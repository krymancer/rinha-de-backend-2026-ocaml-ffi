// ivf.c — IVF (inverted file) k-NN: runtime search + build-time index construction.
// 1:1 port of the validated Zig submission (ivf.zig + refs.zig + index.zig Top5).
// Correctness is paramount: identical int16 vectors, identical distances, identical
// fraud decisions, and a byte-identical on-disk RNHZIVF2 format.

#include "rinha.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ---- small helpers -----------------------------------------------------------

static inline size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

// Read an entire file into an allocator-owned buffer (anonymous RAM, fully
// resident — mirrors os.readFileAlloc which uses the page allocator). The buffer
// is PAGE-ALIGNED (4096): this matches Zig's page-aligned resident pages and,
// crucially, makes `buf + <64-byte-aligned section offset>` 32-byte aligned so
// the dist8 kernel's _mm256_load_si256 (aligned load) on `blocks` is valid.
// Freed with plain free() (posix_memalign pointers are free()-able). NULL on err.
static uint8_t *read_file_alloc(const char *path, size_t *len_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    off_t end = lseek(fd, 0, SEEK_END);
    if (end < 0) { close(fd); return NULL; }
    if (lseek(fd, 0, SEEK_SET) < 0) { close(fd); return NULL; }
    size_t size = (size_t)end;
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, size ? size : 1) != 0) {
        close(fd);
        return NULL;
    }
    size_t off = 0;
    while (off < size) {
        ssize_t rc = read(fd, buf + off, size - off);
        if (rc < 0) { free(buf); close(fd); return NULL; }
        if (rc == 0) break;
        off += (size_t)rc;
    }
    close(fd);
    if (off != size) { free(buf); return NULL; }
    *len_out = size;
    return buf;
}

// ============================================================================
//  RUNTIME (E-critical)
// ============================================================================

// ---- index load --------------------------------------------------------------

int ivf_map(const char *path, ivf_index_t *out, bool pin) {
    size_t len = 0;
    uint8_t *buf = read_file_alloc(path, &len);
    if (!buf) return -1;
    if (len < sizeof(ivf_header_t)) { free(buf); return -2; }
    const ivf_header_t *h = (const ivf_header_t *)buf;
    if (memcmp(h->magic, IVF_MAGIC, 8) != 0) { free(buf); return -3; }
    if (pin) mlockall(MCL_CURRENT | MCL_FUTURE);

    uint64_t nc = h->n_clusters;
    out->n = h->n;
    out->n_clusters = nc;
    out->n_blocks = h->n_blocks;
    out->blocks   = (int16_t  *)(buf + h->blocks_off);    // 32-byte aligned by layout
    out->meta     = (uint32_t *)(buf + h->meta_off);
    out->cl_blk   = (uint32_t *)(buf + h->cl_blk_off);
    out->cl_cnt   = (uint32_t *)(buf + h->cl_cnt_off);
    out->bbox_min = (int16_t  *)(buf + h->bbox_min_off);
    out->bbox_max = (int16_t  *)(buf + h->bbox_max_off);
    out->owned     = buf;
    out->owned_len = len;
    return 0;
}

void rinha_free_index(ivf_index_t *idx) {
    if (idx && idx->owned) {
        free(idx->owned);
        idx->owned = NULL;
        idx->owned_len = 0;
    }
}

// ---- Top5 (from index.zig) ---------------------------------------------------

typedef struct {
    uint64_t keys[K];   // packed (dist<<DIST_SHIFT)|meta ; init UINT64_MAX
    size_t   worst_i;
    uint64_t worst;
} top5_t;

static inline void top5_init(top5_t *t) {
    for (int j = 0; j < K; j++) t->keys[j] = UINT64_MAX;
    t->worst_i = 0;
    t->worst = UINT64_MAX;
}

static inline void top5_offer(top5_t *t, uint64_t key) {
    if (key < t->worst) {
        t->keys[t->worst_i] = key;
        t->worst = t->keys[0];
        t->worst_i = 0;
        // 5-way worst recompute (j = 1..K-1)
        for (int j = 1; j < K; j++) {
            if (t->keys[j] > t->worst) {
                t->worst = t->keys[j];
                t->worst_i = (size_t)j;
            }
        }
    }
}

static inline int64_t top5_worst_dist(const top5_t *t) {
    return (int64_t)(t->worst >> DIST_SHIFT);
}

// ---- clusterLB ---------------------------------------------------------------
// Branchless 16-wide box lower bound: e = max(mn-q,0)+max(q-mx,0) per dim, sum e^2.
// VPAD == 16 fits one 256-bit i16 register exactly. Pad dims contribute 0.
static inline int64_t cluster_lb(const int16_t *q, const int16_t *mn, const int16_t *mx) {
    __m256i qv  = _mm256_loadu_si256((const __m256i *)q);
    __m256i mnv = _mm256_loadu_si256((const __m256i *)mn);
    __m256i mxv = _mm256_loadu_si256((const __m256i *)mx);
    __m256i zero = _mm256_setzero_si256();
    // max(mn-q,0) + max(q-mx,0)  (i16 lanes; mn<=mx so at most one term positive)
    __m256i e = _mm256_add_epi16(_mm256_max_epi16(_mm256_sub_epi16(mnv, qv), zero),
                                 _mm256_max_epi16(_mm256_sub_epi16(qv, mxv), zero));
    // vpmaddwd(e,e) -> 8 i32 lanes, each = e_2k^2 + e_(2k+1)^2 ; sum to i64.
    __m256i sq = _mm256_madd_epi16(e, e);
    // horizontal sum of the 8 i32 into i64 (values <= ~8e8 total, but widen safely)
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, sq);
    int64_t s = 0;
    for (int i = 0; i < 8; i++) s += (int64_t)tmp[i];
    return s;
}

// ---- packQ + dist8 (pair-SoA vpmaddwd kernel) --------------------------------
// qp[p] = set1_epi32( (uint16_t)q[2p] | ((int32_t)q[2p+1] << 16) ): the (a,b)
// dim-pair splatted across all 8 lanes (16 i16). This matches the Zig packQ which
// fills v[lane*2]=a, v[lane*2+1]=b for every lane.
typedef struct { __m256i p[PAIRS]; } qpairs_t;

static inline void pack_q(const int16_t *q, qpairs_t *qp) {
    for (int p = 0; p < PAIRS; p++) {
        // Mask to unsigned before the shift: q dims can be the -10000 sentinel, and
        // left-shifting a negative int is UB (miscompiles at -O3). The int16 bit
        // patterns are preserved, so the packed pair is interpreted correctly.
        uint32_t lo = (uint16_t)q[2 * p];
        uint32_t hi = (uint16_t)q[2 * p + 1];
        qp->p[p] = _mm256_set1_epi32((int32_t)(lo | (hi << 16)));
    }
}

// Squared distance of 8 vectors (one block) to the query; lane j = vector j.
// acc += vpmaddwd(block_p - qp_p, ...) over the 7 pairs.
static inline __m256i dist8(const int16_t *block, const qpairs_t *qp) {
    __m256i acc = _mm256_setzero_si256();
    for (int p = 0; p < PAIRS; p++) {
        __m256i bp = _mm256_load_si256((const __m256i *)(block + p * 16));
        __m256i diff = _mm256_sub_epi16(bp, qp->p[p]);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
    }
    return acc;
}

// Horizontal min of 8 i32 lanes (scalar, matches @reduce(.Min) bit-for-bit).
static inline int32_t hmin8(__m256i v) {
    int32_t a[8];
    _mm256_storeu_si256((__m256i *)a, v);
    int32_t m = a[0];
    for (int i = 1; i < 8; i++) if (a[i] < m) m = a[i];
    return m;
}

// ---- scanCluster -------------------------------------------------------------
static inline void scan_cluster(const ivf_index_t *ivf, uint32_t c,
                                const qpairs_t *qp, top5_t *top) {
    uint32_t cnt  = ivf->cl_cnt[c];
    uint32_t blk0 = ivf->cl_blk[c];
    uint32_t nfull = cnt / LANES;
    uint32_t rem   = cnt % LANES;
    for (uint32_t b = 0; b < nfull; b++) {
        __m256i dv = dist8(ivf->blocks + (size_t)(blk0 + b) * PBLOCK, qp);
        if ((int64_t)hmin8(dv) >= top5_worst_dist(top)) continue;
        int32_t dists[LANES];
        _mm256_storeu_si256((__m256i *)dists, dv);
        size_t mbase = (size_t)(blk0 + b) * LANES;
        for (int lane = 0; lane < LANES; lane++) {
            top5_offer(top, ((uint64_t)(uint32_t)dists[lane] << DIST_SHIFT)
                                 | ivf->meta[mbase + lane]);
        }
    }
    if (rem != 0) {
        __m256i dv = dist8(ivf->blocks + (size_t)(blk0 + nfull) * PBLOCK, qp);
        int32_t dists[LANES];
        _mm256_storeu_si256((__m256i *)dists, dv);
        size_t mbase = (size_t)(blk0 + nfull) * LANES;
        for (uint32_t lane = 0; lane < rem; lane++) {
            top5_offer(top, ((uint64_t)(uint32_t)dists[lane] << DIST_SHIFT)
                                 | ivf->meta[mbase + lane]);
        }
    }
}

// ---- min-heap over cluster keys ----------------------------------------------
static inline void sift_down(uint64_t *h, size_t n, size_t start) {
    size_t i = start;
    for (;;) {
        size_t l = 2 * i + 1;
        size_t r = 2 * i + 2;
        size_t m = i;
        if (l < n && h[l] < h[m]) m = l;
        if (r < n && h[r] < h[m]) m = r;
        if (m == i) break;
        uint64_t t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

// ---- search ------------------------------------------------------------------
result_t ivf_search(const ivf_index_t *idx, const int16_t q[VPAD],
                    size_t init_probe, size_t max_probe,
                    uint64_t *keys, size_t *probed_out) {
    size_t nc = idx->n_clusters;
    for (size_t c = 0; c < nc; c++) {
        int64_t lb = cluster_lb(q, idx->bbox_min + c * VPAD, idx->bbox_max + c * VPAD);
        keys[c] = ((uint64_t)lb << CL_BITS) | (uint64_t)c;
    }
    // heapify keys[0..nc]
    size_t hn = nc;
    {
        size_t i = nc / 2;
        while (i > 0) {
            i -= 1;
            sift_down(keys, hn, i);
        }
    }

    qpairs_t qp;
    pack_q(q, &qp);

    top5_t top;
    top5_init(&top);

    size_t probed = 0;
    const uint64_t mask = ((uint64_t)1 << CL_BITS) - 1;
    while (probed < max_probe && hn > 0) {
        uint64_t kmin = keys[0];
        int64_t lb = (int64_t)(kmin >> CL_BITS);
        if (lb >= top5_worst_dist(&top)) break; // exact prune
        hn -= 1;                                 // pop min
        keys[0] = keys[hn];
        sift_down(keys, hn, 0);
        scan_cluster(idx, (uint32_t)(kmin & mask), &qp, &top);
        probed += 1;
        if (probed >= init_probe) {
            uint8_t fr = 0;
            for (int j = 0; j < K; j++) fr += (uint8_t)(top.keys[j] & 1);
            if (fr == 0 || fr == K) break; // confident
        }
    }
    if (probed_out) *probed_out = probed;

    uint8_t fraud = 0;
    for (int j = 0; j < K; j++) fraud += (uint8_t)(top.keys[j] & 1);
    result_t r;
    r.fraud_count = fraud;
    r.approved = (fraud < 3);
    return r;
}

// ============================================================================
//  BUILD-TIME (used by the indexer)
// ============================================================================

// ---- references.json loader (port of refs.zig loadJson) ----------------------

typedef struct {
    size_t   n;
    int16_t *vecs;   // n * VPAD, AoS padded; dims 14,15 = 0
    uint8_t *labels; // n, 1=fraud 0=legit
} refs_t;

static inline const int16_t *refs_vec(const refs_t *r, size_t i) {
    return r->vecs + i * VPAD;
}

// qref: round(v*10000) -> i16 (round-half-away-from-zero; C round() matches Zig @round)
static inline int16_t qref(double v) {
    return (int16_t)round(v * 10000.0);
}

// Find substring `needle` in buf[pos..len) ; returns offset or (size_t)-1.
static size_t index_of_pos(const uint8_t *buf, size_t len, size_t pos,
                           const char *needle, size_t nlen) {
    if (nlen == 0 || nlen > len) return (size_t)-1;
    for (size_t i = pos; i + nlen <= len; i++) {
        if (buf[i] == (uint8_t)needle[0] && memcmp(buf + i, needle, nlen) == 0)
            return i;
    }
    return (size_t)-1;
}

// Parse a float from buf[start..end) (non-NUL-terminated slice) via strtod.
// Returns false if the slice does not parse to a full float.
static bool parse_float_slice(const uint8_t *buf, size_t start, size_t end, double *out) {
    char tmp[64];
    size_t n = end - start;
    if (n == 0 || n >= sizeof(tmp)) return false;
    memcpy(tmp, buf + start, n);
    tmp[n] = '\0';
    char *endp = NULL;
    double v = strtod(tmp, &endp);
    if (endp != tmp + n) return false; // must consume the whole slice
    *out = v;
    return true;
}

// Load references.json -> refs_t. Returns 0 on success, negative on error.
static int load_json(const char *path, refs_t *refs) {
    size_t len = 0;
    uint8_t *buf = read_file_alloc(path, &len);
    if (!buf) return -1;

    const char *VKEY = "\"vector\"";
    const size_t VKEY_LEN = 8;
    const char *VARR = "\"vector\":[";
    const size_t VARR_LEN = 10;
    const char *LKEY = "\"label\":\"";
    const size_t LKEY_LEN = 9;

    // Count records (one "vector" per record).
    size_t n = 0;
    {
        size_t p = 0;
        size_t at;
        while ((at = index_of_pos(buf, len, p, VKEY, VKEY_LEN)) != (size_t)-1) {
            n += 1;
            p = at + 8;
        }
    }

    refs->n = n;
    refs->vecs = (int16_t *)malloc(n * VPAD * sizeof(int16_t));
    refs->labels = (uint8_t *)malloc(n ? n : 1);
    if (!refs->vecs || !refs->labels) { free(buf); free(refs->vecs); free(refs->labels); return -2; }
    memset(refs->vecs, 0, n * VPAD * sizeof(int16_t));

    size_t pos = 0;
    size_t idx = 0;
    size_t vs;
    while ((vs = index_of_pos(buf, len, pos, VARR, VARR_LEN)) != (size_t)-1) {
        size_t p = vs + VARR_LEN;
        size_t d = 0;
        for (; d < 14; d++) {
            size_t start = p;
            while (p < len && buf[p] != ',' && buf[p] != ']') p++;
            double v;
            if (!parse_float_slice(buf, start, p, &v)) { free(buf); return -3; }
            refs->vecs[idx * VPAD + d] = qref(v);
            if (p < len && buf[p] == ']') { p += 1; break; }
            p += 1; // skip comma
        }
        if (d != 13) { free(buf); return -4; }
        size_t lp = index_of_pos(buf, len, p, LKEY, LKEY_LEN);
        if (lp == (size_t)-1) { free(buf); return -5; }
        size_t ls = lp + LKEY_LEN;
        refs->labels[idx] = (buf[ls] == 'f') ? 1 : 0;
        idx += 1;
        pos = ls;
    }
    if (idx != n) { free(buf); return -6; }
    free(buf);
    return 0;
}

static void free_refs(refs_t *r) {
    free(r->vecs);
    free(r->labels);
    r->vecs = NULL;
    r->labels = NULL;
}

// ---- k-means (port of ivf.zig kmeans) ----------------------------------------
// SoA centroids cent[d*Kp + c], Kp = n_clusters padded to LANES. Stride-init,
// SoA-LANES assignment, scalar i64-accumulator update, `iters` iterations.
static int kmeans(const refs_t *refs, size_t n_clusters, size_t iters, uint32_t *assign) {
    size_t n = refs->n;
    size_t Kp = (n_clusters + LANES - 1) / LANES * LANES;

    int16_t *cent = (int16_t *)malloc(VPAD * Kp * sizeof(int16_t));
    int64_t *sum  = (int64_t *)malloc(n_clusters * VPAD * sizeof(int64_t));
    uint32_t *cnt = (uint32_t *)malloc(n_clusters * sizeof(uint32_t));
    if (!cent || !sum || !cnt) { free(cent); free(sum); free(cnt); return -1; }
    memset(cent, 0, VPAD * Kp * sizeof(int16_t));

    // init: stride-sampled points
    {
        size_t stride = n / n_clusters;
        for (size_t c = 0; c < n_clusters; c++) {
            const int16_t *v = refs_vec(refs, c * stride);
            for (size_t d = 0; d < VPAD; d++) cent[d * Kp + c] = v[d];
        }
        // pad centroids -> far away so they never win
        for (size_t p = n_clusters; p < Kp; p++)
            for (size_t d = 0; d < VPAD; d++) cent[d * Kp + p] = 30000;
    }

    for (size_t it = 0; it < iters; it++) {
        // assign
        for (size_t i = 0; i < n; i++) {
            const int16_t *v = refs_vec(refs, i);
            int32_t best_d = INT32_MAX;
            uint32_t best_c = 0;
            for (size_t cg = 0; cg < Kp; cg += LANES) {
                // acc over VPAD dims, LANES-wide i32
                int32_t acc[LANES];
                for (int l = 0; l < LANES; l++) acc[l] = 0;
                for (size_t d = 0; d < VPAD; d++) {
                    const int16_t *cc = cent + d * Kp + cg;
                    int32_t qd = (int32_t)v[d];
                    for (int l = 0; l < LANES; l++) {
                        int32_t df = (int32_t)cc[l] - qd;
                        acc[l] += df * df;
                    }
                }
                for (int l = 0; l < LANES; l++) {
                    if (acc[l] < best_d) {
                        best_d = acc[l];
                        best_c = (uint32_t)(cg + (size_t)l);
                    }
                }
            }
            assign[i] = (best_c < n_clusters) ? best_c : 0;
        }
        if (it == iters - 1) break; // last assignment is final; skip update
        // update
        memset(sum, 0, n_clusters * VPAD * sizeof(int64_t));
        memset(cnt, 0, n_clusters * sizeof(uint32_t));
        for (size_t i = 0; i < n; i++) {
            uint32_t c = assign[i];
            const int16_t *v = refs_vec(refs, i);
            for (size_t d = 0; d < VPAD; d++) sum[c * VPAD + d] += v[d];
            cnt[c] += 1;
        }
        for (size_t c = 0; c < n_clusters; c++) {
            if (cnt[c] == 0) continue;
            for (size_t d = 0; d < VPAD; d++) {
                int64_t m = sum[c * VPAD + d] / (int64_t)cnt[c]; // @divTrunc
                cent[d * Kp + c] = (int16_t)m;
            }
        }
    }

    free(cent);
    free(sum);
    free(cnt);
    return 0;
}

// ---- build + save (port of ivf.zig build + save) -----------------------------
// Builds the in-memory IVF, then writes the byte-identical RNHZIVF2 file.
int ivf_build_and_save(const char *refs_path, const char *out_path,
                       size_t n_clusters, size_t iters) {
    refs_t refs = {0};
    if (load_json(refs_path, &refs) != 0) return -1;
    size_t n = refs.n;

    uint32_t *assign = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!assign) { free_refs(&refs); return -2; }
    if (kmeans(&refs, n_clusters, iters, assign) != 0) { free(assign); free_refs(&refs); return -3; }

    // counts + total blocks
    uint32_t *cnt = (uint32_t *)calloc(n_clusters, sizeof(uint32_t));
    if (!cnt) { free(assign); free_refs(&refs); return -4; }
    for (size_t i = 0; i < n; i++) cnt[assign[i]] += 1;

    size_t total_blocks = 0;
    for (size_t c = 0; c < n_clusters; c++) total_blocks += (cnt[c] + LANES - 1) / LANES;

    // allocate IVF arrays
    int16_t  *blocks   = NULL;
    uint32_t *meta     = (uint32_t *)malloc(total_blocks * LANES * sizeof(uint32_t));
    uint32_t *cl_blk   = (uint32_t *)malloc(n_clusters * sizeof(uint32_t));
    uint32_t *cl_cnt   = (uint32_t *)malloc(n_clusters * sizeof(uint32_t));
    int16_t  *bbox_min = (int16_t *)malloc(n_clusters * VPAD * sizeof(int16_t));
    int16_t  *bbox_max = (int16_t *)malloc(n_clusters * VPAD * sizeof(int16_t));
    // blocks aligned to 32 (matches alignedAlloc(i16, .@"32"))
    if (posix_memalign((void **)&blocks, 32, total_blocks * PBLOCK * sizeof(int16_t)) != 0)
        blocks = NULL;
    uint32_t *wpos = (uint32_t *)calloc(n_clusters, sizeof(uint32_t));
    if (!meta || !cl_blk || !cl_cnt || !bbox_min || !bbox_max || !blocks || !wpos) {
        free(blocks); free(meta); free(cl_blk); free(cl_cnt);
        free(bbox_min); free(bbox_max); free(wpos); free(cnt);
        free(assign); free_refs(&refs); return -5;
    }
    memset(blocks, 0, total_blocks * PBLOCK * sizeof(int16_t));
    // Zig `@memset([]u32, 0xFF)` sets each ELEMENT to 0xFF (=255), not each byte.
    // Match it exactly so the file is byte-identical to a Zig-built index. (Pad
    // lanes are never read during search, but byte-identity is the contract.)
    for (size_t i = 0; i < total_blocks * LANES; i++) meta[i] = 0xFF;

    // assign block ranges + init bbox
    {
        size_t blk = 0;
        for (size_t c = 0; c < n_clusters; c++) {
            cl_blk[c] = (uint32_t)blk;
            cl_cnt[c] = cnt[c];
            blk += (cnt[c] + LANES - 1) / LANES;
            for (size_t d = 0; d < VPAD; d++) {
                bbox_min[c * VPAD + d] = INT16_MAX;
                bbox_max[c * VPAD + d] = INT16_MIN;
            }
        }
    }

    // fill blocks + bbox + meta
    for (size_t i = 0; i < n; i++) {
        uint32_t c = assign[i];
        uint32_t j = wpos[c];
        wpos[c] += 1;
        size_t b = cl_blk[c] + j / LANES;
        size_t lane = j % LANES;
        const int16_t *v = refs_vec(&refs, i);
        // bbox over all VPAD dims
        for (size_t d = 0; d < VPAD; d++) {
            if (v[d] < bbox_min[c * VPAD + d]) bbox_min[c * VPAD + d] = v[d];
            if (v[d] > bbox_max[c * VPAD + d]) bbox_max[c * VPAD + d] = v[d];
        }
        // block: pair-SoA over the 14 real dims
        for (size_t p = 0; p < PAIRS; p++) {
            blocks[b * PBLOCK + p * 16 + lane * 2 + 0] = v[2 * p];
            blocks[b * PBLOCK + p * 16 + lane * 2 + 1] = v[2 * p + 1];
        }
        meta[b * LANES + lane] = ((uint32_t)i << 1) | refs.labels[i];
    }

    free(wpos);
    free(cnt);
    free(assign);

    // ---- serialize (exact ivf.zig save() offsets/alignment) ------------------
    size_t nc = n_clusters;
    size_t blocks_len = total_blocks * PBLOCK; // i16 elements
    size_t meta_len = total_blocks * LANES;    // u32 elements

    size_t off = align_up(sizeof(ivf_header_t), 64);
    size_t cl_blk_off = off;
    off = align_up(off + nc * 4, 64);
    size_t cl_cnt_off = off;
    off = align_up(off + nc * 4, 64);
    size_t bbox_min_off = off;
    off = align_up(off + nc * VPAD * 2, 64);
    size_t bbox_max_off = off;
    off = align_up(off + nc * VPAD * 2, 64);
    size_t blocks_off = off;
    off = align_up(off + blocks_len * 2, 64);
    size_t meta_off = off;
    size_t total = off + meta_len * 4;

    ivf_header_t h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, IVF_MAGIC, 8);
    h.n = n;
    h.n_clusters = nc;
    h.n_blocks = total_blocks;
    h.cl_blk_off = cl_blk_off;
    h.cl_cnt_off = cl_cnt_off;
    h.bbox_min_off = bbox_min_off;
    h.bbox_max_off = bbox_max_off;
    h.blocks_off = blocks_off;
    h.meta_off = meta_off;
    h.total = total;

    int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        free(blocks); free(meta); free(cl_blk); free(cl_cnt);
        free(bbox_min); free(bbox_max); free_refs(&refs);
        return -6;
    }

    // write-all helper
    int rc = 0;
    static const uint8_t zeros[64] = {0};

    #define WRITE_ALL(ptr, nbytes) do {                                  \
        size_t _w = 0; const uint8_t *_p = (const uint8_t *)(ptr);       \
        size_t _n = (nbytes);                                            \
        while (_w < _n) {                                                \
            ssize_t _r = write(fd, _p + _w, _n - _w);                    \
            if (_r <= 0) { rc = -7; goto done; }                        \
            _w += (size_t)_r;                                            \
        }                                                                \
    } while (0)

    // pad with zeros until `cur` reaches `target`
    #define PAD_TO(curp, target) do {                                    \
        while (*(curp) < (target)) {                                     \
            size_t _m = (64 < (target) - *(curp)) ? 64 : (target) - *(curp); \
            WRITE_ALL(zeros, _m);                                        \
            *(curp) += _m;                                               \
        }                                                                \
    } while (0)

    {
        size_t cur = 0;
        WRITE_ALL(&h, sizeof(ivf_header_t));
        cur = sizeof(ivf_header_t);
        PAD_TO(&cur, cl_blk_off);
        WRITE_ALL(cl_blk, nc * 4);
        cur += nc * 4;
        PAD_TO(&cur, cl_cnt_off);
        WRITE_ALL(cl_cnt, nc * 4);
        cur += nc * 4;
        PAD_TO(&cur, bbox_min_off);
        WRITE_ALL(bbox_min, nc * VPAD * 2);
        cur += nc * VPAD * 2;
        PAD_TO(&cur, bbox_max_off);
        WRITE_ALL(bbox_max, nc * VPAD * 2);
        cur += nc * VPAD * 2;
        PAD_TO(&cur, blocks_off);
        WRITE_ALL(blocks, blocks_len * 2);
        cur += blocks_len * 2;
        PAD_TO(&cur, meta_off);
        WRITE_ALL(meta, meta_len * 4);
    }

done:
    #undef WRITE_ALL
    #undef PAD_TO
    close(fd);
    free(blocks); free(meta); free(cl_blk); free(cl_cnt);
    free(bbox_min); free(bbox_max); free_refs(&refs);
    return rc;
}
