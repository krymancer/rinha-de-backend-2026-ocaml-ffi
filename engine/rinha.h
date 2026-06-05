// Rinha de Backend 2026 — C implementation. Shared contract.
// Ported 1:1 from the validated Zig submission (krymancer-zig, score 5966, E=0,
// p99 1.080ms): IVF int16x10000 exact search with a pair-SoA vpmaddwd distance
// kernel, SCM_RIGHTS fd-passing LB, warm-core epoll API workers.
#ifndef RINHA_H
#define RINHA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// ---- dimensions / layout -----------------------------------------------------
#define VDIM 14          // real dimensions
#define VPAD 16          // padded to 16 (two trailing zero dims; AVX2)
#define K 5              // nearest neighbours
#define LANES 8          // vectors per SoA block
#define PAIRS 7          // VDIM/2 dim-pairs for the vpmaddwd kernel
#define PBLOCK (PAIRS * 16) // 112 int16 per block (8 vectors x 14 dims, pair-interleaved)

#define DIST_SHIFT 23    // packed key = (dist<<23)|(orig_idx<<1)|label
#define CL_BITS 13       // packed cluster key = (lb<<13)|cluster_id  (n_clusters<=8192)

// search probe budget (matches the Zig submission exactly)
#define INIT_PROBE 24
#define MAX_PROBE 96
#define MAX_CLUSTERS 8192

// server limits
#define MAX_FDS 8192
#define CONN_BUF 4096

#define IVF_MAGIC "RNHZIVF2"   // pair-SoA block format

// ---- types -------------------------------------------------------------------
typedef struct { uint8_t fraud_count; bool approved; } result_t;

// On-disk header (must byte-match the Zig `Header` extern struct: 8-byte magic
// then 10 little-endian u64 fields, naturally packed = 88 bytes).
typedef struct {
    char     magic[8];
    uint64_t n;
    uint64_t n_clusters;
    uint64_t n_blocks;
    uint64_t cl_blk_off;
    uint64_t cl_cnt_off;
    uint64_t bbox_min_off;
    uint64_t bbox_max_off;
    uint64_t blocks_off;
    uint64_t meta_off;
    uint64_t total;
} ivf_header_t;

typedef struct {
    uint64_t  n;
    uint64_t  n_clusters;
    uint64_t  n_blocks;
    int16_t  *blocks;    // n_blocks * PBLOCK, pair-SoA, 32-byte aligned
    uint32_t *meta;      // n_blocks * LANES : (orig_idx<<1)|label ; pad lanes = 0xFFFFFFFF
    uint32_t *cl_blk;    // n_clusters : first block index of each cluster
    uint32_t *cl_cnt;    // n_clusters : real vector count of each cluster
    int16_t  *bbox_min;  // n_clusters * VPAD
    int16_t  *bbox_max;  // n_clusters * VPAD
    void     *owned;     // backing buffer (anonymous RAM); free with rinha_free_index
    size_t    owned_len;
} ivf_index_t;

// ---- vec.c : payload JSON -> 14-dim int16 query (dims 14,15 = 0) --------------
// Returns false only if the payload is unparseable (caller emits a safe 200).
bool vectorize(const uint8_t *buf, size_t len, int16_t out[VPAD]);

// ---- ivf.c : index load + search ---------------------------------------------
// Load index.bin into anonymous RAM (read, not file-mmap). pin=true -> mlockall.
// Returns 0 on success, negative on error.
int  ivf_map(const char *path, ivf_index_t *out, bool pin);
void rinha_free_index(ivf_index_t *idx);

// Adaptive IVF search. `keys` is caller scratch of >= n_clusters uint64_t.
// `probed_out` (nullable) receives the number of cells probed.
result_t ivf_search(const ivf_index_t *idx, const int16_t q[VPAD],
                    size_t init_probe, size_t max_probe,
                    uint64_t *keys, size_t *probed_out);

// ---- indexer (build-time) : in ivf.c, used by indexer main --------------------
// Loads references.json, runs k-means, writes the pair-SoA index.bin.
int  ivf_build_and_save(const char *refs_path, const char *out_path,
                        size_t n_clusters, size_t iters);

// ---- http.c : precomputed responses ------------------------------------------
// Full HTTP/1.1 response (headers+body) for a given fraud_count (0..5).
const char *http_resp(int fraud_count, size_t *len_out);
const char *http_ready(size_t *len_out);
// Header parsing on a raw request buffer.
long http_header_end(const uint8_t *buf, size_t len);     // index past "\r\n\r\n", or -1
long http_content_length(const uint8_t *headers, size_t len); // or -1

// ---- net.c : sockets, fd-passing, epoll --------------------------------------
int  tcp_listener(uint16_t port);                 // bound+listening, blocking, NODELAY-ready
int  uds_listener(const char *path, int backlog); // AF_UNIX SEQPACKET, nonblocking
int  uds_connect(const char *path);               // AF_UNIX SEQPACKET connect (retries)
bool send_fd(int sock, int fd);                   // SCM_RIGHTS sendmsg (1 byte payload)
// recv one fd over SCM_RIGHTS: returns fd>=0, or -1 (EAGAIN), or -2 (closed/error)
int  recv_fd(int ctrl);
void set_nonblocking(int fd);
void set_nodelay(int fd);
void set_quickack(int fd);
void set_busy_poll(int fd, int usecs);
// epoll_wait with microsecond timeout (epoll_pwait2, ms fallback). <0 blocks.
int  epoll_wait_us(int epfd, void *events, int maxevents, long timeout_us);

// ---- os.c : misc -------------------------------------------------------------
long now_ns(void);

#endif // RINHA_H
