// Payload (JSON) -> 14-dimension int16 query vector.
// 1:1 port of vectorize.zig (validated Zig submission, E=0). Every coordinate the
// data-generator emits is round4'd, so multiplying by 10000 and rounding yields an
// EXACT int16 in [-10000, 10000]. Dims 14,15 are always 0.

#include "rinha.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---- normalization.json constants (fixed for the edition) -------------------
#define MAX_AMOUNT          10000.0
#define MAX_INSTALLMENTS    12.0
#define AMOUNT_VS_AVG_RATIO 10.0
#define MAX_MINUTES         1440.0
#define MAX_KM              1000.0
#define MAX_TX_24H          20.0
#define MAX_MERCH_AVG       10000.0

// A non-owning text slice (Zig []const u8).
typedef struct { const char *p; size_t len; } slice_t;

static inline double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

// round(v*10000) clamped to int16 sentinel range. C round() = ties away from
// zero, matching Zig @round. clamp before cast.
static inline int16_t q(double v) {
    double r = round(v * 10000.0);
    if (r < -10000.0) r = -10000.0;
    if (r > 10000.0) r = 10000.0;
    return (int16_t)r;
}

// mcc_risk.json lookup; default 0.5 for unknown MCCs. `mcc` is the inner string.
static double mccRisk(slice_t mcc) {
    static const struct { const char *code; double risk; } tbl[] = {
        { "5411", 0.15 }, { "5812", 0.30 }, { "5912", 0.20 }, { "5944", 0.45 },
        { "7801", 0.80 }, { "7802", 0.75 }, { "7995", 0.85 }, { "4511", 0.35 },
        { "5311", 0.25 }, { "5999", 0.50 },
    };
    for (size_t i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++) {
        size_t cl = strlen(tbl[i].code);
        if (mcc.len == cl && memcmp(mcc.p, tbl[i].code, cl) == 0) return tbl[i].risk;
    }
    return 0.5;
}

// ----- timestamp helpers (UTC) ----------------------------------------------

typedef struct { int64_t y, mo, d, h, mi, s; } ts_t;

// Strict base-10 integer parse over exactly [p, p+n), matching Zig parseInt:
// optional single leading '+'/'-', then digits only; any other char -> error.
// Returns true on success.
static bool parseIntFixed(const char *p, size_t n, int64_t *out) {
    if (n == 0) return false;
    size_t i = 0;
    bool neg = false;
    if (p[0] == '+' || p[0] == '-') {
        neg = (p[0] == '-');
        i = 1;
        if (i >= n) return false; // sign with no digits
    }
    int64_t acc = 0;
    for (; i < n; i++) {
        char c = p[i];
        if (c < '0' || c > '9') return false;
        acc = acc * 10 + (int64_t)(c - '0');
    }
    *out = neg ? -acc : acc;
    return true;
}

// Parse "YYYY-MM-DDThh:mm:ssZ" by fixed offsets. Returns false on failure.
static bool parseTs(slice_t s, ts_t *out) {
    if (s.len < 19) return false;
    const char *b = s.p;
    return parseIntFixed(b + 0,  4, &out->y)  &&
           parseIntFixed(b + 5,  2, &out->mo) &&
           parseIntFixed(b + 8,  2, &out->d)  &&
           parseIntFixed(b + 11, 2, &out->h)  &&
           parseIntFixed(b + 14, 2, &out->mi) &&
           parseIntFixed(b + 17, 2, &out->s);
}

// Floored division/modulo, matching Zig @divFloor / @mod for i64.
static inline int64_t divFloor(int64_t a, int64_t b) {
    int64_t qd = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) qd--;
    return qd;
}
static inline int64_t modFloor(int64_t a, int64_t b) {
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

// days since 1970-01-01 (Howard Hinnant). Note: era uses @divFloor, the rest
// @divTrunc (plain C / for the involved nonneg operands).
static int64_t daysFromCivil(int64_t y_in, int64_t m, int64_t d) {
    int64_t y = y_in - (m <= 2 ? 1 : 0);
    int64_t era = divFloor((y >= 0 ? y : y - 399), 400);
    int64_t yoe = y - era * 400;                 // [0, 399]
    int64_t mp = modFloor(m + 9, 12);            // [0,11] with Mar=0
    int64_t doy = (153 * mp + 2) / 5 + d - 1;    // [0,365]
    int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0,146096]
    return era * 146097 + doe - 719468;
}

static int64_t epochSecs(ts_t t) {
    return daysFromCivil(t.y, t.mo, t.d) * 86400 + t.h * 3600 + t.mi * 60 + t.s;
}

// day_of_week, Monday=0..Sunday=6 (Sakamoto). Replicates the generator exactly.
static int64_t dayOfWeek(int64_t y_in, int64_t m, int64_t d) {
    static const int64_t t[12] = { 0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4 };
    int64_t y = y_in - (m < 3 ? 1 : 0);
    int64_t dow = modFloor(y + y / 4 - y / 100 + y / 400 + t[m - 1] + d, 7); // 0=Sun
    return modFloor(dow + 6, 7); // 0=Mon
}

// ----- minimal JSON field extraction ----------------------------------------
// Extract the value text for "key": within buf (first occurrence). For
// objects/arrays/strings the returned slice includes the delimiters; for bare
// values it excludes the trailing ',' '}' ']'. Returns false if not found.

static bool field(slice_t buf, const char *key, slice_t *out) {
    // needle = "\"" ++ key ++ "\":"
    char needle[64];
    size_t klen = strlen(key);
    if (klen + 3 >= sizeof(needle)) return false;
    needle[0] = '"';
    memcpy(needle + 1, key, klen);
    needle[1 + klen] = '"';
    needle[2 + klen] = ':';
    size_t nlen = klen + 3;

    // first occurrence of needle in buf (memmem over the slice)
    if (buf.len < nlen) return false;
    const char *hit = NULL;
    for (size_t off = 0; off + nlen <= buf.len; off++) {
        if (memcmp(buf.p + off, needle, nlen) == 0) { hit = buf.p + off; break; }
    }
    if (!hit) return false;

    size_t at = (size_t)(hit - buf.p);
    size_t i = at + nlen;
    // skip optional whitespace
    while (i < buf.len) {
        char w = buf.p[i];
        if (w == ' ' || w == '\t' || w == '\n' || w == '\r') i++;
        else break;
    }
    if (i >= buf.len) return false;
    size_t start = i;
    char c = buf.p[i];
    if (c == '{' || c == '[') {
        char open = c;
        char close = (c == '{') ? '}' : ']';
        int depth = 0;
        bool in_str = false;
        for (; i < buf.len; i++) {
            char ch = buf.p[i];
            if (in_str) {
                if (ch == '\\') i++;
                else if (ch == '"') in_str = false;
            } else if (ch == '"') {
                in_str = true;
            } else if (ch == open) {
                depth++;
            } else if (ch == close) {
                depth--;
                if (depth == 0) { out->p = buf.p + start; out->len = (i + 1) - start; return true; }
            }
        }
        return false;
    } else if (c == '"') {
        i++;
        for (; i < buf.len; i++) {
            if (buf.p[i] == '\\') i++;
            else if (buf.p[i] == '"') { out->p = buf.p + start; out->len = (i + 1) - start; return true; }
        }
        return false;
    } else {
        while (i < buf.len && buf.p[i] != ',' && buf.p[i] != '}' && buf.p[i] != ']') i++;
        out->p = buf.p + start;
        out->len = i - start;
        return true;
    }
}

// Parse the field's value as f64 (Zig parseFloat). Returns false if missing or
// the value text isn't a valid float over its full extent (parseFloat is strict).
static bool numField(slice_t buf, const char *key, double *out) {
    slice_t v;
    if (!field(buf, key, &v)) return false;
    if (v.len == 0) return false;
    // strtod over a NUL-terminated copy; require it to consume the whole slice
    // (Zig parseFloat rejects trailing garbage).
    char tmp[64];
    if (v.len >= sizeof(tmp)) return false;
    memcpy(tmp, v.p, v.len);
    tmp[v.len] = '\0';
    char *end = NULL;
    double r = strtod(tmp, &end);
    if (end != tmp + v.len) return false;
    *out = r;
    return true;
}

// Strip surrounding quotes if present (Zig strInner).
static inline slice_t strInner(slice_t v) {
    if (v.len >= 2 && v.p[0] == '"' && v.p[v.len - 1] == '"') {
        slice_t r = { v.p + 1, v.len - 2 };
        return r;
    }
    return v;
}

// Search for needle slice within haystack slice (Zig indexOf != null).
static bool sliceContains(slice_t hay, const char *needle, size_t nlen) {
    if (hay.len < nlen) return false;
    for (size_t off = 0; off + nlen <= hay.len; off++) {
        if (memcmp(hay.p + off, needle, nlen) == 0) return true;
    }
    return false;
}

// Vectorize a raw POST body into a 16-wide int16 vector (dims 14,15 = 0).
// Returns false only if the payload is unparseable.
bool vectorize(const uint8_t *buf, size_t len, int16_t out[VPAD]) {
    slice_t body = { (const char *)buf, len };
    slice_t tx, cust, merch, term;
    if (!field(body, "transaction", &tx)) return false;
    if (!field(body, "customer", &cust)) return false;
    if (!field(body, "merchant", &merch)) return false;
    if (!field(body, "terminal", &term)) return false;

    double amount, installments;
    if (!numField(tx, "amount", &amount)) return false;
    if (!numField(tx, "installments", &installments)) return false;
    slice_t req_at_v;
    if (!field(tx, "requested_at", &req_at_v)) return false;
    slice_t req_at = strInner(req_at_v);
    ts_t ts;
    if (!parseTs(req_at, &ts)) return false;

    double avg_amount, tx24;
    if (!numField(cust, "avg_amount", &avg_amount)) return false;
    if (!numField(cust, "tx_count_24h", &tx24)) return false;
    slice_t known;
    if (!field(cust, "known_merchants", &known)) { known.p = "[]"; known.len = 2; }

    slice_t merch_id_v, mcc_v;
    if (!field(merch, "id", &merch_id_v)) return false;
    slice_t merch_id = strInner(merch_id_v);
    if (!field(merch, "mcc", &mcc_v)) return false;
    slice_t mcc = strInner(mcc_v);
    double merch_avg;
    if (!numField(merch, "avg_amount", &merch_avg)) return false;

    slice_t v_online, v_present;
    if (!field(term, "is_online", &v_online)) return false;
    bool is_online = (v_online.len > 0 && v_online.p[0] == 't');
    if (!field(term, "card_present", &v_present)) return false;
    bool card_present = (v_present.len > 0 && v_present.p[0] == 't');
    double km_home;
    if (!numField(term, "km_from_home", &km_home)) return false;

    // last_transaction: null or {timestamp, km_from_current}
    bool has_last = false;
    double minutes = -1.0, last_km = -1.0;
    slice_t lt;
    if (field(body, "last_transaction", &lt)) {
        if (lt.len > 0 && lt.p[0] == '{') {
            slice_t lt_ts_v, lk_present;
            (void)lk_present;
            if (!field(lt, "timestamp", &lt_ts_v)) return false;
            slice_t lt_ts = strInner(lt_ts_v);
            double lk;
            if (!numField(lt, "km_from_current", &lk)) return false;
            ts_t tprev;
            if (!parseTs(lt_ts, &tprev)) return false;
            has_last = true;
            minutes = (double)(epochSecs(ts) - epochSecs(tprev)) / 60.0;
            last_km = lk;
        }
    }

    // unknown_merchant: merchant.id NOT present (quoted) in known_merchants.
    // Zig builds "\"{id}\"" into a 40-byte buffer; id longer than 38 -> null.
    char idbuf[40];
    if (merch_id.len + 2 > sizeof(idbuf)) return false;
    idbuf[0] = '"';
    memcpy(idbuf + 1, merch_id.p, merch_id.len);
    idbuf[1 + merch_id.len] = '"';
    size_t qlen = merch_id.len + 2;
    double known_flag = sliceContains(known, idbuf, qlen) ? 1.0 : 0.0;
    double unknown_merchant = 1.0 - known_flag;

    double hour = (double)ts.h;
    double dow = (double)dayOfWeek(ts.y, ts.mo, ts.d);

    for (int i = 0; i < VPAD; i++) out[i] = 0;
    out[0] = q(clamp01(amount / MAX_AMOUNT));
    out[1] = q(clamp01(installments / MAX_INSTALLMENTS));
    out[2] = q(clamp01((amount / avg_amount) / AMOUNT_VS_AVG_RATIO));
    out[3] = q(hour / 23.0);
    out[4] = q(dow / 6.0);
    if (has_last) {
        out[5] = q(clamp01(minutes / MAX_MINUTES));
        out[6] = q(clamp01(last_km / MAX_KM));
    } else {
        out[5] = -10000;
        out[6] = -10000;
    }
    out[7] = q(clamp01(km_home / MAX_KM));
    out[8] = q(clamp01(tx24 / MAX_TX_24H));
    out[9] = is_online ? 10000 : 0;
    out[10] = card_present ? 10000 : 0;
    out[11] = q(unknown_merchant);
    out[12] = q(mccRisk(mcc));
    out[13] = q(clamp01(merch_avg / MAX_MERCH_AVG));
    return true;
}

// ----- self-test -------------------------------------------------------------
#ifdef VEC_TEST
#include <stdio.h>
#include <assert.h>

static void check(const char *name, const char *payload, const int16_t want[VPAD]) {
    int16_t got[VPAD];
    bool ok = vectorize((const uint8_t *)payload, strlen(payload), got);
    if (!ok) { fprintf(stderr, "FAIL %s: vectorize returned false\n", name); abort(); }
    for (int i = 0; i < VPAD; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL %s: out[%d]=%d want %d\n", name, i, got[i], want[i]);
            abort();
        }
    }
    printf("ok %s\n", name);
}

int main(void) {
    const char *legit =
        "{\"id\":\"tx-1329056812\",\"transaction\":{\"amount\":41.12,\"installments\":2,"
        "\"requested_at\":\"2026-03-11T18:45:53Z\"},\"customer\":{\"avg_amount\":82.24,"
        "\"tx_count_24h\":3,\"known_merchants\":[\"MERC-003\",\"MERC-016\"]},"
        "\"merchant\":{\"id\":\"MERC-016\",\"mcc\":\"5411\",\"avg_amount\":60.25},"
        "\"terminal\":{\"is_online\":false,\"card_present\":true,\"km_from_home\":29.23},"
        "\"last_transaction\":null}";
    const int16_t legit_want[VPAD] =
        { 41, 1667, 500, 7826, 3333, -10000, -10000, 292, 1500, 0, 10000, 0, 1500, 60, 0, 0 };

    const char *fraud =
        "{\"id\":\"tx-3330991687\",\"transaction\":{\"amount\":9505.97,\"installments\":10,"
        "\"requested_at\":\"2026-03-14T05:15:12Z\"},\"customer\":{\"avg_amount\":81.28,"
        "\"tx_count_24h\":20,\"known_merchants\":[\"MERC-008\",\"MERC-007\",\"MERC-005\"]},"
        "\"merchant\":{\"id\":\"MERC-068\",\"mcc\":\"7802\",\"avg_amount\":54.86},"
        "\"terminal\":{\"is_online\":false,\"card_present\":true,\"km_from_home\":952.27},"
        "\"last_transaction\":null}";
    const int16_t fraud_want[VPAD] =
        { 9506, 8333, 10000, 2174, 8333, -10000, -10000, 9523, 10000, 0, 10000, 10000, 7500, 55, 0, 0 };

    const char *smoke =
        "{\"id\":\"tx-smoke-001\",\"transaction\":{\"amount\":384.88,\"installments\":3,"
        "\"requested_at\":\"2026-03-11T20:23:35Z\"},\"customer\":{\"avg_amount\":769.76,"
        "\"tx_count_24h\":3,\"known_merchants\":[\"MERC-009\",\"MERC-001\",\"MERC-001\"]},"
        "\"merchant\":{\"id\":\"MERC-001\",\"mcc\":\"5912\",\"avg_amount\":298.95},"
        "\"terminal\":{\"is_online\":false,\"card_present\":true,\"km_from_home\":13.7090520965},"
        "\"last_transaction\":{\"timestamp\":\"2026-03-11T14:58:35Z\","
        "\"km_from_current\":18.8626479774}}";
    const int16_t smoke_want[VPAD] =
        { 385, 2500, 500, 8696, 3333, 2257, 189, 137, 1500, 0, 10000, 0, 2000, 299, 0, 0 };

    check("legit", legit, legit_want);
    check("fraud", fraud, fraud_want);
    check("smoke", smoke, smoke_want);

    // day_of_week spot checks (2026-03-11 Wed->2, 2026-03-14 Sat->5)
    assert(dayOfWeek(2026, 3, 11) == 2);
    assert(dayOfWeek(2026, 3, 14) == 5);
    printf("ok day_of_week\n");
    printf("all vec tests passed\n");
    return 0;
}
#endif
