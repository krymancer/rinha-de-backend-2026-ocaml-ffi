// http.c — minimal HTTP/1.1 for the fraud-score endpoint, ported 1:1 from http.zig.
// The only possible response bodies are the 6 fraud-count outcomes; we precompute
// each full response (headers + body) once into static buffers. The hot path is a
// single indexed lookup + one write().
#include "rinha.h"
#include <string.h>
#include <strings.h> // strncasecmp
#include <stdio.h>

// Bodies indexed by fraud_count (0..5). approved=true for 0..2, false for 3..5.
static const char *const BODIES[6] = {
    "{\"approved\":true,\"fraud_score\":0.0}",
    "{\"approved\":true,\"fraud_score\":0.2}",
    "{\"approved\":true,\"fraud_score\":0.4}",
    "{\"approved\":false,\"fraud_score\":0.6}",
    "{\"approved\":false,\"fraud_score\":0.8}",
    "{\"approved\":false,\"fraud_score\":1.0}",
};

// Precomputed full responses. Bodies are <=36 bytes, headers ~ 90 bytes; 160 is
// ample. Lengths are recorded so the hot path returns an exact slice.
static char  resp_buf[6][160];
static size_t resp_len[6];

static const char READY[] =
    "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
static const size_t READY_LEN = sizeof(READY) - 1;

static int g_initialized = 0;

static void http_init(void) {
    for (int i = 0; i < 6; i++) {
        const char *body = BODIES[i];
        size_t body_len = strlen(body);
        int n = snprintf(resp_buf[i], sizeof(resp_buf[i]),
                         "HTTP/1.1 200 OK\r\n"
                         "Content-Type: application/json\r\n"
                         "Content-Length: %zu\r\n"
                         "Connection: keep-alive\r\n"
                         "\r\n"
                         "%s",
                         body_len, body);
        resp_len[i] = (size_t)n;
    }
    g_initialized = 1;
}

// Use a constructor so the buffers are ready before any request is served; the
// lazy guard remains as a belt-and-suspenders fallback.
__attribute__((constructor)) static void http_ctor(void) { http_init(); }

const char *http_resp(int fraud_count, size_t *len_out) {
    if (!g_initialized) http_init();
    // fraud_count is always 0..5 from ivf_search; clamp defensively.
    if (fraud_count < 0) fraud_count = 0;
    if (fraud_count > 5) fraud_count = 5;
    if (len_out) *len_out = resp_len[fraud_count];
    return resp_buf[fraud_count];
}

const char *http_ready(size_t *len_out) {
    if (len_out) *len_out = READY_LEN;
    return READY;
}

// Find end of headers ("\r\n\r\n"); returns index just past it, or -1.
long http_header_end(const uint8_t *buf, size_t len) {
    if (len < 4) return -1;
    for (size_t i = 0; i + 4 <= len; i++) {
        if (buf[i] == '\r' && buf[i + 1] == '\n' &&
            buf[i + 2] == '\r' && buf[i + 3] == '\n') {
            return (long)(i + 4);
        }
    }
    return -1;
}

// Parse Content-Length (case-insensitive) within the header block; or -1.
long http_content_length(const uint8_t *headers, size_t len) {
    static const char NEEDLE[] = "content-length:";
    const size_t NLEN = sizeof(NEEDLE) - 1; // 15
    if (len < NLEN) return -1;
    for (size_t i = 0; i + NLEN <= len; i++) {
        if (strncasecmp((const char *)(headers + i), NEEDLE, NLEN) == 0) {
            size_t j = i + NLEN;
            while (j < len && (headers[j] == ' ' || headers[j] == '\t')) j++;
            // Parse digits. Zig's parseInt over the digit run; empty -> null.
            if (j >= len || headers[j] < '0' || headers[j] > '9') return -1;
            size_t val = 0;
            while (j < len && headers[j] >= '0' && headers[j] <= '9') {
                val = val * 10 + (size_t)(headers[j] - '0');
                j++;
            }
            return (long)val;
        }
    }
    return -1;
}
