// Fraud-score API worker. Receives already-accepted client sockets from the load
// balancer over a Unix SOCK_SEQPACKET channel (SCM_RIGHTS), then serves HTTP/1.1
// keep-alive directly on those sockets via a single-threaded epoll loop.
//   server <unix_ctrl_path> <index.bin> [spin_us] [idle_us]
// 1:1 port of api.zig (the validated Zig submission, score 5966, E=0).

#define _GNU_SOURCE
#include "rinha.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>

// ---- connection table --------------------------------------------------------

typedef enum { K_UNUSED = 0, K_LISTEN, K_CTRL, K_CLIENT } kind_t;
typedef struct {
    kind_t   kind;
    uint32_t len;
    uint8_t  buf[CONN_BUF];
} conn_t;

static conn_t       conns[MAX_FDS];
static ivf_index_t  gidx;
static uint64_t     gkeys[MAX_CLUSTERS];
static int          epfd;

// ---- epoll helpers -----------------------------------------------------------

// Adaptive wait: poll, then busy-spin (PAUSE) for spin_ns catching back-to-back
// requests with no sleep, then block with a short finite timeout (idle_us) instead
// of forever — the periodic wakeup keeps the core warm (no deep C-state / freq
// drop), cutting the cold-wakeup latency that dominated the tail. (bmtec's loop.)
static inline int wait_events(struct epoll_event *events, int maxevents,
                              uint64_t spin_ns, long idle_us) {
    if (spin_ns == 0) return epoll_wait_us(epfd, events, maxevents, idle_us);
    int rc = epoll_wait_us(epfd, events, maxevents, 0);
    if (rc > 0) return rc;
    long start = now_ns();
    while ((uint64_t)(now_ns() - start) < spin_ns) {
        rc = epoll_wait_us(epfd, events, maxevents, 0);
        if (rc > 0) return rc;
        __builtin_ia32_pause();
    }
    return epoll_wait_us(epfd, events, maxevents, idle_us);
}

static inline void epoll_add(int fd, uint32_t events) {
    struct epoll_event ev;
    ev.events = events;
    ev.data.fd = fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);
}
static inline void epoll_del(int fd) {
    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
}

static void close_client(int fd) {
    epoll_del(fd);
    close(fd);
    if (fd >= 0 && fd < MAX_FDS) conns[fd].kind = K_UNUSED;
}

// ---- HTTP handling -----------------------------------------------------------

static bool write_all_sock(int fd, const char *bytes, size_t total) {
    size_t off = 0;
    uint32_t spins = 0;
    while (off < total) {
        ssize_t rc = write(fd, bytes + off, total - off);
        if (rc > 0) {
            off += (size_t)rc;
        } else if (rc == 0) {
            return false;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                spins++;
                if (spins > 100000) return false;
            } else if (errno == EINTR) {
                // retry
            } else {
                return false;
            }
        }
    }
    return true;
}

// Try to serve as many complete requests as are buffered in conns[fd].
// Returns false if the connection should be closed.
static bool process(int fd) {
    conn_t *c = &conns[fd];
    size_t consumed = 0;
    while (1) {
        const uint8_t *data = c->buf + consumed;
        size_t data_len = c->len - consumed;
        long hend_rel = http_header_end(data, data_len);
        if (hend_rel < 0) break;                 // need more bytes
        const uint8_t *headers = data;           // header block: data[0..hend_rel]
        size_t total = (size_t)hend_rel;
        const char *resp;
        size_t resp_len;
        resp = http_ready(&resp_len);
        if (data[0] == 'P') {                    // POST /fraud-score
            long cl = http_content_length(headers, (size_t)hend_rel);
            size_t body_len = cl < 0 ? 0 : (size_t)cl;
            total = (size_t)hend_rel + body_len;
            if (data_len < total) break;         // body incomplete
            const uint8_t *body = data + hend_rel;
            int16_t q[VPAD];
            if (vectorize(body, body_len, q)) {
                result_t r = ivf_search(&gidx, q, INIT_PROBE, MAX_PROBE,
                                        gkeys, NULL);
                resp = http_resp(r.fraud_count, &resp_len);
            } else {
                resp = http_resp(0, &resp_len);  // SAFE = RESP[0]
            }
        }
        if (!write_all_sock(fd, resp, resp_len)) return false;
        consumed += total;
        if (consumed >= c->len) break;
    }
    if (consumed > 0) {
        size_t remaining = c->len - consumed;
        if (remaining > 0) memmove(c->buf, c->buf + consumed, remaining);
        c->len = (uint32_t)remaining;
    }
    return true;
}

static void on_client_readable(int fd) {
    conn_t *c = &conns[fd];
    while (1) {
        if (c->len >= CONN_BUF) {
            // request too large for our buffer; drop it safely
            c->len = 0;
        }
        ssize_t rc = read(fd, c->buf + c->len, CONN_BUF - c->len);
        if (rc > 0) {
            c->len += (uint32_t)rc;
            if (!process(fd)) {
                close_client(fd);
                return;
            }
            // keep draining; level-triggered will refire but draining cuts syscalls
        } else if (rc == 0) {
            close_client(fd);
            return;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return;
            if (errno == EINTR) continue;
            close_client(fd);
            return;
        }
    }
}

static void on_ctrl_readable(int ctrl) {
    while (1) {
        int cfd = recv_fd(ctrl);
        if (cfd == -1) return;   // EAGAIN
        if (cfd == -2) return;   // closed/error
        // cfd is a real fd
        if (cfd < 0 || cfd >= MAX_FDS) {
            close(cfd);
            continue;
        }
        conns[cfd].kind = K_CLIENT;
        conns[cfd].len = 0;
        set_busy_poll(cfd, 64);  // SO_BUSY_POLL microseconds (best-effort)
        epoll_add(cfd, EPOLLIN | EPOLLRDHUP);
    }
}

// ---- NAPI busy-poll on the epoll instance ------------------------------------

// Match the Zig submission's extern struct exactly (u32,u16,u8,u8 = 8 bytes).
// glibc's <sys/epoll.h> defines struct epoll_params identically, but we use our
// own name and the Zig-hardcoded ioctl number to stay byte-identical: the Zig
// source uses 0x40086502 (= _IOW('e'=0x65, 0x02, 8)), NOT glibc's current
// EPIOCSPARAMS (0x40088A01 = _IOW(0x8A, 0x01, 8)). This ioctl is best-effort
// (failure ignored), so reproducing the exact same number reproduces the exact
// same kernel outcome the validated run had.
struct rinha_epoll_params {
    uint32_t busy_poll_usecs;
    uint16_t busy_poll_budget;
    uint8_t  prefer_busy_poll;
    uint8_t  pad;
};
#define RINHA_EPIOCSPARAMS 0x40086502u

int main(int argc, char **argv) {
    if (argc < 3) return 1;
    const char *ctrl_path  = argv[1];
    const char *index_path = argv[2];
    // Optional epoll tuning: <spin_us> <idle_us>. Defaults (0, -1) = block forever.
    // The submission passes "30 80" = 30us spin then 80us finite-timeout block.
    uint64_t spin_us = (argc > 3) ? strtoull(argv[3], NULL, 10) : 0;
    long     idle_us = (argc > 4) ? strtol(argv[4], NULL, 10) : -1;
    uint64_t spin_ns = spin_us * 1000;

    if (ivf_map(index_path, &gidx, true) < 0) return 1;

    // Unix SEQPACKET listener for the LB to connect and pass client fds.
    int lfd = uds_listener(ctrl_path, 16);
    if (lfd < 0) return 1;

    epfd = epoll_create1(0);
    if (epfd < 0) return 1;

    // NAPI busy-poll on the epoll instance (kernel >= 6.9): epoll_wait polls the
    // device for up to busy_poll_usecs before sleeping, slashing wakeup latency
    // under CPU contention. Best-effort — ignored on older kernels.
    struct rinha_epoll_params ep_params = {
        .busy_poll_usecs = 64,
        .busy_poll_budget = 8,
        .prefer_busy_poll = 1,
        .pad = 0,
    };
    ioctl(epfd, RINHA_EPIOCSPARAMS, &ep_params);

    if (lfd < MAX_FDS) {
        conns[lfd].kind = K_LISTEN;
        conns[lfd].len = 0;
    }
    epoll_add(lfd, EPOLLIN);

    struct epoll_event events[256];
    while (1) {
        int n = wait_events(events, 256, spin_ns, idle_us);
        if (n < 0) {
            // EINTR or other error -> re-poll/spin
            continue;
        }
        if (n == 0) continue; // finite idle timeout fired with no events
        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;
            uint32_t ev = events[i].events;
            if (fd < 0 || fd >= MAX_FDS) continue;
            switch (conns[fd].kind) {
                case K_LISTEN: {
                    while (1) {
                        int ctrl = accept4(fd, NULL, NULL,
                                           SOCK_NONBLOCK | SOCK_CLOEXEC);
                        if (ctrl < 0) break;
                        if (ctrl >= MAX_FDS) {
                            close(ctrl);
                            continue;
                        }
                        conns[ctrl].kind = K_CTRL;
                        conns[ctrl].len = 0;
                        epoll_add(ctrl, EPOLLIN | EPOLLRDHUP);
                    }
                    break;
                }
                case K_CTRL: {
                    if (ev & (EPOLLHUP | EPOLLERR)) {
                        epoll_del(fd);
                        close(fd);
                        conns[fd].kind = K_UNUSED;
                        continue;
                    }
                    on_ctrl_readable(fd);
                    break;
                }
                case K_CLIENT: {
                    if (ev & (EPOLLHUP | EPOLLERR)) {
                        close_client(fd);
                        continue;
                    }
                    on_client_readable(fd);
                    break;
                }
                case K_UNUSED:
                default:
                    break;
            }
        }
    }
    return 0;
}
