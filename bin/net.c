// net.c — sockets, SCM_RIGHTS fd-passing, epoll, timing.
// Ported 1:1 from lb.zig / api.zig / os.zig. The cmsg layout matches the Zig
// submission (one fd over SOL_SOCKET / SCM_RIGHTS), expressed via the standard
// CMSG_* macros which produce the identical wire layout.
#define _GNU_SOURCE
#include "rinha.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/epoll.h>
#include <sys/syscall.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <stddef.h>

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif
#ifndef TCP_QUICKACK
#define TCP_QUICKACK 12
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0x4000
#endif

// ---- listeners ---------------------------------------------------------------

// TCP listener on the given port: AF_INET SOCK_STREAM, SO_REUSEADDR, bind, listen
// 1024. Blocking (the LB accept loop is blocking). Returns fd, or -1 on error.
int tcp_listener(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, IPPROTO_TCP);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = 0; // INADDR_ANY
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { close(fd); return -1; }
    if (listen(fd, 1024) != 0) { close(fd); return -1; }
    return fd;
}

// AF_UNIX SOCK_SEQPACKET listener (nonblocking, cloexec). Unlinks the stale path
// first, then bind + listen. Returns fd, or -1 on error.
int uds_listener(const char *path, int backlog) {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct sockaddr_un un;
    memset(&un, 0, sizeof(un));
    un.sun_family = AF_UNIX;
    size_t plen = strlen(path);
    if (plen >= sizeof(un.sun_path)) { close(fd); return -1; }
    memcpy(un.sun_path, path, plen);
    // Match Zig's un_len = offsetof(path) + plen + 1 (include trailing NUL).
    socklen_t un_len = (socklen_t)(offsetof(struct sockaddr_un, sun_path) + plen + 1);
    unlink(path);
    if (bind(fd, (struct sockaddr *)&un, un_len) != 0) { close(fd); return -1; }
    if (listen(fd, backlog) != 0) { close(fd); return -1; }
    return fd;
}

// AF_UNIX SOCK_SEQPACKET connect with ~600 attempts x 100ms retries.
// Returns fd, or -1 if all attempts fail.
int uds_connect(const char *path) {
    size_t plen = strlen(path);
    if (plen >= sizeof(((struct sockaddr_un *)0)->sun_path)) return -1;
    for (int attempt = 0; attempt < 600; attempt++) {
        int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (fd < 0) {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000000L };
            nanosleep(&ts, &ts);
            continue;
        }
        struct sockaddr_un un;
        memset(&un, 0, sizeof(un));
        un.sun_family = AF_UNIX;
        memcpy(un.sun_path, path, plen);
        socklen_t un_len = (socklen_t)(offsetof(struct sockaddr_un, sun_path) + plen + 1);
        if (connect(fd, (struct sockaddr *)&un, un_len) == 0) return fd;
        close(fd);
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000000L };
        nanosleep(&ts, &ts);
    }
    return -1;
}

// ---- SCM_RIGHTS fd passing ---------------------------------------------------

// Send one fd over a SEQPACKET socket via SCM_RIGHTS. 1-byte iov payload (0x46),
// MSG_NOSIGNAL. Spin on EAGAIN/EINTR (cap 1,000,000 spins). Returns true on send.
bool send_fd(int sock, int fd) {
    unsigned char byte = 0x46;
    struct iovec iov = { .iov_base = &byte, .iov_len = 1 };
    union {
        struct cmsghdr align;
        char buf[CMSG_SPACE(sizeof(int))];
    } ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl.buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int));
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
    // Zig sets controllen = cmsghdr_aligned + sizeof(i32) (== CMSG_LEN). The
    // kernel accepts either CMSG_LEN or CMSG_SPACE; use CMSG_LEN to match Zig.
    msg.msg_controllen = cmsg->cmsg_len;

    unsigned spins = 0;
    for (;;) {
        ssize_t rc = sendmsg(sock, &msg, MSG_NOSIGNAL);
        if (rc >= 0) return true;
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            if (++spins > 1000000u) return false;
            continue;
        }
        return false;
    }
}

// Receive one fd over SCM_RIGHTS. 1-byte iov. Returns fd>=0, -1 (EAGAIN), or
// -2 (closed/error). Mirrors recvOneFd's union(again/closed/fd).
int recv_fd(int ctrl_sock) {
    unsigned char byte;
    struct iovec iov = { .iov_base = &byte, .iov_len = 1 };
    union {
        struct cmsghdr align;
        char buf[CMSG_SPACE(sizeof(int))];
    } ctrl;
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl.buf;
    msg.msg_controllen = sizeof(ctrl.buf);

    ssize_t rc = recvmsg(ctrl_sock, &msg, 0);
    if (rc < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return -1;
        return -2;
    }
    if (rc == 0) return -2; // closed
    if (msg.msg_controllen < sizeof(struct cmsghdr)) return -1; // -> .again
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL || cmsg->cmsg_level != SOL_SOCKET ||
        cmsg->cmsg_type != SCM_RIGHTS) return -1; // -> .again
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

// ---- socket options ----------------------------------------------------------

void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) flags = 0;
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

void set_quickack(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
}

void set_busy_poll(int fd, int usecs) {
    setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &usecs, sizeof(usecs));
}

// ---- epoll -------------------------------------------------------------------

// epoll_wait with microsecond-precision timeout. timeout_us < 0 blocks forever;
// == 0 returns immediately; otherwise epoll_pwait2 (kernel >= 5.11) falling back
// to millisecond epoll_wait (ceil) on ENOSYS. Returns the epoll_wait return code
// (>=0 number of events, or -1 with errno set). Mirrors api.zig epollWaitUs.
int epoll_wait_us(int epfd, void *events, int maxevents, long timeout_us) {
    struct epoll_event *ev = (struct epoll_event *)events;
    if (timeout_us < 0) return epoll_wait(epfd, ev, maxevents, -1);
    if (timeout_us == 0) return epoll_wait(epfd, ev, maxevents, 0);
    struct timespec ts = {
        .tv_sec  = timeout_us / 1000000L,
        .tv_nsec = (timeout_us % 1000000L) * 1000L,
    };
    long rc = syscall(SYS_epoll_pwait2, (long)epfd, (long)ev,
                      (long)maxevents, (long)&ts, (long)NULL, (long)8);
    if (rc < 0 && errno == ENOSYS) {
        int ms = (int)((timeout_us + 999) / 1000);
        return epoll_wait(epfd, ev, maxevents, ms);
    }
    return (int)rc;
}

// ---- timing ------------------------------------------------------------------

long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + (long)ts.tv_nsec;
}
