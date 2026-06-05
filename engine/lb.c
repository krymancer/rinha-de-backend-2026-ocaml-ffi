// Load balancer: accept TCP connections on <port> and hand each accepted client
// socket to an API worker (round-robin) over a Unix SOCK_SEQPACKET channel using
// SCM_RIGHTS. It never reads the HTTP payload -- pure fd forwarding.
//   lb <port> <api1.sock> <api2.sock> [...]
// Ported 1:1 from lb.zig (final: simple single-fd handoff, no prefetch).

#include "rinha.h"

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#define MAX_APIS 8

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: lb <port> <api1.sock> <api2.sock> [...]\n");
        return 1;
    }

    uint16_t port = (uint16_t)strtoul(argv[1], NULL, 10);

    int api_fds[MAX_APIS];
    size_t napis = 0;
    for (int i = 2; i < argc && napis < MAX_APIS; i++) {
        int fd = uds_connect(argv[i]);
        if (fd < 0) {
            fprintf(stderr, "lb: connect to %s failed\n", argv[i]);
            return 1;
        }
        api_fds[napis++] = fd;
    }
    if (napis == 0) {
        fprintf(stderr, "lb: no apis\n");
        return 1;
    }

    int lfd = tcp_listener(port);
    if (lfd < 0) {
        fprintf(stderr, "lb: tcp_listener failed\n");
        return 1;
    }

    size_t rr = 0;
    for (;;) {
        int client = accept4(lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (client < 0) {
            // INTR, AGAIN, or any other error: just retry (matches lb.zig).
            continue;
        }
        set_nodelay(client);
        send_fd(api_fds[rr], client);
        close(client);
        rr += 1;
        if (rr >= napis) rr = 0;
    }
}
