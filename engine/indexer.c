// Build-time: references.json -> index.bin (IVF k-means index).
//   indexer <references.json> <out.index.bin> [n_clusters] [iters]
// Ported 1:1 from indexer.zig. ivf_build_and_save does the loading + k-means +
// save internally; here we just parse args and print timings.

#include "rinha.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: indexer <references.json> <out.bin> [n_clusters] [iters]\n");
        return 1;
    }
    const char *refs_path = argv[1];
    const char *out_path = argv[2];
    size_t n_clusters = 2048;
    size_t iters = 12;
    if (argc > 3) {
        char *end = NULL;
        unsigned long v = strtoul(argv[3], &end, 10);
        n_clusters = (end != argv[3]) ? (size_t)v : 2048; // parse error -> 2048
    }
    if (argc > 4) {
        char *end = NULL;
        unsigned long v = strtoul(argv[4], &end, 10);
        iters = (end != argv[4]) ? (size_t)v : 12; // parse error -> 12
    }

    long t = now_ns();
    int rc = ivf_build_and_save(refs_path, out_path, n_clusters, iters);
    if (rc != 0) {
        fprintf(stderr, "[indexer] build/save failed rc=%d\n", rc);
        return 1;
    }
    long elapsed_ms = (now_ns() - t) / 1000000;
    fprintf(stderr, "[indexer] clusters=%zu iters=%zu in %ld ms\n",
            n_clusters, iters, elapsed_ms);
    fprintf(stderr, "[indexer] wrote %s\n", out_path);
    return 0;
}
