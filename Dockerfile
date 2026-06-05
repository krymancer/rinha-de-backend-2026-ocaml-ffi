# Build the C index (validated engine) + the OCaml binary (OCaml main + C-FFI engine).
FROM ocaml/opam:debian-12-ocaml-5.2 AS build
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc make curl ca-certificates libc6-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN opam install -y dune

ARG REFS_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz
RUN curl -fsSL "${REFS_URL}" -o refs.json.gz && gunzip refs.json.gz

# Bake the IVF index (RNHZIVF2) with the validated C indexer.
COPY engine ./engine
RUN gcc -O3 -march=haswell -mavx2 -std=gnu11 -D_GNU_SOURCE -o indexer \
    engine/indexer.c engine/ivf.c engine/vec.c engine/net.c -lm
ARG N_CLUSTERS=2048
ARG KMEANS_ITERS=12
RUN ./indexer refs.json /index.bin ${N_CLUSTERS} ${KMEANS_ITERS} && rm -f refs.json

# Build the OCaml binary (links the C engine via dune foreign_stubs).
COPY dune-project ./
COPY bin ./bin
RUN opam exec -- dune build ./bin/main.exe

FROM debian:bookworm-slim AS final
COPY --from=build /build/_build/default/bin/main.exe /rinha
COPY --from=build /index.bin /index.bin
ENTRYPOINT ["/rinha"]
