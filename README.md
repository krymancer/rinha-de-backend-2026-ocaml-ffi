# Rinha de Backend 2026 — OCaml (+ C FFI)

OCaml backend: OCaml owns the binary and request dispatch; a validated C engine
does the AVX2 IVF nearest-neighbour search (vpmaddwd), the epoll event loop, and
SCM_RIGHTS fd-passing — called via OCaml FFI. Removes the "no SIMD in OCaml" wall.

MIT.
