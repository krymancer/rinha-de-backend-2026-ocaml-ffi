(* Rinha de Backend 2026 — OCaml entry. OCaml parses args and dispatches into the
   validated C engine via FFI (the C does the AVX2 IVF search, epoll loop, and
   SCM_RIGHTS fd-passing — the parts OCaml has no native SIMD/epoll for). *)
external engine_main : string -> string array -> int = "caml_engine_main"

let () =
  let a = Sys.argv in
  if Array.length a < 2 then (prerr_endline "usage: rinha <server|lb> ..."; exit 1);
  let mode = a.(1) in
  let rest = Array.sub a 2 (Array.length a - 2) in
  exit (engine_main mode rest)
