/* OCaml<->C FFI: OCaml owns the binary + dispatch; the validated C engine
 * (vec.c/ivf.c/net.c/http.c/server.c/lb.c) does the AVX2 IVF search + epoll +
 * SCM_RIGHTS fd-passing. server_main/lb_main loop forever (never return). */
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <string.h>
#include <stdlib.h>

extern int server_main(int argc, char **argv);
extern int lb_main(int argc, char **argv);

CAMLprim value caml_engine_main(value mode, value rest) {
    CAMLparam2(mode, rest);
    int n = Wosize_val(rest);
    char **argv = malloc(sizeof(char *) * (n + 2));
    argv[0] = strdup("rinha");
    for (int i = 0; i < n; i++) argv[i + 1] = strdup(String_val(Field(rest, i)));
    argv[n + 1] = NULL;
    int argc = n + 1;
    int r = 1;
    const char *m = String_val(mode);
    if (strcmp(m, "server") == 0) r = server_main(argc, argv);
    else if (strcmp(m, "lb") == 0) r = lb_main(argc, argv);
    free(argv);
    CAMLreturn(Val_int(r));
}
