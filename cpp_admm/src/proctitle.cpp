// proctitle.cpp - process title masking, port of golang_mosek/src/main/proctitle.go.
// Two mechanisms (same as PostgreSQL): prctl(PR_SET_NAME) for /proc/pid/comm
// and overwriting the original argv string block for /proc/pid/cmdline.
// C gives us argv directly in main; the block is [argv[0], end of last arg).
#include <cstring>
#include <unistd.h>
#include <sys/prctl.h>

static const char kProcTitle[] = "jobd";

void setComm(const char *name) {
    char b[16] = {0};
    strncpy(b, name, 15);
    prctl(PR_SET_NAME, b, 0, 0, 0);
}

// Overwrites the argv block [argv[0] .. end of argv[argc-1]] with title.
// envp strings follow argv strings contiguously on the initial stack; we only
// wipe up to the end of the last argv element, then NUL-pad the remainder.
void setProcTitle(int argc, char **argv, const char *title) {
    if (argc <= 0 || argv == nullptr || argv[0] == nullptr) return;
    char *start = argv[0];
    char *last = argv[argc - 1];
    char *end = last + strlen(last);
    if (end <= start) return;
    size_t size = (size_t)(end - start);
    size_t tlen = strlen(title);
    if (tlen + 1 > size) tlen = size ? size - 1 : 0;
    memset(start, 0, size);
    memcpy(start, title, tlen);
}

void maskProcessTitle(int argc, char **argv) {
    setComm(kProcTitle);
    setProcTitle(argc, argv, kProcTitle);
}
