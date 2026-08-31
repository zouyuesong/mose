// Process title masking.
//
// This file rewrites how the process appears in /proc and thus in top/htop/ps:
// the binary is a portfolio optimizer whose command line exposes CSV paths;
// rewriting comm + cmdline shows a neutral name ("jobd") to other users.
//
// Two mechanisms, same as used by PostgreSQL and Python's setproctitle:
//  1. comm   (/proc/<pid>/comm)  - 15 chars, shown by top's default COMMAND
//     column and by `ps -o comm`. Set via prctl(PR_SET_NAME).
//  2. cmdline(/proc/<pid>/cmdline) - the full "program args" string shown by
//     htop and `ps -o args`. The kernel does not offer a setter; instead we
//     locate the original argv[] string block on the stack (argv and environ
//     strings are laid out contiguously by the loader) and overwrite it in
//     place.
//
// Limitations (by design, this is masking, not hiding):
//   - the process entry itself, user, CPU% and memory are still visible;
//   - root can always see through it;
//   - /proc/<pid>/exe still points at the real binary.
//
// NOTE: setProcTitle destroys the memory behind os.Args, so it must be called
// only AFTER all flag parsing is done (see main.go).
package main

import (
	"os"
	"reflect"
	"syscall"
	"unsafe"
)

// procTitle is the neutral name shown in top/htop/ps after masking.
const procTitle = "jobd"

// setComm sets the kernel task name (comm, max 15 chars + NUL) via
// prctl(PR_SET_NAME). This affects /proc/<pid>/comm and therefore the default
// COMMAND column of top and `ps -o comm`.
func setComm(name string) {
	b := make([]byte, 16) // PR_SET_NAME copies at most 16 bytes incl. NUL
	copy(b, name)
	syscall.Syscall6(syscall.SYS_PRCTL, syscall.PR_SET_NAME, uintptr(unsafe.Pointer(&b[0])), 0, 0, 0, 0)
}

// stringDataPtr returns the address of the backing array of a string.
// For os.Args elements this points into the original argv string block the
// kernel handed to the process, which is exactly what /proc/<pid>/cmdline reads.
func stringDataPtr(s string) uintptr {
	return (*reflect.StringHeader)(unsafe.Pointer(&s)).Data
}

// setProcTitle overwrites the argv string block so that /proc/<pid>/cmdline
// (shown by htop and `ps -o args`) displays only title.
//
// The loader places all argv strings (and then environ strings) contiguously
// on the initial stack. We compute the block as [addr(argv[0]),
// end(argv[argc-1])), zero it out entirely, then write the new title. Zeroed
// tail bytes terminate cmdline early, so no stale arguments leak.
func setProcTitle(title string) {
	if len(os.Args) == 0 {
		return
	}
	start := stringDataPtr(os.Args[0])
	last := os.Args[len(os.Args)-1]
	end := stringDataPtr(last) + uintptr(len(last))
	if end <= start {
		return
	}
	size := end - start
	if uintptr(len(title))+1 > size {
		title = title[:size-1] // never write past the original block
	}
	buf := (*[1 << 30]byte)(unsafe.Pointer(start))
	for i := uintptr(0); i < size; i++ {
		buf[i] = 0 // wipe original program name and all arguments
	}
	copy(buf[:], title)
}
