// Package cosqp embeds the OSQP solver (v0.6.3, Apache 2.0, see LICENSE/
// NOTICE) as vendored C sources compiled via cgo. It exposes a minimal
// interface for problems already in OSQP canonical form:
//
//	min 0.5 x'Px + q'x   s.t.  l <= Ax <= u
//
// with P (upper triangular CSC) and A (CSC). Solver knobs: adaptive rho,
// Ruiz scaling and polish enabled; verbose off.
package cosqp

/*
#cgo CFLAGS: -I${SRCDIR}/include -DDLONG
#cgo LDFLAGS: -lm

#include "osqp.h"

static c_int solve_raw(c_float *Px, c_int *Pi, c_int *Pp, c_int P_nnz, c_int n,
                       c_float *q,
                       c_float *Ax, c_int *Ai, c_int *Ap, c_int A_nnz, c_int m,
                       c_float *l, c_float *u,
                       c_float eps_abs, c_float eps_rel, c_int max_iter, c_int check_term,
                       c_float rho_set, c_float sigma_set, c_int adapt_interval,
                       c_float *x_out, c_float *y_out, c_float *run_time, c_int *iter_out, c_float *obj_out) {
    c_int exitflag;
    OSQPWorkspace *work = OSQP_NULL;
    OSQPData *data;
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = n;
    data->m = m;
    data->P = csc_spalloc(n, n, P_nnz, 1, 0);
    data->P->nz = -1;
    for (c_int i = 0; i < P_nnz; i++) { data->P->x[i] = Px[i]; data->P->i[i] = Pi[i]; }
    for (c_int i = 0; i <= n; i++)    { data->P->p[i] = Pp[i]; }
    data->q = q;
    data->A = csc_spalloc(m, n, A_nnz, 1, 0);
    data->A->nz = -1;
    for (c_int i = 0; i < A_nnz; i++) { data->A->x[i] = Ax[i]; data->A->i[i] = Ai[i]; }
    for (c_int i = 0; i <= n; i++)    { data->A->p[i] = Ap[i]; }
    data->l = l;
    data->u = u;

    osqp_set_default_settings(settings);
    settings->verbose = 0;
    settings->eps_abs = eps_abs;
    settings->eps_rel = eps_rel;
    settings->max_iter = max_iter;
    settings->polish = 1;
    settings->adaptive_rho = 1;
    settings->check_termination = check_term;

    exitflag = osqp_setup(&work, data, settings);
    if (exitflag == 0) {
        exitflag = osqp_solve(work);
        if (work->solution != OSQP_NULL) {
            for (c_int i = 0; i < n; i++) x_out[i] = work->solution->x[i];
            for (c_int i = 0; i < m; i++) y_out[i] = work->solution->y[i];
        }
        *run_time = (c_float)work->info->run_time;
        *iter_out = work->info->iter;
        *obj_out = (c_float)work->info->obj_val;
        exitflag = work->info->status_val;
    }
    osqp_cleanup(work);
    c_free(data->P); c_free(data->A); c_free(data);
    c_free(settings);
    return exitflag;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Solve runs OSQP once. P must be upper-triangular CSC (n x n), A CSC (m x n).
// Returns x, y, the OSQP status value (1 = solved) and the internal runtime.
func SolveFull(n, m int, Px []float64, Pi, Pp []int64, q []float64,
	Ax []float64, Ai, Ap []int64, l, u []float64,
	epsAbs, epsRel float64, maxIter, checkTerm int,
	rho, sigma float64, adaptInterval int) (x, y []float64, statusVal int, runTime float64, iter int, err error) {

	x = make([]float64, n)
	y = make([]float64, m)
	runTime = 0

	iterOut := int32(0)
	objOut := 0.0
	rv := C.solve_raw(
		(*C.c_float)(unsafe.Pointer(&Px[0])),
		(*C.c_int)(unsafe.Pointer(&Pi[0])),
		(*C.c_int)(unsafe.Pointer(&Pp[0])),
		C.c_int(len(Px)), C.c_int(n),
		(*C.c_float)(unsafe.Pointer(&q[0])),
		(*C.c_float)(unsafe.Pointer(&Ax[0])),
		(*C.c_int)(unsafe.Pointer(&Ai[0])),
		(*C.c_int)(unsafe.Pointer(&Ap[0])),
		C.c_int(len(Ax)), C.c_int(m),
		(*C.c_float)(unsafe.Pointer(&l[0])),
		(*C.c_float)(unsafe.Pointer(&u[0])),
		C.c_float(epsAbs), C.c_float(epsRel), C.c_int(maxIter), C.c_int(checkTerm),
		C.c_float(rho), C.c_float(sigma), C.c_int(adaptInterval),
		(*C.c_float)(unsafe.Pointer(&x[0])),
		(*C.c_float)(unsafe.Pointer(&y[0])),
		(*C.c_float)(unsafe.Pointer(&runTime)),
		(*C.c_int)(unsafe.Pointer(&iterOut)),
		(*C.c_float)(unsafe.Pointer(&objOut)))
	statusVal = int(rv)
	iter = int(iterOut)
	_ = objOut
	if statusVal != 1 { // OSQP_SOLVED
		err = fmt.Errorf("osqp exit status %d (1=solved)", statusVal)
	}
	return x, y, statusVal, runTime, iter, err
}

// Solve is the simple entry with default knobs.
func Solve(n, m int, Px []float64, Pi, Pp []int64, q []float64,
	Ax []float64, Ai, Ap []int64, l, u []float64,
	epsAbs, epsRel float64, maxIter, checkTerm int) (x, y []float64, statusVal int, runTime float64, err error) {
	x, y, statusVal, runTime, _, err = SolveFull(n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u,
		epsAbs, epsRel, maxIter, checkTerm, 0.1, 1e-6, 0)
	return
}
