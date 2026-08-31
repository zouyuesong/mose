//go:build !nocosqp

// backend_osqp.go: OSQP (vendored C via csvport/cosqp) engine. The problem is
// the SAME canonical QP the pure-Go admm engine builds (buildADMMProblem);
// here it is converted to CSC and solved by OSQP with its adaptive-rho,
// Ruiz scaling and built-in polish.
package opt

import (
	"csvport/cosqp"
	"fmt"
	"math"
	"sort"
	"time"
)

func init() {
	registerSolver("osqp", solveOSQP)
}

// osqpPrescale enables the bound-magnitude prescaling experiment.
var osqpPrescale = false
var osqpScaleD []float64

func solveOSQP(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights,
	softLambda float64, relaxMode bool, elasticPenalty float64, verbose bool) (OPT_Result, error) {

	if math.Abs(softLambda) > 1e-10 {
		return nil, fmt.Errorf("osqp engine does not support soft-lambda yet (%g)", softLambda)
	}
	cols := inputs.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, constraints, weights)
	p := buildADMMProblem(cols, det, relaxMode, elasticPenalty)
	n := p.N
	m := len(p.Rows)

	// ---- build CSC of A (m x n, row-major rows -> column-major) ----
	colCount := make([]int64, n+1)
	for _, r := range p.Rows {
		for _, idx := range r.Idx {
			colCount[idx+1]++
		}
	}
	for j := 0; j < n; j++ {
		colCount[j+1] += colCount[j]
	}
	Ap := colCount
	Ai := make([]int64, 0, colCount[n])
	Ax := make([]float64, 0, colCount[n])
	// append (row, val) into column buckets, sorted by row index within column
	type rv struct {
		r int64
		v float64
	}
	buckets := make([][]rv, n)
	boxRowsOf := make([][]int32, n)
	for ri, r := range p.Rows {
		if len(r.Idx) == 1 && math.Abs(math.Abs(r.Val[0])-1) < 1e-12 {
			boxRowsOf[r.Idx[0]] = append(boxRowsOf[r.Idx[0]], int32(ri))
		}
		for t, idx := range r.Idx {
			buckets[idx] = append(buckets[idx], rv{int64(ri), r.Val[t]})
		}
	}
	for j := 0; j < n; j++ {
		sort.SliceStable(buckets[j], func(a, b int) bool { return buckets[j][a].r < buckets[j][b].r })
		for _, e := range buckets[j] {
			Ai = append(Ai, e.r)
			Ax = append(Ax, e.v)
		}
	}

	// ---- P: diagonal, upper triangular CSC ----
	Pp := make([]int64, n+1)
	Pi := make([]int64, 0, n)
	Px := make([]float64, 0, n)
	for j := 0; j < n; j++ {
		if p.PDiag[j] != 0 {
			Pi = append(Pi, int64(j))
			Px = append(Px, p.PDiag[j])
		}
		Pp[j+1] = int64(len(Pi))
	}

	q := p.Q
	l := make([]float64, m)
	u := make([]float64, m)
	for r := 0; r < m; r++ {
		l[r] = p.L[r]
		u[r] = p.U[r]
	}

	// ---- experimental prescaling ----
	// Normalize variable magnitudes (D) and row bound magnitudes (E). OSQP's
	// internal Ruiz equilibrates matrix ENTRIES (all +-1 here => no-op); the
	// ill conditioning of this problem class lives in the bound/value scales
	// (x ~ 1e5..1e6 vs boxes +-1e-6, positions +-1e9).
	if osqpPrescale {
		D := make([]float64, n)
		pre, _ := p.Presolve.(*PresolveInfo)
		for j := 0; j < n; j++ {
			// from the row structure: rows with a single +-1 entry on j are
			// its box rows; use their bound magnitude as the scale. x' = x/scl
			scl := 1.0
			for _, r := range boxRowsOf[j] {
				b := math.Max(math.Abs(p.L[r]), math.Abs(p.U[r]))
				if math.IsInf(b, 1) || b == 0 {
					continue
				}
				scl = b
			}
			D[j] = scl
		}
		if pre != nil {
			// aux blocks (u = |x|, g = |pos+x|) inherit their trade var's scale
			na := len(pre.Active)
			for a := 0; a < na; a++ {
				D[pre.TubBase+a] = D[a]
			}
		}
		for j := 0; j < n; j++ { // apply D: P, q
			qv := q[j] * D[j]
			q[j] = qv
		}
		for j := 0; j < n; j++ {
			p.PDiag[j] *= D[j] * D[j]
		}
		// rebuild Px with scaled P
		Px = Px[:0]
		Pi = Pi[:0]
		Pp = Pp[:0]
		Pp = append(Pp, 0)
		for j := 0; j < n; j++ {
			if p.PDiag[j] != 0 {
				Pi = append(Pi, int64(j))
				Px = append(Px, p.PDiag[j])
			}
			Pp = append(Pp, int64(len(Pi)))
		}
		// A' = A D  (scale CSC values by column)
		for j := 0; j < n; j++ {
			for t := Ap[j]; t < Ap[j+1]; t++ {
				Ax[t] *= D[j]
			}
		}
		// row scaling E and bounds
		for r := 0; r < m; r++ {
			b := math.Max(math.Abs(l[r]), math.Abs(u[r]))
			if math.IsInf(b, 1) || b == 0 {
				continue
			}
			e := 1.0 / b
			for j := 0; j < n; j++ {
				for t := Ap[j]; t < Ap[j+1]; t++ {
					if Ai[t] == int64(r) {
						Ax[t] *= e
					}
				}
			}
			l[r] *= e
			u[r] *= e
		}
		// remember D for unscaling (x = D x')
		osqpScaleD = D
	}

	t0 := time.Now()
	x, _, status, runTime, err := cosqp.Solve(n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u,
		1e-6, 1e-8, 200000, 25)
	wall := time.Since(t0)
	fmt.Printf("osqp engine: status=%d err=%v internalTime=%.1fms wall=%v prescale=%v\n",
		status, err, runTime*1e3, wall, osqpPrescale)
	if err != nil {
		return nil, err
	}
	if osqpPrescale {
		for j := 0; j < n; j++ {
			x[j] *= osqpScaleD[j]
		}
	}
	return resultFromReduced(cols, p, x)
}
