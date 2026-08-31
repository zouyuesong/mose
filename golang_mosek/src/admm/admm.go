// Package admm implements a basic OSQP-style ADMM solver for convex QPs
//
//	min 1/2 x'Px + q'x   s.t.  l <= Ax <= u
//
// with P diagonal PSD. It follows the splitting of Stellato et al. (OSQP,
// Math. Prog. Comp. 2020): ADMM on (x, z) with sigma-regularized x-update
//
//	(P + sigma*I + rho*A'A) x~ = sigma*x - q + rho*A'(z - y)
//	z+ = clamp(alpha*A x~ + (1-alpha)z + y, l, u)
//	y+ = y + alpha*A x~ + (1-alpha)z - z+
//
// The KKT matrix is solved EXACTLY via a diagonal + low-rank (Woodbury)
// factorization. This is valid because the model builder orders rows so that
// rows [0:GeneralStart) are "+-1 pair" rows whose A'A cross terms cancel
// (they contribute to the diagonal only), while rows [GeneralStart:) form the
// low-rank block G (k x n, k small):
//
//	P + sigma*I + rho*A'A = D + rho*G'G,   D diagonal,
//	(D + rho G'G)^-1 = D^-1 - rho D^-1 G' (I + rho G D^-1 G')^-1 G D^-1
//
// so per-iteration cost is O(nnz(A)) plus a dense k x k triangular solve.
// See ./admm.md (repo root) for the full survey and improvement plan.
package admm

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// Row is one sparse constraint row (Idx strictly increasing).
type Row struct {
	Idx []int32
	Val []float64
}

// Problem is the QP data. PDiag/Q/L/U must have len N (PDiag >= 0).
// Rows [0:GeneralStart) must be +-1 pair rows with cancelling cross terms
// (builder guarantee); rows [GeneralStart:) are arbitrary sparse rows.
type Problem struct {
	N            int
	PDiag        []float64
	Q            []float64
	Rows         []Row
	L, U         []float64
	GeneralStart int
	// Presolve is opaque solver-side metadata set by the model builder
	// (fixed-variable elimination bookkeeping); the solver never touches it.
	Presolve interface{}
}

// Settings holds algorithm parameters.
type Settings struct {
	Rho          float64 // ADMM penalty
	Sigma        float64 // x-update regularization
	Alpha        float64 // over-relaxation in [0,2)
	EpsAbs       float64 // absolute residual tolerance
	EpsRel       float64 // relative residual tolerance
	MaxIter      int
	CheckEvery   int // residual check period
	Verbose      bool
	StatsEvery   int // progress print period when Verbose
	X0, Z0, Y0   []float64
	Scaling      bool // Ruiz diagonal equilibration (basic: off)
	ScalingIters int

	// Checkpoint callback: called every CheckpointEvery iterations with a
	// copy of the current x-iterate. Returning true stops the iteration
	// (status becomes "checkpoint"). Used by callers to run early polish.
	Checkpoint      func(snap *Solution) bool
	CheckpointEvery int

	// Adaptive rho (OSQP residual balancing): every RhoInterval iterations,
	// compare tolerance-normalized primal/dual residuals and scale rho by
	// RhoScale (factor^+-1), clamped to [RhoMin, RhoMax]. 0 disables.
	RhoAdaptive bool
	RhoInterval int
	RhoScale    float64
	RhoMin      float64
	RhoMax      float64
}

// Solution is the solver output.
type Solution struct {
	X                  []float64
	Y                  []float64
	Status             string
	Iter               int
	PrimalRes, DualRes float64
	EpsPrim, EpsDua    float64
	Obj                float64
	Duration           time.Duration
	Factorizations     int
}

// DefaultSettings returns the basic-version parameters: fixed rho, no
// relaxation, no scaling (improvements land on top of this baseline).
func DefaultSettings() Settings {
	return Settings{
		Rho:          1e-4,
		Sigma:        1e-6,
		Alpha:        1.0,
		EpsAbs:       1e-6,
		EpsRel:       1e-8,
		MaxIter:      1000000,
		CheckEvery:   25,
		StatsEvery:   5000,
		Scaling:      false,
		ScalingIters: 10,
		RhoAdaptive:  true,
		RhoInterval:  500,
		RhoScale:     5.0,
		RhoMin:       1e-9,
		RhoMax:       1e6,
	}
}

// woodbury holds the factorization of M = D + rho*G'G.
type woodbury struct {
	n, k   int
	rho    float64
	dtilde []float64 // D = PDiag + sigma + rho*diag(A'A)
	invd   []float64 // 1/dtilde
	gRows  []Row     // general rows
	// flat CSR of G (avoids slice-of-slice indirection in the hot loop)
	gPtr  []int32
	gIdx  []int32
	gVal  []float64
	S     []float64 // I + rho*G D^-1 G' (k*k, row-major)
	chol  []float64 // Cholesky factor of S (lower, row-major)
	bufA  []float64
	bufV  []float64
	bufS  []float64
	bufGt []float64
}

func newWoodbury(p *Problem, rho, sigma float64) (*woodbury, error) {
	n := p.N
	w := &woodbury{n: n, rho: rho}
	// D = PDiag + sigma + rho*diag contributions of the PAIR rows only.
	// General rows' full outer products (diagonal squares included) belong to
	// the low-rank term rho*G'G and must NOT be folded into D (double count).
	ad := make([]float64, n)
	for _, r := range p.Rows[:p.GeneralStart] {
		for t, idx := range r.Idx {
			v := r.Val[t]
			ad[idx] += v * v
		}
	}
	w.dtilde = make([]float64, n)
	for j := 0; j < n; j++ {
		w.dtilde[j] = p.PDiag[j] + sigma + rho*ad[j]
		if !(w.dtilde[j] > 0) {
			w.dtilde[j] = 1e-12
		}
	}
	w.gRows = p.Rows[p.GeneralStart:]
	w.k = len(w.gRows)
	k := w.k
	w.bufA = make([]float64, n)
	w.bufV = make([]float64, k)
	w.bufS = make([]float64, k)
	w.bufGt = make([]float64, n)
	w.invd = make([]float64, n)
	for j := 0; j < n; j++ {
		w.invd[j] = 1.0 / w.dtilde[j]
	}
	{ // flat CSR of the general block
		nnz := 0
		for _, r := range w.gRows {
			nnz += len(r.Idx)
		}
		w.gPtr = make([]int32, k+1)
		w.gIdx = make([]int32, 0, nnz)
		w.gVal = make([]float64, 0, nnz)
		for i, r := range w.gRows {
			w.gPtr[i] = int32(len(w.gIdx))
			w.gIdx = append(w.gIdx, r.Idx...)
			w.gVal = append(w.gVal, r.Val...)
		}
		w.gPtr[k] = int32(len(w.gIdx))
	}
	if k == 0 {
		return w, nil
	}
	// T = G D^-1 G' accumulated row-pair-wise by merge (rows sorted).
	invd := make([]float64, n)
	for j := 0; j < n; j++ {
		invd[j] = 1.0 / w.dtilde[j]
	}
	S := make([]float64, k*k)
	for i := 0; i < k; i++ {
		for j := i; j < k; j++ {
			d := weightedRowDot(w.gRows[i], w.gRows[j], invd)
			S[i*k+j] = d
			if j != i {
				S[j*k+i] = d
			}
		}
	}
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			v := 0.0
			if i == j {
				v = 1.0
			}
			S[i*k+j] = v + rho*S[i*k+j]
		}
	}
	w.S = S
	ch := make([]float64, k*k)
	copy(ch, S)
	if !cholFactor(ch, k) {
		return nil, fmt.Errorf("admm: woodbury Cholesky failed (k=%d)", k)
	}
	w.chol = ch
	return w, nil
}

func weightedRowDot(a, b Row, invd []float64) float64 {
	s := 0.0
	ia, ib := 0, 0
	for ia < len(a.Idx) && ib < len(b.Idx) {
		switch {
		case a.Idx[ia] == b.Idx[ib]:
			s += a.Val[ia] * b.Val[ib] * invd[a.Idx[ia]]
			ia++
			ib++
		case a.Idx[ia] < b.Idx[ib]:
			ia++
		default:
			ib++
		}
	}
	return s
}

// solve computes z = M^-1 r.
func (w *woodbury) solve(z, r []float64) {
	n := w.n
	if w.k == 0 {
		for j := 0; j < n; j++ {
			z[j] = r[j] / w.dtilde[j]
		}
		return
	}
	a := w.bufA
	invd := w.invd
	for j := 0; j < n; j++ {
		a[j] = r[j] * invd[j]
	}
	v := w.bufV
	gPtr, gIdx, gVal := w.gPtr, w.gIdx, w.gVal
	for i := 0; i < w.k; i++ {
		s := 0.0
		for t := gPtr[i]; t < gPtr[i+1]; t++ {
			s += gVal[t] * a[gIdx[t]]
		}
		v[i] = s
	}
	ss := w.bufS
	cholSolve(w.chol, w.k, v, ss)
	rho := w.rho
	for j := 0; j < n; j++ {
		z[j] = a[j]
	}
	// z = a - rho * D^-1 (G' s): first Gt = G' s, then divide by D.
	gt := w.bufGt
	for j := 0; j < n; j++ {
		gt[j] = 0
	}
	for i := 0; i < w.k; i++ {
		si := ss[i]
		if si == 0 {
			continue
		}
		for t := gPtr[i]; t < gPtr[i+1]; t++ {
			gt[gIdx[t]] += si * gVal[t]
		}
	}
	for j := 0; j < n; j++ {
		z[j] -= rho * gt[j] * invd[j]
	}
}

// cholFactor factors a symmetric PD matrix (lower triangle used) in place.
func cholFactor(a []float64, k int) bool {
	for i := 0; i < k; i++ {
		for j := 0; j <= i; j++ {
			s := a[i*k+j]
			for l := 0; l < j; l++ {
				s -= a[i*k+l] * a[j*k+l]
			}
			if i == j {
				if s <= 0 {
					return false
				}
				a[i*k+i] = math.Sqrt(s)
			} else {
				a[i*k+j] = s / a[j*k+j]
			}
		}
	}
	return true
}

// cholSolve solves (L L') x = b given cholFactor output L.
func cholSolve(ch []float64, k int, b, x []float64) {
	for i := 0; i < k; i++ {
		s := b[i]
		for l := 0; l < i; l++ {
			s -= ch[i*k+l] * x[l]
		}
		x[i] = s / ch[i*k+i]
	}
	for i := k - 1; i >= 0; i-- {
		s := x[i]
		for l := i + 1; l < k; l++ {
			s -= ch[l*k+i] * x[l]
		}
		x[i] = s / ch[i*k+i]
	}
}

// rowIdxGlobal/rowValGlobal hold the flat CSR built once per solveRaw call
// (kept at package level purely to avoid re-plumbing the loops below).
var rowIdxGlobal []int32
var rowValGlobal []float64

// absMax returns max |v[i]|.
func absMax(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

// Solve runs the ADMM iteration (optionally on a Ruiz-equilibrated problem).
func Solve(p *Problem, st *Settings) *Solution {
	t0 := time.Now()
	if !st.Scaling {
		sol := solveRaw(p, st)
		sol.Duration = time.Since(t0)
		return sol
	}
	D, E := ruizScaling(p, st.ScalingIters)
	ps := scaledProblem(p, D, E)
	st2 := *st
	if st.X0 != nil {
		st2.X0 = make([]float64, p.N)
		for j := range st2.X0 {
			st2.X0[j] = st.X0[j] / D[j]
		}
	}
	if st.Z0 != nil {
		st2.Z0 = make([]float64, len(p.Rows))
		copy(st2.Z0, st.Z0)
		for r := range st2.Z0 {
			st2.Z0[r] *= E[r]
		}
	}
	if st.Y0 != nil {
		st2.Y0 = make([]float64, len(p.Rows))
		copy(st2.Y0, st.Y0)
		for r := range st2.Y0 {
			st2.Y0[r] /= E[r]
		}
	}
	sol := solveRaw(ps, &st2)
	// map back: x = D x~, z = z~ / E, y = y~ * E
	for j := 0; j < p.N; j++ {
		sol.X[j] *= D[j]
	}
	if sol.Y != nil {
		for r := range sol.Y {
			sol.Y[r] *= E[r]
		}
	}
	// recompute residuals & objective in the ORIGINAL space
	residuals(p, sol, st)
	obj := 0.0
	for j := 0; j < p.N; j++ {
		obj += 0.5*p.PDiag[j]*sol.X[j]*sol.X[j] + p.Q[j]*sol.X[j]
	}
	sol.Obj = obj
	sol.Duration = time.Since(t0)
	return sol
}

// residuals fills sol.PrimalRes/DualRes/EpsPrim/EpsDua from (sol.X, sol.Y).
func residuals(p *Problem, sol *Solution, st *Settings) {
	n := p.N
	m := len(p.Rows)
	x, y := sol.X, sol.Y
	Ax := make([]float64, m)
	for r := 0; r < m; r++ {
		row := p.Rows[r]
		s := 0.0
		for t, idx := range row.Idx {
			s += row.Val[t] * x[idx]
		}
		Ax[r] = s
	}
	z := make([]float64, m)
	for r := 0; r < m; r++ {
		z[r] = math.Min(math.Max(Ax[r], p.L[r]), p.U[r])
	}
	rp := 0.0
	for r := 0; r < m; r++ {
		if d := math.Abs(Ax[r] - z[r]); d > rp {
			rp = d
		}
	}
	scratch := make([]float64, n)
	for j := 0; j < n; j++ {
		scratch[j] = p.Q[j]
	}
	for r := 0; r < m; r++ {
		if y[r] == 0 {
			continue
		}
		row := p.Rows[r]
		for t, idx := range row.Idx {
			scratch[idx] += row.Val[t] * y[r]
		}
	}
	qatyInf := absMax(scratch)
	pxInf := 0.0
	for j := 0; j < n; j++ {
		if a := math.Abs(p.PDiag[j] * x[j]); a > pxInf {
			pxInf = a
		}
	}
	for j := 0; j < n; j++ {
		scratch[j] = p.PDiag[j]*x[j] + p.Q[j] + st.Rho*(scratch[j]-p.Q[j])
	}
	sol.PrimalRes = rp
	sol.DualRes = absMax(scratch)
	sol.EpsPrim = st.EpsAbs + st.EpsRel*math.Max(absMax(Ax), absMax(z))
	sol.EpsDua = st.EpsAbs + st.EpsRel*math.Max(pxInf, qatyInf)
}

// ruizScaling computes diagonal D (columns) and E (rows) approximately
// equilibrating diag-block [P, A'; A, 0] in inf-norm.
func ruizScaling(p *Problem, iters int) (D, E []float64) {
	n := p.N
	m := len(p.Rows)
	D = make([]float64, n)
	E = make([]float64, m)
	for j := range D {
		D[j] = 1
	}
	for r := range E {
		E[r] = 1
	}
	colNorm := make([]float64, n)
	rowNorm := make([]float64, m)
	for it := 0; it < iters; it++ {
		for j := range colNorm {
			colNorm[j] = math.Sqrt(p.PDiag[j]) * D[j] // sqrt of scaled P_jj
		}
		for r, row := range p.Rows {
			er := E[r]
			for t, idx := range row.Idx {
				v := math.Abs(row.Val[t]) * D[idx] * er
				if v > rowNorm[r] {
					rowNorm[r] = v
				}
				if v > colNorm[idx] {
					colNorm[idx] = v
				}
			}
		}
		for j := 0; j < n; j++ {
			if colNorm[j] > 0 {
				D[j] /= math.Sqrt(colNorm[j])
			}
		}
		// recompute row norms under new D
		for r := range rowNorm {
			rowNorm[r] = 0
		}
		for r, row := range p.Rows {
			er := E[r]
			for t, idx := range row.Idx {
				v := math.Abs(row.Val[t]) * D[idx] * er
				if v > rowNorm[r] {
					rowNorm[r] = v
				}
			}
		}
		for r := 0; r < m; r++ {
			if rowNorm[r] > 0 {
				E[r] /= math.Sqrt(rowNorm[r])
			}
		}
		for j := range colNorm {
			colNorm[j] = 0
		}
		for r := range rowNorm {
			rowNorm[r] = 0
		}
	}
	return D, E
}

// scaledProblem returns A~ = E A D, P~ = D P D, q~ = D q, l~/u~ = E l/u.
func scaledProblem(p *Problem, D, E []float64) *Problem {
	ps := &Problem{
		N:            p.N,
		PDiag:        make([]float64, p.N),
		Q:            make([]float64, p.N),
		Rows:         make([]Row, len(p.Rows)),
		L:            make([]float64, len(p.Rows)),
		U:            make([]float64, len(p.Rows)),
		GeneralStart: p.GeneralStart,
	}
	for j := 0; j < p.N; j++ {
		ps.PDiag[j] = p.PDiag[j] * D[j] * D[j]
		ps.Q[j] = p.Q[j] * D[j]
	}
	for r, row := range p.Rows {
		idx := append([]int32{}, row.Idx...)
		val := make([]float64, len(row.Val))
		for t, j := range row.Idx {
			val[t] = row.Val[t] * D[j] * E[r]
		}
		ps.Rows[r] = Row{Idx: idx, Val: val}
		ps.L[r] = p.L[r] * E[r]
		ps.U[r] = p.U[r] * E[r]
	}
	return ps
}

// solveRaw runs the ADMM iteration on the given problem (no scaling).
func solveRaw(p *Problem, st *Settings) *Solution {
	n := p.N
	m := len(p.Rows)
	sol := &Solution{X: make([]float64, n)}
	rho := st.Rho
	w, err := newWoodbury(p, rho, st.Sigma)
	if err != nil {
		sol.Status = "error: " + err.Error()
		return sol
	}
	x := make([]float64, n) // x^{k-1} for the sigma term
	z := make([]float64, m)
	y := make([]float64, m)
	if st.X0 != nil {
		copy(x, st.X0)
	}
	if st.Z0 != nil {
		copy(z, st.Z0)
	}
	if st.Y0 != nil {
		copy(y, st.Y0)
	}
	rhs := make([]float64, n)
	xt := make([]float64, n)
	Ax := make([]float64, m)
	scratch := make([]float64, n)
	alpha := st.Alpha
	beta := 1.0 - alpha

	// flat CSR of the whole constraint matrix: the two per-iteration passes
	// (A'(z-y) and Ax) dominate runtime; slice-of-slice indirection costs
	// ~2x vs contiguous arrays
	rowPtr := make([]int32, m+1)
	{
		nnz := 0
		for _, r := range p.Rows {
			nnz += len(r.Idx)
		}
		csrIdx := make([]int32, 0, nnz)
		csrVal := make([]float64, 0, nnz)
		for i, r := range p.Rows {
			rowPtr[i] = int32(len(csrIdx))
			csrIdx = append(csrIdx, r.Idx...)
			csrVal = append(csrVal, r.Val...)
		}
		rowPtr[m] = int32(len(csrIdx))
		// stash for the passes below via closure-free locals
		passIdx, passVal := csrIdx, csrVal
		_ = passIdx
		_ = passVal
		rowIdxGlobal = csrIdx
		rowValGlobal = csrVal
	}
	defer func() { rowIdxGlobal, rowValGlobal = nil, nil }()

	sol.Iter = 0
	for it := 1; it <= st.MaxIter; it++ {
		// rhs = sigma*x - q + rho*A'(z - y)
		for j := 0; j < n; j++ {
			rhs[j] = st.Sigma*x[j] - p.Q[j]
		}
		csrIdx := rowIdxGlobal
		csrVal := rowValGlobal
		for r := 0; r < m; r++ {
			d := rho * (z[r] - y[r])
			if d == 0 {
				continue
			}
			for t := rowPtr[r]; t < rowPtr[r+1]; t++ {
				rhs[csrIdx[t]] += d * csrVal[t]
			}
		}
		w.solve(xt, rhs)
		// Ax~
		for r := 0; r < m; r++ {
			s := 0.0
			for t := rowPtr[r]; t < rowPtr[r+1]; t++ {
				s += rowValGlobal[t] * xt[rowIdxGlobal[t]]
			}
			Ax[r] = s
		}
		// z+ / y+ (OSQP unscaled-y iteration; y converges to dual/rho)
		for r := 0; r < m; r++ {
			tz := alpha*Ax[r] + beta*z[r] + y[r]
			if tz < p.L[r] {
				tz = p.L[r]
			} else if tz > p.U[r] {
				tz = p.U[r]
			}
			y[r] += alpha*Ax[r] + beta*z[r] - tz
			z[r] = tz
		}
		x, xt = xt, x
		sol.Iter = it

		if math.IsNaN(z[0]) || math.IsInf(z[0], 0) {
			sol.Status = "numeric_error"
			copy(sol.X, x)
			return sol
		}

		if st.Checkpoint != nil && st.CheckpointEvery > 0 && it%st.CheckpointEvery == 0 {
			snap := &Solution{X: append([]float64(nil), x...), PrimalRes: sol.PrimalRes, DualRes: sol.DualRes, EpsPrim: sol.EpsPrim, EpsDua: sol.EpsDua, Iter: it}
			if st.Checkpoint(snap) {
				sol.Status = "checkpoint"
				sol.Iter = it
				copy(sol.X, x)
				sol.Y = y
				return sol
			}
		}
		if it%st.CheckEvery == 0 || it == 1 {
			// Per-row scaled stopping criterion (mixed-scale problems): a row
			// with bounds ~1e6 must not dictate the absolute tolerance of rows
			// with bounds ~1e-6, and vice versa. Converged iff for every row r
			//   |Ax_r - z_r| <= epsAbs + epsRel * max(|Ax_r|, |z_r|)
			// and for every dual coordinate j
			//   |mu_j|       <= epsAbs + epsRel * max(|Px_j|, |q_j + (A'y)_j|)
			converged := true
			rpMax := 0.0
			for r := 0; r < m; r++ {
				viol := math.Abs(Ax[r] - z[r])
				if viol > rpMax {
					rpMax = viol
				}
				if converged {
					tol := st.EpsAbs + st.EpsRel*math.Max(math.Abs(Ax[r]), math.Abs(z[r]))
					if viol > tol {
						converged = false
					}
				}
			}
			// A'y into scratch, then dual residual = Px + q + rho*A'y
			// (y is the scaled dual: true dual is rho*y)
			for j := 0; j < n; j++ {
				scratch[j] = p.Q[j]
			}
			for r := 0; r < m; r++ {
				if y[r] == 0 {
					continue
				}
				row := p.Rows[r]
				for t, idx := range row.Idx {
					scratch[idx] += row.Val[t] * y[r]
				}
			}
			rd := 0.0
			for j := 0; j < n; j++ {
				qaty := scratch[j]
				mu := p.PDiag[j]*x[j] + p.Q[j] + rho*(qaty-p.Q[j])
				if a := math.Abs(mu); a > rd {
					rd = a
				}
				tolj := st.EpsAbs + st.EpsRel*math.Max(math.Abs(p.PDiag[j]*x[j]), math.Abs(qaty))
				if math.Abs(mu) > tolj {
					converged = false
				}
			}
			rp := rpMax
			epsP := st.EpsAbs + st.EpsRel*math.Max(absMax(Ax), absMax(z))
			pxInf := 0.0
			for j := 0; j < n; j++ {
				if a := math.Abs(p.PDiag[j] * x[j]); a > pxInf {
					pxInf = a
				}
			}
			epsD := st.EpsAbs + st.EpsRel*math.Max(pxInf, absMax(scratch))
			sol.PrimalRes, sol.DualRes = rp, rd
			sol.EpsPrim, sol.EpsDua = epsP, epsD
			if st.Verbose && (it == 1 || it%st.StatsEvery == 0) {
				fmt.Printf("admm iter %8d: rho %9.3e pri %9.3e dua %9.3e (tols %8.2e/%8.2e) fact=%d\n",
					it, rho, rp, rd, epsP, epsD, sol.Factorizations)
			}
			if converged {
				sol.Status = "solved"
				break
			}
			// adaptive rho: balance tolerance-normalized residuals
			if st.RhoAdaptive && it%st.RhoInterval == 0 && epsP > 0 && epsD > 0 {
				ratioP := rp / epsP
				ratioD := rd / epsD
				newRho := rho
				if ratioP > 10*ratioD {
					newRho = math.Min(rho*st.RhoScale, st.RhoMax)
				} else if ratioD > 10*ratioP {
					newRho = math.Max(rho/st.RhoScale, st.RhoMin)
				}
				if newRho != rho {
					rho = newRho
					w2, err := newWoodbury(p, rho, st.Sigma)
					if err != nil {
						sol.Status = "error: " + err.Error()
						return sol
					}
					w = w2
					sol.Factorizations++
				}
			}
		}
	}
	if sol.Status == "" {
		sol.Status = "max_iter"
	}
	copy(sol.X, x)
	sol.Y = y
	obj := 0.0
	for j := 0; j < n; j++ {
		obj += 0.5*p.PDiag[j]*x[j]*x[j] + p.Q[j]*x[j]
	}
	sol.Obj = obj
	sol.Factorizations = 1
	return sol
}

// DebugRuiz exposes ruizScaling for tests/diagnostics.
func DebugRuiz(p *Problem, iters int) (D, E []float64) { return ruizScaling(p, iters) }

// DebugSolveKKT exposes the factorization for testing: returns M^{-1} r with
// M = P + sigma*I + rho*A'A as computed by the Woodbury path.
func DebugSolveKKT(p *Problem, st *Settings, r []float64) ([]float64, bool) {
	w, err := newWoodbury(p, st.Rho, st.Sigma)
	if err != nil {
		return nil, false
	}
	z := make([]float64, p.N)
	w.solve(z, r)
	return z, true
}

// ValidateRowOrder checks the builder contract: pair rows touch <= 2 vars,
// all row indices strictly increasing. Debug helper.
func ValidateRowOrder(p *Problem) error {
	for r, row := range p.Rows {
		if !sort.SliceIsSorted(row.Idx, func(a, b int) bool { return row.Idx[a] < row.Idx[b] }) {
			return fmt.Errorf("row %d indices not sorted", r)
		}
		if r < p.GeneralStart && len(row.Idx) > 2 {
			return fmt.Errorf("pair row %d has %d entries", r, len(row.Idx))
		}
	}
	return nil
}
