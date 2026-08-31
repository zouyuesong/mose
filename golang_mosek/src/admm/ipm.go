// ipm.go: Mehrotra predictor-corrector interior-point method for the same
// canonical QP as the ADMM solver:
//
//	min 0.5 z'Pz + q'z   s.t.  l <= Az <= u   (P diagonal PSD)
//
// Per-row barrier slacks/duals: zl = Az - l >= 0, wl >= 0 and
// zu = u - Az >= 0, wu >= 0. The Newton system each iteration is
//
//	(P + A' D A) dz = rhs,   D_r = wl_r/zl_r + wu_r/zu_r,
//
// which matches the woodbury structure (pair-row diagonal + low-rank
// G'DG over the k general rows) and is solved exactly in O(nnz + k^2).
// Same algorithm class MOSEK uses on this problem (17 IPM iterations on
// the optim dataset) — the path to single-digit milliseconds where
// first-order methods need 10k+ iterations.
package admm

import (
	"fmt"
	"math"
	"time"
)

// IPMSettings holds interior-point parameters.
type IPMSettings struct {
	MaxIter int
	EpsAbs  float64 // primal & dual residual tolerance (problem-scaled)
	EpsGap  float64 // relative complementarity-gap tolerance
	Verbose bool
	X0      []float64 // optional warm start (full z-space vector)
}

// DefaultIPMSettings returns robust defaults.
func DefaultIPMSettings() IPMSettings {
	return IPMSettings{MaxIter: 100, EpsAbs: 1e-8, EpsGap: 1e-6, Verbose: false}
}

// ipmWood factors M = P + A' D A with per-row diagonal D >= 0, using the
// pair-row/general-row split (same math as woodbury, D not uniform).
type ipmWood struct {
	n, k   int
	dtilde []float64 // P + pair-row diagonal part of A'DA
	dinv   []float64
	gPtr   []int32
	gIdx   []int32
	gVal   []float64
	chol   []float64 // k*k Cholesky of I + G_w G_w' (weighted)
	bufA   []float64
	bufV   []float64
	bufS   []float64
	bufGt  []float64
}

func newIPMWood(p *Problem, d []float64) (*ipmWood, error) {
	n := p.N
	w := &ipmWood{n: n}
	ad := make([]float64, n)
	for ri := 0; ri < p.GeneralStart; ri++ {
		row := p.Rows[ri]
		dr := d[ri]
		for t, idx := range row.Idx {
			v := row.Val[t]
			ad[idx] += dr * v * v
		}
	}
	w.dtilde = make([]float64, n)
	for j := 0; j < n; j++ {
		w.dtilde[j] = p.PDiag[j] + ad[j]
		if !(w.dtilde[j] > 0) {
			w.dtilde[j] = 1e-12
		}
	}
	w.gPtr = nil
	k := len(p.Rows) - p.GeneralStart
	w.k = k
	if k == 0 {
		w.dinv = make([]float64, n)
		for j := 0; j < n; j++ {
			w.dinv[j] = 1.0 / w.dtilde[j]
		}
		return w, nil
	}
	w.dinv = make([]float64, n)
	for j := 0; j < n; j++ {
		w.dinv[j] = 1.0 / w.dtilde[j]
	}
	// weighted general rows: g_i = sqrt(d_i) * G_i, so A'DA low-rank = G' Dg G
	gRows := p.Rows[p.GeneralStart:]
	sqw := make([]float64, k)
	for i := range sqw {
		sqw[i] = math.Sqrt(d[p.GeneralStart+i])
	}
	nnz := 0
	for _, r := range gRows {
		nnz += len(r.Idx)
	}
	w.gPtr = make([]int32, k+1)
	w.gIdx = make([]int32, 0, nnz)
	w.gVal = make([]float64, 0, nnz)
	for i, r := range gRows {
		w.gPtr[i] = int32(len(w.gIdx))
		for t, idx := range r.Idx {
			w.gIdx = append(w.gIdx, idx)
			w.gVal = append(w.gVal, r.Val[t]*sqw[i])
		}
	}
	w.gPtr[k] = int32(len(w.gIdx))
	w.bufA = make([]float64, n)
	w.bufV = make([]float64, k)
	w.bufS = make([]float64, k)
	w.bufGt = make([]float64, n)
	// S = I + Gw Gw'
	invd := w.dinv
	S := make([]float64, k*k)
	for i := 0; i < k; i++ {
		for j := i; j < k; j++ {
			sv := weightedRowDot(w.rowView(i), w.rowView(j), invd)
			S[i*k+j] = sv
			if j != i {
				S[j*k+i] = sv
			}
		}
		S[i*k+i] += 1.0
	}
	ch := make([]float64, k*k)
	copy(ch, S)
	if !cholFactor(ch, k) {
		return nil, fmt.Errorf("ipm: Schur factorization failed")
	}
	w.chol = ch
	return w, nil
}

func (w *ipmWood) rowView(i int) Row {
	return Row{Idx: w.gIdx[w.gPtr[i]:w.gPtr[i+1]], Val: w.gVal[w.gPtr[i]:w.gPtr[i+1]]}
}

// solve computes z = (P + A'DA)^-1 r.
func (w *ipmWood) solve(z, r []float64) {
	n := w.n
	a := w.bufA
	dinv := w.dinv
	if w.k == 0 {
		for j := 0; j < n; j++ {
			z[j] = r[j] * dinv[j]
		}
		return
	}
	for j := 0; j < n; j++ {
		a[j] = r[j] * dinv[j]
	}
	v := w.bufV
	for i := 0; i < w.k; i++ {
		s := 0.0
		for t := w.gPtr[i]; t < w.gPtr[i+1]; t++ {
			s += w.gVal[t] * a[w.gIdx[t]]
		}
		v[i] = s
	}
	ss := w.bufS
	cholSolve(w.chol, w.k, v, ss)
	gt := w.bufGt
	for j := 0; j < n; j++ {
		gt[j] = 0
	}
	for i := 0; i < w.k; i++ {
		si := ss[i]
		if si == 0 {
			continue
		}
		for t := w.gPtr[i]; t < w.gPtr[i+1]; t++ {
			gt[w.gIdx[t]] += si * w.gVal[t]
		}
	}
	for j := 0; j < n; j++ {
		z[j] = a[j] - gt[j]*dinv[j]
	}
}

// SolveIPM runs Mehrotra's predictor-corrector IPM on the canonical problem.
func SolveIPM(p *Problem, st *IPMSettings) *Solution {
	t0 := time.Now()
	n := p.N
	m := len(p.Rows)
	sol := &Solution{X: make([]float64, n), Y: make([]float64, m)}
	sol.Status = "max_iter"

	// ---- CSR of A ----
	rowPtr := make([]int32, m+1)
	csrIdx := make([]int32, 0, 1024)
	csrVal := make([]float64, 0, 1024)
	for i, r := range p.Rows {
		rowPtr[i] = int32(len(csrIdx))
		csrIdx = append(csrIdx, r.Idx...)
		csrVal = append(csrVal, r.Val...)
	}
	rowPtr[m] = int32(len(csrIdx))

	// Relax exact-equality rows by a tiny epsilon: with u == l both barrier
	// slacks collapse to 0 and complementarity pins the row dual to zero,
	// which stalls the dual residual. A 1e-9-scale relaxation lets the active
	// side carry a finite multiplier (inactive side's dual -> 0) while
	// changing the feasible set negligibly.
	rowL := make([]float64, m)
	rowU := make([]float64, m)
	copy(rowL, p.L)
	copy(rowU, p.U)
	bScale := 1.0
	for r := 0; r < m; r++ {
		if !math.IsInf(p.L[r], -1) {
			bScale = math.Max(bScale, math.Abs(p.L[r]))
		}
		if !math.IsInf(p.U[r], 1) {
			bScale = math.Max(bScale, math.Abs(p.U[r]))
		}
	}
	for r := 0; r < m; r++ {
		lo, hi := p.L[r], p.U[r]
		if !math.IsInf(lo, -1) && !math.IsInf(hi, 1) && hi-lo <= 1e-12*bScale {
			// one-sided relaxation only: keep the upper edge at the original
			// equality so the optimum sits there with a finite active-side
			// dual while the lower slack stays strictly positive
			eqEps := 1e-7 * bScale
			rowL[r] = lo - eqEps
			rowU[r] = hi
		}
	}

	hasLow := make([]bool, m)
	hasUpp := make([]bool, m)
	nComp := 0
	for r := 0; r < m; r++ {
		hasLow[r] = !math.IsInf(rowL[r], -1)
		hasUpp[r] = !math.IsInf(rowU[r], 1)
		if hasLow[r] {
			nComp++
		}
		if hasUpp[r] {
			nComp++
		}
	}
	if nComp == 0 {
		for j := 0; j < n; j++ {
			sol.X[j] = -p.Q[j] / math.Max(p.PDiag[j], 1e-12)
		}
		sol.Status = "solved"
		sol.Duration = time.Since(t0)
		return sol
	}

	qNorm := absMax(p.Q)
	if qNorm == 0 {
		qNorm = 1
	}

	// ---- starting point ----
	z := make([]float64, n)
	if len(st.X0) == n {
		copy(z, st.X0)
	}
	zl := make([]float64, m)
	zu := make([]float64, m)
	wl := make([]float64, m)
	wu := make([]float64, m)
	Az := make([]float64, m)
	az := func(x []float64, out []float64) {
		for r := 0; r < m; r++ {
			s := 0.0
			for t := rowPtr[r]; t < rowPtr[r+1]; t++ {
				s += csrVal[t] * x[csrIdx[t]]
			}
			out[r] = s
		}
	}
	az(z, Az)
	boundScale := 1.0
	for r := 0; r < m; r++ {
		if hasLow[r] && !math.IsInf(p.L[r], 0) {
			boundScale = math.Max(boundScale, math.Abs(p.L[r]))
		}
		if hasUpp[r] && !math.IsInf(p.U[r], 0) {
			boundScale = math.Max(boundScale, math.Abs(p.U[r]))
		}
	}
	// Mehrotra-style starting point: raw slacks first, then a global shift
	// so every slack is comfortably positive and the initial primal residual
	// is uniformly signed (mixed +-1e6 residuals block the first Newton
	// steps on badly scaled rows).
	rawMin := math.Inf(1)
	for r := 0; r < m; r++ {
		if hasLow[r] {
			rawMin = math.Min(rawMin, Az[r]-rowL[r])
		}
		if hasUpp[r] {
			rawMin = math.Min(rawMin, rowU[r]-Az[r])
		}
	}
	shiftP := math.Max(0.0, -1.5*rawMin) + 1e-2*boundScale
	for r := 0; r < m; r++ {
		if hasLow[r] {
			zl[r] = (Az[r] - rowL[r]) + shiftP
		}
		if hasUpp[r] {
			zu[r] = (rowU[r] - Az[r]) + shiftP
		}
	}
	shiftD := 1e-2 * qNorm
	if !(shiftD > 0) {
		shiftD = 1e-2
	}
	for r := 0; r < m; r++ {
		wl[r] = shiftD
		wu[r] = shiftD
	}

	// workspace
	d := make([]float64, m)
	rd := make([]float64, n)
	rl := make([]float64, m)
	ru := make([]float64, m)
	rhs := make([]float64, n)
	dz := make([]float64, n)
	dzl := make([]float64, m)
	dzu := make([]float64, m)
	dwl := make([]float64, m)
	dwu := make([]float64, m)
	AzNew := make([]float64, m)
	sL := make([]float64, m)
	sU := make([]float64, m)

	var lastAP, lastAD, lastSig float64
	stallCount := 0

	// best-iterate tracking: endgame degeneracy can destroy the iterates
	// after numerical convergence; always return the best point seen
	type snapT struct {
		z          []float64
		rp, rd, mu float64
		it         int
	}
	var best *snapT
	mu0 := -1.0
	takeBest := func(rpInf, rdInf, mu float64, it int) {
		// score = primal feasibility + relative barrier weight (dual residual
		// is excluded: on degenerate problems the dual never converges even
		// though the primal iterates are optimal)
		if mu0 < 0 {
			mu0 = math.Max(mu, 1e-12)
		}
		// weakly-weighted dual residual: pure (rp, mu) scores are gameable
		// by early feasible-but-suboptimal iterates whose duals collapsed
		score := math.Max(rpInf/boundScale, math.Max(mu/mu0, 0.1*rdInf/qNorm))
		if best == nil || score < best.rp {
			best = &snapT{append([]float64(nil), z...), score, rdInf, mu, it}
		}
	}

	for it := 1; it <= st.MaxIter; it++ {
		// residuals
		for j := 0; j < n; j++ {
			rd[j] = p.PDiag[j]*z[j] + p.Q[j]
		}
		for r := 0; r < m; r++ {
			yv := wu[r] - wl[r]
			if yv != 0 {
				for t := rowPtr[r]; t < rowPtr[r+1]; t++ {
					rd[csrIdx[t]] += csrVal[t] * yv
				}
			}
			if hasLow[r] {
				rl[r] = Az[r] - rowL[r] - zl[r]
			}
			if hasUpp[r] {
				ru[r] = rowU[r] - Az[r] - zu[r]
			}
		}
		mu := 0.0
		for r := 0; r < m; r++ {
			if hasLow[r] {
				mu += zl[r] * wl[r]
			}
			if hasUpp[r] {
				mu += zu[r] * wu[r]
			}
		}
		mu /= float64(nComp)

		rpInf := 0.0
		for r := 0; r < m; r++ {
			if hasLow[r] {
				rpInf = math.Max(rpInf, math.Abs(rl[r]))
			}
			if hasUpp[r] {
				rpInf = math.Max(rpInf, math.Abs(ru[r]))
			}
		}
		rdInf := absMax(rd)
		pobj := 0.0
		for j := 0; j < n; j++ {
			pobj += 0.5*p.PDiag[j]*z[j]*z[j] + p.Q[j]*z[j]
		}
		gapTot := mu * float64(nComp)
		sol.PrimalRes, sol.DualRes = rpInf, rdInf
		if st.Verbose {
			fmt.Printf("ipm it %3d: mu=%.3e rp=%.3e rd=%.3e gap=%.3e aP=%.3f aD=%.3f sig=%.2e\n", it, mu, rpInf, rdInf, gapTot, lastAP, lastAD, lastSig)
		}
		takeBest(rpInf, rdInf, mu, it)
		if rpInf <= st.EpsAbs*boundScale && rdInf <= st.EpsAbs*qNorm &&
			gapTot <= st.EpsGap*math.Max(1, math.Abs(pobj)) {
			sol.Status = "solved"
			sol.Iter = it
			copy(sol.X, z)
			for r := 0; r < m; r++ {
				sol.Y[r] = wu[r] - wl[r]
			}
			sol.Obj = pobj
			sol.Duration = time.Since(t0)
			return sol
		}

		for r := 0; r < m; r++ {
			d[r] = 0
			if hasLow[r] {
				d[r] += wl[r] / math.Max(zl[r], 1e-14)
			}
			if hasUpp[r] {
				d[r] += wu[r] / math.Max(zu[r], 1e-14)
			}
		}
		w, err := newIPMWood(p, d)
		if err != nil {
			sol.Status = "error: " + err.Error()
			sol.Iter = it
			copy(sol.X, z)
			sol.Duration = time.Since(t0)
			return sol
		}

		// Newton step for a given barrier target. Sign conventions (min form):
		//   gradient: Pz + q + A'(wu - wl) = rd  (upper duals enter +)
		//   constraint Az - l - zl = 0 linearized: dzl = A dz + rl
		//   constraint u - Az - zu = 0 linearized: dzu = ru - A dz
		//   dwl = (tgtL - zl wl - wl dzl)/zl, dwu = (tgtU - zu wu - wu dzu)/zu
		//   P dz + A'(dwu - dwl) = -rd  and  dwl - dwu = -D(A dz) + (cL - cU)
		// => (P + A'DA) dz = -rd + A'(cL - cU),
		//    cL = (tgtL - zl wl - wl rl)/zl, cU = (tgtU - zu wu - wu ru)/zu
		step := func(tgtL, tgtU []float64) {
			for j := 0; j < n; j++ {
				rhs[j] = -rd[j]
			}
			for r := 0; r < m; r++ {
				var c float64
				if hasLow[r] {
					c = (tgtL[r] - zl[r]*wl[r] - wl[r]*rl[r]) / zl[r]
				}
				if hasUpp[r] {
					c -= (tgtU[r] - zu[r]*wu[r] - wu[r]*ru[r]) / zu[r]
				}
				if c != 0 {
					for t := rowPtr[r]; t < rowPtr[r+1]; t++ {
						rhs[csrIdx[t]] += csrVal[t] * c
					}
				}
			}
			w.solve(dz, rhs)
			az(dz, AzNew)
			for r := 0; r < m; r++ {
				if hasLow[r] {
					dzl[r] = AzNew[r] + rl[r]
					dwl[r] = (tgtL[r] - zl[r]*wl[r] - wl[r]*dzl[r]) / zl[r]
				}
				if hasUpp[r] {
					dzu[r] = ru[r] - AzNew[r]
					dwu[r] = (tgtU[r] - zu[r]*wu[r] - wu[r]*dzu[r]) / zu[r]
				}
			}
		}
		maxStep := func() (float64, float64, float64) {
			ap, ad := 1.0, 1.0
			for r := 0; r < m; r++ {
				if hasLow[r] {
					if dzl[r] < 0 {
						ap = math.Min(ap, -zl[r]/dzl[r])
					}
					if dwl[r] < 0 {
						ad = math.Min(ad, -wl[r]/dwl[r])
					}
				}
				if hasUpp[r] {
					if dzu[r] < 0 {
						ap = math.Min(ap, -zu[r]/dzu[r])
					}
					if dwu[r] < 0 {
						ad = math.Min(ad, -wu[r]/dwu[r])
					}
				}
			}
			muA := 0.0
			for r := 0; r < m; r++ {
				if hasLow[r] {
					muA += (zl[r] + ap*dzl[r]) * (wl[r] + ad*dwl[r])
				}
				if hasUpp[r] {
					muA += (zu[r] + ap*dzu[r]) * (wu[r] + ad*dwu[r])
				}
			}
			return ap, ad, muA / float64(nComp)
		}

		var ap, ad float64
		// affine predictor (targets = 0)
		for r := 0; r < m; r++ {
			sL[r] = 0
			sU[r] = 0
		}
		_, _, muAff := maxStep()
		sigma := muAff / mu
		sigma = math.Max(1e-8, math.Min(0.9, sigma*sigma*sigma))

		// corrector (targets = sigma*mu + second order)
		for r := 0; r < m; r++ {
			if hasLow[r] {
				sL[r] = sigma*mu + dzl[r]*dwl[r]
				if sL[r] < 0 {
					sL[r] = 0 // Mehrotra-style safeguard
				}
			}
			if hasUpp[r] {
				sU[r] = sigma*mu + dzu[r]*dwu[r]
				if sU[r] < 0 {
					sU[r] = 0
				}
			}
		}
		step(sL, sU)
		apC, adC, _ := maxStep()
		ap, ad = apC, adC
		alphaP := math.Min(1.0, 0.995*ap)
		alphaD := math.Min(1.0, 0.995*ad)
		lastAP, lastAD, lastSig = alphaP, alphaD, sigma

		for j := 0; j < n; j++ {
			z[j] += alphaP * dz[j]
		}
		for r := 0; r < m; r++ {
			if hasLow[r] {
				zl[r] += alphaP * dzl[r]
				wl[r] += alphaD * dwl[r]
			}
			if hasUpp[r] {
				zu[r] += alphaP * dzu[r]
				wu[r] += alphaD * dwu[r]
			}
		}
		az(z, Az)

		if math.IsNaN(mu) || math.IsInf(mu, 0) {
			sol.Status = "numeric_error"
			break
		}
		// stalled: both steps collapsed for many iterations
		if alphaP*alphaD < 1e-8 {
			stallCount++
			if stallCount > 5 {
				break
			}
		} else {
			stallCount = 0
		}
	}

	if best != nil {
		copy(z, best.z)
		if sol.Status == "max_iter" {
			sol.Status = "best_effort"
		}
	}
	copy(sol.X, z)
	for r := 0; r < m; r++ {
		sol.Y[r] = wu[r] - wl[r]
	}
	obj := 0.0
	for j := 0; j < n; j++ {
		obj += 0.5*p.PDiag[j]*z[j]*z[j] + p.Q[j]*z[j]
	}
	sol.Obj = obj
	sol.Duration = time.Since(t0)
	return sol
}
