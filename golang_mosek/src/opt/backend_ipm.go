//go:build !nocosqp

// backend_ipm.go: pure-Go interior-point engine (admm.SolveIPM) with a
// feasibility-repaired warm start:
//
//  1. start from x = 0 (or the analytic unconstrained optimum);
//  2. min-norm repair toward violated NET/NET_TRADE rows (small Schur solve
//     on the violated subset, a few passes with trade-box clamping);
//  3. hand the repaired point to the Mehrotra IPM as X0 (u/g aux blocks
//     consistent: u=|x|, g=|pos+x|).
//
// The repair keeps the IPM start near-feasible, avoiding the blocked-Newton
// pathology of deeply infeasible starts on badly scaled rows.
package opt

import (
	"csvport/admm"
	"fmt"
	"math"
	"time"
)

func init() {
	registerSolver("ipm", solveIPMEngine)
}

// analyticStart computes the unconstrained per-symbol optimum clipped into
// the trade box: for each of the 4 sign branches (sgn(x) in {+1,-1}) the
// unconstrained minimizer of 0.5*(lam+tlam)*x^2 + (lam*pos-alpha)*x + cost*|x|
// is x* = (alpha - lam*pos - cost*sgn) / (lam + tlam); evaluate each branch,
// keep the best feasible one.
func analyticStart(cols OPT_InputsColumns) []float64 {
	n := len(cols.Univ)
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		xl := math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
		xu := math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
		d := cols.Lambda[i] + cols.TradingLambda[i]
		if d <= 0 {
			x[i] = math.Max(xl, math.Min(xu, 0))
			continue
		}
		q := cols.Lambda[i]*cols.Position[i] - cols.Alpha[i] // minimize 0.5 d x^2 + q x + cost |x|
		best := math.Max(xl, math.Min(xu, 0))
		bestObj := math.Inf(1)
		for _, s := range []float64{1, -1} {
			xc := -(q + cols.TradingCost[i]*s) / d
			if xc < xl {
				xc = xl
			} else if xc > xu {
				xc = xu
			}
			// candidate must be on the assumed branch (or at its box edge)
			if (s > 0 && xc < 0) || (s < 0 && xc > 0) {
				continue
			}
			o := 0.5*d*xc*xc + q*xc + cols.TradingCost[i]*math.Abs(xc)
			if o < bestObj {
				bestObj = o
				best = xc
			}
		}
		x[i] = best
	}
	return x
}

// repairFeasibility returns a warm start over the ACTIVE x block: x=0 (or
// analytic) repaired toward the constraint set by min-norm steps on violated
// linear rows, with trade-box clamping between passes.
func repairFeasibility(cols OPT_InputsColumns, det []OPT_AssembledConstraintItem, pre *PresolveInfo, analytic bool) []float64 {
	na := len(pre.Active)
	x := make([]float64, na)
	if analytic {
		xa := analyticStart(cols)
		for a, i := range pre.Active {
			x[a] = xa[i]
		}
	}
	xl := make([]float64, na)
	xu := make([]float64, na)
	for a, i := range pre.Active {
		xl[a] = math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
		xu[a] = math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
	}
	clamp := func() {
		for a := 0; a < na; a++ {
			if x[a] < xl[a] {
				x[a] = xl[a]
			} else if x[a] > xu[a] {
				x[a] = xu[a]
			}
		}
	}
	clamp()

	// gather linear rows over active symbols: NET (pos-shifted) and NET_TRADE
	type linrow struct {
		idx []int32 // active indices (original TradeVariableIndices filtered)
		w   []float64
		lb  float64 // already pos-adjusted and fixed-folded
		ub  float64
	}
	var rows []linrow
	for _, con := range det {
		switch constraintType(con.ConstraintItem.ConstraintType) {
		case "NET":
			constAdj := 0.0
			for j, k := range con.TradeVariableIndices {
				constAdj += con.Weights[j] * cols.Position[k]
			}
			r := linrow{lb: con.ConstraintItem.LowerBound - constAdj, ub: con.ConstraintItem.UpperBound - constAdj}
			for j, k := range con.TradeVariableIndices {
				if !pre.Fixed[k] {
					r.idx = append(r.idx, pre.IdxOf[int(k)])
					r.w = append(r.w, con.Weights[j])
				} else {
					v := con.Weights[j] * pre.XFix[k]
					r.lb -= v
					r.ub -= v
				}
			}
			rows = append(rows, r)
		case "NET_TRADE":
			r := linrow{lb: con.ConstraintItem.LowerBound, ub: con.ConstraintItem.UpperBound}
			for j, k := range con.TradeVariableIndices {
				if !pre.Fixed[k] {
					r.idx = append(r.idx, pre.IdxOf[int(k)])
					r.w = append(r.w, con.Weights[j])
				} else {
					v := con.Weights[j] * pre.XFix[k]
					r.lb -= v
					r.ub -= v
				}
			}
			rows = append(rows, r)
		}
	}

	// ---- POCS: alternate projection onto linear rows and trade boxes ----
	// Precompute S = W W' + eps*I over the linear rows (k x k, one Cholesky).
	k := len(rows)
	if k == 0 {
		return x
	}
	S := make([]float64, k*k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			sv := 0.0
			for t1, a1 := range rows[i].idx {
				for t2, a2 := range rows[j].idx {
					if a1 == a2 {
						sv += rows[i].w[t1] * rows[j].w[t2]
					}
				}
			}
			S[i*k+j] = sv
		}
		S[i*k+i] += 1e-9 * math.Max(1, math.Abs(S[i*k+i]))
	}
	if !cholInPlace(S, k) {
		return x
	}
	// gross rows: sum w_k |pos_k + x_k| <= ub -> weighted L1-ball projection
	// of v = pos+x (Duchi's sorting algorithm)
	type grossrow struct {
		act []int32 // active indices
		w   []float64
		ub  float64 // fixed-folded
	}
	var grossRows []grossrow
	for _, con := range det {
		if constraintType(con.ConstraintItem.ConstraintType) == "GROSS" {
			g := grossrow{ub: con.ConstraintItem.UpperBound}
			for j, kk := range con.TradeVariableIndices {
				if !pre.Fixed[kk] {
					g.act = append(g.act, pre.IdxOf[int(kk)])
					g.w = append(g.w, con.Weights[j])
				} else {
					g.ub -= con.Weights[j] * math.Abs(cols.Position[kk]+pre.XFix[kk])
				}
			}
			grossRows = append(grossRows, g)
		}
	}
	projectGross := func() {
		for gi := range grossRows {
			g := &grossRows[gi]
			// current L1 mass
			mass := 0.0
			for t, a := range g.act {
				i := pre.Active[a]
				mass += g.w[t] * math.Abs(cols.Position[i]+x[a])
			}
			if mass <= g.ub {
				continue
			}
			// find theta: sum w max(|v| - theta w, 0) = ub  (bisection on
			// the sorted breakpoints |v|/w; weights here are > 0)
			lo, hi := 0.0, 0.0
			for t, a := range g.act {
				i := pre.Active[a]
				hi = math.Max(hi, math.Abs(cols.Position[i]+x[a])/g.w[t])
			}
			for bis := 0; bis < 60; bis++ {
				mid := 0.5 * (lo + hi)
				s := 0.0
				for t, a := range g.act {
					i := pre.Active[a]
					v := math.Abs(cols.Position[i]+x[a]) - mid*g.w[t]
					if v > 0 {
						s += g.w[t] * v
					}
				}
				if s > g.ub {
					lo = mid
				} else {
					hi = mid
				}
			}
			theta := 0.5 * (lo + hi)
			for t, a := range g.act {
				i := pre.Active[a]
				v := cols.Position[i] + x[a]
				av := math.Abs(v) - theta*g.w[t]
				if av < 0 {
					av = 0
				}
				if v >= 0 {
					x[a] = cols.Position[i] + av
					x[a] = -cols.Position[i] + av
				} else {
					x[a] = -cols.Position[i] - av
				}
			}
		}
	}

	for pass := 0; pass < 2000; pass++ {
		projectGross()
		clamp()
		vals := make([]float64, k)
		worst := 0.0
		for ri := range rows {
			r := &rows[ri]
			v := 0.0
			for t, a := range r.idx {
				v += r.w[t] * x[a]
			}
			vals[ri] = v
			d := math.Max(r.lb-v, v-r.ub)
			if d > worst {
				worst = d
			}
		}
		grossWorst := 0.0
		for gi := range grossRows {
			g := &grossRows[gi]
			mass := 0.0
			for t, a := range g.act {
				i := pre.Active[a]
				mass += g.w[t] * math.Abs(cols.Position[i]+x[a])
			}
			grossWorst = math.Max(grossWorst, mass-g.ub)
		}
		if worst <= 1e-7*math.Max(1, absMaxF(vals)) && grossWorst <= 1e-7 {
			break
		}
		// target = clip(row value into bounds)
		g := make([]float64, k)
		for ri := range rows {
			r := &rows[ri]
			g[ri] = math.Min(math.Max(vals[ri], r.lb), r.ub) - vals[ri]
		}
		lam := make([]float64, k)
		cholSolveInPlace(S, k, g, lam)
		const relax = 1.7
		for ri := range rows {
			if lam[ri] == 0 {
				continue
			}
			for t, a := range rows[ri].idx {
				x[a] += relax * lam[ri] * rows[ri].w[t]
			}
		}
		clamp()
	}
	return x
}

func absMaxF(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

// solveIPMEngine builds the canonical problem, repairs a warm start and runs
// the pure-Go IPM.
func solveIPMEngine(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights,
	softLambda float64, relaxMode bool, elasticPenalty float64, verbose bool) (OPT_Result, error) {

	if math.Abs(softLambda) > 1e-10 {
		return nil, fmt.Errorf("ipm engine does not support soft-lambda yet (%g)", softLambda)
	}
	cols := inputs.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, constraints, weights)
	p := buildADMMProblem(cols, det, relaxMode, elasticPenalty)
	pre, _ := p.Presolve.(*PresolveInfo)

	// warm start with feasibility repair (analytic base for objective quality)
	x0 := repairFeasibility(cols, det, pre, false)
	// pull the start strictly inside boxes and above the |.| kinks: the
	// canonical rows u>=|x|, g>=|pos+x| have zero slack exactly at the kink,
	// which blocks the first IPM Newton steps
	if pre != nil {
		for a, i := range pre.Active {
			xl := math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
			xu := math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
			// 2% toward box interior (or epsilon when the box is tight)
			x0[a] = xl + 0.98*(x0[a]-xl)
			x0[a] = xu - 0.98*(xu-x0[a])
			// prefer the milder of the two pulls
			v1 := xl + 0.98*(x0[a]-xl)
			v2 := xu - 0.98*(xu-x0[a])
			if math.Abs(v1-x0[a]) < math.Abs(v2-x0[a]) {
				x0[a] = v1
			} else {
				x0[a] = v2
			}
		}
	}
	z0 := make([]float64, p.N)
	copy(z0, x0)
	if pre != nil {
		for a := range pre.Active {
			z0[pre.TubBase+a] = 1.02*math.Abs(x0[a]) + 1e-2
		}
		// g block consistent with |pos+x|: reuse the builder's map by
		// scanning GROSS constraints
		gmap := map[int32]int32{}
		next := int32(2 * len(pre.Active))
		for _, c := range det {
			if constraintType(c.ConstraintItem.ConstraintType) == "GROSS" {
				for _, ti := range c.TradeVariableIndices {
					if !pre.Fixed[ti] {
						ni := pre.IdxOf[int(ti)]
						if _, ok := gmap[ni]; !ok {
							gmap[ni] = next
							next++
						}
					}
				}
			}
		}
		for ni, gi := range gmap {
			i := pre.Active[ni]
			z0[gi] = 1.02*math.Abs(cols.Position[i]+x0[ni]) + 1e-2
		}
	}

	st := admm.DefaultIPMSettings()
	st.Verbose = verbose
	st.X0 = z0
	t0 := time.Now()
	sol := admm.SolveIPM(p, &st)
	fmt.Printf("ipm engine: status=%s iter=%d priRes=%.3e duaRes=%.3e obj=%.6f solveTime=%v\n",
		sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes, sol.Obj, time.Since(t0))
	if sol.Status != "solved" && sol.Status != "best_effort" {
		return nil, fmt.Errorf("ipm solver failed: %s", sol.Status)
	}
	return resultFromReduced(cols, p, sol.X)
}
