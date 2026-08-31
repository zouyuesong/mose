package opt

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"csvport/admm"
)

// TestHybridWarmStart: analytic start -> short ADMM repair -> polish.
// Sweeps the ADMM repair budget to find the minimum that lets polish
// succeed with a certified optimum.
func TestHybridWarmStart(t *testing.T) {
	if os.Getenv("ADMM_WARM") == "" {
		t.Skip("set ADMM_WARM=1 to run")
	}
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	pre := p.Presolve.(*PresolveInfo)
	na := len(pre.Active)

	x0full := analyticStart(cols)
	x0 := make([]float64, na)
	for a, i := range pre.Active {
		x0[a] = x0full[i]
	}
	// full-length start vector: x block, then u=|x|, then g=|pos+x| (aux
	// blocks consistent with the TUB/GUB linearizations)
	xfull := make([]float64, p.N)
	copy(xfull, x0)
	nSym := len(cols.Univ)
	for a := range pre.Active {
		xfull[pre.TubBase+a] = math.Abs(x0[a])
	}
	for ti, gi := range preGUBMap(p, det) {
		xfull[gi] = math.Abs(cols.Position[ti] + x0[pre.IdxOf[int(ti)]])
	}
	_ = nSym
	// z0 = projection of A x0 onto [l,u] (warm the DUAL state, not just x)
	z0 := make([]float64, len(p.Rows))
	for r, row := range p.Rows {
		v := 0.0
		for t, idx := range row.Idx {
			v += row.Val[t] * xfull[idx]
		}
		if v < p.L[r] {
			v = p.L[r]
		} else if v > p.U[r] {
			v = p.U[r]
		}
		z0[r] = v
	}

	// feasible cold start x=0 (book unbreached) -> direct active set
	{
		zero := make([]float64, p.N)
		snap := &admm.Solution{X: zero}
		tP := time.Now()
		px, ok := polishADMM(cols, det, p, snap, 3000)
		dP := time.Since(tP)
		objP := math.NaN()
		if ok {
			objP = polishObjective(cols, pre, px)
		}
		fmt.Printf("x=0 start -> polish(3000): %v obj(min)=%s wall=%v\n", ok, fmtF(objP), dP)
	}
	for _, admmIters := range []int{200, 400, 800, 1600, 3200} {
		st := admm.DefaultSettings()
		st.X0 = append([]float64(nil), xfull...)
		st.Z0 = append([]float64(nil), z0...)
		st.MaxIter = admmIters
		st.Checkpoint = nil
		st.CheckpointEvery = 0
		tA := time.Now()
		sol := admm.Solve(p, &st)
		dA := time.Since(tA)
		tP := time.Now()
		px, ok := polishADMM(cols, det, p, sol, 500)
		_ = px
		dP := time.Since(tP)
		objP := math.NaN()
		if ok {
			objP = polishObjective(cols, pre, px)
		}
		fmt.Printf("admm=%4d it (%6.1fms, pri/eps=%7.2f) -> polish %6.2fms: ok=%-5v obj=%s  total=%6.1fms\n",
			admmIters, float64(dA.Microseconds())/1e3, sol.PrimalRes/sol.EpsPrim,
			float64(dP.Microseconds())/1e3, ok, fmtF(objP), float64(dA.Microseconds()+dP.Microseconds())/1e3)
	}
}

// preGUBMap rebuilds the tradeIdx -> gVarIdx map used at build time.
func preGUBMap(p *admm.Problem, det []OPT_AssembledConstraintItem) map[int32]int32 {
	pre := p.Presolve.(*PresolveInfo)
	nSym := len(pre.Fixed)
	gubBase := 2 * len(pre.Active)
	tradeToGUB := map[int32]int32{}
	next := int32(gubBase)
	for _, c := range det {
		if constraintType(c.ConstraintItem.ConstraintType) == "GROSS" {
			for _, ti := range c.TradeVariableIndices {
				if int(ti) < nSym && !pre.Fixed[ti] {
					ni := pre.IdxOf[int(ti)]
					if _, exists := tradeToGUB[ni]; !exists {
						tradeToGUB[ni] = next
						next++
					}
				}
			}
		}
	}
	// map keyed by original trade index for the caller
	out := map[int32]int32{}
	for ni, gi := range tradeToGUB {
		out[int32(pre.Active[ni])] = gi
	}
	return out
}
