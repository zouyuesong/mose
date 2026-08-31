package opt

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"csvport/admm"
)

// TestWarmStartQuality: analytic start -> straight into polish. Measures
// rounds-to-success, obj, and wall time (decision gate for Phase B).
func TestWarmStartQuality(t *testing.T) {
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
	// how good is the start? obj and constraint violations
	obj0 := polishObjective(cols, pre, x0)
	sol0 := &admm.Solution{X: x0}
	_ = sol0

	// violation profile of the start point
	worstNet, worstGross := 0.0, 0.0
	for _, con := range det {
		switch constraintType(con.ConstraintItem.ConstraintType) {
		case "NET":
			val := 0.0
			for j, k := range con.TradeVariableIndices {
				var xv float64
				if pre.Fixed[k] {
					xv = pre.XFix[k]
				} else {
					xv = x0[pre.IdxOf[int(k)]]
				}
				val += con.Weights[j] * (cols.Position[k] + xv)
			}
			v := math.Max(con.ConstraintItem.LowerBound-val, val-con.ConstraintItem.UpperBound)
			if v > worstNet {
				worstNet = v
			}
		case "GROSS":
			val := 0.0
			for j, k := range con.TradeVariableIndices {
				var xv float64
				if pre.Fixed[k] {
					xv = pre.XFix[k]
				} else {
					xv = x0[pre.IdxOf[int(k)]]
				}
				val += con.Weights[j] * math.Abs(cols.Position[k]+xv)
			}
			if val > con.ConstraintItem.UpperBound && val-con.ConstraintItem.UpperBound > worstGross {
				worstGross = val - con.ConstraintItem.UpperBound
			}
		}
	}
	fmt.Printf("analytic start: obj(min)=%.2f worstNetViol=%.4g worstGrossViol=%.4g\n", obj0, worstNet, worstGross)

	// straight into polish from the analytic start
	for _, rounds := range []int{500, 2000} {
		snap := &admm.Solution{X: append([]float64(nil), x0...)}
		t0 := time.Now()
		px, ok := polishADMM(cols, det, p, snap, rounds)
		dt := time.Since(t0)
		objP := math.NaN()
		if ok {
			objP = polishObjective(cols, pre, px)
		}
		fmt.Printf("polish(rounds<=%d): ok=%v obj(min)=%v wall=%v\n", rounds, ok, fmtF(objP), dt)
	}
}

func fmtF(v float64) string {
	if math.IsNaN(v) {
		return "n/a"
	}
	return fmt.Sprintf("%.2f", v)
}
