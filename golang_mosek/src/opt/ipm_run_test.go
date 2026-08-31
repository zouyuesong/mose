package opt

import (
	"fmt"
	"math"
	"os"
	"testing"

	"csvport/admm"
)

func TestIPMSample(t *testing.T) {
	if os.Getenv("ADMM_BENCH") == "" {
		t.Skip("x")
	}
	in, co, we := loadSample(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	pre := p.Presolve.(*PresolveInfo)
	x0 := repairFeasibility(cols, det, pre, true)
	for a, i := range pre.Active {
		xl := math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
		xu := math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
		v1 := xl + 0.98*(x0[a]-xl)
		v2 := xu - 0.98*(xu-x0[a])
		if math.Abs(v1-x0[a]) < math.Abs(v2-x0[a]) {
			x0[a] = v1
		} else {
			x0[a] = v2
		}
	}
	z0 := make([]float64, p.N)
	copy(z0, x0)
	for a := range pre.Active {
		z0[pre.TubBase+a] = 1.02*math.Abs(x0[a]) + 1e-2
	}
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
	st := admm.DefaultIPMSettings()
	st.Verbose = true
	st.X0 = z0
	sol := admm.SolveIPM(p, &st)
	obj := 0.0
	for i := range cols.Univ {
		x := pre.XFix[i]
		if !pre.Fixed[i] {
			x = sol.X[pre.IdxOf[i]]
		}
		obj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*absF(x)
	}
	fmt.Printf("sample: status=%s it=%d rp=%.2e rd=%.2e obj=%.4f (want 48.7249) wall=%v\n",
		sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes, obj, sol.Duration)
}

func TestIPMOptim(t *testing.T) {
	if os.Getenv("ADMM_BENCH") == "" {
		t.Skip("x")
	}
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	pre := p.Presolve.(*PresolveInfo)
	x0 := repairFeasibility(cols, det, pre, true)
	for a, i := range pre.Active {
		xl := math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
		xu := math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
		v1 := xl + 0.98*(x0[a]-xl)
		v2 := xu - 0.98*(xu-x0[a])
		if math.Abs(v1-x0[a]) < math.Abs(v2-x0[a]) {
			x0[a] = v1
		} else {
			x0[a] = v2
		}
	}
	z0 := make([]float64, p.N)
	copy(z0, x0)
	for a := range pre.Active {
		z0[pre.TubBase+a] = 1.02*math.Abs(x0[a]) + 1e-2
	}
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
	for rep := 0; rep < 1; rep++ {
		st := admm.DefaultIPMSettings()
		st.Verbose = true
		st.X0 = z0
		sol := admm.SolveIPM(p, &st)
		obj := 0.0
		for i := range cols.Univ {
			x := pre.XFix[i]
			if !pre.Fixed[i] {
				x = sol.X[pre.IdxOf[i]]
			}
			obj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*absF(x)
		}
		fmt.Printf("optim: status=%s it=%d rp=%.2e rd=%.2e obj=%.4f (want 6580.23) wall=%v\n",
			sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes, obj, sol.Duration)
	}
}

func absF(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
