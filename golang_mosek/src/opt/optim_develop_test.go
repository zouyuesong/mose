package opt

import (
	"csvport/admm"
	"fmt"
	"math"
	"os"
	"testing"
)

func loadOptim(t *testing.T) (OPT_Inputs, OPT_Constrints, OPT_Weights) {
	t.Helper()
	base := "../../../optim_inputs"
	in, err := LoadOptInputsFromCSV(base + "/optim_inputs.20260520")
	if err != nil {
		t.Skip("optim data not present")
	}
	co, err := LoadOptConstraintsFromCSV(base + "/optim_constraints.20260520")
	if err != nil {
		t.Fatal(err)
	}
	we, err := LoadOptWeightsFromCSV(base + "/optim_weights.20260520")
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range in {
		if i.Position+i.MinTrade > i.MaxPosition {
			i.MaxPosition = i.Position + i.MinTrade + epsilon
		}
		if i.Position+i.MaxTrade < i.MinPosition {
			i.MinPosition = i.Position + i.MaxTrade - epsilon
		}
	}
	return in, co, we
}

// TestOptimRhoSweep sweeps fixed rho on the real optim problem.
func TestOptimRhoSweep(t *testing.T) {
	if os.Getenv("ADMM_SWEEP") == "" {
		t.Skip("set ADMM_SWEEP=1 to run")
	}
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	for _, rho := range []float64{1e-4} {
		st := admm.DefaultSettings()
		st.Rho = rho
		st.MaxIter = 200000
		st.Verbose = false
		st.Alpha = 1.0
		sol := admm.Solve(p, &st)
		maxObj := 0.0
		for i := range cols.Univ {
			x := sol.X[i]
			maxObj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*math.Abs(x)
		}
		fmt.Printf("rho=%-8g alpha=%.1f status=%-9s iter=%-7d pri=%.2e (tol %.1e) dua=%.2e obj=%.6f t=%v\n",
			rho, st.Alpha, sol.Status, sol.Iter, sol.PrimalRes, sol.EpsPrim, sol.DualRes, maxObj, sol.Duration)
		if rho == 1e-3 {
			f, _ := os.Create("/tmp/admm_optim_r1e3.csv")
			f.WriteString("sym,targetTrade,targetPosition\n")
			for i := range cols.Univ {
				fmt.Fprintf(f, "%s,%g,%g\n", cols.Univ[i], sol.X[i], cols.Position[i]+sol.X[i])
			}
			f.Close()
		}
	}
}
