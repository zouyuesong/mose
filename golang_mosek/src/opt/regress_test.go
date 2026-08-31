package opt

import (
	"fmt"
	"os"
	"testing"
	"time"

	"csvport/admm"
)

func TestColdRegress(t *testing.T) {
	if os.Getenv("ADMM_WARM") == "" {
		t.Skip("x")
	}
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	for _, it := range []int{6000, 10000, 20000} {
		st := admm.DefaultSettings()
		st.MaxIter = it
		st.Checkpoint = nil
		st.CheckpointEvery = 0
		t0 := time.Now()
		sol := admm.Solve(p, &st)
		dA := time.Since(t0)
		t0 = time.Now()
		px, ok := polishADMM(cols, det, p, sol, 500)
		dP := time.Since(t0)
		obj := "n/a"
		if ok {
			obj = fmt.Sprintf("%.2f", polishObjective(cols, p.Presolve.(*PresolveInfo), px))
		}
		fmt.Printf("cold it=%5d (%6.1fms pri/eps=%6.2f) polish %6.1fms ok=%v obj=%s\n",
			it, float64(dA.Microseconds())/1e3, sol.PrimalRes/sol.EpsPrim, float64(dP.Microseconds())/1e3, ok, obj)
	}
}
