package opt

import (
	"os"
	"testing"

	"csvport/admm"
)

func TestProfileADMM(t *testing.T) {
	if os.Getenv("ADMM_PROF") == "" {
		t.Skip("x")
	}
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	st := admm.DefaultSettings()
	st.MaxIter = 30000
	st.Checkpoint = nil
	st.CheckpointEvery = 0
	sol := admm.Solve(p, &st)
	t.Logf("iters=%d dur=%v", sol.Iter, sol.Duration)
}
