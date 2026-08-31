package opt

import (
	"fmt"
	"math"

	"os"
	"testing"
	"time"

	"csvport/admm"
)

// benchOne runs the full admm backend on optim data with a settings tune and
// returns (wall, maxObj, err).
func benchOne(t *testing.T, tune func(*admm.Settings)) (time.Duration, float64, error) {
	t.Helper()
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	admmSettingsTune = tune
	defer func() { admmSettingsTune = nil }()
	t0 := time.Now()
	res, err := solveADMM(in, co, we, 0, false, 100.0, false)
	wall := time.Since(t0)
	obj := 0.0
	if err == nil {
		for i := range cols.Univ {
			x := res[i].Trade
			obj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*math.Abs(x)
		}
	}
	return wall, obj, err
}

// TestImproveBench is the improving-phase experiment harness.
func TestImproveBench(t *testing.T) {
	if os.Getenv("ADMM_BENCH") == "" {
		t.Skip("set ADMM_BENCH=1 to run")
	}
	type cfg struct {
		name string
		tune func(*admm.Settings)
	}
	ckpt := func(n int) func(*admm.Settings) {
		return func(s *admm.Settings) { s.CheckpointEvery = n }
	}
	cfgs := []cfg{
		{"basic(ckpt10k)", ckpt(10000)},
		{"ckpt2k", ckpt(2000)},
		{"ckpt5k", ckpt(5000)},
		{"alpha1.6+ckpt2k", func(s *admm.Settings) { s.CheckpointEvery = 2000; s.Alpha = 1.6 }},
		{"alpha1.6+ckpt1k", func(s *admm.Settings) { s.CheckpointEvery = 1000; s.Alpha = 1.6 }},
		{"chkEvery100+ckpt2k", func(s *admm.Settings) { s.CheckpointEvery = 2000; s.CheckEvery = 100 }},
		{"rhoAdaptOff+ckpt2k", func(s *admm.Settings) { s.CheckpointEvery = 2000; s.RhoAdaptive = false; s.Rho = 1e-4 }},
	}
	for _, c := range cfgs {
		wall, obj, err := benchOne(t, c.tune)
		es := "ok"
		if err != nil {
			es = err.Error()
		}
		fmt.Printf("%-22s wall=%-9v obj=%v err=%s\n", c.name, wall, obj, es)
	}
}
