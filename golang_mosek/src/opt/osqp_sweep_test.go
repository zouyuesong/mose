package opt

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	"csvport/admm"
	"csvport/cosqp"
)

// osqpCSC converts the canonical problem to OSQP CSC form.
func osqpCSC(p *admm.Problem) (n, m int, Px []float64, Pi, Pp []int64, q []float64, Ax []float64, Ai, Ap []int64, l, u []float64) {
	n = p.N
	m = len(p.Rows)
	colCount := make([]int64, n+1)
	for _, r := range p.Rows {
		for _, idx := range r.Idx {
			colCount[idx+1]++
		}
	}
	for j := 0; j < n; j++ {
		colCount[j+1] += colCount[j]
	}
	Ap = colCount
	type rv struct {
		r int64
		v float64
	}
	buckets := make([][]rv, n)
	for ri, r := range p.Rows {
		for t, idx := range r.Idx {
			buckets[idx] = append(buckets[idx], rv{int64(ri), r.Val[t]})
		}
	}
	for j := 0; j < n; j++ {
		sort.SliceStable(buckets[j], func(a, b int) bool { return buckets[j][a].r < buckets[j][b].r })
		for _, e := range buckets[j] {
			Ai = append(Ai, e.r)
			Ax = append(Ax, e.v)
		}
	}
	Pp = make([]int64, n+1)
	for j := 0; j < n; j++ {
		if p.PDiag[j] != 0 {
			Pi = append(Pi, int64(j))
			Px = append(Px, p.PDiag[j])
		}
		Pp[j+1] = int64(len(Pi))
	}
	q = append([]float64{}, p.Q...)
	l = make([]float64, m)
	u = make([]float64, m)
	for r := 0; r < m; r++ {
		l[r] = p.L[r]
		u[r] = p.U[r]
	}
	return
}

func maxObjOf(cols OPT_InputsColumns, trades []float64) float64 {
	obj := 0.0
	for i := range cols.Univ {
		x := trades[i]
		obj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*math.Abs(x)
	}
	return obj
}

// TestOSQPSweep: knob sweep on the optim problem (unscaled, vendor defaults
// as baseline).
func TestOSQPSweep(t *testing.T) {
	if os.Getenv("ADMM_BENCH") == "" {
		t.Skip("set ADMM_BENCH=1 to run")
	}
	in, co, we := loadOptim(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u := osqpCSC(p)

	type cfg struct {
		name       string
		epsAbs     float64
		rho, sigma float64
		adaptInt   int
	}
	cfgs := []cfg{
		{"default(1e-6/1e-8,rho.1)", 1e-6, 0.1, 1e-6, 0},
		{"eps1e-4", 1e-4, 0.1, 1e-6, 0},
		{"eps1e-3", 1e-3, 0.1, 1e-6, 0},
		{"eps1e-4+adapt25", 1e-4, 0.1, 1e-6, 25},
		{"eps1e-4+adapt50", 1e-4, 0.1, 1e-6, 50},
		{"eps1e-4+rho1e-2", 1e-4, 0.01, 1e-6, 0},
		{"eps1e-4+rho1", 1e-4, 1.0, 1e-6, 0},
		{"eps1e-4+sig1e-9", 1e-4, 0.1, 1e-9, 0},
		{"eps1e-4+sig1e-5", 1e-4, 0.1, 1e-5, 0},
		{"eps1e-5+adapt25", 1e-5, 0.1, 1e-6, 25},
	}
	for _, c := range cfgs {
		t0 := time.Now()
		x, _, st, rt, it, err := cosqp.SolveFull(n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u,
			c.epsAbs, c.epsAbs, 200000, 25, c.rho, c.sigma, c.adaptInt)
		wall := time.Since(t0)
		obj := math.NaN()
		if err == nil {
			pre := p.Presolve.(*PresolveInfo)
			full := make([]float64, len(cols.Univ))
			for i := range full {
				full[i] = pre.XFix[i]
			}
			for a, i := range pre.Active {
				full[i] = x[a]
			}
			obj = maxObjOf(cols, full)
		}
		fmt.Printf("%-24s wall=%-9v int=%7.2fms iters=%-6d st=%d obj=%v err=%v\n",
			c.name, wall, rt*1e3, it, st, fmtF2(obj), err)
	}
}

func fmtF2(v float64) string {
	if math.IsNaN(v) {
		return "n/a"
	}
	return fmt.Sprintf("%.2f", v)
}
