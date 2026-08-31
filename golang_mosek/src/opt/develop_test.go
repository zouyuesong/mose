package opt

import (
	"csvport/admm"
	"fmt"
	"math"
	"os"
	"testing"
)

func loadSample(t *testing.T) (OPT_Inputs, OPT_Constrints, OPT_Weights) {
	t.Helper()
	base := "../sample"
	in, err := LoadOptInputsFromCSV(base + "/input.csv")
	if err != nil {
		t.Skip("sample data not present")
	}
	co, err := LoadOptConstraintsFromCSV(base + "/constraint.csv")
	if err != nil {
		t.Fatal(err)
	}
	we, err := LoadOptWeightsFromCSV(base + "/weight.csv")
	if err != nil {
		t.Fatal(err)
	}
	// apply STEP1 pre-relax like Solve does
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

// TestRhoSweep is a development harness: fixed-rho basic ADMM convergence
// on the sample problem for several rho/alpha values.
func TestRhoSweep(t *testing.T) {
	if os.Getenv("ADMM_SWEEP") == "" {
		t.Skip("set ADMM_SWEEP=1 to run")
	}
	in, co, we := loadSample(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	for _, rho := range []float64{1e-6, 1e-5, 1e-4} {
		for _, alpha := range []float64{1.0, 1.6} {
			for _, scaling := range []bool{false} {
				st := admm.DefaultSettings()
				st.Rho = rho
				st.Alpha = alpha
				st.Scaling = scaling
				st.EpsRel = 1e-8
				st.MaxIter = 500000
				sol := admm.Solve(p, &st)
				// objective in maximize form for comparison with mosek
				maxObj := 0.0
				for i := range cols.Univ {
					x := sol.X[i]
					maxObj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*math.Abs(x)
				}
				fmt.Printf("rho=%-6g alpha=%.1f scal=%-5v status=%-9s iter=%-7d pri=%.2e dua=%.2e obj=%.6f t=%v\n",
					rho, alpha, scaling, sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes, maxObj, sol.Duration)
			}
		}
	}
}

// TestWarmFromMosek starts ADMM from the mosek reference solution (fixed-point check).
func TestWarmFromMosek(t *testing.T) {
	if os.Getenv("ADMM_SWEEP") == "" {
		t.Skip("set ADMM_SWEEP=1 to run")
	}
	in, co, we := loadSample(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	// read mosek reference trades
	ref, err := os.ReadFile("../sample/output.csv")
	if err != nil {
		t.Fatal(err)
	}
	lines := splitLines(string(ref))
	refTrade := map[string]float64{}
	for i, ln := range lines {
		if i == 0 || ln == "" {
			continue
		}
		f := splitComma(ln)
		refTrade[f[0]] = parseFloat(f[1])
	}
	x0 := make([]float64, p.N)
	for i, s := range cols.Univ {
		x0[i] = refTrade[s]
	}
	st := admm.DefaultSettings()
	st.X0 = x0
	st.MaxIter = 50000
	sol := admm.Solve(p, &st)
	maxObj := 0.0
	for i := range cols.Univ {
		x := sol.X[i]
		maxObj += (cols.Alpha[i]-cols.Lambda[i]*cols.Position[i])*x - 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x - cols.TradingCost[i]*math.Abs(x)
	}
	fmt.Printf("warm-from-mosek: status=%s iter=%d pri=%.2e dua=%.2e maxObj=%.6f t=%v\n",
		sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes, maxObj, sol.Duration)
	// how far did x drift from the mosek trades?
	maxd := 0.0
	for i, s := range cols.Univ {
		d := math.Abs(sol.X[i] - refTrade[s])
		if d > maxd {
			maxd = d
		}
	}
	fmt.Printf("max |x - x_mosek| = %.6g\n", maxd)
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func splitComma(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	out = append(out, s[start:])
	return out
}

func parseFloat(s string) float64 {
	var v float64
	fmt.Sscanf(s, "%g", &v)
	return v
}

// TestWoodburyOnSample validates the low-rank KKT solve against dense algebra
// on the real sample problem structure.
func TestWoodburyOnSample(t *testing.T) {
	if os.Getenv("ADMM_SWEEP") == "" {
		t.Skip("set ADMM_SWEEP=1 to run")
	}
	in, co, we := loadSample(t)
	cols := in.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, co, we)
	p := buildADMMProblem(cols, det, false, 100.0)
	st := admm.DefaultSettings()
	sol := admm.Solve(p, &st) // get a Solver-internal factorization via package API? no - use exported test hook
	_ = sol

	// dense M via exported row data
	n := p.N
	m := len(p.Rows)
	M := make([][]float64, n)
	for i := range M {
		M[i] = make([]float64, n)
		M[i][i] = p.PDiag[i] + st.Sigma
	}
	for _, row := range p.Rows {
		for a := range row.Idx {
			for b := range row.Idx {
				M[row.Idx[a]][row.Idx[b]] += st.Rho * row.Val[a] * row.Val[b]
			}
		}
	}
	r := make([]float64, n)
	for i := range r {
		r[i] = math.Sin(float64(i) * 1.7)
	}
	zd := denseSolveOpt(M, r)
	z, ok := admm.DebugSolveKKT(p, &st, r)
	if !ok {
		t.Fatal("kkt solve failed")
	}
	worst := 0.0
	for i := 0; i < n; i++ {
		d := math.Abs(z[i] - zd[i])
		if d > worst {
			worst = d
		}
	}
	t.Logf("woodbury vs dense: n=%d m=%d k=%d maxdiff=%.3e", n, m, m-p.GeneralStart, worst)
	if worst > 1e-6 {
		t.Fatalf("woodbury mismatch %e", worst)
	}
}

func denseSolveOpt(M [][]float64, b []float64) []float64 {
	n := len(b)
	A := make([][]float64, n)
	for i := range A {
		A[i] = append([]float64{}, M[i]...)
		A[i] = append(A[i], b[i])
	}
	for c := 0; c < n; c++ {
		piv := c
		for rr := c + 1; rr < n; rr++ {
			if math.Abs(A[rr][c]) > math.Abs(A[piv][c]) {
				piv = rr
			}
		}
		A[c], A[piv] = A[piv], A[c]
		for rr := 0; rr < n; rr++ {
			if rr == c {
				continue
			}
			f := A[rr][c] / A[c][c]
			for k := c; k <= n; k++ {
				A[rr][k] -= f * A[c][k]
			}
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = A[i][n] / A[i][i]
	}
	return x
}

// kktDiag prints which variables carry the stuck dual residual.
func kktDiag(p *admm.Problem, sol *admm.Solution, cols OPT_InputsColumns) {
	n := p.N
	mu := make([]float64, n)
	for j := 0; j < n; j++ {
		mu[j] = p.PDiag[j]*sol.X[j] + p.Q[j]
	}
	for r, row := range p.Rows {
		if sol.Y[r] == 0 {
			continue
		}
		for t, idx := range row.Idx {
			mu[idx] += row.Val[t] * sol.Y[r]
		}
	}
	type kv struct {
		j  int
		mu float64
	}
	arr := make([]kv, n)
	for j := range mu {
		arr[j] = kv{j, math.Abs(mu[j])}
	}
	for i := 0; i < 5; i++ {
		for j := i + 1; j < n; j++ {
			if arr[j].mu > arr[i].mu {
				arr[i], arr[j] = arr[j], arr[i]
			}
		}
	}
	nSym := len(cols.Univ)
	for i := 0; i < 5 && i < n; i++ {
		j := arr[i].j
		name := fmt.Sprintf("var%d", j)
		if j < nSym {
			name = "x:" + cols.Univ[j]
		} else if j < 2*nSym {
			name = "u:" + cols.Univ[j-nSym]
		} else {
			name = "g/t:" + fmt.Sprint(j)
		}
		fmt.Printf("  duaRes top: %-45s |mu|=%.4e  x=%.6g\n", name, arr[i].mu, sol.X[j])
	}
}
