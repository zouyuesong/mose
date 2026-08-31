// backend_admm.go: builds the same QP the original mosek solver built, in
// canonical minimize form
//
//	min 1/2 z'Pz + q'z   s.t.  l <= Az <= u
//
// with variable blocks x (trades), u >= |x| (TUB), g >= |pos+x| (GUB) and
// elastic t_k (relax mode), and hands it to the pure-Go ADMM solver in
// csvport/admm. Row layout honors the admm.Problem contract: all "+-1 pair"
// rows first (their A'A cross terms cancel), arbitrary rows (portfolio
// constraints, elastic rows, t bounds) after GeneralStart.
package opt

import (
	"csvport/admm"
	"fmt"
	"math"
	"strings"
)

// admmSettingsTune allows tests to tweak the ADMM settings used by the
// backend (nil in production).
var admmSettingsTune func(*admm.Settings)

func init() {
	registerSolver("admm", solveADMM)
}

func solveADMM(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights,
	softLambda float64, relaxMode bool, elasticPenalty float64, verbose bool) (OPT_Result, error) {

	if math.Abs(softLambda) > 1e-10 {
		return nil, fmt.Errorf("admm engine does not support soft-lambda yet (%g)", softLambda)
	}
	cols := inputs.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, constraints, weights)
	p := buildADMMProblem(cols, det, relaxMode, elasticPenalty)
	if err := admm.ValidateRowOrder(p); err != nil {
		return nil, fmt.Errorf("admm problem build: %v", err)
	}
	st := admm.DefaultSettings()
	if admmSettingsTune != nil {
		admmSettingsTune(&st)
	}
	st.Verbose = verbose
	var polished []float64
	if !relaxMode {
		// try an active-set polish every N iterations; stop at first success
		st.CheckpointEvery = 10000
		st.Checkpoint = func(snap *admm.Solution) bool {
			// pre-filter: iterates far from primal feasibility rarely polish
			// (and polishing them wastes the round budget)
			if snap.EpsPrim > 0 && snap.PrimalRes > 200*snap.EpsPrim {
				return false
			}
			if px, ok := polishADMM(cols, det, p, snap, 400); ok {
				polished = px
				return true
			}
			return false
		}
		if admmSettingsTune != nil {
			admmSettingsTune(&st) // let the harness override polish settings too
		}
	}
	sol := admm.Solve(p, &st)
	fmt.Printf("admm engine: status=%s iter=%d priRes=%.3e duaRes=%.3e obj=%.6f solveTime=%v\n",
		sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes, sol.Obj, sol.Duration)
	if polished != nil {
		fmt.Printf("admm engine: polished solution accepted\n")
		return resultFromReduced(cols, p, polished)
	}
	if sol.Status != "solved" && !relaxMode {
		// last-chance polish on the final iterate
		if px, ok := polishADMM(cols, det, p, sol, 3000); ok {
			fmt.Printf("admm engine: polished solution accepted\n")
			return resultFromReduced(cols, p, px)
		}
	}
	if sol.Status != "solved" && sol.Status != "checkpoint" {
		return nil, fmt.Errorf("admm solver failed: %s (iter=%d, priRes=%.3e duaRes=%.3e)",
			sol.Status, sol.Iter, sol.PrimalRes, sol.DualRes)
	}
	return resultFromReduced(cols, p, sol.X)
}

// resultFromReduced expands a reduced solution vector (or full-length
// fallback) into OPT_Result rows.
func resultFromReduced(cols OPT_InputsColumns, p *admm.Problem, xred []float64) (OPT_Result, error) {
	n := len(cols.Univ)
	res := make(OPT_Result, n)
	var pre *PresolveInfo
	if pp, ok := p.Presolve.(*PresolveInfo); ok {
		pre = pp
	}
	for i := 0; i < n; i++ {
		res[i] = OPT_ResultItem{
			Symbol:          cols.Univ[i],
			Trade:           0.0,
			CurrentPosition: cols.Position[i],
			TargetPosition:  cols.Position[i],
		}
	}
	if pre != nil {
		for a, i := range pre.Active {
			if a < len(xred) {
				res[i].Trade = xred[a]
				res[i].TargetPosition = cols.Position[i] + xred[a]
			}
		}
		for i := 0; i < n; i++ {
			if pre.Fixed[i] {
				res[i].Trade = pre.XFix[i]
				res[i].TargetPosition = cols.Position[i] + pre.XFix[i]
			}
		}
	} else {
		for i := 0; i < n && i < len(xred); i++ {
			res[i].Trade = xred[i]
			res[i].TargetPosition = cols.Position[i] + xred[i]
		}
	}
	return res, nil
}

// buildADMMProblem mirrors opt/backend_mosek.go (the original mosek model)
// exactly: same variables, rows, bounds and relax-mode semantics, but in
// minimize canonical form.
//
// Presolve: symbols whose trade box is numerically closed
// (xu - xl <= 1e-9 * max(1,|xl|)) are eliminated as constants; their rows and
// auxiliary variables are dropped and constraint bounds shifted accordingly
// (the same reduction MOSEK's presolve performs). Fixed trades re-enter the
// result unchanged. This matters on real data where no-trade names carry
// lambda=1 flag values: a 0.5-units absolute solver tolerance on such names
// otherwise buys ~1e5 of fake objective via the -lambda*pos*x term.
func buildADMMProblem(cols OPT_InputsColumns, det []OPT_AssembledConstraintItem,
	relaxMode bool, elasticPenalty float64) *admm.Problem {

	n := len(cols.Univ)
	inf := math.Inf(1)
	ninf := math.Inf(-1)

	// ---- fixed-trade detection and index compaction ----
	xl := make([]float64, n)
	xu := make([]float64, n)
	fixed := make([]bool, n)
	xFix := make([]float64, n)
	for i := 0; i < n; i++ {
		xl[i] = math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
		xu[i] = math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
		if xu[i]-xl[i] <= 1e-5*math.Max(1, math.Abs(xl[i])+math.Abs(xu[i])) {
			fixed[i] = true
			xFix[i] = math.Max(math.Min(0, xu[i]), xl[i])
		}
	}
	var active []int // active symbol -> original index
	var idxOf map[int]int32
	idxOf = map[int]int32{}
	for i := 0; i < n; i++ {
		if !fixed[i] {
			idxOf[i] = int32(len(active))
			active = append(active, i)
		}
	}
	na := len(active)

	// ---- variable layout: x [0,na), u [na,2na), g [...], t at the tail ----
	tubBase := int32(na)
	tradeToGUB := map[int32]int32{}
	for _, c := range det {
		if constraintType(c.ConstraintItem.ConstraintType) == "GROSS" {
			for _, ti := range c.TradeVariableIndices {
				ni := idxOf[int(ti)]
				if int(ti) < n && !fixed[ti] {
					if _, exists := tradeToGUB[ni]; !exists {
						tradeToGUB[ni] = -1
					}
				}
			}
		}
	}
	gubBase := int32(2 * na)
	next := gubBase
	for _, c := range det {
		if constraintType(c.ConstraintItem.ConstraintType) == "GROSS" {
			for _, ti := range c.TradeVariableIndices {
				if fixed[ti] {
					continue
				}
				ni := idxOf[int(ti)]
				if tradeToGUB[ni] < 0 {
					tradeToGUB[ni] = next
					next++
				}
			}
		}
	}
	numFixed := int(next) // x + u + g

	// objective: minimize 1/2 z'Pz + q'z with
	//   P_ii = lambda_i + tlambda_i, q_i = lambda_i*pos_i - alpha_i,
	//   q_{u_i} = cost_i, q_{t_k} = elasticPenalty
	pDiag := make([]float64, numFixed, numFixed+64)
	q := make([]float64, numFixed, numFixed+64)
	for a, i := range active {
		pDiag[a] = cols.Lambda[i] + cols.TradingLambda[i]
		q[a] = cols.Lambda[i]*cols.Position[i] - cols.Alpha[i]
		q[tubBase+int32(a)] = cols.TradingCost[i]
	}

	p := &admm.Problem{N: numFixed}
	p.PDiag, p.Q = pDiag, q

	addRow := func(idx []int32, val []float64, lb, ub float64) {
		p.Rows = append(p.Rows, admm.Row{Idx: idx, Val: val})
		p.L = append(p.L, lb)
		p.U = append(p.U, ub)
	}

	// ---- pair rows (A'A cross terms cancel) ----
	// x variable bounds as a pair: [xl, xu] and [-xu, -xl]
	for a, i := range active {
		aa := int32(a)
		addRow([]int32{aa}, []float64{1}, xl[i], xu[i])
		addRow([]int32{aa}, []float64{-1}, -xu[i], -xl[i])
	}
	// TUB: u + x >= 0, u - x >= 0  (u >= |x|; the u >= 0 bound is implied)
	for a := range active {
		aa, ui := int32(a), tubBase+int32(a)
		addRow([]int32{aa, ui}, []float64{1, 1}, 0, inf)
		addRow([]int32{aa, ui}, []float64{-1, 1}, 0, inf)
	}
	// GUB: g - x >= pos, g + x >= -pos  (g >= |pos + x|)
	for ni, gi := range tradeToGUB {
		i := active[ni]
		addRow([]int32{ni, gi}, []float64{-1, 1}, cols.Position[i], inf)
		addRow([]int32{ni, gi}, []float64{1, 1}, -cols.Position[i], inf)
	}
	p.GeneralStart = len(p.Rows)

	// activeRow shrinks a constraint row to non-fixed symbols and returns the
	// constant contribution of the fixed ones (sum w_k * xFix_k).
	activeRow := func(idxs []int32, ws []float64) ([]int32, []float64, float64) {
		var ri []int32
		var rv []float64
		constSum := 0.0
		for k, ti := range idxs {
			if fixed[ti] {
				constSum += ws[k] * xFix[ti]
			} else {
				ri = append(ri, idxOf[int(ti)])
				rv = append(rv, ws[k])
			}
		}
		return ri, rv, constSum
	}

	// ---- general rows ----
	relaxationAuxVarIdx := next
	for _, con := range det {
		switch constraintType(con.ConstraintItem.ConstraintType) {
		case "NET": // lb <= sum w_k (pos_k + x_k) <= ub
			constAdj := 0.0
			for j, k := range con.TradeVariableIndices {
				constAdj += con.Weights[j] * cols.Position[k]
			}
			ri, rv, fixSum := activeRow(con.TradeVariableIndices, con.Weights)
			lb := con.ConstraintItem.LowerBound - constAdj - fixSum
			ub := con.ConstraintItem.UpperBound - constAdj - fixSum
			breach := con.ConstraintItem.LowerBound-constAdj > 0 || con.ConstraintItem.UpperBound-constAdj < 0
			if relaxMode && breach {
				if lb > 0 {
					lb = -epsilon
				}
				if ub < 0 {
					ub = epsilon
				}
			}
			addRow(ri, rv, lb, ub)
			if relaxMode && breach {
				if con.ConstraintItem.LowerBound > constAdj {
					// t >= LB - finalConstraint, t >= 0 -> finalConstraint + t >= LB
					ti := relaxationAuxVarIdx
					p.PDiag = append(p.PDiag, 0)
					p.Q = append(p.Q, elasticPenalty)
					addRow([]int32{ti}, []float64{1}, 0, inf) // t >= 0 bound row (general: single row)
					addRow(append(append([]int32{}, ri...), ti),
						append(append([]float64{}, rv...), 1.0),
						con.ConstraintItem.LowerBound-constAdj-fixSum, inf)
					relaxationAuxVarIdx++
				}
				if con.ConstraintItem.UpperBound < constAdj {
					// finalConstraint - t <= UB
					ti := relaxationAuxVarIdx
					p.PDiag = append(p.PDiag, 0)
					p.Q = append(p.Q, elasticPenalty)
					addRow([]int32{ti}, []float64{1}, 0, inf)
					addRow(append(append([]int32{}, ri...), ti),
						append(append([]float64{}, rv...), -1.0),
						ninf, con.ConstraintItem.UpperBound-constAdj-fixSum)
					relaxationAuxVarIdx++
				}
			}
		case "NET_TRADE": // lb <= sum w_k x_k <= ub
			ri, rv, fixSum := activeRow(con.TradeVariableIndices, con.Weights)
			addRow(ri, rv, con.ConstraintItem.LowerBound-fixSum, con.ConstraintItem.UpperBound-fixSum)
		case "GROSS": // sum w_k |pos_k + x_k| <= ub (via g), lb ignored
			var posAbsIdx []int32
			var grossWs []float64
			zeroTradedGross := 0.0
			fixGross := 0.0
			for k, ti := range con.TradeVariableIndices {
				zeroTradedGross += math.Abs(cols.Position[ti]) * con.Weights[k]
				if fixed[ti] {
					fixGross += con.Weights[k] * math.Abs(cols.Position[ti]+xFix[ti])
				} else {
					ni := idxOf[int(ti)]
					posAbsIdx = append(posAbsIdx, tradeToGUB[ni])
					grossWs = append(grossWs, con.Weights[k])
				}
			}
			if con.ConstraintItem.LowerBound > 0 {
				fmt.Println(fmt.Sprint("gross constraint", con.ConstraintItem.ConstraintName, " lower bound(", con.ConstraintItem.LowerBound, ") > 0 - not honored"))
			}
			ub := con.ConstraintItem.UpperBound - fixGross
			breach := con.ConstraintItem.UpperBound < zeroTradedGross
			if relaxMode && breach {
				ub = zeroTradedGross + epsilon - fixGross
			}
			addRow(posAbsIdx, grossWs, 0, ub)
			if relaxMode && breach {
				// sum w g - t <= UB (original)
				ti := relaxationAuxVarIdx
				p.PDiag = append(p.PDiag, 0)
				p.Q = append(p.Q, elasticPenalty)
				addRow([]int32{ti}, []float64{1}, 0, inf)
				addRow(append(append([]int32{}, posAbsIdx...), ti),
					append(append([]float64{}, grossWs...), -1.0),
					ninf, con.ConstraintItem.UpperBound-fixGross)
				relaxationAuxVarIdx++
			}
		case "GROSS_TRADE": // sum w_k |x_k| <= ub (via u), lb ignored
			var tradeAbsIdx []int32
			var tradeWs []float64
			fixTrade := 0.0
			for k, ti := range con.TradeVariableIndices {
				if fixed[ti] {
					fixTrade += con.Weights[k] * math.Abs(xFix[ti])
				} else {
					ni := idxOf[int(ti)]
					tradeAbsIdx = append(tradeAbsIdx, tubBase+ni)
					tradeWs = append(tradeWs, con.Weights[k])
				}
			}
			if con.ConstraintItem.LowerBound > 0 {
				fmt.Println(fmt.Sprint("gross traded constraint", con.ConstraintItem.ConstraintName, " lower bound(", con.ConstraintItem.LowerBound, ") > 0 - not honored"))
			}
			addRow(tradeAbsIdx, tradeWs, 0, con.ConstraintItem.UpperBound-fixTrade)
		default:
			panic("unsupported constraint type: " + con.ConstraintItem.ConstraintType + "(" + con.ConstraintItem.ConstraintName + ")")
		}
	}
	p.N = len(p.PDiag)
	p.Presolve = &PresolveInfo{
		Active:  active,
		Fixed:   fixed,
		XFix:    xFix,
		IdxOf:   idxOf,
		TubBase: int(tubBase),
	}
	return p
}

// PresolveInfo records the fixed-variable elimination performed by
// buildADMMProblem so the caller can expand the reduced solution.
type PresolveInfo struct {
	Active  []int         // reduced symbol slot -> original universe index
	Fixed   []bool        // original index -> fixed?
	XFix    []float64     // trade value assigned to fixed symbols
	IdxOf   map[int]int32 // original index -> reduced x index
	TubBase int           // reduced index where TUB block starts
}

// polishADMM performs an OSQP-style polish on the ADMM iterate, hardened
// into a primal active-set refinement:
//
//  0. start from a box-feasible point (ADMM iterate clamped into its boxes);
//  1. classify symbols (at lower/upper bound, at the |x| kink=0, or free with
//     a sign) and rows (active at a bound or currently violated) from the
//     current point;
//  2. solve the equality-constrained QP on that active set exactly via a
//     Schur complement (P = diag(lambda+tlambda) is strictly PD);
//  3. line-search the step to stay feasible on boxes and non-active rows;
//  4. iterate until the active set is stable (max 50 rounds);
//  5. verify the final point on EVERY original row; accept only if feasible
//     and objective <= the ADMM iterate's (minimize form).
//
// eqrowT is one equality row of the polish active-set system.
type eqrowT struct {
	w       []float64 // over free symbols
	b       float64
	full    []float64 // over reduced symbols (for line search)
	atUpper bool      // active at the upper bound (for dual signs)
	sr      float64   // row scale used in the Schur system
	rowIdx  int       // index into the round's rows/isGrossFoldRow
}

func polishADMM(cols OPT_InputsColumns, det []OPT_AssembledConstraintItem,
	p *admm.Problem, sol *admm.Solution, maxRounds int) ([]float64, bool) {

	pre, ok := p.Presolve.(*PresolveInfo)
	if !ok {
		return nil, false
	}
	na := len(pre.Active)
	if len(sol.X) < na {
		return nil, false
	}
	xl := make([]float64, na)
	xu := make([]float64, na)
	xcur := make([]float64, na)
	for a, i := range pre.Active {
		xl[a] = math.Max(cols.MinTrade[i], cols.MinPos[i]-cols.Position[i])
		xu[a] = math.Min(cols.MaxTrade[i], cols.MaxPos[i]-cols.Position[i])
		xcur[a] = math.Max(xl[a], math.Min(xu[a], sol.X[a]))
	}

	// kink positions where a |.| fold changes sign: GROSS rows fold |pos+x|
	// (kink at x = -pos), GROSS_TRADE rows fold |x| (kink at x = 0).
	kink := make([]float64, na)
	hasKink := make([]bool, na)
	for _, con := range det {
		switch constraintType(con.ConstraintItem.ConstraintType) {
		case "GROSS":
			for _, k := range con.TradeVariableIndices {
				if !pre.Fixed[k] {
					a := pre.IdxOf[int(k)]
					kink[a] = -cols.Position[k]
					hasKink[a] = true
				}
			}
		case "GROSS_TRADE":
			for _, k := range con.TradeVariableIndices {
				if !pre.Fixed[k] {
					a := pre.IdxOf[int(k)]
					kink[a] = 0
					hasKink[a] = true
				}
			}
		}
	}

	// linear rows over reduced x: value(x) = sum w_a x_a + c0 in [lb, ub]
	type linrow struct {
		w           []float64
		lb, ub      float64
		isGrossFold bool // GROSS or GROSS_TRADE sign-folded row
	}
	var rows []linrow
	var isGrossFoldRow []bool

	var bestX []float64
	var bestObj float64
	var bestCertified bool
	consider := func(x []float64) {
		if verifyPolish(cols, det, pre, x) {
			o := polishObjective(cols, pre, x)
			if bestX == nil || o < bestObj {
				bestX = append([]float64{}, x...)
				bestObj = o
				bestCertified = false
			}
		}
	}
	consider(xcur)

	// certifyKKT: dual-sign check under a round's classification and scaled
	// multipliers. Convention: g_j = P_j x_j + q_j + cost_j*sgn_j + sum_r w_rj mu_r
	// with mu = lam/sr. Rows active at LOWER need mu <= 0, at UPPER mu >= 0.
	// Vars at box-lower need g >= -tol, box-upper g <= +tol; at the cost kink
	// (x=0) need |g - cost*sgn| <= cost + tol. Variables pinned at a GROSS
	// fold kink are skipped (nonsmooth, documented approximation).
	certifyKKT := func(x []float64, kind []int, sgn []float64, eqs []eqrowT, lam []float64) bool {
		na_ := len(x)
		mu := make([]float64, len(eqs))
		for i := range eqs {
			mu[i] = lam[i] / eqs[i].sr
			tolm := 1e-6 * math.Max(1, math.Abs(mu[i]))
			if eqs[i].atUpper && mu[i] < -tolm {
				if polishDebug {
					fmt.Printf("   cert FAIL rowMu i=%d atUpper mu=%.4g\n", i, mu[i])
				}
				return false
			}
			if !eqs[i].atUpper && mu[i] > tolm {
				if polishDebug {
					fmt.Printf("   cert FAIL rowMu i=%d atLower mu=%.4g\n", i, mu[i])
				}
				return false
			}
		}
		// lin = P x + q + A'mu (smooth part; the |x| cost enters per-branch)
		lin := make([]float64, na_)
		for a := 0; a < na_; a++ {
			i := pre.Active[a]
			lin[a] = (cols.Lambda[i]+cols.TradingLambda[i])*x[a] + (cols.Lambda[i]*cols.Position[i] - cols.Alpha[i])
		}
		// grossFold[a]: sum of w*mu over GROSS-folded equality rows touching a
		// (subgradient slack available at the |pos+x| kink)
		grossFold := make([]float64, na_)
		hasFold := make([]bool, na_)
		for i := range eqs {
			if mu[i] == 0 {
				continue
			}
			for a := 0; a < na_; a++ {
				if eqs[i].full[a] != 0 {
					lin[a] += eqs[i].full[a] * mu[i]
				}
			}
		}
		for a := 0; a < na_; a++ {
			if kind[a] == 3 { // atGKink: accumulate fold subgradient terms
				for i := range eqs {
					if eqs[i].full[a] != 0 && isGrossFoldRow[eqs[i].rowIdx] {
						grossFold[a] += eqs[i].full[a] * mu[i]
						hasFold[a] = true
					}
				}
				// remove the folded term from lin to get the smooth remainder
				lin[a] -= grossFold[a]
			}
		}
		nfail := 0
		for a := 0; a < na_; a++ {
			i := pre.Active[a]
			c := cols.TradingCost[i]
			tol := 1e-6 * math.Max(1, math.Abs(lin[a]))
			bad := false
			switch kind[a] {
			case 0: // atLo: only rightward movement feasible
				bad = lin[a]+c < -tol
			case 1: // atHi: only leftward movement feasible
				bad = lin[a]-c > tol
			case 2: // atKink (x=0, interior): |lin| <= cost
				bad = math.Abs(lin[a]) > c+tol
			case 3: // atGKink (x=-pos): |lin + cost*sgn(x)| <= |grossFold|
				sx := 1.0
				if x[a] < 0 {
					sx = -1
				}
				bad = math.Abs(lin[a]+c*sx) > math.Abs(grossFold[a])+tol
			case 4: // free: stationarity with the active branch
				bad = math.Abs(lin[a]+c*sgn[a]) > tol
			}
			if bad {
				if polishDebug && nfail < 5 {
					fmt.Printf("   cert FAIL var %s kind=%d lin=%.6g cost=%.6g fold=%.6g x=%.6g\n", cols.Univ[i], kind[a], lin[a], c, grossFold[a], x[a])
				}
				nfail++
			}
		}
		return nfail == 0
	}

	stalled := 0
	var lastKind []int
	var lastSgn []float64
	var lastEqs []eqrowT
	var lastLam []float64
	var lastLin []float64
	lastMu := map[int]float64{} // rowIdx -> last unscaled multiplier (for row-drop)
	for round := 0; round < maxRounds; round++ {
		// rebuild GROSS/GROSS_TRADE sign folds from the current point
		rows = rows[:0]
		isGrossFoldRow = isGrossFoldRow[:0]
		for _, con := range det {
			switch constraintType(con.ConstraintItem.ConstraintType) {
			case "NET":
				constAdj := 0.0
				for j, k := range con.TradeVariableIndices {
					constAdj += con.Weights[j] * cols.Position[k]
				}
				r := linrow{w: make([]float64, na), lb: con.ConstraintItem.LowerBound - constAdj, ub: con.ConstraintItem.UpperBound - constAdj}
				for j, k := range con.TradeVariableIndices {
					if !pre.Fixed[k] {
						r.w[pre.IdxOf[int(k)]] += con.Weights[j]
					}
				}
				rows = append(rows, r)
				isGrossFoldRow = append(isGrossFoldRow, false)
			case "NET_TRADE":
				r := linrow{w: make([]float64, na), lb: con.ConstraintItem.LowerBound, ub: con.ConstraintItem.UpperBound}
				for j, k := range con.TradeVariableIndices {
					if !pre.Fixed[k] {
						r.w[pre.IdxOf[int(k)]] += con.Weights[j]
					}
				}
				rows = append(rows, r)
				isGrossFoldRow = append(isGrossFoldRow, false)
			case "GROSS":
				ub := con.ConstraintItem.UpperBound
				for j, k := range con.TradeVariableIndices {
					if pre.Fixed[k] {
						ub -= con.Weights[j] * math.Abs(cols.Position[k]+pre.XFix[k])
					}
				}
				r := linrow{w: make([]float64, na), lb: math.Inf(-1), ub: ub, isGrossFold: true}
				for j, k := range con.TradeVariableIndices {
					if !pre.Fixed[k] {
						a := pre.IdxOf[int(k)]
						v := cols.Position[k] + xcur[a]
						sg := 1.0
						if v < 0 {
							sg = -1.0
						}
						r.w[a] += con.Weights[j] * sg
						r.ub -= con.Weights[j] * sg * cols.Position[k]
						r.lb = math.Inf(-1)
					}
				}
				rows = append(rows, r)
				isGrossFoldRow = append(isGrossFoldRow, true)
			case "GROSS_TRADE":
				ub := con.ConstraintItem.UpperBound
				for j, k := range con.TradeVariableIndices {
					if pre.Fixed[k] {
						ub -= con.Weights[j] * math.Abs(pre.XFix[k])
					}
				}
				r := linrow{w: make([]float64, na), lb: math.Inf(-1), ub: ub, isGrossFold: true}
				for j, k := range con.TradeVariableIndices {
					if !pre.Fixed[k] {
						a := pre.IdxOf[int(k)]
						sg := 1.0
						if xcur[a] < 0 {
							sg = -1.0
						}
						r.w[a] += con.Weights[j] * sg
					}
				}
				rows = append(rows, r)
				isGrossFoldRow = append(isGrossFoldRow, true)
			}
		}

		// classify symbols from xcur
		const atLo, atHi, atKink, atGKink, freeV = 0, 1, 2, 3, 4
		kind := make([]int, na)
		sgn := make([]float64, na)
		freeIdx := []int{}
		fixVal := make([]float64, na)
		for a := 0; a < na; a++ {
			i := pre.Active[a]
			cost := cols.TradingCost[i]
			relTol := 0.0
			kind[a] = -1
			// dual-sign release: a bound var whose gradient pushes it off the
			// bound (last round's lin) becomes free again (primal active-set
			// "drop" step; the line search provides the "add" step)
			if len(lastLin) == na {
				relTol = 1e-6 * math.Max(1, math.Abs(lastLin[a]))
				lin := lastLin[a]
				if xcur[a]-xl[a] <= 1e-6*math.Max(1, math.Abs(xl[a])) && lin+cost >= -relTol {
					kind[a] = atLo
					fixVal[a] = xl[a]
				}
				if kind[a] < 0 && xu[a]-xcur[a] <= 1e-6*math.Max(1, math.Abs(xu[a])) && lin-cost <= relTol {
					kind[a] = atHi
					fixVal[a] = xu[a]
				}
				if kind[a] < 0 && math.Abs(xcur[a]) <= math.Max(1.0, 1e-5*math.Abs(cols.Position[i])) && math.Abs(lin) <= cost+relTol {
					kind[a] = atKink
					fixVal[a] = 0
				}
			} else {
				// bootstrap round (no lastLin yet): use the raw gradient
				// P x + q (no multipliers) to decide releases
				lin0 := (cols.Lambda[i]+cols.TradingLambda[i])*xcur[a] + (cols.Lambda[i]*cols.Position[i] - cols.Alpha[i])
				if xcur[a]-xl[a] <= 1e-6*math.Max(1, math.Abs(xl[a])) && lin0+cost >= -1e-6 {
					kind[a] = atLo
					fixVal[a] = xl[a]
				} else if xu[a]-xcur[a] <= 1e-6*math.Max(1, math.Abs(xu[a])) && lin0-cost <= 1e-6 {
					kind[a] = atHi
					fixVal[a] = xu[a]
				} else if math.Abs(xcur[a]) <= math.Max(1.0, 1e-5*math.Abs(cols.Position[i])) && math.Abs(lin0) <= cost+1e-6 {
					kind[a] = atKink
					fixVal[a] = 0
				}
			}
			if kind[a] < 0 && hasKink[a] && math.Abs(cols.Position[i]+xcur[a]) <= 1e-2*math.Max(1, math.Abs(cols.Position[i])) {
				// near the |pos+x| fold: pin AT the kink (zero gross
				// contribution) to stop sign-flip zig-zag across it
				if kink[a] >= xl[a] && kink[a] <= xu[a] {
					kind[a] = atGKink
					fixVal[a] = kink[a]
				}
			}
			if kind[a] < 0 {
				kind[a] = freeV
			}
			if kind[a] == freeV {
				freeIdx = append(freeIdx, a)
				if xcur[a] >= 0 {
					sgn[a] = 1
				} else {
					sgn[a] = -1
				}
			}
		}
		if len(freeIdx) == 0 {
			break
		}
		nf := len(freeIdx)

		// active rows: at a bound (within tol) or violated
		var eqs []eqrowT
		activeSig := map[string]bool{}
		for ri := range rows {
			r := &rows[ri]
			val := 0.0
			for a := 0; a < na; a++ {
				if r.w[a] != 0 {
					val += r.w[a] * xcur[a]
				}
			}
			tol := 1e-5 * math.Max(1, math.Max(math.Abs(r.lb), math.Abs(r.ub)))
			loAct := !math.IsInf(r.lb, -1) && (val-r.lb) <= tol
			upAct := !math.IsInf(r.ub, 1) && (r.ub-val) <= tol
			// row-drop: skip the one row whose last multiplier had the worst
			// wrong sign (classic primal active-set dual release for rows)
			if mu, seen := lastMu[ri]; seen {
				tolm := 1e-6 * math.Max(1, math.Abs(mu))
				if (upAct && !loAct && mu < -tolm) || (loAct && !upAct && mu > tolm) {
					delete(lastMu, ri)
					continue
				}
			}
			var bnd float64
			isA := false
			if loAct && !upAct {
				bnd, isA = r.lb, true
			} else if upAct {
				bnd, isA = r.ub, true
			}
			if !isA {
				continue
			}
			// skip duplicate active rows (same weight vector, e.g. delta_ptf
			// vs delta_grp3 on single-group data): the twin is implied, and
			// alternating add/drop on identical rows cycles forever
			sig := rowSignature(r.w)
			if seen := activeSig[sig]; seen {
				continue
			}
			activeSig[sig] = true
			er := eqrowT{w: make([]float64, nf), full: r.w, atUpper: upAct && !loAct, rowIdx: ri}
			constSum := 0.0
			for a := 0; a < na; a++ {
				constSum += r.w[a] * fixVal[a]
			}
			er.b = bnd - constSum
			for f, a := range freeIdx {
				er.w[f] = r.w[a]
			}
			eqs = append(eqs, er)
		}

		// Schur solve: P x = -c - Weq' lam ; Weq x = b
		P := make([]float64, nf)
		c := make([]float64, nf)
		for f, a := range freeIdx {
			i := pre.Active[a]
			P[f] = cols.Lambda[i] + cols.TradingLambda[i]
			c[f] = cols.Lambda[i]*cols.Position[i] - cols.Alpha[i] + cols.TradingCost[i]*sgn[a]
		}
		k := len(eqs)
		Pinv := make([]float64, nf)
		for f := 0; f < nf; f++ {
			Pinv[f] = 1.0 / P[f]
		}
		Wn := make([][]float64, k)
		bn := make([]float64, k)
		for i := 0; i < k; i++ {
			sr := math.Max(1, math.Abs(eqs[i].b))
			for f := 0; f < nf; f++ {
				sr = math.Max(sr, math.Abs(eqs[i].w[f])*math.Max(1, math.Abs(c[f]*Pinv[f])))
			}
			Wn[i] = make([]float64, nf)
			for f := 0; f < nf; f++ {
				Wn[i][f] = eqs[i].w[f] / sr
			}
			bn[i] = eqs[i].b / sr
			eqs[i].sr = sr
		}
		S := make([]float64, k*k)
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				sv := 0.0
				for f := 0; f < nf; f++ {
					sv += Wn[i][f] * Wn[j][f] * Pinv[f]
				}
				S[i*k+j] = sv
			}
		}
		smax := 0.0
		for _, v := range S {
			smax = math.Max(smax, v)
		}
		for i := 0; i < k; i++ {
			S[i*k+i] += 1e-10 * math.Max(1e-12, smax)
		}
		rhsL := make([]float64, k)
		for i := 0; i < k; i++ {
			sv := 0.0
			for f := 0; f < nf; f++ {
				sv += Wn[i][f] * Pinv[f] * c[f]
			}
			rhsL[i] = -bn[i] - sv
		}
		lam := make([]float64, k)
		if k > 0 {
			if !cholInPlace(S, k) {
				break
			}
			cholSolveInPlace(S, k, rhsL, lam)
		}
		// gradient at xcur under this classification/multipliers (input to
		// the next round's dual-sign release decisions)
		linNow := make([]float64, na)
		for a := 0; a < na; a++ {
			i := pre.Active[a]
			linNow[a] = (cols.Lambda[i]+cols.TradingLambda[i])*xcur[a] + (cols.Lambda[i]*cols.Position[i] - cols.Alpha[i])
		}
		for i := 0; i < k; i++ {
			mui := lam[i] / eqs[i].sr
			if mui == 0 {
				continue
			}
			full := eqs[i].full
			for a := 0; a < na; a++ {
				if full[a] != 0 {
					linNow[a] += full[a] * mui
				}
			}
		}
		lastKind = append(lastKind[:0], kind...)
		lastSgn = append(lastSgn[:0], sgn...)
		lastEqs = append(lastEqs[:0], eqs...)
		lastLam = append(lastLam[:0], lam...)
		lastLin = append(lastLin[:0], linNow...)
		for i := range eqs {
			lastMu[eqs[i].rowIdx] = lam[i] / eqs[i].sr
		}
		xtar := make([]float64, na)
		copy(xtar, fixVal)
		for f, a := range freeIdx {
			v := -c[f]
			for i := 0; i < k; i++ {
				v -= Wn[i][f] * lam[i]
			}
			xtar[a] = v * Pinv[f]
		}

		// line search xcur -> xtar staying feasible on boxes, folded-row kinks
		// and rows. Kink positions: GROSS rows fold |pos+x| (kink at x=-pos),
		// GROSS_TRADE rows fold |x| (kink at 0). Crossing a kink invalidates
		// the linearization and makes the iterate oscillate, so stop there.
		d := make([]float64, na)
		tMax := 1.0
		blockTag := ""
		for a := 0; a < na; a++ {
			d[a] = xtar[a] - xcur[a]
			if d[a] > 1e-15 && !math.IsInf(xu[a], 1) {
				nt := (xu[a] - xcur[a]) / d[a]
				if nt < tMax {
					tMax = nt
					blockTag = fmt.Sprintf("boxHi %d(%s)", a, cols.Univ[pre.Active[a]])
				}
			} else if d[a] < -1e-15 && !math.IsInf(xl[a], -1) {
				nt := (xl[a] - xcur[a]) / d[a]
				if nt < tMax {
					tMax = nt
					blockTag = fmt.Sprintf("boxLo %d(%s)", a, cols.Univ[pre.Active[a]])
				}
			}
			if hasKink[a] && math.Abs(d[a]) > 1e-15 {
				nt := (kink[a] - xcur[a]) / d[a]
				if nt > 1e-12 && nt < tMax {
					tMax = nt
					blockTag = fmt.Sprintf("kink %d(%s)", a, cols.Univ[pre.Active[a]])
				}
			}
		}
		for ri := range rows {
			r := &rows[ri]
			dv := 0.0
			val := 0.0
			for a := 0; a < na; a++ {
				if r.w[a] != 0 {
					dv += r.w[a] * d[a]
					val += r.w[a] * xcur[a]
				}
			}
			// only currently-FEASIBLE rows may block the step at their bounds;
			// violated rows are active equalities whose step direction always
			// improves them, so they never need to block here.
			// relative-scale zero test: row values live at ~1e5-1e6, where fp
			// noise in dv is ~1e-10..1e-6; treat those as no movement
			dvZero := 1e-9 * math.Max(1, math.Abs(val))
			upOK := !math.IsInf(r.ub, 1) && val <= r.ub+1e-9*math.Max(1, math.Abs(r.ub))
			loOK := !math.IsInf(r.lb, -1) && val >= r.lb-1e-9*math.Max(1, math.Abs(r.lb))
			if dv > dvZero && upOK && val+dv > r.ub {
				nt := (r.ub - val) / dv
				if nt < tMax {
					tMax = nt
					blockTag = fmt.Sprintf("rowUb %d val=%.6g ub=%.6g", ri, val, r.ub)
				}
			} else if dv < -dvZero && loOK && val+dv < r.lb {
				nt := (r.lb - val) / dv
				if nt < tMax {
					tMax = nt
					blockTag = fmt.Sprintf("rowLb %d val=%.6g lb=%.6g", ri, val, r.lb)
				}
			}
		}
		if polishDebug {
			fmt.Printf("polish round %d: tMax=%.3e nFree=%d k=%d blockedBy=%s\n", round, tMax, nf, k, blockTag)
		}
		if tMax < 0 {
			tMax = 0 // rounding put xcur a hair outside a bound; snap back below
		}
		if tMax <= 1e-12 {
			stalled++
			if stalled > 15 {
				break // genuinely stuck
			}
			// snap variables sitting on their bounds (within fp noise) so the
			// next classification frees the line search
			for a := 0; a < na; a++ {
				if math.Abs(xcur[a]-xl[a]) <= 1e-9*math.Max(1, math.Abs(xl[a])) {
					xcur[a] = xl[a]
				} else if math.Abs(xcur[a]-xu[a]) <= 1e-9*math.Max(1, math.Abs(xu[a])) {
					xcur[a] = xu[a]
				}
			}
			consider(xcur)
			continue
		}
		stalled = 0
		for a := 0; a < na; a++ {
			xcur[a] += tMax * d[a]
			// clamp into the box: at this scale (x~1e5, d~1e5) one fused step
			// carries ~1e-6 of fp rounding, which otherwise wedges the line
			// search on a negative ratio forever
			if xcur[a] < xl[a] {
				xcur[a] = xl[a]
			} else if xcur[a] > xu[a] {
				xcur[a] = xu[a]
			}
		}
		consider(xcur)
		if tMax == 1.0 {
			// full step: xcur == xtar, classification/lam are self-consistent.
			// Certify KKT dual signs to reject active sets that are feasible
			// stationary points of the WRONG active set (observed: polish from
			// under-converged iterates lands on suboptimal KKT points).
			if cert := certifyKKT(xcur, kind, sgn, eqs, lam); cert {
				if verifyPolish(cols, det, pre, xcur) {
					o := polishObjective(cols, pre, xcur)
					if !bestCertified || o < bestObj {
						bestX = append([]float64{}, xcur...)
						bestObj = o
						bestCertified = true
					}
					break
				}
			}
			if bestCertified {
				break
			}
		}
	}
	if !bestCertified && bestX != nil && lastLam != nil && certifyKKT(bestX, lastKind, lastSgn, lastEqs, lastLam) {
		bestCertified = true
	}
	if bestX == nil || !bestCertified {
		if polishDebug && bestX != nil {
			fmt.Println("admm polish: rejected (no certified KKT point)")
		}
		return nil, false
	}
	objAdmm := polishObjective(cols, pre, sol.X[:na])
	if bestObj > objAdmm+1e-6*math.Max(1, math.Abs(objAdmm)) {
		fmt.Printf("admm polish: rejected (worse objective %.6f > %.6f)\n", bestObj, objAdmm)
		return nil, false
	}
	fmt.Printf("admm polish: accepted+certified (obj %.6f vs admm %.6f)\n", bestObj, objAdmm)
	return bestX, true
}

var polishDebug = false

func verifyPolish(cols OPT_InputsColumns, det []OPT_AssembledConstraintItem,
	pre *PresolveInfo, xpol []float64) bool {
	n := len(cols.Univ)
	// trade and position boxes (original input bounds)
	for i := 0; i < n; i++ {
		var x float64
		if pre.Fixed[i] {
			x = pre.XFix[i]
		} else {
			x = xpol[pre.IdxOf[i]]
		}
		tp := cols.Position[i] + x
		tolX := 1e-6 * math.Max(1, math.Max(math.Abs(x), math.Abs(cols.MinTrade[i])))
		if x < cols.MinTrade[i]-tolX || x > cols.MaxTrade[i]+tolX {
			if polishDebug {
				fmt.Printf("polish verify FAIL tradebox %s x=%.6g [%.6g,%.6g]\n", cols.Univ[i], x, cols.MinTrade[i], cols.MaxTrade[i])
			}
			return false
		}
		if x < cols.MinPos[i]-cols.Position[i]-2e-6*math.Max(1, math.Abs(cols.MinPos[i])) ||
			x > cols.MaxPos[i]-cols.Position[i]+2e-6*math.Max(1, math.Abs(cols.MaxPos[i])) {
			if polishDebug {
				fmt.Printf("polish verify FAIL posbox %s x=%.6g xbounds=[%.6g,%.6g]\n", cols.Univ[i], x, cols.MinPos[i]-cols.Position[i], cols.MaxPos[i]-cols.Position[i])
			}
			return false
		}
		_ = tp
	}
	// constraints
	for _, con := range det {
		lo, hi := math.Inf(-1), math.Inf(1)
		switch constraintType(con.ConstraintItem.ConstraintType) {
		case "NET":
			cadj := 0.0
			val := 0.0
			for j, k := range con.TradeVariableIndices {
				cadj += con.Weights[j] * cols.Position[k]
				var x float64
				if pre.Fixed[k] {
					x = pre.XFix[k]
				} else {
					x = xpol[pre.IdxOf[int(k)]]
				}
				val += con.Weights[j] * x
			}
			lo, hi = con.ConstraintItem.LowerBound-cadj, con.ConstraintItem.UpperBound-cadj
			tol := 1e-6 * math.Max(1, math.Max(math.Abs(lo), math.Abs(hi)))
			if val < lo-tol || val > hi+tol {
				if polishDebug {
					fmt.Printf("polish verify FAIL NET %s val=%.6g [%.6g,%.6g]\n", con.ConstraintItem.ConstraintName, val, lo, hi)
				}
				return false
			}
		case "NET_TRADE":
			val := 0.0
			for j, k := range con.TradeVariableIndices {
				var x float64
				if pre.Fixed[k] {
					x = pre.XFix[k]
				} else {
					x = xpol[pre.IdxOf[int(k)]]
				}
				val += con.Weights[j] * x
			}
			lo, hi = con.ConstraintItem.LowerBound, con.ConstraintItem.UpperBound
			tol := 1e-6 * math.Max(1, math.Max(math.Abs(lo), math.Abs(hi)))
			if val < lo-tol || val > hi+tol {
				if polishDebug {
					fmt.Printf("polish verify FAIL NET_TRADE %s val=%.6g [%.6g,%.6g]\n", con.ConstraintItem.ConstraintName, val, lo, hi)
				}
				return false
			}
		case "GROSS":
			val := 0.0
			for j, k := range con.TradeVariableIndices {
				var x float64
				if pre.Fixed[k] {
					x = pre.XFix[k]
				} else {
					x = xpol[pre.IdxOf[int(k)]]
				}
				val += con.Weights[j] * math.Abs(cols.Position[k]+x)
			}
			tol := 1e-6 * math.Max(1, math.Abs(con.ConstraintItem.UpperBound))
			if val > con.ConstraintItem.UpperBound+tol {
				if polishDebug {
					fmt.Printf("polish verify FAIL GROSS %s val=%.6g ub=%.6g\n", con.ConstraintItem.ConstraintName, val, con.ConstraintItem.UpperBound)
				}
				return false
			}
		case "GROSS_TRADE":
			val := 0.0
			for j, k := range con.TradeVariableIndices {
				var x float64
				if pre.Fixed[k] {
					x = pre.XFix[k]
				} else {
					x = xpol[pre.IdxOf[int(k)]]
				}
				val += con.Weights[j] * math.Abs(x)
			}
			tol := 1e-6 * math.Max(1, math.Abs(con.ConstraintItem.UpperBound))
			if val > con.ConstraintItem.UpperBound+tol {
				if polishDebug {
					fmt.Printf("polish verify FAIL GROSS_TRADE %s val=%.6g ub=%.6g\n", con.ConstraintItem.ConstraintName, val, con.ConstraintItem.UpperBound)
				}
				return false
			}
		}
	}
	return true
}

// polishObjective evaluates the canonical minimize objective at xpol
// (fixed names included).
func polishObjective(cols OPT_InputsColumns, pre *PresolveInfo, xpol []float64) float64 {
	obj := 0.0
	for i := 0; i < len(cols.Univ); i++ {
		var x float64
		if pre.Fixed[i] {
			x = pre.XFix[i]
		} else {
			x = xpol[pre.IdxOf[i]]
		}
		obj += 0.5*(cols.Lambda[i]+cols.TradingLambda[i])*x*x + (cols.Lambda[i]*cols.Position[i]-cols.Alpha[i])*x + cols.TradingCost[i]*math.Abs(x)
	}
	return obj
}

// cholInPlace factors a symmetric PD matrix (lower triangle) in place.
func cholInPlace(a []float64, k int) bool {
	for i := 0; i < k; i++ {
		for j := 0; j <= i; j++ {
			s := a[i*k+j]
			for l := 0; l < j; l++ {
				s -= a[i*k+l] * a[j*k+l]
			}
			if i == j {
				if s <= 0 {
					return false
				}
				a[i*k+i] = math.Sqrt(s)
			} else {
				a[i*k+j] = s / a[j*k+j]
			}
		}
	}
	return true
}

// cholSolveInPlace solves (L L') x = b given cholInPlace output.
func cholSolveInPlace(ch []float64, k int, b, x []float64) {
	for i := 0; i < k; i++ {
		s := b[i]
		for l := 0; l < i; l++ {
			s -= ch[i*k+l] * x[l]
		}
		x[i] = s / ch[i*k+i]
	}
	for i := k - 1; i >= 0; i-- {
		s := x[i]
		for l := i + 1; l < k; l++ {
			s -= ch[l*k+i] * x[l]
		}
		x[i] = s / ch[i*k+i]
	}
}

// rowSignature builds a canonical key for a weight vector (duplicate-row
// detection in the active set, e.g. delta_ptf vs delta_grp3).
func rowSignature(w []float64) string {
	var b strings.Builder
	b.Grow(len(w))
	for _, v := range w {
		if v != 0 {
			b.WriteByte(byte(int8(v * 1000)))
		} else {
			b.WriteByte(0)
		}
	}
	return b.String()
}
