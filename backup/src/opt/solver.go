package opt

import (
	"errors"
	"fmt"
	"github.com/mosek/mosek.go"
	"math"
	"strings"
)

const (
	epsilon float64 = 1e-10
)

func Solve(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights, softLambda float64, elasticPenalty float64, verbose bool) (OPT_Result, error) {
	// STEP 1. relax max position constraints if current position with maxTrade and minTrade will be in breach
	for _, i := range inputs {
		if i.Position+i.MinTrade > i.MaxPosition {
			i.MaxPosition = i.Position + i.MinTrade + epsilon
		}
		if i.Position+i.MaxTrade < i.MinPosition {
			i.MinPosition = i.Position + i.MaxTrade - epsilon
		}
	}
	if res, constraintInBreachWithZeroTraded, err := solve(inputs, constraints, weights, softLambda, false, elasticPenalty, verbose); err == nil || !constraintInBreachWithZeroTraded {
		return res, err
	} else {
		fmt.Println("normal opt attempt failed, constraint in breach with 0 traded, trying to solve under relaxation mode")
		for _, i := range inputs {
			if i.Position > i.MaxPosition {
				i.MaxPosition = i.Position + epsilon
			}
			if i.Position < i.MinPosition {
				i.MinPosition = i.Position - epsilon
			}
		}
		res, _, err = solve(inputs, constraints, weights, softLambda, true, elasticPenalty, verbose)
		return res, err
	}
}

func solve(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights, softLambda float64, relaxMode bool, elasticPenalty float64, verbose bool) (OPT_Result, bool, error) {
	// preparation - we could move these two steps out, but need to ensure the indices in the univ matches indices in constraints then
	inputColumns := inputs.ToColumns()
	constraintDetails := AssembleConstraintDetails(inputColumns.Univ, constraints, weights)
	numvar := int32(len(inputColumns.Univ))
	quadCoefs := map[int32]map[int32]float64{}
	linearCoefs := map[int32]float64{}

	// ALL variables: x, |x| upper bound (in full since |x| is in the objective function), |x| lower bounds when needed, |pos+x| upper bound for gross constraints, |pos+x| lower bounds when needed
	// collect the univ index of all gross(UB) varialbes ( abs(position + x) ), prepare two maps: tradeIndex <-> posAbsIndex

	// TradeAbs UB auxiliary variables
	tubBaseIdx := numvar
	tradeToTUB := map[int32]int32{}
	tubToTrade := map[int32]int32{}
	tubIdx := tubBaseIdx
	for ti := int32(0); ti < numvar; ti++ {
		tradeToTUB[ti] = tubIdx
		tubToTrade[tubIdx] = ti
		tubIdx += 1
	}

	// GROSS UB auxiliary variables
	gubBaseIdx := tubIdx
	tradeToGUB := map[int32]int32{}
	gubToTrade := map[int32]int32{}
	gubIdx := gubBaseIdx
	for _, c := range constraintDetails {
		if strings.ToUpper(c.ConstraintItem.ConstraintType) == "GROSS" {
			for _, ti := range c.TradeVariableIndices {
				// if ti already has a gi, we can skip it
				if _, exists := tradeToGUB[ti]; !exists {
					tradeToGUB[ti] = gubIdx
					gubToTrade[gubIdx] = ti
					gubIdx += 1
				}
			}
		}
	}

	// start the setup
	constraintInBreachWithZeroTraded := false
	/* Create the mosek environment. */
	env, r := mosek.MakeEnv()
	if r != 0 {
		symname, rstr, _ := mosek.GetCodeDesc(r)
		errmsg := "mosek error code at mosek.MakeEnv: " + fmt.Sprint(r) + " mosek.GetCodeDesc returns:" + symname + ", " + rstr
		fmt.Sprintln(errmsg)
		return nil, constraintInBreachWithZeroTraded, errors.New(errmsg)
	}
	defer env.DeleteEnv()
	task, r := env.MakeTask()
	defer task.DeleteTask()
	if r != 0 {
		symname, rstr, _ := mosek.GetCodeDesc(r)
		errmsg := "mosek error code at mosek.MakeTask: " + fmt.Sprint(r) + " mosek.GetCodeDesc returns:" + symname + ", " + rstr
		fmt.Sprintln(errmsg)
		return nil, constraintInBreachWithZeroTraded, errors.New(errmsg)
	}

	// Append traded variables. The variables will initially be fixed at zero (x=0).
	task.AppendVars(numvar)
	for i := int32(0); i < numvar; i++ {
		// Set the bounds on variable i. minTrade[i] <= x_i <= maxTrade[i]
		// minPos[i] <= x_i + pos <= maxPos[i] ->  minPos[i] - pos <= x_i <= maxPos[i] - pos
		// note: use math.Inf(-1) and math.Inf(1) for infinity
		task.PutVarBound(i, mosek.BK_RA,
			math.Max(inputColumns.MinTrade[i], inputColumns.MinPos[i]-inputColumns.Position[i]),
			math.Min(inputColumns.MaxTrade[i], inputColumns.MaxPos[i]-inputColumns.Position[i]))
	}
	constraintIdx := int32(0)

	// set up TUB
	task.AppendVars(int32(len(tubToTrade)))
	for uidx, i := range tubToTrade {
		// u + x >= 0, u - x >= 0, so u >= |x|
		// task.PutVarBound(uidx, mosek.BK_FR, math.Inf(-1), math.Inf(1))
		task.PutVarBound(uidx, mosek.BK_LO, 0.0, math.Inf(1))
		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{i, uidx}, []float64{1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, 0.0, math.Inf(1))
		constraintIdx += 1

		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{i, uidx}, []float64{-1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, 0.0, math.Inf(1))
		constraintIdx += 1

		linearCoefs[uidx] = -inputColumns.TradingCost[i] // minimize  tradingCost_i * |x|_i
	}

	// set up GUB
	task.AppendVars(int32(len(gubToTrade)))
	for gidx, tradeIdx := range gubToTrade {
		task.PutVarBound(gidx, mosek.BK_LO, 0.0, math.Inf(1))
		// g >= |x + pos| -> g > x + pos and g > -x - pos  ---> g-x > pos   and g+x > -pos
		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{tradeIdx, gidx}, []float64{-1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, inputColumns.Position[tradeIdx], math.Inf(1))
		constraintIdx += 1

		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{tradeIdx, gidx}, []float64{1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, -inputColumns.Position[tradeIdx], math.Inf(1))
		constraintIdx += 1
	}

	// set up the objective function   -------------------------
	// 1. maximize return alpha * (pos + x), since alpha * pos is constant, we only need alpha * x here
	// 2. minimize cost: sum(|cost * x|_i) (done above in |x| aux variable)  ----------------------
	// 3. minimize vol: 1/2 * lambda * (pos + x)^2 = 1/2 * lambda x^2 + lambda * pos * x -------------------------------
	// 4. minimize trades 1/2 * tlambda * x^2 ------------------------------------
	// put them together, we have (lambda + tlambda) * x^2 + lambda * pos * x
	for i := int32(0); i < numvar; i++ {
		//task.PutQObjIJ(i, i, -(inputColumns.Lambda[i] + inputColumns.TradingLambda[i]))
		quadCoefs[i] = map[int32]float64{i: -(inputColumns.Lambda[i] + inputColumns.TradingLambda[i])}
		// here we combine the linear term minimize (lambda * pos * x) together with maximize alpha linear term  x * alpha
		linearCoefs[i] = inputColumns.Alpha[i] - inputColumns.Lambda[i]*inputColumns.Position[i]
		//task.PutCJ(i, inputColumns.Alpha[i]-inputColumns.Lambda[i]*inputColumns.Position[i]) // CJ is the linear term c_j in the objective
	}
	// set up the constraints ----------------------------------------------
	relaxationAuxVarIdx := gubIdx
	for _, con := range constraintDetails {
		switch strings.ToUpper(con.ConstraintItem.ConstraintType) {
		case "NET":
			task.AppendCons(1)
			task.PutARow(constraintIdx, con.TradeVariableIndices, con.Weights)
			// sum(w * (x + pos)) -> constant term is  sum(w*pos), so lb - sum(w*pos) < sum(w*x) < ub - sum(w*pos)
			constAdj := 0.0
			for j, k := range con.TradeVariableIndices {
				constAdj += con.Weights[j] * inputColumns.Position[k]
			}
			lb := con.ConstraintItem.LowerBound - constAdj
			ub := con.ConstraintItem.UpperBound - constAdj
			breach := lb > 0 || ub < 0
			constraintInBreachWithZeroTraded = constraintInBreachWithZeroTraded || breach
			if relaxMode && breach {
				// 0 traded, so x = 0,  lb < sum(w*pos) < ub
				if lb > 0 { // lowerBound > sum(w*pos), meaning 0 traded will break lower bound, relax lower bound
					lb = -epsilon // set lowerBound to sum(w*pos), effectively setting sum(w*x)'s lower bound to 0
				}
				if ub < 0 { // upperBound < sum(w*pos), meaning 0 traded will break upperBound, relax upper bound
					ub = epsilon
				}
			}
			task.PutConBound(constraintIdx, mosek.BK_RA, lb, ub)
			constraintIdx += 1
			// add elastic constraint to pull the bounds back to the pre-set bounds
			if relaxMode && breach {
				if con.ConstraintItem.LowerBound > constAdj { // lowerBound > sum(w*pos), meaning 0 traded will break lower bound, relax lower bound
					// Adding elastic soft constraint
					//// appending aux variable t s.t. t >= LB - finalConstraint && t >= 0
					task.AppendVars(1)
					task.PutVarBound(relaxationAuxVarIdx, mosek.BK_LO, 0.0, math.Inf(1)) // t >= 0
					task.AppendCons(1)
					elasticIndices := append(con.TradeVariableIndices, relaxationAuxVarIdx)
					elasticWeights := append(con.Weights, 1.0)
					// finalConstraint + t >= LB
					task.PutARow(constraintIdx, elasticIndices, elasticWeights)
					task.PutConBound(constraintIdx, mosek.BK_LO, con.ConstraintItem.LowerBound-constAdj, math.Inf(1))
					constraintIdx += 1
					// add linear penalty term on t -> minimize (max(LB - finalConstraint, 0) * elasticLambda -> minimize elasticLambda * t -> maximize -elasticLambda * t
					linearCoefs[relaxationAuxVarIdx] = -elasticPenalty
					relaxationAuxVarIdx += 1
				}
				if con.ConstraintItem.UpperBound < constAdj { // upperBound < sum(w*pos), meaning 0 traded will break upperBound, relax upper bound
					// Adding elastic soft constraint
					//// appending aux variable t s.t. t >= finalConstraint - UB && t >= 0
					task.AppendVars(1)
					task.PutVarBound(relaxationAuxVarIdx, mosek.BK_LO, 0.0, math.Inf(1)) // t >= 0
					task.AppendCons(1)
					elasticIndices := append(con.TradeVariableIndices, relaxationAuxVarIdx)
					elasticWeights := append(con.Weights, -1.0)
					// finalConstraint - t <= UB
					task.PutARow(constraintIdx, elasticIndices, elasticWeights)
					task.PutConBound(constraintIdx, mosek.BK_UP, math.Inf(-1), con.ConstraintItem.UpperBound-constAdj)
					constraintIdx += 1
					// add linear penalty term on t -> minimize (max(finalConstraint - UB, 0) * elasticLambda -> minimize elasticLambda * t -> maximize -elasticLambda * t
					linearCoefs[relaxationAuxVarIdx] = -elasticPenalty
					relaxationAuxVarIdx += 1
				}
			}
			if math.Abs(softLambda) > 1e-10 {
				for wi, i := range con.TradeVariableIndices {
					for wj, j := range con.TradeVariableIndices {
						if i >= j {
							quadCoefs[i][j] -= con.Weights[wi] * con.Weights[wj] * softLambda
						}
						linearCoefs[j] -= con.Weights[wi] * con.Weights[wj] * inputColumns.Position[i] * softLambda
						linearCoefs[i] -= con.Weights[wi] * con.Weights[wj] * inputColumns.Position[j] * softLambda
					}
				}
			}
		case "NET_TRADE":
			task.AppendCons(1)
			task.PutARow(constraintIdx, con.TradeVariableIndices, con.Weights)
			task.PutConBound(constraintIdx, mosek.BK_RA, con.ConstraintItem.LowerBound, con.ConstraintItem.UpperBound)
			constraintIdx += 1
			if math.Abs(softLambda) > 1e-10 {
				for wi, i := range con.TradeVariableIndices {
					for wj, j := range con.TradeVariableIndices {
						if i >= j {
							quadCoefs[i][j] -= con.Weights[wi] * con.Weights[wj] * softLambda
						}
					}
				}
			}
		case "GROSS":
			task.AppendCons(1)
			posAbsVariableIndices := make([]int32, len(con.TradeVariableIndices))
			//grossBaseIdx := numvar + numvar
			zeroTradedGross := 0.0
			for k := 0; k < len(con.TradeVariableIndices); k++ {
				posAbsVariableIndices[k] = tradeToGUB[con.TradeVariableIndices[k]]
				zeroTradedGross += math.Abs(inputColumns.Position[con.TradeVariableIndices[k]]) * con.Weights[k]
			}
			task.PutARow(constraintIdx, posAbsVariableIndices, con.Weights)
			if con.ConstraintItem.LowerBound > 0 {
				fmt.Println(fmt.Sprint("gross constraint", con.ConstraintItem.ConstraintName, " lower bound(", con.ConstraintItem.LowerBound, ") > 0 - not honored"))
			}
			ub := con.ConstraintItem.UpperBound
			breach := ub < zeroTradedGross
			constraintInBreachWithZeroTraded = constraintInBreachWithZeroTraded || breach
			if relaxMode && breach {
				ub = zeroTradedGross + epsilon
			}
			task.PutConBound(constraintIdx, mosek.BK_RA, 0, ub)
			constraintIdx += 1

			// add elastic constraint to pull the bounds back to the pre-set bounds
			if relaxMode && breach {
				// UppderBound < zeroTradeGross
				// Adding elastic soft constraint
				// appending aux variable t s.t. t >= finalConstraint - UB && t >= 0, so t = max(finalConstraint - UB, 0), then minimize t
				task.AppendVars(1)
				task.PutVarBound(relaxationAuxVarIdx, mosek.BK_LO, 0.0, math.Inf(1)) // t >= 0
				task.AppendCons(1)
				elasticIndices := append(posAbsVariableIndices, relaxationAuxVarIdx)
				elasticWeights := append(con.Weights, -1.0)
				// finalConstraint - t <= UB
				task.PutARow(constraintIdx, elasticIndices, elasticWeights)
				task.PutConBound(constraintIdx, mosek.BK_UP, math.Inf(-1), con.ConstraintItem.UpperBound)
				constraintIdx += 1
				// add linear penalty term on t -> minimize (max(finalConstraint - UB, 0) * elasticLambda -> minimize elasticLambda * t -> maximize -elasticLambda * t
				linearCoefs[relaxationAuxVarIdx] = -1.0 * elasticPenalty
				relaxationAuxVarIdx += 1
			}

			if math.Abs(softLambda) > 1e-10 {
				for wi, i := range posAbsVariableIndices {
					for wj, j := range posAbsVariableIndices {
						if i >= j {
							if _, exists := quadCoefs[i]; exists {
								quadCoefs[i][j] -= con.Weights[wi] * con.Weights[wj] * softLambda
							} else {
								quadCoefs[i] = map[int32]float64{j: -con.Weights[wi] * con.Weights[wj] * softLambda}
							}
						}
					}
				}
			}
		case "GROSS_TRADE":
			task.AppendCons(1)
			tradeAbsVariableIndices := make([]int32, len(con.TradeVariableIndices))
			for k := 0; k < len(con.TradeVariableIndices); k++ {
				tradeAbsVariableIndices[k] = tradeToTUB[con.TradeVariableIndices[k]]
			}
			task.PutARow(constraintIdx, tradeAbsVariableIndices, con.Weights)
			if con.ConstraintItem.LowerBound > 0 {
				fmt.Println(fmt.Sprint("gross traded constraint", con.ConstraintItem.ConstraintName, " lower bound(", con.ConstraintItem.LowerBound, ") > 0 - not honored"))
			}
			task.PutConBound(constraintIdx, mosek.BK_RA, 0, con.ConstraintItem.UpperBound)
			constraintIdx += 1

			if math.Abs(softLambda) > 1e-10 {
				for wi, i := range tradeAbsVariableIndices {
					for wj, j := range tradeAbsVariableIndices {
						if i >= j {
							if _, exists := quadCoefs[i]; exists {
								quadCoefs[i][j] -= con.Weights[wi] * con.Weights[wj] * softLambda
							} else {
								quadCoefs[i] = map[int32]float64{j: -con.Weights[wi] * con.Weights[wj] * softLambda}
							}
						}
					}
				}
			}
		default:
			panic("unsupported constraint type: " + con.ConstraintItem.ConstraintType + "(" + con.ConstraintItem.ConstraintName + ")")
		}
	}
	// set up task obj
	for i, jvs := range quadCoefs {
		for j, v := range jvs {
			//fmt.Println(i, ",", j, ",", v)
			task.PutQObjIJ(i, j, v)
		}
	}
	for j, cj := range linearCoefs {
		task.PutCJ(j, cj)
	}

	/* Run optimizer */
	if verbose {
		task.PutStreamFunc(mosek.STREAM_LOG, func(msg string) { fmt.Print(msg) })
	}
	task.PutObjSense(mosek.OBJECTIVE_SENSE_MAXIMIZE)
	trmcode := task.Optimize()

	// produce the result
	if task.GetRes() != mosek.RES_OK {
		symname, rstr, _ := mosek.GetCodeDesc(task.GetRes())
		fmt.Println(fmt.Sprint("taks.GetResult() != mosek.RES_OK, Error code desc = ", symname, ", rstr = ", rstr))
		inputs.PrettyPrint()
		panic(fmt.Sprint("taks.GetResult() != mosek.RES_OK, Error code desc = ", symname, ", rstr = ", rstr))
	}
	res := make([]OPT_ResultItem, numvar)
	// initialize the result to the current state, so if anything is wrong, we do not trade
	for i := int32(0); i < numvar; i++ {
		res[i] = OPT_ResultItem{
			Symbol:          inputColumns.Univ[i],
			Trade:           0.0,
			CurrentPosition: inputColumns.Position[i],
			TargetPosition:  inputColumns.Position[i],
		}
	}
	var err error
	err = nil

	solsta := task.GetSolSta(mosek.SOL_ITR)
	if task.GetRes() != mosek.RES_OK {
		/* Print a summary containing information about the solution
		/* for debugging purposes. */
		task.SolutionSummary(mosek.STREAM_LOG)
		err = errors.New("failed getting mosek task result: task.GetRes() != mosek.RES_OK")
	} else {
		switch solsta {
		case mosek.SOL_STA_OPTIMAL:
			err = nil
		case mosek.SOL_STA_DUAL_INFEAS_CER:
			fallthrough
		case mosek.SOL_STA_PRIM_INFEAS_CER:
			err = errors.New("Primal or dual infeasibility certificate found.")
		case mosek.SOL_STA_UNKNOWN:
			symname, rstr, _ := mosek.GetCodeDesc(trmcode)
			if trmcode == mosek.RES_OK {
				fmt.Println("Solution status is unknown but result code is OK: could be a degenerate case?")
				err = nil
			} else if trmcode != mosek.RES_TRM_STALL {
				err = errors.New(fmt.Sprint("The solution status is unknown.\n", "The optimizer terminitated with code: \n", symname, "\n", rstr))
			} else {
				fmt.Println("RES_TRM_STALL: " + symname + " ," + rstr + ", we return the stalled solution")
				err = nil // we allow RES_TRM_STALL to pass, just with a warninng messaage
			}
		default:
			err = errors.New("unknow solution status: " + fmt.Sprint(solsta))
		}
		if err == nil {
			xx := task.GetXx(mosek.SOL_ITR, nil)
			for i := int32(0); i < numvar; i++ {
				res[i].Trade = xx[i]
				res[i].TargetPosition = inputColumns.Position[i] + xx[i]
			}
		}
	}
	return res, constraintInBreachWithZeroTraded, err
}
