//go:build mosek

// backend_mosek.go: the original MOSEK-based backend, verbatim port of the
// pre-refactor opt/solver.go solve(). Compiled only with build tag "mosek"
// (requires MOSEK 9.3 headers & libs via CGO_CFLAGS/CGO_LDFLAGS, see README).
package opt

import (
	"os"
	"strconv"
	"errors"
	"fmt"
	"github.com/mosek/mosek.go"
	"math"
	"strings"
)

func init() {
	registerSolver("mosek", solveMosek)
}

// solveMosek builds and optimizes one MOSEK problem (original implementation).
//
// Decision variables:
//   - x_i            trade of universe symbol i, i in [0, numvar)
//   - TUB u_i        >= |x_i|          (linearizes |x| for cost & GROSS_TRADE)
//   - GUB g_i        >= |pos_i + x_i|  (linearizes |pos+x| for GROSS)
//   - elastic t_k    >= breach of relaxed constraint k (relax mode only)
//
// Objective (maximized):
//
//	sum_i [ alpha_i*x_i - lambda_i*pos_i*x_i - cost_i*u_i ]
//	- x'*(Lambda+TLambda)*x
//	- elasticPenalty * sum_k t_k               (relax mode only)
func solveMosek(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights,
	softLambda float64, relaxMode bool, elasticPenalty float64, verbose bool) (OPT_Result, error) {

	constraintInBreachWithZeroTraded := false
	inputColumns := inputs.ToColumns()
	constraintDetails := AssembleConstraintDetails(inputColumns.Univ, constraints, weights)
	numvar := int32(len(inputColumns.Univ))
	quadCoefs := map[int32]map[int32]float64{}
	linearCoefs := map[int32]float64{}

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
				if _, exists := tradeToGUB[ti]; !exists {
					tradeToGUB[ti] = gubIdx
					gubToTrade[gubIdx] = ti
					gubIdx += 1
				}
			}
		}
	}

	/* Create the mosek environment. */
	env, r := mosek.MakeEnv()
	if r != 0 {
		symname, rstr, _ := mosek.GetCodeDesc(r)
		errmsg := "mosek error code at mosek.MakeEnv: " + fmt.Sprint(r) + " mosek.GetCodeDesc returns:" + symname + ", " + rstr
		fmt.Sprintln(errmsg)
		return nil, errors.New(errmsg)
	}
	defer env.DeleteEnv()
	task, r := env.MakeTask()
	defer task.DeleteTask()
	if r != 0 {
		symname, rstr, _ := mosek.GetCodeDesc(r)
		errmsg := "mosek error code at mosek.MakeTask: " + fmt.Sprint(r) + " mosek.GetCodeDesc returns:" + symname + ", " + rstr
		fmt.Sprintln(errmsg)
		return nil, errors.New(errmsg)
	}

	// Append traded variables. The variables will initially be fixed at zero (x=0).
	task.AppendVars(numvar)
	for i := int32(0); i < numvar; i++ {
		task.PutVarBound(i, mosek.BK_RA,
			math.Max(inputColumns.MinTrade[i], inputColumns.MinPos[i]-inputColumns.Position[i]),
			math.Min(inputColumns.MaxTrade[i], inputColumns.MaxPos[i]-inputColumns.Position[i]))
	}
	constraintIdx := int32(0)

	// set up TUB
	task.AppendVars(int32(len(tubToTrade)))
	for uidx, i := range tubToTrade {
		task.PutVarBound(uidx, mosek.BK_LO, 0.0, math.Inf(1))
		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{i, uidx}, []float64{1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, 0.0, math.Inf(1))
		constraintIdx += 1

		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{i, uidx}, []float64{-1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, 0.0, math.Inf(1))
		constraintIdx += 1

		linearCoefs[uidx] = -inputColumns.TradingCost[i]
	}

	// set up GUB
	task.AppendVars(int32(len(gubToTrade)))
	for gidx, tradeIdx := range gubToTrade {
		task.PutVarBound(gidx, mosek.BK_LO, 0.0, math.Inf(1))
		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{tradeIdx, gidx}, []float64{-1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, inputColumns.Position[tradeIdx], math.Inf(1))
		constraintIdx += 1

		task.AppendCons(1)
		task.PutARow(constraintIdx, []int32{tradeIdx, gidx}, []float64{1, 1})
		task.PutConBound(constraintIdx, mosek.BK_LO, -inputColumns.Position[tradeIdx], math.Inf(1))
		constraintIdx += 1
	}

	// set up the objective function
	for i := int32(0); i < numvar; i++ {
		quadCoefs[i] = map[int32]float64{i: -(inputColumns.Lambda[i] + inputColumns.TradingLambda[i])}
		linearCoefs[i] = inputColumns.Alpha[i] - inputColumns.Lambda[i]*inputColumns.Position[i]
	}
	// set up the constraints
	relaxationAuxVarIdx := gubIdx
	for _, con := range constraintDetails {
		switch strings.ToUpper(con.ConstraintItem.ConstraintType) {
		case "NET":
			task.AppendCons(1)
			task.PutARow(constraintIdx, con.TradeVariableIndices, con.Weights)
			constAdj := 0.0
			for j, k := range con.TradeVariableIndices {
				constAdj += con.Weights[j] * inputColumns.Position[k]
			}
			lb := con.ConstraintItem.LowerBound - constAdj
			ub := con.ConstraintItem.UpperBound - constAdj
			breach := lb > 0 || ub < 0
			constraintInBreachWithZeroTraded = constraintInBreachWithZeroTraded || breach
			if relaxMode && breach {
				if lb > 0 {
					lb = -epsilon
				}
				if ub < 0 {
					ub = epsilon
				}
			}
			task.PutConBound(constraintIdx, mosek.BK_RA, lb, ub)
			constraintIdx += 1
			if relaxMode && breach {
				if con.ConstraintItem.LowerBound > constAdj {
					task.AppendVars(1)
					task.PutVarBound(relaxationAuxVarIdx, mosek.BK_LO, 0.0, math.Inf(1))
					task.AppendCons(1)
					elasticIndices := append(con.TradeVariableIndices, relaxationAuxVarIdx)
					elasticWeights := append(con.Weights, 1.0)
					task.PutARow(constraintIdx, elasticIndices, elasticWeights)
					task.PutConBound(constraintIdx, mosek.BK_LO, con.ConstraintItem.LowerBound-constAdj, math.Inf(1))
					constraintIdx += 1
					linearCoefs[relaxationAuxVarIdx] = -elasticPenalty
					relaxationAuxVarIdx += 1
				}
				if con.ConstraintItem.UpperBound < constAdj {
					task.AppendVars(1)
					task.PutVarBound(relaxationAuxVarIdx, mosek.BK_LO, 0.0, math.Inf(1))
					task.AppendCons(1)
					elasticIndices := append(con.TradeVariableIndices, relaxationAuxVarIdx)
					elasticWeights := append(con.Weights, -1.0)
					task.PutARow(constraintIdx, elasticIndices, elasticWeights)
					task.PutConBound(constraintIdx, mosek.BK_UP, math.Inf(-1), con.ConstraintItem.UpperBound-constAdj)
					constraintIdx += 1
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

			if relaxMode && breach {
				task.AppendVars(1)
				task.PutVarBound(relaxationAuxVarIdx, mosek.BK_LO, 0.0, math.Inf(1))
				task.AppendCons(1)
				elasticIndices := append(posAbsVariableIndices, relaxationAuxVarIdx)
				elasticWeights := append(con.Weights, -1.0)
				task.PutARow(constraintIdx, elasticIndices, elasticWeights)
				task.PutConBound(constraintIdx, mosek.BK_UP, math.Inf(-1), con.ConstraintItem.UpperBound)
				constraintIdx += 1
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
	for i, jvs := range quadCoefs {
		for j, v := range jvs {
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
	// eps override for precision-tier experiments (matches cpp_admm's
	// --eps-abs/--eps-rel semantics): MOSEK_INTPNT_EPS=v sets all three
	// interior-point tolerances (relative gap + primal/dual feasibility).
	// Also prints the solver-internal time (MSK_DINF_OPTIMIZER_TIME=31)
	// to stderr as "mosekOptMs=..." for sweep harnesses.
	if epsOverride := os.Getenv("MOSEK_INTPNT_EPS"); epsOverride != "" {
		v, err := strconv.ParseFloat(epsOverride, 64)
		if err == nil && v > 0 {
			task.PutDouParam(mosek.DPAR_INTPNT_TOL_REL_GAP, v)
			task.PutDouParam(mosek.DPAR_INTPNT_TOL_PFEAS, v)
			task.PutDouParam(mosek.DPAR_INTPNT_TOL_DFEAS, v)
		}
	}
	task.PutObjSense(mosek.OBJECTIVE_SENSE_MAXIMIZE)
	trmcode := task.Optimize()
	if tmsk := task.GetDouInf(31 /*MSK_DINF_OPTIMIZER_TIME*/); tmsk > 0 {
		fmt.Fprintf(os.Stderr, "mosekOptMs=%.1f\n", tmsk*1000)
	}

	if task.GetRes() != mosek.RES_OK {
		symname, rstr, _ := mosek.GetCodeDesc(task.GetRes())
		fmt.Println(fmt.Sprint("taks.GetResult() != mosek.RES_OK, Error code desc = ", symname, ", rstr = ", rstr))
		inputs.PrettyPrint()
		panic(fmt.Sprint("taks.GetResult() != mosek.RES_OK, Error code desc = ", symname, ", rstr = ", rstr))
	}
	res := make([]OPT_ResultItem, numvar)
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
				err = nil
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
	_ = constraintInBreachWithZeroTraded // computed for parity; Solve precomputes it from data
	return res, err
}
