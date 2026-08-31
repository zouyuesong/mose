package opt

import (
	"fmt"
	"math"
	"strings"
)

const (
	epsilon float64 = 1e-10
)

// solverFunc is the backend contract: build & solve one QP for the given
// (possibly pre-relaxed) inputs. Backends register themselves via
// registerSolver in init() (admm: always; mosek: build tag "mosek").
type solverFunc func(inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights,
	softLambda float64, relaxMode bool, elasticPenalty float64, verbose bool) (OPT_Result, error)

var solverRegistry = map[string]solverFunc{}

func registerSolver(name string, f solverFunc) {
	solverRegistry[name] = f
}

// AvailableSolvers lists registered engine names.
func AvailableSolvers() []string {
	var names []string
	for k := range solverRegistry {
		names = append(names, k)
	}
	return names
}

// Solve runs the trade optimization with a two-pass fallback:
//
// Pass 1 (normal): solve with the user-given bounds. Before building the
// problem, per-symbol position bounds are pre-relaxed when the current
// position already breaches them together with the trade bounds (e.g.
// Position+MinTrade > MaxPosition), so a feasible x always exists per symbol.
//
// Pass 2 (relax mode): only if pass 1 fails AND some NET/GROSS constraint is
// in breach even with zero traded (i.e. the current book already violates a
// risk limit, which no trade can undo without breaching symbol bounds). Then
// position bounds are re-anchored at the current position and breached
// NET/GROSS bounds are relaxed to "no worse than today", with an elastic
// penalty (elasticPenalty per unit) pulling the solution back toward the
// original bounds.
func Solve(engine string, inputs OPT_Inputs, constraints OPT_Constrints, weights OPT_Weights, softLambda float64, elasticPenalty float64, verbose bool) (OPT_Result, error) {
	backend, ok := solverRegistry[engine]
	if !ok {
		return nil, fmt.Errorf("unknown solver engine %q (available: %v)", engine, AvailableSolvers())
	}
	// STEP 1. relax max position constraints if current position with maxTrade and minTrade will be in breach
	for _, i := range inputs {
		if i.Position+i.MinTrade > i.MaxPosition {
			i.MaxPosition = i.Position + i.MinTrade + epsilon
		}
		if i.Position+i.MaxTrade < i.MinPosition {
			i.MinPosition = i.Position + i.MaxTrade - epsilon
		}
	}
	cols := inputs.ToColumns()
	det := AssembleConstraintDetails(cols.Univ, constraints, weights)
	constraintInBreachWithZeroTraded := breachWithZeroTraded(cols, det)

	if res, err := backend(inputs, constraints, weights, softLambda, false, elasticPenalty, verbose); err == nil || !constraintInBreachWithZeroTraded {
		if res == nil {
			res = noTradeResult(cols)
		}
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
		res, err := backend(inputs, constraints, weights, softLambda, true, elasticPenalty, verbose)
		if res == nil {
			res = noTradeResult(cols)
		}
		return res, err
	}
}

// constraintType normalizes the CSV type field (case-insensitive).
func constraintType(s string) string { return strings.ToUpper(s) }

// breachWithZeroTraded reports whether any NET/GROSS constraint is violated
// when x = 0 (the current book already breaks the bound). This mirrors the
// flag the original mosek solve computed during model building.
func breachWithZeroTraded(cols OPT_InputsColumns, det []OPT_AssembledConstraintItem) bool {
	for _, con := range det {
		switch constraintType(con.ConstraintItem.ConstraintType) {
		case "NET":
			constAdj := 0.0
			for j, k := range con.TradeVariableIndices {
				constAdj += con.Weights[j] * cols.Position[k]
			}
			lb := con.ConstraintItem.LowerBound - constAdj
			ub := con.ConstraintItem.UpperBound - constAdj
			if lb > 0 || ub < 0 {
				return true
			}
		case "GROSS":
			zeroTradedGross := 0.0
			for j, k := range con.TradeVariableIndices {
				zeroTradedGross += math.Abs(cols.Position[k]) * con.Weights[j]
			}
			if con.ConstraintItem.UpperBound < zeroTradedGross {
				return true
			}
		}
	}
	return false
}

// noTradeResult is the fail-safe answer: do not trade anything.
func noTradeResult(cols OPT_InputsColumns) OPT_Result {
	res := make(OPT_Result, len(cols.Univ))
	for i := range res {
		res[i] = OPT_ResultItem{
			Symbol:          cols.Univ[i],
			Trade:           0.0,
			CurrentPosition: cols.Position[i],
			TargetPosition:  cols.Position[i],
		}
	}
	return res
}
