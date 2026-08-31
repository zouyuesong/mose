// Package opt implements a portfolio trade optimizer on top of MOSEK:
// CSV input types & loaders (types.go) and the QP model builder (solver.go).
package opt

import (
	"fmt"
	"github.com/gocarina/gocsv"
	"github.com/olekukonko/tablewriter"
	"os"
	"strconv"
	"strings"
)

// OPT_InputItem is one row of the input CSV (one universe symbol).
// Columns without a csv tag in the file (e.g. masterAlpha_exec,
// minPerTradeDv01, isHoliday, ...) are ignored by the loader.
type OPT_InputItem struct {
	Symbol      string  `csv:"sym"`         // instrument identifier, must be unique
	Alpha       float64 `csv:"alpha"`       // expected return signal, drives linear PnL term
	Position    float64 `csv:"position"`    // current position, in risk units (e.g. dv01)
	Lambda      float64 `csv:"lambda"`      // risk aversion: penalizes (pos+x)^2 in the objective
	TLambda     float64 `csv:"tlambda"`     // trade aversion: penalizes x^2 in the objective
	TradingCost float64 `csv:"cost"`        // linear cost applied to |x| (turnover cost)
	MinTrade    float64 `csv:"minTrade"`    // hard lower bound on the trade x
	MaxTrade    float64 `csv:"maxTrade"`    // hard upper bound on the trade x
	MinPosition float64 `csv:"minPosition"` // hard lower bound on final position pos+x
	MaxPosition float64 `csv:"maxPosition"` // hard upper bound on final position pos+x
}

// OPT_Inputs is the ordered list of input rows.
type OPT_Inputs []*OPT_InputItem

// LoadOptInputsFromCSV reads the per-symbol input CSV produced upstream.
func LoadOptInputsFromCSV(fname string) (OPT_Inputs, error) {
	csvFile, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer csvFile.Close()
	res := OPT_Inputs{}
	if err := gocsv.UnmarshalFile(csvFile, &res); err != nil { // Load raw order info from file
		return nil, err
	}
	return res, nil
}

// PrettyPrint dumps the input table to stdout as an aligned table.
func (inputs OPT_Inputs) PrettyPrint() {
	var buffer strings.Builder
	table := tablewriter.NewWriter(&buffer)
	table.SetHeader([]string{"Sym", "Alpha", "Position", "Lambda", "TLambda", "Cost", "MinTrade", "MaxTrade", "MinPos", "MaxPos"})
	table.SetHeaderAlignment(tablewriter.ALIGN_RIGHT)
	table.SetAlignment(tablewriter.ALIGN_RIGHT)
	table.SetFooterAlignment(tablewriter.ALIGN_RIGHT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	for _, i := range inputs {
		v := make([]string, 10)
		v[0] = i.Symbol
		v[1] = strconv.FormatFloat(i.Alpha, 'G', 4, 64)
		v[2] = strconv.FormatFloat(i.Position, 'G', 4, 64)
		v[3] = strconv.FormatFloat(i.Lambda, 'G', 4, 64)
		v[4] = strconv.FormatFloat(i.TLambda, 'G', 4, 64)
		v[5] = strconv.FormatFloat(i.TradingCost, 'G', 5, 64)
		v[6] = strconv.FormatFloat(i.MinTrade, 'G', 2, 64)
		v[7] = strconv.FormatFloat(i.MaxTrade, 'G', 2, 64)
		v[8] = strconv.FormatFloat(i.MinPosition, 'G', 2, 64)
		v[9] = strconv.FormatFloat(i.MaxPosition, 'G', 2, 64)
		table.Append(v)
	}
	table.Render() // Send output
	res := buffer.String()
	fmt.Println(res)
}

// OPT_InputsColumns holds the input table as column vectors; index i of every
// slice corresponds to the same symbol Univ[i]. The solver consumes this form.
type OPT_InputsColumns struct {
	Univ          []string
	Position      []float64
	Alpha         []float64
	TradingCost   []float64
	MinTrade      []float64
	MaxTrade      []float64
	MinPos        []float64
	MaxPos        []float64
	Lambda        []float64
	TradingLambda []float64
}

// ToColumns transposes the row-oriented input list into column vectors.
// Input symbols are assumed unique.
func (inputs OPT_Inputs) ToColumns() OPT_InputsColumns {
	n := len(inputs)
	univ := make([]string, n)
	position := make([]float64, n)
	alpha := make([]float64, n)
	minTrades := make([]float64, n)
	maxTrades := make([]float64, n)
	minPos := make([]float64, n)
	maxPos := make([]float64, n)
	tradingCost := make([]float64, n)
	lambda := make([]float64, n)
	tlambda := make([]float64, n)

	for idx, i := range inputs {
		univ[idx] = i.Symbol
		position[idx] = i.Position
		alpha[idx] = i.Alpha
		minTrades[idx] = i.MinTrade
		maxTrades[idx] = i.MaxTrade
		minPos[idx] = i.MinPosition
		maxPos[idx] = i.MaxPosition
		lambda[idx] = i.Lambda
		tlambda[idx] = i.TLambda
		tradingCost[idx] = i.TradingCost
	}
	res := OPT_InputsColumns{
		Univ:          univ,
		Position:      position,
		Alpha:         alpha,
		TradingCost:   tradingCost,
		MinTrade:      minTrades,
		MaxTrade:      maxTrades,
		MinPos:        minPos,
		MaxPos:        maxPos,
		Lambda:        lambda,
		TradingLambda: tlambda,
	}
	return res
}

// OPT_ConstraintItem is one row of the constraint CSV: a hard bound pair
// [lb, ub] on the quantity identified by (name, type), whose per-symbol
// weights live in the weight CSV.
type OPT_ConstraintItem struct {
	ConstraintName string  `csv:"name"` // links to weight rows and must be unique
	LowerBound     float64 `csv:"lb"`   // lower bound (ignored by GROSS/GROSS_TRADE)
	UpperBound     float64 `csv:"ub"`   // upper bound (the binding bound for GROSS*)
	ConstraintType string  `csv:"type"` // NET | NET_TRADE | GROSS | GROSS_TRADE
}

// OPT_Constrints is the ordered list of constraint rows.
type OPT_Constrints []*OPT_ConstraintItem

// PrettyPrint dumps the constraint list to stdout as an aligned table.
func (constraints OPT_Constrints) PrettyPrint() {
	var buffer strings.Builder
	table := tablewriter.NewWriter(&buffer)
	table.SetHeader([]string{"Name", "LB", "UB", "TYPE"})
	table.SetHeaderAlignment(tablewriter.ALIGN_RIGHT)
	table.SetAlignment(tablewriter.ALIGN_RIGHT)
	table.SetFooterAlignment(tablewriter.ALIGN_RIGHT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	for _, c := range constraints {
		v := make([]string, 4)
		v[0] = c.ConstraintName
		v[1] = strconv.FormatFloat(c.LowerBound, 'G', 2, 64)
		v[2] = strconv.FormatFloat(c.UpperBound, 'G', 2, 64)
		v[3] = c.ConstraintType
		table.Append(v)
	}
	table.Render() // Send output
	res := buffer.String()
	fmt.Println(res)
}

// LoadOptConstraintsFromCSV reads the constraint definition CSV.
func LoadOptConstraintsFromCSV(fname string) (OPT_Constrints, error) {
	csvFile, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer csvFile.Close()
	res := OPT_Constrints{}
	if err := gocsv.UnmarshalFile(csvFile, &res); err != nil { // Load raw order info from file
		return nil, err
	}
	return res, nil
}

// OPT_WeightItem is one row of the weight CSV: symbol sym contributes with
// coefficient weight to the constraint named ConstraintName. Symbols absent
// from a constraint's rows simply do not participate in it (implicit weight 0).
type OPT_WeightItem struct {
	ConstraintName string  `csv:"name"`   // which constraint this weight belongs to
	Symbol         string  `csv:"sym"`    // which universe symbol it weights
	Weight         float64 `csv:"weight"` // linear coefficient in the constraint
}

// OPT_Weights is the ordered list of weight rows.
type OPT_Weights []*OPT_WeightItem

// LoadOptWeightsFromCSV reads the constraint weight mapping CSV.
func LoadOptWeightsFromCSV(fname string) (OPT_Weights, error) {
	csvFile, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer csvFile.Close()
	res := OPT_Weights{}
	if err := gocsv.UnmarshalFile(csvFile, &res); err != nil { // Load raw order info from file
		return nil, err
	}
	return res, nil
}

// PrettyPrint dumps the weight table to stdout as an aligned table.
func (weights OPT_Weights) PrettyPrint() {
	var buffer strings.Builder
	table := tablewriter.NewWriter(&buffer)
	table.SetHeader([]string{"Constraint", "SYM", "Weight"})
	table.SetHeaderAlignment(tablewriter.ALIGN_RIGHT)
	table.SetAlignment(tablewriter.ALIGN_RIGHT)
	table.SetFooterAlignment(tablewriter.ALIGN_RIGHT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	for _, w := range weights {
		v := make([]string, 3)
		v[0] = w.ConstraintName
		v[1] = w.Symbol
		v[2] = strconv.FormatFloat(w.Weight, 'G', 2, 64)
		table.Append(v)
	}
	table.Render() // Send output
	res := buffer.String()
	fmt.Println(res)
}

// Tabulate converts the weight rows into a nested map:
// constraint name -> (symbol -> weight).
func (weights OPT_Weights) Tabulate() map[string](map[string]float64) {
	res := map[string](map[string]float64){}
	for _, wi := range weights {
		if m, exists := res[wi.ConstraintName]; exists {
			m[wi.Symbol] = wi.Weight
			res[wi.ConstraintName] = m
		} else {
			res[wi.ConstraintName] = map[string]float64{
				wi.Symbol: wi.Weight,
			}
		}
	}
	return res
}

// OPT_ResultItem is one row of the output CSV: the trade to execute and the
// resulting target position. CurrentPosition is display-only (not exported).
type OPT_ResultItem struct {
	Symbol          string  `csv:"sym"`
	Trade           float64 `csv:"targetTrade"`
	CurrentPosition float64 `csv:"-"`
	TargetPosition  float64 `csv:"targetPosition"`
}

// OPT_Result is the full result, one row per universe symbol.
type OPT_Result []OPT_ResultItem

// PrettyPrint dumps the result table to stdout as an aligned table.
func (result OPT_Result) PrettyPrint() {
	var buffer strings.Builder
	table := tablewriter.NewWriter(&buffer)
	table.SetHeader([]string{"SYM", "Current", "Target", "Trade"})
	table.SetHeaderAlignment(tablewriter.ALIGN_RIGHT)
	table.SetAlignment(tablewriter.ALIGN_RIGHT)
	table.SetFooterAlignment(tablewriter.ALIGN_RIGHT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	for _, w := range result {
		v := make([]string, 4)
		v[0] = w.Symbol
		v[1] = strconv.FormatFloat(w.CurrentPosition, 'G', 6, 64)
		v[2] = strconv.FormatFloat(w.TargetPosition, 'G', 6, 64)
		v[3] = strconv.FormatFloat(w.Trade, 'G', 6, 64)
		table.Append(v)
	}
	table.Render() // Send output
	res := buffer.String()
	fmt.Println(res)
}

// OPT_AssembledConstraintItem is a constraint joined with its weights:
// TradeVariableIndices[k] is the variable index of Weights[k]'s symbol in the
// universe order, so the constraint row is sum_k Weights[k]*x[idx_k].
type OPT_AssembledConstraintItem struct {
	ConstraintItem       *OPT_ConstraintItem
	TradeVariableIndices []int32 // trade variable indices
	Weights              []float64
}

// AssembleConstraintDetails joins each constraint with the weight table and
// resolves symbols to variable indices following the universe order.
func AssembleConstraintDetails(univ []string, constraints OPT_Constrints, weights OPT_Weights) []OPT_AssembledConstraintItem {
	res := make([]OPT_AssembledConstraintItem, len(constraints))
	numvar := int32(len(univ))
	weightsTab := weights.Tabulate()
	for i, c := range constraints {
		wmap := weightsTab[c.ConstraintName]
		indices := []int32{}
		ww := []float64{}
		for idx := int32(0); idx < numvar; idx++ {
			if w, exists := wmap[univ[idx]]; exists {
				ww = append(ww, w)
				indices = append(indices, idx)
			}
		}
		res[i] = OPT_AssembledConstraintItem{
			ConstraintItem:       c,
			TradeVariableIndices: indices,
			Weights:              ww,
		}
	}
	return res
}

//////// ToCSV ////////////////////////////

// ToCSV serializes inputs back to a CSV file.
func (inputs OPT_Inputs) ToCSV(filename string) error {
	csvFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer csvFile.Close()
	err = gocsv.MarshalFile(&inputs, csvFile)
	if err != nil {
		return err
	}
	return nil
}

// ToCSV serializes constraints back to a CSV file.
func (constraints OPT_Constrints) ToCSV(filename string) error {
	csvFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer csvFile.Close()
	err = gocsv.MarshalFile(&constraints, csvFile)
	if err != nil {
		return err
	}
	return nil
}

// ToCSV serializes weights back to a CSV file.
func (weights OPT_Weights) ToCSV(filename string) error {
	csvFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer csvFile.Close()
	err = gocsv.MarshalFile(&weights, csvFile)
	if err != nil {
		return err
	}
	return nil
}

// ToCSV writes the optimization result (sym, targetTrade, targetPosition)
// to a CSV file.
func (result OPT_Result) ToCSV(filename string) error {
	csvFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer csvFile.Close()
	err = gocsv.MarshalFile(&result, csvFile)
	if err != nil {
		return err
	}
	return nil
}
