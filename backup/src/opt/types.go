package opt

import (
	"fmt"
	"github.com/gocarina/gocsv"
	"github.com/olekukonko/tablewriter"
	"os"
	"strconv"
	"strings"
)

type OPT_InputItem struct {
	Symbol      string  `csv:"sym"`
	Alpha       float64 `csv:"alpha"`
	Position    float64 `csv:"position"`
	Lambda      float64 `csv:"lambda"`
	TLambda     float64 `csv:"tlambda"`
	TradingCost float64 `csv:"cost"`
	MinTrade    float64 `csv:"minTrade"`
	MaxTrade    float64 `csv:"maxTrade"`
	MinPosition float64 `csv:"minPosition"`
	MaxPosition float64 `csv:"maxPosition"`
}
type OPT_Inputs []*OPT_InputItem

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

// ordered lists of the input table
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

// we assume input is unique
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

type OPT_ConstraintItem struct {
	ConstraintName string  `csv:"name"`
	LowerBound     float64 `csv:"lb"`
	UpperBound     float64 `csv:"ub"`
	ConstraintType string  `csv:"type"`
}
type OPT_Constrints []*OPT_ConstraintItem

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

type OPT_WeightItem struct {
	ConstraintName string  `csv:"name"`
	Symbol         string  `csv:"sym"`
	Weight         float64 `csv:"weight"`
}
type OPT_Weights []*OPT_WeightItem

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

// given a constraint name, get the weights map
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

type OPT_ResultItem struct {
	Symbol          string  `csv:"sym"`
	Trade           float64 `csv:"targetTrade"`
	CurrentPosition float64 `csv:"-"`
	TargetPosition  float64 `csv:"targetPosition"`
}
type OPT_Result []OPT_ResultItem

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

type OPT_AssembledConstraintItem struct {
	ConstraintItem       *OPT_ConstraintItem
	TradeVariableIndices []int32 // trade variable indices
	Weights              []float64
}

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
