// csvport is a portfolio trade optimizer CLI built on the MOSEK convex QP
// solver. It reads positions/alphas, risk constraints and constraint weights
// from CSV files, computes target trades, and writes the result back to CSV.
//
// Sub-command: opt  (see CsvPortOptCmd for flags)
package main

import (
	"csvport/opt"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

// CsvPortOptArgsType holds the parsed command line flags of the `opt`
// sub-command. Flag registration happens in init() below.
type CsvPortOptArgsType struct {
	input_file      string  // -i: per-symbol inputs (position, alpha, bounds...)
	constraint_file string  // -c: constraint list (name, lb, ub, type)
	weight_file     string  // -w: constraint-name -> symbol weight mapping
	output_file     string  // -o: trade result written here
	engine          string  // --engine: admm (default, pure Go) | mosek (build tag)
	soft_lambda     float64 // -s: quadratic soft-penalty weight for constraints
	elastic_penalty float64 //     linear penalty pulling relaxed constraints back
	verbose         bool    // -v: forward solver log/stream to stdout
}

// cpargs is the single flag set instance for the opt sub-command.
var cpargs CsvPortOptArgsType

// init registers the `opt` sub-command and its flags on the root command.
func init() {
	c := CsvPortOptCmd
	c.Flags().StringVarP(&cpargs.input_file, "input", "i", "", "input csv file")
	_ = c.MarkFlagRequired("input")
	c.Flags().StringVarP(&cpargs.constraint_file, "constraint", "c", "", "constraint csv file")
	_ = c.MarkFlagRequired("constraint")
	c.Flags().StringVarP(&cpargs.weight_file, "weight", "w", "", "weight csv file")
	_ = c.MarkFlagRequired("weight")
	c.Flags().StringVarP(&cpargs.output_file, "output", "o", "", "output csv file")
	_ = c.MarkFlagRequired("output")

	for _, c_ := range []*cobra.Command{CsvPortOptCmd} {
		c_.Flags().StringVar(&cpargs.engine, "engine", "admm", "solver engine: admm (pure Go, default) or mosek (if compiled with -tags mosek)")
		c_.Flags().Float64VarP(&cpargs.soft_lambda, "soft-lambda", "s", 0.0, "soft constraint lambda")
		c_.Flags().Float64VarP(&cpargs.elastic_penalty, "elastic-penalty", "", 100.0, "elastic penalty to pull back net and gross constraints in breach with zero traded")
		c_.Flags().BoolVarP(&cpargs.verbose, "verbose", "v", false, "print out optimizer streams")
	}

	RootCmd.AddCommand(CsvPortOptCmd)
}

// CsvPortOptCmd is the `opt` sub-command: load the three input CSVs, solve
// the trade optimization, pretty-print the result and write it to --output.
var CsvPortOptCmd = &cobra.Command{
	Use:   "opt",
	Short: "opt",
	RunE: func(cmd *cobra.Command, args []string) (err error) {
		// Mask process name/args in top/htop/ps. Must run AFTER cobra finished
		// flag parsing: setProcTitle wipes the memory behind os.Args. pflag's
		// parsed string values are substrings of os.Args (not copies), so we
		// must clone every path BEFORE masking, otherwise they get wiped too.
		inputFile := strings.Clone(cpargs.input_file)
		constraintFile := strings.Clone(cpargs.constraint_file)
		weightFile := strings.Clone(cpargs.weight_file)
		outputFile := strings.Clone(cpargs.output_file)
		engine := strings.Clone(cpargs.engine)
		setComm(procTitle)
		setProcTitle(procTitle)
		// get inputs
		inputs, err := opt.LoadOptInputsFromCSV(inputFile)
		if err != nil {
			fmt.Println(err)
			return err
		}
		inputs.PrettyPrint()
		// get constraints
		constraints, err := opt.LoadOptConstraintsFromCSV(constraintFile)
		if err != nil {
			fmt.Println(err)
			return err
		}
		constraints.PrettyPrint()
		// get weights
		weights, err := opt.LoadOptWeightsFromCSV(weightFile)
		if err != nil {
			fmt.Println(err)
			return err
		}
		weights.PrettyPrint()

		res, err := opt.Solve(engine, inputs, constraints, weights, cpargs.soft_lambda, cpargs.elastic_penalty, cpargs.verbose)
		res.PrettyPrint()
		res.ToCSV(outputFile)

		// TODO: consider showing constraints breach stats
		return err
	},
}

// RootCmd represents the base command when called without any subcommands
// (the only sub-command today is `opt`).
var RootCmd = &cobra.Command{
	Use:   "csvport",
	Short: "cats CLI - Consolidated Algo Trading Strats",
}

// main parses the command line and dispatches to the sub-command.
func main() {
	if err := RootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
