package main

import (
	"csvport/opt"
	"fmt"
	"github.com/spf13/cobra"
	"os"
)

type CsvPortOptArgsType struct {
	input_file      string
	constraint_file string
	weight_file     string
	output_file     string
	soft_lambda     float64
	elastic_penalty float64
	verbose         bool
}

var cpargs CsvPortOptArgsType

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
		c_.Flags().Float64VarP(&cpargs.soft_lambda, "soft-lambda", "s", 0.0, "soft constraint lambda")
		c_.Flags().Float64VarP(&cpargs.elastic_penalty, "elastic-penalty", "", 100.0, "elastic penalty to pull back net and gross constraints in breach with zero traded")
		c_.Flags().BoolVarP(&cpargs.verbose, "verbose", "v", false, "print out optimizer streams")
	}

	RootCmd.AddCommand(CsvPortOptCmd)
}

var CsvPortOptCmd = &cobra.Command{
	Use:   "opt",
	Short: "opt",
	RunE: func(cmd *cobra.Command, args []string) (err error) {
		// get inputs
		inputs, err := opt.LoadOptInputsFromCSV(cpargs.input_file)
		inputs.PrettyPrint()
		if err != nil {
			fmt.Println(err)
			return err
		}
		// get constraints
		constraints, err := opt.LoadOptConstraintsFromCSV(cpargs.constraint_file)
		constraints.PrettyPrint()
		if err != nil {
			fmt.Println(err)
			return err
		}
		// get weights
		weights, err := opt.LoadOptWeightsFromCSV(cpargs.weight_file)
		weights.PrettyPrint()
		if err != nil {
			fmt.Println(err)
			return err
		}

		res, err := opt.Solve(inputs, constraints, weights, cpargs.soft_lambda, cpargs.elastic_penalty, cpargs.verbose)
		res.PrettyPrint()
		res.ToCSV(cpargs.output_file)

		// TODO: consider showing constraints breach stats
		return err
	},
}

// RootCmd represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "csvport",
	Short: "cats CLI - Consolidated Algo Trading Strats",
}

func main() {
	if err := RootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
