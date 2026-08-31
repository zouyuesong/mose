// model.hpp - QP model builder, port of golang_mosek/src/opt/solver.go
// (two-pass relax orchestration) + backend_admm.go buildADMMProblem
// (canonical form min 1/2 x'Px + q'x s.t. l <= Ax <= u, CSC, with
// fixed-variable presolve). The GeneralStart/pair-row contract and Woodbury
// solver are NOT ported: they exist only for the custom Go solver.
#pragma once
#include "csv.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

// Sparse row of the QP constraint matrix.
struct Row {
    std::vector<int> idx;   // variable indices (any order here; CSC built later)
    std::vector<double> val;
    double lb, ub;          // bounds (may be +-inf, use inf()/ninf())
    double hinge = 0;       // >0: elastic penalty c on dist(Ax, [lb,ub])
                            // (admm_st2 S2-3b Lew folding: the t_k variable
                            // and its rows are NOT materialized; engines with
                            // folding support apply the three-piece prox)
};

struct Problem {
    int n = 0;                  // number of variables
    std::vector<double> pDiag;  // diagonal P (>= 0)
    std::vector<double> q;      // linear term
    std::vector<Row> rows;      // constraints
    // per-row target scale (trade band for per-stock rows, constraint target
    // magnitude for aggregated rows), used by scaleRows() to re-anchor the
    // solvers' global relative tolerance to per-row / per-trade units.
    std::vector<double> rowScale;
    // per-variable band (x/u = trade band, g = |pos|+band, t = 1). scaleRows()
    // also column-scales by this and transforms P/q accordingly; the returned
    // reduced x must be un-scaled by the caller (x *= varScale).
    std::vector<double> varScale;
    int generalStart = -1;    // rows [0, generalStart) = per-symbol structure rows
                              // (box/TUB/GUB: diagonal-only A'A contribution);
                              // rows [generalStart, end) = general rows (rank
                              // term for the Woodbury solver in native engine)
    // fixed-variable presolve bookkeeping (backend_admm.go PresolveInfo)
    std::vector<int> active;    // reduced slot -> original index
    std::vector<bool> fixed;    // original index -> fixed?
    std::vector<double> xFix;   // trade value for fixed symbols
    int numSymbols = 0;         // original universe size
};

// AssembledConstraint mirrors opt.OPT_AssembledConstraintItem.
struct AssembledConstraint {
    const ConstraintItem *item = nullptr;
    std::vector<int> varIndices;    // into input vector order
    std::vector<double> weights;
};

// Port of STEP1 + Solve orchestration (solver.go). Modifies inputs in place
// (pre-relaxes position bounds), returns the assembled model for the given
// relaxMode. breachZero is breachWithZeroTraded(cols, det).
struct InputsCols {
    std::vector<std::string> univ;
    std::vector<double> position, alpha, tradingCost, minTrade, maxTrade,
        minPos, maxPos, lambda, tradingLambda;
};

InputsCols toColumns(std::vector<InputItem> &inputs);
std::vector<AssembledConstraint> assembleConstraints(
    const std::vector<std::string> &univ,
    const std::vector<ConstraintItem> &constraints,
    const std::vector<WeightItem> &weights);
bool breachWithZeroTraded(const InputsCols &cols,
                          const std::vector<AssembledConstraint> &det);
void step1Relax(std::vector<InputItem> &inputs);
Problem buildProblem(const InputsCols &cols,
                     const std::vector<AssembledConstraint> &det,
                     bool relaxMode, double elasticPenalty);
// Divide every row's A coefficients and bounds by its rowScale entry so that
// a solver's global eps_rel becomes "x% of each row's own scale" (=x% of the
// trade band for per-stock rows) instead of x% of the largest row (~5e7 gross).
// Feasible set / optimum unchanged; only row conditioning changes.
void scaleRows(Problem &p);

// Generic matrix-only diagonal equilibration (admm.md §3.4(a) series):
//  - applyRuiz: iterative Ruiz (row & col inf-norms -> 1, iters times);
//    same family as OSQP/PDLP/proxqp built-in scaling, applied at harness
//    level so EVERY engine sees it (engines' own scaling can then be toggled).
//  - applyPockChambolle: single-step alpha=1 variant (sqrt of row/col l1
//    norms), per ABIP+ (arXiv 2209.01793 §3.5): (D1)_rr = sqrt(||A_r.||_{2-a}),
//    (D2)_jj = sqrt(||A_.j||_a), a=1.
// Both update varScale (solution must be un-scaled: x *= varScale), transform
// P/q for column scaling and divide rows/bounds for row scaling. Feasible set
// and optimum unchanged.
void applyRuiz(Problem &p, int iters);
void applyPockChambolle(Problem &p);

// admm.md §3.4(f) harness-level polish (engine-agnostic post-solve step,
// works on the scaled problem where every row is O(1)):
//  1. classify rows as active-at-lower / active-at-upper from the solver's
//     stopping point (tolerance relative to each row's own bound magnitude);
//  2. solve the equality-constrained KKT system
//        [P+delta*I   A_act'] [x]   [-q]
//        [A_act       0     ] [y] = [ b_act]
//     once with a sparse LU (delta=1e-6 regularization, same trick as OSQP's
//     polish, needed because u/g/t variables have zero P diagonal);
//  3. irIters rounds of iterative refinement on that same factorization
//     (the ProxQP nb_iterative_refinement idea);
//  4. verify: EVERY original row feasible (l-tol <= Ax <= u+tol) and
//     min-form objective not worse than the input iterate; otherwise the
//     input x is left untouched and false is returned (polish is a free
//     option: worst case = return the ADMM solution unchanged).
bool polishKKT(Problem &p, std::vector<double> &x, int irIters,
               const std::vector<double> *y = nullptr,
               std::vector<double> *yOut = nullptr);
// Internal dual-gap certificate (admm_st2 S2-3, Moehle-Boyd §4 idea adapted):
//   prim - dual with dual(y) = -sum_{Pjj>0} g_j^2/(2 Pjj) + sum_{Pjj=0} g_j x_j
//                                - sum_r [y_r>0 ? y_r ub_r : y_r lb_r],
//   g = q + A'y. Exact at KKT points (complementarity + stationarity); the
// P-zero-coordinate term is evaluated at x, making the value trustworthy to
// ||g_inf||*||x_inf|| on those coords (LU-tight after polish). Rows whose
// active-side bound is +-inf contribute nothing (wrong-sign multiplier
// leakage -> certificate over-estimates the gap, never under-estimates).
// Returns prim(x) - dual(y); >= 0 up to the residual caveat above.
double dualGapCert(const Problem &p, const std::vector<double> &x,
                   const std::vector<double> &y);
double maxObjective(const InputsCols &cols, const Problem &p,
                    const std::vector<double> &x);  // evaluate max-form objective
