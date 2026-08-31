// nqp_check.cpp - forensic: nativeqpalm vs QPALM library on the real problem,
// first-K-outer-steps trajectory compare (dphi norm, tau, active count).
// Runs both with identical settings (sigma_init=1, proximal on) and dumps
// the library's per-iteration stats via its verbose mode vs ours.
#include "../src/csv.hpp"
#include "../src/model.hpp"
#include "../src/engines/engine.hpp"
#include <cstdio>
#include <vector>

// declared in engines.cpp (both variants)
EngineResult nativeqpalmSolve(const Problem &, double, double, bool,
                              const EngineTuning &);
EngineResult engineSolve(const std::string &, const Problem &, double, double,
                         bool, bool, int, EngineTuning);

int main(int argc, char **argv) {
    std::string inF = argv[1], conF = argv[2], wtF = argv[3];
    std::string err;
    auto inputs = loadInputs(inF, err);
    auto constraints = loadConstraints(conF, err);
    auto weights = loadWeights(wtF, err);
    step1Relax(inputs);
    InputsCols cols = toColumns(inputs);
    auto det = assembleConstraints(cols.univ, constraints, weights);
    Problem p = buildProblem(cols, det, false, 100.0);
    scaleRows(p);

    EngineTuning tune;
    tune.maxIter = 200; // cap: we only need the early trajectory
    printf("== nativeqpalm (sigma=1, capped 200) ==\n");
    auto r1 = nativeqpalmSolve(p, 1e-6, 1e-8, false, tune);
    printf("native: status=%s iter=%d obj=%.6f solveMs=%.1f\n",
           r1.status.c_str(), r1.iters, r1.obj, r1.solveMs);

    printf("\n== qpalm library (scaling=0, sigma_init=1) ==\n");
    EngineTuning t2;
    t2.sigma = 1; // maps to sigma_init
    auto r2 = engineSolve("qpalm", p, 1e-6, 1e-8, true, false, 2 /*off*/, t2);
    printf("library: status=%s iter=%d obj=%.6f solveMs=%.1f\n",
           r2.status.c_str(), r2.iters, r2.obj, r2.solveMs);
    return 0;
}