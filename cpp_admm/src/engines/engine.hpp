// engine.hpp - unified solver engine interface
#pragma once
#include "../model.hpp"
#include <string>
#include <vector>

struct EngineResult {
    std::vector<double> x;     // reduced-space solution (len = p.n)
    std::vector<double> y;     // row multipliers (len = m) when available
    std::string status;        // library-specific status string
    int iters = 0;
    double solveMs = 0;        // solve() wall time in ms
    double setupMs = -1;      // library setup (factorization/preconditioner):
                              // reported only by lib engines; native engines
                              // fold everything into solveMs
    double priRes = 0, duaRes = 0;
    double obj = 0;            // min-form objective reported by library (if any)
    int polishStatus = 0;      // OSQP built-in polish: 1 ok / 0 not run / -1 failed
    int harnessPolish = -1;    // harness --polish: 1 ok / 0 failed-reverted / -1 not run
    double polishMs = 0;       // harness polish wall time (included in solveMs)
};

// Per-engine tuning knobs for the admm.md §3.4(b)/(c) experiments
// (adaptive rho & residual balancing; over-relaxation / acceleration).
// Negative / unset fields keep each library's default.
// Mapping:
//   osqp:   rho->rho, sigma->sigma, alpha->alpha, adaptiveRho->adaptive_rho
//   qpalm:  sigma->sigma_init (proximal penalty init)
//   proxqp: rho->default_rho, sigma->default_mu_in, alpha->alpha_bcl
//   scs:    rho->rho_x, alpha->alpha, sigma->scale
struct EngineTuning {
    double rho = -1;
    double sigma = -1;
    double alpha = -1;
    int adaptiveRho = -1;  // 0 off, 1 on, -1 default
    int osqpPolish = -1;   // 0 off, 1 on, -1 default (built-in polishing)
    int maxIter = -1;      // cap solver iterations (forced mid-accuracy stop)
    int rhoInterval = -1;  // native engine: adaptive-rho period (default 500)
};

// Build & solve with the named engine ("osqp"|"qpalm"|"scs"|"proxqp").
// epsAbs/epsRel follow the Go defaults (1e-6 / 1e-8).
// osqpScaledTerm: OSQP-only, sets scaled_termination=1 (residuals measured on
// the internally-equilibrated problem -> global eps_rel becomes roughly
// per-row relative, comparable to manual scaleRows()).
// libScaling: 0 = library default, 1 = force built-in scaling ON,
//             2 = force built-in scaling OFF (osqp scaling=0, qpalm scaling=0,
//             scs normalize=0, proxqp compute_preconditioner=false).
EngineResult engineSolve(const std::string &engine, const Problem &p,
                         double epsAbs, double epsRel, bool verbose,
                         bool osqpScaledTerm = false, int libScaling = 0,
                         EngineTuning tune = EngineTuning{});

bool engineAvailable(const std::string &engine);
