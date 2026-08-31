// osqp_engine.cpp - OSQP v1.0 C API backend (algorithmic twin of the Go ADMM)
#include "engine.hpp"
#include <osqp/osqp.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdlib>
#include <vector>

#ifdef USE_OSQP
bool osqpAvailable() { return true; }

EngineResult osqpSolve(const Problem &p, double epsAbs, double epsRel, bool verbose,
                       bool osqpScaledTerm, int libScaling, const EngineTuning &tune) {
    int n = p.n, m = (int)p.rows.size();
    // CSC of A (counting sort by column)
    size_t nnzA = 0;
    for (auto &r : p.rows) nnzA += r.idx.size();
    std::vector<OSQPInt> Ap(m + 1, 0);
    // count per column
    std::vector<OSQPInt> cnt(n, 0);
    for (auto &r : p.rows)
        for (auto j : r.idx) cnt[j]++;
    // rows are stored as (rowIdx r, col j, val v): counting sort into CSC
    std::vector<OSQPFloat> Ax(nnzA);
    std::vector<OSQPInt> Ai(nnzA);
    {
        std::vector<OSQPInt> colStart(n + 1, 0);
        for (int j = 0; j < n; j++) colStart[j + 1] = colStart[j] + cnt[j];
        std::vector<OSQPInt> fill(n, 0);
        for (int r = 0; r < m; r++)
            for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
                int j = p.rows[r].idx[t];
                OSQPInt pos = colStart[j] + fill[j]++;
                Ai[pos] = r;
                Ax[pos] = p.rows[r].val[t];
            }
        Ap = colStart;
    }
    std::vector<OSQPFloat> l(m), u(m);
    for (int r = 0; r < m; r++) {
        l[r] = p.rows[r].lb;
        u[r] = p.rows[r].ub;
    }
    // P upper-triangular CSC diagonal
    std::vector<OSQPInt> Pp(n + 1), Pi(n);
    std::vector<OSQPFloat> Px(n);
    OSQPInt pn = 0;
    for (int j = 0; j < n; j++) {
        Pp[j] = pn;
        if (p.pDiag[j] != 0) {
            Pi[pn] = j;
            Px[pn] = p.pDiag[j];
            pn++;
        }
    }
    Pp[n] = pn;
    OSQPCscMatrix Pm, Am;
    Pm.m = n; Pm.n = n; Pm.nzmax = pn; Pm.p = Pp.data(); Pm.i = Pi.data();
    Pm.x = Px.data(); Pm.nz = -1; Pm.owned = 0;
    Am.m = m; Am.n = n; Am.nzmax = (OSQPInt)nnzA; Am.p = Ap.data();
    Am.i = Ai.data(); Am.x = Ax.data(); Am.nz = -1; Am.owned = 0;
    std::vector<OSQPFloat> q(p.q.begin(), p.q.end());

    OSQPSettings *settings = OSQPSettings_new();
    settings->eps_abs = epsAbs;
    settings->eps_rel = epsRel;
    settings->verbose = verbose;
    settings->polishing = 1;
    settings->check_termination = 25;
    settings->max_iter = 1000000;
    settings->check_dualgap = 0; // gap is nan when dual_obj is -inf; block on criteria
    settings->scaled_termination = osqpScaledTerm ? 1 : 0;
    settings->scaling = (libScaling == 2) ? 0 : 10; // default: 10 (Ruiz)
    if (tune.rho >= 0) settings->rho = tune.rho;
    if (tune.sigma >= 0) settings->sigma = tune.sigma;
    if (tune.alpha >= 0) settings->alpha = tune.alpha;
    if (tune.adaptiveRho >= 0) settings->adaptive_rho = tune.adaptiveRho;
    if (tune.osqpPolish >= 0) settings->polishing = tune.osqpPolish;
    if (tune.maxIter > 0) settings->max_iter = tune.maxIter;

    OSQPSolver *solver = nullptr;
    double setupMsLoc = -1;
    auto ts0 = std::chrono::steady_clock::now();
    OSQPInt exitflag = osqp_setup(&solver, &Pm, q.data(), &Am, l.data(), u.data(),
                                  m, n, settings);
    auto ts1 = std::chrono::steady_clock::now();
    setupMsLoc =
        std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    EngineResult res;
    res.setupMs = setupMsLoc;
    if (exitflag != 0 || !solver) {
        res.status = "setup_error_" + std::to_string((long long)exitflag);
        OSQPSettings_free(settings);
        return res;
    }
    auto t0 = std::chrono::steady_clock::now();
    osqp_solve(solver);
    auto t1 = std::chrono::steady_clock::now();
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<OSQPFloat> xsol(n), ysol(m), pic(m), dic(n);
    OSQPSolution sol;
    sol.x = xsol.data();
    sol.y = ysol.data();
    sol.prim_inf_cert = pic.data();
    sol.dual_inf_cert = dic.data();
    if (osqp_get_solution(solver, &sol) == 0) {
        res.x.assign(sol.x, sol.x + n);
        res.y.assign(sol.y, sol.y + m);
    }

    OSQPInfo *info = solver->info;
    res.status = std::string(info->status);
    res.iters = (int)info->iter;
    res.priRes = info->prim_res;
    res.duaRes = info->dual_res;
    res.obj = info->obj_val;
    res.polishStatus = (int)info->status_polish;

    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    return res;
}
#else
bool osqpAvailable() { return false; }
EngineResult osqpSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &) {
    return EngineResult{"", "osqp not compiled in", 0, 0, 0, 0, 0};
}
#endif
