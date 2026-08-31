// proxqp_engine.cpp - ProxQP (proxsuite v0.6.4) sparse backend, C++ API
#include "engine.hpp"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <Eigen/Sparse>
#include <proxsuite/proxqp/sparse/wrapper.hpp>

#ifdef USE_PROXQP
bool proxqpAvailable() { return true; }

using namespace proxsuite::proxqp;

EngineResult proxqpSolve(const Problem &p, double epsAbs, double epsRel, bool verbose,
                         bool, int libScaling, const EngineTuning &tune) {
    int n = p.n, m = (int)p.rows.size();
    std::vector<Eigen::Triplet<double>> atrip, ptrip;
    for (int r = 0; r < m; r++)
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            atrip.push_back({r, p.rows[r].idx[t], p.rows[r].val[t]});
    for (int j = 0; j < n; j++)
        if (p.pDiag[j] != 0) ptrip.push_back({j, j, p.pDiag[j]});
    Eigen::SparseMatrix<double, Eigen::ColMajor> C(m, n), H(n, n);
    C.setFromTriplets(atrip.begin(), atrip.end());
    H.setFromTriplets(ptrip.begin(), ptrip.end());
    Eigen::Matrix<double, -1, 1> g =
        Eigen::Map<const Eigen::Matrix<double, -1, 1>>(p.q.data(), n);
    Eigen::Matrix<double, -1, 1> l(m), u(m);
    for (int r = 0; r < m; r++) {
        l[r] = p.rows[r].lb;
        u[r] = p.rows[r].ub;
    }

    sparse::QP<double, int> qp(n, 0, m); // no equality block
    qp.settings.eps_abs = epsAbs;
    qp.settings.eps_rel = epsRel;
    qp.settings.verbose = verbose;
    qp.settings.max_iter = 100000;
    qp.settings.max_iter_in = 1000;
    if (tune.rho >= 0) qp.settings.default_rho = tune.rho;
    if (tune.sigma >= 0) qp.settings.default_mu_in = tune.sigma;
    if (tune.alpha >= 0) qp.settings.alpha_bcl = tune.alpha;
    double setupMsLoc = -1;
    auto ts0 = std::chrono::steady_clock::now();
    qp.init(H, g, proxsuite::nullopt, proxsuite::nullopt, C, l, u,
            /*compute_preconditioner=*/ libScaling != 2);
    auto ts1 = std::chrono::steady_clock::now();
    setupMsLoc =
        std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    auto t0 = std::chrono::steady_clock::now();
    qp.solve();
    auto t1 = std::chrono::steady_clock::now();

    EngineResult res;
    res.setupMs = setupMsLoc;
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x.assign(qp.results.x.data(), qp.results.x.data() + qp.results.x.size());
    res.y.assign(qp.results.z.data(), qp.results.z.data() + qp.results.z.size()); // B1 FIX: z = ineq multipliers (y is eq, len 0 here)

    res.iters = (int)qp.results.info.iter;
    res.priRes = qp.results.info.pri_res;
    res.duaRes = qp.results.info.dua_res;
    res.obj = qp.results.info.objValue;
    switch (qp.results.info.status) {
    case QPSolverOutput::PROXQP_SOLVED: res.status = "solved"; break;
    case QPSolverOutput::PROXQP_MAX_ITER_REACHED: res.status = "max_iter"; break;
    case QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE: res.status = "primal_infeasible"; break;
    case QPSolverOutput::PROXQP_DUAL_INFEASIBLE: res.status = "dual_infeasible"; break;
    default: res.status = "other"; break;
    }
    return res;
}
#else
bool proxqpAvailable() { return false; }
EngineResult proxqpSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &) {
    return EngineResult{"", "proxqp not compiled in", 0, 0, 0, 0, 0};
}
#endif
