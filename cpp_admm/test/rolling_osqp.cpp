// rolling_osqp.cpp - admm_st2.md S2-0 experiment: warm-start rolling re-solves.
//
// Simulates intraday/next-day rebalances by jittering alpha (the only data
// that changes q; bands/P/A/l/u stay fixed), then measures:
//   cold  = fresh osqp_setup + solve for each variant
//   warm  = persistent solver: osqp_update_data_vec(q) + osqp_warm_start(x,y)
// All in scaled coordinates (--scale-rows semantics), default eps 1e-6/1e-8,
// lib scaling OFF (report_presolve.md best osqp config).
#include "../src/csv.hpp"
#include "../src/model.hpp"
#include <osqp/osqp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// CSC of A (same construction as osqp_engine.cpp)
static void makeCsc(const Problem &p, std::vector<OSQPInt> &Ap,
                    std::vector<OSQPInt> &Ai, std::vector<OSQPFloat> &Ax,
                    int &m, int &n) {
    m = (int)p.rows.size();
    n = p.n;
    size_t nnz = 0;
    for (auto &r : p.rows) nnz += r.idx.size();
    std::vector<OSQPInt> cnt(n, 0), colStart(n + 1, 0), fill(n, 0);
    for (auto &r : p.rows)
        for (int j : r.idx) cnt[j]++;
    for (int j = 0; j < n; j++) colStart[j + 1] = colStart[j] + cnt[j];
    Ax.resize(nnz);
    Ai.resize(nnz);
    for (int r = 0; r < m; r++)
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
            int j = p.rows[r].idx[t];
            OSQPInt pos = colStart[j] + fill[j]++;
            Ai[pos] = r;
            Ax[pos] = p.rows[r].val[t];
        }
    Ap = colStart;
}

static OSQPSettings *mkSettings() {
    OSQPSettings *s = OSQPSettings_new();
    s->eps_abs = 1e-6;
    s->eps_rel = 1e-8;
    s->verbose = 0;
    s->polishing = 1;
    s->check_termination = 25;
    s->max_iter = 1000000;
    s->check_dualgap = 0;
    s->scaling = 0; // manual scaleRows already applied
    s->adaptive_rho_interval = 25; // iteration-based (deterministic; default is time-based)
    return s;
}

int main(int argc, char **argv) {
    std::string inF, conF, wtF;
    double jitter = 0.05;
    int nseq = 8;
    unsigned seed = 42;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a.rfind("-i=", 0) == 0) inF = a.c_str() + 3;
        else if (a.rfind("-c=", 0) == 0) conF = a.c_str() + 3;
        else if (a.rfind("-w=", 0) == 0) wtF = a.c_str() + 3;
        else if (a.rfind("--jitter=", 0) == 0) jitter = atof(a.c_str() + 9);
        else if (a.rfind("--seq=", 0) == 0) nseq = atoi(a.c_str() + 6);
        else if (a.rfind("--seed=", 0) == 0) seed = (unsigned)atoi(a.c_str() + 7);
    }
    std::string err;
    auto inputs = loadInputs(inF, err);
    if (!err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto constraints = loadConstraints(conF, err);
    if (!err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto weights = loadWeights(wtF, err);
    if (!err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    step1Relax(inputs);
    InputsCols cols = toColumns(inputs);
    auto det = assembleConstraints(cols.univ, constraints, weights);

    Problem p = buildProblem(cols, det, false, 100.0);
    scaleRows(p); // scaled coordinates; bands depend on pos/bounds only

    int m, n;
    std::vector<OSQPInt> Ap, Ai;
    std::vector<OSQPFloat> Ax;
    makeCsc(p, Ap, Ai, Ax, m, n);
    std::vector<OSQPFloat> l(m), u(m);
    for (int r = 0; r < m; r++) { l[r] = p.rows[r].lb; u[r] = p.rows[r].ub; }
    std::vector<OSQPInt> Pp(n + 1), Pi(n);
    std::vector<OSQPFloat> Px(n);
    OSQPInt pn = 0;
    for (int j = 0; j < n; j++) {
        Pp[j] = pn;
        if (p.pDiag[j] != 0) { Pi[pn] = j; Px[pn] = p.pDiag[j]; pn++; }
    }
    Pp[n] = pn;
    OSQPCscMatrix Pm, Am;
    Pm.m = n; Pm.n = n; Pm.nzmax = pn; Pm.p = Pp.data(); Pm.i = Pi.data();
    Pm.x = Px.data(); Pm.nz = -1; Pm.owned = 0;
    Am.m = m; Am.n = n; Am.nzmax = (OSQPInt)Ai.size(); Am.p = Ap.data();
    Am.i = Ai.data(); Am.x = Ax.data(); Am.nz = -1; Am.owned = 0;

    // scaled q from perturbed alpha: q[a] = (lambda*pos - alpha) * band
    // (alpha is the only perturbed field; bands/P/A/l/u unchanged)
    auto scaledQ = [&](const InputsCols &c) {
        std::vector<OSQPFloat> q(n, 0.0f);
        for (int a = 0; a < (int)p.active.size(); a++) {
            int i = p.active[a];
            double b = p.varScale[a]; // x-var band
            q[a] = (OSQPFloat)((c.lambda[i] * c.position[i] - c.alpha[i]) * b);
        }
        return q;
    };
    std::vector<OSQPFloat> q0 = scaledQ(cols);

    OSQPSettings *settings = mkSettings();

    // ---------- cold sequence ----------
    printf("jitter=%.0f%% seq=%d seed=%u\n", jitter * 100, nseq, seed);
    double coldMs = 0, coldSolveMs = 0;
    printf("%-10s %-14s %-10s %-12s\n", "variant", "cold iters", "cold ms", "(solve-only ms)");
    std::vector<std::vector<double>> xs; // cold solutions for later warm runs
    for (int k = 0; k <= nseq; k++) {
        InputsCols cc = cols;
        if (k > 0) {
            std::mt19937 rng(seed + k);
            std::uniform_real_distribution<double> d(-jitter, jitter);
            for (auto &a : cc.alpha) a *= (1.0 + d(rng));
        }
        std::vector<OSQPFloat> qk = scaledQ(cc);
        auto t0 = std::chrono::steady_clock::now();
        OSQPSolver *solver = nullptr;
        OSQPInt rc = osqp_setup(&solver, &Pm, qk.data(), &Am, l.data(), u.data(),
                                m, n, settings);
        auto ts0 = std::chrono::steady_clock::now();
        osqp_solve(solver);
        auto t1 = std::chrono::steady_clock::now();
        double msAll = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double msSolve = std::chrono::duration<double, std::milli>(t1 - ts0).count();
        coldMs += msAll;
        coldSolveMs += msSolve;
        printf("%-10d %-14d %-10.1f %-12.1f\n", k, (int)solver->info->iter, msAll, msSolve);
        xs.push_back(std::vector<double>(solver->solution->x, solver->solution->x + n));
        osqp_cleanup(solver);
    }
    printf("COLD total %.1f ms (solve-only %.1f)\n\n", coldMs, coldSolveMs);

    // ---------- warm sequence ----------
    double warmMs = 0;
    printf("%-10s %-14s %-10s\n", "variant", "warm iters", "warm ms");
    OSQPSolver *solver = nullptr;
    OSQPInt rc = osqp_setup(&solver, &Pm, q0.data(), &Am, l.data(), u.data(),
                            m, n, settings);
    if (rc != 0) { fprintf(stderr, "setup failed\n"); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    osqp_solve(solver);
    auto t1 = std::chrono::steady_clock::now();
    double msBase = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("%-10d %-14d %-10.1f  (base, cold)\n", 0, (int)solver->info->iter, msBase);
    warmMs += msBase;
    for (int k = 1; k <= nseq; k++) {
        InputsCols cc = cols;
        std::mt19937 rng(seed + k);
        std::uniform_real_distribution<double> d(-jitter, jitter);
        for (auto &a : cc.alpha) a *= (1.0 + d(rng));
        std::vector<OSQPFloat> qk = scaledQ(cc);
        osqp_update_data_vec(solver, qk.data(), nullptr, nullptr);
        osqp_warm_start(solver, solver->solution->x, solver->solution->y);
        auto tk0 = std::chrono::steady_clock::now();
        osqp_solve(solver);
        auto tk1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(tk1 - tk0).count();
        warmMs += ms;
        printf("%-10d %-14d %-10.1f\n", k, (int)solver->info->iter, ms);
        // sanity: warm solution close to the cold solution of the same variant
        double md = 0;
        for (int j = 0; j < n; j++)
            md = std::max(md, std::fabs(solver->solution->x[j] - xs[k][j]));
        if (k == 1) printf("           (warm vs cold variant-1 max|x diff| = %.3e)\n", md);
    }
    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    printf("WARM total %.1f ms (base + %d updates)  speedup vs cold: %.2fx\n",
           warmMs, nseq, coldMs / warmMs);
    return 0;
}