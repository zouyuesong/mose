// iter_check2.cpp - definitive: step OSQP library 1 iteration at a time
// (max_iter=1 + warm start) and compare its (x,z,y) trajectory against the
// manual formulas (explicit dense KKT per auxil.c semantics).
#include "../src/csv.hpp"
#include "../src/model.hpp"
#include <Eigen/Dense>
#include <osqp/osqp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

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
    int n = p.n, m = (int)p.rows.size();
    double rho = 0.1, sigma = 1e-6, alpha = 1.6, beta = 1 - alpha;

    // ---------- manual loop state ----------
    std::vector<double> x(n, 0.0), z(m, 0.0), y(m, 0.0);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n + m, n + m);
    for (int j = 0; j < n; j++) K(j, j) = p.pDiag[j] + sigma;
    for (int r = 0; r < m; r++) {
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
            int j = p.rows[r].idx[t];
            double v = p.rows[r].val[t];
            K(n + r, j) = v;
            K(j, n + r) = v;
        }
        K(n + r, n + r) = -1.0 / rho;
    }
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(K);

    // ---------- OSQP solver ----------
    std::vector<OSQPInt> Ap, Ai;
    std::vector<OSQPFloat> Axv;
    int mm, nn;
    makeCsc(p, Ap, Ai, Axv, mm, nn);
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
    Am.i = Ai.data(); Am.x = Axv.data(); Am.nz = -1; Am.owned = 0;
    std::vector<OSQPFloat> q(p.q.begin(), p.q.end());
    OSQPSettings *st = OSQPSettings_new();
    st->eps_abs = 1e-6;
    st->eps_rel = 1e-8;
    st->verbose = 0;
    st->polishing = 0;
    st->scaling = 0;
    st->adaptive_rho = 0;     // FIXED rho for the comparison
    st->max_iter = 1;
    st->check_termination = 1000000; // never: keep pure 1-iter steps
    st->warm_starting = 1;
    OSQPSolver *solver = nullptr;
    osqp_setup(&solver, &Pm, q.data(), &Am, l.data(), u.data(), m, n, st);

    for (int step = 1; step <= 5; step++) {
        // manual iteration (formulas = native engine)
        std::vector<double> rhs(n), zt(m);
        for (int j = 0; j < n; j++) rhs[j] = sigma * x[j] - p.q[j];
        for (int r = 0; r < m; r++) {
            double d = rho * z[r] - y[r];
            if (d == 0) continue;
            for (size_t t = 0; t < p.rows[r].idx.size(); t++)
                rhs[p.rows[r].idx[t]] += d * p.rows[r].val[t];
        }
        Eigen::VectorXd xt = lu.solve(Eigen::Map<Eigen::VectorXd>(rhs.data(), n) .eval());
        // NOTE: solve with the x-part rhs only is NOT the KKT solve; do full:
        Eigen::VectorXd full(n + m);
        for (int j = 0; j < n; j++) full[j] = sigma * x[j] - p.q[j];
        for (int r = 0; r < m; r++) full[n + r] = z[r] - y[r] / rho;
        Eigen::VectorXd sol = lu.solve(full);
        for (int j = 0; j < n; j++) xt[j] = sol[j];
        for (int r = 0; r < m; r++) {
            double s = 0;
            for (size_t t = 0; t < p.rows[r].idx.size(); t++)
                s += p.rows[r].val[t] * xt[p.rows[r].idx[t]];
            zt[r] = rho * s + y[r] - rho * z[r];
        }
        for (int r = 0; r < m; r++) {
            double tz = y[r] / rho + alpha * zt[r] + beta * z[r];
            if (tz < p.rows[r].lb) tz = p.rows[r].lb;
            else if (tz > p.rows[r].ub) tz = p.rows[r].ub;
            y[r] += rho * (alpha * zt[r] + beta * z[r] - tz);
            z[r] = tz;
        }
        for (int j = 0; j < n; j++) x[j] = alpha * xt[j] + beta * x[j];

        // OSQP one iteration
        osqp_solve(solver);
        const OSQPFloat *ox = solver->solution->x;
        const OSQPFloat *oy = solver->solution->y;
        // OSQP z is not exposed directly; compare x and y
        double dx = 0, dy = 0;
        for (int j = 0; j < n; j++) dx = std::fmax(dx, std::fabs(x[j] - ox[j]));
        for (int r = 0; r < m; r++) dy = std::fmax(dy, std::fabs(y[r] - oy[r]));
        printf("step %d: max|x-manual - x-osqp| = %.3e   max|y-manual - y-osqp| = %.3e"
               "   (x1 %.6g vs %.6g)\n", step, dx, dy, x[0], ox[0]);
    }
    osqp_cleanup(solver);
    OSQPSettings_free(st);
    return 0;
}