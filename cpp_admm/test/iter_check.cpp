// iter_check.cpp - single-iteration forensic: explicit dense-KKT OSQP v1.0
// step (following /tmp/cppadmm/osqp/src/auxil.c literally) vs the native
// engine's Woodbury formulas. Prints where they diverge.
#include "../src/csv.hpp"
#include "../src/model.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <cstdio>
#include <cmath>
#include <vector>

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

    // ---- dense KKT: [[P + sigma I, A'],[A, -rho^-1 I]] ----
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

    std::vector<double> x(n, 0.0), z(m, 0.0), y(m, 0.0);
    Eigen::VectorXd rhs(n + m), sol;
    for (int j = 0; j < n; j++) rhs[j] = sigma * x[j] - p.q[j];
    for (int r = 0; r < m; r++) rhs[n + r] = y[r] / rho - z[r];
    sol = lu.solve(rhs);
    std::vector<double> xt(n), zt(m);
    for (int j = 0; j < n; j++) xt[j] = sol[j];
    for (int r = 0; r < m; r++) zt[r] = sol[n + r];

    // ---- native formulas for the same step ----
    // x~: (P+sigma I+rho A'A) x~ = sigma x - q + A'(y - rho z)
    std::vector<double> rhs2(n);
    for (int j = 0; j < n; j++) rhs2[j] = sigma * x[j] - p.q[j];
    for (int r = 0; r < m; r++) {
        double d = y[r] - rho * z[r];
        if (d == 0) continue;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            rhs2[p.rows[r].idx[t]] += d * p.rows[r].val[t];
    }
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (int j = 0; j < n; j++) M(j, j) = p.pDiag[j] + sigma;
    for (int r = 0; r < m; r++)
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
            int j = p.rows[r].idx[t];
            double v = p.rows[r].val[t];
            M(j, j) += rho * v * v; // WRONG for cross terms - need full A'A
        }
    // full A'A:
    M = Eigen::MatrixXd::Zero(n, n);
    for (int j = 0; j < n; j++) M(j, j) = p.pDiag[j] + sigma;
    for (int r = 0; r < m; r++)
        for (size_t a = 0; a < p.rows[r].idx.size(); a++)
            for (size_t b = 0; b < p.rows[r].idx.size(); b++)
                M(p.rows[r].idx[a], p.rows[r].idx[b]) +=
                    rho * p.rows[r].val[a] * p.rows[r].val[b];
    Eigen::VectorXd xt2 = M.partialPivLu().solve(
        Eigen::Map<Eigen::VectorXd>(rhs2.data(), n));
    std::vector<double> zt2(m);
    for (int r = 0; r < m; r++) {
        double s = 0;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            s += p.rows[r].val[t] * xt2[p.rows[r].idx[t]];
        zt2[r] = rho * s - y[r] + rho * z[r];
    }
    double dxt = 0, dzt = 0;
    for (int j = 0; j < n; j++) dxt = std::max(dxt, std::fabs(xt[j] - xt2[j]));
    for (int r = 0; r < m; r++) dzt = std::max(dzt, std::fabs(zt[r] - zt2[r]));
    printf("iter-1  max|KKT x~ - native x~| = %.3e\n", dxt);
    printf("iter-1  max|KKT z~ - native z~| = %.3e\n", dzt);
    // also: is KKT z~ == A x~ (identity check)
    double zt_vs_ax = 0;
    for (int r = 0; r < m; r++) {
        double s = 0;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            s += p.rows[r].val[t] * xt[p.rows[r].idx[t]];
        zt_vs_ax = std::max(zt_vs_ax, std::fabs(zt[r] - s));
    }
    printf("iter-1  max|KKT z~ - A*KKT x~|  = %.3e\n", zt_vs_ax);
    return 0;
}