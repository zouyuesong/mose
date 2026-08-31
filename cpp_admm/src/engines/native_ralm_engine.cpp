// native_ralm_engine.cpp - admm_st2 S2 fourth native family: NR-LALM
// (arXiv 2608.19847, "A Fixed-Penalty Linearized Augmented Lagrangian Method
// with Classical Multiplier Updates") ported to our convex QP
//   min 1/2 x'Px + q'x  s.t.  l <= Ax <= u.
//
// Paper fidelity notes (see admm_st2.md S2 notes):
// - The paper handles NONLINEAR EQUALITY constraints only. For linear
//   constraints the constraint-linearization error d_k vanishes identically
//   (their central analysis collapses), leaving the single-loop iteration:
//     (beta*I + rho*A_act'A_act) p = -(grad f + A'(y + rho*c))
//     x+ = x + p;   y+ = y + rho*c(x+)     [classical multiplier update]
//   with FIXED, accuracy-independent (rho, beta).
// - Inequalities are "open" per the paper's conclusion; our adaptation uses
//   the Ax = z splitting with exact box projection:
//     z+ = clip(y/rho + Ax+, l, u)   (proximal projection; without y/rho the
//                                     multiplier could never retreat to 0)
//     y+ = y + rho*(Ax+ - z+)        (= rho*(v - clip(v)), the classical
//                                     residual update on the new iterate)
// - P is NOT part of the system matrix (Gauss-Newton proximal touch: M =
//   beta*I + rho*A'A). OSQP keeps P in M and solves the x-subproblem exactly;
//   NR-LALM trades that for a factorization that does not depend on P at all
//   and stays fixed for the whole run (ONE Woodbury build).
//
// Three pieces harvested from the paper beyond the loop:
//   PhiHat monitor:  Phi_hat_k = L_rho(x,y) + rho*||r||^2
//                    = f(x) + y'r + 1.5*rho*||r||^2, r = Ax - z
//     (Theorem 2.9 Lyapunov function with the constant-free memory term
//      rho*||c||^2 replacing (C_lam,p/rho)||p_{k-1}||^2; drops when the
//      iteration makes progress)
//   argmin output:   return the iterate with the best normalized KKT
//                    residual max(pri/epsP, dua/epsD), not the last one
//                    (Theorem 2.12 output rule; rescues loose-tolerance runs
//                    whose tail drifts after passing near the solution)
//   rho/beta ratio:  paper practice rho/beta in [6, 64]; expose both knobs.
//
// Linear algebra: identical structure win as native_engine.cpp - structure
// rows contribute DIAGONAL to A'A (bilateral +- pairs cancel cross terms),
// general rows (<=67) form the rank-k Woodbury term. Fixed (rho,beta) means
// ONE factorization for the entire solve.
#include "engine.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

bool nativeralmAvailable() { return true; }

namespace {

// Variant switch NATIVERALM_PIN_P=1: fold P into M (= P + beta*I + rho*A'A)
// to isolate the effect of the Gauss-Newton proximal touch (P outside M)
// from the loop shape in the comparison against OSQP-style native.
static bool pinP = getenv("NATIVERALM_PIN_P") != nullptr;

struct WoodburyNR {
    int n = 0, k = 0;
    double rho = 0;
    std::vector<double> dtilde, invd;
    std::vector<int> gPtr;
    std::vector<int> gIdx;
    std::vector<double> gVal;
    std::vector<double> chol;
    std::vector<double> bufA, bufV, bufGt;

    // M = beta*I + rho*A'A  (NO pDiag - differs from native_engine's Woodbury;
    // pinP variant adds pDiag, see the switch above)
    bool init(const Problem &p, double rho_, double beta) {
        n = p.n;
        rho = rho_;
        int m = (int)p.rows.size();
        int gs = p.generalStart >= 0 ? p.generalStart : m;
        std::vector<double> ad(n, 0.0);
        for (int r = 0; r < gs; r++)
            for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
                double v = p.rows[r].val[t];
                ad[p.rows[r].idx[t]] += v * v;
            }
        dtilde.assign(n, 0.0);
        invd.assign(n, 0.0);
        for (int j = 0; j < n; j++) {
            dtilde[j] = beta + rho * ad[j] + (pinP ? p.pDiag[j] : 0.0);
            if (!(dtilde[j] > 0)) dtilde[j] = 1e-12;
            invd[j] = 1.0 / dtilde[j];
        }
        k = m - gs;
        gPtr.assign(k + 1, 0);
        int nnz = 0;
        for (int i = 0; i < k; i++) nnz += (int)p.rows[gs + i].idx.size();
        gIdx.resize(nnz);
        gVal.resize(nnz);
        {
            int c = 0;
            for (int i = 0; i < k; i++) {
                gPtr[i] = c;
                for (size_t t = 0; t < p.rows[gs + i].idx.size(); t++) {
                    gIdx[c] = p.rows[gs + i].idx[t];
                    gVal[c] = p.rows[gs + i].val[t];
                    c++;
                }
            }
            gPtr[k] = c;
        }
        bufA.assign(n, 0.0);
        bufV.assign(k, 0.0);
        bufGt.assign(n, 0.0);
        if (k == 0) return true;
        std::vector<double> S((size_t)k * k, 0.0);
        for (int i = 0; i < k; i++)
            for (int j = i; j < k; j++) {
                double s = 0;
                int ia = gPtr[i], ib = gPtr[j], ea = gPtr[i + 1], eb = gPtr[j + 1];
                while (ia < ea && ib < eb) {
                    if (gIdx[ia] == gIdx[ib]) {
                        s += gVal[ia] * gVal[ib] * invd[gIdx[ia]];
                        ia++;
                        ib++;
                    } else if (gIdx[ia] < gIdx[ib]) ia++;
                    else ib++;
                }
                S[(size_t)i * k + j] = s;
                if (j != i) S[(size_t)j * k + i] = s;
            }
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                S[(size_t)i * k + j] =
                    (i == j ? 1.0 : 0.0) + rho * S[(size_t)i * k + j];
        for (int i = 0; i < k; i++)
            for (int j = 0; j <= i; j++) {
                double s = S[(size_t)i * k + j];
                for (int l = 0; l < j; l++)
                    s -= S[(size_t)i * k + l] * S[(size_t)j * k + l];
                if (i == j) {
                    if (s <= 0) return false;
                    S[(size_t)i * k + i] = std::sqrt(s);
                } else {
                    S[(size_t)i * k + j] = s / S[(size_t)j * k + j];
                }
            }
        chol = std::move(S);
        return true;
    }

    void solve(double *z, const double *r) {
        if (k == 0) {
            for (int j = 0; j < n; j++) z[j] = r[j] * invd[j];
            return;
        }
        double *a = bufA.data();
        for (int j = 0; j < n; j++) a[j] = r[j] * invd[j];
        double *v = bufV.data();
        for (int i = 0; i < k; i++) {
            int len = gPtr[i + 1] - gPtr[i];
            if (len <= 0) { v[i] = 0; continue; }
            int j0 = gIdx[gPtr[i]];
            const double *vv = gVal.data() + gPtr[i];
            const double *aa = a + j0;
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int t = 0;
            for (; t + 4 <= len; t += 4) {
                s0 += vv[t + 0] * aa[t + 0];
                s1 += vv[t + 1] * aa[t + 1];
                s2 += vv[t + 2] * aa[t + 2];
                s3 += vv[t + 3] * aa[t + 3];
            }
            double s = (s0 + s1) + (s2 + s3);
            for (; t < len; t++) s += vv[t] * aa[t];
            v[i] = s;
        }
        for (int i = 0; i < k; i++) {
            double s = v[i];
            const double *ci = &chol[(size_t)i * k];
            for (int l = 0; l < i; l++) s -= ci[l] * v[l];
            v[i] = s / ci[i];
        }
        for (int i = k - 1; i >= 0; i--) {
            double xi = v[i] / chol[(size_t)i * k + i];
            v[i] = xi;
            if (xi != 0) {
                const double *ci = &chol[(size_t)i * k];
                for (int j = 0; j < i; j++) v[j] -= ci[j] * xi;
            }
        }
        double *gt = bufGt.data();
        for (int j = 0; j < n; j++) gt[j] = 0;
        for (int i = 0; i < k; i++) {
            double si = v[i];
            if (si == 0) continue;
            int len = gPtr[i + 1] - gPtr[i];
            if (len <= 0) continue;
            int j0 = gIdx[gPtr[i]];
            double *gg = gt + j0;
            const double *vv = gVal.data() + gPtr[i];
            for (int t = 0; t < len; t++) gg[t] += si * vv[t];
        }
        for (int j = 0; j < n; j++) z[j] = a[j] - rho * gt[j] * invd[j];
    }
};

} // namespace

EngineResult nativeralmSolve(const Problem &p, double epsAbs, double epsRel,
                             bool verbose, const EngineTuning &tune) {
    int n = p.n, m = (int)p.rows.size();
    double rho = tune.rho >= 0 ? tune.rho : 10.0;   // paper practice rho/beta 6..64
    double beta = tune.sigma >= 0 ? tune.sigma : 1.0;
    int maxIter = tune.maxIter > 0 ? tune.maxIter : 1000000;
    int checkEvery = 25;

    EngineResult res;
    WoodburyNR w;
    if (!w.init(p, rho, beta)) {
        res.status = "setup_error";
        return res;
    }

    std::vector<double> x(n, 0.0), xt(n), z(m, 0.0), y(m, 0.0);
    std::vector<double> rhs(n), Ax(m), wy(m), aty(n);
    std::vector<double> L(m), U(m);
    int gs = p.generalStart >= 0 ? p.generalStart : m;
    std::vector<int> bJ;      std::vector<double> bV;
    std::vector<int> pJ1, pJ2; std::vector<double> pV1, pV2;
    std::vector<int> pRowOf, bRowOf;
    std::vector<int> gCol0; std::vector<int> gLen; std::vector<double> gValFlat;
    {
        for (int r = 0; r < m; r++) { L[r] = p.rows[r].lb; U[r] = p.rows[r].ub; }
        for (int r = 0; r < gs; r++) {
            const Row &rw = p.rows[r];
            if (rw.idx.size() == 1) {
                bRowOf.push_back(r); bJ.push_back(rw.idx[0]); bV.push_back(rw.val[0]);
            } else if (rw.idx.size() == 2) {
                pRowOf.push_back(r);
                pJ1.push_back(rw.idx[0]); pV1.push_back(rw.val[0]);
                pJ2.push_back(rw.idx[1]); pV2.push_back(rw.val[1]);
            }
        }
        int nnzG = 0;
        for (int r = gs; r < m; r++) nnzG += (int)p.rows[r].idx.size();
        gValFlat.reserve(nnzG);
        for (int r = gs; r < m; r++) {
            const Row &rw = p.rows[r];
            gCol0.push_back(rw.idx.empty() ? 0 : rw.idx[0]);
            gLen.push_back((int)rw.idx.size());
            for (size_t t = 0; t < rw.idx.size(); t++)
                gValFlat.push_back(rw.val[t]);
        }
    }
    int nBox = (int)bJ.size(), nPair = (int)pJ1.size();
    int nGen = m - gs;
    const int *gCol0p = gCol0.data();
    const int *gLenp = gLen.data();
    const double *gVp = gValFlat.data();

    auto t0 = std::chrono::steady_clock::now();
    static bool dbg = getenv("NATIVERALM_DEBUG") != nullptr;
    double phiHat = 0, phiHatMin = 1e300;
    std::vector<double> xBest(n, 0.0), yBest(m, 0.0);
    double bestRatio = 1e300;
    int bestIt = 0;
    int it = 0;
    for (it = 1; it <= maxIter; it++) {
        // rhs = (beta - Pdiag)*x - q + A'(rho*z - y); with P folded into M
        // (pinP) the proximal-ADMM rhs drops the -Pdiag*x term
        for (int j = 0; j < n; j++)
            rhs[j] = (beta - (pinP ? 0.0 : p.pDiag[j])) * x[j] - p.q[j];
        for (int r = 0; r < m; r++) wy[r] = rho * z[r] - y[r];
        for (int e = 0; e < nBox; e++) {
            double d = wy[bRowOf[e]];
            if (d != 0) rhs[bJ[e]] += d * bV[e];
        }
        for (int e = 0; e < nPair; e++) {
            double d = wy[pRowOf[e]];
            if (d != 0) {
                rhs[pJ1[e]] += d * pV1[e];
                rhs[pJ2[e]] += d * pV2[e];
            }
        }
        {
            int c = 0;
            for (int i = 0; i < nGen; i++) {
                int r = gs + i, len = gLenp[i];
                double d = wy[r];
                if (d != 0 && len > 0) {
                    int j0 = gCol0p[i];
                    const double *v = gVp + c;
                    for (int t = 0; t < len; t++) rhs[j0 + t] += d * v[t];
                }
                c += len;
            }
        }
        w.solve(xt.data(), rhs.data());
        // Ax~ (of the NEW x), z+/y+ classical residual update
        for (int e = 0; e < nBox; e++) Ax[bRowOf[e]] = bV[e] * xt[bJ[e]];
        for (int e = 0; e < nPair; e++) {
            int r = pRowOf[e];
            Ax[r] = pV1[e] * xt[pJ1[e]] + pV2[e] * xt[pJ2[e]];
        }
        {
            int c = 0;
            for (int i = 0; i < nGen; i++) {
                int r = gs + i, len = gLenp[i];
                if (len > 0) {
                    int j0 = gCol0p[i];
                    const double *v = gVp + c;
                    const double *xx = xt.data() + j0;
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    int t = 0;
                    for (; t + 4 <= len; t += 4) {
                        s0 += v[t + 0] * xx[t + 0];
                        s1 += v[t + 1] * xx[t + 1];
                        s2 += v[t + 2] * xx[t + 2];
                        s3 += v[t + 3] * xx[t + 3];
                    }
                    double s = (s0 + s1) + (s2 + s3);
                    for (; t < len; t++) s += v[t] * xx[t];
                    Ax[r] = s;
                } else Ax[r] = 0;
                c += len;
            }
        }
        for (int r = 0; r < m; r++) {
            double v = y[r] / rho + Ax[r];
            double lo = L[r], up = U[r];
            double zc = v < lo ? lo : (v > up ? up : v);
            z[r] = zc;
            y[r] += rho * (Ax[r] - zc);
        }
        for (int j = 0; j < n; j++) x[j] = xt[j];
        if (!std::isfinite(z[0]) || !std::isfinite(y[0])) {
            res.status = "numeric_error";
            break;
        }
        if (it % checkEvery == 0 || it == 1) {
            // primal residual ||Ax - z||_inf (Ax is of the NEW x already)
            double primRes = 0, zInf = 0, axInf = 0;
            for (int r = 0; r < m; r++) {
                double d = std::fabs(Ax[r] - z[r]);
                if (d > primRes) primRes = d;
                if (std::fabs(Ax[r]) > axInf) axInf = std::fabs(Ax[r]);
                if (std::fabs(z[r]) > zInf) zInf = std::fabs(z[r]);
            }
            // dual residual ||Px + q + A'y||_inf with the NEW y
            for (int j = 0; j < n; j++) aty[j] = p.q[j];
            for (int e = 0; e < nBox; e++) {
                double yr = y[bRowOf[e]];
                if (yr != 0) aty[bJ[e]] += bV[e] * yr;
            }
            for (int e = 0; e < nPair; e++) {
                double yr = y[pRowOf[e]];
                if (yr != 0) {
                    aty[pJ1[e]] += pV1[e] * yr;
                    aty[pJ2[e]] += pV2[e] * yr;
                }
            }
            {
                int c = 0;
                for (int i = 0; i < nGen; i++) {
                    int r = gs + i, len = gLenp[i];
                    double yr = y[r];
                    if (yr != 0 && len > 0) {
                        int j0 = gCol0p[i];
                        const double *v = gVp + c;
                        for (int t = 0; t < len; t++) aty[j0 + t] += v[t] * yr;
                    }
                    c += len;
                }
            }
            double dualRes = 0, pxInf = 0, atyInf = 0;
            for (int j = 0; j < n; j++) {
                double px = p.pDiag[j] * x[j];
                double mu = px + aty[j];
                if (std::fabs(mu) > dualRes) dualRes = std::fabs(mu);
                if (std::fabs(px) > pxInf) pxInf = std::fabs(px);
                if (std::fabs(aty[j]) > atyInf) atyInf = std::fabs(aty[j]);
            }
            double epsP = epsAbs + epsRel * std::fmax(axInf, zInf);
            double epsD = epsAbs + epsRel * std::fmax(pxInf, atyInf);
            // PhiHat monitor (paper Theorem 2.9, constant-free memory term)
            phiHat = 0;
            for (int j = 0; j < n; j++)
                phiHat += 0.5 * p.pDiag[j] * x[j] * x[j] + p.q[j] * x[j];
            for (int r = 0; r < m; r++) {
                double rr = Ax[r] - z[r];
                phiHat += y[r] * rr + 1.5 * rho * rr * rr;
            }
            // argmin-KKT-residual output rule (paper Theorem 2.12)
            double ratio = std::fmax(primRes / epsP, dualRes / epsD);
            if (ratio < bestRatio) {
                bestRatio = ratio;
                bestIt = it;
                std::copy(x.begin(), x.end(), xBest.begin());
                std::copy(y.begin(), y.end(), yBest.begin());
            }
            if (phiHat < phiHatMin) phiHatMin = phiHat;
            if (dbg && (it == 1 || it % 100 == 0))
                fprintf(stderr,
                        "[ralm] it=%d pri=%.2e dua=%.2e ratio=%.3f best=%.3f "
                        "(it %d) phi=%.6e phiMin=%.6e\n",
                        it, primRes, dualRes, ratio, bestRatio, bestIt,
                        phiHat, phiHatMin);
            if (verbose && (it == 1 || it % 100 == 0))
                std::printf("nativeralm iter %8d: pri %9.3e dua %9.3e "
                            "(eps %8.2e/%8.2e)\n",
                            it, primRes, dualRes, epsP, epsD);
            res.priRes = primRes;
            res.duaRes = dualRes;
            if (primRes <= epsP && dualRes <= epsD) {
                res.status = "solved";
                break;
            }
        }
    }
    if (res.status.empty()) res.status = it > maxIter ? "max_iter" : "solved";
    auto t1 = std::chrono::steady_clock::now();
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x = xBest;   // argmin-residual iterate (== last when solved)
    res.iters = it;
    res.y = yBest;
    if (dbg)
        fprintf(stderr,
                "[ralm] final it=%d bestIt=%d bestRatio=%.3f rho=%.3g "
                "beta=%.3g phiHatMin=%.6e\n",
                it, bestIt, bestRatio, rho, beta, phiHatMin);
    double obj = 0;
    for (int j = 0; j < n; j++)
        obj += 0.5 * p.pDiag[j] * xBest[j] * xBest[j] + p.q[j] * xBest[j];
    res.obj = obj;
    return res;
}
