// native_engine.cpp - admm_st2.md S2-1: in-house ADMM, faithful port of the
// OSQP v1.0 iteration with the problem-specific Woodbury solver.
//
// Algorithm (OSQP v1.0, rho_is_vec semantics with uniform rho - verified
// against /tmp/cppadmm/osqp/src/auxil.c + osqp_api.c):
//   rhs_x = sigma*x - q
//   rhs_z = y/rho - z                      (y = UNSCALED dual)
//   (P+sigma*I+rho*A'A) x~ = rhs_x + A'(y - rho*z)   [Woodbury, see below]
//   z~ = rho*(A x~ - rhs_z) = rho*A x~ - y + rho*z
//   x+ = alpha*x~ + (1-alpha)*x
//   z+ = clip( y/rho + alpha*z~ + (1-alpha)*z,  l, u )
//   y+ = y + rho*( alpha*z~ + (1-alpha)*z - z+ )
// y stays unscaled across rho changes -> no dual rescale on adapt.
//
// Termination (v1.0, scaling=0): ||A x - z||_inf <= eps_abs + eps_rel*max(||Ax||,||z||)
// and ||P x + q + A'y||_inf <= eps_abs + eps_rel*max(||Px||,||q+A'y||).
// Adaptive rho (compute_rho_estimate): est = rho*sqrt(prim_res/nprim / dual_res/ndua),
// applied when est deviates > 5x (clamped [1e-6,1e6]).
//
// Linear algebra: rows split [structure | general]. Structure rows
// (box/TUB/GUB pairs) contribute DIAGONAL to A'A (pairs cancel cross terms);
// general rows (67 NET/GROSS) form the rank-k term. M = D + rho*G'G solved
// exactly: k*k Cholesky of I + rho*G D^-1 G' (~1e5 flops per rho change),
// O(nnz) per iteration. No sparse factorization of the 1782-row KKT.
#include "engine.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

bool nativeAvailable() { return true; } // zero library dependencies

namespace {

static double g_dbgT[3] = {0, 0, 0}; // NATIVE_TIME2 accumulators

struct Woodbury {
    int n = 0, k = 0;
    double rho = 0;
    std::vector<double> dtilde, invd;
    std::vector<int> gPtr; // CSR of general rows (index-sorted)
    std::vector<int> gIdx;
    std::vector<double> gVal;
    std::vector<double> chol; // k*k lower
    std::vector<double> bufA, bufV, bufGt;

    bool init(const Problem &p, double rho_, double sigma) {
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
            dtilde[j] = p.pDiag[j] + sigma + rho * ad[j];
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
        static bool nt2 = getenv("NATIVE_TIME2") != nullptr;
        static double *tg1 = &g_dbgT[0], *tch = &g_dbgT[1], *tg2 = &g_dbgT[2];
        auto a0 = std::chrono::steady_clock::now();
        auto lap2 = [&a0]() {
            auto a1 = std::chrono::steady_clock::now();
            double d = std::chrono::duration<double, std::milli>(a1 - a0).count();
            a0 = a1;
            return d;
        };
        if (k == 0) {
            for (int j = 0; j < n; j++) z[j] = r[j] * invd[j];
            return;
        }
        double *a = bufA.data();
        for (int j = 0; j < n; j++) a[j] = r[j] * invd[j];
        double *v = bufV.data();
        // contiguous-slice GEMV (row indices are contiguous ascending ranges)
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
        if (nt2) *tg1 += lap2();
        // forward L v = y (row-contiguous)
        for (int i = 0; i < k; i++) {
            double s = v[i];
            const double *ci = &chol[(size_t)i * k];
            for (int l = 0; l < i; l++) s -= ci[l] * v[l];
            v[i] = s / ci[i];
        }
        // backward L' x = v, RIGHT-LOOKING (all accesses row-contiguous;
        // the textbook column version strides k doubles = one cache line
        // per element and dominated the profile)
        for (int i = k - 1; i >= 0; i--) {
            double xi = v[i] / chol[(size_t)i * k + i];
            v[i] = xi;
            if (xi != 0) {
                const double *ci = &chol[(size_t)i * k];
                for (int j = 0; j < i; j++) v[j] -= ci[j] * xi;
            }
        }
        if (nt2) *tch += lap2();
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
        if (nt2) *tg2 += lap2();
    }
};

} // namespace

EngineResult nativeSolve(const Problem &p, double epsAbs, double epsRel,
                         bool verbose, const EngineTuning &tune) {
    int n = p.n, m = (int)p.rows.size();
    double rho = tune.rho >= 0 ? tune.rho : 0.1;
    double sigma = tune.sigma >= 0 ? tune.sigma : 1e-6;
    double alpha = tune.alpha >= 0 ? tune.alpha : 1.6;
    double beta = 1.0 - alpha;
    int maxIter = tune.maxIter > 0 ? tune.maxIter : 1000000;
    int checkEvery = 25;
    int rhoInterval = tune.rhoInterval > 0 ? tune.rhoInterval : 50;
    double rhoTol = 5.0, rhoMin = 1e-6, rhoMax = 1e6;
    bool rhoAdaptive = tune.adaptiveRho != 0;

    // ---- reflected-Halpern acceleration (admm_st2 S2-2, arXiv 2606.16552) ----
    // Base map F = one native iteration with alpha=1 and FIXED rho (adaptive
    // off: the map must not change mid-run). Wrapper (Algorithm 1/2):
    //   shadow   w^ = F(w)                  [rates/identification live here]
    //   reflect  wbar = (1+gamma) w^ - gamma w
    //   anchor   w+ = (w0 + (k+1) wbar) / (k+2),  k resets per epoch
    // Restart wrapper (Algorithm 3 + 6.1 practical triggers): reset anchor to
    // current state, k<-0, when any of
    //   (1) rF <= 0.2 * rF_epoch_start        (sufficient decrease)
    //   (2) rF <= 0.8 * rF0 && rF > rF_prev   (necessary + no local progress)
    //   (3) kEpoch >= 0.36 * kTot             (epoch too long)
    // Termination & argmin output evaluated on the SHADOW (paper Alg output).
    static bool halpern = getenv("NATIVE_HALPERN") != nullptr;
    static double hg = [] {
        const char *s = getenv("NATIVE_HALPERN_GAMMA");
        return s ? atof(s) : 0.7;
    }();
    if (halpern) {
        rhoAdaptive = false; // fixed map (alpha stays as configured: the
                             // over-relaxed map is still a fixed map)
    }
    static bool hdbg = getenv("NATIVE_HALPERN_DEBUG") != nullptr;
    const double hBetaSuff = 0.2, hBetaNec = 0.8, hBetaArt = 0.36;

    EngineResult res;
    Woodbury w;
    if (!w.init(p, rho, sigma)) {
        res.status = "setup_error";
        return res;
    }

    std::vector<double> x(n, 0.0), xt(n, 0.0), z(m, 0.0), y(m, 0.0);
    std::vector<double> rhs(n), Ax(m), wy(m), aty(n);
    std::vector<double> L(m), U(m);
    // hot-loop row storage, split by structure:
    //   box rows (1 nnz):        bJ[r], bV[r]
    //   pair rows (2 nnz):       pJ1[r], pV1[r], pJ2[r], pV2[r]
    //   general rows (>=3 nnz):  CSR; each row's indices are CONTIGUOUS
    //   (delta/barra over active x 0..na, gross over g block) -> slice access
    int gs = p.generalStart >= 0 ? p.generalStart : m;
    std::vector<int> bJ;      std::vector<double> bV;
    std::vector<int> pJ1, pJ2; std::vector<double> pV1, pV2;
    std::vector<int> pRowOf, bRowOf; // row index of each entry (for z/y access)
    std::vector<int> gRowStart;      // per general row: start col, length
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
            // empty structure rows: nothing
        }
        int nnzG = 0;
        for (int r = gs; r < m; r++) nnzG += (int)p.rows[r].idx.size();
        gValFlat.reserve(nnzG);
        for (int r = gs; r < m; r++) {
            const Row &rw = p.rows[r];
            gCol0.push_back(rw.idx.empty() ? 0 : rw.idx[0]);
            gLen.push_back((int)rw.idx.size());
            for (size_t t = 0; t < rw.idx.size(); t++) {
                gValFlat.push_back(rw.val[t]);
                // contiguity asserted by construction (assembleConstraints
                // emits sorted varIndices; scaleRows preserves them)
            }
        }
    }
    int nBox = (int)bJ.size(), nPair = (int)pJ1.size();
    int nGen = m - gs;
    const int *gCol0p = gCol0.data();
    const int *gLenp = gLen.data();
    const double *gVp = gValFlat.data();
    // elastic-hinge rows (admm_st2 S2-3b Lew folding): three-piece prox
    // instead of the plain clip; c = hinge penalty (already row-scaled)
    std::vector<double> cH(m, 0.0);
    bool anyHinge = false;
    int firstHingeRow = -1;
    for (int r = 0; r < m; r++)
        if (p.rows[r].hinge > 0) {
            cH[r] = p.rows[r].hinge;
            anyHinge = true;
            if (firstHingeRow < 0) firstHingeRow = r;
        }
    static bool hDbg2 = getenv("NATIVE_HINGE_DEBUG") != nullptr;

    auto t0 = std::chrono::steady_clock::now();
    int it = 0;
    int factorizations = 1;
    // Halpern wrapper state (unused when halpern == false)
    std::vector<double> xA(n, 0.0), zA(m, 0.0), yA(m, 0.0);   // epoch anchor
    std::vector<double> xs(n, 0.0), zs(m, 0.0), ys(m, 0.0);   // shadow
    double rF0 = -1, rFprev = -1;                              // epoch rF bookkeeping
    int kEpoch = 0, kTot = 0, nRestarts = 0;
    // argmin-KKT-residual output rule (NR-LALM Theorem 2.12)
    std::vector<double> xBest(n, 0.0), yBest(m, 0.0);
    double bestRatio = 1e300;
    int bestIt = 0;
    // section timing (NATIVE_TIME=1)
    static bool ntime = getenv("NATIVE_TIME") != nullptr;
    double tRhs = 0, tSolve = 0, tAx = 0, tZy = 0, tCheck = 0, tRho = 0;
    auto tt0 = std::chrono::steady_clock::now();
    auto lap = [&]() {
        auto tt1 = std::chrono::steady_clock::now();
        double d = std::chrono::duration<double, std::milli>(tt1 - tt0).count();
        tt0 = tt1;
        return d;
    };
    for (it = 1; it <= maxIter; it++) {
        // rhs = sigma*x - q + A'(rho*z - y)
        for (int j = 0; j < n; j++) rhs[j] = sigma * x[j] - p.q[j];
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
        if (ntime) tRhs += lap();
        w.solve(xt.data(), rhs.data());
        if (ntime) tSolve += lap();
        // Ax~ ; z~ = A x~ (linsys wrapper overwrites the KKT companion,
        // see solve_linsys_qdldl)
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
        if (ntime) tAx += lap();
        if (!halpern) {
            if (anyHinge) {
                for (int r = 0; r < m; r++) {
                    double v = y[r] / rho + alpha * Ax[r] + beta * z[r];
                    double lo = L[r], up = U[r];
                    double tz;
                    if (cH[r] > 0) {
                        // prox of c*dist(.,[lo,up]) + (rho/2)(.-v)^2:
                        // soft shoulder c/rho outside the box
                        if (v < lo)
                            tz = (v + cH[r] / rho < lo) ? v + cH[r] / rho : lo;
                        else if (v > up)
                            tz = (v - cH[r] / rho > up) ? v - cH[r] / rho : up;
                        else tz = v;
                    } else {
                        tz = v < lo ? lo : (v > up ? up : v);
                    }
                    y[r] += rho * (alpha * Ax[r] + beta * z[r] - tz);
                    z[r] = tz;
                }
            } else
            for (int r = 0; r < m; r++) {
                double tz = y[r] / rho + alpha * Ax[r] + beta * z[r];
                double lo = L[r], up = U[r];
                if (tz < lo) tz = lo;
                else if (tz > up) tz = up;
                y[r] += rho * (alpha * Ax[r] + beta * z[r] - tz);
                z[r] = tz;
            }
            if (hDbg2 && anyHinge &&
                (it == 1 || it % 200 == 0)) {
                int r = firstHingeRow;
                fprintf(stderr,
                        "[hinge] it=%d row=%d c=%.3e lo=%.3e up=%.3e "
                        "Ax=%.6e z=%.6e y=%.6e\n",
                        it, r, cH[r], L[r], U[r], Ax[r], z[r], y[r]);
            }
            for (int j = 0; j < n; j++) x[j] = alpha * xt[j] + beta * x[j];
        } else {
            // base-map output = shadow (with the configured alpha)
            for (int r = 0; r < m; r++) {
                double tz = y[r] / rho + alpha * Ax[r] + beta * z[r];
                double lo = L[r], up = U[r];
                if (tz < lo) tz = lo;
                else if (tz > up) tz = up;
                zs[r] = tz;
                ys[r] = y[r] + rho * (alpha * Ax[r] + beta * z[r] - tz);
            }
            for (int j = 0; j < n; j++) xs[j] = alpha * xt[j] + beta * x[j];
            // fixed-point residual of the state (cheap, every iteration)
            double rF2 = 0;
            for (int j = 0; j < n; j++) {
                double d = x[j] - xs[j];
                rF2 += d * d;
            }
            for (int r = 0; r < m; r++) {
                double dz = z[r] - zs[r], dy = y[r] - ys[r];
                rF2 += dz * dz + dy * dy;
            }
            double rF = std::sqrt(rF2);
            if (rF0 < 0) { rF0 = rF; rFprev = rF; }
            // restart triggers (paper 6.1: 0.2 / 0.8 / 0.36); pure-anchoring
            // control run disables them (NATIVE_HALPERN_NORESTART=1)
            static bool noRestart = getenv("NATIVE_HALPERN_NORESTART") != nullptr;
            bool trig = false;
            if (!noRestart) {
                if (rF <= hBetaSuff * rF0) trig = true;               // sufficient
                else if (rF <= hBetaNec * rF0 && rF > rFprev) trig = true; // nec+stall
                else if (kEpoch >= 25 && kEpoch >= hBetaArt * kTot) trig = true;
            }
            if (trig) {
                for (int j = 0; j < n; j++) xA[j] = x[j];
                for (int r = 0; r < m; r++) { zA[r] = z[r]; yA[r] = y[r]; }
                kEpoch = 0;
                rF0 = rF;
                nRestarts++;
            } else {
                kEpoch++;
            }
            rFprev = rF;
            // reflect + anchor: w+ = (w0 + (k+1) wbar) / (k+2) with k = kEpoch
            double cRefl = 1.0 + hg, cAnc = (double)(kEpoch + 1) / (kEpoch + 2);
            for (int j = 0; j < n; j++)
                x[j] = (xA[j] + (kEpoch + 1) * (cRefl * xs[j] - hg * x[j])) /
                       (kEpoch + 2);
            for (int r = 0; r < m; r++) {
                z[r] = (zA[r] + (kEpoch + 1) * (cRefl * zs[r] - hg * z[r])) /
                       (kEpoch + 2);
                y[r] = (yA[r] + (kEpoch + 1) * (cRefl * ys[r] - hg * y[r])) /
                       (kEpoch + 2);
            }
            (void)cAnc;
            kTot++;
            if (hdbg && (it == 1 || it % 100 == 0 || trig))
                fprintf(stderr,
                        "[hlp] it=%d kEp=%d rF=%.3e rF0=%.3e rst=%d "
                        "tot=%d\n",
                        it, kEpoch, rF, rF0, nRestarts, kTot);
        }
        if (ntime) tZy += lap();
        if (!std::isfinite(z[0])) {
            res.status = "numeric_error";
            break;
        }
        if (it % checkEvery == 0 || it == 1) {
            // check point: SHADOW (xs,zs,ys) under halpern; the anchored
            // state otherwise. Ax recomputed from the check x (the shadow x
            // is the over-relaxed blend alpha*xt+(1-alpha)*x, not xt)
            const double *cx, *cz, *cy, *xsrc;
            if (!halpern) {
                xsrc = x.data();
                cz = z.data();
                cy = y.data();
            } else {
                xsrc = xs.data();
                cz = zs.data();
                cy = ys.data();
            }
            {
                for (int e = 0; e < nBox; e++) {
                    int r = bRowOf[e];
                    Ax[r] = bV[e] * xsrc[bJ[e]];
                }
                for (int e = 0; e < nPair; e++) {
                    int r = pRowOf[e];
                    Ax[r] = pV1[e] * xsrc[pJ1[e]] + pV2[e] * xsrc[pJ2[e]];
                }
                int c = 0;
                for (int i = 0; i < nGen; i++) {
                    int r = gs + i, len = gLenp[i];
                    if (len > 0) {
                        int j0 = gCol0p[i];
                        const double *v = gVp + c;
                        const double *xx = xsrc + j0;
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
            cx = xsrc;
            // primal residual: ||Ax - z||_inf
            double primRes = 0, zInf = 0, axInf = 0;
            for (int r = 0; r < m; r++) {
                double d = std::fabs(Ax[r] - cz[r]);
                if (d > primRes) primRes = d;
                if (std::fabs(Ax[r]) > axInf) axInf = std::fabs(Ax[r]);
                if (std::fabs(cz[r]) > zInf) zInf = std::fabs(cz[r]);
            }
            // dual residual: ||P x + q + A'y||_inf
            for (int j = 0; j < n; j++) aty[j] = p.q[j];
            for (int e = 0; e < nBox; e++) {
                double yr = cy[bRowOf[e]];
                if (yr != 0) aty[bJ[e]] += bV[e] * yr;
            }
            for (int e = 0; e < nPair; e++) {
                double yr = cy[pRowOf[e]];
                if (yr != 0) {
                    aty[pJ1[e]] += pV1[e] * yr;
                    aty[pJ2[e]] += pV2[e] * yr;
                }
            }
            {
                int c = 0;
                for (int i = 0; i < nGen; i++) {
                    int r = gs + i, len = gLenp[i];
                    double yr = cy[r];
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
                double px = p.pDiag[j] * cx[j];
                double mu = px + aty[j];
                if (std::fabs(mu) > dualRes) dualRes = std::fabs(mu);
                if (std::fabs(px) > pxInf) pxInf = std::fabs(px);
                if (std::fabs(aty[j]) > atyInf) atyInf = std::fabs(aty[j]);
            }
            double qInf = 0;
            for (int j = 0; j < n; j++)
                if (std::fabs(p.q[j]) > qInf) qInf = std::fabs(p.q[j]);
            double epsP = epsAbs + epsRel * std::fmax(axInf, zInf);
            double epsD = epsAbs + epsRel * std::fmax(pxInf, atyInf);
            if (ntime) tCheck += lap();
            res.priRes = primRes;
            res.duaRes = dualRes;
            if (verbose && (it == 1 || it % 50 == 0 || it == 100 || it == 200))
                std::printf("native iter %8d: rho %9.3e pri %9.3e dua %9.3e "
                            "(eps %8.2e/%8.2e) fact=%d\n",
                            it, rho, primRes, dualRes, epsP, epsD, factorizations);
            // argmin-KKT-residual output rule (NR-LALM Thm 2.12): keep the
            // best check point seen anywhere on the trajectory
            {
                double ratio = std::fmax(primRes / epsP, dualRes / epsD);
                if (ratio < bestRatio) {
                    bestRatio = ratio;
                    bestIt = it;
                    if (halpern) {
                        std::copy(xs.begin(), xs.end(), xBest.begin());
                        std::copy(ys.begin(), ys.end(), yBest.begin());
                    } else {
                        std::copy(x.begin(), x.end(), xBest.begin());
                        std::copy(y.begin(), y.end(), yBest.begin());
                    }
                }
            }
            if (primRes <= epsP && dualRes <= epsD) {
                res.status = "solved";
                break;
            }
            if (rhoAdaptive && it % rhoInterval == 0) {
                double pn = std::fmax(axInf, zInf);
                double dn = std::fmax(std::fmax(qInf, atyInf), pxInf); // OSQP: max(||q||,||A'y||,||Px||)
                double est = rho;
                if (pn > 0 && dn > 0 && dualRes > 0)
                    est = rho * std::sqrt((primRes / (pn + 1e-14)) /
                                          (dualRes / (dn + 1e-14)));
                if (est > rho * rhoTol || est < rho / rhoTol) {
                    double newRho = std::fmin(std::fmax(est, rhoMin), rhoMax);
                    if (newRho != rho) {
                        rho = newRho;
                        if (!w.init(p, rho, sigma)) {
                            res.status = "numeric_error";
                            break;
                        }
                        factorizations++;
                    }
                    if (ntime) tRho += lap();
                }
            }
        }
    }
    if (ntime && g_dbgT[0] > 0)
        fprintf(stderr, "[native.solve] G1=%.1f chol=%.1f G2=%.1f ms\n",
                g_dbgT[0], g_dbgT[1], g_dbgT[2]);
    if (ntime)
        fprintf(stderr, "[native] iters=%d fact=%d  rhs=%.1fms solve=%.1fms "
                        "Ax=%.1fms zy=%.1fms check=%.1fms rho=%.1fms\n",
                it, factorizations, tRhs, tSolve, tAx, tZy, tCheck, tRho);
    if (res.status.empty()) res.status = "max_iter";
    auto t1 = std::chrono::steady_clock::now();
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x = xBest;   // argmin-KKT-residual point (== last point when solved)
    res.iters = it;
    res.y = yBest;
    if (hdbg)
        fprintf(stderr, "[hlp] done: it=%d bestIt=%d bestRatio=%.4f "
                        "restarts=%d gamma=%.2f\n",
                it, bestIt, bestRatio, nRestarts, hg);
    double obj = 0;
    for (int j = 0; j < n; j++)
        obj += 0.5 * p.pDiag[j] * xBest[j] * xBest[j] + p.q[j] * xBest[j];
    res.obj = obj;
    return res;
}