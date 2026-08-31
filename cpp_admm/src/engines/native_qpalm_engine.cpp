// native_qpalm_engine.cpp - admm_st2 S2 extension: in-house proximal-ALM +
// semismooth-Newton engine, faithful port of the QPALM algorithm
// (Hermans-Themelis-Patrinos 2022) with a problem-specific Newton solver:
// block-diagonal (per-symbol clusters) + Woodbury (active general rows, k<=67).
//
// Verified semantics (against /tmp/cppadmm/qpalm/src):
//   sigma0  = clip(sigma_init*max(1,|f0|)/max(1,0.5*dist2), 1e-4, 1e4)
//             (initialize_sigma; data-driven, NOT the bare sigma_init)
//   outer   = proximal ALM on Phi(x)=f(x)+sum phi_sigma(b-Ax)+||x-x0||^2/2g
//   inner   = semismooth Newton: active rows at bounds of Axys=Ax+y./sigma;
//             (Q+1/gamma I+sum_active sigma a a') d = -dphi, dphi=Qx+q+(x-x0)/
//             gamma+A'yh, yh=y+sigma.*(Ax-z); exact piecewise linesearch
//   inner stop: ||dphi||<=eps_in  -> y<-yh, sigma boost on violated+active
//             rows (mult=max(1,delta*|r|/(|r|_inf+eps)), cap sigma_max),
//             eps_in<-max(eps,rho*eps_in), x0<-x
//   outer stop: ||Ax-z||<=eps_abs+eps_rel*max(||Ax||,||z||) AND
//             ||Qx+q+A'y||<=eps_abs+eps_rel*max(||Qx||,||q||,||A'y||)
//
// Newton linear algebra (THE structure win):
//   M = diag(Q)+1/gamma + sum_active sigma_i a_i a_i'. Active structure rows
//   (box/TUB/GUB) touch variables of ONE symbol only -> per-symbol clusters
//   (<=3x3 blocks, union-find over active pair rows, CONTIGUOUS two-pass
//   layout). Active general rows (<=67) -> rank-k Woodbury:
//   M^-1 b = Dinv b - Dinv W'(I + W Dinv W')^-1 W Dinv b,
//   with FULL-width Dinv*w rows (block inverses are non-diagonal: Dinv*w
//   leaks to slice-external block-mates u_j/g_j).
#include "engine.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

bool nativeqpalmAvailable() { return true; }

EngineResult nativeqpalmSolve(const Problem &p, double epsAbs, double epsRel,
                              bool verbose, const EngineTuning &tune) {
    (void)verbose;
    int n = p.n, m = (int)p.rows.size();
    int gs = p.generalStart >= 0 ? p.generalStart : m;

    // defaults from qpalm/include/constants.h (+ bc-study sigma_init=1)
    double sigmaInit = tune.sigma >= 0 ? tune.sigma : 1.0;
    double sigmaMax = 1e9;
    double theta = 0.25, deltaUpd = 100.0;
    double gammaInit = 1e7, gammaUpd = 10.0, gammaMax = 1e7;
    double rhoTol = tune.rho >= 0 ? tune.rho : 0.1;
    double epsAbsIn = 1.0, epsRelIn = 1.0;
    int maxIter = tune.maxIter > 0 ? tune.maxIter : 1000000;
    int innerMaxIter = 100;

    EngineResult res;

    // ---- row storage ----
    std::vector<int> rowPtr(m + 1, 0);
    std::vector<int> ri;
    std::vector<double> rv;
    std::vector<double> bmin(m), bmax(m);
    int nnz = 0;
    for (auto &r : p.rows) nnz += (int)r.idx.size();
    ri.resize(nnz);
    rv.resize(nnz);
    {
        int c = 0;
        for (int r = 0; r < m; r++) {
            rowPtr[r] = c;
            bmin[r] = p.rows[r].lb;
            bmax[r] = p.rows[r].ub;
            for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
                ri[c] = p.rows[r].idx[t];
                rv[c] = p.rows[r].val[t];
                c++;
            }
        }
        rowPtr[m] = c;
    }
    std::vector<int> gCol0(m - gs, 0), gLen(m - gs, 0);
    for (int i = 0; i < m - gs; i++) {
        const Row &r = p.rows[gs + i];
        gLen[i] = (int)r.idx.size();
        gCol0[i] = gLen[i] ? r.idx[0] : 0;
        for (size_t t = 1; t < r.idx.size(); t++)
            if (r.idx[t] != r.idx[t - 1] + 1) {
                res.status = "setup_error_noncontig";
                return res;
            }
    }

    // ---- state ----
    std::vector<double> x(n, 0.0), x0(n, 0.0), y(m, 0.0);
    double sigma0 = sigmaInit;
    {
        double f0 = 0; // x = 0
        double dist2 = 0;
        for (int r = 0; r < m; r++) {
            double a = 0; // Ax at x=0
            double t = a < bmin[r] ? bmin[r] : (a > bmax[r] ? bmax[r] : a);
            dist2 += (a - t) * (a - t);
        }
        sigma0 = sigmaInit * std::fmax(1.0, std::fabs(f0)) /
                 std::fmax(1.0, 0.5 * dist2);
        if (sigma0 < 1e-4) sigma0 = 1e-4;
        if (sigma0 > 1e4) sigma0 = 1e4;
    }
    std::vector<double> sigma(m, sigma0), sigmaInv(m, 1.0 / sigma0),
        sqrtSigma(m, std::sqrt(sigma0));
    double gamma = gammaInit;
    std::vector<double> Qx(n, 0.0), Ax(m, 0.0);
    std::vector<double> z(m), Axys(m), priRes(m), yh(m), df(n), Atyh(n),
        dphi(n), Aty(n, 0.0);
    std::vector<char> active(m, 0), activeOld(m, 0);
    std::vector<double> d(n), Qd(n), Ad(m);
    std::vector<double> delta(2 * m), alpha(2 * m);

    auto computeResiduals = [&]() {
        for (int r = 0; r < m; r++) Axys[r] = Ax[r] + y[r] * sigmaInv[r];
        for (int r = 0; r < m; r++) {
            double t = Axys[r];
            z[r] = t < bmin[r] ? bmin[r] : (t > bmax[r] ? bmax[r] : t);
            priRes[r] = Ax[r] - z[r];
            yh[r] = y[r] + sigma[r] * priRes[r];
        }
        for (int j = 0; j < n; j++)
            df[j] = Qx[j] + p.q[j] - x0[j] / gamma;
        std::fill(Atyh.begin(), Atyh.end(), 0.0);
        for (int r = 0; r < m; r++) {
            double yr = yh[r];
            if (yr == 0) continue;
            for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++)
                Atyh[ri[t]] += rv[t] * yr;
        }
        for (int j = 0; j < n; j++) dphi[j] = df[j] + Atyh[j];
    };

    auto setActive = [&]() {
        for (int r = 0; r < m; r++)
            active[r] = (Axys[r] <= bmin[r]) || (Axys[r] >= bmax[r]);
    };

    // ---- Newton system state (rebuilt on active-set change) ----
    std::vector<int> blkOfVar(n), blkSz, blkStart, varSlot(n);
    std::vector<int> blkVar;
    std::vector<double> blkInv;
    int nBlk = 0;
    std::vector<int> uf(n), rootBlk(n);
    auto findUf = [&](int xx) {
        while (uf[xx] != xx) { uf[xx] = uf[uf[xx]]; xx = uf[xx]; }
        return xx;
    };
    std::vector<int> wStart, wLen, wCol0;
    std::vector<double> wVal;
    std::vector<double> cholS, dinvWt;
    int kAct = 0;
    bool nsOk = false;
    bool sigDirty = true;      // sigma changed -> W/S must be rebuilt
    std::vector<int> wRowLast; // general active-row signature
    int nGenSkip = 0, nGenRebuild = 0;
    std::vector<double> dFullBuf(n), wtBuf(n), rhsBuf, negDphi(n);
    std::vector<double> lastPriRes(m, 0.0);

    bool layoutDone = false;
    auto buildNewton = [&]() {
        // FIXED layout: union-find over ALL structure pair rows once (an
        // inactive pair row simply contributes nothing to its block - same
        // matrix, but the block partition/varSlot never need recompute)
        if (!layoutDone) {
            layoutDone = true;
            for (int j = 0; j < n; j++) uf[j] = j;
            for (int r = 0; r < gs; r++) {
                const Row &rw = p.rows[r];
                if (rw.idx.size() == 2) {
                    int a = findUf(rw.idx[0]), b = findUf(rw.idx[1]);
                    if (a != b) uf[a] = b;
                }
            }
            std::fill(rootBlk.begin(), rootBlk.end(), -1);
            std::vector<int> blkOf(n);
            int nb2 = 0;
            for (int j = 0; j < n; j++) {
                int rt = findUf(j);
                if (rootBlk[rt] < 0) rootBlk[rt] = nb2++;
                blkOf[j] = rootBlk[rt];
            }
            blkSz.assign(nb2, 0);
            for (int j = 0; j < n; j++) blkSz[blkOf[j]]++;
            blkStart.assign(nb2, 0);
            int acc2 = 0;
            for (int b = 0; b < nb2; b++) {
                blkStart[b] = acc2;
                acc2 += blkSz[b];
            }
            blkVar.assign(n, 0);
            {
                std::vector<int> fill2(blkStart.begin(), blkStart.end());
                for (int j = 0; j < n; j++) blkVar[fill2[blkOf[j]]++] = j;
            }
            for (int j = 0; j < n; j++) blkOfVar[j] = blkOf[j];
            varSlot.assign(n, -1);
            for (int b = 0; b < nb2; b++)
                for (int u = 0; u < blkSz[b]; u++)
                    varSlot[blkVar[blkStart[b] + u]] = u;
            nBlk = nb2;
        }
        // O(nnz) assembly: base diag, then ONE pass over active structure rows
        blkInv.assign((size_t)nBlk * 16, 0.0);
        for (int b = 0; b < nBlk; b++) {
            int sz = blkSz[b];
            double *D = &blkInv[(size_t)b * 16];
            for (int u = 0; u < sz; u++) {
                int var = blkVar[blkStart[b] + u];
                D[u * 4 + u] = p.pDiag[var] + 1.0 / gamma;
            }
        }
        for (int r = 0; r < gs; r++) {
            if (!active[r]) continue;
            const Row &rw = p.rows[r];
            if (rw.idx.empty()) continue;
            double sg = sigma[r];
            double *D = &blkInv[(size_t)blkOfVar[rw.idx[0]] * 16];
            for (size_t a = 0; a < rw.idx.size(); a++) {
                int ta = varSlot[rw.idx[a]];
                if (ta < 0) continue;
                for (size_t c = 0; c < rw.idx.size(); c++) {
                    int tc = varSlot[rw.idx[c]];
                    if (tc < 0) continue;
                    D[ta * 4 + tc] += sg * rw.val[a] * rw.val[c];
                }
            }
        }
        // invert each block in place (TOP-LEFT sz x sz only; 4x4 padding
        // slots are zero and would zero the determinant)
        for (int b = 0; b < nBlk; b++) {
            int sz = blkSz[b];
            double *D = &blkInv[(size_t)b * 16];
            if (sz == 1) {
                D[0] = std::fabs(D[0]) > 0 ? 1.0 / D[0] : 1e12;
            } else if (sz == 2) {
                double a = D[0], b2 = D[1], c = D[4], dd = D[5];
                double det = a * dd - b2 * c;
                if (std::isfinite(det) && std::fabs(det) > 0) {
                    double idet = 1.0 / det;
                    D[0] = dd * idet; D[1] = -b2 * idet;
                    D[4] = -c * idet; D[5] = a * idet;
                } else { D[0] = D[5] = 1e12; D[1] = D[4] = 0; }
            } else {
                Eigen::Matrix3d Bm;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) Bm(i, j) = D[i * 4 + j];
                double det = Bm.determinant();
                if (std::isfinite(det) && std::fabs(det) > 0) {
                    Eigen::Matrix3d Bi = Bm.inverse();
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++) D[i * 4 + j] = Bi(i, j);
                } else {
                    for (int i = 0; i < sz; i++) D[i * 4 + i] = 1e12;
                }
            }
        }
        // active general rows -> W. Skip the whole Woodbury rebuild when the
        // general active set AND all sigma are unchanged (structure-only
        // enter/leave leaves W, S, cholS, dinvWt identical)
        wStart.clear(); wLen.clear(); wCol0.clear(); wVal.clear();
        for (int i = 0; i < m - gs; i++) {
            int r = gs + i;
            if (!active[r] || gLen[i] == 0) continue;
            double sq = std::sqrt(sigma[r]);
            wStart.push_back((int)wVal.size());
            wLen.push_back(gLen[i]);
            wCol0.push_back(gCol0[i]);
            for (int t = 0; t < gLen[i]; t++)
                wVal.push_back(sq * p.rows[r].val[t]);
        }
        kAct = (int)wStart.size();
        rhsBuf.assign(kAct > 0 ? kAct : 1, 0.0);
        nsOk = true;
        if (kAct == 0) { cholS.clear(); return; }
        (void)nGenSkip; (void)nGenRebuild; (void)sigDirty; (void)wRowLast;
        int k = kAct;
        // FULL-width Dinv*w rows (leak to block-mates outside the slice);
        // the same pass also fills the dinvWt cache (n x k, column i)
        std::vector<double> wdFull((size_t)k * n, 0.0);
        dinvWt.assign((size_t)n * k, 0.0);
        {
            std::vector<double> wFull(n, 0.0);
            for (int i = 0; i < k; i++) {
                int st = wStart[i], len = wLen[i], c0 = wCol0[i];
                std::fill(wFull.begin(), wFull.end(), 0.0);
                for (int t = 0; t < len; t++) wFull[c0 + t] = wVal[st + t];
                double *out = &wdFull[(size_t)i * n];
                for (int b = 0; b < nBlk; b++) {
                    int sz = blkSz[b], bs = blkStart[b];
                    bool any = false;
                    for (int u = 0; u < sz; u++)
                        if (wFull[blkVar[bs + u]] != 0) { any = true; break; }
                    if (!any) continue;
                    double acc[4] = {0, 0, 0, 0};
                    for (int u = 0; u < sz; u++) {
                        double s2 = 0;
                        for (int v = 0; v < sz; v++)
                            s2 += blkInv[(size_t)b * 16 + u * 4 + v] *
                                  wFull[blkVar[bs + v]];
                        acc[u] = s2;
                    }
                    for (int u = 0; u < sz; u++) {
                        int var = blkVar[bs + u];
                        out[var] = acc[u];
                        dinvWt[(size_t)var * k + i] = acc[u];
                    }
                }
            }
        }
        // S = I + W Dinv W': dot full (Dinv w_i) with sparse w_j over slice
        std::vector<double> S2((size_t)k * k, 0.0);
        for (int i = 0; i < k; i++) {
            const double *di = &wdFull[(size_t)i * n];
            for (int j = i; j < k; j++) {
                int sj = wStart[j], lj = wLen[j], cj = wCol0[j];
                const double *wj = &wVal[sj];
                const double *dij = di + cj;
                double s0 = 0, s1 = 0;
                int t = 0;
                for (; t + 2 <= lj; t += 2) {
                    s0 += dij[t] * wj[t];
                    s1 += dij[t + 1] * wj[t + 1];
                }
                for (; t < lj; t++) s0 += dij[t] * wj[t];
                double s2 = s0 + s1;
                S2[(size_t)i * k + j] = s2;
                if (j != i) S2[(size_t)j * k + i] = s2;
            }
        }
        for (int i = 0; i < k; i++) S2[(size_t)i * k + i] += 1.0;
        for (int i = 0; i < k; i++)
            for (int j = 0; j <= i; j++) {
                double s2 = S2[(size_t)i * k + j];
                for (int l = 0; l < j; l++)
                    s2 -= S2[(size_t)i * k + l] * S2[(size_t)j * k + l];
                if (i == j) {
                    if (s2 <= 0) { nsOk = false; return; }
                    S2[(size_t)i * k + i] = std::sqrt(s2);
                } else {
                    S2[(size_t)i * k + j] = s2 / S2[(size_t)j * k + j];
                }
            }
        cholS = std::move(S2);
    };

    auto newtonSolve = [&](const double *b, double *dout) {
        double *v1 = dFullBuf.data();
        for (int j = 0; j < n; j++) v1[j] = b[j];
        for (int blk = 0; blk < nBlk; blk++) {
            int sz = blkSz[blk], bs = blkStart[blk];
            double acc[4] = {0, 0, 0, 0};
            for (int u = 0; u < sz; u++) {
                double s2 = 0;
                for (int v = 0; v < sz; v++)
                    s2 += blkInv[(size_t)blk * 16 + u * 4 + v] *
                          v1[blkVar[bs + v]];
                acc[u] = s2;
            }
            for (int u = 0; u < sz; u++) v1[blkVar[bs + u]] = acc[u];
        }
        if (kAct == 0) {
            for (int j = 0; j < n; j++) dout[j] = v1[j];
            return;
        }
        int k = kAct;
        double *rhs = rhsBuf.data();
        for (int i = 0; i < k; i++) {
            int st = wStart[i], len = wLen[i], c0 = wCol0[i];
            const double *vv = &wVal[st];
            const double *xx = &v1[c0];
            double s0 = 0, s1 = 0;
            int t = 0;
            for (; t + 2 <= len; t += 2) {
                s0 += vv[t] * xx[t];
                s1 += vv[t + 1] * xx[t + 1];
            }
            for (; t < len; t++) s0 += vv[t] * xx[t];
            rhs[i] = s0 + s1;
        }
        for (int i = 0; i < k; i++) {
            double s2 = rhs[i];
            const double *ci = &cholS[(size_t)i * k];
            for (int l = 0; l < i; l++) s2 -= ci[l] * rhs[l];
            rhs[i] = s2 / ci[i];
        }
        for (int i = k - 1; i >= 0; i--) {
            double s2 = rhs[i];
            for (int l = i + 1; l < k; l++)
                s2 -= cholS[(size_t)l * k + i] * rhs[l];
            rhs[i] = s2 / cholS[(size_t)i * k + i];
        }
        for (int j = 0; j < n; j++) dout[j] = v1[j];
        for (int j = 0; j < n; j++) {
            const double *row = &dinvWt[(size_t)j * k];
            double s2 = 0;
            for (int i = 0; i < k; i++) s2 += row[i] * rhs[i];
            dout[j] -= s2;
        }
    };

    // ---- exact piecewise linesearch (port of linesearch.c) ----
    auto alden = [](double a, double dd) -> double {
        if (dd > 0) return a > 0 ? a / dd : -1.0;
        if (dd < 0) return a < 0 ? a / dd : -1.0;
        return -1.0;
    };
    struct AE { double x; int i; };
    std::vector<AE> sArr(2 * m);
    static long g_walkSum = 0, g_walkMax = 0, g_walkN = 0;
    static const bool lsDbg = getenv("NQP_LS_DEBUG") != nullptr;
    auto exactLinesearch = [&]() -> double {
        for (int j = 0; j < n; j++) Qd[j] = p.pDiag[j] * d[j] + d[j] / gamma;
        for (int r = 0; r < m; r++) {
            double s2 = 0;
            for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++)
                s2 += rv[t] * d[ri[t]];
            Ad[r] = s2;
        }
        double eta = 0, beta = 0;
        for (int j = 0; j < n; j++) {
            eta += d[j] * Qd[j];
            beta += d[j] * df[j];
        }
        for (int r = 0; r < m; r++) {
            double sqAd = sqrtSigma[r] * Ad[r];
            delta[r] = -sqAd;
            delta[m + r] = sqAd;
            alpha[r] = (y[r] + sigma[r] * (Ax[r] - bmin[r])) / sqrtSigma[r];
            alpha[m + r] = (-y[r] + sigma[r] * (bmax[r] - Ax[r])) / sqrtSigma[r];
        }
        // min-heap of positive breakpoints; walk pops only as far as the
        // first non-negative gradient (avg 41 pops vs ~2000 breakpoints,
        // replacing the full O(nL log nL) sort)
        int nL = 0;
        for (int i = 0; i < 2 * m; i++) {
            double q = alden(alpha[i], delta[i]);
            if (q > 0) {
                sArr[nL].x = q;
                sArr[nL].i = i;
                nL++;
            }
        }
        double a = eta, b = beta;
        for (int i = 0; i < 2 * m; i++) {
            bool L = alden(alpha[i], delta[i]) > 0;
            bool P = delta[i] > 0;
            if (L != P) {
                a += delta[i] * delta[i];
                b -= delta[i] * alpha[i];
            }
        }
        auto heapGreater = [](const AE &pa, const AE &pb) { return pa.x > pb.x; };
        std::make_heap(sArr.begin(), sArr.begin() + nL, heapGreater);
        auto popMin = [&]() {
            std::pop_heap(sArr.begin(), sArr.begin() + nL, heapGreater);
            nL--;
            return sArr[nL];
        };
        long walked = 0;
        while (nL > 0) {
            AE top = popMin();
            // check gradient just BEFORE this breakpoint using current (a,b):
            // if the segment ending at top.x is already non-negative, the
            // minimizer lies in the previous segment
            int iz = top.i;
            bool P = delta[iz] > 0;
            if (a * top.x + b > 0) {
                // minimizer in previous segment: -b/a with current a,b
                break;
            }
            // cross the breakpoint: update (a, b) for the next segment
            if (P) {
                a += delta[iz] * delta[iz];
                b -= delta[iz] * alpha[iz];
            } else {
                a -= delta[iz] * delta[iz];
                b += delta[iz] * alpha[iz];
            }
            walked++;
        }
        if (lsDbg) { g_walkSum += walked; if (walked > g_walkMax) g_walkMax = walked; g_walkN++; }
        return (a != 0) ? -b / a : 0;
    };

    // ---- main loop (mirror of qpalm_solve) ----
    double epsAbsInCur = epsAbsIn, epsRelInCur = epsRelIn;
    int iterOut = 0, prevIter = -1, noChangeActive = 0;
    bool resetNewton = true;
    double sigmaMaxSeen = 0, tauLast = 0;
    int nBuilds = 0;
    double tBuild = 0, tLS = 0;
    auto __t0 = std::chrono::steady_clock::now();
    auto __lap = [&]() {
        auto __t1 = std::chrono::steady_clock::now();
        double dd = std::chrono::duration<double, std::milli>(__t1 - __t0).count();
        __t0 = __t1;
        return dd;
    };
    int iter = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (iter = 0; iter < maxIter; iter++) {
        computeResiduals();
        double priNorm = 0, axNorm = 0, zNorm = 0;
        for (int r = 0; r < m; r++) {
            double v = std::fabs(priRes[r]);
            if (v > priNorm) priNorm = v;
            v = std::fabs(Ax[r]);
            if (v > axNorm) axNorm = v;
            v = std::fabs(z[r]);
            if (v > zNorm) zNorm = v;
        }
        double epsPri = epsAbs + epsRel * std::fmax(axNorm, zNorm);
        double duaNorm = 0, qxNorm = 0, qNorm = 0, atyhNorm = 0;
        for (int j = 0; j < n; j++) {
            double qaty = Qx[j] - x[j] / gamma + p.q[j] + Atyh[j];
            double v = std::fabs(qaty);
            if (v > duaNorm) duaNorm = v;
            v = std::fabs(Qx[j] - x[j] / gamma);
            if (v > qxNorm) qxNorm = v;
            v = std::fabs(p.q[j]);
            if (v > qNorm) qNorm = v;
            v = std::fabs(Atyh[j]);
            if (v > atyhNorm) atyhNorm = v;
        }
        double maxNorm = std::fmax(qxNorm, std::fmax(qNorm, atyhNorm));
        double epsDua = epsAbs + epsRel * maxNorm;
        double epsDuaIn = epsAbsInCur + epsRelInCur * maxNorm;
        double dua2Norm = 0;
        for (int j = 0; j < n; j++) {
            double v = std::fabs(dphi[j]);
            if (v > dua2Norm) dua2Norm = v;
        }
        if (getenv("NATIVEQP_DEBUG") && iter % 25 == 0)
            fprintf(stderr,
                    "it=%d out=%d pri=%.2e dua=%.2e dua2=%.2e epsIn=%.2e "
                    "gMax=%.1e tau=%.2e builds=%d\n",
                    iter, iterOut, priNorm, duaNorm, dua2Norm, epsDuaIn,
                    sigmaMaxSeen, tauLast, nBuilds);
        if (priNorm < epsPri && duaNorm < epsDua) {
            res.status = "solved";
            break;
        }
        auto boostSigma = [&]() {
            sigDirty = true;
            double priNorm_ = 0;
            for (int r = 0; r < m; r++) {
                double v = std::fabs(priRes[r]);
                if (v > priNorm_) priNorm_ = v;
            }
            for (int r = 0; r < m; r++) {
                bool grew =
                    std::fabs(priRes[r]) > theta * std::fabs(lastPriRes[r]);
                if (grew && active[r]) {
                    double mf = std::fmax(1.0, deltaUpd * std::fabs(priRes[r]) /
                                                    (priNorm_ + 1e-6));
                    double s2 = mf * sigma[r];
                    if (s2 > sigmaMax) s2 = sigmaMax;
                    sigma[r] = s2;
                    sigmaInv[r] = 1.0 / s2;
                    sqrtSigma[r] = std::sqrt(s2);
                    if (s2 > sigmaMaxSeen) sigmaMaxSeen = s2;
                }
            }
        };
        if (dua2Norm <= epsDuaIn || noChangeActive == 3) {
            noChangeActive = 0;
            if (iterOut > 0 && priNorm > epsPri) {
                boostSigma();
                resetNewton = true;
            }
            std::copy(yh.begin(), yh.end(), y.begin());
            std::copy(Atyh.begin(), Atyh.end(), Aty.begin());
            epsAbsInCur = std::fmax(epsAbs, rhoTol * epsAbsInCur);
            epsRelInCur = std::fmax(epsRel, rhoTol * epsRelInCur);
            std::copy(x.begin(), x.end(), x0.begin());
            for (int r = 0; r < m; r++) lastPriRes[r] = priRes[r];
            iterOut++;
            prevIter = iter;
        } else if (prevIter >= 0 && iter == prevIter + innerMaxIter) {
            noChangeActive = 0;
            if (iterOut > 0 && priNorm > epsPri) {
                boostSigma();
                resetNewton = true;
            }
            std::copy(x.begin(), x.end(), x0.begin());
            for (int r = 0; r < m; r++) lastPriRes[r] = priRes[r];
            iterOut++;
            prevIter = iter;
        } else {
            setActive();
            int nbEnter = 0, nbLeave = 0;
            for (int r = 0; r < m; r++) {
                if (active[r] && !activeOld[r]) nbEnter++;
                if (!active[r] && activeOld[r]) nbLeave++;
            }
            if (nbEnter + nbLeave) noChangeActive = 0;
            else noChangeActive++;
            if (resetNewton || nbEnter + nbLeave > 0) {
                nBuilds++;
                __lap();
                buildNewton();
                tBuild += __lap();
                if (!nsOk) {
                    res.status = "numeric_error";
                    break;
                }
            }
            // d = M^-1 (-dphi)
            for (int j = 0; j < n; j++) negDphi[j] = -dphi[j];
            newtonSolve(negDphi.data(), d.data());
            // clip absurd directions (weak-curvature aux vars)
            double dMax = 0;
            for (int j = 0; j < n; j++) {
                double v = std::fabs(d[j]);
                if (v > dMax) dMax = v;
            }
            if (dMax > 1e12)
                for (int j = 0; j < n; j++) d[j] *= 1e12 / dMax;
            double tau = exactLinesearch();
            if (!std::isfinite(tau)) tau = 0;
            tauLast = tau;
            tLS += __lap();
            if (tau > 1e12) tau = 1e12;
            for (int j = 0; j < n; j++) {
                x[j] += tau * d[j];
                Qx[j] += tau * Qd[j];
            }
            for (int r = 0; r < m; r++) Ax[r] += tau * Ad[r];
            for (int r = 0; r < m; r++) activeOld[r] = active[r];
            resetNewton = false;
        }
    }
    if (getenv("NATIVEQP_DEBUG"))
        fprintf(stderr, "[nqp] final iter=%d builds=%d tBuild=%.1f tLS=%.1f "
                        "(ms) walk(avg=%ld max=%ld n=%ld)\n",
                iter, nBuilds, tBuild, tLS,
                g_walkN ? g_walkSum / g_walkN : -1, g_walkMax, g_walkN);
    if (res.status.empty()) res.status = iter >= maxIter ? "max_iter" : "solved";
    auto t1 = std::chrono::steady_clock::now();
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x = x;
    res.iters = iter;
    res.y = y;
    double obj = 0;
    for (int j = 0; j < n; j++)
        obj += 0.5 * p.pDiag[j] * x[j] * x[j] + p.q[j] * x[j];
    res.obj = obj;
    return res;
}