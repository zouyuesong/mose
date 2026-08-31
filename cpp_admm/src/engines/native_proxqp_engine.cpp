// native_proxqp_engine.cpp - admm_st2 task B: in-house port of the ProxQP
// sparse algorithm (proxsuite v0.6.4, /tmp/cppadmm/proxsuite) with the
// problem-specific block-diagonal + Woodbury Newton solver.
//
// Verified semantics (against solver.hpp / workspace.hpp / utils.hpp):
//   KKT = [[H + rho*I, C_act'], [C_act, -mu_in I_act]] (+ I rows for inactive,
//   i.e. dz_inact = -z_inact explicitly) -> Schur on x:
//     M = diag(H) + rho + mu_in^{-1} * sum_active a a'
//     dx = M^{-1} (rx + mu_in^{-1} sum_act a_i rz_i),
//     rz_i = mu_in*z_i - (up_i | lo_i),  rx = -(Hx+g) - sum_act z_i a_i
//     dz_act_i = mu_in^{-1} (a_i.dx - rz_i),  dz_inact_i = -z_i
//   inner: primal-dual piecewise line search (breakpoints -lo/Cdx, -up/Cdx,
//   sorted unique; secant between last-negative and first-positive gradient
//   of a(alpha)*alpha + b(alpha) with alpha-dependent active mask)
//   BCL outer: good -> eta_ext *= mu^beta_bcl, eta_in = max(eta_in*mu, eps_min)
//              bad  -> z<-z_prev, mu *= 0.1 (floor 1e-8), etas reset
//              safeguard (no progress & mu<=1e-5) -> cold reset mu=1/1.1
//   defaults: rho=1e-6, mu_in=1e-1, nu=1, alpha_bcl=.1, beta_bcl=.9,
//   initial guess = equality-constrained with empty active set:
//   x = -(H+rho I)^{-1} g, z = 0.
//
// Newton = the SAME block-diagonal (per-symbol clusters over ACTIVE structure
// rows) + rank-k Woodbury (active general rows) machinery as nativeqpalm,
// with uniform coefficient mu_in^{-1} instead of per-row sigma.
#include "engine.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

bool nativeproxAvailable() { return true; }

EngineResult nativeproxCQSolve(const Problem &p, double epsAbs, double epsRel,
                               bool verbose, const EngineTuning &tune);

EngineResult nativeproxSolve(const Problem &p, double epsAbs, double epsRel,
                            bool verbose, const EngineTuning &tune) {
    (void)verbose;
    int n = p.n, m = (int)p.rows.size();
    int gs = p.generalStart >= 0 ? p.generalStart : m;

    // defaults (proxsuite settings.hpp)
    double rho = tune.rho >= 0 ? tune.rho : 1e-6;
    double mu = tune.sigma >= 0 ? tune.sigma : 1e-1; // mu_in
    double nu = 1.0;
    double alphaBcl = 0.1, betaBcl = 0.9;
    double muMinIn = 1e-8, muMaxInInv = 1e8;
    double muUpdFactor = 0.1, muUpdInvFactor = 10.0;
    double coldMu = 1.0 / 1.1, coldMuInv = 1.1;
    double epsInMin = std::fmin(epsAbs, 1e-9);
    int maxIter = tune.maxIter > 0 ? tune.maxIter : 100000;
    int maxIterIn = 1500;
    double safeguard = 1e4;

    EngineResult res;

    // ---- row storage (structure/general split; general rows contiguous) ----
    std::vector<int> rowPtr(m + 1, 0);
    std::vector<int> ri;
    std::vector<double> rv;
    std::vector<double> l(m), u(m);
    int nnz = 0;
    for (auto &r : p.rows) nnz += (int)r.idx.size();
    ri.resize(nnz);
    rv.resize(nnz);
    {
        int c = 0;
        for (int r = 0; r < m; r++) {
            rowPtr[r] = c;
            l[r] = p.rows[r].lb;
            u[r] = p.rows[r].ub;
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
    std::vector<double> x(n), z(m, 0.0);
    // equality-constrained initial guess with empty active set:
    // x = -(H + rho I)^{-1} g
    for (int j = 0; j < n; j++)
        x[j] = -p.q[j] / (p.pDiag[j] + rho);
    std::vector<double> Hx(n), Cx(m), CTz(n, 0.0), dual(n);
    auto fullPass = [&]() {
        for (int j = 0; j < n; j++) Hx[j] = p.pDiag[j] * x[j];
        std::fill(CTz.begin(), CTz.end(), 0.0);
        for (int r = 0; r < m; r++) {
            double s2 = 0;
            for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++) {
                s2 += rv[t] * x[ri[t]];
                CTz[ri[t]] += rv[t] * z[r];
            }
            Cx[r] = s2;
        }
        for (int j = 0; j < n; j++) dual[j] = Hx[j] + p.q[j] + CTz[j];
    };
    fullPass();

    // ---- Newton system (block + Woodbury, uniform coefficient muInv) ----
    std::vector<char> active(m, 0), activeOld(m, 0);
    std::vector<int> blkOfVar(n), blkSz, blkStart, varSlot(n), blkVar;
    std::vector<double> blkInv;
    int nBlk = 0;
    std::vector<int> uf(n), rootBlk(n);
    auto findUf = [&](int xx) {
        while (uf[xx] != xx) { uf[xx] = uf[uf[xx]]; xx = uf[xx]; }
        return xx;
    };
    std::vector<int> wStart, wLen, wCol0, wRow;
    std::vector<double> wVal;
    std::vector<double> cholS, dinvWt;
    int kAct = 0;
    bool nsOk = false;
    std::vector<double> dBuf(n), rhsBuf, wtBuf(n);

    double muInv = 1.0 / mu;

    auto buildNewton = [&]() {
        for (int j = 0; j < n; j++) uf[j] = j;
        for (int r = 0; r < gs; r++) {
            if (!active[r]) continue;
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
            for (int uu = 0; uu < blkSz[b]; uu++)
                varSlot[blkVar[blkStart[b] + uu]] = uu;
        nBlk = nb2;
        blkInv.assign((size_t)nBlk * 16, 0.0);
        for (int b = 0; b < nBlk; b++) {
            int sz = blkSz[b];
            double *D = &blkInv[(size_t)b * 16];
            for (int uu = 0; uu < sz; uu++) {
                int var = blkVar[blkStart[b] + uu];
                D[uu * 4 + uu] = p.pDiag[var] + rho;
            }
        }
        for (int r = 0; r < gs; r++) {
            if (!active[r]) continue;
            const Row &rw = p.rows[r];
            if (rw.idx.empty()) continue;
            double *D = &blkInv[(size_t)blkOfVar[rw.idx[0]] * 16];
            for (size_t a = 0; a < rw.idx.size(); a++) {
                int ta = varSlot[rw.idx[a]];
                if (ta < 0) continue;
                for (size_t c = 0; c < rw.idx.size(); c++) {
                    int tc = varSlot[rw.idx[c]];
                    if (tc < 0) continue;
                    D[ta * 4 + tc] += muInv * rw.val[a] * rw.val[c];
                }
            }
        }
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
        // active general rows -> W = sqrt(muInv) * a
        wStart.clear(); wLen.clear(); wCol0.clear(); wVal.clear(); wRow.clear();
        for (int i = 0; i < m - gs; i++) {
            int r = gs + i;
            if (!active[r] || gLen[i] == 0) continue;
            double sq = std::sqrt(muInv);
            wStart.push_back((int)wVal.size());
            wLen.push_back(gLen[i]);
            wCol0.push_back(gCol0[i]);
            wRow.push_back(r);
            for (int t = 0; t < gLen[i]; t++)
                wVal.push_back(sq * p.rows[r].val[t]);
        }
        kAct = (int)wStart.size();
        rhsBuf.assign(kAct > 0 ? kAct : 1, 0.0);
        nsOk = true;
        if (kAct == 0) return;
        int k = kAct;
        std::vector<double> wdFull((size_t)k * n, 0.0);
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
                    for (int uu = 0; uu < sz; uu++)
                        if (wFull[blkVar[bs + uu]] != 0) { any = true; break; }
                    if (!any) continue;
                    double acc[4] = {0, 0, 0, 0};
                    for (int uu = 0; uu < sz; uu++) {
                        double s2 = 0;
                        for (int v = 0; v < sz; v++)
                            s2 += blkInv[(size_t)b * 16 + uu * 4 + v] *
                                  wFull[blkVar[bs + v]];
                        acc[uu] = s2;
                    }
                    for (int uu = 0; uu < sz; uu++)
                        out[blkVar[bs + uu]] = acc[uu];
                }
            }
        }
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
                for (int ll = 0; ll < j; ll++)
                    s2 -= S2[(size_t)i * k + ll] * S2[(size_t)j * k + ll];
                if (i == j) {
                    if (s2 <= 0) { nsOk = false; return; }
                    S2[(size_t)i * k + i] = std::sqrt(s2);
                } else {
                    S2[(size_t)i * k + j] = s2 / S2[(size_t)j * k + j];
                }
            }
        cholS = std::move(S2);
        dinvWt.assign((size_t)n * k, 0.0);
        {
            std::vector<double> wFull(n, 0.0);
            for (int i = 0; i < k; i++) {
                int st = wStart[i], len = wLen[i], c0 = wCol0[i];
                std::fill(wFull.begin(), wFull.end(), 0.0);
                for (int t = 0; t < len; t++) wFull[c0 + t] = wVal[st + t];
                for (int b = 0; b < nBlk; b++) {
                    int sz = blkSz[b], bs = blkStart[b];
                    bool any = false;
                    for (int uu = 0; uu < sz; uu++)
                        if (wFull[blkVar[bs + uu]] != 0) { any = true; break; }
                    if (!any) continue;
                    double acc[4] = {0, 0, 0, 0};
                    for (int uu = 0; uu < sz; uu++) {
                        double s2 = 0;
                        for (int v = 0; v < sz; v++)
                            s2 += blkInv[(size_t)b * 16 + uu * 4 + v] *
                                  wFull[blkVar[bs + v]];
                        acc[uu] = s2;
                    }
                    for (int uu = 0; uu < sz; uu++)
                        dinvWt[(size_t)blkVar[bs + uu] * k + i] = acc[uu];
                }
            }
        }
    };

    auto newtonSolve = [&](const double *b, double *dout) {
        double *v1 = dBuf.data();
        for (int j = 0; j < n; j++) v1[j] = b[j];
        for (int blk = 0; blk < nBlk; blk++) {
            int sz = blkSz[blk], bs = blkStart[blk];
            double acc[4] = {0, 0, 0, 0};
            for (int uu = 0; uu < sz; uu++) {
                double s2 = 0;
                for (int v = 0; v < sz; v++)
                    s2 += blkInv[(size_t)blk * 16 + uu * 4 + v] *
                          v1[blkVar[bs + v]];
                acc[uu] = s2;
            }
            for (int uu = 0; uu < sz; uu++) v1[blkVar[bs + uu]] = acc[uu];
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
            for (int ll = 0; ll < i; ll++) s2 -= ci[ll] * rhs[ll];
            rhs[i] = s2 / ci[i];
        }
        for (int i = k - 1; i >= 0; i--) {
            double s2 = rhs[i];
            for (int ll = i + 1; ll < k; ll++)
                s2 -= cholS[(size_t)ll * k + i] * rhs[ll];
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

    // ---- main loop ----
    double etaExtInit = std::pow(0.1, alphaBcl);
    double etaExt = etaExtInit, etaIn = 1.0;
    std::vector<double> lo(m), up(m), dx(n), dz(m), Cdx(m), Hdx(n), CTdz(n);
    std::vector<double> xPrev(n), zPrev(m);
    std::vector<char> actLo(m, 0), actUp(m, 0);
    double machineEps = std::numeric_limits<double>::epsilon();
    std::vector<double> alphas(2 * m);
    int totalIter = 0, iterExt = 0, nBuilds = 0;
    double tBuild = 0, tLS = 0;
    auto __t0 = std::chrono::steady_clock::now();
    auto __lap = [&]() {
        auto __t1 = std::chrono::steady_clock::now();
        double dd =
            std::chrono::duration<double, std::milli>(__t1 - __t0).count();
        __t0 = __t1;
        return dd;
    };
    auto t0 = std::chrono::steady_clock::now();

    // termination helper: fills pri/dua lhs+rhs from current x,z (full pass)
    auto checkTerm = [&](double &priLhs, double &priRhs, double &duaLhs,
                         double &duaRhs) {
        fullPass();
        double pri = 0, cxInf = 0;
        for (int r = 0; r < m; r++) {
            double viol = 0;
            if (Cx[r] > u[r]) viol = Cx[r] - u[r];
            if (Cx[r] < l[r]) viol = std::fmax(viol, l[r] - Cx[r]);
            if (viol > pri) pri = viol;
            double av = std::fabs(Cx[r]);
            if (av > cxInf) cxInf = av;
        }
        double dua = 0, hxInf = 0, gInf = 0, ctzInf = 0;
        for (int j = 0; j < n; j++) {
            double dv = std::fabs(dual[j]);
            if (dv > dua) dua = dv;
            double hv = std::fabs(Hx[j]);
            if (hv > hxInf) hxInf = hv;
            double gv = std::fabs(p.q[j]);
            if (gv > gInf) gInf = gv;
            double cv = std::fabs(CTz[j]);
            if (cv > ctzInf) ctzInf = cv;
        }
        priLhs = pri;
        priRhs = epsAbs + epsRel * cxInf;
        duaLhs = dua;
        duaRhs = epsAbs + epsRel * std::fmax(hxInf, std::fmax(gInf, ctzInf));
    };

    bool solved = false;
    for (iterExt = 0; iterExt < maxIter && !solved; iterExt++) {
        double priLhs, priRhs, duaLhs, duaRhs;
        checkTerm(priLhs, priRhs, duaLhs, duaRhs);
        if (getenv("NATIVEPROX_DEBUG"))
            fprintf(stderr, "ext=%d pri=%.2e/%.2e dua=%.2e/%.2e mu=%.1e "
                            "etaE=%.1e etaI=%.1e builds=%d\n",
                    iterExt, priLhs, priRhs, duaLhs, duaRhs, mu, etaExt, etaIn,
                    nBuilds);
        if (priLhs <= priRhs && duaLhs <= duaRhs) {
            solved = true;
            break;
        }
        xPrev = x;
        zPrev = z;
        // lo/up = Cx + mu*z_prev - {l,u}
        for (int r = 0; r < m; r++) {
            double base = Cx[r] + mu * z[r];
            lo[r] = base - l[r];
            up[r] = base - u[r];
        }
        bool dirty = true; // force first build each outer iter? only if active
                           // set changed - track below

        // ---------- inner loop ----------
        double alphaUsed = 0;
        int inner;
        for (inner = 0; inner < maxIterIn; inner++) {
            totalIter++;
            bool changed = dirty;
            for (int r = 0; r < r && false; r++) {}
            for (int r = 0; r < m; r++) {
                actLo[r] = lo[r] <= 0;
                actUp[r] = up[r] >= 0;
                bool a2 = actLo[r] || actUp[r];
                if (a2 != activeOld[r]) changed = true;
                active[r] = a2;
            }
            if (changed) {
                nBuilds++;
                __lap();
                buildNewton();
                tBuild += __lap();
                if (!nsOk) {
                    res.status = "numeric_error";
                    break;
                }
                activeOld = active;
                dirty = false;
            }
            // rhs
            // rx = -(Hx+g) - sum_act z_i a_i ; b = rx + muInv sum_act a_i rz_i
            std::vector<double> &bvec = wtBuf; // reuse
            for (int j = 0; j < n; j++) bvec[j] = -(Hx[j] + p.q[j]);
            for (int r = 0; r < m; r++) {
                if (!active[r]) continue;
                double zr = z[r];
                if (zr == 0) continue;
                for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++)
                    bvec[ri[t]] -= zr * rv[t];
            }
            // active rz contributions (structure rows via direct add; general
            // rows recorded for the Woodbury rhs)
            for (int r = 0; r < gs; r++) {
                if (!active[r]) continue;
                double rz = mu * z[r] - (actUp[r] ? up[r] : lo[r]);
                double c = muInv * rz;
                for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++)
                    bvec[ri[t]] += c * rv[t];
            }
            for (int i = 0; i < kAct; i++) {
                int r = wRow[i];
                double rz = mu * z[r] - (actUp[r] ? up[r] : lo[r]);
                double c = muInv * rz;
                int st = wStart[i], len = wLen[i], c0 = wCol0[i];
                // wVal = sqrt(muInv)*a -> a = wVal/sqrt(muInv)
                double invSq = 1.0 / std::sqrt(muInv);
                for (int t = 0; t < len; t++)
                    bvec[c0 + t] += c * invSq * wVal[st + t];
            }
            newtonSolve(bvec.data(), dx.data());
            // dz: active = muInv(a.dx - rz); inactive = -z
            for (int r = 0; r < m; r++) dz[r] = -z[r];
            for (int r = 0; r < gs; r++) {
                if (!active[r]) continue;
                double rz = mu * z[r] - (actUp[r] ? up[r] : lo[r]);
                double s2 = 0;
                for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++)
                    s2 += rv[t] * dx[ri[t]];
                dz[r] = muInv * (s2 - rz);
            }
            for (int i = 0; i < kAct; i++) {
                int r = wRow[i];
                double rz = mu * z[r] - (actUp[r] ? up[r] : lo[r]);
                int st = wStart[i], len = wLen[i], c0 = wCol0[i];
                const double *wrow = &wVal[st];
                double s2 = 0;
                for (int t = 0; t < len; t++) s2 += wrow[t] * dx[c0 + t];
                // a.dx = w.dx / sqrt(muInv)
                s2 /= std::sqrt(muInv);
                dz[r] = muInv * (s2 - rz);
            }
            // Cdx, Hdx, CTdz
            std::fill(CTdz.begin(), CTdz.end(), 0.0);
            for (int j = 0; j < n; j++) Hdx[j] = p.pDiag[j] * dx[j];
            for (int r = 0; r < m; r++) {
                double s2 = 0;
                for (int t = rowPtr[r]; t < rowPtr[r + 1]; t++) {
                    s2 += rv[t] * dx[ri[t]];
                    CTdz[ri[t]] += rv[t] * dz[r];
                }
                Cdx[r] = s2;
            }
            // ---- primal-dual piecewise line search ----
            {
                // constant parts
                double dxHdx = 0, dx2 = 0, xHdx = 0, xmxpG = 0;
                for (int j = 0; j < n; j++) {
                    dxHdx += dx[j] * Hdx[j];
                    dx2 += dx[j] * dx[j];
                    xHdx += x[j] * Hdx[j];
                    xmxpG += (rho * (x[j] - xPrev[j]) + p.q[j]) * dx[j];
                }
                int nAlpha = 0;
                for (int r = 0; r < m; r++) {
                    double c1 = -lo[r] / (Cdx[r] + machineEps);
                    double c2 = -up[r] / (Cdx[r] + machineEps);
                    if (c1 > machineEps) alphas[nAlpha++] = c1;
                    if (c2 > machineEps) alphas[nAlpha++] = c2;
                }
                std::sort(alphas.begin(), alphas.begin() + nAlpha);
                nAlpha = (int)(std::unique(alphas.begin(),
                                           alphas.begin() + nAlpha) -
                               alphas.begin());
                // grad(alpha) with alpha-dependent mask
                auto abAt = [&](double ac, double &aOut, double &bOut) {
                    double muInvCd2 = 0, muInvCdZact = 0;
                    double nuA2 = 0, nuB = 0;
                    for (int r = 0; r < m; r++) {
                        bool tl = lo[r] + ac * Cdx[r] < 0;
                        bool tu = up[r] + ac * Cdx[r] > 0;
                        if (tl || tu) {
                            double cd = Cdx[r];
                            muInvCd2 += cd * cd;
                            double zap = (tl ? lo[r] : 0) + (tu ? up[r] : 0);
                            muInvCdZact += cd * zap;
                            double t1 = muInv * cd - dz[r];
                            nuA2 += t1 * t1;
                            nuB += (zap - mu * z[r]) * t1;
                        }
                    }
                    aOut = dxHdx + rho * dx2 + muInv * muInvCd2 +
                           nu * mu * nuA2;
                    bOut = xHdx + xmxpG + muInv * muInvCdZact + nu * nuB;
                };
                double alpha;
                if (nAlpha == 0) {
                    double a0, b0;
                    abAt(0, a0, b0);
                    alpha = a0 > 0 ? -b0 / a0 : 1.0;
                } else {
                    double lastNegA = 0, lastNegG = 0, alphaLastNeg = 0;
                    double firstPosG = 0, alphaFirstPos =
                                              std::numeric_limits<double>::
                                                  infinity();
                    bool anyPos = false;
                    for (int i = 0; i < nAlpha; i++) {
                        double ac = alphas[i];
                        double a2, b2;
                        abAt(ac, a2, b2);
                        double gr = a2 * ac + b2;
                        if (gr < 0) {
                            alphaLastNeg = ac;
                            lastNegG = gr;
                            lastNegA = a2;
                        } else {
                            firstPosG = gr;
                            alphaFirstPos = ac;
                            anyPos = true;
                            break;
                        }
                    }
                    if (alphaLastNeg == 0) {
                        double a2, b2;
                        abAt(0, a2, b2);
                        lastNegG = b2;
                    }
                    if (!anyPos) {
                        double a2, b2;
                        abAt(2 * alphaLastNeg + 1, a2, b2);
                        alpha = a2 > 0 ? -b2 / a2 : 1.0;
                    } else {
                        double denom = firstPosG - lastNegG;
                        alpha = denom != 0
                                    ? alphaLastNeg -
                                          lastNegG * (alphaFirstPos -
                                                       alphaLastNeg) / denom
                                    : alphaLastNeg;
                    }
                }
                alphaUsed = alpha;
                double dwInf = 0;
                for (int j = 0; j < n; j++)
                    dwInf = std::fmax(dwInf, std::fabs(dx[j]));
                for (int r = 0; r < m; r++)
                    dwInf = std::fmax(dwInf, std::fabs(dz[r]));
                if (alpha * dwInf < 1e-11 && inner > 0) break;
                if (!(alpha > 0)) alpha = 0; // safeguard
                // apply step + incremental updates
                for (int j = 0; j < n; j++) x[j] += alpha * dx[j];
                for (int r = 0; r < m; r++) z[r] += alpha * dz[r];
                for (int j = 0; j < n; j++)
                    dual[j] += alpha * (Hdx[j] + CTdz[j] + rho * dx[j]);
                for (int r = 0; r < m; r++) {
                    lo[r] += alpha * Cdx[r];
                    up[r] += alpha * Cdx[r];
                }
            }
            tLS += __lap();
            // inner residual
            double errIn = 0;
            for (int r = 0; r < m; r++) {
                double v = 0;
                if (lo[r] < 0) v -= lo[r];
                if (up[r] > 0) v += up[r];
                v -= mu * z[r];
                double av = std::fabs(v);
                if (av > errIn) errIn = av;
            }
            for (int j = 0; j < n; j++) {
                double av = std::fabs(dual[j]);
                if (av > errIn) errIn = av;
            }
            if (errIn <= etaIn) break;
        }
        if (res.status == "numeric_error") break;

        // re-check termination with fresh residuals
        double priNew, priRhs2, duaNew, duaRhs2;
        checkTerm(priNew, priRhs2, duaNew, duaRhs2);
        if (priNew <= priRhs2 && duaNew <= duaRhs2) {
            solved = true;
            break;
        }
        double priOld = priLhs, duaOld = duaLhs;
        // BCL update
        double newMu = mu, newMuInv = muInv;
        if (priNew <= etaExt || iterExt > safeguard) {
            etaExt *= std::pow(mu, betaBcl);
            etaIn = std::fmax(etaIn * mu, epsInMin);
        } else {
            z = zPrev;
            newMu = std::fmax(mu * muUpdFactor, muMinIn);
            newMuInv = std::fmin(muInv * muUpdInvFactor, muMaxInInv);
            etaExt = etaExtInit * std::pow(newMu, alphaBcl);
            etaIn = std::fmax(newMu, epsInMin);
        }
        // safeguard: no progress & mu small -> cold reset
        if (priNew >= priOld && duaNew >= duaOld && mu <= 1e-5) {
            newMu = coldMu;
            newMuInv = coldMuInv;
        }
        if (newMu != mu) {
            mu = newMu;
            muInv = newMuInv;
            dirty = true; // rebuild with new coefficient
            // activeOld stays: rebuild happens next inner iter if needed
        }
        // recompute Cx etc. for next outer pass (checkTerm does fullPass)
    }
    if (solved) res.status = "solved";
    else if (res.status.empty()) res.status = "max_iter";
    if (getenv("NATIVEPROX_DEBUG"))
        fprintf(stderr, "[npx] final ext=%d inner_total=%d builds=%d "
                        "tBuild=%.1f tLS=%.1f\n",
                iterExt, totalIter, nBuilds, tBuild, tLS);
    auto t1 = std::chrono::steady_clock::now();
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x = x;
    res.iters = totalIter;
    // dual convention: y = z (inequality multipliers)
    res.y = z;
    double obj = 0;
    for (int j = 0; j < n; j++)
        obj += 0.5 * p.pDiag[j] * x[j] * x[j] + p.q[j] * x[j];
    res.obj = obj;
    return res;
}