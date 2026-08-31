// scs_engine.cpp - SCS 2.1 C API backend.
//
// SCS does not support two-sided rows l<=Ax<=u (split each ranged row into
// Ax<=u and -Ax<=-l) and has NO P matrix (pure conic form). The diagonal
// quadratic 0.5 x'Px enters via a single second-order-cone epigraph:
//   0.5*x'Px = 0.5*||sqrt(P)x||^2  <=>  || [sqrt(P)x ; (t-1)/2] ||_2 <= (t+1)/2
// with linear cost 0.5*t. Only columns with pDiag>0 join the SOC tail.
//
// Cone row order required by SCS: [zero(eq) | nonneg(ineq) | SOC].
#include "engine.hpp"
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstdio>
#ifdef __cplusplus
extern "C" {
#endif
#include "scs/scs.h"
#include "scs/amatrix.h"
#ifdef __cplusplus
}
#endif

#ifdef USE_SCS
bool scsAvailable() { return true; }

EngineResult scsSolve(const Problem &p, double epsAbs, double epsRel, bool verbose,
                      bool, int libScaling, const EngineTuning &tune) {
    (void)epsRel;
    int n = p.n;
    // final variable layout: [x (n) | t (1)]
    int nF = n + 1;
    int tVar = n;

    std::vector<std::vector<std::pair<int, double>>> eqRows, ineqRows, socRows;
    std::vector<double> eqB, ineqB, socB;
    // provenance of each split one-sided row: (original row, +1 upper / -1 lower)
    std::vector<int> ineqSrc;
    std::vector<int> ineqSgn;

    for (int ri = 0; ri < (int)p.rows.size(); ri++) {
        const Row &row = p.rows[ri];
        double lb = row.lb, ub = row.ub;
        bool li = lb <= -1e30, ui = ub >= 1e30;
        // SCS 2.x has no equality cone block usable here without f-cone tricks:
        // split EVERY row (equality l==u included) into one-sided nonneg rows.
        if (!ui) {
            ineqRows.push_back({});
            for (size_t t = 0; t < row.idx.size(); t++)
                ineqRows.back().push_back({row.idx[t], row.val[t]});
            ineqB.push_back(ub);
            ineqSrc.push_back(ri); ineqSgn.push_back(1);
        }
        if (!li) {
            ineqRows.push_back({});
            for (size_t t = 0; t < row.idx.size(); t++)
                ineqRows.back().push_back({row.idx[t], -row.val[t]});
            ineqB.push_back(-lb);
            ineqSrc.push_back(ri); ineqSgn.push_back(-1);
        }
    }
    // SOC: head s0 = (t+1)/2  (row: -0.5*t + s = 0.5)
    //      tail s1 = (t-1)/2  (row: -0.5*t + s = -0.5)
    //      tail_j = -sqrt(P_jj)*x_j (row: +sqrt(P_jj)*x_j + s = 0)
    std::vector<double> sq(n);
    for (int j = 0; j < n; j++) sq[j] = std::sqrt(p.pDiag[j]);
    socRows.push_back({{tVar, -0.5}}); socB.push_back(0.5);
    socRows.push_back({{tVar, -0.5}}); socB.push_back(-0.5);
    int qsize = 2;
    for (int j = 0; j < n; j++)
        if (sq[j] > 0) {
            socRows.push_back({{j, sq[j]}});
            socB.push_back(0.0);
            qsize++;
        }

    // assemble final row list: ineq (nonneg cone) -> soc
    std::vector<std::vector<std::pair<int, double>>> Frows = ineqRows;
    std::vector<double> Fb = ineqB;
    Frows.insert(Frows.end(), socRows.begin(), socRows.end());
    Fb.insert(Fb.end(), socB.begin(), socB.end());
    int mF = (int)Frows.size();

    size_t nnzF = 0;
    for (auto &rw : Frows) nnzF += rw.size();

    AMatrix *A = (AMatrix *)calloc(1, sizeof(AMatrix));
    A->m = mF; A->n = nF;
    A->x = (scs_float *)malloc(nnzF * sizeof(scs_float));
    A->i = (scs_int *)malloc(nnzF * sizeof(scs_int));
    A->p = (scs_int *)malloc((nF + 1) * sizeof(scs_int));
    std::vector<scs_int> cnt(nF, 0);
    for (auto &rw : Frows)
        for (auto &kv : rw) cnt[kv.first]++;
    A->p[0] = 0;
    for (int j = 0; j < nF; j++) A->p[j + 1] = A->p[j] + cnt[j];
    std::vector<scs_int> fill(nF, 0);
    for (int r = 0; r < mF; r++)
        for (auto &kv : Frows[r]) {
            int col = kv.first, pos = A->p[col] + fill[col]++;
            A->i[pos] = r;
            A->x[pos] = kv.second;
        }
    std::vector<scs_float> Fc(nF, 0.0);
    for (int j = 0; j < n; j++) Fc[j] = p.q[j];
    Fc[tVar] = 0.5;

    Data *d = (Data *)calloc(1, sizeof(Data));
    d->m = mF;
    d->n = nF;
    d->b = Fb.data();
    d->c = Fc.data();
    d->A = A;
    d->stgs = (Settings *)calloc(1, sizeof(Settings));
    setDefaultSettings(d);
    d->stgs->eps = epsAbs;
    d->stgs->verbose = verbose;
    d->stgs->max_iters = 1000000;
    d->stgs->normalize = (libScaling == 2) ? 0 : 1; // default: 1
    if (tune.rho >= 0) d->stgs->rho_x = tune.rho;
    if (tune.alpha >= 0) d->stgs->alpha = tune.alpha;
    if (tune.sigma >= 0) d->stgs->scale = tune.sigma;

    Cone *k = (Cone *)calloc(1, sizeof(Cone));
    k->f = 0;
    // every original row (ranged or equality) was split into one-sided rows,
    // all living in the nonneg cone; the SOC epigraph rows follow them.
    k->l = (scs_int)(ineqRows.size());
    scs_int *qarr = (scs_int *)malloc(sizeof(scs_int));
    qarr[0] = qsize;
    k->q = qarr;
    k->qsize = 1;

    Sol *sol = (Sol *)calloc(1, sizeof(Sol));
    Info *info = (Info *)calloc(1, sizeof(Info));
    sol->x = (scs_float *)calloc(nF, sizeof(scs_float));
    sol->y = (scs_float *)calloc(mF, sizeof(scs_float));
    sol->s = (scs_float *)calloc(mF, sizeof(scs_float));

    auto t0 = std::chrono::steady_clock::now();
    scs_int flag = scs(d, k, sol, info);
    auto t1 = std::chrono::steady_clock::now();

    EngineResult res;
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.x.assign(sol->x, sol->x + n);
    // B2 FIX: merge one-sided nonneg-cone duals back to per-original-row
    // multipliers in OSQP sign convention (y>0 = active at upper, y<0 = lower):
    // upper split row (+A x <= ub) contributes +y_i, lower (-A x <= -lb) -> -y_i.
    {
        int m = (int)p.rows.size();
        res.y.assign(m, 0.0);
        for (size_t i = 0; i < ineqSrc.size(); i++)
            res.y[ineqSrc[i]] += ineqSgn[i] * sol->y[i];
    }
    res.iters = (int)info->iter;
    res.priRes = info->resPri;
    res.duaRes = info->resDual;
    res.obj = info->pobj;
    res.status = std::string(info->status);

    free(sol->x); free(sol->y); free(sol->s); free(sol); free(info);
    free(qarr); free(k);
    free(d->stgs); free(d);
    free(A->x); free(A->i); free(A->p); free(A);
    return res;
}
#else
bool scsAvailable() { return false; }
EngineResult scsSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &) {
    return EngineResult{"", "scs not compiled in", 0, 0, 0, 0, 0};
}
#endif
