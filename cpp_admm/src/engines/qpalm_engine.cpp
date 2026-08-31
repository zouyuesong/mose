// qpalm_engine.cpp - QPALM 1.x C API backend (proximal ALM + semismooth Newton)
// QPALM takes ladel_sparse matrices; we build them from our row lists.
#include "engine.hpp"
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <string>
// match the qpalm library build configuration (USE_LADEL + long indices)
#define USE_LADEL 1
#define DLONG 1
#include "qpalm/ladel.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "qpalm/qpalm.h"
#ifdef __cplusplus
}
#endif

#ifdef USE_QPALM
bool qpalmAvailable() { return true; }

// build a ladel sparse matrix in CSC from (rows, m, n)
static ladel_sparse_matrix *toCsc(const Problem &p, int n, int m, bool isP) {
    // gather triplets
    std::vector<int> ti, tj;
    std::vector<double> tv;
    if (isP) {
        for (int j = 0; j < n; j++)
            if (p.pDiag[j] != 0) {
                ti.push_back(j);   // row
                tj.push_back(j);   // col
                tv.push_back(p.pDiag[j]);
            }
    } else {
        for (int r = 0; r < m; r++)
            for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
                ti.push_back(r);                    // row = constraint index
                tj.push_back(p.rows[r].idx[t]);     // col = variable index
                tv.push_back(p.rows[r].val[t]);
            }
    }
    // CSC by column, rows sorted within each column
    std::vector<int> cnt(n + 1, 0);
    for (auto c : tj) cnt[c + 1]++;
    for (int j = 0; j < n; j++) cnt[j + 1] += cnt[j];
    std::vector<int> pos(tv.size());
    {
        std::vector<std::vector<int>> buckets(n);
        for (size_t k = 0; k < tv.size(); k++) buckets[tj[k]].push_back((int)k);
        int cur = 0;
        for (int j = 0; j < n; j++) {
            // sort bucket entries by row index ti
            std::sort(buckets[j].begin(), buckets[j].end(),
                      [&](int a, int b) { return ti[a] < ti[b]; });
            for (int k : buckets[j]) pos[cur++] = k;
        }
    }
    ladel_int *col = (ladel_int *)malloc((n + 1) * sizeof(ladel_int));
    ladel_int *row = (ladel_int *)malloc(std::max<size_t>(1, tv.size()) * sizeof(ladel_int));
    ladel_double *val = (ladel_double *)malloc(std::max<size_t>(1, tv.size()) * sizeof(ladel_double));
    for (int j = 0; j <= n; j++) col[j] = cnt[j];
    for (size_t k = 0; k < tv.size(); k++) {
        row[k] = ti[pos[k]];
        val[k] = tv[pos[k]];
    }
    int symmetry = isP ? UPPER : UNSYMMETRIC;
    ladel_sparse_matrix *M = ladel_sparse_alloc(m, n, (ladel_int)tv.size(), symmetry, TRUE, FALSE);
    // copy into the allocated matrix
    for (int j = 0; j <= n; j++) M->p[j] = col[j];
    for (size_t k = 0; k < tv.size(); k++) {
        M->i[k] = row[k];
        M->x[k] = val[k];
    }
    M->nzmax = (ladel_int)tv.size();
    free(col); free(row); free(val);
    return M;
}

EngineResult qpalmSolve(const Problem &p, double epsAbs, double epsRel, bool verbose,
                        bool, int libScaling, const EngineTuning &tune) {
    int n = p.n, m = (int)p.rows.size();
    ladel_sparse_matrix *Q = toCsc(p, n, n, true);   // n x n upper
    ladel_sparse_matrix *A = toCsc(p, n, m, false);  // m x n

    std::vector<c_float> q(p.q.begin(), p.q.end());
    std::vector<c_float> bmin(m), bmax(m);
    for (int r = 0; r < m; r++) {
        bmin[r] = p.rows[r].lb;
        bmax[r] = p.rows[r].ub;
    }
    QPALMData data;
    data.n = n;
    data.m = m;
    data.Q = Q;
    data.A = A;
    data.q = q.data();
    data.c = 0;
    data.bmin = bmin.data();
    data.bmax = bmax.data();

    QPALMSettings *settings = (QPALMSettings *)calloc(1, sizeof(QPALMSettings));
    qpalm_set_default_settings(settings);
    settings->eps_abs = epsAbs;
    settings->eps_rel = epsRel;
    settings->verbose = verbose;
    settings->max_iter = 100000;
    // baseline kept scaling=0; libScaling==1 turns the built-in Ruiz on
    settings->scaling = (libScaling == 1) ? 10 : 0;
    if (tune.sigma >= 0) settings->sigma_init = tune.sigma;

    double setupMsLoc = -1;
    auto ts0 = std::chrono::steady_clock::now();
    QPALMWorkspace *work = qpalm_setup(&data, settings);
    auto ts1 = std::chrono::steady_clock::now();
    setupMsLoc =
        std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    EngineResult res;
    res.setupMs = setupMsLoc;
    if (!work) {
        res.status = "setup_error";
        return res;
    }
    auto t0 = std::chrono::steady_clock::now();
    qpalm_solve(work);
    auto t1 = std::chrono::steady_clock::now();
    res.solveMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    res.x.assign(work->solution->x, work->solution->x + n);
    res.y.assign(work->solution->y, work->solution->y + m);

    res.iters = (int)work->info->iter;
    res.priRes = work->info->pri_res_norm;
    res.duaRes = work->info->dua_res_norm;
    res.obj = work->info->objective;
    std::string st = work->info->status;
    res.status = st.empty() ? "unknown" : st;

    qpalm_cleanup(work);
    free(settings);
    ladel_sparse_free(Q);
    ladel_sparse_free(A);
    return res;
}
#else
bool qpalmAvailable() { return false; }
EngineResult qpalmSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &) {
    return EngineResult{"", "qpalm not compiled in", 0, 0, 0, 0, 0};
}
#endif
