// model.cpp - implementation, see model.hpp
#include "model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

static const double INF = std::numeric_limits<double>::infinity();
static const double EPS = 1e-10;

static std::string upper(std::string s) {
    for (auto &c : s)
        if (c >= 'a' && c <= 'z') c -= 32;
    return s;
}

InputsCols toColumns(std::vector<InputItem> &inputs) {
    InputsCols c;
    size_t n = inputs.size();
    c.univ.resize(n);
    c.position.resize(n); c.alpha.resize(n); c.tradingCost.resize(n);
    c.minTrade.resize(n); c.maxTrade.resize(n);
    c.minPos.resize(n); c.maxPos.resize(n);
    c.lambda.resize(n); c.tradingLambda.resize(n);
    for (size_t i = 0; i < n; i++) {
        c.univ[i] = inputs[i].sym;
        c.position[i] = inputs[i].position;
        c.alpha[i] = inputs[i].alpha;
        c.tradingCost[i] = inputs[i].cost;
        c.minTrade[i] = inputs[i].minTrade;
        c.maxTrade[i] = inputs[i].maxTrade;
        c.minPos[i] = inputs[i].minPosition;
        c.maxPos[i] = inputs[i].maxPosition;
        c.lambda[i] = inputs[i].lambda;
        c.tradingLambda[i] = inputs[i].tlambda;
    }
    return c;
}

std::vector<AssembledConstraint> assembleConstraints(
    const std::vector<std::string> &univ,
    const std::vector<ConstraintItem> &constraints,
    const std::vector<WeightItem> &weights) {
    // weights -> name -> (sym -> weight)
    std::map<std::string, std::map<std::string, double>> tab;
    for (auto &w : weights) tab[w.name][w.sym] = w.weight;
    std::vector<AssembledConstraint> out(constraints.size());
    std::map<std::string, int> idxOf;
    for (size_t i = 0; i < univ.size(); i++) idxOf[univ[i]] = (int)i;
    for (size_t i = 0; i < constraints.size(); i++) {
        out[i].item = &constraints[i];
        auto it = tab.find(constraints[i].name);
        if (it == tab.end()) continue;
        for (size_t j = 0; j < univ.size(); j++) {
            auto w = it->second.find(univ[j]);
            if (w != it->second.end()) {
                out[i].varIndices.push_back((int)j);
                out[i].weights.push_back(w->second);
            }
        }
    }
    return out;
}

bool breachWithZeroTraded(const InputsCols &cols,
                          const std::vector<AssembledConstraint> &det) {
    for (auto &con : det) {
        std::string t = upper(con.item->type);
        if (t == "NET") {
            double adj = 0;
            for (size_t k = 0; k < con.varIndices.size(); k++)
                adj += con.weights[k] * cols.position[con.varIndices[k]];
            if (con.item->lb - adj > 0 || con.item->ub - adj < 0) return true;
        } else if (t == "GROSS") {
            double z = 0;
            for (size_t k = 0; k < con.varIndices.size(); k++)
                z += std::fabs(cols.position[con.varIndices[k]]) * con.weights[k];
            if (con.item->ub < z) return true;
        }
    }
    return false;
}

void step1Relax(std::vector<InputItem> &inputs) {
    for (auto &i : inputs) {
        if (i.position + i.minTrade > i.maxPosition)
            i.maxPosition = i.position + i.minTrade + EPS;
        if (i.position + i.maxTrade < i.minPosition)
            i.minPosition = i.position + i.maxTrade - EPS;
    }
}

Problem buildProblem(const InputsCols &cols,
                     const std::vector<AssembledConstraint> &det,
                     bool relaxMode, double elasticPenalty) {
    int n = (int)cols.univ.size();
    Problem p;
    p.numSymbols = n;

    // fixed-trade detection (backend_admm.go:147-154)
    std::vector<double> xl(n), xu(n);
    p.fixed.resize(n);
    p.xFix.assign(n, 0.0);
    for (int i = 0; i < n; i++) {
        xl[i] = std::max(cols.minTrade[i], cols.minPos[i] - cols.position[i]);
        xu[i] = std::min(cols.maxTrade[i], cols.maxPos[i] - cols.position[i]);
        if (xu[i] - xl[i] <= 1e-5 * std::max(1.0, std::fabs(xl[i]) + std::fabs(xu[i]))) {
            p.fixed[i] = true;
            p.xFix[i] = std::max(std::min(0.0, xu[i]), xl[i]);
        }
    }
    std::vector<int> idxOf(n, -1);
    for (int i = 0; i < n; i++)
        if (!p.fixed[i]) {
            idxOf[i] = (int)p.active.size();
            p.active.push_back(i);
        }
    int na = (int)p.active.size();

    // variable layout: x [0,na), u [na,2na), g afterwards, t at tail
    int tubBase = na;
    std::map<int, int> tradeToGUB; // reduced idx -> g var idx (-1 placeholder first)
    for (auto &con : det) {
        if (upper(con.item->type) == "GROSS") {
            for (auto ti : con.varIndices) {
                if (!p.fixed[ti] && !tradeToGUB.count(idxOf[ti]))
                    tradeToGUB[idxOf[ti]] = -1;
            }
        }
    }
    int gubBase = 2 * na;
    int next = gubBase;
    for (auto &con : det) {
        if (upper(con.item->type) == "GROSS") {
            for (auto ti : con.varIndices) {
                if (p.fixed[ti]) continue;
                int ni = idxOf[ti];
                if (tradeToGUB[ni] < 0) {
                    tradeToGUB[ni] = next;
                    next++;
                }
            }
        }
    }
    int numCore = next; // x + u + g

    p.n = numCore;
    p.pDiag.assign(numCore, 0.0);
    p.q.assign(numCore, 0.0);
    p.varScale.assign(numCore, 1.0);
    for (int a = 0; a < na; a++) {
        int i = p.active[a];
        p.pDiag[a] = cols.lambda[i] + cols.tradingLambda[i];
        p.q[a] = cols.lambda[i] * cols.position[i] - cols.alpha[i];
        p.q[tubBase + a] = cols.tradingCost[i];
    }

    auto addRow = [&](std::vector<int> idx, std::vector<double> val,
                      double lb, double ub, double sigma) {
        p.rows.push_back(Row{std::move(idx), std::move(val), lb, ub});
        p.rowScale.push_back(sigma);
    };

    // x variable box (single row suffices for libraries supporting ranges)
    std::vector<double> band(na, 1.0);
    for (int a = 0; a < na; a++) {
        int i = p.active[a];
        band[a] = std::max(1.0, std::max(std::fabs(xl[i]), std::fabs(xu[i])));
        p.varScale[a] = band[a];
        p.varScale[tubBase + a] = band[a];
        addRow({a}, {1.0}, xl[i], xu[i], band[a]);
    }
    // TUB: u + x >= 0, u - x >= 0
    for (int a = 0; a < na; a++) {
        addRow({a, tubBase + a}, {1.0, 1.0}, 0.0, INF, 2.0 * band[a]);
        addRow({a, tubBase + a}, {-1.0, 1.0}, 0.0, INF, 2.0 * band[a]);
    }
    // GUB: g - x >= pos, g + x >= -pos
    for (auto &kv : tradeToGUB) {
        int i = p.active[kv.first];
        double sigma = std::max(1.0, std::fabs(cols.position[i]) + band[kv.first]);
        p.varScale[kv.second] = sigma;
        addRow({kv.first, kv.second}, {-1.0, 1.0}, cols.position[i], INF, sigma);
        addRow({kv.first, kv.second}, {1.0, 1.0}, -cols.position[i], INF, sigma);
    }

    // general constraints
    p.generalStart = (int)p.rows.size();
    int relaxNext = next;
    auto activeRow = [&](const std::vector<int> &idxs,
                         const std::vector<double> &ws,
                         std::vector<int> &ri, std::vector<double> &rv,
                         double &fixSum) {
        fixSum = 0;
        for (size_t k = 0; k < idxs.size(); k++) {
            if (p.fixed[idxs[k]]) {
                fixSum += ws[k] * p.xFix[idxs[k]];
            } else {
                ri.push_back(idxOf[idxs[k]]);
                rv.push_back(ws[k]);
            }
        }
    };
    for (auto &con : det) {
        std::string t = upper(con.item->type);
        if (t == "NET") {
            double adj = 0;
            for (size_t k = 0; k < con.varIndices.size(); k++)
                adj += con.weights[k] * cols.position[con.varIndices[k]];
            std::vector<int> ri;
            std::vector<double> rv;
            double fixSum;
            activeRow(con.varIndices, con.weights, ri, rv, fixSum);
            double lb = con.item->lb - adj - fixSum;
            double ub = con.item->ub - adj - fixSum;
            bool breach = con.item->lb - adj > 0 || con.item->ub - adj < 0;
            double netSigma =
                std::max(1.0, std::max(std::fabs(con.item->lb), std::fabs(con.item->ub)));
            static bool foldRelax = [] {
                return getenv("RELAX_FOLD") != nullptr;
            }();
            if (relaxMode && breach && foldRelax) {
                // Lew folding (arXiv 2511.08451): no t_k variable, no extra
                // rows; the ORIGINAL bounds become the hinge reference and
                // the row carries the elastic penalty (three-piece prox in
                // engines that honor Row::hinge; identical optimum since
                // min_t c*t s.t. dist(w'x,[lb,ub]) <= t == c*dist)
                addRow(ri, rv, lb, ub, netSigma);
                p.rows.back().hinge = elasticPenalty;
            } else {
            if (relaxMode && breach) {
                if (lb > 0) lb = -EPS;
                if (ub < 0) ub = EPS;
            }
            addRow(ri, rv, lb, ub, netSigma);
            if (relaxMode && breach) {
                if (con.item->lb > adj) { // finalConstraint + t >= LB
                    int ti = relaxNext++;
                    p.pDiag.push_back(0);
                    p.q.push_back(elasticPenalty);
                    p.n++;
                    p.varScale.push_back(1.0);
                    addRow({ti}, {1.0}, 0.0, INF, 1.0);
                    std::vector<int> i2 = ri; i2.push_back(ti);
                    std::vector<double> v2 = rv; v2.push_back(1.0);
                    addRow(i2, v2, con.item->lb - adj - fixSum, INF, netSigma);
                }
                if (con.item->ub < adj) { // finalConstraint - t <= UB
                    int ti = relaxNext++;
                    p.pDiag.push_back(0);
                    p.q.push_back(elasticPenalty);
                    p.n++;
                    p.varScale.push_back(1.0);
                    addRow({ti}, {1.0}, 0.0, INF, 1.0);
                    std::vector<int> i2 = ri; i2.push_back(ti);
                    std::vector<double> v2 = rv; v2.push_back(-1.0);
                    addRow(i2, v2, -INF, con.item->ub - adj - fixSum, netSigma);
                }
            }
            }
        } else if (t == "NET_TRADE") {
            std::vector<int> ri;
            std::vector<double> rv;
            double fixSum;
            activeRow(con.varIndices, con.weights, ri, rv, fixSum);
            double netSigma =
                std::max(1.0, std::max(std::fabs(con.item->lb), std::fabs(con.item->ub)));
            addRow(ri, rv, con.item->lb - fixSum, con.item->ub - fixSum, netSigma);
        } else if (t == "GROSS") {
            std::vector<int> gi;
            std::vector<double> gw;
            double zeroTradedGross = 0, fixGross = 0;
            for (size_t k = 0; k < con.varIndices.size(); k++) {
                int ti = con.varIndices[k];
                zeroTradedGross += std::fabs(cols.position[ti]) * con.weights[k];
                if (p.fixed[ti]) {
                    fixGross += con.weights[k] *
                                std::fabs(cols.position[ti] + p.xFix[ti]);
                } else {
                    gi.push_back(tradeToGUB[idxOf[ti]]);
                    gw.push_back(con.weights[k]);
                }
            }
            if (con.item->lb > 0)
                std::printf("gross constraint %s lower bound(%g) > 0 - not honored\n",
                            con.item->name.c_str(), con.item->lb);
            double ub = con.item->ub - fixGross;
            bool breach = con.item->ub < zeroTradedGross;
            double grossSigma = std::max(1.0, std::fabs(con.item->ub));
            static bool foldRelaxG = [] {
                return getenv("RELAX_FOLD") != nullptr;
            }();
            if (relaxMode && breach && foldRelaxG) {
                addRow(gi, gw, 0.0, ub, grossSigma);
                p.rows.back().hinge = elasticPenalty;
            } else {
            if (relaxMode && breach) ub = zeroTradedGross + EPS - fixGross;
            addRow(gi, gw, 0.0, ub, grossSigma);
            if (relaxMode && breach) {
                int ti = relaxNext++;
                p.pDiag.push_back(0);
                p.q.push_back(elasticPenalty);
                p.n++;
                addRow({ti}, {1.0}, 0.0, INF, 1.0);
                std::vector<int> i2 = gi; i2.push_back(ti);
                std::vector<double> v2 = gw; v2.push_back(-1.0);
                addRow(i2, v2, -INF, con.item->ub - fixGross, grossSigma);
            }
            }
        } else if (t == "GROSS_TRADE") {
            std::vector<int> ui;
            std::vector<double> uw;
            double fixTrade = 0;
            for (size_t k = 0; k < con.varIndices.size(); k++) {
                int ti = con.varIndices[k];
                if (p.fixed[ti]) {
                    fixTrade += con.weights[k] * std::fabs(p.xFix[ti]);
                } else {
                    ui.push_back(tubBase + idxOf[ti]);
                    uw.push_back(con.weights[k]);
                }
            }
            if (con.item->lb > 0)
                std::printf("gross traded constraint %s lower bound(%g) > 0 - not honored\n",
                            con.item->name.c_str(), con.item->lb);
            addRow(ui, uw, 0.0, con.item->ub - fixTrade,
                   std::max(1.0, std::fabs(con.item->ub)));
        } else {
            std::fprintf(stderr, "unsupported constraint type: %s (%s)\n",
                         con.item->type.c_str(), con.item->name.c_str());
            std::exit(1);
        }
    }
    return p;
}

// Divide every row's A coefficients and bounds by its rowScale entry so that
// a solver's global eps_rel becomes "x% of each row's own scale" (=x% of the
// trade band for per-stock rows) instead of x% of the largest row (~5e7 gross).
// Feasible set / optimum unchanged; only row conditioning changes.
// NOTE: also column-scales by varScale (and transforms P/q) so the scaled
// system is fully equilibrated (rows ~O(1), columns ~O(1)); otherwise the
// solvers' infeasibility certificates misfire on the near-singular rows.
// Caller must un-scale x:  x[j] *= varScale[j].
void scaleRows(Problem &p) {
    for (int j = 0; j < p.n; j++) {
        double v = (j < (int)p.varScale.size() && p.varScale[j] > 0)
                       ? p.varScale[j] : 1.0;
        p.pDiag[j] *= v * v;
        p.q[j] *= v;
    }
    for (size_t r = 0; r < p.rows.size(); r++) {
        double s = (r < p.rowScale.size() && p.rowScale[r] > 0) ? p.rowScale[r] : 1.0;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
            int j = p.rows[r].idx[t];
            double v = (j < (int)p.varScale.size() && p.varScale[j] > 0)
                        ? p.varScale[j] : 1.0;
            p.rows[r].val[t] *= v;
        }
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) p.rows[r].val[t] /= s;
        if (std::isfinite(p.rows[r].lb)) p.rows[r].lb /= s;
        if (std::isfinite(p.rows[r].ub)) p.rows[r].ub /= s;
        // elastic penalty lives on row violation: c_eff = c * s (violation
        // shrinks by s under row scaling, so the penalty must grow by s)
        p.rows[r].hinge *= s;
    }
}

// ---- admm.md §3.4(a) preprocessing series, harness-level ----

// apply one simultaneous (row scale s[], col scale v[]) diagonal transform:
//   A'_rj = A_rj * v_j / s_r ; lb'/ub' = (lb,ub)/s_r ;
//   pDiag *= v^2 ; q *= v ; varScale *= v (caller un-scales x).
static void applyDiagScale(Problem &p, const std::vector<double> &s,
                           const std::vector<double> &v) {
    int m = (int)p.rows.size();
    for (int j = 0; j < p.n; j++) {
        double vj = v[j];
        p.pDiag[j] *= vj * vj;
        p.q[j] *= vj;
        if (j < (int)p.varScale.size()) p.varScale[j] *= vj;
        else p.varScale.push_back(vj);
    }
    for (int r = 0; r < m; r++) {
        double sr = s[r];
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            p.rows[r].val[t] *= v[p.rows[r].idx[t]] / sr;
        if (std::isfinite(p.rows[r].lb)) p.rows[r].lb /= sr;
        if (std::isfinite(p.rows[r].ub)) p.rows[r].ub /= sr;
    }
}

void applyRuiz(Problem &p, int iters) {
    int m = (int)p.rows.size();
    std::vector<double> s(m, 1.0), v(p.n, 1.0);
    for (int it = 0; it < iters; it++) {
        // row inf-norm and col inf-norm of the CURRENT matrix
        std::vector<double> rowNorm(m, 0.0), colNorm(p.n, 0.0);
        for (int r = 0; r < m; r++)
            for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
                double a = std::fabs(p.rows[r].val[t]);
                if (a > rowNorm[r]) rowNorm[r] = a;
                int j = p.rows[r].idx[t];
                if (a > colNorm[j]) colNorm[j] = a;
            }
        for (int r = 0; r < m; r++)
            s[r] = rowNorm[r] > 0 ? std::sqrt(rowNorm[r]) : 1.0;
        for (int j = 0; j < p.n; j++)
            v[j] = colNorm[j] > 0 ? std::sqrt(colNorm[j]) : 1.0;
        applyDiagScale(p, s, v);
    }
}

void applyPockChambolle(Problem &p) { // alpha = 1: sqrt of l1 norms, single step
    int m = (int)p.rows.size();
    std::vector<double> s(m, 1.0), v(p.n, 1.0);
    std::vector<double> rowL1(m, 0.0), colL1(p.n, 0.0);
    for (int r = 0; r < m; r++)
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
            double a = std::fabs(p.rows[r].val[t]);
            rowL1[r] += a;
            colL1[p.rows[r].idx[t]] += a;
        }
    for (int r = 0; r < m; r++) s[r] = rowL1[r] > 0 ? std::sqrt(rowL1[r]) : 1.0;
    for (int j = 0; j < p.n; j++) v[j] = colL1[j] > 0 ? std::sqrt(colL1[j]) : 1.0;
    applyDiagScale(p, s, v);
}

// ---- admm.md §3.4(f) harness-level polish ----
// Recover every P=0 auxiliary variable (u/g/t) from its structure rows based
// on the given x-part: u must be |x|, g must be |pos+x| at any sensible
// vertex. Applying this to BOTH the input iterate and the polished candidate
// puts the verify objective on the TRUE cost|x| semantics (the raw ADMM
// stopping point carries slack u > |x| which would otherwise bias the
// acceptance threshold).
static void polishRecoverAux(const Problem &p, const std::vector<int> &kind,
                             std::vector<double> &v) {
    int n = p.n, m = (int)p.rows.size();
    std::vector<char> done(n, 0);
    for (int r = 0; r < m; r++) {
        if (!kind[r] || p.rows[r].idx.size() != 2) continue;
        int j1 = p.rows[r].idx[0], j2 = p.rows[r].idx[1];
        if ((p.pDiag[j1] == 0) == (p.pDiag[j2] == 0)) continue;
        int zj = p.pDiag[j1] == 0 ? j1 : j2;
        int pj = p.pDiag[j1] == 0 ? j2 : j1;
        double cz = p.pDiag[j1] == 0 ? p.rows[r].val[0] : p.rows[r].val[1];
        double cp = p.pDiag[j1] == 0 ? p.rows[r].val[1] : p.rows[r].val[0];
        double b = kind[r] == 1 ? p.rows[r].lb : p.rows[r].ub;
        if (cz != 0 && !done[zj]) {
            v[zj] = (b - cp * v[pj]) / cz;
            done[zj] = 1;
        }
    }
    for (int j = 0; j < n; j++) {
        if (p.pDiag[j] != 0 || done[j]) continue;
        double best = 0; // t-style default (lb 0)
        for (int r = 0; r < m; r++) {
            if (p.rows[r].idx.size() != 2) continue;
            int j1 = p.rows[r].idx[0], j2 = p.rows[r].idx[1];
            if (j1 != j && j2 != j) continue;
            int pj = (j1 == j) ? j2 : j1;
            if (p.pDiag[pj] == 0) continue;
            double cself = (j1 == j) ? p.rows[r].val[0] : p.rows[r].val[1];
            double cpar = (j1 == j) ? p.rows[r].val[1] : p.rows[r].val[0];
            if (cself > 0)
                best = std::max(best, (p.rows[r].lb - cpar * v[pj]) / cself);
        }
        v[j] = best;
        done[j] = 1;
    }
}

// shared acceptance check: feasibility on EVERY row + objective not worse;
// on success overwrite x with the candidate.
static bool polishVerify(const Problem &p, std::vector<double> &x,
                         const std::vector<double> &cand, int dbg) {
    int n = p.n, m = (int)p.rows.size();
    auto objMin = [&](const std::vector<double> &z) {
        double o = 0;
        for (int j = 0; j < n; j++) o += 0.5 * p.pDiag[j] * z[j] * z[j] + p.q[j] * z[j];
        return o;
    };
    bool feas = true;
    for (int r = 0; r < m && feas; r++) {
        double a = 0;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            a += p.rows[r].val[t] * cand[p.rows[r].idx[t]];
        double tl = 1e-6 * std::max(1.0, std::fabs(p.rows[r].lb));
        double tu = 1e-6 * std::max(1.0, std::fabs(p.rows[r].ub));
        if (std::isfinite(p.rows[r].lb) && a < p.rows[r].lb - tl) feas = false;
        if (std::isfinite(p.rows[r].ub) && a > p.rows[r].ub + tu) feas = false;
    }
    if (!feas) {
        // top violators for diagnosis
        std::vector<std::pair<double,int>> v;
        for (int r = 0; r < m; r++) {
            double a = 0;
            for (size_t t = 0; t < p.rows[r].idx.size(); t++)
                a += p.rows[r].val[t] * cand[p.rows[r].idx[t]];
            double tl = 1e-6 * std::max(1.0, std::fabs(p.rows[r].lb));
            double tu = 1e-6 * std::max(1.0, std::fabs(p.rows[r].ub));
            double viol = 0;
            if (std::isfinite(p.rows[r].lb) && a < p.rows[r].lb - tl)
                viol = p.rows[r].lb - a;
            if (std::isfinite(p.rows[r].ub) && a > p.rows[r].ub + tu)
                viol = a - p.rows[r].ub;
            if (viol > 0) v.push_back({viol, r});
        }
        std::sort(v.begin(), v.end(), std::greater<>());
        fprintf(stderr, "[polish] reject: infeasible, top violators:\n");
        for (size_t i = 0; i < v.size() && i < 6; i++) {
            int r = v[i].second;
            fprintf(stderr, "  row %d nnz=%zu viol=%.3e lb=%.3g ub=%.3g firstvar=%d\n",
                    r, p.rows[r].idx.size(), v[i].first, p.rows[r].lb, p.rows[r].ub,
                    p.rows[r].idx.empty() ? -1 : p.rows[r].idx[0]);
        }
        return false;
    }
    double oc = objMin(cand), ox = objMin(x);
    if (dbg > 1) fprintf(stderr, "[polish] verify objMin cand=%.6f input=%.6f\n", oc, ox);
    if (oc > ox + 1e-9 * (1.0 + std::fabs(ox))) {
        if (dbg) fprintf(stderr, "[polish] reject: obj %.6f > %.6f (worse by %.3g)\n", oc, ox, oc-ox);
        return false;
    }
    x = cand;
    return true;
}

bool polishKKT(Problem &p, std::vector<double> &x, int irIters,
               const std::vector<double> *y, std::vector<double> *yOut) {
#if defined(USE_OSQP) || defined(USE_QPALM) || defined(USE_SCS) || defined(USE_PROXQP)
    // (always compiled when any engine exists -> Eigen available)
#endif
    int n = p.n, m = (int)p.rows.size();
    if ((int)x.size() != n) return false;
    const double delta = 1e-6;
    const double epsReg = 1e-10;
    // row activities at the stopping point
    std::vector<double> act(m, 0.0);
    for (int r = 0; r < m; r++)
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            act[r] += p.rows[r].val[t] * x[p.rows[r].idx[t]];
    // classify: prefer the multiplier criterion (active set truth) when the
    // engine returned y (sign of y_r says WHICH side; magnitude says WHETHER);
    // fall back to pure bound proximity otherwise. For two-sided rows stopped
    // mid-interval the proximity-only rule misclassifies -> infeasible polish.
    std::vector<int> kind(m, 0); // 0 free, 1 at lower, 2 at upper
    int k = 0;
    bool haveY = y && y->size() == (size_t)m;
    for (int r = 0; r < m; r++) {
        const Row &rw = p.rows[r];
        double tl = 1e-6 * std::max(1.0, std::fabs(rw.lb));
        double tu = 1e-6 * std::max(1.0, std::fabs(rw.ub));
        bool low = std::isfinite(rw.lb) && act[r] - rw.lb <= tl;
        bool upp = std::isfinite(rw.ub) && rw.ub - act[r] <= tu;
        if (haveY) {
            // loose proximity (the ADMM point is only ~eps-feasible): trust
            // the multiplier for the side, proximity (1e-3 rel) for the gate
            double yr = (*y)[r];
            double gl = 1e-3 * std::max(1.0, std::fabs(rw.lb));
            double gu = 1e-3 * std::max(1.0, std::fabs(rw.ub));
            bool nearL = std::isfinite(rw.lb) && act[r] - rw.lb <= gl;
            bool nearU = std::isfinite(rw.ub) && rw.ub - act[r] <= gu;
            if (yr < -1e-8 && nearL) { kind[r] = 1; k++; }
            else if (yr > 1e-8 && nearU) { kind[r] = 2; k++; }
        } else {
            if (low) { kind[r] = 1; k++; }
            else if (upp) { kind[r] = 2; k++; }
        }
    }
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("POLISH_DEBUG") ? 1 : 0;
    static int solverMode = -1; // 0 = schur (default), 1 = full KKT LU
    if (solverMode < 0) {
        const char *sm = getenv("POLISH_SOLVER");
        solverMode = (sm && strcmp(sm, "schur") == 0) ? 0 : 1; // LU default
    }
    if (dbg) fprintf(stderr, "[polish] k=%d of m=%d solver=%s\n", k, m,
                     solverMode ? "lu" : "schur");

    // ---- path A (default): Schur complement on y ----
    // K = [[D, A'],[A, -eps I]] with D = P + delta*I DIAGONAL, so
    //   x  = D^-1 (-q - A'y)                       (exact, diagonal)
    //   S y = -A D^-1 q - b =: c,  S = A D^-1 A' + eps I   (SPD -> Cholesky)
    // S is assembled per variable: active rows sharing variable j contribute
    // pairwise products; structure rows (box/TUB/GUB) give ~5x5 blocks per
    // symbol, only the few TIGHT general rows add a small dense block.
    // O(sum_j |L_j|^2) which is tiny since general rows are rarely active.
    if (solverMode == 0) {
        std::vector<double> D(n);
        for (int j = 0; j < n; j++) D[j] = p.pDiag[j] + delta;
        std::vector<int> rowToK(m, -1);
        {
            int kr = 0;
            for (int r = 0; r < m; r++)
                if (kind[r]) rowToK[r] = kr++;
        }
        // active rows per variable
        std::vector<std::vector<std::pair<int, double>>> varRows(n);
        for (int r = 0; r < m; r++) {
            if (!kind[r]) continue;
            for (size_t t = 0; t < p.rows[r].idx.size(); t++)
                varRows[p.rows[r].idx[t]].push_back({r, p.rows[r].val[t]});
        }
        Eigen::VectorXd c(k);
        std::vector<Eigen::Triplet<double>> trip;
        for (int r = 0; r < m; r++) {
            if (!kind[r]) continue;
            double b = kind[r] == 1 ? p.rows[r].lb : p.rows[r].ub;
            double adjq = 0;
            for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
                int j = p.rows[r].idx[t];
                adjq += p.rows[r].val[t] * (-p.q[j]) / D[j];
            }
            c[rowToK[r]] = adjq - b;
        }
        for (int j = 0; j < n; j++) {
            auto &L = varRows[j];
            size_t sz = L.size();
            for (size_t a = 0; a < sz; a++)
                for (size_t b2 = a; b2 < sz; b2++) {
                    int ra = rowToK[L[a].first], rb = rowToK[L[b2].first];
                    double v = L[a].second * L[b2].second / D[j];
                    if (ra == rb) trip.push_back({ra, rb, v});
                    else { // store BOTH triangles: SimplicialLDLT reads one
                        trip.push_back({ra, rb, v});
                        trip.push_back({rb, ra, v});
                    }
                }
        }
        // Build pair products once (no eps on the diagonal yet).
        Eigen::SparseMatrix<double> S0(k, k);
        S0.setFromTriplets(trip.begin(), trip.end());
        // diagonal equilibration: S has huge diag spread (u/g vars contribute
        // val^2/delta ~ 2.5e5 while x vars with large P contribute ~1e-4);
        // pivot-free Cholesky on the raw matrix loses accuracy. Solve
        //   Ds^-1 S Ds^-1 z = Ds^-1 c ,  y = Ds^-1 z .
        Eigen::VectorXd ds = S0.diagonal();
        for (int i = 0; i < k; i++) ds[i] = ds[i] > 0 ? std::sqrt(ds[i]) : 1.0;
        Eigen::VectorXd cs = c.cwiseQuotient(ds);
        // shifted Cholesky with equilibration: dependent active rows (duplicate
        // delta rows, TUB/GUB kink pairs) give S a near-null space, so plain
        // epsReg can still trip a non-positive pivot; retry with eps*100.
        // equilibrated pair-only system Sb = Ds^-1 S0 Ds^-1 (NO eps): this is
        // the system IR must satisfy; the eps-shifted factorization below only
        // serves as its preconditioner, so the eps perturbation is iterated
        // out instead of being baked into the solution.
        Eigen::SparseMatrix<double> Sb(k, k); // pre-sized (Eigen never resizes)
        {
            std::vector<Eigen::Triplet<double>> ts;
            ts.reserve(trip.size());
            for (auto &t : trip) {
                double v = t.value() / (ds[t.row()] * ds[t.col()]);
                ts.push_back({t.row(), t.col(), v});
            }
            Sb.setFromTriplets(ts.begin(), ts.end());
        }
        // factorize Sb + epsI (equilibrated) - pivoted LU on the reduced
        // system: robust where pivot-free Cholesky trips on near-null modes
        // (duplicate delta rows / TUB-GUB kink pairs), still k x k (half the
        // size of the full KKT) and diagonally equilibrated.
        Eigen::SparseMatrix<double> Sa(k, k);
        {
            std::vector<Eigen::Triplet<double>> ts;
            ts.reserve(trip.size() + k);
            for (auto &t : trip) {
                double v = t.value() / (ds[t.row()] * ds[t.col()]);
                ts.push_back({t.row(), t.col(), v});
            }
            // uniform shift in EQUILIBRATED space (diag(Sb) == 1 exactly):
            // a per-row epsReg/ds^2 would drop below machine precision for
            // large-diagonal rows (u/g) and blow up for small ones.
            for (int i = 0; i < k; i++)
                ts.push_back({i, i, 1e-6}); // shift
            Sa.setFromTriplets(ts.begin(), ts.end());
        }
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> schurLU;
        schurLU.compute(Sa);
        bool okPath = schurLU.info() == Eigen::Success;
        Eigen::VectorXd y(k);
        if (okPath) {
            y = schurLU.solve(cs).cwiseQuotient(ds);
            okPath = schurLU.info() == Eigen::Success;
        }
        if (okPath && dbg) {
            double r0 = (cs - Sa * y.cwiseProduct(ds)).cwiseAbs().maxCoeff();
            fprintf(stderr, "[polish] schur residual0=%.2e\n", r0);
        }
        if (okPath) {
            // IR toward the SHIFTED system: the pure Sb is singular (duplicate
            // rows / kink pairs) and IR toward it drifts y along null modes,
            // leaking into x via the D=delta=1e-6 amplification of u/g vars.
            for (int it = 0; it < irIters; it++) {
                Eigen::VectorXd resd = cs - Sa * (y.cwiseProduct(ds));
                y += schurLU.solve(resd).cwiseQuotient(ds);
            }
            if (dbg) {
                double rmax = (cs - Sa * y.cwiseProduct(ds)).cwiseAbs().maxCoeff();
                fprintf(stderr, "[polish] schur residualIR=%.2e\n", rmax);
            }
            Eigen::VectorXd Aty = Eigen::VectorXd::Zero(n);
            for (int r = 0; r < m; r++) {
                if (!kind[r]) continue;
                double yr = y[rowToK[r]];
                for (size_t t = 0; t < p.rows[r].idx.size(); t++)
                    Aty[p.rows[r].idx[t]] += p.rows[r].val[t] * yr;
            }
            std::vector<double> cand(n);
            for (int j = 0; j < n; j++) cand[j] = (-p.q[j] - Aty[j]) / D[j];
            // Recover the P=0 variables (u/g/t: D = delta = 1e-6) from their
            // TIGHT 2-nnz structure rows instead of the diagonal formula: the
            // formula amplifies multiplier error by 1/D ~ 1e6 (observed TUB
            // violations ~190), while the row recovery is exact - this is what
            // pivoted LU effectively does for these vars.
            {
                std::vector<char> fixed(n, 0);
                for (int r = 0; r < m; r++) {
                    if (!kind[r] || p.rows[r].idx.size() != 2) continue;
                    int j1 = p.rows[r].idx[0], j2 = p.rows[r].idx[1];
                    double c1 = p.rows[r].val[0], c2 = p.rows[r].val[1];
                    double b = kind[r] == 1 ? p.rows[r].lb : p.rows[r].ub;
                    bool z2 = p.pDiag[j2] == 0 && p.pDiag[j1] != 0;
                    bool z1 = p.pDiag[j1] == 0 && p.pDiag[j2] != 0;
                    if (z2 && !fixed[j2] && c2 != 0) {
                        cand[j2] = (b - c1 * cand[j1]) / c2;
                        fixed[j2] = 1;
                    } else if (z1 && !fixed[j1] && c1 != 0) {
                        cand[j1] = (b - c2 * cand[j2]) / c1;
                        fixed[j1] = 1;
                    }
                }
                // P=0 vars with NO tight structure row: clamp to their
                // structural meaning (u=|x|, g=|pos+x| inferred from the
                // 2-nnz rows; t to its bounds) instead of the amplified
                // diagonal formula.
                for (int j = 0; j < n; j++) {
                    if (p.pDiag[j] != 0 || fixed[j]) continue;
                    double best = 0; // t default
                    for (int r = 0; r < m; r++) {
                        if (p.rows[r].idx.size() != 2) continue;
                        int j1 = p.rows[r].idx[0], j2 = p.rows[r].idx[1];
                        if (j1 != j && j2 != j) continue;
                        int partner = (j1 == j) ? j2 : j1;
                        if (p.pDiag[partner] == 0) continue;
                        double cself = (j1 == j) ? p.rows[r].val[0] : p.rows[r].val[1];
                        double cpar = (j1 == j) ? p.rows[r].val[1] : p.rows[r].val[0];
                        double b = p.rows[r].lb; // structural rows are [b, inf)
                        // row: cself*z + cpar*x >= b, tight at z*:
                        // z* = sup over both rows of (b - cpar*x)/cself
                        if (cself > 0)
                            best = std::max(best, (b - cpar * cand[partner]) / cself);
                    }
                    cand[j] = best;
                    fixed[j] = 1;
                }
            }
            {
                bool okP = polishVerify(p, x, cand, dbg);
                if (okP && yOut) {
                    yOut->assign((size_t)m, 0.0);
                    for (int r = 0; r < m; r++)
                        if (kind[r]) (*yOut)[r] = y[rowToK[r]];
                }
                return okP;
            }
        }
        if (dbg) fprintf(stderr, "[polish] schur fail, fallback lu\n");
    }
    // ---- path B: full KKT SparseLU (fallback / cross-check) ----
    // assemble sparse KKT  K = [P+delta*I  A_act'; A_act  0]  (size n+k)
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve((size_t)n + 1);
    for (int j = 0; j < n; j++) trip.push_back({j, j, p.pDiag[j] + delta});
    Eigen::VectorXd rhs(n + k);
    for (int j = 0; j < n; j++) rhs[j] = -p.q[j];
    std::vector<int> rowToKB(m, -1);
    int kr = 0;
    for (int r = 0; r < m; r++) {
        if (!kind[r]) continue;
        double b = kind[r] == 1 ? p.rows[r].lb : p.rows[r].ub;
        rhs[n + kr] = b;
        rowToKB[r] = kr;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++) {
            trip.push_back({n + kr, p.rows[r].idx[t], p.rows[r].val[t]});
            trip.push_back({p.rows[r].idx[t], n + kr, p.rows[r].val[t]});
        }
        kr++;
    }
    // (2,2) block: -epsReg regularization (see top) keeps redundant/dependent
    // active rows (duplicate delta rows, TUB/GUB kink pairs) from making K
    // singular; IR rounds pull the true equality residual back below tol.
    for (int i = 0; i < k; i++) trip.push_back({n + i, n + i, -epsReg});
    Eigen::SparseMatrix<double> K(n + k, n + k);
    K.setFromTriplets(trip.begin(), trip.end());
    if (dbg) fprintf(stderr, "[polish] assembled nnz=%zu\n", K.nonZeros());
    auto tasm = std::chrono::steady_clock::now();
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>> lu;
    lu.compute(K);
    if (dbg) {
        auto tcmp = std::chrono::steady_clock::now();
        fprintf(stderr, "[polish] lu.compute=%.2fms\n",
                std::chrono::duration<double, std::milli>(tcmp - tasm).count());
    }
    if (lu.info() != Eigen::Success) { if (dbg) fprintf(stderr, "[polish] LU compute fail\n"); return false; }
    Eigen::VectorXd sol = lu.solve(rhs);
    if (lu.info() != Eigen::Success) { if (dbg) fprintf(stderr, "[polish] LU solve fail\n"); return false; }
    // iterative refinement on the same factorization
    for (int it = 0; it < irIters; it++) {
        Eigen::VectorXd resd = rhs - K * sol;
        Eigen::VectorXd corr = lu.solve(resd);
        if (lu.info() != Eigen::Success) return false;
        sol += corr;
    }
    std::vector<double> cand(sol.data(), sol.data() + n);
    polishRecoverAux(p, kind, cand);
    std::vector<double> xref = x; // same true-cost semantics for the baseline
    polishRecoverAux(p, kind, xref);
    if (polishVerify(p, xref, cand, dbg)) {
        x = cand; // update the CALLER's vector (polishVerify writes xref only)
        if (yOut) {
            yOut->assign((size_t)m, 0.0);
            for (int r = 0; r < m; r++)
                if (kind[r]) (*yOut)[r] = sol[n + rowToKB[r]];
        }
        return true;
    }
    return false;
}

double dualGapCert(const Problem &p, const std::vector<double> &x,
                   const std::vector<double> &y) {
    int n = p.n, m = (int)p.rows.size();
    if ((int)x.size() != n || (int)y.size() != m) return NAN;
    std::vector<double> g = p.q;
    for (int r = 0; r < m; r++) {
        double yr = y[r];
        if (yr == 0) continue;
        for (size_t t = 0; t < p.rows[r].idx.size(); t++)
            g[p.rows[r].idx[t]] += p.rows[r].val[t] * yr;
    }
    double prim = 0, dual = 0;
    for (int j = 0; j < n; j++) {
        prim += 0.5 * p.pDiag[j] * x[j] * x[j] + p.q[j] * x[j];
        dual += p.pDiag[j] > 0 ? -g[j] * g[j] / (2.0 * p.pDiag[j]) : g[j] * x[j];
    }
    for (int r = 0; r < m; r++) {
        double yr = y[r];
        if (yr > 0) {
            if (std::isfinite(p.rows[r].ub)) dual -= yr * p.rows[r].ub;
        } else if (yr < 0) {
            if (std::isfinite(p.rows[r].lb)) dual -= yr * p.rows[r].lb;
        }
    }
    return prim - dual;
}

double maxObjective(const InputsCols &cols, const Problem &p,
                    const std::vector<double> &xred) {
    // maximize-form objective on trades (same as Go tests use)
    double obj = 0;
    for (int i = 0; i < p.numSymbols; i++) {
        double x = p.fixed[i] ? p.xFix[i] : xred[0]; // placeholder, replaced below
        (void)x;
    }
    // build reduced lookup
    std::vector<int> idxOf(p.numSymbols, -1);
    for (size_t a = 0; a < p.active.size(); a++) idxOf[p.active[a]] = (int)a;
    for (int i = 0; i < p.numSymbols; i++) {
        double x = p.fixed[i] ? p.xFix[i]
                              : (idxOf[i] >= 0 && idxOf[i] < (int)xred.size()
                                     ? xred[idxOf[i]] : 0.0);
        obj += (cols.alpha[i] - cols.lambda[i] * cols.position[i]) * x -
               0.5 * (cols.lambda[i] + cols.tradingLambda[i]) * x * x -
               cols.tradingCost[i] * std::fabs(x);
    }
    return obj;
}
