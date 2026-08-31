// main.cpp - CLI entry, mirrors golang_mosek/src/main/main.go (opt subcommand).
// Usage: csvport_cpp opt -i input.csv -c constraint.csv -w weight.csv
//                    -o output.csv --engine=osqp|qpalm|scs|proxqp [-v]
#include "csv.hpp"
#include "model.hpp"
#include "engines/engine.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

void maskProcessTitle(int argc, char **argv);

static void usage() {
    std::fprintf(stderr,
        "usage: csvport_cpp opt -i input.csv -c constraint.csv -w weight.csv -o out.csv\n"
        "                      [--engine=osqp|qpalm|scs|proxqp] [--eps-abs=1e-6] [--eps-rel=1e-8]\n"
        "                      [--scale-rows] [--presolve=none|ruiz|pc] [--lib-scaling=default|on|off]\n"
        "                      [--rho=F] [--sigma=F] [--alpha=F] [--adaptive-rho=on|off]\n"
        "                      [--polish=off|kkt|ir] [--osqp-polish=on|off] [--max-iter=N]\n"
        "                      [--osqp-scaled-term] [-v]\n");
}

int main(int argc, char **argv) {
    std::string inFile, conFile, wtFile, outFile, engine = "osqp";
    double epsAbs = 1e-6, epsRel = 1e-8;
    bool verbose = false, doScaleRows = false, osqpScaledTerm = false;
    std::string presolve = "none", libScaling = "default", adRho = "default";
    std::string polish = "off", osqpPolish = "default";
    bool noRelax = false;
    EngineTuning tune;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&](const char *pfx) -> const char * {
            if (a.rfind(pfx, 0) == 0) return a.c_str() + strlen(pfx);
            return nullptr;
        };
        if (const char *v = val("-i=")) inFile = v;
        else if (const char *v = val("--input=")) inFile = v;
        else if (const char *v = val("-c=")) conFile = v;
        else if (const char *v = val("--constraint=")) conFile = v;
        else if (const char *v = val("-w=")) wtFile = v;
        else if (const char *v = val("--weight=")) wtFile = v;
        else if (const char *v = val("-o=")) outFile = v;
        else if (const char *v = val("--output=")) outFile = v;
        else if (const char *v = val("--engine=")) engine = v;
        else if (const char *v = val("--eps-abs=")) epsAbs = atof(v);
        else if (const char *v = val("--eps-rel=")) epsRel = atof(v);
        else if (a == "--scale-rows") doScaleRows = true;
        else if (const char *v = val("--presolve=")) presolve = v;
        else if (const char *v = val("--lib-scaling=")) libScaling = v;
        else if (const char *v = val("--rho=")) tune.rho = atof(v);
        else if (const char *v = val("--sigma=")) tune.sigma = atof(v);
        else if (const char *v = val("--alpha=")) tune.alpha = atof(v);
        else if (const char *v = val("--adaptive-rho=")) adRho = v;
        else if (const char *v = val("--polish=")) polish = v;
        else if (const char *v = val("--osqp-polish=")) osqpPolish = v;
        else if (const char *v = val("--max-iter=")) tune.maxIter = atoi(v);
        else if (const char *v = val("--rho-interval=")) tune.rhoInterval = atoi(v);
        else if (a == "--no-relax") noRelax = true;
        else if (a == "--osqp-scaled-term") osqpScaledTerm = true;
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "opt") { /* subcommand, ignored */ }
        else if (a == "-i" || a == "--input") { if (i + 1 < argc) inFile = argv[++i]; }
        else if (a == "-c" || a == "--constraint") { if (i + 1 < argc) conFile = argv[++i]; }
        else if (a == "-w" || a == "--weight") { if (i + 1 < argc) wtFile = argv[++i]; }
        else if (a == "-o" || a == "--output") { if (i + 1 < argc) outFile = argv[++i]; }
        else { usage(); return 1; }
    }
    if (inFile.empty() || conFile.empty() || wtFile.empty() || outFile.empty()) {
        usage();
        return 1;
    }
    // Copy out the arg strings we still need BEFORE masking (mirror of
    // main.go strings.Clone before setProcTitle).
    std::string inF = inFile, conF = conFile, wtF = wtFile, outF = outFile;
    maskProcessTitle(argc, argv);

    std::string err;
    auto tLoad0 = std::chrono::steady_clock::now();
    std::vector<InputItem> inputs = loadInputs(inF, err);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<ConstraintItem> constraints = loadConstraints(conF, err);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<WeightItem> weights = loadWeights(wtF, err);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto tLoad1 = std::chrono::steady_clock::now();

    if (!engineAvailable(engine)) {
        std::fprintf(stderr, "engine %s not compiled into this binary\n", engine.c_str());
        return 1;
    }

    // Solve orchestration (mirror opt.Solve): STEP1 relax, breach check,
    // pass 1 normal, pass 2 relax only if pass1 failed AND zero-trade breach.
    step1Relax(inputs);
    InputsCols cols = toColumns(inputs);
    std::vector<AssembledConstraint> det = assembleConstraints(cols.univ, constraints, weights);
    bool breach = breachWithZeroTraded(cols, det);

    // internal certificate state (set inside solveOnce, see dualGapCert)
    double gapCert = NAN;
    bool gapCertSet = false;
    // S2-3b experiment: elastic penalty override (default 100 mirrors Go)
    static double elasticPen = [] {
        const char *s = getenv("ELASTIC_PENALTY");
        return s ? atof(s) : 100.0;
    }();
    auto solveOnce = [&](bool relaxMode) -> std::pair<EngineResult, bool> {
        Problem p = buildProblem(cols, det, relaxMode, elasticPen);
        if (doScaleRows) scaleRows(p);
        bool scaled = doScaleRows;
        if (presolve == "ruiz") { applyRuiz(p, 10); scaled = true; }
        else if (presolve == "pc") { applyPockChambolle(p); scaled = true; }
        int libScale = libScaling == "on" ? 1 : libScaling == "off" ? 2 : 0;
        if (adRho == "on") tune.adaptiveRho = 1;
        else if (adRho == "off") tune.adaptiveRho = 0;
        if (osqpPolish == "on") tune.osqpPolish = 1;
        else if (osqpPolish == "off") tune.osqpPolish = 0;
        auto t0 = std::chrono::steady_clock::now();
        EngineResult r = engineSolve(engine, p, epsAbs, epsRel, verbose, osqpScaledTerm, libScale, tune);
        auto t1 = std::chrono::steady_clock::now();
        double engineMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // internal dual-gap certificate (admm_st2 S2-3): at the ENGINE's own
        // stopping point, BEFORE polish replaces x (the (x,y) pair must be
        // consistent; polish's recovered aux coords would desync it). Computed
        // in the SCALED space; the gap value is scale-invariant.
        {
            if (r.y.size() == p.rows.size()) {
                gapCert = dualGapCert(p, r.x, r.y);
                gapCertSet = true;
            }
        }
        // harness-level polish on the scaled coordinates (rows are O(1) here);
        // polishMs is included in solveMs (reported) and tracked separately
        double polishMs = 0;
        int polishOk = -1;
        if (polish == "kkt" || polish == "ir") {
            int ir = polish == "ir" ? 3 : 0;
            auto tp0 = std::chrono::steady_clock::now();
            polishOk = polishKKT(p, r.x, ir, r.y.empty() ? nullptr : &r.y) ? 1 : 0;
            auto tp1 = std::chrono::steady_clock::now();
            polishMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
        }
        r.harnessPolish = polishOk;
        r.polishMs = polishMs;
        if (scaled) {
            for (int j = 0; j < p.n; j++)
                if (j < (int)p.varScale.size()) r.x[j] *= p.varScale[j];
        }
        r.solveMs = engineMs + polishMs;
        bool ok = (r.status == "solved" || r.status == "solved_inaccurate" ||
                   r.status == "solved inaccurate" || r.status == "solved_backtracking" ||
                   r.status == "Solved" || r.status == "Solved/Inaccurate");
        return {r, ok};
    };

    auto tAll0 = std::chrono::steady_clock::now();
    auto [res, ok] = solveOnce(false);
    int hPolish = res.harnessPolish;
    double hPolishMs = res.polishMs;
    bool usedRelax = false;
    // S2-3b experiment hook: force the relaxation pass directly (moderate
    // artificial-breach scenarios; per-constraint breach flags inside
    // buildProblem decide which rows get the elastic treatment)
    static bool forceRelax = getenv("RELAX_FORCE") != nullptr;
    if ((!ok && breach && !noRelax) || (forceRelax && !noRelax)) {
        std::printf("normal opt attempt failed, constraint in breach with 0 traded, "
                    "trying to solve under relaxation mode\n");
        auto r2 = solveOnce(true);
        res = r2.first;
        hPolish = r2.first.harnessPolish;
        ok = r2.second;
        usedRelax = true;
    }
    auto tAll1 = std::chrono::steady_clock::now();

    double loadMs = std::chrono::duration<double, std::milli>(tLoad1 - tLoad0).count();
    double allMs = std::chrono::duration<double, std::milli>(tAll1 - tAll0).count();

    // expand result: reduced x -> per-symbol trades (fixed symbols use xFix)
    std::vector<double> trades(cols.univ.size(), 0.0);
    {
        Problem p = buildProblem(cols, det, usedRelax, elasticPen);
        std::vector<int> idxOf(cols.univ.size(), -1);
        for (size_t a = 0; a < p.active.size(); a++) idxOf[p.active[a]] = (int)a;
        for (size_t i = 0; i < cols.univ.size(); i++) {
            if (p.fixed[i]) trades[i] = p.xFix[i];
            else if (idxOf[i] >= 0 && idxOf[i] < (int)res.x.size())
                trades[i] = res.x[idxOf[i]];
        }
    }
    // fail-safe to no-trade when solver failed
    if (!ok) {
        for (auto &t : trades) t = 0.0;
    }
    res.harnessPolish = hPolish;
    res.polishMs = hPolishMs;
    double maxObj = 0;
    {
        Problem p = buildProblem(cols, det, usedRelax, elasticPen);
        (void)p;
        for (size_t i = 0; i < cols.univ.size(); i++) {
            double x = trades[i];
            maxObj += (cols.alpha[i] - cols.lambda[i] * cols.position[i]) * x -
                      0.5 * (cols.lambda[i] + cols.tradingLambda[i]) * x * x -
                      cols.tradingCost[i] * std::abs(x);
        }
    }

    // report + write output
    std::printf("engine=%s status=%s iter=%d solveMs=%.1f totalMs=%.1f loadMs=%.1f "
                "maxObj=%.6f relax=%d libPolish=%d hPolish=%d polishMs=%.2f setupMs=%.1f gapCert=%.3e\n",
                engine.c_str(), res.status.c_str(), res.iters, res.solveMs, allMs,
                loadMs, maxObj, usedRelax ? 1 : 0, res.polishStatus, hPolish,
                res.polishMs, res.setupMs, gapCertSet ? gapCert : NAN);
    FILE *f = std::fopen(outF.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", outF.c_str()); return 1; }
    std::fprintf(f, "sym,targetTrade,targetPosition\n");
    for (size_t i = 0; i < cols.univ.size(); i++)
        std::fprintf(f, "%s,%.10g,%.10g\n", cols.univ[i].c_str(), trades[i],
                     cols.position[i] + trades[i]);
    std::fclose(f);
    return ok ? 0 : 2;
}
