// engines.cpp - dispatch
#include "engine.hpp"
#include <stdexcept>

EngineResult engineSolve(const std::string &engine, const Problem &p,
                         double epsAbs, double epsRel, bool verbose,
                         bool osqpScaledTerm, int libScaling, EngineTuning tune) {
    // declared in the per-engine translation units (both variants)
    EngineResult osqpSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &);
    EngineResult qpalmSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &);
    EngineResult scsSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &);
    EngineResult proxqpSolve(const Problem &, double, double, bool, bool, int, const EngineTuning &);
    EngineResult nativeSolve(const Problem &, double, double, bool, const EngineTuning &);
    EngineResult nativeqpalmSolve(const Problem &, double, double, bool, const EngineTuning &);
    EngineResult nativeproxSolve(const Problem &, double, double, bool, const EngineTuning &);
    EngineResult nativeralmSolve(const Problem &, double, double, bool, const EngineTuning &);
    if (engine == "native") return nativeSolve(p, epsAbs, epsRel, verbose, tune);
    if (engine == "nativeqpalm") return nativeqpalmSolve(p, epsAbs, epsRel, verbose, tune);
    if (engine == "nativeprox") return nativeproxSolve(p, epsAbs, epsRel, verbose, tune);
    if (engine == "nativeralm") return nativeralmSolve(p, epsAbs, epsRel, verbose, tune);
    if (engine == "osqp") return osqpSolve(p, epsAbs, epsRel, verbose, osqpScaledTerm, libScaling, tune);
    if (engine == "qpalm") return qpalmSolve(p, epsAbs, epsRel, verbose, false, libScaling, tune);
    if (engine == "scs") return scsSolve(p, epsAbs, epsRel, verbose, false, libScaling, tune);
    if (engine == "proxqp") return proxqpSolve(p, epsAbs, epsRel, verbose, false, libScaling, tune);
    throw std::runtime_error("unknown engine: " + engine);
}

bool engineAvailable(const std::string &engine) {
    bool osqpAvailable();
    bool qpalmAvailable();
    bool scsAvailable();
    bool proxqpAvailable();
    bool nativeAvailable();
    bool nativeqpalmAvailable();
    bool nativeproxAvailable();
    bool nativeralmAvailable();
    if (engine == "native") return nativeAvailable();
    if (engine == "nativeqpalm") return nativeqpalmAvailable();
    if (engine == "nativeprox") return nativeproxAvailable();
    if (engine == "nativeralm") return nativeralmAvailable();
    if (engine == "osqp") return osqpAvailable();
    if (engine == "qpalm") return qpalmAvailable();
    if (engine == "scs") return scsAvailable();
    if (engine == "proxqp") return proxqpAvailable();
    return false;
}
