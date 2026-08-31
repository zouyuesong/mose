#!/usr/bin/env python3
"""Data-level equilibration of optim_inputs -> optim_scaled.

Equivalence claim (verified in cpp_admm report §7.6): cpp_admm --scale-rows
per-row/column equilibration is a pure algebra transform, independent of the
solver.  This script bakes that transform into the input/constraint/weight CSV
files, so ANY solver run without any scaling flag sees the equilibrated model.

Per symbol i, trade band b_i = max(1, max(|xl_i|, |xu_i|)) with
  xl_i = max(minTrade, minPos - pos), xu_i = min(maxTrade, maxPos - pos)
(matches cpp_admm model.cpp buildProblem).

Field transforms (strict algebra, same optimum, saves/restores units):
  input:  pos,minTrade,maxTrade,minPosition,maxPosition /= b_i
          alpha,cost *= b_i ; lambda,tlambda *= b_i^2
          grp,rebate,rebateLocate passthrough
  constr: each row sigma_k = max(1, max(|lb|,|ub|)); lb,ub /= sigma_k
  weight: weight *= b_sym / sigma_constr

Solve output x' (scaled trades) must be un-scaled: x = x' * b_i.
scale_map.csv records (sym,band) for that.
"""
import csv
import math
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optim_inputs")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optim_scaled")
DATE = "20260520"

INF = math.inf


def to_f(x):
    x = x.strip()
    if x in ("", "-inf", "+inf", "inf", "Inf", "Infinity", "-Infinity"):
        return -INF if x.startswith("-") else INF
    return float(x)


def fmt_inf(x):
    if x == INF:
        return "inf"
    if x == -INF:
        return "-inf"
    s = repr(x)
    if "e" not in s and "." not in s:
        return s
    return format(x, ".17g")


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def main():
    os.makedirs(DST, exist_ok=True)
    in_rows = load_rows(os.path.join(SRC, f"optim_inputs.{DATE}"))
    co_rows = load_rows(os.path.join(SRC, f"optim_constraints.{DATE}"))
    we_rows = load_rows(os.path.join(SRC, f"optim_weights.{DATE}"))

    in_head = in_rows[0]
    ci = {c: i for i, c in enumerate(in_head)}
    # per-symbol band
    band = {}
    for r in in_rows[1:]:
        pos = to_f(r[ci["position"]])
        xl = max(to_f(r[ci["minTrade"]]), to_f(r[ci["minPosition"]]) - pos)
        xu = min(to_f(r[ci["maxTrade"]]), to_f(r[ci["maxPosition"]]) - pos)
        b = max(1.0, max(abs(xl), abs(xu)))
        band[r[ci["sym"]]] = b

    # constraint row sigma
    chead = co_rows[0]
    cxi = {c: i for i, c in enumerate(chead)}
    sigma = {}
    for r in co_rows[1:]:
        lb, ub = to_f(r[cxi["lb"]]), to_f(r[cxi["ub"]])
        s = max(1.0, max(abs(lb), abs(ub)))
        sigma[r[cxi["name"]]] = s

    # ---- input transform ----
    out_in = [in_head]
    for r in in_rows[1:]:
        b = band[r[ci["sym"]]]
        nr = list(r)
        for name in ("position", "minTrade", "maxTrade", "minPosition", "maxPosition"):
            v = to_f(r[ci[name]])
            if abs(v) < 1e-300:
                nr[ci[name]] = "0"
            else:
                nr[ci[name]] = fmt_inf(v / b)
        for name in ("alpha", "cost"):
            nr[ci[name]] = fmt_inf(to_f(r[ci[name]]) * b)
        for name in ("lambda", "tlambda"):
            nr[ci[name]] = fmt_inf(to_f(r[ci[name]]) * b * b)
        out_in.append(nr)

    # ---- constraint transform ----
    out_co = [chead]
    for r in co_rows[1:]:
        s = sigma[r[cxi["name"]]]
        nr = list(r)
        nr[cxi["lb"]] = fmt_inf(to_f(r[cxi["lb"]]) / s)
        nr[cxi["ub"]] = fmt_inf(to_f(r[cxi["ub"]]) / s)
        out_co.append(nr)

    # ---- weight transform ----
    whead = we_rows[0]
    wxi = {c: i for i, c in enumerate(whead)}
    out_we = [whead]
    for r in we_rows[1:]:
        nr = list(r)
        w = to_f(r[wxi["weight"]]) * band[r[wxi["sym"]]] / sigma[r[wxi["name"]]]
        nr[wxi["weight"]] = fmt_inf(w)
        out_we.append(nr)

    with open(os.path.join(DST, f"optim_inputs.{DATE}"), "w", newline="") as f:
        csv.writer(f).writerows(out_in)
    with open(os.path.join(DST, f"optim_constraints.{DATE}"), "w", newline="") as f:
        csv.writer(f).writerows(out_co)
    with open(os.path.join(DST, f"optim_weights.{DATE}"), "w", newline="") as f:
        csv.writer(f).writerows(out_we)
    with open(os.path.join(DST, "scale_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sym", "band"])
        for s, b in sorted(band.items()):
            w.writerow([s, fmt_inf(b)])

    print("wrote optim_scaled/:",
          len(out_in) - 1, "symbols,", len(out_co) - 1, "constraints,",
          len(out_we) - 1, "weight rows")


if __name__ == "__main__":
    main()