"""G0 of N3.piles_prospective.v1: synthetic decidability of the conversion test (model-only).

Run BEFORE registration; spends no data. The synthetic world is the frozen page model at the
two primary pile cells (``model_side.json`` from build_mapping.py): the true local drawdown
change is the model's nest dD (heads at the nest wells' cells and layers, declared layer
rule), the true pile increment is rho x the model's column increment, and then

* the nest-vs-cell head error: AR(1) on the head LEVEL, marginal sd {0.75, 1.5} m (the
  judge's grid), persistence phi {0.5, 0.9} (declared here; the gate must pass in every
  cell), one error series per pile nest, differenced into dD;
* the print rounding: 0.1 cm on every monthly increment, at the model's own amplitude;
* truths rho in {0.4, 1, 2}, 500 replicates each, one SeedSequence per cell.

The estimator, the 3-month moving-block bootstrap (B 2000) and the verdict are n3lib's, the
ones the scorer uses. Pass at a horizon (12 or 24 months) needs, in every noise cell:
P(conversion_supported | rho = 1) >= 0.80, P(direction_supported_size_not | rho = 0.4) >=
0.80, P(conversion_supported | rho = 0.4 or 2) <= 0.10, and the 90 % interval's coverage of
rho = 1 >= 0.85. Decision (declared before the run): the 12-month test is the primary if G0
passes at 12; else the 24-month continuation if it passes at 24; else the conversion test is
registered report-only (C31 stays N: 'not identifiable at the piles by design').

Also reported, for rank 12 (the after-the-fact diagnostic; never for N3): the same gate on
the page model's pre-origin hindcast months 2016-06..2022-12 at the Yunlin piles, and the
溪州 season estimator in R1c's hindcast world (recovery of k within [1/1.5, 1.5] in >= 80 %
of draws and t-interval coverage >= 0.85 at rho = 1).

Frozen (2026-11-01, after the builder's first full run failed; disclosed in the registration):
SD_GRID, RHOS, REPS, the pass thresholds and the bootstrap are the judge's; PHI_GRID was
declared here before the first run; the estimator and N2's shared decision rule
(n3lib.conversion_verdict -> n2_policy/bootstrap_scorer.decide) are the scorer's. Changing
any of them after this known failure (for example an errors-in-variables or IV estimator)
needs a user ruling and a fresh, disclosed G0 run; every archived run is listed by
prereg_n3.py. The truth world uses the DISPLAYED ZR heads (model_side.json column_pairing):
G0 tests the estimator's attenuation, not the column's pairing with its driver.

Usage: python -I dataprep_staging/n3_piles/g0_n3.py run --model-side <dir>/model_side.json
           --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n3lib as L  # noqa: E402

G0_ID = "N3.G0.v1"
SD_GRID = (0.75, 1.5)
PHI_GRID = (0.5, 0.9)
RHOS = (0.4, 1.0, 2.0)
REPS = 500
SEED = 20261101
P_CORRECT = 0.80
P_FALSE_MAX = 0.10
COVERAGE_MIN = 0.85
HINDCAST_YUNLIN = ("2016-06", "2022-12")
XIZHOU_SEASONS = (2019, 2020, 2021, 2022)
CORRECT = {0.4: "direction_supported_size_not", 1.0: "conversion_supported",
           2.0: "direction_supported_size_not"}


def ar1(n: int, sd: float, phi: float, rng) -> np.ndarray:
    """Stationary AR(1) with marginal sd ``sd``."""
    e = np.empty(n)
    e[0] = rng.normal(0.0, sd)
    s = sd * np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0.0, s)
    return e


def world_arrays(world: dict, keys, span) -> dict:
    out = {}
    for k in keys:
        p = world["piles"][k]
        sel = [i for i, m in enumerate(p["months"]) if span[0] <= m <= span[1]]
        out[k] = {"inc": np.asarray([p["inc_cm"][i] for i in sel], float),
                  "dd": np.asarray([p["dd_nest_m"][i] for i in sel], float),
                  "months": [p["months"][i] for i in sel]}
    return out


def draw(arr: dict, rho: float, sd: float, phi: float, rng) -> tuple:
    ys, xs, gs = [], [], []
    for k, a in arr.items():
        n = len(a["inc"])
        e = ar1(n + 1, sd, phi, rng)
        x = a["dd"] + np.diff(e)
        y = np.round(rho * a["inc"] / L.PRINT_RES_CM) * L.PRINT_RES_CM
        ok = np.isfinite(x) & np.isfinite(y)
        ys.append(y[ok])
        xs.append(x[ok])
        gs.append(np.array([k] * int(ok.sum())))
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(gs)


def conversion_cell(arr: dict, g_mod: float, sd: float, phi: float, rho: float, reps: int,
                    B: int, seed_seq) -> dict:
    rng = np.random.default_rng(seed_seq)
    cats, cover, rhat = [], [], []
    for _ in range(reps):
        y, x, g = draw(arr, rho, sd, phi, rng)
        est = L.conversion_estimate(y, x, g, B=B, rng=rng)
        v = L.conversion_verdict(est["g"], est["lo"], est["hi"], g_mod)
        cats.append(v["verdict"])
        ci = v["rho_ci"]
        cover.append(bool(ci is not None and ci[0] <= rho <= ci[1]))
        rhat.append(v["rho"] if v["rho"] is not None else np.nan)
    s = pd.Series(cats)
    probs = {c: float((s == c).mean()) for c in ("conversion_supported",
                                                 "direction_supported_size_not",
                                                 "not_supported", "not_informative")}
    r = np.asarray(rhat, float)
    return {"rho": rho, "sd": sd, "phi": phi, "reps": reps, "probs": probs,
            "p_correct": probs[CORRECT[rho]], "coverage": float(np.mean(cover)),
            "rho_hat_median": float(np.nanmedian(r)),
            "rho_hat_q10_q90": np.nanquantile(r, [0.1, 0.9]).tolist(),
            "mc_se_p_correct": float(np.sqrt(probs[CORRECT[rho]]
                                             * (1 - probs[CORRECT[rho]]) / reps))}


def judge_cells(cells: list[dict]) -> dict:
    """The declared pass rule over the noise grid."""
    fails = []
    for sd in SD_GRID:
        for phi in PHI_GRID:
            c = {r["rho"]: r for r in cells if r["sd"] == sd and r["phi"] == phi}
            tag = f"sd {sd} phi {phi}"
            if c[1.0]["p_correct"] < P_CORRECT:
                fails.append(f"{tag}: P(supported | rho 1) {c[1.0]['p_correct']:.3f} < 0.80")
            if c[0.4]["p_correct"] < P_CORRECT:
                fails.append(f"{tag}: P(direction, size not | rho 0.4) "
                             f"{c[0.4]['p_correct']:.3f} < 0.80")
            for r in (0.4, 2.0):
                p = c[r]["probs"]["conversion_supported"]
                if p > P_FALSE_MAX:
                    fails.append(f"{tag}: false 'supported' at rho {r} {p:.3f} > 0.10")
            if c[1.0]["coverage"] < COVERAGE_MIN:
                fails.append(f"{tag}: coverage at rho 1 {c[1.0]['coverage']:.3f} < 0.85")
    return {"pass": not fails, "fails": fails}


def conversion_gate(arr: dict, g_mod: float, reps: int, B: int, ss) -> dict:
    cells = []
    kids = iter(ss.spawn(len(SD_GRID) * len(PHI_GRID) * len(RHOS)))
    for sd in SD_GRID:
        for phi in PHI_GRID:
            for rho in RHOS:
                cells.append(conversion_cell(arr, g_mod, sd, phi, rho, reps, B, next(kids)))
    return {"g_mod": g_mod, "n_months": {k: len(a["inc"]) for k, a in arr.items()},
            "cells": cells, **judge_cells(cells)}


def season_gate(world: dict, reps: int, ss) -> dict:
    """溪州 season estimator in R1c's hindcast world (rank 12 only)."""
    p = world["piles"]["xizhou"]
    inc = pd.Series(p["inc_cm"], index=p["months"])
    dd = pd.Series(p["dd_nest_m"], index=p["months"])
    thr = L.discontinuity_threshold(inc.to_numpy())
    truth = L.season_coefficients(inc, dd, XIZHOU_SEASONS, threshold_cm=thr)
    k0 = truth["pooled"]["k"]
    if k0 is None or truth["pooled"]["n"] < 2:
        return {"pass": False, "truth": truth, "fails": [
            "fewer than 2 informative model seasons in R1c's world: the coefficient is not "
            "identifiable from held piles"]}
    cells = []
    kids = iter(ss.spawn(len(SD_GRID) * len(PHI_GRID)))
    for sd in SD_GRID:
        for phi in PHI_GRID:
            rng = np.random.default_rng(next(kids))
            rec, cov = [], []
            for _ in range(reps):
                e = ar1(len(dd) + 1, sd, phi, rng)
                ddo = dd + np.diff(e)
                inco = np.round(inc / L.PRINT_RES_CM) * L.PRINT_RES_CM
                r = L.season_coefficients(inco, ddo, XIZHOU_SEASONS, threshold_cm=thr)
                k = r["pooled"]["k"]
                rec.append(k is not None and 1 / L.RATIO_BAND <= k / k0 <= L.RATIO_BAND)
                ci = r["pooled"]["ci90"]
                cov.append(ci is not None and ci[0] <= k0 <= ci[1])
            cells.append({"sd": sd, "phi": phi, "p_recovery": float(np.mean(rec)),
                          "coverage": float(np.mean(cov))})
    fails = [f"sd {c['sd']} phi {c['phi']}: recovery {c['p_recovery']:.3f} or coverage "
             f"{c['coverage']:.3f}" for c in cells
             if c["p_recovery"] < P_CORRECT or c["coverage"] < COVERAGE_MIN]
    return {"pass": not fails, "fails": fails, "k_true_model": k0, "truth": truth,
            "cells": cells}


def decision(h12: dict, h24: dict) -> dict:
    if h12["pass"]:
        return {"primary_horizon": 12, "conversion_role": "gating",
                "wording": "G0 passed at 12 months: N3P.conversion_12m is the primary; the "
                           "24-month continuation is report-only"}
    if h24["pass"]:
        return {"primary_horizon": 24, "conversion_role": "gating",
                "wording": "G0 failed at 12 months and passed at 24: N3P.conversion_24m is "
                           "the primary (declared before registration); the 12-month line is "
                           "report-only"}
    return {"primary_horizon": None, "conversion_role": "report_only",
            "wording": "G0 failed at 12 and 24 months: the conversion test is registered "
                       "report-only; C31 stays N ('conversion not identifiable at the piles "
                       "by design')"}


def code_shas() -> dict:
    """The code whose behaviour G0 characterises (checked by prereg_n3.g0_block)."""
    return {f"{L.E}/g0_n3.py": L.sha256(HERE / "g0_n3.py"),
            f"{L.E}/n3lib.py": L.sha256(HERE / "n3lib.py"),
            L.SHARED_SCORER: L.sha256(REPO / L.SHARED_SCORER)}


def run(model_side: Path, out: Path, smoke: bool = False, reps: int = REPS,
        B: int = L.BOOT_B) -> dict:
    out = Path(out)
    res_path = out / "result.json"
    if res_path.exists():
        raise FileExistsError(f"{res_path} exists: G0 is run once (archive it to retry)")
    if smoke:
        reps, B = 20, 200
    ms = json.loads(Path(model_side).read_text(encoding="utf-8"))
    t0 = time.time()
    root = np.random.SeedSequence(SEED)
    s12, s24, spre, sxz = root.spawn(4)
    w = ms["world"]
    h12 = conversion_gate(world_arrays(w, L.PRIMARY, L.WINDOW), ms["g_mod_12"]["g_mod"],
                          reps, B, s12)
    h24 = conversion_gate(world_arrays(w, L.PRIMARY, L.WINDOW24), ms["g_mod_24"]["g_mod"],
                          reps, B, s24)
    hw = world_arrays(ms["hindcast"], L.PRIMARY, HINDCAST_YUNLIN)
    ys = np.concatenate([a["inc"] for a in hw.values()])
    xs = np.concatenate([a["dd"] for a in hw.values()])
    gs = np.concatenate([[k] * len(a["inc"]) for k, a in hw.items()])
    ok = np.isfinite(ys) & np.isfinite(xs)
    g_pre = L.fe_slope(ys[ok], xs[ok], gs[ok])
    hpre = conversion_gate(hw, g_pre, reps, B, spre)
    xz = (season_gate(ms["xizhou_r1c"]["hindcast"], reps, sxz) if "xizhou_r1c" in ms
          else {"pass": False, "fails": ["no R1c world in model_side.json"]})
    res = {
        "id": G0_ID, "for": L.ID, "smoke": bool(smoke),
        "code": code_shas(),
        "model_side": {"path": str(Path(model_side).name), "sha256": L.sha256(model_side)},
        "parameters": {"sd_grid_m": SD_GRID, "phi_grid": PHI_GRID, "rhos": RHOS,
                       "reps": reps, "boot_B": B, "boot_block": L.BOOT_BLOCK, "seed": SEED,
                       "print_res_cm": L.PRINT_RES_CM, "p_correct": P_CORRECT,
                       "p_false_max": P_FALSE_MAX, "coverage_min": COVERAGE_MIN},
        "horizon_12": h12, "horizon_24": h24, "decision": decision(h12, h24),
        "rank12_report_only": {"yunlin_hindcast_2016_06_2022_12": hpre, "xizhou_r1c": xz},
        "seconds": round(time.time() - t0, 1),
        "reads": "model_side.json only (frozen model output); no observation"}
    out.mkdir(parents=True, exist_ok=True)
    with open(res_path, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, sort_keys=True, default=float)
        f.write("\n")
    return res


def brief(res: dict) -> dict:
    def b(h):
        return {"pass": h["pass"], "fails": h["fails"][:6],
                "cells": [{k: c[k] for k in ("sd", "phi", "rho", "p_correct", "coverage",
                                             "rho_hat_median")} for c in h["cells"]]}
    r12 = res["rank12_report_only"]
    return {"decision": res["decision"], "h12": b(res["horizon_12"]),
            "h24": b(res["horizon_24"]),
            "rank12_yunlin_pass": r12["yunlin_hindcast_2016_06_2022_12"]["pass"],
            "rank12_xizhou": {k: r12["xizhou_r1c"].get(k) for k in ("pass", "fails")},
            "smoke": res["smoke"], "seconds": res["seconds"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-side", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.model_side, a.out, smoke=a.smoke)
    print(json.dumps(brief(res), indent=1, ensure_ascii=False, default=float))
    print("result sha256:", L.sha256(a.out / "result.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
