"""G0' of N3 (declaration N3.G0prime.v1): synthetic decidability of an attenuation-robust
conversion estimator at the 2 Yunlin deep piles. Model-only; run ONCE, after the declaration
is registered (prereg_g0prime.py --write), on N3's registered model_side.json.

The world is N3 G0's (g0_n3.py, frozen and imported): the frozen page model at the two
primary pile cells, true pile increment = rho x the model's column increment, nest-vs-cell
head error AR(1) on the head LEVEL with marginal sd {0.75, 1.5} m and persistence {0.5, 0.9},
0.1 cm print rounding, truths rho {0.4, 1, 2}, 500 replicates per cell. Only the estimator
changes (g0p_lib: (a) harmonic-amplitude ratio, (b) 2SLS with calendar harmonics, (c) 2SLS
with the model's nest dD as instrument; N3's OLS as a reference line). Added cells:

* seasonal mismatch (judge rank 14): the nest's annual head component is amp x the model's
  (amp {0.7, 1, 1.3}) and lags it by {0, 2} months, the pile still follows the cell. With amp
  != 1 every nest-based estimator measures rho / amp, which no data from the nest can undo;
  a 2-month lag rotates the annual harmonic by 60 deg (biases (b) and (c), not (a));
* forecast deviation {none, cell} (builder): 'cell' adds an independent AR(1) (the cell's sd
  and phi) to the true local head, seen by the nest and converted into the pile at the
  model's OLS slope, so the actual head change is not exactly the model's (the instrument of
  (c)). It is uncorrelated with every instrument: it adds variance only.

Pass of an estimator at a horizon (12 or 24 months) needs, in every GATING cell:
P(conversion_supported | rho 1) >= 0.80, P(direction_supported_size_not | rho 0.4) >= 0.80,
P(conversion_supported | rho 0.4 or 2) <= 0.10, coverage of the 90 % interval at rho 1 >=
0.85 (G0's frozen rule), plus once per horizon a model-side first-stage F >= 10 (the judge's
weak-instrument check) and a finite positive g_mod. The median first-stage F of the draws is
reported per cell, not gated. Which cells gate is the declared gate mode (design.gate_mode,
read from the registered declaration): 'all' (the judge's text; the 'amp' cells cannot be
passed by any nest-based estimator, so a fail is expected) or 'identifiable' (G0's cells and
the forecast-deviation cells gate; the seasonal-mismatch cells are a report-only
sensitivity). Every estimator's pass per cell class (g0, fdev, lag, amp) is reported in
every case. Decision (declared): the first estimator in the order (a), (b), (c) that passes
at 12 months is N3G's estimator with a 12-month gating test; else the first that passes at
24 months (24-month gating test); else G0' fails, no N3G registration is written, and the
fail wording names the cell class that failed. All three estimators are reported.

Before any grid cell runs, the OLS g_mod of both horizons must equal N3's registered g_mod and
be finite and positive; otherwise the run stops with an engineering failure and writes
nothing (fix, archive with a reason, rerun: not a scored attempt).

Usage (repository root, the lead, after the declaration; ~10-20 min CPU):
    python -I dataprep_staging/n3_g0prime/g0prime.py run \
        --model-side results/dataprep/reports/N3/mapping/model_side.json \
        --out results/dataprep/reports/N3.G0prime.v1/g0prime
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
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0p_lib as P  # noqa: E402

L = P.L
RESULT = "result.json"
CODE_FILES = (f"{P.E}/g0p_lib.py", f"{P.E}/g0prime.py", "dataprep_staging/n3_piles/g0_n3.py",
              "dataprep_staging/n3_piles/n3lib.py", L.SHARED_SCORER)


class NotDeclared(RuntimeError):
    """G0' may run on the real model_side.json only after its declaration is registered."""


class EngineeringFailure(RuntimeError):
    """A precondition of the model world failed before any cell ran: nothing is written."""


def code_shas() -> dict:
    return {p: P.sha256(REPO / p) for p in CODE_FILES}


def require_declared(prereg_dir=None, model_side: Path | None = None) -> tuple[dict, dict]:
    """The declaration is in INDEX.jsonl, its file hashes as indexed, its frozen code is this
    code, and ``model_side`` is the declared file. Returns (index record, document)."""
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == P.ID] \
        if (d / "INDEX.jsonl").exists() else []
    if len(rows) != 1:
        raise NotDeclared(f"{P.ID} is not registered in {d} (prereg_g0prime.py --write first)")
    f = d / f"{P.ID}.json"
    if not f.is_file() or P.sha256(f) != rows[0]["sha256"]:
        raise NotDeclared(f"{f} is missing or does not hash as indexed")
    doc = json.loads(f.read_text(encoding="utf-8"))
    want = doc.get("frozen_code") or {}
    have = {p: P.sha256(REPO / p) for p in want if (REPO / p).is_file()}
    if not want or have != want:
        bad = sorted(p for p in want if have.get(p) != want[p])
        raise NotDeclared(f"code differs from the declaration's frozen_code: {bad}")
    if model_side is not None and P.sha256(model_side) != doc["model_side"]["sha256"]:
        raise NotDeclared("model_side.json is not the declared file")
    if (doc.get("design") or {}).get("gate_mode") not in P.GATE_MODES:
        raise NotDeclared("the declaration names no valid design.gate_mode")
    return rows[0], doc


# ----------------------------------------------------------------------------- the gate
def _summary(cats: list, cover: list, rhat: list, rho: float, reps: int) -> dict:
    s = pd.Series(cats)
    probs = {c: float((s == c).mean()) for c in ("conversion_supported",
                                                 "direction_supported_size_not",
                                                 "not_supported", "not_informative")}
    r = np.asarray(rhat, float)
    pc = probs[P.CORRECT[rho]]
    fin = r[np.isfinite(r)]
    return {"probs": probs, "p_correct": pc, "coverage": float(np.mean(cover)),
            "rho_hat_median": float(np.median(fin)) if fin.size else None,
            "rho_hat_q10_q90": np.quantile(fin, [0.1, 0.9]).tolist() if fin.size else None,
            "mc_se_p_correct": float(np.sqrt(pc * (1 - pc) / reps))}


def run_cell(arr: dict, g_mod: dict, cell: tuple, rho: float, reps: int, B: int,
             seed_seq) -> dict:
    sd, phi, amp, lag, fdev = cell
    rng = np.random.default_rng(seed_seq)
    cats = {k: [] for k in P.ALL_LINES}
    cover = {k: [] for k in P.ALL_LINES}
    rhat = {k: [] for k in P.ALL_LINES}
    fs = {"f_harmonic": [], "f_model": []}
    for _ in range(reps):
        y, x, z, m, g = P.draw(arr, rho, sd, phi, amp, lag, fdev, g_mod[P.REFERENCE], rng)
        est = P.estimate_all(y, x, z, m, g, B=B, rng=rng)
        v = P.verdicts(est, g_mod)
        for k in P.ALL_LINES:
            cats[k].append(v[k]["verdict"])
            ci = v[k]["rho_ci"]
            cover[k].append(bool(ci is not None and ci[0] <= rho <= ci[1]))
            rhat[k].append(v[k]["rho"] if v[k]["rho"] is not None else np.nan)
        for f in fs:
            fs[f].append(est[f])
    return {"sd": sd, "phi": phi, "amp": amp, "lag": lag, "fdev": fdev, "rho": rho,
            "reps": reps,
            "lines": {k: _summary(cats[k], cover[k], rhat[k], rho, reps) for k in P.ALL_LINES},
            "f_median": {f: float(np.nanmedian(v)) for f, v in fs.items()}}


def _cell_fails(c: dict, line: str, tag: str) -> list[str]:
    """G0's four rules in one noise cell (``c``: rho -> cell record)."""
    fails = []
    s = {r: c[r]["lines"][line] for r in P.RHOS}
    if s[1.0]["p_correct"] < P.P_CORRECT:
        fails.append(f"{tag}: P(supported | rho 1) {s[1.0]['p_correct']:.3f} < 0.80")
    if s[0.4]["p_correct"] < P.P_CORRECT:
        fails.append(f"{tag}: P(direction, size not | rho 0.4) "
                     f"{s[0.4]['p_correct']:.3f} < 0.80")
    for r in (0.4, 2.0):
        p = s[r]["probs"]["conversion_supported"]
        if p > P.P_FALSE_MAX:
            fails.append(f"{tag}: false 'supported' at rho {r} {p:.3f} > 0.10")
    if s[1.0]["coverage"] < P.COVERAGE_MIN:
        fails.append(f"{tag}: coverage at rho 1 {s[1.0]['coverage']:.3f} < 0.85")
    return fails


def judge(cells: list[dict], line: str, model_f: dict, g_mod: dict,
          gate_mode: str = "all") -> dict:
    """The declared pass rule of one line: G0's four rules in every gating cell of
    ``gate_mode`` plus the once-per-horizon checks (candidates only). The pass of every cell
    class and the per-draw first-stage F are reported, never gated."""
    gating = P.GATE_MODES[gate_mode]
    once = []
    if line != P.REFERENCE:
        g = g_mod[line]
        if not (np.isfinite(g) and g > 0):
            once.append(f"g_mod of {line} is {g} (not finite and positive)")
        fm = model_f[P.F_OF[line]]
        if not (fm >= P.F_MIN):
            once.append(f"model-side first-stage {P.F_OF[line]} {fm} < {P.F_MIN}")
    by_class = {k: [] for k in P.CELL_CLASSES}
    f_med = {k: [] for k in P.CELL_CLASSES}
    for nc in P.noise_cells():
        c = {r["rho"]: r for r in cells
             if (r["sd"], r["phi"], r["amp"], r["lag"], r["fdev"]) == nc}
        k = P.cell_class(*nc[2:])
        by_class[k] += _cell_fails(c, line, P.cell_tag(*nc))
        if line != P.REFERENCE:
            f_med[k].append(float(np.nanmedian([c[r]["f_median"][P.F_OF[line]]
                                                for r in P.RHOS])))
    fails = once + [f for k in gating for f in by_class[k]]
    return {"pass": not fails, "n_fails": len(fails), "fails": fails[:40],
            "gate_mode": gate_mode, "gating_classes": list(gating),
            "once_per_horizon_ok": not once,
            "class_pass": {k: (not once) and not v for k, v in by_class.items()},
            "class_n_fails": {k: len(v) for k, v in by_class.items()},
            "report_only_draw_f_median_min": (
                {k: (float(min(v)) if v else None) for k, v in f_med.items()}
                if line != P.REFERENCE else None),
            "role": "reference (never selectable)" if line == P.REFERENCE else "candidate"}


def precheck(world: dict, ms: dict) -> dict:
    """Engineering preconditions, before any cell runs: per horizon the OLS g_mod equals N3's
    registered g_mod and is finite and positive. Raises EngineeringFailure (nothing written)."""
    out = {}
    for h in (12, 24):
        gm = P.g_mod_lines(P.window_world(world, P.HORIZONS[h]))
        g = float(gm["g_mod"][P.REFERENCE])
        reg = float(ms[P.G_MOD_KEY[h]]["g_mod"])
        if not (np.isfinite(g) and g > 0):
            raise EngineeringFailure(f"h{h}: OLS g_mod {g} is not finite and positive")
        if abs(g - reg) > 1e-6 * max(1.0, abs(reg)):
            raise EngineeringFailure(f"h{h}: OLS g_mod {g} != N3's registered {reg}")
        out[h] = gm
    return out


def horizon_gate(world: dict, h: int, reps: int, B: int, ss, g_mod_registered: float,
                 progress=None, gate_mode: str = "all") -> dict:
    arr = P.window_world(world, P.HORIZONS[h])
    gm = P.g_mod_lines(arr)
    g_mod = gm["g_mod"]
    cons = abs(g_mod[P.REFERENCE] - g_mod_registered) <= 1e-6 * max(1.0, abs(g_mod_registered))
    cells = []
    combos = [(nc, rho) for nc in P.noise_cells() for rho in P.RHOS]
    kids = ss.spawn(len(combos))
    t0 = time.time()
    for i, ((nc, rho), kid) in enumerate(zip(combos, kids, strict=True)):
        cells.append(run_cell(arr, g_mod, nc, rho, reps, B, kid))
        if progress:
            progress(f"h{h} cell {i + 1}/{len(combos)} ({time.time() - t0:.0f} s)")
    lines = {k: judge(cells, k, gm["model_side_f"], g_mod, gate_mode) for k in P.ALL_LINES}
    return {"horizon_months": h, "window": list(P.HORIZONS[h]),
            "n_months": {k: len(a["inc"]) for k, a in arr.items()},
            "g_mod": g_mod, "model_side_f": gm["model_side_f"],
            "g_mod_ols_equals_registered": bool(cons),
            "g_mod_registered_ols": g_mod_registered, "lines": lines, "cells": cells}


FAIL_WORDING = {
    "nest_error": "no estimator identifies the conversion at 2 piles even in N3 G0's own "
                  "world (nest error only; harmonic ratio, 2SLS with calendar harmonics, 2SLS "
                  "with the model's heads)",
    "forecast_deviation": "an estimator identifies the conversion at 2 piles in N3 G0's own "
                          "world, but none does once the actual heads deviate from the "
                          "model's forecast (N3 G0 and G0')",
    "mismatch": "an estimator identifies the conversion at 2 piles when the nest's seasonal "
                "swing tracks the pile cell's, but none can when it differs by 30 % or lags "
                "by 2 months: from nest heads that mismatch cannot be told apart from the "
                "conversion (N3 G0 and G0')"}


def fail_class(h12: dict, h24: dict) -> str:
    """Which cell class the failure is attributed to (first class no estimator passes)."""
    def any_pass(classes):
        return any(all(hz["lines"][k]["class_pass"][c] for c in classes)
                   for hz in (h12, h24) for k in P.ESTIMATORS)
    if not any_pass(("g0",)):
        return "nest_error"
    if not any_pass(("g0", "fdev")):
        return "forecast_deviation"
    return "mismatch"


def decision(h12: dict, h24: dict) -> dict:
    for hz in (h12, h24):
        for k in P.ESTIMATORS:
            if hz["lines"][k]["pass"]:
                h = hz["horizon_months"]
                return {"passed": True, "horizon_months": h, "estimator": k,
                        "g_mod": hz["g_mod"][k],
                        "wording": f"G0' passed at {h} months with {k} (the first passing "
                                   "estimator in the declared order a, b, c at the first "
                                   "passing horizon): N3G may be registered with this "
                                   "estimator gating at this horizon (ruling R7.n3_g0prime), "
                                   "before 2026-11-30T16:00Z"}
    fc = fail_class(h12, h24)
    return {"passed": False, "horizon_months": None, "estimator": None, "fail_class": fc,
            "wording": "G0' failed for every estimator at 12 and 24 months: no N3G "
                       f"registration; C31 keeps N with the note '{FAIL_WORDING[fc]}'"}


def run(model_side: Path, out: Path, *, smoke: bool = False, reps: int = P.REPS,
        B: int = L.BOOT_B, prereg_dir=None, check_declared: bool = True,
        progress=None, gate_mode: str = "all") -> dict:
    out = Path(out)
    res_path = out / RESULT
    if res_path.exists():
        raise FileExistsError(f"{res_path} exists: G0' is run once (archive it to retry)")
    if check_declared:
        rec, ddoc = require_declared(prereg_dir, Path(model_side))
        gate_mode = ddoc["design"]["gate_mode"]
    else:
        rec = {"note": "declaration check disabled (tests only; a real run checks it)"}
    if gate_mode not in P.GATE_MODES:
        raise ValueError(f"gate_mode {gate_mode!r} not in {sorted(P.GATE_MODES)}")
    if smoke:
        reps, B = 20, 200
    ms = json.loads(Path(model_side).read_text(encoding="utf-8"))
    precheck(ms["world"], ms)                     # engineering: raises before any cell runs
    t0 = time.time()
    s12, s24 = np.random.SeedSequence(P.SEED).spawn(2)
    h12 = horizon_gate(ms["world"], 12, reps, B, s12, ms["g_mod_12"]["g_mod"], progress,
                       gate_mode)
    h24 = horizon_gate(ms["world"], 24, reps, B, s24, ms["g_mod_24"]["g_mod"], progress,
                       gate_mode)
    res = {
        "id": P.ID, "for": [L.ID, P.N3G_ID], "smoke": bool(smoke),
        "declaration_index_record": rec, "code": code_shas(),
        "gate": {"mode": gate_mode, "gating_classes": list(P.GATE_MODES[gate_mode]),
                 "cell_classes": list(P.CELL_CLASSES)},
        "model_side": {"path": Path(model_side).name, "sha256": P.sha256(model_side)},
        "parameters": {"sd_grid_m": P.SD_GRID, "phi_grid": P.PHI_GRID, "rhos": P.RHOS,
                       "amp_ratios": P.AMP_RATIOS, "lags_months": P.LAGS_MONTHS,
                       "forecast_dev": P.FORECAST_DEV, "reps": reps, "boot_B": B,
                       "boot_block": L.BOOT_BLOCK, "seed": P.SEED,
                       "print_res_cm": L.PRINT_RES_CM, "p_correct": P.P_CORRECT,
                       "p_false_max": P.P_FALSE_MAX, "coverage_min": P.COVERAGE_MIN,
                       "f_min_model_side": P.F_MIN, "estimator_order": P.ESTIMATORS,
                       "reference": P.REFERENCE},
        "horizon_12": h12, "horizon_24": h24, "decision": decision(h12, h24),
        "seconds": round(time.time() - t0, 1),
        "reads": "model_side.json only (frozen model output: window increments and nest dD); "
                 "no observation"}
    out.mkdir(parents=True, exist_ok=True)
    res = P.jsonable(res)
    with open(res_path, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    return res


def brief(res: dict) -> dict:
    def b(h):
        return {"g_mod": h["g_mod"], "model_side_f": h["model_side_f"],
                "g_mod_ols_equals_registered": h["g_mod_ols_equals_registered"],
                "lines": {k: {"pass": v["pass"], "n_fails": v["n_fails"],
                              "class_pass": v["class_pass"],
                              "first_fails": v["fails"][:3]} for k, v in h["lines"].items()}}
    return {"decision": res["decision"], "gate": res["gate"], "h12": b(res["horizon_12"]),
            "h24": b(res["horizon_24"]), "smoke": res["smoke"], "seconds": res["seconds"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-side", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = run(a.model_side, a.out,
              progress=lambda s: print(s, file=sys.stderr, flush=True))
    print(json.dumps(brief(res), indent=1, ensure_ascii=False, default=float))
    print("result sha256:", P.sha256(Path(a.out) / RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
