"""Post-hoc sensitivity of the state-space Jan-May coverage to the sigma grid (W1 repair round,
verifier minor issue: sigma = 0.45, the upper edge of config.SS_SIGMA_GRID, carries the
largest weight for full/v1 Yunlin and is chosen in two pre-origin folds).

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep_sigma_sensitivity.py OUT_JSON

Reporting only; nothing published changes and no setting is chosen from it. For each variant
(pre-origin 2012-2022, full 2012-2025) the leave-one-year-out coverage of the held-out Jan-May
totals (build.run_variant's loop, same seeds) is recomputed with the registered grid (must
reproduce statespace_loo_coverage.csv) and with the sigma grid extended to 1.0. The Jan-May
units are design-time units; the look is logged before anything is computed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import prereg2  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal import diversion, sources, statespace, validate  # noqa: E402
from w1_canal.build import LEVEL, jsonable  # noqa: E402
from w1_canal.config import BENCH_YEARS, DISTRICTS, SS_SEED, SS_SIGMA_GRID  # noqa: E402

EXT_GRID = tuple(SS_SIGMA_GRID) + (0.6, 0.8, 1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def loo_cov(dd, years, targets, res, grid_list, jm_all, cs) -> pd.DataFrame:
    rows = []
    for k, (yr, sc_k, sy_k, w_k) in enumerate(res["chosen_supply_limited"]):
        g = next(g for g in grid_list if g["cap"] and g["s_c"] == sc_k and g["s_y"] == sy_k
                 and g["w"] == w_k)
        ik = dict(zip(DISTRICTS, diversion.simulate(dd, *g["p"], sc_k, sy_k, w_k),
                      strict=True))
        for j, d in enumerate(DISTRICTS):
            obs = statespace.build_obs(years, targets[d], jm_all[d], cs[d], holdout_year=yr)
            r = statespace.run(ik[d], obs, seed=SS_SEED + 100 * (k + 1) + j)
            jm = statespace.janmay_of(r["draws"], years, yr)
            lo, hi = statespace.interval(jm, LEVEL)
            o = jm_all[d][yr]
            rows.append({"year": yr, "district": d, "obs": o, "p05": lo,
                         "p50": float(np.median(jm)), "p95": hi, "hit": bool(lo <= o <= hi),
                         "best_rho": r["best"]["rho"], "best_sigma": r["best"]["sigma"]})
    return pd.DataFrame(rows)


def summary(cov: pd.DataFrame) -> dict:
    from harness import coverage2
    g = coverage2.coverage_gate(cov.hit.to_numpy(bool), cov.year.to_numpy())
    return {"hits": int(cov.hit.sum()), "n": int(len(cov)),
            "a1_gate2": [float(g["lo"]), float(g["hi"]), bool(g["pass"])],
            "mean_width_90": {d: float((cov[cov.district == d].p95
                                        - cov[cov.district == d].p05).mean())
                              for d in DISTRICTS},
            "best_sigma": {f"{r.year}_{r.district}": r.best_sigma for r in cov.itertuples()},
            "misses": [f"{r.year}_{r.district}" for r in cov[~cov.hit].itertuples()]}


def main(out_json: str):
    prereg2.record_look([f"W1.janmay.{d}.{y}" for d in DISTRICTS for y in BENCH_YEARS],
                        "wra_briefing_110_janmay:2015-01..2021-05", "W1",
                        "post_hoc_sensitivity",
                        note="W1 repair round: state-space Jan-May leave-one-year-out coverage "
                             "with the sigma grid extended to 1.0 (verifier minor issue); "
                             "reporting only, design-time units")
    jiji, _ = sources.jiji_annual()
    bench, _ = sources.janmay_benchmark()
    moa = sources.moa_crop_share()
    v3m = sources.v3_monthly()
    jm_all = {d: {int(y): float(bench.loc[y, d]) for y in BENCH_YEARS} for d in DISTRICTS}
    cs = {d: {int(y): float(v) for y, v in moa[d].dropna().items()} for d in DISTRICTS}
    out = {"note": "post hoc (written after the results were known); reporting only",
           "registered_grid": list(SS_SIGMA_GRID), "extended_grid": list(EXT_GRID)}
    for name in ("preorigin", "full"):
        vdir = C.product_dir_base(name) / "v1"
        years = list(C.VARIANTS[name]["years"])
        comp = pd.read_parquet(vdir / "inflow_composite_daily.parquet")
        inflow = pd.Series(comp["value"].to_numpy(float), index=pd.to_datetime(comp["date"]))
        dd = diversion.make_daily(inflow, jiji, years)
        targets = diversion.annual_targets(jiji, years)
        log(f"{name}: benchmark grid")
        res, grid_list = validate.evaluate(dd, jiji, v3m, bench, years, lambda *a: None)
        pub = pd.read_csv(vdir / "statespace_loo_coverage.csv")
        rec = {}
        for label, grid in (("registered", tuple(SS_SIGMA_GRID)), ("extended", EXT_GRID)):
            statespace.SS_SIGMA_GRID = grid
            t0 = time.time()
            cov = loo_cov(dd, years, targets, res, grid_list, jm_all, cs)
            rec[label] = summary(cov)
            rec[label]["seconds"] = round(time.time() - t0, 1)
            if label == "registered":
                rec[label]["reproduces_published_max_abs"] = float(np.abs(
                    cov[["p05", "p50", "p95"]].to_numpy()
                    - pub[["p05", "p50", "p95"]].to_numpy()).max())
            log(f"{name} {label}: {rec[label]['hits']}/{rec[label]['n']} "
                f"({rec[label]['seconds']} s)")
        statespace.SS_SIGMA_GRID = tuple(SS_SIGMA_GRID)
        out[name] = rec
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(jsonable(out), f, indent=1)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
