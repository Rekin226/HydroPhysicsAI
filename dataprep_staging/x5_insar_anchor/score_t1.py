"""Score X5.insar_leveling_2020_2021.v1 (test X5.T1) ONCE.

Three steps, each its own short process:

    flock <lock> python -I score_t1.py assert    prereg3.assert_can_score (logs the score look)
    python -I score_t1.py score                  reads the leveling rows, writes the report once
    flock <lock> python -I score_t1.py ledger    ledger rows of the reported numbers

``score`` refuses to run unless ``assert`` succeeded (its record file and a 'score' look of
this test in looks.jsonl) and the registered predictions are unchanged; it refuses to
overwrite results/dataprep/reports/X5.insar_leveling_2020_2021.v1/report.json.

Registered rule (x5const): leveling increment = elev(2021-05) - elev(2020-05) in mm (+ = up)
of the benchmarks surveyed in both months; screens: remark marks a disturbed monument
(不列入比較, 異動, 移動) or |increment| > 200 mm; scored = finite primary prediction.
PASS iff n >= 200, OLS slope of the InSAR increment on the leveling increment in [0.8, 1.2]
and RMSE(InSAR - leveling) <= 10 mm, all on the PRIMARY column ``p2_offset_scaled``.
Declared alternatives A1-A3, the diagnostic raw column, county splits, Deming / Theil-Sen /
reverse slopes and the spatial block bootstrap are reported and never gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "s4_insar")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import x5const as K  # noqa: E402
import x5lib as X  # noqa: E402
from common.paths import LEDGER_ROWS, rel_to_repo  # noqa: E402

PRED = K.OUT / "t1_predictions" / "predictions.csv"
SCORE_DIR = K.OUT / "t1_score"
ASSERT_JSON = SCORE_DIR / "assert.json"
REPORT_DIR = K.REPORTS_DIR / K.PREREG_T1
REPORT = REPORT_DIR / "report.json"
PRIMARY = "p2_offset_scaled"
ALTS = {"A1_p2_offset_unscaled": "p2_offset_unscaled", "A2_p2_plane_scaled": "p2_plane_scaled",
        "A3_v3_offset_scaled": "v3_offset_scaled"}
DIAG = {"D_raw_unanchored_p2_scaled": "p2_raw_scaled"}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def do_assert() -> int:
    from harness import prereg3
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    if ASSERT_JSON.exists():
        raise SystemExit(f"{ASSERT_JSON} exists; the test is scored once")
    r = prereg3.assert_can_score(K.PREREG_T1, K.WP, test_id=K.T1_TEST_ID)
    rec = {"prereg": K.PREREG_T1, "sha256": r["sha256"], "tests": r["tests"],
           "inputs_verified": r["inputs_verified"],
           "post_registration_looks": r["post_registration_looks"],
           "unverifiable_prior_looks": r["unverifiable_prior_looks"]}
    with open(ASSERT_JSON, "x", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(rec, indent=1, ensure_ascii=False, default=str)[:3000])
    return 0


def _score_look_logged() -> bool:
    from harness.prereg2 import read_chained
    return any(lk.get("prereg_id") == K.PREREG_T1 and lk.get("test_id") == K.T1_TEST_ID
               and lk.get("purpose") == "score"
               for lk in read_chained(K.PREREG_DIR / "looks.jsonl"))


def leveling_increments(sids) -> pd.DataFrame:
    p = pd.read_parquet(K.LEV_PANEL, columns=["sid", "datetime", "elev_m"])
    p["ym"] = p["datetime"].dt.strftime("%Y-%m")
    p = p[p["sid"].isin(set(sids)) & p["ym"].isin(K.LEV_MONTHS)]
    w = p.groupby(["sid", "ym"])["elev_m"].mean().unstack("ym")
    inc = (w[K.LEV_MONTHS[1]] - w[K.LEV_MONTHS[0]]) * 1000.0
    return inc.rename("lev_mm").reset_index()


def metrics_block(d: pd.DataFrame, col: str) -> dict:
    ok = np.isfinite(d[col].to_numpy(float))
    x, y = d.loc[ok, "lev_mm"].to_numpy(float), d.loc[ok, col].to_numpy(float)
    m = X.core_metrics(x, y)
    m["verdict_if_primary_rule"] = X.t1_verdict(m)
    m["deming_slope"] = X.deming(x, y)
    m["theil_sen_slope"] = X.theil_sen(x, y)
    m["reverse_ols_slope_leveling_on_insar_inverted"] = 1.0 / X.ols(y, x)[0]
    m["bootstrap"] = X.block_bootstrap(x, y, d.loc[ok, "x"], d.loc[ok, "y"])
    m["by_county"] = {}
    for c, g in d[ok].groupby("county"):
        if len(g) >= 20:
            mc = X.core_metrics(g["lev_mm"], g[col])
            m["by_county"][c] = mc
    return m


def do_score() -> int:
    if REPORT.exists():
        raise SystemExit(f"{REPORT} exists; scored once")
    if not ASSERT_JSON.exists() or not _score_look_logged():
        raise SystemExit("assert_can_score has not run for this test; run 'assert' first")
    a = json.loads(ASSERT_JSON.read_text(encoding="utf-8"))
    from harness.prereg2 import load_prereg
    doc, psha = load_prereg(K.PREREG_T1)
    if psha != a["sha256"]:
        raise SystemExit("registration hash differs from the asserted one")
    reg_pred = doc["registered_predictions"]
    if sha(PRED) != reg_pred["sha256"]:
        raise SystemExit("predictions changed since registration")
    pred = pd.read_csv(PRED)
    lev = leveling_increments(pred["sid"])
    d = pred.merge(lev, on="sid", how="left")
    screen = {"n_both_surveys": int(len(d))}
    bad_rem = d["remark"].fillna("").apply(lambda s: any(k in s for k in K.LEV_BAD_REMARK))
    gross = d["lev_mm"].abs() > K.LEV_MAX_ABS_MM
    screen["excluded_remark"] = int(bad_rem.sum())
    screen["excluded_gross"] = int((gross & ~bad_rem).sum())
    d = d[~bad_rem & ~gross & np.isfinite(d["lev_mm"])].copy()
    screen["after_screens"] = int(len(d))
    screen["with_primary_prediction"] = int(np.isfinite(d[PRIMARY]).sum())
    prim = metrics_block(d, PRIMARY)
    verdict = X.t1_verdict(prim)
    res = {
        "prereg": K.PREREG_T1, "prereg_sha256": psha, "test_id": K.T1_TEST_ID,
        "label": "registered test, scored once; unit declared fresh for InSAR-alone at "
                 "registration (see the registration's exposure_disclosure)",
        "rule": {"n_min": K.T1_MIN_N, "slope_band": list(K.T1_SLOPE),
                 "rmse_max_mm": K.T1_RMSE_MM, "estimator": "OLS slope of the anchored InSAR "
                 "increment on the leveling increment; RMSE of InSAR - leveling",
                 "primary_column": PRIMARY},
        "screens": screen,
        "primary": prim, "verdict": verdict,
        "alternatives_reported_not_gating": {k: metrics_block(d, c) for k, c in ALTS.items()},
        "diagnostics_reported_not_gating": {k: metrics_block(d, c) for k, c in DIAG.items()},
        "assert": {"post_registration_looks": a["post_registration_looks"],
                   "unverifiable_prior_looks": a["unverifiable_prior_looks"]},
        "inputs": {"predictions": {"path": rel_to_repo(PRED), "sha256": sha(PRED)},
                   "leveling_panel": {"path": rel_to_repo(K.LEV_PANEL),
                                      "sha256": sha(K.LEV_PANEL)}},
        "code": {rel_to_repo(HERE / f): sha(HERE / f)
                 for f in ("score_t1.py", "x5lib.py", "x5const.py")},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = d[["sid", "county", "lev_mm", PRIMARY, *ALTS.values(), *DIAG.values()]]
    pairs.to_csv(SCORE_DIR / "pairs.csv", index=False)
    with open(REPORT, "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({"verdict": verdict, "n": prim["n"],
                      "slope": prim["slope_ols_insar_on_leveling"], "rmse": prim["rmse_mm"],
                      "bias": prim["bias_mm"], "r": prim["r"]}, indent=1))
    print("wrote", rel_to_repo(REPORT))
    return 0


def do_ledger() -> int:
    from common import ledger
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    pv = f"prereg-sha256:{r['prereg_sha256']}"
    srcs = [(rel_to_repo(K.LEV_PANEL), "validator"), (rel_to_repo(PRED), "input"),
            (rel_to_repo(K.P2_DIR), "input"), (rel_to_repo(K.GNSS), "input")]
    rows = []
    for num in ("n", "slope_ols_insar_on_leveling", "rmse_mm", "bias_mm", "r"):
        for s, role in srcs:
            rows.append({"number_id": f"X5.T1.primary.{num}", "source": s, "role": role,
                         "product_version": pv,
                         "note": f"X5.T1 primary = {r['primary'][num]}; report "
                                 f"{rel_to_repo(REPORT)}"})
    p = ledger.append_rows(K.WP, rows, rows_dir=LEDGER_ROWS)
    print("ledger", rel_to_repo(p), len(rows))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["assert", "score", "ledger"])
    a = ap.parse_args(argv)
    return {"assert": do_assert, "score": do_score, "ledger": do_ledger}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
