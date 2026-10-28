"""X5.T1 post-failure follow-up (the one allowed; exploratory; never validation; cannot
promote): where does the T1 bias come from?

T1 failed on RMSE through a fan-wide bias (anchored InSAR minus leveling = -15.4 mm over
2020-05 -> 2021-05) while the unanchored InSAR (off-fan bedrock reference) showed the same
bias. This splits the bias at GNSS stations into InSAR - GNSS and GNSS - leveling:

* GNSS May increment: mean of the daily heights (flags 0-1, anchor set of T1) within May 15
  +-30 d of 2021 minus that of 2020, scaled to 365 d by the mean dates;
* leveling: the T1-screened increments of the benchmarks within 1.0 km of the station
  (median when several);
* InSAR: run P2 3 x 3 pixels at the station, unanchored and offset-anchored LEAVING THE
  STATION OUT of the median polish, same May windows and scaling as T1.

Reported: per station, and medians over stations of GNSS - leveling, InSAR - GNSS and
InSAR - leveling. The leveling row was spent by X5.T1 (scored 2026-10-25).

    flock <lock> python -I followup_t1.py declare    (declaration, before any number)
    python -I followup_t1.py run                     (writes the report once)
    flock <lock> python -I followup_t1.py look       (logs the post-failure look)
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
from common.paths import rel_to_repo  # noqa: E402

PREREG = "X5.t1_postfailure_gnss_leveling.v1"
RADIUS_M = 1000.0
OUTD = K.OUT / "t1_followup"
REPORT_DIR = K.REPORTS_DIR / PREREG
REPORT = REPORT_DIR / "report.json"
PAIRS = K.OUT / "t1_score" / "pairs.csv"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def declare() -> int:
    from harness import prereg3
    NA = prereg3.not_applicable
    d = {
        "id": PREREG, "wp": K.WP, "kind": "declaration",
        "title": "X5.T1 post-failure follow-up: split the InSAR - leveling bias at GNSS "
                 "stations into InSAR - GNSS and GNSS - leveling (exploratory)",
        "tests": NA("post-failure, exploratory; never validation; cannot promote"),
        "checks": {"design": __doc__.split("\n\n", 1)[1].split("\n    flock")[0],
                   "radius_m": RADIUS_M,
                   "reading_rule": "if median GNSS - leveling is within 5 mm of the T1 bias "
                                   "(-15.4 mm) and median InSAR - GNSS within 5 mm of 0, the "
                                   "T1 failure is read as a GNSS/InSAR-frame vs leveling-"
                                   "datum difference, not an InSAR anchoring defect; "
                                   "otherwise as an InSAR defect at the stations"},
        "zones": NA("stations"), "priors": NA("none"), "burn_in": NA("none"),
        "folds": NA("none"), "fresh_units": NA("spent unit (X5.T1)"),
        "thresholds": NA("reading rule above; no gate"), "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": [rel_to_repo(HERE / "followup_t1.py"), rel_to_repo(PAIRS)],
        "interpretation": "diagnoses the T1 failure only; post-failure, exploratory; never "
                          "validation; cannot promote",
        "written_by": rel_to_repo(Path(__file__)),
    }
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


def gnss_increment(s: pd.Series) -> tuple[float, float, int, int]:
    out = []
    for y in K.T1_YEARS:
        c = pd.Timestamp(f"{y}-{K.MAY_CENTRE}")
        w = s.loc[c - pd.Timedelta(days=K.MAY_HALF_D):c + pd.Timedelta(days=K.MAY_HALF_D)]
        w = w.dropna()
        days = (w.index - pd.Timestamp("1970-01-01")).days.to_numpy(float)
        out.append((float(w.mean()) if len(w) else np.nan,
                    float(days.mean()) if len(w) else np.nan, len(w)))
    (v0, d0, n0), (v1, d1, n1) = out
    return (v1 - v0) * K.NOMINAL_DAYS / (d1 - d0), d1 - d0, n0, n1


def run() -> int:
    if REPORT.exists():
        raise SystemExit(f"{REPORT} exists; written once")
    st = X.anchor_stations()
    series = X.gnss_daily(st, last_day="2022-12-31")
    pairs = pd.read_csv(PAIRS)
    bench = X.bench_geometry().set_index("sid")
    pairs = pairs.join(bench[["x", "y"]], on="sid")
    p2 = X.load_p2()
    S = X.station_inputs(p2, st, series)
    rows = []
    for k, r in st.iterrows():
        s = r["station"]
        if s not in series:
            continue
        g_inc, span, n0, n1 = gnss_increment(series[s])
        dist = np.hypot(pairs["x"] - r["x_twd97"], pairs["y"] - r["y_twd97"])
        near = pairs[dist <= RADIUS_M]
        keep = [j for j in range(len(st)) if j != k]
        corr = X.epoch_corrections(S["I"][:, keep], S["G"][:, keep], S["x"][keep],
                                   S["y"][keep], "offset")
        Ist = S["I"][:, [k]]
        A = Ist + X.correction_at(corr, [r["x_twd97"]], [r["y_twd97"]])
        i_raw = X.may_increment(Ist, p2["dates"], *K.T1_YEARS)["inc"][0]
        i_anc = X.may_increment(A, p2["dates"], *K.T1_YEARS)["inc"][0]
        rows.append({"station": s, "family": r["family"], "n_days_2020": n0,
                     "n_days_2021": n1, "gnss_inc_mm": g_inc,
                     "n_benchmarks_1km": int(len(near)),
                     "lev_inc_mm": float(near["lev_mm"].median()) if len(near) else np.nan,
                     "insar_raw_inc_mm": float(i_raw), "insar_anchored_loso_inc_mm": float(i_anc)})
    t = pd.DataFrame(rows)
    t["gnss_minus_lev"] = t["gnss_inc_mm"] - t["lev_inc_mm"]
    t["insar_anc_minus_gnss"] = t["insar_anchored_loso_inc_mm"] - t["gnss_inc_mm"]
    t["insar_raw_minus_gnss"] = t["insar_raw_inc_mm"] - t["gnss_inc_mm"]
    t["insar_anc_minus_lev"] = t["insar_anchored_loso_inc_mm"] - t["lev_inc_mm"]
    OUTD.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUTD / "stations.csv", index=False)

    def med(c):
        v = t[c].dropna()
        return {"median_mm": float(v.median()) if len(v) else None, "n": int(len(v)),
                "iqr_mm": [float(v.quantile(.25)), float(v.quantile(.75))] if len(v) else None}

    summ = {k: med(k) for k in ("gnss_minus_lev", "insar_anc_minus_gnss",
                                "insar_raw_minus_gnss", "insar_anc_minus_lev")}
    t1_bias = json.loads((K.REPORTS_DIR / K.PREREG_T1 / "report.json").read_text(
        encoding="utf-8"))["primary"]["bias_mm"]
    gl, ig = summ["gnss_minus_lev"]["median_mm"], summ["insar_anc_minus_gnss"]["median_mm"]
    datum = (gl is not None and ig is not None and abs(gl - t1_bias) <= 5.0
             and abs(ig) <= 5.0)
    res = {"prereg": PREREG, "label": "post-failure, exploratory; never validation; cannot "
                                      "promote",
           "t1_primary_bias_mm": t1_bias, "radius_m": RADIUS_M, "summary": summ,
           "reading_rule_outcome": ("frame/datum difference between GNSS-InSAR and leveling"
                                    if datum else "not a pure datum difference (see medians)"),
           "stations_csv": rel_to_repo(OUTD / "stations.csv"),
           "inputs": {"pairs": {"path": rel_to_repo(PAIRS), "sha256": sha(PAIRS)}},
           "code": {rel_to_repo(HERE / "followup_t1.py"): sha(HERE / "followup_t1.py")}}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(t.round(1).to_string())
    return 0


def look() -> int:
    from harness import prereg3
    rec = prereg3.record_look(
        [K.T1_UNIT], "leveling:2020-05..2021-05", K.WP, "post_failure_exploratory",
        prereg_id=PREREG, note="X5.T1 follow-up: GNSS vs leveling within 1 km of the anchor "
                               "stations (spent unit)")
    print(rec["time"], rec["purpose"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["declare", "run", "look"])
    a = ap.parse_args(argv)
    return {"declare": declare, "run": run, "look": look}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
