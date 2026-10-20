"""Score gate set G2 (``I1.G2.fresh_partition.v1``) for the fresh-partition runs R0, R1c and
R1e-pw, exactly as declared, and apply the declared decision rule.

Metrics (declaration ``thresholds.metrics``):

* ``kfold_r2``: ``stage3_flow.csv`` ``r2_kfold``;
* ``county_corr``: per county, median over held-out entries with >= 12 scored months of the
  correlation between predicted and observed own-mean anomalies;
* ``county_mean_rmse``: per county, each month the mean over the county's held-out entries of
  the own-mean anomaly (observed and predicted; each entry's mean over the months where both are
  finite), RMSE between the two county-mean series over months with >= 5 entries;
* ``leveling_r2``: ``coupled_leveling/vep_zonal_leveling.json`` ``r2_outoffold`` (site folds
  ``--seed 2027``; the file must record ``fold_seed`` 2027).

The four I1.R1* gates are reported, never gating. Pass ``--design`` to run the same code on the
design-time runs (a code check on spent units; nothing is written).

Run from the repository root:
    python dataprep_staging/i1/score_g2.py [--design]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

FRESH = {"R0": "f0", "R1c": "fcz", "R1e-pw": "fpw"}
DESIGN = {"R0": "cwa", "R1c": "ccz", "R1e-pw": "cpw"}
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
OUT = REPO / "results/dataprep/reports/I1_G2_fresh_partition"
WEIGHTS = {"kfold_r2": 0.2, "county_corr:changhua": 0.2, "county_corr:yunlin": 0.1,
           "county_mean_rmse:changhua": 0.2, "county_mean_rmse:yunlin": 0.1,
           "leveling_r2": 0.2}


def metrics(suffix: str, st: pd.DataFrame, expect_seed: int | None) -> dict:
    d = REPO / f"results/twin_runs/stage3_datum_gate_{suffix}"
    flow = pd.read_csv(d / "stage3_flow.csv").iloc[0]
    if expect_seed is not None and int(flow["seed"]) != expect_seed:
        raise SystemExit(f"{d}: fold seed {flow['seed']}, the declaration needs {expect_seed}")
    lev = json.loads((d / "coupled_leveling/vep_zonal_leveling.json").read_text())
    if expect_seed is not None and int(lev.get("fold_seed", 0)) != expect_seed:
        raise SystemExit(f"{d}: leveling fold seed {lev.get('fold_seed', 0)}, needs "
                         f"{expect_seed}")
    z = np.load(d / "stage3_per_entry.npz", allow_pickle=True)
    w = pd.read_csv(d / "stage3_kfold_wells.csv", dtype={"sid": str}).set_index("entry")
    county = st.loc[w.loc[z["entry"], "sid"].to_numpy(), "CountyName"].to_numpy()
    out = {"kfold_r2": float(flow["r2_kfold"]), "leveling_r2": float(lev["r2_outoffold"])}
    for cname, cc in COUNTIES.items():
        corrs, rmses, amps, oa_rows, pa_rows = [], [], [], [], []
        for i in np.where(county == cc)[0]:
            o, p = z["obs"][i].astype(float), z["pred"][i].astype(float)
            m = np.isfinite(o) & np.isfinite(p)
            if m.sum() < 12:
                continue
            oa = np.where(m, o - o[m].mean(), np.nan)
            pa = np.where(m, p - p[m].mean(), np.nan)
            corrs.append(np.corrcoef(oa[m], pa[m])[0, 1])
            rmses.append(np.sqrt(np.mean((oa[m] - pa[m]) ** 2)))
            amps.append(pa[m].std() / oa[m].std())
            oa_rows.append(oa)
            pa_rows.append(pa)
        obs_a, pred_a = np.array(oa_rows), np.array(pa_rows)
        n = np.isfinite(obs_a).sum(axis=0)
        keep = n >= 5
        co, cp = np.nanmean(obs_a[:, keep], axis=0), np.nanmean(pred_a[:, keep], axis=0)
        out[f"county_corr:{cname}"] = float(np.median(corrs))
        out[f"county_mean_rmse:{cname}"] = float(np.sqrt(np.mean((co - cp) ** 2)))
        out[f"reported:single_well_rmse:{cname}"] = float(np.mean(rmses))
        out[f"reported:amplitude:{cname}"] = float(np.median(amps))
        out[f"reported:county_mean_corr:{cname}"] = float(np.corrcoef(co, cp)[0, 1])
    bz = lev.get("r2_outoffold_by_zone") or {}
    for k in ("county:changhua", "county:rest"):
        if isinstance(bz.get(k), dict):
            out[f"reported:leveling_r2_{k}"] = bz[k].get("r2")
    return out


def gates(m: dict, r0: dict) -> dict:
    g = {"kfold_r2": m["kfold_r2"] >= r0["kfold_r2"] - 0.01,
         "leveling_r2": m["leveling_r2"] >= r0["leveling_r2"] - 0.02}
    for c in COUNTIES:
        g[f"county_corr:{c}"] = m[f"county_corr:{c}"] >= r0[f"county_corr:{c}"] - 0.02
        g[f"county_mean_rmse:{c}"] = (m[f"county_mean_rmse:{c}"]
                                      <= r0[f"county_mean_rmse:{c}"] * 1.02)
    return g


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", action="store_true", help="code check on the design-time runs")
    a = ap.parse_args(argv)
    runs, seed = (DESIGN, None) if a.design else (FRESH, 2027)
    st = pd.read_parquet(STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    M = {k: metrics(v, st, seed) for k, v in runs.items()}
    res = {"declaration": "I1.G2.fresh_partition.v1", "runs": runs, "metrics": M,
           "gates": {}, "weighted_score": {}}
    for k in ("R1c", "R1e-pw"):
        g = gates(M[k], M["R0"])
        res["gates"][k] = {"each": g, "passes_all": bool(all(g.values()))}
        # standardised deltas: improvement in the gate's direction, over R0's magnitude
        sc = 0.0
        for name, wt in WEIGHTS.items():
            sign = -1.0 if name.startswith("county_mean_rmse") else 1.0
            sc += wt * sign * (M[k][name] - M["R0"][name]) / max(abs(M["R0"][name]), 1e-9)
        res["weighted_score"][k] = sc
    passing = [k for k in ("R1c", "R1e-pw") if res["gates"][k]["passes_all"]]
    promotable = [k for k in passing if k != "R1e-pw"]
    if promotable:
        best = max(promotable, key=lambda k: res["weighted_score"][k])
        res["decision"] = f"{best} becomes the deliverable recipe (passes every G2 gate)"
    elif "R1e-pw" in passing:
        res["decision"] = ("R1e-pw passes every G2 gate but stays diagnostic (P6 sensitivity "
                           "member): promotion needs the user's ruling")
    elif (M["R1c"]["leveling_r2"] >= M["R0"]["leveling_r2"] - 0.02
          and M["R1c"]["leveling_r2"] > M["R0"]["leveling_r2"]):
        res["decision"] = ("no rung passes; the deliverable keeps R0's heads and takes its "
                           "subsidence (VEP column) from R1c")
    else:
        res["decision"] = "no rung passes; R0 stays the deliverable"
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if not a.design:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
