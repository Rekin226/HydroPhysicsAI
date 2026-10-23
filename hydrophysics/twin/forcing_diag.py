"""Does a candidate forcing the model lacks explain the annual head anomalies?

The damping diagnostic (`docs/TWIN_EVIDENCE_2026-10.md`, section 4) left about a third of
the anomaly variance unexplained and named canal deliveries first among the forcings the
model does not see. The Jiji Weir's annual operating table (WRA, `水源運用 > 歷年營運水量
統計表`) gives the surface water delivered to the Changhua and Yunlin irrigation districts
each year since 2006. This module correlates each county's annual mean head anomaly
(observed minus the well's month-of-year climatology, by aquifer layer) with its district's
deliveries, with the weir's inflow (a basin-wide wetness proxy) and with deliveries after
the inflow is regressed out. Fourteen years give a screening statistic, not a forcing: a
partial correlation that survives says the record is worth acquiring at monthly
resolution, nothing more.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .coverage import assemble
from .drift_diag import fair_baselines
from .release import atomic_json, sha256

COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
DELIVERY = {"changhua": "changhua_irrigation_1e8_m3", "yunlin": "yunlin_irrigation_1e8_m3"}


def partial_corr(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, float, int]:
    """Pearson correlation of ``x`` and ``y`` after a linear fit on ``z`` is removed from
    each -> (r, p, n); NaN below six complete years."""
    d = pd.concat([x, y, z], axis=1).dropna()
    if len(d) < 6:
        return float("nan"), float("nan"), int(len(d))
    zz = d.iloc[:, 2].to_numpy(dtype=float)
    rx = d.iloc[:, 0] - np.polyval(np.polyfit(zz, d.iloc[:, 0], 1), zz)
    ry = d.iloc[:, 1] - np.polyval(np.polyfit(zz, d.iloc[:, 1], 1), zz)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p), int(len(d))


def annual_anomaly(obs: np.ndarray, clim: np.ndarray, dates: pd.DatetimeIndex,
                   groups: dict[str, np.ndarray], min_months: int = 6, min_values: int = 20
                   ) -> pd.DataFrame:
    """Year x group table of the mean head anomaly (m) over the group's wells."""
    anom = obs - clim
    years = np.asarray(dates.year)
    rows = []
    for year in sorted(set(years)):
        sel = years == year
        if sel.sum() < min_months:
            continue
        row = {"year": int(year)}
        for name, mask in groups.items():
            block = anom[mask][:, sel]
            row[name] = float(np.nanmean(block)) if np.isfinite(block).sum() >= min_values else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def run(forward: Path, stations: Path, historical_wells: Path, new_wells: Path,
        deliveries_csv: Path, out: Path) -> dict:
    a = assemble(forward, stations, historical_wells, new_wells, None)
    dates, fit = a["dates"], a["fit"]
    obs = np.where(np.isfinite(a["new"]), a["new"], a["historical"])
    clim = fair_baselines(obs, fit, month_offset=0)["clim"]
    table = pd.read_parquet(stations).drop_duplicates("sid").set_index("sid")
    county = np.array([str(table.CountyName.get(s, "?")) for s in a["ids"]])
    layers = np.asarray(a["layers"])
    groups = {f"{name}_L{layer}": (county == cn) & (layers == layer)
              for name, cn in COUNTIES.items() for layer in (1, 2)}
    annual = annual_anomaly(obs, clim, dates, groups)
    forcing = pd.read_csv(deliveries_csv).set_index("year").sort_index()
    joined = annual.join(forcing)
    results = {}
    for name in COUNTIES:
        for layer in (1, 2):
            col, dcol = f"{name}_L{layer}", DELIVERY[name]
            d = joined[[col, dcol, "inflow_1e8_m3"]].dropna()
            if len(d) < 6:
                continue
            r_d, p_d = stats.pearsonr(d[col], d[dcol])
            r_i, p_i = stats.pearsonr(d[col], d["inflow_1e8_m3"])
            r_p, p_p, n = partial_corr(joined[col], joined[dcol], joined["inflow_1e8_m3"])
            results[col] = {"years": int(len(d)), "wells": int(groups[col].sum()),
                            "r_deliveries": float(r_d), "p_deliveries": float(p_d),
                            "r_inflow": float(r_i), "p_inflow": float(p_i),
                            "partial_r_deliveries_given_inflow": r_p, "partial_p": p_p}
    report = {
        "schema": 1, "task": "annual-forcing-screen", "prospective_validation": False,
        "causal_policy_validation": False,
        "forcing": str(deliveries_csv), "forcing_sha256": sha256(deliveries_csv),
        "well_axis_forward_sha256": a["model_sha256"],
        "anomaly": "observed head minus the well's month-of-year climatology of the months "
                   "before the forecast origin; annual mean over the county's wells by layer",
        "caveat": "fourteen annual values per series; deliveries are themselves set from "
                  "water availability, so a partial correlation is a screen for a missing "
                  "forcing, not a causal estimate",
        "results": results,
        "annual_table": json.loads(joined.reset_index().round(3).to_json(orient="records")),
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "jiji_deliveries.json", report)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward", required=True, type=Path)
    ap.add_argument("--stations", required=True, type=Path)
    ap.add_argument("--historical-wells", required=True, type=Path)
    ap.add_argument("--new-wells", required=True, type=Path)
    ap.add_argument("--deliveries", required=True, type=Path,
                    help="CSV with year, inflow_1e8_m3, changhua_irrigation_1e8_m3, "
                         "yunlin_irrigation_1e8_m3")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    rep = run(args.forward, args.stations, args.historical_wells, args.new_wells,
              args.deliveries, args.out)
    print(json.dumps(rep["results"], indent=2))


if __name__ == "__main__":
    main()
