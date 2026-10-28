"""Score rung I1.X5NH.v1 on its declared gates: coastal township bounds 2023-25 (q0_nh vs Q0),
the column's leveling fit, and the rolling-origin no-harm gates (nudged vs free, 2016-21).
Writes results/dataprep/reports/I1_X5NH/report.json once and refuses to overwrite.

Model rates are the forward runs' baseline-scenario ensemble-mean subsidence (cm/yr, + =
sinking). Cells are rebuilt from each run's mask (row-major active cells, the twin's order).

Usage (repo root, after results/twin_runs/par31/run.sh):
    python dataprep_staging/x5_nudge/score_x5nh.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prereg_x5nh import (  # noqa: E402
    COAST,
    DISTAL_KM,
    GNSS_TO_LEVELING,
    LISTED_FLOOR,
    MAX_RATE,
    ORIGINS,
    POST_YEARS,
    SEED,
    TOL,
)

from hydrophysics.twin.leveling import load_panel  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_X5NH"
FWD = REPO / "results/twin_forward"
RUNS = REPO / "results/twin_runs"
PUB = REPO / "results/dataprep/subsidence_public_reports"
TOWNS = REPO / "data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json"
RINGS = REPO / "results/twin/ring_holdout_ref2023/rings_sites.csv"
GNSS = REPO / "results/dataprep/gnss_clean/full/v4"
RING_SITES = ("元長國小", "客厝國小")


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------
class Run:
    """A forward run's baseline subsidence (A, T) in cm, its months and cell centroids."""

    def __init__(self, name: str):
        z = np.load(FWD / f"datum_gate_{name}.npz", allow_pickle=True)
        names = [str(s) for s in z["scenario_names"]]
        self.s = z["subs_mean"][names.index("baseline")] * 100.0
        self.months = pd.DatetimeIndex(pd.to_datetime([str(d) for d in z["dates"]])
                                       ).to_period("M")
        self.origin = int(z["origin"])
        mask = z["mask"]
        ii, jj = np.nonzero(mask)
        x0, y0, dx = float(z["x0"]), float(z["y0"]), float(z["dx"])
        self.cx, self.cy = x0 + (jj + 0.5) * dx, y0 + (ii + 0.5) * dx
        self.cell = -np.ones(mask.shape, dtype="int64")
        self.cell[ii, jj] = np.arange(len(ii))
        self.x0, self.y0, self.dx = x0, y0, dx
        self.opts = json.loads(str(z["forward_options"]))

    def m(self, ym: str | pd.Period) -> int:
        p = pd.Period(ym, freq="M")
        if p not in self.months:
            raise SystemExit(f"month {p} is outside the run {self.months[0]}..{self.months[-1]}")
        return int(self.months.get_loc(p))

    def rate(self, a, b) -> np.ndarray:
        """(A,) cm/yr between months a and b."""
        ia, ib = self.m(a), self.m(b)
        return (self.s[:, ib] - self.s[:, ia]) * 12.0 / (ib - ia)

    def cell_of(self, x: float, y: float) -> int:
        j, i = int((x - self.x0) // self.dx), int((y - self.y0) // self.dx)
        if not (0 <= i < self.cell.shape[0] and 0 <= j < self.cell.shape[1]):
            return -1
        return int(self.cell[i, j])

    def slope(self, cell: int, months: pd.PeriodIndex) -> float:
        idx = np.array([self.m(p) for p in months])
        t = idx / 12.0
        return float(np.polyfit(t, self.s[cell, idx], 1)[0])


# ---------------------------------------------------------------------------------------
# townships
# ---------------------------------------------------------------------------------------
def township_cells(run: Run) -> dict[str, np.ndarray]:
    """{township -> active cell ids whose centroid lies inside it} for the coastal ones."""
    from matplotlib.path import Path as MPath
    from pyproj import Transformer

    t = json.loads(TOWNS.read_text())
    sc, tr = t["transform"]["scale"], t["transform"]["translate"]
    arcs = [np.cumsum(np.asarray(a, float), axis=0) * sc + tr for a in t["arcs"]]

    def ring(ids):
        pts = []
        for k in ids:
            a = arcs[k] if k >= 0 else arcs[~k][::-1]
            pts.extend(a if not pts else a[1:])
        return np.asarray(pts)

    T = Transformer.from_crs(4326, 3826, always_xy=True)
    xy = np.column_stack([run.cx, run.cy])
    out = {}
    for g in next(iter(t["objects"].values()))["geometries"]:
        p = g["properties"]
        name, county = p.get("TOWNNAME"), p.get("COUNTYNAME") or ""
        if name not in COAST or not any(c in county for c in ("彰化", "雲林")):
            continue
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        inside = np.zeros(len(xy), dtype=bool)
        for poly in polys:
            r = ring(poly[0])
            X, Y = T.transform(r[:, 0], r[:, 1])
            inside |= MPath(np.column_stack([X, Y])).contains_points(xy)
        out[name] = np.flatnonzero(inside)
    missing = sorted(set(COAST) - set(out))
    if missing:
        raise SystemExit(f"coastal townships not found in {TOWNS.name}: {missing}")
    return out


def bounds(years) -> pd.DataFrame:
    """Coastal township bounds of the given WRA survey years, with their intervals."""
    rows = []
    for v in ("preorigin", "postorigin"):
        b = pd.read_csv(PUB / v / "v1/township_bounds.csv")
        ci = pd.read_csv(PUB / v / "v1/county_intervals.csv")[
            ["county", "survey_year", "start_ym", "end_ym"]]
        rows.append(b.merge(ci, on=["county", "survey_year"], how="left"))
    b = pd.concat(rows, ignore_index=True)
    b = b[b["township"].isin(COAST) & b["survey_year"].isin(list(years))]
    b = b.drop_duplicates(["county", "survey_year", "township"])
    if b["start_ym"].isna().any():
        raise SystemExit("a township-year has no county interval")
    return b.reset_index(drop=True)


def flag(v) -> bool:
    """A CSV boolean that may be missing (NaN reads as False, never as True)."""
    return bool(v) if pd.notna(v) else False


def check_bounds(run: Run, cells: dict, b: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in b.itertuples():
        peak = float(np.max(run.rate(r.start_ym, r.end_ym)[cells[r.township]]))
        listed, anyl = flag(r.listed_by_all), flag(r.listed_by_any)
        ambiguous = flag(r.ambiguous) or (anyl and not listed)
        upper = float(r.peak_rate_upper_cm_yr)
        out.append({"township": r.township, "survey_year": int(r.survey_year),
                    "interval": f"{r.start_ym}..{r.end_ym}", "peak_cm_yr": round(peak, 3),
                    "listed": listed, "ambiguous": ambiguous, "upper_cm_yr": upper,
                    "lower_met": (peak >= LISTED_FLOOR - TOL) if listed and not ambiguous
                    else None,
                    "upper_met": None if ambiguous else bool(peak <= upper + TOL)})
    return pd.DataFrame(out)


def bound_counts(df: pd.DataFrame) -> dict:
    lo = df["lower_met"].dropna()
    up = df["upper_met"].dropna()
    return {"lower_n": int(len(lo)), "lower_met": int(lo.astype(bool).sum()),
            "upper_n": int(len(up)), "upper_broken": int((~up.astype(bool)).sum())}


# ---------------------------------------------------------------------------------------
# leveling benchmark pairs (rolling origins)
# ---------------------------------------------------------------------------------------
def benchmark_pairs(panel: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Each benchmark's last survey in the origin's year after the origin and its last
    survey the next year -> observed rate (cm/yr, + = sinking)."""
    o = pd.Period(origin, freq="M")
    y = o.year
    w = panel[(panel["datetime"] > o.end_time) & (panel["datetime"].dt.year <= y + 1)]
    rows = []
    for sid, g in w.groupby("sid"):
        a = g[g["datetime"].dt.year == y]
        b = g[g["datetime"].dt.year == y + 1]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[-1], b.iloc[-1]
        yrs = (b["datetime"] - a["datetime"]).days / 365.25
        if yrs < 0.5:
            continue
        rate = (a["elev_m"] - b["elev_m"]) * 100.0 / yrs
        if abs(rate) > MAX_RATE:
            continue
        rows.append({"sid": sid, "x": a["x"], "y": a["y"], "ta": a["datetime"],
                     "tb": b["datetime"], "obs": rate})
    return pd.DataFrame(rows)


def score_pairs(run: Run, pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.copy()
    p["cell"] = [run.cell_of(x, y) for x, y in zip(p["x"], p["y"], strict=True)]
    p = p[p["cell"] >= 0].copy()
    ma = np.array([run.m(t.to_period("M")) for t in p["ta"]])
    mb = np.array([run.m(t.to_period("M")) for t in p["tb"]])
    yrs = (p["tb"] - p["ta"]).dt.days.to_numpy() / 365.25
    p["mod"] = (run.s[p["cell"].to_numpy(), mb] - run.s[p["cell"].to_numpy(), ma]) / yrs
    p["distal"] = run.cx[p["cell"].to_numpy()] < DISTAL_KM * 1000.0
    if (ma <= run.origin).any():
        raise SystemExit("a scored survey precedes the forecast origin")
    return p


def rmse(d: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((d["mod"] - d["obs"]) ** 2))) if len(d) else float("nan")


# ---------------------------------------------------------------------------------------
# reported
# ---------------------------------------------------------------------------------------
def column_r2(arm: str) -> dict:
    p = RUNS / f"stage3_datum_gate_{arm}/coupled_leveling/vep_zonal_leveling.json"
    d = json.loads(p.read_text())
    if int(d.get("fold_seed", 0)) != SEED:
        raise SystemExit(f"{p}: fold seed {d.get('fold_seed')}, needs {SEED}")
    return {"leveling_oof": float(d["r2_outoffold"]), "leveling_insample": d["leveling"]["r2"],
            "hindcast_gain": d.get("hindcast_gain", 0.0),
            "withhold_surveys": d.get("withhold_surveys")}


def rings(runs: dict[str, Run]) -> list[dict]:
    r = pd.read_csv(RINGS)
    st = pd.read_csv(REPO / "chou-shui-data/data/mlcw_stations.csv").set_index("sub_id")
    out = []
    for site in RING_SITES:
        row = r[r["site"] == site].iloc[0]
        x, y = st.loc[site, ["x", "y"]]
        rec = {"site": site, "window": f"{row.ref_month}..{row.end_month}",
               "obs_cm_yr": float(row.obs_rate_cm_yr)}
        for k, run in runs.items():
            c = run.cell_of(float(x), float(y))
            rec[k] = round(float(run.rate(row.ref_month, row.end_month)[c]), 3)
        out.append(rec)
    return out


def gnss(runs: dict[str, Run]) -> list[dict]:
    st = pd.read_csv(GNSS / "stations.csv")
    mon = pd.concat([pd.read_parquet(GNSS / f / "monthly.parquet") for f in ("wra", "ngl")])
    mon = mon[mon["flag"].isin([0, 1])]
    mon["p"] = pd.PeriodIndex(mon["date"], freq="M")
    win = pd.period_range("2023-01", "2025-03", freq="M")
    out = []
    any_run = next(iter(runs.values()))
    for s in st.itertuples():
        if pd.isna(s.twin_cell) or s.twin_cell < 0:
            continue
        c = int(s.twin_cell)
        if any_run.cx[c] >= DISTAL_KM * 1000.0:
            continue
        g = mon[(mon["station"] == s.station) & mon["p"].isin(win)]
        if len(g) < 18:
            continue
        t = np.array([(p - win[0]).n for p in g["p"]]) / 12.0
        rate = -float(np.polyfit(t, g["value"].to_numpy() * 100.0, 1)[0])
        rec = {"station": s.station, "family": s.family, "cell": c, "n_months": int(len(g)),
               "gnss_frame_cm_yr": round(rate, 3),
               "leveling_frame_cm_yr": round(rate - GNSS_TO_LEVELING, 3)}
        for k, run in runs.items():
            rec[k] = round(run.slope(c, pd.PeriodIndex(g["p"])), 3)
        out.append(rec)
    return out


def county_rates(run: Run) -> dict:
    r = run.rate("2023-01", "2025-12") if run.months[-1] >= pd.Period("2025-12", "M") else None
    if r is None:
        return {}
    distal = run.cx < DISTAL_KM * 1000.0
    return {"distal_mean_2023_25": round(float(r[distal].mean()), 3),
            "rest_mean_2023_25": round(float(r[~distal].mean()), 3),
            "rising_share_distal": round(float((r[distal] < -0.1).mean()), 3),
            "rising_share_all": round(float((r < -0.1).mean()), 3)}


# ---------------------------------------------------------------------------------------
def main() -> int:
    if (OUT / "report.json").exists():
        raise SystemExit(f"{OUT}/report.json exists: scored once")
    q0, nh = Run("q0"), Run("q0_nh")
    if nh.opts.get("extend_record_to") is None or q0.opts.get("extend_record_to"):
        raise SystemExit("q0_nh must be the extended run and q0 the registered one")
    cells = township_cells(q0)
    post = bounds(POST_YEARS)
    tb = {"q0": check_bounds(q0, cells, post), "q0_nh": check_bounds(nh, cells, post)}
    post_counts = {k: bound_counts(v) for k, v in tb.items()}
    col = {"q0": column_r2("q0"), "q0_nh": column_r2("q0_nh")}

    panel = load_panel(os.environ.get("HYDROMIND_GW_DATA", str(REPO / "chou-shui-data/data")))
    roll, roll_tb = [], []
    for o in ORIGINS:
        y = int(o[:4])
        f, n = Run(f"q0_r{y}f"), Run(f"q0_r{y}n")
        for run, arm in ((f, "free"), (n, "nudged")):
            if run.opts.get("record_end") != o:
                raise SystemExit(f"q0_r{y}: record_end {run.opts.get('record_end')} != {o}")
            d = check_bounds(run, cells, bounds([y + 1]))
            d["origin"], d["arm"] = o, arm
            roll_tb.append(d)
        pairs = benchmark_pairs(panel, o)
        if pairs.empty:
            continue
        for run, arm in ((f, "free"), (n, "nudged")):
            p = score_pairs(run, pairs)
            p["origin"], p["arm"] = o, arm
            roll.append(p)
    roll = pd.concat(roll, ignore_index=True)
    roll_tb = pd.concat(roll_tb, ignore_index=True)
    pooled = {}
    for arm in ("free", "nudged"):
        a = roll[roll["arm"] == arm]
        pooled[arm] = {"rmse_distal": rmse(a[a["distal"]]), "rmse_all": rmse(a),
                       "rmse_mid_prox": rmse(a[~a["distal"]]),
                       "n_distal": int(a["distal"].sum()), "n_all": int(len(a)),
                       "bias_distal": float((a[a["distal"]]["mod"] - a[a["distal"]]["obs"]
                                             ).mean()),
                       "by_origin": {o: {"rmse_distal": rmse(g[g["distal"]]),
                                         "rmse_all": rmse(g)}
                                     for o, g in a.groupby("origin")},
                       "coastal_lower_met": bound_counts(roll_tb[roll_tb["arm"] == arm])}

    G = {"G1 coastal lower bounds 2023-25 >= 6": post_counts["q0_nh"]["lower_met"] >= 6,
         "G2 coastal upper bounds broken <= 3": post_counts["q0_nh"]["upper_broken"] <= 3,
         "G3 column leveling oof >= Q0 - 0.02":
             col["q0_nh"]["leveling_oof"] >= col["q0"]["leveling_oof"] - 0.02,
         "G4 rolling RMSE distal nudged <= 1.05 x free":
             pooled["nudged"]["rmse_distal"] <= 1.05 * pooled["free"]["rmse_distal"],
         "G4 rolling RMSE all nudged <= 1.05 x free":
             pooled["nudged"]["rmse_all"] <= 1.05 * pooled["free"]["rmse_all"],
         "G5 rolling coastal lower bounds nudged >= free":
             pooled["nudged"]["coastal_lower_met"]["lower_met"]
             >= pooled["free"]["coastal_lower_met"]["lower_met"]}
    G = {k: bool(v) for k, v in G.items()}
    ok = all(G.values())
    res = {"declaration": "I1.X5NH.v1", "label": "design-time (no fresh units)",
           "tol_cm_yr": TOL, "coastal_bounds_2023_25": post_counts,
           "coastal_township_years": {k: v.to_dict("records") for k, v in tb.items()},
           "columns": col, "rolling": pooled,
           "rolling_township_years": roll_tb.to_dict("records"),
           "reported": {"rings": rings({"q0": q0, "q0_nh": nh}),
                        "gnss_distal": gnss({"q0": q0, "q0_nh": nh}),
                        "fan_2023_25": {"q0": county_rates(q0), "q0_nh": county_rates(nh)}},
           "gates": G, "passes_all": ok,
           "decision": ("q0_nh becomes the policy-run baseline, labelled design-time until "
                        "the user rules on it" if ok else
                        "no change; Q0 stays. The meaning follows the declaration's "
                        "interpretation, read with the failed gates, not this line.")}
    text = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "report.json", "x", encoding="utf-8") as fh:   # scored once
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
