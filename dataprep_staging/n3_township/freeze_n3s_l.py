"""Freeze the N3S-L arms (ZR, SL mean3_eb, Z) before registration; the lead runs this.

Steps (repository root; each refuses to overwrite):

    python -I dataprep_staging/n3_township/freeze_n3s_l.py sl-panel
        -> results/dataprep/leveling_statlayer_history/preorigin/v<N>/ (MANIFEST: use
           statistical_layer_history, window 2007-01..2021-12; the column-calibration panel)
    python -I dataprep_staging/n3_township/freeze_n3s_l.py freeze --sl-product <dir>
        --out results/dataprep/reports/N3S-L/freeze [--pier-xy <csv>]
    python -I dataprep_staging/n3_township/freeze_n3s_l.py verify --sl-product <dir>
        --out results/dataprep/reports/N3S-L/freeze [--pier-xy <csv>]

What each step reads (and nothing else):

* ``sl-panel``  the pre-origin leveling panel (chou-shui-data ls_cache, the column's calibration
  leveling), rows dated 2007-01..2021-12 only; written as a product granting
  ``statistical_layer_history`` (SL is the statistical layer, tested in L1) and ``baseline``
  (SL is the comparator ZR must beat in L2; a pre-origin use, fit_window_end 2021-12 <= the
  origin), like N3's pre-origin leveling-cell competitor. Every row is an observed survey
  value (flag 0, U1); the datum-artefact screen (survey-to-survey rate > 50 cm/yr,
  twin.leveling) is applied downstream per interval in ``sl_intervals``, as recorded in the
  MANIFEST.
* ``freeze``    the page model's frozen forward run (model output, sha cf2c7154: geometry and
  the baseline subsidence), the 2021 township atlas (geometry), the SL product (opened under
  both ``statistical_layer_history`` and ``baseline``; rows read under ``baseline``, the
  stricter: pre-origin, gate, flags 0-1) and the pre-origin WRA summaries'
  county_area_by_year (use design_time_evidence, which that version allows) for the L2
  zero-area tolerance, a test-design input only. No post-2022 observation and no township
  summary after the origin is read.
* ``verify``    rebuilds into a sibling directory and requires bit-identical files.

Outputs: cells.csv (township and county cell membership), sl_benchmarks.csv, predictions.csv
(ZR for the nominal survey intervals, SL, Z), tolerance.json, summary.json (hashes), and
verify.json after ``verify``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_n3s_l as S  # noqa: E402

TOWNS = "data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json"
PANEL_DATA_DIR = "chou-shui-data/data"      # hydrophysics.twin.leveling.load_panel's data_dir
SL_PRODUCT = "leveling_statlayer_history/preorigin"
PREORIGIN = "results/dataprep/subsidence_public_reports/preorigin/v1"
FREEZE_OUT = "results/dataprep/reports/N3S-L/freeze"
SL_PANEL_START, SL_PANEL_END = "2007-01-01", "2021-12-31"
SL_LAST_END = "2021-09-30"        # SL's intervals end by 2021-09
SL_N = 3                          # last 3 annual intervals
SL_MIN_N = 2
SL_RADIUS_KM = 3.0                # EB neighbourhood, and SL's cell field radius
SL_DT = (0.6, 1.5)                # years between consecutive surveys of one interval
SL_MAX_RATE = 50.0                # cm/yr: faster is a datum artefact (twin.leveling)
TOL_YEARS = (2012, 2022)          # pre-origin survey years for the zero-area tolerance
SL_USES = ("statistical_layer_history", "baseline")   # SL's two roles (L1 arm, L2 comparator)
SL_ALLOWED = [*SL_USES, "after_the_fact_analysis"]
SL_FORBIDDEN = ["scoring_target", "datum_fit", "prospective_scorer", "calibration_target",
                "initial_conditions", "idw_field", "prior", "prior_width_input", "model_design",
                "selection", "forcing_frozen", "forcing_recorded_arm", "validator",
                "design_time_evidence"]
SL_FLAG_NOTE = ("every row is an observed leveling survey value as held in the panel: flag 0 "
                "(U1). The datum-artefact screen (|survey-to-survey rate| > 50 cm/yr, "
                "twin.leveling) is applied downstream per interval by "
                "freeze_n3s_l.sl_intervals (SL_MAX_RATE), together with the 0.6-1.5 yr "
                "interval rule; no row is flagged here")
FREEZE_FILES = ("cells.csv", "sl_benchmarks.csv", "predictions.csv", "tolerance.json")


# ----------------------------------------------------------------------------- forward run
class ForwardRun:
    """A forward run's baseline subsidence (A, T) in cm, its months and cell centroids
    (score_x5nh.Run's definitions, with an explicit path)."""

    def __init__(self, s, months, mask, x0, y0, dx):
        self.s = np.asarray(s, float)
        self.months = pd.PeriodIndex(months, freq="M")
        ii, jj = np.nonzero(mask)
        self.x0, self.y0, self.dx = float(x0), float(y0), float(dx)
        self.cx = self.x0 + (jj + 0.5) * self.dx
        self.cy = self.y0 + (ii + 0.5) * self.dx
        self.cell = -np.ones(np.shape(mask), dtype="int64")
        self.cell[ii, jj] = np.arange(len(ii))

    @classmethod
    def from_npz(cls, path) -> ForwardRun:
        z = np.load(path, allow_pickle=True)
        names = [str(s) for s in z["scenario_names"]]
        s = z["subs_mean"][names.index("baseline")] * 100.0
        months = pd.DatetimeIndex(pd.to_datetime([str(d) for d in z["dates"]])).to_period("M")
        return cls(s, months, z["mask"], z["x0"], z["y0"], z["dx"])

    def m(self, ym) -> int:
        p = pd.Period(ym, freq="M")
        if p not in self.months:
            raise S.EngineeringFailure(f"month {p} is outside the run")
        return int(self.months.get_loc(p))

    def rate(self, a, b) -> np.ndarray:
        """(A,) cm/yr between months a and b (+ = sinking)."""
        ia, ib = self.m(a), self.m(b)
        return (self.s[:, ib] - self.s[:, ia]) * 12.0 / (ib - ia)

    def cell_of(self, x: float, y: float) -> int:
        j, i = int((x - self.x0) // self.dx), int((y - self.y0) // self.dx)
        if not (0 <= i < self.cell.shape[0] and 0 <= j < self.cell.shape[1]):
            return -1
        return int(self.cell[i, j])


# ----------------------------------------------------------------------------- geometry
def town_polygons(towns_json, counties=S.COUNTIES) -> dict:
    """{(county, township): [ring (N, 2) in TWD97 metres]} from the 2021 atlas topojson."""
    from pyproj import Transformer

    t = json.loads(Path(towns_json).read_text(encoding="utf-8"))
    sc, tr = t["transform"]["scale"], t["transform"]["translate"]
    arcs = [np.cumsum(np.asarray(a, float), axis=0) * sc + tr for a in t["arcs"]]

    def ring(ids):
        pts = []
        for k in ids:
            a = arcs[k] if k >= 0 else arcs[~k][::-1]
            pts.extend(a if not pts else a[1:])
        return np.asarray(pts)

    T = Transformer.from_crs(4326, 3826, always_xy=True)
    out: dict = {}
    for g in next(iter(t["objects"].values()))["geometries"]:
        p = g["properties"]
        cty = next((c for c in counties if c == (p.get("COUNTYNAME") or "")), None)
        if cty is None:
            continue
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        rings = []
        for poly in polys:
            r = ring(poly[0])
            X, Y = T.transform(r[:, 0], r[:, 1])
            rings.append(np.column_stack([X, Y]))
        out[(cty, p.get("TOWNNAME"))] = rings
    missing = [t for t in S.COAST if (S.COAST_COUNTY[t], t) not in out]
    if missing:
        raise SystemExit(f"coastal townships not found in the atlas: {missing}")
    return out


def assign(polys: dict, x, y) -> list:
    """(county, township) of each point (None outside both counties)."""
    from matplotlib.path import Path as MPath

    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    out = [None] * len(xy)
    for key, rings in polys.items():
        inside = np.zeros(len(xy), dtype=bool)
        for r in rings:
            inside |= MPath(r).contains_points(xy)
        for i in np.flatnonzero(inside):
            out[i] = key if out[i] is None else out[i]
    return out


def cells_table(run, polys: dict) -> pd.DataFrame:
    """Every active cell whose centroid lies in a township of the two counties."""
    keys = assign(polys, run.cx, run.cy)
    rows = [{"county": k[0], "township": k[1], "cell": i, "cx": run.cx[i], "cy": run.cy[i],
             "distal": bool(run.cx[i] < S.DISTAL_KM * 1000.0)}
            for i, k in enumerate(keys) if k is not None]
    return pd.DataFrame(rows).sort_values(["county", "township", "cell"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- SL mean3_eb
def sl_intervals(panel: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-survey intervals per benchmark (cm/yr, + = sinking), screened, ending by
    SL_LAST_END."""
    p = panel.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values(["sid", "datetime"])
    out = []
    end_max = pd.Timestamp(SL_LAST_END)
    for sid, g in p.groupby("sid", sort=True):
        t = g["datetime"].to_numpy()
        z = g["elev_m"].to_numpy(float)
        for i in range(1, len(g)):
            if pd.Timestamp(t[i]) > end_max:
                break
            dty = (t[i] - t[i - 1]) / np.timedelta64(1, "D") / 365.25
            if not SL_DT[0] <= dty <= SL_DT[1]:
                continue
            r = (z[i - 1] - z[i]) / dty * 100.0
            if not np.isfinite(r) or abs(r) > SL_MAX_RATE:
                continue
            out.append({"sid": str(sid), "end": pd.Timestamp(t[i]), "rate": float(r)})
    return pd.DataFrame(out, columns=["sid", "end", "rate"])


def sl_mean3_eb(panel: pd.DataFrame) -> pd.DataFrame:
    """mean3_eb: per benchmark the mean of its last SL_N intervals (>= SL_MIN_N), shrunk
    toward the mean of the other benchmarks within SL_RADIUS_KM by an empirical-Bayes weight
    B = (s2w/n) / (s2w/n + tau2), with s2w the pooled within-benchmark variance and tau2 =
    max(0, mean((rbar - m)^2) - mean(s2w/n)) over benchmarks with neighbours. No neighbour
    -> no shrinkage. Decay 0 (persistence), no head terms."""
    iv = sl_intervals(panel)
    xy = panel.groupby("sid")[["x", "y"]].first()
    xy.index = xy.index.astype(str)
    recs, ss, df = [], 0.0, 0
    for sid, g in iv.groupby("sid", sort=True):
        r = g.sort_values("end")["rate"].to_numpy()[-SL_N:]
        if len(r) < SL_MIN_N:
            continue
        recs.append({"sid": sid, "x": float(xy.loc[sid, "x"]), "y": float(xy.loc[sid, "y"]),
                     "n": len(r), "mean_rate": float(r.mean())})
        ss += float(((r - r.mean()) ** 2).sum())
        df += len(r) - 1
    b = pd.DataFrame(recs, columns=["sid", "x", "y", "n", "mean_rate"])
    if b.empty:
        return b.assign(prior_mean=[], shrink_B=[], sl_rate=[])
    s2w = ss / df if df > 0 else 0.0
    X, Y = b["x"].to_numpy(), b["y"].to_numpy()
    d = np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :])
    nb = (d <= SL_RADIUS_KM * 1000.0) & ~np.eye(len(b), dtype=bool)
    rbar = b["mean_rate"].to_numpy()
    has = nb.any(axis=1)
    m = np.where(has, (nb * rbar[None, :]).sum(1) / np.maximum(nb.sum(1), 1), rbar)
    v = s2w / b["n"].to_numpy()
    tau2 = max(0.0, float(np.mean((rbar[has] - m[has]) ** 2) - np.mean(v[has]))) \
        if has.any() else 0.0
    den = v + tau2
    B = np.where(has & (den > 0), v / np.where(den > 0, den, 1.0), 0.0)
    b["prior_mean"] = m
    b["shrink_B"] = B
    b["sl_rate"] = m + (1.0 - B) * (rbar - m)
    b.attrs.update({"s2_within": s2w, "tau2": tau2})
    return b


def sl_town_peaks(sl_cell: np.ndarray, cells: pd.DataFrame) -> dict:
    """SL's township peak, aggregated like ZR's: the max of SL's cell field over the
    township's cells (NaN cells ignored; no cell with a value -> NaN, no prediction)."""
    out = {}
    for (cty, t), g in cells.groupby(["county", "township"]):
        v = sl_cell[g["cell"].to_numpy(int)]
        v = v[np.isfinite(v)]
        out[(cty, t)] = float(v.max()) if len(v) else np.nan
    return out


def sl_cell_rates(sl: pd.DataFrame, cx, cy) -> np.ndarray:
    """SL's cell field: the nearest benchmark's SL rate within SL_RADIUS_KM, else NaN."""
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    out = np.full(len(cx), np.nan)
    if sl.empty:
        return out
    X, Y, R = sl["x"].to_numpy(), sl["y"].to_numpy(), sl["sl_rate"].to_numpy()
    for i in range(len(cx)):
        d = np.hypot(X - cx[i], Y - cy[i])
        k = int(np.argmin(d))
        if d[k] <= SL_RADIUS_KM * 1000.0:
            out[i] = R[k]
    return out


# ----------------------------------------------------------------------------- tolerance
def area_tolerance(area: pd.DataFrame) -> dict:
    """Per county: the median |change| of WRA's printed significant-subsidence area between
    consecutive pre-origin survey years (TOL_YEARS), latest edition per year (km2)."""
    out = {}
    a = area.copy()
    a["year"] = a["year_label"].astype(str).str.extract(r"^\s*(\d+)")[0].astype(float) + 1911
    a = a[(a["year"] >= TOL_YEARS[0]) & (a["year"] <= TOL_YEARS[1])]
    if "edition_roc" in a:
        a = a.sort_values("edition_roc")
    for cty in S.COUNTIES:
        g = a[a["county"].astype(str).str.contains(S.SHORT[cty][0])]
        g = g.groupby("year")["area_km2"].last().dropna().sort_index()
        yrs, v = g.index.to_numpy(), g.to_numpy(float)
        dif = [abs(v[i] - v[i - 1]) for i in range(1, len(v)) if yrs[i] - yrs[i - 1] == 1]
        out[cty] = float(np.median(dif)) if dif else None
    return out


def read_preorigin_area(preorigin_dir) -> pd.DataFrame:
    from common import guard
    prod = guard.open_product(preorigin_dir, "design_time_evidence")
    return prod.read_table("county_area_by_year.csv")


def read_sl_panel(sl_product) -> pd.DataFrame:
    """The SL history, opened under both of SL's uses (each must be granted); the rows are
    read under ``baseline`` (pre-origin, gate, flags 0-1) and must equal the
    statistical_layer_history read."""
    from common import guard
    prods = {u: guard.open_product(sl_product, u) for u in SL_USES}
    a = prods["baseline"].read_table("panel.parquet")
    b = prods["statistical_layer_history"].read_table("panel.parquet")
    if len(a) != len(b):
        raise S.EngineeringFailure("the SL panel differs between its two uses (flags)")
    return a


# ----------------------------------------------------------------------------- predictions
def predictions(run, cells: pd.DataFrame, sl: pd.DataFrame) -> pd.DataFrame:
    """ZR (nominal intervals), SL and Z, per survey year: coastal township peak (and, report
    only, mean and distal-cell peak) and county area > 3 cm/yr (km2)."""
    rows = []
    sl_cell = sl_cell_rates(sl, run.cx, run.cy)
    sl_peak = sl_town_peaks(sl_cell, cells)
    cell_km2 = (run.dx / 1000.0) ** 2
    bm = sl.copy()
    for roc in S.SURVEYS:
        s, e = S.NOMINAL[roc]
        q = S.zr_quantities(run, cells, s, e)
        for t in S.COAST:
            cty = S.COAST_COUNTY[t]
            v = q["town"].get((cty, t), {})
            b = bm[(bm["county"] == cty) & (bm["township"] == t)] if len(bm) else bm
            slb = float(b["sl_rate"].max()) if len(b) else np.nan
            for arm, qty, val in (("ZR", "peak_cm_yr", v.get("peak", np.nan)),
                                  ("ZR", "mean_cm_yr", v.get("mean", np.nan)),
                                  ("ZR", "distal_peak_cm_yr", v.get("distal_peak", np.nan)),
                                  ("SL", "peak_cm_yr", sl_peak.get((cty, t), np.nan)),
                                  ("SL", "benchmark_peak_cm_yr", slb),
                                  ("Z", "peak_cm_yr", 0.0)):
                rows.append({"arm": arm, "survey_roc": roc, "level": "township",
                             "county": cty, "township": t, "quantity": qty, "value": val,
                             "start_ym": s if arm == "ZR" else "", "end_ym": e if arm == "ZR"
                             else ""})
        for cty in S.COUNTIES:
            c = np.unique(cells.loc[cells["county"] == cty, "cell"].to_numpy(int))
            sl_area = float((np.nan_to_num(sl_cell[c], nan=-np.inf) > S.AREA_RATE).sum()
                            * cell_km2)
            for arm, val in (("ZR", q["area"].get(cty, np.nan)), ("SL", sl_area), ("Z", 0.0)):
                rows.append({"arm": arm, "survey_roc": roc, "level": "county", "county": cty,
                             "township": "", "quantity": "area_gt3_km2", "value": val,
                             "start_ym": s if arm == "ZR" else "",
                             "end_ym": e if arm == "ZR" else ""})
    return pd.DataFrame(rows)


def freeze(out, run, polys: dict, panel: pd.DataFrame, pre_area: pd.DataFrame,
           pier_xy=None, provenance: dict | None = None) -> dict:
    """Write the frozen files into ``out`` (must not exist) and return the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cells = cells_table(run, polys)
    sl = sl_mean3_eb(panel)
    keys = assign(polys, sl["x"], sl["y"]) if len(sl) else []
    sl["county"] = [k[0] if k else "" for k in keys]
    sl["township"] = [k[1] if k else "" for k in keys]
    pred = predictions(run, cells, sl)
    tol = area_tolerance(pre_area)
    cells.to_csv(out / "cells.csv", index=False)
    sl.to_csv(out / "sl_benchmarks.csv", index=False)
    pred.to_csv(out / "predictions.csv", index=False)
    (out / "tolerance.json").write_text(json.dumps(
        {"area_tolerance_km2": tol, "rule": "median |change| of the printed county "
         f"significant-subsidence area between consecutive survey years {TOL_YEARS[0]}-"
         f"{TOL_YEARS[1]} (preorigin v1, design_time_evidence)"}, ensure_ascii=False,
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    files = list(FREEZE_FILES)
    if pier_xy is not None:
        shutil.copyfile(pier_xy, out / "pier_xy.csv")
        files.append("pier_xy.csv")
    summ = {"id": S.ID, "files": {n: S.sha256(out / n) for n in files},
            "area_tolerance_km2": tol, "nominal_intervals": {str(k): v for k, v in
                                                             S.NOMINAL.items()},
            "sl": {"n_benchmarks": int(len(sl)), "params": {
                "last_end": SL_LAST_END, "n": SL_N, "min_n": SL_MIN_N,
                "radius_km": SL_RADIUS_KM, "dt_years": list(SL_DT),
                "max_rate_cm_yr": SL_MAX_RATE, "decay": 0, "head_terms": None,
                "township_peak": "max of SL's cell field over the township's cells (as ZR); "
                                 "the benchmark max is report-only",
                "uses": list(SL_USES)}},
            "n_cells": int(len(cells)), "provenance": provenance or {}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1,
                                                 sort_keys=True, default=float) + "\n",
                                      encoding="utf-8")
    return summ


def verify(out, rebuild) -> dict:
    """Rebuild into ``<out>.verify`` (via ``rebuild(path)``) and compare every file."""
    out = Path(out)
    tmp = out.with_name(out.name + ".verify")
    if tmp.exists():
        shutil.rmtree(tmp)
    rebuild(tmp)
    names = sorted(json.loads((out / "summary.json").read_text())["files"]) + ["summary.json"]
    probs = [n for n in names if S.sha256(out / n) != S.sha256(tmp / n)]
    rec = {"identical": not probs, "problems": probs,
           "files": {n: S.sha256(out / n) for n in names if n.endswith(".csv")},
           "summary_sha256": S.sha256(out / "summary.json")}
    (out / "verify.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(tmp)
    return rec


# ----------------------------------------------------------------------------- SL product
def build_sl_panel(root=None, panel: pd.DataFrame | None = None, src=None) -> Path:
    """The SL source product from the pre-origin leveling panel (rows 2007-01..2021-12).
    ``panel``/``src`` are for tests (a synthetic panel and a stand-in input file)."""
    from common import manifest
    from common.paths import DATAPREP

    if panel is None:
        from hydrophysics.twin.leveling import PANEL, load_panel
        src = REPO / PANEL_DATA_DIR / PANEL
        panel = load_panel(str(REPO / PANEL_DATA_DIR))
    p = panel.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p[(p["datetime"] >= SL_PANEL_START) & (p["datetime"] <= SL_PANEL_END)].copy()
    p["sid"] = p["sid"].astype(str)
    p["flag"] = 0
    out = manifest.new_version_dir(SL_PRODUCT, root=root or DATAPREP)
    p.to_parquet(out / "panel.parquet", index=False)
    v = {"schema": "n3sl-sl-panel/1", "rows": int(len(p)), "benchmarks": int(p["sid"].nunique()),
         "first": str(p["datetime"].min()), "last": str(p["datetime"].max())}
    gate = bool(len(p)) and v["last"] <= SL_PANEL_END + " 23:59:59" and \
        v["first"] >= SL_PANEL_START
    v["gate_passed"] = gate
    (out / "validation.json").write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    m = manifest.build_manifest(
        out, product=SL_PRODUCT, layer="L1", inputs=[src],
        code=[Path(__file__), REPO / "hydrophysics/twin/leveling.py"],
        parameters={"window": [SL_PANEL_START, SL_PANEL_END], "columns":
                    ["sid", "datetime", "elev_m", "x", "y", "flag"], "flags": SL_FLAG_NOTE,
                    "uses_note": "statistical_layer_history: SL's history (the L1 arm); "
                                 "baseline: SL as the L2 comparator ZR must beat (pre-origin, "
                                 "fit_window_end 2021-12)"},
        fit_window_end="2021-12", validation_report="validation.json",
        gate={"passed": gate, "criteria": "rows exist and every row is dated within "
              f"{SL_PANEL_START}..{SL_PANEL_END}; fixed before the build"},
        allowed_uses=SL_ALLOWED, forbidden_uses=SL_FORBIDDEN,
        personal_data="none",
        extra={"wp": S.WP, "purpose": "N3S-L SL arm (mean3_eb) history; the column's "
               "calibration leveling, pre-origin, so not independent of the column"})
    manifest.write_manifest(out, m)
    return out


def _real_freeze(out, sl_product, pier_xy):
    run = ForwardRun.from_npz(REPO / S.FORWARD)
    polys = town_polygons(REPO / TOWNS)
    panel = read_sl_panel(sl_product)
    pre = read_preorigin_area(REPO / PREORIGIN)
    prov = {"forward": {"path": S.FORWARD, "sha256": S.sha256(REPO / S.FORWARD)},
            "towns": {"path": TOWNS, "sha256": S.sha256(REPO / TOWNS)},
            "sl_product_manifest": {"path": str(Path(sl_product).resolve().relative_to(REPO)),
                                    "sha256": S.sha256(Path(sl_product) / "MANIFEST.json")},
            "preorigin_manifest": {"path": PREORIGIN,
                                   "sha256": S.sha256(REPO / PREORIGIN / "MANIFEST.json")}}
    if prov["forward"]["sha256"] != S.MODEL_SHA:
        raise SystemExit("forward npz sha differs from the page model's")
    return freeze(out, run, polys, panel, pre, pier_xy, prov)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sl-panel")
    for name in ("freeze", "verify"):
        f = sub.add_parser(name)
        f.add_argument("--sl-product", required=True)
        f.add_argument("--out", default=str(REPO / FREEZE_OUT))
        f.add_argument("--pier-xy", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "sl-panel":
        print(build_sl_panel())
    elif a.cmd == "freeze":
        s = _real_freeze(a.out, a.sl_product, a.pier_xy)
        print(json.dumps({"files": s["files"], "n_cells": s["n_cells"],
                          "sl_benchmarks": s["sl"]["n_benchmarks"]}, indent=1))
    else:
        print(json.dumps(verify(a.out, lambda d: _real_freeze(d, a.sl_product, a.pier_xy)),
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
