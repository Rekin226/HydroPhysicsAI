"""Inventory of the 344 fan stations: why each excluded well is excluded, its coverage under
every definition the plan names, and the H3 candidates (plan WP H3 'Gap', 'Inclusion on
observed months only', 'Reconcile counts').

Inputs (all pre-origin reads, use ``selection``):

* ``heads/L1_clean/v3`` through the loader guard (``window_end`` 2022-12; monthly table and
  ``station_meta_preorigin.parquet``: repaired coordinates, raw and proposed layer codes);
* ``validation_units/v3/wells.csv`` (the 158 model wells);
* the twin's L0 cache ``AMP_V2/data/wells/<sid>.parquet`` and ``fan_stations.parquet``
  (read-only) to reproduce the twin's own QC rule (``heads.build_head_field`` defaults);
* the model grid (``results/twin_forward/datum_gate_innov.npz``).

Coverage definitions over the 132 months 2012-01..2022-12:

* ``twin_hour``: the twin's default rule, despiked samples / hours (10-minute records,
  common since about 2019, inflate it up to six-fold) with the longest gap from the first to
  the last sample <= 180 days (the twin's inclusion needs >= 0.80 and that gap rule);
* ``raw_month``: share of months with at least one raw sample (the twin's opt-in
  ``--strict-coverage`` statistic);
* ``l1_any``: share of months with an L1 flag-0/1 row (at least one clean day);
* ``l1_c20``: share of months that are L1 C20 months with flag 0/1 (>= 20 clean days): the
  settled definition (``config.COVERAGE_DEFINITION``).

No head value is compared with a prediction here: this is metadata and record extent only.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from h3_recovered_wells import config as C

sys.path[:0] = [str(C.REPO / "dataprep_staging"), str(C.REPO)]
from common import guard  # noqa: E402

COUNTY = {"彰化縣": "Changhua", "雲林縣": "Yunlin"}
META_COLS = ["sid", "site_id", "CountyName", "TownName", "x_twd97", "y_twd97", "xy_flag",
             "xy_method", "LocationByTWD97", "layer_code_raw", "layer_code", "layer_flag",
             "layer_method", "WellDepth", "FinishDepth", "WellElevation", "GroundHeight",
             "EstablishDate"]


def grid():
    from hydrophysics.twin.grid import FanGrid
    z = np.load(C.GRID_NPZ, allow_pickle=False)
    return FanGrid(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]),
                   float(z["y0"]), z["mask"])


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


def run_stats(obs: np.ndarray) -> dict:
    """Lead, trail and internal-run statistics of a boolean observed-month vector."""
    T = len(obs)
    if not obs.any():
        return {"lead": T, "trail": T, "max_internal": 0, "n_internal_runs": 0,
                "first": None, "last": None}
    idx = np.where(obs)[0]
    lead, trail = int(idx[0]), int(T - 1 - idx[-1])
    inner = [b - a for a, b in runs_of(~obs) if a > idx[0] and b <= idx[-1]]
    return {"lead": lead, "trail": trail, "max_internal": int(max(inner, default=0)),
            "n_internal_runs": len(inner), "first": int(idx[0]), "last": int(idx[-1])}


def twin_rule(sid: str, stations: pd.DataFrame) -> dict:
    """The twin's default QC on the L0 cache (heads.build_head_field, read-only)."""
    from hydrophysics.twin.heads import _despike
    t0, t1 = pd.Timestamp(f"{C.START}-01"), pd.Timestamp("2023-01-01")
    n_hours = int((t1 - t0).total_seconds() // 3600)
    months = pd.period_range(t0, t1 - pd.Timedelta(seconds=1), freq="M")
    out = {"twin_hour": 0.0, "twin_max_gap_d": np.nan, "raw_month": 0.0, "n_raw": 0,
           "raw_first": None, "raw_last": None}
    p = C.AMP_WELLS / f"{sid}.parquet"
    if not p.exists():
        return out
    s = pd.read_parquet(p)["value"].dropna().sort_index()
    s = s[(s.index >= t0) & (s.index < t1)]
    out["n_raw"] = int(len(s))
    if s.empty:
        return out
    out["raw_first"], out["raw_last"] = str(s.index[0].date()), str(s.index[-1].date())
    out["raw_month"] = float(months.isin(s.index.to_period("M").unique()).mean())
    s = _despike(s)
    out["twin_hour"] = float(len(s) / max(n_hours, 1))
    hourly = s.resample("h").mean()
    miss = hourly.isna()
    if miss.any():
        r = (miss != miss.shift()).cumsum()[miss]
        out["twin_max_gap_d"] = float(r.value_counts().max()) / 24.0
    else:
        out["twin_max_gap_d"] = 0.0
    return out


def load_monthly() -> pd.DataFrame:
    hp = guard.open_product(C.L1_DIR, "selection", window_end=C.PRE_END)
    mon = hp.read_table("heads_monthly.parquet",
                        columns=["sid", "date", "month", "flag", "c20", "n_clean_days"])
    mon["sid"] = mon["sid"].astype(str)
    mon = mon[(mon.date >= pd.Timestamp(f"{C.START}-01"))
              & (mon.date <= pd.Timestamp(f"{C.PRE_END}-01"))]
    return mon


def load_meta() -> pd.DataFrame:
    mp = guard.open_product(C.L1_DIR, "selection")
    md = mp.read_table("station_meta_preorigin.parquet", columns=META_COLS)
    md["sid"] = md["sid"].astype(str)
    return md


def model_wells() -> set[str]:
    p = guard.open_product(C.UNITS_DIR, "selection")
    w = p.read_table("wells.csv")
    return set(w["sid"].astype(str).str.zfill(8))


def build() -> pd.DataFrame:
    from hydrophysics.twin.heads import _station_xy
    mon = load_monthly()
    md = load_meta()
    stations = pd.read_parquet(C.AMP_STATIONS, columns=["sid", "GroundwaterLayerCode",
                                                        "LocationByTWD97"])
    stations["sid"] = stations["sid"].astype(str)
    st = stations.set_index("sid")
    in_model = model_wells()
    g = grid()
    dates = pd.date_range(f"{C.START}-01", f"{C.PRE_END}-01", freq="MS")
    ti = {d: j for j, d in enumerate(dates)}
    rows = []
    for _, m in md.iterrows():
        sid = m.sid
        r = {"sid": sid, "site_id": m.site_id,
             "county": COUNTY.get(m.CountyName, "other"), "town": m.TownName,
             "layer_raw": str(m.layer_code_raw), "layer_code": str(m.layer_code),
             "layer_flag": int(m.layer_flag), "layer_method": m.layer_method,
             "well_depth": m.WellDepth, "finish_depth": m.FinishDepth,
             "x": m.x_twd97, "y": m.y_twd97, "xy_flag": int(m.xy_flag),
             "xy_method": m.xy_method, "in_model": sid in in_model}
        # grid membership: repaired coordinates (L1) and the twin's raw parse
        c = g.active_index(m.x_twd97, m.y_twd97) if np.isfinite(m.x_twd97) else None
        r["cell"] = -1 if c is None else int(c)
        r["on_grid"] = c is not None
        raw_xy = _station_xy(st.loc[sid]) if sid in st.index else None
        c_raw = g.active_index(*raw_xy) if raw_xy is not None else None
        r["on_grid_raw_xy"] = c_raw is not None
        r["twin_layer_ok"] = (str(st.loc[sid, "GroundwaterLayerCode"]).strip() in C.LAYERS
                              if sid in st.index else False)
        r["twin_xy_ok"] = raw_xy is not None
        # observed months (L1 v3, pre-origin)
        w = mon[mon.sid == sid]
        obs01 = w[w.flag.isin([0, 1])]
        any_v = np.zeros(len(dates), dtype=bool)
        c20_v = np.zeros(len(dates), dtype=bool)
        jj = obs01.date.map(ti).dropna().astype(int).to_numpy()
        any_v[jj] = True
        jc = obs01[obs01.c20.astype(bool)].date.map(ti).dropna().astype(int).to_numpy()
        c20_v[jc] = True
        r["n_c20"] = int(c20_v.sum())
        r["n_any"] = int(any_v.sum())
        r["l1_c20"] = float(c20_v.mean())
        r["l1_any"] = float(any_v.mean())
        rs = run_stats(c20_v)
        ra = run_stats(any_v)
        r.update({f"c20_{k}": v for k, v in rs.items()})
        r["record_end_trail"] = ra["trail"]
        r["record_first"] = None if ra["first"] is None else dates[ra["first"]].strftime("%Y-%m")
        r["record_last"] = None if ra["last"] is None else dates[ra["last"]].strftime("%Y-%m")
        r.update(twin_rule(sid, stations))
        rows.append(r)
    X = pd.DataFrame(rows)
    X["twin_pass_qc"] = (X.twin_layer_ok & X.twin_xy_ok & (X.n_raw > 0)
                         & (X.twin_hour >= C.TWIN_MIN_COV)
                         & ~(X.twin_max_gap_d > C.TWIN_MAX_GAP_DAYS))
    X["reason"] = X.apply(reason, axis=1)
    X["candidate"] = X.apply(is_candidate, axis=1)
    X["candidate_fail"] = X.apply(candidate_fail, axis=1)
    return X


def reason(r) -> str:
    """Why a station is not a model well (the twin's QC order), or 'model well'."""
    if r.in_model:
        return "model well"
    if r.n_raw == 0:
        return "no data 2012-22 in cache"
    if not r.twin_layer_ok:
        return "layer code missing"
    if not r.twin_xy_ok:
        return "coordinates invalid"
    if r.twin_hour < C.TWIN_MIN_COV:
        return "hourly coverage < 0.8"
    if r.twin_max_gap_d > C.TWIN_MAX_GAP_DAYS:
        return "single gap > 180 days"
    if not r.on_grid_raw_xy:
        return "passes QC, off grid"
    return "other"


def candidate_fail(r) -> str:
    """First H3 inclusion rule a non-model station fails ('' = candidate)."""
    if r.in_model:
        return "model well"
    if not r.on_grid:
        return "off grid"
    if r.layer_raw not in C.LAYERS:
        return "no layer code (proposal only)"
    if r.n_c20 == 0:
        return "no observed month"
    if r.l1_c20 < C.MIN_COVERAGE:
        return "observed coverage < 0.80"
    if r.n_c20 < C.MIN_OBS_MONTHS:
        return "fewer than 36 observed months"
    if r.c20_max_internal > C.MAX_INTERNAL_RUN:
        return "internal gap > 24 months"
    if r.c20_lead > C.MAX_LEAD:
        return "leading gap > 66 months"
    if r.record_end_trail > C.MAX_TRAIL:
        return "record ends before 2022-12"
    return ""


def is_candidate(r) -> bool:
    return candidate_fail(r) == ""


def counts(X: pd.DataFrame) -> dict:
    """The plan's counts and the reconciliation table."""
    ex = X[~X.in_model]
    og = ex[ex.on_grid]
    out = {"stations": int(len(X)), "model_wells": int(X.in_model.sum()),
           "excluded": int(len(ex)),
           "twin_pass_qc": int(X.twin_pass_qc.sum()),
           "twin_pass_qc_on_grid_raw_xy": int((X.twin_pass_qc & X.on_grid_raw_xy).sum()),
           "excluded_by_reason": ex.reason.value_counts().to_dict(),
           "excluded_on_grid": int(len(og)),
           "excluded_on_grid_raw_xy": int(ex.on_grid_raw_xy.sum()),
           "excluded_on_grid_with_data": int((og.n_raw > 0).sum()),
           "excluded_on_grid_with_data_by_county": og[og.n_raw > 0].county.value_counts()
           .to_dict(),
           "excluded_on_grid_with_data_layer_1_4": int(((og.n_raw > 0)
                                                        & og.layer_raw.isin(C.LAYERS)).sum()),
           "on_grid_changed_by_coordinate_repair": X[X.on_grid != X.on_grid_raw_xy]
           .sid.tolist()}
    rec = {}
    for name, col in (("brief_unknown", None), ("twin_hour", "twin_hour"),
                      ("raw_month", "raw_month"), ("l1_any", "l1_any"), ("l1_c20", "l1_c20")):
        if col is None:
            continue
        for scope, sub in (("excluded_all", ex), ("excluded_on_grid", og),
                           ("excluded_on_grid_raw_xy", ex[ex.on_grid_raw_xy])):
            k = sub[sub[col] >= C.COUNT_THRESHOLD].county.value_counts()
            rec[f"{name}>={C.COUNT_THRESHOLD}:{scope}"] = {c: int(k.get(c, 0)) for c in
                                                         ("Changhua", "Yunlin", "other")}
    out["reconcile_ge_0.30"] = rec
    lay = og[og.layer_raw.isin(C.LAYERS) & (og.n_raw > 0)]
    k80 = lay[lay.l1_c20 >= C.MIN_COVERAGE]
    out["on_grid_layer14_ge_0.80_l1_c20"] = {
        "n": int(len(k80)), "by_county": k80.county.value_counts().to_dict(),
        "with_record_end_before_2022_12": int((k80.record_end_trail > 0).sum())}
    cand = X[X.candidate]
    out["candidates"] = {"n": int(len(cand)), "by_county": cand.county.value_counts().to_dict(),
                         "by_layer": cand.layer_raw.value_counts().to_dict(),
                         "sids": cand.sid.tolist()}
    out["candidate_fail_counts_on_grid_excluded"] = og.candidate_fail.value_counts().to_dict()
    nolay = og[~og.layer_raw.isin(C.LAYERS) & (og.l1_c20 >= C.MIN_COVERAGE)
               & (og.record_end_trail <= C.MAX_TRAIL)]
    out["no_layer_full_coverage_on_grid"] = {"n": int(len(nolay)), "sids": nolay.sid.tolist()}
    ended = X[(X.record_last == "2018-12")]
    out["records_ending_2018_12"] = {
        "all": int(len(ended)), "on_grid": int(ended.on_grid.sum()),
        "on_grid_layer_1_4": int((ended.on_grid & ended.layer_raw.isin(C.LAYERS)).sum()),
        "sids_on_grid": ended[ended.on_grid].sid.tolist(),
        "sids_off_grid": ended[~ended.on_grid].sid.tolist()}
    return out
