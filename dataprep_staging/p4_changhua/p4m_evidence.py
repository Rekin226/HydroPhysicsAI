"""WP P4m T2d: independent pre-origin evidence on Changhua's response (E8-style tides,
barometric efficiency, E9-style nested-site layer ratios), as declared in
``P4m.metered_diagnostics.v1``, with one deviation (see below).

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_evidence     # after p4m_run (needs its P_A1)

* **E8 (rebuilt, deviation from the declaration).** The declaration named the head-signature
  pilot's ``tide_coherence.csv`` as design-time evidence, but that pilot's manifest allows only
  ``after_the_fact_analysis`` (its hourly features run to 2026-09). The M2 statistics are
  therefore rebuilt from the L0 hourly records 2012-01..2022-12 of the 158 calibration wells
  with H1a's own tide code (``h1_heads_qc.tides.o1_phases``, a copy of the pilot's;
  ``rawio.load_well(pre_only=True)``), and the pilot table is only compared after the fact.
* **BE.** H1a v3 ``barometric_efficiency.csv`` (guard use model_design, window 2022-12),
  accepted epochs, joined to the 158 calibration wells' county and model layer (H1a's
  ``station_meta_preorigin`` has no time column, so the guard refuses it under a window).
* **E9.** Nested sites among the 158 calibration wells (wells within 100 m of each other) with
  model layers 0 and 1; H1a v3 ``heads_monthly`` flags 0-1, 2012-01..2022-12; local metered
  energy from C0 preorigin/v3 ``cells_total`` ``value_all`` (all counties) within r of the
  well's cell; the same regression on the A1 model heads of ``p4m_run``.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime as _dt  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.paths import DATAPREP, REPO  # noqa: E402

from p4_changhua import looks_p4m  # noqa: E402
from p4_changhua import p4m_lib as L  # noqa: E402

OUT = L.STAGE / "evidence"
RUN = L.STAGE / "run"
PILOT_TIDES = DATAPREP / "_pilots" / "head_pumping_signatures"
RADII_KM = (2.0, 3.0, 5.0)
PRIMARY_R = 3.0


# ============================================================================ E8
def e8(arr: dict, created: str) -> dict:
    from h1_heads_qc import rawio, tides

    sids = [str(s) for s in arr["sids"]]
    looks_p4m.record(["p4m.E8.tides"], ids=sids, window="2012-01..2022-12",
                     description="hourly L0 head records (AMP_V2 historical cache, pre-origin "
                                 "part only) of the 158 calibration wells, monthly M2/O1 "
                                 "harmonics",
                     purpose="design_evidence", prereg_id=L.PRE_DIAG,
                     tag=f"[p4m_evidence:E8:{created}]",
                     note="E8-style tidal evidence rebuilt pre-origin (deviation: pilot table "
                          "is after-the-fact only)")
    months = pd.period_range("2012-01", "2022-12", freq="M")
    rows = []
    t0 = time.perf_counter()
    for i, sid in enumerate(sids):
        sh, sn = rawio.load_well(sid, keep_sentinels=False, pre_only=True)
        if sh.empty:
            continue
        if sh.index.max() >= pd.Timestamp("2023-01-01"):
            raise SystemExit("post-origin hourly data reached the E8 rebuild")
        hh = sh.resample("h").mean()
        ph = tides.o1_phases(hh, months)
        if ph.empty:
            continue
        rows.append({"sid": sid, "county": str(arr["well_county"][i]),
                     "layer": int(arr["obs_layer"][i]),
                     "x_km": float(arr["well_xy"][i, 0]) / 1000.0,
                     "y_km": float(arr["well_xy"][i, 1]) / 1000.0,
                     "n_months": int(len(ph)), "cohM2": tides.coherence(ph["gM2"].to_numpy()),
                     "cohO1": tides.coherence(ph["gO1"].to_numpy()),
                     "A_M2_mm": 1000.0 * float(ph["A_M2"].median()),
                     "A_O1_mm": 1000.0 * float(ph["A_O1"].median())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e8_tides_preorigin.csv", index=False)
    L.log(f"E8: {len(df)} wells in {time.perf_counter() - t0:.0f}s")
    out = {"n_wells": int(len(df)), "coherence_threshold": 0.8}
    co = df[(df.cohM2 > 0.8) & (df.n_months >= 24)].copy()
    out["n_coherent"] = int(len(co))
    out["median_A_M2_mm"] = {
        f"{c}_layer{lay}": {"n": int(len(g)), "median": float(g.A_M2_mm.median())}
        for (c, lay), g in co.groupby(["county", "layer"])}
    if len(co) >= 8 and co.county.nunique() == 2:
        co["is_ch"] = (co.county == L.CH).astype(float)
        X = [np.ones(len(co)), co.x_km.to_numpy(), co.is_ch.to_numpy()]
        for lay in sorted(co.layer.unique())[1:]:
            X.append((co.layer == lay).astype(float).to_numpy())
        X = np.column_stack(X)
        y = np.log(co.A_M2_mm.to_numpy())
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        cl = (np.floor(co.x_km / L.CLUSTER_KM) * 100000 + np.floor(co.y_km / L.CLUSTER_KM))
        cl = pd.factorize(cl)[0]
        rng = np.random.default_rng(L.SEED)
        uc = np.unique(cl)
        bs = []
        for _ in range(L.N_BOOT):
            pick = rng.choice(uc, uc.size, replace=True)
            idx = np.concatenate([np.flatnonzero(cl == c) for c in pick])
            if np.unique(X[idx, 2]).size < 2:
                continue
            bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][2])
        out["ols_log_AM2"] = {
            "columns": ["const", "easting_km", "changhua", "layer dummies"],
            "easting_coef_per_km": float(beta[1]), "changhua_coef": float(beta[2]),
            "changhua_over_yunlin_ratio": float(math.exp(beta[2])),
            "ratio_p05_p95": [float(math.exp(np.percentile(bs, 5))),
                              float(math.exp(np.percentile(bs, 95)))],
            "label": "cluster bootstrap (10 km), UNCALIBRATED; context only (plan E8)"}
    # after the fact: agreement with the pilot table (2012-01..2026-09 hourly)
    try:
        pp = guard.open_product(PILOT_TIDES, "after_the_fact_analysis")
        pt = pd.read_csv(pp.path("out/tide_coherence.csv"), dtype={"sid": str})
        m = df.merge(pt[["sid", "AM2_mm", "cohM2"]], on="sid", suffixes=("", "_pilot"))
        out["after_the_fact_vs_pilot"] = {
            "n": int(len(m)),
            "corr_log_AM2": float(np.corrcoef(np.log(m.A_M2_mm), np.log(m.AM2_mm))[0, 1]),
            "median_ratio_AM2": float(np.median(m.A_M2_mm / m.AM2_mm)),
            "corr_cohM2": float(np.corrcoef(m.cohM2, m.cohM2_pilot)[0, 1]),
            "use": "after_the_fact_analysis (pilot window to 2026-09); not design evidence"}
    except Exception as exc:          # pragma: no cover - reported, not fatal
        out["after_the_fact_vs_pilot"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ============================================================================ BE
def be(arr: dict, created: str) -> dict:
    p = guard.open_product(L.H1A, "model_design", window_end="2022-12")
    t = p.read_table("barometric_efficiency.csv")
    t["sid"] = t["sid"].astype(str).str.zfill(8)
    sids = [str(s) for s in arr["sids"]]
    info = pd.DataFrame({"sid": sids, "county": arr["well_county"], "layer": arr["obs_layer"]})
    t = t[(t.accepted.astype(bool)) & (t.fit_window == "2012-01..2022-12")]
    m = t.merge(info, on="sid")
    looks_p4m.record(["p4m.BE"], ids=sorted(m.sid.unique()), window="2012-01..2022-12",
                     description="H1a v3 barometric efficiency of accepted pre-origin epochs at "
                                 "calibration wells",
                     purpose="design_evidence", prereg_id=L.PRE_DIAG,
                     tag=f"[p4m_evidence:BE:{created}]", note="BE by county and layer")
    g = m.groupby(["county", "layer"])["BE_long"]
    out = {"n_epochs": int(len(m)), "n_wells": int(m.sid.nunique()),
           "median_BE_long": {f"{c}_layer{lay}": {"n": int(v.size), "median": float(v.median())}
                              for (c, lay), v in g}}
    ratios = {}
    for lay in range(4):
        a = m[(m.county == L.CH) & (m.layer == lay)].BE_long
        b = m[(m.county == L.YL) & (m.layer == lay)].BE_long
        if len(a) >= 2 and len(b) >= 2:
            ratios[f"layer{lay}"] = {"BE_CH_over_YL": float(a.median() / b.median()),
                                     "implied_Ss_CH_over_YL": float(b.median() / a.median()),
                                     "n_CH": int(len(a)), "n_YL": int(len(b))}
    out["ratios"] = ratios
    out["all_layers"] = {"median_CH": float(m[m.county == L.CH].BE_long.median())
                         if (m.county == L.CH).any() else None,
                         "median_YL": float(m[m.county == L.YL].BE_long.median())
                         if (m.county == L.YL).any() else None}
    out["note"] = ("BE = n beta / (alpha + n beta); at equal porosity Ss is proportional to "
                   "1/BE; context only (few accepted epochs, plan H1 notes A_K sign issues)")
    return out


# ============================================================================ E9
def _hp(v: np.ndarray, w: int = 25) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.rolling(w, center=True, min_periods=12).mean()).to_numpy()


def _fit(y, e, reg, p):
    yy, ee, rr = _hp(y), _hp(e), _hp(reg)
    pp = [_hp(p), _hp(np.roll(p, 1)), _hp(np.roll(p, 2))]
    Z = np.column_stack([np.ones(len(yy)), ee, rr] + pp)
    ok = np.isfinite(yy) & np.isfinite(Z).all(1)
    ok[:2] = False
    if ok.sum() < 48 or np.nanstd(ee[ok]) == 0:
        return None
    b, *_ = np.linalg.lstsq(Z[ok], yy[ok], rcond=None)
    res = yy[ok] - Z[ok] @ b
    s2 = res @ res / (ok.sum() - Z.shape[1])
    cov = s2 * np.linalg.pinv(Z[ok].T @ Z[ok])
    return float(b[1]), float(b[1] / math.sqrt(cov[1, 1])), int(ok.sum())


def e9(arr: dict, run: dict, S_grid, created: str) -> dict:
    sids = [str(s) for s in arr["sids"]]
    xy = arr["well_xy"]
    oi, ol, wc = arr["obs_idx"], arr["obs_layer"], arr["well_county"]
    site = pd.factorize([f"{round(x / 100)}_{round(y / 100)}" for x, y in xy])[0]
    nested = []
    for s in np.unique(site):
        w = np.flatnonzero(site == s)
        if {0, 1} <= set(ol[w].tolist()):
            nested.extend(w.tolist())
    nested = np.array(sorted(nested))
    looks_p4m.record(["p4m.E9.nested"], ids=[sids[i] for i in nested],
                     window="2012-01..2022-12",
                     description="H1a v3 monthly heads (flags 0-1) of the calibration wells at "
                                 "nested sites with model layers 0 and 1",
                     purpose="design_evidence", prereg_id=L.PRE_DIAG,
                     tag=f"[p4m_evidence:E9:{created}]", note="E9-style layer ratios")
    p = guard.open_product(L.H1A, "model_design", window_end="2022-12")
    hm = p.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag"])
    hm = hm[hm.flag.isin([0, 1])]
    hm["sid"] = hm["sid"].astype(str)
    months = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    H = hm.pivot_table(index="sid", columns="date", values="value").reindex(
        index=sids, columns=months).to_numpy(float)
    # energy: all counties, C0 cells_total value_all
    pc = L.c0_open("preorigin", "model_design")
    with np.load(pc.path("cells_total.npz"), allow_pickle=False) as z:
        d = [str(x) for x in z["dates"]]
        i0, i1 = d.index("2012-01"), d.index("2022-12") + 1
        Ecell = z["value_all"][:, i0:i1] / 1e6                       # GWh/month
    cent = S_grid.centroids()
    regional = Ecell.sum(axis=0)
    rf = pd.read_csv(REPO / "chou-shui-data/data/rf_timeseries.csv",
                     parse_dates=["date time"]).set_index("date time")
    rfs = pd.read_csv(REPO / "chou-shui-data/data/rf_stations.csv", encoding="utf-8-sig")
    Pm = rf.resample("MS").sum(min_count=20).reindex(months)

    def rain_at(x, y):
        dd = (rfs["TM_X97"] - x) ** 2 + (rfs["TM_Y97"] - y) ** 2
        for rid in rfs.iloc[np.argsort(dd.to_numpy())]["rf_id"]:
            if rid in Pm and Pm[rid].notna().mean() > 0.9:
                return Pm[rid].interpolate().bfill().ffill().to_numpy()
        return Pm.mean(axis=1).to_numpy()

    P_A1 = run["P_A1"]                                            # (W, 131) 2012-02..
    model = np.concatenate([np.full((P_A1.shape[0], 1), np.nan), P_A1], axis=1)
    rows = []
    for r_km in RADII_KM:
        for w in nested:
            c = cent[oi[w]]
            near = ((cent - c) ** 2).sum(axis=1) <= (r_km * 1000.0) ** 2
            e = Ecell[near].sum(axis=0)
            pr = rain_at(xy[w, 0], xy[w, 1])
            fo = _fit(H[w], e, regional, pr)
            fm = _fit(model[w], e, regional, pr)
            rows.append({"sid": sids[w], "site": int(site[w]), "county": str(wc[w]),
                         "layer": int(ol[w]), "r_km": r_km, "E_local_mean_GWh": float(e.mean()),
                         "b_obs": fo[0] if fo else np.nan, "t_obs": fo[1] if fo else np.nan,
                         "n_obs": fo[2] if fo else 0,
                         "b_mod": fm[0] if fm else np.nan, "t_mod": fm[1] if fm else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e9_nested_slopes.csv", index=False)
    out = {"n_nested_wells": int(len(nested)), "n_sites": int(len(np.unique(site[nested]))),
           "regression": "high-passed head on high-passed local energy (GWh/month), "
                         "high-passed regional (fan) energy and rain lags 0-2; >= 48 months",
           "radii": {}}
    for r_km in RADII_KM:
        d = df[df.r_km == r_km]
        res = {}
        for cty, key in ((L.CH, "changhua"), (L.YL, "yunlin")):
            dd = d[d.county == cty]
            po = dd.pivot_table(index="site", columns="layer", values="b_obs")
            pm = dd.pivot_table(index="site", columns="layer", values="b_mod")
            blk = {"slopes_median_m_per_GWh": {
                f"layer{lay}": {"obs": float(dd[dd.layer == lay].b_obs.median()),
                                "model_A1": float(dd[dd.layer == lay].b_mod.median()),
                                "n": int(dd[dd.layer == lay].b_obs.notna().sum())}
                for lay in sorted(dd.layer.unique())}}
            if 0 in po and 1 in po:
                both = po[[0, 1]].dropna()
                bm = pm.reindex(both.index)[[0, 1]].dropna() if (0 in pm and 1 in pm) else None
                blk["ratio_layer0_over_layer1"] = {
                    "n_sites": int(len(both)),
                    "obs_median": float(np.median(both[0] / both[1])) if len(both) else None,
                    "model_A1_median": (float(np.median(bm[0] / bm[1]))
                                        if bm is not None and len(bm) else None),
                    "obs_sites_with_negative_layer1_slope": int((both[1] < 0).sum())}
            res[key] = blk
        out["radii"][str(r_km)] = res
    out["primary_r_km"] = PRIMARY_R
    out["model_parameters"] = {"pump_frac_shallow": 0.0868, "return_frac": 0.417,
                               "note": "the cwa theta's split: 8.7 % of the abstraction from "
                                       "layer 0, 41.7 % returned to layer 0"}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    created = _dt.datetime.now(_dt.timezone.utc).isoformat()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "dataprep_staging"))
    with np.load(L.ENGINE_V2 / "engine_arrays.npz", allow_pickle=False) as z:
        arr = {k: z[k] for k in ("sids", "obs_idx", "obs_layer", "well_county")}
    with np.load(RUN / "arrays.npz", allow_pickle=False) as z:
        run = {"P_A1": z["P_A1"], "sids": z["sids"], "well_xy": z["well_xy"]}
    if not np.array_equal(run["sids"], arr["sids"]):
        raise SystemExit("run arrays and engine wells differ")
    from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS
    from hydrophysics.twin.grid import build_grid

    grid = build_grid(DEFAULT_PATHS["polygon"], dx=1000.0)
    arr["well_xy"] = np.asarray(run["well_xy"], dtype=float)
    R = {"wp": "P4m", "prereg": L.PRE_DIAG, "created": created,
         "well_xy_note": "well coordinates of the twin inputs (TWD97); a site = wells within "
                         "the same 100 m rounded position"}
    R["E8"] = e8(arr, created)
    R["BE"] = be(arr, created)
    R["E9"] = e9(arr, run, grid, created)
    L.jdump(R, OUT / "results.json")
    L.log(f"evidence done -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
