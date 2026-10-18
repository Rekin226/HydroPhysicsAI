"""WP P4m repair round: the computations declared in ``P4m.repair_audit.v1``.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_repair audit    # RA, RB, RC, RG (no engine)
    tools/cpu_slot.sh python -m p4_changhua.p4m_repair engine   # RE, RF (frozen twin, CPU)

Writes ``results/dataprep/reports/P4m_evidence/stage/repair/`` (audit.json, engine.json,
e9_*.csv). Reads P4m preorigin/v1 outputs (never edits them), H1a v3 pre-origin rows, C0
preorigin/v3 and the rain files; every look goes to the A1 look log with members.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
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

PRE = "P4m.repair_audit.v1"
V1 = DATAPREP / "changhua_forcing_diagnostics" / "preorigin" / "v1"
OUT = L.STAGE / "repair"
RF_CWA = REPO / "chou-shui-data/data/rf_timeseries.csv"
RF_VENDOR = REPO / "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
RF_ST = REPO / "chou-shui-data/data/rf_stations.csv"
PRIMARY_R = 3.0
RADII_KM = (2.0, 3.0, 5.0)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _engine_wells() -> dict:
    with np.load(L.ENGINE_V2 / "engine_arrays.npz", allow_pickle=False) as z:
        arr = {k: z[k] for k in ("sids", "obs_idx", "obs_layer", "well_county")}
    with np.load(V1 / "well_arrays.npz", allow_pickle=False) as z:
        if not np.array_equal(z["sids"], arr["sids"]):
            raise SystemExit("v1 well arrays and engine wells differ")
        arr["well_xy"] = np.asarray(z["well_xy"], dtype=float)
        arr["P_A1"] = z["P_A1"]
    arr["sids"] = np.array([str(s) for s in arr["sids"]])
    return arr


# ============================================================================ RA: BE
def ra_be(arr: dict, created: str) -> dict:
    p = guard.open_product(L.H1A, "model_design", window_end="2022-12")
    t = p.read_table("barometric_efficiency.csv")
    t["sid"] = t["sid"].astype(str).str.zfill(8)
    info = pd.DataFrame({"sid": arr["sids"], "county": arr["well_county"],
                         "layer": arr["obs_layer"]})
    t = t[(t.accepted.astype(bool)) & (t.fit_window == "2012-01..2022-12")]
    m = t.merge(info, on="sid")
    looks_p4m.record(["p4m.repair.BE"], ids=sorted(m.sid.unique()), window="2012-01..2022-12",
                     description="H1a v3 barometric efficiency of accepted pre-origin epochs at "
                                 "calibration wells (validity audit)",
                     purpose="design_evidence", prereg_id=PRE, tag=f"[p4m_repair:RA:{created}]",
                     note="RA BE sign / validity audit")
    strata = {}
    for (c, lay), g in m.groupby(["county", "layer"]):
        v = g.BE_long.to_numpy(float)
        med = float(np.median(v))
        strata[f"{c}_layer{lay}"] = {
            "n": int(v.size), "n_le_0": int((v <= 0).sum()), "n_ge_1": int((v >= 1).sum()),
            "median": med, "valid": bool(0 < med < 1 and v.size >= 3)}
    ch1, yl1 = strata.get(f"{L.CH}_layer1"), strata.get(f"{L.YL}_layer1")
    out = {"n_epochs": int(len(m)), "n_wells": int(m.sid.nunique()), "strata": strata,
           "all_epochs": {"n": int(len(m)), "n_le_0": int((m.BE_long <= 0).sum()),
                          "n_ge_1": int((m.BE_long >= 1).sum())}}
    pump_valid = bool(ch1 and yl1 and ch1["valid"] and yl1["valid"])
    out["pump_layer_valid"] = pump_valid
    if pump_valid:
        ratio = yl1["median"] / ch1["median"]
        out["reading"] = {"basis": "pump layer", "implied_Ss_CH_over_YL": ratio,
                          "valid": True, "points_same_way": bool(ratio > 1.25)}
    else:
        ok = m[(m.BE_long > 0) & (m.BE_long < 1)]
        a, b = ok[ok.county == L.CH].BE_long, ok[ok.county == L.YL].BE_long
        fb = {"basis": "fallback: epochs with 0 < BE < 1 pooled over layers",
              "n_CH": int(len(a)), "n_YL": int(len(b))}
        if len(a) >= 5 and len(b) >= 5:
            r = float(b.median() / a.median())
            fb.update({"implied_Ss_CH_over_YL": r, "valid": True,
                       "points_same_way": bool(r > 1.25)})
        else:
            fb.update({"valid": False, "points_same_way": None})
        out["reading"] = fb
    ok1 = m[(m.layer == 1) & (m.BE_long > 0) & (m.BE_long < 1)]
    out["info_pump_layer_valid_epochs_only"] = {
        "n_CH": int((ok1.county == L.CH).sum()), "n_YL": int((ok1.county == L.YL).sum()),
        "median_CH": float(ok1[ok1.county == L.CH].BE_long.median())
        if (ok1.county == L.CH).any() else None,
        "median_YL": float(ok1[ok1.county == L.YL].BE_long.median())
        if (ok1.county == L.YL).any() else None}
    out["v1_reading"] = "implied Ss ratio CH/YL in the pump layer 1.37 = (-0.496) / (-0.362)"
    return out


# ============================================================================ RC: E8
def _cluster_ids(x_km, y_km) -> np.ndarray:
    cl = np.floor(x_km / L.CLUSTER_KM) * 100000 + np.floor(y_km / L.CLUSTER_KM)
    return pd.factorize(np.asarray(cl))[0]


def rc_e8(created: str) -> dict:
    df = pd.read_csv(V1 / "e8_tides_preorigin.csv", dtype={"sid": str})
    looks_p4m.record(["p4m.repair.E8"], ids=sorted(df.sid), window="2012-01..2022-12",
                     description="P4m v1 E8 table (M2 amplitudes from L0 hourly heads to "
                                 "2022-12) of the 158 calibration wells",
                     purpose="design_evidence", prereg_id=PRE, tag=f"[p4m_repair:RC:{created}]",
                     note="RC E8 robustness (no easting, pump-layer raw ratio)")
    co = df[(df.cohM2 > 0.8) & (df.n_months >= 24)].copy().reset_index(drop=True)
    co["is_ch"] = (co.county == L.CH).astype(float)
    y = np.log(co.A_M2_mm.to_numpy())
    lays = sorted(co.layer.unique())[1:]

    def design(with_e: bool, d=co):
        X = [np.ones(len(d))] + ([d.x_km.to_numpy()] if with_e else []) + [d.is_ch.to_numpy()]
        X += [(d.layer == lay).astype(float).to_numpy() for lay in lays]
        return np.column_stack(X), (2 if with_e else 1)

    cl = _cluster_ids(co.x_km.to_numpy(), co.y_km.to_numpy())
    uc = np.unique(cl)
    out = {"n_coherent": int(len(co))}
    for name, with_e in (("a_declared_with_easting", True), ("b_without_easting", False)):
        X, k = design(with_e)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rng = np.random.default_rng(L.SEED)
        bs = []
        for _ in range(L.N_BOOT):
            pick = rng.choice(uc, uc.size, replace=True)
            idx = np.concatenate([np.flatnonzero(cl == c) for c in pick])
            if np.unique(X[idx, k]).size < 2:
                continue
            bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][k])
        out[name] = {"ratio": float(math.exp(beta[k])),
                     "p05": float(math.exp(np.percentile(bs, 5))),
                     "p95": float(math.exp(np.percentile(bs, 95))), "n_boot": len(bs)}
    # (c) pump-layer raw median ratio
    c1 = co[co.layer == 1].reset_index(drop=True)
    cl1 = _cluster_ids(c1.x_km.to_numpy(), c1.y_km.to_numpy())
    uc1 = np.unique(cl1)

    def med_ratio(d):
        a, b = d[d.county == L.CH].A_M2_mm, d[d.county == L.YL].A_M2_mm
        return float(a.median() / b.median()) if len(a) and len(b) else np.nan

    rng = np.random.default_rng(L.SEED)
    bs = []
    for _ in range(L.N_BOOT):
        pick = rng.choice(uc1, uc1.size, replace=True)
        idx = np.concatenate([np.flatnonzero(cl1 == c) for c in pick])
        v = med_ratio(c1.iloc[idx])
        if np.isfinite(v):
            bs.append(v)
    out["c_pump_layer_raw_median_ratio"] = {
        "ratio": med_ratio(c1), "n_CH": int((c1.county == L.CH).sum()),
        "n_YL": int((c1.county == L.YL).sum()), "p05": float(np.percentile(bs, 5)),
        "p95": float(np.percentile(bs, 95)), "n_boot": len(bs)}
    parts = [out["a_declared_with_easting"], out["b_without_easting"],
             out["c_pump_layer_raw_median_ratio"]]
    out["robust"] = bool(all(p["ratio"] > 1 and p["p05"] > 1 for p in parts))
    out["direction"] = ("a larger M2 strain response means a smaller specific storage (plan "
                        "E8): a stronger, faster pumping response, the opposite of the damping "
                        "s_theta < 0.67 calls for")
    out["valid"] = out["robust"]
    out["points_same_way"] = False
    out["reading"] = ("not supporting: " + ("robust, " if out["robust"] else "not robust, ")
                      + "and its physical direction is opposite to damping")
    return out


# ============================================================================ RB / RG: E9
def _hp(v: np.ndarray, w: int = 25) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.rolling(w, center=True, min_periods=12).mean()).to_numpy()


def _fit(y, e, reg, p):
    """The v1 E9 regression (p4m_evidence._fit), unchanged."""
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


def _rain_table(path: Path, months) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = pd.read_csv(path, parse_dates=["date time"], encoding="utf-8-sig").set_index(
        "date time")
    rfs = pd.read_csv(RF_ST, encoding="utf-8-sig")
    return rf.resample("MS").sum(min_count=20).reindex(months), rfs


def e9_variant(arr, H, Ecell, cent, regional, months, *, definition: str, rain: Path,
               meta: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    sids, xy = arr["sids"], arr["well_xy"]
    oi, ol, wc = arr["obs_idx"], arr["obs_layer"], arr["well_county"]
    if definition == "v1_rounded_100m_model_layers_0_1":
        site = pd.factorize(np.array([f"{round(x / 100)}_{round(y / 100)}" for x, y in xy]))[0]
        lay = ol.astype(int)
        pair, pump = {0, 1}, 1
    else:
        mm = meta.set_index("sid").reindex(sids)
        site = pd.factorize(mm.site_id.astype(str).to_numpy())[0]
        lay = pd.to_numeric(mm.layer_code, errors="coerce").fillna(-1).astype(int).to_numpy()
        pair, pump = {1, 2}, 2
    nested = []
    for s in np.unique(site):
        w = np.flatnonzero(site == s)
        if pair <= set(lay[w].tolist()):
            nested.extend(w[np.isin(lay[w], list(pair))].tolist())
    nested = np.array(sorted(nested))
    Pm, rfs = _rain_table(rain, months)

    def rain_at(x, y):
        dd = (rfs["TM_X97"] - x) ** 2 + (rfs["TM_Y97"] - y) ** 2
        for rid in rfs.iloc[np.argsort(dd.to_numpy())]["rf_id"]:
            if rid in Pm and Pm[rid].notna().mean() > 0.9:
                return Pm[rid].interpolate().bfill().ffill().to_numpy()
        return Pm.mean(axis=1).to_numpy()

    P_A1 = arr["P_A1"]
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
                         "layer_def": int(lay[w]), "model_layer": int(ol[w]), "r_km": r_km,
                         "pump_layer": bool(lay[w] == pump),
                         "E_local_mean_GWh": float(e.mean()),
                         "b_obs": fo[0] if fo else np.nan, "t_obs": fo[1] if fo else np.nan,
                         "n_obs": fo[2] if fo else 0,
                         "b_mod": fm[0] if fm else np.nan, "t_mod": fm[1] if fm else np.nan})
    df = pd.DataFrame(rows)
    summ = {"definition": definition, "rain": rain.name,
            "n_nested_wells": int(len(nested)), "n_sites": int(len(np.unique(site[nested]))),
            "by_radius": {str(r): {k: _e9_counts(df[(df.r_km == r) & df.pump_layer
                                                     & (df.county == c)])
                                   for k, c in (("changhua", L.CH), ("yunlin", L.YL))}
                          for r in RADII_KM}}
    return df, summ


def _e9_counts(d: pd.DataFrame) -> dict:
    d = d[np.isfinite(d.b_obs)]
    n = int(len(d))
    if n == 0:
        return {"n": 0, "valid": False, "points_same_way": None}
    mo, mm = float(d.b_obs.median()), float(d.b_mod.median())
    n_neg = int((d.b_obs < 0).sum())
    valid = bool(mo < 0 and mm < 0 and n_neg >= 2 * n / 3)
    ratio = mo / mm if mm else float("nan")
    return {"n": n, "n_obs_negative": n_neg, "n_obs_positive": int((d.b_obs > 0).sum()),
            "n_t_below_minus2": int((d.t_obs < -2).sum()),
            "n_t_above_plus2": int((d.t_obs > 2).sum()),
            "median_b_obs": mo, "median_b_mod": mm,
            "n_mod_negative": int((d.b_mod < 0).sum()),
            "signed_ratio_obs_over_mod": ratio,
            "v1_abs_ratio": abs(mo) / abs(mm) if mm else float("nan"),
            "valid": valid,
            "points_same_way": bool(valid and 0 < ratio < 0.67) if valid else None}


def rb_rg_e9(arr: dict, created: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS
    from hydrophysics.twin.grid import build_grid

    grid = build_grid(DEFAULT_PATHS["polygon"], dx=1000.0)
    cent = grid.centroids()
    pm = guard.open_product(L.H1A, "model_design")
    meta = pm.read_table("station_meta_preorigin.parquet",
                         columns=["sid", "site_id", "layer_code", "layer_flag"])
    meta["sid"] = meta["sid"].astype(str)
    p = guard.open_product(L.H1A, "model_design", window_end="2022-12")
    hm = p.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag"])
    hm = hm[hm.flag.isin([0, 1])]
    hm["sid"] = hm["sid"].astype(str)
    months = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    H = hm.pivot_table(index="sid", columns="date", values="value").reindex(
        index=list(arr["sids"]), columns=months).to_numpy(float)
    pc = L.c0_open("preorigin", "model_design")
    with np.load(pc.path("cells_total.npz"), allow_pickle=False) as z:
        d = [str(x) for x in z["dates"]]
        i0, i1 = d.index("2012-01"), d.index("2022-12") + 1
        Ecell = z["value_all"][:, i0:i1] / 1e6
    regional = Ecell.sum(axis=0)
    looks_p4m.record(["p4m.repair.E9"], ids=list(arr["sids"]), window="2012-01..2022-12",
                     description="H1a v3 monthly heads (flags 0-1) of the calibration wells at "
                                 "nested sites (v1 and as-declared site definitions)",
                     purpose="design_evidence", prereg_id=PRE, tag=f"[p4m_repair:RG:{created}]",
                     note="RB/RG E9 sign audit, site_id definition, vendor-rain sensitivity")
    out, frames = {}, {}
    for definition in ("v1_rounded_100m_model_layers_0_1", "declared_site_id_codes_1_2"):
        for rain in (RF_CWA, RF_VENDOR):
            key = f"{definition}|{'cwa' if rain == RF_CWA else 'vendor'}"
            df, summ = e9_variant(arr, H, Ecell, cent, regional, months,
                                  definition=definition, rain=rain, meta=meta)
            out[key] = summ
            frames[key] = df
    # replication of v1 (v1 definition, installed rain)
    v1 = pd.read_csv(V1 / "e9_nested_slopes.csv", dtype={"sid": str})
    rep = frames["v1_rounded_100m_model_layers_0_1|cwa"]
    mrg = v1.merge(rep, on=["sid", "r_km"], suffixes=("_v1", "_rep"))
    out["replication_v1"] = {
        "n_rows_v1": int(len(v1)), "n_rows_rep": int(len(rep)), "n_merged": int(len(mrg)),
        "max_abs_diff_b_obs": float(np.nanmax(np.abs(mrg.b_obs_v1 - mrg.b_obs_rep))),
        "max_abs_diff_b_mod": float(np.nanmax(np.abs(mrg.b_mod_v1 - mrg.b_mod_rep)))}
    # layer-definition agreement for the 158 wells
    mm = meta.set_index("sid").reindex(list(arr["sids"]))
    code = pd.to_numeric(mm.layer_code, errors="coerce")
    out["layer_code_vs_model_layer"] = {
        "n_wells": int(len(code)),
        "n_code_minus_1_equals_model_layer": int((code - 1 == arr["obs_layer"]).sum())}
    out["primary"] = out["declared_site_id_codes_1_2|cwa"]["by_radius"][str(PRIMARY_R)]
    out["rain_note"] = ("chou-shui-data/data/rf_timeseries.csv is the CWA CODIS rebuild "
                        "installed 2026-10-11 (sha256 42308c8d...), the file P4m v1 read; the "
                        "vendor cache is rf_timeseries_vendor_2012_2022.csv")
    return out, frames


def audit() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    created = _now()
    arr = _engine_wells()
    R = {"wp": "P4m", "prereg": PRE, "created": created}
    R["RA_BE"] = ra_be(arr, created)
    R["RC_E8"] = rc_e8(created)
    R["RB_RG_E9"], frames = rb_rg_e9(arr, created)
    for k, df in frames.items():
        df.to_csv(OUT / ("e9_" + k.replace("|", "_") + ".csv"), index=False)
    refit = json.loads((V1 / "theta_refit_t2b.json").read_text())
    s_theta = refit["s_theta"]
    e9p = R["RB_RG_E9"]["primary"]["changhua"]
    e9v1 = R["RB_RG_E9"]["v1_rounded_100m_model_layers_0_1|cwa"]["by_radius"][
        str(PRIMARY_R)]["changhua"]
    ind = {"BE": R["RA_BE"]["reading"], "E8": {"valid": R["RC_E8"]["valid"],
                                               "points_same_way": False},
           "E9_declared": e9p, "E9_v1_definition": e9v1}
    support = [k for k, v in ind.items() if v.get("valid") and v.get("points_same_way")]
    R["RD_T5_audit"] = {
        "s_theta": s_theta, "s_theta_below_0.67": bool(s_theta < 0.67),
        "indicators": {k: {"valid": v.get("valid"), "points_same_way": v.get("points_same_way")}
                       for k, v in ind.items()},
        "valid_supporting_indicators": support,
        "multiplier_supported": bool(s_theta < 0.67 and support),
        "declared_T5_mechanical_result": "fired (v1, unchanged on record)",
        "label": "post-look audit reading (the 2023-25 recorded arm had been read)"}
    L.jdump(R, OUT / "audit.json")
    L.log(f"audit done: BE reading {R['RA_BE']['reading']}; E8 robust {R['RC_E8']['robust']}; "
          f"E9 primary CH {e9p}; support {support}")
    return 0


# ============================================================================ RE / RF (engine)
def _gls_blocks(X, r, wsel, phi, detrend=False):
    """Per-well AR(1)-whitened blocks with the whitened datum (and trend) projected out:
    {w: (A_w^T A_w, A_w^T y_w)} (the v1 gls_free_datum convention)."""
    out = {}
    for w in np.flatnonzero(wsel):
        m = np.isfinite(r[w])
        if m.sum() < L.MIN_N:
            continue
        t = np.flatnonzero(m)
        cols = [r[w, m], np.ones(t.size)] + ([t - t.mean()] if detrend else [])
        Z = np.column_stack(cols + [X[w, m]])
        Wz = np.empty_like(Z)
        Wz[0] = Z[0] * math.sqrt(1 - phi ** 2)
        for i in range(1, t.size):
            Wz[i] = Z[i] - phi * Z[i - 1] if t[i] == t[i - 1] + 1 else Z[i] * math.sqrt(
                1 - phi ** 2)
        nn = 2 + int(detrend)
        D = Wz[:, 1:nn]
        P = Wz - D @ np.linalg.lstsq(D, Wz, rcond=None)[0]
        A, y = P[:, nn:], P[:, 0]
        out[int(w)] = (A.T @ A, A.T @ y)
    return out


def _solve_blocks(blocks, wells):
    AtA = sum(blocks[w][0] for w in wells if w in blocks)
    Aty = sum(blocks[w][1] for w in wells if w in blocks)
    return np.linalg.solve(AtA, Aty)


def gls_boot(X, r, wsel, cl, phi, detrend=False):
    blocks = _gls_blocks(X, r, wsel, phi, detrend)
    est = _solve_blocks(blocks, np.flatnonzero(wsel))
    boots = []
    for idx in L.boot_indices(cl, wsel):
        try:
            boots.append(_solve_blocks(blocks, idx))
        except np.linalg.LinAlgError:
            continue
    boots = np.asarray(boots)
    res = {"phi": phi, "n_boot": int(len(boots)), "detrend": detrend}
    for k, nm in enumerate(("mean", "seasonal", "anomaly")):
        res[nm] = {"theta": float(est[k]), "p05": float(np.percentile(boots[:, k], 5)),
                   "p95": float(np.percentile(boots[:, k], 95))}
    return res


def engine_part() -> int:
    from p4_changhua import p4m_run as RUN

    OUT.mkdir(parents=True, exist_ok=True)
    created = _now()
    R = {"wp": "P4m", "prereg": PRE, "created": created}
    S = L.setup()
    tw = S["tw"]
    tm = L.step_mask(S)
    obs_m = L.masked_obs(S, tm)
    elig = L.eligible(obs_m)
    G = L.groups(S)
    ch_el = G["CH"] & elig
    cl = L.clusters(S)
    steps = S["steps"]
    v1 = json.loads((V1 / "diagnostics_t1_t3.json").read_text())
    H0 = L.rollout(S)
    Ech, _, _ = L.changhua_forcing(S)
    H1 = L.rollout(S, E_extra=Ech)
    P = {"A0": L.at_wells(S, H0), "A1": L.at_wells(S, H1)}
    with np.load(V1 / "well_arrays.npz", allow_pickle=False) as z:
        rep_A1 = float(np.nanmax(np.abs(z["P_A1"] - P["A1"])))
    R["replication_P_A1_max_abs_m"] = rep_A1
    sids_all = [str(s) for s in S["sids"]]
    looks_p4m.record(["p4m.repair.insample"], ids=sids_all, window="2012-01..2022-12",
                     description="calibration heads of the 158 wells (the cwa theta's fit "
                                 "data), RE GLS intervals and RF scale variants",
                     purpose="design_evidence", prereg_id=PRE,
                     tag=f"[p4m_repair:engine:{created}]",
                     note="RE AR(1)-GLS bootstrap of the T3 mean channel; RF s_AM "
                          "identifiability (LAG+2, LAG-2, PERM, NOSEAS)")
    # ------------------------------------------------------------- RE
    t0 = time.perf_counter()
    mo = np.array([int(s[5:7]) for s in steps])
    mean_c = Ech.mean(axis=1, keepdims=True)
    clim = np.stack([Ech[:, mo == k].mean(axis=1) for k in range(1, 13)], axis=1)
    seas = clim[:, mo - 1] - mean_c
    anom = Ech - mean_c - seas
    parts = np.stack([np.broadcast_to(mean_c, Ech.shape), seas, anom], axis=-1)
    Es0 = tw.spread(S["E"])
    X0 = L.tlm_response(S, H0, Es0, parts)
    r0 = np.where(tm, S["obs"] - P["A0"], np.nan)
    tm2 = L.step_mask(S, "2014-01", L.SCORE_LAST)
    r0_v2 = np.where(tm2, S["obs"] - P["A0"], np.nan)
    el2 = L.eligible(np.where(tm2, S["obs"], np.nan))
    phi = RUN.lag1_phi(r0, elig)
    # V4 spin-up (as p4m_run.e10)
    Rfull = S["inp"].recharge_field.numpy()
    m_all = np.array([int(m[5:7]) for m in S["E_months"]])
    Rclim = np.stack([Rfull[:, m_all == k].mean(axis=1) for k in range(1, 13)], axis=1)
    sp_months = [str(d)[:7] for d in pd.date_range("2007-02-01", "2012-01-01", freq="MS")]
    sp_mo = np.array([int(m[5:7]) for m in sp_months])
    E_sp_yl, mths, _ = L.c0_cells("preorigin", "model_design", counties=("yunlin",),
                                  first="2007-02", last="2012-01",
                                  grid_hash_expected=S["grid_hash"])
    Hs0 = L.rollout(S, E=E_sp_yl, R=Rclim[:, sp_mo - 1])
    H0s = L.rollout(S, h0=Hs0[..., -1])
    P0s = L.at_wells(S, H0s)
    X0s = L.tlm_response(S, H0s, Es0, parts)
    r0s = np.where(tm, S["obs"] - P0s, np.nan)
    phi4 = RUN.lag1_phi(r0s, elig)
    cases = {"V1_12m_about_A0": (X0, r0, elig, False, phi),
             "V2_24m_about_A0": (X0, r0_v2, el2, False, phi),
             "V3_12m_trend_about_A0": (X0, r0, elig, True, phi),
             "V4_spinup_12m_about_A0s": (X0s, r0s, elig, False, phi4)}
    RE = {"phi_V1": phi, "phi_V4": phi4, "variants": {}}
    for name, (X, r, ws, detr, ph) in cases.items():
        ols = RUN.channel_fit(X, r, ws, cl, detrend=detr)
        g = gls_boot(X, r, ws, cl, ph, detrend=detr)
        rec = {"ols": ols, "gls": g}
        vv = v1["T3"]["channels"][name]
        rec["replication"] = {"ols_mean_theta_abs_diff": abs(ols["mean"]["theta"]
                                                             - vv["mean"]["theta"]),
                              "ols_mean_p05_abs_diff": abs(ols["mean"]["p05"] - vv["mean"]["p05"]),
                              "ols_mean_p95_abs_diff": abs(ols["mean"]["p95"] - vv["mean"]["p95"])}
        if "gls_ar1" in vv:
            g_v1fun = RUN.gls_free_datum(X, r, ws, ph)
            rec["replication"]["gls_mean_v1_value_abs_diff_v1fun"] = abs(
                float(g_v1fun[0]) - vv["gls_ar1"]["mean"])
            rec["replication"]["gls_blocks_vs_v1fun_abs_diff"] = float(
                np.max(np.abs(np.array([g[k]["theta"] for k in ("mean", "seasonal", "anomaly")])
                              - g_v1fun)))
        rec["gls_mean_interval_below_0"] = bool(g["mean"]["p95"] < 0)
        RE["variants"][name] = rec
    allneg = all(RE["variants"][v]["gls_mean_interval_below_0"] for v in RE["variants"])
    RE["all_variants_gls_below_0"] = allneg
    RE["reading"] = ("fragile: OLS primary covers 0 in V1, V2, V4; the GLS secondary excludes 0 "
                     "in every variant; log the level channel for plan 7.3 as a possible "
                     "misspecification (declared primary verdict unchanged)" if allneg else
                     "robust to the GLS secondary in: " + ", ".join(
                         v for v, x in RE["variants"].items()
                         if not x["gls_mean_interval_below_0"]))
    RE["t_s"] = round(time.perf_counter() - t0, 1)
    R["RE_gls"] = RE
    L.log(f"RE done in {RE['t_s']}s: {RE['reading']}")
    # ------------------------------------------------------------- RF
    t0 = time.perf_counter()
    grid = list(L.S_GRID)
    variants = {"METERED": Ech, "LAG+2": np.roll(Ech, 2, axis=1),
                "LAG-2": np.roll(Ech, -2, axis=1)}
    pos = np.flatnonzero(Ech.mean(axis=1) > 0)
    perm = np.random.default_rng(L.SEED).permutation(pos)
    Ep = Ech.copy()
    Ep[pos] = Ech[perm]
    variants["PERM"] = Ep
    variants["NOSEAS"] = Ech - seas
    RF = {"n_cells_positive": int(pos.size),
          "nonseas_negative_cell_months": int((variants["NOSEAS"] < 0).sum()),
          "variants": {}}
    for name, Ev in variants.items():
        res, _curves, cache = RUN.scale_family(S, obs_m, Ev, grid, elig, ch_el, f"RF {name}")
        p1 = cache[round(1.0, 6)]
        gm = L.group_metrics(L.per_well_stats(obs_m, p1), ch_el)
        gy = L.group_metrics(L.per_well_stats(obs_m, p1), G["YL"] & elig)
        RF["variants"][name] = {
            "s_LS_all": res["s_LS_all"], "s_LS_changhua": res["s_LS_changhua"],
            "s_AM_changhua": res["s_AM_changhua"], "s_AM_how": res["s_AM_how"],
            "amp_ratio_changhua_grid": res["amp_ratio_changhua"],
            "changhua_at_s1": gm, "yunlin_at_s1": gy,
            "total_kwh": float(Ev.sum())}
    met = RF["variants"]["METERED"]
    t2 = v1["T2"]["metered"]
    RF["replication"] = {"s_LS_all_abs_diff": abs(met["s_LS_all"] - t2["s_LS_all"]),
                         "s_AM_abs_diff": abs(met["s_AM_changhua"] - t2["s_AM_changhua"])}
    c_met = met["changhua_at_s1"]["corr_median"]
    inf, within = {}, []
    for name in ("LAG+2", "LAG-2", "PERM"):
        v = RF["variants"][name]
        drop = c_met - v["changhua_at_s1"]["corr_median"]
        ratio = (v["s_AM_changhua"] / met["s_AM_changhua"]
                 if v["s_AM_changhua"] is not None else None)
        informative = bool(drop >= 0.02)
        inside = bool(ratio is not None and 0.8 <= ratio <= 1.25)
        inf[name] = {"corr_drop": drop, "s_AM_ratio_to_metered": ratio,
                     "informative": informative, "s_AM_within_0.8_1.25": inside}
        if informative:
            within.append(inside)
    RF["identifiability"] = inf
    RF["n_informative"] = len(within)
    if not within:
        RF["reading"] = "no informative variant: the check is inconclusive"
    elif all(within):
        RF["reading"] = ("not identified: every informative timing/location variant keeps s_AM "
                         "within a factor 0.8-1.25 of the metered value; 'the metered magnitude "
                         "fits' is withdrawn ('consistent with, not identified')")
    else:
        RF["reading"] = ("partly informative: at least one informative variant moves s_AM "
                         "outside 0.8-1.25; the statement is weakened to 'consistent with'")
    RF["t_s"] = round(time.perf_counter() - t0, 1)
    R["RF_identifiability"] = RF
    L.log(f"RF done in {RF['t_s']}s: {RF['reading']}")
    L.jdump(R, OUT / "engine.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", choices=("audit", "engine"))
    a = ap.parse_args(argv)
    return audit() if a.part == "audit" else engine_part()


if __name__ == "__main__":
    raise SystemExit(main())
