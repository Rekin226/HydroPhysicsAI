"""P2f (plan WP P2f): local-only layer-response evidence at nested head sites, observed and
modelled (``cwa`` theta), with regional placebos. Declared in P2def.f.layer_response.v1 before
it ran; the units are design-time (Yunlin: the billing pilot's s07 sites; every well's
2012-2022 heads have been read by earlier WPs), so the results are design-time evidence for
the I1 rung-R4 acceptance, not an out-of-sample test.

Definitions:

* heads: heads/L1_clean/v3 monthly values with flags 0-1 (observed), sliced to 2022-12 by the
  guard, months 2012-01..2022-11; wells with layer code 1-4, valid coordinates and >= 60
  observed months; site = station site_id; county from the station polygon;
* local energy at r km: sum over kept pumps of every in-grid county (C0 preorigin v3, recorded
  arm; missing readings 0) within r of the well, GWh/month; r = 3 km primary, 2 and 5 km
  reported; the Chow-Lin arm (P2abc preorigin v2) reported;
* regional covariates (regressed out together with the rain, plan P2f "local-only slopes"):
  the in-grid total of the well's county and the energy of kept pumps (any county) more than
  15 km from the well; rain: high-passed CODIS rain of the nearest gauge with >= 90 % valid
  months, lags 0-2;
* slope b_w: OLS of the high-passed head (x minus its centred 25-month mean, >= 12 months) on
  the high-passed local energy with intercept, rain lags 0-2 and the two regional series
  (>= 48 usable months, non-constant energy);
* site ratio for layer L vs 2: median slope of the site's layer-L wells / median slope of its
  layer-2 wells; statistic = median over sites; 90 % interval = 5th-95th percentiles of 4000
  site-bootstrap medians (seed 2026);
* model side: the same regression on the model head of the well's cell and layer, primary
  ``results/twin_forward/datum_gate_cwa.npz`` heads_mean, baseline scenario (the cwa forward
  archive; ensemble mean over its members); secondary the cwa fit_full CPU hindcast of the
  physics pilot (hindcast_cwa.npz H); supplementary ``datum_gate_chc.npz`` (another session's
  run with Changhua census forcing; not the plan's theta). With cwa, Changhua has no modelled
  pumping, so its Changhua model ratio is reported as not meaningful;
* placebo: 20 year-shuffled copies (derangement of the complete years 2012-2021, seeds
  6000 + r, harness.placebos.year_shuffle) of each well's local energy; D_w = partial R2 of
  the real local energy minus the mean over the placebos; per county and layer (1, 2) the mean
  D_w and a site-cluster bootstrap one-sided p = (1 + #{boot mean <= 0}) / 4001 (4000
  resamples, seed 2026); local response 'identified' when p < 0.05. The L1/L2 ratio is labelled
  locally identified in a county only if both layers are identified there. The ratio the
  placebo regressors produce (median over placebos of the site-median ratio) is reported.
* depth proxy (plan P2f): share of 2012-2022 kept in-grid energy on meters over 10 HP per
  county; coverage of the census well-depth attribute.

Usage: python -I f1_layer_response.py <out_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import looks_def as LK  # noqa: E402
from common import guard  # noqa: E402
from harness import placebos  # noqa: E402

DECL_ID = "P2def.f.layer_response.v1"
HM = pd.date_range("2012-01-01", "2022-11-01", freq="MS")
NH = len(HM)
H0 = D.tidx("2012-01")
RADII = (2.0, 3.0, 5.0)
PRIMARY_R = 3.0
FAR_KM = 15.0
N_PLACEBO = 20
N_BOOT = 4000
SEED = 2026
MIN_MONTHS = 60


# ----------------------------------------------------------------------------- data
def station_meta() -> pd.DataFrame:
    """Candidate wells from the station metadata only (no head values): layer 1-4, valid
    coordinates, Yunlin or Changhua."""
    p_meta = guard.open_product(D.HEADS_L1, "validator")
    meta = p_meta.read_table("station_meta_preorigin.parquet",
                             columns=["sid", "site_id", "layer_code", "x_twd97", "y_twd97",
                                      "poly_county", "CountyName"])
    meta = meta.copy()
    meta["layer"] = pd.to_numeric(meta["layer_code"], errors="coerce")
    meta["county"] = meta["poly_county"].fillna(meta["CountyName"])
    return meta[meta["layer"].isin([1, 2, 3, 4]) & meta["x_twd97"].notna() &
                meta["y_twd97"].notna() &
                meta["county"].isin([D.COUNTY_ZH["yunlin"], D.COUNTY_ZH["changhua"]])]


def wells(meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    p = guard.open_product(D.HEADS_L1, "model_design", window_end="2022-12")
    h = p.read_table("heads_monthly.parquet", columns=["sid", "month", "value", "flag"])
    h = h[h["flag"].isin([0, 1]) & (h["month"] >= "2012-01") & (h["month"] <= "2022-11")]
    piv = h.pivot_table(index="sid", columns="month", values="value", aggfunc="first")
    piv = piv.reindex(columns=HM.strftime("%Y-%m"))
    meta = meta[meta["sid"].isin(piv.index)]
    H = piv.reindex(meta["sid"]).to_numpy(dtype=float)
    meta = meta.assign(n_months=np.isfinite(H).sum(1))
    keep = (meta["n_months"] >= MIN_MONTHS).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    return meta, H[keep]


def rain_for(w: pd.DataFrame) -> np.ndarray:
    rf = pd.read_csv(D.RAIN, parse_dates=["date time"]).set_index("date time")
    rf = rf[(rf.index >= HM[0]) & (rf.index < HM[-1] + pd.offsets.MonthBegin(1))]
    cnt = rf.resample("MS").count()
    tot = rf.resample("MS").sum(min_count=1)
    Pm = tot.where(cnt >= 20).reindex(HM)
    st = pd.read_csv(D.RAIN_ST, encoding="utf-8-sig")
    st = st[st["rf_id"].isin(Pm.columns)]
    good = [g for g in st["rf_id"] if Pm[g].notna().mean() >= 0.9]
    st = st[st["rf_id"].isin(good)].reset_index(drop=True)
    tree = cKDTree(st[["TM_X97", "TM_Y97"]].to_numpy())
    idx = tree.query(w[["x_twd97", "y_twd97"]].to_numpy(), k=1)[1]
    return np.array([Pm[st.loc[i, "rf_id"]].interpolate().bfill().ffill().to_numpy()
                     for i in idx])


def pump_energy(version: str):
    """All kept pumps of the three counties: coordinates, county, cell and the monthly energy
    2012-01..2022-11 (GWh) of the recorded arm (C0 preorigin v3) or the Chow-Lin arm (P2abc
    preorigin v2)."""
    l1 = D.l1_table(use="model_design", with_xy=True)
    xs, ys, ks, cs, Es = [], [], [], [], []
    if version == "chowlin":
        prod = guard.open_product(D.P2ABC["preorigin"], "model_design")
    for k in D.COUNTIES:
        p = D.Pumps("preorigin", k, "model_design", l1=l1, with_xy=True)
        if version == "recorded":
            E = np.nan_to_num(p.value[:, H0:H0 + NH], nan=0.0)
        else:
            z = np.load(prod.path(f"pumps_{D.FILETAG[k]}.npz"), allow_pickle=False)
            if not (z["uid"] == p.uid).all():
                raise ValueError(f"{k}: P2abc pump order differs from C0")
            E = np.nan_to_num(z["value"][:, H0:H0 + NH], nan=0.0)
        xs.append(p.x)
        ys.append(p.y)
        ks.append(np.full(len(p.uid), k))
        cs.append(p.cell)
        Es.append(E / 1e6)
        del p
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ks), np.concatenate(cs),
            np.concatenate(Es))


def weights(px, py, wxy, r_km) -> sparse.csr_matrix:
    ok = np.isfinite(px) & np.isfinite(py)
    pi = np.flatnonzero(ok)
    tree = cKDTree(np.column_stack([px[pi], py[pi]]))
    lists = tree.query_ball_point(wxy, r=r_km * 1000.0)
    rows, cols = [], []
    for i, lst in enumerate(lists):
        if lst:
            rows.append(np.full(len(lst), i))
            cols.append(pi[np.asarray(lst)])
    if not rows:
        return sparse.csr_matrix((len(wxy), len(px)))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(wxy), len(px)))


# ----------------------------------------------------------------------------- regression
def hp(v, w: int = 25) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.rolling(w, center=True, min_periods=12).mean()).to_numpy()


def fit(y, e, rain, regs):
    """(slope, t, partial R2, n) of high-passed y on high-passed e with covariates."""
    yy, ee = hp(y), hp(e)
    pp = [hp(rain), hp(np.roll(rain, 1)), hp(np.roll(rain, 2))]
    Z = np.column_stack([np.ones(len(yy)), ee] + pp + [hp(r) for r in regs])
    ok = np.isfinite(yy) & np.isfinite(Z).all(1)
    ok[:2] = False
    if ok.sum() < 48 or np.nanstd(ee[ok]) == 0:
        return None
    Zo, yo = Z[ok], yy[ok]
    b, *_ = np.linalg.lstsq(Zo, yo, rcond=None)
    res = yo - Zo @ b
    dof = ok.sum() - Z.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(Zo.T @ Zo)
    Z0 = np.delete(Zo, 1, axis=1)
    b0, *_ = np.linalg.lstsq(Z0, yo, rcond=None)
    r0 = yo - Z0 @ b0
    den = r0 @ r0
    pr2 = float(1 - (res @ res) / den) if den > 0 else np.nan
    return float(b[1]), float(b[1] / np.sqrt(cov[1, 1])), pr2, int(ok.sum())


def model_heads(path, key="heads_mean", scen=0) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    d = [str(x)[:7] for x in z["dates"]] if "dates" in z.files else None
    a = z[key]
    if a.ndim == 4:
        a = a[scen]
    if d is not None:
        i0 = d.index("2012-01")
        return np.asarray(a[:, :, i0:i0 + NH], dtype=float)
    return np.asarray(a[:, :, :NH], dtype=float)


def cell_of(x, y) -> np.ndarray:
    g = np.load(D.GRID_NPZ, allow_pickle=False)
    mask, x0, y0, dx = g["mask"], float(g["x0"]), float(g["y0"]), float(g["dx"])
    ny, nx = mask.shape
    cum = np.cumsum(mask.ravel()) - 1
    out = np.full(len(x), -1)
    for i, (a, b) in enumerate(zip(x, y, strict=False)):
        c, r = int((a - x0) // dx), int((b - y0) // dx)
        if 0 <= r < ny and 0 <= c < nx and mask[r, c]:
            out[i] = int(cum[r * nx + c])
    return out


# ----------------------------------------------------------------------------- statistics
def site_ratio_table(df: pd.DataFrame, col: str, lay: int) -> pd.DataFrame:
    piv = df.pivot_table(index="site", columns="layer", values=col, aggfunc="median")
    if lay not in piv or 2 not in piv:
        return pd.DataFrame(columns=["site", "b_l", "b_2", "ratio"])
    both = piv[[lay, 2]].dropna()
    out = pd.DataFrame({"site": both.index, "b_l": both[lay].to_numpy(),
                        "b_2": both[2].to_numpy()})
    out["ratio"] = out["b_l"] / out["b_2"]
    return out


def boot_median(x: np.ndarray, rng) -> tuple[float, float, float]:
    if len(x) < 3:
        return float(np.median(x)) if len(x) else np.nan, np.nan, np.nan
    bs = np.median(x[rng.integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    return float(np.median(x)), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def boot_site_mean(d: pd.DataFrame, rng) -> dict:
    """Site-cluster bootstrap of the well mean of D (one-sided p for mean > 0)."""
    d = d.dropna(subset=["D"])
    if len(d) == 0:
        return {"n_wells": 0}
    sites = d["site"].unique()
    grp = [d.loc[d["site"] == s, "D"].to_numpy() for s in sites]
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(sites), len(sites))
        v = np.concatenate([grp[i] for i in pick])
        means[b] = v.mean()
    return {"n_wells": int(len(d)), "n_sites": int(len(sites)), "mean_D": float(d["D"].mean()),
            "ci90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
            "p_one_sided": float((1 + (means <= 0).sum()) / (N_BOOT + 1))}


# ----------------------------------------------------------------------------- main
def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    meta = station_meta()
    LK.record(
        "P2def-f1-layer-response",
        units={"P2def:f1_candidate_wells": "heads/L1_clean v3 wells of Yunlin and Changhua "
                                           "with layer 1-4 and coordinates (the analysis keeps "
                                           "those with >= 60 observed months 2012-01..2022-11)"},
        scope="heads:2012-01..2022-11", purpose="design_time",
        note=f"P2f analysis declared in {DECL_ID}: local-only slopes of observed and modelled "
             "(cwa) heads on 3 km local energy, site ratios, year-shuffled placebos. Units "
             "are design-time (pilot s07 sites; heads read by earlier WPs).",
        members={"P2def:f1_candidate_wells": {"ids": sorted(meta["sid"].astype(str).tolist())}})
    w, H = wells(meta)
    wxy = w[["x_twd97", "y_twd97"]].to_numpy()
    rain = rain_for(w)
    w["cell"] = cell_of(w["x_twd97"].to_numpy(), w["y_twd97"].to_numpy())
    models = {"cwa_forward": model_heads(D.FWD_CWA),
              "cwa_fitfull_hindcast": model_heads(D.PILOT_PHYS / "hindcast_cwa.npz", key="H"),
              "chc_forward_supplementary": model_heads(D.FWD_CHC)}
    rows = []
    for version in ("recorded", "chowlin"):
        px, py, pk, pc, E = pump_energy(version)
        tot_county = {k: E[(pk == k) & (pc >= 0)].sum(0) for k in D.COUNTIES}
        W15 = weights(px, py, wxy, FAR_KM)
        e_all = E[np.isfinite(px) & np.isfinite(py)].sum(0)
        far = e_all[None, :] - np.asarray(W15 @ E)
        for r_km in RADII:
            Wr = weights(px, py, wxy, r_km)
            loc = np.asarray(Wr @ E)
            n_p = np.asarray(Wr.sum(1)).ravel()
            for i, wr in enumerate(w.itertuples()):
                kk = "yunlin" if wr.county == D.COUNTY_ZH["yunlin"] else "changhua"
                regs = [tot_county[kk], far[i]]
                rec = {"sid": wr.sid, "site": wr.site_id, "layer": int(wr.layer),
                       "county": kk, "r_km": r_km, "version": version,
                       "n_pumps": int(n_p[i]), "E_mean_GWh": float(loc[i].mean()),
                       "n_months": int(wr.n_months), "cell": int(wr.cell)}
                f = fit(H[i], loc[i], rain[i], regs)
                if f:
                    rec.update(b_obs=f[0], t_obs=f[1], pr2_obs=f[2], n_obs=f[3])
                    if version == "recorded" and r_km == PRIMARY_R:
                        pl = []
                        for rr in range(1, N_PLACEBO + 1):
                            sh, _ = placebos.year_shuffle(loc[i], HM, seed=6000 + rr,
                                                          years=list(range(2012, 2022)))
                            fp = fit(H[i], sh, rain[i], regs)
                            pl.append(fp if fp else (np.nan, np.nan, np.nan, 0))
                        pl = np.array(pl, dtype=float)
                        rec.update(pr2_placebo_mean=float(np.nanmean(pl[:, 2])),
                                   b_placebo_median=float(np.nanmedian(pl[:, 0])),
                                   D=f[2] - float(np.nanmean(pl[:, 2])))
                        for rr in range(N_PLACEBO):
                            rec[f"b_pl{rr + 1}"] = float(pl[rr, 0])
                if wr.cell >= 0:
                    for mname, Hm in models.items():
                        fm = fit(Hm[int(wr.layer) - 1, int(wr.cell)], loc[i], rain[i], regs)
                        if fm:
                            rec.update({f"b_{mname}": fm[0], f"t_{mname}": fm[1],
                                        f"pr2_{mname}": fm[2]})
                rows.append(rec)
            print(version, r_km, f"{time.time() - t_start:.0f}s", flush=True)
        del E
    df = pd.DataFrame(rows)
    df.to_parquet(out / "well_slopes.parquet", index=False)
    # ratios
    res = {"declaration": DECL_ID, "n_wells": int(len(w)),
           "wells_by_county_layer": w.groupby(["county", "layer"]).size().rename(
               "n").reset_index().to_dict(orient="records")}
    ratio_rows, site_rows = [], []
    cols = ["b_obs", "b_cwa_forward", "b_cwa_fitfull_hindcast", "b_chc_forward_supplementary"]
    for version in ("recorded", "chowlin"):
        for r_km in RADII:
            d = df[(df["version"] == version) & (df["r_km"] == r_km)]
            for scope, dd in (("yunlin", d[d["county"] == "yunlin"]),
                              ("changhua", d[d["county"] == "changhua"]), ("pooled", d)):
                for lyr in (1, 3, 4):
                    for col in cols:
                        if col not in dd:
                            continue
                        st = site_ratio_table(dd, col, lyr)
                        rng = np.random.default_rng(SEED)
                        med, lo, hi = boot_median(st["ratio"].to_numpy(), rng)
                        ratio_rows.append({
                            "version": version, "r_km": r_km, "scope": scope,
                            "pair": f"L{lyr}/L2", "heads": col[2:], "n_sites": int(len(st)),
                            "median_ratio": med, "ci90_lo": lo, "ci90_hi": hi,
                            "share_b2_negative": float((st["b_2"] < 0).mean())
                            if len(st) else np.nan})
                        if version == "recorded" and r_km == PRIMARY_R and lyr == 1:
                            for s in st.itertuples():
                                site_rows.append({"scope": scope, "heads": col[2:],
                                                  "site": s.site, "b_l1": s.b_l,
                                                  "b_l2": s.b_2, "ratio": s.ratio})
    rt = pd.DataFrame(ratio_rows)
    rt.to_csv(out / "layer_response_localonly.csv", index=False)
    pd.DataFrame(site_rows).to_parquet(out / "site_ratios.parquet", index=False)
    # placebo test and placebo ratio
    d = df[(df["version"] == "recorded") & (df["r_km"] == PRIMARY_R)]
    plc = {}
    for scope in ("yunlin", "changhua", "pooled"):
        dd = d if scope == "pooled" else d[d["county"] == scope]
        plc[scope] = {}
        for lyr in (1, 2):
            rng = np.random.default_rng(SEED)
            plc[scope][f"L{lyr}"] = boot_site_mean(dd[dd["layer"] == lyr], rng)
            plc[scope][f"L{lyr}"]["identified"] = bool(
                plc[scope][f"L{lyr}"].get("p_one_sided", 1.0) < 0.05)
        plc[scope]["L1_L2_ratio_locally_identified"] = bool(
            plc[scope]["L1"]["identified"] and plc[scope]["L2"]["identified"])
        pr = []
        for rr in range(1, N_PLACEBO + 1):
            st = site_ratio_table(dd, f"b_pl{rr}", 1)
            if len(st):
                pr.append(float(np.median(st["ratio"])))
        plc[scope]["placebo_site_median_ratio_median"] = float(np.median(pr)) if pr else None
        plc[scope]["placebo_site_median_ratio_range"] = [float(np.min(pr)), float(np.max(pr))] \
            if pr else None
    res["placebo"] = plc
    prim = rt[(rt["version"] == "recorded") & (rt["r_km"] == PRIMARY_R) & (rt["pair"] == "L1/L2")]
    res["primary"] = prim.to_dict(orient="records")
    for scope in ("yunlin", "changhua", "pooled"):
        o = prim[(prim["scope"] == scope) & (prim["heads"] == "obs")]
        m = prim[(prim["scope"] == scope) & (prim["heads"] == "cwa_forward")]
        if len(o) and len(m):
            res[f"model_cwa_inside_observed_ci90_{scope}"] = bool(
                o["ci90_lo"].iloc[0] <= m["median_ratio"].iloc[0] <= o["ci90_hi"].iloc[0])
    # depth proxy
    dp = {}
    l1 = D.l1_table(use="model_design")
    for k in ("yunlin", "changhua"):
        p = D.Pumps("preorigin", k, "model_design", l1=l1)
        Ek = np.nan_to_num(p.value[:, D.tidx("2012-01"):D.tidx("2022-12") + 1]).sum(1)
        ing = p.cell >= 0
        big = p.meter_hp_sum > 10
        dp[k] = {"share_energy_meters_gt10hp_in_grid": float(Ek[ing & big].sum() /
                                                              Ek[ing].sum()),
                 "share_pumps_on_meters_gt10hp_in_grid": float((ing & big).sum() / ing.sum()),
                 "well_depth_attribute_coverage": float(np.isfinite(p.well_depth).mean())}
    res["depth_proxy"] = dp
    res["seconds"] = time.time() - t_start
    res["code"] = D.p2def_code()
    D.save_json(res, out / "layer_response_summary.json")
    print(prim[prim["heads"].isin(["obs", "cwa_forward", "cwa_fitfull_hindcast",
                                   "chc_forward_supplementary"])].round(3).to_string())
    print({s: {k: v for k, v in plc[s].items() if k != "L1" and k != "L2"} for s in plc})
    print(dp)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
