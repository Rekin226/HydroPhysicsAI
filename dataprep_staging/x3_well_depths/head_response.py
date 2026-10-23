"""Which aquifer does the local pumping stress hit? A head-based estimate from design-time
data only (heads 2012-02..2022-12, the out-of-fold run's observation matrix).

Inputs, all pre-origin:

* heads: ``results/twin_runs/stage3_datum_gate_cwa/stage3_per_entry.npz`` (``obs``, and the
  R0 model's out-of-fold ``pred``), R1c's ``pred`` from ``stage3_datum_gate_ccz``;
* local census electricity: ``pump_electricity/preorigin/v3`` cells_total (kWh/month),
  summed over the cells within ``RADIUS_KM`` of each well;
* rain recharge at the cells (the R0 run's ``recharge_field``), averaged the same way.

Two estimators:

A. **Response per well** (descriptive). The monthly head is regressed on local electricity
   at lags 0-3 plus recharge at lags 0-3 and a linear trend. Gain G = sum of the electricity
   coefficients (m per GWh/month); lag centroid tau. Summaries by county, fan zone, layer.

B. **Nested-site two-layer mass balance** (the split). At a site with wells in two adjacent
   aquifers (upper u, lower l) the lumped balance is
       S_u dh_u/dt = -V_u q - c (h_u - h_l) + R_u,   S_l dh_l/dt = -V_l q + c (h_u - h_l)
   with q the local electricity. In monthly differences:
       dh_u = -a_u q - b_u d + g_u R + k_u,   dh_l = -a_l q + b_l d + g_l R + k_l,
   d = h_u - h_l of the previous month; a = V/S, b = c/S. Since c is common,
   S_l/S_u = b_u/b_l, so the share of the net stress volume taken from the upper aquifer is
       phi = a_u b_l / (a_u b_l + a_l b_u).
   phi is the NET share: extraction minus any return flow into that aquifer (the twin's
   return flow goes to layer 0, so the twin's own net phi is (f - r) / (1 - r)).
   a_u and a_l are free in sign; b_u, b_l >= 0. Moving-block bootstrap over years.

Validation of B: the same estimator on the R0 and R1c out-of-fold predictions, whose net
split is known from their fitted parameters, and a synthetic recovery test with the lumped
model itself.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
import x3const as K
from scipy.optimize import lsq_linear

sys.path.insert(0, str(K.REPO))

RADIUS_KM = 3.0
LAGS = 3
SEED = 2027
N_BOOT = 400
BLOCK = 12


# ----------------------------------------------------------------------------- data
def load() -> dict:
    from common import guard

    z = np.load(K.RUN["R0"] / "stage3_per_entry.npz", allow_pickle=True)
    zc = np.load(K.RUN["R1c"] / "stage3_per_entry.npz", allow_pickle=True)
    wells = pd.read_csv(K.RUN["R0"] / "stage3_kfold_wells.csv", dtype={"sid": str}
                        ).set_index("entry")
    sids = wells.loc[z["entry"], "sid"].to_numpy()
    assert np.array_equal(z["entry"], zc["entry"])
    with open(K.RUN["R0"] / "inputs.local.pkl", "rb") as fh:
        inp = pickle.load(fh)
    dates = pd.DatetimeIndex(inp.dates)[1:]
    assert len(dates) == z["obs"].shape[1] and dates[-1] <= pd.Timestamp("2022-12-01")
    R = np.asarray(inp.recharge_field, dtype=float)[:, 1:]
    del inp
    prod = guard.open_product(K.ELEC_PREORIGIN, "prior")
    e = np.load(prod.path("cells_total.npz"), allow_pickle=True)
    edates = pd.DatetimeIndex(pd.to_datetime(e["dates"], format="%Y-%m"))
    ix = edates.get_indexer(dates)
    assert (ix >= 0).all()
    E = np.asarray(e["value_all"], dtype=float)[:, ix]
    zc_map = np.load(K.ZONE_COUNTY / "zone_county_map.npz", allow_pickle=True)
    cent = np.asarray(zc_map["centroids"], dtype=float)
    st = pd.read_parquet(K.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    county = st.reindex(sids)["CountyName"].map({v: k for k, v in K.COUNTY_ZH.items()})
    return {"sid": sids, "x": z["x"].astype(float), "y": z["y"].astype(float),
            "layer": z["layer"].astype(int), "obs": z["obs"].astype(float),
            "pred_R0": z["pred"].astype(float), "pred_R1c": zc["pred"].astype(float),
            "dates": dates, "E": E, "R": R, "cent": cent, "county": county.to_numpy(),
            "electricity_product": f"{prod.manifest['product']} {prod.manifest['version']}"}


def local_series(d: dict, radius_km: float = RADIUS_KM) -> tuple[np.ndarray, np.ndarray]:
    """(wells, T) local electricity (GWh/month, sum over cells within radius) and recharge
    (mean over the same cells)."""
    xy = np.column_stack([d["x"], d["y"]])
    dist = np.sqrt(((xy[:, None, :] - d["cent"][None, :, :]) ** 2).sum(-1)) / 1000.0
    m = dist <= radius_km
    m[np.arange(len(xy)), dist.argmin(1)] = True
    q = (m.astype(float) @ d["E"]) / 1e6
    r = (m.astype(float) @ d["R"]) / m.sum(1, keepdims=True)
    return q, r


def site_of(sid: str) -> str:
    return sid[:6] + sid[7:]


# ----------------------------------------------------------------------------- estimator A
def lagmat(x: np.ndarray, lags: int) -> np.ndarray:
    out = np.full((len(x), lags + 1), np.nan)
    for k in range(lags + 1):
        out[k:, k] = x[:len(x) - k]
    return out


def response(h: np.ndarray, q: np.ndarray, r: np.ndarray, lags: int = LAGS) -> dict:
    Xq, Xr = lagmat(q, lags), lagmat(r, lags)
    t = np.arange(len(h)) / 120.0
    X = np.column_stack([np.ones(len(h)), t, Xq, Xr])
    ok = np.isfinite(h) & np.isfinite(X).all(1)
    if ok.sum() < 48:
        return {"n": int(ok.sum()), "gain": np.nan, "tau": np.nan, "r_q": np.nan}
    b, *_ = np.linalg.lstsq(X[ok], h[ok], rcond=None)
    bq = b[2:3 + lags]
    g = float(bq.sum())
    tau = float((np.arange(lags + 1) * bq).sum() / g) if abs(g) > 1e-12 else np.nan
    hq = h[ok] - X[ok][:, [0, 1]] @ b[:2]
    rq = float(np.corrcoef(hq, Xq[ok, 0])[0, 1])
    return {"n": int(ok.sum()), "gain": g, "tau": tau, "r_q": rq}


# ----------------------------------------------------------------------------- estimator B
def _fit_pair(hu, hl, q, r) -> dict | None:
    d = hu - hl
    dhu, dhl = np.diff(hu), np.diff(hl)
    qq, rr, dd = q[1:], r[1:], d[:-1]
    ok = np.isfinite(dhu) & np.isfinite(dhl) & np.isfinite(dd) & np.isfinite(qq) \
        & np.isfinite(rr)
    if ok.sum() < 48:
        return None
    one = np.ones(ok.sum())
    # dh_u = -a_u q - b_u d + g_u R + k_u ;  unknowns (a_u, b_u, g_u, k_u)
    Au = np.column_stack([-qq[ok], -dd[ok], rr[ok], one])
    Al = np.column_stack([-qq[ok], dd[ok], rr[ok], one])
    lo = [-np.inf, 0.0, -np.inf, -np.inf]
    hi = [np.inf] * 4
    su = lsq_linear(Au, dhu[ok], bounds=(lo, hi))
    sl = lsq_linear(Al, dhl[ok], bounds=(lo, hi))
    au, bu = su.x[0], su.x[1]
    al, bl = sl.x[0], sl.x[1]
    den = au * bl + al * bu
    phi = au * bl / den if abs(den) > 1e-15 and bu > 1e-6 and bl > 1e-6 else np.nan
    return {"n": int(ok.sum()), "a_u": float(au), "b_u": float(bu), "a_l": float(al),
            "b_l": float(bl), "phi": float(phi),
            "S_ratio_l_over_u": float(bu / bl) if bl > 1e-6 else np.nan}


def fit_pair(hu, hl, q, r, n_boot: int = N_BOOT, seed: int = SEED) -> dict | None:
    est = _fit_pair(hu, hl, q, r)
    if est is None:
        return None
    rng = np.random.default_rng(seed)
    T = len(hu)
    nb = int(np.ceil(T / BLOCK))
    phis = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - BLOCK, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        # keep month-to-month structure inside blocks; break it between blocks
        hu_b, hl_b = hu[idx].copy(), hl[idx].copy()
        cut = np.arange(BLOCK, T, BLOCK)
        hu_b[cut] = np.nan
        hl_b[cut] = np.nan
        e = _fit_pair(hu_b, hl_b, q[idx], r[idx])
        if e is not None and np.isfinite(e["phi"]):
            phis.append(np.clip(e["phi"], -3, 3))
    est["phi_boot_median"] = float(np.median(phis)) if phis else np.nan
    est["phi_lo90"] = float(np.percentile(phis, 5)) if phis else np.nan
    est["phi_hi90"] = float(np.percentile(phis, 95)) if phis else np.nan
    est["n_boot_ok"] = len(phis)
    return est


def nested_pairs(d: dict, upper: int = 0, lower: int = 1) -> list[tuple[int, int]]:
    sites = pd.Series([site_of(s) for s in d["sid"]])
    out = []
    for _s, idx in sites.groupby(sites).groups.items():
        lay = d["layer"][list(idx)]
        iu = [i for i, lk in zip(idx, lay, strict=True) if lk == upper]
        il = [i for i, lk in zip(idx, lay, strict=True) if lk == lower]
        if iu and il:
            out.append((iu[0], il[0]))
    return out


def synthetic_recovery(q: np.ndarray, r: np.ndarray, fit: dict, resid_sd: tuple,
                       phi_true: float, seed: int) -> float:
    """Simulate the lumped two-layer model with a given net split and refit it."""
    rng = np.random.default_rng(seed)
    T = len(q)
    bu, bl = max(fit["b_u"], 0.02), max(fit["b_l"], 0.02)
    tot = abs(fit["a_u"]) / bu * 1.0 + abs(fit["a_l"]) / bl     # stress scale ~ V/c
    tot = tot if tot > 0 else 1.0
    au = phi_true * tot * bu
    al = (1 - phi_true) * tot * bl
    hu, hl = np.zeros(T), np.zeros(T)
    rr = np.nan_to_num(r - np.nanmean(r))
    qq = np.nan_to_num(q - np.nanmean(q))
    for t in range(1, T):
        dd = hu[t - 1] - hl[t - 1]
        hu[t] = hu[t - 1] - au * qq[t] - bu * dd + 0.5 * rr[t] + rng.normal(0, resid_sd[0])
        hl[t] = hl[t - 1] - al * qq[t] + bl * dd + rng.normal(0, resid_sd[1])
    e = _fit_pair(hu, hl, q, r)
    return np.nan if e is None else e["phi"]


def twin_net_phi(run_dir) -> dict:
    """Net upper share of the twin's own stress: (f - r) / (1 - r)."""
    import json
    t = json.loads((run_dir / "stage3_theta.json").read_text())["theta"]
    f, r = float(t.get("pump_frac_shallow", 0.0)), float(t.get("return_frac", 0.0))
    return {"pump_frac_shallow": f, "return_frac": r, "net_phi": (f - r) / (1 - r)}


# ----------------------------------------------------------------------------- summaries
def responses(d: dict, q: np.ndarray, r: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(d["sid"])):
        for src in ("obs", "pred_R0", "pred_R1c"):
            a = response(d[src][i], q[i], r[i])
            rows.append({"entry_i": i, "sid": d["sid"][i], "site": site_of(d["sid"][i]),
                         "src": src, "county": d["county"][i],
                         "zone": K.zone_of_x(d["x"][i]), "layer": int(d["layer"][i]),
                         "q_mean_gwh": float(np.nanmean(q[i])), **a})
    return pd.DataFrame(rows)


def _boot_sites(df: pd.DataFrame, stat, n: int = 1000, seed: int = SEED) -> tuple:
    """90 % interval of ``stat`` under a bootstrap of whole sites (nested wells together)."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    sites = df["site"].to_numpy()
    us = np.unique(sites)
    rows_of = {s: np.flatnonzero(sites == s) for s in us}
    vals = []
    for _ in range(n):
        pick = rng.choice(us, len(us), replace=True)
        v = stat(df.iloc[np.concatenate([rows_of[s] for s in pick])])
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95))) if vals else \
        (np.nan, np.nan)


def summarise_gains(A: pd.DataFrame, by=("src", "county", "layer")) -> pd.DataFrame:
    out = []
    for key, g in A.groupby(list(by)):
        g = g[np.isfinite(g["gain"])]
        if len(g) == 0:
            continue
        lo, hi = _boot_sites(g, lambda x: float(np.median(x["gain"])))
        out.append({**dict(zip(by, key, strict=True)), "n_wells": int(len(g)),
                    "gain_median": float(np.median(g["gain"])), "gain_lo90": lo,
                    "gain_hi90": hi, "tau_median": float(np.nanmedian(g["tau"])),
                    "r_q_median": float(np.nanmedian(g["r_q"]))})
    return pd.DataFrame(out)


def ratio_upper_lower(A: pd.DataFrame, upper: int = 0, lower: int = 1,
                      by=("src", "county")) -> pd.DataFrame:
    """Median gain of the upper layer over the median gain of the lower layer, with a site
    bootstrap; and the median of the same ratio at nested sites only."""
    out = []
    for key, g in A.groupby(list(by)):
        g = g[np.isfinite(g["gain"])]

        def stat(x):
            gu = x.loc[x["layer"] == upper, "gain"]
            gl = x.loc[x["layer"] == lower, "gain"]
            return float(np.median(gu) / np.median(gl)) if len(gu) and len(gl) else np.nan

        lo, hi = _boot_sites(g, stat)
        piv = g.pivot_table(index="site", columns="layer", values="gain", aggfunc="first")
        nest = piv.dropna(subset=[c for c in (upper, lower) if c in piv.columns]) \
            if upper in piv.columns and lower in piv.columns else piv.iloc[0:0]
        ratios = (nest[upper] / nest[lower]) if len(nest) else pd.Series(dtype=float)
        out.append({**dict(zip(by, key, strict=True)), "ratio": stat(g), "ratio_lo90": lo,
                    "ratio_hi90": hi, "n_nested_sites": int(len(nest)),
                    "nested_ratio_median": float(np.median(ratios)) if len(ratios) else
                    np.nan,
                    "tau_upper": float(np.nanmedian(g.loc[g["layer"] == upper, "tau"])),
                    "tau_lower": float(np.nanmedian(g.loc[g["layer"] == lower, "tau"]))})
    return pd.DataFrame(out)


def pair_table(d: dict, q: np.ndarray, r: np.ndarray, upper: int, lower: int,
               n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for iu, il in nested_pairs(d, upper, lower):
        for src in ("obs", "pred_R0", "pred_R1c"):
            e = fit_pair(d[src][iu], d[src][il], q[iu], r[iu], n_boot=n_boot)
            if e:
                rows.append({"site": site_of(d["sid"][iu]), "src": src,
                             "county": d["county"][iu], "zone": K.zone_of_x(d["x"][iu]),
                             "upper": upper, "lower": lower, **e})
    return pd.DataFrame(rows)
