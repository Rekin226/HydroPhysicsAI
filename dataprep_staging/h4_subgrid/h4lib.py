"""WP H4: sub-grid observation operator for the high-swing wells (plan section 3, WP H4).

``h_well(t) = h_cell(t) - c(t)`` with the local-cone correction

    c_i(t) = a_s * sum_j g_s(r_ij) q_j^s(t) + a_l * sum_j g_l(r_ij) Qbar_j(t)

summed over census pumps j within ``R_MAX_M`` of well i:

* ``Q_j(t)`` m3/day: the pump's monthly kWh (Chow-Lin arm of ``yunlin_electricity/preorigin/v2``,
  plan P2a) times the twin's lift-dependent conversion ``eta J / (rho g max(z_g - h_1 + head_extra,
  2 m)) / dt`` at the pump's cell, with the hindcast's previous-month pump-layer head (the twin's
  own rule, ``engine.SparseTwin.rate_factor``);
* ``Qbar_j(t)`` the trailing 12-month mean of ``Q_j`` (slow part), ``q_j^s = Q_j - Qbar_j``
  (seasonal part);
* seasonal kernel (periodic Theis): ``g_s(r) = w_l / (2 pi T) * (|K0(r k sqrt(i))| -
  |K0(R k sqrt(i))|)``, ``k = sqrt(omega S / T)``, omega = 2 pi / 365.25 d;
* slow kernel (Thiem): ``g_l(r) = w_l / (2 pi T) * ln(R / r)``; both zero beyond ``R`` and with
  ``r`` floored at ``R_MIN_M``;
* ``T, S`` of the well's model layer at the well's cell (the calibrated zone values in
  ``changhua_abstraction/engine/v2``); ``w_l`` the point-extraction share of the well's layer
  under the twin's calibrated pump split (layer 0: f_shallow, the pump layer: 1 - f_shallow,
  deeper layers 0; the return flow is diffuse and is not a point source);
* ``a_s, a_l >= 0`` fitted by non-negative least squares on the 2012-02..2019-12 residuals
  ``obs - h_cell`` (per-well means removed: the free datum; each well weighted 1/n_i).

Data access goes through the A0 loader guard (``common.guard``). CPU only; no torch import.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.special import kv

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from common import guard  # noqa: E402
from common.paths import DATAPREP, REPO  # noqa: E402

# ----------------------------------------------------------------------------- products
ENGINE = DATAPREP / "changhua_abstraction/engine/v2"
UNITS = DATAPREP / "validation_units/v3"
ELEC = DATAPREP / "yunlin_electricity/preorigin/v2"
CENSUS = DATAPREP / "pump_electricity/census_L1/v3"
HEADS = DATAPREP / "heads/L1_clean/v3"
FOLD_UNIT = "fresh.folds.wells_all"

# ----------------------------------------------------------------------------- constants
J_PER_KWH = 3.6e6            # hydrophysics.twin.pumping.J_PER_KWH
RHO_G = 9800.0               # hydrophysics.twin.pumping.RHO_G
MIN_LIFT_M = 2.0             # hydrophysics.twin.pumping.MIN_LIFT_M
DT_DAYS = 30.0               # forward.build_model: FlowModel(dt_days=30.0)
PUMP_LAYER = 1               # engine.SparseTwin(pump_layer=1); engine_arrays pump_weight_layers
OMEGA = 2.0 * math.pi / 365.25   # annual angular frequency, 1/day
ELEC_T0 = "2007-01"          # index 0 of the per-pump arrays


@dataclass(frozen=True)
class Config:
    """Every design choice of the operator and of its tests (hashed by the registration)."""
    r_max_m: float = 2000.0
    r_min_m: float = 100.0
    arm: str = "chowlin"                       # chowlin | recorded
    counties: tuple = ("yunlin",)              # census counties whose pumps enter the kernel
    layer_weights: str = "twin_point_split"    # twin_point_split | ones
    n_mult: int = 2                            # 2: (a_s, a_l); 1: a_s = a_l
    slow_window: int = 12                      # trailing months of the slow part
    fit_start: str = "2012-02"                 # 2012-01 is the hindcast's initial condition
    fit_end: str = "2019-12"
    test_start: str = "2020-01"
    test_end: str = "2022-12"
    min_train: int = 24                        # observed fit-window months to fit or score
    min_test: int = 7                          # observed test months to score (harness default)
    c20_only: bool = False                     # True: only months with >= 20 clean days
    t_bound_scale: float = 1.0                 # sensitivity: T x this where T sits at a bound
    n_placebo: int = 39
    seed_shuffle: int = 20261014
    seed_distant: int = 20261114
    distant_km: float = 15.0
    alpha: float = 0.05
    q123_tol: float = 1.02

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["counties"] = list(self.counties)
        return d


PRIMARY = Config()
T_BOUND = 20000.0            # upper T bound reached by the theta (engine_arrays T max)


# ----------------------------------------------------------------------------- loading
@dataclass
class Data:
    cfg: Config
    dates: np.ndarray                    # (D,) 'YYYY-MM' 2012-01..2022-12
    sids: np.ndarray                     # (W,)
    county: np.ndarray                   # (W,) 'CH' / 'YL'
    layer: np.ndarray                    # (W,) model layer 0-3
    cell: np.ndarray                     # (W,)
    xy: np.ndarray                       # (W, 2) m TWD97
    h_cell: np.ndarray                   # (W, D) hindcast head at the well's cell and layer
    obs: np.ndarray                      # (W, D) observed monthly head, flags 0-1, else NaN
    T: np.ndarray                        # (W,) m2/day of the well's layer at its cell
    S: np.ndarray                        # (W,)
    w: np.ndarray                        # (W,) point-extraction share of the well's layer
    t_at_bound: np.ndarray               # (W,) bool
    pump_xy: np.ndarray                  # (P, 2)
    pump_county: np.ndarray              # (P,)
    pump_row: np.ndarray                 # (P,) row in its county's pumps_<county>.npz
    Qbar: np.ndarray                     # (P, D) m3/day trailing mean
    qs: np.ndarray                       # (P, D) m3/day seasonal part
    folds: list                          # [{"fold", "train", "test"}] sids
    engine_scalars: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _month_index(dates, m: str) -> int:
    return int(np.flatnonzero(np.asarray(dates) == m)[0])


def load_engine() -> dict:
    p = guard.open_product(ENGINE, "model_design")
    with np.load(p.path("engine_arrays.npz"), allow_pickle=False) as z:
        keys = ["dates", "obs_idx", "obs_layer", "sids", "well_county", "T", "S",
                "H_hindcast", "ground_elev", "scalars", "scalar_names", "pump_weight_layers",
                "pump_weight_values"]
        e = {k: np.array(z[k]) for k in keys}
    e["notes"] = list(p.notes)
    return e


def load_units() -> tuple[pd.DataFrame, list]:
    p = guard.open_product(UNITS, "selection")
    wells = p.read_table("wells.csv")
    wells["sid"] = wells["sid"].astype(str).str.zfill(8)
    units = json.loads(p.path("units.json").read_text(encoding="utf-8"))
    u = [x for x in units["units"] if x["id"] == FOLD_UNIT][0]
    folds = [{"fold": int(f["fold"]), "train": [str(s) for s in f["train"]],
              "test": [str(s) for s in f["test"]]} for f in u["folds"]]
    return wells, folds


def load_heads(sids: np.ndarray, dates: np.ndarray, cfg: Config) -> tuple[dict, list]:
    """Observed monthly heads (flags 0-1) as (W, D) matrices per declared use.

    Opens heads/L1_clean/v3 four times through the guard, once per use, each with
    ``window_end='2022-12'``: calibration_target (fit rows), datum_fit (datum), baseline
    (climatology), scoring_target (held-out score). All keep flags 0-1 here (U1/U2)."""
    out, notes = {}, []
    for use in ("calibration_target", "datum_fit", "baseline", "scoring_target"):
        p = guard.open_product(HEADS, use, window_end="2022-12")
        notes += [f"{use}: {n}" for n in p.notes]
        df = p.read_table("heads_monthly.parquet", columns=["sid", "month", "value", "flag",
                                                             "c20"])
        df = df[df["flag"].isin([0, 1]) & df["value"].notna()]
        if cfg.c20_only:
            df = df[df["c20"].astype(bool)]
        df = df[df["sid"].astype(str).isin(set(sids)) & df["month"].isin(set(dates))]
        m = np.full((len(sids), len(dates)), np.nan)
        wi = pd.Index(sids).get_indexer(df["sid"].astype(str))
        di = pd.Index(dates).get_indexer(df["month"])
        if (wi < 0).any() or (di < 0).any():
            raise RuntimeError("heads rows outside the well/date axes")
        dup = pd.DataFrame({"w": wi, "d": di}).duplicated().sum()
        if dup:
            raise RuntimeError(f"{dup} duplicate well-months in heads_monthly")
        m[wi, di] = df["value"].to_numpy(float)
        out[use] = m
    ref = out["calibration_target"]
    for k, v in out.items():
        if not np.array_equal(np.isnan(v), np.isnan(ref)) or not np.allclose(
                np.nan_to_num(v), np.nan_to_num(ref)):
            raise RuntimeError(f"heads read for {k!r} differ from calibration_target rows")
    return out, notes


def load_pumps(counties, arm: str) -> tuple[dict, list]:
    """Per-pump monthly kWh (arm) and coordinates for the census counties."""
    pe = guard.open_product(ELEC, "forcing_frozen")
    pc = guard.open_product(CENSUS, "model_design")
    notes = [f"forcing_frozen: {n}" for n in pe.notes] + [f"model_design: {n}" for n in pc.notes]
    cen = pc.read_table("census_pumps.parquet",
                        columns=["uid", "county", "x_twd97", "y_twd97", "cell", "kept"])
    cen = cen.set_index("uid")
    out = {}
    for cty in counties:
        name = {"yunlin": "pumps_yunlin.npz", "changhua": "pumps_changhua.npz",
                "chiayi": "pumps_chiayi_ingrid.npz"}[cty]
        if arm == "chowlin":
            with np.load(pe.path(name), allow_pickle=False) as z:
                uid, cell, val, dates = (np.array(z["uid"]), np.array(z["cell"]),
                                         np.array(z["value"]), np.array(z["dates"]))
        else:
            raise ValueError("per-pump arrays exist for the Chow-Lin arm only; the recorded "
                             "arm is built by recorded_pump_values()")
        c = cen.loc[uid]
        if not bool(c["kept"].all()):
            raise RuntimeError(f"{cty}: per-pump array holds pumps not kept in census_L1")
        out[cty] = {"uid": uid, "cell": cell, "kwh": val, "dates": dates,
                    "xy": c[["x_twd97", "y_twd97"]].to_numpy(float),
                    "cell_census": c["cell"].to_numpy(int)}
    return out, notes


def recorded_pump_values(cty: str, uid: np.ndarray) -> np.ndarray:
    """Recorded arm per pump (C0 preorigin/v3 pumps_<county>.npz, TPC even split), aligned on
    ``uid``; used only by the 'recorded' sensitivity member."""
    p = guard.open_product(DATAPREP / "pump_electricity/preorigin/v3", "forcing_frozen")
    name = {"yunlin": "pumps_yunlin.npz", "changhua": "pumps_changhua.npz",
            "chiayi": "pumps_chiayi_ingrid.npz"}[cty]
    with np.load(p.path(name), allow_pickle=False) as z:
        keys = list(z.files)
        u = np.array(z["uid"])
        vk = "value" if "value" in keys else "kwh"
        v = np.array(z[vk])
        d = np.array(z["dates"]) if "dates" in keys else None
    pos = pd.Index(u).get_indexer(uid)
    if (pos < 0).any():
        raise RuntimeError(f"{cty}: {int((pos < 0).sum())} pumps missing in the recorded arm")
    v = v[pos]
    if d is not None and str(d[0]) != ELEC_T0:
        raise RuntimeError(f"recorded arm starts {d[0]}, expected {ELEC_T0}")
    return v


def rate_factor(ge: np.ndarray, h_pump: np.ndarray, eta: float, hx: float) -> np.ndarray:
    """m3/day per kWh/month at pump-layer head ``h_pump`` (engine.SparseTwin.rate_factor)."""
    head = ge - h_pump + hx
    return eta * J_PER_KWH / (RHO_G * np.maximum(head, MIN_LIFT_M)) / DT_DAYS


def build_data(cfg: Config = PRIMARY, with_heads: bool = True) -> Data:
    """Assemble everything; ``with_heads=False`` never opens the head product (``obs`` is all
    NaN): used by the design-time checks run before the pre-registration."""
    e = load_engine()
    dates = np.array([str(x)[:7] for x in e["dates"]])
    sids = np.array([str(s) for s in e["sids"]])
    wells, folds = load_units()
    wells = wells.set_index("sid").loc[sids]
    if not (np.array_equal(wells["cell"].to_numpy(int), e["obs_idx"])
            and np.array_equal(wells["layer"].to_numpy(int), e["obs_layer"])):
        raise RuntimeError("validation_units wells.csv does not match the engine's wells")
    sc = dict(zip([str(x) for x in e["scalar_names"]], map(float, e["scalars"]), strict=True))
    pw = dict(zip(map(int, e["pump_weight_layers"]), map(float, e["pump_weight_values"]),
                  strict=True))
    f_sh = sc["f_shallow"]
    # the engine's net source weights are layer 0: -f_sh + return, layer 1: -(1 - f_sh)
    if abs(pw[PUMP_LAYER] + (1.0 - f_sh)) > 1e-9:
        raise RuntimeError(f"pump split mismatch: {pw} vs f_shallow {f_sh}")
    oi, ol = e["obs_idx"], e["obs_layer"]
    h_cell = e["H_hindcast"][ol, oi, :]
    T = e["T"][ol, oi].astype(float)
    S = e["S"][ol, oi].astype(float)
    at_bound = np.isclose(T, T_BOUND)
    if cfg.t_bound_scale != 1.0:
        T = np.where(at_bound, T * cfg.t_bound_scale, T)
    if cfg.layer_weights == "twin_point_split":
        w = np.where(ol == 0, f_sh, np.where(ol == PUMP_LAYER, 1.0 - f_sh, 0.0))
    elif cfg.layer_weights == "ones":
        w = np.ones(len(ol))
    else:
        raise ValueError(cfg.layer_weights)
    if with_heads:
        heads, hnotes = load_heads(sids, dates, cfg)
    else:
        heads = {"calibration_target": np.full((len(sids), len(dates)), np.nan)}
        hnotes = ["heads not opened (design-time check)"]
    pumps, pnotes = load_pumps(cfg.counties, "chowlin")
    # conversion kWh -> m3/day with the hindcast's lift (previous month's pump-layer head)
    ge = e["ground_elev"].astype(float)
    H1 = e["H_hindcast"][PUMP_LAYER]                     # (A, D) state at each date
    D = len(dates)
    k0 = None
    xy_l, cty_l, row_l, Qb_l, qs_l = [], [], [], [], []
    for cty, pp in pumps.items():
        pd_dates = np.array([str(x) for x in pp["dates"]])
        if pd_dates[0] != ELEC_T0 or pd_dates[-1] != dates[-1]:
            raise RuntimeError(f"{cty}: per-pump dates {pd_dates[0]}..{pd_dates[-1]}")
        k0 = _month_index(pd_dates, dates[0])
        lo = k0 - (cfg.slow_window - 1)
        kwh = pp["kwh"] if cfg.arm == "chowlin" else recorded_pump_values(cty, pp["uid"])
        kwh = np.nan_to_num(np.asarray(kwh, float)[:, lo:], nan=0.0)   # missing reads as 0
        ok = np.isfinite(pp["xy"]).all(axis=1)
        cell = pp["cell"].astype(int)
        # lift factor per month on the extended axis: months <= dates[0] use h0
        Dx = kwh.shape[1]
        h_prev = np.empty((len(cell), Dx))
        n_pre = cfg.slow_window - 1          # months before dates[0]
        h_prev[:, : n_pre + 1] = H1[cell, 0][:, None]
        h_prev[:, n_pre + 1:] = H1[cell, : D - 1]
        rf = rate_factor(ge[cell][:, None], h_prev, sc["eta"], sc["head_extra_m"])
        Q = kwh * rf
        cs = np.cumsum(np.concatenate([np.zeros((len(cell), 1)), Q], axis=1), axis=1)
        Qbar = (cs[:, cfg.slow_window:] - cs[:, :-cfg.slow_window]) / cfg.slow_window
        Qd = Q[:, n_pre:]
        if Qbar.shape[1] != D or Qd.shape[1] != D:
            raise RuntimeError("slow/seasonal axis mismatch")
        xy_l.append(pp["xy"][ok])
        cty_l.append(np.full(int(ok.sum()), cty))
        row_l.append(np.flatnonzero(ok))
        Qb_l.append(Qbar[ok])
        qs_l.append((Qd - Qbar)[ok])
    notes = list(e["notes"]) + hnotes + pnotes
    return Data(cfg=cfg, dates=dates, sids=sids, county=np.array(
        ["CH" if c == "彰化縣" else "YL" for c in e["well_county"]]), layer=ol.astype(int),
        cell=oi.astype(int), xy=wells[["x", "y"]].to_numpy(float), h_cell=h_cell,
        obs=heads["calibration_target"], T=T, S=S, w=w, t_at_bound=at_bound,
        pump_xy=np.concatenate(xy_l), pump_county=np.concatenate(cty_l),
        pump_row=np.concatenate(row_l), Qbar=np.concatenate(Qb_l), qs=np.concatenate(qs_l),
        folds=folds, engine_scalars=sc, notes=notes)


# ----------------------------------------------------------------------------- kernels
def kernel_values(r: np.ndarray, T: float, S: float, w: float, cfg: Config
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(g_s, g_l) in day/m2 for distances ``r`` (m) < r_max (callers pass only those)."""
    R = cfg.r_max_m
    rr = np.maximum(np.asarray(r, float), cfg.r_min_m)
    kap = math.sqrt(OMEGA * S / T)
    z = np.exp(1j * math.pi / 4.0) * kap
    fs = np.abs(kv(0, rr * z)) - abs(kv(0, R * z))
    fl = np.log(R / rr)
    c = w / (2.0 * math.pi * T)
    return c * fs, c * fl


def operator_matrices(d: Data, pump_xy: np.ndarray | None = None,
                      centres: np.ndarray | None = None, support: np.ndarray | None = None
                      ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Sparse (W, P) kernel matrices G_s, G_l and pumps-within-R counts.

    ``pump_xy`` replaces the pump coordinates (shuffle placebo); ``centres`` (W, 2) replaces the
    point at which each well's kernel is evaluated (distant placebo); ``support`` (W,) bool
    zeroes rows of wells outside it."""
    from scipy.spatial import cKDTree

    cfg = d.cfg
    pxy = d.pump_xy if pump_xy is None else pump_xy
    cxy = d.xy if centres is None else centres
    tree = cKDTree(pxy)
    rows_s, rows_l, ii, jj, n = [], [], [], [], np.zeros(len(d.sids), int)
    for i in range(len(d.sids)):
        if support is not None and not support[i]:
            continue
        if d.w[i] == 0.0 or not np.isfinite(cxy[i]).all():
            continue
        js = np.array(tree.query_ball_point(cxy[i], cfg.r_max_m - 1e-9), dtype=int)
        if js.size == 0:
            continue
        r = np.sqrt(((pxy[js] - cxy[i]) ** 2).sum(axis=1))
        gs, gl = kernel_values(r, d.T[i], d.S[i], d.w[i], cfg)
        rows_s.append(gs)
        rows_l.append(gl)
        ii.append(np.full(js.size, i))
        jj.append(js)
        n[i] = js.size
    shape = (len(d.sids), len(pxy))
    if not ii:
        z = sp.csr_matrix(shape)
        return z, z.copy(), n
    ii, jj = np.concatenate(ii), np.concatenate(jj)
    Gs = sp.csr_matrix((np.concatenate(rows_s), (ii, jj)), shape=shape)
    Gl = sp.csr_matrix((np.concatenate(rows_l), (ii, jj)), shape=shape)
    return Gs, Gl, n


def regressors(d: Data, Gs, Gl) -> tuple[np.ndarray, np.ndarray]:
    """Unit-multiplier correction components (W, D) in m: seasonal S and slow L."""
    return np.asarray(Gs @ d.qs), np.asarray(Gl @ d.Qbar)


# ----------------------------------------------------------------------------- masks
def window_mask(d: Data, a: str, b: str) -> np.ndarray:
    return (d.dates >= a) & (d.dates <= b)


def masks(d: Data) -> dict:
    cfg = d.cfg
    fit = window_mask(d, cfg.fit_start, cfg.fit_end)[None, :] & np.isfinite(d.obs)
    test = window_mask(d, cfg.test_start, cfg.test_end)[None, :] & np.isfinite(d.obs)
    ntr, nte = fit.sum(1), test.sum(1)
    return {"fit": fit, "test": test, "n_train": ntr, "n_test": nte,
            "fit_ok": ntr >= cfg.min_train, "score_ok": (ntr >= cfg.min_train)
            & (nte >= cfg.min_test)}


def quartiles(d: Data, m: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Quartile 1-4 by the sd of the observed fit-window heads (wells with >= min_train
    months; 0 = not classified), the sd, and the cut points."""
    o = np.where(m["fit"], d.obs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(o, axis=1, ddof=1)
    ok = m["fit_ok"] & np.isfinite(sd)
    cuts = list(np.quantile(sd[ok], [0.25, 0.5, 0.75]))
    q = np.zeros(len(sd), int)
    q[ok] = 1 + np.searchsorted(cuts, sd[ok], side="right")
    return q, sd, [float(c) for c in cuts]


# ----------------------------------------------------------------------------- fit / score
def fit_multipliers(d: Data, X: list[np.ndarray], m: dict, train: np.ndarray) -> np.ndarray:
    """Non-negative multipliers for the corrections ``X`` (list of (W, D)), fitted on the
    fit-window residuals ``obs - h_cell`` of the wells in ``train`` (bool (W,)) with at least
    min_train observed months; per-well means removed; each well weighted 1/n_i."""
    sel = train & m["fit_ok"]
    if d.cfg.n_mult == 1:
        X = [X[0] + X[1]]
    ys, As = [], []
    for i in np.flatnonzero(sel):
        t = m["fit"][i]
        y = d.obs[i, t] - d.h_cell[i, t]
        A = np.stack([-x[i, t] for x in X], axis=1)
        y = y - y.mean()
        A = A - A.mean(axis=0)
        wt = 1.0 / math.sqrt(t.sum())
        ys.append(y * wt)
        As.append(A * wt)
    y = np.concatenate(ys)
    A = np.concatenate(As)
    if not np.any(A):
        a = np.zeros(A.shape[1])
    else:
        a, _ = nnls(A, y)
    if d.cfg.n_mult == 1:
        a = np.array([a[0], a[0]])
    return a


def correction(X: list[np.ndarray], a: np.ndarray) -> np.ndarray:
    return a[0] * X[0] + a[1] * X[1]


def climatology(d: Data, m: dict) -> np.ndarray:
    """Per-well month-of-year mean of the observed fit-window heads; a calendar month without
    fit-window data falls back to the well's fit-window mean."""
    o = np.where(m["fit"], d.obs, np.nan)
    moy = np.array([int(s[5:7]) for s in d.dates])
    out = np.full(d.obs.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_all = np.nanmean(o, axis=1)
        for k in range(1, 13):
            sel = moy == k
            mu = np.nanmean(o[:, sel], axis=1)
            mu = np.where(np.isfinite(mu), mu, mean_all)
            out[:, sel] = mu[:, None]
    return out


def per_well_mse(d: Data, m: dict, corr: np.ndarray | None, clim: np.ndarray) -> dict:
    """Per-well test-window MSE of the model (training-mean datum over the fit window) and of
    the climatology; ``corr`` None = base model."""
    pred = d.h_cell if corr is None else d.h_cell - corr
    r = d.obs - pred
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        datum = np.nanmean(np.where(m["fit"], r, np.nan), axis=1)
        e = np.where(m["test"], r - datum[:, None], np.nan)
        ec = np.where(m["test"], d.obs - clim, np.nan)
        mse = np.nanmean(e ** 2, axis=1)
        msec = np.nanmean(ec ** 2, axis=1)
    ok = m["score_ok"]
    return {"mse": np.where(ok, mse, np.nan), "mse_clim": np.where(ok, msec, np.nan)}


def summarise(base: dict, cor: dict, q: np.ndarray, wells: np.ndarray) -> dict:
    """Quartile metrics on the scored wells in ``wells`` (bool (W,))."""
    out = {}
    for k in (1, 2, 3, 4):
        s = wells & (q == k) & np.isfinite(base["mse"]) & np.isfinite(cor["mse"])
        if not s.any():
            out[f"Q{k}"] = {"n_wells": 0}
            continue
        b, c, cl = base["mse"][s], cor["mse"][s], base["mse_clim"][s]
        out[f"Q{k}"] = {"n_wells": int(s.sum()),
                        "rmse_base": float(np.sqrt(b.mean())),
                        "rmse_corrected": float(np.sqrt(c.mean())),
                        "rmse_clim": float(np.sqrt(cl.mean())),
                        "rmse_ratio_corrected_over_base": float(np.sqrt(c.mean() / b.mean())),
                        "excess_base_m2": float((b - cl).sum()),
                        "excess_corrected_m2": float((c - cl).sum()),
                        "excess_reduction_m2": float((b - c).sum())}
    s = wells & (q > 0) & np.isfinite(base["mse"])
    out["all"] = {"n_wells": int(s.sum()),
                  "rmse_base": float(np.sqrt(base["mse"][s].mean())),
                  "rmse_corrected": float(np.sqrt(cor["mse"][s].mean())),
                  "rmse_clim": float(np.sqrt(base["mse_clim"][s].mean()))}
    return out


def run_t1(d: Data, X: list[np.ndarray], m: dict, q: np.ndarray, clim: np.ndarray) -> dict:
    """Held-out years: multipliers from every well's fit window; scored on the test window."""
    a = fit_multipliers(d, X, m, np.ones(len(d.sids), bool))
    corr = correction(X, a)
    base = per_well_mse(d, m, None, clim)
    cor = per_well_mse(d, m, corr, clim)
    return {"a": a, "corr": corr, "base": base, "cor": cor,
            "summary": summarise(base, cor, q, np.ones(len(d.sids), bool))}


def run_t2(d: Data, X: list[np.ndarray], m: dict, q: np.ndarray, clim: np.ndarray) -> dict:
    """Held-out sites (site-grouped folds, A1 fold definition) and years: multipliers from the
    fold's training wells (test and buffer wells excluded), applied to its test wells."""
    pos = {s: i for i, s in enumerate(d.sids)}
    corr = np.zeros_like(d.h_cell)
    tested = np.zeros(len(d.sids), bool)
    fold_a = {}
    for f in d.folds:
        tr = np.zeros(len(d.sids), bool)
        tr[[pos[s] for s in f["train"] if s in pos]] = True
        te = np.zeros(len(d.sids), bool)
        te[[pos[s] for s in f["test"] if s in pos]] = True
        a = fit_multipliers(d, X, m, tr)
        fold_a[f["fold"]] = a
        c = correction(X, a)
        corr[te] = c[te]
        tested |= te
    base = per_well_mse(d, m, None, clim)
    cor = per_well_mse(d, m, corr, clim)
    return {"a_by_fold": fold_a, "corr": corr, "base": base, "cor": cor,
            "summary": summarise(base, cor, q, tested)}


def run_t2_spatial_only(d: Data, X: list[np.ndarray], m: dict, q: np.ndarray) -> dict:
    """Reported only: the same fold fits scored on the test wells' FIT-window months (held-out
    sites, in-sample years; datum and climatology in sample)."""
    m2 = dict(m)
    m2["test"] = m["fit"]
    m2["score_ok"] = m["fit_ok"]
    clim = climatology(d, m)
    return run_t2(d, X, m2, q, clim)


def q4_reduction(res: dict) -> float:
    return float(res["summary"]["Q4"].get("excess_reduction_m2", float("nan")))


def amplitude_ratio(d: Data, m: dict, corr: np.ndarray | None, wells: np.ndarray) -> float:
    """Median over ``wells`` of sd(prediction) / sd(observed) over the scored test months."""
    pred = d.h_cell if corr is None else d.h_cell - corr
    out = []
    for i in np.flatnonzero(wells & m["score_ok"]):
        t = m["test"][i]
        so = d.obs[i, t].std(ddof=1)
        if so > 0:
            out.append(pred[i, t].std(ddof=1) / so)
    return float(np.median(out)) if out else float("nan")


# ----------------------------------------------------------------------------- placebos
def shuffle_xy(d: Data, rng: np.random.Generator) -> np.ndarray:
    """Pump coordinates permuted among the pumps of the same county."""
    xy = d.pump_xy.copy()
    for c in np.unique(d.pump_county):
        idx = np.flatnonzero(d.pump_county == c)
        xy[idx] = d.pump_xy[idx[rng.permutation(idx.size)]]
    return xy


def distant_centres(d: Data, rng: np.random.Generator, support: np.ndarray) -> np.ndarray:
    """For each supported well, a donor point drawn uniformly from the pump positions at least
    ``distant_km + r_max`` from the well (so every pump within r_max of the donor is more than
    ``distant_km`` from the well)."""
    cfg = d.cfg
    lim = cfg.distant_km * 1000.0 + cfg.r_max_m
    cen = np.full_like(d.xy, np.nan)
    for i in np.flatnonzero(support):
        dist = np.sqrt(((d.pump_xy - d.xy[i]) ** 2).sum(axis=1))
        cand = np.flatnonzero(dist >= lim)
        cen[i] = d.pump_xy[cand[rng.integers(cand.size)]]
    return cen


def with_config(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)


def repo_rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO))
