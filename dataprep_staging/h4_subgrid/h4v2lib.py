"""WP H4 repair round: the SUB-GRID observation operator (cell-mean kernel subtracted).

The v1 operator ``h_well = h_cell - sum_j Q_j k(r_j)`` (``h4lib``) subtracts the whole local
cone, although the 1 km hindcast already carries the cell-average cone (the verifier's major
finding: the v1 kernel averaged over the well's own cell correlates 0.993 with the point
value). The repaired operator keeps only the part a cell-mean head cannot carry:

    c_i(t) = a_s sum_j g_s,ij q_j^s(t) + a_l sum_j g_l,ij Qbar_j(t)

    g_ij = w_i / (2 pi T_i) * ( f(|x_i - x_j|) - < f(|x - x_j|) >_{x in cell(i)} )

with ``f = |K0(r k sqrt(i))|`` (seasonal) or ``f = -ln r`` (slow), ``r`` floored at
``r_min`` in both terms, ``< >_cell`` the mean over the model cell (1 km square) that contains
the well, summed over census pumps within ``r_max`` of the well. The truncation offsets of v1
(``|K0(R ..)|``, ``ln R``) cancel in the difference. Averaged over the cell, the correction is
zero by construction (up to the r_max truncation), so it cannot rescale the cell-mean response
the hindcast already carries.

``kernel_mode``:

* ``"cellmean"`` (primary): the difference above;
* ``"centroid"`` (member): the model-consistent reference ``f(max(|c_i - c_j|, r_e))`` with
  ``c`` the cell centroids of the well and of the pump and ``r_e = 0.198 dx`` (Peaceman's
  equivalent radius of a finite-difference cell), i.e. what a cell-centred source and a
  cell-value observation carry;
* ``"full"``: the v1 kernel (``h4lib.kernel_values``), for diagnostics only.

Everything else (Q_j, T, S, w, the NNLS fit, the scorers, the placebos' logic) is ``h4lib``,
imported unchanged (``h4lib.py`` is hashed by subgrid_operator/v1 and is not edited).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import correlate1d, map_coordinates
from scipy.special import kv

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
from common import guard  # noqa: E402
from common.paths import REPO  # noqa: E402

GRID_NPZ = REPO / "results/twin_forward/datum_gate_innov.npz"
PEACEMAN = 0.198                  # r_e / dx for a square finite-difference cell
# quartile cuts of v1 (sd of the observed 2012-02..2019-12 heads of the 158 calibration wells,
# reports/H4_evidence/registered/scores.json quartile_cuts_sd_m)
V1_QUARTILE_CUTS = (0.8758816018678, 1.4998585062879892, 2.3534572169650394)
TABLE_H_M = 20.0                  # quadrature spacing of the cell-mean tables


@dataclass(frozen=True)
class ConfigV2:
    """Design of the repaired operator and its tests (hashed by the v2 registration).

    Field names shared with ``h4lib.Config`` have the same meaning (h4lib functions read them
    by name)."""
    r_max_m: float = 2000.0
    r_min_m: float = 100.0
    arm: str = "chowlin"
    counties: tuple = ("yunlin", "changhua", "chiayi")
    layer_weights: str = "twin_point_split"
    n_mult: int = 2
    slow_window: int = 12
    fit_start: str = "2012-02"
    fit_end: str = "2019-12"
    test_start: str = "2020-01"
    test_end: str = "2022-12"
    min_train: int = 24
    min_test: int = 7
    c20_only: bool = False
    t_bound_scale: float = 1.0
    kernel_mode: str = "cellmean"
    t_profile: tuple = ()                # () = no profile; e.g. (1.0, 2.0, 4.0)
    n_placebo: int = 199
    seed_shuffle: int = 20261017
    seed_distant: int = 20261117
    distant_km: float = 15.0
    alpha: float = 0.05
    q123_tol: float = 1.02
    quartile_cuts: tuple = ()            # () = from the data (v1 rule); else fixed cuts (m)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


PRIMARY = ConfigV2()


def with_config(cfg, **kw):
    return replace(cfg, **kw)


# ----------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    dx: float
    nx: int
    ny: int
    mask: np.ndarray
    centroids: np.ndarray          # (A, 2) in the solver's active-cell order
    index: np.ndarray              # (ny, nx) active index or -1


def load_grid() -> Grid:
    with np.load(GRID_NPZ, allow_pickle=False) as z:
        x0, y0, dx = float(z["x0"]), float(z["y0"]), float(z["dx"])
        nx, ny = int(z["nx"]), int(z["ny"])
        mask = np.asarray(z["mask"], bool)
    rows, cols = np.nonzero(mask)            # hydrophysics.twin.grid.FanGrid.centroids order
    cen = np.column_stack([x0 + (cols + 0.5) * dx, y0 + (rows + 0.5) * dx])
    idx = np.full(mask.shape, -1, int)
    idx[rows, cols] = np.arange(rows.size)
    return Grid(x0, y0, dx, nx, ny, mask, cen, idx)


def cell_centre_of(g: Grid, xy: np.ndarray) -> np.ndarray:
    """Centre of the grid cell (active or not) that contains each point."""
    xy = np.atleast_2d(np.asarray(xy, float))
    ix = np.floor((xy[:, 0] - g.x0) / g.dx)
    iy = np.floor((xy[:, 1] - g.y0) / g.dx)
    return np.column_stack([g.x0 + (ix + 0.5) * g.dx, g.y0 + (iy + 0.5) * g.dx])


def active_cell_of(g: Grid, xy: np.ndarray) -> np.ndarray:
    """Active-cell index of each point, -1 outside the active mask."""
    xy = np.atleast_2d(np.asarray(xy, float))
    fin = np.isfinite(xy).all(axis=1)
    xyf = np.where(fin[:, None], xy, -1e12)
    ix = np.floor((xyf[:, 0] - g.x0) / g.dx).astype(np.int64)
    iy = np.floor((xyf[:, 1] - g.y0) / g.dx).astype(np.int64)
    ok = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny) & fin
    out = np.full(len(xy), -1, int)
    out[ok] = g.index[iy[ok], ix[ok]]
    return out


# ----------------------------------------------------------------------------- kernels
def _f_seasonal(r: np.ndarray, kap: float) -> np.ndarray:
    z = np.exp(1j * math.pi / 4.0) * kap
    return np.abs(kv(0, r * z))


def _f_slow(r: np.ndarray) -> np.ndarray:
    return -np.log(r)


class CellMeanTables:
    """Cell means ``M(s) = < f(|o - s|) >_{o in [-dx/2, dx/2]^2}`` of the floored kernels as
    functions of the pump position ``s`` relative to the cell centre, tabulated on a
    ``TABLE_H_M`` grid (trapezoid rule in each axis) and read by bilinear interpolation. One
    table pair per distinct (T, S)."""

    def __init__(self, dx: float, r_max: float, r_min: float, h: float = TABLE_H_M):
        self.dx, self.r_max, self.r_min, self.h = dx, r_max, r_min, h
        n_half = int(math.ceil((r_max + dx * math.sqrt(0.5) + dx / 2 + 4 * h) / h))
        self.u = np.arange(-n_half, n_half + 1) * h
        n_box = int(round(dx / h))
        if abs(n_box * h - dx) > 1e-9:
            raise ValueError("dx must be a multiple of the table spacing")
        wts = np.ones(n_box + 1)
        wts[0] = wts[-1] = 0.5
        self.wts = wts / wts.sum()
        U, V = np.meshgrid(self.u, self.u, indexing="ij")
        self.r = np.maximum(np.hypot(U, V), r_min)
        self._cache: dict = {}

    def _box(self, f: np.ndarray) -> np.ndarray:
        g = correlate1d(f, self.wts, axis=0, mode="nearest")
        return correlate1d(g, self.wts, axis=1, mode="nearest")

    def tables(self, T: float, S: float) -> tuple[np.ndarray, np.ndarray]:
        key = (float(T), float(S))
        if key not in self._cache:
            kap = math.sqrt(L.OMEGA * S / T)
            self._cache[key] = (self._box(_f_seasonal(self.r, kap)), self._box(_f_slow(self.r)))
        return self._cache[key]

    def lookup(self, T: float, S: float, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell means at relative positions ``s`` (n, 2) m."""
        ms, ml = self.tables(T, S)
        lim = self.u[-1] - self.dx / 2 - 2 * self.h
        if np.abs(s).max(initial=0.0) > lim:
            raise ValueError("relative position outside the cell-mean table")
        c = (np.asarray(s, float) - self.u[0]) / self.h
        coords = np.vstack([c[:, 0], c[:, 1]])
        return (map_coordinates(ms, coords, order=1, mode="nearest"),
                map_coordinates(ml, coords, order=1, mode="nearest"))


def subgrid_values(r_point: np.ndarray, s_rel: np.ndarray, pump_cc: np.ndarray,
                   well_cc: np.ndarray, T: float, S: float, w: float, cfg,
                   tabs: CellMeanTables, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """(g_s, g_l) in day/m2 for the pumps at distance ``r_point`` from the evaluation point;
    ``s_rel`` pump position minus the evaluation cell's centre; ``pump_cc`` the pumps' cell
    centres, ``well_cc`` the evaluation cell's centre (centroid mode)."""
    rp = np.maximum(np.asarray(r_point, float), cfg.r_min_m)
    kap = math.sqrt(L.OMEGA * S / T)
    fs_p, fl_p = _f_seasonal(rp, kap), _f_slow(rp)
    if cfg.kernel_mode == "cellmean":
        fs_c, fl_c = tabs.lookup(T, S, s_rel)
    elif cfg.kernel_mode == "centroid":
        rc = np.maximum(np.sqrt(((pump_cc - well_cc) ** 2).sum(axis=1)), PEACEMAN * dx)
        fs_c, fl_c = _f_seasonal(rc, kap), _f_slow(rc)
    elif cfg.kernel_mode == "full":
        return L.kernel_values(r_point, T, S, w, cfg)
    else:
        raise ValueError(cfg.kernel_mode)
    c = w / (2.0 * math.pi * T)
    return c * (fs_p - fs_c), c * (fl_p - fl_c)


def operator_matrices(d: L.Data, grid: Grid, tabs: CellMeanTables,
                      pump_xy: np.ndarray | None = None, centres: np.ndarray | None = None,
                      support: np.ndarray | None = None
                      ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Sparse (W, P) sub-grid kernel matrices and pumps-within-r_max counts.

    ``pump_xy`` replaces the pump coordinates (shuffle placebo; each pump keeps its Q series
    and the cell centre of its new position); ``centres`` (W, 2) replaces the evaluation point
    (distant placebo: the donor point and the donor's own cell); ``support`` zeroes rows."""
    from scipy.spatial import cKDTree

    cfg = d.cfg
    pxy = d.pump_xy if pump_xy is None else pump_xy
    pcc = cell_centre_of(grid, pxy)
    cxy = d.xy if centres is None else centres
    tree = cKDTree(pxy)
    rows_s, rows_l, ii, jj = [], [], [], []
    n = np.zeros(len(d.sids), int)
    for i in range(len(d.sids)):
        if support is not None and not support[i]:
            continue
        if d.w[i] == 0.0 or not np.isfinite(cxy[i]).all():
            continue
        js = np.array(tree.query_ball_point(cxy[i], cfg.r_max_m - 1e-9), dtype=int)
        if js.size == 0:
            continue
        r = np.sqrt(((pxy[js] - cxy[i]) ** 2).sum(axis=1))
        wc = cell_centre_of(grid, cxy[i])[0]
        gs, gl = subgrid_values(r, pxy[js] - wc, pcc[js], wc, d.T[i], d.S[i], d.w[i], cfg,
                                tabs, grid.dx)
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


def make_tables(cfg, grid: Grid) -> CellMeanTables:
    return CellMeanTables(grid.dx, cfg.r_max_m, cfg.r_min_m)


# ----------------------------------------------------------------------------- wells
def fresh_well_table(grid: Grid, cal_sids) -> tuple[pd.DataFrame, list]:
    """Candidate held-out wells: every station of heads/L1_clean/v3 station_meta_preorigin that
    is not one of the 158 calibration wells, with coordinates in an active model cell and a
    layer code 1-4 that is not a proposal (layer_flag < 5). Model layer = layer_code - 1 (the
    rule that maps the 158 wells, checked here). Reads station metadata only."""
    p = guard.open_product(L.HEADS, "selection")    # only the pre-origin station table opens
    meta = pd.read_parquet(p.path("station_meta_preorigin.parquet"),
                           columns=["sid", "x_twd97", "y_twd97", "layer_code", "layer_flag",
                                    "xy_flag", "CountyName"])
    meta["sid"] = meta["sid"].astype(str)
    cal = set(map(str, cal_sids))
    m = meta[~meta["sid"].isin(cal)].copy()
    lc = pd.to_numeric(m["layer_code"], errors="coerce")
    xy = m[["x_twd97", "y_twd97"]].to_numpy(float)
    m["cell"] = active_cell_of(grid, xy)
    m["layer"] = lc - 1
    reasons = {
        "not_calibration_well": int(len(m)),
        "no_active_cell": int((m["cell"] < 0).sum()),
        "layer_missing": int(lc.isna().sum()),
        "layer_proposal_flag5": int((pd.to_numeric(m["layer_flag"], errors="coerce") >= 5)
                                    .sum())}
    ok = (m["cell"] >= 0) & lc.between(1, 4) & (pd.to_numeric(m["layer_flag"],
                                                              errors="coerce") < 5)
    m = m[ok].copy()
    m["layer"] = m["layer"].astype(int)
    m["county"] = np.where(m["CountyName"].astype(str) == "彰化縣", "CH",
                           np.where(m["CountyName"].astype(str) == "雲林縣", "YL", "OT"))
    return m.sort_values("sid").reset_index(drop=True), [reasons, list(p.notes)]


def check_cal_mapping(grid: Grid, d: L.Data) -> dict:
    """The 158 calibration wells: grid cell of the coordinates == engine obs_idx, and layer
    code - 1 == obs_layer (the rule applied to the held-out wells)."""
    p = guard.open_product(L.HEADS, "selection")
    meta = pd.read_parquet(p.path("station_meta_preorigin.parquet"),
                           columns=["sid", "layer_code"])
    meta["sid"] = meta["sid"].astype(str)
    lc = meta.set_index("sid").loc[d.sids, "layer_code"].astype(int).to_numpy()
    cell = active_cell_of(grid, d.xy)
    return {"cell_matches": int((cell == d.cell).sum()), "layer_matches":
            int((lc - 1 == d.layer).sum()), "n": int(len(d.sids))}


def fresh_data(cal: L.Data, ft: pd.DataFrame, with_heads: bool) -> L.Data:
    """A Data object for the held-out wells that shares the pumps (Qbar, qs) of ``cal`` and
    reads the hindcast, T and S at each well's own cell and layer."""
    e = L.load_engine()
    cfg = cal.cfg
    sids = ft["sid"].to_numpy(str)
    cell, lay = ft["cell"].to_numpy(int), ft["layer"].to_numpy(int)
    h_cell = e["H_hindcast"][lay, cell, :]
    T = e["T"][lay, cell].astype(float)
    S = e["S"][lay, cell].astype(float)
    at_bound = np.isclose(T, L.T_BOUND)
    if cfg.t_bound_scale != 1.0:
        T = np.where(at_bound, T * cfg.t_bound_scale, T)
    f_sh = cal.engine_scalars["f_shallow"]
    if cfg.layer_weights == "twin_point_split":
        w = np.where(lay == 0, f_sh, np.where(lay == L.PUMP_LAYER, 1.0 - f_sh, 0.0))
    else:
        w = np.ones(len(lay))
    if with_heads:
        heads, _ = L.load_heads(sids, cal.dates, cfg)
        obs = heads["calibration_target"]
    else:
        obs = np.full((len(sids), len(cal.dates)), np.nan)
    return replace(cal, sids=sids, county=ft["county"].to_numpy(str), layer=lay, cell=cell,
                   xy=ft[["x_twd97", "y_twd97"]].to_numpy(float), h_cell=h_cell, obs=obs,
                   T=T, S=S, w=w, t_at_bound=at_bound, folds=[])


def quartiles_fixed(d: L.Data, m: dict, cuts) -> tuple[np.ndarray, np.ndarray]:
    """Quartile 1-4 by the observed fit-window sd against FIXED cuts (0 = not classified)."""
    import warnings
    o = np.where(m["fit"], d.obs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(o, axis=1, ddof=1)
    ok = m["fit_ok"] & np.isfinite(sd)
    q = np.zeros(len(sd), int)
    q[ok] = 1 + np.searchsorted(np.asarray(cuts, float), sd[ok], side="right")
    return q, sd


def distant_centres(d: L.Data, pump_xy_all: np.ndarray, rng, support: np.ndarray, cfg
                    ) -> np.ndarray:
    """h4lib.distant_centres with an explicit pump set (all census pumps of the operator)."""
    lim = cfg.distant_km * 1000.0 + cfg.r_max_m
    cen = np.full_like(d.xy, np.nan)
    for i in np.flatnonzero(support):
        dist = np.sqrt(((pump_xy_all - d.xy[i]) ** 2).sum(axis=1))
        cand = np.flatnonzero(dist >= lim)
        cen[i] = pump_xy_all[cand[rng.integers(cand.size)]]
    return cen
