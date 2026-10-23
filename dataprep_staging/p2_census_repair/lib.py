"""WP P2a-c (Yunlin census repair): shared helpers.

Ports of the billing scout's pilot code (``results/dataprep/_pilots/billing_and_census_gaps``:
``s01_build.py``, ``s02_billing.py``, ``s03_codes_by_class.py``, ``disagg_lib.py``,
``s11_chowlin_production.py``) generalised so that every window (profile, Chow-Lin, phase
detection, duty cap) is an explicit argument. That is what the pre-origin variant needs
(plan U3: p(t), beta and rho fitted on data to 2022-12 only).

Conventions (shared with P2d-f, see the WP report):

* Month axis ``DATES``: 2007-01 .. 2025-07 (223 months, index 0 = 2007-01). Phase A pairs
  start at odd calendar months (Jan-Feb, Mar-Apr, ...), i.e. at even indices; phase B pairs
  start at even calendar months (Feb-Mar, ..., Dec-Jan), i.e. at odd indices.
* Meter id = ``pd.factorize(str(電號1))`` in census order (identical to the pilot's
  ``meter_code``); the raw meter number is never written. A meter's series is its first
  pump's copy (the copies are duplicates), as in ``pumping.clean_census``.
* Duty cap and kept set: the twin's frozen convention, ``clean_census(cap_duty=1.0,
  t0=2012-01-01, t1=2023-01-01)`` (mean monthly kWh over 2012-01..2022-12 above 1.0 x the
  meter's summed HP x 0.746 kW x 730 h drops the meter). The same kept set is used by both
  variants, as the forcing experiment does (``forcing_experiment.observed_energy``).
* Pump energy = meter energy x HP share (even split when the meter's HP sum is 0); each pump
  is aggregated into its own active cell (``pumping._cell_lookup``) and its own purpose class
  (``scenario.purpose_class``), so the recorded (C, A, T) arrays reproduce
  ``scenario.energy_by_class`` exactly.

Privacy: the census is read with an explicit column list that never includes 戶名1 or
用電地址2 (plan 2.6).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "dataprep_staging") not in sys.path:
    sys.path.insert(0, str(REPO / "dataprep_staging"))

CENSUS = REPO / "AMP_V2" / "data" / "tpc_pumps.parquet"
KWH = REPO / "AMP_V2" / "data" / "pump_kwh_all.parquet"
STATIONS = REPO / "AMP_V2" / "data" / "fan_stations.parquet"
WELLS_DIR = REPO / "AMP_V2" / "data" / "wells"
RAIN = REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / "rf_timeseries_codis.csv"
RAIN_STATIONS = REPO / "chou-shui-data" / "data" / "rf_stations.csv"
GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
PILOT = REPO / "results" / "dataprep" / "_pilots" / "billing_and_census_gaps"
PRODUCT_ROOT = REPO / "results" / "dataprep" / "yunlin_electricity"
WORK = PRODUCT_ROOT / "_work_p2abc"          # intermediates (not a product; gitignored)
CODE_DIR = Path(__file__).resolve().parent

#: Columns read from the census. 戶名1 (a name) and 用電地址2 (an address) are never read.
CENSUS_COLS = ["sid", "TOWN", "PURPOSE", "PUMP_HP", "W_TUBE_DIA", "每期用電度數1", "新設日期1",
               "電號1", "TWD97_X", "TWD97_Y"]
FORBIDDEN_COLS = ("戶名1", "用電地址2")

DATES = pd.date_range("2007-01-01", "2025-07-01", freq="MS")
T_ALL = len(DATES)                    # 223
KW_PER_HP = 0.746
HOURS_PER_MONTH = 730.0
CAP_DUTY = 1.0
DUTY_WINDOW = ("2012-01", "2022-12")  # the twin's frozen meter selection (inclusive months)
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")
PHASE_A, PHASE_B, MONTHLY = "A(odd-start)", "B(even-start)", "monthly"

#: Variant definitions. ``end`` = last month of data any fitted parameter sees; ``out0`` =
#: first output month; ``cl0`` = first month of the Chow-Lin window (12-month lead-in, as in
#: the pilot's s11); ``phase_window`` = window of the pair statistics of the phase detection.
VARIANTS = {
    "preorigin": {"end": "2022-12", "out0": "2012-01", "cl0": "2011-01",
                  "phase_window": ("2012-01", "2022-12")},
    "full": {"end": "2025-07", "out0": "2012-01", "cl0": "2011-01",
             "phase_window": ("2012-01", "2025-06")},
}


def tidx(month: str) -> int:
    """Index of a 'YYYY-MM' month on ``DATES``."""
    return int(DATES.get_loc(pd.Timestamp(month + "-01")))


def ym(i: int) -> str:
    return DATES[i].strftime("%Y-%m")


# ----------------------------------------------------------------------------- census
def read_census() -> pd.DataFrame:
    """The census with the explicit, privacy-safe column list (plan 2.6)."""
    assert not set(CENSUS_COLS) & set(FORBIDDEN_COLS)
    return pd.read_parquet(CENSUS, columns=CENSUS_COLS)


def roc_to_ts(v) -> pd.Timestamp:
    """ROC date integer (yyymmdd) -> Timestamp, exactly as the pilot's s01_build.py."""
    if pd.isna(v):
        return pd.NaT
    v = int(v)
    y, md = divmod(v, 10000)
    m, d = divmod(md, 100)
    try:
        return pd.Timestamp(year=y + 1911, month=max(m, 1), day=max(d, 1))
    except Exception:  # noqa: BLE001 - invalid dates become NaT, as in the pilot
        return pd.NaT


def fan_grid():
    """The twin's FanGrid from ``datum_gate_innov.npz`` (nx, ny, dx, x0, y0, mask)."""
    from hydrophysics.twin.grid import FanGrid
    g = np.load(GRID_NPZ, allow_pickle=False)
    return FanGrid(nx=int(g["nx"]), ny=int(g["ny"]), dx=float(g["dx"]), x0=float(g["x0"]),
                   y0=float(g["y0"]), mask=g["mask"].astype(bool))


def grid_meta(grid) -> dict:
    from common.hashing import grid_hash, mask_hash
    return {"nx": grid.nx, "ny": grid.ny, "dx": grid.dx, "x0": grid.x0, "y0": grid.y0,
            "n_active": grid.n_active, "mask_hash": mask_hash(grid.mask),
            "grid_hash": grid_hash(grid.nx, grid.ny, grid.dx, grid.x0, grid.y0, grid.mask),
            "source": str(GRID_NPZ.relative_to(REPO)),
            "active_order": "row-major over masked cells (pumping._cell_lookup)"}


# ----------------------------------------------------------------------------- phases
def pair_stats(X: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Identical-pair counts (pilot s02): pairs (start, start+1), (start+2, ...)."""
    a = X[:, start:-1:2]
    b = X[:, start + 1::2]
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    valid = np.isfinite(a) & np.isfinite(b) & ((a > 0) | (b > 0))
    eq = valid & np.isclose(a, b, rtol=0, atol=0.51)
    return eq.sum(1), valid.sum(1)


def classify_phase(fA, fB, nA, nB) -> np.ndarray:
    """Pilot s02 ``phase`` rule, vectorised."""
    fA = np.asarray(fA, float)
    fB = np.asarray(fB, float)
    hi, lo = np.fmax(fA, fB), np.fmin(fA, fB)
    out = np.full(len(fA), "mixed", dtype=object)
    out[lo >= 0.8] = "both-flat"
    out[hi < 0.3] = MONTHLY
    ab = (hi >= 0.8) & (lo <= 0.3)
    out[ab & (fA > fB)] = PHASE_A
    out[ab & ~(fA > fB)] = PHASE_B
    out[(np.asarray(nA) < 6) | (np.asarray(nB) < 6)] = "insufficient"
    return out


def detect_phase(M: np.ndarray, window: tuple[str, str]) -> pd.DataFrame:
    """Billing phase per meter from the identical-pair fractions over ``window``."""
    t0, t1 = tidx(window[0]), tidx(window[1])
    X = M[:, t0:t1 + 1]
    sA = 0 if DATES[t0].month % 2 == 1 else 1
    eqA, nA = pair_stats(X, sA)
    eqB, nB = pair_stats(X, 1 - sA)
    fA = np.where(nA > 0, eqA / np.maximum(nA, 1), np.nan)
    fB = np.where(nB > 0, eqB / np.maximum(nB, 1), np.nan)
    return pd.DataFrame({"fA": fA, "fB": fB, "nA": nA, "nB": nB,
                         "phase": classify_phase(fA, fB, nA, nB)})


# ----------------------------------------------------------------------------- pairs
def pair_starts(phase: str, t_lo: int, t_hi: int) -> np.ndarray:
    """Starts of the phase's pairs lying entirely inside [t_lo, t_hi) on ``DATES``."""
    s = 0 if phase == "A" else 1
    st = np.arange(s, T_ALL - 1, 2)
    return st[(st >= t_lo) & (st + 1 < t_hi)]


def even_split(x: np.ndarray, phase: str, t_lo: int = 0, t_hi: int = T_ALL) -> np.ndarray:
    st = pair_starts(phase, t_lo, t_hi)
    s = x[..., st] + x[..., st + 1]
    out = np.full(x.shape, np.nan)
    out[..., st] = s / 2
    out[..., st + 1] = s / 2
    return out


def prorata(x: np.ndarray, phase: str, ind: np.ndarray, t_lo: int = 0, t_hi: int = T_ALL,
            floor: float = 1e-6) -> np.ndarray:
    st = pair_starts(phase, t_lo, t_hi)
    s = x[..., st] + x[..., st + 1]
    ind = np.broadcast_to(ind, x.shape)
    i1 = np.maximum(ind[..., st], floor)
    i2 = np.maximum(ind[..., st + 1], floor)
    w = i1 / (i1 + i2)
    out = np.full(x.shape, np.nan)
    out[..., st] = s * w
    out[..., st + 1] = s * (1 - w)
    return out


def stagger_deconvolve(a_month: np.ndarray, b_month: np.ndarray, t_end: int,
                       lam: float = 1e-2, win: int = 12, share_a: np.ndarray | None = None,
                       share_b: np.ndarray | None = None) -> np.ndarray:
    """Fan-wide monthly profile from the staggered phase-A and phase-B population totals
    (pilot ``disagg_lib.stagger_deconvolve``), using only months < ``t_end``:

        min_u sum_A (r_t u_t + r_t+1 u_t+1 - Y_A)^2 + sum_B ((1-r) ... - Y_B)^2 + lam ||D2 u||^2

    with the phase-A share r_t a rolling ``win``-month ratio of the recorded energies and both
    terms scaled by the mean pair sum (lam is relative to the data scale). Only pairs that lie
    entirely before ``t_end`` enter, so the pre-origin profile (t_end = index of 2023-01) never
    sees a post-origin bill. ``share_a``/``share_b`` optionally replace the series the rolling
    share is computed from (the full variant passes phase A's July 2025, whose bill is missing,
    as June's value there). Returns u on months [0, t_end).
    """
    T = int(t_end)
    a = np.nan_to_num(np.asarray(a_month, float)[:T])
    b = np.nan_to_num(np.asarray(b_month, float)[:T])
    sa = a if share_a is None else np.nan_to_num(np.asarray(share_a, float)[:T])
    sb = b if share_b is None else np.nan_to_num(np.asarray(share_b, float)[:T])
    k = np.ones(win) / win
    ra = np.convolve(sa, k, mode="same")
    rb = np.convolve(sb, k, mode="same")
    rA = np.where(ra + rb > 0, ra / np.maximum(ra + rb, 1e-12), 0.5)
    stA = np.arange(0, T - 1, 2)
    stB = np.arange(1, T - 1, 2)
    rows = len(stA) + len(stB)
    A = np.zeros((rows, T))
    rhs = np.zeros(rows)
    for i, t in enumerate(stA):
        A[i, t], A[i, t + 1] = rA[t], rA[t + 1]
        rhs[i] = a[t] + a[t + 1]
    for j, t in enumerate(stB):
        i = len(stA) + j
        A[i, t], A[i, t + 1] = 1 - rA[t], 1 - rA[t + 1]
        rhs[i] = b[t] + b[t + 1]
    scale = max(np.mean(np.abs(rhs)), 1e-9)
    D2 = np.diff(np.eye(T), n=2, axis=0)
    ridge = 1e-6 * np.eye(T)
    lhs = A.T @ A / scale**2 + lam * D2.T @ D2 / scale**2 + ridge / scale**2
    r_ = A.T @ rhs / scale**2 + ridge @ np.full(T, np.mean(rhs) / 2) / scale**2
    return np.linalg.solve(lhs, r_)


def year_shuffle(prof: np.ndarray, seed: int, t0: int = 0) -> np.ndarray:
    """Placebo profile: whole calendar years permuted (a derangement), so each year keeps its
    seasonal shape but lands in the wrong year. Months before ``t0`` and a trailing partial
    year are left in place."""
    rng = np.random.default_rng(seed)
    n_years = (len(prof) - t0) // 12
    perm = np.arange(n_years)
    while np.any(perm == np.arange(n_years)):
        perm = rng.permutation(n_years)
    out = prof.copy()
    for i, j in enumerate(perm):
        out[t0 + 12 * i: t0 + 12 * (i + 1)] = prof[t0 + 12 * j: t0 + 12 * (j + 1)]
    return out


# ----------------------------------------------------------------------------- Chow-Lin
class ChowLin:
    """Chow-Lin disaggregation of pair sums for one phase over the window [w_lo, w_hi).

    Model x = X beta + u, X = [1, p_t / mean(p)], u AR(1) with rho chosen per meter by
    likelihood from ``rhos``; estimate x_hat = X beta_hat + V C' (C V C')^-1 (Y - C X beta_hat)
    with beta_hat the GLS estimate (plan P2a). Pairs are those lying entirely inside the
    window; months of the window not covered by a pair (the phase-B month before the first
    pair, or after the last one, e.g. December 2022 in the pre-origin variant) receive the
    Chow-Lin extrapolation X beta_hat + V C' Vai res (flag 5 downstream).

    With one common indicator this is a fixed linear operator per rho, so all meters of a
    phase are processed as one matrix product (pilot s11).
    """

    def __init__(self, phase: str, prof: np.ndarray, w_lo: int, w_hi: int,
                 rhos=(0.0, 0.5, 0.9)):
        self.phase, self.w_lo, self.w_hi = phase, int(w_lo), int(w_hi)
        st = pair_starts(phase, w_lo, w_hi)
        self.st = st
        n, K = self.w_hi - self.w_lo, len(st)
        self.n, self.K = n, K
        C = np.zeros((K, n))
        C[np.arange(K), st - w_lo] = 1
        C[np.arange(K), st - w_lo + 1] = 1
        p = np.asarray(prof, float)[w_lo:w_hi]
        X = np.column_stack([np.ones(n), p / p.mean()])
        Xa = C @ X
        idx = np.arange(n)
        self.ops = []
        for rho in rhos:
            V = rho ** np.abs(idx[:, None] - idx[None, :]) / max(1 - rho**2, 1e-6)
            Va = C @ V @ C.T
            Vai = np.linalg.pinv(Va)
            B = np.linalg.solve(Xa.T @ Vai @ Xa, Xa.T @ Vai)       # beta = B @ S
            R = np.eye(K) - Xa @ B                                  # residual = R @ S
            Lop = X @ B + V @ C.T @ Vai @ R                         # x_hat = Lop @ S
            _, logdet = np.linalg.slogdet(Va)
            self.ops.append((float(rho), Lop, R, Vai, float(logdet)))

    def pair_sums(self, Xm: np.ndarray) -> np.ndarray:
        """(K, N) pair sums of (N, T_ALL) monthly records (NaN treated as 0, as the pilot)."""
        Z = np.nan_to_num(Xm)
        return (Z[:, self.st] + Z[:, self.st + 1]).T

    def run(self, Xm: np.ndarray, return_rho: bool = False, project: bool = True):
        """(N, n) disaggregated months [w_lo, w_hi) for the records Xm (N, T_ALL).
        ``project=False`` skips the [0, Y] pair projection (pilot behaviour, for the port
        check only)."""
        Sp = self.pair_sums(Xm)
        N = Sp.shape[1]
        best = np.full(N, -np.inf)
        out = np.zeros((N, self.n))
        rho_sel = np.full(N, np.nan)
        for rho, Lop, R, Vai, logdet in self.ops:
            res = R @ Sp
            sig2 = np.einsum("kn,kl,ln->n", res, Vai, res) / self.K
            ll = -0.5 * (self.K * np.log(np.maximum(sig2, 1e-12)) + logdet)
            better = ll > best
            out[better] = (Lop @ Sp[:, better]).T
            best[better] = ll[better]
            rho_sel[better] = rho
        if project:
            out = self.project(out, Sp)
        return (out, rho_sel) if return_rho else out

    def project(self, out: np.ndarray, Sp: np.ndarray) -> np.ndarray:
        """Clip each pair's split to [0, Y] keeping the pair sum exactly (energy >= 0); a
        zero pair stays zero. Unpaired months are floored at 0."""
        o = out.copy()
        a = self.st - self.w_lo
        Y = Sp.T                                   # (N, K)
        x1 = np.clip(o[:, a], 0.0, np.maximum(Y, 0.0))
        o[:, a] = x1
        o[:, a + 1] = Y - x1
        paired = np.zeros(self.n, bool)
        paired[a] = paired[a + 1] = True
        o[:, ~paired] = np.maximum(o[:, ~paired], 0.0)
        return o


# ----------------------------------------------------------------------------- masks
def open_edge_group(bill: np.ndarray, phase: np.ndarray, end: int) -> np.ndarray:
    """Meters whose month ``end`` is the first half of a bill not yet issued at ``end``
    (plan P2b: a month whose billing pair is incomplete is NaN for that phase, never zero).

    ``end`` at an even index (odd calendar month, e.g. July 2025) opens a phase-A pair: the
    phase-A meters plus M2-coded meters that are neither phase B nor monthly. ``end`` at an
    odd index (even calendar month, e.g. December 2022) opens a phase-B pair: phase-B meters
    plus M1-coded meters that are neither phase A nor monthly."""
    if end % 2 == 0:
        return (phase == PHASE_A) | ((bill == "M2") & ~np.isin(phase, [PHASE_B, MONTHLY]))
    return (phase == PHASE_B) | ((bill == "M1") & ~np.isin(phase, [PHASE_A, MONTHLY]))


def incomplete_pairs(M: np.ndarray, phase: np.ndarray, t_lo: int, t_hi: int) -> np.ndarray:
    """Boolean (n_meters, T_ALL): months of phase-A/B meters whose pair (inside
    [t_lo, t_hi)) has exactly one finite month; both months are flagged."""
    out = np.zeros(M.shape, bool)
    for p_, lab in (("A", PHASE_A), ("B", PHASE_B)):
        rows = np.where(phase == lab)[0]
        st = pair_starts(p_, t_lo, t_hi)
        a, b = np.isfinite(M[np.ix_(rows, st)]), np.isfinite(M[np.ix_(rows, st + 1)])
        bad = a ^ b
        r, k = np.nonzero(bad)
        out[rows[r], st[k]] = True
        out[rows[r], st[k] + 1] = True
    return out


def masked_matrix(M: np.ndarray, bill: np.ndarray, phase: np.ndarray, end: int):
    """(Mv, edge, inc): float64 copy of M[:, :end+1] with the open edge month of the open-edge
    group and both months of every incomplete A/B pair set to NaN, plus the two masks."""
    Mv = M[:, :end + 1].astype(float)
    edge = open_edge_group(bill, phase, end)
    inc = incomplete_pairs(M, phase, 0, end + 1)[:, :end + 1]
    Mv[inc] = np.nan
    Mv[edge, end] = np.nan
    return Mv, edge, inc


def spans(Mv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last month index with positive use (-1 if none) of a masked matrix."""
    nz = np.nan_to_num(Mv) > 0
    end = Mv.shape[1] - 1
    first = np.where(nz.any(1), nz.argmax(1), -1)
    last = np.where(nz.any(1), end - nz[:, ::-1].argmax(1), -1)
    return first, last


# ----------------------------------------------------------------------------- io
def save_json(obj, path: str | os.PathLike) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=default)
        f.write("\n")


def load_work():
    """(meters DataFrame, meter matrix (nm, T_ALL) float32, pumps DataFrame) from WORK."""
    mt = pd.read_parquet(WORK / "meters.parquet")
    M = np.load(WORK / "meter_mat.npy")
    pumps = pd.read_parquet(WORK / "pumps.parquet")
    return mt, M, pumps
