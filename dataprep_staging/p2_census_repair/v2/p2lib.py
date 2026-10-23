"""WP P2a-c, round 2 (C0-based, both counties): shared library.

Round 1 (``dataprep_staging/p2_census_repair/*.py``, 2026-10-12 morning) worked on the Yunlin
AMP_V2 cache only. After the Changhua census arrived, WP C0 delivered the cleaned census of
every in-grid county (``results/dataprep/pump_electricity/{census_L1,preorigin,full}/v3``).
This package rebuilds P2a-c on those products for Yunlin, Changhua and the 316 kept in-grid
Chiayi pumps. Round-1 code is kept unchanged (its pre-registrations stand).

Conventions (shared with P2d-f; also stated in the WP report):

* **Time axis.** C0's: index 0 = 2007-01. Pre-origin arrays run 2007-01..2022-12 (T = 192),
  full arrays 2007-01..2026-07 (T = 235). Phase-A bills pair (Jan, Feb), (Mar, Apr), ...:
  pairs start at even indices. Phase-B bills pair (Feb, Mar), ..., (Dec, Jan): odd indices.
* **Meters.** County-scoped ``meter_idx`` of C0 (not contiguous). A meter's monthly series is
  ``value / hp_share`` of any of its kept pumps (all pumps of a meter agree; checked). The kept
  set, classes, cells, flags and method codes are C0's (v3), never recomputed.
* **Phase used for the split** = C0's detected phase of the variant (pre-origin
  ``phase_detected_2012_2022``: identical pairs inside 2012-01..2022-11; full
  ``phase_detected``: 2012-01..2026-06). The billing code (M2 = A, M1 = B) selects the profile
  populations together with the detected phase (pilot rule).
* **Profile** (fan-wide, one per county): stagger deconvolution of the in-grid kept irrigation
  phase-A (M2 & detected A) and phase-B (M1 & detected B) population sums (lambda 0.01, 12-month
  share window), only pairs complete by the variant's end. Chiayi's 316 in-grid pumps use the
  Yunlin profile (too few meters for their own).
* **Chow-Lin** (plan P2a): x = X beta + u, X = [1, p/mean p], AR(1) u with rho per meter by
  likelihood from {0, 0.5, 0.9}, window 2011-01..variant end (pairs lying entirely inside),
  each pair's split clipped to [0, Y] keeping the pair sum. A pair is re-split only when both
  months are installed observations (flag 0, method observed / observed before the install
  date / terminated trailing run / bill in the second month / uneven pair); every other month
  keeps its C0 value, flag and method (structural pre-install zeros, missing readings, C0's
  December-2022 fill, 2026-07 masks). Re-split months with a positive pair sum get flag 3,
  method ``chowlin_split``.

Privacy: no output carries a name, address, postal code, raw meter number or coordinate; uid
(well numbers) appear in per-pump arrays only, as in C0 (gitignored, declared personal data).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[2]
for _p in (REPO, REPO / "dataprep_staging"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import guard  # noqa: E402

DATAPREP = REPO / "results" / "dataprep"
C0_DIR = {"preorigin": DATAPREP / "pump_electricity" / "preorigin" / "v3",
          "full": DATAPREP / "pump_electricity" / "full" / "v3"}
C0_L1 = DATAPREP / "pump_electricity" / "census_L1" / "v3"
PRODUCT_ROOT = DATAPREP / "yunlin_electricity"
WORK = PRODUCT_ROOT / "_work_p2abc_v2"
EVID = DATAPREP / "reports" / "P2abc_evidence"
WP = "P2abc"

COUNTIES = ("yunlin", "changhua", "chiayi")
FILETAG = {"yunlin": "yunlin", "changhua": "changhua", "chiayi": "chiayi_ingrid"}
PROFILE_OF = {"yunlin": "yunlin", "changhua": "changhua", "chiayi": "yunlin"}
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")
DATES = pd.date_range("2007-01-01", "2026-07-01", freq="MS")      # C0's time base (235)


def tidx(month: str) -> int:
    return int(DATES.get_loc(pd.Timestamp(month + "-01")))


def ym(i: int) -> str:
    return DATES[int(i)].strftime("%Y-%m")


#: C0 per pump-month method codes (pump_electricity v3) and the codes added here.
C0_METHODS = {0: "observed", 1: "pre_install", 2: "observed_before_install_date",
              3: "missing", 4: "incomplete_pair_2026_07", 5: "partial_reading_cycle_2026_07",
              6: "terminated_trailing_run", 7: "bill_in_second_month", 8: "uneven_pair",
              9: "fill_dec2022_dec_jan_pair"}
M_CHOWLIN = 10            # flag 3: Chow-Lin split of an observed bimonthly bill
M_JULY_IMPUTED = 11       # flag 4: own 3-year same-month mean (open phase-A July 2026, P2b)
METHODS = {**C0_METHODS, M_CHOWLIN: "chowlin_split", M_JULY_IMPUTED: "own_3yr_same_month_mean"}
#: methods whose months may be re-split when both months of a pair carry flag 0
RESPLIT_OK = np.array([0, 2, 6, 7, 8], dtype=np.uint8)

LAMBDA = 1e-2
SHARE_WIN = 12
RHOS = (0.0, 0.5, 0.9)
Z90 = 1.6448536269514722

VARIANTS = {
    "preorigin": {
        "T": tidx("2022-12") + 1, "end": "2022-12", "phase_col": "phase_detected_2012_2022",
        "cl0": tidx("2011-01"),
        # window upper bounds (exclusive): phase A pairs to (Nov, Dec 2022); phase B pairs to
        # (Oct, Nov 2022) - the (Dec 2022, Jan 2023) bill is open; its December keeps C0's
        # flag-4 fill.
        "win_hi": {"A": tidx("2022-12") + 1, "B": tidx("2022-11") + 1},
        "open": {"B": tidx("2022-12")}, "use": "forcing_frozen",
    },
    "full": {
        "T": tidx("2026-07") + 1, "end": "2026-07", "phase_col": "phase_detected",
        "cl0": tidx("2011-01"),
        # phase A pairs to (May, Jun 2026) - the (Jul, Aug 2026) bill is open (C0: NaN, flag
        # 8); phase B pairs to (Jun, Jul 2026).
        "win_hi": {"A": tidx("2026-06") + 1, "B": tidx("2026-07") + 1},
        "open": {"A": tidx("2026-07")}, "use": "forcing_recorded_arm",
    },
}


# ----------------------------------------------------------------------------- io
def save_json(obj, path) -> None:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=default)
        f.write("\n")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rel(p) -> str:
    return os.path.relpath(Path(p).resolve(), REPO)


# ----------------------------------------------------------------------------- C0 inputs
def open_c0(variant: str):
    """C0 product opened through the guard with the use this WP makes of it."""
    return guard.open_product(C0_DIR[variant], VARIANTS[variant]["use"])


def open_l1(use: str = "validator"):
    return guard.open_product(C0_L1, use)


def l1_pumps(cols=("uid", "county", "x_twd97", "y_twd97", "cell", "meter_idx",
                   "meter_first_pump", "kept", "in_grid")) -> pd.DataFrame:
    """Selected census_L1 columns (never a personal column other than uid/coordinates,
    which stay in memory)."""
    p = open_l1()
    return pd.read_parquet(p.path("census_pumps.parquet"), columns=list(cols))


class CountyData:
    """One county's C0 arrays for one variant, at pump and meter level."""

    def __init__(self, variant: str, county: str, prod=None, l1: pd.DataFrame | None = None):
        prod = prod or open_c0(variant)
        tag = FILETAG[county]
        z = np.load(prod.path(f"pumps_{tag}.npz"), allow_pickle=False)
        self.variant, self.county = variant, county
        self.T = int(z["value"].shape[1])
        assert VARIANTS[variant]["T"] == self.T, (self.T, variant)
        self.uid = z["uid"]
        self.cell = z["cell"].astype(np.int64)
        self.pclass = z["purpose_class"]
        self.meter_idx = z["meter_idx"].astype(np.int64)
        self.hp = z["hp"]
        self.share = z["hp_share"]
        self.pvalue = z["value"]
        self.pflag = z["flag"]
        self.pmethod = z["method"]
        self.psd = z["sd"] if "sd" in z.files else np.full(self.pvalue.shape, np.nan)
        mt = pd.read_parquet(prod.path(f"meters_{tag}.parquet"))
        mt = mt.sort_values("meter_idx").reset_index(drop=True)
        self.meters = mt
        self.mpos = pd.Series(np.arange(len(mt)), index=mt["meter_idx"].to_numpy())
        self.prow = self.mpos.reindex(self.meter_idx).to_numpy()
        assert not np.isnan(self.prow.astype(float)).any()
        self.prow = self.prow.astype(np.int64)
        # meter series from the array pump with the largest HP share of each meter (value /
        # share); a pump with zero HP on a meter with positive HP has share 0 and value 0
        order = np.lexsort((np.arange(len(self.uid)), -self.share, self.prow))
        fp = pd.Series(order).groupby(self.prow[order]).first().to_numpy()
        assert len(fp) == len(mt) and (self.share[fp] > 0).all()
        self.first_array_pump = fp
        self.M = self.pvalue[fp] / self.share[fp, None]
        self.F = self.pflag[fp].copy()
        self.Mm = self.pmethod[fp].copy()
        self.Msd = self.psd[fp] / self.share[fp, None]
        # consistency: every pump equals share x its meter series (NaN patterns, flags, methods)
        rec = self.M[self.prow] * self.share[:, None]
        both_nan = np.isnan(rec) & np.isnan(self.pvalue)
        diff = np.where(both_nan, 0.0, np.abs(rec - self.pvalue))
        self.check = {
            "max_abs_pump_vs_meter_kwh": float(np.nanmax(diff)) if diff.size else 0.0,
            "nan_pattern_mismatch": int((np.isnan(rec) ^ np.isnan(self.pvalue)).sum()),
            "flag_mismatch": int((self.F[self.prow] != self.pflag).sum()),
            "method_mismatch": int((self.Mm[self.prow] != self.pmethod).sum()),
            "zero_share_pumps": int((self.share == 0).sum()),
        }
        if l1 is not None:
            self.attach_l1(l1)

    def attach_l1(self, l1: pd.DataFrame) -> None:
        """Pump coordinates and the meter's first pump (census_L1), kept in memory only."""
        sub = l1[l1["county"] == self.county].set_index("uid")
        s = sub.reindex(self.uid)
        self.x = s["x_twd97"].to_numpy(dtype=float)
        self.y = s["y_twd97"].to_numpy(dtype=float)
        # first pump of the meter in the census (may lie outside the array for Chiayi meters
        # that share a meter with an out-of-grid pump): its cell and coordinates
        fp = sub[sub["meter_first_pump"].astype(bool)].drop_duplicates("meter_idx")
        fp = fp.set_index("meter_idx").reindex(self.meters["meter_idx"].to_numpy())
        self.m_x = fp["x_twd97"].to_numpy(dtype=float)
        self.m_y = fp["y_twd97"].to_numpy(dtype=float)
        self.m_cell = fp["cell"].fillna(-1).to_numpy().astype(np.int64)
        self.m_first_in_grid = self.m_cell >= 0

    # convenience
    @property
    def phase(self) -> np.ndarray:
        return self.meters[VARIANTS[self.variant]["phase_col"]].astype(str).to_numpy()

    @property
    def code(self) -> np.ndarray:
        return self.meters["billing_code"].astype(str).to_numpy()

    @property
    def mclass(self) -> np.ndarray:
        return self.meters["class_first_pump"].astype(str).to_numpy()


# ----------------------------------------------------------------------------- pairs
def pair_starts(phase: str, t_lo: int, t_hi: int, T: int) -> np.ndarray:
    """Starts of the phase's pairs lying entirely inside [t_lo, t_hi) on a T-month axis."""
    s = 0 if phase == "A" else 1
    st = np.arange(s, T - 1, 2)
    return st[(st >= t_lo) & (st + 1 < t_hi)]


def even_split(x: np.ndarray, phase: str, t_lo: int = 0, t_hi: int | None = None) -> np.ndarray:
    T = x.shape[-1]
    t_hi = T if t_hi is None else t_hi
    st = pair_starts(phase, t_lo, t_hi, T)
    s = x[..., st] + x[..., st + 1]
    out = np.full(x.shape, np.nan)
    out[..., st] = s / 2
    out[..., st + 1] = s / 2
    return out


def stagger_deconvolve(a_month: np.ndarray, b_month: np.ndarray, t_end: int,
                       lam: float = LAMBDA, win: int = SHARE_WIN,
                       share_a: np.ndarray | None = None,
                       share_b: np.ndarray | None = None) -> np.ndarray:
    """Fan-wide monthly profile from the staggered phase-A and phase-B population sums (port of
    the billing pilot's ``disagg_lib.stagger_deconvolve`` with an explicit end; identical to
    round 1's ``lib.stagger_deconvolve``). Only pairs lying before ``t_end`` enter."""
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
    """Placebo profile: whole calendar years permuted by a derangement (seeded), so each year
    keeps its seasonal shape but lands in another year; a trailing partial year stays."""
    rng = np.random.default_rng(seed)
    n_years = (len(prof) - t0) // 12
    perm = np.arange(n_years)
    while np.any(perm == np.arange(n_years)):
        perm = rng.permutation(n_years)
    out = prof.copy()
    for i, j in enumerate(perm):
        out[t0 + 12 * i: t0 + 12 * (i + 1)] = prof[t0 + 12 * j: t0 + 12 * (j + 1)]
    return out


class ChowLin:
    """Chow-Lin disaggregation of one phase's pair sums over the window [w_lo, w_hi) of a
    T-month axis (round 1's ``lib.ChowLin`` with T explicit). ``run`` returns the window's
    months; the caller decides which months to keep (pairs only, see :func:`apply_chowlin`)."""

    def __init__(self, phase: str, prof: np.ndarray, w_lo: int, w_hi: int, T: int,
                 rhos=RHOS):
        self.phase, self.w_lo, self.w_hi, self.T = phase, int(w_lo), int(w_hi), int(T)
        st = pair_starts(phase, w_lo, w_hi, T)
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
            B = np.linalg.solve(Xa.T @ Vai @ Xa, Xa.T @ Vai)
            R = np.eye(K) - Xa @ B
            Lop = X @ B + V @ C.T @ Vai @ R
            _, logdet = np.linalg.slogdet(Va)
            self.ops.append((float(rho), Lop, R, Vai, float(logdet)))

    def pair_sums(self, Xm: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num(Xm)
        return (Z[:, self.st] + Z[:, self.st + 1]).T

    def run(self, Xm: np.ndarray, return_rho: bool = False):
        """(N, n) disaggregated window months of records Xm (N, T); NaN counts as 0 in the
        pair sums (pilot convention); each pair clipped to [0, Y] keeping its sum."""
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
        a = self.st - self.w_lo
        Y = Sp.T
        x1 = np.clip(out[:, a], 0.0, np.maximum(Y, 0.0))
        out[:, a] = x1
        out[:, a + 1] = Y - x1
        return (out, rho_sel) if return_rho else out


def apply_chowlin(M: np.ndarray, F: np.ndarray, Mm: np.ndarray, rows: np.ndarray, phase: str,
                  prof: np.ndarray, w_lo: int, w_hi: int, batch: int = 20000):
    """Chow-Lin split of the pairs of meters ``rows`` (one phase). Returns (values (len(rows),
    T) with re-split pairs replaced, resplit mask (len(rows), T), rho per meter). Only pairs
    whose two months both carry flag 0 and a re-splittable method are replaced."""
    T = M.shape[1]
    op = ChowLin(phase, prof, w_lo, w_hi, T)
    out = M[rows].astype(np.float64).copy()
    mask = np.zeros(out.shape, bool)
    rho = np.full(len(rows), np.nan)
    st = op.st
    for k in range(0, len(rows), batch):
        r = rows[k:k + batch]
        est, rr = op.run(M[r].astype(np.float64), return_rho=True)
        rho[k:k + batch] = rr
        ok = ((F[np.ix_(r, st)] == 0) & (F[np.ix_(r, st + 1)] == 0)
              & np.isin(Mm[np.ix_(r, st)], RESPLIT_OK) & np.isin(Mm[np.ix_(r, st + 1)], RESPLIT_OK)
              & np.isfinite(M[np.ix_(r, st)]) & np.isfinite(M[np.ix_(r, st + 1)]))
        a = st - w_lo
        sub = out[k:k + batch]
        msub = mask[k:k + batch]
        for j in range(len(st)):
            o = ok[:, j]
            if o.any():
                sub[o, st[j]] = est[o, a[j]]
                sub[o, st[j] + 1] = est[o, a[j] + 1]
                msub[o, st[j]] = True
                msub[o, st[j] + 1] = True
    return out, mask, rho, op


def county_profile(cd: CountyData, in_grid: np.ndarray | None = None) -> dict:
    """The variant's fan-wide profile of one county (see module docstring)."""
    v = VARIANTS[cd.variant]
    T = cd.T
    ing = cd.m_first_in_grid if in_grid is None else in_grid
    irr = cd.mclass == "irrigation"
    ph, code = cd.phase, cd.code
    selA = irr & ing & (code == "M2") & (ph == "A")
    selB = irr & ing & (code == "M1") & (ph == "B")
    Z = np.nan_to_num(cd.M)
    a, b = Z[selA].sum(0), Z[selB].sum(0)
    sa, sb = a.copy(), b.copy()
    for p_, t in v["open"].items():          # the open month never enters the share
        if p_ == "A":
            sa[t] = sa[t - 1]
        else:
            sb[t] = sb[t - 1]
    u = stagger_deconvolve(a, b, t_end=T, share_a=sa, share_b=sb)
    return {"prof": u, "a_month": a, "b_month": b, "n_A": int(selA.sum()),
            "n_B": int(selB.sum()), "t_end": T}


def chowlin_rows(cd: CountyData) -> dict:
    """Meter rows of the Chow-Lin set per phase: kept irrigation meters (meter class) whose
    detected phase of the variant is A or B."""
    irr = cd.mclass == "irrigation"
    ph = cd.phase
    return {p_: np.where(irr & (ph == p_))[0] for p_ in ("A", "B")}


def env_info() -> dict:
    import platform
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def detect_phase(X: np.ndarray, t_lo: int, t_hi: int, atol: float = 0.51) -> np.ndarray:
    """Billing phase from identical-pair fractions over months [t_lo, t_hi) (the billing
    pilot's rule, as C0's ``lib.detect_phase``): a phase needs >= 0.8 identical pairs while
    the other has <= 0.3; both < 0.3 monthly; both >= 0.8 'both-flat'; < 6 valid pairs
    'insufficient'; else 'mixed'."""
    T = X.shape[1]
    out = {}
    for ph in ("A", "B"):
        st = pair_starts(ph, t_lo, t_hi, T)
        a, b = X[:, st], X[:, st + 1]
        valid = np.isfinite(a) & np.isfinite(b) & ((a > 0) | (b > 0))
        eq = valid & np.isclose(a, b, rtol=0, atol=atol)
        out[f"n{ph}"] = valid.sum(1)
        out[f"f{ph}"] = np.where(valid.sum(1) > 0, eq.sum(1) / np.maximum(valid.sum(1), 1),
                                 np.nan)
    fA, fB = out["fA"], out["fB"]
    hi, lo = np.fmax(fA, fB), np.fmin(fA, fB)
    ph = np.full(len(fA), "mixed", dtype=object)
    ph[lo >= 0.8] = "both-flat"
    ph[hi < 0.3] = "monthly"
    ab = (hi >= 0.8) & (lo <= 0.3)
    ph[ab & (fA > fB)] = "A"
    ph[ab & ~(fA > fB)] = "B"
    ph[(out["nA"] < 6) | (out["nB"] < 6)] = "insufficient"
    return ph.astype(str)
