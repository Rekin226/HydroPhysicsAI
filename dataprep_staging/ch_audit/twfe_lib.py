"""X5.CH_twfe_mirror.v1 (round 7 rank 16): the within-township estimator, its bootstrap,
placebos and the synthetic G0 gate. Pure numpy/scipy; no file access.

Unit: screen-year. Model:

    head_iy = a_i + g_{township(i), y} + beta * X_iy + e_iy

``a_i`` a screen fixed effect, ``g`` a township x year fixed effect, ``X`` the local census
energy (GWh/yr) in a Gaussian kernel around the screen's site. beta is the FWL slope
``x~'y / x~'x~`` with ``x~`` the exact least-squares residual of X on both fixed effects.
Because screens nest in townships the design is block-diagonal by township, so only townships
with two or more sites identify beta.

Interval: 90 % percentile interval of a site-cluster bootstrap STRATIFIED by township (each
township redraws its own sites with replacement, so every township-year effect stays
estimable; a site drawn twice enters as two copies with their own screen effects), B 2000.
The township-cluster bootstrap is reported beside it.

kappa = beta_obs / beta_model is formed draw by draw on the SAME bootstrap plan (paired), and
is reported, never applied (user ruling 2026-10-29 (4)).

Placebos that gate: the regressor shifted +/-2 years (literal or conditional, the mode fixed by
G0). The within-township permutation is REPORT-ONLY: its pooled slope is ~0 for any y
(sum_p x_p is constant within a township-year, so the fixed effects absorb it) and in a 2-site
township the only non-identity permutation is the swap (x~ -> -x~), so no permutation
statistic can fail on this design (review of the round 7 build; :func:`placebo_perm`).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

LEVEL = 0.90
B_BOOT = 2000
SEED = 20261107
H_KM_PRIMARY = 3.0
H_KM_REPORT = 5.0
TRUNC_H = 3.0                    # kernel truncated at 3 bandwidths
YEARS = tuple(range(2012, 2023))
REG_YEARS = tuple(range(2007, 2023))
SHIFT_YEARS = 2
N_PERM = 200
MIN_MONTHS_PER_YEAR = 9
FLAGS_SCORED = (0, 1)
DEGENERATE_XTX = 1e-12
MAX_DEGENERATE_FRAC = 0.05

# G0 (synthetic, R1c's world)
G0_DRAWS = 500
G0_DRAWS_REPORT = 100
G0_B_REPORT = 500
G0_COVERAGE_MIN = 0.85
G0_POWER_MIN = 0.80
G0_PLACEBO_PASS_MIN = 0.70
KAPPA_ALT = 0.33
RAMP_START_YEAR = 2016
RAMP_FRAC_2022 = 0.15
RAMP_SITE_SHARE = 0.5            # half of the sites INSIDE each township (floor, at least 1)
MCAR_DROP = 0.25
PERSISTENT_PHI_ANNUAL = 0.6      # gating arm: annual AR(1) residual drift
MONTHS = 12 * len(YEARS)

# observed-design check (after the heads are read, before the model and placebo steps)
MIN_OBS_IDENT_ROW_SHARE = 1.0 - MCAR_DROP


def annual_mean_sd(sigma: float, phi: float, n: int = 12) -> float:
    """sd of the n-month mean of a stationary AR(1) with marginal sd ``sigma``."""
    k = np.arange(n)
    return float(sigma * math.sqrt(np.sum(phi ** np.abs(k[:, None] - k[None, :])) / n ** 2))


# ----------------------------------------------------------------------------- regressor
def kernel_energy(wx, wy, px, py, E, h_km: float, trunc: float = TRUNC_H,
                  chunk: int = 50) -> np.ndarray:
    """(W, Y) kernel-weighted energy: sum_p exp(-d^2 / (2 h^2)) E[p, y] over pumps within
    ``trunc`` x h of each site, in the units of E (GWh/yr when E is GWh/yr). The weight is 1
    at the site (unnormalised)."""
    wx, wy = np.asarray(wx, float), np.asarray(wy, float)
    px, py = np.asarray(px, float), np.asarray(py, float)
    E = np.asarray(E, float)
    h = h_km * 1000.0
    out = np.zeros((len(wx), E.shape[1]))
    for a in range(0, len(wx), chunk):
        dx = wx[a:a + chunk, None] - px[None, :]
        dy = wy[a:a + chunk, None] - py[None, :]
        d2 = dx * dx + dy * dy
        w = np.exp(-d2 / (2 * h * h))
        w[d2 > (trunc * h) ** 2] = 0.0
        out[a:a + chunk] = w @ E
    return out


def annual_from_monthly(values: np.ndarray, flags: np.ndarray, years_of_month: np.ndarray,
                        years=YEARS, min_months: int = MIN_MONTHS_PER_YEAR,
                        flags_ok=FLAGS_SCORED) -> np.ndarray:
    """(W, Y) annual means of monthly values with an accepted flag; NaN where fewer than
    ``min_months`` months qualify."""
    v = np.asarray(values, float)
    ok = np.isfinite(v) & np.isin(np.asarray(flags), flags_ok)
    out = np.full((v.shape[0], len(years)), np.nan)
    for j, y in enumerate(years):
        m = np.asarray(years_of_month) == y
        n = ok[:, m].sum(axis=1)
        s = np.where(ok[:, m], v[:, m], 0.0).sum(axis=1)
        out[:, j] = np.where(n >= min_months, s / np.maximum(n, 1), np.nan)
    return out


# ----------------------------------------------------------------------------- design
@dataclass
class Panel:
    """Long panel rows: screen index, year index, site code, township code."""
    well: np.ndarray
    year: np.ndarray
    site: np.ndarray
    town: np.ndarray

    @property
    def n(self) -> int:
        return len(self.well)

    def group(self) -> np.ndarray:
        return self.town * 1000 + self.year


def make_panel(site_of_well, town_of_well, mask) -> Panel:
    """Rows of the (W, Y) boolean ``mask`` in screen-major order."""
    mask = np.asarray(mask, bool)
    w, y = np.nonzero(mask)
    site = np.asarray(site_of_well)[w]
    town = np.asarray(town_of_well)[w]
    return Panel(w.astype(np.int64), y.astype(np.int64), np.asarray(site, np.int64),
                 np.asarray(town, np.int64))


def _group_demean(X: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """X minus its mean within each code (columns of a 2-D X independently)."""
    u, inv = np.unique(codes, return_inverse=True)
    n = len(codes)
    Gt = sp.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(len(u), n))
    cnt = np.bincount(inv, minlength=len(u)).astype(np.float64)
    return X - (Gt @ X / cnt[:, None])[inv]


def resid_fe(X: np.ndarray, a_codes: np.ndarray, b_codes: np.ndarray) -> np.ndarray:
    """Exact residual of X (N,) or (N, D) on two sets of fixed effects: demean by ``b``, then
    project out the ``b``-demeaned ``a`` dummies by least squares (minimum-norm, so the
    rank deficiency of nested effects is harmless)."""
    X = np.asarray(X, dtype=np.float64)
    vec = X.ndim == 1
    if vec:
        X = X[:, None]
    ua, ia = np.unique(a_codes, return_inverse=True)
    Xb = _group_demean(X, b_codes)
    Da = np.zeros((len(ia), len(ua)))
    Da[np.arange(len(ia)), ia] = 1.0
    Db = _group_demean(Da, b_codes)
    coef, *_ = np.linalg.lstsq(Db, Xb, rcond=None)
    R = Xb - Db @ coef
    return R[:, 0] if vec else R


def beta_fe(x: np.ndarray, Y: np.ndarray, panel: Panel) -> tuple[np.ndarray, float]:
    """FWL slope(s) of Y (N,) or (N, D) on x with screen and township-year effects."""
    xt = resid_fe(x, panel.well, panel.group())
    xtx = float(xt @ xt)
    Y = np.asarray(Y, float)
    if xtx <= DEGENERATE_XTX:
        return np.full(Y.shape[1:] if Y.ndim > 1 else (), np.nan), xtx
    return (xt @ Y) / xtx, xtx


def identifying(panel: Panel) -> dict:
    """Townships and sites that can identify beta (two or more sites in a township)."""
    sites_by_town: dict[int, set] = {}
    for s, t in zip(panel.site, panel.town, strict=True):
        sites_by_town.setdefault(int(t), set()).add(int(s))
    ident = {t: s for t, s in sites_by_town.items() if len(s) >= 2}
    return {"n_townships": len(sites_by_town), "n_sites": len(set(panel.site.tolist())),
            "n_identifying_townships": len(ident),
            "n_identifying_sites": int(sum(len(s) for s in ident.values())),
            "n_identifying_rows": int(np.isin(panel.town, list(ident)).sum())}


# ----------------------------------------------------------------------------- bootstrap
@dataclass
class BootDraw:
    rows: np.ndarray      # indices into the panel rows
    well: np.ndarray      # copy-specific screen codes
    group: np.ndarray     # township-year codes (shared by copies)


def bootstrap_plan(panel: Panel, B: int = B_BOOT, seed: int = SEED,
                   kind: str = "site_in_township") -> list[BootDraw]:
    """``site_in_township``: within each township redraw its sites with replacement;
    ``township``: redraw townships with replacement (copies get distinct township codes)."""
    rng = np.random.default_rng(seed)
    rows_of_site: dict[int, np.ndarray] = {}
    for s in np.unique(panel.site):
        rows_of_site[int(s)] = np.flatnonzero(panel.site == s)
    towns = np.unique(panel.town)
    sites_of_town = {int(t): np.unique(panel.site[panel.town == t]) for t in towns}
    rows_of_town = {int(t): np.flatnonzero(panel.town == t) for t in towns}
    grp = panel.group()
    plan = []
    for _ in range(B):
        r_l, w_l, g_l = [], [], []
        if kind == "site_in_township":
            for t in towns:
                ss = sites_of_town[int(t)]
                pick = rng.choice(ss, size=len(ss), replace=True)
                for k, s in enumerate(pick):
                    r = rows_of_site[int(s)]
                    r_l.append(r)
                    w_l.append(panel.well[r] * 10000 + k)
                    g_l.append(grp[r])
        elif kind == "township":
            pick = rng.choice(towns, size=len(towns), replace=True)
            for k, t in enumerate(pick):
                r = rows_of_town[int(t)]
                r_l.append(r)
                w_l.append(panel.well[r] * 10000 + k)
                g_l.append(grp[r] * 10000 + k)
        else:
            raise ValueError(kind)
        plan.append(BootDraw(np.concatenate(r_l), np.concatenate(w_l), np.concatenate(g_l)))
    return plan


def boot_betas(plan: list[BootDraw], x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(B, D) bootstrap slopes. ``x`` (N,) shared by every column of ``Y`` (N, D), or (N, D)
    one regressor per column. Degenerate draws (x~'x~ ~ 0) are NaN."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    x = np.asarray(x, float)
    out = np.full((len(plan), Y.shape[1]), np.nan)
    for b, d in enumerate(plan):
        xb = x[d.rows]
        xt = resid_fe(xb, d.well, d.group)
        Yb = Y[d.rows]
        if xt.ndim == 1:
            xtx = float(xt @ xt)
            if xtx > DEGENERATE_XTX:
                out[b] = (xt @ Yb) / xtx
        else:
            xtx = (xt * xt).sum(axis=0)
            num = (xt * Yb).sum(axis=0)
            out[b] = np.where(xtx > DEGENERATE_XTX, num / np.where(xtx > 0, xtx, 1.0),
                              np.nan)
    return out


def interval(draws: np.ndarray, level: float = LEVEL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile interval per column and the degenerate fraction."""
    a = (1 - level) / 2 * 100
    d = np.asarray(draws, float)
    if d.ndim == 1:
        d = d[:, None]
    lo = np.nanpercentile(d, a, axis=0)
    hi = np.nanpercentile(d, 100 - a, axis=0)
    deg = np.isnan(d).mean(axis=0)
    return lo, hi, deg


# ----------------------------------------------------------------------------- placebos
def shifted(Xw: np.ndarray, shift: int, years=YEARS, reg_years=REG_YEARS) -> np.ndarray:
    """(W, Y) regressor shifted by ``shift`` years (X at y + shift); NaN outside reg_years."""
    idx = {y: j for j, y in enumerate(reg_years)}
    out = np.full((Xw.shape[0], len(years)), np.nan)
    for j, y in enumerate(years):
        if (y + shift) in idx:
            out[:, j] = Xw[:, idx[y + shift]]
    return out


def on_years(Xw: np.ndarray, years=YEARS, reg_years=REG_YEARS) -> np.ndarray:
    return shifted(Xw, 0, years, reg_years)


def placebo_shift(x_rows: np.ndarray, xs_rows: np.ndarray, Y: np.ndarray, panel: Panel,
                  conditional: bool) -> np.ndarray:
    """Slope on the shifted regressor, on the rows where it exists. ``conditional``: with the
    contemporaneous regressor as a control (FWL)."""
    ok = np.isfinite(xs_rows)
    sub = Panel(panel.well[ok], panel.year[ok], panel.site[ok], panel.town[ok])
    Ys = np.asarray(Y, float)[ok]
    xs = resid_fe(xs_rows[ok], sub.well, sub.group())
    if conditional:
        xc = resid_fe(x_rows[ok], sub.well, sub.group())
        cc = float(xc @ xc)
        if cc > DEGENERATE_XTX:
            xs = xs - (float(xs @ xc) / cc) * xc
    ss = float(xs @ xs)
    if ss <= DEGENERATE_XTX:
        return np.full(Ys.shape[1:] if Ys.ndim > 1 else (), np.nan)
    return (xs @ Ys) / ss


def permutations(panel: Panel, n_wells: int, P: int = N_PERM, seed: int = SEED + 1
                 ) -> list[np.ndarray]:
    """Within-township permutations of the SITE -> regressor-series assignment: each returns,
    for every screen, the screen whose series it takes (screens of one site move together,
    to the first screen of the target site)."""
    rng = np.random.default_rng(seed)
    site_w, town_w = {}, {}
    for w, s, t in zip(panel.well, panel.site, panel.town, strict=True):
        site_w[int(w)] = int(s)
        town_w[int(w)] = int(t)
    first_screen = {}
    for w in sorted(site_w):
        first_screen.setdefault(site_w[w], w)
    sites_by_town: dict[int, list[int]] = {}
    for s, w in first_screen.items():
        sites_by_town.setdefault(town_w[w], []).append(s)
    out = []
    for _ in range(P):
        to = {}
        for _t, ss in sites_by_town.items():
            perm = rng.permutation(ss)
            to.update({int(a): int(b) for a, b in zip(ss, perm, strict=True)})
        src = np.arange(n_wells)
        for w, s in site_w.items():
            src[w] = first_screen[to[s]]
        out.append(src)
    return out


def placebo_perm(Xw_years: np.ndarray, perms: list[np.ndarray], Y: np.ndarray,
                 panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Permutation slopes: (P, D) per-permutation slopes and the (D,) pooled permutation
    slope sum_p x~_p'y / sum_p x~_p'x~_p. REPORT-ONLY, never a gate: the numerator is
    M(sum_p x_p)'y and sum_p x_p is (in expectation, exactly as P grows) constant within a
    township-year, so the residual-maker sends it to ~0 for ANY y; in a 2-site township the
    swap gives x~_swap = -x~. The statistic cannot fail on this design; the per-draw slopes
    feed a report-only permutation p-value."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    out = np.full((len(perms), Y.shape[1]), np.nan)
    num = np.zeros(Y.shape[1])
    den = 0.0
    for i, src in enumerate(perms):
        xp = Xw_years[src][panel.well, panel.year]
        xt = resid_fe(xp, panel.well, panel.group())
        xtx = float(xt @ xt)
        nm = xt @ Y
        num += nm
        den += xtx
        if xtx > DEGENERATE_XTX:
            out[i] = nm / xtx
    pooled = num / den if den > DEGENERATE_XTX else np.full(Y.shape[1], np.nan)
    return out, pooled


# ----------------------------------------------------------------------------- G0 noise
def monthly_noise(rng, block_of_well: np.ndarray, sigma: float, block_share: float,
                  phi: float, D: int, months: int = MONTHS) -> np.ndarray:
    """(W, months, D) AR(1) noise: a 5 km block component (variance share ``block_share``)
    plus an idiosyncratic one, both AR(1) with ``phi``, stationary start, marginal sd
    ``sigma`` (the n2c_reading noise model, declared parameters)."""
    blocks, binv = np.unique(block_of_well, return_inverse=True)
    W = len(block_of_well)
    sb = sigma * math.sqrt(block_share)
    si = sigma * math.sqrt(1 - block_share)
    innov = math.sqrt(1 - phi * phi)
    zb = rng.standard_normal((len(blocks), D))
    zi = rng.standard_normal((W, D))
    out = np.empty((W, months, D))
    for t in range(months):
        if t:
            zb = phi * zb + innov * rng.standard_normal(zb.shape)
            zi = phi * zi + innov * rng.standard_normal(zi.shape)
        out[:, t, :] = sb * zb[binv] + si * zi
    return out


def annual_noise(rng, block_of_well, sigma, block_share, phi, D) -> np.ndarray:
    """(W, Y, D) annual means of :func:`monthly_noise`."""
    m = monthly_noise(rng, block_of_well, sigma, block_share, phi, D)
    return m.reshape(m.shape[0], len(YEARS), 12, D).mean(axis=2)


def persistent_noise(rng, block_of_well, sigma, block_share, phi, D,
                     phi_annual: float = PERSISTENT_PHI_ANNUAL) -> np.ndarray:
    """(W, Y, D) annual AR(1) residual drift (``phi_annual``), stationary start, with the same
    5 km block share and the same annual sd as the annual mean of the declared monthly model
    (:func:`annual_mean_sd`). A gating G0 arm: monthly AR(1) with phi 0.859 decorrelates within
    a year (phi^12 = 0.16), while real screen residuals can drift for years."""
    sd = annual_mean_sd(sigma, phi)
    blocks, binv = np.unique(block_of_well, return_inverse=True)
    W = len(block_of_well)
    sb = sd * math.sqrt(block_share)
    si = sd * math.sqrt(1 - block_share)
    k = math.sqrt(1 - phi_annual ** 2)
    zb = rng.standard_normal((len(blocks), D))
    zi = rng.standard_normal((W, D))
    out = np.empty((W, len(YEARS), D))
    for j in range(len(YEARS)):
        if j:
            zb = phi_annual * zb + k * rng.standard_normal(zb.shape)
            zi = phi_annual * zi + k * rng.standard_normal(zi.shape)
        out[:, j, :] = sb * zb[binv] + si * zi
    return out


# ----------------------------------------------------------------------------- G0
def _rows(A: np.ndarray, panel: Panel) -> np.ndarray:
    """(W, Y[, D]) -> (N[, D]) panel rows."""
    return A[panel.well, panel.year]


def ramp_profile(years=YEARS) -> np.ndarray:
    """r(y): 0 up to RAMP_START_YEAR, rising linearly to RAMP_FRAC_2022 in the last year."""
    yrs = np.asarray(years, float)
    return RAMP_FRAC_2022 * np.clip((yrs - RAMP_START_YEAR) / (years[-1] - RAMP_START_YEAR),
                                    0.0, 1.0)


def ramp_hits(site, town, rng, share: float = RAMP_SITE_SHARE) -> np.ndarray:
    """(W,) True for the screens of the sites whose census understates. Inside every township
    with n >= 2 sites, floor(share x n) (at least 1) sites are drawn without replacement, so
    the understatement differs WITHIN the township (a township-wide factor would be absorbed
    by the township x year effect); a single-site township is hit with probability ``share``
    (it does not identify beta)."""
    site = np.asarray(site)
    town = np.asarray(town)
    hit_sites: list[int] = []
    for t in np.unique(town):
        ss = np.unique(site[town == t])
        if len(ss) >= 2:
            k = max(1, int(math.floor(share * len(ss))))
            hit_sites.extend(int(s) for s in rng.choice(ss, size=k, replace=False))
        elif rng.random() < share:
            hit_sites.append(int(ss[0]))
    return np.isin(site, hit_sites)


def kappa_interval(b_obs: np.ndarray, b_mod: np.ndarray, level: float = LEVEL
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paired percentile interval of b_obs / b_mod over the bootstrap draws. ``b_obs`` (B,) or
    (B, D); ``b_mod`` (B,) shared or (B, D). Draws with a non-finite slope or |b_mod| <= 1e-12
    are dropped and counted."""
    bo = np.asarray(b_obs, float)
    if bo.ndim == 1:
        bo = bo[:, None]
    bm = np.asarray(b_mod, float)
    bm = bm[:, None] if bm.ndim == 1 else bm
    bm = np.broadcast_to(bm, bo.shape)
    ok = np.isfinite(bo) & np.isfinite(bm) & (np.abs(bm) > 1e-12)
    kd = np.where(ok, bo / np.where(ok, bm, 1.0), np.nan)
    a = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(kd, a, axis=0)
        hi = np.nanpercentile(kd, 100 - a, axis=0)
    return lo, hi, (~ok).sum(axis=0)


def g0(H: np.ndarray, Xw_years: np.ndarray, site, town, block, noise: dict,
       Xw_shift: dict, draws: int = G0_DRAWS, B: int = B_BOOT, seed: int = SEED + 2,
       report_arms: bool = True) -> dict:
    """The synthetic gate in a model's world (R1c's).

    ``H`` (W, Y) the model's annual heads at the screens (complete), ``Xw_years`` (W, Y) the
    regressor, ``Xw_shift`` {+2: (W, Y), -2: (W, Y)} the shifted regressors, ``noise`` the
    declared parameters (sigma_m, block_share, phi). Truth beta_model = TWFE slope of H on X.

    Gating arms, all on one bootstrap plan with the model column H beside them (paired kappa,
    as scored):

    * base (y = H + e): coverage of beta_model; coverage of kappa = 1 by the paired kappa
      interval; shifted-placebo pass rates;
    * kappa033 (y = H - (1 - 0.33) beta_model X + e): power; coverage of kappa = 0.33;
    * persistent (y = H + annual AR(1) 0.6 drift of the same annual sd and block share):
      coverage of beta_model;
    * ramp (the estimator sees X (1 - r(y)) at a random half of the SITES inside each
      township, r rising linearly from 0 in 2016 to 15 % in 2022; heads respond to the true
      X): coverage of beta_model.

    Report only: the permutation placebo's pass rate (degenerate; documents why it does not
    gate), kappa power, and mcar25 (25 % of screen-years missing at random).
    """
    rng = np.random.default_rng(seed)
    W, Y = H.shape
    panel = make_panel(site, town, np.ones((W, Y), bool))
    x = _rows(Xw_years, panel)
    hy = _rows(H, panel)
    beta_m, xtx = beta_fe(x, hy, panel)
    beta_m = float(beta_m)
    res: dict = {"beta_model": beta_m, "xtx": xtx, "design": identifying(panel),
                 "draws": draws, "B": B}
    if not np.isfinite(beta_m) or beta_m == 0.0:
        res.update({"passed": False, "reason": "beta_model degenerate (no within-township "
                                               "variation)"})
        return res
    plan = bootstrap_plan(panel, B, seed + 10)
    nz = (noise["sigma_m"], noise["block_share"], noise["phi"])

    def noise_rows(n, persistent=False):
        f = persistent_noise if persistent else annual_noise
        e = f(rng, block, *nz, n)
        return e[panel.well, panel.year]          # (N, n)

    # base, kappa 0.33 and persistent share the design: x~ per bootstrap draw computed once
    Yb = hy[:, None] + noise_rows(draws)
    Yk = (hy - (1 - KAPPA_ALT) * beta_m * x)[:, None] + noise_rows(draws)
    Yp = hy[:, None] + noise_rows(draws, persistent=True)
    pt_b, _ = beta_fe(x, Yb, panel)
    bb = boot_betas(plan, x, np.hstack([Yb, Yk, Yp, hy[:, None]]))
    bm_draws = bb[:, -1]
    lo, hi, deg = interval(bb[:, :-1])
    sb, sk, sq = slice(0, draws), slice(draws, 2 * draws), slice(2 * draws, 3 * draws)
    lo_b, hi_b, lo_k, hi_k, lo_p, hi_p = lo[sb], hi[sb], lo[sk], hi[sk], lo[sq], hi[sq]
    cover_base = float(np.mean((lo_b <= beta_m) & (beta_m <= hi_b)))
    cover_pers = float(np.mean((lo_p <= beta_m) & (beta_m <= hi_p)))
    alt = KAPPA_ALT * beta_m
    p1 = float(np.mean((alt < lo_b) | (alt > hi_b)))
    p2 = float(np.mean((beta_m < lo_k) | (beta_m > hi_k)))
    cover_k = float(np.mean((lo_k <= alt) & (alt <= hi_k)))
    klo_b, khi_b, kdrop_b = kappa_interval(bb[:, sb], bm_draws)
    klo_k, khi_k, kdrop_k = kappa_interval(bb[:, sk], bm_draws)
    kcov_1 = float(np.mean((klo_b <= 1.0) & (khi_b >= 1.0)))
    kcov_alt = float(np.mean((klo_k <= KAPPA_ALT) & (khi_k >= KAPPA_ALT)))
    kpow1 = float(np.mean((klo_b > KAPPA_ALT) | (khi_b < KAPPA_ALT)))
    kpow2 = float(np.mean((klo_k > 1.0) | (khi_k < 1.0)))
    hw = (hi_b - lo_b) / 2
    # placebo calibration in the base arm: the shifted regressors (the gating placebos)
    shifts = {}
    for s, Xs in Xw_shift.items():
        xs = _rows(Xs, panel)
        lit = placebo_shift(x, xs, Yb, panel, conditional=False)
        con = placebo_shift(x, xs, Yb, panel, conditional=True)
        shifts[s] = {"literal": lit, "conditional": con}
    rates = {}
    for mode in ("literal", "conditional"):
        ok = np.ones(draws, bool)
        for s in shifts:
            ok &= np.abs(shifts[s][mode]) < hw
        rates[mode] = float(np.mean(ok))
    placebo_mode = ("literal" if rates["literal"] >= G0_PLACEBO_PASS_MIN else
                    "conditional" if rates["conditional"] >= G0_PLACEBO_PASS_MIN else None)
    # the permutation placebo, report only: it passes whatever the heads (documents why)
    _, perm_pooled = placebo_perm(Xw_years, permutations(panel, W), Yb, panel)
    perm_rate = float(np.mean(np.abs(perm_pooled) < hw))
    # ramp arm (site level): the regressor differs per draw
    r_y = ramp_profile()
    Xr = np.empty((panel.n, draws))
    for d in range(draws):
        hit = ramp_hits(site, town, rng)
        f = np.where(hit[:, None], 1.0 - r_y[None, :], 1.0)
        Xr[:, d] = _rows(Xw_years * f, panel)
    Yr = hy[:, None] + noise_rows(draws)
    br = boot_betas(plan, Xr, Yr)
    lo_r, hi_r, deg_r = interval(br)
    cover_ramp = float(np.mean((lo_r <= beta_m) & (beta_m <= hi_r)))
    pt_r = np.array([beta_fe(Xr[:, d], Yr[:, d], panel)[0] for d in range(draws)])
    deg_frac = float(max(deg.max(), deg_r.max()))
    power = min(p1, p2)
    gates = {"coverage_base": cover_base >= G0_COVERAGE_MIN,
             "coverage_ramp": cover_ramp >= G0_COVERAGE_MIN,
             "coverage_persistent": cover_pers >= G0_COVERAGE_MIN,
             "kappa_coverage_1": kcov_1 >= G0_COVERAGE_MIN,
             "kappa_coverage_033": kcov_alt >= G0_COVERAGE_MIN,
             "power": power >= G0_POWER_MIN,
             "placebo": placebo_mode is not None,
             "degenerate": deg_frac <= MAX_DEGENERATE_FRAC}
    passed = bool(all(gates.values()))
    res.update({
        "coverage_base": cover_base, "coverage_ramp": cover_ramp,
        "coverage_persistent": cover_pers,
        "kappa_coverage_1": kcov_1, "kappa_coverage_033": kcov_alt,
        "coverage_kappa033_beta_report": cover_k,
        "power_excl_033_when_1": p1, "power_excl_1_when_033": p2, "power": power,
        "kappa_power_report": {"excl_033_when_1": kpow1, "excl_1_when_033": kpow2},
        "median_kappa_width_base": float(np.nanmedian(khi_b - klo_b)),
        "median_kappa_width_kappa033": float(np.nanmedian(khi_k - klo_k)),
        "kappa_draws_dropped_max": int(max(kdrop_b.max(), kdrop_k.max())),
        "median_halfwidth_base": float(np.median(hw)),
        "median_point_base": float(np.median(pt_b)),
        "median_point_ramp": float(np.nanmedian(pt_r)),
        "placebo_pass_rate": rates, "placebo_mode": placebo_mode,
        "placebo_false_failure_rate": (None if placebo_mode is None
                                       else 1.0 - rates[placebo_mode]),
        "permutation_pass_rate_report_only": perm_rate,
        "degenerate_bootstrap_frac_max": deg_frac,
        "gates": gates,
        "thresholds": {"coverage_min": G0_COVERAGE_MIN, "power_min": G0_POWER_MIN,
                       "placebo_pass_min": G0_PLACEBO_PASS_MIN,
                       "max_degenerate_frac": MAX_DEGENERATE_FRAC},
        "passed": passed})
    if report_arms:
        res["report_only"] = _g0_report(H, Xw_years, site, town, block, noise, rng)
    return res


def _g0_report(H, Xw_years, site, town, block, noise, rng) -> dict:
    """MCAR missingness (report only): one mask per draw, its own plan; coverage of the masked
    beta_model."""
    W, Y = H.shape
    cov = []
    for _ in range(min(G0_DRAWS_REPORT, 50)):
        mask = rng.random((W, Y)) >= MCAR_DROP
        pn = make_panel(site, town, mask)
        e = annual_noise(rng, block, noise["sigma_m"], noise["block_share"], noise["phi"], 1)
        y = H[pn.well, pn.year] + e[pn.well, pn.year, 0]
        bm, _ = beta_fe(Xw_years[pn.well, pn.year], H[pn.well, pn.year], pn)
        bb = boot_betas(bootstrap_plan(pn, 200, int(rng.integers(1 << 30))),
                        Xw_years[pn.well, pn.year], y)
        lo, hi, _ = interval(bb)
        cov.append(bool(lo[0] <= bm <= hi[0]) if np.isfinite(bm) else False)
    return {"mcar25_coverage_of_masked_beta_model": float(np.mean(cov)),
            "note": "report only; never gates"}


# ----------------------------------------------------------------------------- scoring
def estimate(Xw_years: np.ndarray, Yw: np.ndarray, site, town, plan_kind: str = "site_in_township",
             B: int = B_BOOT, seed: int = SEED, Hm: dict | None = None) -> dict:
    """beta on the rows where ``Yw`` (W, Y) is finite, its interval, and for each model in
    ``Hm`` {name: (W, Y)} the model slope on the SAME rows and the paired kappa interval.
    The caller builds each model's annual means on the months the observation uses."""
    mask = np.isfinite(Yw) & np.isfinite(Xw_years)
    panel = make_panel(site, town, mask)
    x = _rows(Xw_years, panel)
    y = _rows(Yw, panel)
    b, xtx = beta_fe(x, y, panel)
    plan = bootstrap_plan(panel, B, seed, plan_kind)
    names = list(Hm or {})
    cols = [y] + [_rows(Hm[n], panel) for n in names]
    bb = boot_betas(plan, x, np.column_stack(cols))
    lo, hi, deg = interval(bb[:, :1])
    out = {"beta": float(b), "lo": float(lo[0]), "hi": float(hi[0]),
           "halfwidth": float((hi[0] - lo[0]) / 2), "degenerate_frac": float(deg[0]),
           "n_rows": panel.n, "design": identifying(panel), "xtx": xtx, "models": {}}
    for j, n in enumerate(names, start=1):
        if not np.isfinite(cols[j]).all():
            raise ValueError(f"model {n} has no value on an observed screen-year")
        bm, _ = beta_fe(x, cols[j], panel)
        lm, hm, dm = interval(bb[:, j:j + 1])
        k_lo, k_hi, k_drop = kappa_interval(bb[:, 0], bb[:, j])
        out["models"][n] = {
            "beta_model": float(bm), "lo": float(lm[0]), "hi": float(hm[0]),
            "model_ci_excludes_0": bool(lm[0] > 0 or hm[0] < 0),
            "kappa": float(b / bm) if np.isfinite(bm) and bm != 0 else float("nan"),
            "kappa_lo": float(k_lo[0]), "kappa_hi": float(k_hi[0]),
            "kappa_draws_dropped": int(k_drop[0])}
    return out


def design_check(g0_res: dict, est: dict) -> dict:
    """The observed-design check (declared; after the heads are read, before the model and
    placebo steps). G0 validated the complete 74 x 11 panel; scoring runs on the observed
    screen-years (flags 0-1, >= 9 months). The observed rows must keep every identifying
    township and site G0 had, at least MIN_OBS_IDENT_ROW_SHARE of its identifying screen-years
    (the G0 report-only mcar25 arm's missing share), a finite slope and at most
    MAX_DEGENERATE_FRAC degenerate bootstrap draws."""
    g = g0_res.get("design") or {}
    o = est.get("design") or {}
    why = []
    for k in ("n_identifying_townships", "n_identifying_sites"):
        if int(o.get(k, 0)) < int(g.get(k, 0)):
            why.append(f"{k} {o.get(k, 0)} < G0's {g.get(k, 0)}")
    gr = int(g.get("n_identifying_rows", 0))
    share = float(o.get("n_identifying_rows", 0)) / gr if gr else float("nan")
    if not share >= MIN_OBS_IDENT_ROW_SHARE:
        why.append(f"identifying screen-years {share:.2f} of G0's < {MIN_OBS_IDENT_ROW_SHARE}")
    if not np.isfinite(est.get("beta", float("nan"))):
        why.append("observed slope not finite (no within-township variation)")
    dg = est.get("degenerate_frac", 1.0)
    if not dg <= MAX_DEGENERATE_FRAC:
        why.append(f"degenerate bootstrap draws {dg:.3f} > {MAX_DEGENERATE_FRAC}")
    return {"passed": not why, "why": why, "identifying_row_share": share,
            "observed_design": o, "g0_design": g}


def placebos(Xw_years, Xw_shift: dict, Yw, site, town, halfwidth: float, mode: str) -> dict:
    """The declared placebos on the observed rows: shifted regressors (``mode`` literal or
    conditional, fixed by G0; these gate) and the within-township permutation (report only:
    pooled slope, per-draw median, p-value)."""
    mask = np.isfinite(Yw) & np.isfinite(Xw_years)
    panel = make_panel(site, town, mask)
    x = _rows(Xw_years, panel)
    y = _rows(Yw, panel)
    out = {"mode": mode, "halfwidth": halfwidth, "shifts": {}}
    ok = True
    for s, Xs in Xw_shift.items():
        b = float(placebo_shift(x, _rows(Xs, panel), y, panel, conditional=(mode ==
                                                                             "conditional")))
        p = bool(np.isfinite(b) and abs(b) < halfwidth)
        out["shifts"][str(s)] = {"beta": b, "passed": p}
        ok &= p
    bp, pooled = placebo_perm(Xw_years, permutations(panel, Yw.shape[0]), y, panel)
    bp, pooled = bp[:, 0], float(pooled[0])
    b0, _ = beta_fe(x, y, panel)
    pval = float(np.mean(np.abs(bp) >= abs(float(b0)))) if np.isfinite(b0) else float("nan")
    out["permutation_report_only"] = {
        "pooled_beta": pooled, "median_beta": float(np.nanmedian(bp)), "p_value": pval,
        "note": "never gates: degenerate on this design (pooled slope ~0 for any y; the only "
                "non-identity permutation of a 2-site township is the swap)"}
    out["passed"] = bool(ok)
    return out


READINGS = {
    "not_identifiable_g0": "within-township design-time estimate: not identifiable within "
                           "county (synthetic gate G0 failed: {why})",
    "not_identifiable_design": "within-township design-time estimate: not identifiable within "
                               "county (the observed screen-years do not keep the design G0 "
                               "validated: {why})",
    "not_identifiable_model": "within-township design-time estimate: not identifiable within "
                              "county (R1c's own within-township response is not detectable "
                              "on the observed screen-years)",
    "not_identifiable_placebo": "within-township design-time estimate: not identifiable within "
                                "county (placebo failed: {why}; a correct model world fails "
                                "it in {ffr} of G0 draws, so this is not evidence against "
                                "R1c)",
    "opposite_sign": "within-township design-time estimate: Changhua heads respond with the "
                     "opposite sign to R1c's per GWh of local census energy (kappa "
                     "[{lo:.2f}, {hi:.2f}]; not a correction)",
    "uninformative": "within-township design-time estimate: uninformative (kappa "
                     "[{lo:.2f}, {hi:.2f}] of R1c's per GWh covers both 0.33 and 1; not a "
                     "correction)",
    "narrowing": "within-township design-time estimate: Changhua heads respond kappa "
                 "[{lo:.2f}, {hi:.2f}] of R1c's per GWh (not a correction)",
}

G0_GATE_KEYS = (("coverage_base", G0_COVERAGE_MIN), ("coverage_ramp", G0_COVERAGE_MIN),
                ("coverage_persistent", G0_COVERAGE_MIN),
                ("kappa_coverage_1", G0_COVERAGE_MIN), ("kappa_coverage_033", G0_COVERAGE_MIN),
                ("power", G0_POWER_MIN))


def kappa_band(lo: float, hi: float) -> str:
    """Declared bands of the paired kappa interval: entirely below 0 -> opposite_sign; covering
    both 0.33 and 1 (or not finite) -> uninformative; else narrowing."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "uninformative"
    if hi < 0:
        return "opposite_sign"
    if lo <= KAPPA_ALT and hi >= 1.0:
        return "uninformative"
    return "narrowing"


def reading(g0_res: dict, est: dict | None, plc: dict | None) -> dict:
    """The declared reading (R1c is the reference). Order: G0, the observed design, R1c's own
    response, the shifted placebos, then the kappa band."""
    if not g0_res.get("passed"):
        why = []
        for k, thr in G0_GATE_KEYS:
            v = g0_res.get(k)
            if v is None or not v >= thr:
                why.append(f"{k} {v if v is None else round(v, 3)} < {thr}")
        if g0_res.get("placebo_mode") is None and "placebo_pass_rate" in g0_res:
            why.append("no placebo calibrated")
        if g0_res.get("reason"):
            why.append(g0_res["reason"])
        if g0_res.get("degenerate_bootstrap_frac_max", 0) > MAX_DEGENERATE_FRAC:
            why.append("degenerate bootstrap")
        return {"key": "not_identifiable_g0",
                "text": READINGS["not_identifiable_g0"].format(why="; ".join(why) or "failed")}
    dchk = design_check(g0_res, est)
    if not dchk["passed"]:
        return {"key": "not_identifiable_design", "design_check": dchk,
                "text": READINGS["not_identifiable_design"].format(why="; ".join(dchk["why"]))}
    m = est["models"]["R1c"]
    if not m["model_ci_excludes_0"]:
        return {"key": "not_identifiable_model", "text": READINGS["not_identifiable_model"]}
    if not plc["passed"]:
        bad = [f"shift {s}" for s, v in plc["shifts"].items() if not v["passed"]]
        ffr = g0_res.get("placebo_false_failure_rate")
        ffr = "an unknown share" if ffr is None else f"{ffr:.0%}"
        return {"key": "not_identifiable_placebo",
                "text": READINGS["not_identifiable_placebo"].format(why=", ".join(bad),
                                                                    ffr=ffr)}
    band = kappa_band(m["kappa_lo"], m["kappa_hi"])
    return {"key": band, "text": READINGS[band].format(lo=m["kappa_lo"], hi=m["kappa_hi"]),
            "kappa": m["kappa"], "kappa_lo": m["kappa_lo"], "kappa_hi": m["kappa_hi"]}
