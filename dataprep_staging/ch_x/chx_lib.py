"""Pure functions of I1.CHX.v1 and I1.CH_backcast.v1 (round 7 ranks 13 and 18).

No function here reads a file. The runners pass model arrays (hindcast heads, energy) and the
scorers pass the well-month panel; the tests drive everything with synthetic arrays.

Estimators come from the pinned shared N2/N2C scorer ``dataprep_staging/n2_policy/
bootstrap_scorer.py`` (sha256 f8990d61..., cited, never edited): ``beta_response`` (per-well
fixed-effects slope with the 5 km block bootstrap, B 2000, seed 20261101, 90 %),
``informative_gate``, ``rmse``, ``within`` and ``decide``. What this module adds is declared
in I1.CHX.v1 / I1.CH_backcast.v1:

* the contrast arms (a county held at a reference-year monthly pattern, cell by cell);
* the declared fixed effects (``fe_basis``, exact least squares) and the transform of e and c
  before the pinned slope;
* the primary of I1.CHX.v1: ``beta_response`` on e~, c~ under ``chx_const.CHX_PRIMARY_FORM``
  ('twoway': per-well intercept and linear trend + month-of-sample effects) and ``decide()``
  fed with that slope, its interval and RMSEs on the same transformed rows; 'direction' =
  verdict in {response_supported, direction_supported_size_not};
* the informative gate on the TRANSFORMED contrast (the variation that identifies the slope);
* the bootstrap blocks: 5 km squares (``block_ids``), the 07100111-141 nest as one block;
* G0, the synthetic decidability gate (common confounder with heterogeneous loading, common
  trend);
* the CH-J gate statistics (seasonal amplitude ratios, county-mean correlation, no-harm);
* the backcast primary: the per-well FE slope on rows residualised on well x calendar-month
  and month-of-sample effects, and its report-only synthetic power.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "n2_policy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import chx_const as K  # noqa: E402


# ----------------------------------------------------------------------------- months
def month_list(first: str, last: str) -> list[str]:
    """Inclusive ``YYYY-MM`` list."""
    y0, m0 = int(first[:4]), int(first[5:7])
    y1, m1 = int(last[:4]), int(last[5:7])
    a, b = 12 * y0 + m0 - 1, 12 * y1 + m1 - 1
    if b < a:
        raise ValueError(f"{first} > {last}")
    return [f"{k // 12}-{k % 12 + 1:02d}" for k in range(a, b + 1)]


def month_t(months) -> np.ndarray:
    """Integer month index 12 * year + month - 1 (the scorer's time variable)."""
    return np.array([int(m[:4]) * 12 + int(m[5:7]) - 1 for m in months], float)


def window_slice(months: list[str], first: str, last: str) -> slice:
    i0, i1 = months.index(first), months.index(last)
    return slice(i0, i1 + 1)


# ----------------------------------------------------------------------------- arms
def pattern_energy(E: np.ndarray, months: list[str], cells: np.ndarray, pattern_year: int
                   ) -> np.ndarray:
    """The ZR energy: every month after ``pattern_year`` of the ``cells`` (bool, (A,))
    replaced, cell by cell, by the same calendar month of ``pattern_year``. ``E`` is
    ``(..., A, T)`` on ``months``; other cells and the pattern year itself are unchanged."""
    E = np.asarray(E, float)
    cells = np.asarray(cells, bool)
    if E.shape[-1] != len(months) or E.shape[-2] != cells.shape[0]:
        raise ValueError(f"E {E.shape} does not match {cells.shape[0]} cells x "
                         f"{len(months)} months")
    yr = np.array([int(m[:4]) for m in months])
    mo = np.array([int(m[5:7]) for m in months])
    ref = {int(m): int(np.flatnonzero((yr == pattern_year) & (mo == m))[0])
           for m in range(1, 13) if ((yr == pattern_year) & (mo == m)).any()}
    if len(ref) != 12:
        raise ValueError(f"{pattern_year} is not a full year of the axis")
    Z = np.array(E, copy=True)
    idx = np.flatnonzero(cells)
    for t in np.flatnonzero(yr > pattern_year):
        Z[..., idx, t] = E[..., idx, ref[int(mo[t])]]
    return Z


def county_anomaly(E: np.ndarray, months: list[str], cells: np.ndarray) -> np.ndarray:
    """Relative county pumping anomaly per month: total energy of ``cells`` divided by its
    calendar-month mean over the whole axis, minus 1 (classes summed when ``E`` is
    ``(K, A, T)``)."""
    E = np.asarray(E, float)
    if E.ndim == 3:
        E = E.sum(axis=0)
    tot = E[np.asarray(cells, bool)].sum(axis=0)
    mo = np.array([int(m[5:7]) for m in months])
    clim = np.array([tot[mo == m].mean() for m in range(1, 13)])
    base = clim[mo - 1]
    return np.divide(tot, base, out=np.zeros_like(tot), where=base > 0) - 1.0


def screen_series(H: np.ndarray, layers, cells) -> np.ndarray:
    """``(W, T)`` heads at each screen's (layer, cell) from ``H`` ``(L, A, T)``."""
    layers = np.asarray(layers, int)
    cells = np.asarray(cells, int)
    if (layers < 0).any() or (cells < 0).any():
        raise ValueError("a screen has no layer or no active cell")
    return np.asarray(H, float)[layers, cells, :]


# ----------------------------------------------------------------------------- blocks
def well_blocks(sids, x, y, nest=K.NEST, nest_block: str = K.NEST_BLOCK) -> dict:
    """{sid: block}: the scorer's 5 km squares, the declared nest forced into one block."""
    sids = [str(s) for s in sids]
    b = BS.block_ids(np.asarray(x, float), np.asarray(y, float))
    out = dict(zip(sids, (str(v) for v in b), strict=True))
    for s in nest:
        if s in out:
            out[s] = nest_block
    return out


# ----------------------------------------------------------------------------- panel
def long_panel(sids, months, obs: np.ndarray, zr: np.ndarray, c: np.ndarray,
               flag: np.ndarray | None = None) -> dict:
    """Wide ``(W, T)`` arrays -> the scorer's long well-month rows (NaN obs dropped; flags
    other than 0-1 dropped; ``flag`` None = every finite obs has flag 0)."""
    sids = np.asarray([str(s) for s in sids])
    W, T = np.asarray(obs).shape
    if np.asarray(c).shape != (W, T) or np.asarray(zr).shape != (W, T) or len(months) != T:
        raise ValueError("panel arrays differ in shape")
    fl = np.zeros((W, T), int) if flag is None else np.asarray(flag, int)
    well = np.repeat(sids, T)
    mon = np.tile(np.asarray(months), W)
    r = BS._rows(well, mon, np.asarray(obs, float).ravel(), np.asarray(zr, float).ravel(),
                 np.asarray(c, float).ravel(), fl.ravel())
    return r


def eligible(rows: dict, min_months: int = K.MIN_WELL_MONTHS) -> dict:
    """Rows of wells with at least ``min_months`` scored months (others dropped, listed)."""
    u, n = np.unique(rows["well"], return_counts=True)
    keep_w = set(u[n >= min_months])
    k = np.isin(rows["well"], list(keep_w))
    out = {key: (v[k] if isinstance(v, np.ndarray) and v.shape == rows["well"].shape else v)
           for key, v in rows.items()}
    out["dropped_wells"] = sorted(set(u) - keep_w)
    out["n_rows_scored"] = int(k.sum())
    return out


# ----------------------------------------------------------------------------- fixed effects
def fe_basis(wells, months, form: str) -> np.ndarray:
    """Orthonormal basis ``(n, r)`` of the declared fixed effects on these rows (exact least
    squares; rank-deficient designs are fine): ``detrended`` per-well intercept and linear
    trend in month; ``twoway`` that plus one effect per month of sample; ``calmonth`` one
    effect per (well, calendar month); ``calmonth_twoway`` that plus one effect per month of
    sample."""
    if form not in K.FORMS:
        raise ValueError(f"unknown form {form!r}")
    wells = np.asarray(wells).astype(str)
    months = np.asarray([str(m)[:7] for m in months])
    n = len(wells)
    if n == 0:
        return np.zeros((0, 0))
    rows = np.arange(n)
    cols = []
    if form in ("detrended", "twoway"):
        uw, iw = np.unique(wells, return_inverse=True)
        D = np.zeros((n, len(uw)))
        D[rows, iw] = 1.0
        t = month_t(months)
        cols += [D, D * (t - t.mean())[:, None]]
    else:
        key = np.char.add(np.char.add(wells, "|"), np.asarray([m[5:7] for m in months]))
        uk, ik = np.unique(key, return_inverse=True)
        D = np.zeros((n, len(uk)))
        D[rows, ik] = 1.0
        cols.append(D)
    if form in ("twoway", "calmonth_twoway"):
        um, im = np.unique(months, return_inverse=True)
        M = np.zeros((n, len(um)))
        M[rows, im] = 1.0
        cols.append(M)
    U, s, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
    return U[:, s > s[0] * 1e-10]


def fe_resid(v, Q: np.ndarray) -> np.ndarray:
    """``v`` minus its projection on the fixed effects ``Q`` (``fe_basis``)."""
    v = np.asarray(v, float)
    return v - Q @ (Q.T @ v)


def transform(e, c, rows: dict, form: str, Q: np.ndarray | None = None):
    """(e~, c~) under ``form``: ``detrended`` with the pinned scorer's own ``within``
    (identical to the projection), the other forms by ``fe_resid``."""
    w = rows["well"]
    if form == "detrended":
        uniq, inv = BS._codes(w)
        return (BS.within(e, inv, len(uniq), rows["t"], detrend=True),
                BS.within(c, inv, len(uniq), rows["t"], detrend=True))
    Q = fe_basis(w, rows["month"], form) if Q is None else Q
    if Q.shape[0] != len(w):
        raise ValueError("fixed-effect basis and rows differ")
    return fe_resid(e, Q), fe_resid(c, Q)


# ----------------------------------------------------------------------------- primary
def detrended_rmse(e, c, wells, t) -> dict:
    """RMSE(ZR) = equal-well RMS of e~, RMSE(PR) = of e~ - c~, e~ and c~ detrended per well."""
    uniq, inv = BS._codes(wells)
    et = BS.within(e, inv, len(uniq), t, detrend=True)
    ct = BS.within(c, inv, len(uniq), t, detrend=True)
    return {"PR": BS.rmse(et - ct, wells), "ZR": BS.rmse(et, wells)}


def score_direction(rows: dict, blocks: dict, informative: dict, *,
                    form: str = K.CHX_PRIMARY_FORM, Q: np.ndarray | None = None,
                    B: int = BS.B_BOOT, seed: int = BS.SEED, report: bool = True) -> dict:
    """I1.CHX.v1's primary on long rows (``long_panel``/``eligible``): e = obs - ZR and c
    transformed under ``form`` (``transform``), the pinned per-well FE slope of e~ on c~
    (``beta_response``; its per-well demeaning is then a no-op) and its 90 % 5 km block
    interval, ``decide()`` fed with them and RMSE(PR), RMSE(ZR) on the same transformed rows;
    'direction' = verdict in DIRECTION_VERDICTS. ``report`` adds the report-only forms (the
    per-well detrended slope, the demeaned slope = N2C's own form, the level component)."""
    w = rows["well"]
    if len(w) == 0:
        raise ValueError("no scored row")
    miss = sorted(set(w) - set(blocks))
    if miss:
        raise ValueError(f"wells without a block: {miss[:5]}")
    e = rows["obs"] - rows["zr"]
    c = rows["c"]
    et, ct = transform(e, c, rows, form, Q)
    br = BS.beta_response(et, ct, w, blocks, B=B, seed=seed)
    br["form"] = form
    rm = {"PR": BS.rmse(et - ct, w), "ZR": BS.rmse(et, w)}
    lo, hi = br["ci90"]
    verdict = BS.decide(bool(informative["passes"]), br["beta"], lo, hi,
                        rm["PR"]["equal_well"], rm["ZR"]["equal_well"])
    out = {"verdict": verdict, "direction": verdict in K.DIRECTION_VERDICTS, "form": form,
           "beta_primary": br, "rmse_primary_m": rm,
           "rmse_pr_below_zr": bool(rm["PR"]["equal_well"] < rm["ZR"]["equal_well"]),
           "ratio_band_point": bool(1 / BS.RATIO_BAND <= br["beta"] <= BS.RATIO_BAND)
           if np.isfinite(br["beta"]) else False,
           "interval_below_zero_with_positive_beta":
               BS.interval_below_zero_with_positive_beta(br["beta"], lo, hi),
           "informative_gate": informative, "n_wells": br["n_wells"], "n_rows": int(len(w))}
    if report:
        t = rows["t"]
        out["report_only_detrended"] = BS.beta_response(e, c, w, blocks, t=t, detrend=True,
                                                        B=B, seed=seed)
        out["report_only_demeaned"] = BS.beta_response(e, c, w, blocks, detrend=False, B=B,
                                                       seed=seed)
        out["report_only_level_component"] = BS.level_component(e, c, w)
    return out


def informative_from_contrast(c_wide: np.ndarray, sids, blocks: dict, months,
                              form: str = K.CHX_PRIMARY_FORM) -> dict:
    """The pinned informative gate (RMS of the well-demeaned contrast >= 0.05 m, >= 5 blocks,
    >= 6 months) applied to the contrast AFTER the primary's transform ``form``, every screen x
    window month (no head): the variation that identifies the slope. Report only beside it:
    the pinned gate on the untransformed contrast."""
    c_wide = np.asarray(c_wide, float)
    W, T = c_wide.shape
    months = [str(m) for m in months]
    if len(months) != T:
        raise ValueError("contrast and months differ")
    rows = {"well": np.repeat(np.asarray([str(s) for s in sids]), T),
            "month": np.tile(np.asarray(months), W), "t": np.tile(month_t(months), W)}
    cv = c_wide.ravel()
    ct = transform(cv, cv, rows, form)[1]
    g = BS.informative_gate(ct, rows["well"], blocks, T)
    raw = BS.informative_gate(cv, rows["well"], blocks, T)
    g.update({"form": form, "transformed": True,
              "report_only_untransformed_rms_m": raw["rms_well_demeaned_contrast_m"],
              "report_only_untransformed_passes": raw["passes"]})
    return g


# ----------------------------------------------------------------------------- G0 worlds
def hinge(months: list[str], brk: str = K.TREND_BREAK, amp: float = K.TREND_AMP_M
          ) -> np.ndarray:
    """0 up to ``brk``, then linear to ``amp`` at the last month."""
    t = month_t(months)
    tb = month_t([brk])[0]
    span = t[-1] - tb
    if span <= 0:
        raise ValueError("the break is not inside the axis")
    return amp * np.clip(t - tb, 0.0, None) / span


def trend_sign(c: np.ndarray, mask: np.ndarray, h: np.ndarray) -> float:
    """Sign of the common trend that biases the per-well detrended slope upward: sign of sum
    over observed well-months of c~ h~ (both detrended per well); +1 on a tie. (The two-way
    form absorbs any common trend exactly; the sign then does not matter.)"""
    W, T = c.shape
    tt = np.arange(T, dtype=float)
    num = 0.0
    for i in range(W):
        m = mask[i]
        if m.sum() < 3:
            continue
        x = np.vstack([np.ones(m.sum()), tt[m]]).T
        rc = c[i, m] - x @ np.linalg.lstsq(x, c[i, m], rcond=None)[0]
        rh = h[m] - x @ np.linalg.lstsq(x, h[m], rcond=None)[0]
        num += float(rc @ rh)
    return 1.0 if num >= 0 else -1.0


def load_z(c: np.ndarray) -> np.ndarray:
    """Standardised per-screen RMS of the well-demeaned contrast (zeros if constant)."""
    c = np.asarray(c, float)
    amp = np.sqrt(np.mean((c - c.mean(axis=1, keepdims=True)) ** 2, axis=1))
    sd = amp.std()
    return (amp - amp.mean()) / sd if sd > 0 else np.zeros_like(amp)


def ar1(rng, n: int, T: int, s: float, phi: float) -> np.ndarray:
    """Stationary AR(1) rows (synth_n2c.ar1's recursion)."""
    eps = rng.standard_normal((n, T))
    x = np.empty((n, T))
    x[:, 0] = s * eps[:, 0]
    q = math.sqrt(1.0 - phi * phi) * s
    for k in range(1, T):
        x[:, k] = phi * x[:, k - 1] + q * eps[:, k]
    return x


def confounder(rng, a: np.ndarray, rho: float) -> np.ndarray:
    """Unit-sd common series with correlation ``rho`` to ``a`` (synth_n2c.confounder)."""
    a = np.asarray(a, float) - np.mean(a)
    sd = a.std()
    a = a / sd if sd > 0 else a
    xi = rng.standard_normal(a.size)
    xi -= xi.mean()
    if sd > 0:
        xi -= (xi @ a) / (a @ a) * a
    xi /= xi.std()
    return rho * a + math.sqrt(1.0 - rho * rho) * xi


def world(panel: dict, k: float, rho: float, rng, params: dict, *, trend_mult: float = 1.0,
          het_mult: float = K.G0_HET_MULT, het_corr: float = K.G0_HET_CORR,
          kappa_mult: float = 1.0, noise_scale: float = 1.0) -> np.ndarray:
    """One synthetic e = obs - ZR, ``(W, T)``, NaN outside ``panel['mask']``:
    k c + block AR(1) + well AR(1) + kappa g_t (1 + het_mult z_i) + trend_mult s h_t, with g
    the common confounder (corr rho with the Changhua anomaly), z_i = het_corr
    panel['load_z']_i + sqrt(1 - het_corr^2) N(0, 1) and s h the common-trend hinge (sign
    ``panel['trend_sign']``). Draw order: block AR(1), well AR(1), g, z."""
    c, mask = panel["c"], panel["mask"]
    W, T = c.shape
    ub, binv = np.unique(panel["block"], return_inverse=True)
    sig, share, phi = params["sigma_tot_m"], params["block_share"], params["phi"]
    b = ar1(rng, len(ub), T, noise_scale * sig * math.sqrt(share), phi)[binv]
    wn = ar1(rng, W, T, noise_scale * sig * math.sqrt(1.0 - share), phi)
    g = confounder(rng, panel["a_ch"], rho)
    z = rng.standard_normal(W)
    if het_corr:
        z = het_corr * np.asarray(panel["load_z"], float) + math.sqrt(1.0 - het_corr ** 2) * z
    load = 1.0 + het_mult * z
    e = k * c + b + wn + kappa_mult * params["kappa_m"] * load[:, None] * g[None, :]
    if trend_mult:
        e = e + trend_mult * panel["trend_sign"] * panel["hinge"][None, :]
    return np.where(mask, e, np.nan)


def _panel_rows(panel: dict, e: np.ndarray) -> dict:
    W, T = e.shape
    return long_panel(panel["sid"], panel["months"], e, np.zeros((W, T)), panel["c"])


def _template(panel: dict, form: str) -> tuple[dict, np.ndarray | None]:
    """The rows the mask yields (identical in every world) and their fixed-effect basis."""
    r = eligible(_panel_rows(panel, np.where(panel["mask"], 0.0, np.nan)))
    return r, (None if form == "detrended" else fe_basis(r["well"], r["month"], form))


def g0_cell(panel: dict, params: dict, k: float, rho: float, rng, reps: int, *,
            trend_mult: float = 1.0, het_mult: float = K.G0_HET_MULT,
            het_corr: float = K.G0_HET_CORR, kappa_mult: float = 1.0,
            noise_scale: float = 1.0, form: str = K.CHX_PRIMARY_FORM,
            B: int = BS.B_BOOT) -> dict:
    """``reps`` worlds scored by the primary; P(direction), P(response_supported), median
    slope and the 90 % interval coverage of k (report)."""
    blocks = dict(zip(panel["sid"], panel["block"], strict=True))
    tmpl, Q = _template(panel, form)
    n_dir = n_rs = cover = 0
    betas = []
    for _ in range(reps):
        e = world(panel, k, rho, rng, params, trend_mult=trend_mult, het_mult=het_mult,
                  het_corr=het_corr, kappa_mult=kappa_mult, noise_scale=noise_scale)
        r = eligible(_panel_rows(panel, e))
        if len(r["well"]) != len(tmpl["well"]):
            raise RuntimeError("a world's rows differ from the mask's")
        s = score_direction(r, blocks, panel["informative"], form=form, Q=Q, B=B,
                            report=False)
        n_dir += s["direction"]
        n_rs += s["verdict"] == "response_supported"
        b = s["beta_primary"]
        betas.append(b["beta"])
        cover += bool(b["ci90"][0] <= k <= b["ci90"][1])
    return {"k": k, "rho": rho, "trend_mult": trend_mult, "het_mult": het_mult,
            "het_corr": het_corr, "kappa_mult": kappa_mult, "noise_scale": noise_scale,
            "form": form, "replicates": reps, "p_direction": n_dir / reps,
            "p_response_supported": n_rs / reps, "median_beta": float(np.nanmedian(betas)),
            "coverage_k": cover / reps}


def plumbing_exact(panel: dict, ks=(0.5, 1.0, 2.0), form: str = K.CHX_PRIMARY_FORM) -> dict:
    """No noise, no confounder, no trend: the primary's slope must equal k exactly."""
    tmpl, Q = _template(panel, form)
    out = {}
    for k in ks:
        e = np.where(panel["mask"], k * panel["c"], np.nan)
        r = eligible(_panel_rows(panel, e))
        et, ct = transform(r["obs"] - r["zr"], r["c"], r, form, Q)
        out[str(k)] = BS.slope_fe(et, ct, r["well"])
    ok = all(np.isfinite(v) and abs(v - float(k)) <= K.G0_PLUMBING_TOL
             for k, v in out.items())
    return {"betas": out, "ok": bool(ok), "tol": K.G0_PLUMBING_TOL, "form": form,
            "n_rows": int(len(tmpl["well"]))}


def run_g0(panel: dict, params: dict, *, reps: int = K.G0_REPLICATES, seed: int = K.G0_SEED,
           B: int = BS.B_BOOT, form: str = K.CHX_PRIMARY_FORM) -> dict:
    """The declared G0. Order: (1) the informative gate on R1c's transformed contrast
    (``panel['informative']``): if it fails, no cell is run and the reading is
    'not_informative' (the scorer then reports not_informative without reading a head);
    (2) exact plumbing; (3) the grid k in G0_K x rho in G0_RHO (trend on, heterogeneous
    loading G0_HET_MULT / G0_HET_CORR), k outer, rho inner, then the report-only cells
    G0_REPORT_CELLS, generators SeedSequence(seed).spawn(n_grid + n_report) in that order."""
    inf = panel["informative"]
    plumb = plumbing_exact(panel, form=form)
    if not plumb["ok"]:
        return {"plumbing": plumb, "engineering_failure": True, "cells": [],
                "informative": inf}
    if not inf["passes"]:
        return {"plumbing": plumb, "engineering_failure": False, "cells": [],
                "report_only_cells": [], "informative": inf,
                "gate": g0_gate([], informative=False)}
    grid = [(k, r) for k in K.G0_K for r in K.G0_RHO]
    ss = np.random.SeedSequence(seed).spawn(len(grid) + len(K.G0_REPORT_CELLS))
    cells = [g0_cell(panel, params, k, r, np.random.default_rng(ss[i]), reps, form=form, B=B)
             for i, (k, r) in enumerate(grid)]
    report = []
    for j, rc in enumerate(K.G0_REPORT_CELLS):
        kw = {x: rc[x] for x in ("trend_mult", "het_corr") if x in rc}
        report.append(g0_cell(panel, params, rc["k"], rc["rho"],
                              np.random.default_rng(ss[len(grid) + j]), reps, form=form, B=B,
                              **kw))
    return {"plumbing": plumb, "engineering_failure": False, "cells": cells,
            "report_only_cells": report, "informative": inf, "gate": g0_gate(cells)}


def g0_gate(cells: list[dict], informative: bool = True) -> dict:
    """Decidable iff R1c's contrast is informative, P(direction | k = 1, rho = -0.83) >=
    G0_POWER_MIN and the largest P(direction | k = 0) over rho <= G0_FALSE_MAX (grid cells:
    trend on, declared heterogeneous loading)."""
    if not informative:
        return {"informative": False, "decidable": False, "power_k1_rho083": None,
                "false_pass_k0_max_rho": None, "power_min": K.G0_POWER_MIN,
                "false_max": K.G0_FALSE_MAX,
                "reading": "not_informative: R1c's transformed contrast fails the pinned "
                           "informative gate; G0 not run; no observation is opened"}

    def p(k, r):
        for c in cells:
            if (c["trend_mult"] == 1.0 and c["het_corr"] == K.G0_HET_CORR
                    and abs(c["k"] - k) < 1e-12 and abs(c["rho"] - r) < 1e-12):
                return c["p_direction"]
        raise KeyError((k, r))
    power = p(1.0, -0.83)
    false = max(p(0.0, r) for r in K.G0_RHO)
    ok = power >= K.G0_POWER_MIN and false <= K.G0_FALSE_MAX
    return {"informative": True, "power_k1_rho083": power, "false_pass_k0_max_rho": false,
            "power_min": K.G0_POWER_MIN, "false_max": K.G0_FALSE_MAX,
            "decidable": bool(ok),
            "reading": "decidable: the primary may be scored" if ok else
                       "not decidable: stop; no observation is opened"}


def g0_mask(n_months: np.ndarray, T: int) -> np.ndarray:
    """Declared availability approximation: each screen observed on the LAST min(n, T)
    window months (n = N3F's logged pre-origin month count)."""
    n = np.clip(np.asarray(n_months, int), 0, T)
    m = np.zeros((len(n), T), bool)
    for i, v in enumerate(n):
        if v:
            m[i, T - v:] = True
    return m


# ----------------------------------------------------------------------------- CH-J gates
def seasonal_amplitude(series: np.ndarray, months: list[str], smooth2: bool = True) -> float:
    """Peak-to-trough of the calendar-month means of the per-series detrended values
    (linear in time); with ``smooth2`` the 12 means are first replaced by the circular
    2-month mean of each month and the next (the census's bimonthly even split blurs a
    one-month phase). NaN if any calendar month has no value."""
    v = np.asarray(series, float)
    t = month_t(months)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return float("nan")
    x = np.vstack([np.ones(ok.sum()), t[ok]]).T
    r = v[ok] - x @ np.linalg.lstsq(x, v[ok], rcond=None)[0]
    mo = np.array([int(m[5:7]) for m in months])[ok]
    clim = np.array([r[mo == m].mean() if (mo == m).any() else np.nan for m in range(1, 13)])
    if not np.isfinite(clim).all():
        return float("nan")
    if smooth2:
        clim = 0.5 * (clim + np.roll(clim, -1))
    return float(clim.max() - clim.min())


def amplitude_ratios(model: np.ndarray, obs: np.ndarray, months: list[str]) -> np.ndarray:
    """Per-well model/observed seasonal amplitude on the observed months only."""
    out = []
    for m, o in zip(np.asarray(model, float), np.asarray(obs, float), strict=True):
        mm = np.where(np.isfinite(o), m, np.nan)
        a_o, a_m = seasonal_amplitude(o, months), seasonal_amplitude(mm, months)
        out.append(a_m / a_o if np.isfinite(a_o) and a_o > 0 and np.isfinite(a_m)
                   else np.nan)
    return np.array(out)


def county_mean_corr(model: np.ndarray, obs: np.ndarray) -> float:
    """Correlation of the county-mean per-well-demeaned monthly anomaly, model vs observed,
    over months where at least one well is observed (model taken on observed cells only)."""
    o = np.asarray(obs, float)
    m = np.where(np.isfinite(o), np.asarray(model, float), np.nan)
    o = o - np.nanmean(o, axis=1, keepdims=True)
    m = m - np.nanmean(m, axis=1, keepdims=True)
    ok = np.isfinite(o).any(axis=0)
    if ok.sum() < 3:
        return float("nan")
    mo, mm = np.nanmean(o[:, ok], axis=0), np.nanmean(m[:, ok], axis=0)
    if mo.std() == 0 or mm.std() == 0:
        return float("nan")
    return float(np.corrcoef(mo, mm)[0, 1])


def in_band(v: float, band=K.CHJ_AMP_BAND) -> bool:
    return bool(np.isfinite(v) and band[0] <= v <= band[1])


def no_harm(new: float, ref: float, tol: float) -> bool:
    return bool(np.isfinite(new) and np.isfinite(ref) and new >= ref - tol)


def chj_external_gates(ch_model, ch_obs, yl_model, yl_obs, months, ch_q0_model,
                       direction: dict) -> dict:
    """CH-J's gates at the externals: median CH amplitude ratio in the band; CH/YL ratio of
    medians in the band; county-mean correlation >= Q0's - 0.05; direction by the primary
    estimator (``direction`` = ``score_direction`` on CH-J's contrast)."""
    r_ch = amplitude_ratios(ch_model, ch_obs, months)
    r_yl = amplitude_ratios(yl_model, yl_obs, months)
    med_ch, med_yl = float(np.nanmedian(r_ch)), float(np.nanmedian(r_yl))
    cyr = med_ch / med_yl if np.isfinite(med_yl) and med_yl > 0 else float("nan")
    corr, corr_q0 = county_mean_corr(ch_model, ch_obs), county_mean_corr(ch_q0_model, ch_obs)
    g = {"amplitude_ratio_median_ch": med_ch, "amplitude_band": list(K.CHJ_AMP_BAND),
         "amplitude_ok": in_band(med_ch),
         "ch_over_yl_ratio": cyr, "ch_over_yl_ok": in_band(cyr),
         "county_mean_corr": corr, "county_mean_corr_q0": corr_q0,
         "county_mean_corr_ok": no_harm(corr, corr_q0, K.CHJ_NOHARM["county_mean_corr_ext"]),
         "direction_verdict": direction["verdict"], "direction_ok": bool(direction["direction"]),
         "n_ch_ratios": int(np.isfinite(r_ch).sum()), "n_yl_ratios": int(np.isfinite(r_yl).sum())}
    g["passed"] = bool(g["amplitude_ok"] and g["ch_over_yl_ok"] and g["county_mean_corr_ok"]
                       and g["direction_ok"])
    return g


def chj_noharm_gates(m: dict, q0: dict) -> dict:
    """No-harm on spent units (design-time): Yunlin k-fold and county_corr >= Q0 - 0.01;
    out-of-fold leveling >= Q0 - 0.02; Changhua leveling not below Q0's."""
    t = K.CHJ_NOHARM
    g = {"yunlin_kfold_ok": no_harm(m["yunlin_kfold"], q0["yunlin_kfold"], t["yunlin_kfold"]),
         "yunlin_county_corr_ok": no_harm(m["yunlin_county_corr"], q0["yunlin_county_corr"],
                                          t["yunlin_county_corr"]),
         "oof_leveling_ok": no_harm(m["oof_leveling"], q0["oof_leveling"], t["oof_leveling"]),
         "changhua_leveling_ok": no_harm(m["changhua_leveling"], q0["changhua_leveling"], 0.0)}
    g["passed"] = all(g.values())
    return g


def chj_forward_gates(m: dict, q0: dict) -> dict:
    """Model-only forward plausibility: 2023-25 county-mean subsidence >= 0 in both counties;
    Changhua rising share <= Q0's."""
    g = {"subsidence_ch_ok": bool(m["subsidence_2023_25_changhua_m"] >= 0.0),
         "subsidence_yl_ok": bool(m["subsidence_2023_25_yunlin_m"] >= 0.0),
         "rising_share_ok": bool(m["rising_share_changhua"] <= q0["rising_share_changhua"])}
    g["passed"] = all(g.values())
    return g


def chj_sw_outcome(fraction_ch: float, bound: float = K.CHJ_SW_FRACTION_BOUND,
                   rtol: float = 1e-3) -> str:
    """'co_located_recharge_rejected' when the fitted Changhua sw fraction ends at its lower
    bound (a registered outcome, not an engineering retry), else 'off_bound'."""
    return ("co_located_recharge_rejected" if fraction_ch <= bound * (1.0 + rtol)
            else "off_bound")


# ----------------------------------------------------------------------------- backcast
def score_backcast(rows: dict, blocks: dict, informative: dict, *,
                   form: str = K.BC_PRIMARY_FORM, B: int = BS.B_BOOT, seed: int = BS.SEED,
                   report: bool = True) -> dict:
    """I1.CH_backcast.v1's primary: e = obs - ZR and c residualised on well x calendar-month
    AND month-of-sample effects (``fe_basis(form='calmonth_twoway')``), then the pinned
    per-well FE slope (its demeaning is a no-op) and its 90 % 5 km block interval. With two
    years the slope is identified from differences ACROSS wells in the 2011 - 2010 change of
    R1c's contrast, per calendar month; a rain or drought anomaly common to every well is
    absorbed. Pass iff the informative gate (on the transformed contrast) passes and the
    interval lies above 0 (lo > 0). Report only: ``decide()``'s verdict, the well x
    calendar-month-only slope (the v0 form, which a common year shock confounds) and the
    per-well detrended slope."""
    w = rows["well"]
    e, c = rows["obs"] - rows["zr"], rows["c"]
    et, ct = transform(e, c, rows, form)
    br = BS.beta_response(et, ct, w, blocks, B=B, seed=seed)
    br["form"] = form
    lo, hi = br["ci90"]
    inf = bool(informative["passes"])
    rm = {"PR": BS.rmse(et - ct, w), "ZR": BS.rmse(et, w)}
    passed = bool(inf and np.isfinite(lo) and lo > 0.0 and br["beta"] > 0.0)
    verdict = ("not_informative" if not inf else
               "direction_supported" if passed else "not_supported")
    out = {"verdict": verdict, "passed": passed, "form": form, "beta_primary": br,
           "rmse_primary_m": rm,
           "decide_verdict_report_only": BS.decide(inf, br["beta"], lo, hi,
                                                   rm["PR"]["equal_well"],
                                                   rm["ZR"]["equal_well"]),
           "informative_gate": informative, "n_wells": br["n_wells"], "n_rows": int(len(w))}
    if report:
        e0, c0 = transform(e, c, rows, "calmonth")
        out["report_only_calmonth_only"] = BS.beta_response(e0, c0, w, blocks, B=B, seed=seed)
        out["report_only_detrended"] = BS.beta_response(e, c, w, blocks, t=rows["t"],
                                                        detrend=True, B=B, seed=seed)
    return out


def bc_power(P: dict, params: dict, informative: dict, *, reps: int = K.BC_POWER_REPS,
             seed: int = K.BC_POWER_SEED, B: int = BS.B_BOOT) -> dict:
    """Report only (never gates): the backcast primary's synthetic rates on its own panel
    (model only, before any head): P(direction | k) for k = 0 and 1 at rho = -0.83 with G0's
    noise, confounder and heterogeneous loading, no trend, every registered well observed in
    every window month (an upper bound on availability)."""
    months = [str(m) for m in P["months"]]
    c = np.asarray(P["c"], float)
    pan = {"c": c, "mask": np.ones(c.shape, bool), "a_ch": np.asarray(P["a_ch"], float),
           "block": np.asarray(P["block"]).astype(str), "sid": np.asarray(P["sid"]).astype(str),
           "months": months, "hinge": np.zeros(len(months)), "trend_sign": 1.0,
           "load_z": load_z(c)}
    blocks = dict(zip(pan["sid"], pan["block"], strict=True))
    ss = np.random.SeedSequence(seed).spawn(2)
    out = {}
    for i, k in enumerate((0.0, 1.0)):
        rng = np.random.default_rng(ss[i])
        n = 0
        for _ in range(reps):
            e = world(pan, k, -0.83, rng, params, trend_mult=0.0)
            r = eligible(_panel_rows(pan, e), min_months=K.BC_MIN_MONTHS)
            n += score_backcast(r, blocks, informative, B=B, report=False)["passed"]
        out[f"p_direction_k{k:g}"] = n / reps
    out.update({"reps": reps, "seed": seed, "rho": -0.83, "het_mult": K.G0_HET_MULT,
                "het_corr": K.G0_HET_CORR, "mask": "every registered well x month",
                "report_only": True})
    return out
