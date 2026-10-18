"""Repair round 1: the uniform level-shift screen ``H3.step_screen`` (registered in
``H3.repair.v1``).

Why. A recovered well's record has one long internal gap (12 of the 15 admitted wells lost
2019-01/02..2019-07/12, the transition from 60-minute to 10-minute records, which is why the
twin's 180-day gap rule excluded them). H1a's offset detector needs samples < 70 min apart, so
it cannot see a datum step across a multi-month gap. In the first round only 09030212 was
checked for such a step, with a threshold (|bias| > 2 x LOYO RMSE) written after its result
was seen. This module applies one procedure to every well and calibrates the threshold on a
null population instead of fixing it by hand.

Procedure for a target well with pre-split months P and post-split months Q (both C20
months, flag 0/1, of L1 v3; the split is the well's longest internal C20 gap, ties to the
later gap; for 09030212 the split is its switch to once-daily readings, Q = its screened
once-daily months):

* neighbour pool: the 158 model wells (continuous records across 2019, screened by H1a's
  offset detector), minus the target and every well of the target's site (siblings share the
  site's instrument history, so a common step would be invisible);
* candidates within 25 km (L1 repaired coordinates), >= 24 months in common with the target
  in P, observed in >= 80 % of Q; Pearson r over the common P months >= 0.6; the 4 with the
  largest r (ties: nearer first);
* a ridge regression of the target on them (``s7b_snapshot_consistency.ridge_fit``: alpha
  1e-3, standardised predictors, intercept) fitted on the P months where all neighbours are
  observed (>= 24 needed); leave-one-calendar-year-out RMSE over those months (``rmse_loyo``);
* residuals on the Q months where all neighbours are observed (>= 12 needed); ``bias`` =
  their mean; statistic ``z = |bias| / rmse_loyo``;
* untestable when fewer than 2 neighbours qualify or a month minimum is not met.

Null: the same procedure with the same calendar split applied to every model well as a
pseudo-target (its own site excluded from its neighbours). p = (1 + #{null z >= z}) /
(1 + n_null); the target is flagged when p <= 0.05 (n_null >= 30 needed, else untestable).
No head value after 2022-12 is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory, snapshot  # noqa: E402
from h3_recovered_wells.s7b_snapshot_consistency import ridge_fit  # noqa: E402

RADIUS_M = 25000.0
MIN_COMMON_PRE = 24
MIN_POST_COVER = 0.80
MIN_R = 0.6
N_NEIGHBOURS = 4
MIN_NEIGHBOURS = 2
MIN_FIT = 24
MIN_POST = 12
ALPHA_LEVEL = 0.05
MIN_NULL = 30
DATES = pd.date_range(f"{C.START}-01", f"{C.PRE_END}-01", freq="MS")


def c20_matrix() -> pd.DataFrame:
    """L1 v3 C20 values (flag 0/1), months 2012-01..2022-12 x wells."""
    m = snapshot.l1_monthly()
    Y = m.pivot_table(index="date", columns="sid", values="value", aggfunc="first")
    return Y.reindex(DATES)


def longest_gap(obs: np.ndarray) -> tuple[int, int] | None:
    """(first, last) month index of the longest internal run of unobserved months; ties go
    to the later run; None if the record has no internal gap."""
    idx = np.where(obs)[0]
    if len(idx) == 0:
        return None
    runs = [(a, b) for a, b in inventory.runs_of(~obs) if a > idx[0] and b <= idx[-1]]
    if not runs:
        return None
    a, b = max(runs, key=lambda r: (r[1] - r[0], r[0]))
    return int(a), int(b - 1)


def _neighbours(Y: pd.DataFrame, y: pd.Series, pre: pd.DatetimeIndex, post: pd.DatetimeIndex,
                pool: list[str], meta: pd.DataFrame, target: str) -> list[tuple]:
    t = meta.loc[target]
    out = []
    for s in pool:
        if s == target or s not in Y.columns or s not in meta.index:
            continue
        m = meta.loc[s]
        if isinstance(t.site_id, str) and t.site_id and m.site_id == t.site_id:
            continue
        if not (np.isfinite(m.x_twd97) and np.isfinite(t.x_twd97)):
            continue
        dist = float(np.hypot(m.x_twd97 - t.x_twd97, m.y_twd97 - t.y_twd97))
        if dist > RADIUS_M:
            continue
        both = pd.concat([y.reindex(pre), Y[s].reindex(pre)], axis=1).dropna()
        if len(both) < MIN_COMMON_PRE:
            continue
        if Y[s].reindex(post).notna().mean() < MIN_POST_COVER:
            continue
        r = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        if np.isfinite(r) and r >= MIN_R:
            out.append((r, -dist, s, dist))
    out.sort(reverse=True)
    return out[:N_NEIGHBOURS]


def screen_one(Y: pd.DataFrame, y: pd.Series, pre: pd.DatetimeIndex, post: pd.DatetimeIndex,
               pool: list[str], meta: pd.DataFrame, target: str) -> dict:
    """The step statistic of one target (``y``: its monthly values indexed by month)."""
    rec = {"sid": target, "n_pre_months": int(len(pre)), "n_post_months": int(len(post)),
           "testable": False}
    nb = _neighbours(Y, y, pre, post, pool, meta, target)
    rec["neighbours"] = [s for _, _, s, _ in nb]
    rec["neighbour_r"] = [round(r, 4) for r, _, _, _ in nb]
    rec["neighbour_km"] = [round(d / 1000, 2) for _, _, _, d in nb]
    if len(nb) < MIN_NEIGHBOURS:
        rec["reason"] = f"{len(nb)} neighbours qualify (< {MIN_NEIGHBOURS})"
        return rec
    cols = rec["neighbours"]
    fit = pd.concat([y.reindex(pre).rename("_y"), Y[cols].reindex(pre)], axis=1).dropna()
    ev = pd.concat([y.reindex(post).rename("_y"), Y[cols].reindex(post)], axis=1).dropna()
    rec["n_fit"], rec["n_eval"] = int(len(fit)), int(len(ev))
    if len(fit) < MIN_FIT or len(ev) < MIN_POST:
        rec["reason"] = f"fit months {len(fit)} (< {MIN_FIT}) or post months {len(ev)} " \
                        f"(< {MIN_POST})"
        return rec
    years = sorted(set(fit.index.year))
    if len(years) < 2:
        rec["reason"] = "fewer than 2 calendar years to leave out"
        return rec
    loyo = []
    for yr in years:
        a, b = fit[fit.index.year != yr], fit[fit.index.year == yr]
        f = ridge_fit(a[cols].to_numpy(), a["_y"].to_numpy())
        loyo.extend(b["_y"].to_numpy() - f(b[cols].to_numpy()))
    loyo = np.asarray(loyo, float)
    f = ridge_fit(fit[cols].to_numpy(), fit["_y"].to_numpy())
    res = ev["_y"].to_numpy() - f(ev[cols].to_numpy())
    rmse = float(np.sqrt(np.mean(loyo ** 2)))
    bias = float(np.mean(res))
    rec.update(testable=True, rmse_loyo=rmse, bias_loyo=float(np.mean(loyo)), bias=bias,
               rmse_post=float(np.sqrt(np.mean(res ** 2))),
               z=float(abs(bias) / rmse) if rmse > 0 else float("inf"))
    return rec


def split_months(obs_dates: pd.DatetimeIndex, gap: tuple[int, int]) -> tuple:
    """Pre and post months of a split given as (first, last) gap month index (an empty gap
    has last = first - 1)."""
    g0, g1 = DATES[gap[0]], DATES[gap[1]] if gap[1] >= 0 else DATES[0] - pd.offsets.MonthBegin()
    pre = obs_dates[obs_dates < g0]
    post = obs_dates[obs_dates > g1]
    return pre, post


def null_for_split(Y: pd.DataFrame, gap: tuple[int, int], model: list[str],
                   meta: pd.DataFrame) -> pd.DataFrame:
    """The screen applied to every model well as a pseudo-target with the same split."""
    rows = []
    for m in model:
        if m not in Y.columns or m not in meta.index:
            continue
        obs = Y[m].dropna().index
        pre, post = split_months(obs, gap)
        r = screen_one(Y, Y[m], pre, post, model, meta, m)
        rows.append(r)
    return pd.DataFrame(rows)


def p_value(z: float, null_z: np.ndarray) -> float:
    return float((1 + np.sum(null_z >= z)) / (1 + len(null_z)))
