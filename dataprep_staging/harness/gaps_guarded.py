"""Origin-guarded wrappers of the index- and label-based gap generators (A1 repair round).

``harness.gaps`` stays frozen (heads/L1_clean v1 and H1a.qc_acceptance.v1 hash it). Its
year-based generators (families D, J, X, K, W and the season blocks) already raise
:class:`harness.gaps.LeakageError` for years after the origin unless
``allow_post_origin=True``. The generators that take month indices or a caller's matrix did
not check anything: the head families N, S, E, H, the test-period replica T (its docstring
example starts the window at 2023-01), the offset/step injection O, the GNSS steps G, the ring
windows R, the ring-path years Y and the withheld leveling years L. The wrappers here take the
calendar of the matrix (``months``: one ``YYYY-MM`` label, date or period per column; or the
epoch labels) and raise unless every column, window and endpoint is at or before the origin
(2022-12), or ``allow_post_origin=True`` (after-the-fact analysis only). Masks, cells, events
and seeds are exactly those of ``harness.gaps``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common.paths import ORIGIN

from harness import gaps
from harness.gaps import ORIGIN_YEAR, LeakageError

_ORIGIN_P = pd.Period(ORIGIN, freq="M")


def _periods(labels) -> pd.PeriodIndex:
    return pd.PeriodIndex([pd.Period(x, freq="M") for x in labels], freq="M")


def check_months(labels, allow_post_origin: bool, what: str) -> pd.PeriodIndex:
    """Month periods of ``labels``; raises LeakageError if any is after the origin."""
    p = _periods(labels)
    late = p[p > _ORIGIN_P]
    if len(late) and not allow_post_origin:
        raise LeakageError(f"{what}: {len(late)} month(s) after the origin {ORIGIN} (first "
                           f"{late[0]}) (U3/U7); pass allow_post_origin=True for after-the-"
                           f"fact analysis only")
    return p


def _check_matrix(obs, months, allow_post_origin, what):
    n = np.asarray(obs).shape[1]
    if months is None or len(months) != n:
        raise ValueError(f"{what}: months must label each of the {n} columns")
    return check_months(months, allow_post_origin, what)


def head_masks(family: str, obs, targets, months, nreal: int | None = None, *,
               allow_post_origin: bool = False):
    """Families N, S, E, H (``gaps.head_masks``) on a matrix whose columns ``months`` label;
    every column must be at or before the origin."""
    _check_matrix(obs, months, allow_post_origin, f"head family {family}")
    yield from gaps.head_masks(family, obs, targets, nreal)


def replica_test_period(obs, targets, r: int, window: tuple[int, int], months, *,
                        allow_post_origin: bool = False, **kw):
    """Family T (``gaps.replica_test_period``): the matrix and the window must end by the
    origin."""
    p = _check_matrix(obs, months, allow_post_origin, "test-period replica matrix")
    lo, hi = window
    check_months([str(x) for x in p[lo:hi]], allow_post_origin, "test-period replica window")
    return gaps.replica_test_period(obs, targets, r, window, **kw)


def inject_offsets_steps(Y, r: int, months, *, allow_post_origin: bool = False, **kw):
    """Family O (``gaps.inject_offsets_steps``) on series whose samples ``months`` label
    (monthly labels, or daily dates: their months are checked)."""
    _check_matrix(Y, months, allow_post_origin, "offset/step injection")
    return gaps.inject_offsets_steps(Y, r, **kw)


def inject_gnss_steps(epochs, r: int, *, allow_post_origin: bool = False, **kw):
    """Family G (``gaps.inject_gnss_steps``) for the epochs ``epochs`` (dates)."""
    check_months(list(epochs), allow_post_origin, "GNSS step injection")
    return gaps.inject_gnss_steps(len(epochs), r, **kw)


def ring_windows(start: str = "2015-01", end: str = "2021-12", length: int = 13, *,
                 allow_post_origin: bool = False) -> list[dict]:
    """Family R windows (``gaps.ring_windows``); ``end`` must be at or before the origin."""
    check_months([start, end], allow_post_origin, "ring windows")
    return gaps.ring_windows(start, end, length)


def ring_path_years(years=range(2017, 2022), *, allow_post_origin: bool = False) -> list[dict]:
    """Family Y (``gaps.ring_path_years``); the right endpoint (January after the masked year)
    must be at or before the origin, so the last pre-origin masked year is 2021."""
    ys = [int(y) for y in years]
    late = [y for y in ys if y + 1 > ORIGIN_YEAR]
    if late and not allow_post_origin:
        raise LeakageError(f"ring path years {late}: the right endpoint lies after the origin "
                           f"{ORIGIN}; pass allow_post_origin=True for after-the-fact analysis")
    return gaps.ring_path_years(ys)


def leveling_withheld(years=(2020, 2021), *, allow_post_origin: bool = False) -> dict:
    """Family L (``gaps.leveling_withheld``); years must be at or before the origin year."""
    ys = gaps._check_years(years, allow_post_origin, "withheld leveling years")
    return gaps.leveling_withheld(ys)


#: Where each plan 4.4 family is origin-guarded after the repair round.
GUARDS = {
    "D": "gaps.daily_outage_windows (years)", "J": "gaps.rolling_origin_july (years)",
    "X": "gaps.leave_one_year_out / gaps.season_blocks (years)",
    "K": "gaps.matched_lead_backcast (years)", "W": "gaps.canal_loyo (years, no override)",
    "C": "fixed pre-origin design (freeze 2013-06, predict 2014-16)",
    "N": "gaps_guarded.head_masks (months)", "S": "gaps_guarded.head_masks (months)",
    "E": "gaps_guarded.head_masks (months)", "H": "gaps_guarded.head_masks (months)",
    "T": "gaps_guarded.replica_test_period (months)",
    "O": "gaps_guarded.inject_offsets_steps (months)",
    "G": "gaps_guarded.inject_gnss_steps (epochs)", "R": "gaps_guarded.ring_windows (months)",
    "Y": "gaps_guarded.ring_path_years (years)", "L": "gaps_guarded.leveling_withheld (years)",
    "B": "none: billing pairing acts on the caller's billing table (caller responsible)",
    "P": "none: rescaling and emulation act on the caller's series (caller responsible)",
    "Q": "none: station ids only",
}
