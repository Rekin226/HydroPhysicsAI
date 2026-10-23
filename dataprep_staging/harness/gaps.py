"""Artificial-gap and injection generators mirroring the real gaps (plan 4.4, WP A1).

One generator per family of the plan 4.4 table. Randomized families draw from
``np.random.default_rng(1000 * ord(family) + r)`` for realization ``r`` (the test-period
replica uses ``7000 + r``), so every WP that re-runs a family on its final product gets the
same masks. :data:`FAMILIES` is the registry (family letter, product, real gap, test, seed
rule, generator).

The four head families N, S, E and H are copied verbatim from the head gap-fill pilot
(``_pilots/head_gapfill_pilot/scen.py``; same logic, same seeds, same random-number
consumption), so their masks are identical to the pilot's for the same matrix.

Pre-origin rule (U3, U7): a pre-origin product is tested on pre-origin truth only. Pass the
2012-2022 matrix to the head generators, keep the year lists of the July, extension, backcast
and outage families at or before 2022 (``preorigin_years``), and place the test-period
replica inside the pre-origin window. Families whose plan wording reaches past 2022 (the July
backtest to 2024, the backcast shares 2020-23, the outage window to 2025) raise unless
``allow_post_origin=True``, which is for after-the-fact analysis of full variants only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from common.paths import ORIGIN

ORIGIN_YEAR = int(ORIGIN[:4])


class LeakageError(PermissionError):
    """A generator would put post-origin truth into a pre-origin test (U3, U7)."""


def seed_for(family: str, r: int) -> int:
    """The plan's seed rule: ``1000 * ord(family) + r`` (``7000 + r`` for family 'T')."""
    if family == "T":
        return 7000 + r
    if len(family) != 1:
        raise ValueError("family must be one character")
    return 1000 * ord(family) + r


def rng_for(family: str, r: int) -> np.random.Generator:
    return np.random.default_rng(seed_for(family, r))


def _check_years(years, allow_post_origin: bool, what: str) -> list[int]:
    ys = [int(y) for y in years]
    late = [y for y in ys if y > ORIGIN_YEAR]
    if late and not allow_post_origin:
        raise LeakageError(f"{what}: years {late} are after the origin {ORIGIN} (U3/U7); "
                           f"pass allow_post_origin=True for after-the-fact analysis only")
    return ys


def preorigin_years(years) -> list[int]:
    """The years of ``years`` that are not after the origin year."""
    return [int(y) for y in years if int(y) <= ORIGIN_YEAR]


# ============================================================================ heads (H2)
# Families N, S, E, H: verbatim logic of _pilots/head_gapfill_pilot/scen.py (do not edit
# the random-number calls; the masks must stay identical to the pilot's).
def scen_N(rng, obs, targets):
    """Whole-network outages: six windows of 1 or 2 months (three each)."""
    T = obs.shape[1]
    while True:
        starts = np.sort(rng.choice(np.arange(6, T - 8), 6, replace=False))
        if np.all(np.diff(starts) >= 8):
            break
    lens = rng.permutation([1, 1, 1, 2, 2, 2])
    mask = np.zeros_like(obs)
    cells = []
    for g, (s0, L) in enumerate(zip(starts, lens, strict=True)):
        mask[:, s0:s0 + L] = True
        for i in targets:
            for t in range(s0, s0 + L):
                if obs[i, t]:
                    cells.append((g, L, i, t, min(t - s0 + 1, s0 + L - t)))
    return mask, cells


def scen_S(rng, obs, targets, per_len=12, lens=(1, 2, 3, 6, 12, 24)):
    """Single-well internal gaps of 1, 2, 3, 6, 12 and 24 months, 12 per length."""
    T = obs.shape[1]
    mask = np.zeros_like(obs)
    cells = []
    wells = list(rng.permutation(targets))
    g = 0
    for L in lens:
        placed = 0
        tries = 0
        while placed < per_len and tries < 5000 and wells:
            tries += 1
            i = wells[tries % len(wells)]
            s0 = rng.integers(2, T - L - 2)
            if obs[i, s0 - 2:s0 + L + 2].all() and not mask[i].any():
                mask[i, s0:s0 + L] = True
                for t in range(s0, s0 + L):
                    cells.append((g, L, i, t, min(t - s0 + 1, s0 + L - t)))
                wells.remove(i)
                placed += 1
                g += 1
    return mask, cells


def scen_E(rng, obs, targets, n=20):
    """Edge gaps: 66-month leading gaps at n wells and 48-month trailing gaps at n others."""
    T = obs.shape[1]
    mask = np.zeros_like(obs)
    cells = []
    lead_ok = [i for i in targets if obs[i, :66].mean() >= 0.8 and obs[i, 66:].sum() >= 36]
    trail_ok = [i for i in targets if obs[i, -48:].mean() >= 0.8 and obs[i, :-48].sum() >= 36]
    lead = rng.choice(lead_ok, min(n, len(lead_ok)), replace=False)
    rest = [i for i in trail_ok if i not in set(lead)]
    trail = rng.choice(rest, min(n, len(rest)), replace=False)
    g = 0
    for i in lead:
        mask[i, :66] = True
        cells += [(g, 66, i, t, 66 - t) for t in range(66) if obs[i, t]]
        g += 1
    for i in trail:
        mask[i, T - 48:] = True
        cells += [(g, 48, i, t, t - (T - 48) + 1) for t in range(T - 48, T) if obs[i, t]]
        g += 1
    return mask, cells


def scen_H(rng, obs, targets):
    """Half the network (wells with any data) missing for 12 months."""
    T = obs.shape[1]
    t0 = rng.integers(6, T - 18)
    have = np.where(obs.sum(1) > 0)[0]
    gone = rng.choice(have, len(have) // 2, replace=False)
    mask = np.zeros_like(obs)
    mask[np.ix_(gone, np.arange(t0, t0 + 12))] = True
    tset = set(targets)
    cells = [(0, 12, i, t, min(t - t0 + 1, t0 + 12 - t)) for i in gone if i in tset
             for t in range(t0, t0 + 12) if obs[i, t]]
    return mask, cells


HEAD_SCEN = {"N": scen_N, "S": scen_S, "E": scen_E, "H": scen_H}
#: realizations used by the pilot and the plan (S: 4 x 72 gaps; E: 2 x 40 wells)
HEAD_NREAL = {"N": 3, "S": 4, "E": 2, "H": 3}


def head_masks(family: str, obs, targets, nreal: int | None = None):
    """Yield ``(r, mask, cells)`` for head family N, S, E or H; ``cells`` rows are
    ``(gap, length, well, month, edge_distance)`` for the observed masked target cells."""
    obs = np.asarray(obs, dtype=bool)
    for r in range(nreal or HEAD_NREAL[family]):
        yield (r, *HEAD_SCEN[family](rng_for(family, r), obs, targets))


def replica_test_period(obs, targets, r: int, window: tuple[int, int],
                        avoid: tuple[int, int] | None = None, lens=(1, 2, 3, 6, 12),
                        per_len: int = 12, block_len: int = 6):
    """Family T: the test-period replica (pilot ``s8_testperiod.py``, seed ``7000 + r``).

    Single-well gaps of ``lens`` months (``per_len`` each) placed wholly inside
    ``window = (t_start, t_end)`` (month indices, end exclusive), never overlapping the
    ``avoid`` months (inclusive index pair, e.g. a real outage), plus a ``block_len``-month
    block over half of the remaining wells. With ``window = (index of 2023-01, T)`` and
    ``avoid`` = the Feb-Mar 2026 months this is the pilot's test-period check (post-origin
    truth: after-the-fact only). For a pre-origin product use the 2012-2022 matrix and a
    window that ends by 2022-12 (for example its last 45 months, the length of the
    2023-01..2026-09 test period)."""
    obs = np.asarray(obs, dtype=bool)
    rng = rng_for("T", r)
    t_lo, T = window
    mask = np.zeros_like(obs)
    cells = []
    wells = list(rng.permutation(targets))
    g = 0
    for L in lens:
        placed = 0
        for i in list(wells):
            if placed >= per_len:
                break
            s0 = rng.integers(t_lo + 2, T - L - 2)
            clash = avoid is not None and (avoid[0] <= s0 + L - 1 and s0 <= avoid[1])
            if obs[i, s0 - 2:s0 + L + 2].all() and not clash:
                mask[i, s0:s0 + L] = True
                cells += [(f"S{L:02d}", g, i, t) for t in range(s0, s0 + L)]
                wells.remove(i)
                placed += 1
                g += 1
    hi = (avoid[0] - 8) if avoid is not None else (T - block_len - 2)
    t0 = rng.integers(t_lo + 2, hi)
    rest = [i for i in range(obs.shape[0]) if not mask[i].any() and obs[i].sum() > 0]
    gone = rng.choice(rest, len(rest) // 2, replace=False)
    tset = set(targets)
    for i in gone:
        mask[i, t0:t0 + block_len] = True
        if i in tset:
            cells += [(f"H{block_len:02d}", g, i, t) for t in range(t0, t0 + block_len)
                      if obs[i, t]]
    cells = [c for c in cells if obs[c[2], c[3]]]
    return mask, cells


def daily_outage_windows(years, start: str = "02-11", end: str = "03-19",
                         allow_post_origin: bool = False) -> list[dict]:
    """Family D: the Feb-Mar 2026 outage window (11 Feb - 19 Mar, 37 days) removed in every
    year of ``years`` at every well (deterministic). Pre-origin products: 2013-2022."""
    ys = _check_years(years, allow_post_origin, "daily outage window")
    out = []
    for y in ys:
        a, b = pd.Timestamp(f"{y}-{start}"), pd.Timestamp(f"{y}-{end}")
        out.append({"year": y, "start": str(a.date()), "end": str(b.date()),
                    "days": int((b - a).days) + 1})
    return out


# ============================================================================ heads (H1)
def inject_offsets_steps(Y, r: int, *, n_paired: int = 20, n_steps: int = 20,
                         paired_size=(1.0, 3.0), step_size=(3.0, 6.0),
                         paired_len=(15, 365), exclude=None, margin: int = 30):
    """Family O (H1): inject temporary datum offsets (a jump of 1-3 m reversed 15-365 samples
    later, the audit's 'offset episode') and permanent steps (3-6 m) into series ``Y``
    (wells x samples; daily or monthly). Events avoid samples flagged in ``exclude`` (known
    real events) by ``margin`` samples and each well gets at most one event. Returns
    ``(Y_injected, events)``; ``events`` rows: well, kind, start, end (exclusive; ``None``
    for a step), size. Known real events must stay unflagged by the QC under test."""
    Yc = np.array(Y, dtype=float, copy=True)
    W, T = Yc.shape
    rng = rng_for("O", r)
    ex = np.zeros((W, T), bool) if exclude is None else np.asarray(exclude, bool)
    order = list(rng.permutation(W))
    events = []
    kinds = ["paired"] * n_paired + ["step"] * n_steps
    for kind in kinds:
        for _ in range(200):
            if not order:
                break
            w = order[0]
            obs = np.flatnonzero(np.isfinite(Yc[w]))
            if obs.size < 2 * margin:
                order.pop(0)
                continue
            if kind == "paired":
                L = int(rng.integers(paired_len[0], paired_len[1] + 1))
                s0 = int(rng.integers(obs[0] + margin, max(obs[-1] - L - margin,
                                                             obs[0] + margin + 1)))
                s1 = s0 + L
            else:
                s0 = int(rng.integers(obs[0] + margin, max(obs[-1] - margin,
                                                             obs[0] + margin + 1)))
                s1 = None
            lo, hi = max(s0 - margin, 0), min((s1 or s0) + margin, T)
            if ex[w, lo:hi].any():
                order.append(order.pop(0))
                continue
            lo_s, hi_s = (paired_size if kind == "paired" else step_size)
            size = float(rng.uniform(lo_s, hi_s) * rng.choice([-1.0, 1.0]))
            Yc[w, s0:(s1 if s1 is not None else T)] += size
            events.append({"well": int(w), "kind": kind, "start": s0, "end": s1,
                           "size": size})
            order.pop(0)
            break
    return Yc, events


# ============================================================================ rings (S1)
def ring_windows(start: str = "2015-01", end: str = "2021-12", length: int = 13
                 ) -> list[dict]:
    """Family R windows: every ``length``-month window wholly inside [start, end] (months)."""
    months = pd.period_range(start, end, freq="M")
    return [{"start": str(months[i]), "end": str(months[i + length - 1])}
            for i in range(len(months) - length + 1)]


def ring_instrument_injection(site_ids, r: int, *, p_A: float = 10 / 14,
                              b_A: float = 182.2e-6, tau_A: float = 11.4e-6,
                              b_B: float = 1.9e-6, sd_B: float = 14.3e-6,
                              offset_clusters=((-0.163, -0.155, 8), (0.50, 0.54, 3)),
                              offset_other=(-0.10, 0.20)) -> list[dict]:
    """Family R injection: mixed A/B instrument changes ``z_post = a + (1 + b) z_true``.

    Groups are drawn with probability ``p_A`` (default 10/14, the pilot's group-A share;
    replace it by the observed proportions of the evidence-based assignment); b is drawn
    around the group's pooled scale (A: 182.2 ppm, between-site tau 11.4; B: 1.9 ppm, se
    14.3) and the offset a from the two observed clusters (8 and 3 sites) or, for the rest,
    uniformly from ``offset_other`` (metres). Seed ``1000*ord('R') + r``."""
    rng = rng_for("R", r)
    sites = list(site_ids)
    n_cl = sum(c[2] for c in offset_clusters)
    out = []
    for s in sites:
        grp = "A" if rng.random() < p_A else "B"
        b = float(rng.normal(b_A, tau_A) if grp == "A" else rng.normal(b_B, sd_B))
        u = rng.random()
        if u < n_cl / max(len(sites), n_cl):
            pick = rng.choice(len(offset_clusters),
                              p=np.array([c[2] for c in offset_clusters]) / n_cl)
            lo, hi, _ = offset_clusters[pick]
        else:
            lo, hi = offset_other
        a = float(rng.uniform(lo, hi))
        out.append({"site": s, "group": grp, "b": b, "a": a})
    return out


def apply_ring_instrument(z_true, a: float, b: float) -> np.ndarray:
    """Ring depth after the instrument change: ``a + (1 + b) z_true``."""
    return a + (1.0 + b) * np.asarray(z_true, dtype=float)


def ring_path_years(years=range(2017, 2022)) -> list[dict]:
    """Family Y (ring path): mask each calendar year, keeping the December before and the
    January after as known endpoints (deterministic)."""
    return [{"year": int(y), "mask_start": f"{y}-01", "mask_end": f"{y}-12",
             "left_endpoint": f"{y - 1}-12", "right_endpoint": f"{y + 1}-01"} for y in years]


# ============================================================================ GNSS (S3)
def inject_gnss_steps(n_epochs: int, r: int, *, n_steps: int = 3, size=(0.02, 0.06),
                      min_sep: int = 90, exclude=None) -> tuple[np.ndarray, list[dict]]:
    """Family G: additive step series (metres) with ``n_steps`` steps of 2-6 cm (random sign)
    at epochs at least ``min_sep`` apart and away from ``exclude`` epochs (known equipment
    changes). Seed ``1000*ord('G') + r``. Returns (offset series, steps)."""
    rng = rng_for("G", r)
    ex = np.zeros(n_epochs, bool) if exclude is None else np.asarray(exclude, bool)
    chosen: list[int] = []
    for _ in range(10000):
        if len(chosen) >= n_steps:
            break
        t = int(rng.integers(min_sep, n_epochs - min_sep))
        if ex[max(t - min_sep, 0):t + min_sep].any():
            continue
        if all(abs(t - c) >= min_sep for c in chosen):
            chosen.append(t)
    series = np.zeros(n_epochs)
    steps = []
    for t in sorted(chosen):
        s = float(rng.uniform(*size) * rng.choice([-1.0, 1.0]))
        series[t:] += s
        steps.append({"epoch": t, "size": s})
    return series, steps


# ============================================================================ billing (P2a)
def nearest_phase(xy_monthly, xy_bimonthly, phase_bimonthly) -> np.ndarray:
    """Family B: the billing phase ('A'/'B') of each monthly meter's nearest bimonthly meter
    (Euclidean, metres; brute force in chunks)."""
    xm = np.asarray(xy_monthly, dtype=float)
    xb = np.asarray(xy_bimonthly, dtype=float)
    ph = np.asarray(phase_bimonthly)
    out = np.empty(len(xm), dtype=ph.dtype)
    for i0 in range(0, len(xm), 2000):
        d = ((xm[i0:i0 + 2000, None, :] - xb[None, :, :]) ** 2).sum(-1)
        out[i0:i0 + 2000] = ph[np.argmin(d, axis=1)]
    return out


def pair_ids(dates, phase: str) -> np.ndarray:
    """Billing-pair id of each month: phase A pairs Jan-Feb, Mar-Apr, ...; phase B pairs
    Feb-Mar, ..., Dec-Jan."""
    di = pd.DatetimeIndex(pd.to_datetime(dates))
    k = di.year * 12 + di.month - 1
    if phase == "A":
        return np.asarray(k // 2)
    if phase == "B":
        return np.asarray((k - 1) // 2)
    raise ValueError("phase must be 'A' or 'B'")


def pair_sum(monthly, dates, phases) -> tuple[np.ndarray, np.ndarray]:
    """Sum monthly meters (M, T) into their billing pairs. Returns ``(pair_total, even)``,
    both (M, T): the pair total on both months of the pair and the even split (total / 2);
    NaN where a pair is incomplete in the record (its first or last month is missing)."""
    v = np.asarray(monthly, dtype=float)
    tot = np.full_like(v, np.nan)
    for ph in ("A", "B"):
        rows = np.flatnonzero(np.asarray(phases) == ph)
        if rows.size == 0:
            continue
        pid = pair_ids(dates, ph)
        for p in np.unique(pid):
            cols = np.flatnonzero(pid == p)
            if cols.size != 2:
                continue
            s = v[np.ix_(rows, cols)].sum(1)
            tot[np.ix_(rows, cols)] = s[:, None]
    return tot, tot / 2.0


# ============================================================================ P2b / P2d / P1
def rolling_origin_july(years=range(2013, 2023), allow_post_origin: bool = False) -> list[dict]:
    """Family J: rolling-origin July backtest; each fold predicts July of ``year`` from
    data up to June of that year. The plan's 2013-2024 needs ``allow_post_origin``."""
    ys = _check_years(years, allow_post_origin, "July backtest")
    return [{"year": y, "train_end": f"{y}-06", "test_month": f"{y}-07"} for y in ys]


def census_freeze(freeze: str = "2013-06", predict_years=(2014, 2015, 2016)) -> dict:
    """Family C: census frozen at ``freeze``; predict the cohorts installed in
    ``predict_years`` (per township counts and kWh)."""
    return {"freeze": freeze, "predict_years": [int(y) for y in predict_years]}


def leave_one_year_out(years, allow_post_origin: bool = False) -> list[dict]:
    """Family X: leave-one-year-out folds (P1 extension: 2016-2022 pre-origin; the plan's
    2016-2024 needs ``allow_post_origin``)."""
    ys = _check_years(years, allow_post_origin, "leave one year out")
    return [{"test_year": y, "train_years": [z for z in ys if z != y]} for y in ys]


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def season_blocks(years, allow_post_origin: bool = False) -> list[dict]:
    """Family X (whole seasons): one fold per season of each year (DJF takes December of the
    previous year), held out as a block."""
    ys = _check_years(years, allow_post_origin, "season blocks")
    out = []
    for y in ys:
        for name, mons in SEASONS.items():
            ms = [f"{y - 1}-12" if m == 12 and name == "DJF" else f"{y}-{m:02d}" for m in mons]
            out.append({"year": y, "season": name, "months": ms})
    return out


def matched_lead_backcast(fit_years=(2020, 2021, 2022), predict_years=(2016, 2017, 2018),
                          allow_post_origin: bool = False) -> dict:
    """Family K: matched-lead pseudo-backcast. The real backcast predicts 2012-15 from
    shares four years later, so the test fits shares on ``fit_years`` and predicts
    ``predict_years`` at the same lead. Pre-origin default: 2020-22 -> 2016-18 (lead 4).
    The plan's 2020-23 -> 2016-19 uses 2023 and needs ``allow_post_origin``."""
    fy = _check_years(fit_years, allow_post_origin, "backcast fit years")
    py = [int(y) for y in predict_years]
    leads = sorted({a - b for a, b in zip(fy, py, strict=True)})
    return {"fit_years": fy, "predict_years": py, "leads": leads}


# ============================================================================ canal (W1)
def canal_loyo(years=range(2015, 2022), months=(1, 2, 3, 4, 5)) -> list[dict]:
    """Family W: Jan-May leave-one-year-out folds of the canal benchmark (2015-2021)."""
    ys = _check_years(years, False, "canal benchmark")
    return [{"test_year": y, "months": [f"{y}-{m:02d}" for m in months],
             "train_years": [z for z in ys if z != y]} for y in ys]


def paired_bootstrap_indices(n: int, B: int = 2000, r: int = 0) -> np.ndarray:
    """Family W: (B, n) resampling indices shared by the methods being compared (paired
    bootstrap), seed ``1000*ord('W') + r``."""
    return rng_for("W", r).integers(0, n, size=(B, n))


# ============================================================================ InSAR (S4)
def leveling_withheld(years=(2020, 2021)) -> dict:
    """Family L: leveling surveys in ``years`` withheld from the InSAR referencing and used
    only as the check."""
    return {"withheld_years": [int(y) for y in years]}


def leave_one_station_out(station_ids) -> list[dict]:
    """Family Q: GNSS leave-one-station-out folds."""
    ids = list(station_ids)
    return [{"test": s, "train": [t for t in ids if t != s]} for s in ids]


# ============================================================================ P4 (E11)
def cv_mismatch_rescale(series, factor: float) -> np.ndarray:
    """E11: rescale a series' time variation about its own time mean by ``factor`` (0.5,
    1.5) while keeping the mean (last axis = time)."""
    v = np.asarray(series, dtype=float)
    m = v.mean(axis=-1, keepdims=True)
    return m + factor * (v - m)


def mu_prior_emulation(true_total: float, r: int, sd_log: float = 0.5, n: int = 1
                       ) -> np.ndarray:
    """E11: prior means for mu drawn log-normally around the region's true total
    (log-sd 0.5), seed ``1000*ord('P') + r``."""
    return float(true_total) * np.exp(rng_for("P", r).normal(0.0, sd_log, size=n))


def block_resample_residuals(resid, r: int, block: int = 12) -> np.ndarray:
    """E11: circular block bootstrap in time of real residuals (W, T); the same blocks are
    used for every well, keeping the cross-well correlation. Seed ``1000*ord('P') + 500 + r``
    (offset so it does not reuse the mu-prior stream)."""
    x = np.asarray(resid, dtype=float)
    T = x.shape[-1]
    rng = np.random.default_rng(seed_for("P", 500 + r))
    nb = -(-T // block)
    starts = rng.integers(0, T, size=nb)
    idx = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
    return x[..., idx]


# ============================================================================ registry
@dataclass(frozen=True)
class Family:
    letter: str
    product: str
    real_gap: str
    test: str
    seed_rule: str
    generators: tuple[str, ...]


FAMILIES = {f.letter: f for f in (
    Family("D", "heads (H2)", "Feb-Mar 2026 outage (37 days, partial days on both sides)",
           "the same daily window removed in every year at the 158 wells (pre-origin "
           "products: 2013-2022)", "deterministic", ("daily_outage_windows",)),
    Family("S", "heads (H2)", "1-24 month internal gaps",
           "single-well internal gaps of 1, 2, 3, 6, 12, 24 months (4 x 72)",
           "1000*ord('S')+r", ("head_masks", "scen_S")),
    Family("E", "heads (H2)", "late-starting wells; records ending in 2018",
           "66-month leading and 48-month trailing gaps (2 x 40 wells)",
           "1000*ord('E')+r", ("head_masks", "scen_E")),
    Family("H", "heads (H2)", "Jul-Aug 2023 network thinning",
           "50 % of wells missing for 12 months", "1000*ord('H')+r",
           ("head_masks", "scen_H")),
    Family("N", "heads (H2)", "network-wide months", "whole-network 1- and 2-month outages",
           "1000*ord('N')+r", ("head_masks", "scen_N")),
    Family("T", "heads (H2)", "2023-26 pattern",
           "test-period replica (single-well 1-12 month gaps and a half-network 6-month "
           "block placed inside a window; pre-origin products: a window ending by 2022-12)",
           "7000+r", ("replica_test_period",)),
    Family("O", "heads (H1)", "datum offsets, steps, barometric loading",
           "injected paired offsets >= 1 m and steps >= 3 m; known real events must stay "
           "unflagged", "1000*ord('O')+r", ("inject_offsets_steps",)),
    Family("R", "rings (S1)", "13-month hiatus with an instrument change, groups unknown",
           "every 13-month window 2015-2021 masked, mixed A/B jumps injected, "
           "classification included", "1000*ord('R')+r",
           ("ring_windows", "ring_instrument_injection", "apply_ring_instrument")),
    Family("Y", "ring path (S1)", "monthly values inside 2022",
           "masked calendar years 2017-2021 with known endpoints", "deterministic",
           ("ring_path_years",)),
    Family("G", "GNSS (S3)", "equipment steps", "injected steps >= 2 cm",
           "1000*ord('G')+r", ("inject_gnss_steps",)),
    Family("B", "billing (P2a)", "bimonthly pair sums",
           "monthly meters given their nearest bimonthly meter's phase, then pair-summed",
           "deterministic", ("nearest_phase", "pair_ids", "pair_sum")),
    Family("J", "July 2025 (P2b)", "missing phase-A bill",
           "rolling-origin July backtest (pre-origin 2013-2022; plan 2013-2024 after the "
           "fact)", "deterministic", ("rolling_origin_july",)),
    Family("C", "census (P2d)", "frozen after mid-2016",
           "census frozen at mid-2013; predict 2014-16 cohorts", "deterministic",
           ("census_freeze",)),
    Family("X", "extension (P1)", "census ends 2025-06",
           "leave one year out (pre-origin 2016-2022; plan 2016-2024 after the fact), plus "
           "whole seasons", "deterministic", ("leave_one_year_out", "season_blocks")),
    Family("K", "backcast (P1)", "no county data before 2016",
           "matched-lead pseudo-backcast (pre-origin 2020-22 shares predicting 2016-18; "
           "plan 2020-23 -> 2016-19 after the fact)", "deterministic",
           ("matched_lead_backcast",)),
    Family("W", "canal (W1)", "monthly values unknown",
           "Jan-May 2015-2021, leave one year out, paired bootstrap", "1000*ord('W')+r",
           ("canal_loyo", "paired_bootstrap_indices")),
    Family("P", "Changhua pumping (P4)", "no metered data",
           "pseudo-Changhua regions with the census withheld (units module), CV-mismatch "
           "rescaling and mu-prior emulation (E11)", "1000*ord('P')+r",
           ("cv_mismatch_rescale", "mu_prior_emulation", "block_resample_residuals")),
    Family("L", "InSAR (S4)", "leveling after 2021 absent", "leveling 2020-21 withheld",
           "deterministic", ("leveling_withheld",)),
    Family("Q", "InSAR (S4)", "leveling after 2021 absent", "GNSS leave one station out",
           "deterministic", ("leave_one_station_out",)),
)}
