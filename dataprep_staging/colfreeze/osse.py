"""OSSE engine and draft declaration of COLF.column_family_2028.v1 (round-8 rank 7).

Question: on the 2028 pile and township units, can the observations rank the SD v2
leveling-equivalent columns (Tier A+B, deduplicated)? Model-only: every synthetic truth is a
frozen member's own status-quo (baseline) prediction (or a null construction for the
self-check); no observation is read.

Ranking rule (shared with the scorer):

* pile arms: per primary pile, the RMSE over verified months of the cumulative status-quo
  compaction since the arm's reference month, summed over verified months only (member
  unscaled, N3's depth convention); a pair is ranked only if the same member is closer at
  BOTH primary piles (N3's 'every primary pile' rule); otherwise 'split' (not ranked);
* township arm: per survey year and coastal township, the member's peak rate against WRA's
  printed bound row (N3S-L's check_bounds semantics); RMSE of the bound distance over the
  rows of informative years; |RMSE_a - RMSE_b| <= TIE_ABS is a tie.

OSSE: per pair (A, B) and arm, NOMINAL cells = truth in {A, B} x the copied noise grid x
(piles) availability {full, minimum}; P(correct) = P(the truth member is ranked first); the
pair's P = min over the nominal cells. Report-only cells (piles): ac300, rho {0.4, 2} and
g_mod_cell. Self-check per arm, computed before any P: an exact label-swap check (the
engine's outcome for (B, A) is (A, B)'s mirrored, same seed) and a null that must give
P_sym = P(first wins) + P(not ranked) / 2 within 0.5 +- SELF_CHECK_TOL: piles: the midpoint
truth (A + B) / 2; townships: the shared value (A + B) / 2 with the competitors' labels drawn
at random per replicate (the bound metric is one-sided, so a midpoint is not equidistant
there). If any arm fails, no P is computed and only the failure is written.

Usage (repository root; the lead runs these after the freeze):
    $PY -I dataprep_staging/colfreeze/osse.py record --seed <S> --freeze-npz <npz>
        (engineering record at the fixed path, once, before any run)
    $PY -I dataprep_staging/colfreeze/osse.py run --seed <S> --freeze-npz <npz>
        (refuses another seed or freeze; result written once at the fixed path)
    $PY -I dataprep_staging/colfreeze/osse.py draft --seed <S> [--freeze-summary <json>]
        [--freeze-npz <npz>] [--osse-result <json>]          (validates; never writes)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colf_config as C  # noqa: E402
import members as M  # noqa: E402

OSSE_VERSION = "2.0.0"


# ----------------------------------------------------------------------------- seeds
def check_seed(seed, real: bool = False) -> int:
    """The OSSE seed: an int the lead picks at registration (no default)."""
    if seed is None:
        raise ValueError("the seed is required (no default): the lead picks it at registration")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"the seed must be an int, not {type(seed).__name__}")
    seed = int(seed)
    if seed < 0:
        raise ValueError("the seed must be >= 0")
    if seed in C.REFUSED_SEEDS:
        raise ValueError(f"seed {seed} is refused (N3's G0 or bootstrap seed)")
    if real and seed in C.REAL_REFUSED_SEEDS:
        raise ValueError(f"seed {seed} is refused for real runs (the unit tests' seed)")
    return seed


# ----------------------------------------------------------------------------- piles
def pile_increments(piles: np.ndarray, piles_months: list[str], arm: str) -> np.ndarray:
    """(2, n) member increments (cm, + = compaction) over the arm's window months."""
    a = C.ARMS[arm]
    lo, hi = piles_months.index(a["ref_month"]), piles_months.index(a["window"][1])
    s = np.asarray(piles, float)[:, lo:hi + 1] * 100.0
    return np.diff(s, axis=1)


def choose_sensor(series: dict, primary: int, fallback: int | None, min_months: int):
    """N3's rule: the primary depth; the fallback only if the primary has fewer than
    ``min_months`` verified months and the fallback at least that many; never mixed.
    ``series`` = {depth: (n,) increments with NaN where not verified}."""
    s = np.asarray(series.get(primary, []), float)
    n_p = int(np.isfinite(s).sum()) if s.size else 0
    if n_p < min_months and fallback is not None and fallback in series:
        f = np.asarray(series[fallback], float)
        if int(np.isfinite(f).sum()) >= min_months:
            return fallback, f
    return primary, s


def rmse_piles(member_inc: np.ndarray, obs_inc: np.ndarray) -> np.ndarray:
    """(..., P) per-pile RMSE (cm) of cumulative status-quo compaction over verified months.
    member_inc (P, n); obs_inc (..., P, n) with NaN where not verified; NaN for a pile
    without a verified month."""
    obs = np.asarray(obs_inc, float)
    ok = np.isfinite(obs)
    cm = np.cumsum(np.where(ok, member_inc, 0.0), axis=-1)
    co = np.cumsum(np.where(ok, obs, 0.0), axis=-1)
    d2 = np.where(ok, (cm - co) ** 2, 0.0).sum(axis=-1)
    n = ok.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.sqrt(d2 / np.maximum(n, 1)), np.nan)


def ar1(lead: tuple, n: int, sd: float, phi: float, rng) -> np.ndarray:
    """Stationary AR(1) with marginal sd ``sd`` along the last axis (g0_n3.ar1, batched)."""
    e = np.empty(lead + (n,))
    e[..., 0] = rng.normal(0.0, sd, lead)
    s = sd * np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        e[..., t] = phi * e[..., t - 1] + rng.normal(0.0, s, lead)
    return e


def availability_mask(reps: int, P: int, n: int, keep: int, rng) -> np.ndarray:
    """(reps, P, n) True on exactly ``keep`` random months per replicate and pile."""
    order = np.argsort(rng.random((reps, P, n)), axis=-1)
    return order < keep


def synth_pile_obs(truth_inc: np.ndarray, sd: float, phi: float, min_months: int, reps: int,
                   rng, *, anchor: str = "nominal", avail: str = "full",
                   g: float = C.G_MOD_CM_PER_M, rho: float = 1.0) -> np.ndarray:
    """(reps, 2, n) printed increments of a synthetic truth: rho x truth, G0's head-error
    AR(1) through g (cm/m), 0.1 cm print rounding, the declared sensor rule (anchor cell) and
    the availability cell (NaN = not verified)."""
    if anchor not in C.ANCHOR_CELLS:
        raise ValueError(f"unknown anchor cell {anchor!r}")
    if avail not in C.AVAIL_CELLS:
        raise ValueError(f"unknown availability cell {avail!r}")
    P, n = truth_inc.shape
    e = ar1((reps, P), n + 1, sd, phi, rng)
    raw = rho * truth_inc[None] + g * np.diff(e, axis=-1)
    obs = np.empty_like(raw)
    for p, key in enumerate(C.PILE_KEYS):
        spec = C.PILES[key]
        share = 1.0
        if anchor == "ac300" and key == "tuku":
            prim = np.full(n, np.nan)
            prim[:C.AC_400_MONTHS] = 1.0
            depth, _ = choose_sensor({spec["depth_m"]: prim, spec["fallback_depth_m"]:
                                      np.ones(n)}, spec["depth_m"], spec["fallback_depth_m"],
                                     min_months)
            if depth != spec["fallback_depth_m"]:
                raise AssertionError("anchor cell: the sensor rule did not move to 300 m")
            share = C.F300
        obs[:, p] = np.round(share * raw[:, p] / C.PRINT_RES_CM) * C.PRINT_RES_CM
    if avail == "minimum":
        obs = np.where(availability_mask(reps, P, n, min_months, rng), obs, np.nan)
    return obs


def outcome_piles(ra: np.ndarray, rb: np.ndarray) -> dict:
    """Per-pile agreement: 'a' (or 'b') only if closer at EVERY primary pile; ties or a split
    are 'not ranked' (reported as 'tie', with 'split' apart)."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    d = ra - rb
    bad = ~np.isfinite(d)
    lt = (d < -C.TIE_ABS) & ~bad
    gt = (d > C.TIE_ABS) & ~bad
    a = lt.all(axis=-1)
    b = gt.all(axis=-1)
    n = len(a)
    split = (lt.any(axis=-1) & gt.any(axis=-1))
    return {"a": float(a.sum()) / n, "b": float(b.sum()) / n,
            "tie": float((~a & ~b).sum()) / n, "split": float(split.sum()) / n}


# ----------------------------------------------------------------------------- townships
def town_offsets(town_cells: dict) -> list[tuple[str, int, int]]:
    out, off = [], 0
    for t in C.COAST:
        n = len(town_cells[t])
        out.append((t, off, off + n))
        off += n
    return out


def town_peaks(towns: np.ndarray, towns_months: list[str], town_cells: dict,
               interval: tuple[str, str]) -> np.ndarray:
    """(11,) peak rate (cm/yr, + = sinking) per coastal township over ``interval``
    (score_n3s_l.zr_quantities: max over the township's cells of the cell rate)."""
    ia, ib = towns_months.index(interval[0]), towns_months.index(interval[1])
    s = np.asarray(towns, float) * 100.0
    rate = (s[:, ib] - s[:, ia]) * 12.0 / (ib - ia)
    return np.asarray([rate[a:b].max() for _, a, b in town_offsets(town_cells)])


def bound_distance(peaks, listed, upper, use) -> np.ndarray:
    """Distance (cm/yr) of a peak from N3S-L's bound row: shortfall below 3 - TOL on listed
    rows plus excess over upper + TOL; NaN where the row is not used (ambiguous). One-sided
    above the lower bound: a peak inside [3 - TOL, upper + TOL] scores 0."""
    peaks = np.asarray(peaks, float)
    lo = np.where(listed, np.maximum(0.0, (C.LISTED_FLOOR - C.TOL) - peaks), 0.0)
    hi = np.maximum(0.0, peaks - (np.asarray(upper, float) + C.TOL))
    return np.where(use, lo + hi, np.nan)


def rmse_towns(member_peaks: dict, obs: dict) -> np.ndarray:
    """RMSE over the used rows of informative years. member_peaks {y: (..., 11)}; obs {y:
    {'listed', 'upper', 'use': (..., 11), 'informative': (...,)}}; NaN if no year counts."""
    d2, n = 0.0, 0
    for y, o in obs.items():
        d = bound_distance(member_peaks[y], o["listed"], o["upper"], o["use"])
        inf = np.asarray(o["informative"], bool)[..., None]
        ok = np.isfinite(d) & inf
        d2 = d2 + np.where(ok, d ** 2, 0.0).sum(axis=-1)
        n = n + ok.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.asarray(n) > 0, np.sqrt(d2 / np.maximum(n, 1)), np.nan)


def synth_town_obs(truth_peaks: dict, sigma: float, reps: int, rng) -> dict:
    """N3S-L's passability construction on a member truth: N(0, sigma) on each township peak;
    listed = peak >= 3; upper = the year's maximum listed peak when listed, else 3."""
    out = {}
    for y in C.SURVEYS:
        t = truth_peaks[y][None] + sigma * rng.standard_normal((reps, len(C.COAST)))
        listed = t >= C.LISTED_FLOOR
        mx = np.where(listed, t, -np.inf).max(axis=1, keepdims=True)
        upper = np.where(listed, mx, C.LISTED_FLOOR)
        out[y] = {"listed": listed, "upper": upper, "use": np.ones_like(listed, bool),
                  "informative": listed.sum(axis=1) >= C.L1_MIN_LISTED}
    return out


def outcome(ra: np.ndarray, rb: np.ndarray) -> dict:
    """Township outcome: smaller RMSE wins; NaN or |difference| <= TIE_ABS is a tie."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    d = ra - rb
    tie = ~np.isfinite(d) | (np.abs(d) <= C.TIE_ABS)
    a = (d < 0) & ~tie
    b = (d > 0) & ~tie
    n = len(d)
    return {"a": float(a.sum()) / n, "b": float(b.sum()) / n, "tie": float(tie.sum()) / n}


# ----------------------------------------------------------------------------- cells
def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def _arm_inputs(frozen: dict, arm: str, cols: list[str]) -> dict:
    if C.ARMS[arm]["kind"] == "piles":
        return {c: pile_increments(frozen["members"][c]["piles"], frozen["piles_months"], arm)
                for c in cols}
    return {c: {y: town_peaks(frozen["members"][c]["towns"], frozen["towns_months"],
                              frozen["town_cells"], C.NOMINAL[y]) for y in C.SURVEYS}
            for c in cols}


def _mid(x, y):
    if isinstance(x, dict):
        return {k: 0.5 * (x[k] + y[k]) for k in x}
    return 0.5 * (x + y)


def nominal_cells(arm: str) -> list[dict]:
    if C.ARMS[arm]["kind"] == "piles":
        return [{"noise": (sd, phi), "avail": av} for sd in C.G0_SD_GRID_M
                for phi in C.G0_PHI_GRID for av in C.AVAIL_CELLS]
    return [{"noise": s} for s in C.TOWN_SIGMA_GRID]


def sensitivity_cells(arm: str) -> list[dict]:
    """Report-only (piles): ac300, rho 0.4 / 2 and g_mod_cell, each over the noise grid."""
    if C.ARMS[arm]["kind"] != "piles":
        return []
    out = []
    for sd in C.G0_SD_GRID_M:
        for phi in C.G0_PHI_GRID:
            nz = (sd, phi)
            out.append({"family": "ac300", "noise": nz, "anchor": "ac300"})
            out += [{"family": f"rho_{r:g}", "noise": nz, "rho": r} for r in C.RHO_SENS]
            out.append({"family": "g_mod_cell", "noise": nz, "g": C.G_MOD_CELL_CM_PER_M})
    return out


def simulate(arm: str, xa, xb, truth, cell: dict, reps: int, ss, *, swap_labels=False) -> dict:
    """One cell. ``swap_labels``: per replicate, the two competitors' labels are drawn at
    random (the township null); 'a' is then the first-listed label."""
    rng = np.random.default_rng(ss)
    if C.ARMS[arm]["kind"] == "piles":
        sd, phi = cell["noise"]
        obs = synth_pile_obs(truth, sd, phi, C.ARMS[arm]["min_months"], reps, rng,
                             anchor=cell.get("anchor", "nominal"),
                             avail=cell.get("avail", "full"),
                             g=cell.get("g", C.G_MOD_CM_PER_M), rho=cell.get("rho", 1.0))
        ra, rb = rmse_piles(xa, obs), rmse_piles(xb, obs)
        if swap_labels:
            flip = rng.random(reps) < 0.5
            ra, rb = np.where(flip[:, None], rb, ra), np.where(flip[:, None], ra, rb)
        return outcome_piles(ra, rb)
    obs = synth_town_obs(truth, cell["noise"], reps, rng)
    ra, rb = rmse_towns(xa, obs), rmse_towns(xb, obs)
    if swap_labels:
        flip = rng.random(reps) < 0.5
        ra, rb = np.where(flip, rb, ra), np.where(flip, ra, rb)
    return outcome(ra, rb)


def _p_sym(o: dict) -> float:
    return o["a"] + 0.5 * o["tie"]


def self_check_arm(arm: str, x: dict, pairs: list, dd_arm: dict, reps: int, ss) -> dict:
    """The arm's self-check (no P(correct) is formed): the exact label-swap check on the
    first nominal cell, the null in every nominal cell, and each collapsed alias exactly
    0.5."""
    kind = C.ARMS[arm]["kind"]
    fails, rec = [], []
    kids = ss.spawn(len(pairs) + len(C.ALIASES))
    for (a, b), k in zip(pairs, kids[:len(pairs)], strict=True):
        cells = nominal_cells(arm)
        gs = k.spawn(len(cells) + 1)
        o1 = simulate(arm, x[a], x[b], x[a], cells[0], reps, gs[-1])
        o2 = simulate(arm, x[b], x[a], x[a], cells[0], reps, gs[-1])
        swap_ok = (o1["a"], o1["b"], o1["tie"]) == (o2["b"], o2["a"], o2["tie"])
        if not swap_ok:
            fails.append(f"{arm} {a}|{b}: the outcome is not label-symmetric")
        for cell, g in zip(cells, gs[:-1], strict=True):
            o = simulate(arm, x[a], x[b], _mid(x[a], x[b]), cell, reps, g,
                         swap_labels=(kind == "towns"))
            p = _p_sym(o)
            ok = abs(p - 0.5) <= C.SELF_CHECK_TOL
            rec.append({"pair": _pair_key(a, b), **cell, "p_sym": p, "ok": ok,
                        "swap_ok": swap_ok})
            if not ok:
                fails.append(f"{arm} {a}|{b} null {cell}: P_sym {p:.3f}")
    alias = {}
    for (al, canon), k in zip(C.ALIASES.items(), kids[len(pairs):], strict=True):
        if dd_arm["evidence"][al]["bytes_equal_on_arm"]:
            o = simulate(arm, x[canon], x[al], x[canon], nominal_cells(arm)[0], reps, k)
            p = _p_sym(o)
            alias[al] = {"canonical": canon, "p_sym": p, "ok": p == 0.5}
            if p != 0.5:
                fails.append(f"{arm} alias {al}: P_sym {p} != 0.5")
    return {"pass": not fails, "fails": fails, "null_cells": rec, "aliases": alias,
            "null": "midpoint truth" if kind == "piles" else
                    "shared value (A + B) / 2 with labels drawn at random per replicate"}


def run_osse(frozen: dict, seed: int, reps: int = C.REPS, arms=C.ARM_ORDER,
             real: bool = True) -> dict:
    """The decidability study on frozen predictions (members.load_freeze). Deterministic
    given the seed. Self-checks of every arm come first; if any fails, the result carries
    only the failures (no P(correct) is computed)."""
    seed = check_seed(seed, real=real)
    dd = M.dedup(frozen)
    root = np.random.SeedSequence(seed)
    arm_ss = dict(zip(C.ARM_ORDER, root.spawn(len(C.ARM_ORDER)), strict=True))
    prep, checks = {}, {}
    for arm in arms:
        cols = dd[arm]["members"]
        x = _arm_inputs(frozen, arm, sorted(set(cols) | set(C.ALIASES) | set(C.ALIASES.values())))
        pairs = list(itertools.combinations(sorted(cols), 2))
        s_check, s_dec = arm_ss[arm].spawn(2)
        prep[arm] = (cols, x, pairs, s_dec)
        checks[arm] = self_check_arm(arm, x, pairs, dd[arm], reps, s_check)
    base = {"id": C.ID, "osse_version": OSSE_VERSION, "seed": seed, "reps": int(reps),
            "self_check": {"pass": all(c["pass"] for c in checks.values()),
                           "by_arm": {a: {"pass": c["pass"], "fails": c["fails"]}
                                      for a, c in checks.items()},
                           "fails": [f for c in checks.values() for f in c["fails"]]},
            "thresholds": {"p_d": C.P_D, "p_narrow": C.P_NARROW,
                           "self_check_tol": C.SELF_CHECK_TOL}}
    if not base["self_check"]["pass"]:
        base["status"] = "self_check_failed"
        base["decidability"] = {"status": "self_check_failed", "arms": {}}
        return base                                        # no P(correct) exists
    res = {}
    for arm in arms:
        cols, x, pairs, s_dec = prep[arm]
        out = {}
        for (a, b), ss in zip(pairs, s_dec.spawn(len(pairs)), strict=True):
            nom = [(t, c) for t in (a, b) for c in nominal_cells(arm)]
            sen = [(t, c) for t in (a, b) for c in sensitivity_cells(arm)]
            kids = iter(ss.spawn(len(nom) + len(sen)))
            rec_n, rec_s = [], []
            for lst, cells in ((rec_n, nom), (rec_s, sen)):
                for t, cell in cells:
                    o = simulate(arm, x[a], x[b], x[t], cell, reps, next(kids))
                    lst.append({"truth": t, **cell, **o,
                                "p_correct": o["a"] if t == a else o["b"]})
            ps = [r["p_correct"] for r in rec_n]
            fam = {}
            for r in rec_s:
                fam.setdefault(r["family"], []).append(r["p_correct"])
            out[_pair_key(a, b)] = {
                "cells": rec_n, "p_min": float(min(ps)), "p_mean": float(np.mean(ps)),
                "worst_cell": rec_n[int(np.argmin(ps))],
                "report_only_cells": rec_s,
                "report_only_p_min": {k: float(min(v)) for k, v in fam.items()}}
        res[arm] = {"members": cols, "pairs": out, "self_check": checks[arm]}
    result = {**base, "status": "ok", "arms": res}
    result["decidability"] = decidability(result)
    return result


def decidability(result: dict) -> dict:
    """{arm: {pair: {'p', 'label'}}} from the NOMINAL P, plus the multiplicity: sum of
    (1 - P) over the labelled-decidable pairs = the expected number of wrong labels (an upper
    bound, P being a minimum over cells). No label exists before every self-check passes."""
    if not result["self_check"]["pass"]:
        return {"status": "self_check_failed", "arms": {}}
    out, mult = {}, {}
    for arm, r in result["arms"].items():
        hi = "D" if C.ARMS[arm]["role"] == "gating" else "R"
        out[arm] = {k: {"p": v["p_min"], "label": hi if v["p_min"] >= C.P_D else (
            "narrowing" if v["p_min"] <= C.P_NARROW else "neither")}
            for k, v in r["pairs"].items()}
        dec = [v["p"] for v in out[arm].values() if v["label"] == hi]
        mult[arm] = {"label": hi, "n": len(dec), "sum_1_minus_p": float(sum(1 - p for p in dec)),
                     "expected_wrong_labels": float(sum(1 - p for p in dec))}
    tot = sum(m["sum_1_minus_p"] for m in mult.values() if m["label"] == "D")
    return {"status": "ok", "arms": out, "multiplicity": mult,
            "expected_wrong_D_labels": float(tot)}


# ----------------------------------------------------------------------------- page lines
UNIT_WORDS = {
    "r_piles": "the 2026-12..2028-11 printed months of the 土庫國中 and 中科虎尾 deep piles "
               "(N3's months)",
    "r_towns": "WRA's ROC116/117 coastal township bounds (N3S-L's survey years)",
    "g_piles": "the 2028-12..2029-11 printed months of the 土庫國中 and 中科虎尾 deep piles",
}
SCALE_WORDS = {"r_piles": " at pile/column scale 1", "g_piles": " at pile/column scale 1",
               "r_towns": ""}
LINE_TEMPLATES = {
    "D": "D sub-label: {n} column pair(s) of SD v2 Tier A+B ({pairs}) are decidable on "
         "{units}{scale} (COLF OSSE, P(correct ranking) >= 0.80 in every nominal cell, same "
         "ranking at both piles); ranked by the registered screen, decided not before "
         "2030-01-15 (D-pending, R8.4b); expected wrong D labels {wrong:.2f}; no cut number "
         "changes",
    "R": "Report-only: {n} column pair(s) ({pairs}) can be ranked on {units}{scale} (COLF "
         "OSSE, P(correct ranking) >= 0.80); a report-only ranking, never a gate or a D; no "
         "cut number changes",
    "narrowing": "Narrowing: {units} cannot separate {n} leveling-equivalent column pair(s) "
                 "({pairs}) (COLF OSSE, nominal P(correct ranking) <= 0.50); no cut number "
                 "changes",
}


def pair_words(arm: str, key: str) -> str:
    """'a vs b' plus the members' unverified labels and the 大城鄉-only clause."""
    a, b = key.split("|")
    tags = [f"{m}: {C.MEMBER_SPEC[m]['label']}" for m in (a, b) if C.MEMBER_SPEC[m]["label"]]
    if arm == "r_towns" and {a, b} == {"q0_cs", "q0_csk1"}:
        tags.append("separable only at 大城鄉")
    return f"{a} vs {b}" + (f" [{'; '.join(tags)}]" if tags else "")


def page_lines(result: dict | None) -> dict:
    """{claim: [lines]} from the registered OSSE reading (templates if none)."""
    if result is None:
        return {"status": "pending: computed by osse.page_lines from the registered OSSE "
                          "result", "templates": LINE_TEMPLATES, "claims_by_arm":
                {k: list(v) for k, v in C.CLAIMS_BY_ARM.items()},
                "pre_worded": {"r_towns q0_cs|q0_csk1": pair_words("r_towns", "q0_cs|q0_csk1")}}
    dec = result["decidability"]
    if dec.get("status") != "ok":
        return {"status": "no lines: the OSSE self-check failed (engineering; fix and rerun)"}
    out = {c: [] for c in C.CLAIMS}
    for arm in C.ARM_ORDER:
        if arm not in dec["arms"]:
            continue
        for lab in ("D", "R", "narrowing"):
            ks = sorted(k for k, v in dec["arms"][arm].items() if v["label"] == lab)
            if not ks:
                continue
            line = LINE_TEMPLATES[lab].format(
                n=len(ks), pairs="; ".join(pair_words(arm, k) for k in ks),
                units=UNIT_WORDS[arm], scale=SCALE_WORDS[arm],
                wrong=dec["multiplicity"][arm]["expected_wrong_labels"])
            for claim in C.CLAIMS_BY_ARM[arm]:
                out[claim].append({"arm": arm, "label": lab, "line": line})
    return {"status": "ok", "lines": out,
            "multiplicity": dec["multiplicity"],
            "always": "the ranking says which column tracks the status-quo level better at "
                      "these units; it never selects a column, sets a factor or changes a "
                      "cut number (C03 stays W; E24 stays L, 'model-form spread, not an "
                      "interval')"}


# ----------------------------------------------------------------------------- draft
def _code_shas(repo: Path = REPO) -> dict:
    return {f: M.sha256(Path(repo) / f) for f in C.CODE_FILES}


def _inp(repo: Path, rel: str, sha: str | None = None) -> dict:
    p = Path(repo) / rel
    got = M.sha256(p)
    if sha is not None and got != sha:
        raise M.MemberError(f"{rel}: sha256 {got[:12]} != bound {sha[:12]}")
    return {"path": rel, "sha256": got}


def _n3_unit(repo: Path) -> dict:
    d = json.loads((Path(repo) / C.N3["registration"]).read_text(encoding="utf-8"))
    for t in d["tests"]:
        if t["test_id"] == C.N3["level_24m_test"]:
            return {C.N3["unit"]: t["external_units"][C.N3["unit"]]}
    raise M.MemberError("N3 registration has no level_24m unit")


def _n3sl_units(repo: Path) -> dict:
    d = json.loads((Path(repo) / C.N3SL["registration"]).read_text(encoding="utf-8"))
    out = {}
    for t in d["tests"]:
        out.update({u: v for u, v in (t.get("external_units") or {}).items()
                    if u in C.N3SL["units"].values()})
    if sorted(out) != sorted(C.N3SL["units"].values()):
        raise M.MemberError("N3S-L registration units differ")
    return out


def arm_units(repo: Path, arm: str) -> tuple[list[str], dict]:
    """(fresh_units, external_units) of an arm: the report-only arms' own unit ids carry the
    owners' members by value (colf_config.USE_OWNER_UNIT_IDS)."""
    a = C.ARMS[arm]
    if arm == "g_piles":
        return list(a["fresh_units"]), {C.G_UNIT: C.G_UNIT_MEMBERS}
    owner = _n3_unit(repo) if arm == "r_piles" else _n3sl_units(repo)
    if C.USE_OWNER_UNIT_IDS:
        return list(a["owner_units"]), owner
    ext = {}
    for mine, theirs in zip(a["fresh_units"], a["owner_units"], strict=True):
        v = dict(owner[theirs])
        v["description"] = (f"the same printed numbers as {theirs} ({a['owner']}), used "
                            f"here only for a report-only column ranking (R8.4a, one role "
                            f"per number): {v['description']}")
        ext[mine] = v
    return list(a["fresh_units"]), ext


def _xiutan(repo: Path) -> dict:
    """秀潭國小 70 m: an arm only if a deep_benchmarks_printed version >= v3 grants
    prospective_scorer BEFORE registration; a version appearing later cannot add an arm."""
    root = Path(repo) / C.XIUTAN["product_root"]
    found = []
    for d in sorted(root.glob("v*")) if root.is_dir() else []:
        try:
            v = int(d.name[1:])
        except ValueError:
            continue
        mf = d / "MANIFEST.json"
        if v >= C.XIUTAN["min_version"] and mf.is_file():
            m = json.loads(mf.read_text(encoding="utf-8"))
            found.append({"version": d.name, "manifest_sha256": M.sha256(mf),
                          "grants_prospective_scorer": "prospective_scorer" in
                          m.get("allowed_uses", [])})
    after = "a version >= v3 that appears after registration cannot add an arm (R8.4a)"
    if not found:
        return {"status": "excluded", "why": "no deep_benchmarks_printed version >= v3 exists; "
                "v1 and v2 do not list prospective_scorer in allowed_uses (R8.4a)",
                "after_registration": after}
    grants = any(f["grants_prospective_scorer"] for f in found)
    return {"status": "lead decision needed" if grants else "excluded", "versions": found,
            "after_registration": after}


def draft_prereg(seed, *, freeze: dict | None = None, osse_result: dict | None = None,
                 engineering: dict | None = None, repo: Path = REPO) -> dict:
    """The prereg3 document of COLF.column_family_2028.v1 for ``seed`` (required; no
    default). ``freeze`` = the freeze summary (members.freeze, with 'predictions_npz' and
    'freeze_summary_path'); ``osse_result`` = run_osse's result with 'result_path' and
    'result_sha256'. Without them the draft is valid but not ready (see readiness)."""
    seed = check_seed(seed, real=True)
    repo = Path(repo)
    code = _code_shas(repo)
    if osse_result is not None and int(osse_result["seed"]) != seed:
        raise ValueError("the OSSE result was run with another seed")
    inputs = [_inp(repo, f, s) for f, s in code.items()]
    inputs += [_inp(repo, C.SD_V2["registration"], C.SD_V2["registration_sha256"]),
               _inp(repo, C.SD_V2["report"], C.SD_V2["report_sha256"])]
    inputs += [_inp(repo, p, s) for p, s in C.FORWARDS.values()]
    for b, keys in ((C.N3, ("registration", "mapping", "model_side", "g0_result", "g0_code",
                            "n3lib", "scorer")),
                    (C.N3SL, ("registration", "cells", "scorer"))):
        inputs += [_inp(repo, b[k], b[f"{k}_sha256"]) for k in keys]
    inputs.append(_inp(repo, C.RULINGS["path"], C.RULINGS["sha256"]))
    for v in ("v1", "v2"):
        inputs.append(_inp(repo, f"results/dataprep/deep_benchmarks_printed/{v}/MANIFEST.json"))
    fz = None
    if freeze is not None:
        npz = freeze["predictions_npz"]
        inputs.append(_inp(repo, npz["path"], npz["sha256"]))
        if freeze.get("freeze_summary_path"):
            inputs.append(_inp(repo, freeze["freeze_summary_path"],
                               freeze.get("freeze_summary_sha256")))
        fz = {"predictions_npz": npz, "freeze_summary": freeze.get("freeze_summary_path"),
              "prediction_sha256": freeze["prediction_sha256"], "dedup": freeze["dedup"],
              "full_identity": freeze["full_identity"], "pile_cells": freeze["pile_cells"],
              "town_cells_sha256": freeze["town_cells_sha256"],
              "months": {"piles": freeze["piles_months"], "towns": freeze["towns_months"]}}
    if osse_result is not None:
        if osse_result.get("status") != "ok":
            raise ValueError("only a passing OSSE result can be registered")
        inputs.append(_inp(repo, osse_result["result_path"], osse_result["result_sha256"]))
    if engineering is not None:
        if int(engineering["seed"]) != seed:
            raise ValueError("the engineering record names another seed")
        inputs.append(_inp(repo, engineering["path"], engineering["sha256"]))
    tests = []
    for arm in C.ARM_ORDER:
        a = C.ARMS[arm]
        fu, ext = arm_units(repo, arm)
        t = {"test_id": a["test_id"], "role": a["role"], "data_scope": a["data_scope"],
             "fresh_units": fu, "external_units": ext,
             "owner_units": list(a.get("owner_units", [])),
             "claim": f"which SD v2 Tier A+B column tracks the status-quo level better on "
                      f"{UNIT_WORDS[arm]} (claims {', '.join(C.CLAIMS_BY_ARM[arm])})",
             "why_role": a["why_role"],
             "label": "prospective: the arm's printed values did not exist at registration",
             "decision_rule": RANKING_RULE[a["kind"]],
             "pairs": "every pair of the arm's deduplicated members (members.dedup); only "
                      "pairs labelled by the registered OSSE reading carry a page line",
             "informative_gate": (
                 f">= {a['min_months']} verified months at EACH primary pile at the declared "
                 f"sensor; otherwise not evaluable" if a["kind"] == "piles" else
                 ">= 3 listed coastal townships and Yunlin's list verified in at least one "
                 "survey year; otherwise not informative")}
        if a["kind"] == "piles":
            t["window"] = list(a["window"])
            t["ref_month"] = a["ref_month"]
        else:
            t["surveys"] = list(C.SURVEYS)
        tests.append(t)
    doc = {
        "id": C.ID, "wp": C.WP, "kind": "test",
        "title": "Column-family decidability OSSE and freeze: rank the SD v2 Tier A+B "
                 "leveling-equivalent columns on the 2028 pile and township units "
                 "(report-only on N3's and N3S-L's units, gating screen on the unowned "
                 "2028-12..2029-11 pile months)",
        "tests": tests,
        "fresh_units": sorted({u for t in tests for u in t["fresh_units"]}),
        "roles": {C.ARMS[a]["test_id"]: C.ARMS[a]["role"] for a in C.ARM_ORDER},
        "zones": {"not_applicable": "no fit; the frozen forwards' own zones"},
        "burn_in": {"not_applicable": "no fit; SD v2's forwards (hindcast 2012-01..2022-12, "
                                      "origin 2022-12)"},
        "folds": {"not_applicable": "prospective windows"},
        "gate_scores": {"not_applicable": "no weighted gate; each pair's ranking is reported "
                                          "as it falls"},
        "primary_model": {
            "name": "the SD v2 Tier A+B member set (deduplicated per arm), each on its own "
                    "paired heads; no primary column, no selection",
            "sd_v2_registration_sha256": C.SD_V2["registration_sha256"]},
        "members": member_block(),
        "q0_nh": {"status": "included from the fresh origin-2022-12 forward sd_q0nh.npz "
                            "(722c5cd5; SD v1 run 2026-11-01, --hindcast-gain 1.0 "
                            "--hindcast-every 1, never --extend-record-to; members.load_forward "
                            "checks origin, gain, options); label 'unverified pairing (no "
                            "clean reference)'; no new forward is needed"},
        "xiutan_70m": _xiutan(repo),
        "ranking_rule": RANKING_RULE,
        "osse_design": osse_design(),
        "osse_seed": seed,
        "engineering_record": (engineering or {"status": f"pending: osse.py record writes "
                                                         f"{C.ENGINEERING_RECORD} once"}),
        "osse_result": ({"path": osse_result["result_path"],
                         "sha256": osse_result["result_sha256"],
                         "self_check": osse_result["self_check"],
                         "decidability": osse_result["decidability"]}
                        if osse_result is not None else
                        {"status": "pending: the lead runs osse.py run on the frozen "
                                   "predictions before registering"}),
        "thresholds": {"p_d": C.P_D, "p_narrow": C.P_NARROW, "self_check_tol": C.SELF_CHECK_TOL,
                       "reps": C.REPS, "tie_abs": C.TIE_ABS,
                       "min_months": {a: C.ARMS[a].get("min_months") for a in C.ARM_ORDER
                                      if C.ARMS[a]["kind"] == "piles"},
                       "bounds": "inclusive, unrounded; P is the minimum over the pair's "
                                 "NOMINAL cells (truth x noise x availability); the ac300, "
                                 "rho and g_mod_cell cells are report-only"},
        "frozen_predictions": fz or {"status": "pending: members.py freeze --out <dir> "
                                               "(written once)"},
        "page_lines": page_lines(osse_result),
        "page_mapping": {
            "before_scoring": "the registered page_lines: E24, C09 and C11 gain D sub-labels "
                              "(gating arm, P >= 0.80), report-only ranking lines (report-only "
                              "arms, P >= 0.80) or narrowing lines (P <= 0.50); pairs in "
                              "between get no line",
            "after_scoring": "gating-arm D pairs: '<winner> tracks the 2028-12..2029-11 pile "
                             "status-quo level better than <loser> at pile/column scale 1 "
                             "(COLF screen, prospective)', 'split' or 'not evaluable'; R pairs "
                             "the same wording, report-only; narrowing and neither pairs: "
                             "'not decidable (P=...)', never a ranking sentence",
            "never": "no cut number, no factor, no column selection and no status move of "
                     "C03 (W), E24 (L) or the C09 fan total (W); members are never added "
                     "after the freeze"},
        "scoring": {
            "not_before": C.NOT_BEFORE.strftime("%Y-%m-%d"),
            "no_later_than": C.NO_LATER_THAN.strftime("%Y-%m-%d"),
            "once": "one scoring session after N3 (24-month look or its not_scorable record) "
                    "and N3S-L ROC117 (score look or its not_published record); the report is "
                    "written once (mode 'x')",
            "precondition": "the pile product (deep_benchmarks_printed v3 or later) and "
                            "subsidence_public_reports/prospective grant prospective_scorer, "
                            "and the pile MANIFEST states data_end >= 2029-11, checked from "
                            "the MANIFESTs before any read; otherwise the arm is not "
                            "evaluable, nothing read",
            "records": "a not-scorable record of N3 or N3S-L is accepted only by path and "
                       "sha256, both written into the report",
            "scorer": f"{C.E}/score_colf.py score",
            "d_pending": f"a D decided after {C.D_PENDING_AFTER} counts as D-pending (R8.4b)"},
        "sealed_until_scoring": "every pile month after 2026-11 and every ROC116/117 township "
                                "summary; no statistic of an arm's window is computed before "
                                "the scoring look",
        "attempt_policy": {
            "engineering_retries": "free before the decidability reading exists (loader, "
                                   "freeze or OSSE crash, a failed self-check: it writes a "
                                   "numbered failure record with no P), each logged in "
                                   f"{C.ATTEMPTS}; a fix that changes frozen code after "
                                   "registration needs a new id disclosing the attempts",
            "fixed_paths": {"engineering_record": C.ENGINEERING_RECORD,
                            "attempts": C.ATTEMPTS, "result": C.OSSE_RESULT,
                            "self_check_failure": C.SELF_CHECK_FAILURE},
            "seed": "fixed by the engineering record (mode 'x') before any OSSE run; the run "
                    "refuses any other seed or freeze",
            "scored_runs": "the OSSE runs once on the frozen predictions with the registered "
                           "seed; the scorer runs once (score once)",
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "never": "no rerun with another seed, no member added after the freeze, no "
                     "threshold changed after the reading",
            "report": "every attempt, failures included"},
        "frozen_code": code,
        "rulings": {"R8.4a": "report-only ranking on N3's months 2026-12..2028-11 (one role "
                             "per number); gating on 2028-12..2029-11, not evaluable under 8 "
                             "months; 秀潭 70 m excluded unless deep_benchmarks_printed v3 "
                             "grants prospective_scorer",
                    "R8.4b": "a D decided after 2030-01-15 counts as D-pending",
                    "file": C.RULINGS},
        "disclosures": disclosures(),
        "priors": "Model-only: no observation is read. Expected reading: the pile arms "
                  "separate pairs whose columns differ in mid-zone Ske (the q0 family vs the "
                  "page family) and cannot separate tau30 variants of the same column; the "
                  "coastal township arm separates almost nothing (the columns barely move on "
                  "the coast, N3S-L physics_near_z). Pass chance that at least one pair is "
                  "decidable about 0.5 (judge).",
        "interpretation": "Whether the 2028 pile and township units can rank leveling-"
                          "equivalent columns at all, and, for decidable pairs, which column "
                          "tracks the status-quo level better there. A ranking narrows "
                          "E24/C09/C11 wording only; it is not evidence about cut sizes and "
                          "never selects a column.",
        "why": "SD v2 measured the model-form spread of the cut response across columns "
               "that fit leveling equally well; no registered test can tell those columns "
               "apart. This declares, before the 2028 data exist, which pairs the units can "
               "separate and freezes every member's predictions.",
        "inputs": inputs,
    }
    return doc


RANKING_RULE = {
    "piles": "the member closer by RMSE wins: per primary pile and verified month m of the "
             "window, the cumulative status-quo (baseline) compaction since the reference "
             "month summed over verified months only (member unscaled, N3's depth "
             "convention) against the printed increments summed the same way (flag 0, the "
             "declared sensor: 土庫國中 400 m, 300 m only if 400 m has fewer than min_months "
             "and 300 m at least that many, never mixed); RMSE pooled over pile-months; "
             f"|RMSE_a - RMSE_b| <= {C.TIE_ABS:g} cm is a tie",
    "towns": "the member closer by RMSE wins: per survey year (ROC116, ROC117; the printed "
             "survey interval, else the nominal one) and coastal township (N3S-L's 11), the "
             "member's baseline peak rate (max over the township's cells) against WRA's "
             "printed bound row (N3S-L check_bounds semantics): shortfall below 3 - 0.5 "
             "cm/yr on listed rows plus excess over upper + 0.5 cm/yr, ambiguous rows "
             "dropped; RMSE over the rows of informative years (>= 3 listed, Yunlin's list "
             f"verified); |RMSE_a - RMSE_b| <= {C.TIE_ABS:g} is a tie",
}


def member_block() -> dict:
    return {"set": "SD.cut_response_spread.v2 report tiers['A+B'] (11 columns)",
            "spec": C.MEMBER_SPEC, "forwards": {k: {"path": v[0], "sha256": v[1]}
                                                for k, v in C.FORWARDS.items()},
            "aliases": C.ALIASES,
            "dedup_rule": "an alias is collapsed onto its canonical member on an arm only "
                          "where their frozen predictions on that arm's cells and months are "
                          "byte-identical (members.dedup); otherwise both stay members of "
                          "that arm",
            "never": "no member is added after the freeze; no new fit"}


def osse_design() -> dict:
    return {
        "truths": "truth in {A, B} for each pair (model-only: the member's own frozen "
                  "baseline prediction)",
        "nominal_cells": {"piles": "truth x G0 noise (sd {0.75, 1.5} m x phi {0.5, 0.9}) x "
                                   "availability {full, minimum: exactly min_months random "
                                   "months per pile and replicate (8 of 12 gating, 16 of 24 "
                                   "report-only)}",
                          "towns": "truth x sigma {0.5, 1.0, 1.5} cm/yr"},
        "pile_noise": {"source": f"N3 G0 ({C.N3['g0_code']} {C.N3['g0_code_sha256'][:8]}, "
                                 f"result {C.N3['g0_result_sha256'][:8]}; n3lib "
                                 f"{C.N3['n3lib_sha256'][:8]})",
                       "head_error_sd_m": list(C.G0_SD_GRID_M), "phi": list(C.G0_PHI_GRID),
                       "print_res_cm": C.PRINT_RES_CM,
                       "g_mod_cm_per_m": C.G_MOD_CM_PER_M,
                       "g_mod_source": f"N3 model_side.json g_mod_12 "
                                       f"({C.N3['model_side_sha256'][:8]})",
                       "construction": "obs increment = round_0.1(share x (rho x truth "
                                       "increment + g x diff(e))), e AR(1) on the head level, "
                                       "one series per pile and replicate; primary cell rho 1, "
                                       "g = g_mod (pile/column scale 1)",
                       "builder_construction": "G0 put this error on the nest head change and "
                                               "simulated no pile-side noise beyond rounding; "
                                               "passing it to the pile through g_mod is the "
                                               "builder's construction, not G0's model"},
        "report_only_cells": {
            "ac300": f"土庫國中 400 m printed for the first {C.AC_400_MONTHS} months only, so "
                     f"the sensor rule scores the 300 m series with share {C.F300} (N3's "
                     f"report-only anchor-depth share); 中科虎尾 unchanged",
            "rho": f"G0's conversion truths {list(C.RHO_SENS)} (truth increments x rho)",
            "g_mod_cell": f"g = {C.G_MOD_CELL_CM_PER_M} cm/m (N3 model_side "
                          f"g_mod_cell_report_only) in place of g_mod",
            "role": "reported beside each pair's P; never in P, a label or a page line"},
        "pile_ranking": "per-pile agreement: a replicate counts as correct only if the truth "
                        "member is closer at both primary piles",
        "township_noise": {"source": f"N3S-L ({C.N3SL['scorer']} "
                                     f"{C.N3SL['scorer_sha256'][:8]}: l1_passability, "
                                     f"check_bounds)",
                           "sigma_cm_yr": list(C.TOWN_SIGMA_GRID), "tol_cm_yr": C.TOL,
                           "listed_floor_cm_yr": C.LISTED_FLOOR,
                           "min_listed": C.L1_MIN_LISTED,
                           "construction": "peak + N(0, sigma) per township; listed = peak >= "
                                           "3; upper = the year's maximum listed peak when "
                                           "listed, else 3"},
        "reps": C.REPS,
        "p_correct": "P(the truth member is ranked first); the pair's P = min over its "
                     "NOMINAL cells",
        "self_check": {
            "order": "per arm, before any P(correct); if any arm fails, no P is computed and "
                     "only the failure record is written (osse_selfcheck_failure_<k>.json); a "
                     "repair is an engineering retry",
            "swap": "the outcome for (B, A) equals (A, B)'s mirrored, same seed, exactly",
            "piles_null": f"midpoint truth (A + B) / 2: |P(A first) + P(not ranked)/2 - 0.5| "
                          f"<= {C.SELF_CHECK_TOL} in every nominal cell",
            "towns_null": f"the shared value (A + B) / 2 as truth with the two competitors' "
                          f"labels drawn at random per replicate: |P_sym - 0.5| <= "
                          f"{C.SELF_CHECK_TOL} in every nominal cell",
            "towns_null_why": "the bound metric is one-sided (a peak below the year's highest "
                              "listed peak and above 2.5 cm/yr scores 0, a peak above it is "
                              "penalised), so the midpoint is not equidistant and a midpoint "
                              "null fails by construction (review B1). Under an uninformative "
                              "pairing, i.e. labels carrying no information about which "
                              "series is which, any label-symmetric engine gives 0.5; the "
                              "null therefore tests the engine's symmetry and tie handling "
                              "(the exact swap check tests it deterministically), not the "
                              "metric's geometry",
            "aliases": "each collapsed alias pair with truth = canonical gives exactly 0.5"},
        "labels": "nominal P >= 0.80: 'D' (gating arm) or 'R' (report-only arms); nominal P "
                  "<= 0.50: 'narrowing'; else 'neither'",
        "multiplicity": "per arm, sum of (1 - P) over the D (or R) pairs = the expected number "
                        "of wrong labels (an upper bound: P is a minimum over cells)",
        "seeding": "SeedSequence(osse_seed) -> per arm (self-check, decision) -> one child per "
                   "pair and cell in sorted order",
        "reads": "the frozen predictions only (members.load_freeze); no observation"}


def disclosures() -> dict:
    return {
        "dedup_q0_csk1": "q0_cs = q0_csk1 holds by bytes on the pile arms (cells 541, 769, "
                         "all freeze months) and on the 10 Yunlin coastal townships, but NOT "
                         "on 大城鄉 (the columns differ in the mid_changhua zone; 736 cells "
                         "differ in the full field): q0_csk1 is a distinct member of the "
                         "township arm (members.dedup evidence)",
        "dedup_q0_s29": "q0 = q0_s29: the whole subs_mean_by_rheology arrays are "
                        "byte-identical (both scenarios, every cell and month)",
        "osse_is_model_only": "every truth is a member (or a pair midpoint): P(correct) is "
                              "the decidability if the family contains the truth; structural "
                              "error outside the family is not simulated, so P is an upper "
                              "bound on the real decidability",
        "pile_noise_transport": "G0 puts its AR(1) error on the nest head change; here it "
                                "reaches the pile through N3's single frozen g_mod (page "
                                "column, displayed heads) for every member",
        "anchor": "the column is total depth, compared unscaled with 300/400 m anchors (N3); "
                  "the ac300 cell is report-only (review S2): the scorer moves 土庫國中 to "
                  "300 m only under N3's availability rule",
        "per_pile_agreement": "a pile pair is ranked only if the same member is closer at "
                              "both primary piles (N3's 'every primary pile' rule); a split "
                              "is reported, never a ranking",
        "pile_column_scale": "every D and R line holds at pile/column scale 1 (rho 1, g_mod); "
                             "the rho and g_mod_cell cells show how P moves off scale 1",
        "multiplicity": "the reading reports sum(1 - P) over the D pairs = the expected "
                        "number of wrong D labels (and the same for R pairs)",
        "roc118_forfeited": "ROC118 (2028-06..2029-05) is not an arm: R8.4a's gating months "
                            "are pile months; the survey year is forfeited for this "
                            "declaration",
        "not_comparable_to_n3": "N3's ZR level verdict (datum_gate_innov, against zero and the "
                                "leveling-cell rate) and this ranking (sd_page and the other "
                                "members against each other) are not comparable and neither "
                                "reads the other",
        "xiutan_after_registration": "a deep_benchmarks_printed v3 that appears after "
                                     "registration cannot add a 秀潭國小 arm",
        "q0_csk1_line": "q0_cs vs q0_csk1 on the township arm is pre-worded 'separable only at "
                        "大城鄉'",
        "unverified_in_lines": "every page line naming page_aquitard or q0_nh carries its "
                               "'unverified' label",
        "role_reading": "R8.4a gives the N3 and N3S-L numbers a report-only ranking role, so "
                        "those arms carry report-only ranking lines, never D sub-labels; only "
                        "the gating arm's decidable pairs carry a D sub-label",
        "looked_metadata": "the builder read SD v2's registration and report metadata, the "
                           "column JSONs (their spent fit metrics appeared in a diff), N3's "
                           "and N3S-L's registrations, N3 mapping/model_side (g_mod) and G0 "
                           "result metadata, N3S-L's cells.csv (geometry) and the "
                           "deep_benchmarks_printed v1/v2 MANIFESTs; no pile, leveling or "
                           "township observation",
        "no_osse_run_by_builder": "the builder ran no OSSE and computed no P on real members",
        "xiutan": "秀潭國小 70 m has no arm (R8.4a; xiutan_70m)",
        "own_unit_ids": "the report-only arms read the same printed numbers as N3's and "
                        "N3S-L's units under their own unit ids (owner_units name the "
                        "owners'): the owners' ids are blocked for a new registration by the "
                        "look R8:fv_stat_v2_builder_reads_disclosure (2026-11-03T01:39Z, "
                        "universal members, window 2026-11..2027-10, E04 model outputs only; "
                        "no pile or township value); the numbers keep one role here "
                        "(report-only ranking) and N3/N3S-L keep theirs. The lead's "
                        f"correction row {C.LOOK_SCOPE_CORRECTION['unit']} "
                        f"({C.LOOK_SCOPE_CORRECTION['time']}, looks.jsonl row "
                        f"{C.LOOK_SCOPE_CORRECTION['row']}, chain hash "
                        f"{C.LOOK_SCOPE_CORRECTION['chain_hash'][:16]}, "
                        f"{C.LOOK_SCOPE_CORRECTION['chain_hash_source']}) records that row's "
                        "actual reads; it cannot change the row's members, so the owners' "
                        "ids stay blocked",
        "page_member_is_not_n3_zr": "the 'page' member is SD v2's sd_page forward (main "
                                    "parameter set, gain 0, free column heads), not N3's "
                                    "frozen ZR M (datum_gate_innov, 18-member mean); N3's "
                                    "level verdict and this ranking use different page "
                                    "trajectories and are not comparable",
        "status_quo": "the baseline scenario (climatology pumping after the 2022-12 origin) "
                      "is the status-quo level; cut30 is not used",
    }


def readiness(doc: dict, now: dt.datetime | None = None) -> list[str]:
    """Blockers before the lead may register (empty = ready)."""
    out = []
    now = now or dt.datetime.now(C.UTC)
    if "status" in doc["frozen_predictions"]:
        out.append("frozen predictions pending")
    if "status" in doc["engineering_record"]:
        out.append("engineering record pending")
    r = doc["osse_result"]
    if "status" in r:
        out.append("OSSE result pending")
    elif not r["self_check"]["pass"]:
        out.append("OSSE self-check failed")
    fz = doc["frozen_predictions"].get("predictions_npz", {}).get("sha256")
    er = doc["engineering_record"]
    if fz and "status" not in er and er.get("freeze_npz_sha256") != fz:
        out.append("the engineering record names another freeze")
    if now > C.REG_DEADLINE:
        out.append("after the registration deadline 2026-11-30T16:00Z")
    if doc["xiutan_70m"]["status"] != "excluded":
        out.append("秀潭 70 m: deep_benchmarks_printed v3 exists; lead decision needed")
    return out


def probe(doc: dict) -> dict:
    """The prereg3 blocking probe on the draft's tests (read-only; write_prereg's own
    check): looks that would block registration now, and the unverifiable ones."""
    from harness import prereg3

    idx = prereg3.membership_index()
    for t in doc["tests"]:
        for u, v in (t.get("external_units") or {}).items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    now = dt.datetime.now(C.UTC)
    block = prereg3.blocking_looks(doc["tests"], None, prereg3.PREREG_DIR, idx, test_time=now)
    unver = prereg3.unverifiable_looks(doc["tests"], None, prereg3.PREREG_DIR, idx)
    return {"blocking": block, "n_unverifiable": len(unver),
            "unverifiable_tests": sorted({u["test_id"] for u in unver})}


def validate(doc: dict) -> list[str]:
    from harness import prereg3

    return prereg3.validate_prereg(doc)


# ----------------------------------------------------------------------------- runs
class RunRefused(RuntimeError):
    """The OSSE run refuses (no engineering record, another seed or freeze, a rerun)."""


def _write_once(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True, default=float)
            + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def _attempt(attempts: Path, rec: dict) -> None:
    attempts.parent.mkdir(parents=True, exist_ok=True)
    with open(attempts, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": dt.datetime.now(C.UTC).isoformat(), **rec},
                           ensure_ascii=False, default=str) + "\n")


def _rel(p) -> str:
    """Repository-relative path without resolving the results/ symlinks."""
    return os.path.relpath(os.path.abspath(p), REPO)


def _paths(repo: Path, base) -> dict:
    """The fixed paths (colf_config), or the same names under ``base`` (unit tests only)."""
    if base is None:
        return {"record": Path(repo) / C.ENGINEERING_RECORD,
                "attempts": Path(repo) / C.ATTEMPTS, "result": Path(repo) / C.OSSE_RESULT,
                "failure": lambda k: Path(repo) / C.SELF_CHECK_FAILURE.format(k=k)}
    b = Path(base)
    return {"record": b / Path(C.ENGINEERING_RECORD).name,
            "attempts": b / Path(C.ATTEMPTS).name, "result": b / Path(C.OSSE_RESULT).name,
            "failure": lambda k: b / Path(C.SELF_CHECK_FAILURE.format(k=k)).name}


def write_engineering_record(seed, freeze_npz, *, repo: Path = REPO, base=None,
                             real: bool = True) -> dict:
    """The seed and the freeze, written once (mode 'x') before any OSSE run."""
    seed = check_seed(seed, real=real)
    P = _paths(repo, base)
    sha_f = M.sha256(freeze_npz)
    rec = {"id": C.ID, "seed": seed, "freeze_npz": _rel(freeze_npz),
           "freeze_npz_sha256": sha_f, "osse_version": OSSE_VERSION, "reps": C.REPS,
           "code": _code_shas(repo), "written": dt.datetime.now(C.UTC).isoformat()}
    sha = _write_once(P["record"], rec)
    _attempt(P["attempts"], {"event": "engineering_record", "seed": seed,
                             "freeze_npz_sha256": sha_f, "record_sha256": sha})
    return {"path": _rel(P["record"]), "sha256": sha, "seed": seed}


def run_registered(seed, freeze_npz, *, repo: Path = REPO, base=None, reps: int = C.REPS,
                   real: bool = True) -> dict:
    """The one OSSE run: refuses without the engineering record, with another seed or
    freeze, or when the result exists. A failed self-check writes a numbered failure record
    (no P) instead of the result, so the repair stays an engineering retry."""
    P = _paths(repo, base)
    if not P["record"].is_file():
        raise RunRefused(f"write the engineering record first ({C.ENGINEERING_RECORD})")
    rec = json.loads(P["record"].read_text(encoding="utf-8"))
    if int(seed) != int(rec["seed"]):
        raise RunRefused(f"seed {seed} != the engineering record's seed {rec['seed']}")
    if M.sha256(freeze_npz) != rec["freeze_npz_sha256"]:
        raise RunRefused("the frozen predictions differ from the engineering record's")
    if P["result"].exists():
        raise RunRefused(f"{P['result']} exists: the OSSE runs once")
    code = _code_shas(repo)
    base_rec = {"event": "osse_run", "seed": int(seed), "code": code,
                "record_sha256": M.sha256(P["record"])}
    try:
        frozen = M.load_freeze(freeze_npz, rec["freeze_npz_sha256"])
        res = run_osse(frozen, seed, reps=reps, real=real)
    except Exception as exc:
        _attempt(P["attempts"], {**base_rec, "rc": 1, "error": repr(exc)[:300]})
        raise
    res.update({"freeze_npz_sha256": rec["freeze_npz_sha256"], "code": code,
                "engineering_record_sha256": base_rec["record_sha256"]})
    if res["status"] != "ok":
        k = 1
        while P["failure"](k).exists():
            k += 1
        sha = _write_once(P["failure"](k), res)
        _attempt(P["attempts"], {**base_rec, "rc": 2, "self_check_failure": k,
                                 "failure_sha256": sha})
        return {"status": "self_check_failed", "path": str(P["failure"](k)), "sha256": sha}
    sha = _write_once(P["result"], res)
    _attempt(P["attempts"], {**base_rec, "rc": 0, "result_sha256": sha})
    return {"status": "ok", "path": str(P["result"]), "sha256": sha}


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "run"):
        r = sub.add_parser(name)
        r.add_argument("--seed", type=int, required=True)
        r.add_argument("--freeze-npz", type=Path, required=True)
    d = sub.add_parser("draft")
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--freeze-summary", type=Path)
    d.add_argument("--freeze-npz", type=Path)
    d.add_argument("--osse-result", type=Path)
    d.add_argument("--engineering-record", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(json.dumps(write_engineering_record(a.seed, a.freeze_npz)))
        return 0
    if a.cmd == "run":
        out = run_registered(a.seed, a.freeze_npz)
        print(json.dumps(out))
        return 0 if out["status"] == "ok" else 2
    freeze = osse = eng = None
    if a.freeze_summary:
        freeze = json.loads(a.freeze_summary.read_text(encoding="utf-8"))
        npz = a.freeze_npz or a.freeze_summary.parent / freeze["predictions_npz"]["path"]
        freeze["predictions_npz"] = {"path": _rel(npz),
                                     "sha256": freeze["predictions_npz"]["sha256"]}
        freeze["freeze_summary_path"] = _rel(a.freeze_summary)
        freeze["freeze_summary_sha256"] = M.sha256(a.freeze_summary)
    if a.osse_result:
        osse = json.loads(a.osse_result.read_text(encoding="utf-8"))
        osse["result_path"] = _rel(a.osse_result)
        osse["result_sha256"] = M.sha256(a.osse_result)
    if a.engineering_record:
        e = json.loads(a.engineering_record.read_text(encoding="utf-8"))
        eng = {"path": _rel(a.engineering_record), "sha256": M.sha256(a.engineering_record),
               "seed": e["seed"], "freeze_npz_sha256": e["freeze_npz_sha256"]}
    doc = draft_prereg(a.seed, freeze=freeze, osse_result=osse, engineering=eng)
    errs = validate(doc)
    print("validate_prereg:", errs or "valid")
    print("readiness:", readiness(doc) or "ready")
    pr = probe(doc)
    print("blocking probe:", pr["blocking"] or "no blocking look",
          f"({pr['n_unverifiable']} unverifiable prior looks, disclosed at write)")
    return 1 if errs or pr["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
