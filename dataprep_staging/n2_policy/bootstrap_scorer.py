"""The shared N2/N2C scorer: beta_response, its 5 km block cluster bootstrap, the informative
gate, the RMSEs and the decision rule, exactly as the two registrations specify them.

N2C.changhua_response.prospective_2026-11.v1 (scoring.same_session) requires "the scorer code
for beta_response and its 5 km block bootstrap" to be "one implementation shared with N2,
hashed and logged before step (1)". This module is that implementation. It is written and
hashed in round 5 (user ruling 2026-11-01, fourth set, rank 10), well before the 2028-01-15
session, so that X5.n2c_reading.v1 characterises the scorer that will run. It reads no file;
the 2028 scoring wrappers load the window heads, the registered ZR and the CPU contrast and
pass the well-month panel in.

What the registrations fix (N2 and N2C, primary_metric / tests[0].metric, decision_rule,
informative_gate, thresholds), and how it is implemented here:

- beta_response: "OLS slope of e = obs - ZR on c = PR - ZR over the well-months of the
  window (flags 0-1 ...), with well fixed effects (e and c demeaned per well)". Rows are the
  well-months with flag 0 or 1 and finite obs, ZR and c; e and c are demeaned per well over
  those rows; beta = sum(e~ c~) / sum(c~^2), identical to the dummy-variable regression
  (Frisch-Waugh-Lovell).
- interval: "90 % from a 5 km block cluster bootstrap, B 2000, seed 20261101". Blocks are
  5 km squares on TWD97 metres with origin (0, 0) (``block_ids``, the definition N1 step C
  and E02c use). One draw resamples the blocks that hold scored rows, with replacement, as
  many as there are, and refits beta on the resampled wells (a well drawn m times counts m
  times; the per-well demeaning is unchanged by duplication, so the refit is exact with
  per-well sums). Draws: ``numpy.random.default_rng(seed).integers(0, n_blocks, (B,
  n_blocks))`` over the sorted block ids. Interval: the 5 % and 95 % quantiles of the draws
  (numpy's default linear rule), draws with a zero denominator dropped and counted.
- informative gate: "evaluated from the model and the forcing only, before any window head
  is read: RMS of the well-demeaned contrast over the county's well-months >= 0.05 m and >= 5
  bootstrap clusters"; "if fewer than 6 window months carry a recorded r, 'not
  informative'". ``informative_gate`` takes the contrast panel of every eligible well and
  window month (no head) and the number of recorded months.
- RMSE of PR, ZR and CL: the registered prospective test's convention (twin.validation:
  equal weight per well, the mean over wells of each well's mean squared error, then the root)
  on the same rows as beta; PR = ZR + c. Pooled RMSEs are reported beside them.
- decision rule, in this order: not_informative if the informative gate fails; not_supported
  if the 90 % interval includes 0 (lo <= 0 <= hi) or beta <= 0; response_supported if the
  point lies in [1/1.5, 1.5] (inclusive) and RMSE(PR) < RMSE(ZR); otherwise
  direction_supported_size_not. ``skill_vs_cl`` (RMSE(PR) < RMSE(CL)) is reported for the
  skill wording and never changes the verdict. The rule is applied literally: an interval
  lying wholly below 0 (hi < 0) with beta > 0 matches neither not_supported clause and goes on
  to the size test. A scorer may not amend a registered rule, so that case is not overridden;
  it is flagged (``interval_below_zero_with_positive_beta``) for the user's attention.
- declared secondaries that use the same estimator: ``detrend=True`` (N2C 'detrended': e and
  c detrended per well, linear in time, before the slope). The level component (window-mean e
  on window-mean c across wells, OLS with intercept) is report-only.

``python bootstrap_scorer.py --describe`` prints the constants and this file's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCORER_ID = "n2_policy.bootstrap_scorer"
VERSION = "1.0.0"
B_BOOT = 2000
SEED = 20261101
LEVEL = 0.90
BLOCK_KM = 5.0
RATIO_BAND = 1.5
INFORMATIVE_HEAD_M = 0.05
MIN_CLUSTERS = 5
MIN_MONTHS = 6
FLAGS_SCORED = (0, 1)
VERDICTS = ("response_supported", "direction_supported_size_not", "not_supported",
            "not_informative")

REGISTERED = {                 # the registrations' own words for each constant
    "B_BOOT": "interval: 90 % from a 5 km block cluster bootstrap, B 2000, seed 20261101",
    "SEED": "seed 20261101 (N2 primary_metric.interval; N2C tests[0].seed)",
    "LEVEL": "thresholds.level 0.9",
    "BLOCK_KM": "5 km block cluster bootstrap",
    "RATIO_BAND": "thresholds.ratio_band 1.5 (point estimate in [1/1.5, 1.5])",
    "INFORMATIVE_HEAD_M": "thresholds.informative_head_m 0.05",
    "MIN_CLUSTERS": "thresholds.min_clusters 5",
    "MIN_MONTHS": "thresholds.min_months 6 (window months with a recorded r)",
    "FLAGS_SCORED": "flags 0-1",
}


def file_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def describe() -> dict:
    return {"scorer": SCORER_ID, "version": VERSION, "sha256": file_sha256(),
            "constants": {"B_BOOT": B_BOOT, "SEED": SEED, "LEVEL": LEVEL,
                          "BLOCK_KM": BLOCK_KM, "RATIO_BAND": RATIO_BAND,
                          "INFORMATIVE_HEAD_M": INFORMATIVE_HEAD_M,
                          "MIN_CLUSTERS": MIN_CLUSTERS, "MIN_MONTHS": MIN_MONTHS,
                          "FLAGS_SCORED": list(FLAGS_SCORED)},
            "registered_wording": REGISTERED}


# ----------------------------------------------------------------------------- blocks
def block_ids(x, y, km: float = BLOCK_KM) -> np.ndarray:
    """5 km square blocks on TWD97 metres (fixed origin 0, 0): ``"<ix>_<iy>"``."""
    s = km * 1000.0
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("block_ids: non-finite coordinate")
    bx = np.floor(x / s).astype(np.int64)
    by = np.floor(y / s).astype(np.int64)
    return np.array([f"{a}_{b}" for a, b in zip(bx, by, strict=True)])


# ----------------------------------------------------------------------------- estimator
def _codes(labels) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return uniq, inv


def within(v: np.ndarray, inv: np.ndarray, n: int, t: np.ndarray | None = None,
           detrend: bool = False) -> np.ndarray:
    """``v`` residualised per group ``inv`` (0..n-1): demeaned, or (``detrend``) residuals
    of a per-group least-squares line in ``t``. A group with one row (or, when detrending,
    one distinct time) has residual 0."""
    v = np.asarray(v, float)
    cnt = np.bincount(inv, minlength=n).astype(float)
    mv = np.bincount(inv, weights=v, minlength=n) / np.maximum(cnt, 1.0)
    r = v - mv[inv]
    if not detrend:
        return r
    t = np.asarray(t, float)
    mt = np.bincount(inv, weights=t, minlength=n) / np.maximum(cnt, 1.0)
    tc = t - mt[inv]
    stt = np.bincount(inv, weights=tc * tc, minlength=n)
    stv = np.bincount(inv, weights=tc * r, minlength=n)
    slope = np.divide(stv, stt, out=np.zeros(n), where=stt > 0)
    return r - slope[inv] * tc


def well_sums(e, c, wells, t=None, detrend: bool = False
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(well ids, sum e~ c~ per well, sum c~^2 per well) after the per-well transform."""
    uniq, inv = _codes(wells)
    n = len(uniq)
    et = within(e, inv, n, t, detrend)
    ct = within(c, inv, n, t, detrend)
    return (uniq, np.bincount(inv, weights=et * ct, minlength=n),
            np.bincount(inv, weights=ct * ct, minlength=n))


def slope_fe(e, c, wells, t=None, detrend: bool = False) -> float:
    """The per-well fixed-effects slope of ``e`` on ``c`` (nan if c~ is identically 0)."""
    _, sxy, sxx = well_sums(e, c, wells, t, detrend)
    den = float(sxx.sum())
    return float(sxy.sum() / den) if den > 0 else float("nan")


def block_bootstrap(sxy: np.ndarray, sxx: np.ndarray, well_blocks, B: int = B_BOOT,
                    seed: int = SEED, level: float = LEVEL) -> dict:
    """Percentile interval of the per-well FE slope under resampling of blocks with
    replacement (see the module docstring for the exact draw)."""
    ub, binv = _codes(well_blocks)
    nb = len(ub)
    sxy_b = np.bincount(binv, weights=sxy, minlength=nb)
    sxx_b = np.bincount(binv, weights=sxx, minlength=nb)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, nb, size=(B, nb))
    cnt = np.zeros((B, nb))
    np.add.at(cnt, (np.repeat(np.arange(B), nb), draw.ravel()), 1.0)
    num, den = cnt @ sxy_b, cnt @ sxx_b
    ok = den > 0
    betas = num[ok] / den[ok]
    q = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0)
    if betas.size == 0:
        lo = hi = float("nan")
    else:
        lo, hi = (float(v) for v in np.quantile(betas, q))
    return {"lo": lo, "hi": hi, "n_blocks": int(nb), "B": int(B), "seed": int(seed),
            "level": float(level), "n_draws_used": int(betas.size),
            "n_draws_zero_denominator": int((~ok).sum())}


# ----------------------------------------------------------------------------- gate, rmse
def informative_gate(c, wells, well_blocks: dict, n_recorded_months: int) -> dict:
    """The registered gate from the contrast alone: ``c`` over every eligible well and window
    month (``wells`` row labels; no head is involved), ``well_blocks`` {well: block}."""
    c = np.asarray(c, float)
    if not np.isfinite(c).all():
        raise ValueError("informative_gate: the contrast panel must be finite")
    uniq, inv = _codes(wells)
    ct = within(c, inv, len(uniq))
    rms = float(np.sqrt(np.mean(ct ** 2))) if ct.size else 0.0
    nb = len({well_blocks[w] for w in uniq})
    out = {"rms_well_demeaned_contrast_m": rms, "threshold_m": INFORMATIVE_HEAD_M,
           "n_clusters": int(nb), "min_clusters": MIN_CLUSTERS,
           "n_recorded_months": int(n_recorded_months), "min_months": MIN_MONTHS,
           "n_wells": int(len(uniq)), "n_well_months": int(c.size)}
    out["passes"] = bool(rms >= INFORMATIVE_HEAD_M and nb >= MIN_CLUSTERS
                         and n_recorded_months >= MIN_MONTHS)
    return out


def rmse(resid, wells) -> dict:
    """Equal-well RMSE (the registered prospective test's convention) and pooled RMSE."""
    r2 = np.asarray(resid, float) ** 2
    uniq, inv = _codes(wells)
    per = np.bincount(inv, weights=r2, minlength=len(uniq)) / np.bincount(inv)
    return {"equal_well": float(np.sqrt(per.mean())), "pooled": float(np.sqrt(r2.mean()))}


def interval_below_zero_with_positive_beta(beta: float, lo: float, hi: float) -> bool:
    """The case the registered rule does not name (see the module docstring); report only."""
    return bool(np.isfinite(beta) and np.isfinite(hi) and beta > 0.0 and hi < 0.0)


def decide(informative: bool, beta: float, lo: float, hi: float, rmse_pr: float,
           rmse_zr: float, band: float = RATIO_BAND) -> str:
    """The registered decision rule (N2 decision_rule; N2C tests[0].decision_rule), literally."""
    if not informative:
        return "not_informative"
    if not (np.isfinite(beta) and np.isfinite(lo) and np.isfinite(hi)):
        return "not_supported"
    if lo <= 0.0 <= hi or beta <= 0.0:
        return "not_supported"
    if 1.0 / band <= beta <= band and rmse_pr < rmse_zr:
        return "response_supported"
    return "direction_supported_size_not"


# ----------------------------------------------------------------------------- panel
def _rows(well, month, obs, zr, c, flag, cl=None) -> dict:
    well = np.asarray(well).astype(str)
    month = np.asarray(month).astype(str)
    n = len(well)
    arr = {"obs": np.asarray(obs, float), "zr": np.asarray(zr, float),
           "c": np.asarray(c, float), "flag": np.asarray(flag)}
    if cl is not None:
        arr["cl"] = np.asarray(cl, float)
    if any(len(v) != n for v in arr.values()) or len(month) != n:
        raise ValueError("panel columns differ in length")
    key = np.char.add(np.char.add(well, "|"), month)
    if len(np.unique(key)) != n:
        raise ValueError("duplicate (well, month) rows")
    keep = np.isin(arr["flag"].astype(int), FLAGS_SCORED)
    for k in ("obs", "zr", "c") + (("cl",) if cl is not None else ()):
        keep &= np.isfinite(arr[k])
    t = np.array([int(m[:4]) * 12 + int(m[5:7]) - 1 for m in month], float)
    return {"well": well[keep], "month": month[keep], "t": t[keep],
            **{k: v[keep] for k, v in arr.items() if k != "flag"},
            "n_rows_in": int(n), "n_rows_scored": int(keep.sum())}


def beta_response(e, c, wells, well_blocks: dict, t=None, detrend: bool = False,
                  B: int = B_BOOT, seed: int = SEED, level: float = LEVEL) -> dict:
    uniq, sxy, sxx = well_sums(e, c, wells, t, detrend)
    den = float(sxx.sum())
    beta = float(sxy.sum() / den) if den > 0 else float("nan")
    boot = block_bootstrap(sxy, sxx, [well_blocks[w] for w in uniq], B, seed, level)
    return {"beta": beta, "ci90": [boot["lo"], boot["hi"]], "bootstrap": boot,
            "n_wells": int(len(uniq)), "detrended": bool(detrend)}


def level_component(e, c, wells) -> dict:
    """Report only: OLS with intercept of the window-mean e on the window-mean c, across
    wells (confounded with ZR's level bias)."""
    uniq, inv = _codes(wells)
    n = np.bincount(inv).astype(float)
    me = np.bincount(inv, weights=np.asarray(e, float)) / n
    mc = np.bincount(inv, weights=np.asarray(c, float)) / n
    if len(uniq) < 3 or float(np.var(mc)) == 0.0:
        return {"slope": None, "intercept": None, "n_wells": int(len(uniq))}
    slope = float(np.cov(mc, me, bias=True)[0, 1] / np.var(mc))
    return {"slope": slope, "intercept": float(me.mean() - slope * mc.mean()),
            "n_wells": int(len(uniq))}


def score(well, month, obs, zr, c, flag, well_blocks: dict, informative: dict,
          cl=None, detrended_secondary: bool = True) -> dict:
    """The primary statistics and the verdict for one county's well-month panel.

    ``informative`` is :func:`informative_gate`'s result, computed before any head was read.
    ``well_blocks`` maps every well to its 5 km block (:func:`block_ids`)."""
    r = _rows(well, month, obs, zr, c, flag, cl)
    if r["n_rows_scored"] == 0:
        raise ValueError("no scored row (flags 0-1, finite obs, ZR and c)")
    missing = sorted(set(r["well"]) - set(well_blocks))
    if missing:
        raise ValueError(f"wells without a block: {missing[:5]}")
    e = r["obs"] - r["zr"]
    br = beta_response(e, r["c"], r["well"], well_blocks)
    rm = {"PR": rmse(e - r["c"], r["well"]), "ZR": rmse(e, r["well"])}
    if cl is not None:
        rm["CL"] = rmse(r["obs"] - r["cl"], r["well"])
    v = decide(bool(informative["passes"]), br["beta"], br["ci90"][0], br["ci90"][1],
               rm["PR"]["equal_well"], rm["ZR"]["equal_well"])
    out = {"scorer": SCORER_ID, "version": VERSION, "sha256": file_sha256(),
           "verdict": v, "beta_response": br, "rmse_m": rm,
           "rmse_pr_below_zr": bool(rm["PR"]["equal_well"] < rm["ZR"]["equal_well"]),
           "skill_vs_cl": (bool(rm["PR"]["equal_well"] < rm["CL"]["equal_well"])
                           if cl is not None else None),
           "interval_below_zero_with_positive_beta": interval_below_zero_with_positive_beta(
               br["beta"], br["ci90"][0], br["ci90"][1]),
           "informative_gate": informative, "n_rows_in": r["n_rows_in"],
           "n_rows_scored": r["n_rows_scored"],
           "level_component_report_only": level_component(e, r["c"], r["well"])}
    if detrended_secondary:
        out["detrended_secondary"] = beta_response(e, r["c"], r["well"], well_blocks,
                                                   t=r["t"], detrend=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--describe", action="store_true", help="print constants and sha256")
    a = ap.parse_args(argv)
    if a.describe:
        print(json.dumps(describe(), indent=1))
        return 0
    ap.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
