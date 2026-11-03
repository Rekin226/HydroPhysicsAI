"""E02z's fresh first-attempt decidability reading (model-only, synthetic; CPU scoring): if a
model of the right kind were the truth, how often would E02z's strict per-county level rule
pass on years 2 and 3?

Machinery reused read-only, never copied or edited: FV v1 (dataprep_staging/fv_decidability/
osse_fv.py: AR(1) noise ``simulate_ar1``, the datum anchor ``_anchor``, ``well_heads``, the
null tolerance ``null_tolerance``, ``_rate``, ``reading``), FV v2 (fv_decidability_v2/
osse_fv2.py: the Monte Carlo note), FV.stat.v2 (fv_stat_v2/stat_v2.py: a fresh lead-chosen
seed with no default, refused previewed/smoke seeds, new stream ids, an attempts log beside a
write-once report), and E02c's statistic through e02z_lib.county_level.

Truth worlds, per truth T in FV's four distinct-recipe theta sets (R0, Q0, R1c, X5R1), world k:

    obs(w, t) = ZR(w, t) + D_T(w, t) + e_k(w, t)        t in 2027-11..2029-10

* ZR: E02y's registered year-2/3 rows (predicted_m of prospective_2027-11_y2 / _2028-11_y3,
  forward cf2c7154), unchanged.
* D_T: (h_T - h_ZR0) at each well's cell and layer (E02c's wells_158.csv) on the window
  minus its mean over the well's observed 2012-22 months (FV v1's obs_mask.csv, availability
  only), where h_T and h_ZR0 are NEW baseline-only forwards at FV v1's commit 5f3af18 with FV
  v1's flags at horizon 82 (2023-01..2029-10): ZR0 = the page recipe's 18 parameter sets
  (the same-code reference), each truth one member. Each hindcast (2012-01..2022-12) must
  equal FV v1's forward of the same run to 0.0 m at the 158 wells (gate E).
* e: per-well stationary AR(1) on the 24 contiguous months (FV v1's noise_ar1.csv), one path
  per world shared by both windows (same origin: the windows are not independent).
* No pumping-anomaly arm and no recorded weather: the truths run the average-weather
  (climatology) forcing, which is what E02z tests; interannual weather departures enter only
  through e (an optimism, stated). The noise is fitted on hindcast misfit that already holds
  structural error D_T adds again (a pessimism, stated). P is not a strict bound either way.

Null worlds: obs = CL + e (per window, K_NULL). Gate N per county and window: the null
false-pass rate must not exceed FV v1's ``null_tolerance`` on that county's ZR - CL.

Reading, per county and window, applied once: P = equal-weight mean over the four truths of
each truth's pass rate (K_TRUTH worlds); P <= 0.5 -> 'weak_by_design', otherwise
'not_shown_weak' (FV's one-sided rule); a failed gate N -> 'not_evaluable'. The per-truth
split, RMSE(CL) in the worlds and the CL / CLT halves are report-only. The reading only adds
the registered suffix to E02z's wording and the plan's 0.7 / 0.5 grading; it never changes
E02z's decision rule.

Gates, in order, before any truth-world rate: E (registration, frozen code, input sha256s,
forward metadata, hindcast identity, the engine/scorer self-check on N_SELFCHECK worlds),
then N, then the readings. An E failure writes nothing (engineering retry, logged). FV v1's
reference forwards are pinned by sha256 (K.FV1_FWD_SHA, review S4). The forwards run on cuda
only: if the 0.0 m identity cannot be reproduced (driver or platform change) the reading is
'not_run'; there is no CPU fallback (review Q3).

Timing (review B1): ``run`` refuses at or after window B's not_before (2029-01-15) and once any
E02z score report exists; the report carries 'created', and the scorer treats a report
created at or after that instant as 'not_run'. A registration created at or after
REG_DEADLINE is refused (review S6).

CLI (repository root, hydro env):
  decide_e02z.py smoke --out <dir outside results/>   synthetic end to end; prints 'ran ok'
  decide_e02z.py verify --seed N                     registration, code, inputs intact
  decide_e02z.py run --seed N                        once, after registration; mode 'x';
                                                     every attempt in attempts.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_PATHS = (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability_v2"),
          str(REPO / "dataprep_staging" / "fv_decidability"),
          str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO))
for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02z_const as K  # noqa: E402
import e02z_lib as L  # noqa: E402
import osse_fv as O  # noqa: E402  (FV v1, frozen)
import osse_fv2 as V2  # noqa: E402  (FV v2, frozen)
import score_e02c as S  # noqa: E402  (E02c, frozen)

VERSION = "1.0.0"
SELF = f"{K.X}/decide_e02z.py"
if (K.P_WEAK, K.K_TRUTH, K.K_NULL, K.MC_HALF_WIDTH) != (O.P_WEAK, O.K_TRUTH, O.K_NULL,
                                                        V2.MC_HALF_WIDTH):
    raise ImportError("E02z's copy of FV's design constants differs from osse_fv/osse_fv2")
if K.FV1_FWD_SHA != V2.V1_FORWARD_SHA:
    raise ImportError("E02z's pinned FV v1 forward shas differ from osse_fv2.V1_FORWARD_SHA")
EngineeringFailure = L.EngineeringFailure
READING_RULE = ("per county and window: P = equal-weight mean over the four truths of the "
                "pass rate (K_TRUTH worlds each); P <= 0.5 -> 'weak_by_design', otherwise "
                "'not_shown_weak'; gate N failed -> 'not_evaluable'; applied once; "
                "one-sided (no 'a fail is informative' reading)")


# ----------------------------------------------------------------------------- streams
def rng_for(seed: int, stream: int, truth: int, draw: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream, truth, draw]))


def noise_path(ctx: dict, seed: int, stream: int, truth: int, draw: int) -> dict:
    """One AR(1) path over the 24 contiguous months, split into the two windows."""
    e = O.simulate_ar1(ctx["sd"], ctx["phi"], 2 * K.N_WIN, rng_for(seed, stream, truth, draw))
    return {"B": e[:, :K.N_WIN], "C": e[:, K.N_WIN:]}


# ----------------------------------------------------------------------------- context
def context_from_arrays(ids, county, zr: dict, cl: dict, clt: dict, blocks: dict, sd, phi,
                        mask, cells=None, layers=None) -> dict:
    """Aligned context from (W,) and (W, 12) arrays (wells in ``ids`` order). Builds, per
    window, the rows template already joined to CL / CLT by E02c's ``with_baselines``, and the
    (well, month) index of every kept row (so a world's (W, 12) array maps onto it)."""
    ids = [str(s) for s in ids]
    W = len(ids)
    ctx = {"ids": ids, "W": W, "county": np.asarray(county), "zr": {}, "cl": {}, "clt": {},
           "frames": {}, "rows_index": {}, "blocks": dict(blocks),
           "sd": np.asarray(sd, float), "phi": np.asarray(phi, float),
           "mask": np.asarray(mask, bool), "cells": cells, "layers": layers}
    for w in K.WINDOWS:
        months = L.window_months(w)
        for name, src in (("zr", zr), ("cl", cl), ("clt", clt)):
            a = np.asarray(src[w], float)
            if a.shape != (W, K.N_WIN):
                raise EngineeringFailure(f"{name}[{w}] is {a.shape}, expected ({W}, 12)")
            ctx[name][w] = a
        if not np.isfinite(ctx["zr"][w]).all():
            raise EngineeringFailure(f"non-finite ZR in window {w}")
        rows = pd.DataFrame({"station_id": np.repeat(ids, K.N_WIN),
                             "date": np.tile(months, W),
                             "predicted_m": ctx["zr"][w].reshape(-1),
                             "observed_m": np.nan})
        rows["county"] = rows["station_id"].map(S.county_of)
        base = pd.DataFrame({"station_id": np.repeat(ids, K.N_WIN),
                             "date": np.tile(months, W),
                             "cl_m": ctx["cl"][w].reshape(-1),
                             "clt_m": ctx["clt"][w].reshape(-1)})
        f = S.with_baselines(rows, base, ctx["blocks"])
        wi = {s: i for i, s in enumerate(ids)}
        mi = {m: j for j, m in enumerate(months)}
        ctx["frames"][w] = f
        ctx["rows_index"][w] = (f["station_id"].map(wi).to_numpy(),
                                f["date"].map(mi).to_numpy())
    return ctx


def load_context() -> dict:
    """Real context (lead, after registration; called only inside :func:`run`)."""
    pred = {}
    for w, d in K.ABS_DIRS.items():
        p = L.rp(d) / "predictions.csv"
        if L.sha256(p) != K.ABS_SHA[w]["predictions.csv"]:
            raise EngineeringFailure(f"{p} is not E02y's registered predictions")
        t = pd.read_csv(p, dtype={"station_id": str})
        t["date"] = S._month(t["date"])
        pred[w] = t[t["date"].isin(L.window_months(w))]
    ids = sorted(pred["B"]["station_id"].unique())
    wells_p = L.rp(K.E02C_WELLS)
    if L.sha256(wells_p) != K.E02C_WELLS_SHA:
        raise EngineeringFailure("wells_158.csv is not E02c's registered file")
    wells = pd.read_csv(wells_p, dtype={"station_id": str}).set_index("station_id")
    if sorted(wells.index) != ids:
        raise EngineeringFailure("wells_158.csv does not hold exactly E02y's wells")
    wells = wells.loc[ids]
    base = pd.read_csv(L.rp(K.BASE_DIR) / K.BASE_FILE, dtype={"station_id": str})
    base["date"] = S._month(base["date"])
    for name, sha in K.FV_INPUT_SHA.items():
        if L.sha256(L.rp(K.FV_INPUTS) / name) != sha:
            raise EngineeringFailure(f"FV v1 input {name} differs from its registered sha256")
    noise = pd.read_csv(L.rp(K.FV_INPUTS) / "noise_ar1.csv",
                        dtype={"station_id": str}).set_index("station_id")
    if sorted(noise.index) != ids:
        raise EngineeringFailure("noise_ar1.csv does not hold exactly E02y's wells")
    noise = noise.loc[ids]
    mask = O.read_mask(L.rp(K.FV_INPUTS) / "obs_mask.csv", ids)

    def mat(df, col, w):
        return df.pivot(index="station_id", columns="date")[col].reindex(
            index=ids, columns=L.window_months(w)).to_numpy(float)

    return context_from_arrays(
        ids, np.array([S.county_of(s) for s in ids]),
        {w: mat(pred[w], "predicted_m", w) for w in K.WINDOWS},
        {w: mat(base, "cl_m", w) for w in K.WINDOWS},
        {w: mat(base, "clt_m", w) for w in K.WINDOWS},
        dict(zip(ids, wells["block5"].astype(str), strict=True)),
        noise["sd_m"].to_numpy(float), noise["phi"].to_numpy(float), mask,
        wells["cell"].to_numpy(int), wells["layer"].to_numpy(int))


# ----------------------------------------------------------------------------- forwards
def check_forward_meta(tag: str, meta: dict, heads: np.ndarray, want_labels: list[str]
                       ) -> dict:
    """Engineering checks of an E02z forward before any of its values is used: baseline only,
    2012-01..2029-10, origin 2022-12, horizon 82, the declared members, E02's options, no
    recorded-pumping arm, finite heads at the wells."""
    probs = []
    names = [str(x) for x in meta["scenario_names"]]
    if names != ["baseline"]:
        probs.append(f"scenarios {names}, expected ['baseline']")
    dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"])).to_period("M").to_timestamp()
    if len(dates) != K.T_RUN or dates[0] != pd.Timestamp(f"{K.AXIS_START}-01"):
        probs.append(f"dates are not {K.AXIS_START}..2029-10 ({len(dates)} months)")
    if int(meta["origin"]) != K.ORIGIN_IDX or int(meta["horizon"]) != K.HORIZON:
        probs.append("origin or horizon differs")
    labels = [str(x) for x in meta.get("member_labels", [])]
    if labels != list(want_labels) or int(meta["n_members"]) != len(want_labels):
        probs.append(f"members {labels[:3]}... differ from the declared {len(want_labels)}")
    opts = json.loads(str(meta["forward_options"]))
    if opts.get("recorded_pumping"):
        probs.append("a recorded-pumping arm is present")
    if float(meta["gain"]) != 1.0 or float(meta["hindcast_gain"]) != 0.0:
        probs.append("gain / hindcast gain differ from E02's recipe")
    if opts.get("update_method") != "innovation" or opts.get("column_heads") != "free":
        probs.append("forward options differ from E02's recipe")
    if heads.shape[-1] != K.T_RUN or not np.isfinite(heads).all():
        probs.append("heads at the well cells are not finite on the full axis")
    if probs:
        raise EngineeringFailure(f"forward {tag}: " + "; ".join(probs))
    return {"members": len(labels), "gate_git": json.loads(str(meta.get("gate", "{}")))
            .get("git_commit")}


def hindcast_identity(tag: str, h_new: np.ndarray, h_ref: np.ndarray) -> dict:
    """Gate E: 2012-01..2022-12 identical (<= HINDCAST_TOL_M) to FV v1's forward of the same
    run at the wells; the 2023-01..2027-10 overlap difference is report-only."""
    d = np.abs(h_new[:, :K.T_FIT] - h_ref[:, :K.T_FIT])
    mx = float(np.max(d)) if d.size else math.inf
    if not mx <= K.HINDCAST_TOL_M:
        raise EngineeringFailure(f"forward {tag}: hindcast differs from FV v1's by {mx} m")
    n = min(h_new.shape[1], h_ref.shape[1])
    ov = np.abs(h_new[:, K.T_FIT:n] - h_ref[:, K.T_FIT:n])
    return {"hindcast_max_abs_m": mx,
            "overlap_2023_max_abs_m_report_only": float(np.max(ov)) if ov.size else None}


def departure(h_t: np.ndarray, h_zr0: np.ndarray, mask: np.ndarray) -> dict:
    """D_T per window: (h_T - h_ZR0) minus its mean over the well's observed fitted months
    (FV v1's ``_anchor``, the months E02's datum offset used)."""
    diff = np.asarray(h_t, float)[:, :K.T_RUN] - np.asarray(h_zr0, float)[:, :K.T_RUN]
    a = O._anchor(diff, mask)
    return {w: diff[:, t0:t1] - a for w, (t0, t1) in K.FWD_WIN.items()}


def load_departures(ctx: dict) -> tuple[dict, dict]:
    """Real forwards: ZR0 and the four truths at the wells, checked, then D_T per window."""
    out, rec, base = {}, {}, {}
    for tag in (K.ZR0, *K.TAGS):
        ref = L.rp(K.fv1_npz(tag))
        if not ref.is_file() or L.sha256(ref) != K.FV1_FWD_SHA[tag]:
            raise EngineeringFailure(f"FV v1's reference forward {K.fv1_npz(tag)} is missing or "
                                     "not at its pinned sha256 (review S4)")
        ref_meta, href = O.well_heads(L.rp(K.fv1_npz(tag)), ctx["cells"], ctx["layers"])
        want = ([str(x) for x in ref_meta["member_labels"]] if tag == K.ZR0
                else [dict(K.TRUTHS)[tag]])
        meta, h = O.well_heads(L.rp(K.fwd_npz(tag)), ctx["cells"], ctx["layers"])
        rec[tag] = check_forward_meta(tag, meta, h, want)
        rec[tag].update(hindcast_identity(tag, h[0], href[0]))
        rec[tag]["npz_sha256"] = L.sha256(L.rp(K.fwd_npz(tag)))
        base[tag] = h[0]
    for tag in K.TAGS:
        out[tag] = departure(base[tag], base[K.ZR0], ctx["mask"])
    return out, rec


# ----------------------------------------------------------------------------- worlds
def world_level(ctx: dict, obs: np.ndarray, win: str, seed: int) -> dict:
    """E02z's per-county verdicts on one world's (W, 12) heads through ``county_level``."""
    f = ctx["frames"][win].copy()
    wi, mj = ctx["rows_index"][win]
    f["observed_m"] = np.asarray(obs, float)[wi, mj]
    return {c: L.county_level(f, c, ctx["blocks"], seed) for c in K.COUNTIES}


def _brief(v: dict) -> dict:
    out = {"pass": v["verdict"] == "level_supported", "verdict": v["verdict"]}
    if "rmse_equal_well_m" in v:
        out.update({"beat_cl": v["beat_cl"], "beat_clt": v["beat_clt"],
                    "rmse_cl": v["rmse_equal_well_m"]["CL"],
                    "rmse_zr": v["rmse_equal_well_m"]["ZR"]})
    return out


def selfcheck(ctx: dict, D: dict, seed: int) -> list[dict]:
    """Gate E: on N_SELFCHECK worlds the engine's fast path equals the scorer's path (rows
    written as E02y step-(5)-like rows, ``S.annotate``, then ``window_level``)."""
    out = []
    tag = K.TAGS[0]
    for k in range(K.N_SELFCHECK):
        e = noise_path(ctx, seed, K.STREAM_SELFCHECK, 0, k)
        for w in K.WINDOWS:
            obs = ctx["zr"][w] + D[tag][w] + e[w]
            fast = world_level(ctx, obs, w, seed)
            months = L.window_months(w)
            rows = pd.DataFrame({"station_id": np.repeat(ctx["ids"], K.N_WIN),
                                 "date": np.tile(months, ctx["W"]),
                                 "predicted_m": ctx["zr"][w].reshape(-1),
                                 "lower_m": ctx["zr"][w].reshape(-1) - 1.0,
                                 "upper_m": ctx["zr"][w].reshape(-1) + 1.0,
                                 "observed_m": np.asarray(obs).reshape(-1)})
            base = pd.DataFrame({"station_id": np.repeat(ctx["ids"], K.N_WIN),
                                 "date": np.tile(months, ctx["W"]),
                                 "cl_m": ctx["cl"][w].reshape(-1),
                                 "clt_m": ctx["clt"][w].reshape(-1)})
            slow = L.window_level(S.annotate(rows), base, ctx["blocks"], seed, w)["counties"]
            same = all(json.dumps(fast[c], sort_keys=True, default=L.jsonable)
                       == json.dumps(slow[c], sort_keys=True, default=L.jsonable)
                       for c in K.COUNTIES)
            if not same:
                raise EngineeringFailure("the engine's world path differs from the scorer's "
                                         f"path (self-check world {k}, window {w})")
            out.append({"world": k, "window": w, "identical": True})
    return out


def score_worlds(ctx: dict, D: dict, *, seed: int, k_truth: int = K.K_TRUTH,
                 k_null: int = K.K_NULL, log=print) -> dict:
    """Gate N and the readings (gate E is the caller's). Pure given ``ctx`` and ``D``."""
    seed = int(seed)          # checked by run(); the smoke passes SMOKE_SEED
    per = {w: {c: {} for c in K.COUNTIES} for w in K.WINDOWS}
    for ti, tag in enumerate(K.TAGS):
        recs = {w: {c: [] for c in K.COUNTIES} for w in K.WINDOWS}
        for k in range(k_truth):
            e = noise_path(ctx, seed, K.STREAM_NOISE, ti, k)
            for w in K.WINDOWS:
                v = world_level(ctx, ctx["zr"][w] + D[tag][w] + e[w], w, seed)
                for c in K.COUNTIES:
                    recs[w][c].append(_brief(v[c]))
        for w in K.WINDOWS:
            for c in K.COUNTIES:
                per[w][c][tag] = recs[w][c]
        log(f"truth {tag}: {k_truth} worlds scored")
    null = {w: {c: [] for c in K.COUNTIES} for w in K.WINDOWS}
    for k in range(k_null):
        e = noise_path(ctx, seed, K.STREAM_NULL, 0, k)
        for w in K.WINDOWS:
            v = world_level(ctx, ctx["cl"][w] + e[w], w, seed)
            for c in K.COUNTIES:
                null[w][c].append(_brief(v[c]))
    out = {}
    for w in K.WINDOWS:
        out[w] = {}
        for c in K.COUNTIES:
            out[w][c] = cell_reading(ctx, w, c, per[w][c], null[w][c], k_null)
    return out


def cell_reading(ctx: dict, w: str, c: str, per_truth: dict, null: list, k_null: int
                 ) -> dict:
    """One county and window: gate N, P over the truths, the reading; report-only splits."""
    sel = ctx["county"] == c
    tol = O.null_tolerance(ctx["zr"][w][sel] - ctx["cl"][w][sel], ctx["sd"][sel],
                           ctx["phi"][sel], k=k_null)
    fp = O._rate([r["pass"] for r in null])
    gate_n = fp["p"] <= tol["tolerance"]
    split = {}
    for tag, rs in per_truth.items():
        ok = [r for r in rs if "rmse_cl" in r]
        split[tag] = {"passed": O._rate([r["pass"] for r in rs]),
                      "beat_cl": O._rate([r.get("beat_cl", False) for r in rs]),
                      "beat_clt": O._rate([r.get("beat_clt", False) for r in rs]),
                      "not_evaluable": float(np.mean([r["verdict"] == "level_not_evaluable"
                                                      for r in rs])),
                      "median_rmse_cl_m": (float(np.median([r["rmse_cl"] for r in ok]))
                                           if ok else None),
                      "median_rmse_zr_m": (float(np.median([r["rmse_zr"] for r in ok]))
                                           if ok else None)}
    ps = [s["passed"] for s in split.values()]
    p = float(np.mean([x["p"] for x in ps]))
    se = math.sqrt(sum(x["mc_se"] ** 2 for x in ps)) / len(ps)
    rec = {"county": c, "window": w, "P": p, "mc_se": se,
           "mc_half_width_90": V2.Z_MC * se,
           "p_minus_threshold_in_se": (p - K.P_WEAK) / se if se > 0 else None,
           "per_truth_report_only": split,
           "null": {"false_pass": fp, "tolerance": tol, "gate_n_passed": bool(gate_n)}}
    if not gate_n:
        rec.update({"reading": "not_evaluable",
                    "why": f"null false-pass {fp['p']:.3f} > tolerance "
                           f"{tol['tolerance']:.3f}"})
    else:
        rec["reading"] = O.reading(p) or "not_evaluable"
    return rec


# ----------------------------------------------------------------------------- registration
def check_registered(doc: dict, seed: int) -> None:
    d = doc.get("decidability") or {}
    want = {"seed": seed, "k_truth": K.K_TRUTH, "k_null": K.K_NULL,
            "streams": {"noise": K.STREAM_NOISE, "null": K.STREAM_NULL,
                        "selfcheck": K.STREAM_SELFCHECK},
            "truths": [list(t) for t in K.TRUTHS], "p_weak": K.P_WEAK}
    bad = sorted(k for k, v in want.items() if d.get(k) != v)
    if bad:
        raise EngineeringFailure(f"the registered decidability design differs on {bad} (the "
                                 "seed must be the registered one)")


def load_registered(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """Registered, intact, frozen code byte-identical, inputs unchanged."""
    from harness import prereg3

    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(K.ID, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise L.Refused(f"{K.ID} is not registered or not intact: {err}") from err
    fc = doc.get("frozen_code") or {}
    why = L.deadline_problem(doc.get("created"))
    if why:
        raise L.Refused(why)
    if fc.get(SELF) is None or fc.get(K.GOV) != K.GOV_SHA:
        raise EngineeringFailure("frozen_code does not bind this engine and the GOV module")
    bad = sorted(p for p, h in fc.items() if not L.rp(p).is_file() or L.sha256(L.rp(p)) != h)
    if bad:
        raise EngineeringFailure(f"code sha256 differs from the registration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    return doc, sha


def timing_problem(now) -> str | None:
    """Review B1: the reading runs before window B's session and before any E02z score."""
    if now >= L.parse_utc(K.NOT_BEFORE["B"]):
        return (f"the decidability reading runs before {K.NOT_BEFORE['B']} only (it is "
                "'not_run' after)")
    done = [K.SCORE_REPORT[w] for w in K.WINDOWS if L.rp(K.SCORE_REPORT[w]).exists()]
    if done:
        return f"an E02z score report exists ({done}): the reading can no longer run"
    return None


def run(seed, *, out: str = K.DECIDE_REPORT, attempts: str = K.DECIDE_ATTEMPTS,
        prereg_dir=None, rows_dir=None, log=print, now=None) -> dict:
    """Once: refuses unless registered, intact, on the registered seed, before window B's
    session and before any score report; mode 'x'. Every attempt appends 'start', then
    'refused' / 'engineering_failure' / 'error' / 'success'."""
    now = now or L.now_utc()
    outp, att = L.rp(out), L.rp(attempts)
    n = L.attempt_no(att)
    L.log_attempt(att, {"event": "start", "attempt": n, "seed": seed, "report": str(out),
                        "code_sha256": L.sha256(Path(__file__)), "component": "decidability"})
    try:
        seed = L.check_seed(seed)
        why = timing_problem(now)
        if why:
            raise L.Refused(why)
        if outp.exists():
            raise L.Refused(f"{outp} exists: the decidability reading is written once")
        doc, dsha = load_registered(prereg_dir, rows_dir)
        check_registered(doc, seed)
        ctx = load_context()
        D, fwd = load_departures(ctx)
        sc = selfcheck(ctx, D, seed)
        log("gate E passed (engineering)")
        readings = score_worlds(ctx, D, seed=seed, log=log)
        why = timing_problem(L.now_utc())       # B1: re-checked before the write
        if why:
            raise L.Refused(why)
        rep = {"id": K.ID, "component": "decidability (first attempt)", "version": VERSION,
               "created": L.now_utc().isoformat(timespec="seconds"),
               "registration_sha256": dsha, "seed": seed, "k_truth": K.K_TRUTH,
               "k_null": K.K_NULL, "reading_rule": READING_RULE,
               "mc_note": V2.mc_note(len(K.TAGS), K.K_TRUTH),
               "engineering": {"forwards": fwd, "selfcheck": sc}, "readings": readings,
               "frozen_code": doc["frozen_code"], "status": "scored"}
        L.write_once(outp, rep)
    except EngineeringFailure as err:
        L.log_attempt(att, {"event": "engineering_failure", "attempt": n, "message": str(err)})
        raise
    except L.Refused as err:
        L.log_attempt(att, {"event": "refused", "attempt": n, "message": str(err)})
        raise
    except Exception as err:
        L.log_attempt(att, {"event": "error", "attempt": n,
                            "message": f"{type(err).__name__}: {err}"})
        raise
    L.log_attempt(att, {"event": "success", "attempt": n, "report_sha256": L.sha256(outp)})
    return rep


def report_in_time(report: dict | None) -> bool:
    """Review B1: a reading counts only if created before window B's not_before."""
    try:
        return L.parse_utc(str(report["created"])) < L.parse_utc(K.NOT_BEFORE["B"])
    except (KeyError, TypeError, ValueError):
        return False


def reading_for(report: dict | None, win: str, county: str) -> dict:
    """The scorer's view of the decidability report: {'reading', 'p'[, 'why']}; a report
    created at or after window B's not_before (or without 'created') is 'not_run'."""
    if not report or not report_in_time(report):
        return {"reading": "not_run"}
    r = ((report.get("readings") or {}).get(win) or {}).get(county)
    if not r:
        return {"reading": "not_run"}
    return {"reading": r.get("reading", "not_evaluable"), "p": r.get("P", float("nan")),
            "why": r.get("why", "")}


# ----------------------------------------------------------------------------- smoke
def synthetic_context(n_yunlin: int = 24, n_changhua: int = 22, seed: int = 5) -> tuple:
    """A synthetic context and departures (no file, no model output): two counties of fake
    wells on a few 5 km blocks."""
    rng = np.random.default_rng(seed)
    ids = ([f"09{i:06d}" for i in range(n_yunlin)] + [f"07{i:06d}" for i in range(n_changhua)])
    W = len(ids)
    county = np.array([S.county_of(s) for s in ids])
    blocks = {s: f"b{i % 12}" for i, s in enumerate(ids)}
    lvl = rng.normal(0, 3, W)
    season = np.sin(2 * np.pi * np.arange(K.N_WIN) / 12.0)
    zr, cl, clt = {}, {}, {}
    for j, w in enumerate(K.WINDOWS):
        cl[w] = lvl[:, None] + season[None]
        clt[w] = cl[w] - 0.1 * (j + 1)
        zr[w] = cl[w] - 0.8 - 0.2 * j
    D = {t: {w: rng.normal(0, 0.05, (W, K.N_WIN)) for w in K.WINDOWS} for t in K.TAGS}
    mask = np.ones((W, K.T_FIT), bool)
    ctx = context_from_arrays(ids, county, zr, cl, clt, blocks, np.full(W, 0.3),
                              np.full(W, 0.6), mask)
    return ctx, D


def _under_results(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to((REPO / "results").resolve())
        return True
    except ValueError:
        return False


def run_smoke(out: Path, k: int = 3) -> dict:
    """Engineering only on a synthetic context and SMOKE_SEED: self-check, worlds, gate N and
    reading plumbing at a tiny K. Writes 'ran ok' and no rate."""
    out = Path(out)
    if _under_results(out):
        raise EngineeringFailure("the smoke never writes under results/")
    out.mkdir(parents=True, exist_ok=False)
    ctx, D = synthetic_context()
    sc = selfcheck(ctx, D, K.SMOKE_SEED)
    rd = score_worlds(ctx, D, seed=K.SMOKE_SEED, k_truth=k, k_null=k, log=lambda *_: None)
    ok = (all(x["identical"] for x in sc)
          and all(rd[w][c]["reading"] in ("weak_by_design", "not_shown_weak", "not_evaluable")
                  for w in K.WINDOWS for c in K.COUNTIES))
    rep = {"id": K.ID, "smoke": True, "status": "ran ok" if ok else "plumbing failed",
           "seed": K.SMOKE_SEED, "k": k,
           "note": "SMOKE: synthetic context, engineering only; no rate is written"}
    L.write_once(out / "smoke.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("--out", required=True, help="a NEW directory outside results/")
    for name in ("verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True, help="the registered seed")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            print(rep["status"])
            return 0 if rep["status"] == "ran ok" else 1
        if a.cmd == "verify":
            doc, dsha = load_registered(prereg_dir, rows_dir)
            check_registered(doc, L.check_seed(a.seed))
            print(f"{K.ID} registered (sha256 {dsha[:16]}); code and inputs intact")
            return 0
        rep = run(a.seed, prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {K.DECIDE_REPORT} (status {rep['status']})")
        return 0
    except L.Refused as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
