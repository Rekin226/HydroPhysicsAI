"""H1a v2 decision layer (round 2): level-persistent jumps, pairing, steps and the flag rule.

Why a v2. The v1 decisions took the audit's jump classes at face value. A jump of > 1 m
within 70 minutes whose 6-48 h median kept 70 % of the jump was an "offset"; offsets of
opposite sign and similar size within 365 days were paired into datum episodes, and the
unpaired ones of >= 3 m became permanent steps. Two failures followed (v1 report and its
after-the-fact diagnosis, pre-origin data):

* pumping-cycle wells. In wells next to a pump (07080132, 09050111, 08040222, 07080122,
  10050211, 07240223, ...) the level drops and recovers by 1-4 m within an hour whenever the
  pump switches, often several times a week. Many of these switches persist 6-48 h, so the
  audit calls them offsets, and the greedy pairing joined unrelated pump-on and pump-off
  jumps up to a year apart into "episodes" (132 of the 228 v1 episodes at 07080132 alone,
  median 120 days long), which v1 then corrected at flag 1 on co-located backing;
* spurious steps. A jump of 3 m or more whose return was missed (09190132: +18.9 m with
  10-day medians -21.74 / -21.89 m) became a step carried to the record end.

v2 keeps the stage-1 jump detector unchanged (the audit's rule: |dv| > 1 m between samples
<= 70 min apart, 6 h dead time; the jump catalogue, the audit agreement and the spike screen
are those of v1) and replaces the decisions:

1. **Level persistence.** Every stage-1 jump of >= 1 m that is not a spike gets the robust
   local level shift ``delta`` of ``levelcheck.level_shift``: a least-absolute-deviation fit
   of hourly medians within 10 days on either side, with a common linear trend. The jump is
   *persistent* when ``sign(delta) = sign(j)`` and ``|delta| >= 0.5 |j|``. With fewer than
   12 hourly medians on a side (a record gap next to the jump) the check is undetermined and
   the stage-1 class decides (an audit "offset", which kept its level for 6-48 h, counts as
   persistent). The window is not truncated at other jumps. A datum offset that lasts longer than the window
   is persistent at both edges whatever the stage-1 class, and a pump switch that lasts hours
   or a few days is not: the median of the 10-day window stays at the pumping regime's
   typical level.
2. **Typhoon-time jumps are real-event candidates, not datum edges** (plan WP H1: "Real
   events are kept: ... typhoon recharge"; "a typhoon's pressure drop can be mistaken for a
   datum-offset episode"). A jump within +-24 h of a surface-pressure fall of >= 20 hPa at
   the well's ERA5 cell (the largest fall from a running maximum inside the window, as the
   acceptance check measures it) is catalogued and never used as an episode or step edge.
   This rule implements the plan's pressure-drop criterion as a decision rule; it was added
   after the v1 criterion failed (3 v1 episodes on typhoon falls, at pumping-cycle wells), so
   in v2 that criterion holds by construction and is no longer independent evidence.
3. **Episodes.** Persistent jumps are paired: every pair of opposite sign, at most 365 days
   apart, with ``|j1 + j2| < 0.3 max|j|`` (the v1 conditions) is a candidate, and candidates
   are accepted in order of increasing relative size mismatch (then shorter duration), each
   jump in one pair at most (v1 took the first match in time). An episode whose interior is flat and
   more than 3 m low (or below the well bottom) is a dropout. Episodes and dropouts are
   flagged 9 over their interior and **never corrected** (plan WP H1: "Each is flagged 9";
   v1 had extended the co-located backing to episodes, a deviation now withdrawn).
4. **Steps.** Unpaired persistent jumps of >= 3 m are step candidates. Each must also show its
   shift over the segments between its neighbouring accepted candidates, up to a year on
   each side (segment check: the median of the daily medians after minus before, episode
   interiors left out, has the jump's sign and at least half its size; a full year on each
   side spans one seasonal cycle; with fewer than 5 daily medians on a side the local check
   stands);
   failing candidates are rejected one at a time, the most inconsistent first, and the
   segments recomputed. An accepted step gets its
   displaced side (impossible level first, else the shorter side, as v1) and the flag rule
   below; two adjacent accepted steps that both displace the segment between them form one
   excursion with unequal edges and become an episode (flagged 9, never corrected). A
   rejected candidate flags the excursion next to the jump (the run of hourly
   medians closer to the jump's displaced level than to its segment median, at most a year)
   or, without such a run, the 48 h after the jump; never corrected, never carried.
5. **Flag rule for accepted steps** (plan 2.4, WP H1): flag 1 with physical metadata (the
   displaced level is impossible, or a station metadata date lies within 7 days), or with a
   co-located nested well that records through the jump without a simultaneous jump, unless
   the leave-out neighbour regression refutes the step (the neighbours move with the well);
   flag 3 when only the neighbour regression corroborates it; otherwise flag 9.

Unpaired persistent jumps of 1-3 m, non-persistent jumps and typhoon-time jumps are
catalogued, not flagged.

Periods. Pre-origin events are decided on pre-origin samples only. The post-origin pass sees
the 2022 context of the pre-origin record plus the post-origin record; it decides only
events that reach 2023 (episodes with ``t2 >= 2023-01-01``, steps at or after it), never
re-pairs a jump the pre-origin pass already paired, and takes the accepted pre-origin steps
inside the context as fixed segment bounds (U3/U7: no 2023+ sample decides a pre-origin row).

Design record. The level-check constants (``levelcheck``: 10-day windows, >= 12 hourly
medians, half the jump) were fixed before any v2 score and were not changed afterwards. The
round-2 design was then evaluated on pre-origin data and on the family O injections r = 0..9
(already scored by v1, so design-time units): the first version (stage-1 pairs re-checked)
left the paired recall at the stage-1 detection rate; the second (persistent jumps, greedy
pairing, whole-segment medians) missed injected events at record gaps (undetermined
persistence), let natural jumps capture episode edges (greedy pairing), and rejected injected
steps on multi-year trends (whole-segment medians). The third (d3) added the undetermined
fallback and the best-match pairing (paired recall 0.934 on r = 0..9) and a trend-adjusted
one-year segment check, which the seasonal swing of high-amplitude wells defeated (step
recall 0.81); the fourth (d4) replaced it by the one-year median difference. The thresholds
were not changed in any round. The registered test uses the fresh realizations
r = 10..19 (pre-registration H1a.qc_acceptance.v2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import levelcheck as LC

SPLIT_S = int(pd.Timestamp(C.ORIGIN_SPLIT).value // 10**9)
CONTEXT_S = int(pd.Timestamp("2022-01-01").value // 10**9)
DAY = 86400
TYPHOON_FALL_HPA = C.PRESSURE_DROP_HPA      # 20 hPa (plan WP H1 acceptance)
TYPHOON_WINDOW_H = C.PRESSURE_DROP_WINDOW_H  # +-24 h around the jump
EPISODE_KINDS = ("offset_episode", "dropout")
RUN_KINDS = ("excursion", "jump_edge")


# ----------------------------------------------------------------------------- arrays
def segment_arrays(res: dict, period: str) -> dict | None:
    """Arrays used to decide the events of one period: the pre-origin record for 'pre'; the
    2022 context of the pre-origin record plus the post-origin record for 'post'."""
    if period == "pre":
        if "pre" not in res or not len(res["pre"]["t"]):
            return None
        parts = [res["pre"]]
        jumps = list(res["pre_cat"]["jumps"])
    else:
        if "post" not in res or not len(res["post"]["t"]):
            return None
        parts = []
        jumps = []
        if "pre" in res and len(res["pre"]["t"]):
            p = res["pre"]
            m = p["t"] >= CONTEXT_S
            parts.append({k: (p[k][m] if isinstance(p[k], np.ndarray) else p[k]) for k in p})
            jumps += [J for J in res["pre_cat"]["jumps"] if J["t"] >= CONTEXT_S]
        parts.append(res["post"])
        jumps += list(res["post_cat"]["jumps"])
    t = np.concatenate([p["t"] for p in parts])
    v_raw = np.concatenate([p["v_raw"] for p in parts])
    L = np.concatenate([p["L"] for p in parts])
    kb = np.concatenate([p["kind_base"] for p in parts])
    o = np.argsort(t, kind="stable")
    t, v_raw, L, kb = t[o], v_raw[o], L[o], kb[o]
    keep = np.r_[True, np.diff(t) > 0]
    t, v_raw, L, kb = t[keep], v_raw[keep], L[keep], kb[keep]
    seg = {"t": t, "v_raw": v_raw, "L": L, "kind_base": kb}
    tt, v, valid = LC.part_arrays(seg)
    nons = np.isfinite(v_raw) & (np.abs(v_raw) < C.SENTINEL_ABS)
    jumps = sorted(jumps, key=lambda J: J["t"])
    return {"t": tt, "v": v, "valid": valid, "nons": nons, "jumps": jumps,
            "t_min": int(tt[0]), "t_max": int(tt[-1]) + 1, "seg": seg}


def _median_nons(A, a, b, mask=None):
    i0, i1 = np.searchsorted(A["t"], a, "left"), np.searchsorted(A["t"], b, "left")
    ok = A["nons"][i0:i1] if mask is None else (A["nons"][i0:i1] & mask[i0:i1])
    vv = A["v"][i0:i1][ok]
    return float(np.median(vv)) if len(vv) else np.nan


def _times_nons(A, a, b):
    i0, i1 = np.searchsorted(A["t"], a, "left"), np.searchsorted(A["t"], b, "left")
    tt = A["t"][i0:i1][A["nons"][i0:i1]]
    return (int(tt[0]), int(tt[-1])) if len(tt) else (None, None)


# ----------------------------------------------------------------------------- pressure
def pressure_arrays(p: pd.Series | None, t_end_s: int | None = None):
    """(times in s, values in hPa) of a cell's hourly pressure, finite values only, up to
    ``t_end_s`` (exclusive) if given."""
    if p is None:
        return None
    p = p.dropna()
    ts_ = (p.index.to_numpy().astype("datetime64[s]").astype(np.int64))
    vs = p.to_numpy(float)
    if t_end_s is not None:
        k = ts_ < t_end_s
        ts_, vs = ts_[k], vs[k]
    return ts_, vs


def max_fall(pa, t: int, half_h: int = TYPHOON_WINDOW_H) -> float:
    """Largest pressure fall from a running maximum within +-half_h hours of ``t`` (the
    acceptance check's statistic, ``s4_validate.max_fall``)."""
    if pa is None:
        return np.nan
    ts_, vs = pa
    i0 = np.searchsorted(ts_, t - half_h * 3600, "left")
    i1 = np.searchsorted(ts_, t + half_h * 3600, "right")
    w = vs[i0:i1]
    if len(w) < 2:
        return np.nan
    return float(np.max(np.maximum.accumulate(w) - w))


# ----------------------------------------------------------------------------- persistence
def candidate_jumps(A: dict, pa=None) -> list[dict]:
    """Stage-1 jumps of >= 1 m that are not spikes, each with its untruncated 10-day level
    shift, the ``persistent`` verdict and the pressure fall around it (``pa``: the well's
    cell pressure arrays; None = no pressure, no typhoon exclusion)."""
    out = []
    for J in A["jumps"]:
        if J["cls"] == "spike" or abs(J["j"]) < C.EPISODE_MIN_M:
            continue
        r = LC.level_shift(A["t"], A["v"], A["valid"], int(J["t"]), float(J["j"]))
        fall = max_fall(pa, int(J["t"]))
        out.append({"t": int(J["t"]), "j": float(J["j"]), "cls": J["cls"],
                    "pressure_fall_hpa": fall,
                    "typhoon": bool(np.isfinite(fall) and fall >= TYPHOON_FALL_HPA),
                    "pre": J.get("pre"), "post": J.get("post"),
                    "lc_delta": r["delta"], "lc_ratio": r["ratio"], "lc_status": r["status"],
                    "lc_n_before": r["n_hours_before"], "lc_n_after": r["n_hours_after"],
                    "lc_slope_m_per_d": r["slope_m_per_d"],
                    "persistent": r["status"] == "consistent" or (
                        r["status"] == "undetermined" and J["cls"] == "offset")})
    return out


def pair_jumps(P: list[dict]) -> tuple[list[tuple[int, int]], list[int]]:
    """Pair the persistent jumps ``P`` (sorted by time) into episodes: every pair of opposite
    sign, at most 365 days apart, with ``|j1 + j2| < 0.3 max|j|`` (the v1 rule) is a
    candidate; candidates are accepted in order of increasing relative size mismatch
    ``|j1 + j2| / max|j|`` (then shorter duration), each jump in at most one pair. (Round-2
    design: v1's greedy first-match pairing let a natural jump capture the edge of a datum
    episode whose own return matched far better.)"""
    cands = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            if (P[b]["t"] - P[a]["t"]) // DAY > C.PAIR_MAX_DAYS:
                break
            ja, jb = P[a]["j"], P[b]["j"]
            mx = max(abs(ja), abs(jb))
            if np.sign(ja) != np.sign(jb) and abs(ja + jb) < C.PAIR_SIZE_TOL * mx:
                cands.append((abs(ja + jb) / mx, P[b]["t"] - P[a]["t"], a, b))
    used: set[int] = set()
    eps = []
    for _, _, a, b in sorted(cands):
        if a in used or b in used:
            continue
        eps.append((a, b))
        used |= {a, b}
    eps.sort()
    return eps, [k for k in range(len(P)) if k not in used]


def episode_record(A: dict, Ja: dict, Jb: dict, bottom) -> dict:
    """Interior statistics and the dropout test of the episode ``Ja`` -> ``Jb`` (v1 rule of
    ``screens.screen``: interior median > 3 m below the pre-jump level and flat, or below the
    well bottom)."""
    i0 = np.searchsorted(A["t"], Ja["t"], "left")
    i1 = np.searchsorted(A["t"], Jb["t"], "left")
    ok = A["nons"][i0:i1]
    vs = A["v"][i0:i1][ok]
    ts_ = A["t"][i0:i1][ok]
    med = float(np.median(vs)) if len(vs) else np.nan
    rsd = float(1.4826 * np.median(np.abs(vs - med))) if len(vs) else np.nan
    frac_flat = float(np.mean(np.diff(vs) == 0)) if len(vs) > 1 else 1.0
    pre = Ja.get("pre")
    pre = float(pre) if pre is not None and np.isfinite(pre) else np.nan
    below = bool(np.isfinite(med) and np.isfinite(pre) and med - pre < -C.DROPOUT_DEV_M)
    is_drop = bool(below and (rsd < C.DROPOUT_EPISODE_FLAT_SD or frac_flat > 0.5
                              or (bottom is not None and med < bottom - C.BOTTOM_MARGIN_M)))
    return {"t1": int(Ja["t"]), "t2": int(Jb["t"]), "j1": float(Ja["j"]), "j2": float(Jb["j"]),
            "seg_median": med, "seg_rsd": rsd, "frac_flat": frac_flat, "pre_level": pre,
            "duration_d": (Jb["t"] - Ja["t"]) / DAY,
            "t_lo": int(ts_[0]) if len(ts_) else int(Ja["t"]),
            "t_hi": int(ts_[-1]) if len(ts_) else int(Ja["t"]),
            "dropout": is_drop,
            "lc_delta": Ja["lc_delta"], "lc_ratio": Ja["lc_ratio"], "lc_status": Ja["lc_status"],
            "lc_delta2": Jb["lc_delta"], "lc_ratio2": Jb["lc_ratio"],
            "lc_status2": Jb["lc_status"], "cls1": Ja["cls"], "cls2": Jb["cls"]}


# ----------------------------------------------------------------------------- steps
def edge_runs(A: dict, t0: int, j: float, pre_level, post_level, ref_before, ref_after,
              lo: int, hi: int) -> tuple[str, list[tuple]]:
    """Flag runs of a rejected step candidate: forward from t0 while the series sits closer
    to the audit's post-jump level than to ``ref_after`` (up to ``hi``), backward while it
    sits closer to the pre-jump level than to ``ref_before`` (down to ``lo``); each only when
    that level is more than LEVEL_FRAC |j| from its reference. Without a run: the 48 h
    persistence window after the jump (kind ``jump_edge``)."""
    runs = []
    if np.isfinite(ref_after) and np.isfinite(post_level) and \
            abs(post_level - ref_after) > LC.LEVEL_FRAC * abs(j):
        rr = LC.excursion_run(A["t"], A["v"], A["valid"], t0, post_level, ref_after, +1, hi)
        if rr:
            runs.append(("after",) + rr)
    if np.isfinite(ref_before) and np.isfinite(pre_level) and \
            abs(pre_level - ref_before) > LC.LEVEL_FRAC * abs(j):
        rr = LC.excursion_run(A["t"], A["v"], A["valid"], t0, pre_level, ref_before, -1, lo)
        if rr:
            runs.append(("before",) + rr)
    if runs:
        return "excursion", runs
    return "jump_edge", [("after", int(t0), int(t0 + LC.EDGE_WINDOW_H * 3600))]


def classify_steps(A: dict, cands: list[dict], *, bottom, top, seg_valid=None,
                   fixed_bounds=()) -> list[dict]:
    """Decide the step candidates (persistent unpaired jumps >= 3 m) of one period with the
    iterative segment check. ``seg_valid``: sample mask for the segment medians (episode
    interiors removed); ``fixed_bounds``: times that bound segments but are decided elsewhere
    (accepted pre-origin steps inside the 2022 context of the post-origin pass)."""
    t_start, t_end = A["t_min"], A["t_max"]
    sv = A["valid"] if seg_valid is None else (A["valid"] & seg_valid)
    recs = {int(c["t"]): {"cand": c, "B": None, "status": None} for c in cands}
    acc = sorted(recs)

    def bounds_of(tc, accepted):
        bs = sorted(set(accepted) | set(fixed_bounds))
        lo = max([b for b in bs if b < tc], default=t_start)
        hi = min([b for b in bs if b > tc], default=t_end)
        return lo, hi

    while True:
        worst, worst_key = None, None
        for tc in acc:
            lo, hi = bounds_of(tc, acc)
            b = LC.segment_shift_year(A["t"], A["v"], sv, lo, tc, hi, recs[tc]["cand"]["j"])
            recs[tc]["B"] = b
            if b["seg_status"] == "inconsistent":
                key = float(b["seg_ratio"])
                if worst is None or key < worst_key:
                    worst, worst_key = tc, key
        if worst is None:
            break
        acc.remove(worst)
        recs[worst]["status"] = "rejected_segment"
    out = []
    for tc in sorted(recs):
        r = recs[tc]
        c = r["cand"]
        j = float(c["j"])
        rec = {"t": int(tc), "j": j, "pre_level": c.get("pre"), "post_level": c.get("post"),
               "lc_delta": c["lc_delta"], "lc_ratio": c["lc_ratio"],
               "lc_status": c["lc_status"], "lc_n_before": c["lc_n_before"],
               "lc_n_after": c["lc_n_after"], "cls1": c["cls"]}
        lo, hi = bounds_of(tc, acc)
        b = r["B"] if tc in acc else LC.segment_shift_year(A["t"], A["v"], sv, lo, tc, hi, j)
        rec.update({k: b[k] for k in ("seg_median_before", "seg_median_after", "seg_delta",
                                      "seg_ratio", "seg_status", "seg_lo", "seg_hi")})
        if tc in acc:
            mb = _median_nons(A, lo, tc, seg_valid)
            ma = _median_nons(A, tc, hi, seg_valid)

            def impossible(m):
                low = bottom is not None and m < bottom - C.BOTTOM_MARGIN_M
                return bool(np.isfinite(m) and ((top is not None and m > top) or low))
            imp_b, imp_a = impossible(mb), impossible(ma)
            if imp_a and not imp_b:
                side, why = "after", "impossible_level_after"
            elif imp_b and not imp_a:
                side, why = "before", "impossible_level_before"
            else:
                b0, b1 = _times_nons(A, lo, tc)
                a0, a1 = _times_nons(A, tc, hi)
                dur_b = (b1 - b0) if b0 is not None else 0
                dur_a = (a1 - a0) if a0 is not None else 0
                side, why = ("after", "shorter_side") if dur_a <= dur_b else \
                    ("before", "shorter_side")
            t_lo, t_hi = _times_nons(A, tc, hi) if side == "after" else _times_nons(A, lo, tc)
            rec.update({"status": "accepted", "side": side, "side_rule": why,
                        "t_lo": t_lo if t_lo is not None else int(tc),
                        "t_hi": t_hi if t_hi is not None else int(tc),
                        "median_before": mb, "median_after": ma, "impossible_before": imp_b,
                        "impossible_after": imp_a,
                        "to_end": bool(side == "after" and hi >= t_end), "runs": []})
        else:
            kind, runs = edge_runs(A, int(tc), j, rec["pre_level"], rec["post_level"],
                                   rec["seg_median_before"], rec["seg_median_after"], lo, hi)
            rec.update({"status": r["status"], "edge_kind": kind, "runs": runs,
                        "side": None, "side_rule": None, "to_end": False})
        out.append(rec)
    # two adjacent accepted steps that both displace the segment between them are one
    # excursion with unequal edges: an episode (flagged, never corrected), not two steps
    acc_recs = [r for r in out if r["status"] == "accepted"]
    for r1, r2 in zip(acc_recs[:-1], acc_recs[1:], strict=True):
        if r1["side"] == "after" and r2["side"] == "before":
            for r in (r1, r2):
                r["status"] = "unequal_edges"
            r1["partner_t"], r2["partner_t"] = r2["t"], r1["t"]
            r1["partner_j"], r2["partner_j"] = r2["j"], r1["j"]
    return out


# ----------------------------------------------------------------------------- flag rule
def flag_rule_v2(meta_impossible: bool, meta_dates: str, coloc_backed: bool,
                 coloc_covering: str, neighbour_verdict) -> tuple[int, str]:
    """Flag of an accepted step (plan 2.4 / WP H1)."""
    if meta_impossible or bool(meta_dates):
        return 1, ("metadata: impossible displaced level" if meta_impossible else
                   "metadata: station metadata date within 7 days")
    if coloc_backed and neighbour_verdict != "refuted":
        return 1, ("co-located: nested well(s) " + str(coloc_covering) + " record through "
                   "the jump without a simultaneous jump; neighbours do not refute")
    if neighbour_verdict == "corroborated":
        return 3, "neighbour regression residual > 4 calibrated sd"
    if coloc_backed:
        return 9, "none (co-located backing refuted by the neighbour regression)"
    return 9, "none"


# ----------------------------------------------------------------------------- one well
def decide_well(res: dict, *, bottom, top, pressure: pd.Series | None = None) -> list[dict]:
    """v2 decision rows of one well from its stage-1 result. Accepted steps carry
    ``flag = None`` (the stage-2 runner adds the backing evidence and the flag) unless their
    displaced level is impossible (flag 1, metadata). Rows of kind ``jump`` catalogue every
    candidate jump with its persistence statistics (not flagged). ``pressure``: hourly
    surface pressure of the well's ERA5 cell (the pre-origin pass reads it only up to
    2023-01-02, as stage 1 does)."""
    rows = []
    paired_pre: set[int] = set()
    accepted_pre_ctx: list[int] = []
    for period in ("pre", "post"):
        if not res.get(f"{period}_cat"):
            continue
        A = segment_arrays(res, period)
        if A is None:
            continue
        pa = pressure_arrays(pressure, SPLIT_S + DAY if period == "pre" else None)
        cand = candidate_jumps(A, pa)
        for c in cand:
            if period == "post" and c["t"] < SPLIT_S:
                continue        # catalogued by the pre-origin pass
            rows.append({"period": period, "kind": "jump", "t1": c["t"], "t2": c["t"],
                         "j1": c["j"], "cls1": c["cls"], "flag": np.nan,
                         "v2_status": ("typhoon_time" if c["typhoon"] else
                                       "persistent" if c["persistent"] else
                                       "undetermined" if c["lc_status"] == "undetermined"
                                       else "not_persistent"),
                         **{k: c[k] for k in ("lc_delta", "lc_ratio", "lc_status",
                                              "lc_n_before", "lc_n_after",
                                              "lc_slope_m_per_d", "pressure_fall_hpa",
                                              "persistent")}})
        P = [c for c in cand if c["persistent"] and not c["typhoon"]
             and not (period == "post" and c["t"] in paired_pre)]
        eps, unp = pair_jumps(P)
        seg_valid = np.ones(len(A["t"]), bool)
        for a, b in eps:
            E = episode_record(A, P[a], P[b], bottom)
            i0 = np.searchsorted(A["t"], E["t1"], "left")
            i1 = np.searchsorted(A["t"], E["t2"], "left")
            seg_valid[i0:i1] = False
            if period == "pre":
                paired_pre |= {E["t1"], E["t2"]}
            elif E["t2"] < SPLIT_S:
                continue        # an episode of the 2022 context: decided by the pre pass
            kind = "dropout" if E["dropout"] else "offset_episode"
            rows.append({"period": period, "kind": kind, "v2_status": kind, "flag": 9,
                         "backing": "none (dropout: invalid reading)" if E["dropout"] else
                         "none (offset episode: flagged, never corrected)",
                         "spans_origin": bool(E["t1"] < SPLIT_S <= E["t2"]), **E})
        # steps
        sc = [P[k] for k in unp if abs(P[k]["j"]) >= C.STEP_MIN_M
              and not (period == "post" and P[k]["t"] < SPLIT_S)]
        for k in unp:
            J = P[k]
            if abs(J["j"]) < C.STEP_MIN_M and not (period == "post" and J["t"] < SPLIT_S):
                rows.append({"period": period, "kind": "unpaired_offset_lt3m",
                             "v2_status": "info", "t1": J["t"], "t2": J["t"], "j1": J["j"],
                             "cls1": J["cls"], "flag": np.nan, "lc_delta": J["lc_delta"],
                             "lc_ratio": J["lc_ratio"], "lc_status": J["lc_status"]})
        fixed = accepted_pre_ctx if period == "post" else []
        dec_steps = classify_steps(A, sc, bottom=bottom, top=top, seg_valid=seg_valid,
                                   fixed_bounds=fixed)
        for s in dec_steps:
            common = {"period": period, "t1": int(s["t"]), "t2": int(s["t"]), "j1": s["j"],
                      "j2": np.nan, "spans_origin": False, "pre_level": s["pre_level"],
                      "post_level": s["post_level"], "cls1": s["cls1"]} | {
                k: s[k] for k in ("lc_delta", "lc_ratio", "lc_status", "lc_n_before",
                                  "lc_n_after", "seg_median_before", "seg_median_after",
                                  "seg_delta", "seg_ratio", "seg_status", "seg_lo", "seg_hi")}
            if s["status"] == "unequal_edges":
                if s["t"] < s["partner_t"]:
                    E = {"t": s["t"], "j": s["j"], "pre": s["pre_level"], "lc_delta": s["lc_delta"],
                         "lc_ratio": s["lc_ratio"], "lc_status": s["lc_status"], "cls": s["cls1"]}
                    Jb = next(c for c in sc if int(c["t"]) == int(s["partner_t"]))
                    rec = episode_record(A, E, Jb, bottom)
                    rows.append({"period": period, "kind": "offset_episode",
                                 "v2_status": "unequal_edges", "flag": 9,
                                 "backing": "none (two adjacent accepted steps whose "
                                            "displaced sides are the segment between them; "
                                            "flagged, never corrected)",
                                 "spans_origin": bool(rec["t1"] < SPLIT_S <= rec["t2"]),
                                 **rec})
                continue
            if s["status"] == "accepted":
                imp = s["side_rule"].startswith("impossible")
                r = common | {"kind": "step", "v2_status": "accepted", "side": s["side"],
                              "side_rule": s["side_rule"], "t_lo": s["t_lo"],
                              "t_hi": s["t_hi"], "median_before": s["median_before"],
                              "median_after": s["median_after"], "to_end": s["to_end"],
                              "meta_impossible": imp,
                              "j_displaced_side": s["j"] if s["side"] == "after" else -s["j"],
                              "flag": 1 if imp else None,
                              "backing": "metadata: impossible displaced level" if imp
                              else None}
                rows.append(r)
                if period == "pre" and s["t"] >= CONTEXT_S:
                    accepted_pre_ctx.append(int(s["t"]))
            else:
                for side, a, b in s["runs"]:
                    rows.append(common | {
                        "kind": s["edge_kind"], "v2_status": s["status"], "side": side,
                        "side_rule": None, "t_lo": int(a), "t_hi": int(b), "to_end": False,
                        "flag": 9, "backing": "none (v2: unpaired persistent jump >= 3 m "
                                              "whose segment shows no level change; flagged "
                                              "run only)"})
    for r in rows:
        r["sid"] = res["sid"]
    return rows
