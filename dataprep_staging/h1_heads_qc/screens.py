"""Sample-level head screens and jump/episode/step detection for one well segment.

All functions work on numpy arrays of sample times (int64 seconds since the epoch, local
time) and values (m), sorted, duplicate-free, sentinels included (they are flagged 8 here).

Jump detector: the audit's (``_pilots/audit_2026-10-11/artefacts.py``) logic, reproduced
exactly (consecutive samples <= 70 min apart, |dv| > 1 m, 6-hour dead time, persistence in
the 6-48 h window); its offsets are paired into episodes with the gap-fill pilot's rule
(``s5b_cleantruth.py``: opposite jump within 365 days, |j1 + j2| < 0.3 max|j|), and unpaired
offsets >= 3 m are steps (plan WP H1). Further screens (plan WP H1): 15 x MAD despike against
the pre-origin median, flat runs >= 72 h, the 15-day rolling-median dropout screen (with
floor runs), far outliers > 12 m from the rolling median, and readings below the well bottom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# sample-flag kinds (bit field, so one sample may carry several reasons)
K_SENTINEL = 1
K_SPIKE = 2
K_EPISODE = 4
K_DROPOUT = 8
K_STEP = 16
K_FLAT = 32
K_MAD = 64
K_FAR = 128
K_DROPFLAT = 256
K_BOTTOM = 512
KIND_NAMES = {K_SENTINEL: "sentinel_block", K_SPIKE: "spike", K_EPISODE: "offset_episode",
              K_DROPOUT: "dropout", K_STEP: "step", K_FLAT: "flat_run", K_MAD: "mad_outlier",
              K_FAR: "far_outlier", K_DROPFLAT: "dropout_flat", K_BOTTOM: "below_well_bottom"}


def to_arrays(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    t = s.index.values.astype("datetime64[s]").astype(np.int64)
    return t, s.to_numpy(dtype=float)


def ts(t: int) -> pd.Timestamp:
    return pd.Timestamp(int(t), unit="s")


# ----------------------------------------------------------------------------- jumps
def jump_events(t: np.ndarray, v: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """The audit's jump classifier on non-sentinel samples ``(t, v)``.

    Returns (jumps, spike_mask). Each jump: index of the sample after the jump ``i1``, its
    time, size ``j``, ``cls`` in {offset, spike, neither}; ``pre`` level and ``post``."""
    n = len(v)
    spike = np.zeros(n, bool)
    out: list[dict] = []
    if n < 2:
        return out, spike
    dv, dtt = np.diff(v), np.diff(t)
    idx = np.flatnonzero((np.abs(dv) > C.JUMP_M) & (dtt <= C.JUMP_MAX_DT_S))
    last_t = -(10 ** 12)
    for i in idx:
        if t[i + 1] - last_t < C.DEAD_TIME_S:
            continue
        j = dv[i]
        pre = v[max(0, i - 5):i + 1].mean()
        a = np.searchsorted(t, t[i + 1] + C.PERSIST_H[0] * 3600)
        b = np.searchsorted(t, t[i + 1] + C.PERSIST_H[1] * 3600)
        if b - a < C.PERSIST_MIN_SAMPLES:
            continue
        post = float(np.median(v[a:b]))
        if abs(post - pre - j) < C.PERSIST_TOL * abs(j):
            cls = "offset"
        elif abs(post - pre) < C.REVERT_TOL * abs(j):
            cls = "spike"
            spike[i + 1:a] = np.abs(v[i + 1:a] - pre) > C.SPIKE_MASK_FRAC * abs(j)
        else:
            cls = "neither"
        out.append({"i1": int(i + 1), "t": int(t[i + 1]), "j": float(j), "cls": cls,
                    "pre": float(pre), "post": post, "spike_end": int(a)})
        last_t = t[i + 1]
    return out, spike


def pair_offsets(jumps: list[dict]) -> tuple[list[tuple[int, int]], list[int]]:
    """Greedy pairing of offsets (|j| >= 1 m) into episodes (gap-fill pilot rule).

    Returns (episodes as (a, b) indices into ``jumps``, unpaired offset indices)."""
    off = [k for k, J in enumerate(jumps)
           if J["cls"] == "offset" and abs(J["j"]) >= C.EPISODE_MIN_M]
    used: set[int] = set()
    eps = []
    for ai, a in enumerate(off):
        if a in used:
            continue
        for b in off[ai + 1:]:
            if b in used:
                continue
            ja, jb = jumps[a]["j"], jumps[b]["j"]
            # whole days, as the pilot's pandas ``Timedelta.days`` (s5b_cleantruth.py)
            if (jumps[b]["t"] - jumps[a]["t"]) // 86400 > C.PAIR_MAX_DAYS:
                break
            if np.sign(ja) != np.sign(jb) and \
                    abs(ja + jb) < C.PAIR_SIZE_TOL * max(abs(ja), abs(jb)):
                eps.append((a, b))
                used |= {a, b}
                break
    unpaired = [k for k in off if k not in used]
    return eps, unpaired


# ----------------------------------------------------------------------------- runs
def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def flat_runs(t: np.ndarray, v: np.ndarray, min_h: float) -> np.ndarray:
    """Mask of samples inside runs of identical consecutive values lasting >= min_h hours."""
    n = len(v)
    mask = np.zeros(n, bool)
    if n < 2:
        return mask
    same = np.concatenate([[False], np.diff(v) == 0])
    rid = np.cumsum(~same)
    starts = np.flatnonzero(~same)
    ends = np.r_[starts[1:], n]
    dur = (t[ends - 1] - t[starts]) / 3600.0
    for s0, e0, d in zip(starts, ends, dur, strict=True):
        if d >= min_h:
            mask[s0:e0] = True
    del rid
    return mask


def run_len_identical(v: np.ndarray) -> np.ndarray:
    """Length (samples) of the identical-value run each sample belongs to."""
    n = len(v)
    if n == 0:
        return np.zeros(0, int)
    same = np.concatenate([[False], np.diff(v) == 0])
    rid = np.cumsum(~same)
    cnt = np.bincount(rid)
    return cnt[rid]


def rolling_dev(t: np.ndarray, v: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Deviation of each sample from the centred 15-day rolling median of hourly medians
    (computed on samples in ``keep``)."""
    if keep.sum() < 10:
        return np.full(len(v), np.nan)
    s = pd.Series(v[keep], index=pd.to_datetime(t[keep], unit="s"))
    hm = s.resample("h").median()
    rm = hm.rolling(C.ROLL_WINDOW_H, center=True, min_periods=72).median()
    hrs = pd.to_datetime(t, unit="s").floor("h")
    ref = rm.reindex(hrs).to_numpy()
    return v - ref


# ----------------------------------------------------------------------------- main
def screen(t: np.ndarray, v_all: np.ndarray, *, med_ref: float, mad_ref: float,
           elev: float | None, depth: float | None, flag_from: int | None = None) -> dict:
    """Screen one segment. ``v_all`` includes sentinels. ``med_ref``/``mad_ref``: robust
    level and scale for the 15 x MAD despike (fitted on pre-origin data). ``flag_from``:
    flags are only reported for samples with t >= flag_from (context before it is used for
    detection only; jumps before it are kept in the catalogue with ``context=True``).

    Returns dict with ``kind`` (uint16 bit field per sample, 0 = clean), ``jumps``,
    ``episodes`` and ``steps`` (dicts), and ``events`` (list of dicts)."""
    n = len(v_all)
    kind = np.zeros(n, np.uint16)
    sent = ~np.isfinite(v_all) | (np.abs(v_all) >= C.SENTINEL_ABS)
    kind[sent] |= K_SENTINEL
    ok = ~sent
    io = np.flatnonzero(ok)
    t_ok, v_ok = t[ok], v_all[ok]
    cadence = "10min" if (len(t_ok) > 1 and np.median(np.diff(t_ok)) <= 660) else "hourly"

    jumps, spike_mask = jump_events(t_ok, v_ok)
    kind[io[spike_mask]] |= K_SPIKE
    eps_idx, unpaired = pair_offsets(jumps)

    bottom = (elev - depth) if (elev is not None and depth is not None and np.isfinite(elev)
                                and np.isfinite(depth) and depth > 0) else None
    top = elev + C.TOP_MARGIN_M if (elev is not None and np.isfinite(elev)) else None

    ff = flag_from if flag_from is not None else -(10 ** 15)
    episodes = []
    for a, b in eps_idx:
        Ja, Jb = jumps[a], jumps[b]
        seg = slice(Ja["i1"], Jb["i1"])
        vs = v_ok[seg]
        med = float(np.median(vs)) if len(vs) else np.nan
        rsd = float(1.4826 * np.median(np.abs(vs - med))) if len(vs) else np.nan
        frac_flat = float(np.mean(np.diff(vs) == 0)) if len(vs) > 1 else 1.0
        below = med - Ja["pre"] < -C.DROPOUT_DEV_M
        is_drop = bool(below and (rsd < C.DROPOUT_EPISODE_FLAT_SD or frac_flat > 0.5
                                  or (bottom is not None and med < bottom - C.BOTTOM_MARGIN_M)))
        kbit = K_DROPOUT if is_drop else K_EPISODE
        kind[io[seg]] |= kbit
        episodes.append({"a": a, "b": b, "t1": Ja["t"], "t2": Jb["t"], "j1": Ja["j"],
                         "j2": Jb["j"], "i1": Ja["i1"], "i2": Jb["i1"], "dropout": is_drop,
                         "seg_median": med, "seg_rsd": rsd, "pre": Ja["pre"],
                         "n": int(Jb["i1"] - Ja["i1"]),
                         "t_lo": int(t_ok[Ja["i1"]]),
                         "t_hi": int(t_ok[Jb["i1"] - 1]) if Jb["i1"] > Ja["i1"] else Ja["t"],
                         "frac_flat": frac_flat, "context": bool(Jb["t"] < ff),
                         "spans_origin": bool(Ja["t"] < ff <= Jb["t"])})

    # steps: unpaired offsets >= 3 m; displaced side = impossible side, else shorter side
    step_ix = [k for k in unpaired if abs(jumps[k]["j"]) >= C.STEP_MIN_M]
    bounds = [0] + [jumps[k]["i1"] for k in step_ix] + [len(v_ok)]
    steps = []
    for q, k in enumerate(step_ix):
        J = jumps[k]
        lo, mid, hi = bounds[q], bounds[q + 1], bounds[q + 2]
        before, after = v_ok[lo:mid], v_ok[mid:hi]
        mb = float(np.median(before)) if len(before) else np.nan
        ma = float(np.median(after)) if len(after) else np.nan

        def impossible(m):
            low = bottom is not None and m < bottom - C.BOTTOM_MARGIN_M
            return bool(np.isfinite(m) and ((top is not None and m > top) or low))
        imp_b, imp_a = impossible(mb), impossible(ma)
        if imp_a and not imp_b:
            side, why = "after", "impossible_level_after"
        elif imp_b and not imp_a:
            side, why = "before", "impossible_level_before"
        else:
            dur_b = (t_ok[mid - 1] - t_ok[lo]) if mid > lo else 0
            dur_a = (t_ok[hi - 1] - t_ok[mid]) if hi > mid else 0
            side, why = ("after", "shorter_side") if dur_a <= dur_b else ("before", "shorter_side")
        seg = slice(mid, hi) if side == "after" else slice(lo, mid)
        ctx = bool(J["t"] < ff)
        if not ctx:      # context steps (before flag_from) are handled by the pre pass
            kind[io[seg]] |= K_STEP
        steps.append({"k": k, "t": J["t"], "j": J["j"], "side": side, "side_rule": why,
                      "context": ctx, "to_end": bool(side == "after" and hi == len(v_ok)),
                      "i_lo": int(seg.start), "i_hi": int(seg.stop),
                      "t_lo": int(t_ok[seg.start]) if seg.stop > seg.start else J["t"],
                      "t_hi": int(t_ok[seg.stop - 1]) if seg.stop > seg.start else J["t"],
                      "median_before": mb, "median_after": ma, "impossible_before": imp_b,
                      "impossible_after": imp_a})

    # further screens on the non-sentinel samples
    event_bits = kind & np.uint16(K_EPISODE | K_DROPOUT | K_STEP)
    pre_clean = (kind[io] & (K_SPIKE | K_EPISODE | K_DROPOUT)) == 0
    dev = rolling_dev(t_ok, v_ok, pre_clean)
    far = np.abs(dev) > C.FAR_DEV_M
    rl = run_len_identical(v_ok)
    dropflat = (dev < -C.DROPOUT_DEV_M) & (rl >= C.DROPOUT_FLAT_N[cadence])
    flat = flat_runs(t_ok, v_ok, C.FLAT_RUN_MIN_H)
    mad = (np.abs(v_ok - med_ref) > C.MAD_K * mad_ref) if (np.isfinite(med_ref) and mad_ref > 0) \
        else np.zeros(len(v_ok), bool)
    below = (v_ok < bottom - C.BOTTOM_MARGIN_M) if bottom is not None else np.zeros(len(v_ok), bool)
    kind[io[far]] |= K_FAR
    kind[io[dropflat]] |= K_DROPFLAT
    kind[io[flat]] |= K_FLAT
    kind[io[mad]] |= K_MAD
    kind[io[below]] |= K_BOTTOM

    for J in jumps:
        J["context"] = flag_from is not None and J["t"] < flag_from
    kind_base = kind & ~event_bits
    return {"kind": kind, "kind_base": kind_base, "jumps": jumps, "episodes": episodes,
            "steps": steps,
            "unpaired": unpaired, "t_ok_index": io, "cadence": cadence, "bottom": bottom,
            "top": top}


def run_events(t: np.ndarray, kind: np.ndarray, bit: int, max_gap_s: int = 2 * 3600) -> list:
    """Group samples carrying ``bit`` into events (gaps <= max_gap_s join)."""
    ix = np.flatnonzero(kind & bit)
    if ix.size == 0:
        return []
    br = np.flatnonzero(np.diff(t[ix]) > max_gap_s)
    starts = np.r_[0, br + 1]
    ends = np.r_[br, ix.size - 1]
    return [(int(t[ix[a]]), int(t[ix[b]]), int(b - a + 1))
            for a, b in zip(starts, ends, strict=True)]
