"""H1a v2 stage 3 (round 2): apply the v2 decisions (``decide_v2``).

Differences from ``assemble.py`` (v1, unchanged and still used for the daily, monthly and
departure functions):

* kinds: ``offset_episode`` and ``dropout`` are flagged 9 over their interior and never
  corrected; ``step`` (accepted steps only) is corrected at flag 1/3 or flagged 9 over its
  displaced segment; ``excursion`` and ``jump_edge`` (runs of a rejected step candidate) are
  flagged 9 (new sample bit ``K_EXCURSION``); ``jump`` and ``unpaired_offset_lt3m`` rows are
  a catalogue and change no sample;
* carry across the origin (v1 rule on accepted steps only): an accepted pre-origin step
  whose displaced side is "after" and reaches the end of the pre-origin record keeps its
  decision on the post-origin rows until the first of: the return of a post-origin episode
  that starts at the carried jump (the post-origin pass paired it), a post-origin persistent
  jump of opposite sign and matching size, or an accepted post-origin step;
* ``ART_BITS_V2`` = v1 artefact bits plus ``K_EXCURSION``; ``kinds_to_names_v2`` names it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import assemble as A1
from . import config as C
from . import screens as S

SPLIT_S = A1.SPLIT_S
K_EXCURSION = 2048
ART_BITS_V2 = A1.ART_BITS | K_EXCURSION
DAY_KIND_NAMES_V2 = dict(A1.DAY_KIND_NAMES) | {K_EXCURSION: "excursion"}
EPISODE_KINDS = ("offset_episode", "dropout")
RUN_KINDS = ("excursion", "jump_edge")


def kinds_to_names_v2(bits: int) -> str:
    return ";".join(n for b, n in DAY_KIND_NAMES_V2.items() if int(bits) & b)


def carry_from_pre_v2(dec_pre: pd.DataFrame, gap_days: float | None = None) -> dict | None:
    """The accepted pre-origin step to carry into the post-origin rows, if any."""
    if dec_pre is None or dec_pre.empty:
        return None
    if gap_days is not None and gap_days > C.CARRY_MAX_GAP_DAYS:
        return None
    st = dec_pre[dec_pre["kind"] == "step"].sort_values("t1")
    if st.empty:
        return None
    last = st.iloc[-1]
    if last["side"] != "after" or not bool(last.get("to_end", False)):
        return None
    return {"from": SPLIT_S, "until": None, "flag": int(last["flag"]), "corr": float(last["j1"]),
            "j": float(last["j1"]), "t_step": int(last["t1"]), "backing": last.get("backing")}


def carry_until_v2(carry: dict | None, dec_post: pd.DataFrame | None) -> dict | None:
    """End of a carry (see the module docstring)."""
    if carry is None:
        return None
    jc = carry["j"]
    cands = []
    if dec_post is not None and len(dec_post):
        for _, E in dec_post[dec_post.kind.isin(EPISODE_KINDS)].iterrows():
            if abs(int(E["t1"]) - carry["t_step"]) <= C.COLOC_WINDOW_S:
                cands.append((int(E["t2"]), "return_of_post_pass_episode"))
        J = dec_post[(dec_post.kind == "jump") & (dec_post.v2_status == "persistent")]
        for _, x in J.iterrows():
            if int(x["t1"]) < SPLIT_S:
                continue
            j = float(x["j1"])
            if np.sign(j) != np.sign(jc) and abs(j + jc) < C.PAIR_SIZE_TOL * max(abs(j), abs(jc)):
                cands.append((int(x["t1"]), "reversal"))
        for _, St in dec_post[dec_post.kind == "step"].iterrows():
            if int(St["t1"]) >= SPLIT_S:
                cands.append((int(St["t1"]), "accepted_post_step"))
    if cands:
        t, why = min(cands)
        return dict(carry, until=int(t), until_reason=why)
    return dict(carry, until_reason="record_end")


def apply_part_v2(part: dict, dec: pd.DataFrame, carry: dict | None = None):
    """Return (value, corr, sflag, kind, applied) for one segment (v2 decision kinds)."""
    t, v_raw, L = part["t"], part["v_raw"], part["L"]
    kind = part["kind_base"].copy()
    corr = np.zeros(len(t))
    cls = np.zeros(len(t), np.uint8)
    applied = []
    c_lo = carry["from"] if carry else None
    c_hi = (carry["until"] if carry.get("until") is not None else 2**62) if carry else None
    for _, d in (dec.iterrows() if dec is not None else []):
        k = d["kind"]
        if k in EPISODE_KINDS:
            if carry and bool(d.get("spans_origin", False)) and \
                    abs(int(d["t1"]) - carry["t_step"]) <= C.COLOC_WINDOW_S:
                applied.append({"t1": int(d["t1"]), "kind": k, "what": "skipped_same_as_carry",
                                "n": 0})
                continue
            m = (t >= d["t1"]) & (t < d["t2"])
            if not m.any():
                continue
            kind[m] |= np.uint16(S.K_DROPOUT if k == "dropout" else S.K_EPISODE)
            applied.append({"t1": int(d["t1"]), "kind": k, "what": "flagged", "n": int(m.sum())})
        elif k == "step":
            fl = int(d["flag"])
            lo, hi = int(d["t_lo"]), int(d["t_hi"])
            if carry and lo < c_hi and hi >= c_lo:
                lo = max(lo, c_hi)
            m = (t >= lo) & (t <= hi)
            if not m.any():
                applied.append({"t1": int(d["t1"]), "kind": k,
                                "what": "step_range_empty_after_clip", "n": 0})
                continue
            if fl in (1, 3):
                corr[m] += d["j1"] if d["side"] == "after" else -d["j1"]
                cls[m] = np.maximum(cls[m], fl)
                what = f"corrected_flag{fl}"
            else:
                kind[m] |= np.uint16(S.K_STEP)
                what = "flagged"
            applied.append({"t1": int(d["t1"]), "kind": k, "what": what, "n": int(m.sum())})
        elif k in RUN_KINDS:
            m = (t >= int(d["t_lo"])) & (t <= int(d["t_hi"]))
            kind[m] |= np.uint16(K_EXCURSION)
            applied.append({"t1": int(d["t1"]), "kind": k, "what": "flagged",
                            "n": int(m.sum())})
    if carry:
        m = (t >= c_lo) & (t < c_hi)
        if carry["flag"] in (1, 3):
            corr[m] += carry["corr"]
            cls[m] = np.maximum(cls[m], carry["flag"])
            what = f"carry_corrected_flag{carry['flag']}"
        else:
            kind[m] |= np.uint16(S.K_STEP)
            what = "carry_flagged"
        applied.append({"t1": int(carry["t_step"]), "what": what, "n": int(m.sum())})
    sent = (kind & S.K_SENTINEL) > 0
    value = np.where(sent, np.nan, v_raw - L - corr)
    sflag = np.zeros(len(t), np.uint8)
    sflag[(cls == 1) | (L != 0)] = 1
    sflag[cls == 3] = 3
    sflag[(kind & ART_BITS_V2) > 0] = 9
    sflag[sent] = 8
    return value, corr, sflag, kind, applied
