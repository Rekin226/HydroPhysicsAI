"""Model-free predictors of the column's leveling target (round-12 exploratory).

P_hist   : the benchmark's own pre-2012 rate, continued (needs >= 3 surveys before 2012).
P_nbr    : neighbours' same-period elevation change, inverse-distance weighted (kept sites only).
P_nbrhist: neighbours' pre-2012 rate where the site has no history, else P_hist.
P_zero   : no subsidence.
Scored on the same pairs and the same pooled R2 as ``calibrate_coupled``'s gate, under the
gate's random per-benchmark folds and under 10 km spatial blocks.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.panel import (
    build,
    elev_curve,
    pooled_scores,
    random_folds,
    spatial_blocks,
    years,
)


def hist_rate(curve, t_split: float = 2012.0, min_n: int = 3, since: float = 2002.0):
    """Subsidence rate m/yr from surveys in [since, t_split); NaN when too few."""
    t, e = curve
    k = (t >= since) & (t < t_split)
    if k.sum() < min_n or np.ptp(t[k]) < 2.0:
        return np.nan
    return -float(np.polyfit(t[k], e[k], 1)[0])


def nbr_change(sid, tq, kept, tree, kept_xy, curves, xy, k=8, power=2.0):
    """IDW of kept neighbours' elevation drop between tq[0] and each tq (interpolated in
    time inside each neighbour's own survey span)."""
    d, idx = tree.query(xy[sid], k=min(k * 3, len(kept)))
    num = np.zeros(len(tq))
    den = np.zeros(len(tq))
    used = 0
    for dist, j in zip(d, idx, strict=True):
        t, e = curves[kept[j]]
        if tq[0] < t[0] - 0.3 or tq[-1] > t[-1] + 0.3:
            continue
        ei = np.interp(tq, t, e)
        w = 1.0 / max(dist, 50.0) ** power
        num += w * (ei[0] - ei)
        den += w
        used += 1
        if used >= k:
            break
    return num / den if used else np.full(len(tq), np.nan)


def run(folds_kind: str = "random"):
    panel, sub, xy = build()
    curves = elev_curve(panel)
    sids = sorted(sub)
    XY = np.array([xy[s] for s in sids])
    folds = random_folds(len(sids)) if folds_kind == "random" else spatial_blocks(XY)
    rate_h = {s: hist_rate(curves[s]) for s in sids}

    pred = {k: {} for k in ("zero", "hist", "nbr", "nbr_rate", "nbrhist", "blend")}
    for f in np.unique(folds):
        kept = [s for s, ff in zip(sids, folds, strict=True) if ff != f]
        kxy = np.array([xy[s] for s in kept])
        tree = cKDTree(kxy)
        kept_rate = np.array([rate_h[s] for s in kept])
        okr = np.isfinite(kept_rate)
        tree_r = cKDTree(kxy[okr])
        for s in (s for s, ff in zip(sids, folds, strict=True) if ff == f):
            tq = years(sub[s].index)
            el = tq - tq[0]
            pred["zero"][s] = np.zeros(len(tq))
            pred["hist"][s] = rate_h[s] * el
            pred["nbr"][s] = nbr_change(s, tq, kept, tree, kxy, curves, xy)
            d, idx = tree_r.query(xy[s], k=8)
            w = 1.0 / np.maximum(d, 50.0) ** 2
            nr = float((w * kept_rate[okr][idx]).sum() / w.sum())
            pred["nbr_rate"][s] = nr * el
            pred["nbrhist"][s] = (rate_h[s] if np.isfinite(rate_h[s]) else nr) * el
            a, b = pred["nbr"][s], pred["nbrhist"][s]
            pred["blend"][s] = np.where(np.isfinite(a), 0.5 * (a + b), b)
    out = {k: pooled_scores(v, sub, sids) for k, v in pred.items()}
    return out, pred, sub, xy, sids, folds


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "random"
    out, *_ = run(kind)
    print(json.dumps({"folds": kind, **out}, indent=1))
