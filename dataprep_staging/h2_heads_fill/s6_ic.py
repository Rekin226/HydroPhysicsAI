"""Registered initial-condition test of WP H2 (``H2.fill_acceptance.v1``, test H2.ic.lowo).

Leave-one-well-out on the January 2012 L1 heads (flag 0/1 monthly mean, any clean-day count)
of the 158 model wells: the twin's per-layer IDW against kriging with an external drift
(SRTM cell elevation) and a GP field (``ic_field.py``). Decision rule (registered): the
initial condition stays IDW unless KED or GP has a lower pooled LOWO RMSE; if both do, the
lower one. Reported: per-layer RMSE, a well-bootstrap 90 % interval of the RMSE difference
(2000 draws, seed 20261013), native-sd coverage of KED and GP.

Writes ``H2_evidence/registered/H2.ic.lowo.json``.

Usage: python -I s6_ic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import ic_field, io_h2, scoring  # noqa: E402

PREREG_ID = "H2.fill_acceptance.v1"
TEST_ID = "H2.ic.lowo"


def jan2012(h):
    """(sids, xy, value, layer 0-3, dem) of model wells with a January 2012 L1 value."""
    w = io_h2.model_wells().set_index("sid")
    j = io_h2.month_index(h.dates, "2012-01")
    v = np.where(np.isfinite(h.Y[:, j]), h.Y[:, j], h.partial_mean[:, j])
    keep = [i for i, s in enumerate(h.sids) if s in w.index and np.isfinite(v[i])
            and np.isfinite(h.meta.dem.iloc[i])]
    sids = [h.sids[i] for i in keep]
    xy = h.xy()[keep]
    lay = w.loc[sids, "layer"].to_numpy().astype(int)
    return sids, xy, v[keep], lay, h.meta.dem.to_numpy()[keep]


def main():
    h = io_h2.load("pre", with_daily=False)
    ok = prereg3.assert_can_score(PREREG_ID, C.WP, test_id=TEST_ID)
    sids, xy, v, lay, dem = jan2012(h)
    pred, sd = ic_field.lowo(xy, v, lay, dem)
    err = {k: pred[k] - v for k in pred}
    rm = {k: scoring.rmse(e) for k, e in err.items()}
    rng = np.random.default_rng(C.SEED_SPLIT)
    n = len(v)
    boot = {k: [] for k in ("ked", "gp")}
    for _ in range(2000):
        ix = rng.integers(0, n, n)
        base = np.sqrt(np.mean(err["idw"][ix] ** 2))
        for k in boot:
            boot[k].append(np.sqrt(np.mean(err[k][ix] ** 2)) - base)
    better = [k for k in ("ked", "gp") if rm[k] < rm["idw"]]
    decision = "idw" if not better else min(better, key=lambda k: rm[k])
    rep = {"test_id": TEST_ID, "n_wells": n, "sids": sids,
           "assert_can_score": {"sha256": ok["sha256"],
                                "post_registration_looks": ok["post_registration_looks"]},
           "rmse_lowo": rm, "decision": decision,
           "rmse_by_layer": {str(k): {m: scoring.rmse(err[m][lay == k]) for m in err}
                             for k in sorted(set(lay))},
           "n_by_layer": {str(k): int((lay == k).sum()) for k in sorted(set(lay))},
           "bootstrap_diff_vs_idw_p05_p50_p95": {
               k: [float(np.quantile(b, q)) for q in (0.05, 0.5, 0.95)] for k, b in boot.items()},
           "native_cov90": {k: float(np.mean(np.abs(err[k]) <= C.Z90 * sd[k])) for k in sd},
           "max_abs_err": {k: float(np.max(np.abs(e))) for k, e in err.items()},
           "pass": True,
           "note": "a decision test: it has no pass/fail threshold; 'pass' records that the "
                   "registered decision rule was applied"}
    scoring.jdump(rep, C.EVIDENCE_DIR / "registered" / f"{TEST_ID}.json")
    print(rm, decision)


if __name__ == "__main__":
    main()
