"""Smoke test of the T3 code path WITHOUT any head value.

Usage: python -I t3_smoke.py OUT_JSON

1. The patched driver panel on the PRE-ORIGIN sources (only the stations table of the
   pre-origin heads product is read) over 2012-01..2026-09: the X1 electricity drivers must
   equal the I1 drivers on every census-recorded month and be recorded on the downscaled ones.
2. ``prospective.fit`` and ``predict`` end to end with SYNTHETIC heads (random walk plus a
   seasonal cycle; ``heads_matrix`` is replaced in-process), to check that the frozen fit and
   the predictions run with the X1 drivers. No real head is read; no score is computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t3  # noqa: E402

P = t3.P


def main(out: str) -> None:
    sids = pd.read_csv(t3.panel.WELLS_CSV, dtype={"sid": str})["sid"].tolist()
    dd = P.months("2012-01", "2026-09")
    src = P.SOURCES["preorigin"]
    v0, r0, _ = t3._ORIG_DRIVER_PANEL(src, sids, dd)
    with t3.patched():
        v1, r1, attrs = P.driver_panel(src, sids, dd)
    res = {"info": dict(t3._INFO)}
    for k in ("electricity_local_3km", "electricity_county"):
        both = r0[k] & r1[k]
        rel = np.abs(v1[k][both] - v0[k][both]) / np.maximum(np.abs(v0[k][both]), 1.0)
        res[k] = {"months_recorded_I1": int(r0[k][0].sum()),
                  "months_recorded_T3": int(r1[k][0].sum()),
                  "max_rel_diff_on_shared": float(rel.max()),
                  "T3_only_months": [d.strftime("%Y-%m") for d, a, b in
                                     zip(dd, r0[k][0], r1[k][0], strict=True) if b and not a]}
    rng = np.random.default_rng(0)

    def fake_heads(src_, sids_, hd, use_override=None):
        W, T = len(sids_), len(hd)
        season = np.sin(2 * np.pi * (np.asarray(hd.month) - 1) / 12)[None, :]
        return np.cumsum(rng.normal(0, 0.1, (W, T)), axis=1) + season

    orig = P.heads_matrix
    P.heads_matrix = fake_heads
    try:
        with t3.patched():
            prm = P.fit("preorigin", "2026-09", None, sids=sids[:6])
            y = fake_heads(None, sids[:6], dd[1:])
            pr = P.predict(prm, "preorigin", "2026-09", y, sids=sids[:6])
    finally:
        P.heads_matrix = orig
    res["synthetic_fit"] = {"wells_fitted": sum(w.get("status") == "fitted"
                                               for w in prm["wells"].values()),
                            "pred_finite_share_arx1": float(np.isfinite(pr["arx1"]).mean())}
    Path(out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
