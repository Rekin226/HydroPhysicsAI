"""WP H2 design stage 1: re-decide the daily-bridge form on pre-origin design realizations.

The head gap-fill pilot chose the bridge with the well's mean deviation curve ('shape') over
the plain linear bridge ('lin') on the Feb-Mar window in 2013-2025, which includes 2023-25
truth (U7: no 2023+ observation in a design choice). This script repeats the comparison on
pre-origin data only, on design-time units:

* family d, realizations r = 0..4 (random daily gaps of 5-60 days, one per eligible well);
* family D, the 11 Feb - 19 Mar window removed in every year 2013-2022.

Decision rule (declared here before the run): the form with the lower pooled RMSE over the
clean-truth bridged months of d r = 0..4 and D 2013-2022 together; ties keep 'shape' (the
plan's form). Deviation curves use years <= 2022 only. Writes
``H2_evidence/design/bridge_form.json``; ``config.BRIDGE_FORM`` must equal the decision
(checked by ``s2_design.py``).

Usage: python -I s1_bridge_form.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from h2_heads_fill import bridge, io_h2, scoring  # noqa: E402
from h2_heads_fill import config as C  # noqa: E402

OUT = C.EVIDENCE_DIR / "design" / "bridge_form.json"


def main():
    h = io_h2.load("pre")
    clean = io_h2.clean_truth(h)
    years = range(2012, 2023)
    parts = []
    for r in C.DESIGN_R["d"]:
        ev = bridge.evaluate_windows(h, bridge.d_windows(h, r, C.PRE_END), years, f"d{r}")
        parts.append(ev)
    parts.append(bridge.evaluate_windows(h, bridge.D_windows(h, C.D_YEARS_PRE), years, "D"))
    ev = pd.concat(parts, ignore_index=True)
    ev["clean"] = clean[ev.w.to_numpy(), ev.t.to_numpy()]
    ok = ev[ev.ok & ev.clean & np.isfinite(ev.pred_shape) & np.isfinite(ev.pred_lin)]
    res = {}
    for name, g in [("pooled", ok), ("d", ok[ok.kind.str.startswith("d")]),
                    ("D", ok[ok.kind == "D"])]:
        res[name] = {"n": int(len(g)), "n_wells": int(g.sid.nunique()),
                     "rmse_shape": scoring.rmse(g.pred_shape - g.truth),
                     "rmse_lin": scoring.rmse(g.pred_lin - g.truth),
                     "rmse_partial": scoring.rmse(g.partial - g.truth)}
    p = res["pooled"]
    decision = "lin" if p["rmse_lin"] < p["rmse_shape"] else "shape"
    rep = {"schema": "h2-bridge-form/1", "units": ["H2:design_d_r0-4", "H2:D_febmar_2013-2022"],
           "rule": "lower pooled clean-truth RMSE over d r0-4 and D 2013-2022; ties keep "
                   "'shape'", "decision": decision, "tables": res,
           "n_attempted": int(len(ev)), "n_bridgeable": int(ev.ok.sum()),
           "note": "design-time evidence (pre-origin, years <= 2022 only)"}
    scoring.jdump(rep, OUT)
    print(decision, res)


if __name__ == "__main__":
    main()
