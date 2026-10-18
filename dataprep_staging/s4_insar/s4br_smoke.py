"""WP S4b repair round 2 (S4br): smoke test of ``s4br_evaluate.cmd_g1_ngl`` BEFORE the
pre-registration hashes it, on units that no S4b test uses: the OFF-FAN NGL stations inside the
multilooked grid (CHUA, GUKN, LSB0, PAKU, PINT, S103: not on active cells, so never G1 units),
in round 1's G1 window, with the spent WRA stations as tie stations of the combined-tie row.
The scoring and look calls are replaced by no-ops; the look is logged here (purpose
``smoke_test``). Outputs go to a scratch file and are not reported as results.

    python -I s4br_smoke.py      (hydro env, one CPU slot)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pyproj  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s4br_config as R  # noqa: E402
import s4br_evaluate as RE  # noqa: E402
import s4br_lib as RL  # noqa: E402
from harness import prereg3  # noqa: E402

OFFFAN = ["CHUA", "GUKN", "LSB0", "PAKU", "PINT", "S103"]


def main() -> int:
    prereg3.record_look(
        ["s4br.smoke.ngl_offfan", "s4b.wra_gnss.loso_preorigin"],
        "insar_gnss_smoke:2021-08..2022-12", R.WP, "smoke_test",
        members={"s4br.smoke.ngl_offfan": {
            "description": "off-fan NGL stations inside the multilooked grid (not on active "
                           "cells; never units of an S4b test) and their run-P2 pixels; code "
                           "smoke test of s4br_evaluate before the pre-registration",
            "ids": [f"ngl_gnss:{s}" for s in OFFFAN], "window": "2021-08..2022-12"}},
        note="S4br smoke test of the registered code on non-test units; results not reported")
    h = RL.ngl_holdings()
    g = RL.load_geom(C.PRIMARY)
    tr = pyproj.Transformer.from_crs(4326, 3826, always_xy=True)
    st = {}
    for s in OFFFAN:
        x, y = tr.transform(float(h.loc[s, "lon"]), float(h.loc[s, "lat"]))
        i, j, dist = Lb.station_pixel(g, x, y)
        st[s] = {"cell": -1, "ml_i": i, "ml_j": j, "pix_dist_m": dist, "status": "eligible"}
    fake = {"window": {"t0": C.GNSS_T0, "t1": C.GNSS_T1, "scope": "2021-08..2022-12"},
            "stations": st, "eligible": OFFFAN}
    RE._score = lambda tid: {"sha256": "smoke", "post_registration_looks": [],
                             "unverifiable_prior_looks": []}
    RE._look = lambda *a, **k: {"time": "smoke"}
    RE.design_doc = lambda: fake
    RE.ngl_station_set = lambda t0, t1: st
    scratch = R.EVID / "smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    RE.evid = lambda name: scratch / name
    out = RE.cmd_g1_ngl(None)
    rep = {k: (v.get("n_evaluable"), v.get("n_evaluable_ngl")) for k, v in
           out["report_only"].items()}
    print(json.dumps({"ran": True, "primary_n": out["primary"]["n_evaluable"],
                      "report_only_n": rep, "has_rms": out["primary"].get("rms_mm") is not None,
                      "ref_rows": len(out["reference_table_P2"])}, default=Lb._js))
    # (the run of 2026-10-17T21:42Z stopped at a np.loadtxt of this file, after cmd_g1_ngl had
    # completed and written it; replaced by pandas, not re-run)
    import pandas as pd
    d = pd.read_csv(scratch / "ngl_P2_insar_vertical_mm.csv", index_col=0)
    print("valid insar epochs per station:", d.notna().sum().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
