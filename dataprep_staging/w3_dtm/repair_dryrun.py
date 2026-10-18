"""W3r dry run before the registration (no agency-minus-DTM residual is computed).

* hashes of the v3 outputs against the v3 manifest;
* the candidate station table and the eligibility counts (metadata and DTM only);
* the extended displacement model (campaign years, increments) and its equality with the W3
  model on the W3 years;
* a synthetic sign check: agency values replaced by the DTM moved to the survey epoch minus
  0.25 m, so the statistic must return +0.25 m;
* the eligible unit lists that the pre-registration will name.

Usage: python -I repair_dryrun.py <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
from w3_dtm import repair_wells as rw  # noqa: E402


def main(out_json: str) -> int:
    v3 = rw.verify_v3()
    t = rw.eligibility(rw.station_table())
    counts = rw.eligibility_counts(t)
    dm, info = rw.displacement_model()
    e = t[t["eligible"]]
    pts = e[["x", "y"]].to_numpy(dtype="float64")
    eq = rw.check_equals_w3_model(pts, np.array([2011.5, 2014.0, 2017.5, 2020.9]))
    syn = rw.synthetic_check(t, dm, delta=0.25)
    units = {s: sorted(e.loc[e["set"] == s, "sid"].tolist()) for s in ("fan", "nonfan")}
    sites = {s: sorted(e.loc[e["set"] == s, "site"].unique().tolist()) for s in ("fan",
                                                                               "nonfan")}
    out = {"v3_sha256_verified": v3, "eligibility_counts": counts,
           "displacement_model": info, "gap_scaled_equals_w3_model_max_abs_m": eq,
           "synthetic_check": syn, "eligible_units": units, "eligible_sites": sites,
           "survey_epoch_range_eligible": [float(e["t_establish"].min()),
                                           float(e["t_establish"].max())],
           "note": "no agency-minus-DTM residual was computed in this run"}
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "eligible_units"}, indent=1,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
