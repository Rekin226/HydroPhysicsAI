"""W3r: how uncertain is the survey epoch of agency GroundHeight at the fan stations?

Uses only station dates, coordinates and the leveling displacement field (no GroundHeight
value, no DTM value, so no agency-minus-DTM residual). For the stations that pass the
metadata rules E1-E3, E5 and E6 of ``repair_wells`` at any establishment date:

* establishment and modification years (the GroundHeight survey epoch is undocumented and
  lies at one of them or in between);
* leveling displacement between the two candidate epochs, u(EstablishDate) - u(ModificationDate),
  and the epoch correction to the DTM sheet epoch under each assumption. Before the 2005
  campaign the displacement is extrapolated with the first increment's rate (2005-2008), which
  is an estimate only: the panel has 10-165 benchmarks per main campaign before 2005.

Usage: python -I repair_epoch_ambiguity.py <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from w3_dtm import repair_wells as rw  # noqa: E402


def main(out_json: str) -> int:
    rw.verify_v3()
    t = rw.eligibility(rw.station_table())
    rules = ["E1_coords_dtm", "E2_groundheight", "E3_stickup", "E5_flat",
             "E6_no_postorigin_edit"]
    s = t[(t["set"] == "fan") & t[rules].all(axis=1)].copy()
    dm, info = rw.displacement_model()
    s["corr_est"] = rw.corrections(dm, s, "t_establish")
    s["corr_mod"] = rw.corrections(dm, s, "t_modification")
    s["ambiguity"] = s["corr_est"] - s["corr_mod"]
    est_y = (pd.to_numeric(s["EstablishDate"], errors="coerce") // 10000).astype("Int64")
    mod_y = (pd.to_numeric(s["ModificationDate"], errors="coerce") // 10000).astype("Int64")
    out = {
        "stations_passing_E1_E2_E3_E5_E6": int(len(s)),
        "sites": int(s["site"].nunique()),
        "establish_year_counts": {str(k): int(v) for k, v in est_y.value_counts()
                                  .sort_index().items()},
        "modification_year_counts": {str(k): int(v) for k, v in mod_y.value_counts()
                                     .sort_index().items()},
        "established_before_2003": int((pd.to_numeric(s["EstablishDate"], errors="coerce")
                                        < 20030101).sum()),
        "established_2005_2022": int(s["E4_establish"].sum()),
        "epoch_correction_if_survey_at_establishment_m": rw.summarize(s["corr_est"]),
        "epoch_correction_if_survey_at_modification_m": rw.summarize(s["corr_mod"]),
        "ambiguity_u_establish_minus_u_modification_m": rw.summarize(s["ambiguity"]),
        "share_ambiguity_gt_0.30_m": float(np.mean(np.abs(s["ambiguity"].dropna()) > 0.30)),
        "displacement_years": info["years"],
        "note": "no GroundHeight or DTM value enters these numbers; displacement before "
                "2005.37 is a linear extrapolation of the 2005-2008 increment",
    }
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
