"""WP S4b repair round: split the change of the G1 station errors from run P to run F into the
cross-gap increments and the rest (arithmetic on ``S4b_evidence/repair_diag.json`` only; the
GNSS term cancels in the difference, so this part is InSAR-only).

    python -I s4b_r2_split.py   -> S4b_evidence/repair_pf_split.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, float)))))


def main() -> int:
    d = json.loads((C.EVID / "repair_diag.json").read_text())["pieces"]
    P = pd.DataFrame(d["P"]["rows"]).set_index("station")
    F = pd.DataFrame(d["F"]["rows"]).set_index("station")
    c = P.index.intersection(F.index)
    ok = (P.loc[c, ["err_mm", "gap_sum_mm"]].notna().all(axis=1)
          & F.loc[c, ["err_mm", "gap_sum_mm"]].notna().all(axis=1))
    c = c[ok.to_numpy()]
    de = F.loc[c, "err_mm"] - P.loc[c, "err_mm"]
    dg = F.loc[c, "gap_sum_mm"] - P.loc[c, "gap_sum_mm"]
    dr = F.loc[c, "rest_mm"] - P.loc[c, "rest_mm"]
    out = {"n_stations": int(len(c)), "stations": list(c),
           "excluded": sorted(set(P.index) - set(c)),
           "rms_err_change_mm": rms(de), "rms_gap_part_change_mm": rms(dg),
           "rms_within_segment_part_change_mm": rms(dr),
           "rms_gap1_change_mm": rms(F.loc[c, "gap1_mm"] - P.loc[c, "gap1_mm"]),
           "rms_gap2_change_mm": rms(F.loc[c, "gap2_mm"] - P.loc[c, "gap2_mm"]),
           "corr_err_change_gap_change": float(np.corrcoef(de, dg)[0, 1]),
           "g1_rms_P_mm": rms(P.loc[c, "err_mm"]), "g1_rms_F_mm": rms(F.loc[c, "err_mm"]),
           "note": "err(F) - err(P) per station; the GNSS series is the same in both runs, so "
                   "the difference is InSAR (with the registered offset tie) only"}
    Lb.jdump(out, C.EVID / "repair_pf_split.json")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
