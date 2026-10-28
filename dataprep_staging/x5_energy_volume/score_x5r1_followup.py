"""Score I1.X5R1.v2_postfailure (exploratory, cannot promote): leveling R2 of the six-zone
county-split columns (q0_cs, q0c_cs, x5r1_cs) beside their three-zone batch-27 columns, the
reading fixed in the registration, and the forward 2023-25 rate by county (a negative rate
flags the R1c six-zone uplift artefact). Writes results/dataprep/reports/I1_X5R1_followup/
report.json once.

Usage (repo root, after results/twin_runs/par27f/run.sh): python dataprep_staging/x5_energy_volume/score_x5r1_followup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging/i1"))

import score_rung_b as B  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_X5R1_followup"
PAIRS = (("q0", "q0_cs"), ("q0c", "q0c_cs"), ("x5r1", "x5r1_cs"))


def column(suffix: str) -> dict:
    p = B.RUNS / f"stage3_datum_gate_{suffix}/coupled_leveling/vep_zonal_leveling.json"
    d = json.loads(p.read_text())
    z = d["r2_outoffold_by_zone"]
    return {"r2_outoffold": d["r2_outoffold"], "r2_insample": d["r2_insample"],
            "r2_changhua": z["county:changhua"]["r2"], "r2_rest": z["county:rest"]["r2"],
            "by_zone": {k: v["r2"] for k, v in z.items() if isinstance(v, dict) and "r2" in v},
            "zonal": d["zonal"], **B.plausibility(suffix)}


def reading(m: dict, arm: str, ctrl_gain: float) -> dict:
    cs, base = m[f"{arm}_cs"], m[arm]
    gain = cs["r2_outoffold"] - base["r2_outoffold"]
    q0 = m["q0"]
    shared = (cs["r2_outoffold"] >= q0["r2_outoffold"] - 0.02
              and cs["r2_changhua"] >= q0["r2_changhua"] and gain - ctrl_gain >= 0.10)
    verdict = ("SHARED COLUMN supported" if shared else
               "HEADS INCONSISTENT" if cs["r2_outoffold"] < 0.50 else "PARTIAL")
    return {"gain": gain, "gain_minus_control": gain - ctrl_gain, "verdict": verdict,
            "uplift_flag": bool(cs["plaus_ch"] < 0 or cs["plaus_rest"] < 0)}


def main() -> int:
    M = {s: column(s) for pair in PAIRS for s in pair}
    ctrl = M["q0_cs"]["r2_outoffold"] - M["q0"]["r2_outoffold"]
    res = {"declaration": "I1.X5R1.v2_postfailure",
           "label": "post-failure, exploratory; never validation; cannot promote",
           "heads_identity": json.loads((B.RUNS / "par27f/heads_identity.json").read_text()),
           "metrics": M, "control_gain_q0": ctrl,
           "readings": {a: reading(M, a, ctrl) for a in ("q0c", "x5r1")},
           "control_uplift_flag": bool(M["q0_cs"]["plaus_ch"] < 0 or M["q0_cs"]["plaus_rest"] < 0)}
    print(json.dumps(res, indent=1, default=float))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "report.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
