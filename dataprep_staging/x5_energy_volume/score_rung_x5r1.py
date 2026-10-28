"""Score rung X5.R1 (``I1.X5R1.v1``) against Q0 on the declared gates: the same metric code
as rung B (``dataprep_staging/i1/score_rung_b.py``: ``score_g2.metrics``, the temporal run's
fair columns, the forward plausibility rate), with I1.B.v1's six gates verbatim.

Arms: ``x5r1`` (primary) and ``x5r1t`` (pre-declared alternative) are gated against Q0;
``q0c`` (Q0 + the four-county census at the current conversion) is scored and reported only,
to separate "Changhua pumping added" from "irrigation volume doubled". P2f, the calibrated
T and S by zone, and the fitted readouts are reported for every arm, never gating.

Writes ``results/dataprep/reports/I1_X5R1/report.json`` once and refuses to overwrite.

Run from the repository root after batch 27 has finished:
    python dataprep_staging/x5_energy_volume/score_rung_x5r1.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging/i1"))

import score_g2 as S  # noqa: E402
import score_rung_b as B  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_X5R1"
GATED = ("x5r1", "x5r1t")
REPORTED = ("q0c",)


def ts_by_zone(suffix: str) -> dict:
    """Calibrated T (m2/day) and S per zone and layer, exponentiated from the theta file."""
    th = json.loads((B.RUNS / f"stage3_datum_gate_{suffix}/stage3_theta.json").read_text())
    th = th.get("theta", th)
    out = {}
    for k, v in th.items():
        if k.startswith(("log_T_", "log_S_")):
            vals = v if isinstance(v, list) else [v]
            out[k[4:]] = [float(math.exp(x)) for x in vals]
    return out


def score(suffix: str, st: pd.DataFrame) -> dict:
    m = B.score(suffix, st)
    m["T_S_by_zone"] = ts_by_zone(suffix)
    return m


def gates(m: dict, q: dict) -> dict:
    g = B.gates(m, q, p2f_gates=False)   # I1.B.v1's six gates; P2f reported, not gating
    return g


def main(argv=None) -> int:
    st = pd.read_parquet(S.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    M = {"Q0": score("q0", st)}
    M.update({k: score(k, st) for k in GATED + REPORTED})
    G = {k: gates(M[k], M["Q0"]) for k in GATED + REPORTED}
    passes = {k: bool(all(G[k].values())) for k in GATED}
    arm = "x5r1" if passes["x5r1"] else ("x5r1t" if passes["x5r1t"] else None)
    res = {"declaration": "I1.X5R1.v1",
           "partition": "seed 2027, site tol 200 m; leveling folds seed 2027 (design-time)",
           "metrics": M, "gates": {k: G[k] for k in GATED},
           "q0c_gates_reported_only": G["q0c"],
           "passes_all": passes,
           "decision": (f"{arm} passes every gate: run the declared seed-2029 confirmation of Q0 "
                        f"and {arm} (same gates against Q0)" if arm else
                        "x5r1 and x5r1t fail; no seed-2029 confirmation is run (declared: only if "
                        "every gate passes). The meaning follows the declaration's "
                        "interpretation, read with q0c, not this line.")}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report.json"
    with open(path, "x", encoding="utf-8") as f:   # scored once
        f.write(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
