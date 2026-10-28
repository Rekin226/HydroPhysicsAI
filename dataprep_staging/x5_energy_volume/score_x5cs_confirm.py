"""Score I1.X5CS.v1.confirm2029: q0_csk01_s29 against q0_s29 on the declared column gates
(leveling, Changhua leveling, rising share per county, plausibility), with the seed-2027
values beside them. Writes results/dataprep/reports/I1_X5CS_confirm2029/report.json once.

Usage (repo root, after results/twin_runs/par30/run.sh):
    python dataprep_staging/x5_energy_volume/score_x5cs_confirm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging/i1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_rung_b as B  # noqa: E402
from prereg_x5cs import RATE_TOL  # noqa: E402
from score_rung_x5cs import column  # noqa: E402
from uplift import uplift_share  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_X5CS_confirm2029"
SEED = 2029


def metrics(arm: str, seed: int) -> dict:
    p = B.RUNS / f"stage3_datum_gate_{arm}/coupled_leveling/vep_zonal_leveling.json"
    if int(json.loads(p.read_text()).get("fold_seed", 0)) != seed:
        raise SystemExit(f"{p}: fold seed is not {seed}")
    return {**column(arm, seed), **B.plausibility(arm), **B.p2f(arm),
            **uplift_share(arm, RATE_TOL)}


def gates(m: dict, q: dict) -> dict:
    g = {"leveling >= q0_s29 + 0.02": m["leveling"] >= q["leveling"] + 0.02,
         "Changhua leveling >= q0_s29's": m["leveling_changhua"] >= q["leveling_changhua"],
         "rising share Changhua <= q0_s29 + 0.05":
             m["uplift_share_ch"] <= q["uplift_share_ch"] + 0.05,
         "rising share elsewhere <= q0_s29 + 0.05":
             m["uplift_share_rest"] <= q["uplift_share_rest"] + 0.05,
         "plausibility >= 0 both counties": min(m["plaus_ch"], m["plaus_rest"]) >= 0}
    return {k: bool(v) for k, v in g.items()}


def main() -> int:
    M = {"q0_s29": metrics("q0_s29", SEED), "q0_csk01_s29": metrics("q0_csk01_s29", SEED)}
    seed2027 = {a: column(a) for a in ("q0", "q0_csk01")}
    G = gates(M["q0_csk01_s29"], M["q0_s29"])
    ok = all(G.values())
    res = {"declaration": "I1.X5CS.v1.confirm2029", "rate_tol_cm_per_yr": RATE_TOL,
           "metrics": M, "seed2027_columns": seed2027, "gates": G, "passes_all": ok,
           "decision": ("q0_csk01's column is promoted as the column for Q0's heads" if ok else
                        "no promotion; Q0's three-zone column stays")}
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "report.json", "x", encoding="utf-8") as f:   # scored once
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
