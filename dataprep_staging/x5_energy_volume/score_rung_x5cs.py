"""Score rung I1.X5CS.v1 against Q0 on the declared gates: the capped county-split (six-zone)
compaction column refitted on unchanged heads.

Head metrics (kfold, county corr and mean RMSE, temporal) are the parent run's
(``score_rung_b.score`` on q0 or q0c), since the heads are unchanged; leveling, plausibility,
fan subsidence and P2f are the capped arm's own column and forward. Gates: I1.B.v1's six, then
leveling >= Q0 + 0.02, Changhua leveling >= Q0's, and the no-added-rising share per county
(``uplift.uplift_share`` at RATE_TOL) <= Q0's + 0.05. q0_csk1 is primary; q0_csk01 is gated
only if q0_csk1 fails; the q0c arms are reported only.

Writes ``results/dataprep/reports/I1_X5CS/report.json`` once and refuses to overwrite.

Run from the repository root after results/twin_runs/par29/run.sh has finished:
    python dataprep_staging/x5_energy_volume/score_rung_x5cs.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging/i1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_g2 as S  # noqa: E402
import score_rung_b as B  # noqa: E402
from prereg_x5cs import RATE_TOL  # noqa: E402
from uplift import uplift_share  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_X5CS"
ARMS = {"q0_csk1": "q0", "q0_csk01": "q0", "q0c_csk1": "q0c", "q0c_csk01": "q0c"}
GATED = ("q0_csk1", "q0_csk01")


def column(suffix: str, seed: int = 2027) -> dict:
    """Leveling fit, by zone, and Ske/Skv per column zone from the arm's column json."""
    p = B.RUNS / f"stage3_datum_gate_{suffix}/coupled_leveling/vep_zonal_leveling.json"
    d = json.loads(p.read_text())
    if int(d.get("fold_seed", 0)) != seed:
        raise SystemExit(f"{p}: leveling fold seed {d.get('fold_seed')}, needs {seed}")
    z = d["r2_outoffold_by_zone"]
    names = d.get("zone_names") or [str(i) for i in range(len(d["zonal"]))]
    return {"leveling": float(d["r2_outoffold"]),
            "leveling_changhua": float(z["county:changhua"]["r2"]),
            "leveling_rest": float(z["county:rest"]["r2"]),
            "leveling_by_zone": {k: v["r2"] for k, v in z.items()
                                 if isinstance(v, dict) and "r2" in v},
            "ske_skv_max": d.get("ske_skv_max"),
            "ske_skv_by_zone": {n: math.exp(c["log_ske"] - c["log_skv"])
                                for n, c in zip(names, d["zonal"], strict=True)}}


def score(arm: str, parent: str, heads: dict) -> dict:
    m = dict(heads[parent])                   # head metrics: the parent's (heads unchanged)
    m.update(column(arm))                     # leveling: this arm's column
    m.update(B.plausibility(arm))             # forward: this arm's run
    m.update(B.p2f(arm))
    m.update(uplift_share(arm, RATE_TOL))
    m["head_metrics_from"] = parent
    return m


def gates(m: dict, q: dict) -> dict:
    g = B.gates(m, q, p2f_gates=False)        # I1.B.v1's six gates
    g["leveling >= Q0 + 0.02"] = m["leveling"] >= q["leveling"] + 0.02
    g["Changhua leveling >= Q0's"] = m["leveling_changhua"] >= q["leveling_changhua"]
    g["rising share Changhua <= Q0 + 0.05"] = m["uplift_share_ch"] <= q["uplift_share_ch"] + 0.05
    g["rising share elsewhere <= Q0 + 0.05"] = (m["uplift_share_rest"]
                                                <= q["uplift_share_rest"] + 0.05)
    return {k: bool(v) for k, v in g.items()}


def main() -> int:
    st = pd.read_parquet(S.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    heads = {p: B.score(p, st) for p in sorted(set(ARMS.values()))}
    q0 = dict(heads["q0"])
    q0.update(column("q0"))
    q0.update(uplift_share("q0", RATE_TOL))
    M = {"Q0": q0}
    M.update({a: score(a, p, heads) for a, p in ARMS.items()})
    G = {a: gates(M[a], q0) for a in ARMS}
    passes = {a: bool(all(G[a].values())) for a in ARMS}
    arm = "q0_csk1" if passes["q0_csk1"] else ("q0_csk01" if passes["q0_csk01"] else None)
    res = {"declaration": "I1.X5CS.v1",
           "partition": "seed 2027, site tol 200 m; leveling folds seed 2027 (design-time)",
           "rate_tol_cm_per_yr": RATE_TOL,
           "metrics": M, "gates": {a: G[a] for a in GATED},
           "q0c_gates_reported_only": {a: G[a] for a in ARMS if a not in GATED},
           "passes_all": passes,
           "q0_csk01_gated": not passes["q0_csk1"],
           "decision": (f"{arm} passes every gate: run the declared seed-2029 column "
                        f"confirmation of Q0's three-zone column and {arm}'s on the same heads "
                        f"and rescore the column gates" if arm else
                        "q0_csk1 and q0_csk01 fail; no seed-2029 confirmation is run. The "
                        "meaning follows the declaration's interpretation, read with the q0c "
                        "arms, not this line.")}
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "report.json", "x", encoding="utf-8") as f:   # scored once
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
