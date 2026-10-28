"""Score rung I1.X5CW.v1 (route 4) against Q0: I1.B.v1's six gates (the rung B scorer's code)
plus the two registered Changhua interannual gates (interannual_metrics.py). ``cws`` is the
primary arm, ``cwp`` the pre-declared alternative; the base arm (if it is not Q0) is reported
for attribution. Writes results/dataprep/reports/I1_X5CW/report.json once.

Usage (repo root, after batch 28): python dataprep_staging/x5_changhua_water/score_rung_x5cw.py
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
from interannual_metrics import changhua_interannual  # noqa: E402

OUT = REPO / "results/dataprep/reports/I1_X5CW"
ARMS = ("cws", "cwp")


def sw_scale(suffix: str):
    th = json.loads((B.RUNS / f"stage3_datum_gate_{suffix}/stage3_theta.json").read_text())
    th = th.get("theta", th)
    if "sw_scale" in th:                  # readout: a scalar or one value per zone
        return th["sw_scale"]
    v = th.get("log_sw_scale")
    if v is None:
        return None
    return [math.exp(x) for x in v] if isinstance(v, list) else math.exp(v)


def main() -> int:
    base = json.loads((B.RUNS / "par28/base.json").read_text())["base"]
    st = pd.read_parquet(S.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    names = ["q0"] + ([base] if base != "q0" else []) + list(ARMS)
    M = {k: B.score(k, st) for k in names}
    for k in names:
        M[k]["changhua_interannual"] = changhua_interannual(k)
    for k in ARMS:
        M[k]["sw_scale"] = sw_scale(k)
    q = M["q0"]
    a_q0 = q["changhua_interannual"]["amplitude_A"]
    r_q0 = q["changhua_interannual"]["interannual_corr"]
    G = {}
    for k in ARMS:
        g = B.gates(M[k], q, p2f_gates=False)
        ci = M[k]["changhua_interannual"]
        g["Changhua amplitude |ln A| < |ln A_Q0|"] = bool(
            abs(math.log(ci["amplitude_A"])) < abs(math.log(a_q0)))
        g["Changhua interannual corr >= Q0 - 0.02"] = bool(ci["interannual_corr"] >= r_q0 - 0.02)
        G[k] = g
    passes = {k: bool(all(G[k].values())) for k in ARMS}
    arm = "cws" if passes["cws"] else ("cwp" if passes["cwp"] else None)
    res = {"declaration": "I1.X5CW.v1", "base": base,
           "partition": "seed 2027, site tol 200 m; leveling folds seed 2027 (design-time)",
           "metrics": M, "gates": G, "passes_all": passes,
           "decision": (f"{arm} passes every gate: run the declared seed-2029 confirmation"
                        if arm else "cws and cwp fail; no seed-2029 confirmation is run. The "
                        "meaning follows the declaration's interpretation, not this line.")}
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "report.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
