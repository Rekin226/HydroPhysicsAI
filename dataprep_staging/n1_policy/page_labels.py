"""The policy page's N1 labels, derived from the scored N1 results (nothing re-scored).

Reads step B, step C and the Changhua fixes (results/dataprep/reports/N1_policy/) and writes
page_labels.json beside them for ``twin.viewer_app --policy-response``: per county the head
and subsidence verdicts of step C, the Changhua status and the registrations behind them.
Written once.
Usage: python -I page_labels.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, rel_to_repo  # noqa: E402

REP = REPO / "results/dataprep/reports/N1_policy"
OUT = REP / "page_labels.json"
PREREG = REPO / "results/dataprep/preregistration"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def head_row(t: dict) -> dict:
    cl, zr = t["M_vs_CL"], t["M_vs_ZR"]
    return {"rmse_model_m": cl["rmse_model"], "rmse_cl_m": cl["rmse_base"],
            "rmse_zr_m": zr["rmse_base"], "n_wells": cl["n_wells"],
            "verdict_cl": cl["verdict"], "verdict_zr": zr["verdict"],
            "slope_obs_on_model": t["slope_obs_on_M"]["slope"],
            "slope_ci": t["slope_obs_on_M"]["ci"]}


def lev_row(t: dict) -> dict:
    cl, zr = t["M_vs_CL"], t["M_vs_ZR"]
    return {"rms_model_cm": 100 * cl["rmse_model"], "rms_cl_cm": 100 * cl["rmse_base"],
            "rms_zr_cm": 100 * zr["rmse_base"], "n_benchmarks": t["n_benchmarks"],
            "verdict_cl": cl["verdict"], "verdict_zr": zr["verdict"],
            "slope_obs_on_model": t["slope_obs_on_M"]["slope"],
            "slope_ci": t["slope_obs_on_M"]["ci"]}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{rel_to_repo(OUT)} exists; written once")
    paths = {k: REP / k / "results.json" for k in ("step_b", "step_c", "fix_c")}
    b, c, f = (json.loads(paths[k].read_text(encoding="utf-8"))["tests"]
               for k in ("step_b", "step_c", "fix_c"))
    bch = b["N1P.B_dose_changhua"]
    fixes = {"F1": f["N1F.F1_lev_changhua"]["verdict"]}
    f2h, f2l = f["N1F.F2_heads_changhua"]["M_vs_R1c"], f["N1F.F2_lev_changhua"]["M_vs_R1c"]
    fixes["F2"] = f"heads vs R1c {f2h['verdict']}, leveling vs R1c {f2l['verdict']}"
    regs = {r: sha(PREREG / f"{r}.json")
            for r in ("N1.policy_response.v1", "N1.changhua_fix.v1")}
    out = {
        "product": "n1_policy/page_labels", "version": "v1",
        "window": "2016-01..2018-12 (masked refits; design-time window)",
        "heads": {"yunlin": head_row(c["N1P.C_heads_yunlin"]),
                  "changhua": head_row(c["N1P.C_heads_changhua"])},
        "subsidence": {"yunlin": lev_row(c["N1P.C_lev_yunlin"]),
                       "changhua": lev_row(c["N1P.C_lev_changhua"])},
        "changhua": {"status": "not_supported",
                     "dose_obs_cm_per_gwh": bch["beta_obs_cm_per_GWh"],
                     "dose_model_cm_per_gwh": bch["beta_model_cm_per_GWh"],
                     "dose_verdict": bch["verdict"], "fixes": fixes},
        "registrations": regs,
        "inputs": {k: {"path": rel_to_repo(p), "sha256": sha(p)} for k, p in paths.items()},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
