"""WP P4m: write the machine-readable WP report ``results/dataprep/reports/P4m.json``.

    cd dataprep_staging
    python -m p4_changhua.report_p4m

Reads the staged results, the published products (``P4m_evidence/publish.json``) and adds the
post-hoc sensitivity of the T2b theta-refit scale to the refits' predicted remaining decrease
(labelled post hoc; it uses only numbers already in ``refit/results.json``). The prose report
``P4m.md`` is written by hand from the same numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import REPORTS_DIR  # noqa: E402

from p4_changhua import p4m_lib as L  # noqa: E402

ST = L.STAGE


def _vertex(l0: float, l5: float, l1: float) -> float:
    a = 2 * (l0 - 2 * l5 + l1)
    b = -3 * l0 + 4 * l5 - l1
    return -b / (2 * a)


def posthoc_s_theta(refit: dict) -> dict:
    lv = refit["levels"]
    obj = {k: lv[k]["objective"] for k in ("0.0", "0.5", "1.0")}
    rem = {k: lv[k]["predicted_remaining_relative_decrease"] for k in obj}
    sf = refit["frozen"]["s_frozen_cal"]
    adj = {k: obj[k] * (1 - rem[k]) for k in obj}
    v_all = _vertex(adj["0.0"], adj["0.5"], adj["1.0"])
    v_s0 = _vertex(adj["0.0"], obj["0.5"], obj["1.0"])
    return {"label": "post hoc (written after the recorded arm was scored; uses only the T2b "
                     "refit objectives and their predicted remaining decreases)",
            "s_theta_registered": refit["s_theta"],
            "s_theta_all_refits_lowered_by_predicted_remaining": v_all,
            "s_theta_only_s0_lowered": v_s0,
            "theta_bias_share_range": sorted([(v - sf) / (1 - sf) for v in
                                              (refit["s_theta"], v_all, v_s0)])}


def main() -> int:
    run = json.loads((ST / "run" / "results.json").read_text())
    refit = json.loads((ST / "refit" / "results.json").read_text())
    ev = json.loads((ST / "evidence" / "results.json").read_text())
    rec = json.loads((ST / "recorded" / "results.json").read_text())
    reco = json.loads((ST / "recommendation" / "recommendation_preorigin.json").read_text())
    pub = json.loads((L.EVID / "publish.json").read_text())
    t1 = run["T1"]["groups"]
    t2 = run["T2"]
    ph = posthoc_s_theta(refit)
    L.jdump(ph, L.EVID / "posthoc_s_theta.json")
    acc = [dict(a) for a in pub["acceptance"]]
    out = {
        "wp": "P4m", "status": "passed", "date": "2026-10-17",
        "theta": "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json sha256 "
                 "789d0e45... (batch 8's merged cwa theta; not the deliverable's registered "
                 "datum_gate model of commit 709f660)",
        "preregistration": ["results/dataprep/preregistration/P4m.metered_diagnostics.v1.json",
                            "results/dataprep/preregistration/P4m.recorded_arm_2023_25.v1.json"],
        "acceptance": acc,
        "products": [pub["preorigin"], pub["recorded"]],
        "recommendation_sha256": sha256_file(ST / "recommendation" /
                                             "recommendation_preorigin.json"),
        "T1": {g: t1[g] for g in t1},
        "T1_differences": run["T1"]["differences"],
        "T2": {k: {kk: vv for kk, vv in v.items() if kk not in ("J_all", "J_changhua")}
               for k, v in t2.items()},
        "T2b": {"s_frozen_cal": refit["frozen"]["s_frozen_cal"], "s_theta": refit["s_theta"],
                "theta_bias_share_of_gap": refit["theta_bias_share_of_gap"],
                "stop_reasons": {k: v["stop_reason"] for k, v in refit["levels"].items()},
                "predicted_remaining": {k: v["predicted_remaining_relative_decrease"]
                                        for k, v in refit["levels"].items()},
                "insample_scores_at_refits": refit["insample_scores_at_refits"],
                "dtheta": refit["dtheta_refit1_minus_refit0"], "posthoc": ph},
        "T2c_top": run["T2c"]["rows"][:8],
        "T3": {k: run["T3"][k] for k in ("transient", "trend", "channels", "disappears",
                                         "mean_channel_significantly_negative", "spinup")},
        "evidence": ev,
        "T4_recorded_arm": rec["scores"],
        "T4_forcing_GWh_per_yr": {k: sum(v) / 2.5 for k, v in
                                  rec["forcing_GWh_per_month"].items() if k != "months"},
        "recommendation": reco["recommendation"],
        "changhua_multiplier_rule": reco["changhua_specific_response_multiplier"],
        "ledger": {"fragment": "results/dataprep/ledger/rows/P4m.csv",
                   "rows_added_this_round": pub["ledger_rows_added"],
                   "unledgered": pub["unledgered"], "conflicts": pub["conflicts"]},
        "claude_api_tokens": 0, "gpu": "none (CUDA hidden in every run)",
    }
    L.jdump(out, REPORTS_DIR / "P4m_numbers.json")
    print(REPORTS_DIR / "P4m_numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
