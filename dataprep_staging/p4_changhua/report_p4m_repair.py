"""WP P4m repair round: rewrite ``results/dataprep/reports/P4m.json`` (the pre-repair copy is
kept in ``P4m_evidence/repair/P4m_pre_repair.json``).

    cd dataprep_staging
    python -m p4_changhua.report_p4m_repair
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPORTS_DIR  # noqa: E402

from p4_changhua import p4m_lib as L  # noqa: E402


def main() -> int:
    old = json.loads((L.EVID / "repair" / "P4m_pre_repair.json").read_text())
    pub = json.loads((L.EVID / "repair" / "publish_repair.json").read_text())
    audit = json.loads((L.STAGE / "repair" / "audit.json").read_text())
    eng = json.loads((L.STAGE / "repair" / "engine.json").read_text())
    new = dict(old)
    new["date"] = "2026-10-17"
    new["status"] = "passed"
    new["repair_round"] = {
        "preregistration": "results/dataprep/preregistration/P4m.repair_audit.v1.json "
                           "(sha256 451d3c45da012bf40de37a27cc84f1f8431c8c9719960ce25b097f101fd04b83, "
                           "2026-10-17T02:32:41Z, before the first repair look 02:34:57Z)",
        "current_product": pub["preorigin_v2"],
        "superseded": "results/dataprep/changhua_forcing_diagnostics/preorigin/v1",
        "major_1_T5_indicators": {"fixed": True, "audit": audit["RD_T5_audit"],
                                  "BE": audit["RA_BE"]["reading"],
                                  "E8": {k: audit["RC_E8"][k] for k in (
                                      "a_declared_with_easting", "b_without_easting",
                                      "c_pump_layer_raw_median_ratio", "robust", "reading")},
                                  "E9_primary_changhua": audit["RB_RG_E9"]["primary"][
                                      "changhua"]},
        "major_2_item2": {"fixed": True, "RF": {k: {kk: v[kk] for kk in (
            "s_LS_all", "s_LS_changhua", "s_AM_changhua")} | {
            "changhua_corr_s1": v["changhua_at_s1"]["corr_median"],
            "changhua_rmse_s1": v["changhua_at_s1"]["anomaly_rmse"]}
            for k, v in eng["RF_identifiability"]["variants"].items()},
            "reading": eng["RF_identifiability"]["reading"]},
        "minor_gls": {k: v["gls"]["mean"] for k, v in eng["RE_gls"]["variants"].items()}
        | {"reading": eng["RE_gls"]["reading"]},
        "acceptance_repair": [a for a in pub["acceptance"] if a["criterion"].startswith(
            ("Repair", "Hashed", "Post-look"))],
        "ledger_rows_added": pub["ledger_rows_added"], "unledgered": pub["unledgered"],
        "conflicts": pub["conflicts"]}
    new["preregistration"] = old["preregistration"] + [
        "results/dataprep/preregistration/P4m.repair_audit.v1.json"]
    new["recommendation"] = (old["recommendation"] + "; POST-LOOK revision "
                             + pub["preorigin_v2"] + "/recommendation_revision_postlook.json "
                             "(no Changhua multiplier, no R1m; 24-month burn-in; s_AM a "
                             "consistency check only; E10 under OLS and GLS)")
    new["summary"] = (
        "See P4m.md (repair round included). Frozen cwa theta + C0 metered Changhua/Chiayi "
        "forcing (187.99 GWh/yr): in-sample Changhua anomaly RMSE 1.191->1.244 (interval "
        "covers 0), amplitude 0.38->1.02; Yunlin +1.5 %. s_LS 0.29 (attenuated), s_AM 0.97, but "
        "s_AM is not identifying (1.02-1.15 under 2-month shifts and cell permutation), so the "
        "magnitude is consistent with, not shown by, the heads; theta bias about a third of the "
        "LS gap (unconverged refits), remainder response/conversion unresolved. T3: transient "
        "appears (-0.94 m); OLS mean channel covers 0 in V1/V2/V4 but GLS excludes 0 in V1/V3/V4 "
        "(fragile; 24-month burn-in recommended). T5 fired mechanically but no indicator is "
        "valid and supporting (BE negative, E9 sign-reversed, E8 not robust): no Changhua "
        "multiplier and no R1m (post-look revision). Recorded arm 2023-25 (reported only): "
        "Changhua +20.5 %, Yunlin +2.6 %.")
    new["products_current"] = [pub["preorigin_v2"],
                               "results/dataprep/changhua_forcing_diagnostics/recorded_arm/v1"]
    (REPORTS_DIR / "P4m.json").write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
    print(REPORTS_DIR / "P4m.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
