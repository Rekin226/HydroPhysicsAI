"""Version records of the WP H2 products after repair round 3 (W3/H1a ``SUPERSEDED.json``).

Rewrites ``SUPERSEDED.json`` at the root of heads/L2_filled_preorigin and heads/L2_filled_full
(the version directories themselves are never touched). The previous record is embedded with
its sha256. Adds the errata that the independent verification established for the published
versions (E-family freshness; the v2 card's fill-only limits; the IC baseline) and a map from
version numbers to configurations, because the numbering is easy to misread (product v2 is the
v1 configuration's bridge; product v3 is the 'v2' repair configuration).

Usage: python -I s17_versions_r3.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402

ROOT = C.REPO / "results/dataprep"


def versions(product):
    return sorted(p.name for p in (ROOT / product).glob("v*") if (p / "MANIFEST.json").exists())


def release_of(product, v):
    m = json.loads((ROOT / product / v / "MANIFEST.json").read_text())
    return m["parameters"].get("release"), m["allowed_uses"], m["gate"]["passed"]


def write(product: str, doc: dict) -> Path:
    p = ROOT / product / "SUPERSEDED.json"
    prev = None
    if p.exists():
        raw = p.read_bytes()
        prev = {"sha256": hashlib.sha256(raw).hexdigest(), "record": json.loads(raw)}
    doc = {"schema": "dataprep-superseded/1", "product": product,
           "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": "dataprep_staging/h2_heads_fill/s17_versions_r3.py"} | doc
    doc["previous_record"] = prev
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return p


def main():
    ef = json.loads((R3.EV_R3 / "e_freshness.json").read_text())
    v1c = ef["rounds"]["v1"]["pilot_also_excluded"]
    v2c = ef["rounds"]["v2"]["pilot_also_excluded"]
    ic = json.loads((R3.REG_R3 / f"{R3.TEST_IC}.json").read_text())["decision"]
    d3 = json.loads((R3.REG_R3 / f"{R3.TEST_D3}.json").read_text())["pass"]
    pre = versions(C.PRODUCT_PRE)
    full = versions(C.PRODUCT_FULL)
    cur_pre = pre[-1]
    rel_pre, uses_pre, _ = release_of(C.PRODUCT_PRE, cur_pre)
    if not rel_pre.startswith("r3_bridge_pre"):
        raise SystemExit(f"the newest pre-origin version {cur_pre} is not the round-3 release")
    def e(r, s):
        return (f"{r[s]['coverage']:.3f} [{r[s]['lo']:.3f}, {r[s]['hi']:.3f}], k {r[s]['k']}, "
                f"{'pass' if r[s]['pass'] else 'FAIL'}")

    write(C.PRODUCT_PRE, {
        "current": cur_pre,
        "current_note": f"{cur_pre} is the version for use: the observations (flags 0/1) and "
                        "the daily-bridged months (flag 2) for calibration_target (v1 packaging "
                        "rule, gate H2.fresh.d; replicated by H2.fresh3.d: "
                        f"{'passed' if d3 else 'FAILED'})"
                        + (", and the January 2012 gp_bf field for initial_conditions "
                           "(H2.ic.twin confirmed)" if "initial_conditions" in uses_pre else
                           f"; no initial-condition field (H2.ic.twin '{ic['outcome']}')")
                        + ". No complete fill passed its gate (v1, v3): the guard refuses "
                          "idw_field, and initial_conditions on the fills.",
        "configurations": {
            "v1": "registered v1 configuration (H2.fill_acceptance.v1), complete fill",
            "v2": "daily-bridge component of the v1 configuration (v1 packaging rule)",
            "v3": "repair configuration 'v2' (H2.fill_acceptance.v2 sd factors), complete fill",
            cur_pre: "daily-bridge component of the v1 configuration (identical rows to v2 "
                     "apart from product_version), corrected card; repair round 3 "
                     "(H2.repair3.v1)"},
        "superseded": [
            {"version": "v1", "superseded_by": "v3",
             "why": "registered v1 configuration: gate failed (coverage N_net1 0.812, S24 "
                    "0.939, E_lead66 0.844, E_trail48 0.960 as registered).",
             "erratum_repair3": "the registered E gate did not exclude the pilot's E r0-1 "
                                "well-gaps (same cells, design-time units). With them "
                                f"excluded: E_lead66 {e(v1c, 'E_lead66')}; E_trail48 "
                                f"{e(v1c, 'E_trail48')}. v1 then fails 2 strata (N_net1, S24), "
                                "not 4; the gate verdict (failed) is unchanged.",
             "allowed_uses": ["design_time_evidence", "after_the_fact_analysis"]},
            {"version": "v2", "superseded_by": cur_pre,
             "why": f"same rows as {cur_pre}; v2's product card lists fill-only limits (edge "
                    "fills) that do not apply to a bridge-only product and omits the "
                    "sigma_obs_m advice for observed_partial rows, and its manifest also "
                    "allows after_the_fact_analysis beyond the packaging rule's "
                    "'calibration_target only' (harmless). Data and gate are unchanged.",
             "allowed_uses": ["calibration_target", "after_the_fact_analysis"]}],
        "not_superseded": [
            {"version": "v3", "role": "complete fill of the repair configuration "
             "(H2.fill_acceptance.v2): gate failed (as registered: S01 coverage 0.983, N_net2 "
             "RMSE 0.783 vs clim+AR(1) 0.764; 12/14 strata pass)",
             "erratum_repair3": "with the pilot's E r0-1 well-gaps also excluded from the E "
                                f"gate, E_lead66 {e(v2c, 'E_lead66')} (point window 0.85-0.95 "
                                f"applies) and E_trail48 {e(v2c, 'E_trail48')}: 11/14 strata "
                                "pass, not 12/14. Its card's and validation.json's 12/14 and "
                                "the claim that its failing strata touch almost no real cells "
                                "are superseded by this record and reports/H2.md: leading-edge "
                                "fills are 1,321 of the 1,481 router-filled model-well cells.",
             "allowed_uses": ["design_time_evidence", "after_the_fact_analysis"]}],
        "ic_baseline_erratum_repair3": "H2.ic.lowo (v1/v3 ic_field_2012-01.npz) compared the GP "
                                       "field with plain per-layer IDW (13.19 m), not with the "
                                       "twin's stage-3 construction (backfill + merged "
                                       "proximal, about 9.3 m on the same wells). See "
                                       "H2_evidence/repair3/ and H2.ic.twin."})
    full_doc = {"configurations": {
        "v1": "repair configuration 'v2' fitted on 2012-2026, complete fill (gate failed)"}}
    if d3 and len(full) >= 2:
        cur_full = full[-1]
        rel_full, uses_full, _ = release_of(C.PRODUCT_FULL, cur_full)
        if rel_full != "r3_bridge_full":
            raise SystemExit(f"the newest full version {cur_full} is not the round-3 release")
        full_doc.update({
            "current": cur_full,
            "current_note": f"{cur_full} is the version for the statistical layer's history: "
                            "observations and daily-bridged months (Feb-Mar 2026 bridged), "
                            "released by the H2.repair3.v1 packaging rule (gate H2.fresh.d and "
                            "H2.fresh3.d); uses statistical_layer_history and "
                            "after_the_fact_analysis only.",
            "superseded": [],
            "not_superseded": [{"version": "v1", "role": "complete fill of the repair "
                                "configuration, full variant; gate failed",
                                "allowed_uses": ["after_the_fact_analysis"]}]})
        full_doc["configurations"][cur_full] = ("daily-bridge component of the v1 "
                                                "configuration, full variant (H2.repair3.v1)")
    else:
        full_doc.update({"current": "v1", "current_note": "no full-variant release passed: "
                         "H2.fresh3.d failed or the release was not built; after-the-fact "
                         "analysis only", "superseded": []})
    write(C.PRODUCT_FULL, full_doc)
    print("written", cur_pre, full_doc["current"])


if __name__ == "__main__":
    main()
