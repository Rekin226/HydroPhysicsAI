"""WP P4m repair round: publish ``changhua_forcing_diagnostics/preorigin/v<N>`` (superseding
v1), the post-look recommendation revision, ledger rows and ``P4m_evidence/repair/``.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_repair_publish

v1 is never edited. The new version carries v1's outputs unchanged except
``evidence_e8_be_e9.json``, whose after-the-fact comparison with the pilot tide table (hourly
data to 2026-09) moves out of the product into ``P4m_evidence/repair/`` so that the
model_design product no longer lists that pilot table as an input. It adds the repair-round
files of ``P4m.repair_audit.v1`` and ``recommendation_revision_postlook.json``. The hashed
pre-origin recommendation (sha256 ea01963d...) is carried unchanged.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from p4_changhua import p4m_finalize as F  # noqa: E402
from p4_changhua import p4m_lib as L  # noqa: E402
from p4_changhua import p4m_publish as PUBV1  # noqa: E402

V1 = DATAPREP / "changhua_forcing_diagnostics" / "preorigin" / "v1"
REP = L.STAGE / "repair"
EVREP = L.EVID / "repair"
REC_SHA = "ea01963d901e83abe7580bdbc0b666de280b82ed5bdf48db4ff58ccb15870ecd"
PRE_REPAIR = "results/dataprep/preregistration/P4m.repair_audit.v1.json"
CODE = F.CODE + ["dataprep_staging/p4_changhua/" + f for f in (
    "prereg_p4m_repair.py", "p4m_repair.py", "p4m_repair_publish.py")]
UNCHANGED = ("diagnostics_t1_t3.json", "well_arrays.npz", "scale_curves.npz",
             "theta_refit_t2b.json", "refit_predictions.npz", "e8_tides_preorigin.csv",
             "e9_nested_slopes.csv", "recommendation_preorigin.json")
# entries of the hashed expected-shift list that the v1 report already called unreliable
# (at or near a bound in the s = 0 refit)
BOUND_NEAR = ("log_S_mid[1]", "log_S_mid[2]", "pump_split", "log_C_coast[0]",
              "log_C_coast[2]", "log_L_distal[1]", "log_S_distal[0]")


def _row(criterion, threshold, measured, passed, note=""):
    return {"criterion": criterion, "threshold": threshold, "measured": measured,
            "passed": bool(passed), "note": note}


def revision(audit: dict, eng: dict, rec: dict) -> dict:
    re_, rf = eng["RE_gls"], eng["RF_identifiability"]
    shift = rec["recommendation"]["expected_theta_shift_from_cpu_emulation"]
    return {
        "wp": "P4m", "prereg": "P4m.repair_audit.v1",
        "label": "POST-LOOK revision (written in the repair round, after the 2023-25 recorded "
                 "arm of P4m v1 had been read). It does not replace the hashed pre-origin "
                 "recommendation (recommendation_preorigin.json, sha256 ea01963d...), which "
                 "stays the U7 record; it corrects it where the repair audit shows the "
                 "pre-origin evidence was misread. Every change is in the conservative "
                 "direction (less model freedom, weaker claims) and none uses a 2023+ number.",
        "base_recommendation_sha256": REC_SHA,
        "changes": [
            {"item": "Changhua-specific response multiplier",
             "hashed": "recommended (T5 fired: s_theta 0.48 < 0.67 and all three indicators "
                       "passed their fixed readings)",
             "revised": "NOT recommended. No independent indicator is both physically valid "
                        "and supporting (audit RD): BE pump-layer medians are negative in "
                        "both counties (impossible; valid-epoch fallback ratio "
                        f"{audit['RA_BE']['reading'].get('implied_Ss_CH_over_YL', float('nan')):.2f}"
                        ", not > 1.25); E9 Changhua pump-layer slopes have the wrong sign "
                        f"({audit['RB_RG_E9']['primary']['changhua']['n_obs_positive']} of "
                        f"{audit['RB_RG_E9']['primary']['changhua']['n']} positive); E8 is not "
                        "robust (no-easting ratio "
                        f"{audit['RC_E8']['b_without_easting']['ratio']:.2f}, pump-layer raw "
                        f"{audit['RC_E8']['c_pump_layer_raw_median_ratio']['ratio']:.2f}) and "
                        "its physical direction is opposite to damping. The declared T5 rule "
                        "fired mechanically; that result stays on record. No R1m rung is "
                        "recommended on P4m evidence; if I1 runs one, it must be justified "
                        "independently and cannot cite P4m's indicators.",
             "basis": "P4m.repair_audit.v1 RA-RD (pre-origin data only)"},
            {"item": "spin-up / burn-in",
             "hashed": "cites V4 ('the spin-up removes the mean-channel sign problem')",
             "revised": "keep the advice (start R1 from a spin-up with metered pumping or add a "
                        "burn-in to the loss), on the start-up transient evidence alone "
                        "(A1 -0.94 m, interval -1.36..-0.54; spin-up moves layer-1 Changhua IC "
                        "-6.25 m). The V4 citation is withdrawn: V1 already covers 0 under OLS, "
                        "and under the AR(1)-GLS secondary " + (
                            "theta_mean is below 0 in every variant including V4 "
                            "(RE: " + re_["reading"] + ")" if re_["all_variants_gls_below_0"]
                            else re_["reading"]),
             "basis": "v1 T3 and P4m.repair_audit.v1 RE"},
            {"item": "expected theta shift",
             "hashed": "every entry with |dtheta| > 0.05 from the unconverged CPU refits "
                       f"({len(shift)} entries)",
             "revised": "the hashed list stands as the pre-origin record, but it is "
                        "approximate (all three refits stopped at the 15-iteration cap) and "
                        "these entries were at or near a bound in the s = 0 refit, so their "
                        "shifts are bound-limited and should not be read as expected "
                        "directions: " + ", ".join(
                            f"{s['entry']} {s['dtheta']:+.2f}" for s in shift
                            if s["entry"] in BOUND_NEAR) + ". The 'reliable entries' filter in "
                        "the v1 report was added after the recorded-arm look.",
             "basis": "v1 T2b (theta_refit_t2b.json)"},
            {"item": "acceptance tests proposed for I1",
             "hashed": "P4m diagnostics on R1's theta: s_LS in [0.8, 1.25] and s_AM in "
                       "[0.67, 1.5]; multiplier rule re-applied if not",
             "revised": "keep s_LS in [0.8, 1.25]; keep s_AM in [0.67, 1.5] only as a "
                        "consistency check, not as evidence that the magnitude is right "
                        "(RF: " + rf["reading"] + "); do not re-apply the multiplier rule with "
                        "the v1 indicator readings; re-apply it only with indicators that pass "
                        "RA-RC's validity rules",
             "basis": "P4m.repair_audit.v1 RF, RD"},
            {"item": "E10 rerun on R1",
             "hashed": "theta_mean not significantly negative in at least one variant",
             "revised": "report both the OLS primary and the AR(1)-GLS secondary with "
                        "cluster-bootstrap intervals; the test passes only if theta_mean is not "
                        "significantly negative under both estimators in at least one variant",
             "basis": "P4m.repair_audit.v1 RE"},
        ],
        "unchanged": ["forcing: C0 preorigin/v3 Changhua + in-grid Chiayi, all classes primary, "
                      "no-domestic as the declared sensitivity member (choice declared before "
                      "R1's scores)",
                      "free parameters: the R0 recipe's zonal parameters, as "
                      "I1.R1.changhua_census.v1 declares",
                      "widened-bound sensitivity rung for log_T_distal[0] (upper), "
                      "log_L_distal[0], log_L_distal[2], log_spread_km[0] (lower)",
                      "never gating: the 2023-25 recorded arm, the 2023-26 frozen challenge, "
                      "subsidence"],
    }


def acceptance(v1val: dict, audit: dict, eng: dict, rev_path: Path) -> list[dict]:
    rows = [dict(r, note=(r.get("note", "") + " [v1 row, unchanged]").strip())
            for r in v1val["acceptance"]]
    e9r = audit["RB_RG_E9"]["replication_v1"]
    rows.append(_row("Repair RG replication: v1 E9 definition with the installed rain "
                     "reproduces v1's e9_nested_slopes.csv on every common row",
                     "max |db| <= 1e-9", f"b_obs {e9r['max_abs_diff_b_obs']:.1e}, b_mod "
                     f"{e9r['max_abs_diff_b_mod']:.1e} ({e9r['n_merged']} rows)",
                     max(e9r["max_abs_diff_b_obs"], e9r["max_abs_diff_b_mod"]) <= 1e-9))
    rep = eng["replication_P_A1_max_abs_m"]
    rows.append(_row("Repair engine replication: A1 heads equal v1 well_arrays P_A1",
                     "max |dh| <= 1e-9 m", f"{rep:.1e} m", rep <= 1e-9))
    worst_ols = max(max(v["replication"][k] for k in v["replication"] if k.startswith("ols"))
                    for v in eng["RE_gls"]["variants"].values())
    rows.append(_row("Repair RE replication: OLS theta_mean and its interval equal v1 (V1-V4)",
                     "abs diff <= 1e-9", f"{worst_ols:.1e}", worst_ols <= 1e-9))
    g = [v["replication"]["gls_mean_v1_value_abs_diff_v1fun"]
         for v in eng["RE_gls"]["variants"].values()
         if "gls_mean_v1_value_abs_diff_v1fun" in v["replication"]]
    gb = [v["replication"]["gls_blocks_vs_v1fun_abs_diff"]
          for v in eng["RE_gls"]["variants"].values()
          if "gls_blocks_vs_v1fun_abs_diff" in v["replication"]]
    rows.append(_row("Repair RE replication: GLS point estimates equal v1 (V1, V2, V4); the "
                     "bootstrap's normal-equation estimator equals v1's GLS function",
                     "abs diff <= 1e-9", f"v1 values {max(g):.1e}; estimator {max(gb):.1e}",
                     max(g) <= 1e-9 and max(gb) <= 1e-9))
    rf = eng["RF_identifiability"]["replication"]
    rows.append(_row("Repair RF replication: the metered series reproduces v1's s_LS_all and "
                     "s_AM", "abs diff <= 1e-6", f"s_LS {rf['s_LS_all_abs_diff']:.1e}, s_AM "
                     f"{rf['s_AM_abs_diff']:.1e}",
                     max(rf["s_LS_all_abs_diff"], rf["s_AM_abs_diff"]) <= 1e-6))
    rows.append(_row("Repair RA-RD computed (indicator validity audit, T5 audit reading)",
                     "all computed", "valid supporting indicators: "
                     + (", ".join(audit["RD_T5_audit"]["valid_supporting_indicators"]) or "none")
                     + "; multiplier supported: "
                     + str(audit["RD_T5_audit"]["multiplier_supported"]), True,
                     "post-look audit reading; the declared T5 result stays on record"))
    rows.append(_row("Repair RE computed (GLS bootstrap intervals V1-V4)", "all computed",
                     eng["RE_gls"]["reading"], True))
    rows.append(_row("Repair RF computed (s_AM identifiability, LAG+2, LAG-2, PERM, NOSEAS)",
                     "all computed", eng["RF_identifiability"]["reading"], True))
    rows.append(_row("Hashed pre-origin recommendation carried unchanged", "sha256 ea01963d...",
                     sha256_file(V1 / "recommendation_preorigin.json")[:12] + "...",
                     sha256_file(V1 / "recommendation_preorigin.json") == REC_SHA))
    rows.append(_row("Post-look recommendation revision written", "file present",
                     rev_path.name, rev_path.exists()))
    return rows


def card(audit: dict, eng: dict) -> str:
    rd = audit["RD_T5_audit"]
    return f"""# changhua_forcing_diagnostics/preorigin v2 (WP P4m, repair round)

Supersedes v1 (2026-10-17). Same pre-origin, in-sample design-time evidence as v1 (frozen cwa
theta sha256 789d0e45..., C0 preorigin/v3 metered Changhua + in-grid Chiayi forcing; declaration
`P4m.metered_diagnostics.v1`), plus the repair round declared in `P4m.repair_audit.v1`.
Not skill. Allowed uses: model_design, design_time_evidence, after_the_fact_analysis. Two
exemptions inherited from C0 preorigin/v3, as in v1.

What changed from v1:
- `evidence_e8_be_e9.json`: the after-the-fact comparison with the pilot tide table (hourly
  data to 2026-09) was removed; it now sits in `results/dataprep/reports/P4m_evidence/repair/`
  and the pilot table is no longer an input of this product. All other v1 files are
  byte-identical.
- `repair_indicator_audit.json` (RA-RD): BE pump-layer medians are negative in both counties
  (physically invalid); E9 Changhua pump-layer slopes have the wrong sign
  ({audit['RB_RG_E9']['primary']['changhua']['n_obs_positive']} of
  {audit['RB_RG_E9']['primary']['changhua']['n']} positive); E8 is not robust and its physical
  direction is opposite to damping. Valid supporting indicators:
  {', '.join(rd['valid_supporting_indicators']) or 'none'}; multiplier supported:
  {rd['multiplier_supported']}.
- `repair_gls_identifiability.json`: RE ({eng['RE_gls']['reading']}); RF
  ({eng['RF_identifiability']['reading']}).
- `e9_repair_*.csv`: E9 per-well slopes for the v1 and as-declared (site_id, station layer
  codes 1-2) site definitions, each with the installed CWA-rebuilt rain and the vendor cache.
- `recommendation_revision_postlook.json`: POST-LOOK corrections to the hashed pre-origin
  recommendation (`recommendation_preorigin.json`, sha256 ea01963d..., carried unchanged).

Intervals are cluster bootstraps (10 km), UNCALIBRATED.
"""


def main() -> int:
    audit = json.loads((REP / "audit.json").read_text())
    eng = json.loads((REP / "engine.json").read_text())
    v1val = json.loads((V1 / "validation.json").read_text())
    rec = json.loads((V1 / "recommendation_preorigin.json").read_text())
    if sha256_file(V1 / "recommendation_preorigin.json") != REC_SHA:
        raise SystemExit("hashed recommendation changed")
    EVREP.mkdir(parents=True, exist_ok=True)
    d = manifest.new_version_dir("changhua_forcing_diagnostics/preorigin")
    for f in UNCHANGED:
        shutil.copy2(V1 / f, d / f)
    ev = json.loads((V1 / "evidence_e8_be_e9.json").read_text())
    atf = ev["E8"].pop("after_the_fact_vs_pilot")
    ev["v2_note"] = ("E8.after_the_fact_vs_pilot (comparison with the pilot tide table, hourly "
                     "data to 2026-09) moved to results/dataprep/reports/P4m_evidence/repair/"
                     "after_the_fact_pilot_tides.json; everything else is v1's file unchanged. "
                     "The BE ratios and the E9 |obs|/|model| ratio below are the v1 readings; "
                     "repair_indicator_audit.json shows they are physically invalid or "
                     "sign-reversed")
    L.jdump(ev, d / "evidence_e8_be_e9.json")
    L.jdump({"wp": "P4m", "use": "after_the_fact_analysis only (pilot window to 2026-09)",
             "moved_from": "changhua_forcing_diagnostics/preorigin/v1/evidence_e8_be_e9.json "
                           "E8.after_the_fact_vs_pilot", **atf},
            EVREP / "after_the_fact_pilot_tides.json")
    shutil.copy2(REP / "audit.json", d / "repair_indicator_audit.json")
    shutil.copy2(REP / "engine.json", d / "repair_gls_identifiability.json")
    for p in sorted(REP.glob("e9_*.csv")):
        shutil.copy2(p, d / p.name.replace("e9_", "e9_repair_", 1))
    rev = revision(audit, eng, rec)
    rev_path = d / "recommendation_revision_postlook.json"
    L.jdump(rev, rev_path)
    acc = acceptance(v1val, audit, eng, rev_path)
    val = {"wp": "P4m", "prereg": [L.PRE_DIAG, "P4m.repair_audit.v1"], "acceptance": acc,
           "gate_passed": all(a["passed"] for a in acc
                              if not a["criterion"].startswith("T2b replication")),
           "gate_note": "v1's rows plus the repair round's replication and computation rows; "
                        "the T2b replication row is reported, not gating (as in v1)"}
    L.jdump(val, d / "validation.json")
    (d / "PRODUCT_CARD.md").write_text(card(audit, eng))
    ex = F._c0_exemptions()
    inputs = (F.MODEL_INPUTS + F.ENGINE_IN + F.C0_PRE_IN + F.H1A_IN
              + ["results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet",
                 "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv",
                 "results/dataprep/changhua_forcing_diagnostics/preorigin/v1/MANIFEST.json"]
              + F.PREREGS[:1] + [PRE_REPAIR])
    m = manifest.build_manifest(
        d, product="changhua_forcing_diagnostics/preorigin", layer="L3", inputs=inputs,
        input_trees=["AMP_V2/data/wells"], code=CODE,
        parameters={"theta": "stage3_datum_gate_cwa (batch-8 merged cwa)",
                    "steps": "2012-02..2022-12", "scored": "2013-01..2022-12",
                    "min_months": L.MIN_N, "cluster_km": L.CLUSTER_KM, "seed": L.SEED,
                    "n_boot": L.N_BOOT, "scale_grid": list(L.S_GRID),
                    "refit": {"levels": [0, 0.5, 1], "n_iter": 15, "trust": 0.5, "tol": 1e-6},
                    "repair": "P4m.repair_audit.v1 RA-RG"},
        fit_window_end=max(["2022-12"] + [x["data_end"] for x in ex]),
        non_exempt_fit_window_end="2022-12", exemptions=ex, validation_report="validation.json",
        gate={"passed": bool(val["gate_passed"]),
              "criteria": "P4m.metered_diagnostics.v1 engine checks EC1-EC3 and every declared "
                          "diagnostic computed (v1 rows) plus the P4m.repair_audit.v1 "
                          "replication checks"},
        allowed_uses=F.PRE_ALLOWED, forbidden_uses=F._forbidden(F.PRE_ALLOWED),
        personal_data="none: public monitoring-well ids and cell aggregates only (no census "
                      "well numbers, pump coordinates, names or addresses)",
        extra={"preregistration": [F.PREREGS[0], PRE_REPAIR], "wp": "P4m",
               "supersedes": "changhua_forcing_diagnostics/preorigin/v1",
               "provisional": "no: built on batch 8's merged cwa theta (engine/v2)",
               "rain_note": "chou-shui-data/data/rf_timeseries.csv (sha256 42308c8d...) is the "
                            "CWA CODIS rebuild installed 2026-10-11; it holds days to "
                            "2026-10 but every P4m reader reindexes it to 2012-01..2022-12 "
                            "(model recharge and E9 rain covariate, exogenous forcing)"})
    manifest.write_manifest(d, m)
    # superseded record
    sup = {"schema": "dataprep-superseded/1", "product": "changhua_forcing_diagnostics/preorigin",
           "written": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": "dataprep_staging/p4_changhua/p4m_repair_publish.py",
           "current": d.name,
           "superseded": [{"version": "v1", "superseded_by": d.name,
                           "why": "verifier 2026-10-17: the T5 indicators were read without "
                                  "validity checks (negative BE, sign-reversed E9, non-robust "
                                  "E8) and the product listed the post-origin pilot tide table "
                                  "as an input without an exemption; v1's numbers are carried "
                                  "unchanged into v2, which adds the repair audit",
                           "outputs_byte_identical_in_current": list(UNCHANGED),
                           "outputs_differing": ["evidence_e8_be_e9.json", "validation.json",
                                                 "PRODUCT_CARD.md"]}]}
    L.jdump(sup, d.parent / "SUPERSEDED.json")
    # ledger
    pv = rel_to_repo(d).replace("results/dataprep/", "")
    r = PUBV1.load()
    pre_n, _ = PUBV1.numbers(r)
    rows = F.ledger_rows(pre_n, pv)
    heads = "AMP_V2/data/wells"
    h1a = "results/dataprep/heads/L1_clean/v3"
    new_ids = []

    def add(nid, srcs):
        new_ids.append(nid)
        for s, role in srcs.items():
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": "repair round (P4m.repair_audit.v1)"})

    for nid, role in (("P4m.E8", "fit_target"), ("P4m.BE", "design_time_evidence")):
        add(nid, {heads if nid == "P4m.E8" else h1a: role})
    add("P4m.E9", {h1a: "fit_target",
                   "results/dataprep/pump_electricity/preorigin/v3": "input",
                   "chou-shui-data/data/rf_timeseries.csv": "input",
                   "chou-shui-data/data/rf_stations.csv": "input"})
    add("P4m.repair.RA_BE", {h1a: "design_time_evidence"})
    add("P4m.repair.RC_E8", {heads: "fit_target"})
    add("P4m.repair.RG_E9", {h1a: "fit_target",
                             "results/dataprep/pump_electricity/preorigin/v3": "input",
                             "chou-shui-data/data/rf_timeseries.csv": "input",
                             "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv": "input",
                             "chou-shui-data/data/rf_stations.csv": "input"})
    add("P4m.repair.RD_T5_audit", {h1a: "design_time_evidence", heads: "design_time_evidence"})
    model_srcs = {s: "input" for s in F.MODEL_INPUTS}
    model_srcs["results/dataprep/changhua_abstraction/engine/v2"] = "input"
    model_srcs["results/dataprep/pump_electricity/preorigin/v3"] = "input"
    for v in eng["RE_gls"]["variants"]:
        add(f"P4m.repair.RE_gls.{v}", {**model_srcs, heads: "fit_target"})
    for v in eng["RF_identifiability"]["variants"]:
        add(f"P4m.repair.RF.{v}", {**model_srcs, heads: "fit_target"})
    add("P4m.repair.recommendation_revision_postlook", {heads: "design_time_evidence"})
    have = {(x["number_id"], x["source"], x["product_version"])
            for x in ledger._read(ledger.fragment_path("P4m"))}
    new = [x for x in rows if (x["number_id"], x["source"], x["product_version"]) not in have]
    ledger.append_rows("P4m", new)
    unl = ledger.unledgered([n for n, _ in pre_n] + new_ids)
    conf = [c for c in ledger.conflicts(ledger.read_all()) if c[0].startswith("P4m")]
    out = {"preorigin_v2": rel_to_repo(d), "ledger_rows_added": len(new), "unledgered": unl,
           "conflicts": conf, "acceptance": acc}
    L.jdump(out, EVREP / "publish_repair.json")
    print(d, len(new), "rows; unledgered", unl, "conflicts", conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
