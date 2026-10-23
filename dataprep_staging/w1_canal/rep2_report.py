"""W1 report after repair round 2 (reports/W1.md, W1.json).

    python -I dataprep_staging/w1_canal/rep2_report.py STAGE_DIR

Keeps the repair-round-1 report as reports/W1.r3.md / W1.r3.json (refuses to overwrite), and
writes a new W1.md / W1.json with a "Repair round 2" section, the corrected acceptance table,
the new products (preorigin/v2, full/v4) and updated issues. Numbers come from the staged
rep2_state.json, the new validation.json files and W1_evidence/rep2_verify.json. Appends the
ledger rows of the verification numbers.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402

REPORTS = DATAPREP / "reports"
EVID = REPORTS / "W1_evidence"
SW = DATAPREP / C.PRODUCT_BASE
PRE2, FULL4 = SW / "preorigin" / "v2", SW / "full" / "v4"
REP2_CODE = ["rep2_config.py", "rep2_build.py", "rep2_publish.py", "rep2_verify.py",
             "rep2_report.py"]


def f3(x):
    return f"{x:.3f}"


def main(stage: str):
    stage = Path(stage)
    # the repair-round-1 report is kept as W1.r3.*; a rerun rebuilds W1.md/json from it
    if not (REPORTS / "W1.r3.md").exists():
        shutil.copy2(REPORTS / "W1.md", REPORTS / "W1.r3.md")
        shutil.copy2(REPORTS / "W1.json", REPORTS / "W1.r3.json")
    old_md = (REPORTS / "W1.r3.md").read_text()
    old = json.loads((REPORTS / "W1.r3.json").read_text())
    st = json.loads((stage / "rep2_state.json").read_text())
    shutil.copy2(stage / "rep2_state.json", EVID / "rep2_state.json")
    ver = json.loads((EVID / "rep2_verify.json").read_text())
    vpre = json.loads((PRE2 / "validation.json").read_text())
    vfull = json.loads((FULL4 / "validation.json").read_text())
    rp, rf = vpre["repair_r2"], vfull["repair_r2"]
    diag = rf["residual_diagnostics_post_hoc"]
    yl, ch = diag["yunlin"], diag["changhua"]
    scan = ver["label_scan"]
    ident = ver["identity"]
    gp = ver["guard_probes"]
    pq = st["preorigin"]["parse_qc_2012_2022"]
    pid = st["preorigin"]["parse_identity"]
    chk = ver["checker"]
    alt = ver["forward_archive_alternative"]
    probes = ver["forward_archive_probes"]

    def lab(k):
        s = scan[k]
        return f"{s['labelled_values']}/{s['interval_values']}"

    def opened(name, use, idx=None):
        rows = [g for g in gp[name] if g["use"] == use]
        g = rows[idx or 0]
        return g.get("opened") and (g.get("action") or {}).get("ok", True)

    pre_ok = (opened("preorigin/v2", "forcing_frozen") and opened("preorigin/v2", "prior")
              and not opened("preorigin/v2", "prior", 1)
              and not opened("preorigin/v2", "prior_width_input")
              and not opened("preorigin/v2", "forcing_recorded_arm")
              and not opened("preorigin/v2", "scoring_target"))
    full_ok = (opened("full/v4", "forcing_recorded_arm")
               and not any(opened("full/v4", u) for u in (
                   "forcing_frozen", "model_design", "prior", "prior_width_input", "selection",
                   "scoring_target", "calibration_target")))
    ok_checker = all((v.get("status") if isinstance(v, dict) else v) == "ok"
                     for v in chk.values())
    id_pre, id_full = ident["preorigin_v2_vs_v1"], ident["full_v4_vs_v3"]
    id_all = (id_pre["values_identical"] and id_pre["flags_identical"]
              and id_pre["sd_identical_outside_structural"] and id_pre["sd_structural_max"] == 0
              and all(id_pre["district_draws.npz"].values())
              and all(v for k, v in id_pre["grid_sw_v4.npz"].items() if isinstance(v, bool))
              and id_full["values_identical"] and id_full["flags_identical"]
              and id_full["sd_identical_all"] and all(id_full["district_draws.npz"].values())
              and all(v for v in id_full["grid_sw_v4.npz"].values() if isinstance(v, bool))
              and id_pre["composite_value_sd_flag_identical"]
              and id_full["composite_value_sd_flag_identical"])
    archive_codes = ", ".join(f"{p['url'].split('/')[2]} {p['http_code']}" for p in probes)

    # ------------------------------------------------------------------ ledger
    pv_pre = f"{C.PRODUCT_BASE}/preorigin/v2"
    pv_full = f"{C.PRODUCT_BASE}/full/v4"
    rows = []
    for n, srcs in (("W1.rep2.label_scan", [SW / "preorigin" / "v1", PRE2, SW / "full" / "v3",
                                            FULL4]),
                    ("W1.rep2.identity", [SW / "preorigin" / "v1", PRE2, SW / "full" / "v3",
                                          FULL4]),
                    ("W1.rep2.guard_probes", [PRE2, FULL4])):
        for s in srcs:
            rows.append({"number_id": n, "source": rel_to_repo(s), "role": "validator",
                         "product_version": pv_full if "full" in str(s) else pv_pre,
                         "note": "repair round 2 verification (rep2_verify.py)"})
    ledger.append_rows(C.WP, rows)
    with open(ledger.fragment_path(C.WP), encoding="utf-8") as fh:
        n_ledger = sum(1 for _ in fh) - 1
    # ------------------------------------------------------------------ acceptance
    acc = [dict(a) for a in old["acceptance"]]
    for a in acc:
        if a["criterion"].startswith("REPAIR (verifier major issue 1; plan 4.7)"):
            a["measured"] = ("WRONG IN SCOPE (verifier round 2): only district_monthly.csv and "
                             "inflow_predicted_monthly.csv were labelled; full/v3 labels "
                             f"{lab('full/v3')} interval values (the composite, rain-index, "
                             "percolation, grid and draws intervals had no label); "
                             "superseded by the repair-round-2 row below")
            a["passed"] = False
    acc += [
        {"criterion": "REPAIR 2 (verifier major issue 1; plan 4.7): every interval-bearing "
                      "column or array (sd, percentiles, members) of every output of "
                      "preorigin/v2 and full/v4 carries a label (row column, per-month npz "
                      "array, or the manifest's interval_labels inventory)",
         "threshold": "0 unlabelled interval values",
         "measured": f"preorigin/v2 {lab('preorigin/v2')}, full/v4 {lab('full/v4')} "
                     f"interval values labelled, 0 unlabelled columns (before: preorigin/v1 "
                     f"{lab('preorigin/v1')}, full/v3 {lab('full/v3')})",
         "passed": scan["preorigin/v2"]["n_unlabelled_columns"] == 0
         and scan["full/v4"]["n_unlabelled_columns"] == 0},
        {"criterion": "REPAIR 2: labels say what was checked: composite sd, rain-index sd and "
                      "monthly district intervals 'uncalibrated'; 2026 intervals 'uncalibrated "
                      "at every published scale'; Jan-May aggregate of the 2012-2022/2025 draws "
                      "'passes only weakly' with hits, n_eff and the side of every miss",
         "threshold": "label text follows the evidence",
         "measured": f"composite rel sd {rp['composite_sd_check']['rel_sd_flag_1_4']:.3f} "
                     f"(pre-origin) / {rf['composite_sd_check']['rel_sd_flag_1_4']:.3f} (full), "
                     f"x2 on donor days; Jan-May pre-origin "
                     f"{rp['coverage_hist_statespace_janmay']['hits']}/"
                     f"{rp['coverage_hist_statespace_janmay']['n']}, full "
                     f"{rf['coverage_hist_statespace_janmay']['hits']}/"
                     f"{rf['coverage_hist_statespace_janmay']['n']}, all misses "
                     "over-predictions; 2026 label leads with 'uncalibrated'",
         "passed": True},
        {"criterion": "REPAIR 2: values, flags, members and grid values identical to "
                      "preorigin/v1 and full/v3; sd changed only in the pre-origin "
                      "structural-zero months (to 0; the old value kept as "
                      "structural_alt_value)",
         "threshold": "all identity checks true",
         "measured": f"all identity checks {'true' if id_all else 'NOT all true'} (table "
                     "values and flags, members, grid values and fields, composite value/sd/"
                     "flag; sd differs only at pre-origin structural zeros, now 0); pre-origin "
                     "grid sd outside structural months max "
                     f"|diff| {id_pre['grid_sw_v4.npz']['sd_identical_outside_structural_max_abs']:.1e}",
         "passed": bool(id_all)},
        {"criterion": "REPAIR 2 (verifier minor): pre-origin validation.json carries no "
                      "post-origin parse QC; parse QC recomputed from the 2012-2022 yearbooks "
                      "(L0) and identical to preorigin/v1's daily table",
         "threshold": "2012-2022 only; 0 differing station-days",
         "measured": f"{pq['station_years']} station-years; {pq['printed_monthly_off_gt_tol']}/"
                     f"{pq['printed_monthly_compared']} printed monthly means off > 5 %; "
                     f"{pid['matched']}/{pid['l1_flag0_station_days']} station-days matched, "
                     f"max |diff| {pid['max_abs_diff']}; residual: QC_REPORT.md (copied from "
                     "v1) still quotes v1's '30 of 1471 station-months (all years 2012-2025)' "
                     "line, sourced from v1's validation.json (listed as an input; reporting "
                     "only)",
         "passed": pid["max_abs_diff"] == 0 and pid["matched"] == pid["l1_flag0_station_days"]},
        {"criterion": "REPAIR 2: loader guard - preorigin/v2 refuses prior_width_input (no "
                      "coverage-checked sd at a prior-width scale), forcing_recorded_arm, "
                      "scoring_target, calibration_target, baseline; prior and selection read "
                      "the district table but not the LUIMAP grid; full/v4 opens only the "
                      "recorded arm, history and after-the-fact uses",
         "threshold": "every probe as declared",
         "measured": f"preorigin/v2 {pre_ok}; full/v4 {full_ok}",
         "passed": bool(pre_ok and full_ok)},
        {"criterion": "REPAIR 2: manifests re-hash (A0 checker): preorigin/v1-v2, full/v1-v4, "
                      "CODIS SOURCE.json",
         "threshold": "0 problems",
         "measured": "; ".join(f"{k} {v['status'] if isinstance(v, dict) else v}"
                               + (f" ({v['n_hashed']})" if isinstance(v, dict)
                                  and v.get("n_hashed") else "")
                               for k, v in chk.items()),
         "passed": ok_checker},
        {"criterion": "REPAIR 2 (verifier major issue 2): forward archive of the WRA real-time "
                      "discharge and weir feeds started (re-probed, plus the reachable Civil "
                      "IoT SensorThings mirror searched for discharge or stage at the composite "
                      "gauges and the weir)",
         "threshold": "a first snapshot in L0 with SOURCE.json",
         "measured": f"blocked: {archive_codes}; sta.colife.org.tw reachable but its "
                     "STA_WaterResource_v2 service holds flood sensors, gates, CCTV and "
                     "groundwater levels: 0 discharge or stage datastreams in the basin, no "
                     "Thing for 寶石橋 or 內茅埔 (水里玉峰橋 has only CCTV)",
         "passed": False},
        {"criterion": "REPAIR 2 (verifier minor, reported, post hoc): serial dependence of the "
                      "rain chain's annual leave-one-year-out residuals",
         "threshold": "reported; nothing changed",
         "measured": f"Yunlin 2020-2025 all negative (mean {yl['mean_recent_2020_2025']:+.2f} vs "
                     f"{yl['mean_rest']:+.2f} before), Spearman rho vs year "
                     f"{yl['spearman_residual_vs_year']['rho']:+.2f} (p "
                     f"{yl['spearman_residual_vs_year']['p_two_sided']:.4f}), lag-1 "
                     f"{yl['lag1_autocorrelation_over_evaluation_years']:+.2f}, runs test p "
                     f"{yl['runs_test']['p_one_sided_few_runs']:.2f}; Changhua no pattern "
                     f"(rho {ch['spearman_residual_vs_year']['rho']:+.2f}, p "
                     f"{ch['spearman_residual_vs_year']['p_two_sided']:.2f}); 2026 label "
                     "carries the caveat",
         "passed": True},
        {"criterion": "REPAIR 2 (verifier minor): ledger appended, not rewritten; corrected "
                      "state-space roles (fit_target) for the current versions",
         "threshold": "rows only added; 0 conflicts across fragments",
         "measured": f"W1 fragment 286 -> {n_ledger} rows (24 at publish, {len(rows)} "
                     "verification rows), earlier rows untouched; conflicts across "
                     f"fragments {len(ledger.conflicts(ledger.read_all()))}",
         "passed": len(ledger.conflicts(ledger.read_all())) == 0},
    ]

    # ------------------------------------------------------------------ products
    m_pre = json.loads((PRE2 / "MANIFEST.json").read_text())
    m_full = json.loads((FULL4 / "MANIFEST.json").read_text())
    products = [dict(p) for p in old["products"]]
    for p in products:
        if p["path"].endswith("preorigin/v1"):
            p["allowed_uses"] += ("; SUPERSEDED by preorigin/v2 (same values and members; its "
                                  "intervals carry no label and its structural-zero sd is the "
                                  "winter-0.5 alternative)")
        if p["path"].endswith("full/v3"):
            p["allowed_uses"] += ("; SUPERSEDED by full/v4 (same values, sd and members; "
                                  "composite, rain-index, percolation, grid and draws "
                                  "intervals unlabelled)")
    products = [
        {"path": rel_to_repo(PRE2), "layer": "L2 (district series, draws, daily composite) + "
                                             "L3 grid array",
         "manifest": rel_to_repo(PRE2 / "MANIFEST.json"),
         "allowed_uses": ", ".join(m_pre["allowed_uses"]) + " (grid_sw_v4.npz only via the "
                         "LUIMAP 2024 exemption: forcing_frozen, model_design, "
                         "design_time_evidence); prior_width_input now forbidden; every "
                         "interval labelled (monthly uncalibrated; Jan-May aggregate weakly "
                         "coverage-checked; composite sd uncalibrated); gate label 'passed on "
                         "7 design-time years: weak evidence'",
         "fit_window_end": "2022-12 (non-exempt; 2024-12 with the LUIMAP exemption for the "
                           "grid field)"},
        {"path": rel_to_repo(FULL4), "layer": "L2 + L3 grid array",
         "manifest": rel_to_repo(FULL4 / "MANIFEST.json"),
         "allowed_uses": ", ".join(m_full["allowed_uses"]) + " (2012-2025 flag 5 'not "
                         "distinguishable from climatology on the benchmark'; 2026-01..09 flag "
                         "6; every interval labelled; 2026 intervals uncalibrated at every "
                         "published scale); forbidden: " + ", ".join(m_full["forbidden_uses"]),
         "fit_window_end": m_full["fit_window_end"]},
    ] + products

    # ------------------------------------------------------------------ summary
    summary = (
        "W1 rebuilt the monthly Jiji-weir canal deliveries (surface water v4) from L0: a "
        "yearbook inflow composite with per-year gauge QC, a supply-limited diversion "
        "indicator, proportional Denton benchmarking to the annual district totals, 200 "
        "state-space draws per district, a pre-origin variant fitted on 2012-2022 and full "
        "variants (2012-2025, plus Jan-Sep 2026 from a rain-driven chain). The pre-registered "
        "primary test passes narrowly for the pre-origin variant: on the Jan-May "
        "leave-one-year-out benchmark it beats v3 and the climatological fraction in both "
        "districts, with all four 90 % paired-bootstrap intervals below zero (RMSE 0.359/0.339 "
        "vs 0.661/0.563 and 0.839/1.062 x 10^8 m3). The Yunlin-vs-climatology bound is -0.002 "
        "and the test uses seven design-time years. The full variant fails the same rule and "
        "is labelled 'not distinguishable from climatology on the benchmark'. The "
        "pre-registered head check fails in Yunlin layer 1 (-0.0087). In repair round 2, the "
        "verifier's major issue 1 was fixed in two new versions, preorigin/v2 and full/v4, "
        "with values, flags and members unchanged. The first repair labelled only the "
        "district and inflow-prediction tables (full/v3: 1071 of 457963 interval values). Now "
        f"every interval in every output carries a label: {lab('preorigin/v2')} and "
        f"{lab('full/v4')} values, 0 unlabelled. The composite, rain-index and monthly "
        "district intervals are 'uncalibrated'. The Jan-May aggregate passes the coverage rule "
        "only weakly (11/14 and 10/14, all misses over-predictions). The 2026 intervals are "
        "uncalibrated at every published scale. preorigin/v2 forbids prior_width_input, sets "
        "the structural-zero sd to 0, carries the design-time caveat in its gate and reports "
        "only 2012-2022 parse QC, re-parsed from L0 and identical to v1. A post hoc "
        "diagnostic confirms the verifier's Yunlin residual run: 2020-2025 all negative, "
        f"Spearman rho {yl['spearman_residual_vs_year']['rho']:+.2f}, p "
        f"{yl['spearman_residual_vs_year']['p_two_sided']:.3f}. It is disclosed in the 2026 "
        "label. The ledger was appended, with corrected state-space roles. The WP stays "
        "partial: the WRA real-time discharge and weir feeds are still unreachable (503, bot "
        "challenge, TLS), and the one reachable mirror (Civil IoT SensorThings) has no "
        "discharge or stage datastream for the composite gauges. No GPU and 0 Claude API "
        "tokens were used, nothing under hydrophysics/, tests/ or L0 was written, and "
        "nothing was staged in git.")

    # ------------------------------------------------------------------ key numbers
    vm = rf["value_minus_member_mean"]
    keys = [k.replace("-> annual scale coverage-checked;",
                      "-> the unpublished 12-month sums pass (every published 2026 interval "
                      "is uncalibrated);") for k in old["key_numbers"][:-1]] + [
        f"Repair 2: labelled interval values preorigin/v2 {lab('preorigin/v2')}, full/v4 "
        f"{lab('full/v4')} (before: preorigin/v1 {lab('preorigin/v1')}, full/v3 "
        f"{lab('full/v3')})",
        f"Repair 2: composite heuristic relative sd {rp['composite_sd_check']['rel_sd_flag_1_4']:.3f}"
        f" (pre-origin) / {rf['composite_sd_check']['rel_sd_flag_1_4']:.3f} (full), doubled on "
        "donor days; labelled uncalibrated (daily error not observable)",
        f"Repair 2: Denton value - member mean, 2012-2025 max |.|: Changhua "
        f"{vm['changhua']['2012_2025_max_abs']:.3f}, Yunlin {vm['yunlin']['2012_2025_max_abs']:.3f}"
        f" (RMS {vm['changhua']['2012_2025_rms']:.3f} / {vm['yunlin']['2012_2025_rms']:.3f}); "
        f"2026 Jan-Sep {vm['changhua']['2026_jan_sep_sum']:+.3f} / "
        f"{vm['yunlin']['2026_jan_sep_sum']:+.3f} (truncation at 0) x 10^8 m3",
        f"Repair 2 (post hoc): Yunlin annual chain residuals 2020-2025 mean "
        f"{yl['mean_recent_2020_2025']:+.3f} vs {yl['mean_rest']:+.3f} for 2012-2019 evaluation "
        f"years; Spearman rho {yl['spearman_residual_vs_year']['rho']:+.3f} (p "
        f"{yl['spearman_residual_vs_year']['p_two_sided']:.4f}); lag-1 "
        f"{yl['lag1_autocorrelation_over_evaluation_years']:+.3f}; runs test p "
        f"{yl['runs_test']['p_one_sided_few_runs']:.3f}; Jan-Sep residual vs the flag-5 Denton "
        f"series 2020-2025 mean {yl['janSep_residual_vs_flag5_denton_mean_recent']:+.3f}",
        f"Repair 2: pre-origin parse QC 2012-2022 from L0: {pq['station_years']} station-years, "
        f"{pq['printed_monthly_off_gt_tol']}/{pq['printed_monthly_compared']} printed monthly "
        f"means off > 5 %, {pid['matched']}/{pid['l1_flag0_station_days']} station-days "
        "identical to preorigin/v1",
        "Repair 2: checker " + ", ".join(
            f"{k} {v['n_hashed']}" for k, v in chk.items() if isinstance(v, dict)
            and v.get("n_hashed")) + " files ok",
        "Claude API tokens: 0; GPU: none"]

    # ------------------------------------------------------------------ deviations
    devs = list(old["deviations_from_plan"]) + [
        "Repair round 2 (2026-10-13): the verifier's major issue 1 was fixed in new versions "
        "preorigin/v2 and full/v4 (preorigin/v1 and full/v1-v3 unchanged). Nothing was fitted. "
        "Labels, member means, the pre-origin structural-zero sd and the pre-origin parse QC "
        "are new; values, flags and members are identical.",
        "The composite sd, the rain-index sd and every monthly district interval are labelled "
        "'uncalibrated' instead of being coverage-checked. No observation of the same kind "
        "exists: the weir publishes only annual inflow, there is no areal-rain truth, and no "
        "monthly delivery is observed. The plan's alternative (4.7: an unchecked interval is "
        "labelled uncalibrated) is applied. preorigin/v2 forbids prior_width_input for the "
        "same reason.",
        "preorigin/v2 and full/v4 list the previous version's MANIFEST.json and the files read "
        "from it as inputs (provenance chain). They no longer re-list hydrophysics/ modules or "
        "the 475 MB AMP_V2 wells tree. Those stay pinned through preorigin/v1's and full/v3's "
        "manifests, so an edit there flags only the older version.",
        "Ledger: this round only appended rows (286 -> 310 at publish). The earlier rewrite "
        "(superseded/W1.r2.csv) cannot be undone. The mislabelled 'input' roles of "
        "W1.preorigin/full.statespace_hyper stay on the v1 rows; corrected rows "
        "(fit_target) exist for preorigin_v2 and full_v4. The '#' qualifiers on the Jiji "
        "table were kept, for the reason given in the repair-round-2 section.",
        "preorigin/v2's QC_REPORT.md is v1's report plus a repair section, so its line on "
        "printed monthly means still quotes v1's 2012-2025 count (30 of 1471, marked 'all "
        "years 2012-2025'). The pre-origin count (21 of 1162, 2012-2022) is in its "
        "validation.json and in the repair section. No new version was made for this "
        "reporting line.",
        "The serial-dependence diagnostic of the chain residuals is post hoc, on design-time "
        "units, and logged in the A1 look log as 'diagnose'. Nothing was changed because of "
        "it: the 2026 values and members remain the declared ones, and the label carries the "
        "caveat.",
    ]

    # ------------------------------------------------------------------ issues
    iss = []
    for s in old["issues_for_downstream"]:
        if s.startswith("full/v3 supersedes"):
            continue
        if s.startswith("2026 intervals (full/v3)"):
            continue
        if s.startswith("Unresolved from the verifier"):
            continue
        if s.startswith("Ledger: the Jiji annual table appears with '#' qualifiers"):
            continue
        iss.append(s.replace("(inflow_predicted_monthly.csv, full/v3)",
                             "(inflow_predicted_monthly.csv, full/v4)"))
    iss += [
        "Use surface_water_v4/full/v4 for the recorded-forcing arm (U4) and "
        "surface_water_v4/preorigin/v2 for frozen-forcing, design and prior uses. Values and "
        "members are identical to full/v3 and preorigin/v1. The newer versions add a label "
        "to every interval, and preorigin/v2 corrects the structural-zero sd. Never use "
        "full/v2's 2026 sd or draws.",
        "Every published interval is labelled (plan 4.7). Most of them are uncalibrated: the "
        "monthly district sd and members, the daily composite sd (a heuristic: 1.4826 x MAD "
        "of the log annual scale factors, doubled on donor days), the rain-index sd and every "
        "2026 interval (monthly and Jan-Sep). Only the Jan-May aggregate of the 2012-2022/2025 "
        "draws (11/14 and 10/14, all misses over-predictions, n_eff 7) and the annual sums of "
        "the 2026 error model (19/22, not published) passed a coverage rule, both weakly and on "
        "design-time units. Forcing-uncertainty propagation (plan 4.8) should treat the "
        "members as uncalibrated at the monthly scale.",
        "preorigin/v2 refuses prior_width_input. No sd of the product is coverage-checked at "
        "a scale a prior width would use. Re-allow it only after such a check, in a new "
        "version.",
        "The Denton value and the state-space members have different centres in 2012-2025: "
        f"value - member mean up to {vm['changhua']['2012_2025_max_abs']:.2f} (Changhua) and "
        f"{vm['yunlin']['2012_2025_max_abs']:.2f} (Yunlin) x 10^8 m3 in a month (columns "
        "member_mean and value_minus_member_mean). Use the members as members, and do not mix "
        "value +- sd with them. 2026 members are centred on the value before truncation at 0. "
        "Districts are drawn independently before 2026 and jointly in 2026.",
        f"Yunlin 2026 is probably biased high (post hoc). The annual chain residuals of "
        f"2020-2025 are all negative (mean {yl['mean_recent_2020_2025']:+.2f} x 10^8 m3/yr, "
        f"Spearman rho vs year {yl['spearman_residual_vs_year']['rho']:+.2f}, p "
        f"{yl['spearman_residual_vs_year']['p_two_sided']:.3f}), which violates the "
        "exchangeability behind the pooled-residual error model. If that mean persisted, the "
        "2026 Yunlin annual delivery would be about 1.5 lower, and the Jan-Sep total about "
        f"{abs(yl['janSep_residual_vs_flag5_denton_mean_recent']):.1f} lower (against the "
        "flag-5 Denton series, not an observation). A revised error model needs a new "
        "declaration and a new version. Changhua shows no such pattern.",
        "Unresolved (cannot be fixed here): the forward archive of the WRA real-time discharge "
        "and weir feeds needs a reachable endpoint and a user decision on a recurring job "
        "(fhy WraApi 503, opendata bot challenge, sta.ci TLS failure; the reachable "
        "sta.colife.org.tw mirror carries no discharge or stage at the composite gauges). The "
        "head check failed in Yunlin layer 1 (-0.0087). The primary pass rests on 7 "
        "design-time years, with a marginal Yunlin-vs-climatology bound (-0.002) and no "
        "significant sign test. A definitive test needs the monthly intakes (request R-c, "
        "not drafted).",
        "Ledger: the '#' qualifiers on jiji_weir_annual_supply.csv name disjoint parts of the "
        "table in leave-one-year-out numbers: the held-out year and the training years of one "
        "column, and the industrial/public-supply column. Strip the qualifier before joining "
        "on the path. The v1 statespace_hyper rows say 'input'; the preorigin_v2/full_v4 rows "
        "carry the corrected 'fit_target'.",
    ]

    rr2 = {"date": "2026-10-13", "verifier_major_issues": 2, "verifier_minor_issues": 9,
           "fixed": ["major 1 (interval labels on every output; preorigin/v2, full/v4)",
                     "minor 2026 label wording", "minor draw centres (member_mean columns)",
                     "minor preorigin gate caveat", "minor statespace_hyper roles (new rows)",
                     "minor preorigin post-origin parse QC (2012-2022 only)",
                     "minor Yunlin residual serial dependence (diagnosed, disclosed)"],
           "explained_not_changed": ["minor '#' qualifiers (disjoint subsets in "
                                     "leave-one-year-out numbers)",
                                     "minor rewrite vs append (cannot be undone; append-only "
                                     "from now on)"],
           "disclosed_only": ["minor post hoc declaration of the error model",
                              "minor May 2026 zero (pooled-slope misspecification)",
                              "minor full variant uses 2023-2025 (guard refuses selection)",
                              "minor hydrophysics/ files hashed by full/v2-v3"],
           "not_fixable": ["major 2: forward archive blocked; head check failure and weak "
                           "primary evidence are facts of the registered tests"]}

    new = dict(old)
    new.update({
        "status": "partial", "date": "2026-10-13", "summary": summary, "acceptance": acc,
        "products": products,
        "code_paths": list(old["code_paths"]) + [
            str(HERE / f) for f in REP2_CODE],
        "key_numbers": keys, "deviations_from_plan": devs, "issues_for_downstream": iss,
        "evidence": list(old["evidence"]) + [
            rel_to_repo(EVID / "rep2_verify.json"), rel_to_repo(EVID / "rep2_state.json"),
            rel_to_repo(PRE2 / "validation.json"), rel_to_repo(FULL4 / "validation.json"),
            rel_to_repo(PRE2 / "QC_REPORT.md"), rel_to_repo(FULL4 / "QC_REPORT.md")],
        "previous_report": ["results/dataprep/reports/W1.r3.md",
                            "results/dataprep/reports/W1.r3.json"] + old["previous_report"],
        "repair_round_2": rr2})
    new["ledger"] = dict(old["ledger"])
    new["ledger"]["append_only_since_repair_round_2"] = True
    with open(REPORTS / "W1.json", "w") as f:
        json.dump(new, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------------------------ markdown
    parts = re.split(r"(?m)^## ", old_md)
    secs = {p.split("\n", 1)[0]: p for p in parts[1:]}
    order = [p.split("\n", 1)[0] for p in parts[1:]]

    def table(rows):
        out = ["| criterion | threshold | measured | pass |", "|---|---|---|---|"]
        for a in rows:
            out.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                       f"{'yes' if a['passed'] else 'NO'} |")
        return "\n".join(out)

    md = ["# WP W1: monthly canal deliveries, surface water v4 (report, 2026-10-12, updated "
          "in repair rounds 1 and 2)", "", f"Status: **partial**. {summary}", "",
          "## Products", ""]
    for p in products:
        md.append(f"- `{p['path']}` ({p['layer']}); allowed: {p['allowed_uses']}; "
                  f"fit_window_end {p['fit_window_end']}.")
    md += ["", "## Acceptance", "", table(acc), ""]
    md += repair2_section(rp, rf, yl, ch, scan, lab, pq, pid, alt, archive_codes, vm)
    skip = {"Products", "Acceptance", "Deviations from the plan", "Issues for downstream WPs",
            "Key numbers", "Provenance"}
    for h in order:
        if h.split(" (")[0] in skip or h in skip:
            continue
        body = secs[h].rstrip()
        if h.startswith("Repair round (2026-10-12 night)"):
            body = body.replace("Repair round (2026-10-12 night)",
                                "Repair round 1 (2026-10-12 night)", 1)
        md.append("## " + body + "\n")
    md += ["## Deviations from the plan", ""] + [f"- {d}" for d in devs] + [""]
    md += ["## Issues for downstream WPs", ""] + [f"- {s}" for s in iss] + [""]
    md += ["## Key numbers", ""] + [f"- {k}" for k in keys] + [""]
    prov = secs.get("Provenance", "Provenance\n").split("\n", 1)[1].strip()
    md += ["## Provenance", "", prov,
           "- Repair round 2: products `results/dataprep/surface_water_v4/preorigin/v2`, "
           "`results/dataprep/surface_water_v4/full/v4`; evidence "
           "`results/dataprep/reports/W1_evidence/rep2_verify.json`, "
           "`results/dataprep/reports/W1_evidence/rep2_state.json`; previous report "
           "`results/dataprep/reports/W1.r3.md` / `W1.r3.json`; code " + ", ".join(
               f"`dataprep_staging/w1_canal/{f}`" for f in REP2_CODE), ""]
    (REPORTS / "W1.md").write_text("\n".join(md))
    print("wrote", REPORTS / "W1.md", REPORTS / "W1.json", "ledger rows", n_ledger)


def repair2_section(rp, rf, yl, ch, scan, lab, pq, pid, alt, archive_codes, vm) -> list[str]:
    hp, hf = rp["coverage_hist_statespace_janmay"], rf["coverage_hist_statespace_janmay"]
    L = ["## Repair round 2 (2026-10-13): verifier findings and what was done", "",
         "A second independent verifier recomputed every number of the earlier rounds and "
         "found 0 critical, 2 major and 9 minor issues. This round publishes two new versions "
         "(preorigin/v2, full/v4), in which values, flags, members and grid values are "
         "identical to preorigin/v1 and full/v3, and it appends ledger rows. Nothing was "
         "fitted and no published file was edited. The code is in new rep2_*.py modules, "
         "because published manifests hash the older ones.", "",
         "| finding | action | result |", "|---|---|---|",
         "| MAJOR 1: only district_monthly.csv and inflow_predicted_monthly.csv carried "
         "interval labels; the composite sd (5114 daily values), rain-index sd (170), "
         "percolation sd, grid sd and draws had none, including in preorigin/v1, which allowed "
         "prior_width_input | every interval-bearing column or array labelled (row column "
         "interval_calibration, per-month arrays in the npz files, manifest interval_labels "
         "for the coverage-check tables); preorigin/v2 forbids prior_width_input | "
         f"preorigin/v2 {lab('preorigin/v2')}, full/v4 {lab('full/v4')} interval values "
         f"labelled (before {lab('preorigin/v1')} and {lab('full/v3')}) |",
         "| MAJOR 2: WP incomplete (forward archive, head check, weak primary) | re-probed the "
         "WRA feeds; searched the reachable Civil IoT SensorThings mirror | still blocked "
         f"({archive_codes}); the mirror has no discharge or stage datastream at the gauges "
         "or the weir; status partial |",
         "| minor: Yunlin residuals 2020-2025 all negative | post hoc diagnostic (logged look) "
         "and a caveat clause in the 2026 label; no setting changed | "
         f"Spearman rho {yl['spearman_residual_vs_year']['rho']:+.2f} (p "
         f"{yl['spearman_residual_vs_year']['p_two_sided']:.4f}), lag-1 "
         f"{yl['lag1_autocorrelation_over_evaluation_years']:+.2f}, runs p "
         f"{yl['runs_test']['p_one_sided_few_runs']:.2f}; Changhua rho "
         f"{ch['spearman_residual_vs_year']['rho']:+.2f} |",
         "| minor: 2026 'annual-scale coverage-checked' wording misleads; historical label "
         "rests on 10/14 | labels reworded: 2026 'uncalibrated at every published scale' "
         "(the annual check covers unpublished 12-month sums); historical rows state hits, "
         "n_eff and the side of every miss | done |",
         "| minor: draws not centred on the Denton value | member_mean and "
         "value_minus_member_mean columns; npz notes | max |value - member mean| "
         f"{vm['changhua']['2012_2025_max_abs']:.3f} / {vm['yunlin']['2012_2025_max_abs']:.3f} |",
         "| minor: primary fragile; preorigin gate label null | preorigin/v2 gate: label "
         "'passed on 7 design-time years: weak evidence', caveat, evidence_class "
         "design_time | done |",
         "| minor: ledger '#' qualifiers; rewrite; statespace_hyper roles | append-only; "
         "corrected fit_target rows for preorigin_v2/full_v4; qualifiers explained (below) | "
         "done / explained |",
         "| minor: post hoc error-model declaration; May 2026 zero; full variant uses "
         "2023-2025 | disclosed (labels carry the declaration timing; guard refuses "
         "selection and every pre-origin use of full/v4) | disclosed |",
         "| minor: hydrophysics/ files hashed by full/v2-v3; preorigin/v1 parse QC "
         "2023-2025 | full/v4 pins full/v3 through its manifest; preorigin/v2 validation "
         "holds only the 2012-2022 parse QC, re-parsed from L0 | done |", "",
         "### Interval labels (plan 4.7)", "",
         "| output | interval | label |", "|---|---|---|",
         f"| district_monthly.csv, percolation_basis.csv, draws, grid (pre-origin 2012-2022) | "
         f"sd, ss_p05/p50/p95, members | {rp['label_hist']} |",
         f"| same (full 2012-2025) | same | {rf['label_hist']} |",
         f"| same (full 2026, flag 6) | same | {rf['label_2026']} |",
         f"| structural-zero months (Dec-Jan) | sd 0, members 0 | {rp['label_structural']} |",
         f"| inflow_composite_daily.parquet (pre-origin) | sd | {rp['label_composite']} |",
         f"| inflow_composite_daily.parquet (full) | sd | {rf['label_composite']} |",
         f"| basin_rain_index_monthly.csv (full) | sd | {rf['label_rain']} |",
         "| inflow_predicted_monthly.csv (full 2026) | sd | full/v3 label + ' (design-time "
         "years, double leave-one-year-out; rule declared in W1.ext2026_uncertainty.r1 after "
         "the chain's errors were known)' |",
         "| gauges_daily_L1.parquet | sd (empty) | no interval published |",
         "| janmay_validation, statespace_loo_coverage, rep_coverage_*, ext2026_loo_* | "
         "p05/p95, lo/hi | coverage-check tables (manifest interval_labels) |", "",
         f"Coverage behind the 2012-2022/2025 labels (A1 gate-2, clusters = years): pre-origin "
         f"{hp['hits']}/{hp['n']} [{hp['a1_gate2']['lo']:.3f}, {hp['a1_gate2']['hi']:.3f}], "
         f"full {hf['hits']}/{hf['n']} [{hf['a1_gate2']['lo']:.3f}, {hf['a1_gate2']['hi']:.3f}]. "
         "The composite and rain-index sd cannot be coverage-checked: the weir publishes only "
         "annual inflow (the composite is rescaled to it), and areal rain has no truth.", "",
         "### Pre-origin product changes", "",
         "- Structural-zero months: sd 0 (their 200 members are 0); the winter-0.5 alternative "
         "that preorigin/v1 stored as the sd is in structural_alt_value, and the grid field "
         "is value_structural_alt (as full/v3 did). The grid sd is unchanged elsewhere (max "
         "|diff| 8.7e-19).",
         "- prior_width_input moved to forbidden_uses. The guard refuses it, and prior or "
         "selection still read the district table but not the LUIMAP grid.",
         "- Gate: acceptance_passed stays true; label 'pre-registered Jan-May acceptance "
         "passed on 7 design-time years per district: weak evidence'; caveat lists the "
         "scout's design on the same years, the percentile bootstrap at n = 7, the -0.002 "
         "bound, the sign tests and the failed head check.",
         f"- validation.json: parse QC of 2012-2022 only, re-parsed from the L0 PDFs: "
         f"{pq['station_years']} station-years, {pq['printed_monthly_off_gt_tol']}/"
         f"{pq['printed_monthly_compared']} printed monthly means off > 5 %, "
         f"{pid['matched']}/{pid['l1_flag0_station_days']} station-days identical to v1 (max "
         f"|diff| {pid['max_abs_diff']}). The scout comparison (2012-2025) moved to full/v4.",
         "", "### Serial dependence of the chain residuals (post hoc, design-time units)", "",
         "| district | residuals obs - chain, 2012, 2013, 2017-2025 (10^8 m3) | 2020-2025 mean | "
         "earlier mean | Spearman rho (p) | lag-1 | runs test p |", "|---|---|---|---|---|---|---|"]
    for nm, d in (("Changhua", ch), ("Yunlin", yl)):
        L.append(f"| {nm} | {', '.join(f'{x:+.2f}' for x in d['residual_obs_minus_chain'])} | "
                 f"{d['mean_recent_2020_2025']:+.2f} | {d['mean_rest']:+.2f} | "
                 f"{d['spearman_residual_vs_year']['rho']:+.2f} "
                 f"({d['spearman_residual_vs_year']['p_two_sided']:.4f}) | "
                 f"{d['lag1_autocorrelation_over_evaluation_years']:+.2f} | "
                 f"{d['runs_test']['p_one_sided_few_runs']:.2f} |")
    L += ["",
          "Interpretation (post hoc): the Yunlin chain has over-predicted every year since "
          "2020. The trend is significant (p < 0.01) even though the runs test, with 4 "
          "positive and 7 negative signs, is not. The pooled-residual error model assumes "
          "exchangeable years, so the 2026 Yunlin value (6.889 x 10^8 m3 Jan-Sep) is probably "
          "biased high and its interval mis-centred. Changhua shows no pattern. Nothing was "
          "changed: a revised model needs a new declaration and version.", "",
          "### Forward archive (plan step 6)", "",
          f"- WRA endpoints: {archive_codes} (fhy WraApi 503; opendata returns a "
          "bot-challenge HTML page, which was not bypassed; sta.ci.taiwan.gov.tw fails TLS "
          "verification, which was not bypassed).",
          "- The Civil IoT data mirror sta.colife.org.tw/STA_WaterResource_v2 is reachable. "
          "Things matching 寶石橋 and 內茅埔: none. 水里: "
          f"{', '.join(alt['things'].get('水里', []))}. 集集: "
          f"{', '.join(alt['things'].get('集集', []))}. Discharge or stage datastreams in the "
          f"basin: {len(alt.get('flow_or_stage_datastreams_in_basin') or [])}. Its observed "
          "properties are flood depth, CCTV frames, gate openings, groundwater levels and "
          "flood-sensor stage, so it cannot replace the discharge and weir feeds. No snapshot "
          "was archived, because a snapshot of unrelated sensors is not the archive the plan "
          "asks for.", "",
          "### Why the '#' qualifiers stay", "",
          "In a leave-one-year-out number, '#held_out_year_district_total' (validator) and "
          "'#training_year_totals' (fit_target) name disjoint subsets of one column. In every "
          "fold the held-out year never enters its own fit, so no datum plays two roles in "
          "one fold. '#industrial_public_supply' is a different column (input). The P1 and W3 "
          "fragments use the same '#' convention for distinct parts of one file. Reading "
          "U6/4.1 as one role per file would forbid any cross-validation on a single table. "
          "Splitting each statistic into per-fold numbers would satisfy that reading but add "
          "no information. The rows were therefore kept, and this round's new rows use no "
          "qualifier.", ""]
    return L


if __name__ == "__main__":
    main(sys.argv[1])
