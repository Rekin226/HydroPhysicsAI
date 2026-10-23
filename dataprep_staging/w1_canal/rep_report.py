"""W1 report after the repair round (reports/W1.md, W1.json).

    python -I dataprep_staging/w1_canal/rep_report.py

The resume-round report is kept as reports/W1.r2.md / W1.r2.json (the first build's as
W1.r1.*). This script merges the repair round (surface_water_v4/full/v3, the A1
registrations, the ledger role correction, the sigma-grid sensitivity, the provenance
addendum and the forward-archive re-probe) into a new W1.md / W1.json. Numbers are read
from full/v3's validation.json and the evidence files written by rep_verify.py and
rep_sigma_sensitivity.py. Also writes W1_evidence/rep_provenance_addendum.json and appends
the ledger rows of the sensitivity and addendum numbers.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
from common import ledger  # noqa: E402
from common.hashing import sha256_file, sha256_tree  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from w1_canal import checks  # noqa: E402
from w1_canal import config as C  # noqa: E402

REPORTS = DATAPREP / "reports"
EVID = REPORTS / "W1_evidence"
SW = DATAPREP / C.PRODUCT_BASE
V3 = SW / "full" / "v3"
REP_CODE = ["rep_config.py", "rep_uncert.py", "rep_prereg.py", "rep_build.py",
            "rep_publish.py", "rep_verify.py", "rep_ledger.py", "rep_sigma_sensitivity.py",
            "rep_report.py"]
LICENCE_CHECK = {
    "checked_utc": "2026-10-12T15:41Z",
    "codis_footer": "codis.cwa.gov.tw footer: '中央氣象署版權所有｜使用時應註明出處' (CWA copyright; "
                    "cite the source when used) and a link to the '政府網站資料開放宣告' (government "
                    "website open-data declaration, www.cwa.gov.tw/V8/C/information.html)",
    "declaration_page": "HTTP 403 from this VM; not read",
    "conclusion": "attribution required; the open-data declaration that the SOURCE.json licence "
                  "string refers to could not be read; the SOURCE.json is not edited (full/v2 and "
                  "v3 hash it); products stay under the gitignored results/"}


def f3(x):
    return f"{x:.3f}"


def iv(g):
    return f"[{g['lo']:.3f}, {g['hi']:.3f}]"


def provenance_addendum() -> dict:
    yb = C.YEARBOOK_DIR
    pilot = DATAPREP / "_pilots" / "surface_water_monthly"

    def rec(p):
        p = Path(p)
        return {"path": rel_to_repo(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    post = sorted(q for y in (2023, 2024, 2025) for q in (yb / f"hyb{y}").glob("*.pdf"))
    out = {
        "note": "inputs read by the reporting checks in validation.json (common_checks) that "
                "the manifests of preorigin/v1, full/v1 and full/v2 do not list (verifier minor "
                "issue); hashed now (2026-10-12, repair round), not at build time. No fitted "
                "parameter or output value depends on them. full/v3 lists them in its manifest.",
        "preorigin_v1": {
            "parse_qc_post_origin_yearbooks": [rec(p) for p in post],
            "parse_vs_pilot": [rec(pilot / "MANIFEST.json"), rec(pilot / "choushui_daily_q.csv"),
                               rec(pilot / "jiji_janmay_supply_2015_2021.csv")],
            "why_not_reversioned": "data and fitted parameters are unaffected (the pre-origin "
                                   "fit uses 2012-2022 only; the 2023-2025 entries are parse QC "
                                   "statistics in validation.json); a preorigin/v2 would repeat "
                                   "every output byte for byte"},
        "full_v1_v2": {
            "head_check": [rec(C.WELL_META), rec(checks.CODIS_RAIN),
                           {"path": rel_to_repo(C.WELL_DIR), "sha256_tree": sha256_tree(
                               C.WELL_DIR)}],
            "parse_vs_pilot": [rec(pilot / "MANIFEST.json"), rec(pilot / "choushui_daily_q.csv"),
                               rec(pilot / "jiji_janmay_supply_2015_2021.csv")]},
        "codis_licence_check": LICENCE_CHECK}
    return out


def main():
    r2j, r2m = REPORTS / "W1.r2.json", REPORTS / "W1.r2.md"
    if not r2j.exists():
        shutil.copy2(REPORTS / "W1.json", r2j)
        shutil.copy2(REPORTS / "W1.md", r2m)
    r2 = json.loads(r2j.read_text())
    old_md = r2m.read_text()
    val = json.loads((V3 / "validation.json").read_text())
    rep = val["repair_r1"]
    ver = json.loads((EVID / "rep_verify.json").read_text())
    sig = json.loads((EVID / "rep_sigma_sensitivity.json").read_text())
    add = provenance_addendum()
    with open(EVID / "rep_provenance_addendum.json", "w") as f:
        json.dump(add, f, indent=1, ensure_ascii=False)
    ga, gj, gi, gh = (rep["coverage_annual"], rep["coverage_janmay"], rep["coverage_inflow"],
                      rep["coverage_hist_statespace_janmay"])
    ex = ver["exactness"]
    mc = rep["member_checks"]
    s26 = rep["series_2026"]
    prm = rep["error_model"]["params_2026"]
    chk = ver["checker"]
    probes = ver["forward_archive_probes"]
    regs = ver["registrations"]
    decl = rep["declaration"]
    sp, sf = sig["preorigin"], sig["full"]
    jan_sep = {nm: mc[nm]["jan_sep_p05_p50_p95"] for nm in mc}
    labels = ex["interval_calibration_labels"]

    # ------------------------------------------------------------------ acceptance
    acc = []
    for a in r2["acceptance"]:
        a = dict(a)
        if a["criterion"].startswith("Manifests re-hash"):
            a["criterion"] = ("Manifests re-hash (A0 checker) and loader guard enforces the uses "
                              "(preorigin/v1, full/v1, full/v2, full/v3, new L0 SOURCE.json)")
            a["measured"] += ("; repair round: " + ", ".join(
                f"{k.split('surface_water_v4/')[1]} {v['status']} ({v['n_hashed']})"
                for k, v in chk.items()) + "; full/v3 opens for forcing_recorded_arm and "
                "refuses forcing_frozen, model_design, scoring_target, calibration_target, prior")
        if a["criterion"].startswith("Step 6: forward archive"):
            a["measured"] = ("blocked from this VM (first probe and repair-round re-probe "
                             f"{ver['run_utc'][:16]}Z): " + "; ".join(
                                 f"{p['url'].split('/')[2]} -> {p['http_code']}" for p in probes)
                             + " (503, bot-challenge page, TLS name failure; no data endpoint "
                               "reachable; a recurring job is also a user decision)")
        acc.append(a)
    acc += [
        {"criterion": "REPAIR (verifier major issue 1; plan 4.7): every published interval of "
                      "the full product is coverage-checked or labelled 'uncalibrated' "
                      "(full/v3 district_monthly.csv interval_calibration column)",
         "threshold": "every row labelled; the 2026 label follows the declared annual coverage "
                      "rule (W1.ext2026_uncertainty.r1)",
         "measured": f"{ex['rows'] - ex['rows_without_interval_calibration']}/{ex['rows']} rows "
                     "labelled: " + "; ".join(f"{n} x '{k[:60]}...'" for k, n in labels.items()),
         "passed": ex["rows_without_interval_calibration"] == 0},
        {"criterion": "REPAIR (declared, design-time units): annual-scale coverage of the "
                      "repaired 2026 error model, double leave-one-year-out over the 11 "
                      "evaluation years x 2 districts; A1 gate-2 interval (clusters = years) "
                      "must contain 0.90, else every 2026 interval is labelled 'uncalibrated'",
         "threshold": "A1 gate-2 interval contains 0.90",
         "measured": f"{ga['hits']}/{ga['n']} = {ga['coverage']:.3f}; A1 gate-2 {iv(ga['a1_gate2'])}"
                     f"; Clopper-Pearson at n_eff {ga['n_years']} "
                     f"[{ga['clopper_pearson_95_neff_years'][0]:.3f}, "
                     f"{ga['clopper_pearson_95_neff_years'][1]:.3f}]; misses "
                     + ", ".join(f"{m['year']} {m['district']} ({m['side']})"
                                 for m in ga["misses"]),
         "passed": bool(ga["a1_gate2"]["pass"])},
        {"criterion": "REPAIR (reported, not gated): Jan-May 2017-2021 coverage of the same "
                      "double leave-one-year-out ensembles",
         "threshold": "reported",
         "measured": f"{gj['hits']}/{gj['n']}; A1 gate-2 {iv(gj['a1_gate2'])}",
         "passed": True},
        {"criterion": "REPAIR (reported; sets the inflow sd label): monthly 90 % interval of the "
                      "rain-runoff regression on observed composite months (every day flag <= 1)"
                      ", double leave-one-year-out",
         "threshold": "A1 gate-2 interval contains 0.90 (label), otherwise 'uncalibrated'",
         "measured": f"{gi['hits']}/{gi['n']} = {gi['coverage']:.3f}; A1 gate-2 "
                     f"{iv(gi['a1_gate2'])} over {gi['n_years']} year clusters",
         "passed": bool(gi["a1_gate2"]["pass"])},
        {"criterion": "REPAIR (declared): historical rows (full/v1 state-space draws) labelled "
                      "from the A1 gate-2 re-evaluation of the existing full-variant Jan-May "
                      "leave-one-year-out coverage",
         "threshold": "label 'coverage-checked' if the interval contains 0.90, else "
                      "'uncalibrated'; monthly scale 'uncalibrated' either way",
         "measured": f"{gh['hits']}/{gh['n']}; A1 gate-2 {iv(gh['a1_gate2'])} -> "
                     f"'{rep['label_hist'][:40]}...'",
         "passed": True},
        {"criterion": "REPAIR (verifier issues b-d): 2026 members consistent with the shipped "
                      "values and the structural zeros",
         "threshold": "distinct trajectories >> 11; pre-truncation member mean = value; "
                      "structural-zero months sd 0 and every member 0; no negative member; sd = "
                      "members' standard deviation",
         "measured": "; ".join(
             f"{nm}: {mc[nm]['distinct_trajectories']} distinct, pre-truncation mean - value "
             f"{mc[nm]['pre_truncation_mean_minus_value_max_abs']:.1e}, Jan-Sep post-truncation "
             f"mean - value {mc[nm]['jan_sep_mean_minus_value']:+.3f}, structural non-zero "
             f"{mc[nm]['structural_nonzero_members']}, negative {mc[nm]['negative_members']}, "
             f"sd vs members {ex[nm]['sd_2026_equals_member_sd_max_abs']:.1e}" for nm in mc),
         "passed": all(mc[nm]["distinct_trajectories"] > 11 and mc[nm]["negative_members"] == 0
                       and mc[nm]["structural_nonzero_members"] == 0
                       and mc[nm]["pre_truncation_mean_minus_value_max_abs"] < 1e-9
                       for nm in mc)},
        {"criterion": "REPAIR: full/v3 values identical to full/v2 (table values and flags, grid "
                      "value/flag/value_v3level, 2012-2025 draws = full/v1); sd changed only in "
                      "structural-zero and 2026 months; the former structural sd kept as "
                      "structural_alt_value",
         "threshold": "all true",
         "measured": ", ".join(f"{k} {ex[k]}" for k in (
             "values_identical_to_v2", "flags_identical_to_v2",
             "sd_identical_to_v2_outside_structural_and_2026",
             "structural_alt_equals_v2_sd_2012_2025", "draws_2012_2025_identical_to_v1",
             "grid_value_identical_to_v2", "grid_flag_identical_to_v2",
             "grid_value_v3level_identical_to_v2", "mask_hash_unchanged"))
                     + f"; structural sd max {ex['structural_sd_max']}",
         "passed": all(ex[k] for k in (
             "values_identical_to_v2", "flags_identical_to_v2",
             "sd_identical_to_v2_outside_structural_and_2026",
             "structural_alt_equals_v2_sd_2012_2025", "draws_2012_2025_identical_to_v1",
             "grid_value_identical_to_v2", "grid_flag_identical_to_v2",
             "grid_value_v3level_identical_to_v2", "mask_hash_unchanged"))
                   and ex["structural_sd_max"] == 0.0},
        {"criterion": "REPAIR (verifier minor, U1): no flag-5 Denton target among the scores; a "
                      "prospective test that scores only observed composite months against an "
                      "observed-only climatology, registered before any 2026 yearbook value "
                      "exists",
         "threshold": "Denton-target RMSEs out of chain_test.scores; registration loads",
         "measured": f"Denton keys in scores: {ex['denton_target_rmse_in_scores']}; "
                     f"W1.ext2026_prospective_obs.v1 ok={regs['W1.ext2026_prospective_obs.v1']['ok']}"
                     f" ({regs['W1.ext2026_prospective_obs.v1'].get('created', '')})",
         "passed": (not ex["denton_target_rmse_in_scores"])
                   and regs["W1.ext2026_prospective_obs.v1"]["ok"]},
    ]

    # ------------------------------------------------------------------ products
    products = []
    for pr in r2["products"]:
        pr = dict(pr)
        if pr["path"].endswith("full/v2"):
            pr["allowed_uses"] += ("; superseded by full/v3 (same values; its 2026 sd and "
                                   "members are not coverage-checked: do not use them)")
        products.append(pr)
    products.insert(3, {
        "path": "results/dataprep/surface_water_v4/full/v3", "layer": "L2 + L3 grid array",
        "manifest": "results/dataprep/surface_water_v4/full/v3/MANIFEST.json",
        "allowed_uses": "forcing_recorded_arm, statistical_layer_history, after_the_fact_analysis"
                        " (values as full/v2: 2012-2025 flag 5 labelled 'not distinguishable from "
                        "climatology on the benchmark'; 2026-01..2026-09 flag 6, rain-driven "
                        "extension 'not distinguishable from climatology on the annual leave-one-"
                        "year-out test'); 2026 intervals: annual scale coverage-checked on design-"
                        "time years, monthly and Jan-Sep uncalibrated; supersedes full/v1 and v2",
        "fit_window_end": "2026-09"})

    # ------------------------------------------------------------------ numbers and text
    key = []
    for k in r2["key_numbers"]:
        if k.startswith("State-space LOO 90 % coverage"):
            k = ("State-space LOO 90 % coverage of the held-out Jan-May totals: pre-origin 11/14, "
                 "full 10/14; mean 90 % width of the pre-origin variant: Changhua 1.24, Yunlin "
                 "1.62 x 10^8 m3")
        if k.startswith("Claude API"):
            continue
        key.append(k)
    key += [
        f"Repair: annual 90 % coverage of the repaired 2026 error model (double LOO, 11 years x 2)"
        f": {ga['hits']}/{ga['n']}, A1 gate-2 {iv(ga['a1_gate2'])} -> annual scale coverage-"
        f"checked; Jan-May 2017-2021 {gj['hits']}/{gj['n']}; inflow monthly {gi['hits']}/"
        f"{gi['n']} ({iv(gi['a1_gate2'])})",
        "Repair: 2026 error-model scales (Changhua/Yunlin): f = "
        f"{prm['changhua']['f']:.3f}/{prm['yunlin']['f']:.3f}, g = {prm['changhua']['g']:.3f}, "
        f"h = {prm['changhua']['h']:.3f}; annual residual RMSE {prm['changhua']['annual_rmse']:.3f}"
        f"/{prm['yunlin']['annual_rmse']:.3f} x 10^8 m3 (11 pool years)",
        "Repair: 2026 Jan-Sep 90 % member interval (p05/p50/p95, 10^8 m3): Changhua "
        + "/".join(f3(x) for x in jan_sep["changhua"]) + ", Yunlin "
        + "/".join(f3(x) for x in jan_sep["yunlin"])
        + f" (shipped values {s26['changhua']['jan_sep_total']:.3f} / "
          f"{s26['yunlin']['jan_sep_total']:.3f})",
        f"Repair: historical Jan-May state-space coverage re-evaluated: {gh['hits']}/{gh['n']}, "
        f"A1 gate-2 {iv(gh['a1_gate2'])}",
        "Post hoc sigma grid to 1.0 (state-space Jan-May LOO coverage): pre-origin "
        f"{sp['registered']['hits']}/{sp['registered']['n']} -> {sp['extended']['hits']}/"
        f"{sp['extended']['n']}, full {sf['registered']['hits']}/{sf['registered']['n']} -> "
        f"{sf['extended']['hits']}/{sf['extended']['n']}",
        "Claude API tokens: 0; GPU: none"]
    dev = [d + (" (full/v1-v2; full/v3 sets their sd to 0 and keeps that value in "
                "structural_alt_value)" if d.startswith("The structural-zero months (Dec-Jan")
                else "") for d in r2["deviations_from_plan"]] + [
        "Repair round (2026-10-12 night): the verifier's major issue 1 was fixed in a new version "
        "full/v3 (full/v1, full/v2 and preorigin/v1 unchanged). The repaired 2026 error model and "
        "the coverage rule that sets the interval labels were declared (W1.ext2026_uncertainty.r1"
        f", {decl['created'][:19]}Z) after the verifier's report and after the chain's annual "
        "leave-one-year-out RMSE and bias were known (f uses exactly those); the coverage had not "
        "been computed. The code ran on synthetic inputs and once end to end on the real chain "
        "folds with synthetic draws, totals and composite before the declaration (disclosed in "
        "it). It is a declaration on design-time units, not a fresh test.",
        "The earlier repair attempt (an interrupted session) had written rep_config.py, "
        "rep_uncert.py, rep_build.py and rep_prereg.py but registered and computed nothing. "
        "This session kept the design, fixed a crash in rep_uncert.gate for the inflow table, "
        "replaced the fixed historical-row label by one driven by the A1 gate-2 re-evaluation, "
        "added the 2015-2021 Jan-May look, and wrote rep_publish.py, rep_verify.py, "
        "rep_ledger.py, rep_sigma_sensitivity.py and rep_report.py.",
        "2026 members are truncated at 0, so their mean exceeds the shipped value where the value "
        f"is 0 or small (Jan-Sep +{mc['changhua']['jan_sep_mean_minus_value']:.3f} / "
        f"+{mc['yunlin']['jan_sep_mean_minus_value']:.3f} x 10^8 m3; Feb, Mar and May); the "
        "shipped value stays the registered chain prediction (the pre-truncation mean equals it).",
        "Structural-zero convention changed in full/v3: sd 0 (equal to the members' spread); the "
        "winter-0.5 alternative that full/v1-v2 stored as the sd is in structural_alt_value and "
        "the grid field value_structural_alt.",
        "Ledger roles corrected by rewriting W1's fragment once (previous fragment kept as "
        "ledger/superseded/W1.r2.csv): the Jan-May briefing table is design_time_evidence for "
        "the Jan-May numbers (30 rows); in the step-6 rows the Jiji table is split with '#' "
        "qualifiers into the held-out year (validator), the training years (fit_target, or input "
        "for the climatology and persistence baselines) and R (input) (12 rows).",
        "The legacy pre-registration W1_janmay_benchmark_2026-10-12 was put into the hash-chained "
        "INDEX by the declaration W1_janmay_benchmark_2026-10-12.record (A1 repair round), "
        "without changing the test.",
        "Pre-origin manifest provenance (verifier minor issue): preorigin/v1 is not re-versioned; "
        "the unlisted reporting inputs (hyb2023-2025 for the parse QC, the scout's files for "
        "parse_vs_pilot; the head check inputs for full/v1-v2) are hashed in "
        "W1_evidence/rep_provenance_addendum.json and listed in full/v3's manifest.",
        "Verifier major issue 2 (incomplete WP) cannot be fixed in this round: the forward "
        "archive is blocked from this VM (re-probed), the pre-registered head check failed and "
        "stays failed, and the primary pass rests on seven design-time years. Status stays "
        "partial."]
    issues = []
    for k in r2["issues_for_downstream"]:
        if k.startswith("State-space draws (district_draws.npz"):
            k += (f" Post hoc (repair round, reporting only): extending the sigma grid from 0.45 "
                  f"to 1.0 changes the pre-origin coverage from {sp['registered']['hits']}/14 to "
                  f"{sp['extended']['hits']}/14 and the full one from "
                  f"{sf['registered']['hits']}/14 to {sf['extended']['hits']}/14 (mean width "
                  f"Changhua/Yunlin pre-origin {sp['registered']['mean_width_90']['chc']:.2f}/"
                  f"{sp['registered']['mean_width_90']['yl']:.2f} -> "
                  f"{sp['extended']['mean_width_90']['chc']:.2f}/"
                  f"{sp['extended']['mean_width_90']['yl']:.2f}); the registered grid is kept.")
        if k.startswith("full/v2 supersedes full/v1"):
            k = ("full/v3 supersedes full/v1 and full/v2 for the recorded-forcing arm: identical "
                 "values (2012-2025 = full/v1, 2026-01..2026-09 flag 6), repaired 2026 sd and "
                 "members, an interval_calibration label on every row. Use full/v3's "
                 "district_draws.npz as forcing members; never use full/v2's 2026 sd or draws. "
                 "The 2026 months are an extrapolation labelled 'rain-driven extension not "
                 "distinguishable from climatology on the annual leave-one-year-out test': use "
                 "them with their sd and draws, never for calibration or scoring (U1, U4).")
        if k.startswith("2026 caveat"):
            k = ("2026 caveat (post hoc): the registered linear spec fails in dry months. This is "
                 "a pooled-slope misspecification, not extrapolation: May-2026 rain (161 mm) lies "
                 "inside the 2012-2025 May range (110-814 mm). The failure was already visible "
                 "in the design-time leave-one-year-out output: 17 of 132 fold months predict "
                 "inflow below the 2012-2025 monthly minimum and 2 predict 0 (May 2018, April "
                 "2021; May 2018 chain delivery 0 against Denton 0.83). In 2026 the predicted "
                 "inflow is below the monthly minimum in January, February and May, and May "
                 "2026 is delivered as 0. Flooring the inflow at the monthly minimum would give "
                 "Jan-Sep 6.261 / 7.395 instead of 5.519 / 6.889 x 10^8 m3. A revised spec needs "
                 "a new pre-registration and a new version. The repaired members cover part of "
                 "the gap: the Jan-Sep 90 % member intervals are Changhua "
                 f"{jan_sep['changhua'][0]:.2f}-{jan_sep['changhua'][2]:.2f} and Yunlin "
                 f"{jan_sep['yunlin'][0]:.2f}-{jan_sep['yunlin'][2]:.2f}.")
        if k.startswith("Prospective test registered"):
            k = ("Prospective tests of the 2026 inflow prediction: the primary is now "
                 "W1.ext2026_prospective_obs.v1 (repair round). It scores only scorable months "
                 "(all three gauges numeric every day, no donor or interpolated day) against an "
                 "observed-only 2012-2025 climatology, with at least 5 scorable months. The "
                 "original W1.ext2026.prospective_inflow_2026 stays on file as secondary, with "
                 "the caveat that its target and baseline admit flag-3/4 values (U1). Both run "
                 "when the 2026 yearbook and Jiji table appear (about mid-2027). Until then "
                 "there is no skill evidence for 2026.")
        if k.startswith("The manifests list dataprep_staging/harness/coverage.py"):
            k += (" full/v3 also lists harness/prereg2.py, coverage2.py and coverage.py and the "
                  "AMP_V2/data/wells tree (head-check input), so the checker re-hashes about "
                  "475 MB for it.")
        issues.append(k)
    issues += [
        f"2026 intervals (full/v3): the annual-scale error model passed its declared coverage "
        f"check ({ga['hits']}/{ga['n']}, A1 gate-2 {iv(ga['a1_gate2'])}). That is weak "
        f"evidence: 11 design-time years, and the misses are the 2013 under-predictions and the "
        f"2025 Changhua over-prediction. Monthly and Jan-Sep intervals have no observed target "
        f"and are labelled 'uncalibrated' at those scales. Members of the two districts share "
        f"the pool year, draw index and shock (correlated across districts), whereas the "
        f"2012-2025 draws are independent between districts.",
        f"Inflow prediction (inflow_predicted_monthly.csv, full/v3): sd = g x leave-one-year-out "
        f"RMS by calendar month; the 90 % interval (value +- 1.645 sd, floored at 0) covered "
        f"{gi['hits']}/{gi['n']} observed composite months (A1 gate-2 {iv(gi['a1_gate2'])}). The "
        f"RMS uses every composite month of the pool years, including months with donor or "
        f"interpolated days (a fit quantity, not a target).",
        "Ledger: the Jiji annual table appears with '#' qualifiers (#held_out_year_district_total"
        ", #training_year_totals, #industrial_public_supply) where one number uses it in two "
        "roles; consumers joining on the plain path must strip the qualifier.",
        "CODIS licence (minor issue): the CODIS footer requires attribution ('使用時應註明出處') "
        "and links the CWA open-data declaration, which returned HTTP 403 from this VM. The "
        "SOURCE.json licence string is unchanged (full/v2 and v3 hash it); see "
        "W1_evidence/rep_provenance_addendum.json.",
        "Unresolved from the verifier (cannot be fixed here): the forward archive of the WRA "
        "real-time feeds needs a reachable endpoint and a user decision on a recurring job; the "
        "head check failed in Yunlin layer 1 (-0.0087); the primary pass rests on 7 design-time "
        "years with a marginal Yunlin-vs-climatology bound (-0.002) and no significant sign "
        "test. A definitive test needs the monthly intakes (request R-c, not drafted)."]

    summary = (
        "Status partial. W1 rebuilt the monthly Jiji-weir canal deliveries (surface water v4) "
        "from L0: yearbook inflow composite with per-year gauge QC, supply-limited diversion "
        "indicator, proportional Denton benchmarking to the annual district totals, 200 "
        "state-space draws per district, a pre-origin variant fitted on 2012-2022 "
        "(preorigin/v1) and full variants (full/v1 2012-2025; full/v2 adds Jan-Sep 2026 from a "
        "rain-driven chain). On the pre-registered Jan-May leave-one-year-out benchmark the "
        "pre-origin variant beats v3 and the climatological fraction in both districts with all "
        "four 90 % paired-bootstrap intervals below zero (RMSE 0.359 / 0.339 vs climatology "
        "0.661 / 0.563 and v3 0.839 / 1.062 x 10^8 m3), so the primary acceptance passes, "
        "narrowly (Yunlin vs climatology upper bound -0.002) and on seven design-time years. The "
        "full variant fails the same rule (2/4) and is labelled 'not distinguishable from "
        "climatology on the benchmark'; the pre-registered head check fails in one of four "
        "groups (-0.0087). In the repair round the verifier's major issue 1 was fixed in "
        "full/v3. The 2026 sd and members come from a declared error model built on residuals "
        "against the observation-constrained state-space draws, with bias folded into the "
        f"spread, small-sample inflation and a smoothed bootstrap ({mc['changhua']['distinct_trajectories']} distinct "
        "trajectories, centred, structural zeros 0). Its annual 90 % intervals covered "
        f"{ga['hits']}/{ga['n']} held-out district totals (A1 gate-2 {iv(ga['a1_gate2'])}), so "
        "they are labelled coverage-checked at the annual scale and uncalibrated at the monthly "
        "and Jan-Sep scales. Every row now carries an interval label; the Denton-target "
        "agreement is out of the scores; a U1-compliant prospective test is registered; ledger "
        "roles are corrected. Values are unchanged. The forward archive of the WRA real-time "
        "feeds is still blocked (503, bot challenge, TLS failure), so the WP stays partial. No "
        "GPU, 0 Claude API tokens, nothing under hydrophysics/, tests/ or existing L0 written, "
        "nothing staged in git.")

    out = dict(r2)
    out.update({
        "status": "partial", "summary": summary, "acceptance": acc, "products": products,
        "code_paths": list(r2["code_paths"]) + [f"dataprep_staging/w1_canal/{f}"
                                                 for f in REP_CODE],
        "key_numbers": key, "deviations_from_plan": dev, "issues_for_downstream": issues,
        "preregistration": list(r2["preregistration"]) + [
            "results/dataprep/preregistration/W1_janmay_benchmark_2026-10-12.record.json",
            "results/dataprep/preregistration/W1.ext2026_uncertainty.r1.json",
            "results/dataprep/preregistration/W1.ext2026_prospective_obs.v1.json"],
        "evidence": list(r2["evidence"]) + [
            "results/dataprep/reports/W1_evidence/rep_verify.json",
            "results/dataprep/reports/W1_evidence/rep_sigma_sensitivity.json",
            "results/dataprep/reports/W1_evidence/rep_provenance_addendum.json",
            "results/dataprep/surface_water_v4/full/v3/validation.json",
            "results/dataprep/surface_water_v4/full/v3/QC_REPORT.md"],
        "previous_report": ["results/dataprep/reports/W1.r2.md",
                            "results/dataprep/reports/W1.r2.json",
                            "results/dataprep/reports/W1.r1.md",
                            "results/dataprep/reports/W1.r1.json"],
        "repair_round": {
            "date": "2026-10-12", "verifier_major_issues": 2, "verifier_minor_issues": 8,
            "fixed": ["major 1 (full/v3)", "minor U1 (scores, prospective test)",
                      "minor dry-month wording", "minor ledger roles",
                      "minor provenance (full/v3 manifest; addendum for older versions)",
                      "minor sigma grid (post hoc sensitivity, reported)",
                      "minor key-number wording", "minor CODIS licence (checked, partial)"],
            "not_fixable": ["major 2: forward archive blocked; head check failure and weak "
                            "primary evidence are facts of the registered tests"],
            "disclosed_only": ["minor step-6 design timing", "minor design-time "
                               "contamination with post-origin canal totals"]}})

    # ------------------------------------------------------------------ ledger
    pilot = DATAPREP / "_pilots" / "surface_water_monthly"
    brief = rel_to_repo(C.BRIEFING_110_PDF)
    jj = rel_to_repo(C.JIJI_CSV)
    rows = []
    for name in ("preorigin", "full"):
        vdir = SW / name / "v1"
        for gl in ("registered", "extended"):
            n = f"W1.rep.sigma_sensitivity.{name}.{gl}"
            rows += [
                {"number_id": n, "source": brief, "role": "design_time_evidence",
                 "product_version": f"surface_water_v4/{name}/v1",
                 "note": "held-out Jan-May totals 2015-2021 (post hoc, reporting only)"},
                {"number_id": n, "source": jj, "role": "input",
                 "product_version": f"surface_water_v4/{name}/v1",
                 "note": "exact annual constraint of the state space; beta and scales"},
                {"number_id": n, "source": rel_to_repo(vdir / "inflow_composite_daily.parquet"),
                 "role": "input", "product_version": f"surface_water_v4/{name}/v1", "note": ""},
                {"number_id": n, "source": rel_to_repo(C.MOA_691_CSV), "role": "input",
                 "product_version": f"surface_water_v4/{name}/v1",
                 "note": "crop-1 share constraint 2012-2015"}]
    n = "W1.full_v3.rep.coverage_hist_statespace_janmay"
    rows += [{"number_id": n, "source": rel_to_repo(SW / "full" / "v1"
                                                     / "statespace_loo_coverage.csv"),
              "role": "input", "product_version": "surface_water_v4/full/v3",
              "note": "existing hits (10/14) re-evaluated with the A1 gate-2 interval"},
             {"number_id": n, "source": brief, "role": "design_time_evidence",
              "product_version": "surface_water_v4/full/v3",
              "note": "held-out Jan-May totals 2015-2021 behind the existing hits"}]
    for f in ("choushui_daily_q.csv", "jiji_janmay_supply_2015_2021.csv"):
        rows.append({"number_id": "W1.rep.provenance_addendum", "source": rel_to_repo(pilot / f),
                     "role": "input", "product_version": "surface_water_v4/full/v3",
                     "note": "hashed for the provenance addendum"})
    ledger.append_rows(C.WP, rows)
    allrows = ledger.read_all()
    n_conf = len(ledger.conflicts(allrows))
    out["ledger"] = {"fragment": "results/dataprep/ledger/rows/W1.csv",
                     "superseded": "results/dataprep/ledger/superseded/W1.r2.csv",
                     "unledgered_cited_numbers": [], "conflicts_all_fragments": n_conf}
    with open(REPORTS / "W1.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------------------------ markdown
    def section(title_start: str) -> str:
        lines = old_md.splitlines()
        i = next(k for k, s in enumerate(lines) if s.startswith(title_start))
        j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")),
                 len(lines))
        return "\n".join(lines[i:j]).rstrip()

    md = ["# WP W1: monthly canal deliveries, surface water v4 (report, 2026-10-12, updated in "
          "the repair round)", "", f"Status: **partial**. {summary[len('Status partial. '):]}",
          "", "## Products", ""]
    for pr in products:
        md.append(f"- `{pr['path']}` ({pr['layer']}); allowed: {pr['allowed_uses']}; "
                  f"fit_window_end {pr['fit_window_end']}.")
    md += ["", "## Acceptance", "", "| criterion | threshold | measured | pass |",
           "|---|---|---|---|"]
    for a in acc:
        md.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                  f"{'yes' if a['passed'] else 'NO'} |")
    md += ["", "## Repair round (2026-10-12 night): verifier findings and what was done", "",
           "An independent verifier confirmed every recomputed number of the earlier rounds and "
           "found 0 critical, 2 major and 8 minor issues. Nothing published was overwritten; "
           "the fixes are a new version full/v3, three A1 registrations, a ledger role "
           "correction and new evidence files. Files that published manifests or "
           "pre-registrations hash were not edited; the repair code is in new rep_*.py modules.",
           "", "| finding | action | result |", "|---|---|---|",
           "| MAJOR 1: the 2026 sd and draws of full/v2 were never coverage-checked or labelled "
           "(plan 4.7); sd measured against the flag-5 Denton series; 11 distinct trajectories; "
           "draw mean off the value by 0.169 / 0.337; structural-zero January with sd > 0 | "
           "declared error model W1.ext2026_uncertainty.r1 (residuals against the full/v1 "
           "state-space draws, whose annual sums are the observed totals; bias folded into the "
           "spread; t-inflation; smoothed bootstrap; centred; truncated at 0; structural zeros "
           "0) and a declared annual coverage rule; published full/v3 | "
           f"annual {ga['hits']}/{ga['n']}, A1 gate-2 {iv(ga['a1_gate2'])} -> annual scale "
           f"coverage-checked; 200 distinct trajectories; pre-truncation mean = value; every row "
           f"labelled |",
           "| MAJOR 2: WP incomplete (forward archive not started; head check failed; weak "
           "primary) | re-probed the feeds; nothing else can change a registered result | still "
           "blocked (503 / bot challenge / TLS); status partial |",
           "| minor: U1 exposure (prospective target and baseline admit flag 3/4; Denton-target "
           "RMSEs among the scores) | registered W1.ext2026_prospective_obs.v1 (observed months "
           "only); moved the Denton RMSEs to agreement_with_flag5_denton_not_scores | done |",
           "| minor: dry-month failure described as extrapolation | wording corrected "
           "(pooled-slope misspecification; 17 of 132 design-time fold months below the monthly "
           "minimum) | done |",
           "| minor: ledger roles | fragment rewritten once (superseded/W1.r2.csv kept) | 30 + 12 "
           "rows corrected, 0 conflicts |",
           "| minor: incomplete manifest provenance | full/v3 lists the reporting inputs; "
           "addendum hashes them for preorigin/v1 and full/v1-v2 | done (older versions not "
           "re-versioned: data unaffected) |",
           "| minor: step-6 design timing unverifiable | disclosed (no change possible) | "
           "disclosed |",
           "| minor: sigma grid truncated at 0.45 | post hoc sensitivity, grid to 1.0 | "
           f"pre-origin {sp['registered']['hits']}/14 -> {sp['extended']['hits']}/14, full "
           f"{sf['registered']['hits']}/14 -> {sf['extended']['hits']}/14 (reported only) |",
           "| minor: design-time contamination with post-origin totals | disclosed (as before) "
           "| disclosed |",
           "| minor: key-number wording; CODIS licence not re-checked | wording fixed; licence "
           "page checked | footer requires attribution; declaration page 403 |",
           "", "### Repaired 2026 series (full/v3; flag 6; recorded-forcing arm only)", "",
           "| month | Changhua value | sd | p05-p95 | Yunlin value | sd | p05-p95 |",
           "|---|---|---|---|---|---|---|"]
    months26 = pd.date_range("2026-01-01", periods=len(s26["changhua"]["value"]), freq="MS")
    for i, m in enumerate(months26):
        c, y = s26["changhua"], s26["yunlin"]
        md.append(f"| {m:%Y-%m} | {c['value'][i]:.3f} | {c['sd'][i]:.3f} | {c['p05'][i]:.3f}-"
                  f"{c['p95'][i]:.3f} | {y['value'][i]:.3f} | {y['sd'][i]:.3f} | "
                  f"{y['p05'][i]:.3f}-{y['p95'][i]:.3f} |")
    md += ["", f"Jan-Sep member p05/p50/p95: Changhua {'/'.join(f3(x) for x in jan_sep['changhua'])}"
           f", Yunlin {'/'.join(f3(x) for x in jan_sep['yunlin'])} x 10^8 m3 (values "
           f"{s26['changhua']['jan_sep_total']:.3f} / {s26['yunlin']['jan_sep_total']:.3f}).", "",
           "### Coverage (design-time units, double leave-one-year-out)", "",
           "| check | hits | coverage | A1 gate-2 interval | verdict | mean 90 % width |",
           "|---|---|---|---|---|---|"]
    for nm, g in (("annual district totals, 11 years x 2 (declared rule)", ga),
                  ("Jan-May 2017-2021 (reported)", gj),
                  ("inflow monthly, observed months (inflow sd label)", gi),
                  ("historical: full/v1 state-space Jan-May LOO 2015-2021 (row label)", gh)):
        md.append(f"| {nm} | {g['hits']}/{g['n']} | {g['coverage']:.3f} | {iv(g['a1_gate2'])} "
                  f"| {'pass' if g['a1_gate2']['pass'] else 'FAIL'} | "
                  + ", ".join(f"{k} {v:.2f}" for k, v in g["mean_width"].items()) + " |")
    md += ["", "Misses of the annual check: " + ", ".join(
        f"{m['year']} {m['district']} ({m['side']})" for m in ga["misses"]) + ".", "",
        f"Labels: 2026 rows '{rep['label_2026']}'; 2012-2025 rows '{rep['label_hist']}'; "
        "structural-zero rows 'structural zero (winter weight 0): draws and sd are 0; the "
        "winter-0.5 alternative is in structural_alt_value'.", "",
        "Interpretation (written after the results were known): the annual-scale error model "
        "is not contradicted by the 11 design-time years, but the A1 interval is wide at this "
        "n, so this is weak evidence. The intervals are wide (mean annual 90 % width "
        + ", ".join(f"{k} {v:.2f}" for k, v in ga["mean_width"].items())
        + " x 10^8 m3), as the chain's annual error requires. The monthly split of the 2026 "
        "intervals rests on the state-space draws' monthly pattern and is not checked.", ""]
    md += [section("## Jan-May benchmark, pre-origin variant"), "",
           section("## Jan-May benchmark, full variant"), "",
           section("## Resume round").replace(
               "| 2026 month | Changhua value | sd |",
               "(The sd columns below are full/v2's, superseded by the repair round.)\n\n"
               "| 2026 month | Changhua value | sd |").replace(
               "its dry-month extrapolation is the weak point (see issues)",
               "its dry-month behaviour is the weak point (repair-round wording: a pooled-slope "
               "misspecification, not extrapolation; see issues)"), "",
           "## Deviations from the plan", ""]
    md += [f"- {d}" for d in dev]
    md += ["", "## Issues for downstream WPs", ""] + [f"- {i}" for i in issues]
    md += ["", "## Key numbers", ""] + [f"- {k}" for k in key]
    md += ["", "## Provenance", "",
           "- Pre-registrations and declarations: " + "; ".join(
               f"`{p}`" for p in out["preregistration"])
           + "; looks in `results/dataprep/preregistration/looks.jsonl`.",
           f"- Ledger: `results/dataprep/ledger/rows/W1.csv` (role correction; previous fragment "
           f"`results/dataprep/ledger/superseded/W1.r2.csv`); conflicts across all fragments: "
           f"{n_conf}.",
           "- Evidence: " + ", ".join(f"`{e}`" for e in out["evidence"]),
           "- Previous reports: `results/dataprep/reports/W1.r2.md` / `W1.r2.json` (resume "
           "round), `W1.r1.md` / `W1.r1.json` (first build).",
           "- Code: " + ", ".join(f"`{c}`" for c in out["code_paths"]),
           "- Claude API tokens: 0. GPU: none. Nothing under hydrophysics/, tests/ or existing L0 "
           "written; nothing staged in git."]
    (REPORTS / "W1.md").write_text("\n".join(md) + "\n")
    print("wrote", REPORTS / "W1.md", REPORTS / "W1.json", "ledger conflicts", n_conf)


if __name__ == "__main__":
    main()
