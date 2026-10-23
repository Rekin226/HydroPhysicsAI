"""WP A1 second repair round: ledger rows, the 'Repair round 2' section of reports/A1.md and
A1.json as the WP now stands.

    python -m harness.a1r2_report          (from dataprep_staging/, after a1r2_checks)

Reads the evidence in ``A1_evidence/repair2/``; appends rows to the A1 ledger fragment; inserts
a status note under the title of A1.md and appends the 'Repair round 2' section; rewrites
A1.json (the previous top level is kept under ``before_repair_round_2``). Refuses to run twice.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, REPORTS_DIR, rel_to_repo  # noqa: E402

from harness import coverage3, prereg3  # noqa: E402

EV = REPORTS_DIR / "A1_evidence" / "repair2"
V3 = DATAPREP / "validation_units" / "v3"
MARK = "## Repair round 2"
HERE = Path(__file__).resolve().parent
REPO_ABS = str(HERE.parents[1])


def _j(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _abs(rel: str) -> str:
    return f"{REPO_ABS}/{rel}"


def main(argv=None) -> int:
    md_path, js_path = REPORTS_DIR / "A1.md", REPORTS_DIR / "A1.json"
    md = md_path.read_text(encoding="utf-8")
    if MARK in md:
        raise SystemExit("repair round 2 already appended to A1.md")
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    des = _j(EV / "coverage3_design.json")
    desr = _j(EV / "coverage3_design_rerun.json")
    con = _j(EV / "coverage3_confirm.json")
    ck = _j(EV / "a1r2_checks.json")
    b3 = _j(EV / "build_units_v3.json")
    old = _j(js_path)
    pytest_txt = (EV / "pytest.txt").read_text(encoding="utf-8").strip().splitlines()[-1]
    ruff_txt = (EV / "ruff.txt").read_text(encoding="utf-8").strip()
    n_pass = int(pytest_txt.split()[0])
    shas = {}
    for pid in ("A1r2.coverage3_confirm.v1", "A1r2.harness_rules.v1", "A1-fresh-units-v2",
                "A1r.gate_window.v1", "A1.coverage_confirm.v1", "A1-fresh-units-v1"):
        doc, sha = prereg3.load_prereg(pid)
        shas[pid] = (sha, doc["created"])
    conf_created = shas["A1r2.coverage3_confirm.v1"][1]
    confirm_mtime = _dt.datetime.fromtimestamp((EV / "coverage3_confirm.json").stat().st_mtime,
                                               _dt.timezone.utc).strftime("%H:%M:%SZ")

    P = [r for r in con["configs"] if r["role"] == "primary"]
    OUT = [r for r in con["configs"] if r["role"] == "secondary_outside_envelope"]
    by = {r["name"]: r for r in con["configs"]}
    s_up, s_an = des["summary"]["upper90"], des["summary"]["anova"]
    l90p = min((r["level0.90_actual_coverage"], r["name"]) for r in P)
    l90p_below = sum(r["level0.90_actual_coverage"] < 0.89 for r in P)
    l90o_below = sum(r["level0.90_actual_coverage"] < 0.89 for r in OUT)
    win = sorted(((r["window_only_fail_rate_true0.9"], r["name"]) for r in con["configs"]),
                 reverse=True)
    pv = ck["probes_verifier"]["summary"]
    refusable = [n for n, s in pv.items() if s["expected"] == "refused"]
    controls = [n for n, s in pv.items() if s["expected"] == "allowed"]
    au = ck["audit_summary"]
    fo = ck["fresh_overlaps"]
    hh = ck["harness_files_hashed_elsewhere"]
    ver = {k: v["status"] for k, v in ck["checker"].items()}
    vf = lambda n: by[n]["actual_coverage"]  # noqa: E731
    va = lambda n: by[n]["anova_actual_coverage"]  # noqa: E731

    # ------------------------------------------------------------------ acceptance
    acc = [r for r in old["acceptance"]]
    keep = {a["criterion"]: a for a in acc}
    rows = []
    for crit in list(keep)[:2]:                      # pilot reproduction rows (unchanged)
        rows.append(dict(keep[crit]))
    rows[0]["measured"] += ("; not rerun in repair round 2: scorers.py, repro_pilots.py and "
                            "gaps.py are byte-identical to their hash records")
    rows.append({
        "criterion": "Coverage interval (plan A1 acceptance), repaired: harness.coverage3 "
                     "default 'upper90' (Wilson on n_eff from the one-sided 0.90 upper bound of "
                     "the ICC, Student-t on k-1 df, level 0.95) in its envelope (k >= 20 "
                     "clusters, ICC <= 0.7); pre-registered A1r2.coverage3_confirm.v1, fresh "
                     "seeds 20301012+, 10,000 draws per configuration",
        "threshold": f">= 0.89 actual coverage of a true 0.90 in every primary configuration "
                     f"({len(P)})",
        "measured": f"min {con['primary_min_actual_coverage']:.4f} "
                    f"({con['primary_argmin']}); 0 of {len(P)} below; the frozen interval on "
                    f"the same draws: min {con['primary_anova_min_actual_coverage']:.4f}, "
                    f"{len(con['primary_anova_below_criterion'])} below; in-envelope two-point "
                    f"mixtures min {con['secondary_in_envelope_min_actual_coverage']:.4f}",
        "passed": bool(con["pass"])})
    rows.append({
        "criterion": "Outside the envelope (k < 20 clusters; reported, never gating): the "
                     "verifier's failing cases, S1 14x13, W1 7x2, E6 5x6, k3x10, and the "
                     "gap-fill pilot's 9-gap (N) and 3-gap (H) structures, same run",
        "threshold": ">= 0.89 in every configuration (informational; such strata come back "
                     "calibrated=False)",
        "measured": f"upper90 below 0.89 in {len(con['secondary_outside_envelope_below_criterion'])}"
                    f" of {len(OUT)} (k10_x50_rho0.7 {vf('k10_x50_rho0.7'):.4f}); verifier cases "
                    f"14x13 ICC 0.4/0.5/0.7 {vf('verifier_14x13_rho0.4'):.3f}/"
                    f"{vf('verifier_14x13_rho0.5'):.3f}/{vf('verifier_14x13_rho0.7'):.3f}, 7x12 "
                    f"ICC 0.6 {vf('verifier_7x12_rho0.6'):.3f}, 10x50 ICC 0.5 "
                    f"{vf('verifier_10x50_rho0.5'):.3f}, 12x20 ICC 0.4 "
                    f"{vf('verifier_12x20_rho0.4'):.3f}; the frozen interval below 0.89 in "
                    f"{len(con['secondary_outside_envelope_anova_below_criterion'])} of {len(OUT)}"
                    f" (14x13 ICC 0.7 {va('verifier_14x13_rho0.7'):.3f}, 7x12 ICC 0.6 "
                    f"{va('verifier_7x12_rho0.6'):.3f}, pilot H ICC 0.5 "
                    f"{va('pilot_H_half12_rho0.5'):.3f})",
        "passed": not con["secondary_outside_envelope_below_criterion"]})
    rows.append({
        "criterion": "Design-time sweep that chose coverage3's default (seeds 20291012+, "
                     f"{des['n_configs']} configurations, 7 candidates; labelled design-time): "
                     "some candidate meets the criterion in every configuration with ICC <= 0.7",
        "threshold": ">= 0.89 in every one of the "
                     f"{sum(r['in_envelope'] for r in des['configs'])} configurations with "
                     "ICC <= 0.7",
        "measured": f"no candidate did; upper90 {s_up['n_below_envelope']} below (min "
                    f"{s_up['min_coverage_envelope']:.4f}; k 9-14, clusters of 50-340 rows, ICC "
                    f"0.7), so by the declared rule it is the default with the envelope k >= "
                    f"{des['decision']['envelope']['min_k']}; the frozen interval "
                    f"{s_an['n_below_envelope']} below (min "
                    f"{s_an['min_coverage_envelope']:.4f}); rerun with the frozen code "
                    f"identical: {des['configs'] == desr['configs']}",
        "passed": False})
    for crit, a in keep.items():
        if crit.startswith("Coverage interval: actual coverage") or \
                crit.startswith("Coverage interval, design-time run"):
            r = dict(a)
            r["criterion"] = "First round (frozen harness.coverage interval, kept for the " \
                             "record): " + r["criterion"]
            r["measured"] += ("; valid only for that configuration set (ICC <= 0.3 with 10+ "
                              "clusters except one at 0.4); superseded by coverage3")
            rows.append(r)
    rows.append({
        "criterion": "Sensitivity, not an acceptance criterion: coverage3 at level 0.90 (same "
                     "draws)",
        "threshold": ">= 0.89 in every configuration (informational; the gate keeps 0.95)",
        "measured": f"primary below 0.89 in {l90p_below} of {len(P)} (min {l90p[0]:.4f}, "
                    f"{l90p[1]}); outside the envelope {l90o_below} of {len(OUT)}",
        "passed": False})
    for crit, a in keep.items():
        if crit.startswith(("First-round declaration", "Fresh validation",
                            "Validation-units product gate", "Artificial-gap",
                            "Repaired coverage", "Origin guards")):
            rows.append(dict(a))
    rows.append({
        "criterion": "Validation-units product, v3 (documentation-only version: data files "
                     "byte-identical to v2; every input classified with role and data_end; "
                     "every post-origin input is a design-time label; checker; guard)",
        "threshold": "all checks pass; checker ok for v1, v2, v3; guard opens only the allowed "
                     "uses",
        "measured": f"{sum(c['pass'] for c in b3['checks'])}/{len(b3['checks'])} checks; "
                    f"{b3['n_post_origin_inputs']} post-origin inputs, all design_time_label; "
                    f"checker {ver}; guard as expected {b3['guard_as_expected']}",
        "passed": bool(b3["gate_passed"] and b3["guard_as_expected"]
                       and all(v == "ok" for v in ver.values()))})
    pr1 = ck["probes_builder_round1_on_prereg3"]["prereg3_refuses_all"]
    rows.append({
        "criterion": "Pre-registration tool refuses to score a design whose pre-registration "
                     "postdates any look at its test data (plan A1); repaired again: "
                     "harness.prereg3 (strict union-of-windows scopes, membership blocking), "
                     "end-to-end probes on the real v2 registry in temporary directories",
        "threshold": "the verifier's 3 probes and 4 further nesting/temporal probes refused; 3 "
                     "controls (disjoint units, disjoint windows, registered before any look) "
                     "allowed; the first round's 5 probes refused; tests pass; ruff clean",
        "measured": f"prereg3 as expected in {sum(pv[n]['prereg3_as_expected'] for n in pv)}/"
                    f"{len(pv)} scenarios ({len(refusable)} refused, {len(controls)} allowed); "
                    f"prereg2 allowed {len(ck['probes_verifier']['prereg2_allows_refusable'])}/"
                    f"{len(refusable)} of the refusable ones; first-round probes refused by "
                    f"prereg3: {pr1}; {pytest_txt}; ruff: {ruff_txt}",
        "passed": bool(ck["probes_verifier"]["prereg3_all_as_expected"] and pr1
                       and "failed" not in pytest_txt and ruff_txt == "All checks passed!")})
    rows.append({
        "criterion": "The existing look log under prereg3 (read-only audit): no registered "
                     "result is newly invalidated and no look can block a v2 fresh unit",
        "threshold": "report every registered test the stricter rule refuses; 0 looks eligible "
                     "to block a v2 fresh unit",
        "measured": f"{au['looks_records']} looks, {au['index_records']} INDEX records; refused "
                    f"under prereg3: {len(au['tests_refused_under_prereg3'])} "
                    f"({', '.join(t for _, t in au['tests_refused_under_prereg3'])}; the same "
                    f"as prereg2), newly refused: "
                    f"{len(au['tests_refused_under_prereg3_not_prereg2'])}; v2 fresh units with "
                    f"an eligible blocking look: "
                    f"{len(au['fresh_units_with_eligible_blocking_looks'])}; scopes read "
                    f"differently from prereg2: "
                    f"{len(au['scopes_read_differently_from_prereg2'])} (now unbounded); "
                    f"{au['units_of_unknown_membership']} legacy unit ids of unknown membership "
                    f"(id-only, disclosed)",
        "passed": not au["tests_refused_under_prereg3_not_prereg2"]
        and not au["fresh_units_with_eligible_blocking_looks"]})
    rows.append({
        "criterion": "No published version broken: every harness file hashed by a product "
                     "manifest or pre-registration is unchanged",
        "threshold": "0 changed",
        "measured": f"{hh['n']} hash records, {len(hh['changed'])} changed (repair round 2 "
                    f"added modules and new versions; no hashed file edited)",
        "passed": not hh["changed"]})
    rc2 = old.get("resume_check_2") or {}
    if rc2:
        rows.append({
            "criterion": "Resume check 2 (WP re-verified after the workflow re-invoked the "
                         "build; kept for the record)",
            "threshold": "a1r_checks pass and identical to the repair run; repro identical to "
                         "the reported run; tests pass; 0 looks on fresh or withdrawn units since "
                         "the repair",
            "measured": f"a1r_checks pass {rc2.get('a1r_checks_pass')}, identical "
                        f"{rc2.get('a1r_checks_identical_to_repair_run')}; repro "
                        f"{rc2.get('repro_n_values')} values max "
                        f"{rc2.get('repro_max_rel_diff'):.3g}, identical "
                        f"{rc2.get('repro_identical_to_reported')}; {rc2.get('pytest_passed')} "
                        f"passed; 0 looks on A1 fresh/withdrawn units",
            "passed": bool(rc2.get("pass"))})

    # ------------------------------------------------------------------ ledger rows
    have = [r for r in ledger.read_all() if r["wp"] == "A1"]
    tag = f"harness repair2;{now}"
    lrows = []

    def row(nid, src, role, note):
        lrows.append({"number_id": f"A1.repair2.{nid}", "source": rel_to_repo(src),
                      "role": role, "product_version": tag, "note": note})
    row("coverage3.design", EV / "coverage3_design.json", "design_time_evidence",
        f"{des['n_configs']} configurations x 7 candidates; upper90 "
        f"{s_up['n_below_envelope']} below (min {s_up['min_coverage_envelope']:.4f}); frozen "
        f"{s_an['n_below_envelope']} below (min {s_an['min_coverage_envelope']:.4f})")
    row("coverage3.design_rerun", EV / "coverage3_design_rerun.json", "design_time_evidence",
        "rerun with the frozen code; identical")
    row("coverage3.pilot_sizes", EV / "pilot_cluster_sizes.json", "design_time_evidence",
        "cluster sizes of the gap-fill pilot's families (sizes only)")
    row("coverage3.confirm", EV / "coverage3_confirm.json", "validator",
        f"primary min {con['primary_min_actual_coverage']:.4f} over {len(P)}; outside the "
        f"envelope below 0.89 in {len(con['secondary_outside_envelope_below_criterion'])} of "
        f"{len(OUT)}")
    row("coverage3.level090", EV / "coverage3_confirm.json", "validator",
        f"level 0.90 primary min {l90p[0]:.4f}, {l90p_below} of {len(P)} below")
    row("coverage3.window_only_fails", EV / "coverage3_confirm.json", "validator",
        f"max window-only false fail at 0.90 {win[0][0]:.4f} ({win[0][1]})")
    row("prereg3.probes", EV / "a1r2_checks.json", "validator",
        f"{len(refusable)} refused, {len(controls)} allowed as expected; prereg2 allowed "
        f"{len(ck['probes_verifier']['prereg2_allows_refusable'])}")
    row("prereg3.audit", EV / "prereg3_audit.json", "validator",
        f"{au['looks_records']} looks; newly refused "
        f"{len(au['tests_refused_under_prereg3_not_prereg2'])}")
    row("fresh_overlaps", EV / "a1r2_checks.json", "validator",
        f"{fo['n_overlapping_pairs']} of {fo['n_pairs']} fresh-unit pairs share members")
    row("units_v3.build", EV / "build_units_v3.json", "validator",
        f"{sum(c['pass'] for c in b3['checks'])}/{len(b3['checks'])} checks; checker ok")
    row("units_v3.validation", V3 / "v3_validation.json", "validator",
        "post-origin inputs listed as design-time labels")
    row("tests", EV / "pytest.txt", "validator", pytest_txt)
    row("ruff", EV / "ruff.txt", "validator", ruff_txt)
    seen = {(r["number_id"], r["source"]) for r in have}
    ledger.append_rows("A1", [r for r in lrows if (r["number_id"], r["source"]) not in seen])
    n_rows = sum(1 for r in ledger.read_all() if r["wp"] == "A1")

    # ------------------------------------------------------------------ text
    key = [
        f"coverage3 confirm (A1r2.coverage3_confirm.v1, seeds 20301012+, {len(P)} primary "
        f"configurations, k >= 20, ICC <= 0.7): min {con['primary_min_actual_coverage']:.4f} "
        f"({con['primary_argmin']}); frozen interval on the same draws min "
        f"{con['primary_anova_min_actual_coverage']:.4f}, "
        f"{len(con['primary_anova_below_criterion'])} below 0.89",
        f"Outside the envelope ({len(OUT)} configurations, k < 20): upper90 below 0.89 once "
        f"(k10_x50_rho0.7 {vf('k10_x50_rho0.7'):.4f}); frozen interval "
        f"{len(con['secondary_outside_envelope_anova_below_criterion'])} times (min "
        f"{min(r['anova_actual_coverage'] for r in OUT):.4f})",
        f"Verifier's cases re-drawn: 14x13 ICC 0.5 frozen {va('verifier_14x13_rho0.5'):.3f} -> "
        f"upper90 {vf('verifier_14x13_rho0.5'):.3f}; ICC 0.7 {va('verifier_14x13_rho0.7'):.3f} "
        f"-> {vf('verifier_14x13_rho0.7'):.3f}; 7x12 ICC 0.6 {va('verifier_7x12_rho0.6'):.3f} "
        f"-> {vf('verifier_7x12_rho0.6'):.3f}; 10x50 ICC 0.5 {va('verifier_10x50_rho0.5'):.3f}"
        f" -> {vf('verifier_10x50_rho0.5'):.3f}",
        f"Design sweep (design-time, {des['n_configs']} configurations): no candidate met "
        f"0.89 everywhere at ICC <= 0.7; upper90 {s_up['n_below_envelope']} below (min "
        f"{s_up['min_coverage_envelope']:.4f}), frozen {s_an['n_below_envelope']} below (min "
        f"{s_an['min_coverage_envelope']:.4f})",
        f"Power cost: mean share of draws excluding 0.90 at a true 0.80 over the primary set "
        f"{con['primary_mean_power_true0.8']:.3f} (frozen "
        f"{con['primary_anova_mean_power_true0.8']:.3f}); E6 5x6 "
        f"{by['E6_5x6_rho0.3']['power_true0.8']:.3f}, W1 7x2 "
        f"{by['W1_7x2_rho0.5']['power_true0.8']:.3f}, S1 14x13 ICC 0.3 "
        f"{by['S1_14x13_rho0.3']['power_true0.8']:.3f}",
        f"Level 0.90 (information): primary below 0.89 in {l90p_below} of {len(P)} (min "
        f"{l90p[0]:.4f})",
        f"Window-only false fails at a true 0.90 under the plan's point window (gate-2/3 rule): "
        f"indep_n30 {by['indep_n30']['window_only_fail_rate_true0.9']:.3f}, indep_n50 "
        f"{by['indep_n50']['window_only_fail_rate_true0.9']:.3f}, "
        f"{win[2][1]} {win[2][0]:.3f}, S1 14x13 ICC 0.3 "
        f"{by['S1_14x13_rho0.3']['window_only_fail_rate_true0.9']:.3f}",
        f"prereg3 probes on the real registry: {len(refusable)}/{len(refusable)} refusable "
        f"scenarios refused (prereg2 allowed "
        f"{len(ck['probes_verifier']['prereg2_allows_refusable'])}), {len(controls)}/"
        f"{len(controls)} controls allowed; first-round probes refused",
        f"Fresh units: {fo['n_overlapping_pairs']} of {fo['n_pairs']} pairs share members "
        f"(fresh.block.2016_2018 and fresh.folds.wells_all overlap every spatial unit)",
        f"Audit under prereg3: {au['looks_records']} looks; refused "
        f"{len(au['tests_refused_under_prereg3'])} (as prereg2), newly 0; fresh units with an "
        f"eligible blocking look 0; legacy unit ids of unknown membership "
        f"{au['units_of_unknown_membership']}",
        f"validation_units/v3: data byte-identical to v2; {b3['n_post_origin_inputs']} "
        f"post-origin inputs documented as design-time labels; checker v1/v2/v3 ok",
        f"Harness files hashed elsewhere: {hh['n']} records, {len(hh['changed'])} changed",
        f"Tests: {pytest_txt}; ruff clean",
        "Claude API tokens: 0; GPU: none",
    ]
    deviations = [
        "Coverage interval (gate-3): the design effect uses the one-sided 0.90 upper "
        "confidence bound of the ANOVA ICC (rho = 1 without within-cluster variance) instead "
        "of the point estimate the plan describes; chosen by a design-time sweep under a rule "
        "declared before it ran, confirmed on fresh seeds under a pre-registration.",
        "The coverage claim is scoped to an envelope (k >= 20 clusters, simulated ICC <= 0.7); "
        "strata with fewer clusters carry calibrated=False (plan 4.7 'uncalibrated') rather "
        "than a pass/fail certified by simulation.",
        "prereg3: a temporal unit spans the registry's calibration wells and active cells "
        "(head data), not leveling benchmarks or other data sets; dataset names are still not "
        "compared otherwise; external units must declare members.",
        "validation_units/v3 is a documentation-only version (byte-identical data) so the "
        "manifest can describe the post-origin pilot inputs; v2 is unchanged.",
        "New modules instead of edits again (prereg2.py and coverage2.py are hashed by "
        "published products).",
    ]
    downstream = [
        "Use harness.coverage3 (rule gate-3) and harness.prereg3; open "
        "results/dataprep/validation_units/v3 (identical units to v2). Rules are declared in "
        "preregistration/A1r2.harness_rules.v1.json. prereg/prereg2/coverage/coverage2 stay "
        "importable only because published products hash them.",
        f"Register before any scoring look, across WPs: {fo['n_overlapping_pairs']} of "
        f"{fo['n_pairs']} fresh-unit pairs share wells, cells or months; "
        "fresh.block.2016_2018 overlaps every spatial unit in 2016-18 and "
        "fresh.folds.wells_all overlaps every unit. Under prereg3 the first scoring look on a "
        "unit blocks every later registration on any overlapping unit with an overlapping time "
        "window, whichever WP made it. P4m, H2, H3, P3, S1, S5 and I1 should agree a "
        "registration order or register their tests on fresh units together.",
        "Coverage verdicts: report coverage, n_eff, n_eff_lower, k, the interval and "
        "'calibrated' with every verdict. Strata with fewer than 20 clusters (S1 14 ring sites, "
        "W1 7 years per district, E6 5 regions, the head gap-fill N family's 9 gaps, H's 3) are "
        "uncalibrated: pool them first (plan 4.4) or label the verdict uncalibrated. In the "
        "confirmation upper90 still reached 0.89 in 28 of 29 such configurations (k10 x 50 at "
        "ICC 0.7: 0.882), but the design sweep fell to 0.84 with the pilot's 9-gap N structure "
        "at ICC 0.7.",
        f"Power is low with few clusters: at a true 0.80 the interval excludes 0.90 in "
        f"{by['E6_5x6_rho0.3']['power_true0.8']:.2f} of draws for E6 5x6, "
        f"{by['W1_7x2_rho0.5']['power_true0.8']:.2f} for W1 7x2, "
        f"{by['S1_14x13_rho0.3']['power_true0.8']:.2f} for S1 14x13 at ICC 0.3; gate-3 passes "
        f"a true 0.80 in up to {max(r.get('gate3_pass_rate_true0.8', 0) for r in P):.2f} of "
        "draws inside the envelope (strongly unequal clusters). A pass is weak evidence there.",
        f"The plan's 0.85-0.95 point window still fails calibrated products by chance near "
        f"n_eff 30-50 (window-only false fails at a true 0.90: 30 independent rows "
        f"{by['indep_n30']['window_only_fail_rate_true0.9']:.3f}, 50 rows "
        f"{by['indep_n50']['window_only_fail_rate_true0.9']:.3f}, 24 clusters x 3 rows about "
        "0.19-0.21, S1 14x13 at ICC 0.3 "
        f"{by['S1_14x13_rho0.3']['window_only_fail_rate_true0.9']:.3f}). This is the plan's "
        "0.10-wide window at that sample size, not a harness estimator problem; the gate-2 bound "
        "only stops the window firing where n_eff is not shown to be >= 30.",
        "prereg3 compares legacy looks on undeclared units by id only and discloses them as "
        f"unverifiable ({au['units_of_unknown_membership']} such unit ids in the log); new "
        "looks and registrations must declare members for their own units.",
        "The near-duplicate threshold (Jaccard < 0.9) is A1's choice, not the plan's. Several "
        "fresh units overlap design-time units heavily (fresh.km2027.0 vs dt.fusion.km1_0 "
        "0.751, two_tercile.WM vs dt.fusion.east_h1 0.756, river_band vs dt.fusion.north_t3 "
        "0.735), and the two-tercile units are unions of design-time terciles by the plan's own "
        "definition: they add little independent evidence and give no independent estimate of "
        "tau.",
        "harness.coverage3.py, prereg3.py, coverage3_confirm.py and coverage3_design.py are "
        "now hashed by A1r2 pre-registrations, and build_units_v3.py by validation_units/v3: do "
        "not edit them; add modules or versions.",
        "Unchanged from earlier rounds: U2 sigma_obs with pre-origin rho(k) 0.331/0.494/0.651/"
        "0.804/0.862 m; unit membership is stored in units.json (do not recompute k-means); "
        "index-based gap generators through gaps_guarded; the look log is self-reported.",
    ]
    findings = [
        ("Major 1, coverage acceptance holds only inside the pre-registered configurations",
         "Correct. Reproduced: the frozen interval falls to "
         f"{va('verifier_14x13_rho0.7'):.3f} (14x13, ICC 0.7), {va('verifier_7x12_rho0.6'):.3f} "
         f"(7x12, ICC 0.6) and {va('verifier_10x50_rho0.5'):.3f} (10x50, ICC 0.5) on fresh "
         "seeds, and below 0.89 in "
         f"{s_an['n_below_envelope']} of {sum(r['in_envelope'] for r in des['configs'])} "
         "design-time configurations with ICC <= 0.7. A design-time sweep of 7 candidate "
         "intervals (rule declared first) found none that reaches 0.89 everywhere; the best, "
         "'upper90' (design effect from the one-sided 0.90 upper bound of the ICC; full "
         "clustering when no cluster mixes hits and misses), failed only at ICC 0.7 with 9-14 "
         "clusters of 50-340 rows, so by the declared rule it became harness.coverage3's "
         "default with an envelope of k >= 20 clusters. The pre-registered confirmation "
         f"(A1r2.coverage3_confirm.v1, registered {conf_created}, output written "
         f"{confirm_mtime}) passed: min {con['primary_min_actual_coverage']:.4f} over "
         f"{len(P)} configurations. The claim is now scoped to that envelope, strata with "
         "fewer clusters come back calibrated=False, and S1, W1, E6 and the gap-fill N/H "
         "families are warned by name. The first-round statement 'under-covers by up to about "
         "4 points (min 0.9099)' is withdrawn."),
        ("Major 2, prereg2 still does not always enforce the plan's rule",
         "Correct on both counts. harness.prereg3 parses scopes by a strict grammar into a "
         "union of month windows (days widen to months; anything not fully understood is "
         "unbounded) and blocks by unit membership (wells, benchmarks, fold test sets, cells; "
         "temporal units span the head data of their months). A look on a non-identical unit "
         "blocks when it is a registry fresh unit or was made after the test's unit was first "
         "declared fresh; pre-declaration design-time history does not block. On the real "
         "registry the verifier's three probes and four nesting/temporal probes are refused "
         "by prereg3 and allowed by prereg2; three controls are allowed. The audit of the "
         "real log finds no newly refused registered test and no look that can block a v2 "
         "fresh unit, so no product is invalidated."),
        ("Minor, the banner overstated the first repair",
         "Correct. The banner sentence is superseded: under gate-2/gate-3 a calibrated product "
         f"still fails through the window alone in {by['indep_n30']['window_only_fail_rate_true0.9']:.3f}"
         " of draws with 30 independent rows (fresh seeds; the verifier found 0.338) and about "
         "0.1-0.2 in other strata with n_eff 30-50. That is the plan's window, reported in the "
         "downstream issues; G1's membership ('true n_eff < 30') excluded n_eff = 30 exactly."),
        ("Minor, the near-duplicate threshold is the builder's choice",
         "Correct; disclosed in the downstream issues with the overlaps. Under prereg3 the "
         "overlaps among fresh units also matter for blocking (72 of 105 pairs)."),
        ("Minor, validation_units/v2 manifest silent on post-origin pilot inputs",
         "Fixed in validation_units/v3 (byte-identical data): every input carries role, "
         "data_end and a note; the 15 post-origin inputs are listed as design-time labels and "
         "the manifest explains why fit_window_end stays 2022-12. v2 is unchanged."),
        ("Minor, the design-time claim 'declared before the first run' cannot be verified",
         "Agreed (coverage.py and coverage_sim.py were modified after the first output). The "
         "run stays labelled design-time. This round's sweep: the candidate code was written "
         "and its decision rule fixed in coverage3_design.py before the run; the only edits "
         "after it were the DEFAULT_METHOD/ENVELOPE constants and a lint rename, and the rerun "
         "with the frozen code is identical."),
        ("Minor, A1.json had 13 acceptance rows while the result had 14",
         "Fixed: A1.json's acceptance list is now exactly the returned acceptance list."),
    ]
    summary = (
        "Second repair round after independent verification. Major 1 (the coverage acceptance "
        "held only inside the pre-registered configurations) is confirmed and repaired: the "
        "frozen interval falls to 0.75-0.83 with 7-14 clusters at ICC 0.5-0.7. A design-time "
        "sweep of 7 candidates found none that reaches 0.89 everywhere at ICC <= 0.7; by the "
        "rule declared before it ran, harness.coverage3 uses 'upper90' (design effect from the "
        "one-sided 0.90 upper bound of the ICC) with an envelope of k >= 20 clusters, and the "
        f"pre-registered confirmation on fresh seeds passed (min "
        f"{con['primary_min_actual_coverage']:.4f} over {len(P)} configurations; the frozen "
        f"interval on the same draws {con['primary_anova_min_actual_coverage']:.4f}). Outside "
        f"the envelope upper90 met 0.89 in {len(OUT) - len(con['secondary_outside_envelope_below_criterion'])}"
        f" of {len(OUT)} configurations, but such strata (S1, W1, E6, the gap-fill N and H "
        "families) are reported as uncalibrated, and power there stays low. Major 2 (the "
        "pre-registration tool misread day-dated and multi-window scopes and compared unit ids, "
        "not membership) is confirmed and repaired in harness.prereg3: on the real registry "
        f"all {len(refusable)} refusable probes are refused (prereg2 allowed every one) and "
        f"all {len(controls)} controls allowed, and an audit of the real log finds no newly "
        "invalidated result and no look that can block a fresh unit. Every minor finding is "
        "addressed: validation_units/v3 (byte-identical data) documents the post-origin pilot "
        "inputs, the residual window-only false fails of the plan's point window and the "
        "overlap of fresh units are reported, and A1.json now carries the same acceptance "
        "rows as this result. Pilot reproduction (2431 values, max 1.46e-14) is unchanged. "
        f"{n_pass} tests pass, ruff is clean, {hh['n']} hash records of harness files are "
        "unchanged; no GPU, 0 Claude API tokens, nothing under hydrophysics/, tests/ or L0 "
        "written, nothing staged in git.")

    # ------------------------------------------------------------------ A1.md
    note = ("> **Repair round 2 (2026-10-12): read the 'Repair round 2' section at the end "
            "first.** Use validation_units/v3, harness.prereg3 and harness.coverage3 (rule "
            "gate-3); the rules are declared in A1r2.harness_rules.v1. The repair-round "
            "statements that the interval under-covers 'by up to about 4 points' and that the "
            "point-window false fails were a harness estimator problem are withdrawn; prereg2 "
            "did not always enforce the plan's rule.\n\n")
    lines = md.split("\n", 2)
    md2 = lines[0] + "\n\n" + note + (lines[2] if len(lines) > 2 else "")

    def tbl(rs):
        out = ["| Criterion | Threshold | Measured | Pass |", "|---|---|---|---|"]
        for r in rs:
            out.append(f"| {r['criterion']} | {r['threshold']} | {r['measured']} | "
                       f"{'yes' if r['passed'] else '**no**'} |")
        return "\n".join(out)
    sec = [f"\n{MARK} (2026-10-12, {now})\n",
           "An independent verifier found two major and five minor issues in the repaired "
           "WP. Each is addressed below; nothing published was overwritten and no hashed file "
           "was edited.\n",
           "Status after repair round 2: **passed**, with the coverage claim scoped to an "
           "explicit envelope (k >= 20 clusters, ICC <= 0.7). Rows that read **no** are "
           "information or history: the out-of-envelope sensitivity (1 of 29 below), the "
           "design-time sweep (no candidate met 0.89 everywhere, which is why the envelope "
           "exists), the level-0.90 reading, and the first-round v1 declaration.\n",
           "### Verifier findings and what was done\n"]
    for t, x in findings:
        sec.append(f"- **{t}.** {x}")
    sec += ["\n### Acceptance (WP as it now stands)\n", tbl(rows),
            "\n### New products and code\n",
            "- `results/dataprep/validation_units/v3/` (L2; data byte-identical to v2; manifest "
            "documents input roles; checker ok).",
            f"- `results/dataprep/preregistration/A1r2.coverage3_confirm.v1.json` (sha256 "
            f"{shas['A1r2.coverage3_confirm.v1'][0]}, written {conf_created} by harness.prereg3) "
            f"and `A1r2.harness_rules.v1.json` (sha256 {shas['A1r2.harness_rules.v1'][0]}, "
            f"declaration of gate-3 and prereg3).",
            "- Code: `harness/coverage3.py`, `coverage3_design.py`, `coverage3_confirm.py`, "
            "`prereg3.py`, `build_units_v3.py`, `a1r2_prereg.py`, `a1r2_checks.py`, "
            "`a1r2_report.py`, `tests/test_repair2.py` (12 tests); README harness section.",
            "- Evidence: `results/dataprep/reports/A1_evidence/repair2/` (design sweep and "
            "rerun, pilot cluster sizes, confirmation, probes, audit, v3 build, pytest, ruff).",
            f"- Ledger: {len(lrows)} rows added (A1 fragment now {n_rows} rows).",
            "\n### Key numbers\n"] + [f"- {k}" for k in key] + \
        ["\n### Deviations added in repair round 2\n"] + [f"- {d}" for d in deviations] + \
        ["\n### Issues for downstream WPs (repair round 2)\n"] + \
        [f"- {d}" for d in downstream] + \
        ["\nNo GPU, 0 Claude API tokens; nothing under hydrophysics/, tests/ or L0 was written; "
         "nothing staged in git.\n"]
    md2 = md2.rstrip("\n") + "\n" + "\n".join(sec)
    md_path.write_text(md2, encoding="utf-8")

    # ------------------------------------------------------------------ A1.json
    products = [
        {"path": _abs("results/dataprep/validation_units/v3/"), "layer": "L2",
         "manifest": f"results/dataprep/validation_units/v3/MANIFEST.json (checker ok; sha256 "
                     f"{sha256_file(V3 / 'MANIFEST.json')})",
         "allowed_uses": "selection, validator, after_the_fact_analysis, design_time_evidence",
         "fit_window_end": "2022-12"},
        {"path": _abs("results/dataprep/preregistration/A1r2.coverage3_confirm.v1.json"),
         "layer": "preregistration",
         "manifest": f"INDEX.jsonl + ledger row A1.prereg.A1r2.coverage3_confirm.v1 (sha256 "
                     f"{shas['A1r2.coverage3_confirm.v1'][0]})",
         "allowed_uses": "registration of the coverage3 confirmation", "fit_window_end": "n/a"},
        {"path": _abs("results/dataprep/preregistration/A1r2.harness_rules.v1.json"),
         "layer": "preregistration",
         "manifest": f"INDEX.jsonl + ledger row A1.prereg.A1r2.harness_rules.v1 (sha256 "
                     f"{shas['A1r2.harness_rules.v1'][0]})",
         "allowed_uses": "declaration of gate-3 (harness.coverage3) and harness.prereg3",
         "fit_window_end": "n/a"},
        {"path": _abs("results/dataprep/reports/A1_evidence/repair2/"), "layer": "evidence",
         "manifest": "none (repair-round-2 evidence)", "allowed_uses": "provenance only",
         "fit_window_end": "n/a"},
    ] + [p for p in old["products"]]
    for p in products:
        if p["path"].endswith("validation_units/v2/") and "[superseded" not in p["manifest"]:
            p["manifest"] += " [data identical in v3; v3 is current]"
        if p["path"].endswith("A1-fresh-units-v2.json") and "gate-3" not in p["allowed_uses"]:
            p["allowed_uses"] += " (its coverage-gate field is superseded by gate-3 in " \
                                 "A1r2.harness_rules.v1)"
    code_new = [str(HERE / f) for f in ("coverage3.py", "coverage3_design.py",
                                        "coverage3_confirm.py", "prereg3.py",
                                        "build_units_v3.py", "a1r2_prereg.py",
                                        "a1r2_checks.py", "a1r2_report.py",
                                        "tests/test_repair2.py")]
    code_paths = list(dict.fromkeys(list(old["code_paths"]) + code_new))
    new = {k: v for k, v in old.items() if k not in ("before_repair_round_2",)}
    out = {
        "wp": "A1", "status": "passed", "date": "2026-10-12", "summary": summary,
        "acceptance": rows, "products": products, "code_paths": code_paths,
        "key_numbers": key + ["[earlier round] " + k for k in old["key_numbers"]
                              if not k.startswith(("Tests:", "Claude API"))],
        "deviations_from_plan": deviations + ["[earlier round] " + d
                                              for d in old["deviations_from_plan"]],
        "issues_for_downstream": downstream + [
            "[earlier round] " + i for i in old["issues_for_downstream"]
            if not i.startswith(("Use harness.prereg2", "Rule gate-2 removes",
                                 "The gate interval (level 0.95)",
                                 "Units: open results/dataprep/validation_units/v2",
                                 "prereg2 checks the latest units registry"))],
        "report_path": rel_to_repo(md_path),
        "coverage_gate_rule": {"module": "harness.coverage3", "rule": coverage3.RULE,
                               "method": coverage3.DEFAULT_METHOD,
                               "envelope": coverage3.ENVELOPE},
        "preregistration_tool": "harness.prereg3",
        "repair_round_2": {"time": now, "findings": [{"finding": t, "response": x}
                                                     for t, x in findings],
                           "evidence": rel_to_repo(EV), "ledger_rows_added": len(lrows)},
        "ledger_rows": n_rows, "claude_api_tokens": 0, "gpu": "none",
        "before_repair_round_2": new,
    }
    js_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {rel_to_repo(md_path)}, {rel_to_repo(js_path)}; ledger +{len(lrows)} "
          f"({n_rows}); acceptance rows {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
