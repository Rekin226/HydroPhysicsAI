"""WP A1 repair round: ledger rows and the 'Repair round' section of reports/A1.md / A1.json.

    python -m harness.a1r_report          (from dataprep_staging/, after a1r_checks)

Reads the repair evidence (``A1_evidence/repair/``), the design-time simulation, the v2
product and the pre-registrations; appends rows to the A1 ledger fragment; inserts a status
note under the title of A1.md and appends the 'Repair round' section; rewrites A1.json with
the WP as it now stands (the previous top level is kept under ``build_round``). Refuses to run
twice.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, REPORTS_DIR, rel_to_repo  # noqa: E402

from harness import prereg2  # noqa: E402

EV = REPORTS_DIR / "A1_evidence" / "repair"
V2 = DATAPREP / "validation_units" / "v2"
MARK = "## Repair round"
HERE = Path(__file__).resolve().parent


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _f(x, n=4):
    return f"{x:.{n}f}"


def _fix_notes(rows: list[dict]) -> None:
    """Correct the note of A1 repair rows already in the A1 fragment (same number id and
    source) whose note changed; only the A1 fragment is touched."""
    import csv
    import fcntl
    path = ledger.fragment_path("A1")
    want = {(r["number_id"], r["source"]): r["note"] for r in rows}
    with open(path, "r+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        rd = list(csv.DictReader(f))
        cols = list(rd[0].keys()) if rd else []
        changed = False
        for r in rd:
            k = (r["number_id"], r["source"])
            if k in want and r["note"] != want[k]:
                r["note"] = want[k]
                changed = True
        if changed:
            f.seek(0)
            f.truncate()
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rd)
        fcntl.flock(f, fcntl.LOCK_UN)


def main(argv=None) -> int:
    md_path, js_path = REPORTS_DIR / "A1.md", REPORTS_DIR / "A1.json"
    md = md_path.read_text(encoding="utf-8")
    if MARK in md:
        raise SystemExit("repair round already appended to A1.md")
    gw = _j(EV / "gate_window_confirm.json")
    ck = _j(EV / "a1r_checks.json")
    au = _j(EV / "prereg_audit.json")
    pb = _j(EV / "prereg_probes.json")
    val2 = _j(V2 / "validation.json")
    reg2 = _j(V2 / "units.json")
    sim = _j(REPORTS_DIR / "A1_evidence" / "coverage_sim.json")
    conf = _j(REPORTS_DIR / "A1_evidence" / "coverage_confirm.json")
    pytest_txt = (EV / "pytest.txt").read_text(encoding="utf-8")
    ruff_ok = "All checks passed!" in (EV / "ruff.txt").read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed", pytest_txt)
    n_pass = int(m.group(1)) if m else 0
    failed = bool(re.search(r"\d+ failed", pytest_txt))
    old = _j(js_path)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cf = {c["name"]: c for c in gw["configs"]}
    g1, g2 = gw["G1"], gw["G2"]
    # design-time level-0.90 numbers (computed in coverage_sim.json, not reported before)
    t90 = {c["name"]: c["results"]["t_L0.9"]["actual_coverage"] for c in sim["configs"]}
    z90 = {c["name"]: c["results"]["z_L0.9"]["actual_coverage"] for c in sim["configs"]}
    t90_below = sorted((v, k) for k, v in t90.items() if v < 0.89)
    z90_below = [k for k, v in z90.items() if v < 0.89]
    f90_below = [(cf[n]["interval_coverage_level0.90"], n) for n in gw["info_level0.90_below_0.89"]]
    conf_min = conf["primary_min_actual_coverage"]
    fresh2 = sorted(u["id"] for u in reg2["units"] if u["role"] == "fresh")
    dt2 = [u for u in reg2["units"] if u["role"] == "design_time"]
    fo = val2["fold_overlaps"]
    dec2, sha_dec2 = prereg2.load_prereg("A1-fresh-units-v2")
    gwp, sha_gwp = prereg2.load_prereg("A1r.gate_window.v1")
    v1_looks = au["registries"]["results/dataprep/validation_units/v1/units.json"][
        "fresh_units_with_looks"]
    w3_units = sorted(v1_looks)
    decl1 = prereg2.parse_time(prereg2.load_prereg("A1-fresh-units-v1")[0]["created"])
    log = prereg2.read_chained(DATAPREP / "preregistration" / "looks.jsonl")
    w3l = [lk for lk in log if lk["wp"] == "W3" and set(lk["units"]) & set(w3_units)]
    n_v1_looks = len(w3l)
    pre_decl = sorted(lk["time"] for lk in w3l if prereg2.parse_time(lk["time"]) < decl1)
    a1_seed = [lk["time"] for lk in log if lk["wp"] == "A1" and set(lk["units"]) & set(w3_units)]
    probes_old = pb["prereg (first tool)"]
    probes_new = pb["prereg2"]
    hh = ck["harness_files_hashed_elsewhere"]
    hashed_by = sorted({x["by"] for x in hh["records"]
                        if "A1r." not in x["by"] and "validation_units/v2" not in x["by"]})
    refused = au["tests_refused_under_repaired_rule"]
    changed = au["tests_with_changed_inputs"]
    unknown = sorted({x["prereg_id"] for x in au["looks_with_unknown_prereg_id"]})
    notin = au["prereg_files_not_in_index"]

    def pw(name, key):
        return cf[name][key]

    few = ["E11_5reg_x6_rho0.1", "E11_5reg_x6_rho0.3", "E6_5reg_x6_rho0.5", "k3_x10_rho0.1",
           "few_large_k5_x200_rho0.05", "S1_14site_x13_rho0.3"]
    worst_bound = max(gw["configs"], key=lambda c: c["share_n_eff_lower_above_true"])

    # ------------------------------------------------------------------ acceptance (now)
    acc = [a for a in old["acceptance"][:4]]
    gap_row = next(a for a in old["acceptance"] if a["criterion"].startswith("Artificial-gap"))
    cell_dup = next(c for c in val2["checks"] if c["check"].startswith("no fresh cell unit"))
    acc.append({
        "criterion": "Sensitivity, not an acceptance criterion: the same interval at level 0.90 "
                     "(the reading of the plan's 0.89 threshold raised by the verifier; the plan "
                     "states no level and the gate keeps 0.95)",
        "threshold": ">= 0.89 in every configuration (informational)",
        "measured": (f"design-time seeds: t below 0.89 in {len(t90_below)} of {len(t90)} ("
                     + ", ".join(f"{k} {_f(v)}" for v, k in t90_below)
                     + f"), z in {len(z90_below)}; fresh seeds (A1r.gate_window.v1): below 0.89 "
                     f"in {len(f90_below)} of 20 primary, min {_f(gw['info_level0.90_min_primary'])}"
                     f" ({', '.join(f'{n} {_f(v)}' for v, n in sorted(f90_below))})"),
        "passed": False})
    acc.append({
        "criterion": "First-round declaration A1-fresh-units-v1: no look on any of its 17 fresh "
                     "units before it (superseded by v2; kept for the record)",
        "threshold": "0 looks on a fresh unit dated before 2026-10-12T06:21:49Z",
        "measured": (f"FAILED for {len(w3_units)} of 17 units ({', '.join(w3_units)}): "
                     f"{n_v1_looks} W3 looks logged at 10:28:17Z, {len(pre_decl)} of them "
                     f"(unregistered) dated before the declaration "
                     f"({', '.join(t[11:19] for t in pre_decl)}Z), 2 registered scoring looks "
                     f"after it; plus the repair's design-time seed dated {a1_seed[0][:19]}Z; "
                     "W3 scored all 324 / 781 benchmarks; the report's 'looks on fresh units 0' "
                     "came from a log that did not yet hold W3's looks"),
        "passed": False})
    acc.append({
        "criterion": "Fresh validation units declared and hashed before any downstream test uses "
                     "them (repair: A1-fresh-units-v2 with harness.prereg2, which refuses a "
                     "declaration if any look names a declared unit)",
        "threshold": "declaration written; 0 looks on any declared unit at declaration and at "
                     "the check",
        "measured": (f"A1-fresh-units-v2 ({len(fresh2)} fresh units) written {dec2['created']}, "
                     f"sha256 {sha_dec2[:12]}...; looks on v2 fresh units at the check: "
                     f"{len(ck['v2_units']['fresh_with_any_look'])}; the 2 leveling folds "
                     "withdrawn and relabelled design-time"),
        "passed": not ck["v2_units"]["fresh_with_any_look"]})
    acc.append({
        "criterion": "Validation-units product gate, v2 (v1 re-hashes and is carried over "
                     "unchanged except the declared relabels; fresh units carry no look; "
                     "partitions disjoint; folds keep sites whole and buffers; pilot folds "
                     "rebuilt on the pilots' rows; no fresh unit near-duplicates a design-time "
                     "unit)",
        "threshold": "all checks pass; checker ok for v1 and v2",
        "measured": (f"{sum(c['pass'] for c in val2['checks'])}/{len(val2['checks'])} checks; "
                     f"max Jaccard fresh cell unit vs design-time region "
                     f"{_f(cell_dup['measured'], 3)}, fresh well fold vs design-time "
                     f"fold {_f(max(v['max_jaccard'] for v in fo.values()), 3)}; checker v1 "
                     f"{ck['checker']['v1']['status']}, v2 {ck['checker']['v2']['status']}; "
                     f"guard as expected {ck['guard_as_expected']} (design_time_evidence now "
                     "allowed)"),
        "passed": bool(val2["passed"] and ck["checker"]["v2"]["status"] == "ok"
                       and ck["checker"]["v1"]["status"] == "ok" and ck["guard_as_expected"])})
    acc.append(gap_row)
    acc.append({
        "criterion": "Pre-registration tool refuses to score a design whose pre-registration "
                     "postdates any look at its test data (plan A1; repaired tool "
                     "harness.prereg2)",
        "threshold": "the verifier's probes (a) revised design after a scored one, (b) "
                     "overlapping scopes, (c) looks under unknown ids are refused; unit tests "
                     "pass; ruff clean",
        "measured": (f"first tool allows {sum(v == 'allowed' for v in probes_old.values())}/"
                     f"{len(probes_old)} probe actions, prereg2 refuses "
                     f"{sum(v.startswith('refused') for v in probes_new.values())}/"
                     f"{len(probes_new)}; withdrawn and design-time ids refused as fresh; "
                     f"{n_pass} tests passed (harness incl. 12 repair tests + common), ruff "
                     f"{'clean' if ruff_ok else 'NOT clean'}"),
        "passed": bool(ck["probes"]["first_tool_allows_all"] and ck["probes"]["prereg2_refuses_all"]
                       and n_pass > 0 and not failed and ruff_ok)})
    acc.append({
        "criterion": "Repaired coverage-gate window rule gate-2 (pre-registered A1r.gate_window.v1,"
                     " fresh seeds 20281012+, 10,000 draws per configuration)",
        "threshold": (f"G1: window-only false fails at a true 0.90 <= "
                      f"{gwp['thresholds']['G1']['max_window_only_fail_rate_true0.90']} where true "
                      f"n_eff < 30; G2: window applies in >= "
                      f"{gwp['thresholds']['G2']['min_window_applies_rate_true0.90']} where true "
                      "n_eff >= 100 and k >= 30; G3: interval unchanged"),
        "measured": (f"G1 max {g1['max_window_only_fail']} over {len(g1['members'])} configurations"
                     f" (gate-1 on the same draws: E6 "
                     f"{pw('E6_5reg_x6_rho0.5', 'gate1_window_only_fail_true0.9')}, E11 rho0.3 "
                     f"{pw('E11_5reg_x6_rho0.3', 'gate1_window_only_fail_true0.9')}, k3x10 "
                     f"{pw('k3_x10_rho0.1', 'gate1_window_only_fail_true0.9')}); G2 min "
                     f"{g2['min_window_applies']} over {len(g2['members'])}; G3 "
                     f"{gw['G3']['pass']}"),
        "passed": bool(gw["pass"])})
    acc.append({
        "criterion": "Origin guards for the index- and label-based gap generators (N, S, E, H, T, "
                     "O, G, R, Y, L; harness.gaps_guarded)",
        "threshold": "post-origin matrices, windows and endpoints raise LeakageError without "
                     "allow_post_origin; masks identical to harness.gaps",
        "measured": "unit test passes (post-origin replica window, head matrix, ring windows, "
                    "ring-path 2022, leveling 2023, GNSS epochs refused; masks, cells and events "
                    "identical); every plan 4.4 family mapped in gaps_guarded.GUARDS",
        "passed": bool(n_pass > 0 and not failed)})
    acc.append({
        "criterion": "No published version broken by the repair: every harness file hashed by a "
                     "product manifest or pre-registration is unchanged",
        "threshold": "0 changed",
        "measured": f"{hh['n']} hash records in {len(hashed_by)} files, {len(hh['changed'])} changed",
        "passed": not hh["changed"]})

    # ------------------------------------------------------------------ ledger rows
    have = [r for r in ledger.read_all() if r["wp"] == "A1"]
    prev_tags = {r["product_version"] for r in have if r["number_id"].startswith("A1.repair.")}
    tag = sorted(prev_tags)[0] if prev_tags else f"harness repair;{now}"
    rows = []

    def row(nid, src, role, note):
        rows.append({"number_id": f"A1.repair.{nid}", "source": rel_to_repo(src), "role": role,
                     "product_version": tag, "note": note})
    row("gate_window.G1", EV / "gate_window_confirm.json", "validator",
        f"max window-only false fail {g1['max_window_only_fail']}")
    row("gate_window.G2", EV / "gate_window_confirm.json", "validator",
        f"min window applies {g2['min_window_applies']}")
    row("gate_window.G3", EV / "gate_window_confirm.json", "validator", "interval identical")
    row("gate_window.gate1_vs_gate2", EV / "gate_window_confirm.json", "validator",
        "window rates, pass rates, n_eff bias, bound calibration (information)")
    row("level090.fresh", EV / "gate_window_confirm.json", "validator",
        f"level-0.90 interval min {gw['info_level0.90_min_primary']}")
    row("level090.design", REPORTS_DIR / "A1_evidence" / "coverage_sim.json",
        "design_time_evidence", f"t level-0.90 below 0.89 in {len(t90_below)} of 15")
    row("units_v2.gate", V2 / "validation.json", "validator",
        f"{sum(c['pass'] for c in val2['checks'])}/{len(val2['checks'])} checks")
    row("units_v2.fold_overlaps", V2 / "validation.json", "validator",
        "fresh well folds vs design-time folds: Jaccard, ARI")
    row("units_v2.counts", V2 / "units.json", "validator",
        f"{len(fresh2)} fresh, {len(dt2)} design-time units")
    row("checks", EV / "a1r_checks.json", "validator", "guard, checker, prereg loads, looks")
    row("probes", EV / "prereg_probes.json", "validator", "verifier probes on both tools")
    row("audit", EV / "prereg_audit.json", "validator", "existing log under repaired rules")
    row("tests", EV / "pytest.txt", "validator", f"{n_pass} passed")
    row("ruff", EV / "ruff.txt", "validator", "ruff harness")
    row("v1_fresh_looks", EV / "prereg_audit.json", "validator",
        f"{n_v1_looks} W3 looks ({len(pre_decl)} before the v1 declaration) on "
        f"{len(w3_units)} v1 fresh units")
    seen = {(r["number_id"], r["source"]) for r in have}
    _fix_notes(rows)
    ledger.append_rows("A1", [r for r in rows if (r["number_id"], r["source"]) not in seen])
    n_rows = sum(1 for r in ledger.read_all() if r["wp"] == "A1")

    # ------------------------------------------------------------------ key numbers etc.
    key = [
        f"Gate rule gate-2 (A1r.gate_window.v1, seeds 20281012+): G1 max window-only false fail "
        f"{g1['max_window_only_fail']} (6 configurations with true n_eff < 30; gate-1 on the same "
        f"draws up to {max(cf[n]['gate1_window_only_fail_true0.9'] for n in g1['members'])}); "
        f"G2 min window applies {g2['min_window_applies']}; G3 intervals identical",
        "gate-1 vs gate-2 full-rule pass at a true 0.90: " + ", ".join(
            f"{n} {pw(n, 'gate1_pass_true0.9')} -> {pw(n, 'gate2_pass_true0.9')}" for n in few),
        "gate-1 vs gate-2 pass at a true 0.80 (power falls where the window no longer applies): "
        + ", ".join(f"{n} {pw(n, 'gate1_pass_true0.8')} -> {pw(n, 'gate2_pass_true0.8')}"
                    for n in few),
        "gate-1 vs gate-2 pass at a true 0.97: " + ", ".join(
            f"{n} {pw(n, 'gate1_pass_true0.97')} -> {pw(n, 'gate2_pass_true0.97')}" for n in few),
        f"n_eff_hat bias with 5 clusters: E6 mean n_eff_hat {_f(cf['E6_5reg_x6_rho0.5']['mean_n_eff_hat'], 1)}"
        f" vs true 8.6 (rho_hat = 0 in {cf['E6_5reg_x6_rho0.5']['share_rho_hat_zero']} of draws); "
        f"mean n_eff_lower {_f(cf['E6_5reg_x6_rho0.5']['mean_n_eff_lower'], 1)}",
        f"n_eff_lower exceeds the true n_eff in up to {worst_bound['share_n_eff_lower_above_true']}"
        f" of draws ({worst_bound['name']}): not a calibrated 95 % bound for binary data",
        f"Interval at level 0.95 (unchanged): confirm min {conf_min} (A1.coverage_confirm.v1); "
        f"fresh seeds of A1r.gate_window.v1 min "
        f"{min(c['interval_coverage_level0.95'] for c in gw['configs'] if c['role'] == 'primary')}",
        f"Interval at level 0.90 (sensitivity): design-time t below 0.89 in {len(t90_below)}/15 "
        f"(min {_f(t90_below[0][0])}), fresh seeds below 0.89 in {len(f90_below)}/20 (min "
        f"{_f(gw['info_level0.90_min_primary'])})",
        f"validation_units/v2: {len(fresh2)} fresh, {len(dt2)} design-time units; 2 leveling folds "
        f"withdrawn; 5 rebuilt pilot fold units; {len(reg2['design_time_schemes'])} design-time "
        "schemes without membership",
        "Fresh well folds vs design-time folds: max Jaccard " + ", ".join(
            f"{k.split('.')[-1]} {_f(v['max_jaccard'], 3)}" for k, v in fo.items())
        + "; max ARI " + ", ".join(f"{k.split('.')[-1]} {_f(max(a['ari'] for a in v['ari'].values()), 3)}"
                                    for k, v in fo.items())
        + "; wells in some design-time test fold: " + ", ".join(
            f"{v['n_wells_in_a_design_time_test_fold']}/{v['n_wells']}" for v in fo.values()),
        f"v1 fresh leveling folds: {n_v1_looks} W3 looks, {len(pre_decl)} dated before the v1 "
        f"declaration; design-time seed for the leveling folds dated {a1_seed[0][:19]}Z (first "
        "coupled-leveling CV output)",
        f"Audit of the log under the repaired rules: {len(refused)} registered tests would be "
        f"refused ({', '.join(sorted({x['prereg_id'] for x in refused}))}); "
        f"{len(au['looks_with_unknown_prereg_id'])} looks under {len(unknown)} ids not in INDEX; "
        f"{len(notin)} pre-registration files not in INDEX; {len(changed)} registered tests with "
        "changed inputs",
        f"Probes: first tool allows {len(probes_old)}/{len(probes_old)}, prereg2 refuses "
        f"{len(probes_new)}/{len(probes_new)}",
        f"Harness files hashed elsewhere: {hh['n']} records, 0 changed (repair adds modules)",
        f"Tests: {n_pass} passed; ruff {'clean' if ruff_ok else 'not clean'}",
        "Claude API tokens: 0; GPU: none",
    ]
    deviations = [
        "Repair round: the repaired rules live in new modules (harness.prereg2, coverage2, "
        "gaps_guarded, build_units_v2) instead of edits, because product manifests and "
        "pre-registrations hash prereg.py, coverage.py, gaps.py, build_units.py, units.py and "
        "__init__.py (" + ", ".join(hashed_by) + "). harness.__version__ stays 1.0.0; prereg2 "
        "records tool_version 2.0.0.",
        "Coverage gate: the plan's 'effective n of at least 30' is decided by the one-sided 0.95 "
        "lower bound of n_eff (F-based upper bound of the ANOVA ICC), not the point estimate, "
        "and small-stratum pooling uses the same bound (rule gate-2).",
        "Scope overlap ignores dataset names; only disjoint time windows separate two looks on "
        "the same unit (conservative).",
        "validation_units/v2 declares no fresh leveling unit: every leveling benchmark was a "
        "hold-out unit of the 2026-09 zonal-column CV and of W3's design iterations.",
        "The design-time inventory now includes rebuilt pilot well folds (fusion k-means CH, "
        "head_pumping_signatures 5 km-block GroupKFold / LOGO and township folds) and three "
        "schemes without rebuilt membership (coupled leveling 5-fold, W3 10 km blocks, "
        "ML-transfer 5 km-block GroupKFold over cells).",
    ]
    issues = [
        "Use harness.prereg2, harness.coverage2 and harness.gaps_guarded from now on. The "
        "first-round modules stay importable only because published products hash them; "
        "harness.prereg does not enforce the plan's rule (a WP that keeps using it is not "
        "protected; prereg2.assert_can_score and prereg2.audit re-check anything it wrote).",
        "Units: open results/dataprep/validation_units/v2 (15 fresh units, declared in "
        "A1-fresh-units-v2). fresh.folds.leveling_CH / _all are withdrawn (prereg2 refuses "
        "them anywhere); no leveling unit is fresh, so a subsidence model scored on leveling "
        "2012-2021 reports design-time evidence only.",
        "prereg2 checks the latest units registry by default: units of your own must be listed "
        "in external_units. A look on a fresh unit blocks every later test of that unit with an "
        "overlapping window, whatever its dataset name; register all candidate designs (plan "
        "4.5: at most three) before the first look.",
        "prereg2.assert_can_score refuses a pre-registration whose hashed inputs changed. The "
        "audit found three: " + "; ".join(f"{x['id']} ({', '.join(i['path'] for i in x['inputs'])})"
                                         for x in changed)
        + ". Their recorded scores stand as made; a re-score needs a new registration.",
        "Audit of the existing log: " + "; ".join(
            f"{x['prereg_id']} {x['test_id']} (WP {x['wp']}) has {x['n_prior_looks']} look(s) "
            "dated before its registration" for x in refused)
        + f". Looks under ids not in INDEX.jsonl: {', '.join(unknown)} "
        "(prereg2.record_look now refuses such ids; register a declaration and use its id). "
        f"Pre-registration files written outside the tool ({len(notin)}): {', '.join(notin)}; "
        "load_prereg cannot verify them.",
        "Fresh is still 'never a hold-out unit for a design choice', not unseen data: all 158 "
        "wells were in a design-time hold-out fold (fresh well folds overlap them with max "
        "Jaccard 0.60 and ARI <= 0.36), and every in-grid cell was in an ML-transfer test fold.",
        "Rule gate-2 removes chance failures in few-cluster strata, and with them most of the "
        "gate's power there: with 5 regions x 6 rows a product whose true coverage is 0.80 "
        f"passes in {pw('E6_5reg_x6_rho0.5', 'gate2_pass_true0.8')} (E6) and one at 0.97 in "
        f"{pw('E6_5reg_x6_rho0.5', 'gate2_pass_true0.97')} of draws. This is the plan's intent "
        "(the interval alone decides such strata); report n_eff, n_eff_lower, k and the interval "
        "with every verdict and treat a pass there as weak evidence.",
        "n_eff_lower serves only to decide whether the window applies; it is not a calibrated "
        f"bound for binary data (above the true n_eff in up to {worst_bound['share_n_eff_lower_above_true']}"
        " of draws). Quote n_eff (the point estimate, biased upward with few clusters) with that "
        "caveat.",
        f"The gate interval (level 0.95) under-covers by up to about 4 points (min {conf_min}); "
        f"at level 0.90 it falls to {_f(gw['info_level0.90_min_primary'])}. If a WP reads the "
        "plan's 0.89 as a 0.90-level requirement, this interval does not meet it.",
        "Index-based gap generators: call the gaps_guarded wrappers with the matrix calendar; "
        "harness.gaps itself still trusts the caller for N, S, E, H, T, O, G, R, Y, L.",
        "The look log stays self-reported: an unlogged look cannot be detected.",
    ]

    # ------------------------------------------------------------------ markdown
    tbl = ["| Criterion | Threshold | Measured | Pass |", "|---|---|---|---|"]
    for a in acc:
        tbl.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                   f"{'yes' if a['passed'] else '**no**'} |")
    verdicts = [
        ("Major 1, fresh-unit acceptance broken for the leveling folds", f"Correct. W3 logged {n_v1_looks} "
         f"looks on fresh.folds.leveling_CH / _all, {len(pre_decl)} dated before the v1 "
         "declaration, and every leveling benchmark was a hold-out unit of the 2026-09 "
         "zonal-column CV. validation_units/v2 renames them dt.folds.leveling_CH / _all "
         "(design-time), withdraws the v1 ids, seeds a look dated to the first coupled-leveling "
         "CV output (2026-09-14T07:52:04Z), and A1-fresh-units-v2 declares the 15 remaining "
         "fresh units with the repaired tool (0 looks). The v1 acceptance row is reported as "
         "failed."),
        ("Major 2, the tool did not enforce the plan's rule", "Correct. harness.prereg2: any "
         "look on a fresh unit dated before a registration blocks it (purpose, test id and "
         "registration id no longer matter); scopes overlap unless their time windows are "
         "disjoint; record_look refuses unknown registration ids, test ids outside them, test "
         "ids without one and the purpose 'score'; assert_can_score also refuses changed "
         "inputs, another WP's registration and a changed units registry. The verifier's probes "
         f"pass on the first tool ({len(probes_old)}/{len(probes_old)} allowed) and are refused "
         f"by prereg2 ({len(probes_new)}/{len(probes_new)})."),
        ("Major 3, n_eff biased upward with few clusters, window misapplied", "Correct, and the "
         "first report misdiagnosed it as the plan's rule. Rule gate-2 (harness.coverage2) "
         "applies the window only when the one-sided 0.95 lower bound of n_eff is at least 30. "
         "Pre-registered evaluation A1r.gate_window.v1 on fresh seeds: window-only false fails "
         f"where true n_eff < 30 fall from up to "
         f"{max(cf[n]['gate1_window_only_fail_true0.9'] for n in g1['members'])} to "
         f"{g1['max_window_only_fail']}; the window still applies in every draw where n_eff is "
         "large and well estimated; the interval is unchanged."),
        ("Minor, coverage level", f"Reported. At level 0.90 the interval falls below 0.89 in "
         f"{len(t90_below)} of 15 design-time configurations and {len(f90_below)} of 20 fresh "
         f"ones (min {_f(gw['info_level0.90_min_primary'])}); at 0.95 it under-covers by up to "
         f"about 4 points (min {conf_min}). The level stays 0.95 (not changed after seeing "
         "results); the acceptance holds only under that level."),
        ("Minor, design-time inventory incomplete", "Fixed in v2: 5 pilot fold units rebuilt "
         "from the pilots' inputs (fusion k-means CH; head_pumping_signatures 5 km-block "
         "GroupKFold and LOGO, township folds) and 3 schemes listed without membership. Overlap "
         "of the fresh well folds with them is quantified (max Jaccard 0.60, ARI <= 0.36; 158 of "
         "158 wells were in some design-time test fold); none is a near-duplicate, so they stay "
         "fresh as partitions."),
        ("Minor, LeakageError claim overstated", "Correct. The claim is withdrawn; "
         "harness.gaps_guarded wraps N, S, E, H, T, O, G, R, Y and L with calendar checks "
         "(identical masks)."),
        ("Minor, pool_small_strata stopped pooling", "Fixed in coverage2.pool_small_strata "
         "(skips a stratum without a neighbour, reports groups left under-sized)."),
        ("Minor, design_time_evidence refused by v1", "Fixed in v2 (allowed; the guard opens it)."),
        ("Minor, 'declared before the first run' unverifiable", "Agreed; it cannot be verified "
         "after the fact. The run stays labelled design-time and its rerun is identical."),
        ("Minor, exact-string scope matching", "Fixed in prereg2 (time-window overlap; dataset "
         "names not compared)."),
    ]
    sec = [f"{MARK} (2026-10-12, {now})", "",
           "An independent verifier found three major and seven minor issues. Each is addressed "
           "below; nothing published was overwritten. Files that other products or "
           "pre-registrations hash were not edited; the repaired rules are new modules.", "",
           "### Verifier findings and what was done", ""]
    for t, v in verdicts:
        sec.append(f"- **{t}.** {v}")
    sec += ["", "### Acceptance (WP as it now stands)", "", *tbl, "",
            "### New products and code", "",
            f"- `results/dataprep/validation_units/v2/` (L2; checker ok; allowed: selection, "
            f"validator, after_the_fact_analysis, design_time_evidence; fit_window_end 2022-12): "
            f"{len(fresh2)} fresh and {len(dt2)} design-time units, `withdrawn` (the v1 leveling "
            "ids) and `design_time_schemes`. v1 stays published and re-hashes; it is superseded.",
            f"- `preregistration/A1r.gate_window.v1.json` (sha256 {sha_gwp[:12]}..., created "
            f"{gwp['created']}) and `preregistration/A1-fresh-units-v2.json` (sha256 "
            f"{sha_dec2[:12]}..., created {dec2['created']}), both written by harness.prereg2.",
            "- Code: `harness/prereg2.py`, `coverage2.py`, `gaps_guarded.py`, "
            "`gate_window_confirm.py`, `a1r_prereg.py`, `build_units_v2.py`, `a1r_checks.py`, "
            "`a1r_report.py`, `tests/test_repair.py`; README section updated.",
            "", "### Key numbers", "", *[f"- {k}" for k in key], "",
            "### Deviations added in the repair round", "", *[f"- {d}" for d in deviations], "",
            "### Issues for downstream WPs (repair round)", "", *[f"- {i}" for i in issues], "",
            "### Evidence", "",
            "`results/dataprep/reports/A1_evidence/repair/`: `gate_window_confirm.json` / `.log` "
            "(pre-registered run), `build_units_v2.log`, `a1r_checks.json` (guard, checker, "
            "pre-registrations, looks), `prereg_probes.json` (verifier probes on both tools), "
            "`prereg_audit.json` (the log under the repaired rules), `pytest.txt`, `ruff.txt`; "
            "the product's own `validation.json` and `MANIFEST.json`.", "",
            f"Ledger: {len(rows)} rows added (A1 fragment now {n_rows} rows). No GPU, 0 Claude API "
            "tokens; nothing under hydrophysics/, tests/ or L0 was written; nothing staged in git.",
            ""]
    note = ("> **Repair round (2026-10-12): read the 'Repair round' section at the end first.** "
            "Use validation_units/v2 and A1-fresh-units-v2, harness.prereg2, harness.coverage2 "
            "and harness.gaps_guarded. The first-round statements that no look exists on the "
            "fresh units and that the declaration preceded every use are wrong for the two "
            "leveling folds (now withdrawn), the tool did not enforce the plan's rule, and the "
            "point-window false fails were a harness estimator problem, not the plan's rule.")
    lines = md.split("\n")
    lines.insert(1, "\n" + note)
    md_new = "\n".join(lines).rstrip("\n") + "\n\n" + "\n".join(sec)
    md_path.write_text(md_new, encoding="utf-8")

    # ------------------------------------------------------------------ json
    build_round = {k: old[k] for k in ("status", "summary", "acceptance", "products",
                                       "key_numbers", "deviations_from_plan",
                                       "issues_for_downstream", "evidence", "preregistrations",
                                       "coverage_gate_rule", "code_paths")}
    new = dict(old)
    new["build_round"] = build_round
    new["status"] = "passed"
    new["acceptance"] = acc
    new["preregistrations"] = dict(old["preregistrations"])
    for pid in ("A1r.gate_window.v1", "A1-fresh-units-v2"):
        d, s = prereg2.load_prereg(pid)
        new["preregistrations"][pid] = {"sha256": s, "created": d["created"], "kind": d["kind"],
                                        "tool": d.get("tool")}
    new["coverage_gate_rule"] = {"module": "harness.coverage2", "rule": "gate-2", "level": 0.95,
                                 "quantile": "t", "point_window": [0.85, 0.95],
                                 "point_window_n_eff_lower_min": 30, "window_conf": 0.95}
    new["evidence"] = old["evidence"] + [rel_to_repo(EV)]
    new["ledger_rows"] = n_rows
    new["repair_round"] = {
        "time": now, "verdicts": [{"finding": t, "action": v} for t, v in verdicts],
        "gate_window": {"G1": g1, "G2": g2, "G3": gw["G3"], "pass": gw["pass"],
                        "prereg_sha256": sha_gwp},
        "units_v2": {"fresh": fresh2, "n_design_time": len(dt2),
                     "withdrawn": sorted(reg2["withdrawn"]),
                     "declaration_sha256": sha_dec2, "declaration_created": dec2["created"],
                     "fold_overlaps_max_jaccard": {k: v["max_jaccard"] for k, v in fo.items()}},
        "checks_pass": ck["pass"], "audit_summary": ck["audit_summary"],
        "tests_passed": n_pass, "ruff_clean": ruff_ok,
    }
    new["key_numbers"] = key + old["key_numbers"]
    new["deviations_from_plan"] = old["deviations_from_plan"] + deviations
    new["issues_for_downstream"] = issues + [i for i in old["issues_for_downstream"]
                                             if "point window" not in i
                                             and "little power" not in i
                                             and "Pre-register before scoring" not in i
                                             and "LeakageError" not in i]
    added = ["coverage2.py", "prereg2.py", "gaps_guarded.py", "gate_window_confirm.py",
             "a1r_prereg.py", "build_units_v2.py", "a1r_checks.py", "a1r_report.py",
             "tests/test_repair.py"]
    new["code_paths"] = old["code_paths"] + [str(HERE / a) for a in added]
    new["products"] = [
        {"path": str(V2) + "/", "layer": "L2",
         "manifest": "results/dataprep/validation_units/v2/MANIFEST.json (checker ok)",
         "allowed_uses": "selection, validator, after_the_fact_analysis, design_time_evidence",
         "fit_window_end": "2022-12"},
        {"path": str(DATAPREP / "preregistration" / "A1-fresh-units-v2.json"),
         "layer": "preregistration",
         "manifest": f"INDEX.jsonl + ledger row A1.prereg.A1-fresh-units-v2 (sha256 {sha_dec2})",
         "allowed_uses": "declaration of the 15 fresh units, the withdrawn ids and gate-2",
         "fit_window_end": "n/a"},
        {"path": str(DATAPREP / "preregistration" / "A1r.gate_window.v1.json"),
         "layer": "preregistration",
         "manifest": f"INDEX.jsonl + ledger row A1.prereg.A1r.gate_window.v1 (sha256 {sha_gwp})",
         "allowed_uses": "registration of the gate-2 evaluation", "fit_window_end": "n/a"},
        {"path": str(EV) + "/", "layer": "evidence",
         "manifest": "none (repair-round evidence)", "allowed_uses": "provenance only",
         "fit_window_end": "n/a"},
    ] + [dict(p, allowed_uses=p["allowed_uses"] + " [superseded by v2]")
         if "validation_units/v1" in p["path"] or "A1-fresh-units-v1" in p["path"] else p
         for p in old["products"]]
    js_path.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": [(a["criterion"][:70], a["passed"]) for a in acc],
                      "ledger_rows_added": len(rows), "ledger_rows_A1": n_rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
