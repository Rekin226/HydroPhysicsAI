"""Write the WP P2abc report (results/dataprep/reports/P2abc.md and .json) after the repair
round: the round-2 content (kept, with the verifier's corrections) plus a 'Repair round'
section. Numbers come from the evidence JSONs; nothing is computed here beyond formatting.

Usage: python -I r3_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
from p3lib import L  # noqa: E402

REP = L.DATAPREP / "reports"
E3 = P3.EVID3
R2 = L.load_json(E3 / "P2abc.round2.json")
F = L.load_json(E3 / "report_facts3.json")
T6 = L.load_json(E3 / "tests" / "t6_zero_shipped.json")
T7P = L.load_json(E3 / "tests" / "t7_construction3_preorigin.json")
T7F = L.load_json(E3 / "tests" / "t7_construction3_full.json")
T7R = L.load_json(E3 / "tests" / "t7_construction3_full_repair.json")
T8 = L.load_json(E3 / "tests" / "t8_july_member.json")
X1 = L.load_json(E3 / "design_time" / "x1_explore_2025.json")
X2 = L.load_json(E3 / "design_time" / "x2_rain_check.json")
CHK = (E3 / "checker.txt").read_text().strip().splitlines()[-1]
GP = (E3 / "guard_probe.txt").read_text().strip().splitlines()[-1]
with open(L.DATAPREP / "preregistration" / "INDEX.jsonl", encoding="utf-8") as _f:
    IDX = [json.loads(x) for x in _f]
SHA = {r["id"]: r["sha256"] for r in IDX if r["wp"] == "P2abc"}


def pct(x):
    return f"{100 * x:.2f} %"


def x1a(c, k, f="zshare"):
    return X1[c]["A"][k][f]


def code_paths() -> list[str]:
    v3 = sorted(str(p.relative_to(L.REPO)) for p in P3.V3.glob("*.py"))
    return list(v3) + R2["code_paths"]  # repository-relative


def build():
    c6 = T6["checks"]
    t8 = T8["counties"]
    # ---------------------------------------------------------------- acceptance (repair)
    acc3 = [
        {"criterion": "P2c zero classification on the SHIPPED classifier (P2r3b.c1.zero_shipped_"
                      "yunlin, P2abc.r3.repair.v2; part of both v2 gates)",
         "threshold": "each of 7 items within 1 % of the pilot (plan P2c): kz zero-months in "
                      "idle, long_idle, terminated, trailing_short; irrigation zero share in "
                      "April and August; structural meter-months of kept meters",
         "measured": (f"idle {c6['idle']['shipped']:,} vs {c6['idle']['pilot']:,.0f} "
                      f"({pct(c6['idle']['rel_diff'])}); long_idle {c6['long_idle']['shipped']:,}"
                      f" vs {c6['long_idle']['pilot']:,.0f} ({pct(c6['long_idle']['rel_diff'])});"
                      f" terminated {c6['terminated']['shipped']:,} vs "
                      f"{c6['terminated']['pilot']:,.0f} ({pct(c6['terminated']['rel_diff'])}); "
                      f"trailing_short {c6['trailing_short']['shipped']:,} vs "
                      f"{c6['trailing_short']['pilot']:,.0f} "
                      f"({pct(c6['trailing_short']['rel_diff'])}); April "
                      f"{c6['irrigation_zero_apr']['shipped']:.5f} vs 0.0558 "
                      f"({pct(c6['irrigation_zero_apr']['rel_diff'])}); August "
                      f"{c6['irrigation_zero_aug']['shipped']:.5f} vs 0.1563 "
                      f"({pct(c6['irrigation_zero_aug']['rel_diff'])}); structural "
                      f"{c6['structural_kept_meters']['shipped']:,} vs "
                      f"{c6['structural_kept_meters']['pilot']:,} "
                      f"({pct(c6['structural_kept_meters']['rel_diff'])}); max "
                      f"{pct(max(v['rel_diff'] for v in c6.values()))}"),
         "passed": bool(T6["passed"]), "id": "P2abc.r3.c1.zero_shipped"},
        {"criterion": "Construction checks, pre-origin v2 (P2r3b.k1.construction_preorigin; "
                      "product gate)",
         "threshold": "0 violations on items a-l",
         "measured": "0 on every item (incl. (i) exact re-run of zeros3.classify: "
                     f"{T7P['details']['i']['rows']:,} zero rows; (k) member cells rebuilt per "
                     "pump; (h) post-origin inputs only under exemption entries)",
         "passed": bool(T7P["passed"]), "id": "P2abc.r3.k1.construction.preorigin"},
        {"criterion": "Construction checks, full v2, first assembly (P2r3b.k1.construction_full)",
         "threshold": "0 violations on items a-l",
         "measured": "item (c): 1 Yunlin pump-month (a code-M0 monthly meter installed 2026-04 "
                     "had its pre-install 2025-07 month flagged by the M0 rule); all other "
                     "items 0",
         "passed": bool(T7F["passed"]), "id": "P2abc.r3.k1.first_assembly.full"},
        {"criterion": "Construction checks, full v2, re-assembly (P2r3c.k1.construction_full, "
                      "P2abc.r3.k1_repair.v1 registered after the failure; product gate)",
         "threshold": "0 violations on items a-l ((j) restated: installed readings, expected "
                      "set computed in the check)",
         "measured": "0 on every item (M0 suspects: Yunlin "
                     f"{T7R['details']['j']['yunlin']['m0_suspect_meters']:,} meters / "
                     f"{T7R['details']['j']['yunlin']['m0_suspect_pumps']:,} pumps, Changhua "
                     f"{T7R['details']['j']['changhua']['m0_suspect_meters']:,} / "
                     f"{T7R['details']['j']['changhua']['m0_suspect_pumps']:,}, Chiayi "
                     f"{T7R['details']['j']['chiayi']['m0_suspect_meters']} / "
                     f"{T7R['details']['j']['chiayi']['m0_suspect_pumps']})",
         "passed": bool(T7R["passed"]), "id": "P2abc.r3.k1r.construction.full"}]
    for c in L.COUNTIES:
        m = t8[c]["member_2013_2025"]
        acc3.append({
            "criterion": f"P2b July backtest on the synchronized-zero member, {c} "
                         f"(P2r3b.b2.july_syncfill_{c}; reported only)",
            "threshold": "reported against mean |fan-total error| <= 20 % (no decision; "
                         "round-2 P2r2.b1 verdicts stand)",
            "measured": (f"member 2013-2025 {m['mean_abs_total_err_pct']:.1f} % (max "
                         f"{m['max_abs_total_err_pct']:.1f} %); member 2013-2024 "
                         f"{t8[c]['member_2013_2024']['mean_abs_total_err_pct']:.1f} %; recorded "
                         f"2013-2024 {t8[c]['recorded_2013_2024']['mean_abs_total_err_pct']:.1f}"
                         f" %; plan variant (as-read July 2013-2024) "
                         f"{t8[c]['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.1f}"
                         f" %"),
            "passed": bool(m["mean_abs_total_err_pct"] <= 20.0),
            "id": f"P2abc.r3.b2.july_member.{c}"})
    acc3 += [
        {"criterion": "Manifests re-hash (common.check on yunlin_electricity and "
                      "pump_electricity, after publication)",
         "threshold": "0 failures", "measured": CHK, "passed": "(failed 0)" in CHK,
         "id": "P2abc.r3.checker"},
        {"criterion": "Guard probe of the v2 products (U3/U4/U5)",
         "threshold": "every probe as designed", "measured": GP,
         "passed": GP.startswith("26/26"), "id": "P2abc.r3.guard_probe"},
        {"criterion": "Pre-origin staging product rebuilt with the final code (the M0 fix came "
                      "after its assembly; reproducibility, not registered)",
         "threshold": "identical content", "measured": "15/15 files identical in content",
         "passed": True, "id": "P2abc.r3.preorigin_rebuild"}]
    # round-2 rows with corrections
    acc2 = []
    for a, nid in zip(R2["acceptance"], R2["acceptance_number_ids"], strict=True):
        a = dict(a)
        if a["criterion"].startswith("P2c zero classification port"):
            a["criterion"] = ("[round 2, superseded] " + a["criterion"] + ": scored "
                              "t4_zero_port.pilot_stats, a re-implementation, not the "
                              "classifier that wrote the v1 zero table (verifier); the v2 "
                              "products are gated by P2r3b.c1 instead")
        elif a["criterion"].startswith(("Construction checks", "Manifests re-hash",
                                        "Pre-origin U3")):
            a["criterion"] = "[round 2, v1 products] " + a["criterion"]
        a["id"] = nid
        acc2.append(a)
    return acc3, acc2


def md(acc3, acc2) -> str:
    pre, full = F["preorigin"], F["full"]
    zb = F["zeros"]
    t8 = T8["counties"]
    m0e = full["m0_2025_07_energy"]
    st = T6["reported"]["structural"]
    rk = X2["yunlin"]["jul_aug_rank_desc"]
    rmm = X2["yunlin"]["monthly_mm"]
    o = []
    w = o.append
    w("# WP P2abc report: billing phase, Chow-Lin split, record-end mask and zero classes\n")
    w("Round 2 (on WP C0's census: Yunlin, Changhua and in-grid Chiayi) built and scored "
      "2026-10-14 (UTC 2026-10-13 20:15-20:56) and published as yunlin_electricity/{preorigin,"
      "full}/v1. **Repair round** after independent verification, 2026-10-14 (UTC 00:36-01:55): "
      "new versions yunlin_electricity/{preorigin,full}/v2 supersede v1.\n")
    w("**Status: passed.** Both v2 gates pass: the zero check now scores the shipped classifier, "
      "and the construction checks pass (full product after one registered repair). Non-gating "
      "results that fail are unchanged from round 2: the registered head check fails in both "
      "counties, Changhua's Chow-Lin sd is uncalibrated, Chiayi's July backtest fails (its "
      "July-2026 imputation is withheld) and Chiayi's meter validation cannot be evaluated "
      "beyond groups of 10.\n")
    # ---------------------------------------------------------------- summary
    w("## Summary\n")
    w("The repair round fixes the verifier's two major issues and its minor ones, and publishes "
      "yunlin_electricity/preorigin/v2 and full/v2. **P2c:** round 2's registered zero check "
      "scored a re-implementation of the pilot's s05(e), not the classifier that wrote "
      "zero_class.parquet, and the shipped classifier missed the plan's 1 % criterion "
      "(long_idle -1.76 %, terminated +7.3 %). The v2 tables come from a new classifier "
      "(zeros3.py) whose run kinds are the pilot's rules, with structural (pre-install) zeros "
      "as a separate column. The registered check P2r3b.c1 runs that same function on the "
      "pilot's view of the Yunlin record. All 7 items are within 1 % of the pilot, max "
      f"{pct(max(v['rel_diff'] for v in T6['checks'].values()))}, including the structural "
      "class: 902,981 against the pilot's 901,713 kept-meter pre-install months under one "
      "month convention. The pilot's own statistic of 878,738 is recomputed exactly from its "
      "archive. **P2b:** the full product now carries the plan's M0 July-2025 flag 9 (values "
      "kept). The verifier read the Yunlin phase-A July-August 2025 deficit as missing bills "
      "recorded as zero; the evidence says otherwise. The zero-bill meters' next bill is "
      "not raised (ratio 1.15 against 1.12 for meters with a positive bill; a missed reading "
      "of a cumulative meter would give about 2). The delivery records these months as 0, "
      "not as missing. The same pattern occurs in 2019 and 2021, which are synchronized-zero "
      "months of the plan's own list. These bills are therefore treated by the plan's P2c "
      "rule: flag 9 in the zero table (now pair-consistent), plus a sensitivity member. Two "
      "new arms, recorded_syncfill and chowlin_syncfill, deliver the masked-and-imputed "
      "alternative the verifier asked for, for every synchronized-zero month of both "
      "products. The primary arms keep the recorded zeros, and their values are identical "
      "to v1; only the 2025-07 flags change. Process: the first repair registration "
      "(P2abc.r3.repair.v1) was amended before any score (v2), after the first build showed "
      "that its window-end rule turned the pre-origin terminations of Dec-Jan-paired meters "
      "into long idling. The full construction check failed once (1 pump-month: the M0 rule "
      "hit a pre-install month) and passed after a registered repair. The July backtest on "
      "the member (reported only) gives Yunlin 13.2 %, Changhua 8.4 % and Chiayi 33.8 %. "
      "No GPU and no Claude API were used.\n")
    # ---------------------------------------------------------------- repair round
    w("## Repair round (2026-10-14)\n")
    w("### What the verifier found, and what was done\n")
    w("| verifier finding | severity | resolution |\n|---|---|---|")
    w("| The registered P2c gate scored `t4_zero_port.pilot_stats`, not `b2_zeros.classify`, "
      "which wrote zero_class.parquet; the shipped classifier misses 1 % (long_idle -1.76 %, "
      "terminated +7.3 %); structural class dropped from the gate, undeclared | major | Fixed: "
      "new shipped classifier `zeros3.classify_core` (pilot run kinds; structural column); the "
      "registered check P2r3b.c1 scores that function; 7/7 items within 1 % (structural included; "
      "the plan's literal 878,738 explained below) |")
    w("| P2b: M0 July 2025 not flagged 9 although the plan says so; Yunlin phase-A Jul-Aug 2025 "
      "deficit unmasked (read as missing bills recorded as zero); 'the July-2025 problem does "
      "not exist' contradicted | major | M0 flag applied (values kept). The phase-A deficit is "
      "shown not to be incomplete pairs (evidence below) and is handled by the plan's "
      "synchronized-zero rule, with sensitivity members that carry the masked-and-imputed "
      "alternative; the round-2 sentence is retracted |")
    w("| Chiayi Chow-Lin sd labelled 'calibrated' without a coverage check; card claims Chiayi skill "
      "over n = 1..300 | minor | Fixed: label 'uncalibrated (not checked on Chiayi meters; "
      "Yunlin's sd model)' in manifests, cards and conventions; the card no longer quotes it |")
    w("| Pre-origin manifest lists the zero-check JSON (post-origin electricity) without an "
      "exemption entry | minor | Fixed: v2 manifest carries the exemption "
      "P2abc.r3.exemption.zero_check_gate.preorigin (input path, data_end 2025-07, uses, "
      "sensitivity); construction check (h) now requires it |")
    w("| Ledger roles: round-1 units, head-check wells and heads, Yunlin round-2 units recorded "
      "as validator | minor | Fixed: 18 rows changed to design_time_evidence (Changhua head wells, "
      "not in the billing pilot, stay validator); round-2 fragment kept as "
      "ledger/superseded/P2abc.r2.csv |")
    w("| July-2026 imputation flagged 4, plan P2b says 5, undeclared | minor | Declared: flag 4 "
      "follows the plan's table 2.4 (own temporal model); the P2b text's 5 contradicts it |")
    w("| Process traceability: b1 rerun at 20:30Z without its own look; p2lib edited between d0 "
      "and the registration, old version not kept | minor | Cannot be repaired after the fact; "
      "disclosed here (the 20:30Z rerun produced the stage-1 arrays the registration hashes; its "
      "log b1_build_rerun.log equals the first run's log except for the run times) |")
    w("| Wording: pro-rata 'worse at every size' (fan +0.024); July 'fan total' is the county "
      "total of kept M2 phase-A meters | minor | Fixed in 'Results in brief' |")
    w("| Product name yunlin_electricity covers three counties | minor | Kept (plan's name); "
      "stated in bold at the top of each PRODUCT_CARD and in the conventions |\n")
    w("### Registrations and looks of the repair round\n")
    w(f"- `P2abc.r3.repair.v1` (sha256 `{SHA.get('P2abc.r3.repair.v1', '')[:16]}...`): written "
      "after the code was debugged on synthetic data and on a Chiayi-only scratch run (look "
      "[P2-r3-debug-chiayi]). **Superseded before any of its tests was scored.**")
    w(f"- `P2abc.r3.repair.v2` (sha256 `{SHA.get('P2abc.r3.repair.v2', '')[:16]}...`): the "
      "same tests under new ids (P2r3b.*), with one change. A meter's window is trimmed to "
      "its first..last observed month before runs are classified. Reason: the first build "
      "(values only, look [P2-r3-zeros-build-review]) showed that the window-end rule made "
      "every trailing run interior when the last window month is not an observation. In the "
      "pre-origin product, 2022-12 of the Dec-Jan-paired meters is C0's flag-4 fill, so "
      "Yunlin's pre-origin terminated zero-months fell from round 2's 260,542 to 128,907. The "
      "rule is identical to the pilot's where both edge months are observations.")
    w(f"- `P2abc.r3.k1_repair.v1` (sha256 `{SHA.get('P2abc.r3.k1_repair.v1', '')[:16]}...`): "
      "the full construction check failed on the first assembly. Item (c) had 1 Yunlin "
      "pump-month: a code-M0 monthly meter installed 2026-04 had its pre-install 2025-07 "
      "month flagged by the M0 rule. As registered, items (c) and (j) contradicted each "
      "other for that pump-month, and (j) passed only because it reused the product's own "
      "selection. Fix: the M0 rule takes installed readings only. Check (j) now says "
      "'installed' and computes its expected set itself, which tightens it. The failed "
      "result stands for the first assembly, which was never published.")
    w("- Looks (A1 log): [P2-r3-verifier-disclosure] records the verifier's unlogged looks "
      "(outcomes listed there are known to this round). The other looks are "
      "[P2-r3-explore-2025], [P2-r3-explore-structural], [P2-r3-explore-rain], "
      "[P2-r3-debug-chiayi], [P2-r3-zeros-build-review], [P2-r3-k1-diag], "
      "[P2-r3-preorigin-rebuild] and [P2-r3-facts]. Scores were logged by assert_can_score "
      "(P2r3b.c1, P2r3b.k1 x2, P2r3c.k1, P2r3b.b2 x3).")
    w("- Not fresh (declared): P2r3b.c1 runs on the same Yunlin meters and window as round "
      "2's check and the verifier's run. The pilot reference, round 2's 0.17 % and the "
      "verifier's result for the old classifier were known. The new classifier was run "
      "before registration only on synthetic data and Chiayi. P2r3b.b2 uses round-2 July "
      "units whose recorded-arm outcomes were known.\n")
    w("### Acceptance (repair round)\n")
    w("| criterion | threshold | measured | pass |\n|---|---|---|---|")
    for a in acc3:
        w(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
          f"{'PASS' if a['passed'] else 'FAIL'} |")
    w("")
    w("Gates (conjunctive, registered): yunlin_electricity/preorigin/v2 = "
      "P2r3b.k1.construction_preorigin and P2r3b.c1.zero_shipped_yunlin (passed); "
      "yunlin_electricity/full/v2 = P2r3c.k1.construction_full and P2r3b.c1.zero_shipped_yunlin "
      "(passed). The Chow-Lin shipping, the sd labels and the July-2026 imputation keep their "
      "round-2 registered verdicts.\n")
    w("### P2c: the shipped classifier and the structural class\n")
    w("`zeros3.classify_core` implements the pilot's s05(e) run loop. A zero is a reading "
      "exactly 0, and a non-observation (flags 4, 8, 9) is NaN: not a zero, and it breaks a "
      "run. first/last use are taken over the whole record. The kinds are leading, interior "
      "and trailing, and meters without use in the window get no_use_in_window. On synthetic "
      "data it equals a literal transcription of s05(e) exactly. Repair-round rules on top "
      "of the pilot's: (1) the window is trimmed to each meter's observed months (amendment "
      "v2); (2) a bimonthly bill that TPC posted wholly to its second month (C0 method 7: "
      "from 2025-07 for 12-33 % of bills, plus a few thousand earlier pairs, mostly "
      "2021-12/2022-01 and 2013) is split evenly first, so its first month is not a zero "
      "reading; (3) synchronized-zero marks are pair-consistent. The registered pilot-view "
      "check includes rules (1) and (2), and still lands within 0.5 % of the pilot.")
    w("")
    w("Structural zeros are a column, not a precedence class, so the run classes keep the "
      "pilot's definition. The plan's figure (878,738) is the pilot's s05 statistic over ALL "
      "meters (dropped ones included): readings exactly 0 before the install index minus "
      "one, with the install index being the first month start on or after the install "
      "date. The registered check recomputes it exactly from the pilot's archive "
      f"(reproduced: {st['s05_reproduced_exactly']}). A kept-meter product cannot reproduce "
      "that count, for two reasons. First, the product holds kept meters only (pilot kept "
      f"meters: {st['pilot_archive']['s05|kept_meters']['zero']:,} zeros + "
      f"{st['pilot_archive']['s05|kept_meters']['nan']:,} missing). Second, C0 codes every "
      "pre-install month 0, where the AMP_V2 cache left "
      f"{st['pilot_archive']['s05|all_meters']['nan']:,} of them missing. The like-for-like "
      "comparison (kept meters, install date floored to its month, zero or missing) gives "
      f"{st['shipped']['pilot_convention']:,} shipped against "
      f"{st['pilot_archive']['install_month|kept_meters']['zero_or_nan']:,} in the pilot "
      f"({pct(T6['checks']['structural_kept_meters']['rel_diff'])}). C0's whole-record count "
      f"without slack to 2025-07 is {st['shipped']['whole_record_no_slack']:,} (round 2's "
      "923,170 counts the record to 2026-07).\n")
    zp, zf = zb["preorigin"], zb["full"]
    w("Product zero classes (v2; one row per zero meter-month of the window; `structural` "
      "counted separately):\n")
    w("| variant, county | not_yet_used | idle | long_idle | terminated | trailing_short | "
      "never_used_in_window | structural rows | sync rows (flag 9) | terminated meters "
      "(last-12-month GWh) |\n|---|---|---|---|---|---|---|---|---|---|")
    for v, z in (("pre-origin", zp), ("full", zf)):
        fv = pre if v == "pre-origin" else full
        for c in L.COUNTIES:
            cc = z[c]["class_counts"]
            w(f"| {v}, {c} | {cc['not_yet_used']:,} | {cc['idle']:,} | {cc['long_idle']:,} | "
              f"{cc['terminated']:,} | {cc['trailing_short']:,} | "
              f"{cc['never_used_in_window']:,} | {z[c]['structural_rows']:,} | "
              f"{z[c]['sync_suspect_rows']:,} | {fv['terminated_meters'][c]:,} "
              f"({fv['terminated_last12_GWh'][c]:.2f}) |")
    w("")
    w("Against v1 (round 2's classifier) the pre-origin Yunlin terminations are 5,234 meters "
      "/ 6.97 GWh (v1 5,423 / 7.22) and Changhua's 6,261 / 8.36 (v1 6,489 / 9.37). v1 counted "
      "post-install missing readings outside the use span as zeros, and its structural class "
      "took precedence over the run kinds. Agreement with C0's termination flag is "
      f"{pre['agreement_with_c0_terminated']['yunlin']:.4f} (Yunlin) and "
      f"{pre['agreement_with_c0_terminated']['changhua']:.4f} (Changhua).\n")
    w("Screened synchronized-zero months (round 2's screen algorithm, on the classifier's "
      "observed matrix): pre-origin Yunlin "
      + ", ".join(zp["yunlin"]["sync_flagged_months"]) + " (identical to round 2); full "
      "Yunlin " + ", ".join(zf["yunlin"]["sync_flagged_months"]) + ". Splitting the "
      "second-month postings evenly changed the full-variant months: 2025-07 is added and "
      "2021-07 drops at month level, although its zero bills stay marked through the pair "
      "rule. The pair rule added "
      f"{zp['yunlin']['sync_rows_added_by_pair_rule']:,} pre-origin and "
      f"{zf['yunlin']['sync_rows_added_by_pair_rule']:,} full Yunlin rows.\n")
    # ---- P2b
    w("### P2b: the 2025 bills\n")
    w("**M0 July 2025 (plan rule, applied).** In the full product, every kept code-M0 monthly "
      "meter with an installed flag-0 2025-07 reading carries flag 9 and method "
      "`m0_july2025_suspect` (12), with its value kept. That is Yunlin "
      f"{m0e['yunlin']['pumps']:,} pumps ({m0e['yunlin']['GWh_2025_07']:.2f} GWh of the county's "
      f"{m0e['yunlin']['county_GWh_2025_07_all_pumps']:.2f} GWh in 2025-07), Changhua "
      f"{m0e['changhua']['pumps']:,} ({m0e['changhua']['GWh_2025_07']:.2f} of "
      f"{m0e['changhua']['county_GWh_2025_07_all_pumps']:.2f}) and Chiayi "
      f"{m0e['chiayi']['pumps']}. An aggregate is flagged 9 when such pumps hold more than "
      "10 % of its installed HP, C0's threshold for aggregates with missing readings: "
      f"{full['m0_2025_07_cells_flag9_new']:,} class x cell-months and "
      f"{full['m0_2025_07_total_cells_flag9']} of 2,148 total cells at 2025-07 (C0: "
      f"{full['m0_2025_07_total_cells_flag9_c0']}). Against the plan's 2022-2024 reference "
      "the trigger holds: Jul/Jun 1.096 (Yunlin) and 1.091 (Changhua) against 1.156-1.257. "
      "Against 2012-2024 it does not: 0.872-1.298 and 0.905-1.400. The M0 July zero share "
      "in 2025 (0.017 / 0.034) is ordinary. The flag is applied because the plan prescribes "
      "it, and the evidence for incompleteness is weak.\n")
    w("**Phase-A July-August 2025 (Yunlin): synchronized zeros, not incomplete pairs.** "
      "Yunlin phase-A meters (code M2), July-August bill by year (design-time look "
      "[P2-r3-explore-2025]):\n")
    w("| year | pair total GWh | zero-bill share of active meters | next-bill ratio, zero-bill "
      "meters | next-bill ratio, positive-bill meters | Yunlin-side rain Jul+Aug mm (rank of "
      "14) |\n|---|---|---|---|---|---|")
    for y in range(2019, 2026):
        k = f"{y}-07"
        w(f"| {y} | {x1a('yunlin', k, 'total_GWh'):.2f} | {x1a('yunlin', k):.3f} | "
          f"{x1a('yunlin', k, 'next_ratio_zero'):.2f} | "
          f"{x1a('yunlin', k, 'next_ratio_nonzero'):.2f} | {rmm[str(y)]['jul_aug']:.0f} "
          f"({rk[str(y)]}) |")
    w("")
    w("Points 1-4 below are why these bills are not P2b's 'incomplete pairs' (which would be "
      "NaN, never zero):\n")
    w("1. TPC meters are cumulative registers. A skipped or lost reading reappears in the next "
      "bill, which would be about twice the meter's usual next bill. The zero-bill meters' "
      "Sep-Oct 2025 bill is 1.15 times their own 2022-24 mean, against 1.12 for meters with "
      "a positive bill. Nothing was deferred.")
    w("2. The delivery records missing readings as NaN in both sheets (Yunlin phase-A meters: "
      "208-227 missing per month in mid-2019, 2 in 2025, which C0 flags 8). The 2025 zero "
      "bills are recorded zeros.")
    w("3. The same pattern appears in 2019 and 2021, both pre-origin, with zero shares 0.135 "
      "and 0.136 and pair totals of 8.0 and 8.9 GWh. Those are synchronized-zero months of "
      "the plan's own list (2019-07/08, 2021-07/08), which the plan keeps 'in the data with "
      "flag 9' as 'a sensitivity member only'. The monthly meters show the same August "
      "excess in 2025: zero share 0.043 against 0.006-0.027 in 2017-2024 (Changhua 0.098 "
      "against 0.023-0.039). The full-variant screen flags 2025-07 and 2025-08 in Yunlin "
      "(2025-08 in Changhua).")
    w("4. Changhua's phase-A zero share in Jul-Aug 2025 (0.030) lies inside its 2019-2024 "
      "range (0.014-0.042). Its total (11.00 GWh) is above 2021's 10.92 GWh, so Changhua "
      "has no deficit of its own. On the Yunlin side, July-August 2025 was the wettest of "
      "2012-2025 (1,060 mm). That is only partial corroboration: 2019 ranks 8th, and 2024 "
      "(3rd) shows no excess. The cause stays unverified, as the plan says.\n")
    w("The verifier's reading is therefore not taken as the primary treatment; it remains "
      "available as the members. In `recorded_syncfill` and `chowlin_syncfill`, every "
      "synchronized-zero meter-month is replaced by the meter's own mean of that calendar "
      "month over its three most recent eligible earlier years (flag 4, method "
      "`sync_zero_fill` (13), sd in-sample q90/1.645 x fill, uncalibrated). That is the P2b "
      "mask-and-impute treatment, applied to every synchronized month. In the full product, "
      "the members' July-2026 imputation is recomputed from the member record, so it no "
      "longer sees the 2025 zeros. Member effects (in grid):\n")
    w("| variant | county | 2012-2022 GWh/yr, recorded -> member | Jul+Aug 2019 GWh | "
      "Jul+Aug 2021 GWh | Jul+Aug 2025 GWh | filled meter-months |\n|---|---|---|---|---|---|---|")
    for v, fv in (("pre-origin", pre), ("full", full)):
        for c in L.COUNTIES:
            g = fv["member_in_grid_2012_2022_GWh_per_yr"][c]
            ja = fv["member_in_grid_jul_aug_GWh"]

            def jj(y, c=c, ja=ja):
                if str(y) not in ja:
                    return "-"
                r = ja[str(y)][c]
                return f"{r['recorded']:.2f} -> {r['recorded_syncfill']:.2f}"
            w(f"| {v} | {c} | {g['recorded']:.2f} -> {g['recorded_syncfill']:.2f} | {jj(2019)} | "
              f"{jj(2021)} | {jj(2025)} | {fv['sync_fill_meter_months'].get(c, 0):,} |")
    w("")
    w("The member fills only zero meter-months that the screen marks (active two months "
      "before and after). Yunlin's Jul+Aug 2025 rises by "
      f"{full['member_in_grid_jul_aug_GWh']['2025']['yunlin']['recorded_syncfill'] - full['member_in_grid_jul_aug_GWh']['2025']['yunlin']['recorded']:.2f}"
      " GWh in grid, less than the about 2.7 GWh county-wide phase-A deficit. The July "
      "backtest on the member (P2r3b.b2, reported) gives Yunlin "
      f"{t8['yunlin']['member_2013_2025']['mean_abs_total_err_pct']:.1f} % (2025 year +21.8 %), "
      f"Changhua {t8['changhua']['member_2013_2025']['mean_abs_total_err_pct']:.1f} % and Chiayi "
      f"{t8['chiayi']['member_2013_2025']['mean_abs_total_err_pct']:.1f} %. The plan's own "
      "variant (as-read July, 2013-2024) gives "
      f"{t8['yunlin']['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.2f} / "
      f"{t8['changhua']['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.2f} / "
      f"{t8['chiayi']['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.2f} %, as "
      "the verifier found. No July verdict changes. The members' July-2026 imputation is "
      f"{full['july2026']['yunlin']['imputed_recorded_syncfill_kwh']:.2f} GWh for Yunlin "
      f"against {full['july2026']['yunlin']['imputed_recorded_kwh']:.2f} in the recorded arm "
      f"(Changhua {full['july2026']['changhua']['imputed_recorded_syncfill_kwh']:.2f} vs "
      f"{full['july2026']['changhua']['imputed_recorded_kwh']:.2f}).\n")
    w("Round 2's deviation 'the plan's July-2025 problem does not exist' is retracted. The "
      "plan's specific problem, an open July-August 2025 bill at the record end, is absent "
      "from the 2026 delivery, but July-August 2025 does carry a synchronized-zero excess in "
      "Yunlin, which round 2 left unflagged in the arrays and unmentioned in the report.\n")
    w("### v1 -> v2 identity\n")
    w("Primary arms (recorded, chowlin): value, sd, totals and by-county arrays are identical "
      "to v1 in both variants. Flags are identical in the pre-origin product. In the full "
      f"product they differ only at 2025-07 ({full['v1_v2_recorded']['flag']['n_diff']:,} class "
      f"x cell-months and {full['v1_v2_recorded']['flag_total_all']['n_diff']} total cells: "
      "the M0 rule). New in v2: the two member arms, sync_fill.parquet, the "
      "`n_sync` arrays, the new zero tables, the new mask reasons and Chiayi's sd label.\n")
    # ---------------------------------------------------------------- products
    w("## Products\n")
    w("| product | content | allowed uses | fit_window_end |\n|---|---|---|---|")
    w("| `results/dataprep/yunlin_electricity/preorigin/v2` (L2; supersedes v1) | 2007-01..2022-12; "
      "arms recorded, chowlin and members recorded_syncfill, chowlin_syncfill; phase table, "
      "profiles, zero classes (zeros3), mask, sync fills | forcing_frozen, model_design, prior, "
      "prior_width_input, design_time_evidence, validator, after_the_fact_analysis (pre-origin "
      "uses through 3 exemptions) | 2026-07 (non-exempt 2022-12) |")
    w("| `results/dataprep/yunlin_electricity/full/v2` (L2; supersedes v1) | 2007-01..2026-07; "
      "same layout + July-2026 imputation (arms and members); M0 July-2025 flag 9 | "
      "forcing_recorded_arm, statistical_layer_history, validator, after_the_fact_analysis | "
      "2026-07 |")
    w("| `.../{preorigin,full}/v1` (round 2) | superseded: zero table from an unvalidated "
      "classifier; full v1 lacks the M0 flag | unchanged manifests (still re-hash) | |\n")
    w("Evidence: `results/dataprep/reports/P2abc_evidence/repair3/` (tests/, design_time/, "
      "checker, guard probe, report_facts3.json, rebuild comparison, round-2 report copies) and "
      "round 2's `P2abc_evidence/`. Ledger: `results/dataprep/ledger/rows/P2abc.csv`.\n")
    # ---------------------------------------------------------------- round-2 sections
    w("## Method (round 2, unchanged unless noted)\n")
    w("- **Phase.** C0's detected billing phase of the variant (pre-origin "
      "`phase_detected_2012_2022`, full `phase_detected`); the billing code (M2 = A, M1 = B) "
      "selects the profile populations together with the detected phase.")
    w("- **Profile.** One per county (Chiayi uses Yunlin's): stagger deconvolution of the in-grid "
      "kept irrigation phase-A (M2 & A) and phase-B (M1 & B) population sums, lambda 0.01, "
      "12-month share window; only pairs complete by the variant end.")
    w("- **Chow-Lin.** x = X beta + u, X = [1, p/mean p], AR(1) u, rho per meter by likelihood "
      "from {0, 0.5, 0.9}, window 2011-01..variant end; a pair is re-split only when both months "
      "are installed flag-0 observations; split clipped to [0, Y] keeping the pair sum. Every "
      "other month keeps C0's value, flag and method.")
    w("- **sd.** q90(n_eff, month)/1.645 x the even-split value from the validation groups "
      "pooled over both easting halves, log-linear in n_eff, clamped to [1, 300].")
    w("- **July 2026 (full).** The open phase-A month is imputed with the meter's own 3-year "
      "same-month mean of the Jul-Aug pair mean (flag 4, method 11) where the county's backtest "
      "passed.")
    w("- **Zero classes (P2c), repair round.** `zeros3.py` (see 'Repair round'); windows "
      "2012-01..2022-12 (pre-origin) and 2012-01..2026-06 (full); synchronized-zero months flag "
      "9 in the zero table and filled in the members.")
    w("- **M0 July 2025 (full), repair round.** flag 9, method 12, value kept.\n")
    w("## Pre-registrations\n")
    for k in ("P2abc.r2.c0_based.v1", "P2abc.r2.k1_repair.v1", "P2abc.r3.repair.v1",
              "P2abc.r3.repair.v2", "P2abc.r3.k1_repair.v1"):
        tag = " (superseded before scoring)" if k == "P2abc.r3.repair.v1" else ""
        w(f"- `{k}` sha256 `{SHA[k][:16]}...`{tag}")
    for k, v in R2["prereg"].items():
        if k.startswith("P2a.") or k.startswith("P2b.") or k.startswith("P2c."):
            w(f"- `{k}` sha256 `{v['sha256'][:16]}...` (round 1, Yunlin AMP_V2; stands)")
    w("")
    w("## Acceptance (round 2; the repair-round table is above)\n")
    w("| criterion | threshold | measured | pass |\n|---|---|---|---|")
    for a in acc2:
        w(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
          f"{'PASS' if a['passed'] else 'FAIL'} |")
    w("")
    w("## Results in brief (round 2, corrected wording)\n")
    w("**Meters (P2a).** On pseudo-bimonthly monthly meters the pre-origin Chow-Lin operator beats "
      "the even split at every group size: Yunlin skill 0.232-0.324, Changhua 0.205-0.414, "
      "fan-wide 0.50 in both. Pro-rata by the same profile is worse than the even split in "
      "Yunlin at every group size from 1 to 300 (n = 1: -0.259); fan-wide it is slightly better "
      "(+0.024). A year-shuffled placebo profile keeps most of the skill for single meters (Yunlin "
      "n = 1: 0.224 vs 0.232); year-specific timing pays off at aggregate scale (Yunlin n = 300: "
      "0.324 vs 0.056; fan 0.501 vs 0.048). For Changhua the real profile beats every placebo at "
      "every size.\n")
    w("**Head check (P2a).** D_w (real minus 20 year-shuffled placebo partial R2, 1-2 km, regional "
      "covariates recomputed from each energy version): Yunlin +0.0054 (p 0.051), Changhua "
      "-0.0012 (p 0.786), pooled +0.0027 (p 0.099); round 1 +0.0028 (p 0.139). All fail p < 0.05: "
      "the gain is not distinguishable from regional-signal injection. The fixed-nuisance "
      "secondary (p 0.003 / 0.005 / 0.002) changes no decision.\n")
    w("**July (P2b).** Rolling-origin |error| of the county-wide total of kept code-M2 phase-A "
      "meters (meters outside the active cells included), 2013-2025: Yunlin 15.4 % (max 32.1 %; "
      "1 km cell relative RMSE 0.345), Changhua 8.7 % (max 16.5 %; 0.214), Chiayi 42.2 % (max "
      "157 %; 98 meters, fails). July 2026 imputed: 55,591 of 55,606 open Yunlin phase-A meters "
      "and 45,423 of 45,436 Changhua meters; Chiayi's 101 stay NaN (flag 8).\n")
    # ---------------------------------------------------------------- key numbers
    w("## Key numbers (ledger ids in brackets)\n")
    w("Repair round:")
    for k in key_numbers_r3():
        w(f"- {k}")
    w("\nRound 2 (unchanged):")
    for k in R2["key_numbers"]:
        if "zero classes" in k and "pre-origin zero classes" in k:
            k = k + " [superseded by the v2 zero tables]"
        if "the 2025 jump is TPC posting" in k or ("M2 July zero share" in k and "2025" in k):
            k = k + (" [repair round: in Yunlin the 2025 July share also carries the "
                     "synchronized-zero excess of the phase-A July-August bills]")
        w(f"- {k}")
    w("")
    w("## Anti-leakage (U1-U7)\n")
    w("- Pre-origin v2: every fitted quantity uses electricity to 2022-12 (phase, profiles, "
      "beta, rho, sd model, zero classes, the screen, the members' fills and their q90). Its "
      "post-origin information is C0's two exemptions and the zero-check gate input, each "
      "listed in the manifest with a ledger id and a sensitivity statement. The pre-origin "
      "screen months equal round 2's.")
    w("- No head, ring, pile, GNSS or InSAR observation dated 2023-01 or later enters any "
      "test, gate, threshold or choice (U7). The repair-round design looks used electricity "
      "(and rain for corroboration only) and shaped only the full product's 2025 treatment.")
    w("- Imputed and filled values (flags 3 and 4) are forcing inputs with sd, never targets "
      "(U1/U2). Both manifests forbid scoring_target, baseline, datum_fit, calibration_target "
      "and selection. The guard probe passed 26/26.\n")
    w("## Conventions for P2def (and any consumer)\n")
    w(conventions())
    w("\n## Deviations from plan\n")
    for d in deviations():
        w(f"- {d}")
    w("\n## Issues for downstream\n")
    for d in issues():
        w(f"- {d}")
    w("\n## Provenance\n")
    w("- Code (repair round): " + ", ".join(f"`{p}`" for p in sorted(
        str(p.relative_to(L.REPO)) for p in P3.V3.glob("*.py"))))
    w("- Code (round 2, unchanged; hashed in the v1 manifests and imported read-only): "
      "`dataprep_staging/p2_census_repair/v2/*.py`; round 1: `dataprep_staging/p2_census_repair/"
      "*.py`.")
    w("- Compute: CPU only, single-threaded via cpu_slot.sh. Zero tables 82 s; assemblies "
      "213 s (pre-origin) and 272 s (full); checks 14 s (zero), 110-140 s (construction).")
    w("- Claude API tokens: 0. GPU: none. Nothing under hydrophysics/, tests/ or L0 was "
      "written, and nothing was staged in git.")
    return "\n".join(o) + "\n"


def key_numbers_r3() -> list[str]:
    pre, full = F["preorigin"], F["full"]
    t8 = T8["counties"]
    m0e = full["m0_2025_07_energy"]
    out = [
        f"P2c shipped classifier vs pilot (Yunlin, pilot view): max relative difference "
        f"{pct(max(v['rel_diff'] for v in T6['checks'].values()))} over 7 items; idle 0.35 %, "
        "long_idle 0.47 %, terminated 0.01 %, trailing_short 0.02 %, April 0.04 %, August "
        "0.09 %, structural 0.14 % [P2abc.r3.c1.zero_shipped]",
        "Structural: pilot s05 statistic recomputed exactly (878,738 zeros, 49,388 missing, 9,071 "
        "positive, all meters); kept meters with month-precision installs: pilot 901,713 vs "
        "shipped 902,981 [P2abc.r3.c1.zero_shipped]"]
    for v, fv in (("pre-origin", pre), ("full", full)):
        out.append(f"{v} terminated meters (last-12-month GWh): Yunlin "
                   f"{fv['terminated_meters']['yunlin']:,} ({fv['terminated_last12_GWh']['yunlin']:.2f}),"
                   f" Changhua {fv['terminated_meters']['changhua']:,} "
                   f"({fv['terminated_last12_GWh']['changhua']:.2f}), Chiayi "
                   f"{fv['terminated_meters']['chiayi']} ({fv['terminated_last12_GWh']['chiayi']:.2f})"
                   f" [P2abc.r3.zeros.{'preorigin' if v == 'pre-origin' else 'full'}]")
    out += [
        f"Synchronized-zero rows (flag 9): pre-origin Yunlin {F['zeros']['preorigin']['yunlin']['sync_suspect_rows']:,}, "
        f"Changhua {F['zeros']['preorigin']['changhua']['sync_suspect_rows']:,}, Chiayi "
        f"{F['zeros']['preorigin']['chiayi']['sync_suspect_rows']}; full "
        f"{F['zeros']['full']['yunlin']['sync_suspect_rows']:,} / "
        f"{F['zeros']['full']['changhua']['sync_suspect_rows']:,} / "
        f"{F['zeros']['full']['chiayi']['sync_suspect_rows']} [P2abc.r3.zeros.preorigin, "
        "P2abc.r3.zeros.full]",
        "Members, in-grid 2012-2022 GWh/yr (pre-origin): Yunlin 188.45 -> 188.90 (+0.24 %), "
        "Changhua 187.36 -> 187.67 (+0.16 %), Chiayi 0.626 -> 0.629; Yunlin Jul+Aug 2019 "
        "28.79 -> 29.74 GWh, 2021 34.11 -> 34.86 [P2abc.r3.syncfill.preorigin]",
        "Members, full: Yunlin Jul+Aug 2025 in grid 33.16 -> 34.20 GWh, Changhua 35.17 -> 35.40; "
        "filled meter-months Yunlin 162,671, Changhua 117,224, Chiayi 671 "
        "[P2abc.r3.syncfill.full]",
        f"M0 July 2025 (full): Yunlin {m0e['yunlin']['pumps']:,} pumps / "
        f"{m0e['yunlin']['GWh_2025_07']:.2f} GWh of {m0e['yunlin']['county_GWh_2025_07_all_pumps']:.2f};"
        f" Changhua {m0e['changhua']['pumps']:,} / {m0e['changhua']['GWh_2025_07']:.2f} of "
        f"{m0e['changhua']['county_GWh_2025_07_all_pumps']:.2f}; Chiayi {m0e['chiayi']['pumps']};"
        f" {full['m0_2025_07_total_cells_flag9']} of 2,148 total cells flagged 9 at 2025-07 "
        f"(C0 {full['m0_2025_07_total_cells_flag9_c0']}) [P2abc.r3.m0_july2025]",
        "Design-time 2025 evidence (Yunlin M2 phase A, Jul-Aug): zero-bill share 0.164 (2019 "
        "0.135, 2021 0.136, 2020/2022-24 0.061-0.079); next-bill ratio zero-bill 1.15 vs positive "
        "1.12; Changhua 0.030 (2019-24 0.014-0.042); M0 Aug zero share Yunlin 0.043 (2017-24 "
        "0.006-0.027), Changhua 0.098 (0.023-0.039); M0 Jul/Jun 2012-2024 Yunlin 0.872-1.298 "
        "[P2abc.r3.design.explore_2025]",
        "Yunlin-side Jul+Aug rain 2025 1,060 mm (wettest of 2012-2025); 2019 511 (8th), 2021 724 "
        "(5th), 2024 801 (3rd) [P2abc.r3.design.rain]",
        f"July backtest on the member (reported): Yunlin "
        f"{t8['yunlin']['member_2013_2025']['mean_abs_total_err_pct']:.1f} %, Changhua "
        f"{t8['changhua']['member_2013_2025']['mean_abs_total_err_pct']:.1f} %, Chiayi "
        f"{t8['chiayi']['member_2013_2025']['mean_abs_total_err_pct']:.1f} %; plan variant "
        f"(as-read July 2013-2024) {t8['yunlin']['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.2f}"
        f" / {t8['changhua']['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.2f} / "
        f"{t8['chiayi']['recorded_as_read_july_2013_2024']['mean_abs_total_err_pct']:.2f} % "
        "[P2abc.r3.b2.july_member.yunlin, .changhua, .chiayi]",
        f"July 2026 imputation, members vs arms (GWh): Yunlin "
        f"{full['july2026']['yunlin']['imputed_recorded_syncfill_kwh']:.2f} vs "
        f"{full['july2026']['yunlin']['imputed_recorded_kwh']:.2f}; Changhua "
        f"{full['july2026']['changhua']['imputed_recorded_syncfill_kwh']:.2f} vs "
        f"{full['july2026']['changhua']['imputed_recorded_kwh']:.2f} [P2abc.r3.facts.full]",
        "v1 -> v2: primary-arm values, sd and totals identical in both variants; flags differ "
        f"only at 2025-07 of the full product ({full['v1_v2_recorded']['flag']['n_diff']:,} class "
        "x cell-months) [P2abc.r3.facts.preorigin, P2abc.r3.facts.full]",
        "Construction: pre-origin 0 violations; full first assembly item (c) 1; full re-assembly 0 "
        "[P2abc.r3.k1.construction.preorigin, P2abc.r3.k1.first_assembly.full, "
        "P2abc.r3.k1r.construction.full]",
        f"Checker: {CHK} [P2abc.r3.checker]; guard probe {GP} [P2abc.r3.guard_probe]; "
        "pre-origin rebuild identical in content (15/15 files) [P2abc.r3.preorigin_rebuild]",
        "Ledger rows/P2abc.csv: 193 rows (142 before the round, of which 18 roles corrected to "
        "design_time_evidence; 51 added), 0 conflicts across fragments; Claude API tokens 0; GPU "
        "none"]
    return out


def conventions() -> str:
    return (
        "**Paths.** `results/dataprep/yunlin_electricity/preorigin/v2/` and `.../full/v2/` "
        "(MANIFEST.json, PRODUCT_CARD.md, validation.json in each); v1 is superseded. Open them "
        "with the guard (`sys.path.insert(0, '<repo>/dataprep_staging'); from common import "
        "guard`). The pre-origin product allows forcing_frozen, model_design, prior, "
        "prior_width_input, design_time_evidence, validator and after_the_fact_analysis "
        "(through three exemptions). The full product allows forcing_recorded_arm, "
        "statistical_layer_history, validator and after_the_fact_analysis.\n\n"
        "**Coverage.** Despite the name, the product covers Yunlin, Changhua and the 316 "
        "kept in-grid Chiayi pumps.\n\n"
        "**Time axis.** Every array has `dates` ('YYYY-MM'), index 0 = 2007-01. Pre-origin "
        "T = 192 (to 2022-12), full T = 235 (to 2026-07). Phase-A bills pair (Jan, Feb), ...: "
        "pairs start at even indices. Phase-B bills pair (Feb, Mar), ..., (Dec, Jan): odd "
        "indices.\n\n"
        "**Grid.** A = 2148 active cells of `results/twin_forward/datum_gate_innov.npz` (nx 59, "
        "ny 77, dx 1000 m, x0 162191.011, y0 2600419.535), row-major over masked cells; arrays "
        "carry `mask_hash` and `grid_hash`.\n\n"
        "**Labels.** Classes C = (irrigation, aquaculture, livestock, domestic, industry, "
        "other); counties K = (yunlin, changhua, chiayi). Unit: kWh/month in model units (the "
        "cleaned census, duty cap 1.0).\n\n"
        "**Arms.** `kwh_by_class_<arm>.npz` for arm in recorded, chowlin (primary) and "
        "recorded_syncfill, chowlin_syncfill (synchronized-zero sensitivity members, plan "
        "P2c). Each has the same keys: `value, flag (uint8), sd` (C, A, T); "
        "`value_by_county, flag_by_county, sd_by_county` (K, C, A, T); `value_total_all, "
        "flag_total_all, sd_total_all, n_eff_total_all` and `_nodomestic` (A, T); `n_eff`, "
        "`n_cl` (flag-3 pumps), `n_imp` (imputed pumps), `n_sync` (filled pumps; 0 in the "
        "primary arms); C0's `missing_expected_kwh` (+ `_all`, `_nodomestic`); grid metadata; "
        "`arm`. Choose an arm or member weights before results, on pre-origin evidence only "
        "(U7).\n\n"
        "**`pumps_<county>.npz`** (`yunlin`, `changhua`, `chiayi_ingrid`): the Chow-Lin arm "
        "per kept pump in C0's pump order. Keys: `uid` (well number, personal data), "
        "`meter_idx`, `cell` (-1 = kept pump outside the active cells), `purpose_class`, "
        "`hp`, `hp_share`, `value, flag, method, sd` (P, T). The recorded arm per pump is "
        "C0's `pump_electricity/<variant>/v3/pumps_<county>.npz`, plus "
        "`july2026_imputation.parquet` and the M0 flag (full). A member per pump = arm + "
        "hp_share x the meter fill of `sync_fill.parquet`.\n\n"
        "**Method codes (uint8).** C0's codes 0 observed, 1 pre_install, 2 "
        "observed_before_install_date, 3 missing, 4 incomplete_pair_2026_07, 5 "
        "partial_reading_cycle_2026_07, 6 terminated_trailing_run, 7 bill_in_second_month, "
        "8 uneven_pair and 9 fill_dec2022_dec_jan_pair; then 10 chowlin_split (flag 3), 11 "
        "own_3yr_same_month_mean (flag 4, July 2026), 12 m0_july2025_suspect (flag 9, full) "
        "and 13 sync_zero_fill (flag 4, members).\n\n"
        "**Flags.** 0 observed; 3 Chow-Lin split; 4 C0's December-2022 fill (pre-origin), the "
        "July-2026 imputation (full) or a member fill; 8 missing (value NaN); 9 suspect "
        "(C0's partial 2026-07 cycle; M0 July 2025 at pump level, and at aggregate level "
        "when the suspect pumps hold > 10 % of installed HP). Aggregates take the highest "
        "flag in the order 8 > 9 > 4 > 3 > 0, and a flag-8 aggregate is NaN. Missing "
        "post-install readings count as 0 in aggregates. `read_table` for forcing_frozen "
        "keeps flags 0-5; npz users apply flags themselves. Dropping flag 9 removes 650 of "
        "2148 full-product total cells at 2025-07, although their values are as read.\n\n"
        "**sd.** Flag 3: q90(n_eff, calendar month)/1.645 x the even-split energy (Yunlin "
        "calibrated; Changhua uncalibrated, too narrow at n >= 10; Chiayi uncalibrated, not "
        "checked). Flag 4: C0's December-2022 fill sd (uncalibrated per the C0 verifier), "
        "the July-2026 imputation q90_rel/1.645 x value (uncalibrated), or the member fill's "
        "in-sample q90/1.645 x fill (uncalibrated). Components add in quadrature. Cross-cell "
        "correlation is not modelled.\n\n"
        "**Tables (join on `county`, `meter_idx`).**\n"
        "- `meter_phase.parquet`: billing_code, phase_code, phase_detected, dec_jan_paired, "
        "class_first_pump, first_pump_in_grid, chowlin_phase, rho, n_pairs_resplit, "
        "profile_member, profile_county, chowlin_shipped.\n"
        "- `zero_class.parquet`: one row per zero meter-month of the window (pre-origin "
        "2012-01..2022-12, full 2012-01..2026-06). Columns: month, zkind (leading, interior, "
        "trailing, no_use_in_window), run_len, zclass (not_yet_used, idle, long_idle, "
        "terminated, trailing_short, never_used_in_window), structural (C0 pre_install), "
        "sync_suspect, flag (0/9), possible_replacement (= terminated).\n"
        "- `meter_zero_summary.parquet`: first_nz(_t), last_nz(_t), use_in_window, terminated "
        "(a trailing zero run >= 12 months), termination(_t), last12_kwh_before_termination, "
        "c0_terminated.\n"
        "- `incomplete_pair_mask.parquet`: month, reason (pre-origin: open_bill_dec2022_phaseB_"
        "c0_fill/_missing; full: open_bill_jul2026_phaseA, partial_cycle_jul2026_monthly, "
        "m0_july2025_suspect; both: pair_one_month_missing, bill_in_second_month, "
        "sync_zero_month), plus the C0 and arm flags and methods.\n"
        "- `sync_fill.parquet`: month, fill_recorded_kwh, sd_recorded_kwh, fill_chowlin_kwh, "
        "sd_chowlin_kwh (meter level).\n"
        "- `county_class_monthly.parquet`: county x scope (in_grid, county) x class (+ all, "
        "all_nodomestic) x date x arm (4 arms), with value, sd, flag, method and counts "
        "(n_chowlin_pumps, n_imputed_pumps, n_syncfill_pumps).\n"
        "- `profiles.npz`: prof_<county>, a_month_/b_month_<county>, n_A_, n_B_, t_end_.\n"
        "- `july2026_imputation.parquet` (full): imputed_recorded_kwh, imputed_chowlin_kwh, "
        "sd_*, imputed_recorded_syncfill_kwh, imputed_chowlin_syncfill_kwh, shipped.")


def deviations() -> list[str]:
    d = [
        "Scope (task update): Yunlin, Changhua and the 316 kept in-grid Chiayi pumps, built "
        "on C0's pump_electricity v3 (kept set, classes, cells, flags and methods unchanged). "
        "The plan's product name yunlin_electricity is kept.",
        "Profiles are per county. Changhua has its own; Chiayi uses Yunlin's and follows the "
        "Yunlin Chow-Lin verdict, as registered. Repair round: Chiayi's sd label is now "
        "'uncalibrated', because it was never checked on Chiayi meters (plan 4.7).",
        "P2b record end: the open bill of the 2026 delivery is July-2026 phase A. It is "
        "masked by C0 and imputed with flag 4 where the county's backtest passed. The plan's "
        "open July-August 2025 bill is absent. Round 2's sentence 'the July-2025 problem "
        "does not exist' is retracted: July-August 2025 carries a synchronized-zero excess "
        "in Yunlin (see 'Repair round').",
        "The plan's july2025_mask.parquet is `incomplete_pair_mask.parquet`. It lists open "
        "bills, one-month-missing pairs, second-month postings and, from the repair round, "
        "synchronized-zero months and M0 July-2025 suspects.",
        "The P2b imputation base is the (Jul+Aug)/2 pair mean, and the backtest years are "
        "2013-2025 (round 2). The imputation is flagged 4: the plan's table 2.4 gives 4 for "
        "the series' own temporal model, while the P2b text's 'flagged 5' contradicts it.",
        "Yunlin's phase-A July-August 2025 zero bills are treated as synchronized zeros "
        "(plan P2c: flag 9 in the data, sensitivity member), not as P2b incomplete pairs "
        "(NaN), because the evidence shows recorded zeros. The members carry the NaN-and-"
        "impute alternative.",
        "M0 July 2025 (plan P2b) is flagged 9 at pump level, for installed readings only. An "
        "aggregate takes flag 9 when the suspect pumps hold > 10 % of its installed HP (C0's "
        "threshold), not when any pump does. The trigger holds against the plan's 2022-24 "
        "reference only.",
        "P2c: the run classes follow the pilot's rules. The structural class is a separate "
        "column and is checked like for like (kept meters, month-precision installs); the "
        "plan's 878,738 is an all-meter statistic, recomputed exactly but not a product "
        "count. Repair-round additions to the pilot's rules: window trimmed to observed "
        "months (amendment v2); second-month postings split evenly in the classifier and "
        "screen input; pair-consistent synchronized-zero marks. The full-variant screen "
        "months therefore differ from round 2's (2025-07 added).",
        "Synchronized-zero screen (round 2, unchanged algorithm): Chiayi is screened with the "
        "pooled Yunlin + Chiayi population, a design change made after the first round-2 "
        "build; the flag-9 months enter only through the members.",
        "Members (repair round): synchronized-zero months filled in every county, Chiayi "
        "included, with in-sample q90 sd (uncalibrated). No backtest decides the members.",
        "Pre-origin December 2022 of Dec-Jan-paired meters keeps C0's flag-4 fill; round 1's "
        "edge fill is not used.",
        "Construction checks failed once in each round and were repaired under their own "
        "registrations: round 2 item (f), and repair-round full item (c). The fixes were "
        "chosen after reading the failure.",
        "The first repair registration (P2abc.r3.repair.v1) was amended (v2) before any "
        "score, after the first build of the zero tables, which were values only.",
        "Yunlin's round-2 and repair-round units are not fresh, and their outcomes were "
        "known (declared). The head-check wells and 2012-2022 heads of Yunlin are design-time "
        "data of the billing pilot; the ledger now records them as design_time_evidence.",
        "The pre-origin gate includes the zero check, which reads the Yunlin record to "
        "2025-07 (code identity). It is declared as the exemption "
        "P2abc.r3.exemption.zero_check_gate.preorigin; no value depends on it.",
        "The design-time counts (d0) predate a p2lib.py edit made before the round-2 "
        "registration. Changhua's validation-meter count there is 5,406, against 5,434 in "
        "the registered test, and the old p2lib was not kept. The round-2 stage-1 rerun at "
        "20:30Z had no look of its own.",
        "The superseded ledger copy ledger/superseded/P2abc.r2.csv also contains the three "
        "repair-round pre-registration rows, which write_prereg appended before the copy "
        "was taken.",
        "synthetic_post2016_{low,high}.npz and layer_response_localonly.csv belong to "
        "P2def and are not delivered here."]
    return d


def issues() -> list[str]:
    full = F["full"]
    return [
        "Use yunlin_electricity/{preorigin,full}/v2. v1 is superseded: its zero tables come "
        "from an unvalidated classifier, and full v1 lacks the M0 flag. v1 arrays of the "
        "primary arms are value-identical to v2.",
        "P2def conventions: see 'Conventions for P2def'. The new keys are the member arms "
        "(kwh_by_class_<arm>_syncfill.npz), n_sync, method codes 12/13, the zero-table "
        "columns zkind/structural/use_in_window, sync_fill.parquet and the new mask reasons.",
        "P2d inputs: terminated meters (possible_replacement True) and their "
        "last-12-month energy are in zero_class / meter_zero_summary v2. Pre-origin: Yunlin "
        "5,234 meters / 6.97 GWh, Changhua 6,261 / 8.36 GWh (v1 5,423 / 7.22 and 6,489 / "
        "9.37). The census is frozen at mid-2016 installs.",
        "Synchronized-zero members: the primary arms keep the recorded zeros. The members "
        "add +0.24 % (Yunlin) and +0.16 % (Changhua) of 2012-2022 in-grid energy, "
        "concentrated in the flagged months (Yunlin Jul+Aug 2019 +0.96 GWh, 2021 +0.75, "
        "2025 +1.04). Their cause is unverified (plan P2c). Weight or choose them before "
        "results, on pre-origin evidence only (U7).",
        f"Full product, M0 July 2025: {full['m0_2025_07_total_cells_flag9']} of 2,148 total "
        "cells carry flag 9 at 2025-07 (values as read). The flagged pumps hold 44 % "
        "(Yunlin) and 35 % (Changhua) of the county's 2025-07 energy. A consumer that drops "
        "flag 9 loses that month in those cells, so it should keep the values or treat the "
        "month as a sensitivity.",
        "Changhua's Chow-Lin sd is uncalibrated (too narrow at n >= 10), and so is Chiayi's "
        "(never checked). C0's Dec-2022 fill sd, the July-2026 imputation sd and the member "
        "fill sd are uncalibrated too. Forcing-uncertainty propagation (plan 4.8) must not "
        "take them at face value.",
        "The Chow-Lin arm is not a proven improvement at cell scale: a year-shuffled profile "
        "nearly matches it for single meters, and the registered head check finds no "
        "placebo-controlled gain. Treat recorded vs Chow-Lin as a forcing-timing "
        "sensitivity.",
        "Changhua's domestic class is 23 % of in-grid energy; value_total_nodomestic is the "
        "sensitivity member (P4m).",
        "Full product: from 2025-07 the recorded arm keeps TPC's second-month postings, and "
        "the Chow-Lin arm re-splits them. 2026-07 is partial.",
        "Model integration after the freeze: the twin's cached forcing is C0 preorigin/v1 "
        "Yunlin. Recalibrate on this product (an arm chosen before results) or on C0 v3.",
        "Do not delete results/dataprep/yunlin_electricity/_work_p2abc_v2 (stage-1 arrays, "
        "inputs of v1 and v2) or _work_p2abc_v3/zeros (zero tables, inputs of v2). "
        "_work_p2abc_v3/assembled_* are staging copies of v2 and may be removed. "
        "_work_p2abc holds round-1 evidence.",
        "ledger/data_roles.csv was not reassembled (shared file). rows/P2abc.csv has 193 "
        "rows and 0 conflicts across fragments."]


def main() -> None:
    acc3, acc2 = build()
    text = md(acc3, acc2)
    (REP / "P2abc.md").write_text(text, encoding="utf-8")
    acc_all = [{k: a[k] for k in ("criterion", "threshold", "measured", "passed")}
               for a in acc3 + acc2]
    js = {
        "wp": "P2abc", "status": "passed",
        "summary": text.split("## Summary\n\n", 1)[1].split("\n## ", 1)[0].strip(),
        "repair_round": {
            "date": "2026-10-14",
            "prereg": {k: SHA[k] for k in ("P2abc.r3.repair.v1", "P2abc.r3.repair.v2",
                                           "P2abc.r3.k1_repair.v1")},
            "v1_superseded_before_scoring": True,
            "products": ["results/dataprep/yunlin_electricity/preorigin/v2",
                         "results/dataprep/yunlin_electricity/full/v2"],
            "evidence": "results/dataprep/reports/P2abc_evidence/repair3"},
        "prereg": {**R2["prereg"], **{k: {"sha256": SHA[k]} for k in (
            "P2abc.r3.repair.v1", "P2abc.r3.repair.v2", "P2abc.r3.k1_repair.v1")}},
        "acceptance": acc_all,
        "acceptance_number_ids": [a["id"] for a in acc3 + acc2],
        "products": [
            {"path": "results/dataprep/yunlin_electricity/preorigin/v2", "layer": "L2",
             "supersedes": "preorigin/v1"},
            {"path": "results/dataprep/yunlin_electricity/full/v2", "layer": "L2",
             "supersedes": "full/v1"}],
        "deviations_from_plan": deviations(), "issues_for_downstream": issues(),
        "conventions_for_P2def": conventions(),
        "key_numbers": key_numbers_r3() + R2["key_numbers"],
        "ledger": {"fragment": "results/dataprep/ledger/rows/P2abc.csv", "rows": 193,
                   "superseded": "results/dataprep/ledger/superseded/P2abc.r2.csv",
                   "roles_corrected": 18},
        "code_paths": code_paths(),
        "evidence": ["results/dataprep/reports/P2abc_evidence",
                     "results/dataprep/reports/P2abc_evidence/repair3"],
        "claude_api_tokens": 0, "gpu": "none"}
    L.save_json(js, REP / "P2abc.json")
    print("written", REP / "P2abc.md", len(text))


if __name__ == "__main__":
    main()
