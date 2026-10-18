"""WP S1b repair round: ledger rows, repair provenance and the updated ``reports/S1b.json``.

    python -I dataprep_staging/s1_rings/s1b/s1b_r2_report.py

Reads only S1b's own evidence (round-1 files, unchanged, and ``S1b_evidence/repair/*.json``),
the pre-registration and the look log. Appends the repair round's ledger rows (and the
design-time-evidence rows round 1 left out) to ``ledger/rows/S1b.csv``; re-verifies the
registered inputs and every hash of the round-1 ``PROVENANCE.json``; writes
``S1b_evidence/repair/PROVENANCE.json`` and ``reports/S1b.json`` (the round-1 version is kept in
``S1b_evidence/round1/``). ``reports/S1b.md`` is written by hand from the same files.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

EV = K.EVIDENCE
REP = EV / "repair"
R1 = EV / "round1"
REPORTS = K.DATAPREP / "reports"
CODE_R2 = ["s1b_r2_core.py", "s1b_r2_looks.py", "s1b_r2_readings.py", "s1b_r2_synth.py",
           "s1b_r2_u1.py", "s1b_r2_bridge.py", "s1b_r2_report.py", "tests/test_s1b_r2.py"]
EVID_R2 = ["readings.json", "synthetic_plan_literal.json", "u1_and_sensitivity.json",
           "bridge_windows.json", "code_at_run.json"]

GNSS = "results/dataprep/gnss_clean/full/v2"
INSAR = "results/dataprep/insar_069A/v2"
RINGS = "results/dataprep/rings_bridged/v2"
PILOT = "results/dataprep/_pilots/subsidence_gap_sources"
S1A_MD = "results/dataprep/reports/S1a.md"
S3_MD = "results/dataprep/reports/S3.md"
S4B_MD = "results/dataprep/reports/S4b.md"
ROWS = "results/dataprep/reports/S1b_evidence/registered_rows.csv"
REGJ = "results/dataprep/reports/S1b_evidence/registered.json"
AFTJ = "results/dataprep/reports/S1b_evidence/after_the_fact.json"
PV = {GNSS: "gnss_clean/full/v2", INSAR: "insar_069A/v2", RINGS: "rings_bridged/v2",
      PILOT: "pilot 2026-10-12", S1A_MD: "S1a report 2026-10-14", S3_MD: "S3 report (r3 repair)",
      S4B_MD: "S4b report (repair)", ROWS: "S1b round-1 evidence", REGJ: "S1b round-1 evidence",
      AFTJ: "S1b round-1 evidence"}


SUMMARY = (
    "Repair round (2026-10-14) of WP S1b, the independent GNSS (c) and InSAR (d) checks of "
    "rings_bridged/v2. Both verifier majors were accepted; the new evidence is after the fact "
    "(logged looks, S1b_evidence/repair/), and the registered result is unchanged and was not "
    "re-scored. Major 1: under the plan's literal '+-30-40 ppm excess-rate bracket' (the "
    "pilot's unwidened pre/post range) check (c) fails: 3/10 WRA sites contain the bridged "
    "value, the other plan-literal readings give 4-6/10 and k = 1 gives 7/10; the registered "
    "9/10 needs the bracket widened by at least 1.02 sigma. A synthetic run made after scoring "
    "shows that a correct product would almost always fail the plan-literal reading at this "
    "GNSS noise (family pass rate 0.00 unwidened, at most 0.615), and the real counts are what "
    "a correct product would give (3/10 against 3.6 expected), so the plan-literal failure is "
    "uninformative rather than evidence against the product; but the widening was chosen after "
    "the pilot table had shown unwidened failures, so the (c) PASS is a pass of S1b's own rule "
    "only. Major 2: the round-1 U1 sentence was wrong. 19 % of the GNSS days in (c) are flag 3 "
    "(common-mode-only step corrections: all of GFES's post window and endpoints, most of the "
    "JMES and TKJS post windows), and every InSAR gap change crosses two LiCSBAS-bridged network "
    "gaps (Beichen three). Split with 36-day epoch sets, the model-set (flag-4) window "
    "contributes -12.8..+7.0 mm at 13 sites (median share of G 0.09; Beichen -39.6 mm) and "
    "agrees with GNSS within the scatter of same-shaped within-segment windows (mean +3.1 mm "
    "against a 13.6 mm common-mode scatter; no site-specific excess once Kecuo, whose KTES "
    "in-gap step falls in the window, is excluded). The U1-strict readings pass (report only): "
    "(c) on flags 0-1 6/7, (d2) with the model-set window at 100 % uncertainty 14/14, (d1) from "
    "interferometric increments only r 0.593 against raw 0.245. Minor findings: (d1)'s "
    "bridged-over-raw clause is a point comparison not supported at 5 % (Williams p 0.072); "
    "against the post-gap window alone the raw value is closer to InSAR than the bridged one "
    "(24.4 vs 31.7 mm), so d2's RMSE half rests on the midpoint; the ring-residual-sd variant "
    "changes nothing; the declared InSAR epoch noise (6.6 mm) is about half the empirical upper "
    "bound (13-14 mm). Status stays partial: all three products are gate-failed and read after "
    "the fact, the check is design-exposed, the plan-literal (c) fails, and no part detects a "
    "25 % scale error. No product version (validation only); nothing under hydrophysics/, "
    "tests/ or L0; nothing staged; no GPU; 0 Claude API tokens.")
DEVIATIONS = [
    "Validators gate-failed (S3 A1 0.863 < 0.90; S4b G1 13.2 > 10 mm): read as "
    "after_the_fact_analysis (fallback); results are after-the-fact evidence, not a validator "
    "gate",
    "(c) bracket made operational as the pre/post excess-rate bracket widened by 2 sigma; the "
    "80 % and RMSE-vs-raw rules are S1b's (the plan gives no number). Repair round: under the "
    "plan-literal bracket (c) fails (3/10 unwidened; 4-6/10 for +-30/40 ppm readings); the "
    "widening was chosen after the pilot table had shown unwidened failures (design exposure)",
    "NGL family has one pair within 5 km (Jiaxing-DNAN): reported, not judged",
    "(d) 'offset-only InSAR ties (S4)' not implemented: the check uses each instrument's own "
    "change and rates; insar_069A/v2 is stable-area referenced with no GNSS tie",
    "Units declared by ring-site and station id. Corrected wording (repair round): the "
    "registration's cell-based dry run returned no blocking look because design-time looks made "
    "before a unit's first declaration never block under A1's rule; it is not evidence that no "
    "earlier look viewed those cells (S4b's all-cell InSAR looks did)",
    "Resumed session (round 1): code re-checked, GNSS step-se term and family-level operating "
    "characteristics added, synthetic re-run (logged) before registration",
    "No product version: validation only. Repair evidence is a new directory "
    "(S1b_evidence/repair/); round-1 code, evidence and PROVENANCE.json are unchanged (17 "
    "hashes re-verified); the round-1 S1b.md and S1b.json are kept in S1b_evidence/round1/",
    "Repair round: plan-literal and U1-strict readings, a dependent-correlation test, synthetic "
    "operating characteristics of the plan-literal readings (seed 1803) and the InSAR "
    "model-set-window split were computed after scoring (logged looks); they do not replace "
    "the registered result",
    "Repair round: the first split of the InSAR gap change at the network gaps used single "
    "epochs and was dominated by epoch noise (single-epoch 'bridged' sums -4.6..+73.9 mm); it "
    "is superseded by the 36-day epoch-set split (bridge_windows.json, re-run once to add "
    "per-period noise rows; both runs logged)",
    "Repair round: the verifier's unlogged looks were logged on its behalf "
    "(verification_disclosure, 2026-10-14T00:41Z)",
]
ISSUES = [
    "S3 / gnss_clean: SJES carries an uncorrected offset of about -100 mm inside its "
    "2022-12-22..2023-02-24 data gap, reversing the corrected +94.9 mm step of 2022-03-29. With "
    "the registered endpoint rule (+-3 d at 2021-12-01; +-7 d at 2023-02-20, where no flag-0/1/3 "
    "day lies within +-3 d) the corrected gap change is 125.8 mm, the raw one 30.9 mm (31.3 mm "
    "with +-7 d at both ends; corrected 126.2 mm), InSAR 21.4 mm. Levels across that interval "
    "are wrong by about 10 cm",
    "S3: other GNSS data gaps inside the ring gap where an SJES-like offset would go undetected: "
    "TKJS 54 d (2022-04-03..05-28), STES 32 d, HNES 20 d, SSES 18 d; off-pair WRA stations XPES "
    "182 d, MDES 132 d, TCBS 69 d, LYES 51 d, JJES 38 d, CYES no data. InSAR-GNSS gap-change "
    "agreement at TKJS, STES and HNES (-1.3, +4.9, +0.4 mm) argues against large offsets there; "
    "SSES differs by -11.4 mm",
    "S3 / KTES: the in-gap step (2022-08-11, -33.2 mm) falls inside InSAR's model-set window "
    "(2022-05..2022-10), where InSAR is set by LiCSBAS's constraint; Kecuo is the only site "
    "whose window disagrees with GNSS (+43.9 mm), so InSAR cannot adjudicate the step",
    "S4b / insar_069A: empirical per-epoch noise of the 3 x 3 pixel median at the co-located "
    "stations is at most 13-14 mm (6 d and 12 d within-segment InSAR-minus-GNSS increments), "
    "against 6.6 mm declared by S1b and S4b's 9.4-9.7 mm single-interval figure; the epochs next "
    "to the 2022 network gaps scatter by 15-35 mm. Across the 2022 gaps the model-set window "
    "agrees with GNSS within the within-segment null. Post-gap InSAR rates fitted within "
    "segments exceed GNSS by a median 27.6 mm/yr (7.8 without segment intercepts; -22.3 in "
    "2020-21): check rate-based uses of 2023-25 InSAR",
    "S1a (report only, never fed back): GNSS is consistent with every evidence-based "
    "assignment; it reads A at Yuanchang, neither at Hunan (C_A 0.3 mm outside), both at Kecuo "
    "(A with the KTES step kept). An S1a evidence line would need a gate-passing GNSS/InSAR "
    "product, a new registration and fresh windows",
    "S1b cannot detect a 25 % scale error (power 0.06-0.12): the scale half of S1a's (e) stays "
    "untested by independent instruments; S1b supports the existence and rough size (about "
    "5 cm) of the group-A correction only",
    "Plan revision: the plan's literal (c) test (pre/post excess-rate range, about +-30-40 ppm) "
    "has no noise term; a correct product passes it at 0.00-0.615 per family (synthetic). Any "
    "revised test needs a noise term, a new registration and fresh data; this registration is "
    "spent and design-exposed",
    "Citing S1b: cite (d1) as 'r_bridged above the critical r (0.714; 0.593 from interferometric "
    "increments only); bridged-over-raw not significant (p 0.07)'; cite d2's RMSE half with its "
    "per-window caveat; do not cite d2's containment count (raw inside at 13/14)",
    "USER DECISIONS pending: the U7 reading for S1's inherently post-gap checks; the S3 and S4b "
    "gate decisions; whether the plan-literal (c) failure or S1b's registered rule governs",
    "Registered code files must stay unchanged (s1b_run.py registered refuses otherwise); port "
    "s1b/ with the S1 port after the freeze (plan 5.5); ring_holdout.py untouched",
    "Before any commit of dataprep_staging, run the CLAUDE.md Rule 1 scan (over the S1b files it "
    "hits only the word 'tokens'); no coordinates or data in the code",
]


def ledger_rows() -> list[dict]:
    spec = {
        # design-time-evidence rows round 1 left out (verifier minor 5)
        "S1b.d1_insar_correlation": [(S4B_MD, "design_time_evidence",
                                      "S4b's InSAR-GNSS results read before the design")],
        "S1b.d2_insar_bracket": [(S1A_MD, "design_time_evidence",
                                  "S1a's report (GNSS sensitivity, membership) read before the "
                                  "design")],
        "S1b.c_ngl_near": [(PILOT, "design_time_evidence", "gnss_vs_ring*.csv read before the "
                            "design"),
                           (S1A_MD, "design_time_evidence", "S1a's GNSS sensitivity table read "
                            "before the design")],
        "S1b.verdict": [(S3_MD, "design_time_evidence", "A3/A2 read before the design"),
                        (S4B_MD, "design_time_evidence", "S4b InSAR-GNSS read before the "
                         "design")],
        # repair-round numbers
        "S1b.r2.c_plan_literal_readings": [
            (GNSS, "validator", "via the registered rows (after_the_fact_analysis)"),
            (RINGS, "input", "values under test"),
            (ROWS, "input", "registered bracket ends and sigmas"),
            (PILOT, "design_time_evidence", "gnss_vs_ring_stepfixed.csv defines the plan's "
             "+-30-40 ppm bracket (pre/post range)")],
        "S1b.r2.d2_readings": [(INSAR, "validator", "via the registered rows"),
                               (RINGS, "input", "values under test"),
                               (ROWS, "input", "registered bracket ends and sigmas")],
        "S1b.r2.per_window_rmse": [(GNSS, "validator", "via the registered rows"),
                                   (INSAR, "validator", "via the registered rows"),
                                   (RINGS, "input", "values under test"),
                                   (ROWS, "input", "C_pre, C_post per site")],
        "S1b.r2.d1_dependent_correlation": [(INSAR, "validator", "InSAR gap changes"),
                                            (RINGS, "input", "values under test"),
                                            (ROWS, "input", "I_gap, C_primary, C_raw")],
        "S1b.r2.synthetic_plan_literal_oc": [
            (RINGS, "input", "survey dates, sd_s, L, P(A) only (values replaced)"),
            (GNSS, "input", "dates, flags, sigma_hf, rho_hf, step se only"),
            (INSAR, "input", "valid-pixel masks and network segments only")],
        "S1b.r2.gnss_flag_composition": [(GNSS, "input", "daily flags of the used series"),
                                         (RINGS, "input", "survey dates (windows)")],
        "S1b.r2.single_epoch_bridge_split_superseded": [
            (INSAR, "input", "single epochs next to the network gaps"),
            (GNSS, "input", "+-3 d means at those epochs"),
            (RINGS, "input", "survey dates")],
        "S1b.r2.insar_model_set_windows": [(INSAR, "input", "36 d epoch sets around the gaps"),
                                           (GNSS, "input", "+-3 d means at the same epochs"),
                                           (RINGS, "input", "survey dates"),
                                           (ROWS, "input", "registered I_gap")],
        "S1b.r2.insar_epoch_noise": [(INSAR, "input", "6/12 d within-segment increments"),
                                     (GNSS, "input", "same intervals")],
        "S1b.r2.u1_strict_readings": [(GNSS, "validator", "flags 0-1 variant (round-1 "
                                       "after_the_fact.json)"),
                                      (INSAR, "validator", "model-set windows at 100 % sd; "
                                       "interferometric-only d1"),
                                      (RINGS, "input", "values under test"),
                                      (AFTJ, "input", "registered variant gnss_flags_0_1"),
                                      (ROWS, "input", "registered bracket ends")],
        "S1b.r2.ring_residual_sd_variant": [(RINGS, "input", "ring surveys (residual sd) and "
                                             "values under test"),
                                            (GNSS, "validator", "registered pipeline"),
                                            (INSAR, "validator", "registered pipeline")],
        "S1b.r2.gnss_data_gaps_in_ring_gap": [(GNSS, "input", "days with flag 0/1/3"),
                                              (RINGS, "input", "survey dates")],
        "S1b.r2.sjes_endpoints": [(GNSS, "input", "SJES corrected and raw daily heights"),
                                  (RINGS, "input", "Xizhou survey dates")],
        "S1b.r2.unverifiable_prior_looks_summary": [(REGJ, "input", "assert_can_score's "
                                                     "unverifiable_prior_looks")],
    }
    rows = []
    for num, items in spec.items():
        for src, role, note in items:
            rows.append({"number_id": num, "source": src, "role": role,
                         "product_version": PV[src], "note": note})
    return rows


def verify_round1() -> dict:
    p = json.loads((EV / "PROVENANCE.json").read_text())
    bad = []
    for sec in ("code_now", "evidence"):
        for it in p[sec]:
            f = REPO / it["path"]
            if not f.exists() or sha256_file(f) != it["sha256"]:
                bad.append(it["path"])
    return {"n_checked": sum(len(p[s]) for s in ("code_now", "evidence")), "mismatches": bad}


def derived(rd: dict, sy: dict, u1: dict, bw: dict, reg: dict, aft: dict) -> dict:
    out = {}
    # GNSS flag-3 share over the registered c_wra series
    fc = [r for r in u1["gnss_flag_composition"] if r["family"] == "wra_colocated"]
    tot = sum(r["pre_days"] + r["post_days"] for r in fc)
    f3 = sum(r["pre_flag3"] + r["post_flag3"] for r in fc)
    out["gnss_flag3"] = {
        "c_wra_days_total": tot, "c_wra_flag3_days": f3, "share": f3 / tot,
        "stations_with_flag3": {r["station"]: {"pre": r["pre_flag3"], "post": r["post_flag3"],
                                               "post_days": r["post_days"],
                                               "endpoint_days": r["endpoint_flag3_days"]}
                                for r in fc if r["pre_flag3"] + r["post_flag3"] > 0}}
    # model-set windows: Kecuo excluded (the KTES in-gap step falls inside the window)
    per = bw["first_window_vs_gnss"]["per_site_mm"]
    xk = [v for k, v in per.items() if k != "Kecuo"]
    nsd = bw["first_window_vs_gnss"]["null_sd_about_window_mean_mm"]
    out["model_set_windows_excl_Kecuo"] = {
        "n": len(xk), "mean_mm": float(np.mean(xk)), "sd_mm": float(np.std(xk, ddof=1)),
        "site_specific_bridge_error_est_mm": math.sqrt(max(0.0, float(np.var(xk, ddof=1))
                                                           - nsd ** 2))}
    B = [s["B_mm"] for s in bw["sites"] if s["site_en"] != "Beichen"]
    out["model_set_B_range_13_sites_mm"] = [min(B), max(B)]
    out["model_set_B_share_of_G_median"] = float(np.median(
        [s["B_share_of_G"] for s in bw["sites"]]))
    nz = collections.defaultdict(list)
    for r in bw["epoch_noise"]:
        nz[(r["lag_d"], r["period"])].append(r["implied_epoch_sd_mm_upper"])
    out["epoch_sd_upper_median_mm"] = {f"{k[0]}d|{k[1]}": float(np.median(v))
                                       for k, v in nz.items()}
    sb = [r for r in u1["insar_bridges"]]
    out["single_epoch_bridged_sum_range_mm"] = [min(r["bridged_sum_mm"] for r in sb),
                                                max(r["bridged_sum_mm"] for r in sb)]
    # unverifiable looks
    u = [x for x in reg["unverifiable_prior_looks"] if x["test_id"] == "S1b.c_wra_colocated"]
    c = collections.Counter(x["look"]["wp"] for x in u)
    cp = collections.Counter(x["look"]["purpose"] for x in u)
    out["unverifiable_prior_looks"] = {
        "records_in_registered_json": len(reg["unverifiable_prior_looks"]),
        "distinct_looks": len(u), "tests": 4, "by_wp": dict(c.most_common()),
        "by_purpose": dict(cp.most_common()),
        "listed_in_preregistration": 50,
        "note": "looks of other WPs (2026-10-11..13) on units of unknown membership whose "
                "windows overlap S1b's; they block by unit id only and none names an S1b "
                "unit; the pre-registration lists the first 50 (test, look) records"}
    # synthetic plan-literal per-site containment (correct member), WRA family
    ps = [r for r in sy["per_site"] if r["family"] == "wra_colocated"]
    out["synthetic_wra_expected_contained_correct"] = {
        rule: float(sum(r[f"{rule}|correct"] for r in ps)) for rule in sy["rules"]}
    out["flags01_variant"] = {k: aft["variants"]["gnss_flags_0_1"]["verdicts"]["c_wra"][k]
                              for k in ("n_evaluable", "n_contain_primary", "n_contain_raw",
                                        "rmse_primary_mm", "rmse_raw_mm", "pass")}
    return out


def acceptance(reg: dict, rd: dict, bw: dict, u1: dict, aft: dict) -> list[dict]:
    r1 = json.loads((R1 / "S1b.json").read_text())["acceptance"]
    acc = [r1[0]]
    c_ngl = dict(r1[1])
    c_ngl.update({"criterion": "(c) GNSS, NGL family within 5 km [S1b.c_ngl_near]: judgeable "
                               "as a family (>= 3 evaluable sites)",
                  "threshold": "judged only with >= 3 evaluable sites (registered)",
                  "measured": "1 evaluable site (Jiaxing-DNAN, 3.1 km): it contains C_primary "
                              "and excludes raw (RMSE 4.6 vs 45.7 mm), but n = 1, so the family "
                              "is reported, not judged",
                  "passed": False})
    acc += [c_ngl, r1[2], r1[3], r1[4]]
    f = rd["families"]["c_wra"]["rules"]
    lit = {k: f[k]["primary"] for k in ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:30",
                                        "mid_ppm:40", "k:1")}
    acc.append({
        "criterion": "(c) plan-literal reading (repair round, after the fact): C_primary within "
                     "the plan's '+-30-40 ppm excess-rate bracket' (the pilot's unwidened "
                     "pre/post range), WRA co-located family",
        "threshold": ">= 80 % of evaluable sites (S1b's family rule; the plan gives no count)",
        "measured": (f"unwidened range {lit['k:0']['n_contain']}/10; range +-30 ppm "
                     f"{lit['range_ppm:30']['n_contain']}/10; +-40 ppm "
                     f"{lit['range_ppm:40']['n_contain']}/10; |C - C_mid| <= 30 ppm "
                     f"{lit['mid_ppm:30']['n_contain']}/10, <= 40 ppm "
                     f"{lit['mid_ppm:40']['n_contain']}/10; k = 1 {lit['k:1']['n_contain']}/10. "
                     f"The registered 9/10 needs k >= {rd['families']['c_wra']['k_star_primary']:.2f}. "
                     "Synthetic, correct product: family pass rate 0.00 (unwidened), 0.41/0.615 "
                     "(+-30/40 ppm range), 0.03/0.105 (mid +-30/40 ppm)"),
        "passed": False})
    acc.append({
        "criterion": "Plan-literal S1b conjunction (repair round, after the fact)",
        "threshold": "plan-literal (c) AND d2 AND d1",
        "measured": "fails on (c) under every plan-literal reading; d2 and d1 as registered",
        "passed": False})
    us = bw["readings"]
    fl = aft["variants"]["gnss_flags_0_1"]["verdicts"]["c_wra"]
    acc.append({
        "criterion": "U1-strict reading (repair round, after the fact): no flag-3 GNSS day, "
                     "the InSAR model-set (flag-4) window carried at 100 % uncertainty, d1 from "
                     "interferometric increments only",
        "threshold": "the registered rules applied to the U1-strict inputs",
        "measured": (f"c_wra flags 0-1: {fl['n_contain_primary']}/{fl['n_evaluable']} (raw "
                     f"{fl['n_contain_raw']}; Guangfu, Tuku, Zhengmin not evaluable); d2: "
                     f"{us['d2_B_at_100pct']['primary']['n_contain']}/14 (raw "
                     f"{us['d2_B_at_100pct']['raw']['n_contain']}/14); d1: r "
                     f"{us['d1_interferometric_only']['r_bridged']:.3f} vs raw "
                     f"{us['d1_interferometric_only']['r_raw']:.3f} (r_crit 0.458)"),
        "passed": bool(fl["pass"] and us["d2_B_at_100pct"]["primary"]["containment_pass"]
                       and us["d1_interferometric_only"]["pass"])})
    d1r = rd["d1_rates"]
    acc.append({
        "criterion": "(d1) second clause with uncertainty (repair round, report only, not "
                     "registered): r_bridged > r_raw beyond chance",
        "threshold": "Williams t one-sided p < 0.05 (report only)",
        "measured": (f"t {d1r['williams']['t']:.2f}, df 11, p {d1r['williams']['p_one_sided']:.3f};"
                     f" bootstrap 90 % CI of r_b - r_r [{d1r['bootstrap_diff']['ci90'][0]:.2f}, "
                     f"{d1r['bootstrap_diff']['ci90'][1]:.2f}]; Fisher 90 % CI of r_b "
                     f"[{d1r['fisher90_r_bridged'][0]:.2f}, {d1r['fisher90_r_bridged'][1]:.2f}]"),
        "passed": bool(d1r["williams"]["p_one_sided"] < 0.05)})
    acc.append({
        "criterion": "Registration integrity",
        "threshold": "pre-registration predates every look at the test data; registered inputs "
                     "unchanged",
        "measured": "assert_can_score passed at scoring; 0 of 46 registered inputs changed at "
                    "repair time; post-registration looks are after_the_fact, diagnose, "
                    "verification_disclosure and integrity_check only (no re-score)",
        "passed": True})
    acc.append({
        "criterion": "Power against a 25 % scale error (stated before scoring; report only)",
        "threshold": "reported (no plan threshold)",
        "measured": "family pass rate under a +-25 % group-A scale error 0.885-0.94 (WRA, "
                    "InSAR), i.e. power 0.06-0.12",
        "passed": False})
    return acc


def main() -> int:
    doc, sha = prereg3.load_prereg(K.PREREG_ID)
    bad = prereg3.changed_inputs(doc)
    reg = json.loads((EV / "registered.json").read_text())
    aft = json.loads((EV / "after_the_fact.json").read_text())
    rd = json.loads((REP / "readings.json").read_text())
    sy = json.loads((REP / "synthetic_plan_literal.json").read_text())
    u1 = json.loads((REP / "u1_and_sensitivity.json").read_text())
    bw = json.loads((REP / "bridge_windows.json").read_text())
    ledger.append_rows(K.WP, ledger_rows())
    dv = derived(rd, sy, u1, bw, reg, aft)
    looks = [{k: lk.get(k) for k in ("time", "purpose", "test_id", "prereg_id", "units",
                                     "data_scope")}
             for lk in prereg3.read_chained(K.DATAPREP / "preregistration" / "looks.jsonl")
             if lk.get("wp") == K.WP]
    prov = {
        "wp": K.WP, "round": "repair 2026-10-14", "prereg": K.PREREG_ID, "prereg_sha256": sha,
        "registered_inputs_changed_since_registration": bad,
        "round1_provenance_recheck": verify_round1(),
        "round1_reports": [{"path": rel_to_repo(R1 / f), "sha256": sha256_file(R1 / f)}
                           for f in ("S1b.md", "S1b.json")],
        "code": [{"path": rel_to_repo(HERE / c), "sha256": sha256_file(HERE / c)}
                 for c in CODE_R2],
        "evidence": [{"path": rel_to_repo(REP / f), "sha256": sha256_file(REP / f)}
                     for f in EVID_R2],
        "code_at_run": json.loads((REP / "code_at_run.json").read_text()),
        "code_note": "after the evidence runs, lint-only edits (unused imports, zip strict=, "
                     "unused loop variables, closure default binding) changed the hashes of "
                     "s1b_r2_core/readings/synth/u1/bridge; behaviour is unchanged (unit tests "
                     "pass) and the outputs were not regenerated; code_at_run lists the hashes "
                     "of the versions that ran",
        "looks": looks,
        "products_read": {"rings_bridged/v2": "after_the_fact_analysis",
                          "gnss_clean/full/v2": "after_the_fact_analysis",
                          "insar_069A/v2": "after_the_fact_analysis"},
        "products_written": [], "claude_api_tokens": 0, "gpu": "none",
        "note": "Validation only: no product version is created (S1b has no L1-L3 product); the "
                "repair evidence is a new directory; round-1 evidence and code are unchanged."}
    (REP / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))

    r1 = json.loads((R1 / "S1b.json").read_text())
    out = dict(r1)
    out["date"] = "2026-10-14 (repair round)"
    out["status"] = "partial"
    out["summary"] = SUMMARY
    out["deviations_from_plan"] = DEVIATIONS
    out["issues_for_downstream"] = ISSUES
    out["resources"] = {"claude_api_tokens": 0, "gpu": "none",
                        "cpu": "cpu_slot.sh, 1 slot per job (round 1: synthetic about 2 min, "
                               "registered and variants about 1 min each; repair: synthetic "
                               "about 1 min, U1 script about 2 min, bridge windows about 1 min "
                               "per run)"}
    out["acceptance"] = acceptance(reg, rd, bw, u1, aft)
    out["repair_round"] = {
        "verifier_majors": {
            "1_plan_literal_c": "accepted: plan-literal readings reported; (c) fails under "
                                "every one (3/10 .. 6/10); the registered PASS needs k >= "
                                f"{rd['families']['c_wra']['k_star_primary']:.2f}",
            "2_u1_statement": "accepted: the round-1 sentence was wrong; flag-3 GNSS days and "
                              "flag-4 InSAR windows are named and quantified; U1-strict "
                              "readings pass (report only)"},
        "readings": {k: rd["families"][k]["rules"] for k in ("c_wra", "c_wra_without_SJES",
                                                              "d_insar")},
        "k_star": {k: {"primary": rd["families"][k]["k_star_primary"],
                       "raw": rd["families"][k]["k_star_raw"]}
                   for k in ("c_wra", "c_wra_without_SJES", "d_insar", "c_neighbour")},
        "range_half_width_ppm": {k: rd["families"][k]["range_half_width_ppm"]
                                 for k in ("c_wra", "d_insar")},
        "rmse_mm": {k: rd["families"][k]["rmse_mm"] for k in ("c_wra", "c_wra_without_SJES",
                                                               "d_insar")},
        "d1": {"rates": rd["d1_rates"], "changes": rd["d1_changes"]},
        "synthetic_plan_literal_families": sy["families"],
        "derived": dv,
        "u1": {"model_set_windows": bw["first_window_vs_gnss"],
               "readings": bw["readings"],
               "sites": [{k: s.get(k) for k in ("site_en", "G_mm", "B_mm", "G_int_mm",
                                                "B_share_of_G", "B_minus_gnss_mm")}
                         for s in bw["sites"]],
               "single_epoch_split_superseded": "u1_and_sensitivity.json insar_bridges and "
                                                "bridge_vs_gnss use single epochs next to the "
                                                "gaps and are dominated by epoch noise; "
                                                "superseded by bridge_windows.json"},
        "ring_residual_sd_variants": {k: {kk: (v["verdicts"][kk]["n_contain_primary"],
                                               v["verdicts"][kk]["n_contain_raw"],
                                               v["verdicts"][kk]["pass"])
                                          for kk in ("c_wra", "d_insar")}
                                      | {"gate_pass": v["gate_pass"]}
                                      for k, v in u1["ring_sd_variants"].items()},
        "gnss_data_gaps_in_ring_gap": [g for g in u1["gnss_gaps_in_ring_gap"]
                                       if g["max_gap_days"] >= 15],
        "sjes": {"registered_rule": bw["sjes_registered_rule_mm"],
                 "endpoints": u1["sjes_endpoints"]},
        "evidence": [rel_to_repo(REP / f) for f in EVID_R2],
        "provenance": rel_to_repo(REP / "PROVENANCE.json"),
        "round1_report_copies": [rel_to_repo(R1 / f) for f in ("S1b.md", "S1b.json")],
    }
    (REPORTS / "S1b.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                 default=str))
    print(json.dumps({"bad_inputs": bad, "round1": prov["round1_provenance_recheck"],
                      "derived": dv}, indent=1, ensure_ascii=False, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
