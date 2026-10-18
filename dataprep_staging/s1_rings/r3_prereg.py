"""Pre-registrations of the S1a repair round r3 (S1ar), written with harness.prereg3.

    python -m s1_rings.r3_prereg rules     # declaration S1a.rings_bridged_rules.v3
    python -m s1_rings.r3_prereg test      # test S1a.e_synthetic_fresh.v3 (after the rules)

Both are written after the r3 diagnosis (S1a.repair.md Part 1, snapshot sha256 2fb45352...)
and before the r3 method has run on any real ring survey: before the fresh re-test, the second
look on the spent windows and the rings_bridged/v3 build. Thresholds are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

HERE = Path(__file__).resolve().parent
R3DIR = K.EVIDENCE / "repair_r3"
RULES_ID = "S1a.rings_bridged_rules.v3"
TEST_ID = "S1a.e_synthetic_fresh.v3"
PREREG_DIR = K.REPO / "results" / "dataprep" / "preregistration"


def code_hashes(names) -> dict:
    return {f"dataprep_staging/s1_rings/{n}": sha256_file(HERE / n) for n in names
            if (HERE / n).exists()}


def ring_inputs() -> list[str]:
    return [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]


RULES = {
    "basis": "S1a.rings_bridged_rules.v1 as amended by v2 (feed-check day screen, allowed-uses "
             "rule, provenance), with the changes M1-M5 below; each answers a defect of the r3 "
             "diagnosis (results/dataprep/reports/S1a.repair.md Part 1, snapshot "
             "S1a_evidence/repair_r3/snapshots/S1a.repair.part1_diagnosis.md, sha256 "
             "2fb453527f9bcaa1d43eceac508d7a8ad4c18c5db38832458ab2da253de52d43).",
    "M1_length_aware_bdeep_sd": "sd of b_deep = sqrt(max(jackknife sd, 10 ppm)^2 + (sigma_x / "
                                "L_deep)^2), L_deep the summed length of the deep low-rate "
                                "intervals; sigma_x = 1.2673 mm (r3_calibrate: mean of e^2/sd^2 "
                                "= 1 over the 468 uninjected pseudo-gap site-windows of the "
                                "registered (e) geometry, ring surveys 2014-05..2021-12; "
                                "diagnosis 1.2: the excess is a few mm whatever L_deep). Used in "
                                "the b_deep likelihood ratio (v1 formula and vote rule) and the "
                                "pool weights.",
    "M2_pile_not_a_line": "the pile-implied scale is no longer an evidence line (diagnosis 1.3: "
                          "overconfident, 25 wrong votes in 780, Xiutan z sd 3.0; at the 300 m "
                          "piles it uses the whole-interval change being corrected; upstream "
                          "deep_benchmarks/v2 failed its gate). It is still computed for check "
                          "(b) and the product card, and the piles still shape the 2022 path. "
                          "Evidence lines: offset cluster (likelihood ratio 10 in a cluster, v1) "
                          "and b_deep (M1).",
    "M3_empirical_bayes_prior": "P(A) of site i = pi_i L_i / (pi_i L_i + 1 - pi_i), L_i the "
                                "product of its lines' likelihood ratios (each capped at 1e3), "
                                "pi_i = EM over the OTHER sites of the evaluation with the "
                                "Laplace rule pi <- (sum_j P_j(pi) + 1)/(n + 2), start 0.5, "
                                "until |change| < 1e-10 (diagnosis 1.3: prior 0.5 against a "
                                "majority group; P(A) too low in the middle).",
    "M4_pools": "pooled scale of group g for site i: every other site j with a finite b_deep and "
                "P(g | j; pi_i) >= 0.95 (posterior with site i's prior), whatever its vote "
                "count or forcing status; inverse-variance mean with M1's sd, DerSimonian-Laird "
                "tau (s1lib.pool). No qualifying site: the reference value with its spread (A "
                "182.2 +- 11.4, B 1.9 +- 14.3 ppm), as in v1, and the fallback share is reported "
                "(diagnosis 1.1, 1.4: pools of assigned sites only, mean 3.89 / 1.62; the "
                "forcing rule acted through pool size).",
    "M5_common_mode": "sigma_b = sqrt(se^2 + tau^2 + sigma_cm^2), sigma_cm = sigma_c x "
                      "sum_j(w_j/L_deep,j)/sum_j(w_j), w_j = 1/sd_j^2, sigma_c = 1.3495 mm (sd "
                      "of the window-mean excess over the 42 calibration windows; diagnosis 1.4). "
                      "It widens the stated sd; it cannot remove a common bias.",
    "unchanged_from_v1": "gap_stats (window fits, J, J_top_mid, b_deep method C, total "
                         "interval); offset line; likelihood-ratio formula, cap 1e3, vote rule "
                         "(LR >= 10 and |z| <= 3); assignment = >= 2 agreeing lines, none "
                         "against, not forced (with M2: offset and b_deep both vote g); forced "
                         "list (real gap: Neiliao, Yuanchang, Xiutan, Qiaoyi; synthetic: "
                         "|dS_obs| <= 1 cm), which now acts only on the primary and the flags, "
                         "not on the pools; a_g = J_top_mid - b_g z_top; C_g = S(t_lp) - "
                         "S_raw(t_fp)/(1 + b_g) with sigma_g = sqrt((sigma_b L)^2 + 2 sd_s^2); "
                         "primary = the assigned member, else the P(A)-weighted mixture with "
                         "the mixture second moment as sd; flags, monthly series, bands, "
                         "2022 path methods and terms, the v2 feed-check day screen.",
    "corrected_check_a": "Corrected (a), a declared SECOND LOOK on the 14 design-time sites "
                         "(diagnosis 1.5: the registered statistic compares two single-survey "
                         "endpoints, whose departures from the window trends (sd_s 2.2-15.8 mm) "
                         "are unrelated to the scale, with a trend-based reference). Per member "
                         "g: C_trend,g = T_pre(t_lp) - T_post(t_fp)/(1 + b_g), T the OLS trend + "
                         "annual harmonic fits of the total separation in the pre and post "
                         "windows (gap_stats' fits), sd_g^2 = (sigma_b,g L)^2 + (k_pre "
                         "se_pre(t_lp))^2 + (k_post se_post(t_fp))^2, k = sqrt((1+phi)/(1-phi)) "
                         "with phi the lag-1 autocorrelation of the fit residuals clipped to "
                         "[0, 0.95]; primary and mixture sd as for C; sigma_mid as registered "
                         "but with the rate se multiplied by the same k; Z_i = (C_trend,primary "
                         "- C_mid)/sqrt(sd^2 + sigma_mid^2); pooled Z = sum Z_i/sqrt(14); "
                         "alternative and power as registered (shift 0.25 b_eff S_fp; power = "
                         "Phi(mu-2) + Phi(-mu-2)). Thresholds unchanged: |Z| < 2 and power >= "
                         "0.8. The registered (a) is computed with v3's values and reported.",
    "b_and_residual": "(b): pile-implied scale (v1 method) of every pile site with one, against "
                      "the pooled scale of its group (assigned, else more probable) from ALL "
                      "sites with P(g) >= 0.95 under the all-site EM prior (M1 weights); "
                      "tolerances unchanged (A +-25 %, B +-30 ppm). Residual offset unchanged.",
    "gate": "rings_bridged/v3 gate = (e) on FRESH units (S1a.e_synthetic_fresh.v3, test "
            "S1a.e3.pre2012_r3_groupcond) AND corrected (a) AND (b) AND residual offset. Because "
            "the corrected (a) is a second look, a pass is reported as 'passed, conditional "
            "(second look)'.",
    "allowed_uses": "validator and scoring_target only if the gate passes with no second-look "
                    "criterion AND every upstream product shaping flag 0-1 values passed its "
                    "gate (in v3 none does: M2) AND the user has ruled on U7 for S1's post-gap "
                    "checks; otherwise after_the_fact_analysis only.",
    "product": "rings_bridged/v3 (L2): v2's files plus an sd column for corrected depths in "
               "rings_corrected_* (sqrt(se_J_top^2 + (sigma_b (z - z_top))^2)); fit_window_end "
               "= the latest of the inputs' (deep_benchmarks/v2: 2026-01).",
    "second_look_spent_windows": "after the fresh test is scored, the r3 method also runs on the "
                                 "42 spent windows with the registered seeds of "
                                 "S1a.e_synthetic_mixed.v1 (primary and stress), logged as a "
                                 "second_look_score look before it runs: reported beside the "
                                 "registered rows, conditional, never gating.",
}

ACCEPT = {
    "e_fresh": "worst-window RMSE of the primary bridged gap compaction <= 1.5 cm (post hoc, "
               "unchanged) in S1a.e3.pre2012_r3_groupcond",
    "a": "corrected (a): |pooled Z| < 2 and power >= 0.8 (unchanged thresholds; second look); "
         "registered (a) reported",
    "b": "A within +-25 % of the pooled A scale; B within +-30 ppm of the pooled B scale",
    "residual_offset": "within +-1 cm at >= 12 of 14 sites",
    "gate": RULES["gate"],
}


def rules_doc() -> dict:
    return {
        "id": RULES_ID, "wp": "S1a",
        "title": "S1a repair round r3 (S1ar): repaired evidence, membership and pooling rules "
                 "(M1-M5), corrected check (a) as a second look, gate and allowed-uses rule "
                 "(declared before the fresh re-test and the rings_bridged/v3 build)",
        "kind": "declaration",
        "tests": not_applicable("declaration; check (e) is re-tested on fresh units in "
                                "S1a.e_synthetic_fresh.v3; (a), (b) and the residual offset are "
                                "scored on the 14 design-time sites"),
        "zones": not_applicable("no zones"),
        "priors": {"membership_prior": "empirical Bayes (M3)",
                   "reference_groups_ppm": {"A": [182.2, 11.4], "B": [1.9, 14.3]}},
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("leave-one-site-out pooling and EM prior"),
        "fresh_units": not_applicable("none in this declaration"),
        "thresholds": ACCEPT,
        "gate_scores": not_applicable("conjunctive gate: " + RULES["gate"]),
        "primary_model": RULES,
        "inputs": ring_inputs() + [
            rel_to_repo(K.DEEP_BENCH_V2 / "daily.parquet"),
            rel_to_repo(K.DEEP_BENCH_V2 / "MANIFEST.json"),
            rel_to_repo(K.PRINTED_V2 / "printed_monthly.parquet"),
            rel_to_repo(K.PRINTED_V2 / "MANIFEST.json"),
            rel_to_repo(R3DIR / "calibration.json"),
            rel_to_repo(R3DIR / "diag" / "truth_bdeep.csv"),
            rel_to_repo(R3DIR / "diag" / "summary.json"),
            rel_to_repo(R3DIR / "snapshots" / "S1a.repair.part1_diagnosis.md")],
        "interpretation": (
            "rings_bridged stays a post-origin DATA product (every correction parameter rests on "
            "2023-25 surveys). v3 changes how the group and the scale are inferred, not the "
            "instrument model. The common deep excess of a strong dry season is not repairable "
            "from ring data (diagnosis 1.2/1.4): windows that contain one can still fail. "
            "Nothing here changes a threshold; the corrected (a) is labelled a second look "
            "wherever it is quoted."),
        "anti_leakage": "U3/U5: no pre-origin use (fit_window_end 2026-01). U7: the gate, the "
                        "member choice and the injected p_A rest on 2023-25 ring observations; "
                        "flagged for a user ruling (unchanged), and the allowed-uses rule keeps "
                        "scoring uses closed until it is ruled.",
        "design_time_exposure": [
            "Everything S1a rounds 1-2 saw (seeded looks), the r3 diagnosis of the spent (e) "
            "rows and of the uninjected 2014-05..2021-12 surveys (looks 2026-10-17T20:57:24Z), "
            "and the calibration constants computed from them.",
            "The 14 real sites (design-time units for (a), (b) and the residual offset): v2's "
            "published check (a) rows were re-read at about 21:07Z; the r2 report already "
            "contained an after-the-fact (a) without the endpoint term (Z 1.04, power 0.999).",
            "The pre-2012 record: only survey dates and ring read masks were read in r3 "
            "(availability looks 20:57:24Z and 21:03:14Z). The pilot's gap2012.py "
            "(subsidence_gap_sources, 2026-10-11) read surveys 2010-01..2012-03 for its 2012-14 "
            "hiatus scale test; that is not a pseudo-gap test and its output was not read in r3.",
        ],
        "code_at_registration": code_hashes(["s1const.py", "s1lib.py", "r3_lib.py",
                                             "r3_build.py", "r3_calibrate.py", "r3_looks.py",
                                             "r3_prereg.py", "build.py", "build_r2.py"]),
    }


NONE_MEMBERS = ("simulation draws on pre-origin ring surveys (no registered well, leveling "
                "benchmark or model cell)")


def _unit(name: str, desc: str) -> dict:
    return {name: {"none": NONE_MEMBERS, "description": desc}}


def test_doc() -> dict:
    u_pre_p = "S1a:synth.R3.pre2012.groupcond.v3"
    u_pre_h = "S1a:synth.R3.pre2012.harness.v3"
    u_l19_p = "S1a:synth.R3.late2019.groupcond.v3"
    u_l19_h = "S1a:synth.R3.late2019.harness.v3"
    stat = ("worst-window RMSE (over sites x realizations) of C_primary - C_true; also the "
            "median window RMSE, windows above 1.5 cm, oracle (true-group member) worst window, "
            "misclassification and unassigned rates, reference-fallback shares, mean pool "
            "sizes, by-true-group RMSE and the coverage of the nominal 90 % interval "
            "(harness.coverage3, clusters = sites; uncalibrated below 20 clusters)")
    pre_scope, l19_scope = "rings:2003-01..2012-02", "rings+piles:2016-07..2021-12"
    tests = [
        {"test_id": "S1a.e3.pre2012_r3_groupcond", "data_scope": pre_scope,
         "fresh_units": [u_pre_p], "external_units": _unit(
             u_pre_p, "mixed A/B instrument changes injected in fresh pre-2012 windows, offsets "
                      "from the own group's cluster (primary injection)"),
         "method": "r3 (S1a.rings_bridged_rules.v3)", "statistic": stat,
         "threshold": "worst-window RMSE <= 1.5 cm (post hoc, unchanged)", "gating": True},
        {"test_id": "S1a.e3.pre2012_r3_harness", "data_scope": pre_scope,
         "fresh_units": [u_pre_h], "external_units": _unit(
             u_pre_h, "the same windows and seeds with harness offsets independent of the group "
                      "(stress injection)"),
         "method": "r3", "statistic": stat, "threshold": "reported only (stress)",
         "gating": False},
        {"test_id": "S1a.e3.pre2012_registered_groupcond", "data_scope": pre_scope,
         "fresh_units": [u_pre_p], "external_units": _unit(
             u_pre_p, "mixed A/B instrument changes injected in fresh pre-2012 windows, offsets "
                      "from the own group's cluster (primary injection)"),
         "method": "registered v1/v2 method (s1lib.evaluate) on the SAME draws: control",
         "statistic": stat, "threshold": "reported only (control)", "gating": False},
        {"test_id": "S1a.e3.pre2012_registered_harness", "data_scope": pre_scope,
         "fresh_units": [u_pre_h], "external_units": _unit(
             u_pre_h, "the same windows and seeds with harness offsets independent of the group "
                      "(stress injection)"),
         "method": "registered method, control", "statistic": stat,
         "threshold": "reported only (control)", "gating": False},
        {"test_id": "S1a.e3.late2019_r3_groupcond", "data_scope": l19_scope,
         "fresh_units": [u_l19_p], "external_units": _unit(
             u_l19_p, "13-month windows starting 2019-07..2019-12 (start months never used; the "
                      "surveys are design-exposed), primary injection"),
         "method": "r3", "statistic": stat,
         "threshold": "reported only (fresh start months, design-exposed surveys)",
         "gating": False},
        {"test_id": "S1a.e3.late2019_r3_harness", "data_scope": l19_scope,
         "fresh_units": [u_l19_h], "external_units": _unit(
             u_l19_h, "the late-2019 windows with harness offsets (stress injection)"),
         "method": "r3", "statistic": stat, "threshold": "reported only", "gating": False},
        {"test_id": "S1a.e3.late2019_registered_groupcond", "data_scope": l19_scope,
         "fresh_units": [u_l19_p], "external_units": _unit(
             u_l19_p, "13-month windows starting 2019-07..2019-12 (start months never used; the "
                      "surveys are design-exposed), primary injection"),
         "method": "registered method, control", "statistic": stat,
         "threshold": "reported only (control)", "gating": False},
        {"test_id": "S1a.e3.late2019_registered_harness", "data_scope": l19_scope,
         "fresh_units": [u_l19_h], "external_units": _unit(
             u_l19_h, "the late-2019 windows with harness offsets (stress injection)"),
         "method": "registered method, control", "statistic": stat,
         "threshold": "reported only (control)", "gating": False},
    ]
    design = {
        "families": {
            "pre2012": "every 13-month window of harness.gaps_guarded.ring_windows('2003-01', "
                       "'2011-12', 13) whose post window starts by 2012-02-29; pre window <= 36 "
                       "months before the mask (from 2003-01-01), post window <= 28 months after "
                       "it (to 2012-02-29, the last survey month before the 2012-03..2014-04 "
                       "hiatus); a site enters with >= 12 pre and >= 12 post surveys and "
                       "gap_stats valid (>= 3 rings read in >= 80 % of both windows); a window "
                       "is scored with >= 8 sites. From availability only (survey dates and "
                       "read masks): 16 windows expected, starts 2008-11..2010-02, 8-10 sites "
                       "(Yuanchang, Guangfu, Neiliao, Jiaxing, Tuku, Kecuo, Xinsheng, Hunan, "
                       "Xizhou, Xiutan). Entry relaxed from v1's 20/18 surveys and 10 sites "
                       "because v1's rule scores no pre-2012 window; 12 surveys keep one full "
                       "annual cycle per window fit; 8 sites keep pooled groups.",
            "late2019": "the registered geometry (pre <= 36 months from 2014-05-01, post <= 28 "
                        "months to 2021-12-31) for the windows starting 2019-07..2019-12 of "
                        "ring_windows('2015-01', '2021-12', 13), >= 20 pre and >= 12 post "
                        "surveys, >= 10 sites."},
        "realizations": "10 per window; seeds harness.gaps.rng_for('R', r) with r = 30000 + "
                        "100 w + k (pre2012) and 40000 + 100 w + k (late2019), w the window "
                        "index in the family's ring_windows list, k = 0..9 (disjoint from the "
                        "registered r = 100 w + k)",
        "injections": "primary = synthetic.inject_groupcond (b ~ N(182.2, 11.4) ppm for A, "
                      "N(1.9, 14.3) for B; offset from the own group's cluster with probability "
                      "11/14, else uniform -10..+20 cm); stress = harness.gaps."
                      "ring_instrument_injection; p_A = 5/7 (S1a_evidence/assignment.json, as "
                      "in v1); every survey after the mask z -> a + (1 + b) z",
        "pipeline": "r3: r3_lib.evaluate_r3 (rules S1a.rings_bridged_rules.v3; sigma_x, "
                    "sigma_c from repair_r3/calibration.json; no pile line); registered: "
                    "s1lib.evaluate with v1's settings (pile lines where pile rows exist: none "
                    "before 2016); forced = |dS_obs| <= 1 cm in both",
        "truth": "C_true = S_true(t_lp) - S_true(t_fp) of the uninjected surveys on the same "
                 "rings and dates",
        "runner": "dataprep_staging/s1_rings/r3_synthetic.py fresh",
    }
    return {
        "id": TEST_ID, "wp": "S1a",
        "title": "S1a check (e) re-test on FRESH synthetic pseudo-gaps (pre-2012 windows never "
                 "used), method r3, with the registered method as control",
        "kind": "test", "tests": tests,
        "zones": not_applicable("no zones"),
        "priors": {"membership_prior": "empirical Bayes (r3) / 0.5 (registered control)"},
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("leave-one-site-out pooling and EM prior inside each "
                                "realization"),
        "fresh_units": [u_pre_p, u_pre_h, u_l19_p, u_l19_h],
        "thresholds": {"S1a.e3.pre2012_r3_groupcond": "worst-window RMSE <= 0.015 m (post hoc, "
                                                      "unchanged)",
                       "others": "none (reported)"},
        "gate_scores": not_applicable("single pass/fail statistic: "
                                      "S1a.e3.pre2012_r3_groupcond"),
        "primary_model": design,
        "inputs": ring_inputs() + [
            rel_to_repo(K.DEEP_BENCH_V2 / "daily.parquet"),
            rel_to_repo(K.DEEP_BENCH_V2 / "MANIFEST.json"),
            rel_to_repo(K.EVIDENCE / "assignment.json"),
            rel_to_repo(R3DIR / "calibration.json"),
            f"results/dataprep/preregistration/{RULES_ID}.json"],
        "interpretation": (
            "Pass if the r3 worst-window RMSE on the pre-2012 primary family is <= 1.5 cm. The "
            "pre-2012 windows differ from the real gap: <= 10 sites (Qiaoyi, Honglun, Zhengmin "
            "and Beichen have no pre-2012 record), shorter fit windows, no piles, higher "
            "compaction rates. The injection uses the same reference distributions as the "
            "likelihood lines (well-specified classifier: optimistic) and the reference "
            "fallback equals the injection mean (fallback shares are reported). The registered "
            "method on the same draws shows whether the fresh windows are easier than the "
            "spent ones. A failure keeps rings_bridged excluded; a pass alone opens no scoring "
            "use (see the allowed-uses rule of S1a.rings_bridged_rules.v3)."),
        "anti_leakage": "pre-origin data only (ring surveys <= 2012-02 for the gating family; "
                        "<= 2021-12 and pre-origin pile rows for late2019); injection constants "
                        "are design constants from the pilot's real-gap analysis",
        "code_at_registration": code_hashes(["s1const.py", "s1lib.py", "synthetic.py",
                                             "r3_lib.py", "r3_synthetic.py", "r3_calibrate.py",
                                             "r3_looks.py", "r3_prereg.py"]),
    }


def main(argv=None) -> None:
    which = (argv or sys.argv[1:] or [""])[0]
    if which == "rules":
        doc = rules_doc()
    elif which == "test":
        if not (PREREG_DIR / f"{RULES_ID}.json").exists():
            raise SystemExit("write the rules declaration first")
        doc = test_doc()
    else:
        raise SystemExit("usage: r3_prereg {rules|test}")
    errs = prereg3.validate_prereg(doc)
    if errs:
        raise SystemExit(f"invalid: {errs}")
    path, sha = prereg3.write_prereg(doc)
    print(path, sha)


if __name__ == "__main__":
    main()
