"""Pre-registrations of WP S1a (harness.prereg3).

    python -m s1_rings.prereg_s1a rules       # declaration S1a.rings_bridged_rules.v1
    python -m s1_rings.prereg_s1a synthetic   # test S1a.e_synthetic_mixed.v1

The rules declaration fixes every choice of the product before the rebuilt evidence is
computed on the real data. Its real-gap acceptance checks (a), (b) and the residual offset are
scored on the 14 sites the pilot already analysed (design-time units, seeded look), so they
carry no fresh unit; only the mixed A/B synthetic test (e) is a registered test on fresh units.
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


def data_inputs(with_gnss: bool = False) -> list[str]:
    xs = [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]
    xs += [rel_to_repo(K.DEEP_BENCH_V2 / "daily.parquet"),
           rel_to_repo(K.DEEP_BENCH_V2 / "MANIFEST.json"),
           rel_to_repo(K.PRINTED_V2 / "printed_monthly.parquet"),
           rel_to_repo(K.PRINTED_V2 / "MANIFEST.json")]
    if with_gnss:
        xs += [rel_to_repo(K.GNSS_FULL_V1 / "wra" / "monthly.parquet"),
               rel_to_repo(K.GNSS_FULL_V1 / "MANIFEST.json")]
    return xs


def code_hashes(names) -> dict:
    return {f"dataprep_staging/s1_rings/{n}": sha256_file(HERE / n) for n in names
            if (HERE / n).exists()}


RULES_TEXT = {
    "model": "z_post = a_site + (1 + b_group) z_true; z' = (z - a_site)/(1 + b_group) for every "
             "survey from 2023-01-01 on (Beichen: from 2024-07). Rings: read in >= 80 % of the "
             "surveys of both windows (2019-01..2021-12 and 2023-02..2025-05; Beichen "
             "2024-07..2025-05), ordered by their 2019-21 mean depth.",
    "window_fits": "per ring OLS trend + annual harmonic in each window (t_ref = mid-gap); "
                   "J = deseasonalised post fit at the first post survey minus pre fit at the "
                   "last pre survey; J_top_mid = the top ring's post minus pre fit at mid-gap "
                   "(se from the fit covariances).",
    "b_deep": "pilot method C: consecutive intervals with mid-depth below the median ring depth "
              "and shortening rate < 1 mm/yr in both windows; b = sum(dS_k + C_k)/sum(L_k), "
              "C_k = mean of the two windows' rates x gap; jackknife sd; NaN with < 3 intervals.",
    "evidence_lines": {
        "offset": "J_top_mid in [-16.8, -15.0] cm votes A (likelihood ratio 10), in "
                  "[+49.5, +54.5] cm votes B (1/10), otherwise no vote (LR 1). The ranges are "
                  "the plan's -15.5..-16.3 and +50..+54 cm widened by 0.5 cm.",
        "b_deep": "Gaussian likelihood ratio A:B of the site's b_deep (sd floored at 10 ppm) "
                  "against reference groups A N(182.2, 11.4^2 + sd^2) ppm and B "
                  "N(1.9, 14.3^2 + sd^2) ppm, capped at 1e3; votes when LR >= 10 (<= 1/10) "
                  "and the value lies within 3 sd of the favoured group.",
        "pile": "pile-implied scale b = (ratio x pile_gap - ring_gap_raw)/L at Tuku (TKJS(1) "
                "300 m), Xizhou (SJES 300 m), Guangfu and Zhengmin (NTUH(1) 300 m) over the top "
                "ring -> deepest both-sides ring, and Xiutan (STES 70 m) over the top ring -> "
                "ring nearest 70 m; ratio = mean of the ring/pile rate ratios 2017-01 (or the "
                "pile's first day)..2021-12 and 2023-02..2025-05 on survey-matched vendor daily "
                "values (deep_benchmarks/v2, flags 0-2, +-3 days); sd = [half the pre/post "
                "ratio difference x pile_gap] (+) [RMS over the pre window of ring change - "
                "ratio_pre x pile change across 14-month spans], over L. Same likelihood ratio "
                "and vote rule as b_deep. Needs >= 6 matched surveys and >= 80 % coverage in "
                "each window.",
        "gnss": "not a line of the product: gnss_clean (S3) failed its gate, and its A2 failure "
                "is GNSS-ring disagreement, which is what a GNSS-implied scale rests on. "
                "GNSS-implied scales are computed from gnss_clean/full/v1 (after_the_fact_"
                "analysis) and reported as a sensitivity of the assignment only.",
    },
    "assignment": "assigned to g when >= 2 lines vote g and none votes the other group, unless "
                  "the site is on the plan's forced list (Neiliao, Yuanchang, Xiutan, Qiaoyi), "
                  "which is carried as both members in any case. Unassigned sites carry both "
                  "members.",
    "membership_probability": "P(A) = 0.5 L / (0.5 L + 0.5), L = product of the lines' "
                              "likelihood ratios (each capped at 1e3; offset 10).",
    "pooling": "group scale = inverse-variance mean (sd floor 10 ppm) of the b_deep of the "
               "sites ASSIGNED to the group, the corrected site left out; sigma_b = sqrt(se^2 + "
               "tau^2) with the DerSimonian-Laird tau of those sites. No site left: the "
               "reference value with its spread (A 182.2 +- 11.4, B 1.9 +- 14.3 ppm). Every "
               "site gets both members' corrections (b_A, b_B) for the membership term.",
    "offset_a": "a_g = J_top_mid - b_g z_top.",
    "gap_compaction": "C_g = S(t_lp) - S_raw(t_fp)/(1 + b_g) of the total interval (top ring -> "
                      "deepest both-sides ring); sigma_g = sqrt((sigma_b L)^2 + 2 sd_s^2), sd_s "
                      "the residual sd of the 2019-21 trend + annual fit of the total "
                      "separation.",
    "primary": "assigned sites: the assigned member; unassigned: the P-weighted mixture; "
               "sd^2 = sum_g p_g [sigma_g^2 + (C_g - C_primary)^2] (plan: membership "
               "uncertainty enters sigma as the mixture variance).",
    "flags": "pre-gap months 0 (observed); post-gap months of the assigned member 1 (observed, "
             "corrected with an independently assigned group); post-gap months of unassigned "
             "or forced sites (primary and members) and of a non-assigned member 3; 2022 path "
             "3 (pile-shaped, or Xiutan's pile-shaped shallow band + climatology) or 4 "
             "(linear + seasonal climatology); Beichen's 2022-01..2024-06 path 6 (30 months, "
             "beyond the 13-month tested envelope); months without a survey outside the "
             "hiatus 8.",
    "monthly_series": "monthly median ring positions; interval shortening re-zeroed to its "
                      "first valid month; total = top ring -> deepest both-sides ring; bands "
                      "as hydrophysics.twin.mechanism.band_compaction (rings nearest 0, 100, "
                      "200 m below the top ring among the both-sides rings, the deepest closes "
                      "the last band).",
    "path_2022": "months between the last pre month and the first post month at virtual "
                 "survey dates (the site's median 2019-21 day of month). Pile-shaped (unit "
                 "slope, chord + the pile's deviation from its chord, vendor daily values "
                 "with gaps <= 31 days interpolated) for the total at Tuku, Xizhou, Guangfu; "
                 "Xiutan: shallow band (top -> ring nearest 70 m) pile-shaped + deep remainder "
                 "climatology; elsewhere and for every band: linear + seasonal climatology "
                 "(mean anomaly about a 13-month centred running mean, 2014-05..2021-12). A "
                 "path sensor whose 2022 vendor monthly-mean increments differ from the "
                 "printed 007001 values (deep_benchmarks_printed/v2, flag 0) by > 2 mm in any "
                 "month falls back to climatology. Never plain linear, continuous-rate or "
                 "offset-only.",
    "path_sd": "sqrt((sigma_b L_interval tau)^2 + path_term^2) per member (tau = fraction of "
               "the gap elapsed); path_term = the pooled RMSE of the method in the family-Y "
               "path test (masked calendar years 2017-2020, endpoints the December before "
               "(>= Nov 1) and the January after (<= Mar 15), >= 8 masked survey months, "
               "climatology without the masked span, on the product's intervals, pre-origin "
               "values only); Beichen x sqrt(n_months/13). The pilot's 1.8 / 5.3 mm are not "
               "used.",
    "sd_observed": "flag 0: sd_s (residual sd of the 2019-21 fit; an upper bound of the "
                   "measurement error, uncalibrated); flag 1: sqrt(sd_s^2 + (sigma_b L)^2).",
}

ACCEPTANCE = {
    "e_worst_window": "worst-window RMSE of the primary bridged gap compaction <= 1.5 cm in the "
                      "mixed A/B synthetic rerun (test S1a.e_synthetic_mixed.v1, primary "
                      "variant). Threshold set by the plan after seeing the pilot's 1.35 cm: "
                      "POST HOC, labelled so wherever quoted.",
    "a_pooled_z": "Z_i = (C_primary - C_mid)/sqrt(sigma_primary^2 + sigma_mid^2), C_mid = mean "
                  "of the 2019-21 and 2023-25 total shortening rates x gap, sigma_mid^2 = "
                  "((C_hi - C_lo)/sqrt(12))^2 + gap^2 (se_pre^2 + se_post^2)/4; pooled Z = "
                  "sum Z_i / sqrt(14); pass |Z| < 2 AND power >= 0.8 for a 25 % scale error "
                  "(shift 0.25 b_eff S_raw(t_fp) per site; power = Phi(mu - 2) + Phi(-mu - 2), "
                  "mu = sum(shift_i/s_i)/sqrt(14)).",
    "b_pile_scale": "every pile site with a pile-implied scale, tested against the pooled scale "
                    "of its group (all assigned sites; the group of an unassigned site = its "
                    "more probable group): A within +-25 % of the pooled A scale; B within "
                    "+-30 ppm of the pooled B scale.",
    "residual_offset": "after correction (primary member: the assigned one, else the more "
                       "probable), the intercept of the last-pre to first-post survey change "
                       "regressed on depth within +-1 cm at >= 12 of 14 sites.",
    "misassignment_sensitivity": "reported, not gating: every site re-corrected with the group "
                                 "of the nearest offset-cluster centre (A -15.9 cm, B +52 cm) "
                                 "and the change in bridged gap compaction listed, Yuanchang in "
                                 "particular; plus the offset likelihood ratio at 3 and 30, and "
                                 "GNSS as a fourth line (after the fact).",
    "gate": "rings_bridged gate = e_worst_window AND a_pooled_z AND b_pile_scale AND "
            "residual_offset.",
}


def rules_doc() -> dict:
    return {
        "id": "S1a.rings_bridged_rules.v1",
        "wp": "S1a",
        "title": "S1a ring instrument-change correction and 2022 bridge: product rules and "
                 "acceptance (declared before the rebuilt evidence is computed)",
        "kind": "declaration",
        "tests": not_applicable("no fresh test in this declaration: the real-gap checks are "
                                "scored on the 14 sites the pilot analysed (design-time units, "
                                "seeded look 2026-10-11T17:02:48Z); the synthetic test (e) is "
                                "registered separately as S1a.e_synthetic_mixed.v1"),
        "zones": not_applicable("no zones"),
        "priors": {"membership_prior_A": 0.5,
                   "reference_groups_ppm": {"A": [182.2, 11.4], "B": [1.9, 14.3]}},
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("leave-one-site-out pooling only"),
        "fresh_units": not_applicable("none (see tests)"),
        "thresholds": ACCEPTANCE,
        "gate_scores": not_applicable("conjunctive gate: " + ACCEPTANCE["gate"]),
        "primary_model": RULES_TEXT,
        "inputs": data_inputs(with_gnss=True),
        "interpretation": (
            "rings_bridged is a post-origin data product by construction: every correction "
            "parameter (a, b, group) rests on 2023-25 surveys, so its fit_window_end is 2025-05 "
            "and it may serve only post-origin uses (validator, scoring_target for flags 0-1, "
            "after_the_fact_analysis). Its gate decides whether the instrument correction is "
            "valid; it selects no model, so U7 (no 2023+ observation in a model gate or member "
            "choice) is not engaged, and the product must never be used for selection, design, "
            "priors, calibration or datum fits. The 2022 interior months (flags 3/4/6) are "
            "never scoring targets (U1). Real-gap checks are design-time-exposed (the pilot saw "
            "the same sites with sign-based groups)."),
        "anti_leakage": "U1: flags 2-6 never scoring targets; U3/U5: no pre-origin use allowed "
                        "(fit_window_end 2025-05); U7: the synthetic test uses ring and pile "
                        "data 2014-05..2021-12 only; its injection constants come from the pilot "
                        "(post-origin) and are design constants of a validation test.",
        "design_time_exposure": [
            "pilot subsidence_gap_sources (seeded look): per-site b_deep, offsets, sign-based "
            "groups, pooled 182.2/1.9 ppm, rate brackets (10/14), pile-implied scales "
            "(153-203 ppm group A; -10 ppm Zhengmin), residual offsets (13 within 0.9 cm, "
            "Yuanchang +1.6 cm), GNSS- and InSAR-implied values, group-A-only synthetic windows "
            "(worst 1.35 cm), interior-path RMSE (1.8 / 5.3 / 10.6 mm).",
            "plan WP S1 text: the b_deep table and the four forced sites.",
            "S1a session (logged look 2026-10-13): pilot tables re-read; printed pile months "
            "2020-10..2023-06 and vendor daily values near 2021-12/2022-12 read.",
        ],
        "code_at_registration": code_hashes(["s1const.py", "s1lib.py", "seed_looks.py",
                                             "prereg_s1a.py"]),
    }


def synthetic_doc() -> dict:
    members = {"none": "simulation draws on pre-origin ring surveys (no registered well, "
                       "benchmark or cell)"}
    ext_p = {"S1a:synth.R.mixed_groupcond.v1": {
        **members, "description": "mixed A/B instrument changes, offsets drawn from the own "
                                  "group's cluster (primary)"}}
    ext_h = {"S1a:synth.R.mixed_harness.v1": {
        **members, "description": "mixed A/B instrument changes, harness.gaps."
                                  "ring_instrument_injection offsets (independent of group; "
                                  "stress)"}}
    common = {
        "windows": "harness.gaps_guarded.ring_windows('2015-01', '2021-12', 13): every "
                   "13-month window; a site enters a window when it has >= 20 surveys in the "
                   "36 months before it (from 2014-05) and >= 18 in the 28 months after it "
                   "(to 2021-12); a window is scored when >= 10 sites enter.",
        "realizations": "10 per window, r = 100 w + k (w window index from 0, k = 0..9); RNG "
                        "harness.gaps.rng_for('R', r) (seed 1000*ord('R') + r).",
        "pipeline": "the product pipeline (s1lib.evaluate) on the injected surveys: rings read on "
                    "both sides, window fits, b_deep, offset, pile-implied scale (pile rate "
                    "windows = the synthetic windows clipped to the pile record), votes, "
                    "assignment, LOO pooling within evidence-assigned groups, members, "
                    "membership probability, primary; forced (carried as both members) = "
                    "|dS_obs| <= 1.0 cm (the plan's near-threshold criterion); GNSS not used.",
        "truth": "C_true = S_true(t_lp) - S_true(t_fp) of the uninjected surveys on the same "
                 "rings and dates.",
        "p_A": "n_A / (n_A + n_B) of the real-gap evidence assignment (assigned, not forced), "
               "computed by s1lib.evaluate on the real data before this test runs; 10/14 if "
               "none is assigned.",
    }
    return {
        "id": "S1a.e_synthetic_mixed.v1",
        "wp": "S1a",
        "title": "S1a check (e): mixed A/B synthetic pseudo-gaps, every 13-month window "
                 "2015-2021, full pipeline including classification",
        "kind": "test",
        "tests": [
            {"test_id": "S1a.e.mixed_groupcond", "data_scope": "rings+piles:2014-05..2021-12",
             "fresh_units": sorted(ext_p), "external_units": ext_p,
             "statistic": "worst-window RMSE (over sites x realizations) of C_primary - C_true; "
                          "also misclassification rate, unassigned rate, RMSE of misassigned "
                          "site-realizations, oracle-group worst-window RMSE, coverage of the "
                          "nominal 90 % interval (|err| <= 1.645 sigma_primary; "
                          "harness.coverage3, clusters = sites, uncalibrated with 14 clusters)",
             "threshold": "worst-window RMSE <= 1.5 cm (post hoc, plan)",
             "gating": True},
            {"test_id": "S1a.e.mixed_harness", "data_scope": "rings+piles:2014-05..2021-12",
             "fresh_units": sorted(ext_h), "external_units": ext_h,
             "statistic": "as S1a.e.mixed_groupcond with the harness generator's offsets",
             "threshold": "reported only (stress)", "gating": False},
        ],
        "zones": not_applicable("no zones"),
        "priors": {"membership_prior_A": 0.5},
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("leave-one-site-out pooling inside each realization"),
        "fresh_units": sorted(ext_p) + sorted(ext_h),
        "thresholds": {"S1a.e.mixed_groupcond": "worst-window RMSE <= 0.015 m (post hoc)",
                       "S1a.e.mixed_harness": "none (reported)"},
        "gate_scores": not_applicable("single pass/fail statistic (S1a.e.mixed_groupcond)"),
        "primary_model": {
            **common,
            "injection_primary": "group A with probability p_A, else B; b ~ N(182.2, 11.4) ppm "
                                 "(A) or N(1.9, 14.3) ppm (B) (harness defaults); offset: with "
                                 "probability 11/14 uniform in the own group's cluster "
                                 "(A -16.3..-15.5 cm, B +50..+54 cm, harness ranges), else "
                                 "uniform -10..+20 cm (harness offset_other); every survey after "
                                 "the window: a + (1 + b) z.",
            "injection_stress": "harness.gaps.ring_instrument_injection(sites, r, p_A=p_A) "
                                "unchanged (offset cluster independent of the group).",
            "rules": "S1a.rings_bridged_rules.v1 (declaration) for every pipeline choice",
        },
        "inputs": data_inputs(),
        "interpretation": "pass if the primary worst-window RMSE <= 1.5 cm (a post hoc "
                          "threshold); the stress variant shows what the offset line costs when "
                          "offsets carry no group information. The synthetic truth uses the "
                          "same reference group distributions as the evidence likelihoods, so "
                          "the classification is evaluated under a well-specified model "
                          "(optimistic).",
        "code_at_registration": code_hashes(["s1const.py", "s1lib.py", "synthetic.py"]),
    }


def main(argv=None) -> None:
    which = (argv or sys.argv[1:] or ["rules"])[0]
    doc = rules_doc() if which == "rules" else synthetic_doc()
    path, sha = prereg3.write_prereg(doc)
    print(path, sha)


if __name__ == "__main__":
    main()
