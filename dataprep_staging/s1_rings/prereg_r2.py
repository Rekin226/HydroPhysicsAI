"""Declaration S1a.rings_bridged_rules.v2 (WP S1a repair round; harness.prereg3).

    python -m s1_rings.prereg_r2          (from dataprep_staging/)

Written after the verifier's review of rings_bridged/v1 and before rings_bridged/v2 is built.
It amends S1a.rings_bridged_rules.v1 in four points (feed-check day screen, allowed-uses rule,
code provenance, product card) and changes no evidence rule, assignment, pooling, member,
mixture, flag, sd, acceptance check, threshold or registered test. Every amendment is decided
AFTER v1's results were seen and is labelled so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings.prereg_s1a import code_hashes, data_inputs  # noqa: E402

ID = "S1a.rings_bridged_rules.v2"
V1 = "S1a.rings_bridged_rules.v1"
V1_SHA = "263973f04af1884176ed339b34eeaedb679ece1293de8e0608b8a765a36a393e"

AMENDMENTS = {
    "1_feed_check_day_screen": (
        "path_2022 feed check: a vendor monthly-mean increment of a path sensor is compared "
        "with the printed 007001 value (deep_benchmarks_printed/v2, flag 0) only when BOTH of "
        "its months have >= 20 vendor days with flag 0-1 in deep_benchmarks/v2 (that product's "
        "convention for a comparable monthly mean: 'no feed with >= 20 full days in both "
        "months'). The 2 mm tolerance and the 2022 month range are unchanged. POST-RESULT: "
        "decided after v1's check tripped on TKJS(1)'s vendor outage 2022-05-04..23 (a 9-day "
        "May mean, 4.41 mm in 2022-06) and sent Tuku's 2022 path to climatology. Expected "
        "effect: Tuku's total 2022 path pile-shaped (flag 3) as the plan prescribes. No "
        "acceptance check uses the 2022 path."),
    "2_allowed_uses_rule": (
        "validator and scoring_target are declared only when the product gate passes AND "
        "every upstream product whose values shape flag 0-1 values passed its own gate; "
        "otherwise the product declares after_the_fact_analysis only and forbids every other "
        "use. deep_benchmarks/v2 (gate failed) shapes the group assignment through the pile "
        "line, so rings_bridged/v2 declares after_the_fact_analysis only. Closes the route by "
        "which a later version passing its own gate would carry a failed upstream into "
        "scoring targets (verifier, minor issue)."),
    "3_code_provenance": (
        "the manifest hashes every module the outputs depend on: v1's list plus "
        "harness/coverage3.py (coverage of the hashed synthetic result), common/flags.py "
        "(validate_long_table), common/uses.py and common/paths.py (guard), build_r2.py, "
        "r2_diagnose.py and seed_looks.py."),
    "4_product_card_and_validation": (
        "README: 13 synthetic clusters (v1 said 14); the (e) failure diagnosed from the "
        "registered rows (r2_diagnose: scale and membership components); the pile line "
        "described as using the raw ring change across the gap. validation.json adds, "
        "labelled after the fact and outside the gate: residual offsets under both members, "
        "how far check (b) is implied by the pile vote, the assignment with the plan's "
        "unwidened offset ranges, and the (e)/(a) decompositions."),
}

UNCHANGED = (
    "Everything else in S1a.rings_bridged_rules.v1 (sha256 " + V1_SHA + ") stands: model, "
    "windows, ring selection, b_deep, evidence lines and their likelihood ratios, the 0.5 cm "
    "widened offset clusters, the vote and assignment rule, the forced list, the membership "
    "prior 0.5, LOO pooling, members, mixture, flags, sd, path methods other than the feed "
    "check, path terms (family-Y evidence of v1), and the acceptance checks (a), (b), (e), "
    "residual offset with their thresholds and the conjunctive gate. Check (e) is the "
    "registered result of S1a.e_synthetic_mixed.v1 (2026-10-13T07:52Z); it is NOT re-run and "
    "no new seeds are drawn, because a method changed after seeing (e)'s failing windows "
    "could not be tested freshly on the same 2016-2019 windows.")


def doc() -> dict:
    return {
        "id": ID,
        "wp": "S1a",
        "title": "S1a repair round: amendments of the rings_bridged rules for v2 (feed-check "
                 "day screen, allowed-uses rule, provenance, product card); declared before v2 "
                 "is built",
        "kind": "declaration",
        "tests": not_applicable("no fresh test: the gate is v1's, with the registered (e) "
                                "result and the real-gap checks (a), (b), residual offset "
                                "recomputed under unchanged rules on design-time units"),
        "zones": not_applicable("no zones"),
        "priors": {"membership_prior_A": K.PRIOR_A,
                   "reference_groups_ppm": {"A": [182.2, 11.4], "B": [1.9, 14.3]}},
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("leave-one-site-out pooling only (unchanged)"),
        "fresh_units": not_applicable("none"),
        "thresholds": {"unchanged": "all of " + V1 + " (acceptance and gate)"},
        "gate_scores": not_applicable("conjunctive gate of " + V1 + ", unchanged"),
        "primary_model": {"base": V1, "base_sha256": V1_SHA, "amendments": AMENDMENTS,
                          "unchanged": UNCHANGED},
        "inputs": data_inputs(with_gnss=False) + [
            "results/dataprep/preregistration/" + V1 + ".json",
            "results/dataprep/reports/S1a_evidence/synthetic/result.json",
            "results/dataprep/reports/S1a_evidence/repair/e_decomposition.json",
            "results/dataprep/reports/S1a_evidence/repair/a_power_decomposition.json"],
        "interpretation": (
            "rings_bridged/v2 is v1 with Tuku's 2022 path restored to pile-shaped, the "
            "allowed uses narrowed to after_the_fact_analysis while deep_benchmarks/v2's gate "
            "is failed, provenance completed and the product card corrected. Its gate is "
            "expected to stay failed ((e) 2.79 cm, (a) power 0.721). U7: the product's gate, "
            "group (member) choice and the injected p_A of (e) rest on 2023-25 ring and pile "
            "observations, which is inherent to S1's design (the instrument change is visible "
            "only after the gap); U7's text covers 'a gate ... a member choice', so this is "
            "flagged for a plan clarification or user decision, not resolved here. The product "
            "selects no model and stays post-origin by construction (fit_window_end 2025-05; "
            "every pre-origin use forbidden)."),
        "anti_leakage": "unchanged from " + V1 + "; U7 flagged (see interpretation).",
        "design_time_exposure": [
            "all of rings_bridged/v1 (validation.json, ring_corrections.csv, flags) and the "
            "verifier's findings were seen before this declaration",
            "the registered (e) rows were decomposed after scoring (r2_diagnose; looks logged "
            "2026-10-13)",
            "TKJS(1)'s 2022 vendor days and the v1 feed-check result were seen (amendment 1 is "
            "post-result)"],
        "code_at_registration": code_hashes(["s1const.py", "s1lib.py", "s1io.py", "build.py",
                                             "build_r2.py", "r2_diagnose.py", "prereg_r2.py"]),
    }


def main() -> None:
    path, sha = prereg3.write_prereg(doc())
    print(path, sha)


if __name__ == "__main__":
    main()
