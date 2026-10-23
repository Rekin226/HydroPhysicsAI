"""H1a repair-round packager (2026-10-13): write ``results/dataprep/heads/L1_clean/v<N>/``
with its MANIFEST.json, validation report and product card, in one of two roles.

Why a "v2b" packager. The registered packager ``s6_build_product_v2.py`` (hashed in
H1a.qc_acceptance.v2) is kept unchanged; its gate function ``criteria`` and per-well table
``well_qc_summary`` are imported and used as they are, so the gate is the registered one.
This script changes what the round-1 verifier found wrong in the packaging and adds the
role of the build:

* ``--role registered`` packages the registered v2 build (the build the registered tests
  scored). The after-the-fact verification found its 8 co-located-backed step corrections
  unsupported, so this version is published for provenance only: allowed uses
  ``after_the_fact_analysis`` and ``validator``.
* ``--role reviewed`` packages the post-look v3 build (``s2c_review_v3.py``: co-located
  backing alone no longer corrects a step). Its gate is the conjunction of the registered
  scores on the v2 build and the same criteria recomputed on the v3 build (registered
  function, v3 validation and leakage-probe files); the registered injection recall applies
  unchanged because recovery never reads a flag. Full allowed uses.
* the product card states what the product does (which objects are corrected and at which
  flag, which are only flagged), with the after-the-fact verification of every correction
  and the known weaknesses (displaced side, barometric fits with BE > 1 or a positive
  response, typhoon-time jumps, real-data false flags); v1's card claimed "the default is to
  flag, not correct" while v1 corrected 210 of 228 episodes;
* ``scoring_target`` is for pre-origin folds, held-out sites and secondary scores only (plan
  4.6: the primary 2023-26 score and the prospective scorer keep the registered QC); the
  restriction is written into the manifest parameters and the card;
* the manifest code list names every script that produced the product, its registered
  tests, the design-time evidence scripts (``design_v2/``) and the repair-round scripts;
* validation.json carries the after-the-fact verification (reported, never gated).

Usage: s6_build_product_v2b.py WORKDIR BUILD_DIR VALIDATION_JSON INJECTION_JSON
       LEAKPROBE_JSON VERIFY_JSON --role registered|reviewed
       [--registered VALIDATION_V2_JSON LEAKPROBE_V2_JSON] [--dry-run]
(``--registered`` is required with ``--role reviewed``.)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import guard, manifest  # noqa: E402
from common.paths import PILOTS, REPO, rel_to_repo  # noqa: E402

from h1_heads_qc import baro, meta, rawio  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402
from h1_heads_qc.s6_build_product import STATIC_COLS  # noqa: E402
from h1_heads_qc.s6_build_product_v2 import (  # noqa: E402
    CODE_FILES,
    EVID,
    PREREG,
    PRODUCT,
    criteria,
    well_qc_summary,
)

REPAIR_CODE = ["s6_build_product_v2b.py", "v2_verify.py", "looks_h1a_repair.py",
               "s2c_review_v3.py", "run_v2_repair.sh"]
DESIGN_DIR = "design_v2"
FULL_USES = ["scoring_target", "calibration_target", "datum_fit", "baseline",
             "initial_conditions", "idw_field", "model_design", "selection", "validator",
             "statistical_layer_history", "after_the_fact_analysis", "design_time_evidence"]
PROVENANCE_USES = ["after_the_fact_analysis", "validator"]
USE_NOTES = {
    "scoring_target": "pre-origin folds, held-out sites and secondary scores only; the primary "
                      "2023-26 score and the registered prospective scorer keep the "
                      "registered QC (plan 4.6, WP H1 'Prospective tests')",
    "pre_origin_uses": "calibration_target, datum_fit, baseline, initial_conditions, "
                       "idw_field, model_design, selection and design_time_evidence read the "
                       "time slice window_end <= 2022-12 (column 'date')",
}
FLAG1 = {
    "registered": "flag 1 = barometric loading removed, or an accepted step >= 3 m corrected "
                  "with physical metadata or co-located backing that the neighbour regression "
                  "does not refute (the co-located corrections of this build are unsupported: "
                  "see the card); offset episodes and dropouts are never corrected",
    "reviewed": "flag 1 = barometric loading removed, or an accepted step >= 3 m corrected "
                "with physical metadata (none in this build); co-located backing alone no "
                "longer corrects (repair-round review); offset episodes and dropouts are "
                "never corrected",
}


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def card(val: dict, crit: list, passed: bool, version: str, ver: dict, role: str,
         crit_reg: list | None) -> str:
    s = val["summary"]
    cov = s["monthly_coverage"]
    cm = val["corrected_months_v2"]
    cm1 = val["corrected_months_v1"]
    px = val["proxies"]
    lv = ver.get("corrections_level_check", {})
    sb = ver.get("corrections_sibling_check", {})
    title = ("registered build, provenance only" if role == "registered"
             else "reviewed build, the version for use")
    lines = [f"# heads/L1_clean {version} (WP H1a repair round: {title})", ""]
    if role == "registered":
        lines += ["**Do not use for modelling.** This is the build that the registered tests "
                  "of H1a.qc_acceptance.v2 scored (all 7 criteria passed). The after-the-fact "
                  "verification then found that its step corrections at flag 1 rest on "
                  "co-located backing that does not hold: for 7 of the 8 the well minus "
                  "nested-sibling difference did not shift by half the jump over the following "
                  "year (a well-specific real event, not a datum artefact), and the 8th cannot "
                  "be tested (7 sibling days after the jump). The reviewed build "
                  "published right after it (next version) flags those steps instead. Allowed "
                  "uses: after_the_fact_analysis and validator only.", ""]
    else:
        lines += ["Product for use. It is the registered v2 build with one post-look, "
                  "conservative review: an accepted step whose only backing is co-located is "
                  "flagged 9 instead of corrected (the sibling-difference check confirmed "
                  "none of the 8 such corrections of the registered build: 7 not confirmed, 1 "
                  "untestable). The registered scores "
                  "and the same criteria recomputed on this build are listed below.", ""]
    lines += ["Cleaned groundwater heads of the Choushui fan monitoring network, 2012-01 to "
              "2026-09: the historical cache (2012-22) merged with the new network (2023-26), "
              "screened, barometric loading removed where the fitted response is significant, "
              "datum-offset episodes and dropouts flagged 9 (never corrected), steps of 3 m or "
              "more accepted only when the level change persists (10-day robust shift and a "
              "one-year median shift of at least half the jump), then corrected or flagged 9 "
              "by the flag rule; daily and C20 monthly aggregates with sigma_obs.", "",
              f"Gate: {'passed' if passed else 'NOT passed'} (see Acceptance).", "",
              "Supersedes v1. v1's gate failed and its corrected values were largely wrong: "
              "its step rule never checked that the levels on the two sides of a jump differ "
              "by about the jump, so returns from short dropouts or spikes became multi-year "
              "flag-1 'steps' (07230332: +14.42 m over 2012-01..2016-11; 07260122: -6.1 m over "
              "2018-09..2026-09), and v1 corrected 210 of 228 offset episodes on co-located "
              "backing alone. Do not use v1's corrected values.", "",
              "## What is corrected, what is only flagged", ""]
    for per in ("pre", "post"):
        p = px[per]
        lines.append(
            f"- {per}-origin: {p['v2_steps_accepted']} accepted steps (by flag "
            f"{p['v2_steps_by_flag']}), {p['v2_offset_episodes']} offset episodes and "
            f"{p['v2_dropouts']} dropouts (flag 9, never corrected), rejected step candidates "
            f"flagged as runs {p['v2_rejected_candidates_runs']}; corrected well-months "
            f"(non-zero step correction): {cm[per]['well_months']} at {cm[per]['wells']} "
            f"wells (v1: {cm1[per]['well_months']} at {cm1[per]['wells']}).")
    lines += ["", "After-the-fact checks of the step corrections of this build: independent "
              "10-day medians of the uncorrected series (shift with the jump's sign and at "
              f"least half its size): {lv.get('n_pass', 'n/a')} of "
              f"{lv.get('n_corrections', 'n/a')} pass; co-located sibling-difference check "
              f"(365 days): {sb.get('verdicts', {})}. Details in validation.json "
              "(after_the_fact_verification).", "", "## Coverage", ""]
    for per, c in cov.items():
        lines.append(f"- {per}: {c['wells_with_record']} wells with a record; "
                     f"{c['observed_flag01']} observed well-months (flag 0/1), "
                     f"{c['observed_flag01_c20']} of them C20; flag 3: {c['flag3']}, "
                     f"flag 9: {c['flag9']}, flag 8: {c['flag8']}.")
    lines += ["", "## Flags in use", "",
              "0 observed; 1 " + FLAG1[role].split("= ", 1)[1] + "; 3 step corrected with "
              "neighbour-regression corroboration only (never a calibration target, U2); 8 "
              "missing or sentinel only; 9 suspect (kept, excluded downstream).", "",
              "## Use", ""]
    if role == "registered":
        lines += ["- after_the_fact_analysis and validator only (provenance of the registered "
                  "scores).", ""]
    else:
        lines += ["- Pre-origin uses (calibration targets, datum fits, baselines, initial "
                  "conditions, model design, selection, design-time evidence) open the "
                  "product with `window_end <= 2022-12` (time slice on `date`); "
                  "`station_meta_preorigin.parquet` is the pre-origin station table.",
                  "- `scoring_target`: " + USE_NOTES["scoring_target"] + ".",
                  "- `value_raw`, `baro_m` and `corr_m` keep every correction reversible.", ""]
    lines += ["## Acceptance", ""]
    if crit_reg is not None:
        lines += ["Registered scores (H1a.qc_acceptance.v2, on the registered v2 build):", ""]
        for c in crit_reg:
            lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} [{c['type']}]: "
                         f"{c['criterion']} ({c['threshold']}): {c['measured']}")
        lines += ["", "The same criteria recomputed on this build (post-registration checks; "
                  "the injection recall is the registered one, since recovery never reads a "
                  "flag):", ""]
    else:
        lines += ["Registered scores (H1a.qc_acceptance.v2, this build):", ""]
    for c in crit:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} [{c['type']}]: {c['criterion']} "
                     f"({c['threshold']}): {c['measured']}")
    be = ver.get("barometric_plausibility", {})
    ff = ver.get("false_flag_proxies", {})
    o1 = val["conformance"]["o1_coherence"]["pre"]
    o1c, o1a = o1["corrected_wells"], o1["all_wells"]
    ffp = ff.get("pre", {})
    lines += ["", "## Known limits (read before use)", "",
              "- Real-data false flags are not measured by the injection test (which counts "
              "only flags created by the injections). Proxy: "
              f"{_fmt(ffp.get('episodes_fail_level_check'))} of "
              f"{_fmt(ffp.get('episodes_checked'))} flagged pre-origin offset episodes fail an "
              "independent interior-vs-outside level check (plain 10-day medians), so about "
              "that share of flag-9 episode days may be good data; flag-9 day shares by well "
              "are in well_qc_summary.csv.",
              "- Co-located backing (a nested sibling records through the jump without a "
              "simultaneous jump) cannot tell a datum artefact from a well-specific real "
              "event; " + ("in this build it still backs the flag-1 corrections listed above."
                           if role == "registered" else
                           "in this build it backs no correction. Where the sibling evidence "
                           "suggests the raw record is right on both sides (a real event), the "
                           "displaced segment is still flagged 9 (lost, not wrong data)."),
              "- The displaced side of an accepted step is the impossible side, else the "
              "shorter side; on the injections the shorter-side rule picked the injected side "
              "about half the time.",
              "- Barometric fits: accepted where |A_K| > 3 se (calibrated) and "
              f"-1.2 < A_K < 1.2. {_fmt(be.get('n_accepted_A_K_lt_minus1'))} accepted "
              "epochs have A_K < -1 (BE > 1, implausible for a vented sensor) and "
              f"{_fmt(be.get('n_accepted_A_K_positive'))} have A_K > 0 (ambiguous: an "
              "uncompensated sensor or a confounded fit); their corrections are flag 1 "
              f"(99th percentile |baro_m| at those wells {_fmt(be.get('p99_abs_baro_m_flagged'), 2)}"
              f" m, maximum {_fmt(be.get('max_abs_baro_m_flagged'), 2)} m). The O1 coherence "
              f"gain is small: paired median {_fmt(o1c.get('median_paired_diff'), 4)} over "
              f"{o1c.get('n_wells')} corrected wells; over all {o1a.get('n_wells')} wells the "
              f"median is {_fmt(o1a.get('median_raw'), 4)} -> {_fmt(o1a.get('median_corr'), 4)}.",
              "- Typhoon-time jumps (>= 20 hPa ERA5 fall within +-24 h) are kept as "
              "real-event candidates and never used as episode or step edges. ERA5 "
              "under-reads peak typhoon falls by up to about 17 % (CWA Taichung, Sep 2016).",
              "- Jumps across logger gaps longer than 70 minutes are not detected (the "
              "audit's rule); episodes shorter than about 10 days may not pair; in "
              "pumping-cycle wells datum episodes and pump regimes look alike and are flagged "
              "9 (see well_qc_summary.csv).",
              "- GroundHeight 0.0 is a missing-value code (H1b replaces it after W3). Layer "
              "codes proposed from WellDepth are flag 5, unverified. The 9 stations at "
              "'0 0' have no coordinates.", ""]
    return "\n".join(lines)


def main():
    work, build = Path(sys.argv[1]), Path(sys.argv[2])
    val = json.loads(Path(sys.argv[3]).read_text())
    inj = json.loads(Path(sys.argv[4]).read_text())
    leak = json.loads(Path(sys.argv[5]).read_text())
    ver = json.loads(Path(sys.argv[6]).read_text())
    role = sys.argv[sys.argv.index("--role") + 1]
    if role not in ("registered", "reviewed"):
        raise SystemExit("--role registered|reviewed")
    dry = "--dry-run" in sys.argv
    crit = criteria(val, inj, leak)
    crit_reg, reg_files = None, []
    if role == "reviewed":
        i = sys.argv.index("--registered")
        val2 = json.loads(Path(sys.argv[i + 1]).read_text())
        leak2 = json.loads(Path(sys.argv[i + 2]).read_text())
        crit_reg = criteria(val2, inj, leak2)
        reg_files = [rel_to_repo(Path(sys.argv[i + 1]).resolve()),
                     rel_to_repo(Path(sys.argv[i + 2]).resolve())]
        for c in crit:
            c["type"] = "recomputed on this build: " + c["type"]
    passed = all(c["passed"] for c in crit) and all(c["passed"] for c in (crit_reg or []))
    for c in (crit_reg or []) + crit:
        print(("PASS " if c["passed"] else "FAIL ") + c["id"] + ": " + c["measured"])
    if dry:
        print(card(val, crit, passed, "vDRY", ver, role, crit_reg))
        return
    d = manifest.new_version_dir(PRODUCT)
    for f in ("heads_daily.parquet", "heads_monthly.parquet", "events.csv",
              "barometric_efficiency.csv"):
        shutil.copy2(build / f, d / f)
    P = baro.load_pressure()
    P = P[(P.index >= "2011-12-25") & (P.index < "2026-10-01")]
    PL = P.rename_axis("date").reset_index().melt(id_vars="date", var_name="era5_cell",
                                                  value_name="surface_pressure_hpa")
    PL = PL.dropna(subset=["surface_pressure_hpa"]).sort_values(["era5_cell", "date"])
    PL.to_parquet(d / "pressure_hourly.parquet", index=False)
    sm = pd.read_parquet(work / "station_meta_clean.parquet")
    sm.to_parquet(d / "station_meta_clean.parquet", index=False)
    sm[[c for c in STATIC_COLS if c in sm.columns]].to_parquet(
        d / "station_meta_preorigin.parquet", index=False)
    well_qc_summary(work, build).to_csv(d / "well_qc_summary.csv", index=False)
    report = {"wp": "H1a", "version": d.name, "role": role, "prereg": PREREG,
              "criteria": crit, "criteria_registered_v2_build": crit_reg,
              "gate_passed": passed,
              "gate_rule": ("conjunctive over the 7 criteria (registered function "
                            "s6_build_product_v2.criteria)" if role == "registered" else
                            "conjunctive over the 7 registered criteria on the v2 build and "
                            "the same 7 criteria recomputed on this build"),
              "validation": val,
              "injection": {k: v for k, v in inj.items() if k not in ("rows", "spurious")},
              "leakage_probe": leak,
              "after_the_fact_verification": ver,
              "evidence_files": [rel_to_repo(Path(a).resolve()) for a in sys.argv[3:7]]
              + reg_files}
    Path(d / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    (d / "PRODUCT_CARD.md").write_text(card(val, crit, passed, d.name, ver, role, crit_reg))

    here = Path(__file__).resolve().parent
    code = [here / f for f in CODE_FILES + REPAIR_CODE if (here / f).exists()]
    code += sorted((here / DESIGN_DIR).glob("*.py"))
    code += [REPO / "dataprep_staging" / "harness" / "gaps.py",
             REPO / "dataprep_staging" / "harness" / "gaps_guarded.py",
             REPO / "dataprep_staging" / "harness" / "prereg2.py",
             REPO / "dataprep_staging" / "common" / "paths.py"]
    inputs = [rawio.STATIONS, meta.WRA_LIST, meta.TOWNS,
              PILOTS / "audit_2026-10-11" / "artefact_events.csv",
              REPO / "results" / "dataprep" / "reports" / "A0_evidence" / "repair" / "reruns"
              / "rho_k_result.json",
              REPO / "results" / "twin" / "hybrid_datum_gate_innov" / "evaluation.csv",
              REPO / "results" / "dataprep" / "heads" / "L1_clean" / "v1" / "events.csv",
              REPO / "results" / "dataprep" / "heads" / "L1_clean" / "v1"
              / "heads_monthly.parquet",
              REPO / PREREG]
    inputs += sorted((REPO / EVID / "design_time").glob("*"))
    inputs += baro.pressure_files()
    codis = sorted((REPO / "data_fetch_api" / "new_measurements" / "pressure_hourly")
                   .glob("cwa_codis_*"))[-1]
    inputs += [codis / "SOURCE.json"]
    inputs = [rel_to_repo(p) for p in inputs if Path(p).exists()]
    outputs = [{"path": "heads_daily.parquet"}, {"path": "heads_monthly.parquet"},
               {"path": "events.csv"}, {"path": "pressure_hourly.parquet"},
               {"path": "barometric_efficiency.csv"},
               {"path": "station_meta_clean.parquet"},
               {"path": "station_meta_preorigin.parquet", "fit_window_end": "2022-12"},
               {"path": "well_qc_summary.csv"},
               {"path": "validation.json"}, {"path": "PRODUCT_CARD.md"}]
    uses = FULL_USES if role == "reviewed" else PROVENANCE_USES
    forbidden = ["prospective_scorer", "prior", "prior_width_input", "forcing_frozen",
                 "forcing_recorded_arm"]
    if role == "registered":
        forbidden += [u for u in FULL_USES if u not in PROVENANCE_USES]
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L1", inputs=inputs, code=[str(c) for c in code],
        input_trees=[rel_to_repo(rawio.HIST_DIR), rel_to_repo(rawio.NEW_DIR), rel_to_repo(codis)],
        parameters={
            "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            "role": role,
            "decision_layer": "decide_v2 (round 2): 10-day LAD level persistence of every "
                              "stage-1 jump >= 1 m (not spikes); typhoon-time jumps (>= 20 hPa "
                              "ERA5 fall within +-24 h) excluded from edges; persistent jumps "
                              "paired into episodes by best size match (each jump in one pair; "
                              "flag 9, never corrected); unpaired persistent jumps >= 3 m "
                              "accepted as steps only if the one-year median shift is at least "
                              "half the jump, then the flag rule; rejected candidates flag "
                              "their excursion run",
            "flag_rule": ("plan rule: 1 metadata or co-located backing not refuted by the "
                          "neighbour regression, 3 neighbour regression only, else 9"
                          if role == "registered" else
                          "plan rule with the repair-round review (s2c_review_v3.py, post-look): "
                          "1 physical metadata only, 3 neighbour regression only, else 9 "
                          "(co-located backing alone no longer corrects)"),
            "levelcheck": {"window_days": 10, "min_hourly_medians": 12, "frac_of_jump": 0.5,
                           "segment_window_days": 365, "segment_min_daily_medians": 5},
            "merge": "historical cache 2012-01-01..2022-12-31 + new network 2023-01-01.."
                     "2026-09-30 (2022-10..12 overlap identical, p0_profile.json)",
            "pre_origin_rule": "rows dated <= 2022-12 are built from pre-origin data only "
                               "(separate pre-origin pass, pre-origin backing evidence, "
                               "pre-origin daily screen); verified by the leakage probe",
            "barometric_design": C.BARO_DESIGN, "seed": C.BARO_SEED,
            "pressure_source": "Open-Meteo ERA5 surface_pressure, nearest 0.25-degree cell",
            "flags": {"0": "observed, passed QC", "1": FLAG1[role],
                      "3": "step corrected with neighbour-regression corroboration only",
                      "8": "missing / sentinel only", "9": "suspect (kept, excluded "
                      "downstream): offset episodes, dropouts, uncorrected steps, rejected "
                      "step runs, sample screens"},
            "use_notes": USE_NOTES,
            "packager": "s6_build_product_v2b.py (repair round; gate from the registered "
                        "s6_build_product_v2.criteria, unchanged)",
            "supersedes": "heads/L1_clean v1 (gate not passed; corrected values largely "
                          "spurious; kept published for provenance only)",
            "prereg": PREREG},
        fit_window_end="2026-09",
        time_slicing={"column": "date", "params_fit_window_end": "2022-12", "encoding": "iso"},
        validation_report="validation.json",
        gate={"passed": passed,
              "criteria": ("plan WP H1 (H1a) acceptance, pre-registered in "
                           "H1a.qc_acceptance.v2, scored on this build; see validation.json"
                           if role == "registered" else
                           "registered H1a.qc_acceptance.v2 scores on the v2 build plus the "
                           "same criteria recomputed on this post-look reviewed build; see "
                           "validation.json")},
        allowed_uses=uses, forbidden_uses=forbidden,
        personal_data="none (station metadata: non-personal columns only; LocationAddress and "
                      "contact fields never read)",
        outputs=outputs)
    errs = guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit("time slicing errors: " + "; ".join(errs))
    manifest.write_manifest(d, m)
    print("wrote", d, role, "gate passed" if passed else "GATE NOT PASSED")


if __name__ == "__main__":
    main()
