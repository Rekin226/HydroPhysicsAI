"""Declare rung I1.X5NH.v1, kind 'declaration', before any of its columns or forward runs
exists: Q0's hindcast nudged toward the observed heads every month (column refitted on the
nudged heads), scored where Q0's forecast fails, on the coast.

Why (2026-10-28, exploratory diagnosis, look logged): Q0's post-2022 rebound is the distal
column zone. Its forecast heads rise to the 2012-22 climatology while the observed distal
heads fell; WRA's leveling summaries list coastal townships sinking >= 3 cm/yr in 2023 and
2024. Feeding the twin's own 2023-25 pumping made the coast worse (ring_holdout_ref2023), so
this rung carries the observed heads instead.

No fresh units remain after 2022 (registry check 2026-10-28), so the coastal gates are
design-time; the rolling-origin no-harm gates are scored on leveling the flow model saw in
calibration (the column does not see it: surveys after each origin are withheld).

Usage (repo root, under the X5 lock):
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_nudge/prereg_x5nh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

# townships with >= 50 % of their twin cells west of 182 km (the distal column zone)
COAST = ("麥寮鄉", "北港鎮", "東勢鄉", "水林鄉", "臺西鄉", "四湖鄉", "口湖鄉", "褒忠鄉",
         "元長鄉", "崙背鄉", "大城鄉")
COAST_FRAC = 0.5
DISTAL_KM = 182.0
NUDGE = {"hindcast_gain": 1.0, "hindcast_every": 1, "update_method": "innovation",
         "apex_hold": "calibrated"}
EXTEND_TO = "2025-06"
ORIGINS = ("2016-04", "2017-04", "2018-04", "2019-04", "2020-04", "2021-04")
ROLL_HORIZON = 24
POST_YEARS = (2023, 2024, 2025)          # WRA survey years (May-May intervals)
TOL = 0.5                                # cm/yr on every township bound
LISTED_FLOOR = 3.0                       # the WRA listing threshold, cm/yr
MAX_RATE = 50.0                          # cm/yr: a benchmark pair faster is a datum artefact
GNSS_TO_LEVELING = 1.544                 # cm/yr, x5_insar_anchor v1 card (reported only)
SEED = 2027


def doc() -> dict:
    return {
        "id": "I1.X5NH.v1", "wp": "I1", "kind": "declaration",
        "title": "Q0 with the hindcast nudged monthly toward the observed heads through "
                 f"{EXTEND_TO} (column refitted on the nudged heads) vs Q0, on the coastal "
                 "leveling summaries 2023-25 and rolling origins 2016-21",
        "priors": (
            "Q0 (three-zone column, seed-2027 leveling R2 0.653) forecasts every distal cell "
            "rising after 2022 (2023-25 rate < -0.1 cm/yr): its heads rise to the 2012-22 "
            "climatology (+0.9 m in distal Yunlin L2) while 39 observed distal Yunlin wells "
            "fell 1.36 m in 2023; the distal column (tau 57 d, Ske/Skv 0.6) is near-elastic, "
            "about 1.5 cm per m. WRA leveling summaries list 6 coastal townships >= 3 cm/yr in "
            "2023 and 5 in 2024, none in 2025 (Yunlin max 2.3). Rough expectation from "
            "observed heads alone at 1.5 cm/m: the right sign but about 0.3-0.5 cm/yr, so the "
            "listed-township gate may fail on the column, not the forcing. Nudging every 12 "
            "months under a free-run column cost leveling R2 0.599 -> 0.391 (2026-09-19); "
            "here the column is refitted on the nudged heads."),
        "arms": {
            "q0": "Q0 as registered: results/twin_forward/datum_gate_q0.npz (origin 2022-12, "
                  "free hindcast, climatology after), column stage3_datum_gate_q0",
            "q0_nh": f"Q0 theta; column refitted with calibrate_coupled --hindcast-gain "
                     f"{NUDGE['hindcast_gain']} --hindcast-every {NUDGE['hindcast_every']} "
                     f"--update-method {NUDGE['update_method']} --apex-hold "
                     f"{NUDGE['apex_hold']}; forward with the same nudging, the record "
                     f"extended to {EXTEND_TO} (twin_inputs/clean_heads v2), climatology "
                     "after the extended record",
            "q0_r<YYYY>f / q0_r<YYYY>n": f"rolling origins {', '.join(ORIGINS)}: Q0 theta, "
                     "column refitted with --withhold-surveys <YYYY>-05:2022-12, free (f) or "
                     "nudged as q0_nh (n); forward --record-end <origin> --horizon "
                     f"{ROLL_HORIZON}, baseline only"},
        "recipe": (
            "column: calibrate_coupled --target leveling --configs zonal --hpc0-guard-days "
            f"365 --seed {SEED} --device cuda [nudging flags] [--withhold-surveys]; forward: "
            "Q0's (--gain 1.0 --ic-members 1 --update-method innovation --restart-taper-km 5 "
            "--apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 "
            "--compile-matvec --save-members none), plus --hindcast-gain/--hindcast-every "
            f"for nudged arms; q0_nh also --extend-record-to {EXTEND_TO} --wells-dir "
            "results/dataprep/twin_inputs/clean_heads/v2/wells and Q0's cut30 scenario, "
            "horizon 120. Code: main 5f3af18 plus the X5NH patch (inputs.truncate_record/"
            "extend_record, forward --record-end/--extend-record-to and the column pairing "
            "check, calibrate_coupled --hindcast-gain; results/twin_runs/par31/x5nh_code.diff, sha256 "
            "16db367124ed589bf3e2c03de06d0865ecbefbcfcf40eea55187b0f6caf0cd90), frozen in "
            "~/HydroPhysicsAI-x5nh; engineering checks in results/twin_runs/par31/"
            "ENGINEERING.md; batch results/twin_runs/par31/run.sh. "
            "Scorer: dataprep_staging/x5_nudge/score_x5nh.py, written before any arm exists."),
        "zones": f"Q0's three column zones; distal = twin cell centroid easting < {DISTAL_KM} "
                 f"km; coastal townships = >= {COAST_FRAC:.0%} of their cells distal: "
                 + ", ".join(COAST),
        "burn_in": "unchanged",
        "folds": f"heads: Q0's; leveling site folds seed {SEED} (Q0's partition); rolling "
                 "columns: surveys after each origin withheld from the loss",
        "gate_scores": NA("pass/fail gates on counts and ratios; no weighted score"),
        "primary_model": "q0_nh",
        "inputs": ["results/dataprep/subsidence_public_reports/postorigin/v1/township_bounds.csv",
                   "results/dataprep/subsidence_public_reports/postorigin/v1/county_intervals.csv",
                   "results/dataprep/subsidence_public_reports/preorigin/v1/township_bounds.csv",
                   "results/dataprep/subsidence_public_reports/preorigin/v1/county_intervals.csv",
                   "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet",
                   "data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json",
                   "results/dataprep/twin_inputs/clean_heads/v2/MANIFEST.json"],
        "tests": NA("design-time: every post-2022 subsidence family was looked at (S1a, S1b, "
                    "S3, S5, X5 exploration, ring_holdout_ref2023, the 2026-10-28 inventory)"),
        "fresh_units": NA("no post-2022 subsidence unit is fresh; the rolling-origin leveling "
                          "was a calibration target of the flow model"),
        "thresholds": {
            "units": (
                "township bound: model peak = max over the township's cells of the forward "
                "subsidence rate over the WRA survey interval (cm/yr, + = sinking; scenario "
                "baseline, ensemble mean). Listed by all sources: peak >= "
                f"{LISTED_FLOOR} - {TOL} (lower) and <= printed county max + {TOL} (upper); "
                f"unlisted: peak <= printed upper bound + {TOL}; listed by some sources only "
                "(ambiguous): excluded. Benchmark pair: rate between a benchmark's last survey "
                "in the origin's year (after the origin) and its last survey the next year, "
                f"pairs faster than {MAX_RATE} cm/yr dropped, model rate between the same "
                "months at its cell"),
            "passes_if_all": [
                "G1 coastal lower bounds 2023-25: q0_nh meets >= 6 of the listed coastal "
                "township-years (11: 6 in 2023, 5 in 2024)",
                "G2 coastal upper bounds 2023-25: q0_nh breaks <= 3 of all coastal "
                "township-year upper bounds",
                f"G3 column: q0_nh leveling out-of-fold R2 (seed {SEED}) >= Q0's - 0.02",
                "G4 no harm, rolling origins 2016-04..2020-04 (benchmark pairs 2016/17.."
                "2020/21), pooled over origins: RMSE of the rate, nudged <= 1.05 x free, on "
                "the distal benchmarks and on all benchmarks, each",
                "G5 rolling origins 2016-04..2021-04 (WRA survey years 2017-2022): coastal "
                "lower bounds met, nudged >= free"],
            "on_pass": "q0_nh (state nudged to the observed heads through 2025-06, its column) "
                       "becomes the baseline of the policy runs, labelled design-time until "
                       "the user rules on it",
            "on_fail": "no change; Q0 stays. Read with interpretation",
            "reported_not_gating": [
                "the same counts for q0, and per township-year peaks for both arms",
                "rings 元長 and 客厝 (ring_holdout_ref2023 observed 2023-02..2025-03 rates) vs "
                "each arm's rate at their cells",
                "distal GNSS stations (gnss_clean/full v4 flags 0-1, >= 18 months in "
                "2023-01..2025-03): slope in the GNSS frame and shifted by "
                f"{GNSS_TO_LEVELING} cm/yr to leveling, vs each arm",
                "rolling: per-origin RMSE and bias, mid-zone benchmarks, each column's "
                "in-sample leveling R2",
                "fan subsidence 2023-25 and the rising share (rate < -0.1 cm/yr) by county",
                "assimilated hindcast head R2 (not a free-running score)"]},
        "attempt_policy": {
            "scored_runs": "one per arm; no rerun after any metric is seen",
            "engineering_retries": "free before any metric is written, each logged",
            "pre_declared_alternatives": [], "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True, "report": "every attempt, failures included"},
        "interpretation": (
            "G1 failing with G2-G5 passing says the observed heads move the coast the right "
            "way without harm but the near-elastic distal column cannot turn them into the "
            "listed rates: the next defect is the column (a slow component), not the "
            "forcing. G3 failing says the column cannot be fitted on nudged heads as well "
            "as on free ones. G4 or G5 failing says nudging the state worsens the next "
            "year's forecast where leveling exists. A pass is design-time: it shows the "
            "fix reproduces 2023-25 on data already seen, not that it forecasts."),
        "no_prior_look_statement": (
            "No q0_nh or rolling-origin column or forward run exists. Q0's own forward was "
            "looked at against distal GNSS, heads and the ring holdout (diagnosis "
            "2026-10-28); the township summaries were read as listed/unlisted per township "
            "(inventory 2026-10-28) but never compared with any model."),
        "written_by": "dataprep_staging/x5_nudge/prereg_x5nh.py",
    }


def main() -> int:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    if "--validate" in sys.argv[1:]:
        print("valid; not written")
        return 0
    print(*prereg3.write_prereg(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
