"""Declaration X5.n2_reading.v1 (harness.prereg3, kind 'declaration'; round 8 rank 4):
synthetic operating characteristics of N2's decision rule for Yunlin (power at k = 1, false
pass at k = 0, wrong-size pass at k = 0.5 and 2, 90 % interval coverage, a canal-correlated
confounder cell), registered BEFORE any replicate is drawn and before the replay forward runs.

A fresh first attempt, model-only and synthetic. It reuses the X5.n2c_reading.v1 harness
(dataprep_staging/n2c_reading/synth_n2c.py, imported read-only) and the shared N2/N2C scorer
that declaration bound (bootstrap_scorer.py, sha256 f8990d61...), wrapped, never edited. The
anomaly multipliers come from pump_electricity/preorigin/v3, used as design_time_evidence (its
MANIFEST allows it; precedent: the OSSE ruling). X1 is never used (its MANIFEST forbids
design_time_evidence). No observation is read.

``draft_prereg(seed)`` returns the prereg3 document; the seed has no default (the lead picks
it). ``--validate`` checks it and prints readiness; ``--write`` (the lead only) registers it.

    PY=~/miniforge3/envs/hydro/bin/python
    bash dataprep_staging/x5_n2_reading/run_n2_reading.sh replays
    $PY -I dataprep_staging/x5_n2_reading/prereg_n2_reading.py --validate --seed <S> --quiet
    $PY -I dataprep_staging/x5_n2_reading/prereg_n2_reading.py --write --seed <S> --quiet \\
        --expect-self-sha <sha256 printed by --validate>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "dataprep_staging"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import synth_n2 as S  # noqa: E402
from harness import prereg3  # noqa: E402

BS = S.BS
NA = prereg3.not_applicable
ID = S.DECL_ID
THIS = f"{S.X}/prereg_n2_reading.py"
RUNNER = f"{S.X}/run_n2_reading.sh"
TESTS = "tests/test_x5_n2_reading.py"
N2_CODE = "1fff145"
WT_TXT = "$HOME/HydroPhysicsAI-n2rd-1fff145"
HPY_TXT = "$HOME/miniforge3/envs/hydro/bin/python"
G = "results/twin_runs/stage3_datum_gate"
FLAGS = ("--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
         "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
         "--column-hpc0-fast-days 365 --device cpu --save-members none")
RUNNER_LINES = (f"ID={ID}", f"N2_CODE={N2_CODE}", f"WT_TXT='{WT_TXT}'",
                f'FLAGS="{FLAGS}"', "YEARS=(2012 2013 2014 2015 2016 2017 2018)",
                "DECL=results/dataprep/preregistration/$ID.json",
                f"OUT={S.FWD_STEM}", "OUTD=results/dataprep/reports/$ID",
                'export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"')
DEADLINE = dt.datetime(2026, 11, 20, 23, 59, tzinfo=dt.timezone.utc)   # round-8 plan, rank 4
PRODUCT_MANIFEST = f"{S.PRODUCT}/MANIFEST.json"
X1_MANIFEST = "results/dataprep/electricity_x1_tpc/full/v1/MANIFEST.json"
CITED = ("results/dataprep/reports/I1_changhua_local/report.json",
         "results/dataprep/reports/I1_changhua_interannual/report.json",
         "results/dataprep/reports/I1_changhua_drivers/report.json",
         "results/dataprep/reports/I1_changhua_drivers/report.md",
         "results/twin/hybrid_datum_gate_ccz/report.json",
         "results/twin_runs/stage3_datum_gate/stage3_kfold_wells.csv")

# the engineering gate as run before registration (synth_n2.py engineering, 2026-11-03):
# harness vs the reference computation of N2's registered statistic on frozen forwards
ENGINEERING_GATE_RECORD = {
    "run": "2026-11-03 (rerun after review fixes), synth_n2.py engineering (BLAS one "
           "thread); printed output below only",
    "frozen_forwards": [S.PAGE_FORWARD + " (cut30, retire_aqua vs baseline)",
                        S.R1C_REPLAY + " (RP2012..RP2018 vs baseline)"],
    "n_contrasts": 9, "n_worlds": 36, "verdict_agreement": "36/36",
    "max_abs_diff": {"beta": 2.886579864025407e-15, "lo": 1.7763568394002505e-15,
                     "hi": 3.552713678800501e-15, "rms": 0.0,
                     "rmse_pr": 4.440892098500626e-16, "rmse_zr": 4.440892098500626e-16},
    "tol": 1e-08, "exact_plumbing_ok": True,
    "scorer_sha256": "f8990d6182b7b080608f77b7e5f7a449ea722f97f845cd83478997cdc9ae75bd",
    "harness_sha256": {
        "dataprep_staging/x5_n2_reading/synth_n2.py":
            "5ccc1b212b0c3d0a111b93bd4bf64902d0428f971c4fb90e942920582e1ca2bf",
        "dataprep_staging/n2c_reading/synth_n2c.py":
            "3bd79f713e9ad0bc57a5415c4ccf89b95472cd6fcc8124de7be6d88c2ff993c8",
        "dataprep_staging/n2_policy/bootstrap_scorer.py":
            "f8990d6182b7b080608f77b7e5f7a449ea722f97f845cd83478997cdc9ae75bd"},
    "ok": True}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "missing"


def canonical_sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def command() -> str:
    """The forward line, exactly as ``DRY=1 run_n2_reading.sh forward`` prints it."""
    return (f"PYTHONPATH={WT_TXT} {HPY_TXT} -P -m hydrophysics.twin.forward --theta "
            f"{G}/stage3_theta.json --vep-json {G}/coupled_leveling/vep_zonal_leveling.json "
            f"{FLAGS}"
            + "".join(f" --recorded-pumping 'RP{y}={S.REPLAYS}/rp{y}.csv@2023-01'"
                      for y in S.REPLAY_YEARS) + f" --out {S.FWD_STEM}")


def replays_block() -> dict:
    rj = REPO / S.REPLAYS / "replays.json"
    return {"replays_json": f"{S.REPLAYS}/replays.json", "replays_json_sha256": sha(rj),
            "content": json.loads(rj.read_text()) if rj.exists() else None}


def product_use() -> dict:
    """What the electricity product's MANIFEST grants, read from the file."""
    m = json.loads((REPO / PRODUCT_MANIFEST).read_text())
    x1 = json.loads((REPO / X1_MANIFEST).read_text()) if (REPO / X1_MANIFEST).exists() else {}
    return {"product": f"{m['product']} {m['version']}", "manifest": PRODUCT_MANIFEST,
            "manifest_sha256": sha(REPO / PRODUCT_MANIFEST),
            "allowed_uses": m["allowed_uses"], "forbidden_uses": m["forbidden_uses"],
            "exemption_uses": sorted({u for x in m.get("exemptions", []) for u in x["uses"]}),
            "fit_window_end": m.get("fit_window_end"),
            "non_exempt_fit_window_end": m.get("non_exempt_fit_window_end"),
            "gate_passed": m["gate"]["passed"],
            "use_made": f"{S.USE}: the anomaly multipliers of synthetic future months (never "
                        "scored against an observation; never a recorded arm of a registered "
                        "test), opened through common.guard for that use",
            "not_made": "forcing_recorded_arm (forbidden): the replays use twin.forward's "
                        "recorded-pumping MECHANISM as a design device only, as in "
                        "X5.n2c_reading.v1",
            "x1": {"manifest": X1_MANIFEST, "allows_design_time_evidence":
                   "design_time_evidence" in (x1.get("allowed_uses") or []),
                   "use_made": "metadata only: this script opens X1's MANIFEST.json to record "
                               "that it does not allow design_time_evidence; no X1 data file "
                               "is opened by any step; extract loads the N2 rehearsal forward "
                               f"({S.REHEARSAL}), whose PR arm was forced by X1 2023-01.."
                               "2026-08: the arm is loaded with the file, never indexed "
                               "(disclosed)"}}


def inputs(code: dict) -> list:
    return ([{"path": p, "sha256": h} for p, h in code.items()]
            + [TESTS, f"results/dataprep/preregistration/{S.N2_ID}.json",
               f"results/dataprep/preregistration/{S.N2C_READING_ID}.json",
               PRODUCT_MANIFEST, f"{S.PRODUCT}/{S.TABLE}", f"{S.REPLAYS}/replays.json",
               S.WELLS_XY, S.REHEARSAL, S.PAGE_FORWARD, S.R1C_REPLAY,
               f"{G}/stage3_theta.json", f"{G}/coupled_leveling/vep_zonal_leveling.json",
               *CITED])


def draft_prereg(seed) -> dict:
    """The prereg3 document of X5.n2_reading.v1 for ``seed`` (required; no default)."""
    seed = S.check_seed(seed, real=True)
    code = S.frozen_code()
    design = S.design_spec(seed)
    return {
        "id": ID, "wp": "X5", "kind": "declaration",
        "title": "Synthetic operating characteristics of N2's Yunlin decision rule (power at "
                 "k = 1, confounded false pass at k = 0, wrong-size pass at k = 0.5 and 2, "
                 "90 % interval coverage) with the hashed shared N2/N2C scorer; model-only; "
                 "reported either way",
        "priors": (
            "N2 decides the page model's Yunlin head response on 2026-11..2027-10 with the "
            "per-well-demeaned slope beta of e = obs - ZR on c = PR - ZR over 84 wells, a 5 km "
            "block bootstrap 90 % interval and the rule 'response supported' iff the interval "
            "excludes 0, beta in [1/1.5, 1.5] and RMSE(PR) < RMSE(ZR). Its power_check only "
            "shows the contrast is 10-15x the informative threshold and says detectability "
            "'rests on the residual noise of ZR'. Its Changhua twin N2C has a measured "
            "reading (X5.n2c_reading.v1); N2 has none. Yunlin residuals are larger (R0 "
            "anomaly RMSE 1.86 m) and more correlated (0.79 within 5 km, 0.27 beyond 20 km) "
            "than Changhua's, and the observed regional head follows census energy (-0.87) "
            "and canal deliveries (+0.88). This declaration measures, before any N2 "
            "statistic exists, how often N2's rule passes when the response is absent but a "
            "canal-like term co-moves with pumping, when it is half or double the model's, "
            "how well the interval covers a correct response, and the power. Model-only; no "
            "X1."),
        "rulings": {
            "round-8 plan rank 4 (lead)": "register X5.n2_reading.v1, a fresh first attempt, "
                                          "model-only with synthetic worlds; reuse the "
                                          "X5.n2c_reading harness and the shared scorer "
                                          "f8990d61 (wrap, never edit); multipliers from "
                                          "pump_electricity preorigin v3 as "
                                          "design_time_evidence; report either way",
            "page line": "needs ruling R8.8; the ledger and README may record the reading now",
            "seed": "picked by the lead at registration (no default in the code)"},
        "product_use": product_use(),
        "shared_scorer_binding": {
            "path": S.SCORER, "sha256": code[S.SCORER], "version": BS.VERSION,
            "bound_by": f"{S.N2C_READING_ID} (registration sha256 {S.N2C_READING_SHA}), "
                        "code_sha256 of that declaration",
            "rebinds": False,
            "note": "this declaration wraps the scorer (via synth_n2c.score_world) and binds "
                    "the same sha256 in frozen_code; it changes no interpretation fixed there"},
        "design": design,
        "frozen_code": code,
        "replays": replays_block(),
        "recipe": {
            "code": f"N2's registered code commit {N2_CODE} (hydrophysics/twin/forward.py "
                    "088b8bd5, recorded_pumping.py 4a8c99e4, scenario.py c6f250fc) in a clean "
                    f"detached worktree {WT_TXT}",
            "forward": command(),
            "forward_vs_n2": "N2's registered forward_command (page model main set "
                             "stage3_datum_gate, page column, CPU, --ic-members 1, "
                             "--save-members none) with the PR arm replaced by RP2012..RP2018; "
                             "the flag segment is checked identical to N2's (runner_problems)",
            "runner": f"bash {RUNNER} replays (before registration), forward (CPU, tmux), "
                      "extract, run <seed>",
            "cpu": "one forward with 8 scenarios on CPU (hours) plus a few minutes of "
                   "replicates"},
        "zones": "N2's 84 Yunlin wells (Yunlin screens of dataprep_staging/x5_class_share/"
                 "wells_frozen.json, checked equal to N2's external-unit ids and to the cells "
                 f"and layers of {S.WELLS_XY}; 5 km blocks from that file's x, y)",
        "burn_in": NA("synthetic; the AR(1) noise starts stationary"),
        "folds": NA("synthetic declaration; no folds"),
        "gate_scores": NA("declared wording thresholds; no weighted score"),
        "primary_model": "the page model (stage3_datum_gate main set + coupled_leveling column, "
                         "N2's primary model) supplies the contrast; the characterised object "
                         "is N2's decision rule as implemented by the shared scorer",
        "inputs": inputs(code),
        "tests": NA("synthetic declaration: no fresh unit exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "engineering": [
                "pre-registration gate (run): on frozen model output only (the page forward "
                f"{S.PAGE_FORWARD} cut30 and retire_aqua arms and {S.R1C_REPLAY}'s RP arms, "
                "minus their baselines, at the 84 Yunlin screens over N2's window), the harness "
                "(synth_n2c.score_world -> bootstrap_scorer.score) equals the reference "
                "computation of N2's registered statistic from its wording (dummy-variable OLS, "
                "the block-resampled refit, the informative gate, equal-well RMSEs, the "
                "decision rule) within 1e-8 on beta, its interval, the gate's RMS and the "
                "RMSEs, the verdicts agree, exact plumbing passes, and the scorer is f8990d61; "
                "the record is bound to the sha256 of synth_n2.py, synth_n2c.py and the "
                "scorer, and readiness refuses when any differs",
                "exact plumbing (no noise, kappa 0): beta == k for every informative replay; "
                "a failure is a code defect: engineering failure, no reading",
                f"plumbing cell (k = 1, rho = 0, kappa 0, noise x{S.PLUMBING_NOISE_SCALE}, "
                f"{S.PLUMBING_REPLICATES} replicates): median beta in {list(S.PLUMBING_BAND)}; "
                "outside it the TERMINAL reading 'not_estimable' is written and is final",
                f"extract: the run's baseline arm reproduces the N2 rehearsal's baseline arm "
                f"({S.REHEARSAL}, N2's registered command) within {S.REPRO_TOL_M} m at the 84 "
                "screens over all 252 months (only the rehearsal's baseline scenario is "
                "indexed); every replay arm ran on the declared file (sha256)"],
            "engineering_gate_record": ENGINEERING_GATE_RECORD,
            "reading": design["reading"],
            "published_fields": {k: {kk: v[kk] for kk in ("value", "source", "field")}
                                 for k, v in S.PARAMS.items()}},
        "reading_reported_either_way": (
            "the wording (decisive, informative, weak or not_estimable) and every rate are "
            "written to report.json and reported on the ledger whichever way they fall; no "
            "threshold changes after the run"),
        "evidence_and_ledger_mapping": {
            "scope": "C01's D part only (N2's within-window anomaly response).",
            "decisive": "C01's D becomes 'D-measured, decisive by design' with the rates",
            "pass_controlled_low_power": "C01's D becomes 'D-measured, pass-controlled, low "
                                         "power' with the rates (a non-pass says little)",
            "informative": "C01's D becomes 'D-measured, informative' with the rates printed "
                           "beside any N2 verdict",
            "weak": "C01's D is marked 'weak by design'; an N2 'response supported' would be "
                    "reported, not shown (page under R8.8)",
            "not_estimable": "C01's D stays 'decidability unknown'",
            "never": "the N2 verdict word never changes; nothing here shows more than N2's "
                     "own rules allow; page lines only under ruling R8.8"},
        "attempt_policy": {
            "scored_runs": "one forward, one extract, one registered run; report.json is "
                           "written once (mode 'x') and is final, whatever it shows",
            "score_once": True,
            "exploratory_followups_max": 0,
            "pre_declared_alternatives": [],
            "sealing": "the forward's stdout and outputs and the panel are not opened, printed "
                       "or summarised before report.json exists (the runner prints shas; "
                       "extract logs the panel shape, block count and the reproduction check)",
            "engineering_retries": "free before the reading exists: a crash, a missing file, a "
                                   "failed reproduction or a failed EXACT plumbing check; each "
                                   "logged in attempts.jsonl (engineering_failure_*.json "
                                   "carries no panel summary); a fix that changes frozen_code "
                                   "needs a v2 that discloses the failed attempts and may "
                                   "change code only (never PARAMS, GRID, REPLICATES, the "
                                   "seed, the plumbing cell, READING, WORDING or the replays)",
            "terminal_outcomes": ["not_estimable (plumbing cell outside [0.9, 1.1] with exact "
                                  "plumbing passing): final; no v2 may follow it"],
            "thresholds_frozen": True,
            "deadline": "registered by 2026-11-20 (round-8 plan) and before any N2 statistic "
                        "exists (N2 scores not before 2028-01-15)"},
        "interpretation": (
            "A high confounded false-pass rate means an N2 pass could come from an omitted "
            "canal co-movement alone (weak by design). High wrong-size pass rates or a poorly "
            "covering interval mean a shown response must carry those rates (informative). "
            "Neither means the rule is decisive by design under these worlds. Power tells a "
            "reader what an N2 'not supported' can mean. All of this is about the decision "
            "rule under declared synthetic worlds; none of it is evidence about the aquifer."),
        "disclosures": disclosures(),
        "no_prior_look_statement": (
            "No replay forward of the page model on pump_electricity/preorigin/v3 anomalies "
            "exists; no synthetic replicate on the page model's contrast has been drawn; no "
            "power, false-pass, wrong-size or coverage rate of N2 has been computed; no N2 "
            "window head exists."),
        "written_by": THIS,
    }


def disclosures() -> list[str]:
    return [
        "every noise and confounder parameter is a published design-time statistic of spent "
        "2012-22 heads; the drafter read exactly the cited fields (I1_changhua_local "
        "part_a_local.yunlin.rmse_r0_mean and n_wells; I1_changhua_interannual "
        "counties.yunlin.residual.R0.by_distance '0-5 km' and '>20 km' and "
        "counties.yunlin.observed.regional_vs_drivers_r census_energy and canal_deliveries; "
        "I1_changhua_drivers summary.yunlin.median_cv_r2 raw:canal_w1_cell and the report.md "
        "R0 row of the R2 table; hybrid_datum_gate_ccz rho.median_own), the 'runs' labels of "
        "the two I1 reports, and the key names (no values) of those reports",
        "R0 (N2's declared secondary) stands in for sigma, the block and county shares and "
        "kappa. The page model's post-origin RMSE (results/twin/hybrid_datum_gate_innov, "
        "model_datum, 2023-26) EXISTS and is excluded under U7 (no post-2022 data in design; "
        "it was not opened). The page model's PRE-origin out-of-fold per-well rmse_model_m "
        "(stage3_datum_gate/stage3_kfold_wells.csv; the drafter read the sid and rmse_model_m "
        "columns: 158 rows, 84 Yunlin, mean 9.204 m, median 6.278 m, rms 14.91 m) is a "
        "held-out-well absolute-head RMSE and enters two report-only cells only",
        "unlike X5.n2c_reading.v1, the noise has a county-wide component (R0's Yunlin pairs "
        "beyond 20 km correlate +0.27; Changhua's -0.08); 0.27 is a FULL-RECORD pair "
        "correlation (2012-22 residual series), not a within-12-month one; two report-only "
        "cells use N2C's structure (no county part, same total sd)",
        "kappa (lead ruling, review B1): sqrt of the deseasonalised UNIQUE canal R2 (median-"
        "well deseasonalised:all_drivers 0.7435 minus deseasonalised:all_minus_canal_w1_cell "
        "-0.0017 = 0.745; a difference of medians, not a median of differences; the drafter "
        "also read deseasonalised:canal_w1_cell 0.2753) times R0's anomaly sd (1.856 / "
        "sqrt(0.71)) = 1.90 m, capped at sqrt(share_county) sigma = 0.961 m (the cap binds), "
        "and taken out of the county-wide noise (whose AR part is then 0); the anomaly sd "
        "comes from R0's raw-anomaly RMSE and R2, applied to a deseasonalised R2. The "
        "previous raw canal-alone 1.43 m (X5.n2c_reading.v1's derivation) is report-only; "
        "its kappa^2 2.03 m^2 exceeds share_county sigma^2 0.92 m^2, so those cells carry "
        "more than sigma^2 in total",
        "the confounder's within-window sd is kappa x sqrt(0.404), the share of an AR(1) "
        "(phi 0.859) variance that survives per-well demeaning over 12 months; the AR noise "
        "parts are drawn at their marginal sd and lose the same share to demeaning",
        "the cited I1_changhua_drivers and I1_changhua_interannual reports are labelled "
        "after_the_fact_analysis (U7: never a gate, threshold or choice); here they only "
        "parameterise the synthetic worlds, as in X5.n2c_reading.v1",
        "rho_abs 0.87 is the mean of the Yunlin energy and canal magnitudes rounded to 2 "
        "decimals (proxy: no canal-vs-pumping correlation is published). The two published "
        "correlations only bound corr(canal, -energy) to [0.52, 1.00] (positive semi-definite "
        "3 x 3 correlation matrix), so two report-only cells use rho -0.52. The cited values "
        "are INTERANNUAL (regional series) correlations transferred to a monthly within-window "
        "co-movement",
        "the wording tiers 'decisive' and 'pass_controlled_low_power' and the power floor "
        "0.80 are the lead's (review B2); X5.n2c_reading.v1 never used the word 'decisive' "
        "(its display keys were reported_not_shown, rates_printed and power); its thresholds "
        "0.25 / 0.25 / 0.80 are copied; false pass is the max over rho (review S3)",
        "the replay multipliers are pump_electricity/preorigin/v3's census-based Yunlin "
        "in-grid all-class ratios; N2's real PR arm uses X1's in_grid_hybrid (TPC-downscaled) "
        "series, whose seasonal profile correlates 0.37 with the census (N2 power_check "
        "caveat); the gating cells score the rule with the contrast that generated them, so "
        "forcing error between X1 and the true pumping is NOT modelled there; two report-only "
        "cells score with lambda c_j + sqrt(1 - lambda^2) c_(j+1), rescaled, lambda 0.8 fixed "
        "in this draft (review S2)",
        "the replays use twin.forward's recorded-pumping mechanism for synthetic future months "
        "under the product's allowed use design_time_evidence (its MANIFEST forbids "
        "forcing_recorded_arm, which is not the use made here)",
        "while drafting, the builder read pump_electricity/preorigin/v3's MANIFEST.json and "
        "PRODUCT_CARD.md, the schema, county/scope/class row counts, flag counts and date "
        "range of county_class_monthly.parquet (no value)",
        "the engineering gate read frozen model output only (datum_gate_innov cut30, "
        "retire_aqua and baseline; n2c_reading_r1c RP arms and baseline) at the 84 Yunlin "
        "screens over N2's window and printed only differences, counts and booleans. The R1c "
        "replay is a DIFFERENT model (stage3_datum_gate_ccz, saved with yearly members) used "
        "only to test the arithmetic; it says nothing about the page model's contrast",
        "the extract-time reproduction indexes only the baseline scenario of the N2 rehearsal "
        "forward; its PR arm (X1 2023-01..2026-08 forcing) is loaded with the file but never "
        "indexed; the rehearsal's code commit is not recorded in the file, so a mismatch is "
        "an engineering failure (free retry), not a reading. N2's registration names the "
        "rehearsal dir but hashes no rehearsal file, so extract refuses unless the file is the "
        f"sha256 pinned at drafting ({S.REHEARSAL_SHA}); this declaration hashes it as an "
        "input",
        "while drafting, the builder printed the 'what', 'reading' and 'provenance' fields and "
        "the key names of n2_power_rollout.json (no window value) and only the key names of "
        "n2_power_check.json (it holds X1 values); it read N2's and X5.n2c_reading.v1's "
        "registrations and X5.n2c_reading.v1's result.md (synthetic rates, already quoted in "
        "the round-8 plan); no parameter here is set from them",
        "the registered N2 numbers on a replay (power_check: within-window RMS p50 0.65 m, "
        "p90 0.83 m) come from an X1-forced rollout over the record; reproducing them would "
        "use X1, so the engineering gate checks the harness against N2's registered "
        "definition on frozen forwards instead; the report prints the replays' within-window "
        "RMS beside 0.65 / 0.83 m (report only)",
        "the drafter ran the smoke mode on a synthetic stand-in panel (no model output) with "
        f"its own refused seed {S.SMOKE_SEED}; it printed only 'ran ok'",
        "the synthetic panel is complete (84 wells x 12 months, 12 recorded months); the real "
        "window may lack months or wells",
        "phi is the median lag-1 autocorrelation of OBSERVED head anomalies over 158 wells in "
        "both counties (X5.n2c_reading.v1's value), not of the page model's Yunlin residuals",
        f"the unit tests use seed {S.TEST_SEED}, which draft_prereg, simulate and the run "
        "CLI refuse (review M5)",
        "no X1 value enters any parameter"]


# ----------------------------------------------------------------------------- checks
def _get(path: str, keys: list[str]):
    d = json.loads((REPO / path).read_text())
    for k in keys:
        d = d[k]
    return d


def published_problems() -> list[str]:
    """The cited report fields carry the declared values."""
    out = []
    P = S.PARAMS
    loc, ia = CITED[0], CITED[1]
    chk = [("sigma_tot_m", loc, ["part_a_local", "yunlin", "rmse_r0_mean"]),
           ("share_0_5km", ia, ["counties", "yunlin", "residual", "R0", "by_distance",
                                "0-5 km", "median_r"]),
           ("share_county", ia, ["counties", "yunlin", "residual", "R0", "by_distance",
                                 ">20 km", "median_r"]),
           ("phi", CITED[4], ["rho", "median_own"])]
    for name, path, keys in chk:
        try:
            if float(_get(path, keys)) != P[name]["value"]:
                out.append(f"{name}: {path} {'.'.join(keys)} differs")
        except (OSError, KeyError, ValueError) as err:
            out.append(f"{name}: {err}")
    try:
        k = P["kappa_m"]["inputs"]
        if float(_get(loc, ["part_a_local", "yunlin", "rmse_r0_mean"])) != k["rmse_r0_m"]:
            out.append("kappa: R0 RMSE differs")
        md = (REPO / CITED[3]).read_text()
        if not re.search(r"\|\s*R0\s*\|\s*0\.14\s*\|\s*0\.29\s*\|", md):
            out.append("kappa: R0 Yunlin R2 0.29 not in report.md")
        if _get(loc, ["runs", "R0"]) != "cwa" or _get(ia, ["runs", "R0"]) != "cwa":
            out.append("R0 is not the cwa run in the I1 reports")
        rr = _get(ia, ["counties", "yunlin", "observed", "regional_vs_drivers_r"])
        r = P["rho_abs"]["inputs"]
        if rr["census_energy"] != r["census_energy"] or \
                rr["canal_deliveries"] != r["canal_deliveries"]:
            out.append("rho: published correlations differ")
        if round((abs(r["census_energy"]) + abs(r["canal_deliveries"])) / 2, 2) != \
                P["rho_abs"]["value"] or S.GRID["rho"][1] != -P["rho_abs"]["value"]:
            out.append("rho: 0.87 is not the declared rule's value")
        ds = _get(CITED[2], ["summary", "yunlin", "median_cv_r2"])
        if float(ds["deseasonalised:all_drivers"]) != k["r2_deseasonalised_all_drivers"] or \
                float(ds["deseasonalised:all_minus_canal_w1_cell"]) != \
                k["r2_deseasonalised_all_minus_canal"]:
            out.append("kappa: deseasonalised R2s differ")
        if float(_get(CITED[2], ["summary", "yunlin", "median_cv_r2",
                                 "raw:canal_w1_cell"])) != \
                P["kappa_raw_m"]["inputs"]["canal_r2_raw"]:
            out.append("kappa_raw: raw canal R2 differs")
        lo, hi = S.corr_bound(-r["census_energy"], r["canal_deliveries"])
        if round(lo, 2) != -S.RHO_LOW or round(hi, 2) != 1.0:
            out.append("rho_low: the bound is not [0.52, 1.00]")
        n2 = (REPO / f"results/dataprep/preregistration/{S.N2_ID}.json").read_text()
        if S.N2_PUBLISHED_TEXT not in n2:
            out.append("N2 power_check text 0.65 / 0.83 not found")
    except (OSError, KeyError, ValueError) as err:
        out.append(f"kappa/rho: {err}")
    try:
        import pandas as pd

        kf = pd.read_csv(REPO / CITED[5], usecols=["sid", "rmse_model_m"], dtype={"sid": str})
        y = kf[kf["sid"].isin(set(S.yunlin_screens()["sid"]))]
        if len(y) != S.N_YUNLIN or y["sid"].duplicated().any() or \
                float(y["rmse_model_m"].mean()) != P["sigma_page_kfold_m"]["value"]:
            out.append("sigma_page_kfold: stage3_kfold_wells.csv Yunlin mean differs")
    except (OSError, KeyError, ValueError) as err:
        out.append(f"sigma_page_kfold: {err}")
    return out


def scorer_problems() -> list[str]:
    """The shared scorer, its binding and the copied reading thresholds."""
    out = []
    if BS.file_sha256() != S.SCORER_SHA:
        out.append("bootstrap_scorer.py is not f8990d61")
    idx = REPO / "results/dataprep/preregistration/INDEX.jsonl"
    rows = {r["id"]: r for r in prereg3.read_chained(idx)} if idx.exists() else {}
    for pid, want in ((S.N2_ID, S.N2_SHA), (S.N2C_READING_ID, S.N2C_READING_SHA)):
        if (rows.get(pid) or {}).get("sha256") != want:
            out.append(f"{pid}: INDEX sha256 is not {want[:8]}")
    try:
        n2 = json.loads((REPO / f"results/dataprep/preregistration/{S.N2_ID}.json")
                        .read_text())
        rc = json.loads((REPO / f"results/dataprep/preregistration/{S.N2C_READING_ID}.json")
                        .read_text())
    except OSError as err:
        return out + [str(err)]
    if n2["primary_metric"]["interval"] != ("90 % from a 5 km block cluster bootstrap, B "
                                            "2000, seed 20261101"):
        out.append("N2: interval wording changed")
    if n2["decision_rule"]["not_supported"] != "beta 90 % interval includes 0, or beta <= 0":
        out.append("N2: decision rule wording changed")
    if n2.get("thresholds") != {"informative_head_m": BS.INFORMATIVE_HEAD_M,
                                "level": BS.LEVEL, "min_clusters": BS.MIN_CLUSTERS,
                                "min_months": BS.MIN_MONTHS, "ratio_band": BS.RATIO_BAND}:
        out.append("N2: thresholds differ from the scorer")
    if int(n2["tests"][0]["seed"]) != BS.SEED:
        out.append("N2: bootstrap seed differs from the scorer")
    if (rc.get("code_sha256") or {}).get(S.SCORER) != S.SCORER_SHA:
        out.append(f"{S.N2C_READING_ID} did not bind f8990d61")
    r = rc["thresholds"]["reading"]
    if {k: r[k] for k in S.READING} != S.READING:
        out.append(f"READING differs from {S.N2C_READING_ID} thresholds.reading")
    if S.READING != S.SC.READING:
        out.append("READING differs from synth_n2c.READING")
    if sorted(S.yunlin_screens()["sid"].tolist()) != sorted(
            set(S.n2_yunlin_ids()) & set(S.yunlin_screens()["sid"].tolist())) or \
            len(S.yunlin_screens()["sid"]) != S.N_YUNLIN:
        out.append("the Yunlin screens are not 84 of N2's wells")
    return out


def runner_problems() -> list[str]:
    sh = (REPO / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    dry = subprocess.run(["bash", str(REPO / RUNNER), "forward"], cwd=REPO,
                         env={"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
                         capture_output=True, text=True)
    if dry.stdout.splitlines() != [command()]:
        out.append("runner DRY=1 forward line differs from the declared command")
    p = REPO / f"results/dataprep/preregistration/{S.N2_ID}.json"
    if p.exists():
        n2 = json.loads(p.read_text())["code"]["forward_command"]
        seg = n2.split(" -m hydrophysics.twin.forward ")[1].split(" --recorded-pumping")[0]
        if seg not in command():
            out.append("the forward's flags differ from N2's registered forward_command")
        if json.loads(p.read_text())["code"]["commit"] != N2_CODE:
            out.append(f"N2's code commit is not {N2_CODE}")
    return out


def gate_record_problems(rec: dict | None) -> list[str]:
    """The recorded engineering gate passed and is bound to the code now (review S5)."""
    if rec is None or not rec.get("ok"):
        return ["the pre-registration engineering gate is not recorded as passed"]
    now = S.gate_code_sha()
    bad = sorted(k for k, v in now.items() if (rec.get("harness_sha256") or {}).get(k) != v)
    if bad or rec.get("scorer_sha256") != S.SCORER_SHA:
        return [f"the engineering gate record is bound to other code: {bad or [S.SCORER]} "
                "(rerun synth_n2.py engineering and update ENGINEERING_GATE_RECORD)"]
    return []


def readiness(d: dict) -> list[str]:
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
    rep = d["replays"]["content"]
    if rep is None:
        out.append(f"replays not built: bash {RUNNER} replays")
    else:
        for r in rep["replays"].values():
            if sha(REPO / S.REPLAYS / r["file"]) != r["sha256"]:
                out.append(f"replay file {r['file']} differs from replays.json")
    out += gate_record_problems(d["thresholds"]["engineering_gate_record"])
    if (REPO / f"results/dataprep/preregistration/{ID}.json").exists():
        out.append(f"{ID} is already registered")
    for p in (S.FWD_STEM + ".npz", S.PANEL, S.REPORT):
        if (REPO / p).exists():
            out.append(f"{p} exists before the declaration")
    if dt.datetime.now(dt.timezone.utc) >= DEADLINE:
        out.append("past the round-8 deadline 2026-11-20")
    wt = Path.home() / "HydroPhysicsAI-n2rd-1fff145"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return out + [f"worktree {WT_TXT}: {type(err).__name__} (create it: git worktree add "
                      f"--detach {WT_TXT} {N2_CODE})"]
    if head != N2_CODE:
        out.append(f"worktree {WT_TXT} is at {head}, not {N2_CODE}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="dry run: print, check, write nothing")
    g.add_argument("--write", action="store_true", help="register once (the lead only)")
    ap.add_argument("--seed", type=int, required=True, help="the study seed (no default)")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --write")
    ap.add_argument("--quiet", action="store_true", help="do not print the document")
    a = ap.parse_args(argv)
    d = draft_prereg(a.seed)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = runner_problems() + published_problems() + scorer_problems()
    print("binding, published-field and scorer checks:", probs or "ok")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = sha(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    bad = bool(errs or probs or pending)
    if a.validate:
        print("valid; not written" if not bad else "NOT READY: fix the problems above")
        return 1 if bad else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if bad:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path.relative_to(REPO)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
