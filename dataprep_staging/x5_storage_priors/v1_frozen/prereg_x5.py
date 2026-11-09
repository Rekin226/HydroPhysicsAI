"""Declare X5.storage_priors.v1 (harness.prereg3, kind 'declaration') and log X5's looks.

Route 3 of the 2026-10-25 audit plan (Q7): Ss per deep screen from earth tides and from
barometric efficiency on 2017-18 hourly heads, a cross-method agreement gate, coastal ocean-
loading flags, a per-layer/zone prior table, and the stage-2 column rung specification.

Why a declaration and not a test (checked before writing): prereg3 finds no blocking look on
the 211 wells x 2017-18, but (1) every one of them was inside the H1a barometric fit
(L1_clean v3, fit window 2012-01..2022-12) and the P5 monthly tidal fits (head_signatures v1,
period 'pre' 2012-22), both logged with unverifiable membership, and (2) the 2026-10-25
audit computed this very cross-method relation (annual M2 fits 2015/2021/2024 against the
L1 v3 BE; Spearman 0.54, inland 0.45) on 172 / 50 of these wells, unlogged, and the 0.4
threshold was set after it. Ss is a stationary well property (the audit's M2 2021/2015 ratio
is 0.94), so fresh years give fresh noise, not a blind test. Label: design-time.

Every python call that writes the registry runs under the shared lock:
  flock results/dataprep/preregistration/.x5.lock python -I prereg_x5.py --write
  flock ... python -I prereg_x5.py --seed-audit      (the audit's looks, back-dated seeds)
  flock ... python -I prereg_x5.py --look            (X5's own 2017-18 look, before run)
  flock ... python -I prereg_x5.py --ledger          (ledger rows of the reported numbers)
Without a flag: validate and dry-run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5sp_const as K  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
AUDIT = Path("/tmp/claude-1002/-home-abdoul-HydroPhysicsAI/"
             "412eaaa9-2a90-4b16-aa86-d03d5f6c067c/scratchpad/audit2026-10-25")
SEL = K.WORK / "selection.csv"
UNIT = "X5:heads_2017-18_layers2-4"
Q0_COLUMN = REPO / "results/twin_runs/stage3_datum_gate_q0/coupled_leveling/vep_zonal_leveling.json"

STAGE2 = {
    "status": "specification only; run later by the lead through the GPU batch runner; "
              "registered by the lead as its own pre-registration before it runs",
    "precondition": "x5_storage_priors/v1 MANIFEST gate.passed is true (synthetic S1-S6 all "
                    "pass AND the primary Spearman gate PASS). Otherwise the rung is not run.",
    "rung_C1_primary": {
        "what": "Q0 flow model unchanged (results/twin_runs/stage3_datum_gate_q0/"
                "stage3_theta.json); only the leveling column is refitted with a physical "
                "floor on its elastic skeletal storage: --ske-min = the global floor of "
                "x5_storage_priors/v1/column_ske_floor.json (the smallest zonal floor, "
                "median aquifer Ss_k of layers 2-4 x 300 m). Existing flag; no code change.",
        "command": "python -m hydrophysics.twin.calibrate_coupled --theta "
                   "results/twin_runs/stage3_datum_gate_q0/stage3_theta.json --target leveling "
                   "--configs zonal --hpc0-guard-days 365 --seed 2027 --device cuda "
                   "--ske-min <global floor> --out results/twin_runs/stage3_datum_gate_q0_x5c1/"
                   "coupled_leveling; then twin.forward as Q0's job_forward with --vep-json "
                   "of the new column (out results/twin_forward/datum_gate_q0_x5c1)",
        "why": "A0 (TWIN_EVIDENCE 14): proximal Ske/Skv are not identifiable; Q0's fitted "
               "proximal Ske is 7.8e-5 (read before this declaration; mid 8.7e-3, distal "
               "2.0e-2), below any aquifer-only floor of order Ss_k 1e-6..1e-5 /m x 300 m. "
               "The floor is a lower bound only: aquitards (not sampled by screens) add "
               "elastic storage, so it may bind only where the leveling fit went below the "
               "aquifers' own storage.",
    },
    "rung_C2_declared_alternative": {
        "what": "per-zone floors (column_ske_floor.json zones) instead of the global floor; "
                "needs an opt-in flag --ske-min-json (one floor per column zone; default "
                "bit-identical; CPU tests) written by the lead before the batch",
        "rule": "reported beside C1, never substituted for it",
    },
    "gate_0_synthetic_before_real": "M1's declared synthetic recovery (column fitted to "
        "leveling made from a known truth; avoided subsidence within [0.67, 1.5] in every zone "
        "at 1, 3 and 7 years, head damping k 1 and 2) run with the floor; a truth whose Ske "
        "lies above the floor must be recovered as without it (avoided-subsidence ratio change "
        "<= 5 %). Fail -> the real rung is not run.",
    "score_against_Q0": {
        "primary": "leveling out-of-fold R2 (calibrate_coupled --target leveling, site folds "
                   "--seed 2027, the G2 partition) >= Q0 - 0.02 (Q0 0.653)",
        "also_gating": "forward plausibility as I1.B.v1: county-mean VEP subsidence rate "
                       "2023-01..2025-12 >= 0 cm/yr in Changhua and in the rest of the fan "
                       "(county map v2), model-internal",
        "reported": "r2_outoffold_by_zone (proximal especially), fitted Ske/Skv per zone and "
                    "whether the floor binds, the 2023-32 cut30 avoided subsidence vs Q0, and "
                    "A0's identifiability rule re-applied to the column (sd/|mean| <= 0.5)",
        "label": "design-time: the seed-2027 leveling partition was scored before (I1.B.v1), "
                 "so a pass cannot promote alone; promotion needs a confirmation on a fresh "
                 "partition (seed 2029) registered before it runs",
    },
    "not_in_this_rung": "flow-model storativity priors (log_S per layer and zone, for A0's "
                        "Changhua T/S problem) need layer thicknesses the model does not "
                        "carry; the per-layer Ss table is delivered for that later use",
}


def selection_ids() -> list[str]:
    return sorted(pd.read_csv(SEL, dtype={"sid": str})["sid"].tolist())


def doc() -> dict:
    code = [K.CODE_DIR / f for f in ("x5sp_const.py", "x5sp_lib.py", "synthetic_check.py",
                                     "run_x5.py", "prereg_x5.py")]
    est = [REPO / "dataprep_staging/h1_heads_qc/baro.py",
           REPO / "dataprep_staging/h1_heads_qc/config.py"]
    data = [SEL, K.META, K.PRESSURE, K.HEADS_DAILY, Q0_COLUMN]
    return {
        "id": K.PREREG_ID, "wp": K.WP, "kind": "declaration",
        "title": "X5 route 3: independent storage priors (Ss per deep screen from M2/O1 earth "
                 "tides and from barometric efficiency, 2017-18 hourly heads); design-time",
        "tests": NA("declaration: design-time consistency gate, never validation (see "
                    "'why_not_a_test')"),
        "why_not_a_test": __doc__.split("Why a declaration and not a test")[1].split(
            "Every python call")[0].strip(),
        "data_scope": K.DATA_SCOPE,
        "external_units": {UNIT: {
            "description": "AMP hourly heads 2017-01..2018-12 of the 211 layer 2-4 screens "
                           "(L1 v3 pre-origin table, layer_flag 0, with an AMP file and "
                           "coordinates) read by X5 route 3",
            "ids": selection_ids(), "window": "2017-01..2018-12"}},
        "checks": {
            "synthetic_first": {
                "what": "synthetic_check.py: 48 inland + 12 coastal synthetic wells (seed "
                        f"{K.SYN_SEED}), Ss log-uniform {K.SYN_SS_RANGE}, BE = rho g n beta_w "
                        "/ Ss at n 0.3, eight-constituent earth tide, real ERA5 2017-18 "
                        f"pressure of cell {K.SYN_PRESSURE_CELL}, pumping cycles, red noise, "
                        "spikes, sentinels, gaps, 1 cm quantisation; coastal wells add an "
                        "ocean M2 term (1-3 x the earth-tide M2, phase within 60 deg)",
                "criteria": K.SYN_CRITERIA,
                "definitions": "S1: |log10(Ss_M2/Ss_true)| median and p90 over inland wells "
                               "with M2 SNR >= 3. S2: detection rate where true M2 >= 3 mm. "
                               "S3: same for Ss_BE over accepted BE, acceptance rate where "
                               "true BE >= 0.1. S4: Spearman(Ss_M2, Ss_BE) over inland wells "
                               "passing both. S5: ocean flag (gain M2/O1 > 1.3, O1 detected) "
                               "rates, coastal and inland. S6: median |eta_M2| (deg).",
                "rule": "all six must pass before any real 2017-18 head is read; a fail stops "
                        "the real-data part of v1 (a revised estimator needs a new "
                        "declaration X5.storage_priors.v2 before any real look). Engineering "
                        "retries (crash, NaN) are free before a metric is seen; one smoke "
                        "run on one well with a non-declared seed, printing no metric, was "
                        "made before this declaration."},
            "gate_primary": {
                "estimators": "Ss_M2 = c_nu / gain_M2 (gain = head per unit theoretical "
                              "areal strain, block-projected Huber fit of Q1, O1, N2, M2 with "
                              "15-day nuisance blocks: quadratic + local-clock S1..S4; "
                              "c_nu = (1-2nu)/(1-nu), nu 0.25). Ss_BE = rho g n beta_w / "
                              "BE_long (production hourly_grouped barometric design of "
                              "h1_heads_qc.baro refitted on 2017-18 only; BE = -A_K at 48 h; "
                              "n 0.30, beta_w 4.6e-10 /Pa).",
                "wells": "layer_code 3-4, layer_flag 0, x_twd97 > 182 km (twin mid/distal "
                         "line); M2 SNR >= 3 (bootstrap se, 15-day blocks); BE status "
                         "accepted, A_K < 0, 0.02 <= BE <= 1",
                "statistic": "Spearman(Ss_M2, Ss_BE)", "threshold": K.SPEARMAN_MIN,
                "n_min": K.N_MIN,
                "verdict": "PASS if n >= 10 and rho >= 0.4; FAIL if n >= 10 and rho < 0.4; "
                           "NOT_EVALUABLE if n < 10. Reported with a well-bootstrap 90 % "
                           "interval and a one-sided permutation p (not gating)."},
            "secondary_reported_not_gating": [
                "inland deep without ocean-flagged wells", "Ss_O1 vs Ss_BE inland deep",
                "inland layers 2-4", "all deep incl. coastal", "coastal deep"],
            "coastal_ocean_flag": "per well: ocean if gain_M2/gain_O1 > 1.3 with M2 and O1 "
                                  "detected; earth if <= 1.3; else undetermined. Per zone: "
                                  "flag counts, median gain ratio, median Ss_M2/Ss_BE.",
            "hsieh_T_secondary": "M2 phase shift eta = 180 - theta (negative = lag; hourly "
                                 "values taken as means over [hh, hh+1), effective hh:30, "
                                 "+-14.5 deg if instead instantaneous). eta < -2 se: T by "
                                 "Hsieh et al. 1987 (r_w = r_c = 0.0762 m, S = Ss x 50 m); "
                                 "|eta| <= 2 se: T lower bound at eta = -2 se; eta > 2 se: "
                                 "phase lead, model not applicable. Reported, not gating.",
            "prior_table": "per well: Ss_well = sqrt(Ss_M2 Ss_BE) if both and M2 is usable "
                           "(earth, or undetermined and inland), Ss_M2 if only M2 usable, "
                           "Ss_BE if BE ok, else none. Ss_k = Ss_well - rho g n beta_w, "
                           "floored at 0.1 Ss_well (flagged). Groups: layer x zone, layer x "
                           "county, layer, zone (layers 2-4 only; layer 1 is unconfined, Sy "
                           "governs); >= 3 wells else not estimated; median of log10, well-"
                           "bootstrap se and 90 % CI of the median, between-well sd; "
                           "systematic log10 term = quadrature of half the porosity range "
                           "(0.2-0.4) and of the Poisson range (0.2-0.3) (each halved: the "
                           "geometric mean carries half) and half the median inland "
                           "log10(Ss_M2/Ss_BE); prior sd = sqrt(se^2 + sys^2). Column Ske "
                           "floor per zone = median Ss_k (layers 2-4) x 300 m.",
            "alternatives": {
                "A1_earthquake_steps": "exploration only (Chishang 2022-09, Hualien 2024; "
                                       "task ruling). Not an Ss estimator without the static "
                                       "strain at each well; not run in v1.",
                "A2_amp_c_tuku": "1-cpd gain of the 130 m Tuku pile (TKJS_282_29) on head "
                                 "09050321, Mar-May 2017 and 2018, 5-day blocks; S_pile = "
                                 "gain (mm/m) / 1000 vs the well's Ss_k x 130 m. Design-time "
                                 "(the audit read Mar-May 2017-22 at this pair); reported."},
            "U7": "no 2023+ value enters anything here"},
        "stage2_rung_spec": STAGE2,
        "zones": "twin zones by easting: distal <= 182 km < mid <= 205 km < proximal",
        "priors": NA("this declaration produces priors; it uses none"),
        "burn_in": NA("no model fit"), "folds": NA("no folds"),
        "fresh_units": NA("design-time: no fresh units (see why_not_a_test)"),
        "thresholds": {"spearman_min": K.SPEARMAN_MIN, "n_min": K.N_MIN,
                       "snr_min": K.SNR_MIN, "ocean_ratio": K.OCEAN_RATIO,
                       "be_range": [K.BE_MIN, K.BE_MAX], "synthetic": K.SYN_CRITERIA},
        "gate_scores": NA("the gate outcome is written once to results/dataprep/reports/"
                          "X5.storage_priors.v1/report.json"),
        "primary_model": NA("no model; estimators as in checks.gate_primary"),
        "attempt_policy": "the real-data run and the report run once (each output refuses to "
                          "overwrite); engineering retries before a metric is seen are free "
                          "and logged in report.json; at most one exploratory follow-up "
                          "after a failure, labelled 'post-failure, exploratory; never "
                          "validation; cannot promote'",
        "inputs": [rel_to_repo(p) for p in code + est + data],
        "interpretation": "PASS: the two independent physics routes rank the inland deep "
                          "screens' storage alike on 2017-18, so the combined Ss table may "
                          "serve as a prior (design-time evidence; the relation was previewed "
                          "by the audit). FAIL or NOT_EVALUABLE: the routes do not agree "
                          "well enough at the screen level; the table is still written with "
                          "gate.passed false, so the loader refuses its 'prior' use and the "
                          "stage-2 rung is not run.",
        "no_prior_look_statement": "Before this declaration X5 read: station metadata; the "
                                   "Q0 column JSON (model parameters); AMP timestamps and "
                                   "value resolution of 07010131 and 09050321 for March 2013 "
                                   "(format check, 1 cm quantum); the audit report's "
                                   "aggregate numbers; the sid/year columns of the audit's "
                                   "look files. No 2017-18 head value.",
        "written_by": rel_to_repo(Path(__file__)),
    }


def audit_seeds() -> list[dict]:
    t = pd.read_csv(AUDIT / "heads/tidal_lsq.csv", usecols=["sid"], dtype={"sid": str})
    b = pd.read_csv(AUDIT / "lit-subsidence/be_storage.csv", usecols=["sid"],
                    dtype={"sid": str})

    def mtime(p):
        return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    win_t = "2015-01..2015-12,2021-01..2021-12,2024-01..2024-12"
    return [
        {"unit": "X5:audit20261025_tidal_lsq_wells", "scope": f"heads:{win_t}",
         "time": mtime(AUDIT / "heads/tidal_lsq.csv"),
         "members": {"ids": sorted(set(t["sid"])), "window": win_t,
                     "description": "wells of the audit's annual tidal LSQ fits"},
         "note": "2026-10-25 audit (heads agent, scratchpad heads/tides.py): annual LSQ tidal "
                 "fits 2015, 2021, 2024 incl. M2/O1 Ss; logged by X5 (it was not logged)"},
        {"unit": "X5:audit20261025_be_storage_wells", "scope": "heads:2012-01..2022-12",
         "time": mtime(AUDIT / "lit-subsidence/be_storage.csv"),
         "members": {"ids": sorted(set(b["sid"])), "window": "2012-01..2022-12",
                     "description": "wells of the audit's BE-based Ss (L1 v3 BE, fit "
                                    "2012-22) and its Spearman with the tidal Ss"},
         "note": "2026-10-25 audit (lit-subsidence agent, be_storage.csv): Ss from L1 v3 BE "
                 "and Spearman 0.54 (inland 0.45) with tidal Ss; logged by X5"},
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--seed-audit", action="store_true")
    g.add_argument("--look", action="store_true")
    g.add_argument("--ledger", action="store_true")
    a = ap.parse_args(argv)
    if a.seed_audit:
        for s in audit_seeds():
            r = prereg3.record_look([s["unit"]], s["scope"], K.WP, prereg3.SEED_PURPOSE,
                                    members={s["unit"]: s["members"]}, seeded_time=s["time"],
                                    note=s["note"])
            print("seeded", s["unit"], r.get("time"))
        return 0
    if a.look:
        r = prereg3.record_look(
            [UNIT], K.DATA_SCOPE, K.WP, "design_time_estimation", prereg_id=K.PREREG_ID,
            note="X5 route 3 real-data run (run_x5.py run/report): earth-tide and barometric "
                 "fits of 2017-18 hourly heads at the 211 screens, after the synthetic check")
        print("look", r.get("time"))
        return 0
    if a.ledger:
        rep = K.REPORT_DIR / "report.json"
        pv = K.PRIORS_V1
        num = f"{K.WP}.storage_priors.v1"
        rows = [
            {"number_id": f"{num}.gate_spearman", "source": rel_to_repo(K.WELLS_DIR),
             "role": "design_time_evidence", "product_version": "AMP_V2 wells 2017-18",
             "note": "Ss_M2 and Ss_BE inputs"},
            {"number_id": f"{num}.gate_spearman", "source": rel_to_repo(K.PRESSURE),
             "role": "input", "product_version": "heads/L1_clean/v3", "note": "ERA5 pressure"},
            {"number_id": f"{num}.ss_prior_table", "source": rel_to_repo(K.WELLS_DIR),
             "role": "design_time_evidence", "product_version": "AMP_V2 wells 2017-18",
             "note": "per-well Ss"},
            {"number_id": f"{num}.ss_prior_table", "source": rel_to_repo(pv),
             "role": "prior", "product_version": "x5_storage_priors/v1",
             "note": "per-layer/zone Ss prior table (gate in MANIFEST)"},
            {"number_id": f"{num}.report", "source": rel_to_repo(rep), "role": "input",
             "product_version": K.PREREG_ID, "note": "report written once"},
            {"number_id": f"{num}.synthetic", "source": rel_to_repo(K.SYN_DIR),
             "role": "design_time_evidence", "product_version": K.PREREG_ID,
             "note": "synthetic recovery check"},
        ]
        print(ledger.append_rows(K.WP, rows))
        return 0
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("missing inputs:", missing or "none")
    if not a.write or errs or missing:
        print(json.dumps({k: d[k] for k in ("id", "kind", "data_scope")}, indent=1))
        return 1 if (errs or missing) else 0
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
