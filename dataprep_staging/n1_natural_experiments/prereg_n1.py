"""Write the N1 pre-registrations (harness.prereg3), before any head value is read.

Three records:

* ``N1.cpu_response.v1`` (test): T1 (2015, 2020, 2021 events at held-out wells), T2 (2021
  public-well pulse) and T3 (2019 public-well step), run on CPU with the P4.0 engine replica
  of the deliverable after this registration.
* ``N1.phase12_response.v1`` (test): G1 (masked 2015 refit), G3 (canal-delivery response, from
  rung R3 on) and G4 (deep piles), run in Phases 1-2 (GPU refits, column) by I1 or N1.
* ``N1.designtime_checks.v1`` (declaration): the in-sample consistency checks and the fold-theta
  check (CPU, design-time labels), the masked 2020 and 2021 refits (design-time: the 2020-22
  hold-out), and the closure-ramp test that waits for located volumes (P6 / R-i).

Before writing, the member inputs of T2/T3 (public-well Jan-May series, drought wells) and the
unit memberships are written to ``results/dataprep/reports/N1_evidence/`` and hashed as
registration inputs, and every earlier look whose members and window intersect an N1 unit is
listed in the record (disclosure; prereg3 does not block looks on undeclared units).

Usage: python -I prereg_n1.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import read_chained  # noqa: E402

from n1_natural_experiments import merge_charts as MC  # noqa: E402
from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import parse_sources as P  # noqa: E402
from n1_natural_experiments import units_n1 as U  # noqa: E402

NA = prereg3.not_applicable
UNITS_JSON = K.EVID / "units_n1.json"
SERIES_CSV = K.EVID / "public_well_janmay_series.csv"
DWELLS_CSV = K.EVID / "drought_wells_yunlin_changhua.csv"
NO_LOOK = ("No head value (observed or simulated) and no fold prediction had been read by "
           "WP N1 when this record was written; the design rests on the plan (events, "
           "windows), the W1 benchmark's reference years and the intervention table (forcing "
           "and report data only).")


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path} exists with different content; not overwritten")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_inputs() -> dict:
    u = U.units()
    _write_once(UNITS_JSON, (json.dumps(u, ensure_ascii=False, indent=1, sort_keys=True)
                             + "\n").encode())
    df, _ = MC.merge()
    s = MC.janmay_series(df)
    _write_once(SERIES_CSV, s.to_csv(index=False).encode())
    dw = pd.DataFrame(P.drought_wells()["rows"])
    dw["cell"] = U.cell_index(dw.x_twd97.to_numpy(), dw.y_twd97.to_numpy())
    _write_once(DWELLS_CSV, dw.to_csv(index=False).encode())
    return u


# --------------------------------------------------------------------------- disclosure
def disclose(units: dict) -> dict:
    """Earlier looks whose members (ids, or the cells of the unit's wells) and window intersect
    each N1 unit; looks on units of unknown membership listed as unverifiable."""
    idx = prereg3.membership_index()
    ext = U.ext_wells()
    cal = U.calibration_wells()
    cells_of = dict(zip(ext.sid, ext.cell, strict=True))
    cells_of.update(dict(zip(cal.sid, U.cell_index(cal.x.to_numpy(), cal.y.to_numpy()),
                             strict=True)))
    looks = read_chained(PREREG_DIR / "looks.jsonl")
    out = {}
    for name, mem in units.items():
        ids = set(mem.get("ids") or [])
        cells = {int(cells_of[i]) for i in ids if i in cells_of}
        win = prereg3.scope_windows(mem.get("window"))
        by_id, by_cell, unknown = {}, {}, {}
        for lk in looks:
            for u in lk.get("units") or []:
                cu = idx.canon(u)
                w = prereg3.windows_intersect(idx.window(cu, lk.get("data_scope")), win)
                if w is not None and not w:
                    continue
                m = idx.members.get(cu)
                key = f"{lk.get('wp')}|{lk.get('purpose')}"
                if m is None:
                    unknown[key] = unknown.get(key, 0) + 1
                elif m.get("universal") or (m["ids"] & ids):
                    by_id[key] = by_id.get(key, 0) + 1
                elif m["cells"] & cells:
                    by_cell[key] = by_cell.get(key, 0) + 1
        out[name] = {"looks_sharing_ids_or_universal": by_id,
                     "looks_sharing_only_cells": by_cell,
                     "looks_unknown_membership": unknown}
    return out


# --------------------------------------------------------------------------- documents
COMMON_MODEL = (
    "M = the P4.0 engine replica (dataprep_staging/p4_changhua/engine.py SparseTwin, engine/v2) "
    "of the deliverable stage3_datum_gate_cwa (batch-8 merged cwa theta, stage3_theta.json), "
    "built with forward.build_model and inputs_p4.load_inputs exactly as engine/v2 was "
    "(Yunlin census energy E, rain recharge R, the twin's initial heads h0 of 2012-01), "
    "stepwise rollout 2012-02..2022-12 with lift feedback. Simulated head of a well = the "
    "head of its active cell (H1a pre-origin coordinates) in twin layer = layer code - 1.")
COMMON_OBS = (
    "heads/L1_clean/v3 heads_monthly.parquet opened through the loader guard as "
    "scoring_target (flags 0-1), months with c20 true, 2013-01..2022-12 only (no 2023+ value "
    "is read).")
COMMON_ANOM = (
    "n1lib.anomalies: per well, least squares of intercept + linear trend + 2 harmonics on the "
    "observed months of the reference years 2013, 2014, 2016, 2017, 2018, 2019, 2022; event "
    "anomaly = value - fit, for the observation and, on the same months, for each model "
    "series with its own fit. Eligibility: >= 3 observed months of the 5 event months, >= 36 "
    "observed reference months in >= 4 reference years.")
COMMON_SCORE = (
    "Per-well error e_X = sqrt(mean over observed event months of (a_X - a_obs)^2); score "
    "RMSE_X = sqrt(mean_w e_X^2) (equal wells). Comparison D = RMSE_M - RMSE_X, paired "
    "site-cluster bootstrap (site_id; calibration wells: first 6 characters of the station "
    "id), B = 2000, seed 20261014 + test index, 90 % percentile interval: 'model_better' if "
    "its upper bound < 0, 'baseline_better' if its lower bound > 0, else 'indistinguishable'; "
    "'not_evaluable' with fewer than 5 sites. Reported with every verdict: n wells, n sites, "
    "both RMSEs, D, the interval, the ratio, and (never gating) the slope through the origin "
    "of the mean observed on the mean modelled event anomaly per well, its sign agreement "
    "and correlation.")
PW_MEMBER = (
    "M+PW = M plus a direct-source member (SparseTwin.rollout q_extra, m3/day, abstraction "
    "negative): Yunlin Taiwan Water public-well pumping placed at the Yunlin Taiwan Water "
    "drought wells of the 2022 inventory (wra111_drought_wells_report.pdf annex table 1-10, "
    "98 listed wells; drought_wells_yunlin_changhua.csv), one equal share per listed well "
    "(wells outside the active grid dropped with their share); the yearly Jan-May volume "
    "V_y (public_well_janmay_series.csv: county series of Taiwan Water, Yunlin; 2020-2021 the "
    "printed values, other years the two-read chart values; 2012-2022 only) gives the rate "
    "V_y / (days of Jan-May of y) applied to every month of year y (primary: constant within "
    "the year); half in twin layer 2, half in layer 3; divided by 2.58, the WRA-to-model "
    "volume ratio of plan P6 step 3 (primary). Sensitivities (reported, never gating): "
    "physical volumes (x 1); Jan-May only (zero June-December); all in layer 2. Assumptions "
    "stated: the 2022 drought wells locate the earlier and the 2021 Taiwan Water pumping; no "
    "volume outside Jan-May is printed.")


def cpu_response_doc(units: dict, disc: dict) -> dict:
    ext = units["N1:ext_wells"]
    deep = units["N1:ext_deep_yunlin"]
    tests = []
    for i, (ev, e) in enumerate(K.EVENTS.items()):
        y = e["year"]
        tests.append({
            "test_id": f"N1.T1_ext_{ev}", "data_scope": "heads:2013-01..2022-12",
            "fresh_units": ["N1:ext_wells"], "external_units": {"N1:ext_wells": ext},
            "what": f"Does the deliverable reproduce the head response to the {e['what']} at "
                    f"wells it was never fitted to, better than climatology and than its own "
                    f"zero-response counterfactual? Event window {y}-01..{y}-05.",
            "model": COMMON_MODEL,
            "zero_response": f"ZR = M with E in {y}-01..{y}-05 replaced cell by cell by the "
                             "mean of E over the reference years for the same calendar month "
                             "(rain unchanged). The deliverable has no canal-delivery term "
                             "(sw_recharge none), so its only response channel to the delivery "
                             "cut is the metered Yunlin pumping; Changhua has no pumping in R0.",
            "climatology": "CL = zero anomaly (the well's own reference trend + season).",
            "observations": COMMON_OBS, "anomaly": COMMON_ANOM, "score": COMMON_SCORE,
            "wells": {"primary": "N1:ext_wells with site_disjoint (no calibration well at the "
                                 "site) and layer_flag 0",
                      "secondary_reported": "all N1:ext_wells (nested at calibration sites; "
                                            "layer_flag 5 layers included)",
                      "strata_reported": "county; twin layer 0-1 vs 2-3"},
            "comparisons": {"primary_1": "M vs CL on the primary wells",
                            "primary_2": "M vs ZR on the primary wells in Yunlin",
                            "zr_informative": "if RMS over the primary Yunlin well-months of "
                                              "(a_M - a_ZR) < 0.05 m, primary_2 is reported "
                                              "'not informative' (no verdict)"},
            "seed": K.BOOT_SEED + i,
            "label": "skill on held-out wells (never fitted; site-disjoint), frozen theta "
                     "in-sample in time: the theta saw the event months at the calibration "
                     "wells, so this is spatial transfer of an event response, not a "
                     "temporal forecast"})
    tests.append({
        "test_id": "N1.T2_pw_pulse2021", "data_scope": "heads:2013-01..2022-12",
        "fresh_units": ["N1:ext_deep_yunlin"], "external_units": {"N1:ext_deep_yunlin": deep},
        "what": "Does adding the Yunlin Taiwan Water public-well pumping (Jan-May 0.49 Mm3 in "
                "2020, 11.93 in 2021, printed) improve the deliverable's 2021 Jan-May head "
                "anomaly at deep wells it was never fitted to?",
        "model": COMMON_MODEL, "member": PW_MEMBER, "observations": COMMON_OBS,
        "anomaly": COMMON_ANOM + " Event window 2021-01..2021-05.", "score": COMMON_SCORE,
        "wells": {"primary": "N1:ext_deep_yunlin (Yunlin, layer code 3-4; nested and "
                             "site-disjoint)",
                  "secondary_reported": "calibration wells in Yunlin with twin layer 2-3 "
                                        "(theta in-sample: consistency check)",
                  "strata_reported": "distance to the nearest in-grid Yunlin Taiwan Water "
                                     "drought well: <= 3, 3-10, > 10 km"},
        "comparisons": {"primary": "M+PW vs M (paired site bootstrap)",
                        "secondary_reported": "M+PW vs CL; M vs CL; the three sensitivities",
                        "informative": "if RMS over the primary well-months of "
                                       "(a_M+PW - a_M) < 0.05 m: 'not informative'"},
        "seed": K.BOOT_SEED + 3,
        "label": "held-out wells for theta; the member is new to the model (frozen theta, "
                 "calibrated without it)"})
    tests.append({
        "test_id": "N1.T3_pw_step2019", "data_scope": "heads:2013-01..2022-12",
        "fresh_units": ["N1:ext_deep_yunlin"], "external_units": {"N1:ext_deep_yunlin": deep},
        "what": "Does the public-well member reproduce the head change across the 2019 Yunlin "
                "Taiwan Water step (Jan-May about 17-21 Mm3 in 2016-2018, 1.7 in 2019, 0.49 "
                "in 2020; Hushan supply, closure programme) at deep wells, better than the "
                "deliverable without it and than the wells' own trend?",
        "model": COMMON_MODEL, "member": PW_MEMBER, "observations": COMMON_OBS,
        "step": "delta = mean of the observed months of Jun-Dec 2019 and Jun-Dec 2020 minus "
                "mean of Jun-Dec 2017 and Jun-Dec 2018 (>= 8 of 14 months on each side); each "
                "model series on the same months. CL0: delta = 0. CL-trend: slope of the "
                "well's reference fit on 2013, 2014, 2016, 2017, 2018 (>= 36 months, >= 4 "
                "years) times the difference of the mean times of the post and pre months.",
        "score": "per-well error |delta_pred - delta_obs|; RMSE over wells; paired site "
                 "bootstrap as in T1 (B 2000, 90 %, >= 5 sites).",
        "wells": {"primary": "N1:ext_deep_yunlin", "secondary_reported": "calibration wells "
                  "in Yunlin with twin layer 2-3 (consistency)",
                  "strata_reported": "distance to the nearest in-grid Taiwan Water drought "
                                     "well"},
        "comparisons": {"primary": "M+PW vs M", "secondary_reported": "M+PW vs CL-trend; "
                        "M+PW vs CL0; M vs CL-trend; the sensitivities",
                        "informative": "if RMS over wells of (delta_M+PW - delta_M) < 0.05 "
                                       "m: 'not informative'"},
        "seed": K.BOOT_SEED + 4,
        "label": "held-out wells for theta; frozen theta; a step inside the calibration "
                 "window (theta may have absorbed part of it through its trend terms)"})
    return {
        "id": "N1.cpu_response.v1", "wp": K.WP, "kind": "test",
        "title": "N1 natural-experiment response tests run on CPU with the P4.0 engine replica "
                 "of the deliverable: spring shortfalls 2015/2020/2021 at held-out wells, "
                 "and the Yunlin public-well step (2019) and pulse (2021)",
        "tests": tests, "zones": NA("no zones: the deliverable's own parameters"),
        "priors": NA("nothing is fitted except the per-well reference fit of n1lib"),
        "burn_in": "2012 is excluded (12-month start-up, plan P4 fact 5): reference years "
                   "start in 2013",
        "folds": NA("held-out wells were never in any fit; no folds"),
        "fresh_units": NA("per test"),
        "thresholds": {"verdict": "90 % paired site-bootstrap interval of D excludes 0",
                       "min_sites": K.MIN_SITES, "informative_rms_m": K.ZR_INFORMATIVE_M,
                       "eligibility": {"event_months": K.MIN_EVENT_MONTHS,
                                       "reference_months": K.MIN_REF_MONTHS,
                                       "reference_years": K.MIN_REF_YEARS}},
        "gate_scores": NA("no gate: every result is reported whichever way it falls (plan N1 "
                          "acceptance); 8 primary comparisons, no multiplicity correction"),
        "primary_model": "R0 = the deliverable (stage3_datum_gate_cwa, batch-8 merged cwa "
                         "theta) through the P4.0 engine replica; M+PW for T2/T3",
        "inputs": [rel_to_repo(x) for x in (
            K.CODE_DIR / "n1lib.py", K.CODE_DIR / "n1const.py", UNITS_JSON, SERIES_CSV,
            DWELLS_CSV, K.TWIN_THETA, U.TWIN_WELLS, K.HEADS_DIR / "heads_monthly.parquet",
            K.HEADS_DIR / "station_meta_preorigin.parquet", K.HEADS_DIR / "MANIFEST.json",
            K.ENGINE_DIR / "MANIFEST.json", K.ENGINE_DIR / "engine_arrays.npz")],
        "interpretation": (
            "T1: 'model_better' against CL at held-out wells means the deliverable transfers "
            "the event response in space; 'baseline_better' means its event anomaly is worse "
            "than assuming no event. Against ZR (Yunlin only) a 'model_better' credits the "
            "metered pumping response; the delivery cut itself has no channel in R0, so a "
            "pass or fail here says nothing about canal recharge (that is G3, from rung R3). "
            "T2/T3: 'model_better' supports adding a located public-well class (P6, rung R6); "
            "'indistinguishable' with an informative member means the heads do not resolve it "
            "at these wells; 'not informative' means the member is too small to matter there. "
            "Nothing here selects or calibrates a model scored on 2023-26 (U3, U7)."),
        "u7": "heads 2013-01..2022-12 only; the public-well series used is 2012-2022",
        "no_prior_look_statement": NO_LOOK,
        "prior_looks_disclosed": {k: disc[k] for k in ("N1:ext_wells", "N1:ext_deep_yunlin")},
        "written_by": rel_to_repo(Path(__file__)),
    }


def phase12_doc(units: dict, disc: dict) -> dict:
    m15 = units["N1:mask_E2015"]
    piles = units["N1:piles"]
    zr_general = ("ZR = the model with every recorded forcing that answers the delivery cut "
                  "(metered pumping energy, in both counties where the recipe has it; canal "
                  "deliveries where the recipe has a delivery term) replaced in the masked "
                  "months by its reference-year mean for the same calendar month, cell by "
                  "cell; rain unchanged.")
    tests = [
        {"test_id": "N1.G1_mask_E2015", "data_scope": "heads:2015-01..2015-05",
         "fresh_units": ["N1:mask_E2015"], "external_units": {"N1:mask_E2015": m15},
         "what": "Masked refit: does the model, fitted without Jan-May 2015, reproduce the "
                 "2015 spring shortfall at the 158 calibration wells better than CL and ZR?",
         "models": "(i) the R0 recipe (batch-8 cwa, stage3_datum_gate_cwa) and (ii) the I1 "
                   "primary model's recipe as pre-registered by I1, each refitted with "
                   "calibrate_flow --mask-months 2015-01:2015-05 (the I1 hook), same seeds "
                   "and epochs as the recipe; heads from the refit's own rollout.",
         "zero_response": zr_general, "observations": COMMON_OBS,
         "anomaly": COMMON_ANOM + " Event window 2015-01..2015-05 (the masked months).",
         "score": COMMON_SCORE + " Sites = first 6 characters of the station id.",
         "comparisons": {"primary_1": "M vs CL on the 158 calibration wells",
                         "primary_2": "M vs ZR on the 158 calibration wells",
                         "zr_informative": "RMS(a_M - a_ZR) < 0.05 m -> not informative"},
         "seed": K.BOOT_SEED + 10,
         "label": "skill: a masked temporal hold-out that no screen, gate or design choice "
                  "has used (it lies outside the 2020-22 hold-out and the 2016-18 fresh "
                  "block)"},
        {"test_id": "N1.G3_sw_E2015", "data_scope": "heads:2015-01..2015-05",
         "fresh_units": ["N1:mask_E2015"], "external_units": {"N1:mask_E2015": m15},
         "what": "Canal-delivery response: in the masked 2015 refit of a recipe with a "
                 "canal-delivery recharge term (rung R3 or later, surface_water_v4 "
                 "preorigin), does the delivery cut itself improve the event anomaly?",
         "models": "each recipe of G1 that carries a delivery term; ZR_sw = the same masked "
                   "refit's rollout with the canal deliveries of 2015-01..2015-05 replaced by "
                   "their reference-year mean (pumping and rain unchanged).",
         "observations": COMMON_OBS, "anomaly": COMMON_ANOM, "score": COMMON_SCORE,
         "comparisons": {"primary": "M vs ZR_sw on the 158 calibration wells",
                         "informative": "RMS(a_M - a_ZR_sw) < 0.05 m -> not informative"},
         "seed": K.BOOT_SEED + 11,
         "label": "skill (masked temporal hold-out); not run if no recipe has a delivery "
                  "term (reported as 'not run')"},
        {"test_id": "N1.G4_piles_2020_2021", "data_scope": "piles:2020-01..2022-05",
         "fresh_units": ["N1:piles"], "external_units": {"N1:piles": piles},
         "what": "Does the compaction column reproduce the deep-benchmark compression of the "
                 "2020 and 2021 springs relative to 2022 at the 7 printed sensors?",
         "models": "the I1 primary model's compaction column (calibrated on leveling <= 2021; "
                   "piles never fitted) driven by that model's hindcast head path, at the "
                   "pile's cell and the sensor's depth interval",
         "observations": "deep_benchmarks_printed/v2 printed_monthly (flag 0) opened as "
                         "scoring_target; Jan-May sums of monthly compression per sensor for "
                         "2020, 2021 and 2022 (all five months verified, else the sensor-year "
                         "is dropped).",
         "metric": "delta_e = C(Jan-May e) - C(Jan-May 2022) for e in {2020, 2021}, per "
                   "sensor; error |delta_mod - delta_obs|; RMSE over sensors; baseline ZR: "
                   "delta = 0.",
         "verdict": "with 4 stations the bootstrap rule of T1 cannot be applied ('not "
                    "evaluable'); reported: RMSE ratio to ZR per event, sign agreement of "
                    "delta over the 14 sensor-events; 'model_better' only if the ratio < 1 "
                    "in both events and >= 10 of 14 signs agree. Uncalibrated, weak evidence.",
         "label": "column out of sample (piles never fitted), heads in sample; 2019-2021 "
                  "printed months are uncertain by several mm (S2)"},
    ]
    return {
        "id": "N1.phase12_response.v1", "wp": K.WP, "kind": "test",
        "title": "N1 natural-experiment response tests for Phases 1-2: masked 2015 refit, "
                 "canal-delivery response and deep piles",
        "tests": tests, "zones": "the recipe's own (R0: --zone-boundaries 205,182,208)",
        "priors": "the recipe's own", "burn_in": "the recipe's own; scores from 2013",
        "folds": NA("masked temporal hold-out; no folds"), "fresh_units": NA("per test"),
        "thresholds": {"verdict": "90 % paired site-bootstrap interval of D excludes 0 "
                                  "(G1, G3); G4 as stated in the test",
                       "min_sites": K.MIN_SITES, "informative_rms_m": K.ZR_INFORMATIVE_M},
        "gate_scores": NA("no gate; reported whichever way they fall"),
        "primary_model": "the I1 primary model (named in I1's own pre-registration before any "
                         "2023-26 re-score); R0 recipe as the reference run",
        "inputs": [rel_to_repo(x) for x in (
            K.CODE_DIR / "n1lib.py", K.CODE_DIR / "n1const.py", UNITS_JSON, K.TWIN_THETA,
            U.TWIN_WELLS, K.HEADS_DIR / "heads_monthly.parquet", K.HEADS_DIR / "MANIFEST.json",
            K.PILES_PRINTED / "printed_monthly.parquet", K.PILES_PRINTED / "MANIFEST.json",
            K.W1_PRE / "MANIFEST.json")],
        "interpretation": (
            "G1 is the plan's skill test of the event response at the calibration wells (year "
            "masked in a refit). G3 isolates the canal-delivery channel. G4 tests the "
            "subsidence response. A failure is reported as a limit of the policy response, "
            "not fixed by re-tuning (a re-design needs a new registration with new test ids)."),
        "u7": "heads and piles through 2022-12 only",
        "no_prior_look_statement": NO_LOOK,
        "prior_looks_disclosed": {k: disc[k] for k in ("N1:mask_E2015", "N1:piles")},
        "written_by": rel_to_repo(Path(__file__)),
    }


def designtime_doc(units: dict, disc: dict) -> dict:
    return {
        "id": "N1.designtime_checks.v1", "wp": K.WP, "kind": "declaration",
        "title": "N1 checks that are not skill tests: in-sample consistency, fold-theta check, "
                 "masked 2020/2021 refits (design-time), and the closure-ramp test that "
                 "waits for located volumes",
        "tests": NA("declaration: these checks are reported with design-time or consistency "
                    "labels, never as validation"),
        "checks": {
            "C1_insample": "the T1, T2 and T3 metrics with the full theta (R0) at the 158 "
                           "calibration wells (T2/T3 on the Yunlin twin-layer 2-3 ones); label "
                           "'consistency check' (theta fitted on these heads, event months "
                           "included). Look logged with record_look, purpose "
                           "'consistency_check'.",
            "C2_fold_theta": "the T1 metric at each calibration well with the theta of the "
                             "batch-8 fold that held it out (stage3_fold_thetas.json, folds "
                             "of stage3_kfold_wells.csv), M and ZR rebuilt per fold with the "
                             "engine; label 'design-time evidence' (the batch's k-fold "
                             "partition gated the deliverable). Purpose 'design_time_check'.",
            "C3_mask_E2020_E2021": "masked refits (calibrate_flow --mask-months "
                                   "2020-01:2020-05 and 2021-01:2021-05) scored as G1; label "
                                   "'design-time evidence': both windows lie in the 2020-22 "
                                   "hold-out (dt.block.2020_2022) and in the I1 temporal screen "
                                   "that selects rungs (--holdout-months 36).",
            "C4_closure_ramp_pending": "the 2011-18 closure ramp beyond the Yunlin Taiwan "
                                       "Water step (T3) has no located volumes: test design "
                                       "waits for the P6 sensitivity member or request R-i and "
                                       "needs its own registration with new test ids."},
        "zones": NA("as the model"), "priors": NA("none"), "burn_in": "2012 excluded",
        "folds": "C2: stage3_kfold_wells.csv (batch 8, seed 0, 5 folds)",
        "fresh_units": NA("no fresh units: consistency and design-time checks"),
        "external_units": {"N1:cal_wells": units["N1:cal_wells"]},
        "thresholds": NA("same verdict rule as the T1 metric, reported with its label"),
        "gate_scores": NA("no gate"), "primary_model": NA("checks of R0"),
        "inputs": [rel_to_repo(x) for x in (K.CODE_DIR / "n1lib.py", K.CODE_DIR / "n1const.py",
                                            UNITS_JSON, K.TWIN_THETA, K.TWIN_FOLD_THETAS,
                                            K.TWIN_KFOLD_WELLS)],
        "interpretation": "context for T1-T3 and G1; never quoted as skill",
        "no_prior_look_statement": NO_LOOK,
        "prior_looks_disclosed": {"N1:cal_wells": disc["N1:cal_wells"]},
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    u = write_inputs()["units"]
    disc = disclose(u)
    docs = [cpu_response_doc(u, disc), phase12_doc(u, disc), designtime_doc(u, disc)]
    for d in docs:
        errs = prereg3.validate_prereg(d)
        if errs:
            raise SystemExit(f"{d['id']}: {errs}")
    rec = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "disclosure": disc, "ids": [d["id"] for d in docs]}
    if args.dry_run:
        print(json.dumps(rec, ensure_ascii=False, indent=1)[:6000])
        return 0
    out = []
    for d in docs:
        path, sha = prereg3.write_prereg(d)
        out.append({"id": d["id"], "path": rel_to_repo(path), "sha256": sha,
                    "created": json.loads(path.read_text())["created"]})
        print(out[-1], flush=True)
    rec["written"] = out
    rec["input_sha256"] = {p: sha256_file(Path(K.EVID.parents[3] / p)) for p in (
        rel_to_repo(UNITS_JSON), rel_to_repo(SERIES_CSV), rel_to_repo(DWELLS_CSV))}
    (K.EVID / "prereg_written.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1)
                                                + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
