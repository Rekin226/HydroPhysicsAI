"""Registration I1.CH_backcast.v1 (harness.prereg3, kind 'test'; round 7 rank 18; user ruling
R7.l1_pre2012), written after the metadata-only availability look and the S0 reproduction
test, BEFORE heads/L1_pre2012 v1 reads any value.

R1c's Changhua-only pumping contrast in 2010-01..2011-12, months no model has fitted: R1c
rolled out from its own month-0 field at 2007-01 (spin-up 2007-09) on census preorigin v3
energy and its 2012-22 recharge climatology; A recorded energy, B Changhua cells at their 2007
monthly pattern from 2008-01; c = A - B at each registered well's cell and layer, e = obs - B.
Primary: the pinned per-well FE slope on rows residualised on well x calendar-month AND
month-of-sample effects (a rain or drought anomaly common to every well, which the 2012-22
recharge climatology leaves in e, is absorbed; the slope is identified from differences
across wells in the 2011 - 2010 change of R1c's contrast), 90 % 5 km block interval;
DIRECTION SUPPORTED iff the informative gate on the transformed contrast passes and the
interval lies above 0. Report only: the well x calendar-month-only slope, the per-well
detrended slope and the synthetic power on the panel. Label: design-time, high-frequency
band previously read.

    python -I dataprep_staging/ch_x/prereg_bc.py --validate --quiet
    python -I dataprep_staging/ch_x/prereg_bc.py --write --quiet --expect-self-sha <sha256>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import score_bc as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{K.X}/prereg_bc.py"
TESTS_FILE = f"{K.X}/tests/test_backcast.py"
RUNNER = f"{K.X}/run_backcast.sh"
RUNNER_LINES = (f"ID={K.BC_ID}", f"WT_HEAD={K.WT_HEAD}", f'WT="$HOME/{K.WT_NAME}"',
                f"BC={K.BC_DIR}")
X5EV = ("X5.energy_volume.c_vs_a.v1", "X5EV:confined_ls_cache_2010_2011",
        "X5EV:unconfined_ls_cache_2010_2011_reported")


def frozen_code() -> dict:
    return {p: IO.sha_or_missing(CODE / p) for p in S.FROZEN + (THIS, TESTS_FILE)}


def pregates() -> dict:
    out = {}
    for key, path in (("availability", S.AVAIL), ("s0", S.S0)):
        p = IO.rp(path)
        if not p.exists():
            out[key] = {"path": path, "sha256": "missing", "passed": False}
            continue
        r = json.loads(p.read_text())
        passed = (r["feasibility"]["passed"] if key == "availability" else r["passed"])
        out[key] = {"path": path, "sha256": IO.sha256(p), "passed": bool(passed)}
        if key == "availability":
            out[key]["eligible_wells"] = r["feasibility"]["eligible_wells"]
            out[key]["nest_excluded"] = r["nest_excluded"]
        else:
            out[key]["max_abs_diff_m"] = r["max_abs_diff_m"]
            out[key]["swap_gate"] = r.get("swap_gate")
    return out


def doc() -> dict:
    pg = pregates()
    ids = pg["availability"].get("eligible_wells", [])
    fz = frozen_code()
    unit = {"description": "Changhua monitoring wells with ls_cache records giving >= 18 "
                           "monthly values in 2010-01..2011-12 by the registered availability "
                           "rule, ring-nest wells excluded; heads/L1_pre2012 v1 (validator, "
                           "design_time_evidence, initial_conditions; local only); "
                           "design-time, the 1-cpd band of the confined members was read by "
                           "X5EV", "ids": ids, "window": "2010-01..2011-12"}
    test = {
        "test_id": K.BC_TEST, "data_scope": K.BC_SCOPE, "fresh_units": [K.BC_UNIT],
        "external_units": {K.BC_UNIT: unit},
        "label": "design-time, high-frequency band previously read",
        "what": "Does R1c's Changhua-only pumping contrast point the right way in 2010-11, "
                "months no model has fitted?",
        "estimand": "beta = pinned per-well FE slope (bootstrap_scorer.beta_response) of e~ "
                    "on c~, e = obs - B, c = A - B, both residualised on well x calendar-month "
                    "and month-of-sample effects (exact least squares, chx_lib.fe_basis form "
                    f"{K.BC_PRIMARY_FORM!r}); rows flags 0-1; wells with >= 18 scored months; "
                    "identified from differences across wells in the 2011 - 2010 change of "
                    "the contrast, per calendar month",
        "interval": "90 %, 5 km block bootstrap (B 2000, seed 20261101)",
        "informative_gate": "bootstrap_scorer.informative_gate on the TRANSFORMED contrast c~ "
                            "at the scored wells x 24 months (RMS of the well-demeaned c~ >= "
                            "0.05 m, >= 5 blocks): the variation that identifies the slope",
        "decision_rule": "not_evaluable if fewer than 10 wells keep >= 18 scored months after "
                         "QC; not_informative if the gate fails; DIRECTION SUPPORTED iff lo > "
                         "0 and beta > 0; else not supported. decide()'s verdict is reported "
                         "beside it",
        "report_only": ["the well x calendar-month-only slope (the v0 form: identified only "
                        "from the common 2011 - 2010 difference, which a drought year "
                        "confounds)", "the per-well detrended slope",
                        f"synthetic power on the panel (chx_lib.bc_power: k = 0 and 1, rho "
                        f"-0.83, G0's noise and heterogeneous loading, {K.BC_POWER_REPS} "
                        f"replicates, seed {K.BC_POWER_SEED}, every well x month observed), "
                        "computed from model output before any head"],
        "never_substituted": "no secondary replaces the primary"}
    return {
        "id": K.BC_ID, "wp": K.WP, "kind": "test",
        "title": "I1.CH_backcast.v1: R1c's Changhua pumping contrast in 2010-11, months no "
                 "model has fitted (design-time)",
        "priors": "Every model starts in 2012-01; no monthly-anomaly estimand of any model "
                  "exists on 2010-11 Changhua heads. Pass chance about 0.15-0.2 (feasibility "
                  "about 0.5 x direction about 0.3).",
        "rulings": {K.RULING_L1PRE: f"{K.RULINGS}: governed local product L1_pre2012 v1 from "
                                    "ls_cache, never committed, for this backcast, labelled "
                                    "design-time (approved)"},
        "tests": [test], "fresh_units": [K.BC_UNIT],
        "zones": "each well at its own forward-grid cell and twin layer (layer_code - 1); "
                 "Changhua cells = R1c's county map",
        "burn_in": "R1c's own month-0 field (2012-01) placed at 2007-01; spin-up "
                   "2007-01..2009-12 (36 months); scored 2010-01..2011-12",
        "folds": NA("temporal holdout before every fit; no folds"),
        "gate_scores": NA("declared gates; no weighted score"),
        "primary_model": "R1c (stage3_datum_gate_ccz, main parameter set)",
        "recipe": {
            "theta": K.R1C_THETA, "theta_sha256": IO.sha_or_missing(IO.rp(K.R1C_THETA)),
            "worktree_head": K.WT_HEAD,
            "forcing": {"energy": "pump_electricity preorigin v3 cells_by_class value, all "
                                  "classes (forcing_frozen; census_L1 v3 forbids "
                                  "forcing_frozen, model_census v1 starts in 2012)",
                        "recharge": "R1c's own 2012-22 recharge field as a calendar-month "
                                    "climatology (no held rain product with forcing_frozen "
                                    "covers 2010-11: rain_check v1 forbids it) + the month "
                                    "fixed effect in the estimator",
                        "canal_percolation": "none (R1c carries no canal field)",
                        "public_supply": "none (R1c carries no extra abstraction)"},
            "arms": {"A": "recorded energy 2007-01..2011-12",
                     "B": f"Changhua cells at their {K.BC_PATTERN_YEAR} monthly pattern from "
                          f"{K.BC_PATTERN_YEAR + 1}-01, other cells recorded"},
            "order": ["availability look (logged; metadata only)", "S0 (model only)",
                      "THIS REGISTRATION", "product_build look logged", "L1_pre2012 v1 "
                      "build", "backcast hindcast (model only)", "score once"],
            "t0_refactor": "inputs.py/heads.py are not edited: hindcast.py assembles the "
                           "pre-2012 forcing itself (bc_forcing) and rolls R1c out from its "
                           "month-0 field (run_from). S0 (pregates.s0): (a) that path equals "
                           "the native 2012-22 run to 0.01 m (a plumbing identity; the 2007 "
                           "placement has no reference run, the 36-month spin-up is the "
                           "remedy); (b) the declared forcing-swap gate: census v3 energy in "
                           "place of model_census v1 on 2012-22 keeps Changhua energy within "
                           f"{list(K.S0_SWAP_ENERGY_BAND)} and R1c's Changhua-only contrast "
                           f"(per-cell-demeaned pooled correlation >= {K.S0_SWAP_CORR_MIN}, "
                           f"RMS ratio in {list(K.S0_SWAP_RMS_BAND)})"},
        "pregates": pg,
        "thresholds": {"min_wells": K.BC_MIN_WELLS, "min_months": K.BC_MIN_MONTHS,
                       "s0_tol_m": K.S0_TOL_M,
                       "s0_swap": {"energy_band": list(K.S0_SWAP_ENERGY_BAND),
                                   "corr_min": K.S0_SWAP_CORR_MIN,
                                   "rms_band": list(K.S0_SWAP_RMS_BAND)},
                       "day_min_hours": K.BC_DAY_MIN_HOURS,
                       "month_min_days": K.BC_MONTH_MIN_DAYS, "bounds": "inclusive"},
        "frozen_code": fz,
        "prior_looks_disclosed": {
            "x5ev": f"{X5EV[0]} loaded 2010-11 hourly series of the confined wells "
                    f"({X5EV[1]}) for a 1-cpd tidal-band estimand and reported its "
                    f"unconfined companion unit ({X5EV[2]}); every known 2010-11 Changhua "
                    "ls_cache well is a member of one of them",
            "print_2010_01_01": "three hourly values of 2010-01-01 were printed in an earlier "
                                "session (disclosed in X5EV)",
            "mechanism_py": "mechanism.py's long-heads option read monthly 2010-25 anomalies "
                            "of ls_cache wells within 1000 m of the ring sites; those wells "
                            "are excluded from this unit (pregates.availability.nest_excluded)",
            "availability_look": f"{K.BC_AVAIL_UNIT}: timestamps only (logged before this "
                                 "registration)",
            "models": "R1c was fitted on 2012-22; no model has been run before 2012"},
        "evidence_and_ledger_mapping": {
            "direction_supported": "new E-row 'direction of R1c's Changhua pumping response "
                                   "supported in years never used to fit any model (2010-11, "
                                   "design-time)' plus C20/C21 wording; C13a unchanged",
            "not_supported": "E-row 'not supported (2010-11, design-time)'; C20/C21 narrowed",
            "not_informative": "recorded; no page change",
            "not_evaluable": "closes U-F honestly in the ledger ('not evaluable')",
            "page_ruling": "E-row and C20/C21 wording need the user's approval (requested in "
                           "round 7, not in the sixth set)",
            "forecast_part": "the page model and Q0 against CL are not computed by v1 "
                             "(report-only in the plan; no 2010-11 CL was frozen); disclosed"},
        "attempt_policy": {"scored_runs": "one", "engineering_retries": "free before any head "
                           "is read; logged", "exploratory_followups_max": 0,
                           "reason": "keep 2010-11 clean for any later use",
                           "thresholds_frozen": True},
        "u7": "every input is pre-origin; the product's fit_window_end is 2011-12",
        "interpretation": "A pass says wells where R1c's Changhua-only contrast changed more "
                          "from 2010 to 2011 also changed more in the observations, outside "
                          "every fitted year; it says nothing about size or the sustained "
                          "lever. Recharge is a 2012-22 climatology, so the real 2010-11 rain "
                          "anomaly stays in e: a dry year raises pumping and lowers heads, "
                          "which biases a slope TOWARD a pass. The month-of-sample effects "
                          "absorb the part common to every well; a rain anomaly whose spatial "
                          "loading tracks R1c's contrast is not excluded.",
        "inputs": ([{"path": p, "sha256": h} for p, h in fz.items() if h != "missing"]
                   + [K.R1C_THETA, S.AVAIL, S.S0, K.RULINGS, f"{K.CENSUS_V3}/MANIFEST.json"]),
        "no_prior_look_statement": "no 2010-11 head value has been read for this test; the "
                                   "L1_pre2012 product does not exist at registration",
        "written_by": THIS,
    }


def readiness(d: dict) -> list[str]:
    out = []
    for key, g in d["pregates"].items():
        if g["sha256"] == "missing":
            out.append(f"pre-gate {key} not run ({g['path']})")
        elif not g["passed"]:
            out.append(f"pre-gate {key} FAILED: do not register (availability: record 'not "
                       "evaluable'; S0: engineering fix)")
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not IO.rp(p).exists() and not (CODE / p).exists():
            out.append(f"input {p} is missing")
    sh = (CODE / RUNNER).read_text() if (CODE / RUNNER).exists() else ""
    out += [f"{RUNNER} lacks {s!r}" for s in RUNNER_LINES if s not in sh.splitlines()]
    if IO.rp(f"results/dataprep/preregistration/{K.BC_ID}.json").exists():
        out.append(f"{K.BC_ID} is already registered")
    if IO.rp(K.L1PRE_V1).exists():
        out.append(f"{K.L1PRE_V1} exists before the registration")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", IO.canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    idx = prereg3.membership_index()
    m = prereg3._declared_members(d["tests"][0]["external_units"][K.BC_UNIT])
    if m is not None and not idx.known(K.BC_UNIT):
        idx.members[idx.canon(K.BC_UNIT)] = m
    block = prereg3.blocking_looks(d["tests"], None, idx=idx,
                                   test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = IO.sha256(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    bad = bool(errs or block or pending)
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
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
