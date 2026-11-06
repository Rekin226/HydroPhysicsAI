"""Declaration builder of I1.backcast_2010_11.v1 (harness.prereg3, kind 'test'; round 8 plan
#12; user rulings R8.6a-c). The builder DRAFTS; the lead registers.

The registration comes AFTER the logged availability look (done 2026-11-06, PASS, origin
2010-12 only), the logged product_build look and the L1_pre2012 v2 build, E0 (GPU), the swap
gate, the backcast panel, the 2012-22 baselines and the model-only G0 studies, and BEFORE any
2010-11 head is compared with anything. Every one of those files is frozen here by sha256;
the G0 results set each gate's role (gating / report-only) at registration, never after.

One declaration, four tests on two brand-new fresh units (I1BC11T:heads_2010_2011_yunlin and
_changhua; window 2010-01..2011-12):

* I1BC11T.G1_stat_vs_cl_2011 (primary): E04's recipe refitted on 2012-22, origin 2010-12,
  2011 scored. PASS iff the equal-station RMSE ratio <= 1.0 with the 90 % well-cluster upper
  bound < 1.0 against BOTH CL and the level-adjusted CL (G1), and pooled 90 % coverage in
  [0.85, 0.95] (G1b).
* I1BC11T.G2_physics_fair_cl_2011: the page physics free run (recorded preorigin-v3 energy,
  climatological rain) against CL by E07's fair rule, ratio <= 1.25, per county (n >= 10).
* I1BC11T.yunlin_response_2010_2011: the page model's Yunlin-only contrast (E26's primary).
* I1BC11T.G3_pumping_value_2011: F1 against F0; gating only if the swap gate passed.

    python -I dataprep_staging/bc11/prereg_bc11.py --validate --quiet
    python -I dataprep_staging/bc11/prereg_bc11.py --write --quiet --expect-self-sha <sha256>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _p in (str(HERE), str(HERE.parent / "ch_x"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
#: CPU entry point: this checkout's hydrophysics (the hash-bound copy), not an editable install
if str(HERE.parents[1]) not in sys.path[:1]:
    sys.path.insert(0, str(HERE.parents[1]))

import bc11_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import score_bc11 as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{K.X}/prereg_bc11.py"
TESTS_FILE = f"{K.X}/tests/test_bc11.py"
RUNNER = f"{K.X}/run_bc11.sh"
RUNNER_LINES = (f"ID={K.ID}", f"WT_HEAD={K.WT_HEAD}", f'WT="$HOME/{K.WT_NAME}"',
                f"RUN={K.RUN_DIR}")


def frozen_code() -> dict:
    return {p: IO.sha_or_missing(CODE / p) for p in S.FROZEN + (THIS, TESTS_FILE)}


def _json(path: str) -> dict | None:
    p = IO.rp(path)
    return json.loads(p.read_text()) if p.is_file() else None


# ----------------------------------------------------------------------------- pre-gates
def pregates() -> dict:
    out = {}
    for key, path in S.PREGATES.items():
        out[key] = {"path": path, "sha256": IO.sha_or_missing(IO.rp(path))}
    a = _json(K.AVAIL_OUT)
    f = (a or {}).get("feasibility") or {}
    out["availability"].update({
        "passed": bool(f.get("passed")), "unit": K.AVAIL_UNIT,
        "look_time": (a or {}).get("look_time"), "n_eligible": f.get("n_eligible"),
        "n_eligible_by_county": f.get("n_eligible_by_county"),
        "county_line": f.get("county_line"),
        "origin2_report_only_n": (f.get("origin2_report_only") or {}).get("n_eligible"),
        "constants_sha256": f.get("constants_sha256")})
    e0 = _json(K.E0_OUT) or {}
    out["e0"].update({"passed": bool(e0.get("passed")),
                      "max_abs_diff_m": e0.get("max_abs_diff_m"), "tol_m": K.E0_TOL_M,
                      "code_head": e0.get("code_head")})
    sw = _json(K.SWAP_OUT) or {}
    out["swap"].update({"passed": bool(sw.get("passed")),
                        "counties": {c: (sw.get("counties") or {}).get(c, {}).get("passed")
                                     for c in K.COUNTIES}})
    mp = IO.rp(K.PRODUCT) / "MANIFEST.json"
    m = json.loads(mp.read_text()) if mp.is_file() else {}
    out["product"] = {"path": K.PRODUCT, "manifest_sha256": IO.sha_or_missing(mp),
                      "allowed_uses": m.get("allowed_uses"),
                      "gate_passed": (m.get("gate") or {}).get("passed"),
                      "build_look": (m.get("parameters") or {}).get("build_look")}
    return out


def g0_results() -> dict:
    out = {}
    for key, path in K.G0_FILES.items():
        r = _json(path)
        rec = {"path": path, "sha256": IO.sha_or_missing(IO.rp(path)), "seed": K.G0_SEEDS[key]}
        if r is None:
            rec["gate"] = None
        else:
            rec["gate"] = r["gate"]
            rec["reps"] = r.get("reps")
        out[key] = rec
    return out


def _dec(gate) -> bool | None:
    return None if gate is None else bool(gate.get("decidable"))


def roles(pg: dict, g0: dict) -> dict:
    """Each gate's role, fixed by the G0 results and the swap gate BEFORE registration:
    'gating' iff its G0 is decidable (G3: and the swap gate passed); 'report_only'
    otherwise; 'pending' while a G0 file is missing (the draft is then not ready)."""
    def role(d, extra=True):
        return "pending" if d is None else ("gating" if d and extra else "report_only")
    g2 = g0["g2"]["gate"]
    return {"g1": role(_dec(g0["g1"]["gate"])),
            "g2": {c: role(None if g2 is None else _dec(g2.get(c))) for c in K.COUNTIES},
            "yr": role(_dec(g0["yr"]["gate"])),
            "g3": role(_dec(g0["g3"]["gate"]), bool(pg["swap"]["passed"]))}


# ----------------------------------------------------------------------------- units
def wells() -> dict:
    a = _json(K.AVAIL_OUT) or {}
    f = a.get("feasibility") or {}
    ids = sorted(map(str, f.get("eligible_wells") or []))
    pw = f.get("per_well") or {}
    return {c: [s for s in ids if (pw.get(s) or {}).get("county") == c] for c in K.COUNTIES}


def unit_decl(county: str, ids: list[str]) -> dict:
    return {"description": f"{county.capitalize()} monitoring wells that qualified in "
                           f"{K.AVAIL_UNIT}'s logged availability count (>= 9 available "
                           "months of 2011 and one in 2010-10..12, ring-nest wells "
                           "excluded); heads/L1_pre2012 v2 (validator, design_time_evidence, "
                           "initial_conditions; local only); design-time, window flagged",
            "ids": ids, "window": K.WINDOW_TEXT}


def e26_overlap(changhua: list[str]) -> dict:
    """Changhua members shared with E26's registered unit (ids only)."""
    d = _json(f"results/dataprep/preregistration/{K.E26_ID}.json")
    if d is None:
        return {"n_shared": None, "note": "E26's registration not found"}
    e = set()
    for t in d.get("tests") or []:
        e |= set(map(str, ((t.get("external_units") or {}).get(K.E26_UNIT) or {}).get("ids")
                     or []))
    return {"e26_unit": K.E26_UNIT, "n_e26": len(e), "n_shared": len(e & set(changhua)),
            "n_changhua": len(changhua)}


def probe(units: dict, prereg_dir=None, registries=None) -> dict:
    """Per new unit, the earlier looks sharing members with it inside the window (unit ids,
    purposes, dates and shared-member counts only; looks.jsonl metadata), and the number of
    looks on units of unknown membership whose windows intersect (disclosed, uncomparable).
    None of them blocks: every one predates the registration and none names a new unit."""
    import avail_lib as AL

    out = {}
    for u, decl in units.items():
        rows = AL.prior_overlaps(u, decl, K.SCOPE, prereg_dir, registries)
        known = sorted({(r["unit"], r["purpose"], str(r["time"])[:10],
                         "universal" if r["universal"] else r["n_shared"])
                        for r in rows if r["membership"] == "overlap"}, key=str)
        out[u] = {"overlapping_looks": [{"unit": a, "purpose": b, "date": c, "shared": d}
                                        for a, b, c, d in known],
                  "n_unknown_membership_looks": sum(r["membership"] == "unknown"
                                                    for r in rows)}
    return out


# ----------------------------------------------------------------------------- tests
def tests(w: dict) -> list[dict]:
    units = {K.UNIT_OF_COUNTY[c]: unit_decl(c, w[c]) for c in K.COUNTIES}
    yl = {K.UNIT_YL: units[K.UNIT_YL]}
    common = {"data_scope": K.SCOPE, "label": S.S_LABEL + " on pass (R8.6b)"}
    return [
        {"test_id": K.TEST_G1, **common, "fresh_units": list(K.UNITS),
         "external_units": units, "role": "primary",
         "what": "Does E04's statistical recipe, with CL and rho refitted on 2012-22 only, beat "
                 "climatology and level-adjusted climatology on 2011 heads from a 2010-12 "
                 "start, with calibrated intervals?",
         "estimand": "pooled over the scored wells: R_CL = sqrt(mean_w MSE_w(stat)) / "
                     "sqrt(mean_w MSE_w(CL)) and R_adj the same against CL + each well's mean "
                     "2010 anomaly; MSE_w over the well's observed 2011 months; stat = CL + "
                     "rho_w^k x the anomaly at the last 2010 month within 3 months of 2010-12 "
                     "(twin.hybrid.persisted_anomaly); CL = month-of-year mean of 2012-22 "
                     "(drift_diag.fair_baselines); rho = hybrid.lag1_rho on 2012-22 "
                     "anomalies with the pooled-median fallback; sd = coverage._residual_sd "
                     "on 2012-22; 90 % interval = stat +- Z[90] sd",
         "interval": f"well-cluster bootstrap (wells with replacement), B {K.BOOT_B}, seed "
                     f"{K.BOOT_SEED}, upper bound = the {K.BOOT_Q} quantile (the upper end of "
                     "the central 90 % interval); the same resamples for both baselines",
         "decision_rule": f"not_evaluable if fewer than {K.MIN_POOLED_WELLS} wells keep "
                          f">= {K.MIN_FIT_MONTHS} fitted months, a finite sd, a 2010 month "
                          f"within {K.MAX_GAP} of the origin and >= {K.MIN_WINDOW_MONTHS} "
                          "observed 2011 months; G1 iff R_CL <= 1.0 and UB_CL < 1.0 and "
                          "R_adj <= 1.0 and UB_adj < 1.0; G1b iff pooled coverage in "
                          f"[{K.COVERAGE_BAND[0]}, {K.COVERAGE_BAND[1]}] (inclusive); PASS "
                          "iff G1 and G1b (E04's own rule needs both)",
         "county_lines": f"the same rule per county with >= {K.COUNTY_LINE_MIN_WELLS} scored "
                         "wells; reported, never substituted for the pooled verdict",
         "report_only": ["full persistence against CL and level-adjusted CL",
                         "level-adjusted CL against CL", "per-lead ratios (the L part of "
                         "the F/L memory line)"]},
        {"test_id": K.TEST_G2, **common, "fresh_units": list(K.UNITS),
         "external_units": units, "role": "secondary",
         "what": "Is the page physics, free-running from 2007 on recorded preorigin-v3 "
                 "energy with climatological rain, within E07's fair margin of climatology "
                 "on 2011 heads?",
         "estimand": "E07's fair rule (drift_diag.fair_temporal_verdict) with the baseline "
                     "set reduced to CL: datum_w = mean(model - obs) over the well's observed "
                     "2012-22 months (the page forward's own 2012-22 heads); ratio = pooled "
                     "RMSE(model - datum) / pooled RMSE(CL) over the observed 2011 "
                     "well-months; shape R2 about each well's observed 2012-22 mean",
         "decision_rule": f"per county with >= {K.COUNTY_LINE_MIN_WELLS} wells having >= "
                          f"{K.FAIR_MIN_FIT_MONTHS} observed 2012-22 months and an observed "
                          f"2011 month: PASS iff ratio <= {K.FAIR_K} and r2_shape(model) >= "
                          f"r2_shape(CL) - {K.FAIR_SHAPE_TOL} (E07's two clauses, CL only); "
                          "else not_evaluable for that county; the pooled line is "
                          "report-only",
         "report_only": ["the pooled line", "the CF origin-update arm (statistical origin "
                         "update of the datum-adjusted free run; not the page's state "
                         "restart)", "per-lead ratios of F1 and CF (the F part of the F/L "
                         "line) and the p90 over the wells of |F1 - F0| per lead"]},
        {"test_id": K.TEST_YR, **common, "fresh_units": [K.UNIT_YL], "external_units": yl,
         "role": "secondary (R8.6c)",
         "what": "Does the page model's Yunlin-only pumping contrast point the right way in "
                 "2010-11?",
         "estimand": "E26's primary (chx_lib.score_backcast): e = obs - B, c = A - B (A = F1, "
                     f"B = Yunlin cells at their {K.PATTERN_YEAR} monthly pattern from "
                     f"{K.PATTERN_YEAR + 1}-01), both residualised on well x calendar-month "
                     "and month-of-sample effects (form calmonth_twoway), the pinned per-well "
                     "FE slope and its 90 % 5 km block interval (bootstrap_scorer: B 2000, "
                     "seed 20261101)",
         "decision_rule": f"not_evaluable if fewer than {K.YR_MIN_WELLS} Yunlin wells keep "
                          f">= {K.YR_MIN_MONTHS} months; not_informative if the pinned gate on "
                          "the transformed contrast fails; DIRECTION SUPPORTED iff lo > 0 and "
                          "beta > 0"},
        {"test_id": K.TEST_G3, **common, "fresh_units": list(K.UNITS),
         "external_units": units, "role": "report-only unless the swap gate passed",
         "what": "Does recorded 2007-11 pumping (F1) improve the page physics' 2011 heads over "
                 "pumping climatology (F0)?",
         "estimand": "equal-station RMSE ratio of (F1 - datum) to (F0 - datum) on the "
                     "observed 2011 months, well-cluster bootstrap as G1",
         "decision_rule": "gating only if the swap gate (both counties) passed AND its G0 is "
                          "decidable (roles, frozen here); 'value shown' iff ratio <= 1.0 and "
                          "UB < 1.0; ledger only either way"},
    ]


# ----------------------------------------------------------------------------- the document
def doc(prereg_dir=None, registries=None) -> dict:
    pg = pregates()
    g0 = g0_results()
    w = wells()
    fz = frozen_code()
    units = {K.UNIT_OF_COUNTY[c]: unit_decl(c, w[c]) for c in K.COUNTIES}
    rl = roles(pg, g0)
    return {
        "id": K.ID, "wp": K.WP, "kind": "test",
        "title": "I1.backcast_2010_11.v1: 2011 heads from a 2010-12 start, months no model "
                 "has fitted: E04's recipe (primary), the page physics (E07's fair rule) and "
                 "a Yunlin response arm (design-time, window flagged)",
        "priors": "Plan #12: pass chance about 0.25-0.3 net for the S row (availability "
                  "about 0.65, done: PASS; G1 with its interval and the level-adjusted CL "
                  "about 0.5; ruling given). CL fitted on 2012-22 and applied backward makes "
                  "the level-adjusted CL a strong baseline if 2010-11 levels differ from the "
                  "2012-22 mean.",
        "rulings": {K.RULING_PRODUCT: f"{K.RULINGS}: L1_pre2012 v2 with Yunlin, local, built "
                                      "after the logged availability look",
                    K.RULING_S: f"{K.RULINGS}: a pass reads '{S.S_LABEL}', with the earlier "
                                "X5EV and mechanism.py reads of 2010-11 heads disclosed; E07 "
                                "stays N",
                    K.RULING_ARM: f"{K.RULINGS}: the Yunlin response arm with its own G0 "
                                  f"(power >= {K.G0_POWER_MIN}, false pass <= "
                                  f"{K.G0_FALSE_MAX}); report-only if that G0 fails",
                    "rulings_sha256": IO.sha_or_missing(IO.rp(K.RULINGS))},
        "tests": tests(w), "fresh_units": list(K.UNITS),
        "zones": "each well at its own forward-grid cell and twin layer (layer_code - 1); "
                 "counties from the station table; Yunlin / Changhua cells from the county "
                 "polygons (zone_county.county_of_points)",
        "burn_in": f"each page parameter set's own month-0 field (2012-01) placed at "
                   f"{K.BC_RUN[0]}; spin-up {K.BC_RUN[0]}..{K.SPINUP_END}; pre-origin "
                   f"{K.PRE[0]}..{K.PRE[1]}; scored {K.SCORED[0]}..{K.SCORED[1]}",
        "folds": NA("temporal holdout before every fit; no folds"),
        "gate_scores": NA("declared gates; no weighted score"),
        "primary_model": "E04's statistical recipe (twin.hybrid) refitted on 2012-22; the page "
                         "physics ensemble (stage3_datum_gate, 18 parameter sets) for G2, G3 "
                         "and the Yunlin arm",
        "recipe": {
            "thetas": list(K.PAGE_THETAS), "thetas_sha256": theta_shas(),
            "worktree_head": K.WT_HEAD, "rain": K.PAGE_RF,
            "rain_sha256": IO.sha_or_missing(IO.rp(K.PAGE_RF)),
            "page_forward": K.PAGE_FORWARD, "page_forward_sha256": K.PAGE_FORWARD_SHA,
            "model": "page_run.PageRun: a staging copy of hindcast.R1cRun without its "
                     "zone_county_split guard, over the page's 18 parameter sets, ensemble "
                     "mean (E0 binds it to cf2c7154)",
            "forcing": {"F1": "pump_electricity preorigin v3 cells_by_class value, all "
                              "classes, 2007-01..2011-12 (forcing_frozen)",
                        "F0": "F1's per-cell calendar-month climatology over 2007-11",
                        "B_yl": f"Yunlin cells at their {K.PATTERN_YEAR} pattern",
                        "recharge": "the page inputs' 2012-22 recharge (page rain file) as a "
                                    "calendar-month climatology: climatological rain",
                        "canal_percolation": "none (the page theta carries none)",
                        "public_supply": "none (the page theta carries none)"},
            "cl": {"product": K.CL_PRODUCT, "use": K.CL_USE, "window_end": K.CL_WINDOW_END,
                   "frozen_file": K.BASELINES, "e04_frozen_rho": "never used"},
            "order": ["availability look (logged 2026-11-06; metadata only)",
                      "product_build look (own unit) + L1_pre2012 v2 build",
                      "E0 (GPU)", "swap gate (GPU)", "backcast panel (GPU)",
                      "2012-22 baselines (CPU)", "G0 g1, g2, g3, yr (CPU, synthetic)",
                      "final draft", "THIS REGISTRATION", "score once"]},
        "pregates": pg, "g0": g0, "roles": rl,
        "thresholds": {
            "e0_tol_m": K.E0_TOL_M, "max_gap": K.MAX_GAP, "min_fit_months": K.MIN_FIT_MONTHS,
            "min_window_months": K.MIN_WINDOW_MONTHS, "min_pooled_wells": K.MIN_POOLED_WELLS,
            "county_line_min_wells": K.COUNTY_LINE_MIN_WELLS,
            "g1": {"max_ratio": K.G1_MAX_RATIO, "ub_below": K.G1_UB_BELOW,
                   "boot_B": K.BOOT_B, "boot_seed": K.BOOT_SEED, "boot_q": K.BOOT_Q},
            "g1b_coverage_band": list(K.COVERAGE_BAND),
            "g2": {"fair_k": K.FAIR_K, "shape_tol": K.FAIR_SHAPE_TOL,
                   "shape_clause": K.G2_SHAPE_CLAUSE, "min_fit_months": K.FAIR_MIN_FIT_MONTHS},
            "g3": {"max_ratio": K.G3_MAX_RATIO, "ub_below": K.G3_UB_BELOW},
            "swap": {"energy_band": list(K.SWAP_ENERGY_BAND), "corr_min": K.SWAP_CORR_MIN,
                     "rms_band": list(K.SWAP_RMS_BAND), "pattern_year": K.SWAP_PATTERN_YEAR},
            "yr": {"form": K.YR_FORM, "min_wells": K.YR_MIN_WELLS,
                   "min_months": K.YR_MIN_MONTHS},
            "g0": {"power_min": K.G0_POWER_MIN, "false_max": K.G0_FALSE_MAX,
                   "seeds": dict(K.G0_SEEDS), "reps": {"g1": K.G0_K_G1, "g2": K.G0_K_G2,
                                                       "g3": K.G0_K_G3, "yr": K.G0_K_YR},
                   "rho_grid": list(K.G0_RHO_GRID), "share_grid": list(K.G0_SHARE_GRID),
                   "yr_k": list(K.G0_YR_K), "yr_rho": list(K.G0_YR_RHO)},
            "bounds": "as written: '<=' and '<' exactly; coverage band inclusive"},
        "dropped_as_gates": {"PF": "re-originated physics forecast: dropped as a gate (plan)",
                             "FM-D": "dropped as a gate; no report-only line here"},
        "frozen_code": fz,
        "prior_looks_disclosed": {
            "x5ev": f"{K.X5EV_ID} loaded the 2010-11 hourly ls_cache series of its confined "
                    f"wells ({K.X5EV_UNITS[0]}) for a 1-cpd band estimand and reported its "
                    f"unconfined companion unit ({K.X5EV_UNITS[1]})",
            "print_hourly_2010_01_01": f"three hourly 2010-01-01 values of confined well "
                              f"{K.PRINTED_WELL} were printed in a metadata check before X5EV "
                              "was registered (disclosed 2026-10-25)",
            "i1chb": f"{K.E26_ID} (E26; units {', '.join(K.I1CHB_UNITS)}) logged a timestamps "
                     "availability look, built heads/L1_pre2012 v1 and scored 2010-11 "
                     "Changhua heads with R1c (2026-11-01)",
            "e26_overlap": e26_overlap(w["changhua"]),
            "mechanism_py": "mechanism.py's long-heads option read the whole ls_cache files "
                            "of the wells within 1000 m of a ring site; those wells were "
                            "excluded from the gating count and are not in these units",
            "availability_look": f"{K.AVAIL_UNIT}: datetime column only (logged "
                                 f"{pg['availability'].get('look_time')}, PASS, 32 wells, "
                                 "origin 2009-12 had 0 qualifying wells)",
            "product_build_look": f"{K.PRODUCT_UNIT}: the L1_pre2012 v2 build read the "
                                  "2010-11 values for QC and monthly aggregation only (no "
                                  "model-vs-observation statistic), before this registration",
            "cl_time_reversal": "CL, rho, sd and the physics datum are fitted on 2012-22 and "
                                "applied BACKWARD to 2010-11",
            "fv_stat_v2_generator": "the G0 worlds reuse FV.stat.v2's AR(1) generator and "
                                    "grid (synthetic; FV.stat.v2 is scored)",
            "probe": probe(units, prereg_dir, registries),
            "models": "the page model has never been run before 2012; E26 ran R1c on "
                      "2007-11 (Changhua contrast only)"},
        "evidence_and_ledger_mapping": {
            "G1_pass": f"new row '{S.S_LABEL}' beside E04 (E04 unchanged): {S.WORDING['g1_pass']}",
            "G1_fail": f"ledger + page narrowing line: {S.WORDING['g1_fail']}",
            "G2_pass": f"per county, a '{S.S_LABEL}' row beside E07; E07 stays N: "
                       f"{S.WORDING['g2_pass']}",
            "G2_fail": f"pre-written narrowing per county: '{S.NARROW_PHYSICS}'",
            "YR_pass": f"a heads response row for Yunlin: {S.WORDING['yr_pass']}",
            "YR_fail": S.WORDING["yr_fail"] + " (ledger and the C-row wording)",
            "G3": "ledger only: " + S.WORDING["g3_pass"] + " / " + S.WORDING["g3_fail"],
            "report_only_gate": S.WORDING["report_only"],
            "not_evaluable": S.WORDING["not_evaluable"],
            "page_ruling": f"{K.RULING_S} (2026-11-03) approves the label; E07 stays N"},
        "attempt_policy": {"scored_runs": "one", "exploratory_followups_max": 0,
                           "engineering_retries": "free before any 2010-11 head is compared; "
                                                  "logged in attempts.jsonl",
                           "thresholds_frozen": True, "never_substituted":
                               "no secondary or county line replaces the primary"},
        "u7": "no post-2022-12 datum enters selection, calibration or design: CL, rho, sd and "
              "the datum come from 2012-22, the G0 noise scale from 2012-22 residuals, the "
              "wells from 2009-11 timestamps",
        "interpretation": "A G1 pass says E04's recipe, fitted only on 2012-22, would have "
                          "beaten both climatologies in 2011 from a 2010-12 start, with "
                          "calibrated intervals: a design-time out-of-period check of the "
                          "statistical layer (one year, 32 wells, window flagged by the "
                          "disclosed earlier reads). G2 is a non-inferiority check of the "
                          "physics level, not skill. Rain is climatological: a year drier "
                          "than the 2012-22 climatology raises pumping and lowers recharge, "
                          "so the physics run sits too high and the dry-year bias pushes G2 "
                          "toward fail (and the Yunlin arm's month effects absorb its common "
                          "part only). A pass carries the label "
                          f"'{S.S_LABEL}' (R8.6b), never S alone.",
        "inputs": ([{"path": p, "sha256": h} for p, h in fz.items() if h != "missing"]
                   + [K.AVAIL_OUT, K.RULINGS, f"{K.CENSUS_V3}/MANIFEST.json",
                      f"{K.CL_PRODUCT}/MANIFEST.json", K.PAGE_FORWARD]
                   + list(K.PAGE_THETAS)
                   + [v for v in S.PREGATES.values() if v != K.AVAIL_OUT]
                   + [f"{K.PRODUCT}/MANIFEST.json"] + list(K.G0_FILES.values())),
        "no_prior_look_statement": "no 2010-11 head has been compared with any model, "
                                   "baseline or forecast for this test; the only read of "
                                   "2010-11 values for it is the logged product build",
        "written_by": THIS,
    }


def theta_shas() -> dict:
    return {p: IO.sha_or_missing(IO.rp(p)) for p in K.PAGE_THETAS}


def readiness(d: dict) -> list[str]:
    out = []
    pg = d["pregates"]
    for key, g in pg.items():
        if key != "product" and g["sha256"] == "missing":
            out.append(f"pre-gate {key} not run ({g['path']})")
    if pg["availability"]["sha256"] != "missing" and not pg["availability"]["passed"]:
        out.append("availability FAILED: record 'not evaluable'; do not register")
    if pg["e0"]["sha256"] != "missing" and not pg["e0"]["passed"]:
        out.append("E0 FAILED: engineering (code ladder); do not register")
    if pg["product"]["manifest_sha256"] == "missing":
        out.append(f"{K.PRODUCT} not built")
    elif "design_time_evidence" not in (pg["product"]["allowed_uses"] or []):
        out.append("the product does not allow design_time_evidence")
    for key, g in d["g0"].items():
        if g["gate"] is None:
            out.append(f"G0 {key} not run ({g['path']})")
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not IO.rp(p).exists() and not (CODE / p).exists():
            out.append(f"input {p} is missing")
    sh = (CODE / RUNNER).read_text() if (CODE / RUNNER).exists() else ""
    out += [f"{RUNNER} lacks {s!r}" for s in RUNNER_LINES if s not in sh.splitlines()]
    if any(not d["tests"][0]["external_units"][u]["ids"] for u in K.UNITS):
        out.append("a fresh unit has no member")
    if IO.rp(K.PREREG_FILE).exists():
        out.append(f"{K.ID} is already registered")
    if IO.rp(K.REPORT).exists():
        out.append(f"{K.REPORT} exists")
    return sorted(set(out))


def blocking(d: dict, prereg_dir=None, registries=None) -> list:
    """The prereg3 blocking probe as write_prereg runs it (dry run; nothing written)."""
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir}
    idx = prereg3.membership_index(registries=registries, **kw)
    for t in d["tests"]:
        for u, v in (t.get("external_units") or {}).items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    return prereg3.blocking_looks(d["tests"], None, idx=idx,
                                  test_time=dt.datetime.now(dt.timezone.utc), **kw)


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
    block = blocking(d)
    print("blocking looks (dry run):", block or "none")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    print("roles:", json.dumps(d["roles"]))
    self_sha = IO.sha256(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    if a.validate:
        print("draft valid (schema, blocking)" if not (errs or block) else
              "NOT VALID: fix the problems above")
        return 1 if (errs or block) else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if errs or block or pending:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
