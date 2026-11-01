"""Write the two design-time declarations of round 7 ranks 4-5 (harness.prereg3, kind
'declaration'):

  dr     DR.relative_drift.v1 - BEFORE the fitted-month residuals are read (dr_drift.py run)
  power  U1.decomp_power.v1   - AFTER DR's report exists (its grid is derived from it by a
                                declared formula) and BEFORE power_e02d.py run

``--validate`` (dry run) prints the document, its canonical sha256, this script's sha256,
prereg3.validate_prereg and readiness; nothing is written. ``--write`` registers once (the lead
only): it refuses if a slot is missing, validation fails, the E02d deadline has passed or
``--expect-self-sha`` is not this file's sha256. ``--skip-digest`` (dry runs only) skips
hashing the head-well directory (a byte digest; no value is parsed).

    python -I dataprep_staging/e02d/prereg_design.py dr --validate [--skip-digest]
        [--lead-from datum-centre|origin]
    python -I dataprep_staging/e02d/prereg_design.py dr --write --expect-self-sha <sha> \
        --lead-from datum-centre|origin

``--lead-from`` is the lead's sign-off on where the drift lead is measured from (the judge's
spec said 'at a 4-year lead' from the origin; the builder proposes the datum centre, about 10
years to window A). ``dr --write`` refuses without it; the choice is recorded in the
declaration as ``lead_definition`` and dr_drift.py / build_artefact.py follow it.
``--validate`` also checks the source's governance card (heads/L1_clean v3 MANIFEST:
datum_fit and design_time_evidence at window_end 2022-12, metadata only); ``--write``
refuses if that check fails.
    python -I dataprep_staging/e02d/prereg_design.py power --validate
    python -I dataprep_staging/e02d/prereg_design.py power --write --expect-self-sha <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dr_drift as DR  # noqa: E402
import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402
import power_e02d as PW  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{io.X}/prereg_design.py"
TESTS = f"{io.X}/tests/test_e02d.py"

HISTORY = {
    "X5.storage_priors.v2_postfailure": "look 2026-10-25T15:13:47Z, purpose "
                                        "design_time_estimation, unit "
                                        "X5:heads_2012-14_2016-18_layers2-4 (hourly heads "
                                        "2012-14 and 2016-18 at 211 screens: earth-tide and "
                                        "barometric fits)",
    "N1.policy_response.v1 / N1.changhua_fix.v1": "score looks 2026-10-23 on N1P:wells_m1618 "
                                                  "(heads 2016-01..2018-12) of masked-run "
                                                  "models (a different model than the page "
                                                  "model)",
    "E02 (results/twin/prospective_2026-11, 2026-10-11)": "the same rows, use datum_fit: the "
                                                          "per-well datum and the fitted "
                                                          "residual sd of E02's interval",
    "the page model's calibration": "theta, the well datum and the column were fitted on "
                                    "2012-22, both halves",
    "round-5/6 diagnoses": "after-the-fact 2023-26 numbers are known to the designers "
                           "(combined coverage 0.92-0.93, run spread 0.75, hybrid 0.877, "
                           "the 2023 distal drift); none enters any rule here"}


def _wells_digest() -> dict:
    d = io.REPO / L.WELLS_DIR
    files = sorted(d.glob("*.parquet"))
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.name}\0{io.sha256(p)}\n".encode())
    return {"path": L.WELLS_DIR, "glob": "*.parquet", "n_files": len(files),
            "kind": "sha256 over 'name NUL sha256(file)' lines, sorted by name",
            "sha256": h.hexdigest()}


GOV_CARD = "results/dataprep/heads/L1_clean/v3"
GOV_USES = ("datum_fit", "design_time_evidence")


def _governance() -> dict:
    """The source's governance status (MANIFEST metadata only, no table is read): DR reads
    the raw page-model input (AMP_V2/data/wells via build_head_field, E02's exact path, which
    the 1e-6 m datum gate needs); the governed product over the same wells, heads/L1_clean
    v3, must allow datum_fit and design_time_evidence for rows up to 2022-12."""
    from common import guard
    from common.manifest import load_manifest

    out = {"read_path": f"{L.WELLS_DIR} (raw page-model input, no MANIFEST; the path E02's "
                        "registration and the page model's calibration read)",
           "card": GOV_CARD, "uses": list(GOV_USES), "window_end": L.ORIGIN}
    d = io.REPO / GOV_CARD
    try:
        m, _ = load_manifest(d)
    except (OSError, ValueError) as exc:
        return {**out, "ok": False, "why": f"no readable MANIFEST ({exc})"}
    out.update({"product": f"{m.get('product')} {m.get('version')}",
                "manifest_sha256": io.sha256(d / "MANIFEST.json")})
    bad = []
    for u in GOV_USES:
        try:
            guard.check_use(m, u, window_end=L.ORIGIN)
        except Exception as exc:  # noqa: BLE001  (any refusal is a failed check)
            bad.append(f"{u}: {exc}")
    return {**out, "ok": not bad, "why": bad or "both uses allowed (time slice to 2022-12)"}


def _unit_ids() -> list[str]:
    import pandas as pd

    p = io.REPO / L.E02_DIR / "predictions.csv"
    ids = sorted(pd.read_csv(p, dtype={"station_id": str})["station_id"].unique())
    if len(ids) != L.N_UNIT_WELLS:
        raise SystemExit(f"E02 has {len(ids)} wells, expected {L.N_UNIT_WELLS}")
    return ids


# ----------------------------------------------------------------------------- DR
LEAD_TEXT = {
    "datum_centre": "the lead runs from each well's datum centre (the mean decimal year of its "
                    "fitted months), not from the origin: the datum is the 2012-22 mean "
                    "residual, so a linear drift has accumulated over about 10 years by "
                    "window A; dt in the interval is (month - t_c)",
    "origin": f"the lead runs from the forecast origin ({L.ORIGIN_ANCHOR_YEAR:g}, the end of "
              "2022-12) for every well, as in the judge's spec; dt in the interval is "
              f"(month - {L.ORIGIN_ANCHOR_YEAR:g})"}


def doc_dr(skip_digest: bool = False, lead_definition: str = "datum_centre") -> dict:
    if lead_definition not in L.LEAD_DEFINITIONS:
        raise ValueError(f"lead definition {lead_definition!r}")
    ids = _unit_ids()
    frozen = {f: io.sha256(io.REPO / f) for f in DR.FROZEN}
    return {
        "id": L.DR_ID, "wp": L.DR_WP, "kind": "declaration",
        "lead_definition": lead_definition,
        "title": "Split-half drift of each well's county-relative head residual and the "
                 "pre-origin residual ICCs by county, layer group and zone (one logged "
                 "design-time read of the page model's 2012-22 fitted-month residuals)",
        "why": "E02d (round 7 rank 5) scores a per-well interval around the county-wide "
               "departure. Its width needs (a) the grouping inside which the shared part is "
               "removed and (b) whether a constant width is too narrow at a 4-6-year lead. "
               "Both are fixed here by rules written before the residuals are read, from "
               "pre-origin rows only. User ruling R7.e02d_residual_read (2026-11-01, sixth "
               "set): 'Yes, by the pre-declared rule (DR.relative_drift.v1)'.",
        "ruling": {"file": L.RULINGS, "id": L.RULING_IDS["dr"]},
        "read": {
            "rows": "E02's own read: twin.coverage.assemble's arrays (AMP_V2/data/wells via "
                    "twin.heads.build_head_field, t0 2012-01-01, t1 2023-01-01; forward "
                    f"{L.FORWARD} sha {L.MODEL_SHA[:8]}), fitted months 2012-01..2022-12, at "
                    "E02's 158 wells; nothing after 2022-12, no masked or rolling-origin run, "
                    "no external well",
            "uses": "datum_fit (the E02 datum, as E02 itself) and design_time_evidence",
            "governance": _governance(),
            "look": f"logged once by dr_drift.py run: unit {L.DR_UNIT}, scope {L.DR_SCOPE}, "
                    "purpose design_time_estimation, immediately before the arrays load; an "
                    "engineering re-run logs again and is reported",
            "runs_before": "E02d's artefact freeze; if E02d is frozen first this becomes "
                           "report-only and can change wording, never the frozen interval"},
        "steps": {
            "0_engineering_gate": f"E02's recipe rebuilt from the arrays (datum = mean fitted "
                                  f"residual, sd = fitted residual sd ddof 1, combined "
                                  f"sigma, keep = >= 24 fitted months) reproduces every "
                                  f"predicted_m / lower_m / upper_m of {L.E02_DIR}/"
                                  f"predictions.csv to {L.DATUM_TOL_M} m on exactly its 158 "
                                  "wells; otherwise nothing is computed (engineering failure)",
            "1_residuals": "r = observed - (forward member mean + datum), fitted months",
            "2_icc": {
                "icc_county_month": "per county, one-way random-effects ANOVA ICC of r over "
                                    "county-month cells (unequal sizes; (MSB - MSW) / (MSB + "
                                    "(n0 - 1) MSW), clipped to [0, 1], raw reported)",
                "icc_layer_month": "the same ICC of the county-relative r (LOO median, >= "
                                   f"{L.MIN_OTHERS} other wells) over county x layer-group-"
                                   "month cells; layer groups L1, L2, L34 (layers 3-4)",
                "icc_zone_month": "the same over county x zone-month cells; zones prox, mid, "
                                  "dist of twin.zone_county.column_zone_map at the page "
                                  "model's meta",
                "merge": f"a partition group with < {L.MIN_GROUP_WELLS} of the county's unit "
                         "wells joins the next group in the declared order (L1, L2, L34; "
                         "prox, mid, dist), the last one the previous; repeated",
                "grouping_rule": f"per county: the sub-county partition with the larger ICC "
                                 f"if that ICC >= {L.ICC_SWITCH} and it has >= 2 groups "
                                 "after merging (tie: layer), otherwise the county",
                "icc_remaining": "the ICC of the chosen-group relative residual over the "
                                 "other partition's group-months (county grouping: the "
                                 "larger of the two ICCs); it bounds U1.decomp_power.v1's "
                                 "sub-group shock share",
                "icc_remaining_bootstrap": f"well x moving-block ({L.ICC_BOOT_BLOCK} months) "
                                           f"bootstrap of the same ICC on the same relative "
                                           f"residuals (B {L.ICC_BOOT_B}, seed "
                                           f"{L.ICC_BOOT_SEED}; county grouping: the larger "
                                           f"of the two partition ICCs per replicate); its "
                                           f"{L.ICC_BOOT_Q:.0%} quantile is the power grid's "
                                           "sampling allowance (in-sample damping is covered "
                                           "by the power grid's fixed margin, not by it)",
                "icc_well_report_only": "per-well persistence ICC of the relative residual"},
            "3_relative": f"e = r - LOO median of the chosen group's other wells that month "
                          f"(>= {L.MIN_OTHERS}); sd_fit_rel = root mean square of e over the "
                          f"well's fitted months (>= {L.MIN_FIT_MONTHS_SD} months, else the "
                          "well is not scored by E02d); the datum centre t_c = mean decimal "
                          "year of the well's fitted months",
            "4_drift": {
                "statistic": f"d_i = mean e over {L.LATE[0]}..{L.LATE[1]} minus mean e over "
                             f"{L.EARLY[0]}..{L.EARLY[1]}",
                "inclusion": f">= {L.MIN_HALF_MONTHS} months in each half; at least "
                             f"{L.INCLUSION_MIN_SHARE:.0%} of the county's unit wells, else "
                             "'not evaluable' for that county (no drift term, stated)",
                "ar_estimator": f"phi_i = sum x_t x_t+1 / sum (x_t^2 + x_t+1^2) / 2 over "
                                f"consecutive months inside one half of the half-demeaned "
                                f"e, clipped to {list(L.AR_CLIP)}; county median where fewer "
                                f"than {L.MIN_AR_PAIRS} pairs",
                "expected_variance": "E_i = s_i^2 (f(n_e, phi_i) / n_e + f(n_l, phi_i) / "
                                     "n_l), s_i^2 the pooled within-half variance, f(n, phi) "
                                     "= 1 + 2 sum_{k<n} (1 - k/n) phi^k (gaps ignored)",
                "model": f"linear trend: tau^2 = max(0, mean(d^2 - E)) / {L.DELTA_YEARS:g}^2 "
                         "(m^2 per year^2), per county; a well-bootstrap 90 % interval is "
                         "report-only",
                "label": "tau is a LOWER bound: theta, the datum and the column were fitted "
                         "on both halves (in-sample damping); a phi estimated low biases tau "
                         "up (conservative for the interval)"},
            "5_rule": {
                "variants": "exactly two interval variants: sigma_rel = sqrt(sd_fit_rel^2 + "
                            "sigma_ens_rel^2) or sigma_rel_drift = sqrt(sigma_rel^2 + tau^2 "
                            "dt^2)",
                "rule": f"per county: include the drift term iff sqrt(s^2 + tau^2 L^2) / s - "
                        f"1 > {L.DRIFT_RULE_FRACTION}, s = the county median sd_fit_rel, L = "
                        "the county median over included wells of (window A mid-month 2027-05 "
                        "- anchor) in years",
                "s_definition": "s is sd_fit_rel alone, not sigma_rel: the relative member "
                                "spread does not exist when DR runs (the member rerun may "
                                "finish later). Leaving it out makes s smaller, so the rule "
                                "triggers more easily (it errs toward including the drift "
                                "term, the wider interval)",
                "lead_definition": LEAD_TEXT[lead_definition],
                "lead_choice": f"'{lead_definition}', chosen by the lead at write time "
                               "(--lead-from); the builder proposed 'datum_centre' and "
                               "flagged it as a change from the judge's 'at a 4-year lead'",
                "literal_4y_report_only": "the plan's literal 'at a 4-year lead' rule (L = 4) "
                                          "is printed beside it, report-only"}},
        "outputs": {"dir": DR.REPORT_DIR, "files": ["report.json", "wells_e02d.csv"],
                    "wells_e02d_columns": list(DR.WELL_COLS),
                    "written_once": True,
                    "page_line": "C14b (N) gains, per county, 'in-sample, each well's "
                                 "county-relative offset drifted by X over ~6 years "
                                 "(pre-origin; a lower bound)' with X = page_numbers."
                                 "excess_drift_over_6y_m = sqrt(max(0, mean(d^2 - E))) = 6 "
                                 "tau, the noise-corrected excess; the root mean square of d "
                                 "includes sampling noise, overstates the drift and is "
                                 "report-only (never on the page); if the rule triggers: "
                                 "'constant-width intervals are likely too narrow at 4-5-year "
                                 "leads (pre-origin evidence)'; not evaluable: 'drift not "
                                 "evaluable'"},
        "gates": ["engineering: step 0 to 1e-6 m before any statistic",
                  f"inclusion: >= {L.INCLUSION_MIN_SHARE:.0%} per county, else not evaluable "
                  "for that county",
                  "decision rule (not pass/fail): the drift term enters by the 10 % rule"],
        "history_disclosure": HISTORY,
        "e02d_commitments": {
            "member_gate_no_fallback": "if the fold-member gate fails at both declared codes "
                                       "(4c2723e, then 730f65b), E02d is NOT registered: "
                                       "there is no fallback interval without the relative "
                                       "member spread; C14a-r is recorded N ('undecidable: "
                                       "the relative member spread could not be "
                                       "reproduced')"},
        "freshness": "prereg3 does not block this declaration (no fresh unit; the rows are "
                     "design-time). Non-blocking is NOT freshness: these rows were fitted and "
                     "looked at before (history above); the output is a design-time line and "
                     "construction input, never support or validation",
        "external_units": {L.DR_UNIT: {
            "description": "E02's 158 wells, pre-origin fitted months 2012-01..2022-12, "
                           "the page model's calibration-fit residuals",
            "ids": ids, "window": "2012-01..2022-12"}},
        "input_digests": [] if skip_digest else [_wells_digest()],
        "frozen_code": frozen,
        "zones": NA("no fit; the page model's column zones label wells only"),
        "priors": NA("no prior; design-time estimation"),
        "burn_in": NA("no fit"), "folds": NA("no fold"),
        "tests": NA("design-time declaration; nothing is scored"),
        "fresh_units": NA("no fresh unit: design-time rows (see 'freshness')"),
        "thresholds": {"min_others": L.MIN_OTHERS, "min_group_wells": L.MIN_GROUP_WELLS,
                       "icc_switch": L.ICC_SWITCH, "min_half_months": L.MIN_HALF_MONTHS,
                       "inclusion_min_share": L.INCLUSION_MIN_SHARE,
                       "drift_rule_fraction": L.DRIFT_RULE_FRACTION,
                       "datum_tol_m": L.DATUM_TOL_M, "ar_clip": list(L.AR_CLIP),
                       "min_ar_pairs": L.MIN_AR_PAIRS, "delta_years": L.DELTA_YEARS},
        "gate_scores": NA("no promoting gate"),
        "primary_model": {"forward_sha256": L.MODEL_SHA, "e02_predictions_sha256":
                          L.E02_PRED_SHA},
        "inputs": sorted({f"{L.E02_DIR}/predictions.csv", L.FORWARD, L.STATIONS, L.RULINGS,
                          TESTS, *DR.FROZEN}),
        "interpretation": "Design-time only. It sets E02d's grouping and drift term by "
                          "declared rules and gives C14b one in-sample line; it supports no "
                          "claim.",
        "written_by": THIS}


# ----------------------------------------------------------------------------- power
def doc_power(dr_dir: Path) -> dict:
    dr, wells, dsha, wsha = PW.load_dr(dr_dir)
    design = PW.design_spec(dr, wells, dsha, wsha)
    n = {c: len(PW.cells(design["counties"][c])) for c in L.COUNTIES}
    return {
        "id": L.POWER_ID, "wp": L.POWER_WP, "kind": "declaration",
        "title": "Synthetic P-gate of E02d's per-county relative (per-well) coverage test, "
                 "11 months (2026-12..2027-10) and 12 months (years 2-3), ICC grid bounded by "
                 "DR.relative_drift.v1",
        "why": "E02d is registered only if its exact scorer is decisive by design: false-fail "
               f"<= {PW.FALSE_FAIL_MAX} at a true 0.90 and power >= {PW.POWER_MIN} against a "
               f"{PW.POWER_AT} interval, in both counties, over a grid bounded by the "
               "pre-origin estimates (not chosen by designers who know the 2023 drift).",
        "design": design, "n_cells": n,
        "derivation": {"a_s": f"{{0, rem, max(rem + {PW.AS_MARGIN}, rem_up90)}} capped at "
                              f"{PW.AS_CAP}; rem = DR icc_remaining of the county, rem_up90 = "
                              "its well x month-block bootstrap upper-90 % (sampling "
                              f"uncertainty); the +{PW.AS_MARGIN} floor stands for in-sample "
                              "damping, which biases rem low as it biases tau low",
                       "a_s_disclosure": "out-of-sample sub-group shocks (for example zone "
                                         "shocks) may still exceed this bound; a 2028 fail "
                                         "can then reflect a shock and is reported with its "
                                         "by-group coverage (report-only)",
                       "a_w": f"{{0, w, w + {PW.AW_MARGIN}}} capped at {PW.AW_CAP}; w = DR "
                              "window_well_share_hat (tau^2 L^2 / (s^2 + tau^2 L^2))",
                       "phi": "{0, DR phi_median}", "dropout": list(PW.DROPOUT),
                       "groups": "DR's frozen groups and sub-groups; scale = sd_fit_rel / "
                                 "county median"},
        "decision_to_registration": {
            "window_A": "both counties pass at 11 months -> E02d may be registered; "
                        "otherwise E02d is NOT registered and C14a-r is recorded N "
                        "('undecidable even after removing the common mode')",
            "years_2_3": "per county at 12 months: pass -> the relative year-2/3 test is "
                         "primary for that county; fail -> report-only",
            "member_gate": "independently of the P-gate: if the fold-member gate fails at "
                           "both declared codes, E02d is not registered (no fallback; "
                           "declared in DR.relative_drift.v1 e02d_commitments)"},
        "self_checks": "vectorised verdicts equal e02dlib.county_verdict on the same "
                       "replicates; a county-month shock cancels exactly; a failure is an "
                       "engineering failure that gates nothing (a fix to bound code needs a "
                       "new id)",
        "smoke_disclosure": "the builder ran 'power_e02d.py smoke' only on a SYNTHETIC DR "
                            "rehearsal (dr_drift.py rehearsal, seed 7) in a scratch "
                            "directory and in the test suite; no number from it sets "
                            "anything",
        "code_sha256": {f: io.sha256(io.REPO / f) for f in PW.CODE_FILES},
        "run": {"command": f"python -I {io.X}/power_e02d.py run",
                "result": f"{PW.RESULT_DIR}/result.json", "runs_once": True,
                "cost": "CPU only, about 10-20 minutes; 0 GPU-h"},
        "zones": NA("synthetic draws"), "priors": NA("synthetic draws"),
        "burn_in": NA("synthetic draws"), "folds": NA("synthetic draws"),
        "tests": NA("declaration of a synthetic study; no observation is scored"),
        "fresh_units": NA("no observation is involved"),
        "thresholds": design["decision"], "gate_scores": NA("no promoting gate"),
        "primary_model": NA("no model: synthetic residuals and E02d's scorer only"),
        "inputs": sorted({f"{PW.DR_DIR}/report.json", f"{PW.DR_DIR}/wells_e02d.csv", THIS,
                          TESTS, *PW.CODE_FILES}),
        "interpretation": "Decidability of E02d by design; it estimates nothing from heads.",
        "written_by": THIS}


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("dr", "power"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--skip-digest", action="store_true")
    ap.add_argument("--lead-from", choices=("datum-centre", "origin"), default=None,
                    help="dr only: where the drift lead is measured from (required for "
                         "--write; --validate defaults to datum-centre)")
    ap.add_argument("--dr-dir", type=Path, default=io.REPO / PW.DR_DIR)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    lead = (a.lead_from or "datum-centre").replace("-", "_")
    try:
        d = doc_dr(a.skip_digest, lead) if a.which == "dr" else doc_power(a.dr_dir)
    except (io.Refused, OSError) as exc:
        print(f"NOT READY: {exc}")
        return 1
    pdir = prereg_dir or io.prereg_dir_default()
    errs = prereg3.validate_prereg(d, prereg_dir=pdir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    self_sha = io.sha256(Path(__file__))
    print("draft sha256:", io.sha_json(d))
    print("validate:", errs or "ok")
    print("this script sha256:", self_sha)
    if not a.write:
        return 1 if errs else 0
    bad = []
    if a.expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {a.expect_self_sha!r} != {self_sha}")
    if io.now() >= io.DEADLINE:
        bad.append("the E02d deadline 2026-11-30T16:00Z has passed: DR would be report-only")
    if a.which == "dr" and a.skip_digest:
        bad.append("--skip-digest is for dry runs only")
    if a.which == "dr" and a.lead_from is None:
        bad.append("--lead-from datum-centre|origin is required: the lead signs off where "
                   "the drift lead is measured from")
    if a.which == "dr" and not d["read"]["governance"].get("ok"):
        bad.append(f"governance check failed: {d['read']['governance'].get('why')}")
    if a.which == "dr" and (io.REPO / DR.REPORT_DIR / "report.json").exists():
        bad.append("DR's report already exists: the read happened before the declaration")
    if a.which == "power":
        if io.index_record(L.DR_ID, pdir) is None:
            bad.append(f"{L.DR_ID} is not registered")
        if (io.REPO / PW.RESULT_DIR / "result.json").exists():
            bad.append("the power result already exists")
    if errs or bad:
        print("REFUSED:", errs + bad)
        return 1
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
