"""Declaration X5.CH_twfe_mirror.v1 (harness.prereg3, kind 'declaration'; round 7 rank 16):
within-township response of Changhua heads to local census energy on the 74 spent calibration
screens, mirrored on model-world heads (R1c primary; ccz_m1618 and q0c report-only), with
placebos and a synthetic G0 gate. Narrowing only: kappa = beta_obs / beta_model is reported,
never applied (user ruling 2026-10-29 (4)); C13a stays W on the page.

By default it VALIDATES only (dry run: prints the JSON, its sha256, validate_prereg and the
readiness list); ``--write`` registers once and needs ``--expect-self-sha``. Run from the
repository root after ``run_twfe.sh prep`` (the frozen inputs are hashed here) and after the
rank 8 census audit exists:

    ~/miniforge3/envs/hydro/bin/python dataprep_staging/ch_audit/prereg_twfe.py --validate
    ~/miniforge3/envs/hydro/bin/python dataprep_staging/ch_audit/prereg_twfe.py --write \\
        --expect-self-sha <sha256 printed by --validate>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "dataprep_staging"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import twfe_lib as T  # noqa: E402
import twfe_mirror as M  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{M.X}/prereg_twfe.py"
TESTS = f"{M.X}/tests/test_twfe_mirror.py"
FROZEN = (M.DESIGN, *(f"{M.LOG}/model_{n}.npz" for n in M.MODELS))


def sha_text(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_inputs(repo: Path = REPO) -> dict:
    return {p: (M.sha256(repo / p) if (repo / p).exists() else "missing") for p in FROZEN}


def audit_coupling(repo: Path = REPO) -> dict:
    """Rank 8's census audit: D1, D1b, the annual-sum change of the re-split and the parity of
    the regressor's energy (preorigin v3 in-grid) with R1c's forcing (model_census v1 kept
    in-grid), Changhua."""
    p = repo / M.AUDIT
    if not p.exists():
        return {"path": M.AUDIT, "present": False}
    a = json.loads(p.read_text())["decisions"]
    return {"path": M.AUDIT, "present": True, "sha256": M.sha256(p),
            "D1": a["D1_changhua_forcing_version"]["decision"],
            "D1b": a["D1b_duty_cap"]["decision"],
            "annual_sum_max_rel_change": a["annual_sum_max_rel_change"],
            "forcing_parity": a["forcing_parity_changhua"]}


def doc(repo: Path = REPO) -> dict:
    noise = M.noise_params()
    return {
        "id": M.DECL_ID, "wp": "X5", "kind": "declaration",
        "title": "Within-township response of Changhua heads to local census energy on the 74 "
                 "spent calibration screens (screen and township x year fixed effects), "
                 "mirrored on R1c's world, with placebos and a synthetic G0 gate; narrowing "
                 "only, kappa reported never applied",
        "priors": (
            "Without a model, Changhua heads respond about a third as much per GWh as "
            "Yunlin's (-4.2 against -15.3 m per GWh/month, I1_changhua_local) and barely "
            "follow the seasons; R1c's Changhua head lever equals Yunlin's. Between-place and "
            "county-mean designs are confounded by drought years (energy -0.83, canal +0.83). "
            "This declaration asks the within-township question: when local census energy "
            "changes more at one site than at another site of the same township in the same "
            "year, does the head there fall more, and by how much relative to R1c's own "
            "within-township response? Changhua identification is thin: 29 sites in 19 "
            "townships, of which 5 townships hold two or more sites (15 sites), counted from "
            "station metadata only. G0 decides, before any head is read, whether the design "
            "can answer at all."),
        "rulings": {
            "2026-10-29 user (4)": "keep 'no calibrated correction factor follows': kappa is "
                                   "reported, never applied",
            "2026-10-29 user (second set)": "N2C page mapping: the Changhua head lever stays "
                                            "'untested' until N2C is scored; nothing here "
                                            "shows a Changhua lever",
            "judge round 7 rank 16": "showing kappa on the page would need a ruling; this "
                                     "declaration maps to the ledger and README Limits only"},
        "design": {
            "unit": "screen-year, 2012..2022 (calendar years)",
            "screens": f"the 74 Changhua rows of {M.WELLS} (sha256 {M.WELLS_SHA}): cell and "
                       "layer frozen; the 84 Yunlin rows enter the report-only parity reading",
            "head_obs": f"{M.HEADS}: annual mean of monthly 'value' with flag in "
                        f"{list(T.FLAGS_SCORED)}, at least {T.MIN_MONTHS_PER_YEAR} months a "
                        "year, else missing (allowed use design_time_evidence)",
            "head_model": "baseline scenario's heads_mean at the screen's (layer, cell), "
                          "hindcast months 2012-01..2022-12 of the stored forward (keys read: "
                          "dates, origin, scenario_names, heads_mean); for kappa the annual "
                          "mean over exactly the months the observation uses in that "
                          "screen-year (flags 0-1, finite), on the observed screen-years; "
                          "for G0 the complete 12-month annual mean",
            "regressor": f"local census energy, GWh/yr: sum over in-grid kept pumps of "
                         f"{M.PUMPS_DIR}/pumps_*.npz (annual sums, missing months as zero) "
                         f"weighted exp(-d^2 / 2h^2), h = {T.H_KM_PRIMARY} km, truncated at "
                         f"{T.TRUNC_H} h, d from the screen's station coordinates "
                         f"({M.STATION_META}) to the pump's census_L1 v3 coordinates "
                         f"({M.CENSUS_L1}; allowed uses design_time_evidence and model_design); "
                         f"h = {T.H_KM_REPORT} km report-only",
            "fixed_effects": "screen + township x year (township = station_meta_preorigin "
                             "TownName, else poly_town; prefixed by county)",
            "estimator": "FWL slope x~'y / x~'x~, x~ the exact least-squares residual of x "
                         "on both effects (demean by township-year, then project out the "
                         "demeaned screen dummies, minimum-norm)",
            "interval": f"{int(T.LEVEL * 100)} % percentile interval, site-cluster bootstrap "
                        f"stratified by township, B {T.B_BOOT}, seed {T.SEED}; township-"
                        "cluster bootstrap report-only",
            "kappa": "beta_obs / beta_model per bootstrap draw on the same plan (paired); "
                     "percentile interval; draws with |beta_model| <= 1e-12 dropped and counted",
            "models": {"R1c": M.MODELS["R1c"] + " (primary for kappa; N2C's forward)",
                       "ccz_m1618": M.MODELS["ccz_m1618"] + " (report-only; the step C masked "
                                                            "refit, X5.changhua_lever's forward)",
                       "q0c": M.MODELS["q0c"] + " (report-only)"},
            "placebos": {
                "shift": f"regressor shifted +/-{T.SHIFT_YEARS} years (x at y + 2, y - 2); "
                         "literal: the shifted regressor alone; conditional: with the "
                         "contemporaneous regressor as a control (FWL); the mode is fixed by "
                         "G0, never by the observed heads",
                "permutation_report_only": (
                    f"{T.N_PERM} uniform within-township permutations of the site -> "
                    "regressor-series assignment (identity included): pooled slope, median "
                    "slope and p-value REPORTED, never gating. It cannot fail on this design: "
                    "the pooled numerator is M(sum_p x_p)'y with sum_p x_p constant within a "
                    "township-year (absorbed by the fixed effects, ~0 for any y), and in a "
                    "2-site township the only non-identity permutation is the swap, x~ -> "
                    "-x~ (5 identifying townships, mostly 2-3 sites)"),
                "pass": "each shifted-placebo slope in absolute value below the half-width of "
                        "the primary interval"},
            "design_check": (
                "after the heads are read, before R1c's response and the placebos: the "
                "observed screen-years must keep every identifying township and site of G0's "
                "complete panel, at least "
                f"{T.MIN_OBS_IDENT_ROW_SHARE:g} of its identifying screen-years (G0's "
                "report-only mcar25 missing share), a finite slope and at most "
                f"{T.MAX_DEGENERATE_FRAC} degenerate bootstrap draws; else "
                "not_identifiable_design"),
            "kappa_bands": (
                "the paired kappa interval reads: entirely below 0 -> opposite_sign (step B "
                f"saw the opposite sign once); covering both {T.KAPPA_ALT} and 1 (or not "
                "finite) -> uninformative; else narrowing"),
            "pre_declared_alternative": "placebo mode: literal if its G0 pass rate >= "
                                        f"{T.G0_PLACEBO_PASS_MIN}, else conditional if its rate "
                                        f">= {T.G0_PLACEBO_PASS_MIN}, else none (G0 fails)"},
        "g0": {
            "world": "R1c's: y = H_R1c + e, truth beta_model = the estimator on noise-free "
                     "H_R1c (complete 74 x 11 panel)",
            "noise": {**{k: noise[k] for k in ("sigma_m", "block_share", "phi")},
                      "model": "monthly AR(1) with a 5 km block component (bootstrap_scorer "
                               "block_ids), stationary start, annual means",
                      "sources": noise["sources"]},
            "arms": {
                "base": f"{T.G0_DRAWS} draws: coverage of beta_model; coverage of kappa = 1 "
                        "by the paired kappa interval (the model column H_R1c in the same "
                        "bootstrap draws); shifted-placebo pass rates",
                "kappa033": f"y = H - (1 - {T.KAPPA_ALT}) beta_model x + e, {T.G0_DRAWS} "
                            f"draws: power; coverage of kappa = {T.KAPPA_ALT}",
                "persistent": f"y = H + annual AR(1) {T.PERSISTENT_PHI_ANNUAL} drift with the "
                              "declared block share and the annual sd of the declared monthly "
                              f"model, {T.G0_DRAWS} draws: coverage of beta_model (gating: "
                              "monthly AR(1) phi 0.859 decorrelates within a year)",
                "ramp": f"the estimator sees x (1 - r(y)) at floor({T.RAMP_SITE_SHARE:g} n) (at "
                        "least 1) random SITES inside every township of n >= 2 sites, r = 0 to "
                        f"{T.RAMP_FRAC_2022:g} linearly {T.RAMP_START_YEAR}..2022 (a "
                        "site-specific census bookkeeping ramp: install dates frozen mid-2016 "
                        "miss growing sites; a township-wide factor would be absorbed by the "
                        "township x year effect); heads respond to the true x; coverage",
                "report_only": f"{int(T.MCAR_DROP * 100)} % screen-years missing at random; "
                               "the permutation placebo's pass rate; kappa power; the median "
                               "kappa width"},
            "gate": {"coverage_base_min": T.G0_COVERAGE_MIN,
                     "coverage_ramp_min": T.G0_COVERAGE_MIN,
                     "coverage_persistent_min": T.G0_COVERAGE_MIN,
                     "kappa_coverage_1_min": T.G0_COVERAGE_MIN,
                     "kappa_coverage_033_min": T.G0_COVERAGE_MIN,
                     "power_min": T.G0_POWER_MIN,
                     "power": "min(P[CI excludes 0.33 beta_model | kappa 1], P[CI excludes "
                              "beta_model | kappa 0.33])",
                     "placebo": "a shifted-placebo mode calibrated (above); its false-failure "
                                "rate 1 - pass rate (up to "
                                f"{1 - T.G0_PLACEBO_PASS_MIN:.0%}) is reported",
                     "degenerate_bootstrap_max": T.MAX_DEGENERATE_FRAC},
            "seed": T.SEED + 2},
        "readings": T.READINGS,
        "reading_order": [
            "G0 fails -> not_identifiable_g0; no head is read",
            "the observed design check fails -> not_identifiable_design",
            "R1c's own interval on the observed screen-years includes 0 -> "
            "not_identifiable_model",
            "a shifted placebo fails -> not_identifiable_placebo (kappa stays in report.json "
            "only)",
            "else the kappa band of R1c's paired kappa interval: opposite_sign, uninformative "
            "or narrowing"],
        "code_sha256": M.code_sha256(repo),
        "frozen_inputs": frozen_inputs(repo),
        "census_audit_coupling": {
            **audit_coupling(repo),
            "rule": (
                "registration needs: (a) rank 8's annual_sum_max_rel_change (Changhua) <= "
                "0.01 (the regressor uses annual sums; the re-split moves energy across years "
                "only through Dec-Jan pairs), else a new declaration with the re-split "
                "regressor replaces this one; (b) forcing_parity_changhua within its declared "
                "tolerance (the regressor's preorigin v3 in-grid energy matches R1c's "
                "model_census v1 kept in-grid forcing); (c) D1b = cap_immaterial: preorigin v3 "
                "holds only kept pumps, so this design cannot carry the cap-1.5 member D1b "
                "would require, and registration is refused (a v2 would build that regressor "
                "from model_census v1). D1 = resplit_product_required does not block: the "
                "regressor is annual (condition (a)) and R1c's stored forward is the frozen "
                "world mirrored, whatever its forcing; the D1 outcome is recorded in this "
                "block of the declaration")},
        "zones": "Changhua county screens (74); Yunlin's 84 report-only",
        "burn_in": NA("annual panel of a free-running hindcast; no burn-in"),
        "folds": NA("design-time declaration on spent screens; no folds"),
        "gate_scores": NA("declared G0 thresholds and reading order; no weighted score"),
        "primary_model": "R1c (stage3_datum_gate_ccz; N2C's primary), its stored forward "
                         "results/twin_forward/datum_gate_ccz.npz",
        "inputs": ([{"path": p, "sha256": h} for p, h in M.code_sha256(repo).items()]
                   + [THIS, TESTS, M.STATION_META, M.CENSUS_L1, M.HEADS,
                      *(f"{M.PUMPS_DIR}/{f}" for f in M.PUMP_FILES), *M.MODELS.values(),
                      *FROZEN] + ([M.AUDIT] if (repo / M.AUDIT).exists() else [])),
        "tests": NA("design-time declaration: the 74 Changhua and 84 Yunlin screens are "
                    "spent calibration wells; no fresh unit exists or is consumed"),
        "fresh_units": NA("no fresh unit; the score step logs a design_time_evidence look on "
                          "the 158 screens, 2012-01..2022-12"),
        "thresholds": {"g0": "see g0.gate", "placebo": "see design.placebos.pass",
                       "level": T.LEVEL, "B": T.B_BOOT},
        "evidence_and_ledger_mapping": {
            "narrowing": "ledger + README Limits: 'Within-township design-time estimate "
                         "(X5.CH_twfe_mirror.v1, 74 spent Changhua screens, 2012-22): Changhua "
                         "heads respond kappa [lo, hi] of R1c's per GWh of local census energy "
                         "(not a correction)'",
            "opposite_sign": "ledger + README Limits: 'Within-township design-time estimate "
                             "(X5.CH_twfe_mirror.v1, 74 spent Changhua screens, 2012-22): "
                             "Changhua heads respond with the opposite sign to R1c's per GWh "
                             "of local census energy (kappa [lo, hi]; not a correction)'",
            "uninformative": "ledger + README Limits: 'Within-township design-time estimate "
                             "(X5.CH_twfe_mirror.v1, 74 spent Changhua screens, 2012-22): "
                             "uninformative (kappa [lo, hi] of R1c's per GWh covers both 0.33 "
                             "and 1)'",
            "not_identifiable_g0 / _design / _model / _placebo": (
                "ledger + README Limits: 'Within-township design-time estimate: not "
                "identifiable within county (<reason from report.json reading.text>)'"),
            "yunlin_parity": "ledger only, report-only: one line with Yunlin's paired kappa "
                             "interval (X3, site-in-township bootstrap), never README Limits "
                             "and never the page, whatever it shows (Yunlin has no G0, "
                             "placebo or design check here); never suppressed",
            "unchanged": "C13a stays W; N2C, its page mapping and the F cap are untouched; no "
                         "page tile or number; kappa never multiplies any model output"},
        "attempt_policy": {
            "scored_runs": "g0.json once, report.json once (mode 'x'), whatever they show",
            "engineering_retries": "a crash writes nothing and is rerun; every attempt is "
                                   "logged (results/twin_runs/ch_twfe/attempts.jsonl); a fix "
                                   "that changes registered code needs a v2 that discloses "
                                   "the failed attempts",
            "pre_declared_alternatives": ["shifted-placebo mode literal -> conditional, "
                                          "chosen by G0"],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True},
        "interpretation": (
            "kappa near 1: within a township Changhua heads respond to local census energy as "
            "R1c says; kappa near 0.33: they respond about a third as much, as the county-level "
            "data say; either way it is a design-time estimate on spent screens with a census "
            "that carries even-split, duty-cap and frozen-install bookkeeping, and it corrects "
            "nothing. An interval entirely below 0 reads opposite_sign; one covering both "
            "0.33 and 1 reads uninformative. 'Not identifiable within county' is the honest "
            "outcome when 5 townships cannot separate the two. not_identifiable_placebo is "
            "not evidence against R1c: G0 admits a shifted-placebo mode whose pass rate is "
            f"at least {T.G0_PLACEBO_PASS_MIN}, so a correct model world fails it in up to "
            f"{1 - T.G0_PLACEBO_PASS_MIN:.0%} of draws (the realised rate is in g0.json)."),
        "rule_risks_disclosure": [
            "an unlogged 2026-10-25 within-place TWFE on leveling (explorer, design-time) "
            "preceded this declaration; its numbers set nothing here",
            "I1_changhua_local / I1_changhua_interannual / I1_changhua_drivers (after the "
            "fact) published the county-level response (-4.2 against -15.3 m per GWh/month) "
            "and the drought confounding on these same 74 screens; three of their fields "
            "parameterise G0's noise (cited, via n2c_reading's PARAMS)",
            "N1.cpu_response read the Changhua head response to census changes (design-time)",
            "step B (2007-10 leveling) found an observed Changhua E10 coefficient of -0.221 "
            "against the model's +0.165: the within-place census response in Changhua had the "
            "opposite sign once",
            "the 74 screens are R1c's calibration targets: the model was fitted to these heads, "
            "so beta_model and beta_obs are not independent",
            "ccz_m1618's baseline heads at these screens were read by X5.changhua_lever_model "
            "(reproduction checks); R1c's and q0c's forwards are N2C's registered arms",
            "the builder read station metadata (sid, site_id, TownName, poly_town, x, y) of the "
            "158 screens to count townships and sites; no head value",
            "the builder ran the estimator only on synthetic panels (smoke and tests)",
            "review of the build: the within-township permutation placebo was found vacuous "
            "(pooled slope ~0 for any y) and moved to report-only; the township-wide ramp arm "
            "was found absorbed by the fixed effects and replaced by a site-level ramp; both "
            "changes were made before registration, on synthetic panels only",
            "the census carries the bimonthly even split and the duty cap (rank 8 audit) and "
            "an install table frozen mid-2016; the ramp arm stresses the last one"],
        "no_prior_look_statement": (
            "No within-township estimate on the 74 Changhua screens' heads, observed or "
            "modelled, has been computed; the design file and model panels hold inputs only."),
        "written_by": THIS,
    }


def readiness(d: dict, repo: Path = REPO, prereg_dir=None) -> list[str]:
    out = []
    for p, h in d["frozen_inputs"].items():
        if h == "missing":
            out.append(f"{p} missing: bash {M.X}/run_twfe.sh prep")
    cp = d["census_audit_coupling"]
    if not cp.get("present"):
        out.append(f"rank 8 audit {M.AUDIT} missing: run audit_census.py first")
    else:
        if not cp["annual_sum_max_rel_change"].get("within"):
            out.append("rank 8's re-split moves Changhua annual sums by more than 1 %: declare "
                       "with the re-split regressor instead")
        if not cp["forcing_parity"].get("within"):
            out.append("the regressor's preorigin v3 in-grid Changhua energy does not match "
                       "R1c's model_census v1 kept in-grid forcing within the audit's "
                       "tolerance")
        if cp["D1b"] != "cap_immaterial":
            out.append(f"rank 8 D1b = {cp['D1b']}: this design cannot carry a cap-1.5 member "
                       "(preorigin v3 holds kept pumps only); a v2 must build that regressor")
    reg = Path(prereg_dir or prereg3.PREREG_DIR)
    if (reg / f"{M.DECL_ID}.json").exists():
        out.append(f"{M.DECL_ID} is already registered")
    for p in (M.G0_JSON, M.REPORT):
        if (repo / p).exists():
            out.append(f"{p} exists before the declaration")
    return out


def main(argv=None, prereg_dir=None, rows_dir=None, repo: Path = REPO) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="register once (the lead)")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true", help="do not print the JSON")
    a = ap.parse_args(argv)
    d = doc(repo)
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    pend = readiness(d, repo, prereg_dir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    self_sha = M.sha256(Path(__file__))
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("readiness:", pend or "ready to register")
    print("this script sha256:", self_sha)
    if not a.write:
        return 1 if errs else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if errs or pend:
        print("REFUSED:", errs + pend)
        return 1
    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
