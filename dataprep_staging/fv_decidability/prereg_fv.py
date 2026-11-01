"""Declaration FV.decidability.v1 (harness.prereg3, kind 'declaration'), written BEFORE any
truth world is drawn: a synthetic decidability study (OSSE) of E02c's 2028 level and
coverage verdicts. Model-only; no post-origin observation exists or is read, and no fresh
unit is consumed. The study and its readings are in osse_fv.py (bound here by sha256).

User ruling 2026-11-01 (fourth set, rank 11): RUN; P <= 0.5 means 'weak by design' (C30
stays N whatever the 2028 level outcome); one-sided reading (no 'a fail is informative').

Order for the lead (repository root, hydro env; none of this was run by the builder):
  1. osse_fv.py export                 noise + pool inputs (pre-origin real data), once
  2. osse_fv.py export --verify        bit-identical rebuild (verify.json)
  3. osse_fv.py log-look               the design-time look on the residual months
  4. prereg_fv.py --validate           dry run: JSON + sha, readiness, blocking (writes nothing)
  5. prereg_fv.py --write --expect-self-sha <sha of this file>      (registers once)
  6. run_fv.sh arms / forward / score  (GPU forwards under MPS from the frozen worktree)

--write refuses unless: the export exists and verified identical, the look is logged, E02c is
registered and neither E02 nor E02c has a score look, no truth or ZR0 forward, arm file or
report exists, the frozen worktree and py.sh are as declared, the use classification of the
pump product is confirmed (--use-confirmation, recorded in replay.use_confirmation), and
--expect-self-sha matches. The same --use-confirmation text must be given to --validate so
the dry-run sha is the sha written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osse_fv as O  # noqa: E402
from harness import prereg3  # noqa: E402

S = O.S
NA = prereg3.not_applicable
PENDING = "PENDING"
TESTS = f"{O.F}/tests/test_fv_decidability.py"
NOT_BEFORE_E02 = dt.datetime(2028, 1, 15, tzinfo=dt.timezone.utc)
# the runner must carry these lines verbatim (one source of the recipe, checked here)
RUNNER_LINES = (f"ID={O.ID}", f"WT_HEAD={O.WT_HEAD}", f"PY={O.PY_SH}",
                'WT="$HOME/HydroPhysicsAI-x5run"', f'FLAGS="{O.FWD_FLAGS}"',
                f"RUNS={O.RUNS}", f"ARMS={O.ARMS_DIR}", f"REPORT={O.REPORT}",
                "DECL=results/dataprep/preregistration/$ID.json")
RUNNER_LINES += (f"ZR0_DIR={O.ZR0_DIR}", f"ZR0_THETAS=({' '.join(O.ZR0_THETAS)})")
RUNNER_FRAGMENTS = ('"$PY" -m hydrophysics.twin.forward "${THETAS[@]}" $FLAGS',
                    'THETAS=(--theta "$R/$dir/stage3_theta.json")',
                    'THETAS+=(--theta "$R/$dir/$t")',
                    'out=results/twin_forward/fv_zr0',
                    '--vep-json "$R/$dir/coupled_leveling/vep_zonal_leveling.json"',
                    '${SPECS[@]+"${SPECS[@]}"}', '--out "$out" > "$sealed" 2> "$err"',
                    'for t in ZR0 "${TAGS[@]}"; do forward_one "$t"; done')
AFTER_THE_FACT_KNOWN = {
    "head_rmse_ratio_vs_climatology_2023_26": "about 1.22 (results/twin/new_data_challenge_"
                                              "innov; disclosed by E02c)",
    "ensemble_only_coverage_2023_26": "0.75; combined-sigma 0.92-0.93 (U7, E02c)",
    "use": "none of these chose a truth, a noise rule, K, a threshold or a reading here"}


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _rec(p) -> dict:
    return {"path": str(p), "sha256": O.sha256(O.rp(p))}


def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == prereg_id]
    return rows[-1] if rows else None


def score_looks(prereg_ids, prereg_dir=None) -> list[dict]:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    return [lk for lk in prereg3.read_chained(d / "looks.jsonl")
            if lk.get("purpose") == "score" and lk.get("prereg_id") in prereg_ids]


def design_look(noise_sha: str | None, prereg_dir=None) -> dict | None:
    """The logged design-time look on the residual months (osse_fv.py log-look)."""
    if noise_sha is None:
        return None
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    for lk in prereg3.read_chained(d / "looks.jsonl"):
        if (lk.get("units") == [O.LOOK_UNIT] and lk.get("purpose") == O.LOOK_PURPOSE
                and noise_sha in str(lk.get("note", ""))):
            return {k: lk.get(k) for k in ("time", "wp", "purpose", "units", "data_scope")}
    return None


# ----------------------------------------------------------------------------- export slot
def export_block(inputs_dir=O.INPUTS_DIR) -> dict:
    d = O.rp(inputs_dir)
    if not (d / "summary.json").is_file():
        return {"status": PENDING, "dir": str(inputs_dir)}
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    files = {n: O.sha256(d / n) for n in O.INPUT_FILES if (d / n).is_file()}
    out = {"status": "ok", "dir": str(inputs_dir), "files": files,
           "summary_sha256": O.sha256(d / "summary.json"),
           "pool": {k: s.get("pump", {}).get(k) for k in
                    ("pool_years", "frozen_years", "first_frozen_year", "label",
                     "excluded_incomplete_or_flagged", "freeze_rule")},
           "noise": {k: s.get("heads", {}).get(k) for k in
                     ("n_wells", "sd_filled", "phi_filled", "rules", "window_end", "flags")}}
    if len(files) != len(O.INPUT_FILES) or files != s.get("files"):
        out["status"] = "mismatch"
    elif not (d / "verify.json").is_file():
        out["status"] = "not verified"
    else:
        v = json.loads((d / "verify.json").read_text(encoding="utf-8"))
        out["verify"] = {"identical": v.get("identical"), "verified": v.get("verified"),
                         "sha256": O.sha256(d / "verify.json")}
        if v.get("identical") is not True or v.get("files") != files or \
                v.get("summary_sha256") != out["summary_sha256"]:
            out["status"] = "not verified"
    return out


def null_block(export: dict, paths: O.Paths | None = None) -> dict:
    """The declared null-world tolerance, computed from ZR, CL and the noise inputs before any
    draw (osse_fv.null_tolerance)."""
    if export["status"] != "ok":
        return {"status": PENDING, "tolerance": None}
    ctx = O.load_context(paths or O.Paths())
    t = O.null_tolerance(ctx["zr"] - ctx["cl"], ctx["sd"], ctx["phi"], O.K_NULL)
    return {"status": "ok", **t, "gate": "the null world's E02c level false-pass rate (K_NULL "
                                          "worlds, obs = CL + AR(1) noise) must be <= tolerance; "
                                          "otherwise a possible bug: report not_evaluable, no "
                                          "truth world is drawn"}


# ----------------------------------------------------------------------------- the document
def doc(inputs_dir=O.INPUTS_DIR, prereg_dir=None, paths: O.Paths | None = None,
        use_confirmation: str | None = None) -> dict:
    exp = export_block(inputs_dir)
    nul = null_block(exp, paths)
    e02c = index_record(S.ID, prereg_dir)
    frozen = {p: O.sha256(O.rp(p)) for p in O.FROZEN}
    R = "results/twin_runs"
    truths = [{"tag": t, "theta": f"{R}/{d}/stage3_theta.json",
               "column": f"{R}/{d}/coupled_leveling/vep_zonal_leveling.json",
               "forward_out": O.truth_npz(t),
               "role": {"R0": "CWA-rain recipe without Changhua pumping (Yunlin heads lever)",
                        "Q0": "rebuilt-rain recalibration (page-model family, own theta)",
                        "R1c": "census flow with Changhua pumping (ccz)",
                        "X5R1": "irrigation energy-to-volume fix (x5r1)"}[t]}
              for t, d in O.TRUTHS]
    zr0 = {"tag": O.ZR0, "thetas": O.zr0_thetas(), "column": O.zr0_column(),
           "forward_out": O.ZR0_NPZ,
           "role": "same-code reference of the page recipe: the 18 parameter sets of the "
                   "registered run datum_gate_innov (theta, 5 folds, 12 posterior draws, in "
                   "the registered order), run at the frozen worktree with the truths' FLAGS "
                   "(--ic-members 0, horizon 58), baseline only, its coupled_leveling column "
                   "(heads do not depend on the column under --column-heads free). "
                   "D_k = h_k - h_ZR0, so D_k holds the recipe difference only: no code drift "
                   "since the registered run's commit and no IC-member average",
           "checks": "gate E (osse_fv.check_zr0_meta): baseline only, member labels and order "
                     "equal the registered run's, one member per parameter set, E02's "
                     "forward options, no recorded-pumping arm, finite heads at the wells",
           "diagnostic": "REPORT-ONLY, declared now: |h_ZR0 - h_ZR(registered)| at the 158 "
                         "wells, hindcast 2012-22 (code drift) and window datum-removed (adds "
                         "the IC-member average); max and RMS. It gates nothing (no threshold "
                         "could be set without reading model values) and enters no truth "
                         "world"}
    inputs = sorted({*frozen, f"{O.F}/prereg_fv.py", O.E02_PRED, O.E02_PROTOCOL,
                     f"{S.E02_DIR}/manifest.json", S.FORWARD,
                     f"{O.E02C_BASE_DIR}/baselines_e02c.csv", f"{O.E02C_BASE_DIR}/wells_158.csv",
                     f"results/dataprep/preregistration/{S.ID}.json",
                     *[t["theta"] for t in truths], *[t["column"] for t in truths],
                     *zr0["thetas"], zr0["column"],
                     *[p for p, _ in O.REPRO.values()],
                     *[f"{inputs_dir}/{n}" for n in (*O.INPUT_FILES, "summary.json",
                                                     "verify.json")
                       if O.rp(f"{inputs_dir}/{n}").is_file()]})
    readings = {
        "threshold": O.P_WEAK,
        "rule": "P <= 0.5 -> 'weak_by_design'; P > 0.5 -> 'not_shown_weak'. One-sided: no "
                "'a fail is informative' reading exists. Applied once, to the point estimate.",
        "P": "P(pass | right-ish) = equal-weight mean over the four truths of each truth's "
             f"pass rate over K_TRUTH = {O.K_TRUTH} worlds; Monte Carlo SE and per-truth "
             "rates reported, never used to move the threshold",
        "verdicts": {
            "level": "E02c T-level 'level_forecast_supported' (score_e02c.t_level)",
            "coverage_Yunlin": "E02c T-cov 'coverage_supported' for Yunlin (G0 AND gate-3 "
                               "clusters = well; score_e02c.county_coverage + cov_outcome)",
            "coverage_Changhua": "the same for Changhua",
            "e02_own_report_only": "E02's own checks (ratio <= 1 AND pooled coverage in "
                                   "[0.85, 0.95]); reported, no reading"},
        "wording": O.WORDING}
    page = {
        "before_scoring": "E02/E21 line evc_pro_fv (template, ledger rows C30, E02, E21) as "
                          "already drafted: the interpretation is fixed beforehand; no P shown",
        "level_weak_by_design": "evc_pro_fv states P and 'weak by design'; a fail would not "
                                "refute the forecast; C30's heads part stays N whatever E02c's "
                                "2028 level outcome (a T-level pass cannot make it S-limited)",
        "level_not_shown_weak": "evc_pro_fv states P and 'not shown weak'; E02c's own mapping "
                                "for C30 applies unchanged; no other reading",
        "coverage": "per county, the same one-sided wording as an interpretation line beside "
                    "E02c's coverage line; E02c's C14a mapping is unchanged (no ruling covers "
                    "a coverage cap)",
        "not_evaluable": "no interpretation line; E02/E02c read as registered",
        "ledger_slot": "claims_v1.json E02 notes 'SLOT (rank 11)': the reading and P",
        "never": "no head-response claim (C01, C24, C13a) changes; no subsidence claim; "
                 "'a fail is informative' is never written"}
    attempt = {
        "scored_runs": "one forward per truth and one scoring run; the first run that passes "
                       "gate E writes report.json (open mode 'x') and is final, whatever "
                       "gates S and N or the readings say",
        "engineering_retries": "free while gate E fails (crash, OOM, missing or mismatched "
                               "file, metadata, X5R1 hindcast reproduction): nothing is "
                               "written, each attempt is logged in attempts.jsonl, a failed "
                               "attempt's outputs are archived unopened (run_fv.sh archive). "
                               "A fix that changes bound code (frozen_code) needs a new "
                               "version (v2) that discloses this one",
        "s_or_n_failure": "a superposition or null-world gate failure writes a not_evaluable "
                          "report and closes v1; at most ONE follow-up declaration (v2, "
                          "label 'post-failure, exploratory'), which discloses v1",
        "pre_declared_alternatives": [], "exploratory_followups_max": 1,
        "thresholds_frozen": True, "report": "every attempt, failures included"}
    d = {
        "id": O.ID, "wp": O.WP, "kind": "declaration",
        "title": "OSSE: can E02c's 2028 level and coverage verdicts pass if a model of the "
                 "right kind is true? P(pass | right-ish) over four distinct-recipe truths; "
                 "P <= 0.5 reads 'weak by design' (one-sided)",
        "priors": (
            "At leads of 47-58 months with climatological forcing, a correct model differs "
            "from climatology mainly through trend memory, so E02c's strict level rule (beat "
            "CL and CLT with block-bootstrap intervals) may be close to a coin flip even if the "
            "forecast is right; a 2028 fail would then not refute. This study fixes that "
            "interpretation before any truth world is drawn and before E02 or E02c is scored. "
            "Truths are distinct recipes already on disk (structural error stand-ins), not "
            "page-ensemble members. The study is mostly OPTIMISTIC: rain at climatology "
            "(forward has no recorded-rain arm), AR(1) noise independent between wells, every "
            "well-month observed, Changhua pumping at climatology. One element is PESSIMISTIC: "
            "the noise comes from free-run hindcast residuals, which already hold the page "
            "model's structural misfit, and D_k adds a recipe departure on top. P is therefore "
            "not claimed as a strict bound in either direction; the reading is one-sided by "
            "the user's ruling (2026-11-01), not because P bounds anything. For coverage, rain "
            "variability and spatial correlation could move coverage toward or away from "
            "[0.85, 0.95], so no direction is claimed."),
        "truths": truths,
        "zr0": zr0,
        "zr": {"predictions": _rec(O.E02_PRED), "protocol": _rec(O.E02_PROTOCOL),
               "forward": {"path": S.FORWARD, "sha256": S.MODEL_SHA},
               "use": "registered model output, read unchanged; never joined to observations"},
        "truth_world": {
            "formula": "obs = ZR + D_k + R_k(a) + e on 2026-11..2027-10 at E02c's 158 wells",
            "D_k": "(h_k - h_ZR0) in the window minus its mean over the well's observed "
                   "2012-22 months (inputs obs_mask.csv: the months E02's datum offset used), "
                   "at each well's wells_158.csv cell and layer; both terms from the frozen "
                   "worktree, one member per parameter set; model-only, no head value read",
            "R_k": f"superposition of {O.HORIZON} one-month impulse arms (multiplier 1 + "
                   f"{O.DELTA} in one month 2023-01..2027-10, Yunlin cells, forward "
                   "--recorded-pumping ...#yunlin), scaled by a(t) - 1",
            "e": "per-well stationary AR(1), independent between wells",
            "observed": "all 158 wells x 12 months (optimistic)"},
        "replay": {
            "product": f"{O.PUMP} (pump_electricity/preorigin v3)", "use": O.PUMP_USE,
            "slice": "rows 2012-01..2022-12 filtered in code (the product declares no "
                     "time_slicing; its exemptions cover design_time_evidence)",
            "series": O.POOL_SELECT, "flags": list(O.POOL_FLAGS),
            "ratio": "E(y, m) / mean over the POOL years of E(., m): a(t) has pool mean 1 in "
                     "every calendar month, so the anomalies are interannual variability "
                     "around the forward's climatology with no level shift (a truncated or "
                     "trending pool would otherwise add one). Intended; the pool-to-2012-22 "
                     "level ratio is reported only",
            "freeze_rule": "pool = complete years (12 months, flags 0/4) before the first year "
                           "holding a frozen month; a month is frozen when |E(y,m)-E(y-1,m)| "
                           f"<= {O.FREEZE_RTOL} x mean E (year-on-year repeat) or when it lies "
                           f"in a run of >= {O.FREEZE_RUN} consecutive months identical to "
                           "that tolerance (a freeze starting mid-year); labelled '2012-16 "
                           "blocks only' when that year is 2016 or later",
            "blocks": "calendar-year blocks drawn with replacement for 2023..2027 (2027: its "
                      "first 10 months)",
            "export": exp.get("pool"),
            "use_note": "the forward's recorded-pumping MECHANISM carries synthetic "
                        "multipliers drawn from pre-origin statistics; this is not the "
                        "'forcing_recorded_arm' use the MANIFEST forbids (no recorded "
                        "post-origin pumping is used). Read as design_time_evidence (prior "
                        "is also allowed); the product forbids forcing_recorded_arm and "
                        "selection, so this reading needs the lead's or user's "
                        "confirmation before --write (use_confirmation)",
            "use_confirmation": use_confirmation or PENDING},
        "noise": {
            "product": O.L1, "use": O.L1_USE, "window_end": S.ORIGIN,
            "flags": list(O.HEAD_FLAGS),
            "model": f"ZR forward baseline heads_mean at the wells (sha {S.MODEL_SHA[:12]})",
            "rules": {"sd": f"nanstd (ddof 1) of the centred residual, >= {O.MIN_SD_MONTHS} "
                            "months", "phi": f"lag-1 correlation over consecutive observed "
                                             f"months, >= {O.MIN_PHI_PAIRS} pairs, clipped to "
                                             f"{list(O.PHI_CLIP)}",
                      "fill": "county median, else overall median"},
            "export": exp.get("noise"),
            "look": {"unit": O.LOOK_UNIT, "scope": O.LOOK_SCOPE, "purpose": O.LOOK_PURPOSE,
                     "logged": design_look((exp.get("files") or {}).get("noise_ar1.csv"),
                                           prereg_dir) or PENDING}},
        "export": exp,
        "inputs_frozen": exp.get("files") or PENDING,
        "design": {"K_TRUTH": O.K_TRUTH, "K_NULL": O.K_NULL, "SEED": O.SEED,
                   "streams": {"anomaly": [O.SEED, O.STREAM_ANOM, "truth", "draw"],
                               "noise": [O.SEED, O.STREAM_NOISE, "truth", "draw"],
                               "null": [O.SEED, O.STREAM_NULL, 0, "draw"],
                               "selfcheck": [O.SEED, O.STREAM_SELFCHECK, 0, "draw"]},
                   "truth_order": list(O.TAGS), "delta": O.DELTA, "horizon": O.HORIZON,
                   "window": list(S.WINDOW), "check_plan": O.check_plan(),
                   "e02c_constants": {"boot_B": S.BOOT_B, "boot_seed": S.BOOT_SEED,
                                      "block_km": S.BLOCK_KM, "min_wells": S.MIN_WELLS_LEVEL,
                                      "min_k_county": S.MIN_K_COUNTY}},
        "statistics": {
            "code": f"{S.E}/score_e02c.py (t_level, county_coverage, cov_outcome, annotate), "
                    "imported, sha256 bound in frozen_code; G0 = E02's pooled coverage check, "
                    "self-checked against twin.validation.evaluate on two null worlds",
            "baselines": f"E02c's registered {O.E02C_BASE_DIR} (CL/CLT, wells, blocks), sha "
                         "as in E02c.prospective_2026-11.v1"},
        "gates": {
            "E": "engineering, passed completely before any S, N or truth-world number: "
                 "registration, frozen code, inputs, truth and ZR0 forward metadata and arm "
                 "shas, check-draw files, the declared null tolerance recomputed, g0_e02 = "
                 f"twin.validation.evaluate on {O.N_SELFCHECK} self-check worlds (own stream "
                 f"{O.STREAM_SELFCHECK}), X5R1 hindcast within {O.REPRO_TOL_M} m of "
                 "datum_gate_x5r1.npz; Q0 against datum_gate_q0.npz report-only",
            "S": f"superposition within {O.SUP_TOL_M} m (max |basis - full| over the "
                 f"{O.N_CHECK} check forwards, 158 wells, window months)",
            "N": "null-world level false pass <= the declared tolerance (null_world)"},
        "null_world": nul,
        "readings": readings,
        "page_mapping": page,
        "e02c_registration": e02c or PENDING,
        "run": {"worktree": f"$HOME/HydroPhysicsAI-x5run at {O.WT_HEAD} (clean, detached; "
                            "has forward --recorded-pumping), read-only, via "
                            f"{O.PY_SH}",
                "forward": f"{O.PY_SH} -m hydrophysics.twin.forward --theta <truth theta> "
                           f"{O.FWD_FLAGS} --vep-json <truth column> "
                           "--recorded-pumping <each spec of osse_fv.py specs --truth T> "
                           "--out results/twin_forward/fv_truth_<T>",
                "forward_zr0": f"{O.PY_SH} -m hydrophysics.twin.forward "
                               + " ".join(f"--theta {t}" for t in O.zr0_thetas())
                               + f" {O.FWD_FLAGS} --vep-json {O.zr0_column()} --out "
                               + O.ZR0_NPZ[:-4],
                "runner": f"bash {O.RUNNER} arms | forward [ZR0|TAG] | score; sha in "
                          "frozen_code",
                "gpu": f"ZR0 (18 members, baseline only) and four truth forwards of "
                       f"{1 + O.HORIZON}-{2 + O.HORIZON} scenarios x {O.HORIZON} months, one "
                       "member; about 0.6-1.2 GPU-h on the T4 (MPS: two runs in parallel); "
                       "scoring CPU about 5 min"},
        "sealed_until_scoring": "forward stdout (sealed file, sha logged); truth npz values "
                                "are read only by the scorer; the runner prints exit codes "
                                "and sha256s only",
        "frozen_code": frozen,
        "attempt_policy": attempt,
        "rulings": {"2026-11-01 fourth set, rank 11": "RUN; P <= 0.5 = 'weak by design' (C30 "
                                                      "stays N whatever the 2028 level "
                                                      "outcome); one-sided reading"},
        "disclosures": {
            "timing": "written on or after 2026-11-01, the first window month; no window "
                      "observation exists or is read, and the declaration precedes any E02 or "
                      "E02c scoring (not before 2028-01-15); --write refuses after a score "
                      "look on either",
            "after_the_fact_known": AFTER_THE_FACT_KNOWN,
            "builder": "the builder ran only synthetic smokes (osse_fv.py smoke). While "
                       "reading formats it printed the header and first two rows of E02's "
                       "predictions.csv (registered model output, window months, no "
                       "observation), npz metadata of existing forwards (no head value), and "
                       "the label columns of the pump table (county, scope, class, date "
                       "range, flag counts; no energy value). No observation, no truth "
                       "forward and no E02c baseline value was opened",
            "e02c_baselines": "CL/CLT are E02c's frozen pre-origin baselines (registered); "
                              "the truth worlds share the real 2012-22 record",
            "optimism": "rain at climatology; noise independent between wells; full "
                        "availability; Yunlin-only anomalies; truths restart from the same "
                        "observed field at 2022-12",
            "pessimism": "the AR(1) noise is fitted to free-run hindcast residuals, which "
                         "already hold structural misfit; D_k adds a recipe departure on top",
            "tests": {"path": TESTS, "sha256_at_declaration": (
                O.sha256(O.rp(TESTS)) if O.rp(TESTS).is_file() else None),
                      "binding": "disclosure only, not an input: a later test edit does not "
                                 "invalidate this declaration"},
            "review_fixes": "before registration the reviewer's must-fix (D_k against a "
                            "same-code reference ZR0 instead of the registered 36-member "
                            "run) and should-fixes (observed-month anchor, gate order, pool "
                            "normalisation and mid-year freeze, wording without a bound "
                            "claim, use confirmation, tests out of inputs) were applied"},
        "zones": NA("no fit; the frozen forward runs' own zones"),
        "burn_in": NA("no fit; forward origin 2022-12"),
        "folds": NA("synthetic worlds; no folds"),
        "gate_scores": NA("readings by one threshold; no weighted score"),
        "tests": NA("model-only declaration: synthetic worlds, no observation is scored and "
                    "no fresh unit is consumed"),
        "fresh_units": NA("no observation is scored"),
        "primary_model": {"forecast": "ZR, E02's registered predictions (forward cf2c7154)",
                          "truths": list(O.TAGS)},
        "thresholds": {"P_weak": O.P_WEAK, "sup_tol_m": O.SUP_TOL_M,
                       "repro_tol_m": O.REPRO_TOL_M, "null_tolerance": nul.get("tolerance"),
                       "K_TRUTH": O.K_TRUTH, "K_NULL": O.K_NULL},
        "inputs": inputs,
        "interpretation": (
            "A model-only operating characteristic of E02c's verdicts under optimistic "
            "truths. 'weak_by_design' fixes in advance that a 2028 fail would not refute the "
            "forecast (and, for the level verdict, that C30's heads part stays N whatever the "
            "outcome). 'not_shown_weak' adds nothing else: by the user's one-sided ruling a "
            "fail is never read as informative because of this study. P is not a strict "
            "bound (optimistic in rain and well independence, pessimistic in noise that "
            "already holds structural misfit). Nothing here is evidence about the aquifer."),
        "written_by": f"{O.F}/prereg_fv.py",
    }
    return d


# ----------------------------------------------------------------------------- readiness
def runner_problems() -> list[str]:
    p = O.rp(O.RUNNER)
    if not p.is_file():
        return [f"{O.RUNNER} missing"]
    sh = p.read_text()
    lines = set(sh.splitlines())
    out = [f"{O.RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{O.RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    if "/home/" in sh:
        out.append(f"{O.RUNNER} hardcodes a /home path")
    return out


def worktree_problems() -> list[str]:
    wt = Path.home() / "HydroPhysicsAI-x5run"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return [f"worktree $HOME/HydroPhysicsAI-x5run: {err}"]
    out = []
    if head != O.WT_HEAD:
        out.append(f"worktree is at {head}, not {O.WT_HEAD}")
    if dirty:
        out.append("worktree is dirty")
    py = O.rp(O.PY_SH)
    if not py.exists() or f"export PYTHONPATH={wt}" not in py.read_text().splitlines():
        out.append(f"{O.PY_SH} does not pin the worktree")
    return out


def readiness(d: dict, prereg_dir=None, now=None) -> list[str]:
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if d["export"]["status"] != "ok":
        out.append(f"export: {d['export']['status']} (osse_fv.py export, then --verify)")
    if d["noise"]["look"]["logged"] == PENDING:
        out.append("the design-time look is not logged (osse_fv.py log-look)")
    if d["null_world"].get("tolerance") is None:
        out.append("null-world tolerance not computed (needs the export)")
    if d["e02c_registration"] == PENDING:
        out.append(f"{S.ID} is not in the registry")
    if score_looks({S.ID}, prereg_dir) or now >= NOT_BEFORE_E02:
        out.append("E02/E02c scoring has started (or 2028-01-15 passed): too late to declare")
    if index_record(O.ID, prereg_dir):
        out.append(f"{O.ID} is already registered")
    for p in (*[O.truth_npz(t) for t in O.TAGS], O.ZR0_NPZ):
        if O.rp(p).exists():
            out.append(f"{p} exists before the declaration")
    if d["replay"].get("use_confirmation") in (None, "", PENDING):
        out.append("the use classification of pump_electricity/preorigin v3 is not "
                   "confirmed (--use-confirmation '<who, when, reading>')")
    for p in (O.ARMS_DIR, O.REPORT):
        if O.rp(p).exists():
            out.append(f"{p} exists before the declaration")
    for p in d["inputs"]:
        if not O.rp(p).exists():
            out.append(f"input {p} missing")
    return out + runner_problems() + worktree_problems()


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="register once (the lead)")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true", help="do not print the JSON")
    ap.add_argument("--use-confirmation", default=None,
                    help="who confirmed, and when, that the pump product is used as "
                         "design_time_evidence (recorded in replay.use_confirmation)")
    a = ap.parse_args(argv)
    d = doc(prereg_dir=prereg_dir, use_confirmation=a.use_confirmation)
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    pend = readiness(d, prereg_dir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True, default=O._jsonable))
    self_sha = O.sha256(Path(__file__))
    print("draft sha256:", sha_text(json.loads(json.dumps(d, default=O._jsonable))))
    print("validate_prereg:", errs or "ok")
    print("blocking looks (declaration, no tests, no fresh units): none by construction")
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
    path, h = prereg3.write_prereg(json.loads(json.dumps(d, default=O._jsonable)), **kw)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
