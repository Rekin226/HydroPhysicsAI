"""Write E02c.prospective_2026-11.v1 (harness.prereg3, kind 'test') BEFORE the window opens.

The one companion of the frozen E02 head forecast (results/twin/prospective_2026-11, registered
by manifest hash 2026-10-11) on I1's existing unit I1:prospective_heads_2026-11_2027-10_R0wells
(158 wells; the same well-months, so no new unit id). Two tests that only tighten E02:
T-cov (per-county coverage of E02's frozen 90 % interval, harness.coverage3 gate-3, clusters =
well; claim C14a) and T-level (the frozen forecast must beat BOTH pre-origin climatology and
climatology-plus-trend; claim C30, heads part). Rulings: user 2026-10-29, third set (a) and
(b). It binds E02, not N2, and uses no X1 value.

Two slots must be filled before ``--write``:
  --baselines-dir  the output of ``export_baselines.py e02c`` AND its ``--verify`` record
                   (CL/CLT under the declared reference rule and the well table, pre-origin
                   rows only, reproduced bit-identically; hashed here);
  --power-report   the result file of U1.HB.power.v1 (rank 2), which must already be in the
                   registry; its two numbers fix the pass/fail wording (``power_flags``).
                   Declared fallback if rank 2 is not ready before the deadline:
                   ``--power-conservative`` registers with both conservative wordings for
                   both counties (only weakens wording, never a gate).
``--write`` refuses if a slot is missing, if the deadline (2026-10-31T16:00Z) has passed, if
any look blocks the unit, if validation fails, or if --expect-self-sha is not this file's sha.

Usage (repository root):
    python -I dataprep_staging/e02c/prereg_e02c.py --validate [--baselines-dir D] [--power-report F]
    python -I dataprep_staging/e02c/prereg_e02c.py --write --baselines-dir D --power-report F \
        --expect-self-sha <sha256 of this file>          (the lead only; this never logs a look)
    (fallback: --power-conservative in place of --power-report)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_e02c as S  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
# hashed inputs (assert_can_score refuses on any change): this folder's three scripts and the
# judge's bindings (bands.py, coverage3.py, drift_diag.fair_baselines)
FROZEN = (f"{S.E}/score_e02c.py", f"{S.E}/export_baselines.py", f"{S.E}/prereg_e02c.py")
CODE = ("hydrophysics/twin/bands.py", "hydrophysics/twin/drift_diag.py",
        "dataprep_staging/harness/coverage3.py")
# recorded, report-only (a routine edit must not leave E02c unscorable): restored to these
# sha256 from git as an engineering step at scoring time, logged (see 'engineering')
CODE_AT_REGISTRATION = ("hydrophysics/twin/validation.py", "hydrophysics/twin/prospective.py",
                        "hydrophysics/twin/challenge.py", "hydrophysics/twin/coverage.py",
                        "dataprep_staging/harness/coverage2.py",
                        "dataprep_staging/harness/coverage.py",
                        "dataprep_staging/harness/prereg3.py", "dataprep_staging/common/guard.py",
                        "dataprep_staging/common/manifest.py", "dataprep_staging/common/uses.py")
ENGINEERING_RESTORE = ("a git-tracked module listed in code_at_registration_report_only may be "
                       "restored to its registered sha256 (git checkout of that blob) before "
                       "the scoring run as an engineering step, logged in the report; the "
                       "frozen_code and bound code are never edited")
BASELINE_FILES = ("baselines_e02c.csv", "wells_158.csv", "summary.json", "verify.json")
REFERENCES = ("L1_clean_v3", "e02_baseline_aligned")
DESIGN_PHASE_NOTE = (
    "2026-10-29, while this script was being built: one builder pytest run of an earlier "
    "draft reached the scorer's report-only fair-rule branch in real mode, which loaded "
    "heads/L1_clean v3 pre-origin rows (2012-01..2022-12, use 'baseline' through "
    "common.guard, allowed by its MANIFEST) into the test process. No value was printed or "
    "inspected, the temporary output was deleted, and no row after 2022-12 and no window "
    "head was touched. The tests now block every product open. No unit is spent.")
U7_LABEL = {
    "en": "the interval family was chosen after its 2023-26 coverage was seen; the "
          "2026-11..2027-10 window is untouched",
    "zh": "此區間形式是在看過其 2023-26 年涵蓋率之後選定的；2026-11..2027-10 "
          "的檢驗期間未被觸及",
}
PENDING = "PENDING"
CONSERVATIVE = "conservative fallback"


def _sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _rec(p) -> dict:
    p = Path(p)
    p = p if p.is_absolute() else REPO / p
    return {"path": rel_to_repo(p), "sha256": S.sha256(p)}


def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == prereg_id]
    return rows[-1] if rows else None


def i1_unit() -> dict:
    """I1's own member declaration of the unit, copied verbatim (never redeclared differently);
    its ids must equal the E02 prediction wells."""
    import pandas as pd

    d = json.loads((REPO / f"results/dataprep/preregistration/{S.I1_PREREG}.json")
                   .read_text(encoding="utf-8"))
    u = d["tests"][0]["external_units"][S.UNIT]
    ids = sorted(pd.read_csv(REPO / S.E02_DIR / "predictions.csv",
                             dtype={"station_id": str})["station_id"].unique())
    if sorted(map(str, u["ids"])) != ids or len(ids) != 158:
        raise SystemExit("I1's unit members differ from E02's 158 prediction wells")
    return dict(u)


# ----------------------------------------------------------------------------- power slot
def power_flags(report_path: Path | None, conservative: bool = False) -> dict:
    """Per-county wording flags from the U1.HB.power.v1 result file (rank 2), in the format
    ``dataprep_staging/u1_bands/power_u1hb.py`` writes: top-level ``id`` == U1.HB.power.v1,
    ``smoke`` False, ``engineering_failure`` False, and ``wording[<county>]`` with
    ``false_fail_max.value`` (false-fail rate at a true 0.90, worst cell),
    ``power_min_at_0.75.value`` (power at 0.75, ICC_month <= 0.15, weakest cell) and the
    declared ``fail_wording`` / ``pass_wording`` keys.

    Declared rules (rank 2): false-fail > 0.10 -> a FAIL adds 'outside the registered range;
    may reflect one year's shock'; power < 0.8 -> a PASS reads 'no gross under-coverage
    detected', never 'calibrated'. A county whose wording the report leaves 'not
    determined' gets both conservative flags. Without a report (validate only) the slot is
    PENDING with the conservative flags; --write refuses."""
    cons = {"fail_may_reflect_shock": True, "pass_no_gross_only": True}
    if report_path is None and conservative:
        return {"status": CONSERVATIVE, **{c: dict(cons) for c in S.COUNTIES_COV},
                "note": f"declared fallback: no {S.POWER_ID} result before the deadline; a "
                        "PASS reads 'no gross under-coverage detected' and a FAIL adds 'may "
                        "reflect one year's shock' for both counties; only the wording is "
                        "weakened, no gate changes"}
    if report_path is None:
        return {"status": PENDING, **{c: dict(cons) for c in S.COUNTIES_COV},
                "note": "no power report yet; --write refuses without one (or without "
                        "--power-conservative)"}
    r = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if r.get("id") != S.POWER_ID:
        raise SystemExit(f"{report_path} does not name {S.POWER_ID}")
    if r.get("smoke") is not False or r.get("engineering_failure") is not False:
        raise SystemExit(f"{report_path}: a smoke run or an engineering failure cannot fix "
                         "the wording")
    w = r.get("wording")
    if not isinstance(w, dict):
        raise SystemExit(f"{report_path}: no 'wording' block")
    out: dict = {"status": "from U1.HB.power.v1", "declaration": r.get("declaration"),
                 "rules": "false-fail > 0.10 at a true 0.90 -> FAIL wording adds 'may reflect "
                          "one year's shock'; power < 0.8 at 0.75 (ICC_month <= 0.15) -> PASS "
                          "wording 'no gross under-coverage detected'"}
    for c in S.COUNTIES_COV:
        rec = w.get(c.lower())
        if not isinstance(rec, dict):
            raise SystemExit(f"{report_path}: no wording for {c}")
        ff = (rec.get("false_fail_max") or {}).get("value")
        pw = (rec.get("power_min_at_0.75") or {}).get("value")
        f = {"false_fail_at_090": ff, "power_at_075": pw,
             "fail_wording_key": rec.get("fail_wording"),
             "pass_wording_key": rec.get("pass_wording"),
             "fail_may_reflect_shock": rec.get("fail_wording") != "fail_plain",
             "pass_no_gross_only": rec.get("pass_wording") != "pass_plain"}
        if isinstance(ff, (int, float)) and (ff > 0.10) != (rec.get("fail_wording")
                                                          == "fail_shock"):
            raise SystemExit(f"{report_path}: {c} fail wording disagrees with its number")
        if isinstance(pw, (int, float)) and (pw < 0.8) != (rec.get("pass_wording")
                                                         == "pass_weak"):
            raise SystemExit(f"{report_path}: {c} pass wording disagrees with its number")
        out[c] = f
    return out


def baselines_block(bdir: Path | None) -> dict:
    """The frozen export: every file hashed; status 'ok' only with the declared reference
    decision and a verify.json showing a bit-identical rebuild of the same files."""
    if bdir is None:
        return {"status": PENDING, "dir": None, "files": {}}
    bdir = Path(bdir)
    files = {n: _rec(bdir / n) for n in BASELINE_FILES if (bdir / n).is_file()}
    out = {"status": "ok" if len(files) == len(BASELINE_FILES) else "incomplete",
           "dir": rel_to_repo(bdir), "files": files}
    if (bdir / "summary.json").is_file():
        s = json.loads((bdir / "summary.json").read_text(encoding="utf-8"))
        bad = [n for n, h in (s.get("files") or {}).items()
               if n in files and files[n]["sha256"] != h]
        out["summary_mismatch"] = bad
        out["reference"] = s.get("reference")
        out["cell_check"] = s.get("cell_check")
        if bad:
            out["status"] = "mismatch"
        elif (s.get("reference") or {}).get("chosen") not in REFERENCES:
            out["status"] = "no reference decision"
    if out["status"] == "ok":
        v = json.loads((bdir / "verify.json").read_text(encoding="utf-8"))
        csv = {n: files[n]["sha256"] for n in files if n.endswith(".csv")}
        out["verify"] = {"identical": v.get("identical"), "verified": v.get("verified")}
        if v.get("identical") is not True or v.get("files") != csv or \
                v.get("summary_sha256") != files["summary.json"]["sha256"]:
            out["status"] = "not verified"
    return out


# ----------------------------------------------------------------------------- the document
def doc(baselines_dir: Path | None = None, power_report: Path | None = None,
        rehearsal: bool = False, prereg_dir=None, power_conservative: bool = False) -> dict:
    unit = i1_unit() if not rehearsal else {"description": "rehearsal", "ids": [],
                                            "window": "2026-11..2027-10"}
    power = power_flags(power_report, conservative=power_conservative)
    base = baselines_block(baselines_dir)
    prec = index_record(S.POWER_ID, prereg_dir)
    nb = index_record(S.NO_BAND_ID, prereg_dir)
    e02 = {n: _rec(f"{S.E02_DIR}/{n}") for n in ("predictions.csv", "protocol.json",
                                                  "manifest.json")}
    sec = {k: {p.name: _rec(p) for p in sorted((REPO / d).iterdir()) if p.is_file()}
           for k, (d, _) in S.SECONDARIES.items()}
    frozen = {p: S.sha256(REPO / p) for p in FROZEN if (REPO / p).is_file()}
    code = {p: S.sha256(REPO / p) for p in CODE}
    code_reg = {p: S.sha256(REPO / p) for p in CODE_AT_REGISTRATION}
    inputs = sorted({r["path"] for r in e02.values()}
                    | {r["path"] for v in sec.values() for r in v.values()}
                    | {S.FORWARD, f"results/dataprep/preregistration/{S.I1_PREREG}.json"}
                    | set(frozen) | set(code)
                    | {r["path"] for r in base["files"].values()}
                    | ({rel_to_repo(Path(power_report).resolve())} if power_report else set()))
    ext_unit = {S.UNIT: unit}
    common = {"data_scope": S.DATA_SCOPE, "fresh_units": [S.UNIT],
              "external_units": ext_unit, "seed": S.BOOT_SEED,
              "label": "prospective: the window's heads did not exist at registration"}
    t_cov = {
        "test_id": S.TEST_COV, **common,
        "claim": "C14a (U1.no_band.v1 scheme): the per-well registered 90 % head interval "
                 "at the monitoring wells, in the window",
        "what": "Does E02's frozen combined-sigma central 90 % interval (lower_m, upper_m of "
                "predictions.csv, sha 16ac3c37) cover the window heads, per county, under a "
                "cluster-aware gate?",
        "metric": {
            "observations": "exactly E02's step-(5) rows: evaluation.csv and report.json "
                            "written by 'python -m hydrophysics.twin.prospective score' on "
                            "results/twin/prospective_2026-11 (challenge.monthly_observations "
                            "per well: sentinels rejected, >= 12 h/day on >= 20 days/month; "
                            "wells with >= 6 window months), --new-wells inside the window "
                            "product whose MANIFEST grants prospective_scorer; E02c reads no "
                            "heads table of its own. Engineering failures (never verdicts): "
                            "report.json without E02's registered manifest, any evaluation "
                            "row whose predicted_m/lower_m/upper_m/baseline_m differs from "
                            "predictions.csv (sha 16ac3c37), or twin.validation.evaluate on "
                            "that evaluation.csv not reproducing report.json (checks, passed, "
                            "rmse_ratio, coverage, stations, rows, predictions_sha256)",
            "hits": "lower_m <= observed_m <= upper_m on those rows; county from the station "
                    "code (07 Changhua, 09 Yunlin)",
            "G0": "checks.interval_coverage ([0.85, 0.95]) of E02's step-(5) report.json, read "
                  "unchanged (its sha256 recorded) and conjunctive; not evaluable if E02's "
                  "enough_stations or enough_months_per_station check fails",
            "G1": "harness.coverage3.coverage_gate (rule gate-3, interval upper90), clusters = "
                  "well, per county, target 0.90, level 0.95, point window [0.85, 0.95] when "
                  "n_eff_lower >= 30; reported: coverage, n, n_eff, n_eff_lower, k_clusters, "
                  "rho, calibrated",
            "G1b": "month clusters (k = 12), coverage3.coverage_gate and "
                   "bands.coverage_report, report-only, labelled 'uncalibrated (k = 12 < 20)'",
            "G2": "miss_below <= 0.10 AND miss_above <= 0.10 per county; gates only the word "
                  "'balanced'"},
        "decision_rule": {
            "coverage_supported": "for a county: G0 AND that county's G1 pass",
            "pooled_gate_failed_county_report_only": "G0 fails while the county's G1 passes: "
                                                     "'registered pooled gate failed; <county> "
                                                     "stratum within 0.85-0.95 (report-only)' "
                                                     "when its coverage lies in 0.85-0.95; "
                                                     "declared variant when gate-3 passed "
                                                     "without the point window (n_eff_lower "
                                                     "< 30) and the coverage lies outside: "
                                                     "'registered pooled gate failed; "
                                                     "<county> stratum coverage X passed "
                                                     "gate-3 outside 0.85-0.95 (n_eff_lower "
                                                     "< 30; report-only)'",
            "coverage_not_supported": "the county's G1 fails (coverage stated, side stated)",
            "not_evaluable": f"G0 not evaluable, or the county has fewer than "
                             f"{S.MIN_K_COUNTY} wells with >= 6 months (gate-3 envelope)",
            "per_county": "Yunlin decides only Yunlin wording, Changhua only Changhua "
                          "wording; no E02c statistic (Changhua or pooled) before N2C steps "
                          "(1) and (2) (session order; the scorer refuses until then)"},
        "informative_gate": "E02's protocol: >= 100 stations, each >= 6 months (G0); per "
                            f"county k >= {S.MIN_K_COUNTY} wells"}
    t_level = {
        "test_id": S.TEST_LEVEL, **common,
        "claim": "C30, heads part: the frozen 2022-12 level forecast",
        "what": "Does the frozen forecast (E02 predicted_m) beat BOTH the pre-origin "
                "climatology CL and climatology-plus-trend CLT on the window?",
        "metric": {
            "rmse": "equal-well RMSE (E02's convention: sqrt of the mean over wells of each "
                    "well's mean squared error), over E02's step-(5) well-months with finite "
                    "CL and CLT; wells with >= 6 such months",
            "baselines": "CL_L1, CLT_L1 = twin.drift_diag.fair_baselines (clim, clim_trend; "
                         "month_offset 0) of heads/L1_clean/v3 flags 0-1 opened with use "
                         "'baseline' and window_end 2022-12 (pre-origin months only). Declared "
                         "reference rule (export_baselines.reference_rule, pre-origin only, "
                         "decided before registration and recorded under 'baselines'): if the "
                         "median over wells of mean_window |CL_L1 - E02 baseline_m| <= 0.05 m, "
                         "CL = CL_L1 and CLT = CLT_L1; otherwise CL = E02 baseline_m and CLT = "
                         "baseline_m + (CLT_L1 - CL_L1), so ZR and the baselines share E02's "
                         "datum. Frozen in baselines_e02c.csv, rebuilt bit-identically "
                         "(verify.json) and hashed here; PAR dropped",
            "interval": f"paired bootstrap of MSE(ZR) - MSE(baseline): wells resampled by "
                        f"{S.BLOCK_KM:g} km TWD97 blocks (wells_158.csv block5), B "
                        f"{S.BOOT_B}, seed {S.BOOT_SEED}, central {S.BOOT_CI:.0%}; declared "
                        "secondary: months resampled (seed + 1), report-only",
            "report_only": ["E02's own ratio verdict, unchanged (twin.validation.evaluate)",
                            "drift_diag.fair_temporal_verdict (K = 1.25, constants of "
                            "2026-09-24) through a gap adapter (2023-01..2026-10 passed as "
                            "NaN, never loaded); can never move C30 to S",
                            "per-county RMSEs",
                            "secondaries (hybrid, r0, n2c_r1c; bound below): per-county "
                            "coverage, gate-3 (clusters = well) and equal-well RMSE against the "
                            "same CL / CLT on E02's scored well-months; never gate, never "
                            "substitute; n2c_r1c only after N2C step (2)"]},
        "decision_rule": {
            "level_forecast_supported": "RMSE(ZR) < RMSE(CL) AND RMSE(ZR) < RMSE(CLT) AND "
                                        "both block-bootstrap 90 % intervals of the MSE "
                                        "difference lie below 0",
            "level_forecast_not_supported": "otherwise",
            "not_evaluable": f"fewer than {S.MIN_WELLS_LEVEL} wells with >= {S.MIN_MONTHS} "
                             "months, or no window product granting prospective_scorer"},
        "informative_gate": f">= {S.MIN_WELLS_LEVEL} wells with >= {S.MIN_MONTHS} months"}
    disclosures = {
        "U7_interval_family": "E02's combined-sigma interval family was chosen on 2026-10-11 "
                              "after its 2023-26 after-the-fact coverage (0.92-0.93 at a "
                              "nominal 0.90) had been seen (hydrophysics/twin/prospective.py "
                              "docstring). User ruling 2026-10-29 (third set, a): it may be "
                              "labelled 'decided by a registered prospective test' WITH this "
                              "disclosure on the page, en and zh: " + json.dumps(
                                  U7_LABEL, ensure_ascii=False),
        "after_the_fact_numbers_known": {
            "ensemble_only_coverage_2023_26": "0.75 (results/twin/coverage_datum_gate_innov)",
            "hybrid_picp90": "0.877 (prospective_2026-11_hybrid, after the fact)",
            "head_rmse_ratio_vs_climatology_2023_26": "about 1.22 "
                                                      "(results/twin/new_data_challenge_innov)",
            "use": "none of these chose a gate, a threshold, a well or a baseline here; the "
                   "baselines are the fixed fair_baselines pair used since 2026-09"},
        "multiplicity": "the unit already carries about 7 registered comparisons (E02, the "
                        "hybrid manifest, I1 T1/T2, X1 T3a/T3b, N2, N2C on its Changhua "
                        "subset); E02c adds two; each decides only its own claim, no "
                        "family-wise claim is made",
        "c14_run_spread": "the page's run-spread band (bn_head, rng_*) is a different "
                          "interval: 'not tested by E02 (a different interval)'",
        "no_scorer_product_yet": "no held heads product grants prospective_scorer "
                                 "(heads/L1_clean v3 forbids it); scoring needs a window "
                                 "product whose MANIFEST lists it, flags 0-1",
        "scope": "only tightens E02 (AND with E02's checks); binds E02, not N2; no X1 use",
        "design_phase_engineering_note": DESIGN_PHASE_NOTE,
        "u1_no_band_order": "the judge lists U1.no_band.v1 (rank 3) as registered first; it is "
                            "recorded under 'no_band_declaration' but not required, so a late "
                            "rank 3 cannot cost this window; the C14a-e scheme is declared here"}
    page_mapping = {
        "before_scoring": {
            "C14a": "N+D, text only: 'registered interval; coverage decided not before "
                    "2028-01-15'; the evidence lines name the registered per-well 90 % "
                    "interval; no aggregate half-width summary is shown and no station ids; "
                    "per-well bands only under --private",
            "E02": "stays D, with the U7 label (en/zh above) on its coverage line",
            "C30_heads": "N with D sub-label 'strict level rule E02c, pending'",
            "C14_run_spread": "'not tested by E02 (a different interval)'",
            "engineering": "the page build refuses to emit any interval text unless its "
                           "forward sha256 == cf2c7154 (model_sha256 of the E02 manifest) and "
                           "the predictions and protocol sha256 match 16ac3c37 / 8a96c0ea"},
        "T_cov": {"coverage_supported": "C14a -> S (prospective) for that county; wording "
                                        "'passed its registered gate', never 'calibrated'; "
                                        "'balanced' only if G2 passes",
                  "pooled_gate_failed_county_report_only": "C14a stays N for that county "
                                                           "with the report-only wording",
                  "coverage_not_supported": "C14a -> N with the coverage stated; the "
                                            "half-width summary and 'predictive interval' "
                                            "wording are withdrawn; nothing is widened",
                  "not_evaluable": "C14a -> N, 'undecided'",
                  "wording": {k: v for k, v in S.WORDING.items() if k.startswith("cov_")}},
        "T_level": {"level_forecast_supported": "C30 heads part -> S-limited ('beat "
                                                "climatology and trend on 2026-11..2027-10')",
                    "level_forecast_not_supported": "C30 stays N with the decided reason",
                    "not_evaluable": "C30 stays N, 'undecided'",
                    "wording": {k: v for k, v in S.WORDING.items()
                                if k.startswith("level_")}},
        "power_dependence": f"wording fixed by {S.POWER_ID}: see 'power'",
        "never": "no subsidence claim, no policy-response claim, no claim about N2's or "
                 "N2C's verdicts follows from either test"}
    d = {
        "id": S.ID, "wp": S.WP, "kind": "test",
        "title": "E02c: per-county coverage (gate-3) and strict level rule for the frozen E02 "
                 "head forecast, 2026-11..2027-10",
        "why": "E02's only checks are a row-pooled coverage window and an RMSE ratio to "
               "climatology; neither is clustered, split by county, or compared with a trend "
               "baseline. User ruling 2026-10-29 (third set, b) approves one companion on "
               "I1's unit that only tightens E02.",
        "tests": [t_cov, t_level],
        "binds": {"e02": e02, "e02_model_sha256": S.MODEL_SHA,
                  "e02_expected": {"predictions.csv": S.E02_PRED_SHA,
                                   "protocol.json": S.E02_PROTOCOL_SHA},
                  "forward": {"path": S.FORWARD, "sha256": S.MODEL_SHA},
                  "secondaries_report_only": sec,
                  "code": code,
                  "code_at_registration_report_only": code_reg,
                  "engineering": ENGINEERING_RESTORE},
        "baselines": base,
        "power": {"id": S.POWER_ID,
                  "registration": prec or PENDING,
                  "report": (_rec(Path(power_report).resolve()) if power_report else
                             (CONSERVATIVE if power_conservative else PENDING)),
                  "flags": power},
        "no_band_declaration": {"id": S.NO_BAND_ID, "registration": nb or "not registered "
                                "at drafting; the C14a-e scheme above is used as declared"},
        "frozen_code": frozen,
        "scoring": {
            "not_before": "2028-01-15", "no_later_than": S.NO_LATER_THAN,
            "precondition": "the window heads product MANIFEST lists prospective_scorer "
                            "(flags 0-1), checked from the MANIFEST alone (no table read); "
                            "otherwise both tests are not_evaluable, nothing is read and no "
                            "look is logged",
            "run_order": "(1) MANIFEST check; (2) N2C order check; (3) "
                         "prereg3.assert_can_score (logs the score looks); (4) only then "
                         "E02's step-(5) rows are read",
            "same_session_order": list(S.SESSION_ORDER),
            "if_order_breached": "E02c computes and reads nothing (Changhua or pooled) and "
                                 "logs no look until N2C's step-(1) score look is in "
                                 "looks.jsonl AND N2C's step-(2) report (a JSON naming "
                                 f"{S.N2C_ID}) exists; the scorer refuses until then. If N2C "
                                 "is refused or never scorable, the lead writes a record "
                                 "naming N2C with verdict 'not_scorable', and E02c is scored "
                                 "in the first session after; the delay is reported",
            "engineering": ENGINEERING_RESTORE,
            "scorer": f"{S.E}/score_e02c.py score --window-product <dir> --e02-score-dir "
                      "<E02 step-(5) out> --e02-new-wells <dir inside the product> "
                      "--n2c-report <N2C report> --out <dir>",
            "blocking_looks": "prereg3.blocking_looks on the unit must be empty; no look "
                              "dated on or after 2026-11 exists at registration"},
        "sealed_until_scoring": "every head of 2026-11..2027-10 at the 158 wells, and any "
                                "hit, residual or coverage table; nothing from 2023-26 is "
                                "recomputed",
        "page_mapping": page_mapping,
        "disclosures": disclosures,
        "attempt_policy": "User rule 2026-10-23: engineering failures (crash, bug, missing "
                          "file) may be fixed and re-run before the scoring look, each "
                          "logged; an engineering failure found after the score look (an "
                          "inconsistent E02 step-(5) file) writes no verdict and may be "
                          "fixed and re-run, every run's look logged and listed; each test "
                          "is scored ONCE after the window closes; a "
                          "failed, inconclusive or not-evaluable result is reported as such "
                          "and never re-scored on these units; the secondaries are fixed "
                          "now and reported beside the primary (report-only lines), never in "
                          "its place; after a "
                          "failure at most ONE follow-up registration (new test ids, label "
                          "'post-failure, exploratory', never validation) which cannot reuse "
                          "this window; the final report lists every attempt",
        "rulings": {
            "2026-10-29 third set (a)": "the combined-sigma 90 % head interval may be "
                                        "labelled 'decided by a registered prospective test' "
                                        "with the en/zh U7 disclosure",
            "2026-10-29 third set (b)": "register E02c on I1's unit: T-cov per county "
                                        "(coverage3 gate-3, clusters = well, pooled-vs-county "
                                        "outcomes declared) and T-level (beat CL and CLT, "
                                        "pre-origin baselines); only tightens; no X1; binds "
                                        "E02 not N2; before 2026-10-31T16:00Z",
            "2026-10-29 (3)": "the N2 companion is dropped; E02c is not it",
            "2026-10-29 third set (d)": "C14 split into C14a-e"},
        "zones": NA("no fit; the frozen forward run's own zones"),
        "priors": disclosures,
        "burn_in": NA("no fit"), "folds": NA("prospective window"),
        "fresh_units": [S.UNIT],
        "thresholds": {"target": S.TARGET, "gate_level": S.GATE_LEVEL,
                       "point_window": list(S.POINT_WINDOW),
                       "n_eff_window_min": S.N_EFF_WINDOW_MIN, "min_k_county": S.MIN_K_COUNTY,
                       "max_miss_side": S.MAX_MISS_SIDE, "min_months": S.MIN_MONTHS,
                       "min_wells_level": S.MIN_WELLS_LEVEL, "boot_B": S.BOOT_B,
                       "boot_seed": S.BOOT_SEED, "boot_ci": S.BOOT_CI,
                       "block_km": S.BLOCK_KM},
        "gate_scores": NA("no weighted gate; each decision rule is reported as it falls"),
        "primary_model": {"name": "datum_gate_innov (the page model's frozen forward run), "
                                  "E02's registered predictions",
                          "forward_sha256": S.MODEL_SHA,
                          "predictions_sha256": S.E02_PRED_SHA},
        "inputs": inputs,
        "interpretation": "Tighter readings of E02's frozen forecast on fresh months: whether "
                          "its 90 % interval covers per county under a cluster-aware gate, "
                          "and whether its level beats both simple pre-origin baselines. "
                          "Nothing about policy responses, subsidence or the run-spread band.",
        "written_by": f"{S.E}/prereg_e02c.py",
    }
    if rehearsal:
        d["rehearsal"] = True
    return d


# ----------------------------------------------------------------------------- CLI
def problems_for_write(d: dict, baselines_dir, power_report, expect_self_sha,
                       prereg_dir=None, power_conservative: bool = False) -> list[str]:
    bad = []
    if baselines_dir is None or d["baselines"]["status"] != "ok":
        bad.append(f"baselines slot: {d['baselines']['status']} (run export_baselines.py "
                   "e02c, then e02c --verify, first)")
    if power_report is not None and power_conservative:
        bad.append("give either --power-report or --power-conservative, not both")
    elif power_conservative:
        if d["power"]["flags"]["status"] != CONSERVATIVE:
            bad.append("power slot: conservative fallback not applied")
    elif power_report is None or d["power"]["flags"]["status"] == PENDING:
        bad.append("power slot: no U1.HB.power.v1 report (or the declared "
                   "--power-conservative fallback)")
    elif d["power"]["registration"] == PENDING:
        bad.append(f"{S.POWER_ID} is not in the registry INDEX")
    elif (d["power"]["flags"].get("declaration") or {}).get("sha256") != \
            d["power"]["registration"].get("sha256"):
        bad.append(f"the power report was not run under the registered {S.POWER_ID} "
                   "declaration (sha differs or is missing)")
    if set(d["frozen_code"]) != set(FROZEN):
        bad.append(f"frozen code incomplete: {sorted(set(FROZEN) - set(d['frozen_code']))}")
    for n, h in d["binds"]["e02_expected"].items():
        if d["binds"]["e02"][n]["sha256"] != h:
            bad.append(f"E02 {n} sha changed")
    for k, (_, h) in S.SECONDARIES.items():
        if d["binds"]["secondaries_report_only"][k]["predictions.csv"]["sha256"] != h:
            bad.append(f"secondary {k} predictions changed")
    if S.sha256(REPO / S.FORWARD) != S.MODEL_SHA:
        bad.append("forward npz sha changed")
    self_sha = S.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != this script {self_sha}")
    if S._now() >= S.DEADLINE:
        bad.append(f"DEADLINE PASSED ({S.DEADLINE:%Y-%m-%dT%H:%MZ}): do not register E02c "
                   "for this window")
    return bad


def check_blocking(d: dict, prereg_dir=None) -> list:
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    idx = prereg3.membership_index(pdir)
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            if not idx.known(u):
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
    return prereg3.blocking_looks(d["tests"], None, pdir, idx, test_time=S._now())


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="write the registration (the lead)")
    ap.add_argument("--baselines-dir", type=Path, default=None)
    ap.add_argument("--power-report", type=Path, default=None)
    ap.add_argument("--power-conservative", action="store_true",
                    help="declared fallback: no U1.HB.power.v1 result before the deadline")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true", help="do not print the JSON")
    a = ap.parse_args(argv)
    d = doc(a.baselines_dir, a.power_report, prereg_dir=prereg_dir,
            power_conservative=a.power_conservative and a.power_report is None)
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    block = check_blocking(d, prereg_dir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", _sha_text(d))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("baselines:", d["baselines"]["status"], "| power:", d["power"]["flags"]["status"],
          "| power registration:", "ok" if d["power"]["registration"] != PENDING else PENDING)
    print("this script sha256:", S.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = problems_for_write(d, a.baselines_dir, a.power_report, a.expect_self_sha,
                             prereg_dir, power_conservative=a.power_conservative)
    if errs or block or bad:
        print("REFUSED:", errs + [str(b) for b in block] + bad)
        return 1
    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
