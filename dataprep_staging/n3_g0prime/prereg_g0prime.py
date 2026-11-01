"""Write the declaration N3.G0prime.v1 (harness.prereg3, kind 'declaration') BEFORE G0' runs.

Judge rank 14 (round 7), user ruling R7.n3_g0prime (user_rulings_2026-11-01b.json: 'Yes,
both, registered by 2026-11-30T16:00Z'): after N3's known G0 failure, one attenuation-robust
conversion estimator may be declared and checked by a fresh, disclosed synthetic run (G0'),
and, if G0' passes, N3's 2026-12..2027-11 pile months may take a second, gating role
(N3Gprime.piles_conversion.v1, prereg_n3g.py). This declaration hashes the G0' design, code,
model-side input, pass rule and decision before the first run; the run (g0prime.py) refuses
to start until it is registered and its frozen code is unchanged. Model-only: no observation
is read and no fresh unit is consumed. The ring-attenuation N note (ring_attenuation.py,
model-only) is declared here too.

``--write`` refuses unless: the gate mode is chosen explicitly (--gate all | identifiable; see
GATE_TEXT: with 'all' a FAIL is expected by construction, 'identifiable' relaxes the judge's
text and needs --gate-basis naming the judge's or lead's decision); N3.piles_prospective.v1
is registered and intact with the pinned sha (afa017d7) and names the declared model_side.json
(1b40afe9) and mapping.json (eea0d2d8); the ruling file hashes as pinned (5f53909b) and its
record names this declaration; N3's G0 result hashes as pinned (80e260d0); no G0' or ring
result exists yet; it is before 2026-11-30T16:00Z; validation passes; no blocking look;
--expect-self-sha is this file's sha256. The --validate dry run prints the JSON and its sha
(the written document adds only the tool's fields).

Usage (repository root):
    python -I dataprep_staging/n3_g0prime/prereg_g0prime.py --validate [--gate all]
    python -I dataprep_staging/n3_g0prime/prereg_g0prime.py --write --gate all \
        --expect-self-sha <sha256 of this file>                         (the lead only)
    (or --gate identifiable --gate-basis '<who decided, where>', after the judge/lead rule)
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

import g0p_lib as P  # noqa: E402
import g0prime as GP  # noqa: E402
from common.paths import DATAPREP, from_repo, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

L = P.L
NA = prereg3.not_applicable
RULINGS_SCHEMA = "dataprep-rulings/1"
RULINGS_DIR = "results/dataprep/rulings"
FROZEN = (f"{P.E}/g0p_lib.py", f"{P.E}/g0prime.py", f"{P.E}/ring_attenuation.py",
          f"{P.E}/prereg_g0prime.py", "dataprep_staging/n3_piles/g0_n3.py",
          "dataprep_staging/n3_piles/n3lib.py", L.SHARED_SCORER,
          "hydrophysics/twin/mechanism.py", f"{P.E}/run_g0prime.sh")
RUNNER = f"{P.E}/run_g0prime.sh"
TESTS = f"{P.E}/tests/test_g0prime.py"


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


# ----------------------------------------------------------------------------- shared checks
def registered(prereg_id: str, prereg_dir=None, rows_dir=None) -> dict:
    """{'status': 'ok', 'sha256', 'doc', 'path'} of an intact registration, or why not."""
    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as exc:  # noqa: BLE001 - any failure means 'not usable'
        return {"status": f"{prereg_id} not registered or not intact: {exc}"}
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    return {"status": "ok", "sha256": sha, "doc": doc,
            "path": rel_to_repo(d / f"{prereg_id}.json")}


def ruling_block(path, rid: str, must_name: str, rulings_dir=None) -> dict:
    """A user ruling record (schema dataprep-rulings/1, ruling_by 'user') under the rulings
    directory whose ruling ``rid`` names ``must_name`` in applies_to. Never edited."""
    if path is None or not rid:
        return {"status": "absent"}
    p = from_repo(path)
    root = Path(rulings_dir) if rulings_dir else DATAPREP / "rulings"
    if not p.is_file():
        return {"status": f"no ruling file {p.name}"}
    if root.resolve() not in p.resolve().parents:
        return {"status": f"the ruling file is not under {RULINGS_DIR}"}
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {"status": "the ruling file is not JSON"}
    out = {"status": "ok", "id": rid, "file": rel_to_repo(p), "sha256": P.sha256(p),
           "date": r.get("date")}
    hit = [x for x in r.get("rulings") or [] if isinstance(x, dict) and x.get("id") == rid]
    if r.get("schema") != RULINGS_SCHEMA or r.get("ruling_by") != "user":
        out["status"] = f"not a user ruling record ({RULINGS_SCHEMA}, ruling_by 'user')"
    elif len(hit) != 1:
        out["status"] = f"ruling id {rid!r} not found once in the record"
    elif must_name not in [str(a) for a in hit[0].get("applies_to") or []]:
        out["status"] = f"ruling {rid!r} does not name {must_name} in applies_to"
    elif not str(hit[0].get("decision") or "").strip():
        out["status"] = f"ruling {rid!r} has no decision text"
    else:
        out.update({"question": hit[0].get("question"), "decision": hit[0]["decision"],
                    "applies_to": hit[0]["applies_to"]})
    return out


def n3_block(prereg_dir=None, rows_dir=None, model_side_sha: str = P.MODEL_SIDE_SHA) -> dict:
    """N3.piles_prospective.v1 as registered (cited, never changed)."""
    r = registered(L.ID, prereg_dir, rows_dir)
    if r["status"] != "ok":
        return {"status": r["status"]}
    doc = r["doc"]
    out = {"status": "ok", "id": L.ID, "sha256": r["sha256"], "path": r["path"],
           "roles": doc.get("roles"),
           "g0": {k: (doc.get("g0") or {}).get(k) for k in ("result", "decision", "pass_12",
                                                             "pass_24")}}
    ms = ((doc.get("model_side") or {}).get("files") or {}).get("model_side.json") or {}
    mp = ((doc.get("model_side") or {}).get("files") or {}).get("mapping.json") or {}
    if ms.get("sha256") != model_side_sha:
        out["status"] = "N3's registration names another model_side.json"
    out["model_side"] = ms
    out["mapping"] = mp
    return out


# ----------------------------------------------------------------------------- the gate mode
GATE_TEXT = {
    "all": "every cell class gates (the judge's rank-14 text: 'a pass must hold there too'). "
           "The 'amp' cells cannot be passed by any nest-based estimator (rho / amp, see "
           "power_by_cell_class): P(pass) ~ 0 by construction, a FAIL is the expected outcome "
           "and G0' then spends ruling R7.n3_g0prime only to firm (and attribute) C31's note",
    "identifiable": "the 'g0' and 'fdev' classes gate; the seasonal-mismatch classes ('lag', "
                    "'amp') are a declared, report-only sensitivity, because no estimator can "
                    "separate a nest/pile seasonal mismatch from the conversion using nest "
                    "heads (adversarial review of the n3p build, synthetic check). This "
                    "RELAXES the judge's text and is valid only with gate_basis naming the "
                    "judge's or lead's explicit decision"}
GATE_BASIS_ALL = "judge rank 14 text (round 7): 'a pass must hold there too'"
POWER_BY_CELL_CLASS = {
    "g0": "N3 G0's world with the new estimators: unknown before the run; it rises with the "
          "model's seasonal nest swing relative to the nest error (sd 0.75-1.5 m). Synthetic "
          "checks in a test world (not the real model): with a 1 m seasonal swing all three "
          "missed the 0.80 / 0.85 rules at sd 0.75, phi 0.5 (draw first-stage F ~1-7); with "
          "3 m all three recovered rho",
    "fdev": "at most the g0 power: dF is uncorrelated with every instrument and adds variance",
    "lag": "(a) unaffected in expectation (phase-free); (b) reads ~cos 60 deg = 0.5 x rho; "
           "(c) reads between rho and rho / cos 60 deg = 2 rho (2 rho when the seasonal part "
           "dominates dD). Review check at rho 1: (b) ~0.5, (c) ~1.9, P(correct) 0.02-0.05. "
           "Power ~0 for (b), (c)",
    "amp": "every nest-based estimator measures rho / amp; nest data cannot separate the "
           "nest's seasonal swing from the conversion. Review check (3 m seasonal swing, sd "
           "0.75, phi 0.5): amp 1.3, rho 2 reads 1.52-1.53 (band edge 1.5; false 'supported' "
           "0.40-0.45 vs 0.10); amp 1.3, rho 1 coverage 0.07-0.35 vs 0.85. Power ~0 for "
           "every estimator, by construction"}
PRIORS = {
    "all": "N3 G0 (OLS) failed at 12 and 24 months. With gate_mode 'all' P(pass) ~ 0 by "
           "construction (the 'amp' cells; power_by_cell_class): a FAIL is expected and is "
           "attributed by decision.fail_class. The judge's ~0.35-0.4 did not account for the "
           "amp cells being unidentifiable from nest heads",
    "identifiable": "N3 G0 (OLS) failed at 12 and 24 months. With gate_mode 'identifiable' "
                    "the pass chance is the power in the 'g0' and 'fdev' classes: unknown, "
                    "plausibly below the judge's ~0.35-0.4 (the builder's synthetic draws had "
                    "first-stage F ~1-7 at a 1 m seasonal swing)",
    None: "UNDECIDED: the gate mode must be chosen before --write (GATE_TEXT)"}


def design(gate_mode: str | None, gate_basis: str | None) -> dict:
    d = dict(DESIGN)
    d["gate_mode"] = gate_mode
    d["gate_mode_text"] = GATE_TEXT.get(gate_mode, "UNDECIDED: choose --gate all | "
                                                   "identifiable before --write")
    d["gating_classes"] = list(P.GATE_MODES[gate_mode]) if gate_mode in P.GATE_MODES else None
    d["gate_basis"] = (GATE_BASIS_ALL if gate_mode == "all" else gate_basis)
    d["power_by_cell_class"] = POWER_BY_CELL_CLASS
    return d


# ----------------------------------------------------------------------------- the document
DESIGN = {
    "world": "N3 G0's synthetic world (g0_n3.py, frozen, imported): the frozen page model "
             "(datum_gate_innov, coupled_leveling column) at the 2 primary pile cells (土庫國中 "
             "400 m, 中科虎尾 300 m); true pile increment = rho x the model column increment; "
             "observed nest dD = model nest dD + the difference of an AR(1) nest-vs-cell head "
             "error on the head LEVEL; printed increments rounded to 0.1 cm; window "
             "2026-12..2027-11 (12 months) and 2026-12..2028-11 (24 months)",
    "frozen_grid": "sd {0.75, 1.5} m x phi {0.5, 0.9} x rho {0.4, 1, 2}, 500 replicates per "
                   "cell, block bootstrap 3 months B 2000 (all imported from g0_n3.py / "
                   "n3lib.py)",
    "added_cells": {
        "seasonal_mismatch": "nest/pile annual amplitude ratio amp {0.7, 1.0, 1.3} x phase lag "
                             "{0, 2} months: the observed nest head's annual component is amp "
                             "x the model nest's (harmonic fitted by OLS on [1, t, cos, sin] to "
                             "the window head path, D(2026-11) = 0) delayed by lag months "
                             "(g0p_lib.mismatch_dd); the pile still follows the model, so the "
                             "truth stays rho. Adds (1, 2) to the judge's {0.7, 1.3} x {0, 2}",
        "forecast_deviation": "{none, cell}: 'cell' adds an independent AR(1) (the cell's sd "
                              "and phi) to the true local head, seen by the nest (x) and "
                              "converted into the pile at the model's own OLS slope g_mod "
                              "(y), so that the actual head change is not exactly the "
                              "model's (the instrument of (c)); 'none' is G0's world. Added "
                              "by the builder (not in the judge's text). Without it (c) can "
                              "still fail by variance; dF is uncorrelated with every "
                              "instrument, so it adds variance only ((c) stays consistent; "
                              "(a), (b) are not weaker)"},
    "cell_classes": {"g0": "amp 1, lag 0, fdev none: N3 G0's own cells",
                     "fdev": "amp 1, lag 0, fdev cell",
                     "lag": "amp 1, lag 2 (either fdev)",
                     "amp": "amp 0.7 or 1.3 (either lag, either fdev)"},
    "estimators_declared_order": {
        "a_harmonic_ratio": "per pile annual harmonic (sin, cos of the calendar month, pile "
                            "intercept) of increment and dD; pooled sign(sum b_y.b_x) x "
                            "sqrt(sum |b_y|^2 / sum |b_x|^2). Phase-free; Rician bias at low "
                            "SNR (disclosed)",
        "b_iv_harmonics": "2SLS with a pile fixed effect; instruments the per-pile annual "
                          "sin and cos",
        "c_iv_model_heads": "2SLS with a pile fixed effect; instrument the frozen ZR model's "
                            "own nest dD at the same months (model_side.json world)",
        "reference_ols_fe": "N3's pile-FE OLS slope (n3lib.fe_slope) on the same draws; "
                            "reported, never selectable"},
    "g_mod": "each estimator on the noise-free model world (y = column increment, x = z = "
             "model nest dD): a frozen number per estimator and horizon; rho = g / g_mod; the "
             "OLS g_mod must equal N3's registered g_mod_12 / g_mod_24 (checked, reported)",
    "interval_and_verdict": "90 % percentile interval of a 3-month moving-block bootstrap "
                            "within pile (n3lib.block_indices, B 2000, the same resampled "
                            "rows for every estimator); verdict = N2's shared rule via "
                            "n3lib.conversion_verdict (bootstrap_scorer.py sha f8990d61, "
                            "cited, never edited)",
    "pass_rule": "an estimator passes at a horizon iff in EVERY (sd, phi, amp, lag, fdev) "
                 "cell of a GATING class (gate_mode): P(conversion_supported | rho 1) >= "
                 "0.80; P(direction_supported_size_not | rho 0.4) >= 0.80; "
                 "P(conversion_supported | rho 0.4) <= 0.10 and P(conversion_supported | rho "
                 "2) <= 0.10; coverage of rho = 1 by the 90 % interval >= 0.85 (G0's four "
                 "rules, unchanged); and once per horizon: model-side first-stage F >= 10 "
                 "(the judge's weak-instrument check; (a), (b): dD on the per-pile harmonics; "
                 "(c): dD on the model dD) and g_mod finite and > 0. The median first-stage "
                 "F of the draws per cell is REPORTED, not gated (it duplicates the coverage "
                 "and power rules). The pass of every cell class is reported for every "
                 "estimator and horizon",
    "engineering_precheck": "before any cell runs: the OLS g_mod of both horizons equals "
                            "N3's registered g_mod_12 / g_mod_24 and is finite and > 0; "
                            "otherwise g0prime.py stops, writes nothing (an engineering "
                            "failure, logged, not a scored attempt)",
    "decision": "the first estimator in the order (a), (b), (c) that passes at 12 months is "
                "N3G's estimator with a 12-month gating test; else the first that passes at "
                "24 months, with a 24-month gating test; else G0' fails, no N3G "
                "registration is written, and the fail is attributed to the first cell "
                "class (g0, then g0 + fdev, else the mismatch classes) that no estimator "
                "passes at either horizon (g0prime.fail_class). All three estimators and "
                "the OLS reference are "
                "reported in every case; the order was declared before any run, and the "
                "selection uses synthetic worlds only (no observation)",
    "seed": P.SEED,
    "cells_per_horizon": len(P.noise_cells()) * len(P.RHOS)}
PAGE = {
    "g0prime_pass": "C31 moves N -> D now: 'compaction per metre of head change at the 2 "
                    "Yunlin deep-pile cells: decided by N3G (not before 2028-01-15)'; it can "
                    "become S in 2028 only on a joint N2 'response_supported' + N3G "
                    "'conversion_supported' (N3's page mapping, copied unchanged into N3G). "
                    "C27's conversion clause reads 'decided by N3G (pending)'. Only if N3G is "
                    "registered before 2026-11-30T16:00Z",
    "g0prime_fail": {
        "rule": "C31 keeps N with the note keyed to decision.fail_class (g0prime.FAIL_WORDING); "
                "C27 unchanged",
        **{f"note_{k}": v for k, v in GP.FAIL_WORDING.items()}},
    "g0prime_pass_identifiable_caveat": "only under gate_mode 'identifiable': the D line adds "
                                        "'assumes the nest's seasonal swing tracks the pile "
                                        "cell's (a mismatch cannot be detected from nest "
                                        "heads)'",
    "ring_note": "always an N note beside the UNCHANGED 3.4 / 9.3 mm/m ring figure and the 0.92 "
                 "pile line (ring_attenuation.py note text); never a corrected figure, never a "
                 "status",
    "never": "no correction factor (ruling 2026-10-29 (4)); no cut number, coastal, long-run "
             "or Changhua subsidence claim"}


def doc(model_side: Path | None = None, ruling: Path | None = None,
        ruling_id: str = P.RULING_ID, prereg_dir=None, rows_dir=None,
        rulings_dir=None, gate_mode: str | None = None, gate_basis: str | None = None) -> dict:
    msp = Path(model_side) if model_side else from_repo(P.MODEL_SIDE)
    ms_rec = ({"path": rel_to_repo(msp), "sha256": P.sha256(msp), "status": "ok"}
              if msp.is_file() else {"path": rel_to_repo(msp), "status": "missing"})
    if ms_rec["status"] == "ok" and ms_rec["sha256"] != P.MODEL_SIDE_SHA:
        ms_rec["status"] = "model_side.json is not N3's registered file (sha 1b40afe9)"
    n3 = n3_block(prereg_dir, rows_dir, ms_rec.get("sha256", P.MODEL_SIDE_SHA))
    rb = ruling_block(ruling if ruling else from_repo(P.RULING_FILE), ruling_id, P.ID,
                      rulings_dir)
    frozen = {p: P.sha256(REPO / p) for p in FROZEN if (REPO / p).is_file()}
    n3g0 = from_repo(P.N3_G0_RESULT)
    inputs = sorted(set(frozen) | ({ms_rec["path"]} if ms_rec["status"] == "ok" else set())
                    | ({rb["file"]} if rb.get("status") == "ok" else set())
                    | ({n3["path"]} if n3.get("status") == "ok" else set())
                    | ({P.N3_G0_RESULT} if n3g0.is_file() else set()))
    d = {
        "id": P.ID, "wp": P.WP, "kind": "declaration",
        "title": "N3 G0': synthetic decidability of an attenuation-robust conversion "
                 "estimator at the 2 Yunlin deep piles (harmonic ratio, 2SLS with calendar "
                 "harmonics, 2SLS with the model's heads), with seasonal-mismatch and "
                 "forecast-deviation cells; plus a model-only ring attenuation note",
        "why": "N3's G0 failed by errors-in-variables attenuation of the OLS slope (result "
               "sha 80e260d0; rho = 2 called 'supported' 93-100 % of the time at sd 0.75 m, "
               "phi 0.5). User ruling R7.n3_g0prime allows one estimator change, checked by "
               "a fresh disclosed G0 run, and a second gating role for N3's pile months if "
               "it passes.",
        "tests": NA("model-only declaration: synthetic worlds; no observation is scored"),
        "fresh_units": NA("no observation is read; no fresh unit is consumed"),
        "zones": NA("no fit; the frozen forward run's own zones"),
        "priors": PRIORS.get(gate_mode, PRIORS[None]),
        "burn_in": NA("no fit"), "folds": NA("synthetic worlds; no folds"),
        "thresholds": {"p_correct": P.P_CORRECT, "p_false_max": P.P_FALSE_MAX,
                       "coverage_min": P.COVERAGE_MIN, "f_min_model_side": P.F_MIN,
                       "ratio_band": L.RATIO_BAND, "level": L.LEVEL, "boot_B": L.BOOT_B,
                       "boot_block": L.BOOT_BLOCK, "reps": P.REPS,
                       "sd_grid_m": list(P.SD_GRID), "phi_grid": list(P.PHI_GRID),
                       "rhos": list(P.RHOS), "amp_ratios": list(P.AMP_RATIOS),
                       "lags_months": list(P.LAGS_MONTHS),
                       "forecast_dev": list(P.FORECAST_DEV)},
        "gate_scores": NA("one pass rule per estimator; no weighted score"),
        "primary_model": {"name": "the page model: stage3_datum_gate + coupled_leveling "
                                  "column, forward datum_gate_innov (ZR, ensemble mean), at "
                                  "N3's pile cells and nests (N3's model_side.json)",
                          "forward_sha256": L.FORWARD_SHA, "column_sha256": L.COLUMN_SHA},
        "design": design(gate_mode, gate_basis),
        "model_side": ms_rec,
        "n3_registration": n3,
        "n3_g0": {"result": {"path": P.N3_G0_RESULT, "sha256": P.N3_G0_RESULT_SHA,
                             "sha256_now": P.sha256(n3g0) if n3g0.is_file() else None},
                  "outcome": "failed at 12 and 24 months (OLS attenuation); conversion "
                             "registered report-only by user ruling N3.level_only"},
        "ring_attenuation_note": {
            "code": f"{P.E}/ring_attenuation.py",
            "design": "the model's own hindcast at the ring cells (twin.mechanism.ring_cells "
                      "/ model_ring_series) on a frozen forward archive, plus N3 G0's nest-"
                      "error grid (AR(1) level, sd {0.75, 1.5}, phi {0.5, 0.9}, 200 "
                      "replicates, seed 20261117), through twin.mechanism.seasonal_slope "
                      "(lag 0 and best of 0-2; median over sites); lambda = noisy / noise-"
                      "free median slope",
            "inputs_at_run": "--forward (the archive that produced the page's 9.3 mm/m: the "
                             "lead confirms it reproduces the noise-free median) and --sites "
                             "(a CSV read with usecols site, x, y only)",
            "role": "N note only (page mapping 'ring_note')"},
        "outputs": {"g0prime": f"{P.OUT_DIR}/g0prime/{GP.RESULT}",
                    "ring_attenuation": f"{P.OUT_DIR}/ring_attenuation/result.json",
                    "attempts": f"{P.OUT_DIR}/attempts.jsonl",
                    "archive": f"{P.OUT_DIR}/archive/<UTC stamp>/ (REASON.txt)",
                    "archive_rings": f"{P.OUT_DIR}/archive_rings/<UTC stamp>/ (REASON.txt)"},
        "run": {"runner": RUNNER, "order": ["(1) prereg_g0prime.py --write (this "
                                            "declaration)",
                                            "(2) run_g0prime.sh g0prime (CPU, ~20 min)",
                                            "(3) run_g0prime.sh rings --forward F --sites S "
                                            "(CPU, minutes)",
                                            "(4) if G0' passed: prereg_n3g.py --write "
                                            "before 2026-11-30T16:00Z"],
                "refusals": "g0prime.py and ring_attenuation.py refuse unless this "
                            "declaration is registered, intact and its frozen_code equals "
                            "the code; each result.json is written once; G0' retries via "
                            "'run_g0prime.sh archive', ring-note retries via "
                            "'run_g0prime.sh archive-rings' (separate, logged)"},
        "frozen_code": frozen,
        "page_mapping": PAGE,
        "rulings": {P.RULING_ID: rb,
                    "2026-10-29 (4)": "no calibrated correction factor follows from any "
                                      "verdict",
                    "N3.level_only (2026-11-01)": "N3's conversion lines stay report-only in "
                                                  "N3; N3G is a NEW registration, N3 is "
                                                  "never edited"},
        "disclosures": {
            "known_failure": "G0's failure (OLS) was known when this estimator change was "
                             "proposed; the ruling R7.n3_g0prime authorises it. The "
                             "estimators, their order and the seasonal-mismatch cells came "
                             "from the judge and the adversarial review; the forecast-"
                             "deviation cells, the pure-lag cell (amp 1, lag 2) and a per-"
                             "draw first-stage F rule (since made report-only) were added by "
                             "the builder; the gate mode choice and the class reporting "
                             "followed the build review. All before any G0' run on the real "
                             "model_side.json",
            "builder_runs": "the builder ran exploratory synthetic checks in a scratch "
                            "directory outside the repository (test-style worlds, not the "
                            "real model), saw the expected failure (draw first-stage F ~1-7 "
                            "at a 1 m seasonal swing), and then added the forecast-deviation "
                            "cells, the pure-lag cell and the per-draw F rule; the build "
                            "reviewer ran a synthetic check of the mismatch cells (the "
                            "power_by_cell_class numbers). The tests run G0' on synthetic "
                            "model_side files. The real model_side.json was never given to "
                            "g0prime.py before this declaration",
            "per_draw_f": "a per-draw median first-stage F >= 10 rule was added by the builder "
                          "and made report-only after review: it duplicates the coverage and "
                          "power rules; the judge's weak-instrument gate is the model-side F",
            "gate_mode": "chosen explicitly at --write (design.gate_mode, gate_basis); with "
                         "'all' a fail is expected by construction (power_by_cell_class)",
            "n3_scratch_run": "N3's builder scratch G0 run (sha 5deadeba, failed) is "
                              "disclosed in N3's registration",
            "rician_bias": "the harmonic-amplitude ratio is biased at low SNR; reported, not "
                           "corrected",
            "selection": "three candidate estimators with a declared order: the first "
                         "passing estimator is selected on synthetic worlds only; all three "
                         "are reported and scored as report-only lines in N3G",
            "forecast_deviation": "added by the builder (see design.added_cells); it adds "
                                  "variance only (uncorrelated with every instrument)",
            "ring_note": "the ring re-read and the after-the-fact pile re-read were dropped "
                         "(judge); the note reads no ring value"},
        "attempt_policy": "User rule 2026-10-23: engineering failures (crash, bug, missing "
                          "file) may be fixed and re-run, each archived with a reason and "
                          "logged in attempts.jsonl; a completed G0' result is final and is "
                          "never re-run with other code, grid or thresholds; every attempt "
                          "is listed by prereg_n3g.py",
        "inputs": inputs,
        "interpretation": "Whether any of three declared estimators identifies the pile "
                          "conversion rho at the 2 Yunlin piles in N3's model world under "
                          "nest error, forecast deviation and (gating or report-only, per "
                          "gate_mode) seasonal nest/pile mismatch. A pass allows (does not "
                          "require) N3G's registration; a fail firms C31's N note with the "
                          "failing cell class named. Nothing about observed piles.",
        "written_by": f"{P.E}/prereg_g0prime.py"}
    return d


def pinned_problems(n3: dict, rb: dict) -> list[str]:
    """The pinned shas: N3's registration, its mapping.json, the ruling file."""
    bad = []
    if n3.get("status") == "ok":
        if n3.get("sha256") != P.N3_REG_SHA:
            bad.append(f"N3 registration sha {n3.get('sha256')} != pinned {P.N3_REG_SHA}")
        if (n3.get("mapping") or {}).get("sha256") != P.MAPPING_SHA:
            bad.append(f"N3's mapping.json sha {(n3.get('mapping') or {}).get('sha256')} != "
                       f"pinned {P.MAPPING_SHA}")
    if rb.get("status") == "ok" and rb.get("sha256") != P.RULING_SHA:
        bad.append(f"ruling file sha {rb.get('sha256')} != pinned {P.RULING_SHA}")
    return bad


def problems_for_write(d: dict, expect_self_sha, out_root: Path | None = None) -> list[str]:
    bad = []
    des = d["design"]
    if des.get("gate_mode") not in P.GATE_MODES:
        bad.append("gate mode not chosen: --gate all | identifiable (design.gate_mode_text)")
    elif des["gate_mode"] == "identifiable" and not str(des.get("gate_basis") or "").strip():
        bad.append("--gate identifiable relaxes the judge's text: --gate-basis must name the "
                   "judge's or lead's explicit decision")
    if d["model_side"].get("status") != "ok":
        bad.append(f"model_side: {d['model_side'].get('status')}")
    if d["n3_registration"].get("status") != "ok":
        bad.append(f"N3 registration: {d['n3_registration'].get('status')}")
    rb = d["rulings"].get(P.RULING_ID) or {}
    if rb.get("status") != "ok":
        bad.append(f"ruling {P.RULING_ID}: {rb.get('status')}")
    bad += pinned_problems(d["n3_registration"], rb)
    g0r = d["n3_g0"]["result"]
    if g0r.get("sha256_now") != P.N3_G0_RESULT_SHA:
        bad.append(f"N3 G0 result {g0r['path']} sha {g0r.get('sha256_now')} != pinned "
                   f"{P.N3_G0_RESULT_SHA}")
    if set(d["frozen_code"]) != set(FROZEN):
        bad.append(f"frozen code incomplete: {sorted(set(FROZEN) - set(d['frozen_code']))}")
    root = Path(out_root) if out_root else from_repo(P.OUT_DIR)
    for f in ("g0prime/result.json", "ring_attenuation/result.json"):
        if (root / f).exists():
            bad.append(f"{root / f} exists: G0' must be declared before its first run")
    self_sha = P.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != this script {self_sha}")
    if L._now() >= P.DEADLINE:
        bad.append(f"DEADLINE PASSED ({P.DEADLINE:%Y-%m-%dT%H:%MZ}): ruling R7.n3_g0prime "
                   "covers declarations registered before it")
    return bad


def main(argv=None, prereg_dir=None, rows_dir=None, rulings_dir=None, out_root=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="write the declaration (the lead)")
    ap.add_argument("--model-side", type=Path, default=None)
    ap.add_argument("--ruling", type=Path, default=None)
    ap.add_argument("--ruling-id", default=P.RULING_ID)
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--gate", choices=sorted(P.GATE_MODES), default=None,
                    help="the declared gate mode (required for --write; see GATE_TEXT)")
    ap.add_argument("--gate-basis", default=None,
                    help="for --gate identifiable: who decided it and where (judge/lead)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.model_side, a.ruling, a.ruling_id, prereg_dir, rows_dir, rulings_dir,
            a.gate, a.gate_basis)
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    errs = prereg3.validate_prereg(d, prereg_dir=pdir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", sha_text(d))
    print("validate:", errs or "ok")
    print("slots: model_side", d["model_side"]["status"], "| N3",
          d["n3_registration"]["status"], "| ruling", d["rulings"][P.RULING_ID]["status"],
          "| gate", d["design"]["gate_mode"] or "UNDECIDED")
    print("pinned:", pinned_problems(d["n3_registration"], d["rulings"][P.RULING_ID]) or "ok",
          "| N3 G0 result", "ok" if d["n3_g0"]["result"].get("sha256_now") ==
          P.N3_G0_RESULT_SHA else d["n3_g0"]["result"].get("sha256_now"))
    print("this script sha256:", P.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = problems_for_write(d, a.expect_self_sha, out_root)
    if errs or bad:
        print("REFUSED:", errs + bad)
        return 1
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
