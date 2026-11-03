"""Draft declaration of FV.stat.v2 (harness.prereg3 schema, kind 'declaration'), built in code
because dataprep_staging/ ignores *.json. Nothing here writes to the registry: the lead reviews
the draft, picks a FRESH seed and registers it with prereg3.write_prereg after the code is
committed (R7.commit_first).

  prereg_v2.py --seed N        prints the draft JSON, its sha256, validate_prereg and readiness
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stat_v2 as S  # noqa: E402  (puts the FV and staging directories on sys.path)

O = S.O  # noqa: E741  (osse_fv, frozen)
V2 = S.V2
PENDING = "PENDING"
MISSING = "MISSING"
V1_ID = "FV.stat.v1"
V1_DECL_SHA = "95d201dccf485124950ece8bfcf429255af67b1426860a6d21d2c0e625ab996d"
V1_DECL = f"results/dataprep/preregistration/{V1_ID}.json"
V1_FAILURE = "results/dataprep/reports/FV.stat.v1/engineering_failure.json"
V1_FAILURE_SHA = "ac83bd4620ea93dbc7e0f081d5274f8d82efb9e006a87c092e180e9592397464"
V1_PREVIEW_LOOK = ("FV2:stat_preview_k8_declared_seed", "design_time_preview")
R8_PARAMS_LOOK = ("R8:e04_hybrid_parameters_disclosure", "disclosure")
R8_BUILDER_LOOK = ("R8:fv_stat_v2_builder_reads_disclosure", "disclosure")
V1_BUILDER_LOOK = ("FV2:builder_accidental_e04_rows", "accidental_builder_look")
#: looks the draft must cite that readiness checks only when looks.jsonl is present
LOGGED_LOOKS = {"r8_fv_stat_v2_builder_reads": R8_BUILDER_LOOK,
                "v1_builder_accidental_e04_rows": V1_BUILDER_LOOK}
BUILDER_V2 = (
    "The FV.stat.v2 builder (round 8, 2026-11-03) read code and registrations only: "
    "stat_fv.py, prereg_fv2.py, osse_fv.py / osse_fv2.py helpers, harness.prereg3, "
    "twin.hybrid.register and twin.validation.evaluate; FV.stat.v1's declaration and its "
    "engineering_failure.json; INDEX.jsonl rows for FV.stat and the hybrid registrations; the "
    "metadata fields (time, wp, purpose, units, scope) of looks.jsonl rows "
    "FV2:stat_preview_k8_declared_seed and R8:e04_hybrid_parameters_disclosure (the printed "
    "lines also showed those rows' member descriptions, which are already in the look log). "
    "From E04's protocol.json it read the key list and the evaluate-read fields (window, "
    "registered_before, rule, holdout flag) and compared its model_sha256 with the sha256 of "
    "parameters.json without printing either; it opened no parameters.json value, no "
    "predictions row and no observation. It ran engineering checks only, on its own smoke seed "
    "(gate E with the control, and a K = 2 plumbing smoke that writes no rate), and no P. "
    "These reads (E04 protocol.json keys, dates and rule keys; parameters.json and "
    "manifest.json hashed, not opened; looks rows 615 and 624 printed, whose values were "
    f"already disclosed) are logged by the lead as the look {R8_BUILDER_LOOK[0]} "
    "(2026-11-03T01:39Z). A fixer pass the same day read code, this draft, E04's claims-ledger "
    "note and the time, wp, purpose and units fields of looks rows 614, 615, 624 and 628 only; "
    "it computed no statistic and ran only the engineering smoke.")
SEED_NOT_REROLL = (
    "The registered seed is FRESH, chosen by the lead at registration; v1's declared seed "
    "20261103, v1's smoke seed and v2's smoke seed are refused in code (stat_v2.check_seed). "
    "FV.stat.v1 said 'the seed is NOT changed (that would be a reroll)'; that applied to "
    "re-scoring v1's own registered design after its K = 8 preview. v2 is a new declaration "
    "whose plan entry (round 8, #3) requires a fresh seed, and the change cannot be "
    "outcome-shopping: the only outcome seen, v1's K = 8 preview on 20261103, gave pooled P "
    "about 0.92 (cells 0.75-1.0), far above the 0.5 threshold and on the side v2 is expected to "
    "read, so no seed is being chosen to move a reading. v2's new streams 41-43 (v1 used "
    "31-33) alone already make every world fresh, whatever the seed; the fresh seed adds that "
    "no previewed draw is reused.")


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _json(p) -> dict | None:
    q = O.rp(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.is_file() else None


def looks_file(prereg_dir=None) -> Path:
    from harness import prereg3

    return (Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR) / "looks.jsonl"


def logged_look(unit: str, purpose: str, prereg_dir=None) -> dict | None:
    from harness import prereg3

    p = looks_file(prereg_dir)
    if not p.is_file():
        return None
    for lk in prereg3.read_chained(p):
        if lk.get("units") == [unit] and lk.get("purpose") == purpose:
            return {k: lk.get(k) for k in ("time", "wp", "purpose", "units", "data_scope")}
    return None


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def draft_prereg(seed: int, prereg_dir=None) -> dict:
    """FV.stat.v2's declaration for ``seed`` (the lead's fresh choice; refused seeds raise)."""
    seed = S.check_seed(seed)
    man = _json(S.E04_MANIFEST_PATH) or {}
    frozen = {p: _sha(p) for p in S.FROZEN}
    inputs = [*S.FROZEN, S.PREREG, S.E04_MANIFEST_PATH, S.E04_PROTOCOL_PATH, V1_DECL, V1_FAILURE]
    doc = {
        "id": S.ID, "wp": S.WP, "kind": "declaration",
        "title": "Synthetic decidability of the statistical layer's registered test (E04) in "
                 "AR(1) worlds, engineering v2 of FV.stat.v1 (self-check fixed, fresh seed, "
                 "rho = 0 null reading report-only); P > 0.5 reads 'not shown weak', otherwise "
                 "'weak by design' (one-sided)",
        "label": "engineering v2 (FV.stat.v1 stopped in gate E before any statistic; fresh "
                 "seed; not a post-failure follow-up)",
        "v1": {"id": V1_ID, "declared_sha256": V1_DECL_SHA,
               "declaration": {"path": V1_DECL, "sha256": _sha(V1_DECL)},
               "engineering_failure": {"path": V1_FAILURE, "sha256": _sha(V1_FAILURE),
                                       "expected_sha256": V1_FAILURE_SHA},
               "bug": "stat_fv.py:232 writes model_sha256 'synthetic'; stat_fv.py:424 passes "
                      "the real E04 protocol to evaluate (stat_fv.py:234), which refuses the "
                      "rows; stat_fv.py:324 (**protocol) would also let a caller's "
                      "model_sha256 override 'synthetic'",
               "unchanged": "stat_fv.py and prereg_fv2.py are not edited; v2 imports v1's "
                            "functions (ar1_unit, anomalies, e04_layout, e04_frame, e04_pass, "
                            "cells) and binds stat_fv.py by sha256"},
        "e04": {"manifest": S.E04_MANIFEST_PATH, "protocol": S.E04_PROTOCOL_PATH,
                "manifest_sha256": _sha(S.E04_MANIFEST_PATH),
                "protocol_sha256": _sha(S.E04_PROTOCOL_PATH),
                "registered_on": man.get("registered_on", PENDING),
                "stations": man.get("stations", PENDING),
                "layout_used": {"origins": S.E04_ORIGINS, "counties": S.E04_COUNTIES},
                "rule": S.E04_PROTOCOL, "scorable_from": S.E04_SCORABLE_FROM},
        "looks": {"v1_stat_preview": logged_look(*V1_PREVIEW_LOOK, prereg_dir) or PENDING,
                  "r8_e04_parameters_disclosure": logged_look(*R8_PARAMS_LOOK, prereg_dir)
                  or PENDING,
                  **{k: logged_look(*v, prereg_dir) or PENDING for k, v in LOGGED_LOOKS.items()},
                  "v2_builder": "engineering only (see disclosures.builder); no look at any "
                                "observation and no synthetic statistic"},
        "design": {"rho_grid": list(S.RHO_GRID), "share_grid": list(S.SHARE_GRID),
                   "cells": S.cells(), "K_STAT": S.K_STAT, "seed": seed,
                   "seed_chosen_by": "the lead, at registration (fresh; never previewed)",
                   "refused_seeds": {str(k): v for k, v in S.REFUSED_SEEDS.items()},
                   "smoke_seed": S.SMOKE_SEED,
                   "streams": {"e04": S.STREAM_E04, "null": S.STREAM_NULL,
                               "selfcheck": S.STREAM_SELFCHECK},
                   "null_rho": S.NULL_RHO, "K_NULL": S.K_NULL, "null_cells": S.null_cells(),
                   "axis": "2012-01..2027-10; window 2026-11..2027-10; fitted months before",
                   "copied_verbatim_from_v1": ["RHO_GRID", "SHARE_GRID", "K_STAT",
                                               "E04_PROTOCOL", "E04_ORIGINS", "E04_COUNTIES",
                                               "P_WEAK", "RHO_RECOVERY_TOL", "N_RECOVERY",
                                               "N_SELFCHECK (AR worlds)"],
                   "not_carried": "I1.hybrid_tf T1/T2 (v1's report-only rider; see "
                                  "disclosures.t1_t2_dropped)"},
        "truth_world": "anomaly = sqrt(1 - c) x well AR(1) + sqrt(c) x county AR(1), same rho, "
                       "unit variance; every month observed except E04's registered origins (8 "
                       "wells end 2026-07, 6 end 2026-08); null worlds the same with rho = 0",
        "rule": {"e04": "twin.hybrid.register's arithmetic re-run per world (stat_fv.e04_frame); "
                        "E02's evaluation (ratio <= 1.0 AND pooled coverage in [0.85, 0.95], "
                        ">= 100 stations x >= 6 months; stat_fv.e04_pass = osse_fv.g0_e02)"},
        "readings": {"threshold": S.P_WEAK, "P": "equal-weight mean over the nine AR cells",
                     "rule": S.READING_RULE, "wording": S.WORDING,
                     "what_a_2028_pass_shows": S.PASS_CLAUSE,
                     "null_reading": {"report_only": True, "rho": S.NULL_RHO,
                                      "P0": "equal-weight mean over the three null cells",
                                      "expected": "about 0.4-0.5 (planner's prior, not a "
                                                  "gate)",
                                      "use": "quoted in the wording and the pass clause; "
                                             "P / P0 reported as a report-only likelihood "
                                             "ratio of a pass; sets no branch"}},
        "gates": {"E": {
            "selfcheck": "on two AR worlds (0.86/0.15) and one null world: the rows' "
                         "model_sha256 is the sha256 of their synthetic predictions file; "
                         "evaluated against a synthetic copy of the real protocol (its "
                         f"evaluate-read fields {list(S.EVAL_KEYS)}, model_sha256 replaced) they "
                         "must equal osse_fv.g0_e02 exactly (coverage, ratio, pass)",
            "control": f"the same rows handed to the REAL E04 protocol must be refused with "
                       f"'{S.REFUSAL}' and for no other reason; the synthetic sha must differ "
                       "from E04's model_sha256",
            "rho_recovery": f"cell 0.86/0.0, first {S.N_RECOVERY} worlds, mean fitted rho "
                            f"within {S.RHO_RECOVERY_TOL} of 0.86",
            "on_failure": "EngineeringFailure, no report written; free retry before any "
                          "statistic, each attempt appended by stat_v2.run to "
                          f"results/dataprep/reports/{S.ID}/{S.ATTEMPTS_NAME}"},
            "MC": {"rule": f"1.645 SE <= {V2.MC_HALF_WIDTH}",
                   "note": V2.mc_note(len(S.cells()), S.K_STAT)}},
        "page_mapping": {
            "before_scoring": "no change",
            "weak_by_design": "E04 moves to D-measured; interpretation line next to "
                              "evc_hyb_reg: a 2028 fail would not refute the statistical layer, "
                              "with the null-reading caveat and the pass clause (page line "
                              "under R8.7)",
            "not_shown_weak": "E04 moves to D-measured; 'not shown weak' with P, the "
                              "null-reading caveat and the pass clause; no other reading",
            "not_evaluable": "no line",
            "never": "no S; the null reading is never a reading of E04; P / P0 is never a "
                     "page number; the 1.8 persistence weight is never cited as support"},
        "attempt_policy": {"scored_runs": "scored once (report.json, mode 'x')",
                           "engineering_retries": "free while gate E fails, before any "
                                                  "statistic, logged in code",
                           "attempts_log": f"results/dataprep/reports/{S.ID}/"
                                           f"{S.ATTEMPTS_NAME} (stat_v2.run appends start, "
                                           "engineering_failure with its message, error, or "
                                           "success with the report sha256; no statistic)",
                           "exploratory_followups_max": 0, "thresholds_frozen": True,
                           "report": "every attempt"},
        "priors": "With persistence as assumed, the layer beats climatology in expectation at "
                  "leads 2-13, but its constant +-1.645 sd interval over-covers early leads, so "
                  "pooled coverage may exceed 0.95; common shocks widen both. Without "
                  "persistence the forecast is close to climatology, so the ratio sits near 1 "
                  "and roughly half the worlds pass. v1's K = 8 preview (pooled P about 0.92) "
                  "makes 'not_shown_weak' the expected reading. No direction is claimed.",
        "disclosures": {
            "v1_preview": S.V1_PREVIEW_NOTE + "; logged by the lead as the look "
                          f"{V1_PREVIEW_LOOK[0]} (2026-11-01)",
            "v1_engineering_failure": f"{V1_FAILURE} (sha256 {V1_FAILURE_SHA[:8]}): v1 stopped "
                                      "in e04_selfcheck before any statistic; nothing written",
            "r8_e04_parameters_look": f"the round-8 planner's accidental read of E04's "
                                      f"parameters.json (pooled rho and a few wells' fitted "
                                      f"values), logged as {R8_PARAMS_LOOK[0]} (2026-11-02); "
                                      "it set nothing here: the grid, K, rule and thresholds "
                                      "are v1's, fixed on 2026-11-01",
            "grid": "RHO_GRID is the round-7 judge's grid around the E04 fit's median 0.86; the "
                    "E04 fit spans 2012..2026-09 (not purely pre-origin). The post-origin 1.8 "
                    "persistence weight is not used (U7). SHARE_GRID takes values of "
                    "U1.HB.power.v1's registered ICC_month grid",
            "v1_builder_look": f"v1's builder look {V1_BUILDER_LOOK[0]} (logged 2026-11-01, "
                               "cited by FV.stat.v1's declaration): E04 parameters.json pooled "
                               "summary, one well's row and the first predictions row printed "
                               "by accident in round 7; values only in the look log. Nothing "
                               "in v2 was set from it (v2's design is v1's, fixed before it)",
            "seed": SEED_NOT_REROLL,
            "builder": BUILDER_V2,
            "r8_builder_reads_look": f"{R8_BUILDER_LOOK[0]} (logged 2026-11-03T01:39Z): see "
                                     "builder",
            "v1_frozen_code_incomplete": "FV.stat.v1's frozen_code omitted two files its "
                                         "reading path loads: stat_fv.py imports osse_fv2.py, "
                                         "and osse_fv.py imports score_e02c.py at load. Impact "
                                         "low: FV.decidability.v2_postfailure hash-binds both "
                                         "(unchanged since). v2's frozen_code lists both, and "
                                         "hydrophysics/twin/release.py, which "
                                         "twin.validation imports",
            "t1_t2_dropped": "v1's I1.hybrid_tf T1/T2 rider is not carried: it was report-only, "
                             "on its own stream, with no page line; the rider closes with v1 "
                             "(not evaluable before any statistic)",
            "followups_tightening": "exploratory_followups_max is 0, a tightening of v1's 1: "
                                    "v2 is itself the engineering continuation of v1, so no "
                                    "exploratory follow-up of v2 is allowed",
            "optimism": "no gaps, drift, datum steps or trend; equal scale at every well; "
                        "anomalies exactly AR(1)",
            "tests": {"path": S.TESTS, "sha256_at_declaration": _sha(S.TESTS),
                      "binding": "disclosure only, not an input"}},
        "primary_model": {"layer": "E04's registered statistical layer (procedure re-run per "
                                   "world)", "truth": "AR(1) anomalies"},
        "thresholds": {"P_weak": S.P_WEAK, "K_STAT": S.K_STAT, "K_NULL": S.K_NULL,
                       "mc_half_width_90": V2.MC_HALF_WIDTH,
                       "rho_recovery_tol": S.RHO_RECOVERY_TOL},
        "interpretation": "A model-free operating characteristic of E04's registered test under "
                          "its own assumption, with a no-persistence reference. Nothing here is "
                          "evidence about the aquifer.",
        "written_by": S.PREREG,
        "zones": _na("no fit"), "burn_in": _na("no fit"),
        "folds": _na("synthetic worlds; no folds"),
        "gate_scores": _na("readings by one threshold; no weighted score"),
        "tests": _na("model-only declaration: synthetic worlds, no observation is scored and "
                     "no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "frozen_code": frozen, "inputs": sorted(set(inputs)),
    }
    return json.loads(json.dumps(doc, default=O._jsonable))


def _git_problems(paths) -> list[str]:
    out = []
    for p in paths:
        try:
            tracked = subprocess.run(["git", "-C", str(S.REPO), "ls-files", "--error-unmatch",
                                      p], capture_output=True, text=True).returncode == 0
            clean = subprocess.run(["git", "-C", str(S.REPO), "diff", "--quiet", "HEAD", "--",
                                    p], capture_output=True).returncode == 0
        except OSError as err:
            return [f"git: {err}"]
        if not tracked or not clean:
            out.append(f"{p} is not committed as-is (R7.commit_first)")
    return out


def readiness(d: dict, prereg_dir=None, now=None) -> list[str]:
    """Everything that must hold before the lead registers the draft (empty = ready)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if now >= dt.datetime.fromisoformat(S.E04_SCORABLE_FROM).replace(tzinfo=dt.timezone.utc):
        out.append(f"E04's window has closed ({S.E04_SCORABLE_FROM}): too late")
    if d["e04"]["protocol_sha256"] != S.E04_PROTOCOL_SHA:
        out.append("E04 protocol.json missing or not the registered one")
    if d["e04"]["manifest_sha256"] != S.E04_MANIFEST_SHA:
        out.append("E04 manifest.json missing or not the registered one")
    if d["v1"]["engineering_failure"]["sha256"] != V1_FAILURE_SHA:
        out.append("FV.stat.v1's engineering_failure.json missing or changed")
    if d["v1"]["declaration"]["sha256"] != V1_DECL_SHA:
        out.append("FV.stat.v1's declaration missing or changed")
    for k in ("v1_stat_preview", "r8_e04_parameters_disclosure"):
        if d["looks"][k] == PENDING:
            out.append(f"look {k} is not logged in looks.jsonl")
    if looks_file(prereg_dir).is_file():          # skipped when looks.jsonl is absent
        for k, (unit, _) in LOGGED_LOOKS.items():
            if d["looks"].get(k, PENDING) == PENDING:
                out.append(f"look {unit} ({k}) is not logged in looks.jsonl")
    if O.rp(S.REPORT).exists():
        out.append(f"{S.REPORT} exists before the declaration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    out += _git_problems([*S.FROZEN, S.PREREG])
    if S.index_record(S.ID, prereg_dir):
        out.append(f"{S.ID} is already registered")
    return out


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True, help="the lead's fresh seed")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        d = draft_prereg(a.seed, prereg_dir)
    except S.EngineeringFailure as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 1
    from harness import prereg3

    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("readiness:", readiness(d, prereg_dir) or "ready to register")
    print("this script sha256:", O.sha256(Path(__file__)))
    print("not written: the lead registers with prereg3.write_prereg after review")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
