"""Write N3F.ext_level.prospective_<window>.v1 (harness.prereg3, kind 'test'): the frozen 2022-12
head forecast against each well's own pre-origin climatology at the never-calibrated
external wells (N1:ext_wells of N1.cpu_response.v1), on fresh months.

User ruling 2026-10-29 (third set, c); judge rank 5 (R1b). A new evidence line, 'frozen
forecast vs own climatology at wells never used in calibration (prospective)'; NOT mapped to
E08 or C02. Its own registration (a different unit from E02c's, so I1's unit still gets one
companion only), scored AFTER E02c in the same session.

Windows: 2026-11..2027-10 (register before 2026-10-31T16:00Z) or the pre-declared fallback
2026-12..2027-11 (register before 2026-11-30T16:00Z, no fetch before then). New unit id per
window, N3F:ext_heads_<first>_<last>, members = the external wells kept by the rule (active
cell of the forward grid, layer code 1-4, >= 24 observed pre-origin months, residual sd from
>= 12 months) applied by ``export_baselines.py ext`` to pre-origin rows only, BEFORE any
availability look; no well is dropped afterwards.

Usage (repository root):
    python -I dataprep_staging/e02c/prereg_e02c_ext.py --validate --window 2026-11 [--ext-dir D]
    python -I dataprep_staging/e02c/prereg_e02c_ext.py --write --window 2026-11 --ext-dir D \
        --expect-self-sha <sha256 of this file>          (the lead only; this never logs a look)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prereg_e02c as P  # noqa: E402
import score_e02c as S  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
WP = "N3F"
UTC = dt.timezone.utc
WINDOWS = {
    "2026-11": {"first": "2026-11", "last": "2027-10",
                "deadline": dt.datetime(2026, 10, 31, 16, 0, tzinfo=UTC),
                "not_before": "2028-01-15", "role": "primary window"},
    "2026-12": {"first": "2026-12", "last": "2027-11",
                "deadline": dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC),
                "not_before": "2028-01-15", "role": "pre-declared fallback (no fetch of "
                                                    "any head after 2026-09 before it is "
                                                    "registered)"},
}
EXT_FILES = ("predictions.csv", "baselines.csv", "wells.csv", "protocol.json",
             "manifest.json", "summary.json", "verify.json")
# hashed inputs of this registration (E02c's own script is not one: a fallback-window edit
# here must not touch E02c, and an E02c edit must not touch this)
FROZEN_EXT = (f"{S.E}/score_e02c.py", f"{S.E}/export_baselines.py",
              f"{S.E}/prereg_e02c_ext.py")


def prereg_id(window: str) -> str:
    return f"N3F.ext_level.prospective_{window}.v1"


def test_id(window: str) -> str:
    return f"N3FP.ext_level_{window}"


def unit_id(window: str) -> str:
    w = WINDOWS[window]
    return f"N3F:ext_heads_{w['first']}_{w['last']}"


def n1_ext_ids() -> list[str]:
    d = json.loads((REPO / f"results/dataprep/preregistration/{S.N1_PREREG}.json")
                   .read_text(encoding="utf-8"))
    for t in d["tests"]:
        u = (t.get("external_units") or {}).get(S.N1_EXT_UNIT)
        if u:
            return sorted(map(str, u["ids"]))
    raise SystemExit(f"{S.N1_EXT_UNIT} not in {S.N1_PREREG}")


def ext_block(ext_dir: Path | None, window: str) -> dict:
    """The export outputs (hashed) and the kept wells; PENDING without an export."""
    import pandas as pd

    if ext_dir is None:
        return {"status": P.PENDING, "dir": None, "files": {}, "kept_ids": []}
    ext_dir = Path(ext_dir)
    files = {n: P._rec(ext_dir / n) for n in EXT_FILES if (ext_dir / n).is_file()}
    out = {"status": "ok" if len(files) == len(EXT_FILES) else "incomplete",
           "dir": rel_to_repo(ext_dir), "files": files, "kept_ids": []}
    if out["status"] != "ok":
        return out
    s = json.loads((ext_dir / "summary.json").read_text(encoding="utf-8"))
    bad = [n for n, h in (s.get("files") or {}).items() if files[n]["sha256"] != h]
    w = WINDOWS[window]
    if s.get("window") != [w["first"], w["last"]]:
        bad.append(f"export window {s.get('window')} != {[w['first'], w['last']]}")
    wells = pd.read_csv(ext_dir / "wells.csv", dtype={"station_id": str})
    kept = sorted(wells.loc[wells["kept"].astype(bool), "station_id"])
    pred = pd.read_csv(ext_dir / "predictions.csv", dtype={"station_id": str})
    if sorted(pred["station_id"].unique()) != kept:
        bad.append("predictions wells != kept wells")
    if not set(kept) <= set(n1_ext_ids()):
        bad.append("kept wells outside N1:ext_wells")
    v = json.loads((ext_dir / "verify.json").read_text(encoding="utf-8"))
    csv = {n: files[n]["sha256"] for n in files if n.endswith(".csv")}
    if v.get("identical") is not True or v.get("files") != csv or \
            v.get("summary_sha256") != files["summary.json"]["sha256"]:
        bad.append("verify.json: the export was not rebuilt bit-identically "
                   "(export_baselines.py ext --verify)")
    out.update({"kept_ids": kept, "n_candidates": len(wells), "n_kept": len(kept),
                "reasons": s.get("reasons"), "kept_by_county": s.get("kept_by_county")})
    if bad:
        out.update({"status": "mismatch", "problems": bad})
    return out


def doc(ext_dir: Path | None, window: str = "2026-11", prereg_dir=None) -> dict:
    w = WINDOWS[window]
    ex = ext_block(ext_dir, window)
    uid, tid = unit_id(window), test_id(window)
    unit = {"description": f"monthly heads {w['first']}..{w['last']} (flags 0-1) at the "
                          "external wells of N1:ext_wells (N1.cpu_response.v1: fan "
                          "monitoring wells outside the deliverable's 158 calibration wells) "
                          "kept by the rule (active cell, layer code 1-4, >= 24 pre-origin "
                          "months); never used in calibration",
            "ids": ex["kept_ids"], "window": f"{w['first']}..{w['last']}"}
    frozen = {p: S.sha256(REPO / p) for p in FROZEN_EXT if (REPO / p).is_file()}
    code_reg = {p: S.sha256(REPO / p) for p in P.CODE + P.CODE_AT_REGISTRATION}
    inputs = sorted({r["path"] for r in ex["files"].values()}
                    | {S.FORWARD, f"results/dataprep/preregistration/{S.N1_PREREG}.json"}
                    | set(frozen))
    e02c = P.index_record(S.ID, prereg_dir)
    return {
        "id": prereg_id(window), "wp": WP, "kind": "test",
        "title": f"Frozen 2022-12 head forecast vs own climatology at never-calibrated "
                 f"external wells, {w['first']}..{w['last']} (prospective)",
        "why": "Every prospective head test so far scores R0's 158 calibration wells. The "
               "external wells were never used to fit the page model; no retrospective "
               "window at them is unspent, so only fresh months can decide this line.",
        "tests": [{
            "test_id": tid, "data_scope": f"heads:{w['first']}..{w['last']}",
            "fresh_units": [uid], "external_units": {uid: unit},
            "what": "Does the frozen forecast (datum_gate_innov, datum and residual sd from "
                    "each well's own pre-origin record) beat each well's pre-origin "
                    "month-of-year climatology, with a 90 % interval that covers?",
            "metric": {"protocol": "E02's: equal-well RMSE ratio to the well's own "
                                   "pre-origin month-of-year climatology, and pooled central "
                                   "90 % combined-sigma coverage, computed by E02's own "
                                   "scorer 'hydrophysics.twin.prospective score' on the "
                                   "export (predictions, protocol, manifest), run by "
                                   "score_e02c.py score-ext after its score look is logged",
                       "hits": "challenge.monthly_observations per well (as E02), --new-wells "
                               "inside the window product whose MANIFEST grants "
                               "prospective_scorer; wells with >= 6 window months"},
            "decision_rule": {
                "passed": "RMSE ratio <= 1.0 AND coverage in [0.85, 0.95]",
                "not_supported": "otherwise",
                "not_informative": f"fewer than {S.MIN_WELLS_EXT} wells with >= "
                                   f"{S.MIN_MONTHS} window months, or no window product "
                                   "granting prospective_scorer"},
            "informative_gate": f">= {S.MIN_WELLS_EXT} wells x >= {S.MIN_MONTHS} months",
            "label": "prospective: the window's heads did not exist at registration"}],
        "external_predictions": ex, "ext_predictions": {"dir": ex["dir"]},
        "selection_rule": "N1:ext_wells ids (138) -> active cell of the forward grid and "
                          "layer code 1-4 (station_meta_preorigin), >= 24 observed months "
                          "2012-01..2022-12 (flags 0-1), residual sd from >= 12 months; "
                          "applied to pre-origin rows only (export_baselines.py ext), before "
                          "any availability look; no well dropped afterwards",
        "availability_look": "after registration only: the lead may log a look "
                             "(prereg3.record_look, purpose 'availability_metadata', unit "
                             f"{uid}) reading station-metadata end dates, never values; it "
                             "changes no well and no rule and is listed in "
                             "post_registration_looks at scoring",
        "e02c": {"id": S.ID, "registration": e02c or "not yet registered"},
        "scoring": {"not_before": w["not_before"],
                    "order": "after E02c in the same 2028-01-15..31 session: E02c's "
                             "report.json (any verdict, including not_evaluable) or, if E02c "
                             "was refused or never scorable, a lead-written record naming "
                             f"{S.ID} with verdict 'not_scorable' must exist (a report that "
                             f"scored must have its {S.TEST_COV} and {S.TEST_LEVEL} score "
                             "looks logged); the scorer refuses otherwise. If the fallback "
                             "window's last month is not yet published by then, in the "
                             "first later session, still after E02c",
                    "run_order": "(1) MANIFEST check (metadata only); (2) E02c order check; "
                                 "(3) prereg3.assert_can_score (logs the look); (4) only then "
                                 "E02's scorer reads the window heads",
                    "engineering": P.ENGINEERING_RESTORE,
                    "precondition": "the window heads product MANIFEST lists "
                                    "prospective_scorer (flags 0-1); otherwise "
                                    "not_informative and no value read",
                    "scorer": f"{S.E}/score_e02c.py score-ext --window {window} "
                              "--window-product <dir> --new-wells <dir inside it> "
                              "--e02c-report <E02c report.json or record> --out <dir>",
                    "session_order": list(S.SESSION_ORDER)},
        "page_mapping": {
            "before_scoring": "new evidence row (D, pending): 'level forecast at "
                              "never-calibrated wells, decided not before 2028-01-15'",
            "passed": "S (prospective) for this line only",
            "not_supported": "'frozen forecast does not beat own climatology at unseen "
                             "wells'",
            "not_informative": "'undecided'",
            "never": "E08 and C02 are unchanged under every outcome",
            "wording": {k: v for k, v in S.WORDING.items() if k.startswith("ext_")}},
        "disclosures": {
            "availability_risk": "the heads/L1_clean v3 product card lists post-origin "
                                 "records at 158 wells only (the calibration wells); the "
                                 "external wells may not report in the window, giving "
                                 "'not informative'; known before registration, not looked at",
            "U7": "datum, sd and climatology come from 2012-01..2022-12 only; the 2023-26 "
                  "heads at these wells stay unopened; the combined-sigma family was chosen "
                  "after E02's 2023-26 coverage was seen (E02c disclosure applies)",
            "prior_looks": "N1.cpu_response.v1 scored these wells on 2013-01..2022-12 "
                           "(disjoint window)",
            "multiplicity": "one comparison on a new unit; no family-wise claim"},
        "attempt_policy": "User rule 2026-10-23: engineering retries before the scoring look "
                          "only, logged; scored ONCE; a fail or not-informative result is "
                          "reported as such and never re-scored on these units; at most ONE "
                          "post-failure exploratory follow-up with new ids, never validation; "
                          "every attempt reported",
        "rulings": {"2026-10-29 third set (c)": "add a prospective level test at the "
                                                "never-calibrated external wells, scored "
                                                "after E02c in the same session; fallback "
                                                "window 2026-12"},
        "window_role": w["role"],
        "frozen_code": frozen,
        "code_at_registration_report_only": code_reg,
        "zones": NA("no fit"), "priors": NA("see disclosures"), "burn_in": NA("no fit"),
        "folds": NA("prospective window"), "fresh_units": [uid],
        "thresholds": {"max_rmse_ratio": 1.0, "coverage_bounds": [0.85, 0.95],
                       "min_wells": S.MIN_WELLS_EXT, "min_months": S.MIN_MONTHS,
                       "min_fit_months": 24, "min_sd_months": 12},
        "gate_scores": NA("the decision rule is reported as it falls"),
        "primary_model": {"name": "datum_gate_innov frozen forward run",
                          "forward_sha256": S.MODEL_SHA},
        "inputs": inputs,
        "interpretation": "Whether the frozen forecast beats each unseen well's own "
                          "climatology on fresh months. Not E08 (unseen wells vs "
                          "interpolation), not C02 (head differences), nothing about "
                          "subsidence or policy.",
        "written_by": f"{S.E}/prereg_e02c_ext.py",
    }


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--window", choices=sorted(WINDOWS), default="2026-11")
    ap.add_argument("--ext-dir", type=Path, default=None)
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.ext_dir, a.window, prereg_dir)
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    block = P.check_blocking(d, prereg_dir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", P._sha_text(d))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("ext export:", d["external_predictions"]["status"], "| kept wells:",
          len(d["external_predictions"]["kept_ids"]))
    self_sha = S.sha256(Path(__file__))
    print("this script sha256:", self_sha)
    if not a.write:
        return 1 if errs else 0
    bad = []
    if d["external_predictions"]["status"] != "ok":
        bad.append(f"ext export: {d['external_predictions']['status']}")
    if len(d["external_predictions"]["kept_ids"]) < S.MIN_WELLS_EXT:
        bad.append(f"fewer than {S.MIN_WELLS_EXT} kept wells: the test could never be "
                   "informative")
    if a.expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {a.expect_self_sha!r} != {self_sha}")
    if S._now() >= WINDOWS[a.window]["deadline"]:
        bad.append(f"deadline for window {a.window} passed")
    if set(d["frozen_code"]) != set(FROZEN_EXT):
        bad.append("frozen code incomplete")
    if errs or block or bad:
        print("REFUSED:", errs + [str(b) for b in block] + bad)
        return 1
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
