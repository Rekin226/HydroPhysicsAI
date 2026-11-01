"""Write N3Gprime.piles_conversion.v1 (harness.prereg3, kind 'test'): the second, gating role
of N3's printed pile months for the conversion (judge rank 14's N3G'.v1), ONLY if G0' passed.

A NEW registration that cites N3.piles_prospective.v1 (never edited; its conversion lines stay
report-only there) and user ruling R7.n3_g0prime ('Yes, both, registered by
2026-11-30T16:00Z'). The estimator and the horizon are the ones G0' selected by its declared
rule (N3.G0prime.v1: first passing estimator in the order harmonic_ratio, iv_harmonics,
iv_model_heads at 12 months, else at 24); the decision rule (N2's, through
n3lib.conversion_verdict), the informative gate, the data scope, the fresh units and their
members, and the page mapping of the conversion outcomes are copied unchanged from N3's
registered document. The other two estimators and N3's OLS are report-only lines.

``--write`` refuses unless: N3 and N3.G0prime.v1 are registered and intact (N3 and the ruling
file at their pinned shas); the G0' result is a full run (500 reps, B 2000, not smoke) of the
declared code on the declared model_side.json, gated by the registered declaration (its
declaration_index_record names N3.G0prime.v1 with the registered sha), under the declared
gate mode, and its decision passed; the ruling names N3.piles_prospective.v1; it is before
2026-11-30T16:00Z; no pile look reaches 2026-02 or later; no blocking look; validation passes;
--expect-self-sha is this file's sha256.

Usage (repository root):
    python -I dataprep_staging/n3_g0prime/prereg_n3g.py --validate
    python -I dataprep_staging/n3_g0prime/prereg_n3g.py --write \
        --expect-self-sha <sha256 of this file>                          (the lead only)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "n3_piles"),
           str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0p_lib as P  # noqa: E402
import g0prime as GP  # noqa: E402
import prereg_g0prime as D  # noqa: E402
import prereg_n3 as PN  # noqa: E402  (N3's pile-look and blocking checks, cited)
from common.paths import from_repo, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

L = P.L
NA = prereg3.not_applicable
PENDING = "PENDING"
FROZEN = (f"{P.E}/g0p_lib.py", f"{P.E}/score_n3g.py", f"{P.E}/prereg_n3g.py",
          "dataprep_staging/n3_piles/n3lib.py", "dataprep_staging/n3_piles/score_n3.py",
          "dataprep_staging/n3_piles/well_columns.py", "dataprep_staging/n3_piles/g0_n3.py",
          L.SHARED_SCORER)
TEST_OF = {12: P.TEST_12, 24: P.TEST_24}
SCOPE_OF = {12: L.SCOPE_CONV12, 24: L.SCOPE_CONV24}
N3_TEST_OF = {12: L.TEST_CONV, 24: L.TEST_CONV24}
ESTIMATOR_TEXT = {
    "harmonic_ratio": "per pile the annual harmonic (sin, cos of the calendar month, pile "
                      "intercept) of the printed increment c and of the nest dD; g_obs = "
                      "sign(sum_p b_c,p . b_D,p) x sqrt(sum_p |b_c,p|^2 / sum_p |b_D,p|^2) "
                      "(g0p_lib.estimate_all 'harmonic_ratio')",
    "iv_harmonics": "2SLS of c on dD with a pile fixed effect, instruments the per-pile annual "
                    "sin and cos of the calendar month (g0p_lib 'iv_harmonics')",
    "iv_model_heads": "2SLS of c on dD with a pile fixed effect, instrument the frozen ZR "
                      "model's nest dD at the same months (N3's model_side.json world; "
                      "g0p_lib 'iv_model_heads')"}


# ----------------------------------------------------------------------------- slots
def g0prime_block(result: Path | None, decl: dict) -> dict:
    """The G0' result: a full run of the declared code on the declared model_side.json, gated
    by the registered declaration ``decl`` (prereg_g0prime.registered: sha256 + doc)."""
    if result is None or not Path(result).is_file():
        return {"status": PENDING if result is None else "missing", "decision": None}
    r = json.loads(Path(result).read_text(encoding="utf-8"))
    ddoc = decl.get("doc") or {}
    par = r.get("parameters") or {}
    out = {"status": "ok", "result": {"path": rel_to_repo(result),
                                      "sha256": P.sha256(result)},
           "decision": r.get("decision"), "gate": r.get("gate"),
           "lines_12": {k: {"pass": v["pass"], "n_fails": v["n_fails"],
                            "class_pass": v.get("class_pass")}
                        for k, v in (r.get("horizon_12") or {}).get("lines", {}).items()},
           "lines_24": {k: {"pass": v["pass"], "n_fails": v["n_fails"],
                            "class_pass": v.get("class_pass")}
                        for k, v in (r.get("horizon_24") or {}).get("lines", {}).items()},
           "g_mod": {h: (r.get(f"horizon_{h}") or {}).get("g_mod") for h in (12, 24)},
           "model_side_f": {h: (r.get(f"horizon_{h}") or {}).get("model_side_f")
                            for h in (12, 24)},
           "g_mod_ols_equals_registered": {h: (r.get(f"horizon_{h}") or {}).get(
               "g_mod_ols_equals_registered") for h in (12, 24)}}
    rec = r.get("declaration_index_record") or {}
    want = {p: s for p, s in (ddoc.get("frozen_code") or {}).items() if p in GP.CODE_FILES}
    if rec.get("id") != P.ID or not decl.get("sha256") or rec.get("sha256") != decl["sha256"]:
        out["status"] = "G0' was not run under the registered declaration " \
                        "(declaration_index_record)"
    elif r.get("id") != P.ID or r.get("smoke") is not False or par.get("reps") != P.REPS \
            or par.get("boot_B") != L.BOOT_B:
        out["status"] = "not a full G0' run"
    elif not want or r.get("code") != want:
        out["status"] = "G0' was run with code other than the declaration's"
    elif r.get("model_side", {}).get("sha256") != (ddoc.get("model_side") or {}).get("sha256"):
        out["status"] = "G0' was run on another model_side.json"
    elif (r.get("gate") or {}).get("mode") != (ddoc.get("design") or {}).get("gate_mode"):
        out["status"] = "G0' was run under another gate mode than the declaration's"
    elif not all(out["g_mod_ols_equals_registered"].values()):
        out["status"] = "the OLS g_mod of G0' differs from N3's registered g_mod"
    else:
        again = GP.decision(r["horizon_12"], r["horizon_24"])
        if {k: again.get(k) for k in ("passed", "horizon_months", "estimator")} != \
                {k: (r.get("decision") or {}).get(k) for k in ("passed", "horizon_months",
                                                               "estimator")}:
            out["status"] = "the stored decision does not follow from the stored line passes"
    return out


def attempts(root: Path | None = None) -> dict:
    """Every G0' attempt: attempts.jsonl and the archived runs (result sha, reason)."""
    root = Path(root) if root else from_repo(P.OUT_DIR)
    log = []
    if (root / "attempts.jsonl").is_file():
        log = [json.loads(s) for s in (root / "attempts.jsonl").read_text(
            encoding="utf-8").splitlines() if s.strip()]
    def archived(sub: str) -> list:
        runs = []
        if (root / sub).is_dir():
            for d in sorted(p for p in (root / sub).iterdir() if p.is_dir()):
                why = (d / "REASON.txt").read_text(encoding="utf-8").strip() \
                    if (d / "REASON.txt").is_file() else None
                runs.append({"archive": d.name, "reason": why,
                             "results": [{"file": f.relative_to(d).as_posix(),
                                          "sha256": P.sha256(f)}
                                         for f in sorted(d.rglob("result.json"))]})
        return runs
    runs = archived("archive")
    rings = archived("archive_rings")
    return {"log": log, "archived": runs, "n_archived": len(runs),
            "ring_note_archived": rings, "n_ring_note_archived": len(rings)}


def _n3_test(n3doc: dict, tid: str) -> dict:
    hit = [t for t in n3doc.get("tests") or [] if t.get("test_id") == tid]
    if len(hit) != 1:
        raise ValueError(f"N3's registration has no single test {tid}")
    return hit[0]


# ----------------------------------------------------------------------------- the document
def doc(g0p_result: Path | None = None, ruling: Path | None = None, prereg_dir=None,
        rows_dir=None, rulings_dir=None, out_root: Path | None = None) -> dict:
    n3r = D.registered(L.ID, prereg_dir, rows_dir)
    decl = D.registered(P.ID, prereg_dir, rows_dir)
    root = Path(out_root) if out_root else from_repo(P.OUT_DIR)
    res = g0p_result if g0p_result is not None else root / "g0prime" / GP.RESULT
    g0 = g0prime_block(res, decl) if decl["status"] == "ok" else \
        {"status": f"declaration: {decl['status']}", "decision": None}
    g0["attempts"] = attempts(root)
    rb = D.ruling_block(ruling if ruling else from_repo(P.RULING_FILE), P.RULING_ID, L.ID,
                        rulings_dir)
    dec = g0.get("decision") or {}
    h = dec.get("horizon_months") if dec.get("passed") else None
    est = dec.get("estimator") if dec.get("passed") else None
    n3doc = n3r.get("doc") or {}
    frozen = {p: P.sha256(REPO / p) for p in FROZEN if (REPO / p).is_file()}
    n3ms = ((n3doc.get("model_side") or {}).get("files") or {})
    ms_paths = {n: (n3ms.get(n) or {}).get("path") for n in ("mapping.json", "model_side.json")}
    tests = []
    if n3doc and h is not None:
        for hz in (12, 24):
            base = _n3_test(n3doc, N3_TEST_OF[hz])
            gm = (g0.get("g_mod") or {}).get(hz) or {}
            role = "gating" if hz == h else "report_only"
            tests.append({
                "test_id": TEST_OF[hz], "data_scope": base["data_scope"],
                "fresh_units": list(base["fresh_units"]),
                "external_units": copy.deepcopy(base["external_units"]),
                "seed": L.BOOT_SEED, "role": role,
                "label": "prospective: the window's pile months did not exist at "
                         "registration; second role of N3's pile months (ruling "
                         "R7.n3_g0prime)",
                "claim": "C31: compaction per metre of head change at the 2 Yunlin deep-pile "
                         "cells" + ("" if hz == 12 else " (24-month)"),
                "window": list(P.HORIZONS[hz]),
                "metric": {"observed": base["metric"]["observed"],
                           "estimator": est, "estimator_definition": ESTIMATOR_TEXT[est],
                           "g_mod": gm.get(est),
                           "g_mod_definition": "the same estimator on N3's model world (the "
                                               "frozen ZR column increment on the DISPLAYED "
                                               "ZR nest dD at the window months), a frozen "
                                               "number from the G0' result",
                           "rho": "g_obs / g_mod",
                           "interval": f"90 % percentile interval of a {L.BOOT_BLOCK}-month "
                                       f"moving-block bootstrap within pile, B {L.BOOT_B}, "
                                       f"seed {L.BOOT_SEED} (g0p_lib.estimate_all: the same "
                                       "resampled rows for every line); rho interval = that "
                                       "interval / g_mod",
                           "report_only": {"lines": [k for k in P.ALL_LINES if k != est],
                                           "g_mod": {k: gm.get(k) for k in P.ALL_LINES
                                                     if k != est},
                                           "first_stage_f": "observed first-stage F of "
                                                            "both instrument sets"}},
                "decision_rule": copy.deepcopy(base["decision_rule"]),
                "informative_gate": base["informative_gate"],
                "metric_shared_code": f"{P.E}/g0p_lib.py",
                "decision_shared_code": L.SHARED_SCORER})
    n3pm = n3doc.get("page_mapping") or {}
    page = {
        "before_scoring": {
            "C31": "D: 'compaction per metre of head change at the 2 Yunlin deep-pile cells: "
                   f"decided by N3G (not before {'2028' if h == 12 else '2029'}-01-15)'",
            "C27": "conversion clause: 'decided by N3G (pending)'",
            "C09_C30": "unchanged (N3's level sub-label)",
            **({"C31_caveat": "'assumes the nest's seasonal swing tracks the pile cell's (a "
                              "mismatch cannot be detected from nest heads)' (gate_mode "
                              "identifiable)"}
               if (g0.get("gate") or {}).get("mode") == "identifiable" else {})},
        "conversion_copied_from_N3": copy.deepcopy(n3pm.get("conversion")),
        "reading": "the verdict in conversion_copied_from_N3 is N3G's gating test's; N3's own "
                   "conversion lines stay report-only text (N3.level_only); a report-only N3G "
                   "line never moves C31",
        "never": n3pm.get("never"),
        "ring_note": "the ring attenuation N note (N3.G0prime.v1) stays N"}
    session = [
        f"(11) {L.ID}: its step-(11) score looks ({N3_TEST_OF[12]} and {L.TEST_LEVEL}; "
        f"24 months: {N3_TEST_OF[24]} and {L.TEST_LEVEL24}) are logged first",
        f"(11b) {P.N3G_ID}: same session, immediately after (11); score_n3g.py refuses "
        "until N3's step-(11) score looks exist",
        "(12) onward: as declared by N3 and E02c"]
    inputs = sorted(set(frozen)
                    | {p for p in ms_paths.values() if p}
                    | ({g0["result"]["path"]} if "result" in g0 else set())
                    | ({decl["path"]} if decl["status"] == "ok" else set())
                    | ({n3r["path"]} if n3r["status"] == "ok" else set())
                    | ({rb["file"]} if rb.get("status") == "ok" else set()))
    d = {
        "id": P.N3G_ID, "wp": P.WP, "kind": "test",
        "title": "N3G': the conversion at the 2 Yunlin deep piles with the estimator G0' "
                 "selected (second, gating role of N3's printed pile months, ruling "
                 "R7.n3_g0prime)",
        "why": "N3's OLS conversion failed its G0 by attenuation and is report-only in N3; "
               "G0' (N3.G0prime.v1) checked three attenuation-robust estimators; this "
               "registration gives N3's pile months a second, gating role for the one it "
               "selected.",
        "tests": tests if tests else NA("G0' did not pass (or is pending): nothing gates"),
        "roles": {t["test_id"]: t["role"] for t in tests},
        "selected": {"estimator": est, "horizon_months": h,
                     "gate_mode": (g0.get("gate") or {}).get("mode"),
                     "rule": "N3.G0prime.v1 design.decision (declared before G0' ran)"},
        "binds": {"forward": {"path": L.FORWARD, "sha256": L.FORWARD_SHA},
                  "column": {"path": L.COLUMN, "sha256": L.COLUMN_SHA},
                  "model_side": ms_paths},
        "n3_registration": {k: n3r.get(k) for k in ("status", "sha256", "path")},
        "g0prime_declaration": {k: decl.get(k) for k in ("status", "sha256", "path")},
        "g0prime": g0,
        "frozen_code": frozen,
        "scoring": {
            "not_before": "2028-01-15 (12 months); 2029-01-15 (24 months)",
            "no_later_than": "2029-06-30 (12 months); 2030-06-30 (24 months); later: "
                             "not_informative",
            "precondition": "as N3: the pile product (deep_benchmarks_printed v3, future "
                            "months only) and the window heads product grant "
                            "prospective_scorer (MANIFEST only); otherwise not evaluable, "
                            "nothing read, no look",
            "run_order": "(1) MANIFEST checks; (2) N3's step-(11) score looks logged; (3) "
                         "prereg3.assert_can_score for this horizon's test (logs the look); "
                         "(4) only then pile and head rows are read",
            "same_session_order": session,
            "scorer": f"{P.E}/score_n3g.py score --horizon <12|24> --piles-product <v3> "
                      "--heads-product <window heads> --out <dir>"},
        "sealed_until_scoring": "every pile month after 2026-01 and every window head (as N3)",
        "page_mapping": page,
        "rulings": {P.RULING_ID: rb,
                    "N3.level_only (2026-11-01)": "N3's conversion lines stay report-only in "
                                                  "N3; N3 is never edited",
                    "2026-10-29 (4)": "no calibrated correction factor follows from any "
                                      "verdict",
                    "2026-11-01 fourth set (rank 6)": "only C31 can become S; no cut number"},
        "disclosures": {
            "second_role": "the same pile months and nest heads are N3's fresh units; N3 "
                           "reads them for its level test and its report-only OLS "
                           "conversion in the same session; the second gating role is the "
                           "user's ruling R7.n3_g0prime",
            "g0_history": "N3 G0 (OLS) failed (sha 80e260d0); G0' was declared before its "
                          "first run (N3.G0prime.v1) and every G0' attempt is listed under "
                          "g0prime.attempts",
            "selection": "the estimator was selected on synthetic worlds only, by the order "
                         "declared before G0' ran; the other estimators are report-only",
            "priors_both": (n3doc.get("disclosures") or {}).get("priors_both"),
            "column_pairing": (n3doc.get("disclosures") or {}).get("column_pairing"),
            "lag": "lag 0 only (as N3); delayed drainage and inelastic compaction below past "
                   "lows are not in the decision"},
        "attempt_policy": n3doc.get("attempt_policy", PENDING),
        "zones": NA("no fit; the frozen forward run's own zones"),
        "priors": "G0' passed in N3's model world for the selected estimator; the rings' "
                  "3.4 vs 9.3 mm/m and the 0.92 pile line point in opposite directions "
                  "(N3's disclosure); neither set a threshold here",
        "burn_in": NA("no fit"), "folds": NA("prospective window"),
        "fresh_units": [L.UNIT, L.UNIT_HEADS_YL],
        "thresholds": {"ratio_band": L.RATIO_BAND, "level": L.LEVEL, "boot_B": L.BOOT_B,
                       "boot_block": L.BOOT_BLOCK, "boot_seed": L.BOOT_SEED,
                       "min_months": L.MIN_MONTHS, "min_months_24": L.MIN_MONTHS24,
                       "min_rms_dd_m": L.MIN_RMS_DD_M, "min_rms_inc_cm": L.MIN_RMS_INC_CM},
        "gate_scores": NA("one decision rule per test (N2's)"),
        "primary_model": {"name": "the page model (stage3_datum_gate + coupled_leveling, "
                                  "datum_gate_innov ZR), as N3",
                          "forward_sha256": L.FORWARD_SHA, "column_sha256": L.COLUMN_SHA},
        "external_inputs": (n3doc.get("external_inputs") or PENDING),
        "inputs": inputs,
        "interpretation": "Whether the page column's compaction per metre of head change "
                          "holds at the two Yunlin deep-pile cells on fresh months, measured "
                          "with the attenuation-robust estimator G0' selected. Nothing about "
                          "cut sizes, the coast, the long run or Changhua subsidence.",
        "written_by": f"{P.E}/prereg_n3g.py"}
    return d


def problems_for_write(d: dict, expect_self_sha, prereg_dir=None) -> list[str]:
    bad = []
    for k in ("n3_registration", "g0prime_declaration", "g0prime"):
        if d[k].get("status") != "ok":
            bad.append(f"{k}: {d[k].get('status')}")
    if not (d["g0prime"].get("decision") or {}).get("passed"):
        bad.append("G0' did not pass: N3G is not registered (C31 keeps N)")
    rb = d["rulings"].get(P.RULING_ID) or {}
    if rb.get("status") != "ok":
        bad.append(f"ruling {P.RULING_ID}: {rb.get('status')}")
    if d["n3_registration"].get("status") == "ok" and \
            d["n3_registration"].get("sha256") != P.N3_REG_SHA:
        bad.append(f"N3 registration sha {d['n3_registration'].get('sha256')} != pinned "
                   f"{P.N3_REG_SHA}")
    if rb.get("status") == "ok" and rb.get("sha256") != P.RULING_SHA:
        bad.append(f"ruling file sha {rb.get('sha256')} != pinned {P.RULING_SHA}")
    if set(d["frozen_code"]) != set(FROZEN):
        bad.append(f"frozen code incomplete: {sorted(set(FROZEN) - set(d['frozen_code']))}")
    late = PN.pile_looks(prereg_dir)["late"]
    if late:
        bad.append(f"pile looks whose scope reaches 2026-02 or later: {late[:3]}")
    if P.sha256(Path(__file__)) != expect_self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != this script "
                   f"{P.sha256(Path(__file__))}")
    if L._now() >= P.DEADLINE:
        bad.append(f"DEADLINE PASSED ({P.DEADLINE:%Y-%m-%dT%H:%MZ})")
    return bad


def main(argv=None, prereg_dir=None, rows_dir=None, rulings_dir=None, out_root=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="write the registration (the lead)")
    ap.add_argument("--g0prime-result", type=Path, default=None)
    ap.add_argument("--ruling", type=Path, default=None)
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.g0prime_result, a.ruling, prereg_dir, rows_dir, rulings_dir, out_root)
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    errs = prereg3.validate_prereg(d, prereg_dir=pdir)
    block = PN.check_blocking(d, pdir) if isinstance(d["tests"], list) else []
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", D.sha_text(d))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("slots: N3", d["n3_registration"]["status"], "| G0' declaration",
          d["g0prime_declaration"]["status"], "| G0' result", d["g0prime"]["status"],
          "| selected", json.dumps(d["selected"]), "| ruling",
          d["rulings"][P.RULING_ID]["status"])
    print("G0' attempts: archived", d["g0prime"]["attempts"]["n_archived"])
    print("this script sha256:", P.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = problems_for_write(d, a.expect_self_sha, prereg_dir)
    if errs or block or bad:
        print("REFUSED:", errs + [str(b) for b in block] + bad)
        return 1
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
