"""Declaration DRAFT of PB.decomposed_band.v1 (round-8 rank 16, PB part; user ruling R8.13;
harness.prereg3 schema, kind 'declaration'). Nothing here writes a registration: ``--validate``
prints the draft, its canonical sha256, prereg3.validate_prereg's errors and the readiness
problems. The LEAD registers after the code is merged (R7.commit_first), from a checkout that
holds the merged code and the results/ symlinks:

    python -I dataprep_staging/pb_band/prereg_pb.py --validate --twin-dir <results/twin>
    python -I -c "import sys; sys.path[:0] = ['dataprep_staging', 'dataprep_staging/pb_band'];
        import prereg_pb as P; from harness import prereg3;
        print(prereg3.write_prereg(P.draft_prereg()))"

and then builds once (build_pb.py refuses before that):

    python -I dataprep_staging/pb_band/build_pb.py build --twin-dir <results/twin>

The draft fixes the ribbon construction, the zone minimum, the leave-one-family-out estimand,
the inputs and their sha256 (results/twin files are bound by hash under binds.files, because a
checkout's results/twin may be partial; every other input is hashed by write_prereg), the U7
guard and the page mapping. ``draft_prereg`` reads no data file: hashes come from pb_const
(upstream registrations' records), code hashes from the files themselves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d"),
           str(HERE.parent / "fb_family"), str(HERE.parent / "fv_decidability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_pb as B  # noqa: E402
import e02dio as io  # noqa: E402
import pb_const as K  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

PLAN_ENTRY = (
    "round-8 plan #16 (PB): 'a new module reads E02d's sigma_idio_A/B/C.csv by hash (identity "
    "check 1e-9 m). Draw two separately labelled parts: a per-well zone ribbon ('decided by a "
    "registered test, E02d, 2028'; minimum 5 wells per zone; no station ids) and the shared "
    "range ('cannot be tested before 2030 by design'). Publish leave-one-family-out coverage "
    "per county. Drop leave-one-forcing-year-out. The U7 guard (no 2023-26 coverage of the "
    "relative recipe) lives in the PB builder and its tests, never in coverage.py, bands.py or "
    "coverage3.py.' Register before look: 'PB declaration before build'. Gate: 'PB identity "
    "check'.")
BUILDER_READS = [
    "docs/TWIN_ROUND8_PLAN.local.md (#16 and the execution notes)",
    f"{K.RULINGS} (R8.13, verbatim)",
    f"{K.PREREG}/{K.E02D_ID}.json (binds.dr, interval, inputs, page_mapping, "
    "sealed_until_scoring, tests, thresholds)",
    f"{K.PREREG}/{K.E02Y_ID}.json (page_mapping, tests' hit rules, scoring, inputs' sha256)",
    f"{K.PREREG}/{K.FV2_ID}.json (truths: tags and v1_forward sha256)",
    f"{K.PREREG}/INDEX.jsonl (sha256 of E02d, E02y, FV v2, DR, U1.decomp_power, "
    "U1.ext_transfer_power, FB.family_power)",
    f"{K.FB_RESULT} and {K.U1X_RESULT} (decisions only: the gates failed)",
    "results/twin/e02d_relative/summary.json (decisions, files, inputs' sha256)",
    "results/twin/e02d_relative/sigma_idio_{A,B,C}.csv: layout (rows, months, wells per DR "
    "zone and layer, lead range) and, once, the median half_width_m per DR zone of each file "
    "(E02d's frozen model-derived values; printed during engineering, see disclosures)",
    f"{K.MEMBERS_NPZ}: array names and shapes only; {K.WELLS_E02D}: header only",
    "results/twin_forward/fv_truth_R0.npz: nothing (numpy was not on the default interpreter's "
    "path; FV's constants T_RUN / HORIZON were read from osse_fv.py instead)",
    "NOT read or computed: any observed head, any member deviation, family departure, shared "
    "range or LOFO number, any coverage of anything"]
PRINTED_ZONE_MEDIANS_M = {"A": (2.157, 4.904), "B": (2.267, 4.954), "C": (2.408, 5.014)}
ZONE_WELLS = {"Changhua:mid+dist": 61, "Yunlin:dist": 40, "Yunlin:mid": 34,
              "Changhua:prox": 13, "Yunlin:prox": 10}
DISCLOSURES = {
    "engineering_print": "while inspecting the artefact layout on 2026-11-06 the builder "
                         "printed, once, the median half_width_m per DR zone of each "
                         "sigma_idio file, pooled over the zone's wells and the window's "
                         "months (E02d's registered model-derived values, no observation): "
                         "zone medians ranged "
                         + "; ".join(f"{w} {lo:.3f}-{hi:.3f} m" for w, (lo, hi)
                                     in PRINTED_ZONE_MEDIANS_M.items())
                         + ". The ribbon statistic (the median over the zone's wells, per "
                         "month) was the builder's choice, made with this print in hand. The "
                         "5-well zone minimum is R8.13's and removes no zone: the five zones "
                         "hold " + "/".join(str(n) for n in ZONE_WELLS.values())
                         + " of the 158 wells. Besides this print, no member deviation, "
                         "family departure, shared range, LOFO figure or coverage was "
                         "computed",
    "known_to_designers": "FV v2's published per-truth Changhua coverage split (Q0 0.004, R0 "
                          "0.012, R1c 0.992, X5R1 0.996) implies that in Changhua the families "
                          "{R0,Q0} and {R1c,X5R1} depart differently from E02's forward, so "
                          "leaving out an extreme family may leave it outside the hull of the "
                          "others; no departure was computed before this declaration",
    "u7": "E02d's U7 disclosure carries over: the interval family was chosen after its "
          "2023-26 coverage was seen; the ribbon is E02d's frozen relative form of it. PB "
          "computes no coverage of the relative recipe in any month",
    "model_only": "the shared range and its LOFO coverage are model-only and are the spread "
                  "among model variants only (the 36 rerun members and the FV v1 baseline "
                  "forwards of FV v2's four truth families, which share the model family): "
                  "they exclude forcing-year variation (FV v2's registered forcing-year line "
                  "stays on the page beside the range) and the between-zone part of the "
                  "departure; a high LOFO coverage says the variants agree with each other, "
                  "not that the range holds the aquifer's county-wide departure; the range is "
                  "not a predictive interval",
    "window_A_only": "FV v1's truth forwards end in 2027-10, so the shared range and the LOFO "
                     "coverage exist for window A only; the page draws window A's ribbon only "
                     "(lead decision B1/B2); B and C are computed in the output, not drawn",
    "zones": "the ribbon's zones are DR.relative_drift.v1's frozen groups (E02d's relative "
             "operator removes their leave-one-out median); the shared range is per county "
             "(the county median of the per-well parts), as the page draws it, so the "
             "between-zone part of the departure is in neither part",
    "zr0": "fv_zr0.npz (FV's same-code reference ZR0, read by fb_inputs.load_mu for FV's "
           "report-only ZR0 -> ZR offset, which PB does not use) is bound at sha256 a1833d5f, "
           "a hash recorded only by the FB builder on 2026-11-03; no registration records it",
    "multiplicity": "PB scores no test; E02d decides the drawn ribbon's wording",
}


def page_mapping() -> dict:
    lab = K.LABELS
    return {
        "replaces": "the drawn aggregate run spread of the head series only (the fan-average "
                    "layer-2 head's p10-p90 / p25-p75 run band, template bandKind 'runs_head', "
                    "and its note bn_head; ledger C14c), per ruling R8.13",
        "kept": {"bn_head_fy_pb": {
                     "what": "under PB the page shows ONLY the forcing-year clause of FV v2's "
                             "registered C14c wording, verbatim (first letter capitalised), as "
                             "template key bn_head_fy_pb beside the shared range; no "
                             "run-spread sentence (ruling R7.fv_v2_page)",
                     "registration": {"id": K.FV2_ID, "sha256": K.FV2_SHA,
                                      "wording": K.FV2_C14C_PATH},
                     "clause": {"en": K.FV2_C14C_CLAUSE_EN, "zh": K.FV2_C14C_CLAUSE_ZH},
                     "bn_head_fy": "the full FV v2 line (with the run-spread sentence) is "
                                   "shown only when PB is absent",
                     "see_line": "the shared range's '(see FV v2's line below)' is shown only "
                                 "when the clause is shown"},
                 "cov_line": "the head series' registered coverage line (covLine) stays under "
                             "bn_head_pb; PB removes no registered line"},
        "part_1_ribbon": {
            "rows": {"A": "C14a-r"},
            "windows_drawn": list(K.DRAWN_RIBBON_WINDOWS),
            "not_drawn": "windows B and C are computed in the output (drawn = false), never "
                         "on the page",
            "what": lab["ribbon_what"], "label_A": lab["ribbon_A"],
            "small_zones": f"a zone with fewer than {K.MIN_ZONE_WELLS} wells is not drawn "
                           "and the page says so in one line (pb_small, EN and ZH; none at "
                           "present)",
            "zone_names": "DR.relative_drift.v1's group names as stored in E02d's artefact "
                          "(e.g. 'Changhua:mid+dist', 'Yunlin:prox'), shown untranslated in "
                          "both languages",
            "draw": "per zone (zone name and well count, never a station id), +/- the zone's "
                    "median half-width around zero, monthly, 2026-12..2027-10",
            "after_scoring": "E02d's own page_mapping decides the wording; on a county's fail "
                             "its ribbon is withdrawn (rel_fail); PB adds no wording"},
        "part_2_shared": {
            "rows": {"A": "C14c"},
            "what": lab["shared_what"], "label": lab["shared"],
            "after_window": lab["shared_end"], "lofo_line": lab["lofo"],
            "draw": "per county, [lo, hi] around zero (E02's registered centre), monthly, "
                    "2026-12..2027-10; FV v2's bn_head_fy line and the LOFO line (pooled "
                    "inside / family-months per county) beside it"},
        "scale": "both parts are drawn on one common vertical scale (metres per pixel)",
        "separate": "the two parts are drawn and labelled separately and never combined into "
                    "one band",
        "before_build": "the page is unchanged",
        "build_refused": "the page keeps its current band and wording; the refusal is logged",
        "never": list(K.NEVER),
        "station_ids": "none on the page, in the payload or in the output"}


def frozen_code() -> dict:
    return {f: (io.sha256(io.repo_path(f)) if io.repo_path(f).is_file() else "MISSING")
            for f in K.FROZEN}


def draft_prereg(prereg_dir=None, *, code_sha: dict | None = None) -> dict:
    """PB.decomposed_band.v1 (declaration). ``code_sha`` is for tests only: any injection turns
    the id into TEST.*."""
    injected = code_sha is not None
    fc = code_sha if injected else frozen_code()
    files = B.bound_files()
    regs = dict(K.UPSTREAM_REGS)
    repo_inputs = [{"path": f"{K.PREREG}/{i}.json", "sha256": s} for i, s in
                   {**regs, K.FB_ID: K.FB_SHA, K.U1X_ID: K.U1X_SHA}.items()]
    repo_inputs += [{"path": K.RULINGS, "sha256": K.RULINGS_SHA},
                    {"path": K.FB_RESULT, "sha256": K.FB_RESULT_SHA},
                    {"path": K.U1X_RESULT, "sha256": K.U1X_RESULT_SHA}]
    repo_inputs += [{"path": p, "sha256": s} for p, s in sorted(files.items())
                    if not p.startswith(K.TWIN_PREFIX)]
    repo_inputs += [{"path": p} for p in sorted(set(K.FROZEN) | {K.DRAFT, K.TESTS})]
    pid = ("TEST." if injected else "") + K.PB_ID
    return {
        "id": pid, "wp": K.WP, "kind": "declaration",
        "title": "PB: the decomposed page band (ruling R8.13): a per-well zone ribbon from "
                 "E02d's frozen sigma_idio_A/B/C.csv and a shared range per county with its "
                 "leave-one-family-out coverage (design-time, model-only); identity check "
                 "1e-9 m; CPU only",
        "why": "R8.13 replaces the drawn aggregate run spread with a band that shows what can "
               "be decided: the drawn per-well part (window A) is decided by a registered "
               "test (E02d, 2028); the shared part cannot be tested before 2030 by design. "
               "Declared before any member, family, range or LOFO number is computed (plan "
               "#16: 'PB declaration before build'; see disclosures.engineering_print).",
        "plan_entry": PLAN_ENTRY,
        "ruling": {"id": K.RULING_ID, "file": K.RULINGS, "sha256": K.RULINGS_SHA,
                   "decision": K.RULING_DECISION},
        "design": B.design_spec(),
        "binds": {"registrations": regs, "files": files,
                  "files_note": "results/twin/* are read from the builder's --twin-dir and "
                                "verified by sha256 (E02d interval.artefact.files and summary "
                                "inputs; E02y inputs); the rest are also hashed by "
                                "write_prereg; fv_zr0.npz's sha256 was recorded only by the "
                                "FB builder (disclosures.zr0)"},
        "page_mapping": page_mapping(),
        "status_16": {"FB": K.FB_STATUS, "EXS": K.EXS_STATUS},
        "disclosures": DISCLOSURES,
        "builder_reads": BUILDER_READS,
        "smoke_disclosure": "build_pb's functions ran on SYNTHETIC arrays only "
                            "(tests/test_pb_band.py); on real files only the per-zone median "
                            "half-widths pooled over wells and months were printed once "
                            "(disclosures.engineering_print); no member, family, range, LOFO "
                            "or coverage number was computed",
        "frozen_code": fc,
        "run": {"command": f"python -I {K.BUILDER} build --twin-dir <results/twin>",
                "output": f"{K.OUT_DIR}/{K.OUT_JSON} (written once)",
                "attempts": K.ATTEMPTS, "runs_once": True,
                "cost": "CPU, minutes (four truth forwards and E02's forward read at 158 "
                        "wells); 0 GPU-h"},
        "attempt_policy": "engineering failures (identity check, input sha256, forward "
                          "metadata) write nothing, are logged and may be retried; the band is "
                          "written once; no exploratory follow-up; a change to frozen code "
                          "needs a new id",
        "zones": NA("no fit; the ribbon's zones are E02d's frozen DR groups (design.ribbon)"),
        "priors": NA("no fit"), "burn_in": NA("no fit"), "folds": NA("no fit"),
        "tests": NA("declaration: nothing is scored; the LOFO coverage is a model-only "
                    "design-time number with no gate"),
        "fresh_units": NA("no observation is involved"),
        "thresholds": {"identity_tol_m": K.TOL_M, "min_zone_wells": K.MIN_ZONE_WELLS,
                       "z90": K.Z90, "first_coverage_month": K.FIRST_COVERAGE_MONTH},
        "gate_scores": NA("no promoting gate; the identity check is an engineering gate"),
        "primary_model": {"name": "datum_gate_innov (frozen forward), E02's registered centre; "
                                  "E02d's frozen relative artefact",
                          "forward_sha256": "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249"
                                            "c657dd6b39007228"},
        "inputs": repo_inputs,
        "interpretation": "A page-band construction, not a test: the ribbon displays E02d's "
                          "frozen per-well half-widths by zone with the registered tests that "
                          "decide them; the shared range displays model-only recipe and run "
                          "disagreement about the county-wide departure, with its "
                          "leave-one-family-out coverage; neither is calibrated or a "
                          "predictive interval.",
        "written_by": K.DRAFT}


def _registered_sha(doc: dict, path: str) -> str | None:
    for r in doc.get("inputs", []):
        if isinstance(r, dict) and str(r.get("path", "")).endswith(path):
            return r.get("sha256")
    return None


def problems(doc: dict, prereg_dir=None) -> list[str]:
    """Readiness problems before the lead registers (reads registrations and hashes only)."""
    bad = []
    if doc["id"].startswith("TEST."):
        bad.append("a TEST draft (injected code hashes) is never registrable")
    if "MISSING" in (doc.get("frozen_code") or {}).values():
        bad.append("frozen_code has a MISSING file")
    if io.index_record(K.PB_ID, prereg_dir) is not None:
        bad.append(f"{K.PB_ID} is already registered")
    docs = {}
    for rid, want in K.UPSTREAM_REGS.items():
        try:
            docs[rid], got = io.registered(rid, prereg_dir)
        except io.Refused as exc:
            bad.append(str(exc))
            continue
        if got != want:
            bad.append(f"{rid}: registered sha256 {got} != pinned {want}")
    for rid, want in ((K.FB_ID, K.FB_SHA), (K.U1X_ID, K.U1X_SHA)):
        rec = io.index_record(rid, prereg_dir)
        if rec is None or rec["sha256"] != want:
            bad.append(f"{rid}: not registered at {want[:8]}")
    if io.index_record(K.E02X_ID, prereg_dir) is not None:
        bad.append(f"{K.E02X_ID} is registered: EXS's status line is stale")
    e02d, e02y = docs.get(K.E02D_ID), docs.get(K.E02Y_ID)
    if e02d is not None:
        art = e02d["interval"]["artefact"]
        for w in K.WINDOWS:
            if art["files"][f"sigma_idio_{w}.csv"]["sha256"] != K.SIGMA_SHA[w]:
                bad.append(f"sigma_idio_{w}.csv: pinned sha differs from E02d's artefact")
        if art["summary"]["sha256"] != K.SUMMARY_SHA:
            bad.append("summary.json: pinned sha differs from E02d's artefact")
        if _registered_sha(e02d, "prospective_2026-11/predictions.csv") != K.CENTRE_SHA["A"]:
            bad.append("window A centre: pinned sha differs from E02d's inputs")
        for p, s in ((K.WELLS_E02D, K.WELLS_E02D_SHA), (K.MEMBERS_NPZ, K.MEMBERS_SHA)):
            if _registered_sha(e02d, p) != s:
                bad.append(f"{p}: pinned sha differs from E02d's inputs")
    if e02y is not None:
        for w in ("B", "C"):
            if _registered_sha(e02y, K.SIGMA[w]) != K.SIGMA_SHA[w]:
                bad.append(f"sigma_idio_{w}.csv: pinned sha differs from E02y's inputs")
            if _registered_sha(e02y, K.CENTRE[w]) != K.CENTRE_SHA[w]:
                bad.append(f"window {w} centre: pinned sha differs from E02y's inputs")
    fv2 = docs.get(K.FV2_ID)
    if fv2 is not None:
        reg = fv2.get("readings", {}).get("reading_2", {}).get("wording", {}).get("c14c", {})
        for lang, clause in (("en", K.FV2_C14C_CLAUSE_EN), ("zh", K.FV2_C14C_CLAUSE_ZH)):
            if clause not in str(reg.get(lang, "")):
                bad.append(f"FV v2's registered C14c wording ({lang}) does not hold the kept "
                           f"clause verbatim")
        tags = [t["tag"] for t in fv2.get("truths", [])]
        if sorted(tags) != sorted(K.FAMILIES):
            bad.append(f"FV v2 truths {tags} != {list(K.FAMILIES)}")
        files = B.family_files()
        for t in fv2.get("truths", []):
            p = f"results/twin_forward/fv_truth_{t['tag']}.npz"
            if files.get(p) != t["v1_forward"]["sha256"]:
                bad.append(f"{p}: FB-recorded sha differs from FV v2's v1_forward")
    rp = io.repo_path(K.RULINGS)
    if not rp.is_file() or io.sha256(rp) != K.RULINGS_SHA:
        bad.append(f"{K.RULINGS} missing or not at {K.RULINGS_SHA[:8]}")
    else:
        rul = {r["id"]: r for r in json.loads(rp.read_text(encoding="utf-8"))["rulings"]}
        if rul.get(K.RULING_ID, {}).get("decision") != K.RULING_DECISION:
            bad.append(f"{K.RULING_ID}'s decision differs from pb_const.RULING_DECISION")
    B.assert_no_observation_inputs([r["path"] for r in doc["inputs"]]
                                   + list(doc["binds"]["files"]))
    return bad


def _report(doc: dict, bad: list[str], twin_dir=None) -> None:
    errs = prereg3.validate_prereg(doc)
    print(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True, default=str))
    print(f"draft canonical sha256 {io.sha_json(doc)}")
    print(f"validate_prereg: {errs or 'OK'}")
    if twin_dir is not None:
        now = B.file_shas(doc["binds"]["files"], twin_dir)
        bad = bad + B.sha_problems(now, doc["binds"]["files"])
    print(f"readiness problems: {bad or 'none'}")
    print("nothing written (the lead registers with prereg3.write_prereg after merge)")


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true", required=True)
    ap.add_argument("--twin-dir", type=Path, default=None,
                    help="results/twin holding e02d_relative/ and prospective_*/ (hash only)")
    a = ap.parse_args(argv)
    try:
        doc = draft_prereg(prereg_dir)
        _report(doc, problems(doc, prereg_dir), a.twin_dir)
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
