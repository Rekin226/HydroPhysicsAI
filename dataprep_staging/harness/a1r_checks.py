"""WP A1 repair round evidence: checks of validation_units v2, the repaired tool, and an audit
of the existing look log under the repaired rules.

    python -m harness.a1r_checks OUT_DIR       (from dataprep_staging/, in a CPU slot)

Writes ``OUT_DIR/a1r_checks.json``, ``OUT_DIR/prereg_audit.json`` and
``OUT_DIR/prereg_probes.json``; nothing else (the probes use a temporary directory). Checks:

* the loader guard opens ``validation_units/v2`` for its allowed uses (now including
  ``design_time_evidence``) and refuses the forbidden ones;
* ``validation_units/v1`` and ``v2`` re-hash (``common.check``);
* the four A1 pre-registrations load with their recorded sha256; INDEX.jsonl and looks.jsonl
  chains are intact;
* every v2 fresh unit carries no look; every v2 design-time unit carries a seeded look; the
  withdrawn v1 leveling ids and a design-time unit are refused as fresh by ``prereg2``;
* every harness file hashed by another product or pre-registration is unchanged (the repair
  added modules instead of editing them);
* the verifier's three probes on the first tool, re-run on both tools in a temporary
  directory: the first tool allows them, ``prereg2`` refuses them.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, manifest  # noqa: E402
from common.check import check_manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, from_repo, rel_to_repo  # noqa: E402

from harness import prereg, prereg2  # noqa: E402

V1 = DATAPREP / "validation_units" / "v1"
V2 = DATAPREP / "validation_units" / "v2"
USES = ("validator", "selection", "after_the_fact_analysis", "design_time_evidence",
        "model_design", "prior", "prior_width_input", "calibration_target", "datum_fit",
        "baseline", "scoring_target", "initial_conditions")
EXPECT_OPEN = {"validator", "selection", "after_the_fact_analysis", "design_time_evidence"}
A1_PREREGS = ("A1-fresh-units-v1", "A1.coverage_confirm.v1", "A1r.gate_window.v1",
              "A1-fresh-units-v2")


def _hashed_harness_files() -> list[dict]:
    out = []
    srcs = list(DATAPREP.rglob("MANIFEST.json")) + list(PREREG_DIR.glob("*.json"))
    for p in srcs:
        if "_pilots" in p.parts:
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recs = [r for k in ("code", "inputs") for r in (doc.get(k) or [])
                if isinstance(r, dict) and "dataprep_staging/harness/" in str(r.get("path"))
                and "sha256" in r]
        for r in recs:
            f = from_repo(r["path"])
            out.append({"by": rel_to_repo(p), "file": r["path"],
                        "unchanged": f.exists() and sha256_file(f) == r["sha256"]})
    return out


def _probes() -> dict:
    """The verifier's probes (a), (b), (c) on both tools, in a temporary directory."""
    na = prereg.not_applicable
    res = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        reg = td / "units.json"
        reg.write_text(json.dumps({"units": [{"id": "fresh.u1", "role": "fresh"}]}))

        def doc(pid, tid, scope):
            return {"id": pid, "wp": "PRB", "title": "probe", "kind": "test",
                    "tests": [{"test_id": tid, "data_scope": scope,
                               "fresh_units": ["fresh.u1"]}],
                    "zones": na("p"), "priors": na("p"), "burn_in": na("p"), "folds": na("p"),
                    "fresh_units": ["fresh.u1"], "thresholds": {"x": 1},
                    "gate_scores": [{"name": "x", "weight": 1.0}], "primary_model": "p",
                    "inputs": na("p"), "interpretation": "probe"}

        def attempt(fn):
            try:
                fn()
                return "allowed"
            except (prereg.PreregistrationError, ValueError) as exc:
                return f"refused: {type(exc).__name__}: {str(exc)[:160]}"

        for tool_name, tool in (("prereg (first tool)", prereg), ("prereg2", prereg2)):
            d = td / tool_name.split()[0]
            kw = {"prereg_dir": d / "p", "rows_dir": d / "r"}
            r = {}
            tool.write_prereg(doc("A", "A.t", "heads:2012-01..2022-12"), units_registry=reg,
                              **kw)
            tool.assert_can_score("A", "PRB", **kw)
            r["a_register_revised_design_B_after_A_scored"] = attempt(
                lambda: tool.write_prereg(doc("B", "B.t", "heads:2012-01..2022-12"),  # noqa: B023
                                          units_registry=reg, **kw))  # noqa: B023
            if not (kw["prereg_dir"] / "B.json").exists():   # as a first-tool user would
                prereg.write_prereg(doc("B", "B.t", "heads:2012-01..2022-12"),
                                    units_registry=reg, **kw)
            r["a_score_B"] = attempt(lambda: tool.assert_can_score("B", "PRB", **kw))  # noqa: B023
            d2 = td / (tool_name.split()[0] + "_b")
            kw2 = {"prereg_dir": d2 / "p", "rows_dir": d2 / "r"}
            if tool is prereg:
                prereg.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB", "score",
                                   prereg_dir=kw2["prereg_dir"])
            else:
                prereg2.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB", "diagnose",
                                    **kw2)
            r["b_test_2016_18_after_unregistered_look_2012_22"] = attempt(
                lambda: tool.write_prereg(doc("C", "C.t", "heads:2016-01..2018-12"),  # noqa: B023
                                          units_registry=reg, **kw2))  # noqa: B023
            d3 = td / (tool_name.split()[0] + "_c")
            kw3 = {"prereg_dir": d3 / "p", "rows_dir": d3 / "r"}
            if tool is prereg:
                r["c_look_under_nonexistent_prereg_id"] = attempt(
                    lambda: prereg.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB",
                                               "score", test_id="x", prereg_id="nope",
                                               prereg_dir=kw3["prereg_dir"]))  # noqa: B023
            else:
                r["c_look_under_nonexistent_prereg_id"] = attempt(
                    lambda: prereg2.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB",
                                                "diagnose", test_id="x", prereg_id="nope",
                                                **kw3))  # noqa: B023
            # a look under a nonexistent id (written with the first tool) then a new test
            prereg.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB", "score",
                               test_id="x", prereg_id="nope", prereg_dir=kw3["prereg_dir"])
            r["c_test_after_look_under_nonexistent_id"] = attempt(
                lambda: tool.write_prereg(doc("D", "D.t", "heads:2012-01..2022-12"),  # noqa: B023
                                          units_registry=reg, **kw3))  # noqa: B023
            res[tool_name] = r
    old, new = res["prereg (first tool)"], res["prereg2"]
    res["first_tool_allows_all"] = all(v == "allowed" for v in old.values())
    res["prereg2_refuses_all"] = all(v.startswith("refused") for v in new.values())
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"product": rel_to_repo(V2), "guard": {}}
    for use in USES:
        try:
            p = guard.open_product(V2, use)
            out["guard"][use] = {"opened": True, "notes": p.notes}
        except Exception as e:  # noqa: BLE001 - every refusal is recorded
            out["guard"][use] = {"opened": False, "error": f"{type(e).__name__}: {e}"[:200]}
    out["guard_as_expected"] = all(out["guard"][u]["opened"] == (u in EXPECT_OPEN)
                                   for u in USES)
    out["checker"] = {}
    for d in (V1, V2):
        c = check_manifest(d / manifest.MANIFEST_NAME)
        out["checker"][d.name] = {k: c.get(k) for k in ("status", "n_hashed", "schema_errors",
                                                         "hash_problems")}
    out["preregistrations"] = {}
    for pid in A1_PREREGS:
        doc, sha = prereg2.load_prereg(pid)
        out["preregistrations"][pid] = {"sha256": sha, "created": doc["created"],
                                        "kind": doc["kind"], "tool": doc.get("tool",
                                                                             "harness.prereg")}
    idx = prereg.read_chained(PREREG_DIR / "INDEX.jsonl")
    looks = prereg.read_chained(PREREG_DIR / "looks.jsonl")
    out["chains"] = {"INDEX.jsonl_records": len(idx), "looks.jsonl_records": len(looks),
                     "intact": True}
    reg = prereg2.load_registry(V2 / "units.json")
    fresh = sorted(u for u, x in reg["units"].items() if x["role"] == "fresh")
    dt = sorted(u for u, x in reg["units"].items() if x["role"] == "design_time")
    seeded = {u for lk in looks if lk["purpose"] == "design_time_seed" for u in lk["units"]}
    named = {u for lk in looks for u in lk["units"]}
    dec = prereg2.load_prereg("A1-fresh-units-v2")[0]
    out["v2_units"] = {"n_fresh": len(fresh), "n_design_time": len(dt),
                       "fresh_with_any_look": [u for u in fresh if u in named],
                       "design_time_without_seed": [u for u in dt if u not in seeded],
                       "declaration_created": dec["created"],
                       "declaration_fresh_units_equal_registry": sorted(dec["fresh_units"])
                       == fresh, "withdrawn": sorted(reg["withdrawn"])}
    na = prereg2.not_applicable
    probe = {"id": "A1r.probe.never_written", "wp": "A1", "title": "probe", "kind": "test",
             "tests": [{"test_id": "A1r.probe.t", "data_scope": "leveling:2012-01..2022-12",
                        "fresh_units": ["fresh.folds.leveling_CH"],
                        "external_units": {"fresh.folds.leveling_CH": "probe"}},
                       {"test_id": "A1r.probe.t2", "data_scope": "heads:2012-01..2022-12",
                        "fresh_units": ["dt.tercile.west"]}],
             "zones": na("probe"), "priors": na("probe"), "burn_in": na("probe"),
             "folds": na("probe"), "fresh_units": ["dt.tercile.west"], "thresholds": {"x": 1},
             "gate_scores": [{"name": "x", "weight": 1.0}], "primary_model": "probe",
             "inputs": na("probe"), "interpretation": "probe (never written)"}
    errs = prereg2.validate_prereg(probe, V2 / "units.json")
    out["probe_withdrawn_and_design_time_as_fresh"] = {
        "validation_errors": errs,
        "withdrawn_refused": any("withdrawn" in e for e in errs),
        "design_time_refused": any("cannot be a fresh unit" in e for e in errs),
        "blocking_looks_on_withdrawn_leveling_ids": len(prereg2.blocking_looks(
            probe["tests"][:1], None))}
    hh = _hashed_harness_files()
    out["harness_files_hashed_elsewhere"] = {"n": len(hh),
                                             "changed": [x for x in hh if not x["unchanged"]],
                                             "records": hh}
    probes = _probes()
    (out_dir / "prereg_probes.json").write_text(json.dumps(probes, indent=1,
                                                           ensure_ascii=False))
    out["probes"] = {"first_tool_allows_all": probes["first_tool_allows_all"],
                     "prereg2_refuses_all": probes["prereg2_refuses_all"]}
    aud = prereg2.audit(registries=[V1 / "units.json", V2 / "units.json"])
    (out_dir / "prereg_audit.json").write_text(json.dumps(aud, indent=1, ensure_ascii=False))
    v1a = aud["registries"][rel_to_repo(V1 / "units.json")]
    v2a = aud["registries"][rel_to_repo(V2 / "units.json")]
    out["audit_summary"] = {
        "tests_refused_under_repaired_rule": [
            {k: x[k] for k in ("prereg_id", "test_id", "wp", "scored", "n_prior_looks")}
            for x in aud["tests_refused_under_repaired_rule"]],
        "tests_with_changed_inputs": [x["id"] for x in aud["tests_with_changed_inputs"]],
        "looks_with_unknown_prereg_id": len(aud["looks_with_unknown_prereg_id"]),
        "unknown_prereg_ids": sorted({x["prereg_id"] for x in
                                      aud["looks_with_unknown_prereg_id"]}),
        "prereg_files_not_in_index": aud["prereg_files_not_in_index"],
        "v1_fresh_units_with_looks": sorted(v1a["fresh_units_with_looks"]),
        "v2_fresh_units_with_looks": sorted(v2a["fresh_units_with_looks"]),
    }
    ok = (out["guard_as_expected"]
          and all(c["status"] == "ok" for c in out["checker"].values())
          and not out["v2_units"]["fresh_with_any_look"]
          and not out["v2_units"]["design_time_without_seed"]
          and out["v2_units"]["declaration_fresh_units_equal_registry"]
          and out["probe_withdrawn_and_design_time_as_fresh"]["withdrawn_refused"]
          and out["probe_withdrawn_and_design_time_as_fresh"]["design_time_refused"]
          and not out["harness_files_hashed_elsewhere"]["changed"]
          and out["probes"]["first_tool_allows_all"] and out["probes"]["prereg2_refuses_all"])
    out["pass"] = bool(ok)
    with open(out_dir / "a1r_checks.json", "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"guard_as_expected": out["guard_as_expected"],
                      "checker": {k: v["status"] for k, v in out["checker"].items()},
                      "v2_units": out["v2_units"], "probes": out["probes"],
                      "harness_files_changed": out["harness_files_hashed_elsewhere"]["changed"],
                      "audit": out["audit_summary"], "pass": out["pass"]},
                     indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
