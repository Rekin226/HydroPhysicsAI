"""WP A1 second repair round evidence: the verifier's end-to-end probes of the pre-registration
tool on the real units registry, an audit of the real look log under ``harness.prereg3``, the
membership overlaps among the fresh units, and the integrity checks.

    python -m harness.a1r2_checks OUT_DIR       (from dataprep_staging/)

Writes ``OUT_DIR/a1r2_checks.json`` and ``OUT_DIR/prereg3_audit.json``; nothing else (the
probes run in temporary directories against a copy of the latest ``units.json``; the real look
log, INDEX and ledger are only read).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import manifest  # noqa: E402
from common.check import check_manifest  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR  # noqa: E402

from harness import a1r_checks, prereg, prereg2, prereg3  # noqa: E402

A1_PREREGS = ("A1-fresh-units-v1", "A1.coverage_confirm.v1", "A1r.gate_window.v1",
              "A1-fresh-units-v2", "A1r2.coverage3_confirm.v1")


def _attempt(fn) -> str:
    try:
        fn()
        return "allowed"
    except (prereg.PreregistrationError, ValueError) as exc:
        return f"refused: {type(exc).__name__}: {str(exc)[:200]}"


def _doc(pid, tid, scope, units):
    na = prereg2.not_applicable
    return {"id": pid, "wp": "PRB", "title": "probe", "kind": "test",
            "tests": [{"test_id": tid, "data_scope": scope, "fresh_units": list(units)}],
            "zones": na("p"), "priors": na("p"), "burn_in": na("p"), "folds": na("p"),
            "fresh_units": list(units), "thresholds": {"x": 1},
            "gate_scores": [{"name": "x", "weight": 1.0}], "primary_model": "p",
            "inputs": na("p"), "interpretation": "probe"}


def _scenario(tool, root: Path, reg: Path, steps: list) -> dict:
    """Run one probe scenario with ``tool`` in a fresh directory; ``steps`` are tuples:
    ("look", units, scope) | ("score", pid, tid, scope, units) | ("register", pid, tid,
    scope, units) | ("register_and_score", pid, tid, scope, units)."""
    root.mkdir(parents=True)
    kw = {"prereg_dir": root / "p", "rows_dir": root / "r"}
    kw3 = dict(kw, registries=[reg]) if tool is prereg3 else kw
    out = {}
    for i, s in enumerate(steps):
        key = f"{i}:{s[0]}:{s[1] if s[0] == 'look' else s[1]}"
        if s[0] == "look":
            out[key] = _attempt(lambda s=s: tool.record_look(list(s[1]), s[2], "PRB",
                                                             "diagnose", **kw3))
        elif s[0] == "score":           # a registered and scored earlier design
            _, pid, tid, scope, units = s
            out[key] = _attempt(lambda pid=pid, tid=tid, scope=scope, units=units: (
                tool.write_prereg(_doc(pid, tid, scope, units), units_registry=reg, **kw3),
                tool.assert_can_score(pid, "PRB", **kw3)))
        elif s[0] == "register":
            _, pid, tid, scope, units = s
            out[key] = _attempt(lambda pid=pid, tid=tid, scope=scope, units=units:
                                tool.write_prereg(_doc(pid, tid, scope, units),
                                                  units_registry=reg, **kw3))
        elif s[0] == "register_and_score":
            _, pid, tid, scope, units = s
            r = {}
            r["register"] = _attempt(lambda pid=pid, tid=tid, scope=scope, units=units:
                                     tool.write_prereg(_doc(pid, tid, scope, units),
                                                       units_registry=reg, **kw3))
            disclosed = []

            def score(pid=pid, disclosed=disclosed):
                res = tool.assert_can_score(pid, "PRB", **kw3)
                disclosed.extend(res["post_registration_looks"])
            r["score"] = _attempt(score) if r["register"] == "allowed" else "not registered"
            r["post_registration_looks_disclosed"] = len(disclosed)
            out[key] = r
    return out


def probes(reg: Path) -> dict:
    """The verifier's second-round probes plus nesting and negative controls, on both tools."""
    sc = {
        "v1_day_dates": [("look", ["fresh.rot_tercile.1"], "heads:2016-01-01..2018-12-31"),
                         ("register_and_score", "P", "P.t", "heads:2017-01..2018-12",
                          ["fresh.rot_tercile.1"])],
        "v2_multi_window": [("look", ["fresh.rot_tercile.1"],
                             "heads:2012-01..2015-12,2019-01..2022-12"),
                            ("register_and_score", "P", "P.t", "heads:2020-01..2022-12",
                             ["fresh.rot_tercile.1"])],
        "v3_km2027_0_then_rot_tercile_1": [
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.km2027.0"]),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.rot_tercile.1"])],
        "nest_rot_tercile_1_then_two_tercile_WM": [
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.rot_tercile.1"]),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.two_tercile.WM"])],
        "nest_km2027_2_then_two_tercile_ME": [
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.km2027.2"]),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.two_tercile.ME"])],
        "nest_rot_tercile_2_then_folds_wells_YL": [
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.rot_tercile.2"]),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.folds.wells_YL"])],
        "temporal_rot_tercile_3_2012_22_then_block_2016_18": [
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.rot_tercile.3"]),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.block.2016_2018"])],
        "control_disjoint_km2027_0_then_km2027_4": [
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.km2027.0"]),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.km2027.4"])],
        "control_disjoint_window_then_block_2016_18": [
            ("look", ["fresh.rot_tercile.1"], "heads:2012-01..2015-12,2019-01..2022-12"),
            ("register_and_score", "P", "P.t", "heads:2012-01..2022-12",
             ["fresh.block.2016_2018"])],
        "control_registered_before_any_look": [
            ("register", "P", "P.t", "heads:2012-01..2022-12", ["fresh.rot_tercile.1"]),
            ("score", "A", "A.t", "heads:2012-01..2022-12", ["fresh.two_tercile.WM"]),
            ("register_and_score", "Q", "Q.t", "heads:2023-01..2026-09",
             ["fresh.rot_tercile.1"])],
    }
    expect_refused = {"v1_day_dates", "v2_multi_window", "v3_km2027_0_then_rot_tercile_1",
                      "nest_rot_tercile_1_then_two_tercile_WM",
                      "nest_km2027_2_then_two_tercile_ME",
                      "nest_rot_tercile_2_then_folds_wells_YL",
                      "temporal_rot_tercile_3_2012_22_then_block_2016_18"}
    res = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, steps in sc.items():
            res[name] = {}
            for tname, tool in (("prereg2", prereg2), ("prereg3", prereg3)):
                res[name][tname] = _scenario(tool, td / name / tname, reg, steps)
    summ = {}
    for name, r in res.items():
        last = {t: list(r[t].values())[-1] for t in r}
        ok = {t: (last[t]["register"] == "allowed" and last[t]["score"] == "allowed")
              for t in last}
        summ[name] = {"expected": "refused" if name in expect_refused else "allowed",
                      "prereg2": "allowed" if ok["prereg2"] else "refused",
                      "prereg3": "allowed" if ok["prereg3"] else "refused",
                      "prereg2_disclosed": last["prereg2"]["post_registration_looks_disclosed"],
                      "prereg3_disclosed": last["prereg3"]["post_registration_looks_disclosed"]}
        summ[name]["prereg3_as_expected"] = summ[name]["prereg3"] == summ[name]["expected"]
    return {"scenarios": res, "summary": summ,
            "prereg3_all_as_expected": all(s["prereg3_as_expected"] for s in summ.values()),
            "prereg2_allows_refusable": sorted(n for n, s in summ.items()
                                               if s["expected"] == "refused"
                                               and s["prereg2"] == "allowed")}


def builder_probes() -> dict:
    """The first repair round's five probes (a1r_checks) re-run on prereg3."""
    na = prereg.not_applicable
    out = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        reg = td / "units.json"
        reg.write_text(json.dumps({"units": [{"id": "fresh.u1", "role": "fresh",
                                              "kind": "cells", "cells": [1], "wells": ["w"]}]}))

        def doc(pid, tid, scope):
            d = _doc(pid, tid, scope, ["fresh.u1"])
            d["burn_in"] = na("p")
            return d
        kw = {"prereg_dir": td / "a" / "p", "rows_dir": td / "a" / "r", "registries": [reg]}
        prereg3.write_prereg(doc("A", "A.t", "heads:2012-01..2022-12"), units_registry=reg,
                             **kw)
        prereg3.assert_can_score("A", "PRB", **kw)
        out["a_register_revised_design_B_after_A_scored"] = _attempt(
            lambda: prereg3.write_prereg(doc("B", "B.t", "heads:2012-01..2022-12"),
                                         units_registry=reg, **kw))
        prereg.write_prereg(doc("B", "B.t", "heads:2012-01..2022-12"), units_registry=reg,
                            prereg_dir=kw["prereg_dir"], rows_dir=kw["rows_dir"])
        out["a_score_B"] = _attempt(lambda: prereg3.assert_can_score("B", "PRB", **kw))
        kw2 = {"prereg_dir": td / "b" / "p", "rows_dir": td / "b" / "r", "registries": [reg]}
        prereg3.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB", "diagnose", **kw2)
        out["b_test_2016_18_after_look_2012_22"] = _attempt(
            lambda: prereg3.write_prereg(doc("C", "C.t", "heads:2016-01..2018-12"),
                                         units_registry=reg, **kw2))
        kw3 = {"prereg_dir": td / "c" / "p", "rows_dir": td / "c" / "r", "registries": [reg]}
        out["c_look_under_nonexistent_prereg_id"] = _attempt(
            lambda: prereg3.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB",
                                        "diagnose", test_id="x", prereg_id="nope", **kw3))
        prereg.record_look(["fresh.u1"], "heads:2012-01..2022-12", "PRB", "score",
                           test_id="x", prereg_id="nope", prereg_dir=kw3["prereg_dir"])
        out["c_test_after_look_under_nonexistent_id"] = _attempt(
            lambda: prereg3.write_prereg(doc("D", "D.t", "heads:2012-01..2022-12"),
                                         units_registry=reg, **kw3))
    out["prereg3_refuses_all"] = all(v.startswith("refused") for v in out.values())
    return out


def fresh_overlaps(idx: prereg3.MembershipIndex) -> dict:
    fresh = sorted(u for u, r in idx.role.items() if r == "fresh" and u not in idx.withdrawn)
    pairs = []
    for i, a in enumerate(fresh):
        for b in fresh[i + 1:]:
            if idx.overlap(a, b):
                ma, mb = idx.members[a], idx.members[b]
                pairs.append({"a": a, "b": b, "shared_ids": len(ma["ids"] & mb["ids"]),
                              "shared_cells": len(ma["cells"] & mb["cells"]),
                              "a_in_b_ids": bool(ma["ids"]) and ma["ids"] <= mb["ids"],
                              "b_in_a_ids": bool(mb["ids"]) and mb["ids"] <= ma["ids"]})
    disjoint = [(a, b) for i, a in enumerate(fresh) for b in fresh[i + 1:]
                if not idx.overlap(a, b)]
    return {"n_fresh": len(fresh), "n_pairs": len(fresh) * (len(fresh) - 1) // 2,
            "n_overlapping_pairs": len(pairs), "overlapping_pairs": pairs,
            "disjoint_pairs": disjoint}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_real = prereg2.latest_units_registry()
    out: dict = {"units_registry": str(reg_real)}
    with tempfile.TemporaryDirectory() as td:
        reg = Path(td) / "units.json"
        shutil.copyfile(reg_real, reg)
        shutil.copyfile(reg_real.parent / "wells.csv", Path(td) / "wells.csv")
        out["probes_verifier"] = probes(reg)
    out["probes_builder_round1_on_prereg3"] = builder_probes()
    idx = prereg3.membership_index()
    out["fresh_overlaps"] = fresh_overlaps(idx)
    aud = prereg3.audit()
    with open(out_dir / "prereg3_audit.json", "x", encoding="utf-8") as f:
        json.dump(aud, f, indent=1, ensure_ascii=False, default=str)
    out["audit_summary"] = {
        "index_records": aud["index_records"], "looks_records": aud["looks_records"],
        "tests_refused_under_prereg3": [(r["prereg_id"], r["test_id"])
                                        for r in aud["tests_refused_under_prereg3"]],
        "tests_refused_under_prereg3_not_prereg2": [
            (r["prereg_id"], r["test_id"])
            for r in aud["tests_refused_under_prereg3_not_prereg2"]],
        "scopes_read_differently_from_prereg2": aud["scopes_read_differently_from_prereg2"],
        "fresh_units_with_eligible_blocking_looks": {
            u: v["n_eligible_to_block"] for u, v in aud["fresh_units"].items()
            if v["n_eligible_to_block"]},
        "units_of_unknown_membership": len(aud["looks_on_units_of_unknown_membership"])}
    out["preregistrations"] = {}
    for pid in A1_PREREGS:
        doc, sha = prereg3.load_prereg(pid)
        out["preregistrations"][pid] = {"sha256": sha, "created": doc["created"],
                                        "tool": doc.get("tool", "harness.prereg")}
    out["chains"] = {"INDEX.jsonl": len(prereg.read_chained(PREREG_DIR / "INDEX.jsonl")),
                     "looks.jsonl": len(prereg.read_chained(PREREG_DIR / "looks.jsonl")),
                     "intact": True}
    out["checker"] = {}
    for v in sorted(p.name for p in (DATAPREP / "validation_units").iterdir() if p.is_dir()):
        c = check_manifest(DATAPREP / "validation_units" / v / manifest.MANIFEST_NAME)
        out["checker"][v] = {k: c.get(k) for k in ("status", "n_hashed")}
    hashed = a1r_checks._hashed_harness_files()
    out["harness_files_hashed_elsewhere"] = {"n": len(hashed),
                                             "changed": [h for h in hashed if not h["unchanged"]]}
    with open(out_dir / "a1r2_checks.json", "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: out[k] for k in ("audit_summary", "chains", "checker")}, indent=1,
                     default=str)[:4000])
    print(json.dumps(out["probes_verifier"]["summary"], indent=1))
    print("prereg3 as expected:", out["probes_verifier"]["prereg3_all_as_expected"],
          "| builder probes refused:",
          out["probes_builder_round1_on_prereg3"]["prereg3_refuses_all"],
          "| hashed harness files changed:", len(out["harness_files_hashed_elsewhere"]["changed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
