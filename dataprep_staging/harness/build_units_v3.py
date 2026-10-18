"""Build ``results/dataprep/validation_units/v3/`` (WP A1 second repair round, 2026-10-12).

    CUDA_VISIBLE_DEVICES= python -m harness.build_units_v3

v3 changes documentation only. Its data files (``units.json``, ``wells.csv``,
``leveling_benchmarks.csv``, ``validation.json``) are byte-identical copies of v2's, so every
unit keeps its id, role and membership, and the A1-fresh-units-v2 declaration still names
exactly the fresh units of v3. The manifest repairs the A1 verifier's minor finding on v2:
v2 declares ``fit_window_end`` 2022-12 and no exemption although some of its inputs are pilot
outputs built on records that reach past the origin (head_pumping_signatures
``well_summary.csv``; the design_bayesian_fusion and design_ml_transfer scripts; the 2026-09
leveling CV record). Every input record of v3 now carries ``role``, ``data_end`` (the pilot's
fit window from ``_pilots/INDEX.json`` where it applies) and a ``note`` saying what the input
contributes: only membership of design-time hold-out labels and the fresh-vs-design-time
near-duplicate check (which can only relabel a fresh unit to design-time, never create one).
No unit membership, fold or statistic of a fresh unit depends on a post-origin value, so
``fit_window_end`` stays 2022-12 and ``exemptions`` stays empty; the ``post_origin_inputs``
block lists those inputs explicitly instead.

v2 stays published and unchanged. ``harness.prereg3`` reads every registry version; v3 adds
nothing it does not already know.
"""
from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, manifest  # noqa: E402
from common.check import check_manifest  # noqa: E402
from common.paths import DATAPREP, PILOTS, from_repo, rel_to_repo  # noqa: E402

V2 = DATAPREP / "validation_units" / "v2"
DATA_FILES = ["units.json", "wells.csv", "leveling_benchmarks.csv", "validation.json"]
USES = ("validator", "selection", "after_the_fact_analysis", "design_time_evidence",
        "model_design", "prior", "prior_width_input", "calibration_target", "datum_fit",
        "baseline", "scoring_target", "initial_conditions")
EXPECT_OPEN = {"validator", "selection", "after_the_fact_analysis", "design_time_evidence"}
LABEL = ("design-time label only: records which wells or benchmarks a pilot or the 2026-09 "
         "calibration held out; enters the design-time inventory and the fresh-vs-design-time "
         "near-duplicate check (which can only relabel a fresh unit to design-time); never a "
         "member of a fresh unit, a fit or a statistic")


def _pilot_windows() -> dict:
    idx = json.loads((PILOTS / "INDEX.json").read_text(encoding="utf-8"))
    return {p["scout"]: p.get("fit_window_end") for p in idx["pilots"]}


def _role(rec: dict, windows: dict) -> dict:
    p = rec["path"]
    out = {"path": p}
    if p.startswith("results/dataprep/validation_units/v2/"):
        out.update({"role": "carried_over", "data_end": "2022-12",
                    "note": "v2 file; data files of v3 are byte-identical copies"})
        return out
    if p.startswith("results/dataprep/validation_units/v1/"):
        out.update({"role": "provenance", "data_end": "2022-12",
                    "note": "v1 file read by the v2 build (input of v2)"})
        return out
    if p.startswith("results/dataprep/_pilots/"):
        scout = p.split("/")[3]
        de = windows.get(scout)
        if scout == "physics_inversion_hooks":
            note = ("coordinates of the 74 Changhua calibration wells used to rebuild "
                    "dt.folds.fusion_km5_CH (pre-origin pilot input); " + LABEL)
        else:
            note = f"pilot {scout} (fit window {de}); " + LABEL
        out.update({"role": "design_time_label", "data_end": de, "note": note})
        return out
    if p.startswith("results/twin_runs/") or p.startswith("hydrophysics/"):
        out.update({"role": "design_time_label", "data_end": "2026-09",
                    "note": "evidence of the 2026-09 site-grouped leveling CV "
                            "(dt.scheme.coupled_leveling_kfold5_2026-09); " + LABEL})
        return out
    if p.startswith("results/dataprep/reports/W3_evidence/"):
        out.update({"role": "design_time_label", "data_end": "2026-10",
                    "note": "W3's record of its leveling looks (why the v1 leveling folds were "
                            "withdrawn); " + LABEL})
        return out
    out.update({"role": "unknown", "data_end": None, "note": "unclassified input"})
    return out


def main(argv=None) -> int:
    v2m, _ = manifest.load_manifest(V2 / "MANIFEST.json")
    chk2 = check_manifest(V2 / "MANIFEST.json")
    windows = _pilot_windows()
    d = manifest.new_version_dir("validation_units")
    for f in DATA_FILES:
        shutil.copyfile(V2 / f, d / f)
    identical = {f: filecmp.cmp(V2 / f, d / f, shallow=False) for f in DATA_FILES}
    v2_inputs = [_role(r, windows) for r in v2m["inputs"]]
    inputs = [{"path": rel_to_repo(V2 / f)} for f in DATA_FILES + ["MANIFEST.json"]]
    inputs = [_role(r, windows) for r in inputs] + v2_inputs
    post = [r for r in inputs if (r.get("data_end") or "0000-00") > "2022-12"]
    checks = [
        {"check": "v2 re-hashes (common.check)", "measured": chk2.get("status"),
         "pass": chk2.get("status") == "ok"},
        {"check": "data files byte-identical to v2", "measured": identical,
         "pass": all(identical.values())},
        {"check": "every input classified (role and data_end)",
         "measured": [r["path"] for r in inputs if r["role"] == "unknown"],
         "pass": not any(r["role"] == "unknown" for r in inputs)},
        {"check": "every post-origin input is a design-time label",
         "measured": sorted({r["role"] for r in post}),
         "pass": all(r["role"] == "design_time_label" for r in post)},
    ]
    val = {"product": "validation_units", "version": d.name, "supersedes": "v2",
           "change": "documentation only (input roles); data files byte-identical to v2",
           "checks": checks, "post_origin_inputs": post,
           "verifier_finding": "A1 verifier (second round), minor: v2 declares fit_window_end "
                               "2022-12 and no exemptions although pilot outputs built on "
                               "full records enter the near-duplicate gate as design-time "
                               "labels; the manifest should say so"}
    (d / "v3_validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")
    here = Path(__file__).resolve().parent
    passed = all(c["pass"] for c in checks)
    m = manifest.build_manifest(
        d, product="validation_units", layer="L2", inputs=inputs,
        code=[here / "build_units_v3.py"],
        parameters={"source_version": "v2", "data_files": DATA_FILES,
                    "change": "documentation only: input roles, data_end and notes; "
                              "post_origin_inputs listed",
                    "pilot_fold_labels_note": v2m["parameters"].get("pilot_fold_labels_note"),
                    "near_duplicate_jaccard": v2m["parameters"].get("near_duplicate_jaccard"),
                    "renamed": v2m["parameters"].get("renamed"),
                    "sklearn": v2m["parameters"].get("sklearn")},
        fit_window_end="2022-12", validation_report="v3_validation.json",
        gate={"passed": passed,
              "criteria": "A1 second repair round: v2 re-hashes; data files byte-identical to "
                          "v2; every input classified; every post-origin input is a "
                          "design-time label"},
        allowed_uses=v2m["allowed_uses"], forbidden_uses=v2m["forbidden_uses"],
        personal_data=v2m["personal_data"],
        extra={"post_origin_inputs": {
            "inputs": [r["path"] for r in post],
            "role": "design_time_label",
            "why_fit_window_end_stays_2022-12": (
                "no unit membership, fold or statistic of a fresh unit depends on a value "
                "after 2022-12: fresh units are built from 2012-2022 calibration wells and "
                "cells (build_head_field t0 2012-01, t1 2023-01) and leveling 2012-2022; the "
                "post-origin inputs only define design-time hold-out labels and can only "
                "relabel a fresh unit to design-time"),
            "uses_affected": "none of the allowed uses reads a post-origin value"}})
    manifest.write_manifest(d, m)
    chk3 = check_manifest(d / "MANIFEST.json")
    g = {}
    for use in USES:
        try:
            guard.open_product(d, use)
            g[use] = True
        except Exception:  # noqa: BLE001 - every refusal is recorded
            g[use] = False
    res = {"version_dir": rel_to_repo(d), "gate_passed": passed, "checks": checks,
           "checker": {k: chk3.get(k) for k in ("status", "n_hashed", "schema_errors",
                                                 "hash_problems")},
           "guard": g, "guard_as_expected": all(g[u] == (u in EXPECT_OPEN) for u in USES),
           "n_post_origin_inputs": len(post)}
    out = from_repo("results/dataprep/reports/A1_evidence/repair2/build_units_v3.json")
    with open(out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:3000])
    return 0 if passed and chk3.get("status") == "ok" and res["guard_as_expected"] else 1


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
