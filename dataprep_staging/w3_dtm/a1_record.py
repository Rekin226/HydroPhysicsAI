"""Bring W3's pre-registration and its looks under the WP A1 tool (harness.prereg).

W3 wrote ``preregistration/W3.dtm_heldout_bias.json`` at 2026-10-12T02:41:17Z with its own
``prereg.py``, a minute before ``harness/prereg.py`` existed, so the hash-chained
``INDEX.jsonl`` does not hold it and ``looks.jsonl`` holds none of W3's looks at the leveling
benchmarks. A1 later (06:21:49Z) declared ``fresh.folds.leveling_CH`` and
``fresh.folds.leveling_all`` as fresh validation units; every benchmark of both lies among
the 1030 benchmarks W3 had already scored. This script records, without back-dating anything
the tool does not allow:

* a kind "declaration" ``W3.dtm_heldout_bias.record`` that puts the legacy file's sha256 into
  the index (the W2W4 precedent ``W4.rain_kriging_vs_idw.v1.record``);
* the two registered scoring looks (purpose "score", logged now, real times in the note):
  leveling elevations 2012-2021 (primary and pre-registered secondaries) and held-out leveling
  displacements 2011-2021 (the pre-registered epoch-correction error);
* the three unregistered looks, back-dated to their evidence files' modification times with
  the tool's ``seeded_time`` (purpose "design_time_seed", as S2, S3 and P1 did for their
  unregistered looks): the leave-one-block-out coverage of sd_epoch (v3 build), the
  after-the-fact diagnostics, and the continuation round's siting ranks.

The looks name W3's own unit and A1's two fresh leveling units, so ``assert_can_score``
discloses them and an unregistered look blocks any later test with the same data scope.

Usage: python -I a1_record.py --dry-run SCRATCH     (copies INDEX/looks/ledger into SCRATCH)
       python -I a1_record.py                       (writes the real records once)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, LEDGER_ROWS, PREREG_DIR, REPORTS_DIR, rel_to_repo  # noqa: E402
from harness import prereg as P  # noqa: E402

WP = "W3"
NA = P.not_applicable
LEGACY = PREREG_DIR / "W3.dtm_heldout_bias.json"
RECORD_ID = "W3.dtm_heldout_bias.record"
PROD = DATAPREP / "ground_elevation"
V3 = PROD / "v3"
EV = REPORTS_DIR / "W3_evidence"
CONT = EV / "continuation"
UNITS_REG = DATAPREP / "validation_units" / "v1" / "units.json"
A1_DECL = PREREG_DIR / "A1-fresh-units-v1.json"

ELEV = "leveling_elevation:2012-01..2021-12"
DISP = "leveling_displacement:2011-01..2021-12"
W3_UNIT = "W3.leveling_benchmarks_active_cells_1030"
A1_UNITS = ["fresh.folds.leveling_CH", "fresh.folds.leveling_all"]


def mtime(p: Path) -> str:
    """UTC modification time in the tool's look format (string-comparable with its times)."""
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def overlap() -> dict:
    res = pd.read_parquet(V3 / "benchmark_residuals_heldout.parquet", columns=["sid"])
    w3 = set(res["sid"].unique())
    reg = json.loads(UNITS_REG.read_text(encoding="utf-8"))
    out = {"w3_benchmarks": len(w3)}
    for uid in A1_UNITS:
        u = next(x for x in reg["units"] if x["id"] == uid)
        bms = {b for f in u["folds"] for b in f["test"]}
        out[uid] = {"role_in_registry": u["role"], "benchmarks": len(bms),
                    "scored_by_w3": len(bms & w3), "not_scored_by_w3": len(bms - w3)}
    return out


def times() -> dict:
    pre = json.loads(LEGACY.read_text(encoding="utf-8"))
    a1 = json.loads(A1_DECL.read_text(encoding="utf-8"))
    return {"prereg_created": pre["created"],
            "first_scored_output": mtime(PROD / "v1" / "ground_elevation_grid.npz"),
            "v1_manifest": mtime(PROD / "v1" / "MANIFEST.json"),
            "v2_manifest": mtime(PROD / "v2" / "MANIFEST.json"),
            "v3_manifest": mtime(V3 / "MANIFEST.json"),
            "diagnostics_v1": mtime(EV / "diagnostics_v1.json"),
            "diagnostics_v3": mtime(EV / "diagnostics_v3.json"),
            "verify_v3": mtime(CONT / "verify_v3.json"),
            "a1_fresh_units_declared": a1["created"]}


def declaration(T: dict, ov: dict) -> dict:
    val = json.loads((V3 / "validation.json").read_text(encoding="utf-8"))
    p = val["primary"]
    return {
        "id": RECORD_ID,
        "wp": WP,
        "kind": "declaration",
        "title": "Record of the WP-written W3 pre-registration (held-out DTM bias against "
                 "leveling benchmarks) and of the looks already made on its benchmarks",
        "records": {"id": "W3.dtm_heldout_bias", "path": rel_to_repo(LEGACY),
                    "sha256": sha256_file(LEGACY), "created_utc": T["prereg_created"],
                    "first_scored_utc": T["first_scored_output"],
                    "builds_utc": {"v1": T["v1_manifest"], "v2": T["v2_manifest"],
                                   "v3": T["v3_manifest"]},
                    "outcome": f"gate failed: median bias {p['median_m']:+.3f} m (threshold "
                               f"|bias| <= 0.30 m), {p['n_benchmarks']} benchmarks, "
                               f"{p['n_blocks']} blocks; product ground_elevation/v3"},
        "tests": NA("declaration: the test already ran under W3.dtm_heldout_bias"),
        "zones": NA("all active cells; strata as registered in W3.dtm_heldout_bias"),
        "priors": NA("no prior"),
        "burn_in": NA("no model run"),
        "folds": "10 km square blocks anchored at the grid origin, leave-one-block-out "
                 "(31 blocks); secondary variant with a 10 km buffer (as registered)",
        "fresh_units": NA(
            f"declaration; the {ov['w3_benchmarks']} benchmarks ({W3_UNIT}), which include "
            f"every benchmark of A1's {A1_UNITS[0]} ({ov[A1_UNITS[0]]['scored_by_w3']}/"
            f"{ov[A1_UNITS[0]]['benchmarks']}) and {A1_UNITS[1]} "
            f"({ov[A1_UNITS[1]]['scored_by_w3']}/{ov[A1_UNITS[1]]['benchmarks']}), are "
            f"looked-at units for {ELEV} and {DISP} (see looks.jsonl)"),
        "thresholds": "from the record: |median bias| <= 0.30 m of the epoch-corrected 1 km "
                      "cell mean against held-out benchmark elevations (gate)",
        "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": [rel_to_repo(LEGACY), rel_to_repo(V3 / "MANIFEST.json"),
                   rel_to_repo(V3 / "validation.json"),
                   rel_to_repo(V3 / "benchmark_residuals_heldout.parquet"),
                   rel_to_repo(PROD / "SUPERSEDED.json"),
                   rel_to_repo(EV / "diagnostics_v1.json"),
                   rel_to_repo(EV / "diagnostics_v3.json"),
                   rel_to_repo(CONT / "verify_v3.json"),
                   rel_to_repo(UNITS_REG), rel_to_repo(A1_DECL)],
        "units_defined": {
            W3_UNIT: "the leveling benchmarks inside an active 1 km cell with main-campaign "
                     "(Apr-Jul) observations 2012-2021 that W3 validated: column sid of "
                     "ground_elevation/v3/benchmark_residuals_heldout.parquet (hashed input)"},
        "a1_fresh_units_overlap": ov,
        "timeline_utc": T,
        "interpretation": (
            "Puts the legacy pre-registration's sha256 into the hash-chained index and logs "
            "W3's looks. All of them predate A1's fresh-unit declaration "
            f"({T['a1_fresh_units_declared']}), which stated that no look existed on the fresh "
            "units because W3's looks were not in looks.jsonl. A later test that scores leveling "
            "elevations 2012-2021 or leveling displacements 2011-2021 at these benchmarks is "
            "not fresh with respect to W3's analyses; whether A1's fresh leveling folds stay "
            "fresh for other data scopes is A1's call. Nothing here changes W3's result."),
    }


def looks(T: dict) -> list[dict]:
    units = [W3_UNIT] + A1_UNITS
    reg = "W3.dtm_heldout_bias"
    late = (f"logged late (the A1 index did not hold the pre-registration): computed by the "
            f"v1 build (first scored output {T['first_scored_output']}), v2 "
            f"({T['v2_manifest']}) and v3 ({T['v3_manifest']}) under the WP-written "
            f"pre-registration {reg} ({T['prereg_created']}), and recomputed identically by "
            f"the continuation verify.py ({T['verify_v3']})")
    return [
        dict(units=units, data_scope=ELEV, purpose="score",
             test_id="W3.dtm_heldout_bias.primary", prereg_id=reg,
             note=late + "; primary and pre-registered secondary metrics on benchmark "
                         "elevations (cell mean, cell median, uncorrected, point DTM, SRTM, "
                         "buffer, per year, strata, sd_point coverage)"),
        dict(units=units, data_scope=DISP, purpose="score",
             test_id="W3.dtm_heldout_bias.epoch_error", prereg_id=reg,
             note=late + "; pre-registered secondary: epoch-correction error a from the "
                         "held-out benchmarks' own displacements and its in-sample coverage"),
        dict(units=units, data_scope=DISP, purpose="design_time_seed",
             seeded_time=T["v3_manifest"],
             note="not a pilot: unregistered W3 look, back-dated to the v3 manifest time. The "
                  "leave-one-block-out coverage of sd_epoch (held-out displacements) was added "
                  "after the pre-registration and computed in the v3 build (reported only)"),
        dict(units=units, data_scope=ELEV, purpose="design_time_seed",
             seeded_time=T["diagnostics_v1"],
             note="not a pilot: unregistered W3 look, back-dated to diagnostics_v1.json. "
                  f"After-the-fact diagnostics.py runs (diagnostics_v1 {T['diagnostics_v1']}, "
                  f"diagnostics_v3 {T['diagnostics_v3']}): decomposition, local 3x3 window, "
                  "site type, DTM provenance, height-datum label and agency metadata against "
                  "benchmark elevations"),
        dict(units=units, data_scope=ELEV, purpose="design_time_seed",
             seeded_time=T["verify_v3"],
             note="not a pilot: unregistered W3 look, back-dated to verify_v3.json. The "
                  "continuation verify.py ranked each benchmark elevation and its DTM point "
                  "among the cell's 20 m points (siting ranks); report.py then read per-block "
                  "medians from the stored residuals"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", metavar="SCRATCH", default=None,
                    help="copy INDEX.jsonl, looks.jsonl and the W3 ledger fragment here and "
                         "write into the copies")
    a = ap.parse_args(argv)
    prereg_dir, rows_dir = PREREG_DIR, LEDGER_ROWS
    out = CONT / "a1_record.json"
    if a.dry_run:
        s = Path(a.dry_run)
        prereg_dir, rows_dir = s / "preregistration", s / "rows"
        prereg_dir.mkdir(parents=True, exist_ok=True)
        rows_dir.mkdir(parents=True, exist_ok=True)
        for f in ("INDEX.jsonl", "looks.jsonl"):
            shutil.copy2(PREREG_DIR / f, prereg_dir / f)
        shutil.copy2(LEDGER_ROWS / "W3.csv", rows_dir / "W3.csv")
        out = s / "a1_record.json"
    elif out.exists():
        raise SystemExit(f"{out} exists; the records were already written")
    have = [lk for lk in P.read_chained(Path(prereg_dir) / "looks.jsonl") if lk["wp"] == WP]
    if have:
        raise SystemExit(f"W3 looks already in looks.jsonl: {len(have)}")
    T, ov = times(), overlap()
    for uid in A1_UNITS:
        if ov[uid]["role_in_registry"] != "fresh":
            raise SystemExit(f"{uid} is not a fresh unit any more: {ov[uid]}")
    path, sha = P.write_prereg(declaration(T, ov), prereg_dir=prereg_dir, rows_dir=rows_dir)
    recs = []
    for lk in looks(T):
        lk = dict(lk)
        recs.append(P.record_look(lk.pop("units"), lk.pop("data_scope"), WP, lk.pop("purpose"),
                                  prereg_dir=prereg_dir, **lk))
    # the tool's own checks on the result
    idx = [r for r in P.read_chained(Path(prereg_dir) / "INDEX.jsonl") if r["id"] == RECORD_ID]
    doc, sha_loaded = P.load_prereg(RECORD_ID, prereg_dir=prereg_dir, rows_dir=rows_dir)
    probe_tests = [{"test_id": f"probe.{u}.{s.split(':')[0]}", "data_scope": s,
                    "fresh_units": [u]} for u in A1_UNITS for s in (ELEV, DISP)]
    block = P.blocking_looks(probe_tests, None, prereg_dir)
    rep = {"declaration": {"id": RECORD_ID, "path": rel_to_repo(path) if not a.dry_run
                           else str(path), "sha256": sha, "index_records": len(idx),
                           "load_prereg_ok": sha_loaded == sha},
           "looks": [{k: r[k] for k in ("time", "logged", "purpose", "test_id", "prereg_id",
                                        "data_scope", "units", "seeded")} for r in recs],
           "a1_fresh_units_overlap": ov, "timeline_utc": T,
           "blocking_probe": {"tests": [t["test_id"] for t in probe_tests],
                              "blocked_by_unregistered_w3_looks": sorted(
                                  {h["test_id"] for h in block
                                   if h["look"]["wp"] == WP})},
           "all_w3_looks_before_a1_declaration": all(
               T[k] < T["a1_fresh_units_declared"]
               for k in ("first_scored_output", "v1_manifest", "v2_manifest", "v3_manifest",
                         "diagnostics_v1", "diagnostics_v3", "verify_v3")),
           "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "dry_run": bool(a.dry_run)}
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("declaration", "blocking_probe",
                                          "all_w3_looks_before_a1_declaration")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
