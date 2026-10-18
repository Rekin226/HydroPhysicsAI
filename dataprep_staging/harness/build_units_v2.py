"""Build ``results/dataprep/validation_units/v2/`` (WP A1 repair round, 2026-10-12) and declare
its fresh units with the repaired tool before any downstream test uses them.

    CUDA_VISIBLE_DEVICES= tools/cpu_slot.sh python -m harness.build_units_v2 [--dry-run]

v1 stays published and unchanged (``build_units.py``, ``units.py`` and ``harness/__init__.py``
are hashed by it and are not edited). v2 starts from v1's files (inputs, hashed) and changes
roles and the design-time inventory only; every carried-over unit keeps its v1 id and
membership.

Rules fixed before the build ran (not changed after seeing its output):

1. **Leveling folds become design-time.** ``fresh.folds.leveling_CH`` / ``_all`` are renamed
   ``dt.folds.leveling_CH`` / ``_all`` (role ``design_time``) and their v1 ids are listed under
   ``withdrawn``: W3 scored every benchmark of both units for its DTM design iterations
   between 2026-10-12T02:41Z and 06:18Z, before the v1 declaration (06:21:49Z; looks logged at
   10:28Z), and every leveling benchmark had already been a hold-out unit of the site-grouped
   5-fold CV that selected the zonal compaction column (``calibrate_coupled.py --target
   leveling``, first results 2026-09-14). No leveling partition can be fresh, so v2 declares
   no fresh leveling unit.
2. **The pilots' remaining well hold-out schemes are seeded as design-time units** where
   their membership can be rebuilt from the pilot's own inputs: the Bayesian-fusion pilot's
   Changhua well folds (``KMeans(5, n_init=10, random_state=0)`` on the 74 Changhua wells'
   coordinates in km, p3_structural.py / p8_combined_variants.py) and head_pumping_signatures'
   5 km-block GroupKFold(5) and leave-one-block-out folds over Yunlin wells (cv_shallow.py:
   layers 1-2; cv_transfer.py: layers 1-4, 5 km blocks and townships). Schemes whose
   membership cannot be rebuilt here are listed in ``design_time_schemes`` with their evidence.
3. **A fresh unit stays fresh only if** (a) no look in ``looks.jsonl`` names it (any scope)
   and (b) for fold units, none of its test folds is a near-duplicate (Jaccard >= 0.9, the v1
   rule for cell units) of a design-time fold's test set. A unit failing either is relabelled
   design-time. The overlaps are reported in ``validation.json``.

Fresh still means "never used as a hold-out unit for a design choice", not unseen data: every
calibration well, cell and month was inside the pilots' 2012-22 fits, and every well was in a
design-time hold-out fold of some other partition (``validation.json`` quantifies it).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import manifest  # noqa: E402
from common.check import check_manifest  # noqa: E402
from common.paths import DATAPREP, PILOTS, PREREG_DIR, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

from harness import coverage2, prereg2, units  # noqa: E402

V1 = DATAPREP / "validation_units" / "v1"
V1_FILES = ["units.json", "wells.csv", "leveling_benchmarks.csv", "validation.json",
            "MANIFEST.json"]
PILOT_INPUTS = PILOTS / "physics_inversion_hooks" / "inputs_cwa.npz"
HPS = PILOTS / "head_pumping_signatures"
FUSION = PILOTS / "design_bayesian_fusion"
EVIDENCE = {
    "fusion_code": [FUSION / "p3_structural.py", FUSION / "p8_combined_variants.py"],
    "fusion_outputs": [FUSION / "p3_structural.json", FUSION / "p8_combined_variants.csv"],
    "hps_code": [HPS / "scripts" / "cv_shallow.py", HPS / "scripts" / "cv_transfer.py"],
    "hps_outputs": [HPS / "out" / "cv_shallow.csv", HPS / "out" / "cv_well_level.csv",
                    HPS / "out" / "cv_transfer.log"],
    "hps_table": [HPS / "out" / "well_summary.csv"],
    "leveling_cv": [REPO / "results/twin_runs/stage3_open_clean/coupled_leveling/"
                           "vep_shared_leveling.json",
                    REPO / "hydrophysics/twin/calibrate_coupled.py"],
    "w3_record": [REPORTS_DIR / "W3_evidence" / "continuation" / "a1_record.json"],
    "mltransfer_code": [PILOTS / "design_ml_transfer" / "a1_domain_shift.py",
                        PILOTS / "design_ml_transfer" / "a2_robust_transfer.py"],
}
NEAR_DUP = 0.9
RENAME = {"fresh.folds.leveling_CH": "dt.folds.leveling_CH",
          "fresh.folds.leveling_all": "dt.folds.leveling_all"}
WITHDRAW_REASON = (
    "W3 scored every benchmark of this unit (leveling elevations 2012-2021 and displacements "
    "2011-2021) for its DTM design iterations before the v1 declaration (W3 looks dated "
    "2026-10-12T02:43:08Z, 02:47:03Z, 06:18:45Z; logged 10:28:17Z), and every leveling "
    "benchmark was a hold-out unit of the site-grouped 5-fold CV that selected the zonal "
    "compaction column in 2026-09 (calibrate_coupled.py --target leveling)")


def _utc_mtime(p: Path) -> str:
    t = _dt.datetime.fromtimestamp(p.stat().st_mtime, _dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _earliest(paths) -> tuple[str, str]:
    best = min(paths, key=lambda p: p.stat().st_mtime)
    return _utc_mtime(best), rel_to_repo(best)


def fusion_folds(cal_sids: list[str]) -> tuple[list[dict], dict]:
    """The Bayesian-fusion pilot's Changhua well folds (p3_structural.py line 148,
    p8_combined_variants.py line 92), rebuilt from inputs_cwa.npz."""
    from sklearn.cluster import KMeans
    z = np.load(PILOT_INPUTS, allow_pickle=False)
    sids = [str(s) for s in z["sids"]]
    wch = z["well_county"].astype(str) == "彰化縣"
    km = KMeans(5, n_init=10, random_state=0).fit(z["well_xy"][wch] / 1e3)
    ch_sids = [s for s, w in zip(sids, wch, strict=True) if w]
    folds = []
    for f in range(5):
        te = [s for s, lab in zip(ch_sids, km.labels_, strict=True) if lab == f]
        folds.append({"fold": f, "test": te, "train": [s for s in ch_sids if s not in set(te)]})
    info = {"n_wells": len(ch_sids), "fold_sizes": [len(f["test"]) for f in folds],
            "sids_equal_calibration_order": sids == cal_sids}
    return folds, info


def hps_folds(cal_set: set) -> tuple[dict, dict]:
    """head_pumping_signatures' well-level spatial CV folds (cv_shallow.py, cv_transfer.py)."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    ws = pd.read_csv(HPS / "out" / "well_summary.csv", dtype={"sid": str, "layer": str},
                     usecols=["sid", "x", "y", "layer", "county", "town", "n_months",
                              "sectors3"])
    blk = lambda d: ((d.x // 5000).astype(int).astype(str) + "_"  # noqa: E731
                     + (d.y // 5000).astype(int).astype(str))
    w12 = ws[ws.layer.isin(["1", "2"]) & (ws.n_months >= 24)].copy()
    y12 = w12[(w12.county == "Yunlin") & (w12.sectors3 >= 7)].copy()
    y12["blk5"] = blk(y12)
    yw = ws[(ws.county == "Yunlin") & (ws.sectors3 >= 7) & (ws.n_months >= 24)].copy()
    yw = yw[yw.layer.isin(["1", "2", "3", "4"])].copy()
    yw["blk5"] = blk(yw)
    out, info = {}, {}

    def split(df, splitter, groups):
        X = np.zeros((len(df), 1))
        sids = df.sid.tolist()
        folds = []
        for i, (_tr, te) in enumerate(splitter.split(X, np.zeros(len(df)), groups)):
            tset = [sids[j] for j in te]
            folds.append({"fold": i, "test": tset,
                          "train": [s for s in sids if s not in set(tset)]})
        return folds

    specs = {
        "dt.folds.hps_blk5_gkf5_L12": (y12, GroupKFold(5), "blk5",
                                       "cv_shallow.py blk5_5fold (layers 1-2)"),
        "dt.folds.hps_blk5_logo_L12": (y12, LeaveOneGroupOut(), "blk5",
                                       "cv_shallow.py blk5_LOGO (layers 1-2)"),
        "dt.folds.hps_blk5_gkf5_L1to4": (yw, GroupKFold(min(5, yw.blk5.nunique())), "blk5",
                                         "cv_transfer.py well-level CV, groups blk5"),
        "dt.folds.hps_town_gkf5_L1to4": (yw, GroupKFold(min(5, yw.town.nunique())), "town",
                                         "cv_transfer.py well-level CV, groups town"),
    }
    for uid, (df, sp, g, src) in specs.items():
        folds = split(df, sp, df[g].to_numpy())
        out[uid] = {"folds": folds, "source": src}
        info[uid] = {"n_rows": len(df), "n_folds": len(folds),
                     "n_calibration_wells": int(sum(s in cal_set for s in df.sid))}
    info["rows_L12"] = len(y12)
    info["rows_L1to4"] = len(yw)
    return out, info


def _test_sets(u: dict) -> list[set]:
    return [set(f["test"]) for f in u["folds"]]


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv
    t0 = time.time()
    v1chk = check_manifest(V1 / "MANIFEST.json")
    reg1 = json.loads((V1 / "units.json").read_text(encoding="utf-8"))
    wells = pd.read_csv(V1 / "wells.csv", dtype={"sid": str})
    cal_sids = wells.sid.tolist()
    v1_units = {u["id"]: u for u in reg1["units"]}

    # ---- carry v1 units; rename the leveling folds (rule 1)
    U = []
    for u in reg1["units"]:
        u2 = json.loads(json.dumps(u))
        if u["id"] in RENAME:
            u2.update({"id": RENAME[u["id"]], "role": "design_time", "group": "dt.folds",
                       "renamed_from": u["id"], "relabel_reason": WITHDRAW_REASON})
        U.append(u2)

    # ---- pilot well hold-out schemes (rule 2)
    ff, finfo = fusion_folds(cal_sids)
    U.append({"id": "dt.folds.fusion_km5_CH", "role": "design_time", "kind": "folds",
              "group": "dt.folds",
              "definition": "Bayesian-fusion pilot's spatial-block CV over the 74 Changhua "
                            "calibration wells: KMeans(5, n_init=10, random_state=0) on well "
                            "coordinates in km (p3_structural.py, p8_combined_variants.py), "
                            "rebuilt from _pilots/physics_inversion_hooks/inputs_cwa.npz",
              "folds": ff})
    hf, hinfo = hps_folds(set(cal_sids))
    for uid, rec in hf.items():
        U.append({"id": uid, "role": "design_time", "kind": "folds", "group": "dt.folds",
                  "definition": f"head_pumping_signatures well-level spatial CV ({rec['source']})"
                                "; rebuilt from the pilot's out/well_summary.csv with its "
                                "selection (Yunlin, sectors3 >= 7, n_months >= 24); the "
                                "pilot's own well set (not only calibration wells); a "
                                "historical label of a design-time hold-out, never a fitted "
                                "input",
                  "folds": rec["folds"]})
    by = {u["id"]: u for u in U}

    # ---- rule 3: looks and near-duplicates
    looks = prereg2.read_chained(PREREG_DIR / "looks.jsonl")
    looked = {}
    for lk in looks:
        for x in lk["units"]:
            looked.setdefault(x, []).append(lk["time"])
    dt_folds = [u for u in U if u["role"] == "design_time" and u["kind"] == "folds"]
    fold_overlap, relabel = {}, {}
    for u in U:
        if u["role"] != "fresh":
            continue
        if u["id"] in looked:
            relabel[u["id"]] = f"looks in looks.jsonl at {sorted(looked[u['id']])}"
        if u["kind"] != "folds":
            continue
        rows = []
        fresh_sids = set().union(*_test_sets(u))
        for f, ts in enumerate(_test_sets(u)):
            best = max(((_jac(ts, d), dtu["id"], j) for dtu in dt_folds
                        for j, d in enumerate(_test_sets(dtu))), default=(0.0, None, None))
            rows.append({"fold": f, "n_test": len(ts), "max_jaccard": best[0],
                         "with": best[1], "with_fold": best[2]})
        ari = {}
        from sklearn.metrics import adjusted_rand_score
        lab_f = {s: i for i, ts in enumerate(_test_sets(u)) for s in ts}
        for dtu in dt_folds:
            lab_d = {s: i for i, ts in enumerate(_test_sets(dtu)) for s in ts}
            common = sorted(set(lab_f) & set(lab_d))
            if len(common) >= 5:
                ari[dtu["id"]] = {"n_common": len(common), "ari": float(adjusted_rand_score(
                    [lab_f[s] for s in common], [lab_d[s] for s in common]))}
        in_dt_test = sorted(s for s in fresh_sids
                            if any(s in ts for dtu in dt_folds for ts in _test_sets(dtu)))
        mj = max(r["max_jaccard"] for r in rows)
        fold_overlap[u["id"]] = {"folds": rows, "max_jaccard": mj, "ari": ari,
                                 "n_wells": len(fresh_sids),
                                 "n_wells_in_a_design_time_test_fold": len(in_dt_test)}
        u["max_jaccard_design_time_folds"] = {"value": mj, "with": max(
            rows, key=lambda r: r["max_jaccard"])["with"]}
        if mj >= NEAR_DUP:
            relabel[u["id"]] = f"test fold near-duplicate of a design-time fold (Jaccard {mj})"
    for uid, why in relabel.items():
        by[uid].update({"role": "design_time", "relabel_reason": why})

    # ---- validation checks
    checks = []

    def check(name, expected, measured, ok):
        checks.append({"check": name, "expected": expected, "measured": measured,
                       "pass": bool(ok)})

    check("v1 product re-hashes (common.check)", "ok", v1chk["status"], v1chk["status"] == "ok")
    same = 0
    for u in U:
        src = v1_units.get(u.get("renamed_from", u["id"]))
        if src is None:
            continue
        keys = ("cells", "wells", "months", "folds", "kind", "definition")
        same += all(src.get(k) == u.get(k) for k in keys)
    check("carried-over units keep their v1 membership and definition", len(v1_units), same,
          same == len(v1_units))
    check("leveling folds renamed design-time, v1 ids withdrawn", sorted(RENAME.values()),
          sorted(u["id"] for u in U if u.get("renamed_from") in RENAME),
          all(by[v]["role"] == "design_time" for v in RENAME.values())
          and not set(RENAME) & set(by))
    fresh = sorted(u["id"] for u in U if u["role"] == "fresh")
    check("fresh units carry no look (any scope) at build time", 0,
          sum(len(looked.get(f, [])) for f in fresh), not any(f in looked for f in fresh))
    yl = set().union(*(set(v1_units[f"dt.tercile.{k}"]["cells"])
                       for k in ("west", "mid", "east")))
    for grp, ids in (("rotated terciles", [f"fresh.rot_tercile.{i}" for i in (1, 2, 3)]),
                     ("k-means 2027", [f"fresh.km2027.{i}" for i in range(5)])):
        sets = [set(by[i]["cells"]) for i in ids]
        disjoint = sum(len(s) for s in sets) == len(set().union(*sets))
        covers = set().union(*sets) == yl
        check(f"{grp}: disjoint partition of the Yunlin cells", True,
              {"disjoint": disjoint, "covers_yunlin": covers}, disjoint and covers)
    masks = {u["id"]: set(u["cells"]) for u in U if u["kind"] == "cells"}
    dtc = [k for k in masks if by[k]["role"] == "design_time"]
    mjc = max(max(_jac(masks[k], masks[d]) for d in dtc) for k in masks
              if by[k]["role"] == "fresh")
    check("no fresh cell unit is a near-duplicate of a design-time region (Jaccard < 0.9)",
          "< 0.9", mjc, mjc < NEAR_DUP)
    blk = set(by["fresh.block.2016_2018"]["months"]) & set(by["dt.block.2020_2022"]["months"])
    check("fresh temporal block disjoint from the 2020-22 hold-out", 0, len(blk), not blk)
    pos = wells.set_index("sid")[["x", "y"]]
    site = dict(zip(cal_sids, units.well_sites(wells[["x", "y"]].to_numpy(), 100.0),
                    strict=True))
    for name in ("wells_CH", "wells_YL", "wells_all"):
        u = by[f"fresh.folds.{name}"]
        ids = [s for f in u["folds"] for s in f["test"]]
        sites_whole = all(len({i for i, f in enumerate(u["folds"]) for s in f["test"]
                               if site[s] == st}) == 1 for st in {site[s] for s in ids})
        mind = []
        for f in u["folds"]:
            a, b = pos.loc[f["test"]].to_numpy(), pos.loc[f["train"]].to_numpy()
            mind.append(float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min() / 1000))
        ok = len(ids) == len(set(ids)) and sites_whole and min(mind) > 5.0
        check(f"fresh folds {name}: partition, sites whole, train-test distance > 5 km", True,
              {"n": len(ids), "sites_whole": sites_whole,
               "min_train_test_km": round(min(mind), 3)}, ok)
    check("fusion pilot Changhua folds rebuilt: 74 wells, 5 folds, inputs_cwa order equals the "
          "calibration wells", [74, 5, True],
          [finfo["n_wells"], len(ff), finfo["sids_equal_calibration_order"]],
          finfo["n_wells"] == 74 and len(ff) == 5 and finfo["sids_equal_calibration_order"])
    n12 = int(pd.read_csv(HPS / "out" / "cv_shallow.csv")["n"].iloc[0])
    n14 = int(pd.read_csv(HPS / "out" / "cv_well_level.csv")["n"].iloc[0])
    check("head_pumping_signatures folds rebuilt on the pilot's rows (cv_shallow n, "
          "cv_well_level n)", [n12, n14], [hinfo["rows_L12"], hinfo["rows_L1to4"]],
          [hinfo["rows_L12"], hinfo["rows_L1to4"]] == [n12, n14])
    mjf = max(v["max_jaccard"] for v in fold_overlap.values())
    check("no fresh well fold is a near-duplicate of a design-time fold (Jaccard < 0.9)",
          "< 0.9", mjf, mjf < NEAR_DUP)
    passed = all(c["pass"] for c in checks)

    seeds = {
        "leveling": (*_earliest(EVIDENCE["leveling_cv"][:1]),
                     sorted(RENAME.values()) + sorted(RENAME)),
        "fusion": (*_earliest(EVIDENCE["fusion_outputs"]), ["dt.folds.fusion_km5_CH"]),
        "hps": (*_earliest(EVIDENCE["hps_outputs"]), sorted(hf)),
    }
    schemes = [
        {"id": "dt.scheme.coupled_leveling_kfold5_2026-09", "kind": "leveling benchmarks",
         "definition": "calibrate_coupled.py --target leveling: site-grouped 5-fold CV over the "
                       "~800 leveling benchmarks on the grid (default_rng(0) permutation, "
                       "array_split into 5), out-of-fold R2 used to select the zonal "
                       "compaction column (memory: +0.546 zonal vs +0.371 shared)",
         "evidence": [rel_to_repo(p) for p in EVIDENCE["leveling_cv"]],
         "first_evidence_utc": seeds["leveling"][0],
         "membership": "not rebuilt (needs the twin hindcast inputs); covers every leveling "
                       "benchmark of dt.folds.leveling_*"},
        {"id": "dt.scheme.w3_leveling_block10_lobo", "kind": "leveling benchmarks",
         "definition": "W3 DTM held-out bias: leave-one-10 km-block-out over the 1030 "
                       "benchmarks in active cells (design iterations v1-v3, 2026-10-12)",
         "evidence": [rel_to_repo(p) for p in EVIDENCE["w3_record"]],
         "first_evidence_utc": "2026-10-12T02:41:44.898385Z",
         "membership": "W3's product; all 781 benchmarks of dt.folds.leveling_all scored"},
        {"id": "dt.scheme.mltransfer_block5_gkf5", "kind": "cells",
         "definition": "design_ml_transfer a1/a2: GroupKFold(5) over 5 km blocks of in-grid "
                       "cells of both counties (adversarial propensity and transfer scores)",
         "evidence": [rel_to_repo(p) for p in EVIDENCE["mltransfer_code"]],
         "membership": "not rebuilt (needs the pilot's feature table); every in-grid cell was "
                       "in one of its test folds, so the fresh cell units are fresh only as "
                       "partitions"},
    ]
    if dry:
        for c in checks:
            print(("PASS " if c["pass"] else "FAIL ") + c["check"], "->", c["measured"])
        print("relabel", relabel)
        print("fold overlaps", json.dumps({k: {kk: v[kk] for kk in ("max_jaccard", "n_wells",
                                                                     "n_wells_in_a_design_time"
                                                                     "_test_fold")}
                                           for k, v in fold_overlap.items()}))
        print("seeds", seeds)
        print(f"dry run: nothing written; gate would be passed={passed}")
        return 0 if passed else 1
    if not passed:
        print("gate failed; nothing written")
        for c in checks:
            print(("PASS " if c["pass"] else "FAIL ") + c["check"], "->", c["measured"])
        return 1

    # ---- write the product
    import sklearn
    d = manifest.new_version_dir("validation_units")
    withdrawn = {old: {"replaced_by": new, "reason": WITHDRAW_REASON}
                 for old, new in RENAME.items()}
    registry = {
        "schema": "dataprep-validation-units/2", "created_by": "WP A1 (repair round)",
        "supersedes": "validation_units/v1", "sklearn_version": sklearn.__version__,
        "grid": reg1["grid"], "origin": reg1["origin"], "membership_rule": reg1["membership_rule"],
        "roles": {"fresh": "never used as a hold-out unit for a design choice; declared in "
                           "A1-fresh-units-v2 (harness.prereg2) before any downstream test. "
                           "Fresh only as partitions: wells, cells and months were inside the "
                           "pilots' 2012-22 fits and the wells were in design-time hold-out "
                           "folds of other partitions (validation.json fold_overlaps)",
                  "design_time": "used by the pilots, the twin's 2026-09 calibration or another "
                                 "WP for design choices; results on them are design-time "
                                 "evidence only"},
        "withdrawn": withdrawn, "design_time_schemes": schemes, "units": U}
    (d / "units.json").write_text(json.dumps(registry, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
    for f in ("wells.csv", "leveling_benchmarks.csv"):
        shutil.copyfile(V1 / f, d / f)
    val = {"checks": checks, "passed": passed, "relabelled": relabel,
           "fold_overlaps": fold_overlap, "pilot_fold_rebuilds": {"fusion": finfo, **hinfo},
           "fresh_units": fresh, "design_time_look_seeds": {
               k: {"time": v[0], "evidence": v[1], "units": v[2]} for k, v in seeds.items()},
           "overlaps_fresh_cells_vs_design_time": json.loads(
               (V1 / "validation.json").read_text(encoding="utf-8"))
           ["overlaps_fresh_vs_design_time"],
           "seconds": round(time.time() - t0, 1)}
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1) + "\n",
                                       encoding="utf-8")
    here = Path(__file__).resolve().parent
    inputs = [V1 / f for f in V1_FILES] + [PILOT_INPUTS] + \
        [p for k in ("fusion_code", "fusion_outputs", "hps_code", "hps_outputs", "hps_table",
                     "leveling_cv", "w3_record", "mltransfer_code") for p in EVIDENCE[k]]
    m = manifest.build_manifest(
        d, product="validation_units", layer="L2", inputs=inputs,
        code=[here / "build_units_v2.py", here / "units.py"],
        parameters={"source_version": "v1", "near_duplicate_jaccard": NEAR_DUP,
                    "renamed": RENAME, "sklearn": sklearn.__version__,
                    "pilot_fold_labels_note": "dt.folds.hps_* membership follows the pilot's "
                                              "well selection (its full record); the labels "
                                              "only record past hold-outs and never enter a "
                                              "fresh unit, a fit or a statistic"},
        fit_window_end="2022-12", validation_report="validation.json",
        gate={"passed": passed, "criteria": "A1 repair round: v1 re-hashes and is carried over "
                                            "unchanged except the declared relabels; fresh "
                                            "units carry no look; partitions disjoint; folds "
                                            "keep sites whole and buffers; pilot folds rebuilt "
                                            "on the pilots' rows; no fresh unit near-duplicates "
                                            "a design-time unit"},
        allowed_uses=["selection", "validator", "after_the_fact_analysis",
                      "design_time_evidence"],
        forbidden_uses=["model_design", "prior", "prior_width_input", "calibration_target",
                        "datum_fit", "baseline", "initial_conditions", "idw_field",
                        "forcing_frozen", "scoring_target"],
        personal_data="none: public monitoring-station and benchmark ids and coordinates only")
    manifest.write_manifest(d, m)
    print(f"wrote {rel_to_repo(d)}; gate passed={passed}", flush=True)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"], "->", c["measured"])

    # ---- seed the design-time looks (idempotent by note), then declare the fresh units
    notes = {lk.get("note") for lk in looks}
    for k, (t, src, ids) in seeds.items():
        note = (f"design-time units of validation_units/{d.name} ({k}); seeded by WP A1 "
                f"(repair round), dated to the mtime of {src}")
        if note not in notes:
            prereg2.record_look(ids, "*", "A1", "design_time_seed", seeded_time=t, note=note)
    doc = {
        "id": f"A1-fresh-units-{d.name}", "wp": "A1", "kind": "declaration",
        "title": "Fresh validation units, folds and coverage gate rule (plan WP A1, repair "
                 "round)",
        "tests": prereg2.not_applicable("declaration of units; every downstream test that "
                                        "scores on them writes its own pre-registration "
                                        "with harness.prereg2"),
        "zones": {"design_time": ["dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3"],
                  "note": "Z1-Z3 are design-time choices (plan 4.5); the Yunlin mirror of the "
                          "Z2 rule is fresh.river_band_YL10"},
        "priors": prereg2.not_applicable("no model is fitted by A1"),
        "burn_in": prereg2.not_applicable("no model is fitted by A1"),
        "folds": {k: by[f"fresh.folds.{k}"]["params"] for k in ("wells_CH", "wells_YL",
                                                                 "wells_all")},
        "fresh_units": fresh,
        "withdrawn_units": withdrawn,
        "supersedes": "A1-fresh-units-v1 (its leveling folds were looked at before it; the "
                      "other 15 units keep their ids and membership)",
        "thresholds": {"coverage_gate": {
            "module": "harness.coverage2", "rule": coverage2.RULE,
            "target": coverage2.TARGET, "interval": "cluster-adjusted Wilson on n_eff_hat, "
                                                    "ANOVA ICC, Student-t on k-1 df",
            "level": coverage2.GATE_LEVEL, "point_window": list(coverage2.POINT_WINDOW),
            "point_window_if_n_eff_lower_at_least": coverage2.N_EFF_WINDOW_MIN,
            "n_eff_lower": f"one-sided {coverage2.WINDOW_CONF} bound from the F-based upper "
                           "bound of the ICC",
            "evidence": "A1_evidence/coverage_confirm.json (A1.coverage_confirm.v1, interval); "
                        "A1_evidence/repair/gate_window_confirm.json (A1r.gate_window.v1, "
                        "window rule)"}},
        "gate_scores": prereg2.not_applicable("set by each test's pre-registration"),
        "primary_model": prereg2.not_applicable("set by each test's pre-registration"),
        "inputs": [str(d / "units.json"), str(d / "MANIFEST.json"), str(d / "validation.json")],
        "interpretation": (
            "Fresh units (role 'fresh' in units.json) are validation units: no design choice "
            "may be made on them, and every test scored on them is pre-registered first with "
            "harness.prereg2, whose rule refuses a test that postdates any look at its data. "
            "The two-tercile region is fresh.two_tercile.ME (63 of 84 Yunlin wells); "
            "fresh.two_tercile.WM (55) is secondary. No leveling unit is fresh. The "
            "overlapping terciles, halves and k-means sets give no independent estimate of "
            "tau; report the effective n. Design-time units are labelled as such wherever they "
            "appear and never validate the choices made on them."),
    }
    path, sha = prereg2.write_prereg(doc, units_registry=d / "units.json")
    print(f"declared {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
