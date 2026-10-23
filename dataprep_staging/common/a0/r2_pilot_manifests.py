"""WP A0 repair round: revision r2 of every pilot manifest (and of _pilots/INDEX.json).

What r2 changes against the first-round manifests (r1):

* ``fit_window_end`` is set per pilot from the data-window audit
  (``common/a0/r2_window_audit.py`` evidence plus the reading of each pilot's scripts recorded
  in :data:`AUDIT`), instead of the blanket 2026-10. Outputs that a pilot computed from
  pre-origin data only carry their own ``fit_window_end`` (:data:`PRE_ORIGIN_OUTPUTS`).
* ``design_time_evidence`` is a pre-origin use since the repair round, so it stays allowed
  only where it can pass the guard (the pilot or some of its outputs end by 2022-12); the
  other pilots allow ``after_the_fact_analysis`` only.
* ``personal_data`` is corrected where the r1 text missed pump coordinates
  (``landuse_crop_priors/target.npz``: pump_xy, pump_hp, pump_town, pump_cls, pump_cell).
* Inputs also record what the pilots read through repository modules (static scan of the
  quoted data paths in the imported ``hydrophysics`` modules, labelled ``via``; possible,
  not certain) and the directories they read as a whole (``input_trees``, tree hashes).

Versioning: the r1 file is kept byte-identical as ``MANIFEST.r1.json`` (listed and hashed as
an output of r2, with its sha256 in ``parameters.supersedes``) and r2 is written as
``MANIFEST.json`` (``write_manifest`` refuses to overwrite). Nothing else in the pilot
directories changes. ``INDEX.json`` -> ``INDEX.r1.json`` likewise; ``README.md`` ->
``README.r1.md`` and a new README.

Usage: python -m common.a0.r2_pilot_manifests --audit <window_audit.json> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hashing import sha256_file  # noqa: E402
from common.manifest import (  # noqa: E402
    MANIFEST_NAME,
    build_manifest,
    load_manifest,
    validate_manifest,
    verify_hashes,
    write_manifest,
)
from common.paths import ORIGIN, PILOTS, REPO, from_repo, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

VERSION = "2026-10-12.r2"
R1_NAME = "MANIFEST.r1.json"

#: Per-pilot fit_window_end and its basis (U3: inputs included; U7: observations).
AUDIT = {
    "audit_2026-10-11": ("2026-09", [
        "reads the 2023-26 test-network heads (data_fetch_api/new_measurements/"
        "heads_202210_202610, windows 2023-01..2026-09 in artefacts.py and heads_profile.py)",
        "outputs: rescore_rows.csv dates 2023-01..2026-08, artefact_events.csv 2012-01..2026-07"]),
    "billing_and_census_gaps": ("2025-07", [
        "TPC pump census meter matrix 2007-01..2025-07 (s01_build.py; meters_phase last_nz "
        "2025-07); billing-pair statistics 2012-01..2025-06 (s02); Chow-Lin window "
        "2011-01..2023-12 (s11); July-2025 and zero-run analyses use 2022-2025"]),
    "design_bayesian_fusion": ("2025-12", [
        "heads and twin inputs 2012-01..2022-12 (physics_inversion_hooks arrays), but its "
        "inputs include tpc_aggregate_electricity/changhua_pump_electricity_candidates.csv, "
        "whose coefficients were fitted on 2016-01..2025-06 (tools/ch_product.py), "
        "surface_water_monthly/monthly_deliveries_pilot2.csv (annual totals 2012-2025 used in "
        "every fold, pilot2.py) and the land-use prior (LUIMAP 2024 snapshot); U3 counts "
        "inputs, so the pilot is post-origin"]),
    "design_ml_transfer": ("2026-08", [
        "t1_head_to_electricity.py correlates heads with TPC county sales 2016-01..2026-08 and "
        "trains on the census to 2025-06, reading head_gapfill_pilot monthly_long (2012-2026)",
        "the other scripts read the physics_inversion_hooks arrays (2012-2022), the land-use "
        "prior (LUIMAP 2024) and the canal pilot (2012-2025); not audited per output"]),
    "design_physics_inversion": ("2025-12", [
        "Changhua heads and kernels 2012-01..2022-12 (physics_inversion_hooks), but inputs "
        "include the TPC candidate series (coefficients fitted 2016-01..2025-06), the canal "
        "pilot deliveries (2012-2025 annual totals) and the land-use prior (LUIMAP 2024)"]),
    "head_gapfill_pilot": ("2026-09", [
        "daily and monthly matrices 2012-01..2026-09 including the 2023-26 network heads "
        "(s1_build.py, lib.load_matrix('comb')); partial-month RMSE rho(k) "
        "(s4b_partial.py -> out/partial_month_error.csv) uses complete months 2012-2026; daily "
        "bridge 2013-2025; s8 test-period router check inside 2023-01..2026-09",
        "the artificial-gap pilot chain s3/s3b/s5/s5b/s5c runs on lib.load_matrix('hist') "
        "(2012-01..2022-12; daily QC MAD computed per historical file) with rain features on "
        "the same dates and the audit's 'hist' artefact events: those outputs carry "
        "fit_window_end 2022-12"]),
    "head_pumping_signatures": ("2026-09", [
        "well-month features 2012-01..2026-09 incl. the 2023-26 network heads "
        "(well_features2.py); kWh matrix 2007-01..2025-06 (build_kwh_matrix.py)"]),
    "landuse_crop_priors": ("2024-12", [
        "LUIMAP land-use snapshot (chou-shui-data/data/water/irrigation/luimap_z14.png; the "
        "plan's 'LUIMAP 2024 snapshot', U3's listed exemption; month unknown, 2024-12 used as "
        "the upper bound); its sensitivity run belongs to WP P3",
        "everything else is pre-origin: census targets 2012-01..2022-12 (build_target.py, "
        "town_kwh.py), heads 2012-2022 (heads_amp.py), crop statistics fitted on 2014, 2015, "
        "2016 and 2021 (township_crops.py); the 2023 (ROC 112) crop survey is loaded into "
        "township_crops.csv but not fitted",
        "target.npz, town_kwh.csv and heads_amp.csv do not use the snapshot: fit_window_end "
        "2022-12"]),
    "physics_inversion_hooks": ("2022-12", [
        "twin inputs from hydrophysics.twin.inputs with its default window 2012-01..2022-12 "
        "(inputs_cwa.npz dates 2012-01..2022-12) and the stage-3 datum-gate theta calibrated on "
        "that window (batch 8, provisional)",
        "the theta file was rewritten at 03:06 on 2026-10-12, after the pilot ran (00:41-01:09), "
        "but the archived operator's T, S, f_sh, r_ret and rfrac equal the current theta "
        "(value check, A0 repair), so the hashed theta is the one the pilot used",
        "the forcing caches' own construction (CWA rain rebuild, census cleaning) was not "
        "audited here"]),
    "remediation_plan": ("2020-12", [
        "dem_vs_leveling.py compares the DEM with leveling benchmarks of 2012, 2016 and 2020 "
        "only; no outputs are archived (the numbers went into the plan text)"]),
    "review_cf": ("2026-10", [
        "data-availability probes fetched on 2026-10-12: NGL holdings and HUWE tenv3 to "
        "2026, Open-Meteo pressure and ET0 test responses to 2026-10; nothing fitted, upper "
        "bound = fetch month"]),
    "revise_plan": ("2026-10", [
        "ngl_check.py reads NGL holdings (station spans to 2026); nnls_window.py refits the "
        "TPC NNLS on 2016-01..2025-06 and 2016-01..2022-12; upper bound = fetch month"]),
    "subsidence_gap_sources": ("2025-05", [
        "ring, pile, GNSS and InSAR windows 2023-02..2025-05 on the far side of the 2022 gap "
        "(bridge_pilot.py, bridge_product.py, dbm_vs_ring.py, gnss*.py, insar_pilot.py); the "
        "bridged ring product runs to 2025-05",
        "out/dbm_daily_raw.csv carries a datetime in 2027-03, an impossible future timestamp "
        "in the raw deep-benchmark feed (data-quality issue for WP S2), ignored for this bound"]),
    "surface_water_monthly": ("2025-12", [
        "Choushui daily discharge 2012-2025 from the yearbooks, Jiji annual supplies "
        "2012-2025; district shares chosen leave-one-year-out on 2015-2021 Jan-May with the "
        "annual totals of all years 2012-2025 in every fold (pilot2.py, pilot_disagg.py)"]),
    "tpc_aggregate_electricity": ("2026-08", [
        "TPC county x sector sales 2016-01..2026-08 (d007012), census 2007-01..2025-07; "
        "candidate coefficients fitted on 2016-01..2025-06 (tools/ch_product.py)"]),
}

#: Outputs computed from pre-origin data only (per-output fit_window_end).
PRE_ORIGIN_OUTPUTS = {
    "head_gapfill_pilot": ("2022-12", [
        "out/pilot_N.parquet", "out/pilot_S.parquet", "out/pilot_E.parquet",
        "out/pilot_H.parquet", "out/pilot_ext_N.parquet", "out/pilot_ext_S.parquet",
        "out/pilot_ext_E.parquet", "out/pilot_ext_H.parquet", "out/pilot_all_scored.parquet",
        "out/pilot_summary_NSEH.csv", "out/pilot_summary_cleantruth.csv",
        "out/pilot_ensembles.parquet", "out/pilot_final_clean.csv", "out/pilot_final_all.csv",
        "out/offset_episodes_hist.csv", "out/kalman_theta_rain.npy",
        "out/kalman_theta_norain.npy", "log_N.txt", "log_S.txt", "log_E.txt", "log_H.txt",
        "log_s5.txt"]),
    "landuse_crop_priors": ("2022-12", ["target.npz", "town_kwh.csv", "heads_amp.csv"]),
}

PERSONAL_FIX = {
    "landuse_crop_priors": (
        "target.npz holds per-pump coordinates (pump_xy, TWD97), horsepower (pump_hp), "
        "township (pump_town), class (pump_cls) and grid cell (pump_cell) from the TPC census; "
        "no names or addresses (戶名1 and 用電地址2 were never read). Gitignored results/ only; "
        "never commit or copy into a public product. (r1 said 'none': corrected in r2.)"),
}

#: Pilots that call repository loaders with default paths (load_twin_inputs, load_members):
#: their imported modules are scanned for quoted data paths. Other pilots import only
#: functions that take the data the scripts load themselves (already in r1 inputs).
MODULE_SCAN = {"physics_inversion_hooks"}
#: Curated module-mediated inputs (default-path arguments the static scan cannot resolve).
EXTRA_INPUTS = {
    "remediation_plan": [{
        "path": "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json",
        "via": "hydrophysics/twin/calibrate_flow.py DEFAULT_PATHS['polygon'] (build_grid)"}],
    "physics_inversion_hooks": [{
        "path": "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json",
        "via": "hydrophysics/twin/calibrate_flow.py DEFAULT_PATHS['polygon'] "
               "(load_twin_inputs)"}],
}

_QUOTED = re.compile(r"[\"'](AMP_V2/data/[^\"']+|chou-shui-data/data/[^\"']+|"
                     r"results/[^\"']+|data_fetch_api/[^\"']+)[\"']")
_EXCLUDE = ("results/twin_runs/", "results/dataprep/")
#: Bare roots are too broad to hash as a tree; scripts that name only a root keep it listed
#: under parameters.unresolved_input_patterns.
_BROAD = {"AMP_V2/data", "chou-shui-data/data", "data_fetch_api",
          "data_fetch_api/new_measurements", "results"}


def module_inputs(module_paths: list[str]) -> tuple[list[dict], list[dict]]:
    """(file inputs, directory inputs) named as quoted literals in repository modules."""
    files, dirs = {}, {}
    for mp in module_paths:
        txt = from_repo(mp).read_text(encoding="utf-8", errors="replace")
        for m in _QUOTED.finditer(txt):
            s = m.group(1).rstrip("/")
            if any(s.startswith(x) for x in _EXCLUDE) or any(c in s for c in "*{<"):
                continue
            p = REPO / s
            via = f"{mp} (static scan of quoted paths; possible input)"
            if p.is_file():
                files.setdefault(s, {"path": s, "via": via})
            elif p.is_dir() and s not in _BROAD:
                dirs.setdefault(s, {"path": s, "via": via})
    return list(files.values()), list(dirs.values())


def literal_dirs(m1: dict) -> list[dict]:
    """Directories the pilot scripts name literally (r1 'unresolved' patterns ending in /)."""
    out = {}
    for pat in m1["parameters"].get("unresolved_input_patterns", []):
        s = pat.rstrip("/")
        if not pat.endswith("/") or any(c in s for c in "*{<") or s in _BROAD:
            continue
        if (REPO / s).is_dir() and not any(s.startswith(x) for x in _EXCLUDE):
            out[s] = {"path": s, "via": "pilot script (directory named literally)"}
    return list(out.values())


def r2_for(d: Path, audit_ev: dict, audit_path: Path) -> tuple[dict, dict]:
    m1, _ = load_manifest(d)
    probs = verify_hashes(m1, d, ("code", "outputs"))
    if probs:
        raise RuntimeError(f"{d}: r1 code/outputs no longer reproduce: {probs[:3]}")
    name = d.name
    fwe, basis = AUDIT[name]
    pre_fwe, pre_outs = PRE_ORIGIN_OUTPUTS.get(name, (None, []))
    outs = []
    seen = set()
    for o in m1["outputs"]:
        rec = {k: v for k, v in o.items() if k not in ("sha256", "bytes", "rows")}
        if o["path"] in pre_outs:
            rec["fit_window_end"] = pre_fwe
            seen.add(o["path"])
        outs.append(rec)
    missing = sorted(set(pre_outs) - seen)
    if missing:
        raise RuntimeError(f"{name}: pre-origin outputs not in r1 outputs: {missing}")
    code_paths = [c["path"] for c in m1["code"]]
    repo_mods = [c["path"] for c in m1["code"] if c["path"].startswith("hydrophysics/")]
    mfiles, mdirs = module_inputs(repo_mods) if name in MODULE_SCAN else ([], [])
    mfiles = mfiles + EXTRA_INPUTS.get(name, [])
    have = {i["path"] for i in m1["inputs"]}
    inputs = [i["path"] for i in m1["inputs"]]
    for f in mfiles:
        if f["path"] not in have:
            inputs.append(f)
            have.add(f["path"])
    trees = {t["path"]: t for t in mdirs + literal_dirs(m1)}
    reachable = fwe <= ORIGIN or bool(pre_outs)
    allowed = (["design_time_evidence", "after_the_fact_analysis"] if reachable
               else ["after_the_fact_analysis"])
    params = dict(m1["parameters"])
    params.pop("fit_window_end_basis", None)
    params["fit_window_audit"] = {
        "fit_window_end": fwe,
        "basis": basis,
        "pre_origin_outputs": ({"fit_window_end": pre_fwe, "paths": pre_outs}
                               if pre_outs else None),
        "evidence": {"file": rel_to_repo(audit_path), "sha256": sha256_file(audit_path),
                     "code_post_origin_year_mentions": audit_ev["n_code_mentions"],
                     "max_output_table_month": audit_ev["max_output_table_month"]},
        "u3_u7": ("pre-origin: may serve as design-time evidence for models scored on 2023-26"
                  if fwe <= ORIGIN else
                  ("mixed: only the listed pre-origin outputs may serve as design-time "
                   "evidence; every other number is post-origin (after-the-fact analysis)")
                  if pre_outs else
                  "post-origin: after-the-fact analysis only; a design choice that rests on "
                  "this pilot's numbers is selection-contaminated unless re-made on "
                  "pre-origin data (plan 4.2, U3, U7)"),
    }
    params["input_scan"] = (
        "inputs: data files the pilot scripts name literally (r1 static scan) plus quoted data "
        "paths in the repository modules they import (entries with 'via'; possible inputs, "
        "not proven reads); input_trees: directories read as a whole. All hashed by A0 on "
        "2026-10-12, not at pilot run time. Data a module builds from non-literal paths, and "
        "the bare roots in unresolved_input_patterns, remain unrecorded.")
    params["supersedes"] = {"manifest": R1_NAME, "version": m1["version"],
                            "sha256": sha256_file(d / MANIFEST_NAME)}
    params["repair_round"] = (
        "A0 repair 2026-10-12: per-pilot fit_window_end from a data-window audit; per-output "
        "windows; design_time_evidence (now a pre-origin use) kept only where reachable; "
        "personal_data corrected; module-mediated inputs and input trees recorded")
    personal = PERSONAL_FIX.get(name, m1["personal_data"])
    m2 = build_manifest(
        d, product=m1["product"], layer="pilot", version=VERSION,
        inputs=inputs, code=code_paths, parameters=params, fit_window_end=fwe,
        validation_report=None, gate=m1["gate"], allowed_uses=allowed,
        forbidden_uses=sorted(set(USES) - set(allowed)), personal_data=personal,
        outputs=outs, input_trees=list(trees.values()) or None)
    # the r1 manifest becomes an output of r2 (same bytes once renamed)
    r1_rec = {"path": R1_NAME, "sha256": sha256_file(d / MANIFEST_NAME),
              "bytes": (d / MANIFEST_NAME).stat().st_size, "superseded_manifest": True}
    m2["outputs"].append(r1_rec)
    errs = validate_manifest(m2)
    if errs:
        raise RuntimeError(f"{name}: r2 manifest invalid: {errs}")
    return m1, m2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    audit_path = Path(a.audit).resolve()
    ev = json.loads(audit_path.read_text(encoding="utf-8"))
    dirs = sorted(p for p in PILOTS.iterdir() if p.is_dir() and not p.name.startswith("_"))
    names = {p.name for p in dirs}
    if set(AUDIT) != names:
        raise SystemExit(f"audit table does not match pilots: {set(AUDIT) ^ names}")
    built = {}
    for d in dirs:
        if (d / R1_NAME).exists():
            raise SystemExit(f"{d}: {R1_NAME} already exists; r2 already written?")
        m1, m2 = r2_for(d, ev[d.name], audit_path)
        built[d.name] = (m1, m2)
        print(f"{d.name:28s} fwe {m2['fit_window_end']}  allowed {m2['allowed_uses']}  "
              f"inputs {len(m2['inputs'])} (r1 {len(m1['inputs'])})  trees "
              f"{len(m2.get('input_trees') or [])}  pre-origin outputs "
              f"{sum('fit_window_end' in o for o in m2['outputs'])}")
    if a.dry_run:
        return 0
    index = {"created": "2026-10-12", "by": "WP A0 (repair round)", "version": VERSION,
             "supersedes": {"index": "INDEX.r1.json",
                            "sha256": sha256_file(PILOTS / "INDEX.json")},
             "pilots": []}
    for d in dirs:
        m1, m2 = built[d.name]
        mp = d / MANIFEST_NAME
        os.rename(mp, d / R1_NAME)
        try:
            write_manifest(d, m2)
        except Exception:
            os.rename(d / R1_NAME, mp)
            raise
        index["pilots"].append({
            "scout": d.name, "manifest": rel_to_repo(mp), "manifest_sha256": sha256_file(mp),
            "r1_manifest": rel_to_repo(d / R1_NAME),
            "r1_manifest_sha256": sha256_file(d / R1_NAME),
            "fit_window_end": m2["fit_window_end"],
            "allowed_uses": m2["allowed_uses"],
            "pre_origin_outputs": [o["path"] for o in m2["outputs"] if "fit_window_end" in o],
            "u3_u7": m2["parameters"]["fit_window_audit"]["u3_u7"],
            "n_code": len(m2["code"]), "n_outputs": len(m2["outputs"]),
            "n_inputs": len(m2["inputs"]), "n_input_trees": len(m2.get("input_trees") or []),
            "regenerable_large": [{"path": o["path"], "bytes": o["bytes"], "sha256": o["sha256"],
                                   "generator": o.get("generator")}
                                  for o in m2["outputs"] if o.get("regenerable")],
            "key_scripts": m2["parameters"].get("appendix_a_key_scripts", []),
            "key_outputs": m2["parameters"].get("appendix_a_key_outputs", []),
            "personal_data": m2["personal_data"]})
    old_index = json.loads((PILOTS / "INDEX.json").read_text(encoding="utf-8"))
    index["not_copied_to_l0"] = old_index.get("not_copied_to_l0")
    index["empty_download_dirs"] = old_index.get("empty_download_dirs")
    os.rename(PILOTS / "INDEX.json", PILOTS / "INDEX.r1.json")
    with open(PILOTS / "INDEX.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    print("wrote", PILOTS / "INDEX.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
