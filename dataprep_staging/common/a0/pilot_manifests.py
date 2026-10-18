"""WP A0 step 3: a MANIFEST.json (layer "pilot") for every directory under results/dataprep/_pilots/
plus _pilots/INDEX.json and _pilots/README.md.

Each manifest hashes every script (code) and every other file (outputs), marks the large
regenerable arrays with their generating script, maps each download to its L0 copy, and
records the repository data files the scripts name (inputs; static scan, best effort; hashed
now, so a later change of such a file is visible). Pilots are design-time evidence only
(plan 4.5): allowed_uses = design_time_evidence, after_the_fact_analysis; gate not passed;
fit_window_end conservatively 2026-10 (the pilots read 2023-26 data in places).

Usage: python -m common.a0.pilot_manifests --l0-map <l0_migration.json>
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

from common.hashing import iter_files, sha256_file  # noqa: E402
from common.manifest import MANIFEST_NAME, build_manifest, write_manifest  # noqa: E402
from common.paths import PILOTS, REPO, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

CODE_SUFFIXES = {".py", ".sh"}
BIG = 10_000_000                       # bytes: "large regenerable array" threshold
SCRATCH_PLAN = os.environ.get("DATAPREP_PLANNING_SCRATCHPAD", "<planning scratchpad>")
ALLOWED = ["design_time_evidence", "after_the_fact_analysis"]
FORBIDDEN = sorted(set(USES) - set(ALLOWED))

#: Appendix A of the plan: key scripts and outputs per scout.
APPENDIX_A = {
    "tpc_aggregate_electricity": (
        ["tools/tpc_county.py", "tools/shares.py", "tools/decompose.py", "tools/shape_nnls.py",
         "tools/ch_product.py"],
        ["compare_annual.csv", "decompose_annual.csv", "changhua_pump_electricity_candidates.csv",
         "ag_county_backcast_2012_2015.csv"]),
    "billing_and_census_gaps": (
        ["s02_billing.py", "s04_disagg_pilot.py", "s08b_group_size_nearest.py",
         "s11_chowlin_production.py", "s05_census_zeros.py", "s07_layers.py"],
        ["billing_phase.json", "group_size_nearest_phase.json", "chowlin_production.json",
         "census_zeros.json", "layer_response.json"]),
    "head_pumping_signatures": (
        ["scripts/well_features2.py", "scripts/placebo.py", "scripts/cv_shallow.py",
         "scripts/cross_by_layer.py", "scripts/changhua_distance.py", "scripts/tide_coherence.py"],
        ["out/within_well_summary.csv", "out/placebo.log", "out/cv_shallow.csv",
         "out/tide_coherence.csv"]),
    "physics_inversion_hooks": (
        ["b_operator.py", "c_greens.py", "inv_lib.py", "d_pseudo_changhua.py",
         "e_changhua_inversion.py", "f_temporal_holdout.py", "g_ident_burnin.py"],
        ["d_pseudo_changhua.json", "e_changhua_inversion.json", "f_temporal_holdout.json",
         "g_ident_burnin.json"]),
    "landuse_crop_priors": (
        ["features.py", "models.py", "apply.py", "township_crops.py", "combine.py"],
        ["cv_results.csv", "prior_summary.json", "prior_cells_ci.csv", "combine.json"]),
    "subsidence_gap_sources": (
        ["scripts/bridge_pilot.py", "scripts/bridge_product.py", "scripts/dbm_vs_ring.py",
         "scripts/gnss_stepfix.py", "scripts/insar_pilot.py"],
        ["out/bridge_real_gap.csv", "out/bridge_synthetic.csv", "out/product/",
         "out/insar_gnss_loso.csv"]),
    "surface_water_monthly": (
        ["parse_hyb.py", "annual_check.py", "pilot2.py", "head_check.py"],
        ["pilot2_report.json", "monthly_deliveries_pilot2.csv",
         "jiji_janmay_supply_2015_2021.csv"]),
    "head_gapfill_pilot": (
        ["lib.py", "s3_pilot.py", "s4_daily.py", "s5c_ensemble.py", "s6_apply.py",
         "s7_excluded.py"],
        ["out/pilot_final_clean.csv", "out/fill_hist.parquet", "out/recovery_hist.csv",
         "out/offset_episodes_hist.csv"]),
    "design_physics_inversion": (
        ["p1_channels.py", "p2_pseudo.py", "p3_burnin_stress.py"],
        ["p1_channels.json", "p2_pseudo.json", "p3_burnin_stress.json"]),
    "design_ml_transfer": (
        ["a1_domain_shift.py", "a2_robust_transfer.py", "a5_class_composite.py",
         "p3_changhua_ml_level.py"],
        ["a1_results.json", "a2_scores.csv", "a5_summary.csv", "p3_results.json",
         "p3_results_v1_gls_datum.json"]),
    "design_bayesian_fusion": (
        ["fusion_common.py", "p5_meta_fusion.py", "p6_wra_janmay.py", "p8_combined_variants.py",
         "tools/digitize_wra.py"],
        ["p5_meta_fusion.json", "p6_wra_janmay.json", "p8_combined_variants.csv"]),
    "remediation_plan": (["dem_vs_leveling.py"], []),
    "review_cf": (["scripts/ngl_filter.py", "scripts/tenv3_check.py"],
                  ["dl_ngl/DataHoldings.txt", "dl_ngl_huwe/HUWE.tenv3", "dl_openmeteo/om.json",
                   "dl_openmeteo/et0.json", "dl_licsbas_era5/era5.py"]),
    "revise_plan": (["scripts/ngl_check.py", "scripts/nnls_window.py"], []),
    "audit_2026-10-11": (["artefacts.py", "clean_rescore.py", "heads_profile.py"],
                         ["artefact_events.csv", "rescore_rows.csv", "wells_profile.csv"]),
}

#: Generators of large files the static scan cannot resolve (f-string names, caches).
GENERATOR_OVERRIDES = {
    "landuse_crop_priors/classified_w7.npy": "features.py (classified_image cache, win=7)",
    "head_pumping_signatures/out/kwh_matrix.npz": "scripts/build_kwh_matrix.py",
    "head_pumping_signatures/out/well_month_features.parquet": (
        "scripts/well_features2.py (scripts/well_features.py was the killed first version)"),
}

PERSONAL = {
    "billing_and_census_gaps": (
        "pump coordinates (x, y TWD97) and TPC meter codes in census.parquet, meters.parquet and "
        "meters_phase.parquet; no names or addresses (戶名1 and 用電地址2 were never read). "
        "Gitignored results/ only; never commit or copy into a public product."),
    "tpc_aggregate_electricity": (
        "dl_meta/*.json are data.gov.tw metadata records that include publisher contact fields "
        "(public); never copy those fields into products. Census aggregates only otherwise."),
    "head_pumping_signatures": (
        "out/kwh_matrix.npz holds per-pump coordinates (TWD97), HP, class and billing codes; "
        "no names or addresses. Gitignored results/ only."),
}

_DATA_PAT = re.compile(r"(AMP_V2/data|chou-shui-data/data|results/[A-Za-z0-9_.\-]+|"
                       r"data_fetch_api)(/[^\"'\s,)\]}]*)?")
_IMPORT_PAT = re.compile(r"^\s*(?:from|import)\s+(hydrophysics(?:\.[A-Za-z0-9_]+)+)", re.M)
_SAVE_HINT = re.compile(r"save|savez|to_parquet|to_csv|json\.dump|write|open\(", re.I)


def is_code(scout_dir: Path, p: Path) -> bool:
    """Pilot code = .py/.sh files not inside a download directory (dl*/), whose scripts are
    untrusted downloads (data), never pilot code."""
    parts = p.relative_to(scout_dir).parts[:-1]
    return p.suffix in CODE_SUFFIXES and not any(x.startswith("dl") for x in parts)


def scan_inputs(code_files: list[Path]) -> tuple[list[str], list[str], list[str]]:
    """(existing data files named literally, unresolved patterns, imported repo modules)."""
    files, patterns, mods = set(), set(), set()
    for c in code_files:
        txt = c.read_text(encoding="utf-8", errors="replace")
        for m in _DATA_PAT.finditer(txt):
            s = m.group(0).rstrip(".")
            if s.startswith("results/dataprep"):
                continue
            p = REPO / s
            if "{" in s or "*" in s or not p.exists():
                patterns.add(s)
            elif p.is_file():
                files.add(s)
            else:
                patterns.add(s + ("" if s.endswith("/") else "/"))
        for m in _IMPORT_PAT.finditer(txt):
            parts = m.group(1).split(".")
            for k in range(len(parts), 1, -1):
                f = REPO.joinpath(*parts[:k]).with_suffix(".py")
                if f.is_file():
                    mods.add(rel_to_repo(f))
                    break
    return sorted(files), sorted(patterns), sorted(mods)


def find_generator(scout_dir: Path, rel: str, code_files: list[Path]) -> str | None:
    key = f"{scout_dir.name}/{rel}"
    if key in GENERATOR_OVERRIDES:
        return GENERATOR_OVERRIDES[key]
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(Path(rel).name))
    hits = []
    for c in code_files:
        for line in c.read_text(encoding="utf-8", errors="replace").splitlines():
            if (pat.search(line) and _SAVE_HINT.search(line) and "read" not in line
                    and "load(" not in line):
                hits.append(c.relative_to(scout_dir).as_posix())
                break
    return ", ".join(sorted(set(hits))) or None


def manifest_for(d: Path, l0_map: dict) -> dict:
    files = iter_files(d, (MANIFEST_NAME,))
    code = [p for p in files if is_code(d, p)]
    data_inputs, patterns, mods = scan_inputs(code)
    others = [p.name for p in PILOTS.iterdir() if p.is_dir() and p.name != d.name]
    cross = sorted({o for c in code for o in others
                    if o in c.read_text(encoding="utf-8", errors="replace")})
    outputs, unresolved = [], []
    for p in files:
        if is_code(d, p):
            continue
        rel = p.relative_to(d).as_posix()
        rec = {"path": rel}
        key = f"{d.name}/{rel}"
        if key in l0_map:
            rec["l0_copy"] = l0_map[key]
        size = p.stat().st_size
        if size >= BIG and p.suffix in {".npy", ".npz", ".parquet", ".csv"}:
            gen = find_generator(d, rel, code)
            rec["regenerable"] = True
            rec["generator"] = gen or "not found by static scan"
            if gen is None:
                unresolved.append(rel)
        outputs.append(rec)
    key_scripts, key_outputs = APPENDIX_A.get(d.name, ([], []))
    params = {
        "scout": d.name,
        "origin": (f"{SCRATCH_PLAN}/plan/{d.name}" if d.name not in
                   ("review_cf", "revise_plan", "audit_2026-10-11")
                   else f"{SCRATCH_PLAN}/{d.name if d.name != 'audit_2026-10-11' else 'audit'}"),
        "staged_copy": "results/dataprep/pilots_2026-10-12 (main session), moved by A0",
        "appendix_a_key_scripts": key_scripts,
        "appendix_a_key_outputs": key_outputs,
        "input_scan": ("static scan of the scripts for repository data paths; files hashed by A0 "
                       "on 2026-10-12, not at pilot run time"),
        "unresolved_input_patterns": patterns,
        "repo_modules_imported": mods,
        "reads_other_pilots": cross,
        "fit_window_end_basis": ("conservative: pilots were not audited for their data windows "
                                 "and several read 2023-26 data; treat every pilot number as "
                                 "post-origin design-time evidence"),
        "hardcoded_paths": ("scripts may name the planning scratchpad; run them through "
                            "common/a0/rerun_pilots.py-style path patching, never edit the "
                            "archived copies"),
    }
    if unresolved:
        params["regenerable_without_generator"] = unresolved
    code_recs = [p for p in code] + [REPO / m for m in mods]
    return build_manifest(
        d, product=f"_pilots/{d.name}", layer="pilot", version="2026-10-12",
        inputs=data_inputs, code=code_recs, parameters=params, fit_window_end="2026-10",
        validation_report=None,
        gate={"passed": False, "criteria": "pilot evidence, not a validated product; "
                                           "design-time evidence only (plan 4.5)"},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data=PERSONAL.get(d.name, "none"), outputs=outputs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0-map", required=True)
    a = ap.parse_args()
    l0 = json.loads(Path(a.l0_map).read_text(encoding="utf-8"))
    l0_map, not_copied = l0["l0_map"], l0["not_copied"]
    index = {"created": "2026-10-12", "by": "WP A0", "pilots": []}
    for d in sorted(p for p in PILOTS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        mp = d / MANIFEST_NAME
        if mp.exists():
            print("exists, skipped:", mp)
            m = json.loads(mp.read_text(encoding="utf-8"))
        else:
            m = manifest_for(d, l0_map)
            write_manifest(d, m)
        regen = [o for o in m["outputs"] if o.get("regenerable")]
        index["pilots"].append({
            "scout": d.name, "manifest": rel_to_repo(mp), "manifest_sha256": sha256_file(mp),
            "n_code": len(m["code"]), "n_outputs": len(m["outputs"]),
            "n_inputs": len(m["inputs"]),
            "bytes_outputs": sum(o["bytes"] for o in m["outputs"]),
            "n_l0_copies": sum(1 for o in m["outputs"] if "l0_copy" in o),
            "regenerable_large": [{"path": o["path"], "bytes": o["bytes"],
                                   "sha256": o["sha256"], "generator": o["generator"]}
                                  for o in regen],
            "key_scripts": m["parameters"]["appendix_a_key_scripts"],
            "key_outputs": m["parameters"]["appendix_a_key_outputs"],
            "personal_data": m["personal_data"]})
        print(f"{d.name:28s} code {len(m['code']):3d} outputs {len(m['outputs']):4d} "
              f"inputs {len(m['inputs']):3d} regen {len(regen)}")
    index["not_copied_to_l0"] = not_copied
    index["empty_download_dirs"] = l0["empty_download_dirs"]
    (PILOTS / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n",
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
