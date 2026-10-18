"""WP S4a step 7: assemble the hand-over product for S4b (``insar_licsar_subsets/v<N>/``).

Collects, for every frame, the validation and pair tables written by ``validate.py`` (licsbas
env), derives an epoch table, and writes ``inventory.json``: the exact L0 paths, the reference
grids, the network and epoch inventory, the transfer tally against the budget (user decision
D1), the LiCSBAS environment and source, the smoke-test result and the notes S4b needs. The
product holds no pixel values (they stay in L0); it is an index (layer L1) whose manifest hashes
the L0 fetch directories as input trees, so any change to the subsets breaks the checker.

Uses: the subsets are InSAR observations through 2026-01. Plan S4: InSAR is a validator of twin
subsidence from 2022 (never a column target, never a gate: U7), so ``allowed_uses`` is
``validator`` and ``after_the_fact_analysis`` only.

Run with the hydro env, inside a CPU slot (hashes about 2 GB):
  python -I build_product.py [--version vN]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # dataprep_staging (common)
sys.path.insert(0, str(HERE))

from common import manifest  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, rel_to_repo  # noqa: E402
from config import (  # noqa: E402
    FETCH_DATE,
    FRAMES,
    LICENCE,
    LICSBAS_DIR,
    LICSBAS_ENV,
    RUN_CONFIG,
    RUN_DIR,
    TWIN_GRID,
    WP,
    frame_url,
)

PRODUCT = "insar_licsar_subsets"
FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
             "calibration_target", "initial_conditions", "idw_field", "prior",
             "prior_width_input", "model_design", "selection", "forcing_frozen",
             "forcing_recorded_arm", "statistical_layer_history", "design_time_evidence"]
CODE = ["config.py", "network.py", "tiffhead.py", "make_run_config.py", "fetch_listings.py",
        "fetch_pair_dirs.py", "estimate_sizes.py", "order_pairs.py", "fetch_subsets.py",
        "twin_cells.py", "footprint_coverage.py", "validate.py", "check_raw_reader.py",
        "tally.py", "fetch_decision.py", "make_licsbas_workdir.py", "licsbas_env.sh",
        "smoke_licsbas.sh", "smoke_result.py", "hydro_snapshot.sh", "write_sources.py",
        "run_checker.py", "declare_acceptance.py", "evaluate_acceptance.py", "notes_for_s4b.py",
        "build_product.py"]
#: run records copied into the product (provenance of the acceptance evaluation)
RECORDS = ["acceptance_result.json", "transfer_tally.json", "size_estimate.json",
           "fetch_decision.json", "smoke_result.json", "spot_checks.json", "checker_licsar.json",
           "footprint_coverage.json", "notes_for_S4b.json"]


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    a = ap.parse_args()
    out = manifest.new_version_dir(PRODUCT) if not a.version else DATAPREP / PRODUCT / a.version
    if (out / manifest.MANIFEST_NAME).exists():
        raise SystemExit(f"{out} already has a manifest; versions are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    tally = json.loads((RUN_DIR / "transfer_tally.json").read_text())
    smoke = json.loads((RUN_DIR / "smoke_result.json").read_text())
    env = json.loads((RUN_DIR / "env" / "env_record.json").read_text())

    frames, inputs, trees = {}, [], []
    for frame, spec in FRAMES.items():
        l0 = Path(spec["l0_dir"])
        src = l0 / "SOURCE.json"
        inputs.append(str(src))
        trees.append({"path": str(l0), "role": f"L0 fetch directory of {frame}"})
        vfile = RUN_DIR / f"validation_{frame}.json"
        pfile = RUN_DIR / f"pairs_{frame}.csv"
        entry = {"role": spec["role"], "window": [spec["start"], spec["end"]],
                 "portal": frame_url(frame), "l0_dir": rel_to_repo(l0),
                 "source_json": rel_to_repo(src)}
        ref = RUN_DIR / f"reference_grid_{frame}.json"
        if ref.exists():
            entry["reference_grid"] = json.loads(ref.read_text())
        if vfile.exists():
            val = json.loads(vfile.read_text())
            shutil.copy2(vfile, out / vfile.name)
            shutil.copy2(pfile, out / pfile.name)
            with open(pfile, newline="") as fh:
                rows = list(csv.DictReader(fh))
            fetched = [r for r in rows if r["fetched"] == "True"]
            ep = Counter([r["epoch1"] for r in fetched] + [r["epoch2"] for r in fetched])
            with open(out / f"epochs_{frame}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "epoch", "year", "n_pairs_fetched", "preorigin"])
                for e in sorted(ep):
                    w.writerow([frame, e, e[:4], ep[e], e <= "20221231"])
            entry.update({
                "geoc_dir": rel_to_repo(l0 / "GEOC"),
                "metadata_dir": rel_to_repo(l0 / "metadata"),
                "pairs_table": pfile.name, "epochs_table": f"epochs_{frame}.csv",
                "validation": vfile.name,
                "n_pairs_listed": val["n_pairs_listed"],
                "n_pairs_unavailable_on_portal": val["n_pairs_unavailable"],
                "n_pairs_fetched": val["n_pairs_fetched"],
                "n_epochs_fetched": len(val["epochs_fetched"]),
                "first_epoch": val["network_fetched"].get("first_epoch"),
                "last_epoch": val["network_fetched"].get("last_epoch"),
                "components": val["network_fetched"].get("components"),
                "epochs_per_year": val["network_fetched"].get("epochs_per_year"),
                "largest_epoch_gaps_days": val["network_fetched"].get(
                    "largest_epoch_gaps_days"),
                "unw_georef_variants": val["georef_variants_unw"],
                "metadata_georef_variant": val.get("metadata_georef_variant"),
                "n_pairs_off_reference_grid": val["n_pairs_off_reference_grid"],
                "layout_classes": val["layout_classes"],
                "valid_frac_window_unw_min": val["valid_frac_window_unw_min"],
                "pairs_empty_window": val["pairs_empty_window"],
                "network_without_empty_pairs": val["network_fetched_without_empty_pairs"],
                "alignment": val["alignment"],
                "gnss_summary": val.get("gnss_summary"),
                "fetched": True})
        else:
            est = json.loads((RUN_DIR / "size_estimate.json").read_text())["frames"].get(frame,
                                                                                      {})
            entry.update({"fetched": False,
                          "n_pairs_listed": est.get("n_pairs_listed"),
                          "n_pairs_available": est.get("n_pairs_window"),
                          "expected_subset_bytes": est.get("expected_smart_clip_bytes"),
                          "expected_full_bytes": est.get("expected_full_bytes"),
                          "fetch_command": json.loads(
                              (RUN_DIR / "notes_for_S4b.json").read_text())["frames"][frame].get(
                              "fetch_command")})
        frames[frame] = entry

    inv = {
        "wp": WP, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fetch_date": FETCH_DATE, "licence": LICENCE,
        "clip_box": json.loads(RUN_CONFIG.read_text())["box"],
        "run_config": rel_to_repo(RUN_CONFIG),
        "frames": frames, "transfer_tally": tally,
        "licsbas": {"source": rel_to_repo(LICSBAS_DIR), **env.get("licsbas", {})},
        "environment": {"path": str(LICSBAS_ENV), **env.get("env", {})},
        "smoke_test": smoke,
        "acceptance_declaration": rel_to_repo(PREREG_DIR / "S4a.setup_acceptance.v1.json"),
        "notes_for_S4b": json.loads((RUN_DIR / "notes_for_S4b.json").read_text()),
    }
    (out / "inventory.json").write_text(json.dumps(inv, indent=1, ensure_ascii=False) + "\n")
    summary = {"frames": {f: {k: v for k, v in e.items() if k in (
        "fetched", "n_pairs_fetched", "n_pairs_listed", "n_pairs_unavailable_on_portal",
        "n_epochs_fetched", "components", "valid_frac_window_unw_min")}
        for f, e in frames.items()}, "smoke_test_passed": smoke.get("passed"),
        "transfer_total_bytes": tally.get("total_bytes")}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    gate = json.loads((RUN_DIR / "acceptance_result.json").read_text())
    for name in RECORDS:
        shutil.copy2(RUN_DIR / name, out / name)
    m = manifest.build_manifest(
        out, product=PRODUCT, layer="L1",
        inputs=inputs + [str(RUN_CONFIG), str(TWIN_GRID),
                         str(PREREG_DIR / "S4a.setup_acceptance.v1.json"),
                         str(RUN_DIR / "env" / "licsbas_explicit.txt"),
                         str(RUN_DIR / "env" / "env_record.json")]
        + [str(p) for p in sorted(RUN_DIR.glob("fetch_log_*.jsonl"))]
        + [str(RUN_DIR / f"pairs_order_{f}.txt") for f in FRAMES],
        code=[str(HERE / c) for c in CODE],
        parameters={"clip_box_from": "twin grid bbox + 2 km (run_config.json)",
                    "frames": {f: [s["start"], s["end"]] for f, s in FRAMES.items()},
                    "window_rule": "one index window per frame (reference geo.E.tif grid)",
                    "pair_rule": "all available in-window pairs (unw and cc both on portal)"},
        fit_window_end="2026-01",
        validation_report="validation_summary.json",
        gate={"passed": bool(gate["data_gate"]["passed"]),
              "criteria": ("S4a.setup_acceptance.v1, data criteria "
                           + ", ".join(gate["data_gate"]["criteria"])
                           + " (properties of the fetched data); the WP process criteria "
                           "(environment, LiCSBAS source and smoke run, transfer budget, "
                           "full-fan listing) are in acceptance_result.json"),
              "wp_acceptance_passed": bool(gate["passed"]),
              "results": gate["results"]},
        allowed_uses=["validator", "after_the_fact_analysis"],
        forbidden_uses=FORBIDDEN, personal_data="none", input_trees=trees,
        extra={"note": ("index of L0 InSAR subsets (no pixel values here); InSAR is a "
                        "validator of twin subsidence 2022-26, never a column target or gate "
                        "(plan S4, U7); 2020-2022 epochs are observations too and are not a "
                        "pre-origin product either: no fitted parameter, but S4b must not "
                        "use any of it to calibrate or select")})
    p = manifest.write_manifest(out, m)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
