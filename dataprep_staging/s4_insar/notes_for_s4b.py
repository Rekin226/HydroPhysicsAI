"""WP S4a: hand-over notes for S4b (exact paths, grids, network and epoch inventory, caveats).

Assembles ``results/dataprep/insar_tools/s4a_run/notes_for_S4b.json`` from the validation
records (``validate.py``), the reference grids, the footprint coverage, the fetch decision and
the environment record. ``build_product.py`` copies it into the product's ``inventory.json``.
Descriptive only: no deformation value is read.

Usage (any env): python -I notes_for_s4b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # dataprep_staging (common)
sys.path.insert(0, str(HERE))

from common.paths import rel_to_repo  # noqa: E402
from config import FRAMES, LICSBAS_DIR, LICSBAS_ENV, RUN_DIR  # noqa: E402


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    env = jload(RUN_DIR / "env" / "env_record.json")
    cov = jload(RUN_DIR / "footprint_coverage.json")
    dec = jload(RUN_DIR / "fetch_decision.json")
    frames = {}
    for f, spec in FRAMES.items():
        l0 = Path(spec["l0_dir"])
        v = jload(RUN_DIR / f"validation_{f}.json")
        e = {"l0_dir": rel_to_repo(l0), "source_json": rel_to_repo(l0 / "SOURCE.json"),
             "window": [spec["start"], spec["end"]],
             "active_cells_inside_footprint": cov.get(f)}
        if v:
            net = v["network_fetched"]
            e.update({
                "fetched": True, "geoc_dir": rel_to_repo(l0 / "GEOC"),
                "baselines": rel_to_repo(l0 / "metadata" / "baselines"),
                "metadata_txt": rel_to_repo(l0 / "metadata" / "metadata.txt"),
                "pair_file_pattern": "GEOC/<yyyymmdd_yyyymmdd>/<pair>.geo.{unw,cc}.tif",
                "metadata_rasters": {k: m.get("file") for k, m in
                                     v["metadata_rasters"].items()},
                "reference_grid": v["reference_grid"],
                "n_pairs_fetched": v["n_pairs_fetched"],
                "n_pairs_unavailable_on_portal": v["n_pairs_unavailable"],
                "unavailable_pairs": [u["pair"] for u in v["unavailable_pairs"]],
                "n_epochs": net.get("n_epochs"), "first_epoch": net.get("first_epoch"),
                "last_epoch": net.get("last_epoch"), "components": net.get("components"),
                "epochs_per_year": net.get("epochs_per_year"),
                "largest_epoch_gaps_days": net.get("largest_epoch_gaps_days"),
                "btemp_days_median": net.get("btemp_days_median"),
                "unw_georef_variants": v["georef_variants_unw"],
                "metadata_georef_variant": v["metadata_georef_variant"],
                "n_pairs_off_reference_grid": v["n_pairs_off_reference_grid"],
                "valid_frac_window_unw_min": v["valid_frac_window_unw_min"],
                "pairs_empty_window": v["pairs_empty_window"],
                "network_without_empty_pairs": v["network_fetched_without_empty_pairs"],
                "alignment": v["alignment"],
                "reference_geotransform_of_window": v["reference_geotransform_of_window"],
                "gnss_stations_in_box": v["gnss_stations_in_box"],
                "pairs_table": f"pairs_{f}.csv", "epochs_table": f"epochs_{f}.csv"})
        else:
            d = dec.get("frames", {}).get(f, {})
            e.update({"fetched": False, "fetch_rule_outcome": d,
                      "listings": rel_to_repo(l0 / "listings"),
                      "pair_directory_pages": rel_to_repo(l0 / "listings" / "ifg_dirs"),
                      "fetch_command": (
                          "set config.FETCH_TAG to a new fetch directory (this one has a "
                          "write-once SOURCE.json), rerun fetch_listings.py and "
                          "fetch_pair_dirs.py for it, then: dataprep_staging/tools/cpu_slot.sh "
                          "dataprep_staging/s4_insar/licsbas_env.sh python -I "
                          "dataprep_staging/s4_insar/fetch_subsets.py --frame " + f +
                          " --pairs-file results/dataprep/insar_tools/s4a_run/pairs_order_" + f +
                          ".txt --threads 8")})
        frames[f] = e
    notes = {
        "frames": frames,
        "licsbas": {"checkout": rel_to_repo(LICSBAS_DIR),
                    "commit": env.get("licsbas", {}).get("commit"),
                    "env": str(LICSBAS_ENV),
                    "wrapper": "dataprep_staging/s4_insar/licsbas_env.sh <command> (sets PATH, "
                               "PYTHONPATH, GDAL_DATA, single-threaded BLAS, "
                               "CUDA_VISIBLE_DEVICES=''); wrap long runs in "
                               "dataprep_staging/tools/cpu_slot.sh",
                    "workdir_helper": "dataprep_staging/s4_insar/make_licsbas_workdir.py "
                                      "--frame F --out DIR [--start --end --max-pairs]: a GEOC/ "
                                      "of symlinks into L0 (never write into L0)",
                    "smoke_run": "results/dataprep/insar_tools/s4a_run/smoke/ (steps 02, 11, "
                                 "12, 13, 14 on 40 pre-origin pairs; smoke_result.json)",
                    "env_deviation": env.get("env", {}).get(
                        "deviation_from_LiCSBAS_environment_yml")},
        "caveats": [
            "Grids: every file of a frame was clipped with ONE index window of the frame's "
            "reference grid (its geo.E.tif), so all rasters of a frame stack by array index. "
            "Some portal batches carry a stretched unw geotransform (dx = 0.001 W/(W-1), origin "
            "half a pixel west/north): a metadata artefact (0 valid unw pixels where cc == 0 at "
            "zero shift, hundreds at +-1 px). Each L0 file keeps its source geotransform shifted "
            "by the window; S4b must assign the reference geotransform by index (plan S4 step "
            "7) and never resample a stretched file by its own geotransform. LiCSBAS02 stacks "
            "by index and takes the geotransform from the first tif it finds in GEOC.",
            "LiCSBAS georeference: LiCSBAS02 takes the geotransform of the first GeoTIFF it "
            "reads. In the mixed-variant smoke run cum.h5 got the stretched spacing (post_lon "
            "0.00200067 deg after 2x2 multilooking) instead of 0.002. S4b must set the "
            "reference geotransform of the window (reference_geotransform_of_window) explicitly "
            "or build each GEOC from one variant.",
            "Primary frame epoch 20260104: its 4 pairs have no valid pixel in the clip (portal "
            "cc files of 48.7 KB for the whole frame: the acquisition misses the fan), so the "
            "usable primary network ends 20251211 (205 epochs, one component). Epoch 20200211 "
            "is decorrelated (unw valid on 4-5 % of the window in its 7 pairs; pair "
            "20200205_20200211 has no GeoTIFF on the portal); LiCSBAS11 rejects such pairs.",
            "105D_06628_232108: 4 pairs (20250123_20250228, 20250312_20250405, "
            "20250417_20250604, 20250523_20250628) have an empty unw on the portal (182 KB "
            "files; their cc is normal).",
            "Masks: for the portal generation processed 2025-12-27 / 2026-01-05 (deflate-strip "
            "unw), and for all of 105D, a few hundred boundary pixels carry a valid unw where "
            "cc == 0 (at most 787 of about 290,000 valid pixels). This is not a misregistration: "
            "every unw has 0 valid pixels over the sea of the landmask, and every coherent cc "
            "correlates best at shift (0, 0) with the reference cc (the 7 pairs with epoch "
            "20200211 are inconclusive, correlation about 0). AREA_OR_POINT tags differ between "
            "generations with identical GDAL geotransforms; stack by index only.",
            "Footprint: the primary frame covers about 91 % of the twin's active cells; the "
            "southern band (latitude about 24.02-24.20) is outside it. The two full-fan frames "
            "cover all active cells but only from late 2022 (069A_06632) or with GeoTIFFs only "
            "for 2025-01..2026-01 (105D_06628: 108 of 362 listed in-window pairs).",
            "Atmosphere: no correction (decision D11 = none); no GACOS/ICAMS products fetched.",
            "Vertical: the clipped U raster gives the per-pixel vertical LOS component; the "
            "plan's constant U = 0.833 is an approximation of it. Before 2022-11 only the "
            "ascending primary frame exists (east-west leakage, plan S4 risk).",
            "Leakage: InSAR is a validator of twin subsidence 2022-26 (U7): never a column "
            "target, never a gate or threshold input. The product's fit_window_end is 2026-01; "
            "the 2020-2022 epochs are observations, not a pre-origin product. Reference ties "
            "to on-fan GNSS and leveling are offset-only (plan S4 step 5).",
            "Coherence: S4a computed no coherence statistic and no validity at the twin's "
            "cells (only the valid fraction over the whole clip window, as an integrity check, "
            "and unw validity at the 22 NGL station pixels in sampled pairs). S4b's per-zone "
            "summer-coherence acceptance is still unlooked-at: pre-register it before "
            "computing any coherence over Changhua zones (A1 look log).",
            "Transfer: the portal throttles bursts of parallel range requests (HTTP 429); "
            "fetch_subsets.py uses serial multi-range per connection, HTTP/1.1, at most 8 "
            "threads; throughput varied from about 25 KB/s to 1 MB/s per connection on "
            "2026-10-12.",
        ],
    }
    (RUN_DIR / "notes_for_S4b.json").write_text(json.dumps(notes, indent=1) + "\n")
    print(json.dumps({f: {k: e.get(k) for k in ("fetched", "n_pairs_fetched", "n_epochs")}
                      for f, e in frames.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
