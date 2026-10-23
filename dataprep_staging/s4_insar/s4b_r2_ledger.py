"""WP S4b repair round: ledger rows (plan 4.1, U6) for the numbers the repair section quotes.

    python -I s4b_r2_ledger.py      (hydro env) -> appends to ledger/rows/S4b.csv

Roles as in round 1: LiCSAR L0, LiCSBAS run files, geometry and the units registry are
inputs; WRA GNSS heights are validators of the InSAR numbers that involve them; the census
product is the validator of the amplitude-density numbers and an input of its own flag count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

PV = "insar_069A/v2"
L0_A = "data_fetch_api/new_measurements/licsar_069A_06694/fetch_2026-10-12_s4a"
L0_D = "data_fetch_api/new_measurements/licsar_105D_06628/fetch_2026-10-12_s4a"
S4A = "insar_licsar_subsets/v1"
PREREG = f"results/dataprep/preregistration/{C.PREREG_ID}.json"
CENSUS = "results/dataprep/pump_electricity/preorigin/v1"
META = "chou-shui-data/data/ls_cache/ls-wra-gnss-obs__meta.parquet"


def run_files(name: str, gap: bool = False) -> list[str]:
    ts = C.RUN_DIR / name / "TS_GEOCml2"
    out = [rel_to_repo(ts / "cum.h5"), rel_to_repo(ts / "results" / "mask")]
    if gap:
        out += [rel_to_repo(ts / "info" / "12network_gap_info.txt"),
                rel_to_repo(C.RUN_DIR / f"gaps_{name}.npz")]
    return out


def rows_for(number_id, note, inputs, validators=()) -> list[dict]:
    out = [{"number_id": number_id, "source": s, "role": "input", "product_version": PV,
            "note": note} for s in inputs]
    out += [{"number_id": number_id, "source": s, "role": "validator", "product_version": PV,
             "note": note} for s in validators]
    return out


def main() -> int:
    ev = C.EVID
    diag = json.loads((ev / "repair_diag.json").read_text())
    split = json.loads((ev / "repair_pf_split.json").read_text())
    gaps = json.loads((ev / "repair_gaps.json").read_text())
    files = Lb.wra_files()
    geomA = rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz")
    geomD = rel_to_repo(C.RUN_DIR / f"geom_{C.DESC}.npz")
    base = [L0_A, S4A, PREREG, geomA]
    v1 = "results/dataprep/insar_069A/v1"
    rows = []
    for r in ("P", "F", "D"):
        rows += rows_for(f"S4b.repair.gaps.run{r}",
                         f"network gaps {gaps[r]['gaps']}; pixel n_gap {gaps[r]['pixel_n_gap']}",
                         [L0_D if r == "D" else L0_A, S4A, geomD if r == "D" else geomA,
                          rel_to_repo(C.TWIN_GRID)] + run_files(r, gap=True))
    rows += rows_for("S4b.repair.flags_v2", "vertical_cells flag 0/4/8 counts and segment ids",
                     [f"{v1}/vertical_cells.npz", f"{v1}/MANIFEST.json"] + run_files("F", True))
    rows += rows_for("S4b.repair.decomp_2025_gaps", "decomposition cells with a run-F pixel gap "
                     "in 2025 (0) and run D gaps (none)",
                     [f"{v1}/decomposition_2025.npz"] + run_files("F", True)
                     + run_files("D", True))
    pf = diag["pf"]
    rows += rows_for("S4b.repair.pf_intervals",
                     f"InSAR-only P vs F: gap1 {pf['across_gap1'][0]['rms_mm']:.2f}, gap2 "
                     f"{pf['across_gap2'][0]['rms_mm']:.2f}, within 18-30 d max "
                     f"{pf['within_18_30d_rms_mm']['max']:.2f}, last segment "
                     f"{pf['last_segment']['rms_mm']:.2f}, G1 change "
                     f"{pf['g1_change']['rms_mm']:.2f} mm",
                     base + run_files("P", True) + run_files("F", True))
    for nm in ("P", "F"):
        p = diag["pieces"][nm]
        rows += rows_for(f"S4b.repair.g1_pieces.run{nm}",
                         f"G1 split run {nm}: g1 {p['g1_rms_mm']:.2f}, gap1 {p['gap1_rms_mm']:.2f},"
                         f" gap2 {p['gap2_rms_mm']:.2f}, rest {p['rest_rms_mm']:.2f}, single "
                         f"interval median {p['within_single_interval_rms_mm']['median']:.2f} mm",
                         base + run_files(nm, True) + [META, rel_to_repo(C.RUN_DIR /
                                                                         "gnss_inventory.json")],
                         [rel_to_repo(files[s]) for s in p["stations"]])
    rows += rows_for("S4b.repair.pf_station_split",
                     f"err(F)-err(P): {split['rms_err_change_mm']:.2f} mm; gap part "
                     f"{split['rms_gap_part_change_mm']:.2f}; within "
                     f"{split['rms_within_segment_part_change_mm']:.2f}; corr "
                     f"{split['corr_err_change_gap_change']:.3f}",
                     base + run_files("P", True) + run_files("F", True)
                     + [META, rel_to_repo(C.RUN_DIR / "gnss_inventory.json")],
                     [rel_to_repo(files[s]) for s in split["stations"]])
    rc = diag["refcoef"]
    rows += rows_for("S4b.repair.refcoef",
                     f"P east annual amp {rc['P']['east_km']['annual_amp']:.3f} mm/km; F drifts "
                     f"east {rc['F']['east_km']['rate_per_yr']:+.3f}, north "
                     f"{rc['F']['north_km']['rate_per_yr']:+.3f} mm/km/yr; centroid x "
                     f"{rc['stable_centroid_x_km']:.1f} km",
                     base + run_files("P") + run_files("F") + [rel_to_repo(C.TWIN_GRID)])
    ex = "results/dataprep/reports/S4b_evidence/explore_ramp.json"
    rows += rows_for("S4b.repair.ramp_north_slope",
                     "explore_ramp.json north slopes (plane +0.72, offset+height +0.18 mm/yr/km)",
                     [ex] + run_files("F") + [META],
                     [rel_to_repo(files[s]) for s in sorted(
                         {r["station"] for v in json.loads((C.REPO / ex).read_text()).values()
                          for r in v["rows"]})])
    yc = diag["yconf"]
    ybase = base + run_files("P") + [rel_to_repo(C.UNITS_V3), rel_to_repo(C.TWIN_GRID),
                                     rel_to_repo(ev / "seasonal_amp_cells_P.npy")]
    rows += rows_for("S4b.repair.y_amp_easting",
                     f"rho(amplitude, easting) {yc['rho_amp_easting']:.3f}, northing "
                     f"{yc['rho_amp_northing']:.3f}", ybase)
    rows += rows_for("S4b.repair.y_census_easting",
                     f"rho(density 5 km, easting) {yc['rho_density5_easting']:.3f}; partial "
                     f"rho(amp, density | easting) {yc['partial_rho_amp_density5_given_easting']:.3f}"
                     f"; placebo {yc['partial_rho_amp_placebo_given_easting']:.3f}",
                     ybase, [CENSUS])
    rows += rows_for("S4b.repair.census_flag9",
                     f"flag-9 cell-months 2020-22: {yc['census_flag9_cell_months_2020_22']}",
                     [CENSUS, rel_to_repo(C.UNITS_V3)])
    rows += rows_for("S4b.repair.ledger_rows_round1", "rows/S4b.csv had 387 data rows after "
                     "round 1 (388 lines with the header)",
                     ["results/dataprep/ledger/rows/S4b.csv"])
    p = ledger.append_rows(C.WP, rows)
    nums = sorted({r["number_id"] for r in rows})
    print(json.dumps({"fragment": str(p), "rows": len(rows), "numbers": len(nums),
                      "unledgered": ledger.unledgered(nums)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
