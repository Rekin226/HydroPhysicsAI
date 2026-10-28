"""Ledger rows (U6) for the numbers of report X5.shutdown_sdid.v1. Run under the shared lock:

    flock results/dataprep/preregistration/.x5.lock python -I ledger_x5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5const as K  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

LEV = "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
SOURCES = {
    LEV: ("design_time_evidence", "leveling benchmark elevations (outcome)"),
    rel_to_repo(K.RINGS): ("design_time_evidence", "ring band compaction (outcome, rings)"),
    rel_to_repo(K.WELLS_LOCATED): ("input", "proxy locations of the stopped TWC wells"),
    rel_to_repo(K.PUBLIC_WELL_JANMAY): ("input", "event size and B02 township dose"),
    rel_to_repo(K.ZONE_MAP): ("input", "on-grid test and county"),
    rel_to_repo(K.CENSUS_CELLS): ("input", "census kWh within 3 km (G2, secondary)"),
}
VERSION = {rel_to_repo(K.RINGS): "rings_bridged/v3",
           rel_to_repo(K.WELLS_LOCATED): "public_wells/v2",
           rel_to_repo(K.PUBLIC_WELL_JANMAY): "natural_experiments/v1",
           rel_to_repo(K.ZONE_MAP): "zone_county_map/v2",
           rel_to_repo(K.CENSUS_CELLS): "pump_electricity/full/v3"}
NUMBERS = {
    "X5.T2.primary_att": [LEV, rel_to_repo(K.WELLS_LOCATED), rel_to_repo(K.ZONE_MAP)],
    "X5.T2.lead_placebo": [LEV, rel_to_repo(K.WELLS_LOCATED), rel_to_repo(K.ZONE_MAP)],
    "X5.T2.golden_corridor_sensitivities": [LEV, rel_to_repo(K.WELLS_LOCATED),
                                            rel_to_repo(K.ZONE_MAP),
                                            rel_to_repo(K.CENSUS_CELLS)],
    "X5.T1.att": [LEV, rel_to_repo(K.ZONE_MAP), rel_to_repo(K.WELLS_LOCATED)],
    "X5.rings_T1.band_att": [rel_to_repo(K.RINGS)],
    "X5.secondary.long_difference": [LEV, rel_to_repo(K.CENSUS_CELLS), rel_to_repo(K.ZONE_MAP),
                                     rel_to_repo(K.WELLS_LOCATED)],
    "X5.event.dose": [rel_to_repo(K.PUBLIC_WELL_JANMAY), rel_to_repo(K.WELLS_LOCATED)],
}


def main() -> int:
    rows = []
    for num, srcs in NUMBERS.items():
        for s in srcs:
            role, note = SOURCES[s]
            rows.append({"number_id": num, "source": s, "role": role,
                         "product_version": VERSION.get(s, "L0 read-only cache"),
                         "note": f"{note}; report {rel_to_repo(K.REPORT_DIR / 'report.json')}"})
    p = ledger.append_rows(K.WP, rows)
    print(rel_to_repo(p), len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
