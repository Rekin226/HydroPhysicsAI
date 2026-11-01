"""Pre-origin MANIFEST for the forward's own rain input (FW.weather_replay.v1 prerequisite).

The page theta was fitted on the vendor rain cache chou-shui-data/data/rf_timeseries_vendor_
2012_2022.csv (FS.start_invariance.v1: the page run read it), and every page forward reads it
as frozen forcing. No dataprep MANIFEST names it, so a weather replay may not use it as
``forcing_frozen`` until one exists (round-7 plan, rank 11: 'issue a pre-origin rain MANIFEST
first, or run pumping-only and say so'). This script issues that manifest as product
``twin_rain/preorigin`` v1 (layer L1): the file is not copied or changed; the product holds a
structure record (dates, gauge columns, empty-cell count, no rain value) and a validation
record, and names the cache and the gauge table as inputs.

Allowed uses: forcing_frozen (the page model's frozen forcing, as calibrated) and
design_time_evidence. Every other use is forbidden. Known defects carried into any use: the
vendor cache zero-fills gauge rf11 for all of 2021-22 and rf15 in 2015 H2 (CWA station
inventory note); the recalibration on the rebuilt CWA rain changed no score, and the page
stays on the model fitted on this cache.

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/manifest_rain.py \\
        --validate --scratch <dir outside the repository>
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/manifest_rain.py --write

``--validate`` builds the product in the scratch directory and prints the manifest and its
schema check; ``--write`` (the lead only) writes results/dataprep/twin_rain/preorigin/v1/ once.
Neither prints a rain value: the structure check reads the cache's dates, header and the count
of empty cells.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from common import manifest as M  # noqa: E402
from common.uses import USES  # noqa: E402

PRODUCT = "twin_rain/preorigin"
VERSION = "v1"
RF = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
RF_STATIONS = "chou-shui-data/data/rf_stations.csv"
FIRST, LAST = dt.date(2012, 1, 1), dt.date(2022, 12, 31)
ALLOWED = ["forcing_frozen", "design_time_evidence"]
FORBIDDEN = sorted(set(USES) - set(ALLOWED))
SELF = "dataprep_staging/fw_replay/manifest_rain.py"


def _read_header_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def structure(rain: Path, stations: Path) -> dict:
    """Dates, gauge columns and empty-cell count of the rain cache; no value is kept."""
    with rain.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        days, empty, bad_rows, nonnum = [], 0, 0, 0
        for row in rd:
            if len(row) != len(head):
                bad_rows += 1
                continue
            days.append(row[0].strip()[:10])
            for v in row[1:]:
                s = v.strip()
                if not s:
                    empty += 1
                    continue
                try:
                    float(s)
                except ValueError:
                    nonnum += 1
    st = _read_header_ids(stations)
    if "rf_id" not in st:
        raise ValueError(f"{stations}: no rf_id column")
    with stations.open(encoding="utf-8-sig", newline="") as f:
        ids = {r["rf_id"].strip() for r in csv.DictReader(f)}
    gauges = [h.strip() for h in head[1:]]
    try:
        parsed = [dt.date.fromisoformat(d) for d in days]
    except ValueError:
        parsed = []
    want = [FIRST + dt.timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    return {"date_column": head[0].strip(), "n_days": len(days),
            "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
            "daily_and_complete": parsed == want, "n_gauge_columns": len(gauges),
            "gauges": gauges, "gauges_missing_from_station_table":
                sorted(set(gauges) - ids), "empty_cells": empty,
            "non_numeric_cells": nonnum, "malformed_rows": bad_rows}


def gate(rec: dict) -> dict:
    checks = {"daily_2012-01-01_to_2022-12-31": rec["daily_and_complete"],
              "every_gauge_in_station_table": not rec["gauges_missing_from_station_table"],
              "no_empty_cell": rec["empty_cells"] == 0,
              "no_non_numeric_cell": rec["non_numeric_cells"] == 0,
              "no_malformed_row": rec["malformed_rows"] == 0}
    return {"passed": all(checks.values()), "checks": checks,
            "criteria": "structure only: the cache is daily and complete over 2012-01-01.."
                        "2022-12-31, every gauge column is in rf_stations.csv, no empty, "
                        "non-numeric or malformed cell. Provenance: the vendor cache the page "
                        "theta was fitted on (docs/TWIN_RELEASE.md; FS.start_invariance.v1 "
                        "reproduced the page run with it). It says nothing about rain "
                        "accuracy (known zero-filled gauges, see notes)"}


def build(product_dir: Path, rain: Path | None = None, stations: Path | None = None) -> dict:
    """Build the product in ``product_dir`` (created; must not exist) -> the manifest."""
    rain = REPO / RF if rain is None else Path(rain)
    stations = REPO / RF_STATIONS if stations is None else Path(stations)
    product_dir.mkdir(parents=True, exist_ok=False)
    rec = structure(rain, stations)
    g = gate(rec)
    (product_dir / "rain_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    (product_dir / "validation.json").write_text(json.dumps(g, indent=1) + "\n")
    m = M.build_manifest(
        product_dir, product=PRODUCT, layer="L1", version=VERSION,
        inputs=[str(rain), str(stations)], code=[SELF],
        parameters={"source": "vendor rain cache the page theta was fitted on",
                    "window": [FIRST.isoformat(), LAST.isoformat()],
                    "consumer": "FW.weather_replay.v1 (hydrophysics.twin.replay, the "
                                "forward's own rain input)"},
        fit_window_end="2022-12", validation_report="validation.json",
        gate={"passed": g["passed"], "criteria": g["criteria"]},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none (public rain gauge ids and daily totals)",
        extra={"notes": [
            "Pointer product: the cache is read in place and not copied; its sha256 is in "
            "inputs, and any change refuses every consumer that checks it.",
            "Known vendor-cache defects, carried into every use: gauge rf11 is zero for all "
            "of 2021-22 and rf15 in 2015 H2 (zero-filled, not dry); the page model was "
            "calibrated with them, so a replay of those years reproduces the hindcast's own "
            "forcing, artefact included.",
            "forcing_frozen here means the page model's frozen forcing as calibrated "
            "(2012-22); no post-origin month exists in the cache."]})
    errs = M.validate_manifest(m, product_dir)
    if errs:
        raise M.ManifestError("; ".join(errs))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--scratch", default=None, help="--validate: a directory outside the "
                    "repository to build the product in")
    a = ap.parse_args(argv)
    if a.validate:
        if not a.scratch:
            ap.error("--validate needs --scratch")
        d = Path(a.scratch).resolve() / PRODUCT / VERSION
        m = build(d)
        print(json.dumps(m, indent=1, ensure_ascii=False))
        print(f"built in {d}; gate passed: {m['gate']['passed']}; not written to results/")
        return 0 if m["gate"]["passed"] else 3
    d = M.DATAPREP / PRODUCT / VERSION
    if d.exists():
        print(f"REFUSED: {d} exists")
        return 1
    m = build(d)
    if not m["gate"]["passed"]:
        print("REFUSED: the structure gate failed; the product directory is left for "
              "inspection without a manifest")
        return 3
    p = M.write_manifest(d, m)
    print(f"wrote {p} (allowed uses {ALLOWED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
