"""Prepare a local irrigation-electricity index for the origin-safe AMP pilot.

Meter selection uses pre-2021 records only. All pumps sharing a local pump's meter
are retained during deduplication, then the allocated local irrigation share is summed.
This remains electricity, not measured water volume; registry vintage is retrospective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrophysics.twin.pumping import METER_COL, clean_census
from hydrophysics.twin.scenario import purpose_class

from .forecast_experiment import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--kwh", type=Path, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--radius", type=float, default=1000.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pumps = pd.read_parquet(args.census, columns=["sid", "PUMP_HP", METER_COL,
                                                "PURPOSE", "TWD97_X", "TWD97_Y"])
    pumps["sid"] = pumps.sid.astype(str)
    meter = pumps[METER_COL].astype(str)
    pumps[METER_COL] = meter.where(~meter.isin(["", "nan", "None"]), pumps.sid)
    distance = np.hypot(pd.to_numeric(pumps.TWD97_X, errors="coerce") - args.x,
                        pd.to_numeric(pumps.TWD97_Y, errors="coerce") - args.y)
    local = pumps[(distance <= args.radius) & (pumps.PURPOSE.map(purpose_class) == "irrigation")]
    meters = set(local[METER_COL])
    family = pumps[pumps[METER_COL].isin(meters)].copy()
    kwh = pd.read_parquet(args.kwh, filters=[("pump", "in", family.sid.tolist())])
    kwh = kwh[np.isfinite(kwh.electricity_kwh) & (kwh.electricity_kwh >= 0)]
    # Exclude registry entries lacking historical observations before selecting meters.
    train = kwh[kwh.datetime < pd.Timestamp("2021-01-01")]
    historical_meters = set(family.loc[family.sid.isin(train.pump.astype(str)), METER_COL])
    family = family[family[METER_COL].isin(historical_meters)]
    kept, _, _ = clean_census(family, train, cap_duty=1.,
                              t0="2012-01-01", t1="2021-01-01")
    _, allocated, _ = clean_census(kept, kwh, cap_duty=None)
    ids = set(local.sid) & set(kept.sid)
    allocated = allocated[allocated.pump.isin(ids)].copy()
    allocated["month"] = pd.to_datetime(allocated.datetime).dt.to_period("M").dt.to_timestamp()
    # An incomplete meter population is missing evidence, not zero abstraction.
    allocated["meter"] = allocated.pump.map(kept.set_index("sid")[METER_COL])
    grouped = allocated.groupby("month")
    energy = grouped.electricity_kwh.sum(min_count=1)
    nmeters = kept[kept.sid.isin(ids)][METER_COL].nunique()
    energy = energy.where(grouped.meter.nunique() >= .9 * nmeters)
    if nmeters == 0 or energy.notna().sum() < 36:
        raise ValueError("Insufficient local irrigation-meter coverage")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    energy.rename("energy_kwh").rename_axis("datetime").to_csv(args.out)
    manifest = {"census_sha256": sha256(args.census), "kwh_sha256": sha256(args.kwh),
                "code_sha256": sha256(__file__), "index_sha256": sha256(args.out),
                "center_twd97": [args.x, args.y], "radius_m": args.radius,
                "local_irrigation_pumps": len(ids), "meters": nmeters,
                "training_cutoff_exclusive": "2021-01-01", "min_meter_coverage": .9,
                "valid_months": int(energy.notna().sum()),
                "limitations": ["Static registry, not verified historical registry snapshots",
                                "Electricity release dates unknown; apply an explicit lag",
                                "Electricity is a proxy, not water volume"]}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {energy.notna().sum()} covered months from {nmeters} meters")


if __name__ == "__main__":
    main()
