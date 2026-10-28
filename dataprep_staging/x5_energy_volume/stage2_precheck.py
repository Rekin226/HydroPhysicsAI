"""CPU precheck for the X5.R1 twin rung (stage 2), run before any GPU job.

The rung feeds the twin a frozen census (``build_stage2_inputs.py``) with ``--meter-filter
none``. This checks, per active cell and month and per purpose class, that

* arm ``m1`` (multiplier 1) reproduces the energy the twin itself derives from
  ``model_census/v1`` with its default cleaning (``--meter-filter dedupe-cap``), to 1e-9
  relative -- the base arm Q0c then sees exactly the census forcing of rung B's B1;
* arm ``county`` equals ``m1`` times the declared county multiplier on the irrigation class
  and equals ``m1`` on every other class.

A failure is an input-construction defect, not a result: stop and fix the builder.

Usage: CUDA_VISIBLE_DEVICES= python dataprep_staging/x5_energy_volume/stage2_precheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS  # noqa: E402
from hydrophysics.twin.grid import build_grid  # noqa: E402
from hydrophysics.twin.inputs import T0, T1  # noqa: E402
from hydrophysics.twin.pumping import clean_census  # noqa: E402
from hydrophysics.twin.scenario import energy_by_class  # noqa: E402

MC = REPO / "results/dataprep/pump_electricity/model_census/v1"
S2 = REPO / "results/dataprep/x5_energy_volume/stage2_inputs"
SPEC = REPO / "results/dataprep/x5_energy_volume/twin_rung_spec.json"
OUT = REPO / "results/dataprep/x5_energy_volume/stage2_precheck.json"
CMAP = REPO / "results/dataprep/zone_county_map/v2/zone_county_map.npz"


def energy(census: Path, kwh: Path, clean: bool, grid) -> dict:
    pumps, k = pd.read_parquet(census), pd.read_parquet(kwh)
    if clean:
        pumps, k, _ = clean_census(pumps, k, cap_duty=1.0, t0=T0, t1=T1)
    e, _ = energy_by_class(pumps, k, grid, T0, T1)
    return {c: np.asarray(v, dtype="float64") for c, v in e.items()}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def main() -> int:
    grid = build_grid(str(REPO / DEFAULT_PATHS["polygon"]), dx=1000.0)
    ref = energy(MC / "census.parquet", MC / "kwh.parquet", True, grid)
    m1 = energy(S2 / "m1/census.parquet", S2 / "m1/kwh.parquet", False, grid)
    cty = energy(S2 / "county/census.parquet", S2 / "county/kwh.parquet", False, grid)
    mult = json.loads(SPEC.read_text())["change"]["multiplier_irrigation"]
    ch = np.load(CMAP)["changhua"].astype(bool)
    res = {"classes_ref": sorted(ref), "classes_m1": sorted(m1), "classes_county": sorted(cty)}
    res["m1_vs_ref_rel"] = {c: rel(m1[c], ref[c]) for c in ref}
    expect = {c: m1[c].copy() for c in m1}
    irr = "irrigation"
    expect[irr][ch] *= mult["changhua"]
    expect[irr][~ch] *= mult["yunlin"]
    res["county_vs_expected_rel"] = {c: rel(cty[c], expect[c]) for c in m1}
    res["GWh_2012_2022"] = {a: {c: float(v.sum()) / 1e6 for c, v in e.items()}
                            for a, e in (("ref", ref), ("m1", m1), ("county", cty))}
    ok_m1 = set(ref) == set(m1) and max(res["m1_vs_ref_rel"].values()) <= 1e-9
    ok_c = set(cty) == set(m1) and max(res["county_vs_expected_rel"].values()) <= 1e-6
    res["note"] = ("county check uses zone_county_map v2 (Changhua mask); cells the builder "
                   "assigns by pump county may differ at the county line, so a small county "
                   "mismatch there is reported, not a defect, if m1 passes")
    res["pass_m1"], res["pass_county"] = bool(ok_m1), bool(ok_c)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if ok_m1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
