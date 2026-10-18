"""Model-schema pump census for ladder rung R1: Yunlin (the AMP_V2 cache, unchanged) plus the
Changhua and Chiayi pumps of the 2026 TPC census, for ``calibrate_flow --pump-census/--pump-kwh``.

* Yunlin rows are the cached census the registered model was calibrated on, byte-for-byte in
  value (personal columns are never read; ``clean_census`` does not use them).
* Changhua and Chiayi come from the L0 parquet derivative through C0's ``harmonise`` (the same
  column mapping C0 validated); their ``sid`` is C0's ``uid`` (``changhua:<WELL_NO>``,
  ``chiayi:r<row>``) so no id can collide with a Yunlin well number.
* kWh rows for Changhua and Chiayi are restricted to 2012-01..2022-12: ``calibrate_flow`` and
  ``inputs.load_twin_inputs`` clean and aggregate only that window, so nothing changes, and
  the file then carries no post-origin month (the one exception, the December-2022 half of
  a phase-B December/January bill, is the registered model's own convention and is declared
  as an exemption in the manifest).

Checks (written to validation.json): unique sids; meters shared across counties; the model's
own cleaning and aggregation on the combined file reproduces the Yunlin-only forcing exactly
outside Changhua/Chiayi pumps' cells and adds C0's Changhua + Chiayi energy.

Run from the repository root through the CPU slot wrapper:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/i1/build_model_census.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from common import manifest  # noqa: E402

from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS  # noqa: E402
from hydrophysics.twin.grid import build_grid  # noqa: E402
from hydrophysics.twin.pumping import clean_census  # noqa: E402
from hydrophysics.twin.scenario import energy_by_class  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "c0lib", REPO / "dataprep_staging" / "c0_pump_census" / "lib.py")
c0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c0)

T0, T1 = "2012-01-01", "2023-01-01"
MODEL_COLS = ["sid", "PUMP_HP", "電號1", "PURPOSE", "TWD97_X", "TWD97_Y"]
AMP_CENSUS = REPO / DEFAULT_PATHS["pump_census"]
AMP_KWH = REPO / DEFAULT_PATHS["pump_kwh"]


def forcing(pumps: pd.DataFrame, kwh: pd.DataFrame, grid) -> dict[str, np.ndarray]:
    """The model's cleaned per-class (A, T) kWh, exactly as ``load_twin_inputs`` builds it."""
    p, k, _ = clean_census(pumps, kwh, cap_duty=1.0, t0=T0, t1=T1)
    by_cls, _ = energy_by_class(p, k, grid, T0, T1)
    return by_cls


def main() -> None:
    grid = build_grid(str(REPO / DEFAULT_PATHS["polygon"]), dx=1000.0)
    amp_pumps = pd.read_parquet(AMP_CENSUS, columns=c0.AMP_COLS)
    amp_pumps["sid"] = amp_pumps["sid"].astype(str)
    amp_kwh = pd.read_parquet(AMP_KWH, columns=["datetime", "electricity_kwh", "pump"])
    parts_p, parts_k = [amp_pumps.assign(county="yunlin")[MODEL_COLS + ["county"]]], [amp_kwh]
    meters = {"yunlin": set(amp_pumps["電號1"].astype(str))}
    for tag in ("changhua", "chiayi"):
        df = c0.harmonise(tag, c0.read_census_l0(tag), c0.load_grid())
        uid = df["uid"].to_numpy()
        mp = c0.model_pumps(df)
        mp["sid"] = uid
        mp["county"] = tag
        parts_p.append(mp[MODEL_COLS + ["county"]])
        k, _ = c0.read_kwh_l0(tag, uid, len(df))
        k = k[(k["datetime"] >= T0) & (k["datetime"] < T1)]
        parts_k.append(k)
        meters[tag] = set(mp["電號1"].astype(str))
    pumps = pd.concat(parts_p, ignore_index=True)
    # one type per column, applying exactly the coercions clean_census / _cell_lookup apply
    # themselves (pd.to_numeric(errors="coerce") for HP and coordinates, str for the meter
    # key and the sid), so the parquet round trip cannot change the forcing
    for col in ("PUMP_HP", "TWD97_X", "TWD97_Y"):
        pumps[col] = pd.to_numeric(pumps[col], errors="coerce").astype("float64")
    pumps["電號1"] = pumps["電號1"].astype(str)
    pumps["sid"] = pumps["sid"].astype(str)
    pumps["PURPOSE"] = pumps["PURPOSE"].astype(object).where(pumps["PURPOSE"].notna(), None)
    kwh = pd.concat(parts_k, ignore_index=True)
    kwh["pump"] = kwh["pump"].astype(str)
    if pumps["sid"].duplicated().any():
        raise SystemExit("sid not unique across counties")
    shared = {f"{a}-{b}": len((meters[a] & meters[b]) - {"nan", ""})
              for a, b in (("yunlin", "changhua"), ("yunlin", "chiayi"), ("changhua", "chiayi"))}

    E_all = forcing(pumps.drop(columns="county"), kwh, grid)
    E_yl = forcing(amp_pumps, amp_kwh, grid)
    tot_all = sum(E_all.values())
    tot_yl = sum(E_yl.values())
    diff = tot_all - tot_yl
    years = 11.0
    val = {
        "pumps": {c: int((pumps["county"] == c).sum()) for c in ("yunlin", "changhua", "chiayi")},
        "kwh_rows": int(len(kwh)),
        "meters_shared_across_counties": shared,
        "in_grid_GWh_per_yr_2012_2022": {"combined": float(tot_all.sum()) / 1e6 / years,
                                         "yunlin_only": float(tot_yl.sum()) / 1e6 / years,
                                         "added": float(diff.sum()) / 1e6 / years},
        "c0_reference_GWh_per_yr": {"yunlin": 188.46, "changhua": 187.41, "chiayi": 0.63},
        "min_added_cell_month_kwh": float(diff.min()),
        "classes": sorted(E_all),
    }
    # cells with no Changhua/Chiayi pump must carry exactly the Yunlin-only forcing
    ch_cells = np.zeros(grid.n_active, bool)
    from hydrophysics.twin.pumping import _cell_lookup  # read-only helper
    other = pumps[pumps["county"] != "yunlin"]
    cells = pd.Series(_cell_lookup(other, grid)).to_numpy(dtype="int64")
    ch_cells[cells[cells >= 0]] = True
    val["cells_with_changhua_or_chiayi_pumps"] = int(ch_cells.sum())
    val["max_abs_diff_outside_those_cells_kwh"] = float(np.abs(diff[~ch_cells]).max())
    ok = (val["max_abs_diff_outside_those_cells_kwh"] < 1e-6 and val["min_added_cell_month_kwh"]
          > -1e-6 and abs(val["in_grid_GWh_per_yr_2012_2022"]["added"] - 188.04) < 1.0)
    val["passed"] = bool(ok)
    print(json.dumps(val, indent=1, ensure_ascii=False))
    if not ok:
        raise SystemExit("validation failed; nothing written")

    d = manifest.new_version_dir("pump_electricity/model_census")
    pumps.to_parquet(d / "census.parquet", index=False)
    kwh.to_parquet(d / "kwh.parquet", index=False)
    # round trip: the model reading the saved files must get exactly E_all
    E_rt = forcing(pd.read_parquet(d / "census.parquet").drop(columns="county"),
                   pd.read_parquet(d / "kwh.parquet"), grid)
    rt = max(float(np.abs(E_rt[c] - E_all[c]).max()) for c in E_all)
    val["round_trip_max_abs_diff_kwh"] = rt
    if sorted(E_rt) != sorted(E_all) or rt > 1e-6:
        val["passed"] = False
        (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False))
        raise SystemExit(f"round trip changed the forcing (max {rt}); product not gated")
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False))
    l0 = c0.PARQ
    m = manifest.build_manifest(
        d, product="pump_electricity/model_census", layer="L1",
        inputs=[str(AMP_CENSUS), str(AMP_KWH)] + [str(l0 / f"{t}_{s}.parquet")
                                                  for t in ("changhua", "chiayi")
                                                  for s in ("census", "kwh_long")],
        code=[str(Path(__file__)), str(REPO / "dataprep_staging/c0_pump_census/lib.py")],
        parameters={"kwh_window_changhua_chiayi": [T0, T1], "yunlin": "AMP_V2 cache unchanged",
                    "sid_convention": "Yunlin WELL_NO; others C0 uid"},
        fit_window_end="2023-01", non_exempt_fit_window_end="2022-12",
        exemptions=[{
            "input": "phase-B (M1) bimonthly bills read in January 2023",
            "data_end": "2023-01",
            "reason": ("the December-2022 value of every phase-B meter is half of a "
                       "December-2022/January-2023 bill; this is the registered model's own "
                       "convention for Yunlin, kept so rung R1 differs from R0 only by the "
                       "added counties (about 0.1 % of 2012-22 energy)"),
            "ledger_number_id": "R1.model_census.dec2022_halfbill",
            "sensitivity": "C0 report: about 4.2 GWh, 15 % of in-grid December-2022 forcing",
            "uses": ["forcing_frozen", "model_design"]}],
        validation_report="validation.json",
        gate={"passed": True, "criteria": "validation.json passed (Yunlin unchanged, C0 energy added)"},
        allowed_uses=["forcing_frozen", "model_design", "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit"],
        personal_data="none (no name, address or postal code; raw meter numbers kept for "
                      "clean_census de-duplication, gitignored)")
    manifest.write_manifest(d, m)
    print("written", d)


if __name__ == "__main__":
    main()
