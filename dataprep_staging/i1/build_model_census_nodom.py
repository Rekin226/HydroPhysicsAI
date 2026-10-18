"""Model-schema pump census without the domestic class, for ladder rung R1d-dom.

Rung R1c failed only its Changhua head gate; one suspect is the domestic class (家庭用水 and
the other ``scenario.purpose_class`` == "domestic" purposes), 23 % of Changhua's in-grid
metered energy against 5.5 % in Yunlin, with a summer peak like household cooling. The arm
removes the domestic class in every county (one rule, not a Changhua-only choice).

Exactness. The model counts each meter once and splits its energy over the meter's pumps by
horsepower (``pumping.clean_census``). Deleting the domestic pumps of a meter shared with
other classes would hand their share to the remaining pumps. Instead the remaining pumps'
kWh series on such a meter are scaled by ``hp_other / hp_meter`` before the domestic pumps are
removed: the meter's energy is then exactly the non-domestic pumps' share, and its duty
statistic, ``(E * hp_other / hp_meter) / (hp_other * cap)``, equals the original one, so every
duty-cap decision is unchanged. The result must equal R1c's forcing (model_census/v1 through
the model's own cleaning) with the domestic class removed, to rounding; validation.json
records the check.

Run from the repository root through the CPU slot wrapper:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/i1/build_model_census_nodom.py
"""
from __future__ import annotations

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
from hydrophysics.twin.pumping import KW_PER_HP, clean_census  # noqa: E402
from hydrophysics.twin.scenario import energy_by_class, purpose_class  # noqa: E402

V1 = REPO / "results/dataprep/pump_electricity/model_census/v1"
T0, T1 = "2012-01-01", "2023-01-01"
DROP = "domestic"


def forcing(pumps: pd.DataFrame, kwh: pd.DataFrame, grid) -> dict[str, np.ndarray]:
    p, k, _ = clean_census(pumps, kwh, cap_duty=1.0, t0=T0, t1=T1)
    return energy_by_class(p, k, grid, T0, T1)[0]


def main() -> None:
    grid = build_grid(str(REPO / DEFAULT_PATHS["polygon"]), dx=1000.0)
    pumps = pd.read_parquet(V1 / "census.parquet")
    kwh = pd.read_parquet(V1 / "kwh.parquet")
    pumps["sid"] = pumps["sid"].astype(str)
    kwh["pump"] = kwh["pump"].astype(str)
    cls = pumps["PURPOSE"].map(purpose_class)
    hp = pd.to_numeric(pumps["PUMP_HP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # meter key exactly as clean_census builds it
    mstr = pumps["電號1"].astype(str)
    meter = mstr.where(pumps["電號1"].notna() & (mstr != "nan") & (mstr != ""), pumps["sid"])
    is_dom = (cls == DROP).to_numpy()
    hp_meter = hp.groupby(meter).transform("sum")
    hp_other = hp.where(~is_dom, 0.0).groupby(meter).transform("sum")
    n_dom_on_meter = pd.Series(is_dom.astype(int)).groupby(meter.to_numpy()).transform("sum")
    mixed = (~is_dom) & (n_dom_on_meter.to_numpy() > 0)
    # HP unknown on a whole meter: clean_census splits evenly, so scale by the pump count
    n_meter = pd.Series(1, index=pumps.index).groupby(meter).transform("sum")
    n_other = pd.Series((~is_dom).astype(int), index=pumps.index).groupby(meter).transform("sum")
    frac = np.where(hp_meter.to_numpy() > 0,
                    hp_other.to_numpy() / np.where(hp_meter.to_numpy() > 0, hp_meter.to_numpy(), 1.0),
                    n_other.to_numpy() / n_meter.to_numpy())
    scale = pd.Series(np.where(mixed, frac, 1.0), index=pumps["sid"].to_numpy())

    keep = pumps[~is_dom].copy()
    k2 = kwh[kwh["pump"].isin(set(keep["sid"]))].copy()
    k2["electricity_kwh"] = k2["electricity_kwh"] * k2["pump"].map(scale).to_numpy()

    E_v1 = forcing(pumps, kwh, grid)
    E_nd = forcing(keep, k2, grid)
    diffs = {c: float(np.abs(E_nd.get(c, 0.0) - E_v1[c]).max()) for c in E_v1 if c != DROP}
    rel = max(diffs[c] / max(float(np.abs(E_v1[c]).max()), 1.0) for c in diffs)
    years = 11.0
    val = {
        "rule": "domestic class removed in every county; mixed meters scaled by hp_other/hp_meter",
        "pumps_removed": int(is_dom.sum()),
        "mixed_meter_pumps_scaled": int(mixed.sum()),
        "domestic_GWh_per_yr_in_grid_v1": float(E_v1.get(DROP, 0.0).sum()) / 1e6 / years,
        "total_GWh_per_yr_in_grid_v1": float(sum(E_v1.values()).sum()) / 1e6 / years,
        "total_GWh_per_yr_in_grid_nodom": float(sum(E_nd.values()).sum()) / 1e6 / years,
        "domestic_class_left": DROP in E_nd and float(E_nd[DROP].sum()) > 0.0,
        "max_abs_diff_other_classes_kwh": diffs,
        "max_rel_diff_other_classes": rel,
        "kw_per_hp": KW_PER_HP,
    }
    val["passed"] = bool(rel < 1e-9 and not val["domestic_class_left"])
    print(json.dumps(val, indent=1, ensure_ascii=False))
    if not val["passed"]:
        raise SystemExit("validation failed; nothing written")
    d = manifest.new_version_dir("pump_electricity/model_census")
    keep.to_parquet(d / "census.parquet", index=False)
    k2.to_parquet(d / "kwh.parquet", index=False)
    E_rt = forcing(pd.read_parquet(d / "census.parquet"), pd.read_parquet(d / "kwh.parquet"), grid)
    val["round_trip_max_abs_diff_kwh"] = max(float(np.abs(E_rt[c] - E_nd[c]).max()) for c in E_nd)
    val["passed"] = val["passed"] and val["round_trip_max_abs_diff_kwh"] < 1e-6
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False))
    if not val["passed"]:
        raise SystemExit("round trip changed the forcing; product not gated")
    m = manifest.build_manifest(
        d, product="pump_electricity/model_census", layer="L1",
        inputs=[str(V1 / "census.parquet"), str(V1 / "kwh.parquet"), str(V1 / "MANIFEST.json")],
        code=[str(Path(__file__))],
        parameters={"arm": "no_domestic", "rule": val["rule"], "base": "model_census/v1"},
        fit_window_end="2023-01", non_exempt_fit_window_end="2022-12",
        exemptions=[{
            "input": "phase-B (M1) bimonthly bills read in January 2023 (inherited from v1)",
            "data_end": "2023-01",
            "reason": "the registered model's December-2022 half-bill convention, kept so the "
                      "rung differs from R1c only by the domestic class",
            "ledger_number_id": "R1.model_census.dec2022_halfbill",
            "sensitivity": "C0 report: about 4.2 GWh, 15 % of in-grid December-2022 forcing",
            "uses": ["forcing_frozen", "model_design"]}],
        validation_report="validation.json",
        gate={"passed": True, "criteria": "R1c forcing minus the domestic class, exactly"},
        allowed_uses=["forcing_frozen", "model_design", "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit"],
        personal_data="none (no name, address or postal code; meter numbers kept for "
                      "de-duplication, gitignored)")
    manifest.write_manifest(d, m)
    print("written", d)


if __name__ == "__main__":
    main()
