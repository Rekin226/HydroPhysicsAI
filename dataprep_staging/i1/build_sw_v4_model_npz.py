"""Surface-water v4 (W1, pre-origin) in the npz layout ``inputs.load_sw_recharge`` reads, for
ladder rung R1b (``calibrate_flow --sw-recharge <npz> --sw-components both``).

* ``sw_m_per_day`` = W1's ``grid_sw_v4.npz`` ``value``: the pre-origin monthly canal deliveries
  (Denton-benchmarked, supply-limited indicator) spread with v3's farmland allocation.
* ``perc_m_per_day`` = v3's paddy flood-percolation potential (``surface_water_v3.npz``),
  unchanged: W1 did not rebuild it, and it shares v3's LUIMAP-2024 allocation, the exemption
  W1's pre-origin manifest already declares.

The recharge fractions are learned by the calibration, so only timing and spatial pattern
matter; v4's level is about 0.74 of v3's in every year.

Run from the repository root:
    python dataprep_staging/i1/build_sw_v4_model_npz.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from common import manifest  # noqa: E402

W1 = REPO / "results/dataprep/surface_water_v4/preorigin/v2"
V3 = REPO / "results/twin/surface_water_v3.npz"


def main() -> None:
    w = np.load(W1 / "grid_sw_v4.npz", allow_pickle=True)
    v3 = np.load(V3, allow_pickle=True)
    dates_w = [str(d)[:10] for d in w["dates"]]
    dates_3 = [str(d)[:10] for d in v3["dates"]]
    if dates_w != dates_3:
        raise SystemExit("v4 and v3 month axes differ")
    sw = np.asarray(w["value"], dtype="float64")
    perc = np.asarray(v3["perc_m_per_day"], dtype="float64")
    if sw.shape != perc.shape or not (np.isfinite(sw).all() and (sw >= 0).all()):
        raise SystemExit("v4 field is not finite and non-negative on the v3 shape")
    if int(v3["n_active"]) != sw.shape[0] or abs(float(v3["dx"]) - float(w["dx"])) > 1e-6:
        raise SystemExit("grid mismatch between v3 and v4")
    d = manifest.new_version_dir("surface_water_v4/model_npz")
    out = d / "surface_water_v4_model.npz"
    np.savez(out, sw_m_per_day=sw, perc_m_per_day=perc, dates=np.array(dates_w),
             dx=np.float64(w["dx"]), n_active=np.int64(sw.shape[0]),
             sw_version=np.array("v4-preorigin-v2 deliveries + v3 percolation"),
             source=np.array("results/dataprep/surface_water_v4/preorigin/v2/grid_sw_v4.npz "
                             "(value) + results/twin/surface_water_v3.npz (perc_m_per_day)"))
    ratio = {str(y): float(sw[:, [i for i, t in enumerate(dates_w) if t.startswith(str(y))]]
                           .mean() / np.asarray(v3["sw_m_per_day"])[:, [
                               i for i, t in enumerate(dates_w) if t.startswith(str(y))]].mean())
             for y in range(2012, 2023)}
    val = {"passed": True, "shape": list(sw.shape), "months": [dates_w[0], dates_w[-1]],
           "v4_over_v3_annual_mean_ratio": ratio,
           "sw_mean_m_per_day": float(sw.mean()), "perc_mean_m_per_day": float(perc.mean())}
    (d / "validation.json").write_text(json.dumps(val, indent=1))
    m = manifest.build_manifest(
        d, product="surface_water_v4/model_npz", layer="L3",
        inputs=[str(W1 / "grid_sw_v4.npz"), str(W1 / "MANIFEST.json"), str(V3)],
        code=[str(Path(__file__))],
        parameters={"deliveries": "W1 preorigin/v2 value", "percolation": "v3 perc_m_per_day"},
        fit_window_end="2024-12", non_exempt_fit_window_end="2022-12",
        exemptions=[{
            "input": "results/twin/surface_water_v3.npz (LUIMAP 2024 farmland allocation)",
            "data_end": "2024-12",
            "reason": "plan U3 standard exemption: no pre-origin land-use snapshot exists; the "
                      "same exemption is declared in W1's pre-origin manifest",
            "ledger_number_id": "W1.preorigin.luimap_sensitivity",
            "sensitivity": "W1 report: county-area spreading gives cell-depth correlation 0.382; "
                           "district series unaffected",
            "uses": ["forcing_frozen", "model_design"]}],
        validation_report="validation.json",
        gate={"passed": True, "criteria": "shape, finiteness and grid match; W1 preorigin/v2 gate passed (design-time evidence, weak)"},
        allowed_uses=["forcing_frozen", "model_design", "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit"])
    manifest.write_manifest(d, m)
    print(json.dumps(val, indent=1), "\nwritten", out)


if __name__ == "__main__":
    main()
