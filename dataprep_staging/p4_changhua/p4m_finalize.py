"""WP P4m: publish the two product versions from the staged results, with manifests,
validation reports, product cards and ledger rows.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_finalize

* ``results/dataprep/changhua_forcing_diagnostics/preorigin/v<N>/`` - items (1)-(3), (5) and
  the evidence of (2): pre-origin design-time evidence (allowed: model_design,
  design_time_evidence, after_the_fact_analysis), carrying C0 preorigin/v3's two exemptions.
* ``results/dataprep/changhua_forcing_diagnostics/recorded_arm/v<N>/`` - item (4): the
  2023-01..2025-06 recorded-forcing diagnostic (after_the_fact_analysis only; U4/U7).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

from p4_changhua import p4m_lib as L  # noqa: E402

ST = L.STAGE
CODE = ["dataprep_staging/p4_changhua/" + f for f in (
    "__init__.py", "engine.py", "engine_tlm.py", "profiles_v2.py", "inputs_p4.py",
    "p4m_lib.py", "p4m_run.py", "p4m_refit.py", "p4m_evidence.py", "p4m_recommend.py",
    "p4m_recorded.py", "p4m_finalize.py", "p4m_publish.py", "looks_p4m.py", "prereg_p4m.py")] + [
    "dataprep_staging/h1_heads_qc/tides.py", "dataprep_staging/h1_heads_qc/rawio.py",
    "dataprep_staging/harness/scorers.py", "dataprep_staging/common/guard.py"]
MODEL_INPUTS = [
    "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json",
    "results/twin_forward/datum_gate_innov.npz", "results/twin/basemap.npz",
    "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json",
    "AMP_V2/data/fan_stations.parquet", "AMP_V2/data/tpc_pumps.parquet",
    "AMP_V2/data/pump_kwh_all.parquet", "chou-shui-data/data/rf_timeseries.csv",
    "chou-shui-data/data/rf_stations.csv", "chou-shui-data/data/gw_stations.csv",
    "results/et/openmeteo_et0_2012_2022.npz"]
ENGINE_IN = ["results/dataprep/changhua_abstraction/engine/v2/MANIFEST.json",
             "results/dataprep/changhua_abstraction/engine/v2/engine_arrays.npz",
             "results/dataprep/changhua_abstraction/engine/v2/validation.json"]
C0_PRE_IN = ["results/dataprep/pump_electricity/preorigin/v3/" + f for f in (
    "MANIFEST.json", "cells_by_class.npz", "cells_total.npz", "county_class_monthly.parquet")]
C0_FULL_IN = ["results/dataprep/pump_electricity/full/v3/" + f for f in (
    "MANIFEST.json", "cells_by_class.npz")]
H1A_IN = ["results/dataprep/heads/L1_clean/v3/" + f for f in (
    "MANIFEST.json", "barometric_efficiency.csv", "heads_monthly.parquet")]
PILOT_TIDES = "results/dataprep/_pilots/head_pumping_signatures/out/tide_coherence.csv"
FX_IN = ["results/twin/forcing_experiment/forcing.npz",
         "results/twin/forcing_experiment/protocol.json",
         "results/twin/new_data_challenge/observations.csv"]
PREREGS = ["results/dataprep/preregistration/P4m.metered_diagnostics.v1.json",
           "results/dataprep/preregistration/P4m.recorded_arm_2023_25.v1.json"]
PRE_ALLOWED = ["model_design", "design_time_evidence", "after_the_fact_analysis"]
REC_ALLOWED = ["after_the_fact_analysis"]


def _forbidden(allowed):
    return sorted(set(USES) - set(allowed))


def _c0_exemptions() -> list[dict]:
    m = json.loads((REPO / C0_PRE_IN[0]).read_text())
    out = []
    names = {"C0.v3.exemption.census_attributes.preorigin": "census_attributes",
             "C0.v3.exemption.residual_unpaired_dec2022.preorigin": "residual_unpaired_dec2022"}
    for x in m["exemptions"]:
        short = names.get(x["ledger_number_id"], "other")
        out.append({"input": "inherited from pump_electricity/preorigin/v3: " + x["input"],
                    "data_end": x["data_end"], "reason": x["reason"],
                    "ledger_number_id": f"P4m.exemption.{short}",
                    "sensitivity": "C0's: " + x["sensitivity"] + " (forcing-level effects of "
                                   "at most 3.7e-4 of 2012-22 energy and 0.57-0.68 % of in-grid "
                                   "December 2022; they reach P4m's heads at the same order)",
                    "uses": ["model_design", "design_time_evidence"],
                    "c0_ledger_number_id": x["ledger_number_id"]})
    return out


def _copy(src: Path, d: Path, name: str | None = None) -> str:
    name = name or src.name
    shutil.copy2(src, d / name)
    return name


def preorigin_product(val: dict, card: str) -> Path:
    d = manifest.new_version_dir("changhua_forcing_diagnostics/preorigin")
    files = {ST / "run" / "results.json": "diagnostics_t1_t3.json",
             ST / "run" / "arrays.npz": "well_arrays.npz",
             ST / "run" / "scale_curves.npz": "scale_curves.npz",
             ST / "refit" / "results.json": "theta_refit_t2b.json",
             ST / "evidence" / "results.json": "evidence_e8_be_e9.json",
             ST / "evidence" / "e8_tides_preorigin.csv": "e8_tides_preorigin.csv",
             ST / "evidence" / "e9_nested_slopes.csv": "e9_nested_slopes.csv",
             ST / "recommendation" / "recommendation_preorigin.json":
                 "recommendation_preorigin.json"}
    for s, n in files.items():
        _copy(s, d, n)
    import numpy as np

    np.savez_compressed(d / "refit_predictions.npz", **{
        f"pred_refit_s{s}": np.load(ST / "refit" / f"pred_refit_s{s}.npy") for s in
        ("0.0", "0.5", "1.0")})
    L.jdump(val, d / "validation.json")
    (d / "PRODUCT_CARD.md").write_text(card)
    ex = _c0_exemptions()
    m = manifest.build_manifest(
        d, product="changhua_forcing_diagnostics/preorigin", layer="L3",
        inputs=MODEL_INPUTS + ENGINE_IN + C0_PRE_IN + H1A_IN + [PILOT_TIDES] + PREREGS,
        input_trees=["AMP_V2/data/wells"], code=CODE,
        parameters={"theta": "stage3_datum_gate_cwa (batch-8 merged cwa)",
                    "steps": "2012-02..2022-12", "scored": "2013-01..2022-12",
                    "min_months": L.MIN_N, "cluster_km": L.CLUSTER_KM, "seed": L.SEED,
                    "n_boot": L.N_BOOT, "scale_grid": list(L.S_GRID),
                    "refit": {"levels": [0, 0.5, 1], "n_iter": 15, "trust": 0.5,
                              "tol": 1e-6}},
        fit_window_end=max(["2022-12"] + [x["data_end"] for x in ex]),
        non_exempt_fit_window_end="2022-12", exemptions=ex,
        validation_report="validation.json",
        gate={"passed": bool(val["gate_passed"]),
              "criteria": "P4m.metered_diagnostics.v1: engine checks EC1-EC3 pass and every "
                          "declared diagnostic (T1, T2, T2a-d, T3, T5) computed"},
        allowed_uses=PRE_ALLOWED, forbidden_uses=_forbidden(PRE_ALLOWED),
        personal_data="none: public monitoring-well ids and cell aggregates only (no census "
                      "well numbers, pump coordinates, names or addresses)",
        extra={"preregistration": PREREGS[0], "wp": "P4m",
               "provisional": "no: built on batch 8's merged cwa theta (engine/v2)"})
    manifest.write_manifest(d, m)
    return d


def recorded_product(val: dict, card: str) -> Path:
    d = manifest.new_version_dir("changhua_forcing_diagnostics/recorded_arm")
    _copy(ST / "recorded" / "results.json", d, "recorded_arm_t4.json")
    _copy(ST / "recorded" / "predictions.npz", d, "recorded_arm_predictions.npz")
    L.jdump(val, d / "validation.json")
    (d / "PRODUCT_CARD.md").write_text(card)
    m = manifest.build_manifest(
        d, product="changhua_forcing_diagnostics/recorded_arm", layer="L3",
        inputs=MODEL_INPUTS + ENGINE_IN + C0_PRE_IN + C0_FULL_IN + FX_IN + PREREGS
        + ["results/dataprep/reports/P4m_evidence/stage/recommendation/"
           "recommendation_preorigin.json"],
        input_trees=["AMP_V2/data/wells"], code=CODE,
        parameters={"theta": "stage3_datum_gate_cwa (batch-8 merged cwa)",
                    "forecast": "2023-01..2025-06", "members": 1,
                    "protocol": "hydrophysics.twin.forcing_experiment on the sparse twin"},
        fit_window_end="2025-06", validation_report="validation.json",
        gate={"passed": bool(val["gate_passed"]),
              "criteria": "P4m.recorded_arm_2023_25.v1: numerical check (sparse vs torch <= "
                          "1e-5 m) and the recommendation written before the first score; "
                          "results are reported only (U4/U7)"},
        allowed_uses=REC_ALLOWED, forbidden_uses=_forbidden(REC_ALLOWED),
        personal_data="none: public monitoring-well ids only",
        extra={"preregistration": PREREGS[1], "wp": "P4m"})
    manifest.write_manifest(d, m)
    return d


def ledger_rows(numbers: list[tuple[str, str]], product_version: str,
                extra_sources: dict[str, str] | None = None, kind: str = "preorigin"):
    """Rows for each (number_id, heads_role): model inputs, engine, C0 and heads."""
    rows = []
    for nid, heads_role in numbers:
        srcs = {s: "input" for s in MODEL_INPUTS}
        srcs["results/dataprep/changhua_abstraction/engine/v2"] = "input"
        srcs["results/dataprep/pump_electricity/preorigin/v3"] = "input"
        srcs["AMP_V2/data/wells"] = heads_role
        if kind == "recorded":
            srcs["AMP_V2/data/wells"] = "input"
            srcs["results/dataprep/pump_electricity/full/v3"] = "input"
            srcs["results/twin/forcing_experiment/forcing.npz"] = "input"
            srcs["results/twin/new_data_challenge/observations.csv"] = "validator"
        if extra_sources:
            srcs.update(extra_sources)
        for s, role in srcs.items():
            rows.append({"number_id": nid, "source": s, "role": role,
                         "product_version": product_version, "note": ""})
    return rows


def main() -> int:  # pragma: no cover - driven by report_p4m
    raise SystemExit("use report_p4m.py, which builds validation, cards and calls this module")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["preorigin_product", "recorded_product", "ledger_rows", "ledger", "sha256_file",
           "rel_to_repo"]
