"""Build ``results/dataprep/head_signatures/v<N>`` from the stage outputs and the registered
test results (P5.signatures.v1).

Usage: s9_build_product.py WORKDIR EVIDENCE_DIR
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import guard, manifest  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402

from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402
from p5_head_signatures import p5stats as S  # noqa: E402

CODE = ["__init__.py", "config.py", "extractor.py", "inputs.py", "s1_extract.py",
        "s2_assemble.py", "s3_local_kwh.py", "s4_tf.py", "s5_eemd.py", "p5stats.py",
        "t_run.py", "prereg_p5.py", "looks_p5.py", "s9_build_product.py"]
H1_CODE = ["__init__.py", "pipeline.py", "screens.py", "baro.py", "rawio.py", "config.py",
           "assemble.py", "assemble_v2.py"]
COPY = ["well_month_features.parquet", "well_year_features.parquet",
        "nested_pair_months.parquet", "tidal_response.csv", "tf_gains.csv",
        "eemd_features.csv", "eof_modes.json", "regional_kwh.parquet",
        "extraction_summary.json", "local_kwh_info.json", "tf_inputs.json"]


def load_tests(ev: Path) -> dict:
    out = {}
    for f in sorted((ev / "tests").glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def seasonal_table(work: Path) -> pd.DataFrame:
    F = pd.read_parquet(work / "well_month_features.parquet")
    F = F[F["month"] <= C.ORIGIN]
    km = pd.read_parquet(work / "wells_kwh_meta.parquet")
    meta = I.station_meta()[["sid", "layer_flag"]]
    km = km.merge(meta, on="sid", how="left")
    w = km[km.county.isin(["Yunlin", "Changhua"]) & km.layer.isin(list(C.LAYERS_SHALLOW))
           & (km.layer_flag == 0) & (km.sectors3 >= C.SECTORS_MIN)]
    L = pd.read_parquet(work / "local_kwh.parquet")
    rows = []
    for sid in sorted(set(w.sid) & set(F.sid)):
        g = F[F.sid == sid].set_index("month")
        r = {"sid": sid}
        for idx in ("A_S1c_strain", "DN", "dr_med"):
            s = g[idx].dropna()
            r[f"cv_{idx}"] = S.climatology_cv(s) if len(s) >= C.WELL_MIN_MONTHS else np.nan
            r[f"n_months_{idx}"] = int(len(s))
        e = L[L.sid == sid].set_index("month")["E_r2"]
        r["cv_census_local_E_r2"] = S.climatology_cv(e)
        rows.append(r)
    T = pd.DataFrame(rows).merge(w[["sid", "county", "layer"]], on="sid")
    T["window"] = f"{C.PRE_MONTHS[0]}..{C.PRE_MONTHS[1]}"
    T["role"] = "validator only (plan 4.1: never a prior for alpha)"
    return T


def main():
    work, ev = Path(sys.argv[1]), Path(sys.argv[2])
    tests = load_tests(ev)
    g1a = tests["P5.t1a_repro"]["passed"]
    g1b = tests["P5.t1b_epoch"]["passed"]
    g2 = tests["P5.t2_baro"]["passed"]
    gate = bool(g1a and g1b and g2)
    d = manifest.new_version_dir(C.PRODUCT)          # exclusive mkdir of v<N+1>
    for f in COPY:
        shutil.copy2(work / f, d / f)
    L = pd.read_parquet(work / "local_kwh.parquet")
    L.to_parquet(d / "well_local_kwh.parquet", index=False)
    km = pd.read_parquet(work / "wells_kwh_meta.parquet")
    km.to_csv(d / "well_census_coverage.csv", index=False)
    seasonal_table(work).to_csv(d / "seasonal_cv.csv", index=False)
    bench = {k: v for k, v in tests.items() if k not in ("P5.t1a_repro", "P5.t1b_epoch",
                                                          "P5.t2_baro")}
    (d / "benchmarks.json").write_text(json.dumps(bench, indent=1, default=str))
    summ = json.loads((work / "extraction_summary.json").read_text())
    val = {"prereg": "P5.signatures.v1",
           "gate": {"passed": gate, "t1a_repro": g1a, "t1b_epoch": g1b, "t2_baro": g2},
           "construction_check_vs_L1_v3": summ["daily_check"],
           "tests": {k: tests[k] for k in ("P5.t1a_repro", "P5.t1b_epoch", "P5.t2_baro")},
           "claims_summary_file": "benchmarks.json"}
    (d / "validation.json").write_text(json.dumps(val, indent=1, default=str))
    card = (ev / "PRODUCT_CARD.md")
    if card.exists():
        shutil.copy2(card, d / "PRODUCT_CARD.md")
    l1 = C.L1_DIR
    inputs = [rel_to_repo(l1 / f) for f in (
        "MANIFEST.json", "heads_daily.parquet", "heads_monthly.parquet", "events.csv",
        "barometric_efficiency.csv", "pressure_hourly.parquet", "station_meta_clean.parquet")]
    inputs += [rel_to_repo(C.C0_CENSUS_DIR / f) for f in ("MANIFEST.json",
                                                          "census_pumps.parquet")]
    inputs += [rel_to_repo(C.C0_PRE_DIR / f) for f in (
        "MANIFEST.json", "pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz")]
    inputs += [rel_to_repo(C.ET0_DIR / "MANIFEST.json"),
               rel_to_repo(C.ET0_DIR / "et0_daily_2012_2022_preorigin.npz"),
               rel_to_repo(C.SW_PRE_DIR / "MANIFEST.json"),
               rel_to_repo(C.SW_PRE_DIR / "district_monthly.csv"),
               rel_to_repo(C.RAIN_CSV), rel_to_repo(C.RAIN_STATIONS),
               rel_to_repo(C.PILOT_DIR / "out" / "well_month_features.parquet"),
               "results/dataprep/preregistration/P5.signatures.v1.json"]
    trees = ["AMP_V2/data/wells", "data_fetch_api/new_measurements/heads_202210_202610"]
    code = [rel_to_repo(C.CODE_DIR / f) for f in CODE]
    code += [f"dataprep_staging/h1_heads_qc/{f}" for f in H1_CODE]
    code += ["dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py"]
    params = {k: getattr(C, k) for k in dir(C) if k.isupper() and k not in (
        "REPO", "CODE_DIR", "WORK", "EVIDENCE", "L1_DIR", "C0_PRE_DIR", "C0_CENSUS_DIR",
        "ET0_DIR", "SW_PRE_DIR", "UNITS_DIR", "PILOT_DIR", "RAIN_CSV", "RAIN_STATIONS")}
    params = json.loads(json.dumps(params, default=str))
    m = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=inputs, code=code, parameters=params,
        fit_window_end="2026-09", validation_report="validation.json",
        gate={"passed": gate, "criteria": "plan WP P5 acceptance as registered in "
                                          "P5.signatures.v1: t1a (extractor reproduces the "
                                          "pilot, Spearman >= 0.95), t1b (epoch check >= 0.95), "
                                          "t2 (H1 barometric acceptance: O1 coherence improves)"},
        allowed_uses=["validator", "after_the_fact_analysis", "design_time_evidence"],
        forbidden_uses=["prior", "prior_width_input", "calibration_target", "scoring_target",
                        "baseline", "datum_fit", "prospective_scorer", "initial_conditions",
                        "idw_field", "model_design", "selection", "forcing_frozen",
                        "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none: public monitoring-station ids and coordinates; census "
                      "electricity only as sums around stations (no pump id, coordinate, "
                      "name or address); some 1 km sums cover few meters, so the product "
                      "stays under gitignored results/",
        time_slicing={"column": "month", "params_fit_window_end": C.ORIGIN, "encoding": "iso"},
        input_trees=trees,
        extra={"role": "validators and benchmarks only (plan WP P5 'Role'; ledger rule 4.1: "
                       "daily-cycle indices and their seasonal CV may never be a prior, "
                       "including for alpha)",
               "provisional_inputs": "none (P5 does not use the stage-3 calibrated theta)"})
    errs = guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit(f"time slicing errors: {errs}")
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise SystemExit(f"manifest errors: {errs}")
    manifest.write_manifest(d, m)
    print("built", rel_to_repo(d), "gate", gate, "outputs", len(m["outputs"]),
          "repo", REPO.name)


if __name__ == "__main__":
    main()
