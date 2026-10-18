"""Build ``results/dataprep/natural_experiments/v<N>/`` (plan WP N1 outputs).

Files (every csv carries ``month``, the data month a row refers to, for the time slice):

* ``interventions.csv``: the sourced intervention table (``interventions.build``).
* ``public_well_janmay.csv``: Jan-May public-well pumping by county, agency and township,
  2011-2023, one value per year (printed where printed, else the two-read chart value).
* ``public_well_chart_reads.csv``: every chart point with both model reads.
* ``drought_wells.csv``: the Changhua and Yunlin drought-well tables of the 2022 inventory.
* ``jiji_janmay_supply.csv``: the Jiji weir Jan-May supply table (110 briefing).
* ``census_janmay_kwh.csv``: Jan-May in-grid census kWh by county and class (C0 preorigin/v3).
* ``tests.json``: every registered N1 test and check with its status and verdicts.
* ``cpu_tests/``: results.json and per-well tables of the CPU tests (after their registration).
* ``validation.json``, ``README.md``.

Usage: python -I build_product.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402

from n1_natural_experiments import census_events as CE  # noqa: E402
from n1_natural_experiments import interventions as IV  # noqa: E402
from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import parse_sources as P  # noqa: E402
from n1_natural_experiments import units_n1 as U  # noqa: E402

CODE = ["__init__.py", "n1const.py", "parse_sources.py", "transcribe_charts.py",
        "merge_charts.py", "census_events.py", "interventions.py", "units_n1.py", "n1lib.py",
        "prereg_n1.py", "run_cpu_tests.py", "build_product.py"]
VINTAGE = {"moea_annex": "2019-10", "cy53001": "2015-05", "brief110": "2021 survey report",
           "brief112": "2023 survey report", "drought_wells": "2023-04"}


def tests_json(cpu: dict) -> dict:
    out = {"registrations": [], "tests": []}
    for pid in ("N1.cpu_response.v1", "N1.phase12_response.v1", "N1.designtime_checks.v1"):
        p = PREREG_DIR / f"{pid}.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        out["registrations"].append({"id": pid, "kind": doc["kind"], "created": doc["created"],
                                     "sha256": sha256_file(p), "path": rel_to_repo(p)})
        if isinstance(doc.get("tests"), list):
            for t in doc["tests"]:
                tid = t["test_id"]
                rec = {"test_id": tid, "prereg": pid, "fresh_units": t["fresh_units"],
                       "label": t.get("label", ""), "what": t.get("what", "")}
                if tid in cpu.get("tests", {}):
                    rec["status"] = "run (CPU, after registration)"
                    rec["result"] = cpu["tests"][tid]
                elif pid == "N1.phase12_response.v1":
                    rec["status"] = ("registered; runs in Phase 1-2 (needs the --mask-months "
                                     "hook, GPU refits and the I1 primary model)")
                out["tests"].append(rec)
        else:
            for k, v in (doc.get("checks") or {}).items():
                rec = {"test_id": k, "prereg": pid, "label": "declared check", "what": v}
                ck = {"C1_insample": "C1_insample", "C2_fold_theta": "C2_fold_theta"}.get(k)
                if ck and ck in cpu.get("checks", {}):
                    rec["status"] = "run (CPU, after declaration)"
                    rec["result"] = cpu["checks"][ck]
                else:
                    rec["status"] = "declared; Phase 1-2 or pending data"
                out["tests"].append(rec)
    return out


def main() -> int:
    cpu_dir = K.EVID / "cpu_tests"
    cpu = json.loads((cpu_dir / "results.json").read_text(encoding="utf-8"))
    tab, checks, chart_reads, series, dw = IV.build()
    tab["month"] = tab["date_end"]
    src_key = {rel_to_repo(v): k for k, v in K.SRC.items()}
    tab["vintage"] = tab["source"].map(lambda s: VINTAGE.get(src_key.get(s.split(" ")[0]), ""))
    tab.loc[tab.extraction.eq("product"), "vintage"] = "pre-origin product"
    d = manifest.new_version_dir(K.PRODUCT)
    tab.to_csv(d / "interventions.csv", index=False)
    s = series.copy()
    s.to_csv(d / "public_well_janmay.csv", index=False)
    cr = chart_reads.copy()
    cr["month"] = cr["year"].astype(str) + "-05"
    cr.to_csv(d / "public_well_chart_reads.csv", index=False)
    dwf = pd.DataFrame(dw["rows"])
    dwf["cell"] = U.cell_index(dwf.x_twd97.to_numpy(), dwf.y_twd97.to_numpy())
    dwf["month"] = "2022-09"
    dwf["vintage"] = VINTAGE["drought_wells"]
    dwf.to_csv(d / "drought_wells.csv", index=False)
    jj = pd.DataFrame(P.jiji_janmay_table()["rows"])
    jj["month"] = jj["year"].astype(str) + "-05"
    jj.to_csv(d / "jiji_janmay_supply.csv", index=False)
    jm, notes = CE.janmay_kwh()
    jm["month"] = jm["year"].astype(str) + "-05"
    jm.to_csv(d / "census_janmay_kwh.csv", index=False)
    (d / "tests.json").write_text(json.dumps(tests_json(cpu), ensure_ascii=False, indent=1,
                                             default=float) + "\n")
    shutil.copytree(cpu_dir, d / "cpu_tests")
    val = {"wp": K.WP, "intervention_rows": int(len(tab)),
           "every_row_has_source": checks["every_row_has_source"],
           "every_row_verified": checks["every_row_verified"],
           "quotes_found": f"{checks['quotes_found']}/{checks['quotes_total']}",
           "charts": {k: v for k, v in checks["charts"].items() if k != "pages"},
           "drought_wells": checks["drought_wells"], "census_guard_notes": notes,
           "registrations_before_scoring": cpu["prereg"],
           "engine_reproduction_max_abs_m": cpu["engine_reproduction_max_abs_m"]}
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                  default=float) + "\n")
    (d / "README.md").write_text(README, encoding="utf-8")
    inputs = [rel_to_repo(p) for p in K.SRC.values()]
    inputs += [rel_to_repo(p) for p in sorted(K.API_CACHE.glob("*.json"))]
    inputs += [rel_to_repo(x) for x in (
        K.C0_PRE / "county_class_monthly.parquet", K.C0_PRE / "MANIFEST.json",
        K.HEADS_DIR / "heads_monthly.parquet", K.HEADS_DIR / "station_meta_preorigin.parquet",
        K.HEADS_DIR / "MANIFEST.json", K.ENGINE_DIR / "MANIFEST.json",
        K.ENGINE_DIR / "engine_arrays.npz", K.TWIN_THETA, K.TWIN_FOLD_THETAS,
        K.TWIN_KFOLD_WELLS, U.TWIN_WELLS, K.PILES_PRINTED / "printed_monthly.parquet",
        PREREG_DIR / "N1.cpu_response.v1.json", PREREG_DIR / "N1.phase12_response.v1.json",
        PREREG_DIR / "N1.designtime_checks.v1.json")]
    m = manifest.build_manifest(
        d, product=K.PRODUCT, layer="L1", inputs=inputs,
        code=[rel_to_repo(K.CODE_DIR / c) for c in CODE],
        parameters={"events": K.EVENTS, "reference_years": K.REFERENCE_YEARS,
                    "chart_agree_frac_axis": 0.02, "models": [K.MODEL_PDF, K.MODEL_IMG],
                    "boot": {"B": K.BOOT_B, "seed": K.BOOT_SEED, "level": K.CI_LEVEL}},
        fit_window_end="2024-12", validation_report="validation.json",
        gate={"passed": bool(checks["every_row_has_source"] and checks["every_row_verified"]),
              "criteria": "plan WP N1 acceptance: every row of the table has a source (and here "
                          "a verified quote, parsed table, two agreeing chart reads or a gated "
                          "product); every test is pre-registered before it runs "
                          "(harness.prereg3); results are reported whichever way they fall"},
        time_slicing={"column": "month", "params_fit_window_end": "2022-12", "encoding": "iso"},
        allowed_uses=["validator", "after_the_fact_analysis", "design_time_evidence"],
        forbidden_uses=["scoring_target", "calibration_target", "datum_fit", "baseline",
                        "prior", "prior_width_input", "forcing_frozen", "forcing_recorded_arm",
                        "model_design", "selection", "initial_conditions", "idw_field",
                        "prospective_scorer"],
        personal_data="none (agency wells and monitoring stations only; no person names or "
                      "addresses; contact columns never read)")
    errs = guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit(f"time slicing: {errs}")
    manifest.write_manifest(d, m)
    print(f"wrote {rel_to_repo(d)}")
    return 0


README = """# natural_experiments (WP N1)

Plan WP N1 outputs: the sourced intervention table and the registered response tests.

* interventions.csv: one row per intervention or quantity, with source file, page, quote (as
  printed), extraction (text_layer / text_layer_table / chart_two_reads / product),
  verification, pre_origin, the tests that use it and whether the effect is identifiable.
  'basis' separates agency-reported results from plans (the Control Yuan report of 2015 states
  schedules) and measured product values.
* public_well_janmay.csv, public_well_chart_reads.csv: Jan-May public-well pumping (1e4 m3,
  萬噸) read off the WRA 110 and 112 briefing charts by two models (claude-opus-5-5 on the PDF
  page, claude-sonnet-5-5 on the rendered image); every point agreed within 2 % of the axis;
  the 2020-21 county values are the printed numbers. sd = digitization sd (uncalibrated).
* drought_wells.csv: the 2022 drought-well inventory (published 2023-04), Changhua 49 and
  Yunlin 520 printed rows (the table skips number 280), TWD97 coordinates, active cell.
* jiji_janmay_supply.csv, census_janmay_kwh.csv: delivery cuts and the metered pumping
  response (model-unit kWh).
* tests.json, cpu_tests/: registrations, statuses and the CPU results.

Time slice: 'month' is the data month. Pre-origin uses (design_time_evidence) read rows up to
2022-12. Vintage: the drought-well coordinates and the 2022 public-well values were published
after the origin (2023-2024); model_design is forbidden until a version lists them as U3
exemptions. No 2023+ head, ring, pile, GNSS or InSAR value is in any file.
"""

if __name__ == "__main__":
    raise SystemExit(main())
