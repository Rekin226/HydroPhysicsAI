"""WP H4: package the registered run into results/dataprep/subgrid_operator/v<N>/.

Reads the registered evidence (reports/H4_evidence/registered/{scores.json, arrays.npz}),
rebuilds the operator's kernel weights (deterministic, no head data), writes the product files,
validation.json, PRODUCT_CARD.md and MANIFEST.json (through common.manifest). Never overwrites a
version directory. Usage: python b1_build_product.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

PRODUCT = "subgrid_operator"
EVID = DATAPREP / "reports/H4_evidence/registered"
PREREG = DATAPREP / "preregistration/H4.subgrid_operator.v1.json"
CODE = ["dataprep_staging/h4_subgrid/h4lib.py", "dataprep_staging/h4_subgrid/r1_registered.py",
        "dataprep_staging/h4_subgrid/prereg_h4.py",
        "dataprep_staging/h4_subgrid/b1_build_product.py",
        "dataprep_staging/h4_subgrid/d0_design_check.py",
        "dataprep_staging/h4_subgrid/t0_smoke.py"]
PASS_USES = ["model_design", "validator", "design_time_evidence", "after_the_fact_analysis"]
FAIL_USES = ["design_time_evidence", "after_the_fact_analysis"]
ALL_FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "calibration_target",
                 "prospective_scorer", "initial_conditions", "idw_field", "prior",
                 "prior_width_input", "forcing_frozen", "forcing_recorded_arm",
                 "statistical_layer_history", "selection"]


def inherited_exemptions() -> list[dict]:
    """The post-origin inputs of yunlin_electricity/preorigin/v2 and census_L1/v3 reach the
    operator through Q_j and the pump coordinates; their exemptions are inherited (uses
    restricted to this product's pre-origin uses)."""
    out: dict[str, dict] = {}
    for prod in (L.ELEC, L.CENSUS):
        m = json.loads((prod / "MANIFEST.json").read_text())
        for x in m["exemptions"]:
            src = f"{m['product']} {m['version']} exemption {x['ledger_number_id']}"
            if x["input"] in out:
                out[x["input"]]["inherited_from"] += f"; {src}"
                continue
            y = dict(x)
            y["uses"] = ["model_design", "design_time_evidence"]
            y["inherited_from"] = src
            name = x["ledger_number_id"].split(".exemption.")[-1].split(".")[0]
            y["ledger_number_id"] = f"H4.exemption.{name}"
            out[x["input"]] = y
    return list(out.values())


def main() -> None:
    sc = json.loads((EVID / "scores.json").read_text())
    z = np.load(EVID / "arrays.npz", allow_pickle=False)
    passed = bool(sc["gate_passed"])
    d = L.build_data(L.PRIMARY, with_heads=False)
    Gs, Gl, npump = L.operator_matrices(d)
    if not np.array_equal(npump, z["npump"]) or not np.array_equal(d.sids, z["sids"]):
        raise SystemExit("operator rebuild differs from the registered run")
    Sx, Lx = L.regressors(d, Gs, Gl)
    if not (np.allclose(Sx, z["S_unit"], rtol=1e-12, atol=1e-15)
            and np.allclose(Lx, z["L_unit"], rtol=1e-12, atol=1e-15)):
        raise SystemExit("regressor rebuild differs from the registered run")
    out = manifest.new_version_dir(PRODUCT)
    # 1. correction series (2012-02..2022-12; 2012-01 is the initial condition)
    dates = z["dates"]
    keep = dates >= L.PRIMARY.fit_start
    rows = []
    for i, s in enumerate(z["sids"]):
        rows.append(pd.DataFrame({
            "sid": s, "date": dates[keep], "county": z["county"][i], "layer": int(z["layer"][i]),
            "quartile_2012_19": int(z["quartile"][i]),
            "c_t1_m": z["corr_t1"][i, keep], "c_t2_m": z["corr_t2"][i, keep],
            "s_unit_m": z["S_unit"][i, keep], "l_unit_m": z["L_unit"][i, keep]}))
    cs = pd.concat(rows, ignore_index=True)
    cs.to_parquet(out / "correction_series.parquet", index=False)
    # 2. well table
    from scipy.spatial import cKDTree
    tree = cKDTree(d.pump_xy)
    n1 = np.array([len(tree.query_ball_point(x, 1000.0)) for x in d.xy])
    wt = pd.DataFrame({
        "sid": z["sids"], "county": z["county"], "layer": z["layer"], "cell": z["cell"],
        "T_m2_per_day": z["T"], "S": z["S"], "layer_point_weight": z["w"],
        "n_pumps_1km": n1, "n_pumps_2km": z["npump"], "sd_obs_2012_19_m": z["sd_fit"],
        "quartile_2012_19": z["quartile"], "n_fit_months": z["n_train"],
        "n_test_months": z["n_test"], "scored": z["score_ok"],
        "mse_base_2020_22": z["mse_base"], "mse_clim_2020_22": z["mse_clim"],
        "mse_t1_2020_22": z["mse_t1"], "mse_t2_2020_22": z["mse_t2"],
        "sd_c_t1_m": np.nanstd(z["corr_t1"][:, keep], axis=1),
        "mean_c_t1_m": np.nanmean(z["corr_t1"][:, keep], axis=1)})
    wt.to_parquet(out / "well_table.parquet", index=False)
    # 3. operator weights (unit multipliers), pumps indexed by row of the source arrays
    Gs = Gs.tocoo()
    Gl = Gl.tocsr()
    np.savez_compressed(
        out / "operator_weights.npz", well_index=Gs.row.astype(np.int32),
        pump_index=Gs.col.astype(np.int32),
        pump_source_row=d.pump_row[Gs.col].astype(np.int32),
        pump_source_county=d.pump_county[Gs.col], g_seasonal=Gs.data,
        g_slow=np.asarray(Gl[Gs.row, Gs.col]).ravel(), sids=d.sids,
        a_t1=np.asarray(sc["primary"]["t1"]["a"]),
        source_file=np.array(rel_to_repo(L.ELEC / "pumps_yunlin.npz")),
        source_sha256=np.array(sha256_file(L.ELEC / "pumps_yunlin.npz")))
    # 4. multipliers
    mult = {"t1_fit_2012_19_all_wells": {"a_seasonal": sc["primary"]["t1"]["a"][0],
                                          "a_slow": sc["primary"]["t1"]["a"][1]},
            "t2_by_fold": sc["primary"]["t2"]["a_by_fold"],
            "units": "dimensionless multipliers of the model-unit kernels (1 = the twin's "
                     "volume conversion with the zone T, S); spreads across folds are "
                     "uncalibrated",
            "sensitivity_t1": {k: v["t1"]["a"] for k, v in sc["sensitivity"].items()}}
    (out / "multipliers.json").write_text(json.dumps(mult, indent=1) + "\n")
    # 5. validation
    val = {"prereg": "H4.subgrid_operator.v1", "prereg_sha256": sc["prereg_sha256"],
           "gate_passed": passed, "criteria": sc["criteria"], "primary": sc["primary"],
           "placebo_reductions": {f: {k: v for k, v in sc["placebos"][f].items()
                                      if not k.startswith("a_")} for f in sc["placebos"]},
           "sensitivity": sc["sensitivity"],
           "t2_spatial_only_reported": sc["t2_spatial_only_reported"],
           "counts": {k: sc[k] for k in ("n_wells", "n_pumps", "wells_with_kernel",
                                         "wells_with_kernel_by_county", "quartile_cuts_sd_m",
                                         "quartile_counts_classified", "quartile_counts_scored",
                                         "q4_scored_by_county", "q4_scored_with_kernel",
                                         "n_scored", "n_fit_wells")},
           "post_registration_looks": sc["post_registration_looks"],
           "config": sc["config"]}
    (out / "validation.json").write_text(json.dumps(val, indent=1, default=float) + "\n")
    card = card_text(sc, passed)
    (out / "PRODUCT_CARD.md").write_text(card)
    ex = inherited_exemptions()
    inputs = [rel_to_repo(p) for p in (
        L.ENGINE / "MANIFEST.json", L.ENGINE / "engine_arrays.npz", L.UNITS / "MANIFEST.json",
        L.UNITS / "units.json", L.UNITS / "wells.csv", L.ELEC / "MANIFEST.json",
        L.ELEC / "pumps_yunlin.npz", L.CENSUS / "MANIFEST.json",
        L.CENSUS / "census_pumps.parquet", L.HEADS / "MANIFEST.json",
        L.HEADS / "heads_monthly.parquet", PREREG, EVID / "scores.json", EVID / "arrays.npz")]
    m = manifest.build_manifest(
        out, product=PRODUCT, layer="L3", inputs=inputs, code=CODE,
        parameters={"config": L.PRIMARY.as_dict(), "prereg": "H4.subgrid_operator.v1",
                    "multipliers_fit_window": [L.PRIMARY.fit_start, L.PRIMARY.fit_end],
                    "heads_read": "heads/L1_clean/v3 heads_monthly flags 0-1, window_end "
                                  "2022-12 (fit 2012-02..2019-12; score 2020-01..2022-12)",
                    "theta_sha256": "789d0e459a3b35eb3e16002869405c889a86131815fcebfef84a9ffd8370704c"},
        fit_window_end=max(x["data_end"] for x in ex), non_exempt_fit_window_end="2022-12",
        exemptions=ex, validation_report="validation.json",
        gate={"passed": passed, "criteria": "plan 3 WP H4 acceptance as registered in "
                                            "H4.subgrid_operator.v1 (A1-A8, all must pass)",
              "failed": [c["id"] for c in sc["criteria"] if not c["passed"]]},
        allowed_uses=PASS_USES if passed else FAIL_USES,
        forbidden_uses=ALL_FORBIDDEN + ([] if passed else ["model_design", "validator"]),
        personal_data="no names, addresses, coordinates or well numbers; operator_weights.npz "
                      "indexes census pumps by row of yunlin_electricity/preorigin/v2 "
                      "pumps_yunlin.npz and its kernel weights imply pump-to-monitoring-well "
                      "distances (quasi-locational); gitignored under results/")
    errs = manifest.validate_manifest(m, out)
    if errs:
        raise SystemExit("manifest invalid: " + "; ".join(errs))
    manifest.write_manifest(out, m)
    p = guard.open_product(out, "design_time_evidence")
    print(json.dumps({"product": rel_to_repo(out), "gate_passed": passed,
                      "guard_notes": p.notes}, indent=1))


def card_text(sc: dict, passed: bool) -> str:
    t1, t2 = sc["primary"]["t1"], sc["primary"]["t2"]
    lines = [f"# {PRODUCT} v1 (WP H4)", "",
             f"Gate: **{'PASSED' if passed else 'FAILED'}** (pre-registration "
             f"H4.subgrid_operator.v1; criteria A1-A8, all must pass).", "",
             "A data-driven observation operator for the 158 calibration wells: "
             "`h_well(t) = h_cell(t) - c(t)`, with `c` a periodic-Theis (seasonal) plus Thiem "
             "(slow) local-cone correction summed over kept Yunlin census pumps within 2 km, "
             "fitted multipliers (2012-02..2019-12), applied to the engine/v2 hindcast (cwa "
             "theta). No solver change. CPU only.", "",
             "## Files", "",
             "- `correction_series.parquet`: sid x month 2012-02..2022-12: `c_t1_m` "
             "(correction with the multipliers fitted on every well), `c_t2_m` (the well's "
             "site-fold multipliers), `s_unit_m`, `l_unit_m` (unit-multiplier seasonal and slow "
             "components). Subtract `c` from the cell head.",
             "- `well_table.parquet`: per well layer, T, S, layer point weight, pumps within 1 "
             "and 2 km, 2012-19 sd quartile, held-out MSEs (base, corrected T1/T2, "
             "climatology).",
             "- `operator_weights.npz`: unit-multiplier kernel weights per (well, pump), pumps "
             "indexed by row of `yunlin_electricity/preorigin/v2/pumps_yunlin.npz` (sha256 "
             "stored).",
             "- `multipliers.json`, `validation.json` (criteria, placebos, sensitivity).", "",
             "## Multipliers", "",
             f"- T1 (all wells, 2012-19): a_seasonal {t1['a'][0]:.3f}, a_slow {t1['a'][1]:.3f} "
             "(1 = the model-unit cone).",
             f"- T2 folds: {json.dumps(t2['a_by_fold'])} (spread uncalibrated).", "",
             "## Acceptance", ""]
    for c in sc["criteria"]:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} {c['id']}: {c['criterion']} "
                     f"({c['threshold']}): {c['measured']}")
    lines += ["", "## Known limits", "",
              "- The hindcast (theta calibrated on 2012-22 at the same wells) is in sample in "
              "2020-22: held-out years are held out for the operator's multipliers only.",
              "- Only Yunlin pumps enter (the hindcast carries no Changhua pumping); with the "
              "twin's pump split only Yunlin wells in model layers 0-1 receive a correction.",
              "- Model units: the kWh-to-volume conversion is the twin's (eta 0.5, head_extra "
              "40 m); a multiplier far above 1 is not a physical cone strength.",
              "- Quartiles are defined on 2012-19 observed sd (not the audit's 2023-26 "
              "quartiles).",
              "", "## Uses", "",
              ("Allowed: " + ", ".join(PASS_USES if passed else FAIL_USES)
               + (". Refit the multipliers whenever the hindcast changes (I1 recalibration)."
                  if passed else ". The operator failed its registered acceptance: it must not "
                                 "enter I1 as a secondary model; plan 7.3 applies.")), ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
