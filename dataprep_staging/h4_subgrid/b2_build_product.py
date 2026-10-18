"""WP H4 repair round: package the v2 registered run into results/dataprep/subgrid_operator/v<N>/
and write the product-level SUPERSEDED.json (v1 superseded by v2).

Reads reports/H4_evidence/repair/registered/{scores.json, arrays.npz}, rebuilds the sub-grid
operator weights (deterministic, no head data) and checks them against the registered run,
writes the product files, validation.json, PRODUCT_CARD.md and MANIFEST.json (common.manifest).
Never overwrites a version directory. Usage: python b2_build_product.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import h4v2lib as V  # noqa: E402
import h4v2run as RN  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

PRODUCT = "subgrid_operator"
EVID = DATAPREP / "reports/H4_evidence/repair/registered"
DT = DATAPREP / "reports/H4_evidence/repair/design_time"
PREREG = DATAPREP / "preregistration/H4.subgrid_operator.v2.json"
THETA = DATAPREP.parent / "twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
THETA_SHA = "789d0e459a3b35eb3e16002869405c889a86131815fcebfef84a9ffd8370704c"
CODE = ["dataprep_staging/h4_subgrid/h4lib.py", "dataprep_staging/h4_subgrid/h4v2lib.py",
        "dataprep_staging/h4_subgrid/h4v2run.py", "dataprep_staging/h4_subgrid/r2_registered.py",
        "dataprep_staging/h4_subgrid/prereg_h4v2.py",
        "dataprep_staging/h4_subgrid/b2_build_product.py",
        "dataprep_staging/h4_subgrid/d1_design_v2.py",
        "dataprep_staging/h4_subgrid/t1_power_v2.py",
        "dataprep_staging/h4_subgrid/looks_h4v2.py",
        "dataprep_staging/h4_subgrid/ledger_h4v2.py"]
PASS_USES = ["model_design", "validator", "design_time_evidence", "after_the_fact_analysis"]
FAIL_USES = ["design_time_evidence", "after_the_fact_analysis"]
ALL_FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "calibration_target",
                 "prospective_scorer", "initial_conditions", "idw_field", "prior",
                 "prior_width_input", "forcing_frozen", "forcing_recorded_arm",
                 "statistical_layer_history", "selection"]


def inherited_exemptions() -> list[dict]:
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
            y["ledger_number_id"] = f"H4v2.exemption.{name}"
            out[x["input"]] = y
    return list(out.values())


def main() -> None:
    sc = json.loads((EVID / "scores.json").read_text())
    z = np.load(EVID / "arrays.npz", allow_pickle=False)
    passed = bool(sc["gate_passed"])
    if sha256_file(THETA) != THETA_SHA:
        raise SystemExit("stage3_theta.json changed since the registered run: rebuild needed")
    cfg = V.PRIMARY
    grid, cal, fr, _ = RN.setup(cfg, with_heads=False)
    tabs = V.make_tables(cfg, grid)
    Gc_s, Gc_l, nc = V.operator_matrices(cal, grid, tabs)
    Gf_s, Gf_l, nf = V.operator_matrices(fr, grid, tabs)
    if not (np.array_equal(nc, z["cal_npump"]) and np.array_equal(nf, z["f_npump"])
            and np.array_equal(cal.sids, z["cal_sids"]) and np.array_equal(fr.sids, z["f_sids"])):
        raise SystemExit("operator rebuild differs from the registered run")
    for G, Q, key in ((Gc_s, cal.qs, "cal_S_unit"), (Gc_l, cal.Qbar, "cal_L_unit"),
                      (Gf_s, cal.qs, "f_S_unit"), (Gf_l, cal.Qbar, "f_L_unit")):
        if not np.allclose(np.asarray(G @ Q), z[key], rtol=1e-12, atol=1e-15):
            raise SystemExit(f"regressor rebuild differs from the registered run ({key})")
    out = manifest.new_version_dir(PRODUCT)
    ver = out.name
    dates = z["dates"]
    keep = dates >= cfg.fit_start
    a = np.asarray(sc["a"], float)
    folds = np.array([sc["t2"]["a_by_fold"][k] for k in sorted(sc["t2"]["a_by_fold"])])

    def series(prefix, sids, county, layer, S, Lx, setname):
        rows = []
        for i, s in enumerate(sids):
            c = a[0] * S[i] + a[1] * Lx[i]
            cf = folds[:, :1] * S[i][None, :] + folds[:, 1:2] * Lx[i][None, :]
            rows.append(pd.DataFrame({
                "sid": s, "date": dates[keep], "well_set": setname, "county": county[i],
                "layer": int(layer[i]), "c_m": c[keep],
                "sd_m": cf.std(axis=0, ddof=1)[keep],
                "s_unit_m": S[i, keep], "l_unit_m": Lx[i, keep],
                "flag": 4 if np.any(S[i] != 0) else 0,
                "method": "h4v2_subgrid_cellmean"}))
        return rows
    cs = pd.concat(series("cal", z["cal_sids"], z["cal_county"], z["cal_layer"],
                          z["cal_S_unit"], z["cal_L_unit"], "calibration")
                   + series("f", z["f_sids"], z["f_county"], z["f_layer"], z["f_S_unit"],
                            z["f_L_unit"], "heldout"), ignore_index=True)
    cs.to_parquet(out / "correction_series.parquet", index=False)
    wt = pd.concat([
        pd.DataFrame({"sid": z["cal_sids"], "well_set": "calibration", "county": z["cal_county"],
                      "layer": z["cal_layer"], "cell": z["cal_cell"], "T_m2_per_day": z["cal_T"],
                      "S": z["cal_S"], "layer_point_weight": z["cal_w"],
                      "n_pumps_2km": z["cal_npump"], "sd_obs_m": z["cal_sd_fit"],
                      "sd_window": "2012-02..2019-12", "quartile": z["cal_quartile"],
                      "scored": z["cal_score_ok"], "mse_base": z["cal_mse_base"],
                      "mse_corrected": z["cal_mse_t1"], "mse_corrected_t2": z["cal_mse_t2"],
                      "mse_clim": z["cal_mse_clim"], "score_window": "2020-01..2022-12"}),
        pd.DataFrame({"sid": z["f_sids"], "well_set": "heldout", "county": z["f_county"],
                      "layer": z["f_layer"], "cell": z["f_cell"], "T_m2_per_day": z["f_T"],
                      "S": z["f_S"], "layer_point_weight": z["f_w"],
                      "n_pumps_2km": z["f_npump"], "sd_obs_m": z["f_sd"],
                      "sd_window": "2012-02..2022-12", "quartile": z["f_quartile"],
                      "scored": z["f_scored"], "mse_base": z["f_mse_base"],
                      "mse_corrected": z["f_mse_cor"], "mse_corrected_t2": np.nan,
                      "mse_clim": z["f_mse_clim"], "score_window": "2012-02..2022-12 "
                                                                  "(own-mean datum)"})],
        ignore_index=True)
    wt.to_parquet(out / "well_table.parquet", index=False)
    payload = {}
    for tag, Gs, Gl, sids in (("cal", Gc_s, Gc_l, cal.sids), ("heldout", Gf_s, Gf_l, fr.sids)):
        Gs = Gs.tocoo()
        Gl = Gl.tocsr()
        payload.update({f"{tag}_well_index": Gs.row.astype(np.int32),
                        f"{tag}_pump_index": Gs.col.astype(np.int32),
                        f"{tag}_g_seasonal": Gs.data,
                        f"{tag}_g_slow": np.asarray(Gl[Gs.row, Gs.col]).ravel(),
                        f"{tag}_sids": sids})
    payload.update(pump_source_row=cal.pump_row.astype(np.int32), pump_source_county=cal.pump_county,
                   a=a, source_files=np.array([rel_to_repo(L.ELEC / f) for f in (
                       "pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz")]),
                   source_sha256=np.array([sha256_file(L.ELEC / f) for f in (
                       "pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz")]))
    np.savez_compressed(out / "operator_weights.npz", **payload)
    mult = {"fit_2012_19_calibration_wells": {"a_seasonal": a[0], "a_slow": a[1]},
            "t2_by_fold": sc["t2"]["a_by_fold"],
            "units": "dimensionless multipliers of the model-unit SUB-GRID kernels (cell-mean "
                     "subtracted); a large value is a statistical scale, not a cone strength; "
                     "sd_m in correction_series is the spread over the 5 fold multipliers "
                     "(uncalibrated)",
            "sensitivity": {k: v["a"] for k, v in sc["sensitivity"].items()}}
    (out / "multipliers.json").write_text(json.dumps(mult, indent=1, default=float) + "\n")
    val = {"prereg": "H4.subgrid_operator.v2", "prereg_sha256": sc["prereg_sha256"],
           "gate_passed": passed, "criteria": sc["criteria"], "f": sc["f"],
           "t1": sc["t1"], "t2": sc["t2"], "placebo_p_registered": sc["placebo_p_registered"],
           "placebo_p_v1unit_reported": sc["placebo_p_v1unit_reported"],
           "f2_heldout_years_reported": sc["f2_heldout_years_reported"],
           "concentration": sc["concentration"],
           "v1unit_supported_low_quartiles": sc["v1unit_supported_low_quartiles"],
           "sensitivity": sc["sensitivity"],
           "counts": {k: sc[k] for k in sc if k.startswith(("n_", "cal_wells", "heldout_"))},
           "post_registration_looks": sc["post_registration_looks"], "config": sc["config"],
           "theta_sha256": THETA_SHA}
    (out / "validation.json").write_text(json.dumps(val, indent=1, default=float) + "\n")
    (out / "PRODUCT_CARD.md").write_text(card_text(sc, passed, ver))
    ex = inherited_exemptions()
    inputs = [rel_to_repo(p) for p in (
        L.ENGINE / "MANIFEST.json", L.ENGINE / "engine_arrays.npz", L.UNITS / "MANIFEST.json",
        L.UNITS / "units.json", L.UNITS / "wells.csv", L.ELEC / "MANIFEST.json",
        L.ELEC / "pumps_yunlin.npz", L.ELEC / "pumps_changhua.npz",
        L.ELEC / "pumps_chiayi_ingrid.npz", L.CENSUS / "MANIFEST.json",
        L.CENSUS / "census_pumps.parquet", L.HEADS / "MANIFEST.json",
        L.HEADS / "heads_monthly.parquet", L.HEADS / "station_meta_preorigin.parquet",
        DATAPREP / "pump_electricity/preorigin/v3/MANIFEST.json",
        DATAPREP / "pump_electricity/preorigin/v3/pumps_yunlin.npz",
        DATAPREP / "pump_electricity/preorigin/v3/pumps_changhua.npz",
        DATAPREP / "pump_electricity/preorigin/v3/pumps_chiayi_ingrid.npz",
        V.GRID_NPZ, THETA, PREREG, EVID / "scores.json", EVID / "arrays.npz",
        DT / "d1.json", DT / "d1_availability.json", DT / "t1_power.json")]
    m = manifest.build_manifest(
        out, product=PRODUCT, layer="L2", inputs=inputs, code=CODE,
        parameters={"config": cfg.as_dict(), "prereg": "H4.subgrid_operator.v2",
                    "multipliers_fit_window": [cfg.fit_start, cfg.fit_end],
                    "heads_read": "heads/L1_clean/v3 heads_monthly flags 0-1, window_end "
                                  "2022-12 (calibration wells: fit 2012-02..2019-12, v1-unit "
                                  "score 2020-01..2022-12; held-out wells: 2012-02..2022-12)",
                    "theta_sha256": THETA_SHA,
                    "provisional": "built on the cwa theta of batch 8 (merged, sha256 above); "
                                   "the I1 recalibration (Changhua census, preorigin/v3 "
                                   "forcing) changes the hindcast and makes this product "
                                   "stale; refit the multipliers then",
                    "layer_note": "per-well inferred correction with method id and an "
                                  "uncalibrated fold-spread sd (L2 in the plan's sense); not "
                                  "an L3 model-ready array"},
        fit_window_end=max(x["data_end"] for x in ex), non_exempt_fit_window_end="2022-12",
        exemptions=ex, validation_report="validation.json",
        gate={"passed": passed, "criteria": "H4.subgrid_operator.v2: F1-F4 (held-out wells, "
                                            "registered) and C1-C4 (v1 unit, design-time "
                                            "necessary conditions); all must pass",
              "failed": [c["id"] for c in sc["criteria"] if not c["passed"]]},
        allowed_uses=PASS_USES if passed else FAIL_USES,
        forbidden_uses=ALL_FORBIDDEN + ([] if passed else ["model_design", "validator"]),
        personal_data="no names, addresses, coordinates or well numbers of pumps; "
                      "operator_weights.npz indexes census pumps by row of "
                      "yunlin_electricity/preorigin/v2 pumps_<county>.npz and its kernel "
                      "weights imply pump-to-monitoring-well distances (quasi-locational); "
                      "gitignored under results/")
    errs = manifest.validate_manifest(m, out)
    if errs:
        raise SystemExit("manifest invalid: " + "; ".join(errs))
    manifest.write_manifest(out, m)
    sup = {"schema": "dataprep-superseded/1", "product": PRODUCT,
           "written": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": "dataprep_staging/h4_subgrid/b2_build_product.py", "current": ver,
           "superseded": [{
               "version": "v1", "superseded_by": ver,
               "why": "v1 subtracted the whole model-unit cone from the cell head; the 1 km "
                      "hindcast already carries the cell-average cone (v1 kernel averaged "
                      "over the well's cell: median corr 0.994 with the point value; sub-grid "
                      "part 0.197 of the point sd), so v1 tested mostly a cell-scale "
                      "re-scaling, not a sub-grid cone. Its gate also failed (A8). Its card's "
                      "'sub-grid' framing, 'location-specific part of the Q4 swing' as cone "
                      "evidence and the physical reading of the multiplier are withdrawn "
                      "(reports/H4.md, Repair round).",
               "allowed_uses": ["design_time_evidence", "after_the_fact_analysis"]}]}
    (DATAPREP / PRODUCT / "SUPERSEDED.json").write_text(json.dumps(sup, indent=1) + "\n")
    p = guard.open_product(out, "design_time_evidence")
    print(json.dumps({"product": rel_to_repo(out), "gate_passed": passed,
                      "guard_notes": p.notes}, indent=1))


def card_text(sc: dict, passed: bool, ver: str) -> str:
    f = sc["f"]
    lines = [f"# {PRODUCT} {ver} (WP H4, repair round)", "",
             f"Gate: **{'PASSED' if passed else 'FAILED'}** (pre-registration "
             "H4.subgrid_operator.v2: F1-F4 on held-out wells, C1-C4 design-time necessary "
             "conditions on the v1 unit; all must pass).", "",
             "**Provisional:** built on the engine/v2 hindcast with batch 8's merged cwa theta "
             f"(sha256 {THETA_SHA[:12]}...). The I1 recalibration changes the hindcast; refit "
             "then. Supersedes v1 (see ../SUPERSEDED.json).", "",
             "A SUB-GRID observation operator: `h_well = h_cell - c`, where `c` sums, over the "
             "census pumps (Yunlin, Changhua, in-grid Chiayi) within 2 km, the periodic-Theis "
             "(seasonal) and Thiem (slow) kernels MINUS their mean over the well's own 1 km "
             "cell, times two fan-wide multipliers fitted on the 158 calibration wells "
             "(2012-02..2019-12). Averaged over the cell the correction is zero, so it cannot "
             "rescale the cell-mean response the hindcast already carries.", "",
             "## Files", "",
             "- `correction_series.parquet`: sid x month 2012-02..2022-12 for the 158 "
             "calibration wells and the held-out wells: `c_m` (subtract from the cell head), "
             "`sd_m` (spread over the 5 fold multipliers, uncalibrated), unit components, "
             "flag 4 where a kernel exists, method id.",
             "- `well_table.parquet`, `operator_weights.npz`, `multipliers.json`, "
             "`validation.json`.", "",
             f"## Multipliers: a_seasonal {sc['a'][0]:.3f}, a_slow {sc['a'][1]:.3f} (statistical "
             "scales of the model-unit sub-grid kernel; at 1 the correction sd is about "
             f"{sc['unit_sd_supported_cal_m']['seasonal_median']:.4f} m).", "",
             "## Acceptance", ""]
    for c in sc["criteria"]:
        lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} {c['id']}: {c['criterion']} "
                     f"({c['threshold']}): {c['measured']}")
    lines += ["", "## Known limits", "",
              f"- Held-out wells scored: {sc['heldout_scored']} ({sc['heldout_scored_with_kernel']}"
              " with a kernel); high-swing set "
              f"Q{'+Q'.join(map(str, f['high_q']))}. H3 looked at some of these wells earlier "
              "(recovered wells); H4 never did before the registration.",
              "- C1-C4 use the v1 unit (158 wells, 2020-22), already scored by H4 v1: they "
              "are design-time evidence and can only fail the gate.",
              "- The sub-grid sum is not converged in r_max (far-field gradient term); "
              "r_max_3km is a reported member.",
              "- Model units: the twin's kWh-to-volume conversion and zone T, S.", "",
              "## Uses", "",
              ("Allowed: " + ", ".join(PASS_USES if passed else FAIL_USES)
               + (". Refit the multipliers whenever the hindcast changes." if passed else
                  ". Gate failed: not an I1 secondary model.")), ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
