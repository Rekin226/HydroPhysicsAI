"""Package the WP H2 repair-round-3 releases (``H2.repair3.v1`` packaging rule).

* ``full``  heads/L2_filled_full, new version: the daily-bridge component of the full variant
  (L1 flag 0/1 observations and flag-2 daily-bridged months, frozen bridge of
  ``design_params.json``; Feb-Mar 2026 bridged). Released only if ``H2.fresh3.d`` passed (and
  ``H2.fresh.d``, its original test, passed): uses ``statistical_layer_history`` and
  ``after_the_fact_analysis``; fit_window_end 2026-09. The after-the-fact checks on 2023-25
  truth are reported in validation.json and never gate it (U7).
* ``pre``   heads/L2_filled_preorigin, new version: the same daily-bridge component of the
  pre-origin variant for ``calibration_target`` (the v1 packaging rule, gate H2.fresh.d) with a
  corrected product card, plus, only if ``H2.ic.twin`` is 'confirmed', the January 2012
  ``gp_bf`` initial-condition field for ``initial_conditions``.

Existing versions are never touched; each release is a new version directory with
``fill.parquet``, ``validation.json``, ``PRODUCT_CARD.md`` and ``MANIFEST.json`` (written last).

Usage: python -I s15_package_r3.py full|pre
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from common import guard, manifest  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import ic_field, io_h2, s5_build, scoring  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402
from h2_heads_fill.s8_package import ALL_USES, LIB  # noqa: E402

L1 = "results/dataprep/heads/L1_clean/v3"
EV = "results/dataprep/reports/H2_evidence"
CODE = sorted(str(p.relative_to(C.REPO)) for p in C.CODE_DIR.glob("*.py"))
LIB_R3 = sorted(set(LIB) | {"hydrophysics/twin/calibrate_flow.py",
                            "hydrophysics/twin/zones.py", "hydrophysics/subsidence.py"})
INPUTS = [f"{L1}/MANIFEST.json", f"{L1}/heads_monthly.parquet", f"{L1}/heads_daily.parquet",
          "results/dataprep/validation_units/v3/MANIFEST.json",
          "results/dataprep/validation_units/v3/wells.csv",
          "results/twin/basemap.npz", "results/twin_forward/datum_gate_innov.npz",
          f"{EV}/design/design_params.json", f"{EV}/design/bridge_form.json",
          f"{EV}/port_check.json", f"{EV}/registered/H2.fresh.d.json",
          "results/dataprep/preregistration/H2.fill_acceptance.v1.json",
          "results/dataprep/preregistration/H2.repair3.v1.json"]
D3 = R3.REG_R3 / f"{R3.TEST_D3}.json"
IC3 = R3.REG_R3 / f"{R3.TEST_IC}.json"
AFTER = C.EVIDENCE_DIR / "after_the_fact" / "full_variant_checks.json"


def rel(p: Path) -> str:
    return str(Path(p).relative_to(C.REPO))


def gate_record(rec: dict) -> dict:
    v = rec["strata_clean"]["d"]
    c = v["coverage_gate"]
    return {"pass": rec["pass"], "coverage": c["coverage"], "interval": [c["lo"], c["hi"]],
            "n": c["n"], "k_clusters": c["k_clusters"], "n_eff_lower": c["n_eff_lower"],
            "calibrated": c["calibrated"], "window_applies": c["window_applies"],
            "baselines": {k: {"rmse_product": x["rmse_product"],
                              "rmse_baseline": x["rmse_baseline"]}
                          for k, x in v["baselines"].items() if x["defined"]},
            "share_classes_clean": rec["reported"].get("share_classes_clean"),
            "bridgeable_share": rec["reported"].get("bridgeable_share")}


def bridge_card(product, version, variant, stats, gate_txt, uses, extra_use, ic=None,
                edge=None) -> str:
    window = ("pre-origin: every fitted parameter saw 2012-01..2022-12 only" if variant == "pre"
              else "full: the bridge's deviation curves use 2012-01..2026-09, 2023-26 heads "
                   "included")
    L = [f"# {product} {version}: daily-bridge component (observations and flag-2 months)",
         "", f"{window}.", "",
         f"Monthly groundwater heads of the L1 v3 wells with flag 0/1 data: {stats['rows']} "
         f"rows, {stats['wells']} wells, {stats['months']} months. Rows are the L1 flag-0/1 "
         "monthly means (C20 months, and L1 partial months the bridge cannot complete) and the "
         "months completed by the daily bridge (flag 2). No router, GP or edge fill is in this "
         "version.", "", f"Gate: passed. {gate_txt}", "", f"Allowed uses: {', '.join(uses)}.",
         "", "## Flags in this version", ""]
    names = {0: "observed (L1 flag 0; method observed, or observed_partial for an L1 partial "
                "month of 3-19 clean days the bridge could not complete)",
             1: "observed, L1 barometric correction (same methods)",
             2: "daily bridge of a partial or outage month (same instrument; linear bridge "
                "between 3-day edge means plus the well's mean deviation curve)"}
    for f, n in sorted(stats["by_flag"].items(), key=lambda x: int(x[0])):
        L.append(f"- {f}: {n} rows - {names.get(int(f), '')}")
    L += ["", "## Use", ""] + extra_use
    if ic is not None:
        L += ["", "## ic_field_2012-01.npz (initial conditions)", "",
              f"January 2012 head field on the model grid, 4 layers x {ic['n_cells']} active "
              f"cells (`gp_bf`, `gp_bf_sd`): {ic['method']}. Sources: {ic['n_sources']} model "
              f"wells ({ic['n_observed']} observed in January 2012, {ic['n_backfilled']} "
              "backfilled with their first observed pre-origin head, as the twin's stage-3 "
              "construction does). No flag 2-6 fill enters it.",
              f"- Registered test H2.ic.twin (January 2013..2022, cell-level LOWO): "
              f"{ic['test']}.",
              "- At late-starting wells the field is weaker than the twin's own construction "
              f"(design-time evidence, January 2012, {edge['n_wells']} family-E leading-edge "
              f"wells): RMSE twin {edge['rmse']['twin_own']:.2f} m, gp_bf "
              f"{edge['rmse']['gp_bf_own']:.2f} m, H2's leading-edge router fill "
              f"{edge['rmse']['router']:.2f} m (not released: the complete fill failed its "
              "gate). LOWO holds out observed wells only, so it cannot measure the field at "
              f"the {ic['n_backfilled']} late-starting wells of January 2012 (backfilled "
              "sources in the test months, 0-24 per month, were also mostly late starters, but "
              "never held out).",
              f"- Where the gain sits: RMSE in proximal-fan cells "
              f"{ic['stats']['rmse_proximal']['gp_bf']:.2f} vs "
              f"{ic['stats']['rmse_proximal']['twin']:.2f} m ({ic['stats']['n_proximal']} of "
              f"{ic['stats']['n_cells']} well-months), elsewhere "
              f"{ic['stats']['rmse_non_proximal']['gp_bf']:.2f} vs "
              f"{ic['stats']['rmse_non_proximal']['twin']:.2f} m; but gp_bf is closer to the "
              f"truth than the twin's construction in only {ic['stats']['share_gp_closer']:.1%} "
              f"of the well-months (median absolute error {ic['stats']['medae']['gp_bf']:.2f} vs "
              f"{ic['stats']['medae']['twin']:.2f} m, MAE {ic['stats']['mae']['gp_bf']:.2f} vs "
              f"{ic['stats']['mae']['twin']:.2f} m): the gain comes from fewer large errors, not "
              "from the typical cell. The upper end of the interval, "
              f"{ic['stats']['ci_hi']:.2f} m, lies close to 0. Native 90 % coverage of "
              f"gp_bf_sd in the test: {ic['stats']['cov90_gp_bf']:.3f}."]
    L += ["", "## Columns", "",
          "sid, date (month start, ISO), month, head_m (= value), head_sd_m (= sd; 0 for "
          "observations, the bridged-share class sd for flag 2), flag, method, gap_class, "
          "gap_run_len, edge_dist, bridged_share, l1_n_clean_days, sigma_obs_m, u2_weight, "
          "in_model (one of the twin's 158 wells), layer, county, product_version, source_ids.",
          "", "## Known limits", "",
          "- Bridge sd by bridged-day share of the month: 0.181 m (share <= 0.5), 0.288 m "
          "(0.5-0.8), 0.554 m (> 0.8); u2_weight of bridged months 0.03-0.23. Above a share of "
          "0.5 the router (not released) was more accurate than the bridge on fresh tests "
          "(H2.fresh.d: bridge 0.207 / 0.350 / 0.595 m against router 0.262 / 0.279 / 0.274 m "
          "by share class; H2.fresh3.d: 0.200 / 0.307 / 0.607 against 0.311 / 0.272 / 0.282 m); "
          "bridged months remain the only fill admissible as calibration targets (U2), and "
          "their sd carries that error.",
          "- observed_partial rows (L1 partial months, 3-19 clean days) carry flag 0/1 and "
          "u2_weight 1; their representativeness error is only in sigma_obs_m (U2 rho(k), up to "
          "0.86 m). Use sigma_obs_m as the likelihood noise of every observed row.",
          "- Bridge gaps are limited to 60 days with at least 2 observed days in each 3-day "
          "edge window and a deviation curve from >= 3 other years; other partial months are "
          "not bridged.",
          "- Truth for every test is the L1 v3 C20 monthly mean; clean truth excludes months "
          "with or next to flag-9 days and the three pumping-cycle wells (07080132, 07080122, "
          "10050211)."]
    return "\n".join(L) + "\n"


def build_ic_gp_bf(outdir: Path) -> dict:
    """January 2012 gp_bf field (the H2.ic.twin primary candidate) on every active cell."""
    from h2_heads_fill.s10_ic_twin import setup
    S = setup()
    tw = S["tw"]
    t0 = io_h2.month_index(S["h"].dates, R3.IC_MONTH0)
    bf = S["bf"][:, t0]
    obs = np.isfinite(S["vals"][:, t0])
    src = np.isfinite(bf)
    dem = tw.dem
    cx = tw.centroids
    A = len(cx)
    val = np.zeros((4, A))
    sd = np.zeros((4, A))
    for k in range(4):
        val[k], sd[k] = ic_field.gp_layered(S["xy"][src], bf[src], S["lay"][src],
                                            S["dem"][src], cx, np.full(A, k), dem)
    np.savez(outdir / "ic_field_2012-01.npz", gp_bf=val, gp_bf_sd=sd,
             source_sids=np.array([s for s, m in zip(S["sids"], src, strict=True) if m]),
             source_values=bf[src], source_layers=S["lay"][src],
             source_backfilled=~obs[src], cell_centroids=cx, dem=dem,
             method=np.array("gp_bf"))
    return {"n_cells": int(A), "n_sources": int(src.sum()), "n_observed": int((obs & src).sum()),
            "n_backfilled": int((~obs & src).sum()),
            "method": "ic_field.gp_layered (per-layer drift on the SRTM cell elevation, Matern "
                      "3/2 + white-noise GP on the residuals of all layers) fitted on the twin's "
                      "information set"}


def ic_test_stats(ic: dict) -> dict:
    """Robustness figures of the registered H2.ic.twin cells (reported, not gated)."""
    import pandas as pd
    df = pd.read_parquet(R3.REG_R3 / f"{R3.TEST_IC}_cells.parquet")
    e = {c: (df[f"cell_{c}"] - df.truth).abs() for c in ("gp_bf", "twin")}
    return {"n_cells": int(len(df)), "n_proximal": int(df.proximal.sum()),
            "rmse_proximal": ic["cell"]["rmse_proximal"],
            "rmse_non_proximal": ic["cell"]["rmse_non_proximal"],
            "share_gp_closer": float((e["gp_bf"] < e["twin"]).mean()),
            "mae": {c: float(v.mean()) for c, v in e.items()},
            "medae": {c: float(v.median()) for c, v in e.items()},
            "ci_hi": ic["decision"]["site_boot_diff_p05_p50_p95"][2],
            "cov90_gp_bf": ic["cell"]["native_cov90_gp"]["gp_bf"]}


def package_full() -> Path:
    d1 = json.loads((C.EVIDENCE_DIR / "registered" / "H2.fresh.d.json").read_text())
    d3 = json.loads(D3.read_text())
    if not (d1["pass"] and d3["pass"]):
        raise SystemExit("H2.fresh3.d did not pass: no full-variant component (registered rule)")
    product = C.PRODUCT_FULL
    d = manifest.new_version_dir(product)
    stats = s5_build.build("full", d, params_path=R3.PARAMS_V1, mode="bridge", with_ic=False)
    uses = ["statistical_layer_history", "after_the_fact_analysis"]
    crit = ("H2.repair3.v1 packaging rule (declared before H2.fresh3.d was run): the "
            "full-variant daily-bridge component is released when H2.fresh3.d passes; gate = "
            "H2.fresh.d (v1, passed) and H2.fresh3.d (replication on d r = 60..69, passed)")
    after = json.loads(AFTER.read_text())
    p1 = json.loads(R3.PARAMS_V1.read_text())
    p_after = json.loads((C.REPO / after["params_file"]).read_text())
    bkeys = ("bridge_form", "bridge_sd_by_class", "bridge_class_pool", "bridge_disabled")
    same_bridge = all(p1.get(k) == p_after.get(k) for k in bkeys)
    if not same_bridge:
        raise SystemExit("the after-the-fact check used a different bridge configuration")
    val = {"schema": "h2-validation/1", "product": product, "version": d.name,
           "release": "r3_bridge_full", "gate_passed": True, "gate_criteria": crit,
           "gate_tests": {"H2.fresh.d": gate_record(d1), "H2.fresh3.d": gate_record(d3)},
           "after_the_fact_never_gate": {
               "D_2013_2025": after["D_2013_2025"],
               "note": "family D (the Feb-Mar 2026 window removed in every year 2013-2025) with "
                       "the full variant's bridge; logged 2026-10-13T20:05Z; U7: never a gate. "
                       f"Run with {after['params_file']}, whose bridge entries {list(bkeys)} "
                       "are identical to design_params.json (checked at packaging)."},
           "build_stats": stats}
    scoring.jdump(val, d / "validation.json")
    extra = ["- Statistical-layer history (plan WP H2 Use 4: continuity across the Feb-Mar "
             "2026 network outage) and after-the-fact analysis only. Never calibration targets, "
             "initial conditions, IDW fields, design or selection of a model scored on 2023-26 "
             "(U3), and never scoring targets (U1: flag 2 is not an observation).",
             f"- Feb-Mar 2026, model wells: {stats.get('febmar_2026_model_wells_by_flag')} "
             "(flag: rows).",
             "- After-the-fact (U7, never a gate): family D 2013-2025 (the Feb-Mar 2026 window "
             "removed in every year) with this bridge, clean truth: coverage "
             f"{after['D_2013_2025']['clean']['coverage'][0]['coverage']:.3f}, RMSE "
             f"{after['D_2013_2025']['clean']['rmse'][0]['rmse_product']:.3f} m (backfill "
             f"{after['D_2013_2025']['clean']['rmse'][0]['rmse_backfill']:.3f}, clim+AR(1) "
             f"{after['D_2013_2025']['clean']['rmse'][0]['rmse_clim_ar']:.3f})."]
    (d / "PRODUCT_CARD.md").write_text(bridge_card(product, d.name, "full", stats, crit, uses,
                                                   extra))
    inputs = INPUTS + [f"{L1}/station_meta_clean.parquet", rel(D3), rel(AFTER)]
    m = manifest.build_manifest(
        d, product=product, layer="L2", inputs=inputs, code=CODE + LIB_R3,
        parameters={"release": "r3_bridge_full", "mode": "bridge",
                    "design": json.loads(R3.PARAMS_V1.read_text()),
                    "fit_window": [C.START, C.FULL_END], "bridge_form": C.BRIDGE_FORM},
        fit_window_end=C.FULL_END, validation_report="validation.json",
        gate={"passed": True, "criteria": crit}, allowed_uses=uses,
        forbidden_uses=[u for u in ALL_USES if u not in uses],
        personal_data="none (station ids, layer and county codes; no coordinates, no contact "
                      "fields)")
    errs = manifest.validate_manifest(m, d) + guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit("; ".join(errs))
    manifest.write_manifest(d, m)
    print(d, stats["by_flag"])
    return d


def package_pre() -> Path:
    d1 = json.loads((C.EVIDENCE_DIR / "registered" / "H2.fresh.d.json").read_text())
    d3 = json.loads(D3.read_text())
    ic = json.loads(IC3.read_text())
    if not d1["pass"]:
        raise SystemExit("H2.fresh.d did not pass")
    confirmed = ic["decision"]["outcome"] == "confirmed"
    product = C.PRODUCT_PRE
    d = manifest.new_version_dir(product)
    stats = s5_build.build("pre", d, params_path=R3.PARAMS_V1, mode="bridge", with_ic=False)
    uses = ["calibration_target"] + (["initial_conditions"] if confirmed else [])
    dec = ic["decision"]
    ic_txt = (f"outcome '{dec['outcome']}': cell-level LOWO RMSE gp_bf "
              f"{dec['rmse_primary_cell']:.2f} m vs the twin's construction "
              f"{dec['rmse_twin_cell']:.2f} m, 90 % site-bootstrap interval of the difference "
              f"[{dec['site_boot_diff_p05_p50_p95'][0]:.2f}, "
              f"{dec['site_boot_diff_p05_p50_p95'][2]:.2f}] m ({ic['n_cells']} well-months, "
              f"{ic['n_sites']} sites)")
    crit = ("calibration_target: v1 packaging rule (declared in H2.fill_acceptance.v1 before "
            "any fresh result), gate H2.fresh.d (passed); replication H2.fresh3.d "
            f"({'passed' if d3['pass'] else 'FAILED'}). initial_conditions: "
            + ("granted for ic_field_2012-01.npz by the H2.repair3.v1 rule, H2.ic.twin "
               if confirmed else "not granted (H2.repair3.v1 rule), H2.ic.twin ") + ic_txt)
    edge = json.loads((R3.EV_R3 / "edge_ic_jan2012.json").read_text())
    icinfo = None
    if confirmed:
        icinfo = build_ic_gp_bf(d) | {"test": ic_txt, "stats": ic_test_stats(ic)}
    val = {"schema": "h2-validation/1", "product": product, "version": d.name,
           "release": "r3_bridge_pre" + ("_ic" if confirmed else ""), "gate_passed": True,
           "gate_criteria": crit,
           "gate_tests": {"H2.fresh.d": gate_record(d1)},
           "replication": {"H2.fresh3.d": gate_record(d3)},
           "ic_decision": {"H2.ic.twin": {k: ic[k] for k in ("decision", "cell", "pt",
                                                              "n_cells", "n_wells",
                                                              "n_sites", "months")},
                           "H2.ic.lowo_registered": json.loads(
                               (C.EVIDENCE_DIR / "registered" / "H2.ic.lowo.json").read_text()
                           )["rmse_lowo"],
                           "jan2012_post_look": json.loads(
                               (R3.EV_R3 / "ic_jan2012_twin.json").read_text())["cell"]["rmse"],
                           "late_starting_wells_design_time": {
                               k: edge[k] for k in ("n_wells", "rmse", "rmse_non_proximal",
                                                    "rmse_proximal", "n_proximal")}},
           "ic_field": icinfo, "build_stats": stats}
    scoring.jdump(val, d / "validation.json")
    extra = ["- Calibration targets: flags 0-2 (all rows), weighted by u2_weight = sigma_obs^2 "
             "/ (sigma_obs^2 + sd^2) (U2), with sigma_obs_m as the observation noise. A "
             "bridged month has sigma_obs 0.10 m.",
             "- Never scoring targets, baselines, datum fits or the prospective scorer (U1: "
             "flag 2 is not an observation)."]
    if confirmed:
        extra.append("- Initial conditions: ic_field_2012-01.npz (gp_bf) and the rows of "
                     "this table. The twin's 12-month burn-in is kept (plan H2 step 7).")
    else:
        extra.append("- No initial-condition field in this version: H2.ic.twin was not "
                     "confirmed, so I1 keeps the twin's construction.")
    (d / "PRODUCT_CARD.md").write_text(bridge_card(product, d.name, "pre", stats, crit, uses,
                                                   extra, ic=icinfo, edge=edge))
    inputs = INPUTS + [f"{L1}/station_meta_preorigin.parquet", rel(D3), rel(IC3),
                       f"{EV}/registered/H2.ic.lowo.json", f"{EV}/repair3/ic_jan2012_twin.json",
                       f"{EV}/repair3/edge_ic_jan2012.json",
                       "results/twin_runs/stage3_datum_gate_cwa/stage3_fit_summary.json"]
    m = manifest.build_manifest(
        d, product=product, layer="L2", inputs=inputs, code=CODE + LIB_R3,
        parameters={"release": val["release"], "mode": "bridge",
                    "design": json.loads(R3.PARAMS_V1.read_text()),
                    "fit_window": [C.START, C.PRE_END], "bridge_form": C.BRIDGE_FORM,
                    "ic_field": "gp_bf January 2012" if confirmed else None},
        fit_window_end=C.PRE_END, validation_report="validation.json",
        gate={"passed": True, "criteria": crit}, allowed_uses=uses,
        forbidden_uses=[u for u in ALL_USES if u not in uses],
        personal_data="none (station ids, layer and county codes; model-grid cell centroids in "
                      "the IC file; no well coordinates, no contact fields)")
    errs = manifest.validate_manifest(m, d) + guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit("; ".join(errs))
    manifest.write_manifest(d, m)
    print(d, stats["by_flag"], "ic" if confirmed else "no ic")
    return d


def main(argv):
    if argv == ["full"]:
        package_full()
    elif argv == ["pre"]:
        package_pre()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
