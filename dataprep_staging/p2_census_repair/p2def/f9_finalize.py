"""Publish the P2def products as new versions (never overwriting), with validation.json,
PRODUCT_CARD.md and MANIFEST.json:

* ``census_post2016/{preorigin,full}/v<N>`` (P2d): assemble the members from the build work
  directory, run the registered construction checks (P2d.k1.construction_<variant>, scored
  through prereg3.assert_can_score), gate = back-test (Yunlin and Changhua) and construction;
* ``census_extension/full/v<N>`` (P2e): build the arrays, run P2e.k1.construction, gate =
  x1 (both counties) and construction;
* ``layer_response/v<N>`` (P2f): the declared analysis outputs.

Usage: python -I f9_finalize.py p2d|p2e|p2f
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d3_layer as L3  # noqa: E402
import dlib as D  # noqa: E402
from common import manifest  # noqa: E402
from common.uses import USES  # noqa: E402

PREREG_ID = "P2def.v1"
DECL_ID = "P2def.f.layer_response.v1"
TESTS = D.EVID / "tests"
PERSONAL = ("none: township names, cell arrays and county tables only; no uid, coordinate, "
            "name, address or meter number (pump coordinates and uids were read in memory to "
            "place cohorts)")


def c0_exemptions(wp_tag: str, uses: list[str]) -> list[dict]:
    m, _ = manifest.load_manifest(D.C0["preorigin"])
    out = []
    for x in m["exemptions"]:
        kind = "census_attributes" if "census attribute" in x["input"] else \
            "residual_unpaired_dec2022"
        y = dict(x)
        y["ledger_number_id"] = f"P2def.exemption.{wp_tag}.{kind}"
        y["inherited_from"] = f"pump_electricity/preorigin/v3 exemption {x['ledger_number_id']}"
        y["sensitivity"] = x["sensitivity"] + " [P2def: inherited unchanged]"
        y["uses"] = [u for u in x["uses"] if u in uses]
        out.append(y)
    return out


def score_k1(test_id: str) -> dict:
    from harness import prereg3
    return prereg3.assert_can_score(PREREG_ID, D.WP, test_id=test_id)


# ----------------------------------------------------------------------------- P2d
def card_p2d(variant: str, d: Path, val: dict) -> str:
    bt = val["backtest"]
    lines = [
        f"# census_post2016/{variant}/{d.name} (WP P2def, plan P2d)", "",
        f"Gate: **{'PASSED' if val['gate']['passed'] else 'FAILED'}** "
        f"({val['gate']['criteria']}).", "",
        "Synthetic post-2016 census layer: pumps the census lacks because its install record "
        "stops at mid-2016, for every county with kept pumps in the twin's 2148 active cells "
        "(Yunlin, Changhua, in-grid Chiayi). Additive to the cleaned census (C0 v3 / P2abc v2), "
        "model units (kWh of the cleaned census). Two members ship: `synthetic_post2016_low.npz` "
        "(rate = kept installs 2010-2015 without the 2013 surge / 5) and "
        "`synthetic_post2016_high.npz` (2010-2015 / 6); the third member, zero, is the census "
        "alone. **The three members are unweighted** (plan P2d): choose or weight them before "
        "results, on pre-origin evidence only (U7).", "",
        "Rule: monthly cohorts of rate/12 pumps per township x class from 2016-07; placement by "
        "the township's installs per 1 km cell; energy n x g(class, age) x P(class, month), g = "
        "relative age ramp of the 2010-2015 cohorts (ages 1-6, then held), P = mean energy per "
        "installed kept pump of the class in that month (recorded arm). Flags: 0 before "
        "2016-07 (value 0), 5 from 2016-07" +
        (", 8 (NaN) at 2026-07 (incomplete census month)" if variant == "full" else "") + ".",
        "",
        "sd: **uncalibrated** = s_rel x value with s_rel from the county's mid-2013 back-test "
        f"(s_rel {val['s_rel']}); cells and classes of a county fully correlated, counties in "
        "quadrature.", "",
        "Back-test (P2d.bt, mid-2013 freeze, 2014-16 cohorts): "
        + "; ".join(f"{k}: median township count error {bt[k]['median_township_count_rel_err']:.3f}"
                    f", fan kWh error {bt[k]['fan_kwh_rel_err']:+.3f} "
                    f"({'pass' if bt[k]['pass'] else 'FAIL'})" for k in ("yunlin", "changhua"))
        + ".", "",
        "Keys of each member file: value, flag, sd, method (C, A, T); value_by_county, "
        "flag_by_county, sd_by_county (K, C, A, T); value_total_all / _nodomestic with flags "
        "and sd (A, T); n_pumps (C, A, T) and n_pumps_by_county: expected synthetic in-grid "
        "pump counts; grid metadata (mask_hash, grid_hash). county_class_monthly.parquet: "
        "county x member x class x month county-scope and in-grid energy and counts. "
        "layer_parameters.json: rates by township x class, in-grid weights, ramps, per-pump "
        "index.", "",
        "Allowed uses: " + ", ".join(val["allowed_uses"]) + ". Risk (plan P2d): pumps added on "
        "old accounts after 2016 are already in the census, so the high member can double "
        "count. Code: dataprep_staging/p2_census_repair/p2def/."]
    return "\n".join(lines) + "\n"


def finalize_p2d() -> None:
    bt = D.load_json(TESTS / "P2d_backtest.json")
    gate_bt = bool(bt["yunlin"]["pass"] and bt["changhua"]["pass"])
    dirs = {}
    for variant in ("preorigin", "full"):
        d = manifest.new_version_dir(f"census_post2016/{variant}")
        L3.assemble(variant, D.WORK, d)
        dirs[variant] = d
    for variant, d in dirs.items():
        tid = f"P2d.k1.construction_{variant}"
        auth = score_k1(tid)
        chk = L3.check(D.WORK, dirs)
        own = {k: v for k, v in chk["items"].items()
               if k.startswith(variant) or k.startswith("h_")}
        ok = all(v["pass"] for v in own.values())
        D.save_json({"test_id": tid, "items": own, "pass": ok,
                     "prereg_sha256": auth["sha256"]}, TESTS / f"{tid}.json")
        pre = variant == "preorigin"
        allowed = (["forcing_frozen", "prior", "prior_width_input", "model_design",
                    "design_time_evidence", "after_the_fact_analysis"] if pre else
                   ["forcing_recorded_arm", "after_the_fact_analysis"])
        forbidden = sorted(set(USES) - set(allowed))
        params = D.load_json(d / "layer_parameters.json")
        val = {"product": f"census_post2016/{variant}", "version": d.name,
               "prereg": PREREG_ID, "construction": {"test_id": tid, "pass": ok, "items": own},
               "backtest": {k: {x: bt[k][x] for x in (
                   "median_township_count_rel_err", "n_towns_scored", "fan_kwh_rel_err",
                   "county_kwh_rel_err", "count_total_rel_err", "pass")}
                   for k in ("yunlin", "changhua", "chiayi")},
               "s_rel": params["s_rel"], "allowed_uses": allowed,
               "gate": {"passed": bool(ok and gate_bt),
                        "criteria": f"{PREREG_ID}: P2d.bt.yunlin and P2d.bt.changhua "
                                    f"(passed: {gate_bt}) and {tid} (passed: {ok})"}}
        if (D.EVID / "evidence" / "d4_evidence.json").exists():
            ev = D.load_json(D.EVID / "evidence" / "d4_evidence.json")
            val["evidence_tpc_residual_trend"] = {
                k: {x: v[x] for x in v if x.startswith("trend") or "growth" in x}
                for k, v in ev["tpc_residual_trend"].items()
                if k.startswith(variant) or not pre}
        D.save_json(val, d / "validation.json")
        (d / "PRODUCT_CARD.md").write_text(card_p2d(variant, d, val), encoding="utf-8")
        inputs = [D.C0[variant] / "MANIFEST.json", D.C0[variant] / "cells_by_class.npz",
                  D.C0["census_L1"] / "MANIFEST.json", D.C0["census_L1"] / "census_pumps.parquet",
                  TESTS / "P2d_backtest.json", D.WORK / f"{variant}_params.json",
                  D.WORK / f"{variant}_county_table.parquet"]
        inputs += [D.C0[variant] / f"pumps_{D.FILETAG[k]}.npz" for k in D.COUNTIES]
        inputs += [D.WORK / f"{variant}_{m}_raw.npz" for m in L3.MEMBERS]
        code = [D.CODE_DIR / f for f in ("dlib.py", "d2_backtest.py", "d3_layer.py",
                                         "f9_finalize.py", "prereg_def.py", "looks_def.py")]
        m = manifest.build_manifest(
            d, product=f"census_post2016/{variant}", layer="L2",
            inputs=[D.rel(x) for x in inputs], code=[D.rel(x) for x in code],
            parameters={"layer_start": D.LAYER_START, "members": {"low": list(D.LOW_YEARS),
                                                                  "high": list(D.HIGH_YEARS),
                                                                  "zero": "no file"},
                        "member_weights": "unweighted (plan P2d)", "max_age": D.MAX_AGE,
                        "min_class_cohort": D.MIN_CLASS_COHORT, "prereg": PREREG_ID,
                        "last_complete_month": L3.VAR[variant]["last"],
                        "sd_label": "uncalibrated"},
            fit_window_end="2026-07",
            non_exempt_fit_window_end="2022-12" if pre else None,
            exemptions=c0_exemptions("p2d", allowed) if pre else None,
            validation_report="validation.json", gate=val["gate"],
            allowed_uses=allowed, forbidden_uses=forbidden, personal_data=PERSONAL)
        manifest.write_manifest(d, m)
        print("written", d, "gate", val["gate"]["passed"])


# ----------------------------------------------------------------------------- P2e
def finalize_p2e() -> None:
    import e3_build as E3
    bt = D.load_json(TESTS / "P2e_backtest.json")
    d = manifest.new_version_dir("census_extension/full")
    info = E3.build(TESTS, d)
    auth = score_k1("P2e.k1.construction")
    chk = E3.check(TESTS, d)
    D.save_json({"test_id": "P2e.k1.construction", **chk, "prereg_sha256": auth["sha256"]},
                TESTS / "P2e.k1.construction.json")
    shutil.copy2(TESTS / "P2e_backtest.json", d / "backtest.json")
    gate_x1 = bool(bt["yunlin"]["x1_pass"] and bt["changhua"]["x1_pass"])
    allowed = ["forcing_recorded_arm", "after_the_fact_analysis"]
    val = {"product": "census_extension/full", "version": d.name, "prereg": PREREG_ID,
           "selected": bt["selected"],
           "x1": {k: {"rms": bt[k]["x1_selected_rms"], "pass": bt[k]["x1_pass"]}
                  for k in ("yunlin", "changhua")},
           "x2": {k: {c: v["rms_e2_last36"] for c, v in bt[k]["x2"].items()}
                  for k in ("yunlin", "changhua")},
           "sd_log_monthly": {k: bt[k]["sd_monthly_log"] for k in ("yunlin", "changhua")},
           "construction": chk, "build": info, "allowed_uses": allowed,
           "gate": {"passed": bool(gate_x1 and chk["pass"]),
                    "criteria": f"{PREREG_ID}: P2e.x1.annual_yunlin and P2e.x1.annual_changhua "
                                f"(passed: {gate_x1}) and P2e.k1.construction "
                                f"(passed: {chk['pass']})"}}
    D.save_json(val, d / "validation.json")
    sel = bt["selected"]
    card = [
        f"# census_extension/full/{d.name} (WP P2def, plan P2e)", "",
        f"Gate: **{'PASSED' if val['gate']['passed'] else 'FAILED'}** "
        f"({val['gate']['criteria']}).", "",
        "Extension of the cleaned census (C0 v3 recorded arm, county scope, all classes) past "
        "its last complete month (2026-06) to 2026-07 and 2026-08 from TPC d007012 farm-sector "
        "(農林漁牧) sales: census_hat(t) = TPC(t) x ratio, the ratio estimator selected per "
        f"county by a rolling two-month backtest (Yunlin: {sel['yunlin']}, Changhua: "
        f"{sel['changhua']}). **Recorded-forcing diagnostic arm only (U4).** Class x cell by "
        "the county's 2023-2025 same-month shares; in-grid Chiayi by its ratio to the Yunlin "
        "census. Flag 5, sd uncalibrated (monthly log RMS of the backtest). For 2026-07 the "
        "census itself exists in part: yunlin_electricity/full/v2 (recorded arm with the "
        "July-2026 imputation) is the census-based value; county_extension.csv compares them. "
        "Spatial changes after 2025 are not recoverable.", "",
        "Files: census_extension.npz (value/flag/sd (C, A, 2), by county, totals, grid "
        "metadata), county_extension.csv, backtest.json.", ""]
    (d / "PRODUCT_CARD.md").write_text("\n".join(card), encoding="utf-8")
    inputs = [D.TPC_CSV, D.C0["full"] / "MANIFEST.json", D.C0["full"] / "cells_by_class.npz",
              D.C0["full"] / "county_class_monthly.parquet",
              D.P2ABC["full"] / "MANIFEST.json", D.P2ABC["full"] / "county_class_monthly.parquet",
              TESTS / "P2e_backtest.json", TESTS / "P2e_county_extension.csv"]
    code = [D.CODE_DIR / f for f in ("dlib.py", "e2_extension.py", "e3_build.py",
                                     "f9_finalize.py", "prereg_def.py")]
    forbidden = sorted(set(USES) - set(allowed))
    m = manifest.build_manifest(
        d, product="census_extension/full", layer="L2",
        inputs=[D.rel(x) for x in inputs], code=[D.rel(x) for x in code],
        parameters={"months": ["2026-07", "2026-08"], "selected": sel, "prereg": PREREG_ID,
                    "share_years": [2023, 2024, 2025], "sd_label": "uncalibrated"},
        fit_window_end="2026-08", validation_report="validation.json", gate=val["gate"],
        allowed_uses=allowed, forbidden_uses=forbidden, personal_data=PERSONAL)
    manifest.write_manifest(d, m)
    print("written", d, "gate", val["gate"]["passed"])


# ----------------------------------------------------------------------------- P2f
def finalize_p2f() -> None:
    src = D.EVID / "p2f"
    d = manifest.new_version_dir("layer_response")
    for f in ("layer_response_localonly.csv", "site_ratios.parquet", "well_slopes.parquet",
              "layer_response_summary.json"):
        shutil.copy2(src / f, d / f)
    s = D.load_json(src / "layer_response_summary.json")
    allowed = ["validator", "model_design", "design_time_evidence", "after_the_fact_analysis"]
    val = {"product": "layer_response", "version": d.name, "declaration": DECL_ID,
           "primary": s["primary"], "placebo": s["placebo"], "depth_proxy": s["depth_proxy"],
           "model_inside": {k: v for k, v in s.items() if k.startswith("model_cwa_inside")},
           "allowed_uses": allowed,
           "gate": {"passed": True,
                    "criteria": f"evidence product: the analysis declared in {DECL_ID} ran as "
                                "declared (no acceptance threshold of its own; plan P2f is an "
                                "acceptance test of I1 rung R4)"}}
    D.save_json(val, d / "validation.json")
    card = [
        f"# layer_response/{d.name} (WP P2def, plan P2f)", "",
        "Local-only response of layer-1 (and 3, 4) heads relative to layer 2 at nested sites "
        "to local pump energy (3 km primary; 2 and 5 km and the Chow-Lin arm reported), "
        "observed (heads/L1_clean v3, 2012-01..2022-11) and modelled (cwa forward archive, "
        "primary; cwa fit_full hindcast; a Changhua-census forward of another session as "
        "supplementary). Regional signals (county in-grid total, kWh > 15 km away) and rain "
        "lags 0-2 are regressed out; year-shuffled placebos test that a local response exists. "
        "**Design-time evidence** (units not fresh). Use for I1 rung R4 only after R3 is "
        "accepted (plan P2f). The cwa model has no Changhua pumping: its Changhua ratio is not "
        "meaningful. 'selection' is not an allowed use because the pump positions come from the "
        "2026 census attribute table (exemption; exemptions cannot cover selection).", "",
        "Files: layer_response_localonly.csv (ratio table), site_ratios.parquet, "
        "well_slopes.parquet (station ids are public; no personal data), "
        "layer_response_summary.json.", ""]
    (d / "PRODUCT_CARD.md").write_text("\n".join(card), encoding="utf-8")
    inputs = [D.HEADS_L1 / "MANIFEST.json", D.HEADS_L1 / "heads_monthly.parquet",
              D.HEADS_L1 / "station_meta_preorigin.parquet", D.FWD_CWA, D.FWD_CHC,
              D.THETA_CWA, D.PILOT_PHYS / "hindcast_cwa.npz", D.RAIN, D.RAIN_ST,
              D.C0["preorigin"] / "MANIFEST.json", D.C0["census_L1"] / "census_pumps.parquet",
              D.P2ABC["preorigin"] / "MANIFEST.json", D.GRID_NPZ]
    inputs += [D.C0["preorigin"] / f"pumps_{D.FILETAG[k]}.npz" for k in D.COUNTIES]
    inputs += [D.P2ABC["preorigin"] / f"pumps_{D.FILETAG[k]}.npz" for k in D.COUNTIES]
    code = [D.CODE_DIR / f for f in ("dlib.py", "f1_layer_response.py", "f9_finalize.py",
                                     "looks_def.py", "prereg_def.py")]
    forbidden = sorted(set(USES) - set(allowed))
    m = manifest.build_manifest(
        d, product="layer_response", layer="L2",
        inputs=[D.rel(x) for x in inputs], code=[D.rel(x) for x in code],
        parameters={"declaration": DECL_ID, "heads_window": ["2012-01", "2022-11"],
                    "primary_radius_km": 3.0, "placebos": 20, "bootstrap": 4000,
                    "model_primary": D.rel(D.FWD_CWA),
                    "theta_cwa_sha256": D.sha256(D.THETA_CWA),
                    "provisional": "batch 8 folds still running; cwa theta provisional"},
        fit_window_end="2026-07", non_exempt_fit_window_end="2022-12",
        exemptions=c0_exemptions("p2f", allowed), validation_report="validation.json",
        gate=val["gate"], allowed_uses=allowed, forbidden_uses=forbidden,
        personal_data="none (public station ids; no pump uid or coordinate in any output)")
    manifest.write_manifest(d, m)
    print("written", d)


if __name__ == "__main__":
    {"p2d": finalize_p2d, "p2e": finalize_p2e, "p2f": finalize_p2f}[sys.argv[1]]()
