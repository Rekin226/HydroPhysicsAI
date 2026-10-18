"""Build ``results/dataprep/public_wells/v1`` (WP P6 Phase-0 inventory and sensitivity member).

Everything is computed in memory first (tables T1, drought wells D1, census C1, ratio K1,
member M1, reconciliation R1/R2); the version directory is created only when every declared
check has been evaluated. ``--outdir`` writes the same files to another directory without a
manifest (used to check that two builds give identical bytes).

Usage: cpu_slot.sh python -I build_product.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, manifest  # noqa: E402
from common.hashing import grid_hash, mask_hash, sha256_file  # noqa: E402
from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402

from p6_public_wells import census_check as C  # noqa: E402
from p6_public_wells import inventory as INV  # noqa: E402
from p6_public_wells import member as M  # noqa: E402
from p6_public_wells import p6const as K  # noqa: E402
from p6_public_wells import p6lib as L  # noqa: E402
from p6_public_wells import ratio as R  # noqa: E402
from p6_public_wells import tables as T  # noqa: E402
from p6_public_wells import transcribe_p6 as TR  # noqa: E402

SUPERSEDES = {"v2": {
    "product": "public_wells/v1",
    "reason": "its PRODUCT_CARD.md (and the first P6 report draft) said the primary member "
              "ramps to 0 by 2020-07; the arrays reach 0 from 2019-07 (the last cohorts stop at "
              "mid-2019; timing_late from 2020-01). v2 corrects the card; every data file is "
              "byte-identical to v1."}}
EXTRA_TOWNS = ("虎尾", "土庫", "元長")       # B14: 虎尾、土庫、元長地區地下水減抽配套措施
PRE_USES = ["forcing_frozen", "model_design", "prior", "design_time_evidence"]
ALLOWED = PRE_USES + ["validator", "after_the_fact_analysis"]
FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "calibration_target",
             "prospective_scorer", "selection", "initial_conditions", "idw_field",
             "prior_width_input", "forcing_recorded_arm", "statistical_layer_history"]
CODE = ["__init__.py", "p6const.py", "p6lib.py", "transcribe_p6.py", "tables.py",
        "census_check.py", "ratio.py", "member.py", "inventory.py", "build_product.py",
        "prereg_p6.py", "fetch_plans.py"]


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, default=conv) + "\n"


def extra_component(cen: pd.DataFrame, dates: list[str]) -> tuple[np.ndarray, dict]:
    """B14 (475 x 10^4 m3/yr, Tuku-Huwei-Yuanchang groundwater reduction measure of 102):
    on the census public-supply pumps of those townships, aquifers 3-4, stop 2013-07.
    Added after the declaration (identified while building the inventory)."""
    A, T_ = L.grid()["n_active"], len(dates)
    c = cen[(cen.county == "yunlin") & cen.is_public & cen.town_key.isin(EXTRA_TOWNS)]
    c = c[np.isfinite(c.x_twd97) & np.isfinite(c.y_twd97)]
    cells = L.cell_index(c.x_twd97.to_numpy(), c.y_twd97.to_numpy())
    w = np.zeros(A)
    np.add.at(w, cells[cells >= 0], 1.0 / len(cells))
    stop = dates.index(f"{2013:04d}-{K.DISPOSAL_MONTH:02d}")
    lw = M.layer_weights("taiwan_water", "primary")
    rate = 475.0 * 1e4 / K.DAYS_PER_YEAR
    X = lw[:, None, None] * (w * rate)[None, :, None] * (np.arange(T_) < stop)[None, None, :]
    return X, {"volume_1e4m3_per_yr": 475.0, "stop_month": dates[stop], "n_points": len(cells),
               "in_grid_fraction": float(w.sum()), "towns": list(EXTRA_TOWNS),
               "status": "added after P6.inventory_member.v1; not part of any declared variant"}


def location_sensitivity(res: dict) -> dict:
    """Exemption sensitivity: each location variant against the primary member (2010, before
    any disposal): in-grid volume, county split, spatial overlap and centroid shift."""
    g = L.grid()
    cc = L.cell_county()
    ii, jj = np.nonzero(g["mask"])
    xc = float(g["x0"]) + (jj + 0.5) * float(g["dx"])
    yc = float(g["y0"]) + (ii + 0.5) * float(g["dx"])
    t = res["dates"].index("2010-06")
    base = res["arrays"]["primary"][:, :, t].sum(0)
    out = {}
    for var in ("primary", "loc_census_only", "loc_drought_only", "loc_uniform_county"):
        v = res["arrays"][var][:, :, t].sum(0)
        rec = {"in_grid_Mm3_per_yr": float(v.sum() * K.DAYS_PER_YEAR / 1e6),
               "overlap_with_primary": float(np.minimum(v, base).sum() / base.sum())}
        for county in ("changhua", "yunlin"):
            m = cc == K.COUNTY_ZH[county]
            s = v[m].sum()
            rec[f"{county}_Mm3_per_yr"] = float(s * K.DAYS_PER_YEAR / 1e6)
            if s > 0:
                cx, cy = (v[m] * xc[m]).sum() / s, (v[m] * yc[m]).sum() / s
                b = base[m].sum()
                bx, by = (base[m] * xc[m]).sum() / b, (base[m] * yc[m]).sum() / b
                rec[f"{county}_centroid_shift_km"] = float(np.hypot(cx - bx, cy - by) / 1e3)
        out[var] = rec
    return out


def compute() -> dict:
    t22, t23, t45, t25 = T.table_2_2(), T.table_2_3(), T.table_4_5(), T.table_2_5()
    t29 = T.table_2_9_check(t22)
    dw, dwc = C.wells_located()
    n1 = pd.read_csv(guard.open_product(K.N1_PRODUCT, "after_the_fact_analysis").path(
        "drought_wells.csv"))
    cols = ["county", "table_no", "agency", "township", "x_twd97", "y_twd97", "cell"]
    d1_equal = bool(dw[cols].reset_index(drop=True).astype(str).equals(
        n1[cols].reset_index(drop=True).astype(str)))
    cen = C.census_table()
    energy = C.pump_energy()
    pc = C.purpose_counts(cen)
    eb = C.energy_by_purpose(cen, energy)
    matches, msum = C.drought_matches(dw, cen, energy)
    k1 = R.run()
    res = M.build(t22, t23, dw, cen)
    extra, extra_info = extra_component(cen, res["dates"])
    rec = INV.reconcile(t22, t45, t23, t25, t29)
    facts = pd.concat([INV.facts(), INV.table_rows(t45, t23, dwc)], ignore_index=True)
    facts["flag"] = 0
    facts["method"] = np.where(facts.extraction == "text_layer", "printed_text",
                               "printed_table")
    clos = INV.closures(t22, t23)
    return {"t22": t22, "t23": t23, "t45": t45, "t25": t25, "t29": t29, "dw": dw, "dwc": dwc,
            "d1_equal_n1": d1_equal, "cen": cen, "pc": pc, "eb": eb, "matches": matches,
            "msum": msum, "plan_counts": C.plan_counts_check(cen), "k1": k1, "res": res,
            "extra": extra, "extra_info": extra_info, "rec": rec, "facts": facts,
            "closures": clos}


def checks(x: dict) -> dict:
    t22 = x["t22"]
    t1 = {"two_read_differences": {k: t22[k]["read_differences"] for k in ("counts",
                                                                          "volumes")},
          "text_layer_problems": {k: t22[k]["text_layer_problems"] for k in ("counts",
                                                                            "volumes")},
          "row_arithmetic_exceptions": {k: t22[k]["row_arithmetic_exceptions"]
                                        for k in ("counts", "volumes")},
          "column_arithmetic_exceptions": {k: t22[k]["column_arithmetic_exceptions"]
                                           for k in ("counts", "volumes")},
          "header_order_ok": t22["header_order_ok"],
          "table_2_5": {"two_read_differences": x["t25"]["read_differences"],
                        "text_layer_problems": x["t25"]["text_layer_problems"],
                        "merged_groups": x["t25"]["merged_groups"]},
          "table_4_5_arithmetic_ok": x["t45"]["all_ok"]}
    explained = {"counts": [{"row": "105年", "cells_sum": 73.0, "printed_total": 72.0},
                            {"column": "ts_yunlin", "cells_sum": 8.0, "printed_total": 7.0}]}
    unexplained = []
    for k in ("counts", "volumes"):
        for e in t22[k]["row_arithmetic_exceptions"] + t22[k]["column_arithmetic_exceptions"]:
            e2 = {kk: e[kk] for kk in ("row", "column", "cells_sum", "printed_total") if kk in e}
            if e2 not in explained.get(k, []):
                unexplained.append({k: e})
    t1["exceptions_explained_by_printed_note"] = explained["counts"]
    t1["unexplained_exceptions"] = unexplained
    t1["passed"] = bool(
        not any(t22[k]["read_differences"] for k in ("counts", "volumes"))
        and not any(t22[k]["text_layer_problems"] for k in ("counts", "volumes"))
        and not unexplained and t22["header_order_ok"]
        and not x["t25"]["text_layer_problems"] and x["t45"]["all_ok"])
    t25_diffs = [d for d in x["t25"]["read_differences"]
                 if not ({T.norm_cell(d.get("pdf_read", "")), T.norm_cell(d.get("img_read", ""))}
                         <= {"-"})]
    t1["table_2_5_substantive_read_differences"] = t25_diffs
    t1["passed"] = t1["passed"] and not t25_diffs
    dwc = x["dwc"]
    d1 = {"checks": dwc, "equal_to_natural_experiments_v1": x["d1_equal_n1"],
          "compared_columns": "county, table_no, agency, township, x_twd97, y_twd97, cell "
                              "(well_name differs by design: N1's parser drops every token "
                              "equal to the count, P6 drops only the trailing count)",
          "passed": bool(
              dwc["Changhua"]["rows_parsed"] == 49 and not dwc["Changhua"]["missing_numbers"]
              and dwc["Changhua"]["by_agency_parsed"] == dwc["Changhua"][
                  "by_agency_stated_table3"]
              and dwc["Yunlin"]["rows_parsed"] == 520 and dwc["Yunlin"]["missing_numbers"]
              == [280] and dwc["Yunlin"]["by_agency_parsed"].get(
                  "Irrigation Agency (農水署)") == 420 and x["d1_equal_n1"])}
    m1 = {"mass_balance": x["res"]["checks"],
          "max_rel_err": max(v["mass_balance_max_rel_err"] for v in x["res"]["checks"].values()),
          "months": [x["res"]["dates"][0], x["res"]["dates"][-1], len(x["res"]["dates"])]}
    m1["passed"] = bool(m1["max_rel_err"] <= 1e-9 and x["res"]["dates"][-1] == K.MEMBER_T1)
    return {"T1": t1, "D1": d1, "M1": m1, "R1": x["rec"]["R1"], "R2": x["rec"]["R2"]}


def member_npz(x: dict) -> dict:
    res = x["res"]
    g = L.grid()
    out = {"dates": np.array(res["dates"]), "variants": np.array(K.VARIANTS),
           "twin_layer_aquifer_code": np.arange(1, K.N_LAYERS + 1),
           "wra_to_model": np.float64(K.WRA_TO_MODEL),
           "units": np.array("m3/day of abstraction (positive = abstraction; pass -rate as the "
                             "engine's q_extra), physical (WRA) volumes; model units = "
                             "physical / wra_to_model"),
           "flag_note": np.array("flag 5 (inferred): synthetic sensitivity member from "
                                 "agency-reported volumes and proxy locations; no sd "
                                 "(uncalibrated); the variants span the stated assumptions"),
           "nx": np.int64(g["nx"]), "ny": np.int64(g["ny"]), "dx": np.float64(g["dx"]),
           "x0": np.float64(g["x0"]), "y0": np.float64(g["y0"]),
           "n_active": np.int64(g["n_active"]), "mask_hash": np.array(mask_hash(g["mask"])),
           "grid_hash": np.array(grid_hash(int(g["nx"]), int(g["ny"]), float(g["dx"]),
                                           float(g["x0"]), float(g["y0"]), g["mask"]))}
    for var, X in res["arrays"].items():
        out[f"rate_phys_m3d__{var}"] = X.astype(np.float32)
    out["rate_model_m3d__primary"] = (res["arrays"]["primary"] / K.WRA_TO_MODEL).astype(
        np.float32)
    out["rate_phys_m3d__pending_tw_changhua"] = res["pending"].astype(np.float32)
    out["rate_phys_m3d__extra_tuku_huwei_2013"] = x["extra"].astype(np.float32)
    co = res["cohorts"]
    out["cohort_id"] = co.cohort.to_numpy().astype(str)
    out["cohort_year"] = co.year.to_numpy()
    out["cohort_volume_1e4m3_per_yr"] = co.volume_1e4m3_per_yr.to_numpy()
    out["cohort_rate_phys_m3d"] = res["rate_m3d"]
    for loc, v in res["locs"].items():
        out[f"cohort_weights__{loc}"] = v["W"].astype(np.float32)
    return out


def member_cohorts(x: dict) -> pd.DataFrame:
    res = x["res"]
    co = res["cohorts"].copy()
    for loc, v in res["locs"].items():
        co[f"proxy__{loc}"] = v["proxy_used"]
        co[f"n_points__{loc}"] = v["n_points"]
        co[f"n_points_in_grid__{loc}"] = v["n_points_in_grid"]
        co[f"in_grid_fraction__{loc}"] = v["W"].sum(1)
    co["rate_phys_m3d"] = res["rate_m3d"]
    co["stop_month_primary"] = [f"{y}-{K.DISPOSAL_MONTH:02d}" for y in co.year]
    return co


def write_outputs(d: Path, x: dict, chk: dict) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    x["facts"].to_csv(d / "inventory.csv", index=False)
    x["closures"].to_csv(d / "closures.csv", index=False)
    wl = x["dw"][["county", "table_no", "agency", "township", "well_name", "n_wells",
                  "x_twd97", "y_twd97", "cell", "in_grid", "source", "vintage", "month"]]
    wl.to_csv(d / "wells_located.csv", index=False)
    x["pc"].to_csv(d / "census_purpose_counts.csv", index=False)
    x["eb"].to_csv(d / "census_energy_by_purpose.csv", index=False)
    x["matches"].to_csv(d / "drought_census_match.csv", index=False)
    ms = x["msum"].copy()
    ms["nearest_purposes_le50m"] = ms.nearest_purposes_le50m.map(
        lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
    ms.to_csv(d / "drought_census_match_summary.csv", index=False)
    member_cohorts(x).to_csv(d / "member_cohorts.csv", index=False)
    ann = M.annual_summary({**x["res"], "arrays": {**x["res"]["arrays"],
                                                    "extra_tuku_huwei_2013": x["extra"]}})
    ann.to_csv(d / "member_annual.csv", index=False)
    np.savez_compressed(d / "sensitivity_member.npz", **member_npz(x))
    k1 = dict(x["k1"])
    (d / "ratio_check.json").write_text(_json(k1), encoding="utf-8")
    return {"annual": ann}


def summary_numbers(x: dict, ann: pd.DataFrame) -> dict:
    def yr(var, y, county=None):
        s = ann[(ann.variant == var) & (ann.year == y)]
        if county:
            s = s[s.county == county]
        return float(s.in_grid_Mm3_per_yr_physical.sum())
    clos = x["closures"]
    disposed = clos[clos.status == "disposed"]
    dates = x["res"]["dates"]

    def zero_from(var):
        tot = x["res"]["arrays"][var].sum((0, 1))
        nz = [i for i in range(len(dates)) if tot[i] > 0]
        return dates[nz[-1] + 1] if nz and nz[-1] + 1 < len(dates) else None
    return {
        "member_primary_zero_from": zero_from("primary"),
        "member_timing_late_zero_from": zero_from("timing_late"),
        "phase1_disposed_wells": float(x["t22"]["counts"]["rows"]["合計"]["total"]),
        "phase1_disposed_volume_Mm3_per_yr": float(disposed.volume_1e4m3_per_yr.sum() / 100),
        "by_agency_Mm3_per_yr": {k: float(v / 100) for k, v in disposed.groupby(
            "agency").volume_1e4m3_per_yr.sum().items()},
        "by_county_Mm3_per_yr": {k: float(v / 100) for k, v in disposed.groupby(
            "county").volume_1e4m3_per_yr.sum().items()},
        "member_primary_in_grid_Mm3_per_yr": {str(y): yr("primary", y) for y in
                                              (2010, 2012, 2015, 2018, 2019, 2020)},
        "member_primary_in_grid_2010_model_units": yr("primary", 2010) / K.WRA_TO_MODEL,
        "member_variants_2010_Mm3": {v: yr(v, 2010) for v in K.VARIANTS},
        "pending_tw_changhua_in_grid_Mm3_per_yr": yr("pending_tw_changhua", 2015),
        "extra_tuku_huwei_in_grid_Mm3_per_yr_2012": yr("extra_tuku_huwei_2013", 2012),
        "member_primary_changhua_yunlin_2010": [yr("primary", 2010, "changhua"),
                                                yr("primary", 2010, "yunlin")],
    }


def manifest_for(d: Path, x: dict, chk: dict, sens: dict) -> dict:
    prereg = PREREG_DIR / f"{K.PREREG_ID}.json"
    srcs = ["moea_annex", "cy53001", "drought_wells", "brief110", "plan_rev1", "plan_phase2"]
    inputs = [K.SRC[s] for s in srcs] + list(K.SOURCE_JSON) + [
        K.API_CACHE / "p2_at2_2.json", K.API_CACHE / "p2_at2_5.json",
        K.API_CACHE / "b110_irr_yl.json",
        K.C0_L1 / "census_pumps.parquet", K.C0_L1 / "MANIFEST.json",
        K.C0_PRE / "pumps_yunlin.npz", K.C0_PRE / "pumps_changhua.npz",
        K.C0_PRE / "pumps_chiayi_ingrid.npz", K.C0_PRE / "cells_by_class.npz",
        K.C0_PRE / "MANIFEST.json", K.ENGINE / "engine_arrays.npz", K.ENGINE / "MANIFEST.json",
        K.GRID_NPZ, K.N1_PRODUCT / "drought_wells.csv", prereg]
    code = [K.CODE_DIR / c for c in CODE] + [
        K.CODE_DIR.parent / "n1_natural_experiments" / "parse_sources.py",
        K.CODE_DIR.parent / "n1_natural_experiments" / "n1const.py"]
    ex_uses = list(PRE_USES)
    exemptions = [
        {"input": "data_fetch_api/new_measurements/wra_reports_misc/"
                  "wra111_drought_wells_report.pdf (2022 drought-well inventory: agency, "
                  "township, TWD97 coordinates; table 3 counts)",
         "data_end": K.VINTAGE["drought_wells"],
         "reason": "the only printed coordinates of Taiwan Water, irrigation-agency and Taiwan "
                   "Sugar wells in the fan; plan WP P6 step 3 names them as the location proxy. "
                   "The wells existed in 2022 (pre-origin facts) but the report was published "
                   "2023-04",
         "ledger_number_id": "P6.v1.exemption.drought_well_locations",
         "sensitivity": "member variants loc_census_only and loc_uniform_county against the "
                        "primary (validation.json exemption_sensitivity: in-grid volume, "
                        "county split, overlap with the primary, centroid shift)",
         "uses": ex_uses},
        {"input": "pump_electricity/census_L1/v3 census_pumps.parquet (census attribute table "
                  "of the TPC 2026 delivery: purpose, coordinates, HP; C0 exemption "
                  "C0.v3.exemption.census_attributes.census_L1)",
         "data_end": "2026-07",
         "reason": "census public-supply and industrial pumps locate the cohorts without "
                   "drought wells (plan WP P6 step 3); no attribute snapshot as of 2022-12 "
                   "exists (C0)",
         "ledger_number_id": "P6.v1.exemption.census_attributes",
         "sensitivity": "member variants loc_drought_only and loc_uniform_county against the "
                        "primary (validation.json exemption_sensitivity); C0 R9 (pumps "
                        "installed after 2022-12: -3.7e-4 / -1.2e-4 of 2012-2022 in-grid "
                        "energy)",
         "uses": ex_uses},
        {"input": "pump_electricity/preorigin/v3 kept set and kWh (C0 exemption "
                  "C0.v3.exemption.residual_unpaired_dec2022.preorigin: December-2022 halves "
                  "of bills read in 2023 kept in the duty statistic)",
         "data_end": "2023-01",
         "reason": "the census energy tables (C1) and the K1 ratio's irrigation share use C0's "
                   "kept pumps; C0 cannot identify those meters without the January-2023 "
                   "reading",
         "ledger_number_id": "P6.v1.exemption.c0_residual_dec2022",
         "sensitivity": "C0 repair3_residual.py: 0 kept-meter flips without those December "
                        "values; P6 energy summaries use 2012-2021 and K1 Jan-May 2013-2021 "
                        "only, so no December-2022 value enters them",
         "uses": ex_uses}]
    gate_ok = bool(chk["R1"]["passed"] and chk["T1"]["passed"] and chk["D1"]["passed"]
                   and chk["M1"]["passed"])
    return manifest.build_manifest(
        d, product=K.PRODUCT, layer="L2", inputs=inputs, code=code,
        parameters={"prereg": {"id": K.PREREG_ID, "sha256": sha256_file(prereg)},
                    "member": {"t0": K.MEMBER_T0, "t1": K.MEMBER_T1,
                               "disposal_month": K.DISPOSAL_MONTH,
                               "timing_variants": K.TIMING_VARIANTS,
                               "days_per_year": K.DAYS_PER_YEAR,
                               "wra_to_model": K.WRA_TO_MODEL,
                               "aquifers": list(K.MEMBER_AQUIFERS),
                               "aquifer_split": list(K.MEMBER_AQUIFER_SPLIT),
                               "ia_variant_aquifer": K.IA_VARIANT_AQUIFER,
                               "variants": list(K.VARIANTS),
                               "public_purposes": list(K.PUBLIC_PURPOSES),
                               "industrial_purposes": list(K.INDUSTRIAL_PURPOSES)},
                    "reconcile": {"annex_wells": K.ANNEX_WELLS,
                                  "annex_volume_1e4m3": K.ANNEX_VOLUME_1E4,
                                  "tolerance": K.RECONCILE_TOL},
                    "api_models": [K.MODEL_PDF, K.MODEL_IMG]},
        fit_window_end="2026-07", non_exempt_fit_window_end="2022-12",
        validation_report="validation.json",
        gate={"passed": gate_ok,
              "criteria": "plan WP P6 acceptance (inventory totals reconcile with the MOEA "
                          "annex within 5 %: R1 of P6.inventory_member.v1) together with the "
                          "declared T1 transcription, D1 drought-well and M1 member checks",
              "R1": chk["R1"]["passed"], "T1": chk["T1"]["passed"],
              "D1": chk["D1"]["passed"], "M1": chk["M1"]["passed"]},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none: agency well names and TWD97 coordinates of public drought wells "
                      "(Taiwan Water, irrigation agency, Taiwan Sugar) as printed by WRA; "
                      "census-derived files hold aggregates, cell weights and nearest-pump "
                      "distances only (no well numbers, private pump coordinates, names or "
                      "addresses)",
        exemptions=exemptions,
        extra={"supersedes": SUPERSEDES[d.name]} if d.name in SUPERSEDES else None)


def product_card(x: dict, chk: dict, sens: dict, nums: dict, version: str = "v1") -> str:
    k1 = x["k1"]["primary"]
    supersede = ("" if version not in SUPERSEDES else
                 "\n\n**Supersedes " + SUPERSEDES[version]["product"] + "**: "
                 + SUPERSEDES[version]["reason"] + " Data files are byte-identical.")
    r1 = chk["R1"]
    return f"""# public_wells/v1 (WP P6, Phase-0 inventory)

Gate: **{'PASSED' if r1['passed'] else 'FAILED'}** (plan WP P6 acceptance, R1 of
{K.PREREG_ID}: inventory totals of the public wells disposed in 2011-2018 against the MOEA 2019
annex, within 5 %). Inventory {r1['N_printed_row_totals_100_107']:.0f} wells (annex 774,
{100 * r1['rel_diff_wells']:.2f} %) and {r1['V_cells_100_107']:,.2f} x 10^4 m3/yr (annex 8,793.9,
{100 * r1['rel_diff_volume']:.4f} %). The totals were seen before the declaration (disclosed);
the threshold is the plan's.

Contents:
- `inventory.csv`: {len(x['facts'])} printed quantities (MOEA 2019 annex, first-revision and
  phase-2 plans of 2020, Control Yuan 53001, table 4-5, annex table 2-3, drought-well table 3),
  each with file, PDF page, quote as printed (found on that page) and vintage.
- `closures.csv`: the cohort table of annex table 2-2 (agency x year 2011-2019, counts and
  agency-reported abstraction cuts, 10^4 m3/yr; two Claude API reads + text layer, all cells
  agree) and the 129 Changhua Taiwan Water wells planned for 2020-2024.
- `wells_located.csv`: 569 printed drought wells of 2022 (Changhua 49, Yunlin 520) with TWD97
  coordinates and active cell (published 2023-04: exemption).
- `census_purpose_counts.csv`, `census_energy_by_purpose.csv`, `drought_census_match*.csv`:
  the census PURPOSE reconciliation (C0 census_L1/v3, preorigin/v3).
- `sensitivity_member.npz`, `member_cohorts.csv`, `member_annual.csv`: the fan-scale closure-
  ramp member and its variants; `ratio_check.json`: the pre-origin WRA-to-model ratio check.

**Sensitivity member only. It never enters the primary model (plan P6 step 3).** Rates in
m3/day of abstraction (positive), shape (4 twin layers, 2148 cells, 192 months 2007-01..2022-12),
physical (WRA) volumes; model units = physical / 2.58 (plan, stated assumption). Primary in-grid
volume {nums['member_primary_in_grid_Mm3_per_yr']['2010']:.1f} Mm3/yr before 2011
({nums['member_primary_in_grid_2010_model_units']:.1f} in model units), ramping to 0 from {nums['member_primary_zero_from']}
(timing_late: {nums['member_timing_late_zero_from']}).{supersede}
Variants: {', '.join(K.VARIANTS)}; components to add: pending_tw_changhua
({nums['pending_tw_changhua_in_grid_Mm3_per_yr']:.1f} Mm3/yr in grid, constant through 2022)
and extra_tuku_huwei_2013 (added after the declaration). Flag 5 (inferred); no sd
(uncalibrated); the variants span the assumptions.

Known limits:
- The volumes are agency-reported abstraction cuts (可減抽水量), not metered pumping. WRA's
  Jan-May public-well series (N1, two chart reads, after the fact) shows the Yunlin Taiwan Water
  drop in 2019, while the table credits 2,193.5 x 10^4 m3/yr of it to 2017-2018; the Yunlin
  irrigation-association series does not fall as much as the 44.5 Mm3/yr credited.
- Locations are proxies: the 2022 drought wells are wells that still existed in 2022, and the
  census public-supply pumps are small (max 20 HP Yunlin, 11 HP Changhua). Neither marks a
  closed well. Census pumps within 50 m of a drought well are mostly small irrigation pumps;
  the public wells themselves are not in the census.
- Layers 3-4 are the plan's assumption; Control Yuan 53001 puts Yunlin irrigation-association
  wells at 120-150 m (aquifer 2): variant layers_ia_aq2.
- K1 (pre-origin, 2013-2021): WRA / model = {k1['ratio']:.3f} (per-year sd {k1['per_year_sd']:.3f},
  falling from {k1['per_year'][2013]:.2f} in 2013 to {k1['per_year'][2021]:.2f} in 2021); the
  member keeps 2.58 as declared.

Allowed uses: {', '.join(ALLOWED)} (pre-origin uses through 3 exemptions: drought-well
locations 2023-04, census attributes 2026-07, C0's residual December-2022 halves 2023-01).
Forbidden: scoring_target, calibration_target, baseline, datum_fit, selection and the rest
listed in MANIFEST.json.

Pre-registration: {K.PREREG_ID} (declaration). Validation: validation.json. Report:
results/dataprep/reports/P6.md.
"""


def finish(d: Path, x: dict, chk: dict, write: bool) -> dict:
    sens = location_sensitivity(x["res"])
    ann = M.annual_summary({**x["res"], "arrays": {**x["res"]["arrays"],
                                                    "extra_tuku_huwei_2013": x["extra"]}})
    nums = summary_numbers(x, ann)
    if write:
        d = manifest.new_version_dir(K.PRODUCT)
    else:
        d.mkdir(parents=True, exist_ok=True)
    card = product_card(x, chk, sens, nums, d.name)
    w = write_outputs(d, x, chk)
    if not w["annual"].equals(ann):
        raise SystemExit("annual summary differs between the two computations")
    val = {"product": f"{K.PRODUCT}/{d.name}", "prereg": K.PREREG_ID,
           "checks": chk, "C1_census": {"plan_counts": x["plan_counts"],
                                        "drought_census_match_summary": json.loads(
                                            x["msum"].to_json(orient="records",
                                                              force_ascii=False))},
           "K1_ratio_preorigin": {kk: x["k1"][kk] for kk in ("primary", "pilot_definition",
                                                             "wra_m3_per_census_kwh",
                                                             "plan_value", "pilot_value")},
           "K1_transcription": {"max_abs_diff_frac_axis": x["k1"]["wra"][
               "max_abs_diff_frac_axis"], "reads": {kk: {kq: v[kq] for kq in (
                   "model", "stated_axis", "county_axis", "rescaled")}
                   for kk, v in x["k1"]["wra"]["reads"].items()}},
           "exemption_sensitivity": sens, "pending_tw_changhua": x["res"]["pending_info"],
           "extra_tuku_huwei_2013": x["extra_info"], "summary": nums,
           "api_token_usage": TR.token_usage()}
    (d / "validation.json").write_text(_json(val), encoding="utf-8")
    (d / "PRODUCT_CARD.md").write_text(card, encoding="utf-8")
    m = manifest_for(d, x, chk, sens)
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise SystemExit(f"manifest invalid: {errs}")
    if write:
        manifest.write_manifest(d, m)
    return {"dir": d, "gate": m["gate"]["passed"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=None, help="scratch directory named v<N> (no "
                                                   "manifest written; validated only)")
    a = ap.parse_args(argv)
    x = compute()
    chk = checks(x)
    if a.outdir:
        r = finish(Path(a.outdir), x, chk, write=False)
        print(f"wrote {r['dir']} (manifest validated, not written) gate {r['gate']}")
        return 0
    r = finish(Path("."), x, chk, write=True)
    print(f"wrote {rel_to_repo(r['dir'])} gate {r['gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
