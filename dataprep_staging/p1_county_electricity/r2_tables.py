"""WP P1 round 2: render the report tables from the product files (numbers are never typed by
hand). Writes ``P1_evidence/r2/tables_r2.md``.

Usage: python -I r2_tables.py <out_md>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CE = REPO / "results/dataprep/county_electricity"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def main(out: str) -> int:
    ev = load(CE / "census_check/v1/evaluation.json")
    dg = load(CE / "census_check/v1/diagnostics_post_hoc.json")
    ob = load(CE / "tpc_observed/v1/validation.json")
    L = ["# WP P1 round 2: generated tables", "",
         "Source: county_electricity/census_check/v1 (evaluation.json, "
         "diagnostics_post_hoc.json) and county_electricity/tpc_observed/v1 (validation.json).",
         ""]
    c1 = ev["C1_a_vs_census"]
    L += ["## C1. a against the census (declaration; after the fact, not fresh)", "",
          "| Window | a* (census CH/YL, all classes) | v1 a | v1 / a* | v1 raw 90 % | "
          "v1 remedy 90 % | Reading |", "|---|---|---|---|---|---|---|"]
    for v in ("preorigin", "full"):
        x = c1[v]
        L.append(f"| {x['window'][0]}..{x['window'][1]} | {x['a_star']:.3f} | {x['a_v1']:.3f} | "
                 f"{x['a_v1_over_a_star']:.3f} | {x['raw90'][0]:.3f}-{x['raw90'][1]:.3f} | "
                 f"{x['remedy90'][0]:.3f}-{x['remedy90'][1]:.3f} | {x['reading']} |")
    L += ["", "| Window | a*_irr | v1 a_irr | v1 raw 90 % | inside | CH excl. domestic / YL "
          "all | both excl. domestic |", "|---|---|---|---|---|---|---|"]
    for v in ("preorigin", "full"):
        x, i = c1[v], c1[v]["irrigation"]
        L.append(f"| {v} | {i['a_star_irr']:.3f} | {i['a_irr_v1']:.3f} | "
                 f"{i['raw90_v1'][0]:.3f}-{i['raw90_v1'][1]:.3f} | {i['inside_raw90']} | "
                 f"{x['changhua_domestic_excluded_over_yunlin_all']:.3f} | "
                 f"{x['both_domestic_excluded']:.3f} |")
    L += ["", f"a* over 2012-01..2022-12: {c1['preorigin']['window_2012_2022']:.3f}", ""]
    c2 = ev["C2_misspecification_link"]
    L += ["## C2. Misspecification link (declaration)", "",
          f"- a*_pre - a_pre = {c2['a_star_minus_a_pre']:+.3f}",
          f"- month-of-year CV 2012-22, all classes: CH {c2['moy_cv_all']['changhua']:.3f}, "
          f"YL {c2['moy_cv_all']['yunlin']:.3f}, ratio {c2['moy_cv_all']['ratio']:.2f}",
          f"- irrigation: CH {c2['moy_cv_irrigation']['changhua']:.3f}, YL "
          f"{c2['moy_cv_irrigation']['yunlin']:.3f}, ratio {c2['moy_cv_irrigation']['ratio']:.2f}",
          "- v1 T9 bias (a_hat - a_eff): " + ", ".join(
              f"{k} {v:+.3f}" for k, v in c2["v1_T9_bias_a_hat_minus_a_eff"].items()), ""]
    c4 = ev["C4_phi_in"]
    L += ["## C4. phi_in (declaration)", "",
          f"- census in-grid electricity fraction 2012-22 phi* = {c4['phi_star']:.3f}; v1 phi_in "
          f"{c4['phi_in_v1']:.3f} (sigma_ln {c4['sigma_ln_phi']:.3f}); ln ratio "
          f"{c4['ln_ratio']:+.3f}, z {c4['z']:+.2f}; band half-width {c4['band_halfwidth_ln']:.3f}"
          f" -> inside: {c4['inside_band']}",
          f"- checks: pump count {c4['pump_count_fraction_all_pumps']:.3f} (kept "
          f"{c4['pump_count_fraction_kept']:.3f}); dedup electricity "
          f"{c4['dedup_electricity_fraction_2012_2022']:.3f}; irrigation "
          f"{c4['irrigation_fraction_2012_2022']:.3f}; Yunlin in-grid fraction "
          f"{c4['yunlin_in_grid_fraction_2012_2022']:.3f}", ""]
    L += ["## C5. In-grid anchor (declaration)", "",
          "| Window | v1 in-grid GWh/yr | census in-grid | ratio | z (mu log-sd 0.549) | inside "
          "90 % | v1 county | census county | ratio | Yunlin in-grid | Yunlin county |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w, x in ev["C5_in_grid_anchor"].items():
        L.append(f"| {w} | {x['estimate_in_grid_GWh_yr']:.1f} | {x['truth_in_grid_GWh_yr']:.1f} "
                 f"| {x['ratio_in_grid']:.3f} | {x['z_mu_prior']:+.2f} | "
                 f"{x['inside_mu90_0.549']} | {x['estimate_county_GWh_yr']:.1f} | "
                 f"{x['truth_county_GWh_yr']:.1f} | {x['ratio_county']:.3f} | "
                 f"{x['yunlin_in_grid_GWh_yr']:.1f} | {x['yunlin_county_GWh_yr']:.1f} |")
    L += ["", "## T10 / T11. Extension past the old census end, 2025-07..2026-06 "
          "(pre-registered tests)", "",
          "| Test | Candidate | months | sum pred | sum census | error | pass (<= 10 %) | "
          "monthly rel RMS | inside +-1.645 sd (Wilson 95 %) |", "|---|---|---|---|---|---|---|---|---|"]
    t = ev["T10_ext_yunlin"]
    cov = t["coverage_1.645sd"]
    L.append(f"| T10 Yunlin | YL_clean_hat | {t['n']} | {t['sum_pred']:.1f} | {t['sum_obs']:.1f} "
             f"| {pct(t['sum_err'])} | {t['pass']} | {t['monthly_rel_rms']:.3f} | "
             f"{cov['inside']}/{cov['n']} ({cov['wilson95'][0]:.2f}-{cov['wilson95'][1]:.2f}) |")
    for nm in ("scaled_apre", "scaled", "transfer"):
        t = ev["T11_ext_changhua"][nm]
        cov = t["coverage_1.645sd"]
        tag = "T11 Changhua (primary)" if nm == "scaled_apre" else "T11 Changhua (reported)"
        L.append(f"| {tag} | {nm} | {t['n']} | {t['sum_pred']:.1f} | {t['sum_obs']:.1f} | "
                 f"{pct(t['sum_err'])} | {t['pass']} | {t['monthly_rel_rms']:.3f} | "
                 f"{cov['inside']}/{cov['n']} ({cov['wilson95'][0]:.2f}-{cov['wilson95'][1]:.2f})"
                 f" |")
    d = ev["T11_decomposition"]
    r = ev["T10_T11_month_rule"]
    L += ["", f"T11 decomposition: census CH/YL over the scored months "
          f"{d['truth_CH_over_YL_scored_months']:.3f} vs a_pre {d['a_pre']:.3f}; "
          f"(1 + T10) x a_pre / ratio - 1 = {pct(d['check_scaled_apre'])}.",
          f"Month rule: all of {', '.join(r['scored_months'][:1])}..{r['scored_months'][-1]} "
          f"scored (lowest meter ratio "
          f"{min(r['meters_nonzero_ratio_to_m_minus_12'].values()):.2f}); 2026-07 excluded a "
          f"priori (ratio CH {r['jul_2026_ratio_reported']['changhua']:.2f}, YL "
          f"{r['jul_2026_ratio_reported']['yunlin']:.2f}).", "",
          "Monthly relative errors, T10 YL_clean_hat: " + ", ".join(
              f"{k} {pct(v)}" for k, v in ev["T10_ext_yunlin"]["monthly_rel_err"].items()), ""]
    L += ["## T12. Monthly shape of the v1 estimators against the census (declaration)", "",
          "| Estimator | window | level ratio | monthly rel RMS | corr | corr deseasonalised | "
          "moy CV est / census | annual RMS |", "|---|---|---|---|---|---|---|---|"]
    for nm, ws in ev["T12_monthly_shape"].items():
        for w, x in ws.items():
            L.append(f"| {nm} | {w} | {x['level_ratio']:.3f} | {x['monthly_rel_rms']:.3f} | "
                     f"{x['corr']:.3f} | {x['corr_deseasonalised']:.3f} | "
                     f"{x['moy_cv_ratio']:.2f} | {x['annual_rms_pct']:.1f} % |")
    L += ["", "## T13. Ensemble rank against the census (declaration)", "",
          "| Variant | member | ratio to census | rank | months |", "|---|---|---|---|---|"]
    for v in ("preorigin", "full"):
        for nm, x in sorted(ev["T13_ensemble_rank"][v].items(), key=lambda kv: kv[1]["rank"]):
            L.append(f"| {v} | {nm} | {x['ratio_to_truth']:.3f} | {x['rank']} | "
                     f"{x['months'][0]}..{x['months'][1]} ({x['months'][2]}) |")
    L += ["", f"z of the census inside the mu prior (log-sd 0.549 around a_pre): "
          f"{ev['T13_ensemble_rank']['truth_z_in_mu_prior']:+.2f}", ""]
    c8 = ev["C8_coverage_validator_role"]
    L += ["## C8. Census coverage of TPC 農林漁牧 (declaration; descriptive)", "",
          "| Year | CH TPC ag GWh | CH dedup agri / TPC | CH clean all / TPC | YL TPC ag | "
          "YL dedup agri / TPC | YL clean all / TPC | TPC CH/YL | census clean CH/YL |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in c8["changhua"]:
        a, b, q = c8["changhua"][k], c8["yunlin"][k], c8["ratio_CH_over_YL"][k]
        L.append(f"| {k} | {a['tpc_ag_GWh']:.1f} | {a['cov_dedup_agri']:.3f} | "
                 f"{a['cov_clean_all']:.3f} | {b['tpc_ag_GWh']:.1f} | {b['cov_dedup_agri']:.3f} "
                 f"| {b['cov_clean_all']:.3f} | {q['tpc_ag']:.3f} | {q['census_clean_all']:.3f} |")
    c9 = ev["C9_yunlin_consistency"]
    cr = dg["c9_amp_vs_new_yunlin_rows_le_2025_06"]
    L += ["", "## C9. Yunlin consistency (declaration)", "",
          "- max |new L0 - v1 YL_clean|: " + ", ".join(
              f"{k} {v['max_abs_GWh']:.2e} GWh" for k, v in c9.items() if k != "identical")
          + f"; identical (<= 1e-6): {c9['identical']}",
          f"- post hoc: {cr['rows_only_in_new']} pump-month rows only in the new sheet "
          f"({cr['rows_only_in_new_nonzero']} non-zero, {cr['kwh_only_in_new_GWh'] * 1000:.1f} "
          f"MWh in total, {cr['months_with_rows_only_in_new']} months); rows only in AMP_V2: "
          f"{cr['rows_only_in_amp']}; common rows with a value difference: "
          f"{cr['common_rows_value_diff_gt0']} of {cr['rows_common']}", ""]
    L += ["## Post hoc: census over TPC 農林漁牧, July-June years (not pre-registered)", "",
          "| Year | CH clean | CH TPC | CH clean/TPC | YL clean | YL TPC | YL clean/TPC |",
          "|---|---|---|---|---|---|---|"]
    for k in dg["census_over_tpc_jul_jun"]["changhua"]:
        a = dg["census_over_tpc_jul_jun"]["changhua"][k]
        b = dg["census_over_tpc_jul_jun"]["yunlin"][k]
        L.append(f"| {k} | {a['census_clean_GWh']:.1f} | {a['tpc_ag_GWh']:.1f} | "
                 f"{a['clean_over_tpc']:.3f} | {b['census_clean_GWh']:.1f} | "
                 f"{b['tpc_ag_GWh']:.1f} | {b['clean_over_tpc']:.3f} |")
    my = dg["meters_nonzero_yoy_median_by_year"]
    L += ["", "Median year-on-year ratio of meters with a non-zero reading: "
          + "; ".join(f"{c}: " + ", ".join(f"{y} {v:.3f}" for y, v in my[c].items()
                                             if int(y) >= 2020) for c in my), ""]
    g = ob["gate"]
    g3 = ob["G3_internal_share"]
    q = ob["qc_flag9"]
    ad = ob["ag_share_diagnostic_post_hoc"]
    L += ["## tpc_observed/v1 gate (pre-registered P1r2.tpc_observed_gate)", "",
          "| Check | Result | Detail |", "|---|---|---|",
          f"| G1 archive hashes | {g['G1_archive_hash']} | 3 files equal their SOURCE.json |",
          f"| G2 completeness | {g['G2_completeness']} | CH and YL 農林漁牧 128/128 months, 4 "
          f"sectors every month, no duplicates; national 236/236 months |",
          f"| G3 internal shares | {g['G3_internal_share']} | CH/YL: {g3['rows_over_0.01pp']} of "
          f"{g3['rows_checked']} rows over 0.01 pp (max {g3['max_abs_diff_pp']:.2f} pp); other "
          f"counties {g3['other_counties']['rows_over_0.01pp']} of "
          f"{g3['other_counties']['rows']} |",
          f"| G4 re-fetch (reported) | identical {ob['G4_refetch']['identical']} | "
          f"{ob['G4_refetch']['run_utc']} |",
          f"| Gate | {g['passed']} | |", "",
          f"QC flag 9: {q['rows_flagged_9']} rows in {q['failing_county_months']} county-months "
          f"(Changhua/Yunlin 農林漁牧 rows flagged: {q['changhua_yunlin_ag_rows_flagged']}). "
          "Changhua/Yunlin detail: " + "; ".join(
              f"{x['county']} {x['month']}: " + ", ".join(
                  f"{f['sector']} kWh = {f['kwh_over_implied']:.3f} x implied"
                  for f in x["flagged"]) for x in q["detail"]
              if x["county"] in ("彰化縣", "雲林縣")),
          f"Post hoc ag-share diagnostic: CH max {ad['彰化縣']['max_abs_diff_pp']:.4f} pp, YL max "
          f"{ad['雲林縣']['max_abs_diff_pp']:.4f} pp over 128 months each.", ""]
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
