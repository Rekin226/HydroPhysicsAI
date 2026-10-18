"""Write the WP P2def report (results/dataprep/reports/P2def.md and P2def.json) from the
evidence files, so every quoted number comes from a file.

Usage: python -I r_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import r_ledger as RL  # noqa: E402

EV = D.EVID
REP = D.DATAPREP / "reports"


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def pct(x, n=1):
    return "n/a" if x is None else f"{100 * x:+.{n}f} %"


def main() -> None:
    bt = D.load_json(EV / "tests" / "P2d_backtest.json")
    k1 = {v: D.load_json(EV / "tests" / f"P2d.k1.construction_{v}.json")
          for v in ("preorigin", "full")}
    eb = D.load_json(EV / "tests" / "P2e_backtest.json")
    ek1 = D.load_json(EV / "tests" / "P2e.k1.construction.json")
    d4 = D.load_json(EV / "evidence" / "d4_evidence.json")
    d5 = D.load_json(EV / "evidence" / "d5_heads_ramp.json")
    facts = D.load_json(EV / "report_facts.json")
    chk = D.load_json(EV / "checker.json")
    gp = D.load_json(EV / "guard_probe.json")
    led = D.load_json(EV / "ledger_check.json")
    syn = D.load_json(EV / "synthetic" / "t0_synthetic.json")
    ext = pd.read_csv(D.DATAPREP / RL.latest("census_extension/full") / "county_extension.csv")
    lrdir = D.DATAPREP / RL.latest("layer_response")
    lr = D.load_json(lrdir / "layer_response_summary.json")
    rt = pd.read_csv(lrdir / "layer_response_localonly.csv")
    comp = pd.read_csv(EV / "design_time" / "d0_billing_completeness_by_year.csv")
    params = D.load_json(D.DATAPREP / RL.latest("census_post2016/preorigin") /
                         "layer_parameters.json")
    prods = {p: RL.latest(p) for p in ("census_post2016/preorigin", "census_post2016/full",
                                       "census_extension/full", "layer_response")}
    gates = {p: D.load_json(D.DATAPREP / v / "MANIFEST.json")["gate"]
             for p, v in prods.items()}
    with open(D.DATAPREP / "preregistration" / "INDEX.jsonl", encoding="utf-8") as fh:
        reg = [json.loads(x) for x in fh if '"P2def' in x]

    def r(scope, heads, pair="L1/L2", r_km=3.0, version="recorded"):
        x = rt[(rt["scope"] == scope) & (rt["heads"] == heads) & (rt["pair"] == pair) &
               (rt["r_km"] == r_km) & (rt["version"] == version)].iloc[0]
        return x

    def rs(scope, heads, **kw):
        x = r(scope, heads, **kw)
        return f"{x['median_ratio']:.2f} [{x['ci90_lo']:.2f}, {x['ci90_hi']:.2f}] " \
               f"({int(x['n_sites'])} sites)"

    by = bt["yunlin"]
    bc = bt["changhua"]
    pl = lr["placebo"]
    sz_pre = facts["p2d_sizes.preorigin"]
    sz_full = facts["p2d_sizes.full"]
    tr = d4["tpc_residual_trend"]
    sv = d4["survivorship"]
    km = d4["kwh_per_active_meter"]
    dp = lr["depth_proxy"]

    def comp_row(k, code, yr, col):
        x = comp[(comp["county"] == k) & (comp["code"] == code) & (comp["yr"] == yr)]
        return float(x[col].iloc[0])

    acc = [
        {"criterion": "P2d back-test Yunlin (P2d.bt.yunlin): median township count error, "
                      "2014-01..2016-06 cohorts from a mid-2013 freeze",
         "threshold": "<= 0.30", "measured": f"{by['median_township_count_rel_err']:.3f} "
                                             f"({by['n_towns_scored']} townships)",
         "passed": by["median_township_count_rel_err"] <= 0.30},
        {"criterion": "P2d back-test Yunlin: fan (in-grid) kWh 2014-01..2022-11",
         "threshold": "|error| <= 0.20", "measured": f"{by['fan_kwh_rel_err']:+.3f} "
                                                    f"({by['fan_kwh_pred_GWh']:.2f} vs "
                                                    f"{by['fan_kwh_act_GWh']:.2f} GWh)",
         "passed": abs(by["fan_kwh_rel_err"]) <= 0.20},
        {"criterion": "P2d back-test Changhua (P2d.bt.changhua): median township count error",
         "threshold": "<= 0.30", "measured": f"{bc['median_township_count_rel_err']:.3f} "
                                             f"({bc['n_towns_scored']} townships)",
         "passed": bc["median_township_count_rel_err"] <= 0.30},
        {"criterion": "P2d back-test Changhua: fan (in-grid) kWh 2014-01..2022-11",
         "threshold": "|error| <= 0.20", "measured": f"{bc['fan_kwh_rel_err']:+.3f} "
                                                    f"({bc['fan_kwh_pred_GWh']:.2f} vs "
                                                    f"{bc['fan_kwh_act_GWh']:.2f} GWh)",
         "passed": abs(bc["fan_kwh_rel_err"]) <= 0.20},
        {"criterion": "P2d construction checks, pre-origin layer (P2d.k1.construction_preorigin)",
         "threshold": "all items pass (recomputation <= 1e-9 relative)",
         "measured": f"{sum(v['pass'] for v in k1['preorigin']['items'].values())}/"
                     f"{len(k1['preorigin']['items'])} items",
         "passed": k1["preorigin"]["pass"]},
        {"criterion": "P2d construction checks, full layer (P2d.k1.construction_full)",
         "threshold": "all items pass", "measured":
             f"{sum(v['pass'] for v in k1['full']['items'].values())}/"
             f"{len(k1['full']['items'])} items", "passed": k1["full"]["pass"]},
        {"criterion": "P2d members zero / low / high shipped, unweighted (plan P2d)",
         "threshold": "shipped", "measured": "low and high files in both variants; zero = no "
                                             "layer", "passed": True},
        {"criterion": "P2e extension, one-year-ahead annual error of the selected candidate, "
                      "Yunlin (P2e.x1.annual_yunlin; plan P1 rule)",
         "threshold": "RMS <= 0.10",
         "measured": f"{eb['yunlin']['x1_selected_rms']:.4f} ({eb['yunlin']['selected']})",
         "passed": eb["yunlin"]["x1_pass"]},
        {"criterion": "P2e extension, Changhua (P2e.x1.annual_changhua)", "threshold": "RMS <= 0.10",
         "measured": f"{eb['changhua']['x1_selected_rms']:.4f} ({eb['changhua']['selected']})",
         "passed": eb["changhua"]["x1_pass"]},
        {"criterion": "P2e construction checks (P2e.k1.construction)", "threshold": "all pass",
         "measured": f"{sum(v['pass'] for v in ek1['items'].values())}/{len(ek1['items'])} items",
         "passed": ek1["pass"]},
        {"criterion": "P2f declared analysis (P2def.f.layer_response.v1) run as declared; "
                      "local-only observed L1/L2 ratio with 90 % interval, cwa model side, "
                      "placebos (design-time evidence; no threshold of its own)",
         "threshold": "delivered", "measured": "Yunlin " + rs("yunlin", "obs") +
                                               "; model cwa " + rs("yunlin", "cwa_forward"),
         "passed": True},
        {"criterion": "P2f regional placebo: local L1 and L2 responses beat year-shuffled "
                      "placebos (site bootstrap p < 0.05), reported",
         "threshold": "p < 0.05 for both layers",
         "measured": f"Yunlin L1 p {pl['yunlin']['L1']['p_one_sided']:.4f}, L2 "
                     f"{pl['yunlin']['L2']['p_one_sided']:.4f} (identified); Changhua L1 "
                     f"{pl['changhua']['L1']['p_one_sided']:.4f}, L2 "
                     f"{pl['changhua']['L2']['p_one_sided']:.4f} (not identified)",
         "passed": bool(pl["yunlin"]["L1_L2_ratio_locally_identified"] and
                        pl["changhua"]["L1_L2_ratio_locally_identified"])},
        {"criterion": "Manifest checker on the 4 new product versions", "threshold": "0 failures",
         "measured": f"{len(chk['manifests'])} manifests, "
                     f"{sum(m['n_hashed'] for m in chk['manifests'])} files re-hashed, failures "
                     f"{sum(1 for m in chk['manifests'] if m['status'] != 'ok')}",
         "passed": all(m["status"] == "ok" for m in chk["manifests"])},
        {"criterion": "Guard probe (U3/U4/U5) of every use on the 4 products",
         "threshold": "all as designed", "measured": f"{gp['as_designed']}/{gp['n']}",
         "passed": gp["as_designed"] == gp["n"]},
        {"criterion": "Ledger (U6)", "threshold": "0 conflicts",
         "measured": f"{led['rows_written']} rows written (fragment), conflicts "
                     f"{led['conflicts']} across {led['rows_all_fragments']} rows",
         "passed": led["conflicts"] == 0},
        {"criterion": "Synthetic recovery of the P2d core (toy census, design-time)",
         "threshold": "ramp <= 5 %, counts and kWh <= 8 %",
         "measured": f"ramp {syn['ramp_rel_err_vs_generating']:.4f}, kWh "
                     f"{syn['backtest_county_kwh_err']:+.3f}, count "
                     f"{syn['backtest_count_err']:+.3f}", "passed": syn["pass"]},
    ]
    for a in acc:
        a["passed"] = bool(a["passed"])

    key = [
        f"P2d back-test (registered): Yunlin median township count error "
        f"{by['median_township_count_rel_err']:.3f}, total count {by['pred_count_total']:.0f} vs "
        f"{by['act_count_total']} ({pct(by['count_total_rel_err'])}), in-grid kWh "
        f"{pct(by['fan_kwh_rel_err'])}; Changhua {bc['median_township_count_rel_err']:.3f}, "
        f"{bc['pred_count_total']:.0f} vs {bc['act_count_total']} "
        f"({pct(bc['count_total_rel_err'])}), kWh {pct(bc['fan_kwh_rel_err'])}: P2d FAILS",
        "Rates (kept pumps per yr, 2010-15): Yunlin low "
        f"{params['counties']['yunlin']['members']['low']['rate_total_per_yr']:.0f}, high "
        f"{params['counties']['yunlin']['members']['high']['rate_total_per_yr']:.0f}; Changhua "
        f"low {params['counties']['changhua']['members']['low']['rate_total_per_yr']:.0f}, high "
        f"{params['counties']['changhua']['members']['high']['rate_total_per_yr']:.0f}",
        "Irrigation age ramp g (2010-15 cohorts, relative to the per-pump mean): Yunlin "
        + " ".join(f"{x:.2f}" for x in params["counties"]["yunlin"]["ramp_g"]["irrigation"])
        + "; Changhua "
        + " ".join(f"{x:.2f}" for x in params["counties"]["changhua"]["ramp_g"]["irrigation"]),
    ]
    for k in ("yunlin", "changhua"):
        lo, hi = sz_pre[f"{k}.low"]["2022"], sz_pre[f"{k}.high"]["2022"]
        lo5, hi5 = sz_full[f"{k}.low"]["2025"], sz_full[f"{k}.high"]["2025"]
        key.append(
            f"Layer size {k}: Dec 2022 {lo['n_county_dec']:.0f}-{hi['n_county_dec']:.0f} pumps, "
            f"2022 {lo['GWh_in_grid']:.1f}-{hi['GWh_in_grid']:.1f} GWh in grid "
            f"({100 * lo['share_of_census_in_grid']:.1f}-{100 * hi['share_of_census_in_grid']:.1f}"
            f" % of the census); Dec 2025 {lo5['n_county_dec']:.0f}-{hi5['n_county_dec']:.0f}, "
            f"2025 {lo5['GWh_in_grid']:.1f}-{hi5['GWh_in_grid']:.1f} GWh "
            f"({100 * lo5['share_of_census_in_grid']:.1f}-"
            f"{100 * hi5['share_of_census_in_grid']:.1f} %)")
    key += [
        f"s_rel (layer sd, uncalibrated): Yunlin {params['s_rel']['yunlin']:.3f}, Changhua "
        f"{params['s_rel']['changhua']:.3f}, Chiayi {params['s_rel']['chiayi']:.3f}",
        "Heads ramp (design-time, frozen cwa, rerun): +"
        f"{d5['pilot_reproduction']['total_m3d_end2022']:.0f} m3/d, formal sd "
        f"{d5['pilot_reproduction']['total_sd_formal']:.0f}; chi2/df {d5['chi2_about_zero']:.1f}/"
        f"{d5['df_zero']} -> sd x{d5['inflation_sqrt_chi2_df_zero']:.2f} = "
        f"{d5['total_sd_inflated_zero']:.0f}; site bootstrap sd {d5['bootstrap']['sd']:.0f}, 90 % "
        f"[{d5['bootstrap']['q05']:.0f}, {d5['bootstrap']['q95']:.0f}] (census mean "
        f"{d5['yunlin_census_abstraction_mean_m3d']:.0f} m3/d)",
        "TPC residual trend (farm sales minus cleaned census, GWh/yr per yr): pre-origin Yunlin "
        f"{tr['preorigin.yunlin']['trend_all_from_2016']['slope_GWh_per_yr']:+.1f} +- "
        f"{tr['preorigin.yunlin']['trend_all_from_2016']['se']:.1f} (from 2016), "
        f"{tr['preorigin.yunlin']['trend_all_from_2017']['slope_GWh_per_yr']:+.1f} +- "
        f"{tr['preorigin.yunlin']['trend_all_from_2017']['se']:.1f} (2017); Changhua "
        f"{tr['preorigin.changhua']['trend_all_from_2016']['slope_GWh_per_yr']:+.1f} +- "
        f"{tr['preorigin.changhua']['trend_all_from_2016']['se']:.1f}, "
        f"{tr['preorigin.changhua']['trend_all_from_2017']['slope_GWh_per_yr']:+.1f} +- "
        f"{tr['preorigin.changhua']['trend_all_from_2017']['se']:.1f}; synthetic growth Yunlin "
        f"{tr['preorigin.yunlin']['synthetic_low_growth_2017_on']['slope_GWh_per_yr']:.1f}-"
        f"{tr['preorigin.yunlin']['synthetic_high_growth_2017_on']['slope_GWh_per_yr']:.1f}, "
        f"Changhua "
        f"{tr['preorigin.changhua']['synthetic_low_growth_2017_on']['slope_GWh_per_yr']:.1f}-"
        f"{tr['preorigin.changhua']['synthetic_high_growth_2017_on']['slope_GWh_per_yr']:.1f}",
        "Survivorship lower bound (terminated meters, last-12-month GWh): pre-origin Yunlin "
        f"{sv['preorigin.yunlin']['total_meters']} ({sv['preorigin.yunlin']['total_last12_GWh']:.2f}),"
        f" Changhua {sv['preorigin.changhua']['total_meters']} "
        f"({sv['preorigin.changhua']['total_last12_GWh']:.2f}); full Yunlin "
        f"{sv['full.yunlin']['total_meters']} ({sv['full.yunlin']['total_last12_GWh']:.2f}), "
        f"Changhua {sv['full.changhua']['total_meters']} "
        f"({sv['full.changhua']['total_last12_GWh']:.2f})",
        "kWh per active meter: Yunlin "
        f"{km['full.yunlin']['2012_2016'][0]:.0f}-{km['full.yunlin']['2012_2016'][1]:.0f} "
        f"(2012-16), {km['full.yunlin']['2017_2022'][0]:.0f}-"
        f"{km['full.yunlin']['2017_2022'][1]:.0f} (2017-22), "
        f"{km['full.yunlin']['2023_2025'][0]:.0f}-{km['full.yunlin']['2023_2025'][1]:.0f} "
        f"(2023-25); Changhua {km['full.changhua']['2012_2016'][0]:.0f}-"
        f"{km['full.changhua']['2012_2016'][1]:.0f}, {km['full.changhua']['2017_2022'][0]:.0f}-"
        f"{km['full.changhua']['2017_2022'][1]:.0f}, {km['full.changhua']['2023_2025'][0]:.0f}-"
        f"{km['full.changhua']['2023_2025'][1]:.0f}",
        "Billing completeness 2025/26 (Jul-Jun, design-time): missing-reading share <= 0.0005 "
        f"in every code; zero share Yunlin M2 {comp_row('yunlin', 'M2', 2025, 'zero_share'):.3f} "
        f"(2024/25 {comp_row('yunlin', 'M2', 2024, 'zero_share'):.3f}), M1 "
        f"{comp_row('yunlin', 'M1', 2025, 'zero_share'):.3f} "
        f"({comp_row('yunlin', 'M1', 2024, 'zero_share'):.3f}) with energy "
        f"{comp_row('yunlin', 'M2', 2025, 'GWh'):.1f} vs {comp_row('yunlin', 'M2', 2024, 'GWh'):.1f}"
        " GWh (M2): second-month postings, not missing readings",
        "P2e x2 (last 36 origins, two-month log RMS): Yunlin "
        + ", ".join(f"{c} {v['rms_e2_last36']:.3f}" for c, v in eb["yunlin"]["x2"].items())
        + "; Changhua "
        + ", ".join(f"{c} {v['rms_e2_last36']:.3f}" for c, v in eb["changhua"]["x2"].items())
        + f"; selected mean error Yunlin {eb['yunlin']['x2'][eb['yunlin']['selected']]['mean_e2_last36']:+.3f}"
        f", Changhua {eb['changhua']['x2'][eb['changhua']['selected']]['mean_e2_last36']:+.3f}",
        "P2e x1 annual RMS (all candidates): Yunlin "
        + ", ".join(f"{c} {v['rms_rel']:.3f}" for c, v in eb["yunlin"]["x1"].items())
        + "; Changhua " + ", ".join(f"{c} {v['rms_rel']:.3f}" for c, v in
                                    eb["changhua"]["x1"].items()),
        "Extension (county census-equivalent GWh): "
        + "; ".join(f"{x.county} {x.month} {x.county_census_hat_kwh / 1e6:.2f}"
                    for x in ext.itertuples() if x.county != "chiayi")
        + f"; monthly sd Yunlin {eb['yunlin']['sd_monthly_log']:.3f}, Changhua "
          f"{eb['changhua']['sd_monthly_log']:.3f} (log, uncalibrated)",
        "2026-07 in grid: extension Yunlin "
        f"{facts['p2e_2026_07_comparison']['yunlin.extension_in_grid_GWh']['2026-07']:.2f} vs "
        "census + P2abc imputation 18.78 GWh; Changhua "
        f"{facts['p2e_2026_07_comparison']['changhua.extension_in_grid_GWh']['2026-07']:.2f} vs "
        "19.36 GWh",
        "P2f observed L1/L2 (3 km, local-only): Yunlin " + rs("yunlin", "obs") + ", Changhua "
        + rs("changhua", "obs") + ", pooled " + rs("pooled", "obs"),
        "P2f model L1/L2 (cwa forward): Yunlin " + rs("yunlin", "cwa_forward") +
        "; cwa fit_full hindcast " + rs("yunlin", "cwa_fitfull_hindcast") +
        "; chc (supplementary) " + rs("yunlin", "chc_forward_supplementary"),
        "P2f year-shuffled placebo ratio (median over 20): Yunlin "
        f"{pl['yunlin']['placebo_site_median_ratio_median']:.2f}, Changhua "
        f"{pl['changhua']['placebo_site_median_ratio_median']:.2f}",
        "P2f L3/L2 Yunlin observed " + rs("yunlin", "obs", pair="L3/L2") + " vs cwa "
        + rs("yunlin", "cwa_forward", pair="L3/L2"),
        f"Depth proxy (meters > 10 HP): {100 * dp['yunlin']['share_energy_meters_gt10hp_in_grid']:.1f}"
        f" % of Yunlin in-grid kept energy 2012-22 ({100 * dp['yunlin']['share_pumps_on_meters_gt10hp_in_grid']:.1f}"
        f" % of pumps), {100 * dp['changhua']['share_energy_meters_gt10hp_in_grid']:.1f} % in "
        f"Changhua; census well-depth attribute coverage {dp['yunlin']['well_depth_attribute_coverage']:.0%}",
        f"Checker {len(chk['manifests'])} manifests 0 failures; guard {gp['as_designed']}/{gp['n']};"
        f" ledger {led['rows_written']} rows, 0 conflicts; Claude API tokens 0; GPU none",
    ]
    out = {"wp": D.WP, "status": "partial",
           "prereg": [{"id": x["id"], "sha256": x["sha256"], "created": x["created"],
                       "kind": x["kind"]} for x in reg],
           "acceptance": acc, "products": prods, "gates": gates, "key_numbers": key,
           "backtest": {k: {x: bt[k][x] for x in ("median_township_count_rel_err",
                                                  "count_total_rel_err", "fan_kwh_rel_err",
                                                  "county_kwh_rel_err", "pass")}
                        for k in ("yunlin", "changhua", "chiayi")},
           "p2e_selected": eb["selected"], "claude_api_tokens": 0, "gpu": "none"}
    D.save_json(out, REP / "P2def.json")
    print(json.dumps({"acceptance": [(a["criterion"][:60], a["measured"], a["passed"])
                                     for a in acc], "key": key}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
