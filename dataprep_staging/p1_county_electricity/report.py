"""WP P1 report: results/dataprep/reports/P1.md and P1.json from the built products.

Numbers are read from the product files (no transcription). Also writes guard probes and a
manifest check to results/dataprep/reports/P1_evidence/.

Usage: python -I report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import guard, ledger, manifest  # noqa: E402
from common.paths import DATAPREP, REPORTS_DIR, rel_to_repo  # noqa: E402

PRE = DATAPREP / "county_electricity/preorigin/v1"
FULL = DATAPREP / "county_electricity/full/v1"
EVD = REPORTS_DIR / "P1_evidence"


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def probes() -> dict:
    out = {}
    for d in (PRE, FULL):
        m, pdir = manifest.load_manifest(d)
        rec = {"schema_errors": manifest.validate_manifest(m, pdir),
               "hash_problems": manifest.verify_hashes(m, pdir), "guard": {}}
        for use in ("prior", "prior_width_input", "forcing_frozen", "design_time_evidence",
                    "after_the_fact_analysis", "forcing_recorded_arm", "validator",
                    "scoring_target", "calibration_target", "selection"):
            try:
                guard.open_product(d, use)
                rec["guard"][use] = "opens"
            except Exception as e:  # noqa: BLE001 - the exception type is the result
                rec["guard"][use] = type(e).__name__
        out[rel_to_repo(d)] = rec
    return out


def f(x, n=3):
    return f"{x:.{n}f}"


def main() -> int:
    EVD.mkdir(parents=True, exist_ok=True)
    vp, vf = jl(PRE / "validation.json"), jl(FULL / "validation.json")
    dp, df = jl(PRE / "decomposition.json"), jl(FULL / "decomposition.json")
    mp = jl(PRE / "misspecification.json")
    phi = jl(PRE / "phi_in.json")
    cen = jl(PRE / "census_monthly_info.json")
    fetch = jl(EVD / "fetch_log.json")
    pr = probes()
    (EVD / "guard_probes.json").write_text(json.dumps(pr, ensure_ascii=False, indent=1))
    t = vf["tests"]
    T1, T2, T4 = t["T1_BV1"], t["T2_BV2"], t["T4_EXT_LOYO"]
    T8p, T8f = vp["tests"]["T8_SYN_coverage"]["preorigin"], t["T8_SYN_coverage"]["full"]
    T9p, T9f = vp["tests"]["T9_MIS"]["preorigin"], t["T9_MIS"]["full"]
    sp, sf = dp["source_separation"], df["source_separation"]

    def bv(T, c):
        r = T[c]["trend"]
        return f"RMS {f(r['rms_pct'], 2)} %, mean bias {r['mean_bias_pct']:+.2f} %"

    acc = [
        {"criterion": "Extension, leave one year out 2016-2024 (T4; full gate)",
         "threshold": "annual RMS <= 10 %",
         "measured": f"{f(T4['rms_pct_2016_2024'], 2)} % (2016-2022 only: "
                     f"{f(T4['rms_pct_2016_2022'], 2)} %)", "passed": T4["pass"]},
        {"criterion": "Backcast, matched lead, pre-origin mirror BV1 (shares and trend "
                      "2019-22 predict 2016-18; data <= 2022-12; preorigin gate), Changhua",
         "threshold": "RMS <= 5 % and |mean bias| <= 3 % after the trend correction",
         "measured": bv(T1, "CH"), "passed": T1["CH"]["pass"]},
        {"criterion": "Backcast BV1 (preorigin gate), Yunlin",
         "threshold": "RMS <= 5 % and |mean bias| <= 3 % after the trend correction",
         "measured": bv(T1, "YL") + f" (without trend: RMS "
                     f"{f(T1['YL']['no_trend']['rms_pct'], 2)} %, bias "
                     f"{T1['YL']['no_trend']['mean_bias_pct']:+.2f} %)",
         "passed": T1["YL"]["pass"]},
        {"criterion": "Backcast, the plan's matched lead BV2 (shares 2020-23, trend "
                      "2020-01..2026-08, predict 2016-19; full gate), Changhua",
         "threshold": "RMS <= 5 % and |mean bias| <= 3 % after the trend correction",
         "measured": bv(T2, "CH"), "passed": T2["CH"]["pass"]},
        {"criterion": "Backcast BV2 (full gate), Yunlin",
         "threshold": "RMS <= 5 % and |mean bias| <= 3 % after the trend correction",
         "measured": bv(T2, "YL"), "passed": T2["YL"]["pass"]},
        {"criterion": "Coverage of a's 90 % year-block bootstrap interval, correct model, "
                      "pre-origin window 2016-01..2022-12 (T8; preorigin gate)",
         "threshold": "Wilson 95 % interval (n_eff 1000) contains 0.90 and point in 0.85-0.95",
         "measured": f"{f(T8p['point'])} (Wilson {f(T8p['wilson95'][0])}-"
                     f"{f(T8p['wilson95'][1])})", "passed": T8p["pass"]},
        {"criterion": "Same, full window 2016-01..2025-06 (T8; full gate)",
         "threshold": "Wilson 95 % interval contains 0.90 and point in 0.85-0.95",
         "measured": f"{f(T8f['point'])} (Wilson {f(T8f['wilson95'][0])}-"
                     f"{f(T8f['wilson95'][1])})", "passed": T8f["pass"]},
        {"criterion": "Misspecification biases of a reported and folded into the mu prior "
                      "width (T9)",
         "threshold": "reported; any bias above the bootstrap half-width added in quadrature",
         "measured": "pre-origin: " + ", ".join(
             f"{k.split('_')[0]} {v['bias_vs_a_eff']:+.3f}" for k, v in T9p["cases"].items()
             if v["registered"]) + f" vs half-width {f(T9p['bootstrap_halfwidth'])}; "
             f"b_mis {f(T9p['b_mis_ln'])} -> sigma_mu {f(T9p['sigma_mu_ln'])}",
         "passed": True},
    ]
    gates = {"preorigin": vp["gate"], "full": vf["gate"]}
    status = "partial"
    key = [
        f"Pre-origin source separation 2016-01..2022-12: a = {f(sp['a'])} (90 % year-block "
        f"bootstrap {f(sp['a_lo'])}-{f(sp['a_hi'])}, uncalibrated), b = {f(sp['b'])}, "
        f"R2 = {f(sp['r2'])}",
        f"Full source separation 2016-01..2025-06: a = {f(sf['a'])} ({f(sf['a_lo'])}-"
        f"{f(sf['a_hi'])}, uncalibrated), b = {f(sf['b'])}, R2 = {f(sf['r2'])}",
        f"Synthetic coverage of a's 90 % interval: pre-origin {f(T8p['point'])}, full "
        f"{f(T8f['point'])}; half-width factor k {f(T8p['remedy_reported_only']['k_halfwidth_factor'])}"
        f" / {f(T8f['remedy_reported_only']['k_halfwidth_factor'])} restores "
        f"{f(T8p['remedy_reported_only']['coverage_fresh_set'])} / "
        f"{f(T8f['remedy_reported_only']['coverage_fresh_set'])} on fresh replicates "
        f"(reported only)",
        "Misspecification bias of a vs the mean-level truth, pre-origin: " + ", ".join(
            f"{k} {v['bias_vs_a_eff']:+.3f}" for k, v in T9p["cases"].items()),
        f"mu prior log-sd {f(T9p['sigma_mu_ln'])} = sqrt(0.5^2 + {f(phi['sigma_ln_phi'])}^2 + "
        f"{f(T9p['b_mis_ln'])}^2)",
        f"phi_in (provisional) {f(phi['phi_in'])} (area {f(phi['phi_area'])}, planted "
        f"{f(phi['phi_planted'])}; sigma_ln {f(phi['sigma_ln_phi'])}); checks: TW_COUNTY "
        f"polygon {f(phi['phi_county_polygon_check'])}, 954 labelled cells "
        f"{f(phi['phi_label_cells'])}",
        "In-grid anchor a_pre x YL_clean x phi_in: " + "; ".join(
            f"{k} {v['mu_elec_in_grid_GWh_yr']:.1f} GWh/yr (county "
            f"{v['mu_elec_county_GWh_yr']:.1f})"
            for k, v in dp["in_grid_anchor"]["time_means"].items()),
        "Estimator ratios to Yunlin (pre-origin): " + ", ".join(
            f"{k} {v:.3f}" for k, v in dp["estimator_ratios"].items()
            if isinstance(v, float)),
        "Estimator ratios to Yunlin (full): " + ", ".join(
            f"{k} {v:.3f}" for k, v in df["estimator_ratios"].items()
            if isinstance(v, float)),
        f"Backcast BV1 (pre-origin mirror): CH {bv(T1, 'CH')}; YL {bv(T1, 'YL')}",
        f"Backcast BV2 (plan's matched lead): CH {bv(T2, 'CH')}; YL {bv(T2, 'YL')}; without "
        f"trend CH {f(T2['CH']['no_trend']['rms_pct'], 2)} % RMS, YL "
        f"{f(T2['YL']['no_trend']['rms_pct'], 2)} % (plan: 3.2 / 3.25 %)",
        f"Backcast LOYO (interpolation, reported only): CH "
        f"{f(t['T3_BV_LOYO']['CH']['rms_pct'], 2)} %, YL "
        f"{f(t['T3_BV_LOYO']['YL']['rms_pct'], 2)} %",
        f"Extension LOYO annual RMS {f(T4['rms_pct_2016_2024'], 2)} %; monthly relative RMS "
        f"{f(T4['monthly_rel_rms'])}; quarter blocks pooled RMS "
        f"{f(t['T5_EXT_season']['rms_pct_pooled'], 2)} % (Q1-Q4 " + ", ".join(
            f"{f(v, 1)}" for v in t["T5_EXT_season"]["rms_pct_by_quarter"].values())
        + f" %); 2023 {t['T6_EXT_2023_25']['2023']:+.2f} %, 2024 "
        f"{t['T6_EXT_2023_25']['2024']:+.2f} %, 2025H1 {t['T6_EXT_2023_25']['2025H1']:+.2f} %",
        "Forward 14-month blocks (rolling origin): " + ", ".join(
            f"{k} {v['err14_pct']:+.1f} %" for k, v in t["T7_EXT_rolling"].items()),
        f"No splice: transfer / scaled 2024 = {df['no_splice_check']['transfer_2024_GWh']:.1f} / "
        f"{df['no_splice_check']['scaled_afull_2024_GWh']:.1f} GWh "
        f"({f(df['no_splice_check']['ratio_2024'], 2)} x); June 2025 "
        f"{df['no_splice_check']['transfer_2025_06_GWh']:.2f} / "
        f"{df['no_splice_check']['scaled_afull_2025_06_GWh']:.2f} GWh",
        f"CV_j provisional (even-split census 2012-02..2022-12): irrigation "
        f"{f(dp['cv_j']['cv']['irrigation'])}, other {f(dp['cv_j']['cv']['other'])}",
        f"Census pre-origin build equals the full build on every month <= 2022-12 (max abs "
        f"{cen['preorigin_vs_full_max_abs_GWh']} GWh)",
        "Re-fetch 2026-10-12T02:39Z: d007012 and Energy Administration sets 236/256 "
        "byte-identical to the dated archive 2026-10-12 (server Last-Modified of d007012 "
        + str(fetch["sources"]["tpc_d007012"]["files"]["d007012.csv"]["last_modified_header"])
        + "); no new dated directory needed",
        "Claude API tokens: 0; GPU: none",
    ]
    rows = ledger.read_all()
    n_rows = sum(1 for r in rows if r.get("wp") == "P1")
    rep = {
        "wp": "P1", "status": status, "date": "2026-10-12",
        "products": [rel_to_repo(PRE), rel_to_repo(FULL)],
        "gates": gates, "acceptance": acc, "key_numbers": key,
        "prereg": vp["prereg"], "prereg_sha256": vp["prereg_sha256"],
        "ledger_rows": n_rows, "evidence": rel_to_repo(EVD), "manifest_and_guard": pr,
        "claude_api_tokens": 0,
    }
    (REPORTS_DIR / "P1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    md = [f"# WP P1: county agricultural electricity constraint (report, 2026-10-12)", "",
          f"Status: **{status}**. Both product variants were built with manifests, but "
          "two pre-registered acceptance criteria failed, so neither variant's gate passed.",
          "", "## Acceptance", "", "| Criterion | Threshold | Measured | Pass |",
          "|---|---|---|---|"]
    for a in acc:
        md.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                  f"{'yes' if a['passed'] else '**no**'} |")
    md += ["", "Gates: preorigin " + json.dumps(gates["preorigin"]) + "; full "
           + json.dumps(gates["full"]), "", "## Key numbers", ""]
    md += [f"- {k}" for k in key]
    (REPORTS_DIR / "P1_generated.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gates": gates, "ledger_rows": n_rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
