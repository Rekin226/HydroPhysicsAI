"""Update the W1 report with the 2026 extension (resume round of 2026-10-12, evening).

    python -I dataprep_staging/w1_canal/ext_report.py

The first W1 report (written by report.py from full/v1) is kept as reports/W1.r1.md and
W1.r1.json; this script merges the extension (full/v2, W1 step 6, the forward-archive probes)
into a new W1.md / W1.json. Numbers are read from full/v2's validation.json and
reports/W1_evidence/ext2026_verify.json; the narrative was written after the results were
known and says so where it interprets them.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal import ext_config as E  # noqa: E402
from w1_canal.ext_prereg import PREREG_ID  # noqa: E402

REPORTS = DATAPREP / "reports"
EVID = REPORTS / "W1_evidence"
V2 = C.product_dir_base("full") / "v2"
PV2 = f"{C.PRODUCT_BASE}/full/v2"
EXT_CODE = ["ext_config.py", "ext_fetch.py", "ext_model.py", "ext_prereg.py", "ext_build.py",
            "ext_publish.py", "ext_verify.py", "ext_report.py"]


def f3(x):
    return f"{x:.3f}"


def extra_rows(ver: dict) -> list[dict]:
    rain = f"data_fetch_api/new_measurements/{E.L0_SOURCE_ID}"
    comp = rel_to_repo(C.product_dir_base("full") / "v1" / "inflow_composite_daily.parquet")
    jj = rel_to_repo(C.JIJI_CSV)
    rows = []

    def add(n, s, r, note=""):
        rows.append({"number_id": n, "source": s, "role": r, "product_version": PV2,
                     "note": note})

    for d in ("chc", "yl"):
        n = f"W1.full_v2.ext2026.post_hoc_floor.{d}"
        add(n, rain, "input", "post hoc sensitivity, not in the product")
        add(n, comp, "input", "historical monthly minimum inflow 2012-2025")
        add(n, jj, "input", "beta, district scales, R of 2025")
    n = "W1.full_v2.ext2026.extrapolation"
    add(n, rain, "input")
    add(n, comp, "validator", "2012-2025 monthly inflow range")
    n = "W1.full_v2.check.manifest_and_guard"
    add(n, rel_to_repo(V2 / "MANIFEST.json"), "input")
    n = "W1.full_v2.check.identical_2012_2025"
    add(n, rel_to_repo(C.product_dir_base("full") / "v1" / "district_monthly.csv"), "validator")
    add(n, rel_to_repo(V2 / "district_monthly.csv"), "input")
    n = "W1.full_v2.forward_archive_probes"
    add(n, rel_to_repo(EVID / "ext2026_verify.json"), "input", "HTTP status codes only")
    n = "W1.full_v2.ext2026.look"
    add(n, "results/dataprep/preregistration/looks.jsonl", "input",
        "design-time look logged by ext_build.py after the pre-registration")
    return rows


CITED_EXT = ([f"W1.full_v2.ext2026.loo_annual_rmse.{m}.{d}" for d in ("chc", "yl")
              for m in ("rain_chain", "actual_inflow", "climatology", "persistence")]
             + [f"W1.full_v2.ext2026.{k}.{d}" for d in ("chc", "yl")
                for k in ("loo_boot_vs_climatology", "series_2026", "sd_2026",
                          "post_hoc_floor")]
             + ["W1.full_v2.ext2026.inflow_loo_rmse", "W1.full_v2.ext2026.rain_index_"
                "availability", "W1.full_v2.ext2026.extrapolation",
                "W1.full_v2.check.manifest_and_guard", "W1.full_v2.check.identical_2012_2025",
                "W1.full_v2.forward_archive_probes", "W1.full_v2.ext2026.look",
                f"W1.prereg.{PREREG_ID}"])


def main():
    old_json = REPORTS / "W1.json"
    old_md = REPORTS / "W1.md"
    r1j, r1m = REPORTS / "W1.r1.json", REPORTS / "W1.r1.md"
    if not r1j.exists():
        shutil.copy2(old_json, r1j)
        shutil.copy2(old_md, r1m)
    r1 = json.loads(r1j.read_text())
    val = json.loads((V2 / "validation.json").read_text())
    ext = val["ext2026"]
    ver = json.loads((EVID / "ext2026_verify.json").read_text())
    ledger.append_rows(C.WP, extra_rows(ver))
    from common.ledger import read_all

    from w1_canal.report import CITED
    missing = ledger.unledgered(CITED + CITED_EXT)
    conflicts = ledger.conflicts(read_all())
    ct = ext["chain_test"]
    sc, cmp_ = ct["scores"], ct["compare"]
    rule = ext["shipping_rule"]
    s26 = ext["series_2026"]
    ph = ver["post_hoc"]
    fl = ph["floored_at_hist_min_inflow"]
    below = [k for k, r in ph["extrapolation"].items() if r["q_pred_below_hist_min"]]
    ex = ver["exactness"]
    gp = {g["use"]: g.get("opened") for g in ver["guard_probes"]}
    probes = ver["forward_archive_probes"]
    ey = ext["evaluation_years"]

    def bt(d):
        r = cmp_[d]["rain_chain_vs_climatology"]
        return f"{r['rmse_diff']:+.3f} [{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}]"

    acceptance = [a for a in r1["acceptance"]
                  if not a["criterion"].startswith("Manifests re-hash")]
    acceptance += [
        {"criterion": "Manifests re-hash (A0 checker) and loader guard enforces the uses "
                      "(preorigin/v1, full/v1, full/v2, new L0 SOURCE.json)",
         "threshold": "0 problems; forbidden uses refused",
         "measured": f"preorigin/v1 ok (48 files), full/v1 ok (49), full/v2 "
                     f"{ver['checker']['status']} ({ver['checker']['n_hashed']} files + the "
                     f"raw-rain input tree), SOURCE.json {ver['source_json']}; full/v2 opens "
                     f"for forcing_recorded_arm ({gp.get('forcing_recorded_arm')}) and "
                     f"refuses forcing_frozen, model_design, scoring_target, "
                     f"calibration_target",
         "passed": ver["checker"]["status"] == "ok" and ver["source_json"] == "ok"
                   and not any(gp.get(u) for u in ("forcing_frozen", "model_design",
                                                   "scoring_target", "calibration_target"))},
        {"criterion": "Step 6 shipping rule (pre-registered W1.ext2026_rain_runoff.v1, "
                      "design-time units): rain-driven chain's leave-one-year-out annual "
                      "RMSE below annual climatology in both districts",
         "threshold": "point RMSE(chain) < RMSE(climatology), chc and yl",
         "measured": f"chc {f3(sc['chc']['rain_chain']['rmse'])} vs "
                     f"{f3(sc['chc']['climatology']['rmse'])}; yl "
                     f"{f3(sc['yl']['rain_chain']['rmse'])} vs "
                     f"{f3(sc['yl']['climatology']['rmse'])} (10^8 m3, {len(ey)} evaluation "
                     f"years); ships {rule['ships']}",
         "passed": rule["ships"] == "rain_chain"},
        {"criterion": "Step 6 label: 90 % paired-bootstrap interval of the RMSE difference "
                      "(chain - climatology) below zero in both districts",
         "threshold": "upper bound < 0, chc and yl",
         "measured": f"chc {bt('chc')}; yl {bt('yl')}; sign test "
                     f"{cmp_['chc']['rain_chain_vs_climatology']['sign_wins']}/{len(ey)} and "
                     f"{cmp_['yl']['rain_chain_vs_climatology']['sign_wins']}/{len(ey)}; "
                     f"label '{rule['label']}'",
         "passed": all(rule["interval_below_zero"].values())},
        {"criterion": "Step 6 product full/v2: 2012-2025 identical to full/v1 (table, draws, "
                      "grid); 2026 months flag 6; no negative value or draw; twin mask "
                      "unchanged",
         "threshold": "all true",
         "measured": f"identical table/draws/grid {ex['monthly_2012_2025_identical_to_v1']}/"
                     f"{ex['draws_2012_2025_identical']}/{ex['grid_2012_2025_identical']}; "
                     f"{ex['rows_2026']} rows {ex['dates_2026'][0]}..{ex['dates_2026'][1]} "
                     f"flags {ex['flags_2026']}; negative values {ex['negative_values']}, "
                     f"draws {ex['negative_draws']}; mask {ex['mask_hash_unchanged']}",
         "passed": bool(ex["monthly_2012_2025_identical_to_v1"] and ex["draws_2012_2025_"
                                                                        "identical"]
                        and ex["grid_2012_2025_identical"] and ex["flags_2026"] == [6]
                        and ex["negative_values"] == 0 and ex["negative_draws"] == 0
                        and ex["mask_hash_unchanged"])},
        {"criterion": "Step 6: forward archive of the WRA real-time discharge and weir feeds "
                      "started",
         "threshold": "a first snapshot in L0 with SOURCE.json",
         "measured": "blocked from this VM: " + "; ".join(
             f"{p['url'].split('/')[2]} -> {p['http_code']} {p['content_type'] or ''}".strip()
             for p in probes) + " (no data endpoint reachable; a recurring job is also a "
                                "user decision)",
         "passed": False},
    ]
    summary = (
        "W1 rebuilt the monthly Jiji-weir canal deliveries (surface water v4) from L0: 14 "
        "yearbook PDFs parsed with pdftotext (123 station-years, monthly values from the "
        "daily table), a daily inflow composite (H075+H049+H064) with per-year gauge QC and "
        "donor swaps rescaled to the weir's published inflow, the supply-limited diversion "
        "indicator, proportional first-difference Denton benchmarking to the annual district "
        "totals (0 negative months, totals exact), 200 state-space posterior draws per "
        "district, and the v3 farmland allocation onto the twin grid. Products: "
        "surface_water_v4/preorigin/v1 (beta, demand scales and QC medians fitted on "
        "2012-2022; LUIMAP 2024 is the U3 exemption for the grid field only), full/v1 "
        "(2012-2025) and, new in this resume round, full/v2 = full/v1 plus Jan-Sep 2026 "
        "(plan step 6; recorded-forcing arm only). On the pre-registered Jan-May leave-one-"
        "year-out benchmark the pre-origin variant scores RMSE 0.359 / 0.339 x 10^8 m3 "
        "(Changhua / Yunlin) against climatology 0.661 / 0.563 and v3 0.839 / 1.062; all "
        "four 90 % paired-bootstrap intervals exclude zero, so the primary acceptance "
        "passes, narrowly (Yunlin vs climatology upper bound -0.002; sign tests not "
        "significant), on seven design-time-contaminated years. The full variant fails the "
        "same rule (2 of 4 intervals include zero) and ships labelled 'not distinguishable "
        "from climatology on the benchmark'. The supporting head check fails in one of four "
        "groups (Yunlin layer 1, -0.0087). Step 6: 16 upper-basin CODIS rain records were "
        f"fetched into L0 (259 station-years, 0 failures) and a pre-registered rainfall-"
        f"runoff chain (linear regression on rain lags 0-2, refitted beta, R of 2025) was "
        f"scored leave-one-year-out on {len(ey)} years: annual RMSE "
        f"{f3(sc['chc']['rain_chain']['rmse'])} / {f3(sc['yl']['rain_chain']['rmse'])} vs "
        f"climatology {f3(sc['chc']['climatology']['rmse'])} / "
        f"{f3(sc['yl']['climatology']['rmse'])}, so the chain ships (flag 6) but is labelled "
        f"'not distinguishable from climatology' (intervals {bt('chc')}, {bt('yl')}). "
        f"Post hoc, the linear spec predicts inflow below the 2012-2025 monthly minimum in "
        f"{len(below)} of 9 months ({', '.join(below)}), giving zero May-2026 deliveries. "
        "The forward archive of the WRA real-time feeds could not be started (no endpoint "
        "reachable from this VM), so status stays partial. No GPU, 0 Claude API tokens, "
        "nothing under hydrophysics/, tests/ or existing L0 written, nothing staged in git.")
    products = list(r1["products"]) + [
        {"path": rel_to_repo(V2), "layer": "L2 + L3 grid array",
         "manifest": rel_to_repo(V2 / "MANIFEST.json"),
         "allowed_uses": "forcing_recorded_arm, statistical_layer_history, "
                         "after_the_fact_analysis (2012-2025 as full/v1; 2026-01..2026-09 "
                         "flag 6, labelled '" + rule["label"] + "')",
         "fit_window_end": ext["extension_months"][1]},
        {"path": rel_to_repo(E.L0_DIR), "layer": "L0",
         "manifest": rel_to_repo(E.L0_DIR / "SOURCE.json"),
         "allowed_uses": "raw input for L1+ builders; untrusted data; CWA open data licence",
         "fit_window_end": "n/a (raw; 2011-01-01..2026-10-11, fetched 2026-10-12)"}]
    code_paths = list(r1["code_paths"]) + [f"dataprep_staging/w1_canal/{f}" for f in EXT_CODE]
    key_numbers = [k for k in r1["key_numbers"] if not k.startswith("Claude API")] + [
        f"Step 6 chain LOO annual RMSE ({len(ey)} years {ey[0]}-{ey[-1]} without 2014-16): "
        f"rain chain {f3(sc['chc']['rain_chain']['rmse'])}/"
        f"{f3(sc['yl']['rain_chain']['rmse'])}, yearbook-inflow indicator "
        f"{f3(sc['chc']['actual_inflow']['rmse'])}/{f3(sc['yl']['actual_inflow']['rmse'])}, "
        f"climatology {f3(sc['chc']['climatology']['rmse'])}/"
        f"{f3(sc['yl']['climatology']['rmse'])}, persistence "
        f"{f3(sc['chc']['persistence']['rmse'])}/{f3(sc['yl']['persistence']['rmse'])} "
        f"(10^8 m3, chc/yl)",
        f"Step 6 chain - climatology [90 % paired bootstrap]: chc {bt('chc')}, yl {bt('yl')}",
        f"Inflow regression monthly LOO RMSE {f3(ct['inflow_monthly_rmse_loo'])} vs "
        f"in-sample monthly climatology {f3(ct['inflow_monthly_rmse_climatology_insample'])}"
        f" x 10^8 m3; spec linear_lag2 in {ct['spec_counts'].get('linear_lag2', 0)}/{len(ey)}"
        f" folds",
        f"2026 Jan-Sep deliveries (shipped): Changhua {f3(s26['chc']['jan_sep_total'])}, "
        f"Yunlin {f3(s26['yl']['jan_sep_total'])} x 10^8 m3 (post hoc floor at the monthly "
        f"minimum inflow: {f3(fl['jan_sep_chc'])} / {f3(fl['jan_sep_yl'])})",
        "CODIS upper basin: 16 records, 259 station-years, 225 MB; 2-7 of 16 records valid "
        "mid-2014..2015 (19 months without index)",
        "Claude API tokens: 0; GPU: none"]
    deviations = []
    for dv in r1["deviations_from_plan"]:
        if dv.startswith("Plan step 6"):
            deviations += [
                "Plan step 6 was done in the resume round (2026-10-12 evening), not in the "
                "first build: full/v2 adds Jan-Sep 2026 to the full variant (full/v1 kept "
                "unchanged). It needed a new L0 fetch of 16 upper-basin CODIS records "
                "(data_fetch_api/new_measurements/cwa_codis_rain_upper_choushui/, SOURCE.json)"
                " via a read-only import of hydrophysics.twin.cwa_rain.",
                "Step 6 design choices (ext_config.py): a station rule written from the CWA "
                "station list (drainage areas of H075, H049, H064; Wu-River-side, divide and "
                "below-gauge stations excluded); normal-ratio rain index with area-weighted "
                "sub-basins; regression on rain lags 0-2 with calendar-month intercepts "
                "(linear or log-log by nested leave-one-year-out); R of 2026 = R of 2025; "
                "the plan's 'indicator's annual leave-one-year-out error' is replaced by the "
                "chain's own leave-one-year-out error (rain-predicted inflow, refitted beta), "
                "which is what a 2026 value carries.",
                "Two settings were revised after a data-availability check and before the "
                "pre-registration and any score: CODIS holds only 2-7 of the 16 records from "
                "mid-2014 to 2015, so station normals use calendar-month means and the chain "
                f"is scored on the {len(ey)} evaluation years whose months (with lags) all "
                f"have an index ({', '.join(map(str, ey))}), not on all 14. Before "
                "registering, the code was debugged on synthetic random rain with the real "
                "availability pattern (scores not printed; the debug run also computed the "
                "climatology, persistence and yearbook-inflow references on the real annual "
                "totals, unread); the real index was first scored after the registration.",
                "The plan's forward archive of the WRA real-time discharge and weir feeds was "
                "not started: fhy.wra.gov.tw WraApi returned 503, opendata.wra.gov.tw "
                "returned a bot-challenge HTML page, the Civil IoT SensorThings host failed "
                "TLS name verification (not bypassed); a recurring fetch job also needs a "
                "user decision.",
                "The plan's step-6 RMSE (0.89 / 1.28) is the yearbook-inflow indicator's; "
                f"on the {len(ey)} evaluation years that indicator scores "
                f"{f3(sc['chc']['actual_inflow']['rmse'])} / "
                f"{f3(sc['yl']['actual_inflow']['rmse'])}."]
        else:
            deviations.append(dv)
    issues = []
    for dv in r1["issues_for_downstream"]:
        if dv.startswith("2026 deliveries and the real-time archive"):
            continue
        if dv.startswith("The manifests list dataprep_staging/harness/coverage.py"):
            dv += (" full/v2 also lists hydrophysics/twin/cwa_rain.py and "
                   "field_observations.py (read-only imports) and harness/prereg.py; an edit "
                   "to any of them makes the checker flag full/v2 (provenance drift).")
        issues.append(dv)
    issues += [
        "full/v2 supersedes full/v1 for the recorded-forcing arm (identical 2012-2025 "
        "values, draws and grid; adds 2026-01..2026-09, flag 6). The 2026 months are an "
        f"extrapolation labelled '{rule['label']}': use them with their sd and draws, never "
        "for calibration or scoring (U1, U4).",
        "2026 caveat (post hoc, found after the results were known): the registered linear "
        "spec extrapolates poorly in dry months. Its predicted inflow falls below the "
        f"2012-2025 minimum for that month in {', '.join(below)}; May 2026 (rain index 161 "
        "mm vs 110-814 mm in 2012-2025) gets 2 cms, so the May delivery is 0 in both "
        "districts. Flooring the inflow at the monthly minimum would give Jan-Sep "
        f"{f3(fl['jan_sep_chc'])} / {f3(fl['jan_sep_yl'])} instead of "
        f"{f3(s26['chc']['jan_sep_total'])} / {f3(s26['yl']['jan_sep_total'])} x 10^8 m3 "
        "(May 0.567 / 0.386). A revised spec (for example floored or log-linear) would need "
        "a new pre-registration and full/v3; the draws partly cover the gap (fold residuals "
        "are added, so the draw mean differs from the value by up to "
        f"{f3(ex['district_draw_mean_vs_value_2026']['chc'])} / "
        f"{f3(ex['district_draw_mean_vs_value_2026']['yl'])}).",
        "The chain over-predicts Yunlin on average (annual bias "
        f"{sc['yl']['rain_chain']['bias']:+.3f}) and under-predicts Changhua "
        f"({sc['chc']['rain_chain']['bias']:+.3f}); the 2026 values carry those biases.",
        "Prospective test registered (W1.ext2026.prospective_inflow_2026, fresh units "
        "W1.inflow.2026-01..09): when the 2026 yearbook and Jiji table appear (about mid-"
        "2027), score inflow_predicted_monthly.csv against the yearbook composite and its "
        "2012-2025 monthly climatology. Until then there is no skill evidence for 2026.",
        "The CODIS record has 19 months without an index (2014-01, 2014-07..2015-12): too few "
        "of the 16 upper-basin records are valid (station renumbering). Any reuse of the "
        "index for 2014-16 needs another rain source (WRA gauges)."]
    md_ext = [
        "## Resume round (2026-10-12 evening): plan step 6, the 2026 extension", "",
        "The first W1 session's products and report were checked first: the three manifests "
        "re-hash, the registered code and inputs of W1_janmay_benchmark_2026-10-12 still "
        "match, and nothing in full/v1 or preorigin/v1 was changed. Step 6 was then built as "
        "a new version of the full variant.", "",
        f"- Pre-registration {PREREG_ID} (A1 tool, "
        f"{ext['preregistration']['created']}), written after the fetch and a data-"
        "availability check, before any regression or chain was scored. The look on the "
        "design-time units was logged in the A1 look log by ext_build.py.",
        f"- Rain index: {sum(len(v) for v in ext['rain_index']['members'].values())} CODIS "
        "records (mainstem 9, 陳有蘭溪 5, 水里溪 2), area weights "
        f"{ext['rain_index']['area_weights']}; months without index: "
        f"{len(ext['months_without_index_2011_11_on'])} (2014-01, 2014-07..2015-12).",
        f"- Evaluation years: {ey}.", "",
        "| method (annual district total, leave-one-year-out) | RMSE Changhua | bias Changhua "
        "| RMSE Yunlin | bias Yunlin |", "|---|---|---|---|---|"]
    for m in ("rain_chain", "actual_inflow", "climatology", "persistence"):
        a, b = sc["chc"][m], sc["yl"][m]
        md_ext.append(f"| {m} | {f3(a['rmse'])} | {a['bias']:+.3f} | {f3(b['rmse'])} | "
                      f"{b['bias']:+.3f} |")
    md_ext += ["", "| comparison | district | RMSE difference | 90 % paired bootstrap | sign "
               "test |", "|---|---|---|---|---|"]
    for d in ("chc", "yl"):
        for k, r in cmp_[d].items():
            md_ext.append(f"| {k} | {d} | {r['rmse_diff']:+.3f} | [{r['boot_lo']:+.3f}, "
                          f"{r['boot_hi']:+.3f}] | {r['sign_wins']}/{r['sign_n']}, p = "
                          f"{r['sign_p_two_sided']:.3f} |")
    md_ext += ["", f"Shipping rule: {rule['ships']}; label: {rule['label']}.", "",
               "| 2026 month | Changhua value | sd | Yunlin value | sd | inflow pred (10^8 m3) "
               "| below 2012-25 monthly min |", "|---|---|---|---|---|---|---|"]
    for k, (mo, r) in enumerate(ph["extrapolation"].items()):
        md_ext.append(f"| {mo} | {f3(s26['chc']['value'][k])} | {f3(s26['chc']['sd'][k])} | "
                      f"{f3(s26['yl']['value'][k])} | {f3(s26['yl']['sd'][k])} | "
                      f"{f3(r['q_pred'])} | {'yes' if r['q_pred_below_hist_min'] else ''} |")
    md_ext += ["", "Interpretation (post hoc): the chain beats climatology on point RMSE but "
               "not distinguishably; its dry-month extrapolation is the weak point (see "
               "issues). The 2026 values are forcing for the recorded arm only.", ""]
    out = dict(r1)
    out.update({"status": "partial", "date": "2026-10-12", "summary": summary,
                "acceptance": acceptance, "products": products, "code_paths": code_paths,
                "key_numbers": key_numbers, "deviations_from_plan": deviations,
                "issues_for_downstream": issues,
                "preregistration": [r1["preregistration"],
                                    f"results/dataprep/preregistration/{PREREG_ID}.json"],
                "evidence": list(r1["evidence"]) + [
                    rel_to_repo(EVID / "ext2026_verify.json"),
                    rel_to_repo(V2 / "validation.json"), rel_to_repo(V2 / "QC_REPORT.md")],
                "ledger": {"fragment": "results/dataprep/ledger/rows/W1.csv",
                           "unledgered_cited_numbers": missing,
                           "conflicts_all_fragments": len(conflicts)},
                "previous_report": ["results/dataprep/reports/W1.r1.md",
                                    "results/dataprep/reports/W1.r1.json"],
                "claude_api_tokens": 0, "gpu": "none"})
    old = r1m.read_text()
    _head, _, rest = old.partition("\n## Products")
    md = ["# WP W1: monthly canal deliveries, surface water v4 (report, 2026-10-12, "
          "updated in the resume round)", "", f"Status: **partial**. {summary}", "",
          "## Products", ""]
    for p in products:
        md.append(f"- `{p['path']}` ({p['layer']}); allowed: {p['allowed_uses']}; "
                  f"fit_window_end {p['fit_window_end']}.")
    md += ["", "## Acceptance", "", "| criterion | threshold | measured | pass |",
           "|---|---|---|---|"]
    for a in acceptance:
        md.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                  f"{'yes' if a['passed'] else 'NO'} |")
    sec = rest.split("\n## ")
    keep = [s for s in sec if s.startswith("Jan-May benchmark")]
    md += [""] + ["## " + s.rstrip() + "\n" for s in keep] + md_ext
    md += ["## Deviations from the plan", ""] + [f"- {x}" for x in deviations]
    md += ["", "## Issues for downstream WPs", ""] + [f"- {x}" for x in issues]
    md += ["", "## Key numbers", ""] + [f"- {x}" for x in key_numbers]
    md += ["", "## Provenance", "",
           f"- Pre-registrations: `{r1['preregistration']}`; `results/dataprep/"
           f"preregistration/{PREREG_ID}.json`; looks in `results/dataprep/preregistration/"
           "looks.jsonl`.",
           f"- Ledger: `results/dataprep/ledger/rows/W1.csv`; cited numbers without a row: "
           f"{missing}; conflicts across all fragments: {len(conflicts)}.",
           "- Evidence: " + ", ".join(f"`{e}`" for e in out["evidence"]),
           "- Previous report (first build): `results/dataprep/reports/W1.r1.md`, "
           "`W1.r1.json`.",
           "- Code: " + ", ".join(f"`{c}`" for c in code_paths),
           "- Claude API tokens: 0. GPU: none. Nothing under hydrophysics/, tests/ or "
           "existing L0 written; nothing staged in git."]
    (REPORTS / "W1.md").write_text("\n".join(md) + "\n")
    with open(REPORTS / "W1.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(out["status"], out["ledger"], len(acceptance))


if __name__ == "__main__":
    main()
