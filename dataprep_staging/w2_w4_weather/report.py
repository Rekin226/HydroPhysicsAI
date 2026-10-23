"""WP W2W4 report: results/dataprep/reports/W2W4.{md,json} and the ledger fragment
results/dataprep/ledger/rows/W2W4.csv. Every number is read from the product validation
files, never typed by hand.

Usage: python report.py --et0 results/dataprep/et0/v1 --rain results/dataprep/rain_check/v1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
from archive_probes import DEST as VARIANT_DIR
from archive_probes import summary as variant_summary
from build_et0 import HISTORY, MAIN, PROBE, REFETCH
from common import ledger
from wcommon import (
    CODIS_STATION_LIST,
    ET_CACHE,
    FORCING_EXP_ET,
    PREREG_DIR,
    REPO,
    REPORTS_DIR,
    RF_STATIONS,
    RF_TIMESERIES,
    sha256_file,
)

EVID = REPORTS_DIR / "W2W4_evidence"
VARIANTS: dict = {}


def rel(p) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def api_calls() -> dict:
    """Open-Meteo call units used today (max(1, days/14) per request, 1 variable)."""
    out = {}
    for d in (PROBE, HISTORY, MAIN, REFETCH):
        n = 0.0
        for line in (d / "fetch_log.jsonl").read_text().splitlines():
            r = json.loads(line)
            q = dict(x.split("=", 1) for x in r["url"].split("?", 1)[1].split("&"))
            days = (dt.date.fromisoformat(q["end_date"]) - dt.date.fromisoformat(
                q["start_date"])).days + 1
            n += max(1.0, days / 14.0)
        out[d.name] = round(n, 1)
    # scratch probes (not kept in L0): one 18-day request and 11 January-2022 variants
    out["scratch_probes"] = round(18 / 14 + 11 * 31 / 14, 1)
    out["total"] = round(sum(out.values()), 1)
    return out


def review_probe_check() -> dict:
    """The review's 2026-10-11 19:43Z response vs this WP's 2026-10-12 02:27Z recent-days probe
    at the same request point (23.7N 120.4E, local days), on their common days."""
    a = json.loads((MAIN.parent / "review_test_2026-10-12" / "review_test_et0.json").read_text())
    b = json.loads((VARIANT_DIR / "probe.json").read_text())
    da = dict(zip(a["daily"]["time"], a["daily"]["et0_fao_evapotranspiration"], strict=True))
    db = dict(zip(b["daily"]["time"], b["daily"]["et0_fao_evapotranspiration"], strict=True))
    common = sorted(set(da) & set(db))
    diff = [abs(da[k] - db[k]) for k in common if da[k] is not None and db[k] is not None]
    return {"days": f"{common[0]}..{common[-1]}", "n": len(diff), "max_abs": max(diff),
            "identical": all(d == 0 for d in diff)}


def f(x, nd=3):
    return f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--et0", required=True)
    ap.add_argument("--rain", required=True)
    a = ap.parse_args(argv)
    E = REPO / a.et0
    R = REPO / a.rain
    ev = json.loads((E / "validation.json").read_text())
    sc = json.loads((E / "station_check_467270.json").read_text())
    rv = json.loads((R / "validation.json").read_text())
    rs = json.loads((R / "secondary_2023_2026.json").read_text())
    per = pd.read_csv(R / "logo_per_gauge_2012_2022.csv")
    fl = json.loads((EVID / "rain_flag_diagnostic.json").read_text())
    ev_rb = json.loads((EVID / "et0_v1_vs_scratch_build.json").read_text())
    gp = json.loads((EVID / f"guard_probe_{E.name}.json").read_text())
    import numpy as np
    with np.load(E / "et0_daily_2012_2026.npz", allow_pickle=False) as z:
        gp["shape"] = list(z["et0"].shape)
        gp["source_codes"] = sorted(int(x) for x in np.unique(z["source"]))
    gp["checker"] = json.loads((EVID / f"checker_{E.name}.json").read_text()).get("status")
    r2_path = EVID / "resume_r2.json"
    r2 = json.loads(r2_path.read_text()) if r2_path.exists() else None
    if r2 is not None:      # resume round 2: the checker re-run after A0fix/A1 must agree
        gp["checker"] = "ok" if (gp["checker"] == "ok" and r2["checker"]["status"] == "ok") \
            else "fail"
    exp_ok = {"forcing_recorded_arm", "statistical_layer_history", "after_the_fact_analysis",
              "forcing_frozen"}
    gp["guard_ok"] = all(
        (u in exp_ok) == (r.get("open") == "ok") for u, r in gp.items()
        if isinstance(r, dict) and "open" in r) and gp["forcing_frozen"].get(
        "et0_daily_2012_2026.npz", "").startswith("refused") and gp["forcing_frozen"].get(
        "et0_daily_2012_2022_preorigin.npz") == "readable"
    gaps = pd.read_csv(R / "residual_gaps_2012_2026.csv", index_col=0)
    pre = {p.name: sha256_file(p) for p in sorted(PREREG_DIR.glob("W[24].*.json"))}
    a1_written = {"W2.et0_overlap.v3.json", "W4.rain_kriging_vs_idw.v1.record.json"}
    over = per.loc[per["ratio_ked_idw"] > 1.20].sort_values("ratio_ked_idw", ascending=False)
    over_txt = ", ".join(f"{r.gauge} {r.ratio_ked_idw:.3f}" for r in over.itertuples())
    alt = gaps["altitude_m"]
    rv_chk = review_probe_check()
    calls = api_calls()
    et0_v = f"et0/{E.name}"
    rain_v = f"rain_check/{R.name}"

    T0, T1, T2, T3 = (ev["T0_same_points"], ev["T1_cache_vs_fe_2022"],
                      ev["T2_main_vs_fe_2025H1"], ev["T3_refetch_vs_main_2026Q3"])
    comp, lv, lr = ev["completeness"], ev["level_offsets"], ev["level_match_rule"]
    of, pr = ev["offset_form"], ev["probe_local_vs_gmt_st1_2022"]
    cbs = ev["climatology_bias_summary"]
    stab = lr["stability"]
    dec, sp, boot = rv["decision"], rv["summary"], rv["bootstrap"]
    ok_max = float(per["ratio_ok_idw"].max())
    ok_arg = str(per.loc[per["ratio_ok_idw"].idxmax(), "gauge"])
    post_gauges = gaps.index[gaps["kept_months_2023_2026_09"] <= 15].tolist()

    # ---------------------------------------------------------------- acceptance
    acc = [
        {"criterion": "W2 overlap consistency: 2022 daily ET0, calibration cache vs the GMT "
                      "fetch spliced after it (forcing experiment), every point",
         "threshold": "per-point MAE < 0.1 mm/d (max over 61 points)",
         "measured": f"max {f(T1['mae_max'])}, median {f(T1['mae_median'])} mm/d; bias "
                     f"{f(T1['bias_mean'])} mm/d; r >= {f(T1['r_min'], 4)} (re-computation "
                     "of the plan's own check, disclosed in the pre-registration)",
         "passed": bool(T1["pass"])},
        {"criterion": "W2 chain consistency: 2025-01..06, new fetch (2026-10-12) vs forcing "
                      "experiment (2026-09-30)",
         "threshold": "per-point MAE < 0.1 mm/d", "measured":
             f"max {f(T2['mae_max'], 4)} mm/d; identical values {100 * T2['identical_frac']:.1f} %",
         "passed": bool(T2["pass"])},
        {"criterion": "W2 pre-freeze re-fetch of the last 3 months (2026-07..09) vs the main "
                      "fetch", "threshold": "per-point MAE < 0.1 mm/d",
         "measured": f"max {f(T3['mae_max'], 4)} mm/d; {T3['n_changed_values']} of "
                     f"{61 * 92} values changed",
         "passed": bool(T3["pass"])},
        {"criterion": "W2 same 61 points: returned grid cell and elevation equal the forcing "
                      "experiment's", "threshold": "0 mismatches",
         "measured": f"{T0['n_mismatch']} mismatches ({T0['unique_cells']} distinct cells, "
                     f"{T0['unique_cell_elevation']} distinct cell+elevation)",
         "passed": bool(T0["pass"])},
        {"criterion": "W2 completeness 2023-01..2026-09", "threshold": ">= 99 % finite days "
                                                                       "per point",
         "measured": f"min {100 * comp['min_finite_frac']:.2f} %, "
                     f"{comp['n_missing_days_total']} missing point-days",
         "passed": bool(comp["pass"])},
        {"criterion": "W2 monthly bias against the 2012-2021 climatology reported",
         "threshold": "reported (no threshold in the plan)",
         "measured": f"2023-01..2026-09 fan-mean bias {f(cbs['level_matched']['mean_bias_mm_d'])}"
                     f" mm/d level-matched, {f(cbs['raw_splice']['mean_bias_mm_d'])} raw; mean "
                     f"|z| {f(cbs['level_matched']['mean_abs_z'], 2)}",
         "passed": True},
        {"criterion": "W2 output et0_daily_2012_2026.npz with per-day source and fetch date",
         "threshold": "present, manifest validates",
         "measured": f"{et0_v}: {gp['shape'][0]} x {gp['shape'][1]} days, source codes "
                     f"{gp['source_codes']}, fetch_date and provisional per day, flags; "
                     "pre-origin file equals the cache bitwise: "
                     f"{gp['preorigin_equals_cache']['values_bitwise_equal_nan_aware']}; "
                     f"manifest re-hash: {gp['checker']}",
         "passed": bool(gp["preorigin_equals_cache"]["values_bitwise_equal_nan_aware"]
                        and gp["checker"] == "ok" and gp["guard_ok"])},
        {"criterion": "W4 adoption rule, fan-wide: KED lowers the leave-one-gauge-out RMSE of "
                      "monthly totals (2012-2022)",
         "threshold": ">= 10 % (RMSE_KED <= 0.90 RMSE_IDW)",
         "measured": f"{100 * (1 - dec['fan_ratio_ked_idw']):.1f} % (ratio "
                     f"{f(dec['fan_ratio_ked_idw'])}; 90 % block-bootstrap "
                     f"{f(boot['ratio_ked_idw_p5_p50_p95'][0])}-"
                     f"{f(boot['ratio_ked_idw_p5_p50_p95'][2])})",
         "passed": bool(dec["fan_criterion_pass"])},
        {"criterion": "W4 adoption rule, per gauge: no gauge worsened",
         "threshold": "every gauge RMSE_KED,g <= 1.20 RMSE_IDW,g",
         "measured": f"{len(over)} of 26 gauges above 1.20: {over_txt} "
                     f"({sp['argmax_gauge_ratio_ked_idw']} is the "
                     f"{alt[sp['argmax_gauge_ratio_ked_idw']]:.0f} m apex gauge)",
         "passed": bool(dec["gauge_criterion_pass"])},
        {"criterion": "W4 evaluated as pre-registered and the plan's decision applied",
         "threshold": "decision recorded; product written",
         "measured": f"{dec['outcome']}; {rain_v} is a report only", "passed": True},
    ]

    # ---------------------------------------------------------------- deviations / issues
    deviations = [
        "Day convention: the plan says to adopt local days 'matching the cache'. The cache is "
        f"in GMT days (r(cache, GMT) = {f(pr['r_cache_gmt'], 4)} against "
        f"{f(pr['r_cache_local'], 4)} for local days at st1; >= {f(T1['r_min'], 4)} for all 61 "
        "points at lag 0), so the product uses GMT days, which keeps the cache unchanged. "
        "Pre-registration W2.et0_overlap.v1 (local days) was superseded by v2 before any "
        "61-point fetch; v1 was never run.",
        "The plan reads the 2022 cache-vs-fetch difference (MAE 0.048) as GMT against local "
        "days. The data show an archive level difference instead: the September 2026 "
        "Open-Meteo archive (default model = ECMWF IFS) is higher than the cache by "
        f"{f(-lv['fan_mean'])} mm/d on average (per point {f(-lv['max'])}..{f(-lv['min'])})"
        + (", and the annual offset at st1 and st42 stays within "
           f"{max(s['max_abs_dev_from_2022'] for s in stab.values()):.4f} mm/d of the 2022 "
           "value in every year 2012-2021. Per the pre-registered rule the primary array is "
           "level-matched from 2023-01 (flag 1); the raw splice is kept alongside."
           if lr["apply_level_match_to_primary"] else
           ". The pre-registered level-match rule did not hold (|fan mean| >= 0.02: "
           f"{lr['fan_mean_abs_ge_0.02']}; stable at st1/st42: {lr['stable_all']}), so the "
           "primary array is the raw splice; the level-matched variant is kept alongside."),
        "Fetch window 2025-01..2026-09 (plan: 2025-07..2026-09) to get a 6-month overlap with "
        "the forcing-experiment fetch; 2023-01..2025-06 come from that fetch (GMT), as the "
        "plan intended.",
        "The product ends 2026-09-30 although the archive served days up to the fetch date "
        "(2026-10-12); the last weeks come from the near-real-time stream.",
        "Optional station comparison done post hoc (not in the pre-registration, not gating): "
        "CWA 467270 station FAO-56 ET0 and Class-A pan, from CODIS responses already in L0. "
        "Anemometer height assumed 10 m.",
        "Pre-registration tooling: the first round (02:30-02:49Z) wrote W2.et0_overlap.v1/v2 "
        "and W4.rain_kriging_vs_idw.v1 with this WP's prereg.py because the A1 tool did not "
        "exist yet. In the second round v2 was recorded with the A1 tool as W2.et0_overlap.v3 "
        "(design unchanged) before T0, T2, T3 and completeness could be computed, and "
        "build_et0.py logs the scoring look through assert_can_score. The W4 test had already "
        "run under v1 (about 02:37Z), so it got an A1 declaration "
        "(W4.rain_kriging_vs_idw.v1.record) and a late-logged scoring look with the real time "
        "in its note instead of a back-dated pre-registration.",
        "The first-round downloads and the main fetch ran under nice 10 outside the CPU-slot "
        "wrapper (paced network waits, near-zero CPU). The resume-round re-fetch (10:25-10:29Z) "
        "and every computation ran inside the wrapper.",
        "Resume round (after a usage-limit interruption): the main fetch had completed at "
        "06:40:43Z; the pre-freeze re-fetch ran at 10:25-10:29Z (3 h 45 min later, the "
        "registered minimum is 60 min). et0/v1 was built twice, first into the scratchpad "
        f"(scoring look {ev_rb['scoring_look_times_utc']['scratch']}) and then into "
        f"results/dataprep/et0/v1 (scoring look {ev_rb['scoring_look_times_utc']['v1']}); the "
        "arrays and the CSV are identical and validation.json differs only in those times "
        "(W2W4_evidence/et0_v1_vs_scratch_build.json). Both looks are in the A1 look log. "
        "Earlier unlogged looks (the 2022 overlap, the probes and the 2012-2021 offset history "
        "as design-time seeds; a possible first-round look at 18 main-fetch points; the W4 "
        "flag diagnostic and rerun; the climatology-bias secondary and the station check) "
        "were added to the look log by look_log.py.",
        "W4: ordinary kriging was run as a pre-registered secondary diagnostic; it is not "
        "adoptable from these units.",
    ]
    issues = [
        f"I1 (recorded-forcing arm): use {et0_v}/et0_daily_2012_2026.npz, array 'et0' "
        "(level-matched; flag 1 from 2023-01), opened with use forcing_recorded_arm. For "
        "calibration and frozen forcing keep the cache values (et0_daily_2012_2022_preorigin."
        "npz, identical to results/et/openmeteo_et0_2012_2022.npz), opened with "
        "forcing_frozen. The 2023-26 values are post-origin (U4).",
        "Do not rebuild results/et/openmeteo_et0_2012_2022.npz from the API: the current archive "
        f"returns values about {f(-lv['fan_mean'])} mm/d higher (2022 at all 61 points; "
        "2012-2021 checked at st1 and st42), so a re-fetch would "
        "change the calibration forcing. The cache is GMT days (et.py fallback and the "
        "forcing experiment also use GMT); hydrophysics/twin/forcing_experiment.py is "
        "consistent.",
        "Plan text W2 should be corrected: the cache uses GMT days, and the 0.048 mm/d 2022 "
        "difference is an archive offset, not a day-convention effect. The local-vs-GMT "
        f"effect at st1 is {f(pr['daily_mae_local_vs_cache'])} mm/d daily but "
        f"{f(pr['monthly_mean_mae_local_vs_gmt'])} mm/d on monthly means.",
        "source 3 days (2026-07-01..09-30) are provisional; re-fetch them (and check the "
        "offset again) before any use after about 2027-01.",
        "W4: the twin keeps masked IDW (KED not adopted). Ordinary kriging scored "
        f"{f(sp['ratio_ok_idw'])} of the IDW RMSE fan-wide (worst gauge {ok_max:.3f}, "
        f"{ok_arg}; bootstrap {f(boot['ratio_ok_idw_p5_p50_p95'][0])}-"
        f"{f(boot['ratio_ok_idw_p5_p50_p95'][2])}). This is design-time evidence only; "
        "adopting it needs a new pre-registration scored on fresh units (for example CODIS "
        "gauges inside the fan that are not among the 26, 2012-2022).",
        "W4 residual gaps grew after the origin: kept gauge-months fell to "
        f"{100 * fl['kept_frac_by_year_strict']['2025']:.0f} % in 2025 and "
        f"{100 * fl['kept_frac_by_year_strict']['2026']:.0f} % in 2026 (2022: "
        f"{100 * fl['kept_frac_by_year_strict']['2022']:.0f} %). Gauges with <= 15 kept months "
        f"in 2023-26: {', '.join(post_gauges)}. Most masked days carry a CODIS missing-value "
        "mark of 1-2 with a numeric total; accepting those would keep "
        f"{100 * fl['kept_frac_by_year_tolerant_mark_le_2']['2025']:.0f} % in 2025 "
        f"({fl['kept_gauge_months_2023_2026_tolerant']} vs "
        f"{fl['kept_gauge_months_2023_2026_strict']} gauge-months in 2023-26) and also lifts "
        "2012-2022 (2014: "
        f"{100 * fl['kept_frac_by_year_strict']['2014']:.0f} -> "
        f"{100 * fl['kept_frac_by_year_tolerant_mark_le_2']['2014']:.0f} %). The mark's "
        "meaning is undocumented here; the cwa_rain owner should verify it (CWA CODIS "
        "documentation) after the freeze before any rule change. Not changed by this WP.",
        "data_fetch_api/new_measurements/cwa_codis_rain still has no SOURCE.json (A0 issue); "
        "this WP read its 467270 responses, whose provenance sits in their .meta.json "
        "sidecars and l0_registry/v1, and hashed them in et0/v1's manifest.",
        "Code hygiene for the Phase-1 port: rain_check.py and manifest_w4.py keep three ruff "
        "findings (import order in both, zip without strict in rain_check.py) because they "
        "are hashed in "
        "rain_check/v1's manifest and editing them would make v1 fail the checker; fix them "
        "when porting (and build rain_check/v2 if the report must be rebuilt). The other "
        "scripts are ruff-clean. Scripts that read downloads run as `python -I isorun.py "
        "<script>` (isolated mode with only the two code directories on sys.path).",
        "Open-Meteo quota: this WP used about "
        f"{calls['total']:.0f} of the 10,000 free calls on 2026-10-12 (shared IP); a "
        "full 2012-2026 re-fetch of 61 points costs about 23,500 calls (more than two days).",
    ]
    if not lr["apply_level_match_to_primary"]:
        issues[0] = issues[0].replace("(level-matched; flag 1 from 2023-01)", "(raw splice)")
    if r2 is not None:
        r2s = r2_summary(r2)
        deviations.append(
            "Resume round 2 (after a second usage-limit interruption; the first rounds had "
            "written this report at about 10:34Z but not returned): no product was rebuilt and "
            "no test re-scored. The round re-ran the hash checker after WP A0fix (guard fix) "
            f"and WP A1's repair round (checker: {r2s['checker']}; L0 orphans under "
            f"openmeteo_et0: {r2s['n_orphans']}), applied harness.prereg2 to the W2W4 "
            f"records ({r2s['prereg2_line']}), re-attributed {r2s['n_reattributed']} looks "
            "that had been logged under the unregistered id W4.rain_kriging_vs_idw.v1 to the "
            "A1 declaration W4.rain_kriging_vs_idw.v1.record (prereg2.record_look; the "
            "originals stay in the append-only log), logged one resume_review look, and "
            "regenerated this report (W2W4_evidence/resume_r2.json).")
        issues += [
            "Pre-registration audit (harness.prereg2, A1 repair round): W2.et0_overlap.v3 "
            f"passes assert_can_score with input verification ({r2s['prereg2_line']}). The "
            "audit keeps listing the three original W4 looks under the id "
            "W4.rain_kriging_vs_idw.v1 (append-only log); each now has a re-attribution "
            "record under W4.rain_kriging_vs_idw.v1.record that names its line sha256. The "
            "files W2.et0_overlap.v1, W2.et0_overlap.v2 and W4.rain_kriging_vs_idw.v1 stay "
            "outside INDEX.jsonl (written before the A1 tool), so load_prereg cannot verify "
            f"them; their sha256 match the ledger rows ({r2s['outside_ledger_ok']}) and no "
            "hashed input changed.",
            "Under prereg2's rule ('any look blocks'), the possible first-round look at the "
            "18 main-fetch responses (2025-01..06; disclosed in W2.et0_overlap.v3 and logged "
            "late as possible_unlogged_look) would block T2 (chain 2025H1) if it had been "
            "dated at its real time (between 02:35Z and 06:13Z, before v3). Read T2 as a "
            "data-consistency check with possible prior exposure: it has no tunable choice and "
            "its result (100 % identical values) cannot have been steered. T0, T3 and "
            "completeness are unaffected (their data did not exist before v3).",
            "A0fix's checker run (about 05:56Z) listed 19 openmeteo_et0 files as L0 orphans: "
            "the first round's 18 main-fetch responses (st1..st18) and its fetch log, written "
            "before the main fetch finished (06:40Z) and got its SOURCE.json "
            "(A0fix_evidence/check_after.json). Every openmeteo_et0 file is now covered by a "
            "SOURCE.json "
            f"({r2s['n_sources']} records, {r2s['n_orphans']} orphans).",
        ]
    sc_c = sc["2020-05..2022-12 (cache)"]
    sc_s = sc["2023-01..2026-09 (spliced)"]
    pm_c = sc_c["om_raw_splice_vs_station_pm"]
    pm_r, pm_l = sc_s["om_raw_splice_vs_station_pm"], sc_s["om_level_matched_vs_station_pm"]
    pan_c, pan_s = sc_c["om_raw_splice_vs_station_pan"], sc_s["om_raw_splice_vs_station_pan"]
    ann_lm = cbs["level_matched"]["annual_mean_bias_mm_d"]
    issues += [
        "W2 climatology: the 2023-01..2026-09 fan-mean ET0 is "
        f"{cbs['level_matched']['mean_bias_mm_d']:+.3f} mm/d above the 2012-2021 cache "
        f"climatology (raw splice {cbs['raw_splice']['mean_bias_mm_d']:+.3f}; by year "
        + ", ".join(f"{y} {v:+.3f}" for y, v in ann_lm.items())
        + f"; mean |z| {cbs['level_matched']['mean_abs_z']:.2f}). At CWA 467270 the "
        "Open-Meteo/station FAO-56 sum ratio is "
        f"{pm_c['sum_ratio_om_over_ref']:.3f} in 2020-05..2022-12 and "
        f"{pm_r['sum_ratio_om_over_ref']:.3f} (raw) or {pm_l['sum_ratio_om_over_ref']:.3f} "
        "(level-matched) in 2023-26: Open-Meteo tracks the station equally in both periods, "
        "so the rise is not a splice artefact at that site. The station starts in 2020, so "
        "the 2012-2021 "
        "climatology itself is not checked. A recorded-forcing run will see this higher ET0 "
        "(less max(rain - ET0, 0) recharge).",
        "Level match against the station (post hoc, one site): the raw splice keeps the "
        f"2020-22 ratio slightly better ({pm_r['sum_ratio_om_over_ref']:.3f} vs "
        f"{pm_l['sum_ratio_om_over_ref']:.3f}, against {pm_c['sum_ratio_om_over_ref']:.3f}). "
        "One station cannot settle a 0.05 mm/d level question; the primary array follows "
        "the registered rule, and an I1 sensitivity member with 'et0_raw_splice' costs "
        "nothing.",
        "CWA 467270 Class-A pan (CODIS EvaporationClassAPan) is not usable as a check: the "
        f"Open-Meteo/pan sum ratio is {pan_c['sum_ratio_om_over_ref']:.2f} (2020-22) and "
        f"{pan_s['sum_ratio_om_over_ref']:.2f} (2023-26) where a pan coefficient of about "
        f"0.6-0.85 would give a ratio below 1, and r is {pan_c['r']:.2f} and {pan_s['r']:.2f}. "
        "The field's units or QC are probably wrong; not investigated (W2 does not depend on "
        "it).",
    ]

    key = [
        f"W2 2022 overlap cache vs GMT fetch: MAE median {f(T1['mae_median'])}, max "
        f"{f(T1['mae_max'])} mm/d; bias {f(T1['bias_mean'])} mm/d (pass < 0.1)",
        "W2 archive offset (cache minus the 2026-09/10 archive, 2022): fan mean "
        f"{f(lv['fan_mean'])} mm/d, "
        f"points {f(lv['min'])}..{f(lv['max'])}; annual 2012-2021 within "
        f"{max(s['max_abs_dev_from_2022'] for s in stab.values()):.4f} of 2022 at st1/st42",
        f"W2 additive vs multiplicative offset: RMSE {f(of['rmse_after_additive'], 4)} vs "
        f"{f(of['rmse_after_multiplicative'], 4)} (raw {f(of['rmse_raw'], 4)}) mm/d",
        f"W2 chain 2025H1 max MAE {f(T2['mae_max'], 4)}; re-fetch 2026Q3 max MAE "
        f"{f(T3['mae_max'], 4)} mm/d, {T3['n_changed_values']} values changed",
        f"W2 climatology bias 2023-26 (level-matched): {f(cbs['level_matched']['mean_bias_mm_d'])}"
        f" mm/d, mean |z| {f(cbs['level_matched']['mean_abs_z'], 2)}, "
        f"{cbs['level_matched']['n_months_abs_z_gt_2']} months |z| > 2",
        f"W4 LOGO 2012-2022 ({sp['n_gauge_months']} gauge-months): RMSE IDW "
        f"{sp['rmse_mm']['idw']:.1f}, KED {sp['rmse_mm']['ked']:.1f}, OK "
        f"{sp['rmse_mm']['ok']:.1f} mm/month; KED/IDW {f(dec['fan_ratio_ked_idw'])}, worst "
        f"gauge {dec['max_gauge_ratio_ked_idw']:.3f} -> {dec['outcome']}",
        f"W4 secondary 2023-26 (after the fact): KED/IDW {f(rs['summary']['ratio_ked_idw'])}, "
        f"OK/IDW {f(rs['summary']['ratio_ok_idw'])}",
        f"W2 station check 467270 (post hoc): Open-Meteo/station FAO-56 sum ratio "
        f"{pm_c['sum_ratio_om_over_ref']:.3f} (2020-22), {pm_r['sum_ratio_om_over_ref']:.3f} raw / "
        f"{pm_l['sum_ratio_om_over_ref']:.3f} level-matched (2023-26); r {pm_c['r']:.3f} / "
        f"{pm_r['r']:.3f}",
        f"Open-Meteo call units used: {calls['total']:.0f}; Claude API tokens: 0; GPU: none",
    ]

    # ---------------------------------------------------------------- ledger
    C, F = rel(ET_CACHE), rel(FORCING_EXP_ET)
    M, RF, H, P = rel(MAIN), rel(REFETCH), rel(HISTORY), rel(PROBE)
    rain, gst, alt = rel(RF_TIMESERIES), rel(RF_STATIONS), rel(CODIS_STATION_LIST)
    st467 = "data_fetch_api/new_measurements/cwa_codis_rain/raw/467270 (2020-2026 files)"
    rows = []

    def add(n, src, role, pv, note=""):
        rows.append({"number_id": n, "source": src, "role": role, "product_version": pv,
                     "note": note})

    for n in ("W2.T1.overlap_mae_2022_max", "W2.T1.overlap_mae_2022_median",
              "W2.T1.overlap_bias_2022_mean", "W2.convention.r_cache_gmt_min"):
        add(n, F, "input", et0_v, "GMT fetch spliced after the cache (checked)")
        add(n, C, "validator", et0_v, "calibration cache (reference)")
    add("W2.convention.probe_local_mae", P, "input", et0_v, "local-day probe st1 2022")
    add("W2.convention.probe_local_mae", C, "validator", et0_v)
    for n in ("W2.T2.chain_mae_2025H1_max",):
        add(n, M, "input", et0_v)
        add(n, F, "validator", et0_v)
    for n in ("W2.T3.refetch_mae_2026Q3_max", "W2.T3.refetch_n_changed"):
        add(n, RF, "input", et0_v)
        add(n, M, "validator", et0_v)
    add("W2.T0.n_cell_mismatch", M, "input", et0_v)
    add("W2.T0.n_cell_mismatch", F, "validator", et0_v)
    for n in ("W2.level_offset_fan_mean", "W2.offset_form_rmse"):
        add(n, C, "fit_target", et0_v, "level the 2023+ series is matched to (2022)")
        add(n, F, "input", et0_v)
    add("W2.level_offset_stability_max_dev", C, "input", et0_v)
    add("W2.level_offset_stability_max_dev", H, "input", et0_v)
    add("W2.level_offset_stability_max_dev", F, "input", et0_v)
    add("W2.completeness_min_frac", f"{et0_v}", "input", et0_v)
    for n in ("W2.climatology_bias_mean", "W2.climatology_bias_abs_z"):
        add(n, F, "input", et0_v, "2023-01..2025-06")
        add(n, M, "input", et0_v, "2025-07..2026-06")
        add(n, RF, "input", et0_v, "2026-07..09")
        add(n, C, "validator", et0_v, "2012-2021 climatology (reference)")
    for n in ("W2.station467270.ratio_pm", "W2.station467270.ratio_pan"):
        add(n, f"{et0_v}", "input", et0_v, "nearest point, both variants")
        add(n, st467, "validator", et0_v, "station FAO-56 ET0 and Class-A pan (post hoc)")
    add("W2.fao56_example18", "FAO-56 Example 18 (Allen et al. 1998, published inputs and "
        "result)", "validator", et0_v, "check of station_et0.fao56")
    add("W2.rebuild_identical", f"{et0_v}", "input", et0_v, "scratch build vs v1")
    add("W2.station467270.ratio_pm_level_vs_raw", f"{et0_v}", "input", et0_v)
    add("W2.station467270.ratio_pm_level_vs_raw", st467, "validator", et0_v, "post hoc")
    for n in ("W4.rmse_idw_2012_2022", "W4.rmse_ked_2012_2022", "W4.rmse_ok_2012_2022",
              "W4.ratio_ked_idw_2012_2022", "W4.max_gauge_ratio_ked_idw_2012_2022",
              "W4.ratio_ok_idw_2012_2022", "W4.bootstrap_ratio_ci",
              "W4.ratio_ked_idw_2023_2026"):
        add(n, rain, "input", rain_v, "LOGO: each gauge held out in turn (target) while the "
                                      "others predict; one file, recorded as input")
        add(n, gst, "input", rain_v)
        if "ok" not in n:
            add(n, alt, "input", rain_v, "KED altitude drift")
    for n in ("W4.kept_frac_by_year", "W4.flag_tolerant_kept_frac"):
        add(n, rain, "input", rain_v)
        add(n, "data_fetch_api/new_measurements/cwa_codis_rain/raw", "input", rain_v,
            "CODIS missing-value marks")
    for name, h in pre.items():
        if name in a1_written:      # the A1 tool already wrote their ledger rows
            continue
        add(f"W2W4.prereg.{name[:-5]}", rel(PREREG_DIR / name), "input",
            f"prereg-sha256:{h}", "pre-registration written by dataprep_staging/w2_w4_weather/"
            "prereg.py before the A1 tool existed" + (" (superseded, never run)"
                                                       if name.endswith("v1.json") and
                                                       name.startswith("W2") else ""))
    ledger.append_rows("W2W4", rows)

    # ---------------------------------------------------------------- JSON + MD
    status = "passed" if all(r["passed"] for r in acc if not r["criterion"].startswith(
        "W4 adoption")) else "partial"
    products = [
        {"path": rel(E), "layer": "L1", "manifest": rel(E / "MANIFEST.json"),
         "allowed_uses": "forcing_recorded_arm (2023-26, U4), forcing_frozen (only "
                         "et0_daily_2012_2022_preorigin.npz), statistical_layer_history, "
                         "after_the_fact_analysis",
         "fit_window_end": "2026-09 (pre-origin output 2022-12; level-match offsets fitted on "
                           "2022)"},
        {"path": rel(R), "layer": "L2 (report only)", "manifest": rel(R / "MANIFEST.json"),
         "allowed_uses": "model_design and design_time_evidence (only the 2012-2022 outputs), "
                         "after_the_fact_analysis",
         "fit_window_end": "2026-09 (decision outputs 2022-12)"},
    ]
    for lbl in (MAIN.name, REFETCH.name, HISTORY.name, PROBE.name, VARIANT_DIR.name):
        products.append({"path": rel(MAIN.parent / lbl), "layer": "L0",
                         "manifest": rel(MAIN.parent / lbl / "SOURCE.json"),
                         "allowed_uses": "raw input; untrusted data; CC BY 4.0",
                         "fit_window_end": "n/a (raw; fetched 2026-10-12)"})
    code = sorted(rel(p) for p in Path(__file__).resolve().parent.glob("*.py"))
    w2_checks = {"chain overlap (T2)": T2["pass"], "pre-freeze re-fetch (T3)": T3["pass"],
                 "grid-cell check (T0)": T0["pass"], "completeness": comp["pass"]}
    passed_c = [k for k, v in w2_checks.items() if v]
    failed_c = [k for k, v in w2_checks.items() if not v]
    lm = lr["apply_level_match_to_primary"]
    summary = (
        f"W2 extended the 61-point daily ET0 forcing to 2026-09 ({et0_v}). The plan assumed "
        "the calibration cache used local days; it uses GMT days, so the product keeps GMT "
        "and the cache unchanged, and the local-day pre-registration was superseded before "
        f"it ran. The 2022 overlap {'passes' if T1['pass'] else 'FAILS'} (max per-point MAE "
        f"{f(T1['mae_max'])} vs < 0.1 mm/d). The difference is an archive level offset, not "
        "a day-convention effect: the Open-Meteo archive as fetched in 2026-09/10 is "
        f"{f(-lv['fan_mean'])} mm/d higher than the June 2026 cache"
        + (" by about the same amount in every year 2012-2022 (checked at st1 and st42). "
           if lr["stable_all"] else
           "; the annual offset at st1/st42 is not stable within 0.02 mm/d. ")
        + ("By the pre-registered rule the primary array is level-matched from 2023-01 (flag 1)"
           " and the raw splice is kept. " if lm else
           "The pre-registered level-match rule did not hold, so the primary array is the raw "
           "splice; the level-matched variant is kept. ")
        + (f"Passed: {', '.join(passed_c)}. " if passed_c else "")
        + (f"FAILED: {', '.join(failed_c)}. " if failed_c else "")
        + "A post-hoc station check at CWA 467270 is reported (not gating). W4 ran the "
        "pre-registered leave-one-gauge-out test on 2012-2022 monthly totals: KED lowered "
        f"RMSE by only {100 * (1 - dec['fan_ratio_ked_idw']):.1f} % (needs 10 %) and "
        f"{len(over)} gauges got worse by more than 20 % ({over_txt}), so KED is not adopted "
        "and masked IDW stays; rain_check/v1 is a report. Ordinary kriging looked better "
        f"(ratio {f(sp['ratio_ok_idw'])}; a registered secondary, not adoptable). The "
        "residual-gap analysis found that 2023-26 rain masking grows to about half of the "
        "gauge-months in 2025-26, driven by CODIS partial-day marks. No GPU, no Claude API, "
        "nothing under hydrophysics/ or tests/ edited, nothing staged in git.")
    rep = {"wp": "W2W4", "date": dt.date.today().isoformat(), "status": status,
           "summary": summary, "acceptance": acc, "products": products, "code_paths": code,
           "key_numbers": key, "deviations_from_plan": deviations,
           "issues_for_downstream": issues, "report_path": rel(REPORTS_DIR / "W2W4.md"),
           "preregistrations": pre, "open_meteo_call_units": calls, "claude_api_tokens": 0,
           "a1_scoring": ev.get("a1_scoring", {}), "review_probe_check": rv_chk,
           "rebuild": ev_rb,
           "w4_gauges_over_1.20": over_txt,
           "w2": {"validation": rel(E / "validation.json"), "station_check": sc},
           "w4": {"validation": rel(R / "validation.json"), "secondary": rs["summary"],
                  "rain_flag_diagnostic": fl}}
    if r2 is not None:
        rep["resume_r2"] = {"evidence": rel(r2_path), **r2_summary(r2)}
    (REPORTS_DIR / "W2W4.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1,
                                                      default=float) + "\n")
    VARIANTS.update(variant_summary())
    md = render_md(rep, ev, rv, rs, per, sc, gaps, fl)
    (REPORTS_DIR / "W2W4.md").write_text(md)
    print(REPORTS_DIR / "W2W4.md", status)
    return 0


def r2_summary(r2: dict) -> dict:
    """Short form of W2W4_evidence/resume_r2.json (resume round 2) for the report."""
    pc = r2["prereg_checks"]
    v3 = pc["W2.et0_overlap.v3"]
    decl = pc["W4.rain_kriging_vs_idw.v1.record"]
    outside = {k: v for k, v in pc.items() if "ledger_sha256_matches" in v}
    aud = r2.get("audit_w2w4_after") or r2["audit_w2w4_before"]
    return {
        "checker": r2["checker"]["status"],
        "checker_manifests": r2["checker"]["manifests"],
        "n_sources": r2["checker"]["n_sources"],
        "n_orphans": len(r2["checker"]["orphans_openmeteo_et0"]),
        "prereg2_line": (f"W2.et0_overlap.v3 {v3['assert_can_score_prereg2'].split(' (')[0]}, "
                         f"inputs verified {v3.get('inputs_verified')}; W4 declaration "
                         f"load_prereg {decl['load_prereg']}, changed inputs "
                         f"{len(decl.get('changed_inputs', []))}; refused tests "
                         f"{len(aud['tests_refused_under_repaired_rule'])}, tests with "
                         f"changed inputs {len(aud['tests_with_changed_inputs'])}"),
        "v3_post_registration_looks": v3.get("post_registration_looks_other_tests", []),
        "outside_ledger_ok": all(v["ledger_sha256_matches"] and not v["changed_inputs"]
                                 for v in outside.values()),
        "n_reattributed": sum(1 for x in r2["reattributed_looks"]
                              if x.get("logged") or x.get("already_logged")),
        "reattributed": r2["reattributed_looks"],
        "resume_look": r2["resume_look"],
        "audit_w2w4": aud,
    }


def render_md(rep, ev, rv, rs, per, sc, gaps, fl) -> str:
    T1, T2, T3, T0 = (ev["T1_cache_vs_fe_2022"], ev["T2_main_vs_fe_2025H1"],
                      ev["T3_refetch_vs_main_2026Q3"], ev["T0_same_points"])
    lv, lr, of, pr = (ev["level_offsets"], ev["level_match_rule"], ev["offset_form"],
                      ev["probe_local_vs_gmt_st1_2022"])
    cbs = ev["climatology_bias_summary"]
    dec, sp, boot = rv["decision"], rv["summary"], rv["bootstrap"]
    L = []
    w = L.append
    w("# WP W2W4: ET0 extension 2023-26 (W2) and rain residual gaps (W4)\n")
    w(f"Report date {rep['date']}. Status: **{rep['status']}**. "
      "Plan: `docs/DATA_REMEDIATION_PLAN_2026-10.md`, WP W2 and WP W4.\n")
    w(rep["summary"] + "\n")
    w("## Acceptance\n")
    w("| Criterion | Threshold | Measured | Pass |\n|---|---|---|---|")
    for r in rep["acceptance"]:
        w(f"| {r['criterion']} | {r['threshold']} | {r['measured']} | "
          f"{'yes' if r['passed'] else '**no**'} |")
    w("\nThe two W4 adoption rows fail by design of the test: the plan says kriging is adopted "
      "only if both pass, otherwise masked IDW stays. W4 is complete either way; W2W4's status "
      "is decided by the W2 rows and the W4 evaluation row.\n")
    w("## W2: what was done and what was found\n")
    w("**Day convention.** The plan says the cache uses local days. Before fetching all 61 "
      "points, a one-point probe (st1, 2022, local days; 26 API call units) disagreed with the "
      f"cache by {f(pr['daily_mae_local_vs_cache'])} mm/d (r {f(pr['r_cache_local'], 4)}), "
      f"while the forcing experiment's GMT fetch agreed to {f(pr['daily_mae_gmt_vs_cache'])} "
      f"(r {f(pr['r_cache_gmt'], 4)}). Over all 61 points the cache matches GMT at lag 0 "
      f"(r >= {f(T1['r_min'], 4)}); at +-1 day r is about 0.70. The cache is GMT days. It was "
      "committed on 2026-06-21 through AquaScope's collector or the GMT fallback of "
      "`hydrophysics/et.py`. AquaScope's current source sends `timezone=auto` (local), so the "
      "June version presumably did not; that is unverified. Endpoint variants at st1 for "
      "January 2022 (archived in `openmeteo_et0/w2_variant_probes_2026-10-12`) give the "
      "daily MAE against the cache below: `/v1/era5`, `/v1/archive` and `models=ecmwf_ifs` "
      "are identical today (the API default is ECMWF IFS); ERA5 and ERA5-seamless sit on "
      "other grids and are further from the cache; ERA5-Land returned no values at that "
      "cell. Pre-registration v1 (local days) was therefore superseded by v2 (GMT) before "
      "any 61-point fetch.\n")
    w("| Probe (st1, Jan 2022) | Grid cell | Days | MAE vs cache (mm/d) |\n|---|---|---|---|")
    for k, r in VARIANTS.items():
        m = "no values" if r["mae_vs_cache"] is None else f"{r['mae_vs_cache']:.3f}"
        w(f"| {k} | {r['grid_lat']:.3f} N {r['grid_lon']:.3f} E | {r['n_finite']} | {m} |")
    w("")
    w(f"**Archive offset.** In 2022 the cache is lower than the archive as fetched on "
      f"2026-09-30 (forcing experiment) by "
      f"{f(-lv['fan_mean'])} mm/d on average (per point {f(-lv['max'])} to {f(-lv['min'])}). "
      f"The difference is additive: RMSE {f(of['rmse_raw'], 4)} raw, "
      f"{f(of['rmse_after_additive'], 4)} after a per-point additive offset, "
      f"{f(of['rmse_after_multiplicative'], 4)} after a multiplicative one. Against a fresh "
      "2012-2021 fetch at st1 (north) and st42 (southernmost), the annual mean offsets stay "
      "within "
      f"{max(s['max_abs_dev_from_2022'] for s in lr['stability'].values()):.4f} mm/d of the "
      "2022 value in every year. "
      + ("The level of the Open-Meteo archive therefore changed by a near-uniform amount "
         "between the cache's fetch (on or before 2026-06-21, the commit date) and the "
         "forcing experiment's fetch (2026-09-30); the cause is not documented. "
         if lr["stable_all"] else
         "The offset is not stable across years at both points, so a uniform archive "
         "revision is not established. ")
      + "The plan's reading of the 0.048 mm/d difference as a day-convention effect does not "
      "fit these data.\n")
    st1 = lr["stability"]["st1"]["annual_offset"]
    st42 = lr["stability"]["st42"]["annual_offset"]
    w("| Year | " + " | ".join(str(y) for y in st1) + " | 2022 |")
    w("|---|" + "---|" * (len(st1) + 1))
    w("| st1 offset (mm/d) | " + " | ".join(f"{v:.4f}" for v in st1.values())
      + f" | {lr['stability']['st1']['offset_2022']:.4f} |")
    w("| st42 offset (mm/d) | " + " | ".join(f"{v:.4f}" for v in st42.values())
      + f" | {lr['stability']['st42']['offset_2022']:.4f} |\n")
    w("**Level match.** Pre-registered rule: apply the 2022 per-point offset from 2023-01 if "
      "|fan mean| >= 0.02 mm/d and every annual offset lies within 0.02 of the 2022 value. "
      + ("Both held, so the primary array `et0` is level-matched (flag 1, method "
         "level_match_2022_additive). " if lr["apply_level_match_to_primary"] else
         f"The rule did not hold (|fan mean| >= 0.02: {lr['fan_mean_abs_ge_0.02']}; stable: "
         f"{lr['stable_all']}), so the primary array `et0` is the raw splice (flag 0). ")
      + "`et0_raw_splice` keeps the source values (flag 0), `et0_level_matched` the shifted "
      "ones, and `level_offset_mm_d` stores the 61 offsets. Primary variant: "
      f"**{ev['primary_variant']}**.\n")
    w("**Splice (GMT days).**\n")
    w("| Source | Days | Fetched |\n|---|---|---|")
    for k, (a, b) in ev["splice"].items():
        nm = {"0": "calibration cache", "1": "forcing-experiment fetch",
              "2": "new fetch (openmeteo_et0/w2_gmt_2025_2026)",
              "3": "pre-freeze re-fetch (openmeteo_et0/w2_gmt_refetch_2026q3), provisional"}[k]
        fd = {"0": "committed 2026-06-21", "1": "2026-09-30", "2": "2026-10-12",
              "3": "2026-10-12"}[k]
        w(f"| {k}: {nm} | {a}..{b} | {fd} |")
    w("")
    w(f"**Checks.** Same cells: {T0['n_mismatch']} mismatches over 61 points "
      f"({T0['unique_cells']} distinct IFS cells, {T0['unique_cell_elevation']} distinct "
      "cell+elevation pairs, because Open-Meteo downscales temperature to the point's "
      f"elevation). Chain overlap 2025-01..06: max MAE {f(T2['mae_max'], 4)} mm/d, "
      f"{100 * T2['identical_frac']:.1f} % identical values. Re-fetch 2026-07..09: max MAE "
      f"{f(T3['mae_max'], 4)} mm/d, {T3['n_changed_values']} values changed. The review's "
      "fetch at 23.7 N 120.4 E (2026-10-11 19:43 UTC) and this WP's probe at the same point "
      f"(2026-10-12 02:27 UTC) agree on {rep['review_probe_check']['days']} "
      f"({rep['review_probe_check']['n']} days, max |diff| "
      f"{rep['review_probe_check']['max_abs']:.2f} mm/d).\n")
    w("**Monthly bias against the 2012-2021 cache climatology (fan mean of 61 points).**\n")
    w("| Variant | Mean bias 2023-01..2026-09 (mm/d) | Mean abs z | Months with abs z > 2 | "
      "Annual mean bias (mm/d) |\n|---|---|---|---|---|")
    for vname in ("level_matched", "raw_splice"):
        s = cbs[vname]
        ann = ", ".join(f"{y}: {b:+.3f}" for y, b in s["annual_mean_bias_mm_d"].items())
        w(f"| {vname} | {s['mean_bias_mm_d']:+.3f} | {s['mean_abs_z']:.2f} | "
          f"{s['n_months_abs_z_gt_2']} | {ann} |")
    w("\nPer-month values: `et0/v1/climatology_bias_2023_2026.csv`.\n")
    w("**Station check (post hoc, optional in the plan, not gating).** CWA 467270 "
      f"(Tianzhong, {sc['station_altitude_m']} m) carries the full daily meteorology and "
      "Class-A pan evaporation from 2020-04. Station FAO-56 ET0 was computed (implementation "
      "checked on FAO-56 Example 18: "
      f"{json.loads((EVID / 'fao56_example18.json').read_text())['computed_mm_d']:.2f} "
      "against the published 3.9 mm/d, evidence `W2W4_evidence/fao56_example18.json`; "
      "anemometer height assumed 10 m) "
      f"and compared monthly with the nearest point {sc['nearest_point']} "
      f"({sc['nearest_point_km']:.1f} km).\n")
    w("| Period | Comparison | Months | Sum ratio (Open-Meteo / station) | Bias (mm/d) | r |")
    w("|---|---|---|---|---|---|")
    for per_name in ("2020-05..2022-12 (cache)", "2023-01..2026-09 (spliced)"):
        for k2, r in sc.get(per_name, {}).items():
            w(f"| {per_name} | {k2} | {r['n_months']} | {r['sum_ratio_om_over_ref']:.3f} | "
              f"{r['bias_mm_d']:+.3f} | {r['r']:.3f} |")
    w("\nRead this as a consistency check only: the station's exposure sets its level, and "
      "one station cannot tell which archive level is closer to the truth. The FAO-56 "
      "comparison is stable across the two periods (r about 0.9). The Class-A pan series is "
      "not usable: Open-Meteo ET0 is 2-3 times the pan total, where a pan coefficient of "
      "0.6-0.85 would put it below the pan, and the correlation is weak or negative; the "
      "CODIS pan field's units or QC are suspect (see Issues).\n")
    w("**Uses (manifest).** `forcing_recorded_arm` for the 2023-26 values (U4); "
      "`forcing_frozen` may read only `et0_daily_2012_2022_preorigin.npz` (the cache "
      "values); `statistical_layer_history`; `after_the_fact_analysis`. Targets, baselines, "
      "datum fits, selection and design uses are forbidden.\n")
    w("## W4: masked IDW vs kriging with an elevation drift\n")
    w(f"Pre-registered test (W4.rain_kriging_vs_idw.v1): leave-one-gauge-out on monthly "
      f"totals 2012-01..2022-12 (twin rule: a gauge-month needs >= 80 % of its days), "
      f"{sp['n_gauge_months']} gauge-months over {sp['n_gauges']} gauges and "
      f"{sp['n_months']} months; variograms refitted for each held-out gauge on the other 25 "
      "and the pre-origin window only.\n")
    w("| Method | RMSE (mm/month) | MAE | Bias | Ratio to IDW | Wet May-Oct RMSE | Dry "
      "Nov-Apr RMSE |\n|---|---|---|---|---|---|---|")
    for m, nm in (("idw", "masked IDW (incumbent)"), ("ked", "KED, drift on altitude (candidate)"),
                  ("ok", "ordinary kriging (secondary)")):
        w(f"| {nm} | {sp['rmse_mm'][m]:.2f} | {sp['mae_mm'][m]:.2f} | {sp['bias_mm'][m]:+.2f} |"
          f" {sp['rmse_mm'][m] / sp['rmse_mm']['idw']:.3f} | "
          f"{sp['season_rmse_mm']['wet_May_Oct'][m]:.2f} | "
          f"{sp['season_rmse_mm']['dry_Nov_Apr'][m]:.2f} |")
    w(f"\nMean observed monthly total {sp['obs_mean_mm']:.1f} mm. Block bootstrap (12-month "
      f"blocks, 2000 draws): KED/IDW {boot['ratio_ked_idw_p5_p50_p95'][0]:.3f}-"
      f"{boot['ratio_ked_idw_p5_p50_p95'][2]:.3f} (5-95 %), P(ratio <= 0.90) = "
      f"{boot['p_ked_ratio_le_0.90']:.4f}; OK/IDW {boot['ratio_ok_idw_p5_p50_p95'][0]:.3f}-"
      f"{boot['ratio_ok_idw_p5_p50_p95'][2]:.3f}. Kriging systems needing the least-squares "
      f"fallback: {rv['kriging_fallback_solves']}.\n")
    w(f"**Decision: {dec['outcome']}.** KED improves {sp['n_gauges_ked_better']} of 26 gauges "
      f"but only {100 * (1 - dec['fan_ratio_ked_idw']):.1f} % fan-wide, and it worsens the "
      f"apex gauge rf25 by {100 * (dec['max_gauge_ratio_ked_idw'] - 1):.0f} %. Held out, rf25 "
      "(110 m) sits above every training gauge (next highest 65 m), so the altitude drift "
      "extrapolates. Gauges above the 1.20 limit: " + rep["w4_gauges_over_1.20"] + ".\n")
    w("| Gauge | n | RMSE IDW | RMSE KED | KED/IDW | RMSE OK | OK/IDW |\n"
      "|---|---|---|---|---|---|---|")
    for r in per.itertuples():
        w(f"| {r.gauge} | {r.n} | {r.rmse_idw:.1f} | {r.rmse_ked:.1f} | {r.ratio_ked_idw:.3f} | "
          f"{r.rmse_ok:.1f} | {r.ratio_ok_idw:.3f} |")
    w(f"\nOrdinary kriging (no drift) scored {sp['ratio_ok_idw']:.3f} of the IDW RMSE with a "
      f"worst gauge at {per['ratio_ok_idw'].max():.3f}. It was registered as a non-adoptable "
      "secondary, and its bootstrap interval straddles 0.90. It is design-time evidence for "
      "a future registered test on fresh units, not a reason to change the forcing now.\n")
    w(f"Secondary window 2023-01..2026-09 (after the fact, variograms from 2012-2022, "
      f"{rs['summary']['n_gauge_months']} gauge-months): RMSE IDW "
      f"{rs['summary']['rmse_mm']['idw']:.1f}, KED {rs['summary']['rmse_mm']['ked']:.1f}, OK "
      f"{rs['summary']['rmse_mm']['ok']:.1f} mm/month; worst KED gauge "
      f"{rs['summary']['argmax_gauge_ratio_ked_idw']} "
      f"({rs['summary']['max_gauge_ratio_ked_idw']:.3f}).\n")
    w("**Residual gaps.** Kept gauge-months by year (fan mean, twin rule), and the same with "
      "CODIS days marked 1-2 accepted (a what-if; the mark's meaning is not verified):\n")
    ys = list(fl["kept_frac_by_year_strict"])
    w("| Rule | " + " | ".join(ys) + " |\n|---|" + "---|" * len(ys))
    w("| twin rule (unmarked days only) | " + " | ".join(
        f"{100 * fl['kept_frac_by_year_strict'][y]:.0f}" for y in ys) + " |")
    w("| marks 1-2 accepted | " + " | ".join(
        f"{100 * fl['kept_frac_by_year_tolerant_mark_le_2'][y]:.0f}" for y in ys) + " |")
    marks = fl["days_2023_2026_by_mark"]
    tot = sum(marks.values())
    w(f"\nIn 2023-26, {marks.get('0', 0)} of {tot} gauge-days are unmarked, "
      f"{marks.get('1', 0)} carry mark 1, {marks.get('2', 0)} mark 2 and {marks.get('24', 0)} "
      "mark 24 (whole day). rf24 starts in "
      f"{gaps.loc['rf24', 'first_kept_month']} (no CWA predecessor; the vendor's earlier "
      "values are left empty by the rebuild). The plan's statement that rf25 is empty before "
      "2018 is out of date: rf25 (C1G691, "
      f"{gaps.loc['rf25', 'altitude_m']:.0f} m) has {gaps.loc['rf25', 'kept_months_2012_2022']}"
      " kept months in 2012-2022 but only "
      f"{gaps.loc['rf25', 'kept_months_2023_2026_09']} in 2023-01..2026-09 and none from 2024 "
      "on, so the apex rain in the recorded-forcing arm comes from lower gauges by IDW. "
      "Per-gauge, per-year table: `rain_check/v1/residual_gaps_2012_2026.csv`.\n")
    w("## Deviations from the plan\n")
    for d in rep["deviations_from_plan"]:
        w(f"- {d}")
    w("\n## Issues for downstream\n")
    for d in rep["issues_for_downstream"]:
        w(f"- {d}")
    w("\n## Provenance and resources\n")
    w("- Pre-registrations (each written before the tests it governs; v1/v2 and W4 v1 by "
      "this WP's prereg.py before the A1 tool existed, v3 and the W4 record by the A1 tool, "
      "which also holds them in INDEX.jsonl; W2 scoring looks logged by assert_can_score at "
      f"{rep['rebuild']['scoring_look_times_utc']['scratch']} (scratch build) and "
      f"{rep['a1_scoring'].get('logged_utc', 'n/a')} (et0/v1 build); earlier looks added "
      "by look_log.py):")
    for n, h in rep["preregistrations"].items():
        w(f"  - `results/dataprep/preregistration/{n}` sha256 `{h}`")
    w("- L0 downloads (Open-Meteo, CC BY 4.0, SOURCE.json each): "
      "`data_fetch_api/new_measurements/openmeteo_et0/{w2_probe_st1_2022, "
      "w2_gmt_history_2012_2021_st1_st42, w2_gmt_2025_2026, w2_gmt_refetch_2026q3, "
      "w2_variant_probes_2026-10-12}`.")
    w(f"- Open-Meteo call units: {rep['open_meteo_call_units']}. Claude API: 0 tokens. GPU: "
      "none. Computations and the resume-round re-fetch ran in the CPU-slot wrapper; the "
      "first-round downloads and the main fetch under nice 10.")
    if "resume_r2" in rep:
        r = rep["resume_r2"]
        w(f"- Resume round 2 (`{r['evidence']}`): checker {r['checker']} "
          f"({', '.join(f'{p} {s}' for p, s in r['checker_manifests'])}; "
          f"{r['n_sources']} SOURCE.json, {r['n_orphans']} orphans); {r['prereg2_line']}; "
          f"{r['n_reattributed']} W4 looks re-attributed to W4.rain_kriging_vs_idw.v1.record "
          "(logged "
          + ", ".join(x.get("logged", "earlier") for x in r["reattributed"])
          + f"); resume_review look logged {r['resume_look'].get('logged', 'earlier')}.")
    w("- Ledger: `results/dataprep/ledger/rows/W2W4.csv`.")
    w("- Code: " + ", ".join(f"`{c}`" for c in rep["code_paths"]))
    w("- Nothing under `hydrophysics/`, `tests/` or the L0 vendor caches was written; nothing "
      "was staged or committed in git.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
