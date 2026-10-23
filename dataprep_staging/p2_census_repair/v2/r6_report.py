"""WP P2abc round 2 (resumed 2026-10-14): ledger rows and the WP report.

Reads only already-computed files: the two published products' validation.json and
manifests, the registered test outputs under ``P2abc_evidence/tests``, stage1.json,
d0_design_counts.json, the checker and guard-probe outputs, the round-1 results and
``report_facts.json`` (r5_resume.py facts). Writes

* ledger rows to ``results/dataprep/ledger/rows/P2abc.csv`` (common.ledger.append_rows; the
  shared data_roles.csv is not reassembled here);
* ``results/dataprep/reports/P2abc.json`` and ``P2abc.md``.

Every number quoted in the report carries a ledger number id; the script fails if one is
missing (common.ledger.unledgered).

Usage: python -I r6_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
from common import ledger  # noqa: E402

R = L.REPO
DP = "results/dataprep"
PP = f"{DP}/yunlin_electricity/preorigin/v1"
PF = f"{DP}/yunlin_electricity/full/v1"
C0P = f"{DP}/pump_electricity/preorigin/v3"
C0F = f"{DP}/pump_electricity/full/v3"
C0L1 = f"{DP}/pump_electricity/census_L1/v3"
STG = f"{DP}/yunlin_electricity/_work_p2abc_v2"
R1W = f"{DP}/yunlin_electricity/_work_p2abc"
EV = f"{DP}/reports/P2abc_evidence"
WELLS = "AMP_V2/data/wells"
STATIONS = "AMP_V2/data/fan_stations.parquet"
RAIN = "data_fetch_api/new_measurements/cwa_codis_rain/rf_timeseries_codis.csv"
RAINST = "chou-shui-data/data/rf_stations.csv"
AMP_P = "AMP_V2/data/tpc_pumps.parquet"
AMP_K = "AMP_V2/data/pump_kwh_all.parquet"
PILOT_Z = [f"{DP}/_pilots/billing_and_census_gaps/zero_runs_summary.csv",
           f"{DP}/_pilots/billing_and_census_gaps/census_zeros.json"]
L0T = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/parquet"
PREREG = {"r2": "P2abc.r2.c0_based.v1", "k1r": "P2abc.r2.k1_repair.v1"}
COUNTIES = ("yunlin", "changhua", "chiayi")


def ld(rel: str):
    with open(R / rel, encoding="utf-8") as f:
        return json.load(f)


def f3(x) -> str:
    return f"{x:.3f}"


def pct(x, nd=1) -> str:
    return f"{100 * x:.{nd}f} %"


# ----------------------------------------------------------------------------- data
VP = ld(f"{PP}/validation.json")
VF = ld(f"{PF}/validation.json")
MP = ld(f"{PP}/MANIFEST.json")
MF = ld(f"{PF}/MANIFEST.json")
FACTS = ld(f"{EV}/report_facts.json")
ST1 = ld(f"{STG}/stage1.json")
D0 = ld(f"{EV}/design_time/d0_design_counts.json")
CHK = ld(f"{EV}/checker.json")
T3 = ld(f"{EV}/tests/t3_july_backtest.json")
H1 = ld(f"{EV}/tests/t2_head_check_round1.json")
H2 = ld(f"{EV}/tests/t2_head_check_round2.json")
BIP = ld(f"{PP}/build_info.json")
BIF = ld(f"{PF}/build_info.json")
R1V = ld(f"{R1W}/validation_preorigin.json")
R1J = ld(f"{R1W}/july_backtest.json")
R1REP = ld(f"{EV}/r1_reproduce.json")


A1P = VP["P2r2.a1_a2"]["counties"]
A1F = ld(f"{EV}/tests/t1_meter_validation_full.json")["counties"]  # full-variant sd coverage
A1Fsec = VF["P2r2.a1_full_variant_secondary"]
B1 = VF["P2r2.b1"]["counties"]
C1 = VP["P2r2.c1"]


# ----------------------------------------------------------------------------- ledger
ROWS: list[dict] = []


def row(nid, source, role, pv, note=""):
    ROWS.append({"number_id": nid, "source": source, "role": role, "product_version": pv,
                 "wp": L.WP, "note": note})


def build_ledger() -> None:
    pv_r1 = f"{R1W} (round 1, design-time; superseded by round 2)"
    # round 1 (AMP_V2 Yunlin; scored by the interrupted session, logged as design-time seeds)
    for nid, units, note in (
            ("P2abc.r1.meter_validation", "P2r1:yunlin_amp_monthly_irrigation_meters",
             "round-1 P2a.meter_validation.v1: min skill 0.229 over n 1..300 (passed)"),
            ("P2abc.r1.sd_coverage", "P2r1:yunlin_amp_monthly_irrigation_meters",
             "round-1 P2a.sd_coverage.v1: coverage 0.878-0.899 in 5 strata"),
            ("P2abc.r1.july", "P2r1:yunlin_amp_phaseA_julys_2013_2024",
             "round-1 P2b.july_backtest.v1: mean fan-total error 14.0 % (max 29.2 %)")):
        row(nid, AMP_P, "input", pv_r1, note)
        row(nid, AMP_K, "input", pv_r1, note)
        row(nid, units, "validator", pv_r1, note + "; truth = the named meters of the cache")
    row("P2abc.r1.zero_reproduction", AMP_P, "input", pv_r1, "round-1 P2c (exact)")
    row("P2abc.r1.zero_reproduction", AMP_K, "input", pv_r1, "round-1 P2c (exact)")
    for p in PILOT_Z:
        row("P2abc.r1.zero_reproduction", p, "validator", pv_r1, "pilot counts reproduced")
    row("P2abc.r1.reproduce", R1W, "input", pv_r1,
        "r1_reproduce.py: rerun of the round-1 scripts identical in every compared file")
    for nid in ("P2abc.r1.head_placebo",):
        for s, r in ((WELLS, "validator"), (STATIONS, "input"), (RAIN, "input"),
                     (RAINST, "input"), (R1W, "input"), (AMP_P, "input"), (AMP_K, "input")):
            row(nid, s, r, pv_r1, "P2a.head_placebo.v1 scored in round 2 (never before)")
    # round-2 registered tests
    for c in COUNTIES:
        for var, c0, pv in (("preorigin", C0P, PP), ("full", C0F, PF)):
            nid = f"P2abc.r2.a1.skill.{c}.{var}"
            row(nid, c0, "input", pv, "bimonthly bills and the county operator")
            row(nid, STG, "input", pv, "stage-1 profiles and Chow-Lin operators")
            row(nid, f"P2r2:val_meters_{c}", "validator", pv,
                "truth = the county's monthly (M0/M9) irrigation meters of the same C0 "
                "product, pseudo-bimonthly")
        if c != "chiayi":
            nid = f"P2abc.r2.a1.placebo.{c}.preorigin"
            row(nid, C0P, "input", PP, "year-shuffled profile, seeds 1..20")
            row(nid, STG, "input", PP, "stage-1 profiles")
            row(nid, f"P2r2:val_meters_{c}", "validator", PP, "same meters as the skill")
            for var, c0, pv in (("preorigin", C0P, PP), ("full", C0F, PF)):
                nid = f"P2abc.r2.a2.sdcov.{c}.{var}"
                row(nid, c0, "input", pv, "Chow-Lin split of the validation groups")
                row(nid, STG, "input", pv, "stage-1 operators")
                row(nid, f"P2r2:sd_halves_{c}", "validator", pv,
                    "q90 fitted on one easting half, scored on the other")
    for nid in ("P2abc.r2.a3.heads.yunlin", "P2abc.r2.a3.heads.changhua",
                "P2abc.r2.a3.heads.pooled", "P2abc.r2.a3.heads.fixed_nuisance"):
        for s, r in ((WELLS, "validator"), (STATIONS, "input"), (RAIN, "input"),
                     (RAINST, "input"), (C0P, "input"), (STG, "input")):
            row(nid, s, r, PP, "heads 2012-01..2022-11 (daily record cut at 2022-12-31)")
    for c in COUNTIES:
        for nid in (f"P2abc.r2.b1.july.{c}", f"P2abc.r2.b1.july_sd.{c}"):
            row(nid, C0F, "input", PF, "recorded C0 full values, Julys 2010-2025")
            row(nid, f"P2r2:july_units_{c}", "validator", PF,
                "truth = (Jul + Aug)/2 of the M2 phase-A meters of year Y")
    row("P2abc.r2.b1.m0_ratio", C0F, "input", PF, "M0 July/June ratios and July zero shares")
    row("P2abc.r2.c1.zero_port", C0F, "input", PP, "C0 Yunlin record in the pilot view")
    for p in PILOT_Z:
        row("P2abc.r2.c1.zero_port", p, "validator", PP, "pilot counts")
    for var, c0, pv in (("preorigin", C0P, PP), ("full", C0F, PF)):
        row(f"P2abc.r2.k1.first_assembly.{var}", f"{STG}/assembled_{var}", "input", pv,
            "first assembly (failed item f; never published; directory later overwritten by "
            "the re-assembly)")
        row(f"P2abc.r2.k1.first_assembly.{var}", c0, "validator", pv, "C0 reference")
        row(f"P2abc.r2.k1r.construction.{var}", pv, "input", pv, "published product")
        row(f"P2abc.r2.k1r.construction.{var}", c0, "validator", pv, "C0 reference")
        row(f"P2abc.r2.k1r.construction.{var}", C0L1, "validator", pv, "kept set, classes")
        for cty in ("yunlin", "changhua", "chiayi"):
            row(f"P2abc.r2.k1r.construction.{var}", f"{L0T}/{cty}_census.parquet",
                "validator", pv, "meter-number column only, for the privacy scan (item g)")
        row(f"P2abc.r2.build.{var}", c0, "input", pv, "stage-1 build (b1_build.py)")
        row(f"P2abc.r2.build.{var}", C0L1, "input", pv, "census attributes")
        row(f"P2abc.r2.facts.{var}", pv, "input", pv, "r5_resume.py facts (after the fact)")
        row(f"P2abc.r2.facts.{var}", c0, "validator", pv, "C0 reference for the spot checks")
        row(f"P2abc.r2.zeros.{var}", c0, "input", pv, "b2_zeros.py on C0 v3")
        row(f"P2abc.r2.checker.{var}", pv, "input", pv, "common.check re-hash")
        row(f"P2abc.r2.guard_probe.{var}", pv, "input", pv, "guard.open_product probes")
    row("P2abc.r2.d0.design_counts", C0P, "input", PP, "design-time counts (no estimate)")
    row("P2abc.r2.d0.design_counts", C0F, "input", PF, "design-time counts (no estimate)")
    # exemptions referenced by the pre-origin manifest
    for cty in ("yunlin", "changhua", "chiayi"):
        row("P2abc.r2.exemption.census_attributes.preorigin", f"{L0T}/{cty}_census.parquet",
            "input", PP, "exemption (U3) inherited from C0 v3: 2026 census attribute table")
        row("P2abc.r2.exemption.residual_unpaired_dec2022.preorigin",
            f"{L0T}/{cty}_kwh_long.parquet", "input", PP,
            "exemption (U3) inherited from C0 v3: residual unpaired Dec-2022 halves")
    row("P2abc.r2.exemption.census_attributes.preorigin",
        f"{DP}/reports/C0_evidence/repair3/checks/R9.json", "input", PP, "sensitivity R9")
    row("P2abc.r2.exemption.residual_unpaired_dec2022.preorigin",
        f"{DP}/reports/C0_evidence/repair3/residual_sensitivity.json", "input", PP,
        "sensitivity (repair3_residual.py)")
    # the two round-2 pre-registration rows were appended when registering (already present)


# ----------------------------------------------------------------------------- numbers
def skill_line(d: dict) -> str:
    gs = d["group_skill"]
    return ", ".join(f"n={n}: {gs[n]['chowlin']:.3f}" for n in gs)


def placebo_line(d: dict) -> str:
    ps = d.get("placebo_skill") or {}
    return ", ".join(f"n={n}: {v['mean']:.3f} (max {v['max']:.3f})" for n, v in ps.items())


def strata_line(d: dict) -> str:
    st = d["sd_coverage"]["strata"]
    return "; ".join(f"{k}: {v['coverage']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] "
                     f"{'pass' if v['pass'] else 'FAIL'}" for k, v in st.items())


def head_line(s: dict) -> str:
    p = s["primary_D_w"]
    return (f"mean D_w {p['mean']:+.4f}, 90 % bootstrap [{p['boot_q05_q95'][0]:+.4f}, "
            f"{p['boot_q05_q95'][1]:+.4f}], one-sided p {p['p_one_sided']:.3f} "
            f"({p['n_wells']} wells, {p['n_sites']} sites)")


def main(dry: bool = False) -> None:
    build_ledger()
    old = ledger._read(ledger.fragment_path(L.WP))
    cleaned = [ledger._clean(r, L.WP) for r in ROWS]
    bad = ledger.conflicts(old + cleaned)
    if bad:
        raise SystemExit(f"ledger conflicts: {bad}")
    if not dry:
        ledger.append_rows(L.WP, ROWS)

    fx = FACTS
    cp, cf = fx["cells_preorigin"], fx["cells_full"]
    st_p, st_f = ST1["preorigin"]["counties"], ST1["full"]["counties"]
    h2 = H2["summary"]
    h2f = H2["summary_fixed_nuisance"]
    h1 = H1["summary"]
    h1f = H1["summary_fixed_nuisance"]

    # ------------------------------------------------------------------ acceptance table
    acc = []

    def add(criterion, threshold, measured, passed, nid):
        acc.append({"criterion": criterion, "threshold": threshold, "measured": measured,
                    "passed": passed, "number_id": nid})

    for c in COUNTIES:
        d = A1P[c]
        if c == "chiayi":
            add("P2a meter validation, Chiayi (P2r2.a1.skill_chiayi, reported only: 16 "
                "candidate meters, groups of 30-300 cannot be formed; Chiayi's 316 pumps use "
                "the Yunlin profile and follow the Yunlin verdict as registered)",
                "skill >= 0.15 at every n in {1,3,10,30,100,300}",
                f"n=1/3/10 only: min {d['min_skill_scored']:.3f} ({d['n_validation_meters']} "
                "meters); n=30-300 not evaluable", False, f"P2abc.r2.a1.skill.{c}.preorigin")
        else:
            add(f"P2a meter validation, {c} (P2r2.a1.skill_{c}; pre-origin operator)",
                "pooled skill 1-SSE_CL/SSE_even >= 0.15 at every n in {1,3,10,30,100,300}",
                f"min {d['min_skill_scored']:.3f} over n ({d['n_validation_meters']} meters; "
                f"{skill_line(d)})", bool(d["passed"]), f"P2abc.r2.a1.skill.{c}.preorigin")
    for c in ("yunlin", "changhua"):
        sc = A1P[c]["sd_coverage"]
        add(f"Chow-Lin sd coverage, {c} (P2r2.a2.sdcov_{c}; plan 4.4/4.7, A1 coverage3 "
            "gate-3; sets the sd label only)",
            "every stratum's cluster-adjusted 95 % interval contains 0.90 and the point lies "
            "in 0.85-0.95 (n_eff >= 30), k >= 20 clusters",
            f"{strata_line(A1P[c])} -> label '{sc['label']}'",
            bool(sc["all_pass"]), f"P2abc.r2.a2.sdcov.{c}.preorigin")
    for c, key in (("yunlin", "yunlin"), ("changhua", "changhua")):
        add(f"P2a placebo-controlled head check, {c} (P2r2.a3.heads_{c}; not gating: P2a "
            "ships on the meter validation)", "one-sided site-bootstrap p < 0.05",
            head_line(h2[key]), bool(h2[key]["primary_pass_p_lt_0.05"]),
            f"P2abc.r2.a3.heads.{c}")
    add("P2a head check, round-1 registration P2a.head_placebo.v1 (Yunlin AMP_V2 energy, all "
        "wells; first scored in round 2)", "one-sided p < 0.05",
        head_line(h1["all_wells"]), bool(h1["all_wells"]["primary_pass_p_lt_0.05"]),
        "P2abc.r1.head_placebo")
    for c in COUNTIES:
        b = B1[c]["recorded_pair_mean"]
        add(f"P2b July backtest, {c} (P2r2.b1.july_{c}; ships the July-2026 imputation)",
            "mean |fan-total error| <= 20 % (rolling origin, Y = 2013..2025)",
            f"{b['mean_abs_total_err_pct']:.1f} % (max {b['max_abs_total_err_pct']:.1f} %; "
            f"2013-2022 {b['mean_abs_total_err_pct_2013_2022']:.1f} %; 1 km cell relative "
            f"RMSE {b['cell_rel_rmse_mean']:.3f})", bool(B1[c]["passed"]),
            f"P2abc.r2.b1.july.{c}")
    worst = max(C1.values(), key=lambda v: v["rel_diff"])
    add("P2c zero classification port (P2r2.c1.zero_port_yunlin; part of both gates)",
        "zero-month counts by run class and irrigation zero share in April / August each "
        "within 1 % of the pilot",
        "; ".join(f"{k} {v['mine']:.4g} vs {v['pilot']:.4g} ({100 * v['rel_diff']:.2f} %)"
                  for k, v in C1.items()) + f"; max {100 * worst['rel_diff']:.2f} %",
        all(v["pass"] for v in C1.values()), "P2abc.r2.c1.zero_port")
    for var, V in (("preorigin", VP), ("full", VF)):
        fa = V["P2r2.k1_first_assembly_failed"]
        add(f"Construction checks, first assembly ({var}; P2r2.k1.construction_{var})",
            "0 violations on items a-i", f"item f: {fa['item_f']['cell_violations']} cell "
            "sd violations (cells holding only zero-energy flag-3 or imputed pumps had sd "
            "NaN); all other items 0", False, f"P2abc.r2.k1.first_assembly.{var}")
        add(f"Construction checks, re-assembly ({var}; P2r2.k1r.construction_{var}, "
            "registered after the failure; product gate)", "0 violations on items a-i",
            "0 on every item (a pair sums, b C0 identity outside the split, c pre-install "
            "zeros, d recorded = C0 cells, e sums, f sd/flag rules, g privacy, h no full "
            "input in pre-origin, i zero table)",
            all(v == 0 for v in V["P2r2.k1r"]["violations"].values()),
            f"P2abc.r2.k1r.construction.{var}")
    add("Manifests re-hash (common.check, run again 2026-10-14)", "0 failures",
        f"2 manifests ok, {CHK['summary']['files_hashed']} files re-hashed, 0 failures "
        "(C0 v3 inputs: 10 manifests ok)", CHK["summary"]["manifests_failed"] == 0,
        "P2abc.r2.checker.preorigin")
    add("Pre-origin U3 / U5 (guard probe): pre-origin uses allowed only through the declared "
        "exemptions; scoring_target, selection, calibration_target, forcing_recorded_arm "
        "refused; full refused for forcing_frozen and model_design",
        "all probes as designed", "14/14 probes as designed (8 pre-origin, 6 full; guard_probe.txt)", True,
        "P2abc.r2.guard_probe.preorigin")

    # ------------------------------------------------------------------ key numbers
    kn = []

    def k(text, nid):
        kn.append(f"{text} [{nid}]")

    for c in ("yunlin", "changhua", "chiayi"):
        a = cp["annual_in_grid"][c]["mean_2012_2022"]
        k(f"{c}: in-grid electricity 2012-2022 {a['recorded_GWh_per_yr']:.2f} GWh/yr recorded "
          f"arm vs {a['chowlin_GWh_per_yr']:.2f} Chow-Lin arm (pre-origin); max annual "
          f"difference {100 * a['max_abs_rel_annual_diff']:.2f} % (Dec-Jan pairs at year "
          "ends)", "P2abc.r2.facts.preorigin")
    for c in ("yunlin", "changhua", "chiayi"):
        s = st_p[c]
        k(f"{c}: kept pumps {BIP['counties'][c]['n_pumps']:,} / meters "
          f"{BIP['counties'][c]['n_meters']:,}; Chow-Lin meters A {s['chowlin_meters']['A']:,}"
          f" + B {s['chowlin_meters']['B']:,}; pairs re-split {s['pairs_resplit']:,} "
          f"(pre-origin); flag-3 pump-months {BIP['counties'][c]['chowlin_pump_months_flag3']:,};"
          f" rho shares 0/0.5/0.9 = {s['rho_share']['0.0']:.2f}/{s['rho_share']['0.5']:.2f}/"
          f"{s['rho_share']['0.9']:.2f}", "P2abc.r2.build.preorigin")
    for c in ("yunlin", "changhua"):
        d = A1P[c]
        k(f"{c}: meter skill pre-origin {skill_line(d)}; year-shuffled placebo "
          f"{placebo_line(d)}", f"P2abc.r2.a1.skill.{c}.preorigin")
        k(f"{c}: pro-rata baseline skill n=1 {d['group_skill']['1']['prorata']:+.3f}, "
          f"fan {d['group_skill']['fan']['prorata']:+.3f}", f"P2abc.r2.a1.skill.{c}.preorigin")
        k(f"{c}: meter skill full variant (secondary) min {A1Fsec[c]['min_skill_scored']:.3f}",
          f"P2abc.r2.a1.skill.{c}.full")
        k(f"{c}: sd coverage pre-origin {strata_line(d)}; full: "
          f"{strata_line(A1F[c])}", f"P2abc.r2.a2.sdcov.{c}.preorigin")
    k(f"chiayi: meter skill (16 meters) {skill_line(A1P['chiayi'])}",
      "P2abc.r2.a1.skill.chiayi.preorigin")
    for c in ("yunlin", "changhua", "pooled"):
        k(f"head check round 2 {c}: {head_line(h2[c])}; fixed-nuisance secondary "
          f"{head_line({'primary_D_w': h2f[c]['primary_D_w']})}",
          f"P2abc.r2.a3.heads.{c}" if c != "pooled" else "P2abc.r2.a3.heads.pooled")
    k(f"head check round 1 (AMP_V2, all wells): {head_line(h1['all_wells'])}; fixed-nuisance "
      f"{head_line({'primary_D_w': h1f['all_wells']['primary_D_w']})}",
      "P2abc.r1.head_placebo")
    for c in COUNTIES:
        imp = B1[c]["recorded_pair_mean"]["imputation_sd"]
        k(f"{c}: July-2026 imputation sd LOYO coverage meter "
          f"{imp['meter']['loyo_coverage']['coverage']:.3f}, cell "
          f"{imp['cell']['loyo_coverage']['coverage']:.3f} (13 year clusters < 20: label "
          f"'{imp['meter']['label']}')", f"P2abc.r2.b1.july_sd.{c}")
    for c in ("yunlin", "changhua", "chiayi"):
        j = st_f[c].get("july2026_open_phaseA", {})
        k(f"{c}: July-2026 open phase-A meters {j.get('n_meters')}, imputed "
          f"{j.get('imputed_recorded')} (shipped {BIF['counties'][c]['july_imputed_pumps'] > 0})"
          f", left missing {j.get('left_missing')}", "P2abc.r2.build.full")
    for c in ("yunlin", "changhua"):
        m0 = T3["counties"][c]["m0_july_ratio"]
        k(f"{c}: M0 July/June ratio " + ", ".join(
            f"{y} {m0[y]['M0_jul_over_jun']:.3f}" for y in sorted(m0) if y <= "2025") +
          "; M2 July zero share " + ", ".join(
            f"{y} {m0[y]['zero_share_july_by_code']['M2']:.3f}" for y in sorted(m0)
            if y <= "2025"), "P2abc.r2.b1.m0_ratio")
    for c in ("yunlin", "changhua", "chiayi"):
        e = cp["energy_moved"][c]
        pw = fx["pump_window_totals_preorigin"][c]
        k(f"{c}: energy moved by the split (pre-origin) {pct(pw['energy_moved_share_in_window'])}"
          f" of pump energy 2011-2022, {pct(e['irrigation_cell_level_2012_2022'])} of in-grid "
          f"irrigation at cell level, {pct(e['irrigation_fan_total_level_2012_2022'])} at fan "
          f"level; per-pump window totals equal C0 to {pw['max_abs_total_diff_kwh']:.1e} kWh",
          "P2abc.r2.facts.preorigin")
        s = cp["irrigation_seasonality_2012_2022"][c]
        k(f"{c}: in-grid irrigation seasonality (CV of calendar-month means 2012-2022) "
          f"recorded {s['recorded']['cv_calendar_means']:.3f} (peak month "
          f"{s['recorded']['peak_month']}) vs Chow-Lin {s['chowlin']['cv_calendar_means']:.3f} "
          f"(peak month {s['chowlin']['peak_month']})", "P2abc.r2.facts.preorigin")
        r = cp["flag3_irrigation_cell_rel_sd"][c]
        k(f"{c}: flag-3 irrigation cell-month relative sd median {r['median_rel_sd']:.3f}, "
          f"q90 {r['q90_rel_sd']:.3f}", "P2abc.r2.facts.preorigin")
    k(f"recorded arm = C0 cells on {cp['recorded_vs_c0']['cell_months_compared']:,} "
      f"(pre-origin) and {cf['recorded_vs_c0']['cell_months_compared']:,} (full) cell-months "
      f"without imputation, max |d| {max(cp['recorded_vs_c0']['max_abs_diff_kwh'], cf['recorded_vs_c0']['max_abs_diff_kwh']):.1e} kWh, 0 flag or NaN differences",
      "P2abc.r2.facts.full")
    dd = cp["dec2022"]
    k(f"December 2022 in grid (pre-origin): recorded {dd['in_grid_total_kWh_recorded'] / 1e6:.2f}"
      f" GWh (= C0 v3), Chow-Lin {dd['in_grid_total_kWh_chowlin'] / 1e6:.2f} GWh; "
      f"{dd['cells_flag4_recorded']} cells carry C0's flag-4 fill", "P2abc.r2.facts.preorigin")
    jj = cf["jul2026"]
    k(f"July 2026 in grid (full): recorded {jj['in_grid_total_kWh_recorded'] / 1e6:.2f} GWh, "
      f"Chow-Lin {jj['in_grid_total_kWh_chowlin'] / 1e6:.2f} GWh (partial month; imputed or "
      "masked)", "P2abc.r2.facts.full")
    for c in ("yunlin", "changhua", "chiayi"):
        z = fx["zeros_preorigin"][c]
        k(f"{c}: pre-origin zero classes (2012-01..2022-12) " + ", ".join(
            f"{a} {b:,}" for a, b in z["by_class"].items()) +
          f"; terminated meters {z['terminated_meters']:,} (last-12-month energy "
          f"{z['terminated_last12_GWh']:.2f} GWh); synchronized-zero rows (flag 9) "
          f"{z['flag9_rows']:,} in months {', '.join(z['sync_months'])}",
          "P2abc.r2.zeros.preorigin")
    for c in ("yunlin", "changhua"):
        ccm = fx["county_class_monthly_preorigin"][f"{c}|in_grid|recorded"]
        k(f"{c}: in-grid domestic class {ccm['domestic']:.2f} of {ccm['all']:.2f} GWh/yr "
          f"2012-2022 ({100 * ccm['domestic'] / ccm['all']:.1f} %); no-domestic member "
          f"{ccm['all_nodomestic']:.2f}", "P2abc.r2.facts.preorigin")
    for c in ("yunlin", "changhua"):
        dc = D0[f"full/{c}"]
        k(f"{c}: design-time count (before registration) of M2 July zero share "
          + ", ".join(f"{y} {v:.3f}" for y, v in dc["july_zero_share_M2"].items())
          + " (the 2025 jump is TPC posting whole bimonthly bills to the second month)",
          "P2abc.r2.d0.design_counts")
    k("round 1 (Yunlin AMP_V2, design-time): meter skill min 0.229, sd coverage "
      "0.878-0.899, July 14.0 % (max 29.2 %), zero reproduction exact; round-1 scripts "
      "rerun identical in every compared file", "P2abc.r1.meter_validation")

    # ------------------------------------------------------------------ coverage of ids
    ids = sorted({a["number_id"] for a in acc} | {x.rsplit("[", 1)[1][:-1] for x in kn})
    ids += ["P2abc.r1.sd_coverage", "P2abc.r1.july", "P2abc.r1.zero_reproduction",
            "P2abc.r1.reproduce", "P2abc.r2.a3.heads.fixed_nuisance",
            "P2abc.r2.exemption.census_attributes.preorigin",
            "P2abc.r2.exemption.residual_unpaired_dec2022.preorigin"]
    miss = ledger.unledgered(ids, rows=old + cleaned)
    if miss:
        raise SystemExit(f"unledgered numbers: {miss}")
    if dry:
        print("dry run: rows", len(cleaned), "ids", len(set(ids)), "no conflicts")
        for a in acc:
            print("ACC", a["passed"], "|", a["criterion"][:90], "|", a["measured"][:200])
        for x in kn:
            print("KN", x[:260])
        return

    out = {"acc": acc, "kn": kn, "ids": sorted(set(ids)),
           "n_ledger_rows": len(ledger._read(ledger.fragment_path(L.WP)))}
    L.save_json(out, L.EVID / "report_tables.json")
    print("acceptance rows", len(acc), "key numbers", len(kn), "ids", len(set(ids)),
          "ledger rows", out["n_ledger_rows"])


if __name__ == "__main__":
    main(dry=sys.argv[1:] == ["--dry"])
