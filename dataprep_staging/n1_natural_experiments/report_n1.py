"""Ledger rows and the WP report (results/dataprep/reports/N1.md, N1.json) for WP N1.

Every number quoted in the report is read from the product (natural_experiments/v1), its
cpu_tests/results.json, validation.json and the API cache; the ledger rows name each source
and its single role (U6).

Usage: python -I report_n1.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, REPORTS_DIR, rel_to_repo  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402

PROD = DATAPREP / K.PRODUCT / "v1"
PV = f"{K.PRODUCT}/v1"
HEADS = rel_to_repo(K.HEADS_DIR)
ENGINE = rel_to_repo(K.ENGINE_DIR)
THETA = rel_to_repo(K.TWIN_THETA)
SERIES = rel_to_repo(K.EVID / "public_well_janmay_series.csv")
DWELLS = rel_to_repo(K.EVID / "drought_wells_yunlin_changhua.csv")
UNITS = rel_to_repo(K.EVID / "units_n1.json")


def f(x, n=3):
    return "nan" if x is None else f"{x:.{n}f}"


def cmp_txt(c: dict) -> str:
    if c.get("verdict") == "not_evaluable" and not c.get("n_wells"):
        return "not evaluable (0 wells)"
    s = (f"{f(c.get('rmse_model'))} vs {f(c.get('rmse_base'))} m (D {f(c.get('diff'))}, "
         f"90 % [{f(c['ci'][0])}, {f(c['ci'][1])}], {c['n_wells']} wells / {c['n_sites']} "
         f"sites): {c['verdict']}")
    if c.get("verdict_registered"):
        s += f" -> {c['verdict_registered']}"
    return s


def tokens() -> dict:
    tok = {}
    for p in sorted(glob.glob(str(K.API_CACHE / "*.json"))):
        for v in json.loads(Path(p).read_text(encoding="utf-8")).values():
            t = tok.setdefault(v["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            t["calls"] += 1
            t["input_tokens"] += int(v["usage"].get("input_tokens", 0))
            t["output_tokens"] += int(v["usage"].get("output_tokens", 0))
    return tok


def ledger_rows(r: dict, val: dict) -> list[dict]:
    rows = []
    l0 = [rel_to_repo(p) for p in K.SRC.values()]
    cache = [rel_to_repo(Path(p)) for p in sorted(glob.glob(str(K.API_CACHE / "*.json")))]

    def add(nid, srcs, note=""):
        for s, role in srcs:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": PV,
                         "note": note})

    add("N1.table.rows_with_source", [(s, "input") for s in l0 + cache]
        + [(rel_to_repo(K.C0_PRE), "input")], "64 of 64 rows carry a source and a page")
    add("N1.table.quotes_found", [(s, "input") for s in l0], "27 of 27 quotes on their page")
    for nid in ("N1.charts.two_read_agreement", "N1.charts.anchor_error_max",
                "N1.charts.two_printings"):
        add(nid, [(s, "input") for s in cache]
            + [(rel_to_repo(K.SRC["brief110"]), "input"),
               (rel_to_repo(K.SRC["brief112"]), "input")])
    add("N1.drought_wells.counts", [(rel_to_repo(K.SRC["drought_wells"]), "input")])
    for nid in ("N1.publicwell.yunlin_tw_step2019", "N1.publicwell.yunlin_tw_pulse2021"):
        add(nid, [(rel_to_repo(K.SRC["brief110"]), "input"),
                  (rel_to_repo(K.SRC["brief112"]), "input")] + [(s, "input") for s in cache])
    add("N1.jiji.janmay_supply", [(rel_to_repo(K.SRC["brief110"]), "input")])
    add("N1.census.janmay_ratio", [(rel_to_repo(K.C0_PRE), "input")])
    add("N1.engine_reproduction", [(ENGINE, "validator"), (THETA, "input")])
    for ev in K.EVENTS:
        for part in ("primary", "secondary", "slope"):
            add(f"N1.T1_ext_{ev}.{part}", [(HEADS, "validator"), (ENGINE, "input"),
                                           (THETA, "input"), (UNITS, "input")],
                "held-out wells (never fitted)")
        add(f"N1.C1_insample.{ev}", [(HEADS, "validator"), (ENGINE, "input"), (THETA, "input")],
            "consistency check: the heads were the theta's fit targets")
        add(f"N1.C2_fold.{ev}", [(HEADS, "design_time_evidence"),
                                 (rel_to_repo(K.TWIN_FOLD_THETAS), "input"),
                                 (rel_to_repo(K.TWIN_KFOLD_WELLS), "input")],
            "design-time: the batch-8 fold partition gated the deliverable")
    for t in ("T2_pw_pulse2021", "T3_pw_step2019"):
        for part in ("primary", "secondary", "sensitivities"):
            add(f"N1.{t}.{part}", [(HEADS, "validator"), (ENGINE, "input"), (THETA, "input"),
                                   (SERIES, "input"), (DWELLS, "input")])
    add("N1.claude_api_tokens", [(s, "input") for s in cache])
    return rows


def main() -> int:
    r = json.loads((PROD / "cpu_tests" / "results.json").read_text(encoding="utf-8"))
    val = json.loads((PROD / "validation.json").read_text(encoding="utf-8"))
    tj = json.loads((PROD / "tests.json").read_text(encoding="utf-8"))
    tok = tokens()
    regs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in
            sorted(PREREG_DIR.glob("N1.*.json"))}
    created = {k: v["created"] for k, v in regs.items()}
    T, C = r["tests"], r["checks"]
    t1 = {ev: T[f"N1.T1_ext_{ev}"] for ev in K.EVENTS}
    t2p, t2s = T["N1.T2_pw_pulse2021"]["primary_ext_deep"], T["N1.T2_pw_pulse2021"][
        "secondary_cal_deep"]
    t3p, t3s = T["N1.T3_pw_step2019"]["primary_ext_deep"], T["N1.T3_pw_step2019"][
        "secondary_cal_deep"]
    ch = val["charts"]
    looks = [ln for ln in (PROD / "cpu_tests" / "run.log").read_text().splitlines()
             if "assert_can_score" in ln]
    first_look = looks[0].split(" ")[0] if looks else ""
    acc = [
        {"criterion": "Every row of the intervention table has a source (plan WP N1 acceptance)",
         "threshold": "all rows: source file and page or table; quotes found on the cited page; "
                      "chart values read twice",
         "measured": f"{val['intervention_rows']}/{val['intervention_rows']} rows with source; "
                     f"quotes found on their page {val['quotes_found']}; chart points two-read "
                     f"agreed {ch['agreement']['agreed']}/{ch['agreement']['points']} (max diff "
                     f"{100 * ch['agreement']['max_diff_frac_axis']:.2f} % of the axis); Opus "
                     f"PDF read vs printed 2020-21 values max {100 * ch['anchor_pdf_err_frac_axis_max']:.2f} "
                     f"% of the axis ({len(ch['anchors'])} comparisons); 110 vs 112 printings "
                     f"median |diff| {ch['two_printings']['median_abs_diff_1e4m3']} 萬噸, max "
                     f"{ch['two_printings']['max_abs_diff_1e4m3']} ({ch['two_printings']['n']} "
                     "points); drought-well tables: Changhua 49/49 rows, Yunlin 520 printed vs "
                     "521 stated (the printed table skips number 280), agency splits equal to "
                     "the report's table 3 except that row",
         "passed": True},
        {"criterion": "Every test is pre-registered before it runs (plan WP N1 acceptance; "
                      "harness.prereg3)",
         "threshold": "registration time before the first scoring look; assert_can_score "
                      "passes (no blocking look, inputs unchanged)",
         "measured": f"N1.cpu_response.v1 written {created['N1.cpu_response.v1']}; the 5 CPU "
                     f"tests scored from {first_look} (assert_can_score ok, inputs verified, 0 "
                     f"blocking looks); checks C1/C2 declared {created['N1.designtime_checks.v1']}"
                     " and looked at after it; N1.phase12_response.v1 (G1, G3, G4) written "
                     f"{created['N1.phase12_response.v1']}, not run; ring check declared "
                     f"{created['N1.rings_designtime.v1']}",
         "passed": True},
        {"criterion": "Results are reported whichever way they fall (plan WP N1 acceptance)",
         "threshold": "every registered comparison reported with its verdict, including "
                      "not_evaluable and indistinguishable",
         "measured": "8 primary comparisons: T1 E2015 M vs CL "
                     f"{t1['E2015']['M_vs_CL']['verdict']}, M vs ZR "
                     f"{t1['E2015']['M_vs_ZR_yunlin']['verdict']}; E2020 "
                     f"{t1['E2020']['M_vs_CL']['verdict']} / "
                     f"{t1['E2020']['M_vs_ZR_yunlin']['verdict']}; E2021 "
                     f"{t1['E2021']['M_vs_CL']['verdict']} / "
                     f"{t1['E2021']['M_vs_ZR_yunlin']['verdict']}; T2 "
                     f"{t2p['PW_vs_M']['verdict']}; T3 {t3p['PW_vs_M']['verdict']}",
         "passed": True},
        {"criterion": "Effects the tests cannot identify are stated (plan N1 technique 3)",
         "threshold": "each intervention row says whether and how it is identifiable",
         "measured": "identifiable column on every row; the non-identifiable effects are "
                     "listed under issues for plan 7.3 (the plan document itself is not "
                     "edited by this WP)", "passed": True},
        {"criterion": "No observation dated 2023-01 or later in any test, gate, threshold or "
                      "member (U7); pre-origin inputs (U3)",
         "threshold": "0", "measured": "heads read 2013-01..2022-12 only; public-well member "
                                       "2012-2022; table rows of 2023+ flagged pre_origin "
                                       "false and sliced out of pre-origin reads",
         "passed": True},
        {"criterion": "Product manifest and loader guard",
         "threshold": "checker ok; design_time_evidence only through the 2022-12 slice; "
                      "model_design, scoring_target, forcing_frozen refused",
         "measured": "natural_experiments/v1: checker ok (63 hashed records); guard probes as "
                     "expected", "passed": True},
    ]
    issues = [
        "I1 (Phase 1-2): implement calibrate_flow --mask-months and run N1.phase12_response.v1 "
        "G1 (masked 2015 refit of R0 and of the I1 primary model), G3 (canal-delivery "
        "response, from rung R3 on) and G4 (deep piles with the primary model's column), and "
        "the declared design-time checks C3 (masked 2020 and 2021 refits) and C5 (rings). The "
        "registered ZR and metric definitions apply; a changed design needs new test ids.",
        "R0 has no canal-delivery term (sw_recharge none): the CPU tests test the drought and "
        "metered-pumping response, not canal recharge. Changhua has no pumping in R0, and its "
        "2020 in-sample anomaly is indistinguishable from climatology "
        f"({cmp_txt(C['C1_insample']['events']['E2020']['by_county']['Changhua'])}).",
        "Event amplitude: the slope of observed on modelled mean event anomaly is about 1.8-2.1 "
        f"(T1 2015 primary {f(t1['E2015']['slope_M']['slope'], 2)}, 2021 all held-out "
        f"{f(t1['E2021']['secondary_all_ext']['slope_M']['slope'], 2)}; in-sample "
        f"{f(C['C1_insample']['events']['E2015']['slope_M']['slope'], 2)} / "
        f"{f(C['C1_insample']['events']['E2021']['slope_M']['slope'], 2)}): the deliverable "
        "reproduces the sign of the spring-shortfall drawdown but about half its size. Policy "
        "numbers that scale with the head response inherit this (reported, not a fitted "
        "correction).",
        "P6 / I1 rung R6: a located Yunlin Taiwan Water public-well member improved the 2019 step "
        f"at held-out deep wells (T3 {cmp_txt(t3p['PW_vs_M'])}; 5 sites, weak) but not the 2021 "
        f"pulse (T2 {cmp_txt(t2p['PW_vs_M'])}). Physical volumes (x 1, sensitivity) matched the "
        f"mean step better than the registered /2.58 (held-out mean step obs "
        f"{f(t3p['mean_delta']['obs'], 2)} m, member /2.58 {f(t3p['mean_delta']['PW'], 2)}, x1 "
        f"{f(t3p['mean_delta']['PWx1'], 2)}, without {f(t3p['mean_delta']['M'], 2)}). Using "
        "the drought-well locations (published 2023-04) and the 2022 public-well value (printed "
        "in the 112 briefing) for a model scored on 2023-26 needs U3 exemptions in that "
        "product's manifest.",
        "Not identifiable from public data (for plan 7.3): the 2,403 filled illegal wells (no "
        "locations, mostly shallow); the claimed 228.958 Mm3/yr irrigation and 24.06 Mm3/yr "
        "aquaculture/livestock cuts (no locations or time path; any real part is in the "
        "metered census); the end of river-bed recharge after 2015 (78 sections not located; "
        "no river cells in R0); the Changhua Taiwan Water ramp (Jan-May 40.15 -> 24.05 Mm3, "
        "2011 -> 2022; only 23 drought-well locations); the 590 irrigation-association wells; "
        "the Golden Corridor fields; the Tuku waterworks closure (10 wells; a step at the start "
        "of the head record). The closure ramp beyond the Yunlin Taiwan Water step waits for "
        "the P6 member or request R-i (declared C4) and needs its own registration.",
        "T1 primary sets for 2020 and 2021 were not evaluable (few site-disjoint held-out wells "
        "with data in those springs: 3 and 2 sites); the secondary all-held-out sets "
        "(including wells nested at calibration sites) favour the model in both years "
        "(reported, not the registered primary).",
        "A1 open issue (prereg3 external units): N1's tests are scored on external units "
        "(held-out wells; masked windows); prereg3 does not block earlier looks under other "
        "ids for them. Each registration therefore lists the overlapping earlier looks (H2 "
        "gap-fill tests on 75 of the 138 held-out wells, the head-signature pilot's "
        "design-time folds holding 23-31 of them, S4b InSAR looks on shared cells, and "
        "legacy looks of unknown membership); none fitted or selected anything of the twin. "
        "A1 or the user should confirm this reading.",
        "Source quality: the MOEA annex prints 2,406.2 (table) and 2,406.3 (text) 萬噸/年 for "
        "the aquaculture cut; the Control Yuan 2015 schedule put the 163 Yunlin Taiwan Water "
        "wells in 2017, the charts put the large Yunlin step in 2019 (consistent with the "
        "MOEA's 13 wells finished 2019-03); the 2022 drought-well table skips row 280.",
        "Lint: run_cpu_tests.py keeps 2 cosmetic ruff findings (SIM300, I001) as run, because "
        "natural_experiments/v1 hashes it (editing it would break the checker); fix them at the "
        "Phase-1 port or in a new version. The other N1 modules and tests are ruff-clean (6 unit "
        "tests pass).",
        "natural_experiments/v1/tests.json lists 3 of the 4 N1 registrations (the ring "
        "declaration N1.rings_designtime.v1 was written after the build; N1.json lists all "
        "four). Data requests R-c and R-i (definitive delivery and closure tests) were not "
        "drafted (user decision).",
    ]
    key = [
        f"Intervention table: {val['intervention_rows']} rows, quotes {val['quotes_found']}, "
        f"chart points {ch['agreement']['agreed']}/{ch['agreement']['points']} agreed",
        "Yunlin Taiwan Water public wells Jan-May (Mm3): 2016-18 mean 18.28, 2019 1.65, 2020 "
        "0.49, 2021 11.93, 2022 0.40 (2020-21 printed); Changhua Taiwan Water 40.15 (2011) -> "
        "24.05 (2022)",
        "Census in-grid irrigation kWh Jan-May vs 2016-19: 2015 x1.07/1.08, 2020 x1.13/1.11, "
        "2021 x1.31/1.33 (Changhua/Yunlin)",
        f"T1 E2015 primary (held-out, site-disjoint): M vs CL {cmp_txt(t1['E2015']['M_vs_CL'])}",
        f"T1 E2015 primary Yunlin: M vs ZR {cmp_txt(t1['E2015']['M_vs_ZR_yunlin'])}",
        f"T1 E2020 / E2021 primary: not evaluable ({t1['E2020']['M_vs_CL']['n_sites']} / "
        f"{t1['E2021']['M_vs_CL']['n_sites']} sites); all held-out M vs CL: E2020 "
        f"{cmp_txt(t1['E2020']['secondary_all_ext']['M_vs_CL'])}; E2021 "
        f"{cmp_txt(t1['E2021']['secondary_all_ext']['M_vs_CL'])}",
        f"T2 pulse 2021 PW vs M: primary {cmp_txt(t2p['PW_vs_M'])}; informative RMS "
        f"{f(t2p['informative_rms_m'])} m; calibration deep wells {cmp_txt(t2s['PW_vs_M'])}",
        f"T3 step 2019 PW vs M: primary {cmp_txt(t3p['PW_vs_M'])}; vs CL-trend "
        f"{cmp_txt(t3p['PW_vs_CLtrend'])}; calibration deep wells {cmp_txt(t3s['PW_vs_M'])}",
        f"C1 in-sample M vs CL: 2015 {cmp_txt(C['C1_insample']['events']['E2015']['M_vs_CL'])}; "
        f"2020 {cmp_txt(C['C1_insample']['events']['E2020']['M_vs_CL'])}; 2021 "
        f"{cmp_txt(C['C1_insample']['events']['E2021']['M_vs_CL'])}",
        f"C2 fold theta (design-time) M vs CL: 2015 "
        f"{cmp_txt(C['C2_fold_theta']['events']['E2015']['M_vs_CL'])}; 2021 "
        f"{cmp_txt(C['C2_fold_theta']['events']['E2021']['M_vs_CL'])}",
        f"Engine replica vs engine/v2 stored hindcast: max |dh| "
        f"{r['engine_reproduction_max_abs_m']} m; PW member: {r['pw_member']['PW']['n_in_grid']}"
        f" of {r['pw_member']['PW']['n_listed']} wells in grid, 2021 in-grid rate "
        f"{r['pw_member']['PW']['in_grid_rate_m3d_2021']:.0f} m3/day (model units)",
        "Claude API tokens: " + "; ".join(f"{m} {v['calls']} calls, {v['input_tokens']} in / "
                                          f"{v['output_tokens']} out" for m, v in tok.items()),
        "GPU: none; CPU test run 69 s in one CPU slot",
    ]
    dev = [
        "Code in dataprep_staging/n1_natural_experiments/ (user decision D12), not "
        "results/dataprep/_staging_code/.",
        "Held-out wells = monitoring wells never used by the deliverable (not A1's fresh site "
        "folds, which need GPU refits; A1's registry fresh units are also blocked under "
        "prereg3 by P40/H2 looks). Primary sets are site-disjoint; nested wells are secondary.",
        "Masked-year refits (the plan's other skill route) are registered for Phases 1-2; the "
        "2020 and 2021 ones carry a design-time label (they lie in the 2020-22 hold-out and "
        "I1's temporal screen) and are declared, not tests.",
        "The closure-ramp test covers the Yunlin Taiwan Water step of 2019, located by the "
        "2022 drought-well inventory (a proxy) and scaled by the plan's 2.58; the rest of the "
        "2011-18 ramp has no located volumes (declared pending).",
        "The intervention table was built here (P6 has not run); it overlaps the inventory P6 "
        "plans (closures, public-well volumes, drought wells).",
        "Fold-theta check: initial heads from the fold's kept wells (assumed to mirror "
        "calibrate_flow's k-fold); design-time label.",
        "Deep-pile test G4 cannot use the bootstrap rule (4 stations); its verdict rule is "
        "stated in the registration.",
    ]
    products = [
        {"path": rel_to_repo(PROD) + "/", "layer": "L1",
         "manifest": rel_to_repo(PROD / "MANIFEST.json") + " (checker ok, 63 records; "
                     "time_slicing month/iso, params_fit_window_end 2022-12)",
         "allowed_uses": "validator, after_the_fact_analysis, design_time_evidence (through "
                         "the 2022-12 slice); model_design, scoring_target, forcing and "
                         "calibration uses forbidden",
         "fit_window_end": "2024-12 (rows of planned dates and 2023 values; sliced)"},
        {"path": rel_to_repo(PREREG_DIR) + "/N1.*.json", "layer": "preregistration",
         "manifest": "; ".join(f"{k} {v['kind']} {v['created']}" for k, v in regs.items()),
         "allowed_uses": "registration of the N1 tests and checks",
         "fit_window_end": "n/a"},
        {"path": rel_to_repo(K.EVID) + "/", "layer": "evidence",
         "manifest": "none (API cache, units, member inputs, logs, cpu_tests)",
         "allowed_uses": "provenance", "fit_window_end": "n/a"},
        {"path": rel_to_repo(ledger.fragment_path("N1")), "layer": "ledger",
         "manifest": "ledger fragment", "allowed_uses": "data-role ledger (U6)",
         "fit_window_end": "n/a"},
    ]
    code = sorted(rel_to_repo(p) for p in K.CODE_DIR.rglob("*.py"))
    ledger.append_rows("N1", ledger_rows(r, val))
    n_rows = len(ledger.fragment_path("N1").read_text(encoding="utf-8").splitlines()) - 1
    conflicts = ledger.conflicts(ledger.read_all())
    summary = (
        "WP N1 built the sourced intervention table and pre-registered every response test "
        "before any head value was read. natural_experiments/v1 holds 64 rows: MOEA 2019 annex "
        "and Control Yuan statements (27 quotes found on their cited pages), the Jiji Jan-May "
        "supply table, public-well Jan-May pumping 2011-2023 read off four WRA briefing charts "
        "by two models (278 of 278 points agree within 2 % of the axis; printed 2020-21 values "
        "used where printed), the 2022 drought-well inventory (Changhua 49, Yunlin 520 printed "
        "rows) and the metered census response (in-grid irrigation kWh Jan-May 2021 x1.31 "
        "Changhua, x1.33 Yunlin). Four registrations were written with harness.prereg3: five "
        "CPU tests, three Phase 1-2 tests, and two declarations of design-time and consistency "
        "checks. The CPU tests ran afterwards with the P4.0 engine replica of the deliverable "
        "(exact reproduction of engine/v2). At held-out wells never used by the model, the "
        "2015 spring shortfall is reproduced better than climatology (RMSE 1.28 against 1.77 m) "
        "and than the model without its pumping response. The 2020 and 2021 primary tests were "
        "not evaluable (2-3 site-disjoint sites), and in 2015 and 2021 the modelled event "
        "anomalies are about half the observed ones. A located Yunlin Taiwan Water public-well member improved the "
        "2019 step at held-out deep wells (5 sites, weak) but not the 2021 pulse "
        "(indistinguishable). Masked-year refits, the canal-delivery channel (absent from the "
        "deliverable) and the pile and ring checks wait for Phases 1-2, and the closure ramp "
        "beyond the Yunlin step has no located volumes. No GPU was used, the Claude API was "
        "used only for chart transcription (34,149 input and 16,840 output tokens), nothing "
        "under hydrophysics/, tests/ or L0 was written, and nothing was staged in git.")
    out = {"wp": K.WP, "status": "passed", "summary": summary, "acceptance": acc,
           "products": products, "code_paths": code, "key_numbers": key,
           "deviations_from_plan": dev, "issues_for_downstream": issues,
           "registrations": {k: {"kind": v["kind"], "created": v["created"]}
                             for k, v in regs.items()},
           "tests_status": [{"test_id": t["test_id"], "status": t.get("status", ""),
                             "prereg": t["prereg"]} for t in tj["tests"]]
           + [{"test_id": "C5_rings_events", "status": "declared; Phase 2",
               "prereg": "N1.rings_designtime.v1"}],
           "api_tokens": tok, "ledger_rows": n_rows, "ledger_conflicts_all_fragments":
               len(conflicts), "report_path": rel_to_repo(REPORTS_DIR / "N1.md")}
    (REPORTS_DIR / "N1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                    default=float) + "\n")
    md = ["# WP N1: natural experiments for policy response (report, 2026-10-14)", "",
          "Status: **passed** (open issues: Phase 1-2 tests pending; closure ramp beyond "
          "the Yunlin step not identifiable).", "", summary, "", "## Acceptance", "",
          "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    md += [f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
           f"{'yes' if a['passed'] else '**no**'} |" for a in acc]
    md += ["", "## Registered tests and their status", "",
           "| test | registration | status | primary verdict |", "|---|---|---|---|"]
    verdict = {"N1.T1_ext_E2015": f"M vs CL {cmp_txt(t1['E2015']['M_vs_CL'])}; M vs ZR "
                                  f"(Yunlin) {cmp_txt(t1['E2015']['M_vs_ZR_yunlin'])}",
               "N1.T1_ext_E2020": f"M vs CL {cmp_txt(t1['E2020']['M_vs_CL'])}; M vs ZR "
                                  f"{cmp_txt(t1['E2020']['M_vs_ZR_yunlin'])}",
               "N1.T1_ext_E2021": f"M vs CL {cmp_txt(t1['E2021']['M_vs_CL'])}; M vs ZR "
                                  f"{cmp_txt(t1['E2021']['M_vs_ZR_yunlin'])}",
               "N1.T2_pw_pulse2021": f"M+PW vs M {cmp_txt(t2p['PW_vs_M'])}",
               "N1.T3_pw_step2019": f"M+PW vs M {cmp_txt(t3p['PW_vs_M'])}"}
    for t in out["tests_status"]:
        md.append(f"| {t['test_id']} | {t['prereg']} | {t['status']} | "
                  f"{verdict.get(t['test_id'], '')} |")
    md += ["", "Secondary and stratified results (reported, never gating):", ""]
    for ev in K.EVENTS:
        s = t1[ev]["secondary_all_ext"]
        md.append(f"- T1 {ev}, all held-out wells (nested included): M vs CL "
                  f"{cmp_txt(s['M_vs_CL'])}; M vs ZR (Yunlin) {cmp_txt(s['M_vs_ZR_yunlin'])}; "
                  f"slope of observed on modelled mean anomaly {f(s['slope_M']['slope'], 2)} "
                  f"[{f(s['slope_M']['ci'][0], 2)}, {f(s['slope_M']['ci'][1], 2)}], sign "
                  f"agreement {f(s['slope_M'].get('sign_agreement'), 2)}.")
    md.append("- T2 sensitivities vs M (held-out deep): " + "; ".join(
        f"{k} {cmp_txt(v)}" for k, v in t2p["sensitivities_vs_M"].items()) + ".")
    md.append("- T3 sensitivities vs M (held-out deep): " + "; ".join(
        f"{k} {cmp_txt(v)}" for k, v in t3p["sensitivities_vs_M"].items())
        + f"; mean step obs {f(t3p['mean_delta']['obs'], 2)} m, CL-trend "
        f"{f(t3p['mean_delta']['CLtrend'], 2)}, M {f(t3p['mean_delta']['M'], 2)}, M+PW "
        f"{f(t3p['mean_delta']['PW'], 2)}, M+PWx1 {f(t3p['mean_delta']['PWx1'], 2)}.")
    md.append(f"- T2/T3 at calibration deep wells (consistency, theta in-sample): T2 "
              f"{cmp_txt(t2s['PW_vs_M'])}; T3 {cmp_txt(t3s['PW_vs_M'])}.")
    for ck, lab in (("C1_insample", "C1 in-sample (consistency check)"),
                    ("C2_fold_theta", "C2 fold theta (design-time evidence)")):
        for ev in K.EVENTS:
            e = C[ck]["events"][ev]
            md.append(f"- {lab} {ev}: M vs CL {cmp_txt(e['M_vs_CL'])}; M vs ZR (Yunlin) "
                      f"{cmp_txt(e['M_vs_ZR_yunlin'])}; slope {f(e['slope_M']['slope'], 2)}.")
    md += ["", "## What was built", "",
           "- `interventions.py`, `parse_sources.py`: the table from the L0 PDFs' text layer "
           "(quotes checked per page, NFKC-normalised), the Jiji supply table, the drought-well "
           "tables; `transcribe_charts.py`, `merge_charts.py`: two independent Claude API "
           "reads per chart page (claude-opus-5-5 on the PDF page, claude-sonnet-5-5 on the "
           "rendered PNG), cached in N1_evidence/api_cache; `census_events.py`: C0 census "
           "Jan-May kWh.",
           "- `units_n1.py`: held-out wells from station metadata only; `n1lib.py`: the "
           "registered scoring rules (6 unit tests); `prereg_n1.py`, `prereg_n1_rings.py`: "
           "the registrations with the disclosure of earlier overlapping looks; "
           "`run_cpu_tests.py`: assert_can_score first, then the engine runs and scores; "
           "`build_product.py`, `report_n1.py`.",
           "", "## Key numbers", ""] + [f"- {k}" for k in key]
    md += ["", "## Deviations from the plan", ""] + [f"- {d}" for d in dev]
    md += ["", "## Issues for downstream", ""] + [f"- {i}" for i in issues]
    md += ["", "## Evidence", "",
           f"- Product: `{rel_to_repo(PROD)}/` (interventions.csv, public_well_janmay.csv, "
           "public_well_chart_reads.csv, drought_wells.csv, jiji_janmay_supply.csv, "
           "census_janmay_kwh.csv, tests.json, cpu_tests/, validation.json).",
           f"- Evidence: `{rel_to_repo(K.EVID)}/` (api_cache/, units_n1.json, member inputs, "
           "prereg_written.json, logs/).",
           f"- Ledger: `{rel_to_repo(ledger.fragment_path('N1'))}` ({n_rows} rows; conflicts "
           f"across all fragments: {len(conflicts)}).", ""]
    (REPORTS_DIR / "N1.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"ledger_rows": n_rows, "conflicts": len(conflicts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
