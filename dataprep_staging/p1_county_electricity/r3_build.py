"""WP P1 repair round (r3): package ``county_electricity/census_check/v2``, fix the P1 ledger
fragment, write the ledger rows of the repair round, and probe the loader guard.

census_check/v2 = census_check/v1 (truth tables unchanged) with:

* ``evaluation.json``: the round-2 numbers unchanged, plus the repair labels. T10 and T11 are
  after-the-fact checks, not fresh-unit tests: ``inspection/profile2.json``, opened by P1 before
  ``P1r2.census_tests`` was written, held the truth of both (P1 saw the Changhua part, as its
  pre-registration says; the Yunlin part was computed before the pre-registration and P1's
  statement that it was not read cannot be verified). The corrective look-log seeds and the A1
  tool's verdict (``assert_can_score`` now refuses both) are recorded under ``r3_test_status``.
  Also: the two round-2 scoring runs, the like-for-like domestic sensitivity, and the
  pre-install coding sensitivity (``census_sensitivity.json``).
* ``diagnostics_post_hoc.json``: v1 content plus the C9 single-pump finding, census energy per
  non-zero meter, and the T10 explanation restated as hypotheses.
* the libraries the evaluation used (guard, uses, flags, harness.prereg) are now listed as code
  (WP A0fix finished editing guard.py at 05:53Z, before round 2 ran).

Usage: python -I r3_build.py <evidence_dir>
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import guard, ledger, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import LEDGER_ROWS, REPO, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

import data as dmod  # noqa: E402
import r3_preorigin_v2 as r3p  # noqa: E402

CE = "results/dataprep/county_electricity"
NEW = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026"
CEN = f"{NEW}/parquet"
CC1, PRE1, FULL1 = f"{CE}/census_check/v1", f"{CE}/preorigin/v1", f"{CE}/full/v1"
OBS = f"{CE}/tpc_observed/v1"
PRODUCT = "county_electricity/census_check"
TPC = "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv"
NAT = ["data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/set_236.csv",
       "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/set_256.csv"]
AMP_KWH, AMP_PUMPS = "AMP_V2/data/pump_kwh_all.parquet", "AMP_V2/data/tpc_pumps.parquet"
BUILD3 = REPO / CE / "_build_r3"
LIB = HERE.parent
CODE = [HERE / n for n in ("r3_build.py", "r3_census_sensitivity.py", "r3_looks.py",
                           "r3_preorigin_v2.py", "census_truth.py", "r2_evaluate.py",
                           "r2_diagnostics.py", "r2_build.py", "r2_prereg.py", "lib.py",
                           "data.py")] + [
    REPO / "hydrophysics/twin/pumping.py", REPO / "hydrophysics/twin/scenario.py",
    LIB / "common/manifest.py", LIB / "common/ledger.py", LIB / "common/hashing.py",
    LIB / "common/guard.py", LIB / "common/uses.py", LIB / "common/flags.py",
    LIB / "harness/prereg.py"]
COPY = ["meter_counts.parquet", "truth_info.json", "annual_comparison.parquet",
        "truth_series_used.parquet"]
SCORING_RUNS = [
    {"time": "2026-10-12T06:06:05Z", "looks": "looks.jsonl lines for P1r2.T10/T11 at 06:06:05Z",
     "outputs": "overwritten by the second run"},
    {"time": "2026-10-12T06:08:36Z", "looks": "looks.jsonl lines for P1r2.T10/T11 at 06:08:36Z",
     "outputs": "published in census_check/v1 (evaluation.json 06:08:36Z)",
     "note": "r2_evaluate.py and r2_diagnostics.py were edited at 06:08:28Z, between the runs; "
             "the edit is not recoverable (no version control of the staging folder). The "
             "published values equal the verifier's independent recomputation from the "
             "pre-registered definitions, and the candidates (v1 files, hashed in the "
             "pre-registration), thresholds and month rule are fixed in the hashed "
             "pre-registration, so the edit cannot have changed what was scored."}]

CARD = """# county_electricity/census_check/{pv}

Measured pump electricity of Changhua and Yunlin (Taiwan Power pump census, obtained
2026-10-12) in model units, used by WP P1 to score the frozen v1 inference
(`county_electricity/{{preorigin,full}}/v1`). After-the-fact analysis only. Re-issue of
census_check/v1 (repair round, 2026-10-12): the truth values are unchanged.

- **Content.** `census_truth_monthly.parquet` (county x variant x series x scope x class x month,
  GWh), `meter_counts.parquet`, `evaluation.json` (round-2 readings plus the repair labels),
  `diagnostics_post_hoc.json` (post hoc only), `census_sensitivity.json` (pre-install coding,
  C9 extra rows, energy per meter), `annual_comparison.parquet`, `truth_series_used.parquet`,
  `truth_info.json`.
- **T10 and T11 are after-the-fact checks**, not fresh-unit out-of-sample tests: the truth of
  both was in `inspection/profile2.json` before `P1r2.census_tests` was written (P1 saw the
  Changhua part). Their numbers are unchanged (T10 +15.8 %, fails the 10 % rule; T11 +8.6 %).
  The A1 look log now carries the corrective seeds and the tool refuses both as pre-registered
  scores (`evaluation.json` r3_test_status).
- **Flags.** 0 (observed census aggregates); 9 for 2026-07 (partial month).
- **Gate.** Construction check (pre-registered C9 'identical' and preorigin == full): passed =
  {gate}. C9 is not identical because the new Yunlin sheet carries one pump (222 pump-months,
  8.3 MWh in total) that has no kWh row in the AMP_V2 cache; all common rows agree exactly.
- **Allowed uses.** after_the_fact_analysis only. WP C0 builds the census forcing product.
- **Licence.** The census is restricted (local research use, no redistribution; SOURCE.json).
- **Personal data.** None: county x class x month totals and monthly meter counts only.
"""


def ledger_fix(evd: Path) -> dict:
    """Rewrite the 10 tpc_observed rows whose product_version used the repo-relative form."""
    path = ledger.fragment_path("P1", LEDGER_ROWS)
    text = path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    bad = f"results/dataprep/{OBS.split('results/dataprep/')[1]}"
    n = sum(r["product_version"] == bad for r in rows)
    if n == 0:
        return {"rows_fixed": 0, "note": "already fixed"}
    sup = path.parent.parent / "superseded" / "P1.r2.csv"
    if not sup.exists():
        shutil.copy2(path, sup)
    good = OBS.split("results/dataprep/")[1]
    for r in rows:
        if r["product_version"] == bad:
            r["product_version"] = good
    if ledger.conflicts(rows):
        raise SystemExit("ledger conflicts after the fix")
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ledger.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ledger.COLUMNS})
    os.replace(tmp, path)
    return {"rows_fixed": n, "from": bad, "to": good, "old_fragment": rel_to_repo(sup)}


def t3_port_check() -> dict:
    """After the fact: r3_preorigin_v2.loyo_backcast reproduces v1's T3 on the full data."""
    full = dmod.load("full")
    got = r3p.loyo_backcast(full["d"], range(2016, 2026))
    ref = json.loads((REPO / FULL1 / "validation.json").read_text(encoding="utf-8"))
    ref = ref["tests"]["T3_BV_LOYO"]
    diff = max(abs(got[c]["annual_err_pct"][y] - ref[c]["annual_err_pct"][str(y)])
               for c in ("CH", "YL") for y in range(2016, 2026))
    return {"max_abs_diff_pct_points": diff, "identical": diff == 0.0,
            "note": "evidence only (uses full-variant data); not written into any pre-origin "
                    "product"}


def evaluation_v2(sens: dict, looks: dict) -> dict:
    ev = json.loads((REPO / CC1 / "evaluation.json").read_text(encoding="utf-8"))
    c1 = ev["C1_a_vs_census"]
    lab = ("after_the_fact_check: the truth of these months was in inspection/profile2.json "
           "before P1r2.census_tests was written; the candidate, threshold and month rule were "
           "fixed before scoring, but the units were not fresh (repair round)")
    ev["T10_ext_yunlin"]["status"] = lab + "; P1 declared it had not read the Yunlin part " \
        "(unverifiable)"
    ev["T11_ext_changhua"]["status"] = lab + "; P1 had seen the Changhua county totals " \
        "(declared in the pre-registration's prior_exposure)"
    pc = sens["changhua_preinstall_coding"]["codings"]
    ev["r3_test_status"] = {
        "P1r2.T10_ext_yunlin": {"label": "after-the-fact check", "pass_rule_result": "fail",
                                "sum_err": ev["T10_ext_yunlin"]["sum_err"]},
        "P1r2.T11_ext_changhua": {"label": "after-the-fact check", "pass_rule_result": "pass",
                                  "sum_err": ev["T11_ext_changhua"]["scaled_apre"]["sum_err"]},
        "why": "verifier major issue: P1r2.census_tests prior_exposure states that P1 read "
               "inspection/profile2.json before the pre-registration, and that file holds "
               "changhua_last14_cleaned_GWh (the T11 truth) and yunlin_last14_cleaned_GWh (the "
               "T10 truth). The round-2 seed named only 'changhua_census_vs_P1_v1', so the A1 "
               "tool could not block T11, and the round-2 report wrongly said the tool cleared "
               "both tests because no prior look existed on their units.",
        "look_log_corrections": looks["appended"],
        "assert_can_score_now (log=False)": {
            k: {"can_score_as_preregistered": v["can_score_as_preregistered"]}
            for k, v in looks["assert_can_score_log_false"].items()},
        "pre_registration_unchanged_sha256": looks["prereg_file_unchanged"],
        "what_still_holds": "T10's failure is informative either way: exposure cannot have "
                            "improved a frozen v1 candidate (built 02:43Z, before the census "
                            "arrived)."}
    ev["r3_scoring_runs"] = SCORING_RUNS
    c1["r3_domestic_sensitivity"] = {
        "headline_round2": "Changhua without domestic / Yunlin with domestic",
        "preorigin_changhua_only_excluded": c1["preorigin"][
            "changhua_domestic_excluded_over_yunlin_all"],
        "preorigin_both_excluded_like_for_like": c1["preorigin"]["both_domestic_excluded"],
        "full_changhua_only_excluded": c1["full"]["changhua_domestic_excluded_over_yunlin_all"],
        "full_both_excluded_like_for_like": c1["full"]["both_domestic_excluded"],
        "note": "the pre-registered wording compares Changhua without its domestic class to "
                "Yunlin with all classes; if domestic is non-pump load in both counties, the "
                "like-for-like ratio (both excluded) is the relevant one"}
    c1["r3_preinstall_coding_sensitivity"] = {
        how: {"pumps_kept": pc[how]["pumps_kept"],
              "a_star_2016_2022": pc[how]["preorigin_2016_2022"]["a_star"],
              "a_star_2016_01_2025_06": pc[how]["full_2016_01_2025_06"]["a_star"],
              "a_star_2012_2022": pc[how]["long_2012_2022"]["a_star"],
              "phi_star_2012_2022": pc[how]["long_2012_2022"]["phi_star"],
              "in_grid_GWh_per_yr_2012_2022": pc[how]["in_grid_GWh_per_yr_2012_2022"]}
        for how in pc}
    return ev


def diagnostics_v2(sens: dict) -> dict:
    d = json.loads((REPO / CC1 / "diagnostics_post_hoc.json").read_text(encoding="utf-8"))
    epm = sens["energy_per_nonzero_meter_jul_jun"]
    d["r3"] = {
        "status": "post hoc (repair round); no gate, threshold or selection depends on it",
        "c9_rows_only_in_new_sheet": sens["c9_rows_only_in_new_sheet"],
        "energy_per_nonzero_meter_jul_jun": epm,
        "t10_explanation_as_hypotheses": {
            "H1_ratio_drift": "TPC 農林漁牧 grew faster than the cleaned census after about "
                              "2021 (cleaned/TPC Yunlin 0.50-0.53 in 2017/18-2020/21, 0.443 in "
                              "2024/25, 0.425 in 2025/26), so a flat r(moy) fitted on "
                              "2016-01..2025-06 over-predicts the census",
            "H2_census_incomplete_in_newest_months": "meters with a non-zero reading fell "
                                                     "year on year by a median 14 % (Yunlin) "
                                                     "and 10 % (Changhua) in 2026 and 5 % / 4 % "
                                                     "in 2025, against about 1.00 before; if "
                                                     "readings are missing rather than pumps "
                                                     "idle, the census under-states the "
                                                     "newest months and part of the +15.8 % "
                                                     "is census incompleteness",
            "can_they_be_separated": "not with these data: stable energy with fewer non-zero "
                                     "meters fits both 'fewer, busier meters' and 'late or "
                                     "missing readings'. The round-2 sentence 'the census "
                                     "energy in the new months is complete and stable' is "
                                     "withdrawn; stable energy does not establish "
                                     "completeness."}}
    return d


def rows_r3(ev: dict, sens: dict, pre_ev: dict, cc_pv: str, pre_pv: str) -> list[dict]:
    rows = []

    def add(num, src, role, pv, note=""):
        rows.append({"number_id": f"P1.r3.{num}", "source": src, "role": role,
                     "product_version": pv, "note": note})
    add("T10.after_the_fact", FULL1, "input", cc_pv,
        f"sum err {ev['T10_ext_yunlin']['sum_err']:+.4f}; relabelled after-the-fact")
    add("T10.after_the_fact", CEN, "validator", cc_pv)
    add("T10.after_the_fact", "results/dataprep/reports/P1_evidence/r3/looks_r3.json",
        "input", cc_pv, "look-log corrections and the A1 verdict")
    add("T11.after_the_fact", FULL1, "input", cc_pv,
        f"sum err {ev['T11_ext_changhua']['scaled_apre']['sum_err']:+.4f}; relabelled "
        f"after-the-fact")
    add("T11.after_the_fact", CEN, "validator", cc_pv)
    add("T11.after_the_fact", "results/dataprep/reports/P1_evidence/r3/looks_r3.json",
        "input", cc_pv)
    ds = ev["C1_a_vs_census"]["r3_domestic_sensitivity"]
    for var in ("preorigin", "full"):
        add(f"C1.domestic_both_excluded_{var}", CEN, "input", cc_pv,
            f"{ds[f'{var}_both_excluded_like_for_like']:.4f}")
    pc = sens["changhua_preinstall_coding"]["codings"]
    for how in ("nan_to_zero_all", "nan_to_zero_leading"):
        for wn in ("preorigin_2016_2022", "full_2016_01_2025_06", "long_2012_2022"):
            add(f"sens.{how}.a_star_{wn}", CEN, "input", cc_pv,
                f"{pc[how][wn]['a_star']:.4f} (baseline {pc['baseline'][wn]['a_star']:.4f})")
            add(f"sens.{how}.a_star_{wn}", PRE1, "input", cc_pv, "a_v1 compared")
        add(f"sens.{how}.phi_star_2012_2022", CEN, "input", cc_pv,
            f"{pc[how]['long_2012_2022']['phi_star']:.4f}")
        add(f"sens.{how}.phi_star_2012_2022", "results/twin_forward/datum_gate_innov.npz",
            "input", cc_pv)
        add(f"sens.{how}.pumps_kept", CEN, "input", cc_pv, str(pc[how]["pumps_kept"]))
    c9 = sens["c9_rows_only_in_new_sheet"]
    add("c9.single_pump", CEN, "input", cc_pv,
        f"{c9['distinct_pumps']} pump(s); {c9['these_pumps_kwh_rows_in_amp_any_month']} AMP rows")
    add("c9.single_pump", AMP_KWH, "input", cc_pv)
    add("c9.single_pump", AMP_PUMPS, "input", cc_pv)
    add("diag.energy_per_nonzero_meter", CEN, "input", cc_pv, "post hoc")
    for c in ("CH", "YL"):
        cn = "彰化縣" if c == "CH" else "雲林縣"
        add(f"T3_BV_LOYO_preorigin_{c}", f"{TPC}#{cn}農林漁牧", "validator", pre_pv,
            f"rms {pre_ev['T3_BV_LOYO_preorigin'][c]['rms_pct']:.2f} % (2016-2022, reported "
            f"only)")
        for n in NAT:
            add(f"T3_BV_LOYO_preorigin_{c}", n, "input", pre_pv)
    return rows


def preorigin_v2_rows(pre_pv: str) -> list[dict]:
    """The v1 pre-origin numbers are carried unchanged into v2: same rows, v2 product."""
    old = [r for r in ledger.read_all() if r["wp"] == "P1"
           and r["product_version"] == "county_electricity/preorigin/v1"]
    return [{k: (pre_pv if k == "product_version" else r[k]) for k in ledger.COLUMNS
             if k != "wp"} | {"note": (r["note"] + " | carried into v2 unchanged").strip(" |")}
            for r in old]


def probe(path: str, use: str, **kw) -> dict:
    try:
        p = guard.open_product(REPO / path, use, **kw)
        return {"opened": True, "notes": p.notes}
    except Exception as exc:          # noqa: BLE001 - the refusal class is the result
        return {"opened": False, "error": type(exc).__name__, "message": str(exc)[:300]}


def main(evidence: str) -> int:
    evd = Path(evidence)
    evd.mkdir(parents=True, exist_ok=True)
    sens = json.loads((BUILD3 / "census_sensitivity.json").read_text(encoding="utf-8"))
    looks = json.loads((evd / "looks_r3.json").read_text(encoding="utf-8"))
    pre_ev = json.loads((evd / "preorigin_v2_build.json").read_text(encoding="utf-8"))
    if max(sens["baseline_reproduces_census_check_v1"].values()) > 1e-9:
        raise SystemExit("sensitivity baseline does not reproduce census_check/v1")
    for tid, v in looks["assert_can_score_log_false"].items():
        if v["can_score_as_preregistered"]:
            raise SystemExit(f"{tid}: the look-log correction did not take effect")
    m1 = json.loads((REPO / CC1 / "MANIFEST.json").read_text(encoding="utf-8"))
    d = manifest.new_version_dir(PRODUCT)
    pv = f"{PRODUCT}/{d.name}"
    t = pd.read_parquet(REPO / CC1 / "census_truth_monthly.parquet")
    t2 = t.assign(product_version=pv)
    pd.testing.assert_frame_equal(t.drop(columns="product_version"),
                                  t2.drop(columns="product_version"))
    t2.to_parquet(d / "census_truth_monthly.parquet", index=False)
    for f in COPY:
        shutil.copy2(REPO / CC1 / f, d / f)
    ev = evaluation_v2(sens, looks)
    (d / "evaluation.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    (d / "diagnostics_post_hoc.json").write_text(
        json.dumps(diagnostics_v2(sens), ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(BUILD3 / "census_sensitivity.json", d / "census_sensitivity.json")
    gate_ok = bool(m1["gate"]["passed"])
    (d / "PRODUCT_CARD.md").write_text(CARD.format(pv=d.name, gate=gate_ok), encoding="utf-8")
    params = {k: v for k, v in m1["parameters"].items()
              if k != "library_hashes_not_listed_as_code"}
    params["r3_reissue"] = {
        "reissue_of": f"{PRODUCT}/v1", "v1_manifest_sha256": sha256_file(REPO / CC1
                                                                         / "MANIFEST.json"),
        "changes": ["T10/T11 relabelled after-the-fact checks (look-log seeds corrected)",
                    "evaluation adds the scoring-run history, the like-for-like domestic "
                    "ratio and the pre-install coding sensitivity",
                    "diagnostics add the C9 single-pump finding, energy per non-zero meter "
                    "and the T10 hypotheses; the 'complete and stable' wording is withdrawn",
                    "guard.py, uses.py, flags.py and harness/prereg.py listed as code",
                    "census_truth_monthly.parquet: product_version rewritten (values "
                    "unchanged); other truth tables byte-identical"],
        "looks_evidence": "results/dataprep/reports/P1_evidence/r3/looks_r3.json"}
    inputs = [i["path"] for i in m1["inputs"]] + [
        f"{CC1}/MANIFEST.json", f"{CC1}/census_truth_monthly.parquet", f"{CC1}/evaluation.json",
        f"{CC1}/diagnostics_post_hoc.json"] + [f"{CC1}/{f}" for f in COPY] + [
        AMP_PUMPS, "results/dataprep/reports/P1_evidence/r3/looks_r3.json"]
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L2", inputs=inputs, code=CODE, parameters=params,
        fit_window_end=m1["fit_window_end"], validation_report="evaluation.json",
        gate=dict(m1["gate"], c9_cause="one pump with 222 pump-months (8.3 MWh) in the new "
                                       "Yunlin sheet has no kWh row in AMP_V2 (repair round)"),
        allowed_uses=["after_the_fact_analysis"],
        forbidden_uses=sorted(u for u in USES if u != "after_the_fact_analysis"),
        personal_data=m1["personal_data"])
    p = manifest.write_manifest(d, m)
    print("wrote", rel_to_repo(p))
    fix = ledger_fix(evd)
    pre_pv = pre_ev["product_dir"].split("results/dataprep/")[1]
    rows = rows_r3(ev, sens, pre_ev, pv, pre_pv) + preorigin_v2_rows(pre_pv)
    lp = ledger.append_rows("P1", rows)
    allrows = ledger.read_all()
    port = t3_port_check()
    probes = {
        f"{pre_pv}": {u: probe(f"results/dataprep/{pre_pv}", u)
                      for u in ("prior", "validator", "design_time_evidence",
                                "after_the_fact_analysis", "forcing_recorded_arm")},
        f"{pv}": {u: probe(f"results/dataprep/{pv}", u)
                  for u in ("after_the_fact_analysis", "validator", "design_time_evidence")},
        "ledger": {"rows_all_fragments": len(allrows), "conflicts": ledger.conflicts(allrows),
                   "p1_rows": sum(1 for r in allrows if r["wp"] == "P1"),
                   "r3_rows_added": len(rows), "fix": fix}}
    out = {"census_check": {"dir": rel_to_repo(d), "manifest": rel_to_repo(p)},
           "t3_port_check": port, "probes": probes, "ledger_fragment": rel_to_repo(lp)}
    (evd / "r3_build.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                  default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
