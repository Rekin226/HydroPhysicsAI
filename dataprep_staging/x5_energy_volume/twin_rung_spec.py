"""Write the stage-2 twin rung specification of the X5 conversion fix (no twin run here).

Reads the step0_ab report (estimator (a) and the twin's conversion) and, if it exists, the
scored test report (for the corroboration label fixed in the test's ``stage2_rule``), and
writes results/dataprep/x5_energy_volume/twin_rung_spec.json.

Usage: python -I twin_rung_spec.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402

from x5_energy_volume import x5const as K  # noqa: E402

OUT = K.OUT / "twin_rung_spec.json"
I1B_GATES = [
    "k-fold out-of-fold R2 >= Q0 - 0.01",
    "Changhua and Yunlin county_corr >= Q0 - 0.02 (score_g2 definitions)",
    "Changhua and Yunlin county_mean_rmse <= Q0 x 1.02",
    "leveling out-of-fold R2 >= Q0 - 0.02 (column refitted on the rung's own heads)",
    "temporal screen (--holdout-months 36 --fit-only): fair RMSE ratio <= Q0 x 1.01 and "
    "fair verdict PASS",
    "forward plausibility: county-mean VEP subsidence rate Dec 2022 -> Dec 2025 / 3 >= 0 "
    "cm/yr in Changhua and in the rest of the fan (county map v2)",
]


def main() -> int:
    s = json.loads((K.REPORT_DIR["step0_ab"] / "report.json").read_text())["counties"]
    tr = K.REPORT_DIR["test"] / "report.json"
    test = json.loads(tr.read_text()) if tr.exists() else None
    verdict = test["primary"]["verdict"] if test else "not scored yet"
    label = ("conversion independently corroborated within 1.5x (X5.energy_volume.c_vs_a.v1 "
             "PASS)" if verdict == "PASS" else
             "conversion set by WRA's electricity-based ratio only (test verdict: "
             f"{verdict})")
    ct = s["changhua"]["step0"]["c_twin_m3_per_kwh"]
    a = {cn: s[cn]["a_m3_per_kwh"]["kept"]["janmay_2011_22"] for cn in K.COUNTIES}
    m = {cn: a[cn] / ct for cn in K.COUNTIES}
    hx = {cn: K.ETA_TWIN * K.M3_PER_KWH_PER_M / a[cn] - K.LIFT_MEDIAN for cn in K.COUNTIES}
    caveat = None
    if test:
        pr = test["primary"]
        dt = test.get("designtime_confined_amp_2012_2019", {})
        caveat = {
            "test_median_ratio": pr["median_ratio"], "test_n": pr["n"],
            "test_ci90": pr["ci90_median_ratio_well_bootstrap"],
            "county_median_ratio": {cn: pr[cn]["median_ratio"] for cn in K.COUNTIES
                                    if cn in pr},
            "designtime_2012_2019_median_ratio": dt.get("median_ratio"),
            "reading": "the PASS is weak: its 90 % interval also covers the twin's current "
                       "conversion (0.45-0.47 of a), per-well ratios span four orders of "
                       "magnitude and are stable across windows (a fixed per-well model "
                       "error), and the design-time 2012-2019 replication gives 0.57; the "
                       "multiplier rests on WRA's ratio, with the head route agreeing on the "
                       "median only (results/dataprep/reports/X5.energy_volume.c_vs_a.v1/"
                       "reading.md)"}
    spec = {
        "caveat": caveat,
        "name": "X5.R1 conversion fix: irrigation-class kWh x m(county), applied after the "
                "census cleaning",
        "label": label,
        "base_model": {"id": "Q0", "run": "results/twin_runs/stage3_datum_gate_q0",
                       "theta": rel_to_repo(K.Q0_THETA),
                       "key_flags": "fix_eta 0.5, fix_head_extra 40, meter_filter dedupe-cap, "
                                    "cap_duty 1.0, pump_split, return_flow, zonal coast-apex "
                                    "205,182,208, census model_census/v1 (Changhua included)"},
        "change": {
            "inputs": "--pump-census results/dataprep/x5_energy_volume/stage2_inputs/county/"
                      "census.parquet --pump-kwh .../county/kwh.parquet --meter-filter none "
                      "(built by dataprep_staging/x5_energy_volume/build_stage2_inputs.py "
                      "county: Q0's own clean_census kept set, frozen, one meter key per pump)",
            "multiplier_irrigation": m,
            "reference_a_m3_per_kwh_kept_janmay_2011_22": a,
            "twin_conversion_m3_per_kwh_at_median_lift": ct,
            "other_classes": "unchanged (WRA's own per-kWh volumes for aquaculture, "
                             "livestock and domestic are at or below the twin's 3.94)",
            "equivalent_head_extra_m_at_eta_0.5_median_lift": hx,
            "why_not_fix_head_extra": "one global h_extra converts every class; it would also "
                                      "double aquaculture/livestock/domestic volumes, and the "
                                      "twin has no fixed per-county ratio flag (--eta-county "
                                      "learns one); the data-side multiplier needs no code "
                                      "change",
            "everything_else": "as Q0 (same flags, seed 2027 partition, site tol 200 m)"},
        "precheck_cpu": "arm m1 (multiplier 1, --meter-filter none) must reproduce Q0's "
                        "per-cell monthly kWh tensor to 1e-9 relative before any GPU run; "
                        "if not, stop (an input-construction defect, not a result)",
        "declared_alternative": {
            "name": "X5.R1t township-varying factor shrunk to the county mean",
            "construction": "factor_t = exp(0.5 x log(r_t / r_county)), r_t = Lin (2012) "
                            "Table 2-2 township irrigation volume / census kept irrigation "
                            "kWh 2007-08 of the town, capped to [1/3, 3], renormalised so "
                            "the county's kWh-weighted mean factor is 1; multiplier = "
                            "m(county) x factor_t (build_stage2_inputs.py township; Lin 2012 "
                            "from X3 evidence v1 township_pumping_by_use.csv)",
            "rule": "reported beside X5.R1, never substituted; run only if X5.R1 runs"},
        "gates_against_Q0": I1B_GATES,
        "gate_source": "as I1.B.v1 (results/dataprep/preregistration/I1.B.v1.json); scorer "
                       "dataprep_staging/i1/score_rung_b.py logic with the rung's suffixes",
        "column": "refit the coupled_leveling column on the rung's own head field (project "
                  "memory: the column pairs with its head field)",
        "promotion": "only if every gate passes: rerun Q0 and the arm on the never-scored "
                     "partition (calibrate_flow --seed 2029 --fold-site-tol-m 200, column "
                     "--seed 2029) with the same gates",
        "registration": "the lead registers the rung (kind declaration, design-time "
                        "partition seed 2027) before the GPU batch; this file is its draft",
        "reported_not_gating": [
            "fan and county irrigation volume (Mm3/yr) against WRA's 2011-22 Jan-May values",
            "calibrated T and S by zone against Q0 (A0: storage and volume trade off)",
            "Changhua and Yunlin amplitude ratio and interannual correlation",
            "P2f Yunlin ratio (score_p2f_rung.py)"],
        "expected_risk": "doubling irrigation volume is likely to raise the calibrated T "
                         "(and/or lower S) rather than the head amplitude, because the "
                         "heads constrain Q/T; a pass then says little about volume and the "
                         "calibrated T should be read against Liu et al.'s 58-6034 m2/day",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(spec, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
