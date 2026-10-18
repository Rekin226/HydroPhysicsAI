"""Pre-registration of the WP H2 acceptance tests (A1 tool ``harness.prereg3``).

Written after the design stage (design-time realizations r <= 4 and family D) and before any
fresh realization is generated. Registers six fill tests (families N, S, E, H, T, d on fresh
realizations) and the initial-condition decision test, with the frozen code, the design
parameters, the L1 inputs and the harness modules as hashed inputs.

Usage: python -I s3_prereg.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import io_h2  # noqa: E402

PREREG_ID = "H2.fill_acceptance.v1"
SCOPE = "heads:2012-01..2022-12"
CODE = ["config.py", "io_h2.py", "methods.py", "kriging.py", "bridge.py", "runner.py",
        "scoring.py", "s1_bridge_form.py", "s2_design.py", "s4_score.py", "ic_field.py",
        "s6_ic.py", "looks_h2.py", "port_check.py", "s3_prereg.py", "__init__.py"]
HARNESS = ["dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py",
           "dataprep_staging/harness/coverage.py", "dataprep_staging/harness/coverage2.py",
           "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/scorers.py",
           "dataprep_staging/common/guard.py", "dataprep_staging/common/paths.py",
           "hydrophysics/twin/grid.py"]
DATA = ["results/dataprep/heads/L1_clean/v3/MANIFEST.json",
        "results/dataprep/heads/L1_clean/v3/heads_monthly.parquet",
        "results/dataprep/heads/L1_clean/v3/heads_daily.parquet",
        "results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet",
        "results/dataprep/validation_units/v3/wells.csv",
        "results/dataprep/validation_units/v3/MANIFEST.json",
        "results/twin/basemap.npz", "results/twin_forward/datum_gate_innov.npz",
        "results/dataprep/reports/H2_evidence/design/design_params.json",
        "results/dataprep/reports/H2_evidence/design/bridge_form.json",
        "results/dataprep/reports/H2_evidence/port_check.json"]
STRATA = {"N": "N_net1, N_net2", "S": "S01, S02, S03, S06, S12, S24",
          "E": "E_lead66, E_trail48 (gated on the fresh well-gaps that no design "
               "realization r = 0..4 contains; all fresh E cells reported)",
          "H": "H_half12", "T": "T_S (single-well gaps of 1, 2, 3, "
          "6 and 12 months inside the last 45 pre-origin months), T_H06 (half the network "
          "for 6 months)", "d": "d (all bridged months pooled; bridged-share classes reported)"}
CLUSTERS = {"N": "the gap window (realization x gap)", "S": "the well", "E": "the well",
            "H": "the realization", "T": "T_S: the well; T_H06: the realization",
            "d": "the well"}
DESC = {"N": "whole-network outages of 1 and 2 months (3 each per realization)",
        "S": "single-well internal gaps of 1, 2, 3, 6, 12 and 24 months (12 each)",
        "E": "66-month leading gaps at 20 wells and 48-month trailing gaps at 20 others",
        "H": "half the network missing for 12 months",
        "T": "test-period replica inside 2019-04..2022-12",
        "d": "one random daily gap of 5-60 days per eligible well (H2's own family, seed "
             "1000*ord('d') + r); the months it makes non-C20 are bridged"}


def fresh_unit(fam):
    rs = C.FRESH_R[fam]
    return f"H2:fresh_{fam}_r{rs.start}-{rs.stop - 1}"


def doc(h, params):
    ids = sorted(h.sids)
    tests = []
    for fam in ("N", "S", "E", "H", "T", "d"):
        u = fresh_unit(fam)
        rs = C.FRESH_R[fam]
        tests.append({
            "test_id": f"H2.fresh.{fam}", "data_scope": SCOPE, "fresh_units": [u],
            "external_units": {u: {
                "description": f"family {fam} ({DESC[fam]}) realizations r = {rs.start}.."
                               f"{rs.stop - 1} on the pre-origin L1 v3 matrix; never run or "
                               f"scored before this registration (for family E only the "
                               f"well selection of r = 10..19 was computed beforehand, from "
                               f"the observation mask alone, to size the fresh-only E "
                               f"stratum); truth cells are C20 months of these wells",
                "ids": ids, "window": "2012-01..2022-12"}},
            "metric": f"per stratum ({STRATA[fam]}), clean truth: (a) coverage of the "
                      f"product's 90 % interval (pred +- 1.6449 sd) by rule gate-3 "
                      f"(harness.coverage3.coverage_gate, clusters = {CLUSTERS[fam]}); (b) "
                      f"pooled RMSE of the product below that of each baseline: the twin's "
                      f"backfill, climatology + AR(1), space-time kriging of standardized "
                      f"anomalies, and same-layer IDW of anomalies on the cells where IDW is "
                      f"defined (product on the same cells)",
            "threshold": "every stratum: coverage gate passes AND the product's RMSE is "
                         "strictly lower than every defined baseline's (a baseline that the "
                         "design made the method of a gap class present in the stratum: not "
                         "worse; none was, see primary_model)"})
    u = "H2:ic_lowo_2012-01"
    tests.append({
        "test_id": "H2.ic.lowo", "data_scope": "heads:2012-01", "fresh_units": [u],
        "external_units": {u: {
            "description": "leave-one-well-out on the January 2012 L1 heads of the 158 model "
                           "wells (any well with a flag 0/1 value); never scored before",
            "ids": sorted(io_h2.model_wells().sid.tolist()), "window": "2012-01..2012-01"}},
        "metric": "pooled leave-one-well-out RMSE of the twin's per-layer IDW, kriging with "
                  "an external drift (SRTM cell elevation) and a GP field (ic_field.py)",
        "threshold": "decision rule: the initial condition stays IDW unless KED or GP has a "
                     "lower pooled LOWO RMSE; if both, the lower; not a gate criterion"})
    return {
        "id": PREREG_ID, "wp": C.WP, "kind": "test",
        "title": "WP H2: head gap filling with calibrated uncertainty - acceptance on fresh "
                 "artificial-gap realizations and the initial-condition decision",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H2 (technique, validation, "
                        "acceptance), 2.4-2.5 (flags, U1-U7), 4.4 (families, coverage rule), "
                        "4.5",
        "tests": tests,
        "zones": prereg3.not_applicable("point records; no zones"),
        "priors": prereg3.not_applicable("no priors"),
        "burn_in": prereg3.not_applicable("no model run"),
        "folds": prereg3.not_applicable("artificial-gap realizations; clusters per family "
                                        "are given in each test's metric"),
        "fresh_units": [fresh_unit(f) for f in ("N", "S", "E", "H", "T", "d")]
        + ["H2:ic_lowo_2012-01"],
        "design_time_units": ["H2:design_N_r0-4", "H2:design_S_r0-4", "H2:design_E_r0-4",
                              "H2:design_H_r0-4", "H2:design_T_r0-4", "H2:design_d_r0-4",
                              "H2:D_febmar_2013-2022 (both design passes: 2026-10-13 and "
                              "2026-10-14)", "pilot head_gapfill_pilot N r0-2, "
                              "S r0-3, E r0-1, H r0-2, T r0-2 (post-origin window), D "
                              "2013-2025"],
        "thresholds": {"coverage": "rule gate-3: cluster-adjusted interval (upper90 ICC, "
                                   "Student-t on k-1 df, level 0.95) contains 0.90; point "
                                   "window 0.85-0.95 when n_eff_lower >= 30",
                       "rmse": "product < backfill, clim+AR(1), STK, IDW (where defined), "
                               "pooled over the stratum's clean-truth cells",
                       "ic": "lower pooled LOWO RMSE than IDW"},
        "gate_scores": [{"name": f"H2.fresh.{f}", "rule": "pass/fail", "weight": 1 / 6}
                        for f in ("N", "S", "E", "H", "T", "d")],
        "gate_rule": "conjunctive: the product gate passes only if all six fill tests pass "
                     "(every stratum); H2.ic.lowo is a registered decision, not a gate "
                     "criterion; equal weights only satisfy the schema. Packaging rule "
                     "declared now, before any fresh result: if the conjunctive gate fails "
                     "but H2.fresh.d passes, the daily-bridge component (flag-2 months, plus "
                     "the flag 0/1 observations) is released as a separate product version "
                     "whose gate is H2.fresh.d alone, for calibration_target only; the full "
                     "fill is then published with its gate failed (design-time evidence and "
                     "after-the-fact analysis only). No other partial release.",
        "primary_model": "the frozen H2 configuration: config.py, methods.py (pilot router "
                         "port, verified identical by port_check.py), kriging.py, bridge.py "
                         f"(form {params['bridge_form']!r}, decided on pre-origin design "
                         "realizations by s1_bridge_form.py), "
                         "runner.py, scoring.py and the design parameters design_params.json "
                         f"(factor classes {params['factor_by_class']}, methods "
                         f"{params['method_by_class']}, bridge sd {params['bridge_sd_by_class']}"
                         f"), all hashed in inputs",
        "inputs": [str((C.CODE_DIR / f).relative_to(C.REPO)) for f in CODE] + HARNESS + DATA,
        "conformance_checks": [
            {"id": "H2.D_febmar", "data_scope": "heads:2013-01..2022-12",
             "metric": "family D (the 2026 outage window removed in every year 2013-2022): "
                       "product (daily bridge) RMSE below backfill, clim+AR(1) and STK, and "
                       "coverage gate on clean truth",
             "units": ["H2:D_febmar_2013-2022"], "status": "design-time unit (seen in the "
             "design stage and by the pilot); a pass there is necessary, not independent"}],
        "reported_not_gated": [
            "all-truth tables (coverage and RMSE) of every test",
            "model-well subset (158 wells)",
            "within-fresh split-conformal multiplier per stratum (20 half splits by cluster)",
            "bridged-share classes of family d; the bridgeable share; the partial-mean RMSE",
            "family E on all fresh cells, and the share of fresh E well-gaps also present in "
            "the design realizations (E masks are deterministic per well)",
            "both bridge forms (shape, lin) on family d",
            "every method's RMSE (router, neighbour, softImpute, GP, climatology)",
            "after-the-fact checks of the full variant on 2023-26 truth (family D 2013-2025, "
            "test-period replica in 2023-01..2026-09): never a gate (U7)"],
        "interpretation": (
            "Plan WP H2 acceptance: 'coverage passes the cluster-adjusted rule of 4.4 in every "
            "stratum; RMSE below the backfill, climatology + AR(1), same-layer IDW and "
            "space-time kriging in every family where the baseline is defined; a family where "
            "a baseline wins falls back to that baseline; results reported on both all truth "
            "and clean truth'. The fall-backs were decided on the design-time realizations "
            "(none is chosen after this registration); the gate is on clean truth (months with "
            "no flag-9 or flag-3 day in the month or an adjacent observed month; the three "
            "pumping-cycle wells excluded), all truth is reported. Truth cells are L1 v3 C20 "
            "months of 2012-22 only; the base matrix includes the daily-bridged months as "
            "pseudo-observations, as in the product. The sd factors per gap class and the "
            "bridge sds per share class were set on the design realizations (r <= 4) and the "
            "fresh realizations (r >= 10) check them out of sample: that split is the plan's "
            "split-conformal check. The E masks are deterministic per well (first 66 or last 48 "
            "months), so fresh E realizations can repeat design well-gaps; the E gate is "
            "computed on the fresh well-gaps no design realization contains. The gap classes "
            "(the plan's 1-2 / 3-23 / >= 24 months refined to 1-2 / 3-5 / 6-11 / 12-23 / >= 24, "
            "network outages split into 1 and >= 2 months, a 'thin' class for months with "
            "fewer than 0.6 x the median number of observed wells) and the bridge form were "
            "fixed on design realizations only. With 14 strata, a calibrated product still "
            "fails some stratum by chance (each interval is a 95 % interval); thresholds are "
            "the plan's and are not changed after the results. No observation dated 2023-01 or "
            "later entered any design choice, threshold or factor (U7); the design used "
            "pre-origin L1 rows read through the loader guard with window_end 2022-12."),
    }


def main(argv):
    params = json.loads((C.EVIDENCE_DIR / "design" / "design_params.json").read_text())
    h = io_h2.load("pre", with_daily=False)
    d = doc(h, params)
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if argv and argv[0] == "--dry-run":
        print("valid; tests:", [t["test_id"] for t in d["tests"]])
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main(sys.argv[1:])
