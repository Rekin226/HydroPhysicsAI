"""Step 4: pre-registration ``H3.recovery.v1`` (A1 ``prereg3``), written before any H3 test
cell exists.

Tests:

* ``H3.d_wells``: per-well coverage of H2's frozen daily bridge (the H2 stratum that applies
  to a recovered well's calibration targets, U2) on family-d realizations r = 100..159, never
  generated before, at the 15 envelope candidates and the R-h well 09030212.
* ``H3.snapshot``: representativeness of the once-daily (00:00) readings that continue
  09030212's record in 2019-2022 (the R-h find), tested on its own 2012-2018 hourly-era
  months with sigma_snap estimated on the other wells of its raw layer code.

Writes ``H3_evidence/registered_wells.json`` (the test wells and the snapshot estimation
wells; hashed by the registration) and the registration.

Usage: python -I s4_prereg.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import pandas as pd  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
from harness import prereg3  # noqa: E402

RH_WELL = "09030212"
CODE = ["__init__.py", "config.py", "inventory.py", "snapshot.py", "s1_inventory.py",
        "s2_rh_probe.py", "s3_layer_review.py", "looks_h3.py", "looks_h3_dryrun.py",
        "s4_prereg.py", "s5_runs.py",
        "s6_score.py"]
H2_CODE = ["__init__.py", "config.py", "io_h2.py", "bridge.py", "runner.py", "methods.py",
           "kriging.py", "scoring.py"]
LIB = ["dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py",
       "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/prereg3.py",
       "dataprep_staging/common/guard.py", "dataprep_staging/common/paths.py",
       "hydrophysics/twin/grid.py", "hydrophysics/twin/heads.py"]
EV = "results/dataprep/reports/H3_evidence"
DATA = ["results/dataprep/heads/L1_clean/v3/MANIFEST.json",
        "results/dataprep/heads/L1_clean/v3/heads_monthly.parquet",
        "results/dataprep/heads/L1_clean/v3/heads_daily.parquet",
        "results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet",
        "results/dataprep/validation_units/v3/wells.csv",
        "results/dataprep/validation_units/v3/MANIFEST.json",
        "results/twin/basemap.npz", "results/twin_forward/datum_gate_innov.npz",
        "results/dataprep/reports/H2_evidence/design/design_params.json",
        "results/dataprep/reports/H2_evidence/run_tables/base_pre.npz",
        "AMP_V2/data/fan_stations.parquet",
        f"{EV}/inventory/inventory.csv", f"{EV}/inventory/counts.json",
        f"{EV}/rh_probe/rh_probe.json", f"{EV}/rh_probe/rh_probe_wells.csv",
        f"{EV}/layer_review/layer_review.json", f"{EV}/registered_wells.json"]
D_UNIT = f"H3:d_r{C.FRESH_D.start}-{C.FRESH_D.stop - 1}_wells16"
S_UNIT = f"H3:snapshot_{RH_WELL}_2012-2018"


def registered_wells() -> dict:
    inv = pd.read_csv(C.EVIDENCE / "inventory" / "inventory.csv",
                      dtype={"sid": str, "layer_raw": str})
    cand = sorted(inv[inv.candidate.astype(bool)].sid)
    lay = str(inv.loc[inv.sid == RH_WELL, "layer_raw"].iloc[0])
    has = inv[(inv.n_raw > 0) & inv.layer_raw.isin(C.LAYERS)
              & inv.sid.map(lambda s: (C.AMP_WELLS / f"{s}.parquet").exists())]
    by_layer = {lv: sorted(g.sid) for lv, g in has[has.sid != RH_WELL].groupby("layer_raw")}
    return {"candidates_envelope": cand, "rh_well": RH_WELL,
            "test_wells": sorted(set(cand) | {RH_WELL}),
            "snapshot": {"well": RH_WELL, "layer": lay,
                         "estimation_wells": by_layer.get(lay, []), "by_layer": by_layer}}


def doc(reg: dict) -> dict:
    wells = reg["test_wells"]
    est = reg["snapshot"]["estimation_wells"]
    amp = sorted({f"AMP_V2/data/wells/{s}.parquet" for s in est + [RH_WELL]})
    tests = [
        {"test_id": C.TEST_D, "data_scope": "heads:2012-01..2022-12", "fresh_units": [D_UNIT],
         "external_units": {D_UNIT: {
             "description": "H2 family d (one random daily gap of 5-60 days per eligible "
                            "well per realization, seed 1000*ord('d') + r; generator "
                            "h2_heads_fill.bridge.d_windows) realizations r = 100..159 on the "
                            "pre-origin L1 v3 matrix, never generated before; every eligible "
                            "well masked as in H2; only the 16 H3 test wells' cells kept and "
                            "scored (truth: the C20 months the gaps make non-C20)",
             "ids": wells, "window": "2012-01..2022-12"}},
         "metric": "per test well: rule gate-3 (harness.coverage3.coverage_gate) on the "
                   "coverage of the frozen bridge's 90 % interval (pred +- 1.6449 sd, sd of "
                   "the bridged-share class in H2 design_params.json), clean truth "
                   "(io_h2.clean_truth), bridgeable cells only (scoring.product_daily), "
                   "cluster = the gap window (well, realization); pooled over the test wells "
                   "(cluster = the well): coverage (reported) and the RMSE of the bridge "
                   "against the twin's backfill, climatology + AR(1), space-time kriging, "
                   "and same-layer IDW on the IDW cells (run tables of runner.run_daily)",
         "threshold": "a well passes if its gate passes and is calibrated (k >= 20 windows); "
                      "the pooled RMSE rule passes if the bridge's pooled clean RMSE is "
                      "strictly lower than every defined baseline's"},
        {"test_id": C.TEST_SNAP, "data_scope": "heads:2012-01..2018-12", "fresh_units": [S_UNIT],
         "external_units": {S_UNIT: {
             "description": f"well {RH_WELL}'s own L1 C20 months 2012-01..2018-12 with >= 20 "
                            "raw 00:00 readings on L1 flag-0/1 valid days: the monthly mean "
                            "of those readings against the C20 value; never computed before",
             "ids": [RH_WELL], "window": "2012-01..2018-12"}},
         "metric": "sigma_snap = pooled RMSE of (mean of a month's raw 00:00 readings on L1 "
                   "valid days - L1 C20 value) over the other wells of raw layer code "
                   f"{reg['snapshot']['layer']} with >= 24 such months in 2012-01..2018-12 "
                   "(snapshot.sigma_snap); then rule gate-3 on hits |error| <= 1.6449 "
                   f"sigma_snap over {RH_WELL}'s own months, cluster = calendar quarter",
         "threshold": "passes if the gate passes and is calibrated (k >= 20 quarters)"}]
    gate_rule = (
        "Admission (packaging rule for heads/recovered_wells/v1, declared before any test "
        "cell exists). (1) Envelope, deterministic and pre-origin (inventory.py; observed "
        "months only, fills never count): not a model well; in an active cell with the L1 "
        "repaired coordinates; raw layer code 1-4; L1 v3 C20 coverage >= 0.80 of 2012-01.."
        "2022-12; >= 36 observed months; internal runs <= 24 months; leading run <= 66 "
        "months; record reaching 2022-12 (months after a record end would be flag 6, never "
        "calibration). 15 wells meet it (counts.json). (2) H3.d_wells: an envelope well whose "
        "per-well gate passes (calibrated) keeps its daily-bridged months (flag 2 rows of "
        "heads/L2_filled_preorigin/v4) as calibration targets, provided the pooled RMSE rule "
        "passes; otherwise it enters with flags 0-1 only. (3) H3.snapshot: if it passes, "
        f"{RH_WELL}'s 2019-01..2022-12 months with >= 20 once-daily readings that survive "
        "the persistence screen (snapshot.screen_daily) enter as flag 0, method "
        "'observed_daily_snapshot', sigma_obs = max(0.10, sigma_snap), and the well is "
        "admitted if its coverage counting those months meets the envelope of (1) (its "
        "bridged months then follow (2)); if it fails, the well is not admitted. (4) The "
        "product carries the admitted wells' pre-origin targets with allowed uses "
        "calibration_target (only in the pre-declared I1 ladder rung R5), datum_fit, "
        "validator and after_the_fact_analysis; never a target of the 2023-26 test (the "
        "registered 158-well set stays fixed). The six no-code wells (layer proposals flag "
        "5, unverified) and the wells whose records stop in 2018-12 without a found record "
        "are listed, not admitted.")
    return {
        "id": C.PREREG_ID, "wp": C.WP, "kind": "test",
        "title": "WP H3: recovered wells judged on observed months; per-well daily-bridge "
                 "coverage; once-daily continuation of an R-h well",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H3, 2.4-2.5, 4.4-4.5; "
                        "WP I1 rung R5",
        "tests": tests,
        "zones": prereg3.not_applicable("point records"),
        "priors": prereg3.not_applicable("no priors"),
        "burn_in": prereg3.not_applicable("no model run"),
        "folds": prereg3.not_applicable("artificial-gap realizations (d) and a single-well "
                                        "representativeness check; clusters in each metric"),
        "fresh_units": [D_UNIT, S_UNIT],
        "design_time_units": [
            "H3:inventory_2012-2022", "H3:rh_probe", "H3:switch_09030212",
            "H3:layer_review_nantou6",
            "H2 family-d units containing every test well's 2012-22 record (H2:design_d_r0-4, "
            "H2:fresh_d_r10-19, H2:fresh3_d_r60-69: bridge errors pooled over 254-258 "
            "wells, never broken out by H3 well) and H2's router families",
            "pilot head_gapfill_pilot recovery_hist.csv (after-fill coverage counts)"],
        "thresholds": {C.TEST_D: "per well: gate-3 pass and calibrated; pooled RMSE strictly "
                                 "below every defined baseline",
                       C.TEST_SNAP: "gate-3 pass and calibrated"},
        "gate_scores": [{"name": C.TEST_D, "rule": "per-well admission of bridged months",
                         "weight": 0.5},
                        {"name": C.TEST_SNAP, "rule": "admission of the once-daily months",
                         "weight": 0.5}],
        "gate_rule": gate_rule,
        "primary_model": ("daily bridge of H2 (form 'shape', bridged-share sds of "
                          "design_params.json, unchanged since H2.fill_acceptance.v1); "
                          "once-daily monthly mean with sigma_snap"),
        "inputs": ([f"dataprep_staging/h3_recovered_wells/{f}" for f in CODE]
                   + [f"dataprep_staging/h2_heads_fill/{f}" for f in H2_CODE] + LIB + DATA
                   + amp),
        "reported_not_gated": [
            "per-well RMSE of the bridge, backfill and climatology + AR(1); pooled coverage "
            "(cluster = the well; 16 clusters, uncalibrated by the A1 envelope); all-truth "
            "coverage; bridged-share classes; both bridge forms",
            "sigma_snap by layer for all wells; 09030212's snapshot bias and RMSE",
            "after the scoring: the H2 registered family-d and family-S run tables restricted "
            "to the test wells (after-the-fact, logged first)"],
        "interpretation": (
            "Plan WP H3: recovered wells enter calibration only in the pre-declared I1 rung "
            "R5, judged on observed months, with no extrapolated record. U2 admits only "
            "flags 0-2 as calibration targets, and H2's complete fill (router, flags 3-4) "
            "failed its gate (H2 partial), so the H2 stratum that applies to a recovered "
            "well's targets is the daily bridge (d); the router strata are reported after "
            "the fact only. Per-well testing with ~30-40 windows per well puts each well "
            "inside the A1 envelope (k >= 20; 60 realizations were chosen after a dry run on "
            "the design realization r = 0 printed only how many test wells kept a scored "
            "window, 10 of 16); with 16 wells, about one calibrated well is "
            "expected to fail by chance, which only drops its bridged months. The R-h probe "
            "(s2_rh_probe.py) found no post-2018 pre-origin record for 18 of the 19 on-grid "
            "wells whose records stop in 2018-12; well 09030212 continues in the twin's own "
            "cache with one 00:00 reading per day in 2019-2022, which L1's 12-hour day rule "
            "discards. The builder looked at its December 2018 / January 2019 values before "
            "this registration (logged: H3:switch_09030212). A dry run of the scoring code "
            "(stubbed registration, logged: H3:dryrun_paths) computed the snapshot metric "
            "and the r = 0 cells into scratch files that were deleted unread; it printed "
            "only counts (09030212: 75 own months in 26 quarters, 52 estimation wells, 33 "
            "once-daily months passing the screen). "
            "Family d r >= 100 was never generated (H2 used 0-4, 10-19, 60-69). No "
            "observation dated 2023-01 or later enters any test, threshold or choice (U7)."),
    }


def main(argv):
    reg = registered_wells()
    p = C.EVIDENCE / "registered_wells.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old != reg:
            raise SystemExit(f"{p} exists with different content; the registration hashes it")
    else:
        p.write_text(json.dumps(reg, indent=1))
    d = doc(reg)
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if argv and argv[0] == "--dry-run":
        print("valid; tests", [t["test_id"] for t in d["tests"]], "inputs", len(d["inputs"]),
              "test wells", len(reg["test_wells"]), "estimation wells",
              len(reg["snapshot"]["estimation_wells"]))
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main(sys.argv[1:])
