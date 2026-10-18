"""Pre-register WP H1b's acceptance (``H1b.ground_height.v1``) with the A1 tool (prereg3).

Run once, before the product build (``h1b_build.py`` without ``--dry``):

    cpu_slot.sh python -I h1b_prereg.py

It first appends a ``history_disclosure`` look for what this session saw before the
registration (W3 table summaries, five station rows, structural counts, the synthetic dry
run), then writes the pre-registration. The fresh unit is the set of stations whose agency
GroundHeight is a real value, that have a W3 DTM value, were established by 2022-12, and
whose residual was not displayed before the registration. Its members are station-metadata
records, written ``station_meta:<sid>``: no head, leveling or model-cell data is read by
H1b, so these units do not overlap the head-data units of the validation registry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(1, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common.paths import PREREG_DIR, TWIN_GRID_NPZ  # noqa: E402
from harness import prereg3  # noqa: E402

from h1_heads_qc import h1b_lib as lib  # noqa: E402

DATA_SCOPE = "station_metadata_GroundHeight+w3_dtm_well_points"
NA = prereg3.not_applicable


def structural() -> dict:
    """Station sets from the H1a v3 table and W3's finite well points (no residual)."""
    from w3_dtm import validation as w3val
    inp = lib.open_inputs()
    meta, w3 = inp["meta"], inp["w3"]
    st = lib.raw_status(meta["GroundHeight"])
    fin = set(w3.loc[np.isfinite(w3["ground_elev_dtm_ref_m"]), "sid"])
    est = pd.to_numeric(meta["EstablishDate"], errors="coerce") <= lib.EST_CUTOFF
    cal = (st == "valid") & meta["sid"].isin(fin).to_numpy() & est.to_numpy() & \
        ~meta["sid"].isin(lib.SEEN_BEFORE_REGISTRATION).to_numpy()
    g = lib.model_grid()
    blocks = w3val.block_ids(meta.loc[cal, "x_twd97"], meta.loc[cal, "y_twd97"], g["x0"],
                             g["y0"])
    return {"all": sorted(meta["sid"]), "cal": sorted(meta.loc[cal, "sid"]),
            "n_blocks": int(len(set(blocks.tolist())))}


def history_logged() -> bool:
    for lk in prereg3.read_chained(PREREG_DIR / "looks.jsonl"):
        if lib.HISTORY_UNIT in (lk.get("units") or []):
            return True
    return False


def main() -> int:
    s = structural()
    print(f"stations {len(s['all'])}; calibration (fresh) {len(s['cal'])} in "
          f"{s['n_blocks']} blocks", flush=True)
    if not history_logged():
        lk = prereg3.record_look(
            [lib.HISTORY_UNIT], data_scope=DATA_SCOPE, wp=lib.WP, purpose="history_disclosure",
            members={lib.HISTORY_UNIT: {
                "description": "station-metadata records of the 344 stations of "
                               "heads/L1_clean/v3 and their W3 DTM well points, as seen by "
                               "the H1b session before its pre-registration",
                "ids": [lib.NS + x for x in s["all"]]}},
            note="H1b session 2026-10-13, before H1b.ground_height.v1: (1) describe() of W3 "
                 "well_ground_elevation.parquet over all 344 stations (marginal statistics of "
                 "ground_height_meta, well_elevation_meta, DTM point, sd_m; no residual) and "
                 "its first 5 rows: 09160132 (GroundHeight NaN), 070213M3 (GroundHeight 6.84 "
                 "vs DTM 7.003), 070215T1 (3.09 vs 1.855), 07050111 and 07050121 (0.0 vs "
                 "8.854); the two residuals shown exclude 070213M3 and 070215T1 from the "
                 "fresh unit; (2) structural counts (status x model well x DTM, "
                 "EstablishDate, ModificationDate by status, coordinate agreement H1a v3 = "
                 "W3: max 0.0 m); (3) a dry run of h1b_build.py with synthetic agency values "
                 "at the 182 valid stations (seed 20261013): its log (C1-C4, C2 max |diff| 0, "
                 "T1 machinery on synthetic values) and three flag-5 rows (WellElevation vs "
                 "DTM at 09160132, 07050111, 07050121) were read; its other reported fields "
                 "(real WellElevation-minus-DTM summaries) were written to the scratchpad "
                 "and not read. W3's report (read first) states agency GroundHeight minus "
                 "DTM point +0.415 m (n 182) and WellElevation +0.686 m at the sheet epoch.")
        print(f"history look logged at {lk.get('time') if isinstance(lk, dict) else lk}",
              flush=True)
    code = [HERE.parent / "h1b_build.py", HERE.parent / "h1b_lib.py",
            HERE.parents[1] / "w3_dtm/build.py", HERE.parents[1] / "w3_dtm/leveling.py",
            HERE.parents[1] / "w3_dtm/mosaic.py", HERE.parents[1] / "w3_dtm/validation.py",
            HERE.parents[1] / "w3_dtm/dtm_io.py", HERE.parents[1] / "harness/coverage3.py",
            HERE.parents[1] / "harness/coverage2.py", HERE.parents[1] / "harness/coverage.py"]
    data = [lib.H1A_DIR / lib.H1A_TABLE, lib.H1A_DIR / "station_meta_clean.parquet",
            lib.H1A_DIR / "MANIFEST.json", lib.W3_DIR / lib.W3_MOSAIC, lib.W3_DIR / lib.W3_WELLS,
            lib.W3_DIR / "MANIFEST.json", lib.LEVELING, TWIN_GRID_NPZ]
    t1 = {
        "test_id": lib.TEST_ID, "data_scope": DATA_SCOPE, "fresh_units": [lib.FRESH_UNIT],
        "external_units": {lib.FRESH_UNIT: {
            "description": "stations of heads/L1_clean/v3 whose agency GroundHeight is a real "
                           "value (not 0.0, not NaN), that have a W3 DTM well point, were "
                           "established by 2022-12-31 and whose residual was not displayed "
                           "before this registration; members are station-metadata records "
                           "(agency GroundHeight and W3 DTM value), not head records",
            "ids": [lib.NS + x for x in s["cal"]]}},
        "gating": True,
        "metric": "leave-one-block-out coverage over 10 km squares anchored at the twin grid "
                  "origin (w3_dtm.validation.block_ids): for each block, tau is the "
                  "split-conformal 90 % quantile (rank ceil((n+1) 0.9)) of tau_i = sqrt(max(0, "
                  "(|r_i|/1.6449)^2 - sd_w3_i^2)) over the other blocks' stations, r = agency "
                  "GroundHeight - DTM well point (2017.5, recomputed from ground_elevation/v3 "
                  "at the H1a v3 coordinates), sd_w3 = W3's well-point sd; a station is a hit "
                  "when |r| <= 1.6449 sqrt(sd_w3^2 + tau^2); verdict harness.coverage3."
                  "coverage_gate(hits, blocks)",
        "threshold": "coverage3 'pass' (rule gate-3: cluster-adjusted Wilson interval "
                     "contains 0.90; the 0.85-0.95 point window applies when the lower bound "
                     "of n_eff >= 30) AND coverage3 'calibrated' (k_clusters >= 20)",
        "n_stations_expected": len(s["cal"]), "n_blocks_expected": s["n_blocks"]}
    doc = {
        "id": lib.PREREG_ID, "wp": lib.WP, "kind": "test",
        "title": "WP H1b: GroundHeight 0.0/NaN replaced by the W3 DTM well point - calibration "
                 "of the replacement sd and product conformance",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H1 (H1b bullet), WP W3 "
                        "(well-point output), 2.4-2.5, 4.4 (coverage rule), 4.5, 4.7",
        "tests": [t1],
        "zones": NA("station metadata; no zones"),
        "priors": NA("no priors"),
        "burn_in": NA("no model run"),
        "folds": {"rule": "leave one 10 km block out (w3_dtm.validation.block_ids with the twin "
                          "grid origin)", "n_blocks_expected": s["n_blocks"]},
        "fresh_units": [lib.FRESH_UNIT],
        "thresholds": {
            "T1": t1["threshold"],
            "C1_completeness": "every station whose agency GroundHeight is 0.0 or NaN is "
                               "flag 5 with a finite sd > 0, or flag 8 with a not_replaced "
                               "reason; all model wells with GroundHeight 0.0 are flag 5",
            "C2_reproduction": "DTM well points recomputed at the H1a v3 coordinates equal "
                               "W3's well_ground_elevation.parquet: same finite set, max |diff| "
                               "<= 1e-6 over ground_elev_dtm_ref_m, sd_m, local sd, epoch, "
                               "correction and rate; coordinates within 0.01 m",
            "C3_integrity": "common.flags.validate_long_table passes; flag-0 values equal the "
                            "agency GroundHeight exactly; no 0.0 code left as a value; flag-5 "
                            "sd > 0; no personal or contact columns; 344 unique sids",
            "C4_use_policy": "guard.check_use on the product's policy: design_time_evidence "
                             "and after_the_fact_analysis granted; validator and model_design "
                             "GateNotPassed while the gate is closed; every forbidden use "
                             "UseNotAllowed"},
        "gate_scores": [{"name": "T1_sd_coverage_lobo", "rule": "pass/fail", "weight": 0.2},
                        {"name": "C1_completeness", "rule": "pass/fail", "weight": 0.2},
                        {"name": "C2_reproduction", "rule": "pass/fail", "weight": 0.2},
                        {"name": "C3_integrity", "rule": "pass/fail", "weight": 0.2},
                        {"name": "C4_use_policy", "rule": "pass/fail", "weight": 0.2}],
        "gate_rule": "conjunctive: H1b's verdict passes only if T1 and C1-C4 all pass (the equal "
                     "weights only satisfy the schema). The product gate further requires the "
                     "upstream W3 gate (ground_elevation/v3, FAILED, no user decision recorded "
                     "at registration): it stays closed until a decision on W3 is recorded",
        "primary_model": {
            "value": "flag 0: agency GroundHeight as stored; flag 5 (agency 0.0 or NaN): W3 "
                     "DTM well point at 2017.5, bilinear on the published 20 m mosaic with "
                     "W3's leveling epoch correction, at the H1a v3 coordinates; flag 8: no "
                     "coordinates. No offset is applied to the DTM value",
            "sd": "flag 5: sqrt(sd_w3^2 + tau^2), tau = split-conformal 90 % over every "
                  "calibration station (the T1 rule without hold-out)",
            "inputs_opened": "heads/L1_clean/v3 for model_design (station_meta_preorigin "
                             "only) and after_the_fact_analysis (ModificationDate, reported); "
                             "ground_elevation/v3 for design_time_evidence (its gate failed)",
            "code": "dataprep_staging/h1_heads_qc/h1b_build.py and h1b_lib.py (hashed in "
                    "inputs)"},
        "inputs": [str(p) for p in code + data],
        "interpretation": (
            "H1b is the GroundHeight part of plan WP H1, after W3. W3's gate failed (median "
            "held-out bias of the cell mean -0.842 m vs |0.30| m) and no user decision on it "
            "is recorded, so this product inherits a closed gate whatever H1b's own verdict; "
            "H1b's verdict tells the user whether the replacement itself is sound. Prior "
            "knowledge (design-time, disclosed): W3's after-the-fact diagnostics put agency "
            "GroundHeight minus the DTM point at +0.415 m (n 182, sheet epoch) and "
            "WellElevation at +0.686 m; W3's sd_point covered 0.689 at benchmarks; the H1b "
            "session saw W3 table summaries, two station residuals (excluded from the fresh "
            "unit) and ran the pipeline on synthetic agency values (history look "
            f"{lib.HISTORY_UNIT}). No coverage of any interval against agency GroundHeight "
            "at stations had been computed. The calibration is against agency GroundHeight "
            "(construction-era surveys of unknown epoch), so a passing sd includes the "
            "agency-vs-DTM epoch and datum mismatch and is conservative for the 2017.5 "
            "ground; the DTM value itself is not shifted (plan: replace by the DTM). "
            "Exchangeability of replaced and calibration stations is an assumption, "
            "reported (WellElevation proxy) but not tested. No head data is read by H1b; "
            "U7: only stations established by 2022-12 enter the calibration and none of them "
            "carries a metadata modification dated 2023 or later (checked, reported)."),
        "conformance_checks": ["C1_completeness", "C2_reproduction", "C3_integrity",
                               "C4_use_policy"],
        "reported_not_gated": [
            "agency minus DTM residual summary at the calibration stations (median, IQR, "
            "MAD-sd; design-time knowledge: W3 +0.415 m)",
            "coverage of W3's published sd_m (tau = 0)",
            "LOBO coverage by county and model-well stratum",
            "tau per held-out block, sd_cal and sd_w3 at flag-5 rows",
            "stick-up proxy at replaced stations: WellElevation - median(WellElevation - "
            "GroundHeight) minus DTM, and the share inside the interval widened by the "
            "stick-up MAD-sd",
            "WellElevation - DTM at calibration vs replaced stations (Mann-Whitney p)",
            "stations with |agency - DTM| > 5 m (informational column, flag unchanged)",
            "new values at the 60 model wells with GroundHeight 0.0, well-bottom elevations",
            "calibration stations with a metadata modification dated >= 2023-01-01"],
    }
    path, sha = prereg3.write_prereg(doc)
    print(f"wrote {path} sha256 {sha}", flush=True)
    print(json.dumps({"n_cal": len(s["cal"]), "n_blocks": s["n_blocks"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
