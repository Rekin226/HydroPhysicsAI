"""WP S4b: write the pre-registration ``S4b.insar_acceptance.v1`` with ``harness.prereg3``
BEFORE any test statistic is computed on the pre-origin run P (and before any coherence,
GNSS comparison or amplitude at the test cells exists).

    python -I dataprep_staging/s4_insar/s4b_prereg.py [--dry-run]

Registers three tests on pre-origin data only (U7): G1, the plan's numeric acceptance (WRA
GNSS leave-one-station-out 14-month change RMS <= 10 mm); C1, the plan's Changhua summer
coherence coverage per zone (>= 30 % of a zone's cells with coherence >= 0.3, otherwise the
zone is reported as not covered); Y, the plan's Yunlin test of the seasonal amplitude as a
validator (beat the placebo and the compressible-thickness proxies under the P3 battery).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

CODE = ["s4b_config.py", "s4b_lib.py", "s4b_geom.py", "s4b_run_licsbas.sh", "s4b_coherence.py",
        "s4b_battery.py", "s4b_gnss_inventory.py", "s4b_evaluate.py", "s4b_prereg.py",
        "make_licsbas_workdir.py", "licsbas_env.sh"]
PILOT_STATIONS = ["TKJS", "KTES", "GFES", "HLES", "STES", "YCES", "LYES", "JMES", "SNES", "JJES"]


def units_cells(ids) -> list[int]:
    u = {x["id"]: x for x in json.loads(C.UNITS_V3.read_text())["units"]}
    return sorted(set().union(*[set(u[i]["cells"]) for i in ids]))


def doc() -> dict:
    na = prereg3.not_applicable
    inv = json.loads((C.RUN_DIR / "gnss_inventory.json").read_text())
    cand = sorted(r["station"] for r in inv if r.get("on_fan") and r.get("gnss_ok")
                  and not r.get("catalogue_steps_in_span"))
    bat = json.loads((C.RUN_DIR / "battery.json").read_text())
    yl_cells = sorted(set(units_cells(C.YUNLIN_UNITS)) | set(bat["train_cells"]))
    zone_cells = {z: units_cells([z]) for z in C.CH_ZONES}
    ext_g = {
        "s4b.wra_gnss.loso_preorigin": {
            "description": "WRA GNSS stations (vendor cache, L0) on active twin cells with >= "
                           f"{C.GNSS_MIN_DAYS_PER_WINDOW} daily heights in both +-30 d windows "
                           "and no catalogued step in the change span; their run-P InSAR "
                           "pixels, 2021-08..2022-12",
            "ids": [f"wra_gnss:{s}" for s in cand], "window": "2021-08..2022-12"},
        "s4b.pilot.insar_gnss_2021_2023": {
            "description": "design-time: the subsidence scout's mini-SBAS (fan-south subset, "
                           "2021-10..2023-03, no atmospheric correction) compared with 10 WRA "
                           "stations (insar_gnss_loso.py: 9.6 mm time-series RMS, 8.2 mm on "
                           "the 2021-12-05 -> 2023-02-20 change)",
            "ids": [f"wra_gnss:{s}" for s in PILOT_STATIONS], "window": "2021-10..2023-03"}}
    ext_c = {f"s4b.coh.{z.split('.')[-1]}": {
        "description": f"native-resolution LiCSAR coherence of the primary frame at the "
                       f"active cells of Changhua zone {z} (registry design-time unit), "
                       f"pre-origin summer pairs", "cells": zone_cells[z],
        "window": "2020-06..2022-09"} for z in C.CH_ZONES}
    ext_y = {"s4b.seasamp.yunlin": {
        "description": "annual-harmonic vertical amplitude (run P, 2020-01..2022-12) at the "
                       "Yunlin cells (registry terciles plus the P3 battery cells)",
        "cells": yl_cells, "window": "2020-01..2022-12"}}
    tests = [
        {"test_id": "S4b.G1_gnss_loso_14mo_preorigin",
         "data_scope": "insar_wra_gnss_change:2021-08..2022-12",
         "fresh_units": ["s4b.wra_gnss.loso_preorigin"], "external_units": ext_g,
         "design_time_units": ["s4b.pilot.insar_gnss_2021_2023"],
         "procedure": (
             "Run P (LiCSBAS, pre-origin pairs only) -> stable-area reference (offset + "
             "east/north plane + height term per epoch, robust, stable pixels only) -> "
             "vertical-equivalent LOS/U (per-pixel U). Station value per epoch = median of "
             "the 3 x 3 multilooked pixels around the station (>= 4 valid in the LiCSBAS15 "
             "mask). GNSS = WRA daily height (raw vendor cache), 7-day centred rolling "
             "median, sampled at the InSAR epochs (+-3 d). Stations: on-fan WRA stations "
             "with >= 20 daily heights in each +-30 d window and no plan-catalogue step "
             f"inside the span ({', '.join(cand)}); a station is evaluable with >= 3 epochs "
             "having both values in each window. Tie: per epoch, subtract from the held-out "
             "station's InSAR the mean over the OTHER evaluable stations of (InSAR - GNSS), "
             "each station's difference demeaned over the analysis epochs (offset only; no "
             "rate, no plane). Error = tied InSAR change - GNSS change between the means "
             f"over +-30 d around {C.GNSS_T0} and {C.GNSS_T1} (14 months, pre-origin)."),
         "threshold": (f"RMS over evaluable stations <= {C.GNSS_LOSO_RMS_MAX_MM} mm with >= "
                       f"{C.GNSS_MIN_STATIONS} evaluable stations (fewer = fail)")},
        {"test_id": "S4b.C1_changhua_summer_coherence_preorigin",
         "data_scope": "insar_069A_coherence:2020-06..2022-09",
         "fresh_units": sorted(ext_c), "external_units": ext_c,
         "procedure": (
             "Pairs of the primary frame with both epochs in June-September of the same year "
             "2020, 2021 or 2022 and a temporal baseline <= 24 d (57 pairs). Per pair and "
             "active cell: median coherence (cc/255, cc 0 = NoData) of the native pixels whose "
             "centre lies in the cell. Zone statistic = median over the pairs of the fraction "
             "of the zone's active cells (denominator: every cell of the zone, cells outside "
             "the frame footprint count as not coherent) with median coherence >= 0.30."),
         "threshold": "per zone: statistic >= 0.30 -> covered; otherwise 'not covered'"},
        {"test_id": "S4b.Y_seasonal_amplitude_yunlin_preorigin",
         "data_scope": "insar_069A_seasonal_amplitude:2020-01..2022-12",
         "fresh_units": sorted(ext_y), "external_units": ext_y,
         "procedure": (
             "Per multilooked pixel of run P (stable-referenced vertical-equivalent, LiCSBAS15 "
             "mask, every epoch 2020-01..2022-12 finite): LS of a + b t + annual + "
             "semiannual; amplitude of the annual term; cell value = median over >= 5 valid "
             "pixels. Census density at r = 2, 5, 10 km: mean over the active cells whose "
             "centre lies within r of the cell centre of the 2020-2022 mean annual kWh "
             "(pump_electricity/preorigin/v1 cells_total value_all, all classes, all "
             "counties). Placebo: same over cells 15-20 km away. Thickness proxies from the "
             "model's layer geometry: zonal elastic skeletal storage Ske of the "
             "leveling-calibrated VEP column (zones at 205 / 182 km), and the distance west of "
             "the aquitard pinch-out line, max(0, 205 km - x). Spearman correlations. Y1: "
             "all Yunlin cells with an amplitude; 5 km-block bootstrap (1000 draws, seed "
             "4242). Y2: on each of the eight P3 battery subsets (s4b_battery.py, rebuilt "
             "from the design pilot, 1091 cells), a win when rho(amplitude, density 5 km) > "
             "|rho| of every competitor (placebo, Ske, west distance); a competitor constant "
             "within a subset (undefined rho) counts as 0."),
         "threshold": ("Y1: bootstrap 5th percentile of rho(amplitude, density 5 km) > 0; Y2: "
                       ">= 6 wins of 8. Pass = Y1 and Y2 -> the amplitude may serve as a rank "
                       "validator in Changhua (never a prior); otherwise it may not")}]
    return {
        "id": C.PREREG_ID, "wp": C.WP, "kind": "test",
        "title": "InSAR 069A (LiCSBAS, no atmospheric correction): GNSS leave-one-station-out "
                 "14-month change (gate), Changhua summer coherence coverage per zone, and the "
                 "Yunlin seasonal-amplitude validator test, on the pre-origin run",
        "tests": tests,
        "zones": {"changhua": list(C.CH_ZONES), "yunlin": list(C.YUNLIN_UNITS),
                  "source": rel_to_repo(C.UNITS_V3)},
        "priors": na("no model priors; InSAR is a validator only (U7)"),
        "burn_in": na("no model run"),
        "folds": {"Y1": "5 km x 5 km cell blocks (row // 5, col // 5), block bootstrap",
                  "Y2": "the eight P3 transfer-battery subsets (cell sets, no fitting)"},
        "fresh_units": sorted(set(ext_g) - {"s4b.pilot.insar_gnss_2021_2023"} | set(ext_c)
                              | set(ext_y)),
        "thresholds": {
            "G1": f"RMS <= {C.GNSS_LOSO_RMS_MAX_MM} mm, >= {C.GNSS_MIN_STATIONS} stations",
            "C1": f"per zone >= {C.COH_ZONE_MIN_FRAC} (coherence >= {C.COH_MIN})",
            "Y": f"Y1 bootstrap p05 > 0 and Y2 >= {C.BATTERY_MIN_SPLITS}/8",
            "product_gate": "insar_069A/v1 gate = G1 (the plan's numeric acceptance). C1 "
                            "labels each Changhua zone covered / not covered (not covered = "
                            "the product is not a validator there). Y decides only whether "
                            "the seasonal amplitude may be a Changhua rank validator.",
            "report_only": [
                "G1 rows: no GNSS tie; steps corrected by the catalogue size (TKJS, SJES "
                "included); reference offset + height only; offset only; constant U 0.833",
                "after the tests, on run F (full network): G1 in the same pre-origin window; "
                "the pilot's post-origin window 2021-12-05 -> 2023-02-20; NGL on-fan stations "
                "(DNAN, FUNY, HUWE, JSAN, PUS2) with the WRA tie",
                "C1 rows: per-year medians, cells with data, all pre-origin pairs",
                "Y rows: 2 and 10 km scales, all-Yunlin competitor correlations",
                "2025 ascending/descending decomposition (runs F and D) of the 2025 rate: "
                "east-west leakage of the vertical-equivalent, after the fact"]},
        "gate_scores": na("conjunctive per-test pass/fail; no weighted score"),
        "primary_model": {
            "licsbas_commit": C.LICSBAS_COMMIT, "licsbas_steps": C.LICSBAS_STEPS,
            "runs": {k: list(v) for k, v in C.RUNS.items()},
            "atmosphere": "none (D11)",
            "stable_area": {"land": True, "hgt_m": [C.STABLE_MIN_HGT_M, C.STABLE_MAX_HGT_M],
                            "min_dist_active_cell_centre_km": C.STABLE_MIN_DIST_KM,
                            "licsbas15_mask": True, "min_pixels_flag9": C.STABLE_MIN_PIXELS},
            "reference_terms": list(C.REF_TERMS),
            "vertical": C.VERTICAL, "cell_min_pixels": C.CELL_MIN_PIXELS,
            "georeference": "by index on the S4a window geotransform (cum.h5 geotransform "
                            "ignored)",
            "code": [rel_to_repo(HERE / c) for c in CODE]},
        "inputs": [rel_to_repo(HERE / c) for c in CODE] + [
            rel_to_repo(C.S4A_NOTES), rel_to_repo(C.S4A_PRODUCT / "MANIFEST.json"),
            rel_to_repo(C.S4A_RUN / f"pairs_{C.PRIMARY}.csv"),
            rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz"),
            rel_to_repo(C.RUN_DIR / "gnss_inventory.json"),
            rel_to_repo(C.RUN_DIR / "battery.json"), rel_to_repo(C.UNITS_V3),
            rel_to_repo(C.TWIN_GRID),
            "results/dataprep/pump_electricity/preorigin/v1/MANIFEST.json",
            "results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json",
            "chou-shui-data/data/ls_cache/ls-wra-gnss-obs__meta.parquet"],
        "interpretation": (
            "Context and disclosures. (1) S3's gnss_clean gate failed twice, so its products "
            "are refused as validators and as the off-fan reference; the reference here uses "
            "stable areas only (plan step 5 allows 'off-fan GNSS stations (S3) and stable "
            "areas'), and G1 uses the L0 WRA heights with the plan's step catalogue (stations "
            "with a catalogued step in the span left out). Stations are selected only by "
            "pre-origin data availability (LNJS, MDES, YSLL have no heights in the second "
            "window; CYES starts 2023), never by post-origin evidence. (2) Design-time "
            "evidence: the scout's pilot compared InSAR with 10 of these WRA stations over "
            "2021-10..2023-03 (8.2 mm on its 14-month change, post-origin end); the plan's "
            "10 mm was set after that pilot. G1 uses a fully pre-origin window instead "
            "(2021-09-30 -> 2022-11-30) and the full-fan processing; the pilot's window is "
            "reported after the fact only. (3) The reference scheme (offset + plane + height "
            "on stable pixels) is the plan's and was fixed before any InSAR displacement was "
            "read. A geometric concern was noted before any value was read: stable pixels lie "
            "only east of the fan (x about 196-223 km against a fan at 162-221 km), so the "
            "fitted east-west gradient is extrapolated up to about 50 km west; development "
            "looks at the 2025 descending run D (logged, post-origin, disjoint from these "
            "windows) showed per-epoch east gradients of order 0.2-2 mm/km. The scheme was "
            "not changed after that look; the offset-only and offset + height variants are "
            "registered as report-only rows. (4) Run P uses pre-origin pairs only, so no "
            "2023+ interferogram enters the gate (U7); the product run F (2020-01..2025-12) "
            "is checked against run P after the fact. (5) Thresholds are the plan's "
            "(provisional, fixed here): 10 mm; 30 % of cells at coherence 0.3; the amplitude "
            "test rules are this registration's reading of plan step 8 ('beat the placebo "
            "and a compressible-thickness proxy under the P3 transfer battery'), with the "
            "battery's own 6-of-8 split count. (6) Expectations: paddy summers decorrelate "
            "C-band, so some Changhua zones may fail C1; the northern band of Changhua lies "
            "outside the primary footprint (counted as not coherent). Thresholds are not "
            "changed after results."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs or a.dry_run:
        print(json.dumps(errs or {"ok": True, "n_inputs": len(d["inputs"]),
                                  "fresh_units": d["fresh_units"]}, indent=1))
        return 1 if errs else 0
    path, sha = prereg3.write_prereg(d)
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
