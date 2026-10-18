"""WP S4b repair round 2 (S4br): write the pre-registration ``S4b.insar_repair.v1`` with
``harness.prereg3`` BEFORE any re-test statistic exists.

    python -I dataprep_staging/s4_insar/s4br_prereg.py [--dry-run]

Registers ONE test on fresh units: G1 (the plan's numeric acceptance, threshold unchanged) on
the NGL GNSS family, which has never been compared with InSAR in the fresh window, on the
repaired pre-origin run P2. It also DECLARES, before they are computed, the second looks on
spent units (the registered G1 on the WRA family; Y) and the report-only rows; the A1 tool
cannot register a test without a fresh unit, so each second look is logged with
``prereg3.record_look`` (purpose ``second_look_score``) immediately before it is computed.
Requires ``S4b_evidence/repair_2026-10-18/design.json`` (``s4br_evaluate.py design``: the fresh
window and the NGL station set from metadata and epoch dates only).
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
import s4br_config as R  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

CODE = ["s4br_config.py", "s4br_lib.py", "s4br_evaluate.py", "s4br_prereg.py",
        "s4br_run_licsbas.sh", "s4br_remask.sh", "licsbas_env.sh", "make_licsbas_workdir.py",
        "s4b_config.py", "s4b_lib.py", "s4b_evaluate.py", "s4br_smoke.py"]
T_NGL = "S4b.r2.G1_ngl_fresh_preorigin"
UNIT = "s4b.r2.ngl_gnss.loso_fresh"


def doc() -> dict:
    na = prereg3.not_applicable
    des = json.loads((R.EVID / "design.json").read_text())
    w = des["window"]
    t0, t1, scope = w["t0"], w["t1"], w["scope"]
    elig = des["eligible"]
    st = des["stations"]
    ext = {UNIT: {
        "description": "NGL GNSS stations (L0 tenv3, IGS20) on active twin cells inside the "
                       "multilooked grid, eligible by the pre-registered rule in the fresh "
                       f"window; their run-P2 InSAR pixels, {scope}",
        "ids": [f"ngl_gnss:{s}" for s in elig], "window": scope}}
    ts = R.RUN_DIR / "P2" / "TS_GEOCml2"
    ngl_files = [rel_to_repo(next(d / f"{s}.tenv3" for d in R.NGL_DIRS
                                  if (d / f"{s}.tenv3").is_file())) for s in elig]
    procedure = (
        "Run P2 = LiCSBAS on pre-origin pairs only (20200101..20221231) with the repaired "
        "steps (M1: LiCSBAS12 --multi_prime --nullify, loop threshold pi; LiCSBAS13 "
        "--nullify_noloops; M2: LiCSBAS16 defaults, no deramp, no height term, no NaN fill; "
        "the product series is cum_filt.h5). Reference (M3): per epoch a robust two-pass LS "
        "(3 x 1.4826 MAD rejection) of offset + height term on the round-1 stable strip "
        "(land, DEM 100-600 m, >= 2.5 km from active cells, LiCSBAS15 mask), subtracted "
        "everywhere; vertical-equivalent = LOS / U per pixel. Station value per epoch = "
        "median of the 3 x 3 multilooked pixels around the station's nearest pixel (>= 4 "
        "valid in the LiCSBAS15 mask). GNSS = NGL daily up (u0 + up, tenv3), 7-day centred "
        "rolling median (>= 3 values), sampled at the InSAR epochs (nearest within +-3 d). "
        "Stations: every NGL station with a tenv3 file in L0 whose DataHoldings position lies "
        "in an active twin cell within 150 m of a multilooked pixel centre, with >= 20 daily "
        "solutions in each +-30 d window and no equipment change (steps.txt type 1) in "
        f"[t0 - 30 d, t1 + 30 d]; eligible here: {', '.join(elig)}. A station is evaluable "
        "with >= 3 epochs having both values in each window. Tie: per epoch, subtract from "
        "the held-out station's InSAR the mean over the OTHER evaluable NGL stations of "
        "(InSAR - GNSS), each station's difference demeaned over the analysis epochs "
        "(offset only; no rate, no plane). Error = tied InSAR change - GNSS change between "
        f"the means over +-30 d around t0 = {t0} and t1 = {t1} (14 months; the window rule "
        "of design.json). Code: s4br_evaluate.py g1_ngl (s4b_evaluate.loso_eval, "
        "s4b_lib.loso).")
    tests = [{
        "test_id": T_NGL, "data_scope": f"insar_ngl_gnss_change:{scope}",
        "fresh_units": [UNIT], "external_units": ext, "procedure": procedure,
        "threshold": (f"RMS over evaluable stations <= {C.GNSS_LOSO_RMS_MAX_MM} mm with >= "
                      f"{C.GNSS_MIN_STATIONS} evaluable stations (fewer = fail); the plan's "
                      "G1 threshold, unchanged")}]
    second = {
        "S4b.r2.G1_wra_second_look_preorigin": (
            "SECOND LOOK (spent units s4b.wra_gnss.loso_preorigin): the registered G1 of "
            "S4b.insar_acceptance.v1 unchanged (18 WRA stations, 2021-09-30 -> 2022-11-30, "
            "catalogue step rule, offset-only leave-one-out tie, 10 mm with >= 5 stations) on "
            "run P2 with M3. A pass is reported as 'conditional (second look)' and cannot open "
            "the gate. Logged with record_look before it is computed."),
        "S4b.r2.Y_second_look_preorigin": (
            "SECOND LOOK (spent units s4b.seasamp.yunlin): the registered Y of "
            "S4b.insar_acceptance.v1 unchanged (Y1 block bootstrap p05 > 0 and Y2 >= 6/8) on "
            "run P2 with M3. Conditional; logged before it is computed."),
        "C1": ("not re-tested: coherence is a LiCSAR product property no repair changes "
               "(diagnosis 1.2); Z2 stays 'not covered'")}
    report_only = [
        "NGL fresh window: the round-1 reference (offset + east/north plane + height) on P2; "
        "M3 on the unfiltered P2 series (cum.h5); the leave-one-out tie over the WRA and NGL "
        "stations together (WRA as tie stations only; logged look), errors at NGL only",
        "NGL in round 1's G1 window 2021-09-30 -> 2022-11-30 on P2 (round 1 looked there "
        "after the fact): SECOND LOOK, logged look",
        "WRA second look: the round-1 reference on P2; unfiltered P2; S3 gnss_clean full/v4 "
        "step-corrected heights (after_the_fact_analysis)",
        "network structure of P2, F2 and S2 (gaps, epochs, interferograms kept)"]
    return {
        "id": R.PREREG_ID, "wp": R.WP, "kind": "test",
        "title": "InSAR 069A repair round 2: G1 (GNSS leave-one-station-out 14-month change) "
                 "on the fresh NGL family on the repaired pre-origin run; declared second "
                 "looks on the WRA family and Y",
        "tests": tests,
        "zones": na("G1 is scored at GNSS stations; no zone"),
        "priors": na("no model priors; InSAR is a validator only (U7)"),
        "burn_in": na("no model run"),
        "folds": {"G1": "leave-one-station-out within the evaluable NGL stations"},
        "fresh_units": [UNIT],
        "thresholds": {
            "G1_ngl": f"RMS <= {C.GNSS_LOSO_RMS_MAX_MM} mm, >= {C.GNSS_MIN_STATIONS} "
                      "evaluable stations (unchanged from S4b.insar_acceptance.v1)",
            "product_gate": "insar_069A gate = G1 on fresh units (this NGL test). C1 labels "
                            "stay those of round 1 (Z2 not covered).",
            "second_looks": second, "report_only": report_only},
        "gate_scores": na("conjunctive per-test pass/fail; no weighted score"),
        "primary_model": {
            "licsbas_commit": C.LICSBAS_COMMIT,
            "licsbas_steps": R.LICSBAS_STEPS + [
                "amendment (S4b.repair.md Part 2b, before this registration): "
                "LiCSBAS15_mask_ts.py -t TS_GEOCml2 -i 500, then LiCSBAS16_filt_ts.py -t "
                "TS_GEOCml2 --n_para 1 --nopngs (s4br_remask.sh)"],
            "runs": {k: list(v) for k, v in R.RUNS.items()}, "gate_run": "P2",
            "series": R.CUM_FILE, "atmosphere": "none (D11); LiCSBAS16 filter (M2)",
            "reference_terms": list(R.REF_TERMS),
            "stable_area": {"land": True, "hgt_m": [C.STABLE_MIN_HGT_M, C.STABLE_MAX_HGT_M],
                            "min_dist_active_cell_centre_km": C.STABLE_MIN_DIST_KM,
                            "licsbas15_mask": True},
            "vertical": C.VERTICAL, "fresh_window": w,
            "ngl_station_rule": {k: v for k, v in st.items() if v.get("cell", -1) >= 0},
            "code": [rel_to_repo(HERE / c) for c in CODE]},
        "inputs": [rel_to_repo(HERE / c) for c in CODE] + [
            rel_to_repo(R.EVID / "design.json"),
            rel_to_repo(ts / R.CUM_FILE), rel_to_repo(ts / "cum.h5"),
            rel_to_repo(ts / "results" / "mask"),
            rel_to_repo(ts / "info" / "12network_gap_info.txt"),
            rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz"),
            rel_to_repo(C.RUN_DIR / "gnss_inventory.json"),
            rel_to_repo(C.RUN_DIR / "battery.json"),
            rel_to_repo(R.NGL_HOLDINGS), rel_to_repo(R.NGL_STEPS), *ngl_files,
            rel_to_repo(C.UNITS_V3), rel_to_repo(C.TWIN_GRID),
            "results/dataprep/preregistration/S4b.insar_acceptance.v1.json",
            "results/dataprep/reports/S4b_evidence/repair_2026-10-18/snapshots/"
            "S4b.repair.part1-2.20261017T211120Z.md",
            "results/dataprep/pump_electricity/preorigin/v1/MANIFEST.json",
            "results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json",
            "chou-shui-data/data/ls_cache/ls-wra-gnss-obs__meta.parquet"],
        "interpretation": (
            "Repair round 2 of WP S4b. (1) Diagnosis first: S4b.repair.md Parts 1-2 (snapshot "
            "sha256 dbacb416e8eabf3e626ce804ccdb8ba0480b04beefcbf9e2fbc3672c68d1ffd0, "
            "2026-10-17T21:11:20Z) were written and hashed before any repaired method was "
            "coded; every change M1-M5 is tied there to a diagnosed defect. The diagnosis read "
            "only spent units (the 18 G1 WRA stations 2021-08..2022-12, the Y Yunlin cells, the "
            "C1 zone cells) and InSAR-internal structure, never the NGL family. (2) Freshness: "
            "the only earlier InSAR-vs-NGL comparison is S4b round 1's after-the-fact row "
            "(look 2026-10-13T19:40:01Z, unit s4b.ngl_onfan.preorigin, 2021-08..2022-12, run P "
            "with the WRA tie: n 4, 17.6 mm; run F 14.5 mm; JSAN had the largest error, HUWE "
            "had no valid pixel). The author knows those numbers; the fresh window "
            f"({scope}) is disjoint from that look, and the station rule is mechanical (no "
            "station is chosen or dropped by name). Other looks at NGL series: S3 (GNSS-only "
            "cleaning and injections), S1b (rings against NGL DNAN, 2020-01..2025-05; no "
            "InSAR), S5 (data build of the fused field over every instrument, 2012-01..2026-01, "
            "unit of unknown membership; no InSAR-vs-NGL score reported). These are disclosed; "
            "none compared InSAR with NGL in the fresh window. (3) Step rule: NGL's steps.txt "
            "equipment changes (type 1) exclude a station, as in round 1's NGL row; type-2 "
            "entries are distance-based earthquake flags (in 2020-21: M5.5-6.1 at 86-180 km, "
            "static offsets far below 1 mm) and are listed, not applied. This rule was fixed "
            "before any NGL height was read. (4) Window rule: fixed before any value was "
            "read; its result depends on run P2's epoch dates only (LiCSAR has no epoch "
            "between 2021-04-30 and 2021-08-16). (5) Tie: within the NGL family (the WRA "
            "family's own common mode would otherwise enter every NGL error); the combined "
            "WRA+NGL tie is report only. (6) Limits stated in advance: the fresh window lies "
            "in the Sentinel-1A/B era (6-day revisit) and has no network gap in P2, while the "
            "product validates 2022-26 (12-day revisit, summer decorrelation); a pass shows the "
            "repaired processing meets 10 mm in a gap-free pre-origin window, not that every "
            "2022-26 summer does. With 5-6 candidate stations the test fails by rule if fewer "
            "than 5 are evaluable. (7) S3 gnss_clean v4 (gate failed) is used after the fact "
            "only. (8) Thresholds are the plan's and unchanged. (9) Amendment to M1 before this "
            "registration (S4b.repair.md Part 2b, snapshot 20261017T214019Z, sha256 "
            "75d097f7...): LiCSBAS15 re-run with -i 500, LiCSBAS's documented no-loop "
            "threshold for --nullify_noloops (the default 'global no-loop count + 1' = 1 masked "
            "41 % of P2's pixels whose no-loop data step 13 had already removed); P2 keeps "
            "74.6 % of its pixels. Only InSAR-internal summaries were read. (10) A smoke test "
            "of the registered code (s4br_smoke.py) ran on the OFF-FAN NGL stations inside the "
            "grid (CHUA, GUKN, LSB0, PAKU, PINT, S103: never G1 units) in round 1's G1 window, "
            "with the spent WRA stations as tie stations (look logged, purpose smoke_test; its "
            "numbers are not reported and were not read beyond completion and valid-epoch "
            "counts).")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if a.dry_run:
        print(json.dumps({"tests": [t["test_id"] for t in d["tests"]],
                          "fresh_units": d["fresh_units"],
                          "ids": d["tests"][0]["external_units"][UNIT]["ids"],
                          "scope": d["tests"][0]["data_scope"]}, indent=1))
        return 0
    path, sha = prereg3.write_prereg(d)
    print(json.dumps({"path": rel_to_repo(path), "sha256": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
