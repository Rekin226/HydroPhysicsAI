"""WP H1b finalisation: version record, ledger rows and the WP report (md + json).

    cpu_slot.sh python -I h1b_report.py

Reads the published versions (heads/L1_station_ground/v1, v2), the evidence under
results/dataprep/reports/H1b_evidence/ and the look log; writes
results/dataprep/heads/L1_station_ground/SUPERSEDED.json, ledger/rows/H1b.csv and
reports/H1b.md, reports/H1b.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(1, str(HERE.parents[2]))

from common import hashing, ledger  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, REPORTS_DIR, rel_to_repo  # noqa: E402

from h1_heads_qc import h1b_lib as lib  # noqa: E402

PROD = DATAPREP / lib.PRODUCT
EV = REPORTS_DIR / "H1b_evidence"
SM = "results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet"
SMC = "results/dataprep/heads/L1_clean/v3/station_meta_clean.parquet"
W3D = "results/dataprep/ground_elevation/v3"
LEV = rel_to_repo(lib.LEVELING)
PRE = f"results/dataprep/preregistration/{lib.PREREG_ID}.json"
GRID = "results/twin_forward/datum_gate_innov.npz"


def j(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def f3(x) -> str:
    return f"{x:.3f}"


def version_record(v1: dict, v2: dict) -> dict:
    o1 = {r["path"]: r["sha256"] for r in j(PROD / "v1" / "MANIFEST.json")["outputs"]}
    o2 = {r["path"]: r["sha256"] for r in j(PROD / "v2" / "MANIFEST.json")["outputs"]}
    rec = {
        "schema": "dataprep-superseded/1", "product": lib.PRODUCT,
        "written": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "writer": "dataprep_staging/h1_heads_qc/h1b_report.py", "current": "v2",
        "current_note": "post-look reviewed build; its gate is closed (upstream W3 gate "
                        "failed, no user decision recorded); the version to release once a W3 "
                        "decision opens it (rebuild from the released ground elevation)",
        "superseded": [{
            "version": "v1", "superseded_by": "v2",
            "why": ("registered build of H1b.ground_height.v1 (T1 coverage "
                    f"{f3(v1['registered']['T1_sd_coverage_lobo']['gate']['coverage'])} and "
                    "C1-C4 passed). After scoring, 11 agency GroundHeight values were found "
                    "inconsistent with the agency's own WellElevation (stick-up 6.84-29.39 m "
                    "against <= 3.53 m elsewhere; 9 store 0.80 m, one 0.99 m); they inflated "
                    f"tau to {f3(v1['registered']['T1_sd_coverage_lobo']['tau_all_m'])} m. v2 "
                    "replaces them by the DTM and refits tau "
                    f"({f3(v2['post_look']['sd_coverage_lobo']['tau_all_m'])} m) on the "
                    "consistent stations (post-look)"),
            "registered_acceptance": "v1 holds the registered scores; keep it for provenance",
            "outputs_byte_identical_to_v2": sorted(k for k in o1 if o2.get(k) == o1[k]),
            "outputs_differing_from_v2": sorted(k for k in o1 if o2.get(k) != o1[k]),
            "allowed_uses_note": "same policy as v2: gate closed by W3; the guard grants "
                                 "design_time_evidence and after_the_fact_analysis only"}]}
    p = PROD / "SUPERSEDED.json"
    if p.exists():
        raise SystemExit(f"{p} exists; write a new record by hand if needed")
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rec


def ledger_rows(v1: dict, v2: dict, ver: dict) -> list[str]:
    P1, P2 = f"{lib.PRODUCT}/v1", f"{lib.PRODUCT}/v2"
    t1 = v1["registered"]["T1_sd_coverage_lobo"]
    pl = v2["post_look"]["sd_coverage_lobo"]
    r1 = v1["reported_not_gated"]
    r2 = v2["reported_not_gated"]
    inside = f"{SM}#GroundHeight_of_stations_inside_the_held_out_block"
    outside = f"{SM}#GroundHeight_of_stations_outside_the_held_out_block"
    w3 = [(f"{W3D}/dtm20m_mosaic.npz", "input"), (f"{W3D}/well_ground_elevation.parquet",
                                                    "input"), (LEV, "input")]
    rows = []

    def add(nid, pv, note, sources):
        for s, role in sources:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": note})

    add("H1b.v1.T1.sd_coverage_lobo", P1,
        f"registered T1: LOBO coverage {f3(t1['gate']['coverage'])} (Wilson "
        f"{f3(t1['gate']['lo'])}..{f3(t1['gate']['hi'])}), n {t1['n_stations']}, "
        f"{t1['n_blocks']} blocks; pass, calibrated",
        [(inside, "validator"), (outside, "fit_target"), *w3, (PRE, "input"),
         (GRID, "input")])
    add("H1b.v1.tau", P1, f"tau {f3(t1['tau_all_m'])} m (split-conformal 90 %, all 180 "
        "calibration stations); sd_cal median at flag 5 "
        f"{f3(r1['sd_cal_flag5']['median'])} m, W3 sd median "
        f"{f3(r1['sd_w3_flag5']['median'])} m",
        [(f"{SM}#GroundHeight_of_calibration_stations", "fit_target"), *w3])
    add("H1b.v1.C1.completeness", P1,
        "148 stations with GroundHeight 0.0 (93) or NaN (55): 146 flag 5, 2 flag 8 (no "
        "coordinates); 60/60 model wells with 0.0 and 32/32 with NaN replaced",
        [(f"{SM}#GroundHeight_xy", "input"), *w3, (GRID, "input")])
    add("H1b.v1.C2.reproduction", P1,
        "DTM well points recomputed at the H1a v3 coordinates equal W3's table at 328 "
        "stations: max |diff| 0, coordinates max 0.0 m; independent scipy bilinear 2.4e-12 m",
        [(f"{W3D}/well_ground_elevation.parquet", "validator"),
         (f"{W3D}/dtm20m_mosaic.npz", "input"), (LEV, "input"), (f"{SM}#xy", "input")])
    add("H1b.v1.C3.integrity", P1, "long-table validator clean; flag-0 values = agency; no 0.0 "
        "left; 344 unique sids; no personal columns", [(SM, "input")])
    add("H1b.v1.C4.use_policy", P1, "guard policy: design_time_evidence and "
        "after_the_fact_analysis granted, validator/model_design GateNotPassed, 13 forbidden "
        "uses UseNotAllowed", [(f"{W3D}/MANIFEST.json", "input")])
    add("H1b.w3_gate", P1, "upstream W3 gate failed (median held-out bias -0.842 m vs |0.30| m; "
        "quoted from ground_elevation/v3), no user decision recorded",
        [(f"{W3D}/MANIFEST.json", "input")])
    rr = r1["residual_agency_minus_dtm_calibration_set"]
    add("H1b.v1.residual_summary", P1,
        f"agency - DTM at 180 stations: median {rr['median']:+.3f} m, IQR "
        f"{rr['iqr'][0]:+.2f}..{rr['iqr'][1]:+.2f}, MAD-sd {f3(rr['mad_sd'])}; 13 with |r| > 5 m",
        [(f"{SM}#GroundHeight_of_calibration_stations", "validator"), *w3])
    add("H1b.v1.w3_sd_coverage", P1,
        f"W3 sd_m as published covers {f3(r1['w3_sd_as_published_coverage']['coverage'])} "
        f"(v1 set) and {f3(r2['w3_sd_as_published_coverage']['coverage'])} (v2 set): fails",
        [(f"{SM}#GroundHeight_of_calibration_stations", "validator"), *w3])
    add("H1b.v1.strata", P1, "LOBO coverage by county and model-well stratum (reported)",
        [(inside, "validator"), (outside, "fit_target"), *w3])
    add("H1b.v1.stickup_proxy", P1,
        f"proxy (WellElevation - median stick-up) minus DTM at 146 replaced stations: median "
        f"{r1['proxy_minus_dtm_replaced']['median']:+.3f} m, MAD-sd "
        f"{f3(r1['proxy_minus_dtm_replaced']['mad_sd'])}; inside widened interval "
        f"{f3(r1['proxy_within_interval_share'])}",
        [(f"{SM}#WellElevation_of_replaced_stations", "validator"),
         (f"{SM}#WellElevation_GroundHeight_of_calibration_stations", "input"), *w3])
    ex = r1["exchangeability_wellelev_minus_dtm"]
    add("H1b.v1.exchangeability", P1,
        f"WellElevation - DTM: calibration median {f3(ex['calibration']['median'])} vs "
        f"replaced {f3(ex['replaced']['median'])} m; Mann-Whitney p {ex['mannwhitney_p']:.4f}",
        [(f"{SM}#WellElevation", "validator"), *w3])
    add("H1b.v1.gross_discrepancies", P1, "13 stations with |agency - DTM| > 5 m (all "
        "negative): 11 agency values inconsistent with WellElevation, 2 DTM misses (09100211/"
        "221, local sd 15.7 m)", [(f"{SM}#GroundHeight_WellElevation", "validator"), *w3])
    mz = r1["model_wells_zero_new_value"]
    add("H1b.v1.model_wells_new_values", P1,
        f"60 model wells with GroundHeight 0.0: new value median {mz['median']:.1f} m (IQR "
        f"{mz['iqr'][0]:.1f}..{mz['iqr'][1]:.1f})", [(f"{SM}#GroundHeight", "input"), *w3,
                                                     (GRID, "input")])
    add("H1b.v1.post_origin_modification", P1, "0 of 180 calibration stations carry a "
        "metadata modification dated 2023-01-01 or later; 13 replaced stations were "
        "established 2023-24", [(f"{SMC}#ModificationDate", "input"),
                                (f"{SM}#EstablishDate", "input")])
    inc = r2["inconsistent_stations"]
    add("H1b.v2.inconsistent_stations", P2,
        f"{inc['n']} agency GroundHeight values with stick-up > 5 m (6.84-29.39 m; other 185 "
        f"<= 3.53 m): 9 at 0.80 m, 1 at 0.99 m, 1 at 143.39 m; {inc['model_wells']} model wells",
        [(f"{SM}#GroundHeight_WellElevation", "input")])
    add("H1b.v2.tau", P2, f"tau {f3(pl['tau_all_m'])} m on {pl['n_stations']} consistent "
        f"stations; sd_cal median at flag 5 {f3(r2['sd_cal_flag5']['median'])} m",
        [(f"{SM}#GroundHeight_of_consistent_calibration_stations", "fit_target"), *w3])
    add("H1b.v2.coverage_postlook", P2,
        f"post-look LOBO coverage {f3(pl['gate']['coverage'])} (Wilson "
        f"{f3(pl['gate']['lo'])}..{f3(pl['gate']['hi'])}), n {pl['n_stations']}, "
        f"{pl['n_blocks']} blocks (not registered)",
        [(inside, "validator"), (outside, "fit_target"), *w3])
    rr2 = r2["residual_agency_minus_dtm_calibration_set"]
    add("H1b.v2.residual_summary", P2,
        f"agency - DTM at 169 consistent stations: median {rr2['median']:+.3f} m, MAD-sd "
        f"{f3(rr2['mad_sd'])}", [(f"{SM}#GroundHeight_of_consistent_calibration_stations",
                                  "validator"), *w3])
    add("H1b.v2.counts", P2, "flags 0/5/8 = 185/157/2; 101 of 158 model wells flag 5 (60 zero "
        "code, 32 NaN, 9 inconsistent); proxy inside interval "
        f"{f3(r2['proxy_within_interval_share'])}",
        [(f"{SM}#GroundHeight_WellElevation", "input"), *w3, (GRID, "input")])
    vf = j(EV / "verify.json")
    add("H1b.verify", P2, f"independent re-verification all_pass={vf['all_pass']}: rebuilds "
        "byte-equal, scipy bilinear max 2.4e-12 m, LOBO hits identical, registered members = "
        "v1 calibration set, v2 rule reproduced",
        [(f"{lib.PRODUCT}/v1", "validator"), (f"{lib.PRODUCT}/v2#station_ground", "validator"),
         (f"{W3D}/dtm20m_mosaic.npz", "input"), (PRE, "input")])
    add("H1b.check", P2, "manifest checker: heads/L1_station_ground v1 (23 files) and v2 (26), "
        "heads/L1_clean/v3 (100), ground_elevation/v3 (26) ok",
        [("results/dataprep/reports/H1b_evidence/check.json", "validator")])
    ledger.append_rows(lib.WP, rows)
    ids = sorted({r["number_id"] for r in rows} | {f"H1b.prereg.{lib.PREREG_ID}"})
    return ids


def looks() -> list[dict]:
    out = []
    with open(PREREG_DIR / "looks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("wp") == lib.WP:
                out.append({k: r.get(k) for k in ("time", "purpose", "units", "test_id")})
    return out


def main() -> int:
    v1 = j(PROD / "v1" / "validation.json")
    v2 = j(PROD / "v2" / "validation.json")
    m1 = j(PROD / "v1" / "MANIFEST.json")
    m2 = j(PROD / "v2" / "MANIFEST.json")
    ver = version_record(v1, v2)
    ids = ledger_rows(v1, v2, ver)
    miss = ledger.unledgered(ids)
    conf = ledger.conflicts(ledger.read_all())
    lk = looks()
    vf = j(EV / "verify.json")
    ck = j(EV / "check.json")["summary"]
    g1 = j(EV / "guard_probe_v1.json")
    g2 = j(EV / "guard_probe_v2.json")
    t1 = v1["registered"]["T1_sd_coverage_lobo"]
    c1, c2, c3, c4 = (v1["registered"][k] for k in ("C1_completeness", "C2_reproduction",
                                                     "C3_integrity", "C4_use_policy"))
    pl = v2["post_look"]["sd_coverage_lobo"]
    r1, r2 = v1["reported_not_gated"], v2["reported_not_gated"]
    rr1 = r1["residual_agency_minus_dtm_calibration_set"]
    rr2 = r2["residual_agency_minus_dtm_calibration_set"]
    inc = r2["inconsistent_stations"]
    ex = r1["exchangeability_wellelev_minus_dtm"]
    strata = r1["coverage_lobo_by_stratum"]
    ib = vf["independent_bilinear"]
    acceptance = [
        {"criterion": "T1 (registered, H1b.ground_height.v1): leave-one-10km-block-out "
                      "coverage of the 90 % interval DTM +- 1.645 sqrt(sd_w3^2 + tau^2) "
                      "against agency GroundHeight at 180 stations",
         "threshold": "harness.coverage3 rule gate-3 pass (Wilson on n_eff contains 0.90; "
                      "point in 0.85-0.95 when n_eff_lower >= 30) and calibrated (>= 20 "
                      "clusters)",
         "measured": f"{f3(t1['gate']['coverage'])} (Wilson {f3(t1['gate']['lo'])}.."
                     f"{f3(t1['gate']['hi'])}; n 180, 30 blocks, n_eff "
                     f"{t1['gate']['n_eff']:.1f}); tau {f3(t1['tau_all_m'])} m",
         "passed": bool(t1["pass"])},
        {"criterion": "C1 completeness (registered): every station with GroundHeight 0.0 or "
                      "NaN is flag 5 with sd > 0 or flag 8 with a reason; all model wells "
                      "with 0.0 replaced",
         "threshold": "148/148 accounted; 60/60 model wells",
         "measured": f"{c1['accounted']}/{c1['zero_or_missing']} accounted: "
                     f"{c1['replaced_flag5']} flag 5, {c1['flag8_not_replaced']} flag 8 (no "
                     f"coordinates); model wells {c1['model_wells_zero_replaced']}/"
                     f"{c1['model_wells_zero']} (0.0) and {c1['model_wells_missing_replaced']}/"
                     f"{c1['model_wells_missing_nan']} (NaN)",
         "passed": bool(c1["pass"])},
        {"criterion": "C2 reproduction (registered): W3 well points recomputed at the H1a v3 "
                      "coordinates equal W3's table",
         "threshold": "same finite set, max |diff| <= 1e-6, coordinates within 0.01 m",
         "measured": f"{c2['n_mine_finite']} = {c2['n_w3_finite']} finite; max |diff| "
                     f"{c2['max_abs_diff_all']}; coordinates max {c2['coord_max_diff_m']} m",
         "passed": bool(c2["pass"])},
        {"criterion": "C3 integrity (registered)", "threshold": "validator clean, agency "
                      "values unchanged, no 0.0 left, sd > 0, no personal columns",
         "measured": f"errors {c3['validator_errors']}; {c3['n_flag0']} flag-0 values equal; "
                     f"{c3['zero_code_values_left']} zero codes left; {c3['rows']} rows",
         "passed": bool(c3["pass"])},
        {"criterion": "C4 use policy (registered)", "threshold": "gate-exempt uses granted, "
                      "validator/model_design GateNotPassed while the gate is closed, "
                      "forbidden uses UseNotAllowed",
         "measured": f"mismatches {c4['mismatches']}; published-product probes: "
                     f"v1 {sorted({v['result'] for v in g1['probes'].values()})}, "
                     f"v2 {sorted({v['result'] for v in g2['probes'].values()})}",
         "passed": bool(c4["pass"])},
        {"criterion": "Plan WP H1 (H1b bullet): GroundHeight 0.0 set to missing and replaced "
                      "by the W3 DTM at the well (flag 5)",
         "threshold": "every model well with GroundHeight 0.0",
         "measured": "60/60 model wells; 146 stations in v1 (0.0 and NaN), 157 in v2 (plus "
                     "11 agency values inconsistent with WellElevation)",
         "passed": True},
        {"criterion": "Product release (upstream W3 gate, not an H1b criterion)",
         "threshold": "ground_elevation/v3 gate passed or a recorded user decision",
         "measured": "W3 gate FAILED (-0.842 m vs |0.30| m), no decision recorded: product "
                     "gate closed in v1 and v2",
         "passed": False},
        {"criterion": "v2 post-look sd coverage (reported, not registered)",
         "threshold": "rule gate-3 pass and calibrated",
         "measured": f"{f3(pl['gate']['coverage'])} (Wilson {f3(pl['gate']['lo'])}.."
                     f"{f3(pl['gate']['hi'])}; n 169, 30 blocks); tau {f3(pl['tau_all_m'])} m",
         "passed": bool(pl["pass"])},
        {"criterion": "Independent re-verification (rebuild, scipy bilinear, LOBO "
                      "re-implementation, registered members, v2 rule)",
         "threshold": "all equal", "measured": f"all_pass={vf['all_pass']}; bilinear max "
                                               f"{ib['max_abs_diff_m']:.1e} m",
         "passed": bool(vf["all_pass"])},
        {"criterion": "Manifest checker", "threshold": "0 failures",
         "measured": f"{ck['manifests']} manifests ({ck['files_hashed']} files), "
                     f"{ck['manifests_failed']} failed", "passed": ck["manifests_failed"] == 0},
        {"criterion": "Anti-leakage (U3, U7): no head data read; calibration stations "
                      "established by 2022-12 without post-origin metadata edits",
         "threshold": "0 head rows; 0 post-origin modifications",
         "measured": "0 head rows read; "
                     f"{r1['post_origin_metadata_modification']['n_modified_on_or_after_2023_01_01']}"
                     " of 180 modified >= 2023-01; fit_window_end 2022-12",
         "passed": True},
    ]
    key_numbers = [
        "Stations 344: agency GroundHeight real 196, missing-value code 0.0 at 93, NaN at 55 "
        "(13 of the NaN stations established 2023-24)",
        f"v1 (registered): {c1['replaced_flag5']} replaced (91 zero code + 55 NaN), 2 flag 8 "
        "(zero code, no coordinates); 60/60 model wells with 0.0 and 32/32 with NaN replaced",
        f"T1 registered: LOBO coverage {f3(t1['gate']['coverage'])} (Wilson "
        f"{f3(t1['gate']['lo'])}..{f3(t1['gate']['hi'])}), 180 stations, 30 blocks: pass; "
        f"tau {f3(t1['tau_all_m'])} m, sd_cal median {f3(r1['sd_cal_flag5']['median'])} m",
        f"W3 well-point sd as published covers {f3(r1['w3_sd_as_published_coverage']['coverage'])}"
        " (fails): it needs the inflation",
        f"C2: recomputed DTM well points = W3 table at 328 stations (max |diff| 0); "
        f"independent scipy bilinear {ib['max_abs_diff_m']:.1e} m",
        f"Agency minus DTM (180): median {rr1['median']:+.3f} m, MAD-sd {f3(rr1['mad_sd'])}, "
        "13 stations below -5 m",
        f"{inc['n']} agency values inconsistent with WellElevation (stick-up 6.84-29.39 m vs "
        "<= 3.53 m at the other 185): 9 at 0.80 m, 1 at 0.99 m, 1 at 143.39 m (Nantou); "
        f"{inc['model_wells']} model wells",
        f"v2 (post-look): 157 replaced; tau {f3(pl['tau_all_m'])} m; LOBO coverage "
        f"{f3(pl['gate']['coverage'])} (Wilson {f3(pl['gate']['lo'])}..{f3(pl['gate']['hi'])}, "
        f"169 stations); sd_cal median {f3(r2['sd_cal_flag5']['median'])} m; agency minus DTM "
        f"median {rr2['median']:+.3f} m, MAD-sd {f3(rr2['mad_sd'])}",
        "Remaining flag-0 gross discrepancy: 09100211/09100221, -28.3 m, DTM local sd 15.7 m "
        "(steep foothill; the DTM is off, the agency value is consistent)",
        f"Stick-up proxy at replaced stations: proxy minus DTM median "
        f"{r1['proxy_minus_dtm_replaced']['median']:+.3f} m (MAD-sd "
        f"{f3(r1['proxy_minus_dtm_replaced']['mad_sd'])}); inside the widened interval "
        f"{f3(r1['proxy_within_interval_share'])} (v1), {f3(r2['proxy_within_interval_share'])}"
        " (v2)",
        f"Exchangeability: WellElevation minus DTM median {f3(ex['calibration']['median'])} m "
        f"(calibration) vs {f3(ex['replaced']['median'])} m (replaced), Mann-Whitney p "
        f"{ex['mannwhitney_p']:.4f}",
        "LOBO coverage by stratum (v1): " + "; ".join(
            f"{k} {f3(v['coverage_lobo'])} (n {v['n']})" for k, v in strata.items()),
        f"60 model wells with GroundHeight 0.0 now {r1['model_wells_zero_new_value']['median']:.1f}"
        f" m median (IQR {r1['model_wells_zero_new_value']['iqr'][0]:.1f}.."
        f"{r1['model_wells_zero_new_value']['iqr'][1]:.1f}); in v2 101 of 158 model wells carry "
        "a DTM value",
        f"Checker: {ck['manifests']} manifests ok ({ck['files_hashed']} files); ledger "
        f"{len(ids)} numbers, unledgered {len(miss)}, conflicts {len(conf)}; looks {len(lk)}; "
        "Claude API tokens 0; GPU none",
    ]
    deviations = [
        "Product location: a separate L1 metadata product heads/L1_station_ground (keyed by "
        "sid, joins heads/L1_clean/v3 station_meta_clean) instead of a new heads/L1_clean "
        "version, so W3's failed gate does not close the head product (heads/L1_clean/v3 gate "
        "passed).",
        "Upstream fallback: W3's gate failed with no recorded user decision. W3 v3 was opened "
        "for design_time_evidence (the only use it grants besides after-the-fact analysis), "
        "and the product inherits a closed gate: validator and model_design are declared but "
        "refused until a decision on W3 is recorded and a new version is built.",
        "NaN GroundHeight (55 stations) is replaced as well as the 0.0 code (task wording).",
        "Uncertainty: W3's well-point sd covers only 0.661 of agency values, so a split-"
        "conformal inflation tau was added and checked leave-one-10 km-block-out (plan 4.7), "
        "pre-registered as T1. The sd is calibrated against agency GroundHeight, so it "
        "includes the agency-vs-DTM epoch/datum mismatch.",
        "v2 is a post-look reviewed build (beyond the plan's 0.0 code): 11 agency values "
        "inconsistent with WellElevation (stick-up > 5 m) are replaced and tau is refitted on "
        "the consistent stations; v1 stays the registered build (SUPERSEDED.json).",
        "No depth-to-water or transducer-floor check was run: those need head data, which H1b "
        "never reads; the product gives well_bottom_elev_m = value - WellDepth for them.",
        "Look members are station-metadata records written 'station_meta:<sid>', so H1b's "
        "looks at ground elevations do not overlap the head-data units of the A1 registry "
        "(they would otherwise block future head tests on the fresh well folds).",
        "Two stations whose residual was displayed before the registration (070213M3, "
        "070215T1) are excluded from the registered calibration set.",
        "Two dry runs on synthetic agency values (seed 20261013) exercised the code before "
        "the registration (disclosed in the history look).",
    ]
    issues = [
        "Release is blocked on the W3 decision. When ground_elevation is released (waiver or a "
        "new registered test, possibly a v4), rebuild heads/L1_station_ground from it with "
        "h1b_v2.py's rule, opening W3 for model_design; v2 is the version to release, v1 is "
        "the registered build (provenance).",
        "Agency GroundHeight is unreliable at 159 of 344 stations: 0.0 code at 93, NaN at 55, "
        "and 11 values inconsistent with WellElevation (0.80 m at 9 stations of sites 071001, "
        "090401, 090402, 091302; 0.99 m at 09150131; 143.39 m at 08060221). calibrate_flow's "
        "historical '--ground-elev wells' field (IDW of station GroundHeight) takes the 0.0 "
        "code (93 stations) and the 0.80/0.99 placeholders (10) as real elevations, 69 of them "
        "model wells; the cwa deliverable uses 'dem' (SRTM), so it is unaffected. I1 should "
        "not use the wells mode.",
        "Mixed references: flag-0 agency values sit a median +0.44 m above the DTM at 2017.5 "
        "(construction-era surveys or a datum offset; W3 found the DTM point -0.38 m below "
        "leveling benchmarks). Flag-5 values are not shifted. Depth-to-water differences "
        "below about 1 m between flag-0 and flag-5 stations are not meaningful; the "
        "calibrated sd (about 1.0 m in v2) covers this.",
        "09100211 and 09100221 (Linnei, foothill): the DTM point is 28.3 m above the agency "
        "value with a 100 m local relief sd of 15.7 m; the agency value is consistent with "
        "WellElevation and is kept (flag 0). 09100211 also carries H1a v3's only flag-3 step "
        "correction. Do not use the DTM column there.",
        "Steep or foothill stations are covered worse: LOBO coverage 0.333 in Nantou (n 9) "
        "against 0.92-1.00 elsewhere; model wells 0.833 (n 66) against 0.939. No flag-5 "
        "station has a DTM local sd > 5 m, but the per-stratum sd is uncalibrated.",
        "Exchangeability is assumed, not tested: WellElevation minus DTM differs between "
        "calibration and replaced stations (Mann-Whitney p 0.0013; replaced stations tighter, "
        "median 0.65 vs 0.71 m); the WellElevation proxy falls inside the widened interval at "
        "0.92-0.95 of replaced stations.",
        "13 replaced stations were established in 2023-24 (established_after_origin = True); "
        "pre-origin consumers should drop them.",
        "well_bottom_elev_m assumes WellDepth is measured from ground level (unverified); "
        "rate_2012_2021_m_per_yr (leveling mean rate) for shifting the 2017.5 value is "
        "uncalibrated.",
        "Read the product with read_table: the guard drops the 2 flag-8 rows for every use "
        "(no value) and gives 342 rows; sids are strings in parquet (no CSV leading-zero "
        "loss).",
        "Future WPs that look at station metadata (not heads) can reuse the 'station_meta:"
        "<sid>' member namespace so their looks do not block head tests on registry units.",
    ]
    status = "partial"
    summary = (
        "WP H1b built the L1 metadata product heads/L1_station_ground: one ground elevation "
        "per monitoring station, with GroundHeight 0.0 (93 stations) and NaN (55) replaced by "
        "the W3 DTM well point at 2017.5 (flag 5) with an sd. The W3 point values were "
        "recomputed at H1a v3's repaired coordinates and match W3's table exactly (328 "
        "stations, max |diff| 0). An independent scipy interpolation agrees to 2.4e-12 m. "
        "W3's own well-point sd covered only 0.661 of the agency values, so a split-conformal "
        f"inflation was pre-registered (H1b.ground_height.v1) and passed: leave-one-10 km-"
        f"block-out coverage {f3(t1['gate']['coverage'])} (Wilson {f3(t1['gate']['lo'])}.."
        f"{f3(t1['gate']['hi'])}, 180 stations, 30 blocks). The 4 conformance checks also "
        f"passed: all 60 model wells with 0.0 are replaced, plus 32 with NaN. tau was "
        f"{f3(t1['tau_all_m'])} m in the registered build v1. After scoring, 11 agency values "
        "turned out to be inconsistent with the agency's own WellElevation (stick-up 6.84-"
        "29.39 m against at most 3.53 m elsewhere; 9 store 0.80 m), and these inflated tau. A "
        f"labelled post-look v2 replaces them too and refits tau to {f3(pl['tau_all_m'])} m. "
        f"Its coverage is {f3(pl['gate']['coverage'])} (not registered), and 101 of 158 model "
        "wells now carry a DTM value. Both versions keep W3's closed gate: W3 failed its own "
        "test (-0.842 m) and no user decision is recorded. The guard therefore grants only "
        "design_time_evidence and after_the_fact_analysis, and the WP is partial until that "
        "decision. No head data was read, 0 Claude API tokens were used, there was no GPU, "
        "nothing was written under hydrophysics/, tests/ or L0, and nothing was staged in git.")
    products = [
        {"path": rel_to_repo(PROD / "v2") + "/", "layer": "L1",
         "manifest": f"{rel_to_repo(PROD / 'v2')}/MANIFEST.json (checker ok, 26 files; gate "
                     "passed=false: own criteria pass (post-look), upstream W3 failed)",
         "allowed_uses": "declared: validator, model_design (gated, refused while the gate is "
                         "closed), design_time_evidence, after_the_fact_analysis (granted); "
                         "forbidden: scoring_target, baseline, datum_fit, prospective_scorer, "
                         "calibration_target, selection, prior, prior_width_input, "
                         "forcing_frozen, forcing_recorded_arm, initial_conditions, idw_field, "
                         "statistical_layer_history",
         "fit_window_end": "2022-12"},
        {"path": rel_to_repo(PROD / "v1") + "/", "layer": "L1",
         "manifest": f"{rel_to_repo(PROD / 'v1')}/MANIFEST.json (checker ok, 23 files; "
                     "registered build of H1b.ground_height.v1; gate passed=false: upstream "
                     "W3 failed)",
         "allowed_uses": "same policy as v2; provenance of the registered acceptance",
         "fit_window_end": "2022-12"},
        {"path": rel_to_repo(PROD / "SUPERSEDED.json"), "layer": "L1 (version record)",
         "manifest": "dataprep-superseded/1: current v2, v1 superseded (registered build)",
         "allowed_uses": "provenance record", "fit_window_end": "n/a"},
        {"path": rel_to_repo(PRE), "layer": "preregistration",
         "manifest": f"sha256 {m1['parameters']['preregistration']['sha256']}; in "
                     "INDEX.jsonl (harness.prereg3)",
         "allowed_uses": "registration of H1b's acceptance", "fit_window_end": "n/a"},
    ]
    code_paths = [rel_to_repo(HERE.parent / f) for f in (
        "h1b_lib.py", "h1b_build.py", "h1b_prereg.py", "h1b_v2.py", "h1b_verify.py",
        "h1b_report.py")]
    out = {"wp": lib.WP, "status": status, "summary": summary, "acceptance": acceptance,
           "products": products, "code_paths": code_paths, "key_numbers": key_numbers,
           "deviations_from_plan": deviations, "issues_for_downstream": issues,
           "report_path": rel_to_repo(REPORTS_DIR / "H1b.md"),
           "ledger": {"numbers": ids, "unledgered": miss, "conflicts_all_fragments":
                      len(conf)},
           "looks": lk, "evidence": sorted(rel_to_repo(p) for p in EV.iterdir()),
           "manifest_sha256": {"v1": hashing.sha256_file(PROD / "v1" / "MANIFEST.json"),
                               "v2": hashing.sha256_file(PROD / "v2" / "MANIFEST.json")},
           "claude_api_tokens": 0, "gpu": "none", "version_record": ver["current"],
           "manifest_gates": {"v1": m1["gate"]["passed"], "v2": m2["gate"]["passed"]}}
    (REPORTS_DIR / "H1b.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    md = [f"# WP H1b: GroundHeight repair from the W3 DTM well points (report, "
          f"{_dt.date.today().isoformat()})", "",
          f"Status: **{status}**. {summary}", "", "## Acceptance", "",
          "| Criterion | Threshold | Measured | Pass |", "|---|---|---|---|"]
    for a in acceptance:
        md.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                  f"{'yes' if a['passed'] else '**no**'} |")
    md += ["", "Rows 1-5 are the registered gate of H1b (conjunctive; all pass). Row 7 is the "
           "release condition inherited from W3, which keeps both versions closed. Row 8 is "
           "post-look and reported only.", "", "## Key numbers", ""]
    md += [f"- {k}" for k in key_numbers]
    md += ["", "## Versions", "",
           "- `heads/L1_station_ground/v1`: the registered build (pre-registration "
           f"`{lib.PREREG_ID}`, sha256 {m1['parameters']['preregistration']['sha256'][:12]}..., "
           "written before any agency-vs-DTM residual of the scored stations was computed; "
           "`assert_can_score` verified its inputs and logged the score look).",
           "- `heads/L1_station_ground/v2`: post-look reviewed build (current in "
           "`SUPERSEDED.json`). It adds the stick-up rule (WellElevation - GroundHeight > 5 m, "
           "a metadata-only rule whose threshold lies in the observed 3.53-6.84 m gap) and "
           "refits tau with the registered rule on the 169 consistent stations.",
           "- Both: 344 rows (`sid`, `value`, `sd`, `flag`, `method`, `epoch`, `sd_status`, "
           "`product_version`, `source_ids`, raw agency value and status, DTM point and sds, "
           "agency minus DTM, `well_bottom_elev_m`, `well_elevation_m`), "
           "`calibration_blocks.csv`, `validation.json`, `PRODUCT_CARD.md`.", "",
           "## Deviations from the plan", ""]
    md += [f"- {d}" for d in deviations]
    md += ["", "## Issues for downstream", ""]
    md += [f"- {i}" for i in issues]
    md += ["", "## Looks (A1 look log)", ""]
    md += [f"- {x['time']} {x['purpose']} {x['units']} {x['test_id'] or ''}" for x in lk]
    md += ["", f"Ledger: `results/dataprep/ledger/rows/H1b.csv`, {len(ids)} numbers, "
           f"unledgered {len(miss)}, conflicts across all fragments {len(conf)}.", "",
           "## Code", ""]
    md += [f"- `{c}`" for c in code_paths]
    md += ["", "## Evidence", ""]
    md += [f"- `{e}`" for e in out["evidence"]]
    (REPORTS_DIR / "H1b.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"unledgered": miss, "conflicts": len(conf), "looks": len(lk)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
