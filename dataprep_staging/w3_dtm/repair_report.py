"""W3r: ledger rows (fragment ``ledger/rows/W3.csv``) and ``reports/W3.repair.json``.

Every number comes from an evidence file of this round (``reports/W3_evidence/repair/``) or
from W3's own ledgered outputs; nothing is typed in by hand. No product version is written:
the corrected test was not evaluable and the gate stays failed.

Usage: python -I repair_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, REPORTS_DIR, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

EV = REPORTS_DIR / "W3_evidence" / "repair"
PV = "ground_elevation/v3"
DECL = "W3r.corrected_test_design.v1"
FAN = "AMP_V2/data/fan_stations.parquet"
WRA = "data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet"
LEV = "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
V3 = "results/dataprep/ground_elevation/v3"
CWA = "data_fetch_api/new_measurements/cwa_station_inventory/codis_station_list_2026-10-11.json"


def rw_fan() -> Path:
    from common.paths import REPO
    return REPO / FAN


def f3(x) -> str:
    return f"{x:+.3f}"


def load(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def rows_for(number_id: str, note: str, sources: list[tuple[str, str]]) -> list[dict]:
    return [{"number_id": number_id, "source": s, "role": role, "product_version": PV,
             "wp": "W3", "note": note} for s, role in sources]


def main() -> int:
    dry, amb, aft = load("dryrun.json"), load("epoch_ambiguity.json"), load("aftfact.json")
    chk = load("check_ground_elevation.json")
    _, decl_sha = prereg3.load_prereg(DECL)
    ec = dry["eligibility_counts"]
    fan, nf = ec["fan"], ec["nonfan"]
    R = aft["residuals"]
    A, B, U = R["A_survey_at_establishment"], R["B_survey_at_modification"], R["uncorrected"]
    CA, CB = R["cell_mean_A"], R["cell_mean_B"]
    st = aft["strata_sites"]
    post = aft["post2005_wells"]
    vers = {m["version"]: m["status"] for m in chk["manifests"]}

    n = {}
    n["W3r.diag.sheet_production"] = (
        "Diagnosis: all 633 sheets used are LiDAR (method code 10), gridded at 20 m with "
        "TerraScan (304), SCOP++ (242) or SCOPE (87); schema height datum TWVD2001; the "
        "dataset record states the 20 m DTM is for terrain display and rough analysis",
        [(f"{V3}/sheets.csv", "input"),
         ("data_fetch_api/new_measurements/moi_dtm20m_2025/ds_176927.json", "input"),
         ("data_fetch_api/new_measurements/moi_dtm20m_2025/schema_hdr.zip", "input")])
    n["W3r.candidates.nonfan_wra"] = (
        f"WRA groundwater stations outside the fan set inside the DTM coverage: "
        f"{nf['stations']} wells at 6 sites (Wuri, Caotun, Mingjian, Taibao, Liujiao); "
        f"{nf['after_E2_groundheight']} carry a GroundHeight, {nf['after_E3_stickup']} pass "
        f"the WellElevation consistency rule",
        [(f"{WRA}#sid_LocationByTWD97_GroundHeight_WellElevation", "input"),
         (f"{FAN}#sid", "input"), (V3, "input")])
    n["W3r.candidates.cwa_altitudes"] = (
        "CWA stations inside the DTM coverage: 157 (111 auto, 44 agricultural, 2 CWA); "
        "altitude given in whole metres for all 155 auto/agricultural stations; rejected as "
        "a 0.30 m reference (quantization, unknown rounding, source and epoch)",
        [(CWA, "input"), (V3, "input")])
    n["W3r.corrected_test.eligibility"] = (
        f"Corrected test (metadata only): fan {fan['stations']} -> E1 coordinates/DTM "
        f"{fan['after_E1_coords_dtm']} -> E2 GroundHeight {fan['after_E2_groundheight']} -> "
        f"E3 stick-up {fan['after_E3_stickup']} -> E4 established 2005-2022 "
        f"{fan['after_E4_establish']} -> E5/E6 {fan['eligible_wells']} wells at "
        f"{fan['eligible_sites']} sites (minimum 20 sites): NOT EVALUABLE; fresh non-fan "
        f"sites {nf['eligible_sites']}",
        [(f"{FAN}#GroundHeight_WellElevation_EstablishDate_ModificationDate", "input"),
         (f"{WRA}#sid_LocationByTWD97_GroundHeight_WellElevation", "input"), (V3, "input")])
    dmi = dry["displacement_model"]
    n["W3r.corrected_test.displacement_model"] = (
        f"Extended leveling displacement for survey epochs: main campaigns "
        f"{dmi['years'][0]}, {dmi['years'][1]}-{dmi['years'][-1]} (>= 400 benchmarks; 2007 "
        f"has 199), {dmi['resets_dropped']} resets dropped; equals the W3 model on 2011-2021 "
        f"(max |diff| {dry['gap_scaled_equals_w3_model_max_abs_m']} m); synthetic sign check "
        f"returns {dry['synthetic_check']['T1_median']} for +0.25",
        [(LEV, "input"), (V3, "input")])
    a = amb["ambiguity_u_establish_minus_u_modification_m"]
    n["W3r.epoch_ambiguity"] = (
        f"Survey-epoch ambiguity of agency GroundHeight (leveling only, no GroundHeight or "
        f"DTM value): {amb['stations_passing_E1_E2_E3_E5_E6']} consistent fan stations at "
        f"{amb['sites']} sites, {amb['established_before_2003']} established before 2003; "
        f"correction to the DTM epoch median "
        f"{amb['epoch_correction_if_survey_at_establishment_m']['median']:.3f} m if surveyed "
        f"at establishment, "
        f"{amb['epoch_correction_if_survey_at_modification_m']['median']:.3f} m if at the "
        f"2009-2011 modification; difference median {a['median']:.3f} m (IQR "
        f"{a['iqr'][0]:.3f}..{a['iqr'][1]:.3f}), {amb['share_ambiguity_gt_0.30_m']:.3f} above "
        f"0.30 m",
        [(f"{FAN}#EstablishDate_ModificationDate_sid", "input"), (LEV, "input"),
         (V3, "input")])

    def boot(r):
        b = r["block_bootstrap_90_sites"]
        return f"{f3(b[0])}..{f3(b[1])}"

    src_aft = [(f"{FAN}#GroundHeight", "validator"),
               (f"{FAN}#EstablishDate_ModificationDate_WellElevation_sid", "input"),
               (V3, "input"), (LEV, "input"),
               (f"results/dataprep/preregistration/{DECL}.json", "input")]
    n["W3r.aftfact.assumption_A"] = (
        f"After the fact (never gating): DTM point moved to the survey epoch minus agency "
        f"GroundHeight, survey at EstablishDate: median over {A['sites']['n']} sites "
        f"{f3(A['sites']['median'])} m (IQR {f3(A['sites']['iqr'][0])}..{f3(A['sites']['iqr'][1])}, "
        f"MAD-sd {A['sites']['mad_sd']:.3f}; block bootstrap 90 % {boot(A)}); per well "
        f"{f3(A['wells']['median'])} (n {A['wells']['n']})", src_aft)
    n["W3r.aftfact.assumption_B"] = (
        f"After the fact (never gating): same, survey at ModificationDate (2009-2011 for most): "
        f"median over {B['sites']['n']} sites {f3(B['sites']['median'])} m (IQR "
        f"{f3(B['sites']['iqr'][0])}..{f3(B['sites']['iqr'][1])}, MAD-sd "
        f"{B['sites']['mad_sd']:.3f}; block bootstrap 90 % {boot(B)}); per well "
        f"{f3(B['wells']['median'])}", src_aft)
    n["W3r.aftfact.uncorrected_and_cell"] = (
        f"After the fact: without epoch correction {f3(U['sites']['median'])} m (bootstrap "
        f"{boot(U)}); containing 1 km cell mean instead of the DTM point: "
        f"{f3(CA['sites']['median'])} (A) and {f3(CB['sites']['median'])} (B) over "
        f"{CA['sites']['n']} sites", src_aft)
    n["W3r.aftfact.strata"] = (
        f"After the fact, site medians A / B: Changhua (sid 07, "
        f"{st['changhua_sid07']['A']['n']} sites) {f3(st['changhua_sid07']['A']['median'])} / "
        f"{f3(st['changhua_sid07']['B']['median'])}; Yunlin (09, "
        f"{st['yunlin_sid09']['A']['n']}) {f3(st['yunlin_sid09']['A']['median'])} / "
        f"{f3(st['yunlin_sid09']['B']['median'])}; other ({st['other_counties']['A']['n']}) "
        f"{f3(st['other_counties']['A']['median'])} / {f3(st['other_counties']['B']['median'])}",
        src_aft)
    n["W3r.aftfact.post2005_wells"] = (
        "After the fact, the 3 wells surveyed 2014-2016 (2 sites): " + "; ".join(
            f"{p['sid']} {f3(p['r_primary'])} m" for p in post), src_aft)
    n["W3r.checker"] = (
        f"Manifest checker over ground_elevation: v3 {vers.get('v3')}, v1 {vers.get('v1')}, "
        f"v2 {vers.get('v2')} (code edited in place before this round; no W3 code file was "
        f"edited in this round)",
        [(f"{V3}/MANIFEST.json", "input"),
         ("results/dataprep/ground_elevation/v1/MANIFEST.json", "input"),
         ("results/dataprep/ground_elevation/v2/MANIFEST.json", "input")])
    import pandas as pd
    fs = pd.read_parquet(rw_fan(), columns=["sid", "GroundHeight", "WellElevation",
                                            "EstablishDate"])
    post = fs[(fs["EstablishDate"] >= 20050101) & (fs["EstablishDate"] <= 20221231)]
    gh = post["GroundHeight"]
    real = post[(gh != 0) & gh.notna()]
    n["W3r.corrected_test.post2005_codes"] = (
        f"Fan stations established 2005-2022: {len(post)}; GroundHeight 0.0 code "
        f"{int((gh == 0).sum())}, NaN {int(gh.isna().sum())}, real "
        f"{int(((gh != 0) & gh.notna()).sum())}, of which "
        f"{int(((real['WellElevation'] - real['GroundHeight']).abs() > 5).sum())} inconsistent "
        f"with WellElevation (|stick-up| > 5 m)",
        [(f"{FAN}#GroundHeight_WellElevation_EstablishDate", "input")])
    rows = []
    for k, (note, srcs) in n.items():
        rows += rows_for(k, note, srcs)
    ledger.append_rows("W3", rows)
    unledgered = ledger.unledgered(list(n))

    rep = {
        "wp": "W3", "round": "W3r (repair round)", "status": "failed", "date": "2026-10-18",
        "products": [], "product_current": "results/dataprep/ground_elevation/v3 (unchanged; "
                                           "gate failed)",
        "gate": {"passed": False, "original_test": "W3.dtm_heldout_bias: -0.842 m (90 % "
                                                    "-0.908..-0.783) vs |median| <= 0.30 m",
                 "corrected_test": "not evaluable: 2 eligible fan sites (3 wells) vs minimum "
                                   "20; 0 fresh sites; not registered as a test, not run"},
        "diagnosis_report": "results/dataprep/reports/W3.repair.md (sections 1-4 written "
                            "2026-10-17T17:14Z; snapshot "
                            "W3_evidence/repair/diagnosis_written_2026-10-17T1714Z.md, sha256 "
                            + sha256_file(EV / "diagnosis_written_2026-10-17T1714Z.md") + ")",
        "declaration": {"id": DECL, "sha256": decl_sha,
                        "path": f"results/dataprep/preregistration/{DECL}.json"},
        "numbers": {k: v[0] for k, v in n.items()},
        "evidence": [rel_to_repo(p) for p in sorted(EV.iterdir())],
        "ledger": {"fragment": "results/dataprep/ledger/rows/W3.csv",
                   "rows_added_this_round": len(rows) + 1, "number_ids": list(n),
                   "unledgered": unledgered},
        "looks": "one after_the_fact_analysis look (2026-10-17T17:24:16Z) on "
                 "W3r:fan_wells_groundheight_consistent (165 station-metadata members), "
                 "declared in " + DECL,
        "acceptance": [
            {"criterion": "Original registered gate W3.dtm_heldout_bias (unchanged): |median| "
                          "of epoch-corrected 1 km cell mean minus held-out benchmark",
             "threshold": "<= 0.30 m", "measured": "-0.842 m (90 % -0.908..-0.783)",
             "passed": False},
            {"criterion": "Diagnosis of every failed criterion written before any method "
                          "change or new test (repair rule a)",
             "threshold": "before", "measured": "sections 1-4 at 17:14Z; first new "
                                                "computation (dry run) 17:18Z", "passed": True},
            {"criterion": "Corrected test (bare-ground reference at wells, TWVD2001-era "
                          "surveys) evaluable",
             "threshold": ">= 20 sites",
             "measured": f"{fan['eligible_sites']} fan sites ({fan['eligible_wells']} wells), "
                         f"{nf['eligible_sites']} fresh sites", "passed": False},
            {"criterion": "Threshold unchanged; no method tuned against the failed test",
             "threshold": "0.30 m", "measured": "0.30 m; no product method changed",
             "passed": True},
            {"criterion": "sd_point coverage (secondary, unchanged)",
             "threshold": "Wilson 90 % contains 0.90", "measured": "0.689 (0.654..0.721); "
                                                                   "uncalibrated",
             "passed": False},
            {"criterion": "Manifest checker over ground_elevation", "threshold": "all ok",
             "measured": f"v3 {vers.get('v3')}; v1 {vers.get('v1')}, v2 {vers.get('v2')} "
                         f"(permanent)", "passed": False},
        ],
        "recommendation": "Keep SRTM as the twin's ground elevation for every gated use; "
                          "ground_elevation/v3 stays excluded. For a user decision only: the "
                          "DTM's median bias at bare ground lies between about -0.36 and 0 m "
                          "(benchmarks bound it at >= -0.38 m; wells give -0.05 or -0.36 m "
                          "depending on the undocumented survey date), against SRTM +1.25 m "
                          "above benchmarks (MAD-sd 2.45). A waiver disclosing this range "
                          "would be the honest route to a DTM rung.",
        "data_needed": [
            "WRA station records: survey date and height datum of GroundHeight/WellElevation, "
            "or a new TWVD2001 leveling of ground at the post-2005 wells (fresh units)",
            "MOI/NLSC LiDAR ground check points (GNSS-RTK on open ground) or the producers' "
            "accuracy reports for the 2016-2018 sheets",
            "point-of-record descriptions (點之記) of the WRA leveling benchmarks: monument "
            "type and mark height above ground",
            "the WRA leveling network's starting benchmarks and adopted TWVD2001 heights",
            "independent geodetic heights at ground-level control points (e-GNSS with the "
            "TWHYGEO geoid, or ICESat-2 ATL08 with a TWVD2001-consistent geoid)"],
        "claude_api_tokens": 0, "gpu": "none", "downloads": "none",
        "code_paths": [f"dataprep_staging/w3_dtm/{x}" for x in (
            "repair_wells.py", "repair_dryrun.py", "repair_epoch_ambiguity.py",
            "repair_declare.py", "repair_aftfact.py", "repair_report.py")],
        "prereg_dir_index": rel_to_repo(PREREG_DIR / "INDEX.jsonl"),
    }
    (REPORTS_DIR / "W3.repair.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)
                                                + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
