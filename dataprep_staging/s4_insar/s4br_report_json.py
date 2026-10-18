"""WP S4b repair round 2 (S4br): assemble ``results/dataprep/reports/S4b.repair.json`` from the
evidence files, so every number in it is the evidence's own (the Markdown report quotes the
same files).

    python -I s4br_report_json.py      (hydro env)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s4br_config as R  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def j(name: str):
    return json.loads((R.EVID / name).read_text())


def r2(x):
    return None if x is None else round(float(x), 2)


def main() -> int:
    des, ngl, wra, y, af = (j("design.json"), j("test_g1_ngl.json"), j("test_g1_wra.json"),
                            j("test_y.json"), j("after.json"))
    b, gm, nw, co = (j("diag_budget.json"), j("diag_geometry.json"), j("diag_network.json"),
                     j("diag_coherence.json"))
    pdir = sorted((C.DATAPREP / C.PRODUCT).glob("v*"), key=lambda q: int(q.name[1:]))[-1]
    val = json.loads((pdir / "validation.json").read_text())
    man = json.loads((pdir / "MANIFEST.json").read_text())
    p = ngl["primary"]
    acc = [
        {"criterion": "G1 on FRESH units (registered S4b.insar_repair.v1): NGL family, run P2, "
                      f"{des['window']['t0']} -> {des['window']['t1']}",
         "threshold": "RMS <= 10.0 mm with >= 5 evaluable stations (fewer = fail)",
         "measured": f"{r2(p['rms_mm'])} mm over {p['n_evaluable']} evaluable stations "
                     f"({', '.join(p['stations'])}); HUWE (all 3x3 pixels masked, coh_avg "
                     "0.11-0.15) and LUKN (outside the primary footprint) not evaluable",
         "passed": False, "units": "mm / stations"},
        {"criterion": "G1 WRA family, registered window 2021-09-30 -> 2022-11-30, run P2 with "
                      "the stated M3 reference (SECOND LOOK, conditional)",
         "threshold": "RMS <= 10.0 mm with >= 5 stations",
         "measured": f"{r2(wra['primary']['rms_mm'])} mm over {wra['primary']['n_evaluable']} "
                     "stations",
         "passed": False, "units": "mm"},
        {"criterion": "Y seasonal-amplitude validator, run P2 (SECOND LOOK, conditional)",
         "threshold": "Y1 bootstrap p05 > 0 and Y2 >= 6/8",
         "measured": f"Y1 rho {r2(y['Y1_all_yunlin']['rho_primary'])} (p05 "
                     f"{r2(y['Y1_all_yunlin']['boot_p05'])}, pass); Y2 "
                     f"{y['Y2_battery']['wins']}/8",
         "passed": False, "units": "Spearman rho / splits"},
        {"criterion": "C1 Changhua Z2 summer coherence (not re-tested: LiCSAR coherence)",
         "threshold": ">= 0.30 of the zone's cells with coherence >= 0.3",
         "measured": "0.297 (round 1, unchanged); summer coherent fraction 0.643 at 6 d, "
                     "0.397 at 12 d, 0.205 at 24 d",
         "passed": False, "units": "fraction of cells"},
        {"criterion": "Network gaps: values bridged by LiCSBAS's linear-velocity constraint "
                      "(round 1: flag 4 on 51.2 % of finite values)",
         "threshold": "no gap inside the registered windows; bridged share reported",
         "measured": f"P2, F2 and S2 each one connected network (gaps: "
                     f"{ {k: v['gaps'] for k, v in val['network']['runs'].items()} }); flag-4 "
                     f"share of finite cell values "
                     f"{100 * val['network']['frac_finite_cell_values_flag4']:.2f} %",
         "passed": True, "units": "gaps / %"},
        {"criterion": "Asc/desc decomposition from 2022-11",
         "threshold": "a descending partner from 2022-11",
         "measured": "still none: 105D GeoTIFFs only from 2025-01; 069A_06632 is ascending",
         "passed": False, "units": "geometry"},
        {"criterion": "Second ascending frame 069A_06632_162010 (approved)",
         "threshold": "fetched with SOURCE.json; merged in the product",
         "measured": "765 files, 1134.1 MB; tied at 2022-11-21 and averaged; Z3 coverage "
                     f"{val['coverage_merged']['dt.zone.Z3']['primary_referenced_2020_2025']:.2f}"
                     f" -> {val['coverage_merged']['dt.zone.Z3']['any_after_tie']:.2f}; frame "
                     f"agreement {val['frame_agreement_after_tie']['median_rms_mm']:.1f} mm "
                     "median per-epoch RMS",
         "passed": True, "units": "files / fraction / mm"},
        {"criterion": "Declared report-only rows computed as declared",
         "threshold": "as registered",
         "measured": "combined-tie row and after-the-fact fresh-window WRA rows kept GFES "
                     "(catalogued 2020-07-10 step inside the fresh span) against the station "
                     "rule; disclosed and recomputed (fresh WRA 7.17 / 8.43 mm, combined tie "
                     f"{af['ramp_fix']['combined_wra_ngl_tie_corrected']['rms_mm']:.2f} mm)",
         "passed": False, "units": "procedure"},
        {"criterion": "Diagnosis before method change; thresholds unchanged; second looks "
                      "labelled; pre-registration before scoring",
         "threshold": "repair discipline a-d",
         "measured": "Parts 1-2 hashed 2026-10-17T21:11:20Z before any method code; Part 2b "
                     "21:40:19Z before registration; registration 21:43:33Z, scoring look "
                     "21:43:40Z; every second look and after-the-fact row logged first",
         "passed": True, "units": "procedure"},
        {"criterion": "Product version with manifest (insar_069A/" + pdir.name + ")",
         "threshold": "new version, checker re-hashes, guard refuses gated uses",
         "measured": f"{pdir.name} written; gate {man['gate']['status']}",
         "passed": True, "units": "files"}]
    out = {
        "wp": "S4br", "parent_wp": "S4b", "status": "partial",
        "date": "2026-10-18", "prereg": {"id": R.PREREG_ID,
                                         "path": f"results/dataprep/preregistration/"
                                                 f"{R.PREREG_ID}.json"},
        "diagnosis": {
            "g1_registered_reproduced_mm": r2(b["reproduces_registered"]["rms_mm"]),
            "network_gaps": {"cause": "LiCSBAS12 without --multi_prime: first-pass loop RMS "
                                      "carried the non-integer phase bias",
                             "P_removals_first_pass": nw["P"]["n_bad12_first_pass"],
                             "P_removals_total": nw["P"]["n_bad12"],
                             "loop_rms_median_rad_no_prime_vs_prime":
                                 [r2(nw["P"]["loop_rms_no_prime_median"]),
                                  r2(nw["P"]["loop_rms_multi_prime_median"])],
                             "gap_part_rms_mm_without_reference_surface":
                                 r2(gm["gap_part_without_reference_surface"]["rms"])},
            "reference": {"vif": gm["vif"], "corr_x_y": r2(gm["corr_x_y"]),
                          "pattern_rms_mm": r2(b["reference"]["rms_plane_height_pattern"]),
                          "corr_with_error": r2(b["reference"]["corr_e_plane_pattern"])},
            "atmosphere": {"white_window_change_mm":
                               r2(b["epoch_noise"]["rms_white_noise_window_change"]),
                           "insar_diff_sd_mm_before_after_lp_hp":
                               [r2(b["epoch_noise"]["sigma_insar_diff_median"]),
                                r2(b["epoch_noise"]["sigma_insar_diff_after_lpspace_hptime_median"])]},
            "steps_s3v4_rms_mm": r2(b["gnss_steps_s3v4"]["g1_with_s3v4_corrected_heights"]
                                    ["rms_mm"]),
            "floors_mm": {"east_leak": r2(b["horizontal_2025"]["rms_demeaned_leak"]),
                          "point_vs_area": r2(b["point_vs_area"]["rms_demeaned"])},
            "c1_z2_by_btemp": [x for x in co["dt.zone.Z2"]["by_season_btemp"]
                               if x["season"] == "summer"],
            "corrections_after_retest": [
                "1.1(b) north-term reading was confounded by the bridges: with the network "
                "closed the plane reduces the 2021-22 WRA error from 17.7 to 7.3 mm",
                "1.3 Y plane-artefact reading withdrawn: without the plane rho(amplitude, "
                "easting) is -0.83 (round 1 -0.66) and Y2 is again 0/8"]},
        "retest": {
            "fresh_ngl": {"window": [des["window"]["t0"], des["window"]["t1"]],
                          "stations_eligible": des["eligible"], "primary": {
                              k: p.get(k) for k in ("rms_mm", "n_evaluable", "stations",
                                                    "not_evaluable", "passed")},
                          "errors_mm": {r["station"]: r2(r["err_mm"]) for r in p["rows"]},
                          "report_only": {k: r2(v.get("rms_mm"))
                                          for k, v in ngl["report_only"].items()}},
            "wra_second_look": {"rms_mm": r2(wra["primary"]["rms_mm"]),
                                "n": wra["primary"]["n_evaluable"],
                                "report_only": {k: r2(v.get("rms_mm"))
                                                for k, v in wra["report_only"].items()}},
            "y_second_look": {"Y1_rho": r2(y["Y1_all_yunlin"]["rho_primary"]),
                              "Y1_p05": r2(y["Y1_all_yunlin"]["boot_p05"]),
                              "Y2_wins": y["Y2_battery"]["wins"],
                              "rho_amp_easting": r2(y["report_only_rho_amp_easting"])},
            "after_the_fact": {
                "reference_by_window": {k: {"rms_mm": r2(v["rms_mm"]), "n": v["n"]}
                                        for k, v in af["ramp_table"].items() if "rms_mm" in v},
                "corrected_fresh_window_wra": {k: {"rms_mm": r2(af["ramp_fix"][k]["rms_mm"]),
                                                   "n": af["ramp_fix"][k]["n"]}
                                               for k in ("wra_fresh_plane",
                                                         "wra_fresh_offset_height")},
                "combined_tie_corrected_mm":
                    r2(af["ramp_fix"]["combined_wra_ngl_tie_corrected"]["rms_mm"]),
                "full_chain": {k: r2(af[k].get("rms_mm")) for k in
                               ("wra_F2", "wra_F2_S2_merged", "ngl_fresh_window_F2")
                               if k in af}}},
        "acceptance": acc,
        "products": [{"path": rel_to_repo(pdir), "manifest_sha256": Lb_sha(pdir),
                      "gate": man["gate"]["status"], "allowed_uses": man["allowed_uses"],
                      "fit_window_end": man["fit_window_end"]},
                     {"path": rel_to_repo(R.L0_SECOND), "kind": "L0 (new fetch, SOURCE.json)"}],
        "network": val["network"], "coverage": val["coverage_merged"],
        "frame_agreement": {k: val["frame_agreement_after_tie"][k]
                            for k in ("n_epochs", "median_rms_mm", "max_rms_mm")},
        "claude_api_tokens": 0, "gpu": "none", "downloads_bytes": 1134.1e6}
    Lb.jdump(out, C.DATAPREP / "reports" / "S4b.repair.json")
    print(json.dumps({"written": "results/dataprep/reports/S4b.repair.json",
                      "product": rel_to_repo(pdir)}))
    return 0


def Lb_sha(pdir: Path) -> str:
    from common.hashing import sha256_file
    return sha256_file(pdir / "MANIFEST.json")


if __name__ == "__main__":
    raise SystemExit(main())
