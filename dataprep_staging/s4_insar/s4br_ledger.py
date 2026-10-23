"""WP S4b repair round 2 (S4br): ledger rows (plan 4.1, U6) for every number the repair report
quotes. Appends to ``results/dataprep/ledger/rows/S4b.csv`` through ``common.ledger``.

Roles as in round 1: LiCSAR L0 subsets, LiCSBAS run files, geometry, the units registry, the
battery and the thickness proxy are inputs; GNSS heights (WRA and NGL L0; S3 gnss_clean v4 for
its after-the-fact rows) are validators of the InSAR numbers that involve them; the census
product is the validator of the amplitude-density numbers.

    python -I s4br_ledger.py      (hydro env)
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
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

L0_A = "data_fetch_api/new_measurements/licsar_069A_06694/fetch_2026-10-12_s4a"
L0_S = rel_to_repo(R.L0_SECOND)
S4A = "insar_licsar_subsets/v1"
PRE1 = f"results/dataprep/preregistration/{C.PREREG_ID}.json"
PRE2 = f"results/dataprep/preregistration/{R.PREREG_ID}.json"
CENSUS = "results/dataprep/pump_electricity/preorigin/v1"
S3V4 = "results/dataprep/gnss_clean/full/v4"
EVR = rel_to_repo(R.EVID)


def j(name: str):
    return json.loads((R.EVID / name).read_text())


def run_files(name: str, round2: bool = True) -> list[str]:
    base = R.RUN_DIR if round2 else C.RUN_DIR
    ts = base / name / "TS_GEOCml2"
    out = [rel_to_repo(ts / "cum.h5"), rel_to_repo(ts / "results" / "mask"),
           rel_to_repo(ts / "info" / "12network_gap_info.txt")]
    if round2:
        out.append(rel_to_repo(ts / R.CUM_FILE))
    return out


def rows(number_id: str, note: str, inputs, validators=(), pv: str = "insar_069A/v3"):
    out = [{"number_id": number_id, "source": s, "role": "input", "product_version": pv,
            "note": note} for s in dict.fromkeys(inputs)]
    out += [{"number_id": number_id, "source": s, "role": "validator", "product_version": pv,
             "note": note} for s in dict.fromkeys(validators)]
    return out


def main() -> int:
    wra = {k: rel_to_repo(v) for k, v in Lb.wra_files().items()}
    des = j("design.json")
    ngl = {s: rel_to_repo(next(d / f"{s}.tenv3" for d in R.NGL_DIRS
                               if (d / f"{s}.tenv3").is_file())) for s in des["eligible"]}
    ngl_meta = [rel_to_repo(R.NGL_HOLDINGS), rel_to_repo(R.NGL_STEPS)]
    geomA = rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz")
    geomS = rel_to_repo(R.RUN_DIR / f"geom_{R.SECOND}.npz")
    inv = rel_to_repo(C.RUN_DIR / "gnss_inventory.json")
    g1_st = ["ANES", "CHSG", "FRES", "GFES", "HLES", "HNES", "JJES", "JMES", "KTES", "LYES",
             "NGES", "NTUH", "SLES", "SNES", "SSES", "STES", "YCES", "YWJS"]
    wra_v = [wra[s] for s in g1_st]
    ngl_v = [ngl[s] for s in des["eligible"]]
    base1 = [L0_A, S4A, PRE1, geomA, inv] + run_files("P", round2=False)
    base2 = [L0_A, S4A, PRE2, geomA, inv, f"{EVR}/design.json"] + run_files("P2")
    out = []
    # ---- diagnosis (round-1 runs)
    b = j("diag_budget.json")
    gm = j("diag_geometry.json")
    nw = j("diag_network.json")
    co = j("diag_coherence.json")
    yc = j("diag_yconf.json")
    s, e = b["split"], b["epoch_noise"]
    out += rows("S4b.r2.diag.g1_reproduced", f"registered G1 reproduced {b['reproduces_registered']['rms_mm']:.3f} mm, max diff 0.0",
                base1 + [f"{EVR}/diag_budget.json"], wra_v, pv="insar_069A/v2")
    out += rows("S4b.r2.diag.gap_part",
                f"gap-interval residual increments RMS {s['rms_gap_part']:.1f} mm; without the "
                f"reference surface {gm['gap_part_without_reference_surface']['rms']:.1f} mm "
                f"(corr northing {gm['gap_part_without_reference_surface']['corr_northing']:.2f}, "
                f"with error {gm['gap_part_without_reference_surface']['corr_with_err']:.2f}); "
                f"reference part {gm['gap_part_reference_surface_mm']['rms']:.1f} mm",
                base1 + [f"{EVR}/diag_budget.json", f"{EVR}/diag_geometry.json"], wra_v,
                pv="insar_069A/v2")
    out += rows("S4b.r2.diag.epoch_noise",
                f"sigma resid {e['median_sigma_resid']:.1f} mm, white window change "
                f"{e['rms_white_noise_window_change']:.1f} mm, InSAR-only "
                f"{e['sigma_insar_diff_median']:.1f} -> {e['sigma_insar_diff_after_lpspace_hptime_median']:.1f} mm, "
                f"smooth HP change {e['rms_window_change_of_smooth_hp_part']:.1f} mm",
                base1 + [f"{EVR}/diag_budget.json"], wra_v, pv="insar_069A/v2")
    for r in ("P", "F"):
        x = nw[r]
        out += rows(f"S4b.r2.diag.loops_run{r}",
                    f"step-12 removals {x['n_bad12']} (first pass {x['n_bad12_first_pass']}); "
                    f"loop RMS median {x['loop_rms_no_prime_median']:.2f} / "
                    f"{x['loop_rms_multi_prime_median']:.2f} rad; >= 1.5 rad "
                    f"{x['frac_loops_ge_1p5_no_prime']:.2f} / {x['frac_loops_ge_1p5_multi_prime']:.2f}; "
                    f"recomputed first pass {x['first_pass_bad_recomputed']}",
                    [L0_A, S4A, geomA, f"{EVR}/diag_network.json"] + run_files(r, False)
                    + [rel_to_repo(C.RUN_DIR / r / "GEOCml2")], pv="insar_069A/v2")
    out += rows("S4b.r2.diag.reference_geometry",
                f"VIF {gm['vif']}; corr x-y {gm['corr_x_y']:.2f}; pattern RMS "
                f"{b['reference']['rms_plane_height_pattern']:.1f} mm (east "
                f"{gm['parts']['east']['rms_mm']:.1f}, north {gm['parts']['north']['rms_mm']:.1f}); "
                f"coef scatter {b['reference']['coef_east_detrended_rsd_mm_per_km']:.2f}/"
                f"{b['reference']['coef_north_detrended_rsd_mm_per_km']:.2f} mm/km",
                base1 + [f"{EVR}/diag_budget.json", f"{EVR}/diag_geometry.json"], wra_v,
                pv="insar_069A/v2")
    out += rows("S4b.r2.diag.unwrapping",
                f"{b['unwrapping']['total_increments_gt_25mm']} increments > 25 mm",
                base1 + [f"{EVR}/diag_budget.json"], wra_v, pv="insar_069A/v2")
    out += rows("S4b.r2.diag.steps_s3v4",
                f"G1 with S3 v4 heights {b['gnss_steps_s3v4']['g1_with_s3v4_corrected_heights']['rms_mm']:.1f} mm; "
                "1 step in span (KTES 2022-08-26)", base1 + [f"{EVR}/diag_budget.json"], [S3V4],
                pv="insar_069A/v2")
    out += rows("S4b.r2.diag.floors",
                f"leak {b['horizontal_2025']['rms_demeaned_leak']:.1f} mm; point vs area "
                f"{b['point_vs_area']['rms_demeaned']:.1f} mm",
                base1 + [f"{EVR}/diag_budget.json", "results/dataprep/insar_069A/v2"],
                pv="insar_069A/v2")
    out += rows("S4b.r2.diag.c1_coherence",
                "Z2 summer coherent fraction by baseline: " + ", ".join(
                    f"{r_['btemp_days']} d {r_['median_frac_coherent']:.3f}" for r_ in
                    co["dt.zone.Z2"]["by_season_btemp"] if r_["season"] == "summer"),
                [L0_A, S4A, f"{EVR}/diag_coherence.json", rel_to_repo(C.UNITS_V3),
                 rel_to_repo(C.RUN_DIR / "coh_P_all_preorigin.npz")], pv="insar_069A/v2")
    out += rows("S4b.r2.diag.yconf",
                f"injected annual amplitude median {yc['injected_amp_mm_vertical_p10_p50_p90'][1]:.1f} mm; "
                f"Spearman with amplitude {yc['spearman_injected_vs_amplitude']:.2f}",
                base1 + [f"{EVR}/diag_yconf.json", "results/dataprep/insar_069A/v2",
                         rel_to_repo(C.UNITS_V3)], pv="insar_069A/v2")
    # ---- repair: fetch, runs, masks
    out += rows("S4b.r2.fetch_069A_06632", "765 files (380 pairs), 1134.1 MB transferred",
                [L0_S, rel_to_repo(R.L0_SECOND / "SOURCE.json")])
    af = j("after.json")
    for r in ("P2", "F2", "S2"):
        x = af["network"][r]
        out += rows(f"S4b.r2.network.{r}",
                    f"{x['n_ifg_used']}/{x['n_ifg_all']} ifgs, {x['n_epochs']} epochs, gaps "
                    f"{x['network_gaps']}, active pixels kept {x['frac_active_pixels_kept']:.3f}",
                    [L0_S if r == "S2" else L0_A, geomS if r == "S2" else geomA,
                     f"{EVR}/after.json"] + run_files(r))
    # ---- registered fresh test and second looks
    t = j("test_g1_ngl.json")
    p = t["primary"]
    out += rows("S4b.r2.G1_ngl_fresh",
                f"RMS {p['rms_mm']:.2f} mm over {p['n_evaluable']} evaluable (>= 5 needed): FAIL",
                base2 + ngl_meta + [f"{EVR}/test_g1_ngl.json"], ngl_v)
    for k, v in t["report_only"].items():
        vv = wra_v if k == "combined_wra_ngl_tie" else []
        out += rows(f"S4b.r2.G1_ngl.report_only.{k}", f"RMS {v.get('rms_mm')}",
                    base2 + ngl_meta + [f"{EVR}/test_g1_ngl.json"], ngl_v + vv)
    w = j("test_g1_wra.json")
    out += rows("S4b.r2.G1_wra_second_look", f"RMS {w['primary']['rms_mm']:.2f} mm, 18 stations",
                base2 + [f"{EVR}/test_g1_wra.json"], wra_v)
    for k, v in w["report_only"].items():
        vv = [S3V4] if k == "s3v4_step_corrected_heights" else wra_v
        out += rows(f"S4b.r2.G1_wra.report_only.{k}", f"RMS {v.get('rms_mm')}",
                    base2 + [f"{EVR}/test_g1_wra.json"], vv)
    y = j("test_y.json")
    out += rows("S4b.r2.Y_second_look",
                f"Y1 rho {y['Y1_all_yunlin']['rho_primary']:.3f} p05 "
                f"{y['Y1_all_yunlin']['boot_p05']:.3f}; Y2 {y['Y2_battery']['wins']}/8; rho "
                f"easting {y['report_only_rho_amp_easting']:.3f}",
                base2 + [f"{EVR}/test_y.json", rel_to_repo(C.UNITS_V3),
                         rel_to_repo(C.RUN_DIR / "battery.json"),
                         "results/twin_runs/stage3_datum_gate/coupled_leveling/"
                         "vep_zonal_leveling.json"], [CENSUS])
    rt = af["ramp_table"]
    for k, v in rt.items():
        if "rms_mm" not in v:
            continue
        vv = wra_v if k.startswith("wra") else ngl_v
        out += rows(f"S4b.r2.after.ramp.{k}", f"RMS {v['rms_mm']} n {v['n']} (after the fact)",
                    base2 + ngl_meta + [f"{EVR}/after.json"], vv)
    out += rows("S4b.r2.after.ramp.plane_change", json.dumps(
        rt["plane_coef_change_w1_minus_w0_mm_per_km_los"]),
        base2 + [f"{EVR}/after.json"])
    rf = af["ramp_fix"]
    wra_f = [wra[s] for s in rf["wra_station_set_fresh_window"]]
    for k in ("wra_fresh_plane", "wra_fresh_offset_height"):
        out += rows(f"S4b.r2.after.ramp_fix.{k}",
                    f"RMS {rf[k]['rms_mm']:.2f} n {rf[k]['n']} (registered WRA station rule "
                    "applied to the fresh window; GFES excluded; after the fact)",
                    base2 + [f"{EVR}/after.json"], wra_f)
    ct = rf["combined_wra_ngl_tie_corrected"]
    out += rows("S4b.r2.after.ramp_fix.combined_tie",
                f"RMS {ct['rms_mm']:.2f} at {ct['n_evaluable_ngl']} NGL stations, "
                f"{ct['n_tie_stations']} tie stations (after the fact)",
                base2 + ngl_meta + [f"{EVR}/after.json"], ngl_v + wra_f)
    for k in ("wra_F2", "wra_F2_S2_merged", "ngl_fresh_window_F2"):
        if k in af:
            vv = ngl_v if k.startswith("ngl") else wra_v
            out += rows(f"S4b.r2.after.{k}", f"RMS {af[k].get('rms_mm')} n "
                        f"{af[k].get('n_evaluable')} (after the fact)",
                        [L0_A, L0_S, S4A, PRE2, geomA, geomS, inv, f"{EVR}/after.json"]
                        + run_files("F2") + run_files("S2") + ngl_meta, vv)
    # ---- product
    pdir = sorted((C.DATAPREP / C.PRODUCT).glob("v*"), key=lambda q: int(q.name[1:]))[-1]
    if pdir.name != "v2":
        val = json.loads((pdir / "validation.json").read_text())
        net = val["network"]
        out += rows("S4b.r2.product.flags",
                    f"flag counts {net['flag_counts_vertical_cells']}; flag-4 share of finite "
                    f"values {net['frac_finite_cell_values_flag4']:.4f}",
                    [rel_to_repo(pdir / "MANIFEST.json"), L0_A, L0_S] + run_files("F2")
                    + run_files("S2"), pv=f"{C.PRODUCT}/{pdir.name}")
        out += rows("S4b.r2.product.coverage", json.dumps(val["coverage_merged"]),
                    [rel_to_repo(pdir / "MANIFEST.json"), rel_to_repo(C.UNITS_V3)]
                    + run_files("F2") + run_files("S2"), pv=f"{C.PRODUCT}/{pdir.name}")
        fa = val["frame_agreement_after_tie"]
        out += rows("S4b.r2.product.frame_agreement",
                    f"median per-epoch RMS {fa['median_rms_mm']} mm over {fa['n_epochs']} "
                    "epochs", [rel_to_repo(pdir / "MANIFEST.json")] + run_files("F2")
                    + run_files("S2"), pv=f"{C.PRODUCT}/{pdir.name}")
    p = ledger.append_rows(R.WP, out)
    ids = sorted({r["number_id"] for r in out})
    print(json.dumps({"rows_appended": len(out), "numbers": len(ids), "fragment": str(p)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
