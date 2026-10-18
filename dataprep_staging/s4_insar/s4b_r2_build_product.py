"""WP S4b repair round: build ``results/dataprep/insar_069A/v<N>/`` (v2; never an overwrite) from
v1 plus the LiCSBAS network-gap structure.

The displacement values, sd, reference table, coherence, amplitude and decomposition arrays are
v1's, unchanged (v1's hashes are verified first). What changes:

* flags: a cell value whose offset from the first epoch includes at least one increment that
  LiCSBAS13 bridged with its linear-velocity constraint (no interferogram spans the interval
  at one or more of the cell's pixels) carries flag 4 (plan 2.4: filled from the series' own
  temporal model) instead of 0;
* ``segment`` (A, T): the cell's network segment per epoch; a change between two epochs is
  interferometric only when both share the segment id. ``vertical_ml.npz`` gets the same per
  pixel (``pix_segment``) and LiCSBAS's ``n_gap``;
* ``network_segments.json``: gaps and segments of runs P, F, D and where they cut the
  registered windows;
* ``gnss_tie_offsets.csv`` is dropped (a GNSS-derived file that spends the G1 stations has no
  place inside a validator product; its computation is now disclosed in the look log);
* ``fit_window_end`` is 2026-01: run D (behind ``decomposition_2025.npz``) uses pairs up to
  2026-01-30;
* validation.json gains a ``repair_round`` section; PRODUCT_CARD.md is rewritten.

    python -I s4b_r2_build_product.py     (hydro env, one CPU slot)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402
from common import manifest  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

V1 = C.DATAPREP / C.PRODUCT / "v1"
CODE = ["s4b_config.py", "s4b_lib.py", "s4b_evaluate.py", "s4b_r2_looks.py", "s4b_r2_gaps.py",
        "s4b_r2_diag.py", "s4b_r2_split.py", "s4b_r2_build_product.py"]
PREORIGIN_USES = ["calibration_target", "datum_fit", "baseline", "initial_conditions",
                  "idw_field", "prior", "prior_width_input", "model_design", "selection",
                  "forcing_frozen", "design_time_evidence"]
FLAG_NOTE = ("0 interferometric: same network segment as the first epoch (2020-01-12) at every "
             "contributing pixel; 4 value includes >= 1 increment bridged by LiCSBAS13's "
             "linear-velocity (gamma) constraint (plan 2.4 flag 4); 8 missing; 9 epoch whose "
             "stable set has < 200 pixels (none). A CHANGE between two epochs is "
             "interferometric only when both have the same segment id; across segments it "
             "includes the bridged offset whatever the flags")


def j(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    m1, _ = manifest.load_manifest(V1)
    probs = manifest.verify_hashes(m1, V1)
    if probs:
        raise SystemExit(f"v1 hashes do not reproduce: {probs}")
    gz = C.RUN_DIR / "gaps_F.npz"
    G = np.load(gz, allow_pickle=False)
    gaps = j(C.EVID / "repair_gaps.json")
    diag = j(C.EVID / "repair_diag.json")
    split = j(C.EVID / "repair_pf_split.json")
    d = manifest.new_version_dir(C.PRODUCT)

    # ---- vertical_cells.npz
    z = np.load(V1 / "vertical_cells.npz", allow_pickle=False)
    vc = {k: z[k] for k in z.files}
    dates = [str(x) for x in vc["dates"]]
    assert [x.replace("-", "") for x in dates] == [str(x) for x in G["dates"]], "epochs differ"
    val = vc["value"]                                         # (A, T)
    seg = G["seg_cell"].astype(np.int16)                      # (A, T), -1 no data
    fin = np.isfinite(val)
    assert np.all(seg[fin] >= 0), "a finite value lies in a cell without gap structure"
    assert set(np.unique(vc["flag"])) <= {0, 8}, "v1 flags were 0/8 only"
    flag = np.where(fin, np.where(seg > 0, 4, 0), 8).astype(np.uint8)
    flag = np.where(vc["flag"] == 9, 9, flag).astype(np.uint8)
    seg_out = np.where(fin, seg, -1).astype(np.int16)
    vc.update(flag=flag, segment=seg_out, cell_gap=G["cell_gap"],
              frac_pix_gap=G["frac_pix_gap"].astype(np.float16),
              network_segment=G["seg_net"].astype(np.int16),
              network_gaps=np.array(["_".join(g) for g in gaps["F"]["gaps"]]),
              flag_note=np.array(FLAG_NOTE),
              segment_note=np.array("segment id per (cell, epoch): number of gap intervals "
                                    "before the epoch at which ANY contributing pixel has no "
                                    "spanning interferogram; cell_gap (A, T-1) marks them, "
                                    "frac_pix_gap gives the pixel fraction"))
    np.savez_compressed(d / "vertical_cells.npz", **vc)

    # ---- vertical_ml.npz
    z = np.load(V1 / "vertical_ml.npz", allow_pickle=False)
    vm = {k: z[k] for k in z.files}
    assert np.array_equal(vm["licsbas15_mask"], G["licsbas15_mask"])
    vm.update(pix_segment=G["pix_seg"], n_gap=G["n_gap_pix"],
              segment_note=np.array("pix_segment (T, L, W): LiCSBAS13 gap intervals "
                                    "(cum.h5 'gap') accumulated over time; -1 outside the "
                                    "LiCSBAS15 mask; a change is interferometric only within "
                                    "one segment"))
    np.savez_compressed(d / "vertical_ml.npz", **vm)

    # ---- reference_epochs.csv (+ network segment)
    ref = pd.read_csv(V1 / "reference_epochs.csv")
    assert list(ref["date"]) == dates
    ref["network_segment"] = G["seg_net"]
    ref.to_csv(d / "reference_epochs.csv", index=False)

    # ---- unchanged copies
    for f in ("coherence_summer_preorigin.npz", "zone_coverage.json", "decomposition_2025.npz"):
        shutil.copyfile(V1 / f, d / f)

    # ---- seasonal amplitude (run P): v1 arrays + status and gap notes
    z = np.load(V1 / "seasonal_amplitude_preorigin.npz", allow_pickle=False)
    sa = {k: z[k] for k in z.files}
    yc = diag["yconf"]
    sa.update(status=np.array(
        "NOT a validator: registered test Y failed (Y2 0/8). The test cannot separate local "
        "pumping from east-west structure: both thickness proxies are functions of easting, "
        f"rho(amplitude, easting) {yc['rho_amp_easting']:.2f}, and the run-P reference "
        "plane's east coefficient has an annual cycle; no physical reading is offered"),
        network_gaps_runP=np.array(["_".join(g) for g in gaps["P"]["gaps"]]),
        gap_note=np.array("the harmonic fit spans run P's two summer-2022 gaps, whose "
                          "increments are bridged by LiCSBAS13's linear-velocity constraint"))
    np.savez_compressed(d / "seasonal_amplitude_preorigin.npz", **sa)

    # ---- network_segments.json
    dec = np.load(V1 / "decomposition_2025.npz", allow_pickle=False)
    lo, hi = (pd.Timestamp(t).strftime("%Y%m%d") for t in C.DECOMP_SPAN)
    gd = [str(x) for x in G["dates"]]
    ks = [k for k in range(len(gd) - 1) if lo <= gd[k] and gd[k + 1] <= hi]
    dgap = G["cell_gap"][:, ks].any(1)
    ok = dec["ok"]
    ns = {"rule": gaps["F"]["cell_rule"],
          "runs": {r: {k: gaps[r][k] for k in ("first", "last", "n_epochs", "gaps", "segments",
                                               "pixel_n_gap",
                                               "frac_pixels_with_gaps_beyond_network",
                                               "cells_with_data", "cell_n_gaps_any_pixel",
                                               "intervals_with_most_pixel_gaps")}
                   for r in ("P", "F", "D")},
          "registered_windows": {
              "G1 (run P, 2021-08-31..2022-12-30)": gaps["G1_window_gaps"]["P"],
              "G1 after the fact (run F)": gaps["G1_window_gaps"]["F"],
              "pilot window (run F, 2021-11-05..2023-03-22)": gaps["pilot_window_gaps_F"],
              "seasonal amplitude Y (run P, 2020-01..2022-12)": ["_".join(g) for g in
                                                                 gaps["P"]["gaps"]],
              "decomposition 2025 (runs F and D, 2025-01-23..2025-12-11)": {
                  "network_gaps": [], "frac_ok_cells_with_a_pixel_gap_in_span_runF":
                  float(dgap[ok].mean()), "run_D_pixel_gaps": gaps["D"]["pixel_n_gap"]}},
          "flag_counts_vertical_cells": {int(k): int(v) for k, v in
                                         zip(*np.unique(flag, return_counts=True), strict=True)},
          "guidance": "Compare InSAR with any validator or the twin only through changes whose "
                      "two epochs share a segment id, or carry the bridged offsets as an "
                      "extra uncertainty. Every summer 2022-2024 of run F has a network gap "
                      "(2022-06-06/06-30, 2022-08-05/08-29, 2023-07-31/08-12, "
                      "2024-07-01/08-06); 2025 has none."}
    Lb.jdump(ns, d / "network_segments.json")

    # ---- validation.json
    val1 = j(V1 / "validation.json")
    pfd = diag["pf"]
    val1["repair_round"] = {
        "date": "2026-10-14", "supersedes": f"{C.PRODUCT}/v1",
        "changes": ["flag 4 and segment ids for values that include LiCSBAS-bridged gap "
                    "increments (v1 flagged every value 0)",
                    "gnss_tie_offsets.csv removed (GNSS-derived; spends the G1 stations)",
                    "fit_window_end 2026-01 (run D pairs up to 2026-01-30)",
                    "seasonal-amplitude status: physical interpretation withdrawn"],
        "no_retest": "values unchanged from v1; the registered tests are not re-scored; the "
                     "gate stays failed (G1 13.2 mm on run P)",
        "network_gaps": {r: gaps[r]["gaps"] for r in ("P", "F", "D")},
        "g1_failure_diagnostics_after_the_fact": {
            "insar_only_P_vs_F": {
                "across_gap1_rms_mm": pfd["across_gap1"][0]["rms_mm"],
                "across_gap2_rms_mm": pfd["across_gap2"][0]["rms_mm"],
                "within_segment_intervals_rms_mm": pfd["within_rms_mm"],
                "within_18_30d_rms_mm": pfd["within_18_30d_rms_mm"],
                "last_segment": pfd["last_segment"], "g1_change": pfd["g1_change"]},
            "station_error_split_P_to_F": split,
            "single_epoch_noise": {nm: diag["pieces"][nm]["within_single_interval_rms_mm"]
                                   for nm in ("P", "F")},
            "reading": "the P-to-F change of the G1 station errors comes from the bridged "
                       "cross-gap increments (InSAR only, corr 0.98); within-segment and "
                       "last-segment increments agree to about 1 mm; the 'end effect' "
                       "explanation of round 1 is withdrawn"},
        "reference_coefficients": diag["refcoef"],
        "y_easting_confound": diag["yconf"]}
    Lb.jdump(val1, d / "validation.json")

    write_card(d, gaps, ns, pfd, split, diag)

    # ---- manifest
    g1 = val1["G1"]
    gate = dict(m1["gate"])
    gate["criteria"] = (m1["gate"]["criteria"] + "; v2 = v1 values with network-gap flags, "
                        "not re-tested")
    outputs = []
    for p in sorted(x.name for x in d.iterdir()):
        rec = {"path": p}
        if p == "coherence_summer_preorigin.npz":
            rec["fit_window_end"] = "2022-09"
        if p == "seasonal_amplitude_preorigin.npz":
            rec["fit_window_end"] = "2022-12"
        if p == "vertical_cells.npz" or p == "vertical_ml.npz":
            rec["fit_window_end"] = "2025-12"
        if p == "decomposition_2025.npz":
            rec["fit_window_end"] = "2026-01"
        outputs.append(rec)
    inputs = [rel_to_repo(V1 / "MANIFEST.json")]
    inputs += [rel_to_repo(V1 / f) for f in (
        "vertical_cells.npz", "vertical_ml.npz", "reference_epochs.csv",
        "coherence_summer_preorigin.npz", "zone_coverage.json", "decomposition_2025.npz",
        "seasonal_amplitude_preorigin.npz", "validation.json")]
    for nm in ("P", "F", "D"):
        ts = C.RUN_DIR / nm / "TS_GEOCml2"
        inputs += [rel_to_repo(ts / "cum.h5"), rel_to_repo(ts / "results" / "mask"),
                   rel_to_repo(ts / "info" / "12network_gap_info.txt"),
                   rel_to_repo(C.RUN_DIR / f"gaps_{nm}.npz")]
    inputs += [rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz"),
               rel_to_repo(C.RUN_DIR / f"geom_{C.DESC}.npz"),
               rel_to_repo(C.EVID / "repair_gaps.json"), rel_to_repo(C.EVID / "repair_diag.json"),
               rel_to_repo(C.EVID / "repair_pf_split.json"), rel_to_repo(C.TWIN_GRID),
               "results/dataprep/preregistration/S4b.insar_acceptance.v1.json"]
    params = dict(m1["parameters"])
    params.update({"derived_from": f"{C.PRODUCT}/v1", "flag_rule": FLAG_NOTE,
                   "cell_gap_rule": gaps["F"]["cell_rule"]})
    m = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=inputs,
        code=[rel_to_repo(HERE / c) for c in CODE], parameters=params,
        fit_window_end="2026-01", validation_report="validation.json", gate=gate,
        allowed_uses=["validator", "after_the_fact_analysis"],
        forbidden_uses=PREORIGIN_USES + ["scoring_target", "prospective_scorer",
                                         "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none", outputs=outputs,
        extra={"licence": C.LICENCE, "supersedes": f"{C.PRODUCT}/v1",
               "zone_coverage": m1["zone_coverage"],
               "notes": "InSAR is a validator of twin subsidence 2022-2026 only (U7): never a "
                        "column target, never a gate, threshold or member choice input. The "
                        "gate failed: the guard refuses 'validator' until a user decision; "
                        "'after_the_fact_analysis' stays open. Changes across network "
                        "segments include LiCSBAS-bridged offsets (flag 4; segment ids)."})
    manifest.write_manifest(d, m)
    print(json.dumps({"product": str(d), "gate": gate["passed"], "n_outputs": len(outputs),
                      "flags": ns["flag_counts_vertical_cells"],
                      "decomp_frac_gap": ns["registered_windows"][
                          "decomposition 2025 (runs F and D, 2025-01-23..2025-12-11)"],
                      "g1_rms": g1["primary"]["rms_mm"]}))
    return 0


def write_card(d, gaps, ns, pfd, split, diag) -> None:
    v = j(d / "validation.json")
    g1, c1, y, af = v["G1"], v["C1"], v["Y"], v["after_the_fact"]
    cov = j(d / "zone_coverage.json")["time_series_coverage_runF"]
    fc = ns["flag_counts_vertical_cells"]
    tot = sum(fc.values())
    rc = diag["refcoef"]
    yc = diag["yconf"]
    z = c1["zones"]
    yrs = pd.Series([int(str(q)[:4]) for q in c1["pairs"]]).value_counts().sort_index()
    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S4b, repair round 2026-10-14)", "",
        "Gate: **FAILED** (pre-registered S4b.insar_acceptance.v1: G1 WRA GNSS leave-one-"
        f"station-out 14-month change RMS {g1['primary']['rms_mm']:.1f} mm over "
        f"{g1['primary']['n_evaluable']} stations on the pre-origin run P, threshold "
        f"{C.GNSS_LOSO_RMS_MAX_MM} mm). v2 carries v1's values unchanged; it adds network-gap "
        "flags and segment ids, drops gnss_tie_offsets.csv and corrects fit_window_end "
        "(2026-01). Nothing was re-tested.", "",
        "LiCSAR frame 069A_06694_131402 (ascending), LiCSBAS commit "
        f"{C.LICSBAS_COMMIT[:12]} at defaults (2 x 2 multilook, NSBAS with gamma 1e-4, no "
        "atmospheric correction, D11), run F 2020-01-12..2025-12-11 (192 epochs). Each epoch "
        "is referenced to a stable off-fan area (land, LiCSAR DEM 100-600 m, >= 2.5 km from "
        "every active cell centre, LiCSBAS15 mask) by a robust offset + east/north plane + "
        "height fit; vertical-equivalent = LOS / U (per-pixel U 0.80-0.85). No GNSS, ring, "
        "pile or leveling value is in the product.", "",
        "## Network gaps (read before using any change)", "",
        "LiCSBAS12 left the network in disconnected segments. Across a gap no interferogram "
        "spans the interval, so LiCSBAS13 sets that increment by its linear-velocity "
        "constraint: the offset between segments is a model value, not a measurement.", "",
        f"- Run F (product): gaps {', '.join(a + '/' + b for a, b in gaps['F']['gaps'])}; "
        f"{len(gaps['F']['segments'])} segments. Every summer 2022-2024 has one; 2025 has none.",
        f"- Run P (gate run): gaps {', '.join(a + '/' + b for a, b in gaps['P']['gaps'])}; "
        "both lie inside the G1 window and inside the seasonal-amplitude fit.",
        "- Run D (descending, 2025-01..2026-01): no network gap.",
        f"- {100 * (1 - gaps['F']['frac_pixels_with_gaps_beyond_network']):.1f} % of the "
        "active-cell pixels have exactly the network gaps; "
        f"{100 * gaps['F']['cell_n_gaps_any_pixel']['frac_cells_equal_to_network']:.1f} % of "
        "the cells with data have no extra cell gap.", "",
        "Flags in vertical_cells.npz: 0 = same segment as the first epoch at every "
        "contributing pixel; 4 = the value includes at least one bridged increment; 8 = "
        "missing; 9 = epoch with < 200 stable pixels (none). Counts: " + ", ".join(
            f"flag {k} {100 * n / tot:.1f} %" for k, n in sorted(fc.items())) + ". "
        "`segment` (A, T) gives each value's segment id: **a change between two epochs is "
        "interferometric only when both share the segment id.** vertical_ml.npz carries "
        "`pix_segment` and LiCSBAS's `n_gap` per pixel.", "",
        "Evidence that the bridges matter (after the fact, report only): runs P and F differ "
        f"across the 2022-08 gap by RMS {pfd['across_gap2'][0]['rms_mm']:.1f} mm and across "
        f"the 2022-06 gap by {pfd['across_gap1'][0]['rms_mm']:.1f} mm, against at most "
        f"{pfd['within_18_30d_rms_mm']['max']:.1f} mm over within-segment 18-30 d intervals "
        f"and {pfd['last_segment']['rms_mm']:.1f} mm over the last segment of run P. The "
        f"change of the G1 station errors from P to F (RMS "
        f"{split['rms_err_change_mm']:.1f} mm) is carried by the cross-gap increments "
        f"(RMS {split['rms_gap_part_change_mm']:.1f} mm, correlation "
        f"{split['corr_err_change_gap_change']:.2f}); the within-segment part changes by "
        f"{split['rms_within_segment_part_change_mm']:.1f} mm.", "",
        "## Files", "",
        "vertical_cells.npz (value/sd/flag/segment (A, T) mm, positive up, relative to "
        "2020-01-12; cell_gap, frac_pix_gap (A, T-1); network_segment (T,)), vertical_ml.npz "
        "(multilooked grid, pix_segment, n_gap), reference_epochs.csv (+ network_segment), "
        "network_segments.json, coherence_summer_preorigin.npz and zone_coverage.json, "
        "seasonal_amplitude_preorigin.npz, decomposition_2025.npz (fit_window_end 2026-01), "
        "validation.json (v1 tests + repair_round). sd is UNCALIBRATED and does not include "
        "the bridged-offset uncertainty.", "",
        "## Registered tests (pre-origin run P; unchanged)", "",
        f"- G1: RMS {g1['primary']['rms_mm']:.1f} mm, FAIL. The window spans both 2022 gaps "
        "of run P.",
        "- C1 Changhua summer coherence: " + "; ".join(
            f"{k.split('.')[-1]} {vv['median_frac_coherent']:.3f} "
            f"({'covered' if vv['covered'] else 'NOT covered'}; 2022 alone "
            f"{vv['per_year_median']['2022']:.3f})" for k, vv in z.items()) +
        f". The pooled median is dominated by 2020 ({int(yrs.get(2020, 0))} of {int(yrs.sum())} "
        "pairs; " + ", ".join(f"{k}: {int(n)}" for k, n in yrs.items()) + ").",
        f"- Y: Y1 rho {y['Y1_all_yunlin']['rho_primary']:.3f} pass; Y2 "
        f"{y['Y2_battery']['wins']}/8 FAIL. The amplitude is NOT a validator. The test "
        "design cannot separate local pumping from east-west structure: both thickness "
        f"proxies are functions of easting, rho(amplitude, easting) "
        f"{yc['rho_amp_easting']:.2f}, rho(census density, easting) "
        f"{yc['rho_density5_easting']:.2f}, and the run-P plane's east coefficient has an "
        f"annual amplitude of {rc['P']['east_km']['annual_amp']:.3f} mm/km (LOS), i.e. "
        f"{rc['P_east_annual_amp_los_mm_at_30_50km'][0]:.0f}-"
        f"{rc['P_east_annual_amp_los_mm_at_30_50km'][1]:.0f} mm 30-50 km west of the stable "
        f"centroid (x {rc['stable_centroid_x_km']:.1f} km).", "",
        "## After the fact (report only)", "",
        f"- Run F, G1 stations and window: {af['F_g1']['loso']['rms_mm']:.1f} mm; pilot "
        f"window: {af['pilot_window_F']['loso']['rms_mm']:.1f} mm. Both windows also span "
        "the two 2022 gaps.",
        f"- 2025 asc/desc decomposition: vertical-equivalent minus vertical "
        f"{af['decomp_2025']['median_vert_eq_minus_up_mm_yr']:.1f} mm/yr median, RMS "
        f"{af['decomp_2025']['rms_vert_eq_minus_up_mm_yr']:.1f} mm/yr.",
        f"- Reference ramp (exploratory): the run-F plane coefficients drift by "
        f"{rc['F']['east_km']['rate_per_yr']:+.2f} (east) and "
        f"{rc['F']['north_km']['rate_per_yr']:+.2f} (north) mm/km/yr (LOS).", "",
        "## Coverage", "",
        "Active cells with >= 5 valid pixels (run F): " + "; ".join(
            f"{k} {vv:.2f}" for k, vv in cov.items()) + ". The primary footprint misses the "
        "northern band of Changhua.", "",
        "## Uses", "",
        "Allowed: validator (refused by the guard while the gate fails), "
        "after_the_fact_analysis. Every pre-origin use is forbidden (U7). Licence: "
        + C.LICENCE]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
