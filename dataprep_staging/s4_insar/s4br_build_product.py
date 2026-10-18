"""WP S4b repair round 2 (S4br): build ``results/dataprep/insar_069A/v<N>/`` (v3; never an
overwrite) from the repaired runs.

* Full product: run F2 (primary frame 069A_06694, 2020-01..2025-12) merged with run S2 (second
  ascending frame 069A_06632, 2022-11..2025-12; M5). Both use M1 (LiCSBAS12 --multi_prime
  --nullify -l pi, LiCSBAS13 --nullify_noloops), M2 (LiCSBAS16 filtered series) and M3 (robust
  per-epoch offset + height reference on each frame's stable strip; no plane). S2 is placed on
  the primary multilooked grid by index (1/6 pixel E-W, 1/36 N-S offset), tied pixel by pixel to
  F2 at their first common epoch, and averaged with F2 where both are valid
  (``s4br_lib.tie_and_merge``).
* Cells: median of >= 5 valid pixels (round 1's rule). ``value`` is relative to 2020-01-12 and
  uses primary-referenced pixels only (F2, tied S2, or their mean); cells that only the second
  frame covers get ``value_second_frame_only``, relative to the second frame's first epoch.
* Flags: 0 observed (interferometric chain from the reference epoch at every contributing
  pixel); 4 the value includes >= 1 increment LiCSBAS13 bridged with its linear-velocity
  constraint (any contributing pixel, either frame); 8 missing; 9 epoch whose stable set has
  < 200 pixels.
* ``value_round1_plane_reference`` (cells only): the same repaired, merged series referenced
  with round 1's offset + east/north plane + height fit instead of M3. It is an AFTER-THE-FACT
  alternative, never tested on fresh units: the registered M3 reference failed in the 2021-22
  window (WRA second look 17.7 mm) where the plane gave 7.3 mm, while the plane gave 17.5 mm at
  the 4 NGL stations in 2020-21 where M3 gave 7.3 mm (S4b.repair.md). Users must not pick the
  variant that fits a validator.
* Seasonal amplitude: run P2 (pre-origin), with the Y second-look status.
* Coherence and zone coverage: copied from v2 (LiCSAR coherence; C1 is unchanged).

    python -I s4br_build_product.py     (hydro env, one CPU slot)
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
import s4br_config as R  # noqa: E402
import s4br_lib as RL  # noqa: E402
from common import manifest  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

V2 = C.DATAPREP / C.PRODUCT / "v2"
CODE = ["s4br_config.py", "s4br_lib.py", "s4br_run_licsbas.sh", "s4br_workdir.py",
        "s4br_geom.py", "s4br_fetch_06632.py", "s4br_looks.py", "s4br_diag.py",
        "s4br_evaluate.py", "s4br_prereg.py", "s4br_after.py", "s4br_build_product.py",
        "s4b_config.py", "s4b_lib.py", "s4b_evaluate.py", "make_licsbas_workdir.py",
        "licsbas_env.sh"]
PREORIGIN_USES = ["calibration_target", "datum_fit", "baseline", "initial_conditions",
                  "idw_field", "prior", "prior_width_input", "model_design", "selection",
                  "forcing_frozen", "design_time_evidence"]
FLAG_NOTE = ("0 observed: interferometric chain from the reference epoch at every contributing "
             "pixel of either frame; 4 the value includes >= 1 increment bridged by LiCSBAS13's "
             "linear-velocity (gamma) constraint at a contributing pixel (plan 2.4 flag 4); "
             "8 missing; 9 epoch whose stable set has < 200 pixels in the frame used")
SRC_NOTE = ("source per (pixel or cell, epoch): 1 primary frame only; 2 second frame only, "
            "tied to the primary at the first common epoch; 3 mean of both; 4 second frame "
            "only, not tied (relative to the second frame's first epoch); 0 none")


def j(p: Path):
    return json.loads(Path(p).read_text())


def pixel_segments(run: dict) -> np.ndarray:
    """(T, L, W) int16: number of LiCSBAS13 gap intervals before each epoch per pixel."""
    g = run["gap"]
    T = run["cum"].shape[0]
    seg = np.zeros((T,) + run["cum"].shape[1:], np.int16)
    if g is not None:
        seg[1:] = np.cumsum(g.astype(np.int16), axis=0)
    return seg


def cells_median(field: np.ndarray, valid: np.ndarray, g: dict, A: int):
    return Lb.to_cells(field, valid, g, A)


def main() -> int:
    tests = {k: j(R.EVID / f"test_{k}.json") for k in ("g1_ngl", "g1_wra", "y")}
    after = j(R.EVID / "after.json")
    design = j(R.EVID / "design.json")
    A = Lb.n_active()
    runF, gF, vF, tabF = RL.processed("F2")
    runS, gS, vS, tabS = RL.processed("S2")
    assert gF["x"].shape == gS["x"].shape
    segF, segS = pixel_segments(runF), pixel_segments(runS)
    val, src, dates, tstar = RL.tie_and_merge(vF, runF["dates"], runF["mask"],
                                              vS, runS["dates"], runS["mask"])
    T = len(dates)
    L, W = val.shape[1:]
    iF = {d: i for i, d in enumerate(runF["dates"])}
    iS = {d: i for i, d in enumerate(runS["dates"])}
    ts = pd.Timestamp(tstar)
    segF_t = segF[iF[ts]]
    bridged = np.zeros((T, L, W), bool)
    stable_bad = np.zeros(T, bool)
    badF = set(tabF.loc[tabF["n_kept"] < C.STABLE_MIN_PIXELS, "date"])
    badS = set(tabS.loc[tabS["n_kept"] < C.STABLE_MIN_PIXELS, "date"])
    none = np.zeros((L, W), bool)
    for k, d in enumerate(dates):
        bF = segF[iF[d]] > 0 if d in iF else none            # primary chain from 2020-01-12
        bS = (segF_t + segS[iS[d]]) > 0 if d in iS else none  # tied second-frame chain
        b4 = segS[iS[d]] > 0 if d in iS else none             # untied second-frame chain
        s = src[k]
        bridged[k] = ((s == 1) & bF) | ((s == 2) & bS) | ((s == 3) & (bF | bS)) | \
            ((s == 4) & b4)
        ds = str(d.date())
        stable_bad[k] = (d in iF and ds in badF) or (d in iS and ds in badS)
    # ---- after-the-fact alternative: round 1's plane reference on the same repaired runs
    _, _, vFp, tabFp = RL.processed("F2", terms=C.REF_TERMS)
    _, _, vSp, tabSp = RL.processed("S2", terms=C.REF_TERMS)
    valp, srcp, datesp, _ = RL.tie_and_merge(vFp, runF["dates"], runF["mask"],
                                             vSp, runS["dates"], runS["mask"])
    assert datesp.equals(dates)
    del vFp, vSp
    # ---- cells
    prim = np.isin(src, (1, 2, 3))
    value = np.full((A, T), np.nan, np.float32)
    sd = np.full((A, T), np.nan, np.float32)
    flag = np.full((A, T), 8, np.uint8)
    srcc = np.zeros((A, T), np.int8)
    v2only = np.full((A, T), np.nan, np.float32)
    valuep = np.full((A, T), np.nan, np.float32)
    pix, cid = Lb.cell_groups(gF)
    for k in range(T):
        okp = prim[k] & np.isfinite(val[k])
        vv, ss, n = cells_median(val[k], okp, gF, A)
        value[:, k], sd[:, k] = vv, ss
        okq = np.isin(srcp[k], (1, 2, 3)) & np.isfinite(valp[k])
        valuep[:, k] = cells_median(valp[k], okq, gF, A)[0]
        fin = np.isfinite(vv)
        b = pd.Series((bridged[k].ravel()[pix] & okp.ravel()[pix]).astype(int)).groupby(cid).max()
        bc = np.zeros(A, bool)
        bc[b.index.to_numpy()] = b.to_numpy() > 0
        flag[:, k] = np.where(fin, np.where(bc, 4, 9 if stable_bad[k] else 0), 8)
        sm = pd.Series(np.where(okp.ravel()[pix], src[k].ravel()[pix], 0)).groupby(cid).agg(
            lambda z: int(pd.Series(z[z > 0]).mode().iloc[0]) if (z > 0).any() else 0)
        srcc[sm.index.to_numpy(), k] = sm.to_numpy()
        ok4 = (src[k] == 4) & np.isfinite(val[k])
        if ok4.any():
            v4, _, _ = cells_median(val[k], ok4, gF, A)
            v2only[:, k] = v4
    after_t = np.array([d >= ts for d in dates])
    has_prim_after = np.isfinite(value[:, after_t]).any(1)
    v2only[has_prim_after] = np.nan
    d = manifest.new_version_dir(C.PRODUCT)
    mask, dx, x0, y0 = Lb.twin_grid()
    np.savez_compressed(
        d / "vertical_cells.npz", value=value, sd=sd, flag=flag, source=srcc,
        value_second_frame_only=v2only, value_round1_plane_reference=valuep,
        value_round1_plane_reference_note=np.array(
            "AFTER-THE-FACT alternative, not tested on fresh units: the same repaired merged "
            "series referenced with round 1's robust offset + east/north plane + height fit on "
            "the stable strip; flags and sources as for value; never choose the variant by its "
            "fit to a validator"),
        dates=np.array([str(x.date()) for x in dates]),
        reference_epoch=np.array(str(dates[0].date())),
        reference_epoch_second_frame_only=np.array(tstar),
        U_cell=Lb.to_cells(gF["U"], np.isfinite(gF["U"]), gF, A)[0],
        nx=np.array(mask.shape[1]), ny=np.array(mask.shape[0]), dx=np.array(dx),
        x0=np.array(x0), y0=np.array(y0),
        units=np.array("mm, vertical-equivalent (LOS / U), positive up, relative to the "
                       "reference epoch; cell = median of >= 5 valid multilooked pixels"),
        sd_note=np.array("UNCALIBRATED robust standard error of the cell median "
                         "(1.2533 x 1.4826 x MAD / sqrt(n)); excludes reference, atmosphere "
                         "and bridged-offset errors; the calibrated statement is G1"),
        flag_note=np.array(FLAG_NOTE), source_note=np.array(SRC_NOTE))
    np.savez_compressed(
        d / "vertical_ml.npz", value=val, source=src, bridged=bridged,
        dates=np.array([str(x.date()) for x in dates]), lon=gF["lon"], lat=gF["lat"],
        x=gF["x"], y=gF["y"], cell=gF["cell"], U=gF["U"].astype(np.float32),
        E=gF["E"].astype(np.float32), hgt=gF["hgt"].astype(np.float32),
        licsbas15_mask_primary=runF["mask"], licsbas15_mask_second=runS["mask"],
        stable_primary=gF["stable_geom"] & runF["mask"],
        stable_second=gS["stable_geom"] & runS["mask"],
        pix_segment_primary=segF, pix_segment_second=segS,
        dates_primary=np.array([str(x.date()) for x in runF["dates"]]),
        dates_second=np.array([str(x.date()) for x in runS["dates"]]),
        tie_epoch=np.array(tstar), flag_note=np.array(FLAG_NOTE), source_note=np.array(SRC_NOTE),
        grid_note=np.array("primary frame's 2 x 2 multilooked grid; the second frame is placed "
                           "by index (offset 1/6 pixel E-W, 1/36 N-S)"))
    ref = pd.DataFrame({"date": [str(x.date()) for x in dates]})
    for nm, tb in (("primary", tabF), ("second", tabS), ("primary_plane", tabFp),
                   ("second_plane", tabSp)):
        t2 = tb.rename(columns={c: f"{nm}_{c}" for c in tb.columns if c != "date"})
        ref = ref.merge(t2, on="date", how="left")
    ref.to_csv(d / "reference_epochs.csv", index=False)
    for f in ("coherence_summer_preorigin.npz", "zone_coverage.json"):
        shutil.copyfile(V2 / f, d / f)
    # ---- seasonal amplitude (P2)
    runP, gP, vP, _ = RL.processed("P2")
    lo, hi = (pd.Timestamp(t) for t in C.SEAS_WINDOW)
    sel = (runP["dates"] >= lo) & (runP["dates"] <= hi)
    h = Lb.harmonic_fit(vP[sel], runP["dates"][sel])
    validP = runP["mask"] & np.isfinite(h["amp"])
    amp_c, _, _ = Lb.to_cells(h["amp"], validP, gP, A)
    y = tests["y"]
    np.savez_compressed(
        d / "seasonal_amplitude_preorigin.npz", amp=h["amp"].astype(np.float32),
        phase_doy=h["phase_doy"], rate=h["rate"].astype(np.float32), resid_sd=h["resid_sd"],
        valid=validP, amp_cells=amp_c,
        status=np.array(
            f"Y second look on run P2 (spent units, conditional): Y1 p05 "
            f"{y['Y1_all_yunlin']['boot_p05']:.3f}, Y2 {y['Y2_battery']['wins']}/8; "
            f"{'passes' if y['passed'] else 'fails'}. The registered Y failed (round 1); a "
            "second look cannot make the amplitude a validator. Not a validator."),
        network_gaps_runP2=np.array(["_".join(gp) for gp in runP["network_gaps"]]))
    # ---- coverage of the merged product
    U = {x["id"]: x for x in j(C.UNITS_V3)["units"]}
    cov = {}
    groups = {**{z: U[z]["cells"] for z in C.CH_ZONES},
              "yunlin": sorted(set().union(*[set(U[u]["cells"]) for u in C.YUNLIN_UNITS])),
              "all_active": list(range(A))}
    for nm, cells in groups.items():
        cells = np.array(cells, int)
        cov[nm] = {
            "primary_referenced_2020_2025": float(np.isfinite(value[cells]).any(1).mean()),
            "primary_referenced_after_tie": float(np.isfinite(value[cells][:, after_t]).any(1)
                                                  .mean()),
            "second_frame_only_after_tie": float(np.isfinite(v2only[cells]).any(1).mean()),
            "any_after_tie": float((np.isfinite(value[cells][:, after_t]).any(1)
                                    | np.isfinite(v2only[cells]).any(1)).mean())}
    # ---- frame agreement where both are valid (after the tie epoch)
    both_k = [k for k, dd in enumerate(dates) if dd in iF and dd in iS and dd > ts]
    diffs = []
    for k in both_k:
        dd = dates[k]
        a = np.where(runF["mask"], vF[iF[dd]], np.nan)
        b = np.where(runS["mask"], vS[iS[dd]], np.nan)
        off = (np.where(runF["mask"], vF[iF[ts]], np.nan)
               - np.where(runS["mask"], vS[iS[ts]], np.nan))
        dlt = a - (b + off)
        dlt = dlt[(gF["cell"] >= 0) & np.isfinite(dlt)]
        if dlt.size:
            diffs.append({"date": str(dd.date()), "n_pix": int(dlt.size),
                          "rms_mm": float(np.sqrt(np.mean(dlt ** 2))),
                          "median_mm": float(np.median(dlt))})
    dfd = pd.DataFrame(diffs)
    fc = {int(k): int(n) for k, n in zip(*np.unique(flag, return_counts=True), strict=True)}
    fin_n = int(np.isfinite(value).sum())
    ng = {"P2": runP["network_gaps"], "F2": runF["network_gaps"], "S2": runS["network_gaps"]}
    netw = {"runs": {nm: {"gaps": ng[nm], "n_epochs": int(r["cum"].shape[0]),
                          "first": str(r["dates"][0].date()), "last": str(r["dates"][-1].date())}
                     for nm, r in (("P2", runP), ("F2", runF), ("S2", runS))},
            "tie_epoch": tstar, "flag_counts_vertical_cells": fc,
            "frac_finite_cell_values_flag4": float(fc.get(4, 0) / max(fin_n, 1)),
            "guidance": "changes between two epochs are interferometric where both values are "
                        "flag 0; flag-4 values include bridged increments"}
    Lb.jdump(netw, d / "network_segments.json")
    validation = {
        "prereg": R.PREREG_ID, "repair_round": "2026-10-18 (S4br)",
        "supersedes": f"{C.PRODUCT}/v2", "methods": {
            "M1": R.LICSBAS_STEPS[2:4], "M2": R.LICSBAS_STEPS[6],
            "M3": list(R.REF_TERMS), "M4": "NGL steps.txt type 1; S3 v4 after the fact only",
            "M5": f"{R.SECOND} merged ({R.MERGE_RULE})"},
        "G1_ngl_fresh": tests["g1_ngl"], "G1_wra_second_look": tests["g1_wra"],
        "Y_second_look": tests["y"], "after_the_fact": after, "design": design,
        "coverage_merged": cov, "frame_agreement_after_tie": {
            "n_epochs": int(len(dfd)),
            "median_rms_mm": float(dfd["rms_mm"].median()) if len(dfd) else None,
            "max_rms_mm": float(dfd["rms_mm"].max()) if len(dfd) else None,
            "per_epoch": dfd.round(3).to_dict("records")},
        "network": netw}
    Lb.jdump(validation, d / "validation.json")
    gate_pass = bool(tests["g1_ngl"]["primary"]["passed"])
    gate = {"passed": gate_pass, "status": "passed" if gate_pass else "failed",
            "criteria": ("pre-registered S4b.insar_repair.v1: G1 (plan S4 acceptance, 10 mm "
                         "with >= 5 stations) on the FRESH NGL family on the repaired "
                         "pre-origin run P2; the WRA family and Y are second looks "
                         "(conditional, no evidential weight)"),
            "tests": {"G1_ngl_fresh": gate_pass,
                      "G1_wra_second_look": tests["g1_wra"]["primary"]["passed"],
                      "Y_second_look": tests["y"]["passed"],
                      "C1_zone_covered": j(V2 / "MANIFEST.json")["gate"]["tests"][
                          "C1_zone_covered"]},
            "rms_mm": {"G1_ngl_fresh": tests["g1_ngl"]["primary"]["rms_mm"],
                       "G1_wra_second_look": tests["g1_wra"]["primary"]["rms_mm"]}}
    write_card(d, validation, gate, cov, netw, dfd)
    outputs = []
    for p in sorted(x.name for x in d.iterdir()):
        rec = {"path": p}
        if p == "coherence_summer_preorigin.npz":
            rec["fit_window_end"] = "2022-09"
        if p == "seasonal_amplitude_preorigin.npz":
            rec["fit_window_end"] = "2022-12"
        outputs.append(rec)
    inputs = [rel_to_repo(V2 / "MANIFEST.json"),
              rel_to_repo(V2 / "coherence_summer_preorigin.npz"),
              rel_to_repo(V2 / "zone_coverage.json")]
    for nm in ("P2", "F2", "S2"):
        tsd = R.RUN_DIR / nm / "TS_GEOCml2"
        inputs += [rel_to_repo(tsd / R.CUM_FILE), rel_to_repo(tsd / "cum.h5"),
                   rel_to_repo(tsd / "results" / "mask"),
                   rel_to_repo(tsd / "info" / "12network_gap_info.txt")]
    inputs += [rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz"),
               rel_to_repo(R.RUN_DIR / f"geom_{R.SECOND}.npz"),
               rel_to_repo(R.L0_SECOND / "SOURCE.json"),
               "data_fetch_api/new_measurements/licsar_069A_06694/fetch_2026-10-12_s4a/"
               "SOURCE.json",
               rel_to_repo(R.EVID / "test_g1_ngl.json"), rel_to_repo(R.EVID / "test_g1_wra.json"),
               rel_to_repo(R.EVID / "test_y.json"), rel_to_repo(R.EVID / "after.json"),
               rel_to_repo(R.EVID / "design.json"), rel_to_repo(C.TWIN_GRID),
               rel_to_repo(C.UNITS_V3),
               f"results/dataprep/preregistration/{R.PREREG_ID}.json"]
    m2 = j(V2 / "MANIFEST.json")
    params = {"licsbas_commit": C.LICSBAS_COMMIT, "licsbas_steps": R.LICSBAS_STEPS,
              "runs": {k: list(v) for k, v in R.RUNS.items()}, "product_runs": ["F2", "S2"],
              "gate_run": "P2", "series": R.CUM_FILE, "atmosphere": "none (D11); LiCSBAS16",
              "stable_area": m2["parameters"]["stable_area"],
              "reference_terms": list(R.REF_TERMS), "vertical": C.VERTICAL,
              "cell_min_pixels": C.CELL_MIN_PIXELS, "merge_rule": R.MERGE_RULE,
              "tie_epoch": tstar, "prereg": R.PREREG_ID, "flag_rule": FLAG_NOTE,
              "source_rule": SRC_NOTE}
    m = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=inputs,
        code=[rel_to_repo(HERE / c) for c in CODE], parameters=params,
        fit_window_end="2025-12", validation_report="validation.json", gate=gate,
        allowed_uses=["validator", "after_the_fact_analysis"],
        forbidden_uses=PREORIGIN_USES + ["scoring_target", "prospective_scorer",
                                         "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none", outputs=outputs,
        extra={"licence": C.LICENCE, "supersedes": f"{C.PRODUCT}/v2",
               "zone_coverage": m2["zone_coverage"],
               "notes": "InSAR is a validator of twin subsidence 2022-2026 only (U7): never a "
                        "column target, never a gate, threshold or member choice input. "
                        "Changhua Z2 is NOT covered in summer (C1). Flag-4 values include "
                        "bridged increments."})
    manifest.write_manifest(d, m)
    print(json.dumps({"product": rel_to_repo(d), "gate": gate, "flags": fc,
                      "coverage_all_active": cov["all_active"],
                      "frame_agreement_median_rms": validation["frame_agreement_after_tie"][
                          "median_rms_mm"]}, default=Lb._js))
    return 0


def write_card(d: Path, v: dict, gate: dict, cov: dict, netw: dict, dfd: pd.DataFrame) -> None:
    gn, gw, y = v["G1_ngl_fresh"]["primary"], v["G1_wra_second_look"]["primary"], \
        v["Y_second_look"]
    fc = netw["flag_counts_vertical_cells"]
    tot = sum(fc.values())
    gaps = {k: ", ".join("/".join(x) for x in r["gaps"]) or "none"
            for k, r in netw["runs"].items()}
    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S4b, repair round 2, 2026-10-18)", "",
        f"Gate: **{'PASSED' if gate['passed'] else 'FAILED'}** (pre-registered "
        f"{R.PREREG_ID}: G1 on the fresh NGL family, run P2, RMS "
        f"{gn['rms_mm'] if gn['rms_mm'] is None else round(gn['rms_mm'], 1)} mm over "
        f"{gn['n_evaluable']} evaluable stations; threshold {C.GNSS_LOSO_RMS_MAX_MM} mm with "
        f">= {C.GNSS_MIN_STATIONS} stations). WRA family, second look (conditional): "
        f"{gw['rms_mm']:.1f} mm over {gw['n_evaluable']} stations. Y second look: "
        f"{'pass' if y['passed'] else 'fail'} ({y['Y2_battery']['wins']}/8); the amplitude is "
        "not a validator.", "",
        "Method (each change tied to a diagnosed defect in reports/S4b.repair.md): LiCSBAS12 "
        "with the multi-prime bias removal and per-pixel nullification of loop errors (loop "
        "threshold pi), LiCSBAS13 with no-loop nullification, LiCSBAS16 spatio-temporal "
        "filter (defaults), and a stated off-fan reference: robust per-epoch offset + height "
        "term on each frame's stable strip (land, DEM 100-600 m, >= 2.5 km from active cells, "
        "LiCSBAS15 mask), no plane. No atmospheric correction product (D11). No GNSS, ring, "
        "pile or leveling value is in the product.", "",
        f"Frames: primary 069A_06694 (run F2, 2020-01-12..2025-12-11) and second ascending "
        f"069A_06632 (run S2, from {netw['tie_epoch']}), merged on the primary grid: S2 is "
        "tied to F2 at their first common epoch pixel by pixel and the two are averaged where "
        f"both are valid. Frame agreement after the tie: median per-epoch RMS "
        f"{float(dfd['rms_mm'].median()) if len(dfd) else float('nan'):.1f} mm (same "
        "acquisitions, so this measures processing differences, not atmosphere).", "",
        "## Network gaps", "",
        f"- P2 (gate run): {gaps['P2']}.", f"- F2: {gaps['F2']}.", f"- S2: {gaps['S2']}.",
        "Flags in vertical_cells.npz: " + ", ".join(
            f"flag {k} {100 * n / tot:.1f} %" for k, n in sorted(fc.items())) +
        f" of cell-epochs; flag 4 on {100 * netw['frac_finite_cell_values_flag4']:.1f} % of "
        "finite values (v2: 51.2 %).", "",
        "## Coverage (cells with >= 5 valid pixels at some epoch)", ""] + [
        f"- {k}: 2020-25 primary-referenced {c['primary_referenced_2020_2025']:.2f}; after the "
        f"tie epoch, any frame {c['any_after_tie']:.2f} (second frame only "
        f"{c['second_frame_only_after_tie']:.2f})" for k, c in cov.items()] + [
        "", "## Files", "",
        "vertical_cells.npz (value/sd/flag/source (A, T), value_second_frame_only), "
        "vertical_ml.npz (merged multilooked values, source, bridged, per-frame masks, stable "
        "pixels and gap segments), reference_epochs.csv (both frames), network_segments.json, "
        "seasonal_amplitude_preorigin.npz (run P2), coherence_summer_preorigin.npz and "
        "zone_coverage.json (unchanged from v2), validation.json. sd is UNCALIBRATED.", "",
        "## Limits", "",
        "- The fresh NGL window lies in 2020-21 (6-day revisit, no network gap in P2); the "
        "product validates 2022-26 (12-day revisit, summer decorrelation).",
        "- Changhua Z2 is not covered in summer (C1, unchanged). The asc/desc decomposition "
        "from 2022-11 remains impossible (no descending GeoTIFFs before 2025-01).",
        "- East-west motion leaks into LOS / U (about 3 mm RMS across stations over 14 "
        "months, from the 2025 decomposition); the systematic part of long-wavelength "
        "atmospheric and ionospheric ramps that the filter cannot remove stays in the "
        "product.", "",
        "## Uses", "",
        "Allowed: validator (only while the gate is passed; the guard decides), "
        "after_the_fact_analysis. Every pre-origin use is forbidden (U7). Licence: "
        f"{C.LICENCE}"]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
