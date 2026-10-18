"""WP S5 repair round 2: build results/dataprep/subsidence_fused/v<N> (v4) with the repaired
model (R1-R6 of s5r2_const) under S5.repair_r2.v1.

    python -I s5r2_looks.py product_build_v4 && tools/cpu_slot.sh python -I s5r2_product.py

Full data: leveling 2012-2021 (round-1 QC), gnss_clean/full/v4 with offset segments at
daily-record gaps > 30 d, insar_069A/v3 (flag 0, sources 1-3), rings to 2025-05 over the S1
interval (rings_bridged/v2 ring definitions, free segment offsets), printed piles to 2025-11.
Hyperparameters and depth fractions from H2 main (pre-origin, hyper_main.json). Rules carried
from v3: GNSS gap segments, stepped benchmarks (scan on a first fit), GNSS and ring/pile
screening by one |z| > 5 pass on the first fit, mask_year excludes 2026 (flag 6). Not carried:
the monthly field (stage 2 / M1 were registered on the round-1 annual model).

Outputs: fused_annual.npz, fused_cells_annual.parquet, gnss_segments.csv,
leveling_steps.csv, hyperparameters.json, validation.json, PRODUCT_CARD.md, MANIFEST.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5fit as F  # noqa: E402
import s5product as P1  # noqa: E402
import s5r2_const as K  # noqa: E402
import s5r2_data as RD  # noqa: E402
import s5r2_model as RM  # noqa: E402
import s5r_core as RC  # noqa: E402
import s5r_vconst as V  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import grid_hash  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5tests.py", "s5product.py",
        "s5r_core.py", "s5r_vconst.py", "s5r2_const.py", "s5r2_model.py", "s5r2_data.py",
        "s5r2_hyper.py", "s5r2_tests.py", "s5r2_insarP2.py", "s5r2_prereg.py",
        "s5r2_product.py", "tests/test_s5r2.py"]
SHARED = ["dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
          "dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
          "dataprep_staging/common/hashing.py", "dataprep_staging/common/paths.py",
          "dataprep_staging/common/uses.py", "dataprep_staging/common/flags.py",
          "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/prereg3.py",
          "dataprep_staging/s4_insar/s4b_lib.py", "dataprep_staging/s4_insar/s4br_lib.py",
          "dataprep_staging/s4_insar/s4br_config.py", "dataprep_staging/s4_insar/s4b_config.py"]
UPSTREAM = [K.GNSS_FULL_V4, K.GNSS_PRE_V4, K.INSAR_V3, C.PILES, C.RINGS_BRIDGED]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jload(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def input_files() -> list[str]:
    from s1_rings import s1const as K1
    f = [C.LEVELING_PANEL, C.RING_STATIONS, C.TWIN_GRID,
         K.GNSS_FULL_V4 / "MANIFEST.json", K.GNSS_FULL_V4 / "stations.csv",
         K.GNSS_FULL_V4 / "wra" / "monthly.parquet", K.GNSS_FULL_V4 / "ngl" / "monthly.parquet",
         K.GNSS_FULL_V4 / "wra" / "ncku_monthly.parquet"]
    f += [K.GNSS_FULL_V4 / v for v in V.GNSS_DAILY.values()]
    f += [K.INSAR_V3 / "MANIFEST.json", K.INSAR_V3 / "vertical_cells.npz",
          C.PILES / "MANIFEST.json", C.PILES / "printed_monthly.parquet",
          C.RINGS_BRIDGED / "MANIFEST.json", C.RINGS_BRIDGED / "ring_corrections.csv",
          K.GNSS_PRE_V4 / "MANIFEST.json", K.INSAR_P2_CELLS, K.INSAR_P2_CELLS.with_suffix(".json"),
          K.EVID / "hyper_main.json", K.EVID / "hyper_t1.json", K.EVID / "hyper_t1f.json",
          C.DATAPREP / "preregistration" / f"{K.PREREG_ID}.json",
          C.DATAPREP / "preregistration" / f"{C.PREREG_ID}.json"]
    f += [C.RING_DIR / f"{s}.parquet" for s in K1.SITES]
    f += [K.EVID / n for n in ("t1f.json", "t1.json", "t2.json", "t2b.json", "t3.json")]
    return [rel_to_repo(p) for p in f if Path(p).exists()]


def gate_state() -> dict:
    t1f, t2, t2b, t1 = (jload(K.EVID / n) for n in ("t1f.json", "t2.json", "t2b.json",
                                                    "t1.json"))
    t1f_pass = bool((t1f or {}).get("pass"))
    t2_pass = bool((t2 or {}).get("pass"))
    if t1f_pass and t2_pass:
        status = "conditional (second look): T1F (fresh) passed, T2 passed on a second look"
    else:
        status = "FAILED"
    return {"passed": False if status == "FAILED" else None, "status": status,
            "T1F_fresh_pass": t1f_pass, "T2_second_look_pass": t2_pass,
            "T2b_second_look_pass": bool((t2b or {}).get("pass")),
            "T1_second_look_pass": bool((t1 or {}).get("pass"))}


def main() -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.time()
    base = jload(K.EVID / "hyper_main.json")
    h, fdict = base["hyper"], base["fdict"]
    data, qc = RD.load_all2("full")
    g = data["grid"]
    breaks = RC.gnss_breaks(RC.read_daily("full"))
    flagged = {"lev": base["flagged"]["lev"]}
    # leveling rids are identical between the pre and full loads (same leveling table)
    m0 = RD.masks_from(data, flagged)
    pr0 = RM.make_problem2(data, C.KNOTS_FULL, F.FAMILIES, masks=m0, fdict=fdict)
    post0 = RM.fit2(pr0, h)
    steps = RC.leveling_steps(pr0, post0, h)
    flagged["gnss"] = RM.outliers2(pr0, post0, h, "gnss")
    flagged["cmp"] = RM.outliers2(pr0, post0, h, "cmp")
    del post0, pr0
    data["lev"]["seg"] = RC.lev_segments(data["lev"], steps)
    log(f"first fit: {int(steps.flagged.sum())} stepped benchmarks; screening gnss "
        f"{len(flagged['gnss'])}, cmp {len(flagged['cmp'])} ({time.time() - t0:.0f} s)")
    m = RD.masks_from(data, flagged)
    pr = RM.make_problem2(data, C.KNOTS_FULL, F.FAMILIES, masks=m, fdict=fdict)
    post = RM.fit2(pr, h, want_cov=True)
    data_end = P1.data_end_month(pr)
    log(f"final fit: N {pr.N}, data to {data_end} ({time.time() - t0:.0f} s)")
    F.set_ell(pr, h["ell_km"])
    Phi = pr.lat.weights(g["cell_xy_km"])
    tb = pr.tb
    A = g["A"]
    years = np.array(C.PRODUCT_YEARS)
    disp = np.zeros((tb.T, A))
    disp_sd = np.zeros((tb.T, A))
    inc = np.zeros((years.size, A))
    inc_sd = np.zeros((years.size, A))
    for k in range(tb.T):
        disp[k], disp_sd[k] = P1.cell_stats(pr, post, Phi, tb.knot_row(k), tb.knots[k] - C.T_REF)
    for i in range(years.size):
        inc[i], inc_sd[i] = P1.cell_stats(pr, post, Phi, tb.knot_row(i + 1) - tb.knot_row(i),
                                          1.0)
    a_test = tb.knot_row(2026 - 2012) - tb.knot_row(2023 - 2012)
    test, test_sd = P1.cell_stats(pr, post, Phi, a_test, 3.0)
    ec = np.zeros(tb.K)
    ec[tb.ic] = 1.0
    es = np.zeros(tb.K)
    es[tb.isn] = 1.0
    s_cos, s_cos_sd = P1.cell_stats(pr, post, Phi, ec, 0.0)
    s_sin, s_sin_sd = P1.cell_stats(pr, post, Phi, es, 0.0)
    th = post.theta
    glob = {n: {"mean": float(th[pr.L + i]), "sd": float(np.sqrt(post.cov[pr.L + i, pr.L + i]))}
            for i, n in enumerate(pr.gnames)
            if not n.startswith(("ioff:", "goff:", "gcm:", "irate:", "lcamp:"))}
    camp = {n[6:]: {"mean": float(th[pr.L + i]), "sd": float(np.sqrt(post.cov[pr.L + i,
                                                                              pr.L + i]))}
            for i, n in enumerate(pr.gnames) if n.startswith("lcamp:")}
    n_par = pr.N
    seg_tab = RC.segment_table(pr.gnss, breaks)
    del post
    prl = RM.make_problem2(data, C.KNOTS_FULL, ("lev",), masks={"lev": m["lev"]})
    postl = RM.fit2(prl, h, want_cov=True)
    F.set_ell(prl, h["ell_km"])
    inc_l = np.zeros_like(inc)
    inc_sd_l = np.zeros_like(inc_sd)
    for i in range(years.size):
        inc_l[i], inc_sd_l[i] = P1.cell_stats(prl, postl, Phi, tb.knot_row(i + 1)
                                              - tb.knot_row(i), 1.0)
    test_l, test_sd_l = P1.cell_stats(prl, postl, Phi, a_test, 3.0)
    del postl
    flag = np.where(years[:, None] >= C.EXTRAPOLATED_FROM, 6, 3).astype(np.uint8) * \
        np.ones((1, A), np.uint8)
    dflag = np.where(tb.knots[:, None] > 2026.0, 6, 3).astype(np.uint8) * np.ones((1, A),
                                                                                  np.uint8)
    dflag[0] = 0
    vr_year = 1.0 - inc_sd ** 2 / inc_sd_l ** 2
    vr_test = 1.0 - test_sd ** 2 / test_sd_l ** 2
    mask_year = (vr_year >= C.VR_MIN) & (flag != 6)
    mask_test = vr_test >= C.VR_MIN
    log(f"leveling-only fit done ({time.time() - t0:.0f} s)")
    gate = gate_state()
    sd_label = ("UNCALIBRATED unless the card states a passed coverage check: posterior sd of "
                "the 4 km-lattice field under the repaired model; gate status: "
                + gate["status"] + ". A point (benchmark) adds sig_rb * years, sig_lev and "
                "sig_camp.")
    d = manifest.new_version_dir(C.PRODUCT)
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    gh = grid_hash(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]), float(z["y0"]),
                   z["mask"])
    np.savez_compressed(
        d / "fused_annual.npz", knots=tb.knots, years=years,
        disp=disp.astype(np.float32), disp_sd=disp_sd.astype(np.float32), disp_flag=dflag,
        inc=inc.astype(np.float32), inc_sd=inc_sd.astype(np.float32), inc_flag=flag,
        inc_levonly=inc_l.astype(np.float32), inc_sd_levonly=inc_sd_l.astype(np.float32),
        vr_year=vr_year.astype(np.float32), mask_year=mask_year,
        change_2023_2026=test.astype(np.float32), change_2023_2026_sd=test_sd.astype(np.float32),
        change_2023_2026_levonly=test_l.astype(np.float32),
        change_2023_2026_sd_levonly=test_sd_l.astype(np.float32),
        vr_test=vr_test.astype(np.float32), mask_test=mask_test,
        seasonal_cos=s_cos.astype(np.float32), seasonal_cos_sd=s_cos_sd.astype(np.float32),
        seasonal_sin=s_sin.astype(np.float32), seasonal_sin_sd=s_sin_sd.astype(np.float32),
        units=np.array("mm (displacement, + = sinking) and mm/yr (annual increments); relative "
                       "to 2012-01-01; cells in np.nonzero(mask) order"),
        sd_note=np.array(sd_label),
        mask_note=np.array("mask_year = vr_year >= 0.5 and flag != 6; vr against a leveling-only "
                           "fit with the same hyperparameters"),
        flag_note=np.array("3 = fused estimate from several instruments; 6 = extrapolated "
                           "(year 2026, knot 2027-01-01); disp at 2012-01-01 = 0 (flag 0)"),
        nx=z["nx"], ny=z["ny"], dx=z["dx"], x0=z["x0"], y0=z["y0"], grid_hash=np.array(gh))
    rows = []
    for i, y in enumerate(years):
        rows.append(pd.DataFrame({
            "cell": np.arange(A), "year": int(y), "value": inc[i], "sd": inc_sd[i],
            "flag": flag[i], "vr": vr_year[i], "in_mask": mask_year[i],
            "sd_levonly": inc_sd_l[i], "disp_end": disp[i + 1], "disp_end_sd": disp_sd[i + 1],
            "method": "s5 space-time fusion v4 (repaired: reverting temporal prior, correlated "
                      "instrument noise, campaign offsets, InSAR unit rates)",
            "product_version": f"{C.PRODUCT}/{d.name}",
            "source_ids": "leveling L0; gnss_clean/full/v4; insar_069A/v3; rings L0; "
                          "deep_benchmarks_printed/v2"}))
    pd.concat(rows, ignore_index=True).to_parquet(d / "fused_cells_annual.parquet", index=False)
    seg_tab.to_csv(d / "gnss_segments.csv", index=False)
    steps.to_csv(d / "leveling_steps.csv", index=False)
    F.dump({"hyper": h, "hyper_fit_window_end": "2022-12", "f": base["f"], "fdict": fdict,
            "screening_full": {k: len(v) for k, v in flagged.items()}, "screened_rids": flagged,
            "globals": glob, "campaign_offsets": camp, "n_parameters": n_par,
            "lattice_nodes": pr.m, "knots": list(tb.knots),
            "stepped_benchmarks": int(steps.flagged.sum())}, d / "hyperparameters.json")
    act = mask_year[:, :]
    yi = {int(y): i for i, y in enumerate(years)}
    val = {"gate": gate, "tests": {n: jload(K.EVID / f"{n}.json") and {
        k: v for k, v in jload(K.EVID / f"{n}.json").items()
        if k in ("pass", "criteria", "scored", "coverage_fused", "ratio_fused_idw",
                 "rmse_fused", "rmse_idw", "n_benchmarks_scored", "registration", "rms_z",
                 "families", "test", "test_id")} for n in ("t1f", "t1", "t2", "t2b", "t3")},
           "qc": qc, "data_end": data_end,
           "median_increment_mm_yr": {int(y): float(np.median(inc[yi[y]])) for y in years},
           "median_inc_sd_mm_yr": {int(y): float(np.median(inc_sd[yi[y]])) for y in years},
           "mask_year_share": {int(y): float(act[yi[y]].mean()) for y in years},
           "mask_test_share": float(mask_test.mean()),
           "change_2023_2026": {"median": float(np.median(test)), "max": float(np.max(test)),
                                "median_sd": float(np.median(test_sd))},
           "stepped_benchmarks": int(steps.flagged.sum()),
           "gnss_extra_segments": int(len(seg_tab) - seg_tab.station.nunique())}
    F.dump(val, d / "validation.json")
    write_card(d, val, h, gate)
    fwe = max([data_end, "2022-12"] + [str((jload(p / "MANIFEST.json") or {}).get(
        "fit_window_end") or "0000-00") for p in UPSTREAM])
    mm = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=input_files(),
        code=[rel_to_repo(HERE / c) for c in CODE] + SHARED,
        parameters={"hyper": h, "hyper_fit_window_end": "2022-12", "fdict": fdict,
                    "lattice_km": C.LATTICE_KM, "knots": list(tb.knots), "vr_min": C.VR_MIN,
                    "gnss_gap_days": V.GNSS_GAP_DAYS,
                    "step_rule": {"z": V.STEP_Z, "min_mm": V.STEP_MIN_MM,
                                  "min_side": V.STEP_MIN_SIDE},
                    "prereg": [C.PREREG_ID, K.PREREG_ID], "data_end": data_end,
                    "screening_full": {k: len(v) for k, v in flagged.items()}},
        fit_window_end=fwe, validation_report="validation.json",
        gate={"passed": bool(gate["passed"]) if gate["passed"] is not None else False,
              "status": gate["status"],
              "criteria": (f"{K.PREREG_ID}: T1F (fresh temporal block, 2018-19 leveling "
                           "withheld at a pseudo-origin 2020-01; VR >= 0.5; RMSE < continue-"
                           "rate baseline; coverage3) AND T2 (registered design, second look); "
                           "a pass on these terms is conditional (second look) and is not "
                           "recorded as passed")},
        allowed_uses=["after_the_fact_analysis"], forbidden_uses=P1.FORBIDDEN,
        personal_data="none (agency benchmark and station coordinates only)",
        extra={"upstream_gate_rule": (
            "allowed uses are after_the_fact_analysis only whatever the S5 gate: the GNSS "
            "(gnss_clean v4, S3), InSAR (insar_069A v3, S4b) and ring-interval (rings_bridged, "
            "S1a) inputs failed their gates"),
            "supersedes": ["subsidence_fused/v3 (annual part; v3's monthly field is not "
                           "rebuilt in v4)"]})
    errs = manifest.validate_manifest(mm, d)
    if errs:
        raise SystemExit(f"manifest invalid: {errs}")
    manifest.write_manifest(d, mm)
    p = guard.open_product(d, "after_the_fact_analysis")
    log(f"wrote {d}; guard opens after_the_fact_analysis: {p.use}; {time.time() - t0:.0f} s")
    return 0


def write_card(d: Path, val: dict, h: dict, gate: dict) -> None:
    t = val["tests"]

    def g(n, *ks):
        x = t.get(n) or {}
        for k in ks:
            x = (x or {}).get(k) if isinstance(x, dict) else None
        return x

    def f3(x):
        return "n/a" if x is None else f"{x:.3f}"

    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S5, repair round 2, 2026-10-18)", "",
        f"Gate: **{gate['status']}**. Allowed use: after_the_fact_analysis only (upstream GNSS, "
        "InSAR and ring inputs are gate-failed products).", "",
        "## Tests (S5.repair_r2.v1; thresholds of S5.subsidence_fusion.v1)", "",
        "| Test | Units | Ratio | Coverage | Pass |", "|---|---|---|---|---|",
        f"| T1F temporal block 2018-19 (pseudo-origin 2020-01) | fresh window | "
        f"{f3(g('t1f', 'scored', 'ratio_fused_baseline'))} | "
        f"{f3(g('t1f', 'scored', 'coverage_fused', 'coverage'))} | {g('t1f', 'pass')} |",
        f"| T1 registered design 2020-21 | second look | "
        f"{f3(g('t1', 'scored', 'ratio_fused_baseline'))} | "
        f"{f3(g('t1', 'scored', 'coverage_fused', 'coverage'))} | {g('t1', 'pass')} |",
        f"| T2 registered design (no buffer) | second look | {f3(g('t2', 'ratio_fused_idw'))} | "
        f"{f3(g('t2', 'coverage_fused', 'coverage'))} | {g('t2', 'pass')} |",
        f"| T2b with 10 km buffer (plan 4.3) | second look | {f3(g('t2b', 'ratio_fused_idw'))} | "
        f"{f3(g('t2b', 'coverage_fused', 'coverage'))} | {g('t2b', 'pass')} |", "",
        "## Model", "",
        "Matern 3/2 field on a 4 km lattice x annual-knot temporal basis with a reverting prior "
        "(long-term velocity, white knot-velocity anomalies, reverting displacement anomalies), "
        "seasonal pair; leveling with campaign offsets; GNSS (gnss_clean/full/v4, gap segments) "
        "with correlated station noise; InSAR (insar_069A/v3) with per-month planes, APS and "
        "per-unit random rates; rings and piles with depth fractions and correlated sensor "
        "noise. Hyperparameters: H2 on pre-origin data (U7).", "",
        "Hyperparameters: " + ", ".join(f"{k} {v:.3g}" for k, v in h.items()), "",
        "## Limits", "",
        "- sd: calibrated only to the extent the tests above pass; a T2-type check that is a "
        "second look gives no evidential weight.",
        "- No monthly field in v4 (v3's monthly field was built on the round-1 annual model).",
        "- T1F has no InSAR (none before 2020): the InSAR operator is checked only by second "
        "looks (T1) and T3 (report only).",
        "- Depth fractions f are estimated, then held fixed (f uncertainty not propagated).", ""]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
