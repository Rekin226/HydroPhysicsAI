"""WP S5 repair round: build results/dataprep/subsidence_fused/v<N> (v2) under the
declaration ``S5.subsidence_fused_rules.v2`` (L2 observation product).

    python -I s5r_looks.py product_build_v2 && tools/cpu_slot.sh python -I s5r_product.py
    tools/cpu_slot.sh python -I s5r_product.py --dry <scratch>/v9999   (synthetic dry run)

Full data as v1 (leveling 2012-2021, gnss_clean/full/v2, insar_069A/v2 run F, printed piles
to 2025-11, rings to 2025-05), the pre-origin hyperparameters and depth fractions of
hyper_pre.json, and the v2 amendments:

1. GNSS offset segments at every gap > 30 d of the daily record (s5r_core.gnss_breaks);
2. stepped benchmarks: scanned on a first fit (leveling screening of hyper_pre.json), the
   GNSS and ring/pile |z| > 5 screening taken on the same fit, a second offset from the step;
3. mask_year excludes flag 6 (year 2026);
4. the monthly field 2020-01..2025-12 from stage 2 on the run-F residuals of the final fit with
   the pre-origin stage-2 hyperparameters (S5_evidence/repair/hyper_stage2.json);
5-8. fit_window_end with inputs, provenance, labels, sensitivity fits (report only).

Outputs: fused_annual.npz, fused_cells_annual.parquet, fused_monthly.npz,
fused_cells_monthly.parquet, gnss_segments.csv, leveling_steps.csv, hyperparameters.json,
sensitivity.json, validation.json, PRODUCT_CARD.md.
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
import s5data as D  # noqa: E402
import s5fit as F  # noqa: E402
import s5hyper as Hy  # noqa: E402
import s5product as P1  # noqa: E402  (round-1 helpers, read-only)
import s5r_const as R  # noqa: E402
import s5r_core as RC  # noqa: E402
import s5r_stage2 as S2  # noqa: E402
import s5r_vconst as V  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import grid_hash  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5hyper.py", "s5tests.py",
        "s5insarP.py", "s5looks.py", "s5prereg.py", "s5product.py", "s5r_const.py",
        "s5r_vconst.py", "s5r_core.py", "s5r_stage2.py", "s5r_mtest.py", "s5r_prereg_m1.py",
        "s5r_prereg_v2.py", "s5r_prereg_v3.py", "s5r_product.py", "tests/test_s5r_stage2.py",
        "tests/test_s5r_core.py"]
SHARED = ["dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
          "dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
          "dataprep_staging/common/hashing.py", "dataprep_staging/common/paths.py",
          "dataprep_staging/common/uses.py", "dataprep_staging/common/flags.py",
          "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/prereg3.py",
          "dataprep_staging/s4_insar/s4b_lib.py", "dataprep_staging/s4_insar/s4b_evaluate.py",
          "dataprep_staging/s4_insar/s4b_config.py"]
UPSTREAM = [C.GNSS_FULL, C.GNSS_PRE, C.INSAR_V2, C.PILES, C.RINGS_BRIDGED]
SD_NOTE = ("UNCALIBRATED. Posterior sd of the 4 km-lattice field. The registered coverage "
           "checks failed (T1 0.631 with rms z 2.47 across the 2020-21 gap: too small; T2 0.940: "
           "too wide inside the leveling era); after 2021 the sd is probably too small by about "
           "2.5. A point (benchmark) adds sig_rb * years and sig_lev.")


def monthly_sd_label(m1: dict | None) -> str:
    """Declaration S5.subsidence_fused_rules.v3 (labels): the monthly VALUE sd is uncalibrated
    (it carries the annual posterior variance, whose T1/T2 coverage checks failed); M1's
    coverage check concerns the anomaly relative to the annual model only."""
    m1 = m1 or {}
    cov = (m1.get("scored") or {}).get("coverage") or {}
    if not cov:
        return "uncalibrated (M1 coverage not computed)"
    return ("uncalibrated for the monthly value (the annual part is uncalibrated: T1 0.631, "
            "T2 0.940); the anomaly relative to the annual model passed"
            if cov.get("pass") else "uncalibrated; the anomaly relative to the annual model "
            "failed") + (f" M1's coverage check: {cov.get('coverage', float('nan')):.3f} "
                         f"(interval {cov.get('lo', float('nan')):.3f}-"
                         f"{cov.get('hi', float('nan')):.3f}, {cov.get('k_clusters')} station "
                         "clusters" + ("" if cov.get("calibrated") else
                                       "; outside the envelope in which coverage3 was "
                                       "validated") + "); registered M1 label: '"
                         + str(m1.get("sd_label")) + "'")


def m1_years(m1: dict | None) -> str:
    by = (m1 or {}).get("scored_by_year") or {}
    return ", ".join(f"{y} {b.get('ratio', float('nan')):.3f}" for y, b in by.items())


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jload(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def increments(pr, theta, Phi, years) -> np.ndarray:
    tb = pr.tb
    Z = theta[:pr.L].reshape(pr.K, pr.m)
    return np.stack([Phi @ ((tb.knot_row(i + 1) - tb.knot_row(i)) @ Z) + theta[pr.L]
                     for i in range(years.size)])


def change_23_26(pr, theta, Phi) -> np.ndarray:
    tb = pr.tb
    Z = theta[:pr.L].reshape(pr.K, pr.m)
    a = tb.knot_row(2026 - 2012) - tb.knot_row(2023 - 2012)
    return Phi @ (a @ Z) + 3.0 * theta[pr.L]


def gnss_with_segments(data: dict, breaks: dict) -> None:
    data["gnss"]["seg"] = RC.gnss_segments(data["gnss"], breaks)


def main(dry_dir: Path | None = None) -> int:
    """Build the next product version; with ``dry_dir`` a DRY RUN on SYNTHETIC values
    (s5dryrun loader, the dry-run stage-2 hyperparameters and M1) written to ``dry_dir``
    instead of a product version (code-path check before the declaration)."""
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.time()
    base = jload(C.EVIDENCE / "hyper_pre.json")
    s2dir = R.EVIDENCE_R if dry_dir is None else R.EVIDENCE_R / "dryrun_monthly" / "anomaly" \
        / "repair"
    hs = jload(s2dir / "hyper_stage2.json")
    m1 = jload(s2dir / "m1.json")
    if dry_dir is not None:
        import s5dryrun as DR
        D.load_all = DR.load_all
    if hs is None:
        raise SystemExit("hyper_stage2.json missing: run s5r_mtest.py hyper first")
    h, fdict, h2 = base["hyper"], base["fdict"], hs["hyper"]
    data, qc = Hy.load("full")
    g = data["grid"]
    # ---- amendment 1: GNSS offset segments
    breaks = RC.gnss_breaks(RC.read_daily("full"))
    gnss_with_segments(data, breaks)
    data["lev"]["seg"] = 0
    # ---- first fit: stepped benchmarks and the GNSS / ring-pile screening
    flagged = {"lev": base["flagged"]["lev"]}
    m0 = Hy.masks_from(data, flagged)
    pr0 = RC.make_problem_v2(data, C.KNOTS_FULL, F.FAMILIES, masks=m0, fdict=fdict)
    post0 = RC.fit_v2(pr0, h)
    steps = RC.leveling_steps(pr0, post0, h)
    flagged["gnss"] = F.outliers(pr0, post0, h, "gnss")
    flagged["cmp"] = F.outliers(pr0, post0, h, "cmp")
    del post0, pr0
    data["lev"]["seg"] = RC.lev_segments(data["lev"], steps)
    log(f"first fit: {int(steps.flagged.sum())} stepped benchmarks of {len(steps)} scanned; "
        f"screening gnss {len(flagged['gnss'])}, cmp {len(flagged['cmp'])} "
        f"({time.time() - t0:.0f} s)")
    # ---- final fit with covariance
    m = Hy.masks_from(data, flagged)
    pr = RC.make_problem_v2(data, C.KNOTS_FULL, F.FAMILIES, masks=m, fdict=fdict)
    post = RC.fit_v2(pr, h, want_cov=True)
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
            for i, n in enumerate(pr.gnames) if not n.startswith(("ioff:", "goff:", "gcm:"))}
    # ---- monthly field (amendment 4)
    months = S2.month_range(R.MONTHLY_FIRST, R.MONTHLY_LAST)
    groups = S2.residual_groups(pr, post)
    s2 = S2.Stage2(pr.lat.Z, pr.cell_xy, pr._unit_xy, pr.insar["unit_cells"], months, groups)
    sol = s2.solve(h2)
    mom = s2.cell_moments(h2, sol)
    tm = D.month_mid(months)
    trend = np.zeros((len(months), A))
    trend_sd = np.zeros((len(months), A))
    for k, t in enumerate(tm):
        trend[k], trend_sd[k] = P1.cell_stats(pr, post, Phi, tb.rows([t])[0], t - C.T_REF)
    an, an_sd = mom["mean"], np.sqrt(mom["var"])
    support_vr = mom["support"]
    m1_pass = bool((m1 or {}).get("pass"))
    # amendment 4: monthly values only if the registered test M1 passed
    support = support_vr if m1_pass else np.zeros_like(support_vr)
    mval = np.where(support, trend + an, np.nan)
    msd = np.where(support, np.sqrt(trend_sd ** 2 + an_sd ** 2), np.nan)
    mflag = np.where(support, 3, 8).astype(np.uint8)
    n_par = pr.N
    log(f"monthly: {len(groups)} run-F groups, M1 pass {m1_pass}, VR support "
        f"{support_vr.mean():.3f}, delivered {support.mean():.3f} of cell-months "
        f"({time.time() - t0:.0f} s)")
    del post
    # ---- leveling-only fit (same operators) for the VR mask
    prl = RC.make_problem_v2(data, C.KNOTS_FULL, ("lev",), masks=m)
    postl = RC.fit_v2(prl, h, want_cov=True)
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
    # ---- sensitivity fits (report only)
    sens = sensitivity(data, m, fdict, h, breaks, g, Phi, years, inc, test, flagged, base)
    log(f"sensitivity fits done ({time.time() - t0:.0f} s)")
    # ---- write
    if dry_dir is None:
        d = manifest.new_version_dir(C.PRODUCT)
    else:
        d = Path(dry_dir)
        d.mkdir(parents=True, exist_ok=False)
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    gh = grid_hash(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]), float(z["y0"]),
                   z["mask"])
    common_meta = dict(nx=z["nx"], ny=z["ny"], dx=z["dx"], x0=z["x0"], y0=z["y0"],
                       grid_hash=np.array(gh))
    np.savez_compressed(
        d / "fused_annual.npz", knots=tb.knots, years=years,
        disp=disp.astype(np.float32), disp_sd=disp_sd.astype(np.float32), disp_flag=dflag,
        inc=inc.astype(np.float32), inc_sd=inc_sd.astype(np.float32), inc_flag=flag,
        inc_levonly=inc_l.astype(np.float32), inc_sd_levonly=inc_sd_l.astype(np.float32),
        vr_year=vr_year.astype(np.float32), mask_year=mask_year,
        change_2023_2026=test.astype(np.float32), change_2023_2026_sd=test_sd.astype(np.float32),
        change_2023_2026_sd_levonly=test_sd_l.astype(np.float32),
        vr_test=vr_test.astype(np.float32), mask_test=mask_test,
        seasonal_cos=s_cos.astype(np.float32), seasonal_cos_sd=s_cos_sd.astype(np.float32),
        seasonal_sin=s_sin.astype(np.float32), seasonal_sin_sd=s_sin_sd.astype(np.float32),
        units=np.array("mm (displacement, + = sinking) and mm/yr (annual increments); "
                       "relative to 2012-01-01; cells in np.nonzero(mask) order"),
        sd_note=np.array(SD_NOTE),
        mask_note=np.array("mask_year = vr_year >= 0.5 and flag != 6 (2026 never in the mask); "
                           "the mask is almost non-binding (q at its upper bound): see the card"),
        flag_note=np.array("3 = fused estimate from several instruments (plan 2.4: filled using "
                           "other sites or instruments); 6 = extrapolated beyond the record "
                           "(year 2026, knot 2027-01-01); disp at 2012-01-01 = 0 by definition "
                           "(flag 0)"), **common_meta)
    np.savez_compressed(
        d / "fused_monthly.npz", months=np.array(months), t_mid=tm,
        value=mval.astype(np.float32), sd=msd.astype(np.float32), flag=mflag,
        support=support, support_vr=support_vr, m1_pass=np.array(m1_pass),
        trend=trend.astype(np.float32), trend_sd=trend_sd.astype(np.float32),
        anomaly=an.astype(np.float32), anomaly_sd=an_sd.astype(np.float32),
        vr_anomaly_planefree=mom["vr_dt"].astype(np.float32), footprint=s2.footprint,
        units=np.array("mm, + = sinking, displacement relative to 2012-01-01 at mid-month; "
                       "cells in np.nonzero(mask) order"),
        sd_note=np.array("sd^2 = annual posterior variance at mid-month + stage-2 anomaly "
                         "variance (two-stage, independent). Label: " + monthly_sd_label(m1)
                         + ". anomaly_sd alone is the stage-2 posterior sd."),
        support_note=np.array("value is given (flag 3) only if the registered test M1 "
                              "(S5.monthly_field.v1) passed (m1_pass) and only where the cell "
                              "lies in the month's InSAR footprint and the plane-free anomaly "
                              "variance reduction vr_anomaly_planefree >= 0.5 (support_vr); "
                              "elsewhere NaN (flag 8). trend = "
                              "annual fused field incl. the constant seasonal pair; anomaly = "
                              "stage-2 posterior mean (its planar part is not resolved by "
                              "InSAR and is prior-driven)."), **common_meta)
    rows = []
    for i, y in enumerate(years):
        rows.append(pd.DataFrame({
            "cell": np.arange(A), "year": int(y), "value": inc[i], "sd": inc_sd[i],
            "flag": flag[i], "vr": vr_year[i], "in_mask": mask_year[i],
            "sd_levonly": inc_sd_l[i], "disp_end": disp[i + 1], "disp_end_sd": disp_sd[i + 1],
            "method": "s5 space-time fusion v2 (separable Matern x IRW velocity; leveling, "
                      "GNSS with gap offsets, InSAR, rings, piles)",
            "product_version": f"{C.PRODUCT}/{d.name}",
            "source_ids": "leveling L0; gnss_clean/full/v2; insar_069A/v2; rings L0; "
                          "deep_benchmarks_printed/v2"}))
    pd.concat(rows, ignore_index=True).to_parquet(d / "fused_cells_annual.parquet",
                                                  index=False)
    k_i, c_i = np.nonzero(support)
    pd.DataFrame({"cell": c_i, "month": np.array(months)[k_i], "value": mval[k_i, c_i],
                  "sd": msd[k_i, c_i], "flag": mflag[k_i, c_i], "trend": trend[k_i, c_i],
                  "anomaly": an[k_i, c_i], "anomaly_sd": an_sd[k_i, c_i],
                  "vr_anomaly_planefree": mom["vr_dt"][k_i, c_i],
                  "method": "s5 v2 monthly: annual fusion + stage-2 InSAR anomaly",
                  "product_version": f"{C.PRODUCT}/{d.name}",
                  "source_ids": "subsidence_fused annual fit; insar_069A/v2 run F"}
                 ).to_parquet(d / "fused_cells_monthly.parquet", index=False)
    seg_tab = RC.segment_table(pr.gnss, breaks)
    seg_tab.to_csv(d / "gnss_segments.csv", index=False)
    steps.to_csv(d / "leveling_steps.csv", index=False)
    F.dump({"hyper": h, "hyper_fit_window_end": "2022-12", "f": base["f"], "fdict": fdict,
            "stage2": h2, "stage2_fit_window_end": hs.get("fit_window_end"),
            "screening_full": {k: len(v) for k, v in flagged.items()}, "screened_rids": flagged,
            "globals": glob, "n_parameters": n_par, "lattice_nodes": pr.m,
            "knots": list(tb.knots), "gnss_segments_extra": int(len(seg_tab) -
                                                               seg_tab.station.nunique()),
            "stepped_benchmarks": int(steps.flagged.sum())}, d / "hyperparameters.json")
    F.dump(sens, d / "sensitivity.json")
    val = build_validation(base, qc, vr_year, vr_test, mask_year, mask_test, years, g, m1, hs,
                           steps, seg_tab, support, mom, months, support_vr)
    F.dump(val, d / "validation.json")
    fwe = fit_window_end_with_inputs(data_end)
    write_card(d, val, h, h2, sens, years, steps, seg_tab, support, months, fwe, data_end,
               support_vr)
    m_ = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=input_files(dry=dry_dir is not None),
        code=[rel_to_repo(HERE / c) for c in CODE] + SHARED,
        parameters={"hyper": h, "hyper_fit_window_end": "2022-12", "fdict": fdict,
                    "stage2": h2, "lattice_km": C.LATTICE_KM, "knots": list(tb.knots),
                    "vr_min": C.VR_MIN, "vr_a_min": R.VR_A_MIN,
                    "gnss_gap_days": V.GNSS_GAP_DAYS, "step_rule": {
                        "z": V.STEP_Z, "min_mm": V.STEP_MIN_MM, "min_side": V.STEP_MIN_SIDE},
                    "prereg": C.PREREG_ID,
                    "declaration": [R.DECL_ID, "S5.subsidence_fused_rules.v3"],
                    "monthly_test": R.MTEST_ID, "data_end": data_end,
                    "screening_full": {k: len(v) for k, v in flagged.items()}},
        fit_window_end=fwe, validation_report="validation.json",
        gate={"passed": False,
              "criteria": ("S5.subsidence_fusion.v1: T1 (2020-2021 leveling withheld; VR >= "
                           "0.5; RMSE < continue-rate baseline; coverage3) and T2 (10 km "
                           "leveling blocks; coverage3; RMSE < IDW), registered and scored on "
                           "the round-1 model: both FAILED. The v2 amendments ("
                           + R.DECL_ID + ") are not covered by a registered gate test. "
                           "Monthly component: " + R.MTEST_ID + " M1 "
                           + ("PASS" if (m1 or {}).get("pass") else "FAIL or not scored"))},
        allowed_uses=["after_the_fact_analysis"], forbidden_uses=P1.FORBIDDEN,
        personal_data="none (agency benchmark and station coordinates only)",
        extra={"upstream_gate_rule": (
            "allowed uses are after_the_fact_analysis only whatever the S5 gate: the GNSS "
            "(gnss_clean, S3), InSAR (insar_069A, S4b) and ring-interval (rings_bridged, S1a) "
            "inputs failed their gates; validator use in VR >= 0.5 cells needs the user's "
            "decisions on those gates and a new version"),
            "supersedes": ["subsidence_fused/v1", "subsidence_fused/v2"]})
    errs = manifest.validate_manifest(m_, d)
    if errs:
        raise SystemExit(f"manifest invalid: {errs}")
    manifest.write_manifest(d, m_)
    if dry_dir is not None:
        log(f"DRY RUN written to {d} ({time.time() - t0:.0f} s)")
        return 0
    p = guard.open_product(d, "after_the_fact_analysis")
    log(f"wrote {d}; guard opens after_the_fact_analysis: {p.use}; {time.time() - t0:.0f} s")
    return 0


# ============================================================================ sensitivity
def _fit_variant(data, m, fdict, h, families=F.FAMILIES, v1_operator=False):
    if v1_operator:
        pr = F.make_problem(data, C.KNOTS_FULL, families, masks=m, fdict=fdict)
        return pr, F.fit(pr, h)
    pr = RC.make_problem_v2(data, C.KNOTS_FULL, families, masks=m, fdict=fdict)
    return pr, RC.fit_v2(pr, h)


def sensitivity(data, m, fdict, h, breaks, g, Phi, years, inc, test, flagged, base) -> dict:
    """v2 minus four variants, annual increments 2022-2025 and the 2023-2026 change."""
    st = pd.read_csv(C.GNSS_FULL / "stations.csv").set_index(["family", "station"])
    sx, sy = (st.loc[V.SJES, ["x_twd97", "y_twd97"]].to_numpy(float) / 1000.0)
    dj = np.sqrt((g["cell_xy_km"][:, 0] - sx) ** 2 + (g["cell_xy_km"][:, 1] - sy) ** 2)
    yi = (years >= 2022) & (years <= 2025)
    seg_v2 = data["gnss"]["seg"].copy()
    lev_v2 = data["lev"]["seg"].copy()
    out = {"what": "product (v2) minus variant: annual increments 2022-2025 (mm/yr) and the "
                   "2023-01-01 -> 2026-01-01 change (mm); report only",
           "sjes_cells_3km": int((dj <= 3.0).sum()), "sjes_cells_10km": int((dj <= 10.0).sum())}
    variants = {}
    try:
        # (a) the v1 operator: no GNSS segments, no leveling steps
        data["gnss"]["seg"], data["lev"]["seg"] = 0, 0
        pr, po = _fit_variant(data, m, fdict, h, v1_operator=True)
        variants["v1_operator"] = (pr, po)
        # (b) GNSS breaks at SJES only (leveling steps kept)
        only = {V.SJES: breaks.get(V.SJES, [])}
        data["gnss"]["seg"] = RC.gnss_segments(data["gnss"], only)
        data["lev"]["seg"] = lev_v2
        pr, po = _fit_variant(data, m, fdict, h)
        variants["sjes_breaks_only"] = (pr, po)
        # (c) no leveling steps (GNSS segments kept)
        data["gnss"]["seg"], data["lev"]["seg"] = seg_v2, 0
        pr, po = _fit_variant(data, m, fdict, h)
        variants["no_leveling_steps"] = (pr, po)
        # (d) rings and piles left out
        data["lev"]["seg"] = lev_v2
        pr, po = _fit_variant(data, m, fdict, h, families=("lev", "gnss", "insar"))
        variants["no_rings_piles"] = (pr, po)
    finally:
        data["gnss"]["seg"], data["lev"]["seg"] = seg_v2, lev_v2
    for name, (pr, po) in variants.items():
        F.set_ell(pr, h["ell_km"])
        iv = increments(pr, po.theta, Phi, years)
        cv = change_23_26(pr, po.theta, Phi)
        dlt = inc[yi] - iv[yi]
        dc = test - cv
        r = {"rms_inc_2022_2025": float(np.sqrt(np.mean(dlt ** 2))),
             "max_abs_inc_2022_2025": float(np.max(np.abs(dlt))),
             "cells_inc_moved_gt5mm": int((np.abs(dlt) > 5.0).any(0).sum()),
             "rms_change_2023_2026": float(np.sqrt(np.mean(dc ** 2))),
             "cells_change_moved_gt10mm": int((np.abs(dc) > 10.0).sum()),
             "max_abs_change_2023_2026": float(np.max(np.abs(dc)))}
        for rad in (3.0, 10.0):
            sel = dj <= rad
            r[f"mean_inc_diff_within_{int(rad)}km"] = {
                int(y): float(np.mean(dlt[j][sel])) for j, y in enumerate(years[yi])}
            r[f"mean_change_diff_within_{int(rad)}km"] = float(np.mean(dc[sel]))
        out[name] = r
    out["ring_pile_cells"] = sorted({int(x) for x in P1.ring_pile_cells(g, data["cmp"])})
    out["f"] = base["f"]
    # v2 product vs v1 product files (same cells)
    v1 = np.load(C.DATAPREP / C.PRODUCT / "v1" / "fused_annual.npz", allow_pickle=False)
    dv = inc[yi] - v1["inc"][yi]
    dcv = test - v1["change_2023_2026"]
    out["v2_minus_v1_product"] = {
        "rms_inc_2022_2025": float(np.sqrt(np.mean(dv ** 2))),
        "max_abs_inc_2022_2025": float(np.max(np.abs(dv))),
        "cells_inc_moved_gt5mm": int((np.abs(dv) > 5.0).any(0).sum()),
        "cells_change_moved_gt10mm": int((np.abs(dcv) > 10.0).sum()),
        **{f"mean_inc_diff_within_{int(rad)}km": {
            int(y): float(np.mean(dv[j][dj <= rad])) for j, y in enumerate(years[yi])}
           for rad in (3.0, 10.0)},
        "sjes_window_v1_inc_within_3km": {int(y): float(np.mean(v1["inc"][i][dj <= 3.0]))
                                          for i, y in enumerate(years) if 2021 <= y <= 2025},
        "sjes_window_v2_inc_within_3km": {int(y): float(np.mean(inc[i][dj <= 3.0]))
                                          for i, y in enumerate(years) if 2021 <= y <= 2025}}
    return out


# ============================================================================ files
def fit_window_end_with_inputs(data_end: str) -> str:
    ends = [data_end, "2022-12"]
    for p in UPSTREAM:
        mm = jload(p / "MANIFEST.json")
        if mm and mm.get("fit_window_end"):
            ends.append(str(mm["fit_window_end"]))
    return max(ends)


def input_files(dry: bool = False) -> list[str]:
    from s1_rings import s1const as K1
    files = [C.LEVELING_PANEL, C.RING_STATIONS, C.TWIN_GRID, C.UNITS_V3,
             C.GNSS_FULL / "MANIFEST.json", C.GNSS_FULL / "stations.csv",
             C.GNSS_FULL / "wra" / "monthly.parquet", C.GNSS_FULL / "ngl" / "monthly.parquet",
             C.GNSS_FULL / "wra" / "ncku_monthly.parquet"]
    files += [C.GNSS_FULL / v for v in V.GNSS_DAILY.values()]
    files += [C.INSAR_V2 / "MANIFEST.json", C.INSAR_V2 / "vertical_cells.npz",
              C.PILES / "MANIFEST.json", C.PILES / "printed_monthly.parquet",
              C.RINGS_BRIDGED / "MANIFEST.json", C.RINGS_BRIDGED / "ring_corrections.csv",
              C.GNSS_PRE / "MANIFEST.json", C.GNSS_PRE / "stations.csv",
              C.GNSS_PRE / "wra" / "monthly.parquet", C.GNSS_PRE / "ngl" / "monthly.parquet",
              C.WORK / "insar_runP_cells.npz", C.WORK / "insar_runP_cells.json",
              C.EVIDENCE / "hyper_pre.json", C.EVIDENCE / "hyper_t1.json",
              R.EVIDENCE_R / "hyper_stage2.json",
              C.DATAPREP / "preregistration" / f"{C.PREREG_ID}.json",
              C.DATAPREP / "preregistration" / f"{R.MTEST_ID}.json",
              C.DATAPREP / "preregistration" / f"{R.DECL_ID}.json",
              C.DATAPREP / "preregistration" / "S5.subsidence_fused_rules.v3.json",
              C.DATAPREP / C.PRODUCT / "v2" / "MANIFEST.json",
              C.DATAPREP / C.PRODUCT / "v1" / "MANIFEST.json",
              C.DATAPREP / C.PRODUCT / "v1" / "fused_annual.npz"]
    files += [C.RING_DIR / f"{s}.parquet" for s in K1.SITES]
    for n in ("t1.json", "t2.json", "t3.json", "r2.json"):
        files.append(C.EVIDENCE / n)
    files.append(R.EVIDENCE_R / "m1.json")
    if dry:                                   # the dry run predates some of these files
        files = [f for f in files if f.exists()]
    return [rel_to_repo(f) for f in files]


def build_validation(base, qc, vr_year, vr_test, mask_year, mask_test, years, g, m1, hs,
                     steps, seg_tab, support, mom, months, support_vr) -> dict:
    t1, t2, t3, r2 = (jload(C.EVIDENCE / n) for n in ("t1.json", "t2.json", "t3.json",
                                                       "r2.json"))
    units = json.loads(C.UNITS_V3.read_text())["units"]
    zc = {u["id"]: np.array(u["cells"]) for u in units if u["id"] in
          ("dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3")}
    yl = sorted(set().union(*[set(u["cells"]) for u in units
                              if u["id"] in ("dt.tercile.west", "dt.tercile.mid",
                                             "dt.tercile.east")]))
    groups = {"all_active": np.arange(g["A"]), "changhua_Z1": zc["dt.zone.Z1"],
              "changhua_Z2": zc["dt.zone.Z2"], "changhua_Z3": zc["dt.zone.Z3"],
              "yunlin": np.array(yl)}
    vr, sup = {}, {}
    yrs = np.array([int(mo[:4]) for mo in months])
    for gname, cells in groups.items():
        vr[gname] = {"test_2023_2026": float(mask_test[cells].mean()),
                     **{str(y): float(mask_year[i, cells].mean())
                        for i, y in enumerate(years) if y >= 2020}}
        sup[gname] = {str(y): {"delivered": float(support[yrs == y][:, cells].mean()),
                               "vr_rule": float(support_vr[yrs == y][:, cells].mean())}
                      for y in np.unique(yrs)}
    return {"prereg": C.PREREG_ID, "declaration": R.DECL_ID, "monthly_test": R.MTEST_ID,
            "gate": {"passed": False, "T1": t1.get("pass") if t1 else None,
                     "T2": t2.get("pass") if t2 else None,
                     "note": "registered T1/T2 scored on the round-1 model (v1); the v2 "
                             "amendments are not covered by a registered gate test",
                     "M1_monthly": (m1 or {}).get("pass")},
            "T1": t1, "T2": t2, "T3": t3, "R2": r2, "M1": m1,
            "stage2_hyper": hs, "vr_mask_fraction": vr,
            "monthly_support_fraction": sup,
            "vr_test_quantiles": {q: float(np.quantile(vr_test, q))
                                  for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "stepped_benchmarks": {"scanned": int(len(steps)),
                                   "flagged": int(steps.flagged.sum()),
                                   "list": steps[steps.flagged].to_dict("records")},
            "gnss_segments": {"stations": int(seg_tab.station.nunique()),
                              "segments": int(len(seg_tab)),
                              "stations_with_breaks": sorted(
                                  seg_tab[seg_tab.seg > 0].station.unique().tolist())},
            "hyper_stages": base["stages"], "data_qc_full": qc,
            "sd_label": {"annual": "uncalibrated (T1 0.631, T2 0.940 failed)",
                         "monthly": monthly_sd_label(m1),
                         "monthly_registered_M1_label": (m1 or {}).get("sd_label")}}


def write_card(d, val, h, h2, sens, years, steps, seg_tab, support, months, fwe,
               data_end, support_vr) -> None:
    t1, t2, m1 = val["T1"] or {}, val["T2"] or {}, val["M1"] or {}
    sc = t1.get("scored", {})
    v1op = sens.get("v1_operator", {})
    vv = sens.get("v2_minus_v1_product", {})
    m1s = m1.get("scored", {})
    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S5, repair round; supersedes v1 and v2)", "",
        "v2 and this version hold identical values; this version corrects the monthly sd label "
        "and the card (declaration S5.subsidence_fused_rules.v3).", "",
        "Gate: **FAILED** (pre-registered " + C.PREREG_ID + "; T1 False, T2 False, scored on "
        "the round-1 model; the v2 amendments of " + R.DECL_ID + " are not covered by a "
        "registered gate test). Allowed use: after_the_fact_analysis only, whatever the gate, "
        "because the GNSS (S3), InSAR (S4b) and ring-interval (S1a) inputs failed their gates. "
        "Forbidden: every pre-origin use, calibration and scoring targets, validator use.", "",
        "## What it is",
        "Annual vertical displacement of the Choushui fan on the twin's 2,148 active 1 km "
        "cells, 2012-2026, and a monthly field 2020-01..2025-12 where InSAR supports it, "
        "fused from leveling (2012-2021, L0), GNSS (gnss_clean/full/v2: WRA 2020-2025, NGL "
        "2012-2024, NCKU 2023-2025), InSAR (insar_069A/v2 run F, within-segment changes "
        "only), compaction rings (L0, free offsets per segment) and printed deep piles "
        "(deep_benchmarks_printed/v2). Separable Matern x integrated random walk on velocity at "
        "annual knots on a 4 km lattice; exact Gaussian posterior; pre-origin hyperparameters "
        f"(U7). fit_window_end {fwe} (inputs included, plan 2.3); last observation used "
        f"{data_end}.", "",
        "## Changes from v1 (declaration " + R.DECL_ID + ")",
        f"1. **GNSS offset segments.** A gap > {V.GNSS_GAP_DAYS} d in a station's daily record "
        f"starts a new free offset ({len(seg_tab) - seg_tab.station.nunique()} extra segments "
        f"at {int(seg_tab[seg_tab.seg > 0].station.nunique())} stations). v1 tied every "
        "station across every gap, so SJES's uncorrected ~100 mm offset inside its "
        "2022-12-22..2023-02-24 gap (S1b; plan S3 catalogue) entered v1 as a spurious bowl: "
        "v1 mean increments within 3 km of SJES "
        + ", ".join(f"{y}: {v:.1f}" for y, v in vv.get("sjes_window_v1_inc_within_3km",
                                                        {}).items())
        + " mm/yr; v2 " + ", ".join(f"{y}: {v:.1f}" for y, v in vv.get(
            "sjes_window_v2_inc_within_3km", {}).items()) + " mm/yr. **Do not use v1 near SJES.**",
        f"2. **Stepped benchmarks.** {int(steps.flagged.sum())} of {len(steps)} scanned "
        f"benchmarks carry a step (|z| > {V.STEP_Z:g}, >= {V.STEP_MIN_MM:g} mm) and get a "
        "second offset from the step on (leveling_steps.csv).",
        "3. **mask_year excludes 2026** (flag 6).",
        "4. **Monthly field** (fused_monthly.npz, fused_cells_monthly.parquet): annual trend "
        "incl. the seasonal pair + a stage-2 InSAR monthly anomaly (AR(1) x Matern, fitted on "
        "pre-origin run P: " + ", ".join(f"{k} {v:.3g}" for k, v in h2.items()) + "). Values "
        f"only if M1 passes and only where supported (footprint and plane-free VR >= "
        f"{R.VR_A_MIN:g}, {support_vr.mean():.2f} of the 2020-2025 cell-months): delivered "
        f"{support.mean():.2f}. Registered test "
        + R.MTEST_ID + " (M1, pre-origin, WRA GNSS months withheld): "
        + ("PASS" if m1.get("pass") else "FAIL") + f" on {m1.get('n_scored')} supported "
        f"station-months at {m1.get('stations_scored')} stations: RMSE ratio "
        f"{m1s.get('ratio', float('nan')):.3f} against the annual model, bootstrap P(ratio "
        f">= 1) {(m1s.get('boot_ratio') or {}).get('p_ge_1', float('nan')):.3f} (gate <= "
        f"{R.M1_BOOT_P_MAX:g}). **The skill is marginal and comes from 2021** (ratio by "
        f"year: {m1_years(m1)}); stage 2 found no month-to-month correlation (rho at 0), so "
        "the anomaly is not separated from atmospheric delay by its time behaviour: read "
        "monthly detail outside 2021-like conditions as unconfirmed, possibly atmospheric. "
        "sd: " + monthly_sd_label(m1) + ".",
        "5. Provenance completed; fit_window_end with inputs; every sd labelled uncalibrated "
        "(v3: the monthly value sd too; declaration S5.subsidence_fused_rules.v3).",
        "", "## Hyperparameters (pre-origin, restricted marginal likelihood)",
        ", ".join(f"{k} {v:.3g}" for k, v in h.items()) + ". Several sit at a bound (q 400, "
        "sig_ins 0.2, ell_aps 40, sig_rc 0.05) and the Nelder-Mead stages A, B (re-fit) and D "
        "stopped at their evaluation cap (not converged); stage A's returned vertex is 2.07 "
        "units of -2 log L above the best point it evaluated.", "",
        "## Variance-reduction mask (plan: score the post-2021 field only where VR >= 0.5)",
        "| group | 2022 | 2023 | 2024 | 2025 | 2026 | 2023-2026 change |",
        "|---|---|---|---|---|---|---|"]
    for gname, v in val["vr_mask_fraction"].items():
        lines.append(f"| {gname} | " + " | ".join(f"{v.get(str(y), float('nan')):.2f}"
                                                  for y in range(2022, 2027))
                     + f" | {v['test_2023_2026']:.2f} |")
    lines += [
        "", "The mask is almost non-binding: q sits at its upper bound, so the leveling-only "
        "increment sd explodes after 2021. Do not read mask_year/mask_test as a meaningful "
        "anti-persistence restriction.", "",
        "## Validation",
        f"- T1 (registered, round-1 model; leveling 2020-2021 withheld): RMSE fused "
        f"{sc.get('rmse_fused', float('nan')):.2f} vs continue-rate baseline "
        f"{sc.get('rmse_baseline', float('nan')):.2f} mm (ratio "
        f"{sc.get('ratio_fused_baseline', float('nan')):.3f}); 90 % coverage "
        f"{(sc.get('coverage_fused') or {}).get('coverage', float('nan')):.3f}: FAIL. T1 is "
        "mildly optimistic: its stages B-D and screening come from the main fit, which held "
        "the withheld surveys.",
        f"- T2 (registered; 10 km leveling blocks out, NO buffer): RMSE "
        f"{t2.get('rmse_fused', float('nan')):.2f} vs IDW {t2.get('rmse_idw', float('nan')):.2f} "
        f"mm; coverage {(t2.get('coverage_fused') or {}).get('coverage', float('nan')):.3f} "
        "(its interval excludes 0.90: over-covers): FAIL. Plan 4.3 asks for a >= 10 km buffer "
        "in leveling folds; T2 had none, so its skill describes short-range interpolation.",
        "- T3 and R2 are report only (validation.json). R2 is after the fact and partly "
        "contaminated: run F's SBAS inversion uses 2023+ interferograms within the segment "
        "that spans the origin, and its targets were screened on a full fit.",
        "- Everything here is design-time evidence in A1's sense (no fresh leveling units).", "",
        "## Known limits",
        f"- {SD_NOTE}",
        "- 4 km node lattice: features below about 4 km are smoothed.",
        "- Annual knots plus one constant seasonal pair; intra-annual detail only through the "
        "monthly anomaly, and only where supported. The monthly anomaly's fan-wide plane is "
        "not resolved (InSAR plane per month and GNSS common modes are nuisances).",
        "- InSAR long wavelengths are free per month; fan-wide tilts after 2021 come from GNSS "
        "(now with fewer cross-gap ties) and rings/piles.",
        f"- Rings and piles: fixed depth fractions f (pre-origin regression, nearly collinear "
        f"with a local rate); leaving them out moves 2022-2025 increments by "
        f"{sens.get('no_rings_piles', {}).get('rms_inc_2022_2025', float('nan')):.2f} mm/yr RMS.",
        f"- v2 vs the v1 operator: {v1op.get('rms_inc_2022_2025', float('nan')):.2f} mm/yr RMS "
        f"over cells (2022-2025), {v1op.get('cells_inc_moved_gt5mm', 0)} cells moved > 5 mm in "
        "some year (sensitivity.json).",
        "- 2026 is extrapolated (flag 6). Northern Changhua lies outside the InSAR footprint.",
        "- W3 cannot take an epoch correction from this product: every pre-origin use is "
        "forbidden.", "",
        "## Files",
        "fused_annual.npz, fused_cells_annual.parquet, fused_monthly.npz, "
        "fused_cells_monthly.parquet, gnss_segments.csv, leveling_steps.csv, "
        "hyperparameters.json, sensitivity.json, validation.json, PRODUCT_CARD.md."]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--dry"
                          else None))
