"""WP S5: build results/dataprep/subsidence_fused/v<N> (L2 observation product).

    tools/cpu_slot.sh python -I s5product.py          (after s5hyper main and s5tests t1-t3, r2)

Full data (leveling 2012-2021, gnss_clean/full/v2, insar_069A/v2 run F, printed piles to
2025-11, rings to 2025-05), pre-origin hyperparameters and depth fractions (hyper_pre.json).
Screening: leveling flags of the pre-origin run; GNSS and ring/pile flags from one pass on the
full fit (same rule). Outputs on the twin's active cells:

* fused_annual.npz - disp (16 knots, A): trend displacement at Jan 1 2012..2027 (mm, + =
  sinking, 0 at 2012-01-01) and sd; inc (15 years, A): annual increments 2012..2026 and sd;
  the same sds of the leveling-only fit; vr_year (15, A) = 1 - Var_fused / Var_levonly of each
  annual increment; mask_year = vr_year >= 0.5; vr_test / mask_test for the 2023-01-01 ->
  2026-01-01 change; seasonal pair (cos, sin) and sd; flags (3 = fused estimate, 6 = year 2026,
  extrapolated beyond the record); grid hash.
* fused_cells_annual.parquet - long table (cell, year, value, sd, flag, vr, in_mask, ...).
* hyperparameters.json, sensitivity.json (rings and piles left out), validation.json,
  PRODUCT_CARD.md.
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
import s5hyper as Hy  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import grid_hash  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5hyper.py", "s5tests.py",
        "s5insarP.py", "s5looks.py", "s5prereg.py", "s5product.py"]
FORBIDDEN = ["calibration_target", "datum_fit", "baseline", "initial_conditions", "idw_field",
             "prior", "prior_width_input", "model_design", "selection", "forcing_frozen",
             "design_time_evidence", "scoring_target", "prospective_scorer",
             "forcing_recorded_arm", "statistical_layer_history", "validator"]


def data_end_month(pr) -> str:
    """Last month of any observation used by the fit (decimal years -> YYYY-MM; a pile
    cumulative value at a month start belongs to the month before)."""
    ts = [pr.lev.t.to_numpy(), pr.gnss.t.to_numpy(), pr.cmp.t.to_numpy(),
          np.array([gr["t"] for gr in pr.insar["groups"]])]
    t = max(float(np.max(a)) for a in ts if len(a)) - 1e-6
    y = int(np.floor(t))
    return f"{y}-{int(np.floor((t - y) * 12.0)) + 1:02d}"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cell_stats(pr, post, Phi, a, c):
    """Mean and sd over cells of sum_k a_k u-block_k + c mu_v (posterior with cov)."""
    K, m, L = pr.K, pr.m, pr.L
    th = post.theta
    Z = th[:L].reshape(K, m)
    mean = Phi @ (a @ Z) + c * th[L]
    cov = post.cov
    nz = np.flatnonzero(a)
    S = np.zeros((m, m))
    s_mu = np.zeros(m)
    for k in nz:
        s_mu += a[k] * cov[k * m:(k + 1) * m, L]
        for j in nz:
            S += a[k] * a[j] * cov[k * m:(k + 1) * m, j * m:(j + 1) * m]
    var = np.einsum("ij,jk,ik->i", Phi, S, Phi) + c * c * cov[L, L] + 2 * c * (Phi @ s_mu)
    return mean, np.sqrt(np.clip(var, 0, None))


def jload(name: str):
    p = C.EVIDENCE / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.time()
    base = jload("hyper_pre.json")
    h, fdict = base["hyper"], base["fdict"]
    data, qc = Hy.load("full")
    g = data["grid"]
    flagged = {"lev": base["flagged"]["lev"]}
    m0 = Hy.masks_from(data, flagged)
    pr = F.make_problem(data, C.KNOTS_FULL, F.FAMILIES, masks=m0, fdict=fdict)
    post = F.fit(pr, h)
    flagged["gnss"] = F.outliers(pr, post, h, "gnss")
    flagged["cmp"] = F.outliers(pr, post, h, "cmp")
    del post
    m = Hy.masks_from(data, flagged)
    pr = F.make_problem(data, C.KNOTS_FULL, F.FAMILIES, masks=m, fdict=fdict)
    post = F.fit(pr, h, want_cov=True)
    fwe = data_end_month(pr)
    log(f"fused fit with covariance: N {pr.N}, data to {fwe} ({time.time() - t0:.0f} s)")
    F.set_ell(pr, h["ell_km"])
    Phi = pr.lat.weights(g["cell_xy_km"])
    tb = pr.tb
    nk = tb.T
    A = g["A"]
    years = np.array(C.PRODUCT_YEARS)
    disp = np.zeros((nk, A))
    disp_sd = np.zeros((nk, A))
    inc = np.zeros((years.size, A))
    inc_sd = np.zeros((years.size, A))
    for k in range(nk):
        disp[k], disp_sd[k] = cell_stats(pr, post, Phi, tb.knot_row(k), tb.knots[k] - C.T_REF)
    for i in range(years.size):
        a = tb.knot_row(i + 1) - tb.knot_row(i)
        inc[i], inc_sd[i] = cell_stats(pr, post, Phi, a, 1.0)
    a_test = tb.knot_row(2026 - 2012) - tb.knot_row(2023 - 2012)
    test, test_sd = cell_stats(pr, post, Phi, a_test, 3.0)
    ec = np.zeros(tb.K)
    ec[tb.ic] = 1.0
    es = np.zeros(tb.K)
    es[tb.isn] = 1.0
    s_cos, s_cos_sd = cell_stats(pr, post, Phi, ec, 0.0)
    s_sin, s_sin_sd = cell_stats(pr, post, Phi, es, 0.0)
    th = post.theta
    glob = {n: {"mean": float(th[pr.L + i]), "sd": float(np.sqrt(post.cov[pr.L + i, pr.L + i]))}
            for i, n in enumerate(pr.gnames) if not n.startswith(("ioff:", "goff:", "gcm:"))}
    n_par = pr.N
    del post
    # ---- leveling-only fit (same hyperparameters)
    prl = F.make_problem(data, C.KNOTS_FULL, ("lev",), masks=m)
    postl = F.fit(prl, h, want_cov=True)
    F.set_ell(prl, h["ell_km"])
    inc_sd_l = np.zeros_like(inc_sd)
    inc_l = np.zeros_like(inc)
    for i in range(years.size):
        a = tb.knot_row(i + 1) - tb.knot_row(i)
        inc_l[i], inc_sd_l[i] = cell_stats(prl, postl, Phi, a, 1.0)
    test_l, test_sd_l = cell_stats(prl, postl, Phi, a_test, 3.0)
    del postl
    vr_year = 1.0 - inc_sd ** 2 / inc_sd_l ** 2
    vr_test = 1.0 - test_sd ** 2 / test_sd_l ** 2
    mask_year = vr_year >= C.VR_MIN
    mask_test = vr_test >= C.VR_MIN
    flag = np.where(years[:, None] >= C.EXTRAPOLATED_FROM, 6, 3).astype(np.uint8) * \
        np.ones((1, A), np.uint8)
    dflag = np.where(tb.knots[:, None] > 2026.0, 6, 3).astype(np.uint8) * np.ones((1, A),
                                                                                  np.uint8)
    dflag[0] = 0
    log(f"leveling-only fit done ({time.time() - t0:.0f} s)")
    # ---- sensitivity: rings and piles left out
    prs = F.make_problem(data, C.KNOTS_FULL, ("lev", "gnss", "insar"), masks=m)
    posts = F.fit(prs, h)
    F.set_ell(prs, h["ell_km"])
    Zs = posts.theta[:prs.L].reshape(prs.K, prs.m)
    inc_nc = np.stack([Phi @ ((tb.knot_row(i + 1) - tb.knot_row(i)) @ Zs) + posts.theta[prs.L]
                       for i in range(years.size)])
    del posts
    ring_cells = sorted({int(x) for x in ring_pile_cells(g, data["cmp"])})
    yi = (years >= 2022) & (years <= 2025)
    sens = {"what": "fused field without rings and piles minus the product, annual increments "
                    "2022-2025 (mm/yr)",
            "rms_all_cells": float(np.sqrt(np.mean((inc_nc[yi] - inc[yi]) ** 2))),
            "rms_ring_pile_cells": float(np.sqrt(np.mean((inc_nc[yi][:, ring_cells]
                                                          - inc[yi][:, ring_cells]) ** 2))),
            "max_abs_ring_pile_cells": float(np.max(np.abs(inc_nc[yi][:, ring_cells]
                                                           - inc[yi][:, ring_cells]))),
            "ring_pile_cells": ring_cells, "f": base["f"]}
    # ---- write
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
        change_2023_2026_sd_levonly=test_sd_l.astype(np.float32),
        vr_test=vr_test.astype(np.float32), mask_test=mask_test,
        seasonal_cos=s_cos.astype(np.float32), seasonal_cos_sd=s_cos_sd.astype(np.float32),
        seasonal_sin=s_sin.astype(np.float32), seasonal_sin_sd=s_sin_sd.astype(np.float32),
        nx=z["nx"], ny=z["ny"], dx=z["dx"], x0=z["x0"], y0=z["y0"], grid_hash=np.array(gh),
        units=np.array("mm (displacement, + = sinking) and mm/yr (annual increments); "
                       "relative to 2012-01-01; cells in np.nonzero(mask) order"),
        sd_note=np.array("posterior sd of the 4 km-lattice field (calibration: validation.json "
                         "T1/T2 coverage); a point (benchmark) adds sig_rb * years and "
                         "sig_lev"),
        flag_note=np.array("3 = fused estimate from several instruments (plan 2.4: filled using "
                           "other sites or instruments); 6 = extrapolated beyond the record "
                           "(year 2026, knot 2027-01-01); disp at 2012-01-01 = 0 by definition "
                           "(flag 0)"))
    rows = []
    for i, y in enumerate(years):
        rows.append(pd.DataFrame({
            "cell": np.arange(A), "year": int(y), "value": inc[i], "sd": inc_sd[i],
            "flag": flag[i], "vr": vr_year[i], "in_mask": mask_year[i],
            "sd_levonly": inc_sd_l[i], "disp_end": disp[i + 1], "disp_end_sd": disp_sd[i + 1],
            "method": "s5 space-time fusion (separable Matern x IRW velocity; leveling, GNSS, "
                      "InSAR, rings, piles)", "product_version": f"{C.PRODUCT}/{d.name}",
            "source_ids": "leveling L0; gnss_clean/full/v2; insar_069A/v2; rings L0; "
                          "deep_benchmarks_printed/v2"}))
    pd.concat(rows, ignore_index=True).to_parquet(d / "fused_cells_annual.parquet",
                                                  index=False)
    F.dump({"hyper": h, "hyper_fit_window_end": "2022-12", "f": base["f"],
            "fdict": fdict, "screening_full": {k: len(v) for k, v in flagged.items()},
            "screened_rids": flagged, "globals": glob, "n_parameters": n_par,
            "lattice_nodes": pr.m, "knots": list(tb.knots)}, d / "hyperparameters.json")
    F.dump(sens, d / "sensitivity.json")
    val = build_validation(base, qc, vr_year, vr_test, mask_year, mask_test, years, g)
    F.dump(val, d / "validation.json")
    write_card(d, val, h, base, sens, years, mask_year, mask_test, A)
    gate = val["gate"]
    inputs = input_files(data)
    m_ = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2",
        inputs=inputs, code=[rel_to_repo(HERE / c) for c in CODE] + [
            "dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
            "dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
            "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/prereg3.py"],
        parameters={"hyper": h, "hyper_fit_window_end": "2022-12", "fdict": fdict,
                    "lattice_km": C.LATTICE_KM, "knots": list(tb.knots),
                    "vr_min": C.VR_MIN, "prereg": C.PREREG_ID,
                    "screening_full": {k: len(v) for k, v in flagged.items()}},
        fit_window_end=fwe, validation_report="validation.json",
        gate={"passed": bool(gate["passed"]),
              "criteria": "S5.subsidence_fusion.v1: T1 (2020-2021 leveling withheld; VR >= 0.5; "
                          "RMSE < continue-rate baseline; coverage3) and T2 (10 km leveling "
                          "blocks; coverage3; RMSE < IDW), pre-origin data only (U7)"},
        allowed_uses=["after_the_fact_analysis"], forbidden_uses=FORBIDDEN,
        personal_data="none (agency benchmark and station coordinates only)",
        extra={"upstream_gate_rule": (
            "allowed uses are after_the_fact_analysis only whatever the S5 gate: the GNSS "
            "(gnss_clean, S3), InSAR (insar_069A, S4b) and ring-interval (rings_bridged, S1a) "
            "inputs failed their gates; validator use in VR >= 0.5 cells needs the user's "
            "decisions on those gates and a new version"),
            "supersedes": None})
    errs = manifest.validate_manifest(m_, d)
    if errs:
        raise SystemExit(f"manifest invalid: {errs}")
    manifest.write_manifest(d, m_)
    p = guard.open_product(d, "after_the_fact_analysis")
    log(f"wrote {d} (gate {gate['passed']}); guard opens after_the_fact_analysis: "
        f"{p.use}; {time.time() - t0:.0f} s")
    return 0


def ring_pile_cells(g, cmp: pd.DataFrame) -> np.ndarray:
    from s5data import cell_of
    xy = cmp.groupby("sensor")[["x_km", "y_km"]].first().to_numpy() * 1000.0
    c = cell_of(g, xy[:, 0], xy[:, 1])
    return c[c >= 0]


def input_files(data) -> list[str]:
    from s1_rings import s1const as K1
    files = [C.LEVELING_PANEL, C.RING_STATIONS, C.TWIN_GRID,
             C.GNSS_FULL / "MANIFEST.json", C.GNSS_FULL / "stations.csv",
             C.GNSS_FULL / "wra" / "monthly.parquet", C.GNSS_FULL / "ngl" / "monthly.parquet",
             C.GNSS_FULL / "wra" / "ncku_monthly.parquet",
             C.INSAR_V2 / "MANIFEST.json", C.INSAR_V2 / "vertical_cells.npz",
             C.PILES / "MANIFEST.json", C.PILES / "printed_monthly.parquet",
             C.RINGS_BRIDGED / "MANIFEST.json", C.RINGS_BRIDGED / "ring_corrections.csv",
             C.GNSS_PRE / "MANIFEST.json", C.WORK / "insar_runP_cells.npz",
             C.EVIDENCE / "hyper_pre.json", C.EVIDENCE / "hyper_t1.json",
             C.DATAPREP / "preregistration" / f"{C.PREREG_ID}.json"]
    files += [C.RING_DIR / f"{s}.parquet" for s in K1.SITES]
    for n in ("t1.json", "t2.json", "t3.json", "r2.json"):
        if (C.EVIDENCE / n).exists():
            files.append(C.EVIDENCE / n)
    return [rel_to_repo(f) for f in files]


def build_validation(base, qc, vr_year, vr_test, mask_year, mask_test, years, g) -> dict:
    t1, t2, t3, r2 = (jload(n) for n in ("t1.json", "t2.json", "t3.json", "r2.json"))
    units = json.loads(C.UNITS_V3.read_text())["units"]
    zc = {u["id"]: np.array(u["cells"]) for u in units if u["id"] in
          ("dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3")}
    yl = sorted(set().union(*[set(u["cells"]) for u in units
                              if u["id"] in ("dt.tercile.west", "dt.tercile.mid",
                                             "dt.tercile.east")]))
    groups = {"all_active": np.arange(g["A"]), "changhua_Z1": zc["dt.zone.Z1"],
              "changhua_Z2": zc["dt.zone.Z2"], "changhua_Z3": zc["dt.zone.Z3"],
              "yunlin": np.array(yl)}
    vr = {}
    for gname, cells in groups.items():
        vr[gname] = {"test_2023_2026": float(mask_test[cells].mean()),
                     **{str(y): float(mask_year[i, cells].mean())
                        for i, y in enumerate(years) if y >= 2020}}
    passed = bool(t1 and t2 and t1.get("pass") and t2.get("pass"))
    return {"prereg": C.PREREG_ID, "gate": {"passed": passed,
                                            "T1": t1.get("pass") if t1 else None,
                                            "T2": t2.get("pass") if t2 else None},
            "T1": t1, "T2": t2, "T3": t3, "R2": r2, "vr_mask_fraction": vr,
            "vr_test_quantiles": {q: float(np.quantile(vr_test, q))
                                  for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "hyper_stages": base["stages"], "data_qc_full": qc}


def write_card(d, val, h, base, sens, years, mask_year, mask_test, A) -> None:
    t1, t2 = val["T1"] or {}, val["T2"] or {}
    sc = t1.get("scored", {})
    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S5)", "",
        f"Gate: **{'PASSED' if val['gate']['passed'] else 'FAILED'}** "
        f"(pre-registered {C.PREREG_ID}; T1 {t1.get('pass')}, T2 {t2.get('pass')}). Allowed use: "
        "after_the_fact_analysis only, whatever the gate, because the GNSS (S3), InSAR (S4b) and "
        "ring-interval (S1a) inputs failed their gates. Forbidden: every pre-origin use, "
        "calibration and scoring targets, and (until the upstream decisions and a new version) "
        "validator use.", "",
        "## What it is",
        "Annual vertical displacement of the Choushui fan on the twin's 2,148 active 1 km cells, "
        "2012-2026, fused from leveling (2012-2021, L0), GNSS (gnss_clean/full/v2: WRA 2020-2025, "
        "NGL 2012-2024, NCKU 2023-2025), InSAR (insar_069A/v2 run F, 2020-01..2025-12, "
        "within-segment changes only), compaction rings (L0, 14 sites, free offsets per "
        "segment) and printed deep piles (deep_benchmarks_printed/v2, 7 sensors). Model: "
        "separable Matern (nu 3/2) x integrated random walk on velocity at annual knots, "
        f"predictive-process basis on a {C.LATTICE_KM:g} km lattice, instrument operators in "
        "s5fit.py, exact Gaussian posterior. Hyperparameters and depth fractions from "
        "pre-origin data only (U7).", "",
        "## Hyperparameters (pre-origin, restricted marginal likelihood)",
        ", ".join(f"{k} {v:.3g}" for k, v in h.items()), "",
        "## Variance-reduction mask (plan: score the post-2021 field only where VR >= 0.5)",
        "VR = 1 - Var(fused) / Var(leveling-only, same hyperparameters) of each annual "
        "increment (mask_year) and of the 2023-01-01 -> 2026-01-01 change (mask_test). Share of "
        "cells in the mask:",
        "| group | 2022 | 2023 | 2024 | 2025 | 2026 | 2023-2026 change |", "|---|---|---|---|---|---|---|"]
    for gname, v in val["vr_mask_fraction"].items():
        lines.append(f"| {gname} | " + " | ".join(f"{v.get(str(y), float('nan')):.2f}"
                                                  for y in range(2022, 2027))
                     + f" | {v['test_2023_2026']:.2f} |")
    lines += [
        "", "## Validation (pre-origin, registered)",
        f"- T1 (leveling 2020-2021 withheld, the mirror of the real gap): scored "
        f"{t1.get('n_benchmarks_scored')} benchmarks / {t1.get('n_targets_scored')} targets "
        f"with VR >= 0.5; RMSE fused {sc.get('rmse_fused', float('nan')):.2f} mm vs "
        f"continue-rate baseline {sc.get('rmse_baseline', float('nan')):.2f} mm (ratio "
        f"{sc.get('ratio_fused_baseline', float('nan')):.3f}); 90 % coverage "
        f"{(sc.get('coverage_fused') or {}).get('coverage', float('nan')):.3f} "
        f"(gate-3 {(sc.get('coverage_fused') or {}).get('pass')}).",
        f"- T2 (10 km leveling blocks out): {t2.get('n_targets')} increments; RMSE fused "
        f"{t2.get('rmse_fused', float('nan')):.2f} vs IDW {t2.get('rmse_idw', float('nan')):.2f} "
        f"mm (ratio {t2.get('ratio_fused_idw', float('nan')):.3f}); coverage "
        f"{(t2.get('coverage_fused') or {}).get('coverage', float('nan')):.3f} (gate-3 "
        f"{(t2.get('coverage_fused') or {}).get('pass')}).",
        "- T3 (family out) and R2 (post-2022 hold-out, after the fact) are report only: see "
        "validation.json and the WP report.", "",
        "## Known limits",
        f"- Spatial resolution: a {C.LATTICE_KM:g} km node lattice; features below about 4 km "
        "are smoothed. A point (benchmark) differs from the field by a random velocity "
        f"(sd {h['sig_rb']:.2f} mm/yr) and survey noise (sd {h['sig_lev']:.2f} mm).",
        "- Annual knots plus one constant seasonal harmonic; no monthly field (plan: monthly "
        "where InSAR supports it - not delivered).",
        "- InSAR long wavelengths are free (a plane per month is projected out), so fan-wide "
        "tilts after 2021 come from GNSS and rings only.",
        "- GNSS station rates are nuisance parameters (sd sig_d); GNSS mostly constrains changes "
        "of velocity, not the level of the rate.",
        f"- Rings and piles: fixed depth fractions f (pre-origin regression); leaving them out "
        f"changes the 2022-2025 annual increments by {sens['rms_all_cells']:.2f} mm/yr RMS "
        f"over all cells and {sens['rms_ring_pile_cells']:.2f} mm/yr at their cells "
        "(sensitivity.json).",
        "- 2026 is extrapolated (flag 6). Northern Changhua lies outside the InSAR footprint.",
        "- sd is calibrated only by the T1/T2 coverage checks on pre-origin leveling; after "
        "2022 it is uncalibrated.", "",
        "## Files",
        "fused_annual.npz, fused_cells_annual.parquet, hyperparameters.json, sensitivity.json, "
        "validation.json, PRODUCT_CARD.md (this file)."]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
