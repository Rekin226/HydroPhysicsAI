"""P2d heads evidence (plan WP P2d, 'Heads'): rerun of the physics scout's post-2016 ramp test
(``_pilots/physics_inversion_hooks/e_changhua_inversion.py``, section 'post-2016 Yunlin
unregistered pumping') from its archived pre-origin intermediates, with the standard error
the plan asks for: inflated by the block over-dispersion (chi2/df) and replaced by a
site-cluster bootstrap. Frozen cwa theta (fit_full of batch 8, provisional), heads 2012-2022
(pre-origin). Design-time evidence only, labelled uncalibrated; it does not weight the members.

The pilot's code is not edited: inv_lib is imported read-only from the pilot directory without
writing bytecode, and the post-2016 section is transcribed below.

Usage: python -I d5_heads_ramp.py <out_json>
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import looks_def as LK  # noqa: E402

N_BOOT = 500
SEED = 2026


def load_inv_lib():
    spec = importlib.util.spec_from_file_location("inv_lib_pilot",
                                                  D.PILOT_PHYS / "inv_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(out: str) -> None:
    LK.record(
        "P2def-d5-heads-ramp",
        units={"P2def:d5_yunlin_calibration_wells": "the 84 Yunlin calibration wells of the "
                                                    "physics pilot (inputs_cwa.npz), heads "
                                                    "2012-02..2022-12, residuals against the "
                                                    "cwa fit_full hindcast"},
        scope="heads:2012-02..2022-12", purpose="design_time",
        note="Rerun of the physics scout's post-2016 ramp test with chi2/df-inflated and "
             "site-bootstrap standard errors (plan P2d 'Heads'); design-time evidence, "
             "uncalibrated, never weights the members.",
        members={"P2def:d5_yunlin_calibration_wells": {"ids": sorted(
            str(s) for s, c in zip(np.load(D.PILOT_PHYS / "inputs_cwa.npz")["sids"],
                                   np.load(D.PILOT_PHYS / "inputs_cwa.npz")["well_county"],
                                   strict=True) if str(c) == "雲林縣")}})
    il = load_inv_lib()
    d = np.load(D.PILOT_PHYS / "inputs_cwa.npz")
    hc = np.load(D.PILOT_PHYS / "hindcast_cwa.npz")
    K = np.load(D.PILOT_PHYS / "K_pump_f32.npy").astype(np.float64)
    NT = il.NT
    cen = il.centroids(d)
    cc = d["cell_county"]
    yl = cc == "雲林縣"
    oi, ol = d["obs_idx"], d["obs_layer"]
    wch = d["well_county"] == "彰化縣"
    obs = d["obs_h"][:, 1:]
    H = hc["H"]
    pred = H[ol, oi, 1:]
    Qf = hc["Q"]
    Wsp = hc["W"]
    Es_cls = np.einsum("ij,cjt->cit", Wsp, d["E_cls"][:, :, 1:])
    Es_tot = Es_cls.sum(0)
    classes = list(d["classes"])
    Q_irr = Qf * Es_cls[classes.index("irrigation")] / np.where(Es_tot > 0, Es_tot, 1)
    dates = pd.date_range("2012-02-01", periods=NT, freq="MS")
    resid = obs - pred
    phi = float(il.ar1_phi(resid))
    anom_sd = np.array([np.nanstd(r - np.nanmean(r)) for r in resid])
    s0 = float(np.median(anom_sd[~wch]))
    phi_irr = Q_irr[yl].sum(0)
    phi_irr /= phi_irr.mean()
    cut = np.flatnonzero(dates >= "2016-07-01")[0]
    ramp = np.clip((np.arange(NT) - cut) / (NT - cut), 0, 1)
    By, laby, pairsy = il.make_blocks(cen, yl, cells_per_block=150)
    nby = By.shape[1]
    Yy = il.block_responses(K, By, np.stack([ramp * phi_irr]))
    sol = il.solve(resid, Yy, ~wch, phi, sigma0=s0, J=1, bounds_pos=False)
    th = sol["theta"]
    C = sol["cov_theta"]
    sd = np.sqrt(np.diag(C))
    one = np.ones(nby)
    tot = float(th.sum())
    tot_sd = float(np.sqrt(one @ C @ one))
    Ci = np.linalg.inv(C)
    chi2_zero = float(th @ Ci @ th)
    n_cells = np.array([(laby == b).sum() for b in range(nby)], dtype=float)
    alpha = float((n_cells @ Ci @ th) / (n_cells @ Ci @ n_cells))
    r = th - alpha * n_cells
    chi2_common = float(r @ Ci @ r)
    f_zero = float(np.sqrt(max(1.0, chi2_zero / nby)))
    f_common = float(np.sqrt(max(1.0, chi2_common / (nby - 1))))
    # site-cluster bootstrap of the Yunlin wells (sites: coordinates rounded to 100 m)
    yw = np.flatnonzero(~wch)
    xy = np.round(d["well_xy"][yw] / 100.0)
    site_id = pd.Series([f"{a:.0f}_{b:.0f}" for a, b in xy]).factorize()[0]
    sites = np.unique(site_id)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(sites, len(sites), replace=True)
        idx = np.concatenate([yw[site_id == s] for s in pick])
        sb = il.solve(resid[idx], Yy[idx], np.ones(len(idx), bool), phi, sigma0=s0, J=1,
                      bounds_pos=False)
        boots.append(float(sb["theta"].sum()))
    boots = np.array(boots)
    census_mean = float(Qf[yl].sum(0).mean())
    res = {
        "pilot_reproduction": {"total_m3d_end2022": tot, "total_sd_formal": tot_sd,
                               "pilot_json_total": 29.9816767545708,
                               "pilot_json_sd": 34387.705151002665},
        "n_blocks": int(nby), "theta_blocks_m3d": th.tolist(), "sd_blocks": sd.tolist(),
        "blocks_beyond_2sd_pos": int((th > 2 * sd).sum()),
        "blocks_beyond_2sd_neg": int((th < -2 * sd).sum()),
        "chi2_about_zero": chi2_zero, "df_zero": int(nby),
        "chi2_about_uniform_rate": chi2_common, "df_uniform": int(nby - 1),
        "inflation_sqrt_chi2_df_zero": f_zero, "inflation_sqrt_chi2_df_uniform": f_common,
        "total_sd_inflated_zero": tot_sd * f_zero, "total_sd_inflated_uniform": tot_sd * f_common,
        "bootstrap": {"n": N_BOOT, "sd": float(boots.std(ddof=1)),
                      "q05": float(np.quantile(boots, 0.05)),
                      "q95": float(np.quantile(boots, 0.95)), "n_sites": int(len(sites))},
        "sigma0": s0, "sigma_hat": sol["sigma_hat"], "phi_ar1": phi,
        "yunlin_census_abstraction_mean_m3d": census_mean,
        "label": "design-time evidence, uncalibrated; frozen cwa theta (provisional); does "
                 "not weight the P2d members",
        "code": D.p2def_code(),
    }
    D.save_json(res, out)
    print({k: v for k, v in res.items() if k not in ("theta_blocks_m3d", "sd_blocks", "code")})


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main(sys.argv[1])
