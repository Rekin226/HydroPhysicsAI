"""WP H4 repair round: registered run of H4.subgrid_operator.v2 (sub-grid operator,
cell-mean kernel subtracted; h4v2lib.PRIMARY).

Calls harness.prereg3.assert_can_score first (refuses if the registration postdates a look at
its test data or a registered input changed; logs the scoring look), then computes:

* the registered F test on the held-out wells (h4v2run): multipliers fitted on the 158
  calibration wells' 2012-02..2019-12 residuals, applied to the held-out wells, high-swing MSE
  reduction, both placebo families (199 seeded draws each; the same pump shuffle and the same
  donor draws for the calibration and the held-out wells), no-harm check;
* the declared conservative conditions on the v1 unit (158 wells, T1 held-out years and T2 A1
  site folds, 2020-22; post-look, design-time evidence): Q4 excess falls and Q1-Q3 no-harm;
* reported only: placebo p on the v1 unit, F2 (held-out wells with 2020-22 data, held-out
  years), sensitivity members (no placebos), Monte-Carlo sd of every p, concentration of the
  reductions.

Writes results/dataprep/reports/H4_evidence/repair/registered/{scores.json, arrays.npz}.
Usage: python r2_registered.py
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import h4v2lib as V  # noqa: E402
import h4v2run as RN  # noqa: E402
from common.paths import DATAPREP  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "H4.subgrid_operator.v2"
OUT = DATAPREP / "reports/H4_evidence/repair/registered"
SENSITIVITY = {
    "yunlin_only": {"counties": ("yunlin",)},
    "r_max_3km": {"r_max_m": 3000.0},
    "centroid_reference": {"kernel_mode": "centroid"},
    "one_multiplier": {"n_mult": 1},
    "recorded_arm": {"arm": "recorded"},
    "t_profile_up": {"t_profile": (1.0, 2.0, 4.0)},
}


def _q(x):
    return [float(v) for v in np.asarray(x, float).ravel()]


def fit_loss(cal: L.Data, X, m: dict, a) -> float:
    """The NNLS objective of h4lib.fit_multipliers at ``a`` (all fit wells)."""
    tot = 0.0
    for i in np.flatnonzero(m["fit_ok"]):
        t = m["fit"][i]
        y = cal.obs[i, t] - cal.h_cell[i, t]
        A = np.stack([-x[i, t] for x in X], axis=1)
        y = y - y.mean()
        A = A - A.mean(axis=0)
        r = y - A @ np.asarray(a[: A.shape[1]], float)
        tot += float((r ** 2).sum() / t.sum())
    return tot


def p_value(real: float, plac) -> dict:
    v = np.asarray(plac, float)
    n_ge = int(np.sum(v >= real))
    p = (1 + n_ge) / (1 + v.size)
    return {"p": p, "n_ge": n_ge, "n": int(v.size), "median": float(np.median(v)),
            "max": float(v.max()), "q95": float(np.quantile(v, 0.95)),
            "mc_sd": math.sqrt(p * (1 - p) / (1 + v.size))}


def concentration(base_mse, cor_mse, sel) -> dict:
    d = (base_mse - cor_mse)[sel]
    d = d[np.isfinite(d)]
    if not d.size:
        return {}
    pos = np.sort(d[d > 0])[::-1]
    tot = float(d.sum())
    return {"n": int(d.size), "n_improved": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
            "total_m2": tot, "top5_m2": _q(pos[:5]),
            "top5_share_of_total": float(pos[:5].sum() / tot) if tot > 0 else float("nan")}


def build_members(cfg, cal0: L.Data, with_heads: bool):
    grid, cal, fr, notes = RN.setup(cfg, with_heads, cal=cal0)
    return grid, cal, fr, notes


def run(cfg, gate=None) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    grid, cal, fr, sel_notes = RN.setup(cfg, with_heads=True)
    tabs = V.make_tables(cfg, grid)
    mcal = L.masks(cal)
    qcal, sdcal, cuts = L.quartiles(cal, mcal)
    if not np.allclose(cuts, V.V1_QUARTILE_CUTS, rtol=0, atol=1e-12):
        raise RuntimeError(f"158-well quartile cuts {cuts} differ from v1")
    climcal = L.climatology(cal, mcal)
    Xc, nc = RN.regressors(cfg, grid, cal, tabs)
    Xf, nf = RN.regressors(cfg, grid, fr, tabs)
    sup_c, sup_f = nc > 0, nf > 0
    fm = RN.f_masks(fr)
    fb0 = RN.f_mse(fr, fm, None)
    cls = RN.f_classes(fb0["sd"], fm["ok"], sup_f, V.V1_QUARTILE_CUTS)
    real = RN.evaluate(cfg, cal, fr, Xc, Xf, mcal, qcal, climcal, fm, cls, sup_f)
    # placebos
    plac = {f: {"f": [], "t1": [], "t2": [], "a": []} for f in ("shuffle", "distant")}
    for r in range(cfg.n_placebo):
        rng = np.random.default_rng(cfg.seed_shuffle + r)
        pxy = L.shuffle_xy(cal, rng)
        xc = RN.regressors(cfg, grid, cal, tabs, pump_xy=pxy)[0]
        xf = RN.regressors(cfg, grid, fr, tabs, pump_xy=pxy)[0]
        e = RN.evaluate(cfg, cal, fr, xc, xf, mcal, qcal, climcal, fm, cls, sup_f)
        for k, v in (("f", e["f"]["high"]["mse_reduction_m2"]), ("t1", L.q4_reduction(e["t1"])),
                     ("t2", L.q4_reduction(e["t2"])), ("a", _q(e["a"]))):
            plac["shuffle"][k].append(v)
        rng = np.random.default_rng(cfg.seed_distant + r)
        cc = V.distant_centres(cal, cal.pump_xy, rng, sup_c, cfg)
        cf = V.distant_centres(fr, cal.pump_xy, rng, sup_f, cfg)
        xc = RN.regressors(cfg, grid, cal, tabs, centres=cc, support=sup_c)[0]
        xf = RN.regressors(cfg, grid, fr, tabs, centres=cf, support=sup_f)[0]
        e = RN.evaluate(cfg, cal, fr, xc, xf, mcal, qcal, climcal, fm, cls, sup_f)
        for k, v in (("f", e["f"]["high"]["mse_reduction_m2"]), ("t1", L.q4_reduction(e["t1"])),
                     ("t2", L.q4_reduction(e["t2"])), ("a", _q(e["a"]))):
            plac["distant"][k].append(v)
    t_pl = time.perf_counter() - t0
    # ---- registered criteria ---------------------------------------------------------------
    fh = real["f"]["high"]
    crit = []
    crit.append({"id": "F1", "criterion": "held-out wells (fresh for H4): high-swing MSE falls "
                                          f"(high-swing = Q{'+Q'.join(map(str, real['f']['high_q']))}"
                                          " by the v1 sd cuts)",
                 "threshold": "sum over high-swing held-out wells of (base MSE - corrected MSE) "
                              "> 0 (strict)",
                 "measured": f"{fh['mse_reduction_m2']:.4f} m2 over {fh['n_wells']} wells "
                             f"({fh['n_supported']} with a kernel); RMSE {fh['rmse_base']:.4f}"
                             f" -> {fh['rmse_corrected']:.4f} m",
                 "value": fh["mse_reduction_m2"], "passed": bool(fh["mse_reduction_m2"] > 0)})
    pvals = {}
    for k, fam, name in (("F2", "shuffle", "pump positions shuffled within the county"),
                         ("F3", "distant", "only pumps more than 15 km away (donor "
                                           "neighbourhoods)")):
        pv = p_value(fh["mse_reduction_m2"], plac[fam]["f"])
        pvals[fam] = pv
        crit.append({"id": k, "criterion": f"held-out wells: the real kernel's high-swing MSE "
                                           f"reduction beats the placebo ({name})",
                     "threshold": f"p = (1 + #placebo >= real) / (1 + {cfg.n_placebo}) <= "
                                  f"{cfg.alpha}",
                     "measured": f"real {fh['mse_reduction_m2']:.4f} m2; placebo median "
                                 f"{pv['median']:.4f}, q95 {pv['q95']:.4f}, max {pv['max']:.4f};"
                                 f" {pv['n_ge']} of {pv['n']} >= real; p = {pv['p']:.4f} "
                                 f"(MC sd {pv['mc_sd']:.4f})",
                     "value": pv["p"], "passed": bool(pv["p"] <= cfg.alpha)})
    nh_ok, nh_txt = RN.f_noharm(real["f"], cfg.q123_tol)
    crit.append({"id": "F4", "criterion": "held-out wells: no harm outside the high-swing set",
                 "threshold": f"equal-well RMSE corrected / base <= {cfg.q123_tol} pooled over "
                              f"the non-high-swing quartiles and in each of them with >= "
                              f"{RN.NOHARM_MIN_WELLS} scored wells",
                 "measured": nh_txt, "value": real["f"]["low_pooled"]["ratio"],
                 "passed": nh_ok})
    for k, tk, label in (("C1", "t1", "held-out years 2020-22"),
                         ("C2", "t2", "A1 site folds and years 2020-22")):
        s = real[tk]["summary"]["Q4"]
        crit.append({"id": k, "criterion": f"[design-time necessary condition, v1 unit, "
                                           f"post-look] Q4 excess falls, {label}",
                     "threshold": "corrected < base (strict)",
                     "measured": f"{s['excess_base_m2']:.3f} -> {s['excess_corrected_m2']:.3f} m2"
                                 f" (reduction {s['excess_reduction_m2']:.3f}; "
                                 f"{s['n_wells']} Q4 wells)",
                     "value": s["excess_reduction_m2"],
                     "passed": bool(s["excess_corrected_m2"] < s["excess_base_m2"])})
    for k, tk, label in (("C3", "t1", "held-out years"), ("C4", "t2", "A1 site folds")):
        sm = real[tk]["summary"]
        rat = {qq: sm[qq]["rmse_ratio_corrected_over_base"] for qq in ("Q1", "Q2", "Q3")}
        crit.append({"id": k, "criterion": f"[design-time necessary condition, v1 unit, "
                                           f"post-look] {label}: Q1, Q2, Q3 RMSE each rise by no "
                                           f"more than 2 %",
                     "threshold": f"corrected / base <= {cfg.q123_tol} in each quartile",
                     "measured": ", ".join(f"{qq} {v:.4f}" for qq, v in rat.items()),
                     "value": max(rat.values()),
                     "passed": bool(all(v <= cfg.q123_tol for v in rat.values()))})
    passed = all(c["passed"] for c in crit)
    # ---- reported --------------------------------------------------------------------------
    rep_p = {f"{tk}_{fam}": p_value(L.q4_reduction(real[tk]), plac[fam][tk])
             for tk in ("t1", "t2") for fam in ("shuffle", "distant")}
    f2 = RN.f2_eval(cfg, fr, Xf, real["a"])
    conc = {"f_high": concentration(real["f_base"]["mse"], real["f_cor"]["mse"],
                                    cls["high"] & sup_f),
            "t1_q4": concentration(real["t1"]["base"]["mse"], real["t1"]["cor"]["mse"],
                                   (qcal == 4) & sup_c),
            "t2_q4": concentration(real["t2"]["base"]["mse"], real["t2"]["cor"]["mse"],
                                   (qcal == 4) & sup_c)}
    low_sup = {}
    for tk in ("t1", "t2"):
        for k in (1, 2, 3):
            s = (qcal == k) & sup_c & mcal["score_ok"]
            b, c = real[tk]["base"]["mse"][s], real[tk]["cor"]["mse"][s]
            low_sup[f"{tk}_Q{k}_supported"] = {"n": int(s.sum()), "n_improved": int((c < b).sum()),
                                               "rmse_base": float(np.sqrt(b.mean())),
                                               "rmse_corrected": float(np.sqrt(c.mean()))}
    # sensitivity members
    sens = {}
    for name, kw in SENSITIVITY.items():
        c2 = V.with_config(cfg, **kw)
        reuse = c2.counties == cfg.counties and c2.arm == cfg.arm
        g2, cal2, fr2, _ = RN.setup(c2, True, cal=cal if reuse else None)
        tb2 = V.make_tables(c2, g2)
        prof = None
        if c2.t_profile:
            losses = {}
            for s in c2.t_profile:
                ca = RN.scale_t_at_bound(cal2, s)
                xa = RN.regressors(c2, g2, ca, tb2)[0]
                aa = L.fit_multipliers(replace(ca, cfg=c2), xa, mcal,
                                       np.ones(len(ca.sids), bool))
                losses[str(s)] = fit_loss(ca, xa, mcal, aa)
            s_best = float(min(losses, key=losses.get))
            prof = {"fit_loss_by_T_scale": losses, "chosen_T_scale": s_best,
                    "n_cal_at_bound_with_w": int((cal2.t_at_bound & (cal2.w > 0)).sum()),
                    "n_heldout_at_bound_with_w": int((fr2.t_at_bound & (fr2.w > 0)).sum())}
            cal2, fr2 = RN.scale_t_at_bound(cal2, s_best), RN.scale_t_at_bound(fr2, s_best)
        x2c, n2c = RN.regressors(c2, g2, cal2, tb2)
        x2f, n2f = RN.regressors(c2, g2, fr2, tb2)
        m2 = L.masks(replace(cal2, cfg=c2))
        q2, _, _ = L.quartiles(cal2, m2)
        cl2 = L.climatology(cal2, m2)
        fm2 = RN.f_masks(fr2)
        cls2 = RN.f_classes(RN.f_mse(fr2, fm2, None)["sd"], fm2["ok"], n2f > 0,
                            V.V1_QUARTILE_CUTS)
        e2 = RN.evaluate(c2, cal2, fr2, x2c, x2f, m2, q2, cl2, fm2, cls2, n2f > 0)
        nh2, nh2t = RN.f_noharm(e2["f"], cfg.q123_tol)
        sens[name] = {"config_change": {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in kw.items()},
                      "t_profile": prof, "a": _q(e2["a"]),
                      "cal_wells_with_kernel": int((n2c > 0).sum()),
                      "heldout_wells_with_kernel": int(((n2f > 0) & fm2["ok"]).sum()),
                      "f_high": e2["f"]["high"], "f_high_q": e2["f"]["high_q"],
                      "f_noharm": [nh2, nh2t],
                      "t1_q4_reduction": L.q4_reduction(e2["t1"]),
                      "t2_q4_reduction": L.q4_reduction(e2["t2"]),
                      "t1_q123": [e2["t1"]["summary"][f"Q{k}"]["rmse_ratio_corrected_over_base"]
                                  for k in (1, 2, 3)],
                      "t2_q123": [e2["t2"]["summary"][f"Q{k}"]["rmse_ratio_corrected_over_base"]
                                  for k in (1, 2, 3)]}
    sc_ok = mcal["score_ok"]
    out = {
        "prereg_id": PREREG_ID, "prereg_sha256": gate["sha256"] if gate else None,
        "post_registration_looks": gate["post_registration_looks"] if gate else None,
        "unverifiable_prior_looks_n": len(gate["unverifiable_prior_looks"]) if gate else None,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": cfg.as_dict(), "notes": cal.notes, "heldout_selection": sel_notes[0],
        "n_cal": int(len(cal.sids)), "n_pumps": int(len(cal.pump_xy)),
        "n_pumps_by_county": {c: int((cal.pump_county == c).sum())
                              for c in np.unique(cal.pump_county)},
        "cal_wells_with_kernel": int(sup_c.sum()),
        "cal_wells_with_kernel_by_county": {c: int((sup_c & (cal.county == c)).sum())
                                            for c in ("YL", "CH")},
        "heldout_candidates": int(len(fr.sids)), "heldout_scored": int(fm["ok"].sum()),
        "heldout_scored_with_kernel": int((fm["ok"] & sup_f).sum()),
        "heldout_scored_by_county": {c: int((fm["ok"] & (fr.county == c)).sum())
                                     for c in np.unique(fr.county)},
        "heldout_quartile_counts": {f"Q{k}": int((cls["q"] == k).sum()) for k in (1, 2, 3, 4)},
        "heldout_quartile_counts_supported": {f"Q{k}": int(((cls["q"] == k) & sup_f).sum())
                                              for k in (1, 2, 3, 4)},
        "quartile_cuts_sd_m": list(V.V1_QUARTILE_CUTS),
        "a": _q(real["a"]),
        "unit_sd_supported_cal_m": {"seasonal_median": float(np.median(Xc[0][sup_c].std(1))),
                                    "slow_median": float(np.median(Xc[1][sup_c].std(1)))},
        "fitted_correction_sd_cal_m": {
            "median": float(np.median(L.correction(Xc, real["a"])[sup_c].std(1))),
            "max": float(np.max(L.correction(Xc, real["a"])[sup_c].std(1)))},
        "fitted_correction_sd_heldout_m": {
            "median": float(np.median(real["corr_fr"][sup_f].std(1))) if sup_f.any() else None,
            "max": float(np.max(real["corr_fr"][sup_f].std(1))) if sup_f.any() else None},
        "f": real["f"],
        "t1": {"summary": real["t1"]["summary"],
               "q4_amplitude_ratio_base": L.amplitude_ratio(cal, mcal, None, qcal == 4),
               "q4_amplitude_ratio_corrected": L.amplitude_ratio(cal, mcal, real["t1"]["corr"],
                                                                 qcal == 4)},
        "t2": {"summary": real["t2"]["summary"],
               "a_by_fold": {str(k): _q(v) for k, v in real["t2"]["a_by_fold"].items()}},
        "criteria": crit, "gate_passed": bool(passed),
        "placebo_p_registered": pvals, "placebo_p_v1unit_reported": rep_p,
        "placebos": {f: {k: (v if k == "a" else _q(v)) for k, v in plac[f].items()}
                     for f in plac},
        "f2_heldout_years_reported": f2, "concentration": conc,
        "v1unit_supported_low_quartiles": low_sup, "sensitivity": sens,
        "runtime_s": round(time.perf_counter() - t0, 1), "placebo_s": round(t_pl, 1)}
    arrays = dict(
        cal_sids=cal.sids, dates=cal.dates, cal_county=cal.county, cal_layer=cal.layer,
        cal_cell=cal.cell, cal_quartile=qcal, cal_sd_fit=sdcal, cal_score_ok=sc_ok,
        cal_npump=nc, cal_S_unit=Xc[0], cal_L_unit=Xc[1], a=real["a"],
        cal_corr_t1=real["t1"]["corr"], cal_corr_t2=real["t2"]["corr"],
        cal_mse_base=real["t1"]["base"]["mse"], cal_mse_clim=real["t1"]["base"]["mse_clim"],
        cal_mse_t1=real["t1"]["cor"]["mse"], cal_mse_t2=real["t2"]["cor"]["mse"],
        cal_T=cal.T, cal_S=cal.S, cal_w=cal.w,
        f_sids=fr.sids, f_county=fr.county, f_layer=fr.layer, f_cell=fr.cell, f_xy=fr.xy,
        f_npump=nf, f_S_unit=Xf[0], f_L_unit=Xf[1], f_corr=real["corr_fr"],
        f_quartile=cls["q"], f_high=cls["high"], f_scored=fm["ok"], f_n=fm["n"],
        f_sd=real["f_base"]["sd"], f_mse_base=real["f_base"]["mse"],
        f_mse_cor=real["f_cor"]["mse"], f_mse_clim=real["f_base"]["mse_clim"],
        f_T=fr.T, f_S=fr.S, f_w=fr.w)
    return out, arrays


def main() -> None:
    gate = prereg3.assert_can_score(PREREG_ID, "H4")
    OUT.mkdir(parents=True, exist_ok=True)
    out, arrays = run(V.PRIMARY, gate)
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                default=float) + "\n")
    np.savez_compressed(OUT / "arrays.npz", **arrays)
    print(json.dumps({"criteria": [(c["id"], c["passed"], c["measured"]) for c in
                                   out["criteria"]], "gate_passed": out["gate_passed"],
                      "a": out["a"]}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
