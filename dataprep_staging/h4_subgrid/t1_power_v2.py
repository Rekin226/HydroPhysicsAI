"""WP H4 repair round: design-time power and bias check of the v2 F test on SYNTHETIC heads.

No head value is read. The observation masks are the real flag 0-1 availability (columns sid,
month, flag; the value column is not read) of the 158 calibration wells and the held-out
candidates, so the synthetic design has the real sample sizes. Synthetic observations:

    obs = h_cell - truth + datum + seasonal residual + AR(1) noise (sd 0.6 m, phi 0.7)

Scenarios (each R replicate datasets; placebo regressors precomputed once, 99 draws each):

* ``null_seasonal``: truth 0, seasonal residual (lognormal amplitude, median 1.5 m);
* ``real_seasonal``: truth = a_true (S + L) on the REAL sub-grid operator, a_true set so that
  the median sd of the truth at supported calibration wells is 0.25 m; seasonal residual;
* ``real_clean``: as above without the seasonal residual (multiplier recovery);
* ``shuffled_seasonal``: truth carried by a shuffled-position operator (a cone at the wrong
  pumps); seasonal residual.

Reports per scenario: median fitted (a_s, a_l), the rate of a positive high-swing MSE
reduction, of p <= 0.05 against each placebo, of the no-harm check, and of the joint F pass.
Writes results/dataprep/reports/H4_evidence/repair/design_time/t1_power.json.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import h4v2lib as V  # noqa: E402
import h4v2run as RN  # noqa: E402
from common import guard  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

OUT = DATAPREP / "reports/H4_evidence/repair/design_time"
N_DRAWS = 99
N_REP = 20
TRUTH_SD = 0.25


def availability_mask(sids, dates) -> np.ndarray:
    p = guard.open_product(L.HEADS, "selection", window_end="2022-12")
    df = p.read_table("heads_monthly.parquet", columns=["sid", "month", "flag"])
    if "value" in df.columns:
        raise RuntimeError("value column read")
    df["sid"] = df["sid"].astype(str)
    df = df[df["flag"].isin([0, 1]) & df["sid"].isin(set(sids)) & df["month"].isin(set(dates))]
    m = np.zeros((len(sids), len(dates)), bool)
    m[pd.Index(sids).get_indexer(df["sid"]), pd.Index(dates).get_indexer(df["month"])] = True
    return m


def synth(d: L.Data, truth: np.ndarray, mask: np.ndarray, rng, seasonal: bool) -> np.ndarray:
    W, D = d.h_cell.shape
    datum = rng.normal(0, 5, W)[:, None]
    moy = np.array([int(s[5:7]) for s in d.dates])
    amp = (1.5 * rng.lognormal(0, 0.8, W)[:, None]) if seasonal else np.zeros((W, 1))
    seas = amp * np.sin(2 * np.pi * (moy - 3) / 12.0)[None, :]
    e = np.zeros((W, D))
    z = rng.normal(0, 0.6, (W, D))
    for t in range(D):
        e[:, t] = (0.7 * e[:, t - 1] if t else 0.0) + z[:, t]
    obs = d.h_cell - truth + datum + seas + e
    return np.where(mask, obs, np.nan)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cfg = V.PRIMARY
    grid, cal, fr, _ = RN.setup(cfg, with_heads=False)
    tabs = V.make_tables(cfg, grid)
    mask_cal = availability_mask(cal.sids, cal.dates)
    mask_fr = availability_mask(fr.sids, fr.dates)
    Xc, nc = RN.regressors(cfg, grid, cal, tabs)
    Xf, nf = RN.regressors(cfg, grid, fr, tabs)
    sup_c, sup_f = nc > 0, nf > 0
    a_true = TRUTH_SD / float(np.median((Xc[0] + Xc[1])[sup_c].std(axis=1, ddof=1)))
    # placebo regressors (precomputed; the same for every replicate)
    pl = {"shuffle": [], "distant": []}
    for r in range(N_DRAWS):
        rng = np.random.default_rng(cfg.seed_shuffle + 1000 + r)
        pxy = L.shuffle_xy(cal, rng)
        pl["shuffle"].append((RN.regressors(cfg, grid, cal, tabs, pump_xy=pxy)[0],
                              RN.regressors(cfg, grid, fr, tabs, pump_xy=pxy)[0]))
        rng = np.random.default_rng(cfg.seed_distant + 1000 + r)
        cc = V.distant_centres(cal, cal.pump_xy, rng, sup_c, cfg)
        cf = V.distant_centres(fr, cal.pump_xy, rng, sup_f, cfg)
        pl["distant"].append((RN.regressors(cfg, grid, cal, tabs, centres=cc, support=sup_c)[0],
                              RN.regressors(cfg, grid, fr, tabs, centres=cf, support=sup_f)[0]))
    t_pl = time.perf_counter() - t0
    shuf_truth_c, _ = RN.regressors(cfg, grid, cal, tabs,
                                    pump_xy=L.shuffle_xy(cal, np.random.default_rng(99)))
    shuf_truth_f, _ = RN.regressors(cfg, grid, fr, tabs,
                                    pump_xy=L.shuffle_xy(cal, np.random.default_rng(99)))
    scen = {
        "null_seasonal": (0.0, None, True),
        "real_seasonal": (a_true, "real", True),
        "real_clean": (a_true, "real", False),
        "shuffled_seasonal": (a_true, "shuffled", True)}
    res = {"reads_head_values": False, "n_draws": N_DRAWS, "n_rep": N_REP,
           "a_true": a_true, "truth_sd_target_m": TRUTH_SD,
           "n_cal": int(len(cal.sids)), "n_fresh_candidates": int(len(fr.sids)),
           "placebo_precompute_s": round(t_pl, 1), "scenarios": {}}
    for name, (amp, kind, seas) in scen.items():
        rows = []
        for rep in range(N_REP):
            rng = np.random.default_rng(5000 + rep)
            if kind == "real":
                tc, tf = amp * (Xc[0] + Xc[1]), amp * (Xf[0] + Xf[1])
            elif kind == "shuffled":
                tc = amp * (shuf_truth_c[0] + shuf_truth_c[1])
                tf = amp * (shuf_truth_f[0] + shuf_truth_f[1])
            else:
                tc, tf = 0.0, 0.0
            c2 = replace(cal, obs=synth(cal, tc, mask_cal, rng, seas))
            f2 = replace(fr, obs=synth(fr, tf, mask_fr, rng, seas))
            mcal = L.masks(c2)
            fm = RN.f_masks(f2)
            fb = RN.f_mse(f2, fm, None)
            cls = RN.f_classes(fb["sd"], fm["ok"], sup_f, V.V1_QUARTILE_CUTS)
            ev = RN.evaluate(cfg, c2, f2, Xc, Xf, mcal, None, None, fm, cls, sup_f,
                             with_v1unit=False)
            red = ev["f"]["high"]["mse_reduction_m2"]
            ps = {}
            for fam in ("shuffle", "distant"):
                vals = []
                for xc, xf in pl[fam]:
                    e2 = RN.evaluate(cfg, c2, f2, xc, xf, mcal, None, None, fm, cls, sup_f,
                                     with_v1unit=False)
                    vals.append(e2["f"]["high"]["mse_reduction_m2"])
                vals = np.array(vals)
                ps[fam] = float((1 + np.sum(vals >= red)) / (1 + vals.size))
            nh, _ = RN.f_noharm(ev["f"], cfg.q123_tol)
            rows.append({"a_s": float(ev["a"][0]), "a_l": float(ev["a"][1]), "red": red,
                         "p_shuffle": ps["shuffle"], "p_distant": ps["distant"],
                         "noharm": nh, "high_q": ev["f"]["high_q"],
                         "n_high": ev["f"]["high"]["n_wells"],
                         "n_high_sup": ev["f"]["high"]["n_supported"],
                         "n_scored": int(fm["ok"].sum())})
        df = pd.DataFrame(rows)
        joint = (df.red > 0) & (df.p_shuffle <= 0.05) & (df.p_distant <= 0.05) & df.noharm
        res["scenarios"][name] = {
            "a_s_median": float(df.a_s.median()), "a_l_median": float(df.a_l.median()),
            "rate_reduction_pos": float((df.red > 0).mean()),
            "rate_p_shuffle": float((df.p_shuffle <= 0.05).mean()),
            "rate_p_distant": float((df.p_distant <= 0.05).mean()),
            "rate_noharm": float(df.noharm.mean()), "rate_joint_pass": float(joint.mean()),
            "n_scored": int(df.n_scored.iloc[0]), "n_high_median": float(df.n_high.median()),
            "n_high_supported_median": float(df.n_high_sup.median()),
            "high_q_counts": df.high_q.astype(str).value_counts().to_dict(),
            "reduction_median": float(df.red.median())}
        print(name, json.dumps(res["scenarios"][name], default=float))
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "t1_power.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
