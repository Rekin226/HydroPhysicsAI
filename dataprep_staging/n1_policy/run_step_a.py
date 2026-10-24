"""Step A of N1.policy_response.v1 (declared checks, model-internal, never skill).

A1 compensation: does a leveling-fitted column recover the subsidence a pumping cut avoids
when the true head response to pumping is k times the model's? Truth heads
H_k = H0 + k (H - H0) (H0 = zero pumping energy); synthetic leveling = the model's own
column on H_k at the real benchmark cells and survey months 2012-21 (survey DATES only;
>= 5 surveys, nearest month within 45 days), re-zeroed at each benchmark's first survey,
plus N(0, 0.5 cm). A fresh three-zone column is fitted to it on the model's own heads H
(calibrate_coupled._fit, 2000 epochs, h_pc0 guard 365 d). The cut30 lever (irrigation
energy x 0.7 from 2013-01) is applied to both systems; reported per column zone and county:
fitted / true subsidence avoided at 2021-12. 'compensates' if within [0.8, 1.25] at k = 2.

A2 ranking: the page's response basis perturbed per zone x county (log-uniform [0.5, 2],
500 draws): Kendall tau of the lever order, top-decile overlap, lever-total ranges.

Usage: CUDA_VISIBLE_DEVICES="" results/twin_runs/par24/py.sh -I run_step_a.py [--a1] [--a2]
Outputs: results/dataprep/reports/N1_policy/step_a/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402

OUT = REPORTS_DIR / "N1_policy" / "step_a"
SEED = 20261223
KS = (1.0, 1.5, 2.0, 3.0)
NOISE_M = 0.005
CUT, CUT_FROM = 0.3, pd.Period("2013-01", "M")
END = pd.Period("2021-12", "M")
R = "results/twin_runs"
MODELS = {
    "R0": (f"{R}/stage3_datum_gate_cwa/stage3_theta.json",
           f"{R}/stage3_datum_gate_cwa/coupled_leveling/vep_zonal_leveling.json"),
    "R1c_z3": (f"{R}/stage3_datum_gate_ccz/stage3_theta.json",
               f"{R}/stage3_datum_gate_ccz/coupled_leveling_z3/vep_zonal_leveling.json"),
}
BASIS = REPO / "results/twin_forward/response_basis.npz"


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def survey_mask(tm: PL.TwinModel) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark cells (n,) and survey months (n, T) from the panel's dates only."""
    from hydrophysics.config import Config

    meta = pd.read_parquet(os.path.join(str(Config().data_dir), "ls_cache",
                                        "ls-wra-lsp-obs__choushui_panel.parquet"),
                           columns=["sid", "datetime", "x_3826", "y_3826"])
    meta["datetime"] = pd.to_datetime(meta.datetime)
    meta = meta[(meta.datetime >= "2012-01-01") & (meta.datetime < "2022-01-01")]
    dates = pd.DatetimeIndex([p.to_timestamp() for p in PL.MONTHS])
    cells, M = [], []
    for _, g in meta.groupby("sid"):
        if len(g) < 5:
            continue
        c = tm.cell_of(g[["x_3826", "y_3826"]].to_numpy()[:1])[0]
        if c < 0:
            continue
        m = np.zeros(len(dates), bool)
        for t in g.datetime:
            j = int(np.argmin(np.abs((dates - t).days)))
            if abs((dates[j] - t).days) <= 45:
                m[j] = True
        if m.sum() >= 3:
            cells.append(c)
            M.append(m)
    return np.array(cells), np.array(M)


def cut_energy(tm: PL.TwinModel) -> np.ndarray:
    E = tm.energy()
    step_month = PL.MONTHS[1:E.shape[-1] + 1]
    on = np.array([p >= CUT_FROM for p in step_month])
    ec = tm.meta.get("eta_classes")
    if ec:
        k = list(ec).index("irrigation")
        E[k][:, on] *= 1.0 - CUT
    else:
        irr = np.asarray(tm.inp.E_by_class["irrigation"], float)[:, 1:]
        E[:, on] -= CUT * irr[:, on]
    return E


def a1(name: str, theta: str, col: str, rng) -> dict:
    import torch

    import hydrophysics.twin.calibrate_coupled as cc
    from hydrophysics.twin.zone_county import DEFAULT_MAP, column_zone_map, county_flags_for_grid

    cc.HPC0_GUARD_DAYS = 365.0
    tm = PL.TwinModel(str(REPO / theta), str(REPO / col))
    H, Hc = tm.heads(), tm.heads(cut_energy(tm))
    H0 = tm.heads(np.zeros_like(tm.energy()))
    log(f"[{name}] rollouts done")
    cells, M = survey_mask(tm)
    zone = column_zone_map(tm.inp.grid, tm.meta) % 3
    ch = np.asarray(county_flags_for_grid(tm.inp.grid, DEFAULT_MAP)[0], bool)
    e = list(PL.MONTHS).index(END)
    t32 = torch.float32

    def fitted_column(obs_rows):
        model = cc._make("zonal", "cpu", 3)
        Ht = torch.tensor(H[:, cells, :].transpose(1, 0, 2), dtype=t32)
        cc._fit(model, Ht, torch.tensor(obs_rows, dtype=t32), torch.tensor(M, dtype=t32),
                torch.tensor(zone[cells], dtype=torch.long), 2000, 0.05, rezero=True)
        return model

    def predict(model, heads):
        Ha = torch.tensor(heads.transpose(1, 0, 2), dtype=t32)
        with torch.no_grad():
            return cc._predict(model, Ha, torch.tensor(zone, dtype=torch.long)).numpy()

    groups = {f"zone{z}_{c}": (zone == z) & (ch == (c == "changhua"))
              for z in range(3) for c in ("yunlin", "changhua")}
    out = {"model": {"theta": theta, "column": col}, "n_benchmarks": int(len(cells)),
           "k": {}}
    for k in KS:
        Hk, Hkc = H0 + k * (H - H0), H0 + k * (Hc - H0)
        S_true = tm.subsidence(Hk)
        rows = S_true[cells]
        first = M.argmax(axis=1)
        rows = rows - rows[np.arange(len(cells)), first][:, None]
        obs = np.where(M, rows + rng.normal(0, NOISE_M, rows.shape), 0.0)
        mdl = fitted_column(obs)
        av_true = S_true[:, e] - tm.subsidence(Hkc)[:, e]
        av_fit = predict(mdl, H)[:, e] - predict(mdl, Hc)[:, e]
        res = {}
        for g, sel in groups.items():
            tt, ff = float(av_true[sel].sum()), float(av_fit[sel].sum())
            res[g] = {"true_avoided_m_sum": tt, "fitted_avoided_m_sum": ff,
                      "ratio": ff / tt if abs(tt) > 1e-9 else float("nan"),
                      "n_cells": int(sel.sum())}
        tt, ff = float(av_true.sum()), float(av_fit.sum())
        res["fan"] = {"true_avoided_m_sum": tt, "fitted_avoided_m_sum": ff,
                      "ratio": ff / tt if abs(tt) > 1e-9 else float("nan")}
        out["k"][str(k)] = res
        log(f"[{name}] k={k}: fan fitted/true avoided {res['fan']['ratio']:.3f}")
    r2 = out["k"]["2.0"]["fan"]["ratio"]
    out["verdict_k2"] = ("compensates" if 0.8 <= r2 <= 1.25 else "does_not_compensate")
    return out


def a2(rng) -> dict:
    from scipy.stats import kendalltau

    from hydrophysics.twin.app.prep import CLASSES, STARTS, basis_deltas, year_ends
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid
    from hydrophysics.twin.zones import fan_zones

    z = np.load(BASIS, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    ye, years = year_ends(pd.DatetimeIndex(z["dates"]))
    b = basis_deltas(z["subs_mean"] * 100.0, z["heads_mean"], names, ye)
    dsub = -b["dsub"][..., -1]                     # (R, A) cm avoided at the last year-end
    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
    ch = np.asarray(county_flags_for_grid(grid, DEFAULT_MAP)[0], bool)
    zone = fan_zones(grid.centroids())
    grp = zone + 3 * ch.astype(int)
    K = len(CLASSES)
    out = {"basis": rel_to_repo(BASIS), "year_end": int(years[-1]), "starts": {}}
    for si, s in enumerate(STARTS):
        R_ = dsub[si * K:(si + 1) * K]
        base_tot = R_.sum(axis=1)
        base_order = np.argsort(-base_tot)
        irr = R_[CLASSES.index("irrigation")] * 0.3
        top = irr >= np.quantile(irr, 0.9)
        taus, jac, tots = [], [], []
        for _ in range(500):
            f = np.exp(rng.uniform(np.log(0.5), np.log(2.0), 6))[grp]
            P = R_ * f[None, :]
            tot = P.sum(axis=1)
            taus.append(kendalltau(base_tot, tot).statistic)
            pt = P[CLASSES.index("irrigation")] * 0.3
            tp = pt >= np.quantile(pt, 0.9)
            jac.append(float((top & tp).sum() / max((top | tp).sum(), 1)))
            tots.append(tot)
        tots = np.array(tots)
        taus = np.array(taus)
        out["starts"][str(s)] = {
            "lever_order": [CLASSES[i] for i in base_order],
            "lever_total_cm_sum": dict(zip(CLASSES, map(float, base_tot), strict=True)),
            "share_tau_1": float(np.mean(taus >= 1.0 - 1e-12)),
            "tau_p5_p50": [float(np.quantile(taus, 0.05)), float(np.median(taus))],
            "top_decile_jaccard_p5_p50": [float(np.quantile(jac, 0.05)),
                                          float(np.median(jac))],
            "lever_total_p5_p95": {c: [float(np.quantile(tots[:, k], 0.05)),
                                       float(np.quantile(tots[:, k], 0.95))]
                                   for k, c in enumerate(CLASSES)},
            "ranking_survives": bool(np.mean(taus >= 1.0 - 1e-12) >= 0.9)}
        log(f"A2 start {s}: tau=1 in {out['starts'][str(s)]['share_tau_1']:.0%} of draws")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a1", action="store_true")
    ap.add_argument("--a2", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if a.a2:
        p = OUT / "a2.json"
        if p.exists():
            raise SystemExit(f"{p} exists; results are never recomputed")
        p.write_text(json.dumps(a2(rng), indent=1) + "\n")
        log(f"wrote {rel_to_repo(p)}")
    if a.a1:
        p = OUT / "a1.json"
        if p.exists():
            raise SystemExit(f"{p} exists; results are never recomputed")
        res = {name: a1(name, th, col, rng) for name, (th, col) in MODELS.items()}
        p.write_text(json.dumps(res, indent=1) + "\n")
        log(f"wrote {rel_to_repo(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
