"""WP H4 smoke test of the registered pipeline on SYNTHETIC heads (no head data is read).

Runs ``r1_registered.compute`` with a builder that replaces the observations by
``h_cell - (a_s S + a_l L) + datum + seasonal + AR(1) noise`` built from the real operator and
the real hindcast (``h4lib.build_data(with_heads=False)``), with 3 placebo draws, and checks
that (i) every code path runs, (ii) the fit recovers known multipliers when the synthetic
truth is the operator itself, (iii) a truth carried by shuffled positions is not credited to
the real kernel. Writes results/dataprep/reports/H4_evidence/design_time/t0_smoke.json.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import r1_registered as R  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

OUT = DATAPREP / "reports/H4_evidence/design_time"
A_TRUE = (8.0, 4.0)


def make_builder(kind: str, seasonal_amp: float):
    cache: dict = {}

    def builder(cfg: L.Config) -> L.Data:
        key = (cfg,)
        if key in cache:
            return cache[key]
        d = L.build_data(cfg, with_heads=False)
        rng = np.random.default_rng(11)
        W, D = d.h_cell.shape
        if kind == "real":
            Gs, Gl, _ = L.operator_matrices(d)
        else:                      # truth from shuffled pump positions
            Gs, Gl, _ = L.operator_matrices(d, pump_xy=L.shuffle_xy(d, np.random.default_rng(99)))
        S, Lx = L.regressors(d, Gs, Gl)
        truth = A_TRUE[0] * S + A_TRUE[1] * Lx
        datum = rng.normal(0, 5, W)[:, None]
        moy = np.array([int(s[5:7]) for s in d.dates])
        amp = seasonal_amp * rng.lognormal(0, 0.8, W)[:, None]
        seas = amp * np.sin(2 * np.pi * (moy - 3) / 12.0)[None, :]
        e = np.zeros((W, D))
        z = rng.normal(0, 0.6, (W, D))
        for t in range(D):
            e[:, t] = (0.7 * e[:, t - 1] if t else 0.0) + z[:, t]
        obs = d.h_cell - truth + datum + seas + e
        miss = rng.random((W, D)) < 0.1
        miss[rng.random(W) < 0.05, :] = True          # some wells without any record
        obs[miss] = np.nan
        d2 = replace(d, obs=obs)
        cache[key] = d2
        return d2

    return builder


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = L.with_config(L.PRIMARY, n_placebo=3)
    res = {"reads_head_data": False, "a_true": A_TRUE}
    out, arrays = R.compute(cfg, make_builder("real", 0.0), R.SENSITIVITY, None)
    res["real_truth_no_seasonal"] = {
        "a_t1": out["primary"]["t1"]["a"], "a_t2": out["primary"]["t2"]["a_by_fold"],
        "criteria": [(c["id"], c["passed"], c["measured"]) for c in out["criteria"]],
        "sensitivity_ran": sorted(out["sensitivity"]),
        "sens_a_t1": {k: v["t1"]["a"] for k, v in out["sensitivity"].items()},
        "quartile_counts_scored": out["quartile_counts_scored"],
        "arrays": sorted(arrays)}
    out, _ = R.compute(cfg, make_builder("real", 1.5), {}, None)
    res["real_truth_with_seasonal"] = {
        "a_t1": out["primary"]["t1"]["a"],
        "criteria": [(c["id"], c["passed"], c["measured"]) for c in out["criteria"]]}
    out, _ = R.compute(cfg, make_builder("shuffled", 1.5), {}, None)
    res["shuffled_truth_with_seasonal"] = {
        "a_t1": out["primary"]["t1"]["a"],
        "criteria": [(c["id"], c["passed"], c["measured"]) for c in out["criteria"]]}
    (OUT / "t0_smoke.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
