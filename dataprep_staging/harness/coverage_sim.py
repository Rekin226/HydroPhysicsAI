"""WP A1 acceptance: actual coverage of the cluster-adjusted coverage interval (plan 4.4).

    python -m harness.coverage_sim OUT.json        (from dataprep_staging/, in a CPU slot)

Criterion (plan WP A1): the coverage interval reaches at least 0.89 actual coverage of a true
0.90 in 10,000 simulated clustered binomial draws. Read here as: when the true coverage rate
of a product's intervals is 0.90, the gate's interval for that rate (level ``GATE_LEVEL`` =
0.95, the level the gate uses) contains 0.90 in at least 89 % of 10,000 draws, in **every**
configuration declared below.

Declared before the first run (2026-10-12), not changed after seeing results:

* Configurations mirror the plan's use cases: independent rows (n = 30, 200); W1 (7 years x 2
  districts); E11 / E6 (5 regions x 6 runs); S1 (14 sites x 13 months); V3 (74 wells x 42
  months); H2 (300 gaps of 1-24 months); unequal clusters (k = 40, log-normal sizes); few
  large clusters (k = 5 x 200, k = 10 x 50). ICC rho from 0 to 0.3.
* Clustered draws: cluster hit probabilities ~ Beta(a, b) with mean p and
  rho = 1 / (a + b + 1) (beta-binomial); rho = 0 gives independent Bernoulli rows.
* Seeds: ``20261012 + 100 * config_index + variant`` with numpy's default_rng.
* Decision rule: the default quantile rule is the plan's method (``small_k="z"``). If it
  meets the criterion in every configuration it stays the default. If it fails any
  configuration and the Student-t variant (``small_k="t"``, k - 1 df) meets it in every
  configuration, ``t`` becomes the default and the ``z`` failures are reported. If both fail
  somewhere, the criterion fails for those configurations and the variant with the higher
  minimum coverage is the default.
* Reported for information (not acceptance): the 0.90-level interval's coverage, mean n_eff
  against the true n / deff, and how often the full 4.4 rule passes a product whose true
  coverage is 0.80 or 0.97 (power).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.coverage import (  # noqa: E402
    GATE_LEVEL,
    N_EFF_WINDOW_MIN,
    POINT_WINDOW,
    TARGET,
    interval_from_totals_batch,
)

N_DRAWS = 10_000
CRITERION = 0.89
SEED0 = 20261012


def _sizes(name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "h2":
        return np.tile(np.array([1, 2, 3, 6, 12, 24], float), 50)
    if name == "lognormal40":
        return np.maximum(1, np.round(rng.lognormal(np.log(12), 0.9, 40))).astype(float)
    raise ValueError(name)


#: (name, k, m (int or sizes key), rho)
CONFIGS = [
    ("indep_n30", 30, 1, 0.0),
    ("indep_n200", 200, 1, 0.0),
    ("W1_7yr_x2", 7, 2, 0.2),
    ("E11_5reg_x6_rho0.1", 5, 6, 0.1),
    ("E11_5reg_x6_rho0.3", 5, 6, 0.3),
    ("S1_14site_x13_rho0.1", 14, 13, 0.1),
    ("S1_14site_x13_rho0.3", 14, 13, 0.3),
    ("V3_74well_x42_rho0.05", 74, 42, 0.05),
    ("V3_74well_x42_rho0.2", 74, 42, 0.2),
    ("H2_300gap_rho0.0", 300, "h2", 0.0),
    ("H2_300gap_rho0.1", 300, "h2", 0.1),
    ("H2_300gap_rho0.3", 300, "h2", 0.3),
    ("unequal_k40_rho0.2", 40, "lognormal40", 0.2),
    ("few_large_k5_x200_rho0.05", 5, 200, 0.05),
    ("k10_x50_rho0.1", 10, 50, 0.1),
]


def draw_totals(m: np.ndarray, p: float, rho: float, n: int, rng) -> np.ndarray:
    k = m.size
    if rho <= 0:
        return rng.binomial(m.astype(int)[None, :].repeat(n, 0), p).astype(float)
    ab = 1.0 / rho - 1.0
    pi = rng.beta(p * ab, (1 - p) * ab, size=(n, k))
    return rng.binomial(m.astype(int)[None, :].repeat(n, 0), pi).astype(float)


def run() -> dict:
    out = {"criterion": f"actual coverage of a true {TARGET} >= {CRITERION} with the "
                        f"level-{GATE_LEVEL} interval, {N_DRAWS} draws per configuration",
           "n_draws": N_DRAWS, "gate_level": GATE_LEVEL, "configs": []}
    for ci, (name, k, msz, rho) in enumerate(CONFIGS):
        rng_sizes = np.random.default_rng(SEED0 + 100 * ci + 99)
        m = (np.full(k, float(msz)) if isinstance(msz, int) else _sizes(msz, rng_sizes))
        assert m.size == k
        N = float(m.sum())
        m_w = float((m ** 2).sum() / N)
        true_neff = N / (1 + (m_w - 1) * rho)
        row = {"name": name, "k": k, "n": N, "m_w": m_w, "rho": rho,
               "true_n_eff": true_neff, "results": {}}
        Y = draw_totals(m, TARGET, rho, N_DRAWS, np.random.default_rng(SEED0 + 100 * ci))
        for sk in ("z", "t"):
            for level in (GATE_LEVEL, 0.90):
                r = interval_from_totals_batch(Y, m, level, sk)
                cov = float(np.mean((r["lo"] <= TARGET) & (r["hi"] >= TARGET)))
                row["results"][f"{sk}_L{level}"] = {
                    "actual_coverage": cov, "mean_n_eff": float(np.mean(r["n_eff"])),
                    "median_rho_hat": float(np.median(r["rho"])),
                    "mean_width": float(np.mean(r["hi"] - r["lo"])),
                    "quantile": r["quantile"]}
        # power of the full 4.4 rule (interval contains 0.90; point window if n_eff >= 30)
        for variant, ptrue in ((1, 0.80), (2, 0.97)):
            Yp = draw_totals(m, ptrue, rho, N_DRAWS,
                             np.random.default_rng(SEED0 + 100 * ci + variant))
            for sk in ("z", "t"):
                r = interval_from_totals_batch(Yp, m, GATE_LEVEL, sk)
                in_ci = (r["lo"] <= TARGET) & (r["hi"] >= TARGET)
                big = r["n_eff"] >= N_EFF_WINDOW_MIN
                in_win = (r["p"] >= POINT_WINDOW[0]) & (r["p"] <= POINT_WINDOW[1])
                passes = in_ci & (in_win | ~big)
                row["results"][f"{sk}_pass_rate_true{ptrue}"] = float(np.mean(passes))
        # the full rule at the true 0.90 (false-fail rate = 1 - pass rate)
        for sk in ("z", "t"):
            r = interval_from_totals_batch(Y, m, GATE_LEVEL, sk)
            in_ci = (r["lo"] <= TARGET) & (r["hi"] >= TARGET)
            big = r["n_eff"] >= N_EFF_WINDOW_MIN
            in_win = (r["p"] >= POINT_WINDOW[0]) & (r["p"] <= POINT_WINDOW[1])
            row["results"][f"{sk}_pass_rate_true0.9"] = float(np.mean(in_ci & (in_win | ~big)))
        out["configs"].append(row)
        print(name, {kk: (round(v["actual_coverage"], 4) if isinstance(v, dict) else round(v, 3))
                     for kk, v in row["results"].items()}, flush=True)
    mins = {sk: min(c["results"][f"{sk}_L{GATE_LEVEL}"]["actual_coverage"]
                    for c in out["configs"]) for sk in ("z", "t")}
    fails = {sk: [c["name"] for c in out["configs"]
                  if c["results"][f"{sk}_L{GATE_LEVEL}"]["actual_coverage"] < CRITERION]
             for sk in ("z", "t")}
    if not fails["z"]:
        default = "z"
    elif not fails["t"]:
        default = "t"
    else:
        default = max(("z", "t"), key=lambda s: mins[s])
    out["min_actual_coverage"] = mins
    out["configs_below_criterion"] = fails
    out["decision"] = {"default_small_k": default,
                       "criterion_met_by_default": not fails[default],
                       "rule": "z if it passes everywhere, else t if it passes everywhere, "
                               "else the higher minimum (criterion failed)"}
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.time()
    res = run()
    res["seconds"] = round(time.time() - t0, 1)
    if argv:
        Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
        with open(argv[0], "x", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("min_actual_coverage", "configs_below_criterion",
                                          "decision", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
