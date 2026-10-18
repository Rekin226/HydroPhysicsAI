"""WP A1 acceptance, confirmatory run: the frozen coverage-gate interval on fresh draws.

    python -m harness.coverage_confirm OUT.json      (from dataprep_staging/, in a CPU slot)

The design-time simulation (``coverage_sim.py``, seeds ``20261012 + ...``) chose the
Student-t quantile (``coverage.DEFAULT_SMALL_K = "t"``) after the plan's normal quantile fell
below 0.89 in one configuration. A choice made on those draws cannot be validated on them, so
this script re-tests the frozen rule on fresh seeds and on configurations that the design run
did not contain. It is pre-registered with the A1 tool as ``A1.coverage_confirm.v1`` before it
runs (configurations, seeds, criterion and interpretation are in the pre-registration), and it
refuses to run unless :func:`harness.prereg.assert_can_score` passes.

Criterion (plan WP A1): with a true coverage rate of 0.90, the gate's interval (cluster-adjusted
Wilson on n_eff, ANOVA ICC, Student-t quantile on k - 1 df, level 0.95) contains 0.90 in at
least 89 % of 10,000 draws, in every PRIMARY configuration. SECONDARY configurations (two-point
mixtures of cluster coverage, a heterogeneity shape the beta-binomial does not have) are
reported and never gate. No rule is changed after the run; a changed rule needs a new id and
new seeds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import coverage  # noqa: E402
from harness.coverage_sim import _sizes as _sizes_design  # noqa: E402
from harness.coverage_sim import draw_totals  # noqa: E402

PREREG_ID = "A1.coverage_confirm.v1"
TEST_ID = "A1.coverage_confirm.v1.T1"
N_DRAWS = 10_000
CRITERION = 0.89
SEED0 = 20271012          # fresh: the design run used 20261012 + 100 * i + v
MIX_P = (1.0, 0.6)        # two-point mixture of cluster coverage ...
MIX_W = (0.75, 0.25)      # ... with weights giving mean 0.90 and ICC 1/3

#: (name, k, m (int or sizes key), rho, kind, role)
CONFIGS = [
    # the 15 design-time configurations, re-drawn with fresh seeds
    ("indep_n30", 30, 1, 0.0, "beta", "primary"),
    ("indep_n200", 200, 1, 0.0, "beta", "primary"),
    ("W1_7yr_x2", 7, 2, 0.2, "beta", "primary"),
    ("E11_5reg_x6_rho0.1", 5, 6, 0.1, "beta", "primary"),
    ("E11_5reg_x6_rho0.3", 5, 6, 0.3, "beta", "primary"),
    ("S1_14site_x13_rho0.1", 14, 13, 0.1, "beta", "primary"),
    ("S1_14site_x13_rho0.3", 14, 13, 0.3, "beta", "primary"),
    ("V3_74well_x42_rho0.05", 74, 42, 0.05, "beta", "primary"),
    ("V3_74well_x42_rho0.2", 74, 42, 0.2, "beta", "primary"),
    ("H2_300gap_rho0.0", 300, "h2", 0.0, "beta", "primary"),
    ("H2_300gap_rho0.1", 300, "h2", 0.1, "beta", "primary"),
    ("H2_300gap_rho0.3", 300, "h2", 0.3, "beta", "primary"),
    ("unequal_k40_rho0.2", 40, "lognormal40", 0.2, "beta", "primary"),
    ("few_large_k5_x200_rho0.05", 5, 200, 0.05, "beta", "primary"),
    ("k10_x50_rho0.1", 10, 50, 0.1, "beta", "primary"),
    # new configurations, not in the design run
    ("W1_7yr_x2_rho0.5", 7, 2, 0.5, "beta", "primary"),
    ("E6_5reg_x6_rho0.5", 5, 6, 0.5, "beta", "primary"),
    ("k20_x30_rho0.4", 20, 30, 0.4, "beta", "primary"),
    ("k3_x10_rho0.1", 3, 10, 0.1, "beta", "primary"),
    ("unequal_k30_sigma1.5_rho0.2", 30, "lognormal30_s1.5", 0.2, "beta", "primary"),
    ("mix_S1_14site_x13", 14, 13, None, "mix", "secondary"),
    ("mix_k30_x10", 30, 10, None, "mix", "secondary"),
    ("mix_V3_74well_x42", 74, 42, None, "mix", "secondary"),
]


def sizes(key, k: int, rng) -> np.ndarray:
    if isinstance(key, int):
        return np.full(k, float(key))
    if key == "lognormal30_s1.5":
        return np.maximum(1, np.round(rng.lognormal(np.log(12), 1.5, 30))).astype(float)
    return _sizes_design(key, rng)


def draw_mix(m: np.ndarray, n: int, rng) -> np.ndarray:
    pick = rng.random((n, m.size)) < MIX_W[0]
    pi = np.where(pick, MIX_P[0], MIX_P[1])
    return rng.binomial(m.astype(int)[None, :].repeat(n, 0), pi).astype(float)


def mix_icc() -> float:
    mean = MIX_W[0] * MIX_P[0] + MIX_W[1] * MIX_P[1]
    var = MIX_W[0] * (MIX_P[0] - mean) ** 2 + MIX_W[1] * (MIX_P[1] - mean) ** 2
    return var / (mean * (1 - mean))


def _rule(r, target=coverage.TARGET):
    in_ci = (r["lo"] <= target) & (r["hi"] >= target)
    big = r["n_eff"] >= coverage.N_EFF_WINDOW_MIN
    in_win = (r["p"] >= coverage.POINT_WINDOW[0]) & (r["p"] <= coverage.POINT_WINDOW[1])
    return in_ci, in_ci & (in_win | ~big)


def run() -> dict:
    sk, level = coverage.DEFAULT_SMALL_K, coverage.GATE_LEVEL
    out = {"prereg_id": PREREG_ID, "test_id": TEST_ID, "n_draws": N_DRAWS,
           "criterion": f"actual coverage of a true {coverage.TARGET} >= {CRITERION}, "
                        f"every primary configuration", "rule": {"small_k": sk, "level": level},
           "seed0": SEED0, "configs": []}
    for ci, (name, k, msz, rho, kind, role) in enumerate(CONFIGS):
        m = sizes(msz, k, np.random.default_rng(SEED0 + 100 * ci + 99))
        assert m.size == k
        N = float(m.sum())
        m_w = float((m ** 2).sum() / N)
        rho_true = mix_icc() if kind == "mix" else rho
        rng = np.random.default_rng(SEED0 + 100 * ci)
        Y = draw_mix(m, N_DRAWS, rng) if kind == "mix" else \
            draw_totals(m, coverage.TARGET, rho, N_DRAWS, rng)
        r = coverage.interval_from_totals_batch(Y, m, level, sk)
        in_ci, full = _rule(r)
        row = {"name": name, "role": role, "kind": kind, "k": k, "n": N, "m_w": m_w,
               "rho_true": rho_true, "true_n_eff": N / (1 + (m_w - 1) * rho_true),
               "actual_coverage": float(np.mean(in_ci)),
               "meets_criterion": bool(np.mean(in_ci) >= CRITERION),
               "mean_n_eff": float(np.mean(r["n_eff"])),
               "median_rho_hat": float(np.median(r["rho"])),
               "mean_width": float(np.mean(r["hi"] - r["lo"])), "quantile": r["quantile"],
               "full_rule_pass_rate_true0.9": float(np.mean(full))}
        if kind == "beta":       # power of the full 4.4 rule (information only)
            for v, ptrue in ((1, 0.80), (2, 0.97)):
                Yp = draw_totals(m, ptrue, rho, N_DRAWS,
                                 np.random.default_rng(SEED0 + 100 * ci + v))
                rp = coverage.interval_from_totals_batch(Yp, m, level, sk)
                row[f"full_rule_pass_rate_true{ptrue}"] = float(np.mean(_rule(rp)[1]))
        out["configs"].append(row)
        print(f"{name:32s} {role:9s} cov {row['actual_coverage']:.4f} "
              f"n_eff {row['mean_n_eff']:.1f} (true {row['true_n_eff']:.1f})", flush=True)
    prim = [c for c in out["configs"] if c["role"] == "primary"]
    sec = [c for c in out["configs"] if c["role"] == "secondary"]
    out["primary_min_actual_coverage"] = min(c["actual_coverage"] for c in prim)
    out["primary_below_criterion"] = [c["name"] for c in prim if not c["meets_criterion"]]
    out["secondary_min_actual_coverage"] = min(c["actual_coverage"] for c in sec)
    out["secondary_below_criterion"] = [c["name"] for c in sec if not c["meets_criterion"]]
    out["pass"] = not out["primary_below_criterion"]
    return out


def main(argv=None) -> int:
    from harness import prereg
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m harness.coverage_confirm OUT.json")
    ok = prereg.assert_can_score(PREREG_ID, "A1", test_id=TEST_ID)
    t0 = time.time()
    res = run()
    res["prereg_sha256"] = ok["sha256"]
    res["seconds"] = round(time.time() - t0, 1)
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(argv[0], "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("primary_min_actual_coverage",
                                          "primary_below_criterion",
                                          "secondary_min_actual_coverage",
                                          "secondary_below_criterion", "pass", "seconds")},
                     indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
