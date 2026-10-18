"""WP A1 repair round: pre-registered evaluation of the repaired gate rule ``gate-2``.

    python -m harness.gate_window_confirm OUT.json      (from dataprep_staging/, in a CPU slot)

The verifier showed that the first gate (``gate-1``: point window when ``n_eff_hat >= 30``)
applies the 0.85-0.95 window in strata whose true n_eff is far below 30 (5 regions x 6 rows at
ICC 0.5: true n_eff 8.6, ``n_eff_hat >= 30`` in 47 % of draws, 43 % of calibrated products
failed through the window alone). ``harness.coverage2`` (rule ``gate-2``) applies the window
only when the one-sided 0.95 lower confidence bound of n_eff is at least 30. This script draws
fresh clustered binomial data (seeds ``20281012 + 100 * i + v``; the design run used
``20261012 + ...`` and the coverage confirmation ``20271012 + ...``) for the 23 configurations
of ``coverage_confirm.CONFIGS`` and scores both rules on the same draws. It is pre-registered
as ``A1r.gate_window.v1`` with ``harness.prereg2`` and refuses to run unless
``prereg2.assert_can_score`` passes.

Criteria (all must hold; fixed in the pre-registration before any draw):

* G1 - in every primary configuration whose true n_eff is below 30, a product whose true
  coverage is 0.90 fails through the point window alone (interval contains 0.90, window
  applies, point outside 0.85-0.95) in at most 10 % of draws under ``gate-2``;
* G2 - in every primary configuration with true n_eff >= 100 and at least 30 clusters, the
  window applies in at least 90 % of draws at a true 0.90 under ``gate-2`` (the repair does
  not switch the window off where the plan wants it);
* G3 - the interval is unchanged: ``gate-2`` and ``gate-1`` agree on "interval contains 0.90"
  in every draw of every configuration.

Reported, never gating: both rules' window rates, full-rule pass rates at true 0.80 / 0.90 /
0.97, mean ``n_eff_hat`` and ``n_eff_lower`` against the true n_eff, the share of draws whose
``rho_hat`` is truncated to 0 and whose lower bound exceeds the true n_eff, the secondary
mixtures, and the interval's actual coverage at level 0.90 (the reading of the plan's 0.89
threshold that the A1 verifier raised; the gate's level stays 0.95).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import coverage, coverage2  # noqa: E402
from harness.coverage_confirm import CONFIGS, draw_mix, mix_icc, sizes  # noqa: E402
from harness.coverage_sim import draw_totals  # noqa: E402

PREREG_ID = "A1r.gate_window.v1"
TEST_ID = "A1r.gate_window.v1.T1"
N_DRAWS = 10_000
SEED0 = 20281012
G1_MAX_WINDOW_ONLY_FAILS = 0.10
G2_MIN_WINDOW_APPLIES = 0.90
G2_MIN_TRUE_N_EFF = 100.0
G2_MIN_K = 30
G1_TRUE_N_EFF_BELOW = 30.0


def _both(Y, m):
    g2 = coverage2.gate_from_totals_batch(Y, m)
    big1 = g2["n_eff"] >= coverage.N_EFF_WINDOW_MIN
    pass1 = g2["in_ci"] & (g2["in_window"] | ~big1)
    return g2, big1, pass1


def run() -> dict:
    out = {"prereg_id": PREREG_ID, "test_id": TEST_ID, "n_draws": N_DRAWS, "seed0": SEED0,
           "rule": {"name": coverage2.RULE, "window_conf": coverage2.WINDOW_CONF,
                    "level": coverage.GATE_LEVEL, "small_k": coverage.DEFAULT_SMALL_K,
                    "n_eff_window_min": coverage.N_EFF_WINDOW_MIN,
                    "point_window": list(coverage.POINT_WINDOW)},
           "criteria": {"G1": f"window-only fails at true 0.90 <= {G1_MAX_WINDOW_ONLY_FAILS} "
                              f"in every primary configuration with true n_eff < "
                              f"{G1_TRUE_N_EFF_BELOW}",
                        "G2": f"window applies in >= {G2_MIN_WINDOW_APPLIES} of draws at true "
                              f"0.90 in every primary configuration with true n_eff >= "
                              f"{G2_MIN_TRUE_N_EFF} and k >= {G2_MIN_K}",
                        "G3": "gate-2 and gate-1 agree on 'interval contains 0.90' in every "
                              "draw"},
           "configs": []}
    for ci, (name, k, msz, rho, kind, role) in enumerate(CONFIGS):
        m = sizes(msz, k, np.random.default_rng(SEED0 + 100 * ci + 99))
        assert m.size == k
        N = float(m.sum())
        m_w = float((m ** 2).sum() / N)
        rho_true = mix_icc() if kind == "mix" else rho
        true_neff = N / (1 + (m_w - 1) * rho_true)
        rng = np.random.default_rng(SEED0 + 100 * ci)
        Y = draw_mix(m, N_DRAWS, rng) if kind == "mix" else \
            draw_totals(m, coverage.TARGET, rho, N_DRAWS, rng)
        g2, big1, pass1 = _both(Y, m)
        r1 = coverage.interval_from_totals_batch(Y, m, coverage.GATE_LEVEL,
                                                 coverage.DEFAULT_SMALL_K)
        same = bool(np.array_equal(r1["lo"], g2["lo"]) and np.array_equal(r1["hi"], g2["hi"]))
        r90 = coverage.interval_from_totals_batch(Y, m, 0.90, coverage.DEFAULT_SMALL_K)
        cov90 = float(np.mean((r90["lo"] <= coverage.TARGET) & (r90["hi"] >= coverage.TARGET)))
        wonly2 = g2["in_ci"] & g2["window_applies"] & ~g2["in_window"]
        wonly1 = g2["in_ci"] & big1 & ~g2["in_window"]
        row = {"name": name, "role": role, "kind": kind, "k": k, "n": N, "m_w": m_w,
               "rho_true": rho_true, "true_n_eff": true_neff,
               "g1_member": bool(role == "primary" and true_neff < G1_TRUE_N_EFF_BELOW),
               "g2_member": bool(role == "primary" and true_neff >= G2_MIN_TRUE_N_EFF
                                 and k >= G2_MIN_K),
               "interval_coverage_level0.95": float(np.mean(g2["in_ci"])),
               "interval_coverage_level0.90": cov90,
               "g3_interval_identical": same,
               "gate2_window_applies": float(np.mean(g2["window_applies"])),
               "gate1_window_applies": float(np.mean(big1)),
               "gate2_window_only_fail_true0.9": float(np.mean(wonly2)),
               "gate1_window_only_fail_true0.9": float(np.mean(wonly1)),
               "gate2_pass_true0.9": float(np.mean(g2["pass"])),
               "gate1_pass_true0.9": float(np.mean(pass1)),
               "mean_n_eff_hat": float(np.mean(g2["n_eff"])),
               "mean_n_eff_lower": float(np.mean(g2["n_eff_lower"])),
               "share_rho_hat_zero": float(np.mean(g2["rho"] == 0.0)),
               "share_n_eff_lower_above_true": float(np.mean(g2["n_eff_lower"] > true_neff))}
        if kind == "beta":
            for v, ptrue in ((1, 0.80), (2, 0.97)):
                Yp = draw_totals(m, ptrue, rho, N_DRAWS,
                                 np.random.default_rng(SEED0 + 100 * ci + v))
                h2, _b1, p1 = _both(Yp, m)
                row[f"gate2_pass_true{ptrue}"] = float(np.mean(h2["pass"]))
                row[f"gate1_pass_true{ptrue}"] = float(np.mean(p1))
        out["configs"].append(row)
        print(f"{name:30s} {role:9s} true n_eff {true_neff:7.1f} | window g1 "
              f"{row['gate1_window_applies']:.3f} g2 {row['gate2_window_applies']:.3f} | "
              f"window-only fails g1 {row['gate1_window_only_fail_true0.9']:.3f} g2 "
              f"{row['gate2_window_only_fail_true0.9']:.3f}", flush=True)
    cf = out["configs"]
    g1 = [c for c in cf if c["g1_member"]]
    g2m = [c for c in cf if c["g2_member"]]
    out["G1"] = {"members": [c["name"] for c in g1],
                 "max_window_only_fail": max(c["gate2_window_only_fail_true0.9"] for c in g1),
                 "failing": [c["name"] for c in g1
                             if c["gate2_window_only_fail_true0.9"] > G1_MAX_WINDOW_ONLY_FAILS]}
    out["G1"]["pass"] = not out["G1"]["failing"]
    out["G2"] = {"members": [c["name"] for c in g2m],
                 "min_window_applies": min(c["gate2_window_applies"] for c in g2m),
                 "failing": [c["name"] for c in g2m
                             if c["gate2_window_applies"] < G2_MIN_WINDOW_APPLIES]}
    out["G2"]["pass"] = not out["G2"]["failing"]
    out["G3"] = {"pass": all(c["g3_interval_identical"] for c in cf)}
    prim = [c for c in cf if c["role"] == "primary"]
    out["info_level0.90_min_primary"] = min(c["interval_coverage_level0.90"] for c in prim)
    out["info_level0.90_below_0.89"] = [c["name"] for c in prim
                                         if c["interval_coverage_level0.90"] < 0.89]
    out["pass"] = bool(out["G1"]["pass"] and out["G2"]["pass"] and out["G3"]["pass"])
    return out


def main(argv=None) -> int:
    from harness import prereg2
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m harness.gate_window_confirm OUT.json")
    ok = prereg2.assert_can_score(PREREG_ID, "A1", test_id=TEST_ID)
    t0 = time.time()
    res = run()
    res["prereg_sha256"] = ok["sha256"]
    res["seconds"] = round(time.time() - t0, 1)
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(argv[0], "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("G1", "G2", "G3", "info_level0.90_min_primary",
                                          "info_level0.90_below_0.89", "pass", "seconds")},
                     indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
