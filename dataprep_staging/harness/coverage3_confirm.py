"""WP A1 second repair round: pre-registered confirmation of the coverage3 interval.

    python -m harness.coverage3_confirm OUT.json     (from dataprep_staging/, in a CPU slot)

The design-time sweep (``coverage3_design.py``, seeds ``20291012 + ...``) chose
``coverage3.DEFAULT_METHOD = "upper90"`` (Wilson on n_eff from the one-sided 0.90 upper bound
of the ANOVA ICC, Student-t on k - 1 df, level 0.95) and the envelope k >= 20 clusters. This
script re-tests the frozen choice on fresh seeds (``20301012 + 100 * i + v``) and on
configurations the sweep did not contain. It is pre-registered as ``A1r2.coverage3_confirm.v1``
with ``harness.prereg3`` and refuses to run unless ``prereg3.assert_can_score`` passes.

Criterion (A1 acceptance, scoped): with a true coverage rate of 0.90, the interval contains 0.90
in at least 89 % of 10,000 draws in EVERY primary configuration (inside the envelope: k >= 20
clusters, ICC <= 0.7). Reported, never gating: the secondary configurations (outside the
envelope: k < 20, including the A1 verifier's failing cases, S1 14 x 13, W1 7 x 2, E6 5 x 6 and
the gap-fill pilot's 9-gap and 3-gap structures; and two-point mixtures inside it), the frozen
``anova`` interval on the same draws, power (share of draws whose interval excludes 0.90 at a
true 0.80 or 0.97), rule ``gate-3`` pass rates at 0.80 / 0.90 / 0.97, the window-only false-fail
rate at 0.90, and the level-0.90 coverage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import coverage3  # noqa: E402
from harness.coverage_sim import draw_totals  # noqa: E402

PREREG_ID = "A1r2.coverage3_confirm.v1"
TEST_ID = "A1r2.coverage3_confirm.v1.T1"
N_DRAWS = 10_000
CRITERION = 0.89
SEED0 = 20301012
TARGET = 0.90
MIN_K = 20
MAX_RHO = 0.7

#: cluster sizes of the head gap-fill pilot's families (clean wells, ens_avg; pre-origin pilot
#: output, design-time evidence), copied from coverage3_design.json pilot sizes
PILOT_SIZES_FILE = "results/dataprep/reports/A1_evidence/repair2/pilot_cluster_sizes.json"


def _eq(k, m):
    return [float(m)] * k


def configs(pilot_sizes: dict) -> list[dict]:
    """Declared configurations: name, sizes (key or list), rho, kind, role."""
    c = []
    for k in (20, 24, 40, 60, 100):
        for m in (3, 8, 24, 100):
            for r in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.7):
                c.append({"name": f"eq_k{k}_m{m}_rho{r}", "k": k, "sizes": _eq(k, m),
                          "rho": r, "kind": "beta", "role": "primary"})
    for k in (20, 30, 50):
        for sig in (0.9, 1.5):
            for r in (0.2, 0.45, 0.7):
                c.append({"name": f"lognorm_k{k}_s{sig}_rho{r}", "k": k,
                          "sizes": f"lognormal:{k}:{sig}", "rho": r, "kind": "beta",
                          "role": "primary"})
    for fam in ("E_lead66", "E_trail48", "S_gap06", "S_gap12", "S_gap24"):
        for r in (0.15, 0.45, 0.7):
            sz = pilot_sizes[fam]
            c.append({"name": f"pilot_{fam}_rho{r}", "k": len(sz),
                      "sizes": [float(v) for v in sz], "rho": r, "kind": "beta",
                      "role": "primary"})
    for n in (30, 50, 200):
        c.append({"name": f"indep_n{n}", "k": n, "sizes": _eq(n, 1), "rho": 0.0,
                  "kind": "beta", "role": "primary"})
    for r in (0.05, 0.2, 0.4, 0.6):
        c.append({"name": f"V3_74well_x42_rho{r}", "k": 74, "sizes": _eq(74, 42), "rho": r,
                  "kind": "beta", "role": "primary"})
    for r in (0.0, 0.1, 0.3, 0.5, 0.7):
        c.append({"name": f"H2_300gap_rho{r}", "k": 300, "sizes": "h2", "rho": r,
                  "kind": "beta", "role": "primary"})
    # inside the envelope, a shape the beta-binomial does not have (never gating)
    for name, k, m, pw in (("mix_k30_x10_1.0/0.6", 30, 10, (1.0, 0.6, 0.75)),
                           ("mix_k74_x42_1.0/0.6", 74, 42, (1.0, 0.6, 0.75)),
                           ("mix_k24_x24_1.0/0.5", 24, 24, (1.0, 0.5, 0.8))):
        c.append({"name": name, "k": k, "sizes": _eq(k, m), "rho": None, "kind": "mix",
                  "mix": list(pw), "role": "secondary_in_envelope"})
    # outside the envelope (k < 20): reported, never gating
    out = [("verifier_14x13_rho0.4", 14, 13, 0.4), ("verifier_14x13_rho0.5", 14, 13, 0.5),
           ("verifier_14x13_rho0.7", 14, 13, 0.7), ("verifier_7x12_rho0.6", 7, 12, 0.6),
           ("verifier_10x50_rho0.5", 10, 50, 0.5), ("verifier_12x20_rho0.4", 12, 20, 0.4),
           ("S1_14x13_rho0.0", 14, 13, 0.0), ("S1_14x13_rho0.1", 14, 13, 0.1),
           ("S1_14x13_rho0.3", 14, 13, 0.3), ("S1_14x13_rho0.6", 14, 13, 0.6),
           ("S1_14x13_rho0.8", 14, 13, 0.8), ("W1_7x2_rho0.2", 7, 2, 0.2),
           ("W1_7x2_rho0.5", 7, 2, 0.5), ("E6_5x6_rho0.1", 5, 6, 0.1),
           ("E6_5x6_rho0.3", 5, 6, 0.3), ("E6_5x6_rho0.5", 5, 6, 0.5),
           ("k3_x10_rho0.1", 3, 10, 0.1), ("k10_x50_rho0.7", 10, 50, 0.7),
           ("k14_x50_rho0.7", 14, 50, 0.7), ("k12_x100_rho0.3", 12, 100, 0.3)]
    for name, k, m, r in out:
        c.append({"name": name, "k": k, "sizes": _eq(k, m), "rho": r, "kind": "beta",
                  "role": "secondary_outside_envelope"})
    for fam in ("N_net1", "N_net2", "H_half12"):
        for r in (0.1, 0.3, 0.5):
            sz = pilot_sizes[fam]
            c.append({"name": f"pilot_{fam}_rho{r}", "k": len(sz),
                      "sizes": [float(v) for v in sz], "rho": r, "kind": "beta",
                      "role": "secondary_outside_envelope"})
    return c


def sizes(spec, rng) -> np.ndarray:
    if isinstance(spec, list):
        return np.asarray(spec, float)
    if spec == "h2":
        return np.tile(np.array([1, 2, 3, 6, 12, 24], float), 50)
    if spec.startswith("lognormal:"):
        _, k, sig = spec.split(":")
        return np.maximum(1, np.round(rng.lognormal(np.log(12), float(sig), int(k)))).astype(
            float)
    raise ValueError(spec)


def draw_mix(m: np.ndarray, pw, n: int, rng) -> np.ndarray:
    hi, lo, w = pw
    pi = np.where(rng.random((n, m.size)) < w, hi, lo)
    return rng.binomial(m.astype(int)[None, :].repeat(n, 0), pi).astype(float)


def mix_stats(pw) -> tuple[float, float]:
    hi, lo, w = pw
    mean = w * hi + (1 - w) * lo
    var = w * (hi - mean) ** 2 + (1 - w) * (lo - mean) ** 2
    return mean, var / (mean * (1 - mean))


def _cov(r) -> np.ndarray:
    return (r["lo"] <= TARGET) & (r["hi"] >= TARGET)


def run(pilot_sizes: dict) -> dict:
    meth = coverage3.DEFAULT_METHOD
    out = {"prereg_id": PREREG_ID, "test_id": TEST_ID, "n_draws": N_DRAWS, "seed0": SEED0,
           "method": meth, "envelope": coverage3.ENVELOPE, "level": coverage3.GATE_LEVEL,
           "criterion": f"actual coverage of a true {TARGET} >= {CRITERION} in every primary "
                        f"configuration (k >= {MIN_K}, ICC <= {MAX_RHO})", "configs": []}
    for i, c in enumerate(configs(pilot_sizes)):
        m = sizes(c["sizes"], np.random.default_rng(SEED0 + 100 * i + 99))
        k, N = m.size, float(m.sum())
        m_w = float((m ** 2).sum() / N)
        rng = np.random.default_rng(SEED0 + 100 * i)
        if c["kind"] == "mix":
            mean, rho = mix_stats(c["mix"])
            Y = draw_mix(m, c["mix"], N_DRAWS, rng)
        else:
            rho = c["rho"]
            Y = draw_totals(m, TARGET, rho, N_DRAWS, rng)
        g = coverage3.gate_from_totals_batch(Y, m, meth)
        a = coverage3.interval_batch(Y, m, "anova")
        l90 = coverage3.interval_batch(Y, m, meth, level=0.90)
        row = {"name": c["name"], "role": c["role"], "kind": c["kind"], "k": k, "n": N,
               "m_w": m_w, "rho_true": rho, "true_n_eff": N / (1 + (m_w - 1) * rho),
               "actual_coverage": float(np.mean(g["in_ci"])),
               "meets_criterion": bool(np.mean(g["in_ci"]) >= CRITERION),
               "anova_actual_coverage": float(np.mean(_cov(a))),
               "level0.90_actual_coverage": float(np.mean(_cov(l90))),
               "mean_n_eff": float(np.mean(g["n_eff"])),
               "anova_mean_n_eff": float(np.mean(a["n_eff"])),
               "mean_width": float(np.mean(g["hi"] - g["lo"])),
               "gate3_pass_rate_true0.9": float(np.mean(g["pass"])),
               "window_only_fail_rate_true0.9": float(np.mean(
                   g["in_ci"] & g["window_applies"] & ~g["in_window"])),
               "window_applies_rate_true0.9": float(np.mean(g["window_applies"]))}
        if c["kind"] == "beta":
            for v, pp in ((1, 0.80), (2, 0.97)):
                Yp = draw_totals(m, pp, rho, N_DRAWS,
                                 np.random.default_rng(SEED0 + 100 * i + v))
                gp = coverage3.gate_from_totals_batch(Yp, m, meth)
                ap = coverage3.interval_batch(Yp, m, "anova")
                row[f"power_true{pp}"] = float(np.mean(~gp["in_ci"]))
                row[f"anova_power_true{pp}"] = float(np.mean(~_cov(ap)))
                row[f"gate3_pass_rate_true{pp}"] = float(np.mean(gp["pass"]))
        out["configs"].append(row)
        if i % 20 == 0:
            print(f"{i:3d} {row['name']:34s} {row['role'][:9]} cov {row['actual_coverage']:.4f}"
                  f" (anova {row['anova_actual_coverage']:.4f})", flush=True)
    rows = out["configs"]
    prim = [r for r in rows if r["role"] == "primary"]
    out["primary_n"] = len(prim)
    out["primary_min_actual_coverage"] = min(r["actual_coverage"] for r in prim)
    out["primary_argmin"] = min(prim, key=lambda r: r["actual_coverage"])["name"]
    out["primary_below_criterion"] = [r["name"] for r in prim if not r["meets_criterion"]]
    out["primary_anova_min_actual_coverage"] = min(r["anova_actual_coverage"] for r in prim)
    out["primary_anova_below_criterion"] = [r["name"] for r in prim
                                            if r["anova_actual_coverage"] < CRITERION]
    for role in ("secondary_in_envelope", "secondary_outside_envelope"):
        rr = [r for r in rows if r["role"] == role]
        out[f"{role}_min_actual_coverage"] = min(r["actual_coverage"] for r in rr)
        out[f"{role}_below_criterion"] = [r["name"] for r in rr if not r["meets_criterion"]]
        out[f"{role}_anova_below_criterion"] = [r["name"] for r in rr
                                                if r["anova_actual_coverage"] < CRITERION]
    out["primary_mean_power_true0.8"] = float(np.mean([r["power_true0.8"] for r in prim
                                                       if "power_true0.8" in r]))
    out["primary_anova_mean_power_true0.8"] = float(np.mean([r["anova_power_true0.8"]
                                                             for r in prim
                                                             if "anova_power_true0.8" in r]))
    out["pass"] = not out["primary_below_criterion"]
    return out


def main(argv=None) -> int:
    from harness import prereg3
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m harness.coverage3_confirm OUT.json")
    ok = prereg3.assert_can_score(PREREG_ID, "A1", test_id=TEST_ID)
    from common.paths import from_repo
    pilot_sizes = json.loads(from_repo(PILOT_SIZES_FILE).read_text(encoding="utf-8"))["sizes"]
    t0 = time.time()
    res = run(pilot_sizes)
    res["prereg_sha256"] = ok["sha256"]
    res["post_registration_looks"] = ok["post_registration_looks"]
    res["unverifiable_prior_looks"] = len(ok["unverifiable_prior_looks"])
    res["seconds"] = round(time.time() - t0, 1)
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(argv[0], "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in (
        "primary_n", "primary_min_actual_coverage", "primary_argmin", "primary_below_criterion",
        "primary_anova_min_actual_coverage", "secondary_in_envelope_min_actual_coverage",
        "secondary_outside_envelope_min_actual_coverage",
        "secondary_outside_envelope_below_criterion", "primary_mean_power_true0.8",
        "primary_anova_mean_power_true0.8", "pass", "seconds")}, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
