"""WP A1 second repair round: design-time sweep of candidate coverage intervals (coverage3).

    python -m harness.coverage3_design OUT.json     (from dataprep_staging/, in a CPU slot)

DESIGN-TIME EVIDENCE. This sweep chooses ``harness.coverage3.DEFAULT_METHOD``; a choice made on
these draws is re-tested on fresh seeds by the pre-registered ``harness.coverage3_confirm``.

Declared before the first run (2026-10-12, second repair round), not changed after:

* Candidates: ``harness.coverage3.METHODS`` (anova = the frozen interval, anova_z0, median,
  upper75, upper90, cr, kg_cp), all at level 0.95.
* Draws: beta-binomial cluster coverage with mean p and ICC rho (``coverage_sim.draw_totals``);
  10,000 draws per configuration at p = 0.90 (coverage), 0.80 and 0.97 (power: share of draws
  whose interval excludes 0.90). Seeds ``20291012 + 100 * i + v`` (v = 0 / 1 / 2 for p = 0.90 /
  0.80 / 0.97; never used before; earlier runs used 20261012, 20271012, 20281012).
* Configurations: (a) equal clusters, k in {3, 5, 7, 10, 14, 20, 30, 74} x m in {2, 6, 13, 50}
  x rho in {0, 0.1, ..., 0.8} (288); (b) the cluster sizes of the head gap-fill pilot's
  families N_net1, N_net2 (9 gaps), H_half12 (3), E_lead66 (35) and S_gap12 (39), clean wells,
  method ens_avg (pre-origin pilot output, design-time evidence), at rho in {0.1, 0.3, 0.5, 0.7}
  (20); (c) the 20 primary configurations of A1.coverage_confirm.v1 (sizes re-drawn from these
  seeds).
* Envelope R: every configuration with rho <= 0.7 (the head gap-fill pilot's ANOVA ICC
  estimates reach 0.69, S_gap02; the verifier's failing cases reach 0.7). Configurations with
  rho = 0.8 are reported, never decide.
* Decision rule: S = candidates whose actual coverage is >= 0.89 in every configuration of R.
  If S is not empty, DEFAULT_METHOD = the member of S with the highest mean power at p = 0.80
  over R (ties: the earlier name in METHODS). If S is empty, DEFAULT_METHOD = the candidate
  meeting 0.89 in the most configurations of R (ties: the higher minimum coverage over R), and
  the envelope is restricted to the smallest number of clusters k_min such that it meets 0.89
  in every configuration of R with k >= k_min; strata with fewer clusters are reported as
  "uncalibrated" by ``coverage3.coverage_gate``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PILOTS, rel_to_repo  # noqa: E402

from harness import coverage3  # noqa: E402
from harness.coverage_confirm import CONFIGS as CONFIRM1_CONFIGS  # noqa: E402
from harness.coverage_confirm import sizes as confirm1_sizes  # noqa: E402
from harness.coverage_sim import draw_totals  # noqa: E402

N_DRAWS = 10_000
CRITERION = 0.89
SEED0 = 20291012
TARGET = 0.90
P_POWER = (0.80, 0.97)
ENVELOPE_MAX_RHO = 0.7
EQUAL_K = (3, 5, 7, 10, 14, 20, 30, 74)
EQUAL_M = (2, 6, 13, 50)
RHOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
PILOT_FAMILIES = ("N_net1", "N_net2", "H_half12", "E_lead66", "S_gap12")
PILOT_RHOS = (0.1, 0.3, 0.5, 0.7)
PILOT_METHOD = "ens_avg"


def pilot_cluster_sizes() -> tuple[dict, dict]:
    """Cluster (gap) sizes of the gap-fill pilot's families (clean wells, ens_avg)."""
    import pandas as pd
    prod = guard.open_product(PILOTS / "head_gapfill_pilot", "design_time_evidence")
    p = prod.path("out/pilot_ensembles.parquet")
    D = pd.read_parquet(p, columns=["Lname", "method", "gid", "suspect", "step_well"])
    D = D[~D.suspect & ~D.step_well & (D.method == PILOT_METHOD)]
    out = {}
    for fam in PILOT_FAMILIES:
        g = D[D.Lname == fam]
        out[fam] = sorted(int(v) for v in g.groupby("gid").size().to_numpy())
    return out, {"path": rel_to_repo(p), "sha256": sha256_file(p),
                 "use": "design_time_evidence", "selection": f"clean wells, method "
                                                            f"{PILOT_METHOD}, cluster = gid"}


def configs() -> tuple[list, dict]:
    cfg = []
    for k in EQUAL_K:
        for m in EQUAL_M:
            for r in RHOS:
                cfg.append({"name": f"eq_k{k}_m{m}_rho{r}", "group": "equal", "k": k,
                            "sizes": [float(m)] * k, "rho": r})
    ps, prov = pilot_cluster_sizes()
    for fam, sz in ps.items():
        for r in PILOT_RHOS:
            cfg.append({"name": f"pilot_{fam}_rho{r}", "group": "pilot", "k": len(sz),
                        "sizes": [float(s) for s in sz], "rho": r})
    for i, (name, k, msz, rho, _kind, role) in enumerate(CONFIRM1_CONFIGS):
        if role != "primary":
            continue
        m = confirm1_sizes(msz, k, np.random.default_rng(SEED0 + 90_000 + i))
        cfg.append({"name": f"c1_{name}", "group": "confirm1_primary", "k": k,
                    "sizes": [float(v) for v in m], "rho": rho})
    return cfg, prov


def run() -> dict:
    cfg, prov = configs()
    meths = coverage3.METHODS
    rows = []
    for i, c in enumerate(cfg):
        m = np.asarray(c["sizes"], float)
        N = float(m.sum())
        m_w = float((m ** 2).sum() / N)
        row = {k: c[k] for k in ("name", "group", "k", "rho")}
        row.update({"n": N, "m_w": m_w, "true_n_eff": N / (1 + (m_w - 1) * c["rho"]),
                    "in_envelope": c["rho"] <= ENVELOPE_MAX_RHO, "coverage": {},
                    "power_p0.8": {}, "power_p0.97": {}, "mean_n_eff": {}})
        Y = draw_totals(m, TARGET, c["rho"], N_DRAWS, np.random.default_rng(SEED0 + 100 * i))
        Yp = {pp: draw_totals(m, pp, c["rho"], N_DRAWS,
                              np.random.default_rng(SEED0 + 100 * i + v + 1))
              for v, pp in enumerate(P_POWER)}
        for meth in meths:
            r = coverage3.interval_batch(Y, m, meth)
            row["coverage"][meth] = float(np.mean((r["lo"] <= TARGET) & (r["hi"] >= TARGET)))
            row["mean_n_eff"][meth] = float(np.mean(r["n_eff"]))
            for pp, Yq in Yp.items():
                rq = coverage3.interval_batch(Yq, m, meth)
                row[f"power_p{pp}"][meth] = float(np.mean((rq["lo"] > TARGET)
                                                          | (rq["hi"] < TARGET)))
        rows.append(row)
        if i % 25 == 0:
            print(i, row["name"], {k: round(v, 3) for k, v in row["coverage"].items()},
                  flush=True)
    env = [r for r in rows if r["in_envelope"]]
    summ = {}
    for meth in meths:
        covs = [r["coverage"][meth] for r in env]
        summ[meth] = {
            "min_coverage_envelope": min(covs),
            "n_below_envelope": int(sum(c < CRITERION for c in covs)),
            "below_envelope": [r["name"] for r in env if r["coverage"][meth] < CRITERION],
            "mean_power_p0.8_envelope": float(np.mean([r["power_p0.8"][meth] for r in env])),
            "mean_power_p0.97_envelope": float(np.mean([r["power_p0.97"][meth] for r in env])),
            "min_coverage_rho0.8": min(r["coverage"][meth] for r in rows
                                       if not r["in_envelope"]),
            "min_coverage_confirm1_primary": min(r["coverage"][meth] for r in rows
                                                 if r["group"] == "confirm1_primary"),
            "k_min_all_pass": None}
        for kmin in sorted({r["k"] for r in env}):
            if all(r["coverage"][meth] >= CRITERION for r in env if r["k"] >= kmin):
                summ[meth]["k_min_all_pass"] = kmin
                break
    S = [mm for mm in meths if summ[mm]["n_below_envelope"] == 0]
    if S:
        best = max(S, key=lambda mm: (summ[mm]["mean_power_p0.8_envelope"],
                                      -meths.index(mm)))
        decision = {"S": S, "default_method": best, "criterion_met_in_envelope": True,
                    "envelope": {"max_rho_simulated": ENVELOPE_MAX_RHO, "min_k": 1}}
    else:
        best = max(meths, key=lambda mm: (-summ[mm]["n_below_envelope"],
                                          summ[mm]["min_coverage_envelope"]))
        decision = {"S": [], "default_method": best, "criterion_met_in_envelope": False,
                    "envelope": {"max_rho_simulated": ENVELOPE_MAX_RHO,
                                 "min_k": summ[best]["k_min_all_pass"]}}
    return {"design_time": True, "seed0": SEED0, "n_draws": N_DRAWS, "criterion": CRITERION,
            "level": coverage3.GATE_LEVEL, "envelope_max_rho": ENVELOPE_MAX_RHO,
            "pilot_sizes_provenance": prov, "n_configs": len(rows),
            "summary": summ, "decision": decision, "configs": rows}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m harness.coverage3_design OUT.json")
    t0 = time.time()
    res = run()
    res["seconds"] = round(time.time() - t0, 1)
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(argv[0], "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "below_envelope"}
                                  for k, v in res["summary"].items()},
                      "decision": res["decision"], "seconds": res["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
