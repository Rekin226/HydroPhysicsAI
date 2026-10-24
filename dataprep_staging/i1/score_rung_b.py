"""Score rung B (``I1.B.v1``) against Q0 on the declared gates, exactly as I1.Rstar.v1 was
scored on 2026-10-22 (the same metric code: ``score_g2.metrics``, the temporal run's fair
columns, the forward plausibility rate and ``score_p2f_rung.py``).

Gates (declaration ``thresholds.passes_against_Q0_if_all``):

* k-fold out-of-fold R2 >= Q0 - 0.01;
* Changhua and Yunlin county_corr >= Q0 - 0.02; county_mean_rmse <= Q0 x 1.02;
* leveling out-of-fold R2 >= Q0 - 0.02;
* temporal fair RMSE ratio <= Q0 x 1.01 and fair verdict PASS;
* forward plausibility: county-mean VEP subsidence rate Dec 2022 -> Dec 2025 / 3 >= 0 cm/yr in
  Changhua and in the rest of the fan (county map v2);
* B2 only: the P2f Yunlin model median ratio inside the observed 90 % interval [0.521, 0.954].
  B1's P2f value is reported, not gating.

``--check rs`` runs the same code on R* and Q0 (spent runs; a code check against
results/dataprep/reports/I1_Rstar/report.json; nothing is written).

Run from the repository root:
    python dataprep_staging/i1/score_rung_b.py [--check rs]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_g2 as S  # noqa: E402

RUNS = REPO / "results/twin_runs"
FWD = REPO / "results/twin_forward"
CMAP = REPO / "results/dataprep/zone_county_map/v2/zone_county_map.npz"
OUT = REPO / "results/dataprep/reports/I1_B"
P2F_OBS = (0.521, 0.954)


def temporal(suffix: str) -> dict:
    t = pd.read_csv(RUNS / f"stage3_datum_gate_{suffix}_temporal/stage3_temporal.csv").iloc[0]
    return {"temporal_fair_ratio": float(t["fair_rmse_ratio_fair"]),
            "temporal_fair_verdict": str(t["fair_verdict_fair"])}


def plausibility(suffix: str) -> dict:
    ch = np.load(CMAP)["changhua"].astype(bool)
    z = np.load(FWD / f"datum_gate_{suffix}.npz")
    d = pd.DatetimeIndex(z["dates"])
    s = (z["subs_mean_by_rheology"][0][0] if "subs_mean_by_rheology" in z.files
         else z["subs_mean"][0]) * 100
    i22 = np.where(d <= pd.Timestamp("2022-12-31"))[0][-1]
    i25 = np.where(d <= pd.Timestamp("2025-12-31"))[0][-1]
    i32 = np.where(d <= pd.Timestamp("2032-12-31"))[0][-1]

    def rate(sel):
        return float((s[sel][:, i25] - s[sel][:, i22]).mean() / 3)

    return {"plaus_ch": rate(ch), "plaus_rest": rate(~ch),
            "fan_subsidence_2023_2032_cm": float((s[:, i32] - s[:, i22]).mean())}


def p2f(suffix: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("score_p2f_rung.py")),
         str(FWD / f"datum_gate_{suffix}.npz")],
        cwd=REPO, check=True, capture_output=True, text=True).stdout
    r = json.loads(out)
    yl = next(v["yunlin"] for v in r.values() if "yunlin" in v)
    return {"p2f_yl": float(yl["model_median_ratio"]),
            "p2f_yl_ci90": [float(x) for x in yl["model_ci90"]]}


def score(suffix: str, st: pd.DataFrame) -> dict:
    m = S.metrics(suffix, st, 2027)
    th = json.loads((RUNS / f"stage3_datum_gate_{suffix}/stage3_theta.json").read_text())
    th = th.get("theta", th)  # the readouts sit under "theta", beside "meta"
    return {"kfold": m["kfold_r2"], "corr_ch": m["county_corr:changhua"],
            "corr_yl": m["county_corr:yunlin"], "cmrmse_ch": m["county_mean_rmse:changhua"],
            "cmrmse_yl": m["county_mean_rmse:yunlin"], "leveling": m["leveling_r2"],
            **temporal(suffix), **plausibility(suffix), **p2f(suffix),
            "reported": {k.split(":", 1)[1]: v for k, v in m.items()
                         if k.startswith("reported:")},
            "fitted": {k: v for k, v in th.items()
                       if any(s in k for s in ("eta", "pump_frac", "pump_layer", "tau_ret",
                                               "return_frac"))
                       and not k.startswith("log_") and not k.endswith("_logit")}}


def gates(m: dict, q: dict, p2f_gates: bool) -> dict:
    g = {"kfold >= Q0 - 0.01": m["kfold"] >= q["kfold"] - 0.01,
         "corr_ch >= Q0 - 0.02": m["corr_ch"] >= q["corr_ch"] - 0.02,
         "corr_yl >= Q0 - 0.02": m["corr_yl"] >= q["corr_yl"] - 0.02,
         "cmrmse_ch <= Q0 x 1.02": m["cmrmse_ch"] <= q["cmrmse_ch"] * 1.02,
         "cmrmse_yl <= Q0 x 1.02": m["cmrmse_yl"] <= q["cmrmse_yl"] * 1.02,
         "leveling >= Q0 - 0.02": m["leveling"] >= q["leveling"] - 0.02,
         "temporal ratio <= Q0 x 1.01 and PASS": (
             m["temporal_fair_ratio"] <= q["temporal_fair_ratio"] * 1.01
             and m["temporal_fair_verdict"] == "PASS"),
         "plausibility >= 0 both counties": min(m["plaus_ch"], m["plaus_rest"]) >= 0}
    if p2f_gates:
        g["P2f Yunlin inside [0.521, 0.954]"] = P2F_OBS[0] <= m["p2f_yl"] <= P2F_OBS[1]
    return {k: bool(v) for k, v in g.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", choices=["rs"], help="code check on a spent rung; writes nothing")
    a = ap.parse_args(argv)
    st = pd.read_parquet(S.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    arms = {"rs": True} if a.check else {"b1": False, "b2": True}
    M = {"Q0": score("q0", st)}
    M.update({k: score(k, st) for k in arms})
    G = {k: gates(M[k], M["Q0"], p) for k, p in arms.items()}
    res = {"declaration": "I1.Rstar.v1 (code check)" if a.check else "I1.B.v1",
           "partition": "seed 2027, site tol 200 m; leveling folds seed 2027 (design-time)",
           "metrics": M, "gates": G,
           "passes_all": {k: bool(all(g.values())) for k, g in G.items()}}
    if not a.check:
        p = res["passes_all"]
        arm = "b1" if p["b1"] else ("b2" if p["b2"] else None)
        res["decision"] = (
            f"{arm.upper()} passes every gate: run the declared seed-2029 confirmation of R0, Q0 "
            f"and {arm.upper()} (same gates against Q0)" if arm else
            "B1 and B2 fail; no seed-2029 confirmation is run (declared: only if every gate "
            "passes). What the failure means for the public-supply term follows the "
            "declaration's interpretation, read beside the metrics, not this line.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if not a.check:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "report.json"
        if path.exists():
            raise SystemExit(f"{path} exists; scored once")
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
