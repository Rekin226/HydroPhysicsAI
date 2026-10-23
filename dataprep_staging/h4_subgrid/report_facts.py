"""WP H4: descriptive facts for the report, from the registered evidence only (no new fit).

Reads reports/H4_evidence/registered/{scores.json, arrays.npz} and writes
reports/H4_evidence/report_facts.json: per-quartile and per-county breakdowns, supported Q4
wells, the pumping-cycle wells H1a flagged, multiplier spreads and placebo distributions.
Usage: python report_facts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.paths import DATAPREP  # noqa: E402

EVID = DATAPREP / "reports/H4_evidence"
PUMP_CYCLE = ("07080132", "07080122", "10050211")     # H1a issues for downstream


def main() -> None:
    sc = json.loads((EVID / "registered/scores.json").read_text())
    z = np.load(EVID / "registered/arrays.npz", allow_pickle=False)
    q, ok, n = z["quartile"], z["score_ok"], z["npump"]
    sup = n > 0
    f = {}
    for name, key in (("t1", "mse_t1"), ("t2", "mse_t2")):
        b, c, cl = z["mse_base"], z[key], z["mse_clim"]
        rows = {}
        for k in (1, 2, 3, 4):
            for grp, sel in (("supported", sup), ("unsupported", ~sup)):
                s = ok & (q == k) & sel
                if s.any():
                    rows[f"Q{k}_{grp}"] = {
                        "n": int(s.sum()), "rmse_base": float(np.sqrt(b[s].mean())),
                        "rmse_corrected": float(np.sqrt(c[s].mean())),
                        "rmse_clim": float(np.sqrt(cl[s].mean())),
                        "excess_base": float((b[s] - cl[s]).sum()),
                        "excess_corrected": float((c[s] - cl[s]).sum()),
                        "n_improved": int((c[s] < b[s]).sum())}
        f[name] = rows
    f["q4_wells"] = [{"sid": str(z["sids"][i]), "county": str(z["county"][i]),
                      "layer": int(z["layer"][i]), "npump_2km": int(n[i]),
                      "sd_fit": float(z["sd_fit"][i]), "scored": bool(ok[i]),
                      "mse_base": float(z["mse_base"][i]), "mse_clim": float(z["mse_clim"][i]),
                      "mse_t1": float(z["mse_t1"][i]), "mse_t2": float(z["mse_t2"][i]),
                      "sd_corr_t1": float(np.nanstd(z["corr_t1"][i, 1:]))}
                     for i in np.flatnonzero(q == 4)]
    f["pump_cycle_wells"] = {s: {"quartile": int(q[i]), "scored": bool(ok[i]),
                                 "npump_2km": int(n[i]), "n_fit": int(z["n_train"][i]),
                                 "n_test": int(z["n_test"][i])}
                             for i, s in enumerate(z["sids"]) if s in PUMP_CYCLE}
    a = np.array([v for v in sc["primary"]["t2"]["a_by_fold"].values()])
    f["t2_multiplier_range"] = {"a_s": [float(a[:, 0].min()), float(a[:, 0].max())],
                                "a_l": [float(a[:, 1].min()), float(a[:, 1].max())]}
    for fam in ("shuffle", "distant"):
        for t in ("t1", "t2"):
            v = np.asarray(sc["placebos"][fam][t])
            f[f"placebo_{fam}_{t}"] = {"median": float(np.median(v)),
                                       "p90": float(np.quantile(v, 0.9)),
                                       "max": float(v.max()), "min": float(v.min())}
        aa = np.asarray(sc["placebos"][fam]["a_t1"])
        f[f"placebo_{fam}_a_t1_median"] = [float(x) for x in np.median(aa, axis=0)]
    corr = z["corr_t1"][:, 1:]
    f["corr_t1_sd_supported_m"] = {str(p): float(np.percentile(np.std(corr[sup], axis=1), p))
                                   for p in (0, 25, 50, 75, 100)}
    (EVID / "report_facts.json").write_text(json.dumps(f, indent=1) + "\n")
    print(json.dumps({k: v for k, v in f.items() if k != "q4_wells"}, indent=1))


if __name__ == "__main__":
    main()
