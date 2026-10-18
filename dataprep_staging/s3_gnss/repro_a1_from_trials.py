"""WP S3 resume check: the gate's A1 rates recomputed from the archived trial table.

Recomputes, from ``S3_evidence/acceptance/injection_trials_preorigin.csv`` alone, the
registered criterion "found within +-3 days and size within +-5 mm" per trial, and the
recovery rates per family and per step size, and compares them with the trial table's own
``recovered`` column and with ``validation_preorigin.json``. Logs a ``reproduce`` look
under ``S3.gnss_clean_acceptance.v1`` first. No new trial is run and no threshold changes.

    python -I dataprep_staging/s3_gnss/repro_a1_from_trials.py

Writes ``S3_evidence/acceptance/repro_a1_from_trials.json``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import prereg  # noqa: E402

REPO = HERE.parents[1]
EVID = REPO / "results/dataprep/reports/S3_evidence/acceptance"
PREREG_ID = "S3.gnss_clean_acceptance.v1"
TOL_M, TOL_D, THRESH = 0.005, 3.0, 0.90


def main() -> int:
    prereg.load_prereg(PREREG_ID)
    look = prereg.record_look(
        ["s3.injection.seed303.wra.preorigin", "s3.injection.seed303.ngl.preorigin"],
        "gnss_injection:preorigin_background:seed303", "S3", "reproduce",
        prereg_id=PREREG_ID,
        note="resume check: A1 rates recomputed from the archived trial table "
             "(injection_trials_preorigin.csv); no new trial, no threshold change")
    trials_path = EVID / "injection_trials_preorigin.csv"
    t = pd.read_csv(trials_path)
    gate = json.loads((EVID / "validation_preorigin.json").read_text())["A1_injected_steps"]
    crit = (t.found.astype(bool) & (t.date_err_d.abs() <= TOL_D)
            & (t.size_err_m.abs() <= TOL_M + 1e-12))
    out = {"look_logged": {"time": look["time"], "purpose": "reproduce"},
           "trials_sha256": hashlib.sha256(trials_path.read_bytes()).hexdigest(),
           "criterion": "found and |date_err_d| <= 3 and |size_err_m| <= 0.005",
           "rows_where_recomputed_differs_from_recorded": int(
               (crit != t.recovered.astype(bool)).sum()),
           "families": {}}
    worst = 0.0
    for fam, g in t.groupby("family"):
        c = crit[g.index]
        rec = {"n": int(len(g)), "recovered_recomputed": float(c.mean()),
               "recovered_gate_run": gate[fam]["recovered"],
               "pass_at_0.90": bool(c.mean() >= THRESH),
               "by_size": {}}
        worst = max(worst, abs(rec["recovered_recomputed"] - rec["recovered_gate_run"]))
        for amp, gg in g.groupby(g.abs_size_m.round(3)):
            key = f"{int(round(amp * 100))}cm"
            r = float(crit[gg.index].mean())
            rec["by_size"][key] = {"n": int(len(gg)), "recomputed": r,
                                   "gate_run": gate[fam]["by_size"][key]["recovered"]}
            worst = max(worst, abs(r - gate[fam]["by_size"][key]["recovered"]))
        se = g.loc[g.found.astype(bool), "size_err_m"].abs() * 1000
        rec["size_err_mm_p90_recomputed"] = float(np.percentile(se, 90))
        rec["size_err_mm_p90_gate_run"] = gate[fam]["size_err_mm_p90"]
        out["families"][fam] = rec
    out["max_abs_rate_diff_vs_gate_run"] = worst
    (EVID / "repro_a1_from_trials.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
