"""After-the-fact diagnostic of step B (N1.policy_response.v1, scored 2026-10-23): what drives
the within-benchmark variation of the 2007-10 exposure E3?

Changhua's observed beta came out negative (more nearby pumping, less subsidence), which no
pumping response produces. Hypothesis: the back-cast census makes E3's year-to-year changes
record pump ENTRY and EXIT (installations, terminations) rather than pumping intensity. This
splits E3 per interval into pumps with a positive reading in both the first and the last half
year of 2007-01..2010-12 ('continuous') and the rest ('entry/exit'), and reports each part's
share of the within-benchmark variance (after removing benchmark and campaign means, the
variation the registered regression uses), per county.

Reads electricity (pump_electricity/full/v3 pumps_<county>.npz, months <= 2010-12 only) and
the scored intervals' sid, cell, t0, t1, years and county columns; NOT their rates.
Logged as an after_the_fact look. Report only: changes no registered verdict.
Usage: results/twin_runs/par24/py.sh dataprep_staging/n1_policy/diag_b_exposure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR  # noqa: E402
from harness import prereg3  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402
from n1_policy import run_step_b as B  # noqa: E402

FULL = REPO / "results/dataprep/pump_electricity/full/v3"
OUT = REPORTS_DIR / "N1_policy" / "step_b" / "diag_exposure.json"


def pump_split(county: str, n_active: int):
    z = np.load(FULL / f"pumps_{county}.npz", allow_pickle=False)
    dates = pd.PeriodIndex([str(d) for d in z["dates"]], freq="M")
    keep = dates <= pd.Period("2010-12", "M")
    v = np.where(np.isfinite(z["value"][:, keep]), z["value"][:, keep], 0.0)
    d = dates[keep]
    first = (d <= pd.Period("2007-06", "M"))
    last = (d >= pd.Period("2010-07", "M"))
    cont = (v[:, first] > 0).any(1) & (v[:, last] > 0).any(1)
    cell = z["cell"]
    ok = cell >= 0
    out = {}
    for name, sel in (("continuous", cont & ok), ("entry_exit", ~cont & ok)):
        E = np.zeros((n_active, len(d)))
        np.add.at(E, cell[sel], v[sel])
        out[name] = E
    return out, d, {"n_pumps": int(ok.sum()), "n_continuous": int((cont & ok).sum())}


def within(x: np.ndarray, sid: np.ndarray, camp: np.ndarray) -> np.ndarray:
    groups = []
    for g in (sid, camp):
        _, inv = np.unique(g, return_inverse=True)
        groups.append((inv, np.bincount(inv).astype(float)))
    return PL._demean(x[:, None], groups)[:, 0]


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    prereg3.record_look(
        ["N1P:lev_pre2012"], "electricity:2007-01..2010-12", "N1", "after_the_fact",
        prereg_id="N1.policy_response.v1",
        note="after the step B score: split of the 2007-10 exposure E3 into continuous vs "
             "entry/exit pumps at the scored intervals (sid, cell, dates only; no rate or "
             "elevation read); informs a possible post-failure exploratory follow-up")
    iv = pd.read_csv(REPORTS_DIR / "N1_policy/step_b/intervals.csv",
                     usecols=["sid", "cell", "t0", "t1", "years", "changhua", "campaign"])
    grid, _ = B.grid_and_county()
    cent = grid.centroids()
    res = {}
    for county, flag in (("yunlin", False), ("changhua", True)):
        s = iv[iv.changhua == flag].reset_index(drop=True)
        parts, d, cnt = pump_split(county, len(cent))
        pos = {p: k for k, p in enumerate(d)}
        m0 = np.array([pos[pd.Period(t, "M")] for t in s.t0])
        m1 = np.array([pos[pd.Period(t, "M")] for t in s.t1])
        W3 = PL.ring_weights(cent, s.cell.to_numpy(), 0.0, B.R3)
        e = {k: PL.interval_exposure(E, W3, m0, m1, s.years.to_numpy()) for k, E in parts.items()}
        w = {k: within(v, s.sid.to_numpy(), s.campaign.to_numpy()) for k, v in e.items()}
        tot = w["continuous"] + w["entry_exit"]
        vt = float(np.var(tot))
        res[county] = {
            **cnt, "n_intervals": int(len(s)),
            "mean_E3_GWh_per_yr": {k: float(np.mean(v) / 1e6) for k, v in e.items()},
            "within_sd_GWh_per_yr": {k: float(np.std(v) / 1e6) for k, v in w.items()},
            "share_of_within_variance": {
                "continuous": float(np.var(w["continuous"]) / vt),
                "entry_exit": float(np.var(w["entry_exit"]) / vt),
                "covariance_term": float(2 * np.cov(w["continuous"], w["entry_exit"])[0, 1] / vt)},
            "corr_within_continuous_vs_entry_exit": float(np.corrcoef(w["continuous"],
                                                                      w["entry_exit"])[0, 1])}
        print(county, json.dumps(res[county], indent=1))
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
