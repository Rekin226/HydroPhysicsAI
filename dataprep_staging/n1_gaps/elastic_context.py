"""Context for the elastic split: why no avoided subsidence is inelastic in the page's run.

Companion to ``elastic_split.py`` (same column, same ensemble-mean heads). It reports, for
the baseline forward run: the elastic / inelastic parts of the baseline's own forward
subsidence, how many cells' heads fall below their preconsolidation head after the
policies start (Jan 2026), and the head margin above preconsolidation that the forecast
keeps (how much extra drawdown a dry year would need before new inelastic compaction
starts, which is when a pumping cut would begin to avoid permanent compaction). It also
records how the forecast forcing is built (``forward_options``), since the forecast
repeats mean conditions and never revisits the record's dry-year minima.

Model output only; no observation read; nothing scored. CPU, seconds.
Usage: python -I dataprep_staging/n1_gaps/elastic_context.py
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-13/elastic_context.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from n1_gaps import elastic_split as ES  # noqa: E402
from n1_gaps import gaps_lib as G  # noqa: E402

START = "2026-01"


def main() -> int:
    os.chdir(G.REPO)
    from hydrophysics.twin.zone_county import column_zone_map

    sets = json.loads(ES.COLUMN.read_text())["zonal"]
    meta = json.loads(ES.THETA.read_text())["meta"]
    gg = G.grid_groups()
    zone = np.asarray(column_zone_map(gg["grid"], meta), int)
    ch = gg["changhua"]
    z = np.load(ES.FORWARD, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    dates = pd.DatetimeIndex(z["dates"])
    o = int(z["origin"])
    s0 = int(np.nonzero(dates == pd.Timestamp(START))[0][0])
    Hbar = z["heads_mean"].astype("float64").mean(axis=1)
    par = {k: np.array([float(sets[zz][k]) for zz in zone]) for k in sets[0]}
    res = {"what": "baseline forward run of the page's model: elastic/inelastic parts and "
                   "the head margin above preconsolidation after Jan 2026",
           "forward_options": json.loads(str(z["forward_options"])), "scenarios": {}}
    for s in names:
        h = Hbar[names.index(s)]
        el, inel, _ = ES.vep_terms(h, np.exp(par["log_ske"]), np.exp(par["log_skv"]),
                                   np.exp(par["log_tau"]), par["h_pc0"])
        h_pc_before = np.minimum(h[:, 0] + par["h_pc0"], h[:, 1:s0].min(axis=1))
        margin = h[:, s0:].min(axis=1) - h_pc_before          # > 0: never below h_pc again
        below = margin < 0
        yH = len(dates) - 1
        fwd_el = (el[:, yH] - el[:, o]) * 100.0
        fwd_in = (inel[:, yH] - inel[:, o]) * 100.0
        blk = {"forward_dec2022_dec2032_cm": {
                   "elastic_fan": float(fwd_el.mean()), "inelastic_fan": float(fwd_in.mean()),
                   "total_fan": float((fwd_el + fwd_in).mean()),
                   **{f"zone{k}_inelastic": float(fwd_in[zone == k].mean())
                      for k in range(len(sets))}},
               "cells_below_preconsolidation_after_2026_01": int(below.sum()),
               "head_margin_above_preconsolidation_m": {
                   "p5": float(np.percentile(margin, 5)), "p50": float(np.median(margin)),
                   "min": float(margin.min()),
                   **{f"zone{k}_p5_p50": [float(np.percentile(margin[zone == k], 5)),
                                          float(np.median(margin[zone == k]))]
                      for k in range(len(sets))},
                   "yunlin_p5_p50": [float(np.percentile(margin[~ch], 5)),
                                     float(np.median(margin[~ch]))],
                   "changhua_p5_p50": [float(np.percentile(margin[ch], 5)),
                                       float(np.median(margin[ch]))]},
               "month_of_record_minimum_head_p50": str(
                   dates[int(np.median(np.argmin(h[:, 1:s0], axis=1)) + 1)].date())[:7]}
        res["scenarios"][s] = blk
        G.log(f"{s}: forward elastic {blk['forward_dec2022_dec2032_cm']['elastic_fan']:.3f} "
              f"inelastic {blk['forward_dec2022_dec2032_cm']['inelastic_fan']:.3f} cm; cells "
              f"below h_pc after {START}: {int(below.sum())}; margin p5/p50 "
              f"{np.percentile(margin, 5):.2f}/{np.median(margin):.2f} m")
    res["column_tau_days"] = [math.exp(p["log_tau"]) for p in sets]
    res["reading"] = (
        "In the forward run no cell's layer-mean head falls below its preconsolidation head "
        "after Jan 2026 when the count above is 0, so neither the baseline nor a policy adds "
        "new inelastic loading; the baseline's inelastic forward subsidence is creep toward "
        "loads set before 2026, identical with or without a policy. A pumping cut then "
        "changes only the elastic term. The forecast repeats mean forcing; a dry year that "
        "drew heads down by more than the margin above would add inelastic loading that a "
        "cut could avoid. That case is not in the page's runs.")
    res["provenance"] = {"code": "dataprep_staging/n1_gaps/elastic_context.py",
                         "git_head": G.git_head(),
                         "inputs": [G.source(ES.FORWARD), G.source(ES.COLUMN),
                                    G.source(ES.THETA)],
                         "status": "post-hoc model-only analysis; nothing scored"}
    G.write_new(ES.OUT / "elastic_context.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
