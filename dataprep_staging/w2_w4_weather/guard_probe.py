"""Loader-guard probe of a built et0/v<N> product (WP W2): which uses can open which outputs.

Expected (plan U3-U5): forcing_recorded_arm, statistical_layer_history and
after_the_fact_analysis read every output; forcing_frozen (a pre-origin use) reads only
et0_daily_2012_2022_preorigin.npz; scoring_target, calibration_target, baseline, datum_fit,
selection, model_design and prior are refused. Also checks that the pre-origin file holds the
calibration cache values unchanged.

Usage: python -I isorun.py guard_probe.py --dir results/dataprep/et0/v1 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from common import guard
from wcommon import ET_CACHE, REPO, write_json_new

USES = ["forcing_recorded_arm", "statistical_layer_history", "after_the_fact_analysis",
        "forcing_frozen", "scoring_target", "calibration_target", "baseline", "datum_fit",
        "selection", "model_design", "prior", "initial_conditions"]
FILES = ["et0_daily_2012_2026.npz", "et0_daily_2012_2022_preorigin.npz"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.dir) if Path(a.dir).is_absolute() else REPO / a.dir
    res = {}
    for use in USES:
        r = {}
        try:
            p = guard.open_product(d, use)
            r["open"] = "ok"
            for f in FILES:
                try:
                    p.path(f)
                    r[f] = "readable"
                except Exception as e:  # noqa: BLE001 - record the refusal
                    r[f] = f"refused: {type(e).__name__}"
        except Exception as e:  # noqa: BLE001
            r["open"] = f"refused: {type(e).__name__}: {str(e)[:160]}"
        res[use] = r
    # pre-origin output equals the cache
    c = np.load(ET_CACHE, allow_pickle=False)
    z = np.load(d / "et0_daily_2012_2022_preorigin.npz", allow_pickle=False)
    same_ids = bool(np.array_equal(c["well_ids"], z["well_ids"]))
    same_dates = bool(np.array_equal(c["dates"], z["dates"]))
    a1, a2 = c["et0"], z["et0"]
    res["preorigin_equals_cache"] = {
        "well_ids": same_ids, "dates": same_dates, "dtype": [str(a1.dtype), str(a2.dtype)],
        "values_bitwise_equal_nan_aware": bool(np.array_equal(a1, a2, equal_nan=True))}
    full = np.load(d / "et0_daily_2012_2026.npz", allow_pickle=False)
    n_pre = a1.shape[1]
    res["full_file_2012_2022_equals_cache"] = {
        "et0_raw_splice": bool(np.array_equal(full["et0_raw_splice"][:, :n_pre], a1,
                                              equal_nan=True)),
        "et0": bool(np.array_equal(full["et0"][:, :n_pre], a1, equal_nan=True))}
    write_json_new(a.out, res)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
