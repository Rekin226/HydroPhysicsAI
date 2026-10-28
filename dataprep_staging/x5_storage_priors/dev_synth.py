"""Estimator development on synthetic DEV seeds after the v1 synthetic failure (no real data).

Runs the synthetic pipeline of synthetic_check.py for one variant (window x barometric
design) on a dev seed and writes results/dataprep/x5_storage_priors/dev/<tag>.json. Dev seeds
are never the declared seed of a synthetic check; every dev run is listed in the v2
declaration and the report.

Usage: python -I dev_synth.py --window 2y|3y|6y --be hourly_grouped|daily_lag2 --seed N
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5sp_const as K  # noqa: E402

WINDOWS = {
    "2y": (("2017-01-01 00:00", "2018-12-31 23:00"), ()),
    "3y": (("2016-01-01 00:00", "2018-12-31 23:00"), ()),
    "6y": (("2012-01-01 00:00", "2018-12-31 23:00"), (("2015-01-01", "2015-12-31 23:00"),)),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", choices=sorted(WINDOWS), required=True)
    ap.add_argument("--be", default="hourly_grouped")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if a.seed == K.SYN_SEED:
        raise SystemExit("the declared seed is not a dev seed")
    K.WINDOW, K.EXCLUDE = WINDOWS[a.window]
    K.BE_DESIGN = a.be
    K.SYN_SEED = a.seed
    import synthetic_check as S
    import x5sp_lib as L
    P = L.load_pressure()
    p = P[K.SYN_PRESSURE_CELL]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(S.one, [(d, p) for d in S.draws()]))
    df = pd.DataFrame(rows)
    res = S.evaluate(df)
    inl = df[~df["coastal"] & (df["M2_snr"] >= K.SNR_MIN)]
    w = 1.0 / np.maximum(inl["M2_se_phase_deg"], 0.1) ** 2
    res["eta_weighted_mean_deg"] = float(np.sum(w * inl["M2_eta_deg"]) / np.sum(w))
    hi = inl[inl["M2_snr"] >= 20]
    res["median_abs_eta_snr20"] = float(np.median(np.abs(hi["M2_eta_deg"]))) if len(hi) else None
    tag = f"{a.window}_{a.be}_seed{a.seed}"
    out = K.OUT / "dev" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(tag, json.dumps({k: (v.get("pass"), {kk: vv for kk, vv in v.items()
                                               if kk != "pass"}) if isinstance(v, dict) else v
                           for k, v in res.items()}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
