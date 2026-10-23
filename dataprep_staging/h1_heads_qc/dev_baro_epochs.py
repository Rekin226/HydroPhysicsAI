"""H1a design-time check 2 (pre-origin data only): is the barometric response stable within a
logger epoch and different across the logger change? Run with ``python -I`` in a CPU slot.

The pre-origin record of most stations switches from hourly to 10-minute sampling (a logger
change, mostly in 2019). For every station with both epochs (>= 365 days each), the
production design is fitted on epoch A (before the first sustained 10-minute sampling),
epoch B (after), and on the two halves of each epoch. Reported: the median |z| of the
within-epoch half differences and of the across-epoch difference (z = dA / sqrt(se1^2 +
se2^2); about 0.67 if stable and the se are right), and the split-half z of the whole-window
fit from ``dev_baro_methods.py`` for comparison.

Usage: dev_baro_epochs.py WORKDIR OUT_JSON
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import baro, pipeline, rawio, screens  # noqa: E402


def logger_switch(t: np.ndarray) -> int | None:
    """Time of the first sample after which >= 90 % of the next 1000 intervals are <= 11
    minutes (sustained 10-minute logging)."""
    if len(t) < 2000:
        return None
    dt = np.diff(t)
    short = (dt <= 660).astype(float)
    c = np.convolve(short, np.ones(1000) / 1000.0, mode="valid")
    idx = np.flatnonzero(c >= 0.9)
    return int(t[idx[0] + 1]) if len(idx) else None


def fit(hh, p, a, b):
    hs = hh[(hh.index >= a) & (hh.index < b)]
    f = baro.fit_be(hs, p)
    return f.get("A_K", np.nan), f.get("se_A_K", np.nan)


def z(a, sa, b, sb):
    return (a - b) / np.sqrt(sa ** 2 + sb ** 2)


def main():
    work, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    P = baro.load_pressure()
    rows = []
    t00 = time.time()
    for sid in rawio.all_sids():
        if sid not in meta.index or not meta.loc[sid, "era5_cell"]:
            continue
        sh, _ = rawio.load_well(sid, keep_sentinels=True)
        if len(sh) < 1000:
            continue
        m = meta.loc[sid]
        t, v = screens.to_arrays(sh)
        sw = logger_switch(t[np.isfinite(v) & (np.abs(v) < 1000)])
        if sw is None:
            continue
        tau = pd.Timestamp(sw, unit="s")
        t0, t1 = pd.Timestamp(t[0], unit="s"), pd.Timestamp("2023-01-01")
        if (tau - t0).days < 365 or (t1 - tau).days < 365:
            continue
        med, mad = pipeline.robust_ref(v)
        elev = float(m.WellElevation) if np.isfinite(m.WellElevation) else None
        depth = float(m.WellDepth) if np.isfinite(m.WellDepth) and m.WellDepth > 0 else None
        r1 = screens.screen(t, v, med_ref=med, mad_ref=mad, elev=elev, depth=depth)
        hh = pipeline.hourly_clean(t, v, r1["kind"])
        p = P[m.era5_cell]
        mA, mB = t0 + (tau - t0) / 2, tau + (t1 - tau) / 2
        A, sA = fit(hh, p, t0, tau)
        B, sB = fit(hh, p, tau, t1)
        A1, sA1 = fit(hh, p, t0, mA)
        A2, sA2 = fit(hh, p, mA, tau)
        B1, sB1 = fit(hh, p, tau, mB)
        B2, sB2 = fit(hh, p, mB, t1)
        rows.append({"sid": sid, "switch": str(tau), "AK_A": A, "se_A": sA, "AK_B": B,
                     "se_B": sB, "z_across": z(A, sA, B, sB), "z_within_A": z(A1, sA1, A2, sA2),
                     "z_within_B": z(B1, sB1, B2, sB2)})
        print(f"{sid} {tau.date()} A={A:.3f}+-{sA:.3f} B={B:.3f}+-{sB:.3f} "
              f"{time.time() - t00:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    within = pd.concat([R.z_within_A, R.z_within_B]).abs().dropna()
    summ = {"n_wells": int(len(R)),
            "switch_year_counts": pd.to_datetime(R.switch).dt.year.value_counts().sort_index()
            .to_dict(),
            "median_abs_z_within_epoch_halves": float(within.median()),
            "median_abs_z_across_epochs": float(R.z_across.abs().median()),
            "share_abs_z_gt3_within": float((within > 3).mean()),
            "share_abs_z_gt3_across": float((R.z_across.abs() > 3).mean())}
    Path(out_json).write_text(json.dumps({"note": "design-time, pre-origin only", "summary": summ,
               "rows": R.to_dict("records")}, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
