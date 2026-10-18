"""H1a stage 1 runner: pipeline.process_well over a shard of stations (python -I, CPU slot).

Usage: s1_screen.py WORKDIR SHARD NSHARDS [--pre-only] [--sids a,b,c]
Writes WORKDIR/stage1/<sid>.pkl (per-well arrays and catalogues) and a shard log.
"""
from __future__ import annotations

import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import baro, pipeline, rawio  # noqa: E402


def main():
    work = Path(sys.argv[1])
    shard, nsh = int(sys.argv[2]), int(sys.argv[3])
    pre_only = "--pre-only" in sys.argv
    sids = rawio.all_sids()
    if "--sids" in sys.argv:
        sids = sys.argv[sys.argv.index("--sids") + 1].split(",")
    sids = [s for i, s in enumerate(sids) if i % nsh == shard]
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    P = baro.load_pressure()
    outd = work / ("stage1_pre" if pre_only else "stage1")
    outd.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    for k, sid in enumerate(sids):
        t0 = time.time()
        try:
            m = meta.loc[sid].to_dict() if sid in meta.index else {"era5_cell": None}
            for key in ("WellElevation", "WellDepth"):
                m[key] = float(m[key]) if m.get(key) is not None and np.isfinite(m.get(key)) \
                    else None
            res = pipeline.process_well(sid, m, P, pre_only=pre_only)
            with open(outd / f"{sid}.pkl", "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            be = res.get("be", {})
            aks = ",".join(f"{x.get('A_K', float('nan')):.3f}" for x in be.get("segments", []))
            print(f"{k + 1}/{len(sids)} {sid} {time.time() - t0:.1f}s be={be.get('status')} "
                  f"A_K={aks}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"FAIL {sid}\n{traceback.format_exc()}", flush=True)
    print(f"shard {shard}/{nsh} done in {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
