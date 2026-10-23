"""P5 stage 1: per-well head-signature extraction over a shard of wells (CPU slot, python -I).

Usage: s1_extract.py WORKDIR SHARD NSHARDS [--sids a,b,c]
Writes WORKDIR/wells/<sid>.parquet (well-month features) and WORKDIR/diag/<sid>.json, plus a
shard log line per well. Wells: every station with at least one flag-0/1 day in L1 v3.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p5_head_signatures import extractor as X  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402


def main():
    work = Path(sys.argv[1])
    shard, nsh = int(sys.argv[2]), int(sys.argv[3])
    p = I.l1()
    meta = I.station_meta(p).set_index("sid")
    be = I.be_table(p)
    P = I.pressure_wide(p)
    d = I.daily_flags(p)
    d = d[d.flag.isin(list(X.C.GOOD_DAY_FLAGS))]
    ev = pd.read_csv(p.path("events.csv"), dtype={"sid": str})
    sids = sorted(d.sid.unique())
    if "--sids" in sys.argv:
        sids = sys.argv[sys.argv.index("--sids") + 1].split(",")
    sids = [s for i, s in enumerate(sids) if i % nsh == shard]
    (work / "wells").mkdir(parents=True, exist_ok=True)
    (work / "diag").mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    for k, sid in enumerate(sids):
        t0 = time.time()
        try:
            m = meta.loc[sid].to_dict() if sid in meta.index else {"era5_cell": None}
            for key in ("WellElevation", "WellDepth"):
                v = m.get(key)
                m[key] = float(v) if v is not None and np.isfinite(v) else None
            dd = d[d.sid == sid]
            F, diag = X.extract_well(sid, m, P, be[be.sid == sid], dd["date"].values,
                                     ev[ev.sid == sid],
                                     dd[["date", "value", "corr_m", "baro_m"]])
            if len(F):
                F.to_parquet(work / "wells" / f"{sid}.parquet", index=False)
            diag["n_rows"] = int(len(F))
            diag["seconds"] = round(time.time() - t0, 2)
            (work / "diag" / f"{sid}.json").write_text(json.dumps(diag, default=str))
            print(f"{k + 1}/{len(sids)} {sid} rows={len(F)} {time.time() - t0:.1f}s "
                  f"check={diag.get('daily_check')}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"FAIL {sid}\n{traceback.format_exc()}", flush=True)
    print(f"shard {shard}/{nsh} done in {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
