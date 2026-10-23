# Design-time (r=0..9, already seen): misses whose injected edges are persistent: what v2 did.
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from h1_heads_qc import baro, pipeline, rawio, screens
from h1_heads_qc import config as C
from h1_heads_qc import decide_v2 as DV
from h1_heads_qc.s5_inject_v2 import bounds_of

W = Path(sys.argv[1]); J = json.loads(Path(sys.argv[2]).read_text())
pick = [tuple(x.split(":")) for x in sys.argv[3].split(",")]
meta = pd.read_parquet(W / "station_meta_clean.parquet").set_index("sid")
P = baro.load_pressure()
R = pd.DataFrame(J["rows"]); R = R[(R.status == "injected")]
for real, sid in pick:
    e = R[(R.real == int(real)) & (R.sid == sid)].iloc[0]
    base = pickle.loads((W / "stage1" / f"{sid}.pkl").read_bytes())
    segs = [{"start": x["start"], "end": x["end"], "a": x.get("a")} for x in base["be"].get("segments", [])]
    sh, _ = rawio.load_well(sid, keep_sentinels=True, pre_only=True)
    t, v = screens.to_arrays(sh)
    good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
    tau0 = int(pd.Timestamp(e.tau0).value // 10**9)
    tau1 = int(pd.Timestamp(e.tau1).value // 10**9) if isinstance(e.tau1, str) else None
    m = good & (t >= tau0)
    if tau1 is not None: m &= t < tau1
    v2 = v.copy(); v2[m] += e["size"]
    mrow = meta.loc[sid].to_dict()
    for key in ("WellElevation", "WellDepth"):
        mrow[key] = float(mrow[key]) if mrow.get(key) is not None and np.isfinite(mrow.get(key)) else None
    o = pipeline.process_well(sid, mrow, P, pre_only=True, hist_override=pd.Series(v2, index=sh.index),
                              be_override={"segments": segs, "p_ref_hpa": base.get("p_ref_hpa")}, o1=False)
    bottom, top = bounds_of(mrow)
    cid = mrow.get("era5_cell")
    rows = DV.decide_well({k: x for k, x in o.items() if k not in ("post", "post_cat")}, bottom=bottom, top=top, pressure=P[cid] if cid in P.columns else None)
    lo, hi = tau0 - 400 * 86400, (tau1 or tau0) + 400 * 86400
    print(f"== r{real} {sid} {e.kind} size {e['size']:.2f} tau0 {e.tau0} tau1 {e.tau1}")
    for x in rows:
        if x["kind"] == "jump" and x["v2_status"] != "persistent":
            continue
        if lo <= x["t1"] <= hi or lo <= x["t2"] <= hi:
            print("  ", x["kind"], x.get("v2_status"), pd.Timestamp(x["t1"], unit="s"), pd.Timestamp(x["t2"], unit="s"),
                  round(x["j1"], 2), x.get("j2"), x.get("seg_ratio"), x.get("flag"))
