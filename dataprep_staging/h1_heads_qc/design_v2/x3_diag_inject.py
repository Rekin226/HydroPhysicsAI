# Design-time diagnosis on family O r=0..9 (already seen): why v2 rejects detected injections.
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

W = Path(sys.argv[1]); J = json.loads(Path(sys.argv[2]).read_text()); out = Path(sys.argv[3])
meta = pd.read_parquet(W / "station_meta_clean.parquet").set_index("sid")
P = baro.load_pressure()
R = pd.DataFrame(J["rows"]); R = R[R.status == "injected"]
sel = R[((R.kind == "paired") & (R.stage1_detected == True) & (R.recovered == False)) |
        ((R.kind == "step") & (R.recovered == False)) |
        ((R.kind == "paired") & (R.recovered == True))].copy()
res_rows = []
for _, e in sel.iterrows():
    sid = e.sid
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
    A = DV.segment_arrays(o, "pre")
    if e.kind == "paired":
        for E in o["pre_cat"]["episodes"]:
            if abs(E["t1"] - tau0) <= 86400 and abs(E["t2"] - tau1) <= 86400:
                c = DV.classify_episode(A, E)
                res_rows.append({"real": e.real, "sid": sid, "kind": "paired", "size": e["size"],
                                 "dur": (tau1 - tau0) / 86400, "recovered": e.recovered, **c,
                                 "dropout": E["dropout"]})
    else:
        bottom, top = bounds_of(mrow)
        st = DV.classify_steps(A, list(o["pre_cat"]["steps"]), bottom=bottom, top=top)
        hit = [s for s in st if abs(s["t"] - tau0) <= 86400]
        for s in hit:
            res_rows.append({"real": e.real, "sid": sid, "kind": "step", "size": e["size"],
                             "recovered": e.recovered, "status": s["status"], "lc_status": s["lc_status"],
                             "lc_ratio": s["lc_ratio"], "seg_status": s["seg_status"], "seg_ratio": s["seg_ratio"],
                             "seg_lo": s["seg_lo"], "seg_hi": s["seg_hi"], "t": s["t"]})
        if not hit:
            res_rows.append({"real": e.real, "sid": sid, "kind": "step", "size": e["size"],
                             "recovered": e.recovered, "status": "no stage-1 step at jump"})
pd.DataFrame(res_rows).to_parquet(out, index=False)
print(len(res_rows))
