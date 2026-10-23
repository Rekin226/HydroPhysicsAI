# Design-time (r=0..9, already seen): segment-check statistics of the missed injected steps.
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(sys.argv[4]).resolve()))
from h1_heads_qc import baro, pipeline, rawio, screens
from h1_heads_qc import config as C
from h1_heads_qc import decide_v2 as DV
from h1_heads_qc import levelcheck as LC
from h1_heads_qc.s5_inject_v2 import bounds_of

W = Path(sys.argv[1]); J = json.loads(Path(sys.argv[2]).read_text()); out = Path(sys.argv[3])
meta = pd.read_parquet(W / "station_meta_clean.parquet").set_index("sid")
P = baro.load_pressure()
R = pd.DataFrame(J["rows"]); R = R[(R.status == "injected") & (R.kind == "step") & (R.recovered == False)]
rows = []
for _, e in R.iterrows():
    sid = e.sid
    base = pickle.loads((W / "stage1" / f"{sid}.pkl").read_bytes())
    segs = [{"start": x["start"], "end": x["end"], "a": x.get("a")} for x in base["be"].get("segments", [])]
    sh, _ = rawio.load_well(sid, keep_sentinels=True, pre_only=True)
    t, v = screens.to_arrays(sh)
    good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
    tau0 = int(pd.Timestamp(e.tau0).value // 10**9)
    v2 = v.copy(); v2[good & (t >= tau0)] += e["size"]
    mrow = meta.loc[sid].to_dict()
    for key in ("WellElevation", "WellDepth"):
        mrow[key] = float(mrow[key]) if mrow.get(key) is not None and np.isfinite(mrow.get(key)) else None
    o = pipeline.process_well(sid, mrow, P, pre_only=True, hist_override=pd.Series(v2, index=sh.index),
                              be_override={"segments": segs, "p_ref_hpa": base.get("p_ref_hpa")}, o1=False)
    bottom, top = bounds_of(mrow)
    cid = mrow.get("era5_cell")
    dec = DV.decide_well({k: x for k, x in o.items() if k not in ("post", "post_cat")}, bottom=bottom, top=top,
                         pressure=P[cid] if cid in P.columns else None)
    hit = [x for x in dec if x["kind"] in ("excursion", "jump_edge", "step") and abs(x["t1"] - tau0) <= 86400]
    A = DV.segment_arrays(o, "pre")
    lo, hi = A["t_min"], A["t_max"]
    full = LC.segment_shift(A["t"], A["v"], A["valid"], lo, tau0, hi, e["size"])
    for x in hit[:1]:
        rows.append({"real": e.real, "sid": sid, "size": round(e["size"], 2), "tau0": e.tau0,
                     "kind": x["kind"], "seg_ratio_trend": x.get("seg_ratio"), "seg_lo": str(pd.Timestamp(x["seg_lo"], unit="s").date()),
                     "seg_hi": str(pd.Timestamp(x["seg_hi"], unit="s").date()),
                     "seg_ratio_wholemedian_norebounds": full["seg_ratio"]})
D = pd.DataFrame(rows); D.to_csv(out, index=False); print(D.to_string())
