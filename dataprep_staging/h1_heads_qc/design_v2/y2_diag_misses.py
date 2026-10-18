# Design-time diagnosis (family O r=0..9, already seen): why v2 misses injected events.
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

W = Path(sys.argv[1]); J = json.loads(Path(sys.argv[2]).read_text()); out = Path(sys.argv[3])
meta = pd.read_parquet(W / "station_meta_clean.parquet").set_index("sid")
P = baro.load_pressure()
R = pd.DataFrame(J["rows"]); R = R[(R.status == "injected") & (R.recovered == False)]
rows = []
for _, e in R.iterrows():
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
    cid = mrow.get("era5_cell")
    pa = DV.pressure_arrays(P[cid] if cid in P.columns else None, DV.SPLIT_S + 86400)
    cand = DV.candidate_jumps(A, pa)
    alljumps = A["jumps"]
    def near(tt):
        js = [x for x in alljumps if abs(x["t"] - tt) <= 86400]
        cs = [x for x in cand if abs(x["t"] - tt) <= 86400]
        return js, cs
    rec = {"real": e.real, "sid": sid, "kind": e.kind, "size": round(e["size"], 2), "dur_d": e.get("effective_days")}
    for lab, tt in (("j0", tau0), ("j1", tau1)):
        if tt is None: continue
        js, cs = near(tt)
        rec[f"{lab}_n_stage1_jumps"] = len(js)
        rec[f"{lab}_cls"] = ";".join(f"{x['cls']}:{x['j']:.2f}" for x in js)
        rec[f"{lab}_cand"] = ";".join(f"{x['lc_status']}:{x['lc_ratio']:.2f}:{'T' if x['typhoon'] else ''}" for x in cs)
        # raw diff at the injected sample
        i = np.searchsorted(t, tt)
        rec[f"{lab}_raw_dv"] = float(v2[i] - v2[i-1]) if 0 < i < len(v2) else None
        rec[f"{lab}_dt_min"] = float((t[i] - t[i-1]) / 60) if 0 < i < len(t) else None
    rows.append(rec)
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).to_string())
