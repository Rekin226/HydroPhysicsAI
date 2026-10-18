# Design-time exploration (pre-origin events only): level check on the v1 decisions.
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from h1_heads_qc import levelcheck as LC

W = Path(sys.argv[1]); out = Path(sys.argv[2])
D = pd.read_parquet(W / "decisions_stage1.parquet")
D = D[D.period == "pre"].copy()
rows = []
t0 = time.time()
for sid, g in D.groupby("sid"):
    res = pickle.loads((W / "stage1" / f"{sid}.pkl").read_bytes())
    part, jumps = res["pre"], res["pre_cat"]["jumps"]
    for ix, d in g.iterrows():
        if d["kind"] == "step":
            c = LC.check_jump(part, jumps, int(d.t1), float(d.j1))
            rows.append({"ix": ix, "edge": 1, **c})
        else:
            c1 = LC.check_jump(part, jumps, int(d.t1), float(d.j1), other=int(d.t2))
            c2 = LC.check_jump(part, jumps, int(d.t2), float(d.j2), other=int(d.t1))
            rows.append({"ix": ix, "edge": 1, **c1}); rows.append({"ix": ix, "edge": 2, **c2})
R = pd.DataFrame(rows)
R.to_parquet(out, index=False)
print(f"{len(R)} checks in {time.time()-t0:.0f}s")
