# Design-time exploration: v2 decisions on PRE-ORIGIN events only (post parts removed).
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from h1_heads_qc import config as C
from h1_heads_qc import decide_v2 as DV

W = Path(sys.argv[1]); out = Path(sys.argv[2])
D = pd.read_parquet(W / "decisions_stage1.parquet")
D = D[D.period == "pre"]
meta = pd.read_parquet(W / "station_meta_clean.parquet").set_index("sid")
rows = []; t0 = time.time()
for p in sorted((W / "stage1").glob("*.pkl")):
    res = pickle.loads(p.read_bytes())
    sid = res["sid"]
    if sid not in set(D.sid):
        continue
    res = {k: v for k, v in res.items() if k not in ("post", "post_cat", "o1_post", "be_oos")}
    m = meta.loc[sid]
    elev, depth = m.get("WellElevation"), m.get("WellDepth")
    bottom = (elev - depth) if (pd.notna(elev) and pd.notna(depth) and depth > 0) else None
    top = elev + C.TOP_MARGIN_M if pd.notna(elev) else None
    rows += DV.decide_well(res, D[D.sid == sid], bottom=bottom, top=top)
R = pd.DataFrame(rows)
for c in R.columns:
    if R[c].dtype == object:
        R[c] = R[c].map(lambda x: x if (x is None or isinstance(x, (str, float, int, bool, np.floating, np.integer))) else str(x))
R.to_parquet(out, index=False)
print(len(R), f"{time.time()-t0:.0f}s")
