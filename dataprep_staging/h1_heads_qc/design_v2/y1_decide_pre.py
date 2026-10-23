# Design-time exploration (round 2): v2 decisions on PRE-ORIGIN data only (post parts removed).
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
sids = sys.argv[3].split(",") if len(sys.argv) > 3 else None
meta = pd.read_parquet(W / "station_meta_clean.parquet").set_index("sid")
from h1_heads_qc import baro

P = baro.load_pressure()
rows = []; t0 = time.time()
for p in sorted((W / "stage1").glob("*.pkl")):
    if sids and p.stem not in sids:
        continue
    res = pickle.loads(p.read_bytes())
    res = {k: v for k, v in res.items() if k not in ("post", "post_cat", "o1_post", "be_oos")}
    m = meta.loc[res["sid"]]
    elev, depth = m.get("WellElevation"), m.get("WellDepth")
    bottom = (elev - depth) if (pd.notna(elev) and pd.notna(depth) and depth > 0) else None
    top = elev + C.TOP_MARGIN_M if pd.notna(elev) else None
    t1 = time.time()
    cid = m.get("era5_cell"); R = DV.decide_well(res, bottom=bottom, top=top, pressure=P[cid] if cid in P.columns else None)
    rows += R
    print(res["sid"], len(R), f"{time.time()-t1:.1f}s", flush=True)
R = pd.DataFrame(rows)
for c in R.columns:
    if R[c].dtype == object:
        R[c] = R[c].map(lambda x: x if (x is None or isinstance(x, (str, float, int, bool, np.floating, np.integer))) else str(x))
R.to_parquet(out, index=False)
print(len(R), f"{time.time()-t0:.0f}s")
