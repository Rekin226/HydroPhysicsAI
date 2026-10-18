"""WP S5 repair round 2 (R6): observation tables with the repaired inputs.

Round-1 loaders (s5data) are reused read-only; this module only points them at the repaired
upstream products and adds the repair-round fields:

* GNSS: gnss_clean/preorigin/v4 (``pre``) or full/v4 (``full``); offset segments at every gap
  > 30 d of the v4 daily record (repair round 1's rule, s5r_core);
* InSAR: run P2 cells (``pre``; s5r2_insarP2.py) or insar_069A/v3 vertical_cells (``full``;
  flag 0 and source 1-3 only, i.e. chains tied to the first epoch, untied second-frame values
  and LiCSBAS-bridged values excluded; one network segment);
* leveling: campaign id (county x survey month) per survey (R2); towns from the L0 panel;
* optional ends: leveling (``lev_end``) and the other instruments (``other_end``) for the
  pseudo-origin of the fresh test T1F; ``ring_win`` sets the pre-origin ring-interval window.

Gate-failed inputs are opened for after_the_fact_analysis through the loader guard, as in
round 1 (s5data).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5r2_const as K  # noqa: E402
import s5r2_model as RM  # noqa: E402
import s5r_core as RC  # noqa: E402
from common import guard  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

# R6: every round-1 loader reads the v4 GNSS products in this process
C.GNSS_PRE = K.GNSS_PRE_V4
C.GNSS_FULL = K.GNSS_FULL_V4


def towns() -> pd.Series:
    df = pd.read_parquet(C.LEVELING_PANEL, columns=["sid", "town"])
    return df.dropna(subset=["town"]).groupby("sid")["town"].first()


def insar_cells2(variant: str) -> dict:
    if variant == "pre":
        z = np.load(K.INSAR_P2_CELLS, allow_pickle=False)
        return {k: z[k] for k in ("value", "flag", "segment", "network_segment", "dates")}
    p = guard.open_product(K.INSAR_V3, D.USE)
    z = np.load(p.path("vertical_cells.npz"), allow_pickle=False)
    flag = np.where((z["flag"] == 0) & np.isin(z["source"], [1, 2, 3]), 0, 8).astype(np.uint8)
    A, T = z["value"].shape
    return {"value": z["value"], "flag": flag, "segment": np.zeros((A, T), np.int16),
            "network_segment": np.zeros(T, np.int16), "dates": z["dates"]}


def load_all2(variant: str, *, lev_end: str | None = None, other_end: str | None = None,
              ring_win: tuple | None = None) -> tuple[dict, dict]:
    g = D.grid()
    lev, q_lev = D.leveling(g)
    if lev_end is not None:
        lev = lev[lev["date"].astype(str) <= lev_end].reset_index(drop=True)
    lev["camp"] = RM.campaign_ids(lev, towns())
    lev["seg"] = 0
    gn, q_g = D.gnss(g, variant)
    breaks = RC.gnss_breaks(RC.read_daily(variant))
    gn["seg"] = RC.gnss_segments(gn, breaks)
    end_ins = C.ORIGIN_END if variant == "pre" else None
    if other_end is not None:
        end_ins = other_end
    ins, q_i = D.insar_groups(g, insar_cells2(variant), end=end_ins)
    if other_end is not None and len(ins["groups"]) == 0:
        q_i["note"] = "no InSAR before the pseudo-origin"
    old = C.RING_PRE_WIN
    try:
        if ring_win is not None:
            C.RING_PRE_WIN = ring_win
        rg, q_r = D.rings(g, variant)
    finally:
        C.RING_PRE_WIN = old
    pl, q_p = D.piles(g, variant)
    cmp_ = pd.concat([rg, pl], ignore_index=True)
    if other_end is not None:
        te = D.dec_year([other_end])[0] + 1.0 / 365.25
        gn = gn[gn.t <= te].reset_index(drop=True)
        cmp_ = cmp_[cmp_.t <= te].reset_index(drop=True)
    data = {"grid": g, "lev": lev, "gnss": gn, "insar": ins, "cmp": cmp_}
    for k in ("lev", "gnss", "cmp"):
        data[k] = data[k].copy()
        data[k]["rid"] = np.arange(len(data[k]))
    qc = {"variant": variant, "lev_end": lev_end, "other_end": other_end,
          "ring_win": list(ring_win) if ring_win else list(C.RING_PRE_WIN),
          "leveling": q_lev, "leveling_rows": int(len(lev)),
          "campaigns": int(lev["camp"].nunique()), "gnss": q_g,
          "gnss_rows": int(len(gn)), "gnss_extra_segments": int(
              (gn.groupby(["family", "station"]).seg.max()).sum()),
          "insar": q_i, "rings": q_r, "piles": q_p, "cmp_rows": int(len(cmp_)),
          "inputs": {"gnss": rel_to_repo(C.GNSS_PRE if variant == "pre" else C.GNSS_FULL),
                     "insar": rel_to_repo(K.INSAR_P2_CELLS if variant == "pre"
                                          else K.INSAR_V3)}}
    return data, qc


def masks_from(data: dict, flagged: dict, extra: dict | None = None) -> dict:
    m = {}
    for fam in ("lev", "gnss", "cmp"):
        keep = ~data[fam]["rid"].isin(flagged.get(fam, [])).to_numpy()
        if extra and fam in extra:
            keep &= np.asarray(extra[fam], bool)
        m[fam] = keep
    return m
