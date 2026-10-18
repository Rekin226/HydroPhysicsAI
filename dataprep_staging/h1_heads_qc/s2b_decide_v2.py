"""H1a v2 stage 2 (round 2): v2 decisions (``decide_v2``) with the backing evidence of the
accepted steps (python -I, CPU slot).

Usage: s2b_decide_v2.py WORKDIR [STAGE1_SUBDIR]
Reads WORKDIR/<stage1>/*.pkl, WORKDIR/station_meta_clean.parquet and the ERA5 cell pressure
(``baro.load_pressure``, for the typhoon-time rule); writes WORKDIR/decisions_v2_<stage1>.parquet.

Backing of an accepted step (``backing.py``, as v1 stage 2 computed it for its steps): the
co-located test (nested wells of the same site recording through the jump without a
simultaneous jump), station metadata dates within 7 days, and the leave-out neighbour
regression (60 days after vs 60 days before; pre-origin fit; a pre-origin step is tested on
pre-origin days only). The neighbour regression excludes the well's v2 episodes from its fit.
The daily matrices that supply neighbours are the stage-1 clean days (v1's evidence base).
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import backing as B  # noqa: E402
from h1_heads_qc import baro  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402
from h1_heads_qc import decide_v2 as DV  # noqa: E402
from h1_heads_qc import screens as S  # noqa: E402


def bounds(m: pd.Series):
    elev, depth = m.get("WellElevation"), m.get("WellDepth")
    elev = float(elev) if elev is not None and np.isfinite(elev) else None
    depth = float(depth) if depth is not None and np.isfinite(depth) and depth > 0 else None
    bottom = (elev - depth) if (elev is not None and depth is not None) else None
    top = elev + C.TOP_MARGIN_M if elev is not None else None
    return bottom, top


def main():
    work = Path(sys.argv[1])
    sub = sys.argv[2] if len(sys.argv) > 2 else "stage1"
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    t0 = time.time()
    light, daily, jumps_by_sid, files = {}, {}, {}, {}
    for p in sorted((work / sub).glob("*.pkl")):
        res = pickle.loads(Path(p).read_bytes())
        sid = res["sid"]
        files[sid] = p
        daily[sid] = B.well_daily(res)
        J = []
        for part in ("pre_cat", "post_cat"):
            if part in res:
                J += res[part]["jumps"]
        jumps_by_sid[sid] = J
        lt = {}
        for part in ("pre", "post"):
            if part in res:
                a = res[part]
                tt = a["t"][(a["kind"] & S.K_SENTINEL) == 0]
                lt[part] = {"t": tt, "kind": np.zeros(len(tt), np.uint16)}
        light[sid] = lt
    print(f"loaded {len(light)} wells in {time.time() - t0:.0f}s", flush=True)
    pre_c, post_c = B.build_matrices(daily, "clean")
    pre_d, post_d = B.build_matrices(daily, "disp")
    xy = {s: (r.x_twd97, r.y_twd97) for s, r in meta.iterrows()
          if np.isfinite(r.x_twd97) and r.xy_flag in (0, 1)}
    sites = meta.groupby("site_id").groups
    P = baro.load_pressure()

    rows = []
    for sid, p in files.items():
        res = pickle.loads(Path(p).read_bytes())
        m = meta.loc[sid] if sid in meta.index else pd.Series(dtype=object)
        bottom, top = bounds(m)
        cid = m.get("era5_cell") if len(m) else None
        R = DV.decide_well(res, bottom=bottom, top=top,
                           pressure=P[cid] if cid in P.columns else None)
        site = sites.get(m.get("site_id"), []) if len(m) else []
        mdates = B.metadata_dates(m) if len(m) else []
        spans = [(pd.Timestamp(r["t1"], unit="s"), pd.Timestamp(r["t2"], unit="s"))
                 for r in R if r["kind"] in DV.EPISODE_KINDS]
        for r in R:
            if r["kind"] != "step":
                continue
            co = B.colocated(sid, [r["t1"]], r["j1"], list(site), light, jumps_by_sid)
            md = [d for d in mdates
                  if abs((pd.Timestamp(r["t1"], unit="s") - d).days) <= C.META_DATE_DAYS]
            disp = B.days_after(r["t1"], C.NEIGH_STEP_DAYS)
            ref = B.days_before(r["t1"], C.NEIGH_STEP_DAYS)
            nt = B.neighbour_test(sid, disp, ref, r["j1"], pre_c, post_c, pre_d, post_d, xy,
                                  spans, use_post=r["period"] == "post")
            mds = ",".join(str(d.date()) for d in md)
            cov = ",".join(co["covering"])
            r.update({"meta_dates": mds, "coloc_backed": bool(co["backed"]),
                      "coloc_covering": cov, "coloc_jumped": ",".join(co["jumped"]),
                      "neighbour_verdict": nt.get("verdict"), "neighbour_why": nt.get("why"),
                      "neighbour_delta_m": nt.get("delta_m"),
                      "neighbour_sd_m": nt.get("sd_cal_m"), "neighbour_z": nt.get("z"),
                      "neighbours": ",".join(nt.get("neighbours", []) or [])})
            r["flag"], r["backing"] = DV.flag_rule_v2(bool(r.get("meta_impossible")), mds,
                                                      bool(co["backed"]), cov,
                                                      nt.get("verdict"))
        rows += R
        print(f"{sid} {len(R)} rows {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    for c in D.columns:
        if D[c].dtype == object:
            D[c] = D[c].map(lambda x: x if (x is None or isinstance(
                x, (str, float, int, bool, np.floating, np.integer))) else str(x))
    D = D.sort_values(["sid", "period", "t1", "kind"]).reset_index(drop=True)
    D.to_parquet(work / f"decisions_v2_{sub}.parquet", index=False)
    g = D[D.kind != "jump"]
    print(g.groupby(["period", "kind", "v2_status"]).size().to_string())
    print(g.groupby(["period", "kind", "flag"], dropna=False).size().to_string())
    print(D[D.kind == "jump"].groupby(["period", "cls1", "v2_status"]).size().to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
