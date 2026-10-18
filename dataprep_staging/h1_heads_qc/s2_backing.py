"""H1a stage 2 runner: co-located, metadata and neighbour evidence for every episode and
step; writes the decision table (python -I, CPU slot).

Usage: s2_backing.py WORKDIR [STAGE1_SUBDIR]
Writes WORKDIR/decisions.parquet and WORKDIR/daily_clean_<part>.parquet (helper matrices).
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
from h1_heads_qc import config as C  # noqa: E402
from h1_heads_qc import screens as S  # noqa: E402


def main():
    work = Path(sys.argv[1])
    sub = sys.argv[2] if len(sys.argv) > 2 else "stage1"
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    t0 = time.time()
    light, daily, jumps_by_sid = {}, {}, {}
    for p in sorted((work / sub).glob("*.pkl")):
        res = pickle.loads(Path(p).read_bytes())
        sid = res["sid"]
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
        light[sid] = lt | {k: res[k] for k in ("pre_cat", "post_cat") if k in res}
    print(f"loaded {len(light)} wells in {time.time() - t0:.0f}s", flush=True)
    pre_c, post_c = B.build_matrices(daily, "clean")
    pre_d, post_d = B.build_matrices(daily, "disp")
    pre_c.to_parquet(work / f"daily_clean_pre_{sub}.parquet")
    post_c.to_parquet(work / f"daily_clean_post_{sub}.parquet")
    xy = {s: (r.x_twd97, r.y_twd97) for s, r in meta.iterrows()
          if np.isfinite(r.x_twd97) and r.xy_flag in (0, 1)}
    sites = meta.groupby("site_id").groups

    rows = []
    for sid, res in light.items():
        m = meta.loc[sid] if sid in meta.index else pd.Series(dtype=object)
        site = sites.get(m.get("site_id"), []) if len(m) else []
        mdates = B.metadata_dates(m) if len(m) else []
        spans = []
        for part in ("pre_cat", "post_cat"):
            if part not in res:
                continue
            for E in res[part]["episodes"]:
                spans.append((pd.Timestamp(E["t1"], unit="s"), pd.Timestamp(E["t2"], unit="s")))
        for part in ("pre_cat", "post_cat"):
            if part not in res:
                continue
            cat = res[part]
            for E in cat["episodes"]:
                row = {"sid": sid, "period": E["period"], "kind": "dropout" if E["dropout"]
                       else "offset_episode", "t1": E["t1"], "t2": E["t2"], "j1": E["j1"],
                       "j2": E["j2"], "seg_median": E["seg_median"], "pre_level": E["pre"],
                       "duration_d": (E["t2"] - E["t1"]) / 86400.0, "t_lo": E["t_lo"],
                       "t_hi": E["t_hi"], "spans_origin": bool(E.get("spans_origin", False)),
                       "frac_flat": E.get("frac_flat")}
                if E["dropout"]:
                    row.update(flag=9, backing="none (dropout: invalid reading)",
                               neighbour_verdict="not_tested")
                    rows.append(row)
                    continue
                co = B.colocated(sid, [E["t1"], E["t2"]], E["j1"], list(site),
                                 light, jumps_by_sid)
                md = [d for d in mdates
                      for t in (E["t1"], E["t2"])
                      if abs((pd.Timestamp(t, unit="s") - d).days) <= C.META_DATE_DAYS]
                elev = m.get("WellElevation")
                depth = m.get("WellDepth")
                imp = False
                if elev is not None and np.isfinite(elev):
                    top = elev + C.TOP_MARGIN_M
                    bot = (elev - depth - C.BOTTOM_MARGIN_M) if (depth is not None and
                                                                  np.isfinite(depth)) else -1e9
                    imp = bool(E["seg_median"] > top or E["seg_median"] < bot)
                disp = B.interior_days(E["t1"], E["t2"])
                ref = B.days_before(E["t1"], C.NEIGH_REF_DAYS).union(
                    B.days_after(E["t2"], C.NEIGH_REF_DAYS))
                if len(disp) == 0:
                    nt = {"verdict": "untestable", "why": "shorter than one whole day"}
                else:
                    nt = B.neighbour_test(sid, disp, ref, E["j1"], pre_c, post_c, pre_d, post_d,
                                          xy, [s for s in spans
                                               if s[0] != pd.Timestamp(E["t1"], unit="s")],
                                          use_post=E["period"] == "post")
                row.update(_decide(co, md, imp, nt))
                rows.append(row)
            for St in cat["steps"]:
                jd = St["j"] if St["side"] == "after" else -St["j"]
                row = {"sid": sid, "period": St["period"], "kind": "step", "t1": St["t"],
                       "t2": St["t"], "j1": St["j"], "j2": np.nan, "side": St["side"],
                       "side_rule": St["side_rule"], "t_lo": St["t_lo"], "t_hi": St["t_hi"],
                       "median_before": St["median_before"], "median_after": St["median_after"],
                       "to_end": bool(St.get("to_end", False)), "spans_origin": False}
                co = B.colocated(sid, [St["t"]], St["j"], list(site), light, jumps_by_sid)
                md = [d for d in mdates
                      if abs((pd.Timestamp(St["t"], unit="s") - d).days) <= C.META_DATE_DAYS]
                imp = St["side_rule"].startswith("impossible")
                disp = B.days_after(St["t"], C.NEIGH_STEP_DAYS)
                ref = B.days_before(St["t"], C.NEIGH_STEP_DAYS)
                # the statistic is (after - before); the displacement after - before is +j
                nt = B.neighbour_test(sid, disp, ref, St["j"], pre_c, post_c, pre_d, post_d, xy,
                                      spans, use_post=St["period"] == "post")
                row.update(_decide(co, md, imp, nt))
                row["j_displaced_side"] = jd
                rows.append(row)
    D = pd.DataFrame(rows)
    for c in ("neighbours", "coloc_covering", "coloc_jumped", "meta_dates"):
        if c in D:
            D[c] = D[c].map(lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple))
                            else (x if isinstance(x, str) else ""))
    D.to_parquet(work / f"decisions_{sub}.parquet", index=False)
    print(D.groupby(["period", "kind", "flag"]).size().to_string())
    print(D.groupby(["period", "kind", "neighbour_verdict"]).size().to_string())
    print(f"done in {time.time() - t0:.0f}s")


def _decide(co, md, imp, nt) -> dict:
    meta_backed = bool(imp or md)
    if meta_backed:
        flag, backing = 1, ("metadata: impossible displaced level" if imp else
                            "metadata: station metadata date within 7 days")
    elif co["backed"]:
        flag, backing = 1, "co-located: nested well(s) " + ",".join(co["covering"]) + \
            " record through the jump(s) without a simultaneous jump"
    elif nt.get("verdict") == "corroborated":
        flag, backing = 3, "neighbour regression residual > 4 calibrated sd"
    else:
        flag, backing = 9, "none"
    return {"flag": flag, "backing": backing, "meta_impossible": imp,
            "meta_dates": [str(d.date()) for d in md], "coloc_backed": co["backed"],
            "coloc_covering": co["covering"], "coloc_jumped": co["jumped"],
            "neighbour_verdict": nt.get("verdict"), "neighbour_why": nt.get("why"),
            "neighbour_delta_m": nt.get("delta_m"), "neighbour_sd_m": nt.get("sd_cal_m"),
            "neighbour_z": nt.get("z"), "neighbours": nt.get("neighbours", [])}


if __name__ == "__main__":
    main()
