"""Shared scoring for WP H2: product predictions from run tables, RMSE and coverage strata.

The product's prediction for a monthly cell is the method chosen for its gap class
(``router`` unless the design stage set a fall-back) with sd = max(factor[class] x native sd
of that method, SD_FLOOR); for a daily-family cell it is the bridge value with the sd of its
bridged-share class. Coverage uses the A1 rule gate-3 (``harness.coverage3``) with the
clusters stored in the run tables (wells for S, E, T_S and d; the gap window for N; the
realization for H and T_H06).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from h2_heads_fill import config as C

BASELINES = ("backfill", "clim_ar", "idw", "stk")
REPORTED = ("clim", "neigh", "softimpute_tb", "gp", "router")
NATIVE_SD = ("router", "neigh", "softimpute_tb", "gp", "clim", "clim_ar", "stk")


def load_runs(variant, families, rs, workdir=C.WORKDIR):
    parts = []
    for f in families:
        for r in rs[f] if isinstance(rs, dict) else rs:
            p = workdir / "runs" / variant / f"{f}_r{r}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
            else:
                raise FileNotFoundError(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def class_key(cls: str, params: dict) -> str:
    """Factor class used for a gap class: merged classes map to their pool
    (``class_map``); a class the design never saw follows the design's pooling rule
    (``pool_rule``) to the first class that has a factor, and otherwise takes the class with
    the largest factor (conservative)."""
    fac = params.get("factor_by_class", {})
    k = params.get("class_map", {}).get(cls, cls)
    pool, seen = params.get("pool_rule", {}), set()
    while k not in fac and k in pool and k not in seen:
        seen.add(k)
        k = pool[k]
    return k if k in fac else max(fac, key=fac.get)


def product_monthly(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add pred_product / sd_product / method_product to a monthly run table."""
    out = df.copy()
    keys = [class_key(c, params) for c in out.cls]
    meth = [params["method_by_class"].get(k, "router") for k in keys]
    ref = [params["sd_ref_by_class"].get(k, m) for k, m in zip(keys, meth, strict=True)]
    fac = np.array([params["factor_by_class"].get(k, np.nan) for k in keys])
    pm = np.array([r[f"pred_{m}"] for r, m in zip(out.to_dict("records"), meth, strict=True)])
    sn = np.array([r[f"sd_{s}"] for r, s in zip(out.to_dict("records"), ref, strict=True)])
    out["class_key"] = keys
    out["method_product"] = meth
    out["pred_product"] = pm
    out["sd_product"] = np.maximum(fac * sn, C.SD_FLOOR)
    return out


def product_daily(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from h2_heads_fill.bridge import frac_class
    out = df[df.ok].copy()
    sd = params["bridge_sd_by_class"]
    out["frac_class"] = [frac_class(f) for f in out.frac]
    out["pred_product"] = out.pred_bridge
    out["sd_product"] = [max(sd[str(k)], C.SD_FLOOR) for k in out.frac_class]
    out["method_product"] = "daily_bridge"
    return out


def rmse(e) -> float:
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def rmse_table(df: pd.DataFrame, methods, by="stratum") -> pd.DataFrame:
    rows = []
    for s, g in df.groupby(by):
        rec = {by: s, "n": len(g), "n_clusters": g.cluster.nunique()}
        for m in methods:
            col = f"pred_{m}"
            if col not in g:
                continue
            sel = g
            if m == "idw":
                sel = g[g.idw_defined.astype(bool)] if "idw_defined" in g else g
            rec[f"rmse_{m}"] = rmse(sel[col] - sel.truth)
            rec[f"n_{m}"] = int(np.isfinite(sel[col] - sel.truth).sum())
            if m == "idw" and "pred_product" in g:
                rec["rmse_product_on_idw_cells"] = rmse(sel.pred_product - sel.truth)
        rows.append(rec)
    return pd.DataFrame(rows)


def coverage_rows(df: pd.DataFrame, by="stratum") -> list[dict]:
    from harness import coverage3
    out = []
    for s, g in df.groupby(by):
        err = (g.pred_product - g.truth).to_numpy()
        hits = np.abs(err) <= C.Z90 * g.sd_product.to_numpy()
        gate = coverage3.coverage_gate(hits, g.cluster.astype(str).to_numpy())
        gate.update({by: s, "n_cells": int(len(g)), "rmse_product": rmse(err),
                     "median_width90": float(np.median(2 * C.Z90 * g.sd_product))})
        out.append(gate)
    return out


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=conv, ensure_ascii=False))
