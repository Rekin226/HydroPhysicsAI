"""Per-well H1a pipeline, stage 1: screens, barometric response, pass-2 detection, O1 phases.

Pre-origin and post-origin records are processed separately so that every value dated
<= 2022-12 depends on pre-origin data only:

* pre-origin pass: historical cache 2012-01-01 .. 2022-12-31 only; screen (pass 1), fit the
  barometric response on its clean hourly means, remove the loading, screen again (pass 2,
  final for these rows). The 15 x MAD level/scale and the barometric coefficients are
  fitted here.
* post-origin pass: new network from 2023-01-01, with 2022 of the historical cache as
  detection context (so an episode that starts before the origin is seen); flags are kept
  for rows >= 2023-01-01 only; it uses the pre-origin coefficients (no 2023+ data in any
  parameter of a well that has pre-origin data). Steps whose jump lies in the context are
  not applied here: the pre-origin decision is carried forward in stage 3.

Wells with no pre-origin record take their MAD reference from their own post-origin data
(these parameters touch post-origin rows only) and get no barometric correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import baro, rawio, screens, tides
from . import config as C

SPLIT = pd.Timestamp(C.ORIGIN_SPLIT)
SPLIT_S = int(SPLIT.value // 10**9)
CONTEXT_START = pd.Timestamp("2022-01-01")
PRE_MONTHS = pd.period_range("2012-01", "2022-12", freq="M")
POST_MONTHS = pd.period_range("2023-01", "2026-09", freq="M")

EP_KEYS = ("t1", "t2", "j1", "j2", "dropout", "seg_median", "seg_rsd", "pre", "n", "t_lo",
           "t_hi", "frac_flat", "spans_origin")
ST_KEYS = ("t", "j", "side", "side_rule", "t_lo", "t_hi", "median_before", "median_after",
           "impossible_before", "impossible_after", "to_end")


def robust_ref(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)]
    if len(v) < 100:
        return np.nan, np.nan
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad <= 0:
        mad = float(np.std(v)) or 1.0
    return med, mad


def hourly_clean(t: np.ndarray, v: np.ndarray, kind: np.ndarray) -> pd.Series:
    good = kind == 0
    if good.sum() == 0:
        return pd.Series(dtype=float)
    s = pd.Series(v[good], index=pd.to_datetime(t[good], unit="s"))
    return s.resample("h").mean()


def _segment_out(t, v_raw, L, res, flag_from_s=None):
    """Arrays of one segment restricted to t >= flag_from_s (if given)."""
    keep = np.ones(len(t), bool) if flag_from_s is None else (t >= flag_from_s)
    return {"t": t[keep], "v_raw": v_raw[keep], "L": L[keep], "kind": res["kind"][keep],
            "kind_base": res["kind_base"][keep], "cadence": res["cadence"]}


def _catalogue(res, flag_from_s=None, period="pre"):
    """Jumps, episodes and steps (absolute times) kept for this period."""
    ff = flag_from_s if flag_from_s is not None else -(10 ** 15)
    jumps = [{k: J[k] for k in ("t", "j", "cls", "pre", "post")} | {"period": period}
             for J in res["jumps"] if J["t"] >= ff]
    eps = [{k: E[k] for k in EP_KEYS} | {"period": period}
           for E in res["episodes"] if E["t2"] >= ff]
    steps = [{k: S[k] for k in ST_KEYS} | {"period": period}
             for S in res["steps"] if not S["context"]]
    unp = []
    for k in res["unpaired"]:
        J = res["jumps"][k]
        if abs(J["j"]) < C.STEP_MIN_M and J["t"] >= ff:
            unp.append({"t": J["t"], "j": J["j"], "period": period})
    return {"jumps": jumps, "episodes": eps, "steps": steps, "unpaired_small": unp}


def process_well(sid: str, meta: dict, P: pd.DataFrame | None, pre_only: bool = False,
                 hist_override: pd.Series | None = None, be_override: dict | None = None,
                 o1: bool = True) -> dict:
    """Stage 1 for one station. ``hist_override`` replaces the pre-origin raw series (for
    injection tests); ``be_override`` = {"segments": [{"start", "end", "a"}], "p_ref_hpa"}
    fixes the barometric loading instead of fitting it.

    Barometric segments: the pre-origin record is split at the logger change (first sustained
    10-minute sampling, ``baro.logger_switch``) when ``config.BARO_EPOCHS`` is set and both
    epochs hold >= 365 days; each epoch's response is fitted and accepted on its own, and the
    last epoch's response also applies to the post-origin rows (same logger at the origin)."""
    sh, sn = rawio.load_well(sid, keep_sentinels=True, pre_only=pre_only)
    if hist_override is not None:
        sh = hist_override
    elev = meta.get("WellElevation")
    depth = meta.get("WellDepth")
    elev = float(elev) if elev is not None and np.isfinite(elev) else None
    depth = float(depth) if depth is not None and np.isfinite(depth) and depth > 0 else None
    cid = meta.get("era5_cell")
    p = P[cid] if (P is not None and cid in P.columns) else None
    out = {"sid": sid, "cell": cid, "elev": elev, "depth": depth}

    # ------------------------------------------------------------- pre-origin
    t_pre, v_pre = screens.to_arrays(sh)
    med, mad = robust_ref(v_pre)
    ref_window = "pre_origin"
    if not np.isfinite(med) and not pre_only:
        t_n, v_n = screens.to_arrays(sn)
        med, mad = robust_ref(v_n)
        ref_window = "post_origin_only_well"
    out.update(med_ref=med, mad_ref=mad, ref_window=ref_window)
    p_ref = float(p[(p.index >= rawio.T0) & (p.index < SPLIT)].mean()) if p is not None \
        else np.nan
    segments = []
    if be_override is not None:
        segments = [dict(x) for x in be_override.get("segments", [])]
        p_ref = be_override.get("p_ref_hpa", p_ref)
        be = {"status": "fixed_override", "segments": segments,
              "accepted": any(x.get("a") is not None for x in segments)}
    else:
        be = {"status": "no_pre_origin_record" if not len(t_pre) else "no_pressure",
              "accepted": False, "segments": []}
        if len(t_pre):
            r1 = screens.screen(t_pre, v_pre, med_ref=med, mad_ref=mad, elev=elev, depth=depth)
            hh = hourly_clean(t_pre, v_pre, r1["kind"])
            if p is not None and len(hh.dropna()) > 0:
                hh = hh[(hh.index >= rawio.T0) & (hh.index < SPLIT)]
                pp = p[p.index < SPLIT + pd.Timedelta(days=1)]
                bounds = [int(t_pre[0]), SPLIT_S]
                good = np.isfinite(v_pre) & (np.abs(v_pre) < C.SENTINEL_ABS)
                sw = baro.logger_switch(t_pre[good]) if C.BARO_EPOCHS else None
                if sw is not None and (sw - bounds[0]) >= 365 * 86400 and \
                        (SPLIT_S - sw) >= 365 * 86400:
                    bounds = [bounds[0], sw, SPLIT_S]
                for k, (a_, b_) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
                    ta, tb = pd.Timestamp(a_, unit="s"), pd.Timestamp(b_, unit="s")
                    f = baro.fit_be(hh[(hh.index >= ta) & (hh.index < tb)], pp)
                    f.update(epoch=k, n_epochs=len(bounds) - 1, start=a_, end=b_,
                             a=f["a"] if f.get("accepted") else None,
                             a_fitted=f.get("a"))
                    segments.append(f)
                be = {"status": ";".join(str(x.get("status")) for x in segments),
                      "accepted": any(x.get("accepted") for x in segments),
                      "segments": segments, "logger_switch": sw}
            be["n_clean_samples_pass1"] = int((r1["kind"] == 0).sum())
    out["be"] = be
    out["p_ref_hpa"] = p_ref
    # loading per segment; the last segment extends to the end of the record
    L_parts = []
    for i, x in enumerate(segments):
        if x.get("a") is None or p is None:
            continue
        hi = 2 ** 62 if i == len(segments) - 1 else x["end"]
        L_parts.append((x["start"] if i else -(2 ** 62), hi,
                        baro.loading(np.asarray(x["a"]), p, p_ref)))
    a_last = np.asarray(segments[-1]["a"]) if segments and segments[-1].get("a") is not None \
        else None

    def corrected(t, v):
        if not L_parts:
            return v.copy(), np.zeros(len(v))
        L = np.zeros(len(t))
        for lo, hi, Lh in L_parts:
            m = (t >= lo) & (t < hi)
            if m.any():
                L[m] = baro.loading_at(t[m], Lh)
        L = np.where(np.isfinite(L), L, 0.0)
        good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
        vc = v.copy()
        vc[good] = v[good] - L[good]
        return vc, np.where(good, L, 0.0)

    if len(t_pre):
        vc, L = corrected(t_pre, v_pre)
        r2 = screens.screen(t_pre, vc, med_ref=med, mad_ref=mad, elev=elev, depth=depth)
        out["pre"] = _segment_out(t_pre, v_pre, L, r2)
        out["pre_cat"] = _catalogue(r2, None, "pre")
        if o1:
            out["o1_pre"] = _o1(t_pre, v_pre, vc, r2["kind"], PRE_MONTHS)
    if pre_only:
        return out

    # ------------------------------------------------------------- post-origin
    if len(sn):
        ctx = sh[sh.index >= CONTEXT_START]
        seg = pd.concat([ctx, sn]).sort_index()
        seg = seg[~seg.index.duplicated(keep="last")]
        t_s, v_s = screens.to_arrays(seg)
        vc, L = corrected(t_s, v_s)
        r = screens.screen(t_s, vc, med_ref=med, mad_ref=mad, elev=elev, depth=depth,
                           flag_from=SPLIT_S)
        out["post"] = _segment_out(t_s, v_s, L, r, SPLIT_S)
        out["post_cat"] = _catalogue(r, SPLIT_S, "post")
        keep = t_s >= SPLIT_S
        if o1:
            out["o1_post"] = _o1(t_s[keep], v_s[keep], vc[keep], r["kind"][keep], POST_MONTHS)
        # out-of-sample barometric check (after-the-fact, reported only; U7)
        if a_last is not None and p is not None:
            hk = hourly_clean(t_s[keep], v_s[keep], r["kind"][keep])
            out["be_oos"] = baro.oos_ratio(hk, p, a_last)
    return out


def _o1(t, v_raw, v_corr, kind, months):
    h_raw = hourly_clean(t, v_raw, kind)
    h_cor = hourly_clean(t, v_corr, kind)
    ph_r = tides.o1_phases(h_raw, months) if len(h_raw) else pd.DataFrame()
    ph_c = tides.o1_phases(h_cor, months) if len(h_cor) else pd.DataFrame()
    return {"raw": ph_r.to_dict("list"), "corr": ph_c.to_dict("list")}
