"""WP S1b repair round: what the validators are made of (U1), and sensitivities.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r2_u1.py

Logs an ``after_the_fact`` look first, then reads the three products (guarded,
``after_the_fact_analysis``) and the registered rows:

A. GNSS flag composition of every series the registered run used: days by flag in the clipped
   pre- and post-gap windows and at the two endpoints (gnss_clean flag 3 = a step correction
   corroborated only by the on-fan common mode, plan 2.4 "filled or split using other sites").
B. InSAR bridged increments: each site's gap change G split into the increments LiCSBAS set by
   its linear-velocity constraint across the network gaps (flag 4 in insar_069A/v2) and the
   interferometric remainder G_obs.
C. U1-strict readings (report only): (c) GNSS flags 0-1 is the registered variant (cited from
   after_the_fact.json); (d2) each bridged increment carried at 100 % uncertainty
   (sd = max(6 mm, |increment|)) and at 12 mm; (d1) from the interferometric increments only.
D. Empirical check of the a-priori 6 mm per bridge: InSAR-minus-GNSS changes across each network
   gap at the co-located WRA stations against within-segment intervals of the same length.
E. Ring-noise sensitivity: the registered computation with the ring-fit residual sd in place of
   the product's sd_s (an upper bound).
F. GNSS data gaps (days without a flag 0/1/3 height) inside each ring gap, every WRA station.
G. SJES endpoint means (+-3 d and +-7 d; corrected and raw; flags 0/1/3 and all flags).

Writes ``S1b_evidence/repair/u1_and_sensitivity.json``. Nothing is fed back into S1a; the
registered result is not changed.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_core as C  # noqa: E402
from s1_rings.s1b import s1b_io as IO  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r2_looks as LK  # noqa: E402
from s1_rings.s1b import s1b_run as RUN  # noqa: E402

OUT = K.EVIDENCE / "repair"
FLAGS = tuple(K.GNSS_FLAGS)


def _rows() -> dict:
    df = pd.read_csv(K.EVIDENCE / "registered_rows.csv")
    df["evaluable"] = df["evaluable"].astype(str).str.lower().eq("true")
    return {t: g.set_index("site", drop=False) for t, g in df.groupby("test")}


# ----------------------------------------------------------------------------- A
def gnss_flag_composition(g: dict, prs: dict, rings: dict) -> list[dict]:
    d = g["daily"]
    out = []
    for fam in ("wra_colocated", "ngl_near", "neighbour"):
        for p in prs[fam]:
            r = rings[p["site"]]
            x = d[(d.family == p["family"]) & (d.station == p["station"])
                  & d["flag"].isin(list(FLAGS))].sort_values("date")
            t = pd.DatetimeIndex(x["date"])
            f = x["flag"].to_numpy()
            pre = (max(pd.Timestamp(K.PRE_WIN[0]), t.min()),
                   min(pd.Timestamp(K.PRE_WIN[1]), pd.Timestamp(r["t_lp"])))
            post = (max(pd.Timestamp(K.POST_WIN[0]), pd.Timestamp(r["t_fp"])),
                    min(pd.Timestamp(K.POST_WIN[1]), t.max()))
            row = {"family": fam, "site_en": K.SITE_EN[p["site"]], "station": p["station"]}
            for w, (lo, hi) in (("pre", pre), ("post", post)):
                m = (t >= lo) & (t <= hi)
                row[f"{w}_days"] = int(m.sum())
                for fl in FLAGS:
                    row[f"{w}_flag{fl}"] = int((m & (f == fl)).sum())
            for e, c in (("end_lp", r["t_lp"]), ("end_fp", r["t_fp"])):
                dd = np.abs((t - pd.Timestamp(c)).days.to_numpy())
                h = K.GNSS_END_HALF_D if (dd <= K.GNSS_END_HALF_D).sum() >= K.GNSS_END_MIN \
                    else K.GNSS_END_HALF_D_FALLBACK
                m = dd <= h
                row[f"{e}_half_d"] = int(h)
                for fl in FLAGS:
                    row[f"{e}_flag{fl}"] = int((m & (f == fl)).sum())
            row["post_flag3_share"] = (row["post_flag3"] / row["post_days"]
                                       if row["post_days"] else None)
            row["pre_flag3_share"] = (row["pre_flag3"] / row["pre_days"]
                                      if row["pre_days"] else None)
            row["endpoint_flag3_days"] = row["end_lp_flag3"] + row["end_fp_flag3"]
            out.append(row)
    return out


# ----------------------------------------------------------------------------- B
def insar_bridges(ins: dict, xy: dict, rings: dict, rows: pd.DataFrame) -> list[dict]:
    out = []
    for s in K.SITES:
        r = rings[s]
        x, y = xy[s]
        px = IO.insar_pixel_series(ins, x, y)
        t, d, seg = ins["dates"], px["down_mm"], ins["network_segment"]
        e0 = C.endpoint_mean(t, d, r["t_lp"], K.INSAR_END_HALF_D, K.INSAR_END_MIN, seg)
        e1 = C.endpoint_mean(t, d, r["t_fp"], K.INSAR_END_HALF_D, K.INSAR_END_MIN, seg)
        G = e1["mean"] - e0["mean"]
        br = R.bridge_increments(t, d, seg, e0["seg"], e1["seg"])
        inc = [b["inc_mm"] for b in br]
        days = sum(b["days"] or 0 for b in br)
        reg = rows.loc[s]
        out.append({"site": s, "site_en": K.SITE_EN[s], "G_mm": G,
                    "G_registered_mm": float(reg["I_gap_mm"]),
                    "seg_lp": e0["seg"], "seg_fp": e1["seg"], "n_bridges": len(br),
                    "bridges": br, "bridged_sum_mm": float(np.nansum(inc)),
                    "bridged_days": int(days), "gap_days": int((pd.Timestamp(r["t_fp"])
                                                                - pd.Timestamp(r["t_lp"])).days),
                    "G_obs_mm": G - float(np.nansum(inc)),
                    "bridged_share_of_G": (float(np.nansum(inc)) / G) if G else None})
    return out


# ----------------------------------------------------------------------------- C
def d2_bridge_sd(rows: pd.DataFrame, br: list[dict], mode: str) -> dict:
    """(d2) with the bridge term replaced: mode '12mm' or 'full' (sd = max(6, |inc|))."""
    res = []
    for b in br:
        r = rows.loc[b["site"]]
        if not r["evaluable"]:
            continue
        n = int(r["n_bridges"]) if np.isfinite(r["n_bridges"]) else 0
        if mode == "12mm":
            v_new = n * 12.0 ** 2
        else:
            v_new = sum(max(K.INSAR_BRIDGE_SD_MM, abs(x["inc_mm"])) ** 2 for x in b["bridges"]
                        if np.isfinite(x["inc_mm"]))
        dv = v_new - n * K.INSAR_BRIDGE_SD_MM ** 2
        sp = math.sqrt(r["sigma_pre_mm"] ** 2 + dv)
        sq = math.sqrt(r["sigma_post_mm"] ** 2 + dv)
        L = 1000.0 * r["L_m"]
        res.append({"site": b["site"], "evaluable": True,
                    "C_primary_mm": r["C_primary_mm"], "C_raw_mm": r["C_raw_mm"],
                    "C_pre_mm": r["C_pre_mm"], "C_post_mm": r["C_post_mm"],
                    "C_mid_mm": r["C_mid_mm"], "sigma_pre_mm": sp, "sigma_post_mm": sq,
                    "L_m": L / 1000.0})
    fp = R.family_counts(res, "k:2", "C_primary_mm")
    fr = R.family_counts(res, "k:2", "C_raw_mm")
    return {"mode": mode, "primary": fp, "raw": fr,
            "rmse_primary_mm": R.rmse(res, "C_primary_mm", "C_mid_mm"),
            "rmse_raw_mm": R.rmse(res, "C_raw_mm", "C_mid_mm"),
            "pass": bool(fp["containment_pass"] and R.rmse(res, "C_primary_mm", "C_mid_mm")
                         < R.rmse(res, "C_raw_mm", "C_mid_mm"))}


def d1_obs(rows: pd.DataFrame, br: list[dict], denom: str) -> dict:
    x, yb, yr = [], [], []
    for b in br:
        r = rows.loc[b["site"]]
        gy = float(r["gap_yr"])
        if denom == "observed_span":
            x.append(b["G_obs_mm"] / (gy - b["bridged_days"] / 365.25))
        else:
            x.append(b["G_obs_mm"] / gy)
        yb.append(r["C_primary_mm"] / gy)
        yr.append(r["C_raw_mm"] / gy)
    out = C.corr_check(x, yb, yr, K.CORR_ALPHA)
    r23 = float(np.corrcoef(yb, yr)[0, 1])
    out["williams"] = R.williams_t(out["r_bridged"], out["r_raw"], r23, out["n"])
    out["denominator"] = denom
    return out


# ----------------------------------------------------------------------------- D
def _gnss_mean(t: pd.DatetimeIndex, d: np.ndarray, c, h: int = 3, nmin: int = 3) -> float:
    dd = np.abs((t - pd.Timestamp(c)).days.to_numpy())
    m = (dd <= h) & np.isfinite(d)
    return float(d[m].mean()) if m.sum() >= nmin else float("nan")


def bridge_vs_gnss(g: dict, ins: dict, xy: dict, prs: dict) -> dict:
    dates = pd.DatetimeIndex(ins["dates"])
    netseg = np.asarray(ins["network_segment"])
    gaps = []
    for s in range(int(netseg.max())):
        i = int(np.flatnonzero(netseg == s)[-1])
        j = int(np.flatnonzero(netseg == s + 1)[0])
        gaps.append((s, i, j, int((dates[j] - dates[i]).days)))
    rec_b, rec_w = [], []
    end = pd.Timestamp(K.POST_WIN[1])
    for p in prs["wra_colocated"]:
        x, y = xy[p["site"]]
        px = IO.insar_pixel_series(ins, x, y)
        di = px["down_mm"]
        t, dg, _ = IO.gnss_series(g, "wra", p["station"], flags=FLAGS)
        st = g["steps"]
        sd = pd.DatetimeIndex(pd.to_datetime(st[(st.family == "wra")
                                                & (st.station == p["station"])]["date"]))
        gm = {}

        def gmean(k, gm=gm, t=t, dg=dg):
            if k not in gm:
                gm[k] = _gnss_mean(t, dg, dates[k])
            return gm[k]

        def stepfree(a, b, sd=sd):
            return not ((sd > dates[a] - pd.Timedelta(days=3))
                        & (sd < dates[b] + pd.Timedelta(days=3))).any()

        for _s, i, j, days in gaps:
            if dates[j] > end or not (np.isfinite(di[i]) and np.isfinite(di[j])):
                continue
            if not stepfree(i, j):
                continue
            g0, g1 = gmean(i), gmean(j)
            if not (np.isfinite(g0) and np.isfinite(g1)):
                continue
            rec_b.append({"station": p["station"], "site_en": K.SITE_EN[p["site"]],
                          "gap": f"{dates[i].date()}/{dates[j].date()}", "days": days,
                          "d_insar": float(di[j] - di[i]), "d_gnss": g1 - g0,
                          "diff": float(di[j] - di[i]) - (g1 - g0)})
        lens = sorted({gp[3] for gp in gaps})
        idx = np.flatnonzero(np.isfinite(di) & (dates <= end))
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                dd = int((dates[b] - dates[a]).days)
                if dd > max(lens):
                    break
                if dd not in lens or netseg[a] != netseg[b] or not stepfree(a, b):
                    continue
                g0, g1 = gmean(a), gmean(b)
                if not (np.isfinite(g0) and np.isfinite(g1)):
                    continue
                rec_w.append({"station": p["station"], "days": dd,
                              "year": int(dates[a].year),
                              "diff": float(di[b] - di[a]) - (g1 - g0)})
    B = pd.DataFrame(rec_b)
    W = pd.DataFrame(rec_w)
    summ = {}
    for gp, gb in (B.groupby("gap") if len(B) else []):
        days = int(gb["days"].iloc[0])
        w = W[W.days == days]["diff"] if len(W) else pd.Series(dtype=float)
        ms_b = float(np.mean(gb["diff"] ** 2))
        ms_w = float(np.mean(w ** 2)) if len(w) else float("nan")
        summ[gp] = {"days": days, "n_stations": int(len(gb)),
                    "mean_diff_mm": float(gb["diff"].mean()),
                    "rms_diff_mm": math.sqrt(ms_b),
                    "sd_diff_mm": float(gb["diff"].std(ddof=1)) if len(gb) > 1 else None,
                    "within_segment_rms_same_length_mm": math.sqrt(ms_w) if len(w) else None,
                    "n_within": int(len(w)),
                    "bridge_error_rms_est_mm": (math.sqrt(max(0.0, ms_b - ms_w))
                                                if len(w) else None)}
    allb = B[B["gap"].str.startswith("2022")] if len(B) else B
    pooled = {}
    if len(allb):
        w = W[W.days.isin(allb["days"].unique())]["diff"]
        ms_b, ms_w = float(np.mean(allb["diff"] ** 2)), float(np.mean(w ** 2))
        pooled = {"n": int(len(allb)), "rms_diff_mm": math.sqrt(ms_b),
                  "mean_diff_mm": float(allb["diff"].mean()),
                  "within_rms_mm": math.sqrt(ms_w),
                  "bridge_error_rms_est_mm": math.sqrt(max(0.0, ms_b - ms_w)),
                  "per_site_sum_of_two_2022_bridges_mm": allb.groupby("site_en")["diff"]
                  .sum().round(2).to_dict()}
    return {"gaps": [{"from_seg": s, "last": str(dates[i].date()), "first": str(dates[j].date()),
                      "days": d} for s, i, j, d in gaps],
            "per_bridge": B.round(3).to_dict(orient="records") if len(B) else [],
            "summary": summ, "pooled_2022": pooled,
            "within_by_length": {int(k): {"n": int(len(v)),
                                          "rms_mm": float(np.sqrt(np.mean(v["diff"] ** 2))),
                                          "mean_mm": float(v["diff"].mean())}
                                 for k, v in W.groupby("days")} if len(W) else {}}


# ----------------------------------------------------------------------------- E
def ring_resid_sd(rings: dict) -> dict:
    out = {}
    for s, r in rings.items():
        t = pd.DatetimeIndex(r["t"])
        c = np.asarray(r["c"], float)
        t_mid = C.years([pd.Timestamp(r["t_lp"]) + (pd.Timestamp(r["t_fp"])
                                                    - pd.Timestamp(r["t_lp"])) / 2])[0]
        mp = C.window_select(t, K.PRE_WIN[0], min(pd.Timestamp(K.PRE_WIN[1]),
                                                  pd.Timestamp(r["t_lp"]))) & np.isfinite(c)
        fp = C.fit_pre(C.years(t[mp]), c[mp], t_mid, True)
        mq = C.window_select(t, max(pd.Timestamp(K.POST_WIN[0]), pd.Timestamp(r["t_fp"])),
                             K.POST_WIN[1]) & np.isfinite(c)
        yq = C.years(t[mq])
        zq = c[mq] - fp.season(yq)
        X = np.stack([np.ones_like(yq), yq - t_mid], 1)
        beta, *_ = np.linalg.lstsq(X, zq, rcond=None)
        res = zq - X @ beta
        sq = float(np.sqrt((res ** 2).sum() / max(len(zq) - 2, 1)))
        out[s] = {"sd_s_mm": r["sd_s_mm"], "resid_pre_mm": fp.resid_sd, "resid_post_mm": sq,
                  "n_pre": int(mp.sum()), "n_post": int(mq.sum())}
    return out


def ring_sd_variant(rings, xy, g, ins, sds: dict, key: str) -> dict:
    rr = {s: {**r, "sd_s_mm": (sds[s][key] if key != "max" else
                               max(sds[s]["resid_pre_mm"], sds[s]["resid_post_mm"]))}
          for s, r in rings.items()}
    res = RUN.compute(RUN.PRIMARY, rr, xy, g, ins)
    return {"sd": key, "verdicts": {k: {kk: res["verdicts"][k][kk] for kk in
                                        ("n_evaluable", "n_contain_primary", "n_contain_raw",
                                         "rmse_primary_mm", "rmse_raw_mm", "pass",
                                         "sites_failing")}
                                    for k in ("c_wra", "c_ngl", "c_neighbour", "d_insar")},
            "d1": res["d1"], "gate_pass": res["gate_pass"]}


# ----------------------------------------------------------------------------- F, G
def gnss_gaps(g: dict, rings: dict, prs: dict) -> list[dict]:
    d = g["daily"]
    st = g["stations"]
    pair_site = {p["station"]: p["site"] for p in prs["wra_colocated"]}
    out = []
    for r in st[st.family == "wra"].itertuples():
        x = d[(d.family == "wra") & (d.station == r.station) & d["flag"].isin(list(FLAGS))]
        if not len(x):
            continue
        if r.station in pair_site:
            rg = rings[pair_site[r.station]]
            lo, hi, basis = rg["t_lp"], rg["t_fp"], K.SITE_EN[pair_site[r.station]]
        else:
            lo, hi, basis = pd.Timestamp("2021-12-01"), pd.Timestamp("2023-02-28"), \
                "common window 2021-12-01..2023-02-28"
        gp = R.longest_gap(x["date"], lo, hi)
        out.append({"station": r.station, "window": basis, "lo": str(pd.Timestamp(lo).date()),
                    "hi": str(pd.Timestamp(hi).date()), **gp})
    return sorted(out, key=lambda z: -z["max_gap_days"])


def sjes_endpoints(g: dict, rings: dict) -> dict:
    d = g["daily"]
    r = rings["溪州國小"]
    out = {"t_lp": str(pd.Timestamp(r["t_lp"]).date()), "t_fp": str(pd.Timestamp(r["t_fp"]).date())}
    for fl_name, fl in (("flags013", FLAGS), ("all_flags", None)):
        x = d[(d.family == "wra") & (d.station == "SJES")]
        if fl is not None:
            x = x[x["flag"].isin(list(fl))]
        t = pd.DatetimeIndex(x["date"])
        for col in ("value", "value_raw"):
            v = -1000.0 * x[col].to_numpy(float)
            for h in (3, 7):
                e = []
                ns = []
                for c in (r["t_lp"], r["t_fp"]):
                    m = np.abs((t - pd.Timestamp(c)).days.to_numpy()) <= h
                    e.append(float(v[m].mean()) if m.any() else float("nan"))
                    ns.append(int(m.sum()))
                out[f"{fl_name}|{'corrected' if col == 'value' else 'raw'}|pm{h}d"] = {
                    "gap_change_mm": e[1] - e[0], "n_lp": ns[0], "n_fp": ns[1]}
    return out


def main() -> int:
    LK.u1()
    rings, _ = IO.rings()
    xy = IO.site_xy()
    g = IO.gnss()
    ins = IO.insar()
    rows = _rows()
    prs = IO.pairs(g["stations"], xy)
    out = {"gnss_flag_composition": gnss_flag_composition(g, prs, rings)}
    br = insar_bridges(ins, xy, rings, rows["d_insar"])
    out["insar_bridges"] = br
    out["u1_strict"] = {
        "d2_bridge_12mm": d2_bridge_sd(rows["d_insar"], br, "12mm"),
        "d2_bridge_full": d2_bridge_sd(rows["d_insar"], br, "full"),
        "d1_observed_increments_observed_span": d1_obs(rows["d_insar"], br, "observed_span"),
        "d1_observed_increments_full_gap": d1_obs(rows["d_insar"], br, "full_gap"),
    }
    out["bridge_vs_gnss"] = bridge_vs_gnss(g, ins, xy, prs)
    sds = ring_resid_sd(rings)
    out["ring_resid_sd"] = {K.SITE_EN[s]: v for s, v in sds.items()}
    out["ring_sd_variants"] = {k: ring_sd_variant(rings, xy, g, ins, sds, k)
                               for k in ("resid_pre_mm", "max")}
    out["gnss_gaps_in_ring_gap"] = gnss_gaps(g, rings, prs)
    out["sjes_endpoints"] = sjes_endpoints(g, rings)
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.dump(out, OUT / "u1_and_sensitivity.json")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out["gnss_flag_composition"]).to_string())
    print(pd.DataFrame([{k: v for k, v in b.items() if k != "bridges"} for b in br])
          .round(2).to_string())
    for k, v in out["u1_strict"].items():
        print(k, {kk: v[kk] for kk in v if kk not in ("primary", "raw")},
              v.get("primary", {}).get("n_contain"), v.get("raw", {}).get("n_contain"))
    print(out["bridge_vs_gnss"]["summary"])
    print(out["bridge_vs_gnss"]["pooled_2022"])
    print(out["bridge_vs_gnss"]["within_by_length"])
    print(pd.DataFrame(out["ring_resid_sd"]).T.round(2).to_string())
    for k, v in out["ring_sd_variants"].items():
        print(k, {kk: (vv["n_contain_primary"], vv["n_contain_raw"], vv["pass"])
                  for kk, vv in v["verdicts"].items()}, v["gate_pass"])
    print(pd.DataFrame(out["gnss_gaps_in_ring_gap"]).head(15).to_string())
    print(out["sjes_endpoints"])
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
