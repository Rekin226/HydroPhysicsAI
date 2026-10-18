"""WP S1b repair round: the model-set part of the InSAR gap changes, with epoch averaging.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r2_bridge.py

Follows ``s1b_r2_u1.py``, whose single-epoch split of G at the bridges turned out to be
dominated by single-epoch noise (the epochs next to the 2022 network gaps scatter by 15-30 mm).
Logs a ``diagnose`` look first. For every ring site:

* G (the registered InSAR gap change) is split into interferometric spans and model-set
  windows. A model-set window runs from the last 36 d of the long segment before a network gap
  (mean of its epochs, LATE) to the first 36 d of the next long segment (EARLY); segments with
  fewer than 6 epochs (the 2022-06-30..08-05 segment) lie inside the window, because their level
  relative to both neighbours is set by LiCSBAS's linear-velocity constraint. B = sum over
  windows of mean(EARLY) - mean(LATE) is model-set (flag 4); G_int = G - B is interferometric.
* At the 10 co-located WRA stations and Jiaxing-DNAN, the same epoch sets are compared with the
  GNSS (means of the daily heights within +-3 d of each epoch): the InSAR-minus-GNSS difference
  of B, against a null of within-segment windows of the same shape (36 d sets, 72-96 d apart),
  per station and as cross-station means per window (common mode).
* Report-only readings: (d1) from G_int over the observed span; (d2) with B carried at 100 %
  uncertainty (sd = max(registered bridge sd, |B|)) and with the empirical bridge error.
* Empirical InSAR epoch noise from 12 d within-segment InSAR-minus-GNSS increments, against the
  declared 6.6 mm.
* SJES: the raw gap change with the registered endpoint rule (+-3 d at t_lp, +-7 d fallback at
  t_fp).

Writes ``S1b_evidence/repair/bridge_windows.json``. Report only; nothing is fed back.
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
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_core as C  # noqa: E402
from s1_rings.s1b import s1b_io as IO  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b.s1b_r2_looks import REG_UNITS  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

OUT = K.EVIDENCE / "repair"
FLAGS = tuple(K.GNSS_FLAGS)
SET_D = 36            # days spanned by a LATE / EARLY epoch set
LONG_SEG = 6          # segments with fewer epochs lie inside a model-set window
NULL_SEP = (72, 96)   # null windows: days from the last LATE epoch to the first EARLY epoch


def log_look(extra: str = "") -> None:
    m = {"S1b:repair.bridge_windows_2026-10-14": {
        "none": "InSAR pixels at the 14 ring sites and GNSS at their stations; registered rows "
                "(no registered member)",
        "description": "repair round: model-set windows of the InSAR gap changes with epoch "
                       "averaging, against GNSS"}}
    prereg3.record_look(REG_UNITS + sorted(m), "rings+gnss+insar:2020-01..2025-12", K.WP,
                        "diagnose", members=m, prereg_id=K.PREREG_ID,
                        note="Repair round: s1b_r2_u1.py's single-epoch split of the InSAR gap "
                             "change at the network gaps was dominated by epoch noise; this "
                             "re-does the split with 36 d epoch sets (model-set windows vs "
                             "interferometric spans), compares the windows with GNSS against a "
                             "within-segment null, and recomputes the report-only d1/d2 "
                             "readings and the empirical InSAR epoch noise. Report only."
                             + extra)


def _rows() -> pd.DataFrame:
    df = pd.read_csv(K.EVIDENCE / "registered_rows.csv")
    df["evaluable"] = df["evaluable"].astype(str).str.lower().eq("true")
    return df


def epoch_sets(dates, d, seg, s: int, which: str) -> np.ndarray:
    idx = np.flatnonzero((seg == s) & np.isfinite(d))
    if not len(idx):
        return idx
    if which == "late":
        t_end = dates[idx[-1]]
        return idx[(dates[idx] >= t_end - pd.Timedelta(days=SET_D))]
    t0 = dates[idx[0]]
    return idx[(dates[idx] <= t0 + pd.Timedelta(days=SET_D))]


def split_site(dates, d, seg, netseg_n: dict, t_lp, t_fp) -> dict:
    e0 = C.endpoint_mean(dates, d, t_lp, K.INSAR_END_HALF_D, K.INSAR_END_MIN, seg)
    e1 = C.endpoint_mean(dates, d, t_fp, K.INSAR_END_HALF_D, K.INSAR_END_MIN, seg)
    s0, s1 = int(e0["seg"]), int(e1["seg"])
    longs = [s for s in range(s0, s1 + 1) if s in (s0, s1) or netseg_n.get(s, 0) >= LONG_SEG]
    windows = []
    for a, b in zip(longs[:-1], longs[1:], strict=True):
        la, eb = epoch_sets(dates, d, seg, a, "late"), epoch_sets(dates, d, seg, b, "early")
        windows.append({"from_seg": a, "to_seg": b, "late_idx": la.tolist(),
                        "early_idx": eb.tolist(),
                        "late_dates": [str(dates[i].date()) for i in la],
                        "early_dates": [str(dates[i].date()) for i in eb],
                        "B_mm": float(d[eb].mean() - d[la].mean()),
                        "span_days": float((dates[eb].mean() - dates[la].mean()).days)})
    B = sum(w["B_mm"] for w in windows)
    G = e1["mean"] - e0["mean"]
    span = sum(w["span_days"] for w in windows)
    return {"G_mm": G, "B_mm": B, "G_int_mm": G - B, "windows": windows,
            "model_span_days": span, "seg_lp": s0, "seg_fp": s1,
            "e0_idx": None, "n_windows": len(windows)}


def gnss_at(t: pd.DatetimeIndex, dg: np.ndarray, when) -> float:
    dd = np.abs((t - pd.Timestamp(when)).days.to_numpy())
    m = (dd <= 3) & np.isfinite(dg)
    return float(dg[m].mean()) if m.sum() >= 3 else float("nan")


def main() -> int:
    log_look(" " + " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "")
    rings, _ = IO.rings()
    xy = IO.site_xy()
    g = IO.gnss()
    ins = IO.insar()
    rows = _rows()
    prs = IO.pairs(g["stations"], xy)
    dates = pd.DatetimeIndex(ins["dates"])
    netseg = np.asarray(ins["network_segment"])
    netseg_n = {int(s): int((netseg == s).sum()) for s in np.unique(netseg)}
    ins_rows = rows[rows.test == "d_insar"].set_index("site")
    sta_of = {p["site"]: ("wra", p["station"]) for p in prs["wra_colocated"]}
    sta_of.update({p["site"]: ("ngl", p["station"]) for p in prs["ngl_near"]})

    sites, gn_cache = [], {}
    for s in K.SITES:
        r = rings[s]
        x, y = xy[s]
        d = IO.insar_pixel_series(ins, x, y)["down_mm"]
        sp = split_site(dates, d, netseg, netseg_n, r["t_lp"], r["t_fp"])
        rec = {"site": s, "site_en": K.SITE_EN[s], **{k: v for k, v in sp.items()
                                                       if k != "e0_idx"},
               "G_registered_mm": float(ins_rows.loc[s, "I_gap_mm"]),
               "gap_days": int((pd.Timestamp(r["t_fp"]) - pd.Timestamp(r["t_lp"])).days)}
        rec["B_share_of_G"] = rec["B_mm"] / rec["G_mm"] if rec["G_mm"] else None
        if s in sta_of:
            fam, st = sta_of[s]
            t, dg, _ = IO.gnss_series(g, fam, st, flags=FLAGS)
            gn_cache[s] = (t, dg, d, fam, st)
            gv = np.array([gnss_at(t, dg, dates[i]) for i in range(len(dates))])
            bg = []
            for w in sp["windows"]:
                le, ea = gv[w["late_idx"]], gv[w["early_idx"]]
                bg.append(float(np.nanmean(ea) - np.nanmean(le))
                          if np.isfinite(le).any() and np.isfinite(ea).any() else float("nan"))
            rec["gnss_station"] = st
            rec["B_gnss_mm"] = float(np.sum(bg))
            rec["B_minus_gnss_mm"] = rec["B_mm"] - rec["B_gnss_mm"]
            rec["per_window_B_minus_gnss_mm"] = [w["B_mm"] - b for w, b in zip(sp["windows"], bg,
                                                                       strict=True)]
        sites.append(rec)

    # ---- null: within-segment windows of the same shape, at the GNSS-paired sites
    null = []
    for s, (t, dg, d, _fam, st) in gn_cache.items():
        gv = np.array([gnss_at(t, dg, dates[i]) for i in range(len(dates))])
        ok = np.isfinite(d) & np.isfinite(gv) & (dates <= pd.Timestamp(K.POST_WIN[1]))
        for sg in np.unique(netseg):
            idx = np.flatnonzero((netseg == sg) & ok)
            for a in idx:
                la = idx[(dates[idx] <= dates[a]) & (dates[idx] >= dates[a]
                                                     - pd.Timedelta(days=SET_D))]
                cand = idx[((dates[idx] - dates[a]).days >= NULL_SEP[0])
                           & ((dates[idx] - dates[a]).days <= NULL_SEP[1])]
                if len(la) < 2 or not len(cand):
                    continue
                b = cand[0]
                eb = idx[(dates[idx] >= dates[b]) & (dates[idx] <= dates[b]
                                                     + pd.Timedelta(days=SET_D))]
                if len(eb) < 2:
                    continue
                di = d[eb].mean() - d[la].mean()
                dgg = gv[eb].mean() - gv[la].mean()
                null.append({"site_en": K.SITE_EN[s], "station": st,
                             "a": str(dates[a].date()), "b": str(dates[b].date()),
                             "diff": float(di - dgg)})
    N = pd.DataFrame(null)
    bw = pd.DataFrame([{"site_en": x["site_en"], "station": x.get("gnss_station"),
                        "diff": x["per_window_B_minus_gnss_mm"][0]}
                       for x in sites if "per_window_B_minus_gnss_mm" in x
                       and x["n_windows"] >= 1 and np.isfinite(x["per_window_B_minus_gnss_mm"][0])])
    cm = N.groupby(["a", "b"])["diff"].agg(["mean", "count"]).reset_index()
    cm = cm[cm["count"] >= 5]
    site_null = N.groupby("site_en")["diff"].agg(lambda v: float(np.sqrt(np.mean(v ** 2))))
    null_sd_within_window = float(np.sqrt(np.mean(
        N.groupby(["a", "b"])["diff"].transform(lambda v: v - v.mean()) ** 2)))
    first = {
        "n_sites": int(len(bw)), "mean_diff_mm": float(bw["diff"].mean()),
        "sd_diff_mm": float(bw["diff"].std(ddof=1)),
        "rms_diff_mm": float(np.sqrt(np.mean(bw["diff"] ** 2))),
        "per_site_mm": bw.set_index("site_en")["diff"].round(2).to_dict(),
        "null_n_windows": int(len(N)), "null_rms_mm": float(np.sqrt(np.mean(N["diff"] ** 2))),
        "null_sd_about_window_mean_mm": null_sd_within_window,
        "null_common_mode_rms_mm": float(np.sqrt(np.mean(cm["mean"] ** 2))),
        "null_common_mode_n": int(len(cm)),
        "null_common_mode_abs_q95_mm": float(np.quantile(np.abs(cm["mean"]), 0.95)),
        "null_per_site_rms_mm": site_null.round(2).to_dict()}
    first["site_specific_bridge_error_est_mm"] = math.sqrt(max(
        0.0, first["sd_diff_mm"] ** 2 - null_sd_within_window ** 2))
    first["common_bias_z_vs_null"] = (first["mean_diff_mm"] / first["null_common_mode_rms_mm"]
                                      if first["null_common_mode_rms_mm"] else None)

    # ---- empirical epoch noise (12 d within-segment single-epoch increments)
    noise = []
    for s, (t, dg, d, _fam, st) in gn_cache.items():
        gv = np.array([gnss_at(t, dg, dates[i]) for i in range(len(dates))])
        ok = np.flatnonzero(np.isfinite(d) & np.isfinite(gv))
        for lag, per in ((12, "all"), (6, "2020-2021"), (12, "2020-2021"), (12, "2022-2025")):
            df = [(d[j] - d[i]) - (gv[j] - gv[i]) for i, j in zip(ok[:-1], ok[1:], strict=True)
                  if netseg[i] == netseg[j] and (dates[j] - dates[i]).days == lag
                  and (per == "all" or (per == "2020-2021") == (dates[i].year <= 2021))]
            if df:
                rms = float(np.sqrt(np.mean(np.square(df))))
                noise.append({"site_en": K.SITE_EN[s], "station": st, "lag_d": lag,
                              "period": per, "n": len(df), "rms_mm": rms,
                              "implied_epoch_sd_mm_upper": rms / math.sqrt(2)})

    # ---- report-only readings
    x, yb, yr = [], [], []
    d2rows_full, d2rows_emp = [], []
    emp = math.sqrt(first["site_specific_bridge_error_est_mm"] ** 2
                    + first["mean_diff_mm"] ** 2)
    for rec in sites:
        rr = ins_rows.loc[rec["site"]]
        gy = float(rr["gap_yr"])
        obs_yr = (rec["gap_days"] - rec["model_span_days"]) / 365.25
        x.append(rec["G_int_mm"] / obs_yr)
        yb.append(rr["C_primary_mm"] / gy)
        yr.append(rr["C_raw_mm"] / gy)
        if not rr["evaluable"]:
            continue
        n = int(rr["n_bridges"])
        v_reg = n * K.INSAR_BRIDGE_SD_MM ** 2
        for tgt, v_new in ((d2rows_full, max(v_reg, rec["B_mm"] ** 2)),
                           (d2rows_emp, max(v_reg, emp ** 2))):
            dv = v_new - v_reg
            tgt.append({"site": rec["site"], "evaluable": True,
                        "C_primary_mm": rr["C_primary_mm"], "C_raw_mm": rr["C_raw_mm"],
                        "C_pre_mm": rr["C_pre_mm"], "C_post_mm": rr["C_post_mm"],
                        "C_mid_mm": rr["C_mid_mm"],
                        "sigma_pre_mm": math.sqrt(rr["sigma_pre_mm"] ** 2 + dv),
                        "sigma_post_mm": math.sqrt(rr["sigma_post_mm"] ** 2 + dv),
                        "L_m": rr["L_m"]})
    d1 = C.corr_check(x, yb, yr, K.CORR_ALPHA)
    d1["williams"] = R.williams_t(d1["r_bridged"], d1["r_raw"],
                                  float(np.corrcoef(yb, yr)[0, 1]), d1["n"])
    d1["x"] = "G_int / observed span (gap minus model-set window spans)"
    d1_B = C.corr_check([rec["B_mm"] / (rec["model_span_days"] / 365.25) for rec in sites],
                        yb, yr, K.CORR_ALPHA)
    readings = {"d1_interferometric_only": d1,
                "d1_model_set_windows_only": d1_B,
                "d2_B_at_100pct": {"primary": R.family_counts(d2rows_full, "k:2"),
                                   "raw": R.family_counts(d2rows_full, "k:2", "C_raw_mm")},
                "d2_empirical_bridge_error": {"sd_mm": emp,
                                              "primary": R.family_counts(d2rows_emp, "k:2"),
                                              "raw": R.family_counts(d2rows_emp, "k:2",
                                                                     "C_raw_mm")}}

    # ---- SJES raw with the registered endpoint rule
    t, _, _ = IO.gnss_series(g, "wra", "SJES", flags=FLAGS)
    dd = g["daily"]
    x_s = dd[(dd.family == "wra") & (dd.station == "SJES") & dd["flag"].isin(list(FLAGS))]
    ts = pd.DatetimeIndex(x_s["date"])
    rs = rings["溪州國小"]
    sj = {}
    for col in ("value", "value_raw"):
        v = -1000.0 * x_s[col].to_numpy(float)
        e = []
        for c in (rs["t_lp"], rs["t_fp"]):
            ddd = np.abs((ts - pd.Timestamp(c)).days.to_numpy())
            h = K.GNSS_END_HALF_D if (ddd <= K.GNSS_END_HALF_D).sum() >= K.GNSS_END_MIN \
                else K.GNSS_END_HALF_D_FALLBACK
            e.append(float(v[ddd <= h].mean()))
        sj["corrected" if col == "value" else "raw"] = e[1] - e[0]

    out = {"sites": sites, "first_window_vs_gnss": first,
           "null_windows_sample": N.head(50).to_dict(orient="records"),
           "epoch_noise": noise, "readings": readings, "sjes_registered_rule_mm": sj,
           "params": {"SET_D": SET_D, "LONG_SEG": LONG_SEG, "NULL_SEP": NULL_SEP}}
    OUT.mkdir(parents=True, exist_ok=True)
    dump(out, OUT / "bridge_windows.json")
    pd.set_option("display.width", 250)
    print(pd.DataFrame([{k: v for k, v in z.items() if k not in ("windows",
                                                                  "per_window_B_minus_gnss_mm")}
                        for z in sites]).round(1).to_string())
    print({k: v for k, v in first.items() if k not in ("per_site_mm", "null_per_site_rms_mm")})
    print(first["per_site_mm"])
    nz = pd.DataFrame(noise)
    print(nz.groupby(["lag_d", "period"])[["rms_mm", "implied_epoch_sd_mm_upper"]]
          .median().round(1).to_string())
    for k, v in readings.items():
        if "primary" in v:
            print(k, v["primary"]["n_contain"], v["raw"]["n_contain"], v.get("sd_mm"))
        else:
            print(k, {kk: v[kk] for kk in ("n", "r_bridged", "r_raw", "r_crit", "pass")},
                  v.get("williams"))
    print("SJES", sj)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
