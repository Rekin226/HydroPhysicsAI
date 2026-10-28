"""X5 E1 (exploration, U7; never validation; cannot promote): the anchored InSAR field
2020-01..2025-12 as product ``results/dataprep/x5_insar_anchor/v1`` and its comparison with the
WRA county leveling summaries (max rate, area sinking > 2 and > 3 cm/yr).

Supersedes ``prereg_x5.doc_e1`` (left unchanged because prereg_x5.py is a hashed input of
X5.T1): after T1 failed on a fan-wide level offset, E1 adds a leveling-datum variant.

Steps (each its own process):

    flock <lock> python -I build_e1.py declare   declaration X5.post2022_exploration.v2
    flock <lock> python -I build_e1.py look      look log entry (WRA aggregates, piles, GNSS)
    python -I build_e1.py build                  product v1 (refuses to overwrite)
    python -I build_e1.py explore                report (written once)

Product (GNSS frame): insar_069A/v3 merged ml field (valid = finite, not bridged, source 1-3)
+ per-epoch offset c(t) from the T1 median polish over the T1 GNSS anchors to 2025-03-31 and,
after each pile site's last GNSS day, a pile pseudo-station G*(t) = Q_L + beta (t - t_QL) -
P(t) at TKJS, STES, SJES, NTUH (P = cumulative printed compression of the deepest sensor,
flag 0, missing months interpolated; Q = monthly GNSS + P; beta = OLS rate of Q over
2020-06..2024-12; Q_L, t_QL = mean of the last 3 GNSS months). Epochs need >= 5 stations
while GNSS exists and >= 3 anchors afterwards. May increments (May 15 +-30 d, scaled to
365 d) per pixel; 1-km cells = median of >= 5 valid pixels.

Leveling-datum variant L: the GNSS-frame increments + 15.44 mm/yr (minus the X5.T1 primary
bias, the scored 2020-21 offset of anchored InSAR against leveling); sensitivity L2: + 13.53
mm/yr (median GNSS - leveling of the T1 follow-up). Both assume the GNSS-leveling datum
offset of 2020-21 holds in later years, which no held data can check after 2021-09.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "s4_insar")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import x5const as K  # noqa: E402
import x5lib as X  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

PREREG = "X5.post2022_exploration.v2"
PROD = K.OUT / "v1"
REPORT_DIR = K.REPORTS_DIR / PREREG
REPORT = REPORT_DIR / "report.json"
INTERVALS = ((2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))
T1_REPORT = K.REPORTS_DIR / K.PREREG_T1 / "report.json"
FU_REPORT = K.REPORTS_DIR / "X5.t1_postfailure_gnss_leveling.v1" / "report.json"
COUNTY_WRA = {"changhua": "彰化", "rest": "雲林"}
GNSS_END = pd.Timestamp(K.E1_GNSS_LAST)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shifts() -> dict:
    t1 = json.loads(T1_REPORT.read_text(encoding="utf-8"))["primary"]["bias_mm"]
    fu = json.loads(FU_REPORT.read_text(encoding="utf-8"))["summary"]["gnss_minus_lev"][
        "median_mm"]
    return {"L": -float(t1), "L2": -float(fu)}


# ----------------------------------------------------------------------------- declare
def declare() -> int:
    from harness import prereg3
    NA = prereg3.not_applicable
    s = shifts()
    d = {
        "id": PREREG, "wp": K.WP, "kind": "declaration",
        "title": "X5 E1: anchored InSAR field 2020-2025 (product v1, GNSS frame and leveling-"
                 "datum variants) against the WRA county leveling summaries (exploration)",
        "tests": NA("exploration: 2023+ data never gate (U7); never validation; cannot "
                    "promote"),
        "checks": {
            "product_and_variants": __doc__.split("\n\nProduct (GNSS frame): ", 1)[1],
            "datum_shifts_mm_per_yr": s,
            "intervals": [list(i) for i in INTERVALS],
            "statistics": "per county (zone_county_map v2: changhua vs rest of the twin "
                          "domain, compared with WRA 彰化 and 雲林) and interval: max sinking "
                          "rate over cells (cm/yr), 99th percentile of sinking, area of cells "
                          "sinking > 2 and > 3 cm/yr (1 km2 per cell), cells with a value / "
                          "active cells; pixel-area alternative (pixels inside active cells)",
            "comparison": "WRA county_intervals (pre-origin v1 for 2020-21 and 2021-22, "
                          "post-origin v1 for 2022-23..2024-25): difference field - WRA per "
                          "statistic; sign agreement of the year-to-year changes",
            "loso": "per GNSS station: c(t) without it; anchored increment at the station "
                    "vs its GNSS increment for 2021-22, 2022-23, 2023-24 (May windows)",
            "labels": {"2020-21": "in-sample for the datum shift L (it was estimated there)",
                       "2021-22": "pre-origin; per-benchmark leveling ends 2021-09, the "
                                  "2022-05 survey exists only as WRA aggregates; the design "
                                  "is informed by T1, so this is exploration too",
                       "2022-25": "post-origin exploration (U7)"}},
        "zones": NA("counties"), "priors": NA("none"), "burn_in": NA("none"),
        "folds": NA("none"), "fresh_units": NA("exploration; no fresh unit"),
        "thresholds": NA("reported, no gate"), "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": [rel_to_repo(HERE / f) for f in ("build_e1.py", "x5lib.py", "x5const.py")]
        + [rel_to_repo(T1_REPORT), rel_to_repo(FU_REPORT)],
        "interpretation": "how far an anchored InSAR field reproduces the only post-2022 "
                          "leveling truth (county aggregates): context for Q6, never a skill "
                          "claim",
        "prior_knowledge_disclosure": "the post-2022 WRA county values were quoted in the "
                                      "2026-10-25 audit report read by this WP; X5.T1 and "
                                      "its follow-up are spent and set the datum shifts; no "
                                      "other choice uses a WRA value",
        "supersedes": "prereg_x5.doc_e1 (never registered)",
        "written_by": rel_to_repo(Path(__file__)),
    }
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


def look() -> int:
    from harness import prereg3
    piles = [f"pile:{s}" for v in K.PILE_SITES.values() for s in v]
    st = X.anchor_stations()
    gn = [f"gnss:{f}:{s}" for f, s in zip(st["family"], st["station"], strict=True)]
    rec = prereg3.record_look(
        ["x5.e1.wra_county_aggregates", "x5.e1.piles", "x5.e1.gnss_anchors"],
        "subsidence_exploration:2020-01..2026-01", K.WP, "exploration", prereg_id=PREREG,
        members={"x5.e1.wra_county_aggregates": {
                     "none": "county aggregates (max rate, exceedance areas); no registered "
                             "well, benchmark or cell", "description": "WRA county leveling "
                             "summaries 2020-21..2024-25 (subsidence_public_reports v1)"},
                 "x5.e1.piles": {"ids": piles, "window": "2019-04..2026-01",
                                 "description": "printed deep piles (anchors after 2025-03)"},
                 "x5.e1.gnss_anchors": {"ids": gn, "window": "2020-01..2025-03",
                                        "description": "T1 GNSS anchor set, full period"}},
        note="E1 exploration: product v1 and county comparison (never validation)")
    print(rec["time"], rec["purpose"])
    return 0


# ----------------------------------------------------------------------------- build
def pile_pseudo(st: pd.DataFrame, series: dict) -> tuple[dict, list]:
    pm = pd.read_parquet(K.PILES, columns=["sensor_id", "month", "value", "flag"])
    pm = pm[pm["flag"] == 0]
    w0, w1 = (pd.Period(m, "M") for m in K.PILE_RATE_WINDOW)
    out, notes = {}, []
    for site, sensors in K.PILE_SITES.items():
        if site not in series:
            notes.append({"site": site, "used": False, "why": "no GNSS series"})
            continue
        for sen in sensors:
            p = pm[pm["sensor_id"] == sen].copy()
            p["per"] = pd.PeriodIndex(p["month"], freq="M")
            p = p.set_index("per")["value"].sort_index()
            inwin = p[(p.index >= w0) & (p.index <= w1)]
            if len(inwin) >= 24:
                break
        else:
            notes.append({"site": site, "used": False, "why": "pile record too short"})
            continue
        full = pd.period_range(p.index.min(), p.index.max(), freq="M")
        mon = p.reindex(full).interpolate(limit_area="inside")
        n_interp = int(p.reindex(full).isna().sum())
        cum_end = mon.cumsum()                                 # compression at month end
        ends = full.to_timestamp(how="end").normalize()
        P_day = pd.Series(np.r_[0.0, cum_end.to_numpy()],
                          index=pd.DatetimeIndex(np.r_[full[0].to_timestamp(how="start")
                                                       .to_datetime64(), ends.values]))
        g = series[site].dropna()
        gm = g.resample("MS").mean().dropna()
        gm.index = gm.index + pd.Timedelta(days=14)
        days = pd.date_range(P_day.index.min(), P_day.index.max(), freq="D")
        Pd = P_day.reindex(P_day.index.union(days)).interpolate("time").reindex(days)
        Q = (gm + Pd.reindex(gm.index)).dropna()
        qw = Q[(Q.index >= w0.to_timestamp()) & (Q.index <= w1.to_timestamp(how="end"))]
        if len(qw) < 24:
            notes.append({"site": site, "used": False, "why": "Q window too short"})
            continue
        tq = (qw.index - qw.index[0]).days.to_numpy(float)
        beta = float(np.polyfit(tq, qw.to_numpy(), 1)[0])       # mm/day
        tL = g.index.max()
        last = Q[Q.index <= tL].iloc[-K.PILE_Q_LAST_MONTHS:]
        QL = float(last.mean())
        tQL = last.index.mean()
        after = days[(days > tL) & (days <= pd.Timestamp("2025-12-31"))]
        ext = QL + beta * (after - tQL).days.to_numpy(float) - Pd.reindex(after).to_numpy()
        ext = pd.Series(ext, index=after).dropna()
        out[site] = pd.concat([series[site], ext]).sort_index()
        notes.append({"site": site, "used": True, "sensor": sen, "beta_mm_per_yr":
                      beta * 365.25, "gnss_last_day": str(tL.date()),
                      "pile_last_month": str(p.index.max()), "months_interpolated": n_interp,
                      "extension_days": int(len(ext))})
    return out, notes


def anchor(ins: dict, st: pd.DataFrame, series: dict, gnss_end_by_station: dict,
           drop: str | None = None) -> dict:
    keep = st if drop is None else st[st["station"] != drop]
    S = X.station_inputs(ins, keep, series)
    c, k, n = X.median_polish(S["G"] - S["I"], min_st=K.E1_MIN_STATIONS_PILE)
    dates = ins["dates"]
    n_gnss = np.zeros(len(dates), int)
    for j, s in enumerate(S["names"]):
        end = gnss_end_by_station.get(s)
        if end is not None:
            n_gnss += (np.isfinite(S["G"][:, j] - S["I"][:, j])
                       & np.asarray(dates <= end)).astype(int)
    in_gnss = np.asarray(dates <= GNSS_END)
    c = np.where(in_gnss & (n < K.MIN_STATIONS_EPOCH), np.nan, c)
    ref = int(np.flatnonzero(np.isfinite(c))[0])
    return {"c": c - c[ref], "n": n, "n_gnss": n_gnss, "ref": ref, "S": S}


def build() -> int:
    if PROD.exists():
        raise SystemExit(f"{PROD} exists; products are never overwritten")
    v3 = X.load_v3()
    st = X.anchor_stations()
    series = X.gnss_daily(st)
    gend = {s: v.dropna().index.max() for s, v in series.items()}
    ext, notes = pile_pseudo(st, series)
    series_ext = dict(series)
    series_ext.update(ext)
    an = anchor(v3, st, series_ext, gend)
    c = an["c"]
    dates = v3["dates"]
    T, L, W = v3["value"].shape
    PROD.mkdir(parents=True)
    pd.DataFrame({"date": [str(d.date()) for d in dates], "c_mm": c, "n_anchors": an["n"],
                  "n_gnss_anchors": an["n_gnss"],
                  "anchor_kind": np.where(np.asarray(dates <= GNSS_END), "gnss",
                                          "piles")}).to_csv(PROD / "anchor_offsets.csv",
                                                            index=False)
    inc = np.full((len(INTERVALS), L, W), np.nan, np.float32)
    nwin = np.zeros((len(INTERVALS), 2, L, W), np.int8)
    span = np.full((len(INTERVALS), L, W), np.nan, np.float32)
    for q, (y0, y1) in enumerate(INTERVALS):
        sel = X.may_window(dates, y0) | X.may_window(dates, y1)
        idx = np.flatnonzero(sel)
        A = np.where(v3["valid"][idx], v3["value"][idx], np.nan).reshape(len(idx), -1)
        A = A + c[idx][:, None]
        r = X.may_increment(A, dates[idx], y0, y1, scaled=True)
        inc[q] = r["inc"].reshape(L, W)
        span[q] = r["span_days"].reshape(L, W)
        nwin[q, 0] = r["n0"].reshape(L, W)
        nwin[q, 1] = r["n1"].reshape(L, W)
    g = {"cell": v3["cell"]}
    A_n = int(np.load(K.COUNTY_MAP)["n_active"])
    cell_inc = np.full((len(INTERVALS), A_n), np.nan, np.float32)
    cell_n = np.zeros((len(INTERVALS), A_n), int)
    for q in range(len(INTERVALS)):
        v, _, nv = Lb.to_cells(inc[q], np.isfinite(inc[q]), g, A_n)
        cell_inc[q], cell_n[q] = v, nv
    np.savez_compressed(
        PROD / "may_increments.npz", inc_ml=inc, n_epochs_ml=nwin, span_days_ml=span,
        inc_cells=cell_inc, n_pixels_cells=cell_n,
        intervals=np.array([f"{a}-05..{b}-05" for a, b in INTERVALS]),
        x=v3["x"], y=v3["y"], cell=v3["cell"],
        units=np.array("mm per 365 d, + = up (sinking negative), GNSS frame; the leveling-"
                       "datum variants add the shifts in PRODUCT_CARD.md"),
        label=np.array("EXPLORATION (X5.post2022_exploration.v2): never validation, never a "
                       "scoring target, cannot promote"))
    s = shifts()
    card = f"""# x5_insar_anchor/v1 (WP X5, 2026-10-25): EXPLORATION product

**Label: exploration (U7). Never validation, never a scoring target or calibration target, cannot
promote.** Declaration: `{PREREG}`.

## What it is
May-to-May vertical increments (mm per 365 d, + up) of insar_069A/v3 (merged ml field, valid =
finite, not bridged, source 1-3) anchored per epoch to GNSS (gnss_clean/full/v4, T1 anchor set,
to 2025-03-31) and, after each pile site's last GNSS day, to pile pseudo-stations (TKJS, STES,
SJES, NTUH). Intervals: {', '.join(f'{a}-05..{b}-05' for a, b in INTERVALS)}.
Cells: median of >= 5 valid pixels in each active twin cell.

## Frames
- `inc_*` are in the **GNSS frame**.
- Leveling datum: add **{s['L']:.2f} mm/yr** (minus the X5.T1 primary bias, 2020-21) or, as a
  sensitivity, {s['L2']:.2f} mm/yr (T1 follow-up median GNSS - leveling). Whether this offset
  holds after 2021 cannot be checked with held data.

## Why the label
X5.T1 (anchored InSAR alone vs the 2020-05 -> 2021-05 leveling, 709 benchmarks) FAILED: slope
0.906 (in band) but RMSE 17.5 mm (> 10) from a -15.4 mm fan-wide bias. The follow-up puts most
of it between GNSS and leveling (median -13.5 mm at 23 stations), not in the InSAR anchoring.
After 2023 the GNSS anchors thin (2023 outage, drifting stations) and after 2025-03 only four
pile sites anchor the level.

## Files
anchor_offsets.csv (c(t), anchors per epoch, kind), may_increments.npz (inc_ml, n_epochs_ml,
span_days_ml, inc_cells, n_pixels_cells, grid), pile_anchors.json, MANIFEST.json.
"""
    (PROD / "PRODUCT_CARD.md").write_text(card, encoding="utf-8")
    (PROD / "pile_anchors.json").write_text(json.dumps(notes, indent=1, default=str) + "\n",
                                            encoding="utf-8")
    man = {"product": "x5_insar_anchor", "version": "v1", "created": "2026-10-25",
           "label": "exploration", "declaration": PREREG,
           "allowed_uses": ["after_the_fact_analysis"],
           "forbidden_uses": ["scoring_target", "calibration_target", "validator", "prior",
                              "selection"],
           "inputs": {rel_to_repo(p): sha(p) for p in (K.INSAR_ML, K.GNSS_WRA, K.GNSS_NGL,
                                                       K.PILES, K.COUNTY_MAP, T1_REPORT,
                                                       FU_REPORT)},
           "code": {rel_to_repo(HERE / f): sha(HERE / f)
                    for f in ("build_e1.py", "x5lib.py", "x5const.py")},
           "outputs": {f.name: sha(f) for f in sorted(PROD.iterdir())}}
    (PROD / "MANIFEST.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    print(json.dumps({"epochs": T, "anchored_epochs": int(np.isfinite(c).sum()),
                      "piles": notes, "cells_with_value": cell_n.astype(bool).sum(1).tolist()},
                     indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------- explore
def county_stats(cell_inc: np.ndarray, ch: np.ndarray, shift: float) -> dict:
    out = {}
    for cname, m in (("changhua", ch), ("rest", ~ch)):
        v = cell_inc[m] + shift
        f = v[np.isfinite(v)]
        sink = -f / 10.0                                        # cm/yr, + = sinking
        out[cname] = {"n_cells": int(m.sum()), "n_with_value": int(len(f)),
                      "max_sinking_cm_yr": float(sink.max()) if len(f) else None,
                      "p99_sinking_cm_yr": float(np.percentile(sink, 99)) if len(f) else None,
                      "area_gt2_km2": float((sink > K.E1_THRESH_CM[0]).sum()
                                            * K.CELL_AREA_KM2),
                      "area_gt3_km2": float((sink > K.E1_THRESH_CM[1]).sum()
                                            * K.CELL_AREA_KM2),
                      "median_sinking_cm_yr": float(np.median(sink)) if len(f) else None}
    return out


def wra_table() -> pd.DataFrame:
    rows = []
    for sub in ("preorigin", "postorigin"):
        t = pd.read_csv(K.PUBLIC_POST.parent.parent / sub / "v1" / "county_intervals.csv",
                        usecols=["county", "start_ym", "end_ym", "max_rate_cm_yr",
                                 "area_gt2_km2", "area_gt3_km2"])
        rows.append(t)
    t = pd.concat(rows)
    return t[t["start_ym"].isin([f"{a}-05" for a, _ in INTERVALS])
             & t["end_ym"].isin([f"{b}-05" for _, b in INTERVALS])]


def loso(v3, st, series, gend) -> pd.DataFrame:
    rows = []
    base = anchor(v3, st, series, gend)
    S = base["S"]
    for j, s in enumerate(S["names"]):
        an = anchor(v3, st, series, gend, drop=s)
        ins_j = S["I"][:, j] + an["c"]
        for y0, y1 in INTERVALS[1:4]:
            ri = X.may_increment(ins_j[:, None], v3["dates"], y0, y1)["inc"][0]
            gs = series.get(s)
            if gs is None:
                continue
            G = X.at_epochs({s: gs}, [s], v3["dates"])
            rg = X.may_increment(G, v3["dates"], y0, y1)["inc"][0]
            rows.append({"station": s, "interval": f"{y0}-{y1}", "insar_anchored_mm": ri,
                         "gnss_mm": rg, "diff_mm": ri - rg})
    return pd.DataFrame(rows)


def explore() -> int:
    if REPORT.exists():
        raise SystemExit(f"{REPORT} exists; written once")
    z = np.load(PROD / "may_increments.npz")
    cm = np.load(K.COUNTY_MAP)
    ch = cm["changhua"].astype(bool)
    s = shifts()
    wra = wra_table()
    res = {"prereg": PREREG, "label": "exploration (U7); never validation; cannot promote",
           "product": rel_to_repo(PROD), "datum_shifts_mm_per_yr": s, "intervals": {}}
    for q, iv in enumerate(z["intervals"]):
        y0, y1 = (int(t[:4]) for t in iv.split(".."))
        ent = {"field": {fr: county_stats(z["inc_cells"][q], ch, sh)
                         for fr, sh in (("G", 0.0), ("L", s["L"]), ("L2", s["L2"]))},
               "wra": {}}
        for cn, wn in COUNTY_WRA.items():
            w = wra[(wra["county"] == wn) & (wra["start_ym"] == f"{y0}-05")
                    & (wra["end_ym"] == f"{y1}-05")]
            if len(w):
                r0 = w.iloc[0]
                ent["wra"][cn] = {"county": wn, "max_rate_cm_yr": float(r0["max_rate_cm_yr"]),
                                  "area_gt2_km2": (None if pd.isna(r0["area_gt2_km2"])
                                                   else float(r0["area_gt2_km2"])),
                                  "area_gt3_km2": (None if pd.isna(r0["area_gt3_km2"])
                                                   else float(r0["area_gt3_km2"]))}
        ent["label"] = ("in-sample for the datum shift" if y0 == 2020 else
                        "pre-origin exploration" if y0 == 2021 else "post-origin exploration")
        res["intervals"][f"{y0}-{y1}"] = ent
    # direction agreement of year-to-year changes (field L vs WRA)
    keys = list(res["intervals"])
    agree = {}
    for cn in COUNTY_WRA:
        for stat, fk in (("max_rate_cm_yr", "max_sinking_cm_yr"),
                         ("area_gt2_km2", "area_gt2_km2"), ("area_gt3_km2", "area_gt3_km2")):
            seq = []
            for a, b in zip(keys[:-1], keys[1:], strict=True):
                wa = res["intervals"][a]["wra"].get(cn, {}).get(stat)
                wb = res["intervals"][b]["wra"].get(cn, {}).get(stat)
                fa = res["intervals"][a]["field"]["L"][cn][fk]
                fb = res["intervals"][b]["field"]["L"][cn][fk]
                if None in (wa, wb, fa, fb):
                    continue
                seq.append({"from": a, "to": b, "wra_change": wb - wa, "field_L_change": fb - fa,
                            "same_sign": bool(np.sign(wb - wa) == np.sign(fb - fa))})
            agree[f"{cn}:{stat}"] = seq
    res["direction_agreement_field_L_vs_wra"] = agree
    v3 = X.load_v3()
    st = X.anchor_stations()
    series = X.gnss_daily(st)
    gend = {k: v.dropna().index.max() for k, v in series.items()}
    lo = loso(v3, st, series, gend)
    lo.to_csv(K.OUT / "e1_loso.csv", index=False)
    res["loso"] = {iv: {"n": int(g["diff_mm"].notna().sum()),
                        "median_diff_mm": float(g["diff_mm"].median()),
                        "rmse_mm": float(np.sqrt(np.nanmean(g["diff_mm"] ** 2))),
                        "mad_sd_mm": float(1.4826 * np.nanmedian(np.abs(
                            g["diff_mm"] - g["diff_mm"].median())))}
                   for iv, g in lo.groupby("interval")}
    res["loso_csv"] = rel_to_repo(K.OUT / "e1_loso.csv")
    res["code"] = {rel_to_repo(HERE / f): sha(HERE / f) for f in ("build_e1.py", "x5lib.py",
                                                                  "x5const.py")}
    res["product_manifest_sha256"] = sha(PROD / "MANIFEST.json")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float)[:6000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["declare", "look", "build", "explore"])
    a = ap.parse_args(argv)
    return {"declare": declare, "look": look, "build": build, "explore": explore}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
