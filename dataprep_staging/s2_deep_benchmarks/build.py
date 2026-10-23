"""Build the deep-benchmark product ``results/dataprep/deep_benchmarks/v<N>/`` (WP S2).

Steps (rules fixed in results/dataprep/preregistration/S2.deep_benchmark_acceptance.v1.json):

1. load both feeds (vendor cache, raw agency dumps) and the verified printed values;
2. vendor-to-sensor identity (offset or sign-inverted sum constant on common hours);
3. pre-origin envelope of natural changes from artificial gaps (vendor feed, 2022-12 or
   earlier) and the recall of injected steps;
4. step catalogue of every raw and vendor series; raw steps removed, spikes dropped;
5. daily means, month-end levels and monthly increments per feed; primary feed per month;
6. printed values merged (primary where printed and verified; feed elsewhere);
7. cumulative compression per sensor by weighted least squares (printed monthly and
   trailing-twelve-month values, feed increments), referenced to the end of 2022-12;
8. acceptance A (printed against feed <= 2 mm) and B (mapping correlation >= 0.85), and the
   secondary diagnostics; manifest.

Usage: python -I build.py [--scratch DIR]   (with --scratch: compute and write the tables to
DIR without creating a product version or a manifest)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import flags as F  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, REPO, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import c_test as T  # noqa: E402
from s2_deep_benchmarks import s2lib as L  # noqa: E402

PRODUCT = "deep_benchmarks"
PREREG_V1 = PREREG_DIR / "S2.deep_benchmark_acceptance.v1.json"
PREREG = PREREG_DIR / "S2r2.deep_benchmarks_core.json"
PREREG_C_ID = "S2r2.c_fresh_cy007001"
PREREG_C = PREREG_DIR / f"{PREREG_C_ID}.json"
PRINTED_CY = REPO / "results" / "dataprep" / "deep_benchmarks_printed" / "v1"
ENVELOPE_KEYS = ("TKJS_281_29", "TKJS_282_29", "NTUH_281_29", "NTUH_282_29", "SJES", "STES",
                 "WNES")
#: overlap tolerance when matching events of two feeds
TOL_1H = pd.Timedelta(hours=1)
CODE = [Path(__file__), Path(L.__file__), Path(__file__).with_name("prereg.py"),
        Path(__file__).with_name("archive_opendata.py"),
        Path(__file__).with_name("probe_reports.py"), Path(__file__).with_name("prereg_r2.py"),
        Path(T.__file__), Path(__file__).with_name("cy007001.py"),
        Path(__file__).with_name("fetch_007001.py"),
        Path(__file__).with_name("transcribe_007001.py"),
        Path(__file__).with_name("prereg_c.py")]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- sensors
def sensor_registry(raw_names: dict, ident: pd.DataFrame) -> pd.DataFrame:
    sites = pd.read_csv(L.SITES_CSV)
    rows = []
    for rid, name in sorted(raw_names.items()):
        sid = rid.replace("BM_", "")
        pile = str(name).split("(")[0]
        srow = sites[sites.identifier == rid]
        if len(srow):
            depth = int(srow.depth_m.iloc[0])
            x, y = float(srow.x_twd97.iloc[0]), float(srow.y_twd97.iloc[0])
            dsrc = "deep_pile_sites.csv (station metadata)"
        else:
            depth = next((k[1] for k, v in L.PRINTED_TO_SENSOR.items() if v == sid), None)
            x = y = float("nan")
            dsrc = "printed table (not in the station metadata)"
        cell, row, col = L.grid_cell((x, y)) if np.isfinite(x) else (-1, -1, -1)
        vk = ident[(ident.raw_id == rid) & ident.chosen]
        rows.append({"sensor_id": sid, "raw_id": rid, "pile": pile, "depth_m": depth,
                     "depth_source": dsrc, "x_twd97": x, "y_twd97": y,
                     "grid_cell": cell, "grid_row": row, "grid_col": col,
                     "on_grid": bool(cell >= 0),
                     "vendor_key": vk.vendor_key.iloc[0] if len(vk) else ""})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- helpers
def overlaps(t0a, t1a, t0b, t1b, tol=TOL_1H) -> bool:
    return (pd.Timestamp(t0a) <= pd.Timestamp(t1b) + tol) and \
           (pd.Timestamp(t0b) <= pd.Timestamp(t1a) + tol)


def classify_stitches(stitches: list[dict], raw_ev: pd.DataFrame) -> list[dict]:
    """Vendor stitch -> agrees with a raw step / removed movement the raw catalogue keeps."""
    out = []
    for s in stitches:
        m = raw_ev[[overlaps(s["t_before"], s["t_after"], a, b)
                    for a, b in zip(raw_ev.t_before, raw_ev.t_after, strict=True)]] \
            if len(raw_ev) else raw_ev
        kinds = sorted(set(m.kind)) if len(m) else []
        if s["kind"] == "sign_change":
            cls = "sign_inversion"
        elif "step" in kinds:
            cls = "agrees_raw_step"
        elif "spike" in kinds:
            cls = "raw_spike"
        else:
            cls = "removed_real_movement"
        out.append({**s, "raw_kinds": ";".join(kinds), "classification": cls})
    return out


def daily_frame(sensor: dict, feed: str, d: pd.DataFrame, source_id: str, method: str,
                version: str) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    d.insert(0, "sensor_id", sensor["sensor_id"])
    d.insert(1, "pile", sensor["pile"])
    d.insert(2, "depth_m", sensor["depth_m"])
    d.insert(3, "feed", feed)
    d["month"] = d["date"].dt.strftime("%Y-%m")
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    d["method"] = np.where(d["flag"] == 2, method + "; partial day", method)
    d["product_version"] = version
    d["source_ids"] = source_id
    return d


# ------------------------------------------------------------------------------- compute
def compute(version: str) -> dict:
    t_start = dt.datetime.now()
    # 1. feeds
    vendor_samples = {k: L.vendor_raw(k) for k in L.VENDOR_KEYS}
    vendor_h = {k: L.to_hourly(s) for k, s in vendor_samples.items()}
    raw_h, raw_info = L.load_open([L.OPEN_DIR / f for f in L.OPEN_FILES])
    log("raw sensors", sorted(raw_h), raw_info["rows"], "rows")
    # 2. identity
    irows, ident_full = [], {}
    for vk in L.VENDOR_KEYS:
        for rid, rh in raw_h.items():
            out = L.identity(vendor_h[vk], rh)
            ident_full[(vk, rid)] = out
            irows.append({"vendor_key": vk, "raw_id": rid, "n_common_hours": out["n_common_hours"],
                          "n_informative_hours": out.get("n_informative_hours", 0),
                          "share_same": out.get("share_same"),
                          "share_inverted": out.get("share_inverted"),
                          "share_union": out.get("share_union"),
                          "n_segments": len(out.get("segments", [])),
                          "n_stitches": len(out.get("stitches", []))})
    ident = pd.DataFrame(irows)
    # prereg: "share of informative hours with an unchanged offset or sum >= 0.98", i.e. the
    # union of the two conventions (round 1 used the larger of the two, which would have
    # dropped the sign-inverting 灣內 series: 0.64 same + 0.36 inverted)
    ident["best_share"] = ident["share_union"].astype(float)
    ident["chosen"] = False
    for _vk, g in ident.groupby("vendor_key"):
        g = g[g.n_common_hours >= 48].sort_values("best_share", ascending=False)
        if len(g) and g.best_share.iloc[0] >= 0.98:
            ident.loc[g.index[0], "chosen"] = True
    dup = ident[ident.chosen].raw_id.duplicated(keep=False)
    if dup.any():
        raise RuntimeError(f"one raw sensor identified with several vendor series: "
                           f"{ident[ident.chosen][dup].to_dict('records')}")
    log(ident[ident.chosen].to_string())
    sensors = sensor_registry(raw_info["names"], ident)
    vendor_of = dict(zip(sensors.sensor_id, sensors.vendor_key, strict=True))
    unmapped_vendor = [k for k in L.VENDOR_KEYS if k not in set(sensors.vendor_key)]
    log("unmapped vendor series:", unmapped_vendor)
    # 3. envelope
    samples = []
    for k in ENVELOPE_KEYS:
        h = vendor_h[k].loc[:L.ORIGIN_END]
        s = L.artificial_gaps(h.to_numpy(), n_per=3000, seed=2026)
        s["sensor"] = k
        samples.append(s)
    samples = pd.concat(samples, ignore_index=True)
    env = L.envelope_table(samples)
    env["chosen"] = [L.best_estimator(env, d) == e for d, e in zip(env.d_h, env.estimator,
                                                                   strict=True)]
    recall = L.injected_recall(samples, env)
    det = L.Detector(env)
    # 4-5. per sensor: catalogue, daily, monthly
    daily_rows, monthly_rows, step_rows, stitch_rows, sig = [], [], [], [], {}
    per_sensor = {}
    for srow in sensors.to_dict("records"):
        sid, rid, vk = srow["sensor_id"], srow["raw_id"], srow["vendor_key"]
        rec = {"sensor": srow}
        # sign segments of the raw feed relative to the vendor
        segs, stitches = [], []
        if vk:
            idt = ident_full[(vk, rid)]
            segs, stitches = idt["segments"], idt["stitches"]
        rh = raw_h[rid]
        # prereg: an inversion runs to the next same-sign segment or the record end, so every
        # segment (both signs) sets the sign from its start (round 1 passed only the -1
        # segments, which could never end an inversion)
        sgn = L.sign_series(rh.index, segs) if segs else np.ones(len(rh))
        rh_s = rh * sgn
        ev_r, xc_r, hf_r = L.detect(rh_s, det)
        ev_r.insert(0, "feed", "raw")
        ev_r.insert(0, "sensor_id", sid)
        rec["raw_events"] = ev_r
        rec["sign_inverted_hours"] = int((sgn < 0).sum())
        # vendor
        if vk:
            vh = vendor_h[vk]
            ev_v, xv_c, hf_v = L.detect(vh, det)
            # the vendor is kept as delivered: drop only removed hours, never apply offsets
            xv = vh.where(hf_v != 9)
            ev_v.insert(0, "feed", "vendor")
            ev_v.insert(0, "sensor_id", sid)
            rec["vendor_events"] = ev_v
        st = classify_stitches(stitches, ev_r)
        for s in st:
            stitch_rows.append({"sensor_id": sid, "vendor_key": vk, **s})
        # sigma_day: vendor pre-origin (longest record), else raw pre-origin
        sd_candidates = []
        if vk:
            dv0 = L.daily_from_hourly(xv, hf_v.where(hf_v != 9, 0))
            sd_candidates.append(L.sigma_day_preorigin(dv0))
        dr0 = L.daily_from_hourly(xc_r, hf_r)
        sd_candidates.append(L.sigma_day_preorigin(dr0))
        sig[sid] = next((s for s in sd_candidates if np.isfinite(s)), float("nan"))
        rec["xc_r"], rec["hf_r"], rec["rh"] = xc_r, hf_r, rh
        if vk:
            rec["xv"], rec["hf_v"] = xv, hf_v
        per_sensor[sid] = rec
    pooled = float(np.nanmedian([v for v in sig.values() if np.isfinite(v)]))
    sig_src = {k: ("pre-origin" if np.isfinite(v) else "pooled median") for k, v in sig.items()}
    sig = {k: (v if np.isfinite(v) else pooled) for k, v in sig.items()}
    log("sigma_day (mm):", {k: round(v, 3) for k, v in sig.items()})
    for sid, rec in per_sensor.items():
        srow, vk = rec["sensor"], rec["sensor"]["vendor_key"]
        dr = L.daily_from_hourly(rec["xc_r"], rec["hf_r"], raw=rec["rh"], sigma_day=sig[sid])
        daily_rows.append(daily_frame(
            srow, "raw", dr, f"raw:{srow['raw_id']}",
            "raw agency feed, step catalogue applied (removed steps -> flag 1)", version))
        ev_r = rec["raw_events"]
        mr = L.monthly_from_daily(dr, ev_r.assign(event_id=[f"{sid}:raw:{i}" for i in
                                                            range(len(ev_r))]),
                                  det, sig[sid])
        mr["feed"] = "raw"
        mv = None
        if vk:
            dvd = L.daily_from_hourly(rec["xv"], rec["hf_v"].where(rec["hf_v"] != 9, 0),
                                      raw=rec["xv"], sigma_day=sig[sid])
            daily_rows.append(daily_frame(srow, "vendor", dvd, f"vendor:{vk}",
                                          "vendor feed as delivered (zeros = missing)", version))
            ev_v = rec["vendor_events"]
            sus = [ev_v[ev_v.kind == "step"][["t_before", "t_after"]]]
            st = [s for s in stitch_rows if s["sensor_id"] == sid
                  and s["classification"] == "removed_real_movement"]
            if st:
                sus.append(pd.DataFrame(st)[["t_before", "t_after"]])
            rs = ev_r[ev_r.kind == "step"]
            kept = [r for r in rs.itertuples()
                    if not any(overlaps(r.t_before, r.t_after, s["t_before"], s["t_after"])
                               for s in stitch_rows if s["sensor_id"] == sid)]
            vcover = rec["xv"].dropna()
            kept = [r for r in kept if len(vcover) and vcover.index.min() < r.t_before
                    and vcover.index.max() > r.t_after]
            if kept:
                sus.append(pd.DataFrame({"t_before": [r.t_before for r in kept],
                                         "t_after": [r.t_after for r in kept]}))
            sus = pd.concat(sus, ignore_index=True)
            mv = L.monthly_from_daily(dvd, pd.DataFrame(columns=["kind"]), det, sig[sid],
                                      suspect_events=sus)
            mv["feed"] = "vendor"
            rec["vendor_suspects"] = sus
        rec["mr"], rec["mv"] = mr, mv
    # vendor series without identity (defective): kept, every row flag 9
    defective = {}
    for vk in unmapped_vendor:
        s = vendor_samples[vk].dropna()
        g = s.groupby(s.index.floor("D"))
        d = pd.DataFrame({"date": g.mean().index, "value": g.mean().to_numpy(),
                          "n_hours": g.size().to_numpy(), "flag": 9})
        d["raw_value"] = d["value"]
        d["sd"] = np.nan
        nominal = {"TKJS_283_29": "TKJS(3)"}.get(vk, vk)
        srow = sensors[sensors.sensor_id == nominal].to_dict("records")
        srow = srow[0] if srow else {"sensor_id": nominal, "pile": "", "depth_m": -1}
        defective[vk] = nominal
        daily_rows.append(daily_frame(
            srow, "vendor", d, f"vendor:{vk}",
            "vendor series without identity to any raw sensor: defective (flag 9)", version))
    # primary monthly
    for sid, rec in per_sensor.items():
        srow = rec["sensor"]
        mr = rec["mr"].set_index("month")
        mv = rec["mv"].set_index("month") if rec["mv"] is not None else None
        months = sorted(set(mr.index) | (set(mv.index) if mv is not None else set()))
        for m in months:
            r = mr.loc[m] if m in mr.index else None
            v = mv.loc[m] if (mv is not None and m in mv.index) else None
            pick, feed = None, ""
            if r is not None and r["flag"] in (0, 1, 2):
                pick, feed = r, "raw"
            elif v is not None and v["flag"] in (0, 1, 2):
                pick, feed = v, "vendor"
            elif r is not None and r["flag"] == 9:
                pick, feed = r, "raw"
            elif v is not None and v["flag"] == 9:
                pick, feed = v, "vendor"
            row = {"sensor_id": sid, "pile": srow["pile"], "depth_m": srow["depth_m"],
                   "month": m, "feed": feed,
                   "value": float(pick["inc_cm"]) if pick is not None else np.nan,
                   "sd": float(pick["sd_cm"]) if pick is not None else np.nan,
                   "flag": int(pick["flag"]) if pick is not None else 8,
                   "n_steps": int(pick["n_steps"]) if pick is not None else 0,
                   "step_ids": pick["step_ids"] if pick is not None else "",
                   "raw_cm": float(r["inc_cm"]) if r is not None else np.nan,
                   "raw_flag": int(r["flag"]) if r is not None else 8,
                   "vendor_cm": float(v["inc_cm"]) if v is not None else np.nan,
                   "vendor_flag": int(v["flag"]) if v is not None else 8,
                   "level_end_mm_raw": float(r["level_end_mm"]) if r is not None else np.nan,
                   "level_end_mm_vendor": float(v["level_end_mm"]) if v is not None else np.nan,
                   "mean_level_mm": float(pick["mean_level_mm"]) if pick is not None else np.nan,
                   "n_days": int(pick["n_days"]) if pick is not None else 0}
            meth = {"raw": "raw feed, catalogued", "vendor": "vendor feed as delivered",
                    "": "no feed increment"}[feed]
            if row["flag"] == 1:
                meth += "; removed step inside the month"
            elif row["flag"] == 2:
                meth += "; month end bridged from an earlier day or a partial day"
            elif row["flag"] == 9:
                meth += "; suspect: a stitch or step inside the month"
            if np.isfinite(row["raw_cm"]) and np.isfinite(row["vendor_cm"]) and \
                    abs(row["raw_cm"] - row["vendor_cm"]) > 0.2:
                meth += "; raw and vendor differ by more than 2 mm"
            row["method"] = meth
            row["product_version"] = version
            row["source_ids"] = {"raw": f"raw:{srow['raw_id']}",
                                 "vendor": f"vendor:{srow['vendor_key']}", "": ""}[feed]
            monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    # steps catalogue
    for sid, rec in per_sensor.items():
        for key in ("raw_events", "vendor_events"):
            if key in rec and len(rec[key]):
                e = rec[key].copy()
                e["event_id"] = [f"{sid}:{e.feed.iloc[0]}:{i}" for i in range(len(e))]
                step_rows.append(e[e.kind != "noise"])
    steps = pd.concat(step_rows, ignore_index=True)
    stitches = pd.DataFrame(stitch_rows)
    steps = annotate_steps(steps, stitches)
    # 6. printed merge
    printed = L.read_printed()
    last_month = str(printed.loc[printed['verified'].astype(bool), 'month'].max())
    merged = merge_printed(monthly, printed, sensors, sig, version, last_month)
    # 7. cumulative
    cum = cumulative_all(merged, printed, sig, version)
    # 8. acceptance
    val = acceptance(merged, monthly, printed, sensors, per_sensor, sig)
    val["identity"] = ident[ident.chosen][["vendor_key", "raw_id", "n_common_hours",
                                           "share_same", "share_inverted"]].to_dict("records")
    val["unmapped_vendor_series"] = unmapped_vendor
    val["sigma_day_mm"] = {k: {"value": round(v, 4), "source": sig_src[k]} for k, v in sig.items()}
    val["raw_feed_info"] = raw_info
    val["envelope"] = env.to_dict("records")
    val["recall"] = recall_summary(recall)
    val["steps_summary"] = steps_summary(steps, stitches)
    val["runtime_s"] = (dt.datetime.now() - t_start).total_seconds()
    mapping = mapping_table(sensors, merged, monthly, printed, ident, vendor_of, defective)
    val["last_month"] = last_month
    cy = test_c(daily, monthly, printed, version)
    val["C"] = cy["validation"]
    merged = merged.merge(cy["printed_cols"], on=["sensor_id", "month"], how="left")
    return {"daily": daily, "monthly": monthly, "merged": merged, "cum": cum, "steps": steps,
            "stitches": stitches, "identity": ident, "envelope": env, "recall": recall,
            "mapping": mapping, "validation": val, "sensors": sensors, "last_month": last_month,
            "cy_units": cy["units"], "mm_inc": cy["mm_inc"]}


def annotate_steps(steps: pd.DataFrame, stitches: pd.DataFrame) -> pd.DataFrame:
    s = steps.copy()
    s["t_before"] = pd.to_datetime(s["t_before"])
    s["t_after"] = pd.to_datetime(s["t_after"])
    s["month"] = s["t_after"].dt.strftime("%Y-%m")
    tags = []
    for r in s.itertuples():
        tag = ""
        if r.kind == "step" and pd.Timestamp("2024-02-28") <= r.t_after <= pd.Timestamp("2024-03-07"):
            tag = "plan: step at every pile 2024-02-29/03-01"
        if r.kind == "step" and r.sensor_id == "SJES" and r.month == "2025-03":
            tag = "plan: 溪州 reset March 2025"
        tags.append(tag)
    s["plan_item"] = tags
    # vendor treatment of raw events
    vt = []
    for r in s.itertuples():
        if r.feed != "raw" or not len(stitches):
            vt.append("")
            continue
        m = stitches[(stitches.sensor_id == r.sensor_id)
                     & np.array([overlaps(a, b, r.t_before, r.t_after)
                                 for a, b in zip(stitches.t_before, stitches.t_after,
                                                 strict=True)])]
        vt.append(";".join(sorted(set(m.classification))) if len(m) else "not stitched")
    s["vendor_treatment"] = vt
    s["value"] = s["correction_mm"].astype(float)
    s["sd"] = s["sd_mm"].astype(float)
    s["flag"] = np.where(s["kind"] == "step", 1, np.where(s["kind"] == "spike", 9, 0))
    s["method"] = "step catalogue (pre-origin envelope; prereg S2.deep_benchmark_acceptance.v1)"
    s["product_version"] = "deep_benchmarks/v1"
    s["source_ids"] = s["feed"] + ":" + s["sensor_id"]
    s["t_before"] = s["t_before"].dt.strftime("%Y-%m-%d %H:%M")
    s["t_after"] = s["t_after"].dt.strftime("%Y-%m-%d %H:%M")
    return s.drop(columns=[c for c in ("a", "b") if c in s.columns])


def merge_printed(monthly: pd.DataFrame, printed: pd.DataFrame, sensors: pd.DataFrame,
                  sig: dict, version: str, last_month: str) -> pd.DataFrame:
    out = []
    pv = printed[printed["verified"].astype(bool)]
    for srow in sensors.to_dict("records"):
        sid = srow["sensor_id"]
        mo = monthly[monthly.sensor_id == sid].set_index("month")
        pr = pv[pv.sensor_id == sid].set_index("month")
        if not len(mo) and not len(pr):
            continue
        first = min([m for m in list(mo.index) + list(pr.index)])
        months = [str(p) for p in pd.period_range(first, last_month, freq="M")]
        for m in months:
            f = mo.loc[m] if m in mo.index else None
            p = pr.loc[m] if m in pr.index else None
            pm = float(p["month_cm"]) if p is not None and pd.notna(p["month_cm"]) else np.nan
            py = float(p["year_cm"]) if p is not None and pd.notna(p["year_cm"]) else np.nan
            fv = float(f["value"]) if f is not None else np.nan
            ff = int(f["flag"]) if f is not None else 8
            row = {"sensor_id": sid, "pile": srow["pile"], "depth_m": srow["depth_m"],
                   "month": m, "printed_cm": pm, "printed_year_cm": py,
                   "printed_file": p["file"] if p is not None else "",
                   "feed_cm": fv, "feed_flag": ff, "feed": f["feed"] if f is not None else "",
                   "feed_sd_cm": float(f["sd"]) if f is not None else np.nan}
            noise = math.sqrt(2.0) * sig[sid] / 10.0
            if np.isfinite(pm):
                rsd = float(p["month_round_sd_cm"])
                row.update({"value": pm, "sd": math.sqrt(rsd ** 2 + noise ** 2), "flag": 0,
                            "source": "printed",
                            "method": "printed monthly value (WRA monthly report, verified "
                                      "transcription); primary"})
                if sid == "TKJS(1)":
                    row["method"] += "; printed depth label 300 m, unverifiable (no feed " \
                                     "overlap, no chain check)"
            elif f is not None and ff in (0, 1, 2, 9) and np.isfinite(fv):
                row.update({"value": fv, "sd": float(f["sd"]), "flag": ff,
                            "source": f"feed_{f['feed']}", "method": f["method"]})
            else:
                row.update({"value": np.nan, "sd": np.nan, "flag": 8, "source": "none",
                            "method": "no printed value and no feed increment"})
            row["diff_printed_minus_feed_mm"] = (pm - fv) * 10.0 if np.isfinite(pm) and \
                np.isfinite(fv) else np.nan
            row["product_version"] = version
            row["source_ids"] = (f"printed:{row['printed_file']}" if row["source"] == "printed"
                                 else (f["source_ids"] if f is not None else ""))
            out.append(row)
    return pd.DataFrame(out)


def cumulative_all(merged: pd.DataFrame, printed: pd.DataFrame, sig: dict,
                   version: str) -> pd.DataFrame:
    out = []
    pv = printed[printed["verified"].astype(bool) & printed["year_cm"].notna()]
    for sid, g in merged.groupby("sensor_id"):
        g = g.sort_values("month")
        months = [str(pd.Period(g.month.iloc[0], freq="M") - 1)] + g.month.tolist()
        incs = g[["month", "value", "sd", "flag"]].copy()
        incs = incs[incs.flag.isin([0, 1, 2])]
        p = pv[pv.sensor_id == sid].set_index("month")
        yrs = []
        noise = math.sqrt(2.0) * sig[sid] / 10.0
        for m, r in p.iterrows():
            m12 = str(pd.Period(m, freq="M") - 12)
            if m12 in p.index:          # same sensor printed twelve months earlier
                yrs.append({"month": m, "value": float(r["year_cm"]),
                            "sd": math.sqrt(float(r["year_round_sd_cm"]) ** 2 + noise ** 2)})
        yrs = pd.DataFrame(yrs, columns=["month", "value", "sd"])
        c = L.cumulative(months, incs, yrs)
        c.insert(0, "sensor_id", sid)
        c.insert(1, "pile", g.pile.iloc[0])
        c.insert(2, "depth_m", g.depth_m.iloc[0])
        c["n_year_constraints"] = len(yrs)
        # the month end before the first product month is a helper node: keep it only when
        # a constraint reaches it (it is the start level of the first increment)
        if c["flag"].iloc[0] == 8:
            c = c.iloc[1:]
        out.append(c)
    cum = pd.concat(out, ignore_index=True)
    cum["method"] = ("weighted least squares on printed monthly (primary) and feed increments "
                     "plus printed trailing-twelve-month values; cm of compression since the "
                     "reference month end")
    cum.loc[cum.flag == 2, "method"] += "; reached only through flag-2 increments"
    cum.loc[cum.flag == 8, "method"] = "not connected to the reference"
    cum["product_version"] = version
    cum["source_ids"] = "merged_monthly.parquet"
    return cum


# ------------------------------------------------------------------------------- test C
def _vs(units: pd.DataFrame, feed: pd.DataFrame, lo: str, hi: str, flags_ok=(0, 1)) -> dict:
    u = units[(units.month >= lo) & (units.month <= hi)]
    return T.score(u, feed, flags_ok=flags_ok)


def test_c(daily: pd.DataFrame, monthly: pd.DataFrame, printed_int: pd.DataFrame,
           version: str) -> dict:
    """Pre-registered test C (S2r2.c_fresh_cy007001) and its secondary comparisons."""
    from harness import prereg as pr
    prod = guard.open_product(PRINTED_CY, "validator")
    cy = prod.read_table("printed_monthly.parquet")
    auth = pr.assert_can_score(PREREG_C_ID, "S2")          # refuses if registered too late
    inc = T.monthly_mean_increments(daily, monthly)
    units_all = T.printed_units(cy, last_month="2099-12")
    units = units_all[units_all.month <= "2023-12"]
    out = {"preregistration": {"id": PREREG_C_ID, "sha256": auth["sha256"],
                               "other_tests_prior_looks": auth["other_tests_prior_looks"]},
           "printed_product": rel_to_repo(PRINTED_CY), "printed_notes": prod.notes}
    c1 = _vs(units, inc, "2019-01", "2022-12")
    c2 = _vs(units, inc, "2023-01", "2023-12")
    out["C1_preorigin"] = c1["summary"]
    out["C2_2023"] = c2["summary"]
    # secondary 1: month-end increments of the product (test A's convention)
    me = monthly[["sensor_id", "month", "value", "flag", "feed"]].copy()
    me["inc_mm"] = me["value"] * 10.0
    out["secondary_month_end_preorigin"] = _vs(units, me, "2019-01", "2022-12")["summary"]
    out["secondary_month_end_2023"] = _vs(units, me, "2023-01", "2023-12")["summary"]
    # secondary 2: feeds as delivered (vendor as delivered; raw without the catalogue)
    dv = daily[daily.feed == "vendor"].copy()
    mv = monthly[["sensor_id", "month"]].assign(raw_flag=8, vendor_flag=0)
    inc_v = T.monthly_mean_increments(dv, mv)
    dr = daily[daily.feed == "raw"].copy()
    dr["value"] = dr["raw_value"]
    dr["flag"] = np.where(dr["flag"].isin([0, 1]), 0, dr["flag"])
    mr = monthly[["sensor_id", "month"]].assign(raw_flag=0, vendor_flag=8)
    inc_r = T.monthly_mean_increments(dr, mr)
    for name, feed in (("vendor_as_delivered", inc_v), ("raw_uncatalogued", inc_r)):
        out[f"secondary_{name}_preorigin"] = _vs(units, feed, "2019-01", "2022-12")["summary"]
        out[f"secondary_{name}_2023"] = _vs(units, feed, "2023-01", "2023-12")["summary"]
    # secondary 3: printed 007001 against printed integrated values (no feed)
    pi = printed_int[printed_int["verified"].astype(bool) & printed_int["month_cm"].notna()
                     & printed_int["sensor_id"].notna()]
    pj = units_all.merge(pi[["sensor_id", "month", "month_cm"]], on=["sensor_id", "month"])
    pj["diff_mm"] = pj["value"] - pj["month_cm"] * 10.0
    out["secondary_printed_cy_vs_integrated"] = {
        "n": int(len(pj)), "median_abs_mm": float(pj.diff_mm.abs().median()) if len(pj)
        else float("nan"), "max_abs_mm": float(pj.diff_mm.abs().max()) if len(pj)
        else float("nan"), "share_within_0p5mm": float((pj.diff_mm.abs() <= 0.5 + 1e-9).mean())
        if len(pj) else float("nan"), "share_within_1mm": float((pj.diff_mm.abs() <= 1.0
                                                                 + 1e-9).mean())
        if len(pj) else float("nan"),
        "r": T.pearson(pj["value"], pj["month_cm"]) if len(pj) >= 3 else float("nan"),
        "rows": pj[["sensor_id", "month", "value", "month_cm", "diff_mm"]].round(3)
        .to_dict("records")}
    # secondary 4: design-time overlap with the feeds (units of tests A and B)
    out["secondary_designtime_2024_2025"] = _vs(units_all, inc, "2024-01", "2025-04")["summary"]
    # per-unit table (C1 + C2 + later months, both conventions)
    u = c1["units"].assign(test="C1")
    u = pd.concat([u, c2["units"].assign(test="C2"),
                   _vs(units_all, inc, "2024-01", "2099-12")["units"].assign(
                       test="design_time_or_no_feed")], ignore_index=True)
    u = u.merge(me[["sensor_id", "month", "inc_mm", "flag"]].rename(
        columns={"inc_mm": "month_end_inc_mm", "flag": "month_end_flag"}),
        on=["sensor_id", "month"], how="left")
    u = u.rename(columns={"inc_mm": "monthly_mean_inc_mm", "flag": "monthly_mean_flag",
                          "value": "printed_compression_mm"})
    u["product_version"] = version
    mm_inc = inc.copy()
    mm_inc["value"] = mm_inc["inc_mm"] / 10.0
    mm_inc["sd"] = np.nan
    mm_inc["method"] = np.where(mm_inc["flag"] == 8, "no feed with >= 20 full days in both "
                                "months", "difference of consecutive monthly means of full days "
                                "(c_test.monthly_mean_increments); cm")
    mm_inc["product_version"] = version
    mm_inc["source_ids"] = mm_inc["feed"].where(mm_inc["feed"] != "", "")
    cols = units_all[["sensor_id", "month", "value", "report", "column"]].rename(
        columns={"value": "printed_cy007001_mm", "report": "printed_cy007001_report",
                 "column": "printed_cy007001_column"})
    return {"validation": out, "units": u, "mm_inc": mm_inc, "printed_cols": cols}


# ------------------------------------------------------------------------------- acceptance
def acceptance(merged, monthly, printed, sensors, per_sensor, sig) -> dict:
    pv = printed[printed["verified"].astype(bool) & printed["month_cm"].notna()].copy()
    on_grid = dict(zip(sensors.sensor_id, sensors.on_grid, strict=True))
    res = {}

    def a_units(feedcol, flagcol, flags_ok, src):
        rows = []
        for r in pv.itertuples():
            sid = r.sensor_id
            mo = src[(src.sensor_id == sid) & (src.month == r.month)]
            if not len(mo):
                continue
            f = mo.iloc[0]
            if int(f[flagcol]) in flags_ok and np.isfinite(f[feedcol]):
                d = (float(r.month_cm) - float(f[feedcol])) * 10.0
                rows.append({"sensor_id": sid, "pile": r.pile, "depth_m": int(r.depth_m),
                             "month": r.month, "printed_cm": float(r.month_cm),
                             "feed_cm": float(f[feedcol]), "feed_flag": int(f[flagcol]),
                             "diff_mm": d, "ok": abs(d) <= 2.0 + 1e-9,
                             "on_grid": bool(on_grid.get(sid, False))})
        return pd.DataFrame(rows)

    def summ(u):
        if not len(u):
            return {"n": 0}
        a = u.diff_mm.abs()
        return {"n": int(len(u)), "n_ok": int(u.ok.sum()), "share_ok": float(u.ok.mean()),
                "median_abs_mm": float(a.median()), "p95_abs_mm": float(a.quantile(0.95)),
                "max_abs_mm": float(a.max()), "pass": bool(u.ok.all()),
                "failures": u[~u.ok][["sensor_id", "month", "printed_cm", "feed_cm",
                                      "feed_flag", "diff_mm"]].round(3).to_dict("records")}

    uA = a_units("value", "flag", (0, 1), monthly)
    res["A_primary"] = summ(uA)
    res["A_primary"]["criterion"] = "|printed - feed| <= 2.0 mm for every unit (flags 0-1)"
    res["A_by_sensor"] = {s: summ(g) for s, g in uA.groupby("sensor_id")} if len(uA) else {}
    res["A_secondary_flag012"] = summ(a_units("value", "flag", (0, 1, 2), monthly))
    res["A_secondary_vendor_as_delivered"] = summ(a_units("vendor_cm", "vendor_flag",
                                                          (0, 1, 2, 9), monthly))
    # raw without the catalogue: month-end levels of the uncatalogued raw daily means
    raw_unc = []
    for sid, rec in per_sensor.items():
        rh = rec["rh"]
        d = rh.groupby(rh.index.floor("D")).agg(["mean", "count"])
        full = d[d["count"] >= L.DAY_MIN_HOURS]["mean"]
        if not len(full):
            continue
        me = full.groupby(full.index.to_period("M")).apply(
            lambda x: x.iloc[-1] if x.index[-1].day == x.index[-1].days_in_month else np.nan)
        inc = me.diff() / 10.0
        for p, v in inc.items():
            raw_unc.append({"sensor_id": sid, "month": str(p), "raw_unc_cm": v,
                            "raw_unc_flag": 0 if np.isfinite(v) else 8})
    raw_unc = pd.DataFrame(raw_unc)
    res["A_secondary_raw_without_catalogue"] = summ(a_units("raw_unc_cm", "raw_unc_flag", (0,),
                                                            raw_unc))
    res["A_secondary_on_grid_only"] = summ(uA[uA.on_grid]) if len(uA) else {"n": 0}
    res["A_units"] = uA.round(4).to_dict("records")
    # B: mapping correlation
    brows = []
    for (pile, depth), g in pv.groupby(["pile", "depth_m"]):
        sid = L.PRINTED_TO_SENSOR.get((pile, int(depth)))
        cand = {}
        for s2 in monthly.sensor_id.unique():
            mo = monthly[(monthly.sensor_id == s2) & monthly.flag.isin([0, 1, 2])]
            j = g.merge(mo[["month", "value"]], on="month")
            if len(j) >= 5:
                cand[s2] = {"n": int(len(j)), "r": L.pearson(j.month_cm, j.value),
                            "median_abs_mm": float(((j.month_cm - j.value).abs() * 10).median())}
        mine = cand.get(sid, {"n": 0, "r": float("nan"), "median_abs_mm": float("nan")})
        best_r = max(cand.items(), key=lambda kv: (kv[1]["r"] if np.isfinite(kv[1]["r"])
                                                   else -9))[0] if cand else ""
        best_mad = min(cand.items(), key=lambda kv: kv[1]["median_abs_mm"])[0] if cand else ""
        brows.append({"pile": pile, "depth_m": int(depth), "sensor_id": sid,
                      "on_grid": bool(on_grid.get(sid, False)), "n": mine["n"], "r": mine["r"],
                      "median_abs_mm": mine["median_abs_mm"], "testable": mine["n"] >= 5,
                      "pass": bool(mine["n"] >= 5 and mine["r"] >= 0.85),
                      "best_by_r": best_r, "best_by_median_abs": best_mad,
                      "candidates": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                         for kk, vv in v.items()} for k, v in cand.items()}})
    bt = [b for b in brows if b["testable"]]
    res["B"] = {"criterion": "Pearson r >= 0.85 for every printed (pile, depth) with >= 5 "
                             "overlapping months",
                "units": brows, "n_testable": len(bt),
                "min_r": float(min(b["r"] for b in bt)) if bt else float("nan"),
                "pass": bool(bt) and all(b["pass"] for b in bt),
                "untestable": [f"{b['pile']} {b['depth_m']} m" for b in brows if not b["testable"]]}
    res["gate_passed"] = bool(res["A_primary"].get("pass", False) and res["B"]["pass"])
    # printed 土庫國中 depth-label diagnostic (2025-05/06/08 printed at 300 m)
    tk = printed[(printed.pile == "土庫國中")][["month", "depth_m", "month_cm", "year_cm"]]
    res["tuku_depth_label"] = tk[(tk.month >= "2025-03") & (tk.month <= "2025-10")].astype(
        {"depth_m": "float"}).to_dict("records")
    return res


def recall_summary(recall: pd.DataFrame) -> dict:
    out = {}
    for est, g in recall.groupby("estimator"):
        out[est] = {"false_positive_share": {int(r.d_h): round(r.detected_share, 4)
                                             for r in g[g.size_mm == 0].itertuples()},
                    "recall": {f"{int(r.d_h)}h_{r.size_mm:g}mm": round(r.detected_share, 4)
                               for r in g[g.size_mm > 0].itertuples()},
                    "mean_recall": round(float(g[g.size_mm > 0].detected_share.mean()), 4)}
    return out


def steps_summary(steps: pd.DataFrame, stitches: pd.DataFrame) -> dict:
    s = steps
    out = {"by_feed_kind": s.groupby(["feed", "kind"]).size().rename("n").reset_index()
           .to_dict("records"),
           "raw_steps": s[(s.feed == "raw") & (s.kind == "step")][
               ["sensor_id", "t_before", "t_after", "duration_h", "delta_obs_mm",
                "resid_mm", "threshold_mm", "estimator", "plan_item", "vendor_treatment"]]
           .round(3).to_dict("records"),
           "vendor_steps": s[(s.feed == "vendor") & (s.kind == "step")][
               ["sensor_id", "t_before", "t_after", "duration_h", "delta_obs_mm", "resid_mm",
                "threshold_mm"]].round(3).to_dict("records")}
    if len(stitches):
        out["vendor_stitches"] = stitches.groupby("classification").size().to_dict()
        out["vendor_stitches_removed_real_movement"] = stitches[
            stitches.classification == "removed_real_movement"][
            ["sensor_id", "t_before", "t_after", "gap_h", "raw_change_mm", "vendor_change_mm"]
        ].round(3).to_dict("records")
        out["sign_inversions"] = stitches[stitches.classification == "sign_inversion"][
            ["sensor_id", "t_before", "t_after"]].to_dict("records")
    plan = {"step at every pile 2024-02-29/03-01": sorted(set(
        s[s.plan_item.str.startswith("plan: step at every pile")].sensor_id)),
        "溪州 reset March 2025": bool((s.plan_item == "plan: 溪州 reset March 2025").any()),
        "灣內 sign inversion": bool(len(stitches) and (
            (stitches.sensor_id == "WNES") & (stitches.classification == "sign_inversion")).any())}
    out["plan_items_found"] = plan
    return out


def mapping_table(sensors, merged, monthly, printed, ident, vendor_of, defective) -> pd.DataFrame:
    rows = []
    for s in sensors.to_dict("records"):
        sid = s["sensor_id"]
        pk = [f"{k[0]} {k[1]} m" for k, v in L.PRINTED_TO_SENSOR.items() if v == sid]
        g = merged[(merged.sensor_id == sid) & merged.printed_cm.notna() & merged.feed_cm.notna()
                   & merged.feed_flag.isin([0, 1, 2])]
        idr = ident[(ident.vendor_key == s["vendor_key"]) & (ident.raw_id == s["raw_id"])]
        mo = monthly[monthly.sensor_id == sid]
        rows.append({**s, "printed_key": ";".join(pk),
                     "n_printed_months": int(merged[(merged.sensor_id == sid)
                                                    & merged.printed_cm.notna()].shape[0]),
                     "n_overlap": int(len(g)),
                     "r_printed_feed": L.pearson(g.printed_cm, g.feed_cm) if len(g) >= 3
                     else float("nan"),
                     "identity_share_same": float(idr.share_same.iloc[0]) if len(idr) else np.nan,
                     "identity_share_inverted": float(idr.share_inverted.iloc[0]) if len(idr)
                     else np.nan,
                     "identity_hours": int(idr.n_common_hours.iloc[0]) if len(idr) else 0,
                     "first_month": mo.month.min() if len(mo) else "",
                     "last_month": mo[mo.flag != 8].month.max() if len(mo) else "",
                     "domain": ("on the model grid" if s["on_grid"] else
                                ("off the fan (sign-inverted raw feed)" if sid == "WNES" else
                                 ("Pingtung, out of domain" if sid == "DLIO" else "off grid")))})
    for vk, nominal in defective.items():
        rows.append({"sensor_id": nominal, "vendor_key": vk, "raw_id": "",
                     "domain": "vendor series defective: no identity with any raw sensor; "
                               "flag 9 (not used)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- write
def write_tables(res: dict, d: Path) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    names = []

    def pq(df, name):
        df.to_parquet(d / name, index=False)
        names.append(name)

    def csv(df, name):
        df.to_csv(d / name, index=False)
        names.append(name)

    pq(res["daily"], "daily.parquet")
    pq(res["monthly"], "monthly.parquet")
    pq(res["merged"], "merged_monthly.parquet")
    pq(res["cum"], "cumulative_monthly.parquet")
    pq(res["mm_inc"], "monthly_mean_increments.parquet")
    pq(res["cy_units"], "test_c_units.parquet")
    csv(res["steps"], "steps.csv")
    csv(res["stitches"], "vendor_stitches.csv")
    csv(res["identity"], "identity.csv")
    csv(res["mapping"], "mapping.csv")
    csv(res["envelope"], "envelope.csv")
    csv(res["recall"], "injected_recall.csv")
    return names


def table_checks(res: dict) -> dict:
    out = {}
    for k in ("daily", "monthly", "merged", "cum", "steps"):
        out[k] = F.validate_long_table(res[k])
    return out


def date_checks(res: dict) -> list[str]:
    """Calendar sanity: every month/date lies between the first feed month (2016-06) and the
    last product month, and no time key is missing (the guard refuses missing values)."""
    errs = []
    hi = res["last_month"]
    for k in ("daily", "monthly", "merged", "cum", "steps"):
        df = res[k]
        if "month" not in df.columns:
            errs.append(f"{k}: no month column")
            continue
        m = df["month"]
        if m.isna().any():
            errs.append(f"{k}: {int(m.isna().sum())} rows without a month")
            continue
        bad = m[(m < "2016-06") | (m > hi)]
        if len(bad):
            errs.append(f"{k}: {len(bad)} rows outside 2016-06..{hi} (e.g. {bad.iloc[0]})")
    d = pd.to_datetime(res["daily"]["date"])
    if d.max() > pd.Timestamp("2025-04-30"):
        errs.append(f"daily: date {d.max()} after the last feed day (2025-04-16)")
    return errs


def card(res: dict, version: str) -> str:
    v = res["validation"]
    m = res["mapping"]
    lines = [f"# deep_benchmarks/{version}: deep benchmarks (深層樁) as a first-class series",
             "", "WP S2 of docs/DATA_REMEDIATION_PLAN_2026-10.md. Built "
             f"{dt.date.today().isoformat()} by dataprep_staging/s2_deep_benchmarks/build.py; "
             "rules in results/dataprep/preregistration/S2.deep_benchmark_acceptance.v1.json, "
             "declared with the A1 tool as S2r2.deep_benchmarks_core; fresh test C registered "
             "as S2r2.c_fresh_cy007001.", "",
             "## Coverage", ""]
    for r in m.to_dict("records"):
        lines.append(f"- {r.get('sensor_id')} {r.get('pile', '')} {r.get('depth_m', '')} m: "
                     f"{r.get('domain', '')}; months {r.get('first_month', '')}.."
                     f"{r.get('last_month', '')}; vendor {r.get('vendor_key', '')}")
    lines += ["", "## Tables", "",
              "- daily.parquet: one row per sensor, feed (raw | vendor) and day; value = daily "
              "mean reading (mm, rising = compression; raw values after the step catalogue).",
              "- monthly.parquet: monthly increments (cm) per sensor; value = primary feed "
              "(raw catalogued where flag 0-2, else vendor), raw_cm and vendor_cm kept.",
              "- merged_monthly.parquet: printed monthly values primary (2024-01..2026-01), "
              "feed elsewhere; printed_cm, feed_cm and their difference.",
              "- cumulative_monthly.parquet: cumulative compression (cm) since the end of "
              "2022-12 (or the first connected month end) by weighted least squares.",
              "- steps.csv, vendor_stitches.csv: the step catalogue and the vendor's stitches.",
              "- mapping.csv, identity.csv: sensor to pile, vendor series to sensor.",
              "- envelope.csv, injected_recall.csv: pre-origin thresholds and their recall.",
              "- monthly_mean_increments.parquet: differences of consecutive monthly means "
              "(cm), the convention of the 007001 reports; test_c_units.parquet: every printed "
              "007001 value against both feed conventions (tests C1, C2 and later months).",
              "- merged_monthly.parquet also carries printed_cy007001_mm (compression, mm) "
              "from results/dataprep/deep_benchmarks_printed/v1 (not merged as primary).",
              "", "## Flags", "",
              "0 observed; 1 a removed sensor step inside the interval (or, for daily levels, "
              "after it); 2 month end bridged from up to 3 days earlier or a partial day; 8 "
              "missing; 9 suspect (vendor stitch that removed movement, unremoved step, spike "
              "hours, the defective vendor TKJS_283_29).",
              "", "## Uses", "",
              "Validator and scoring target for twin subsidence 2023-26 (never fitted, never in a "
              "gate or a choice). Pre-origin uses (datum_fit, calibration_target, baseline, "
              "design_time_evidence: S1 path shape, S4/S5 calibration points) only through a "
              "time slice ending by 2022-12. Use increments, not levels (the ring/pile rate "
              "ratio varies by window).",
              "", "## Known limits", "",
              "- The raw agency feed exists only as two 100,000-row dumps: all sensors "
              "2023-09-21..2025-04-16, and 土庫國中 300 m and 130 m from 2016. Elsewhere the "
              "vendor feed is the only feed; its stitches are invisible there.",
              "- 土庫國中 400 m (the printed pile) exists only in the raw feed from 2023-09; the "
              "vendor TKJS_283_29 does not track it.",
              "- Printed 土庫國中 values labelled 300 m (2025-05, 06, 08) cannot be checked "
              "against a feed (no feed after 2025-04).",
              "- The integrated reports' monthly convention is assumed to be month-end daily "
              "means (test A); the 007001 reports state monthly means (test C).",
              f"- Gate (A and B) passed: {v['gate_passed']}; A {v['A_primary'].get('n_ok')}/"
              f"{v['A_primary'].get('n')} within 2 mm. Test C1 (pre-origin) passed: "
              f"{v.get('C', {}).get('C1_preorigin', {}).get('pass')}; C2 (2023) passed: "
              f"{v.get('C', {}).get('C2_2023', {}).get('pass')}."]
    return "\n".join(lines) + "\n"


def manifest_notes(res: dict, val: dict) -> list[str]:
    a, b, c = val["A_primary"], val["B"], val.get("C", {})
    c1, c2 = c.get("C1_preorigin", {}), c.get("C2_2023", {})
    return [
        f"Observed product 2016-06..{res['last_month']} (feeds to 2025-04-16; printed "
        "integrated values 2024-01..2026-01 primary where they exist). Numeric parameters "
        "(envelope Q(d), floor, sigma_day) fitted on data up to 2022-12 only. The "
        "vendor-to-sensor identity was established on 2023-25 common hours; it labels series "
        "and sets the sign of the raw 灣內 feed after 2024-08-08, it fits no number.",
        f"Gate (plan WP S2, declared in S2r2.deep_benchmarks_core): A {a.get('n_ok')}/"
        f"{a.get('n')} units within 2 mm (max {a.get('max_abs_mm', float('nan')):.2f} mm; "
        f"pass {a.get('pass')}); B min r {b.get('min_r', float('nan')):.3f} over "
        f"{b.get('n_testable')} piles (pass {b.get('pass')}); gate passed "
        f"{val['gate_passed']}. A and B were scored on design-time units (seeded look).",
        f"Fresh test C (S2r2.c_fresh_cy007001, printed 007001 table, monthly-mean "
        f"convention): C1 pre-origin {c1.get('n_ok')}/{c1.get('n_testable')} within 2 mm, "
        f"pass {c1.get('pass')}; C2 2023 {c2.get('n_ok')}/{c2.get('n_testable')}, pass "
        f"{c2.get('pass')}.",
        "Role (plan 4.1): validator of twin subsidence 2023-26; never fitted, never in a gate, "
        "ladder decision, member choice or threshold (U7). Deep piles that shape S1's 2022 "
        "path cannot be an independent check of S1 at those sites. Use increments, not levels.",
        "Pre-origin uses (datum_fit, calibration_target, baseline, design_time_evidence) only "
        "through the time slice (window_end <= 2022-12); flag 9 marks suspect increments "
        "(vendor stitches that the raw record contradicts, unremoved steps).",
        "No integrated WRA monthly report newer than 2026-01 exists (portal listing "
        "GetDataReport/007003 checked 2026-10-12; newest 115年01月).",
    ]


def manifest_for(d: Path, res: dict, val: dict) -> dict:
    last = res["last_month"]
    inputs = [L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet" for k in L.VENDOR_KEYS]
    inputs += [*(L.OPEN_DIR / f for f in L.OPEN_FILES), L.OPEN_DIR / "SOURCE.json",
               L.PRINTED_CSV, L.PRINTED_REPORT, L.SITES_CSV,
               REPO / "data_fetch_api/new_measurements/wra_monthly_reports/SOURCE.json",
               L.GRID_NPZ, PREREG_V1, PREREG, PREREG_C,
               PRINTED_CY / "MANIFEST.json", PRINTED_CY / "printed_monthly.parquet"]
    inputs += [Path(x) for x in res.get("extra_inputs", [])]
    gate = {"passed": bool(val["gate_passed"]),
            "criteria": "plan 3 WP S2: A printed vs feed monthly |diff| <= 2 mm after the step "
                        "catalogue (every unit, flags 0-1); B mapping r >= 0.85 for every "
                        "printed pile with >= 5 overlapping months (declared in "
                        f"{PREREG.name}; rules of {PREREG_V1.name})"}
    return manifest.build_manifest(
        d, product=PRODUCT, layer="L2", inputs=[rel_to_repo(p) for p in inputs],
        code=[rel_to_repo(p) for p in CODE + res.get("extra_code", [])],
        parameters={"gap_h": L.GAP_H, "merge_h": L.MERGE_H, "side_h": L.SIDE_H,
                    "min_side_pts": L.MIN_SIDE_PTS, "floor_mm": L.FLOOR_MM,
                    "q_level": L.Q_LEVEL, "durations_h": list(L.DURS_H),
                    "envelope_series": list(ENVELOPE_KEYS), "envelope_end": str(L.ORIGIN_END),
                    "envelope_draws_per_sensor_duration": 3000, "seed": 2026,
                    "day_min_hours": L.DAY_MIN_HOURS, "bridge_days": L.BRIDGE_DAYS,
                    "reference_month": L.ORIGIN_MONTH, "last_month": last,
                    "identity_min_share": 0.98,
                    "claude_api_tokens": res.get("claude_api_tokens", 0)},
        fit_window_end=last,
        validation_report="validation.json", gate=gate,
        time_slicing={"column": "month", "params_fit_window_end": L.ORIGIN_MONTH,
                      "encoding": "iso"},
        allowed_uses=["validator", "scoring_target", "after_the_fact_analysis", "datum_fit",
                      "calibration_target", "baseline", "design_time_evidence"],
        forbidden_uses=["selection", "model_design", "prior", "prior_width_input",
                        "initial_conditions", "idw_field", "forcing_frozen",
                        "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none (public benchmark station coordinates only; station contact "
                      "columns never read)",
        extra={"notes": manifest_notes(res, val)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    for p in (PREREG_V1, PREREG, PREREG_C):
        if not p.exists():
            raise SystemExit(f"pre-registration missing: {p}")
    version = "deep_benchmarks/scratch" if a.scratch is not None else None
    if version is None:   # name the next version without creating it (a crash burns none)
        base = PREREG.parents[1] / PRODUCT
        ns = [int(q.name[1:]) for q in base.glob("v*") if q.name[1:].isdigit()] \
            if base.exists() else []
        version = f"{PRODUCT}/v{max(ns, default=0) + 1}"
    res = compute(version)
    checks = {"table_checks": table_checks(res), "date_checks": date_checks(res)}
    bad = {k: v for k, v in checks["table_checks"].items() if v}
    if bad or checks["date_checks"]:
        raise SystemExit(f"table/date checks failed: {bad} {checks['date_checks']}")
    val = {**res["validation"], **checks,
           "preregistration": [{"path": rel_to_repo(p), "sha256": sha256_file(p)}
                               for p in (PREREG_V1, PREREG, PREREG_C)]}
    if a.scratch is not None:
        names = write_tables(res, a.scratch)
        (a.scratch / "validation.json").write_text(json.dumps(
            val, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        log("scratch tables:", names)
        return 0
    d = manifest.new_version_dir(PRODUCT)
    if f"{PRODUCT}/{d.name}" != version:
        raise SystemExit(f"version race: computed for {version}, got {d}")
    write_tables(res, d)
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                  default=str) + "\n", encoding="utf-8")
    (d / "README.md").write_text(card(res, d.name), encoding="utf-8")
    m = manifest_for(d, res, val)
    errs = guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit(f"time_slicing errors (manifest not written): {errs}")
    manifest.write_manifest(d, m)
    log(f"wrote {d}")
    log(json.dumps({"gate_passed": val["gate_passed"],
                    "A": {k: val["A_primary"].get(k) for k in ("n", "n_ok", "max_abs_mm",
                                                               "pass")},
                    "B": {k: val["B"].get(k) for k in ("n_testable", "min_r", "pass")}},
                   ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
