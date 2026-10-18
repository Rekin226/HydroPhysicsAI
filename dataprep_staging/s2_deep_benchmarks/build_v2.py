"""Build ``results/dataprep/deep_benchmarks/v<N>/`` version 2 (WP S2, round 4, 2026-10-12).

v1 (``build.py``, unchanged: it is the code of deep_benchmarks/v1) declared a time slice on
``month`` with ``params_fit_window_end`` 2022-12, but some of its pre-origin rows were decided
by post-origin observations (independent verification of round 3):

* the vendor stitch of 土庫國中 130 m (TKJS(2)) across the raw outage 2021-08-19..2023-05-11
  was classified with the raw reading of 2023-05-11, and the classification flagged the 17
  pre-origin months 2021-08..2022-12 suspect (flag 9);
* the vendor-to-sensor labels came from the identity of the two feeds on 2023-25 hours.

v2 therefore computes every row dated up to 2022-12 in a **pre-origin run** and every later
row in a **full run** (rules declared in ``S2r4.deep_benchmarks_v2`` before the build):

1. Pre-origin run: the vendor samples and the raw agency hours are cut before
   2023-01-01T00:00, so no detection window, stitch, outage or identity reaches past the
   origin. Vendor series are labelled by ``label_confirm.py`` from pre-origin data only
   (evidence ``S2_evidence/r4/preorigin_labels.json``): identity with a raw sensor on
   pre-origin hours, else the agency station code, else the printed 007001 table up to
   2022-12. A series whose label is ambiguous keeps its nominal label (the plan's table) and
   every pre-origin row of it is flag 9; a defective series is flag 9 as in v1. The printed
   integrated values start in 2024-01, so the pre-origin run merges none. Its cumulative
   series is solved on pre-origin increments only, anchored at 0 at the end of 2022-12.
2. Full run: v1's computation with v1's labels (identity on common hours, 2016-2025); it
   reproduces v1's tables (checked table by table).
3. Assembly: rows dated <= 2022-12 (day, month, or a step's month) come from the pre-origin
   run, later rows from the full run; column ``variant`` says which. Monthly-mean increments
   are computed within each run. Tables without a month column (identity, mapping,
   stitches, envelope, recall) hold both runs and cannot be read through a time slice.
4. The step envelope Q(d), the floor and sigma_day are pre-origin in both runs (identical).
5. Gate: the plan's A and B, unchanged (declared in S2r2.deep_benchmarks_core), on the full
   run's units (2024-01..2025-04, design-time units).
6. Uncertainty (plan 4.7): every sd column is labelled ``uncalibrated``; the coverage of the
   nominal 95 % interval of the month-end increments is measured on the test-A units and
   reported (``validation.json``), never used to rescale.
7. Test C is not re-registered: its units were scored on v1. C on v2's rows is computed after
   the fact (look logged) and reported as such.

Usage: python -I build_v2.py [--scratch DIR] [--debug-labels-v1]
  --scratch DIR        write the tables to DIR (no product version, no manifest)
  --debug-labels-v1    (development only, with --scratch) label the pre-origin run with v1's
                       labels instead of the evidence file; computes no acceptance and no C
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import flags as F  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import build as B  # noqa: E402
from s2_deep_benchmarks import c_test as T  # noqa: E402
from s2_deep_benchmarks import s2lib as L  # noqa: E402

PRODUCT = "deep_benchmarks"
CUT = pd.Timestamp("2023-01-01 00:00")
ORIGIN = L.ORIGIN_MONTH
DECL_ID = "S2r4.deep_benchmarks_v2"
DECL = PREREG_DIR / f"{DECL_ID}.json"
PREREG_FILES = [B.PREREG_V1, B.PREREG, B.PREREG_C, DECL]
LABELS_JSON = REPORTS_DIR / "S2_evidence" / "r4" / "preorigin_labels.json"
PRINTED_V1 = DATAPREP / "deep_benchmarks_printed" / "v1"
PRINTED_V2 = DATAPREP / "deep_benchmarks_printed" / "v2"
V1 = DATAPREP / PRODUCT / "v1"
HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
HARNESS = HERE.parent / "harness"
CODE = [Path(__file__), Path(B.__file__), Path(L.__file__), Path(T.__file__),
        HERE / "label_confirm.py", HERE / "__init__.py",
        COMMON / "__init__.py", COMMON / "flags.py", COMMON / "guard.py",
        COMMON / "manifest.py", COMMON / "hashing.py", COMMON / "paths.py",
        COMMON / "uses.py", COMMON / "gitinfo.py",
        HARNESS / "__init__.py", HARNESS / "prereg.py"]
SD_STATUS = "uncalibrated"
EMPTY_EVENTS = ["kind", "estimator", "a", "b", "t_before", "t_after", "duration_h",
                "delta_obs_mm", "delta_pred_mm", "resid_mm", "threshold_mm", "rate_mm_per_day",
                "rate_ok", "corrected", "correction_mm", "sd_mm", "n_flagged_pairs"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- inputs
def load_feeds(cut: pd.Timestamp | None):
    """Vendor samples and hourly series, raw hourly series (cut before ``cut`` if given)."""
    vendor_samples = {k: L.vendor_raw(k) for k in L.VENDOR_KEYS}
    raw_h, raw_info = L.load_open([L.OPEN_DIR / f for f in L.OPEN_FILES])
    if cut is not None:
        vendor_samples = {k: s[s.index < cut] for k, s in vendor_samples.items()}
        raw_h = {k: s[s.index < cut] for k, s in raw_h.items()}
    vendor_h = {k: L.to_hourly(s) for k, s in vendor_samples.items()}
    return vendor_samples, vendor_h, raw_h, raw_info


def identity_table(vendor_h: dict, raw_h: dict) -> tuple[pd.DataFrame, dict]:
    irows, full = [], {}
    for vk in L.VENDOR_KEYS:
        for rid, rh in raw_h.items():
            out = L.identity(vendor_h[vk], rh)
            full[(vk, rid)] = out
            irows.append({"vendor_key": vk, "raw_id": rid, "n_common_hours": out["n_common_hours"],
                          "n_informative_hours": out.get("n_informative_hours", 0),
                          "share_same": out.get("share_same"),
                          "share_inverted": out.get("share_inverted"),
                          "share_union": out.get("share_union"),
                          "n_segments": len(out.get("segments", [])),
                          "n_stitches": len(out.get("stitches", []))})
    ident = pd.DataFrame(irows)
    ident["best_share"] = ident["share_union"].astype(float)
    ident["chosen"] = False
    return ident, full


def choose_v1(ident: pd.DataFrame) -> pd.DataFrame:
    """v1's rule (build.compute): per vendor series the raw sensor with the largest share of
    informative common hours with an unchanged offset or sum, if >= 0.98 on >= 48 hours."""
    ident = ident.copy()
    for _vk, g in ident.groupby("vendor_key"):
        g = g[g.n_common_hours >= 48].sort_values("best_share", ascending=False)
        if len(g) and g.best_share.iloc[0] >= 0.98:
            ident.loc[g.index[0], "chosen"] = True
    dup = ident[ident.chosen].raw_id.duplicated(keep=False)
    if dup.any():
        raise RuntimeError(f"one raw sensor identified with several vendor series: "
                           f"{ident[ident.chosen][dup].to_dict('records')}")
    return ident


def preorigin_label_table(labels: dict, raw_names: dict) -> tuple[pd.DataFrame, dict]:
    """(vendor_key, raw_id, chosen) rows for build.sensor_registry from the pre-origin label
    evidence, and the status per vendor series."""
    rows, status = [], {}
    for vk in L.VENDOR_KEYS:
        st = labels["status"].get(vk, "unlabelled")
        status[vk] = st
        if st == "confirmed":
            sid = labels["labels"][vk]
        elif st == "ambiguous":
            sid = labels["nominal"][vk]
        else:
            continue
        rid = f"BM_{sid}"
        if rid not in raw_names:
            raise RuntimeError(f"label {vk} -> {sid}: no agency sensor {rid}")
        rows.append({"vendor_key": vk, "raw_id": rid, "chosen": True})
    t = pd.DataFrame(rows, columns=["vendor_key", "raw_id", "chosen"])
    if t.raw_id.duplicated().any():
        raise RuntimeError(f"a sensor labelled twice before the origin: {t.to_dict('records')}")
    return t, status


# ------------------------------------------------------------------------------- one run
def note_283(status: str | None) -> str:
    """Justification of the defective vendor series TKJS_283_29 (plan WP S2 maps it to
    土庫國中 400 m)."""
    st = {"defective": "it does not track the printed 土庫國中 400 m values before the origin",
          "confirmed": "it tracks the printed 400 m values before the origin, but no full-period "
                       "identity exists"}.get(status or "", "pre-origin tracking test not run")
    return ("vendor daily series TKJS_283_29: its identity with a raw sensor is undeterminable "
            f"(one sample a day, 0 informative hours); {st} (S2_evidence/r4/"
            "preorigin_labels.json); defective (flag 9, not used)")


def run_variant(version: str, variant: str, cut: pd.Timestamp | None,
                pre_labels: dict | None, status_283: str | None = None) -> dict:
    """One computation of the product (v1's steps 1-7). ``variant`` 'full' reproduces v1;
    'preorigin' cuts the feeds at ``cut`` and labels the vendor series with ``pre_labels``."""
    vendor_samples, vendor_h, raw_h, raw_info = load_feeds(cut)
    has_raw = {k: bool(len(s) and s.notna().any()) for k, s in raw_h.items()}
    ident, ident_full = identity_table(vendor_h, raw_h)
    if pre_labels is None:
        ident = choose_v1(ident)
        chosen = ident
        status = {vk: ("confirmed" if vk in set(ident[ident.chosen].vendor_key) else
                       "unlabelled") for vk in L.VENDOR_KEYS}
    else:
        chosen, status = preorigin_label_table(pre_labels, raw_info["names"])
        lab = {(r.vendor_key, r.raw_id) for r in chosen.itertuples()}
        ident["chosen"] = [(a, b) in lab for a, b in zip(ident.vendor_key, ident.raw_id,
                                                          strict=True)]
    sensors = B.sensor_registry(raw_info["names"], chosen)
    unmapped_vendor = [k for k in L.VENDOR_KEYS if k not in set(sensors.vendor_key)]
    ambiguous = {vk for vk, s in status.items() if s == "ambiguous"}
    # envelope (pre-origin vendor data in both runs)
    samples = []
    for k in B.ENVELOPE_KEYS:
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
    # per sensor: catalogue, daily, monthly
    daily_rows, monthly_rows, step_rows, stitch_rows, sig = [], [], [], [], {}
    per_sensor = {}
    for srow in sensors.to_dict("records"):
        sid, rid, vk = srow["sensor_id"], srow["raw_id"], srow["vendor_key"]
        hr = has_raw.get(rid, False)
        hv = bool(vk) and bool(vendor_h[vk].notna().any())
        if not hr and not hv:
            continue
        rec = {"sensor": srow, "has_raw": hr, "has_vendor": hv}
        segs, stitches = [], []
        if vk and hr:
            idt = ident_full[(vk, rid)]
            segs, stitches = idt["segments"], idt["stitches"]
        if hr:
            rh = raw_h[rid]
            sgn = L.sign_series(rh.index, segs) if segs else np.ones(len(rh))
            ev_r, xc_r, hf_r = L.detect(rh * sgn, det)
            rec["sign_inverted_hours"] = int((sgn < 0).sum())
            rec["xc_r"], rec["hf_r"], rec["rh"] = xc_r, hf_r, rh
        else:
            ev_r = pd.DataFrame(columns=EMPTY_EVENTS)
        ev_r.insert(0, "feed", "raw")
        ev_r.insert(0, "sensor_id", sid)
        rec["raw_events"] = ev_r
        if hv:
            vh = vendor_h[vk]
            ev_v, _xv_c, hf_v = L.detect(vh, det)
            xv = vh.where(hf_v != 9)
            ev_v.insert(0, "feed", "vendor")
            ev_v.insert(0, "sensor_id", sid)
            rec["vendor_events"] = ev_v
            rec["xv"], rec["hf_v"] = xv, hf_v
        for s in B.classify_stitches(stitches, ev_r):
            stitch_rows.append({"sensor_id": sid, "vendor_key": vk, **s})
        cands = []
        if hv:
            cands.append(L.sigma_day_preorigin(L.daily_from_hourly(
                rec["xv"], rec["hf_v"].where(rec["hf_v"] != 9, 0))))
        if hr:
            cands.append(L.sigma_day_preorigin(L.daily_from_hourly(rec["xc_r"], rec["hf_r"])))
        sig[sid] = next((s for s in cands if np.isfinite(s)), float("nan"))
        per_sensor[sid] = rec
    pooled = float(np.nanmedian([v for v in sig.values() if np.isfinite(v)]))
    sig_src = {k: ("pre-origin" if np.isfinite(v) else "pooled median") for k, v in sig.items()}
    sig = {k: (v if np.isfinite(v) else pooled) for k, v in sig.items()}
    for sid, rec in per_sensor.items():
        srow, vk = rec["sensor"], rec["sensor"]["vendor_key"]
        rec["mr"] = rec["mv"] = None
        if rec["has_raw"]:
            dr = L.daily_from_hourly(rec["xc_r"], rec["hf_r"], raw=rec["rh"], sigma_day=sig[sid])
            daily_rows.append(B.daily_frame(
                srow, "raw", dr, f"raw:{srow['raw_id']}",
                "raw agency feed, step catalogue applied (removed steps -> flag 1)", version))
            ev_r = rec["raw_events"]
            mr = L.monthly_from_daily(dr, ev_r.assign(event_id=[f"{sid}:raw:{i}" for i in
                                                                range(len(ev_r))]),
                                      det, sig[sid])
            mr["feed"] = "raw"
            rec["mr"] = mr
        if rec["has_vendor"]:
            dvd = L.daily_from_hourly(rec["xv"], rec["hf_v"].where(rec["hf_v"] != 9, 0),
                                      raw=rec["xv"], sigma_day=sig[sid])
            daily_rows.append(B.daily_frame(srow, "vendor", dvd, f"vendor:{vk}",
                                            "vendor feed as delivered (zeros = missing)",
                                            version))
            ev_v = rec["vendor_events"]
            ev_r = rec["raw_events"]
            sus = [ev_v[ev_v.kind == "step"][["t_before", "t_after"]]]
            st = [s for s in stitch_rows if s["sensor_id"] == sid
                  and s["classification"] == "removed_real_movement"]
            if st:
                sus.append(pd.DataFrame(st)[["t_before", "t_after"]])
            rs = ev_r[ev_r.kind == "step"]
            kept = [r for r in rs.itertuples()
                    if not any(B.overlaps(r.t_before, r.t_after, s["t_before"], s["t_after"])
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
            rec["mv"] = mv
    # vendor series without a label: kept, every row flag 9 (as v1)
    defective = {}
    for vk in unmapped_vendor:
        s = vendor_samples[vk].dropna()
        if s.empty:
            continue
        g = s.groupby(s.index.floor("D"))
        d = pd.DataFrame({"date": g.mean().index, "value": g.mean().to_numpy(),
                          "n_hours": g.size().to_numpy(), "flag": 9})
        d["raw_value"] = d["value"]
        d["sd"] = np.nan
        nominal = {"TKJS_283_29": "TKJS(3)"}.get(vk, vk)
        srow = sensors[sensors.sensor_id == nominal].to_dict("records")
        srow = srow[0] if srow else {"sensor_id": nominal, "pile": "", "depth_m": -1}
        defective[vk] = nominal
        why = note_283(status_283) if vk == "TKJS_283_29" else \
            "vendor series without a label: defective (flag 9, not used)"
        daily_rows.append(B.daily_frame(srow, "vendor", d, f"vendor:{vk}", why, version))
    # primary monthly (v1's rule)
    for sid, rec in per_sensor.items():
        srow = rec["sensor"]
        mr = rec["mr"].set_index("month") if rec["mr"] is not None else None
        mv = rec["mv"].set_index("month") if rec["mv"] is not None else None
        months = sorted((set(mr.index) if mr is not None else set())
                        | (set(mv.index) if mv is not None else set()))
        for m in months:
            r = mr.loc[m] if (mr is not None and m in mr.index) else None
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
    # ambiguous pre-origin labels: every vendor row of the sensor is suspect (flag 9)
    amb_sensors = sorted(sensors[sensors.vendor_key.isin(sorted(ambiguous))].sensor_id)
    if amb_sensors:
        note = "; pre-origin label of the vendor series ambiguous (label_confirm): flag 9"
        mvx = monthly.sensor_id.isin(amb_sensors) & (monthly.feed == "vendor")
        monthly.loc[mvx & (monthly.flag != 8), "method"] += note
        monthly.loc[mvx & (monthly.flag != 8), "flag"] = 9
        monthly.loc[monthly.sensor_id.isin(amb_sensors) & (monthly.vendor_flag != 8),
                    "vendor_flag"] = 9
    # monthly-mean increments within this run (before the daily override, see S2r4)
    mm_inc = T.monthly_mean_increments(daily, monthly)
    if amb_sensors:
        dvx = daily.sensor_id.isin(amb_sensors) & (daily.feed == "vendor") & (daily.flag != 8)
        daily.loc[dvx, "method"] += note
        daily.loc[dvx, "flag"] = 9
    # steps
    for sid, rec in per_sensor.items():
        for key in ("raw_events", "vendor_events"):
            if key in rec and len(rec[key]):
                e = rec[key].copy()
                e["event_id"] = [f"{sid}:{e.feed.iloc[0]}:{i}" for i in range(len(e))]
                step_rows.append(e[e.kind != "noise"])
    steps = pd.concat(step_rows, ignore_index=True) if step_rows else \
        pd.DataFrame(columns=["sensor_id", "feed", *EMPTY_EVENTS, "event_id"])
    stitches = pd.DataFrame(stitch_rows)
    steps = B.annotate_steps(steps, stitches)
    steps["product_version"] = version
    # printed merge
    printed = L.read_printed()
    if variant == "preorigin":
        printed = printed[printed.month <= ORIGIN]
        last_month = ORIGIN
    else:
        last_month = str(printed.loc[printed["verified"].astype(bool), "month"].max())
    merged = B.merge_printed(monthly, printed, sensors, sig, version, last_month)
    cum = B.cumulative_all(merged, printed, sig, version)
    return {"variant": variant, "daily": daily, "monthly": monthly, "merged": merged,
            "cum": cum, "steps": steps, "stitches": stitches, "identity": ident,
            "envelope": env, "recall": recall, "sensors": sensors, "sig": sig,
            "sig_src": sig_src, "per_sensor": per_sensor, "printed": printed,
            "last_month": last_month, "mm_inc": mm_inc, "status": status,
            "ambiguous_sensors": amb_sensors, "defective": defective,
            "unmapped_vendor": unmapped_vendor, "raw_info": raw_info}


# ------------------------------------------------------------------------------- assembly
def _take(df: pd.DataFrame, col: str, pre: bool) -> pd.DataFrame:
    m = df[col].astype(str) <= ORIGIN if col == "month" else \
        pd.to_datetime(df[col]) < CUT
    return df[m] if pre else df[~m]


def assemble(pre: dict, full: dict, version: str) -> dict:
    out = {}
    for k, col in (("daily", "date"), ("monthly", "month"), ("merged", "month"),
                   ("cum", "month"), ("steps", "month")):
        a = _take(pre[k], col, True).assign(variant="preorigin")
        b = _take(full[k], col, False).assign(variant="full")
        t = pd.concat([a, b], ignore_index=True)
        t["product_version"] = version
        out[k] = t
    mi_a = pre["mm_inc"][pre["mm_inc"].month <= ORIGIN].assign(variant="preorigin")
    mi_b = full["mm_inc"][full["mm_inc"].month > ORIGIN].assign(variant="full")
    out["mm_inc_raw"] = pd.concat([mi_a, mi_b], ignore_index=True)
    for k in ("stitches", "identity"):
        out[k] = pd.concat([pre[k].assign(variant="preorigin"),
                            full[k].assign(variant="full")], ignore_index=True)
    out["envelope"], out["recall"] = full["envelope"], full["recall"]
    return out


def sort_tables(t: dict) -> dict:
    keys = {"daily": ["sensor_id", "feed", "date"], "monthly": ["sensor_id", "month"],
            "merged": ["sensor_id", "month"], "cum": ["sensor_id", "month"],
            "steps": ["sensor_id", "feed", "t_after"]}
    for k, by in keys.items():
        t[k] = t[k].sort_values(by, kind="mergesort").reset_index(drop=True)
    return t


def sd_labels(t: dict, cov: dict) -> None:
    """Plan 4.7: every sd column labelled; the check behind each label is in validation.json."""
    c = cov.get("monthly_month_end_vs_test_A", {})
    lab_m = (f"{SD_STATUS}: nominal 95 % interval covers {c.get('coverage_95', float('nan')):.2f}"
             f" of {c.get('n', 0)} test-A differences (design-time units)")
    t["monthly"]["sd_status"] = np.where(t["monthly"]["sd"].notna(), lab_m, "no sd")
    t["merged"]["sd_status"] = np.where(t["merged"]["sd"].notna(),
                                        f"{SD_STATUS}: see validation.json uncertainty",
                                        "no sd")
    for k, why in (("daily", "no independent daily truth"),
                   ("cum", "propagated from the increments' uncalibrated sd"),
                   ("steps", "sigma_fill of the pre-origin envelope; 3 agency records only")):
        t[k]["sd_status"] = np.where(t[k]["sd"].notna(), f"{SD_STATUS}: {why}", "no sd")


# ------------------------------------------------------------------------------- checks
def coverage(units: pd.DataFrame) -> dict:
    """Coverage of the nominal 90/95 % intervals of printed - feed on test-A units:
    |diff| <= z * sqrt(sd_feed^2 + sd_round^2) (both in mm)."""
    if not len(units):
        return {"n": 0}
    s = np.sqrt(units["sd_feed_mm"] ** 2 + units["sd_round_mm"] ** 2)
    z = units["diff_mm"].abs() / s
    k95 = float(np.quantile(z, 0.95) / 1.959964)
    return {"n": int(len(units)), "coverage_90": float((z <= 1.644854).mean()),
            "coverage_95": float((z <= 1.959964).mean()),
            "median_sd_mm": float(s.median()), "median_abs_diff_mm": float(
                units["diff_mm"].abs().median()),
            "sd_scale_for_95_after_the_fact": k95,
            "calibrated": False,
            "note": "design-time units (test A, integrated reports 2024-01..2025-04, month-end "
                    "convention); the factor is reported, never applied"}


def a_units_with_sd(monthly: pd.DataFrame, printed: pd.DataFrame) -> pd.DataFrame:
    pv = printed[printed["verified"].astype(bool) & printed["month_cm"].notna()]
    j = pv[["sensor_id", "month", "month_cm", "month_round_sd_cm"]].merge(
        monthly[["sensor_id", "month", "value", "sd", "flag"]], on=["sensor_id", "month"])
    j = j[j.flag.isin([0, 1]) & j.value.notna()]
    return pd.DataFrame({"sensor_id": j.sensor_id, "month": j.month,
                         "diff_mm": (j.month_cm - j.value) * 10.0,
                         "sd_feed_mm": j.sd * 10.0, "sd_round_mm": j.month_round_sd_cm * 10.0})


def compare_preorigin(pre: dict, v1dir: Path) -> dict:
    """Pre-origin rows of the pre-origin run against v1's rows of the same months (after the
    fact; quantifies what the post-origin dependence changed)."""
    out = {}
    v1m = pd.read_parquet(v1dir / "monthly.parquet")
    a = pre["monthly"][pre["monthly"].month <= ORIGIN]
    b = v1m[v1m.month <= ORIGIN]
    j = a.merge(b, on=["sensor_id", "month"], how="outer", suffixes=("_v2", "_v1"),
                indicator=True)
    both = j[j["_merge"] == "both"]
    dv = (both.value_v2 - both.value_v1).abs()
    diff_flag = both.flag_v2 != both.flag_v1
    diff_val = ~((dv <= 1e-9) | (both.value_v2.isna() & both.value_v1.isna()))
    per = {}
    for sid, g in both.groupby("sensor_id"):
        gf = g.flag_v2 != g.flag_v1
        per[sid] = {"n": int(len(g)), "flag_changed": int(gf.sum()),
                    "value_changed": int((~(((g.value_v2 - g.value_v1).abs() <= 1e-9)
                                           | (g.value_v2.isna() & g.value_v1.isna()))).sum()),
                    "flag_transitions": {f"{int(x)}->{int(y)}": int(n) for (x, y), n in
                                         g[gf].groupby(["flag_v1", "flag_v2"]).size().items()},
                    "feed_changed": int((g.feed_v2 != g.feed_v1).sum())}
    out["monthly"] = {"n_rows_both": int(len(both)),
                      "only_v1": int((j["_merge"] == "right_only").sum()),
                      "only_v2": int((j["_merge"] == "left_only").sum()),
                      "flag_changed": int(diff_flag.sum()), "value_changed": int(diff_val.sum()),
                      "per_sensor": per,
                      "only_v1_by_sensor": j[j["_merge"] == "right_only"].groupby(
                          "sensor_id").size().astype(int).to_dict(),
                      "only_v2_by_sensor": j[j["_merge"] == "left_only"].groupby(
                          "sensor_id").size().astype(int).to_dict()}
    v1d = pd.read_parquet(v1dir / "daily.parquet")
    da = pre["daily"][pd.to_datetime(pre["daily"]["date"]) < CUT]
    db = v1d[pd.to_datetime(v1d["date"]) < CUT]
    jd = da.merge(db, on=["sensor_id", "feed", "date"], how="outer", suffixes=("_v2", "_v1"),
                  indicator=True)
    bd = jd[jd["_merge"] == "both"]
    dval = ~(((bd.value_v2 - bd.value_v1).abs() <= 1e-9) | (bd.value_v2.isna()
                                                            & bd.value_v1.isna()))
    out["daily"] = {"n_rows_both": int(len(bd)),
                    "only_v1": int((jd["_merge"] == "right_only").sum()),
                    "only_v2": int((jd["_merge"] == "left_only").sum()),
                    "flag_changed": int((bd.flag_v2 != bd.flag_v1).sum()),
                    "value_changed": int(dval.sum()),
                    "value_changed_by_sensor_feed": bd[dval].groupby(
                        ["sensor_id", "feed"]).size().rename("n").reset_index()
                    .to_dict("records")}
    v1s = pd.read_csv(v1dir / "steps.csv")
    sa = pre["steps"][pre["steps"].month <= ORIGIN]
    sb = v1s[v1s.month <= ORIGIN]
    key = ["sensor_id", "feed", "kind", "t_before", "t_after"]
    js = sa[key].merge(sb[key], on=key, how="outer", indicator=True)
    out["steps"] = {"n_v2": int(len(sa)), "n_v1": int(len(sb)),
                    "same_events": int((js["_merge"] == "both").sum()),
                    "only_v1": js[js["_merge"] == "right_only"][key].to_dict("records"),
                    "only_v2": js[js["_merge"] == "left_only"][key].to_dict("records")}
    tk = both[(both.sensor_id == "TKJS(2)") & (both.month >= "2021-08")]
    out["tkjs2_2021_08_to_2022_12"] = {
        "n": int(len(tk)), "v1_flags": tk.flag_v1.value_counts().astype(int).to_dict(),
        "v2_flags": tk.flag_v2.value_counts().astype(int).to_dict()}
    return out


def compare_full_with_v1(full: dict, v1dir: Path) -> dict:
    """The full run must reproduce v1's tables (value for value)."""
    res = {}
    pairs = (("daily", "daily.parquet"), ("monthly", "monthly.parquet"),
             ("cum", "cumulative_monthly.parquet"), ("steps", "steps.csv"),
             ("stitches", "vendor_stitches.csv"), ("envelope", "envelope.csv"),
             ("recall", "injected_recall.csv"), ("identity", "identity.csv"))
    for k, f in pairs:
        a = full[k].copy()
        b = pd.read_parquet(v1dir / f) if f.endswith(".parquet") else pd.read_csv(v1dir / f)
        a = a.drop(columns=[c for c in ("product_version",) if c in a.columns])
        b = b.drop(columns=[c for c in ("product_version",) if c in b.columns])
        method_diff = None
        if k == "daily":     # v2 rewrites only the defective TKJS_283_29 justification
            md = a["method"].reset_index(drop=True) != b["method"].reset_index(drop=True) \
                if len(a) == len(b) else None
            if md is not None:
                method_diff = {"n_rows": int(md.sum()), "source_ids": sorted(set(
                    a.loc[md.to_numpy(), "source_ids"]))}
            a, b = a.drop(columns="method"), b.drop(columns="method")
        if f.endswith(".csv"):
            a = pd.read_csv(io.StringIO(a.to_csv(index=False)))
        cols = [c for c in b.columns if c in a.columns]
        missing = sorted(set(b.columns) - set(a.columns))
        try:
            pd.testing.assert_frame_equal(a[cols].reset_index(drop=True),
                                          b[cols].reset_index(drop=True), check_dtype=False,
                                          rtol=1e-9, atol=1e-12)
            res[k] = {"identical": True, "rows": int(len(a)), "missing_cols": missing}
            if method_diff is not None:
                res[k]["method_differs_only"] = method_diff
        except AssertionError as exc:
            res[k] = {"identical": False, "rows": [int(len(a)), int(len(b))],
                      "error": str(exc)[:600], "missing_cols": missing}
    mv1 = pd.read_parquet(v1dir / "merged_monthly.parquet")
    a = full["merged"]
    cols = [c for c in a.columns if c in mv1.columns and c != "product_version"]
    try:
        pd.testing.assert_frame_equal(a[cols].reset_index(drop=True),
                                      mv1[cols].reset_index(drop=True), check_dtype=False,
                                      rtol=1e-9, atol=1e-12)
        res["merged"] = {"identical": True, "rows": int(len(a)), "compared_cols": len(cols)}
    except AssertionError as exc:
        res["merged"] = {"identical": False, "error": str(exc)[:600]}
    res["all_identical"] = all(v.get("identical") for v in res.values() if isinstance(v, dict))
    return res


def b_scope(full: dict, val_b: dict) -> dict:
    """Every mapped sensor, whether test B could test it, and why not (plan: 'every mapped
    pile')."""
    printed = full["printed"]
    pv = printed[printed["verified"].astype(bool) & printed["month_cm"].notna()]
    rows = []
    tested = {b["sensor_id"]: b for b in val_b["units"]}
    for s in full["sensors"].to_dict("records"):
        sid = s["sensor_id"]
        n_pr = int((pv.sensor_id == sid).sum())
        b = tested.get(sid)
        why = ""
        if n_pr == 0:
            why = "never printed in the integrated reports (2024-01..2026-01)"
        elif b is not None and not b["testable"]:
            why = f"printed {n_pr} months, {b['n']} overlap a feed month (< 5)"
        rows.append({"sensor_id": sid, "pile": s["pile"], "depth_m": s["depth_m"],
                     "on_grid": bool(s["on_grid"]), "printed_months": n_pr,
                     "testable": bool(b is not None and b["testable"]),
                     "r": b["r"] if b is not None else float("nan"),
                     "why_untestable": why})
    return {"sensors": rows, "n_mapped": len(rows),
            "n_testable": int(sum(r["testable"] for r in rows)),
            "untestable": [f"{r['sensor_id']} ({r['why_untestable']})" for r in rows
                           if not r["testable"]]}


def tkjs283_diagnostic(full: dict) -> dict:
    """Full-period after-the-fact check of the defective vendor series TKJS_283_29 against
    the raw TKJS(3) feed: correlation of daily changes on common days."""
    v = L.vendor_raw("TKJS_283_29").dropna()
    vd = v.groupby(v.index.floor("D")).mean()
    rec = full["per_sensor"].get("TKJS(3)")
    if rec is None or not rec["has_raw"]:
        return {"n_days": 0}
    rh = rec["rh"].dropna()
    g = rh.groupby(rh.index.floor("D"))
    rd = g.mean()[g.size() >= L.DAY_MIN_HOURS]
    j = pd.concat([vd.rename("v"), rd.rename("r")], axis=1).dropna()
    j = j.asfreq("D")
    dj = j.diff().dropna()
    return {"n_common_days": int(j.notna().all(axis=1).sum()),
            "n_daily_change_pairs": int(len(dj)),
            "r_daily_change": L.pearson(dj.v, dj.r) if len(dj) >= 3 else float("nan"),
            "period": [str(j.dropna().index.min().date()), str(j.dropna().index.max().date())]
            if len(j.dropna()) else None,
            "vendor_zero_samples": int((L.vendor_raw("TKJS_283_29").isna()).sum())}


def c_after_the_fact(pre: dict, full: dict, version: str) -> dict:
    """Test C's statistic on v2's rows: C1 units against the pre-origin run, C2 units against
    the full run. After the fact (the units were scored on v1); look logged by main()."""
    prod = guard.open_product(PRINTED_V1, "validator")
    cy = prod.read_table("printed_monthly.parquet")
    units = T.printed_units(cy, last_month="2099-12")
    mi_pre = pre["mm_inc"]
    mi_full = full["mm_inc"]
    c1 = T.score(units[(units.month >= "2019-01") & (units.month <= "2022-12")], mi_pre)
    c2 = T.score(units[(units.month >= "2023-01") & (units.month <= "2023-12")], mi_full)
    by_year = {}
    u1 = c1["units"]
    u1 = u1[u1.testable]
    for y, g in u1.groupby(u1.month.str[:4]):
        by_year[y] = {"n": int(len(g)), "n_ok": int((g.diff_mm.abs() <= T.TOL_MM + 1e-9).sum())}
    u = pd.concat([c1["units"].assign(test="C1_on_v2_preorigin_run"),
                   c2["units"].assign(test="C2_on_v2_full_run")], ignore_index=True)
    u["product_version"] = version
    return {"C1_on_v2": c1["summary"], "C1_by_year": by_year, "C2_on_v2": c2["summary"],
            "units": u, "printed_product": rel_to_repo(PRINTED_V1),
            "note": "after the fact: the units were scored on v1 under S2r2.c_fresh_cy007001; "
                    "the NTUH labels of the pre-origin run were chosen on the same printed "
                    "2019-2022 values (label_confirm), so C1 on v2 is not a test"}


def printed_cy_columns() -> pd.DataFrame:
    prod = guard.open_product(PRINTED_V2, "validator")
    p = prod.read_table("printed_monthly.parquet",
                        columns=["sensor_id", "month", "value", "report", "column",
                                 "vintage_month", "flag"])
    p = p[p.flag == 0]
    return p.rename(columns={"value": "printed_cy007001_mm", "report": "printed_cy007001_report",
                             "column": "printed_cy007001_column",
                             "vintage_month": "printed_cy007001_vintage"}).drop(columns="flag")


def mm_inc_table(mi: pd.DataFrame, version: str) -> pd.DataFrame:
    m = mi.copy()
    m["value"] = m["inc_mm"] / 10.0
    m["sd"] = np.nan
    m["method"] = np.where(m["flag"] == 8, "no feed with >= 20 full days in both months",
                           "difference of consecutive monthly means of full days "
                           "(c_test.monthly_mean_increments); cm")
    m["product_version"] = version
    m["source_ids"] = m["feed"].where(m["feed"] != "", "")
    m["sd_status"] = "no sd"
    return m


# ------------------------------------------------------------------------------- build
def compute(version: str, debug_v1_labels: bool = False) -> dict:
    t0 = dt.datetime.now()
    log("full run ...")
    lab_file = None if debug_v1_labels else json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    s283 = lab_file["status"].get("TKJS_283_29") if lab_file else None
    full = run_variant(version, "full", None, None, status_283=s283)
    if debug_v1_labels:
        lab = {"status": full["status"],
               "labels": {r.vendor_key: r.raw_id.replace("BM_", "")
                          for r in full["identity"][full["identity"].chosen].itertuples()},
               "nominal": {}}
        lab["status"] = {k: ("confirmed" if k in lab["labels"] else "defective")
                         for k in L.VENDOR_KEYS}
    else:
        lab = lab_file
    log("pre-origin run ...", {k: (lab["labels"].get(k), lab["status"].get(k))
                               for k in L.VENDOR_KEYS})
    pre = run_variant(version, "preorigin", CUT, lab, status_283=s283)
    # checks: inputs of the pre-origin run end before the cut; shared parameters identical
    checks = {}
    for k in ("daily",):
        mx = pd.to_datetime(pre[k]["date"]).max()
        checks["preorigin_last_day"] = str(mx.date())
        if mx >= CUT:
            raise SystemExit(f"pre-origin run has a day after the cut: {mx}")
    for k in ("monthly", "merged", "cum"):
        mx = pre[k]["month"].max()
        checks[f"preorigin_last_month_{k}"] = mx
        if mx > ORIGIN:
            raise SystemExit(f"pre-origin {k} has a month after the origin: {mx}")
    st_after = pd.to_datetime(pre["steps"]["t_after"]).max() if len(pre["steps"]) else None
    checks["preorigin_last_step_t_after"] = str(st_after)
    if st_after is not None and st_after >= CUT:
        raise SystemExit(f"pre-origin step after the cut: {st_after}")
    pd.testing.assert_frame_equal(pre["envelope"], full["envelope"])
    pd.testing.assert_frame_equal(pre["recall"], full["recall"])
    checks["envelope_identical_in_both_runs"] = True
    same_sig = {s: bool(abs(pre["sig"][s] - full["sig"][s]) < 1e-12)
                for s in pre["sig"] if s in full["sig"]}
    checks["sigma_day_same_in_both_runs"] = same_sig
    t = sort_tables(assemble(pre, full, version))
    t["mm_inc"] = mm_inc_table(t.pop("mm_inc_raw"), version)
    # printed 007001 (deep_benchmarks_printed/v2 primary printing) carried alongside
    cyc = printed_cy_columns()
    t["merged"] = t["merged"].merge(cyc, on=["sensor_id", "month"], how="left")
    # mapping: v1's table plus the pre-origin labels
    full_map = B.mapping_table(full["sensors"], full["merged"], full["monthly"],
                               full["printed"], full["identity"],
                               dict(zip(full["sensors"].sensor_id, full["sensors"].vendor_key,
                                        strict=True)), full["defective"])
    pre_of = {r.sensor_id: r.vendor_key for r in pre["sensors"].itertuples() if r.vendor_key}
    full_map["vendor_key_preorigin"] = full_map["sensor_id"].map(pre_of).fillna("")
    full_map["preorigin_label_status"] = [
        lab["status"].get(k, "") if k else "" for k in full_map["vendor_key_preorigin"]]
    full_map["preorigin_label_method"] = [
        lab.get("method", {}).get(k, "") if k else "" for k in full_map["vendor_key_preorigin"]]
    full_map.loc[full_map.vendor_key == "TKJS_283_29", "domain"] = note_283(s283)
    t["mapping"] = full_map
    val = {"variants": {"cut": str(CUT), "labels_evidence": rel_to_repo(LABELS_JSON),
                        "labels_evidence_sha256": sha256_file(LABELS_JSON)
                        if LABELS_JSON.exists() else None,
                        "preorigin_status": pre["status"],
                        "preorigin_ambiguous_sensors": pre["ambiguous_sensors"],
                        "preorigin_defective": pre["defective"],
                        "full_defective": full["defective"]},
           "checks": checks,
           "full_run_reproduces_v1": compare_full_with_v1(full, V1),
           "preorigin_rows_vs_v1": compare_preorigin(pre, V1),
           "sigma_day_mm": {k: {"value": round(v, 4), "source": full["sig_src"][k]}
                            for k, v in full["sig"].items()},
           "raw_feed_info": full["raw_info"],
           "envelope": full["envelope"].to_dict("records"),
           "recall": B.recall_summary(full["recall"]),
           "steps_summary_full": B.steps_summary(full["steps"], full["stitches"]),
           "steps_summary_preorigin": B.steps_summary(pre["steps"], pre["stitches"])
           if len(pre["steps"]) else {},
           "last_month": full["last_month"]}
    res = {"tables": t, "validation": val, "full": full, "pre": pre, "labels": lab}
    val["development_run_with_v1_labels"] = bool(debug_v1_labels)
    # acceptance (plan A and B, unchanged) on the full run
    acc = B.acceptance(full["merged"], full["monthly"], full["printed"], full["sensors"],
                       full["per_sensor"], full["sig"])
    val.update(acc)
    val["B_scope"] = b_scope(full, acc["B"])
    val["uncertainty"] = {
        "monthly_month_end_vs_test_A": coverage(a_units_with_sd(full["monthly"],
                                                                full["printed"])),
        "merged_printed_rows": "sd = rounding of the printed decimal and the feed noise, "
                               "uncalibrated (the printed and feed values are the two sides "
                               "of the same test-A comparison)",
        "monthly_mean_increments": "no sd",
        "daily": "sigma_day scaled by the hour count; uncalibrated (no independent daily "
                 "truth)",
        "cumulative": "propagated by weighted least squares from uncalibrated sd; uncalibrated",
        "steps": "sigma_fill(d) of the pre-origin envelope; uncalibrated (3 agency records "
                 "only, max |catalogue - agency| 0.08 mm, S2_evidence/r3)"}
    val["tkjs283_full_period"] = tkjs283_diagnostic(full)
    c = c_after_the_fact(pre, full, version)
    t["c_units"] = c.pop("units")
    val["C_after_the_fact"] = c
    sd_labels(t, val["uncertainty"])
    val["runtime_s"] = (dt.datetime.now() - t0).total_seconds()
    return res


def write_tables(t: dict, d: Path) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for k, f in (("daily", "daily.parquet"), ("monthly", "monthly.parquet"),
                 ("merged", "merged_monthly.parquet"), ("cum", "cumulative_monthly.parquet"),
                 ("mm_inc", "monthly_mean_increments.parquet"),
                 ("c_units", "test_c_units_after_the_fact.parquet")):
        if k in t:
            t[k].to_parquet(d / f, index=False)
            names.append(f)
    for k, f in (("steps", "steps.csv"), ("stitches", "vendor_stitches.csv"),
                 ("identity", "identity.csv"), ("mapping", "mapping.csv"),
                 ("envelope", "envelope.csv"), ("recall", "injected_recall.csv")):
        t[k].to_csv(d / f, index=False)
        names.append(f)
    return names


def card(res: dict, name: str) -> str:
    v = res["validation"]
    t = res["tables"]
    a, b = v["A_primary"], v["B"]
    cov = v["uncertainty"]["monthly_month_end_vs_test_A"]
    c1, c2 = v["C_after_the_fact"]["C1_on_v2"], v["C_after_the_fact"]["C2_on_v2"]
    lines = [f"# deep_benchmarks/{name}: deep benchmarks (深層樁) as a first-class series", "",
             "WP S2 round 4 (2026-10-12), dataprep_staging/s2_deep_benchmarks/build_v2.py; "
             f"rules declared in results/dataprep/preregistration/{DECL_ID}.json. Supersedes "
             "deep_benchmarks/v1, whose pre-origin rows depended on post-origin observations.",
             "", "## What changed from v1", "",
             "- Rows dated up to 2022-12 come from a pre-origin run (feeds cut before "
             "2023-01-01, vendor labels from pre-origin data only, "
             "S2_evidence/r4/preorigin_labels.json); later rows from the full run, which "
             "reproduces v1. Column `variant` says which. Pre-origin label status: "
             + ", ".join(f"{k} {res['labels'].get('status', {}).get(k)}"
                         for k in L.VENDOR_KEYS) + ".",
             "- Every sd is labelled `uncalibrated` (column `sd_status`; plan 4.7). The month-"
             f"end increments' nominal 95 % interval covers {cov.get('coverage_95', 0):.2f} of "
             f"{cov.get('n')} test-A differences.",
             f"- TKJS_283_29: {note_283(res['labels'].get('status', {}).get('TKJS_283_29'))}"
             ".",
             "- merged_monthly carries the printed 007001 value of deep_benchmarks_printed/v2 "
             "(one primary printing per sensor-month).", "",
             "## Tables", "",
             "- daily.parquet, monthly.parquet, merged_monthly.parquet, "
             "cumulative_monthly.parquet, monthly_mean_increments.parquet, steps.csv: as v1, "
             "plus `variant` and `sd_status`. Cumulative values are anchored at 0 at the end of "
             "2022-12 in both runs; a difference across the origin mixes two solves.",
             "- test_c_units_after_the_fact.parquet: test C's statistic on v2's rows (after the "
             "fact; not a test).",
             "- vendor_stitches.csv, identity.csv, mapping.csv: both runs (no month column, so a "
             "time slice cannot read them); envelope.csv, injected_recall.csv: pre-origin "
             "envelope (identical in both runs).", "",
             "## Flags", "", "0 observed; 1 a removed sensor step inside the interval; 2 month "
             "end bridged or partial day; 8 missing; 9 suspect (vendor stitch that removed "
             "movement, unremoved step, spike hours, a vendor series without a pre-origin "
             "label, the defective vendor TKJS_283_29).", "",
             "## Gate (plan WP S2: A and B, unchanged)", "",
             f"A {a.get('n_ok')}/{a.get('n')} units within 2 mm (max "
             f"{a.get('max_abs_mm', float('nan')):.2f} mm), pass {a.get('pass')}; B min r "
             f"{b.get('min_r', float('nan')):.3f} over {b.get('n_testable')} printed piles "
             f"({v['B_scope']['n_testable']} of {v['B_scope']['n_mapped']} mapped sensors "
             f"testable), pass {b.get('pass')}. Gate passed: {v['gate_passed']}. The loader "
             "therefore refuses validator, scoring, calibration, datum and baseline uses; use "
             "deep_benchmarks_printed/v2 for those.", "",
             f"After the fact, C on v2's rows: C1 (2019-2022, pre-origin run) {c1.get('n_ok')}/"
             f"{c1.get('n_testable')} within 2 mm; C2 (2023, full run) {c2.get('n_ok')}/"
             f"{c2.get('n_testable')}.", "",
             f"Rows: daily {len(t['daily'])}, monthly {len(t['monthly'])}, merged "
             f"{len(t['merged'])}, steps {len(t['steps'])}."]
    return "\n".join(lines) + "\n"


def manifest_for(d: Path, res: dict, preview: bool = False) -> dict:
    v = res["validation"]
    last = v["last_month"]
    inputs = [L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet" for k in L.VENDOR_KEYS]
    inputs += [*(L.OPEN_DIR / f for f in L.OPEN_FILES), L.OPEN_DIR / "SOURCE.json",
               L.PRINTED_CSV, L.PRINTED_REPORT, L.SITES_CSV,
               REPO / "data_fetch_api/new_measurements/wra_monthly_reports/SOURCE.json",
               L.GRID_NPZ, *PREREG_FILES, LABELS_JSON,
               PRINTED_V1 / "MANIFEST.json", PRINTED_V1 / "printed_monthly.parquet",
               PRINTED_V2 / "MANIFEST.json", PRINTED_V2 / "printed_monthly.parquet",
               V1 / "MANIFEST.json"]
    if preview:                      # development preview: inputs not yet written are skipped
        inputs = [p for p in inputs if p.exists()]
    a, b = v["A_primary"], v["B"]
    cov = v["uncertainty"]["monthly_month_end_vs_test_A"]
    c1 = v["C_after_the_fact"]["C1_on_v2"]
    notes = [
        f"Observed product 2016-06..{last}. Rows dated up to 2022-12 (variant 'preorigin') "
        "come from a run whose feeds end before 2023-01-01 and whose vendor labels use "
        "pre-origin data only; later rows (variant 'full') from the full run, which reproduces "
        f"deep_benchmarks/v1 (check: {v['full_run_reproduces_v1']['all_identical']}). The step "
        "envelope Q(d), the 1 mm floor and sigma_day use data up to 2022-12 only.",
        f"Gate (plan WP S2, declared in S2r2.deep_benchmarks_core, carried into {DECL_ID}): "
        f"A {a.get('n_ok')}/{a.get('n')} within 2 mm (pass {a.get('pass')}); B min r "
        f"{b.get('min_r', float('nan')):.3f} over {b.get('n_testable')} piles (pass "
        f"{b.get('pass')}; {v['B_scope']['n_testable']} of {v['B_scope']['n_mapped']} mapped "
        f"sensors testable); gate passed {v['gate_passed']}. A and B are design-time units.",
        "Uncertainty (plan 4.7): every sd is uncalibrated (column sd_status). The month-end "
        f"increments' nominal 95 % interval covers {cov.get('coverage_95', float('nan')):.2f} "
        f"of {cov.get('n')} test-A differences (design-time units); no sd is rescaled.",
        f"Test C (registered on v1) on v2's rows, after the fact: C1 {c1.get('n_ok')}/"
        f"{c1.get('n_testable')} within 2 mm. Not a test: the pre-origin NTUH labels were "
        "chosen on the same printed values.",
        "Role (plan 4.1): validator of twin subsidence 2023-26; never fitted, never in a gate, "
        "ladder decision, member choice or threshold (U7). Use increments, not levels. Deep "
        "piles that shape S1's 2022 path cannot be an independent check of S1 at those sites.",
        "Pre-origin uses only through the time slice (window_end <= 2022-12). With the gate "
        "failed the loader opens design_time_evidence (slice) and after_the_fact_analysis only.",
        "Supersedes deep_benchmarks/v1 (its pre-origin slice carries post-origin flags and "
        "labels). No WRA monthly report newer than 2026-01 exists (re-checked 2026-10-12, "
        "S2_evidence/r4).",
    ]
    gate = {"passed": bool(v["gate_passed"]),
            "criteria": "plan 3 WP S2: A printed vs feed monthly |diff| <= 2 mm after the step "
                        "catalogue (every unit, flags 0-1); B mapping r >= 0.85 for every "
                        "printed pile with >= 5 overlapping months (declared in "
                        f"{B.PREREG.name}; carried unchanged into {DECL_ID}.json)"}
    return manifest.build_manifest(
        d, product=PRODUCT, layer="L2", inputs=[rel_to_repo(p) for p in inputs],
        code=[rel_to_repo(p) for p in CODE],
        parameters={"gap_h": L.GAP_H, "merge_h": L.MERGE_H, "side_h": L.SIDE_H,
                    "min_side_pts": L.MIN_SIDE_PTS, "floor_mm": L.FLOOR_MM,
                    "q_level": L.Q_LEVEL, "durations_h": list(L.DURS_H),
                    "envelope_series": list(B.ENVELOPE_KEYS), "envelope_end": str(L.ORIGIN_END),
                    "envelope_draws_per_sensor_duration": 3000, "seed": 2026,
                    "day_min_hours": L.DAY_MIN_HOURS, "bridge_days": L.BRIDGE_DAYS,
                    "reference_month": ORIGIN, "last_month": last,
                    "identity_min_share": 0.98, "preorigin_cut": str(CUT),
                    "preorigin_labels": res["labels"].get("labels"),
                    "preorigin_label_status": res["labels"].get("status"),
                    "claude_api_tokens": 0},
        fit_window_end=last, validation_report="validation.json", gate=gate,
        time_slicing={"column": "month", "params_fit_window_end": ORIGIN, "encoding": "iso"},
        allowed_uses=["validator", "scoring_target", "after_the_fact_analysis", "datum_fit",
                      "calibration_target", "baseline", "design_time_evidence"],
        forbidden_uses=["selection", "model_design", "prior", "prior_width_input",
                        "initial_conditions", "idw_field", "forcing_frozen",
                        "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none (public benchmark station coordinates only; station contact "
                      "columns never read)",
        extra={"supersedes": "deep_benchmarks/v1", "notes": notes})


def log_looks(res: dict, dev: bool = False) -> None:
    from harness import prereg as pr
    pre = ("DEVELOPMENT RUN with v1's labels before the declaration (code test); "
           if dev else "")
    a_units = sorted({f"dbm_integrated_printed:{u['sensor_id']}"
                      for u in res["validation"]["A_units"]})
    pr.record_look(a_units, "deep_piles_integrated_printed_vs_feed:2024-01..2025-04", "S2",
                   "after_the_fact", test_id="S2r4.A_B_on_v2", prereg_id=DECL_ID,
                   note=pre + "S2 round 4: tests A and B recomputed on deep_benchmarks/v2's full run "
                        "(design-time units; same rules) and the coverage of the month-end "
                        "sd on the test-A units (plan 4.7)")
    sids = ["SJES", "TKJS(1)", "TKJS(2)", "TKJS(3)", "NTUH(1)", "NTUH(2)", "STES"]
    pr.record_look([f"dbm_cy007001_printed:{s}:pre" for s in sids],
                   "deep_piles_cy007001_printed_vs_feed:2019-01..2022-12", "S2",
                   "after_the_fact", test_id="S2r4.C_on_v2", prereg_id=DECL_ID,
                   note=pre + "S2 round 4: test C's statistic on v2's pre-origin rows (after the "
                        "fact; units scored on v1)")
    pr.record_look([f"dbm_cy007001_printed:{s}:2023" for s in sids],
                   "deep_piles_cy007001_printed_vs_feed:2023-01..2023-12", "S2",
                   "after_the_fact", test_id="S2r4.C_on_v2", prereg_id=DECL_ID,
                   note=pre + "S2 round 4: test C's statistic on v2's 2023 rows (full run = v1)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--debug-labels-v1", action="store_true")
    ap.add_argument("--no-log", action="store_true",
                    help="with --scratch only: do not log the looks (reproduction runs)")
    a = ap.parse_args(argv)
    if a.debug_labels_v1 and a.scratch is None:
        raise SystemExit("--debug-labels-v1 needs --scratch")
    if a.no_log and a.scratch is None:
        raise SystemExit("--no-log needs --scratch")
    need = [PRINTED_V2 / "MANIFEST.json"] if a.debug_labels_v1 else \
        PREREG_FILES + [LABELS_JSON, PRINTED_V2 / "MANIFEST.json"]
    for p in need:
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    if a.scratch is not None:
        version = f"{PRODUCT}/scratch"
    else:
        base = DATAPREP / PRODUCT
        ns = [int(q.name[1:]) for q in base.glob("v*") if q.name[1:].isdigit()]
        version = f"{PRODUCT}/v{max(ns, default=0) + 1}"
    res = compute(version, debug_v1_labels=a.debug_labels_v1)
    t, val = res["tables"], res["validation"]
    probe = {"daily": t["daily"], "monthly": t["monthly"], "merged": t["merged"],
             "cum": t["cum"], "steps": t["steps"], "last_month": val["last_month"]}
    checks = {"table_checks": B.table_checks(probe), "date_checks": B.date_checks(probe)}
    checks["table_checks"]["mm_inc"] = F.validate_long_table(t["mm_inc"])
    dup = {k: int(t[k].duplicated(by).sum()) for k, by in
           (("monthly", ["sensor_id", "month"]), ("merged", ["sensor_id", "month"]),
            ("cum", ["sensor_id", "month"]), ("daily", ["sensor_id", "feed", "date"]),
            ("mm_inc", ["sensor_id", "month"]))}
    checks["duplicate_keys"] = dup
    bad = {k: v for k, v in checks["table_checks"].items() if v}
    if bad or checks["date_checks"] or any(dup.values()):
        raise SystemExit(f"table/date checks failed: {bad} {checks['date_checks']} {dup}")
    val.update(checks)
    val["preregistration"] = [{"path": rel_to_repo(p), "sha256": sha256_file(p)}
                              for p in PREREG_FILES if p.exists()]
    if not a.no_log:
        log_looks(res, dev=a.debug_labels_v1)
    if a.scratch is not None:
        write_tables(t, a.scratch)
        (a.scratch / "validation.json").write_text(json.dumps(
            val, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        (a.scratch / "README.md").write_text(card(res, "scratch"), encoding="utf-8")
        mp = manifest_for(a.scratch, res, preview=True)
        (a.scratch.parent / f"{a.scratch.name}.MANIFEST.preview.json").write_text(
            json.dumps(mp, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        log(json.dumps({"full_run_reproduces_v1": val["full_run_reproduces_v1"]["all_identical"],
                        "preorigin_vs_v1": val["preorigin_rows_vs_v1"]["monthly"]
                        ["flag_changed"]}, default=str))
        log(f"scratch tables written to {a.scratch}")
        return 0
    d = manifest.new_version_dir(PRODUCT)
    if f"{PRODUCT}/{d.name}" != version:
        raise SystemExit(f"version race: computed for {version}, got {d}")
    write_tables(t, d)
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                  default=str) + "\n", encoding="utf-8")
    (d / "README.md").write_text(card(res, d.name), encoding="utf-8")
    m = manifest_for(d, res)
    errs = guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit(f"time_slicing errors (manifest not written): {errs}")
    manifest.write_manifest(d, m)
    log(f"wrote {d}")
    log(json.dumps({"gate_passed": val["gate_passed"],
                    "A": {k: val["A_primary"].get(k) for k in ("n", "n_ok", "max_abs_mm",
                                                               "pass")},
                    "B": {k: val["B"].get(k) for k in ("n_testable", "min_r", "pass")},
                    "full_run_reproduces_v1": val["full_run_reproduces_v1"]["all_identical"]},
                   ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
