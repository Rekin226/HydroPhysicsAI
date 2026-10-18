"""Pre-origin labels of the vendor deep-benchmark series (WP S2 repair round, 2026-10-12).

Why: ``deep_benchmarks/v1`` labelled every vendor series by its identity with a raw agency
sensor on common hours. Except for 土庫國中 300 m and 130 m, the raw feed starts in 2023-09,
so those labels rested on post-origin hours. ``deep_benchmarks/v2`` computes its pre-origin
rows from inputs cut at the end of 2022-12, so it needs labels that only pre-origin data
support. This script derives them and writes the evidence file that build_v2.py reads.

Rule (fixed in this docstring and in the declaration S2r4.deep_benchmarks_v2 before the run;
data up to 2022-12-31 only):

1. **Identity.** A vendor series is the copy of a raw sensor when the share of informative
   common pre-origin hours with an unchanged offset or sum is >= 0.98 over >= 48 common hours
   (``s2lib.identity``; the v1 rule). The best raw sensor per vendor series is taken; a raw
   sensor may be claimed once.
2. **Station code.** Otherwise a vendor series whose key equals an agency identifier without
   its ``BM_`` prefix (SJES, STES, WNES, DLIO) takes that sensor (metadata, no observation).
3. **Printed pre-origin table.** Otherwise (中科虎尾 NTUH_281_29 / NTUH_282_29, 土庫國中
   TKJS_283_29) the vendor series is compared with the printed 007001 monthly values of its
   station (``deep_benchmarks_printed/v2``, opened as design-time evidence through the time
   slice ending 2022-12; primary printings, flag 0). The vendor statistic is the difference of
   consecutive monthly means (daily means of >= 12 valid hours, monthly means of >= 20 days),
   the convention the 007001 reports state; no step catalogue is applied. A series sampled
   once a day (median spacing of its samples >= 20 h; TKJS_283_29) counts every sampled day
   as a full day, so that its increments are defined at all.

   * The NTUH pair takes the assignment (281 -> 70 m and 282 -> 300 m, or the swap) under
     which **each** series correlates better (Pearson r, >= 5 common months) with its assigned
     depth than with the other depth. If neither assignment satisfies this for both series the
     pair is ambiguous and its pre-origin rows are flagged 9.
   * TKJS_283_29 takes 土庫國中 400 m (TKJS(3)) when its r against the printed 400 m column
     (>= 5 common months) is at least the smaller r of the two identity-confirmed 土庫國中
     series against their own printed depths over the same window (the first to the last
     month common to TKJS_283_29 and the printed 400 m column); otherwise it is defective
     before the origin (flag 9), as v1 already treated it on other grounds.
   * The median absolute difference of every pair is reported beside r; it does not decide.

The labels of v1 (full-period identity, ``deep_benchmarks/v1/mapping.csv``) are compared
afterwards and reported; they do not enter the rule. Computing correlations on the printed
2019-2022 table is a look at the already-scored units of test C1: it is logged in the A1 look
log (purpose ``diagnose``).

Usage: python -I label_confirm.py [--out PATH] [--no-log]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, REPORTS_DIR, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

PRINTED_V2 = DATAPREP / "deep_benchmarks_printed" / "v2"
V1_MAPPING = DATAPREP / "deep_benchmarks" / "v1" / "mapping.csv"
OUT = REPORTS_DIR / "S2_evidence" / "r4" / "preorigin_labels.json"
CUT = pd.Timestamp("2023-01-01")            # samples strictly before this instant
DECL_ID = "S2r4.deep_benchmarks_v2"
TEST_ID = "S2r4.preorigin_labels"
MIN_HOURS_DAY = 12
MIN_DAYS_MONTH = 20
MIN_MONTHS_R = 5
DAILY_SPACING_H = 20.0       # a series sampled once a day
NAME_KEYS = ("SJES", "STES", "WNES", "DLIO")
STATION_OF = {"NTUH": "中科虎尾", "TKJS": "土庫國中"}
#: the plan's mapping (plan WP S2 table; metadata, no observation): the nominal sensor of a
#: vendor series whose pre-origin label is ambiguous (its pre-origin rows are then flag 9)
PLAN_MAP = {"TKJS_281_29": "TKJS(1)", "TKJS_282_29": "TKJS(2)", "TKJS_283_29": "TKJS(3)",
            "NTUH_282_29": "NTUH(1)", "NTUH_281_29": "NTUH(2)", "SJES": "SJES", "STES": "STES",
            "WNES": "WNES", "DLIO": "DLIO"}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- pure
def monthly_mean_increments(hourly: pd.Series) -> pd.Series:
    """Difference of consecutive monthly means (mm) of one hourly series: daily means of days
    with >= MIN_HOURS_DAY valid hours, monthly means of months with >= MIN_DAYS_MONTH such
    days; NaN unless both months qualify. Index: 'YYYY-MM' strings."""
    h = L.as_ns(hourly.dropna())
    if h.empty:
        return pd.Series(dtype=float)
    g = h.groupby(h.index.floor("D"))
    spacing_h = float(h.index.to_series().diff().dt.total_seconds().median() / 3600.0) \
        if len(h) > 1 else float("nan")
    min_hours = 1 if (np.isfinite(spacing_h) and spacing_h >= DAILY_SPACING_H) \
        else MIN_HOURS_DAY
    day = g.mean()[g.size() >= min_hours]
    if day.empty:
        return pd.Series(dtype=float)
    mg = day.groupby(day.index.strftime("%Y-%m"))
    mm = mg.mean()[mg.size() >= MIN_DAYS_MONTH]
    full = pd.period_range(min(mm.index), max(mm.index), freq="M").astype(str)
    mm = mm.reindex(full)
    return mm.diff().dropna()


def corr_table(inc: dict[str, pd.Series], printed: pd.DataFrame,
               window: tuple[str, str] | None = None) -> pd.DataFrame:
    """Pearson r, n and median |diff| of every vendor series against every printed sensor of
    the same station (printed: sensor_id, month, value = compression mm), optionally only on
    months inside ``window`` (first, last; 'YYYY-MM')."""
    rows = []
    for vk, s in inc.items():
        code = vk.split("_")[0]
        if code not in STATION_OF:
            continue
        for sid, g in printed[printed.station == STATION_OF[code]].groupby("sensor_id"):
            j = g.set_index("month")["value"].to_frame("printed").join(s.rename("vendor"),
                                                                       how="inner").dropna()
            if window is not None:
                j = j[(j.index >= window[0]) & (j.index <= window[1])]
            n = int(len(j))
            r = L.pearson(j["printed"], j["vendor"]) if n >= 3 else float("nan")
            rows.append({"vendor_key": vk, "sensor_id": sid,
                         "depth_m": int(g["depth_m"].iloc[0]), "n_months": n, "r": r,
                         "median_abs_mm": float((j["printed"] - j["vendor"]).abs().median())
                         if n else float("nan"),
                         "months": f"{j.index.min()}..{j.index.max()}" if n else ""})
    return pd.DataFrame(rows)


def _r(ct: pd.DataFrame, vk: str, sid: str) -> tuple[float, int]:
    x = ct[(ct.vendor_key == vk) & (ct.sensor_id == sid)]
    if not len(x):
        return float("nan"), 0
    return float(x.r.iloc[0]), int(x.n_months.iloc[0])


def decide_pair(ct: pd.DataFrame, k1: str, k2: str, s_a: str, s_b: str) -> dict:
    """Assignment of two vendor series k1, k2 to two sensors s_a, s_b: the one under which each
    series has a larger r with its own sensor than with the other (>= MIN_MONTHS_R months)."""
    def ok(own: tuple, other: tuple) -> bool:
        return own[1] >= MIN_MONTHS_R and other[1] >= MIN_MONTHS_R and np.isfinite(own[0]) \
            and np.isfinite(other[0]) and own[0] > other[0]
    r1a, r1b, r2a, r2b = _r(ct, k1, s_a), _r(ct, k1, s_b), _r(ct, k2, s_a), _r(ct, k2, s_b)
    out = {"pair": [k1, k2], "sensors": [s_a, s_b],
           "r": {f"{k1}~{s_a}": r1a[0], f"{k1}~{s_b}": r1b[0], f"{k2}~{s_a}": r2a[0],
                 f"{k2}~{s_b}": r2b[0]},
           "n": {f"{k1}~{s_a}": r1a[1], f"{k1}~{s_b}": r1b[1], f"{k2}~{s_a}": r2a[1],
                 f"{k2}~{s_b}": r2b[1]}}
    if ok(r1a, r1b) and ok(r2b, r2a):
        out.update({"decision": {k1: s_a, k2: s_b}, "status": "confirmed"})
    elif ok(r1b, r1a) and ok(r2a, r2b):
        out.update({"decision": {k1: s_b, k2: s_a}, "status": "confirmed"})
    else:
        out.update({"decision": {k1: None, k2: None}, "status": "ambiguous"})
    return out


def decide_tracking(ct: pd.DataFrame, vk: str, sid: str, references: list[tuple[str, str]],
                    ct_window: pd.DataFrame | None = None) -> dict:
    """vk tracks sid when r(vk, sid) >= min r of the reference (vendor, own sensor) pairs,
    the references computed on ``ct_window`` (the same months as vk's comparison)."""
    r, n = _r(ct, vk, sid)
    cw = ct if ct_window is None else ct_window
    refs = {f"{a}~{b}": _r(cw, a, b) for a, b in references}
    ref_r = [v[0] for v in refs.values() if v[1] >= MIN_MONTHS_R and np.isfinite(v[0])]
    bar = float(min(ref_r)) if ref_r else float("nan")
    tracks = bool(n >= MIN_MONTHS_R and np.isfinite(r) and np.isfinite(bar) and r >= bar)
    return {"vendor_key": vk, "sensor_id": sid, "r": r, "n_months": n,
            "reference_r": {k: v[0] for k, v in refs.items()},
            "reference_n": {k: v[1] for k, v in refs.items()}, "bar": bar,
            "status": "tracks" if tracks else "defective_preorigin",
            "decision": sid if tracks else None}


# ------------------------------------------------------------------------------- run
def compute() -> dict:
    vendor = {k: L.to_hourly(L.vendor_raw(k).loc[lambda s: s.index < CUT]) for k in
              L.VENDOR_KEYS}
    raw_full, info = L.load_open([L.OPEN_DIR / f for f in L.OPEN_FILES])
    raw = {k: s[s.index < CUT].dropna() for k, s in raw_full.items()}
    raw = {k: s for k, s in raw.items() if len(s)}
    # 1. identity on pre-origin hours
    irows, ident_full = [], {}
    for vk in L.VENDOR_KEYS:
        for rid, rh in raw.items():
            out = L.identity(vendor[vk], rh)
            ident_full[(vk, rid)] = out
            irows.append({"vendor_key": vk, "raw_id": rid, "n_common_hours": out["n_common_hours"],
                          "n_informative_hours": out.get("n_informative_hours", 0),
                          "share_same": out.get("share_same"),
                          "share_inverted": out.get("share_inverted"),
                          "share_union": out.get("share_union"),
                          "n_stitches": len(out.get("stitches", []))})
    ident = pd.DataFrame(irows)
    labels, how = {}, {}
    for vk, g in ident.groupby("vendor_key"):
        g = g[g.n_common_hours >= 48].sort_values("share_union", ascending=False)
        if len(g) and float(g.share_union.iloc[0]) >= 0.98:
            labels[vk] = str(g.raw_id.iloc[0]).replace("BM_", "")
            how[vk] = "pre-origin identity"
    claimed = list(labels.values())
    if len(claimed) != len(set(claimed)):
        raise RuntimeError(f"a raw sensor claimed twice by pre-origin identity: {labels}")
    # 2. station code
    raw_ids = {str(k).replace("BM_", "") for k in raw_full}
    for vk in NAME_KEYS:
        if vk not in labels and vk in raw_ids:
            labels[vk] = vk
            how[vk] = "station code (agency identifier = vendor key)"
    # 3. printed pre-origin table
    prod = guard.open_product(PRINTED_V2, "design_time_evidence", window_end=L.ORIGIN_MONTH)
    pr = prod.read_table("printed_monthly.parquet",
                         columns=["sensor_id", "station", "depth_m", "month", "value", "flag"])
    pr = pr[(pr.flag == 0) & (pr.month <= L.ORIGIN_MONTH)]
    inc = {k: monthly_mean_increments(vendor[k]) for k in L.VENDOR_KEYS}
    inc = {k: v for k, v in inc.items() if len(v)}
    ct = corr_table(inc, pr)
    pair = decide_pair(ct, "NTUH_281_29", "NTUH_282_29", "NTUH(2)", "NTUH(1)")
    for k, s in pair["decision"].items():
        labels[k] = s
        how[k] = f"printed 007001 pre-origin table: pair assignment {pair['status']}"
    refs = [(k, labels[k]) for k in ("TKJS_281_29", "TKJS_282_29")
            if how.get(k) == "pre-origin identity"]
    x = ct[(ct.vendor_key == "TKJS_283_29") & (ct.sensor_id == "TKJS(3)") & (ct.n_months > 0)]
    win = tuple(x.months.iloc[0].split("..")) if len(x) else None
    ct_win = corr_table(inc, pr, window=win) if win else ct.iloc[0:0]
    trk = decide_tracking(ct, "TKJS_283_29", "TKJS(3)", refs, ct_window=ct_win)
    trk["window"] = list(win) if win else None
    labels["TKJS_283_29"] = trk["decision"]
    how["TKJS_283_29"] = f"printed 007001 pre-origin table: {trk['status']}"
    # status per vendor series (build_v2 reads it): confirmed / ambiguous (nominal label, rows
    # flag 9) / defective (rows flag 9) / no_preorigin_data
    status = {}
    for k in L.VENDOR_KEYS:
        hk = how.get(k, "none")
        if not vendor[k].notna().any():
            status[k] = "no_preorigin_data"
        elif hk in ("pre-origin identity", "station code (agency identifier = vendor key)"):
            status[k] = "confirmed"
        elif k in pair["decision"]:
            status[k] = "confirmed" if pair["status"] == "confirmed" else "ambiguous"
        elif k == "TKJS_283_29":
            status[k] = "confirmed" if trk["status"] == "tracks" else "defective"
        else:
            status[k] = "unlabelled"
    # after the fact: v1 labels (full-period identity)
    v1 = pd.read_csv(V1_MAPPING)
    v1lab = {r.vendor_key: (r.sensor_id if isinstance(r.raw_id, str) and r.raw_id else None)
             for r in v1.itertuples() if isinstance(r.vendor_key, str) and r.vendor_key}
    agree = {k: {"preorigin": labels.get(k), "v1": v1lab.get(k),
                 "agree": labels.get(k) == v1lab.get(k)} for k in L.VENDOR_KEYS}
    return {"labels": {k: labels.get(k) for k in L.VENDOR_KEYS},
            "status": status, "nominal": {k: PLAN_MAP[k] for k in L.VENDOR_KEYS},
            "method": {k: how.get(k, "none") for k in L.VENDOR_KEYS},
            "identity_preorigin": ident.round(4).to_dict("records"),
            "stitches_preorigin": {f"{vk}~{rid}": v["stitches"]
                                   for (vk, rid), v in ident_full.items()
                                   if labels.get(vk) == rid.replace("BM_", "")},
            "corr_table": ct.round(4).to_dict("records"),
            "corr_table_tkjs283_window": ct_win.round(4).to_dict("records"), "ntuh_pair": pair,
            "tkjs_283": trk, "agreement_with_v1": agree,
            "n_disagree_with_v1": int(sum(not v["agree"] for v in agree.values())),
            "printed_access_notes": prod.notes,
            "raw_preorigin_span": {k: [str(v.index.min()), str(v.index.max()), int(len(v))]
                                   for k, v in raw.items()},
            "vendor_preorigin_span": {k: [str(v.dropna().index.min()),
                                          str(v.dropna().index.max()), int(v.notna().sum())]
                                      for k, v in vendor.items() if v.notna().any()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    t0 = dt.datetime.now()
    res = compute()
    res = {"schema": "s2-preorigin-labels/1", "built": t0.isoformat(timespec="seconds"),
           "cut": "samples before 2023-01-01T00:00", "declaration": DECL_ID,
           "rule": __doc__.split("Rule (")[1].split("The labels of v1")[0].strip(),
           "printed_product": rel_to_repo(PRINTED_V2),
           "printed_manifest_sha256": sha256_file(PRINTED_V2 / "MANIFEST.json"), **res,
           "runtime_s": (dt.datetime.now() - t0).total_seconds()}
    if not a.no_log:
        from harness import prereg as pr
        sids = sorted({r["sensor_id"] for r in res["corr_table"]})
        pr.record_look([f"dbm_cy007001_printed:{s}:pre" for s in sids],
                       "deep_piles_cy007001_printed_vs_feed:2019-01..2022-12", "S2", "diagnose",
                       test_id=TEST_ID, prereg_id=DECL_ID,
                       note="S2 repair round: pre-origin labelling of the vendor series "
                            "(correlation of uncatalogued vendor monthly-mean increments with "
                            "the printed 007001 table, 2019-2022) for deep_benchmarks/v2; "
                            "the units were already scored by test C1 (v1)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n",
                     encoding="utf-8")
    log(json.dumps({k: res[k] for k in ("labels", "status", "method", "n_disagree_with_v1")},
                   ensure_ascii=False, indent=1))
    log(json.dumps({"ntuh": res["ntuh_pair"]["r"], "tkjs_283": {k: res["tkjs_283"][k] for k in
                                                                 ("r", "n_months", "bar",
                                                                  "status")}},
                   ensure_ascii=False, indent=1, default=str))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
