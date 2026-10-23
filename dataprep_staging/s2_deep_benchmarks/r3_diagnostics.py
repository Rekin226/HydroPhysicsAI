"""WP S2 round 3: after-the-fact diagnostics of deep_benchmarks/v1 (no product, no rule change).

1. Agency probe-adjustment records. The 007001 reports print the dates and sizes of the
   probe adjustments (探針調整 / 讀數調整 / 伸張儀調整) of the deep benchmarks. They were not used
   to build the step catalogue, so they are an independent check of it: wherever the raw feed
   covers an adjustment, the catalogue should hold a removed step of the documented size, and
   the vendor feed (which stitches resets) should show no jump. The records below were copied
   by hand from the poppler layout text of the cited page; every printed number is checked to
   occur on that page (``verify_records``), so a copying slip fails the run. Maintenance
   tables in the same reports carry staff names; nothing here reads or writes them.
2. Failure pattern of the registered test S2r2.C1 (already scored; look logged before this
   diagnostic): share within 2 mm by year, sensor and report column, and whether the other
   printed column of the same month (the report twelve months later) agrees with the feed.
3. The steps removed inside the months that fail the declared test A (design-time units,
   look logged), and the catalogue at the hour of the 2024-04-03 Hualien earthquake.

Usage: python -I r3_diagnostics.py <out_dir>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

PRODUCT = REPO / "results" / "dataprep" / "deep_benchmarks" / "v1"
PRINTED = REPO / "results" / "dataprep" / "deep_benchmarks_printed" / "v1"
L0 = REPO / "data_fetch_api" / "new_measurements" / "wra_cy_monthly_reports_007001"
VENDOR_OF = {"TKJS(1)": "TKJS_281_29", "TKJS(2)": "TKJS_282_29", "TKJS(3)": "TKJS_283_29",
             "NTUH(1)": "NTUH_282_29", "NTUH(2)": "NTUH_281_29", "SJES": "SJES", "STES": "STES"}

#: (report, page, table label, date, time or None, printed station, sensor, reading before,
#: reading after, printed adjustment). Readings in mm as printed; None where not printed.
RECORDS = [
    ("cy_2022-04.pdf", 5, "note 3 under table 3", "2022-04-13", "11:30", "土庫 400m", "TKJS(3)",
     None, None, "1.27 cm"),
    ("cy_2022-04.pdf", 5, "note 4 under table 3", "2022-04-13", "11:30", "土庫 130m", "TKJS(2)",
     None, None, "0.5 cm"),
    ("cy_2023-02.pdf", 10, "表 5 深層水準樁探針調整紀錄", "2023-02-02", "14:10", "土庫 300m",
     "TKJS(1)", "23.40", "-21.79", "45.19"),
    ("cy_2023-02.pdf", 10, "表 5 深層水準樁探針調整紀錄", "2023-02-02", "15:30", "秀潭 70m",
     "STES", "20.36", "-14.86", "35.22"),
    ("cy_2023-02.pdf", 10, "表 5 深層水準樁探針調整紀錄", "2023-02-02", "16:15", "虎尾 300m",
     "NTUH(1)", "93.55", "20.59", "72.96"),
    ("cy_2023-02.pdf", 10, "表 5 深層水準樁探針調整紀錄", "2023-02-02", "16:40", "虎尾 70m",
     "NTUH(2)", "47.74", "17.39", "30.35"),
    ("cy_2023-05.pdf", 5, "表 1 深層水準樁讀數調整表", "2023-05-11", "13:30", "土庫 130m",
     "TKJS(2)", "16.8973", "-22.5522", "39.45"),
    ("cy_2023-05.pdf", 5, "表 1 深層水準樁讀數調整表", "2023-05-11", "13:50", "土庫 300m",
     "TKJS(1)", "-0.6395", "-28.8803", "28.24"),
    ("cy_2023-05.pdf", 5, "表 1 深層水準樁讀數調整表", "2023-05-11", "13:19", "土庫 400m",
     "TKJS(3)", "47.3063", "-25.0817", "72.39"),
    ("cy_2023-05.pdf", 5, "表 1 深層水準樁讀數調整表", "2023-05-11", "14:40", "秀潭 70m",
     "STES", "-5.6093", "-35.5698", "29.96"),
    ("cy_2023-05.pdf", 5, "表 1 深層水準樁讀數調整表", "2023-05-11", "11:05", "虎尾 300m",
     "NTUH(1)", "39.1792", "8.8519", "20.67"),
    ("cy_2023-05.pdf", 5, "表 1 深層水準樁讀數調整表", "2023-05-11", "11:15", "虎尾 70m",
     "NTUH(2)", "21.2489", "0.5805", "30.33"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "土庫國中 (300m)", "TKJS(1)", "-28.74", "-28.79", "0.05"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "土庫國中 (130m)", "TKJS(2)", "-32.24", "-36.00", "3.76"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "秀潭國小 (70m)", "STES", "-36.99", "-31.66", "5.33"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "溪州國小 (300m)", "SJES", "-25.07", "-38.00", "12.93"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "中科虎尾 (300m)", "NTUH(1)", "35.14", "10.55", "24.59"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "中科虎尾 (70m)", "NTUH(2)", "-6.31", "-7.19", "-0.9"),
    ("cy_2025-04.pdf", 14, "表 10 彰雲地區深層水準樁伸張儀調整表", "2025-03-06", None,
     "土庫國中 (400m)", "TKJS(3)", "-18.30", "-34.30", "16.00"),
]
COLS = ["report", "page", "table", "date", "time", "station_printed", "sensor_id",
        "before_mm", "after_mm", "adjustment_printed"]


def page_text(report: str, page: int) -> str:
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout",
                           str(L0 / report), "-"], check=True, capture_output=True,
                          text=True).stdout


def verify_records(recs: pd.DataFrame) -> list[str]:
    """Every printed number of a record occurs on its cited page (whitespace-normalised)."""
    errs, cache = [], {}
    for r in recs.itertuples():
        key = (r.report, r.page)
        if key not in cache:
            cache[key] = re.sub(r"\s+", " ", page_text(*key))
        t = cache[key]
        for v in (r.before_mm, r.after_mm, r.adjustment_printed, r.date[5:7].lstrip("0")):
            if v is None:
                continue
            pat = re.escape(str(v)).replace(r"\ ", r"\s*")
            if not re.search(pat, t):
                errs.append(f"{r.report} p{r.page}: {v!r} not on the page ({r.station_printed})")
    return errs


def agency_check(recs: pd.DataFrame) -> pd.DataFrame:
    steps = pd.read_csv(PRODUCT / "steps.csv")
    stitches = pd.read_csv(PRODUCT / "vendor_stitches.csv")
    daily = pd.read_parquet(PRODUCT / "daily.parquet", columns=["sensor_id", "feed", "date"])
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = {}
    out = []
    for r in recs.itertuples():
        timed = r.time is not None
        t0 = pd.Timestamp(f"{r.date} {r.time or '12:00'}")
        tol = pd.Timedelta(hours=3 if timed else 36)
        sid = r.sensor_id
        delta = (float(r.after_mm) - float(r.before_mm)) if r.before_mm is not None else np.nan
        s = steps[steps.sensor_id == sid].copy()
        tb, ta = pd.to_datetime(s.t_before), pd.to_datetime(s.t_after)
        near = s[((tb - tol) <= t0) & (t0 <= (ta + tol))]
        raw_near = near[near.feed == "raw"]
        raw_days = daily[(daily.sensor_id == sid) & (daily.feed == "raw")
                         & (daily.date.between(t0.normalize() - pd.Timedelta(days=1),
                                               t0.normalize() + pd.Timedelta(days=1)))]
        st = stitches[stitches.sensor_id == sid]
        stb, sta = pd.to_datetime(st.t_before), pd.to_datetime(st.t_after)
        st_near = st[((stb - tol) <= t0) & (t0 <= (sta + tol))]
        vk = VENDOR_OF.get(sid)
        if vk not in hourly:
            hourly[vk] = L.to_hourly(L.vendor_raw(vk))
        h = hourly[vk]
        w = h[t0 - tol - pd.Timedelta(hours=1): t0 + tol + pd.Timedelta(hours=1)]
        vjump = float(np.nanmax(np.abs(w.diff().values))) if w.notna().sum() > 1 else np.nan
        pre = h[t0 - tol - pd.Timedelta(hours=24): t0 - tol].mean()
        post = h[t0 + tol: t0 + tol + pd.Timedelta(hours=24)].mean()
        steps_raw = raw_near[raw_near.kind == "step"]
        outage = raw_near[raw_near.kind == "outage"]
        if len(steps_raw):
            cat = float(steps_raw.delta_obs_mm.sum())
            status = "raw step catalogued"
        elif len(outage):
            cat = float(outage.delta_obs_mm.sum())
            status = (f"inside a raw outage of {float(outage.duration_h.max()):.0f} h "
                      "(kept as real movement)")
        elif len(raw_days):
            cat, status = np.nan, "raw feed present, no step catalogued"
        else:
            cat, status = np.nan, "no raw feed (vendor only)"
        out.append({**{c: getattr(r, c) for c in COLS},
                    "after_minus_before_mm": round(delta, 4) if delta == delta else None,
                    "printed_consistent": (None if r.before_mm is None else
                                           bool(abs(abs(delta) - abs(float(
                                               r.adjustment_printed))) <= 0.051)),
                    "raw_status": status,
                    "catalogue_delta_obs_mm": None if cat != cat else round(cat, 3),
                    "catalogue_minus_agency_mm": (None if cat != cat or delta != delta
                                                  or not len(steps_raw)
                                                  else round(cat - delta, 3)),
                    "vendor_key": vk, "vendor_hours_in_window": int(w.notna().sum()),
                    "vendor_max_hourly_change_mm": None if vjump != vjump else round(vjump, 3),
                    "vendor_level_change_mm": None if (pre != pre or post != post)
                    else round(float(post - pre), 3),
                    "vendor_stitch_classification": ";".join(
                        st_near.classification.astype(str)) or ""})
    return pd.DataFrame(out)


def c1_pattern() -> dict:
    u = pd.read_parquet(PRODUCT / "test_c_units.parquet")
    c1 = u[(u.test == "C1") & u.testable].copy()
    c1["ok"] = c1.diff_mm.abs() <= 2.0 + 1e-9
    c1["year"] = c1.month.str[:4]
    pm = pd.read_parquet(PRINTED / "printed_monthly.parquet")
    pm = pm[pm.flag == 0][["sensor_id", "month", "column", "report", "value"]]
    alt = c1.merge(pm.rename(columns={"column": "alt_column", "report": "alt_report",
                                      "value": "alt_value"}), on=["sensor_id", "month"])
    alt = alt[alt.alt_column != alt.column].copy()
    alt["alt_diff"] = alt.alt_value - alt.monthly_mean_inc_mm
    fail = alt[~alt.ok]

    def grp(keys):
        return {"|".join(map(str, k if isinstance(k, tuple) else (k,))):
                {"n": int(len(g)), "n_ok": int(g.ok.sum()),
                 "median_abs_mm": round(float(g.diff_mm.abs().median()), 3)}
                for k, g in c1.groupby(keys)}

    f = c1[~c1.ok].sort_values(["sensor_id", "month"])
    return {"label": "after-the-fact diagnostic of the already-scored registered test "
                     "S2r2.C1 (look logged 2026-10-12T10:28:53Z); no rule or threshold changed",
            "n_testable": int(len(c1)), "n_ok": int(c1.ok.sum()),
            "by_year": grp("year"), "by_sensor_year": grp(["sensor_id", "year"]),
            "by_column": grp("column"), "by_feed": grp("feed"),
            "alternate_printed_column": {
                "n_units_with_both_columns": int(len(alt)),
                "n_failing": int(len(fail)),
                "failing_with_alternate_within_2mm_of_feed": int(
                    (fail.alt_diff.abs() <= 2.0 + 1e-9).sum()),
                "n_passing": int(alt.ok.sum()),
                "passing_with_alternate_within_2mm_of_feed": int(
                    (alt[alt.ok].alt_diff.abs() <= 2.0 + 1e-9).sum()),
                "printed_vs_alternate_over_2mm": int(
                    ((alt.printed_compression_mm - alt.alt_value).abs() > 2.0).sum())},
            "failures": f[["sensor_id", "month", "printed_compression_mm",
                           "monthly_mean_inc_mm", "diff_mm", "feed", "column", "report",
                           "month_end_inc_mm"]].round(3).to_dict("records")}


def a_failures() -> dict:
    """Steps removed inside the months that fail test A, and every catalogue event or vendor
    stitch in the hour of the 2024-04-03 Hualien earthquake (07:58 local time)."""
    val = json.loads((PRODUCT / "validation.json").read_text(encoding="utf-8"))
    steps = pd.read_csv(PRODUCT / "steps.csv")
    stitches = pd.read_csv(PRODUCT / "vendor_stitches.csv")
    rows = []
    for f in val["A_primary"]["failures"]:
        s = steps[(steps.sensor_id == f["sensor_id"]) & (steps.month == f["month"])
                  & (steps.kind == "step")]
        rows.append({**f, "removed_steps": s[["feed", "t_before", "t_after", "duration_h",
                                              "correction_mm", "threshold_mm"]]
                     .round(3).to_dict("records"),
                     "sum_removed_mm": round(float(s.correction_mm.sum()), 3)})
    hr = "2024-04-03 07:00"
    q = steps[steps.t_before.astype(str).str.startswith(hr)]
    v = stitches[stitches.t_before.astype(str).str.startswith(hr)]
    return {"label": "after-the-fact look at design-time units of test A (look logged "
                     "2026-10-12T10:36:23Z); no rule or threshold changed",
            "failures": rows,
            "earthquake_hour_2024_04_03_07h": {
                "catalogue": q[["sensor_id", "feed", "kind", "delta_obs_mm", "corrected"]]
                .round(3).to_dict("records"),
                "vendor_stitches": v[["sensor_id", "raw_change_mm", "vendor_change_mm",
                                      "classification"]].round(3).to_dict("records")}}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    recs = pd.DataFrame(RECORDS, columns=COLS)
    errs = verify_records(recs)
    if errs:
        raise SystemExit("record check failed: " + "; ".join(errs))
    chk = agency_check(recs)
    chk.to_csv(out / "agency_adjustments_vs_catalogue.csv", index=False)
    cov = chk[chk.raw_status == "raw step catalogued"]
    # the vendor TKJS_283_29 tracks no raw sensor (product: defective, flag 9, not used)
    ven = chk[(chk.sensor_id != "TKJS(3)") & (chk.vendor_hours_in_window >= 2)]
    summary = {
        "schema": "s2-r3-diagnostics/1",
        "label": "after-the-fact analysis of deep_benchmarks/v1; no product, no rule change",
        "records": int(len(chk)), "records_verified_on_page": int(len(chk)),
        "printed_adjustment_inconsistent_with_readings": chk[
            chk.printed_consistent == False][["report", "station_printed",  # noqa: E712
                                              "after_minus_before_mm",
                                              "adjustment_printed"]].to_dict("records"),
        "raw_step_catalogued": int(len(cov)),
        "max_abs_catalogue_minus_agency_mm": float(cov.catalogue_minus_agency_mm.abs().max())
        if len(cov) else None,
        "inside_raw_outage": chk[chk.raw_status.str.startswith("inside")][
            ["sensor_id", "date", "after_minus_before_mm", "raw_status",
             "catalogue_delta_obs_mm", "vendor_stitch_classification"]].to_dict("records"),
        "vendor_only": int((chk.raw_status == "no raw feed (vendor only)").sum()),
        "vendor_records_checked": int(len(ven)),
        "vendor_note": "vendor series of mapped sensors only (TKJS_283_29 excluded: defective)",
        "vendor_max_hourly_change_mm_at_adjustments": float(
            ven.vendor_max_hourly_change_mm.max()),
        "vendor_max_abs_level_change_mm": float(ven.vendor_level_change_mm.abs().max()),
        "agency_abs_adjustment_mm_range_vendor_checked": [
            float(ven.after_minus_before_mm.abs().min()),
            float(ven.after_minus_before_mm.abs().max())],
        "c1_failure_pattern": c1_pattern(),
        "a_failures": a_failures(),
    }
    (out / "r3_diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                        default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "c1_failure_pattern"},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
