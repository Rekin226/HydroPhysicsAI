"""Task 1: driver availability after 2022-12 (dates, flags and coverage only).

For each driver of the hybrid TF layer: the source product and component, its allowed uses,
the last month present and the last month usable as a recorded value (flag rules below), and
the lag from that month to a test window starting 2026-11. For the heads, only the timestamps
of the last fetch (``data_fetch_api/new_measurements/heads_202210_202610``) are read: the time
column of each file, never its values.

Every product with a manifest is opened with ``common.guard.open_product`` under the use
``after_the_fact_analysis`` (an inventory feeds no model); driver values are not read except
for missing-value counts of the rain cache.

Writes ``results/dataprep/reports/I1_hybrid_tf/availability.json`` and ``availability.csv``.

Run from the repository root:
    PYTHONPATH=dataprep_staging dataprep_staging/tools/cpu_slot.sh \
        ~/miniforge3/envs/hydro/bin/python -m i1.hybrid_tf.inventory
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C

sys.path.insert(0, str(C.REPO / "dataprep_staging"))

from common.guard import open_product  # noqa: E402

WINDOW_START = pd.Period("2026-11", "M")
HEADS_FETCH = C.REPO / "data_fetch_api/new_measurements/heads_202210_202610"
HEADS_META = C.REPO / "data_fetch_api/new_measurements/heads_download.json"
SW_FULL = C.REPO / "results/dataprep/surface_water_v4/full/v4"


def _lag(last: str | None) -> int | None:
    """Months not yet recorded before the window: last recorded + 1 .. window start - 1
    (the ``config.AVAIL_LAG`` convention)."""
    if last is None:
        return None
    return int((WINDOW_START - pd.Period(last, "M")).n) - 1


def _uses(path) -> tuple[list, str]:
    m = json.loads((path / "MANIFEST.json").read_text())
    return m.get("allowed_uses", []), m.get("fit_window_end", "")


def _rel(p) -> str:
    return str(p.relative_to(C.REPO))


def grid_flags_by_month(path, npz: str, value_key: str | None, flag_key: str, ok_flags,
                        use: str = "after_the_fact_analysis") -> dict:
    pr = open_product(path, use)
    with np.load(pr.path(npz)) as z:
        dates = [str(x)[:7] for x in z["dates"]]
        f = np.asarray(z[flag_key])
        if value_key is not None:            # cells that carry energy at some time
            active = np.nanmax(np.asarray(z[value_key], dtype=float), axis=-1) > 0
            f = f[active]
    share = np.isin(f, list(ok_flags)).mean(axis=tuple(range(f.ndim - 1)))
    return {"first": dates[0], "last_present": dates[-1],
            "share_ok_last_6": {d: round(float(s), 4)
                                for d, s in zip(dates[-6:], share[-6:], strict=True)},
            "last_ok_99": next((d for d, s in zip(dates[::-1], share[::-1], strict=True)
                                if s >= 0.99), None)}


def heads_inventory() -> dict:
    meta = json.loads(HEADS_META.read_text())
    last, n_last, firsts = [], 0, []
    files = sorted(HEADS_FETCH.glob("*.parquet"))
    per_month = {}
    for f in files:
        t = pq.read_table(f, columns=["datetime"]).column("datetime").to_pandas()
        if t.empty:
            continue
        last.append(t.max())
        firsts.append(t.min())
        months = pd.DatetimeIndex(t).to_period("M").unique()
        for m in months[months >= pd.Period("2026-01", "M")]:
            per_month[str(m)] = per_month.get(str(m), 0) + 1
    lastp = pd.Period(max(last), "M")
    n_last = sum(1 for x in last if pd.Period(x, "M") == lastp)
    return {"source": _rel(HEADS_FETCH), "dataset": meta.get("dataset"),
            "request_start": meta.get("start"), "request_end_exclusive": meta.get("end_exclusive"),
            "n_files": len(files), "first_timestamp": str(min(firsts)),
            "last_timestamp": str(max(last)), "last_month": str(lastp),
            "wells_with_data_in_last_month": n_last,
            "wells_with_any_record_per_month_2026": dict(sorted(per_month.items())),
            "note": "timestamps only; no head value was read. The release full/heads_observed "
                    "component also ends 2026-09 (PRODUCT_CARD); 2026-10 heads exist at the "
                    "source but have not been fetched"}


def main() -> None:
    rows = []
    # electricity census (C0): full to 2026-07, the last month partial (flags 8/9)
    p = C.REL / "full/electricity_census/v1"
    g = grid_flags_by_month(p, "cells_total.npz", "value_all", "flag_all", range(6))
    uses, fwe = _uses(p)
    rows.append({"driver": "census electricity (3 km and county)", "product":
                 "release_20261012b2 full/electricity_census (C0 pump_electricity/full/v3)",
                 "allowed_uses": uses, "fit_window_end": fwe, "last_present": g["last_present"],
                 "last_recorded": g["last_ok_99"], "detail": g,
                 "note": "2026-07 is partial (flags 8/9 on the readings not yet posted); "
                         "pre-origin variant ends 2022-12"})
    p = C.REL / "full/electricity_extension/v1"
    uses, fwe = _uses(p)
    rows.append({"driver": "census electricity extension (TPC farm-sector sales)",
                 "product": "release_20261012b2 full/electricity_extension (P2def)",
                 "allowed_uses": uses, "fit_window_end": fwe, "last_present": "2026-08",
                 "last_recorded": None,
                 "note": "2026-07..2026-08, flag 5, sd uncalibrated; its manifest does not grant "
                         "statistical_layer_history, so the TF layer cannot use it"})
    # canal (W1): full to 2026-09; 2026 months flag 6 (rain-driven extension)
    p = C.REL / "full/surface_water/v1"
    g = grid_flags_by_month(p, "grid_sw_v4.npz", None, "flag", range(6))
    uses, fwe = _uses(p)
    rows.append({"driver": "canal deliveries (W1 cell and district)", "product":
                 "release_20261012b2 full/surface_water (W1 surface_water_v4/full/v4)",
                 "allowed_uses": uses, "fit_window_end": fwe, "last_present": g["last_present"],
                 "last_recorded": "2025-12", "detail": g,
                 "note": "flag 5 (inferred) to 2025-12; 2026-01..09 flag 6, a rain-driven "
                         "extrapolation, not a delivery record; the forward archive of the "
                         "real-time feeds is blocked (W1)"})
    # ET0 (W2)
    p = C.REL / "full/et0/v1"
    pr = open_product(p, "after_the_fact_analysis")
    with np.load(pr.path("et0_daily_2012_2026.npz")) as z:
        d = pd.to_datetime(z["dates"])
        prov = np.asarray(z["provisional"], bool)
    uses, fwe = _uses(p)
    prov_m = sorted({str(x)[:7] for x in d[prov]})
    rows.append({"driver": "ET0 (W2, part of rain recharge)", "product":
                 "release_20261012b2 full/et0 (et0/v1)", "allowed_uses": uses,
                 "fit_window_end": fwe, "last_present": str(d.max())[:7],
                 "last_recorded": str(d.max())[:7],
                 "note": f"daily to {str(d.max())[:10]}; provisional months {prov_m}; "
                         "level-matched from 2023-01 (flag 1). Open-Meteo archive, re-fetchable"})
    # rain (twin cache, CODIS rebuild, no manifest)
    rf = pd.read_csv(C.RAIN, parse_dates=["date time"]).set_index("date time")
    last = rf.index.max()
    mon = rf[rf.index >= "2026-01-01"].notna().groupby(rf[rf.index >= "2026-01-01"].index
                                                         .to_period("M")).mean()
    good = (mon >= 0.9).sum(axis=1)
    # a month counts as recorded once it has ended and >= RAIN_MIN_GAUGES of the 26 gauges
    # have >= 90 % of their days (the rule of prospective.py; every month 2012-01..2026-09
    # meets it, the thinnest being 2019-11 with 7 gauges)
    complete = [str(m) for m, n in good.items()
                if n >= C.RAIN_MIN_GAUGES and m.end_time <= last]
    rows.append({"driver": "rain (part of rain recharge)", "product":
                 "chou-shui-data/data/rf_timeseries.csv (CODIS rebuild of the vendor cache; "
                 "no manifest)", "allowed_uses": ["no manifest"], "fit_window_end": "",
                 "last_present": str(last)[:7],
                 "last_recorded": complete[-1] if complete else None,
                 "detail": {"gauges_with_90pct_days_2026": {str(k): int(v)
                                                            for k, v in good.items()}},
                 "note": f"daily to {str(last)[:10]} (26 gauges); CODIS re-fetchable each "
                         "month (memory: CODIS daily route works for C0/agr/cwb ids)"})
    # river (W1 L1 daily gauges, yearbook)
    pr = open_product(SW_FULL, "after_the_fact_analysis")
    gd = pr.read_table("gauges_daily_L1.parquet", columns=["date", "station_no"])
    gd["date"] = pd.to_datetime(gd["date"])
    lastg = {code: str(gd.loc[gd["station_no"] == code, "date"].max())[:10]
             for code in C.GAUGES.values()}
    uses, fwe = _uses(SW_FULL)
    rows.append({"driver": "Zhuoshui discharge (H057 apex, H071 mid-fan)", "product":
                 "surface_water_v4/full/v4 gauges_daily_L1.parquet (hydrological yearbook)",
                 "allowed_uses": uses, "fit_window_end": fwe,
                 "last_present": max(lastg.values())[:7], "last_recorded": "2025-12",
                 "detail": lastg, "note": "yearbook daily means; the 2026 yearbook is "
                 "published about a year after the year ends"})
    # public wells (P6)
    p = C.REL / "preorigin/public_wells_member/v1"
    pr = open_product(p, "after_the_fact_analysis")
    with np.load(pr.path("sensitivity_member.npz")) as z:
        dates = [str(x) for x in z["dates"]]
    uses, fwe = _uses(p)
    rows.append({"driver": "public-supply deep wells (P6 member)", "product":
                 "release_20261012b2 preorigin/public_wells_member (public_wells/v2)",
                 "allowed_uses": uses, "fit_window_end": fwe, "last_present": dates[-1],
                 "last_recorded": dates[-1],
                 "note": "closure-ramp sensitivity member, 2007-01..2022-12 only; no full "
                         "variant; annual steps (held at the last value after 2022-12)"})
    for r in rows:
        r["months_missing_before_2026_11"] = _lag(r["last_recorded"])
    heads = heads_inventory()
    C.OUT.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "i1-hybrid-tf-availability/1", "window_start": str(WINDOW_START),
           "role": "inventory (dates, flags, coverage); no head value read",
           "drivers": rows, "heads": heads}
    (C.OUT / "availability.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False,
                                                        default=str))
    pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items()
                   if k != "detail"} for r in rows]).to_csv(C.OUT / "availability.csv",
                                                            index=False)
    for r in rows:
        print(f"{r['driver']:48s} last {r['last_present']}  recorded {r['last_recorded']}  "
              f"missing before window {r['months_missing_before_2026_11']}")
    print("heads:", heads["last_timestamp"], heads["last_month"],
          heads["wells_with_data_in_last_month"], "of", heads["n_files"])


if __name__ == "__main__":
    main()
