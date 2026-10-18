"""Step 2: probe R-h - do post-2018 records exist for the wells whose records stop in
December 2018 (plan WP H3 'Acquisition', section 6.2 R-h)? Existing access only.

The task allows the probe "only through existing repository tooling/access (no new
credentials, no letters)". Routes checked, all read-only:

1. The twin's L0 cache ``AMP_V2/data/wells`` (vendor dataset ``gw-wra-gw10min-obs`` fetched
   for 2012-01-01..2023-01-01): last sample of each well.
2. The raw vendor files in ``chou-shui-data/data/ls_cache/gw__<sid>__raw.parquet`` (the
   original delivery's cache): samples per year, 2010-2025.
3. The vendor station list fetched 2026-09-30
   (``data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet``): the dataset's
   per-station record window ``start/end_datetime_from_dataset`` and ``DisuseDate`` (only
   these non-personal columns are read). The window's semantics are checked first on every
   station with cache data in 2020-2022: its start must equal the first cache sample on or
   after 2020-01-01.
4. The new-network fetch ``data_fetch_api/new_measurements/heads_202210_202610`` (station
   files present).
5. The original delivery's wide files ``chou-shui-data/data/raw/gw_timeseries_raw.csv``
   (174 wells) and ``gw_timeseries.csv`` (61 wells): values after 2018 are checked for
   provenance (whether the file is gap-filled), never used as observations.
6. A credentialed refetch with ``hydrophysics.twin.fetch_amp`` is possible only with
   ``WISENVR_*`` variables in the environment; their presence is tested by name only.

Writes ``H3_evidence/rh_probe/rh_probe.json`` and ``rh_probe_wells.csv``. Reads no head value
after 2022 into any product (the probe's 2023-25 sample counts are availability metadata).

Usage: python -I s2_rh_probe.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

LS_CACHE = C.REPO / "chou-shui-data/data/ls_cache"
CS_RAW = C.REPO / "chou-shui-data/data/raw/gw_timeseries_raw.csv"
CS_61 = C.REPO / "chou-shui-data/data/gw_timeseries.csv"
CS_61_ST = C.REPO / "chou-shui-data/data/gw_stations.csv"
VENDOR_ST = C.REPO / "data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet"
NEW_NET = C.REPO / "data_fetch_api/new_measurements/heads_202210_202610"
VENDOR_COLS = ["sid", "DisuseDate", "start_datetime_from_dataset",
               "end_datetime_from_dataset"]


def cache_series(sid: str) -> pd.Series:
    p = C.AMP_WELLS / f"{sid}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_parquet(p)["value"]
    s.index = pd.to_datetime(s.index)
    return s.where(s.abs() < 1000).dropna().sort_index()


def ls_raw(sid: str) -> pd.Series | None:
    p = LS_CACHE / f"gw__{sid}__raw.parquet"
    if not p.exists():
        return None
    s = pd.to_numeric(pd.read_parquet(p)["value"], errors="coerce")
    s.index = pd.to_datetime(s.index)
    return s.where(s.abs() < 1000).dropna().sort_index()


def vendor_window_check(vendor: pd.DataFrame, sids: list[str]) -> dict:
    """Does start_datetime_from_dataset equal the first cache sample on or after 2020-01-01
    for stations with cache samples in 2020-2022?"""
    t20 = pd.Timestamp("2020-01-01")
    agree, disagree, n = 0, [], 0
    v = vendor.set_index("sid")
    for sid in sids:
        if sid not in v.index:
            continue
        s = cache_series(sid)
        s = s[s.index >= t20]
        if s.empty:
            continue
        n += 1
        st = v.loc[sid, "start_datetime_from_dataset"]
        if pd.isna(st):
            disagree.append({"sid": sid, "cache_first_2020": str(s.index[0]), "vendor": None})
            continue
        if abs((pd.Timestamp(st) - s.index[0]).total_seconds()) <= 3600:
            agree += 1
        else:
            disagree.append({"sid": sid, "cache_first_2020": str(s.index[0]),
                             "vendor": str(st)})
    return {"stations_checked": n, "agree_within_1h": agree, "disagree": disagree[:20],
            "n_disagree": len(disagree)}


def cs_fill_check(cs: pd.DataFrame) -> dict:
    """Is the delivery's 174-well file gap-filled? NaN share, and how it treats the hours the
    vendor cache lacks."""
    tot_nan = float(cs.isna().to_numpy().mean())
    same, filled_hours, n_wells = [], 0, 0
    for sid in cs.columns:
        a = cache_series(sid)
        if a.empty:
            continue
        n_wells += 1
        ah = a.resample("h").mean().reindex(cs.index)
        both = ah.notna() & cs[sid].notna()
        d = (ah[both] - cs[sid][both]).abs()
        same.append(float((d <= 0.0051).mean()) if both.any() else np.nan)
        filled_hours += int((ah.isna() & cs[sid].notna()).sum())
    return {"nan_share_whole_file": tot_nan, "wells_compared_with_cache": n_wells,
            "median_share_identical_where_both_observed": float(np.nanmedian(same)),
            "hours_present_in_file_but_absent_from_cache": filled_hours}


def main():
    inv = pd.read_csv(C.EVIDENCE / "inventory" / "inventory.csv",
                      dtype={"sid": str, "layer_raw": str})
    ended = inv[inv.record_last == "2018-12"].copy()
    cand = inv[inv.candidate.astype(bool)].copy()
    vendor = pd.read_parquet(VENDOR_ST, columns=VENDOR_COLS)
    vendor["sid"] = vendor["sid"].astype(str)
    vd = vendor.set_index("sid")
    new_net = {p.stem for p in NEW_NET.glob("*.parquet")}
    cs = pd.read_csv(CS_RAW, parse_dates=[0], index_col=0)
    cs.columns = [str(c) for c in cs.columns]
    st61 = pd.read_csv(CS_61_ST)
    st61["sid"] = st61.Station.astype(str).str.zfill(8)
    in61 = dict(zip(st61.sid, st61.st_id, strict=True))
    rows = []
    for _, r in ended.iterrows():
        sid = r.sid
        a = cache_series(sid)
        ls = ls_raw(sid)
        rec = {"sid": sid, "county": r.county, "layer": r.layer_raw, "on_grid": r.on_grid,
               "cache_last_sample": str(a.index[-1]) if len(a) else None,
               "cache_samples_2019_2022": int(((a.index >= "2019-01-01")
                                               & (a.index < "2023-01-01")).sum()),
               "ls_cache_raw": ls is not None,
               "ls_samples_2019_2022": (int(((ls.index >= "2019-01-01")
                                             & (ls.index < "2023-01-01")).sum())
                                        if ls is not None else None),
               "ls_first_after_2018": (str(ls[ls.index >= "2019-01-01"].index.min())
                                       if ls is not None and (ls.index >= "2019-01-01").any()
                                       else None),
               "in_new_network": sid in new_net,
               "vendor_listed": sid in vd.index}
        if sid in vd.index:
            v = vd.loc[sid]
            rec.update({"vendor_window_start": None if pd.isna(v.start_datetime_from_dataset)
                        else str(v.start_datetime_from_dataset),
                        "vendor_window_end": None if pd.isna(v.end_datetime_from_dataset)
                        else str(v.end_datetime_from_dataset),
                        "vendor_disuse_date": None if pd.isna(v.DisuseDate)
                        else str(v.DisuseDate)})
        if sid in cs.columns:
            post = cs[sid].loc["2019-01-01":"2022-12-31"]
            rec["delivery_174_values_2019_2022"] = int(post.notna().sum())
        rec["in_delivery_61"] = sid in in61
        pre_origin_found = (rec["cache_samples_2019_2022"] > 0
                            or (rec["ls_samples_2019_2022"] or 0) > 0
                            or rec["in_new_network"])
        rec["post2018_preorigin_observed_record_found"] = bool(pre_origin_found)
        rows.append(rec)
    W = pd.DataFrame(rows)
    out = C.EVIDENCE / "rh_probe"
    out.mkdir(parents=True, exist_ok=True)
    W.to_csv(out / "rh_probe_wells.csv", index=False)
    # semantics of the vendor window, on stations with cache data in 2020-2022
    chk = vendor_window_check(vendor, inv.sid.tolist())
    # candidates: post-origin availability (secondary test set, plan H3 'Use')
    cvd = []
    for sid in cand.sid:
        v = vd.loc[sid] if sid in vd.index else None
        cvd.append({"sid": sid, "in_new_network": sid in new_net,
                    "vendor_window_start": None if v is None or pd.isna(
                        v.start_datetime_from_dataset) else str(v.start_datetime_from_dataset),
                    "vendor_window_end": None if v is None or pd.isna(
                        v.end_datetime_from_dataset) else str(v.end_datetime_from_dataset)})
    rep = {
        "question": "R-h: post-2018 records of the wells whose records stop in December 2018",
        "wells": {"n": int(len(W)), "on_grid": int(W.on_grid.sum()),
                  "on_grid_layer_1_4": int((W.on_grid & W.layer.isin(C.LAYERS)).sum())},
        "found_preorigin_post2018_observed": int(W.post2018_preorigin_observed_record_found
                                                 .sum()),
        "routes": {
            "amp_cache": {"wells_with_2019_2022_samples":
                          int((W.cache_samples_2019_2022 > 0).sum())},
            "ls_cache_raw": {"wells_with_file": int(W.ls_cache_raw.sum()),
                             "wells_with_2019_2022_samples":
                             int((W.ls_samples_2019_2022.fillna(0) > 0).sum()),
                             "first_sample_after_2018": W.ls_first_after_2018.dropna()
                             .tolist()},
            "vendor_station_list_2026_09_30": {
                "window_semantics_check": chk,
                "window_start_by_year": W.vendor_window_start.str[:4].value_counts(
                    dropna=False).to_dict() if "vendor_window_start" in W else {},
                "disuse_dates": W.get("vendor_disuse_date", pd.Series(dtype=str))
                .value_counts(dropna=False).to_dict()},
            "new_network_2022_10_2026_09": {"wells_present": int(W.in_new_network.sum())},
            "delivery_174_file": {"wells_in_file": int(W.get(
                "delivery_174_values_2019_2022", pd.Series(dtype=float)).notna().sum()),
                "fill_check": cs_fill_check(cs)},
            "delivery_61_file": {"wells_in_file": int(W.in_delivery_61.sum())},
            "credentialed_refetch": {
                "tool": "hydrophysics.twin.fetch_amp (wells --start/--end)",
                "credential_variables_present": sorted(
                    k for k in os.environ if k.startswith("WISENVR_")),
                "run": False,
                "why": "the task allows existing access only (no new credentials); no "
                       "WISENVR_* variable is set in this environment"}},
        "candidates_post_origin_availability": cvd,
    }
    (out / "rh_probe.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False,
                                                  default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "candidates_post_origin_availability"},
                     indent=1, ensure_ascii=False, default=str))
    print(pd.DataFrame(cvd).to_string())


if __name__ == "__main__":
    main()
