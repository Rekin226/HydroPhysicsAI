"""WP P1 round 2: the measured pump-electricity truth for Changhua and Yunlin, in model units.

Definitions are fixed in the pre-registrations ``P1r2.census_confirmations`` and
``P1r2.census_tests`` (``truth_definition``):

* L0: ``tpc_changhua_yunlin_chiayi_pingtung_2026/parquet/<county>_census.parquet`` and
  ``<county>_kwh_long.parquet`` only. ``raw/`` and ``extracted/`` (names, addresses) are never
  opened; census columns are selected explicitly (no 戶名1, 用電地址2 or postal code exist in
  parquet/, and none is requested).
* sid = WELL_NO (stripped, trailing '.0' removed); the kWh rows are keyed by row_idx.
* cleaning exactly as P1 v1 ``census_monthly.py``: ``clean_census`` cap 1.0 with the duty
  statistic on 2012-01..2022-12 selects the kept pumps, then the kept pumps' whole record is
  de-duplicated and HP-split (cap None). ``dedup`` = cap None on every pump.
* variants: ``preorigin`` truncates the kWh table to <= 2022-12 before cleaning; ``full``
  keeps every month to 2026-07 (2026-07 partial, never scored).
* in-grid: ``pumping._cell_lookup`` with the grid of results/twin_forward/datum_gate_innov.npz.

Outputs (county x scope x class x month totals and counts only; no pump rows, coordinates or
meter ids leave this script):

* ``truth_monthly.parquet``   long table: county, variant, series (clean|dedup), scope
                              (county|in_grid), class, date, gwh
* ``meter_counts.parquet``    county, date, meters with a non-missing reading, meters with a
                              non-zero reading (for the pre-registered month rule)
* ``truth_info.json``         pump counts (all, kept, in-grid), clean reports, timings,
                              preorigin-vs-full agreement on months <= 2022-12

Usage: python -I census_truth.py <out_dir>   (run inside tools/cpu_slot.sh; ~10 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hydrophysics.twin.pumping import _cell_lookup, clean_census  # noqa: E402
from hydrophysics.twin.scenario import purpose_class  # noqa: E402

L0 = REPO / "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/parquet"
GRID = REPO / "results/twin_forward/datum_gate_innov.npz"
#: explicit column list (plan 2.6); the personal columns are not in parquet/ and never asked for
PUMP_COLS = ["row_idx", "WELL_NO", "COUNTRYCITY", "TOWN", "TWD97_X", "TWD97_Y", "PUMP_HP",
             "PURPOSE", "電號1", "每期用電度數1"]
KWH_COLS = ["row_idx", "datetime", "electricity_kwh"]
DUTY_T0, DUTY_T1 = "2012-01-01", "2023-01-01"
ORIGIN_END = pd.Timestamp("2023-01-01")
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
INPUTS = [L0 / f"{c}_{k}.parquet" for c in COUNTIES for k in ("census", "kwh_long")] + [GRID]


def load_grid() -> SimpleNamespace:
    g = np.load(GRID, allow_pickle=False)
    mask = g["mask"].astype(bool)
    return SimpleNamespace(mask=mask, nx=int(g["nx"]), ny=int(g["ny"]), dx=float(g["dx"]),
                           x0=float(g["x0"]), y0=float(g["y0"]), n_active=int(mask.sum()))


def load_county(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pumps = pd.read_parquet(L0 / f"{tag}_census.parquet", columns=PUMP_COLS)
    sid = pumps["WELL_NO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if sid.duplicated().any():
        raise SystemExit(f"{tag}: WELL_NO not unique")
    if not (pumps["row_idx"].to_numpy() == np.arange(len(pumps))).all():
        raise SystemExit(f"{tag}: row_idx is not 0..n-1")
    pumps = pumps.assign(sid=sid.to_numpy()).drop(columns=["WELL_NO"])
    k = pd.read_parquet(L0 / f"{tag}_kwh_long.parquet", columns=KWH_COLS)
    k["datetime"] = pd.to_datetime(k["datetime"])
    sid_arr = pumps["sid"].to_numpy(dtype=object)
    k["pump"] = sid_arr[k["row_idx"].to_numpy()]
    k = k.drop(columns=["row_idx"])[["datetime", "electricity_kwh", "pump"]]
    return pumps, k


def meter_counts(pumps: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """Per month: unique meters (電號1, else sid) with a non-missing / non-zero reading."""
    meter = pumps["電號1"].astype(str)
    meter = meter.where(meter.notna() & (meter != "nan") & (meter != "") & (meter != "None"),
                        pumps["sid"])
    m_of = pd.Series(meter.to_numpy(), index=pumps["sid"].to_numpy())
    kk = k[["datetime", "electricity_kwh"]].copy()
    kk["meter"] = k["pump"].map(m_of).to_numpy()
    nn = kk[kk["electricity_kwh"].notna()]
    nz = nn[nn["electricity_kwh"] != 0]
    return pd.DataFrame({"meters_nonmissing": nn.groupby("datetime")["meter"].nunique(),
                         "meters_nonzero": nz.groupby("datetime")["meter"].nunique()})


def aggregate(kw: pd.DataFrame, cls_of: pd.Series, cell_of: dict) -> pd.DataFrame:
    """GWh per month x class for the county and the in-grid subset."""
    kw = kw.dropna(subset=["electricity_kwh"])
    cls = kw["pump"].map(cls_of)
    ingrid = kw["pump"].isin(set(cell_of))
    out = []
    for scope, sel in (("county", np.ones(len(kw), bool)), ("in_grid", ingrid.to_numpy())):
        g = kw.loc[sel].groupby([kw.loc[sel, "datetime"], cls[sel]])["electricity_kwh"].sum()
        g = (g / 1e6).rename("gwh").reset_index()
        g.columns = ["date", "class", "gwh"]
        g["scope"] = scope
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main(out_dir: str) -> int:
    t_all = time.time()
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    grid = load_grid()
    rows, counts = [], []
    info = {"grid": {"nx": grid.nx, "ny": grid.ny, "dx": grid.dx, "n_active": grid.n_active},
            "counties": {}}
    for tag, cname in COUNTIES.items():
        t0 = time.time()
        pumps, k = load_county(tag)
        ci = {"county_label": sorted(pumps["COUNTRYCITY"].astype(str).unique().tolist()),
              "pumps": int(len(pumps)), "kwh_rows": int(len(k)),
              "kwh_first": str(k["datetime"].min().date()),
              "kwh_last": str(k["datetime"].max().date()),
              "kwh_missing_frac": float(k["electricity_kwh"].isna().mean())}
        mc = meter_counts(pumps, k)
        mc["county"] = tag
        counts.append(mc.reset_index().rename(columns={"index": "date", "datetime": "date"}))
        cls_all = pumps.set_index("sid")["PURPOSE"].map(purpose_class)
        cell_all = _cell_lookup(pumps, grid)
        ci["pumps_in_grid"] = int(len(cell_all))
        ci["pumps_in_grid_frac"] = float(len(cell_all) / len(pumps))
        for variant, kv in (("full", k), ("preorigin", k[k["datetime"] < ORIGIN_END])):
            tv = time.time()
            kept, _, rep = clean_census(pumps, kv, cap_duty=1.0, t0=DUTY_T0, t1=DUTY_T1)
            _, kc, _ = clean_census(kept, kv, cap_duty=None)
            cell_kept = _cell_lookup(kept, grid)
            a = aggregate(kc, cls_all, cell_kept)
            a["series"] = "clean"
            parts = [a]
            if variant == "full":
                _, kd, _ = clean_census(pumps, kv, cap_duty=None)
                d = aggregate(kd, cls_all, cell_all)
                d["series"] = "dedup"
                parts.append(d)
                del kd
            for p in parts:
                p["county"], p["variant"] = tag, variant
                rows.append(p)
            ci[variant] = {
                "pumps_kept": int(len(kept)), "pumps_kept_in_grid": int(len(cell_kept)),
                "kept_class_counts": kept["PURPOSE"].map(purpose_class).value_counts()
                .to_dict(),
                "clean_report": json.loads(rep.reset_index().to_json(orient="records",
                                                                      force_ascii=False)),
                "seconds": round(time.time() - tv, 1)}
            del kc, kept
        ci["seconds"] = round(time.time() - t0, 1)
        info["counties"][tag] = ci
        del pumps, k
        print(tag, json.dumps({x: ci[x] for x in ci if x not in ("full", "preorigin")},
                              ensure_ascii=False), flush=True)
    tm = pd.concat(rows, ignore_index=True)
    tm["date"] = pd.to_datetime(tm["date"])
    tm = tm[["county", "variant", "series", "scope", "class", "date", "gwh"]].sort_values(
        ["county", "variant", "series", "scope", "class", "date"]).reset_index(drop=True)
    tm.to_parquet(od / "truth_monthly.parquet", index=False)
    mc = pd.concat(counts, ignore_index=True)
    mc["date"] = pd.to_datetime(mc["date"])
    mc.to_parquet(od / "meter_counts.parquet", index=False)
    # preorigin must equal full on every month <= 2022-12 (the duty window is pre-origin)
    agree = {}
    for tag in COUNTIES:
        x = tm[(tm.county == tag) & (tm.series == "clean")]
        f = x[(x.variant == "full") & (x.date < ORIGIN_END)].set_index(
            ["scope", "class", "date"])["gwh"]
        p = x[x.variant == "preorigin"].set_index(["scope", "class", "date"])["gwh"]
        j = pd.concat([f.rename("f"), p.rename("p")], axis=1).fillna(0.0)
        agree[tag] = {"max_abs_GWh": float((j.f - j.p).abs().max()), "n": int(len(j))}
    info["preorigin_vs_full_le_2022_12"] = agree
    info["seconds"] = round(time.time() - t_all, 1)
    (od / "truth_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1,
                                                   default=str), encoding="utf-8")
    print(json.dumps({"agree": agree, "seconds": info["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
