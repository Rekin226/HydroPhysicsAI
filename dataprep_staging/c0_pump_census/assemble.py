"""WP C0: assemble the build intermediates (results/dataprep/pump_electricity/_build_c0/) into the
product directories (no MANIFEST yet; ``finalize.py`` writes it after the checks):

* ``pump_electricity/census_L1/v<N>/census_pumps.parquet``: one row per census pump (Yunlin,
  Changhua; Chiayi pumps in active cells plus their meter-mates), harmonised field names, a
  county column, no personal column and no raw meter number.
* ``pump_electricity/{preorigin,full}/v<N>/``:
  - ``cells_by_class.npz``: value / flag / sd (C, A, T) summed over counties, the same per
    county (K, C, A, T), installed and missing HP, pumps on terminated meters, dates, grid;
  - ``cells_total.npz``: all classes and domestic excluded (A, T) with flags (the domestic-
    class sensitivity variant); full also the even-split alternative;
  - ``cells_by_class_evensplit.npz`` (full only): bimonthly bills posted wholly to the second
    month re-split evenly (flag 1);
  - ``pumps_<county>.npz``: per-pump value / flag / method (P, T), HP share, cell, class;
  - ``meters_<county>.parquet``: billing phase (code and detected), install month, termination;
  - ``county_class_monthly.parquet``: long table (county and in-grid scope, class, all,
    all_nodomestic) with flags;
  - ``build_info.json``: build statistics per county (no acceptance number).

Flags (plan 2.4): 0 observed (including structural pre-install zeros and terminated meters'
zeros, distinguished by ``method``); 1 even-split alternative only; 8 missing, never filled
(pump-month NaN reading after installation; any cell, class or total containing a pump-month of
a phase-A meter in 2026-07, whose July-August bill is not yet issued); 9 suspect (aggregate whose
missing pump readings exceed 10 % of its installed HP, or containing a monthly meter's 2026-07
reading). Missing pump readings count as zero in aggregates, as in the model.

Usage: python -I assemble.py [<results root>]     (default results/dataprep; a scratch root
       is used for dry runs)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import manifest  # noqa: E402

COUNTIES = lib.COUNTIES
K = len(COUNTIES)
PART_KEYS = ("value", "n_incomplete", "n_partial", "hp_installed", "hp_missing",
             "n_terminated", "n_resplit")


def flags_from_parts(p: dict) -> np.ndarray:
    """Aggregate flag from summed parts (see module docstring)."""
    f = np.zeros(p["value"].shape, dtype=np.uint8)
    hpi, hpm = p["hp_installed"], p["hp_missing"]
    f[(hpi > 0) & (hpm > lib.HP_MISSING_FLAG_FRAC * hpi)] = 9
    f[p["n_partial"] > 0] = 9
    f[p["n_incomplete"] > 0] = 8
    return f


def masked(value: np.ndarray, flag: np.ndarray) -> np.ndarray:
    v = value.astype(np.float64).copy()
    v[flag == 8] = np.nan
    return v


def load_cells(work: Path, variant: str) -> tuple[dict, dict]:
    """Per county: cell parts (C, A, T) and county-scope parts (C, T)."""
    cells, county = {}, {}
    for tag in COUNTIES:
        z = np.load(work / f"{variant}_{tag}_cells.npz")
        cells[tag] = {k[5:]: z[k] for k in z.files if k.startswith("cell_")}
        county[tag] = {k[7:]: z[k] for k in z.files if k.startswith("county_")}
    return cells, county


def long_rows(arrs: dict, county: str, scope: str, dates: list[str], version: str,
              source_ids: str) -> list[pd.DataFrame]:
    """Long-table rows from parts of shape (C, T) (classes) for one county x scope, plus the
    'all' and 'all_nodomestic' totals."""
    out = []
    dom = lib.CLASSES.index("domestic")
    groups = [(c, [i]) for i, c in enumerate(lib.CLASSES)]
    groups += [("all", list(range(len(lib.CLASSES)))),
               ("all_nodomestic", [i for i in range(len(lib.CLASSES)) if i != dom])]
    for name, idx in groups:
        p = {k: arrs[k][idx].sum(0) for k in PART_KEYS if k in arrs}
        f = flags_from_parts(p)
        v = masked(p["value"], f)
        method = np.where(f == 8, "incomplete_billing_pair_2026_07",
                          np.where(f == 9, "observed_suspect", "observed")).astype(object)
        out.append(pd.DataFrame({
            "county": county, "scope": scope, "class": name, "date": dates, "value": v,
            "sd": np.nan, "flag": f.astype("int64"), "method": method,
            "hp_installed": p["hp_installed"], "hp_missing": p["hp_missing"],
            "unit": "kWh/month, model units (cleaned-census kWh)",
            "product_version": version, "source_ids": source_ids}))
    return out


def assemble_variant(variant: str, work: Path, root: Path, grid, gmeta: dict) -> Path:
    d = manifest.new_version_dir(f"pump_electricity/{variant}", root=root)
    version = f"pump_electricity/{variant}/{d.name}"
    cells, county = load_cells(work, variant)
    any_c = cells[COUNTIES[0]]["value"]
    C, A, T = any_c.shape
    dates = [lib.ym(i) for i in range(T)]
    # per county and summed
    by_c_val = np.zeros((K, C, A, T))
    by_c_flag = np.zeros((K, C, A, T), dtype=np.uint8)
    tot = {k: sum(cells[t][k] for t in COUNTIES) for k in PART_KEYS}
    for i, tag in enumerate(COUNTIES):
        f = flags_from_parts(cells[tag])
        by_c_flag[i] = f
        by_c_val[i] = masked(cells[tag]["value"], f)
    flag = flags_from_parts(tot)
    value = masked(tot["value"], flag)
    sd = np.full(value.shape, np.nan)
    common = dict(classes=np.array(lib.CLASSES), counties=np.array(COUNTIES),
                  dates=np.array(dates), mask_hash=np.array(gmeta["mask_hash"]),
                  grid_hash=np.array(gmeta["grid_hash"]), nx=grid.nx, ny=grid.ny, dx=grid.dx,
                  x0=grid.x0, y0=grid.y0, n_active=grid.n_active,
                  unit=np.array("kWh/month, model units (cleaned-census kWh)"))
    lib.save_npz(d / "cells_by_class.npz", value=value, flag=flag, sd=sd,
                        value_by_county=by_c_val, flag_by_county=by_c_flag,
                        hp_installed=tot["hp_installed"], hp_missing=tot["hp_missing"],
                        n_terminated=tot["n_terminated"], **common)
    # totals and the domestic-excluded sensitivity
    dom = lib.CLASSES.index("domestic")
    nd = [i for i in range(C) if i != dom]
    tot_all = {k: tot[k].sum(0) for k in PART_KEYS}
    tot_nd = {k: tot[k][nd].sum(0) for k in PART_KEYS}
    f_all, f_nd = flags_from_parts(tot_all), flags_from_parts(tot_nd)
    extra = {}
    if variant == "full":
        ev = {t: cells[t]["value_evensplit"] for t in COUNTIES}
        ev_tot = sum(ev.values())
        f_ev = flag.copy()
        f_ev[(tot["n_resplit"] > 0) & (f_ev == 0)] = 1
        lib.save_npz(d / "cells_by_class_evensplit.npz", value=masked(ev_tot, f_ev),
                            flag=f_ev, sd=np.full(value.shape, np.nan), **common)
        fa = f_all.copy()
        fa[(tot_all["n_resplit"] > 0) & (fa == 0)] = 1
        extra = {"value_all_evensplit": masked(ev_tot.sum(0), fa), "flag_all_evensplit": fa}
    lib.save_npz(d / "cells_total.npz",
                        value_all=masked(tot_all["value"], f_all), flag_all=f_all,
                        sd_all=np.full(f_all.shape, np.nan),
                        value_nodomestic=masked(tot_nd["value"], f_nd), flag_nodomestic=f_nd,
                        sd_nodomestic=np.full(f_nd.shape, np.nan), **extra, **common)
    # per pump and per meter
    for tag in COUNTIES:
        suffix = "_ingrid" if tag == "chiayi" else ""
        shutil.copyfile(work / f"{variant}_{tag}_pumps.npz", d / f"pumps_{tag}{suffix}.npz")
        if variant == "full":
            shutil.copyfile(work / f"{variant}_{tag}_pumps_evensplit.npz",
                            d / f"pumps_{tag}{suffix}_evensplit.npz")
        mt = pd.read_parquet(work / f"{variant}_{tag}_meters.parquet")
        if tag == "chiayi":
            mt = mt[mt["in_grid_any"]].reset_index(drop=True)
        mt.to_parquet(d / f"meters_{tag}{suffix}.parquet", index=False)
    # long table: county scope (whole county's kept pumps; Chiayi = whole county) and in grid
    src = {t: ";".join(lib.rel(lib.PARQ / f"{t}_{x}.parquet") for x in ("census", "kwh_long"))
           for t in COUNTIES}
    rows = []
    for tag in COUNTIES:
        rows += long_rows(county[tag], tag, "county", dates, version, src[tag])
        ing = {k: cells[tag][k].sum(1) for k in PART_KEYS}
        rows += long_rows(ing, tag, "in_grid", dates, version, src[tag])
    ing_all = {k: tot[k].sum(1) for k in PART_KEYS}
    rows += long_rows(ing_all, "all", "in_grid", dates, version, ";".join(src.values()))
    lt = pd.concat(rows, ignore_index=True)
    lt.to_parquet(d / "county_class_monthly.parquet", index=False)
    info = {t: lib.load_json(work / f"{variant}_{t}_info.json") for t in COUNTIES}
    lib.save_json({"variant": variant, "version": version, "grid": gmeta, "counties": info,
                   "flag_rule": {"hp_missing_frac_for_flag_9": lib.HP_MISSING_FLAG_FRAC},
                   "methods": {str(k): {"name": v[0], "flag": v[1]}
                               for k, v in lib.METHODS.items()}},
                  d / "build_info.json")
    return d


def assemble_l1(work: Path, root: Path) -> Path:
    d = manifest.new_version_dir("pump_electricity/census_L1", root=root)
    parts = [pd.read_parquet(work / f"l1_{t}.parquet") for t in COUNTIES]
    l1 = pd.concat(parts, ignore_index=True)
    bad = [c for c in l1.columns if c in lib.FORBIDDEN_COLS or c in ("電號", "電號1", "電號2")]
    if bad:
        raise SystemExit(f"forbidden columns in L1: {bad}")
    l1.to_parquet(d / "census_pumps.parquet", index=False)
    summ = {t: lib.load_json(work / f"summary_{t}.json") for t in COUNTIES}
    lib.save_json({"version": f"pump_electricity/census_L1/{d.name}",
                   "rows": int(len(l1)), "rows_by_county": l1["county"].value_counts().to_dict(),
                   "build_summaries": summ}, d / "build_info.json")
    return d


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else lib.DATAPREP
    work = lib.WORK
    grid = lib.load_grid()
    gmeta = lib.grid_meta(grid)
    out = {"census_L1": str(assemble_l1(work, root))}
    for variant in ("preorigin", "full"):
        out[variant] = str(assemble_variant(variant, work, root, grid, gmeta))
    if root == lib.DATAPREP:
        lib.save_json({k: lib.rel(v) for k, v in out.items()}, work / "products.json")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
