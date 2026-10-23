"""WP C0 repair round: assemble the v2 build intermediates (``repair_build.py``) into the
product directories ``pump_electricity/{census_L1,preorigin,full}/v<N>`` (no MANIFEST yet;
``repair_finalize.py`` writes it after the registered checks).

Same layout as v1 (``assemble.py``) plus:

* pre-origin: the 2022-12 value of Dec-Jan-paired meters is the fill (flag 4) with its sd in
  ``cells_by_class.npz`` (``sd``, ``sd_by_county``), ``cells_total.npz`` (``sd_all``,
  ``sd_nodomestic``), the per-pump ``sd`` array and ``county_class_monthly.parquet`` (``sd``);
  ``fill_parameters.json`` holds the ratio parameters of every county;
* both variants: ``missing_expected_kwh`` per cell x class (and in the totals and the long
  table) and ``n_fill``;
* aggregate flags: 8 > 9 > 4 > 0 (an aggregate holding a filled pump-month is 4 unless 8 or 9
  applies; its sd is given whatever the flag).

Usage: python -I repair_assemble.py <build dir> [<results root>]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import repair_lib as rl  # noqa: E402
from common import manifest  # noqa: E402

COUNTIES = lib.COUNTIES
K = len(COUNTIES)
C = len(lib.CLASSES)
DOM = lib.CLASSES.index("domestic")
ND = [i for i in range(C) if i != DOM]
PART_KEYS = ("value", "n_incomplete", "n_partial", "hp_installed", "hp_missing",
             "n_terminated", "n_resplit", "n_fill", "missing_expected_kwh")
UNIT = "kWh/month, model units (cleaned-census kWh)"
J = rl.J_DEC22


def flags_from_parts(p: dict) -> np.ndarray:
    f = np.zeros(p["value"].shape, dtype=np.uint8)
    f[p["n_fill"] > 0] = 4
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
    cells, county = {}, {}
    for tag in COUNTIES:
        z = np.load(work / f"{variant}_{tag}_cells.npz")
        cells[tag] = {k[5:]: z[k] for k in z.files if k.startswith("cell_")}
        county[tag] = {k[7:]: z[k] for k in z.files if k.startswith("county_")}
    return cells, county


def fill_frame(work: Path, variant: str) -> pd.DataFrame:
    rows = []
    for ki, tag in enumerate(COUNTIES):
        z = np.load(work / f"{variant}_{tag}_fill.npz")
        n = len(z["pump"])
        rows.append(pd.DataFrame({"county": np.full(n, ki), "cell": z["cell"], "cls": z["cls"],
                                  "meter": ki * 10**7 + z["meter_idx"].astype("int64"),
                                  "cc": z["cc"], "ic": z["ic"], "value": z["value"]}))
    return pd.concat(rows, ignore_index=True)


def sd_array(ff: pd.DataFrame, key: np.ndarray, shape_lead: tuple, T: int) -> np.ndarray:
    """(shape_lead..., T) array, NaN except month 2022-12 where a group holds a fill."""
    out = np.full(int(np.prod(shape_lead)) * T, np.nan).reshape(-1, T)
    if len(ff) and J < T:
        s = rl.group_sd(key, ff["meter"].to_numpy(), ff["cc"].to_numpy(), ff["ic"].to_numpy())
        out[s.index.to_numpy().astype("int64"), J] = s.to_numpy()
    return out.reshape(*shape_lead, T)


def long_rows(arrs: dict, sd_by_group: dict, county: str, scope: str, dates: list[str],
              version: str, source_ids: str) -> list[pd.DataFrame]:
    out = []
    groups = [(c, [i]) for i, c in enumerate(lib.CLASSES)]
    groups += [("all", list(range(C))), ("all_nodomestic", ND)]
    for name, idx in groups:
        p = {k: arrs[k][idx].sum(0) for k in PART_KEYS if k in arrs}
        f = flags_from_parts(p)
        v = masked(p["value"], f)
        sd = np.full(len(dates), np.nan)
        if name in sd_by_group and len(dates) > J:
            sd[J] = sd_by_group[name]
        method = np.select([f == 8, f == 9, f == 4],
                           ["incomplete_billing_pair_2026_07", "observed_suspect",
                            "observed_with_fill_dec2022_dec_jan_pair"], "observed").astype(object)
        out.append(pd.DataFrame({
            "county": county, "scope": scope, "class": name, "date": dates, "value": v,
            "sd": sd, "flag": f.astype("int64"), "method": method,
            "hp_installed": p["hp_installed"], "hp_missing": p["hp_missing"],
            "missing_expected_kwh": p["missing_expected_kwh"], "n_fill": p["n_fill"],
            "unit": UNIT, "product_version": version, "source_ids": source_ids}))
    return out


def group_sds(ff: pd.DataFrame) -> dict:
    """sd of the fill for the long-table groups (class, all, all_nodomestic) of a frame."""
    out = {}
    if not len(ff):
        return out
    m, cc, ic = ff["meter"].to_numpy(), ff["cc"].to_numpy(), ff["ic"].to_numpy()
    cls = ff["cls"].to_numpy()
    s = rl.group_sd(cls, m, cc, ic)
    for i, c in enumerate(lib.CLASSES):
        if i in s.index:
            out[c] = float(s.loc[i])
    out["all"] = float(rl.group_sd(np.zeros(len(ff), "int64"), m, cc, ic).iloc[0])
    nd = np.where(cls != DOM, 0, -1)
    if (nd == 0).any():
        out["all_nodomestic"] = float(rl.group_sd(nd, m, cc, ic).iloc[0])
    return out


def assemble_variant(variant: str, work: Path, root: Path, grid, gmeta: dict) -> Path:
    d = manifest.new_version_dir(f"pump_electricity/{variant}", root=root)
    version = f"pump_electricity/{variant}/{d.name}"
    cells, county = load_cells(work, variant)
    C_, A, T = cells[COUNTIES[0]]["value"].shape
    dates = [lib.ym(i) for i in range(T)]
    ff = fill_frame(work, variant)
    ffg = ff[ff["cell"] >= 0]
    by_c_val = np.zeros((K, C, A, T))
    by_c_flag = np.zeros((K, C, A, T), dtype=np.uint8)
    by_c_sd = np.full((K, C, A, T), np.nan)
    tot = {k: sum(cells[t][k] for t in COUNTIES) for k in PART_KEYS}
    for i, tag in enumerate(COUNTIES):
        f = flags_from_parts(cells[tag])
        by_c_flag[i] = f
        by_c_val[i] = masked(cells[tag]["value"], f)
        fc = ffg[ffg["county"] == i]
        by_c_sd[i] = sd_array(fc, (fc["cls"] * A + fc["cell"]).to_numpy(), (C, A), T)
    flag = flags_from_parts(tot)
    value = masked(tot["value"], flag)
    sd = sd_array(ffg, (ffg["cls"] * A + ffg["cell"]).to_numpy(), (C, A), T)
    common = dict(classes=np.array(lib.CLASSES), counties=np.array(COUNTIES),
                  dates=np.array(dates), mask_hash=np.array(gmeta["mask_hash"]),
                  grid_hash=np.array(gmeta["grid_hash"]), nx=grid.nx, ny=grid.ny, dx=grid.dx,
                  x0=grid.x0, y0=grid.y0, n_active=grid.n_active, unit=np.array(UNIT))
    lib.save_npz(d / "cells_by_class.npz", value=value, flag=flag, sd=sd,
                 value_by_county=by_c_val, flag_by_county=by_c_flag, sd_by_county=by_c_sd,
                 hp_installed=tot["hp_installed"], hp_missing=tot["hp_missing"],
                 n_terminated=tot["n_terminated"], n_fill=tot["n_fill"],
                 missing_expected_kwh=tot["missing_expected_kwh"], **common)
    tot_all = {k: tot[k].sum(0) for k in PART_KEYS}
    tot_nd = {k: tot[k][ND].sum(0) for k in PART_KEYS}
    f_all, f_nd = flags_from_parts(tot_all), flags_from_parts(tot_nd)
    sd_all = sd_array(ffg, ffg["cell"].to_numpy(), (A,), T)
    ffn = ffg[ffg["cls"] != DOM]
    sd_nd = sd_array(ffn, ffn["cell"].to_numpy(), (A,), T)
    extra = {}
    if variant == "full":
        ev_tot = sum(cells[t]["value_evensplit"] for t in COUNTIES)
        f_ev = flag.copy()
        f_ev[(tot["n_resplit"] > 0) & (f_ev == 0)] = 1
        lib.save_npz(d / "cells_by_class_evensplit.npz", value=masked(ev_tot, f_ev),
                     flag=f_ev, sd=np.full(value.shape, np.nan), **common)
        fa = f_all.copy()
        fa[(tot_all["n_resplit"] > 0) & (fa == 0)] = 1
        extra = {"value_all_evensplit": masked(ev_tot.sum(0), fa), "flag_all_evensplit": fa}
    lib.save_npz(d / "cells_total.npz",
                 value_all=masked(tot_all["value"], f_all), flag_all=f_all, sd_all=sd_all,
                 value_nodomestic=masked(tot_nd["value"], f_nd), flag_nodomestic=f_nd,
                 sd_nodomestic=sd_nd,
                 missing_expected_kwh_all=tot_all["missing_expected_kwh"],
                 missing_expected_kwh_nodomestic=tot_nd["missing_expected_kwh"],
                 **extra, **common)
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
    src = {t: ";".join(lib.rel(lib.PARQ / f"{t}_{x}.parquet") for x in ("census", "kwh_long"))
           for t in COUNTIES}
    rows = []
    for ki, tag in enumerate(COUNTIES):
        fc = ff[ff["county"] == ki]
        rows += long_rows(county[tag], group_sds(fc), tag, "county", dates, version, src[tag])
        ing = {k: cells[tag][k].sum(1) for k in PART_KEYS}
        rows += long_rows(ing, group_sds(fc[fc["cell"] >= 0]), tag, "in_grid", dates, version,
                          src[tag])
    ing_all = {k: tot[k].sum(1) for k in PART_KEYS}
    rows += long_rows(ing_all, group_sds(ffg), "all", "in_grid", dates, version,
                      ";".join(src.values()))
    lt = pd.concat(rows, ignore_index=True)
    lt.to_parquet(d / "county_class_monthly.parquet", index=False)
    info = {t: lib.load_json(work / f"{variant}_{t}_info.json") for t in COUNTIES}
    summ = {t: lib.load_json(work / f"summary_{t}.json") for t in COUNTIES}
    if variant == "preorigin":
        lib.save_json({"method": "fill = rho_bar(county, class) x (Oct-2022 + Nov-2022)/2 for "
                                 "Dec-Jan-paired kept meters; see repair_lib docstring",
                       "fit_years_dec_y_jan_y1": list(rl.FIT_YEARS),
                       "n_min_meters_per_class_year": rl.N_MIN_CLASS,
                       "counties": {t: summ[t]["fill_params"] for t in COUNTIES}},
                      d / "fill_parameters.json")
    lib.save_json({"variant": variant, "version": version, "grid": gmeta, "counties": info,
                   "flag_rule": {"hp_missing_frac_for_flag_9": lib.HP_MISSING_FLAG_FRAC,
                                 "precedence": "8 > 9 > 4 > 0"},
                   "methods": {str(k): {"name": v[0], "flag": v[1]}
                               for k, v in rl.METHODS.items()}},
                  d / "build_info.json")
    return d


def assemble_l1(work: Path, root: Path) -> Path:
    d = manifest.new_version_dir("pump_electricity/census_L1", root=root)
    l1 = pd.concat([pd.read_parquet(work / f"l1_{t}.parquet") for t in COUNTIES],
                   ignore_index=True)
    bad = [c for c in l1.columns if c in lib.FORBIDDEN_COLS or c in ("電號", "電號1", "電號2")]
    if bad:
        raise SystemExit(f"forbidden columns in L1: {bad}")
    l1.to_parquet(d / "census_pumps.parquet", index=False)
    summ = {t: lib.load_json(work / f"summary_{t}.json") for t in COUNTIES}
    for s in summ.values():
        s.pop("fill_params", None)
    lib.save_json({"version": f"pump_electricity/census_L1/{d.name}",
                   "rows": int(len(l1)), "rows_by_county": l1["county"].value_counts().to_dict(),
                   "build_summaries": summ}, d / "build_info.json")
    return d


def main(argv: list[str]) -> int:
    work = Path(argv[0])
    root = Path(argv[1]) if len(argv) > 1 else lib.DATAPREP
    grid = lib.load_grid()
    gmeta = lib.grid_meta(grid)
    out = {"census_L1": str(assemble_l1(work, root))}
    for variant in ("preorigin", "full"):
        out[variant] = str(assemble_variant(variant, work, root, grid, gmeta))
    rec = {k: (lib.rel(v) if root == lib.DATAPREP else v) for k, v in out.items()}
    lib.save_json(rec, work / "products_v2.json")
    if root == lib.DATAPREP:
        lib.save_json(rec, lib.EVIDENCE / "repair" / "products_v2.json")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
