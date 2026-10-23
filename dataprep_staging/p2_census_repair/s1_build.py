"""P2a step 1: sanitized pump and meter tables, dense meter x month kWh matrix, duty cap,
pump shares and grid cells (port of pilot s01_build.py, s02_billing.py, s03).

Writes intermediates to ``results/dataprep/yunlin_electricity/_work_p2abc/`` (gitignored):
  pumps.parquet    per pump: sid, meter_id, town, cls, purpose, hp, dia, bill, inst, x, y,
                   cell (active-cell index or -1), share (HP share of its meter), kept
  meters.parquet   per meter: meter_id, idx (first pump), n_pumps, hp, cls, bill, n_bill, town,
                   inst, x, y, duty (2012-01..2022-12), kept
  meter_mat.npy    float32 (n_meters, 223) kWh, NaN preserved, 2007-01..2025-07
  build.json       counts and checks

The raw meter number (電號1) is used only to group pumps and is never written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

from hydrophysics.twin.pumping import _cell_lookup  # noqa: E402
from hydrophysics.twin.scenario import purpose_class  # noqa: E402


def main() -> None:
    t_start = time.time()
    lib.WORK.mkdir(parents=True, exist_ok=True)
    p = lib.read_census()
    p["sid"] = p["sid"].astype(str)
    meter_key = p["電號1"].astype(str)          # Int64, no nulls (checked); grouping only
    meter_id = pd.factorize(meter_key)[0]
    hp_raw = pd.to_numeric(p["PUMP_HP"], errors="coerce")
    pumps = pd.DataFrame({
        "idx": np.arange(len(p)), "sid": p["sid"].values, "meter_id": meter_id,
        "town": p["TOWN"].values, "purpose": p["PURPOSE"].values,
        "cls": p["PURPOSE"].map(purpose_class).values,
        "hp": hp_raw.values, "dia": pd.to_numeric(p["W_TUBE_DIA"], errors="coerce").values,
        "bill": p["每期用電度數1"].values,
        "inst": [lib.roc_to_ts(v) for v in p["新設日期1"].values],
        "x": pd.to_numeric(p["TWD97_X"], errors="coerce").values,
        "y": pd.to_numeric(p["TWD97_Y"], errors="coerce").values,
    })
    del p, meter_key
    assert not pumps["sid"].duplicated().any()

    # ---- dense pump x month matrix (pilot s01)
    k = pd.read_parquet(lib.KWH)
    k["pump"] = k["pump"].astype(str)
    row = pd.Series(pumps["idx"].values, index=pumps["sid"].values)
    r = k["pump"].map(row)
    n_unmatched = int(r.isna().sum())
    k = k[r.notna()]
    r = r[r.notna()].astype(int).to_numpy()
    tpos = pd.Series(np.arange(lib.T_ALL), index=lib.DATES)
    c = tpos.reindex(pd.to_datetime(k["datetime"]).values).to_numpy()
    assert not np.isnan(c).any(), "kWh month outside 2007-01..2025-07"
    mat = np.full((len(pumps), lib.T_ALL), np.nan, dtype=np.float32)
    mat[r, c.astype(int)] = k["electricity_kwh"].to_numpy(dtype=np.float32)
    n_rows = len(k)
    del k

    # ---- meters (pilot s02; first pump's series and attributes)
    g = pumps.groupby("meter_id")
    hp0 = pumps["hp"].fillna(0.0).clip(lower=0.0)           # twin convention for the cap
    mt = pd.DataFrame({
        "idx": g["idx"].first(), "n_pumps": g.size(), "hp": hp0.groupby(pumps["meter_id"]).sum(),
        "hp_pilot": g["hp"].sum(), "cls": g["cls"].first(), "bill": g["bill"].first(),
        "n_bill": g["bill"].nunique(), "town": g["town"].first(), "inst": g["inst"].min(),
        "x": g["x"].first(), "y": g["y"].first(),
    })
    mt.index.name = "meter_id"
    mt = mt.reset_index()
    M = mat[mt["idx"].to_numpy()]
    # identical copies on shared meters (the twin keeps the first copy): every pump's row
    # should equal its meter's first row
    first_idx = mt.set_index("meter_id")["idx"]
    first_of_pump = first_idx.reindex(pumps["meter_id"]).to_numpy()
    same = np.all(np.isclose(np.nan_to_num(mat, nan=-1), np.nan_to_num(mat[first_of_pump], nan=-1)),
                  axis=1)
    n_diff = int((~same).sum())
    del mat

    # ---- duty cap on the frozen window (clean_census, cap_duty=1.0, 2012-01..2022-12)
    t0, t1 = lib.tidx(lib.DUTY_WINDOW[0]), lib.tidx(lib.DUTY_WINDOW[1])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(M[:, t0:t1 + 1].astype(float), axis=1)
    cap = mt["hp"].to_numpy() * lib.KW_PER_HP * lib.HOURS_PER_MONTH
    duty = mean / np.where(cap > 0, cap, np.nan)
    mt["duty"] = duty
    mt["kept"] = ~(duty > lib.CAP_DUTY)

    # ---- pump shares (HP share within the meter; even split if the meter's HP sum is 0)
    pumps["kept"] = pumps["meter_id"].map(mt.set_index("meter_id")["kept"]).to_numpy()
    hp_sum = hp0.groupby(pumps["meter_id"]).transform("sum")
    n_on = pumps.groupby("meter_id")["idx"].transform("size")
    share = (hp0 / hp_sum.replace(0.0, np.nan)).fillna(1.0 / n_on)
    pumps["share"] = share.to_numpy()
    grid = lib.fan_grid()
    cells = _cell_lookup(pumps.rename(columns={"x": "TWD97_X", "y": "TWD97_Y"}), grid)
    pumps["cell"] = pumps["sid"].map(cells).fillna(-1).astype(int).to_numpy()

    pumps.to_parquet(lib.WORK / "pumps.parquet", index=False)
    mt.to_parquet(lib.WORK / "meters.parquet", index=False)
    np.save(lib.WORK / "meter_mat.npy", M)
    out = {
        "n_pumps": len(pumps), "n_meters": len(mt), "kwh_rows": n_rows,
        "kwh_rows_unmatched": n_unmatched, "pumps_differing_from_meter_first_copy": n_diff,
        "n_meters_shared": int((mt["n_pumps"] > 1).sum()),
        "n_meters_kept": int(mt["kept"].sum()), "n_meters_dropped": int((~mt["kept"]).sum()),
        "n_pumps_on_active_grid": int((pumps["cell"] >= 0).sum()),
        "n_pumps_kept_on_grid": int(((pumps["cell"] >= 0) & pumps["kept"]).sum()),
        "matrix_nan_frac": float(np.isnan(M).mean()),
        "grid": lib.grid_meta(grid), "seconds": round(time.time() - t_start, 1),
    }
    lib.save_json(out, lib.WORK / "build.json")
    print(out)


if __name__ == "__main__":
    main()
