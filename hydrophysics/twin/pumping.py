"""Registered-pump electricity census -> per-cell monthly abstraction.

Groundwater abstraction on this fan is unmetered, but electricity is not: the wisenvr
`etc-tpc-etc1mon-obs` dataset carries monthly kWh for 116,769 registered pumps, each
georeferenced and labelled by purpose. Energy converts to volume through the pump's own
hydraulics,

    Q = eta * E / (rho * g * lift),        lift = ground elevation - head,

so the only unknown is the wire-to-water efficiency ``eta`` -- one parameter per purpose
class, not a free spatiotemporal field. Because ``lift`` comes from the model's simulated
head, falling heads make the same electricity deliver less water, which is a real
energy-water feedback and not a modelling artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

RHO_G = 9800.0          # N/m3, rho * g for fresh water
J_PER_KWH = 3.6e6
MIN_LIFT_M = 2.0        # floor: a near-zero lift must not imply unbounded volume
KW_PER_HP = 0.746
HOURS_PER_MONTH = 730.0
METER_COL = "電號1"      # the electricity meter number in the TPC census


def clean_census(pumps: pd.DataFrame, kwh: pd.DataFrame, cap_duty: float | None = 1.0,
                 t0: str | None = None, t1: str | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """De-duplicate shared meters and drop meters that cannot physically be pumps.

    Two defects in the raw census, found 2026-09-11 and both large:

    1. **Shared meters.** 7,869 meter numbers (``電號1``) serve more than one registered
       pump, and the monthly kWh series is attached to *every* pump on the meter. Those
       pumps carry 62% of all recorded electricity, most of it duplicated -- the two
       largest "industry" entries are one 30 HP meter counted twice at 1.28 TWh each.
       Here each meter's series is counted once and split across its pumps by rated
       horsepower, so per-cell and per-class aggregation stay pump-based.
    2. **Meters that are not pumps.** A motor cannot draw more than its rating. Yet the
       median *industrial* meter runs at 194% of ``HP x 0.746 kW x 730 h`` per month and
       the top one at 59,000%; livestock and aquaculture have 90th percentiles at 5x and
       1.5x. These are farm or factory supplies billed under an agricultural-water
       tariff, not groundwater lifted. Irrigation -- 86% of installed HP -- runs at a
       plausible 6% duty. Meters whose *mean* monthly duty exceeds ``cap_duty`` are
       dropped whole; ``cap_duty=None`` keeps them (the pre-2026-09-11 forcing).

    Returns ``(pumps, kwh, report)``: the census restricted to surviving pumps, the kWh
    table re-keyed per pump with each meter's energy shared out, and a per-class report of
    what was removed. ``t0``/``t1`` restrict the duty statistic to the modelled window.
    """
    pumps = pumps.copy()
    pumps["sid"] = pumps["sid"].astype(str)
    pumps["_hp"] = pd.to_numeric(pumps["PUMP_HP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    meter = pumps[METER_COL].astype(str) if METER_COL in pumps else pumps["sid"]
    pumps["_meter"] = meter.where(meter.notna() & (meter != "nan") & (meter != ""),
                                  pumps["sid"])

    k = kwh.copy()
    k["pump"] = k["pump"].astype(str)
    k["datetime"] = pd.to_datetime(k["datetime"])
    if t0 is not None:
        k = k[k["datetime"] >= pd.Timestamp(t0)]
    if t1 is not None:
        k = k[k["datetime"] < pd.Timestamp(t1)]
    k = k.dropna(subset=["electricity_kwh"])

    # one series per meter: the first pump's copy (they are identical duplicates)
    first_sid = pumps.groupby("_meter")["sid"].first()
    meter_of_sid = pumps.set_index("sid")["_meter"]
    k["_meter"] = k["pump"].map(meter_of_sid)
    k = k.dropna(subset=["_meter"])
    k_meter = k[k["pump"].isin(set(first_sid.values))].drop(columns=["pump"])

    hp_meter = pumps.groupby("_meter")["_hp"].sum()
    e_meter = k_meter.groupby("_meter")["electricity_kwh"].agg(["mean", "sum"])
    cap = hp_meter.reindex(e_meter.index).fillna(0.0) * KW_PER_HP * HOURS_PER_MONTH
    duty = e_meter["mean"] / cap.replace(0.0, np.nan)
    drop = set(duty[(cap_duty is not None) & (duty > (cap_duty if cap_duty else np.inf))].index)

    from .scenario import purpose_class
    pumps["_cls"] = pumps["PURPOSE"].map(purpose_class)
    pumps["_meter_kwh"] = pumps["_meter"].map(e_meter["sum"]).fillna(0.0)
    pumps["_dropped"] = pumps["_meter"].isin(drop)
    rep_rows = []
    # a meter is reported under its first pump's class, so the per-class sums add up to
    # the totals even where one meter's pumps straddle two classes
    meter_cls = pumps.groupby("_meter")["_cls"].first()
    for cls, g in pumps.groupby("_cls"):
        raw = float(k[k["pump"].isin(set(g["sid"]))]["electricity_kwh"].sum())
        per_meter = g.drop_duplicates("_meter")
        per_meter = per_meter[per_meter["_meter"].map(meter_cls) == cls]
        rep_rows.append({"class": cls, "n_pumps": int(len(g)),
                         "n_meters": int(per_meter.shape[0]),
                         "kwh_raw_GWh": raw / 1e6,
                         "kwh_dedup_GWh": float(per_meter["_meter_kwh"].sum()) / 1e6,
                         "kwh_kept_GWh": float(per_meter.loc[~per_meter["_dropped"],
                                                             "_meter_kwh"].sum()) / 1e6,
                         "n_meters_dropped": int(per_meter["_dropped"].sum())})
    report = pd.DataFrame(rep_rows).set_index("class")

    keep = pumps[~pumps["_dropped"]]
    hp_share = keep["_hp"] / keep.groupby("_meter")["_hp"].transform("sum").replace(0.0, np.nan)
    n_on_meter = keep.groupby("_meter")["sid"].transform("size")
    share = hp_share.fillna(1.0 / n_on_meter)          # HP unknown: split evenly
    share_of_sid = pd.Series(share.values, index=keep["sid"].values)
    km = k_meter.merge(keep[["sid", "_meter"]], on="_meter", how="inner")
    km["electricity_kwh"] = km["electricity_kwh"] * km["sid"].map(share_of_sid).to_numpy()
    kwh_out = km.rename(columns={"sid": "pump"})[["datetime", "electricity_kwh", "pump"]]
    pumps_out = keep.drop(columns=["_hp", "_meter", "_cls", "_meter_kwh", "_dropped"])
    return pumps_out, kwh_out.reset_index(drop=True), report


def _cell_lookup(pumps: pd.DataFrame, grid) -> dict[str, int]:
    """Map ``sid`` -> active-cell index, dropping pumps outside the grid/mask.

    ``FanGrid.active_index`` recomputes ``mask[:row].sum()`` on every call, which is
    O(nx * ny) per lookup. Over ~116k pumps that adds up, so we precompute a cumulative
    active-cell count per row once here (without touching FanGrid's interface) and reuse
    it for every pump instead of calling ``active_index`` in a loop.
    """
    mask = grid.mask
    row_counts = mask.sum(axis=1)
    row_offset = np.concatenate([[0], np.cumsum(row_counts)[:-1]]).astype("int64")
    col_cumsum = np.cumsum(mask, axis=1)  # (ny, nx), running count of active cells to the left

    x = pd.to_numeric(pumps["TWD97_X"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(pumps["TWD97_Y"], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(x) & np.isfinite(y)
    safe_x = np.where(finite, x, grid.x0)
    safe_y = np.where(finite, y, grid.y0)
    col = np.floor((safe_x - grid.x0) / grid.dx).astype("int64")
    row = np.floor((safe_y - grid.y0) / grid.dx).astype("int64")

    in_bounds = (
        finite
        & (row >= 0) & (row < grid.ny)
        & (col >= 0) & (col < grid.nx)
    )
    row_c = np.where(in_bounds, row, 0)
    col_c = np.where(in_bounds, col, 0)
    active = in_bounds & mask[row_c, col_c]

    idx = np.full(len(pumps), -1, dtype="int64")
    active_rows = row_c[active]
    active_cols = col_c[active]
    idx[active] = row_offset[active_rows] + col_cumsum[active_rows, active_cols] - 1

    sids = pumps["sid"].astype(str).to_numpy()
    return {sids[i]: int(idx[i]) for i in range(len(pumps)) if idx[i] >= 0}


def aggregate_pumps(pumps: pd.DataFrame, kwh: pd.DataFrame, grid,
                    t0: str, t1: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Monthly kWh summed into active grid cells -> ``((A, T) array, dates)``.

    Pumps whose coordinates fall outside the grid or the fan mask are dropped, never
    snapped to the nearest cell.
    """
    dates = pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq="MS", inclusive="left")
    cell = _cell_lookup(pumps, grid)

    k = kwh.copy()
    k["datetime"] = pd.to_datetime(k["datetime"]).dt.to_period("M").dt.to_timestamp()
    k = k[(k["datetime"] >= dates[0]) & (k["datetime"] <= dates[-1])]
    k["cell"] = k["pump"].astype(str).map(cell)
    k = k.dropna(subset=["cell"])
    # A NaN reading must be treated as *no contribution*, not summed in: np.add.at
    # accumulates in place, so a single NaN would NaN the whole (cell, month) bucket
    # and silently poison every other pump sharing that cell and month -- not just the
    # pump with the missing reading. Drop NaN readings before accumulating.
    k = k.dropna(subset=["electricity_kwh"])

    E = np.zeros((grid.n_active, len(dates)), dtype="float64")
    if k.empty:
        return E, dates
    tpos = {d: j for j, d in enumerate(dates)}
    k["tcol"] = k["datetime"].map(tpos)
    np.add.at(E, (k["cell"].astype(int).to_numpy(), k["tcol"].astype(int).to_numpy()),
              k["electricity_kwh"].to_numpy(dtype="float64"))
    return E, dates


def energy_to_volume(E: torch.Tensor, lift: torch.Tensor,
                     log_eta: torch.Tensor,
                     head_extra: torch.Tensor | None = None) -> torch.Tensor:
    """Convert period energy (kWh) and head (m) to pumped volume (m3) for the period.

    ``lift`` is the *static* lift, ground elevation minus the aquifer head. ``head_extra``
    is everything else the pump works against -- well drawdown, entrance and friction
    losses, and the discharge head the distribution system needs -- so the denominator is
    the **total dynamic head**, which is what a pump's energy consumption actually
    measures. Omitting it (``head_extra=None``, the pre-2026-09-09 behaviour) is retained
    only so old results reproduce.

    Why it matters here, in numbers. On the Choushui fan ground elevation is median 9.1 m
    and heads sit near the surface, so static lift is **median 6.65 m** and 20% of
    cell-months fall below ``MIN_LIFT_M``; the 1st percentile is -14.8 m (artesian). Since
    volume goes as 1/head, treating 6.65 m as the whole head implies **~25e9 m3/yr at a
    physical eta = 0.45, against a published Choushui abstraction of ~1.5-2.0e9** -- a
    12-16x overestimate. That is what drove the Stage-3 gate's ``log_eta`` onto its lower
    clamp in every fold: efficiency was the only lever available, and it was being used as
    a proxy for a missing head term. An extra 48 m at eta = 0.30 (total dynamic head
    ~55 m) reconciles the published figure, which is an ordinary number for an irrigation
    well once drawdown and 20-40 m of sprinkler discharge head are counted.
    """
    E = E.to(dtype=torch.float64)
    lift = lift.to(dtype=torch.float64)
    log_eta = log_eta.to(dtype=torch.float64)
    eta = torch.exp(log_eta)
    head = lift if head_extra is None else lift + head_extra.to(dtype=torch.float64)
    safe_head = torch.clamp(head, min=MIN_LIFT_M)
    return eta * E * J_PER_KWH / (RHO_G * safe_head)
