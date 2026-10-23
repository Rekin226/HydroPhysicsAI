"""L3: monthly district deliveries spread on the twin's 1 km grid with the v3 allocation.

The v3 field (results/twin/surface_water_v3.npz, hydrophysics.twin.surface_water) spreads a
district's monthly volume over its farmland (NLSC LUIMAP 2024 paddy + dry field, dry weight
1) inside the bounding box, so a district's in-fan fraction is the in-fan share of that
farmland (constant over time). The per-cell share is recovered from v3's stored arrays:
share_d(c) = farm_d(c) / farm_d_total_bbox, with farm_d_total_bbox = sum_c farm_d(c) /
infan_fraction_d and infan_fraction_d = vol_fan_m3 / district volume of the month (checked
constant). Reusing it is checked by rebuilding v3's own sw_m_per_day from its district
volumes.

The LUIMAP 2024 snapshot is post-origin: for the pre-origin variant it is an exemption (U3),
with a sensitivity run that spreads the same in-fan volume by district area instead of
farmland (allocation_sensitivity).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from w1_canal.config import CALENDAR, TWIN_GRID_NPZ, V3_NPZ

DISTRICT_ROWS = {"chc": 0, "yl": 1}           # v3 order: changhua, yunlin


def _v3_month_weights() -> dict[int, float]:
    tot = {1: 0.0, 2: 0.0}
    for _m, (c, w) in CALENDAR.items():
        tot[c] += w
    return {m: w / tot[c] for m, (c, w) in CALENDAR.items()}


def v3_allocation() -> dict:
    z = np.load(V3_NPZ, allow_pickle=True)
    seasons = pd.read_csv(io.StringIO(str(z["seasons_csv"])), index_col=[0, 1])
    dates = pd.to_datetime(z["dates"])
    wm = _v3_month_weights()
    out = {"n_active": int(z["n_active"]), "dx": float(z["dx"]), "dates_v3": dates}
    vol_district = np.zeros_like(z["vol_fan_m3"])
    for d, di in DISTRICT_ROWS.items():
        name = str(z["districts"][di])
        farm = z["paddy_m2"][di] + float(z["dry_weight"]) * z["dry_m2"][di]
        frac = []
        for t, day in enumerate(dates):
            if day.month not in CALENDAR:
                continue
            V = seasons.loc[(name, day.year), f"crop{CALENDAR[day.month][0]}"] * wm[day.month]
            vol_district[di, t] = V
            frac.append(z["vol_fan_m3"][di, t] / V)
        frac = np.array(frac)
        if np.ptp(frac) > 1e-9:
            raise ValueError(f"{name}: in-fan fraction not constant ({frac.min()}..{frac.max()})")
        f = float(frac.mean())
        out[d] = {"share": farm * f / farm.sum(), "infan_fraction": f,
                  "farm_infan_m2": float(farm.sum()),
                  "area_share": z["district_share"][di]}
    # reproduction check of v3's field from its own district volumes
    rebuilt = spread({d: vol_district[di] / 1e8 for d, di in DISTRICT_ROWS.items()},
                     out, dates)
    err = np.abs(rebuilt - z["sw_m_per_day"])
    out["v3_rebuild_max_abs_m_per_day"] = float(err.max())
    out["v3_rebuild_max_rel"] = float(err.max() / np.abs(z["sw_m_per_day"]).max())
    return out


def spread(vol_1e8: dict, alloc: dict, dates: pd.DatetimeIndex, key: str = "share"):
    """(A, T) m/day of the cell from district monthly volumes (10^8 m3)."""
    area = alloc["dx"] ** 2
    days = np.array([d.days_in_month for d in dates], dtype=float)
    sw = np.zeros((alloc["n_active"], len(dates)))
    for d in DISTRICT_ROWS:
        sw += np.outer(alloc[d][key], np.asarray(vol_1e8[d]) * 1e8) / area / days[None, :]
    return sw


def spread_sd(sd_1e8: dict, alloc: dict, dates: pd.DatetimeIndex):
    area = alloc["dx"] ** 2
    days = np.array([d.days_in_month for d in dates], dtype=float)
    var = np.zeros((alloc["n_active"], len(dates)))
    for d in DISTRICT_ROWS:
        var += (np.outer(alloc[d]["share"], np.asarray(sd_1e8[d]) * 1e8) / area
                / days[None, :]) ** 2
    return np.sqrt(var)


def allocation_sensitivity(alloc: dict, vol_1e8: dict, dates) -> dict:
    """Spread the same in-fan volume by district area (no land use) instead of farmland."""
    alt = {"n_active": alloc["n_active"], "dx": alloc["dx"]}
    for d in DISTRICT_ROWS:
        a = alloc[d]["area_share"]
        alt[d] = {"share": a * alloc[d]["infan_fraction"] / a.sum()}
    base = spread(vol_1e8, alloc, dates).mean(1)
    other = spread(vol_1e8, alt, dates).mean(1)
    sel = (base > 0) | (other > 0)
    rel = np.abs(other[sel] - base[sel]) / np.maximum(base[sel], 1e-12)
    return {"cells_compared": int(sel.sum()),
            "corr_mean_depth": float(np.corrcoef(base[sel], other[sel])[0, 1]),
            "median_abs_rel_diff": float(np.median(rel)),
            "frac_cells_rel_diff_gt_0.5": float((rel > 0.5).mean()),
            "fan_total_unchanged": bool(abs(base.sum() - other.sum()) / base.sum() < 1e-9),
            "note": "same in-fan volume per district; only the within-district pattern "
                    "changes (farmland LUIMAP 2024 -> county area)"}


def twin_grid() -> dict:
    g = np.load(TWIN_GRID_NPZ, allow_pickle=True)
    return {k: g[k] for k in ("nx", "ny", "dx", "x0", "y0", "mask")}
