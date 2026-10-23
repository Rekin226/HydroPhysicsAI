"""Fan-scale sensitivity member of the public-well closure ramp (plan WP P6 step 3; check M1).

Fixed by ``p6const.py`` and declared in ``P6.inventory_member.v1`` before construction:

* cohorts = the cells of annex table 2-2 (agency column x year 100-108), volume v (10^4
  m3/yr, WRA physical, agency-reported abstraction cut); a cohort of year Y abstracts at
  v x 1e4 / 365.25 m3/day in every month before its stop month (Y-07; variants Y-01 and
  (Y+1)-01) and nothing afterwards; time axis 2007-01..2022-12;
* layers: aquifers 3-4 equally (twin layers 2, 3); variant: irrigation-association cohorts
  in aquifer 2 (twin layer 1);
* locations: proxy points per cohort (drought wells of the agency, census public-supply or
  industrial pumps), equal share per point, out-of-grid shares dropped; variants census only,
  drought wells only, uniform over the county's active cells;
* the pending Changhua Taiwan Water wells (129, 109-113) are a separate, constant component.

Arrays are abstraction rates in m3/day (positive = abstraction; pass ``-rate`` as the
engine's ``q_extra``), shape (4 twin layers, 2148 active cells, 192 months), physical units;
model units = physical / 2.58 (plan, stated assumption). The member never enters the primary
model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p6_public_wells import p6const as K  # noqa: E402
from p6_public_wells import p6lib as L  # noqa: E402

LOC_VARIANT = {"primary": "primary", "annex_2011_2018": "primary", "timing_early": "primary",
               "timing_late": "primary", "layers_ia_aq2": "primary",
               "loc_census_only": "census_only", "loc_drought_only": "drought_only",
               "loc_uniform_county": "uniform_county"}
AGENCY_LABEL = {"taiwan_water": "Taiwan Water Corporation (台水)",
                "irrigation_association": "Irrigation Agency (農水署)",
                "taiwan_sugar": "Taiwan Sugar (台糖)"}


def cohorts(t22: dict) -> pd.DataFrame:
    """One row per (agency column, year) of annex table 2-2 with a count or a volume."""
    cr, vr = t22["counts"]["rows"], t22["volumes"]["rows"]
    rows = []
    for roc in K.YEARS_ROC:
        lab = f"{roc}年"
        for j, col in enumerate(K.COHORT_COLUMNS):
            n = cr[lab]["values"][j]
            v = vr[lab]["values"][j]
            note = vr[lab]["notes"].get(col["key"])
            if n is None and v is None and note is None:
                continue
            rows.append({"cohort": f"{col['key']}_{roc + 1911}", "key": col["key"],
                         "county": col["county"], "agency": col["agency"],
                         "column_label": col["label"], "year_roc": roc, "year": roc + 1911,
                         "n_wells_printed": n, "volume_1e4m3_per_yr": v or 0.0,
                         "volume_printed": v is not None, "note": note or "",
                         "proxy": col["proxy"], "town": col.get("town", "")})
    return pd.DataFrame(rows)


def _points(spec: str, county: str, dw: pd.DataFrame, cen: pd.DataFrame) -> tuple[
        np.ndarray, str]:
    """Cells (-1 = outside the grid) of the proxy points of a cohort, and the proxy used."""
    if spec.startswith("drought:"):
        ag = spec.split(":", 1)[1]
        p = dw[(dw.county_key == county) & (dw.agency == ag)]
        return p.cell.to_numpy(), spec
    if spec == "drought_all":
        return dw[dw.county_key == county].cell.to_numpy(), spec
    if spec == "census_public":
        p = cen[(cen.county == county) & cen.is_public]
        return p.cell_xy.to_numpy(), spec
    if spec.startswith("census_industrial_town:"):
        town = spec.split(":", 1)[1]
        for name, sel in (("census_industrial_town", cen.is_industrial),
                          ("census_public_town", cen.is_public)):
            p = cen[(cen.county == county) & sel & (cen.town_key == town)]
            if len(p):
                return p.cell_xy.to_numpy(), f"{name}:{town}"
        return _points("census_public", county, dw, cen)
    raise ValueError(f"unknown proxy {spec!r}")


def cohort_weights(co: pd.DataFrame, loc: str, dw: pd.DataFrame, cen: pd.DataFrame,
                   cc: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """(n_cohorts, A) weights (sum = in-grid fraction), proxy used, points, in-grid points."""
    A = L.grid()["n_active"]
    W = np.zeros((len(co), A))
    used, npts, nin = [], np.zeros(len(co), int), np.zeros(len(co), int)
    for i, r in enumerate(co.itertuples()):
        if loc == "uniform_county":
            cells = np.flatnonzero(cc == K.COUNTY_ZH[r.county])
            spec = "uniform_county"
        else:
            spec = r.proxy
            if loc == "census_only" and r.agency != "factory":
                spec = "census_public"
            if loc == "drought_only" and not spec.startswith("drought:"):
                spec = "drought_all"
            if spec.startswith("drought:"):
                has = ((dw.county_key == r.county) & (dw.agency == spec.split(":", 1)[1])).any()
                if not has:
                    raise ValueError(f"{r.cohort}: no drought well of {spec}")
            cells, spec = _points(spec, r.county, dw, cen)
        if len(cells) == 0:
            raise ValueError(f"{r.cohort}: no proxy point ({spec})")
        npts[i], nin[i] = len(cells), int((cells >= 0).sum())
        ok = cells[cells >= 0]
        np.add.at(W[i], ok, 1.0 / len(cells))
        used.append(spec)
    return W, used, npts, nin


def stop_index(year: np.ndarray, month: int, dates: list[str]) -> np.ndarray:
    """Index of the stop month (first month without abstraction); month 13 = next January."""
    y = np.asarray(year, int) + (month - 1) // 12
    m = (month - 1) % 12 + 1
    keys = [f"{a:04d}-{m:02d}" for a in y]
    pos = {d: i for i, d in enumerate(dates)}
    return np.array([pos.get(k, len(dates)) for k in keys])


def layer_weights(agency: str, variant: str) -> np.ndarray:
    w = np.zeros(K.N_LAYERS)
    if variant == "layers_ia_aq2" and agency == "irrigation_association":
        w[K.IA_VARIANT_AQUIFER - 1] = 1.0
    else:
        for aq, s in zip(K.MEMBER_AQUIFERS, K.MEMBER_AQUIFER_SPLIT, strict=True):
            w[aq - 1] = s
    return w


def build(t22: dict, t23: dict, dw: pd.DataFrame, cen: pd.DataFrame) -> dict:
    dates = L.month_axis()
    T = len(dates)
    A = L.grid()["n_active"]
    cc = L.cell_county()
    cen = cen.copy()
    ok = np.isfinite(cen.x_twd97) & np.isfinite(cen.y_twd97)
    cen = cen[ok].reset_index(drop=True)
    cen["cell_xy"] = L.cell_index(cen.x_twd97.to_numpy(), cen.y_twd97.to_numpy())
    co = cohorts(t22)
    rate = co.volume_1e4m3_per_yr.to_numpy() * 1e4 / K.DAYS_PER_YEAR
    locs = {}
    for loc in sorted(set(LOC_VARIANT.values())):
        W, used, npts, nin = cohort_weights(co, loc, dw, cen, cc)
        locs[loc] = {"W": W, "proxy_used": used, "n_points": npts, "n_points_in_grid": nin}
    out, checks = {}, {}
    for var in K.VARIANTS:
        loc = locs[LOC_VARIANT[var]]
        month = K.TIMING_VARIANTS.get(var, K.DISPOSAL_MONTH)
        stop = stop_index(co.year.to_numpy(), month, dates)
        keep = np.ones(len(co), bool)
        if var == "annex_2011_2018":
            keep = co.year_roc.isin(K.ANNEX_YEARS_ROC).to_numpy()
        active = (np.arange(T)[None, :] < stop[:, None]) & keep[:, None]       # (C, T)
        X = np.zeros((K.N_LAYERS, A, T))
        for agency in co.agency.unique():
            sel = (co.agency == agency).to_numpy()
            lw = layer_weights(agency, var)
            cell_t = (loc["W"][sel] * rate[sel][:, None]).T @ active[sel].astype(float)
            X += lw[:, None, None] * cell_t[None, :, :]
        want = (rate[:, None] * active * loc["W"].sum(1)[:, None]).sum(0)
        got = X.sum((0, 1))
        rel = float(np.max(np.abs(got - want) / np.maximum(want, 1e-12)))
        checks[var] = {"mass_balance_max_rel_err": rel, "stop_month_of_year": month,
                       "cohorts_used": int(keep.sum())}
        out[var] = X
    # pending Changhua Taiwan Water component (constant through 2022-12)
    full = t23["volume"]["tw_changhua"]["full_target"]
    done = t23["volume"]["tw_changhua"]["executed_to_108"]
    pend_v = full - done
    cen_cells, _ = _points("drought:" + AGENCY_LABEL["taiwan_water"], "changhua", dw, cen)
    wp = np.zeros(A)
    np.add.at(wp, cen_cells[cen_cells >= 0], 1.0 / len(cen_cells))
    lw = layer_weights("taiwan_water", "primary")
    pend = lw[:, None, None] * (wp * pend_v * 1e4 / K.DAYS_PER_YEAR)[None, :, None] * np.ones(
        (1, 1, T))
    pend_info = {"volume_1e4m3_per_yr": pend_v, "full_target": full, "executed_to_108": done,
                 "n_wells": K.PENDING_TW_CH_WELLS, "in_grid_fraction": float(wp.sum()),
                 "proxy": "drought:" + AGENCY_LABEL["taiwan_water"]}
    return {"dates": dates, "cohorts": co, "rate_m3d": rate, "locs": locs, "arrays": out,
            "checks": checks, "pending": pend, "pending_info": pend_info}


def annual_summary(res: dict) -> pd.DataFrame:
    """In-grid member volume (Mm3/yr, physical and model units) by variant, year and county of
    the cell (engine cell_county; 'other_cells' = cells of Chiayi or Nantou)."""
    cc = L.cell_county()
    dates = res["dates"]
    yrs = np.array([int(d[:4]) for d in dates])
    rows = []
    arrs = dict(res["arrays"])
    arrs["pending_tw_changhua"] = res["pending"]
    for var, X in arrs.items():
        cell_t = X.sum(0)                                           # (A, T) m3/day
        for county in ("changhua", "yunlin", "other_cells"):
            m = (cc == K.COUNTY_ZH[county]) if county != "other_cells" else ~np.isin(
                cc, [K.COUNTY_ZH["changhua"], K.COUNTY_ZH["yunlin"]])
            s = cell_t[m].sum(0)
            for y in np.unique(yrs):
                v = float(s[yrs == y].mean() * K.DAYS_PER_YEAR / 1e6)
                rows.append({"variant": var, "county": county, "year": int(y),
                             "in_grid_Mm3_per_yr_physical": v,
                             "in_grid_Mm3_per_yr_model_units": v / K.WRA_TO_MODEL})
    return pd.DataFrame(rows)
