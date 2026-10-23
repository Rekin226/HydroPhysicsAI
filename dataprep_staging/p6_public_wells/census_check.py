"""Census reconciliation of the public wells (check C1 of ``P6.inventory_member.v1``).

Descriptive, no threshold. Inputs (loader guard, use 'prior'): C0 ``census_L1/v3`` pump table
(explicit columns: county, town, coordinates, cell, HP, purpose, kept; never names or
addresses) and C0 ``preorigin/v3`` per-pump kWh arrays (kept pumps, in grid, 2007-01..2022-12;
energy summaries use 2012-2021, flags 0-5). The drought wells are those of
:func:`wells_located`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from n1_natural_experiments import parse_sources as N1P  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402
from p6_public_wells import p6lib as L  # noqa: E402

AGENCY_KEY = {"Taiwan Water Corporation (台水)": "taiwan_water",
              "Irrigation Agency (農水署)": "irrigation_association",
              "Taiwan Sugar (台糖)": "taiwan_sugar"}


def _well_name(r) -> str:
    """Well name as printed on the row's line. N1's parser drops every token equal to the
    count (so '二林 1 號井重鑿' became '二林 號井重鑿'); here only the trailing count goes."""
    pat = re.compile(rf"^\s*{int(r.table_no)}\s.*{r.x_twd97:.1f}\s+{r.y_twd97:.1f}")
    lines = [ln for ln in N1P.doc_text("drought_wells").splitlines() if pat.search(ln)]
    if not lines:
        return str(r.well_name or "")
    ln = lines[0]
    head = ln[:ln.index(f"{r.x_twd97:.1f}")]
    head = re.sub(rf"^\s*{int(r.table_no)}\s+", "", head)
    for k in N1P.AGENCIES:
        head = head.replace(k, " ")
    head = re.sub(r"第[五十一]+區處|雲林管理處|彰化管理處|管理處", " ", head)
    toks = [t for t in head.split() if not re.search(r"[鄉鎮市區]$", t)]
    if toks and toks[-1].isdigit():
        toks = toks[:-1]
    return " ".join(toks)


def wells_located() -> tuple[pd.DataFrame, dict]:
    """The 2022 drought wells of Changhua and Yunlin (wra111 annex tables 1-8 and 1-10), parsed
    from the L0 text layer with N1's parser, with the active cell. Published 2023-04."""
    dw = N1P.drought_wells()
    df = pd.DataFrame(dw["rows"])
    df["well_name"] = [_well_name(r) for r in df.itertuples()]
    df["county_key"] = df.county.str.lower()
    df["agency_key"] = df.agency.map(AGENCY_KEY)
    df["cell"] = L.cell_index(df.x_twd97.to_numpy(), df.y_twd97.to_numpy())
    df["in_grid"] = df.cell >= 0
    df["town_key"] = df.township.map(L.town_key)
    df["source"] = "wra111_drought_wells_report.pdf annex table " + np.where(
        df.county == "Changhua", "1-8", "1-10")
    df["vintage"] = K.VINTAGE["drought_wells"]
    df["month"] = "2022-12"            # the inventory describes wells existing in 2022
    return df, dw["checks"]


def census_table() -> pd.DataFrame:
    c = L.census("prior")
    c["town_key"] = c.town.map(L.town_key)
    c["is_public"] = c.purpose_raw.isin(K.PUBLIC_PURPOSES)
    c["is_industrial"] = c.purpose_raw.isin(K.INDUSTRIAL_PURPOSES)
    return c


def purpose_counts(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby(["county", "purpose_raw", "purpose_class"], dropna=False)
    out = g.agg(n=("pump_hp", "size"), in_grid=("in_grid", "sum"), kept=("kept", "sum"),
                hp_ge50=("pump_hp", lambda s: int((s >= 50).sum())),
                hp_gt10=("pump_hp", lambda s: int((s > 10).sum())),
                hp_max=("pump_hp", "max"), hp_median=("pump_hp", "median")).reset_index()
    out["purpose_raw"] = out.purpose_raw.fillna("(null)")
    return out


def plan_counts_check(c: pd.DataFrame) -> dict:
    """The plan's Yunlin counts (WP P6 'Gap'), recomputed on census_L1/v3."""
    y = c[c.county == "yunlin"]
    got = {"public_supply_公共用水": int((y.purpose_raw == "公共用水").sum()),
           "industrial_工業用水": int((y.purpose_raw == "工業用水").sum()),
           "domestic_and_public_家用及公共給水": int((y.purpose_raw == "家用及公共給水").sum()),
           "hp_ge50": int((y.pump_hp >= 50).sum()), "hp_gt10": int((y.pump_hp > 10).sum()),
           "hp_max": float(y.pump_hp.max())}
    plan = {"public_supply_公共用水": 430, "industrial_工業用水": 326,
            "domestic_and_public_家用及公共給水": 91, "hp_ge50": 9, "hp_gt10": 1341,
            "hp_max": 100.0}
    return {"census_L1_v3_yunlin": got, "plan": plan,
            "equal": {k: got[k] == plan[k] for k in plan}}


def pump_energy(years=K.CENSUS_ENERGY_YEARS) -> pd.DataFrame:
    """Mean annual kWh (model units) per kept in-grid pump over ``years`` (flags 0-5)."""
    p = guard.open_product(K.C0_PRE, "prior")
    frames = []
    for county in K.COUNTIES:
        f = "pumps_chiayi_ingrid.npz" if county == "chiayi" else f"pumps_{county}.npz"
        z = np.load(p.path(f), allow_pickle=False)
        dates = np.asarray(z["dates"]).astype(str)
        yr = np.array([int(d[:4]) for d in dates])
        sel = (yr >= years[0]) & (yr <= years[1])
        v = np.where(z["flag"][:, sel] <= 5, z["value"][:, sel], 0.0)
        v = np.nan_to_num(v, nan=0.0)
        frames.append(pd.DataFrame({"uid": np.asarray(z["uid"]).astype(str),
                                    "kwh_per_year": v.sum(1) / (years[1] - years[0] + 1),
                                    "cell_npz": z["cell"]}))
    return pd.concat(frames, ignore_index=True)


def energy_by_purpose(c: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    """GWh/yr (model units) of kept pumps by county and purpose: whole county and in grid."""
    m = c.merge(e, on="uid", how="inner", validate="one_to_one")
    m["kwh_in_grid"] = np.where(m.cell_npz >= 0, m.kwh_per_year, 0.0)
    g = m.groupby(["county", "purpose_raw", "purpose_class"], dropna=False)
    out = g.agg(n_pumps=("uid", "size"), n_pumps_in_grid=("cell_npz", lambda s: int(
        (s >= 0).sum())), gwh_per_year=("kwh_per_year", lambda s: float(s.sum()) / 1e6),
        gwh_per_year_in_grid=("kwh_in_grid", lambda s: float(s.sum()) / 1e6))
    out = out.reset_index()
    tot = out.groupby("county").gwh_per_year_in_grid.transform("sum")
    out["share_of_county_in_grid"] = out.gwh_per_year_in_grid / tot
    out["purpose_raw"] = out.purpose_raw.fillna("(null)")
    return out


def drought_matches(dw: pd.DataFrame, c: pd.DataFrame, e: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame]:
    """Nearest census pump (any purpose; public-supply purpose) of each drought well."""
    cc = c[np.isfinite(c.x_twd97) & np.isfinite(c.y_twd97)].reset_index(drop=True)
    cc = cc.merge(e[["uid", "kwh_per_year"]], on="uid", how="left")
    d_any, k_any = L.nearest_distance(dw.x_twd97, dw.y_twd97, cc.x_twd97, cc.y_twd97)
    pub = cc[cc.is_public].reset_index(drop=True)
    d_pub, _ = L.nearest_distance(dw.x_twd97, dw.y_twd97, pub.x_twd97, pub.y_twd97)
    rows = dw[["county", "table_no", "agency", "agency_key", "township", "in_grid"]].copy()
    rows["d_any_m"] = d_any
    rows["d_public_m"] = d_pub
    near = cc.iloc[k_any]
    rows["nearest_purpose"] = near.purpose_raw.fillna("(null)").to_numpy()
    rows["nearest_hp"] = near.pump_hp.to_numpy()
    rows["nearest_kept"] = near.kept.to_numpy()
    rows["nearest_kwh_per_year"] = near.kwh_per_year.to_numpy()
    s = []
    for (cty, ag), g in rows.groupby(["county", "agency"]):
        rec = {"county": cty, "agency": ag, "n": len(g)}
        for r in K.CENSUS_MATCH_M:
            w = g.d_any_m <= r
            rec[f"share_any_le{int(r)}m"] = float(w.mean())
            rec[f"share_public_le{int(r)}m"] = float((g.d_public_m <= r).mean())
            rec[f"gwh_nearest_le{int(r)}m"] = float(
                np.nansum(g.nearest_kwh_per_year.where(w, 0.0)) / 1e6)
        rec["median_d_any_m"] = float(g.d_any_m.median())
        rec["nearest_purposes_le50m"] = g[g.d_any_m <= 50].nearest_purpose.value_counts(
        ).to_dict()
        rec["nearest_hp_le50m_median"] = float(g[g.d_any_m <= 50].nearest_hp.median()) if (
            g.d_any_m <= 50).any() else None
        s.append(rec)
    return rows, pd.DataFrame(s)


def run() -> dict:
    dw, dw_checks = wells_located()
    c = census_table()
    e = pump_energy()
    pc = purpose_counts(c)
    eb = energy_by_purpose(c, e)
    rows, summ = drought_matches(dw, c, e)
    # census cell column vs this grid (consistency of the cell convention)
    ok = np.isfinite(c.x_twd97) & np.isfinite(c.y_twd97)
    mine = L.cell_index(c.x_twd97[ok], c.y_twd97[ok])
    cell_agree = float(np.mean(mine == c.cell[ok].fillna(-1).astype(int).to_numpy()))
    return {"drought_wells": dw, "drought_checks": dw_checks, "census": c, "energy": e,
            "purpose_counts": pc, "energy_by_purpose": eb, "matches": rows,
            "match_summary": summ, "plan_counts": plan_counts_check(c),
            "cell_convention_agreement": cell_agree}
