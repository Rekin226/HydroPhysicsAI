"""County series of Taiwan Water (台水) groundwater abstraction, Changhua and Yunlin.

Changhua (第十一區管理處 = Changhua county). Every 11th-district system draws on wells only
in 2007-2021 (表1); 彰化供水系統 adds 人工湖 (the 鳥嘴潭 lake, supplied from 2023 as purchased
raw water from the WRA central branch, 表4). Groundwater = district plant output (表2,
monthly) minus purchased raw water (表4, annual, spread over the year's days).

Yunlin (served by 第五區管理處 with Chiayi). Its systems mix wells, the 集集 weir and (from
2016) the 湖山 reservoir, and no table splits their output by source. Two estimates:

* E2 (primary in normal years): the WRA briefings' measured Jan-May pumping of the Yunlin
  Taiwan Water public wells (two-model chart reads, printed values where printed) times the
  same year's annual/Jan-May ratio of the Changhua Taiwan Water output (a seasonality
  transfer, stated assumption).
* E1 (validator; primary in drought years 2021 and 2023, whose pumping is concentrated in
  Jan-May): Yunlin systems' output minus the raw water bought from the WRA central branch
  (集集, 湖山). Biased up by the systems' own surface intakes (三疊溪) and down by treatment
  losses.

A ``vintage`` restricts inputs by publication date: 'strict' uses only documents published by
2022-12 (yearbooks 96-110, the 110 briefing) and holds 2022 at 2020, the last year that was
not a drought year (flag 6; 2021 was a drought year with extra pumping); 'exempt' adds
the 2022 values of yearbook 111 (published 2023-04) and the 112 briefing (created 2024-04);
'full' uses everything (to 2025).
"""
from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import drought_wells as DWL  # noqa: E402
from x2_public_supply import x2const as K  # noqa: E402

D11, D5 = "第十一區管理處", "第五區管理處"
#: yearbook (ROC) published by 2022-12 -> data years to 2021
STRICT_MAX_YEARBOOK = 110
PUB = {"yearbook_111": "2023-04", "briefing_112": "2024-04"}
#: strict pre-origin: 2022 is held at the last year that was not a drought year
HOLD_YEAR = 2020


def load_tables() -> dict:
    d = json.loads((K.EVID / "twc_tables.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in d["tables"].items()}


def load_janmay() -> pd.DataFrame:
    return pd.read_csv(K.EVID / "wra_janmay_controlled_wells.csv")


def days(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def ydays(y: int) -> int:
    return 366 if calendar.isleap(y) else 365


def _yb(year: int) -> int:
    return year - 1911


def district_annual(tab: dict, name: str, max_yb: int) -> dict[int, tuple[float, str]]:
    """Annual output (m3) of a district from 表2(2) of the latest allowed yearbook (every
    printing agrees: twc_tables cross_yearbook_same_value)."""
    yb = max(y for y in tab if y <= max_yb)
    vals = tab[yb]["by_district"]["values"]
    pages = tab[yb]["by_district"]["column_pages"]
    return {int(r) + 1911: (v[name], f"twc_yearbook_{yb}.pdf 表2(2) p.{pages.get(name)}")
            for r, v in vals.items() if name in v and v[name] is not None and int(r) >= 96}


def district_monthly(tab: dict, name: str, year: int) -> tuple[list[float], str] | None:
    t = tab.get(_yb(year))
    if not t:
        return None
    rec = t["monthly"]["district"].get(name)
    if not rec:
        return None
    return rec["months"], f"twc_yearbook_{_yb(year)}.pdf 表2(1) pp.{rec['pages']}"


def purchases(tab: dict, name: str, year: int) -> tuple[float | None, dict, str]:
    t = tab.get(_yb(year))
    r = (t or {}).get("purchases", {}).get("districts", {}).get(name)
    if not r or "raw_clear" not in r:
        return None, {}, ""
    pages = [r.get("page1"), r.get("page2")]
    return r["raw_clear"] * 1e3, r, f"twc_yearbook_{_yb(year)}.pdf 表4 pp.{pages}"


def systems(tab: dict, name: str, year: int) -> dict[str, float]:
    t = tab.get(_yb(year))
    if not t:
        return {}
    d = t["systems"]["districts"].get(name, {})
    return {k: v["output"] for k, v in d.get("systems", {}).items()}


def changhua(tab: dict, vintage: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Annual, monthly and per-system tables for Changhua (m3)."""
    last = 2025 if vintage == "full" else 2022
    max_yb = 114 if vintage == "full" else (111 if vintage == "exempt" else STRICT_MAX_YEARBOOK)
    ann = district_annual(tab, D11, max_yb)
    shapes = []
    for y in range(2012, 2017):
        mo = district_monthly(tab, D11, y)
        shapes.append(np.array(mo[0]) / sum(mo[0]))
    mean_shape = np.mean(shapes, axis=0)
    rows_a, rows_m, rows_s = [], [], []
    for y in range(2007, last + 1):
        held = vintage == "strict" and y == 2022
        ys = HOLD_YEAR if held else y
        out, src = ann[ys]
        mo = district_monthly(tab, D11, ys)
        if mo is not None:
            months, msrc = np.array(mo[0], float), mo[1]
            mflag = None
        else:
            months, msrc, mflag = out * mean_shape, f"{src} x mean 2012-2016 shape", 4
        if abs(months.sum() - out) > 1.0:
            raise SystemExit(f"Changhua {ys}: months {months.sum()} != annual {out}")
        pur, prec, psrc = purchases(tab, D11, ys)
        sd2 = (K.SD_PLANT_FRAC * out / 1e6) ** 2
        if pur is None:
            pur, sd2 = 0.0, sd2 + K.SD_UNKNOWN_PURCHASE ** 2
            psrc = "not parsed (before 2012): assumed 0"
        if ys >= 2021:
            sd2 += K.SD_RIVERBED_2021 ** 2
        g = out - pur
        flag = 6 if held else (1 if pur > 0 else 0)
        method = ("held at 2020 (strict pre-origin: no document published by 2022-12 gives "
                  "2022; 2021 was a drought year)" if held else
                  "district output (groundwater-only systems)" + (
                      " minus purchased raw water" if pur > 0 else ""))
        rows_a.append({"county": "changhua", "year": y, "value_m3": g, "value_Mm3": g / 1e6,
                       "sd_Mm3": float(np.sqrt(sd2)), "flag": flag, "method": method,
                       "district_output_m3": out, "purchased_raw_m3": pur,
                       "purchased_from": json.dumps({k: prec.get(k) for k in (
                           "irrigation_raw", "central_raw", "south_raw", "north_raw",
                           "other_raw")}) if prec else "", "source": src,
                       "source_monthly": msrc, "source_purchases": psrc,
                       "riverbed_intake_listed": ys >= 2021})
        for m in range(1, 13):
            v = months[m - 1] - pur * days(ys, m) / ydays(ys)
            rows_m.append({"county": "changhua", "month": f"{y}-{m:02d}", "value_m3": v,
                           "rate_m3d": v / days(y, m),
                           "sd_m3": float(np.sqrt(sd2)) * 1e6 * days(ys, m) / ydays(ys),
                           "flag": 6 if held else (mflag if mflag is not None else flag),
                           "method": method if mflag is None else method + "; months from "
                           "the mean 2012-2016 shape", "source": msrc})
        sy = systems(tab, D11, ys)
        tot = sum(sy.values())
        for k, v in sy.items():
            if prec.get("central_raw"):        # the lake water (人工湖) enters 彰化供水系統
                share_pur = pur if k == "彰化供水系統" else 0.0
            else:                               # small irrigation raw water: pro rata
                share_pur = pur * v / tot if tot else 0.0
            rows_s.append({"county": "changhua", "year": y, "system": k, "output_m3": v,
                           "purchased_assigned_m3": share_pur,
                           "groundwater_m3": v - share_pur, "held": held,
                           "source": f"twc_yearbook_{_yb(ys)}.pdf 表3"})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_m), pd.DataFrame(rows_s)


def yunlin(tab: dict, jm: pd.DataFrame, ch_month: pd.DataFrame, vintage: str
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    last = 2025 if vintage == "full" else 2022
    tw = jm[(jm.county == "yunlin") & (jm.agency == "taiwan_water")]
    pre = tw[tw.vintage == "b110"].set_index("year")
    new = tw[tw.vintage == "b112"].set_index("year")
    # Changhua district output shape (not net of purchases) for the seasonality transfer
    rows_a, rows_m = [], []
    shape_cache = {}

    def shape(y: int) -> np.ndarray:
        if y not in shape_cache:
            mo = district_monthly(tab, D11, y)
            if mo is None:
                s = np.mean([np.array(district_monthly(tab, D11, z)[0]) /
                             sum(district_monthly(tab, D11, z)[0]) for z in range(2012, 2017)],
                            axis=0)
            else:
                s = np.array(mo[0]) / sum(mo[0])
            shape_cache[y] = s
        return shape_cache[y]

    def janmay(y: int) -> tuple[float | None, float, str]:
        if y <= 2021 and y in pre.index:
            r = pre.loc[y]
            return float(r.value_1e4m3) * 1e4, float(r.sd_1e4m3) * 1e4, (
                f"wra_cy_subsidence_results_110.pdf p.{int(r.source_page)} 雲林台水1~5月 "
                + ("(printed)" if pd.notna(r.printed_value) else "(two-model chart read)"))
        if vintage != "strict" and y in new.index:
            r = new.loc[y]
            return float(r.value_1e4m3) * 1e4, float(r.sd_1e4m3) * 1e4, (
                f"wra_cy_subsidence_results_112.pdf p.{int(r.source_page)} 雲林台水1~5月 "
                "(two-model chart read)")
        return None, 0.0, ""

    def e1(y: int) -> tuple[float | None, str]:
        if vintage == "strict" and _yb(y) > STRICT_MAX_YEARBOOK:
            return None, ""
        if vintage == "exempt" and _yb(y) > 111:
            return None, ""
        sy = systems(tab, D5, y)
        out = sum(v for k, v in sy.items() if k in K.YUNLIN_SYSTEMS)
        _, prec, psrc = purchases(tab, D5, y)
        if not out or prec.get("central_raw") is None:
            return None, ""
        return out - prec["central_raw"] * 1e3, (f"twc_yearbook_{_yb(y)}.pdf 表3 (Yunlin "
                                                  f"systems) minus {psrc} central-branch raw")

    dw = {}
    for r in DWL.parse()["values"]:
        if r["county"] == "yunlin" and (vintage != "strict" or r["published"] <= "2022-12"):
            dw[r["year"]] = (r["volume_1e4m3"] * 1e4, f"{r['source']} ({r['period']})")
    for y in range(2007, last + 1):
        held = vintage == "strict" and y == 2022
        ys = HOLD_YEAR if held else max(y, 2011)
        J, Jsd, jsrc = janmay(ys)
        E1, e1src = e1(ys)
        mo = district_monthly(tab, D11, ys)
        if mo is not None:
            r_ch = sum(mo[0]) / sum(mo[0][:5])
        else:
            r_ch = float(np.mean([sum(district_monthly(tab, D11, z)[0]) /
                                  sum(district_monthly(tab, D11, z)[0][:5])
                                  for z in range(2012, 2017)]))
        E2 = J * r_ch if J is not None else None
        drought = ys in K.DROUGHT_YEARS
        if J is None:
            V = max(E1, 0.0) if E1 is not None else None
            flag, method = 5, "E1 residual (systems output minus purchased raw water), >= 0"
            sd = max(0.1 * V, 3.0e6) if V is not None else None
        elif drought and E1 is not None:
            V = max(E1, J)
            flag, method = 5, "drought year: max(E1 residual, measured Jan-May)"
            sd = max(abs(E1 - E2), 0.1 * V, 1.0e6)
        else:
            V = E2
            flag, method = 5, "E2: measured Jan-May x Changhua Taiwan Water annual/Jan-May"
            sd = max(abs(E1 - E2), 0.1 * V, 1.0e6) if E1 is not None else max(0.15 * V, 1e6)
        dwv, dws = dw.get(ys, (None, ""))
        if dwv is not None and V is not None and dwv > V:
            V, method = dwv, method + "; raised to the drought-well survey volume (lower bound)"
            sd = max(sd, 0.25 * V)
        if held:
            flag, method = 6, "held at 2020 (strict pre-origin) - " + method
        elif y < 2011:
            flag, method = 6, "held at 2011 (no Yunlin Taiwan Water series before 2011)"
            sd = max(sd, 0.25 * V)
        rows_a.append({"county": "yunlin", "year": y, "value_m3": V, "value_Mm3": V / 1e6,
                       "sd_Mm3": sd / 1e6, "flag": flag, "method": method,
                       "janmay_measured_m3": J, "janmay_sd_m3": Jsd, "seasonal_ratio_ch": r_ch,
                       "E2_m3": E2, "E1_m3": E1, "drought_year": drought,
                       "drought_wells_m3": dwv, "source_drought_wells": dws,
                       "source": jsrc, "source_E1": e1src})
        s = shape(ys)
        for m in range(1, 13):
            if drought and J is not None and ys >= 2011:
                v = (J * s[m - 1] / s[:5].sum() if m <= 5 else
                     (V - J) * s[m - 1] / s[5:].sum())
            else:
                v = V * s[m - 1]
            mflag = flag if (flag == 6 or J is None) else (3 if m <= 5 else 5)
            rows_m.append({"county": "yunlin", "month": f"{y}-{m:02d}", "value_m3": v,
                           "rate_m3d": v / days(y, m), "sd_m3": sd * s[m - 1],
                           "flag": mflag, "method": method + "; months by the Changhua Taiwan "
                           "Water monthly shape of the year", "source": jsrc})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_m)


def build(vintage: str) -> dict:
    tab = load_tables()
    jm = load_janmay()
    ca, cm, cs = changhua(tab, vintage)
    ya, ym = yunlin(tab, jm, cm, vintage)
    return {"annual": pd.concat([ca, ya], ignore_index=True),
            "monthly": pd.concat([cm, ym], ignore_index=True), "systems": cs, "tables": tab}
