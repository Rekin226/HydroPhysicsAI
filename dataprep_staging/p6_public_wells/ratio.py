"""Pre-origin recomputation of the WRA-to-model volume ratio (check K1, reported only).

WRA side: the 110 briefing's chart 灌溉抽水量比較 (p. 18), series 雲林合計 (Yunlin county
Jan-May irrigation pumping, 萬噸; WRA's estimate from registered-well electricity x a
water/electricity ratio), two reads (``transcribe_p6.py``). The axis a read assigns the series
to is checked against the axis titles: the county total belongs to the axis titled
全縣抽水量. A read that put it on the other axis is rescaled by the ratio of the two axis
ranges (the plotted position is kept; nothing is estimated). The WRA value of a year is the
mean of the two reads.

Model side: engine/v2 ``Q_rate`` (m3/day, the P4.0 replica of the deliverable's census
abstraction with lift feedback; step t = month ``dates[t+1]``), summed over the cells whose
``cell_county`` is 雲林縣, times the days of the month, times the Yunlin irrigation share of
census energy in that month (C0 preorigin/v3 ``value_by_county``). Primary ratio: mean WRA /
mean model over 2013-2021 (complete Jan-May on both sides).
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402
from p6_public_wells import transcribe_p6 as TR  # noqa: E402

SERIES = "雲林合計"
COUNTY_AXIS = "全縣"
YEARS_PRIMARY = list(range(2013, 2022))
YEARS_PILOT = list(range(2012, 2022))


def wra_series() -> dict:
    c = TR.load_cached("b110_irr_yl")
    reads = {}
    for name in ("pdf_read", "img_read"):
        ch = c[name]["result"]["charts"]
        if len(ch) != 1:
            raise ValueError(f"{name}: expected one chart, got {len(ch)}")
        ch = ch[0]
        axes = {"left": (ch["left_axis_title"], ch["left_axis_min"], ch["left_axis_max"]),
                "right": (ch["right_axis_title"], ch["right_axis_min"], ch["right_axis_max"])}
        county_side = [k for k, v in axes.items() if COUNTY_AXIS in v[0]]
        if len(county_side) != 1:
            raise ValueError(f"{name}: cannot tell the county axis from {axes}")
        cs = county_side[0]
        s = [x for x in ch["series"] if SERIES in x["legend_label"]]
        if len(s) != 1:
            raise ValueError(f"{name}: series {SERIES} not found once")
        s = s[0]
        lo_c, hi_c = axes[cs][1], axes[cs][2]
        lo_s, hi_s = axes[s["axis"]][1], axes[s["axis"]][2]
        vals = {}
        for p in s["points"]:
            v = p["value"]
            if v is None:
                continue
            frac = (v - lo_s) / (hi_s - lo_s)
            vals[int(p["x_label"]) + 1911] = lo_c + frac * (hi_c - lo_c)
        reads[name] = {"model": c[name]["model"], "stated_axis": s["axis"],
                       "county_axis": cs, "rescaled": s["axis"] != cs,
                       "axis_range": [lo_c, hi_c], "values_1e4m3": vals}
    years = sorted(set(reads["pdf_read"]["values_1e4m3"]) & set(reads["img_read"]["values_1e4m3"]))
    a = np.array([reads["pdf_read"]["values_1e4m3"][y] for y in years])
    b = np.array([reads["img_read"]["values_1e4m3"][y] for y in years])
    rng = reads["pdf_read"]["axis_range"][1] - reads["pdf_read"]["axis_range"][0]
    return {"reads": reads, "years": years, "mean_1e4m3": dict(zip(years, (a + b) / 2,
                                                                     strict=True)),
            "max_abs_diff_frac_axis": float(np.max(np.abs(a - b)) / rng)}


def model_janmay() -> dict:
    eng = guard.open_product(K.ENGINE, "model_design")
    z = np.load(eng.path("engine_arrays.npz"), allow_pickle=False)
    q = np.asarray(z["Q_rate"], float)                      # (A, 131) m3/day
    dates = np.asarray(z["dates"]).astype(str)[1:]          # month of each step
    yl = np.asarray(z["cell_county"]).astype(str) == K.COUNTY_ZH["yunlin"]
    c0 = guard.open_product(K.C0_PRE, "prior")
    cb = np.load(c0.path("cells_by_class.npz"), allow_pickle=False)
    counties = list(np.asarray(cb["counties"]).astype(str))
    classes = list(np.asarray(cb["classes"]).astype(str))
    cdates = list(np.asarray(cb["dates"]).astype(str))
    v = np.asarray(cb["value_by_county"], float)[counties.index("yunlin")]   # (C, A, T)
    f = np.asarray(cb["flag_by_county"])[counties.index("yunlin")]
    v = np.where(f <= 5, np.nan_to_num(v, nan=0.0), 0.0)
    irr = v[classes.index("irrigation")].sum(0)            # (T,) kWh in grid
    tot = v.sum((0, 1))
    out = {}
    for t, d in enumerate(dates):
        y, m = int(d[:4]), int(d[5:7])
        if m > 5:
            continue
        k = cdates.index(d[:7])
        share = irr[k] / tot[k]
        days = calendar.monthrange(y, m)[1]
        r = out.setdefault(y, {"model_m3": 0.0, "months": 0, "census_irr_kwh": 0.0})
        r["model_m3"] += float(q[yl, t].sum() * days * share)
        r["months"] += 1
        r["census_irr_kwh"] += float(irr[k])
    for y in (2012,):                                       # January 2012 is not a step
        if y in out:
            out[y]["census_irr_kwh_janmay_full"] = float(sum(
                irr[cdates.index(f"{y}-{m:02d}")] for m in range(1, 6)))
    return out


def run() -> dict:
    w = wra_series()
    m = model_janmay()

    def ratio(years):
        wv = np.array([w["mean_1e4m3"][y] * 1e4 for y in years])
        mv = np.array([m[y]["model_m3"] for y in years])
        return {"years": [years[0], years[-1]], "wra_mean_Mm3": float(wv.mean() / 1e6),
                "model_mean_Mm3": float(mv.mean() / 1e6), "ratio": float(wv.mean() / mv.mean()),
                "per_year": {int(y): float(a / b) for y, a, b in zip(years, wv, mv, strict=True)},
                "per_year_sd": float(np.std(wv / mv, ddof=1))}
    prim = ratio(YEARS_PRIMARY)
    pilot = ratio(YEARS_PILOT)
    wv = np.array([w["mean_1e4m3"][y] * 1e4 for y in YEARS_PRIMARY])
    kv = np.array([m[y]["census_irr_kwh"] for y in YEARS_PRIMARY])
    return {"wra": w, "model": {int(k): v for k, v in m.items()}, "primary": prim,
            "pilot_definition": pilot,
            "wra_m3_per_census_kwh": float(wv.mean() / kv.mean()),
            "plan_value": K.WRA_TO_MODEL,
            "pilot_value": 2.575886505123094,
            "note": "reported only; the member keeps the plan's 2.58 (P6.inventory_member.v1)"}
