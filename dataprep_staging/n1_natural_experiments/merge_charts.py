"""Merge the two independent chart reads into the public-well Jan-May pumping table.

For every (page, chart, series, ROC year) the Opus PDF read and the Sonnet image read are
compared. A point is *two-read agreed* when the reads differ by at most ``AGREE_FRAC`` of the
chart's y-axis range; its value is the mean of the reads and its digitization sd is
``max(|a - b| / 2, SD_FLOOR_FRAC * range)`` (uncalibrated). A point the reads disagree on keeps
the mean but gets flag 9 (suspect). Where the 110 briefing prints the value (Jan-May 2020 and
2021, county totals by agency), the printed number replaces the chart value (method
``printed_text``, sd 0.5 in the last printed digit).

Checks reported (validation): agreement share; the Opus PDF read against the printed anchors
(the image read is not an independent check there: it may have copied the printed sentence);
and the two printings (110 vs 112 briefing) of the county series over 2011-2021.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import parse_sources as P  # noqa: E402

AGREE_FRAC = 0.02
SD_FLOOR_FRAC = 0.005
TW = "Taiwan Water Corporation (台水)"
IA = "Irrigation Agency (農水署)"
TOWNS = ("溪湖", "溪州", "二林", "土庫", "元長", "虎尾", "大埤")


def _sq(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _agency(title: str, label: str) -> str | None:
    t = _sq(title) + "|" + _sq(label)
    lab = _sq(label)
    if "台水" in lab or "自來水" in lab:
        return TW
    if "農田水利署" in lab or "農水署" in lab:
        return IA
    if "台水" in _sq(title):
        return TW
    if "農田水利署" in _sq(title) or "農水署" in _sq(title):
        return IA
    return None if not t else None


def _township(label: str) -> str | None:
    lab = _sq(label)
    for t in TOWNS:
        if lab.startswith(t) or lab == t:
            return t
    return None


def _year(x_label: str) -> int | None:
    m = re.search(r"(\d{2,3})", x_label or "")
    return int(m.group(1)) + 1911 if m else None


def _charts(read: dict) -> dict:
    return {_sq(c["title"]): c for c in read["result"]["charts"]}


def merge() -> tuple[pd.DataFrame, dict]:
    rows, checks = [], {"pages": {}}
    anchors = P.public_well_anchors()
    for spec in K.CHART_PAGES:
        cache = json.loads((K.API_CACHE / f"{spec['key']}.json").read_text(encoding="utf-8"))
        a, b = _charts(cache["pdf_read"]), _charts(cache["img_read"])
        titles_a, titles_b = list(a), list(b)
        pairs = [(t, t) for t in titles_a if t in b]
        if len(pairs) < min(len(a), len(b)):           # fall back to the order on the page
            pairs = list(zip(titles_a, titles_b, strict=False))
        checks["pages"][spec["key"]] = {"charts_pdf_read": titles_a, "charts_img_read": titles_b,
                                        "paired": pairs}
        for ta, tb in pairs:
            ca, cb = a[ta], b[tb]
            rng = float(ca["y_axis_max"]) - float(ca["y_axis_min"])
            sb = {_sq(s["legend_label"]): s for s in cb["series"]}
            for sa in ca["series"]:
                key = _sq(sa["legend_label"])
                s2 = sb.get(key)
                if s2 is None:
                    continue
                pb = {_year(p["x_label"]): p["value"] for p in s2["points"]}
                for p in sa["points"]:
                    yr = _year(p["x_label"])
                    va, vb = p["value"], pb.get(yr)
                    if yr is None or va is None or vb is None:
                        continue
                    d = abs(float(va) - float(vb))
                    agree = d <= AGREE_FRAC * rng
                    rows.append({
                        "page_key": spec["key"], "source_file": K.SRC[spec["src"]].name,
                        "source_page": spec["page"], "county": spec["county"],
                        "chart_title": ca["title"], "series_label": sa["legend_label"],
                        "agency": _agency(ca["title"], sa["legend_label"]),
                        "township": _township(sa["legend_label"]), "year": yr,
                        "value_1e4m3": (float(va) + float(vb)) / 2.0,
                        "sd_1e4m3": max(d / 2.0, SD_FLOOR_FRAC * rng),
                        "pdf_read": float(va), "img_read": float(vb), "axis_range": rng,
                        "agree": bool(agree), "flag": 0 if agree else 9,
                        "method": "chart_read_two_models"})
    df = pd.DataFrame(rows)
    county = df.township.isna()
    # printed anchors (110 briefing) replace the chart value for the county series
    an_check = []
    for an in anchors:
        m = (county & (df.county == an["county"]) & (df.agency == an["agency"])
             & (df.year == an["year"]))
        for i in df.index[m]:
            an_check.append({"page_key": df.at[i, "page_key"], "county": an["county"],
                             "agency": an["agency"], "year": an["year"],
                             "printed": an["value_1e4m3"], "pdf_read": df.at[i, "pdf_read"],
                             "img_read": df.at[i, "img_read"],
                             "pdf_err_frac_axis": abs(df.at[i, "pdf_read"] - an["value_1e4m3"])
                             / df.at[i, "axis_range"]})
            df.at[i, "value_1e4m3"] = an["value_1e4m3"]
            df.at[i, "sd_1e4m3"] = 0.5
            df.at[i, "flag"] = 0
            df.at[i, "method"] = "printed_text"
    checks["anchors"] = an_check
    checks["anchor_pdf_err_frac_axis_max"] = float(max(x["pdf_err_frac_axis"] for x in an_check))
    checks["agreement"] = {"points": int(len(df)), "agreed": int(df.agree.sum()),
                           "share": float(df.agree.mean()),
                           "max_diff_frac_axis": float((abs(df.pdf_read - df.img_read)
                                                        / df.axis_range).max())}
    # two printings of the county series, 2011-2021
    c = df[county & df.year.le(2021)]
    piv = c.pivot_table(index=["county", "agency", "year"], columns="page_key",
                        values="value_1e4m3", aggfunc="first")
    cmp = []
    for (cty, ag, yr), r in piv.iterrows():
        k110 = "b110_p10" if cty == "Changhua" else "b110_p19"
        k112 = "b112_p23" if cty == "Changhua" else "b112_p26"
        if k110 in r and k112 in r and np.isfinite(r[k110]) and np.isfinite(r[k112]):
            cmp.append({"county": cty, "agency": ag, "year": int(yr), "b110": float(r[k110]),
                        "b112": float(r[k112]), "diff": float(r[k112] - r[k110])})
    cmp = pd.DataFrame(cmp)
    checks["two_printings"] = {
        "n": int(len(cmp)), "median_abs_diff_1e4m3": float(cmp["diff"].abs().median()),
        "max_abs_diff_1e4m3": float(cmp["diff"].abs().max()),
        "rows": cmp.to_dict(orient="records")}
    return df, checks


def janmay_series(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (county, agency, township, year): the printed anchors where they exist,
    else the earliest printing (the 110 briefing, survey year 2021, for 2011-2021), else the
    112 briefing (survey year 2023; the only printing of 2022, and of 2023, post-origin).
    ``vintage_survey_year`` is the report's survey year (published at the earliest at its
    end; the exact publication date is not printed) and ``month`` the data month (May)."""
    pri = {"printed_text": 0, "chart_read_two_models": 1}
    d = df.copy()
    d["prio_method"] = d["method"].map(pri)
    d["prio_page"] = d["page_key"].str.startswith("b110").map({True: 0, False: 1})
    d["township"] = d["township"].fillna("(county)")
    d = d.sort_values(["prio_method", "prio_page"])
    out = d.groupby(["county", "agency", "township", "year"], as_index=False).first()
    out["vintage_survey_year"] = out["page_key"].str[1:4].astype(int) + 1911
    out["month"] = out["year"].astype(str) + "-05"
    return out.drop(columns=["prio_method", "prio_page"])


if __name__ == "__main__":
    df, ch = merge()
    print(json.dumps({k: v for k, v in ch.items() if k != "two_printings"}, ensure_ascii=False,
                     indent=1, default=str)[:4000])
    print({k: v for k, v in ch["two_printings"].items() if k != "rows"})
