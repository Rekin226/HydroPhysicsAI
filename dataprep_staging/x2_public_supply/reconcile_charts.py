"""Reconcile the two chart reads of transcribe_x2.py and cross-check them against N1's reads.

For each county chart (彰化公井抽水量, 雲林公井抽水量) and series (台水 / 農田水利署 1~5月),
the two reads must agree within ``TOL_FRAC`` of the axis range at every year; the value kept
is their mean, with sd = max(half the difference, ``SD_FRAC`` of the axis range). A third,
independent check is N1's reading of the same series in the 110 briefing (2011-2021), copied
in P6's after-the-fact evidence file (read here as a plain JSON file, validator only).

Output: ``reports/X2_public_supply/evidence/wra_janmay_controlled_wells.csv`` and
``.../chart_reconciliation.json``. Values are 10^4 m3 (萬噸) of Jan-May pumping.

Usage: python -I reconcile_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import x2const as K  # noqa: E402

TOL_FRAC = 0.015
SD_FRAC = 0.005
CHARTS = {"b112_p23": ("changhua", "彰化公井抽水量"), "b112_p26": ("yunlin", "雲林公井抽水量"),
          "b110_p10": ("changhua", "彰化公井抽水量"), "b110_p19": ("yunlin", "雲林公井抽水量")}
#: values printed as text on the 110 briefing pages (台水公司抽水量 109年..., 110年...), 萬噸
PRINTED = {("b110_p10", "taiwan_water", 2020): 2665.0, ("b110_p10", "taiwan_water", 2021): 2721.0,
           ("b110_p10", "irrigation_agency", 2020): 283.0,
           ("b110_p10", "irrigation_agency", 2021): 344.0,
           ("b110_p19", "taiwan_water", 2020): 49.0, ("b110_p19", "taiwan_water", 2021): 1193.0,
           ("b110_p19", "irrigation_agency", 2020): 2854.0,
           ("b110_p19", "irrigation_agency", 2021): 3193.0}
PRINTED_TEXT = {"b110_p10": "台水公司抽水量 109年2,665萬噸、110年2,721萬噸。 / 農水署抽水量 109年"
                            "283萬噸、110年344萬噸。",
                "b110_p19": "台水公司抽水量 109年49萬噸、110年1,193萬噸。 / 農水署抽水量 109年"
                            "2,854萬噸、110年3,193萬噸"}
SERIES = {"台水": "taiwan_water", "農田水利署": "irrigation_agency"}
N1_FILE = K.REPORT_DIR.parent / "P6_evidence" / "after_the_fact_n1_janmay.json"


def _chart(read: dict, title: str) -> dict:
    hits = [c for c in read["result"]["charts"] if c["title"].strip() == title]
    if len(hits) != 1:
        raise SystemExit(f"chart {title!r}: {len(hits)} matches")
    return hits[0]


def main() -> None:
    rows, issues = [], []
    for key, (county, title) in CHARTS.items():
        cache = json.loads((K.API_CACHE / f"{key}.json").read_text(encoding="utf-8"))
        a, b = _chart(cache["pdf_read"], title), _chart(cache["img_read"], title)
        rng = float(a["left_axis_max"]) - float(a["left_axis_min"])
        if (a["left_axis_min"], a["left_axis_max"]) != (b["left_axis_min"], b["left_axis_max"]):
            issues.append(f"{key}: axis ranges differ")
        for sa in a["series"]:
            agency = next((v for k, v in SERIES.items() if k in sa["legend_label"]), None)
            sb = next((s for s in b["series"] if s["legend_label"] == sa["legend_label"]), None)
            if agency is None or sb is None:
                issues.append(f"{key}: series {sa['legend_label']!r} unmatched")
                continue
            pb = {p["x_label"]: p["value"] for p in sb["points"]}
            for p in sa["points"]:
                va, vb = p["value"], pb.get(p["x_label"])
                if va is None or vb is None:
                    issues.append(f"{key} {sa['legend_label']} {p['x_label']}: missing read")
                    continue
                d = abs(va - vb)
                roc = int(str(p["x_label"]).strip())
                printed = PRINTED.get((key, agency, roc + 1911))
                txt = " ".join(cache["pdf_read"]["result"]["printed_sentences"] +
                               cache["img_read"]["result"]["printed_sentences"])
                if printed is not None and f"{int(printed):,}萬噸" not in txt:
                    issues.append(f"{key}: printed value {printed} not found in both reads")
                rows.append({"county": county, "agency": agency, "year": roc + 1911,
                             "vintage": key.split("_")[0], "roc_year": roc,
                             "value_1e4m3": printed if printed is not None else (va + vb) / 2,
                             "printed_value": printed,
                             "sd_1e4m3": 0.5 if printed is not None else max(d / 2,
                                                                              SD_FRAC * rng),
                             "pdf_read": va,
                             "img_read": vb, "abs_diff": d, "axis_range": rng,
                             "agree": d <= TOL_FRAC * rng, "source_file": cache["pdf_read"][
                                 "source"]["file"], "source_page": cache["pdf_read"]["source"][
                                 "page"], "chart_title": title,
                             "series_label": sa["legend_label"]})
    df = pd.DataFrame(rows).sort_values(["county", "agency", "vintage", "year"])
    # cross-vintage check: the 110 and 112 briefings print the same 2011-2021 series
    piv = df.pivot_table(index=["county", "agency", "year"], columns="vintage",
                         values="value_1e4m3")
    both = piv.dropna()
    df["cross_vintage_diff"] = [
        (piv.loc[(r.county, r.agency, r.year), "b112"] - piv.loc[(r.county, r.agency, r.year),
                                                                 "b110"])
        if (r.county, r.agency, r.year) in both.index else float("nan")
        for r in df.itertuples()]
    # third check: N1's reads of the 110 briefing (2011-2021)
    n1 = json.loads(N1_FILE.read_text(encoding="utf-8")) if N1_FILE.exists() else {}
    key_map = {("changhua", "taiwan_water"): "tw_changhua", ("yunlin", "taiwan_water"):
               "tw_yunlin", ("changhua", "irrigation_agency"): "ia_changhua",
               ("yunlin", "irrigation_agency"): "ia_yunlin"}
    n1v = []
    for r in df.itertuples():
        s = n1.get(key_map[(r.county, r.agency)], {}).get("janmay_1e4m3", {})
        n1v.append(s.get(str(r.year)))
    df["n1_b110_read"] = n1v
    df["n1_diff"] = df.value_1e4m3 - df.n1_b110_read
    df["n1_agree"] = df.n1_diff.abs() <= TOL_FRAC * df.axis_range
    out = K.EVID / "wra_janmay_controlled_wells.csv"
    df.to_csv(out, index=False)
    with_n1 = df[df.n1_b110_read.notna()]
    rep = {"n_points": len(df), "n_disagree": int((~df.agree).sum()),
           "tolerance": f"{TOL_FRAC:.1%} of the axis range",
           "max_abs_diff_by_chart": df.groupby("county").abs_diff.max().to_dict(),
           "cross_vintage_points": int(len(both) * 1),
           "cross_vintage_max_abs_diff": float((both["b112"] - both["b110"]).abs().max()),
           "cross_vintage_disagree": int(((both["b112"] - both["b110"]).abs() >
                                          TOL_FRAC * 5000).sum()),
           "printed_text": PRINTED_TEXT,
           "n1_points": len(with_n1), "n1_disagree": int((~with_n1.n1_agree).sum()),
           "n1_max_abs_diff": float(with_n1.n1_diff.abs().max()) if len(with_n1) else None,
           "issues": issues}
    (K.EVID / "chart_reconciliation.json").write_text(json.dumps(rep, indent=1,
                                                                 ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print(df[df.agency == "taiwan_water"][["county", "vintage", "year", "value_1e4m3",
                                           "sd_1e4m3", "n1_b110_read",
                                           "cross_vintage_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
