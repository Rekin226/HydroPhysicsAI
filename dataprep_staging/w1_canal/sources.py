"""Readers for the W1 aggregates, each re-validated against its L0 page.

* Jiji weir annual table (集集攔河堰歷年標的供水量表, 10^8 m3, 2006-2025): the repository CSV
  (chou-shui-data/data/water/jiji_weir_annual_supply.csv) is compared cell by cell with the
  L0 HTML page it was read from (data_fetch_api/new_measurements/jiji_weir_pages/annual.html).
* Jan-May supply per purpose 2015-2021 (WRA 110年度彰雲地區地層下陷檢測成果, slide 19, 萬噸):
  parsed from the L0 PDF with pdftotext (no hand transcription).
* MOA 691 (農田水利署各管理處歷年實際灌溉用水量, per crop season to ROC 104 = 2015): the
  crop-1 share of the Changhua and Yunlin offices' totals (whole office, all sources).
"""
from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from w1_canal.config import (
    BRIEFING_110_PAGE,
    BRIEFING_110_PDF,
    JIJI_CSV,
    JIJI_HTML,
    MOA_691_CSV,
    V3_NPZ,
)

JIJI_COLUMNS = ["inflow_1e8_m3", "yunlin_irrigation_1e8_m3", "industrial_1e8_m3",
                "public_1e8_m3", "changhua_irrigation_1e8_m3", "total_supply_1e8_m3"]


def jiji_annual() -> tuple[pd.DataFrame, dict]:
    """The annual table (index = year AD) and a cross-check against the L0 HTML."""
    csv = pd.read_csv(JIJI_CSV).set_index("year").sort_index()
    text = JIJI_HTML.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
    i = text.find("集集攔河堰歷年標的供水量表")
    seg = text[i:i + 4000]
    rows = re.findall(r"(\d{2,3}) ?年度 ((?:\d+(?:\.\d+)? ){5}\d+(?:\.\d+)?)", seg)
    h = {int(r) + 1911: [float(v) for v in vals.split()] for r, vals in rows}
    diffs = []
    for y, vals in h.items():
        if y not in csv.index:
            diffs.append({"year": y, "problem": "missing in CSV"})
            continue
        for c, v in zip(JIJI_COLUMNS, vals, strict=True):
            if abs(float(csv.loc[y, c]) - v) > 1e-9:
                diffs.append({"year": y, "column": c, "csv": float(csv.loc[y, c]), "html": v})
    miss = sorted(set(csv.index) - set(h))
    # internal consistency: total = sum of purposes
    tot = csv[["yunlin_irrigation_1e8_m3", "industrial_1e8_m3", "public_1e8_m3",
               "changhua_irrigation_1e8_m3"]].sum(axis=1)
    bad_tot = {int(y): round(float(t - csv.loc[y, "total_supply_1e8_m3"]), 3)
               for y, t in tot.items() if abs(t - csv.loc[y, "total_supply_1e8_m3"]) > 0.015}
    check = {"html_rows": len(h), "csv_rows": int(len(csv)), "cell_differences": diffs,
             "csv_years_not_in_html": [int(y) for y in miss],
             "total_minus_sum_of_purposes_gt_0.015": bad_tot,
             "passed": not diffs and not miss}
    return csv, check


def janmay_benchmark() -> tuple[pd.DataFrame, dict]:
    """Jan-May supply 2015-2021 per purpose (10^8 m3), parsed from the briefing PDF."""
    with tempfile.TemporaryDirectory(prefix="w1_b110_") as tmp:
        out = Path(tmp) / "p.txt"
        subprocess.run(["pdftotext", "-layout", "-f", str(BRIEFING_110_PAGE), "-l",
                        str(BRIEFING_110_PAGE + 1), str(BRIEFING_110_PDF), str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        txt = out.read_text(encoding="utf-8", errors="replace")
    if "1~5月供水量" not in txt or "彰化灌溉" not in txt:
        raise ValueError("Jan-May table header not found on the expected briefing page")
    rows = re.findall(r"^\s*(1[01]\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", txt,
                      flags=re.M)
    t = pd.DataFrame([(int(r) + 1911, *map(float, v)) for r, *v in rows],
                     columns=["year", "chc", "yl", "pub", "ind"]).set_index("year") / 1e4
    check = {"page": BRIEFING_110_PAGE, "rows": int(len(t)),
             "years": [int(y) for y in t.index], "unit_source": "萬噸 (1e4 m3) -> 1e8 m3",
             "header": "1~5月供水量: 彰化灌溉 雲林灌溉 公共 工業",
             "passed": list(t.index) == list(range(2015, 2022))}
    return t, check


def moa_crop_share() -> pd.DataFrame:
    """Crop-1 share of the MOA 691 office totals (Changhua, Yunlin), years with both crops."""
    m = pd.read_csv(MOA_691_CSV, encoding="utf-8-sig")
    m["year"] = m["年別"].astype(int) + 1911
    out = {}
    for d, off in (("chc", "彰化"), ("yl", "雲林")):
        sub = m[m["會別"].astype(str).str.strip() == off]
        p = sub.pivot_table(index="year", columns="期作", values="灌溉用水量", aggfunc="sum")
        out[d] = p["一"] / (p["一"] + p["二"])
        out[d + "_total"] = p["一"] + p["二"]
    return pd.DataFrame(out)


def v3_monthly() -> pd.DataFrame:
    """v3 in-fan monthly volumes per district (10^8 m3), 2012-2022 (its timing only is used)."""
    z = np.load(V3_NPZ, allow_pickle=True)
    if str(z["sw_version"]) != "v3":
        raise ValueError("surface_water_v3.npz is not a v3 build")
    return pd.DataFrame(z["vol_fan_m3"].T / 1e8, columns=["chc", "yl"],
                        index=pd.to_datetime(z["dates"]))
