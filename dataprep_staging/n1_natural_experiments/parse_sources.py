"""Deterministic reads of the L0 report PDFs' text layer (poppler ``pdftotext -layout``).

* :func:`page_text` / :func:`find_quote`: the text of one page, and a whitespace-insensitive
  search for a quoted phrase on it (every intervention row's quote is checked this way).
* :func:`drought_wells`: the Changhua (附表 1-8) and Yunlin (附表 1-10) drought-well tables of
  the 2022 inventory report (agency, township, well name, TWD97 x/y), checked against the
  report's own count table (表 3).
* :func:`jiji_janmay_table`: the Jiji weir Jan-May supply table of the 110 briefing (p. 20).
* :func:`public_well_anchors`: the printed Jan-May public-well volumes of 2020 and 2021 (110
  briefing pp. 10 and 19), used to check the chart transcription.

The PDFs are untrusted data: only poppler reads them; nothing in them is executed.
"""
from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from n1_natural_experiments import n1const as K  # noqa: E402

AGENCIES = {"自來水公司": "Taiwan Water Corporation (台水)", "農田水利署": "Irrigation Agency (農水署)",
            "台糖": "Taiwan Sugar (台糖)", "臺糖": "Taiwan Sugar (台糖)"}
_XY = re.compile(r"(\d{6}\.\d+)\s+(\d{7}\.\d+)")


@cache
def page_text(src: str, page: int, layout: bool = True) -> str:
    args = ["pdftotext", "-f", str(page), "-l", str(page)]
    if layout:
        args.append("-layout")
    out = subprocess.run(args + [str(K.SRC[src]), "-"], check=True, capture_output=True)
    return out.stdout.decode("utf-8", errors="replace")


@cache
def doc_text(src: str) -> str:
    out = subprocess.run(["pdftotext", "-layout", str(K.SRC[src]), "-"], check=True,
                         capture_output=True)
    return out.stdout.decode("utf-8", errors="replace")


def _squash(s: str) -> str:
    """Whitespace removed, NFKC-normalised (the PDFs mix CJK compatibility ideographs)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def find_quote(src: str, page: int, quote: str) -> bool:
    """True when ``quote`` occurs on the page once all whitespace is removed from both."""
    q = _squash(quote)
    return bool(q) and (q in _squash(page_text(src, page)) or
                        q in _squash(page_text(src, page, layout=False)))


def n_pages(src: str) -> int:
    out = subprocess.run(["pdfinfo", str(K.SRC[src])], check=True, capture_output=True)
    m = re.search(r"Pages:\s+(\d+)", out.stdout.decode())
    return int(m.group(1))


# ------------------------------------------------------------------ drought wells (2022)
def _table_block(text: str, start_pat: str, end_pat: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if re.search(start_pat, ln) and "彙整表" in _squash(ln))
    end = next((i for i in range(start + 1, len(lines)) if re.search(end_pat, lines[i])
                and "彙整表" in _squash(lines[i])), len(lines))
    return lines[start:end]


def _agency(line: str, prev: list[str]) -> str | None:
    for cand in [line] + prev[::-1]:
        for k, v in AGENCIES.items():
            if k in cand:
                return v
    return None


def drought_wells() -> dict:
    """Rows of the Changhua and Yunlin drought-well tables with a consistency report."""
    text = doc_text("drought_wells")
    out = {"rows": [], "checks": {}}
    for county, start, end in (("Changhua", r"附\s*表\s*1-8\s+彰", r"附\s*表\s*1-9\s"),
                               ("Yunlin", r"附\s*表\s*1-10\s+雲", r"附\s*表\s*1-11\s")):
        block = _table_block(text, start, end)
        nums = []
        for i, ln in enumerate(block):
            m = _XY.search(ln)
            if not m:
                continue
            x, y = float(m.group(1)), float(m.group(2))
            head = ln[:m.start()]
            mnum = re.match(r"^\s*(\d{1,3})\s+", head)
            if not mnum:
                raise ValueError(f"row without a number: {ln!r}")
            num = int(mnum.group(1))
            rest = head[mnum.end():]
            ag = _agency(ln, block[max(0, i - 2):i])
            for k in AGENCIES:
                rest = rest.replace(k, " ")
            rest = re.sub(r"第[五十一]+區處|雲林管理處|彰化管理處|管理處", " ", rest)
            toks = rest.split()
            town = next((t for t in toks if re.search(r"[鄉鎮市區]$", t)), None)
            cnt = toks[-1] if toks and toks[-1].isdigit() else None
            name_toks = [t for t in toks if t != town and t != cnt]
            nums.append(num)
            out["rows"].append({"county": county, "table_no": num, "agency": ag,
                                "township": town, "well_name": " ".join(name_toks),
                                "n_wells": int(cnt) if cnt else None, "x_twd97": x,
                                "y_twd97": y})
        expected = 49 if county == "Changhua" else 521
        missing = [i for i in range(1, expected + 1) if i not in set(nums)]
        out["checks"][county] = {"rows_parsed": len(nums), "stated_total": expected,
                                 "missing_numbers": missing,
                                 "duplicate_numbers": sorted({n for n in nums
                                                              if nums.count(n) > 1})}
    # the report's own count table (表 3): Changhua 49 = 26 irrigation + 23 Taiwan Water;
    # Yunlin 521 = 421 irrigation + 98 Taiwan Water + 2 Taiwan Sugar
    stated = {"Changhua": {"Irrigation Agency (農水署)": 26, "Taiwan Water Corporation (台水)": 23},
              "Yunlin": {"Irrigation Agency (農水署)": 421, "Taiwan Water Corporation (台水)": 98,
                         "Taiwan Sugar (台糖)": 2}}
    for c, st in stated.items():
        got = {}
        for r in out["rows"]:
            if r["county"] == c:
                got[r["agency"]] = got.get(r["agency"], 0) + 1
        out["checks"][c]["by_agency_parsed"] = got
        out["checks"][c]["by_agency_stated_table3"] = st
    return out


# ------------------------------------------------------------------ Jiji Jan-May supply
def jiji_janmay_table() -> dict:
    """The 110 briefing's table 集集堰地面水供應概況 (1~5月供水量, 萬噸), 104-110."""
    t = page_text("brief110", 20)
    rows = []
    for ln in t.splitlines():
        m = re.match(r"^\s*(10[4-9]|110)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m:
            roc = int(m.group(1))
            rows.append({"year": roc + 1911, "roc_year": roc,
                         "changhua_irrigation_1e4m3": float(m.group(2)),
                         "yunlin_irrigation_1e4m3": float(m.group(3)),
                         "public_1e4m3": float(m.group(4)),
                         "industrial_1e4m3": float(m.group(5))})
    if len(rows) != 7:
        raise ValueError(f"expected 7 rows (104-110), parsed {len(rows)}")
    return {"src": "brief110", "page": 20, "unit": "萬噸 (1e4 m3), January-May", "rows": rows,
            "quote": "集集堰地面水供應概況"}


# ------------------------------------------------------------------ printed anchors
_ANCHOR = re.compile(r"(台水公司|農水署)抽水量\s*109年([\d,]+)萬噸、\s*110年([\d,]+)萬噸")


def public_well_anchors() -> list[dict]:
    out = []
    for page, county in ((10, "Changhua"), (19, "Yunlin")):
        t = _squash(page_text("brief110", page))
        for m in _ANCHOR.finditer(t):
            ag = ("Taiwan Water Corporation (台水)" if m.group(1) == "台水公司"
                  else "Irrigation Agency (農水署)")
            for yr, v in ((2020, m.group(2)), (2021, m.group(3))):
                out.append({"src": "brief110", "page": page, "county": county, "agency": ag,
                            "year": yr, "value_1e4m3": float(v.replace(",", "")),
                            "quote": m.group(0)})
    if len(out) != 8:
        raise ValueError(f"expected 8 printed anchors, found {len(out)}")
    return out
