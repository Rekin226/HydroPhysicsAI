"""Table 1 of the WRA Changhua-Yunlin monthly data-analysis reports (series 007001).

Each report (``data_fetch_api/new_measurements/wra_cy_monthly_reports_007001/cy_<YYYY-MM>.pdf``)
has a table 彰雲地區深層水準樁 <ROC year> 年 <month> 月觀測成果 with one row per deep benchmark:
station name, depth (m), 單月變化量 (mm) of the same month a year earlier and of the report month,
and two cumulative changes (mm) over fixed periods. Values print with one decimal, positive =
rebound, negative = compression, ``-`` where missing; an arrow after a value compares it with
the year before and is kept only as a symbol.

Pure functions here: locate the table page, parse its poppler layout text deterministically,
convert a structured-output transcription into the same rows, and compare two reads. The
deterministic parse and the model transcription of the page image are independent reads; a
value is verified when they agree (a third read, the second model on the layout text, settles
a disagreement). Nothing here computes, fills or rounds a printed number.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

#: printed (station, depth m) -> sensor id of the agency feed
STATION_TO_SENSOR = {("溪州國小", 300): "SJES", ("土庫國中", 400): "TKJS(3)",
                     ("土庫國中", 300): "TKJS(1)", ("土庫國中", 130): "TKJS(2)",
                     ("中科虎尾", 300): "NTUH(1)", ("中科虎尾", 70): "NTUH(2)",
                     ("秀潭國小", 70): "STES", ("灣內國小", 100): "WNES",
                     ("頂寮安檢所", 150): "DLIO"}
ALIAS = {"中科虎尾園區": "中科虎尾", "溪州": "溪州國小", "土庫": "土庫國中", "秀潭": "秀潭國小",
         "虎尾": "中科虎尾", "灣內": "灣內國小"}
CAPTION = re.compile(r"表\s*\d+\s*彰雲地區深層水準樁\s*(\d{3})\s*年\s*(\d{1,2})\s*月\s*觀測成果")
_NUM = r"[-−–]?\d+(?:\.\d+)?"
_ARROWS = "↑↓↘↗↖↙→←▲▼"
_TOKEN = re.compile(rf"^({_NUM}|[-−–—]+)([{_ARROWS}*＊]*)$")
_SYMBOL = re.compile(rf"^[{_ARROWS}]+$")
_SIGN_COMP = re.compile(r"正值[為代表]*(?:壓縮|下陷)")
_SIGN_REB = re.compile(r"正值[為代表]*回脹")
_ROW = re.compile(r"^\s*(\d{1,2})\s+(?:([彰化雲林])\s+)?([^\s\d]{2,8})\s+(\d{2,3})\s+(.*)$")
_YM = re.compile(r"\d{3}/\d{1,2}")
_RANGE = re.compile(r"\d{3}/\d{1,2}\s*[~～-]\s*\d{3}/\d{1,2}")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["report_roc_year", "report_month", "positive_means", "col_month_prev",
                 "col_month_curr", "col_cum_prev", "col_cum_curr", "rows"],
    "properties": {
        "positive_means": {"type": "string", "enum": ["compression", "rebound", "not printed"],
                           "description": "what a positive value means according to the "
                                          "table's own note (註), e.g. 正值代表壓縮 or "
                                          "正值代表下陷 -> compression, 正值代表回脹 -> "
                                          "rebound"},
        "report_roc_year": {"type": "integer", "description": "民國 year in the table caption"},
        "report_month": {"type": "integer", "description": "month in the table caption"},
        "col_month_prev": {"type": "string",
                           "description": "header of the first 單月變化量 column, e.g. 110/12"},
        "col_month_curr": {"type": "string",
                           "description": "header of the second 單月變化量 column, e.g. 111/12"},
        "col_cum_prev": {"type": "string",
                         "description": "header of the first 累積變化量 column, e.g. "
                                        "109/4~110/12"},
        "col_cum_curr": {"type": "string",
                         "description": "header of the second 累積變化量 column"},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["row", "county", "station", "depth_m", "month_prev_mm",
                             "month_curr_mm", "cum_prev_mm", "cum_curr_mm"],
                "properties": {
                    "row": {"type": "integer", "description": "編號"},
                    "county": {"type": "string", "description": "縣市 column (彰化 or 雲林), "
                                                                "empty if not printed"},
                    "station": {"type": "string", "description": "站名 exactly as printed"},
                    "depth_m": {"type": ["integer", "null"], "description": "設置深度 (m)"},
                    "month_prev_mm": {"type": ["number", "null"],
                                      "description": "first 單月變化量 value, sign as printed; "
                                                     "null if printed as -"},
                    "month_curr_mm": {"type": ["number", "null"],
                                      "description": "second 單月變化量 value"},
                    "cum_prev_mm": {"type": ["number", "null"],
                                    "description": "first 累積變化量 value"},
                    "cum_curr_mm": {"type": ["number", "null"],
                                    "description": "second 累積變化量 value"},
                },
            },
        },
    },
}

SYSTEM = (
    "You transcribe one printed table from a Taiwanese government report: 彰雲地區深層水準樁 "
    "觀測成果 (deep benchmark results). Copy every row exactly as printed: 編號, 縣市, 站名, "
    "設置深度 (m), the two monthly values (單月變化量 or 單月下陷量, mm) in column order, and "
    "the two cumulative values (累積變化量 or 累積下陷量, mm) in column order, plus the four "
    "column headers (year/month labels) and what a positive value means according to the "
    "table's note (註). Keep the sign as printed. Ignore trend arrows (↑ ↓ ↘ ↗), trend "
    "columns (趨勢) and footnote stars. Never compute, round, infer or fill a number: if a "
    "cell is printed as - (with or without a star) or is empty, return null. Return only the "
    "JSON."
)


# ------------------------------------------------------------------------------- PDF plumbing
def page_text(pdf: Path, page: int) -> str:
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf),
                           "-"], check=True, capture_output=True, text=True).stdout


def n_pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True,
                         text=True).stdout
    return int(next(ln.split()[-1] for ln in out.splitlines() if ln.startswith("Pages")))


def find_table_page(pdf: Path) -> int | None:
    for p in range(1, n_pages(pdf) + 1):
        if CAPTION.search(re.sub(r"[ \t]+", " ", page_text(pdf, p))):
            return p
    return None


# ------------------------------------------------------------------------------- parsing
def _num(tok: str) -> tuple[float | None, str]:
    m = _TOKEN.match(tok)
    if not m:
        raise ValueError(f"unparsed cell {tok!r}")
    v, sym = m.group(1), m.group(2)
    if set(v) <= set("-−–—"):
        return None, sym
    return float(v.replace("−", "-").replace("–", "-")), sym


def canonical_station(name: str) -> str:
    name = name.strip()
    return ALIAS.get(name, name)


def parse_layout(text: str) -> dict:
    """Deterministic read of the table from poppler layout text (one page)."""
    flat = re.sub(r"[ \t]+", " ", text)
    cap = CAPTION.search(flat)
    if not cap:
        raise ValueError("caption not found")
    lines = text[text.find(text.splitlines()[0]):].splitlines()
    start = next(i for i, ln in enumerate(lines) if CAPTION.search(re.sub(r"[ \t]+", " ", ln)))
    out = {"report_roc_year": int(cap.group(1)), "report_month": int(cap.group(2)),
           "positive_means": "not printed",
           "col_month_prev": None, "col_month_curr": None, "col_cum_prev": None,
           "col_cum_curr": None, "rows": []}
    trend_cols = False
    note = ""
    for k, ln in enumerate(lines[start + 1:]):
        if ln.strip().startswith("註"):
            note = " ".join(x.strip() for x in lines[start + 1 + k:start + 5 + k])
            break
        if "趨勢" in ln:
            trend_cols = True
        ranges = _RANGE.findall(ln)
        if ranges and out["col_cum_prev"] is None:
            rest = _RANGE.sub(" ", ln)
            yms = _YM.findall(rest)
            if len(yms) >= 2:
                out["col_month_prev"], out["col_month_curr"] = yms[0], yms[1]
            if len(ranges) >= 2:
                out["col_cum_prev"] = re.sub(r"\s+", "", ranges[0]).replace("～", "~")
                out["col_cum_curr"] = re.sub(r"\s+", "", ranges[1]).replace("～", "~")
            continue
        m = _ROW.match(ln)
        if not m:
            continue
        toks = m.group(5).split()
        if trend_cols:      # value, value, trend, value, value, trend ('-' = no trend)
            if len(toks) != 6:
                raise ValueError(f"row {m.group(1)}: {len(toks)} cells (6 expected) in {ln!r}")
            toks = [toks[0], toks[1], toks[3], toks[4]]
        else:
            toks = [t for t in toks if not _SYMBOL.match(t)]
            if len(toks) != 4:
                raise ValueError(f"row {m.group(1)}: {len(toks)} value cells in {ln!r}")
        vals = [_num(t)[0] for t in toks]
        out["rows"].append({"row": int(m.group(1)), "county": m.group(2) or "",
                            "station": m.group(3), "depth_m": int(m.group(4)),
                            "month_prev_mm": vals[0], "month_curr_mm": vals[1],
                            "cum_prev_mm": vals[2], "cum_curr_mm": vals[3]})
    if not out["rows"]:
        raise ValueError("no table row parsed")
    comp, reb = bool(_SIGN_COMP.search(note)), bool(_SIGN_REB.search(note))
    if comp != reb:
        out["positive_means"] = "compression" if comp else "rebound"
    return out


def rows_frame(table: dict, report_month: str, read: str) -> pd.DataFrame:
    """Long rows (one per station) with the canonical station and sensor id."""
    rows = []
    for r in table.get("rows", []):
        st = canonical_station(str(r.get("station", "")))
        dep = r.get("depth_m")
        rows.append({"report_month": report_month, "read": read, "row": r.get("row"),
                     "station": st, "depth_m": dep,
                     "sensor_id": STATION_TO_SENSOR.get((st, int(dep)) if dep is not None
                                                        else (st, -1), ""),
                     "month_prev_mm": r.get("month_prev_mm"),
                     "month_curr_mm": r.get("month_curr_mm"),
                     "cum_prev_mm": r.get("cum_prev_mm"), "cum_curr_mm": r.get("cum_curr_mm"),
                     "positive_means": table.get("positive_means"),
                     "caption_roc_year": table.get("report_roc_year"),
                     "caption_month": table.get("report_month"),
                     "col_month_prev": table.get("col_month_prev"),
                     "col_month_curr": table.get("col_month_curr"),
                     "col_cum_prev": table.get("col_cum_prev"),
                     "col_cum_curr": table.get("col_cum_curr")})
    return pd.DataFrame(rows)


VALUE_COLS = ("month_prev_mm", "month_curr_mm", "cum_prev_mm", "cum_curr_mm")


def _same(x, y) -> bool:
    if pd.isna(x) and pd.isna(y):
        return True
    if pd.isna(x) or pd.isna(y):
        return False
    return abs(float(x) - float(y)) < 1e-9


def _label(x) -> str:
    return re.sub(r"\s+", "", str(x)) if x is not None and not (isinstance(x, float)
                                                                and pd.isna(x)) else ""


def compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Merge two reads of one report on (station, depth) and flag per-row agreement of the
    depth, the four values and the column headers."""
    key = ["report_month", "station", "depth_m"]
    m = a.merge(b, on=key, how="outer", suffixes=("_a", "_b"), indicator="present")
    agree = []
    for r in m.itertuples():
        ok = r.present == "both"
        if ok:
            for c in VALUE_COLS + ("caption_month",):
                if not _same(getattr(r, f"{c}_a"), getattr(r, f"{c}_b")):
                    ok = False
            for c in ("col_month_prev", "col_month_curr", "positive_means"):
                if _label(getattr(r, f"{c}_a")) != _label(getattr(r, f"{c}_b")):
                    ok = False
        agree.append(bool(ok))
    m["agree"] = agree
    return m


def ym_label_to_month(label: str | None) -> str | None:
    """'111/12' -> '2022-12'."""
    if not label or not _YM.fullmatch(str(label).strip()):
        return None
    y, mo = str(label).strip().split("/")
    return f"{int(y) + 1911}-{int(mo):02d}"
