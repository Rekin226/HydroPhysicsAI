"""Readers for the MOI 20 m DTM sheet archives (data.gov.tw 176927, 2025 edition), WP W3.

Each county archive ``分幅_<county>20MDEM(2025).zip`` holds one pair of files per 1:5000 map
sheet: ``<sheet>dem.hdr`` (Big5 text, one value per line) and ``<sheet>dem.grd`` (ASCII
``E N H`` triples, TWD97 TM2 metres, TWVD2001 heights). The header field order is the one in
``schema_hdr.csv`` (the CSV's description column is shifted by one row from the 14th field on;
the order below was checked against the values themselves).

Downloaded archives are untrusted data: they are only read here (zipfile + numpy text
parsing), never executed or extracted to disk. Run callers with ``python -I``.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Header lines in order (schema_hdr.csv, verified against the sheet values).
HDR_FIELDS = (
    "sheet_name",          # 圖幅名稱 (1:5000 sheet name)
    "sheet_id",            # 圖幅號碼
    "plane_datum",         # 平面坐標基準 (TWD97[2010])
    "height_datum",        # 高程坐標基準 (TWVD2001)
    "scale",               # 比例尺等級
    "dx",                  # 東西向網格間距 (m)
    "dy",                  # 南北向網格間距 (m)
    "n_points",            # 總網格點數
    "n_cols",              # 行數 (east-west)
    "n_rows",              # 列數 (north-south)
    "sw_e",                # 圖幅西南隅E坐標
    "sw_n",                # 圖幅西南隅N坐標
    "method_code",         # 生產方式代碼 (00-04 photogrammetry, 10-12 LiDAR)
    "production_equipment",  # 生產設備名稱
    "acquisition_equipment",  # 原始資料採集設備名稱
    "flying_height_m",     # 飛行高度
    "max_ground_m",        # 最高地面高度
    "min_ground_m",        # 最低地面高度
    "mean_ground_m",       # 平均地面高
    "acquisition_date",    # 原始資料生產日期 (YYYYMMDD): the epoch of the surface
    "acquisition_agency",  # 原始資料生產單位
    "dtm_date",            # DTM（DSM）生成日期 (YYYYMMDD)
    "dtm_producer",        # DTM（DSM）製作單位名稱
)
_NUM = {"scale", "dx", "dy", "n_points", "n_cols", "n_rows", "sw_e", "sw_n",
        "flying_height_m", "max_ground_m", "min_ground_m", "mean_ground_m"}
_SHEET = re.compile(r"^(?P<id>\d+)dem\.(?P<ext>hdr|grd)$", re.I)


def _decode(b: bytes) -> str:
    for enc in ("big5", "cp950", "utf-8-sig"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin1")


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def decimal_year(yyyymmdd: str) -> float:
    """YYYYMMDD -> decimal year (NaN if unparseable)."""
    s = str(yyyymmdd).strip()
    if not re.fullmatch(r"\d{8}", s):
        return float("nan")
    import datetime as dt
    try:
        d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return float("nan")
    y0 = dt.date(d.year, 1, 1)
    y1 = dt.date(d.year + 1, 1, 1)
    return d.year + (d - y0).days / (y1 - y0).days


@dataclass
class Sheet:
    archive: str
    member_hdr: str
    member_grd: str
    header: dict


def parse_header(text: str) -> dict:
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    h = {}
    for i, k in enumerate(HDR_FIELDS):
        v = lines[i] if i < len(lines) else ""
        h[k] = _num(v) if k in _NUM else v
    h["n_header_lines"] = len(lines)
    h["acquisition_year"] = decimal_year(h["acquisition_date"])
    h["dtm_year"] = decimal_year(h["dtm_date"])
    return h


def list_sheets(archive: str | Path) -> list[Sheet]:
    """Sheets in one county archive (header parsed; grid not read)."""
    out = []
    with zipfile.ZipFile(archive) as z:
        names = {}
        for info in z.infolist():
            base = info.filename.rsplit("/", 1)[-1]
            m = _SHEET.match(base)
            if m:
                names.setdefault(m["id"], {})[m["ext"].lower()] = info.filename
        for sid, d in sorted(names.items()):
            if "hdr" not in d or "grd" not in d:
                continue
            h = parse_header(_decode(z.read(d["hdr"])))
            h["file_sheet_id"] = sid
            out.append(Sheet(str(archive), d["hdr"], d["grd"], h))
    return out


def read_grid(archive: str | Path, member: str) -> np.ndarray:
    """(n, 3) float64 array of E, N, H for one sheet's .grd member."""
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member)
    arr = np.loadtxt(io.BytesIO(raw), dtype="float64", ndmin=2)
    if arr.shape[1] != 3:
        raise ValueError(f"{member}: expected 3 columns, got {arr.shape[1]}")
    return arr
