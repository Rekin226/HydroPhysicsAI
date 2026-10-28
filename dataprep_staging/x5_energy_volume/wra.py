"""Parse the WRA 2025 trial report's county groundwater-use cells (estimator (a) input).

The tables are read from the report's text layer (``x5const.WRA_TXT``) by anchor: the table
caption, then the first row starting with 地下水 (groundwater), whose seven numbers are the
columns 生活, 工業, 灌溉, 養殖, 畜牧, 小計, 合計 (``x5const.WRA_COLS``). The 2023/2024 Jan-May
irrigation pumping is the row 灌溉抽水量(億噸) after the 近 2 年 table caption. Every parsed
row is checked: the agricultural subtotal and the total must add up to 0.011 (rounding).

Usage: python -I wra.py   (prints the parsed cells)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_energy_volume import x5const as K  # noqa: E402

_NUM = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")


def _lines() -> list[str]:
    return K.WRA_TXT.read_text(encoding="utf-8").split("\n")


def _after(lines: list[str], anchor: str, start: str, limit: int = 12) -> tuple[int, str]:
    i0 = next(i for i, s in enumerate(lines) if anchor in s.replace("　", " "))
    for j in range(i0, i0 + limit):
        if lines[j].strip().startswith(start):
            return j, lines[j]
    raise ValueError(f"no row {start!r} within {limit} lines of {anchor!r}")


def groundwater_row(lines: list[str], anchor: str) -> dict:
    j, row = _after(lines, anchor, "地下水")
    v = [float(x) for x in _NUM.findall(row)]
    if len(v) != len(K.WRA_COLS):
        raise ValueError(f"{anchor}: expected {len(K.WRA_COLS)} numbers, got {v}")
    d = dict(zip(K.WRA_COLS, v, strict=True))
    agri = d["irrigation"] + d["aquaculture"] + d["livestock"]
    tot = d["domestic"] + d["industry"] + d["agri_subtotal"]
    if abs(agri - d["agri_subtotal"]) > 0.011 or abs(tot - d["total"]) > 0.011:
        raise ValueError(f"{anchor}: row does not add up: {d}")
    d["line"] = j + 1
    return d


def _recent(lines: list[str], anchor: str) -> dict:
    i0 = next(i for i, s in enumerate(lines) if anchor in s)
    for j in range(i0, i0 + 16):
        if "灌溉抽水量(億噸)" in lines[j]:
            v = [float(x) for x in _NUM.findall(lines[j].split("灌溉抽水量(億噸)")[-1])]
            if len(v) == 2:
                return {"2023": v[0], "2024": v[1], "line": j + 1}
    raise ValueError(f"{anchor}: no 灌溉抽水量(億噸) row with two values")


def parse() -> dict:
    """{county: {"annual": row, "janmay": row, "recent": {2023, 2024}}} in 億 m3."""
    lines = _lines()
    out = {}
    for c, a in K.WRA_TABLES.items():
        out[c] = {"annual": groundwater_row(lines, a["annual"]),
                  "janmay": groundwater_row(lines, a["janmay"]),
                  "recent": _recent(lines, a["recent"])}
    return out


def irrigation_reference(w: dict) -> dict:
    """Irrigation groundwater, m3/yr or m3 per Jan-May, per county:
    ``annual_2011_24`` and ``janmay_2011_24`` (the printed 14-year means) and
    ``janmay_2011_22`` = (14 x Jan-May mean - 2023 - 2024) / 12, the pre-2023 mean (U7)."""
    out = {}
    for c, d in w.items():
        jm = d["janmay"]["irrigation"]
        r = d["recent"]
        out[c] = {"annual_2011_24": d["annual"]["irrigation"] * 1e8,
                  "janmay_2011_24": jm * 1e8,
                  "janmay_2011_22": (K.N_WRA_YEARS * jm - r["2023"] - r["2024"])
                  / (K.N_WRA_YEARS - 2) * 1e8,
                  "janmay_2011_22_rounding_halfwidth": 0.005 * K.N_WRA_YEARS
                  / (K.N_WRA_YEARS - 2) * 1e8}
    return out


if __name__ == "__main__":
    w = parse()
    print(json.dumps({"cells": w, "irrigation_m3": irrigation_reference(w)}, indent=1,
                     ensure_ascii=False))
