"""Drought-well pumping into the public supply (WRA water-use statistics, 附錄四 抗旱水井抽水量
調查, ROC 110 and 111), parsed from the PDF text layer and cross-checked.

* 110 appendix (created 2022-11): 附表4-1, 2021-01-01..2021-07-15, wells and 累計抽水量
  (萬立方公尺) by county.
* 111 appendix (created 2023-12): 附表4-1 (2022-01-01..2022-12-31) and 附表4-2 (2020, 2021,
  2022). The 111 text names Yunlin 1,095 萬立方公尺, but both of its tables give 1,095.00 to
  Changhua and 291.00 to Yunlin: the text swaps the counties (recorded as a source issue).

Checks: 附表4-2's 2021 row equals the 110 appendix's 附表4-1; 附表4-2's 2022 row equals the
111 附表4-1; each table's county rows sum to its 全台 row.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import x2const as K  # noqa: E402

FILES = {110: "wra_wateruse_110_drought.pdf", 111: "wra_wateruse_111_drought.pdf"}
PUBLISHED = {110: "2022-11", 111: "2023-12"}
COUNTIES = {"彰化": "changhua", "雲林": "yunlin"}
AREAS = ("基隆", "新北", "桃園", "新竹", "苗栗", "臺中", "彰化", "雲林", "嘉義", "臺南", "高雄",
         "屏東", "宜花東", "花蓮", "臺東", "全台")


def text(roc: int) -> str:
    p = K.L0["wra_water_use_annual"] / FILES[roc]
    return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True,
                          check=True).stdout.decode("utf-8", "replace")


def _rows(block: str) -> dict[str, list[float]]:
    out = {}
    for ln in block.splitlines():
        m = re.match(r"^\s*(\S+)\s+(.*)$", ln)
        if not m:
            continue
        name = m.group(1)
        nums = [float(t.replace(",", "")) if t not in ("-", "－") else 0.0
                for t in m.group(2).split() if re.match(r"^([0-9][0-9,]*(\.[0-9]+)?|-|－)$", t)]
        if nums and name in AREAS:
            out[name] = nums
    return out


def parse() -> dict:
    t110, t111 = text(110), text(111)
    a = t110.split("附表 4-1", 1)[1].split("備註：", 1)[0]
    b1 = t111.split("附表 4-1", 1)[1].split("附表 4-2", 1)[0].split("備註：", 1)[0]
    b2 = t111.split("附表 4-2", 1)[1]
    r110, r111, r112 = _rows(a), _rows(b1), _rows(b2)
    res: dict = {"values": [], "checks": [], "issues": []}
    for zh, county in COUNTIES.items():
        res["values"].append({"county": county, "year": 2021, "wells": r110[zh][-2],
                              "volume_1e4m3": r110[zh][-1], "period": "2021-01-01..07-15",
                              "source": f"{FILES[110]} 附表4-1", "published": PUBLISHED[110]})
        res["values"].append({"county": county, "year": 2022, "wells": r111[zh][-2],
                              "volume_1e4m3": r111[zh][-1], "period": "2022-01-01..12-31",
                              "source": f"{FILES[111]} 附表4-1", "published": PUBLISHED[111]})
        w20, v20, w21, v21, w22, v22 = r112[zh][:6]
        res["values"].append({"county": county, "year": 2020, "wells": w20,
                              "volume_1e4m3": v20, "period": "2020 (from 2020-11-19)",
                              "source": f"{FILES[111]} 附表4-2", "published": PUBLISHED[111]})
        res["checks"].append({"check": "4-2 2021 = 110 4-1", "county": county,
                              "ok": (w21, v21) == (r110[zh][-2], r110[zh][-1])})
        res["checks"].append({"check": "4-2 2022 = 111 4-1", "county": county,
                              "ok": (w22, v22) == (r111[zh][-2], r111[zh][-1])})
    for name, rows in (("110 4-1", r110), ("111 4-1", r111)):
        tot = rows.get("全台")
        s = sum(v[-1] for k, v in rows.items() if k != "全台")
        res["checks"].append({"check": f"{name} counties sum to 全台", "ok":
                              tot is not None and abs(s - tot[-1]) < 0.05, "sum": s,
                              "printed": tot[-1] if tot else None})
    if "雲林抽水量為\n1,095" in t111 or re.search(r"雲林抽水量為\s*1,095", t111):
        res["issues"].append("111 appendix text gives Yunlin 1,095 萬立方公尺; its tables give "
                             "Changhua 1,095.00 and Yunlin 291.00 (tables used)")
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(parse(), ensure_ascii=False, indent=1))
