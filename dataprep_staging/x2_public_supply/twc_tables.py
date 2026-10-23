"""Parse the Taiwan Water Corporation statistical yearbooks (台灣自來水事業統計年報) from their
PDF text layer (poppler ``pdftotext -layout``; the PDFs are untrusted data and only read).

Tables used (same numbering in every yearbook 105-114):

* 表2 出水量 (1) 按月別分: monthly plant output (m3) of every district (區管理處) for the
  yearbook's own year, plus the company's monthly totals for every year since 1974.
* 表2 出水量 (2) 按區處別分: annual output by district for every year since 1974.
* 表3 出水量、用電量暨藥品使用量: annual output of every supply system (供水系統) of the year.
* 表1 現有水源與供水區域: each system's sources in four columns (地下水, 地面水, 水庫水,
  海/鹽水); used to classify the Changhua and Yunlin systems as groundwater-only or mixed.

Validation (``validate``): months sum to the printed annual total of each row; districts sum
to the company total in each month and year; systems sum to their district; the same
district-year printed in different yearbooks agrees (later yearbooks state that a revised
number supersedes: 凡例 八).

Usage: python -I twc_tables.py   (writes reports/X2_public_supply/evidence/twc_tables.json)
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import x2const as K  # noqa: E402

YEARS = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
         114]
DISTRICTS = ["第一區管理處", "第二區管理處", "第三區管理處", "第四區管理處", "第五區管理處",
             "第六區管理處", "第七區管理處", "第八區管理處", "第九區管理處", "第十區管理處",
             "第十一區管理處", "第十二區管理處", "屏東區管理處"]
TARGET = {"第五區管理處": "district5", "第十一區管理處": "district11"}
NUM = re.compile(r"^-?[\d,]+(\.\d+)?$")
ASCII_NUM = re.compile(r"^-?[0-9][0-9,]*(\.[0-9]+)?$")


def pdf(y: int) -> Path:
    return K.L0["twc_statistics"] / f"twc_yearbook_{y}.pdf"


def page_text(path: Path, page: int) -> str:
    out = K.TEXT_DIR / f"{path.stem}_p{page:03d}.txt"
    if out.exists():
        return out.read_text(encoding="utf-8")
    t = subprocess.run(["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(path),
                        "-"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(t, encoding="utf-8")
    return t


def n_pages(path: Path) -> int:
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, check=True).stdout
    return int(re.search(rb"Pages:\s+(\d+)", info).group(1))


def norm(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def find_pages(path: Path, marks: tuple[str, ...], lo: int = 20, hi: int = 200) -> list[int]:
    pages = []
    for p in range(lo, min(hi, n_pages(path)) + 1):
        head = norm("\n".join(page_text(path, p).splitlines()[:8]))
        if all(m in head for m in marks):
            pages.append(p)
    return pages


def split_row(line: str) -> tuple[str, list[str]]:
    """Row label (normalised) and the printed number tokens ('－' and '…' kept). A value is an
    ASCII number or a dash; full-width digits (ROC years such as 民 國 １ ０ １ 年) belong to
    the label."""
    toks = line.split()
    i = 0
    while i < len(toks) and not (ASCII_NUM.match(toks[i]) or toks[i] in ("－", "…")):
        i += 1
    label = norm(" ".join(t for t in toks[:i] if not re.search(r"[A-Za-z]", t)))
    return label, toks[i:]


def to_num(t: str) -> float | None:
    if t in ("－", "-"):
        return 0.0
    if t == "…":
        return None
    return float(t.replace(",", ""))


def roc_label(label: str) -> int | None:
    m = re.match(r"^民國(\d+)年$", label)
    return int(m.group(1)) if m else None


def parse_monthly(y: int) -> dict:
    """表2(1): {'pages', 'district': {name: {'total', 'months'[12]}}, 'company': {roc: ...}}."""
    path = pdf(y)
    pages = find_pages(path, ("表2", "出水量", "按月別分"))
    first, second = {}, {}
    for p in pages:
        lines = page_text(path, p).splitlines()
        hdr = norm(" ".join(lines[:12]))
        is_first = re.search(r"(?<!\d)1月", hdr) is not None
        for ln in lines:
            label, nums = split_row(ln)
            if not nums or not (label in DISTRICTS or roc_label(label)):
                continue
            vals = [to_num(t) for t in nums]
            key = label if label in DISTRICTS else roc_label(label)
            (first if is_first else second)[key] = {"vals": vals, "page": p}
    out = {"pages": pages, "district": {}, "company": {}}
    for key, a in list(first.items()):
        if len(a["vals"]) == 13:          # older yearbooks print the 12 months on one page
            rec = {"total": a["vals"][0], "months": a["vals"][1:], "pages": [a["page"]]}
            (out["district"] if isinstance(key, str) else out["company"])[
                key if isinstance(key, str) else str(key)] = rec
            first.pop(key)
    for key, a in first.items():
        b = second.get(key)
        if b is None or len(a["vals"]) != 7 or len(b["vals"]) != 6:
            continue
        rec = {"total": a["vals"][0], "months": a["vals"][1:] + b["vals"],
               "pages": [a["page"], b["page"]]}
        (out["district"] if isinstance(key, str) else out["company"])[
            key if isinstance(key, str) else str(key)] = rec
    return out


def parse_by_district(y: int) -> dict:
    """表2(2): {roc: {district or '總計': value}} with the page of each column block."""
    path = pdf(y)
    pages = find_pages(path, ("表2", "出水量", "按區處別分"))
    res: dict[str, dict] = {}
    src: dict[str, int] = {}
    for p in pages:
        lines = page_text(path, p).splitlines()
        hdr_line = next((ln for ln in lines if "時期及區處別" in norm(ln)), "")
        cols = re.findall(r"總計|第[一二三四五六七八九十]+區管理處|屏東區管理處", norm(hdr_line))
        for ln in lines:
            label, nums = split_row(ln)
            r = roc_label(label)
            if r is None or len(nums) != len(cols):
                continue
            for c, t in zip(cols, nums, strict=True):
                res.setdefault(str(r), {})[c] = to_num(t)
                src[c] = p
    return {"pages": pages, "values": res, "column_pages": src}


def parse_systems(y: int) -> dict:
    """表3: annual output of the systems of districts 5 and 11 (first number of the row)."""
    path = pdf(y)
    pages = find_pages(path, ("表3", "出水量", "用電量"))
    cur, out = None, {}
    for p in pages:
        lines = page_text(path, p).splitlines()
        if "全年總計" not in norm(" ".join(lines[:12])):
            continue
        for ln in lines:
            label, nums = split_row(ln)
            if label in DISTRICTS:
                cur = label
                if label in TARGET:
                    out.setdefault(label, {"total": to_num(nums[0]), "page": p, "systems": {}})
                continue
            if cur in TARGET and label.endswith("供水系統") and nums:
                out[cur]["systems"][label] = {"output": to_num(nums[0]), "page": p}
    return {"pages": pages, "districts": out}


def parse_purchases(y: int) -> dict:
    """表4 購入原、清水量值統計 (千立方公尺): the rows of districts 5 and 11 on both pages.
    Page 1: 合計, 原清水, 借道輸送, 金額, 北水處 清水, 金額, 農田水利 合計, 原水, 借道輸送,
    金額. Page 2: 北區 原水, 金額, 中區 原水, 金額, 南區 原水, 金額, 其他 原(清)水, 金額."""
    path = pdf(y)
    pages = find_pages(path, ("表4", "購入"))
    out: dict = {}
    for p in pages:
        lines = page_text(path, p).splitlines()
        second = "中區水資源" in norm(" ".join(lines[:14]))
        for ln in lines:
            label, nums = split_row(ln)
            if label in TARGET and nums:
                rec = out.setdefault(label, {})
                vals = [to_num(t) for t in nums]
                if second and len(vals) == 8:
                    rec.update({"north_raw": vals[0], "central_raw": vals[2],
                                "south_raw": vals[4], "other_raw": vals[6], "page2": p})
                elif not second and len(vals) == 10:
                    rec.update({"total": vals[0], "raw_clear": vals[1], "wheeling": vals[2],
                                "taipei_clear": vals[4], "irrigation_total": vals[6],
                                "irrigation_raw": vals[7], "irrigation_wheeling": vals[8],
                                "page1": p})
                elif not second and len(vals) == 1 and vals[0] == 0.0:
                    rec.update({"total": 0.0, "raw_clear": 0.0, "page1": p, "all_dash": True})
    return {"pages": pages, "unit": "1000 m3", "districts": out}


def page_words(path: Path, page: int) -> list[dict]:
    """Words with bounding boxes (poppler ``pdftotext -bbox``), parsed with a regex."""
    out = K.TEXT_DIR / f"{path.stem}_p{page:03d}.bbox.html"
    if not out.exists():
        t = subprocess.run(["pdftotext", "-bbox", "-f", str(page), "-l", str(page), str(path),
                            "-"], capture_output=True, check=True).stdout
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(t)
    words = []
    for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" '
                         r'yMax="([\d.]+)">([^<]*)</word>',
                         out.read_text(encoding="utf-8", errors="replace")):
        words.append({"x0": float(m.group(1)), "y0": float(m.group(2)),
                      "x1": float(m.group(3)), "y1": float(m.group(4)),
                      "t": html.unescape(m.group(5))})
    return words


def parse_sources(y: int) -> dict:
    """表1: for the systems of district 11 and the Yunlin system, the source text in each of
    the 地下水 / 地面水 / 水庫水 / 海鹽水 columns. Columns are bounded by the midpoints between
    the header words' centres (word boxes); numbers (capacity, population) and the service
    area text right of them are excluded."""
    path = pdf(y)
    pages = find_pages(path, ("表1", "現有水源與供水區域"))
    res = {}
    for p in pages:
        w = page_words(path, p)
        hdr = {}
        for x in w:
            if x["t"] in ("地下水", "地面水", "水庫水") or x["t"].startswith("海/"):
                hdr.setdefault("sea" if x["t"].startswith("海") else
                               {"地下水": "gw", "地面水": "sw", "水庫水": "res"}[x["t"]],
                               (x["x0"] + x["x1"]) / 2)
        if not {"gw", "sw", "res"} <= set(hdr):
            continue
        cols = sorted(hdr.items(), key=lambda kv: kv[1])
        cuts = [(cols[i][1] + cols[i + 1][1]) / 2 for i in range(len(cols) - 1)]
        starts = [x for x in w if re.match(r"^\d{4}\S+$", x["t"]) and x["x0"] < 90]
        ends = [x["y0"] for x in w if re.match(r"^(第[一二三四五六七八九十]+區|屏東區)管理處$",
                                                 norm(x["t"]))]
        starts.sort(key=lambda x: x["y0"])
        for i, s in enumerate(starts):
            code, name = s["t"][:4], s["t"][4:]
            if not (code.startswith("11") or name == "雲林"):
                continue
            y_end = min([t["y0"] for t in starts[i + 1:]] + [e for e in ends if e > s["y0"]]
                        + [10_000])
            sysw = [x for x in w if x["t"] == "供水系統" and abs(x["y0"] - s["y0"]) < 1.5]
            x_left = sysw[0]["x1"] - 0.5 if sysw else s["x1"] + 30
            block = [x for x in w if s["y0"] - 1 <= x["y0"] < y_end - 1 and x["x0"] >= x_left]
            num_x = [x["x0"] for x in block if re.match(r"^[\d,]+$", x["t"])]
            x_stop = min(num_x) - 1 if num_x else 10_000
            rec = {"name": name + "供水系統", "page": p, "gw": "", "sw": "", "res": "", "sea": ""}
            for x in sorted(block, key=lambda x: (round(x["y0"]), x["x0"])):
                if x["x0"] >= x_stop:
                    continue
                xc = (x["x0"] + x["x1"]) / 2
                k = cols[sum(xc > c for c in cuts)][0]
                rec[k] += x["t"]
            rec["groundwater_only"] = not (rec["sw"] or rec["res"] or rec["sea"])
            rec["well_count"] = well_count(rec["gw"])
            res[code] = rec
    return {"pages": pages, "systems": res}


CN_DIGIT = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
            "九": 9}


def cn_int(t: str) -> int:
    """Chinese numeral up to 99 (十, 十四, 二十, 二十三) or ASCII digits."""
    if t.isdigit():
        return int(t)
    if "十" in t:
        a, _, b = t.partition("十")
        return (CN_DIGIT.get(a, 1) if a else 1) * 10 + (CN_DIGIT.get(b, 0) if b else 0)
    return CN_DIGIT[t]


def well_count(text: str) -> int:
    """Sum of the well and intake counts printed in a 地下水 cell ('等十四口井', '4口井',
    '一口寬口井', '十二井'); well numbers such as '5.6.7.8號' are not counts."""
    n = 0
    for m in re.finditer(r"([0-9]+|[一二兩三四五六七八九十]+)(?=口|井)", text):
        pre = text[max(0, m.start() - 1):m.start()]
        if pre in ("." , "號") or re.match(r"[0-9.]", pre or " "):
            continue
        n += cn_int(m.group(1))
    return n


def validate(tab: dict) -> dict:
    checks = []
    for y, t in tab.items():
        mo = t["monthly"]
        for name, rec in mo["district"].items():
            s = sum(v or 0 for v in rec["months"])
            checks.append({"yearbook": y, "check": "months_sum_to_total", "row": name,
                           "sum": s, "printed": rec["total"], "diff": s - rec["total"]})
        roc = str(y)
        comp = mo["company"].get(roc)
        if comp and len(mo["district"]) == 13:
            for i in range(12):
                s = sum(mo["district"][d]["months"][i] or 0 for d in DISTRICTS)
                checks.append({"yearbook": y, "check": "districts_sum_to_company_month",
                               "month": i + 1, "sum": s, "printed": comp["months"][i],
                               "diff": s - comp["months"][i]})
        bd = t["by_district"]["values"]
        for r, row in bd.items():
            if int(r) >= 96 and "總計" in row:
                ds = [row[d] for d in DISTRICTS if d in row and row[d] is not None]
                if len(ds) >= 12:
                    checks.append({"yearbook": y, "check": "districts_sum_to_total_year",
                                   "roc": r, "sum": sum(ds), "printed": row["總計"],
                                   "diff": sum(ds) - row["總計"]})
        for d, rec in t["systems"]["districts"].items():
            s = sum(v["output"] for v in rec["systems"].values())
            checks.append({"yearbook": y, "check": "systems_sum_to_district", "row": d,
                           "sum": s, "printed": rec["total"], "diff": s - rec["total"]})
            m = mo["district"].get(d)
            if m:
                checks.append({"yearbook": y, "check": "table3_equals_table2_1", "row": d,
                               "sum": rec["total"], "printed": m["total"],
                               "diff": rec["total"] - m["total"]})
    for y, t in tab.items():
        for d, r in t.get("purchases", {}).get("districts", {}).items():
            if {"raw_clear", "taipei_clear", "irrigation_raw", "north_raw", "central_raw",
                    "south_raw", "other_raw"} <= set(r):
                parts = (r["taipei_clear"] + r["irrigation_raw"] + r["north_raw"]
                         + r["central_raw"] + r["south_raw"] + r["other_raw"])
                checks.append({"yearbook": y, "check": "purchases_sum_by_agency", "row": d,
                               "sum": parts, "printed": r["raw_clear"],
                               "diff": parts - r["raw_clear"]})
    # the same district-year printed in different yearbooks (表2(2))
    for d in TARGET:
        for r in range(101, 115):
            vals = {y: t["by_district"]["values"].get(str(r), {}).get(d)
                    for y, t in tab.items()}
            vals = {y: v for y, v in vals.items() if v is not None}
            if len(vals) >= 2:
                checks.append({"check": "cross_yearbook_same_value", "row": d, "roc": r,
                               "n_printings": len(vals),
                               "spread": max(vals.values()) - min(vals.values()),
                               "values": vals})
    # printed rounding (凡例 七): agency columns of 表4 are rounded to 1000 m3 each
    tol = {"purchases_sum_by_agency": 3.0}
    bad = [c for c in checks
           if abs(c.get("diff", c.get("spread", 0))) > tol.get(c["check"], 0.5)]
    return {"n_checks": len(checks), "n_failed": len(bad), "failed": bad, "checks": checks}


def main() -> None:
    tab = {}
    for y in YEARS:
        tab[y] = {"monthly": parse_monthly(y), "by_district": parse_by_district(y),
                  "systems": parse_systems(y), "sources": parse_sources(y),
                  "purchases": parse_purchases(y)}
    val = validate(tab)
    K.EVID.mkdir(parents=True, exist_ok=True)
    (K.EVID / "twc_tables.json").write_text(
        json.dumps({"tables": {str(k): v for k, v in tab.items()}, "validation": val},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    for y, t in tab.items():
        d11 = t["monthly"]["district"].get("第十一區管理處", {})
        d5 = t["monthly"]["district"].get("第五區管理處", {})
        srcs = {k: (v["name"], v["groundwater_only"]) for k, v in t["sources"]["systems"].items()}
        print(y, "d11", d11.get("total"), "d5", d5.get("total"), "systems",
              {k: len(v["systems"]) for k, v in t["systems"]["districts"].items()}, srcs)
    print("checks", val["n_checks"], "failed", val["n_failed"])
    for c in val["failed"][:30]:
        print(" FAIL", {k: v for k, v in c.items() if k != "values"}, c.get("values", ""))


if __name__ == "__main__":
    main()
