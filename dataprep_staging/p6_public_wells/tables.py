"""Read and verify the MOEA public-well tables (check T1 of ``P6.inventory_member.v1``).

* :func:`table_2_2`: phase-2 main plan, annex table 2-2 (附表2-2, p. 99): public wells
  disposed in ROC 100-108 by agency column and year, counts and abstraction cuts (萬噸/年).
  Two API reads (``transcribe_p6.py``) are compared cell by cell; each row's numbers must
  equal the numbers of the same row in the text layer, in order; rows and columns must add up
  to the printed totals.
* :func:`table_2_5`: annex table 2-5 (p. 101), irrigation-association and Taiwan Sugar wells
  by year (two reads; merged cells of rows 100-102 compared as a group).
* :func:`table_4_5`: table 4-5 (p. 47), disposal types (reduce/stop/fill) actual 100-108 and
  planned 109-113, parsed from the text layer.
* :func:`table_2_3`: annex table 2-3 (p. 100), targets and execution to 108 by agency.

Nothing here is computed from a value the API produced alone: every accepted number is
printed in the PDF's own text layer.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p6_public_wells import p6const as K  # noqa: E402
from p6_public_wells import p6lib as L  # noqa: E402
from p6_public_wells import transcribe_p6 as TR  # noqa: E402

DASHES = {"─", "—", "–", "-", "－", "―"}
VOL_TOL = 0.15          # printed rounding of volume sums (10^4 m3/yr), declared in T1


def norm_cell(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return "-" if s in DASHES else s


def cell_value(s: str) -> tuple[float | None, str | None]:
    """(number, note) of a printed cell: '' or a dash -> (None, None); a parenthesised note or
    text -> (None, text); otherwise the number (thousands separators removed)."""
    s = norm_cell(s).rstrip("*")
    if s in ("", "-"):
        return None, None
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", s):
        return float(s.replace(",", "")), None
    return None, s


def _reads(key: str) -> tuple[list[dict], list[dict]]:
    c = TR.load_cached(key)
    return c["pdf_read"]["result"]["tables"], c["img_read"]["result"]["tables"]


def _compare_reads(ta: dict, tb: dict) -> list[dict]:
    """Cells where the two reads differ (after NFKC and dash normalisation)."""
    out = []
    if [r["row_label"] for r in ta["rows"]] != [r["row_label"] for r in tb["rows"]]:
        out.append({"what": "row labels differ"})
    for ra, rb in zip(ta["rows"], tb["rows"], strict=False):
        if len(ra["cells"]) != len(rb["cells"]):
            out.append({"row": ra["row_label"], "what": "cell count differs"})
            continue
        for j, (a, b) in enumerate(zip(ra["cells"], rb["cells"], strict=True)):
            if norm_cell(a) != norm_cell(b):
                out.append({"row": ra["row_label"], "col": j, "pdf_read": a, "img_read": b})
    return out


def _text_rows(src: str, page: int, start_quote: str, labels: list[str],
               stop_quote: str | None = None) -> dict[str, list[float]]:
    """Numbers of each labelled row in the text layer, scanning sequentially from the line
    holding ``start_quote``. A label line without numbers takes the next line with numbers
    (two-line labels such as '108年 / 目標值')."""
    lines = L.page_text(src, page).splitlines()
    sq = L.squash(start_quote)
    i0 = next(i for i, ln in enumerate(lines) if sq in L.squash(ln))
    i1 = len(lines)
    if stop_quote:
        st = L.squash(stop_quote)
        i1 = next((i for i in range(i0 + 1, len(lines)) if st in L.squash(lines[i])), i1)
    out, i = {}, i0 + 1
    for lab in labels:
        head = re.sub(r"目標值$", "", L.squash(lab)) or L.squash(lab)
        while i < i1 and not L.squash(lines[i]).startswith(head):
            i += 1
        if i >= i1:
            raise ValueError(f"row {lab!r} not found in the text layer of {src} p. {page}")
        raw = unicodedata.normalize("NFKC", lines[i])
        if head not in raw:
            raise ValueError(f"row head {head!r} is not contiguous in {raw!r}")
        nums = L.numbers(raw.replace(head, " ", 1))
        j = i
        while not nums and j + 1 < i1:
            j += 1
            nums = L.numbers(lines[j])
        out[lab] = nums
        i = j + 1
    return out


def _row_numbers(cells: list[str]) -> tuple[list[float], list[str]]:
    vals, notes = [], []
    for c in cells:
        v, note = cell_value(c)
        if v is not None:
            vals.append(v)
        if note:
            notes.append(note)
    return vals, notes


# ------------------------------------------------------------------ 附表2-2
COUNT_TITLE = "附表2-2第一期計畫100~108年水井處置統計(處置口數)"
VOL_TITLE = "附表2-2(續)第一期計畫100~108年公有水井處置統計(可減抽水量)"
BODY = [f"{y}年" for y in K.YEARS_ROC]
TAIL = ["合計", "108年目標值", "全程目標值"]


def _table_from_reads(ta: dict, tb: dict, text: dict[str, list[float]], kind: str) -> dict:
    ncol = len(K.COHORT_COLUMNS)
    diffs = _compare_reads(ta, tb)
    rows, problems = {}, []
    for r in ta["rows"]:
        lab = unicodedata.normalize("NFKC", r["row_label"]).replace(" ", "")
        cells = r["cells"]
        if len(cells) != ncol + 1:
            problems.append({"row": lab, "what": f"{len(cells)} cells, expected {ncol + 1}"})
            continue
        vals, notes = _row_numbers(cells)
        tl = text.get(lab)
        if tl is None:
            problems.append({"row": lab, "what": "row not in the text layer"})
        elif [round(x, 4) for x in vals] != [round(x, 4) for x in tl]:
            problems.append({"row": lab, "what": "numbers differ from the text layer",
                             "reads": vals, "text_layer": tl})
        parsed = [cell_value(c) for c in cells]
        rows[lab] = {"values": [p[0] for p in parsed[:ncol]], "total": parsed[ncol][0],
                     "notes": {K.COHORT_COLUMNS[j]["key"]: parsed[j][1]
                               for j in range(ncol) if parsed[j][1]}}
    # arithmetic
    tol = 0.0 if kind == "count" else VOL_TOL
    arith = []
    for lab in BODY:
        if lab not in rows:
            continue
        s = sum(v or 0.0 for v in rows[lab]["values"])
        t = rows[lab]["total"]
        if t is None or abs(s - t) > tol + 1e-9:
            arith.append({"row": lab, "cells_sum": round(s, 4), "printed_total": t,
                          "notes": rows[lab]["notes"]})
    col_arith = []
    tot = rows.get("合計")
    if tot:
        for j, col in enumerate(K.COHORT_COLUMNS):
            s = sum(rows[lab]["values"][j] or 0.0 for lab in BODY if lab in rows)
            t = tot["values"][j]
            if t is None or abs(s - t) > tol + 1e-9:
                col_arith.append({"column": col["key"], "cells_sum": round(s, 4),
                                  "printed_total": t})
        s = sum(rows[lab]["total"] or 0.0 for lab in BODY if lab in rows)
        if abs(s - (tot["total"] or 0.0)) > tol + 1e-9:
            col_arith.append({"column": "合計 (row totals)", "cells_sum": round(s, 4),
                              "printed_total": tot["total"]})
        s = sum(v or 0.0 for v in tot["values"])
        if abs(s - (tot["total"] or 0.0)) > tol + 1e-9:
            col_arith.append({"column": "合計 (column totals)", "cells_sum": round(s, 4),
                              "printed_total": tot["total"]})
    return {"rows": rows, "read_differences": diffs, "text_layer_problems": problems,
            "row_arithmetic_exceptions": arith, "column_arithmetic_exceptions": col_arith,
            "titles": [ta["title"], tb["title"]],
            "headers_pdf_read": ta["column_headers"]}


def table_2_2() -> dict:
    src, page = K.PAGES["p2_at2_2"]
    a, b = _reads("p2_at2_2")
    if len(a) != 2 or len(b) != 2:
        raise ValueError("annex table 2-2: expected two tables per read")
    labels = BODY + TAIL
    tc = _text_rows(src, page, COUNT_TITLE, labels, stop_quote="附表2-2(續)")
    tv = _text_rows(src, page, VOL_TITLE, labels)
    counts = _table_from_reads(a[0], b[0], tc, "count")
    vols = _table_from_reads(a[1], b[1], tv, "volume")
    hdr_ok = all(col["label"] in L.squash(h).split("|")[-1] or
                 L.squash(col["label"]) in L.squash(h)
                 for col, h in zip(K.COHORT_COLUMNS, a[0]["column_headers"], strict=False))
    return {"src": src, "page": page, "counts": counts, "volumes": vols,
            "header_order_ok": bool(hdr_ok)}


# ------------------------------------------------------------------ 附表2-5
def table_2_5() -> dict:
    src, page = K.PAGES["p2_at2_5"]
    a, b = _reads("p2_at2_5")
    ia = [t for t in a if "2-5" in t["title"]][0]
    ib = [t for t in b if "2-5" in t["title"]][0]
    diffs = _compare_reads(ia, ib)
    lines = L.page_text(src, page).splitlines()
    i0 = next(i for i, ln in enumerate(lines) if "附表2-5" in L.squash(ln))
    body = [ln for ln in lines[i0 + 1:] if re.match(r"^\s*(10\d|合計)\s", ln)]
    text_nums = {}
    for ln in body:
        lab = ln.split()[0]
        text_nums[lab] = L.numbers(ln.split(lab, 1)[1])
    rows = {}
    for r in ia["rows"]:
        lab = unicodedata.normalize("NFKC", r["row_label"]).strip()
        rows[lab] = [cell_value(c)[0] for c in r["cells"]]
    groups = [["100", "101", "102"]] + [[y] for y in ("103", "104", "105", "106", "107",
                                                         "108", "合計")]
    problems = []
    for g in groups:
        got = sorted(round(v, 4) for lab in g for v in rows.get(lab, []) if v is not None)
        want = sorted(round(v, 4) for lab in g for v in text_nums.get(lab, []))
        if got != want:
            problems.append({"rows": g, "reads": got, "text_layer": want})
    return {"src": src, "page": page, "rows": rows, "columns": ia["column_headers"],
            "read_differences": diffs, "text_layer_problems": problems,
            "merged_groups": [g for g in groups if len(g) > 1]}


# ------------------------------------------------------------------ 表4-5 (text layer)
T45_BLOCKS = [("taiwan_water", "changhua"), ("taiwan_water", "yunlin"),
              ("taiwan_sugar", "changhua"), ("taiwan_sugar", "yunlin"),
              ("irrigation_association", "changhua"), ("irrigation_association", "yunlin"),
              ("schools_factories_prison", "changhua"), ("schools_factories_prison", "yunlin"),
              ("total", "both")]
TYPES = {"減抽": "reduce", "停用": "stop", "填塞": "fill"}


def table_4_5() -> dict:
    src, page = K.PAGES["p2_t4_5"]
    lines = L.page_text(src, page).splitlines()
    i0 = next(i for i, ln in enumerate(lines) if "表4-5" in L.squash(ln))
    recs, k = [], 0
    for ln in lines[i0 + 1:]:
        types = re.findall(r"減抽|停用|填塞", ln)
        if not types:
            continue
        nums = [int(x) for x in L.numbers(ln)]
        blk = T45_BLOCKS[k // 3]
        t = TYPES[types[0]]
        actual = nums[0]
        planned = nums[1] if len(types) == 3 else 0
        type_total = nums[2] if len(types) == 3 else nums[1]
        group_total = nums[len(types)] if len(nums) > len(types) else None
        recs.append({"agency": blk[0], "county": blk[1], "type": t, "actual_100_108": actual,
                     "planned_109_113": planned, "type_total": type_total,
                     "group_total": group_total, "line": ln.strip()})
        k += 1
        if k == 27:
            break
    if k != 27:
        raise ValueError(f"table 4-5: parsed {k} type lines, expected 27")
    checks = []
    for b in T45_BLOCKS:
        rs = [r for r in recs if (r["agency"], r["county"]) == b]
        gt = [r["group_total"] for r in rs if r["group_total"] is not None]
        s = sum(r["type_total"] for r in rs)
        ok_types = all(r["actual_100_108"] + r["planned_109_113"] == r["type_total"]
                       for r in rs)
        checks.append({"block": "/".join(b), "type_totals_sum": s,
                       "printed_group_total": gt[0] if gt else None,
                       "ok": ok_types and gt and gt[0] == s})
    for t in TYPES.values():
        parts = sum(r["actual_100_108"] for r in recs if r["type"] == t and r["agency"] != "total")
        tot = [r["actual_100_108"] for r in recs if r["type"] == t and r["agency"] == "total"][0]
        checks.append({"block": f"total actual {t}", "sum": parts, "printed": tot,
                       "ok": parts == tot})
    return {"src": src, "page": page, "rows": recs, "checks": checks,
            "all_ok": all(c["ok"] for c in checks)}


# ------------------------------------------------------------------ 附表2-3 (text layer)
T23_ORDER = ["tw_changhua", "ia_changhua", "ts_changhua", "factory_changhua",
             "school_changhua", "tw_yunlin", "ia_yunlin", "ts_yunlin", "factory_yunlin",
             "school_yunlin", "prison_yunlin", "total"]


def table_2_3() -> dict:
    src, page = K.PAGES["p2_at2_3"]
    lines = L.page_text(src, page).splitlines()
    out = {}
    for part, title in (("count", "附表2-3第一期計畫100~108年水井處置達成率統計(處置口數)"),
                        ("volume", "附表2-3(續)第一期計畫100~108年水井處置達成率統計(可減抽水量)")):
        i0 = next(i for i, ln in enumerate(lines) if L.squash(title) in L.squash(ln))
        rows = []
        for ln in lines[i0 + 1:]:
            if not re.search(r"(公司|水利會|工廠|小學|監獄|合計)", ln):
                continue
            nums = L.numbers(ln.replace("(11區)", "").replace("(5區)", ""))
            if len(nums) < 3:
                continue
            rows.append({"full_target": nums[0], "target_to_108": nums[1],
                         "executed_to_108": nums[2], "line": ln.strip()})
            if len(rows) == len(T23_ORDER):
                break
        if len(rows) != len(T23_ORDER):
            raise ValueError(f"table 2-3 {part}: parsed {len(rows)} rows")
        out[part] = dict(zip(T23_ORDER, rows, strict=True))
    return {"src": src, "page": page, **out}


# ------------------------------------------------------------------ 附表2-9 (cross-check)
def table_2_9_check(t22: dict) -> dict:
    """Taiwan Water disposals by year in annex table 2-9 (p. 104) against annex table 2-2: the
    year's Taiwan Water counts and combined volume must be printed in the 2-9 row (R2)."""
    src = "plan_phase2"
    page = L.find_page(src, "附表2-9第一期計畫100~108年公共用水減抽地下水達成率統計")
    lines = L.page_text(src, page).splitlines()
    i0 = next(i for i, ln in enumerate(lines) if "附表2-9" in L.squash(ln))
    text = {}
    for ln in lines[i0 + 1:]:
        m = re.match(r"^\s*(10\d|合計)\s", ln)
        if m:
            text[m.group(1)] = L.numbers(ln.split(m.group(1), 1)[1])
    cr, vr = t22["counts"]["rows"], t22["volumes"]["rows"]
    keys = [c["key"] for c in K.COHORT_COLUMNS]
    jc, jy = keys.index("tw_changhua"), keys.index("tw_yunlin")
    out, ok = [], True
    for roc in K.YEARS_ROC:
        lab = f"{roc}年"
        nc, ny = cr[lab]["values"][jc] or 0, cr[lab]["values"][jy] or 0
        v = (vr[lab]["values"][jc] or 0) + (vr[lab]["values"][jy] or 0)
        nums = [round(x, 4) for x in text.get(str(roc), [])]
        need = [x for x in (nc, ny) if x] + ([round(v, 4)] if v else [])
        hit = all(round(x, 4) in nums for x in need)
        ok &= hit
        out.append({"year_roc": roc, "tw_changhua_n": nc, "tw_yunlin_n": ny,
                    "tw_volume": round(v, 4), "printed_in_2_9_row": hit})
    tot = text.get("合計", [])
    return {"page": page, "rows": out, "all_found": bool(ok), "total_row": tot}
