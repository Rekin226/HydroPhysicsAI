"""Daily mean discharge of the Choushui (basin 1510) gauges from the WRA Hydrological Year
Book, Part 2 (river discharge and stage), 2012-2025.

Adapted from the 2026-10-12 scout's ``parse_hyb.py`` (results/dataprep/_pilots/
surface_water_monthly/), re-validated here: the text comes from ``pdftotext -layout`` run on
the L0 PDFs (data_fetch_api/new_measurements/wra_hydro_yearbook/), a page counts only when it
carries "DAILY MEAN DISCHARGE" and a ``1510 H...`` header line; month columns are located
from the day rows that carry all 12 values. Added checks: the printed monthly mean
(月平均) and the printed annual mean discharge (平均流量 of the annual summary) are parsed
and compared with the means of the parsed days; a station-year parsed from two pages is
reported, never averaged silently.

The PDFs are untrusted data: they are only read by pdftotext into a temporary directory.
Licence (plan 2.6): yearbook copying is disallowed; daily extractions stay under results/.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from w1_canal.config import BASIN, YEARBOOK_DIR, YEARBOOK_YEARS

NUM = re.compile(r"-?\d[\d,]*\.\d+|-?\d+")
HEAD = re.compile(r"\s*" + BASIN + r"\s+(H\w+)\s+\S+\s+(\S+)\s+(\S+)\s*(\d[\d,]*\.\d+)?\s*$")


def yearbook_pdf(year: int) -> Path:
    d = YEARBOOK_DIR / f"hyb{year}"
    pdfs = sorted(d.glob("*.pdf"))
    if len(pdfs) != 1:
        raise FileNotFoundError(f"expected one yearbook PDF in {d}, found {len(pdfs)}")
    return pdfs[0]


def pdf_text(pdf: Path, workdir: Path) -> str:
    out = workdir / (pdf.stem + ".txt")
    subprocess.run(["pdftotext", "-layout", str(pdf), str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out.read_text(encoding="utf-8", errors="replace")


def column_ends(day_rows: list[str]):
    """Median end position of the 12 month columns, from day rows carrying all 12 values."""
    ends = []
    for line in day_rows:
        toks = [m.end() for m in re.finditer(r"\S+", line)]
        if len(toks) == 13:
            ends.append(toks[1:])
    if not ends:
        return None
    return np.median(np.array(ends, dtype=float), axis=0)


def assign(line: str, ends: np.ndarray, tol: int = 6) -> list[float]:
    vals = [np.nan] * 12
    for m in re.finditer(r"\S+", line):
        try:
            v = float(m.group().replace(",", ""))
        except ValueError:
            continue
        j = int(np.argmin(np.abs(ends - m.end())))
        if abs(ends[j] - m.end()) <= tol:
            vals[j] = v
    return vals


def parse_page(page: str, year: int):
    """(daily records, printed monthly means, annual summary) or None."""
    if "DAILY MEAN DISCHARGE" not in page:
        return None
    lines = page.splitlines()
    head = None
    for i, ln in enumerate(lines):
        m = HEAD.match(ln)
        if m:
            head = (i, m.group(1), m.group(2), m.group(3), m.group(4))
            break
    if head is None:
        return None
    i0, sno, trib, name, area = head
    if area is None:
        for ln in lines[i0 + 1:i0 + 3]:
            mm = re.match(r"\s*(\d[\d,]*\.\d+)\s*$", ln)
            if mm:
                area = mm.group(1)
                break
    area = float(area.replace(",", "")) if area else np.nan
    day_rows: dict[int, str] = {}
    k_month = None
    for k, ln in enumerate(lines):
        if k_month is None and re.search(r"Month\s+月", ln):
            k_month = k
        if k_month is None:
            continue
        m = re.match(r"\s{0,8}(\d{1,2})\s+[-\d.*]", ln)   # '*' = not printed
        if m and 1 <= int(m.group(1)) <= 31 and int(m.group(1)) not in day_rows:
            day_rows[int(m.group(1))] = ln
        if "上旬" in ln:
            break
    ends = column_ends(list(day_rows.values()))
    if ends is None:
        return None
    recs = []
    for d, ln in day_rows.items():
        m = re.match(r"(\s*\d{1,2})", ln)
        body = " " * len(m.group(1)) + ln[len(m.group(1)):]
        for j, v in enumerate(assign(body, ends)):
            recs.append((year, sno, name, trib, area, j + 1, d, v))
    mon = [np.nan] * 12
    for k, ln in enumerate(lines):
        if ln.strip().startswith("月平均"):
            nxt = ln[ln.find("月平均") + 3:]
            if not NUM.search(nxt):
                nxt = lines[k + 1] if k + 1 < len(lines) else ""
            mon = assign(nxt, ends, tol=8)
            break
    ann_mean = ann_runoff = np.nan
    for k, ln in enumerate(lines):
        if "Average Discharge" in ln:
            for nxt in lines[k + 1:k + 5]:
                toks = nxt.split()
                if len(toks) >= 2:          # first value line; '*' = not printed
                    ann_mean = _num(toks[0])
                    ann_runoff = _num(toks[1])
                    break
            break
    return recs, (year, sno, name, area, mon), (year, sno, ann_mean, ann_runoff)


def parse_years(years=YEARBOOK_YEARS, log=print):
    """Parse every yearbook. Returns (daily, printed_monthly, annual_summary, pdf_inputs)."""
    allrecs, allmon, allann, inputs = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="w1_hyb_") as tmp:
        for y in years:
            pdf = yearbook_pdf(y)
            inputs.append(pdf)
            txt = pdf_text(pdf, Path(tmp))
            n_pages = 0
            for page_no, page in enumerate(txt.split("\f")):
                r = parse_page(page, y)
                if r:
                    n_pages += 1
                    allrecs += [rec + (page_no,) for rec in r[0]]
                    allmon.append(r[1] + (page_no,))
                    allann.append(r[2] + (page_no,))
            log(f"  {y}: {n_pages} daily-discharge pages for basin {BASIN}")
    df = pd.DataFrame(allrecs, columns=["year", "station_no", "station", "tributary",
                                        "area_km2", "month", "day", "q_cms", "page"])
    ok = np.array([_valid_date(y, m, d) for y, m, d in zip(df.year, df.month, df.day, strict=True)])
    df = df[ok].reset_index(drop=True)
    mon = pd.DataFrame([(y, s, n, a, m + 1, v, p) for y, s, n, a, vals, p in allmon
                        for m, v in enumerate(vals)],
                       columns=["year", "station_no", "station", "area_km2", "month",
                                "q_printed_cms", "page"])
    ann = pd.DataFrame(allann, columns=["year", "station_no", "q_annual_printed_cms",
                                        "runoff_printed_cms_day", "page"])
    return df, mon, ann, inputs


def _num(tok: str) -> float:
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return float("nan")


def _valid_date(y, m, d) -> bool:
    try:
        pd.Timestamp(year=int(y), month=int(m), day=int(d))
        return True
    except ValueError:
        return False


def parse_qc(df: pd.DataFrame, mon: pd.DataFrame, ann: pd.DataFrame, tol: float) -> dict:
    """Checks of the parse against the yearbook's own printed aggregates."""
    dup = df.groupby(["station_no", "year"]).page.nunique()
    dup = dup[dup > 1]
    # keep the first page of a duplicated station-year (reported)
    first = df.groupby(["station_no", "year"]).page.transform("min")
    d1 = df[df.page == first]
    m1 = mon[mon.page == mon.groupby(["station_no", "year"]).page.transform("min")]
    a1 = ann[ann.page == ann.groupby(["station_no", "year"]).page.transform("min")]
    cm = d1.groupby(["station_no", "year", "month"]).q_cms.agg(qmean="mean", n="count").reset_index()
    j = cm.merge(m1[["station_no", "year", "month", "q_printed_cms"]],
                 on=["station_no", "year", "month"], how="left")
    j["rel_diff"] = (j["qmean"] - j.q_printed_cms).abs() / j.q_printed_cms.abs().clip(lower=1e-6)
    off = j[(j.rel_diff > tol) & j.q_printed_cms.notna()]
    ca = d1.groupby(["station_no", "year"]).q_cms.agg(qmean="mean", n="count").reset_index()
    ja = ca.merge(a1[["station_no", "year", "q_annual_printed_cms"]], on=["station_no", "year"],
                  how="left")
    ja["rel_diff"] = (ja["qmean"] - ja.q_annual_printed_cms).abs() / \
        ja.q_annual_printed_cms.abs().clip(lower=1e-6)
    ndays = d1.groupby(["station_no", "year"]).q_cms.apply(lambda s: int(s.notna().sum()))
    exp_days = {y: 366 if pd.Timestamp(year=y, month=12, day=31).dayofyear == 366 else 365
                for y in d1.year.unique()}
    missing = {f"{s}_{y}": int(exp_days[y] - n) for (s, y), n in ndays.items()
               if exp_days[y] - n > 0}
    return {
        "station_years": int(len(ndays)),
        "duplicate_station_years": {f"{s}_{y}": int(n) for (s, y), n in dup.items()},
        "missing_days": missing,
        "printed_monthly_compared": int(j.q_printed_cms.notna().sum()),
        "printed_monthly_off_gt_tol": int(len(off)),
        "printed_monthly_off_list": [
            {"station": r.station_no, "year": int(r.year), "month": int(r.month),
             "daily_mean": round(float(r.qmean), 3), "printed": float(r.q_printed_cms),
             "rel_diff": round(float(r.rel_diff), 4), "n_days": int(r.n)}
            for r in off.itertuples()],
        "printed_annual_compared": int(ja.q_annual_printed_cms.notna().sum()),
        "printed_annual_max_rel_diff": float(ja.rel_diff.max()),
        "printed_annual_off_gt_1pct": [
            {"station": r.station_no, "year": int(r.year), "daily_mean": round(float(r.qmean), 3),
             "printed": float(r.q_annual_printed_cms), "rel_diff": round(float(r.rel_diff), 4)}
            for r in ja[ja.rel_diff > 0.01].itertuples()],
    }, d1.drop(columns="page").reset_index(drop=True)
