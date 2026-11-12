"""Parse WRA hydrological yearbook groundwater volumes (自記地下水位年報表 pages).

Input PDFs are untrusted data: they are only converted with `pdftotext -bbox` and the
word boxes are read with regular expressions.
Usage: python -I parse_yearbook.py 2001 2002 ...
Writes tmp/parsed/pages_{year}.parquet (one row per station-month) for every year.
"""
from __future__ import annotations

import os  # noqa
# ruff: noqa
import html, os, re, subprocess, sys, unicodedata
import numpy as np
import pandas as pd

ROOT = os.environ.get("HYDRO_YEARBOOK_ROOT", "data_fetch_api/new_measurements/wra_hydro_yearbook_gw")
WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')
MONTHS_EN = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
MONTHS_ZH = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"]
NUM = re.compile(r"^(-?\d+\.\d+)([*＊@＠]*)$")
MISSING = re.compile(r"^[*＊]{2,}$")


def pages(year: int):
    bb = f"{ROOT}/tmp/bbox/gw_{year}.html"
    if not os.path.exists(bb):
        os.makedirs(os.path.dirname(bb), exist_ok=True)
        subprocess.run(["pdftotext", "-bbox", f"{ROOT}/raw/gw_{year}.pdf", bb], check=True,
                       stderr=subprocess.DEVNULL)
    cur, n = None, 0
    with open(bb, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "<page " in line:
                n += 1; cur = []
            elif "</page>" in line:
                yield n, cur; cur = None
            elif cur is not None:
                m = WORD.search(line)
                if m:
                    x0, y0, x1, y1 = map(float, m.groups()[:4])
                    cur.append((x0, y0, x1, y1, unicodedata.normalize('NFKC', html.unescape(m.group(5))).strip()))


def parse_page(words, year):
    srt = sorted(words, key=lambda w: (round(w[1] / 4), w[0]))
    txt = "".join(w[4] for w in srt).replace(" ", "")
    if "自記地下水位年報表" not in txt or "月平均" not in txt:
        return None
    # --- column anchors
    def anchors(labels):
        out = {}
        for i, lab in enumerate(labels):
            c = [w for w in words if w[4].replace(" ", "") == lab and w[1] < 260]
            if c:
                w = min(c, key=lambda w: w[1]); out[i] = ((w[0] + w[2]) / 2, w[1])
        return out
    an = anchors(MONTHS_EN)
    if len(an) < 12:
        an2 = anchors(MONTHS_ZH)
        if len(an2) > len(an): an = an2
    if len(an) < 10:
        return {"error": f"month header anchors {len(an)}"}
    xs = np.array([an[i][0] if i in an else np.nan for i in range(12)])
    if np.isnan(xs).any():  # interpolate missing anchors linearly
        idx = np.arange(12); ok = ~np.isnan(xs)
        xs = np.interp(idx, idx[ok], xs[ok])
    hdr_y = max(v[1] for v in an.values())
    pitch = float(np.median(np.diff(xs)))
    x_left = xs[0] - 0.75 * pitch
    # --- row labels
    mm = [w for w in words if w[4].replace(" ", "") == "月平均" and w[1] > hdr_y]
    if not mm: return {"error": "no monthly-mean row"}
    mrow = min(mm, key=lambda w: w[1]); my = (mrow[1] + mrow[3]) / 2
    rows = {"M": my}
    for w in words:
        if re.fullmatch(r"\d{1,2}", w[4]) and w[2] < x_left and hdr_y < w[1] < mrow[1] and 1 <= int(w[4]) <= 31:
            rows.setdefault(int(w[4]), (w[1] + w[3]) / 2)
    days = sorted(k for k in rows if k != "M")
    if len(days) < 28: return {"error": f"day rows {len(days)}"}
    rpitch = float(np.median(np.diff([rows[d] for d in days])))
    keys = list(rows); ry = np.array([rows[k] for k in keys])
    daily = {m: {} for m in range(12)}; mon = {}; monflag = {}
    for w in words:
        if w[0] < x_left - 5 or w[1] <= hdr_y or (w[1] + w[3]) / 2 > my + 0.5 * rpitch: continue
        s = w[4].replace(" ", "")
        m = NUM.match(s); miss = MISSING.match(s)
        if not m and not miss: continue
        yc = (w[1] + w[3]) / 2; j = int(np.argmin(np.abs(ry - yc)))
        if abs(ry[j] - yc) > 0.55 * rpitch: continue
        xc = (w[0] + w[2]) / 2; c = int(np.argmin(np.abs(xs - xc)))
        if abs(xs[c] - xc) > 0.6 * pitch: continue
        k = keys[j]
        if k == "M":
            if c in mon or c in monflag: return {"error": f"duplicate monthly cell col {c}"}
            if m: mon[c] = float(m.group(1)); monflag[c] = m.group(2).replace("＊", "*").replace("＠", "@")
            else: monflag[c] = "***"
        elif m:
            if k in daily[c]: return {"error": f"duplicate daily cell {k}/{c}"}
            daily[c][k] = (float(m.group(1)), m.group(2))
    # --- header: station id, name, well top
    ids = [w for w in words if re.fullmatch(r"(?=(?:.*\d){5})[0-9A-Za-z]{6,10}", w[4]) and w[1] < hdr_y - 12]
    if len(ids) != 1: return {"error": f"station id candidates {len(ids)}"}
    idw = ids[0]; iy = (idw[1] + idw[3]) / 2
    line = sorted([w for w in words if abs((w[1] + w[3]) / 2 - iy) < 5 and w[0] > idw[2]], key=lambda w: w[0])
    name, nums, rest = [], [], []
    for w in line:
        if re.fullmatch(r"-?\d+(\.\d+)?", w[4]) and (name or nums): nums.append(float(w[4]))
        elif not nums: name.append(w[4])
        else: rest.append(w[4])
    basin = "".join(w[4] for w in sorted(words, key=lambda w: (w[1], w[0]))
                    if w[2] < idw[0] and idw[1] - 25 < w[1] < hdr_y - 5 and re.search(r"[一-鿿]", w[4])
                    and "地下水分區" not in w[4] and len(w[4]) > 2)
    ty = re.search(r"民\s*國\s*(\d+)\s*年", "".join(w[4] for w in srt[:30]))
    out = []
    for c in range(12):
        dv = [v for v, f in daily[c].values()]
        out.append(dict(
            yb_station_id=idw[4], yb_name="".join(name).replace(" ", ""), basin=basin,
            well_top_m=nums[0] if nums else np.nan, well_depth_m=nums[1] if len(nums) > 1 else np.nan,
            agency="".join(rest), title_roc_year=int(ty.group(1)) if ty else -1,
            year=year, month=c + 1, value_m=mon.get(c, np.nan), yb_mark=monflag.get(c, "blank"),
            n_days=len(dv), daily_mean_m=float(np.mean(dv)) if dv else np.nan,
            n_days_filled=sum(1 for v, f in daily[c].values() if "@" in f or "＠" in f)))
    return out


def main():
    os.makedirs(f"{ROOT}/tmp/parsed", exist_ok=True)
    for year in map(int, sys.argv[1:]):
        recs, errs, ntab = [], [], 0
        for n, words in pages(year):
            r = parse_page(words, year)
            if r is None: continue
            ntab += 1
            if isinstance(r, dict): errs.append((n, r["error"])); continue
            for d in r: d["page"] = n
            recs.extend(r)
        df = pd.DataFrame(recs)
        df.to_parquet(f"{ROOT}/tmp/parsed/pages_{year}.parquet")
        pd.DataFrame(errs, columns=["page", "error"]).to_csv(f"{ROOT}/tmp/parsed/errors_{year}.csv", index=False)
        bad = (df.title_roc_year != year - 1911).sum() // 12 if len(df) else 0
        print(year, "table pages", ntab, "parsed", len(df) // 12, "errors", len(errs), "title-year mismatch pages", bad,
              "stations", df.yb_station_id.nunique() if len(df) else 0, errs[:5], flush=True)


if __name__ == "__main__":
    main()
