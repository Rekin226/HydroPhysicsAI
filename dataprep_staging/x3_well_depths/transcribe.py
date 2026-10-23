"""Transcribe the public well-depth evidence from the L0 PDFs into tables (no hand typing).

Every value is parsed from poppler's text layer of the downloaded PDF (untrusted data, read
only), and checked against the printed totals:

* ``township_layer_split.csv`` - Lin (2012, NTU thesis), Table 4-10 (private-well pumping
  share of each township and its split over aquifers 1-4, set from the 2000-2008 YunTech
  well census) and Table 4-14 (the same after calibration of a MODFLOW model on 2005-2009
  heads). Aquifers are the CGS hydrogeological units, i.e. twin layers 0-3.
* ``township_pumping_by_use.csv`` - Lin (2012) Table 2-2: census pumping (t/yr) by use and
  township (YunTech / ARDC census 2000-2008).
* ``township_wells_deep.csv`` - WRA well-management report (2013, marc_id 74288) Tables
  2-1.3 and 2-1.4: wells found by the latest county survey of each township (survey year
  89-99) and the number deeper than 100 m, by use.
* ``ia_wells_depth_classes.csv`` - MOA agricultural statistics yearbook, table "Ground-water
  development - number of water supply wells": irrigation-association wells by depth class,
  Changhua and Yunlin associations, yearbooks 96-103.

Usage: python -I transcribe.py <out_dir>
"""
from __future__ import annotations

import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x3const as K  # noqa: E402
from pdftext import pdf_text  # noqa: E402

NUM = r"-?\d+(?:\.\d+)?"


def _text(pdf: Path, first: int | None = None, last: int | None = None) -> str:
    """Text layer, NFKC-normalised (some pages print CJK compatibility ideographs)."""
    with tempfile.TemporaryDirectory() as td:
        return unicodedata.normalize("NFKC", pdf_text(pdf, Path(td) / "t.txt", first, last))


def _between(txt: str, start: str, end: str, must: str | None = None) -> str:
    """Text from the first occurrence of ``start`` (whose slice contains ``must``, which
    skips tables-of-contents entries) to the next ``end``."""
    start, end = unicodedata.normalize("NFKC", start), unicodedata.normalize("NFKC", end)
    must = unicodedata.normalize("NFKC", must) if must else must
    i = -1
    while True:
        i = txt.index(start, i + 1)
        j = txt.index(end, i)
        if must is None or must in txt[i:j]:
            return txt[i:j]


# ----------------------------------------------------------------------------- Lin (2012)
def lin_split_tables(txt: str) -> pd.DataFrame:
    """Tables 4-10 (initial, from the well census) and 4-14 (calibrated)."""
    rows = []
    blocks = {
        "census_initial": _between(txt, "表 4-10 鄉鎮水井抽水空間分佈表", "第四節", "二崙"),
        "calibrated": _between(txt, "表 4-14 率定後各鄉鎮市民井抽水量之空間分布",
                               "由於第一、二含水層觀測井數", "二崙"),
    }
    pat = re.compile(rf"^\s*(\S{{2,3}})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$")
    for variant, blk in blocks.items():
        for ln in blk.splitlines():
            m = pat.match(ln)
            if not m:
                continue
            town = K.town_key(m.group(1))
            v = [float(m.group(i)) for i in range(2, 7)]
            rows.append({"variant": variant, "town": town,
                         "county": K.county_of_town(town) or ("chiayi" if town == "嘉義"
                                                              else None),
                         "town_pump_share": v[0], "a1": v[1], "a2": v[2], "a3": v[3],
                         "a4": v[4]})
    df = pd.DataFrame(rows)
    df["sum_a"] = df[["a1", "a2", "a3", "a4"]].sum(axis=1)
    return df


def lin_pumping_by_use(txt: str) -> pd.DataFrame:
    blk = _between(txt, "表 2-2 彰化及雲林各鄉鎮地下抽水井普查結果", "第三章", "二崙")
    val = r"(-|[\d,]+\.\d+)"
    pat = re.compile(rf"^\s*(\S{{2,3}})\s+{val}\s+{val}\s+{val}\s+{val}\s+{val}\s*$")
    rows = []
    for ln in blk.splitlines():
        m = pat.match(ln)
        if not m or m.group(1) == "加總":
            if m:
                total = [0.0 if g == "-" else float(g.replace(",", "")) for g in m.groups()[1:]]
            continue
        town = K.town_key(m.group(1))
        v = [0.0 if g == "-" else float(g.replace(",", "")) for g in m.groups()[1:]]
        rows.append({"town": town, "county": K.county_of_town(town), "irrigation": v[0],
                     "aquaculture": v[1], "domestic": v[2], "industry": v[3], "other": v[4]})
    df = pd.DataFrame(rows)
    df.attrs["printed_total"] = total
    return df


# ----------------------------------------------------------------------------- WRA (2013)
def wra_deep_wells(txt: str) -> pd.DataFrame:
    """Tables 2-1.3 (totals by township) and 2-1.4 (wells deeper than 100 m by use)."""
    t13 = _between(txt, "表 2-1.3   彰化、雲林地區歷年水井調查成果統計表", "表 2-1.4", "溪湖鎮")
    t14 = _between(txt, "表 2-1.4   彰化、雲林地區歷年水井調查井深大於 100 公尺者統計表",
                   "此外，因現地無法確認", "溪湖鎮")
    n = r"([\d,]+)"
    tok = r"(－|-|[\d,]+)"
    tot = {}
    p13 = re.compile(rf"(\S{{2,3}}[鄉鎮市])\s+(\d{{2}})\s+{n}\s+{n}\s+{n}")
    for ln in t13.splitlines():
        for m in p13.finditer(ln):
            tot[K.town_key(m.group(1))] = {
                "survey_year_roc": int(m.group(2)), "wells_total": int(m.group(3).replace(",", "")),
                "wells_agri": int(m.group(4).replace(",", "")),
                "wells_ind_dom": int(m.group(5).replace(",", ""))}
    p14 = re.compile(rf"(\S{{2,3}}[鄉鎮市])\s+(\d{{2}})\s+{n}\s+{tok}\s+{tok}\s+{tok}")
    rows = []
    for ln in t14.splitlines():
        for m in p14.finditer(ln):
            town = K.town_key(m.group(1))
            g = [0 if x in ("－", "-") else int(x.replace(",", "")) for x in m.groups()[2:]]
            rows.append({"town": town, "county": K.county_of_town(town),
                         "deep_gt100_total": g[0], "deep_gt100_agri": g[1],
                         "deep_gt100_industry": g[2], "deep_gt100_domestic": g[3],
                         **tot.get(town, {})})
    df = pd.DataFrame(rows)
    df["deep_count_share"] = df["deep_gt100_total"] / df["wells_total"]
    # a '*' in Table 2-1.3 marks deep counts taken from the water-rights register (legal
    # wells only); those townships' shares are lower bounds
    star = set()
    for ln in t13.splitlines():
        for m in re.finditer(r"(\S{2,3}[鄉鎮市])\s+\d{2}\s+[\d,]+\s+[\d,]+\s+[\d,]+\s+[\d,]+(\*)",
                             ln):
            star.add(K.town_key(m.group(1)))
    df["deep_from_water_rights_register"] = df["town"].isin(star)
    return df


# ----------------------------------------------------------------------------- MOA yearbooks
def _split_row(tokens: list[str]) -> dict | None:
    """Total yield (thousands separated by a space) | total wells | 6 depth classes."""
    for k in (1, 2):
        if len(tokens) < k + 7:
            continue
        try:
            vals = [0 if t == "-" else int(t) for t in tokens[k:k + 7]]
        except ValueError:
            continue
        if sum(vals[1:7]) == vals[0] and vals[0] > 0:
            return {"yield_m3h": int("".join(tokens[:k])), "wells_total": vals[0],
                    **dict(zip(K.MOA_DEPTH_CLASSES, vals[1:7], strict=True))}
    return None


def moa_ia_wells() -> pd.DataFrame:
    rows = []
    for fname, year in K.MOA_PAGES.items():
        txt = _text(K.L0["moa_dir"] / fname)
        for name, county in (("彰", "changhua"), ("雲", "yunlin")):
            pat = re.compile(rf"^\s*{name}\s+(化|林)\s+(.*)$")
            for ln in txt.splitlines():
                m = pat.match(ln)
                if not m:
                    continue
                rec = _split_row(m.group(2).split())
                if rec:
                    rows.append({"yearbook_roc": year, "year": year + 1911, "county": county,
                                 "association": f"{'彰化' if county == 'changhua' else '雲林'}"
                                                "農田水利會", "file": fname, **rec})
                    break
    return pd.DataFrame(rows).drop_duplicates(["yearbook_roc", "county"])


def main(out_dir: str) -> dict:
    out = Path(out_dir)
    thesis = _text(K.L0["thesis"])
    split = lin_split_tables(thesis)
    use = lin_pumping_by_use(thesis)
    wra = wra_deep_wells(_text(K.L0["wra_well_mgmt"], 1, 80))
    moa = moa_ia_wells()
    split.to_csv(out / "township_layer_split.csv", index=False)
    use.to_csv(out / "township_pumping_by_use.csv", index=False)
    wra.to_csv(out / "township_wells_deep.csv", index=False)
    moa.to_csv(out / "ia_wells_depth_classes.csv", index=False)
    # checks against printed totals
    chk = {}
    for v, g in split.groupby("variant"):
        chk[f"lin_{v}_n_towns"] = int(len(g))
        chk[f"lin_{v}_max_abs_sum_minus_1"] = float((g["sum_a"] - 1).abs().max())
        chk[f"lin_{v}_town_share_sum"] = float(g["town_pump_share"].sum())
    printed = use.attrs["printed_total"]
    got = use[["irrigation", "aquaculture", "domestic", "industry", "other"]].sum().tolist()
    chk["lin_use_n_towns"] = int(len(use))
    chk["lin_use_max_rel_err_vs_printed_total"] = float(max(
        abs(a - b) / b for a, b in zip(got, printed, strict=True)))
    for c, tot, deep in (("changhua", 75266, 519), ("yunlin", 107689, 1709)):
        g = wra[wra["county"] == c]
        chk[f"wra_{c}_n_towns"] = int(len(g))
        chk[f"wra_{c}_wells_total_matches_printed"] = bool(g["wells_total"].sum() == tot)
        chk[f"wra_{c}_deep_total_matches_printed"] = bool(g["deep_gt100_total"].sum() == deep)
    chk["moa_rows"] = int(len(moa))
    chk["moa_years"] = sorted(int(y) for y in moa["year"].unique())
    return chk


if __name__ == "__main__":
    import json
    print(json.dumps(main(sys.argv[1]), ensure_ascii=False, indent=1))
