"""Task 1: inventory of the TPC / MOEAEA open datasets already in hand (and of later fetches).

Usage: python -I inventory.py OUT_JSON DIR [DIR ...]

Each DIR is a dated L0 download directory (``data_fetch_api/new_measurements/<id>/<date>``).
The files are untrusted data: they are parsed as text tables only. For each file the script
reports its columns, the time span (first and last month), the sectors or classes, the
geography (counties, townships, villages) and the units, and for the county x sector table
the Changhua and Yunlin rows. Nothing is written except OUT_JSON.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

CH, YL = "彰化縣", "雲林縣"


def _ym_roc(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    roc = pd.to_numeric(s.str[:-2], errors="coerce")
    mm = pd.to_numeric(s.str[-2:], errors="coerce")
    ok = roc.notna() & mm.notna() & mm.between(1, 12)
    out = pd.Series(pd.NaT, index=s.index)
    out[ok] = pd.to_datetime(dict(year=(roc[ok] + 1911).astype(int),
                                  month=mm[ok].astype(int), day=1))
    return out


def _span(d: pd.Series) -> dict:
    d = pd.to_datetime(d.dropna())
    if d.empty:
        return {"first": None, "last": None, "n_months": 0}
    return {"first": d.min().strftime("%Y-%m"), "last": d.max().strftime("%Y-%m"),
            "n_months": int(d.dt.to_period("M").nunique())}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def d007012(p: Path) -> dict:
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig").iloc[:, :5]
    df.columns = ["ym", "county", "sector", "kwh", "pct"]
    df = df.dropna(subset=["ym"])
    df["date"] = _ym_roc(df["ym"])
    df["county"] = df["county"].str.strip().str.replace("臺", "台")
    df["kwh"] = _num(df["kwh"])
    out = {"columns": ["年月 (ROC yyymm)", "縣市", "用電性質", "售電量(度) kWh", "用電佔比(%)"],
           "span": _span(df["date"]), "rows": int(len(df)),
           "sectors": sorted(df["sector"].str.strip().unique().tolist()),
           "n_counties": int(df["county"].nunique()), "unit": "kWh per county, sector, month",
           "geography": "county (22)"}
    for c in (CH, YL):
        g = df[(df.county == c) & (df.sector.str.strip() == "農林漁牧")]
        out[f"ag_{c}"] = {"span": _span(g["date"]), "rows": int(len(g)),
                          "last_month_GWh": round(float(g.sort_values("date").kwh.iloc[-1])
                                                  / 1e6, 3)}
    return out


def d007021(p: Path) -> dict:
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    cols = list(df.columns)
    df["date"] = _ym_roc(df.iloc[:, 0])
    cty = df.iloc[:, -1].astype(str).str.strip().str.replace("臺", "台")
    out = {"columns": cols, "span": _span(df["date"]), "rows": int(len(df)),
           "n_counties": int(cty.nunique()), "n_townships": int(df.iloc[:, 6].nunique()),
           "n_villages": int(df.iloc[:, 3].astype(str).str.cat(df.iloc[:, 5].astype(str))
                             .nunique()),
           "unit": "kWh per first-level dissemination area (一級發布區) and billing month",
           "months": sorted(df["date"].dropna().dt.strftime("%Y-%m").unique().tolist())}
    for c in (CH, YL):
        g = df[cty == c]
        out[f"rows_{c}"] = int(len(g))
        out[f"kwh_{c}_GWh"] = round(float(_num(g.iloc[:, 2]).sum()) / 1e6, 3)
    return out


def d007019(p: Path) -> dict:
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig").iloc[:, :4]
    df.columns = ["year", "county", "sector", "pct"]
    yr = pd.to_numeric(df["year"], errors="coerce")
    return {"columns": ["年度", "縣市", "用電性質", "用電比例 (%)"],
            "years": [int(yr.min()), int(yr.max())],
            "sectors": sorted(df["sector"].dropna().str.strip().unique().tolist()),
            "unit": "percent of county annual sales (no kWh)", "rows": int(len(df))}


def d007023(p: Path) -> dict:
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    ag = df[df.iloc[:, 0].astype(str).str.strip().str[:2].isin(["01", "02", "03"])]
    return {"columns": list(df.columns), "rows": int(len(df)),
            "unit": "kWh per industry class, national, last calendar year (表燈營業 only)",
            "agri_rows": {str(r.iloc[1]).strip(): float(_num(pd.Series([r.iloc[2]]))[0])
                          for _, r in ag.iterrows()}}


def d003005(p: Path) -> dict:
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    yr = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return {"columns": list(df.columns), "years": [int(yr.min()), int(yr.max())],
            "uses": sorted(df.iloc[:, 1].dropna().unique().tolist()),
            "unit": "kWh per year, national, by use"}


def d003004(p: Path) -> dict:
    out = {"members": [], "unit": "kWh per month, national, 電力營業用戶 by industry class"}
    with zipfile.ZipFile(p) as z:
        frames = []
        for i in z.infolist():
            n = (i.filename.encode("cp437").decode("big5", "replace")
                 if i.flag_bits & 0x800 == 0 else i.filename)
            out["members"].append(n)
            if n.startswith("行業別售電_") and n.endswith(".csv"):
                t = pd.read_csv(io.BytesIO(z.read(i)), encoding="utf-8-sig", dtype=str)
                t.columns = ["big", "mid", "small", "month", "kwh"]
                t["year"] = int(n[6:10])
                frames.append(t)
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(dict(year=df.year, month=pd.to_numeric(df.month), day=1))
    out["span"] = _span(df["date"])
    out["agri_classes"] = sorted(df[df.big.str.startswith("A")]["small"].unique().tolist())
    return out


def moeaea(p: Path) -> dict:
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    d = pd.to_datetime(df.iloc[:, 0].str.strip(), format="%Y%m", errors="coerce")
    agcols = [c for c in df.columns if "農業" in c]
    return {"span": _span(d), "unit": str(df.iloc[0, 1]), "n_columns": len(df.columns),
            "agri_columns": agcols, "geography": "national"}


READERS = {"d007012": d007012, "d007021": d007021, "d007019": d007019,
           "d007023": d007023, "d003005": d003005, "d003004": d003004, "set_": moeaea}


def main(argv: list[str]) -> None:
    out_json, dirs = Path(argv[0]), [Path(a) for a in argv[1:]]
    res = {}
    for d in dirs:
        src = json.loads((d / "SOURCE.json").read_text(encoding="utf-8"))
        for f in src["files"]:
            p = d / f["file"]
            key = next((k for k in READERS if p.name.startswith(k)), None)
            if key is None and p.parent.parent.name.startswith("tpc_"):
                key = p.parent.parent.name[4:]
            rec = {"source_id": src["source_id"], "title": src["title"],
                   "licence": src["licence"], "fetched": src["fetched"], "url": f["url"],
                   "bytes": f["bytes"], "sha256": f["sha256"]}
            try:
                rec.update(READERS[key](p))
            except Exception as e:  # noqa: BLE001 - inventory reports the failure
                rec["error"] = f"{type(e).__name__}: {e}"
            res[f"{src['source_id']}/{d.name}/{f['file']}"] = rec
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n",
                        encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps({x: v.get(x) for x in ("span", "years", "sectors", "unit",
                                                    "error")}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
