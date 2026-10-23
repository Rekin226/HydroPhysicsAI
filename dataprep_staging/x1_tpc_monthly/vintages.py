"""Compare vintages (fetches) of TPC d007012: span, publication lag and revisions.

Usage: python -I vintages.py OUT_JSON LABEL=CAPTURE_UTC=FILE [LABEL=CAPTURE_UTC=FILE ...]

CAPTURE_UTC is when the vintage was observed online (Wayback capture time, or the HTTP
Last-Modified of a direct fetch). For each vintage: first and last month, the lag from the
end of its last month to the capture time (an upper bound on the publication lag, since the
file may have been posted earlier), and for 彰化縣 / 雲林縣 農林漁牧 the relative difference of
every month shared with the newest vintage (revisions). Files are parsed as text tables only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CH, YL, AG = "彰化縣", "雲林縣", "農林漁牧"


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").iloc[:, :5]
    df.columns = ["ym", "county", "sector", "kwh", "pct"]
    df = df.dropna(subset=["ym"])
    s = df["ym"].str.strip()
    df["date"] = pd.to_datetime(dict(year=s.str[:-2].astype(int) + 1911,
                                     month=s.str[-2:].astype(int), day=1))
    df["county"] = df["county"].str.strip().str.replace("臺", "台")
    df["sector"] = df["sector"].str.strip()
    df["kwh"] = pd.to_numeric(df["kwh"].str.replace(",", "").str.strip(), errors="coerce")
    return df


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    vint = []
    for a in argv[1:]:
        label, cap, f = a.split("=", 2)
        vint.append((label, pd.Timestamp(cap), read(Path(f))))
    vint.sort(key=lambda v: v[1])
    newest = vint[-1][2]
    ref = newest[newest.sector == AG].set_index(["county", "date"])["kwh"]
    res = {}
    for label, cap, df in vint:
        last = df["date"].max()
        month_end = last + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        rec = {"capture_utc": cap.isoformat(), "first": df["date"].min().strftime("%Y-%m"),
               "last": last.strftime("%Y-%m"),
               "days_from_last_month_end_to_capture": round(
                   (cap.tz_localize(None) - month_end).total_seconds() / 86400, 1),
               "rows": int(len(df))}
        ag = df[df.sector == AG].set_index(["county", "date"])["kwh"]
        for c in (CH, YL):
            a = ag.loc[c]
            r = ref.loc[c].reindex(a.index)
            rel = (a / r - 1.0).dropna()
            rec[f"revision_vs_newest_{c}"] = {
                "shared_months": int(len(rel)),
                "max_abs_rel": float(np.abs(rel).max()) if len(rel) else None,
                "months_differing_gt_0.1pct": [d.strftime("%Y-%m") for d in
                                               rel.index[np.abs(rel) > 1e-3]]}
        res[label] = rec
        print(label, json.dumps(rec, ensure_ascii=False))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
