"""WP W4 residual-gap diagnostic: how much of the 2023-26 rain masking comes from CODIS
'missing-value marks' (Accumulationf > 0) on days that still carry a numeric total?

After-the-fact analysis for the report only (not a product, changes nothing): for each gauge
chain station-year in data_fetch_api/new_measurements/cwa_codis_rain/raw, count days by
status, and recompute the kept gauge-months (>= 80 % of days) under the rebuild's rule (only
unmarked numeric days) and under a tolerant rule that also accepts marked days with a numeric
non-negative total and a mark of 1 or 2 (magnitude semantics unverified).

Usage: python rain_flags.py --out <evidence_dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from wcommon import REPO, read_json_untrusted, write_json_new

sys.path.insert(0, str(REPO))
from hydrophysics.twin.cwa_rain import CHAINS  # noqa: E402  (read-only import)

RAW = REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / "raw"


def day_table() -> pd.DataFrame:
    rows = []
    for rf, chain in CHAINS.items():
        for sid, _typ, first, last in chain:
            for p in sorted(RAW.glob(f"{sid}_*.json")):
                if p.name.endswith(".meta.json"):
                    continue
                j = read_json_untrusted(p)
                for blk in j.get("data", []):
                    for x in blk.get("dts", []):
                        day = pd.Timestamp(x["DataDate"][:10])
                        if day < pd.Timestamp(first) or (last and day > pd.Timestamp(last)):
                            continue
                        pr = x.get("Precipitation") or {}
                        a, f = pr.get("Accumulation"), pr.get("Accumulationf")
                        try:
                            a = float(a)
                        except (TypeError, ValueError):
                            a = None
                        f = 0 if f is None else int(f) if str(f).lstrip("-").isdigit() else -1
                        rows.append({"rf": rf, "station": sid, "date": day, "value": a,
                                     "mark": f})
    return pd.DataFrame(rows)


def kept(df: pd.DataFrame, tolerant: bool) -> pd.DataFrame:
    ok = df["value"].notna() & ((df["value"] >= 0) | (df["value"] == -9.8))
    good = ok & ((df["mark"] == 0) | (tolerant & df["mark"].between(1, 2)))
    d = df.assign(good=good)
    # newer station wins where chains overlap (as in the rebuild)
    d = d.sort_values(["rf", "date", "station"]).groupby(["rf", "date"])["good"].max()
    d = d.unstack("rf").reindex(pd.date_range("2012-01-01", "2026-09-30", freq="D"))
    d = d.fillna(False).astype(bool)
    m = d.resample("MS").sum()
    return m.ge(0.8 * m.index.days_in_month.to_numpy()[:, None])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = day_table()
    post = df[df["date"] >= "2023-01-01"]
    marks = post["mark"].value_counts().sort_index()
    strict, tol = kept(df, False), kept(df, True)
    res = {"label": "after-the-fact residual-gap diagnostic (W4 report only)",
           "days_2023_2026_by_mark": {str(k): int(v) for k, v in marks.items()},
           "kept_frac_by_year_strict": {str(y): float(x) for y, x in
                                        strict.groupby(strict.index.year).mean().mean(1)
                                        .items()},
           "kept_frac_by_year_tolerant_mark_le_2": {str(y): float(x) for y, x in
                                                    tol.groupby(tol.index.year).mean()
                                                    .mean(1).items()},
           "kept_gauge_months_2023_2026_strict": int(strict.loc["2023-01":].sum().sum()),
           "kept_gauge_months_2023_2026_tolerant": int(tol.loc["2023-01":].sum().sum()),
           "note": "Accumulationf magnitude semantics are not documented here (likely hours "
                   "missing); the tolerant rule is a what-if, not a recommendation to keep "
                   "marked totals without QA."}
    write_json_new(out / "rain_flag_diagnostic.json", res)
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
