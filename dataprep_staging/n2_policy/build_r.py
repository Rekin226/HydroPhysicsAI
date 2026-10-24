"""The N2 recorded-pumping multiplier r(t) from the X1 county-monthly electricity product.

r(t) = Yunlin in-grid electricity of month t / its 2012-2022 mean for that month of year
(N2.policy_response.prospective_2026-11.v1, forcing_rule_frozen_now). Writes a NEW
``month,multiplier`` CSV for ``twin.forward --recorded-pumping`` plus a JSON record of the
X1 file's sha256 and the months covered; refuses to overwrite. Reads electricity only (no
head, leveling or pile value).

Usage: python -I build_r.py --x1 <county_monthly.parquet> --first 2023-01 --out <r.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def ratio_series(x1: pd.DataFrame, first: str, county: str = "yunlin") -> pd.DataFrame:
    x = x1[(x1["county"] == county) & (x1["scope"] == "in_grid_hybrid")].copy()
    x["date"] = pd.PeriodIndex(x["date"], freq="M")
    x = x.set_index("date").sort_index()
    ref = x[(x.index.year >= 2012) & (x.index.year <= 2022)]
    clim = ref.groupby(ref.index.month)["value"].mean()
    out = x[x.index >= pd.Period(first, freq="M")]
    return pd.DataFrame({"month": out.index.strftime("%Y-%m"),
                         "multiplier": (out["value"] / out.index.month.map(clim).to_numpy()).to_numpy(),
                         "flag": out["flag"].to_numpy()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--x1", required=True, type=Path)
    ap.add_argument("--first", default="2023-01")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; written once")
    r = ratio_series(pd.read_parquet(a.x1), a.first)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    r[["month", "multiplier"]].to_csv(a.out, index=False)
    rec = {"x1": str(a.x1), "x1_sha256": hashlib.sha256(a.x1.read_bytes()).hexdigest(),
           "first": r["month"].iloc[0], "last": r["month"].iloc[-1], "n_months": len(r),
           "flags": {str(k): int(v) for k, v in r["flag"].value_counts().items()},
           "csv_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
