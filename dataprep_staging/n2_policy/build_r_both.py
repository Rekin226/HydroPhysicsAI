"""The N2C recorded-pumping multiplier: a PER-CELL field carrying each county's own r(t).

N2C (the Changhua head-response companion of N2) forces a model that carries Changhua
pumping, so a single fan-wide r(t) is wrong for it, and a Changhua-only scope (``#changhua``)
would leave the recorded Yunlin anomaly out of the PR arm while it is in the observations
(drought years move both counties together; review of plan S3, 2026-10-28). This builder
writes one ``.npz`` for ``twin.forward --recorded-pumping 'PR=<npz>@2023-01'``:

    month       (n_months,)            "YYYY-MM"
    multiplier  (n_months, n_active)   r_CH(t) on Changhua cells, r_YL(t) on every other cell
    n_active    ()                     the grid the county map was built on
    start       ()                     the first month

r_county(t) is N2's frozen rule (``build_r.ratio_series``, imported unchanged): the X1
in-grid county value of month t over its 2012-2022 mean for that month of year. The Changhua
flags are the cached county map (``zone_county_map/v2``, the map R1c was calibrated on; its
sha256 is recorded and, with ``--expect-county-sha``, checked). A month one county lacks takes
r = 1 for that county (N2's 'unrecorded' rule). ``--scope changhua`` writes the declared
secondary (Changhua cells r_CH, every other cell 1); ``--column lo90|hi90`` the X1-band
secondaries (numerator at the band, denominator the 2012-2022 mean of ``value``).

Reads electricity only (X1 is a post-origin product: allowed use ``forcing_recorded_arm``);
no head, leveling or pile value. Writes NEW files only (refuses to overwrite) plus a JSON
record of every input hash.

Usage: python -I build_r_both.py --x1 <county_monthly.parquet> --first 2023-01 --out <r.npz>
       [--last YYYY-MM] [--scope both|changhua] [--column value|lo90|hi90]
       [--county-map <zone_county_map.npz>] [--expect-county-sha <sha256>]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from n2_policy.build_r import ratio_series  # noqa: E402

DEFAULT_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
R1C_COUNTY_SHA = "6ca7840401a5f77ace183acc605f38c936b79de94df54bec8b1134bbfd86529e"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_round_trip(r: pd.DataFrame) -> pd.DataFrame:
    """The multipliers as N2's forward reads them: written by ``to_csv`` and parsed back
    (``build_r`` writes a CSV), so the Yunlin cells carry N2's r(t) bit for bit."""
    txt = r[["month", "multiplier"]].to_csv(index=False)
    back = pd.read_csv(io.StringIO(txt), dtype={"month": str})
    out = r.copy()
    out["multiplier"] = back["multiplier"].to_numpy(dtype="float64")
    return out


def county_ratio(x1: pd.DataFrame, first: str, county: str, column: str = "value"
                 ) -> pd.DataFrame:
    """``ratio_series`` with the numerator taken from ``column`` (``value`` reproduces it
    exactly; ``lo90``/``hi90`` keep the 2012-2022 ``value`` climatology as denominator)."""
    base = ratio_series(x1, first, county=county)
    if column == "value":
        return base
    x = x1[(x1["county"] == county) & (x1["scope"] == "in_grid_hybrid")].copy()
    x["date"] = pd.PeriodIndex(x["date"], freq="M")
    x = x.set_index("date").sort_index()
    ref = x[(x.index.year >= 2012) & (x.index.year <= 2022)]
    clim = ref.groupby(ref.index.month)["value"].mean()
    out = x[x.index >= pd.Period(first, freq="M")]
    r = pd.DataFrame({"month": out.index.strftime("%Y-%m"),
                      "multiplier": (out[column] / out.index.month.map(clim).to_numpy()
                                     ).to_numpy(),
                      "flag": out["flag"].to_numpy()})
    if not r["month"].equals(base["month"]):
        raise ValueError("band series months differ from the value series")
    return r


def load_flags(path: Path, expect: str | None) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        flags = np.asarray(z["changhua"]).astype(bool)
        n_active = int(z["n_active"])
        stored = str(z["sha256"])
    own = hashlib.sha256(np.ascontiguousarray(flags.astype("uint8")).tobytes()).hexdigest()
    if flags.shape != (n_active,):
        raise ValueError(f"{path}: flags {flags.shape} vs n_active {n_active}")
    if expect is not None and stored != expect:
        raise ValueError(f"{path}: county map sha256 {stored[:12]} != expected {expect[:12]}")
    return flags, {"path": str(path), "file_sha256": sha(path), "flags_sha256": stored,
                   "flags_sha256_recomputed_uint8": own, "n_active": n_active,
                   "n_changhua": int(flags.sum())}


def build(x1: pd.DataFrame, first: str, flags: np.ndarray, scope: str = "both",
          column: str = "value", last: str | None = None) -> tuple[dict, dict]:
    """-> (npz arrays, record). Months are the union of both counties' recorded months
    from ``first`` (to ``last`` when given); a county without a month takes 1."""
    per = {c: csv_round_trip(county_ratio(x1, first, c, column))
           for c in ("yunlin", "changhua")}
    if last is not None:
        per = {c: r[r["month"] <= last].reset_index(drop=True) for c, r in per.items()}
    months = sorted(set(per["yunlin"]["month"]) | set(per["changhua"]["month"]))
    if not months:
        raise ValueError("no recorded month in range")
    idx = {c: r.set_index("month")["multiplier"].reindex(months).fillna(1.0).to_numpy()
           for c, r in per.items()}
    yl = idx["yunlin"] if scope == "both" else np.ones(len(months))
    field = np.where(flags[None, :], idx["changhua"][:, None], yl[:, None]).astype("float64")
    if not np.isfinite(field).all() or (field < 0).any():
        raise ValueError("non-finite or negative multiplier")
    arrays = {"month": np.array(months), "multiplier": field,
              "n_active": np.int64(flags.size), "start": np.str_(months[0])}
    rec = {"rule": "r_county(t) = X1 in_grid_hybrid county value of month t / its 2012-2022 "
                   "mean for that month of year (N2 forcing_rule_frozen_now, per county); "
                   "Changhua cells r_CH, other cells "
                   + ("r_YL" if scope == "both" else "1"),
           "scope": scope, "column": column, "first": months[0], "last": months[-1],
           "n_months": len(months),
           "months_recorded": {c: int(len(r)) for c, r in per.items()},
           "flags": {c: {str(k): int(v) for k, v in r["flag"].value_counts().items()}
                     for c, r in per.items()}}
    return arrays, rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--x1", required=True, type=Path)
    ap.add_argument("--first", default="2023-01")
    ap.add_argument("--last", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scope", choices=("both", "changhua"), default="both")
    ap.add_argument("--column", choices=("value", "lo90", "hi90"), default="value")
    ap.add_argument("--county-map", type=Path, default=Path(DEFAULT_MAP))
    ap.add_argument("--expect-county-sha", default=R1C_COUNTY_SHA)
    a = ap.parse_args(argv)
    if a.out.suffix != ".npz":
        raise SystemExit("--out must end in .npz")
    if a.out.exists() or a.out.with_suffix(".json").exists():
        raise SystemExit(f"{a.out} (or its .json) exists; written once")
    flags, cmap = load_flags(a.county_map, a.expect_county_sha or None)
    arrays, rec = build(pd.read_parquet(a.x1), a.first, flags, a.scope, a.column, a.last)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **arrays)
    rec.update({"x1": str(a.x1), "x1_sha256": sha(a.x1), "county_map": cmap,
                "builder": str(Path(__file__).name), "builder_sha256": sha(Path(__file__)),
                "build_r_sha256": sha(Path(__file__).with_name("build_r.py")),
                "npz_sha256": sha(a.out)})
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "flags"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
