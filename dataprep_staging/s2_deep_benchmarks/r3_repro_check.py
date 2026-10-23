"""WP S2 round 3: compare a ``build.py --scratch`` rebuild with deep_benchmarks/v1, and probe
the pandas 2.3.3 microsecond-index defect that ``s2lib.as_ns`` works around.

Usage: python -I r3_repro_check.py <scratch_dir> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO  # noqa: E402

PRODUCT = REPO / "results" / "dataprep" / "deep_benchmarks" / "v1"
LABEL_COLS = ("product_version",)


def compare(scratch: Path) -> dict:
    out = {}
    for p in sorted(scratch.iterdir()):
        q = PRODUCT / p.name
        if not q.exists():
            out[p.name] = "only in scratch"
            continue
        if sha256_file(p) == sha256_file(q):
            out[p.name] = "byte-identical"
            continue
        if p.suffix in (".parquet", ".csv"):
            rd = pd.read_parquet if p.suffix == ".parquet" else pd.read_csv
            a, b = rd(q), rd(p)
            a = a.drop(columns=[c for c in LABEL_COLS if c in a.columns])
            b = b.drop(columns=[c for c in LABEL_COLS if c in b.columns])
            try:
                pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                              check_exact=False, rtol=1e-12, atol=1e-12)
                out[p.name] = "values identical (only the product_version label differs)"
            except AssertionError as exc:
                out[p.name] = f"DIFFERS: {str(exc)[:300]}"
        elif p.name == "validation.json":
            va = json.loads(q.read_text(encoding="utf-8"))
            vb = json.loads(p.read_text(encoding="utf-8"))
            diff = sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k))
            out[p.name] = {"keys_differing": diff}
    return out


def pandas_probe() -> dict:
    a = pd.date_range("2016-06-23 16:00", "2025-03-12 13:00", freq="h", unit="us")
    b = pd.date_range("2018-12-27 13:00", "2025-03-31 00:00", freq="h", unit="us")
    u = a.union(b)
    un = a.as_unit("ns").union(b.as_unit("ns"))
    table = {}
    for unit in ("s", "ms", "us", "ns"):
        for f in ("10min", "h", "D"):
            x = pd.date_range("2016-06-23 16:00", "2025-03-12 13:00", freq=f, unit=unit)
            y = pd.date_range("2018-12-27 13:00", "2025-03-31 00:00", freq=f, unit=unit)
            exp = x.as_unit("ns").append(y.as_unit("ns")).unique().sort_values()
            raw = x.union(y)
            try:
                got = raw.as_unit("ns")
                ok = bool(len(got) == len(exp) and (got == exp).all())
                table[f"{unit}/{f}"] = "correct" if ok else (
                    f"WRONG: {len(got)} entries (expected {len(exp)}), max {raw.max()}")
            except (OverflowError, ValueError) as exc:     # e.g. a year-21029 timestamp
                table[f"{unit}/{f}"] = f"WRONG: {len(raw)} entries, {type(exc).__name__}: {exc}"
    s = pd.Series(1.0, index=pd.date_range("2020-01-01", "2020-03-01", freq="h", unit="us"))
    r = pd.Series(2.0, index=pd.date_range("2020-02-01", "2020-04-01", freq="h", unit="us"))
    return {"pandas": pd.__version__,
            "us_union_len": int(len(u)), "us_union_max": str(u.max()),
            "us_union_after_inputs": int((u > b.max()).sum()),
            "ns_union_len": int(len(un)), "ns_union_max": str(un.max()),
            "expected_len": int(len(a.as_unit("ns").append(b.as_unit("ns")).unique())),
            "union_correct_by_unit_and_freq": table,
            "concat_axis1_two_hourly_us_series_2020": {
                "rows": int(len(pd.concat([s, r], axis=1))),
                "expected": int(len(pd.date_range("2020-01-01", "2020-04-01", freq="h")))}}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    res = {"schema": "s2-r3-repro/1", "scratch": argv[0], "product": "deep_benchmarks/v1",
           "files": compare(Path(argv[0])), "pandas_us_union_probe": pandas_probe()}
    Path(argv[1]).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
