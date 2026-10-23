"""WP A0fix: probe the loader guard's time slicing with every time encoding a product may carry.

Each case writes a synthetic product (temporary directory, no real data) whose manifest
declares ``time_slicing`` on a column ``t`` stored with one encoding (ISO strings, datetimes,
YYYYMM integers, integer years, epoch seconds/milliseconds/days, out-of-range YYYYMM, mixed or
float columns), opens it for ``datum_fit`` with a pre-origin ``window_end`` and records what
``read_table`` returns. Every row carries the true last calendar month it covers
(``true_end``), so the probe counts the rows returned that end after the window (leaks).

The script uses only the public guard API, so the same probe runs against the guard before
and after the A0fix edit (the verifier's open defect; plan 2.5 U3/U5).

Usage: python -m common.a0fix.guard_time_probe --out <probe.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import guard, manifest  # noqa: E402


def _epoch_s(y: int, m: int, d: int) -> int:
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


_DAYS = [(2021, 6, 1), (2022, 12, 15), (2023, 1, 15), (2025, 6, 1)]
_DAYS_END = ["2021-06", "2022-12", "2023-01", "2025-06"]
_YEARS = [2020, 2021, 2022, 2023, 2025]
_YEARS_END = [f"{y}-12" for y in _YEARS]


def cases() -> list[dict]:
    """(name, values, true_end, encoding, fmt, window_end, dtype) for every probe."""
    c = []

    def add(name, values, true_end, encoding=None, fmt="csv", window_end="2022-12", **kw):
        c.append(dict(name=name, values=values, true_end=true_end, encoding=encoding, fmt=fmt,
                      window_end=window_end, **kw))

    add("iso month strings, no encoding (control)",
        ["2021-11", "2022-06", "2022-12", "2023-01", "2025-06"],
        ["2021-11", "2022-06", "2022-12", "2023-01", "2025-06"])
    add("iso dates with times, encoding iso",
        ["2021-06-01", "2022-12-31 23:00:00", "2023-01-01T00:30:00", "2025-06-01"],
        ["2021-06", "2022-12", "2023-01", "2025-06"], encoding="iso")
    add("integer year, no encoding (verifier case)", _YEARS, _YEARS_END)
    add("integer year, encoding year", _YEARS, _YEARS_END, encoding="year")
    add("integer year, encoding year, window_end 2022-06", _YEARS, _YEARS_END,
        encoding="year", window_end="2022-06")
    add("integer year, encoding yyyymm (wrong declaration)", _YEARS, _YEARS_END,
        encoding="yyyymm")
    add("integer year in parquet, no encoding", _YEARS, _YEARS_END, fmt="parquet")
    es = [_epoch_s(*d) for d in _DAYS]
    add("epoch seconds, no encoding (verifier case)", es, _DAYS_END)
    add("epoch seconds, encoding yyyymm", es, _DAYS_END, encoding="yyyymm")
    add("epoch seconds, encoding year", es, _DAYS_END, encoding="year")
    add("epoch milliseconds, no encoding", [v * 1000 for v in es], _DAYS_END)
    add("epoch days, no encoding", [v // 86400 for v in es], _DAYS_END)
    add("epoch days, encoding year", [v // 86400 for v in es], _DAYS_END, encoding="year")
    ym = [202111, 202212, 202301, 202506]
    ym_end = ["2021-11", "2022-12", "2023-01", "2025-06"]
    add("YYYYMM integers, encoding yyyymm", ym, ym_end, encoding="yyyymm")
    add("YYYYMM integers, no encoding (undeclared)", ym, ym_end)
    add("YYYYMM month 13, encoding yyyymm", [202111, 202213], ["2021-11", "2023-01"],
        encoding="yyyymm")
    add("YYYYMM month 00, encoding yyyymm", [202111, 202300], ["2021-11", "2023-00"],
        encoding="yyyymm")
    add("YYYYMM year 1899, encoding yyyymm", [189912, 202111], ["1899-12", "2021-11"],
        encoding="yyyymm")
    add("YYYYMM year 2100, encoding yyyymm", [202111, 210001], ["2021-11", "2100-01"],
        encoding="yyyymm")
    add("YYYYMMDD integers, encoding yyyymm", [20211115, 20230115], ["2021-11", "2023-01"],
        encoding="yyyymm")
    add("datetime64 parquet, no encoding", [dt.datetime(*d) for d in _DAYS], _DAYS_END,
        fmt="parquet")
    add("tz-aware datetime64 parquet, encoding iso",
        [pd.Timestamp(dt.datetime(*d), tz="Asia/Taipei") for d in _DAYS], _DAYS_END,
        encoding="iso", fmt="parquet")
    add("datetime64 parquet, encoding yyyymm (wrong declaration)",
        [dt.datetime(*d) for d in _DAYS], _DAYS_END, encoding="yyyymm", fmt="parquet")
    add("mixed csv strings (ISO, YYYYMM, YYYY), no encoding",
        ["2022-06", "202301", "2023"], ["2022-06", "2023-01", "2023-12"])
    add("bare year strings in parquet, no encoding", ["2021", "2022", "2023"],
        ["2021-12", "2022-12", "2023-12"], fmt="parquet")
    add("float year in parquet, encoding year", [2021.0, 2022.0, 2023.0],
        ["2021-12", "2022-12", "2023-12"], encoding="year", fmt="parquet")
    add("nullable Int64 year with NA, encoding year", [2021, None, 2023],
        ["2021-12", "9999-12", "2023-12"], encoding="year", fmt="parquet", dtype="Int64")
    add("invalid ISO strings (month 13, slash, padded)", ["2022-06", "2022-13"],
        ["2022-06", "2023-01"])
    add("unknown encoding 'epoch'", es, _DAYS_END, encoding="epoch")
    return c


def _write(tmp: Path, i: int, case: dict) -> Path:
    d = manifest.new_version_dir(f"probe/c{i:02d}", root=tmp)
    n = len(case["values"])
    vals = case["values"]
    col = pd.Series(vals, dtype=case.get("dtype")) if case.get("dtype") else \
        pd.Series(vals)
    df = pd.DataFrame({"value": np.arange(n, dtype=float), "sd": [np.nan] * n, "flag": [0] * n,
                       "method": ["obs"] * n, "product_version": d.name, "source_ids": "s",
                       "true_end": case["true_end"], "t": col})
    name = f"t.{case['fmt']}"
    if case["fmt"] == "csv":
        df.to_csv(d / name, index=False)
    else:
        df.to_parquet(d / name, index=False)
    (d / "validation.json").write_text("{}")
    code = tmp / "build.py"
    code.write_text("pass\n")
    ts = {"column": "t", "params_fit_window_end": "2022-12"}
    if case["encoding"] is not None:
        ts["encoding"] = case["encoding"]
    m = manifest.build_manifest(
        d, product=f"probe/c{i:02d}", layer="L1", inputs=[], code=[code], parameters={},
        fit_window_end="2026-09", validation_report="validation.json",
        gate={"passed": True, "criteria": "probe"},
        allowed_uses=["datum_fit", "scoring_target"], forbidden_uses=[], time_slicing=ts)
    manifest.write_manifest(d, m)
    return d


def run_case(tmp: Path, i: int, case: dict) -> dict:
    d = _write(tmp, i, case)
    rec = {"case": case["name"], "format": case["fmt"], "encoding": case["encoding"],
           "window_end": case["window_end"], "n_rows": len(case["values"]),
           "values": [str(v) for v in case["values"]]}
    try:
        p = guard.open_product(d, "datum_fit", window_end=case["window_end"])
        t = p.read_table(f"t.{case['fmt']}")
    except Exception as exc:  # noqa: BLE001 - the probe records the exception
        rec.update(outcome="refused", error=f"{type(exc).__name__}: {str(exc)[:240]}",
                   rows_returned=0, leaked_rows=0)
        return rec
    ends = t["true_end"].astype(str).tolist()
    leaked = [e for e in ends if e > case["window_end"]]
    rec.update(outcome="read", rows_returned=len(t), returned_true_end=ends,
               leaked_rows=len(leaked))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    res = []
    with tempfile.TemporaryDirectory() as t:
        for i, case in enumerate(cases()):
            res.append(run_case(Path(t), i, case))
    summary = {"cases": len(res), "read": sum(r["outcome"] == "read" for r in res),
               "refused": sum(r["outcome"] == "refused" for r in res),
               "cases_with_leaks": sum(r["leaked_rows"] > 0 for r in res),
               "leaked_rows": sum(r["leaked_rows"] for r in res)}
    out = {"label": a.label, "guard_file_sha256": _sha(Path(guard.__file__)),
           "summary": summary, "cases": res}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    for r in res:
        extra = r.get("error", "") if r["outcome"] == "refused" else \
            f"rows {r['rows_returned']} leaked {r['leaked_rows']}"
        print(f"{r['outcome']:8s} {r['case']}: {extra[:150]}")
    print(json.dumps(summary))
    return 0


def _sha(p: Path) -> str:
    from common.hashing import sha256_file
    return sha256_file(p)


if __name__ == "__main__":
    raise SystemExit(main())
