"""WP A0 repair round: per-pilot data-window audit (evidence for each pilot's fit_window_end).

The first-round pilot manifests carried a blanket ``fit_window_end = 2026-10``. This script
collects, for every ``results/dataprep/_pilots/<scout>/``, the evidence needed to set a
per-pilot value:

* every literal year 2023-2026 (and month strings 2023-01..2026-12) in the pilot's own
  scripts (download directories ``dl*/`` are data and are skipped), with file:line;
* the date range of every tabular or array output, read from date-like columns or keys only
  (CSV header + date columns, parquet schema + date columns, npz/npy date keys, JSON
  ``YYYY-MM`` strings). Downloads (``dl*/``) are summarised separately because a download's
  extent is what was available, not necessarily what was used.

Privacy: only date-like columns are read (the census columns 戶名1 / 用電地址2 are never
selected; parquet schemas are read first and only matching columns loaded). No values other
than dates leave this script.

Usage: python -m common.a0.r2_window_audit --out <evidence.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.hashing import iter_files  # noqa: E402
from common.manifest import MANIFEST_NAME  # noqa: E402
from common.paths import PILOTS  # noqa: E402

FORBIDDEN_COLUMNS = {"戶名1", "用電地址2"}
_DATE_COL = re.compile(r"(date|month|^ym$|yyyymm|^ym_|_ym$|time|period|^year$|^yr$|^t$|epoch|"
                       r"^day$|^dt$|日期|年月|年|月)", re.I)
_DATE_KEY = re.compile(r"^(dates?|months?|t|time|times|years?|ym|yms|epochs?|tdays?|day)$",
                       re.I)
_YEAR_LIT = re.compile(r"(?<![0-9])(202[3-6])(?:[-/]?(0[1-9]|1[0-2]))?(?![0-9])")
_YM_STR = re.compile(r"(?<![0-9])((?:19|20)\d{2})-(0[1-9]|1[0-2])(?![0-9])")
CODE_SUFFIXES = {".py", ".sh"}
MAX_CSV_BYTES = 400_000_000


def _is_download(scout_dir: Path, p: Path) -> bool:
    return any(x.startswith("dl") for x in p.relative_to(scout_dir).parts[:-1])


def _to_month(values) -> tuple[str | None, str | None]:
    """(min, max) as YYYY-MM from an array of dates, datetimes, YYYYMM ints or years."""
    s = pd.Series(np.asarray(values).ravel())
    s = s.dropna()
    if s.empty:
        return None, None
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        v = v[np.isfinite(v)]
        if v.empty:
            return None, None
        lo, hi = float(v.min()), float(v.max())
        if lo >= 1900 and hi <= 2100:                        # years
            return f"{int(lo):04d}-01", f"{int(hi):04d}-12"
        if lo >= 190001 and hi <= 210012:                    # YYYYMM
            return f"{int(lo) // 100:04d}-{int(lo) % 100:02d}", \
                f"{int(hi) // 100:04d}-{int(hi) % 100:02d}"
        if lo >= 19000101 and hi <= 21001231:                # YYYYMMDD
            return f"{int(lo) // 10000:04d}-{int(lo) // 100 % 100:02d}", \
                f"{int(hi) // 10000:04d}-{int(hi) // 100 % 100:02d}"
        return None, None
    try:
        d = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
    except (TypeError, ValueError):
        d = pd.to_datetime(s.astype(str), errors="coerce")
    d = d.dropna()
    d = d[(d.dt.year >= 1900) & (d.dt.year <= 2100)]
    if d.empty:
        return None, None
    return d.min().strftime("%Y-%m"), d.max().strftime("%Y-%m")


def _table_ranges(p: Path) -> dict:
    out = {}
    try:
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            schema = pq.read_schema(p)
            cols = [n for n, t in zip(schema.names, schema.types, strict=True)
                    if n not in FORBIDDEN_COLUMNS and (_DATE_COL.search(n)
                                                       or "timestamp" in str(t)
                                                       or str(t).startswith("date"))]
            if not cols:
                return {}
            df = pd.read_parquet(p, columns=cols)
        elif p.suffix == ".csv":
            if p.stat().st_size > MAX_CSV_BYTES:
                return {"_skipped": "too large"}
            head = pd.read_csv(p, nrows=0)
            cols = [c for c in head.columns if c not in FORBIDDEN_COLUMNS and _DATE_COL.search(str(c))]
            if not cols:
                return {}
            df = pd.read_csv(p, usecols=cols, low_memory=False)
        else:
            return {}
    except Exception as exc:  # noqa: BLE001 - evidence only
        return {"_error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    for c in df.columns:
        lo, hi = _to_month(df[c].to_numpy())
        if hi:
            out[str(c)] = [lo, hi]
    return out


def _array_ranges(p: Path) -> dict:
    out = {}
    try:
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    if _DATE_KEY.match(k):
                        lo, hi = _to_month(z[k])
                        if hi:
                            out[k] = [lo, hi]
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return out


def _json_range(p: Path) -> dict:
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    yms = [f"{a}-{b}" for a, b in _YM_STR.findall(txt)]
    yms = [x for x in yms if "1990-01" <= x <= "2030-12"]
    return {"ym_strings": [min(yms), max(yms)], "n_after_origin":
            sum(x > "2022-12" for x in yms)} if yms else {}


def audit_pilot(d: Path) -> dict:
    code_hits, outputs, downloads = [], {}, {}
    for p in iter_files(d, (MANIFEST_NAME,)):
        rel = p.relative_to(d).as_posix()
        dl = _is_download(d, p)
        if p.suffix in CODE_SUFFIXES and not dl:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace")
                                     .splitlines(), 1):
                if _YEAR_LIT.search(line):
                    code_hits.append(f"{rel}:{i}: {line.strip()[:150]}")
            continue
        if p.suffix in (".csv", ".parquet"):
            r = _table_ranges(p)
        elif p.suffix == ".npz":
            r = _array_ranges(p)
        elif p.suffix == ".json":
            r = _json_range(p)
        else:
            r = {}
        if r:
            (downloads if dl else outputs)[rel] = r
    maxes = [v[1] for r in outputs.values() for k, v in r.items()
             if isinstance(v, list) and k != "ym_strings" and v and v[1]]
    jmax = [r["ym_strings"][1] for r in outputs.values() if "ym_strings" in r]
    return {
        "scout": d.name,
        "code_post_origin_year_mentions": code_hits,
        "n_code_mentions": len(code_hits),
        "output_date_ranges": outputs,
        "download_date_ranges": downloads,
        "max_output_table_month": max(maxes) if maxes else None,
        "max_output_json_month": max(jmax) if jmax else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {}
    for d in sorted(p for p in PILOTS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        r = audit_pilot(d)
        res[d.name] = r
        print(f"{d.name:28s} code mentions {r['n_code_mentions']:4d}  max table month "
              f"{r['max_output_table_month']}  max json month {r['max_output_json_month']}")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
