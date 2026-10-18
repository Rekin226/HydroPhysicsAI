"""H1a step 0: profile the raw head records (read-only).

Per file: rows, first/last timestamp, sampling-interval mix (hourly vs 10-minute), sentinel
counts (|value| >= 1000, the -999998 blocks), constant floors, duplicated timestamps, and the
2022-10..12 overlap between the historical cache and the new network (daily means).

Usage: p0_profile.py OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from h1_heads_qc import rawio  # noqa: E402


def profile(path, src):
    f = pd.read_parquet(path)
    idx = pd.to_datetime(f.get("datetime", f.index))
    v = pd.to_numeric(f["value"], errors="coerce").to_numpy()
    out = {"sid": path.stem, "src": src, "rows": int(len(v))}
    if len(v) == 0:
        return out
    order = np.argsort(idx.values, kind="stable")
    t = idx.values[order]
    v = v[order]
    dt = np.diff(t).astype("timedelta64[s]").astype(np.int64)
    out.update({
        "t0": str(pd.Timestamp(t[0])), "t1": str(pd.Timestamp(t[-1])),
        "n_dup_ts": int((dt == 0).sum()),
        "n_nan": int(np.isnan(v).sum()),
        "n_sentinel": int((np.abs(v) >= 1000).sum()),
        "n_m999998": int(np.isclose(v, -999998).sum()),
        "frac_10min": float(np.mean((dt > 0) & (dt <= 660))) if len(dt) else 0.0,
        "frac_hourly": float(np.mean((dt > 660) & (dt <= 3660))) if len(dt) else 0.0,
        "first_10min": None,
    })
    ten = np.flatnonzero((dt > 0) & (dt <= 660))
    if len(ten):
        out["first_10min"] = str(pd.Timestamp(t[ten[0]]))
    ok = np.isfinite(v) & (np.abs(v) < 1000)
    if ok.any():
        out["vmin"] = float(np.min(v[ok]))
        out["vmax"] = float(np.max(v[ok]))
        out["vmed"] = float(np.median(v[ok]))
    return out


def main():
    out_json = sys.argv[1]
    rows = []
    for src, d in (("hist", rawio.HIST_DIR), ("new", rawio.NEW_DIR)):
        for p in sorted(d.glob("*.parquet")):
            rows.append(profile(p, src))
    P = pd.DataFrame(rows)
    # overlap check: 2022-10-01..2022-12-31 daily means, hist vs new (raw, sentinels removed)
    pairs, maxdiff, wells = 0, 0.0, 0
    for p in sorted(rawio.NEW_DIR.glob("*.parquet")):
        h = rawio.HIST_DIR / p.name
        if not h.exists():
            continue
        a = rawio.load_raw(h)
        b = rawio.load_raw(p)
        a = a[(a.index >= "2022-10-01") & (a.index < "2023-01-01")]
        b = b[(b.index >= "2022-10-01") & (b.index < "2023-01-01")]
        if a.empty or b.empty:
            continue
        da = a.resample("D").mean().dropna()
        db = b.resample("D").mean().dropna()
        j = da.index.intersection(db.index)
        if len(j):
            wells += 1
            pairs += len(j)
            maxdiff = max(maxdiff, float(np.max(np.abs(da[j].to_numpy() - db[j].to_numpy()))))
    summ = {
        "files": {s: int((P.src == s).sum()) for s in ("hist", "new")},
        "new_not_in_hist": sorted(set(P[P.src == "new"].sid) - set(P[P.src == "hist"].sid)),
        "hist_t1_max": str(P[P.src == "hist"].t1.dropna().max()),
        "hist_t0_min": str(P[P.src == "hist"].t0.dropna().min()),
        "new_t0_min": str(P[P.src == "new"].t0.dropna().min()),
        "new_t1_max": str(P[P.src == "new"].t1.dropna().max()),
        "empty_files": P[P.rows == 0].sid.tolist(),
        "files_with_sentinels": int((P.n_sentinel > 0).sum()),
        "sentinel_rows": int(P.n_sentinel.sum()),
        "m999998_rows": int(P.n_m999998.sum()),
        "dup_ts_rows": int(P.n_dup_ts.sum()),
        "overlap_2022q4": {"wells": wells, "daily_pairs": pairs, "max_abs_diff_m": maxdiff},
    }
    Path(out_json).write_text(json.dumps({"summary": summ, "files": rows}, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
