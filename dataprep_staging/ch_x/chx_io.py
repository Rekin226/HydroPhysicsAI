"""File access of I1.CHX.v1 / I1.CH_backcast.v1 (metadata and the scorers' observation reads).

* ``ext_wells`` reads ONLY the metadata columns ``chx_const.EXT_COLUMNS`` of N3F's external
  wells export (the file also carries N3F's pre-origin datum and residual sd; they are never
  read here). ``ext_counts`` reads the logged pre-origin month count (G0's mask only).
* ``l1_monthly`` is the scorers' only path to observed heads: L1_clean v3 through the loader
  guard with use ``design_time_evidence`` and ``window_end = 2022-12`` (no post-origin row can
  be returned), flags 0-1, the declared wells and months only. It is called after
  ``prereg3.assert_can_score`` and nowhere else.

Data paths resolve against ``common.paths.REPO`` (override with DATAPREP_REPO).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
from common.paths import REPO  # noqa: E402


def rp(p: str | Path) -> Path:
    return Path(p) if Path(p).is_absolute() else REPO / p


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_or_missing(path: str | Path) -> str:
    p = Path(path)
    return sha256(p) if p.is_file() else "missing"


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- externals
def ext_wells(path: str | Path = K.EXT_WELLS) -> pd.DataFrame:
    """Kept external wells, metadata columns only, sorted by station id."""
    t = pd.read_csv(rp(path), usecols=list(K.EXT_COLUMNS), dtype={"station_id": str,
                                                                  "layer_code": str})
    t = t[t["kept"].astype(str).str.lower().isin(("true", "1"))].copy()
    t["layer"] = t["layer"].astype(int)
    t["cell"] = t["cell"].astype(int)
    return t.sort_values("station_id").reset_index(drop=True)


def county_rows(t: pd.DataFrame, county: str) -> pd.DataFrame:
    return t[t["county"].astype(str).str.lower() == county].reset_index(drop=True)


def changhua_ext(path: str | Path = K.EXT_WELLS) -> pd.DataFrame:
    t = county_rows(ext_wells(path), "changhua")
    if len(t) != K.N_EXT_CHANGHUA:
        raise SystemExit(f"{path}: {len(t)} kept Changhua external wells, expected "
                         f"{K.N_EXT_CHANGHUA}")
    return t


def rows_sha(t: pd.DataFrame) -> str:
    """sha256 of the canonical JSON of the declared metadata columns of these rows."""
    cols = [c for c in K.EXT_COLUMNS if c in t.columns]
    recs = [{c: (None if pd.isna(r[c]) else
                 (int(r[c]) if c in ("layer", "cell") else str(r[c]))) for c in cols}
            for _, r in t[cols].iterrows()]
    return canonical_sha(sorted(recs, key=lambda d: d["station_id"]))


def ext_counts(sids, path: str | Path = K.EXT_WELLS) -> np.ndarray:
    """N3F's logged pre-origin month count per screen (G0's availability mask only)."""
    t = pd.read_csv(rp(path), usecols=["station_id", K.EXT_COUNT_COLUMN],
                    dtype={"station_id": str}).drop_duplicates("station_id")
    t = t.set_index("station_id")
    miss = [s for s in sids if s not in t.index]
    if miss:
        raise SystemExit(f"{path}: no month count for {miss[:5]}")
    return t.loc[list(sids), K.EXT_COUNT_COLUMN].to_numpy(int)


# ----------------------------------------------------------------------------- observations
def l1_monthly(sids, months: list[str], product: str = K.L1, use: str = K.L1_USE,
               window_end: str = K.ORIGIN, table: str = "heads_monthly.parquet"
               ) -> tuple[np.ndarray, np.ndarray, list]:
    """(obs (W, T), flag (W, T), guard notes) for ``sids`` x ``months`` from a heads product,
    flags 0-1 kept (others NaN, flag 9). Scorers only, after assert_can_score."""
    from common import guard

    if months[-1] > window_end:
        raise ValueError(f"months reach {months[-1]} beyond window_end {window_end}")
    prod = guard.open_product(rp(product), use, window_end=window_end)
    long = prod.read_table(table, columns=["sid", "date", "value", "flag"])
    long = long[long["sid"].astype(str).isin(set(map(str, sids)))].copy()
    long["ym"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m")
    long = long[long["ym"].isin(set(months)) & long["flag"].astype(int).isin((0, 1))]
    W, T = len(sids), len(months)
    obs = np.full((W, T), np.nan)
    flag = np.full((W, T), 9, int)
    wi = {str(s): i for i, s in enumerate(sids)}
    ti = {m: j for j, m in enumerate(months)}
    for s, ym, v, f in long[["sid", "ym", "value", "flag"]].itertuples(index=False):
        i, j = wi[str(s)], ti[ym]
        obs[i, j], flag[i, j] = float(v), int(f)
    return obs, flag, list(prod.notes)
