"""Flag table and table/grid conventions common to all products (plan 2.4).

Long tables carry ``value, sd, flag, method, product_version, source_ids``. Grid arrays carry
``value``, ``sd`` and ``flag`` (uint8) with the same shape, plus ``dates`` and the mask hash.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class Flag(IntEnum):
    OBSERVED = 0          # observed, passed QC
    CORRECTED = 1         # observed, corrected by a documented model (metadata or co-located)
    SUBPERIOD = 2         # completed from sub-period data of the same instrument
    FILLED_OTHER = 3      # filled or split using other sites or instruments
    FILLED_TEMPORAL = 4   # filled from the series' own temporal model
    INFERRED = 5          # inferred from an aggregate or a model
    EXTRAPOLATED = 6      # beyond the tested envelope or the record end (never calibration)
    MISSING = 8           # missing, not filled
    SUSPECT = 9           # suspect observation (kept in L1; excluded downstream by default)


FLAG_MEANING = {
    Flag.OBSERVED: "observed, passed QC",
    Flag.CORRECTED: "observed, corrected by a documented model backed by physical metadata or "
                    "a co-located instrument",
    Flag.SUBPERIOD: "completed from sub-period data of the same instrument (daily bridge)",
    Flag.FILLED_OTHER: "filled or split using other sites or instruments",
    Flag.FILLED_TEMPORAL: "filled from the series' own temporal model",
    Flag.INFERRED: "inferred from an aggregate or a model",
    Flag.EXTRAPOLATED: "extrapolated beyond the tested envelope or beyond the record end "
                       "(never used for calibration)",
    Flag.MISSING: "missing, not filled",
    Flag.SUSPECT: "suspect observation (kept in L1; excluded downstream unless a product says "
                  "otherwise)",
}

VALID_FLAGS = frozenset(int(f) for f in Flag)
OBSERVED_FLAGS = frozenset({0, 1})
#: Flags whose value is not a direct observation; they must carry an sd and a method (2.1 L2).
NON_OBSERVED_FLAGS = frozenset({2, 3, 4, 5, 6})

LONG_TABLE_COLUMNS = ("value", "sd", "flag", "method", "product_version", "source_ids")
GRID_KEYS = ("value", "sd", "flag", "dates", "mask_hash")


def validate_long_table(df, require_all_columns: bool = True) -> list[str]:
    """Errors (empty list = valid) for a long table against plan 2.4 and the L2 rule that
    every non-observed value carries an sd and a method."""
    errs = []
    cols = set(df.columns)
    missing = [c for c in LONG_TABLE_COLUMNS if c not in cols]
    if missing and require_all_columns:
        errs.append(f"missing columns {missing}")
    if "flag" not in cols:
        return errs + ["no flag column"]
    f = df["flag"]
    if f.isna().any():
        errs.append(f"{int(f.isna().sum())} rows with null flag")
    bad = sorted(set(int(x) for x in f.dropna().unique()) - VALID_FLAGS)
    if bad:
        errs.append(f"invalid flag values {bad}")
    nonobs = f.isin(sorted(NON_OBSERVED_FLAGS))
    if "sd" in cols:
        n = int((nonobs & df["sd"].isna()).sum())
        if n:
            errs.append(f"{n} non-observed rows (flags 2-6) without sd")
        neg = int((df["sd"] < 0).sum())
        if neg:
            errs.append(f"{neg} rows with negative sd")
    if "method" in cols:
        n = int((nonobs & (df["method"].isna() | (df["method"].astype(str) == ""))).sum())
        if n:
            errs.append(f"{n} non-observed rows (flags 2-6) without method")
    if "value" in cols:
        miss = f == int(Flag.MISSING)
        n = int((~miss & df["value"].isna()).sum())
        if n:
            errs.append(f"{n} rows with null value but flag != 8")
    return errs


def validate_grid(value: np.ndarray, sd: np.ndarray, flag: np.ndarray,
                  dates=None) -> list[str]:
    """Errors for an L3 grid triple (same shape; flag uint8 in the table; sd >= 0 where the
    value is not observed; last axis matches ``dates`` when given)."""
    errs = []
    if value.shape != sd.shape or value.shape != flag.shape:
        errs.append(f"shape mismatch value{value.shape} sd{sd.shape} flag{flag.shape}")
        return errs
    if flag.dtype != np.uint8:
        errs.append(f"flag dtype {flag.dtype} != uint8")
    bad = sorted(set(np.unique(flag).tolist()) - VALID_FLAGS)
    if bad:
        errs.append(f"invalid flag values {bad}")
    nonobs = np.isin(flag, sorted(NON_OBSERVED_FLAGS))
    if np.any(nonobs & ~np.isfinite(sd)):
        errs.append("non-observed cells without finite sd")
    if np.any(np.isfinite(sd) & (sd < 0)):
        errs.append("negative sd")
    if dates is not None and value.shape[-1] != len(dates):
        errs.append(f"last axis {value.shape[-1]} != len(dates) {len(dates)}")
    return errs
