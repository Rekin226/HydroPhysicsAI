"""Conservative, offline import of public field observations discovered via AquaScope.

These are candidate observations, not approved validation targets. Keep raw source
files alongside their hashes. Acquisition time is the earliest *verified* availability
of this exact revision; historical publication times must not be inferred from dates.
No network access, interpolation, model fitting, or automatic cache replacement occurs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TAIPEI = "Asia/Taipei"


def _read(path: str | Path, acquired_at: str) -> tuple[dict, dict]:
    raw = Path(path).read_bytes()
    acquired = pd.Timestamp(acquired_at)
    if pd.isna(acquired) or acquired.tzinfo is None:
        raise ValueError("acquired_at must include a timezone")
    return json.loads(raw), {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "available_at": acquired.tz_convert("UTC"),
        "availability_basis": "retrieval_of_this_revision",
        "independent_holdout": False,
        "validation_approved": False,
    }


def _day(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("missing observation date")
    stamp = stamp.tz_localize(TAIPEI) if stamp.tzinfo is None else stamp.tz_convert(TAIPEI)
    if stamp != stamp.normalize():
        raise ValueError("daily observation must be stamped at local midnight")
    return stamp


def _finish(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise ValueError("source returned no observations")
    out = pd.DataFrame(rows).sort_values(["station_id", "period_start"])
    if out.duplicated(["station_id", "period_start"]).any():
        raise ValueError("duplicate station-date; resolve source revisions explicitly")
    if (out.period_end > out.available_at).any():
        raise ValueError("acquired_at precedes the end of an observation period")
    return out.reset_index(drop=True)


def read_codis_rainfall(
    path: str | Path, *, station_id: str, start: str, end: str, acquired_at: str,
) -> pd.DataFrame:
    """Import daily CODIS rain with flags preserved and doubtful totals masked.

    Bounds are inclusive. A positive ``Accumulationf`` receives CWA's missing-value
    star in the official UI; its numeric magnitude is not interpreted here. Keep
    those partial totals in value_raw, but exclude them from numeric use pending QA.
    Trace (-9.8) remains an interval-censored amount, never an invented zero.
    """
    payload, provenance = _read(path, acquired_at)
    if payload.get("metadata", {}).get("timezone") != "+08:00":
        raise ValueError("unrecognized CODIS timezone")
    lo, hi = _day(start), _day(end)
    if lo > hi:
        raise ValueError("start must precede end")
    rows = []
    for block in payload.get("data", []):
        if block.get("StationID") != station_id:
            raise ValueError("CODIS station mismatch")
        for record in block.get("dts", []):
            date = _day(record["DataDate"])
            if not lo <= date <= hi:
                raise ValueError("CODIS returned a date outside the requested window")
            rain = record.get("Precipitation") or {}
            raw = rain.get("Accumulation")
            value = pd.to_numeric(raw, errors="coerce")
            flag = rain.get("Accumulationf")
            flag_number = pd.to_numeric(flag, errors="coerce")
            if not np.isfinite(value):
                status = "missing_or_nonfinite"
            elif value == -9.8:
                status = "trace_less_than_0.5_mm"
            elif value < 0:
                status = "source_sentinel"
            elif not np.isfinite(flag_number) or flag_number < 0:
                status = "flag_unconfirmed"
            elif flag_number > 0:
                status = "source_missing_value_mark"
            else:
                status = "numeric_unmarked"
            rows.append({
                **provenance, "source": "CWA_CODIS", "station_id": station_id,
                "period_start": date, "period_end": date + pd.Timedelta(days=1),
                "value_raw": raw, "source_flag": flag, "unit": "mm/day",
                "value": float(value) if status == "numeric_unmarked" else np.nan,
                "qa_status": status,
            })
    return _finish(rows)


def read_gweb_groundwater(
    path: str | Path, *, station_id: str, start: str, end: str, acquired_at: str,
) -> pd.DataFrame:
    """Import an inclusive daily WRA chart window, preserving provisional status.

    The response is an undated array. Reject an unexpected length rather than
    silently shifting or extending dates. Head is in source-reported metres; no
    conversion to drawdown or depth below land occurs. Datum version, screens and
    manual quality review are not established by this endpoint.
    """
    payload, provenance = _read(path, acquired_at)
    if payload.get("StationID") != station_id:
        raise ValueError("WRA station mismatch")
    lo, hi = _day(start), _day(end)
    if lo > hi:
        raise ValueError("start must precede end")
    dates = pd.date_range(lo, hi, freq="D")
    values = payload.get("WaterLevelData")
    if not isinstance(values, list) or len(values) != len(dates):
        raise ValueError("undated WRA array length differs from requested days")
    rows = []
    for date, raw in zip(dates, values, strict=True):
        value = pd.to_numeric(raw, errors="coerce")
        usable = np.isfinite(value) and value > -9990
        rows.append({
            **provenance, "source": "WRA_GWEB", "station_id": station_id,
            "period_start": date, "period_end": date + pd.Timedelta(days=1),
            "value_raw": raw, "value": float(value) if usable else np.nan, "unit": "m",
            "qa_status": "not_manually_checked" if usable else "missing_or_sentinel",
            "datum_status": "version_unconfirmed", "screen_status": "unconfirmed",
        })
    return _finish(rows)


def observations_available_at(observations: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Exclude revisions not yet acquired and observation periods not yet completed.

    This only enforces timing. It does not approve quality, provenance independence,
    or the observation's fitness as a forecast target.
    """
    cutoff = pd.Timestamp(origin)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("forecast origin must include a timezone")
    return observations.loc[
        (observations.available_at <= cutoff) & (observations.period_end <= cutoff)
    ].copy()
