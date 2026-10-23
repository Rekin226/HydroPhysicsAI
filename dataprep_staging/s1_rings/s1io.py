"""Guarded reads of the upstream products used by WP S1a."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from common import guard  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402

DAILY_COLS = ["sensor_id", "feed", "date", "value", "flag", "variant"]


def pile_table(preorigin: bool) -> tuple[pd.DataFrame, list[str]]:
    """deep_benchmarks/v2 daily rows. ``preorigin=True``: design_time_evidence through the
    2022-12 time slice (rows of the pre-origin run only), for the pre-origin tests;
    ``False``: after_the_fact_analysis (all rows), for the post-origin product."""
    if preorigin:
        prod = guard.open_product(K.DEEP_BENCH_V2, "design_time_evidence", window_end="2022-12")
    else:
        prod = guard.open_product(K.DEEP_BENCH_V2, "after_the_fact_analysis")
    df = prod.read_table("daily.parquet", columns=DAILY_COLS)
    return df, list(prod.notes)


def piles(preorigin: bool) -> tuple[dict, list[str]]:
    """sensor -> daily vendor Series (mm, compression positive)."""
    df, notes = pile_table(preorigin)
    sensors = sorted({v[0] for v in K.PILES.values()})
    return {s: L.pile_daily(df, s) for s in sensors}, notes


def printed_monthly() -> tuple[pd.DataFrame, list[str]]:
    """deep_benchmarks_printed/v2 printed_monthly opened as a validator (gate passed)."""
    prod = guard.open_product(K.PRINTED_V2, "validator")
    df = prod.read_table("printed_monthly.parquet",
                         columns=["sensor_id", "month", "value", "flag", "depth_m"])
    return df, list(prod.notes)


def gnss_monthly() -> tuple[pd.DataFrame, list[str]]:
    """gnss_clean/full/v1 WRA monthly heights, after_the_fact_analysis (gate failed)."""
    prod = guard.open_product(K.GNSS_FULL_V1, "after_the_fact_analysis")
    df = prod.read_table("wra/monthly.parquet",
                         columns=["station", "date", "value", "sd", "flag", "n_days"])
    return df, list(prod.notes)
