"""Load the groundwater modeling dataset from the CSV inputs (see docs/DATA_FORMAT.md).

Produces a :class:`GWData` bundle of daily, well-aligned arrays ready for both the
baselines and (later) the neural models. Hourly groundwater levels are resampled to
daily means to match the gray-box ODE (dt = 1 day); rainfall is already daily.

Shapes: with ``W`` active wells and ``T`` days,
    target, rainfall, upstream : (W, T) float arrays  (NaN where missing)
    doy                        : (T,) int array (day-of-year)
    train_mask, val_mask       : (T,) bool arrays
    attrs                      : DataFrame indexed by well id (static features)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, default_config


@dataclass
class GWData:
    well_ids: list[str]
    dates: pd.DatetimeIndex
    target: np.ndarray      # (W, T) daily groundwater level, the prediction target
    rainfall: np.ndarray    # (W, T) paired daily rainfall driver
    upstream: np.ndarray    # (W, T) paired upstream daily groundwater level driver
    doy: np.ndarray         # (T,) day-of-year (1..366) for the seasonal cycle
    train_mask: np.ndarray  # (T,) calibration period
    val_mask: np.ndarray    # (T,) validation period
    attrs: pd.DataFrame      # static per-well attributes, indexed by well id
    split_date: str

    @property
    def n_wells(self) -> int:
        return len(self.well_ids)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def well_index(self, well_id: str) -> int:
        return self.well_ids.index(well_id)

    def summary(self) -> str:
        valid = np.isfinite(self.target)
        return (
            f"GWData: {self.n_wells} wells x {self.n_days} days "
            f"({self.dates[0].date()} to {self.dates[-1].date()}), "
            f"split {self.split_date} "
            f"[train {int(self.train_mask.sum())}d / val {int(self.val_mask.sum())}d], "
            f"target coverage {100 * valid.mean():.1f}%"
        )


def _read_csv_bom(path) -> pd.DataFrame:
    """Read a CSV that may carry a UTF-8 BOM on the first column name."""
    return pd.read_csv(path, encoding="utf-8-sig")


def _daily_gw(path) -> pd.DataFrame:
    """Read hourly groundwater levels and resample to daily means, indexed by date."""
    df = pd.read_csv(path)
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    daily = df.resample("D").mean()
    daily.index = daily.index.normalize()
    return daily


def _daily_rf(path) -> pd.DataFrame:
    df = _read_csv_bom(path)
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    df.index = df.index.normalize()
    return df


def load_dataset(cfg: Config | None = None) -> GWData:
    """Load and align the full modeling dataset.

    Active wells (``active == 1`` in gray_box_input.csv) define the well set and their
    paired drivers (upstream well ``ups_id`` and rainfall gauge ``rf_id``).
    """
    cfg = cfg or default_config()

    pairing = pd.read_csv(cfg.gray_box_input)
    pairing = pairing[pairing["active"] == 1].copy()
    well_ids = pairing["st_id"].astype(str).tolist()

    gw = _daily_gw(cfg.gw_timeseries)
    rf = _daily_rf(cfg.rf_timeseries)

    # Common daily index across both sources.
    dates = gw.index.union(rf.index)
    dates = pd.DatetimeIndex(dates).sort_values()
    gw = gw.reindex(dates)
    rf = rf.reindex(dates)

    W, T = len(well_ids), len(dates)
    target = np.full((W, T), np.nan)
    rainfall = np.full((W, T), np.nan)
    upstream = np.full((W, T), np.nan)

    for i, row in enumerate(pairing.itertuples(index=False)):
        st = str(row.st_id)
        if st in gw.columns:
            target[i] = gw[st].to_numpy()
        ups = str(getattr(row, "ups_id", "")) if pd.notna(getattr(row, "ups_id", np.nan)) else ""
        if ups and ups in gw.columns:
            upstream[i] = gw[ups].to_numpy()
        rid = str(getattr(row, "rf_id", "")) if pd.notna(getattr(row, "rf_id", np.nan)) else ""
        if rid and rid in rf.columns:
            rainfall[i] = rf[rid].to_numpy()

    doy = dates.dayofyear.to_numpy().astype(int)

    split = pd.Timestamp(cfg.split_date)
    train_mask = np.asarray(dates < split)
    val_mask = np.asarray(dates >= split)

    attrs = _build_attrs(cfg, pairing, well_ids)

    return GWData(
        well_ids=well_ids,
        dates=dates,
        target=target,
        rainfall=rainfall,
        upstream=upstream,
        doy=doy,
        train_mask=train_mask,
        val_mask=val_mask,
        attrs=attrs,
        split_date=cfg.split_date,
    )


def _build_attrs(cfg: Config, pairing: pd.DataFrame, well_ids: list[str]) -> pd.DataFrame:
    """Assemble static per-well features for conditioning the neural operator."""
    cols = {
        "st_id": pairing["st_id"].astype(str),
        "group": pairing.get("group"),
        "ups_id": pairing.get("ups_id"),
        "rf_id": pairing.get("rf_id"),
        "ups_lag_days": pairing.get("ups_lag_days"),
        "rf_lag_days": pairing.get("lag_days"),
        "tm_x": pairing.get("gw_TM_X97"),
        "tm_y": pairing.get("gw_TM_Y97"),
    }
    attrs = pd.DataFrame(cols).set_index("st_id")

    # Merge coastal/inland classification and tidal descriptors if available.
    if cfg.coastal_inland.exists():
        ci = pd.read_csv(cfg.coastal_inland)
        ci["st_id"] = ci["st_id"].astype(str)
        keep = ["st_id", "dist_to_coast_m", "is_near_coast", "dom_freq_cpd",
                "dom_amp", "m2_amp", "is_m2_like"]
        keep = [c for c in keep if c in ci.columns]
        attrs = attrs.join(ci[keep].set_index("st_id"), how="left")

    attrs["is_coastal"] = (attrs["group"].astype(str) == "coastal").astype(int)
    return attrs.reindex([str(w) for w in well_ids])
