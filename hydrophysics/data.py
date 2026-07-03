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

from .config import DEFAULT_SPLIT_DATE, Config, default_config

# Static per-well features the neural operator conditions on. Missing columns are
# zero-filled by the model (see models/gru.py::_static_features); validate() reports them.
_OPERATOR_STATIC_FEATURES = [
    "tm_x", "tm_y", "is_coastal", "dist_to_coast_m", "dom_amp",
    "ups_lag_days", "rf_lag_days",
]


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
    crs: str = "EPSG:3826"   # CRS of attrs.tm_x/tm_y; used only by the ET driver

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

    @classmethod
    def from_arrays(cls, *, well_ids, dates, target, rainfall, upstream=None,
                    attrs=None, split_date: str = DEFAULT_SPLIT_DATE,
                    crs: str = "EPSG:3826") -> GWData:
        """Build a :class:`GWData` from in-memory arrays, no CSV files needed.

        The derived fields (day-of-year, train/val masks) are computed for you.

        Args:
            well_ids: sequence of W well ids.
            dates: sequence of T dates (anything ``pd.to_datetime`` accepts).
            target, rainfall: ``(W, T)`` arrays (NaN where missing).
            upstream: ``(W, T)`` upstream-well driver, or None for no coupling (all-NaN).
            attrs: per-well static features as a DataFrame (indexed by well id, or with an
                ``st_id``/``well_id`` column) or a dict of column -> values aligned to
                ``well_ids``. Include ``tm_x``/``tm_y`` (needed by the ET driver and spatial
                features); a ``group`` column ("coastal"/"inland") derives ``is_coastal``.
                Missing optional features are handled by the operator (zero-filled).
            split_date: calibration/validation boundary (ISO date string).
            crs: CRS of ``tm_x``/``tm_y`` (see :class:`~hydrophysics.config.Config`).
        """
        well_ids = [str(w) for w in well_ids]
        dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        W, T = len(well_ids), len(dates)
        if W == 0 or T == 0:
            raise ValueError(f"need >=1 well and >=1 date (got W={W}, T={T})")

        target = np.asarray(target, dtype=float)
        rainfall = np.asarray(rainfall, dtype=float)
        upstream = (np.full((W, T), np.nan) if upstream is None
                    else np.asarray(upstream, dtype=float))
        for nm, arr in (("target", target), ("rainfall", rainfall), ("upstream", upstream)):
            if arr.shape != (W, T):
                raise ValueError(
                    f"{nm} must have shape (n_wells={W}, n_days={T}); got {arr.shape}")

        split = pd.Timestamp(split_date)
        return cls(
            well_ids=well_ids,
            dates=dates,
            target=target,
            rainfall=rainfall,
            upstream=upstream,
            doy=dates.dayofyear.to_numpy().astype(int),
            train_mask=np.asarray(dates < split),
            val_mask=np.asarray(dates >= split),
            attrs=_normalize_attrs(attrs, well_ids),
            split_date=str(split_date),
            crs=crs,
        )

    def validate(self) -> list[str]:
        """Preflight check: return human-readable issues with this dataset (empty = OK).

        Non-raising, so call it and print the result before training. Flags shape
        mismatches, unsorted/duplicate dates, an empty train or validation split, wells
        with no finite target/rainfall, missing static-feature columns (which the operator
        would silently zero-fill), and non-finite coordinates.
        """
        issues: list[str] = []
        W, T = self.n_wells, self.n_days
        for nm, arr in (("target", self.target), ("rainfall", self.rainfall),
                        ("upstream", self.upstream)):
            if arr.shape != (W, T):
                issues.append(f"{nm} shape {arr.shape} != (n_wells={W}, n_days={T})")
        if not self.dates.is_monotonic_increasing:
            issues.append("dates are not sorted ascending")
        if self.dates.has_duplicates:
            issues.append("dates contain duplicates")
        if not self.train_mask.any():
            issues.append(f"no training days before split_date {self.split_date}")
        if not self.val_mask.any():
            issues.append(f"no validation days on/after split_date {self.split_date}")

        def _all_nan(arr):
            return [self.well_ids[i] for i in range(W) if not np.isfinite(arr[i]).any()]

        if self.target.shape == (W, T) and (bad := _all_nan(self.target)):
            issues.append(f"{len(bad)} well(s) have no finite target values: {bad[:5]}")
        if self.rainfall.shape == (W, T) and (bad := _all_nan(self.rainfall)):
            issues.append(f"{len(bad)} well(s) have no finite rainfall: {bad[:5]}")

        missing = [c for c in _OPERATOR_STATIC_FEATURES if c not in self.attrs.columns]
        if missing:
            issues.append(
                "attrs missing static feature column(s) (the operator will zero-fill "
                f"these, degrading conditioning): {missing}")
        for c in ("tm_x", "tm_y"):
            if c in self.attrs.columns:
                v = pd.to_numeric(self.attrs[c], errors="coerce").to_numpy()
                if not np.isfinite(v).all():
                    issues.append(f"attrs['{c}'] has missing/non-numeric values")
        return issues


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
        crs=cfg.crs,
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


def _normalize_attrs(attrs, well_ids: list[str]) -> pd.DataFrame:
    """Coerce a user-supplied attrs (DataFrame / dict / None) to a per-well frame indexed
    by well id, deriving ``is_coastal`` from a ``group`` column when present."""
    well_ids = [str(w) for w in well_ids]
    if attrs is None:
        df = pd.DataFrame(index=pd.Index(well_ids, name="st_id"))
    else:
        df = attrs.copy() if isinstance(attrs, pd.DataFrame) else pd.DataFrame(dict(attrs))
        idcol = next((c for c in ("st_id", "well_id", "id") if c in df.columns), None)
        if idcol is not None:
            df = df.set_index(df[idcol].astype(str)).drop(columns=[idcol])
        elif df.index.name in ("st_id", "well_id", "id"):
            df.index = df.index.astype(str)
        elif len(df) == len(well_ids):
            df.index = pd.Index(well_ids, name="st_id")
        df.index = df.index.astype(str)
    if "group" in df.columns and "is_coastal" not in df.columns:
        df["is_coastal"] = (df["group"].astype(str) == "coastal").astype(int)
    return df.reindex(well_ids)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s) {missing}. Present: {list(df.columns)}. "
            f"Use column_map to alias your column names (see docs/DATA_FORMAT.md).")


def load_dataset_from_frames(gw_long, rf_long, stations, pairing=None, *,
                             column_map: dict | None = None,
                             split_date: str = DEFAULT_SPLIT_DATE,
                             crs: str = "EPSG:3826") -> GWData:
    """Build a :class:`GWData` from tidy long-format tables — "declare your columns"
    instead of matching the wide CSV schema byte-for-byte.

    Args:
        gw_long: long groundwater levels, canonical columns ``date, st_id, level``.
        rf_long: long rainfall, canonical columns ``date, rf_id, rainfall``.
        stations: one row per well, canonical columns ``st_id, tm_x, tm_y`` (plus any
            optional attributes such as ``group``, ``dist_to_coast_m`` — carried through).
        pairing: optional per-well topology with columns ``st_id`` and any of
            ``ups_id, rf_id, ups_lag_days, lag_days, group``. If omitted, each well's
            rainfall gauge is assumed to share its ``st_id`` and there is no upstream
            coupling.
        column_map: optional ``{your_column: canonical_name}`` rename applied to every
            input frame before parsing (e.g. ``{"DateTime": "date", "WellID": "st_id"}``).
        split_date, crs: as in :meth:`GWData.from_arrays`.
    """
    def _prep(df):
        return (pd.DataFrame(df).rename(columns=column_map) if column_map
                else pd.DataFrame(df).copy())

    gw_long, rf_long, stations = _prep(gw_long), _prep(rf_long), _prep(stations)
    _require_cols(gw_long, ["date", "st_id", "level"], "gw_long")
    _require_cols(rf_long, ["date", "rf_id", "rainfall"], "rf_long")
    _require_cols(stations, ["st_id", "tm_x", "tm_y"], "stations")

    def _wide(df, id_col, val_col):
        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        w = df.pivot_table(index="date", columns=id_col, values=val_col, aggfunc="mean")
        w.index = pd.DatetimeIndex(w.index).normalize()
        return w.resample("D").mean().sort_index()

    gw = _wide(gw_long, "st_id", "level")
    rf = _wide(rf_long, "rf_id", "rainfall")

    stations["st_id"] = stations["st_id"].astype(str)
    well_ids = stations["st_id"].tolist()

    pmap = None
    if pairing is not None:
        pairing = _prep(pairing)
        _require_cols(pairing, ["st_id"], "pairing")
        pairing["st_id"] = pairing["st_id"].astype(str)
        pmap = pairing.set_index("st_id")

    dates = pd.DatetimeIndex(gw.index.union(rf.index)).sort_values()
    gw, rf = gw.reindex(dates), rf.reindex(dates)

    W, T = len(well_ids), len(dates)
    target = np.full((W, T), np.nan)
    rainfall = np.full((W, T), np.nan)
    upstream = np.full((W, T), np.nan)
    for i, st in enumerate(well_ids):
        if st in gw.columns:
            target[i] = gw[st].to_numpy()
        rid, ups = st, None
        if pmap is not None and st in pmap.index:
            r = pmap.loc[st].get("rf_id", np.nan)
            rid = str(r) if pd.notna(r) else st
            u = pmap.loc[st].get("ups_id", np.nan)
            ups = str(u) if pd.notna(u) else None
        if rid in rf.columns:
            rainfall[i] = rf[rid].to_numpy()
        if ups and ups in gw.columns:
            upstream[i] = gw[ups].to_numpy()

    attrs = stations.set_index("st_id")
    if pmap is not None:
        for src, dst in (("ups_lag_days", "ups_lag_days"), ("lag_days", "rf_lag_days"),
                         ("group", "group"), ("ups_id", "ups_id"), ("rf_id", "rf_id")):
            if src in pmap.columns and dst not in attrs.columns:
                attrs[dst] = pmap[src]

    return GWData.from_arrays(well_ids=well_ids, dates=dates, target=target,
                              rainfall=rainfall, upstream=upstream, attrs=attrs,
                              split_date=split_date, crs=crs)
