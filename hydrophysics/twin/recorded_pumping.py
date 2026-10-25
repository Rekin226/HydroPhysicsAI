"""Recorded-pumping arm of a frozen forward run: a month-by-month multiplier on the metered
pumping energy past the forecast origin, everything else frozen.

The frozen forecast (``forward``) runs the projection on month-of-year climatology of the
recorded electricity. A policy-response test (N2 draft, 2026-10-24) needs a second arm of
the *same* frozen run that differs only in pumping: the climatology times the recorded
ratio ``r(t)`` of each month to its climatological mean. This module is that multiplier.

    month,multiplier                 one multiplier for every class (and every cell, or the
    2023-01,1.07                     cells of one county with ``county="yunlin"``)
    2023-02,0.94

    month,irrigation,aquaculture     per lever class (``scenario.CLASSES``); a class not
    2023-01,1.07,1.00                named keeps 1

An ``.npz`` holds ``month`` (``"YYYY-MM"`` strings) and either ``multiplier`` -- shape
``(n_months,)``, or ``(n_months, n_active)`` for a per-cell field -- or one ``(n_months,)``
array per class; optionally ``start`` (``"YYYY-MM"``) and ``n_active`` (the grid it was
built on).

Rules, all enforced, because a registered test rides on them:

- the multiplier acts only on months ``>= start``, and ``start`` must be inside the
  projection (on or after the first month past the origin): the recorded hindcast and the
  origin restart are never touched;
- a row dated before ``start`` is refused, not silently dropped;
- every month must be a calendar month on the projection's monthly grid (``YYYY-MM``; a
  day other than the 1st, a duplicate, or a month past the horizon is refused);
- a per-cell field or an ``n_active`` that does not match the model grid is refused;
- a month of the projection with no row keeps multiplier 1 (the N2 rule for a month not
  yet published), and so does every month after the last row;
- multipliers must be finite and >= 0.

Only the electricity-driven (metered) abstraction is scaled. The extra abstraction field
(``inputs.load_extra_abstraction``), canal deliveries, recharge, theta and the column are
untouched. A multiplier of exactly 1 leaves the energy field bit-identical.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .scenario import CLASSES

ALL = "*"                       # key of a multiplier that applies to every class
COUNTIES = ("yunlin", "changhua")


def _month(value, what: str) -> pd.Timestamp:
    """``"YYYY-MM"`` (or a first-of-month date) -> the month-start Timestamp; anything
    off the monthly grid is refused."""
    s = str(value).strip()
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{what}: {s!r} is not a month (expected YYYY-MM)") from exc
    if ts.day != 1 or ts.hour or ts.minute or ts.second or ts.nanosecond or ts.microsecond:
        raise ValueError(f"{what}: {s!r} is not on the monthly grid (expected YYYY-MM or "
                         "the 1st of the month)")
    return pd.Timestamp(ts.year, ts.month, 1)


@dataclass(frozen=True)
class RecordedMultiplier:
    """A monthly multiplier series on the metered pumping energy.

    ``values`` maps ``ALL`` or a class name to ``(n_months,)`` multipliers (``ALL`` may also
    be ``(n_months, n_active)``, a per-cell field). ``county`` restricts the multiplier to
    the cells of one county; it needs :meth:`bind` with the model's county flags.
    """

    months: pd.DatetimeIndex
    start: pd.Timestamp
    values: dict
    county: str | None = None
    source: str = "in-memory"
    sha256: str = ""
    n_active: int | None = None            # the grid the series declares (npz) or is bound to
    cell_mask: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.months, pd.DatetimeIndex):
            raise ValueError("months must be a DatetimeIndex")
        if len(self.months) == 0:
            raise ValueError(f"{self.source}: no months")
        if self.months.has_duplicates:
            dup = sorted({m.strftime("%Y-%m") for m in self.months[self.months.duplicated()]})
            raise ValueError(f"{self.source}: duplicate month(s) {dup}")
        if not self.months.is_monotonic_increasing:
            raise ValueError(f"{self.source}: months must be in increasing order")
        early = self.months[self.months < self.start]
        if len(early):
            raise ValueError(
                f"{self.source}: {len(early)} row(s) dated before the start month "
                f"{self.start:%Y-%m} (first {early[0]:%Y-%m}); the multiplier acts only from "
                "its start, so a row before it is refused rather than dropped")
        if not self.values:
            raise ValueError(f"{self.source}: no multiplier column")
        if ALL in self.values and len(self.values) > 1:
            raise ValueError(f"{self.source}: give either 'multiplier' or per-class columns, "
                             "not both")
        bad = sorted(k for k in self.values if k != ALL and k not in CLASSES)
        if bad:
            raise ValueError(f"{self.source}: unknown class column(s) {bad}; expected "
                             f"'multiplier' or a subset of {CLASSES}")
        if self.county is not None and self.county not in COUNTIES:
            raise ValueError(f"{self.source}: county must be one of {COUNTIES}, got "
                             f"{self.county!r}")
        n = len(self.months)
        for k, v in self.values.items():
            v = np.asarray(v)
            per_cell = k == ALL and v.ndim == 2
            if v.shape[0] != n or not (v.ndim == 1 or per_cell):
                raise ValueError(f"{self.source}: column {k!r} has shape {v.shape}, expected "
                                 f"({n},)" + (" or (n_months, n_active)" if k == ALL else ""))
            if not np.isfinite(v).all():
                raise ValueError(f"{self.source}: column {k!r} has a missing or non-finite "
                                 "multiplier; write 1 for a month without a record")
            if (v < 0).any():
                raise ValueError(f"{self.source}: column {k!r} has a negative multiplier")
            if per_cell:
                if self.county is not None:
                    raise ValueError(f"{self.source}: a per-cell field cannot also take a "
                                     "county scope")
                if self.n_active is not None and v.shape[1] != self.n_active:
                    raise ValueError(f"{self.source}: per-cell field has {v.shape[1]} cells "
                                     f"but declares n_active {self.n_active} (wrong grid)")

    # -----------------------------------------------------------------------------------
    @property
    def per_cell(self) -> bool:
        return ALL in self.values and np.asarray(self.values[ALL]).ndim == 2

    def bind(self, n_active: int, county_flags: np.ndarray | None = None
             ) -> RecordedMultiplier:
        """Fix the series to the model grid: ``n_active`` cells, and (for a county scope)
        the model's Changhua flags ``(n_active,)`` bool. Refuses a series declared or
        built for another grid."""
        if self.n_active is not None and int(self.n_active) != int(n_active):
            raise ValueError(f"{self.source}: built for {self.n_active} active cells, the "
                             f"model grid has {n_active} (wrong grid)")
        if self.per_cell and np.asarray(self.values[ALL]).shape[1] != n_active:
            raise ValueError(f"{self.source}: per-cell field has "
                             f"{np.asarray(self.values[ALL]).shape[1]} cells, the model grid "
                             f"has {n_active} (wrong grid)")
        mask = None
        if self.county is not None:
            if county_flags is None:
                raise ValueError(f"{self.source}: county={self.county!r} needs the model's "
                                 "county flags")
            ch = np.asarray(county_flags, bool)
            if ch.shape != (n_active,):
                raise ValueError(f"{self.source}: county flags have shape {ch.shape}, the "
                                 f"model grid has {n_active} cells (wrong grid)")
            mask = ch if self.county == "changhua" else ~ch
        return replace(self, n_active=int(n_active), cell_mask=mask)

    def check_window(self, dates: pd.DatetimeIndex) -> None:
        """Refuse a series that does not fit the projection months ``dates`` (the months
        past the origin): a start before them, a month past the horizon, or a month that
        is not on their grid."""
        dates = pd.DatetimeIndex(dates)
        if len(dates) == 0:
            raise ValueError("no projection months")
        if self.start < dates[0]:
            raise ValueError(
                f"{self.source}: start {self.start:%Y-%m} is before the first projected month "
                f"{dates[0]:%Y-%m}; the recorded-pumping arm acts only past the forecast "
                "origin (the hindcast and the restart stay frozen)")
        missing = self.months[~self.months.isin(dates)]
        if len(missing):
            raise ValueError(
                f"{self.source}: {len(missing)} month(s) not on the projection's monthly grid "
                f"{dates[0]:%Y-%m}..{dates[-1]:%Y-%m} (first {missing[0]:%Y-%m}); extend the "
                "horizon or trim the series")

    def factor(self, name: str, dates: pd.DatetimeIndex, n_cells: int) -> np.ndarray | None:
        """The multiplier field ``(n_cells, len(dates))`` for class ``name``, or ``None``
        when it is exactly 1 everywhere (the caller then leaves the energy untouched)."""
        self.check_window(dates)
        if self.n_active is not None and n_cells != self.n_active:
            raise ValueError(f"{self.source}: energy field has {n_cells} cells, the series "
                             f"is bound to {self.n_active} (wrong grid)")
        if (self.per_cell or self.county is not None) and self.n_active is None:
            raise ValueError(f"{self.source}: a per-cell or county-scoped series must be "
                             "bound to the model grid first (RecordedMultiplier.bind)")
        key = ALL if ALL in self.values else name
        if key not in self.values:
            return None
        v = np.asarray(self.values[key], dtype="float64")
        cols = dates.get_indexer(self.months)               # every month is on the grid
        out = np.ones((n_cells, len(dates)), dtype="float64")
        if v.ndim == 2:
            out[:, cols] = v.T
        else:
            out[:, cols] = v[None, :]
        if self.cell_mask is not None:
            out[~self.cell_mask, :] = 1.0                   # exact 1 outside the county
        if (out == 1.0).all():
            return None
        return out

    def scale(self, name: str, E: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        """``E`` ``(A, T)`` of class ``name`` on months ``dates`` under the series. Months
        and cells whose multiplier is 1 keep their values bit for bit."""
        f = self.factor(name, pd.DatetimeIndex(dates), E.shape[0])
        if f is None:
            return E
        out = np.array(E, dtype=np.result_type(E, np.float64), copy=True)
        hit = f != 1.0
        out[hit] = out[hit] * f[hit]
        return out

    def describe(self) -> str:
        cols = ("all classes" if ALL in self.values
                else "/".join(k for k in CLASSES if k in self.values))
        s = (f"recorded monthly multiplier ({cols}"
             + (", per cell" if self.per_cell else "")
             + (f", {self.county} cells" if self.county else "")
             + f") from {self.start:%Y-%m}, {len(self.months)} month(s) to "
             f"{self.months[-1]:%Y-%m}, {self.source}")
        if self.sha256:
            s += f" sha256 {self.sha256[:12]}"
        return s

    def record(self) -> dict:
        """What a run writes into its options: enough to re-identify the series."""
        return {"source": self.source, "sha256": self.sha256,
                "start": f"{self.start:%Y-%m}", "first": f"{self.months[0]:%Y-%m}",
                "last": f"{self.months[-1]:%Y-%m}", "n_months": int(len(self.months)),
                "columns": sorted(self.values), "county": self.county,
                "per_cell": bool(self.per_cell), "n_active": self.n_active}


def from_series(months, values: dict, start=None, county: str | None = None,
                source: str = "in-memory", sha256: str = "",
                n_active: int | None = None) -> RecordedMultiplier:
    """Build a series from ``months`` (``"YYYY-MM"``) and ``{column: array}`` where column
    is ``"multiplier"`` or a class name. ``start`` defaults to the first month."""
    idx = pd.DatetimeIndex([_month(m, f"{source}: month") for m in months])
    vals = {}
    for k, v in values.items():
        vals[ALL if k == "multiplier" else k] = np.asarray(v, dtype="float64")
    st = _month(start, f"{source}: start") if start is not None else (
        idx.min() if len(idx) else None)
    if st is None:
        raise ValueError(f"{source}: no months")
    return RecordedMultiplier(months=idx, start=st, values=vals, county=county,
                              source=source, sha256=sha256, n_active=n_active)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_recorded_multiplier(path: str, start=None, county: str | None = None
                             ) -> RecordedMultiplier:
    """Read a ``.csv`` or ``.npz`` series (module docstring). ``start`` overrides the
    file's own ``start`` (npz) and defaults to the first month."""
    sha = _file_sha256(path)
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            if "month" not in keys:
                raise ValueError(f"{path}: needs a 'month' array of YYYY-MM strings")
            months = [str(m) for m in z["month"]]
            cols = {k: z[k] for k in keys - {"month", "start", "n_active"}}
            st = start if start is not None else (str(z["start"]) if "start" in keys
                                                  else None)
            n_active = int(z["n_active"]) if "n_active" in keys else None
    else:
        df = pd.read_csv(path, dtype={"month": str})
        if "month" not in df.columns:
            raise ValueError(f"{path}: needs a 'month' column (YYYY-MM)")
        months = df["month"].tolist()
        cols = {k: df[k].to_numpy(dtype="float64") for k in df.columns if k != "month"}
        st, n_active = start, None
    unknown = sorted(k for k in cols if k != "multiplier" and k not in CLASSES)
    if unknown:
        raise ValueError(f"{path}: unknown column(s) {unknown}; expected 'multiplier' or a "
                         f"subset of {CLASSES}")
    return from_series(months, cols, start=st, county=county, source=str(path),
                       sha256=sha, n_active=n_active)


def parse_recorded_spec(text: str) -> tuple[str, str, str | None, str | None]:
    """``"NAME=PATH[@YYYY-MM][#county]"`` -> ``(name, path, start, county)``."""
    if "=" not in text:
        raise ValueError(f"--recorded-pumping {text!r}: expected 'NAME=PATH[@YYYY-MM]"
                         "[#yunlin|#changhua]'")
    name, rest = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--recorded-pumping {text!r}: empty arm name")
    county = None
    if "#" in rest:
        rest, county = rest.rsplit("#", 1)
        county = county.strip()
    start = None
    if "@" in rest:
        rest, start = rest.rsplit("@", 1)
        start = start.strip()
    return name, rest.strip(), start, county
