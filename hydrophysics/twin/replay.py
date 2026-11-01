"""Weather-year replay as projection forcing (opt-in engine; FW.weather_replay.v1).

The forward twin (``forward.py``) projects past the record under month-of-year
*climatology* of every forcing field. That is the honest default for a level, but the
compaction column is threshold-nonlinear: inelastic (permanent) compaction happens only when
the head falls below its past minimum, which mean forcing never does after 2026 (the page's
margin is 0.24 m at the 5th percentile). This module replays whole pre-origin calendar years
instead, so a dry year can draw heads to a new minimum, and records how much of a policy's
avoided subsidence is the column's inelastic term.

    python -m hydrophysics.twin.replay --projection-forcing replay:2015-2022,2012-2014 \\
        --replay-pumping detrended --replay-fields energy+weather \\
        --capture <out>.fw.npz -- <every forward.py argument, unchanged>

What it changes, and what it does not:

- ``forward.py`` is not edited (its sha256 is pinned by registered scorers). This module
  wraps ``forward.main`` and, only inside :func:`installed`, swaps six of its module-level
  names: ``future_forcing`` (energy by class and recharge), ``climatology`` (the canal
  deliveries and the canal-substitution energy, which call it), ``parse_scenario`` (adds an
  end month, ``@2026-01..2029-12``), ``compaction`` (records the column's elastic and
  inelastic parts beside the unchanged total) and ``load_columns``/``run`` (labels). Every
  name is restored on exit. ``--projection-forcing climatology`` delegates every call to the
  original functions, so its outputs are the plain forward's, bit for bit.
- A replay sequence names one source calendar year per projected calendar year, cycling:
  ``replay:2015-2022,2012-2014`` projects 2023 with 2015's months, 2024 with 2016's, ...
  Every field the climatology builds (energy per policy class, recharge = rain - ET0, canal
  deliveries) is taken from the SAME source year, so the replay never mixes years.
- Pumping (``--replay-pumping``): ``detrended`` (default) divides each class's energy by
  its fitted 2012-22 trend factor ``g_c(y)`` (ordinary least squares on the fan-total annual
  energy of the class, normalised to mean 1 over the trend years; :func:`class_trend_factors`),
  so policy steps and the census trend do not count as weather; ``raw`` replays the record
  as metered; ``climatology`` keeps the energy at climatology (a weather-only arm).
- ``--replay-fields energy`` replays the pumping only and keeps recharge and canal water at
  climatology (the arm used when the rain input carries no forcing manifest).

Nothing here reads an observation beyond what ``forward.main`` already reads. The capture
holds model output only: member-mean elastic and inelastic compaction at every December,
per-run group means, and counts of projection months in which the column's driver set a new
minimum (crossed its preconsolidation head). ``forcing_identity_gate`` is the engineering
check that a 2012..2022 replay started at 2012-01 reproduces the record's forcing arrays.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from . import forward as fwd
from .compaction import VEPColumn, aquitard_compaction
from .scenario import PumpingScenario
from .scenario import climatology as _climatology

TREND_YEARS = (2012, 2022)
G_MIN = 0.25                 # a trend factor below this refuses the detrended replay
PUMPING_MODES = ("detrended", "raw", "climatology")
FIELD_MODES = ("energy+weather", "energy")
_YEARS_RE = re.compile(r"^\d{4}(-\d{4})?(,\d{4}(-\d{4})?)*$")
_WINDOW_RE = re.compile(r"@(\d{4}-\d{2})\.\.(\d{4}-\d{2})")


# ---------------------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------------------
def parse_years(text: str) -> tuple[int, ...]:
    """``"2015-2022,2012-2014"`` -> ``(2015, ..., 2022, 2012, 2013, 2014)``."""
    s = str(text).replace(" ", "")
    if not _YEARS_RE.match(s):
        raise ValueError(f"replay years {text!r}: expected 'YYYY[-YYYY][,YYYY[-YYYY]...]'")
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        if hi < lo:
            raise ValueError(f"replay years {text!r}: inverted range {part}")
        out.extend(range(lo, hi + 1))
    return tuple(out)


def format_years(years) -> str:
    """Inverse of :func:`parse_years` (consecutive runs folded into ranges)."""
    ys = [int(y) for y in years]
    parts, i = [], 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1:
            j += 1
        parts.append(str(ys[i]) if i == j else f"{ys[i]}-{ys[j]}")
        i = j + 1
    return ",".join(parts)


def cyclic_sequences(first: int = TREND_YEARS[0], last: int = TREND_YEARS[1]
                     ) -> list[tuple[int, ...]]:
    """The ``last - first + 1`` cyclic orders of ``first..last``, one starting at each year."""
    base = list(range(int(first), int(last) + 1))
    return [tuple(base[k:] + base[:k]) for k in range(len(base))]


def source_years(fut_dates: pd.DatetimeIndex, years) -> np.ndarray:
    """The source calendar year of every projected month: projected calendar year ``k``
    (counted from the first projected month's year) replays ``years[k % len(years)]``."""
    ys = np.asarray(tuple(years), dtype="int64")
    if ys.size == 0:
        raise ValueError("an empty replay sequence")
    k = fut_dates.year.to_numpy() - int(fut_dates[0].year)
    return ys[k % ys.size]


def replay_extend(arr, dates: pd.DatetimeIndex, horizon: int, years, end: str | None = None,
                  start: pd.Timestamp | str | None = None,
                  divisor: dict[int, float] | None = None
                  ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Extend ``(..., T)`` by ``horizon`` months with the record's own months of the source
    years (``scenario.climatology``'s contract: the extension only, plus its dates).

    ``start`` (default: the month after ``dates[-1]``) moves the extension's first month;
    the engineering gate uses ``start=dates[0]``. ``end`` (``YYYY-MM``) refuses a source
    month after it (the climatology window of ``--climatology-end``). ``divisor`` maps a
    source year to the factor its values are divided by (the detrended pumping); ``None``
    copies the record's values bit for bit. A source month missing from the record, or a
    non-finite replayed value, is refused rather than filled."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    a = np.asarray(arr)
    first = dates[-1] + pd.offsets.MonthBegin(1) if start is None else pd.Timestamp(start)
    fut = pd.date_range(first, periods=int(horizon), freq="MS")
    src = source_years(fut, years)
    col = {(int(d.year), int(d.month)): i for i, d in enumerate(dates)}
    last = None if end is None else pd.Period(_check_ym(end), freq="M")
    idx = np.empty(len(fut), dtype="int64")
    for i, (y, m) in enumerate(zip(src.tolist(), fut.month.tolist(), strict=True)):
        if last is not None and pd.Period(year=y, month=m, freq="M") > last:
            raise ValueError(f"replay source {y}-{m:02d} lies after the climatology end {end}")
        if (y, m) not in col:
            raise ValueError(f"replay source {y}-{m:02d} is not in the record "
                             f"({dates[0]:%Y-%m}..{dates[-1]:%Y-%m})")
        idx[i] = col[(y, m)]
    ext = a[..., idx]
    if divisor is not None:
        f = np.array([float(divisor[int(y)]) for y in src.tolist()], dtype="float64")
        ext = ext / f
    if not np.isfinite(ext).all():
        raise ValueError("the replayed forcing holds non-finite values")
    return ext, fut


def _check_ym(s: str) -> str:
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s):
        raise ValueError(f"{s!r}: expected YYYY-MM")
    return s


def _class_lines(E_by_class: dict[str, np.ndarray], dates: pd.DatetimeIndex, first: int,
                 last: int) -> dict[str, tuple[np.ndarray, float, np.ndarray]]:
    """Per class, ``(Y, a, x)``: the fan-total annual energy ``Y`` of ``first..last``, its
    mean ``a`` and the centred years ``x``. Refuses an incomplete year or a non-finite
    value."""
    years = list(range(int(first), int(last) + 1))
    yr = dates.year.to_numpy()
    for y in years:
        if int((yr == y).sum()) != 12:
            raise ValueError(f"trend year {y} is not a complete year of the record")
    x = np.asarray(years, dtype="float64") - float(np.mean(years))
    out = {}
    for c, E in E_by_class.items():
        e = np.asarray(E, dtype="float64")
        tot = np.array([e[..., yr == y].sum() for y in years], dtype="float64")
        if not np.isfinite(tot).all():
            raise ValueError(f"class {c}: non-finite energy in the trend years")
        out[c] = (tot, float(tot.mean()), x)
    return out


def class_trend_factors(E_by_class: dict[str, np.ndarray], dates: pd.DatetimeIndex,
                        first: int = TREND_YEARS[0], last: int = TREND_YEARS[1]
                        ) -> dict[str, dict[int, float]]:
    """Per policy class, the 2012-22 trend factor of its fan-total annual energy.

    ``Y_c(y)`` is the sum over cells and months of class ``c`` in calendar year ``y``; the
    OLS line ``L_c(y) = a + b (y - ybar)`` through ``first..last`` (``a`` is the mean of
    ``Y_c``) gives ``g_c(y) = L_c(y) / a`` (mean exactly 1 over the trend years). A class
    with zero total energy gets ``g = 1``. A year that is not complete in the record, a
    non-finite value, or a factor below ``G_MIN`` is refused."""
    years = list(range(int(first), int(last) + 1))
    out: dict[str, dict[int, float]] = {}
    for c, (tot, a, x) in _class_lines(E_by_class, dates, first, last).items():
        if a == 0.0:
            out[c] = {y: 1.0 for y in years}
            continue
        if a < 0.0:
            raise ValueError(f"class {c}: negative mean annual energy")
        b = float((x * tot).sum() / (x * x).sum())
        g = (a + b * x) / a
        if float(g.min()) < G_MIN:
            raise ValueError(f"class {c}: trend factor {float(g.min()):.3f} < {G_MIN} "
                             "(the linear trend does not describe this class)")
        out[c] = {y: float(v) for y, v in zip(years, g.tolist(), strict=True)}
    return out


def class_annual_anomalies(E_by_class: dict[str, np.ndarray], dates: pd.DatetimeIndex,
                           first: int = TREND_YEARS[0], last: int = TREND_YEARS[1]
                           ) -> dict[str, dict[int, float | None]]:
    """Per class and year, ``Y_c(y) / L_c(y)``: the annual energy over its fitted line, i.e.
    what a detrended replay of year ``y`` carries as its pumping anomaly (1 = on the trend).
    A step in the record (a policy or census change) shows here as a run of ratios on one
    side of 1. ``None`` where the line is not positive. Recorded in the capture; never used
    to build forcing."""
    years = list(range(int(first), int(last) + 1))
    out: dict[str, dict[int, float | None]] = {}
    for c, (tot, a, x) in _class_lines(E_by_class, dates, first, last).items():
        b = float((x * tot).sum() / (x * x).sum()) if a != 0.0 else 0.0
        line = a + b * x
        out[c] = {y: (float(t / v) if v > 0.0 else None)
                  for y, t, v in zip(years, tot.tolist(), line.tolist(), strict=True)}
    return out


@dataclass
class ReplayForcing:
    """A declared replay: the source-year sequence, the pumping mode and the fields."""

    years: tuple[int, ...]
    pumping: str = "detrended"
    fields: str = "energy+weather"
    trend_years: tuple[int, int] = TREND_YEARS
    factors: dict[str, dict[int, float]] | None = field(default=None, repr=False)
    # engineering only (the forcing gate): the first projected month; None = the forward's
    # own (the month after the record)
    start: pd.Timestamp | None = None
    anomaly: dict[str, dict[int, float | None]] | None = field(default=None, repr=False)
    anomaly_note: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.years = tuple(int(y) for y in self.years)
        if not self.years:
            raise ValueError("an empty replay sequence")
        if self.pumping not in PUMPING_MODES:
            raise ValueError(f"pumping must be one of {PUMPING_MODES}, got {self.pumping!r}")
        if self.fields not in FIELD_MODES:
            raise ValueError(f"fields must be one of {FIELD_MODES}, got {self.fields!r}")
        if self.pumping == "climatology" and self.fields == "energy":
            raise ValueError("pumping 'climatology' with fields 'energy' replays nothing")

    @property
    def weather(self) -> bool:
        return self.fields == "energy+weather"

    def bind(self, inp) -> None:
        """Fit the trend factors on the run's record (detrended pumping only) and refuse a
        sequence whose years the record does not hold in full."""
        yr = inp.dates.year.to_numpy()
        for y in sorted(set(self.years)):
            if int((yr == y).sum()) != 12:
                raise ValueError(f"replay year {y} is not a complete year of the record "
                                 f"({inp.dates[0]:%Y-%m}..{inp.dates[-1]:%Y-%m})")
        if self.pumping == "detrended":
            self.factors = class_trend_factors(inp.E_by_class, inp.dates, *self.trend_years)
            missing = sorted(set(self.years) - set(next(iter(self.factors.values()))))
            if missing:
                raise ValueError(f"replay years {missing} lie outside the trend years "
                                 f"{self.trend_years}")
        # the replayed pumping anomaly per class and source year (record only)
        try:
            self.anomaly = class_annual_anomalies(inp.E_by_class, inp.dates, *self.trend_years)
            self.anomaly_note = None
        except ValueError as err:
            self.anomaly, self.anomaly_note = None, str(err)

    def energy(self, name: str, E, dates, horizon: int, end: str | None):
        if self.pumping == "climatology":
            return self._clim(E, dates, horizon, end)
        div = None
        if self.pumping == "detrended":
            if self.factors is None:
                raise RuntimeError("detrended replay used before bind()")
            div = self.factors[name]
        return replay_extend(E, dates, horizon, self.years, end=end, start=self.start,
                             divisor=div)

    def weather_field(self, arr, dates, horizon: int, end: str | None):
        if self.weather:
            return replay_extend(arr, dates, horizon, self.years, end=end, start=self.start)
        return self._clim(arr, dates, horizon, end)

    def _clim(self, arr, dates, horizon: int, end: str | None):
        if self.start is not None:
            raise ValueError("a start override is for the replayed fields only (the gate)")
        return _climatology(arr, dates, horizon, end=end)

    def record(self) -> dict:
        return {"years": list(self.years), "years_text": format_years(self.years),
                "pumping": self.pumping, "fields": self.fields,
                "trend_years": list(self.trend_years),
                "trend_factors": ({c: {str(y): g for y, g in f.items()}
                                   for c, f in self.factors.items()}
                                  if self.factors is not None else None),
                "annual_anomaly_Y_over_L": ({c: {str(y): v for y, v in f.items()}
                                             for c, f in self.anomaly.items()}
                                            if self.anomaly is not None else None),
                "annual_anomaly_note": self.anomaly_note}


# ---------------------------------------------------------------------------------------
# scenarios with an end month (the lift test)
# ---------------------------------------------------------------------------------------
@dataclass
class WindowedScenario(PumpingScenario):
    """``PumpingScenario`` in force from ``start`` through ``end`` (inclusive, ``YYYY-MM``)
    only; ``end=None`` is the parent exactly."""

    end: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.end is not None:
            _check_ym(self.end)
            if self.start is None:
                raise ValueError("an end month needs a start month")
            if pd.Period(self.end, freq="M") < pd.Period(self.start, freq="M"):
                raise ValueError(f"end {self.end} precedes start {self.start}")

    def _active(self, dates: pd.DatetimeIndex) -> np.ndarray:
        active = np.ones(len(dates), dtype="float64")
        if self.start is not None:
            active = (dates >= pd.Timestamp(self.start)).astype("float64")
        if self.end is not None:
            stop = pd.Period(self.end, freq="M").to_timestamp() + pd.offsets.MonthBegin(1)
            active = active * (dates < stop).astype("float64")
        return active

    def _scope(self, n_cells, dates, zone_of_cell):
        base = super()._scope(n_cells, dates, zone_of_cell)
        if self.end is None:
            return base
        return base * self._active(dates)[None, :]

    def apply(self, e_by_class, dates, zone_of_cell=None):
        if self.end is None:
            return super().apply(e_by_class, dates, zone_of_cell=zone_of_cell)
        from .zones import ZONE_NAMES, fan_band_of

        first = next(iter(e_by_class.values()))
        total = np.zeros_like(first)
        active = self._active(dates)
        cell_sel = np.ones(first.shape[0], dtype="float64")
        if self.zones is not None:
            if zone_of_cell is None:
                raise ValueError("zones= requires zone_of_cell")
            idx = [ZONE_NAMES.index(z) for z in self.zones]
            cell_sel = np.isin(fan_band_of(zone_of_cell), idx).astype("float64")
        for name, E in e_by_class.items():
            if self.recorded is not None:
                E = self.recorded.scale(name, E, dates)
            f = float(self.factors.get(name, 1.0))
            if f == 1.0:
                total = total + E
                continue
            total = total + E * (1.0 + (f - 1.0) * cell_sel[:, None] * active[None, :])
        return total

    def describe(self) -> str:
        s = super().describe()
        return s if self.end is None else f"{s} to {self.end}"


def parse_scenario(text: str) -> tuple[PumpingScenario, float]:
    """``forward.parse_scenario`` plus an end month: ``"name:irrigation=0.7@2026-01..2029-12"``.
    A string without ``..`` is parsed by the forward's own parser, unchanged."""
    m = _WINDOW_RE.search(text)
    if m is None:
        if ".." in text:
            raise ValueError(f"scenario {text!r}: an end month reads '@YYYY-MM..YYYY-MM'")
        return _FWD["parse_scenario"](text)
    base, rain = _FWD["parse_scenario"](text[:m.start()] + "@" + m.group(1) + text[m.end():])
    kw = {f.name: getattr(base, f.name) for f in dataclasses.fields(PumpingScenario)}
    return WindowedScenario(**kw, end=m.group(2)), rain


# ---------------------------------------------------------------------------------------
# the column, split into its elastic and inelastic parts
# ---------------------------------------------------------------------------------------
def vep_parts(h: torch.Tensor, log_ske, log_skv, log_tau, h_pc0, dt_days: float = 30.0
              ) -> dict[str, torch.Tensor]:
    """``compaction.vep_compaction`` with its parts kept: ``total`` (the same arithmetic, so
    bit-identical), ``elastic`` (``Ske (h0 - h)``), ``inelastic`` (the relaxed ``eps_i``) and
    ``newlow`` (bool ``(n, T)``: the head fell below the running preconsolidation head that
    month, i.e. the column loaded new inelastic strain)."""
    ske = torch.exp(log_ske)
    skv = torch.exp(log_skv)
    tau = torch.exp(log_tau)
    decay = torch.exp(-torch.tensor(dt_days, dtype=h.dtype, device=h.device) / tau)
    n, T = h.shape
    h_pc = h[:, 0] + h_pc0
    eps_i = torch.zeros(n, dtype=h.dtype, device=h.device)
    eq = torch.zeros(n, dtype=h.dtype, device=h.device)
    zero = torch.zeros(n, dtype=h.dtype, device=h.device)
    tot, el, inel = [zero], [zero], [zero]
    low = [torch.zeros(n, dtype=torch.bool, device=h.device)]
    for t in range(1, T):
        below = torch.clamp(h_pc - h[:, t], min=0.0)
        eq = eq + skv * below
        h_pc = torch.minimum(h_pc, h[:, t])
        eps_i = eq + (eps_i - eq) * decay
        eps_e = ske * (h[:, 0] - h[:, t])
        tot.append(eps_e + eps_i)
        el.append(eps_e)
        inel.append(eps_i)
        low.append(below > 0)
    return {"total": torch.stack(tot, dim=1), "elastic": torch.stack(el, dim=1),
            "inelastic": torch.stack(inel, dim=1), "newlow": torch.stack(low, dim=1)}


def _single_parts(c: VEPColumn, drv: torch.Tensor) -> dict[str, torch.Tensor]:
    p = vep_parts(drv, c.log_ske, c.log_skv, c.log_tau, c.h_pc0, c.dt_days)
    p["aquitard"] = torch.zeros_like(p["total"])
    if getattr(c, "creep", "vep") == "aquitard":
        aq = aquitard_compaction(drv, c.log_ska, c.log_tau_a, c.h_a0, c.dt_days)
        p["total"] = p["total"] + aq
        p["aquitard"] = aq
    return p


def column_parts(col: torch.nn.Module, drv: torch.Tensor) -> dict[str, torch.Tensor]:
    """The parts of ``col(drv)`` (``VEPColumn`` or ``forward.ZonalColumn``, sharp or
    blended): ``total`` equals ``col(drv)`` bit for bit; ``aquitard`` is the delayed-drainage
    term (zero for a pure VEP column), counted with the inelastic part by the scorer."""
    with torch.no_grad():
        if isinstance(col, VEPColumn):
            return _single_parts(col, drv)
        if isinstance(col, fwd.ZonalColumn):
            if col.zone_w is not None:
                if any(getattr(c, "creep", "vep") != "vep" for c in col.cols):
                    raise ValueError("a blended column carries the four VEP parameters only")
                W = col.zone_w.to(dtype=drv.dtype, device=drv.device)
                per = {k: torch.cat([getattr(c, k).reshape(1) for c in col.cols]).to(drv.dtype)
                       @ W for k in fwd.COLUMN_KEYS}
                p = vep_parts(drv, per["log_ske"], per["log_skv"], per["log_tau"],
                              per["h_pc0"], float(col.cols[0].dt_days))
                p["aquitard"] = torch.zeros_like(p["total"])
                return p
            out = {k: torch.zeros_like(drv) for k in ("total", "elastic", "inelastic",
                                                      "aquitard")}
            out["newlow"] = torch.zeros(drv.shape, dtype=torch.bool, device=drv.device)
            for z, c in enumerate(col.cols):
                sel = col.zone == z
                if sel.any():
                    for k, v in _single_parts(c, drv[sel]).items():
                        out[k][sel] = v
            return out
    raise TypeError(f"no part decomposition for a {type(col).__name__}")


def column_zone_ids(col: torch.nn.Module, n_cells: int) -> np.ndarray:
    """The column's zone id per cell (argmax of a blend's weights; 0 for a single column)."""
    if isinstance(col, fwd.ZonalColumn):
        if col.zone_w is not None:
            return col.zone_w.argmax(dim=0).cpu().numpy().astype("int64")
        return col.zone.cpu().numpy().astype("int64")
    return np.zeros(int(n_cells), dtype="int64")


# ---------------------------------------------------------------------------------------
# capture: model output only, accumulated over the forward's member runs
# ---------------------------------------------------------------------------------------
def default_county_flags(grid) -> tuple[np.ndarray | None, dict]:
    """``(flags, provenance)``: the Changhua flags of the twin grid from the cached county
    map ``zone_county.DEFAULT_MAP`` (rebuilt from the gpkg only when the cache is on another
    grid, as ``county_flags_for_grid`` does) and where they came from (``source``, the cache
    file's sha256, the flags' ``county_sha256``). On failure the flags are ``None`` and the
    provenance carries the error: the county split is a report aggregate only, and the
    scorer reports why it is missing rather than failing the run."""
    try:
        from .zone_county import DEFAULT_MAP, county_flags_for_grid, county_sha256

        flags, source = county_flags_for_grid(grid, DEFAULT_MAP)
        flags = np.asarray(flags, dtype=bool)
        file_sha = None
        if source == DEFAULT_MAP:
            with open(DEFAULT_MAP, "rb") as fh:
                file_sha = hashlib.sha256(fh.read()).hexdigest()
        return flags, {"source": source, "file_sha256": file_sha,
                       "county_sha256": county_sha256(flags), "error": None}
    except (OSError, ValueError, KeyError, ImportError) as err:
        return None, {"source": None, "file_sha256": None, "county_sha256": None,
                      "error": f"{type(err).__name__}: {err}"}


class Capture:
    """Elastic / inelastic parts of every compaction the forward computes, per scenario and
    rheology: member-mean fields at each December, per-run group means, new-minimum month
    counts by cell and projected year (summed over runs) and, per run, cell and projected
    year, whether a new minimum was set. It never changes what the forward returns."""

    def __init__(self, county_flags=default_county_flags):
        self.county_fn = county_flags
        self.county: dict = {"source": None, "error": "not bound"}
        self.bound = False
        self.scenario: str | None = None
        self.labels: dict[int, str] = {}
        self.max_total_diff = 0.0
        self._acc: dict[tuple[int, int], dict] = {}

    # -- wiring ------------------------------------------------------------------------
    def bind(self, inp, scenarios, horizon: int) -> None:
        T = len(inp.dates)
        self.origin = T - 1
        self.dates = inp.dates.append(pd.date_range(inp.dates[-1] + pd.offsets.MonthBegin(1),
                                                    periods=int(horizon), freq="MS"))
        self.ye_idx = np.array([i for i, d in enumerate(self.dates) if d.month == 12],
                               dtype="int64")
        self.years = [int(self.dates[i].year) for i in self.ye_idx]
        fut = self.dates[T:]
        self.proj_years = sorted({int(y) for y in fut.year})
        self.proj_year_of = np.asarray(fut.year, dtype="int64")
        self.scen_names = [s.name for s, _ in scenarios]
        self.n_cells = int(inp.grid.n_active)
        ch, prov = None, {"source": None, "error": "no county function"}
        if self.county_fn is not None:
            got = self.county_fn(inp.grid)
            ch, prov = got if isinstance(got, tuple) else (got, {"source": "caller-supplied",
                                                                 "error": None})
        self.changhua = None if ch is None else np.asarray(ch, dtype=bool)
        self.county = prov
        self.rheology: list[str] = []
        self.zone_ids: list[np.ndarray] = []
        self.bound = True

    def label(self, col, name: str) -> None:
        self.labels[id(col)] = name

    def _rheology_index(self, col) -> int:
        name = self.labels.get(id(col))
        if name is None:
            name = f"column{len(self.rheology)}" if self.rheology else "column"
            self.labels[id(col)] = name
        if name not in self.rheology:
            self.rheology.append(name)
            self.zone_ids.append(column_zone_ids(col, self.n_cells))
        return self.rheology.index(name)

    def groups(self, ri: int) -> tuple[list[str], np.ndarray]:
        """Group names and ``(G, A)`` masks: the fan, each column zone, then the counties."""
        z = self.zone_ids[ri]
        names, masks = ["fan"], [np.ones(self.n_cells, dtype=bool)]
        for k in range(int(z.max()) + 1):
            names.append(f"zone{k}")
            masks.append(z == k)
        if self.changhua is not None:
            names += ["yunlin_rest", "changhua"]
            masks += [~self.changhua, self.changhua]
        return names, np.stack(masks)

    # -- accumulation ------------------------------------------------------------------
    def add(self, col, heads: torch.Tensor, out: np.ndarray) -> None:
        if not self.bound:
            raise RuntimeError("Capture.add before bind (forward.run was not wrapped)")
        if self.scenario not in self.scen_names:
            raise RuntimeError(f"compaction for an unknown scenario {self.scenario!r}")
        si = self.scen_names.index(self.scenario)
        ri = self._rheology_index(col)
        with torch.no_grad():
            drv = heads.mean(dim=0).to(dtype=torch.float32)
            p = column_parts(col, drv)
        tot = p["total"].cpu().numpy()
        self.max_total_diff = max(self.max_total_diff,
                                  float(np.max(np.abs(tot.astype("float64")
                                                      - np.asarray(out, dtype="float64")))))
        el = p["elastic"].cpu().numpy().astype("float64")
        inel = p["inelastic"].cpu().numpy().astype("float64")
        aq = p["aquitard"].cpu().numpy().astype("float64")
        low = p["newlow"].cpu().numpy()[:, self.origin + 1:]
        if el.shape[1] != len(self.dates):
            raise RuntimeError(f"compaction over {el.shape[1]} months, the run has "
                               f"{len(self.dates)}")
        acc = self._acc.get((si, ri))
        if acc is None:
            acc = {"el": 0.0, "inel": 0.0, "aq": 0.0, "n": 0, "low": 0, "run_el": [],
                   "run_inel": [], "run_year": []}
            self._acc[(si, ri)] = acc
        ye = self.ye_idx
        acc["el"] = acc["el"] + el[:, ye]
        acc["inel"] = acc["inel"] + inel[:, ye] + aq[:, ye]
        acc["aq"] = acc["aq"] + aq[:, ye]
        acc["n"] += 1
        by_year = [low[:, self.proj_year_of == y] for y in self.proj_years]
        acc["low"] = acc["low"] + np.stack([b.sum(axis=1) for b in by_year],
                                           axis=1).astype("int64")
        acc["run_year"].append(np.stack([b.any(axis=1) for b in by_year],
                                        axis=1).astype("uint8"))
        _, masks = self.groups(ri)
        w = masks / masks.sum(axis=1, keepdims=True).clip(min=1)
        acc["run_el"].append(w @ el[:, ye])
        acc["run_inel"].append(w @ (inel[:, ye] + aq[:, ye]))

    # -- output ------------------------------------------------------------------------
    def arrays(self) -> dict:
        S, R = len(self.scen_names), len(self.rheology)
        if not self._acc or set(self._acc) != {(s, r) for s in range(S) for r in range(R)}:
            raise RuntimeError("the capture is missing scenario x rheology cells")
        ns = {self._acc[(s, r)]["n"] for s in range(S) for r in range(R)}
        if len(ns) != 1:
            raise RuntimeError(f"unequal member-run counts across cells: {sorted(ns)}")

        def stack(key, mean):
            return np.stack([np.stack([
                (self._acc[(s, r)][key] / self._acc[(s, r)]["n"]) if mean
                else np.asarray(self._acc[(s, r)][key]) for r in range(R)])
                for s in range(S)])

        gnames = [self.groups(r)[0] for r in range(R)]
        if any(g != gnames[0] for g in gnames):
            raise RuntimeError("the rheologies' zone groups differ")
        return {
            "dates": np.array([d.strftime("%Y-%m") for d in self.dates]),
            "origin": np.int64(self.origin), "years": np.asarray(self.years, dtype="int64"),
            "proj_years": np.asarray(self.proj_years, dtype="int64"),
            "scenario_names": np.array(self.scen_names),
            "rheology_labels": np.array(self.rheology),
            "zone_ids": np.stack(self.zone_ids),
            "changhua": (self.changhua.astype("int8") if self.changhua is not None
                         else np.full(self.n_cells, -1, dtype="int8")),
            "group_names": np.array(gnames[0]),
            "n_runs": np.int64(next(iter(ns))),
            "elastic_ye_mean": stack("el", True),
            "inelastic_ye_mean": stack("inel", True),
            "aquitard_ye_mean": stack("aq", True),
            "newlow_runs": stack("low", False),
            "newlow_run_year": stack("run_year", False),
            "run_group_elastic": stack("run_el", False),
            "run_group_inelastic": stack("run_inel", False),
            "total_check_max_abs_m": np.float64(self.max_total_diff),
        }


# ---------------------------------------------------------------------------------------
# installation into the forward's namespace
# ---------------------------------------------------------------------------------------
_PATCHED = ("future_forcing", "climatology", "parse_scenario", "compaction", "load_columns",
            "run")
# the forward's own functions as imported (the scenario grammar extends its parser)
_FWD = {k: getattr(fwd, k) for k in _PATCHED}


def _mark(f):
    f._fw_replay = True
    return f


@contextlib.contextmanager
def installed(replay: ReplayForcing | None = None, capture: Capture | None = None):
    """Swap the forward's projection forcing (``replay``) and record the column's parts
    (``capture``) for the duration of the block. ``replay=None`` keeps the climatology:
    every forcing call goes to the function the forward holds, unchanged. Every name is
    restored on exit; a nested installation is refused."""
    cur = {k: getattr(fwd, k) for k in _PATCHED}
    if any(getattr(v, "_fw_replay", False) for v in cur.values()):
        raise RuntimeError("hydrophysics.twin.forward is already patched (nested replay?)")

    @_mark
    def future_forcing(inp, scenario, horizon, rain_scale=1.0, zone_of_cell=None,
                       eta_classes=None, climatology_end=None):
        if capture is not None:
            capture.scenario = scenario.name
        if replay is None:
            return cur["future_forcing"](inp, scenario, horizon, rain_scale=rain_scale,
                                         zone_of_cell=zone_of_cell, eta_classes=eta_classes,
                                         climatology_end=climatology_end)
        fut_by_class, fut_dates = {}, None
        for k, E in inp.E_by_class.items():
            fut_by_class[k], fut_dates = replay.energy(k, E, inp.dates, horizon,
                                                       climatology_end)
        if eta_classes:
            E_fut = np.stack([scenario.apply({c: fut_by_class[c]}, fut_dates,
                                             zone_of_cell=zone_of_cell) for c in eta_classes])
        else:
            E_fut = scenario.apply(fut_by_class, fut_dates, zone_of_cell=zone_of_cell)
        r_fut, _ = replay.weather_field(inp.recharge_field.numpy(), inp.dates, horizon,
                                        climatology_end)
        return (torch.tensor(E_fut, dtype=torch.float64),
                torch.tensor(r_fut * rain_scale, dtype=torch.float64), fut_dates)

    @_mark
    def climatology(arr, dates, horizon, end=None):
        # reached by future_sw, sw_substitution_energy and the window checks: the canal
        # deliveries follow the weather fields
        if replay is None:
            return cur["climatology"](arr, dates, horizon, end=end)
        return replay.weather_field(arr, dates, horizon, end)

    @_mark
    def compaction(col, heads):
        out = cur["compaction"](col, heads)
        if capture is not None:
            capture.add(col, heads, out)
        return out

    @_mark
    def load_columns(*a, **k):
        cols = cur["load_columns"](*a, **k)
        if capture is not None:
            for label, col, _ in cols:
                capture.label(col, label)
        return cols

    @_mark
    def run(inp, members, scenarios, horizon, *a, **k):
        if replay is not None:
            replay.bind(inp)
        if capture is not None:
            capture.bind(inp, scenarios, horizon)
        return cur["run"](inp, members, scenarios, horizon, *a, **k)

    new = {"future_forcing": future_forcing, "climatology": climatology,
           "parse_scenario": _mark(_parse_wrapper()), "compaction": compaction,
           "load_columns": load_columns, "run": run}
    try:
        for k, v in new.items():
            setattr(fwd, k, v)
        yield
    finally:
        for k, v in cur.items():
            setattr(fwd, k, v)


def _parse_wrapper():
    def parse(text):
        return parse_scenario(text)
    return parse


# ---------------------------------------------------------------------------------------
# engineering gate: a 2012..2022 replay from 2012-01 is the record
# ---------------------------------------------------------------------------------------
def forcing_identity_gate(inp, years=None, trend_years: tuple[int, int] | None = None
                          ) -> dict:
    """Replay the record's own years from its first month and compare with the record.

    Two layers, booleans and shapes only (no forcing value):

    - ``fields``: :func:`replay_extend` on every forcing array (energy by class, recharge,
      canal deliveries if present) equals the array exactly; ``detrended_times_factor_equal``:
      the detrended energy times its factors equals the record (relative 1e-12);
      ``trend_factors_positive``: every factor >= ``G_MIN`` (a refusal is caught and reported
      False with its message, so the gate can run as a design-time dry run).
    - ``pipeline``: the forward's own projection functions as the run calls them, patched by
      :func:`installed` with a replay whose first projected month is the record's first
      (``ReplayForcing.start``): ``forward.future_forcing`` under the baseline scenario, per
      class stacked (the ``eta_classes`` path) and summed, equals the record's energy; its
      recharge equals the record's recharge (``rain_scale`` 1, torch conversion included);
      ``forward.future_sw`` (if the inputs carry canal water) equals the record's canal
      deliveries; the detrended stacked energy times its factors equals the record.

    ``trend_years`` defaults to the replayed years' span."""
    yrs = tuple(years) if years is not None else tuple(sorted({int(y) for y in inp.dates.year}))
    T = len(inp.dates)
    if len(yrs) * 12 != T:
        raise ValueError(f"the gate replays whole years: {len(yrs)} years for {T} months")
    d0 = inp.dates[0]
    ty = trend_years or (min(yrs), max(yrs))
    res: dict = {"years": format_years(yrs), "months": T, "fields": {}}

    def same(name, arr):
        a = np.asarray(arr)
        ext, fut = replay_extend(a, inp.dates, T, yrs, start=d0)
        res["fields"][name] = {"shape": list(a.shape), "dates_equal": bool(fut.equals(inp.dates)),
                               "array_equal": bool(np.array_equal(ext, a))}

    for c, E in inp.E_by_class.items():
        same(f"energy:{c}", E)
    same("recharge", inp.recharge_field.numpy())
    if getattr(inp, "sw_field", None) is not None:
        same("canal", np.asarray(inp.sw_field))
    det: dict[str, bool] = {}
    rf = ReplayForcing(yrs, pumping="detrended", trend_years=ty)
    try:
        rf.bind(inp)
        res["trend_factors_positive"] = True
        res["trend_refusal"] = None
    except ValueError as err:
        rf = None
        res["trend_factors_positive"] = False
        res["trend_refusal"] = str(err)
    if rf is not None:
        for c, E in inp.E_by_class.items():
            e = np.asarray(E, dtype="float64")
            ext, _ = replay_extend(e, inp.dates, T, yrs, start=d0, divisor=rf.factors[c])
            g = np.array([rf.factors[c][int(y)] for y in inp.dates.year], dtype="float64")
            det[c] = bool(np.allclose(ext * g, e, rtol=1e-12, atol=0.0))
    res["detrended_times_factor_equal"] = det
    res["pipeline"] = _pipeline_gate(inp, yrs, ty, d0, rf is not None)
    res["passed"] = bool(all(v["array_equal"] and v["dates_equal"]
                             for v in res["fields"].values())
                         and res["trend_factors_positive"] and all(det.values())
                         and res["pipeline"]["passed"])
    return res


def _pipeline_gate(inp, yrs, ty, d0, detrended: bool) -> dict:
    """The ``pipeline`` layer of :func:`forcing_identity_gate`."""
    T = len(inp.dates)
    classes = list(inp.E_by_class)
    want_stack = np.stack([np.asarray(inp.E_by_class[c], dtype="float64") for c in classes])
    want_sum = np.asarray(fwd.BASELINE.apply(inp.E_by_class, inp.dates), dtype="float64")
    want_r = inp.recharge_field.numpy()
    out: dict = {}
    raw = ReplayForcing(yrs, pumping="raw", fields="energy+weather", trend_years=ty, start=d0)
    raw.bind(inp)
    with installed(raw):
        Es, rs, ds = fwd.future_forcing(inp, fwd.BASELINE, T, eta_classes=classes)
        E1, r1, d1 = fwd.future_forcing(inp, fwd.BASELINE, T)
        sw = fwd.future_sw(inp, fwd.BASELINE, T)
    out["dates_equal"] = bool(ds.equals(inp.dates) and d1.equals(inp.dates))
    out["energy_by_class_equal"] = bool(np.array_equal(Es.numpy(), want_stack))
    out["energy_summed_equal"] = bool(np.array_equal(E1.numpy(), want_sum))
    out["recharge_equal"] = bool(np.array_equal(rs.numpy(), want_r)
                                 and np.array_equal(r1.numpy(), want_r))
    if getattr(inp, "sw_field", None) is not None:
        out["canal_equal"] = bool(sw is not None
                                  and np.array_equal(sw.numpy(), np.asarray(inp.sw_field)))
    else:
        out["canal_equal"] = sw is None
    if detrended:
        dt_ = ReplayForcing(yrs, pumping="detrended", fields="energy+weather", trend_years=ty,
                            start=d0)
        dt_.bind(inp)
        with installed(dt_):
            Ed, _, _ = fwd.future_forcing(inp, fwd.BASELINE, T, eta_classes=classes)
        ok = True
        for i, c in enumerate(classes):
            g = np.array([dt_.factors[c][int(y)] for y in inp.dates.year], dtype="float64")
            ok &= bool(np.allclose(Ed.numpy()[i] * g, want_stack[i], rtol=1e-12, atol=0.0))
        out["detrended_by_class_times_factor_equal"] = bool(ok)
    else:
        out["detrended_by_class_times_factor_equal"] = False
    out["passed"] = bool(all(out.values()))
    return out


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------
def code_shas() -> dict:
    """sha256 of the engine's code files as imported."""
    import hydrophysics.twin.compaction as cmp
    import hydrophysics.twin.scenario as scn

    out = {}
    for name, mod in (("replay.py", sys.modules[__name__]), ("forward.py", fwd),
                      ("scenario.py", scn), ("compaction.py", cmp)):
        with open(mod.__file__, "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def parse_projection_forcing(text: str) -> tuple[int, ...] | None:
    """``climatology`` -> ``None``; ``replay:<years>`` -> the source-year sequence."""
    if text == "climatology":
        return None
    if not text.startswith("replay:"):
        raise ValueError(f"--projection-forcing {text!r}: expected 'climatology' or "
                         "'replay:<years>'")
    return parse_years(text.split(":", 1)[1])


def _out_stem(rest: list[str]) -> str | None:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--out", default=None)
    return p.parse_known_args(rest)[0].out


def gate_main(argv) -> int:
    """``replay gate``: load the twin inputs as the forward does and run the identity gate;
    prints booleans only and writes the JSON record."""
    ap = argparse.ArgumentParser(prog="replay gate", allow_abbrev=False)
    ap.add_argument("--theta", required=True, help="the in-sample theta (input options)")
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--wells-from", default=None)
    ap.add_argument("--out", required=True, help="gate record (JSON, written once)")
    for k in ("polygon", "wells_dir", "stations", "pump_census", "pump_kwh",
              "rf_timeseries", "rf_stations", "gw_stations", "et_npz"):
        ap.add_argument(f"--{k.replace('_', '-')}", default=None)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"REFUSED: {a.out} exists", file=sys.stderr)
        return 2
    meta = fwd.load_members([a.theta])[0].meta
    paths = {k: getattr(a, k) for k in ("polygon", "wells_dir", "stations", "pump_census",
                                        "pump_kwh", "rf_timeseries", "rf_stations",
                                        "gw_stations", "et_npz")}
    inp = fwd.load_twin_inputs(paths, dx=a.dx, wells_from=a.wells_from,
                               meter_filter=meta.get("meter_filter", "none"),
                               cap_duty=float(meta.get("cap_duty", 1.0)),
                               **fwd.input_options(meta))
    fwd.attach_sw_recharge(inp, meta, log=lambda *_: None)
    res = forcing_identity_gate(inp)
    res["code_sha256"] = code_shas()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    for name, r in res["fields"].items():
        print(f"  {name}: replay equals the record {r['array_equal'] and r['dates_equal']}")
    for c, ok in res["detrended_times_factor_equal"].items():
        print(f"  energy:{c} detrended x factor equals the record {ok}")
    print(f"  trend factors >= {G_MIN}: {res['trend_factors_positive']}")
    for k, v in res["pipeline"].items():
        if k != "passed":
            print(f"  pipeline {k}: {v}")
    print(f"forcing identity gate: {'PASS' if res['passed'] else 'FAIL'} ({a.out})")
    return 0 if res["passed"] else 3


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["gate"]:
        return gate_main(argv[1:])
    ap = argparse.ArgumentParser(
        description="forward twin with a replayed-weather projection (opt-in); every "
                    "argument after '--' goes to hydrophysics.twin.forward unchanged",
        allow_abbrev=False)
    ap.add_argument("--projection-forcing", default="climatology",
                    help="'climatology' (the forward's own, bit-identical) or "
                         "'replay:<years>', e.g. replay:2015-2022,2012-2014")
    ap.add_argument("--replay-pumping", choices=PUMPING_MODES, default="detrended")
    ap.add_argument("--replay-fields", choices=FIELD_MODES, default="energy+weather")
    ap.add_argument("--trend-years", default=f"{TREND_YEARS[0]}-{TREND_YEARS[1]}")
    ap.add_argument("--capture", default=None,
                    help="capture file (default <out>.fw.npz); 'none' records nothing")
    if "--" not in argv:
        ap.error("separate the forward's arguments with '--'")
    cut = argv.index("--")
    a = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    years = parse_projection_forcing(a.projection_forcing)
    ty = parse_years(a.trend_years)
    if len(ty) < 2 or ty != tuple(range(ty[0], ty[-1] + 1)):
        ap.error("--trend-years must be a range YYYY-YYYY")
    replay = (None if years is None else
              ReplayForcing(years, pumping=a.replay_pumping, fields=a.replay_fields,
                            trend_years=(ty[0], ty[-1])))
    cap_path = a.capture
    if cap_path is None:
        stem = _out_stem(rest)
        if stem is None:
            ap.error("the forward needs --out")
        cap_path = stem + ".fw.npz"
    capture = None if cap_path == "none" else Capture()
    if capture is not None and os.path.exists(cap_path):
        ap.error(f"{cap_path} exists")
    with installed(replay, capture):
        fwd.main(rest)
    if capture is None:
        return 0
    rec = {"projection_forcing": a.projection_forcing,
           "replay": None if replay is None else replay.record(),
           "code_sha256": code_shas(), "forward_argv": rest, "county": capture.county}
    os.makedirs(os.path.dirname(cap_path) or ".", exist_ok=True)
    with open(cap_path, "xb") as fh:
        np.savez_compressed(fh, record=json.dumps(rec, sort_keys=True), **capture.arrays())
    print(f"wrote {cap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
