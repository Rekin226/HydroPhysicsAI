"""Forward pumping scenarios: a policy, not a multiplier.

``explorer3d`` currently dials "1.5x drawdown", which is not a decision anyone can take.
The TPC census carries ``PURPOSE`` and ``PUMP_HP`` per pump, so scenarios can instead be
expressed the way abstraction is actually managed -- cut irrigation by 30%, retire
aquaculture, cap the proximal fan -- and converted into the per-cell energy field the flow
model already consumes.

Scale, for calibration of intuition: of 457,750 installed HP inside the fan mask,
**irrigation is 86%** (dry crop 140k, first rice 116k, second rice 72k, other 63k,
greenhouse 4k). Aquaculture is 35k, livestock 15k, domestic 6k, industry under 1k. A
scenario that does not touch irrigation is not moving the fan.

Running forward
---------------
Beyond the observed record there is no forcing, so one has to be assumed. ``climatology``
repeats the month-of-year mean of the historical record, which is the honest default: it
carries the seasonal cycle that drives the system without pretending to forecast weather.
Anything better is a climate scenario and should be passed in explicitly.

**Caveat that governs every number this module produces:** the pumping -> head map is the
four-layer flow model, whose Stage-3 gate has not returned a verdict. Until it does, a
projection here is mechanically correct and physically uncalibrated. The plumbing is
right; the numbers are not yet evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Coarse policy classes, keyed off the substrings that actually appear in PURPOSE.
# Order matters: 灌溉 (irrigation) and 養殖 (aquaculture) and 畜牧 (livestock) are all
# nested inside 農業用水 (agricultural water), so the specific tests must run first.
_CLASS_RULES = (
    ("irrigation", ("灌溉",)),
    ("aquaculture", ("養殖",)),
    ("livestock", ("畜牧",)),
    ("domestic", ("家庭用水", "家用及公共給水", "公共用水")),
    ("industry", ("工業用水",)),
)
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")


def purpose_class(purpose: object) -> str:
    """Map a raw ``PURPOSE`` string to one of ``CLASSES``.

    Bare ``農業用水`` with no qualifier falls through to ``other`` rather than being
    guessed into irrigation -- 228 pumps, and inventing a class for them would quietly
    move HP into the lever the scenarios act on.
    """
    s = str(purpose)
    for name, needles in _CLASS_RULES:
        if any(n in s for n in needles):
            return name
    return "other"


def energy_by_class(pumps: pd.DataFrame, kwh: pd.DataFrame, grid,
                    t0: str, t1: str) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
    """Monthly kWh per active cell, split by policy class -> ``({class: (A, T)}, dates)``.

    Splitting once here is what lets a scenario act on one class without re-aggregating
    116k pumps per scenario.
    """
    from .pumping import aggregate_pumps

    out: dict[str, np.ndarray] = {}
    dates = None
    cls = pumps["PURPOSE"].map(purpose_class)
    for name in CLASSES:
        sub = pumps[cls == name]
        if sub.empty:
            continue
        E, dates = aggregate_pumps(sub, kwh, grid, t0, t1)
        out[name] = E
    if dates is None:
        raise ValueError("no pumps matched any policy class")
    return out, dates


@dataclass
class PumpingScenario:
    """A pumping policy, applied multiplicatively to abstraction energy.

    ``factors`` maps a policy class to its multiplier (1.0 = unchanged, 0.7 = a 30% cut,
    0.0 = retired). Classes not named are left alone.

    ``zones`` optionally restricts the policy to named fan zones (``zones.ZONE_NAMES``);
    ``start`` optionally delays it, so "cut irrigation 30% from 2026" is expressible
    rather than being applied retroactively over the whole record.
    """

    name: str
    factors: dict[str, float] = field(default_factory=dict)
    zones: tuple[str, ...] | None = None
    start: str | None = None

    def __post_init__(self) -> None:
        bad = set(self.factors) - set(CLASSES)
        if bad:
            raise ValueError(f"unknown policy class(es) {sorted(bad)}; expected {CLASSES}")
        neg = {k: v for k, v in self.factors.items() if v < 0}
        if neg:
            raise ValueError(f"negative pumping factor(s) {neg}")

    def apply(self, e_by_class: dict[str, np.ndarray], dates: pd.DatetimeIndex,
              zone_of_cell: np.ndarray | None = None) -> np.ndarray:
        """Combine the per-class energy fields under this policy -> ``(A, T)``."""
        from .zones import ZONE_NAMES

        first = next(iter(e_by_class.values()))
        total = np.zeros_like(first)

        active = np.ones(len(dates), dtype="float64")
        if self.start is not None:
            active = (dates >= pd.Timestamp(self.start)).astype("float64")

        cell_sel = np.ones(first.shape[0], dtype="float64")
        if self.zones is not None:
            if zone_of_cell is None:
                raise ValueError("zones= requires zone_of_cell")
            idx = [ZONE_NAMES.index(z) for z in self.zones]
            cell_sel = np.isin(zone_of_cell, idx).astype("float64")

        for name, E in e_by_class.items():
            f = float(self.factors.get(name, 1.0))
            if f == 1.0:
                total = total + E
                continue
            # Blend toward the factor only where and when the policy is in force; a cell
            # or month outside its scope keeps the unmodified field.
            scale = 1.0 + (f - 1.0) * cell_sel[:, None] * active[None, :]
            total = total + E * scale
        return total

    def describe(self) -> str:
        if not self.factors:
            return f"{self.name}: baseline (no change)"
        parts = [f"{k} x{v:g}" for k, v in sorted(self.factors.items())]
        s = f"{self.name}: " + ", ".join(parts)
        if self.zones:
            s += f" in {'/'.join(self.zones)}"
        if self.start:
            s += f" from {self.start}"
        return s


BASELINE = PumpingScenario("baseline")


def climatology(arr: np.ndarray, dates: pd.DatetimeIndex,
                horizon: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Extend ``(..., T)`` forward by ``horizon`` months using its month-of-year mean.

    Returns the extension only (not concatenated), plus its dates, so callers stay explicit
    about what is observation and what is assumption.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    months = dates.month.to_numpy()
    clim = np.stack([np.nanmean(arr[..., months == m], axis=-1) for m in range(1, 13)],
                    axis=-1)                                   # (..., 12)
    future = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return clim[..., future.month.to_numpy() - 1], future


def project(twin, h0, recharge: np.ndarray, pumping: np.ndarray, device=None):
    """Run a ``CoupledTwin`` over a forcing sequence -> ``(heads, subsidence)`` as numpy.

    ``recharge``/``pumping`` are ``(n_layers, A, T)``; the run is ``T`` steps and no
    gradient is kept, since projection is a forward question.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    twin = twin.to(dev)
    steps = pumping.shape[-1]
    t_h0 = torch.as_tensor(h0, dtype=torch.float64, device=dev)
    t_r = torch.as_tensor(recharge, dtype=torch.float64, device=dev)
    t_p = torch.as_tensor(pumping, dtype=torch.float64, device=dev)
    with torch.no_grad():
        heads, subs = twin(t_h0, t_r, t_p, steps)
    return heads.cpu().numpy(), subs.cpu().numpy()
