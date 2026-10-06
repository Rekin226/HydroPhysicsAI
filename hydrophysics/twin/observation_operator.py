"""Explicit depth and datum operators for compaction observations.

Positive values mean shortening/subsidence. A total-column archive alone cannot
predict shortening between two monitoring rings: its vertical distribution is unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthProfile:
    """Compaction (metres, time x layer) between absolute depths below land surface.

    Each layer value is its full-layer shortening relative to ``reference_month``.
    Only ring boundaries coincident with model layer boundaries are supported; a
    fractional-layer allocation would assume an unverified uniform strain profile.
    """

    edges_m_bgs: tuple[float, ...]
    compaction_m: np.ndarray
    reference_month: str

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges_m_bgs, dtype=float)
        values = np.asarray(self.compaction_m, dtype=float)
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
            raise ValueError("Depth edges must increase below land surface in metres")
        if (values.ndim != 2 or values.shape[1] != len(edges) - 1
                or not np.isfinite(values).all()):
            raise ValueError("Compaction must be finite time x depth-layer metres")
        if not self.reference_month:
            raise ValueError("A displacement reference month is required")


def ring_interval(profile: DepthProfile, shallow_m_bgs: float,
                  deep_m_bgs: float, *, reference_month: str) -> np.ndarray:
    """Predicted shortening within exactly resolved ring boundaries, in metres.

    Ring positions and layer edges must share the same land-surface depth datum.
    Depths outside or inside a layer are refused, rather than silently extrapolated.
    """
    if reference_month != profile.reference_month:
        raise ValueError("Ring and profile displacement reference months differ")
    edges = np.asarray(profile.edges_m_bgs)
    hits = []
    for depth in (shallow_m_bgs, deep_m_bgs):
        hit = np.flatnonzero(np.isclose(edges, depth, rtol=0, atol=1e-6))
        if len(hit) != 1:
            raise ValueError("Ring depth lacks an exactly resolved model boundary")
        hits.append(int(hit[0]))
    if hits[0] >= hits[1]:
        raise ValueError("Deep ring must lie below shallow ring")
    return np.asarray(profile.compaction_m)[:, hits[0]:hits[1]].sum(axis=1)


def total_column(profile: DepthProfile) -> np.ndarray:
    """Shortening across the complete *modeled* depth, in metres."""
    return np.asarray(profile.compaction_m).sum(axis=1)


def surface_displacement(profile: DepthProfile, *, residual_m: np.ndarray | None,
                         reference_month: str) -> np.ndarray:
    """Surface subsidence including motion outside the modeled column.

    ``residual_m`` is explicit motion from deeper/shallower material, tectonics,
    benchmarks and other unresolved processes; zero must be an explicit assumption.
    """
    if reference_month != profile.reference_month:
        raise ValueError("Surface and profile displacement reference months differ")
    if residual_m is None:
        raise ValueError("Surface displacement requires an explicit residual term")
    residual = np.asarray(residual_m, dtype=float)
    column = total_column(profile)
    if residual.shape != column.shape or not np.isfinite(residual).all():
        raise ValueError("Residual must be finite and match the time axis")
    if profile.edges_m_bgs[0] != 0:
        raise ValueError("Modeled column must start at land surface")
    return column + residual
