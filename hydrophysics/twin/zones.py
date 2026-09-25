"""Fan coordinates -> proximal/mid/distal zone id.

The Stage-3 homogeneous fit failed because one transmissivity cannot describe coarse
proximal gravel and fine distal silt at once: three of four ``log_T`` layers sat pinned
at the lower clamp, below the 58 m2/day floor Liu et al. (2002) measured at Choushui.
This module supplies the geometry for the structural fix.

Boundaries, and how much each is worth trusting:

- **proximal/mid at x = 205 km is well-constrained.** The published criterion is that the
  proximal fan is where the confining mud layers are absent. In this project's own data,
  wells screened in layers 3-4 stop at x = 207.9 km while layers 1-2 continue to
  214.8 km, so the aquitards pinch out at roughly 203-208 km.
- **mid/distal at x = 182 km is NOT constrained.** The transition is a gradual grain-size
  gradient with no structure to locate; 182 km is the equal-width third, a default rather
  than a finding. Spec §4.2 requires re-running the gate at 178 and 186 km and reporting
  whether the verdict moves.

Intervals are half-open and inclusive on the high (eastern) side.
"""

from __future__ import annotations

import numpy as np

PROXIMAL = 0
MID = 1
DISTAL = 2
N_ZONES = 3
ZONE_NAMES = ("proximal", "mid", "distal")
# Opt-in proximal split (``--zone-boundaries P,D,S`` with S > P, round 3): the proximal
# cells west of S become a fourth zone. The id is APPENDED, so ids 0-2 keep their meaning
# for every consumer that only knows three zones. ``collapse_zones`` maps it back.
PROXIMAL_W = 3
ZONE_NAMES_SPLIT = ZONE_NAMES + ("proximal_w",)


def zone_names(n_zones: int = N_ZONES) -> tuple[str, ...]:
    """The zone names of an ``n_zones`` zonation (3, or 4 with the proximal split)."""
    if n_zones == N_ZONES:
        return ZONE_NAMES
    if n_zones == len(ZONE_NAMES_SPLIT):
        return ZONE_NAMES_SPLIT
    raise ValueError(f"only 3 or 4 fan zones are supported, got {n_zones}")


def collapse_zones(zone: np.ndarray) -> np.ndarray:
    """A zone-id array with the proximal split -> the three-zone ids (``proximal_w`` goes
    back into ``proximal``). The compaction column, scenario zones and river conductance
    split stay three-zone, and this is how they get their ids."""
    z = np.asarray(zone)
    return np.where(z == PROXIMAL_W, PROXIMAL, z).astype(z.dtype, copy=False)


def fan_zones(xy: np.ndarray,
              proximal_km: float = 205.0,
              distal_km: float = 182.0,
              split_km: float | None = None) -> np.ndarray:
    """TWD97/EPSG:3826 easting -> zone id. 0 = proximal (E), 1 = mid, 2 = distal (W).

    ``xy`` is ``(n, 2)`` in **metres**; the boundaries are in **kilometres**. Only the
    easting is read -- the zonation is a west-east banding, so northing is ignored (see
    the northern-lobe open question in spec §10).

    ``split_km`` (opt-in, must lie east of ``proximal_km``) splits the proximal zone: cells
    with ``proximal_km <= x < split_km`` get id ``PROXIMAL_W`` (3), and ``PROXIMAL`` (0)
    keeps the part east of ``split_km``. ``None`` = the three-zone map exactly.
    """
    arr = np.asarray(xy, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"xy must be (n, 2) coordinates in metres, got shape {arr.shape}")
    if not distal_km < proximal_km:
        raise ValueError(
            f"distal_km ({distal_km}) must sit west of proximal_km ({proximal_km}); "
            "otherwise the mid zone is empty and every downstream count is wrong"
        )
    x_km = arr[:, 0] / 1000.0
    zone = np.where(x_km >= proximal_km, PROXIMAL,
                    np.where(x_km >= distal_km, MID, DISTAL)).astype("int64")
    if split_km is not None:
        if not float(split_km) > proximal_km:
            raise ValueError(f"split_km ({split_km}) must sit east of proximal_km "
                             f"({proximal_km}): it splits the proximal zone")
        zone[(zone == PROXIMAL) & (x_km < float(split_km))] = PROXIMAL_W
    return zone


def zone_blend_weights(xy: np.ndarray,
                       proximal_km: float = 205.0,
                       distal_km: float = 182.0,
                       blend_km: float = 0.0,
                       split_km: float | None = None) -> np.ndarray:
    """Per-cell zone weights ``(N_ZONES, n)``, each column summing to 1.

    ``blend_km <= 0`` is the one-hot of :func:`fan_zones`, which reproduces the sharp
    zonation exactly. ``blend_km > 0`` (``--zone-blend-km``, 2026-09-23) softens
    the mid/distal line only. The distal weight is ``sigmoid((distal_km - x) / blend_km)``
    and the mid weight is its complement. The proximal/mid line stays sharp: the aquitard
    pinch-out sets it physically (module docstring), while 182 km is an equal-width
    default. The app review found a subsidence step on that line of about 2.5 cm/yr in the
    hindcast, and the leveling rates show no such step (A3 diagnosis). Parameters are
    mixed with these weights (``cols @ W``). For the log-parameters (T, S, L, Ske, Skv,
    tau) that is a geometric mean, and for ``h_pc0`` it is linear.

    ``split_km`` (the opt-in proximal split, see :func:`fan_zones`) returns ``(4, n)``
    weights; both proximal parts stay one-hot (the split line is sharp).
    """
    arr = np.asarray(xy, dtype="float64")
    zone = fan_zones(arr, proximal_km=proximal_km, distal_km=distal_km, split_km=split_km)
    n = zone.shape[0]
    w = np.zeros((N_ZONES if split_km is None else len(ZONE_NAMES_SPLIT), n),
                 dtype="float64")
    if blend_km is None or blend_km <= 0.0:
        w[zone, np.arange(n)] = 1.0
        return w
    x_km = arr[:, 0] / 1000.0
    prox = (zone == PROXIMAL) | (zone == PROXIMAL_W)
    w_d = 0.5 * (1.0 + np.tanh(0.5 * (distal_km - x_km) / float(blend_km)))  # sigmoid
    w[PROXIMAL] = zone == PROXIMAL
    if split_km is not None:
        w[PROXIMAL_W] = zone == PROXIMAL_W
    w[DISTAL] = np.where(prox, 0.0, w_d)
    w[MID] = np.where(prox, 0.0, 1.0 - w_d)
    return w
