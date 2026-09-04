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


def fan_zones(xy: np.ndarray,
              proximal_km: float = 205.0,
              distal_km: float = 182.0) -> np.ndarray:
    """TWD97/EPSG:3826 easting -> zone id. 0 = proximal (E), 1 = mid, 2 = distal (W).

    ``xy`` is ``(n, 2)`` in **metres**; the boundaries are in **kilometres**. Only the
    easting is read -- the zonation is a west-east banding, so northing is ignored (see
    the northern-lobe open question in spec §10).
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
    return np.where(x_km >= proximal_km, PROXIMAL,
                    np.where(x_km >= distal_km, MID, DISTAL)).astype("int64")
