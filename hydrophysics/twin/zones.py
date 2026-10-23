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
# Opt-in county split (``--zone-county-split``, rung R1c, 2026-10-15): every zone gets a
# Changhua copy. With ``n_base`` easting zones (3, or 4 with the proximal split) a Changhua
# cell of zone ``z`` gets id ``z + n_base``, so the rest-of-fan ids keep their meaning, and
# the names are the base names followed by the same names with this suffix.
COUNTY_SUFFIX = "_changhua"


def county_zone_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """``names`` followed by their Changhua copies (``proximal`` -> ``proximal_changhua``)."""
    return tuple(names) + tuple(f"{z}{COUNTY_SUFFIX}" for z in names)


ALL_ZONE_NAMES = county_zone_names(ZONE_NAMES_SPLIT)
# the zones with the proximal (merged-aquifer) form and the raised log_T floor
PROXIMAL_ZONE_NAMES = ("proximal", "proximal_w", f"proximal{COUNTY_SUFFIX}",
                       f"proximal_w{COUNTY_SUFFIX}")


def zone_labels(split: bool = False, county: bool = False) -> tuple[str, ...]:
    """The ordered zone names of a zonation: the three easting zones, ``proximal_w`` with
    the proximal split, then the Changhua copy of each with the county split. Zone id
    ``i`` is ``zone_labels(...)[i]``."""
    base = ZONE_NAMES_SPLIT if split else ZONE_NAMES
    return county_zone_names(base) if county else base


def zone_names(n_zones: int = N_ZONES) -> tuple[str, ...]:
    """The zone names of an ``n_zones`` zonation: 3, 4 with the proximal split, 6 or 8
    with the county split on top."""
    for split in (False, True):
        for county in (False, True):
            names = zone_labels(split, county)
            if n_zones == len(names):
                return names
    raise ValueError(f"only 3, 4, 6 or 8 fan zones are supported, got {n_zones}")


def collapse_zones(zone: np.ndarray) -> np.ndarray:
    """A zone-id array with the proximal split -> the three-zone ids (``proximal_w`` goes
    back into ``proximal``). The compaction column, scenario zones and river conductance
    split stay three-zone, and this is how they get their ids. A county-split map is
    collapsed by :func:`collapse_county_zones` instead (its id 3 is not ``proximal_w``)."""
    z = np.asarray(zone)
    return np.where(z == PROXIMAL_W, PROXIMAL, z).astype(z.dtype, copy=False)


def county_split(zone: np.ndarray, changhua: np.ndarray, n_base: int) -> np.ndarray:
    """``--zone-county-split``: easting zone ids ``0..n_base-1`` -> the Changhua cells'
    ids moved to ``id + n_base``. ``changhua`` is the per-cell county flag
    (``zone_county``)."""
    z = np.asarray(zone, dtype="int64")
    ch = np.asarray(changhua).astype(bool)
    if z.shape != ch.shape:
        raise ValueError(f"zone {z.shape} and county flag {ch.shape} differ in shape")
    if z.size and (z.min() < 0 or z.max() >= int(n_base)):
        raise ValueError(f"zone ids must lie in 0..{int(n_base) - 1} before the county split")
    return np.where(ch, z + int(n_base), z).astype("int64")


def split_county(zone: np.ndarray, n_base: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`county_split` -> ``(easting zone ids, Changhua flag)``."""
    z = np.asarray(zone, dtype="int64")
    if z.size and (z.min() < 0 or z.max() >= 2 * int(n_base)):
        raise ValueError(f"county-split zone ids must lie in 0..{2 * int(n_base) - 1}")
    ch = z >= int(n_base)
    return np.where(ch, z - int(n_base), z), ch


def collapse_county_zones(zone: np.ndarray, n_base: int) -> np.ndarray:
    """A county-split flow zone map -> the column's map: :func:`collapse_zones` on the
    easting ids (``proximal_w`` back into ``proximal``) with the county kept, i.e. ids
    0-2 for the rest of the fan and 3-5 for their Changhua copies
    (``zone_labels(county=True)``)."""
    base, ch = split_county(zone, n_base)
    return county_split(collapse_zones(base), ch, N_ZONES)


def county_split_weights(w: np.ndarray, changhua: np.ndarray) -> np.ndarray:
    """``(n, A)`` zone weights (``zone_blend_weights``) -> ``(2n, A)``: a Changhua cell's
    weights move to the Changhua copies of its zones. Columns still sum to 1, and one-hot
    weights stay the one-hot of :func:`county_split`."""
    w = np.asarray(w, dtype="float64")
    ch = np.asarray(changhua).astype(bool).reshape(-1)
    if w.ndim != 2 or w.shape[1] != ch.shape[0]:
        raise ValueError(f"weights {w.shape} do not match {ch.shape[0]} cells")
    c = ch.astype("float64")[None, :]
    return np.concatenate([w * (1.0 - c), w * c], axis=0)


def fan_band_of(zone: np.ndarray) -> np.ndarray:
    """The proximal/mid/distal band (ids 0-2) of a column or scenario zone map: the
    three-zone map unchanged; the four-zone map's ``proximal_w`` (3) and the county map's
    Changhua copies (3-5) back onto their band (``id % 3``). Scenario ``zones=`` select a
    band, which is a place on the fan, not a parameter set."""
    z = np.asarray(zone, dtype="int64")
    if z.size and (z.min() < 0 or z.max() >= 2 * N_ZONES):
        raise ValueError(f"a column/scenario zone map has ids 0..{2 * N_ZONES - 1}, got "
                         f"{int(z.min())}..{int(z.max())}: collapse a county-split flow map "
                         "first (collapse_county_zones)")
    return z % N_ZONES


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


# ---------------------------------------------------------------------------------------
# Banded column (opt-in, 2026-09-29): column parameters on N bands along the down-fan axis
# ---------------------------------------------------------------------------------------
BAND_KINDS = ("hat", "step")


def band_centres_km(x_km: np.ndarray, n_bands: int,
                    lo_km: float | None = None, hi_km: float | None = None) -> np.ndarray:
    """``n_bands`` evenly spaced band centres (km) from ``lo_km`` to ``hi_km``.

    The defaults are the westmost and eastmost cell eastings, so the end bands sit on the
    coast and on the apex."""
    if int(n_bands) < 2:
        raise ValueError(f"a banded column needs at least 2 bands, got {n_bands}")
    x = np.asarray(x_km, dtype="float64")
    lo = float(np.min(x)) if lo_km is None else float(lo_km)
    hi = float(np.max(x)) if hi_km is None else float(hi_km)
    if not hi > lo:
        raise ValueError(f"band range must have hi > lo, got {lo:g}..{hi:g} km")
    return np.linspace(lo, hi, int(n_bands))


def band_weights(xy: np.ndarray, centres_km, kind: str = "hat") -> np.ndarray:
    """Per-cell band weights ``(n_bands, n)``, each column summing to 1.

    Only the easting is read: the bands run along the fan's down-fan (west-east) axis,
    the same axis as :func:`fan_zones`.

    ``kind="hat"`` interpolates linearly between the two nearest band centres (a cell
    beyond the end centres takes the end band). Mixed through ``forward.blended_column``,
    each log-parameter is then a piecewise-geometric and ``h_pc0`` a piecewise-linear
    function of easting, continuous everywhere: no line on the map where the column
    parameters jump. ``kind="step"`` gives each cell the band of its nearest centre
    (one-hot), which does jump, at the midpoints between centres."""
    if kind not in BAND_KINDS:
        raise ValueError(f"band kind must be one of {BAND_KINDS}, got {kind!r}")
    arr = np.asarray(xy, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"xy must be (n, 2) coordinates in metres, got shape {arr.shape}")
    c = np.asarray(centres_km, dtype="float64").reshape(-1)
    if c.size < 2 or not np.all(np.diff(c) > 0):
        raise ValueError("band centres must be >= 2 strictly increasing values (km)")
    x = arr[:, 0] / 1000.0
    n = x.shape[0]
    w = np.zeros((c.size, n), dtype="float64")
    if kind == "step":
        w[np.argmin(np.abs(x[:, None] - c[None, :]), axis=1), np.arange(n)] = 1.0
        return w
    xc = np.clip(x, c[0], c[-1])
    i = np.clip(np.searchsorted(c, xc, side="right") - 1, 0, c.size - 2)
    f = (xc - c[i]) / (c[i + 1] - c[i])
    w[i, np.arange(n)] = 1.0 - f
    w[i + 1, np.arange(n)] += f
    return w


def column_band_weights(params: dict, xy: np.ndarray) -> np.ndarray:
    """The ``(n_bands, n)`` weights of a banded column JSON (``calibrate_coupled
    --configs banded``) on the cells ``xy``, rebuilt from the centres and kind it
    records."""
    centres = params.get("band_centres_km")
    if not params.get("banded") or centres is None:
        raise ValueError("not a banded column: needs 'banded' and 'band_centres_km'")
    if len(centres) != len(params["banded"]):
        raise ValueError(f"banded column has {len(params['banded'])} parameter sets but "
                         f"{len(centres)} band centres")
    return band_weights(xy, centres, params.get("band_kind", "hat"))
