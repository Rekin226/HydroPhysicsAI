# Stage-3 Zonal Remediation (P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Stage-3 flow model's single homogeneous parameter set with a
structural proximal/mid/distal zonation (26 free parameters), then re-run the Stage-3
k-fold gate against pre-registered decision rules.

**Architecture:** A new pure module `hydrophysics/twin/zones.py` maps EPSG:3826 eastings to
a zone id. `fit_flow` gains `param_mode="zonal"`, which optimises one small tensor per
(parameter, zone) and gathers it to `(n_layers, n_active)` on every forward call — exactly
the trick `homogeneous` already uses with `.expand`, swapping the broadcast for an advanced
index so autograd accumulates gradients per zone. `FlowModel`'s constructor and parameter
shapes are untouched. Because the zonal fit copies its expanded per-cell field back into
`model.log_T/log_S/log_L` (as `homogeneous` does), the existing evaluation path
`_predict_homogeneous` works for zonal with only its guard widened.

**Tech Stack:** Python 3.12, PyTorch (float64, CPU/CUDA), NumPy, pandas, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-29-choushui-stage3-zonal-remediation-design.md`

## Global Constraints

- **Zone ids are `0 = proximal (E)`, `1 = mid`, `2 = distal (W)`.** Fixed everywhere.
- **Default boundaries: `proximal_km = 205.0`, `distal_km = 182.0`** (EPSG:3826 easting, km).
- **Zone intervals are half-open, high side inclusive:** `x_km >= 205` → proximal;
  `182 <= x_km < 205` → mid; `x_km < 182` → distal.
- **Free-parameter count is exactly 26** for 4 layers with both drivers active:
  proximal 2 + mid 11 + distal 11 + global 2.
- **Proximal `log_L` is FIXED at `BOUNDS["log_L"][1]` (= `log(1e-1)`), not optimised.**
  It is a constant buffer and must never appear in the optimiser's parameter list. This
  encodes "the proximal fan has no aquitards" structurally, at zero parameter cost.
- **`bounds_hit` for zonal is reported PER ZONE, never pooled.** A pinned proximal `log_T`
  must not be maskable by interior mid/distal values.
- **`homogeneous` and `percell` behaviour must not change.** Their tests are the regression
  gate.
- **ruff config:** line-length 100, rules `E4, E7, E9, F, I, B, UP, SIM`.
- **Verified data constants** (measured 2026-08-31 against the real fan data at `dx=1000`,
  both match the spec exactly — a future mismatch means the data or the boundary moved and
  must be investigated, not edited away):
  - 2,148 active cells split **264 / 1,235 / 649**
  - 136 well entries over 66 physical sites; sites split **12 / 33 / 21**
- **Pre-registered decision rules (spec §6) are fixed before any run.** PASS = clamp
  released AND mean margin > 0. PARTIAL = clamp released, margin improves but stays < 0.
  FAIL = clamp still pinned → stop, do not retry. Do not read the result post-hoc; this
  stage has already produced four confident wrong answers that way.

---

### Task 1: Zone definition module

**Files:**
- Create: `hydrophysics/twin/zones.py`
- Test: `tests/test_twin_zones.py`

**Interfaces:**
- Consumes: nothing (pure NumPy).
- Produces:
  - `PROXIMAL: int = 0`, `MID: int = 1`, `DISTAL: int = 2`
  - `ZONE_NAMES: tuple[str, str, str] = ("proximal", "mid", "distal")`
  - `N_ZONES: int = 3`
  - `fan_zones(xy: np.ndarray, proximal_km: float = 205.0, distal_km: float = 182.0) -> np.ndarray`
    — takes `(n, 2)` EPSG:3826 metres, returns `(n,)` `int64` zone ids.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_twin_zones.py`:

```python
import os

import numpy as np
import pytest

from hydrophysics.twin.zones import (
    DISTAL,
    MID,
    N_ZONES,
    PROXIMAL,
    ZONE_NAMES,
    fan_zones,
)

POLYGON = ("chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/"
           "Zhuoshui Alluvial Fan.json")
STATIONS = "AMP_V2/data/fan_stations.parquet"
WELLS_DIR = "AMP_V2/data/wells"


def _xy_km(*xs_km):
    """(n, 2) EPSG:3826 metres from a list of eastings in km; northing is irrelevant."""
    return np.array([[x * 1000.0, 2_640_000.0] for x in xs_km], dtype="float64")


def test_zone_constants_are_the_documented_ids():
    assert (PROXIMAL, MID, DISTAL) == (0, 1, 2)
    assert ZONE_NAMES == ("proximal", "mid", "distal")
    assert N_ZONES == 3


def test_fan_zones_assigns_each_zone_by_easting():
    z = fan_zones(_xy_km(214.0, 195.0, 170.0))
    assert z.tolist() == [PROXIMAL, MID, DISTAL]


def test_fan_zones_uses_half_open_intervals_inclusive_on_the_high_side():
    """205.0 is proximal, 204.999 is mid; 182.0 is mid, 181.999 is distal. The
    convention is documented in the spec and every count in this plan depends on it.
    """
    assert fan_zones(_xy_km(205.0))[0] == PROXIMAL
    assert fan_zones(_xy_km(204.999))[0] == MID
    assert fan_zones(_xy_km(182.0))[0] == MID
    assert fan_zones(_xy_km(181.999))[0] == DISTAL


def test_fan_zones_covers_every_point_with_exactly_one_zone_no_gaps_no_overlaps():
    xs = np.linspace(150.0, 240.0, 4001)
    z = fan_zones(_xy_km(*xs))
    assert z.shape == (xs.size,)
    assert set(np.unique(z).tolist()) <= {PROXIMAL, MID, DISTAL}
    assert np.isin(z, [PROXIMAL, MID, DISTAL]).all()


def test_fan_zones_is_monotone_west_to_east():
    """Zone id must decrease as easting increases: distal -> mid -> proximal, no
    interleaving. Guards against an off-by-one in the nested where().
    """
    xs = np.linspace(150.0, 240.0, 2001)
    z = fan_zones(_xy_km(*xs))
    assert (np.diff(z) <= 0).all()


def test_fan_zones_honours_custom_boundaries():
    z = fan_zones(_xy_km(190.0, 180.0), proximal_km=186.0, distal_km=178.0)
    assert z.tolist() == [PROXIMAL, MID]


def test_fan_zones_is_pure_and_deterministic():
    xy = _xy_km(214.0, 195.0, 170.0)
    before = xy.copy()
    a = fan_zones(xy)
    b = fan_zones(xy)
    assert np.array_equal(a, b)
    assert np.array_equal(xy, before)      # input not mutated
    assert a.dtype == np.int64


def test_fan_zones_returns_an_empty_array_for_empty_input():
    z = fan_zones(np.zeros((0, 2), dtype="float64"))
    assert z.shape == (0,)
    assert z.dtype == np.int64


def test_fan_zones_rejects_a_wrong_shaped_array():
    with pytest.raises(ValueError):
        fan_zones(np.zeros((5, 3), dtype="float64"))


def test_fan_zones_rejects_reversed_boundaries():
    """distal_km must sit west of proximal_km, or the mid zone is empty and every
    count downstream is silently wrong.
    """
    with pytest.raises(ValueError):
        fan_zones(_xy_km(200.0), proximal_km=180.0, distal_km=190.0)


@pytest.mark.skipif(not os.path.exists(POLYGON), reason="fan polygon not available")
def test_default_boundaries_split_the_real_grid_264_1235_649():
    """Spec §4/§7 pre-registered count, verified 2026-08-31 against the real polygon at
    dx=1000. A mismatch means the polygon or the boundary moved -- investigate it, do
    not edit this number.
    """
    from hydrophysics.twin.grid import build_grid

    grid = build_grid(POLYGON, dx=1000.0)
    z = fan_zones(grid.centroids())
    assert grid.n_active == 2148
    assert [int((z == k).sum()) for k in range(N_ZONES)] == [264, 1235, 649]


@pytest.mark.skipif(
    not (os.path.exists(POLYGON) and os.path.exists(STATIONS) and os.path.isdir(WELLS_DIR)),
    reason="fan well data not available",
)
def test_default_boundaries_split_the_66_calibration_sites_12_33_21():
    """Spec §4.1/§7 pre-registered count, verified 2026-08-31. The proximal zone holding
    only 12 of 66 sites is exactly why §5 gives it 2 free parameters and not 11.
    """
    import pandas as pd

    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.heads import build_head_field

    grid = build_grid(POLYGON, dx=1000.0)
    stn = pd.read_parquet(STATIONS)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(WELLS_DIR, stn)
    xy = np.array(
        [hf.xy[w] for w in range(len(hf))
         if grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1])) is not None],
        dtype="float64",
    )
    assert xy.shape[0] == 136
    sites = np.unique(np.round(xy, 3), axis=0)
    assert sites.shape[0] == 66
    z = fan_zones(sites)
    assert [int((z == k).sum()) for k in range(N_ZONES)] == [12, 33, 21]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_twin_zones.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'hydrophysics.twin.zones'`

- [ ] **Step 3: Write the implementation**

Create `hydrophysics/twin/zones.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_twin_zones.py -v`
Expected: PASS (12 tests; the two real-data tests skip if the data is absent)

- [ ] **Step 5: Lint**

Run: `uv run ruff check hydrophysics/twin/zones.py tests/test_twin_zones.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/twin/zones.py tests/test_twin_zones.py
git commit --author="Claude <noreply@anthropic.com>" -m "feat(twin): fan_zones, proximal/mid/distal banding for Stage-3 zonation

Boundaries: proximal/mid at 205 km is constrained by the layer-3/4 screen
pinch-out at 207.9 km; mid/distal at 182 km is an equal-width default and
spec 4.2 requires a sensitivity check at 178 and 186 km.

Verified splits: 2148 cells -> 264/1235/649, 66 sites -> 12/33/21."
```

---

### Task 2: Zonal parameter factory and expansion

**Files:**
- Modify: `hydrophysics/twin/calibrate_flow.py` (add helpers next to
  `_make_homogeneous_params` at line 224)
- Test: `tests/test_twin_flow.py` (append)

**Interfaces:**
- Consumes: `fan_zones`, `PROXIMAL`, `MID`, `DISTAL`, `N_ZONES`, `ZONE_NAMES` from Task 1;
  `BOUNDS` and `_clamp_` already in `calibrate_flow.py`.
- Produces:
  - `_base_param_name(name: str) -> str` — `"log_T_mid"` → `"log_T"`, `"log_eta"` → `"log_eta"`
  - `_make_zonal_params(model, use_pumping: bool = False, use_recharge: bool = False) -> dict[str, nn.Parameter]`
    with keys `log_T_proximal`, `log_S_proximal`, `log_T_mid`, `log_S_mid`, `log_L_mid`,
    `log_T_distal`, `log_S_distal`, `log_L_distal`, plus optional `log_eta`,
    `recharge_frac_logit`
  - `_expand_zonal(theta, zone_t: torch.Tensor, n_layers: int) -> tuple[Tensor, Tensor, Tensor | None]`
    returning `(log_T, log_S, log_L)` each `(n_layers, A)` / `(n_layers - 1, A)`
  - `_zonal_bounds_hit(theta) -> dict[str, dict[str, int]]` keyed by zone name plus
    `"global"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_twin_flow.py`:

```python
def test_make_zonal_params_has_exactly_26_free_parameters():
    """Spec §5: proximal 2 (one merged aquifer) + mid 11 + distal 11 + global 2.
    Fewer than naive 3x uniform zoning's 35, and every one physically motivated.
    """
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    assert sum(p.numel() for p in theta.values()) == 26
    assert theta["log_T_proximal"].shape == (1, 1)
    assert theta["log_S_proximal"].shape == (1, 1)
    assert theta["log_T_mid"].shape == (4, 1)
    assert theta["log_L_mid"].shape == (3, 1)
    assert theta["log_T_distal"].shape == (4, 1)
    assert theta["log_L_distal"].shape == (3, 1)


def test_make_zonal_params_does_not_expose_a_proximal_log_L():
    """Spec §5: fixing proximal log_L at the top of its range IS the statement 'there is
    no aquitard here'. It must be a constant, never an optimised parameter -- if it
    appears in theta the optimiser can walk it away from the published geology.
    """
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    assert "log_L_proximal" not in theta


def test_make_zonal_params_without_drivers_drops_the_two_globals():
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=False, use_recharge=False)
    assert sum(p.numel() for p in theta.values()) == 24
    assert "log_eta" not in theta
    assert "recharge_frac_logit" not in theta


def test_base_param_name_strips_zone_suffixes_but_not_log_eta():
    from hydrophysics.twin.calibrate_flow import BOUNDS, _base_param_name

    assert _base_param_name("log_T_mid") == "log_T"
    assert _base_param_name("log_S_proximal") == "log_S"
    assert _base_param_name("log_L_distal") == "log_L"
    assert _base_param_name("log_eta") == "log_eta"
    assert _base_param_name("recharge_frac_logit") == "recharge_frac_logit"
    for name in ("log_T_mid", "log_S_proximal", "log_L_distal", "log_eta"):
        assert _base_param_name(name) in BOUNDS


def test_expand_zonal_gathers_each_zones_value_to_its_own_cells():
    """The expansion is an advanced index into a (k, 3) column stack, so a cell must
    receive its own zone's value and nothing else.
    """
    from hydrophysics.twin.calibrate_flow import _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m)
    with torch.no_grad():
        theta["log_T_proximal"].fill_(1.0)
        theta["log_T_mid"].fill_(2.0)
        theta["log_T_distal"].fill_(3.0)
    zone_t = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    log_T, log_S, log_L = _expand_zonal(theta, zone_t, n_layers=4)
    assert log_T.shape == (4, 5)
    assert log_S.shape == (4, 5)
    assert log_L.shape == (3, 5)
    assert log_T[0].tolist() == [1.0, 2.0, 3.0, 2.0, 1.0]
    assert torch.allclose(log_T[3], log_T[0])       # proximal value shared across layers


def test_expand_zonal_shares_one_value_across_all_proximal_layers():
    """Spec §5: the proximal zone is ONE merged aquifer, so its four layers must carry
    an identical log_T and log_S -- not four values that happen to start equal.
    """
    from hydrophysics.twin.calibrate_flow import _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m)
    with torch.no_grad():
        theta["log_T_mid"].copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64))
    zone_t = torch.tensor([0, 1], dtype=torch.long)
    log_T, log_S, _ = _expand_zonal(theta, zone_t, n_layers=4)
    assert len(set(log_T[:, 0].tolist())) == 1       # proximal: one value, four layers
    assert log_T[:, 1].tolist() == [1.0, 2.0, 3.0, 4.0]   # mid: four distinct layers
    assert len(set(log_S[:, 0].tolist())) == 1


def test_expand_zonal_pins_proximal_log_L_at_the_upper_bound():
    """The merged-aquifer statement, checked at the value level: proximal leakage sits at
    the top of BOUNDS, and it is a constant, so it carries no gradient.
    """
    from hydrophysics.twin.calibrate_flow import BOUNDS, _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m)
    zone_t = torch.tensor([0, 1, 2], dtype=torch.long)
    _, _, log_L = _expand_zonal(theta, zone_t, n_layers=4)
    assert torch.allclose(log_L[:, 0],
                       torch.full((3,), BOUNDS["log_L"][1], dtype=torch.float64))


def test_expand_zonal_accumulates_gradient_onto_each_zones_parameter():
    """Advanced indexing must scatter-add the per-cell gradient back onto the small
    per-zone tensor, the same way homogeneous mode relies on expand-backward. If a zone
    receives a zero or None gradient it is being silently frozen -- this project has
    already shipped a backward() returning None for log_T once.
    """
    from hydrophysics.twin.calibrate_flow import _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    zone_t = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
    log_T, log_S, log_L = _expand_zonal(theta, zone_t, n_layers=4)
    (log_T.sum() + log_S.sum() + log_L.sum()).backward()
    for name in ("log_T_proximal", "log_S_proximal", "log_T_mid", "log_S_mid",
                 "log_L_mid", "log_T_distal", "log_S_distal", "log_L_distal"):
        gr = theta[name].grad
        assert gr is not None, f"{name} received no gradient"
        assert torch.isfinite(gr).all(), f"{name} gradient is not finite"
        assert (gr != 0).all(), f"{name} gradient is zero -- the zone is frozen"
    # proximal log_T is shared across 4 layers x 2 cells -> gradient of 8
    assert float(theta["log_T_proximal"].grad) == pytest.approx(8.0)
    # mid log_T is per-layer over 3 cells -> gradient of 3 per layer
    assert theta["log_T_mid"].grad.flatten().tolist() == pytest.approx([3.0] * 4)


def test_zonal_bounds_hit_reports_per_zone_and_never_pools():
    """Spec §6 primary rule: a pinned proximal log_T must not be maskable by interior
    mid/distal values. Pin proximal at the lower clamp, leave the rest interior, and the
    report must still show it.
    """
    from hydrophysics.twin.calibrate_flow import (
        BOUNDS,
        _make_zonal_params,
        _zonal_bounds_hit,
    )

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    with torch.no_grad():
        theta["log_T_proximal"].fill_(BOUNDS["log_T"][0])
        theta["log_T_mid"].fill_(math.log(500.0))
        theta["log_T_distal"].fill_(math.log(500.0))
    hits = _zonal_bounds_hit(theta)
    assert set(hits) == {"proximal", "mid", "distal", "global"}
    assert hits["proximal"]["log_T"] == 1
    assert hits["mid"]["log_T"] == 0
    assert hits["distal"]["log_T"] == 0
    assert "log_L" not in hits["proximal"]        # fixed, not a free parameter
    assert "log_eta" in hits["global"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_twin_flow.py -k "zonal or base_param_name" -v`
Expected: FAIL — `ImportError: cannot import name '_make_zonal_params'`

- [ ] **Step 3: Write the implementation**

In `hydrophysics/twin/calibrate_flow.py`, add to the imports near line 75:

```python
from .zones import DISTAL, MID, N_ZONES, PROXIMAL, ZONE_NAMES, fan_zones
```

Then insert after `_make_homogeneous_params` (which ends at line 250):

```python
def _base_param_name(name: str) -> str:
    """``"log_T_mid"`` -> ``"log_T"``. Zonal theta keys carry a zone suffix; BOUNDS and
    _clamp_ are keyed on the bare physical name. Only the three known zone suffixes are
    stripped, so ``log_eta`` and ``recharge_frac_logit`` survive untouched.
    """
    for zone in ZONE_NAMES:
        suffix = f"_{zone}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _make_zonal_params(model: FlowModel, use_pumping: bool = False,
                       use_recharge: bool = False) -> dict[str, nn.Parameter]:
    """Structural proximal/mid/distal parameters -- 26 free values for a 4-layer model
    with both drivers, against the homogeneous mode's 13 (spec §5).

    Zones differ in **form**, not only in value:

    - **proximal** is one merged aquifer: a single ``log_T`` and a single ``log_S``
      shared across all four layers, and NO ``log_L`` at all. Its leakage is pinned at
      the top of BOUNDS by ``_expand_zonal`` as a constant. That IS the published
      geology -- thick gravel, indistinct stratification, unrestricted vertical flow --
      encoded structurally rather than left for the optimiser to discover. 2 parameters.
    - **mid** and **distal** keep the full 4 aquifers + 3 aquitards. 11 each.
    - **global**: ``log_eta`` and the recharge fraction, as in homogeneous mode. 2.

    Shapes are ``(k, 1)`` so ``_expand_zonal`` can column-stack them into ``(k, N_ZONES)``
    and gather to ``(k, n_active)``. Initialised from the model's own uniform starting
    values, so a zonal run and a homogeneous run start from the same physics.
    """
    log_T0 = model.log_T[:, :1].detach().clone()
    log_S0 = model.log_S[:, :1].detach().clone()
    theta = {
        # one merged aquifer: mean of the layer starts, a single shared value
        "log_T_proximal": nn.Parameter(log_T0.mean(dim=0, keepdim=True)),
        "log_S_proximal": nn.Parameter(log_S0.mean(dim=0, keepdim=True)),
        "log_T_mid": nn.Parameter(log_T0.clone()),
        "log_S_mid": nn.Parameter(log_S0.clone()),
        "log_T_distal": nn.Parameter(log_T0.clone()),
        "log_S_distal": nn.Parameter(log_S0.clone()),
    }
    if model.n_layers > 1:
        log_L0 = model.log_L[:, :1].detach().clone()
        theta["log_L_mid"] = nn.Parameter(log_L0.clone())
        theta["log_L_distal"] = nn.Parameter(log_L0.clone())
    dev = model.log_T.device
    if use_pumping:
        theta["log_eta"] = nn.Parameter(
            torch.tensor(float(np.log(0.3)), dtype=torch.float64, device=dev))
    if use_recharge:
        theta["recharge_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))
    return theta


def _expand_zonal(theta: dict[str, torch.Tensor], zone_t: torch.Tensor,
                  n_layers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Column-stack the per-zone tensors into ``(k, N_ZONES)`` and gather to ``(k, A)``.

    The gather ``cols[:, zone_t]`` is advanced indexing, whose backward is a scatter-add:
    every cell's gradient accumulates onto its own zone's small tensor, exactly the way
    homogeneous mode relies on expand-backward summing over the broadcast axis. That is
    what lets ``FlowModel``'s frozen constructor and parameter shapes stay untouched.

    Proximal ``log_L`` is a CONSTANT at the upper bound, not a parameter: the four
    proximal layers equilibrate instead of being independently fitted.
    """
    dev = theta["log_T_mid"].device
    prox_T = theta["log_T_proximal"].expand(n_layers, 1)
    prox_S = theta["log_S_proximal"].expand(n_layers, 1)
    cols_T = torch.cat([prox_T, theta["log_T_mid"], theta["log_T_distal"]], dim=1)
    cols_S = torch.cat([prox_S, theta["log_S_mid"], theta["log_S_distal"]], dim=1)
    assert cols_T.shape == (n_layers, N_ZONES)
    log_T = cols_T[:, zone_t]
    log_S = cols_S[:, zone_t]
    log_L = None
    if n_layers > 1 and "log_L_mid" in theta:
        prox_L = torch.full((n_layers - 1, 1), BOUNDS["log_L"][1],
                            dtype=torch.float64, device=dev)
        cols_L = torch.cat([prox_L, theta["log_L_mid"], theta["log_L_distal"]], dim=1)
        log_L = cols_L[:, zone_t]
    return log_T, log_S, log_L


def _zonal_bounds_hit(theta: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    """Clamp every zonal parameter into BOUNDS in place and report hits **per zone**.

    Spec §6's primary decision rule reads this. Pooling would let an interior mid value
    mask a pinned proximal one, and a model still fitting outside the measured physical
    range gives confident wrong counterfactuals -- the failure that matters at the
    operational bar.
    """
    report: dict[str, dict[str, int]] = {name: {} for name in ZONE_NAMES}
    report["global"] = {}
    for name, par in theta.items():
        base = _base_param_name(name)
        if base not in BOUNDS:
            continue                     # recharge_frac_logit: unconstrained by design
        bucket = next((z for z in ZONE_NAMES if name.endswith(f"_{z}")), "global")
        report[bucket][base] = _clamp_({base: par})[base]
    return report
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_twin_flow.py -k "zonal or base_param_name" -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/calibrate_flow.py tests/test_twin_flow.py
git commit --author="Claude <noreply@anthropic.com>" -m "feat(twin): zonal parameter factory and per-zone expansion

26 free parameters: proximal 2 (one merged aquifer, log_L pinned at the
upper bound as a constant), mid 11, distal 11, global 2. The gather
cols[:, zone] scatter-adds gradients back per zone, so FlowModel's
constructor and parameter shapes stay untouched.

bounds_hit is reported per zone so a pinned proximal log_T cannot be
masked by interior mid/distal values."
```

---

### Task 3: Wire `param_mode="zonal"` into `fit_flow`

**Files:**
- Modify: `hydrophysics/twin/calibrate_flow.py:250-281` (signature + validation),
  `:415` (return), and add the zonal branch
- Test: `tests/test_twin_flow.py` (append)

**Interfaces:**
- Consumes: `_make_zonal_params`, `_expand_zonal`, `_zonal_bounds_hit` from Task 2.
- Produces: `fit_flow(..., param_mode="zonal", zone_of_cell: np.ndarray | None = None)`
  returning the usual dict, where `bounds_hit` is the nested per-zone dict and
  `theta` keys carry zone suffixes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_twin_flow.py`:

```python
def _zoned_synthetic_case(seed=0):
    """The synthetic case, plus a zone assignment that splits the square grid into three
    west-east bands so all three zones carry cells.
    """
    from hydrophysics.twin.zones import fan_zones

    g, h, rech, obs_idx, obs_layer = _synthetic_case(seed=seed)
    xy = g.centroids()
    # the uniform grid spans 0..4100 m; rescale the eastings onto 170..215 km so the
    # default boundaries (205 / 182) cut it into three non-empty bands
    span = xy[:, 0].max() - xy[:, 0].min()
    x_km = 170.0 + 45.0 * (xy[:, 0] - xy[:, 0].min()) / max(span, 1e-9)
    zoned = np.column_stack([x_km * 1000.0, xy[:, 1]])
    return g, h, rech, obs_idx, obs_layer, fan_zones(zoned)


def test_fit_flow_zonal_reports_the_mode_and_runs():
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=30, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    assert out["param_mode"] == "zonal"
    assert math.isfinite(out["loss"])
    assert math.isfinite(out["r2"])


def test_fit_flow_zonal_has_the_expected_free_parameter_count():
    """2 layers: proximal 2 + mid (2+2+1) + distal (2+2+1) = 12, no drivers here."""
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=1, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    assert out["n_params"] == 2 + 5 + 5


def test_fit_flow_zonal_requires_a_zone_assignment():
    """Running zonal without zones would silently fall back to something -- refuse."""
    g, h, rech, obs_idx, obs_layer, _ = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError, match="zone_of_cell"):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="zonal", zone_of_cell=None)


def test_fit_flow_zonal_rejects_a_zone_vector_of_the_wrong_length():
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="zonal", zone_of_cell=zones[:-1])


def test_fit_flow_zonal_reports_bounds_hit_per_zone():
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=5, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    hits = out["bounds_hit"]
    assert set(hits) == {"proximal", "mid", "distal", "global"}
    assert "log_T" in hits["proximal"] and "log_T" in hits["mid"]
    assert "log_L" not in hits["proximal"]


def test_fit_flow_zonal_writes_a_piecewise_constant_field_back_to_the_model():
    """The copy-back must produce a field that is constant WITHIN each zone and
    (generally) different between them. This is what lets the existing evaluation path
    read model.log_T directly for zonal, exactly as it does for homogeneous.
    """
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
             epochs=20, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    for zone_id in (0, 1, 2):
        sel = torch.tensor(zones == zone_id)
        assert sel.any(), f"zone {zone_id} has no cells in this fixture"
        block = m.log_T[0][sel]
        assert torch.allclose(block, block[0].expand_as(block))


def test_fit_flow_zonal_gradients_reach_every_zone_and_are_finite():
    """The _ImplicitSolve.backward returning None for log_T is a bug this project has
    already shipped once. Check the real fit path, not just the expansion helper: after
    one step every zone's parameter must have moved.
    """
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    before = {k: v.detach().clone()
              for k, v in _make_zonal_params(m).items()}
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=3, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    for name, start in before.items():
        moved = np.asarray(out["theta"][name], dtype="float64")
        assert np.isfinite(moved).all(), f"{name} went non-finite"
        assert not np.allclose(moved, start.squeeze(-1).numpy()), \
            f"{name} did not move -- no gradient reached this zone"


def test_fit_flow_zonal_proximal_layers_equilibrate():
    """Spec §7: with proximal log_L fixed high, heads across the proximal layers must
    actually converge -- assert it rather than assuming the fixed leakage did its job.
    Compare against the distal zone, where leakage is free and low.
    """
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    fit = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=40, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    assert fit["param_mode"] == "zonal"
    A = g.n_active
    h0 = torch.zeros(2, A, dtype=torch.float64)
    with torch.no_grad():
        hh = m(h0, rech, torch.zeros(2, A, rech.shape[-1], dtype=torch.float64),
               rech.shape[-1])
    prox = torch.tensor(zones == 0)
    dist = torch.tensor(zones == 2)
    spread_prox = (hh[0][prox] - hh[1][prox]).abs().mean()
    spread_dist = (hh[0][dist] - hh[1][dist]).abs().mean()
    assert spread_prox < spread_dist, (
        f"proximal layers did not equilibrate: spread {spread_prox:.4g} is not below "
        f"the distal spread {spread_dist:.4g}"
    )


def test_fit_flow_still_rejects_an_unknown_param_mode_after_zonal_lands():
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="zonal_typo")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_twin_flow.py -k "zonal" -v`
Expected: FAIL — `ValueError: param_mode must be 'homogeneous' or 'percell', got 'zonal'`

- [ ] **Step 3: Change the signature and validation**

In `fit_flow`, change the signature line (currently `calibrate_flow.py:256`) from

```python
             param_mode: str = "homogeneous", h0: torch.Tensor | None = None,
```

to

```python
             param_mode: str = "homogeneous", h0: torch.Tensor | None = None,
             zone_of_cell: np.ndarray | None = None,
```

and replace the validation block at `:280-281`:

```python
    if param_mode not in ("homogeneous", "percell", "zonal"):
        raise ValueError(
            "param_mode must be 'homogeneous', 'percell' or 'zonal', "
            f"got {param_mode!r}"
        )
    if param_mode == "zonal" and zone_of_cell is None:
        raise ValueError(
            "param_mode='zonal' needs zone_of_cell: an (n_active,) zone id per active "
            "cell, from hydrophysics.twin.zones.fan_zones(grid.centroids())"
        )
```

Add to the docstring, after the `"percell"` bullet:

```
    - ``"zonal"`` (spec 2026-08-29 §5): structural proximal/mid/distal zonation --
      proximal is one merged aquifer (2 parameters, ``log_L`` fixed at its upper bound),
      mid and distal each keep 4 aquifers + 3 aquitards (11 each), plus the same two
      global driver scalars: 26 parameters for a 4-layer model. Needs ``zone_of_cell``.
      ``bounds_hit`` comes back nested by zone, never pooled.
```

- [ ] **Step 4: Add the zonal branch**

Immediately before the `# homogeneous: optimise a small (k, 1) tensor ...` comment
(currently `calibrate_flow.py:337`), the homogeneous branch begins. Make the two modes
share it by parameterising the theta factory and the forward. Replace the block from

```python
    use_pumping = E is not None and ground_elev is not None
    use_recharge = recharge_field is not None
    theta = _make_homogeneous_params(model, use_pumping=use_pumping, use_recharge=use_recharge)
```

with

```python
    use_pumping = E is not None and ground_elev is not None
    use_recharge = recharge_field is not None
    zone_t = None
    if param_mode == "zonal":
        zone_arr = np.asarray(zone_of_cell, dtype="int64").reshape(-1)
        if zone_arr.shape[0] != A:
            raise ValueError(
                f"zone_of_cell has {zone_arr.shape[0]} entries but the grid has {A} "
                "active cells"
            )
        zone_t = torch.tensor(zone_arr, dtype=torch.long, device=dev)
        theta = _make_zonal_params(model, use_pumping=use_pumping,
                                   use_recharge=use_recharge)
    else:
        theta = _make_homogeneous_params(model, use_pumping=use_pumping,
                                         use_recharge=use_recharge)
```

Replace the `_forward` closure's first three lines:

```python
    def _forward() -> torch.Tensor:
        if zone_t is not None:
            log_T, log_S, log_L = _expand_zonal(theta, zone_t, model.n_layers)
        else:
            log_T = theta["log_T"].expand(-1, A)
            log_S = theta["log_S"].expand(-1, A)
            log_L = theta["log_L"].expand(-1, A) if "log_L" in theta else None
        return _rollout(
```

Replace the in-loop clamp line:

```python
        hits = (_zonal_bounds_hit(theta) if zone_t is not None
                else _clamp_({k: v for k, v in theta.items() if k in BOUNDS}))
```

Replace the copy-back block:

```python
    with torch.no_grad():
        h = _forward()
        pred = h[obs_layer, obs_idx, 1:]
        if zone_t is not None:
            zT, zS, zL = _expand_zonal(theta, zone_t, model.n_layers)
            model.log_T.copy_(zT)
            model.log_S.copy_(zS)
            if zL is not None and model.n_layers > 1:
                model.log_L.copy_(zL)
        else:
            model.log_T.copy_(theta["log_T"].expand(-1, A))
            model.log_S.copy_(theta["log_S"].expand(-1, A))
            if "log_L" in theta and model.n_layers > 1:
                model.log_L.copy_(theta["log_L"].expand(-1, A))
```

The `init_scatter`, `theta_out`, `n_params` and return blocks need no change: they iterate
`theta` generically, and `_make_zonal_params`'s keys carry the zone suffix, so
`out["theta"]["log_T_mid"]` falls out for free.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_twin_flow.py -k "zonal or base_param_name" -v`
Expected: PASS (18 tests: Task 2's 9 plus Task 3's 9)

- [ ] **Step 6: Verify the regression tests still pass**

Run: `uv run pytest tests/test_twin_flow.py -v`
Expected: PASS — in particular `test_fit_flow_recovers_synthetic_heads`,
`test_fit_flow_homogeneous_writes_a_spatially_constant_field_back_to_the_model`,
`test_fit_flow_percell_mode_runs_with_many_more_parameters`

- [ ] **Step 7: Commit**

```bash
git add hydrophysics/twin/calibrate_flow.py tests/test_twin_flow.py
git commit --author="Claude <noreply@anthropic.com>" -m "feat(twin): param_mode=zonal in fit_flow

Shares the homogeneous branch, swapping expand(-1, A) for a per-zone
gather. bounds_hit comes back nested by zone. The copy-back writes a
piecewise-constant per-cell field, so the existing evaluation path reads
model.log_T directly for zonal exactly as it does for homogeneous.

Tests assert the proximal layers actually equilibrate rather than
assuming the pinned log_L did its job."
```

---

### Task 4: Thread zones through the k-fold gate

**Files:**
- Modify: `hydrophysics/twin/calibrate_flow.py:505-513` (`kfold_wells` signature),
  `:570-576` (the `fit_flow` call), `:577` (the eval-path guard), `:437-440`
  (`_predict_homogeneous` docstring)
- Test: `tests/test_twin_flow.py` (append)

**Interfaces:**
- Consumes: `fit_flow(..., param_mode="zonal", zone_of_cell=...)` from Task 3.
- Produces: `kfold_wells(..., zone_of_cell: np.ndarray | None = None)` — the returned dict
  is unchanged in shape.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_twin_flow.py`:

```python
def test_kfold_wells_runs_in_zonal_mode_and_uses_the_calibrated_field():
    """The gate must reach the same evaluation path zonal's copy-back feeds. If the
    param_mode guard at the eval branch is not widened, zonal silently falls through to
    the static-forcing m(...) call and scores a model that never saw the drivers.
    """
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    out = kfold_wells(g, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                      n_layers=2, epochs=5, lr=0.1, n_folds=3, seed=0,
                      param_mode="zonal", zone_of_cell=zones)
    assert math.isfinite(out["r2_kfold"])
    assert math.isfinite(out["r2_idw"])
    assert out["n_wells"] == obs_idx.numel()


def test_kfold_wells_zonal_rejects_a_missing_zone_assignment():
    g, h, rech, obs_idx, obs_layer, _ = _zoned_synthetic_case()
    with pytest.raises(ValueError, match="zone_of_cell"):
        kfold_wells(g, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                    n_layers=2, epochs=1, n_folds=2, param_mode="zonal",
                    zone_of_cell=None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_twin_flow.py -k "kfold_wells_runs_in_zonal or kfold_wells_zonal" -v`
Expected: FAIL — `TypeError: kfold_wells() got an unexpected keyword argument 'zone_of_cell'`

- [ ] **Step 3: Write the implementation**

In `kfold_wells`, add to the signature (after `pump_layer: int = 1, recharge_layer: int = 0,`):

```python
                zone_of_cell: np.ndarray | None = None,
```

Add right after the docstring, before `W = obs_h.shape[0]`:

```python
    if param_mode == "zonal" and zone_of_cell is None:
        raise ValueError(
            "param_mode='zonal' needs zone_of_cell -- the zone assignment is a property "
            "of the grid, not of the fold, so it is passed once and reused unchanged"
        )
```

In the per-fold `fit_flow` call, add the argument:

```python
        fit = fit_flow(m, obs_h[keep], obs_idx[keep], obs_layer[keep], recharge,
                       E=E, ground_elev=ground_elev, epochs=epochs, lr=lr,
                       param_mode=param_mode, h0=h0_fold, recharge_field=recharge_field,
                       pump_layer=pump_layer, recharge_layer=recharge_layer,
                       zone_of_cell=zone_of_cell)
```

Widen the evaluation guard (currently `:577`):

```python
            if (param_mode in ("homogeneous", "zonal")
                    and (E is not None or recharge_field is not None)):
```

Extend `_predict_homogeneous`'s docstring first line to record why it serves both modes:

```python
    """Re-run the rollout for evaluation (e.g. at wells held out of a k-fold's fit),
    reusing ``model``'s own calibrated per-cell log_T/log_S/log_L. This serves BOTH
    ``homogeneous`` and ``zonal``: both copy their expanded field back into the model, so
    by this point the parameter *source* is identical and only the field's spatial
    structure differs (constant vs piecewise-constant). The name is historical.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_twin_flow.py -k "kfold" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/calibrate_flow.py tests/test_twin_flow.py
git commit --author="Claude <noreply@anthropic.com>" -m "feat(twin): thread zone_of_cell through kfold_wells

The zone assignment is a property of the grid, not of the fold, so it is
passed once and reused unchanged across folds -- same reasoning as the
driver fields. Widens the eval guard so zonal reaches the driver-aware
prediction path instead of silently falling through to static forcing."
```

---

### Task 5: CLI — `--param-mode zonal`, `--zone-boundaries`, per-zone reporting

**Files:**
- Modify: `hydrophysics/twin/calibrate_flow.py:725` (`--param-mode` choices), `:757` (new
  flag), `:762` (build zones), `:805-810` and `:833-845` (fit/gate calls and stdout),
  `:863-875` (csv row)
- Test: `tests/test_twin_flow.py` (append)

**Interfaces:**
- Consumes: `fan_zones` (Task 1), zonal `fit_flow`/`kfold_wells` (Tasks 3-4).
- Produces: CLI flags `--param-mode zonal` and `--zone-boundaries PROXIMAL_KM,DISTAL_KM`;
  new `stage3_flow.csv` columns `zone_proximal_km`, `zone_distal_km`, `zone_cell_counts`;
  `bounds_hit` column carries the nested per-zone dict for zonal.
- Also produces: `_parse_zone_boundaries(text: str) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_twin_flow.py`:

```python
def test_parse_zone_boundaries_reads_a_comma_pair():
    from hydrophysics.twin.calibrate_flow import _parse_zone_boundaries

    assert _parse_zone_boundaries("205,182") == (205.0, 182.0)
    assert _parse_zone_boundaries(" 186 , 178 ") == (186.0, 178.0)


def test_parse_zone_boundaries_rejects_malformed_input():
    from hydrophysics.twin.calibrate_flow import _parse_zone_boundaries

    for bad in ("205", "205,182,170", "a,b", ""):
        with pytest.raises(ValueError):
            _parse_zone_boundaries(bad)


def test_parse_zone_boundaries_rejects_a_reversed_pair():
    """The sensitivity check varies the mid/distal boundary; a transposed pair would
    silently empty the mid zone and produce a verdict on a two-zone model.
    """
    from hydrophysics.twin.calibrate_flow import _parse_zone_boundaries

    with pytest.raises(ValueError):
        _parse_zone_boundaries("178,186")


def test_cli_exposes_zonal_and_zone_boundaries():
    """Argparse-level check: the gate is expensive, so verify the surface without
    running it. Reading _PARAM_MODES rather than re-declaring the list means a silent
    removal of "zonal" fails here instead of at hour two of a run.
    """
    from hydrophysics.twin import calibrate_flow as cf

    assert "zonal" in cf._PARAM_MODES
    assert cf._DEFAULT_ZONE_BOUNDARIES == "205,182"
    with pytest.raises(SystemExit):
        cf.main(["--help"])          # the parser builds and --help exits 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_twin_flow.py -k "zone_boundaries or cli_exposes" -v`
Expected: FAIL — `ImportError: cannot import name '_parse_zone_boundaries'`

- [ ] **Step 3: Write the implementation**

Add near `_DEFAULT_PUMP_CENSUS` (around `calibrate_flow.py:710`):

```python
_PARAM_MODES = ("homogeneous", "percell", "zonal")
_DEFAULT_ZONE_BOUNDARIES = "205,182"


def _parse_zone_boundaries(text: str) -> tuple[float, float]:
    """``"205,182"`` -> ``(205.0, 182.0)``: the proximal/mid and mid/distal eastings in km.

    Spec §4.2 requires re-running the gate at 178 and 186 km, because the mid/distal
    boundary is an equal-width default with no independent justification. A transposed
    pair would silently empty the mid zone, so it is rejected rather than tolerated.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"--zone-boundaries wants PROXIMAL_KM,DISTAL_KM (e.g. '205,182'), got {text!r}"
        )
    try:
        proximal_km, distal_km = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"--zone-boundaries values must be numbers, got {text!r}") from exc
    if not distal_km < proximal_km:
        raise ValueError(
            f"--zone-boundaries wants PROXIMAL_KM,DISTAL_KM with the proximal boundary "
            f"east of the distal one; got proximal={proximal_km}, distal={distal_km}"
        )
    return proximal_km, distal_km
```

Change the `--param-mode` argument (`:725`):

```python
    ap.add_argument("--param-mode", choices=list(_PARAM_MODES), default="homogeneous")
    ap.add_argument("--zone-boundaries", default=_DEFAULT_ZONE_BOUNDARIES,
                    help="PROXIMAL_KM,DISTAL_KM for --param-mode zonal (default "
                         "'205,182'). The mid/distal value is an unjustified default; "
                         "spec 4.2 requires re-running at 178 and 186 to report whether "
                         "the verdict moves.")
```

After `grid = build_grid(args.polygon, dx=args.dx)` (`:762`), add:

```python
    zone_of_cell = zone_counts = None
    if args.param_mode == "zonal":
        proximal_km, distal_km = _parse_zone_boundaries(args.zone_boundaries)
        zone_of_cell = fan_zones(grid.centroids(), proximal_km=proximal_km,
                                 distal_km=distal_km)
        zone_counts = {name: int((zone_of_cell == i).sum())
                       for i, name in enumerate(ZONE_NAMES)}
        print(f"zones: proximal/mid at {proximal_km:.0f} km, mid/distal at "
              f"{distal_km:.0f} km -> cells {zone_counts}", flush=True)
        empty = [n for n, c in zone_counts.items() if c == 0]
        if empty:
            raise SystemExit(
                f"zone(s) {empty} contain no active cells at these boundaries; "
                "the gate would score a model with fewer zones than it reports"
            )
```

Add `zone_of_cell=zone_of_cell` to both the `fit_flow` call (`:806`) and the
`kfold_wells` call (`:834`).

Replace the two `bounds_hit` print lines (`:817` and `:846`) with a shared helper. Add
next to `_parse_zone_boundaries`:

```python
def _format_bounds_hit(hits: dict) -> str:
    """Per-zone dicts print one line per zone; flat dicts print as before.

    Spec §6's primary rule reads this, and pooling would let an interior mid value mask
    a pinned proximal one -- so the zonal form is never collapsed into a single number.
    """
    if hits and all(isinstance(v, dict) for v in hits.values()):
        return "\n".join(f"    bounds_hit[{zone}]={vals}" for zone, vals in hits.items())
    return f"  bounds_hit={hits}"
```

There are two print sites and their indentation differs. Change the `--fit-only` one at
`calibrate_flow.py:818` (8 spaces, inside `if args.fit_only:`) from

```python
        print(f"  in-sample R2={ins['r2']:+.3f}  bounds_hit={ins['bounds_hit']}  "
              f"fit_time={t_fit:.1f}s")
```

to

```python
        print(f"  in-sample R2={ins['r2']:+.3f}  fit_time={t_fit:.1f}s")
        print(_format_bounds_hit(ins["bounds_hit"]))
```

and change the gate one at `calibrate_flow.py:847` (4 spaces, function level) from

```python
    print(f"  in-sample R2={ins['r2']:+.3f}  bounds_hit={ins['bounds_hit']}  "
          f"fit_time={t_fit:.1f}s")
```

to

```python
    print(f"  in-sample R2={ins['r2']:+.3f}  fit_time={t_fit:.1f}s")
    print(_format_bounds_hit(ins["bounds_hit"]))
```

Add three columns to the `stage3_flow.csv` row (`:863`), after `"param_mode"`:

```python
                   "zone_proximal_km": (proximal_km if args.param_mode == "zonal"
                                        else ""),
                   "zone_distal_km": (distal_km if args.param_mode == "zonal" else ""),
                   "zone_cell_counts": (str(zone_counts) if args.param_mode == "zonal"
                                        else ""),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_twin_flow.py -k "zone_boundaries or cli_exposes" -v`
Expected: PASS

- [ ] **Step 5: Smoke the CLI surface**

Run: `uv run python -m hydrophysics.twin.calibrate_flow --help`
Expected: the help text lists `--param-mode {homogeneous,percell,zonal}` and
`--zone-boundaries`

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/twin/calibrate_flow.py tests/test_twin_flow.py
git commit --author="Claude <noreply@anthropic.com>" -m "feat(twin): --param-mode zonal, --zone-boundaries, per-zone bounds_hit

Per-zone bounds_hit prints one line per zone and lands in stage3_flow.csv
alongside the boundaries and cell counts used, so a run's verdict can be
read back against the geometry that produced it.

Refuses boundaries that empty a zone: a three-zone verdict on a two-zone
model is exactly the kind of silent wrong answer this stage keeps hitting."
```

---

### Task 6: Full-suite and lint verification gate

**Files:**
- Modify: none expected (fix whatever the run surfaces)

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: a green suite, which is the precondition for spending compute in Task 7.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: the pre-existing 117 collected / 116 passed / 1 skipped, PLUS the 36 new tests
(12 in `test_twin_zones.py`; 24 appended to `test_twin_flow.py` — 9 from Task 2, 9 from
Task 3, 2 from Task 4, 4 from Task 5), zero failures. The two real-data zone-split tests
in `test_twin_zones.py` show as skipped if the fan data is not present.

- [ ] **Step 2: Run ruff**

Run: `uv run ruff check hydrophysics tests`
Expected: `All checks passed!`

- [ ] **Step 3: Confirm the homogeneous path is byte-for-byte unaffected**

Run:
```bash
uv run pytest tests/test_twin_flow.py -k "homogeneous or percell or theis or synthetic" -v
```
Expected: PASS. If any of these changed behaviour, the shared-branch refactor in Task 3
leaked into `homogeneous` — stop and fix before Task 7.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit --author="Claude <noreply@anthropic.com>" -m "test(twin): green suite and ruff clean with zonal in place"
```

---

### Task 7: Run the gate and record the verdict

**Files:**
- Modify: `docs/superpowers/specs/2026-08-29-choushui-stage3-zonal-remediation-design.md` (§7)
- Modify: `.superpowers/sdd/2026-08-25-choushui-twin-plan-b/progress.md`
- Modify: `README.md` (model table — Plan B Task 7 Step 2, still outstanding)

**Interfaces:**
- Consumes: the CLI from Task 5.
- Produces: the P1 verdict, which gates P2.

**Read before running.** The decision rules are pre-registered in spec §6 and repeated in
this plan's Global Constraints. Do not read the result post-hoc. Cost is driven by CG
iteration count, not epoch count — a 400-epoch fold ran cheaper than a 100-epoch one. Do
not estimate runtime from epochs; fold times swing 12.6–33.4 min at identical settings.
Never write a shell wait like `until [ -f <task-id>.done ]`; no such marker exists.

- [ ] **Step 1: Smoke run at seed 0**

Run (background, ~2 h):
```bash
uv run python -m hydrophysics.twin.calibrate_flow \
  --param-mode zonal --epochs 400 --n-folds 5 --seed 0 \
  --out results/twin/zonal_seed0
```
Expected in stdout, before anything else is trusted:
- `zones: ... -> cells {'proximal': 264, 'mid': 1235, 'distal': 649}`
- `co-location rate=0.000` — if it is not 0.000 the folds leaked and the verdict is void
- `bounds_hit[proximal]=...`, `bounds_hit[mid]=...`, `bounds_hit[distal]=...`
- `n_params=26`

- [ ] **Step 2: Inspect before spending more**

Record from `results/twin/zonal_seed0/stage3_flow.csv`: `r2_kfold`, `r2_idw`, their
difference, and the per-zone `bounds_hit`. Compare the margin against the homogeneous
baseline of −0.048 ± 0.013.

Stop and escalate now, without running seeds 1–4, if a majority of `log_T` values remain
pinned across the three zones. That is spec §6's FAIL: the parameterization is not the
binding constraint and further zoning will not help.

- [ ] **Step 3: Run seeds 1–4**

Only if Step 2 was sane. Run (~8 h):
```bash
for s in 1 2 3 4; do
  uv run python -m hydrophysics.twin.calibrate_flow \
    --param-mode zonal --epochs 400 --n-folds 5 --seed "$s" \
    --out "results/twin/zonal_seed$s"
done
```

- [ ] **Step 4: Boundary sensitivity at 178 and 186 km**

Run at seed 0 only (~4 h):
```bash
uv run python -m hydrophysics.twin.calibrate_flow \
  --param-mode zonal --zone-boundaries 205,178 --epochs 400 --n-folds 5 --seed 0 \
  --out results/twin/zonal_b178
uv run python -m hydrophysics.twin.calibrate_flow \
  --param-mode zonal --zone-boundaries 205,186 --epochs 400 --n-folds 5 --seed 0 \
  --out results/twin/zonal_b186
```

- [ ] **Step 5: Compute the verdict**

Run:
```bash
uv run python - <<'PY'
import glob

import pandas as pd

rows = [pd.read_csv(p) for p in sorted(glob.glob("results/twin/zonal_seed*/stage3_flow.csv"))]
df = pd.concat(rows, ignore_index=True)
df["margin"] = df["r2_kfold"] - df["r2_idw"]
print(df[["seed", "r2_kfold", "r2_idw", "margin", "bounds_hit"]].to_string(index=False))
print(f"\nmean margin {df['margin'].mean():+.4f} +/- {df['margin'].std(ddof=1):.4f} "
      f"over {len(df)} seeds  (homogeneous baseline: -0.048 +/- 0.013)")
for p in sorted(glob.glob("results/twin/zonal_b*/stage3_flow.csv")):
    d = pd.read_csv(p)
    print(f"{p}: margin {float(d['r2_kfold'] - d['r2_idw']):+.4f}  "
          f"boundaries {d['zone_proximal_km'][0]},{d['zone_distal_km'][0]}")
PY
```

Apply spec §6 exactly as written:
- **PASS** — clamp released AND mean margin > 0.
- **PARTIAL** — clamp released, margin improves but stays < 0. Proceed to P2 with the
  limitation documented.
- **FAIL** — clamp still pinned. Stop, escalate to a design conversation, do not retry.

- [ ] **Step 6: Record the verdict in the spec**

Append to spec §7 the verdict, the per-seed table, the per-zone `bounds_hit`, and the two
sensitivity margins. State it plainly, pass or fail. If the verdict moves between 178 and
186 km, that is a result to report, not a knob to tune.

- [ ] **Step 7: Update the SDD ledger and the README model table**

Add the ruling and the gate result to
`.superpowers/sdd/2026-08-25-choushui-twin-plan-b/progress.md`. Update the README model
table (Plan B Task 7 Step 2, outstanding since the Plan B merge).

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-08-29-choushui-stage3-zonal-remediation-design.md \
        .superpowers/sdd/2026-08-25-choushui-twin-plan-b/progress.md README.md \
        results/twin
git commit --author="Claude <noreply@anthropic.com>" -m "docs(twin): Stage-3 zonal gate verdict, seeds 0-4 plus boundary sensitivity"
```

---

## Deferred, deliberately out of this plan

- **The distance-degradation analysis.** Code is committed (`8f53147`, `--dump-predictions`);
  only compute is outstanding. It asks whether the flow model degrades faster than IDW as
  held-out sites isolate — which, if true, breaks the argument for a physics model
  independently of any gate. Worth running, but it is not P1.
- **P2 Stage-4 coupling, P3 scenarios + UQ, P4 the 3D twin.** P4 can be prototyped in
  parallel against Plan A's compaction column, which passed its gate.
- **The four open questions in spec §10**: the unjustified mid/distal boundary, the
  northern lobe, `--dx 500` grid convergence not being like-for-like, and the over-tight
  CG tolerance.
