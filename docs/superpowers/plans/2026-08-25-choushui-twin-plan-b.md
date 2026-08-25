# Choushui Twin — Plan B (Stage 3): Differentiable Four-Layer Flow Solver

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GPU-native, end-to-end differentiable four-layer groundwater flow model of the Choushui fan whose transmissivity, storage, leakance and pumping are calibrated by gradient descent against 147 layer-coded head records — and which predicts held-out wells better than the IDW interpolation everything so far has relied on.

**Architecture:** Four stacked 2D layers on a 1 km grid over the fan polygon, coupled by vertical leakage through aquitards. Five-point finite volume, backward Euler, monthly steps. Gradients come from implicit differentiation of the linear solve rather than backprop through solver iterations. Pumping is not a free field: it is reconstructed from a census of 116,769 registered pumps' monthly electricity via `Q = η·E / (ρ g · lift)`, where lift depends on the model's own simulated head, so only a handful of efficiency parameters are unknown.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch 2.11+cu128 (RTX 4070 SUPER, 12 GB), pytest, ruff (line-length 100, E4/E7/E9/F/I/B/UP/SIM). `uv pip` for packages — the venv has no pip.

**Spec:** `docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md` (§2 forward model, §4 pumping, §7 staging, §9 validation)

## Global Constraints

- **Held-out wells and the 2019+ window are sealed.** Model selection uses inner splits only. Never tune against a held-out number. (Spec §8, §9.)
- **All physical parameters log-parameterized** for positivity: `T`, `S`, `L`, `η`.
- **Bound every parameter the loss is flat in.** Plan A lost a week to `log_tau` running to 9,400 days while the loss stayed flat. Anything unbounded gets a physically-defensible clamp, and the report says whether the bound binds.
- **No datum-dependent formulations.** Plan A's preconsolidation gate silently disabled itself at 7 of 14 sites because it compared head against absolute zero. Express thresholds relative to a state the model owns, never to a survey datum.
- **Baselines on identical arrays.** A comparison between an in-sample number and an out-of-sample one is not a comparison. Every gate reports its baseline computed on exactly the same cells.
- **Units:** metres, days, m³/day. Coordinates EPSG:3826 metres. Heads in metres.
- Run tests with `.venv/bin/python -m pytest`; lint with `.venv/bin/python -m ruff check hydrophysics tests`.
- Data under `data/`-named directories and `results/` are gitignored. Never commit agency data.
- Real data: `export HYDROMIND_GW_DATA="$(pwd)/chou-shui-data/chou-shui-data/data"`.

---

## Why 1 km and not 500 m

The fan polygon gives **8,567 active cells at 500 m** (34,268 four-layer unknowns) against **2,135 at 1 km** (8,540 unknowns). Plan B calibrates on a 12 GB card with a full 132-month backprop, and every calibration epoch solves 132 linear systems. **1 km is the base grid**; 500 m is a convergence check in Task 7, not the working resolution. If 1 km and 500 m disagree materially, that is a finding, not a failure.

## File Structure

| File | Responsibility |
|---|---|
| `hydrophysics/twin/grid.py` | Fan polygon → masked grid, layer geometry, cell↔coordinate mapping |
| `hydrophysics/twin/flow.py` | Differentiable four-layer flow solver (the core) |
| `hydrophysics/twin/pumping.py` | Electricity census → per-cell monthly abstraction |
| `hydrophysics/twin/calibrate_flow.py` | Stage-3 calibration driver + gate CLI |
| `tests/test_twin_grid.py` | Masking, geometry, index round-trips |
| `tests/test_twin_flow.py` | Theis, Hantush, mass balance, gradients, convergence |
| `tests/test_twin_pumping.py` | Energy→volume conversion, aggregation, lift coupling |

Reused unchanged: `twin/heads.py` (the 147-well field), `subsidence.idw_interp` (the baseline to beat), `twin/leveling.py`.

---

### Task 1: Grid and layer geometry

**Files:**
- Create: `hydrophysics/twin/grid.py`
- Test: `tests/test_twin_grid.py`

**Interfaces:**
- Consumes: the fan polygon at `chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json` (GeoJSON, EPSG:4326).
- Produces:
  - `class FanGrid` with fields `nx, ny, dx, x0, y0, mask (ny,nx) bool, n_active int`
  - `build_grid(polygon_path: str, dx: float = 1000.0) -> FanGrid`
  - `FanGrid.cell_of(x, y) -> tuple[int, int] | None` — coordinates to (row, col), `None` outside the mask
  - `FanGrid.active_index(x, y) -> int | None` — coordinates to flat active-cell index
  - `FanGrid.centroids() -> np.ndarray` — `(n_active, 2)` EPSG:3826 metres

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twin_grid.py
import numpy as np
import pytest

from hydrophysics.twin.grid import FanGrid, build_grid

POLY = ("chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/"
        "Zhuoshui Alluvial Fan.json")


def test_grid_masks_the_fan_polygon():
    g = build_grid(POLY, dx=1000.0)
    assert g.mask.shape == (g.ny, g.nx)
    # the fan is ~2,144 km2; at 1 km cells that is ~2,100 active cells
    assert 1900 < g.n_active < 2400
    assert g.n_active == int(g.mask.sum())
    # the bounding box is not all fan: masking must actually reject cells
    assert g.n_active < g.nx * g.ny * 0.75


def test_cell_lookup_round_trips_through_centroids():
    g = build_grid(POLY, dx=1000.0)
    c = g.centroids()
    assert c.shape == (g.n_active, 2)
    for i in (0, g.n_active // 2, g.n_active - 1):
        x, y = c[i]
        assert g.active_index(float(x), float(y)) == i


def test_coordinates_outside_the_fan_return_none():
    g = build_grid(POLY, dx=1000.0)
    assert g.active_index(0.0, 0.0) is None
    assert g.cell_of(0.0, 0.0) is None


def test_finer_grid_gives_more_cells_covering_similar_area():
    coarse = build_grid(POLY, dx=1000.0)
    fine = build_grid(POLY, dx=500.0)
    area_c = coarse.n_active * 1.0        # km2
    area_f = fine.n_active * 0.25         # km2
    assert fine.n_active > coarse.n_active
    assert abs(area_f - area_c) / area_c < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.grid'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/grid.py
"""Fan polygon -> masked regular grid, with coordinate <-> cell lookups.

The Choushui fan covers ~2,144 km2. At 1 km that is ~2,100 active cells (8,400 four-layer
unknowns), which fits a full 132-month differentiable rollout on a 12 GB card. 500 m gives
~8,600 cells and is used only as a convergence check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from pyproj import Transformer


@dataclass
class FanGrid:
    """Regular grid over the fan's bounding box, masked to the polygon."""

    nx: int
    ny: int
    dx: float
    x0: float          # western edge of column 0, EPSG:3826 metres
    y0: float          # southern edge of row 0
    mask: np.ndarray   # (ny, nx) bool, True inside the fan

    @property
    def n_active(self) -> int:
        return int(self.mask.sum())

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, col) for a coordinate, or None if outside the grid or the mask."""
        col = int((x - self.x0) // self.dx)
        row = int((y - self.y0) // self.dx)
        if not (0 <= row < self.ny and 0 <= col < self.nx):
            return None
        if not self.mask[row, col]:
            return None
        return row, col

    def active_index(self, x: float, y: float) -> int | None:
        """Flat index into the active-cell vector (row-major over masked cells)."""
        rc = self.cell_of(x, y)
        if rc is None:
            return None
        row, col = rc
        return int(self.mask[:row].sum() + self.mask[row, :col].sum())

    def centroids(self) -> np.ndarray:
        """(n_active, 2) cell-centre coordinates in EPSG:3826 metres."""
        rows, cols = np.nonzero(self.mask)
        return np.column_stack([self.x0 + (cols + 0.5) * self.dx,
                                self.y0 + (rows + 0.5) * self.dx])


def build_grid(polygon_path: str, dx: float = 1000.0) -> FanGrid:
    """Read the fan GeoJSON (EPSG:4326), reproject to 3826, and mask a dx-metre grid."""
    with open(polygon_path) as fh:
        gj = json.load(fh)
    ring = np.array(gj["features"][0]["geometry"]["coordinates"][0], dtype="float64")[:, :2]
    tf = Transformer.from_crs(4326, 3826, always_xy=True)
    X, Y = tf.transform(ring[:, 0], ring[:, 1])
    poly = MplPath(np.column_stack([X, Y]))

    x0, y0 = float(np.min(X)), float(np.min(Y))
    nx = int(np.ceil((float(np.max(X)) - x0) / dx))
    ny = int(np.ceil((float(np.max(Y)) - y0) / dx))
    cx = x0 + (np.arange(nx) + 0.5) * dx
    cy = y0 + (np.arange(ny) + 0.5) * dx
    GX, GY = np.meshgrid(cx, cy)
    inside = poly.contains_points(np.column_stack([GX.ravel(), GY.ravel()]))
    return FanGrid(nx=nx, ny=ny, dx=dx, x0=x0, y0=y0,
                   mask=inside.reshape(ny, nx))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_grid.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/grid.py tests/test_twin_grid.py
git commit -m "feat(twin): fan polygon -> masked grid with cell lookups"
```

---

### Task 2: Single-layer transient flow, verified against Theis

**Files:**
- Create: `hydrophysics/twin/flow.py`
- Test: `tests/test_twin_flow.py`

**Interfaces:**
- Consumes: `FanGrid` (Task 1).
- Produces:
  - `class FlowModel(torch.nn.Module)` — `__init__(self, grid: FanGrid, n_layers: int = 4, dt_days: float = 30.0, device=None)`
  - `FlowModel.forward(self, h0: Tensor, recharge: Tensor, pumping: Tensor, n_steps: int) -> Tensor` — `h0` is `(L, A)`, `recharge`/`pumping` are `(L, A, T)` in m/day and m³/day, returning heads `(L, A, T+1)`. `A = grid.n_active`.
  - Learnable log-parameters `log_T`, `log_S` each `(L, A)`; `log_L` is `(L-1, A)` (leakance between adjacent layers).
- In this task the layers are independent — `log_L` is declared but unused, so Task 3 adds leakage without changing the constructor.

**The one hard part.** Backward Euler needs a linear solve each step. Do **not** backpropagate through conjugate-gradient iterations — memory grows with iteration count and the gradient is only as good as the convergence. Solve with a matrix-free CG inside `torch.no_grad()`, then attach the gradient with a custom `autograd.Function` whose backward solves the *transposed* system with the same operator. That is exact and its memory does not depend on iteration count.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twin_flow.py
import numpy as np
import pytest
import torch
from scipy.special import exp1

from hydrophysics.twin.flow import FlowModel
from hydrophysics.twin.grid import FanGrid


def _uniform_grid(n=41, dx=100.0):
    """A square all-active grid, so the solver is tested without polygon masking."""
    return FanGrid(nx=n, ny=n, dx=dx, x0=0.0, y0=0.0,
                   mask=np.ones((n, n), dtype=bool))


def test_theis_drawdown_matches_the_analytical_solution():
    """Confined, homogeneous, single well: the solver must reproduce Theis."""
    g = _uniform_grid()
    T_val, S_val, Q = 500.0, 1e-4, 1000.0          # m2/day, -, m3/day
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    with torch.no_grad():
        m.log_T.fill_(float(np.log(T_val)))
        m.log_S.fill_(float(np.log(S_val)))

    A = g.n_active
    centre = g.active_index(20.5 * g.dx, 20.5 * g.dx)
    steps = 10
    pump = torch.zeros(1, A, steps)
    pump[0, centre, :] = Q
    h = m(torch.zeros(1, A), torch.zeros(1, A, steps), pump, steps)

    xy = g.centroids()
    c = xy[centre]
    t_days = steps * 1.0
    for probe_r in (300.0, 500.0):
        d = np.linalg.norm(xy - c, axis=1)
        i = int(np.argmin(np.abs(d - probe_r)))
        r = float(d[i])
        u = r ** 2 * S_val / (4 * T_val * t_days)
        analytic = Q / (4 * np.pi * T_val) * exp1(u)
        numeric = float(-(h[0, i, -1]))            # drawdown is positive
        assert numeric == pytest.approx(analytic, rel=0.20)


def test_mass_balance_closes_each_step():
    g = _uniform_grid(n=21)
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    A = g.n_active
    rech = torch.full((1, A, 5), 1e-3)
    h = m(torch.zeros(1, A), rech, torch.zeros(1, A, 5), 5)
    S = torch.exp(m.log_S)[0]
    cell_area = g.dx ** 2
    stored = float(((h[0, :, -1] - h[0, :, 0]) * S).sum() * cell_area)
    added = float(rech.sum() * cell_area)          # no-flow boundaries: nothing leaves
    assert stored == pytest.approx(added, rel=1e-4)


def test_gradients_reach_the_parameters_and_are_finite():
    g = _uniform_grid(n=15)
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    A = g.n_active
    h = m(torch.zeros(1, A), torch.full((1, A, 3), 1e-3), torch.zeros(1, A, 3), 3)
    h.sum().backward()
    for name in ("log_T", "log_S"):
        gr = getattr(m, name).grad
        assert gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0


def test_adjoint_gradient_matches_finite_differences():
    """The implicit-differentiation backward must agree with a numerical gradient."""
    g = _uniform_grid(n=9)
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    A = g.n_active
    rech = torch.full((1, A, 2), 1e-3)

    def loss_of(logT_delta: float) -> float:
        with torch.no_grad():
            m.log_T.fill_(float(np.log(500.0)) + logT_delta)
        return float(m(torch.zeros(1, A), rech, torch.zeros(1, A, 2), 2).pow(2).sum())

    with torch.no_grad():
        m.log_T.fill_(float(np.log(500.0)))
    out = m(torch.zeros(1, A), rech, torch.zeros(1, A, 2), 2).pow(2).sum()
    out.backward()
    analytic = float(m.log_T.grad.sum())
    eps = 1e-4
    numeric = (loss_of(eps) - loss_of(-eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.flow'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/flow.py
"""Differentiable multi-layer transient groundwater flow on a masked grid.

Five-point finite volume, backward Euler, monthly steps by default. Each step solves

    (S*A/dt + K) h^{n+1} = S*A/dt h^n + q

with K the symmetric conductance operator. The solve runs matrix-free under no_grad; the
gradient is attached by implicit differentiation (a transposed solve with the same
operator), so memory does not grow with iteration count and the gradient does not depend on
how tightly CG converged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .grid import FanGrid


def _neighbour_index(grid: FanGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Active-cell index pairs (i, j) for every shared face, each face listed once."""
    idx = -np.ones((grid.ny, grid.nx), dtype="int64")
    rows, cols = np.nonzero(grid.mask)
    idx[rows, cols] = np.arange(rows.size)
    a, b = [], []
    for dr, dc in ((0, 1), (1, 0)):
        r2, c2 = rows + dr, cols + dc
        ok = (r2 < grid.ny) & (c2 < grid.nx)
        ok[ok] &= grid.mask[r2[ok], c2[ok]]
        a.append(idx[rows[ok], cols[ok]])
        b.append(idx[r2[ok], c2[ok]])
    return (torch.as_tensor(np.concatenate(a)), torch.as_tensor(np.concatenate(b)))


class _ImplicitSolve(torch.autograd.Function):
    """y = M^{-1} b with an exact adjoint, where M is supplied as a matvec closure."""

    @staticmethod
    def forward(ctx, b, matvec, solve, *params):
        with torch.no_grad():
            y = solve(b)
        ctx.save_for_backward(y, *params)
        ctx.matvec, ctx.solve = matvec, solve
        return y

    @staticmethod
    def backward(ctx, grad_y):
        with torch.no_grad():
            lam = ctx.solve(grad_y)          # M is symmetric, so M^T == M
        return (lam, None, None) + tuple(None for _ in ctx.saved_tensors[1:])


def _cg(matvec, b, x0=None, tol=1e-8, maxiter=400):
    """Matrix-free conjugate gradient for a symmetric positive-definite operator."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = (r * r).sum()
    b_norm = (b * b).sum().sqrt().clamp(min=1e-30)
    for _ in range(maxiter):
        Ap = matvec(p)
        alpha = rs / (p * Ap).sum().clamp(min=1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if (rs_new.sqrt() / b_norm) < tol:
            break
        p = r + (rs_new / rs.clamp(min=1e-30)) * p
        rs = rs_new
    return x


class FlowModel(nn.Module):
    """Multi-layer transient flow. ``forward`` returns heads ``(L, A, T+1)``."""

    def __init__(self, grid: FanGrid, n_layers: int = 4, dt_days: float = 30.0,
                 device=None):
        super().__init__()
        self.grid = grid
        self.n_layers = int(n_layers)
        self.dt = float(dt_days)
        self.area = float(grid.dx) ** 2
        A = grid.n_active
        ia, ib = _neighbour_index(grid)
        self.register_buffer("ia", ia.to(device) if device else ia)
        self.register_buffer("ib", ib.to(device) if device else ib)
        z = torch.zeros(self.n_layers, A, device=device)
        self.log_T = nn.Parameter(z.clone() + float(np.log(500.0)))     # m2/day
        self.log_S = nn.Parameter(z.clone() + float(np.log(1e-4)))      # -
        self.log_L = nn.Parameter(torch.zeros(max(self.n_layers - 1, 1), A, device=device)
                                  + float(np.log(1e-4)))               # 1/day

    def _matvec(self, T, S):
        """Return a closure applying (S*area/dt + K) to a head vector of shape (L, A)."""
        ia, ib = self.ia, self.ib
        Tf = 2.0 * T[:, ia] * T[:, ib] / (T[:, ia] + T[:, ib]).clamp(min=1e-30)  # harmonic

        def mv(h):
            out = S * self.area / self.dt * h
            dh = h[:, ia] - h[:, ib]
            flux = Tf * dh
            out = out.index_add(1, ia, flux)
            out = out.index_add(1, ib, -flux)
            # Vertical leakage is added in Task 3; layers are independent here.
            return out

        return mv

    def forward(self, h0: torch.Tensor, recharge: torch.Tensor,
                pumping: torch.Tensor, n_steps: int) -> torch.Tensor:
        T = torch.exp(self.log_T)
        S = torch.exp(self.log_S)
        mv = self._matvec(T, S)
        h = h0
        out = [h0]
        for t in range(n_steps):
            q = (recharge[..., t] - pumping[..., t] / self.area) * self.area
            b = S * self.area / self.dt * h + q
            h = _ImplicitSolve.apply(b, mv, lambda rhs: _cg(mv, rhs), self.log_T, self.log_S)
            out.append(h)
        return torch.stack(out, dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -v`
Expected: 4 passed. If Theis fails, check the sign convention on `pumping` (positive = abstraction) and that the well cell's rate is divided by cell area consistently — do **not** loosen the tolerance to make it pass.

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/flow.py tests/test_twin_flow.py
git commit -m "feat(twin): differentiable single-layer flow verified against Theis"
```

---

### Task 3: Vertical leakage between layers, verified against Hantush

**Files:**
- Modify: `hydrophysics/twin/flow.py`
- Test: `tests/test_twin_flow.py`

**Interfaces:**
- Consumes: `FlowModel` from Task 2 (constructor unchanged).
- Produces: `forward` now couples layers through `log_L`, added to the `_matvec` closure where Task 2 left a comment.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_flow.py
def test_leakage_moves_water_between_layers_and_conserves_it():
    g = _uniform_grid(n=15)
    m = FlowModel(g, n_layers=2, dt_days=1.0)
    A = g.n_active
    with torch.no_grad():
        m.log_S.fill_(float(np.log(1e-3)))
        m.log_L.fill_(float(np.log(1e-3)))
    # recharge the upper layer only
    rech = torch.zeros(2, A, 8)
    rech[0] = 1e-3
    h = m(torch.zeros(2, A), rech, torch.zeros(2, A, 8), 8)
    assert float(h[0, :, -1].mean()) > 0.0
    assert float(h[1, :, -1].mean()) > 0.0            # leakage reached the lower layer
    assert float(h[0, :, -1].mean()) > float(h[1, :, -1].mean())

    S = torch.exp(m.log_S)
    stored = float(((h[:, :, -1] - h[:, :, 0]) * S).sum() * g.dx ** 2)
    added = float(rech.sum() * g.dx ** 2)
    assert stored == pytest.approx(added, rel=1e-4)


def test_zero_leakance_decouples_the_layers():
    g = _uniform_grid(n=11)
    m = FlowModel(g, n_layers=3, dt_days=1.0)
    A = g.n_active
    with torch.no_grad():
        m.log_L.fill_(-40.0)                          # effectively zero
    rech = torch.zeros(3, A, 4)
    rech[0] = 1e-3
    h = m(torch.zeros(3, A), rech, torch.zeros(3, A, 4), 4)
    assert float(h[1, :, -1].abs().max()) < 1e-9
    assert float(h[2, :, -1].abs().max()) < 1e-9


def test_four_layers_run_and_stay_finite():
    g = _uniform_grid(n=11)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    A = g.n_active
    h = m(torch.zeros(4, A), torch.full((4, A, 6), 1e-4), torch.zeros(4, A, 6), 6)
    assert h.shape == (4, A, 7)
    assert torch.isfinite(h).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -k "leakage or decouple or four_layers" -v`
Expected: FAIL — Task 2 leaves the layers independent, so `test_leakage_moves_water_between_layers_and_conserves_it` sees no head in the lower layer.

- [ ] **Step 3: Write minimal implementation**

Replace the `# Vertical leakage is added in Task 3` comment inside `_matvec` with:

```python
            if self.n_layers > 1:
                # Leakance L (1/day) between layer k and k+1, acting on the head difference.
                # Written as an explicit accumulation so it reads the same for any n_layers.
                Lk = torch.exp(self.log_L) * self.area          # (L-1, A)
                inter = Lk * (h[:-1] - h[1:])                   # downward-positive flux
                lay = torch.zeros_like(out)
                lay[:-1] = lay[:-1] + inter
                lay[1:] = lay[1:] - inter
                out = out + lay
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/flow.py tests/test_twin_flow.py
git commit -m "feat(twin): vertical leakage couples the four layers"
```

---

### Task 4: Pumping from the electricity census

**Files:**
- Create: `hydrophysics/twin/pumping.py`
- Test: `tests/test_twin_pumping.py`

**Interfaces:**
- Consumes: `FanGrid` (Task 1); `AMP_V2/data/pump_kwh_all.parquet` (columns `datetime, electricity_kwh, pump`); the pump census parquet with `sid, TWD97_X, TWD97_Y, PUMP_HP, PURPOSE`.
- Produces:
  - `aggregate_pumps(pumps: pd.DataFrame, kwh: pd.DataFrame, grid: FanGrid, t0, t1) -> tuple[np.ndarray, pd.DatetimeIndex]` — monthly kWh per active cell, shape `(A, T)`.
  - `energy_to_volume(E: Tensor, lift: Tensor, log_eta: Tensor) -> Tensor` — `Q = η·E / (ρ g · lift)` in m³/day.

**The physics that makes this work.** `E` is measured; `lift` is `ground elevation − simulated head`, so it comes from the model's own state. Only `η` (wire-to-water efficiency) is unknown, and it is one parameter per purpose class rather than a free field. Lift must be floored at a few metres so a shallow-head cell cannot produce an infinite volume.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twin_pumping.py
import numpy as np
import pandas as pd
import pytest
import torch

from hydrophysics.twin.grid import FanGrid
from hydrophysics.twin.pumping import aggregate_pumps, energy_to_volume


def _grid():
    return FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0,
                   mask=np.ones((4, 4), dtype=bool))


def test_energy_to_volume_follows_the_hydraulic_relation():
    """Q = eta * E / (rho g h). Doubling lift halves volume; doubling energy doubles it."""
    E = torch.tensor([[1000.0, 1000.0]])          # kWh in the period
    lift = torch.tensor([[10.0, 20.0]])
    q = energy_to_volume(E, lift, torch.zeros(1))
    assert float(q[0, 1]) == pytest.approx(float(q[0, 0]) / 2.0, rel=1e-6)

    q2 = energy_to_volume(E * 2, lift, torch.zeros(1))
    assert float(q2[0, 0]) == pytest.approx(2.0 * float(q[0, 0]), rel=1e-6)


def test_energy_to_volume_is_bounded_as_lift_goes_to_zero():
    E = torch.tensor([[1000.0]])
    q_small = energy_to_volume(E, torch.tensor([[1e-6]]), torch.zeros(1))
    q_ref = energy_to_volume(E, torch.tensor([[2.0]]), torch.zeros(1))
    assert torch.isfinite(q_small).all()
    assert float(q_small[0, 0]) <= 2.0 * float(q_ref[0, 0])


def test_aggregate_pumps_sums_into_the_right_cells():
    g = _grid()
    pumps = pd.DataFrame({"sid": ["a", "b", "c"],
                          "TWD97_X": [500.0, 1500.0, 500.0],
                          "TWD97_Y": [500.0, 500.0, 500.0],
                          "PUMP_HP": [5.0, 5.0, 5.0],
                          "PURPOSE": ["irrigation"] * 3})
    months = pd.date_range("2015-01-01", periods=2, freq="MS")
    kwh = pd.DataFrame({"pump": ["a", "a", "b", "b", "c", "c"],
                        "datetime": list(months) * 3,
                        "electricity_kwh": [10.0, 20.0, 100.0, 200.0, 1.0, 2.0]})
    E, dates = aggregate_pumps(pumps, kwh, g, "2015-01-01", "2015-03-01")
    assert E.shape == (g.n_active, 2)
    i_a = g.active_index(500.0, 500.0)
    i_b = g.active_index(1500.0, 500.0)
    assert E[i_a, 0] == pytest.approx(11.0)        # pumps a and c share a cell
    assert E[i_b, 1] == pytest.approx(200.0)
    assert E.sum() == pytest.approx(333.0)


def test_pumps_outside_the_grid_are_dropped_not_snapped():
    g = _grid()
    pumps = pd.DataFrame({"sid": ["far"], "TWD97_X": [999999.0], "TWD97_Y": [999999.0],
                          "PUMP_HP": [5.0], "PURPOSE": ["irrigation"]})
    kwh = pd.DataFrame({"pump": ["far"], "datetime": [pd.Timestamp("2015-01-01")],
                        "electricity_kwh": [50.0]})
    E, _ = aggregate_pumps(pumps, kwh, g, "2015-01-01", "2015-02-01")
    assert E.sum() == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_pumping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.pumping'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/pumping.py
"""Registered-pump electricity census -> per-cell monthly abstraction.

Groundwater abstraction on this fan is unmetered, but electricity is not: the wisenvr
`etc-tpc-etc1mon-obs` dataset carries monthly kWh for 116,769 registered pumps, each
georeferenced and labelled by purpose. Energy converts to volume through the pump's own
hydraulics,

    Q = eta * E / (rho * g * lift),        lift = ground elevation - head,

so the only unknown is the wire-to-water efficiency ``eta`` -- one parameter per purpose
class, not a free spatiotemporal field. Because ``lift`` comes from the model's simulated
head, falling heads make the same electricity deliver less water, which is a real
energy-water feedback and not a modelling artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

RHO_G = 9800.0          # N/m3, rho * g for fresh water
J_PER_KWH = 3.6e6
MIN_LIFT_M = 2.0        # floor: a near-zero lift must not imply unbounded volume


def aggregate_pumps(pumps: pd.DataFrame, kwh: pd.DataFrame, grid,
                    t0: str, t1: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Monthly kWh summed into active grid cells -> ``((A, T) array, dates)``.

    Pumps whose coordinates fall outside the grid or the fan mask are dropped, never
    snapped to the nearest cell.
    """
    dates = pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq="MS", inclusive="left")
    cell = {}
    for _, r in pumps.iterrows():
        try:
            i = grid.active_index(float(r["TWD97_X"]), float(r["TWD97_Y"]))
        except (TypeError, ValueError):
            i = None
        if i is not None:
            cell[str(r["sid"])] = i

    k = kwh.copy()
    k["datetime"] = pd.to_datetime(k["datetime"]).dt.to_period("M").dt.to_timestamp()
    k = k[(k["datetime"] >= dates[0]) & (k["datetime"] <= dates[-1])]
    k["cell"] = k["pump"].astype(str).map(cell)
    k = k.dropna(subset=["cell"])

    E = np.zeros((grid.n_active, len(dates)), dtype="float64")
    if k.empty:
        return E, dates
    tpos = {d: j for j, d in enumerate(dates)}
    k["tcol"] = k["datetime"].map(tpos)
    np.add.at(E, (k["cell"].astype(int).to_numpy(), k["tcol"].astype(int).to_numpy()),
              k["electricity_kwh"].to_numpy(dtype="float64"))
    return E, dates


def energy_to_volume(E: torch.Tensor, lift: torch.Tensor,
                     log_eta: torch.Tensor) -> torch.Tensor:
    """Convert period energy (kWh) and lift (m) to pumped volume (m3) for the period."""
    eta = torch.exp(log_eta)
    safe_lift = torch.clamp(lift, min=MIN_LIFT_M)
    return eta * E * J_PER_KWH / (RHO_G * safe_lift)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_pumping.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/pumping.py tests/test_twin_pumping.py
git commit -m "feat(twin): electricity census -> per-cell monthly abstraction"
```

---

### Task 5: Stage-3 calibration and gate

**Files:**
- Create: `hydrophysics/twin/calibrate_flow.py`
- Test: `tests/test_twin_flow.py`

**Interfaces:**
- Consumes: `FanGrid`, `FlowModel`, `aggregate_pumps`/`energy_to_volume`, `twin.heads.build_head_field`, `subsidence.idw_interp`.
- Produces:
  - `fit_flow(model, obs_h, obs_idx, obs_layer, recharge, E, ground_elev, epochs, lr) -> dict`
  - `loso_wells(...) -> dict` — leave-one-well-out over the 147 head records, pooled R² on held-out wells.
  - `main(argv=None)` — CLI writing `results/twin/stage3_flow.csv`.

**The gate.** Stage 3 passes if the calibrated flow model predicts **held-out wells** better than `idw_interp` from the remaining wells — the interpolation every result so far has relied on. Both are scored on the identical held-out cells. A physics model that cannot beat inverse-distance weighting has not earned its complexity.

**Parameter bounds (Global Constraints).** After each optimiser step, clamp inside `torch.no_grad()`: `log_T` to `log(1)…log(1e5)` m²/day, `log_S` to `log(1e-6)…log(0.3)`, `log_L` to `log(1e-8)…log(1e-1)` 1/day, `log_eta` to `log(0.05)…log(0.9)`. Report which bounds bind and at how many cells.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_flow.py
from hydrophysics.twin.calibrate_flow import fit_flow, loso_wells


def _synthetic_case(n=13, steps=12, seed=0):
    """Heads generated BY the model, so calibration is checkable against a known truth."""
    g = _uniform_grid(n=n, dx=1000.0)
    A = g.n_active
    truth = FlowModel(g, n_layers=2, dt_days=30.0)
    with torch.no_grad():
        truth.log_T.fill_(float(np.log(800.0)))
        truth.log_S.fill_(float(np.log(5e-4)))
        truth.log_L.fill_(float(np.log(1e-4)))
    rng = torch.Generator().manual_seed(seed)
    rech = torch.rand(2, A, steps, generator=rng) * 1e-3
    rech[1] = 0.0
    h = truth(torch.zeros(2, A), rech, torch.zeros(2, A, steps), steps).detach()
    obs_idx = torch.arange(0, A, max(A // 20, 1))
    obs_layer = torch.zeros_like(obs_idx)
    return g, h, rech, obs_idx, obs_layer


def test_fit_flow_recovers_synthetic_heads():
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   E=None, ground_elev=None, epochs=300, lr=0.1)
    assert out["r2"] > 0.9


def test_loso_wells_returns_a_finite_pooled_number():
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    out = loso_wells(g, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                     n_layers=2, epochs=120, lr=0.1)
    assert np.isfinite(out["r2"])
    assert np.isfinite(out["r2_idw"])
    assert out["n_wells"] == len(obs_idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -k "fit_flow or loso_wells" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.calibrate_flow'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/calibrate_flow.py
"""Stage 3: calibrate the four-layer flow model and run the leave-one-well-out gate.

The gate is deliberately unkind: the physics model must beat `subsidence.idw_interp`, the
inverse-distance interpolation every result so far has relied on, scored on the identical
held-out cells. A flow model that cannot beat IDW has not earned its complexity.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from ..subsidence import idw_interp
from .flow import FlowModel
from .grid import build_grid

# Physically defensible bounds. The loss is flat in several of these directions, and Plan A
# showed unbounded parameters run away with the epoch budget while the loss does not move.
BOUNDS = {
    "log_T": (math.log(1.0), math.log(1e5)),        # m2/day
    "log_S": (math.log(1e-6), math.log(0.3)),       # -
    "log_L": (math.log(1e-8), math.log(1e-1)),      # 1/day
}


def _clamp(model: FlowModel) -> dict[str, int]:
    """Clamp parameters into BOUNDS; return how many entries sit on a bound."""
    hits = {}
    with torch.no_grad():
        for name, (lo, hi) in BOUNDS.items():
            par = getattr(model, name, None)
            if par is None:
                continue
            par.clamp_(min=lo, max=hi)
            hits[name] = int(((par <= lo + 1e-9) | (par >= hi - 1e-9)).sum())
    return hits


def _r2(pred: np.ndarray, obs: np.ndarray) -> float:
    ss_res = float(((obs - pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def fit_flow(model: FlowModel, obs_h: torch.Tensor, obs_idx: torch.Tensor,
             obs_layer: torch.Tensor, recharge: torch.Tensor,
             E=None, ground_elev=None, epochs: int = 1500, lr: float = 0.1,
             init_scatter: float = 0.0, seed: int | None = None) -> dict:
    """Fit log-parameters to observed head series by masked MSE.

    ``obs_h`` is ``(W, T)``; ``obs_idx``/``obs_layer`` locate each well in the active-cell
    vector and the layer stack.
    """
    if init_scatter > 0.0:
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        with torch.no_grad():
            for name in BOUNDS:
                par = getattr(model, name, None)
                if par is not None:
                    par.add_(torch.randn(par.shape, generator=g).to(par.device) * init_scatter)

    n_steps = recharge.shape[-1]
    A = model.grid.n_active
    pumping = torch.zeros(model.n_layers, A, n_steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    h0 = torch.zeros(model.n_layers, A)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        opt.zero_grad()
        h = model(h0, recharge, pumping, n_steps)
        pred = h[obs_layer, obs_idx, 1:]
        loss = ((pred - obs_h) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hits = _clamp(model)
    with torch.no_grad():
        pred = model(h0, recharge, pumping, n_steps)[obs_layer, obs_idx, 1:]
    return {"loss": float(loss.detach()), "epochs": epochs, "bounds_hit": hits,
            "r2": _r2(pred.numpy(), obs_h.numpy())}


def loso_wells(grid, obs_h: torch.Tensor, obs_idx: torch.Tensor,
               obs_layer: torch.Tensor, recharge: torch.Tensor, n_layers: int = 4,
               epochs: int = 1500, lr: float = 0.1) -> dict:
    """Leave-one-well-out, against an IDW baseline on the identical held-out cells."""
    W = obs_h.shape[0]
    xy = grid.centroids()
    preds, idws, targets = [], [], []
    for held in range(W):
        keep = [i for i in range(W) if i != held]
        m = FlowModel(grid, n_layers=n_layers, dt_days=30.0)
        fit_flow(m, obs_h[keep], obs_idx[keep], obs_layer[keep], recharge,
                 epochs=epochs, lr=lr)
        with torch.no_grad():
            A = grid.n_active
            h = m(torch.zeros(n_layers, A), recharge,
                  torch.zeros(n_layers, A, recharge.shape[-1]), recharge.shape[-1])
            p = h[obs_layer[held], obs_idx[held], 1:].numpy()
        src = xy[obs_idx[keep].numpy()]
        tgt = xy[obs_idx[held].numpy()][None, :]
        idw = idw_interp(tgt, src, obs_h[keep].numpy())[0]
        preds.append(p)
        idws.append(idw)
        targets.append(obs_h[held].numpy())
    pred = np.concatenate(preds)
    obs = np.concatenate(targets)
    return {"r2": _r2(pred, obs), "r2_idw": _r2(np.concatenate(idws), obs),
            "n_wells": W}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage-3 flow calibration and gate")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--wells-dir", default="AMP_V2/data/wells")
    ap.add_argument("--stations", default="AMP_V2/data/fan_stations.parquet")
    ap.add_argument("--polygon",
                    default="chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/"
                            "Zhuoshui Alluvial Fan.json")
    ap.add_argument("--out", default="results/twin")
    args = ap.parse_args(argv)

    from .heads import build_head_field

    grid = build_grid(args.polygon, dx=args.dx)
    stn = pd.read_parquet(args.stations)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(args.wells_dir, stn)

    idx, lay, series = [], [], []
    for w in range(len(hf)):
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        s = hf.heads[w]
        if not np.isfinite(s).all():
            s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
        idx.append(i)
        lay.append(max(int(hf.layers[w]) - 1, 0))
        series.append(s)
    obs_h = torch.tensor(np.stack(series), dtype=torch.float32)
    obs_idx = torch.tensor(idx, dtype=torch.long)
    obs_layer = torch.tensor(lay, dtype=torch.long)
    n_steps = obs_h.shape[1]
    recharge = torch.zeros(4, grid.n_active, n_steps)

    m = FlowModel(grid, n_layers=4, dt_days=30.0)
    ins = fit_flow(m, obs_h, obs_idx, obs_layer, recharge,
                   epochs=args.epochs, lr=args.lr)
    gate = loso_wells(grid, obs_h, obs_idx, obs_layer, recharge,
                      epochs=max(args.epochs, 1), lr=args.lr)
    print(f"wells={gate['n_wells']} cells={grid.n_active} dx={args.dx:.0f}m")
    print(f"  in-sample R2={ins['r2']:+.3f}  bounds_hit={ins['bounds_hit']}")
    print(f"  LOSO R2={gate['r2']:+.3f}   IDW baseline R2={gate['r2_idw']:+.3f}")
    print(f"GATE: {'PASS' if gate['r2'] > gate['r2_idw'] else 'FAIL'}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage3_flow.csv")
    pd.DataFrame([{"n_wells": gate["n_wells"], "n_cells": grid.n_active, "dx": args.dx,
                   "loss": ins["loss"], "r2_insample": ins["r2"],
                   "r2_loso": gate["r2"], "r2_idw": gate["r2_idw"],
                   "bounds_hit": str(ins["bounds_hit"])}]).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

**Note on recharge.** `main` above passes zero recharge, which makes the first run a pure
pumping/storage calibration. Wiring real forcing (26 rain gauges in `rf_timeseries.csv`
minus the cached ET at `results/et/openmeteo_et0_2012_2022.npz`) is the first thing to try
if the gate fails — see the gate table.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the real Stage-3 gate**

```bash
export HYDROMIND_GW_DATA="$(pwd)/chou-shui-data/chou-shui-data/data"
.venv/bin/python -m hydrophysics.twin.calibrate_flow --epochs 1500
```

Expected: `r2_loso` and `r2_idw` printed side by side. **The gate passes if `r2_loso > r2_idw`.** Record both numbers, and report which parameter bounds bind.

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/twin/calibrate_flow.py tests/test_twin_flow.py
git commit -m "feat(twin): Stage-3 flow calibration and leave-one-well-out gate

LOSO R2 = <value> vs IDW baseline <value> on identical held-out cells."
```

---

### Task 6: Robustness — ensemble and grid convergence

**Files:**
- Modify: `hydrophysics/twin/calibrate_flow.py`
- Test: `tests/test_twin_flow.py`

**Interfaces:**
- Consumes: `fit_flow`, `loso_wells` (Task 5).
- Produces: `--ensemble N --init-scatter S` (same contract as `calibrate_mlcw`) and `--dx` so the gate can be run at 1000 m and 500 m.

**Why both.** Plan A's Stage-2 number moved from −0.841 to +0.324 on a single-line model fix, and a fixed initialisation hid that the landscape was flat in several directions. An ensemble is the cheapest guard against reporting one lucky start. Grid convergence is the equivalent guard for the discretisation: if 1 km and 500 m disagree materially, the 1 km number is a discretisation artefact.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_flow.py
def test_init_scatter_changes_the_starting_point_but_not_the_recovered_fit():
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    outs = []
    for seed in range(2):
        m = FlowModel(g, n_layers=2, dt_days=30.0)
        outs.append(fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                             E=None, ground_elev=None, epochs=300, lr=0.1,
                             init_scatter=0.5, seed=seed)["r2"])
    assert all(o > 0.85 for o in outs)
    assert abs(outs[0] - outs[1]) < 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -k init_scatter -v`
Expected: FAIL with `TypeError: fit_flow() got an unexpected keyword argument 'init_scatter'`

- [ ] **Step 3: Write minimal implementation**

`fit_flow` already accepts `init_scatter` and `seed` (Task 5). Add the ensemble driver and
its CLI flags to `calibrate_flow.py`:

```python
def ensemble(grid, obs_h, obs_idx, obs_layer, recharge, n: int = 5,
             init_scatter: float = 0.5, **kw) -> dict:
    """Repeat the leave-one-well-out gate from scattered starts.

    FlowModel's init is a fixed constant, so a single fit says nothing about whether the
    gate number is representative of a landscape that may be flat in several directions.
    """
    vals = []
    for seed in range(n):
        m = FlowModel(grid, n_layers=4, dt_days=30.0)
        fit_flow(m, obs_h, obs_idx, obs_layer, recharge,
                 init_scatter=init_scatter, seed=seed, **kw)
        g = loso_wells(grid, obs_h, obs_idx, obs_layer, recharge, **kw)
        vals.append(g["r2"])
        print(f"  seed {seed}: LOSO R2={g['r2']:+.4f}", flush=True)
    a = np.array(vals, dtype="float64")
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max()), "values": vals}
```

and in `main`, after the gate print:

```python
    if args.ensemble > 0:
        print(f"\n=== ensemble: {args.ensemble} runs, init scatter {args.init_scatter} ===")
        ens = ensemble(grid, obs_h, obs_idx, obs_layer, recharge, n=args.ensemble,
                       init_scatter=args.init_scatter, epochs=args.epochs, lr=args.lr)
        print(f"  mean {ens['mean']:+.4f}  sd {ens['sd']:+.4f}  "
              f"min {ens['min']:+.4f}  max {ens['max']:+.4f}")
        pd.DataFrame({"seed": range(len(ens["values"])), "r2_loso": ens["values"],
                      "r2_idw": gate["r2_idw"]}).to_csv(
            os.path.join(args.out, "stage3_ensemble.csv"), index=False)
```

with these flags added to the parser:

```python
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--init-scatter", type=float, default=0.5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_flow.py -v`
Expected: 10 passed

- [ ] **Step 5: Run the robustness checks**

```bash
export HYDROMIND_GW_DATA="$(pwd)/chou-shui-data/chou-shui-data/data"
.venv/bin/python -m hydrophysics.twin.calibrate_flow --epochs 1500 --ensemble 5
.venv/bin/python -m hydrophysics.twin.calibrate_flow --epochs 1500 --dx 500
```

Record the ensemble mean/sd and whether the 500 m gate agrees with 1 km.

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/twin/calibrate_flow.py tests/test_twin_flow.py
git commit -m "feat(twin): Stage-3 ensemble and grid-convergence checks"
```

---

### Task 7: Record the result and update the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the CSVs from Tasks 5–6.

- [ ] **Step 1: Add a Stage-3 Results subsection to spec §7**

Include: well count, cell count, grid resolution, `r2_insample`, `r2_loso`, `r2_idw`, the ensemble mean/sd, the 500 m convergence result, which parameter bounds bind and where, and the reproduce command. State the gate verdict plainly, pass or fail.

- [ ] **Step 2: Update the README's model table** with the Stage-3 row, labelling each number for what it is (in-sample vs leave-one-well-out) — the mislabel corrected during Plan A must not recur.

- [ ] **Step 3: Commit**

```bash
git add docs README.md
git commit -m "docs(twin): Stage-3 flow calibration results"
```

---

## Stage gate

| Gate | Where | Pass condition | If it fails |
|---|---|---|---|
| Stage 3 | Task 5 | `r2_loso > r2_idw` on identical held-out cells | The flow model does not beat inverse-distance weighting. Do **not** start Plan C. Investigate whether recharge forcing (26 rain gauges + cached ET) or pump-layer allocation is the limiter, and report that as the finding |

Record the gate numbers in the commit message and carry them into spec §7 before Plan C is written.

## Carried forward from Plan A — read before starting

1. **A gate comparison is only meaningful on identical arrays.** −0.28 vs −0.871 wasted a review cycle because one was in-sample.
2. **Bound anything the loss is flat in**, and report whether the bound binds.
3. **Never express a physical threshold against an absolute datum.** That bug disabled the inelastic term at half the sites and inverted the Stage-2 conclusion.
4. **Test fixtures need enough signal to fail a no-op.** Three Plan A fixtures passed against stub implementations.
5. **The head field matters more than the model.** Moving from 61 to 147 wells changed the Stage-2 baseline from −0.556 to +0.106. Data adequacy is a first-class hypothesis, not a footnote.
