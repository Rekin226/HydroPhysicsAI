# Spatial PINN Head Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a physics-informed neural network that learns a continuous groundwater head field `h(x, y, t)` over the Zhuoshui fan, scored in the existing simulation-mode harness three ways (in-sample, leave-one-well-out spatial, continuous map).

**Architecture:** A Fourier-feature MLP maps `(x, y, t)` → head, trained against the 61 wells as scattered points and regularized by the 2D depth-averaged groundwater-flow PDE `S·∂h/∂t = ∇·(T∇h) + α·R − d`, with `T(x,y)`, `α(x,y)`, `d(x,y)` learned as small spatial sub-networks. All derivatives come from autodiff (no mesh). The model implements the existing `GroundwaterModel` interface so `bench.py` and the benchmark table wire up unchanged.

**Tech Stack:** Python, NumPy, PyTorch (optional dep, same pattern as `ude.py`), pytest. Reuses `hydrophysics.data.GWData`, `hydrophysics.eval`, `hydrophysics.metrics`.

---

## Design notes (read once before starting)

- **Nondimensionalization.** Coordinates are normalized to `[0,1]` by `X=(x−xmin)/L`, `Y=(y−ymin)/L` with `L=max(x_range, y_range)` (preserves aspect ratio). Time is `τ = day_index / 365.25` (years). Head is standardized `H=(h−μ)/σ` using **training** observations only. The PDE constants are absorbed into the **learned** fields `S, T, α, d`, so the residual is written purely in normalized coordinates — it only needs to be self-consistent, not carry SI units. `T(x,y)` is therefore interpretable up to a scale.
- **Rainfall field.** Instead of depending on `rf_stations.csv` (absent from the synthetic sample), the rainfall field is interpolated by inverse-distance weighting (IDW) from the **per-well paired rainfall** already in `GWData.rainfall` at the well coordinates. This works identically on the real data and the synthetic sample and keeps everything inside `GWData`.
- **Boundary condition.** Coastal wells are themselves observation points, so the **data loss already pins head near the coast** to sea-influenced observed values. We add an explicit **L2 smoothness prior on `log T`** (the regularizer the spec calls for — 61 points is thin for a 2D field). An explicit sea-shapefile Dirichlet is a documented Phase-4 refinement to add only if the map edges misbehave.
- **No initial-value rollout.** Unlike the ODE models, a field model has no IVP — `simulate` just evaluates the field at each well's coordinate over every day. It never reads validation-period targets, so the `GroundwaterModel` fairness contract holds.

## File structure

- **Create** `hydrophysics/field_inputs.py` — pure-NumPy domain geometry: `Normalizer` (nondimensionalization), `well_coords_norm`, `RainfallField` (IDW). No torch.
- **Create** `hydrophysics/models/pinn_field.py` — torch: positional encoding, `pde_residual`, `SpatialPINN(GroundwaterModel)`, `leave_one_well_out_field`, CLI `main`.
- **Create** `tests/test_pinn_field.py` — residual analytic check, no-leakage, interface smoke, rainfall field.
- **Modify** `hydrophysics/train.py` — register `"pinn"` in `build_model`.
- **Modify** `hydrophysics/viz.py` — add `plot_head_field` for the continuous map.
- **Modify** `README.md`, `MODEL_CARD.md` — results + how-to (final task).

---

## Task 1: Nondimensionalization (`Normalizer`)

**Files:**
- Create: `hydrophysics/field_inputs.py`
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py
"""Tests for the spatial PINN head field. Torch tests skip when torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

from hydrophysics import Config, load_dataset
from hydrophysics.sample import write_sample


@pytest.fixture()
def data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=2)
    return load_dataset(Config(data_dir=d, baseline_results=d / "gw_fit_results.csv"))


def test_normalizer_roundtrips_coords_and_head(data):
    from hydrophysics.field_inputs import Normalizer

    norm = Normalizer.from_data(data)
    x = data.attrs["tm_x"].astype(float).to_numpy()
    y = data.attrs["tm_y"].astype(float).to_numpy()
    X, Y = norm.xy(x, y)
    # normalized coords land in [0, 1]
    assert X.min() >= -1e-6 and X.max() <= 1 + 1e-6
    assert Y.min() >= -1e-6 and Y.max() <= 1 + 1e-6
    # head standardize/unstandardize round-trips
    h = np.array([norm.h_mu - 2.0, norm.h_mu, norm.h_mu + 3.0])
    assert np.allclose(norm.h_from_norm(norm.h_to_norm(h)), h, atol=1e-5)
    # time in years
    assert np.isclose(norm.tau(np.array([365.25]))[0], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_normalizer_roundtrips_coords_and_head -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.field_inputs'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/field_inputs.py
"""Domain geometry for the spatial head field: nondimensionalization + rainfall field.

Pure NumPy (no torch) so it is importable in the foundation/test path. The PINN
(``models/pinn_field.py``) consumes these helpers. Constants are absorbed into the
PINN's learned fields, so normalization only needs to be self-consistent: coordinates
to [0,1], time to years, head standardized on training observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import GWData


@dataclass
class Normalizer:
    xmin: float
    ymin: float
    length: float        # shared scale for x and y (preserves aspect ratio)
    h_mu: float
    h_sd: float
    year: float = 365.25

    @classmethod
    def from_data(cls, data: GWData) -> "Normalizer":
        x = data.attrs["tm_x"].astype(float).fillna(0.0).to_numpy()
        y = data.attrs["tm_y"].astype(float).fillna(0.0).to_numpy()
        xr = float(x.max() - x.min())
        yr = float(y.max() - y.min())
        length = max(xr, yr, 1.0)
        h = data.target[:, data.train_mask]
        h = h[np.isfinite(h)]
        h_mu = float(h.mean()) if h.size else 0.0
        h_sd = float(h.std()) + 1e-6 if h.size else 1.0
        return cls(xmin=float(x.min()), ymin=float(y.min()), length=length,
                   h_mu=h_mu, h_sd=h_sd)

    def xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(x) - self.xmin) / self.length, (np.asarray(y) - self.ymin) / self.length

    def tau(self, day_index: np.ndarray) -> np.ndarray:
        return np.asarray(day_index, dtype=float) / self.year

    def h_to_norm(self, h: np.ndarray) -> np.ndarray:
        return (np.asarray(h) - self.h_mu) / self.h_sd

    def h_from_norm(self, hn: np.ndarray) -> np.ndarray:
        return np.asarray(hn) * self.h_sd + self.h_mu
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_normalizer_roundtrips_coords_and_head -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/field_inputs.py tests/test_pinn_field.py
git commit -m "feat(pinn): nondimensionalization Normalizer for the spatial head field"
```

---

## Task 2: Rainfall field (IDW)

**Files:**
- Modify: `hydrophysics/field_inputs.py`
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
def test_rainfall_field_idw(data):
    from hydrophysics.field_inputs import Normalizer, RainfallField, well_coords_norm

    norm = Normalizer.from_data(data)
    wc = well_coords_norm(data, norm)            # (W, 2)
    field = RainfallField(wc, np.nan_to_num(data.rainfall))

    # querying exactly at a well coordinate returns that well's rainfall that day
    day = int(np.flatnonzero(data.train_mask)[10])
    pts = wc[:1]                                  # first well
    val = field.at(pts, np.array([day]))
    assert np.isclose(val[0], np.nan_to_num(data.rainfall)[0, day], atol=1e-4)

    # interpolating between wells is finite and within the data range
    mid = wc.mean(axis=0, keepdims=True)
    v = field.at(mid, np.array([day]))
    assert np.isfinite(v).all()
    assert v[0] >= np.nan_to_num(data.rainfall)[:, day].min() - 1e-6
    assert v[0] <= np.nan_to_num(data.rainfall)[:, day].max() + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_rainfall_field_idw -v`
Expected: FAIL with `ImportError: cannot import name 'RainfallField'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/field_inputs.py  (append)

def well_coords_norm(data: GWData, norm: Normalizer) -> np.ndarray:
    """Normalized (W, 2) well coordinates in [0, 1]^2."""
    x = data.attrs["tm_x"].astype(float).fillna(0.0).to_numpy()
    y = data.attrs["tm_y"].astype(float).fillna(0.0).to_numpy()
    X, Y = norm.xy(x, y)
    return np.stack([X, Y], axis=-1).astype("float64")


class RainfallField:
    """Continuous rainfall R(x, y, t) by inverse-distance weighting of the per-well
    paired rainfall at the (normalized) well coordinates.

    Exact at a well coordinate (returns that well's series); a smooth IDW blend
    elsewhere. Power 2, small epsilon to avoid singularities. Pure NumPy.
    """

    def __init__(self, well_xy: np.ndarray, rainfall: np.ndarray,
                 power: float = 2.0, eps: float = 1e-6):
        self.well_xy = np.asarray(well_xy, dtype="float64")   # (W, 2)
        self.rainfall = np.asarray(rainfall, dtype="float64")  # (W, T)
        self.power = power
        self.eps = eps

    def at(self, points_xy: np.ndarray, day_index: np.ndarray) -> np.ndarray:
        """Rainfall at N query points on N day indices -> (N,)."""
        pts = np.asarray(points_xy, dtype="float64")          # (N, 2)
        days = np.asarray(day_index, dtype=int)               # (N,)
        d2 = ((pts[:, None, :] - self.well_xy[None, :, :]) ** 2).sum(-1)  # (N, W)
        w = 1.0 / (d2 ** (self.power / 2.0) + self.eps)        # (N, W)
        w /= w.sum(axis=1, keepdims=True)
        rain_nw = self.rainfall[:, days].T                    # (N, W)
        return (w * rain_nw).sum(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_rainfall_field_idw -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/field_inputs.py tests/test_pinn_field.py
git commit -m "feat(pinn): IDW rainfall field over well coordinates"
```

---

## Task 3: Network primitives (positional encoding + MLP)

**Files:**
- Create: `hydrophysics/models/pinn_field.py`
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
torch = pytest.importorskip("torch")  # torch tests below skip if torch is absent


def test_positional_encoding_shape_and_grad():
    from hydrophysics.models.pinn_field import positional_encoding

    coords = torch.rand(5, 3, requires_grad=True)
    enc = positional_encoding(coords, n_bands=4)
    # original dims + sin/cos for each dim and band
    assert enc.shape == (5, 3 + 3 * 2 * 4)
    enc.sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_positional_encoding_shape_and_grad -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.models.pinn_field'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/models/pinn_field.py
"""SpatialPINN: a physics-informed continuous head field h(x, y, t) over the fan.

The lumped UDE models one ODE per well (0D in space). This learns a single continuous
field across the whole alluvial fan, trained against the wells as scattered observation
points and regularized by the 2D depth-averaged groundwater-flow PDE

    S * dh/dt = div(T grad h) + alpha(x,y) * R(x,y,t) - d(x,y)

All derivatives come from autodiff (no mesh). T, alpha, d are small spatial sub-networks;
S is a learned positive scalar. The model satisfies the GroundwaterModel interface, so
bench.py and the benchmark table wire up unchanged. See
docs/superpowers/specs/2026-06-16-spatial-pinn-head-field-design.md.
"""

from __future__ import annotations

import numpy as np

from ..data import GWData
from ..field_inputs import Normalizer, RainfallField, well_coords_norm
from .base import GroundwaterModel
from .gru import _default_device

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def positional_encoding(coords, n_bands: int):
    """NeRF-style encoding: [coords, sin(2^k pi c), cos(2^k pi c) for k in 0..n_bands-1].

    coords: (N, D) tensor. Returns (N, D + D*2*n_bands). Differentiable.
    """
    feats = [coords]
    for k in range(n_bands):
        freq = (2.0 ** k) * np.pi
        feats.append(torch.sin(freq * coords))
        feats.append(torch.cos(freq * coords))
    return torch.cat(feats, dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_positional_encoding_shape_and_grad -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/models/pinn_field.py tests/test_pinn_field.py
git commit -m "feat(pinn): NeRF-style positional encoding for the field network"
```

---

## Task 4: The PDE residual (the physics — analytic verification)

This is the load-bearing physics test: the differential operator is factored into a
standalone `pde_residual` that takes any differentiable head callable, so it can be
checked against a closed-form solution independent of the trained network.

**Files:**
- Modify: `hydrophysics/models/pinn_field.py`
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
def test_pde_residual_matches_analytic_solution():
    """For h = sin(pi X) sin(pi Y) exp(-lam tau) with T=1, S=1, lam=2pi^2, the
    diffusion residual dh/dt - (h_xx + h_yy) is identically zero."""
    from hydrophysics.models.pinn_field import pde_residual

    lam = 2.0 * (np.pi ** 2)

    def h_fn(X, Y, tau):
        return torch.sin(np.pi * X) * torch.sin(np.pi * Y) * torch.exp(-lam * tau)

    n = 64
    X = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    Y = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    tau = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    rain = torch.zeros(n, 1, dtype=torch.float64)

    res = pde_residual(
        h_fn, X, Y, tau, rain,
        T_fn=lambda X, Y: torch.ones_like(X),
        alpha_fn=lambda X, Y: torch.zeros_like(X),
        d_fn=lambda X, Y: torch.zeros_like(X),
        S=1.0,
    )
    assert res.shape == (n, 1)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_pde_residual_matches_analytic_solution -v`
Expected: FAIL with `ImportError: cannot import name 'pde_residual'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/models/pinn_field.py  (append)

def _grad(outputs, inputs):
    """d(outputs)/d(inputs), summed over the batch, with graph kept for higher orders."""
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]


def pde_residual(h_fn, X, Y, tau, rain, *, T_fn, alpha_fn, d_fn, S):
    """2D depth-averaged groundwater-flow residual in normalized coordinates.

        res = S * dH/dtau - div(T grad H) - alpha * R + d

    h_fn(X, Y, tau) -> (N, 1) head; T_fn/alpha_fn/d_fn(X, Y) -> (N, 1) spatial fields;
    S: scalar or (N,1). X, Y, tau must be leaf tensors with requires_grad=True. rain:
    (N, 1). Returns (N, 1). Used by both training and the analytic test.
    """
    h = h_fn(X, Y, tau)
    hX = _grad(h, X)
    hY = _grad(h, Y)
    ht = _grad(h, tau)
    T = T_fn(X, Y)
    fx = T * hX
    fy = T * hY
    div = _grad(fx, X) + _grad(fy, Y)
    return S * ht - div - alpha_fn(X, Y) * rain + d_fn(X, Y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_pde_residual_matches_analytic_solution -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/models/pinn_field.py tests/test_pinn_field.py
git commit -m "feat(pinn): autodiff PDE residual, verified against analytic diffusion"
```

---

## Task 5: `SpatialPINN` model (fit / simulate + no-leakage)

**Files:**
- Modify: `hydrophysics/models/pinn_field.py`
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
def test_spatial_pinn_fit_simulate_shapes_and_benchmark(data):
    from hydrophysics import benchmark_table
    from hydrophysics.models.pinn_field import SpatialPINN

    model = SpatialPINN(device="cpu", epochs=3, n_collocation=128, seed=0)
    model.fit(data)
    pred = model.simulate(data)
    assert pred.shape == data.target.shape
    assert np.isfinite(pred).all()
    table = benchmark_table(data, {model.name: pred}, period="val")
    assert model.name in table.index


def test_spatial_pinn_lowo_no_leakage(data):
    """A held-out well contributes no observation rows to the data loss."""
    from hydrophysics.models.pinn_field import SpatialPINN

    held = np.zeros(data.n_wells, dtype=bool)
    held[0] = True
    model = SpatialPINN(device="cpu", epochs=1, n_collocation=32, seed=0)
    rows = model._obs_rows(data, train_wells=~held)
    # well index 0 must never appear among the observation-row well indices
    assert (rows["well"] != 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_spatial_pinn_fit_simulate_shapes_and_benchmark tests/test_pinn_field.py::test_spatial_pinn_lowo_no_leakage -v`
Expected: FAIL with `ImportError: cannot import name 'SpatialPINN'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/models/pinn_field.py  (append)

def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("SpatialPINN requires torch. Install with: pip install 'torch>=2.0'")


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpatialPINN(GroundwaterModel):
    """Physics-informed continuous head field h(x, y, t). See module docstring."""

    name = "pinn"

    def __init__(self, hidden: int = 64, n_bands: int = 6, epochs: int = 1500,
                 lr: float = 1e-3, n_collocation: int = 2048, physics_weight: float = 0.1,
                 smooth_weight: float = 1e-3, depth: int = 4,
                 device: str | None = None, seed: int = 0):
        _require_torch()
        self.hidden, self.n_bands, self.epochs, self.lr = hidden, n_bands, epochs, lr
        self.n_collocation = n_collocation
        self.physics_weight, self.smooth_weight = physics_weight, smooth_weight
        self.depth, self.seed = depth, seed
        self.device = device or _default_device()
        self.norm: Normalizer | None = None
        self.h_net: nn.Module | None = None
        self.field_net: nn.Module | None = None
        self.log_S = None
        self._wc: np.ndarray | None = None

    # --- helpers -----------------------------------------------------------
    def _obs_rows(self, data: GWData, train_wells: np.ndarray | None):
        """Observation rows entering the data loss: training days, finite obs, and
        (for LOWO) only wells in ``train_wells``. Returns a dict of int/float arrays."""
        keep_well = (np.ones(data.n_wells, dtype=bool) if train_wells is None
                     else np.asarray(train_wells, dtype=bool))
        day_idx = np.flatnonzero(data.train_mask)
        wi, ti, hv = [], [], []
        for i in range(data.n_wells):
            if not keep_well[i]:
                continue
            h = data.target[i, day_idx]
            fin = np.isfinite(h)
            wi.append(np.full(int(fin.sum()), i))
            ti.append(day_idx[fin])
            hv.append(h[fin])
        return {"well": np.concatenate(wi), "day": np.concatenate(ti),
                "h": np.concatenate(hv)}

    def _build(self):
        enc_dim3 = 3 + 3 * 2 * self.n_bands
        enc_dim2 = 2 + 2 * 2 * self.n_bands
        self.h_net = _MLP(enc_dim3, self.hidden, 1, self.depth).to(self.device)
        # field net outputs [log_T, alpha_raw, d_raw]
        self.field_net = _MLP(enc_dim2, self.hidden, 3, depth=2).to(self.device)
        self.log_S = nn.Parameter(torch.zeros(1, device=self.device))

    def _h_forward(self, X, Y, tau):
        enc = positional_encoding(torch.cat([X, Y, tau], dim=-1), self.n_bands)
        return self.h_net(enc)

    def _fields(self, X, Y):
        enc = positional_encoding(torch.cat([X, Y], dim=-1), self.n_bands)
        out = self.field_net(enc)
        log_T = out[:, 0:1]
        T = torch.nn.functional.softplus(log_T) + 1e-3
        alpha = torch.nn.functional.softplus(out[:, 1:2])
        d = out[:, 2:3]
        return T, alpha, d, log_T

    # --- interface ---------------------------------------------------------
    def fit(self, data: GWData, train_wells: np.ndarray | None = None) -> "SpatialPINN":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.norm = Normalizer.from_data(data)
        self._wc = well_coords_norm(data, self.norm)            # (W, 2)
        rain_field = RainfallField(self._wc, np.nan_to_num(data.rainfall))
        rain_std = float(np.nan_to_num(data.rainfall)[:, data.train_mask].std()) + 1e-6
        self._build()

        rows = self._obs_rows(data, train_wells)
        obs_X = torch.tensor(self._wc[rows["well"], 0:1], dtype=torch.float32, device=self.device)
        obs_Y = torch.tensor(self._wc[rows["well"], 1:2], dtype=torch.float32, device=self.device)
        obs_tau = torch.tensor(self.norm.tau(rows["day"])[:, None], dtype=torch.float32, device=self.device)
        obs_H = torch.tensor(self.norm.h_to_norm(rows["h"])[:, None], dtype=torch.float32, device=self.device)

        train_days = np.flatnonzero(data.train_mask)
        opt = torch.optim.Adam(
            list(self.h_net.parameters()) + list(self.field_net.parameters()) + [self.log_S],
            lr=self.lr,
        )
        for _ in range(self.epochs):
            opt.zero_grad()
            # data loss (no autograd on coords needed)
            h_pred = self._h_forward(obs_X, obs_Y, obs_tau)
            data_loss = torch.mean((h_pred - obs_H) ** 2)

            # physics loss on random collocation points
            n = self.n_collocation
            cx = torch.rand(n, 1, device=self.device, requires_grad=True)
            cy = torch.rand(n, 1, device=self.device, requires_grad=True)
            cdays = np.random.choice(train_days, size=n)
            ctau = torch.tensor(self.norm.tau(cdays)[:, None], dtype=torch.float32,
                                device=self.device).requires_grad_(True)
            crain = torch.tensor(
                rain_field.at(np.concatenate([cx.detach().cpu().numpy(),
                                              cy.detach().cpu().numpy()], axis=1), cdays)[:, None] / rain_std,
                dtype=torch.float32, device=self.device,
            )
            S = torch.nn.functional.softplus(self.log_S)
            res = pde_residual(
                self._h_forward, cx, cy, ctau, crain,
                T_fn=lambda X, Y: self._fields(X, Y)[0],
                alpha_fn=lambda X, Y: self._fields(X, Y)[1],
                d_fn=lambda X, Y: self._fields(X, Y)[2],
                S=S,
            )
            phys_loss = torch.mean(res ** 2)

            # L2 smoothness prior on log T (keeps the field near baseline; 61 pts is thin)
            _, _, _, log_T = self._fields(cx.detach(), cy.detach())
            smooth = torch.mean(log_T ** 2)

            loss = data_loss + self.physics_weight * phys_loss + self.smooth_weight * smooth
            loss.backward()
            opt.step()
        return self

    def simulate(self, data: GWData) -> np.ndarray:
        if self.h_net is None or self.norm is None:
            raise RuntimeError("call fit() before simulate()")
        W, T = data.target.shape
        days = np.arange(T)
        wc = self._wc
        Xs, Ys, taus = [], [], []
        for i in range(W):
            Xs.append(np.full(T, wc[i, 0]))
            Ys.append(np.full(T, wc[i, 1]))
            taus.append(self.norm.tau(days))
        X = torch.tensor(np.concatenate(Xs)[:, None], dtype=torch.float32, device=self.device)
        Y = torch.tensor(np.concatenate(Ys)[:, None], dtype=torch.float32, device=self.device)
        tau = torch.tensor(np.concatenate(taus)[:, None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Hn = self._h_forward(X, Y, tau).cpu().numpy().reshape(W, T)
        return self.norm.h_from_norm(Hn)

    def head_field(self, points_xy_phys: np.ndarray, day_index: int) -> np.ndarray:
        """Head at arbitrary physical (x, y) points on one day -> (M,). For maps."""
        if self.h_net is None or self.norm is None:
            raise RuntimeError("call fit() before head_field()")
        x = np.asarray(points_xy_phys)[:, 0]
        y = np.asarray(points_xy_phys)[:, 1]
        X, Y = self.norm.xy(x, y)
        tau = self.norm.tau(np.full(len(x), day_index))
        Xt = torch.tensor(X[:, None], dtype=torch.float32, device=self.device)
        Yt = torch.tensor(Y[:, None], dtype=torch.float32, device=self.device)
        Tt = torch.tensor(tau[:, None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Hn = self._h_forward(Xt, Yt, Tt).cpu().numpy().ravel()
        return self.norm.h_from_norm(Hn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pinn_field.py::test_spatial_pinn_fit_simulate_shapes_and_benchmark tests/test_pinn_field.py::test_spatial_pinn_lowo_no_leakage -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/models/pinn_field.py tests/test_pinn_field.py
git commit -m "feat(pinn): SpatialPINN fit/simulate over the GroundwaterModel interface"
```

---

## Task 6: Register `pinn` in `build_model`

**Files:**
- Modify: `hydrophysics/train.py` (the `build_model` function shown above)
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
def test_build_model_registers_pinn():
    from hydrophysics.train import build_model
    from hydrophysics.models.pinn_field import SpatialPINN

    model = build_model("pinn", device="cpu", epochs=3)
    assert isinstance(model, SpatialPINN)
    assert model.name == "pinn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_build_model_registers_pinn -v`
Expected: FAIL with `ValueError: unknown model 'pinn' (choose: gru, ude, ude_nemo)`

- [ ] **Step 3: Write minimal implementation**

In `hydrophysics/train.py`, inside `build_model`, add this branch before the final `raise`:

```python
    if name == "pinn":
        from .models.pinn_field import SpatialPINN
        return SpatialPINN(device=device, epochs=epochs)
```

And update the error message:

```python
    raise ValueError(f"unknown model '{name}' (choose: gru, ude, ude_nemo, pinn)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_build_model_registers_pinn -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/train.py tests/test_pinn_field.py
git commit -m "feat(pinn): register 'pinn' in build_model so train/bench wire up"
```

---

## Task 7: Spatial leave-one-well-out (unanchored headline + anchored)

**Files:**
- Modify: `hydrophysics/models/pinn_field.py` (add `leave_one_well_out_field` + CLI `main`)
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
def test_leave_one_well_out_field_modes(data):
    from hydrophysics.models.pinn_field import leave_one_well_out_field

    raw = leave_one_well_out_field(data, device="cpu", epochs=2, folds=2,
                                   n_collocation=32, anchor=False)
    anc = leave_one_well_out_field(data, device="cpu", epochs=2, folds=2,
                                   n_collocation=32, anchor=True)
    assert raw.shape == data.target.shape and anc.shape == data.target.shape
    assert np.isfinite(raw).all() and np.isfinite(anc).all()
    # anchoring shifts each well's train-period mean toward its observed mean
    i = 0
    obs = data.target[i, data.train_mask]
    obs = obs[np.isfinite(obs)]
    anc_mean = anc[i, data.train_mask].mean()
    assert abs(anc_mean - obs.mean()) <= abs(raw[i, data.train_mask].mean() - obs.mean()) + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_leave_one_well_out_field_modes -v`
Expected: FAIL with `ImportError: cannot import name 'leave_one_well_out_field'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/models/pinn_field.py  (append)

def leave_one_well_out_field(
    data: GWData, device: str, epochs: int, folds: int = 6, seed: int = 0,
    anchor: bool = False, **model_kwargs,
) -> np.ndarray:
    """(W, T) prediction where each well was predicted while held out of the data loss.

    Wells are assigned to folds round-robin by index. For each fold the held-out wells
    contribute no observation rows; their head is read from the field the other wells +
    physics built. ``anchor=False`` (the headline) returns the raw field; ``anchor=True``
    shifts each held-out well's series so its training-period mean matches the well's
    observed training mean (comparable to the lumped-UDE anchored LOWO).
    """
    assign = np.arange(data.n_wells) % folds
    pred = np.full_like(data.target, np.nan)
    for f in range(folds):
        held = assign == f
        model = SpatialPINN(device=device, epochs=epochs, seed=seed, **model_kwargs)
        model.fit(data, train_wells=~held)
        pred[held] = model.simulate(data)[held]
        print(f"fold {f + 1}/{folds}: trained on {int((~held).sum())} wells, "
              f"predicted {int(held.sum())} held-out")
    if anchor:
        for i in range(data.n_wells):
            obs = data.target[i, data.train_mask]
            obs = obs[np.isfinite(obs)]
            if obs.size:
                pred[i] += obs.mean() - pred[i, data.train_mask].mean()
    return pred


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from ..baselines import climatology_prediction
    from ..config import Config, default_config
    from ..data import load_dataset
    from ..eval import evaluate_predictions
    from ..train import pick_device

    ap = argparse.ArgumentParser(description="Spatial-PINN leave-one-well-out generalization")
    ap.add_argument("--data", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--anchor", action="store_true",
                    help="report the anchored variant (held-out well's observed mean) "
                         "instead of the unanchored headline.")
    args = ap.parse_args(argv)

    cfg = (Config(data_dir=Path(args.data)) if args.data else default_config())
    data = load_dataset(cfg)
    print(data.summary())
    device = pick_device(args.device)
    pred = leave_one_well_out_field(data, device, args.epochs, folds=args.folds,
                                    anchor=args.anchor)
    clim = evaluate_predictions(data, climatology_prediction(data), period="val")["kge"]
    per = evaluate_predictions(data, pred, period="val")
    k = per["kge"]
    n = int(k.notna().sum())
    mode = "anchored" if args.anchor else "unanchored (headline)"
    print(f"\n=== spatial PINN LOWO [{mode}] (held-out wells, validation) ===")
    print(f"KGE  median {k.median():.3f} | clipped[-1,1] mean {k.clip(-1, 1).mean():.3f}")
    print(f"NSE  median {per['nse'].median():.3f} | RMSE median {per['rmse'].median():.3f} m")
    print(f"beats own climatology on {int((k > clim).sum())}/{n} wells "
          f"(climatology median KGE {clim.median():.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_leave_one_well_out_field_modes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/models/pinn_field.py tests/test_pinn_field.py
git commit -m "feat(pinn): spatial leave-one-well-out (unanchored headline + anchored)"
```

---

## Task 8: Continuous head-field map rendering

**Files:**
- Modify: `hydrophysics/viz.py`
- Test: `tests/test_pinn_field.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pinn_field.py  (append)
def test_plot_head_field_returns_grid(data):
    from hydrophysics.models.pinn_field import SpatialPINN
    from hydrophysics.viz import head_field_grid

    model = SpatialPINN(device="cpu", epochs=2, n_collocation=32, seed=0)
    model.fit(data)
    XX, YY, HH = head_field_grid(model, data, day_index=int(np.flatnonzero(data.val_mask)[0]),
                                 n=16)
    assert XX.shape == (16, 16) and HH.shape == (16, 16)
    assert np.isfinite(HH).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pinn_field.py::test_plot_head_field_returns_grid -v`
Expected: FAIL with `ImportError: cannot import name 'head_field_grid'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/viz.py  (append; keep existing imports, add numpy if not present)
import numpy as np


def head_field_grid(model, data, day_index: int, n: int = 80):
    """Evaluate the PINN head field on an n x n grid over the well bounding box.

    Returns (XX, YY, HH) physical-coordinate meshgrids and head values (n, n).
    """
    x = data.attrs["tm_x"].astype(float).fillna(0.0).to_numpy()
    y = data.attrs["tm_y"].astype(float).fillna(0.0).to_numpy()
    xs = np.linspace(x.min(), x.max(), n)
    ys = np.linspace(y.min(), y.max(), n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    HH = model.head_field(pts, day_index).reshape(n, n)
    return XX, YY, HH


def plot_head_field(model, data, day_index: int, n: int = 80, ax=None):
    """Filled contour of the head field with wells overplotted. Lazy matplotlib."""
    import matplotlib.pyplot as plt

    XX, YY, HH = head_field_grid(model, data, day_index, n=n)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 8))
    cf = ax.contourf(XX, YY, HH, levels=20, cmap="viridis")
    ax.scatter(data.attrs["tm_x"].astype(float), data.attrs["tm_y"].astype(float),
               c="white", edgecolor="black", s=20, zorder=3)
    ax.set_title(f"PINN head field, day index {day_index}")
    ax.figure.colorbar(cf, ax=ax, label="head (m)")
    return ax
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pinn_field.py::test_plot_head_field_returns_grid -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/viz.py tests/test_pinn_field.py
git commit -m "feat(pinn): continuous head-field map rendering"
```

---

## Task 9: Full-suite check, real-data run, and docs

**Files:**
- Modify: `README.md`, `MODEL_CARD.md`
- Reference: results land under `results/pinn/`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all green (torch tests run if torch is installed; otherwise skipped). Fix any regressions before continuing.

- [ ] **Step 2: Lint**

Run: `ruff check hydrophysics/field_inputs.py hydrophysics/models/pinn_field.py hydrophysics/viz.py tests/test_pinn_field.py`
Expected: no errors. Fix anything reported.

- [ ] **Step 3: Train + benchmark on the real data (GPU box)**

Ensure `HYDROMIND_GW_DATA` points at the real `data/` directory (see `hydrophysics/config.py`). Then:

```bash
python -m hydrophysics.train --model pinn --device cuda --epochs 1500 --out results/pinn
python -m hydrophysics.models.pinn_field --device cuda --epochs 1500 --folds 6           # unanchored LOWO headline
python -m hydrophysics.models.pinn_field --device cuda --epochs 1500 --folds 6 --anchor  # anchored LOWO
```

Record the in-sample KGE/NSE/RMSE and both LOWO numbers. These are the three reported results from the spec.

- [ ] **Step 4: Update docs with the measured numbers**

In `README.md`, add a "Spatial head field (PINN)" subsection near the existing results table: state the in-sample KGE vs gray-box 0.736 / lumped UDE 0.591, and the **unanchored LOWO KGE as the headline** with the anchored number alongside (and today's lumped anchored 0.565 for context). Note the deliverable continuous map. In `MODEL_CARD.md`, add `pinn` to the model list with its method summary (continuous `h(x,y,t)`, 2D flow residual, learned `T(x,y)`), intended use, and limitations (depth-averaged, not volumetric; `T` interpretable up to scale; thin spatial data regularized by the `log T` L2 prior).

- [ ] **Step 5: Commit**

```bash
git add README.md MODEL_CARD.md results/pinn
git commit -m "docs(pinn): report spatial head-field results (in-sample, LOWO, map)"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** continuous `h(x,y,t)` field (Tasks 3-5); 2D flow residual with learned `T/α/d`, `S` (Task 4-5); rainfall field forcing (Task 2); dropped upstream coupling (not used anywhere — residual carries lateral flow); nondimensionalization (Task 1); in-sample benchmark (Task 5-6, 9); LOWO unanchored headline + anchored (Task 7); continuous map (Task 8); `log T` L2 regularizer for thin data (Task 5); analytic residual test, no-leakage test, interface smoke, rainfall test (Tasks 1-8); README + model card (Task 9). Coast Dirichlet realized via coastal wells' data loss + documented as a Phase-4 shapefile refinement (design-note section) — intentional, recorded.
- **Placeholder scan:** none — every code step is complete and runnable.
- **Type consistency:** `pde_residual(h_fn, X, Y, tau, rain, *, T_fn, alpha_fn, d_fn, S)` signature is identical in Task 4 (def + test) and Task 5 (call). `_obs_rows` returns `{"well","day","h"}`, consumed consistently in `fit` and the no-leakage test. `head_field(points_xy_phys, day_index)` matches `head_field_grid`'s call. `SpatialPINN.name == "pinn"` matches `build_model` and the benchmark assertion.
