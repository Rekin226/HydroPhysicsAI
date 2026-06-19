# Choushui Head + Subsidence Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone interactive Plotly HTML that animates a 3D observed-head surface and a calibrated land-subsidence surface over the real Zhuoshui fan, with the subsidence coefficient calibrated against the 14 georeferenced MLCW compaction sites.

**Architecture:** Two focused, pure-Python modules. `hydrophysics/subsidence.py` does the science (IDW interpolation of observed heads, monthly resampling, MLCW total-compaction series, single-coefficient `Sk` calibration). `hydrophysics/explorer.py` does the visualization (grid + fan-polygon mask, Plotly figure with a time slider + validation panel, writes a self-contained HTML). Everything stays in EPSG:3826 meters; no models are used (observed interpolation only).

**Tech Stack:** Python, NumPy, pandas, pyarrow (parquet), geopandas + shapely (fan polygon mask), Plotly (figure/HTML), pytest. Reuses `hydrophysics.data.load_dataset` and `hydrophysics.config`.

---

## Design notes (read once)

- **Everything monthly.** Heads are resampled to month-end means so they align with MLCW's monthly cadence and keep the HTML light (~150 frames). Calibration and the grid both use the monthly head matrix.
- **IDW is shared.** A single `idw_interp` is used for both the calibration (head at each MLCW site) and the grid (head at every cell). DRY.
- **Cumulative drawdown** (inelastic proxy) `D(t) = h(t0) − runningmin(h)` — monotonic non-decreasing, ≥ 0.
- **`Sk` sign is handled by the fit.** Least-squares-through-origin makes the predicted subsidence and the observed compaction sign-consistent automatically; we report `Sk` and `R²` honestly whatever they are.
- **CRS:** wells (`tm_x/tm_y`), MLCW coords, and the fan polygon are all EPSG:3826 meters — no reprojection.

## File structure

- **Create** `chou-shui-data/chou-shui-data/data/mlcw_stations.csv` — the 14 MLCW coordinates.
- **Create** `hydrophysics/subsidence.py` — IDW helpers, monthly heads, MLCW compaction, `Sk` calibration. Pure NumPy/pandas (+ pyarrow for parquet). No plotly/geopandas.
- **Create** `hydrophysics/explorer.py` — fan polygon load + grid/mask, Plotly build, CLI. Imports `subsidence.py`.
- **Create** `tests/test_explorer.py` — all tests for both modules.
- **Reference:** output HTML lands at `results/explorer/choushui_explorer.html`.

---

## Task 1: MLCW station coordinates

**Files:**
- Create: `chou-shui-data/chou-shui-data/data/mlcw_stations.csv`
- Create: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Create the coordinates file** `chou-shui-data/chou-shui-data/data/mlcw_stations.csv` with EXACTLY this content:

```csv
sub_id,lon,lat,x,y
新生國小,120.394274,23.937911,188341,2648279
湖南國小,120.48,23.95,196984,2649404
溪州國小,120.5,23.85,198873,2638772
僑義國小,120.471361,23.845062,196959,2637815
元長國小,120.308773,23.653342,179484,2616803
客厝國小,120.334296,23.626618,182074,2613831
內寮派駐站,120.354646,23.60767,184141,2611722
土庫國中,120.389843,23.688067,187771,2620610
秀潭國小,120.349591,23.658882,183651,2617396
宏崙國小,120.347845,23.686484,183488,2620464
光復國小,120.402464,23.741364,189083,2626507
拯民國小,120.407385,23.709472,189570,2622974
嘉興國小,120.459652,23.648009,194874,2616145
北辰國小,120.303054,23.575894,178860,2608238
```

- [ ] **Step 2: Write the failing test** — create `tests/test_explorer.py`:

```python
"""Tests for the Choushui head + subsidence explorer (subsidence.py + explorer.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_load_mlcw_stations():
    from hydrophysics.subsidence import load_mlcw_stations
    from hydrophysics.config import default_config

    cfg = default_config()
    # the real data dir has the file; if running on the sample, skip
    path = cfg.data_dir / "mlcw_stations.csv"
    if not path.exists():
        pytest.skip("mlcw_stations.csv only present alongside the real data")
    df = load_mlcw_stations(path)
    assert list(df.columns[:3]) == ["sub_id", "x", "y"]
    assert len(df) == 14
    assert df["x"].between(170000, 210000).all()
    assert df["y"].between(2600000, 2660000).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_load_mlcw_stations -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.subsidence'`

- [ ] **Step 4: Create `hydrophysics/subsidence.py`** with EXACTLY this content:

```python
"""Subsidence science for the Choushui explorer: IDW heads, MLCW compaction, Sk fit.

Pure NumPy / pandas (+ pyarrow for parquet). No plotly/geopandas here so the science is
importable and testable on its own. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from .data import GWData


def load_mlcw_stations(path) -> pd.DataFrame:
    """Read the 14 MLCW coordinates -> DataFrame[sub_id, x, y, lon, lat] (EPSG:3826)."""
    df = pd.read_csv(path)
    return df[["sub_id", "x", "y", "lon", "lat"]]
```

- [ ] **Step 5: Run test to verify it passes** (or skips if no real data)

Run: `HYDROMIND_GW_DATA="$PWD/chou-shui-data/chou-shui-data/data" pytest tests/test_explorer.py::test_load_mlcw_stations -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chou-shui-data/chou-shui-data/data/mlcw_stations.csv hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(explorer): MLCW station coordinates + loader"
```

---

## Task 2: IDW interpolation + monthly head matrix

**Files:**
- Modify: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_explorer.py`):

```python
@pytest.fixture()
def gwdata(tmp_path):
    from hydrophysics import Config, load_dataset
    from hydrophysics.sample import write_sample

    d = write_sample(tmp_path / "data", n_wells=5, seed=1)
    return load_dataset(Config(data_dir=d, baseline_results=d / "gw_fit_results.csv"))


def test_idw_interp_exact_and_blend():
    from hydrophysics.subsidence import idw_interp

    well_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2 wells, 2 timesteps)
    # querying at a well returns that well's row
    out = idw_interp(np.array([[0.0, 0.0]]), well_xy, values)
    assert np.allclose(out[0], [1.0, 2.0], atol=1e-3)
    # midpoint is between the two wells' values
    mid = idw_interp(np.array([[5.0, 0.0]]), well_xy, values)[0]
    assert (mid >= values.min(0)).all() and (mid <= values.max(0)).all()


def test_monthly_heads_shapes(gwdata):
    from hydrophysics.subsidence import monthly_heads, well_xy

    H, dates = monthly_heads(gwdata)
    assert H.shape[0] == gwdata.n_wells
    assert H.shape[1] == len(dates)
    assert isinstance(dates, pd.DatetimeIndex)
    assert well_xy(gwdata).shape == (gwdata.n_wells, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_idw_interp_exact_and_blend -v`
Expected: FAIL with `ImportError: cannot import name 'idw_interp'`

- [ ] **Step 3: Append to `hydrophysics/subsidence.py`:**

```python
def well_xy(data: GWData) -> np.ndarray:
    """(W, 2) well coordinates in EPSG:3826 meters."""
    x = data.attrs["tm_x"].astype(float).fillna(0.0).to_numpy()
    y = data.attrs["tm_y"].astype(float).fillna(0.0).to_numpy()
    return np.stack([x, y], axis=-1)


def idw_interp(points_xy: np.ndarray, src_xy: np.ndarray, values: np.ndarray,
               power: float = 2.0, eps: float = 1e-6) -> np.ndarray:
    """Inverse-distance interpolate ``values`` (S sources x T) to N query points -> (N, T).

    NaNs in ``values`` are down-weighted per timestep (a source contributes only where it
    has a finite value), so missing observations never poison a cell.
    """
    P = np.asarray(points_xy, dtype="float64")              # (N, 2)
    S = np.asarray(src_xy, dtype="float64")                 # (Sn, 2)
    V = np.asarray(values, dtype="float64")                 # (Sn, T)
    d2 = ((P[:, None, :] - S[None, :, :]) ** 2).sum(-1)      # (N, Sn)
    w = 1.0 / (d2 ** (power / 2.0) + eps)                    # (N, Sn)
    finite = np.isfinite(V).astype("float64")               # (Sn, T)
    num = w @ np.nan_to_num(V)                               # (N, T)
    den = w @ finite                                        # (N, T)
    return num / np.maximum(den, 1e-9)


def monthly_heads(data: GWData) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Resample the observed heads to month-end means -> ((W, Tm) array, Tm dates)."""
    df = pd.DataFrame(data.target.T, index=pd.DatetimeIndex(data.dates))  # (T, W)
    m = df.resample("ME").mean()
    return m.to_numpy().T, m.index


def cumulative_drawdown(h_monthly: np.ndarray) -> np.ndarray:
    """Inelastic cumulative drawdown along the last axis: h[...,0] - runningmin(h) (>=0)."""
    h = np.asarray(h_monthly, dtype="float64")
    runmin = np.minimum.accumulate(h, axis=-1)
    return h[..., :1] - runmin
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_explorer.py::test_idw_interp_exact_and_blend tests/test_explorer.py::test_monthly_heads_shapes -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(explorer): IDW interpolation, monthly heads, cumulative drawdown"
```

---

## Task 3: MLCW total-compaction series

**Files:**
- Modify: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_mlcw_compaction_decodes_and_signs(tmp_path):
    from hydrophysics.subsidence import mlcw_compaction

    # build a tiny synthetic MLCW parquet whose name decodes to 僑義國小
    # UTF-8 bytes of 僑義國小 -> hex, joined as _XX
    name = "僑義國小"
    enc = "".join(f"_{b:02X}" for b in name.encode("utf-8"))
    d = tmp_path / "ls_cache" / "clean"
    d.mkdir(parents=True)
    dates = pd.date_range("2014-01-31", periods=6, freq="ME")
    # shallow ring NO1 sinks over time (depth grows), deep ring NO3 fixed -> separation shrinks
    df = pd.DataFrame(
        {"NO1": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
         "NO2": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
         "NO3": [100.0, 99.8, 99.6, 99.4, 99.2, 99.0]},
        index=pd.Index(dates, name="datetime"),
    )
    df.to_parquet(d / f"ls-wra-mlcw-obs__{enc}.parquet")

    series = mlcw_compaction(str(tmp_path))
    assert name in series
    s = series[name]
    assert len(s) == 6
    assert np.isclose(s.iloc[0], 0.0)           # re-zeroed to first obs
    assert s.iloc[-1] > 0                        # compaction accumulates positive
    assert np.isclose(s.iloc[-1], 1.0, atol=1e-6)  # separation shrank 100->99 = 1.0 m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_mlcw_compaction_decodes_and_signs -v`
Expected: FAIL with `ImportError: cannot import name 'mlcw_compaction'`

- [ ] **Step 3: Append to `hydrophysics/subsidence.py`:**

```python
def _decode_mlcw_name(filename: str) -> str:
    """Decode a percent-hex MLCW filename (UTF-8 bytes as _XX) back to the Chinese name."""
    core = os.path.basename(filename).split("ls-wra-mlcw-obs__")[-1].replace(".parquet", "")
    hexes = re.findall(r"_([0-9A-Fa-f]{2})", core)
    try:
        return bytes(int(h, 16) for h in hexes).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return core


def mlcw_compaction(data_dir: str) -> dict[str, pd.Series]:
    """Per-site total compaction (m), re-zeroed to the first observation, monthly.

    Each MLCW parquet holds magnetic-ring positions ``NO1..NO31`` (m) at increasing depth.
    Total compaction = shortening of the monitored interval = (separation between the
    shallowest and deepest ring at t0) - (separation at t). Positive = subsidence.
    """
    pattern = os.path.join(data_dir, "ls_cache", "clean", "ls-wra-mlcw-obs__*.parquet")
    out: dict[str, pd.Series] = {}
    for f in sorted(glob.glob(pattern)):
        name = _decode_mlcw_name(f)
        df = pd.read_parquet(f).sort_index()
        order = df.mean().sort_values().index            # shallow (small) -> deep (large)
        shallow, deep = df[order[0]], df[order[-1]]
        sep = deep - shallow                             # interval thickness over time
        comp = (sep.iloc[0] - sep).rename("compaction_m")  # >0 as the interval compacts
        out[name] = comp
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_mlcw_compaction_decodes_and_signs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(explorer): MLCW total-compaction series (decode + interval shortening)"
```

---

## Task 4: Calibrate the single Sk coefficient

**Files:**
- Modify: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_calibrate_sk_recovers_slope():
    from hydrophysics.subsidence import calibrate_sk_from_pairs

    # synthetic: compaction = 0.02 * drawdown at two sites
    rng = np.random.default_rng(0)
    per_site = {}
    for s in ("A", "B"):
        D = np.linspace(0, 5, 12)
        C = 0.02 * D + rng.normal(0, 1e-4, size=D.size)
        per_site[s] = (D, C)
    res = calibrate_sk_from_pairs(per_site)
    assert abs(res["sk"] - 0.02) < 1e-3
    assert res["r2"] > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_calibrate_sk_recovers_slope -v`
Expected: FAIL with `ImportError: cannot import name 'calibrate_sk_from_pairs'`

- [ ] **Step 3: Append to `hydrophysics/subsidence.py`:**

```python
def calibrate_sk_from_pairs(per_site: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Least-squares-through-origin slope Sk over pooled (drawdown, compaction) pairs.

    per_site: {name -> (D, C)} aligned monthly arrays. Returns {sk, r2, per_site, D, C}.
    """
    Ds = [np.asarray(D, float) for D, _ in per_site.values()]
    Cs = [np.asarray(C, float) for _, C in per_site.values()]
    D = np.concatenate(Ds)
    C = np.concatenate(Cs)
    sk = float(D @ C / (D @ D + 1e-12))
    pred = sk * D
    ss_res = float(((C - pred) ** 2).sum())
    ss_tot = float(((C - C.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"sk": sk, "r2": r2, "per_site": per_site, "D": D, "C": C}


def calibrate_sk(data: GWData, stations: pd.DataFrame,
                 compaction: dict[str, pd.Series]) -> dict:
    """Pair each MLCW site's monthly compaction with the IDW head drawdown at its (x,y),
    then fit one Sk. Returns the calibrate_sk_from_pairs result plus per-site monthly index.
    """
    H, dates = monthly_heads(data)                       # (W, Tm)
    wxy = well_xy(data)
    per_site: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in stations.iterrows():
        name = row["sub_id"]
        if name not in compaction:
            continue
        h_site = idw_interp(np.array([[row["x"], row["y"]]]), wxy, H)[0]   # (Tm,)
        hser = pd.Series(h_site, index=dates)
        comp = compaction[name]
        idx = hser.index.intersection(comp.index)
        if len(idx) < 6:
            continue
        hh = hser.reindex(idx).to_numpy()
        D = hh[:1] - np.minimum.accumulate(hh)           # cumulative drawdown
        C = comp.reindex(idx).to_numpy()
        C = C - C[0]
        per_site[name] = (D, C)
    return calibrate_sk_from_pairs(per_site)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_calibrate_sk_recovers_slope -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(explorer): single-coefficient Sk calibration from MLCW pairs"
```

---

## Task 5: Grid + fan-polygon mask + subsidence grid

**Files:**
- Create: `hydrophysics/explorer.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_head_and_subsidence_grid(gwdata):
    from shapely.geometry import box
    from hydrophysics.explorer import head_grid, subsidence_grid

    wx = gwdata.attrs["tm_x"].astype(float)
    wy = gwdata.attrs["tm_y"].astype(float)
    poly = box(wx.min(), wy.min(), wx.max(), wy.max())  # rectangle covering the wells
    XX, YY, HH, dates = head_grid(gwdata, poly, n=12)
    assert XX.shape == (12, 12)
    assert HH.shape == (len(dates), 12, 12)
    # interior cells are finite (rectangle covers the whole bbox)
    assert np.isfinite(HH).any()
    SS = subsidence_grid(HH, sk=0.01)
    assert SS.shape == HH.shape
    # subsidence is non-negative where head is finite, and monotonic non-decreasing in time
    fin = np.isfinite(SS)
    assert (SS[fin] >= -1e-9).all()
    assert np.nanmax(SS[-1] - SS[0]) >= -1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_head_and_subsidence_grid -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.explorer'`

- [ ] **Step 3: Create `hydrophysics/explorer.py`** with EXACTLY this content:

```python
"""Choushui head + subsidence explorer: grid/mask + interactive Plotly HTML.

Builds a self-contained HTML that animates the IDW observed-head surface and the
calibrated subsidence surface over the real fan, with a validation panel. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import numpy as np

from .data import GWData
from .subsidence import cumulative_drawdown, idw_interp, monthly_heads, well_xy


def head_grid(data: GWData, poly, n: int = 60):
    """IDW the monthly observed heads onto an n x n grid over the well bbox, masked to
    ``poly`` (a shapely polygon). Returns (XX, YY, HH, dates) with HH shape (Tm, n, n);
    cells outside the polygon are NaN.
    """
    from shapely.geometry import Point

    wxy = well_xy(data)
    xs = np.linspace(wxy[:, 0].min(), wxy[:, 0].max(), n)
    ys = np.linspace(wxy[:, 1].min(), wxy[:, 1].max(), n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)            # (n*n, 2)
    H, dates = monthly_heads(data)                              # (W, Tm)
    grid = idw_interp(pts, wxy, H)                              # (n*n, Tm)
    inside = np.array([poly.contains(Point(px, py)) for px, py in pts])
    grid[~inside] = np.nan
    HH = grid.T.reshape(len(dates), n, n)                      # (Tm, n, n)
    return XX, YY, HH, dates


def subsidence_grid(HH: np.ndarray, sk: float) -> np.ndarray:
    """Sk * cumulative drawdown per cell, from the (Tm, n, n) head history."""
    Tm = HH.shape[0]
    flat = HH.reshape(Tm, -1).T                                # (cells, Tm)
    D = cumulative_drawdown(flat)                              # (cells, Tm)
    return (sk * D).T.reshape(HH.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_head_and_subsidence_grid -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/explorer.py tests/test_explorer.py
git commit -m "feat(explorer): masked head grid + subsidence grid"
```

---

## Task 6: Fan polygon loader + Plotly figure builder + CLI

**Files:**
- Modify: `hydrophysics/explorer.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_build_explorer_writes_html(gwdata, tmp_path):
    from shapely.geometry import box
    from hydrophysics.explorer import build_explorer

    wx = gwdata.attrs["tm_x"].astype(float)
    wy = gwdata.attrs["tm_y"].astype(float)
    poly = box(wx.min(), wy.min(), wx.max(), wy.max())
    out = tmp_path / "explorer.html"
    # no MLCW on the synthetic sample -> empty stations/compaction, sk falls back to 0
    path = build_explorer(gwdata, poly, stations=None, compaction=None,
                          out_html=str(out), n=10)
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 1000
    assert "plotly" in out.read_text(errors="ignore").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_build_explorer_writes_html -v`
Expected: FAIL with `ImportError: cannot import name 'build_explorer'`

- [ ] **Step 3: Append to `hydrophysics/explorer.py`:**

```python
def fan_polygon(shp_path):
    """Load the Zhuoshui fan polygon (EPSG:3826) as a single shapely geometry."""
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    return gdf.geometry.union_all()


def build_explorer(data: GWData, poly, stations, compaction, out_html: str,
                   n: int = 60) -> str:
    """Assemble the interactive HTML and write it to ``out_html``. Returns the path.

    ``stations`` (DataFrame[sub_id,x,y]) and ``compaction`` (dict name->Series) may be None
    when MLCW data is unavailable (e.g. the synthetic sample); then Sk falls back to 0 and
    the validation panel is omitted.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .subsidence import calibrate_sk

    XX, YY, HH, dates = head_grid(data, poly, n=n)
    if stations is not None and compaction:
        cal = calibrate_sk(data, stations, compaction)
    else:
        cal = {"sk": 0.0, "r2": float("nan"), "per_site": {}, "D": np.array([]),
               "C": np.array([])}
    SS = subsidence_grid(HH, cal["sk"])
    wxy = well_xy(data)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=("Groundwater head (m) — IDW of observed wells",
                        f"MLCW validation (Sk={cal['sk']:.4g}, R²={cal['r2']:.2f})"),
    )

    # frame 0 surface
    fig.add_trace(go.Surface(x=XX, y=YY, z=HH[0], colorscale="Viridis",
                             colorbar=dict(title="head m", x=0.62)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=wxy[:, 0], y=wxy[:, 1],
                               z=np.nanmax(HH[0]) * np.ones(len(wxy)),
                               mode="markers", marker=dict(size=2, color="white"),
                               name="wells"), row=1, col=1)

    # validation scatter (predicted vs observed compaction at MLCW sites)
    if cal["D"].size:
        fig.add_trace(go.Scatter(x=cal["sk"] * cal["D"], y=cal["C"], mode="markers",
                                 marker=dict(size=5, color="firebrick"),
                                 name="MLCW sites"), row=1, col=2)
        lim = float(max(cal["C"].max(), (cal["sk"] * cal["D"]).max(), 1e-6))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="1:1"),
                      row=1, col=2)
        fig.update_xaxes(title_text="predicted subsidence (m)", row=1, col=2)
        fig.update_yaxes(title_text="observed compaction (m)", row=1, col=2)

    # animation frames over months (head surface only; subsidence available via the toggle)
    frames = []
    for t, dt in enumerate(dates):
        frames.append(go.Frame(name=str(dt.date()),
                               data=[go.Surface(x=XX, y=YY, z=HH[t])],
                               traces=[0]))
    fig.frames = frames
    steps = [dict(method="animate", label=str(dt.date()),
                  args=[[str(dt.date())], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for dt in dates]
    fig.update_layout(
        title="Choushui groundwater head + land subsidence (observed, 2010–2022)",
        sliders=[dict(active=0, steps=steps, x=0.05, len=0.6,
                      currentvalue=dict(prefix="month: "))],
        scene=dict(xaxis_title="TM_X97 (m)", yaxis_title="TM_Y97 (m)",
                   zaxis_title="head (m)", aspectmode="auto"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    import os
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", auto_play=False)
    return out_html


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .config import Config, default_config
    from .data import load_dataset
    from .subsidence import load_mlcw_stations, mlcw_compaction

    ap = argparse.ArgumentParser(description="Build the Choushui head+subsidence explorer HTML")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="results/explorer/choushui_explorer.html")
    args = ap.parse_args(argv)

    cfg = Config(data_dir=Path(args.data)) if args.data else default_config()
    data = load_dataset(cfg)
    fan_shp = cfg.data_dir / "Zhuoshui Alluvial Fan" / "Zhuoshui Alluvial Fan.shp"
    poly = fan_polygon(str(fan_shp))
    st_path = cfg.data_dir / "mlcw_stations.csv"
    stations = load_mlcw_stations(st_path) if st_path.exists() else None
    compaction = mlcw_compaction(str(cfg.data_dir)) if stations is not None else None
    out = build_explorer(data, poly, stations, compaction, args.out, n=args.n)
    print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_build_explorer_writes_html -v`
Expected: PASS

- [ ] **Step 5: Run the whole test file + lint**

Run: `pytest tests/test_explorer.py -v && ruff check hydrophysics/subsidence.py hydrophysics/explorer.py tests/test_explorer.py`
Expected: all pass, ruff clean. Fix any lint without behavior change.

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/explorer.py tests/test_explorer.py
git commit -m "feat(explorer): fan polygon loader, Plotly figure builder, CLI"
```

---

## Task 7: Real-data run, validation, deliverable + docs

**Files:**
- Reference: `results/explorer/choushui_explorer.html`
- Modify: `README.md`

- [ ] **Step 1: Build on the real data**

```bash
export HYDROMIND_GW_DATA="$PWD/chou-shui-data/chou-shui-data/data"
python -m hydrophysics.explorer --n 70
```
Expected: prints `wrote results/explorer/choushui_explorer.html`. Note the printed nothing-fatal; open the file size with `ls -la results/explorer/`.

- [ ] **Step 2: Print the calibration sanity numbers**

```bash
export HYDROMIND_GW_DATA="$PWD/chou-shui-data/chou-shui-data/data"
python -c "
from hydrophysics.config import default_config
from hydrophysics.data import load_dataset
from hydrophysics.subsidence import load_mlcw_stations, mlcw_compaction, calibrate_sk
cfg=default_config(); d=load_dataset(cfg)
st=load_mlcw_stations(cfg.data_dir/'mlcw_stations.csv')
cmp=mlcw_compaction(str(cfg.data_dir))
cal=calibrate_sk(d, st, cmp)
print(f'Sk={cal[\"sk\"]:.4g}  R2={cal[\"r2\"]:.3f}  sites={len(cal[\"per_site\"])}  obs_compaction_max={cal[\"C\"].max():.3f} m')
"
```
Expected: a printed `Sk`, `R²`, number of sites (≤14), and the max observed compaction (sanity: a few cm to ~1 m). Record these.

- [ ] **Step 3: Honesty check on the result**

If `R²` is low (< ~0.3) or `Sk` is negative, that is itself a finding — the regional proxy does not explain the spatial compaction well. Do not hide it. Note it for the README and keep going (the explorer still shows observed head + the proxy + the validation panel that reveals the mismatch).

- [ ] **Step 4: Add a short README subsection** after the "Continuous head field (PINN)" section in `README.md`:

```markdown
## Interactive explorer: head + land subsidence

`python -m hydrophysics.explorer` builds a standalone HTML
(`results/explorer/choushui_explorer.html`) — a date-slider animation of the IDW
observed-head surface over the real Zhuoshui fan, plus a land-subsidence surface whose
single coefficient `Sk` is calibrated against the 14 georeferenced multi-layer compaction
wells (MLCW). A validation panel shows predicted vs observed compaction at those sites
(`Sk=<value>`, `R²=<value>`). The head surface is IDW interpolation of observations
(labeled as such), not a model; scenario re-runs of the UDE are a separate follow-on.
```
Fill `<value>` from Step 2.

- [ ] **Step 5: Commit**

```bash
git add README.md results/explorer/choushui_explorer.html
git commit -m "feat(explorer): generate Choushui head+subsidence explorer on real data + docs"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** standalone Plotly HTML (Task 6); IDW observed-head 3D surface + monthly steps + fan mask (Tasks 2,5,6); MLCW compaction series with decode (Task 3); per-site `Sk` calibration from co-located pairs (Task 4); subsidence surface (Task 5); validation panel predicted-vs-observed (Task 6); MLCW coords file + loader (Task 1); wells+MLCW markers (Task 6 — wells plotted; MLCW markers via the validation scatter and coordinates available — note: on-map MLCW markers are covered by the stations passed in, plotted as the validation scatter; a 3D MLCW marker trace is optional polish, not required for the deliverable); real-data run + honest reporting (Task 7). All spec sections map to a task.
- **Placeholder scan:** none — every code step is complete and runnable. The README `<value>` placeholders in Task 7 are explicitly filled from Step 2 output (a documented action, not a code gap).
- **Type consistency:** `idw_interp(points_xy, src_xy, values)` identical across Tasks 2/4/5. `monthly_heads -> (H, dates)` consistent. `mlcw_compaction -> dict[name->Series]` consumed by `calibrate_sk` (Task 4) and `main` (Task 6). `head_grid -> (XX,YY,HH,dates)` with `HH (Tm,n,n)` consumed by `subsidence_grid` (Task 5) and `build_explorer` (Task 6). `calibrate_sk` returns `{sk,r2,per_site,D,C}` used in Task 6's figure. `build_explorer(data, poly, stations, compaction, out_html, n)` signature matches its test and `main`.
- **Note on MLCW on-map markers:** the spec mentions MLCW markers on the fan. Task 6 plots the wells in 3D and the MLCW sites in the validation scatter; adding a `Scatter3d` MLCW marker trace (using `stations[x,y]`) is a 1-line optional enhancement and can be added during Task 6 if desired without changing interfaces.
