# Per-site Sk via Distance-to-Coast Regression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model the per-site compaction coefficient `Sk` as `exp(β₀+β₁·distance_to_coast)`, gate it by leave-one-site-out, and only wire a continuous subsidence surface into the explorer if the gate passes (LOSO R² > 0).

**Architecture:** Append a small, leakage-safe regression layer to `hydrophysics/subsidence.py` (distance-to-coast, weighted log-linear fit, leave-one-site-out gate) and coast-surface builders to `hydrophysics/explorer.py`. A `subsidence_report` CLI prints the verdict table. The explorer surface and README claims are updated **conditionally** on the measured LOSO number.

**Tech Stack:** NumPy, pandas, geopandas + shapely (distance-to-coast), Plotly (existing explorer), pytest. Reuses `calibrate_sk`, `cumulative_drawdown`, `head_grid`, `subsidence_grid` from the existing explorer code.

---

## Design notes (read once)

- **Leakage discipline:** the predictor path takes only `distance` + the `(D, C)` pairs. `D` (drawdown) comes from head, the predictor is a function of `distance_to_coast` *only*. No feature is derived from the compaction magnitude `C`. Tests assert this.
- **Model compaction directly:** `C ≈ Sk(dc)·D` with `Sk(dc)=exp(β₀+β₁·dc)`. Fit `log Sk_i` on `dc` weighted by `ΣD²` (reliability of each site's `Sk_i`).
- **The gate is leave-one-site-out pooled R² on compaction.** Pass bar > 0. Reference baselines (this repo): single-Sk −0.28, spatial-IDW LOSO −2.40, per-site in-sample 0.81.
- **Distances are grid-evaluable:** `distance_to_coast` can be computed at every grid cell (needed for the surface). Column-depth (borehole-only) is deliberately excluded.

## File structure

- **Modify** `hydrophysics/subsidence.py` — append `_distances_to_geom`, `site_distance_to_coast`, `fit_sk_regression`, `loso_sk_regression`.
- **Modify** `hydrophysics/explorer.py` — append `coast_distance_grid`, `subsidence_grid_from_sk_field`; extend `build_explorer` with a `coupling` switch.
- **Create** `hydrophysics/subsidence_report.py` — CLI printing the verdict table.
- **Modify** `tests/test_explorer.py` — tests for all new functions.
- **Modify** `README.md` — conditional on the verdict (Task 6).

---

## Task 1: Distance-to-coast per MLCW site

**Files:**
- Modify: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_explorer.py`):

```python
def test_distances_to_geom():
    from shapely.geometry import LineString
    from hydrophysics.subsidence import _distances_to_geom

    coast = LineString([(0.0, 0.0), (0.0, 100.0)])  # the y-axis
    stations = pd.DataFrame({"sub_id": ["a", "b"], "x": [10.0, 30.0], "y": [50.0, 50.0]})
    d = _distances_to_geom(stations, coast)
    assert abs(d["a"] - 10.0) < 1e-6
    assert abs(d["b"] - 30.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_distances_to_geom -v`
Expected: FAIL with `ImportError: cannot import name '_distances_to_geom'`

- [ ] **Step 3: Append to `hydrophysics/subsidence.py`:**

```python
def _distances_to_geom(stations: pd.DataFrame, geom) -> dict[str, float]:
    """Distance from each station (x, y) to a shapely geometry. Pure shapely (testable)."""
    from shapely.geometry import Point

    return {row["sub_id"]: float(Point(row["x"], row["y"]).distance(geom))
            for _, row in stations.iterrows()}


def site_distance_to_coast(stations: pd.DataFrame, coast_shp) -> dict[str, float]:
    """Distance (m, EPSG:3826) from each MLCW site to the coastline polygon/line."""
    import geopandas as gpd

    geom = gpd.read_file(coast_shp).geometry.union_all()
    return _distances_to_geom(stations, geom)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_distances_to_geom -v`
Expected: PASS

- [ ] **Step 5: Real-data sanity** (report, do not gate):

```bash
HYDROMIND_GW_DATA="$PWD/chou-shui-data/chou-shui-data/data" python -c "
from hydrophysics.config import default_config
from hydrophysics.subsidence import load_mlcw_stations, site_distance_to_coast
cfg=default_config()
st=load_mlcw_stations(cfg.data_dir/'mlcw_stations.csv')
d=site_distance_to_coast(st, cfg.data_dir/'water'/'sea_TWD97.shp')
print('sites:', len(d), 'km range:', round(min(d.values())/1000,1), '-', round(max(d.values())/1000,1))
"
```
Expected: `sites: 14 km range: 6.4 - 25.7` (or close). If it errors, STOP and report.

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(subsidence): distance-to-coast per MLCW site"
```

---

## Task 2: Weighted log-linear Sk regression

**Files:**
- Modify: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_fit_sk_regression_recovers_beta():
    from hydrophysics.subsidence import fit_sk_regression

    rng = np.random.default_rng(0)
    b0, b1 = -3.0, -1.5
    dist = {}
    per_site = {}
    for i in range(8):
        dc = 0.2 * i                      # spread of distances
        sk = np.exp(b0 + b1 * dc)
        D = np.linspace(0.5, 5.0, 20)     # drawdown
        C = sk * D + rng.normal(0, 1e-4, size=D.size)
        name = f"s{i}"
        dist[name] = dc
        per_site[name] = (D, C)
    fit = fit_sk_regression(per_site, dist)
    assert abs(fit["b0"] - b0) < 0.1
    assert abs(fit["b1"] - b1) < 0.1
    # predict_sk is a function of distance only
    assert abs(fit["predict_sk"](0.0) - np.exp(b0)) < 1e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_fit_sk_regression_recovers_beta -v`
Expected: FAIL with `ImportError: cannot import name 'fit_sk_regression'`

- [ ] **Step 3: Append to `hydrophysics/subsidence.py`:**

```python
def _per_site_sk(per_site: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[float, float]]:
    """Per-site (Sk_i, weight=ΣD²). Sk_i is the through-origin slope of C on D."""
    out = {}
    for name, (D, C) in per_site.items():
        D = np.asarray(D, float)
        C = np.asarray(C, float)
        denom = float(D @ D)
        out[name] = ((D @ C) / (denom + 1e-12), denom)
    return out


def fit_sk_regression(per_site: dict[str, tuple[np.ndarray, np.ndarray]],
                      dist: dict[str, float]) -> dict:
    """Weighted least-squares fit of log(Sk_i) on distance-to-coast.

    Sk(dc) = exp(b0 + b1*dc). Only sites with Sk_i > 0 enter the log fit; each is weighted
    by ΣD² (how well its Sk_i is determined). The predictor is a function of distance ONLY
    (no compaction-derived feature) -> leakage-safe. Returns {b0, b1, r2_insample,
    predict_sk, sk_per_site}.
    """
    sk_w = _per_site_sk(per_site)
    use = [n for n in sk_w if n in dist and sk_w[n][0] > 0]
    dc = np.array([dist[n] for n in use], float)
    y = np.log(np.array([sk_w[n][0] for n in use], float))
    w = np.array([sk_w[n][1] for n in use], float)
    w = w / w.sum()
    X = np.stack([np.ones_like(dc), dc], axis=1)               # (m, 2)
    WX = X * w[:, None]
    beta = np.linalg.solve(X.T @ WX, X.T @ (w * y))            # weighted normal equations
    b0, b1 = float(beta[0]), float(beta[1])

    def predict_sk(d):
        d = np.asarray(d, float)
        return np.exp(b0 + b1 * d)

    sk_true = np.array([sk_w[n][0] for n in use], float)
    pred = predict_sk(dc)
    ss_res = float(((sk_true - pred) ** 2).sum())
    ss_tot = float(((sk_true - sk_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"b0": b0, "b1": b1, "r2_insample": r2, "predict_sk": predict_sk,
            "sk_per_site": {n: sk_w[n][0] for n in sk_w}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_fit_sk_regression_recovers_beta -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(subsidence): weighted log-linear Sk vs distance-to-coast regression"
```

---

## Task 3: Leave-one-site-out gate

**Files:**
- Modify: `hydrophysics/subsidence.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_loso_discriminates_signal_from_noise():
    from hydrophysics.subsidence import loso_sk_regression

    rng = np.random.default_rng(1)
    D = np.linspace(0.5, 5.0, 20)

    # (a) real coast gradient: Sk depends on distance -> LOSO should be positive
    grad = {}
    grad_dist = {}
    for i in range(10):
        dc = 0.3 * i
        sk = np.exp(-2.0 - 1.2 * dc)
        name = f"g{i}"
        grad_dist[name] = dc
        grad[name] = (D, sk * D + rng.normal(0, 1e-4, size=D.size))
    res_grad = loso_sk_regression(grad, grad_dist)
    assert res_grad["r2"] > 0.3

    # (b) Sk independent of distance (random) -> LOSO should NOT be positive
    rnd = {}
    rnd_dist = {}
    for i in range(10):
        sk = float(rng.uniform(0.02, 0.1))
        name = f"r{i}"
        rnd_dist[name] = float(rng.uniform(0, 3))
        rnd[name] = (D, sk * D + rng.normal(0, 1e-4, size=D.size))
    res_rnd = loso_sk_regression(rnd, rnd_dist)
    assert res_rnd["r2"] <= 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_loso_discriminates_signal_from_noise -v`
Expected: FAIL with `ImportError: cannot import name 'loso_sk_regression'`

- [ ] **Step 3: Append to `hydrophysics/subsidence.py`:**

```python
def loso_sk_regression(per_site: dict[str, tuple[np.ndarray, np.ndarray]],
                       dist: dict[str, float]) -> dict:
    """Leave-one-site-out gate: predict each held-out site's Sk from the others' fit,
    score pooled compaction R². Returns {r2, n_sites, per_site_pred}.
    """
    names = [n for n in per_site if n in dist]
    preds, obs, per_pred = [], [], {}
    for held in names:
        train = {n: per_site[n] for n in names if n != held}
        fit = fit_sk_regression(train, dist)
        sk_pred = float(fit["predict_sk"](dist[held]))
        D = np.asarray(per_site[held][0], float)
        C = np.asarray(per_site[held][1], float)
        preds.append(sk_pred * D)
        obs.append(C)
        per_pred[held] = sk_pred
    pred = np.concatenate(preds)
    ob = np.concatenate(obs)
    ss_res = float(((ob - pred) ** 2).sum())
    ss_tot = float(((ob - ob.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "n_sites": len(names),
            "per_site_pred": per_pred}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_loso_discriminates_signal_from_noise -v`
Expected: PASS (positive on the gradient case, ≤0.3 on the random case — the gate discriminates)

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/subsidence.py tests/test_explorer.py
git commit -m "feat(subsidence): leave-one-site-out gate for the Sk regression"
```

---

## Task 4: Coast-distance grid + Sk-field subsidence grid

**Files:**
- Modify: `hydrophysics/explorer.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_coast_grid_and_sk_field_subsidence():
    from shapely.geometry import LineString
    from hydrophysics.explorer import coast_distance_grid, subsidence_grid_from_sk_field

    xs = np.linspace(0.0, 10.0, 5)
    ys = np.linspace(0.0, 10.0, 5)
    XX, YY = np.meshgrid(xs, ys)
    coast = LineString([(0.0, 0.0), (0.0, 10.0)])     # the y-axis -> distance == x
    DC = coast_distance_grid(XX, YY, coast)
    assert DC.shape == (5, 5)
    assert np.allclose(DC[0], xs, atol=1e-6)          # first row distances == x coords

    # subsidence grid from a constant Sk field == subsidence_grid with that scalar
    HH = np.stack([np.full((5, 5), 3.0), np.full((5, 5), 1.0)])  # (2,5,5), head drops 3->1
    sk_field = np.full((5, 5), 0.01)
    SS = subsidence_grid_from_sk_field(HH, sk_field)
    assert SS.shape == HH.shape
    assert np.allclose(SS[0], 0.0)                    # t0 drawdown is zero
    assert np.allclose(SS[1], 0.01 * 2.0)             # drawdown 2 m * Sk 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_coast_grid_and_sk_field_subsidence -v`
Expected: FAIL with `ImportError: cannot import name 'coast_distance_grid'`

- [ ] **Step 3: Append to `hydrophysics/explorer.py`:**

```python
def coast_distance_grid(XX, YY, coast_geom):
    """Distance from each grid cell to the coastline geometry -> (n, n)."""
    from shapely.geometry import Point

    flat = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    d = np.array([Point(px, py).distance(coast_geom) for px, py in flat])
    return d.reshape(XX.shape)


def subsidence_grid_from_sk_field(HH: np.ndarray, sk_field: np.ndarray) -> np.ndarray:
    """sk_field(x,y) * cumulative drawdown per cell, from the (Tm, n, n) head history."""
    from .subsidence import cumulative_drawdown

    Tm = HH.shape[0]
    flat = HH.reshape(Tm, -1).T                       # (cells, Tm)
    D = cumulative_drawdown(flat)                      # (cells, Tm)
    sk = sk_field.reshape(-1)[:, None]                 # (cells, 1)
    return (sk * D).T.reshape(HH.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_explorer.py::test_coast_grid_and_sk_field_subsidence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/explorer.py tests/test_explorer.py
git commit -m "feat(explorer): coast-distance grid + Sk-field subsidence grid"
```

---

## Task 5: `coupling` switch in `build_explorer`

**Files:**
- Modify: `hydrophysics/explorer.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_build_explorer_coast_coupling_smoke(gwdata, tmp_path):
    from shapely.geometry import LineString, box
    from hydrophysics.explorer import build_explorer

    wx = gwdata.attrs["tm_x"].astype(float)
    wy = gwdata.attrs["tm_y"].astype(float)
    poly = box(wx.min(), wy.min(), wx.max(), wy.max())
    coast = LineString([(wx.min(), wy.min()), (wx.min(), wy.max())])  # west edge
    # synthetic MLCW: 3 stations inside the bbox with fake compaction
    stations = pd.DataFrame({"sub_id": ["a", "b", "c"],
                             "x": np.linspace(wx.min(), wx.max(), 3),
                             "y": [wy.mean()] * 3})
    dates = pd.date_range("2012-01-31", periods=12, freq="ME")
    compaction = {n: pd.Series(np.linspace(0, 0.1, 12), index=dates)
                  for n in stations["sub_id"]}
    out = tmp_path / "coast.html"
    path = build_explorer(gwdata, poly, stations, compaction, out_html=str(out),
                          n=8, coupling="coast", coast_geom=coast)
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_explorer.py::test_build_explorer_coast_coupling_smoke -v`
Expected: FAIL with `TypeError` (unexpected keyword argument `coupling`)

- [ ] **Step 3: Replace the ENTIRE `build_explorer` function in `hydrophysics/explorer.py` with:**

```python
def build_explorer(data: GWData, poly, stations, compaction, out_html: str,
                   n: int = 60, coupling: str = "single", coast_geom=None) -> str:
    """Assemble the interactive HTML and write it to ``out_html``. Returns the path.

    coupling="single" (default): subsidence surface = single calibrated Sk * drawdown,
    validation panel = pooled in-sample pairs. coupling="coast": subsidence surface =
    Sk(distance_to_coast) * drawdown using the leave-one-site-out-validated regression;
    validation panel = LOSO predicted vs observed. ``coast_geom`` (shapely) is required
    for coupling="coast". ``stations``/``compaction`` may be None (sk=0, no panel).
    """
    import os

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .subsidence import (calibrate_sk, fit_sk_regression, loso_sk_regression,
                             _distances_to_geom)

    XX, YY, HH, dates = head_grid(data, poly, n=n)
    wxy = well_xy(data)
    zmark = float(np.nanmax(HH)) if np.isfinite(HH).any() else 0.0
    have_mlcw = stations is not None and compaction

    # --- subsidence surface + validation data, per coupling mode ---
    vx = vy = np.array([])           # validation scatter x (pred), y (obs)
    if not have_mlcw:
        SS = subsidence_grid(HH, 0.0)
        panel_title = "MLCW validation (no MLCW data)"
    elif coupling == "coast":
        if coast_geom is None:
            raise ValueError("coupling='coast' requires coast_geom")
        cal = calibrate_sk(data, stations, compaction)
        dist = _distances_to_geom(stations, coast_geom)
        fit = fit_sk_regression(cal["per_site"], dist)
        loso = loso_sk_regression(cal["per_site"], dist)
        sk_field = fit["predict_sk"](coast_distance_grid(XX, YY, coast_geom))
        SS = subsidence_grid_from_sk_field(HH, sk_field)
        # LOSO predicted vs observed (per-site final compaction)
        for name, (D, C) in cal["per_site"].items():
            vx = np.append(vx, loso["per_site_pred"][name] * np.asarray(D)[-1])
            vy = np.append(vy, np.asarray(C)[-1])
        panel_title = f"MLCW LOSO validation (R²={loso['r2']:.2f}, β1={fit['b1']:.3g})"
    else:
        cal = calibrate_sk(data, stations, compaction)
        SS = subsidence_grid(HH, cal["sk"])
        vx, vy = cal["sk"] * cal["D"], cal["C"]
        panel_title = f"MLCW validation (single Sk={cal['sk']:.4g}, R²={cal['r2']:.2f})"

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=("Groundwater head (m) — IDW of observed wells", panel_title),
    )
    fig.add_trace(go.Surface(x=XX, y=YY, z=HH[0], colorscale="Viridis",
                             colorbar=dict(title="head m", x=0.62)), row=1, col=1)
    fig.add_trace(go.Surface(x=XX, y=YY, z=SS[0], colorscale="Reds", visible=False,
                             showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=wxy[:, 0], y=wxy[:, 1], z=zmark * np.ones(len(wxy)),
                               mode="markers", marker=dict(size=2, color="white"),
                               name="wells"), row=1, col=1)
    if stations is not None:
        sx = stations["x"].to_numpy(dtype=float)
        sy = stations["y"].to_numpy(dtype=float)
        fig.add_trace(go.Scatter3d(x=sx, y=sy, z=zmark * np.ones(len(sx)), mode="markers",
                                   marker=dict(size=4, color="red", symbol="diamond"),
                                   name="MLCW sites"), row=1, col=1)
    if vx.size:
        fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers",
                                 marker=dict(size=6, color="firebrick"),
                                 name="MLCW sites"), row=1, col=2)
        lim = float(max(vy.max(), vx.max(), 1e-6))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="1:1"),
                      row=1, col=2)
        fig.update_xaxes(title_text="predicted subsidence (m)", row=1, col=2)
        fig.update_yaxes(title_text="observed compaction (m)", row=1, col=2)

    frames = [go.Frame(name=str(dt.date()),
                       data=[go.Surface(x=XX, y=YY, z=HH[t]),
                             go.Surface(x=XX, y=YY, z=SS[t])], traces=[0, 1])
              for t, dt in enumerate(dates)]
    fig.frames = frames
    steps = [dict(method="animate", label=str(dt.date()),
                  args=[[str(dt.date())], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for dt in dates]
    fig.update_layout(
        title="Choushui groundwater head + land subsidence (observed, 2010–2022)",
        sliders=[dict(active=0, steps=steps, x=0.05, len=0.6,
                      currentvalue=dict(prefix="month: "))],
        updatemenus=[dict(type="buttons", direction="right", x=0.05, y=1.12, buttons=[
            dict(label="Head", method="restyle", args=[{"visible": [True, False]}, [0, 1]]),
            dict(label="Subsidence", method="restyle",
                 args=[{"visible": [False, True]}, [0, 1]]),
        ])],
        scene=dict(xaxis_title="TM_X97 (m)", yaxis_title="TM_Y97 (m)",
                   zaxis_title="head (m) / subsidence (m)", aspectmode="auto"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", auto_play=False)
    return out_html
```

- [ ] **Step 4: Run the whole file to confirm nothing regressed**

Run: `pytest tests/test_explorer.py -v`
Expected: all pass (the existing `test_build_explorer_writes_html` single-mode test still passes, plus the new coast smoke).

- [ ] **Step 5: Lint + commit**

Run: `ruff check hydrophysics/explorer.py hydrophysics/subsidence.py tests/test_explorer.py`
Then:
```bash
git add hydrophysics/explorer.py tests/test_explorer.py
git commit -m "feat(explorer): coast-coupling mode (Sk(distance) surface + LOSO panel)"
```

---

## Task 6: Verdict report + conditional real-data wiring + docs

**Files:**
- Create: `hydrophysics/subsidence_report.py`
- Modify: `README.md`

- [ ] **Step 1: Create `hydrophysics/subsidence_report.py`:**

```python
"""Print the head->subsidence coupling verdict: the four R² numbers + regression beta.

This is the honest gate. Run on the real data:
    python -m hydrophysics.subsidence_report
"""

from __future__ import annotations


def main(argv=None) -> None:
    import argparse

    import numpy as np

    from .config import Config, default_config
    from .data import load_dataset
    from .subsidence import (calibrate_sk, fit_sk_regression, load_mlcw_stations,
                             loso_sk_regression, mlcw_compaction, site_distance_to_coast)

    ap = argparse.ArgumentParser(description="Head->subsidence coupling verdict")
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)

    cfg = Config(data_dir=args.data) if args.data else default_config()
    data = load_dataset(cfg)
    stations = load_mlcw_stations(cfg.data_dir / "mlcw_stations.csv")
    compaction = mlcw_compaction(str(cfg.data_dir))
    dist = site_distance_to_coast(stations, cfg.data_dir / "water" / "sea_TWD97.shp")
    cal = calibrate_sk(data, stations, compaction)

    # per-site own-Sk in-sample (upper bound)
    sk_i = {n: float(np.asarray(D) @ np.asarray(C) / (np.asarray(D) @ np.asarray(D) + 1e-12))
            for n, (D, C) in cal["per_site"].items()}
    obs = np.concatenate([np.asarray(C) for _, C in cal["per_site"].values()])
    res_ins = sum(float(((np.asarray(C) - sk_i[n] * np.asarray(D)) ** 2).sum())
                  for n, (D, C) in cal["per_site"].items())
    r2_insample = 1.0 - res_ins / max(((obs - obs.mean()) ** 2).sum(), 1e-12)

    fit = fit_sk_regression(cal["per_site"], dist)
    loso = loso_sk_regression(cal["per_site"], dist)

    print("=== Head -> subsidence coupling verdict (2019+ MLCW pairs) ===")
    print(f"  single basin-wide Sk        R² = {cal['r2']:+.3f}")
    print(f"  per-site own Sk (in-sample) R² = {r2_insample:+.3f}   [upper bound]")
    print(f"  coast regression  (in-sample) R² = {fit['r2_insample']:+.3f}")
    print(f"  coast regression  LOSO      R² = {loso['r2']:+.3f}   [THE GATE; pass if > 0]")
    print(f"  beta: log Sk = {fit['b0']:.3f} + ({fit['b1']:.3g}) * distance_to_coast_m")
    verdict = "PASS -> ship coast surface" if loso["r2"] > 0 else "FAIL -> keep negative result"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 2: Run the verdict on real data**

```bash
HYDROMIND_GW_DATA="$PWD/chou-shui-data/chou-shui-data/data" python -m hydrophysics.subsidence_report 2>/dev/null
```
Record the printed table verbatim — especially the **LOSO R²** line. This is the gate.

- [ ] **Step 3: Branch on the verdict**

**If LOSO R² > 0 (PASS):** regenerate the explorer with the coast coupling and confirm the panel shows the LOSO R²:
```bash
HYDROMIND_GW_DATA="$PWD/chou-shui-data/chou-shui-data/data" python -c "
from hydrophysics.config import default_config
from hydrophysics.data import load_dataset
from hydrophysics.explorer import build_explorer, fan_polygon
from hydrophysics.subsidence import load_mlcw_stations, mlcw_compaction
import geopandas as gpd
cfg=default_config(); d=load_dataset(cfg)
poly=fan_polygon(str(cfg.data_dir/'Zhuoshui Alluvial Fan'/'Zhuoshui Alluvial Fan.shp'))
coast=gpd.read_file(cfg.data_dir/'water'/'sea_TWD97.shp').geometry.union_all()
st=load_mlcw_stations(cfg.data_dir/'mlcw_stations.csv'); cmp=mlcw_compaction(str(cfg.data_dir))
print(build_explorer(d, poly, st, cmp, 'results/explorer/choushui_explorer.html', n=50, coupling='coast', coast_geom=coast))
"
```

**If LOSO R² ≤ 0 (FAIL):** do NOT regenerate with coast coupling. Leave the single-mode explorer as is.

- [ ] **Step 4: Update the README explorer section** (`README.md`, the "Interactive explorer: head + land subsidence" section). Replace the paragraph beginning "A validation panel plots predicted vs observed compaction..." with one of:

**If PASS** (fill `<R2>`, `<b1>` from Step 2):
```markdown
A validation panel plots predicted vs observed compaction. The single basin-wide `Sk`
fails (R² = −0.28), so subsidence is modelled per-site as `Sk = exp(β₀ + β₁·distance_to_coast)`
(coastal marine clay compacts more per metre of drawdown). This **passes a leave-one-site-out
gate (R² = <R2>)**, so the explorer's subsidence surface is the coast-calibrated field
(`--coupling coast`), validated on held-out sites. Verdict: `python -m hydrophysics.subsidence_report`.
```

**If FAIL** (fill `<R2>`):
```markdown
A validation panel plots predicted vs observed compaction. Two couplings were tested and
**both fail the honest gate**: a single basin-wide `Sk` (R² = −0.28) and a per-site
`Sk = exp(β₀+β₁·distance_to_coast)` regression (leave-one-site-out R² = <R2>, still ≤ 0).
The distance-to-coast signal is real (corr = −0.68) but does not generalize across only 14
scattered sites. The subsidence surface stays an illustrative single-`Sk` proxy; the
observed-head animation and the real MLCW data are the trustworthy parts. Verdict:
`python -m hydrophysics.subsidence_report`.
```

- [ ] **Step 5: Run the full suite + lint + commit**

Run: `pytest -q && ruff check hydrophysics/subsidence_report.py`
Then:
```bash
git add hydrophysics/subsidence_report.py README.md
git commit -m "feat(subsidence): coupling verdict report + honest README update"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** distance-to-coast predictor (Task 1); weighted log-linear `Sk(dc)=exp(β₀+β₁dc)` model (Task 2); leave-one-site-out gate (Task 3); grid-evaluable coast distance + Sk-field surface (Task 4); conditional coast coupling in the explorer with LOSO panel (Task 5); verdict report + conditional shipping + honest README both-ways (Task 6); leakage-safety (predictor path takes only `dist`+pairs, asserted by the synthetic tests where the predictor is a pure function of distance); risks (n=14, gate may fail — handled by the explicit PASS/FAIL branch). All spec sections map to a task.
- **Placeholder scan:** none. The README `<R2>`/`<b1>` are explicitly filled from Step 2 output (documented action). The PASS/FAIL branch is real conditional logic, not a deferral.
- **Type consistency:** `per_site` is `{name:(D,C)}` everywhere (matches `calibrate_sk`'s return). `dist` is `{name:float}` from `_distances_to_geom`/`site_distance_to_coast`, consumed identically by `fit_sk_regression`, `loso_sk_regression`, and `build_explorer`. `fit_sk_regression` returns `predict_sk` (callable of distance) used in Task 5 and Task 6. `loso_sk_regression` returns `{r2,n_sites,per_site_pred}` used in Task 5's panel and Task 6's gate. `coast_distance_grid(XX,YY,coast_geom)` and `subsidence_grid_from_sk_field(HH,sk_field)` signatures match Task 4 defs and Task 5 calls. `build_explorer(..., coupling, coast_geom)` matches its test and the Task 6 invocation.
