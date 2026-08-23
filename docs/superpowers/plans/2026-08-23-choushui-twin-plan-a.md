# Choushui Twin — Plan A (Stages 0–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leveling-based subsidence panel, run the decisive `Sk` refit that determines whether the published spatial-sparsity explanation holds, and deliver a differentiable visco-elasto-plastic compaction column that beats the algebraic `Sk` at leave-one-site-out.

**Architecture:** Three layers, each independently testable. `twin/leveling.py` turns the cached benchmark panel into per-site cumulative subsidence with an optional planar tectonic correction. `twin/sk_leveling.py` pairs those sites with IDW-interpolated head drawdown and reuses the existing `loso_sk_regression` gate — this is Stage 1, and it is a genuine kill/continue decision. `twin/compaction.py` is a PyTorch visco-elasto-plastic column (elastic + preconsolidation-gated inelastic + viscous relaxation) built up one physical term at a time, each with an analytic limit test, then calibrated against the depth-resolved MLCW sites.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch 2.11+cu128 (CUDA available, RTX 4070 SUPER 12 GB), pytest, ruff (line-length 100, rules E4/E7/E9/F/I/B/UP/SIM). Package managed with `uv pip` — the venv has no pip.

**Spec:** `docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md`

## Global Constraints

- **The 2019+ window and any held-out site are sealed.** Model selection uses inner splits only. Never tune against a leave-one-site-out or 2019+ number. (Spec §8 guardrail 1, §9.)
- **All physical parameters are log-parameterized** to enforce positivity: `T`, `S`, `L`, `S_ke`, `S_kv`, `τ`, `h_pc`. (Spec §3.)
- **Positive compaction means subsidence** throughout — matches `subsidence.mlcw_compaction`, which returns `(base - separation)`.
- **Units:** lengths in metres, coordinates EPSG:3826 metres, time in months for Stage 1 and days for the compaction ODE, heads in metres.
- **No RL, no extra DL module, no full-3D Biot** in this plan. (Spec, Scope decisions.)
- **Data lives under `data/`-named directories and is gitignored.** Never commit agency data.
- Real data dir: `chou-shui-data/chou-shui-data/data`, exported as `HYDROMIND_GW_DATA`.
- Run tests with `.venv/bin/python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `hydrophysics/twin/__init__.py` | Package marker; re-exports nothing (keeps import graph explicit) |
| `hydrophysics/twin/leveling.py` | Benchmark panel → per-site cumulative subsidence; planar tectonic correction |
| `hydrophysics/twin/sk_leveling.py` | Pair leveling sites with IDW head drawdown; Stage-1 gate runner + CLI |
| `hydrophysics/twin/compaction.py` | Differentiable VEP column (torch): elastic, inelastic, viscous |
| `hydrophysics/twin/calibrate_mlcw.py` | Fit the VEP column to MLCW sites; Stage-2 LOSO gate + CLI |
| `tests/test_twin_leveling.py` | Panel loading, re-zeroing, tectonic modes |
| `tests/test_twin_sk_leveling.py` | Pairing logic and gate plumbing on synthetic data |
| `tests/test_twin_compaction.py` | Analytic limits of each physical term; gradient flow |

Existing code reused unchanged: `hydrophysics.subsidence.{idw_interp, monthly_heads, well_xy, mlcw_compaction, loso_sk_regression, calibrate_sk_from_pairs}` and `hydrophysics.data.GWData`.

---

### Task 1: Leveling panel → per-site cumulative subsidence

**Files:**
- Create: `hydrophysics/twin/__init__.py`
- Create: `hydrophysics/twin/leveling.py`
- Test: `tests/test_twin_leveling.py`

**Interfaces:**
- Consumes: the cached parquet `chou-shui-data/chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet` with columns `datetime, elev_m, sid, x_3826, y_3826, lon, lat, town, name`.
- Produces:
  - `load_panel(data_dir: str) -> pd.DataFrame` — columns `sid, datetime, elev_m, x, y`
  - `site_subsidence(panel: pd.DataFrame, t0: str, t1: str, min_obs: int = 5) -> dict[str, pd.Series]` — `{sid -> Series(index=datetime, values=cumulative subsidence in metres, positive = sinking, re-zeroed to the first observation in window)}`
  - `site_xy(panel: pd.DataFrame) -> dict[str, tuple[float, float]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twin_leveling.py
import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin import leveling


def _panel():
    """Two sites: A sinks 1 cm/yr over 6 annual surveys, B is stable.

    Columns mirror ``load_panel``'s output schema (x/y, already renamed), which is what
    ``site_subsidence`` and ``site_xy`` consume.
    """
    rows = []
    for i, yr in enumerate(range(2012, 2018)):
        rows.append({"sid": "A", "datetime": pd.Timestamp(f"{yr}-06-15"),
                     "elev_m": 10.0 - 0.01 * i, "x": 180000.0, "y": 2620000.0})
        rows.append({"sid": "B", "datetime": pd.Timestamp(f"{yr}-06-15"),
                     "elev_m": 5.0, "x": 190000.0, "y": 2630000.0})
    return pd.DataFrame(rows)


def test_site_subsidence_is_positive_and_rezeroed():
    sub = leveling.site_subsidence(_panel(), "2012-01-01", "2018-01-01", min_obs=5)
    assert set(sub) == {"A", "B"}
    a = sub["A"]
    assert a.iloc[0] == pytest.approx(0.0)          # re-zeroed to first observation
    assert a.iloc[-1] == pytest.approx(0.05)        # 5 cm of sinking, positive
    assert np.allclose(sub["B"].to_numpy(), 0.0)    # stable site stays flat


def test_min_obs_filter_drops_short_records():
    p = _panel()
    p = p[~((p.sid == "B") & (p.datetime > pd.Timestamp("2013-01-01")))]
    sub = leveling.site_subsidence(p, "2012-01-01", "2018-01-01", min_obs=5)
    assert set(sub) == {"A"}


def test_site_xy_returns_epsg3826_metres():
    xy = leveling.site_xy(_panel())
    assert xy["A"] == (180000.0, 2620000.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_leveling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/__init__.py
"""Differentiable Choushui digital twin. See
docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md."""
```

```python
# hydrophysics/twin/leveling.py
"""Leveling benchmarks -> per-site cumulative subsidence.

The WRA leveling network (`ls-wra-lsp-obs`) surveys benchmark orthometric elevation roughly
annually. Cumulative subsidence at a site is its elevation drop relative to the first survey
inside the analysis window, so positive values mean sinking -- the same sign convention as
``subsidence.mlcw_compaction``.
"""

from __future__ import annotations

import os

import pandas as pd

PANEL = os.path.join("ls_cache", "ls-wra-lsp-obs__choushui_panel.parquet")


def load_panel(data_dir: str) -> pd.DataFrame:
    """Read the cached benchmark panel -> DataFrame[sid, datetime, elev_m, x, y]."""
    df = pd.read_parquet(os.path.join(data_dir, PANEL))
    out = df[["sid", "datetime", "elev_m", "x_3826", "y_3826"]].copy()
    out = out.rename(columns={"x_3826": "x", "y_3826": "y"})
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values(["sid", "datetime"]).reset_index(drop=True)


def site_subsidence(panel: pd.DataFrame, t0: str, t1: str,
                    min_obs: int = 5) -> dict[str, pd.Series]:
    """{sid -> cumulative subsidence (m, positive = sinking) re-zeroed to the first survey}.

    Sites with fewer than ``min_obs`` surveys inside [t0, t1) are dropped.
    """
    w = panel[(panel.datetime >= pd.Timestamp(t0)) & (panel.datetime < pd.Timestamp(t1))]
    out: dict[str, pd.Series] = {}
    for sid, g in w.groupby("sid"):
        g = g.sort_values("datetime")
        if len(g) < min_obs:
            continue
        s = pd.Series(g.elev_m.to_numpy(), index=pd.DatetimeIndex(g.datetime))
        out[sid] = (s.iloc[0] - s).rename("subsidence_m")
    return out


def site_xy(panel: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """{sid -> (x, y)} in EPSG:3826 metres, taken from the first row of each site."""
    first = panel.groupby("sid").first()
    return {sid: (float(r.x), float(r.y)) for sid, r in first.iterrows()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_leveling.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/__init__.py hydrophysics/twin/leveling.py tests/test_twin_leveling.py
git commit -m "feat(twin): leveling panel -> per-site cumulative subsidence"
```

---

### Task 2: Planar tectonic correction

**Files:**
- Modify: `hydrophysics/twin/leveling.py`
- Test: `tests/test_twin_leveling.py`

**Interfaces:**
- Consumes: `site_subsidence` output and `site_xy` from Task 1.
- Produces: `remove_tectonic(sub, xy, mode) -> tuple[dict[str, pd.Series], dict]` where `mode` is `"none"` or `"planar"`. Returns the corrected series plus `{"a": float, "b": float, "c": float, "var_removed": float}` — the fitted velocity plane `v(x,y) = a + b·x + c·y` in m/yr and the fraction of total variance it absorbed.

**Why a plane and not a per-site trend.** Ching et al. 2011 (`10.1029/2011jb008242`) show Taiwan's tectonic vertical field is spatially smooth. Subsidence itself is a *trend*, so removing a per-site linear trend would delete the very signal being modelled. A single global plane has three parameters across ~556 sites, so it can absorb a regional tilt without absorbing site-specific compaction. **The Stage-1 gate must be reported both with and without this correction** (Spec §6, §7).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_leveling.py
def test_planar_tectonic_removes_a_regional_tilt_but_not_local_signal():
    rng = np.random.default_rng(0)
    xy, sub = {}, {}
    times = pd.DatetimeIndex([pd.Timestamp(f"{y}-06-15") for y in range(2012, 2020)])
    yrs = np.array([(t - times[0]).days / 365.25 for t in times])
    for i in range(40):
        x = 170000.0 + 1000.0 * i
        y = 2620000.0 + 500.0 * i
        xy[f"S{i}"] = (x, y)
        tect = 0.002 + 1e-8 * (x - 170000.0)              # regional tilt, m/yr
        local = 0.01 if i % 2 == 0 else 0.0               # site-specific compaction
        sub[f"S{i}"] = pd.Series((tect + local) * yrs, index=times)

    out, info = leveling.remove_tectonic(sub, xy, mode="planar")
    even = np.mean([out[f"S{i}"].iloc[-1] for i in range(0, 40, 2)])
    odd = np.mean([out[f"S{i}"].iloc[-1] for i in range(1, 40, 2)])
    assert even - odd == pytest.approx(0.01 * yrs[-1], rel=0.05)   # local signal survives
    assert abs(odd) < 0.2 * abs(even)                              # regional tilt removed
    assert 0.0 <= info["var_removed"] <= 1.0


def test_tectonic_mode_none_is_identity():
    p = _panel()
    sub = leveling.site_subsidence(p, "2012-01-01", "2018-01-01", min_obs=5)
    out, info = leveling.remove_tectonic(sub, leveling.site_xy(p), mode="none")
    assert np.allclose(out["A"].to_numpy(), sub["A"].to_numpy())
    assert info["var_removed"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_leveling.py -k tectonic -v`
Expected: FAIL with `AttributeError: module 'hydrophysics.twin.leveling' has no attribute 'remove_tectonic'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to hydrophysics/twin/leveling.py
import numpy as np


def _site_rate(s: pd.Series) -> float:
    """Least-squares subsidence rate (m/yr) of one site's cumulative series."""
    t = np.array([(i - s.index[0]).days / 365.25 for i in s.index], dtype="float64")
    if t.size < 2 or t.std() == 0:
        return 0.0
    return float(np.polyfit(t, s.to_numpy(dtype="float64"), 1)[0])


def remove_tectonic(sub: dict[str, pd.Series], xy: dict[str, tuple[float, float]],
                    mode: str = "planar") -> tuple[dict[str, pd.Series], dict]:
    """Subtract a global planar vertical-velocity field v(x,y) = a + b*x + c*y.

    Taiwan's tectonic vertical field is spatially smooth (Ching et al. 2011), so three
    global parameters over hundreds of sites absorb a regional tilt without absorbing
    site-specific compaction. ``mode="none"`` returns the input unchanged.
    """
    if mode == "none":
        return dict(sub), {"a": 0.0, "b": 0.0, "c": 0.0, "var_removed": 0.0}
    if mode != "planar":
        raise ValueError(f"unknown tectonic mode: {mode!r}")

    names = [n for n in sub if n in xy]
    rates = np.array([_site_rate(sub[n]) for n in names], dtype="float64")
    X = np.array([[1.0, xy[n][0], xy[n][1]] for n in names], dtype="float64")
    Xc = X.copy()
    Xc[:, 1] -= Xc[:, 1].mean()          # centre the coordinates for conditioning
    Xc[:, 2] -= Xc[:, 2].mean()
    beta, *_ = np.linalg.lstsq(Xc, rates, rcond=None)
    fitted = Xc @ beta
    denom = float(((rates - rates.mean()) ** 2).sum())
    var_removed = float((fitted ** 2).sum() / denom) if denom > 0 else 0.0

    out: dict[str, pd.Series] = {}
    for n, v in zip(names, fitted, strict=True):
        s = sub[n]
        t = np.array([(i - s.index[0]).days / 365.25 for i in s.index], dtype="float64")
        out[n] = s - v * t
    return out, {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]),
                 "var_removed": min(max(var_removed, 0.0), 1.0)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_leveling.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/leveling.py tests/test_twin_leveling.py
git commit -m "feat(twin): planar tectonic correction for leveling series"
```

---

### Task 3: Stage-1 gate — refit `Sk` on the leveling network

**This is the decisive experiment.** It settles whether the published leave-one-site-out failure was caused by having only 14 sites (spatial sparsity) or by the model form. Its result determines how much of Stage 2 is justified.

**Files:**
- Create: `hydrophysics/twin/sk_leveling.py`
- Test: `tests/test_twin_sk_leveling.py`

**Interfaces:**
- Consumes: `leveling.load_panel`, `leveling.site_subsidence`, `leveling.site_xy`, `leveling.remove_tectonic` (Tasks 1–2); `subsidence.{monthly_heads, well_xy, idw_interp, loso_sk_regression, calibrate_sk_from_pairs}`; `data.GWData`.
- Produces:
  - `build_pairs(data, sub, xy) -> dict[str, tuple[np.ndarray, np.ndarray]]` — `{sid -> (D, C)}` with `D` cumulative head drawdown and `C` cumulative subsidence, both sampled at the site's survey dates and re-zeroed to the first.
  - `main(argv=None) -> None` — CLI writing `results/twin/stage1_sk_leveling.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twin_sk_leveling.py
import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin import sk_leveling


class _FakeData:
    """Minimal GWData stand-in: two wells, monthly heads declining linearly."""

    def __init__(self):
        self.dates = pd.date_range("2012-01-31", periods=96, freq="ME")
        decline = np.linspace(0.0, -12.0, 96)
        self.target = np.stack([decline, decline])                       # (W, T)
        self.attrs = pd.DataFrame({"tm_x": [175000.0, 185000.0],
                                   "tm_y": [2615000.0, 2625000.0]})


def test_build_pairs_aligns_drawdown_and_subsidence_at_survey_dates():
    data = _FakeData()
    times = pd.DatetimeIndex([pd.Timestamp(f"{y}-06-30") for y in range(2013, 2019)])
    sub = {"S1": pd.Series(np.linspace(0.0, 0.06, len(times)), index=times)}
    xy = {"S1": (180000.0, 2620000.0)}

    pairs = sk_leveling.build_pairs(data, sub, xy)
    D, C = pairs["S1"]
    assert D.shape == C.shape == (len(times),)
    assert D[0] == pytest.approx(0.0)          # drawdown re-zeroed to first survey
    assert C[0] == pytest.approx(0.0)
    assert np.all(np.diff(D) >= -1e-9)         # heads fall -> drawdown accumulates
    assert D[-1] > 0.0


def test_build_pairs_skips_sites_with_no_overlapping_surveys():
    data = _FakeData()
    times = pd.DatetimeIndex([pd.Timestamp("2001-06-30"), pd.Timestamp("2002-06-30")])
    sub = {"OLD": pd.Series([0.0, 0.01], index=times)}
    pairs = sk_leveling.build_pairs(data, sub, {"OLD": (180000.0, 2620000.0)})
    assert pairs == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_sk_leveling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.sk_leveling'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/sk_leveling.py
"""Stage 1: refit the algebraic Sk coupling on the leveling network instead of 14 MLCW sites.

The README reports leave-one-site-out R2 of -0.28 to -2.40 for every head->subsidence
coupling tried on 14 multi-layer compaction wells, and attributes the failure to spatial
sparsity. The WRA leveling network provides ~40x more sites over the same window. If the
gate is still negative here, the wall is the model form, not the sampling -- which is what
justifies the visco-elasto-plastic column in Tasks 4-7.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import GWData, load_dataset
from ..subsidence import (
    calibrate_sk_from_pairs,
    idw_interp,
    loso_sk_regression,
    monthly_heads,
    well_xy,
)
from . import leveling


def build_pairs(data: GWData, sub: dict[str, pd.Series],
                xy: dict[str, tuple[float, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{sid -> (cumulative drawdown, cumulative subsidence)} sampled at survey dates.

    Heads are interpolated to each site by IDW from the observed wells, aggregated to
    month-end, then read at the nearest month to each survey. Both series are re-zeroed to
    the site's first survey inside the head record.
    """
    H, dates = monthly_heads(data)
    wxy = well_xy(data)
    lo, hi = dates[0], dates[-1]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid, s in sub.items():
        if sid not in xy:
            continue
        keep = s.index[(s.index >= lo) & (s.index <= hi)]
        if len(keep) < 3:
            continue
        h = idw_interp(np.array([list(xy[sid])], dtype="float64"), wxy, H)[0]
        hser = pd.Series(h, index=dates)
        draw = (hser.iloc[0] - hser.cummin()).clip(lower=0.0)   # inelastic cumulative drawdown
        at = draw.reindex(keep, method="nearest").to_numpy(dtype="float64")
        C = s.loc[keep].to_numpy(dtype="float64")
        out[sid] = (at - at[0], C - C[0])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage-1 Sk gate on the leveling network")
    ap.add_argument("--data", default=None, help="data dir (else HYDROMIND_GW_DATA)")
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--t0", default="2012-01-01")
    ap.add_argument("--t1", default="2023-01-01")
    ap.add_argument("--out", default="results/twin")
    args = ap.parse_args(argv)

    cfg = Config(data_dir=Path(args.data)) if args.data else Config()
    data = load_dataset(cfg)
    ddir = str(cfg.data_dir)

    panel = leveling.load_panel(ddir)
    sub_raw = leveling.site_subsidence(panel, args.t0, args.t1, min_obs=args.min_obs)
    xy = leveling.site_xy(panel)
    coast = {sid: float(xy[sid][0]) for sid in xy}     # proxy: easting increases inland

    rows = []
    for mode in ("none", "planar"):
        sub, info = leveling.remove_tectonic(sub_raw, xy, mode=mode)
        pairs = build_pairs(data, sub, xy)
        if not pairs:
            continue
        single = calibrate_sk_from_pairs(pairs)
        gate = loso_sk_regression(pairs, {k: coast[k] for k in pairs})
        rows.append({"tectonic": mode, "n_sites": len(pairs),
                     "var_removed": info["var_removed"], "sk_single": single["sk"],
                     "r2_insample": single["r2"], "loso_r2": gate["r2"],
                     "loso_r2_sk": gate["r2_sk"]})
        print(f"[{mode:6s}] sites={len(pairs):4d}  var_removed={info['var_removed']:.3f}  "
              f"single-Sk in-sample R2={single['r2']:+.3f}  "
              f"LOSO compaction R2={gate['r2']:+.3f}  LOSO Sk R2={gate['r2_sk']:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage1_sk_leveling.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path}")
    print("GATE: LOSO compaction R2 > 0 means spatial sparsity was the wall; "
          "still negative means the model form is, and Tasks 4-7 are justified.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_sk_leveling.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the real gate and record the number**

```bash
export HYDROMIND_GW_DATA="$(pwd)/chou-shui-data/chou-shui-data/data"
.venv/bin/python -m hydrophysics.twin.sk_leveling --min-obs 5
```

Expected: a table for `tectonic=none` and `tectonic=planar` with `n_sites` in the 400–600 range. **Record both `loso_r2` values in the commit message** — this is the Stage-1 gate and the number decides how the rest of the plan is read.

- [ ] **Step 6: Commit**

```bash
git add hydrophysics/twin/sk_leveling.py tests/test_twin_sk_leveling.py
git commit -m "feat(twin): Stage-1 Sk gate on the leveling network

LOSO compaction R2: none=<value>, planar=<value>, n_sites=<n>."
```

---

### Task 4: Elastic compaction column (torch)

**Files:**
- Create: `hydrophysics/twin/compaction.py`
- Test: `tests/test_twin_compaction.py`

**Interfaces:**
- Produces: `class VEPColumn(torch.nn.Module)` with `__init__(self, n_sites: int, dt_days: float = 30.0, device=None)` and `forward(self, h: torch.Tensor) -> torch.Tensor`. `h` has shape `(n_sites, T)` in metres; the return has the same shape and is cumulative compaction in metres, positive for subsidence, re-zeroed to `t=0`. Learnable log-parameters `log_ske`, `log_skv`, `log_tau`, `h_pc0`, each shape `(n_sites,)`.
- In this task only `log_ske` is active; `log_skv`, `log_tau`, `h_pc0` are declared but unused so later tasks do not change the constructor signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twin_compaction.py
import pytest
import torch

from hydrophysics.twin.compaction import VEPColumn


def test_elastic_compaction_is_fully_recoverable():
    """Pure elastic loading then unloading must return to zero compaction."""
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_ske.fill_(torch.log(torch.tensor(1e-3)).item())
        col.log_skv.fill_(-30.0)     # inelastic off
        col.log_tau.fill_(-10.0)     # instantaneous
        col.h_pc0.fill_(-1e3)        # preconsolidation far below -> never gated on
    h = torch.tensor([[0.0, -5.0, -10.0, -5.0, 0.0]])
    s = col(h)
    assert s[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert s[0, 2].item() > 0.0                       # head fell -> subsidence
    assert s[0, 4].item() == pytest.approx(0.0, abs=1e-6)   # head recovered -> recovered


def test_elastic_magnitude_follows_ske():
    col = VEPColumn(n_sites=2)
    with torch.no_grad():
        col.log_ske[0] = torch.log(torch.tensor(1e-3))
        col.log_ske[1] = torch.log(torch.tensor(2e-3))
        col.log_skv.fill_(-30.0)
        col.log_tau.fill_(-10.0)
        col.h_pc0.fill_(-1e3)
    h = torch.tensor([[0.0, -10.0], [0.0, -10.0]])
    s = col(h)
    assert s[1, 1].item() == pytest.approx(2.0 * s[0, 1].item(), rel=1e-5)


def test_gradients_flow_to_parameters():
    col = VEPColumn(n_sites=1)
    h = torch.tensor([[0.0, -3.0, -6.0]])
    col(h).sum().backward()
    assert col.log_ske.grad is not None
    assert torch.isfinite(col.log_ske.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.compaction'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/compaction.py
"""Differentiable visco-elasto-plastic compaction column.

Tsai & Hsu 2018 (10.1016/j.enggeo.2018.07.025) show deformation on the Choushui fan is
visco-elasto-plastic: elastic, plastic *and* viscous with a delay. Lees et al. 2022
(10.1029/2021WR031390) find residual clay time constants of decades. The algebraic
``S = Sk * cumulative_drawdown`` in ``hydrophysics/subsidence.py`` has none of that memory,
which is why it fits in-sample and fails out-of-sample.

This module builds the rheology one term at a time so each has its own analytic test:
elastic (Task 4), preconsolidation-gated inelastic (Task 5), viscous relaxation (Task 6).
All parameters are log-parameterized to keep them positive.
"""

from __future__ import annotations

import torch
from torch import nn


class VEPColumn(nn.Module):
    """Visco-elasto-plastic compaction driven by head, one column per site.

    ``forward(h)`` maps heads ``(n_sites, T)`` in metres to cumulative compaction
    ``(n_sites, T)`` in metres, positive for subsidence and re-zeroed to ``t=0``.
    """

    def __init__(self, n_sites: int, dt_days: float = 30.0, device=None):
        super().__init__()
        self.n_sites = int(n_sites)
        self.dt_days = float(dt_days)
        z = torch.zeros(self.n_sites, device=device)
        self.log_ske = nn.Parameter(z.clone() + torch.log(torch.tensor(1e-3)))
        self.log_skv = nn.Parameter(z.clone() + torch.log(torch.tensor(2e-2)))
        self.log_tau = nn.Parameter(z.clone() + torch.log(torch.tensor(365.0)))
        self.h_pc0 = nn.Parameter(z.clone())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ske = torch.exp(self.log_ske).unsqueeze(1)          # (n, 1)
        drop = (h[:, :1] - h)                                # (n, T), >0 when head falls
        return ske * drop
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/compaction.py tests/test_twin_compaction.py
git commit -m "feat(twin): elastic term of the differentiable compaction column"
```

---

### Task 5: Preconsolidation-gated inelastic term

**Files:**
- Modify: `hydrophysics/twin/compaction.py`
- Test: `tests/test_twin_compaction.py`

**Interfaces:**
- Consumes: `VEPColumn` from Task 4 (same constructor signature).
- Produces: `forward` now accumulates permanent strain whenever head falls below the running preconsolidation head, which itself tracks the running minimum starting from `h_pc0`. `S_kv` (from `log_skv`) is typically 10–100× `S_ke`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_compaction.py
def test_inelastic_strain_is_never_recovered():
    """Below the preconsolidation head, compaction is permanent."""
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_ske.fill_(torch.log(torch.tensor(1e-4)).item())
        col.log_skv.fill_(torch.log(torch.tensor(1e-2)).item())
        col.log_tau.fill_(-10.0)      # instantaneous: isolate the gate
        col.h_pc0.fill_(0.0)          # preconsolidated at h = 0
    h = torch.tensor([[0.0, -10.0, 0.0]])
    s = col(h)
    assert s[0, 1].item() > 0.0
    assert s[0, 2].item() > 0.5 * s[0, 1].item()   # most of it stays after recovery


def test_no_inelastic_above_preconsolidation_head():
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_ske.fill_(torch.log(torch.tensor(1e-4)).item())
        col.log_skv.fill_(torch.log(torch.tensor(1e-2)).item())
        col.log_tau.fill_(-10.0)
        col.h_pc0.fill_(-20.0)        # already preconsolidated well below the loading
    h = torch.tensor([[0.0, -10.0, 0.0]])
    s = col(h)
    assert s[0, 2].item() == pytest.approx(0.0, abs=1e-6)   # purely elastic, recovers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -k inelastic -v`
Expected: FAIL — `test_inelastic_strain_is_never_recovered` asserts a permanent residual the elastic-only model does not produce.

- [ ] **Step 3: Write minimal implementation**

```python
# replace VEPColumn.forward in hydrophysics/twin/compaction.py
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ske = torch.exp(self.log_ske)                       # (n,)
        skv = torch.exp(self.log_skv)
        n, T = h.shape
        h_pc = torch.minimum(self.h_pc0, h[:, 0])           # (n,) running preconsolidation
        eps_i = torch.zeros(n, dtype=h.dtype, device=h.device)
        out = [torch.zeros(n, dtype=h.dtype, device=h.device)]
        for t in range(1, T):
            below = torch.clamp(h_pc - h[:, t], min=0.0)    # how far below preconsolidation
            eps_i = eps_i + skv * below
            h_pc = torch.minimum(h_pc, h[:, t])
            eps_e = ske * (h[:, 0] - h[:, t])
            out.append(eps_e + eps_i)
        return torch.stack(out, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/compaction.py tests/test_twin_compaction.py
git commit -m "feat(twin): preconsolidation-gated inelastic term"
```

---

### Task 6: Viscous relaxation

**Files:**
- Modify: `hydrophysics/twin/compaction.py`
- Test: `tests/test_twin_compaction.py`

**Interfaces:**
- Consumes: `VEPColumn` from Task 5.
- Produces: inelastic strain now relaxes toward its equilibrium instead of arriving instantly: `τ dε_i/dt + ε_i = ε_i^eq`, integrated with the exponential (exact for piecewise-constant forcing) update `ε_i ← ε_i^eq + (ε_i − ε_i^eq)·exp(−Δt/τ)`. This is the term Tsai & Hsu found dominant and the algebraic `Sk` lacks entirely.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_compaction.py
def _step_response(tau_days, n_steps=60, dt=30.0):
    col = VEPColumn(n_sites=1, dt_days=dt)
    with torch.no_grad():
        col.log_ske.fill_(-30.0)                                  # elastic off
        col.log_skv.fill_(torch.log(torch.tensor(1e-2)).item())
        col.log_tau.fill_(torch.log(torch.tensor(float(tau_days))).item())
        col.h_pc0.fill_(0.0)
    h = torch.cat([torch.zeros(1, 1), torch.full((1, n_steps - 1), -10.0)], dim=1)
    return col(h)[0]


def test_tau_small_reproduces_the_instantaneous_limit():
    fast = _step_response(1e-3)
    assert fast[1].item() == pytest.approx(fast[-1].item(), rel=1e-3)   # arrives at once


def test_tau_large_suppresses_compaction_within_the_window():
    slow = _step_response(1e6)
    fast = _step_response(1e-3)
    assert slow[-1].item() < 0.05 * fast[-1].item()


def test_intermediate_tau_relaxes_monotonically_toward_equilibrium():
    mid = _step_response(365.0)
    d = torch.diff(mid[1:])
    assert torch.all(d >= -1e-9)                    # monotone approach
    assert mid[-1].item() > mid[2].item()           # still rising: memory present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -k tau -v`
Expected: FAIL — `test_tau_large_suppresses_compaction_within_the_window` fails because compaction is currently instantaneous regardless of `log_tau`.

- [ ] **Step 3: Write minimal implementation**

```python
# replace VEPColumn.forward in hydrophysics/twin/compaction.py
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ske = torch.exp(self.log_ske)
        skv = torch.exp(self.log_skv)
        tau = torch.exp(self.log_tau)
        decay = torch.exp(-torch.tensor(self.dt_days, dtype=h.dtype, device=h.device) / tau)
        n, T = h.shape
        h_pc = torch.minimum(self.h_pc0, h[:, 0])
        eps_i = torch.zeros(n, dtype=h.dtype, device=h.device)
        eq = torch.zeros(n, dtype=h.dtype, device=h.device)
        out = [torch.zeros(n, dtype=h.dtype, device=h.device)]
        for t in range(1, T):
            below = torch.clamp(h_pc - h[:, t], min=0.0)
            eq = eq + skv * below                     # equilibrium inelastic strain
            h_pc = torch.minimum(h_pc, h[:, t])
            eps_i = eq + (eps_i - eq) * decay         # exact for piecewise-constant forcing
            eps_e = ske * (h[:, 0] - h[:, t])
            out.append(eps_e + eps_i)
        return torch.stack(out, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add hydrophysics/twin/compaction.py tests/test_twin_compaction.py
git commit -m "feat(twin): viscous relaxation completes the VEP column"
```

---

### Task 7: Stage-2 gate — calibrate the VEP column against MLCW

**Files:**
- Create: `hydrophysics/twin/calibrate_mlcw.py`
- Test: `tests/test_twin_compaction.py`

**Interfaces:**
- Consumes: `VEPColumn` (Task 6); `subsidence.{mlcw_compaction, load_mlcw_stations, monthly_heads, well_xy, idw_interp}`; `data.load_data`.
- Produces:
  - `fit_column(h, obs, mask, epochs=2000, lr=0.05, device=None) -> tuple[VEPColumn, dict]` — fits one `VEPColumn` to `h` and `obs`, both `(n_sites, T)`, using masked MSE; returns the model and `{"loss": float, "epochs": int}`.
  - `loso(h, obs, mask, **kw) -> dict` — leave-one-site-out. Each held-out site's parameters are the ΣD²-weighted mean of the trained sites' log-parameters, and the gate is pooled compaction R² over held-out sites, directly comparable to `subsidence.loso_sk_regression`'s `r2`.
  - `main(argv=None) -> None` — CLI writing `results/twin/stage2_vep_mlcw.csv`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_twin_compaction.py
import numpy as np

from hydrophysics.twin.calibrate_mlcw import fit_column, loso


def _synthetic(n_sites=6, T=80, seed=0):
    """Heads and compaction generated BY the model, so recovery is checkable."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T, dtype=torch.float32)
    h = -(t / T) * (5.0 + 10.0 * torch.rand(n_sites, 1, generator=g))
    h = h + 0.5 * torch.sin(2 * torch.pi * t / 12.0)
    truth = VEPColumn(n_sites=n_sites)
    with torch.no_grad():
        truth.log_ske.copy_(torch.log(torch.full((n_sites,), 5e-4)))
        truth.log_skv.copy_(torch.log(torch.full((n_sites,), 8e-3)))
        truth.log_tau.copy_(torch.log(torch.full((n_sites,), 200.0)))
        truth.h_pc0.zero_()
    obs = truth(h).detach()
    return h, obs, torch.ones_like(obs, dtype=torch.bool)


def test_fit_column_recovers_synthetic_compaction():
    h, obs, mask = _synthetic()
    model, info = fit_column(h, obs, mask, epochs=800, lr=0.05)
    dev = next(model.parameters()).device          # fit_column may have moved to CUDA
    pred = model(h.to(dev)).detach().cpu()
    ss_res = ((pred - obs) ** 2).sum().item()
    ss_tot = ((obs - obs.mean()) ** 2).sum().item()
    assert 1.0 - ss_res / ss_tot > 0.95
    assert info["loss"] < 1e-4


def test_loso_returns_a_finite_gate_number():
    h, obs, mask = _synthetic()
    out = loso(h, obs, mask, epochs=300, lr=0.05)
    assert np.isfinite(out["r2"])
    assert out["n_sites"] == h.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -k "fit_column or loso" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrophysics.twin.calibrate_mlcw'`

- [ ] **Step 3: Write minimal implementation**

```python
# hydrophysics/twin/calibrate_mlcw.py
"""Stage 2: fit the VEP column to the depth-resolved MLCW sites and run the LOSO gate.

The gate is pooled compaction R2 over held-out sites, the same statistic
``subsidence.loso_sk_regression`` reports, so the VEP column is directly comparable to the
algebraic Sk baseline (-0.28 single-Sk, -2.40 spatial-IDW in the README).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data import load_dataset
from ..subsidence import idw_interp, load_mlcw_stations, mlcw_compaction, monthly_heads, well_xy
from .compaction import VEPColumn


def fit_column(h: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor,
               epochs: int = 2000, lr: float = 0.05, device=None) -> tuple[VEPColumn, dict]:
    """Fit one VEPColumn to (h, obs) under ``mask`` with masked MSE."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h, obs, mask = h.to(dev), obs.to(dev), mask.to(dev)
    model = VEPColumn(n_sites=h.shape[0], device=dev).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(h)
        loss = (((pred - obs) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, {"loss": float(loss.detach().cpu()), "epochs": epochs}


def loso(h: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, **kw) -> dict:
    """Leave-one-site-out gate: pooled compaction R2 over held-out sites."""
    n = h.shape[0]
    preds, targets = [], []
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        model, _ = fit_column(h[keep], obs[keep], mask[keep], **kw)
        with torch.no_grad():
            w = (obs[keep] ** 2).sum(dim=1)
            w = w / w.sum().clamp(min=1e-12)
            out = VEPColumn(n_sites=1, device=h.device)
            for name in ("log_ske", "log_skv", "log_tau", "h_pc0"):
                src = getattr(model, name).detach().cpu()
                getattr(out, name).copy_((src * w.cpu()).sum().reshape(1))
            p = out(h[held: held + 1].cpu())[0]
        m = mask[held].cpu()
        preds.append(p[m].numpy())
        targets.append(obs[held].cpu()[m].numpy())
    pred = np.concatenate(preds)
    obsv = np.concatenate(targets)
    ss_res = float(((obsv - pred) ** 2).sum())
    ss_tot = float(((obsv - obsv.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "n_sites": n}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage-2 VEP gate on MLCW sites")
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="results/twin")
    args = ap.parse_args(argv)

    cfg = Config(data_dir=Path(args.data)) if args.data else Config()
    data = load_dataset(cfg)
    ddir = str(cfg.data_dir)
    stations = load_mlcw_stations(os.path.join(ddir, "mlcw_stations.csv"))
    comp = mlcw_compaction(ddir)

    H, dates = monthly_heads(data)
    wxy = well_xy(data)
    rows_h, rows_o, rows_m, names = [], [], [], []
    for _, r in stations.iterrows():
        name = r["sub_id"]
        if name not in comp:
            continue
        h_site = idw_interp(np.array([[r["x"], r["y"]]], dtype="float64"), wxy, H)[0]
        c = comp[name].reindex(dates, method="nearest")
        ok = c.notna().to_numpy()
        if ok.sum() < 24:
            continue
        rows_h.append(h_site)
        rows_o.append(np.nan_to_num(c.to_numpy(dtype="float64")))
        rows_m.append(ok)
        names.append(name)

    h = torch.tensor(np.stack(rows_h), dtype=torch.float32)
    obs = torch.tensor(np.stack(rows_o), dtype=torch.float32)
    mask = torch.tensor(np.stack(rows_m))
    obs = obs - obs[:, :1]

    _, info = fit_column(h, obs, mask, epochs=args.epochs, lr=args.lr)
    gate = loso(h, obs, mask, epochs=max(args.epochs // 4, 200), lr=args.lr)
    print(f"sites={len(names)}  in-sample loss={info['loss']:.3e}  "
          f"LOSO compaction R2={gate['r2']:+.3f}")
    print("Baselines to beat (README): single-Sk -0.28, spatial-IDW Sk -2.40")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage2_vep_mlcw.csv")
    pd.DataFrame([{"n_sites": len(names), "loss": info["loss"], "loso_r2": gate["r2"]}]).to_csv(
        path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_twin_compaction.py -v`
Expected: 10 passed

- [ ] **Step 5: Run the real Stage-2 gate**

```bash
export HYDROMIND_GW_DATA="$(pwd)/chou-shui-data/chou-shui-data/data"
.venv/bin/python -m hydrophysics.twin.calibrate_mlcw --epochs 2000
```

Expected: `LOSO compaction R2` printed for the MLCW sites. **The gate passes if it beats −0.28.** Record the number.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/python -m ruff check hydrophysics/twin tests/test_twin_compaction.py tests/test_twin_leveling.py tests/test_twin_sk_leveling.py
git add hydrophysics/twin/calibrate_mlcw.py tests/test_twin_compaction.py
git commit -m "feat(twin): Stage-2 VEP calibration and LOSO gate on MLCW

LOSO compaction R2 = <value> vs single-Sk baseline -0.28."
```

---

## Coverage note — Stage 0

The spec's Stage 0 (data foundation) is already satisfied by cached artefacts and needs no
task of its own: the leveling panel is at
`ls_cache/ls-wra-lsp-obs__choushui_panel.parquet` (13,997 rows, 1,239 sites), the MLCW
series at `ls_cache/clean/ls-wra-mlcw-obs__*.parquet` (14 sites) with coordinates in
`mlcw_stations.csv`, and the tectonic correction is Task 2. Task 3 and Task 7 fail loudly
with a `FileNotFoundError` if any of these are missing, which is the intended behaviour —
they are inputs, not steps.

## Stage gates — what each result means

| Gate | Where | Pass condition | If it fails |
|---|---|---|---|
| Stage 1 | Task 3 | `loso_r2 > 0` on ~556 leveling sites | Spatial sparsity was **not** the wall — the model form is. Tasks 4–7 are justified on evidence |
| Stage 1 | Task 3 | `loso_r2 > 0` **and** large | Sparsity *was* the wall. Simplify the rheology in Tasks 4–7 and report that finding — it contradicts the spec's premise and is publishable on its own |
| Stage 2 | Task 7 | `loso_r2 > -0.28` (beats single-Sk) | The VEP form does not help at 16 sites. Do not proceed to Plan B; investigate whether MLCW's 16 sites are themselves too few |

Both gate numbers must be recorded in commit messages and carried into `docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md` §7 before Plan B is written.
