"""Signal-decomposition simulation model (leakage-free, simulation-mode).

Research idea under test: instead of predicting the raw groundwater level directly,
decompose each well's signal into smoother sub-signals, model each, and recombine.

    level(t) = seasonal(t) + trend(t) + anomaly(t) + noise

The crux is LEAKAGE-FREE projection: every component is fit on the TRAINING period
only and projected forward from forcing (rainfall, upstream level, day-of-year) plus an
initial condition -- never from observed validation groundwater. This is the exact same
information the gray-box ODE and the PhysicsUDE get, so the comparison is apples-to-apples
in SIMULATION mode (free-running hindcast).

Components and how each is made causally projectable:

  - seasonal(t): day-of-year harmonic regression (Fourier, K harmonics). Cyclic, so it
    projects forward indefinitely with zero leakage.
  - trend(t): optional slow linear drift fit on training. Extrapolating a multi-year
    trend is risky (it can run away over a 4-year val window); off by default, available
    for ablation.
  - anomaly(t): forcing-driven part. A distributed-lag transfer function on rainfall and
    upstream-level anomalies. Rainfall enters through an exponential-memory state
    (antecedent precipitation index, APor recharge memory); upstream enters as a lagged
    anomaly. Coefficients are fit by ridge regression on the TRAINING residual
    (level - seasonal - trend). Driven entirely by known forcing => projectable.

Recombination is anchored to the per-well initial condition (last training level) so the
free-run hindcast starts at the right level, exactly like the other simulation models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import GWData
from .models.base import GroundwaterModel


def _harmonic_design(doy: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Fourier design matrix [1, sin, cos, sin2, cos2, ...] -> (T, 1 + 2K)."""
    ang = 2 * np.pi * doy.astype(float) / 365.25
    cols = [np.ones_like(ang)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * ang))
        cols.append(np.cos(k * ang))
    return np.stack(cols, axis=-1)


def _api_state(rain: np.ndarray, decay: float) -> np.ndarray:
    """Antecedent-precipitation-index / exponential recharge memory.

    s[t] = decay * s[t-1] + rain[t].  A causal IIR filter: today's recharge state is a
    decayed sum of past rainfall. ``decay`` in (0,1) sets the memory timescale
    (tau = -1/ln(decay) days). Purely a function of the (known) rainfall forcing.
    """
    out = np.zeros_like(rain, dtype=float)
    acc = 0.0
    for t in range(rain.size):
        acc = decay * acc + rain[t]
        out[t] = acc
    return out


@dataclass
class DecompConfig:
    n_harmonics: int = 3            # seasonal Fourier harmonics
    use_trend: bool = False         # extrapolate a linear training trend (risky)
    trend_damping: float = 1.0      # geometric damping of the extrapolated trend per day
                                    # (1.0 = pure linear; <1 reins in runaway drift)
    rain_decays: tuple = (0.95, 0.85)   # exponential memory timescales for rainfall
    use_upstream: bool = True       # include upstream-level anomaly as a driver
    ups_lag: int = 0                # upstream lag in days
    ridge: float = 10.0             # ridge penalty on the anomaly transfer-function coefs
    anomaly_clip: float = 4.0       # clip anomaly to +/- this * training residual std


class DecompModel(GroundwaterModel):
    """Decomposition simulation model satisfying the GroundwaterModel contract."""

    name = "decomp"

    def __init__(self, cfg: DecompConfig | None = None):
        self.cfg = cfg or DecompConfig()
        self._per_well: list[dict] = []

    # ---- per-well fit on the training period only ----
    def fit(self, data: GWData) -> "DecompModel":
        cfg = self.cfg
        tm = data.train_mask
        doy_all = data.doy
        self._per_well = []
        for i in range(data.n_wells):
            h = data.target[i].astype(float)
            rain = np.nan_to_num(data.rainfall[i]).astype(float)
            ups = data.upstream[i].astype(float)
            # upstream may have gaps (or be entirely absent); fill with its training
            # mean for the driver state, falling back to 0 when no training value exists.
            ups_tm = ups[tm]
            ups_tm = ups_tm[np.isfinite(ups_tm)]
            ups_mean = float(ups_tm.mean()) if ups_tm.size else 0.0
            ups_fill = np.nan_to_num(ups, nan=ups_mean)

            train_fin = tm & np.isfinite(h)
            info: dict = {}

            # 1) seasonal: harmonic regression on training days
            Xs = _harmonic_design(doy_all, cfg.n_harmonics)
            coef_s, *_ = np.linalg.lstsq(Xs[train_fin], h[train_fin], rcond=None)
            seasonal = Xs @ coef_s
            info["seasonal"] = seasonal

            resid = h - seasonal

            # 2) trend: optional slow linear drift on training residual. Extrapolation
            # past the last training day is geometrically damped (trend_damping) so a
            # multi-year free run cannot run away on a noisy training slope.
            t_idx = np.arange(data.n_days, dtype=float)
            if cfg.use_trend:
                tfit = t_idx[train_fin]
                slope, intercept = np.polyfit(tfit, resid[train_fin], 1)
                t_last = tfit[-1] if tfit.size else 0.0
                trend = slope * t_idx + intercept
                if cfg.trend_damping < 1.0:
                    # within training: linear; after t_last: damped geometric extension
                    future = t_idx > t_last
                    n_ahead = t_idx[future] - t_last
                    # cumulative damped increment = slope * sum_{k=1..n} damping^k
                    d = cfg.trend_damping
                    damped_extra = slope * d * (1 - d ** n_ahead) / (1 - d)
                    base_last = slope * t_last + intercept
                    trend[future] = base_last + damped_extra
            else:
                trend = np.zeros(data.n_days)
            info["trend"] = trend

            resid2 = resid - trend  # forcing-driven anomaly target on training

            # 3) anomaly: distributed-lag transfer function on rainfall + upstream anomalies
            drivers = []
            for dec in cfg.rain_decays:
                drivers.append(_api_state(rain, dec))
            if cfg.use_upstream:
                u = ups_fill.copy()
                if cfg.ups_lag > 0:
                    u = np.concatenate([np.full(cfg.ups_lag, u[0]), u[:-cfg.ups_lag]])
                drivers.append(u)
            D = np.stack(drivers, axis=-1)  # (T, n_drivers)
            # standardize drivers on training for a stable ridge fit
            d_mu = D[tm].mean(0)
            d_sd = D[tm].std(0) + 1e-9
            Dn = (D - d_mu) / d_sd
            Dn1 = np.concatenate([np.ones((data.n_days, 1)), Dn], axis=-1)
            A = Dn1[train_fin]
            y = resid2[train_fin]
            # ridge (do not penalize the intercept)
            P = cfg.ridge * np.eye(A.shape[1])
            P[0, 0] = 0.0
            coef_a = np.linalg.solve(A.T @ A + P, A.T @ y)
            anomaly = Dn1 @ coef_a
            info["anomaly_full"] = anomaly
            info["resid_std"] = float(resid2[train_fin].std()) if train_fin.sum() > 1 else 1.0

            # 4) initial-condition anchor: free-run must start at the last training level.
            train_obs_idx = np.where(train_fin)[0]
            if train_obs_idx.size:
                t0 = train_obs_idx[-1]
                ic = h[t0]
                base_t0 = seasonal[t0] + trend[t0] + anomaly[t0]
                info["anchor"] = ic - base_t0  # constant offset so recombination hits IC
            else:
                info["anchor"] = 0.0
            self._per_well.append(info)
        return self

    def simulate(self, data: GWData) -> np.ndarray:
        if not self._per_well:
            raise RuntimeError("call fit() before simulate()")
        cfg = self.cfg
        pred = np.full(data.target.shape, np.nan)
        for i, info in enumerate(self._per_well):
            anomaly = info["anomaly_full"]
            clip = cfg.anomaly_clip * info["resid_std"]
            anomaly = np.clip(anomaly, -clip, clip)
            pred[i] = info["seasonal"] + info["trend"] + anomaly + info["anchor"]
        return pred
