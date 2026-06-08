"""GlobalForecastLSTM: operational multi-horizon groundwater forecasting (Task B).

Unlike the simulation-mode models (free-running hindcast), this is a FORECASTER: at an
origin day t0 it predicts the level at t0+1 .. t0+H using observed levels up to t0 (data
assimilation) plus forcing. One LSTM is shared across all wells and conditioned on their
static attributes, so it is a single global model -- not one fit per well.

Design (see docs/superpowers/specs/2026-06-08-groundwater-forecasting-design.md):
  - Encoder: an LSTM over a lookback window of [level-anomaly, rainfall, upstream, sin,
    cos]. Level is fed as a deviation from y[t0] scaled by the well's training std, so the
    network is well-agnostic and assimilates the current state directly.
  - Head: a small MLP over [encoder state, static attributes, future forcing] predicts the
    H-step anomaly relative to y[t0]. Future forcing (rain/upstream/season over the
    horizon) is the perfect-forcing assumption, standard for forecast skill assessment.
  - Target/loss: per-well-normalized MSE on the anomaly, so every well weighs equally.

This class is self-contained and does NOT implement the simulation-mode GroundwaterModel
contract; it is scored by hydrophysics.forecast_eval, which keeps forecast and simulation
benchmarks separate (mixing them is the classic groundwater evaluation trap).
"""

from __future__ import annotations

import numpy as np

from ..data import GWData
from .gru import _default_device, _static_features

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

ENC_FEATURES = 5   # level-anomaly, rainfall, upstream, sin, cos
FUT_FEATURES = 4   # rainfall, upstream, sin, cos


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("GlobalForecastLSTM needs PyTorch. pip install 'hydrophysics[gpu]'.")


def _seasonal(doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = 2 * np.pi * doy.astype(float) / 365.25
    return np.sin(rad), np.cos(rad)


class GlobalForecastLSTM:
    name = "forecast_lstm"

    def __init__(self, lookback: int = 90, horizon: int = 30, hidden: int = 64,
                 layers: int = 1, epochs: int = 40, lr: float = 1e-3, batch: int = 4096,
                 probabilistic: bool = False, device: str | None = None, seed: int = 0):
        _require_torch()
        self.L, self.H = lookback, horizon
        self.hidden, self.layers, self.epochs = hidden, layers, epochs
        self.lr, self.batch, self.seed = lr, batch, seed
        # Probabilistic mode: the head emits a per-horizon mean AND log-variance, trained
        # by Gaussian NLL, so each forecast is a distribution N(mu, sigma^2). Lets us read
        # off prediction intervals / exceedance probabilities for early warning.
        self.probabilistic = probabilistic
        self.device = device or _default_device()
        self.net: nn.Module | None = None
        self._stats: dict = {}

    # --- feature construction --------------------------------------------------
    def _level_std(self, data: GWData) -> np.ndarray:
        sd = np.ones(data.n_wells, dtype="float32")
        for i in range(data.n_wells):
            obs = data.target[i, data.train_mask]
            obs = obs[np.isfinite(obs)]
            if obs.size and obs.std() > 1e-6:
                sd[i] = obs.std()
        return sd

    def _channels(self, data: GWData):
        """Per-well daily channels (W, T) used to cut windows. NaNs kept for masking."""
        sin, cos = _seasonal(data.doy)
        W, T = data.target.shape
        level = data.target.astype("float32")
        rain = np.nan_to_num(data.rainfall).astype("float32")
        ups = np.nan_to_num(data.upstream).astype("float32")
        sin = np.broadcast_to(sin, (W, T)).astype("float32")
        cos = np.broadcast_to(cos, (W, T)).astype("float32")
        return level, rain, ups, sin, cos

    def _build_samples(self, data: GWData, origin_ok: np.ndarray):
        """Cut (encoder, future-forcing, target) windows for every valid (well, origin).

        origin_ok: (W, T) bool -- True where day t may serve as a forecast origin under
        the period rule (training vs all). A valid origin also needs a full finite
        lookback and a full finite horizon span. Returns torch tensors + index arrays.
        """
        L, H = self.L, self.H
        level, rain, ups, sin, cos = self._channels(data)
        finite = np.isfinite(data.target)
        s = self._stats
        W, T = level.shape

        enc_list, fut_list, tgt_list, wi, t0i = [], [], [], [], []
        for i in range(W):
            sd = s["level_sd"][i]
            for t0 in range(L - 1, T - H):
                lo, hi = t0 - L + 1, t0 + H        # span [lo, t0] + (t0, t0+H]
                if not origin_ok[i, t0]:
                    continue
                if not finite[i, lo:hi + 1].all():
                    continue
                y0 = level[i, t0]
                enc = np.stack([
                    (level[i, lo:t0 + 1] - y0) / sd,
                    (rain[i, lo:t0 + 1] - s["rain_mu"]) / s["rain_sd"],
                    (ups[i, lo:t0 + 1] - s["ups_mu"]) / s["ups_sd"],
                    sin[i, lo:t0 + 1], cos[i, lo:t0 + 1],
                ], axis=-1)                          # (L, ENC_FEATURES)
                fut = np.stack([
                    (rain[i, t0 + 1:hi + 1] - s["rain_mu"]) / s["rain_sd"],
                    (ups[i, t0 + 1:hi + 1] - s["ups_mu"]) / s["ups_sd"],
                    sin[i, t0 + 1:hi + 1], cos[i, t0 + 1:hi + 1],
                ], axis=-1)                          # (H, FUT_FEATURES)
                tgt = (level[i, t0 + 1:hi + 1] - y0) / sd   # (H,) normalized anomaly
                enc_list.append(enc)
                fut_list.append(fut)
                tgt_list.append(tgt)
                wi.append(i)
                t0i.append(t0)

        if not enc_list:
            raise RuntimeError("no valid forecast windows (check lookback/horizon vs data)")
        enc = torch.from_numpy(np.asarray(enc_list, dtype="float32")).to(self.device)
        fut = torch.from_numpy(np.asarray(fut_list, dtype="float32")).to(self.device)
        tgt = torch.from_numpy(np.asarray(tgt_list, dtype="float32")).to(self.device)
        wi = np.asarray(wi)
        t0i = np.asarray(t0i)
        stat = torch.from_numpy(s["stat_n"][wi]).to(self.device)
        return enc, fut, stat, tgt, wi, t0i

    # --- fit / forecast --------------------------------------------------------
    def fit(self, data: GWData) -> "GlobalForecastLSTM":
        torch.manual_seed(self.seed)
        s = self._stats
        s["level_sd"] = self._level_std(data)
        tm = data.train_mask
        rain_t = np.nan_to_num(data.rainfall)[:, tm]
        ups_t = np.nan_to_num(data.upstream)[:, tm]
        s["rain_mu"], s["rain_sd"] = float(rain_t.mean()), float(rain_t.std() + 1e-6)
        s["ups_mu"], s["ups_sd"] = float(ups_t.mean()), float(ups_t.std() + 1e-6)
        stat = _static_features(data)
        s["stat_mu"], s["stat_sd"] = stat.mean(0), stat.std(0) + 1e-6
        s["stat_n"] = ((stat - s["stat_mu"]) / s["stat_sd"]).astype("float32")

        origin_ok = np.broadcast_to(tm, data.target.shape).copy()
        enc, fut, stat_t, tgt, wi, _ = self._build_samples(data, origin_ok)
        sd_w = torch.from_numpy(s["level_sd"][wi]).to(self.device)   # for per-well weight

        self.net = _ForecastNet(ENC_FEATURES, FUT_FEATURES, stat_t.shape[-1],
                                self.hidden, self.layers, self.H,
                                probabilistic=self.probabilistic).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs, eta_min=self.lr * 1e-2)
        n = enc.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for b in range(0, n, self.batch):
                idx = perm[b:b + self.batch]
                opt.zero_grad()
                out = self.net(enc[idx], fut[idx], stat_t[idx])
                if self.probabilistic:
                    mean, log_var = out
                    # Gaussian NLL in normalized anomaly space (constants dropped).
                    loss = 0.5 * (log_var + (tgt[idx] - mean) ** 2 / log_var.exp()).mean()
                else:
                    # Targets are already per-well normalized, so plain MSE weights wells
                    # equally; sd_w kept for clarity / future weighting.
                    loss = ((out - tgt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            sched.step()
        del sd_w
        return self

    def _predict(self, data: GWData):
        """Return (mean_cube, sigma_cube) of absolute levels, each (W, T, H).

        [i, t0, h-1] = forecast of the level at t0+h. sigma_cube is None unless the model
        is probabilistic. NaN where the origin lacks a full finite lookback/horizon span;
        origins are not restricted to a period -- the eval harness picks target days.
        """
        if self.net is None:
            raise RuntimeError("call fit() before forecast()")
        s = self._stats
        level = data.target.astype("float32")
        shape = (data.n_wells, data.n_days, self.H)
        mean_cube = np.full(shape, np.nan, dtype="float32")
        sigma_cube = np.full(shape, np.nan, dtype="float32") if self.probabilistic else None
        origin_ok = np.ones(data.target.shape, dtype=bool)
        enc, fut, stat_t, _, wi, t0i = self._build_samples(data, origin_ok)
        self.net.eval()
        means, sigmas = [], []
        with torch.no_grad():
            for b in range(0, enc.shape[0], self.batch):
                out = self.net(enc[b:b + self.batch], fut[b:b + self.batch],
                               stat_t[b:b + self.batch])
                if self.probabilistic:
                    m, log_var = out
                    means.append(m.cpu().numpy())
                    sigmas.append((0.5 * log_var).exp().cpu().numpy())   # std in norm space
                else:
                    means.append(out.cpu().numpy())
        mean = np.concatenate(means, axis=0)            # (N, H) normalized anomaly
        sig = np.concatenate(sigmas, axis=0) if self.probabilistic else None
        for k in range(len(wi)):
            i, t0 = wi[k], t0i[k]
            sd = s["level_sd"][i]
            mean_cube[i, t0] = level[i, t0] + mean[k] * sd
            if self.probabilistic:
                sigma_cube[i, t0] = sig[k] * sd          # de-normalize std to level units
        return mean_cube, sigma_cube

    def forecast(self, data: GWData) -> np.ndarray:
        """Point forecast (W, T, H): [i, t0, h-1] = predicted level at t0+h."""
        return self._predict(data)[0]

    def forecast_dist(self, data: GWData) -> tuple[np.ndarray, np.ndarray]:
        """Probabilistic forecast: (mean, sigma) cubes, each (W, T, H). Needs probabilistic=True."""
        if not self.probabilistic:
            raise RuntimeError("model was not built with probabilistic=True")
        mean, sigma = self._predict(data)
        return mean, sigma


if _HAS_TORCH:
    class _ForecastNet(nn.Module):
        def __init__(self, n_enc: int, n_fut: int, n_stat: int, hidden: int,
                     layers: int, horizon: int, probabilistic: bool = False):
            super().__init__()
            self.horizon = horizon
            self.probabilistic = probabilistic
            self.lstm = nn.LSTM(n_enc, hidden, layers, batch_first=True)
            head_in = hidden + n_stat + n_fut * horizon
            out_dim = horizon * (2 if probabilistic else 1)
            self.head = nn.Sequential(
                nn.Linear(head_in, hidden), nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, enc, fut, stat):
            _, (h, _) = self.lstm(enc)             # h: (layers, B, hidden)
            z = torch.cat([h[-1], stat, fut.flatten(1)], dim=-1)
            out = self.head(z)
            if not self.probabilistic:
                return out
            mean, log_var = out[:, :self.horizon], out[:, self.horizon:]
            # clamp log-variance for numerical stability of the NLL
            return mean, log_var.clamp(-8.0, 8.0)
