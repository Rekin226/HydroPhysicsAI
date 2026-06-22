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


def _grad(outputs, inputs):
    """Elementwise d(output_i)/d(input_i) via the grad_outputs=1 trick, with the graph
    kept for higher-order derivatives. Assumes a pointwise network (no cross-sample
    coupling such as batch norm). Returns zeros if the term is structurally independent.
    """
    g = torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, allow_unused=True,
    )[0]
    return torch.zeros_like(inputs) if g is None else g


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
    """Physics-informed continuous head field h(x, y, t). See module docstring.

    Outcome on the 61-well Zhuoshui data (documented baseline, not the headline model):
    a *pure spatial-coordinate* operator is not competitive here. With the residual
    enforced over the full record and ``physics_weight`` selected on the inner pre-2019
    split (3e-3), it reaches in-sample KGE ~0.33 and leave-one-well-out KGE ~0.10 --
    below the per-well gray-box (0.74), the lumped UDE (0.59), and even climatology
    (0.45). The reason is structural: a single field conditioned only on (x, y) cannot
    match per-well dynamics or place an unseen well from coordinates alone, whereas the
    UDE conditions on each well's observable history and anchors to its observed mean
    (LOWO 0.565). Kept as an honest continuous-field baseline and for the head-field map;
    a competitive field model would need per-well conditioning (i.e. the UDE + a spatial
    deviation field). See docs/superpowers/specs/2026-06-16-spatial-pinn-head-field-design.md.
    """

    name = "pinn"

    def __init__(self, hidden: int = 64, n_bands: int = 6, epochs: int = 1500,
                 lr: float = 1e-3, n_collocation: int = 2048, physics_weight: float = 3e-3,
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
        self.log_S: nn.Parameter | None = None
        self._wc: np.ndarray | None = None

    # --- helpers -----------------------------------------------------------
    def _obs_rows(self, data: GWData, train_wells: np.ndarray | None):
        """Observation rows entering the data loss: training days, finite obs, and
        (for LOWO) only wells in ``train_wells``. Returns a dict of int/float arrays."""
        keep_well = (np.ones(data.n_wells, dtype=bool) if train_wells is None
                     else np.asarray(train_wells, dtype=bool))
        assert len(keep_well) == data.n_wells, "train_wells must be a length-n_wells boolean mask"
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
        if not wi:
            empty_i = np.array([], dtype=int)
            return {"well": empty_i, "day": empty_i.copy(),
                    "h": np.array([], dtype=float)}
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
    def fit(self, data: GWData, train_wells: np.ndarray | None = None) -> SpatialPINN:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.norm = Normalizer.from_data(data)
        self._wc = well_coords_norm(data, self.norm)            # (W, 2)
        rain_field = RainfallField(self._wc, np.nan_to_num(data.rainfall))
        rain_std = float(np.nan_to_num(data.rainfall)[:, data.train_mask].std()) + 1e-6
        self._build()

        rows = self._obs_rows(data, train_wells)
        if rows["h"].size == 0:
            raise ValueError("No finite training observations for the selected wells.")
        obs_X = torch.tensor(self._wc[rows["well"], 0:1], dtype=torch.float32, device=self.device)
        obs_Y = torch.tensor(self._wc[rows["well"], 1:2], dtype=torch.float32, device=self.device)
        obs_tau = torch.tensor(self.norm.tau(rows["day"])[:, None], dtype=torch.float32, device=self.device)
        obs_H = torch.tensor(self.norm.h_to_norm(rows["h"])[:, None], dtype=torch.float32, device=self.device)

        # Physics is enforced over the FULL record: collocation in time spans both the
        # training and forecast periods. The residual uses no observed levels -- only the
        # PDE and the always-available rainfall forcing -- so this respects the no-leakage
        # contract, and it is what gives the field any constraint in the forecast window
        # (sampling only training days, as an earlier version did, left 2019+ unconstrained
        # and collapsed forecast skill).
        colloc_days = np.arange(data.n_days)
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
            cdays = np.random.choice(colloc_days, size=n)
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
