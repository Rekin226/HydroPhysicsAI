"""PhysicsNeMo FNO surrogate of the calibrated flow solver.

    python -m hydrophysics.twin.surrogate build --theta <stage3_theta.json> \\
        --n-samples 256 --horizon 24 --out results/surrogate/data.npz
    python -m hydrophysics.twin.surrogate train --data results/surrogate/data.npz \\
        --out results/surrogate/fno
    python -m hydrophysics.twin.surrogate eval --model results/surrogate/fno \\
        --theta <stage3_theta.json> --scenario "cut30:irrigation=0.7" --horizon 24

What it is for, stated carefully. A single scenario run of the differentiable solver
costs seconds on the GPU (about four minutes on CPU), so the surrogate is **not** what
makes the twin re-runnable on demand -- ``twin.forward`` already is. What the solver
cannot do cheaply is *many* runs: a policy sweep, a large initial-condition ensemble, or
an interactive slider that wants a new head field per frame. That is where an operator
that maps (state, forcing) -> next state in a few milliseconds earns its place, and it
is the one place in the twin where PhysicsNeMo is used.

Design. One-step autoregressive: input channels are the four layer heads, this month's
electricity and recharge, ground elevation and the fan mask, all on the ``(ny, nx)``
raster; the output is the four-layer head **increment** for the month, which keeps the
target zero-mean and the rollout stable. Training pairs come from solver rollouts under
randomised policies (per-class multipliers, log-uniform in [0.25, 2]) and rain scales
([0.5, 1.5]) started from states along the hindcast, so the surrogate sees the range of
forcing a scenario can ask for. fp32 throughout: the card is Turing and has no bf16.

The surrogate inherits every defect of the solver it is trained on. Its accuracy is
reported *against the solver*, never against observations; the gate is the solver's.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..train import pick_device
from .forward import (
    N_LAYERS,
    attach_sw_recharge,
    build_model,
    delay_state_path,
    future_forcing,
    future_sw,
    load_members,
    parse_scenario,
    rollout,
    sw_hist,
)
from .inputs import TwinInputs, input_options, load_twin_inputs
from .scenario import BASELINE, CLASSES, PumpingScenario

IN_CHANNELS = N_LAYERS + 4          # heads x4, log1p(E), recharge, ground elevation, mask
E_SCALE = 1.0                       # E enters as log1p(kWh / E_SCALE)


# ---------------------------------------------------------------------------------------
# raster <-> active-cell vector
# ---------------------------------------------------------------------------------------
def rasterize(grid, vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """``(..., A)`` -> ``(..., ny, nx)`` with ``fill`` outside the fan."""
    out = np.full(vec.shape[:-1] + grid.mask.shape, fill, dtype="float32")
    out[..., grid.mask] = vec
    return out


def unrasterize(grid, ras: np.ndarray) -> np.ndarray:
    return ras[..., grid.mask]


# ---------------------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------------------
def _random_policy(rng: np.random.Generator, classes: list[str]) -> tuple[PumpingScenario, float]:
    factors = {c: float(np.exp(rng.uniform(np.log(0.25), np.log(2.0)))) for c in classes}
    rain = float(rng.uniform(0.5, 1.5))
    return PumpingScenario("random", factors=factors), rain


def build_dataset(inp: TwinInputs, member, n_samples: int, horizon: int, seed: int,
                  device, log=print) -> dict:
    """Solver rollouts under random policies -> one-step training pairs on the raster."""
    rng = np.random.default_rng(seed)
    model, scalars, zone_of_cell = build_model(inp.grid, member, device)
    eta_classes = member.meta.get("eta_classes")
    classes = [c for c in CLASSES if c in inp.E_by_class]
    T = len(inp.dates)
    if eta_classes:
        E_hist = torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                              dtype=torch.float64)[..., 1:]
    else:
        E_hist = inp.E_total[:, 1:]
    h0 = inp.initial_heads(0, n_layers=N_LAYERS)
    t0 = time.perf_counter()
    attach_sw_recharge(inp, member.meta, log=log)
    h_hist = rollout(model, scalars, h0, E_hist, inp.recharge_field[:, 1:], inp.ground_elev,
                     sw_field=sw_hist(inp))
    # a delay-bed model's projection must start from the record's slow-store state, as
    # forward.run and policy_gate do; resetting it to equilibrium is different dynamics
    u_hist = delay_state_path(model, scalars, h_hist)
    if u_hist is not None:
        log("WARNING: delay-bed model -- the solver targets carry the slow store, but the "
            "FNO does not see it as an input; check the one-step error before trusting it")
    if sw_hist(inp) is not None:
        log("WARNING: canal-water model -- the solver targets carry the canal deliveries, "
            "but the FNO has no canal-water input channel; a sw policy lever is invisible "
            "to it")
    log(f"hindcast for start states: {time.perf_counter() - t0:.1f}s")
    ge = rasterize(inp.grid, inp.ground_elev.numpy())
    mask = inp.grid.mask.astype("float32")

    X, Y = [], []
    for i in range(n_samples):
        scen, rain = _random_policy(rng, classes)
        start = int(rng.integers(0, T))
        E_fut, r_fut, _ = future_forcing(inp, scen, horizon, rain_scale=rain,
                                         zone_of_cell=zone_of_cell, eta_classes=eta_classes)
        h_start = h_hist[..., start]
        h = rollout(model, scalars, h_start, E_fut, r_fut, inp.ground_elev,   # (L, A, H+1)
                    sw_field=future_sw(inp, scen, horizon, zone_of_cell),
                    u0=None if u_hist is None else u_hist[..., start])
        hn = h.cpu().numpy()
        E_tot = E_fut.sum(dim=0).numpy() if E_fut.dim() == 3 else E_fut.numpy()
        for t in range(horizon):
            x = np.concatenate([
                rasterize(inp.grid, hn[:, :, t]),
                rasterize(inp.grid, np.log1p(E_tot[:, t] / E_SCALE))[None],
                rasterize(inp.grid, r_fut[:, t].numpy() * 1000.0)[None],   # mm/day
                ge[None], mask[None]], axis=0)
            X.append(x.astype("float32"))
            Y.append(rasterize(inp.grid, hn[:, :, t + 1] - hn[:, :, t]).astype("float32"))
        if (i + 1) % max(1, n_samples // 10) == 0:
            log(f"  sample {i + 1}/{n_samples} ({time.perf_counter() - t0:.0f}s)")
    return {"X": np.stack(X), "Y": np.stack(Y), "mask": inp.grid.mask,
            "horizon": horizon, "n_samples": n_samples}


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------
def make_fno(modes: int = 12, width: int = 32, layers: int = 4, padding: int = 8):
    from physicsnemo.models.fno import FNO

    return FNO(in_channels=IN_CHANNELS, out_channels=N_LAYERS, dimension=2,
               latent_channels=width, num_fno_layers=layers, num_fno_modes=modes,
               padding=padding, decoder_layers=1, decoder_layer_size=width)


class Normalizer:
    """Per-channel affine normalisation, fitted on the training set, saved beside the
    weights so inference reproduces training exactly."""

    def __init__(self, x_mean, x_std, y_std):
        self.x_mean = np.asarray(x_mean, dtype="float32")
        self.x_std = np.asarray(x_std, dtype="float32")
        self.y_std = np.asarray(y_std, dtype="float32")

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> Normalizer:
        m = mask[None, None]
        n = float(mask.sum()) * X.shape[0]
        x_mean = (X * m).sum(axis=(0, 2, 3)) / n
        x_std = np.sqrt(((X - x_mean[None, :, None, None]) ** 2 * m).sum(axis=(0, 2, 3)) / n)
        y_std = np.sqrt((Y ** 2 * m).sum(axis=(0, 2, 3)) / n)
        x_std[IN_CHANNELS - 1] = 1.0          # the mask channel stays 0/1
        x_mean[IN_CHANNELS - 1] = 0.0
        return cls(x_mean, np.maximum(x_std, 1e-6), np.maximum(y_std, 1e-6))

    def x(self, X: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.x_mean, device=X.device)[None, :, None, None]
        s = torch.as_tensor(self.x_std, device=X.device)[None, :, None, None]
        return (X - m) / s

    def y(self, Y: torch.Tensor) -> torch.Tensor:
        return Y / torch.as_tensor(self.y_std, device=Y.device)[None, :, None, None]

    def y_inv(self, Yn: torch.Tensor) -> torch.Tensor:
        return Yn * torch.as_tensor(self.y_std, device=Yn.device)[None, :, None, None]

    def to_json(self) -> dict:
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "y_std": self.y_std.tolist()}


def masked_rel_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[None, None].to(pred.dtype)
    num = (((pred - target) ** 2) * m).sum(dim=(1, 2, 3))
    den = ((target ** 2) * m).sum(dim=(1, 2, 3)).clamp(min=1e-12)
    return (num / den).sqrt().mean()


def train(data: dict, out: str, epochs: int = 200, lr: float = 1e-3, batch: int = 16,
          modes: int = 12, width: int = 32, layers: int = 4, val_frac: float = 0.1,
          seed: int = 0, device=None, log=print) -> dict:
    dev = device or pick_device()
    torch.manual_seed(seed)
    X, Y, mask = data["X"], data["Y"], data["mask"]
    n = X.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_frac))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    norm = Normalizer.fit(X[tr_i], Y[tr_i], mask)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    mt = torch.tensor(mask, dtype=torch.float32, device=dev)
    model = make_fno(modes=modes, width=width, layers=layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(tr_i)
        tot, nb = 0.0, 0
        for b in range(0, len(perm), batch):
            bi = perm[b:b + batch]
            xb = norm.x(Xt[bi].to(dev))
            yb = norm.y(Yt[bi].to(dev))
            opt.zero_grad()
            loss = masked_rel_l2(model(xb), yb, mt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            xv = norm.x(Xt[val_i].to(dev))
            yv = norm.y(Yt[val_i].to(dev))
            vl = float(masked_rel_l2(model(xv), yv, mt))
        hist.append({"epoch": ep + 1, "train": tot / max(nb, 1), "val": vl})
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            log(f"  epoch {ep + 1:4d}: train rel-L2 {tot / max(nb, 1):.4f}  val {vl:.4f}  "
                f"({time.perf_counter() - t0:.0f}s)")
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "fno.mdlus"))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"norm": norm.to_json(), "modes": modes, "width": width, "layers": layers,
                   "in_channels": IN_CHANNELS, "n_train": int(len(tr_i)), "n_val": int(n_val),
                   "epochs": epochs, "history": hist, "mask_shape": list(mask.shape)},
                  fh, indent=1)
    np.save(os.path.join(out, "mask.npy"), mask)
    return {"model": model, "norm": norm, "history": hist}


def load_surrogate(path: str, device=None):
    from physicsnemo.models.fno import FNO

    dev = device or pick_device()
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    model = FNO.from_checkpoint(os.path.join(path, "fno.mdlus")).to(dev)
    model.eval()
    norm = Normalizer(**meta["norm"])
    return model, norm, meta


@torch.no_grad()
def surrogate_rollout(model, norm: Normalizer, grid, h0: np.ndarray, E: np.ndarray,
                      recharge: np.ndarray, ground_elev: np.ndarray, device) -> np.ndarray:
    """Autoregressive surrogate rollout -> heads ``(L, A, T+1)`` (numpy, float32)."""
    ge = torch.tensor(rasterize(grid, ground_elev), device=device)[None, None]
    mask = torch.tensor(grid.mask.astype("float32"), device=device)[None, None]
    mask_b = torch.tensor(grid.mask, device=device)
    h = torch.tensor(rasterize(grid, h0), device=device)[None]              # (1, L, ny, nx)
    out = [h0.astype("float32")]
    for t in range(E.shape[-1]):
        e = torch.tensor(rasterize(grid, np.log1p(E[:, t] / E_SCALE)), device=device)[None, None]
        r = torch.tensor(rasterize(grid, recharge[:, t] * 1000.0), device=device)[None, None]
        x = torch.cat([h, e, r, ge, mask], dim=1)
        dh = norm.y_inv(model(norm.x(x)))
        h = (h + dh) * mask
        out.append(h[0][:, mask_b].cpu().numpy())
    return np.stack(out, axis=-1)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def _add_input_args(ap):
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--device", default=None)
    for k in ("polygon", "wells_dir", "stations", "pump_census", "pump_kwh",
              "rf_timeseries", "rf_stations", "gw_stations", "et_npz"):
        ap.add_argument(f"--{k.replace('_', '-')}", default=None)


def _inputs_from(args, meta) -> TwinInputs:
    paths = {k: getattr(args, k) for k in ("polygon", "wells_dir", "stations", "pump_census",
                                           "pump_kwh", "rf_timeseries", "rf_stations",
                                           "gw_stations", "et_npz")}
    return load_twin_inputs(paths, dx=args.dx, meter_filter=meta.get("meter_filter", "none"),
                            cap_duty=float(meta.get("cap_duty", 1.0)),
                            **input_options(meta))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="FNO surrogate of the calibrated flow solver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="solver rollouts under random policies -> training pairs")
    b.add_argument("--theta", required=True)
    b.add_argument("--n-samples", type=int, default=256)
    b.add_argument("--horizon", type=int, default=24)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    _add_input_args(b)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--modes", type=int, default=12)
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)

    e = sub.add_parser("eval", help="surrogate vs solver on named scenarios from the origin")
    e.add_argument("--model", required=True)
    e.add_argument("--theta", required=True)
    e.add_argument("--scenario", action="append", default=[])
    e.add_argument("--horizon", type=int, default=24)
    e.add_argument("--out", default=None, help="CSV of per-layer errors")
    _add_input_args(e)

    args = ap.parse_args(argv)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    if args.cmd == "build":
        member = load_members([args.theta])[0]
        inp = _inputs_from(args, member.meta)
        data = build_dataset(inp, member, args.n_samples, args.horizon, args.seed, device)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(args.out, **data)
        print(f"wrote {args.out}: X {data['X'].shape} Y {data['Y'].shape}")
    elif args.cmd == "train":
        d = np.load(args.data)
        data = {k: d[k] for k in ("X", "Y", "mask")}
        res = train(data, args.out, epochs=args.epochs, lr=args.lr, batch=args.batch,
                    modes=args.modes, width=args.width, layers=args.layers, seed=args.seed,
                    device=device)
        print(f"final val rel-L2 {res['history'][-1]['val']:.4f}; wrote {args.out}")
    else:
        member = load_members([args.theta])[0]
        inp = _inputs_from(args, member.meta)
        fno, norm, _ = load_surrogate(args.model, device)
        model, scalars, zone_of_cell = build_model(inp.grid, member, device)
        eta_classes = member.meta.get("eta_classes")
        if eta_classes:
            E_hist = torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                                  dtype=torch.float64)[..., 1:]
        else:
            E_hist = inp.E_total[:, 1:]
        attach_sw_recharge(inp, member.meta)
        h_hist = rollout(model, scalars, inp.initial_heads(0), E_hist,
                         inp.recharge_field[:, 1:], inp.ground_elev, sw_field=sw_hist(inp))
        h_start = h_hist[..., -1]
        u_hist = delay_state_path(model, scalars, h_hist)
        u_start = None if u_hist is None else u_hist[..., -1]
        scenarios = [(BASELINE, 1.0)] + [parse_scenario(s) for s in args.scenario]
        rows = []
        for scen, rain in scenarios:
            E_fut, r_fut, _ = future_forcing(inp, scen, args.horizon, rain_scale=rain,
                                             zone_of_cell=zone_of_cell, eta_classes=eta_classes)
            t0 = time.perf_counter()
            ref = rollout(model, scalars, h_start, E_fut, r_fut, inp.ground_elev,
                          sw_field=future_sw(inp, scen, args.horizon, zone_of_cell),
                          u0=u_start).cpu().numpy()
            t_solver = time.perf_counter() - t0
            E_tot = E_fut.sum(dim=0).numpy() if E_fut.dim() == 3 else E_fut.numpy()
            t0 = time.perf_counter()
            sur = surrogate_rollout(fno, norm, inp.grid, h_start.cpu().numpy(), E_tot,
                                    r_fut.numpy(), inp.ground_elev.numpy(), device)
            t_sur = time.perf_counter() - t0
            for k in range(N_LAYERS):
                err = sur[k, :, -1] - ref[k, :, -1]
                chg = ref[k, :, -1] - ref[k, :, 0]
                rows.append({"scenario": scen.name, "layer": k + 1,
                             "rmse_m": float(np.sqrt((err ** 2).mean())),
                             "rmse_over_change": float(np.sqrt((err ** 2).mean())
                                                       / max(np.sqrt((chg ** 2).mean()), 1e-9)),
                             "solver_s": t_solver, "surrogate_s": t_sur})
            print(f"{scen.name}: solver {t_solver:.2f}s, surrogate {t_sur:.3f}s "
                  f"({t_solver / max(t_sur, 1e-9):.0f}x); RMSE at month {args.horizon}: "
                  + ", ".join(f"L{r['layer']} {r['rmse_m']:.3f} m"
                              for r in rows[-N_LAYERS:]), flush=True)
        import pandas as pd

        df = pd.DataFrame(rows)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
