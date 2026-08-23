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
