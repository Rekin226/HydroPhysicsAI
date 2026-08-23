"""Stage 2: fit the VEP column to the depth-resolved MLCW sites and run the LOSO gate.

The gate is pooled compaction R2 over held-out sites. The comparison baseline is computed
on the IDENTICAL (h, obs, mask) arrays used for the VEP fit -- both a pooled through-origin
single-Sk LOSO and a distance-to-coast-regression single-Sk LOSO (the same statistic
``subsidence.loso_sk_regression`` reports for Task 3). The README's -0.28/-2.40 numbers are
NOT used as the gate: they are ``calibrate_sk_from_pairs(...)["r2"]``, an IN-SAMPLE fit on a
different sample set, and are not directly comparable to an out-of-sample number.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data import load_dataset
from ..subsidence import (
    calibrate_sk_from_pairs,
    idw_interp,
    load_mlcw_stations,
    loso_sk_regression,
    mlcw_compaction,
    monthly_heads,
    well_xy,
)
from .compaction import VEPColumn


def fit_column(h: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor,
               epochs: int = 2000, lr: float = 0.05, device=None,
               n_sites: int | None = None) -> tuple[VEPColumn, dict]:
    """Fit one VEPColumn to (h, obs) under ``mask`` with masked MSE.

    ``n_sites`` defaults to ``h.shape[0]`` (one parameter set per row). Passing
    ``n_sites=1`` fits a SINGLE global parameter set against all rows of ``h`` at once --
    ``VEPColumn.forward`` broadcasts its (1,)-shaped parameters against the (n, T) input,
    so this is exactly "one 4-vector explains every row", used by ``loso_shared``.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h, obs, mask = h.to(dev), obs.to(dev), mask.to(dev)
    model = VEPColumn(n_sites=n_sites if n_sites is not None else h.shape[0], device=dev).to(dev)
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
        with torch.no_grad():
            # The loss is flat in these directions, so unbounded parameters run away with the
            # epoch budget and their cross-site mean becomes meaningless. Bound them to
            # physically defensible ranges: Ske/Skv from published aquifer values, tau no
            # longer than the observation window, h_pc0 within the observed head range.
            model.log_ske.clamp_(min=math.log(1e-6), max=math.log(1e-1))
            model.log_skv.clamp_(min=math.log(1e-5), max=math.log(1e0))
            model.log_tau.clamp_(min=math.log(1.0), max=math.log(float(h.shape[1]) * model.dt_days))
            model.h_pc0.clamp_(min=float(h.min()), max=float(h.max()))
        sched.step()
    return model, {"loss": float(loss.detach().cpu()), "epochs": epochs}


def loso(h: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, **kw) -> dict:
    """Leave-one-site-out gate: pooled compaction R2 over held-out sites.

    Also returns ``per_site``, a per-fold diagnostic of {site_index: {r2, loss}}.
    """
    n = h.shape[0]
    preds, targets, per_site = [], [], {}
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        model, _ = fit_column(h[keep], obs[keep], mask[keep], **kw)
        with torch.no_grad():
            # Weight by how much signal (observation count) a site contributed, not by the
            # size of its compaction -- weighting by Sum(C^2) lets one high-subsidence site
            # dominate the cross-site average regardless of how well-observed it is.
            w = mask[keep].sum(dim=1).to(obs.dtype)
            w = w / w.sum().clamp(min=1e-12)
            # built and evaluated on CPU, consistently; dt_days must match the fitted
            # model's -- it is NOT the VEPColumn default (30.0) whenever the caller fits
            # with a different cadence.
            out = VEPColumn(n_sites=1, dt_days=model.dt_days)
            for name in ("log_ske", "log_skv", "log_tau", "h_pc0"):
                src = getattr(model, name).detach().cpu()
                getattr(out, name).copy_((src * w.cpu()).sum().reshape(1))
            p = out(h[held: held + 1].cpu())[0]
        m = mask[held].cpu()
        p_site = p[m].numpy()
        t_site = obs[held].cpu()[m].numpy()
        preds.append(p_site)
        targets.append(t_site)
        ss_res_i = float(((t_site - p_site) ** 2).sum())
        ss_tot_i = float(((t_site - t_site.mean()) ** 2).sum())
        per_site[held] = {
            "r2": 1.0 - ss_res_i / max(ss_tot_i, 1e-12),
            "loss": float(((t_site - p_site) ** 2).mean()) if len(t_site) else float("nan"),
        }
    pred = np.concatenate(preds)
    obsv = np.concatenate(targets)
    ss_res = float(((obsv - pred) ** 2).sum())
    ss_tot = float(((obsv - obsv.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "n_sites": n, "per_site": per_site}


def loso_shared(h: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, **kw) -> dict:
    """Structural analogue of the pooled single-``Sk`` baseline for the VEP column.

    ``loso`` fits ``VEPColumn(n_sites=13)`` per fold -- 4 params x 13 sites = 52 free
    parameters -- then collapses them post hoc with a weighted mean of log-parameters.
    Diagnostics show that mean is not an estimate of the rheology: 6/14 folds give ZERO
    gradient on ``log_skv``/``log_tau`` (those sites just carry their initialization
    value into the average) and 4/14 pin ``log_tau`` at its clamp bound.

    This function instead fits ONE global 4-vector jointly against every training site's
    rows at once (``VEPColumn(n_sites=1)`` broadcasts against the (n_train, T) batch, so
    it is equivalent to fitting against the concatenation of all training sites), then
    predicts the held-out site with that same global vector. 4 parameters over ~1200
    training cells, so identifiability is not in question by construction.

    This is the missing structural analogue of the pooled single-``Sk`` baseline. Only if
    THIS loses to single-``Sk`` has the rheology itself been tested -- rather than the
    per-site-fit-then-average transfer rule ``loso`` tests.

    Residuals are pooled across folds into one R2 exactly as ``loso`` does, so the two
    numbers are directly comparable. Also returns ``per_site``, a per-fold diagnostic of
    {site_index: {r2, loss}}.
    """
    n = h.shape[0]
    preds, targets, per_site = [], [], {}
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        model, _ = fit_column(h[keep], obs[keep], mask[keep], n_sites=1, **kw)
        model = model.cpu()
        with torch.no_grad():
            p = model(h[held: held + 1].cpu())[0]
        m = mask[held].cpu()
        p_site = p[m].numpy()
        t_site = obs[held].cpu()[m].numpy()
        preds.append(p_site)
        targets.append(t_site)
        ss_res_i = float(((t_site - p_site) ** 2).sum())
        ss_tot_i = float(((t_site - t_site.mean()) ** 2).sum())
        per_site[held] = {
            "r2": 1.0 - ss_res_i / max(ss_tot_i, 1e-12),
            "loss": float(((t_site - p_site) ** 2).mean()) if len(t_site) else float("nan"),
        }
    pred = np.concatenate(preds)
    obsv = np.concatenate(targets)
    ss_res = float(((obsv - pred) ** 2).sum())
    ss_tot = float(((obsv - obsv.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "n_sites": n, "per_site": per_site}


def _loso_sk_pooled(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """LOSO for the single-Sk baseline: pooled through-origin fit on the other sites,
    predict Sk * D_held, pool residuals into one R2 -- exactly as ``loso`` does for VEP."""
    names = list(pairs)
    preds, obsv = [], []
    for held in names:
        train = {n: pairs[n] for n in names if n != held}
        fit = calibrate_sk_from_pairs(train)
        d_held, c_held = pairs[held]
        preds.append(fit["sk"] * np.asarray(d_held, dtype="float64"))
        obsv.append(np.asarray(c_held, dtype="float64"))
    pred = np.concatenate(preds)
    ob = np.concatenate(obsv)
    ss_res = float(((ob - pred) ** 2).sum())
    ss_tot = float(((ob - ob.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


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
    rows_h, rows_o, rows_m, rows_x, names = [], [], [], [], []
    for _, r in stations.iterrows():
        name = r["sub_id"]
        if name not in comp:
            continue
        h_site = idw_interp(np.array([[r["x"], r["y"]]], dtype="float64"), wxy, H)[0]
        # tolerance=45D: MLCW is quarterly, so an unbounded nearest-reindex back/forward-fills
        # across the whole monthly head record. A mask cell must mean "an observation exists
        # near this date", not "the last one still holds".
        c = comp[name].reindex(dates, method="nearest", tolerance=pd.Timedelta("45D"))
        ok = c.notna().to_numpy()
        if ok.sum() < 24:
            continue
        rows_h.append(h_site)
        rows_o.append(np.nan_to_num(c.to_numpy(dtype="float64")))
        rows_m.append(ok)
        rows_x.append(float(r["x"]))
        names.append(name)

    h = torch.tensor(np.stack(rows_h), dtype=torch.float32)
    obs = torch.tensor(np.stack(rows_o), dtype=torch.float32)
    mask = torch.tensor(np.stack(rows_m))
    n_cells = int(mask.sum().item())

    # anchor each site at its first OBSERVED sample; column 0 may be masked out
    first = mask.float().argmax(dim=1)
    anchor = obs.gather(1, first.unsqueeze(1))
    obs = obs - anchor

    _, info = fit_column(h, obs, mask, epochs=args.epochs, lr=args.lr)
    # LOSO folds get the SAME epoch budget as the in-sample fit. Diagnostics showed
    # parameters still moving between 500 and 6000 epochs, so a quartered budget
    # (epochs // 4) under-trained every held-out fold and made the gate unreliable.
    gate = loso(h, obs, mask, epochs=args.epochs, lr=args.lr)
    vep_loso = gate["r2"]
    # Structural analogue of the pooled single-Sk baseline (see loso_shared's docstring):
    # one global 4-vector fit jointly to all training sites, not fit-per-site-then-averaged.
    shared_gate = loso_shared(h, obs, mask, epochs=args.epochs, lr=args.lr)
    vep_shared_loso = shared_gate["r2"]

    # Like-for-like single-Sk baselines on the IDENTICAL arrays (same sites, same mask,
    # same anchoring) -- not the README's in-sample table.
    h_np = h.numpy()
    runmin = np.minimum.accumulate(h_np, axis=1)
    D = np.clip(h_np[:, :1] - runmin, a_min=0.0, a_max=None)
    obs_np = obs.numpy()
    mask_np = mask.numpy().astype(bool)
    pairs = {names[i]: (D[i][mask_np[i]], obs_np[i][mask_np[i]]) for i in range(len(names))}

    single = calibrate_sk_from_pairs(pairs)
    sk_insample = single["r2"]
    sk_loso = _loso_sk_pooled(pairs)
    dist = dict(zip(names, rows_x, strict=True))
    sk_coast_gate = loso_sk_regression(pairs, dist)
    sk_coast_loso = sk_coast_gate["r2"]

    print(f"sites={len(names)}  cells_within_tolerance={n_cells}  "
          f"in-sample loss={info['loss']:.3e}")
    print(f"sk_insample={sk_insample:+.3f}  sk_loso={sk_loso:+.3f}  "
          f"sk_coast_loso={sk_coast_loso:+.3f}  vep_loso={vep_loso:+.3f}  "
          f"vep_shared_loso={vep_shared_loso:+.3f}")
    print(f"GATE (VEP LOSO vs single-Sk LOSO, identical arrays): "
          f"{'PASS' if vep_loso > sk_loso else 'FAIL'}")
    print(f"GATE (shared-parameter VEP LOSO vs single-Sk LOSO -- the rheology itself): "
          f"{'PASS' if vep_shared_loso > sk_loso else 'FAIL'}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage2_vep_mlcw.csv")
    pd.DataFrame([{
        "n_sites": len(names),
        "n_cells": n_cells,
        "loss": info["loss"],
        "sk_insample": sk_insample,
        "sk_loso": sk_loso,
        "sk_coast_loso": sk_coast_loso,
        "vep_loso": vep_loso,
        "vep_shared_loso": vep_shared_loso,
    }]).to_csv(path, index=False)
    print(f"wrote {path}")

    per_site_path = os.path.join(args.out, "stage2_vep_per_site.csv")
    pd.DataFrame([
        {"site_index": i, "site_name": names[i], "r2": v["r2"], "loss": v["loss"]}
        for i, v in sorted(gate["per_site"].items())
    ]).to_csv(per_site_path, index=False)
    print(f"wrote {per_site_path}")


if __name__ == "__main__":
    main()
