"""Recompute the fair held-out-years verdict from finished temporal screens.

    python -m hydrophysics.twin.rescore_temporal results/twin_runs/temporal_ref10 \\
        results/twin_runs/temporal_predictions.npz ...

Each argument is a run directory holding ``stage3_temporal_pred.npz`` or an ``.npz`` file
directly. A file with a ``pred`` array is one screen; a file without one (the older
``temporal_predictions.npz`` bundle) is scored once per ``(W, T)`` array that is not
``obs``/``clim``/``persist``/``T_fit``/``sids`` (the 36-column ``*_restart`` arrays cannot
carry a fitted-period datum and are skipped, with a note).

The dumps store the BACK-FILLED record as ``obs``. Runs from 2026-09-23 on also store
``obs_raw``; for older ones the raw record is reloaded from the data cache
(``heads.build_head_field`` with ``calibrate_flow.DEFAULT_PATHS``) and matched to the dump
by ``sids`` -- or, without ``sids``, by content -- and every raw observation is checked to
equal the dump's ``obs`` before anything is scored.

Nothing is written into the run directories (they may still be in use): one row per screen
goes to ``--out`` (default ``results/twin/temporal_fair_rescore.csv``). CPU and numpy only.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .drift_diag import fair_temporal_verdict

NON_PRED_KEYS = ("obs", "clim", "persist", "T_fit", "sids", "obs_raw")
LEGACY_COLS = ("verdict", "rmse_ratio", "r2_shape_model", "r2_shape_clim", "rmse_model_m",
               "rmse_clim_m")


def load_raw_record(paths: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """The QC'd monthly head record ``(W_all, 132)`` with NaN where never observed, and
    its station ids -- the same ``build_head_field`` call ``calibrate_flow.main`` makes."""
    from .calibrate_flow import DEFAULT_PATHS
    from .heads import build_head_field

    P = dict(DEFAULT_PATHS)
    P.update(paths or {})
    stn = pd.read_parquet(P["stations"])
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(P["wells_dir"], stn)
    return np.asarray(hf.heads, dtype="float64"), [str(s) for s in hf.sids]


def match_raw(obs_filled: np.ndarray, raw_all: np.ndarray, raw_sids: list[str],
              sids: np.ndarray | None = None, tol: float = 1e-6) -> np.ndarray:
    """Rows of ``raw_all`` (target columns 1..) aligned to the dump's ``obs`` rows.

    With ``sids`` the rows are looked up by id; without, each dump row is matched to the
    unique raw row whose observed months equal it. Either way every raw observation must
    equal the dump's value to ``tol`` (the dump is the back-filled copy of the raw record),
    or this raises -- a mismatched record must never be scored."""
    raw = raw_all[:, 1:]
    W, T = obs_filled.shape
    if raw.shape[1] != T:
        raise ValueError(f"raw record has {raw.shape[1]} target months, dump has {T}")
    if sids is not None:
        pos = {s: i for i, s in enumerate(raw_sids)}
        missing = [str(s) for s in sids if str(s) not in pos]
        if missing:
            raise ValueError(f"{len(missing)} dump well(s) not in the raw record: {missing[:5]}")
        rows = raw[[pos[str(s)] for s in sids]]
    else:
        rows = np.empty_like(obs_filled)
        fin = np.isfinite(raw)
        for w in range(W):
            d = np.where(fin, np.abs(raw - obs_filled[w][None, :]), 0.0).max(axis=1)
            hit = np.where((d <= tol) & (fin.sum(axis=1) > 0))[0]
            if hit.size != 1:
                raise ValueError(f"dump row {w}: {hit.size} raw rows match by content")
            rows[w] = raw[hit[0]]
    fin = np.isfinite(rows)
    bad = np.abs(np.where(fin, rows - obs_filled, 0.0)).max() if fin.any() else 0.0
    if not np.isfinite(bad) or bad > tol:
        raise ValueError(f"raw record differs from the dump's obs by {bad:g} m")
    return rows


def _screens(path: str) -> tuple[str, str]:
    if os.path.isdir(path):
        return os.path.join(path, "stage3_temporal_pred.npz"), os.path.basename(
            os.path.normpath(path))
    return path, os.path.splitext(os.path.basename(path))[0]


def rescore(paths: list[str], raw_loader=load_raw_record) -> pd.DataFrame:
    """One row per screen found under ``paths``; ``raw_loader`` is called at most once."""
    cache: dict = {}
    rows = []
    for p in paths:
        f, label = _screens(p)
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        obs = np.asarray(d["obs"], dtype="float64")
        T_fit = int(d["T_fit"])
        if "obs_raw" in d:
            raw, src = np.asarray(d["obs_raw"], dtype="float64"), "dump"
        else:
            if "raw" not in cache:
                cache["raw"] = raw_loader()
            raw = match_raw(obs, *cache["raw"], sids=d.get("sids"))
            src = "reloaded"
        preds = ({label: d["pred"]} if "pred" in d else
                 {f"{label}:{k}": v for k, v in d.items()
                  if k not in NON_PRED_KEYS and np.shape(v) == obs.shape})
        skipped = [k for k, v in d.items()
                   if k not in NON_PRED_KEYS and k != "pred" and np.shape(v) != obs.shape]
        legacy = {}
        csv = os.path.join(os.path.dirname(f), "stage3_temporal.csv")
        if "pred" in d and os.path.exists(csv):
            row0 = pd.read_csv(csv).iloc[0]
            legacy = {f"legacy_{c}": row0[c] for c in LEGACY_COLS if c in row0}
        for name, pred in preds.items():
            res = fair_temporal_verdict(np.asarray(pred, dtype="float64"), raw, T_fit)
            rows.append({"screen": name, "source": f, "raw_obs": src, "T_fit": T_fit,
                         "n_wells": obs.shape[0], "n_backfilled_cells": int(
                             (~np.isfinite(raw)).sum()),
                         **res, **legacy,
                         "skipped_arrays": ";".join(skipped)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="run dirs with stage3_temporal_pred.npz, or npz")
    ap.add_argument("--out", default="results/twin/temporal_fair_rescore.csv")
    args = ap.parse_args(argv)
    df = rescore(args.paths)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["screen", "n_wells_scored", "rmse_raw_model_m", "rmse_datum_model_m",
            "level_err_mean_abs_m", "rmse_clim_m", "rmse_clim_trend_m", "rmse_persist_m",
            "best_baseline", "rmse_ratio_fair", "r2_shape_datum_model", "r2_shape_clim",
            "verdict_fair", "legacy_verdict"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[[c for c in cols if c in df]].to_string(index=False,
                                                          float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
