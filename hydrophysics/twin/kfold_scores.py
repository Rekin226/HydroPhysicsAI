"""K-fold spatial ANOMALY verdict, beside the legacy absolute one (pre-registered 2026-09-26).

    python -m hydrophysics.twin.kfold_scores results/twin_runs/stage3_spreadL_gate [...]

rescores a run's ``stage3_per_entry.npz`` (``calibrate_flow --dump-predictions``) on this
verdict and writes ``stage3_kfold_anom.csv`` beside it; ``calibrate_flow`` and
``merge_folds`` compute it for every k-fold run and write it into ``stage3_flow.csv``.

**The rule.** For each held-out well, over its scored months (observation, model and IDW
all finite), each series is taken minus its OWN mean over those months:
``obs - mean(obs)``, ``pred - mean(pred)``, ``idw - mean(idw)``. The pooled anomaly R2
of the model and of the IDW baseline are computed over every held-out well of every
fold, and

    verdict_anom_kfold = PASS  iff  r2_anom_kfold - r2_anom_idw >= KFOLD_ANOM_MARGIN_MIN

(``KFOLD_ANOM_MARGIN_MIN = 0``; the legacy margin ``r2_kfold - r2_idw`` has the same
shape, on the absolute heads). The per-well median anomaly R2 of both is reported beside
it, not ruled on.

**Why.** A model with a per-well datum (``calibrate_flow --well-datum fit``) states that a
well's absolute level carries a sub-grid offset the 1 km aquifer cannot represent (30 m
vertical differences inside single well nests). A held-out well's datum is unknowable --
it is fitted from that well's own record, which the k-fold hides -- so the legacy gate
scores such a model on exactly the part it declares outside the physics. The twin is
driven by head CHANGES (the compaction column responds to drawdown, the policy verdict
to a head difference between scenarios), so the spatial test that matches what a datum
model claims is whether it predicts an unseen well's departures from its own level
better than interpolating the neighbours' series. The legacy absolute verdict is still
computed, written and printed for every run, and it is never replaced by this one.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

KFOLD_ANOM_MARGIN_MIN = 0.0
PER_ENTRY = "stage3_per_entry.npz"
ANOM_KEYS = ("r2_anom_kfold", "r2_anom_idw", "margin_anom_kfold", "r2_anom_well_median_kfold",
             "r2_anom_well_median_idw", "n_wells_anom", "verdict_anom_kfold")


def _r2(pred: np.ndarray, obs: np.ndarray) -> float:
    """Pooled R2 over the finite pairs -- the same formula as ``calibrate_flow._r2``
    (kept here so this module and its CLI need numpy only)."""
    finite = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[finite], obs[finite]
    if obs.size == 0:
        return float("nan")
    ss_res = float(((obs - pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def scored_mask(pred: np.ndarray, idw: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """``(N, T)`` months scored for both the model and IDW: all three finite."""
    return np.isfinite(pred) & np.isfinite(idw) & np.isfinite(obs)


def own_mean_anomaly(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each row minus its own mean over ``mask``; NaN outside ``mask`` (and for a row with
    no scored month)."""
    x = np.asarray(x, dtype="float64")
    n = mask.sum(axis=1, keepdims=True)
    s = np.where(mask, x, 0.0).sum(axis=1, keepdims=True)
    mean = np.divide(s, n, out=np.full_like(s, np.nan), where=n > 0)
    return np.where(mask, x - mean, np.nan)


def _per_row_r2(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """R2 per row; NaN for a row with fewer than two finite months or no variance."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = np.isfinite(pred[i]) & np.isfinite(obs[i])
        if m.sum() >= 2 and float(np.var(obs[i, m])) > 1e-12:
            out[i] = _r2(pred[i, m], obs[i, m])
    return out


def kfold_anomaly_scores(pred: np.ndarray, idw: np.ndarray, obs: np.ndarray) -> dict:
    """The anomaly verdict (module docstring) over held-out rows ``(N, T)`` pooled from
    every fold. Returns the ``ANOM_KEYS``."""
    pred, idw, obs = (np.asarray(a, dtype="float64") for a in (pred, idw, obs))
    mask = scored_mask(pred, idw, obs)
    a_obs, a_pred, a_idw = (own_mean_anomaly(a, mask) for a in (obs, pred, idw))
    r2_m, r2_i = _r2(a_pred, a_obs), _r2(a_idw, a_obs)
    w_m, w_i = _per_row_r2(a_pred, a_obs), _per_row_r2(a_idw, a_obs)
    ok = np.isfinite(w_m) & np.isfinite(w_i)
    margin = r2_m - r2_i
    return {"r2_anom_kfold": r2_m, "r2_anom_idw": r2_i, "margin_anom_kfold": margin,
            "r2_anom_well_median_kfold": float(np.median(w_m[ok])) if ok.any() else np.nan,
            "r2_anom_well_median_idw": float(np.median(w_i[ok])) if ok.any() else np.nan,
            "n_wells_anom": int(ok.sum()),
            "verdict_anom_kfold": ("PASS" if np.isfinite(margin)
                                   and margin >= KFOLD_ANOM_MARGIN_MIN else "FAIL")}


def per_well_metrics(pred: np.ndarray, idw: np.ndarray, obs: np.ndarray) -> dict:
    """``(N,)`` arrays per held-out row: absolute R2/RMSE (each on its own finite months,
    as the legacy pooled R2) and anomaly R2 (on the months scored for both)."""
    pred, idw, obs = (np.asarray(a, dtype="float64") for a in (pred, idw, obs))
    mask = scored_mask(pred, idw, obs)
    a_obs, a_pred, a_idw = (own_mean_anomaly(a, mask) for a in (obs, pred, idw))

    def _rmse(p):
        d = np.where(np.isfinite(p) & np.isfinite(obs), p - obs, np.nan)
        n = np.isfinite(d).sum(axis=1)
        s = np.nansum(d ** 2, axis=1)
        return np.divide(s, n, out=np.full(len(n), np.nan), where=n > 0) ** 0.5

    return {"n_months": mask.sum(axis=1).astype("int64"),
            "r2_model": _per_row_r2(pred, obs), "r2_idw": _per_row_r2(idw, obs),
            "r2_anom_model": _per_row_r2(a_pred, a_obs),
            "r2_anom_idw": _per_row_r2(a_idw, a_obs),
            "rmse_model_m": _rmse(pred), "rmse_idw_m": _rmse(idw)}


def format_verdicts(gate: dict) -> str:
    """Both k-fold verdicts, legacy first, for the run logs."""
    legacy = "PASS" if gate["r2_kfold"] > gate["r2_idw"] else "FAIL"
    return (f"GATE (k-fold, absolute heads, legacy): {legacy} -- R2 {gate['r2_kfold']:+.3f} vs "
            f"IDW {gate['r2_idw']:+.3f}\n"
            f"GATE (k-fold, own-mean anomalies): {gate['verdict_anom_kfold']} -- anomaly R2 "
            f"{gate['r2_anom_kfold']:+.3f} vs IDW {gate['r2_anom_idw']:+.3f} (margin "
            f"{gate['margin_anom_kfold']:+.3f}, rule >= {KFOLD_ANOM_MARGIN_MIN:g}); per-well "
            f"median {gate['r2_anom_well_median_kfold']:+.3f} vs "
            f"{gate['r2_anom_well_median_idw']:+.3f} on {gate['n_wells_anom']} wells")


def rescore(run_dir: str, write: bool = True) -> dict:
    """Score a finished run's ``stage3_per_entry.npz`` on both verdicts; with ``write``,
    write ``stage3_kfold_anom.csv`` (one row) beside it. The legacy R2 is recomputed from
    the dump and checked against ``stage3_flow.csv`` when that exists."""
    path = os.path.join(run_dir, PER_ENTRY)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no per-entry dump (the run needs "
                                "--dump-predictions)")
    with np.load(path) as z:
        pred, idw, obs = z["pred"], z["idw"], z["obs"]
        n_folds = int(np.unique(z["fold"]).size)
    out = {"run": run_dir, "n_entries": int(pred.shape[0]), "n_folds": n_folds,
           "r2_kfold": _r2(pred, obs), "r2_idw": _r2(idw, obs)}
    out["verdict_kfold"] = "PASS" if out["r2_kfold"] > out["r2_idw"] else "FAIL"
    out.update(kfold_anomaly_scores(pred, idw, obs))
    csv = os.path.join(run_dir, "stage3_flow.csv")
    if os.path.exists(csv):
        row = pd.read_csv(csv).iloc[-1]
        out["r2_kfold_recorded"] = float(row["r2_kfold"])
        out["r2_idw_recorded"] = float(row["r2_idw"])
        if (abs(out["r2_kfold"] - out["r2_kfold_recorded"]) > 1e-9
                or abs(out["r2_idw"] - out["r2_idw_recorded"]) > 1e-9):
            print(f"WARNING {run_dir}: the dump's legacy R2 differs from stage3_flow.csv "
                  "(was the dump written by a different run?)", flush=True)
    if write:
        pd.DataFrame([out]).to_csv(os.path.join(run_dir, "stage3_kfold_anom.csv"),
                                   index=False)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="rescore k-fold dumps on the anomaly verdict")
    ap.add_argument("runs", nargs="+", help="run directories holding stage3_per_entry.npz")
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not write stage3_kfold_anom.csv")
    args = ap.parse_args(argv)
    for d in args.runs:
        out = rescore(d, write=not args.no_write)
        print(f"{d} ({out['n_entries']} held-out entries, {out['n_folds']} folds)")
        print("  " + format_verdicts(out).replace("\n", "\n  "))
        if not args.no_write:
            print(f"  wrote {os.path.join(d, 'stage3_kfold_anom.csv')}")
        print(json.dumps({k: out[k] for k in ANOM_KEYS}), flush=True)


if __name__ == "__main__":
    main()
