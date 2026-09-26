"""Merge a parallel k-fold into the outputs of a sequential run (2026-09-26).

    python -m hydrophysics.twin.calibrate_flow <recipe> --fit-only --out DIR
    python -m hydrophysics.twin.calibrate_flow <recipe> --only-fold K --out DIR   # K = 0..n-1
    python -m hydrophysics.twin.merge_folds DIR

reads ``stage3_fit_summary.json`` (the ``--fit-only`` job), ``stage3_theta.json`` and every
``stage3_fold{K}.json`` and writes what the sequential run writes after its folds:
``stage3_flow.csv``, ``stage3_fold_thetas.json``, ``stage3_kfold_wells.csv``,
``stage3_per_entry.npz`` (when the folds ran with ``--dump-predictions``) and the gate
stamped into ``stage3_theta.json``. It pools through the same code as the sequential run
(``calibrate_flow.summarise_kfold`` / ``write_kfold_outputs``), so every metric is
identical to a sequential run on the same device and inductor state; only the timing
columns differ (``gate_time_s`` is the SUM of the fold jobs' times, the compute a
sequential gate would have spent).

It refuses rather than guesses: a missing fold, folds of different partitions (n_folds,
seed, wells) or a fold run with a configuration different from the full fit's
(``stage3_flow.csv`` would describe a recipe no fold used) is an error. ``cg_nonconverged``
is the sum and ``cg_worst_residual`` the maximum over the fit and the folds, as the
sequential counter accumulates them.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .calibrate_flow import (
    FIT_SUMMARY,
    FOLD_FILE,
    fold_record_from_json,
    summarise_kfold,
    write_kfold_outputs,
    write_per_entry_npz,
)
from .kfold_scores import format_verdicts


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def merge(run_dir: str, log=print) -> dict:
    """Merge ``run_dir``'s fold files (module docstring); returns the pooled gate."""
    fs_path = os.path.join(run_dir, FIT_SUMMARY)
    if not os.path.exists(fs_path):
        raise FileNotFoundError(f"{fs_path}: run the full fit with --fit-only into "
                                f"{run_dir} first")
    fs = _load(fs_path)
    first = os.path.join(run_dir, FOLD_FILE.format(k=0))
    if not os.path.exists(first):
        raise FileNotFoundError(f"{first}: no fold files in {run_dir}")
    n_folds = int(_load(first)["n_folds"])
    missing = [k for k in range(n_folds)
               if not os.path.exists(os.path.join(run_dir, FOLD_FILE.format(k=k)))]
    if missing:
        raise FileNotFoundError(f"{run_dir}: fold file(s) missing for fold(s) {missing} of "
                                f"{n_folds}")
    folds = [_load(os.path.join(run_dir, FOLD_FILE.format(k=k))) for k in range(n_folds)]
    ref = folds[0]
    for k, f in enumerate(folds):
        if int(f["fold"]) != k:
            raise ValueError(f"{FOLD_FILE.format(k=k)} holds fold {f['fold']}")
        for key in ("n_folds", "seed", "n_wells", "n_sites", "colocation_rate", "sids",
                    "dump_predictions"):
            if json.dumps(f[key]) != json.dumps(ref[key]):     # NaN-safe equality
                raise ValueError(f"fold {k}: {key}={f[key]!r} differs from fold 0's "
                                 f"{ref[key]!r} -- not one partition")
        if f["cfg"] != fs["cfg"]:
            diff = sorted(key for key in set(f["cfg"]) | set(fs["cfg"])
                          if f["cfg"].get(key) != fs["cfg"].get(key))
            raise ValueError(f"fold {k} ran a different configuration from the full fit "
                             f"({FIT_SUMMARY}): {diff}")
    if ref["sids"] != fs["sids"]:
        raise ValueError("the folds and the full fit used different wells")
    wells_csv = os.path.join(run_dir, "stage3_wells.csv")
    if os.path.exists(wells_csv):
        csv_sids = pd.read_csv(wells_csv, dtype={"sid": str})["sid"].tolist()
        if csv_sids != fs["sids"]:
            raise ValueError(f"{wells_csv} does not list the wells the folds used")
    per_fold = [fold_record_from_json(f["record"]) for f in folds]
    gate = summarise_kfold(per_fold, n_wells=int(ref["n_wells"]), n_folds=n_folds,
                           n_sites=int(ref["n_sites"]),
                           colocation_rate=float(ref["colocation_rate"]))
    cg = (int(fs["cg_nonconverged"]) + sum(int(f["cg_nonconverged"]) for f in folds),
          max([float(fs["cg_worst_residual"])]
              + [float(f["cg_worst_residual"]) for f in folds]))
    t_gate = float(sum(float(f["gate_time_s"]) for f in folds))
    if ref["dump_predictions"]:
        dump = os.path.join(run_dir, "stage3_per_entry.npz")
        write_per_entry_npz(dump, per_fold)
        log(f"wrote {dump}")
    path = write_kfold_outputs(run_dir, fs["cfg"], fs["fit"], gate, cg,
                               float(fs["fit_time_s"]), t_gate, fs["sids"])
    log(f"merged {n_folds} folds (seed {ref['seed']}) of {run_dir}")
    for f in per_fold:
        log(f"  fold {f['fold']}: n_held={f['n_held']} R2 {f['r2_kfold']:+.3f} vs IDW "
            f"{f['r2_idw']:+.3f}")
    log(format_verdicts(gate))
    log(f"wrote {path}")
    return gate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="merge --only-fold jobs into stage3_flow.csv")
    ap.add_argument("run_dir")
    merge(ap.parse_args(argv).run_dir)


if __name__ == "__main__":
    main()
