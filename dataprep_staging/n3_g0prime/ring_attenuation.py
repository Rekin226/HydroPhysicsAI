"""Model-only N note of N3.G0prime.v1: the expected OLS attenuation of the seasonal
compaction-per-metre slope at the ring-site geometry (judge rank 14, 'model-only step').

The page prints the rings' seasonal slope (3.4 mm/m) beside the column's (9.3 mm/m) from
twin.mechanism (rings to 2025, after the fact). The ring figure regresses ring compaction on a
NEST's head anomaly (wells within 1 km), the column figure regresses the model column on the
model's own layer-mean head at the ring cell (no head error). This script asks only how much an
errors-in-variables attenuation of that assumed size would shrink the column's figure if it
were measured the rings' way: the model's own hindcast at the ring cells
(twin.mechanism.ring_cells and model_ring_series), plus N3 G0's nest-error grid (AR(1) on the
head level, sd {0.75, 1.5} m, phi {0.5, 0.9}; g0_n3.ar1; ASSUMED for the pile nests at 1.5 km,
never measured at the ring nests, which are a multi-well mean within 1 km), through the same
estimator (twin.mechanism.seasonal_slope: lag 0 and the best of lags 0-2, median over sites).
No ring value, no nest head and no observation is read: the inputs are a frozen forward archive
and the ring sites' coordinates.

Output (written once): per (sd, phi) the median noise-free and noisy column slopes (mm/m) and
their ratio lambda. Page use (declared in N3.G0prime.v1): an N note printed beside the
UNCHANGED 3.4 / 9.3 figure and the 0.92 pile line; never a corrected figure, never a status.

Refuses unless N3.G0prime.v1 is registered, intact and its frozen code is this code (the same
check as g0prime.py). Written once; a retry (e.g. the wrong forward archive) goes through
``run_g0prime.sh archive-rings '<reason>'``, which never touches G0'.

Usage (repository root, the lead, after the declaration; CPU, ~1-3 min):
    python -I dataprep_staging/n3_g0prime/ring_attenuation.py \
        --forward results/twin_forward/datum_gate_innov.npz \
        --sites <csv with columns site, x, y (TWD97 m)> \
        --out results/dataprep/reports/N3.G0prime.v1/ring_attenuation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0p_lib as P  # noqa: E402
import g0prime as GP  # noqa: E402  (the declaration check)

G0 = P.G0
RESULT = "result.json"
REPS = 200
SEED = 20261117
SITE_COLUMNS = ("site", "x", "y")


def read_sites(path: Path) -> pd.DataFrame:
    """Only the coordinate columns are read (usecols): no ring value can enter."""
    df = pd.read_csv(path, usecols=list(SITE_COLUMNS))
    return df.dropna().drop_duplicates("site").reset_index(drop=True)


def model_series(npz_path: Path, sites: pd.DataFrame) -> tuple[dict, dict]:
    from hydrophysics.twin import mechanism as M

    with np.load(npz_path, allow_pickle=False) as z:
        cells = M.ring_cells(z, sites[["x", "y"]].to_numpy(float))
        C, D, _ = M.model_ring_series(z, cells, sites["site"].tolist())
    return C, D


def attenuation(C: dict, D: dict, reps: int = REPS, seed: int = SEED) -> dict:
    from hydrophysics.twin.mechanism import seasonal_slope

    names = sorted(C)
    if not names:
        raise ValueError("no ring site lies on the model grid")
    true0 = {s: seasonal_slope(C[s], D[s])["by_lag"][0]["slope"] for s in names}
    trueb = {s: seasonal_slope(C[s], D[s])["slope"] for s in names}
    out = {"n_sites": len(names), "sites": names, "reps": reps, "seed": seed,
           "noise_free": {"median_lag0_mm_per_m": 1000 * float(np.nanmedian(list(
               true0.values()))), "median_best_lag_mm_per_m": 1000 * float(np.nanmedian(list(
                   trueb.values())))}, "cells": []}
    kids = iter(np.random.SeedSequence(seed).spawn(len(G0.SD_GRID) * len(G0.PHI_GRID)))
    for sd in G0.SD_GRID:
        for phi in G0.PHI_GRID:
            rng = np.random.default_rng(next(kids))
            n0, nb = {s: [] for s in names}, {s: [] for s in names}
            for _ in range(reps):
                for s in names:
                    e = pd.Series(G0.ar1(len(D[s]), sd, phi, rng), index=D[s].index)
                    r = seasonal_slope(C[s], D[s] + e)
                    n0[s].append(r["by_lag"][0]["slope"])
                    nb[s].append(r["slope"])
            m0 = float(np.nanmedian([np.nanmean(n0[s]) for s in names]))
            mb = float(np.nanmedian([np.nanmean(nb[s]) for s in names]))
            f0 = out["noise_free"]["median_lag0_mm_per_m"] / 1000
            fb = out["noise_free"]["median_best_lag_mm_per_m"] / 1000
            out["cells"].append({
                "sd": sd, "phi": phi, "noisy_median_lag0_mm_per_m": 1000 * m0,
                "noisy_median_best_lag_mm_per_m": 1000 * mb,
                "lambda_lag0": m0 / f0 if f0 else None,
                "lambda_best_lag": mb / fb if fb else None,
                "lambda_lag0_site_median": float(np.nanmedian(
                    [np.nanmean(n0[s]) / true0[s] for s in names if true0[s]]))})
    lam = [c["lambda_lag0"] for c in out["cells"] if c["lambda_lag0"] is not None]
    out["lambda_range_lag0"] = [float(min(lam)), float(max(lam))] if lam else None
    return out


def note(res: dict) -> str:
    r = res.get("lambda_range_lag0")
    nf = res["noise_free"]["median_lag0_mm_per_m"]
    if not r:
        return "ring attenuation note: not computable (no on-grid ring site)"
    return (f"N (model-only): if the ring nests had the head error N3 assumed for the pile "
            f"nests (AR(1), sd 0.75-1.5 m, persistence 0.5-0.9; assumed, not measured at the "
            f"ring nests), the column's own seasonal slope at the ring sites, measured the "
            f"rings' way, would read lambda = {r[0]:.2f}-{r[1]:.2f} x its noise-free value "
            f"({nf:.1f} mm/m -> {nf * r[0]:.1f}-{nf * r[1]:.1f} mm/m). The rings' and the "
            "column's figures are printed unchanged; no figure is corrected and no status "
            "moves.")


def run(forward: Path, sites: Path, out: Path, reps: int = REPS, *, prereg_dir=None,
        check_declared: bool = True) -> dict:
    out = Path(out)
    if (out / RESULT).exists():
        raise FileExistsError(f"{out / RESULT} exists: written once (archive-rings to retry)")
    if check_declared:
        rec, _ = GP.require_declared(prereg_dir)
    else:
        rec = {"note": "declaration check disabled (tests only; a real run checks it)"}
    t0 = time.time()
    st = read_sites(sites)
    C, D = model_series(forward, st)
    res = attenuation(C, D, reps=reps)
    res.update({"id": P.ID, "part": "ring_attenuation_note (N)",
                "declaration_index_record": rec,
                "noise_grid_status": "assumed for the pile nests (N3 G0); not measured at "
                                     "the ring nests",
                "forward": {"path": Path(forward).name, "sha256": P.sha256(forward)},
                "sites_file": {"path": Path(sites).name, "sha256": P.sha256(sites),
                               "columns_read": list(SITE_COLUMNS)},
                "code": {f"{P.E}/ring_attenuation.py": P.sha256(Path(__file__)),
                         f"{P.E}/g0p_lib.py": P.sha256(HERE / "g0p_lib.py"),
                         "hydrophysics/twin/mechanism.py":
                             P.sha256(REPO / "hydrophysics/twin/mechanism.py")},
                "note": note(res), "seconds": round(time.time() - t0, 1),
                "reads": "a frozen forward archive (hindcast at the ring cells) and the ring "
                         "sites' coordinates; no ring value, no observation"})
    out.mkdir(parents=True, exist_ok=True)
    res = P.jsonable(res)
    with open(out / RESULT, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--sites", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = run(a.forward, a.sites, a.out)
    print(res["note"])
    print("result sha256:", P.sha256(Path(a.out) / RESULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
