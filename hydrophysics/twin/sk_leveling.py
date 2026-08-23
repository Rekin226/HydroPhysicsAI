"""Stage 1: refit the algebraic Sk coupling on the leveling network instead of 14 MLCW sites.

The README reports leave-one-site-out R2 of -0.28 to -2.40 for every head->subsidence
coupling tried on 14 multi-layer compaction wells, and attributes the failure to spatial
sparsity. The WRA leveling network provides ~40x more sites over the same window. If the
gate is still negative here, the wall is the model form, not the sampling -- which is what
justifies the visco-elasto-plastic column in Tasks 4-7.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import GWData, load_dataset
from ..subsidence import (
    calibrate_sk_from_pairs,
    idw_interp,
    loso_sk_regression,
    monthly_heads,
    well_xy,
)
from . import leveling


def build_pairs(data: GWData, sub: dict[str, pd.Series],
                xy: dict[str, tuple[float, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{sid -> (cumulative drawdown, cumulative subsidence)} sampled at survey dates.

    Heads are interpolated to each site by IDW from the observed wells, aggregated to
    month-end, then read at the nearest month to each survey. Both series are re-zeroed to
    the site's first survey inside the head record.
    """
    H, dates = monthly_heads(data)
    wxy = well_xy(data)
    lo, hi = dates[0], dates[-1]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid, s in sub.items():
        if sid not in xy:
            continue
        keep = s.index[(s.index >= lo) & (s.index <= hi)]
        if len(keep) < 3:
            continue
        h = idw_interp(np.array([list(xy[sid])], dtype="float64"), wxy, H)[0]
        hser = pd.Series(h, index=dates)
        draw = (hser.iloc[0] - hser.cummin()).clip(lower=0.0)   # inelastic cumulative drawdown
        at = draw.reindex(keep, method="nearest").to_numpy(dtype="float64")
        C = s.loc[keep].to_numpy(dtype="float64")
        out[sid] = (at - at[0], C - C[0])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage-1 Sk gate on the leveling network")
    ap.add_argument("--data", default=None, help="data dir (else HYDROMIND_GW_DATA)")
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--t0", default="2012-01-01")
    ap.add_argument("--t1", default="2023-01-01")
    ap.add_argument("--out", default="results/twin")
    args = ap.parse_args(argv)

    cfg = Config(data_dir=Path(args.data)) if args.data else Config()
    data = load_dataset(cfg)
    ddir = str(cfg.data_dir)

    panel = leveling.load_panel(ddir)
    sub_raw = leveling.site_subsidence(panel, args.t0, args.t1, min_obs=args.min_obs)
    xy = leveling.site_xy(panel)
    coast = {sid: float(xy[sid][0]) for sid in xy}     # proxy: easting increases inland

    rows = []
    for mode in ("none", "planar"):
        sub, info = leveling.remove_tectonic(sub_raw, xy, mode=mode)
        pairs = build_pairs(data, sub, xy)
        if not pairs:
            continue
        single = calibrate_sk_from_pairs(pairs)
        gate = loso_sk_regression(pairs, {k: coast[k] for k in pairs})
        rows.append({"tectonic": mode, "n_sites": len(pairs),
                     "var_removed": info["var_removed"], "sk_single": single["sk"],
                     "r2_insample": single["r2"], "loso_r2": gate["r2"],
                     "loso_r2_sk": gate["r2_sk"]})
        print(f"[{mode:6s}] sites={len(pairs):4d}  var_removed={info['var_removed']:.3f}  "
              f"single-Sk in-sample R2={single['r2']:+.3f}  "
              f"LOSO compaction R2={gate['r2']:+.3f}  LOSO Sk R2={gate['r2_sk']:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage1_sk_leveling.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path}")
    print("GATE: LOSO compaction R2 > 0 means spatial sparsity was the wall; "
          "still negative means the model form is, and Tasks 4-7 are justified.")


if __name__ == "__main__":
    main()
