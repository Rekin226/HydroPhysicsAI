"""Refit the page column with surveys withheld and keep its subsidence field.

``calibrate_coupled`` computes the fitted column's field on every cell and month and only
scores it. This wrapper runs the calibration unchanged and saves that field, so the
withheld surveys can be scored by ``retro_score``. No twin module is edited.

    python -m dataprep_staging.r12_layer.retro_column --theta <stage3_theta.json> \\
        --target leveling --configs zonal --hpc0-guard-days 365 --device cuda \\
        --withhold-surveys 2017-01:2022-12 --out <dir>
"""

from __future__ import annotations

import os
import sys

import numpy as np

from hydrophysics.twin import calibrate_coupled, explorer3d


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = argv[argv.index("--out") + 1]
    original = explorer3d.validate_against_leveling

    def keep(ddir, grid, field, dates, **kw):
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "field.npz"), field=np.asarray(field, "float32"),
                            dates=np.array([str(d.date()) for d in dates]),
                            centroids=np.asarray(grid.centroids(), "float64"))
        return original(ddir, grid, field, dates, **kw)

    explorer3d.validate_against_leveling = keep
    calibrate_coupled.main(argv)


if __name__ == "__main__":
    main()
