"""Write the W3 pre-registration (plan 4.5) BEFORE the held-out benchmark test runs.

WP A1's pre-registration tool does not exist yet, so this W3-local writer follows the
convention of ``results/dataprep/preregistration/README.md``: one JSON per registered test,
never edited (a changed design is a new id), its sha256 recorded in the ledger. ``build.py``
refuses to run the validation unless this file exists, its sha256 matches the ledger row and
its design constants equal the code's.

Usage: python -I prereg.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from common import hashing, ledger  # noqa: E402
from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402
from w3_dtm import build  # noqa: E402


def main() -> int:
    p = PREREG_DIR / f"{build.PREREG_ID}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; a changed design needs a new id")
    inputs = [*build.ARCHIVES, build.DTM_DIR / "SOURCE.json", build.LEVELING,
              build.TWIN_GRID_NPZ, build.SRTM_NPZ]
    rec = {
        "id": build.PREREG_ID,
        "wp": "W3",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "writer": "dataprep_staging/w3_dtm/prereg.py (WP A1's tool is not available yet)",
        "test": "held-out bias of the epoch-corrected 1 km DTM cell mean against leveling "
                "benchmarks (plan section 3, WP W3, Validation)",
        "hypothesis": "the MOI 20 m DTM aggregated to the twin's active 1 km cells (mean of "
                      "the 20 m points inside the cell) and moved from its sheet acquisition "
                      "epochs to the benchmark observation epoch with leveling displacement "
                      "interpolated from benchmarks outside the held-out block agrees with the "
                      "held-out benchmark elevations to |median bias| <= 0.30 m",
        "primary": {
            "residual": "r = cell_mean(sheet epochs) + [u(c, t_obs) - sum_k w_k u(c, t_k)] - "
                        "benchmark elevation(t_obs), c = centroid of the active cell containing "
                        "the benchmark, u from benchmarks outside the held-out block",
            "per_benchmark": "median of r over the benchmark's main-campaign observations "
                             "2012-2021",
            "statistic": "median over all held-out benchmarks (pooled over folds)",
            "acceptance": "|statistic| <= 0.30 m (plan WP W3); the gate of the product",
            "units": "benchmarks inside an active cell (point in cell); folds = 10 km square "
                     "blocks anchored at the grid origin (x0, y0); leave-one-block-out",
        },
        "secondary_reported_never_gating": [
            "block-bootstrap 90% interval of the primary statistic (2000 draws, seed 1000)",
            "cell median instead of cell mean",
            "uncorrected cell mean (no epoch correction)",
            "point DTM (bilinear at the benchmark, epoch-corrected at the benchmark)",
            "SRTM basemap cell value (the field it replaces) on the same benchmarks",
            "10 km buffer variant (training also excludes benchmarks within 10 km of the "
            "block)",
            "per-year medians 2012, 2016, 2020 (comparison with the plan's SRTM numbers)",
            "IQR, MAD-sd, fractions |r| > 1 m and > 3 m",
            "coverage of value +- 1.645 sd_point, sd_point = sqrt(sd_within^2 + (a |t - "
            "t_sheet|)^2), block-cluster-adjusted Wilson interval (plan 4.4 rule)",
            "epoch-correction error: a in sigma = a |dt| from held-out benchmarks' own "
            "displacement between their cell's sheet epochs and the observation epoch",
            "strata: north/south of y = 2640 km (approximate county split), cell elevation "
            "bands, subsidence-rate bands; per-block medians",
        ],
        "design_constants": json.loads(json.dumps(build.DESIGN)),
        "inputs": [{"path": rel_to_repo(x), "sha256": hashing.sha256_file(x)} for x in inputs],
        "code_at_registration": [{"path": rel_to_repo(x), "sha256": hashing.sha256_file(x)}
                                 for x in build.CODE],
        "data_windows": "leveling main campaigns 2011-2021 only (pre-origin); DTM sheets "
                        "acquired <= 2022-12 (inside the grid 2016-11..2018-07); no 2023+ "
                        "observation anywhere (U7)",
        "looked_at_before_registration": [
            "plan WP W3 facts: SRTM cell value minus leveling benchmark, median +1.18/+1.23/"
            "+1.43 m (2012/2016/2020) - design-time evidence",
            "development run without leveling comparison: DTM cell means against SRTM "
            "(median SRTM - DTM +1.88 m), coverage, sheet epochs, the leveling displacement "
            "field itself, DTM against station GroundHeight/WellElevation metadata",
            "a synthetic check of the validation code (benchmark elevations replaced by the "
            "DTM point value plus modelled displacement; residuals came back ~0, confirming "
            "the sign conventions) also showed a DTM-only statistic: at the benchmark "
            "locations the DTM cell mean is a median 0.42 m BELOW the DTM point value "
            "(benchmarks sit on locally higher ground: roads, embankments). The primary "
            "statistic (cell mean, fixed in validation.py before that check) is NOT changed "
            "in response; this foreseeable siting offset is disclosed here",
            "no DTM-versus-real-leveling-elevation residual of any kind was computed",
        ],
        "if_it_fails": "the gate is recorded as failed with the number; the threshold is not "
                       "changed; consumers needing a passed gate are blocked by the loader",
    }
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.chmod(p, 0o444)
    sha = hashing.sha256_file(p)
    ledger.append_rows("W3", [{
        "number_id": "W3.preregistration", "source": rel_to_repo(p), "role": "input",
        "product_version": "ground_elevation (pre-registration)",
        "note": f"sha256={sha}; written {rec['created']} before the held-out test"}])
    print(f"wrote {p} sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
