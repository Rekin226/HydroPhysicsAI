"""WP S4a: declare the setup acceptance criteria and the fetch-budget rule BEFORE the subsets
are fetched and validated (A1 tool, kind 'declaration'; plan 4.5).

Written after the header-only size survey and four single-file transfer probes (design-time
facts: per-file sizes, layouts, about 25 KB/s per connection), before any subset of the primary
frame existed. No 2023+ observation value enters any criterion: the criteria concern data
completeness and integrity (U7); the coherence and GNSS-pixel numbers are reported, and the one
GNSS criterion uses pre-origin pairs only.

Run with the hydro env: python -I declare_acceptance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from config import FRAMES, RUN_DIR  # noqa: E402
from harness.prereg import not_applicable, write_prereg  # noqa: E402

NA = not_applicable

CRITERIA = [
    {"id": "S4a-1", "criterion": "separate licsbas conda env; hydro untouched",
     "threshold": "env at ~/miniforge3/envs/licsbas from conda-forge; hydro conda-meta listing "
                  "and pip freeze byte-identical before/after; LiCSBAS_check_install.py OK"},
    {"id": "S4a-2", "criterion": "official LiCSBAS source obtained",
     "threshold": "git clone of github.com/comet-licsar/LiCSBAS, commit recorded, clean "
                  "worktree, git fsck without errors, outside the repo tree's versioned code"},
    {"id": "S4a-3", "criterion": "LiCSBAS runs end to end on fetched subsets",
     "threshold": "LiCSBAS02_ml_prep, 11, 12, 13 (n_para 1) exit 0 on >= 20 single-grid pairs "
                  "of the primary frame and write cum.h5"},
    {"id": "S4a-4", "criterion": "primary frame completeness 2020-01..2026-01",
     "threshold": "fetched / available in-window pairs >= 0.99; fetched network has 1 "
                  "connected component from the first to the last available epoch and >= 25 "
                  "epochs in 2022"},
    {"id": "S4a-5", "criterion": "integrity of every fetched file",
     "threshold": "0 validation problems (opens; geotransform = native geotransform shifted "
                  "by the logged window; size = window; dtype unw Float32 / cc Byte; no .part "
                  "leftovers); SOURCE.json per fetch directory; checker 0 failures and 0 "
                  "orphans under data_fetch_api/new_measurements/licsar_*"},
    {"id": "S4a-6", "criterion": "raw multi-range reader is exact",
     "threshold": "bitwise equality with GDAL's own read on the first and last 12 window rows "
                  "of one fetched L0 file per uncompressed portal batch"},
    {"id": "S4a-7", "criterion": "transfer budget",
     "threshold": "total S4a transfer (every request of the WP) <= 3.0 GB (hard cap); primary "
                  "transfer reported against D1's 'about 1.4 GB' (recorded as fail if above "
                  "1.5 GB, with sizes reported)"},
    {"id": "S4a-8", "criterion": "full-fan frames listed and priced",
     "threshold": "069A_06632_162010 and 105D_06628_232108: availability of every in-window "
                  "pair and an expected subset size; fetched only if the S4a total stays "
                  "<= 3.0 GB (rule below)"},
    {"id": "S4a-9", "criterion": "off-fan GNSS references usable by S4b",
     "threshold": ">= 2 NGL stations inside the clip box and outside the twin's active cells "
                  "fall on valid unw pixels in >= 50 % of sampled pre-origin primary pairs "
                  "(epoch2 <= 20221231)"},
]

FETCH_RULE = (
    "1) Primary 069A_06694_131402: all available in-window pairs (unw+cc) plus E/N/U/hgt/"
    "landmask, in the tiered order of order_pairs.py, with --max-gb 2.8 (3.0 GB minus the "
    "listings, survey and probes, minus 0.1 GB reserve). 2) After it completes: remaining = "
    "3.0 GB minus every byte transferred by S4a. Fetch 105D_06628_232108 (108 available pairs "
    "2025-01..2026-01; the only descending geometry) only if its expected subset size x 1.10 "
    "fits in the remaining; then 069A_06632_162010 under the same rule. A frame is fetched "
    "whole or not at all (no thinning). 3) Otherwise the frame is listed with sizes and a "
    "ready command for the user.")


def main() -> int:
    doc = {
        "id": "S4a.setup_acceptance.v1", "wp": "S4a", "kind": "declaration",
        "title": "S4a InSAR setup: acceptance criteria and fetch-budget rule",
        "tests": NA("declaration of setup acceptance criteria; no scored test"),
        "zones": NA("no zones; S4b defines the Changhua coherence zones"),
        "priors": NA("no model"), "burn_in": NA("no model"), "folds": NA("no model"),
        "fresh_units": NA("no scored units"),
        "thresholds": {"criteria": CRITERIA, "fetch_rule": FETCH_RULE},
        "gate_scores": NA("pass/fail per criterion"),
        "primary_model": NA("no model"),
        "inputs": [str(RUN_DIR / "run_config.json")]
                  + [str(RUN_DIR / f"pairs_order_{f}.txt") for f in FRAMES]
                  + [str(Path(s["l0_dir"]) / "listings" / "interferograms.html")
                     for s in FRAMES.values()],
        "interpretation": (
            "Criteria concern completeness and integrity of the L0 subsets and the readiness "
            "of the LiCSBAS environment; they never depend on deformation values. Known before "
            "writing: the fan-wide clip of the primary frame costs about 2.3-2.5 GB by "
            "header-only estimate (D1's 1.4 GB came from the scout's 0.22 x 0.20 deg box), so "
            "S4a-7's 1.4 GB comparison is expected to fail and is reported with sizes; "
            "105D_06628_232108 has GeoTIFFs for only 108 of 362 in-window pairs (2025-01.."
            "2026-01), so a 2022-11 asc/desc decomposition is not possible from it."),
    }
    p, sha = write_prereg(doc)
    print(p, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
