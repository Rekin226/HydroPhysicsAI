"""WP S3 repair round 3 (S3r): crash and runtime test of the v3 pipeline on the pre-origin base
runs (no injection, no A1/A2/A3 score), before the v3 design ran.

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/smoke_v3.py OUT_DIR

Runs the WRA and NGL pre-origin families once with the given starting parameters (default:
design run 2's stage-1 start, scan half-window 30 d, z_min 4.0, delta_min 15 mm, level-
difference size on +-10 d) and prints the run time and the number of detected steps per
family. Looks logged beforehand (``looks_r5.py smoke_v3`` for design run 1's start, ``smoke_v3b``
for run 2's).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import pipeline as P  # noqa: E402


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    p = replace(M.Params(), win=30, z_min=4.0, delta_min=0.015, local_half=10, trend_order=2,
                size_order=0)
    end = P.ORIGIN_END
    t0 = time.time()
    wra = P.run_wra(p, window_end=end, series=P.wra_series(end))
    t1 = time.time()
    nser, _ = P.ngl_series(end)
    ngl = P.run_ngl(p, ref=P.NGL_REF_CANDIDATES_PRE, window_end=end, series=nser)
    t2 = time.time()
    rep = {"params": p.as_dict(), "seconds_wra": round(t1 - t0, 1),
           "seconds_ngl": round(t2 - t1, 1),
           "n_detected": {f: sum(sum(1 for s in r.steps if s.evidence.startswith("detected"))
                                 for r in fam.results.values())
                          for f, fam in (("wra", wra), ("ngl", ngl))},
           "n_excursions": {f: sum(sum(1 for x in r.notes if x.startswith("excursion"))
                                   for r in fam.results.values())
                            for f, fam in (("wra", wra), ("ngl", ngl))}}
    (od / "smoke_v3.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
