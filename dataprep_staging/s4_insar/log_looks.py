"""WP S4a: record looks at the InSAR data in the A1 look log (``harness.prereg.record_look``).

S4a scores nothing, but its integrity checks read pixel values, so each one is logged before
it runs (the earlier design-time alignment test is back-dated to its evidence file). Unit
names are ``insar:<frame>``; scopes name what was read and the epoch range. S4b's coherence
acceptance has no look: S4a computes no coherence statistic.

Usage (hydro env): python -I log_looks.py <key> [--frame F]
  keys: align_seed | validate | spot_check | smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # dataprep_staging (common, harness)
sys.path.insert(0, str(HERE))

from config import FRAMES, RUN_DIR, WP  # noqa: E402
from harness.prereg import record_look  # noqa: E402

PRIMARY = "069A_06694_131402"
DECL = "S4a.setup_acceptance.v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=["align_seed", "validate", "spot_check", "smoke"])
    ap.add_argument("--frame", default=PRIMARY, choices=sorted(FRAMES))
    a = ap.parse_args()
    f = a.frame
    win = f"{FRAMES[f]['start'][:4]}-{FRAMES[f]['start'][4:6]}..{FRAMES[f]['end'][:4]}-" \
          f"{FRAMES[f]['end'][4:6]}"
    if a.key == "align_seed":
        ev = RUN_DIR / "probes" / "align_shift_matrix_20200112_20200118.txt"
        t = dt.datetime.fromtimestamp(ev.stat().st_mtime, dt.timezone.utc).isoformat()
        rec = record_look([f"insar:{PRIMARY}"], "insar_unw_cc_alignment:2020-01..2020-01", WP,
                          "design_time_seed", seeded_time=t, prereg_id=DECL,
                          note="S4a run 1 alignment probe: counts of valid unw pixels where "
                               "cc == 0 at 0/+-1 px shifts for pair 20200112_20200118 (scratch "
                               f"clip); evidence {ev.name}; no deformation value read")
    elif a.key == "validate":
        rec = record_look([f"insar:{f}"], f"insar_integrity_validity_window_ngl_pixels:{win}",
                          WP, "integrity_check", prereg_id=DECL,
                          note="validate.py r3: opens every fetched file; valid-pixel fraction "
                               "over the whole clip window; unw-valid where cc == 0 (index "
                               "alignment); unw validity at the NGL station pixels of sampled "
                               "pairs (S4a-9 on pre-origin pairs). No coherence statistic, no "
                               "validity at twin cells, no deformation value")
    elif a.key == "spot_check":
        rec = record_look([f"insar:{f}"], f"insar_raw_reader_rows:{win}", WP,
                          "integrity_check", prereg_id=DECL,
                          note="check_raw_reader.py: bitwise comparison of the first and last 12 "
                               "window rows of one raw-reader file per portal batch with GDAL's "
                               "own read; no value statistics")
    else:
        rec = record_look([f"insar:{PRIMARY}"], "insar_licsbas_smoke:2020-01..2022-12", WP,
                          "integrity_check", prereg_id=DECL,
                          note="smoke_licsbas.sh: LiCSBAS 02/11/12/13/14 on 40 pre-origin pairs; "
                               "exit codes and cum.h5 shape only; no displacement value read")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
