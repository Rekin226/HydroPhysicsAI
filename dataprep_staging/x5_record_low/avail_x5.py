"""X5 availability smoke test (before registration): counts only, no value is printed.

Loads the same inputs as the scorer and prints, per ring site and band, the number of ring
months in FIT / TEST / PERSIST, whether an anchor survey exists, the chosen head well and
whether the primary and nest-mean head series pass the gap rule. Purpose: catch loader
crashes before the code is hashed into the declaration. Ring compaction and head values are
never printed or summarised.

Usage: python avail_x5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402


def main() -> int:
    recs, months = L.build_inputs()
    n_ok = 0
    for r in recs:
        if r["obs"] is None:
            print(r["site"], "no ring bands")
            continue
        f = int((L.in_window(months, K.FIT) & np.isfinite(r["obs"])).sum())
        t = int((L.in_window(months, K.TEST) & np.isfinite(r["obs"])).sum())
        p = int((L.in_window(months, K.PERSIST) & np.isfinite(r["obs"])).sum())
        a = L.anchor_index(r["obs"], months) is not None
        ok = (f >= K.MIN_FIT_OBS and t >= K.MIN_TEST_OBS and p >= K.MIN_PERSIST_OBS and a
              and r["h_primary"] is not None)
        n_ok += ok
        print(f"{r['site']} {r['band']:>10} fit {f:3d} test {t:3d} pers {p:3d} anchor {a!s:5} "
              f"well {r['well']} primary {r['h_primary'] is not None!s:5} "
              f"nest {r['h_nest'] is not None!s:5} support {ok}")
    print("site-bands meeting support:", n_ok, "of", len(recs),
          "sites:", len({r['site'] for r in recs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
