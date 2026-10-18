"""WP S4a: fetch order for a frame's available pairs, so that a fetch stopped early (budget cap,
network failure) still leaves a connected small-baseline network.

Tier 1: for every epoch, its two shortest available pairs to later epochs (the chain and the
        triangles LiCSBAS12's loop closure needs).
Tier 2: pairs of 60 days or longer not in tier 1 (they tie the network across gaps and limit
        the short-pair phase-bias accumulation).
Tier 3: everything else.
Within a tier, pairs are in date order. The rule uses only the portal listing (dates), never
pixel values. Output: ``results/dataprep/insar_tools/s4a_run/pairs_order_<frame>.txt``.

Usage (any env): python -I order_pairs.py --frame F
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import network as nw  # noqa: E402
from config import FRAMES, IFG_EXTS, RUN_DIR  # noqa: E402


def btemp(p: str) -> int:
    d = [dt.date(int(x[:4]), int(x[4:6]), int(x[6:8])) for x in (p[:8], p[9:])]
    return (d[1] - d[0]).days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=sorted(FRAMES))
    a = ap.parse_args()
    spec = FRAMES[a.frame]
    d = Path(spec["l0_dir"])
    listed = nw.window_pairs(nw.parse_pairs(d / "listings/interferograms.html"),
                             spec["start"], spec["end"])
    avail = [p for p in listed if all(nw.pair_products(
        d / "listings" / "ifg_dirs" / f"{p}.html", p, IFG_EXTS).values())]
    by_first = defaultdict(list)
    for p in avail:
        by_first[p[:8]].append(p)
    t1 = sorted({p for v in by_first.values() for p in sorted(v, key=btemp)[:2]})
    t2 = sorted(p for p in avail if btemp(p) >= 60 and p not in set(t1))
    rest = sorted(set(avail) - set(t1) - set(t2))
    order = t1 + t2 + rest
    out = RUN_DIR / f"pairs_order_{a.frame}.txt"
    out.write_text("\n".join(order) + "\n")
    s1 = nw.network_stats(t1)
    print(f"{a.frame}: {len(avail)} available; tier1 {len(t1)} (components "
          f"{s1['n_components']}, epochs {s1['n_epochs']}), tier2 {len(t2)}, tier3 {len(rest)}"
          f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
