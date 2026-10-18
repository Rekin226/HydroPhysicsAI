"""Loader-guard probe of every WP H2 product version after repair round 3.

For each version of heads/L2_filled_preorigin and heads/L2_filled_full and each use of the
vocabulary: granted (with the flags that ``read_table('fill.parquet')`` returns) or the refusal
type. Also probes a pre-origin read (``window_end`` 2022-12) of every full-variant version and
access to ``ic_field_2012-01.npz`` where a version has one. Writes
``H2_evidence/repair3/guard_probe_r3.json`` (the builder's ``guard_probe_h2.json`` is kept).

Usage: python -I guard_probe_r3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from common import guard  # noqa: E402
from common.uses import USES  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402


def probe(d, use, window_end=None):
    try:
        p = guard.open_product(d, use, window_end=window_end)
        df = p.read_table("fill.parquet", columns=["sid", "flag"])
        rec = {"granted": True, "rows": int(len(df)),
               "flags": sorted(int(x) for x in df.flag.unique())}
        if (d / "ic_field_2012-01.npz").exists():
            try:
                rec["ic_field_path"] = bool(p.path("ic_field_2012-01.npz").exists())
            except Exception as e:  # noqa: BLE001 - the refusal type is the record
                rec["ic_field_path"] = type(e).__name__
        return rec
    except Exception as e:  # noqa: BLE001 - the refusal type is the record
        return {"granted": False, "error": type(e).__name__, "message": str(e)[:200]}


def main():
    out = {}
    for product in (C.PRODUCT_PRE, C.PRODUCT_FULL):
        for d in sorted((C.REPO / "results/dataprep" / product).glob("v*")):
            rec = {use: probe(d, use) for use in sorted(USES)}
            if product == C.PRODUCT_FULL:
                rec["calibration_target@window_end=2022-12"] = probe(d, "calibration_target",
                                                                     "2022-12")
            out[f"{product}/{d.name}"] = rec
    R3.jdump(out, R3.EV_R3 / "guard_probe_r3.json")
    for k, v in out.items():
        print(k, {u: (r["flags"], r.get("ic_field_path", "")) if r["granted"] else None
                  for u, r in v.items() if r["granted"]})


if __name__ == "__main__":
    main()
