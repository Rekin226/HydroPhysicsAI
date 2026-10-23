"""Loader-guard probe of the WP H2 products: which uses are granted, and which flags come back.

For every published version of heads/L2_filled_preorigin and heads/L2_filled_full, opens the
product for each use of the vocabulary and records
the outcome (granted with the flags returned by ``read_table('fill.parquet')``, or the
refusal type). The products declare no time slice (the pre-origin product's fit_window_end
is 2022-12 as a whole), so every use is opened without a window; pre-origin uses of the full
variant are refused on its fit_window_end. Writes ``H2_evidence/guard_probe_h2.json``.

Usage: python -I guard_probe_h2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from common import guard  # noqa: E402
from common.uses import USES  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402


def main():
    out = {}
    for product in (C.PRODUCT_PRE, C.PRODUCT_FULL):
        for d in sorted((C.REPO / "results/dataprep" / product).glob("v*")):
            rec = {}
            for use in sorted(USES):
                try:
                    p = guard.open_product(d, use)
                    df = p.read_table("fill.parquet", columns=["sid", "flag"])
                    rec[use] = {"granted": True, "rows": int(len(df)),
                                "flags": sorted(int(x) for x in df.flag.unique())}
                except Exception as e:  # noqa: BLE001 - the refusal type is the record
                    rec[use] = {"granted": False, "error": type(e).__name__,
                                "message": str(e)[:200]}
            out[f"{product}/{d.name}"] = rec
    scoring.jdump(out, C.EVIDENCE_DIR / "guard_probe_h2.json")
    for k, v in out.items():
        print(k, {u: (r["flags"] if r["granted"] else r["error"]) for u, r in v.items()
                  if r["granted"]})


if __name__ == "__main__":
    main()
