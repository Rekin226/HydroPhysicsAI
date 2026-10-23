"""Step 1: build the station inventory and the H3 candidate list (metadata only).

Writes ``H3_evidence/inventory/inventory.csv`` (one row per station, non-personal columns)
and ``counts.json`` (the plan's counts and the reconciliation table). ``looks_h3.py`` logs a
``design_time`` record on its own unit ``H3:inventory_2012-2022`` (record extent only; no
head value is compared with a prediction).

Usage: python -I s1_inventory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402


def main():
    X = inventory.build()
    out = C.EVIDENCE / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    X.to_csv(out / "inventory.csv", index=False)
    cnt = inventory.counts(X)
    (out / "counts.json").write_text(json.dumps(cnt, indent=1, ensure_ascii=False,
                                                default=str))
    print(json.dumps(cnt, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
