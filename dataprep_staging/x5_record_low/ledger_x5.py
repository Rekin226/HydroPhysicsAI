"""Ledger rows for the X5 numbers (run once under the shared lock, after the scripts ran).

    flock results/dataprep/preregistration/.x5.lock python ledger_x5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5const as K  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

RINGS = "chou-shui-data/data/ls_cache/clean"
HEADS = rel_to_repo(K.HEADS_DIR)
PV = f"prereg:{K.PREREG_ID}"


def main() -> int:
    rows = []
    for nid, rr, hr, note in (
            ("X5.synthetic_check", None, "input",
             "synthetic recovery: real heads drive the twin column; ring survey pattern only"),
            ("X5.primary_sites_pass", "design_time_evidence", "input",
             "design-time comparison 2020-21 (0 of 14 sites pass)"),
            ("X5.followup_separation", "design_time_evidence", "input",
             "post-failure, exploratory; never validation; cannot promote"),
            ("X5.exploration_rate_ratio", "design_time_evidence", "input",
             "U7 exploration 2023-25 vs 2020-21; never gates")):
        if rr:
            rows.append({"number_id": nid, "source": RINGS, "role": rr, "product_version": PV,
                         "note": note})
        rows.append({"number_id": nid, "source": HEADS, "role": hr, "product_version": PV,
                     "note": note})
    rows.append({"number_id": "X5.exploration_rate_ratio", "source": rel_to_repo(K.PAGE_COLUMN),
                 "role": "input", "product_version": PV,
                 "note": "page column driven by observed nest-mean heads (exploration)"})
    print(ledger.append_rows(K.WP, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
