"""Repair registration of the repair-round construction check of the full product (A1 tool
``harness.prereg3``): P2abc.r3.k1_repair.v1.

P2r3b.k1.construction_full (P2abc.r3.repair.v2) failed on the first assembly, item (c) only:
1 Yunlin pump-month (look [P2-r3-k1-diag]): a code-M0 monthly meter installed 2026-04 had its
2025-07 month - a pre-install structural zero (C0 method pre_install, flag 0) - flagged by the
M0 July-2025 rule (flag 9, method m0_july2025_suspect). Product defect: the rule selected
flag-0 2025-07 months without excluding pre-install months. As registered, items (c) (pre-
install months keep 0 / flag 0 / pre_install) and (j) ("exactly the kept code-M0 monthly meters
with a flag-0 2025-07 reading carry flag 9") contradict each other for that pump-month; (j)
passed only because it reused the product's own selection. Fix (b5_assemble3.m0_suspect_rows):
an installed (not pre_install) flag-0 2025-07 reading. Check restated: (j) reads "installed"
and computes its expected set in t7_construction3.py, independently of the product code. Every
other item and every threshold is unchanged. The failed result stands for the first assembly,
which is never published. The pre-origin product (no M0 rule) passed its check and is not
re-assembled.

Usage: python -I prereg_k1r3.py [--dry-run]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
import prereg_r3 as R3  # noqa: E402
import prereg_r3b as R3B  # noqa: E402
import prereg_v2 as P  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import not_applicable  # noqa: E402
from p3lib import L  # noqa: E402

PREREG_ID = "P2abc.r3.k1_repair.v1"


def doc() -> dict:
    base = {t["test_id"]: t for t in R3B.tests()}
    t = copy.deepcopy(base["P2r3b.k1.construction_full"])
    t["test_id"] = "P2r3c.k1.construction_full"
    u = "P2r3c:reassembled_full"
    t["fresh_units"] = [u]
    t["external_units"] = {u: P.none_unit(
        "the re-assembled repair-round yunlin_electricity/full product (built after this "
        "registration with the repaired b5_assemble3.m0_suspect_rows)"),
        "P2r3b:assembled_full": P.none_unit(
            "the first repair-round assembly of yunlin_electricity/full (failed "
            "P2r3b.k1.construction_full item c; never published)")}
    t["design_time_units"] = ["P2r3b:assembled_full"]
    t["what"] = t["what"].replace(
        "(j) M0 July-2025 flags exactly on the kept code-M0 monthly meters with a flag-0 "
        "2025-07 reading (full; none in pre-origin)",
        "(j) M0 July-2025 flags exactly on the kept code-M0 monthly meters with an installed "
        "(not pre_install) flag-0 2025-07 reading, expected set computed in the check itself "
        "(full; none in pre-origin)")
    assert "installed (not pre_install)" in t["what"]
    t["gating"] = ("yunlin_electricity/full/v2 gate (with P2r3b.c1.zero_shipped_yunlin), "
                   "replacing P2r3b.k1.construction_full, which failed on the first assembly")
    first = L.load_json(P3.EVID3 / "tests" / "t7_construction3_full.json")
    return {
        "id": PREREG_ID, "wp": L.WP,
        "title": "P2abc repair round: construction check of the re-assembled full product after "
                 "the P2r3b.k1 item (c) failure (M0 rule on a pre-install month)",
        "kind": "test", "tests": [t], "fresh_units": [u],
        "zones": not_applicable("no zones"), "priors": not_applicable("no priors"),
        "burn_in": not_applicable("no model run"), "folds": not_applicable("no folds"),
        "thresholds": {t["test_id"]: t["threshold"]},
        "gate_scores": not_applicable(
            "conjunctive gate: yunlin_electricity/full/v2 = P2r3c.k1.construction_full and "
            "P2r3b.c1.zero_shipped_yunlin (passed); yunlin_electricity/preorigin/v2 = "
            "P2r3b.k1.construction_preorigin (passed) and P2r3b.c1.zero_shipped_yunlin"),
        "primary_model": "as P2abc.r3.repair.v2, with the M0 July-2025 rule restricted to "
                         "installed readings (C0 method not pre_install)",
        "inputs": R3.inputs(),
        "interpretation": {
            "failure": {"violations": first["violations"], "item_c": first["details"]["c"]},
            "diagnosis": "one Yunlin code-M0 monthly meter installed 2026-04: its 2025-07 "
                         "pre-install month (flag 0, method pre_install) was flagged 9 / "
                         "m0_july2025_suspect; M0 monthly flag-0 2025-07 months by C0 method: "
                         "Yunlin observed 4608, terminated_trailing_run 621, pre_install 1; "
                         "Changhua 7481 / 727 / 0; Chiayi 18 / 1 / 0 (look [P2-r3-k1-diag])",
            "fix": "b5_assemble3.m0_suspect_rows adds C0 method != pre_install; nothing else in "
                   "the product changes (the zero tables, members' fills and every other value "
                   "are recomputed by the same code)",
            "check_restatement": "(j) 'installed' added and its expected set computed in "
                                 "t7_construction3.py instead of importing the product's "
                                 "selection; a tightening (it would have failed the defective "
                                 "product), not a loosening",
            "not_fresh": "construction checks, not out-of-sample tests; the fix was decided "
                         "after the failure was read",
            "code_at_registration": P3.v3_code()},
    }


def main(dry: bool) -> None:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps({k: v for k, v in d.items() if k not in ("inputs",)},
                         ensure_ascii=False, indent=1)[:3000])
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
