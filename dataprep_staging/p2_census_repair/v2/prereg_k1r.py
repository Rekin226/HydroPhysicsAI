"""Repair registration of the P2abc round-2 construction checks (A1 ``harness.prereg3``).

The registered checks P2r2.k1.construction_preorigin / _full (P2abc.r2.c0_based.v1) failed on
the first assembly, item (f) only: 4050 / 5560 class x cell-months held a flag-3 (Chow-Lin) or
imputed pump but had a NaN sd. Diagnosis ([P2-k1-diag-1], [P2-k1-diag-2]): every one of them
holds only zero-energy such pumps (a zero-HP pump with HP share 0 on a shared meter, or a
zero July-2026 imputation), and the assembly defined the sd from positive energy only. The
registered rule says the cell sd is finite iff a contributing pump has one (a zero-share
pump's sd is 0). Product defect; fixed in b3_assemble.py (the sd of such a cell is 0); no
other change. The k1 results stand as failed for that first assembly, which was never
published. This registration re-states the identical checks and thresholds under new test
ids for the re-assembled products.

Usage: python -I prereg_k1r.py [--dry-run]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
import prereg_v2 as P  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import not_applicable  # noqa: E402

PREREG_ID = "P2abc.r2.k1_repair.v1"


def doc() -> dict:
    base = {t["test_id"]: t for t in P.tests()}
    ts = []
    for v in ("preorigin", "full"):
        t = dict(base[f"P2r2.k1.construction_{v}"])
        t["test_id"] = f"P2r2.k1r.construction_{v}"
        u = f"P2r2:reassembled_{v}"
        t["fresh_units"] = [u]
        t["external_units"] = {u: P.none_unit(
            f"the re-assembled yunlin_electricity/{v} product (built after this registration "
            f"with the repaired b3_assemble.py)")}
        t["design_time_units"] = [f"P2r2:assembled_{v}"]
        t["external_units"][f"P2r2:assembled_{v}"] = P.none_unit(
            f"the first assembly of yunlin_electricity/{v} (failed P2r2.k1, never published)")
        t["gating"] = (f"yunlin_electricity/{v} gate (with P2r2.c1.zero_port_yunlin), "
                       f"replacing P2r2.k1.construction_{v}, which failed on the first assembly")
        ts.append(t)
    first = {v: L.load_json(L.EVID / "tests" / f"t5_construction_{v}.json")
             for v in ("preorigin", "full")}
    return {
        "id": PREREG_ID, "wp": L.WP,
        "title": "P2abc round 2: construction checks of the re-assembled products after the "
                 "P2r2.k1 item (f) failure",
        "kind": "test", "tests": ts,
        "fresh_units": sorted({u for t in ts for u in t["fresh_units"]}),
        "zones": not_applicable("no zones"), "priors": not_applicable("no priors"),
        "burn_in": not_applicable("no model run"), "folds": not_applicable("no folds"),
        "thresholds": {t["test_id"]: t["threshold"] for t in ts},
        "gate_scores": not_applicable(
            "conjunctive gates: yunlin_electricity/preorigin = P2r2.k1r.construction_preorigin "
            "and P2r2.c1.zero_port_yunlin; yunlin_electricity/full = P2r2.k1r.construction_full "
            "and P2r2.c1.zero_port_yunlin"),
        "primary_model": "as P2abc.r2.c0_based.v1, with the sd of an aggregate holding only "
                         "zero-energy flag-3 or imputed pumps set to 0 instead of NaN "
                         "(b3_assemble.py)",
        "inputs": P.inputs(),
        "interpretation": {
            "failure": {v: {"violations": first[v]["violations"],
                            "item_f": first[v]["details"]["f"]} for v in first},
            "diagnosis": "all violating cells hold only zero-energy flag-3 pumps (HP share 0) "
                         "or zero imputations; no violating cell holds a positive-share flag-3 "
                         "pump or a positive imputation (look [P2-k1-diag-2])",
            "fix": "b3_assemble.py: has |= n_cl > 0 or n_imp > 0 in every aggregate; nothing "
                   "else changed (values, flags, other sds unchanged by construction)",
            "not_fresh": "the checks are construction checks, not out-of-sample tests; the "
                         "fix was decided after the failure was read",
            "code_at_registration": {L.rel(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(L.CODE_DIR.glob("*.py"))}},
    }


def main(dry: bool) -> None:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps(d["interpretation"]["failure"]), [t["test_id"] for t in d["tests"]])
        return
    print(*prereg3.write_prereg(d))


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
