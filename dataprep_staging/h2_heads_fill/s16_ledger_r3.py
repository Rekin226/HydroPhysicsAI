"""Ledger rows of WP H2 repair round 3 (plan 4.1, U6): one row per source and reported number.

Same conventions as ``s9_ledger.py``. The ledger allows one role per source per number, so the
L1 v3 heads stay the ``input`` of every artificial-gap and leave-one-well-out number although
their masked (held-out) values are also the truth; the row note says so. The registered result
record is the ``validator``; post-look numbers on already-scored units (January 2012, the E
recount, the late-starting wells) are ``design_time_evidence``.

Usage: python -I s16_ledger_r3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from common import ledger  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402

L1 = "results/dataprep/heads/L1_clean/v3"
UNITS = "results/dataprep/validation_units/v3"
EV = "results/dataprep/reports/H2_evidence"
R3 = f"{EV}/repair3"
STAGE3 = "results/twin_runs/stage3_datum_gate_cwa/stage3_fit_summary.json"
GRID = "results/twin_forward/datum_gate_innov.npz"
PILOT_E = "results/dataprep/_pilots/head_gapfill_pilot/out/pilot_E.parquet"


def pv_of(product: str, prefix: str) -> str:
    """Newest product version whose validation.json ``release`` starts with ``prefix``."""
    base = C.REPO / "results/dataprep" / product
    hits = []
    for d in sorted(base.glob("v*"), key=lambda p: int(p.name[1:])):
        v = d / "validation.json"
        if v.exists() and str(json.loads(v.read_text()).get("release", "")).startswith(prefix):
            hits.append(f"{product}/{d.name}")
    if not hits:
        raise SystemExit(f"no version of {product} with release {prefix!r}*")
    return hits[-1]


def rows():
    out = []
    pv_pre = pv_of(C.PRODUCT_PRE, "r3_bridge_pre")
    try:
        pv_full = pv_of(C.PRODUCT_FULL, "r3_bridge_full")
    except SystemExit:
        pv_full = None
    pv_v3 = pv_of(C.PRODUCT_PRE, "v2_fill")

    def add(num, src, role, pv, note):
        out.append({"number_id": num, "source": src, "role": role, "product_version": pv,
                    "note": note})

    # registered replication of the daily bridge (gate of the full-variant release)
    for kind in ("coverage", "rmse"):
        num = f"H2.fresh3.d.{kind}"
        pv = pv_full or pv_pre
        add(num, L1, "input", pv, "pre-origin daily and C20 monthly heads (window_end 2022-12) "
            "masked by family d r = 60..69; the masked C20 months are also the truth (one role "
            "per source, U6)")
        add(num, "dataprep_staging/harness/gaps_guarded.py", "input", pv,
            "artificial-gap generator (family d, seed 1000*ord('d')+r)")
        add(num, f"{EV}/design/design_params.json", "input", pv,
            "frozen bridge (form 'shape', bridged-share sds), unchanged since v1")
        add(num, f"{R3}/registered/H2.fresh3.d.json", "validator", pv,
            "registered score record (H2.repair3.v1)")
    # registered initial-condition decision against the twin's construction
    for num in ("H2.ic.twin.rmse", "H2.ic.twin.decision"):
        add(num, L1, "input", pv_pre, "January 2013..2022 L1 flag-0/1 monthly heads of the "
            "model wells (held-out values are also the truth) and the pre-origin record for "
            "the backfill")
        add(num, f"{UNITS}/wells.csv", "input", pv_pre, "model wells and layers")
        add(num, "results/twin/basemap.npz", "input", pv_pre, "SRTM cell elevation (GP drift)")
        add(num, GRID, "input", pv_pre, "model grid")
        add(num, STAGE3, "input", pv_pre, "stage-3 initial-condition recipe (backfill, merged "
            "proximal, zones 205,182,208)")
        add(num, f"{R3}/registered/H2.ic.twin.json", "validator", pv_pre,
            "registered decision record (H2.repair3.v1)")
    # post-look evidence on already-scored units
    for num, rec, note in (
            ("H2.ic.jan2012_twin.rmse", f"{R3}/ic_jan2012_twin.json",
             "post-look LOWO on January 2012 (H2.ic.lowo's unit) against the twin's actual "
             "construction"),
            ("H2.edge_ic.jan2012.rmse", f"{R3}/edge_ic_jan2012.json",
             "late-starting wells: January 2012 leading-edge cells of the H2 E realizations")):
        add(num, L1, "design_time_evidence", pv_pre, "pre-origin L1 heads (already scored "
            "units)")
        add(num, rec, "design_time_evidence", pv_pre, note)
    num = "H2.e_freshness.recount"
    add(num, L1, "input", pv_v3, "pre-origin matrix of the registered E realizations")
    add(num, PILOT_E, "input", pv_v3, "identity of the pilot's E r0-1 well-gaps only (values "
        "unused)")
    add(num, f"{R3}/e_freshness.json", "validator", pv_v3,
        "E strata of H2.fill_acceptance.v1/v2 recomputed with the pilot's well-gaps excluded")
    # product counts of the round-3 releases
    add("H2.product.r3_pre.counts", L1, "input", pv_pre,
        "L1 v3 monthly and daily heads (window_end 2022-12)")
    add("H2.product.r3_pre.counts", f"{EV}/design/design_params.json", "input", pv_pre,
        "frozen bridge")
    add("H2.product.r3_pre.counts", f"{EV}/registered/H2.fresh.d.json", "validator", pv_pre,
        "gate of the bridge component (v1 packaging rule)")
    if pv_full:
        add("H2.product.r3_full.counts", L1, "input", pv_full,
            "L1 v3 monthly and daily heads 2012-2026 (statistical_layer_history)")
        add("H2.product.r3_full.counts", f"{EV}/design/design_params.json", "input", pv_full,
            "frozen bridge")
        add("H2.product.r3_full.counts", f"{R3}/registered/H2.fresh3.d.json", "validator",
            pv_full, "gate of the full-variant bridge component (H2.repair3.v1 rule)")
    return out


def main():
    p = ledger.append_rows(C.WP, rows())
    print(p, len(rows()))


if __name__ == "__main__":
    main()
