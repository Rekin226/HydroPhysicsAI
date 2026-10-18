"""Ledger rows of WP H2 (plan 4.1, U6): one row per source and reported number.

Roles: the L1 v3 heads are the ``input`` of every artificial-gap number (the masked matrix and
its truth cells are the same product); the mask generators and the frozen design parameters
are ``input``; the registered result record is the ``validator``; design-time numbers use
``design_time_evidence``; after-the-fact numbers use the 2023-26 L1 heads as ``validator``.

Usage: python -I s9_ledger.py
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


def pv_of(product: str, release: str) -> str:
    """Product version (``<product>/v<N>``) whose validation.json records ``release``."""
    base = C.REPO / "results/dataprep" / product
    for d in sorted(base.glob("v*")):
        v = d / "validation.json"
        if v.exists() and json.loads(v.read_text()).get("release") == release:
            return f"{product}/{d.name}"
    raise SystemExit(f"no version of {product} with release {release!r}")


PV_PRE = pv_of(C.PRODUCT_PRE, "v1_fill")
PV_PRE2 = pv_of(C.PRODUCT_PRE, "v2_fill")
PV_FULL2 = pv_of(C.PRODUCT_FULL, "v2_fill")
PV_BRIDGE = pv_of(C.PRODUCT_PRE, "v1_bridge")


def rows():
    out = []

    def add(num, src, role, pv, note):
        out.append({"number_id": num, "source": src, "role": role, "product_version": pv,
                    "note": note})

    for fam in ("N", "S", "E", "H", "T", "d"):
        for kind in ("coverage", "rmse"):
            num = f"H2.fresh.{fam}.{kind}"
            add(num, L1, "input", PV_PRE, "pre-origin C20 monthly matrix (window_end 2022-12) "
                "masked by the fresh realizations; its masked C20 cells are the truth")
            add(num, "dataprep_staging/harness/gaps_guarded.py", "input", PV_PRE,
                "artificial-gap generator (seed 1000*ord(family)+r; T 7000+r)")
            add(num, f"{EV}/design/design_params.json", "input", PV_PRE,
                "frozen design: gap-class sd factors, methods, bridge sds")
            add(num, f"{EV}/registered/H2.fresh.{fam}.json", "validator", PV_PRE,
                "registered score record (H2.fill_acceptance.v1)")
    for fam in ("N", "S", "E", "H", "T"):
        for kind in ("coverage", "rmse"):
            num = f"H2.fresh2.{fam}.{kind}"
            add(num, L1, "input", PV_PRE2, "pre-origin C20 monthly matrix (window_end "
                "2022-12) masked by the v2 realizations (r >= 30); its masked C20 cells are "
                "the truth")
            add(num, "dataprep_staging/harness/gaps_guarded.py", "input", PV_PRE2,
                "artificial-gap generator (seed 1000*ord(family)+r; T 7000+r)")
            add(num, f"{EV}/design_v2/design_params_v2.json", "input", PV_PRE2,
                "v2 design: sd factors re-estimated after the v1 look")
            add(num, f"{EV}/registered_v2/H2.fresh2.{fam}.json", "validator", PV_PRE2,
                "registered score record (H2.fill_acceptance.v2)")
    add("H2.design_v2.factors", L1, "design_time_evidence", PV_PRE2,
        "v1 design and v1 fresh realizations (design-time for v2)")
    add("H2.design_v2.factors", f"{EV}/design_v2", "design_time_evidence", PV_PRE2,
        "v2 design outputs")
    add("H2.product.bridge.counts", L1, "input", PV_BRIDGE, "L1 v3 (window_end 2022-12)")
    add("H2.product.bridge.counts", f"{EV}/registered/H2.fresh.d.json", "validator",
        PV_BRIDGE, "gate of the bridge component (v1 packaging rule)")
    for num, pv in (("H2.product.pre2.counts", PV_PRE2), ("H2.product.full2.counts", PV_FULL2)):
        add(num, L1, "input", pv, "L1 v3 monthly and daily heads (pre: window_end 2022-12)")
        add(num, f"{EV}/design_v2/design_params_v2.json", "input", pv, "v2 design")
    for num in ("H2.ic.lowo.rmse", "H2.ic.decision"):
        add(num, L1, "input", PV_PRE, "January 2012 L1 heads of the model wells")
        add(num, f"{UNITS}/wells.csv", "input", PV_PRE, "model wells and layers")
        add(num, "results/twin/basemap.npz", "input", PV_PRE, "SRTM cell elevation (drift)")
        add(num, f"{EV}/registered/H2.ic.lowo.json", "validator", PV_PRE, "registered record")
    for num in ("H2.design.bridge_form", "H2.design.factors", "H2.design.bridge_sd",
                "H2.design.D_febmar_conformance"):
        add(num, L1, "design_time_evidence", PV_PRE, "design-time realizations r = 0..4 and "
            "family D 2013-2022 on the pre-origin matrix")
        add(num, f"{EV}/design", "design_time_evidence", PV_PRE, "design-stage outputs")
    for num, pv in (("H2.product.pre.counts", PV_PRE),):
        add(num, L1, "input", pv, "L1 v3 monthly and daily heads (pre: window_end 2022-12)")
        add(num, f"{EV}/design/design_params.json", "input", pv, "frozen design")
    ap = C.EVIDENCE_DIR / "after_the_fact" / "full_variant_checks.json"
    used = json.loads(ap.read_text()).get("params_file", "") if ap.exists() else ""
    pfile = (f"{EV}/design_v2/design_params_v2.json" if "design_v2" in used
             else f"{EV}/design/design_params.json")
    for num in ("H2.after_fact.T_testperiod", "H2.after_fact.D_2013_2025"):
        add(num, L1, "validator", PV_FULL2, "2023-26 L1 heads as after-the-fact truth (U7: "
            "never a gate)")
        add(num, pfile, "input", PV_FULL2, "design parameters of the checked configuration")
        add(num, f"{EV}/after_the_fact/full_variant_checks.json", "validator", PV_FULL2,
            "after-the-fact record")
    return out


def main():
    p = ledger.append_rows(C.WP, rows())
    print(p)


if __name__ == "__main__":
    main()
