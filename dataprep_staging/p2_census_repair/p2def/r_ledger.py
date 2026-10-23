"""Ledger rows of WP P2def (plan 4.1, U6): one row per (reported number, source), one role per
source per number. Written after the products are published.

Usage: python -I r_ledger.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
from common import ledger  # noqa: E402


def latest(product: str) -> str:
    base = D.DATAPREP / product
    vs = sorted((p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")),
                key=lambda p: int(p.name[1:]))
    return f"{product}/{vs[-1].name}"


def rows() -> list[dict]:
    pre_c0 = "pump_electricity/preorigin/v3"
    full_c0 = "pump_electricity/full/v3"
    l1 = "pump_electricity/census_L1/v3"
    tpc = D.rel(D.TPC_CSV)
    pre_l, full_l = latest("census_post2016/preorigin"), latest("census_post2016/full")
    ext, lr = latest("census_extension/full"), latest("layer_response")
    reg = "results/dataprep/preregistration/P2def.v1.json"
    decl = "results/dataprep/preregistration/P2def.f.layer_response.v1.json"
    out = []

    def add(nid, src, role, pv, note=""):
        out.append({"number_id": nid, "source": src, "role": role, "product_version": pv,
                    "note": note})

    # design-time looks
    for nid, srcs in (("P2def.d0.census_design", [(l1, l1)]),
                      ("P2def.d0.billing_completeness", [(full_c0, full_c0)]),
                      ("P2def.d0.tpc_parse", [(tpc, "tpc_d007012/2026-10-12")])):
        for s, pv in srcs:
            add(nid, s, "design_time_evidence", pv, "design-time look before P2def.v1")
    # P2d back-test
    for k in D.COUNTIES:
        nid = f"P2def.d.backtest.{k}"
        add(nid, f"{pre_c0}:frozen census (pumps installed <= 2013-06, readings <= 2013-06)",
            "input", pre_c0, "predictor side of the mid-2013 freeze back-test")
        add(nid, f"{pre_c0}:cohorts 2014-01..2016-06 (counts by township, kWh to 2022-11)",
            "validator", pre_c0, "truth of the back-test")
        add(nid, l1, "input", l1, "install months, townships, classes, cells")
        add(nid, reg, "input", "P2def.v1", "registered rule and thresholds")
    # P2d layer and checks
    for v, prod, c0 in (("preorigin", pre_l, pre_c0), ("full", full_l, full_c0)):
        for nid in (f"P2def.d.layer.{v}", f"P2def.d.k1.construction_{v}"):
            add(nid, prod, "input", prod)
            add(nid, c0, "input", c0, "per-pump energy (recorded arm), per-pump index, ramp")
            add(nid, l1, "input", l1)
            add(nid, D.rel(D.EVID / "tests" / "P2d_backtest.json"), "input", "P2def.v1",
                "s_rel of the sd")
    # P2d evidence
    nid = "P2def.d.evidence.tpc_residual_trend"
    add(nid, tpc, "input", "tpc_d007012/2026-10-12")
    add(nid, pre_c0, "input", pre_c0)
    add(nid, full_c0, "input", full_c0)
    add(nid, pre_l, "input", pre_l)
    add(nid, full_l, "input", full_l)
    nid = "P2def.d.evidence.heads_ramp"
    for f in ("inputs_cwa.npz", "hindcast_cwa.npz", "K_pump_f32.npy", "inv_lib.py"):
        add(nid, D.rel(D.PILOT_PHYS / f), "design_time_evidence",
            "_pilots/physics_inversion_hooks", "frozen cwa theta, heads 2012-2022")
    nid = "P2def.d.evidence.survivorship"
    add(nid, "yunlin_electricity/preorigin/v2", "input", "yunlin_electricity/preorigin/v2")
    add(nid, "yunlin_electricity/full/v2", "input", "yunlin_electricity/full/v2")
    nid = "P2def.d.evidence.kwh_per_active_meter"
    add(nid, pre_c0, "input", pre_c0)
    add(nid, full_c0, "input", full_c0)
    add(nid, l1, "input", l1)
    # P2e
    for k in ("yunlin", "changhua"):
        for t in ("x2", "x1"):
            nid = f"P2def.e.{t}.{k}"
            add(nid, f"{full_c0}:county census months <= origin (fit)", "input", full_c0)
            add(nid, f"{full_c0}:county census months after origin (truth)", "validator",
                full_c0)
            add(nid, tpc, "input", "tpc_d007012/2026-10-12")
            add(nid, reg, "input", "P2def.v1")
    for nid in ("P2def.e.extension", "P2def.e.k1.construction"):
        add(nid, ext, "input", ext)
        add(nid, full_c0, "input", full_c0, "census to 2026-06 and 2023-2025 shares")
        add(nid, tpc, "input", "tpc_d007012/2026-10-12", "TPC 2026-07 and 2026-08")
    nid = "P2def.e.compare_census_2026_07"
    add(nid, ext, "input", ext)
    add(nid, "yunlin_electricity/full/v2", "validator", "yunlin_electricity/full/v2",
        "census-based 2026-07 (recorded arm with the July-2026 imputation)")
    # P2f
    for nid in ("P2def.f.ratio_observed", "P2def.f.placebo"):
        add(nid, "heads/L1_clean/v3", "design_time_evidence", "heads/L1_clean/v3",
            "2012-01..2022-11 slice; design-time units")
        add(nid, pre_c0, "input", pre_c0, "local energy (recorded arm)")
        add(nid, "yunlin_electricity/preorigin/v2", "input", "yunlin_electricity/preorigin/v2",
            "Chow-Lin arm (sensitivity)")
        add(nid, l1, "input", l1, "pump coordinates (memory only)")
        add(nid, D.rel(D.RAIN), "input", "cwa_codis_rain", "rain lags 0-2")
        add(nid, decl, "input", "P2def.f.layer_response.v1")
        add(nid, lr, "input", lr)
    nid = "P2def.f.ratio_model"
    add(nid, D.rel(D.FWD_CWA), "input", "twin_forward/datum_gate_cwa", "primary model side")
    add(nid, D.rel(D.PILOT_PHYS / "hindcast_cwa.npz"), "design_time_evidence",
        "_pilots/physics_inversion_hooks", "secondary (fit_full hindcast)")
    add(nid, D.rel(D.FWD_CHC), "input", "twin_forward/datum_gate_chc",
        "supplementary; another session's run")
    add(nid, pre_c0, "input", pre_c0)
    add(nid, lr, "input", lr)
    nid = "P2def.f.depth_proxy"
    add(nid, pre_c0, "input", pre_c0)
    add(nid, l1, "input", l1)
    # process facts
    for nid in ("P2def.checker", "P2def.guard_probe"):
        for p in (pre_l, full_l, ext, lr):
            add(nid, p, "input", p)
    add("P2def.synthetic_core_check", D.rel(D.CODE_DIR / "t0_synthetic.py"), "input",
        "code", "toy census; no real data")
    return out


def main() -> None:
    r = rows()
    p = ledger.append_rows(D.WP, r)
    allr = ledger.read_all()
    bad = ledger.conflicts(allr)
    print("rows", len(r), "fragment", p, "conflicts across fragments", len(bad))
    D.save_json({"rows_written": len(r), "fragment": D.rel(p), "conflicts": len(bad),
                 "rows_all_fragments": len(allr)}, D.EVID / "ledger_check.json")


if __name__ == "__main__":
    main()
