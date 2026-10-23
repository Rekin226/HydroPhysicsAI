"""Ledger fragment of the P2abc repair round (rule U6, plan 4.1).

1. Role corrections (verifier minor issue): units whose outcomes were known when they were
   scored (round-1 Yunlin units, the Yunlin round-2 units declared not fresh, the billing
   pilot's head-check wells and 2012-2022 heads for Yunlin and the pooled statistics, the
   pilot references of the round-2 zero check) carry the role design_time_evidence, not
   validator. The round-2 fragment is kept unchanged as ledger/superseded/P2abc.r2.csv and the
   fragment is rewritten with the corrected roles (every other row unchanged).
2. Rows for every number the repair round reports.

Usage: python -I r3_ledger.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from common import ledger  # noqa: E402
from common.paths import LEDGER_ROWS  # noqa: E402
from common.uses import ROLES  # noqa: E402
from p3lib import L  # noqa: E402

FRAG = Path(LEDGER_ROWS) / "P2abc.csv"
SUPERSEDED = Path(LEDGER_ROWS).parent / "superseded" / "P2abc.r2.csv"
TO_DTE = {
    "P2abc.r1.meter_validation", "P2abc.r1.sd_coverage", "P2abc.r1.july",
    "P2abc.r1.zero_reproduction", "P2abc.r1.head_placebo",
    "P2abc.r2.a1.skill.yunlin.preorigin", "P2abc.r2.a1.skill.yunlin.full",
    "P2abc.r2.a1.placebo.yunlin.preorigin", "P2abc.r2.a2.sdcov.yunlin.preorigin",
    "P2abc.r2.a2.sdcov.yunlin.full", "P2abc.r2.a3.heads.yunlin", "P2abc.r2.a3.heads.pooled",
    "P2abc.r2.a3.heads.fixed_nuisance", "P2abc.r2.b1.july.yunlin", "P2abc.r2.b1.july_sd.yunlin",
    "P2abc.r2.c1.zero_port"}
C0 = {v: f"results/dataprep/pump_electricity/{v}/v3" for v in ("preorigin", "full")}
L1 = "results/dataprep/pump_electricity/census_L1/v3"
P2V2 = {v: f"results/dataprep/yunlin_electricity/{v}/v2" for v in ("preorigin", "full")}
P2V1 = {v: f"results/dataprep/yunlin_electricity/{v}/v1" for v in ("preorigin", "full")}
PILOT = "results/dataprep/_pilots/billing_and_census_gaps"
L0 = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/parquet"
T3 = "results/dataprep/reports/P2abc_evidence/repair3/tests"


def rows() -> list[dict]:
    R = []

    def add(nid, src, role, pv, note):
        R.append({"number_id": nid, "source": src, "role": role, "product_version": pv,
                  "note": note})
    for v in ("preorigin", "full"):
        add(f"P2abc.r3.zeros.{v}", C0[v], "input", P2V2[v],
            "zero classes, structural rows, terminated meters and last-12-month energy, "
            "synchronized-zero months and rows (zeros3.py)")
        add(f"P2abc.r3.syncfill.{v}", C0[v], "input", P2V2[v],
            "synchronized-zero members: filled meter-months and energy, member minus arm by "
            "year and in July-August")
        add(f"P2abc.r3.syncfill.{v}", P2V2[v], "input", P2V2[v], "member and arm arrays")
        add(f"P2abc.r3.facts.{v}", P2V2[v], "input", P2V2[v],
            "v1-v2 array comparison, July-2026 imputations of arms and members, mask reasons")
        add(f"P2abc.r3.facts.{v}", P2V1[v], "input", P2V2[v], "v1 arrays (comparison)")
    add("P2abc.r3.c1.zero_shipped", C0["full"], "input", P2V2["preorigin"],
        "P2r3b.c1.zero_shipped_yunlin: the shipped classifier's input (pilot view, Yunlin)")
    for f in ("meter_mat.npy", "meters_phase.parquet", "dates.npy", "census_zeros.json"):
        add("P2abc.r3.c1.zero_shipped", f"{PILOT}/{f}", "design_time_evidence",
            P2V2["preorigin"], "pilot reference of a reproduction check on Yunlin units whose "
            "round-2 outcome was known (not fresh)")
    add("P2abc.r3.c1.zero_shipped", "results/dataprep/yunlin_electricity/_work_p2abc/zeros.json",
        "design_time_evidence", P2V2["preorigin"],
        "round-1 exact reproduction of the pilot's run table (reference)")
    add("P2abc.r3.k1.construction.preorigin", C0["preorigin"], "validator", P2V2["preorigin"],
        "P2r3b.k1.construction_preorigin (construction reference)")
    add("P2abc.r3.k1.first_assembly.full", C0["full"], "validator", P2V2["full"],
        "P2r3b.k1.construction_full on the first assembly (failed, item c)")
    add("P2abc.r3.k1r.construction.full", C0["full"], "validator", P2V2["full"],
        "P2r3c.k1.construction_full on the re-assembly")
    for nid in ("P2abc.r3.k1.construction.preorigin", "P2abc.r3.k1r.construction.full"):
        add(nid, L1, "validator", P2V2["preorigin" if "preorigin" in nid else "full"],
            "census attributes (construction reference)")
        for c in L.COUNTIES:
            add(nid, f"{L0}/{c}_census.parquet", "validator",
                P2V2["preorigin" if "preorigin" in nid else "full"],
                "raw meter numbers for the privacy scan (item g); never written")
    for c in L.COUNTIES:
        add(f"P2abc.r3.b2.july_member.{c}", f"P2r2:july_units_{c}", "design_time_evidence",
            P2V2["full"], "reported-only backtest on round-2 July units (outcomes known)")
        add(f"P2abc.r3.b2.july_member.{c}", P2V2["full"], "input", P2V2["full"],
            "recorded synchronized-zero member (sync_fill.parquet)")
    add("P2abc.r3.m0_july2025", C0["full"], "input", P2V2["full"],
        "M0 July-2025 flags: meters, pumps, energy, cells flagged 9")
    add("P2abc.r3.design.explore_2025", C0["full"], "design_time_evidence", P2V2["full"],
        "x1_explore_2025.py: pair totals, zero-bill shares, next-bill ratios, monthly ratios "
        "2012-2026 (design of the P2b repair)")
    add("P2abc.r3.design.rain", "data_fetch_api/new_measurements/cwa_codis_rain/"
        "rf_timeseries_codis.csv", "design_time_evidence", P2V2["full"],
        "x2_rain_check.py: July-August rain 2012-2025 by side of the Choushui")
    for kind, srcs in (("census_attributes", [f"{L0}/{c}_census.parquet" for c in L.COUNTIES]
                        + ["results/dataprep/reports/C0_evidence/repair3/checks/R9.json"]),
                       ("residual_unpaired_dec2022",
                        [f"{L0}/{c}_kwh_long.parquet" for c in L.COUNTIES]
                        + ["results/dataprep/reports/C0_evidence/repair3/"
                           "residual_sensitivity.json"])):
        for s in srcs:
            add(f"P2abc.r3.exemption.{kind}.preorigin", s, "input", P2V2["preorigin"],
                "exemption (U3) inherited from C0 v3, unchanged")
    add("P2abc.r3.exemption.zero_check_gate.preorigin", f"{T3}/t6_zero_shipped.json", "input",
        P2V2["preorigin"], "exemption (U3): the gate's zero check reads the Yunlin record to "
        "2025-07; no value depends on it")
    add("P2abc.r3.exemption.zero_check_gate.preorigin", C0["full"], "input", P2V2["preorigin"],
        "data of the zero check (pilot view)")
    for nid, note in (("P2abc.r3.checker", "common.check re-hash of yunlin_electricity and "
                       "pump_electricity manifests"),
                      ("P2abc.r3.guard_probe", "guard probe of both v2 products"),
                      ("P2abc.r3.preorigin_rebuild", "rebuild of the pre-origin staging product "
                       "with the final code: identical content")):
        for v in ("preorigin", "full"):
            if nid == "P2abc.r3.preorigin_rebuild" and v == "full":
                continue
            add(nid, P2V2[v], "input", P2V2[v], note)
    return R


def main() -> None:
    SUPERSEDED.parent.mkdir(parents=True, exist_ok=True)
    if not SUPERSEDED.exists():
        shutil.copy2(FRAG, SUPERSEDED)
    with open(FRAG, newline="", encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    n_changed = 0
    for r in old:
        if r["number_id"] in TO_DTE and r["role"] == "validator":
            r["role"] = "design_time_evidence"
            r["note"] = (r["note"] + " [role corrected to design_time_evidence in the repair "
                         "round: outcome known when scored]").strip()
            n_changed += 1
    bad = ledger.conflicts(old)
    if bad:
        raise SystemExit(f"conflicts after role correction: {bad}")
    with open(FRAG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ledger.COLUMNS)
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in ledger.COLUMNS})
    ledger.append_rows(L.WP, rows())
    allrows = ledger.read_all()
    conf = ledger.conflicts(allrows)
    mine = [r for r in allrows if r["wp"] == L.WP]
    print({"roles_corrected": n_changed, "fragment_rows": len(mine),
           "roles": {k: sum(1 for r in mine if r["role"] == k) for k in ROLES
                     if any(r["role"] == k for r in mine)},
           "cross_fragment_conflicts": len(conf), "all_rows": len(allrows)})


if __name__ == "__main__":
    main()
