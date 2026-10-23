"""WP C0: ledger rows (plan 4.1, U6) for the numbers the C0 report quotes that ledger_c0.py and
resume_c0.py did not already cover. Each number is recomputed here from its source file.

Usage: python -I report_ledger_c0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import ledger  # noqa: E402

PE = lib.PRODUCT_ROOT
PRE, L1 = PE / "preorigin" / "v1", PE / "census_L1" / "v1"


def main() -> int:
    bi = lib.load_json(PRE / "build_info.json")["counties"]
    dom = {r["class"]: r for r in bi["changhua"]["clean_report"]}["domestic"]
    dg = lib.load_json(lib.EVIDENCE / "diagnostics.json")["stats_preorigin"]
    moy = {c: dg[c]["domestic_month_of_year_GWh_2012_2022"] for c in ("yunlin", "changhua")}
    ratio = {c: moy[c]["08"] / moy[c]["01"] for c in moy}
    tot = sum(dg[c]["in_grid_GWh_per_yr_2012_2022_all"] for c in lib.COUNTIES)
    cy_share = dg["chiayi"]["in_grid_GWh_per_yr_2012_2022_all"] / tot
    l1 = pd.read_parquet(L1 / "census_pumps.parquet", columns=["county", "in_grid"])
    n = l1["county"].value_counts().to_dict()
    cy_in = int(l1.loc[l1["county"] == "chiayi", "in_grid"].sum())
    pv, lv = lib.rel(PRE), lib.rel(L1)
    rows = []
    for nid, ver, src, note in (
        ("C0.report.changhua_domestic_clean_report", pv, pv,
         f"Changhua domestic 2012-2022: {dom['n_meters_dropped']} of {dom['n_meters']} meters "
         f"dropped by the duty cap; kept {dom['kwh_kept_GWh']:.0f} of "
         f"{dom['kwh_dedup_GWh']:.0f} de-duplicated GWh"),
        ("C0.report.domestic_aug_jan_ratio", pv, pv,
         f"domestic Aug/Jan 2012-2022: changhua {ratio['changhua']:.2f}, "
         f"yunlin {ratio['yunlin']:.2f}"),
        ("C0.report.chiayi_share_in_grid", pv, pv,
         f"Chiayi share of in-grid 2012-2022 energy {cy_share:.4f}"),
        ("C0.report.l1_rows_by_county", lv, lv,
         f"L1 rows {sum(n.values())}: {n}; Chiayi in grid {cy_in}"),
    ):
        rows.append({"number_id": nid, "source": src, "role": "input", "product_version": ver,
                     "note": note})
    ledger.append_rows(lib.WP, rows)
    allr = ledger.read_all()
    bad = ledger.conflicts(allr)
    mine = sum(r["wp"] == lib.WP for r in allr)
    lib.save_json({"c0_rows": mine, "conflicts_all_fragments": bad, "rows_all_fragments":
                   len(allr), "added": rows}, lib.EVIDENCE / "resume" / "report_ledger.json")
    for r in rows:
        print(r["number_id"], "|", r["note"])
    print("C0 rows", mine, "conflicts", len(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
