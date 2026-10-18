"""WP S5 repair round: ledger fragment (plan 4.1, U6).

    python -I s5r_ledger.py rebuild_r1   move rows/S5.csv to ledger/superseded/S5.r1.csv and
                                         write the round-1 rows again with corrected roles
    python -I s5r_ledger.py repair       append the repair-round numbers
                                         (S5_evidence/repair/ledger_numbers_r2.json)

Role correction (verifier, minor): T1 and T2 are registered tests on leveling units that A1
rules are not fresh, so their withheld values are design-time evidence, not validation. The
T1/T2 target tables (and the T1/T2 breakdowns) therefore carry the role
``design_time_evidence``; the L0 leveling file stays the training ``input`` of those numbers
(the target table holds exactly the withheld rows of that file, noted in each row).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5ledger as L1  # noqa: E402  (round-1 rules, read-only)
import s5r_const as R  # noqa: E402
import s5r_vconst as V  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import LEDGER_ROWS, rel_to_repo  # noqa: E402

SUPERSEDED = LEDGER_ROWS.parent / "superseded" / "S5.r1.csv"
DTE_NOTE = (" | role corrected in the repair round: the T1/T2 leveling units are not fresh "
            "(A1), so the withheld rows (this target table, a subset of the L0 leveling file "
            "that is the training input) are design_time_evidence")


def rebuild_r1() -> dict:
    frag = LEDGER_ROWS / "S5.csv"
    SUPERSEDED.parent.mkdir(exist_ok=True)
    if frag.exists() and not SUPERSEDED.exists():
        os.rename(frag, SUPERSEDED)
    with open(SUPERSEDED, newline="", encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    S = L1.sources()
    dte = {S["t1"], S["t2"]}
    rows = []
    for r in old:
        r = {k: r[k] for k in ledger.COLUMNS}
        nid = r["number_id"]
        if (nid.startswith(("S5.T1.", "S5.T2.", "S5.diag.t1", "S5.diag.t2"))
                and r["role"] == "validator" and r["source"] in dte):
            r["role"] = "design_time_evidence"
            r["note"] = r["note"] + DTE_NOTE
        rows.append(r)
    p = ledger.append_rows(C.WP, rows)
    return {"fragment": str(p), "superseded": rel_to_repo(SUPERSEDED), "rows": len(rows),
            "roles_changed": sum(1 for r in rows if r["role"] == "design_time_evidence")}


# ============================================================================ repair numbers
def sources_r() -> dict:
    S = L1.sources()
    ev = R.EVIDENCE_R
    pv2 = C.DATAPREP / C.PRODUCT
    S.update({
        "gnss_daily": [rel_to_repo(C.GNSS_FULL / v) for v in V.GNSS_DAILY.values()],
        "hyper_s2": rel_to_repo(ev / "hyper_stage2.json"),
        "m1_targets": rel_to_repo(ev / "m1_targets.parquet"),
        "prereg_m1": rel_to_repo(C.DATAPREP / "preregistration" / f"{R.MTEST_ID}.json"),
        "decl_v2": rel_to_repo(C.DATAPREP / "preregistration" / f"{R.DECL_ID}.json"),
        "v1_product": rel_to_repo(pv2 / "v1" / "fused_annual.npz"),
        "t2b": rel_to_repo(ev / "diag_t2buffer_targets.parquet"),
    })
    return S


def rows_r(pv: str, numbers: dict) -> list[dict]:
    S = sources_r()
    rws = L1.rows
    pre_obs_nognss = [S["lev"], S["insar_p"], S["piles"]] + S["rings"]
    pre_aux = [S["ring_st"], S["gnss_full"], S["grid"]]
    full_obs = [S["lev"], S["gnss_full"], S["insar_v2"], S["piles"], S["rings_bridged"]] + \
        S["rings"]
    full_aux = [S["ring_st"], S["grid"], S["hyper_pre"], S["hyper_s2"], S["decl_v2"]]
    out = []
    for nid, note in numbers.items():
        if nid.startswith("S5.M1.stage2"):
            out += rws(nid, pv, note, fit_target=[S["insar_p"]],
                       input=[S["lev"], S["piles"]] + S["rings"] + pre_aux
                       + [S["hyper_pre"], S["prereg_m1"]])
        elif nid.startswith("S5.M1."):
            out += rws(nid, pv, note, input=pre_obs_nognss + pre_aux
                       + [S["hyper_pre"], S["hyper_s2"], S["prereg_m1"]],
                       design_time_evidence=[S["gnss_pre"], S["m1_targets"]])
        elif nid.startswith("S5.v2.dryrun"):
            out += rws(nid, pv, note, input=[S["lev"], S["gnss_pre"], S["insar_p"],
                                              S["piles"]] + S["rings"] + pre_aux
                       + [S["hyper_pre"]])
        elif nid.startswith("S5.v2.compare_v1"):
            out += rws(nid, pv, note, input=full_obs + full_aux + S["gnss_daily"]
                       + [S["v1_product"]])
        elif nid.startswith("S5.v2."):
            out += rws(nid, pv, note, input=full_obs + full_aux + S["gnss_daily"]
                       + [S["units"]])
        elif nid.startswith("S5.diag2.stageA"):
            out += rws(nid, pv, note, fit_target=[S["lev"]],
                       input=[S["hyper_pre"]] + full_obs[1:] + full_aux + S["gnss_daily"])
        elif nid.startswith("S5.diag2.campaign"):
            out += rws(nid, pv, note, fit_target=[S["lev"]], input=[S["hyper_pre"], S["units"],
                                                                     S["grid"]])
        elif nid.startswith("S5.diag2.t2buffer"):
            out += rws(nid, pv, note, input=[S["lev"], S["gnss_pre"], S["insar_p"], S["piles"]]
                       + S["rings"] + pre_aux + [S["hyper_pre"]],
                       design_time_evidence=[S["t2b"]])
        else:
            raise ValueError(f"no ledger rule for {nid}")
    return out


def repair() -> dict:
    nums = json.loads((R.EVIDENCE_R / "ledger_numbers_r2.json").read_text())
    rws = rows_r(nums["product_version"], nums["numbers"])
    p = ledger.append_rows(C.WP, rws)
    allr = ledger.read_all()
    return {"fragment": str(p), "rows_added_or_present": len(rws),
            "numbers": len(nums["numbers"]), "conflicts_all_fragments":
            len(ledger.conflicts(allr)), "rows_all_fragments": len(allr)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rebuild_r1", "repair"])
    a = ap.parse_args(argv)
    res = rebuild_r1() if a.what == "rebuild_r1" else repair()
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
