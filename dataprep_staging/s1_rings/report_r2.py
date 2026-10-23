"""WP S1a repair round (r2): the ledger fragment.

    python -m s1_rings.report_r2          (from dataprep_staging/)

Ledger: the round-1 fragment is kept unchanged in ``ledger/superseded/S1a.r1.csv``. The new
fragment keeps every round-1 row except two corrected roles (the verifier's minor issue and
its analogue), adds the rows of rings_bridged/v2 and of the repair-round numbers, and is
written only if no conflict exists across all fragments. The report (S1a.md, S1a.json) is
written by hand from the evidence files.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import LEDGER_ROWS, rel_to_repo  # noqa: E402

from s1_rings import report as R1  # noqa: E402
from s1_rings import s1const as K  # noqa: E402

WP = "S1a"
V1 = K.REPO / "results" / "dataprep" / "rings_bridged" / "v1"
V2 = K.REPO / "results" / "dataprep" / "rings_bridged" / "v2"
REP = K.EVIDENCE / "repair"
DBM = "results/dataprep/deep_benchmarks/v2"
PRN = "results/dataprep/deep_benchmarks_printed/v2"
#: (number_id, source) -> corrected role and the note appended (round-1 rows and v2 rows).
ROLE_FIX = {
    ("S1a.b_pile_scale", DBM): (
        "input", "r2: was 'validator'; the pile line is a group-evidence input of the assignment "
                 "that check (b) depends on, so it is not an independent check (plan 4.1)"),
    ("S1a.path_2022", PRN): (
        "input", "r2: was 'validator'; the printed values decide the path method through the "
                 "feed check, they do not check the path values"),
}


def fix_roles(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        key = (r["number_id"], ledger.normalise_source(r["source"]))
        if key in ROLE_FIX and r["role"] != ROLE_FIX[key][0]:
            r["role"], note = ROLE_FIX[key]
            r["note"] = (r.get("note", "") + "; " + note).strip("; ")
        out.append(r)
    return out


def r2_rows() -> list[dict]:
    l0 = [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]
    syn = rel_to_repo(K.EVIDENCE / "synthetic" / "synthetic_rows.parquet")
    res = rel_to_repo(K.EVIDENCE / "synthetic" / "result.json")
    pre = "results/dataprep/preregistration/S1a.e_synthetic_mixed.v1.json"
    v1 = "results/dataprep/rings_bridged/v1"
    v2 = "results/dataprep/rings_bridged/v2"
    rows = []

    def add(num, src, role, note, ver="rings_bridged/v2"):
        rows.append({"number_id": f"S1a.{num}", "source": src, "role": role,
                     "product_version": ver, "note": note})

    for num, note in (("r2.e_decomposition", "scale/membership decomposition of the registered "
                                              "(e) rows and counterfactuals (after the fact)"),
                      ("r2.e_bdeep_bias", "b_deep error per group and window (after the fact)")):
        add(num, syn, "input", note)
        add(num, res, "input", "registered result (p_A, windows)")
        add(num, pre, "input", "registered design")
    for f in l0:
        add("r2.e_window_coverage", f, "input", "site entry rules of the registered windows "
            "(surveys <= 2021-12)")
        add("r2.a_power_decomposition", f, "input", "S_fp and rate brackets of the real gap")
        for num in ("r2.residual_offset_both_members", "r2.b_implied_by_pile_vote",
                    "r2.plan_offset_ranges_sensitivity", "r2.tuku_path_2022"):
            add(num, f, "input", "real-gap ring surveys (after the fact)")
    add("r2.a_power_decomposition", v1, "input", "validation.json and ring_corrections.csv")
    for num in ("r2.residual_offset_both_members", "r2.b_implied_by_pile_vote",
                "r2.plan_offset_ranges_sensitivity"):
        add(num, DBM, "input", "pile line of the assignment (after_the_fact_analysis read)")
    add("r2.tuku_path_2022", DBM, "input", "TKJS(1) vendor daily values (path shape)")
    add("r2.tuku_path_2022", PRN, "input", "feed check with the 20-day screen decides the method")
    add("r2.v1_v2_compare", v1, "input", "every output table compared")
    add("r2.v1_v2_compare", v2, "input", "every output table compared")
    return rows


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_fragment() -> list[dict]:
    old = read_csv(LEDGER_ROWS / f"{WP}.csv")
    rows = fix_roles(old)
    rows += fix_roles(R1.ledger_rows(V2, {}))
    rows += r2_rows()
    seen, out = set(), []
    for r in rows:
        k = tuple(str(r.get(c, "") or "") for c in ("number_id", "source", "role",
                                                    "product_version", "note"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def write_fragment(rows: list[dict]) -> dict:
    others = [r for p in sorted(LEDGER_ROWS.glob("*.csv")) if p.stem != WP
              for r in read_csv(p)]
    with tempfile.TemporaryDirectory(dir=LEDGER_ROWS.parent) as td:
        ledger.append_rows(WP, rows, rows_dir=td)       # validates, refuses own conflicts
        new = read_csv(Path(td) / f"{WP}.csv")
        bad = ledger.conflicts(others + new)
        if bad:
            raise SystemExit(f"cross-fragment conflicts: {bad[:10]}")
        os.replace(Path(td) / f"{WP}.csv", LEDGER_ROWS / f"{WP}.csv")
    allrows = ledger.read_all()
    return {"rows": len(new), "rows_all_fragments": len(allrows),
            "conflicts_all_fragments": len(ledger.conflicts(allrows)),
            "superseded": "results/dataprep/ledger/superseded/S1a.r1.csv"}


def main() -> None:
    frag = write_fragment(build_fragment())
    (REP / "ledger_r2.json").write_text(json.dumps(frag, indent=1))
    print(json.dumps(frag))


if __name__ == "__main__":
    main()
