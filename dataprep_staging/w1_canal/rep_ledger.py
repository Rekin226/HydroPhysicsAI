"""Correct the roles in W1's ledger fragment (W1 repair round, verifier minor issue).

    python -I dataprep_staging/w1_canal/rep_ledger.py

The fragment is rewritten once; the previous fragment is kept unchanged as
``ledger/superseded/W1.r2.csv`` (refuses to run if that file exists). Changes:

* The Jan-May briefing table (wra_cy_subsidence_results_110.pdf) was recorded as
  ``validator`` for the Jan-May benchmark and state-space coverage numbers. Those seven years
  are design-time units (the scout scored them first; seeded look 2026-10-11T16:56Z), so the
  role becomes ``design_time_evidence`` (plan 4.1).
* In the step-6 rows (W1.full_v2.ext2026.*) the Jiji annual table was recorded only as
  ``validator`` (held-out year) or ``input``, although the same file also gives the training
  years' totals that refit beta and the district scales (fit targets) and R, the previous
  year's industrial + public supply (input). One file cannot carry two roles for one number
  (U6), so the rows name the part of the table that plays each role with a ``#`` qualifier,
  as other WPs do: ``#held_out_year_district_total`` (validator),
  ``#training_year_totals`` (fit_target; input for the climatology and persistence
  baselines, which are computed from them), ``#industrial_public_supply`` (input, R).
Every other row, including the pre-registration rows, is kept byte for byte.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import ledger  # noqa: E402
from common.paths import LEDGER_ROWS, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402

BRIEF = rel_to_repo(C.BRIEFING_110_PDF)
JJ = rel_to_repo(C.JIJI_CSV)
NOTE = "role corrected in the W1 repair round (verifier minor issue; superseded/W1.r2.csv)"


def corrected(rows: list[dict]) -> tuple[list[dict], dict]:
    out, n_brief, n_jj = [], 0, 0
    for r in rows:
        n = r["number_id"]
        if (r["source"] == BRIEF and r["role"] == "validator"
                and n.startswith(("W1.preorigin.", "W1.full."))):
            out.append({**r, "role": "design_time_evidence",
                        "note": (r["note"] + "; " if r["note"] else "") + NOTE})
            n_brief += 1
            continue
        if r["source"] == JJ and n.startswith("W1.full_v2.ext2026."):
            n_jj += 1
            parts = []
            if ".loo_annual_rmse." in n or ".loo_boot_vs_climatology." in n:
                parts.append(("#held_out_year_district_total", "validator",
                              "observed annual total of the held-out year"))
                base_only = n.split(".loo_annual_rmse.")[-1].split(".")[0] in (
                    "climatology", "persistence")
                parts.append(("#training_year_totals", "input" if base_only else "fit_target",
                              "the other years' totals: the baseline itself" if base_only else
                              "refit beta and the district scales of the fold"))
                if not base_only:
                    parts.append(("#industrial_public_supply", "input",
                                  "R of the previous year (rain chain) or of the year "
                                  "(actual-inflow reference)"))
            elif ".series_2026." in n:
                parts += [("#training_year_totals", "fit_target",
                           "beta and district scales fitted on 2012-2025"),
                          ("#industrial_public_supply", "input", "R of 2026 = R of 2025")]
            else:
                out.append(r)
                n_jj -= 1
                continue
            for q, role, note in parts:
                out.append({**r, "source": JJ + q, "role": role,
                            "note": note + "; " + NOTE})
            continue
        out.append(r)
    return out, {"briefing_rows_to_design_time_evidence": n_brief, "jiji_rows_split": n_jj}


def main():
    frag = ledger.fragment_path(C.WP)
    sup = LEDGER_ROWS.parent / "superseded" / "W1.r2.csv"
    if sup.exists():
        raise FileExistsError(sup)
    with open(frag, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    new, stats = corrected(rows)
    bad = ledger.conflicts(new)
    if bad:
        raise ledger.LedgerConflict(bad)
    others = [r for r in ledger.read_all() if r.get("wp") != C.WP]
    cross = ledger.conflicts(others + new)
    if cross:
        raise ledger.LedgerConflict(cross)
    sup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frag, sup)
    tmp = frag.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ledger.COLUMNS)
        w.writeheader()
        for r in new:
            w.writerow({k: r.get(k, "") for k in ledger.COLUMNS})
    tmp.replace(frag)
    print({"rows_before": len(rows), "rows_after": len(new), **stats,
           "superseded": rel_to_repo(sup)})


if __name__ == "__main__":
    main()
