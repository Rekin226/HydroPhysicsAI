"""WP P6 report (results/dataprep/reports/P6.md and P6.json) and ledger fragment rows.

Reads the published ``public_wells/v<N>`` (validation.json, MANIFEST.json, member_annual.csv,
closures.csv), the guard probes and the checker output written by ``finalize_p6.py``, and the
after-the-fact comparison of the agency-credited cuts with WRA's Jan-May public-well series
(natural_experiments/v1, opened as after_the_fact_analysis). Every number quoted in the report
gets ledger rows (U6) with explicit sources and one role each.

Usage: python -I report_p6.py <product_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, ledger  # noqa: E402
from common.paths import REPORTS_DIR, rel_to_repo  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402

EV = K.EVID


def after_the_fact_n1(clos: pd.DataFrame) -> dict:
    """Agency-credited cuts (closures.csv) against WRA's Jan-May public-well pumping read
    off the 110/112 briefing charts by N1 (two reads; after the fact, never fed back)."""
    p = guard.open_product(K.N1_PRODUCT, "after_the_fact_analysis")
    d = pd.read_csv(p.path("public_well_janmay.csv"))
    s = d[d.township == "(county)"].pivot_table(index="year", columns=["county", "agency"],
                                                values="value_1e4m3")
    out = {}
    for county, ag, key in (("Yunlin", "Taiwan Water Corporation (台水)", "tw_yunlin"),
                            ("Changhua", "Taiwan Water Corporation (台水)", "tw_changhua"),
                            ("Yunlin", "Irrigation Agency (農水署)", "ia_yunlin"),
                            ("Changhua", "Irrigation Agency (農水署)", "ia_changhua")):
        ser = s[(county, ag)]
        cred = clos[clos.cohort.str.startswith(key + "_") & (clos.status == "disposed")]
        out[key] = {"janmay_1e4m3": {int(y): float(v) for y, v in ser.items() if y <= 2021},
                    "credited_cut_1e4m3_per_yr_by_year": {int(r.year): float(
                        r.volume_1e4m3_per_yr or 0) for r in cred.itertuples()},
                    "credited_total_1e4m3_per_yr": float(cred.volume_1e4m3_per_yr.sum())}
    return out


def main(argv=None) -> int:
    d = Path(sys.argv[1] if argv is None else argv[0]).resolve()
    val = json.loads((d / "validation.json").read_text(encoding="utf-8"))
    man = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    fin = json.loads((EV / "finalize.json").read_text(encoding="utf-8"))
    clos = pd.read_csv(d / "closures.csv")
    ann = pd.read_csv(d / "member_annual.csv")
    atf = after_the_fact_n1(clos)
    (EV / "after_the_fact_n1_janmay.json").write_text(
        json.dumps(atf, ensure_ascii=False, indent=1), encoding="utf-8")
    from p6_public_wells import report_text as RT
    md, js, rows = RT.compose(d, val, man, fin, clos, ann, atf)
    (REPORTS_DIR / "P6.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "P6.json").write_text(json.dumps(js, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    have = {(r["number_id"], r["source"]) for r in ledger._read(ledger.fragment_path("P6"))}
    new = [r for r in rows if (r["number_id"], ledger.normalise_source(r["source"]))
           not in have]
    if new:
        ledger.append_rows("P6", new)
    print(f"wrote P6.md / P6.json; ledger rows added {len(new)} (total "
          f"{len(ledger._read(ledger.fragment_path('P6')))}); "
          f"report {rel_to_repo(REPORTS_DIR / 'P6.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
