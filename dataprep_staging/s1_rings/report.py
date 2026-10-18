"""WP S1a report (results/dataprep/reports/S1a.md, S1a.json) and ledger rows.

    python -m s1_rings.report results/dataprep/rings_bridged/v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import REPORTS_DIR, from_repo, rel_to_repo  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

WP = "S1a"


def load(d: Path) -> dict:
    return {"val": json.loads((d / "validation.json").read_text()),
            "man": json.loads((d / "MANIFEST.json").read_text()),
            "corr": pd.read_csv(d / "ring_corrections.csv"),
            "syn": json.loads((K.EVIDENCE / "synthetic" / "result.json").read_text()),
            "path": json.loads((K.EVIDENCE / "path_test" / "path_terms.json").read_text()),
            "gnss": json.loads((K.EVIDENCE / "gnss_sensitivity.json").read_text()),
            "chk": json.loads((K.EVIDENCE / f"checks_{d.name}.json").read_text())}


def ledger_rows(d: Path, X: dict) -> list[dict]:
    ver = f"{K.PRODUCT}/{d.name}"
    l0 = [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]
    dbm = "results/dataprep/deep_benchmarks/v2"
    prn = "results/dataprep/deep_benchmarks_printed/v2"
    gns = "results/dataprep/gnss_clean/full/v1"
    pil = "results/dataprep/_pilots/subsidence_gap_sources"
    rows = []

    def add(num, src, role, note=""):
        rows.append({"number_id": f"S1a.{num}", "source": src, "role": role,
                     "product_version": ver, "note": note})

    for num, note in [
        ("assignment", "group votes, membership probabilities, assigned/unassigned sites"),
        ("pooled_scale_A", "pooled b of the assigned group-A sites"),
        ("pooled_scale_B", "pooled b of the assigned group-B sites"),
        ("gap_compaction_primary", "bridged gap compaction per site (primary)"),
        ("a_pooled_z", "check (a) pooled standardized difference and power"),
        ("residual_offset", "residual offset across the gap after correction"),
        ("misassignment_sensitivity", "offset-only groups and offset-LR 3/30"),
    ]:
        for f in l0:
            add(num, f, "fit_target" if num.startswith("pooled") else "input", note)
        if num in ("assignment", "gap_compaction_primary", "pooled_scale_A",
                   "pooled_scale_B", "misassignment_sensitivity"):
            add(num, dbm, "input", "pile-implied scale line (after_the_fact_analysis read)")
        add(num, pil, "design_time_evidence", "pilot saw these sites (seeded look)")
    for f in l0:
        add("b_pile_scale", f, "input", "ring intervals for the pile-implied scale")
        add("path_2022", f, "input", "2022 path endpoints and climatology")
        add("path_terms", f, "fit_target", "family-Y masked years 2017-2020 (pre-origin)")
        add("e_worst_window_rmse", f, "input", "synthetic truth: surveys <= 2021-12")
        add("e_misclassification", f, "input", "synthetic truth: surveys <= 2021-12")
        add("e_coverage", f, "input", "synthetic truth: surveys <= 2021-12")
    add("b_pile_scale", dbm, "validator", "check (b): pile-implied scale vs pooled b_deep; "
        "not independent where the pile is a group line or shapes the path")
    add("path_2022", dbm, "input", "pile shape at Tuku, Xizhou, Guangfu, Xiutan (vendor daily)")
    add("path_2022", prn, "validator", "2022 feed months checked against printed 007001 values")
    add("feed_check_2022", prn, "validator", "printed monthly 007001 (gate passed)")
    add("feed_check_2022", dbm, "input", "vendor daily monthly-mean increments")
    add("path_terms", dbm, "input", "pre-origin run, design_time_evidence 2022-12 slice")
    for num in ("e_worst_window_rmse", "e_misclassification", "e_coverage"):
        add(num, dbm, "input", "pile line in the synthetic pipeline (pre-origin run)")
        add(num, "results/dataprep/preregistration/S1a.e_synthetic_mixed.v1.json", "input",
            "registered design")
    add("e_worst_window_rmse", pil, "design_time_evidence",
        "threshold 1.5 cm set after the pilot's 1.35 cm (post hoc); injection constants")
    for f in l0:
        add("gnss_sensitivity", f, "input", "ring gaps and rates")
    add("gnss_sensitivity", gns, "input", "after_the_fact_analysis read (gate failed); not "
        "used by the product")
    return rows


def acceptance(X: dict) -> list[dict]:
    v, syn = X["val"], X["syn"]
    a, b, r = v["a_pooled_standardized_difference"], v["b_pile_implied_scale"], \
        v["residual_offset"]
    p = syn["primary"]
    btxt = "; ".join(f"{x['site']} {x['b_pile_ppm']} ppm vs {x['tolerance']} "
                     f"({'pass' if x['pass'] else 'FAIL'})" for x in b["sites"] if x.get("tested"))
    sens = v["misassignment_sensitivity"]["offset_only"]
    ych = [x for x in sens if x["site"] == "元長國小"][0]
    nchg = sum(1 for x in sens if x["delta_cm"] and abs(x["delta_cm"]) >= 0.5)
    return [
        {"criterion": "(e) worst-window RMSE of the bridged gap compaction, mixed A/B synthetic "
                      "rerun with classification (registered S1a.e_synthetic_mixed.v1, primary "
                      "variant)",
         "threshold": "<= 1.5 cm (POST HOC: set by the plan after the pilot's 1.35 cm)",
         "measured": f"{100 * p['worst_window_rmse_m']:.2f} cm (window "
                     f"{p['worst_window']['start']}..{p['worst_window']['end']}, bias "
                     f"{p['worst_window']['bias_cm']:.2f} cm; {p['n_windows']} windows; median "
                     f"window {p['median_window_rmse_cm']:.2f} cm; misclassification "
                     f"{p['misclassification_rate']:.3f}, unassigned "
                     f"{p['unassigned_share']:.3f})",
         "passed": bool(p["pass"])},
        {"criterion": "(a) pooled standardized difference between bridged and rate-midpoint gap "
                      "compaction over 14 sites, with power for a 25 % scale error",
         "threshold": "|Z| < 2 and power >= 0.8",
         "measured": f"Z = {a['pooled_z']:.2f}, power {a['power_25pct']:.3f} "
                     f"({a['n_inside_rate_bracket']}/14 inside the rate bracket, reported only)",
         "passed": bool(a["pass"])},
        {"criterion": "(b) pile-implied scale against the pooled scale of the site's group",
         "threshold": "group A within +-25 % of the pooled A scale; group B within +-30 ppm",
         "measured": btxt + f" (pooled A {b['pools']['A']['mean_ppm']} ppm, B "
                            f"{b['pools']['B']['mean_ppm']} ppm; not independent: plan 4.1)",
         "passed": bool(b["pass"])},
        {"criterion": "residual offset across the gap after correction",
         "threshold": "within +-1 cm at >= 12 of 14 sites",
         "measured": f"{r['n_within']}/{r['n']} within 1 cm (max |offset| "
                     f"{max(abs(x['offset_cm']) for x in r['sites']):.2f} cm)",
         "passed": bool(r["pass"])},
        {"criterion": "misassignment sensitivity (groups by offset cluster alone; change in "
                      "bridged gap compaction for every site, Yuanchang in particular)",
         "threshold": "reported (no threshold in the plan)",
         "measured": f"{nchg} sites change by >= 0.5 cm; Yuanchang {ych['C_primary_cm']} -> "
                     f"{ych['C_offset_only_cm']} cm (offset-only group "
                     f"{ych['group_offset_only']})",
         "passed": True},
        {"criterion": "product gate (conjunction of (e), (a), (b), residual offset)",
         "threshold": "all pass", "measured": json.dumps(v["gate"]),
         "passed": bool(v["gate"]["passed"])},
        {"criterion": "manifest, guard and table checks (checks.py)",
         "threshold": "manifest valid, hashes reproduce, guard opens exactly the allowed uses, "
                      "scoring reads flags 0-1 only, every non-observed row has an sd",
         "measured": f"ok={X['chk']['ok']}; guard_as_expected={X['chk']['guard_as_expected']}",
         "passed": bool(X["chk"]["ok"])},
    ]


def main(argv=None) -> None:
    d = from_repo((argv or sys.argv[1:])[0])
    X = load(d)
    rows = ledger_rows(d, X)
    ledger.append_rows(WP, rows)
    acc = acceptance(X)
    (REPORTS_DIR / "S1a.acceptance.json").write_text(json.dumps(acc, indent=1,
                                                                ensure_ascii=False))
    print(json.dumps(acc, indent=1, ensure_ascii=False))
    print("ledger rows", len(rows))


if __name__ == "__main__":
    main()
