"""Pre-registration of the WP H2 repair round (``H2.fill_acceptance.v2``; A1 ``prereg3``).

Written after the v2 design (``s2b_design_v2.py``) and before any v2 realization is generated.
Registers five fill tests (families N, S, E, H, T on realizations never generated before)
with the v2 design parameters. The daily-bridge test H2.fresh.d (v1, passed) and the
initial-condition decision H2.ic.lowo (v1) carry over unchanged: the bridge configuration and
the initial-condition code are identical (their v1 records are hashed inputs).

Usage: python -I s3b_prereg_v2.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import io_h2  # noqa: E402
from h2_heads_fill import repair_v2 as R  # noqa: E402
from h2_heads_fill.s3_prereg import CLUSTERS, CODE, DATA, DESC, HARNESS, SCOPE  # noqa: E402

CODE_V2 = CODE + ["repair_v2.py", "s2b_design_v2.py", "s3b_prereg_v2.py", "s4b_score_v2.py"]
DATA_V2 = DATA + [
    "results/dataprep/preregistration/H2.fill_acceptance.v1.json",
    "results/dataprep/reports/H2_evidence/registered/H2.fresh.d.json",
    "results/dataprep/reports/H2_evidence/registered/H2.ic.lowo.json",
    "results/dataprep/reports/H2_evidence/design_v2/design_params_v2.json"]
STRATA_V2 = {"N": "N_net1, N_net2", "S": "S01, S02, S03, S06, S12, S24",
             "E": "E_lead66, E_trail48 (gated on the v2 well-gaps that no realization "
                  "r = 0..4 or 10..19 contains; all v2 E cells reported)",
             "H": "H_half12", "T": "T_S, T_H06"}


def fresh_unit(fam):
    rs = R.FRESH_R_V2[fam]
    return f"H2:fresh2_{fam}_r{rs.start}-{rs.stop - 1}"


def doc(h, p2):
    ids = sorted(h.sids)
    tests = []
    for fam in R.FAMILIES_V2:
        u = fresh_unit(fam)
        rs = R.FRESH_R_V2[fam]
        tests.append({
            "test_id": f"H2.fresh2.{fam}", "data_scope": SCOPE, "fresh_units": [u],
            "external_units": {u: {
                "description": f"family {fam} ({DESC[fam]}) realizations r = {rs.start}.."
                               f"{rs.stop - 1} on the pre-origin L1 v3 matrix; never run or "
                               f"scored before this registration (for family E only the well "
                               f"selection of r = 30..69 was computed beforehand, from the "
                               f"observation mask alone, to size the unseen-well stratum); "
                               f"truth cells are C20 months of these wells",
                "ids": ids, "window": "2012-01..2022-12"}},
            "metric": f"per stratum ({STRATA_V2[fam]}), clean truth: (a) coverage of the "
                      f"product's 90 % interval (pred +- 1.6449 sd) by rule gate-3 "
                      f"(harness.coverage3.coverage_gate, clusters = {CLUSTERS[fam]}); (b) "
                      f"pooled RMSE of the product below that of each baseline: the twin's "
                      f"backfill, climatology + AR(1), space-time kriging of standardized "
                      f"anomalies, and same-layer IDW of anomalies on the cells where IDW is "
                      f"defined (product on the same cells)",
            "threshold": "every stratum: coverage gate passes AND the product's RMSE is "
                         "strictly lower than every defined baseline's"})
    return {
        "id": R.PREREG_ID_V2, "wp": C.WP, "kind": "test",
        "title": "WP H2 repair round: sd factors re-estimated after the v1 coverage failures; "
                 "acceptance on realizations never generated before",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H2, 2.4-2.5, 4.4, 4.5",
        "tests": tests,
        "zones": prereg3.not_applicable("point records; no zones"),
        "priors": prereg3.not_applicable("no priors"),
        "burn_in": prereg3.not_applicable("no model run"),
        "folds": prereg3.not_applicable("artificial-gap realizations; clusters per family "
                                        "are given in each test's metric"),
        "fresh_units": [fresh_unit(f) for f in R.FAMILIES_V2],
        "design_time_units": ["H2:design_{N,S,E,H,T,d}_r0-4", "H2:D_febmar_2013-2022",
                              "H2:fresh_N_r10-19", "H2:fresh_S_r10-19", "H2:fresh_E_r10-19",
                              "H2:fresh_H_r10-29", "H2:fresh_T_r10-29 (v1 fresh units, scored "
                              "under H2.fill_acceptance.v1 and design-time for v2)"],
        "thresholds": {"coverage": "rule gate-3 (as v1)", "rmse": "as v1",
                       "carried_over": "H2.fresh.d (v1, passed) and H2.ic.lowo (v1 decision "
                                       "'gp') stand: bridge configuration and IC code unchanged"},
        "gate_scores": [{"name": f"H2.fresh2.{f}", "rule": "pass/fail", "weight": 1 / 5}
                        for f in R.FAMILIES_V2],
        "gate_rule": "conjunctive: the v2 product gate passes only if all five v2 fill tests "
                     "pass in every stratum and the carried-over v1 test H2.fresh.d passed (it "
                     "did); equal weights only satisfy the schema. If it fails, no further "
                     "partial release beyond the v1 packaging rule.",
        "primary_model": "the v1 configuration with the v2 sd factors per gap class "
                         f"{p2['factor_by_class']} (design_params_v2.json); methods "
                         f"{p2['method_by_class']}; bridge sds {p2['bridge_sd_by_class']} "
                         "(unchanged)",
        "inputs": [str((C.CODE_DIR / f).relative_to(C.REPO)) for f in CODE_V2] + HARNESS
        + DATA_V2,
        "reported_not_gated": ["all-truth tables", "model-well subset",
                               "within-fresh split-conformal multiplier per stratum",
                               "family E on all v2 cells", "every method's RMSE"],
        "interpretation": (
            "Repair round after the registered v1 test failed on coverage in N_net1 (0.812), "
            "S24 (0.939), E_lead66 (0.844, point window) and E_trail48 (0.960); RMSE passed "
            "in all 14 strata, and H, T and d passed. This is a post-look repair: the v2 "
            "factors were re-estimated with the v1 rule on the union of the v1 design "
            "realizations and the v1 fresh realizations, which are design-time for v2. "
            "Nothing else changed. The v2 realizations (seeds 1000*ord(F)+r, T 7000+r, r >= "
            "30) were never generated before this registration; the A1 tool cannot verify "
            "that for external units (A1 verifier's open issue), so the evidence of freshness "
            "is the seed ranges and the look log. Thresholds are the plan's, unchanged. The "
            "E stratum is gated on wells whose leading or trailing gap appeared in no earlier "
            "realization (about 25 and 29 wells: calibrated by A1's k >= 20 rule but with "
            "little power). With 14 strata a calibrated product fails some stratum by chance "
            "about half the time; the result is reported as it comes. No observation dated "
            "2023-01 or later entered any factor, threshold or choice (U7)."),
    }


def main(argv):
    p2 = json.loads(R.PARAMS_V2.read_text())
    h = io_h2.load("pre", with_daily=False)
    d = doc(h, p2)
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if argv and argv[0] == "--dry-run":
        print("valid; tests:", [t["test_id"] for t in d["tests"]])
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main(sys.argv[1:])
