"""Registered scoring of the WP H2 repair round (pre-registration ``H2.fill_acceptance.v2``).

Same scoring as ``s4_score.py`` (v1, unchanged and imported), with the v2 design parameters
(``design_params_v2.json``) and the v2 fresh realizations (``repair_v2.FRESH_R_V2``). Family
E is gated on the fresh v2 well-gaps that no realization r = 0..4 or 10..19 contains. Each
test first calls ``prereg3.assert_can_score`` (logs the scoring look).

Usage: python -I s4b_score_v2.py TEST_ID [TEST_ID ...]   (or 'all')
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import repair_v2 as R  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402
from h2_heads_fill.s4_score import STRATA, split_conformal, stratum_verdicts  # noqa: E402

FAMILY_OF = {f"H2.fresh2.{f}": f for f in R.FAMILIES_V2}


def seen_e_gaps():
    des = scoring.load_runs("pre", ["E"], {"E": R.E_SEEN_R})
    return set(zip(des.sid, des.stratum, strict=True))


def score_test(tid, params):
    fam = FAMILY_OF[tid]
    ok = prereg3.assert_can_score(R.PREREG_ID_V2, C.WP, test_id=tid)
    raw = scoring.load_runs("pre", [fam], R.FRESH_R_V2)
    p = scoring.product_monthly(raw, params)
    p_all = p
    if fam == "E":
        dk = seen_e_gaps()
        new_gap = np.array([(a, b) not in dk for a, b in zip(p.sid, p.stratum, strict=True)])
        p = p[new_gap]
    clean = p[p.clean.astype(bool)]
    verdict = stratum_verdicts(clean)
    missing = [s for s in STRATA[fam] if s not in verdict]
    rs = R.FRESH_R_V2[fam]
    rep = {"test_id": tid, "family": fam, "realizations": [rs.start, rs.stop - 1],
           "assert_can_score": {"sha256": ok["sha256"], "inputs_verified": ok["inputs_verified"],
                                "post_registration_looks": ok["post_registration_looks"],
                                "unverifiable_prior_looks": len(ok["unverifiable_prior_looks"])},
           "strata_clean": verdict, "strata_missing": missing,
           "pass": (not missing) and all(v["pass"] for v in verdict.values()),
           "reported": {
               "all_truth": {"coverage": scoring.coverage_rows(p),
                             "rmse": scoring.rmse_table(
                                 p, ("product",) + scoring.BASELINES + scoring.REPORTED
                             ).to_dict("records")},
               "model_wells_clean": {"coverage": scoring.coverage_rows(
                   clean[clean.in_model.astype(bool)]),
                   "rmse": scoring.rmse_table(clean[clean.in_model.astype(bool)],
                                              ("product",) + scoring.BASELINES
                                              ).to_dict("records")},
               "split_conformal_within_fresh_clean": split_conformal(clean),
               "n_cells_all": int(len(raw)), "n_cells_clean_scored": int(len(clean))}}
    if fam == "E":
        fk = set(zip(raw.sid, raw.stratum, strict=True))
        ca = p_all[p_all.clean.astype(bool)]
        rep["reported"]["well_gaps_seen_before"] = {
            "fresh_well_gaps": len(fk), "seen_in_r0-4_or_r10-19": len(fk & dk),
            "gated_on": "fresh v2 well-gaps not in any realization r = 0..4 or 10..19"}
        rep["reported"]["all_fresh_E_clean"] = {
            "coverage": scoring.coverage_rows(ca),
            "rmse": scoring.rmse_table(ca, ("product",) + scoring.BASELINES).to_dict("records")}
    scoring.jdump(rep, R.REG_V2 / f"{tid}.json")
    print(tid, "PASS" if rep["pass"] else "FAIL",
          {s: (round(v["coverage_gate"]["coverage"], 3), v["coverage_pass"], v["rmse_pass"])
           for s, v in verdict.items()}, flush=True)
    return rep


def main(argv):
    params = json.loads(R.PARAMS_V2.read_text())
    tids = list(FAMILY_OF) if argv == ["all"] else argv
    for tid in tids:
        score_test(tid, params)


if __name__ == "__main__":
    main(sys.argv[1:])
