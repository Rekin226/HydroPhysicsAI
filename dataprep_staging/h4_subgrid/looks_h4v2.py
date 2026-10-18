"""WP H4 repair round: log the repair's looks in the A1 look log (harness.prereg3.record_look).

Usage: python looks_h4v2.py <name> [<name> ...]   (names in LOOKS; each logged once per call)
Scoring looks of the v2 registration are logged by prereg3.assert_can_score in
r2_registered.py, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from harness import prereg3  # noqa: E402

V1_PREREG = "H4.subgrid_operator.v1"
V2_PREREG = "H4.subgrid_operator.v2"
V1_UNIT = "H4:wells158_2020-2022"
D1 = Path(__file__).resolve().parents[2] / \
    "results/dataprep/reports/H4_evidence/repair/design_time/d1.json"


def _candidates() -> list[str]:
    d = json.loads(D1.read_text(encoding="utf-8"))
    return sorted(r["sid"] for r in d["heldout_wells"])


def looks() -> dict:
    return {
        "v1_review": dict(
            units=[V1_UNIT], data_scope="heads:2012-02..2022-12", purpose="after_the_fact",
            prereg_id=V1_PREREG,
            note="repair round: read the v1 report, scores.json and the verifier findings "
                 "(v1 registered results on the 158 wells, 2012-22) before designing v2; no new "
                 "computation on head data"),
        "verifier_disclosure": dict(
            units=[V1_UNIT], data_scope="heads:2012-01..2022-12", purpose="after_the_fact",
            prereg_id=V1_PREREG,
            note="logged by the H4 repair round ON BEHALF OF the independent H4 verifier, who "
                 "states that it read heads/L1_clean/v3 (flags 0-1, 2012-22) at the 158 "
                 "calibration wells to recompute v1 and did not log the look (verifiers may "
                 "not write). Actual time: between the v1 build (2026-10-14T16:30Z) and this "
                 "entry; logged time is this entry's time"),
        "d1": dict(
            units=["H4r:d1_operator_no_heads"], data_scope="census_pumps:2007-01..2022-12",
            purpose="design_time",
            members={"H4r:d1_operator_no_heads": {
                "none": "no head observation read: census pumps, engine/v2 hindcast and "
                        "operator fields, station metadata (coordinates, layer codes)",
                "description": "d1_design_v2.py: v1 kernel split into cell-mean and sub-grid "
                               "parts (verifier diagnostic), sub-grid regressors at the 158 "
                               "wells and the held-out candidates, members, placebo cost"}},
            note="before the v2 pre-registration (reports/H4_evidence/repair/design_time/"
                 "d1.json)"),
        "d1_availability": dict(
            units=["H4r:heldout_availability"], data_scope="heads:2012-02..2022-12",
            purpose="design_time",
            members={"H4r:heldout_availability": {
                "ids": _candidates(), "window": "2012-02..2022-12",
                "description": "month COUNTS of flag 0-1 rows of the held-out candidate wells "
                               "in 2012-02..2019-12 and 2020-01..2022-12 (columns sid, month, "
                               "flag; the value column was not read)"}},
            note="before the v2 pre-registration; availability only, no head value "
                 "(reports/H4_evidence/repair/design_time/d1_availability.json)"),
        "fit_sd": dict(
            units=["H4r:heldout_fitwindow_sd"], data_scope="heads:2012-02..2019-12",
            purpose="design_time",
            members={"H4r:heldout_fitwindow_sd": {
                "ids": _candidates(), "window": "2012-02..2019-12",
                "description": "sd of the observed 2012-02..2019-12 heads of the held-out "
                               "candidates (their fit window: datum, climatology and quartile "
                               "only), to count wells per quartile before registering; no "
                               "2020-22 value read"}},
            note="before the v2 pre-registration (reports/H4_evidence/repair/design_time/"
                 "d2_fit_sd.json)"),
        "t1_power": dict(
            units=["H4r:t1_synthetic_heads"], data_scope="synthetic_heads",
            purpose="design_time",
            members={"H4r:t1_synthetic_heads": {
                "none": "synthetic observations from the hindcast and the operator; no head "
                        "observation read",
                "description": "t1_power_v2.py: multiplier recovery with and without an "
                               "unmodelled seasonal residual, placebo power"}},
            note="before the v2 pre-registration (reports/H4_evidence/repair/design_time/"
                 "t1_power.json)"),
        "v1_after_the_fact_v2kernel": dict(
            units=[V1_UNIT], data_scope="heads:2012-02..2022-12", purpose="after_the_fact",
            prereg_id=V2_PREREG,
            note="r2_registered.py: the v2 operator scored on the v1 unit (158 wells, held-out "
                 "years and A1 site folds, 2020-22). Post-look (H4 v1 scored this unit), so "
                 "design-time evidence only; enters the v2 gate only as a conservative "
                 "no-harm condition declared in the registration"),
        "build": dict(
            units=["H4r:heldout_wells_2020_2022"], data_scope="heads:2020-01..2022-12",
            purpose="product_build", prereg_id=V2_PREREG,
            note="b2_build_product.py packages the v2 registered scores into "
                 "subgrid_operator/v2; no new computation on head data"),
        "report": dict(
            units=["H4r:heldout_wells_2020_2022", V1_UNIT], data_scope="heads:2012-02..2022-12",
            purpose="report_facts", prereg_id=V2_PREREG,
            note="report_h4v2.py reads the v2 registered evidence to write the repair-round "
                 "section; no new fit"),
    }


def main() -> None:
    lk = looks()
    for name in sys.argv[1:]:
        kw = dict(lk[name])
        rec = prereg3.record_look(kw.pop("units"), kw.pop("data_scope"), "H4", kw.pop("purpose"),
                                  **kw)
        print(json.dumps({"name": name, **{k: rec.get(k) for k in ("time", "purpose", "units",
                                                                   "data_scope")}}, indent=1))


if __name__ == "__main__":
    main()
