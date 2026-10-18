"""Repair round 1, step 0: disclose the independent verifier's looks in the A1 look log before
anything else of the repair round is computed or registered.

The H3 verifier (after the first round, before this repair round) re-ran the after-the-fact
neighbour check of ``s7b_snapshot_consistency.py`` on the 15 admitted wells of
``heads/recovered_wells/v1`` (2012-2022 C20 months; neighbour sets with and without siblings)
and applied ``s8_package.py``'s hold-back criterion to them. It also applied the
``H3.snapshot`` coverage test to each snapshot estimation well's own months. Those numbers
are in the verifier's findings (quoted in the H3 report's Repair round section); they were
read by this repair round's builder before the ``H3.repair.v1`` registration.

``record_look`` cannot back-date a look (only design-time seeds), so the looks are logged now
with the verifier's window in the note; they precede the registration, which lists them as
design-time units.

Usage: python -I r1_looks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

U_STEP = "H3:verifier_neighbour_check_admitted15"
U_SNAP = "H3:verifier_snapshot_power_estimation_wells"


def main():
    v1 = C.REPO / "results/dataprep/heads/recovered_wells/v1/recovered_wells.csv"
    import pandas as pd
    W = pd.read_csv(v1, dtype={"sid": str})
    adm = sorted(W[W.admitted.astype(bool)].sid)
    reg = json.loads((C.EVIDENCE / "registered_wells.json").read_text())
    est = sorted(reg["snapshot"]["estimation_wells"])
    when = ("the verifier's run happened after the H3 first round (last H3 look "
            "2026-10-14T01:18:59Z) and before this repair round; logged now because "
            "record_look cannot back-date")
    out = []
    out.append(prereg3.record_look(
        [U_STEP], "heads:2012-01..2022-12", C.WP, "verification_disclosure",
        members={U_STEP: {
            "description": "independent verifier: s7b neighbour regression (fit 2012-2018 or "
                           "pre-gap, residuals after the gap) re-run on the 15 admitted wells "
                           "of heads/recovered_wells/v1, with and without same-site "
                           "neighbours, and s8's hold-back criterion |bias| > 2 x LOYO RMSE "
                           "applied; per-well biases and ratios read by the repair builder",
            "ids": adm, "window": "2012-01..2022-12"}},
        note=when))
    out.append(prereg3.record_look(
        [U_SNAP], "heads:2012-01..2018-12", C.WP, "verification_disclosure",
        members={U_SNAP: {
            "description": "independent verifier: H3.snapshot's gate-3 coverage test applied "
                           "to each snapshot estimation well's own 2012-2018 months (24 of "
                           "52 fail, mainly over-coverage); read by the repair builder",
            "ids": est, "window": "2012-01..2018-12"}},
        note=when))
    print(json.dumps([{k: r.get(k) for k in ("time", "purpose", "units")} for r in out],
                     indent=1))


if __name__ == "__main__":
    main()
