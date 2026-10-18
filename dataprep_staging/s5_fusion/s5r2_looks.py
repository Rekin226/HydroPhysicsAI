"""WP S5 repair round 2 (S5r, 2026-10-18): look log entries outside a registered score.

Every step logs its look here BEFORE it reads data (A1, harness.prereg3):

    python -I s5r2_looks.py diagnose_lev       spent leveling evidence (T1 2020-21 targets, T2
                                               targets OUTSIDE 2018-2019) and leveling surveys
                                               2012-2017 / 2020-2021 for the diagnosis
    python -I s5r2_looks.py diagnose_inst      spent GNSS / InSAR / ring / pile evidence and the
                                               repaired upstream inputs (pre-origin), diagnosis
    (later entries are added below as the round proceeds, each before its step)

The leveling window 2018-01..2019-12 is deliberately excluded from every diagnosis look: it is
the fresh temporal window of this round (never an S5 temporal target).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
from harness import prereg3  # noqa: E402

LEV_UNITS = ["s5.lev.t1_targets", "s5.lev.t2_blocks"]
INST_UNITS = ["s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP"]
SCOPE_LEV_DIAG = "leveling:2012-01..2017-12,2020-01..2021-12"
SCOPE_INST_PRE = "gnss+insar+rings+piles:2012-01..2022-12"
K_PREREG = "S5.repair_r2.v1"

LOOKS = {
    "diagnose_lev": dict(
        units=LEV_UNITS, scope=SCOPE_LEV_DIAG, purpose="diagnose",
        note=("S5 repair round 2, diagnosis BEFORE any method change: (a) the spent T1 targets "
              "(2020-21 withheld surveys; t1_targets.parquet, t1.json) broken down by year, "
              "county, VR, GNSS distance, InSAR support, campaign; regression of the observed "
              "change on the fused and baseline predictions; error decomposition; (b) the spent "
              "T2 targets (t2_targets.parquet) restricted to increments whose both surveys lie "
              "outside 2018-01..2019-12; (c) leveling surveys 2012-2017 and 2020-2021 (rate "
              "persistence across years, campaign structure). The 2018-2019 surveys are not read "
              "in any form (filtered on load).")),
    "diagnose_inst": dict(
        units=INST_UNITS, scope=SCOPE_INST_PRE, purpose="diagnose",
        note=("S5 repair round 2, diagnosis BEFORE any method change: round-1 hyper_pre.json, "
              "T3 evidence and diagnostics; the repaired upstream inputs (gnss_clean/preorigin/v4 "
              "vs v2 monthly tables; insar_069A run P2 vs run P cell series 2020-01..2022-12) "
              "compared with each other and with the round-1 fused field; no leveling value is "
              "read in this look.")),
    "data_build_insarP2": dict(
        units=["s5.insar.runP"], scope="insar:2020-01..2022-12", purpose="data_build",
        note=("S5 repair round 2 (R6): s5r2_insarP2.py builds twin-cell values of S4br's "
              "repaired pre-origin run P2 (started 2026-10-18T03:0xZ, this record logged "
              "seconds after the job started; record_look cannot back-date). Only counts and "
              "coverage are inspected; no fit, no comparison with any other instrument.")),
    "dryrun_r2": dict(
        units=LEV_UNITS + INST_UNITS, scope="subsidence_preorigin:2012-01..2022-12",
        purpose="data_build",
        note=("S5 repair round 2, before the registration S5.repair_r2.v1: dry runs of H2 and "
              "of the re-tests (s5r2_hyper.py / s5r2_tests.py --dry) on the real data "
              "STRUCTURE with SYNTHETIC values (s5r2_dryrun.synth replaces every leveling, "
              "GNSS, InSAR, ring and pile value before any fit). Only shapes, run times, "
              "convergence, the VR geometry (value-independent) and synthetic results are "
              "inspected; no real displacement, residual or score is computed.")),
    "hyper_t1f_lev": dict(
        units=["s5.lev.t2_blocks"], scope="leveling:2012-01..2017-12",
        purpose="hyperparameter_fit", prereg_id=K_PREREG,
        note=("S5 repair round 2, H2 chain 't1f' (registered in S5.repair_r2.v1): stage A on "
              "leveling <= 2017-12 only; the 2018-2019 surveys are not loaded.")),
    "hyper_t1f_inst": dict(
        units=INST_UNITS, scope="gnss+rings+piles:2012-01..2019-12",
        purpose="hyperparameter_fit", prereg_id=K_PREREG,
        note=("S5 repair round 2, H2 chain 't1f': stages B and D on GNSS (preorigin/v4), rings "
              "and printed piles <= 2019-12 with the leveling <= 2017-12; no InSAR.")),
    "hyper_main": dict(
        units=LEV_UNITS + INST_UNITS, scope="subsidence_preorigin:2012-01..2022-12",
        purpose="hyperparameter_fit", prereg_id=K_PREREG,
        note=("S5 repair round 2, H2 chains 'main' (pre-origin: leveling 2012-2021, GNSS v4 <= "
              "2022-12, run P2, rings <= 2021, piles <= 2022-12) and 't1' (stage A on leveling "
              "<= 2019-12), registered in S5.repair_r2.v1; logged AFTER the fresh test T1F was "
              "scored. No score is computed.")),
    "product_build_v4": dict(
        units=LEV_UNITS + INST_UNITS + ["s5.post2022.families"],
        scope="subsidence_all:2012-01..2026-01", purpose="product_build", prereg_id=K_PREREG,
        note=("subsidence_fused/v4 build (s5r2_product.py): full-data fit with the H2 main "
              "hyperparameters and depth fractions, GNSS full/v4 gap segments, insar_069A/v3, "
              "stepped benchmarks and |z| > 5 screening on a first fit, leveling-only fit for "
              "the VR mask. Nothing enters a gate, threshold or hyperparameter (U7).")),
    "after_t1sl": dict(
        units=["s5.lev.t1_targets", "s5.gnss.preorigin", "s5.insar.runP", "s5.cmp.preorigin"],
        scope="leveling+gnss+insar+rings+piles:2012-01..2022-12", purpose="after_the_fact",
        prereg_id=K_PREREG,
        note=("S5 repair round 2, AFTER THE FACT (report only, nothing feeds back): the T1 "
              "second-look configuration refitted without InSAR, without rings/piles and "
              "without GNSS, to attribute its remaining under-coverage.")),
    "report_facts": dict(
        units=LEV_UNITS + INST_UNITS + ["s5.post2022.families", "s5r2.lev.t1f_2018_2019"],
        scope="subsidence_all:2012-01..2026-01", purpose="report_facts", prereg_id=K_PREREG,
        note=("S5 repair round 2: reading the round-2 evidence and the v4 product files to "
              "write reports/S5.repair.md/.json and the ledger rows; no new fit.")),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(LOOKS))
    a = ap.parse_args(argv)
    lk = LOOKS[a.what]
    prereg_id = lk.get("prereg_id", C.PREREG_ID)
    rec = prereg3.record_look(lk["units"], lk["scope"], C.WP, lk["purpose"],
                              prereg_id=prereg_id, members=lk.get("members"), note=lk["note"])
    print(json.dumps({k: rec[k] for k in ("time", "purpose", "units", "prereg_id",
                                          "scope_windows")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
