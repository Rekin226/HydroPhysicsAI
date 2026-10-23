"""WP S5 repair round (2026-10-17): look log entries outside a registered score.

Every step of the repair round logs its look here BEFORE it reads data (A1, harness.prereg3):

    python -I s5r_looks.py verifier_disclosure   the verifier's looks, logged on its behalf
    python -I s5r_looks.py repair_inspect        GNSS daily-record gaps (dates and flags only),
                                                 leveling survey dates, the v1 product files
    python -I s5r_looks.py stage2_hyper          monthly-anomaly (stage 2) hyperparameters on
                                                 pre-origin InSAR run P (GNSS-out annual fit)
    python -I s5r_looks.py product_build_v2      subsidence_fused/v2 build (full data)
    python -I s5r_looks.py repair_diagnostics    after-the-fact diagnostics of the repair round
    python -I s5r_looks.py report_facts          reading the evidence to write the report

Registered units of ``S5.subsidence_fusion.v1`` carry their members in that pre-registration;
no new unit is declared here.
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

PRE_UNITS = ["s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP", "s5.lev.t2_blocks"]
ALL_UNITS = PRE_UNITS + ["s5.post2022.families"]
SCOPE_ALL = "subsidence_all:2012-01..2026-01"
SCOPE_PRE = "subsidence_preorigin:2012-01..2022-12"

LOOKS = {
    "verifier_disclosure": dict(
        units=ALL_UNITS + ["s5.lev.t1_targets"], scope=SCOPE_ALL,
        purpose="verification_disclosure",
        note=("Disclosure by S5 (repair round, 2026-10-17): the independent verifier of WP S5 "
              "looked at the already-scored S5 units after the round-1 report (S5.md written "
              "2026-10-17T02:50Z) and before this repair round; the verifier did not log. It "
              "(1) regenerated the T1 predictions from the registered code and hyper_t1.json "
              "and recomputed T1, T2, T3 and R2 from the saved target tables; (2) re-ran the "
              "full-data product fit and compared it with subsidence_fused/v1; (3) re-fitted "
              "the full-data product without SJES GNSS months >= 2023 and compared the 2022 "
              "and 2023 annual increments and the 2023-2026 change with v1 (cells within 3 and "
              "10 km of SJES, counts of cells moved by > 5 and > 10 mm); (4) re-ran "
              "s5diag.f_tradeoff and the VR-mask summaries; (5) re-ran the unit tests. Logged "
              "now because record_look cannot back-date.")),
    "repair_inspect": dict(
        units=ALL_UNITS, scope=SCOPE_ALL, purpose="diagnose",
        note=("S5 repair round, before the v2 declaration: (a) GNSS daily records of "
              "gnss_clean/full/v2 and preorigin/v2 (WRA, NGL, NCKU): dates and flags only, to "
              "list the gaps longer than 30 days per station (no height value, residual or "
              "fit is read); (b) leveling panel survey dates and counts per benchmark (no "
              "elevation); (c) the v1 product files (fused_annual.npz, validation.json) to "
              "restate the verifier's SJES and mask findings in the report. No fit, "
              "prediction or score is computed.")),
    "dryrun_monthly": dict(
        units=ALL_UNITS, scope=SCOPE_PRE, purpose="data_build",
        note=("S5 repair round, before the registration S5.monthly_field.v1: dry run of the "
              "monthly-field procedure and test (s5r_dryrun.py) on the real data STRUCTURE "
              "with SYNTHETIC values (s5dryrun's synthetic truth plus a synthetic AR(1) "
              "monthly anomaly, or none): every leveling, GNSS, InSAR, ring and pile value is "
              "replaced before any fit. Only counts, shapes, run times and the synthetic "
              "results are inspected; no real displacement, residual or score is computed.")),
    "dryrun_product": dict(
        units=ALL_UNITS, scope=SCOPE_ALL, purpose="data_build",
        note=("S5 repair round, before the registration S5.monthly_field.v1: run-time and "
              "memory check of the product's stage-2 path (72 months, all cells) on the FULL "
              "data structure with SYNTHETIC values (s5dryrun's synthetic truth; every value "
              "replaced before any fit; the annual fit uses round-1 code). Only shapes, run "
              "times and memory are inspected; no real displacement, residual or score.")),
    "stage2_hyper": dict(
        units=["s5.cmp.preorigin", "s5.insar.runP", "s5.lev.t2_blocks"], scope=SCOPE_PRE,
        purpose="hyperparameter_fit", prereg_id="S5.monthly_field.v1",
        note=("S5 repair round, monthly field (stage 2): the pre-origin annual fit WITHOUT "
              "GNSS (leveling 2012-2021, InSAR run P, rings <= 2021, piles <= 2022-12; "
              "hyper_pre.json) and the restricted marginal likelihood of the InSAR run-P "
              "month-group residuals under the stage-2 model (AR(1)-in-month x Matern "
              "anomaly + exponential APS + white, plane per month projected out), maximised "
              "over sig_ins2, sig_aps2, ell_aps2, sig_a, rho, ell_a. Logged after the "
              "registration S5.monthly_field.v1 and before the fit; no GNSS value is read and "
              "no score is computed.")),
    "product_build_v2": dict(
        units=ALL_UNITS, scope=SCOPE_ALL, purpose="product_build",
        note=("subsidence_fused/v2 build (s5r_product.py) under the declaration "
              "S5.subsidence_fused_rules.v2: full-data annual fit with the pre-origin "
              "hyperparameters and depth fractions, GNSS offset segments at daily-record gaps "
              "> 30 d, stepped-benchmark offsets, GNSS and ring/pile screening by one |z| > 5 "
              "pass on the full fit; the leveling-only fit for the VR mask; the rings/piles-out "
              "and v1-operator sensitivity fits; the stage-2 monthly anomaly on run F "
              "residuals (2020-01..2025-12) with the pre-origin stage-2 hyperparameters. No "
              "pre-origin variant is fitted. Nothing here enters a gate, threshold, "
              "hyperparameter or member choice (U7).")),
    "product_build_v3": dict(
        units=ALL_UNITS, scope=SCOPE_ALL, purpose="product_build",
        prereg_id="S5.subsidence_fused_rules.v3",
        note=("subsidence_fused/v3 build (s5r_product.py) under S5.subsidence_fused_rules.v3: "
              "the same fits and values as v2 (same code path, data, hyperparameters, M1 "
              "result), rebuilt so that the monthly sd label and the card are corrected (label "
              "amendment). Nothing enters a gate, threshold, hyperparameter or member choice.")),
    "correct_r1_product_note": dict(
        units=ALL_UNITS, scope=SCOPE_ALL, purpose="history_disclosure",
        note=("Correction (repair round) of the look 2026-10-17T02:37:23Z (product_build, "
              "subsidence_fused/v1): its note says the build included a 'pre-origin variant "
              "fit with the same code on pre-origin data'. That clause is wrong: s5product.py "
              "fits no pre-origin variant (only the full-data fit, the leveling-only fit for "
              "the VR mask and the rings/piles-out sensitivity fit). The pre-origin fits of S5 "
              "are those of procedure H and the registered tests. The look log is append-only, "
              "so the correction is a new record.")),
    "repair_diagnostics": dict(
        units=ALL_UNITS + ["s5.lev.t1_targets"], scope=SCOPE_ALL, purpose="after_the_fact",
        note=("S5 repair round, after-the-fact diagnostics (report only; nothing feeds back "
              "into v2, a gate or a threshold): v1-operator vs v2 increments near SJES and "
              "elsewhere; the stepped-benchmark list; a continuation of the stage-A optimiser "
              "from the returned vertex and the product sensitivity to it; T2 re-run with a "
              "10 km buffer around each held-out block (design-exposed; plan 4.3); a stage-A "
              "variant with per-campaign leveling offsets (q at its bound?).")),
    "report_facts": dict(
        units=ALL_UNITS + ["s5.lev.t1_targets"], scope=SCOPE_ALL, purpose="report_facts",
        note=("S5 repair round: s5r_report.py reads the registered evidence, the repair "
              "evidence and the v1/v2 product files to write the repair-round section of "
              "reports/S5.md and S5.json and the ledger number list; no new fit.")),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(LOOKS))
    a = ap.parse_args(argv)
    lk = LOOKS[a.what]
    prereg_id = lk.get("prereg_id", C.PREREG_ID)
    rec = prereg3.record_look(lk["units"], lk["scope"], C.WP, lk["purpose"],
                              prereg_id=prereg_id, note=lk["note"])
    print(json.dumps({k: rec[k] for k in ("time", "purpose", "units", "prereg_id")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
