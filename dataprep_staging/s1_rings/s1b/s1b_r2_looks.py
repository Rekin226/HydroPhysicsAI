"""WP S1b repair round: look-log records (A1 look log, harness.prereg3).

    python -I dataprep_staging/s1_rings/s1b/s1b_r2_looks.py verifier   (once, first)

``verifier``: a ``verification_disclosure`` record of the independent verifier's looks at the
already-scored S1b units (it re-ran the registered computation read-only and recomputed
readings on the registered rows without logging). The repair scripts log their own looks at
start (``readings``, ``synth``, ``u1``) through the functions below, before reading anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

REG_UNITS = ["s1b.c.ngl_near", "s1b.c.wra_colocated", "s1b.d.insar_ring_sites"]


def verifier() -> dict:
    note = (
        "Disclosure by S1b (repair round, 2026-10-14): the independent verifier of WP S1b looked "
        "at the already-scored S1b units after the round-1 report (written 2026-10-14T00:21Z) "
        "and before this repair round started (about 00:34Z); the verifier did not log. It "
        "re-ran s1b_run.compute(PRIMARY) read-only (rings_bridged/v2, gnss_clean/full/v2, "
        "insar_069A/v2 at the 14 ring sites and their stations, 2020-01..2025-05), recomputed "
        "the (c) readings on the registered rows (unwidened bracket, +-30/40 ppm, k = 1), a "
        "sensitivity with the ring-fit residual sd, the GNSS flag-3 day counts of the used "
        "series, GNSS data gaps inside the ring gap (TKJS, STES, HNES, SSES) and the SJES "
        "daily heights around 2022-12-22..2023-02-24, and the GNSS-InSAR gap-change agreement. "
        "No S1b rule, threshold or registered result was changed; nothing is re-scored.")
    return prereg3.record_look(REG_UNITS, "rings+gnss+insar:2020-01..2025-12", K.WP,
                               "verification_disclosure", prereg_id=K.PREREG_ID, note=note)


def readings() -> dict:
    m = {"S1b:repair.readings_2026-10-14": {
        "none": "the registered S1b rows (registered_rows.csv) only; no product is read",
        "description": "after-the-fact readings of S1b.ring_bridge_checks.v1 for the repair "
                       "round"}}
    note = ("Repair round (verifier majors 1, minor 1-2): from registered_rows.csv only, the "
            "plan-literal readings of (c) and (d2) (unwidened pre/post excess-rate bracket; "
            "+-30/40 ppm range extension; |C - C_mid| <= 30/40 ppm), the per-site minimum k and "
            "the family k*, RMSE against each window's implied value, and a dependent-"
            "correlation test and site bootstrap for (d1). Report only; the registered result "
            "stands as registered. Before this record the repair session had read the round-1 "
            "reports/S1b.md and S1b.json, the verifier's findings (task brief), the first four "
            "rows of registered_rows.csv, unverifiable_prior_looks of registered.json, and the "
            "pilot's gnss_vs_ring*.csv (already disclosed in the design look).")
    return prereg3.record_look(REG_UNITS + sorted(m), "rings+gnss+insar:2020-01..2025-05",
                               K.WP, "after_the_fact", members=m, prereg_id=K.PREREG_ID,
                               note=note)


def synth() -> dict:
    m = {"S1b:availability.synthetic_run3_2026-10-14": {
        "none": "dates, flags, valid-pixel masks and the GNSS step catalogue (dates, se) at the "
                "14 ring sites and their stations; no displacement value enters any computation",
        "description": "repair round: availability-preserving synthetic operating "
                       "characteristics of the plan-literal bracket readings (after scoring)"}}
    note = ("Repair round (verifier major 1, minor 1): the round-1 synthetic generator "
            "(s1b_synth.py, imported read-only) with a new seed (1803) and the same availability "
            "reads; every value synthetic. Reports, for the correct member, the other member, "
            "the raw value and a +-25 % scale error, the per-site containment and the family "
            "pass rate under the plan-literal readings (k = 0, +-30/40 ppm) and k = 0.5..2.5, "
            "and the (d1) clause pass rates. Run after the registered scoring; report only.")
    return prereg3.record_look(sorted(m), "rings+gnss+insar_availability:2020-01..2025-12",
                               K.WP, "integrity_check", members=m, note=note)


def u1() -> dict:
    m = {"S1b:repair.u1_and_sensitivity_2026-10-14": {
        "none": "ring sites, their GNSS stations and InSAR pixels; the registered S1b rows "
                "(no registered member)",
        "description": "repair round: flag composition of the validators, InSAR bridged "
                       "increments, U1-strict and ring-noise sensitivities, InSAR vs GNSS "
                       "across the network gaps, GNSS data gaps inside the ring gap, SJES "
                       "endpoint table"}}
    note = ("Repair round (verifier major 2, minors 6-8): reads rings_bridged/v2, gnss_clean/"
            "full/v2 (daily heights and flags) and insar_069A/v2 (multilooked pixels, segment "
            "ids) at the 14 ring sites and their stations, 2020-01..2025-12. Computes the "
            "flag-3 share of the GNSS days used, the LiCSBAS-bridged increments inside each "
            "InSAR gap change, U1-strict readings (GNSS flags 0-1; InSAR bridges at 100 % "
            "uncertainty; d1 from interferometric increments only), the ring-fit residual-sd "
            "variant, InSAR-minus-GNSS changes across the network gaps against within-segment "
            "intervals of the same length, GNSS data gaps inside each ring gap, and SJES "
            "endpoint means. Report only; never a gate, never fed back into S1a.")
    return prereg3.record_look(REG_UNITS + sorted(m), "rings+gnss+insar:2020-01..2025-12",
                               K.WP, "after_the_fact", members=m, prereg_id=K.PREREG_ID,
                               note=note)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fn = {"verifier": verifier}.get(argv[0] if argv else "")
    if fn is None:
        raise SystemExit("usage: s1b_r2_looks.py verifier")
    rec = fn()
    print(json.dumps({k: rec.get(k) for k in ("time", "purpose", "units", "prereg_id")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
