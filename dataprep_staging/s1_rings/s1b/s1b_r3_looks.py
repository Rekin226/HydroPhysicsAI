"""WP S1b repair round r3 (S1br): look-log records (A1 look log, ``harness.prereg3``).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_looks.py <record>

Records (each logged before the read or computation it covers, unless its note says otherwise):

* ``disclose_session``: a ``design_time`` record of what this session read before its first
  record (reports, the verifier's findings, product cards and the v3 ring product's README,
  which prints the v3 bridged values), logged AFTER those reads (disclosed in the note);
* ``disclose_r2_verifier``: a ``verification_disclosure`` record of the r2 verifier's unlogged
  re-runs (its minor issue 7 asked for it);
* ``diagnose``: before the diagnosis reads S1b's round-1 and r2 evidence files;
* ``design``: before the design reads metadata of the repaired upstreams (step catalogue,
  station table, InSAR masks and segment ids at the ring sites; no height or displacement);
* ``synth``: before the availability-preserving synthetic run (fresh seeds);
* ``second_look``: before the re-run on the spent real units (rings_bridged/v3,
  gnss_clean/full/v4, insar_069A/v3);
* ``after``: after-the-fact diagnostics once the second look is computed.

The real-data units are S1b's registered units (spent in round 1), so every real-data number
of this round is a SECOND LOOK.
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
PREREG_V1 = K.PREREG_ID
DECL_ID = "S1b.r3.rings_v3_second_look.v1"
SCOPE_REAL = "rings+gnss+insar:2020-01..2025-12"


def _rec(units, scope, purpose, members=None, note="", prereg_id=None) -> dict:
    return prereg3.record_look(sorted(units), scope, K.WP, purpose, members=members or None,
                               prereg_id=prereg_id, note=note)


def disclose_session() -> dict:
    m = {"S1b:r3.session_reads_2026-10-18": {
        "none": "reports, verifier findings, product cards and the v3 ring product README "
                "(no S1b statistic computed; no GNSS or InSAR value read)",
        "description": "S1b repair round r3: what the session read before its first look "
                       "record"}}
    note = (
        "S1b repair round r3 (S1br), logged AFTER the reads it covers (about 2026-10-17 "
        "22:55-23:20Z): reports/S1b.md (round 1 and r2, including the per-site 2-sigma "
        "brackets, the plan-literal readings per site and the synthetic expectations), "
        "PHASE0_STATUS.json and the status document's S1 rows, the plan's WP S1 section, the r2 "
        "verifier's full findings (workflow state and its transcript: seed-99 residual-sd "
        "sensitivity, the +-25 % expectations, k=1 power), reports/S1a.repair.md (rings_bridged/"
        "v3: assignments, P(A), pooled scales, changes of the primary bridged values of "
        "-0.56..+0.71 cm), rings_bridged/v3 README.md (prints C_gap_primary_cm and "
        "sigma_C_primary_cm of all 14 sites) and the header of ring_corrections.csv, "
        "S3.repair.md excerpts (full/v4: HNES 2021-04-13 and SJES 2024-09 step changes, KTES "
        "flag-0/1 days, the A3 per-site excess changes), gnss_clean/full/v4 PRODUCT_CARD.md and "
        "MANIFEST fields, S4b.repair.md excerpts (insar_069A/v3: no network gap, M3 reference, "
        "the reference-term flip between windows, 'report both reference variants'), "
        "insar_069A/v3 PRODUCT_CARD.md, network_segments.json and the array names and shapes "
        "of its npz files (no value). S1b's round-1 and r2 evidence files were NOT read yet.")
    return _rec(m, "rings+gnss+insar:2020-01..2025-12", "design_time", m, note)


def disclose_r2_verifier() -> dict:
    note = (
        "Disclosure by S1b repair round r3 (2026-10-18): the independent verifier of S1b's r2 "
        "repair round re-ran s1b_r2_readings, s1b_r2_synth, s1b_r2_u1 and s1b_r2_bridge with "
        "logging disabled, its own read-only re-implementations (GNSS and ring C_pre/C_post at "
        "YCES, KTES, HNES, GFES, TKJS, HLES; InSAR G and B at the 14 sites) and a synthetic "
        "ring-noise sensitivity (s1b_r2_synth.one, seed 99, 200 realizations, ring noise = "
        "ring-fit residual sd), on 2026-10-14 about 01:00-01:20Z through cpu_slot.sh, outputs "
        "in its scratchpad only. Its minor issue 7 asked for this record. Nothing re-scored.")
    return _rec(REG_UNITS, SCOPE_REAL, "verification_disclosure", None, note, PREREG_V1)


def diagnose() -> dict:
    m = {"S1b:r3.diagnose_2026-10-18": {
        "none": "S1b's own round-1 and r2 evidence files (registered rows, readings, synthetic "
                "operating characteristics, U1 and bridge-window files); no product is read",
        "description": "S1b repair round r3: diagnosis of each failed criterion"}}
    note = (
        "S1b repair round r3, logged BEFORE the reads: S1b_evidence/registered_rows.csv, "
        "registered.json, after_the_fact.json, diagnostics.json, "
        "synthetic_operating_characteristics_v2.json and repair/{readings, "
        "synthetic_plan_literal, u1_and_sensitivity, bridge_windows}.json, to diagnose the "
        "failed criteria (plan-literal (c), the plan-literal conjunction, NGL n = 1, d1's "
        "bridged-over-raw clause, power against a 25 % scale error) and the r2 verifier's "
        "major issue. Spent units, numbers already computed; nothing new is scored.")
    return _rec(REG_UNITS + sorted(m), "rings+gnss+insar:2020-01..2025-05", "diagnose", m,
                note, PREREG_V1)


def design() -> dict:
    m = {"S1b:r3.design_metadata_2026-10-18": {
        "none": "metadata of the repaired upstreams at the ring sites: gnss_clean/full/v4 "
                "steps.csv and stations.csv rows, insar_069A/v3 masks, segment ids and frame "
                "source codes at the ring-site pixels, ring survey dates; no height, "
                "displacement or compaction value",
        "description": "S1b repair round r3: design-time metadata reads"}}
    note = (
        "S1b repair round r3, logged BEFORE the reads: the step catalogue (dates, sizes, se, "
        "basis) and station noise parameters of the 10 co-located WRA stations and DNAN in "
        "gnss_clean/full/v4 (round 1 read v2's in its design look), the flag-0/1/3 day counts "
        "per window, and the insar_069A/v3 per-pixel validity, segment ids and frame source "
        "codes in the 3 x 3 windows of the 14 ring sites; ring survey dates of rings_bridged/"
        "v3. No GNSS height, InSAR displacement or ring compaction value enters.")
    return _rec(sorted(m), "rings+gnss+insar_metadata:2020-01..2025-12", "design_time", m, note)


def synth(seed_note: str) -> dict:
    m = {"S1b:r3.synthetic_oc_2026-10-18": {
        "none": "dates, flags, valid-pixel masks, station noise parameters, the GNSS step "
                "catalogue and per-site ring noise levels at the 14 ring sites; every "
                "compaction, height and displacement value is synthetic",
        "description": "S1b repair round r3: availability-preserving synthetic operating "
                       "characteristics on fresh seeds"}}
    note = ("S1b repair round r3: synthetic operating characteristics of every (c) and (d) "
            "reading under two ring-noise models (sd_s and the ring-fit residual sd) and "
            "scenarios correct / +-25 % / other member / raw / mixture, on the v3/v4/v3 "
            "availability. " + seed_note)
    return _rec(sorted(m), "rings+gnss+insar_availability:2020-01..2025-12",
                "integrity_check", m, note, DECL_ID)


def synth_smoke() -> dict:
    m = {"S1b:r3.synthetic_smoke_2026-10-18": {
        "none": "dates, flags, valid-pixel masks, station noise parameters, the GNSS step "
                "catalogue and per-site ring noise levels at the 14 ring sites; every "
                "compaction, height and displacement value is synthetic",
        "description": "S1b repair round r3: smoke run of the synthetic code path (seed 1, "
                       "2 realizations; not the declared seeds)"}}
    note = ("S1b repair round r3, before the declaration: smoke run of s1b_r3_synth.py on the "
            "v3/v4/v3 availability (seed 1, 2 realizations, ring noise N2), to catch code "
            "errors before the code is hashed; output in repair_r3/logs/, never used as "
            "evidence. The ring surveys are read for the residual sd (already known from r2).")
    return _rec(sorted(m), "rings+gnss+insar_availability:2020-01..2025-12",
                "integrity_check", m, note)


def second_look() -> dict:
    note = (
        "S1b repair round r3: SECOND LOOK on the spent S1b units (round-1 score 2026-10-14). "
        "The declared design (S1b.r3.rings_v3_second_look.v1) re-run on rings_bridged/v3, "
        "gnss_clean/full/v4 and insar_069A/v3 (all after_the_fact_analysis): plan-literal "
        "(c) readings, the registered 2-sigma rule (labelled S1b's own), (d1) and (d2), and the "
        "declared variants. Conditional wherever it passes; never a gate; nothing fed back "
        "into S1a.")
    return _rec(REG_UNITS, SCOPE_REAL, "second_look_score", None, note, DECL_ID)


def after(what: str) -> dict:
    m = {f"S1b:r3.after_{what}_2026-10-18": {
        "none": "S1b r3 second-look rows and the S1b units' series (no registered member)",
        "description": f"S1b repair round r3: after-the-fact {what}"}}
    note = (f"S1b repair round r3, after the second look: {what}. Report only; never a gate; "
            f"nothing fed back into S1a.")
    return _rec(REG_UNITS + sorted(m), SCOPE_REAL, "after_the_fact", m, note, DECL_ID)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else ""
    fns = {"disclose_session": disclose_session, "disclose_r2_verifier": disclose_r2_verifier,
           "diagnose": diagnose, "design": design}
    if what not in fns:
        raise SystemExit(f"usage: s1b_r3_looks.py {'|'.join(fns)}")
    rec = fns[what]()
    print(json.dumps({k: rec.get(k) for k in ("time", "purpose", "units", "prereg_id")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
