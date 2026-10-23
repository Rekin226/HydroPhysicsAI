"""WP S1b repair round r3 (S1br): the declaration of the second-look design (harness.prereg3).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_prereg.py --dry-run
    python -I dataprep_staging/s1_rings/s1b/s1b_r3_prereg.py

Writes ``results/dataprep/preregistration/S1b.r3.rings_v3_second_look.v1.json`` (kind
``declaration``: no registered test, because S1b's real units were scored in round 1). It fixes,
before any computation on the repaired products: the inputs (hashed), the code (hashed, so the
run scripts refuse to run if any file changes), the configuration, the readings, the declared
variants and decompositions, the synthetic seeds, and the interpretation rules of Part 2 of
``reports/S1b.repair.md`` (its snapshot is hashed too).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r3_looks as LK  # noqa: E402
from s1_rings.s1b import s1b_r3_run as RR  # noqa: E402
from s1_rings.s1b import s1b_r3_synth as SY  # noqa: E402

STAGING = REPO / "dataprep_staging"
CODE = [
    "s1_rings/__init__.py", "s1_rings/s1lib.py", "s1_rings/s1const.py",
    "s1_rings/s1b/__init__.py", "s1_rings/s1b/s1b_const.py", "s1_rings/s1b/s1b_core.py",
    "s1_rings/s1b/s1b_io.py", "s1_rings/s1b/s1b_run.py", "s1_rings/s1b/s1b_synth.py",
    "s1_rings/s1b/s1b_r2_core.py", "s1_rings/s1b/s1b_r2_readings.py",
    "s1_rings/s1b/s1b_r2_looks.py", "s1_rings/s1b/s1b_r3_io.py", "s1_rings/s1b/s1b_r3_run.py",
    "s1_rings/s1b/s1b_r3_synth.py", "s1_rings/s1b/s1b_r3_after.py",
    "s1_rings/s1b/s1b_r3_looks.py", "s1_rings/s1b/s1b_r3_snapshot.py",
    "s1_rings/s1b/s1b_r3_diagnose.py", "s1_rings/s1b/s1b_r3_prereg.py",
    "s1_rings/s1b/tests/test_s1b_r3.py", "common/guard.py", "common/manifest.py",
    "common/uses.py", "common/paths.py", "common/hashing.py", "harness/prereg3.py"]
DATA = [
    "results/dataprep/rings_bridged/v3/MANIFEST.json",
    "results/dataprep/rings_bridged/v3/ring_corrections.csv",
    "results/dataprep/gnss_clean/full/v4/MANIFEST.json",
    "results/dataprep/gnss_clean/full/v4/wra/daily.parquet",
    "results/dataprep/gnss_clean/full/v4/ngl/daily.parquet",
    "results/dataprep/gnss_clean/full/v4/stations.csv",
    "results/dataprep/gnss_clean/full/v4/steps.csv",
    "results/dataprep/gnss_clean/full/v4/common_mode.parquet",
    "results/dataprep/insar_069A/v3/MANIFEST.json",
    "results/dataprep/insar_069A/v3/vertical_ml.npz",
    "results/dataprep/insar_069A/v3/vertical_cells.npz",
    "results/dataprep/rings_bridged/v2/MANIFEST.json",
    "results/dataprep/gnss_clean/full/v2/MANIFEST.json",
    "results/dataprep/insar_069A/v2/MANIFEST.json",
    "chou-shui-data/data/mlcw_stations.csv",
    "results/twin_forward/datum_gate_innov.npz",
    "results/dataprep/reports/S1b_evidence/registered_rows.csv",
    "results/dataprep/reports/S1b_evidence/repair_r3/diag/diagnosis.json"]


def snapshots() -> list[str]:
    d = K.EVIDENCE / "repair_r3" / "snapshots"
    recs = [json.loads(x) for x in (d / "index.jsonl").read_text().splitlines() if x.strip()]
    want = {"part1_diagnosis", "part2_design"}
    got = [r for r in recs if r["label"] in want]
    if {r["label"] for r in got} != want:
        raise SystemExit("Part 1 and Part 2 snapshots are required before the declaration")
    return [r["path"] for r in got]


def constants() -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(K).items()
            if k.isupper() and isinstance(v, (int, float, str, tuple))
            and k not in ("WP", "PREREG_ID")}


def doc() -> dict:
    code = [{"path": rel_to_repo(STAGING / c), "sha256": sha256_file(STAGING / c)} for c in CODE]
    na = prereg3.not_applicable
    return {
        "id": LK.DECL_ID, "wp": K.WP, "kind": "declaration",
        "title": ("S1b repair round r3: second look of checks (c) and (d) on the repaired "
                  "products (rings_bridged/v3, gnss_clean/full/v4, insar_069A/v3) under the "
                  "plan-literal readings, with S1b's 2-sigma rule labelled beside them, and "
                  "fresh-seed synthetic operating characteristics of every reading"),
        "tests": na("declaration only: S1b's real units (10 WRA pairs, DNAN, 4 neighbour pairs, "
                    "14 InSAR sites, the one 2022 gap) were scored in round 1 "
                    "(S1b.ring_bridge_checks.v1), so every real-data number of this round is a "
                    "declared SECOND LOOK, conditional where it passes and never a gate"),
        "zones": na("no zones; ring sites"),
        "priors": {"report_only_evidence_line": "S1a's s1lib.line_lr against the plan's "
                   "reference groups (never fed back)"},
        "burn_in": na("no model run"),
        "folds": na("no fitting across sites"),
        "fresh_units": na("no fresh real unit exists (Part 1, 1.6); the only fresh units are "
                          "the synthetic seeds 18031 / 18032 (truth draws 18131 / 18132), "
                          "which calibrate the readings and do not test the product"),
        "thresholds": {
            "family_rule": "per family: >= 80 % of evaluable sites contain C_primary AND "
                           "RMSE(primary) < RMSE(raw) against the bracket midpoint; judged only "
                           "with >= 3 evaluable sites (unchanged)",
            "plan_literal_c_rerun": "reading k0 (unwidened excess-rate range) on the WRA "
                                    "family under the family rule: the criterion that failed "
                                    "(3/10), re-run as a second look",
            "other_readings": list(R.RULES),
            "plan_literal_readings": list(R.PLAN_LITERAL),
            "s1b_registered_rule": "k = 2 (S1b's own rule of S1b.ring_bridge_checks.v1; not the "
                                   "plan's), labelled wherever quoted",
            "d1": "r_bridged >= one-sided 5 % critical r AND r_bridged > r_raw (unchanged)",
            "plan_literal_conjunction": "plan-literal (c) (k0) AND d1",
            "verdict_row": "the governing reading (plan-literal or S1b's rule) is a user "
                           "decision; no reading is made governing here"},
        "gate_scores": na("no gate: validation only; nothing opens a use of any product"),
        "primary_model": {
            "code": code,
            "configuration": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in RR.PRIMARY.items()},
            "constants": constants(),
            "pipeline": "the registered round-1 functions s1b_run.compute / gnss_row / "
                        "insar_site / d1 and s1b_core, unchanged, through s1b_r3_io.Adapter",
            "insar_pixel_rule_v3": IO_RULE,
            "insar_cell_variants": "flag-0 cell values of value (M3) and "
                                   "value_round1_plane_reference (plane)",
            "variants": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                             for kk, vv in v.items()} for k, v in RR.VARIANTS.items()}
            | {"insar_cells_plane": {"insar_source": "cells", "cell_values": "plane"},
               "ring_noise_resid_sd": {"sd_s_mm": "pre-window ring-fit residual sd"}},
            "decompositions": ["rings_bridged/v2 with gnss_clean/full/v4 and insar_069A/v3",
                               "rings_bridged/v3 with gnss_clean/full/v2 and insar_069A/v2"],
            "synthetic": {"models": SY.MODELS, "truth_seed_offset": SY.TRUTH_SEED_OFFSET,
                          "realizations": SY.N_REAL, "scenarios": list(SY.SCEN),
                          "families": list(SY.FAMILIES),
                          "ring_noise": "N1 sd_s; N2 pre-window ring-fit residual sd; bracket "
                                        "sigma always with the product's sd_s",
                          "insar_noise": "round 1: 6.6 mm white per epoch, no bridges",
                          "order": "computed before the second look"},
            "real_vs_synthetic": "per reading, model and family: P(N = n_obs | s), "
                                 "P(N <= n_obs | s) for every scenario; LR correct : +-25 % "
                                 "and drawn mixture : +-25 %; zero frequency -> '< 1/400', no "
                                 "ratio"},
        "inputs": [{"path": c["path"]} for c in code] + DATA + snapshots(),
        "interpretation": (
            "Every real-data number is a SECOND LOOK on units scored in round 1, made with the "
            "round-1 brackets and the v3 bridged values known (Part 1, 1.6): a pass is "
            "'conditional (second look)' and opens no use; a fail is a fail of a second look. "
            "A reading is called discriminating under a ring-noise model when a correct product "
            "passes S1b's family rule at >= 0.5 and a uniform +-25 % scale error at <= 0.3 "
            "(fresh-seed synthetic); otherwise its real outcome is reported without inference "
            "about the scale. No reading is made governing: whether the plan-literal reading "
            "or S1b's 2-sigma rule governs remains a user decision. The products checked and "
            "both validators are gate-failed and read as after_the_fact_analysis, so nothing "
            "here opens a use of rings_bridged; nothing is fed back into S1a (validators only). "
            "U7: the checks use 2023-25 observations by necessity (flagged, unchanged)."),
        "design_time_exposure": [
            "round-1 registered rows (per-site brackets, readings) and r2 evidence, read for "
            "the diagnosis (look 2026-10-17T23:18:20Z)",
            "rings_bridged/v3 README (prints C_gap_primary_cm of all 14 sites) and "
            "S1a.repair.md, read before the first look record (disclosed in the 23:18:19Z "
            "design_time record)",
            "S3.repair.md (v4 step changes at HNES, SJES, KTES; A3 per-site excess changes) "
            "and S4b.repair.md (the M3 reference error in 2021-22), read before the first "
            "record",
            "v4 step catalogue and flag counts, v3 InSAR masks and source codes at the ring "
            "sites (metadata look 23:24:37Z)",
            "the r2 verifier's seed-99 residual-sd numbers and +-25 % expectations",
            "consequence: much of the second look is predictable; it carries no more weight "
            "than a second look"],
        "anti_leakage": ("validators only: rings_bridged/v3 is read, never written; no S1a "
                         "parameter, member or rule is changed; membership evidence is "
                         "reported, never used"),
    }


IO_RULE = ("3 x 3 window around the nearest multilooked pixel; a pixel is usable when it lies in "
           "licsbas15_mask_primary and has no bridged epoch (v2: segment ids equal to the "
           "network's at every epoch); at an epoch its value counts when finite and its source "
           "code is 1, 2 or 3 (code 4, untied second frame, excluded); station value = median "
           "of >= 4 values; one network segment (n_bridges 0, one post-gap intercept)")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if "--dry-run" in argv:
        print(json.dumps({k: (v if k not in ("primary_model", "inputs") else "...")
                          for k, v in d.items()}, ensure_ascii=False, indent=1)[:4000])
        print("inputs", len(d["inputs"]), "valid")
        return 0
    path, sha = prereg3.write_prereg(d)
    print(rel_to_repo(path), sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
