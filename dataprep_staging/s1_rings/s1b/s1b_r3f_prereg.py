"""WP S1b repair round r3, fix round: declaration of the step-policy grid (harness.prereg3).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_prereg.py --dry-run
    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_prereg.py

Writes ``results/dataprep/preregistration/S1b.r3f.step_policy_symmetry.v1.json`` (kind
``declaration``: after the fact on spent units, no registered test). It fixes, before the
computation, the grid, the readings, the interpretation rules of the Fix round's F.2 in
``reports/S1b.repair.md`` (its F.1 and F.2 snapshots are hashed), the inputs and the code.
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
from s1_rings.s1b import s1b_r3f_symmetry as SYM  # noqa: E402

STAGING = REPO / "dataprep_staging"
CODE = [
    "s1_rings/__init__.py", "s1_rings/s1lib.py", "s1_rings/s1const.py",
    "s1_rings/s1b/__init__.py", "s1_rings/s1b/s1b_const.py", "s1_rings/s1b/s1b_core.py",
    "s1_rings/s1b/s1b_io.py", "s1_rings/s1b/s1b_run.py", "s1_rings/s1b/s1b_synth.py",
    "s1_rings/s1b/s1b_r2_core.py", "s1_rings/s1b/s1b_r2_readings.py",
    "s1_rings/s1b/s1b_r3_io.py", "s1_rings/s1b/s1b_r3_after.py",
    "s1_rings/s1b/s1b_r3_snapshot.py", "s1_rings/s1b/s1b_r3f_looks.py",
    "s1_rings/s1b/s1b_r3f_symmetry.py", "s1_rings/s1b/s1b_r3f_prereg.py",
    "s1_rings/s1b/tests/test_s1b_r3f.py", "common/guard.py", "common/manifest.py",
    "common/uses.py", "common/paths.py", "common/hashing.py", "harness/prereg3.py"]
DATA = [
    "results/dataprep/rings_bridged/v3/MANIFEST.json",
    "results/dataprep/rings_bridged/v3/ring_corrections.csv",
    "results/dataprep/gnss_clean/full/v4/MANIFEST.json",
    "results/dataprep/gnss_clean/full/v4/wra/daily.parquet",
    "results/dataprep/gnss_clean/full/v4/steps.csv",
    "results/dataprep/gnss_clean/full/v4/stations.csv",
    "results/dataprep/gnss_clean/full/v2/MANIFEST.json",
    "results/dataprep/gnss_clean/full/v2/wra/daily.parquet",
    "results/dataprep/gnss_clean/full/v2/steps.csv",
    "results/dataprep/gnss_clean/full/v2/stations.csv",
    "results/dataprep/insar_069A/v3/MANIFEST.json",
    "results/dataprep/insar_069A/v2/MANIFEST.json",
    "results/dataprep/reports/S1b_evidence/repair_r3/synthetic/oc_N1_sd_s.json",
    "results/dataprep/reports/S1b_evidence/repair_r3/synthetic/oc_N2_resid.json",
    "results/dataprep/reports/S1b_evidence/repair_r3/retest/second_look.json"]


def snapshots() -> list[str]:
    d = K.EVIDENCE / "repair_r3" / "snapshots"
    recs = [json.loads(x) for x in (d / "index.jsonl").read_text().splitlines() if x.strip()]
    want = {"fix_f1_diagnosis", "fix_f2_design"}
    got = [r for r in recs if r["label"] in want]
    if {r["label"] for r in got} != want:
        raise SystemExit("F.1 and F.2 snapshots are required before the declaration")
    return [r["path"] for r in got]


def doc() -> dict:
    code = [{"path": rel_to_repo(STAGING / c), "sha256": sha256_file(STAGING / c)} for c in CODE]
    na = prereg3.not_applicable
    return {
        "id": SYM.DECL_ID, "wp": K.WP, "kind": "declaration",
        "title": ("S1b r3 fix round: after-the-fact GNSS step-policy grid of check (c) on "
                  "rings_bridged/v3 (validators gnss_clean/full/v2 and v4 x step policies "
                  "product / catalogue_in_gap / catalogue_all), replacing the one-sided r3 "
                  "'both GNSS artefacts removed' statement"),
        "tests": na("declaration only: after the fact on S1b's spent round-1 units (10 WRA "
                    "pairs); report only, never a gate, opens nothing"),
        "zones": na("no zones; ring sites"),
        "priors": na("no prior"),
        "burn_in": na("no model run"),
        "folds": na("no fitting across sites"),
        "fresh_units": na("no fresh real unit exists; no synthetic run in this declaration "
                          "(the r3 synthetic families are reused for count probabilities)"),
        "thresholds": {
            "family_rule": "per family: >= 80 % of evaluable sites contain C_primary AND "
                           "RMSE(primary) < RMSE(raw) against the bracket midpoint (unchanged)",
            "readings": list(SYM.READINGS),
            "s1b_registered_rule": "k = 2 (S1b's own rule; not the plan's), labelled"},
        "gate_scores": na("no gate"),
        "primary_model": {
            "code": code,
            "configuration": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in SYM.PRIMARY.items()},
            "pipeline": "the registered round-1 functions s1b_run.compute and s1b_core, "
                        "unchanged, with rings_bridged/v3",
            "grid": {"versions": {"v2": "gnss_clean/full/v2 (with insar_069A/v2, unused "
                                        "here)", "v4": "gnss_clean/full/v4 (with "
                                                       "insar_069A/v3, unused here)"},
                     "policies": {
                         "product": "every step the validator corrects",
                         "catalogue_in_gap": "as product, but every in-gap step whose basis is "
                                             "not metadata_or_catalogue is added back (r3's "
                                             "'artefacts removed' policy)",
                         "catalogue_all": "only steps with basis metadata_or_catalogue are "
                                          "corrected; every other step is added back at every "
                                          "date (value + size from the step date); added-back "
                                          "steps carry no variance; flags unchanged"},
                     "station_set": "SJES excluded in all six cells (plan catalogue: unusable "
                                    "across the gap); catalogue_all also with SJES kept"},
            "outputs": "per cell: counts and family-rule outcome per reading, RMSE, per-site "
                       "C_pre/C_post/in-out; P(N = n_obs | s) and LR correct : +-25 % against "
                       "the r3 synthetic families (9-site wra_noSJES; 10-site wra), "
                       "approximate for v2 and catalogue_all; add-back integrity per station"},
        "inputs": [{"path": c["path"]} for c in code] + DATA + snapshots(),
        "interpretation": (
            "1. No cell is selected: the artefact-removal result is the minimum and maximum "
            "count over the six SJES-excluded cells per reading, with the per-site table. "
            "2. The statement 'with the GNSS artefacts removed, range +-30/40 ppm pass' is made "
            "only if both readings pass the family rule in all six cells; otherwise the headline "
            "says that the count depends on the GNSS step policy and gives the range, and the "
            "r3 8/9 is quoted only as the v4/catalogue_in_gap cell. 3. Likelihood ratios are "
            "quoted only as a range over the cells. 4. Nothing changes a threshold, a "
            "registered result, a verdict of Part 4.6 or the status; nothing is fed back into "
            "S1a."),
        "design_time_exposure": [
            "the r3 second look (all its cells, including v4/catalogue_in_gap and the "
            "decomposition v3 product with round-1 validators) was computed and read",
            "the r3 verifier's estimate (about 7/9 for catalogue_in_gap on v2) and the fix "
            "round's own prediction (F.1 point 5) were written before this computation",
            "the v2 and v4 step catalogues of the 10 stations were read (F.1 look 02:26:24Z)",
            "consequence: the grid is entirely after the fact; it measures dependence on the "
            "step policy and carries no evidential weight about the product beyond that"],
        "anti_leakage": "validators only; rings_bridged/v3 read, never written; nothing fed "
                        "back into S1a",
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if "--dry-run" in argv:
        print("inputs", len(d["inputs"]), "valid")
        return 0
    path, sha = prereg3.write_prereg(d)
    print(rel_to_repo(path), sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
