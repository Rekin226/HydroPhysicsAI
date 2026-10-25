"""Write N2.policy_response.prospective_2026-11.v1 (harness.prereg3) BEFORE the window opens.

The prospective test of the frozen page model's head response to RECORDED Yunlin pumping
anomalies, 2026-11..2027-10, drafted on 2026-10-24 (results/dataprep/reports/
N1_gaps_2026-10-24/science-15/, power check there) and approved by the user on 2026-10-24
(page model primary, R0 declared secondary; U4 extended to a separate recorded-pumping
counterfactual arm). The code that produces the arm is commit 1fff145 (branch
twin/n2-pumping-arm, `twin.forward --recorded-pumping`), rehearsed on 2023-01..2026-08
before registration (results/dataprep/reports/N2_policy/rehearsal/).

Usage: python -I prereg_n2.py [--write]   (default: validate and dry-run)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
DRAFT = (REPO / "results/dataprep/reports/N1_gaps_2026-10-24/science-15/"
         "N2.policy_response.prospective_2026-11.v1.DRAFT.json")
ZR = REPO / "results/twin/prospective_2026-11/predictions.csv"
R0_ZR = REPO / "results/twin/prospective_2026-11_r0"
REH = REPO / "results/dataprep/reports/N2_policy/rehearsal"
HOOK_COMMIT = "1fff145"
HOOK_FILES = {  # sha256 at commit 1fff145 (branch twin/n2-pumping-arm)
    "hydrophysics/twin/recorded_pumping.py":
        "4a8c99e47c902496ac8959ca6c76ac3320ade27ccaa3793063e3854ce0c36ecb",
    "hydrophysics/twin/forward.py":
        "088b8bd5775f2f019b0b35b52fbebb5e543c3d3b06171a0dd54b0fa7031461bd",
    "hydrophysics/twin/scenario.py":
        "c6f250fcce5a3a146091c3d531a847bca1da3503dc6e5146a001e5aa064e7997",
    "tests/test_twin_recorded_pumping.py":
        "118fa5b81372ff78dd9a6137abeb928bcb59222ba52eb5f06d01494ee298b58d",
}
DROP = ("status", "drafted", "drafted_by", "not_registered_statement",
        "registration_deadline", "preconditions_before_registration", "missing_inputs_at_drafting")

FORWARD = ("PYTHONPATH=<checkout of 1fff145> python -P -m hydrophysics.twin.forward "
           "--theta results/twin_runs/stage3_datum_gate/stage3_theta.json "
           "--vep-json results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json "
           "--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
           "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
           "--column-hpc0-fast-days 365 --device cpu --save-members none "
           "--recorded-pumping 'PR=<r.csv>@2023-01' --out <scoring dir>/forward_pr_page")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def doc() -> dict:
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    for k in DROP:
        d.pop(k, None)
    ids = sorted(pd.read_csv(ZR, dtype={"station_id": str})["station_id"].unique())
    unit = d.pop("unit")
    uid = "N2:heads_prospective_2026-11_2027-10"
    d["declared_secondary_model"]["zr_predictions"] = {
        p.name: {"path": rel_to_repo(p), "sha256": sha(p)} for p in sorted(R0_ZR.iterdir())}
    d["declared_secondary_model"]["note"] = (
        "R0's ZR arm: results/twin/prospective_2026-11_r0 (twin.prospective register on "
        "datum_gate_cwa.npz, same stations, wells and window as the page's registration; the "
        "same command reproduces the page's registered predictions byte for byte, sha 16ac3c37)")
    d.update({
        "id": "N2.policy_response.prospective_2026-11.v1", "wp": "N2", "kind": "test",
        "tests": [{
            "test_id": "N2P.heads_yunlin_response", "data_scope": "heads:2026-11..2027-10",
            "fresh_units": [uid],
            "external_units": {uid: {"description": unit["description"], "ids": ids}},
            "what": "Do the observed Yunlin head anomalies of 2026-11..2027-10 follow the frozen "
                    "page model's response to the RECORDED pumping anomaly (PR - ZR)?",
            "metric": d["primary_metric"], "decision_rule": d["decision_rule"],
            "informative_gate": d["informative_gate"], "seed": 20261101,
            "label": "prospective: the window's heads did not exist at registration"}],
        "unit_detail": unit,
        "rulings": {
            "2026-10-24 user": "register N2 with the page model (datum_gate_innov) primary and "
                               "R0 as the declared secondary",
            "U4 extension (user, 2026-10-24)": "recorded post-origin pumping may feed a SEPARATE "
                                               "counterfactual arm of the frozen physics model; "
                                               "the registered forecast (ZR) is unchanged",
            "changhua": "Changhua wells are report-only: the page model carries no Changhua "
                        "pumping. A Changhua arm may be added only by a separate registration "
                        "written before 2026-10-31 16:00 UTC (rung I1.B.v1 pending)"},
        "code": {"commit": HOOK_COMMIT, "branch": "twin/n2-pumping-arm", "files": HOOK_FILES,
                 "forward_command": FORWARD,
                 "r_builder": {"path": "dataprep_staging/n2_policy/build_r.py",
                               "sha256": sha(Path(__file__).with_name("build_r.py")),
                               "usage": "--x1 <X1 county_monthly.parquet> --first 2023-01"},
                 "contrast": "c = heads_mean[PR] - heads_mean[baseline] of the SAME forward run "
                             "at each well's cell and layer (never against the GPU-written "
                             "datum_gate_innov.npz); PR prediction = registered ZR + c"},
        "rehearsal": {"what": "the full PR path on 2023-01..2026-08 recorded months before "
                              "registration (engineering check; no window value exists)",
                      "dir": rel_to_repo(REH)},
        "attempt_policy": d["attempt_policy"],
        "zones": "the page model's own", "priors": "the frozen page model; nothing is fitted",
        "burn_in": NA("no fit"), "folds": NA("prospective window"),
        "fresh_units": [uid],
        "thresholds": {"ratio_band": 1.5, "informative_head_m": 0.05, "min_clusters": 5,
                       "min_months": 6, "level": 0.90},
        "gate_scores": NA("no gate; the decision rule is reported whichever way it falls"),
        "primary_model": d["primary_model"],
        "inputs": sorted({v["path"] for v in d["inputs"].values()}
                         | {rel_to_repo(p) for p in sorted(R0_ZR.iterdir())}
                         | {rel_to_repo(Path(__file__)),
                            rel_to_repo(Path(__file__).with_name("build_r.py"))}),
        "inputs_detail": d["inputs"],
        "interpretation": d["what_a_pass_supports"],
        "written_by": rel_to_repo(Path(__file__)),
    })
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("missing inputs:", missing or "none")
    bad = [k for k, v in d["inputs_detail"].items()
           if (REPO / v["path"]).exists() and sha(REPO / v["path"]) != v["sha256"]]
    print("inputs changed since drafting:", bad or "none")
    if not a.write or errs or missing or bad:
        return 1 if (errs or missing or bad) else 0
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
