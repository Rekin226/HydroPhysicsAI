"""WP S5 repair round: the declaration ``S5.subsidence_fused_rules.v3`` (label amendment of
subsidence_fused, written BEFORE v3 is built), harness.prereg3.

    python -I s5r_prereg_v3.py [--dry-run]

subsidence_fused/v2 was built under S5.subsidence_fused_rules.v2 after M1 passed. Its card and
validation.json carried M1's registered sd label ('calibrated at monthly resolution on
pre-origin WRA station-months') for the monthly value, although that value's sd includes the
annual posterior variance, whose coverage checks (T1, T2) failed, and coverage3 marks M1's
stratum as outside its validated envelope. v3 changes labels and text only: every value,
flag, mask and array is v2's. POST-RESULT (decided after M1's score and v2's build).
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
import s5r_const as R  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import from_repo, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

ID = "S5.subsidence_fused_rules.v3"
V2_DECL_SHA = "a97921139bdf79a38136dd9a91a353b97f375c20facc54c1d5f800ec963654c1"

AMENDMENTS = {
    "9_monthly_sd_label": (
        "The monthly VALUE sd (annual posterior variance at mid-month + stage-2 anomaly "
        "variance) is labelled 'uncalibrated': its annual part failed T1/T2 coverage. M1's "
        "coverage result (S5.monthly_field.v1) is reported as a check of the anomaly relative "
        "to the annual model only, with coverage3's envelope flag; M1's registered label is "
        "kept alongside (validation.json sd_label.monthly_registered_M1_label). More "
        "conservative than the registered labelling rule of M1 (which named 'the monthly "
        "sd'); disclosed as a deviation."),
    "10_card_text": (
        "PRODUCT_CARD.md item 4 states M1's bootstrap P(ratio >= 1), its ratio by year (the "
        "skill is from 2021) and that stage 2 estimated rho at 0 (the anomaly is not separated "
        "from atmospheric delay by its time behaviour), and asks users to read monthly detail "
        "outside 2021-like conditions as unconfirmed. No delivery rule changes: monthly values "
        "stay delivered wherever support_vr holds, as fixed before M1 was scored."),
    "11_versions": (
        "v3 supersedes v1 and v2; v2 stays published and unchanged (its values equal v3's). "
        "Manifest: the v3 declaration and v2's manifest are added as inputs, s5r_prereg_v3.py "
        "as code."),
}


def doc() -> dict:
    na = prereg3.not_applicable
    code = [rel_to_repo(HERE / n) for n in (
        "s5r_product.py", "s5r_prereg_v2.py", "s5r_prereg_v3.py", "s5r_core.py", "s5r_vconst.py",
        "s5r_stage2.py", "s5r_const.py", "s5product.py")]
    data = [rel_to_repo(p) for p in (
        R.EVIDENCE_R / "m1.json", R.EVIDENCE_R / "hyper_stage2.json",
        C.DATAPREP / "preregistration" / f"{R.DECL_ID}.json",
        C.DATAPREP / "preregistration" / f"{R.MTEST_ID}.json",
        C.DATAPREP / C.PRODUCT / "v2" / "MANIFEST.json")]
    return {
        "id": ID, "wp": C.WP,
        "title": ("S5 repair round: label amendment of subsidence_fused (v3 = v2 values with "
                  "the monthly sd labelled uncalibrated and M1's caveats on the card); declared "
                  "before v3 is built"),
        "kind": "declaration",
        "tests": na("no test; labels only"),
        "zones": na("no zones"),
        "priors": na("unchanged"),
        "burn_in": na("no spin-up"),
        "folds": na("no folds"),
        "fresh_units": na("none"),
        "thresholds": {"unchanged": "all of " + R.DECL_ID},
        "gate_scores": na("unchanged (failed)"),
        "primary_model": {"base": R.DECL_ID, "base_sha256": V2_DECL_SHA,
                          "amendments": AMENDMENTS,
                          "unchanged": "every rule, value, flag and mask of " + R.DECL_ID},
        "inputs": code + data,
        "interpretation": (
            "A wording and labelling correction found while reviewing v2 before the report: "
            "the registered M1 label described 'the monthly sd' as calibrated, but M1 checked "
            "only the anomaly relative to the annual model, and the annual part's coverage "
            "failed (T1 0.631, T2 0.940). The verifier of round 1 flagged the same kind of "
            "over-statement on v1's card."),
        "design_time_exposure": ["M1's score, subsidence_fused/v2 and its files were seen"],
        "code_at_registration": {p: sha256_file(from_repo(p)) for p in code},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    if a.dry_run:
        print(json.dumps({"errors": prereg3.validate_prereg(d), "inputs": len(d["inputs"])}))
        return 0
    path, sha = prereg3.write_prereg(d)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
