"""Repair round 1: ledger rows (plan 4.1, U6) for the repair-round numbers, and a checker note
appended to the product root's ``SUPERSEDED.json`` (the previous record is embedded by
sha256).

Numbers: ``H3.step_screen`` (registered, ``H3.repair.v1``) and ``H3.product_v3.counts``.
The model wells are the ``input`` of the step screen's null (their C20 values are not a test
target); the registered result is the ``validator``.

Usage: python -I r1_ledger.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from common import ledger  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

L1 = "results/dataprep/heads/L1_clean/v3"
H2V4 = "results/dataprep/heads/L2_filled_preorigin/v4"
UNITS = "results/dataprep/validation_units/v3"
EV = "results/dataprep/reports/H3_evidence"
PV = "heads/recovered_wells/v3"


def rows():
    out = []

    def add(num, src, role, note, version=PV):
        out.append({"number_id": num, "source": src, "role": role, "product_version": version,
                    "note": note})

    n = "H3.step_screen"
    add(n, L1, "input", "C20 monthly heads 2012-2022 (window_end 2022-12) of the 16 screened "
                        "wells and the model wells (neighbours and null); station_meta "
                        "coordinates and site ids")
    add(n, UNITS, "input", "the 158 model wells: neighbour pool and null population")
    add(n, f"{EV}/registered/H3.snapshot_months_2019_2022.csv", "input",
        "09030212's screened once-daily months (post-split values; reported only)")
    add(n, "results/dataprep/heads/recovered_wells/v1/recovered_wells.csv", "input",
        "the 15 wells admitted in v1 (screened)")
    add(n, f"{EV}/repair1/registered/H3.step_screen.json", "validator",
        "registered result (H3.repair.v1)")
    add(n, "results/dataprep/preregistration/H3.repair.v1.json", "input",
        "registered procedure and threshold")
    m = "H3.product_v3.counts"
    add(m, H2V4, "input", "calibration targets (flags 0-2) of the admitted wells")
    add(m, L1, "input", "envelope months; segment check")
    add(m, f"{EV}/registered/{C.TEST_D}.json", "validator",
        "per-well admission of bridged months (first round, unchanged)")
    add(m, f"{EV}/repair1/registered/H3.step_screen.json", "validator",
        "admission screen of every admitted well (none flagged)")
    add(m, "results/dataprep/preregistration/H3.repair.v1.json", "input",
        "admission rule of the repair round (gate_rule)")
    add(m, f"results/dataprep/preregistration/{C.PREREG_ID}.json", "input",
        "first-round admission rule (envelope, bridge rule, fallback)")
    return out


def supersede_note():
    p = C.REPO / "results/dataprep" / C.PRODUCT / "SUPERSEDED.json"
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("checker_note"):
        return "note already present"
    arch = f"{EV}/repair1/code_as_packaged_v2/r1_package.py"
    sha = hashlib.sha256((C.REPO / arch).read_bytes()).hexdigest()
    doc["checker_note"] = {
        "written_by": "dataprep_staging/h3_recovered_wells/r1_ledger.py",
        "previous_record_sha256": hashlib.sha256(raw).hexdigest(),
        "v2_expected_code_mismatch": (
            "v2's MANIFEST hashes dataprep_staging/h3_recovered_wells/r1_package.py as it was "
            "when v2 was packaged; the packager was then corrected to build v3, so the "
            "checker reports one code-hash mismatch on v2 (inputs and outputs re-hash). The "
            f"exact bytes v2 was built with are archived at {arch} (sha256 {sha}, equal to "
            "v2's manifest entry)."),
        "v1": "re-hashes clean (no first-round file was edited)"}
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return "note written"


def main():
    r = rows()
    have = {(x["number_id"], x["source"]) for x in ledger.read_all() if x.get("wp") == C.WP}
    new = [x for x in r if (x["number_id"], x["source"]) not in have]
    if new:
        ledger.append_rows(C.WP, new)
    allr = ledger.read_all()
    print("added", len(new), "H3 rows", sum(1 for x in allr if x.get("wp") == C.WP),
          "conflicts", len(ledger.conflicts(allr)), supersede_note())


if __name__ == "__main__":
    main()
