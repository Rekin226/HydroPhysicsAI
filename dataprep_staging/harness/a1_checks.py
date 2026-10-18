"""WP A1 evidence: guard and pre-registration probes on the validation-units product.

    python -m harness.a1_checks OUT.json         (from dataprep_staging/, in a CPU slot)

Checks, without writing anything outside ``OUT.json``:

* the loader guard opens ``validation_units/v1`` for its allowed uses and refuses the
  forbidden ones (model design, priors, scoring targets, ...);
* every input, code file, input tree and output hash of the product reproduces
  (``common.check``);
* the fresh-units declaration and the confirmatory coverage pre-registration load with their
  recorded sha256 (INDEX.jsonl and the ledger), and the hash chains of INDEX.jsonl and
  looks.jsonl are intact;
* the tool refuses a test that names a design-time unit as fresh, and every design-time unit
  carries the seeded look while no fresh unit carries any look (no downstream test has
  scored on the fresh units yet).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, manifest  # noqa: E402
from common.check import check_manifest  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, rel_to_repo  # noqa: E402

from harness import prereg  # noqa: E402

UNITS = DATAPREP / "validation_units" / "v1"
USES = ("validator", "selection", "after_the_fact_analysis", "model_design", "prior",
        "prior_width_input", "calibration_target", "datum_fit", "baseline", "scoring_target",
        "design_time_evidence", "initial_conditions")
EXPECT_OPEN = {"validator", "selection", "after_the_fact_analysis"}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out: dict = {"product": rel_to_repo(UNITS), "guard": {}}
    for use in USES:
        try:
            p = guard.open_product(UNITS, use)
            out["guard"][use] = {"opened": True, "notes": p.notes}
        except Exception as e:  # noqa: BLE001 - every refusal is recorded
            out["guard"][use] = {"opened": False, "error": f"{type(e).__name__}: {e}"[:200]}
    out["guard_as_expected"] = all(out["guard"][u]["opened"] == (u in EXPECT_OPEN)
                                   for u in USES)
    chk = check_manifest(UNITS / manifest.MANIFEST_NAME)
    out["checker"] = {k: chk.get(k) for k in ("status", "n_hashed", "schema_errors",
                                              "problems", "hash_problems")}
    pre = {}
    for pid in ("A1-fresh-units-v1", "A1.coverage_confirm.v1"):
        doc, sha = prereg.load_prereg(pid)
        pre[pid] = {"sha256": sha, "created": doc["created"], "kind": doc["kind"]}
    out["preregistrations"] = pre
    idx = prereg.read_chained(PREREG_DIR / "INDEX.jsonl")
    looks = prereg.read_chained(PREREG_DIR / "looks.jsonl")
    out["chains"] = {"INDEX.jsonl_records": len(idx), "looks.jsonl_records": len(looks),
                     "intact": True}
    reg = json.loads((UNITS / "units.json").read_text(encoding="utf-8"))
    roles = {u["id"]: u["role"] for u in reg["units"]}
    fresh = sorted(k for k, v in roles.items() if v == "fresh")
    dt = sorted(k for k, v in roles.items() if v == "design_time")
    looked = {}
    for lk in looks:
        for u in lk["units"]:
            looked.setdefault(u, []).append({"time": lk["time"], "purpose": lk["purpose"],
                                             "wp": lk["wp"], "test_id": lk.get("test_id")})
    out["fresh_units"] = {"n": len(fresh), "with_any_look": [u for u in fresh if u in looked]}
    out["design_time_units"] = {"n": len(dt),
                                "without_seeded_look": [u for u in dt if not any(
                                    x["purpose"] == "design_time_seed"
                                    for x in looked.get(u, []))]}
    na = prereg.not_applicable
    probe = {"id": "A1.probe.never_written", "wp": "A1", "title": "probe", "kind": "test",
             "tests": [{"test_id": "A1.probe.t", "data_scope": "heads:2012-01..2022-12",
                        "fresh_units": ["dt.tercile.west"]}],
             "zones": na("probe"), "priors": na("probe"), "burn_in": na("probe"),
             "folds": na("probe"), "fresh_units": ["dt.tercile.west"], "thresholds": {"x": 1},
             "gate_scores": [{"name": "x", "weight": 1.0}], "primary_model": "probe",
             "inputs": na("probe"), "interpretation": "probe (never written)"}
    out["probe_design_time_as_fresh"] = {
        "validation_errors": prereg.validate_prereg(probe, UNITS / "units.json"),
        "blocking_looks": len(prereg.blocking_looks(probe["tests"], None))}
    ok = (out["guard_as_expected"] and chk.get("status") == "ok"
          and not out["fresh_units"]["with_any_look"]
          and not out["design_time_units"]["without_seeded_look"]
          and bool(out["probe_design_time_as_fresh"]["validation_errors"])
          and out["probe_design_time_as_fresh"]["blocking_looks"] > 0)
    out["pass"] = bool(ok)
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(argv[0], "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"guard_as_expected": out["guard_as_expected"],
                      "checker": out["checker"]["status"], "fresh_with_looks":
                      out["fresh_units"]["with_any_look"], "pass": out["pass"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
