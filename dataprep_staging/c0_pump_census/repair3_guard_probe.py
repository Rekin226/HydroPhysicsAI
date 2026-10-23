"""WP C0 repair round, v3: guard probe and manifest check of the published v3 products (and of
v1/v2, to record what the guard still grants on the superseded versions).

For each (product, version, use) the probe calls ``guard.open_product`` and, for the L2
variants, ``Product.read_table('county_class_monthly.parquet')``, recording what is granted,
which exemption notes the guard attaches and which flags survive; then
``common.check.check_manifest`` on every pump_electricity manifest.

Output: reports/C0_evidence/repair3/guard_probe.json.  Usage: python -I repair3_guard_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import guard  # noqa: E402
from common.check import check_manifest  # noqa: E402

OUT = lib.EVIDENCE / "repair3" / "guard_probe.json"
PE = lib.PRODUCT_ROOT
CASES = [
    ("preorigin", "v3", "forcing_frozen"), ("preorigin", "v3", "model_design"),
    ("preorigin", "v3", "design_time_evidence"), ("preorigin", "v3", "prior"),
    ("preorigin", "v3", "scoring_target"), ("preorigin", "v3", "selection"),
    ("preorigin", "v3", "calibration_target"), ("preorigin", "v3", "forcing_recorded_arm"),
    ("full", "v3", "forcing_recorded_arm"), ("full", "v3", "statistical_layer_history"),
    ("full", "v3", "forcing_frozen"), ("full", "v3", "after_the_fact_analysis"),
    ("census_L1", "v3", "model_design"), ("census_L1", "v3", "prior"),
    ("census_L1", "v3", "scoring_target"),
    ("preorigin", "v2", "forcing_frozen"), ("preorigin", "v2", "design_time_evidence"),
    ("preorigin", "v1", "forcing_frozen"),
]


def main() -> int:
    out = {"cases": [], "manifests": []}
    for prod, ver, use in CASES:
        d = PE / prod / ver
        rec = {"product": prod, "version": ver, "use": use}
        try:
            p = guard.open_product(d, use)
            rec["result"] = "granted"
            rec["notes"] = list(p.notes)
            if prod != "census_L1":
                df = p.read_table("county_class_monthly.parquet")
                rec["rows"] = int(len(df))
                rec["flags_read"] = sorted(int(x) for x in df["flag"].unique())
        except Exception as exc:  # noqa: BLE001 - the refusal is the result
            rec["result"] = f"refused: {type(exc).__name__}: {str(exc)[:220]}"
        out["cases"].append(rec)
        print(prod, ver, use, rec["result"], rec.get("flags_read", ""))
    for m in sorted(PE.glob("*/v*/MANIFEST.json")):
        c = check_manifest(m)
        out["manifests"].append({k: c.get(k) for k in ("path", "status", "n_hashed",
                                                       "schema_errors", "hash_problems")})
        print(lib.rel(m), c["status"], c.get("n_hashed"))
    lib.save_json(out, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
