"""WP H4: probe the loader guard on subgrid_operator/v1 and re-hash its manifest.

Writes reports/H4_evidence/guard_probe.json. Usage: python guard_probe_h4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import check, guard  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

PROD = DATAPREP / "subgrid_operator/v1"
USES = ["design_time_evidence", "after_the_fact_analysis", "model_design", "validator",
        "scoring_target", "calibration_target", "forcing_frozen", "selection"]


def main() -> None:
    out = {}
    for u in USES:
        try:
            p = guard.open_product(PROD, u)
            df = p.read_table("correction_series.parquet")
            out[u] = {"granted": True, "rows": int(len(df)), "notes": p.notes}
        except Exception as e:  # noqa: BLE001 - the probe records every refusal
            out[u] = {"granted": False, "error": f"{type(e).__name__}: {e}"[:300]}
    rc = check.main([str(PROD), "--no-sources", "--json",
                     str(DATAPREP / "reports/H4_evidence/check_v1.json")])
    out["checker_exit"] = rc
    (DATAPREP / "reports/H4_evidence/guard_probe.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if k == "checker_exit" else {kk: vv for kk, vv in v.items()
                                                          if kk != "notes"})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
