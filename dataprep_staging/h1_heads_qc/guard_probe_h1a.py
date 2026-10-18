"""H1a repair round (2026-10-13): loader-guard probe of heads/L1_clean v1, v2 and v3
(python -I). Records which declared uses each version grants and what a pre-origin consumer
of v3 receives.

Usage: guard_probe_h1a.py OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import guard  # noqa: E402
from common.paths import REPO  # noqa: E402

PROD = REPO / "results" / "dataprep" / "heads" / "L1_clean"
CASES = [("v3", "calibration_target", "2022-12"), ("v3", "calibration_target", None),
         ("v3", "datum_fit", "2022-12"), ("v3", "baseline", "2022-12"),
         ("v3", "initial_conditions", "2022-12"), ("v3", "scoring_target", None),
         ("v3", "prospective_scorer", None), ("v3", "after_the_fact_analysis", None),
         ("v2", "calibration_target", "2022-12"), ("v2", "after_the_fact_analysis", None),
         ("v2", "validator", None), ("v1", "calibration_target", "2022-12"),
         ("v1", "after_the_fact_analysis", None)]


def main():
    out = []
    for v, use, we in CASES:
        rec = {"version": v, "use": use, "window_end": we}
        try:
            P = guard.open_product(PROD / v, use, window_end=we)
            rec["granted"] = True
            if v == "v3" and use in ("calibration_target", "datum_fit", "scoring_target"):
                M = P.read_table("heads_monthly.parquet")
                rec.update({"monthly_rows": int(len(M)),
                            "max_date": str(M["date"].max())[:10],
                            "flags": sorted(int(x) for x in M["flag"].unique()),
                            "rows_with_step_correction": int((M["corr_m"].abs() > 0).sum())})
        except Exception as exc:  # noqa: BLE001 - every refusal is recorded
            rec.update({"granted": False, "error": type(exc).__name__,
                        "message": str(exc)[:300]})
        out.append(rec)
        print(json.dumps({k: rec[k] for k in rec if k != "message"}))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
