"""WP S3: loader-guard probes of the two gnss_clean variants (after ``build.py``).

    python -I dataprep_staging/s3_gnss/guard_probe.py OUT_JSON

Checks that the pre-origin variant serves the pre-origin uses (flags 0-1 for datum fits and
scoring, 0-2 for calibration) and that the full variant refuses them but serves validators.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import guard  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

PRE = DATAPREP / "gnss_clean" / "preorigin" / "v1"
FULL = DATAPREP / "gnss_clean" / "full" / "v1"
CASES = [(PRE, "datum_fit"), (PRE, "calibration_target"), (PRE, "scoring_target"),
         (PRE, "design_time_evidence"), (PRE, "after_the_fact_analysis"),
         (PRE, "validator"), (PRE, "forcing_frozen"), (FULL, "validator"),
         (FULL, "scoring_target"), (FULL, "datum_fit"), (FULL, "calibration_target"),
         (FULL, "design_time_evidence"), (FULL, "after_the_fact_analysis")]


def main(argv: list[str]) -> int:
    out = []
    for d, use in CASES:
        rec = {"product": d.relative_to(DATAPREP).as_posix(), "use": use}
        try:
            p = guard.open_product(d, use)
            t = p.read_table("wra/daily.parquet")
            rec.update(ok=True, rows=int(len(t)), flags=sorted(int(x) for x in t.flag.unique()),
                       last_date=str(t.date.max()))
        except Exception as e:  # noqa: BLE001 - a refusal is the expected outcome
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    Path(argv[0]).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
