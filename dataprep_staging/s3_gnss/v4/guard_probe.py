"""WP S3: loader-guard probes of the two gnss_clean v4 variants (after ``v4/build.py``).

    python -I -B dataprep_staging/s3_gnss/v4/guard_probe.py OUT_JSON

Records, for 13 (variant, use) cases, whether the loader guard opens the product and with which
flags, or refuses it and why. With the v4 gate closed (failed), only the gate-exempt
uses (design_time_evidence on the pre-origin variant,
after_the_fact_analysis) should open.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import guard  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

PRE = DATAPREP / "gnss_clean" / "preorigin" / "v4"
FULL = DATAPREP / "gnss_clean" / "full" / "v4"
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
