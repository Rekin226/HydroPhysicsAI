"""WP S3 fix round of the S3r repair (2026-10-18): daily check behind S3.repair.md 7.6 on the new
KTES common-mode-only step of 2024-04-25 in gnss_clean/full/v4: how many days around it lack a
neighbour, and how far the v3-rule (plain) median and the R7' common mode differ there.

    dataprep_staging/tools/cpu_slot.sh python -I -B \
        dataprep_staging/s3_gnss/v4/adhoc/trace_ktes_daily.py OUT_DIR [--ledger]

Run under the look ``fix_trace_repro`` (KTES, 2024-02..2024-07). No injection, no score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))   # dataprep_staging
sys.path.insert(0, str(HERE.parent))       # v4 modules
import pipeline as P  # noqa: E402
import smoke_v4 as S  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import REPO  # noqa: E402

WINDOWS = (("2024-04-15", "2024-05-04", "size_window_pm10d"),
           ("2024-03-26", "2024-05-24", "scan_window_pm30d"))


def main(argv: list[str]) -> int:
    od = Path(argv[0])
    od.mkdir(parents=True, exist_ok=True)
    p, _ = S.load_tuned("full")
    run = S.wra_unrolled(p, None)
    R = run["R2"].sort_index()
    R = R.reindex(pd.date_range(R.index.min(), R.index.max(), freq="D"))
    nb = P.knn_neighbours("KTES", [m for m in run["members"] if m in R], run["xy"])
    diff = (R[nb].median(axis=1) - run["cm"]["KTES"].reindex(R.index)).abs() * 1000
    out = {"neighbours": nb}
    for lo, hi, lab in WINDOWS:
        n = R.loc[lo:hi, nb].notna().sum(axis=1)
        d = diff.loc[lo:hi]
        out[lab] = {"window": [lo, hi], "days": int(len(n)),
                    "days_fewer_than_5": int((n < 5).sum()),
                    "max_abs_plain_minus_r7b_mm": round(float(d.max()), 2),
                    "mean_abs_plain_minus_r7b_mm": round(float(d.mean()), 3)}
    f = od / "trace_KTES_daily_2024-04.json"
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    src = f.resolve().relative_to(REPO).as_posix()
    rows = [{"number_id": f"S3.fix.trace.KTES.r7b.{lab}", "source": src, "role": "input",
             "product_version": "gnss_clean/v4 smoke traces (no product; S3r fix round)",
             "note": "adhoc/trace_ktes_daily.py"} for _, _, lab in WINDOWS]
    if "--ledger" in argv:
        ledger.append_rows("S3", rows)
        print(f"ledger: {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
