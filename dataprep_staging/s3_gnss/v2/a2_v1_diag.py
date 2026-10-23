"""WP S3 repair round: the v2 A2 statistics on the published v1 pre-origin tables (after-the-
fact diagnostic for the verifier's major issue 2; never a gate).

    python -I dataprep_staging/s3_gnss/v2/a2_v1_diag.py OUT_JSON

Runs ``v2/validate.test_2021_extra`` on ``gnss_clean/preorigin/v1`` (wra/daily, common_mode,
stations): per common mode the evaluable pairs, same sign, pooled ratio, pooled GNSS extra and
its ratio to the uncorrected one, the unique-station summary, and each common mode's own 2021
extra (for ``ngl_offfan`` with its interannual spread over 2008-2022). Logs a ``diagnose``
look under the v1 pre-registration before computing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import build_tables as BT  # noqa: E402
import validate as V  # noqa: E402
from common.paths import DATAPREP  # noqa: E402
from harness import prereg2  # noqa: E402

V1 = DATAPREP / "gnss_clean" / "preorigin" / "v1"
A2_UNITS = ["s3.pair.GFES-光復國小", "s3.pair.HLES-宏崙國小", "s3.pair.HNES-湖南國小",
            "s3.pair.JMES-拯民國小", "s3.pair.KTES-客厝國小", "s3.pair.SJES-溪州國小",
            "s3.pair.SSES-新生國小", "s3.pair.STES-秀潭國小", "s3.pair.TKJS-土庫國中",
            "s3.pair.YCES-元長國小", "s3.pair.GFES-NTUH_282_29", "s3.pair.SJES-SJES",
            "s3.pair.STES-STES", "s3.pair.TKJS-TKJS_281_29"]


def main(argv: list[str]) -> int:
    lk = prereg2.record_look(A2_UNITS, "gnss_wra+rings+piles:2020-01..2021-12:extra_2021",
                             "S3", "diagnose", prereg_id="S3.gnss_clean_acceptance.v1",
                             note="S3 repair round: v2 A2 statistics (each common mode, its "
                                  "own 2021 extra, unique stations) on gnss_clean/preorigin/v1 "
                                  "(after-the-fact diagnostic of the verifier's major issue 2)")
    wd = pd.read_parquet(V1 / "wra" / "daily.parquet")
    cms = pd.read_parquet(V1 / "common_mode.parquet")
    cms.index = pd.DatetimeIndex(pd.to_datetime(cms.pop("date")))
    st = pd.read_csv(V1 / "stations.csv")
    res = V.test_2021_extra(wd, cms, st)
    out = {"source": "results/dataprep/gnss_clean/preorigin/v1", "look": lk["time"],
           "A2_v2_statistics_on_v1": res}
    Path(argv[0]).write_text(json.dumps(BT.clean_json(out), indent=1, ensure_ascii=False,
                                        allow_nan=False, default=str))
    show = {k: v for k, v in res.items() if k != "pairs"}
    print(json.dumps(BT.clean_json(show), indent=1, ensure_ascii=False, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
