"""WP S3 fix round of the S3r repair (2026-10-18), diagnosis: composition part of the WRA
network median (the product common modes ``wra_onfan_fullband`` and ``wra_hf``) at the
January 2023 outage edges, from the gnss_clean/full/v3 product tables.

    python -I -B dataprep_staging/s3_gnss/v4/diag_product_cm.py OUT_JSON

The network median is recomputed from the product's daily ``resid`` of the member stations
(flags 0, 1, 3) and checked against ``common_mode.parquet``; its jump over +-10 days at each
edge is split into the jump of the median over the members reporting on both sides
(constant subset) and the remainder (composition part).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
V3 = REPO / "results/dataprep/gnss_clean/full/v3"
EDGES = ["2023-01-07", "2023-02-08", "2023-01-01", "2023-01-09"]
W = 10


def main(out: str) -> int:
    st = pd.read_csv(V3 / "stations.csv")
    members = st[(st.family == "wra") & (st.role == "member")].station.tolist()
    d = pd.read_parquet(V3 / "wra" / "daily.parquet", columns=["station", "date", "resid", "flag"])
    d = d[d.station.isin(members) & d.flag.isin([0, 1, 3])]
    R = d.pivot(index="date", columns="station", values="resid")
    R.index = pd.DatetimeIndex(R.index)
    R = R.reindex(pd.date_range(R.index.min(), R.index.max(), freq="D"))
    med = R.median(axis=1).where(R.notna().sum(axis=1) >= 5)
    cms = pd.read_parquet(V3 / "common_mode.parquet")
    cms.index = pd.DatetimeIndex(cms.pop("date"))
    chk = (med - cms["wra_onfan_fullband"].reindex(med.index)).abs()
    res = {"members": len(members),
           "recomputed_vs_product_fullband_max_abs_mm": float(1000 * chk.max()),
           "edges": []}
    for e in EDGES:
        t = pd.Timestamp(e)
        lo, hi = t - pd.Timedelta(days=W), t + pd.Timedelta(days=W - 1)
        b, a = R.loc[lo:t - pd.Timedelta(days=1)], R.loc[t:hi]
        both = [k for k in R if b[k].notna().sum() >= 3 and a[k].notna().sum() >= 3]
        j = float(med.loc[t:hi].median() - med.loc[lo:t - pd.Timedelta(days=1)].median())
        cs = R[both].median(axis=1)
        jc = float(cs.loc[t:hi].median() - cs.loc[lo:t - pd.Timedelta(days=1)].median())
        hf = cms["wra_hf"]
        jh = float(hf.loc[t:hi].median() - hf.loc[lo:t - pd.Timedelta(days=1)].median())
        res["edges"].append({"date": e, "n_before": int((b.notna().sum() >= 3).sum()),
                             "n_after": int((a.notna().sum() >= 3).sum()),
                             "n_constant_subset": len(both),
                             "fullband_jump_mm": 1000 * j, "constant_subset_jump_mm": 1000 * jc,
                             "composition_part_mm": 1000 * (j - jc),
                             "wra_hf_jump_mm": 1000 * jh})
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
