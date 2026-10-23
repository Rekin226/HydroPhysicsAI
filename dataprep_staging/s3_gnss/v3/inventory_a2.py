"""WP S3 repair round 3 (S3r): inventory of candidate A2 references by GEOMETRY and DATA
AVAILABILITY only (no height, compaction or extra value is read into any statistic).

    python -I dataprep_staging/s3_gnss/v3/inventory_a2.py OUT_DIR

Lists, for every GNSS station of the v2 pre-origin station sets (WRA cache stations, NGL fan
box + JSAN) and every ring site and deep-benchmark pile in the vendor cache, the distance to
the nearest ring site and to the nearest pile, and the number of observed days of the GNSS
series and of the reference in the four A2 months (2020-01, 2020-12, 2021-01, 2021-12).
Values are only counted, never summarised.

Look logged beforehand: ``looks_r5.py diag_a2_candidates_geometry``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # dataprep_staging
sys.path.insert(0, str(HERE.parent / "v2"))       # v2 readers
import io_gnss as io  # noqa: E402
import pipeline as P  # noqa: E402
import products as PR  # noqa: E402

MONTHS = ("2020-01", "2020-12", "2021-01", "2021-12")


def count_months(idx: pd.DatetimeIndex) -> dict:
    ym = pd.DatetimeIndex(idx).strftime("%Y-%m")
    return {m: int((ym == m).sum()) for m in MONTHS}


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    rings = pd.read_csv(io.REPO / "chou-shui-data" / "data" / "mlcw_stations.csv")
    rings = rings.rename(columns={"sub_id": "site"})
    meta = pd.read_parquet(io.LS_CACHE / "ls-wra-dbm-obs__meta.parquet",
                           columns=["_station_id", "TWD97坐標_X", "TWD97坐標_Y", "深度"])
    piles = []
    for p in sorted(io.LS_CACHE.glob("ls-wra-dbm-obs__*.parquet")):
        k = p.name.split("__", 1)[1][:-len(".parquet")]
        if k == "meta":
            continue
        d = pd.read_parquet(p, columns=["value"])
        s = pd.to_numeric(d["value"], errors="coerce").dropna()
        piles.append({"pile": k, "idx": pd.DatetimeIndex(pd.to_datetime(s.index))})
    print(meta.to_string())
    pmeta = {}
    for _, r in meta.dropna(subset=["TWD97坐標_X"]).iterrows():
        pmeta[str(r["_station_id"])] = (float(r["TWD97坐標_X"]), float(r["TWD97坐標_Y"]))
    rows = []
    have = set(io.wra_files())
    gnss = {("wra", k): xy for k, xy in PR.wra_xy().items() if k in have}
    ngl_st = P.ngl_policy(P.ORIGIN_END)[0]
    gnss.update({("ngl", k): xy for k, xy in PR.ngl_xy(ngl_st).items()})
    for (fam, k), (x, y) in gnss.items():
        dr = np.hypot(rings.x - x, rings.y - y)
        jr = int(np.argmin(dr.to_numpy()))
        rec = {"family": fam, "station": k, "nearest_ring": rings.site.iloc[jr],
               "ring_dist_m": round(float(dr.iloc[jr]))}
        best = (np.inf, None)
        for pid, (px, py) in pmeta.items():
            dd = float(np.hypot(px - x, py - y))
            if dd < best[0]:
                best = (dd, pid)
        rec["nearest_pile_meta_id"], rec["pile_dist_m"] = best[1], round(best[0])
        s = io.read_wra(k) if fam == "wra" else io.read_tenv3(io.ngl_tenv3_path(k))["up"]
        rec.update({f"gnss_days_{m}": v for m, v in count_months(s.index).items()})
        rows.append(rec)
    T = pd.DataFrame(rows).sort_values(["family", "ring_dist_m"])
    T.to_csv(od / "a2_candidate_inventory.csv", index=False)
    print(T.to_string())
    P_ = pd.DataFrame([{"pile": p["pile"], **{f"days_{m}": v for m, v in
                                              count_months(p["idx"]).items()}} for p in piles])
    P_.to_csv(od / "a2_pile_availability.csv", index=False)
    print(P_.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
