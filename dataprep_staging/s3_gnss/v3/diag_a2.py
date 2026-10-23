"""WP S3 repair round 3 (S3r): why A2 failed in v1 and v2 (diagnosis only; failed units).

    python -I dataprep_staging/s3_gnss/v3/diag_a2.py OUT_DIR

Reads the published pre-origin tables ``gnss_clean/preorigin/v1`` and ``v2`` (daily values,
steps, common modes) and the rings/piles exactly as the registered A2 does (v2
``validate.rings_monthly``, ``piles_monthly``, ``pairs``). For every pair (GNSS station within
1 km of a ring site or a deep pile) and each GNSS series:

* ``raw_all``: vendor height ``value_raw``, every day (no step correction, no flag filter);
* ``raw_flag01``: ``value_raw`` on flag-0/1 days;
* ``clean_flag01`` (the registered "uncorrected"): ``value`` on flag-0/1 days;
* ``clean_flag013``: ``value`` on flag 0/1/3 days (what a flag-3 step does to the pair set);
* ``ngl_offfan`` / ``wra_hf`` / ``wra_onfan_fullband`` removed from ``clean_flag01``;

it gives D2020, D2021 and the extra (monthly medians, >= 10 days, as registered), the steps
dated 2019-12..2022-01 at that station with their basis and flag, and the flag-0/1 day counts of
the four January/December months. It also prints each common mode's own D2020, D2021 and
extra. No threshold and no pass/fail is computed here.

Look logged beforehand: ``looks_r5.py diag_a2_pairs``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # dataprep_staging
sys.path.insert(0, str(HERE.parent / "v2"))       # registered v2 A2 helpers
import validate as V  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

MONTHS = ("2020-01", "2020-12", "2021-01", "2021-12")


def load(version: str) -> dict:
    d = DATAPREP / "gnss_clean" / "preorigin" / version
    cms = pd.read_parquet(d / "common_mode.parquet")
    cms.index = pd.DatetimeIndex(pd.to_datetime(cms.pop("date")))
    return {"daily": pd.read_parquet(d / "wra" / "daily.parquet"),
            "steps": pd.read_csv(d / "steps.csv"), "cms": cms,
            "stations": pd.read_csv(d / "stations.csv")}


def series_variants(daily: pd.DataFrame, cms: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"raw_all": -V.monthly(daily, "value_raw", flags=tuple(range(10))),
           "raw_flag01": -V.monthly(daily, "value_raw"),
           "clean_flag01": -V.monthly(daily, "value"),
           "clean_flag013": -V.monthly(daily, "value", flags=(0, 1, 3))}
    for col in ("ngl_offfan", "wra_hf", "wra_onfan_fullband"):
        d = V._dt(daily).merge(cms[[col]].rename(columns={col: "cmx"}), left_on="date",
                               right_index=True, how="left")
        d["v"] = d["value"] - d["cmx"]
        out[f"cmrm_{col}"] = -V.monthly(d, "v")
    return out


def extra(s: pd.Series) -> tuple[float, float, float]:
    a, b = V.ann_increment(s, 2020), V.ann_increment(s, 2021)
    return 1000 * a, 1000 * b, 1000 * (b - a)


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    rings, rxy = V.rings_monthly()
    piles, pxy = V.piles_monthly()
    res: dict = {}
    for ver in ("v1", "v2"):
        tab = load(ver)
        gs = series_variants(tab["daily"], tab["cms"])
        rows = []
        prs = [("ring", st, ref, dist, rings[ref]) for st, ref, dist in
               V.pairs(tab["stations"], rxy, "site") if ref in rings]
        prs += [("pile", st, ref, dist, piles[ref]) for st, ref, dist in
                V.pairs(tab["stations"], pxy, "pile")]
        dd = V._dt(tab["daily"])
        for kind, st, ref, dist, rs in prs:
            r20, r21, rex = extra(rs)
            rec = {"kind": kind, "gnss": st, "reference": ref, "dist_m": round(dist),
                   "ref_D2020": round(r20, 1), "ref_D2021": round(r21, 1),
                   "ref_extra": round(rex, 1)}
            for name, g in gs.items():
                if st in g:
                    a, b, e = extra(g[st])
                    rec[f"{name}_extra"] = round(e, 1)
                    rec[f"{name}_D2020"] = round(a, 1)
                    rec[f"{name}_D2021"] = round(b, 1)
            x = dd[dd.station == st]
            for m in MONTHS:
                xm = x[x.date.dt.strftime("%Y-%m") == m]
                rec[f"n01_{m}"] = int(xm.flag.isin([0, 1]).sum())
                rec[f"n3_{m}"] = int((xm.flag == 3).sum())
            s = tab["steps"]
            s = s[(s.family == "wra") & (s.station == st) & (s.date >= "2019-12-01")
                  & (s.date <= "2022-01-31")]
            rec["steps_2020_21"] = [f"{r.date} {1000 * r.size_m:+.1f}mm se{1000 * r.se_m:.1f} "
                                    f"{r.basis} flag{r.flag_after}" for r in s.itertuples()]
            rows.append(rec)
        T = pd.DataFrame(rows)
        T.to_csv(od / f"a2_pairs_{ver}.csv", index=False)
        cm = {}
        for col in ("ngl_offfan", "wra_hf", "wra_onfan_fullband"):
            c = tab["cms"][col]
            m = c.groupby(c.index.to_period("M")).mean()
            m.index = m.index.to_timestamp()
            a, b, e = extra(m)
            cm[col] = {"D2020": round(a, 2), "D2021": round(b, 2), "extra": round(e, 2)}
        res[ver] = {"pairs": T.to_dict(orient="records"), "cm_own": cm}
        cols = ["kind", "gnss", "reference", "ref_extra", "raw_all_extra", "raw_flag01_extra",
                "clean_flag01_extra", "clean_flag013_extra", "cmrm_ngl_offfan_extra",
                "cmrm_wra_hf_extra", "cmrm_wra_onfan_fullband_extra"]
        print("==", ver, json.dumps(cm))
        print(T[[c for c in cols if c in T]].to_string())
        print(T[["gnss", "reference", "ref_D2020", "ref_D2021", "raw_all_D2020", "raw_all_D2021",
                 "clean_flag01_D2020", "clean_flag01_D2021"]].to_string())
        for r in rows:
            if r["steps_2020_21"]:
                print(r["gnss"], r["reference"], r["steps_2020_21"])
        print(T[["gnss", "reference"] + [c for c in T if c.startswith("n01_")
                                         or c.startswith("n3_")]].to_string())
    (od / "a2_diagnosis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
