"""I1.CH_backcast.v1 step 1: the registered, metadata-only availability look on ls_cache
2010-11 (round 7 rank 18; user ruling R7.l1_pre2012).

Two steps, in order, both run by the lead:

``look --validate`` / ``look --write``
    prints / logs (``prereg3.record_look``, purpose ``availability_check``) the look on unit
    I1CHB:ls_cache_rowcounts_2010_2011 whose members are the Changhua monitoring wells that
    have an ``ls_cache/gw__<sid>__raw.parquet`` file (file names and the station table only).
``count --out DIR``
    refuses unless that look is logged; then, per candidate, reads ONLY the ``datetime``
    column (``pyarrow.parquet.read_table(columns=['datetime'])``; the ``value`` column is
    never read), counts the 2010-01..2011-12 months whose sampling could make a monthly value
    (>= 20 days with >= 12 distinct hours; ``bc_lib.available_months``) and applies the
    registered stop rule: at least 10 Changhua wells with at least 18 such months, excluding
    the wells within 1000 m of a ring site (mechanism.py's nest reads). Writes avail.json
    once; prints only counts and the pass/fail word.

    python dataprep_staging/ch_x/avail_l1pre.py look --validate
    python dataprep_staging/ch_x/avail_l1pre.py count --out results/twin_runs/chx/backcast/avail
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bc_lib as BL  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402

PURPOSE = "availability_check"
STATION_COLS = ["sid", "CountyName", "LocationByTWD97", "GroundwaterLayerCode",
                "GroundwaterZoneIdentifier", "WellElevation", "WellDepth"]


def stations() -> pd.DataFrame:
    st = pd.read_parquet(IO.rp(K.STATIONS), columns=STATION_COLS)
    st = st[st["GroundwaterZoneIdentifier"] == 50].copy()
    st["sid"] = st["sid"].astype(str)
    st["county"] = st["CountyName"].map(BL.COUNTY_ZH).fillna("other")
    xy = st["LocationByTWD97"].map(BL.parse_xy)
    st["x"] = [p[0] if p else np.nan for p in xy]
    st["y"] = [p[1] if p else np.nan for p in xy]
    st["layer_code"] = st["GroundwaterLayerCode"].astype(str)
    return st.drop_duplicates("sid").sort_values("sid").reset_index(drop=True)


def raw_path(sid: str) -> Path:
    return IO.rp(K.LS_CACHE) / K.LS_RAW.format(sid=sid)


def candidates(st: pd.DataFrame | None = None) -> list[str]:
    st = stations() if st is None else st
    ch = st[st["county"] == "changhua"]
    return sorted(s for s in ch["sid"] if raw_path(s).exists())


def ring_xy() -> np.ndarray:
    t = pd.read_csv(IO.rp(K.MLCW_STATIONS), usecols=["x", "y"])
    return t[["x", "y"]].to_numpy(float)


def look_record(ids: list[str]) -> dict:
    return {"units": [K.BC_AVAIL_UNIT], "data_scope": K.BC_SCOPE, "wp": K.WP,
            "purpose": PURPOSE,
            "members": {K.BC_AVAIL_UNIT: {
                "description": "Changhua monitoring wells with an ls_cache gw__<sid>__raw "
                               "file: per-month sampling counts 2010-01..2011-12 from the "
                               "datetime column only (no value), I1.CH_backcast.v1's "
                               "registered feasibility look (user ruling R7.l1_pre2012)",
                "ids": list(ids), "window": "2010-01..2011-12"}},
            "note": "metadata-only availability look; stop rule >= 10 wells x >= 18 months, "
                    "ring-nest wells excluded"}


def look_logged(prereg_dir=None) -> bool:
    from common.paths import PREREG_DIR
    from harness import prereg3

    d = Path(prereg_dir or PREREG_DIR)
    for lk in prereg3.read_chained(d / "looks.jsonl"):
        if K.BC_AVAIL_UNIT in (lk.get("units") or []) and lk.get("purpose") == PURPOSE:
            return True
    return False


def timestamps(sid: str) -> np.ndarray:
    import pyarrow.parquet as pq

    if "datetime" not in pq.read_schema(raw_path(sid)).names:
        return np.array([], dtype="datetime64[ns]")   # no timestamps: the well is unavailable
    tab = pq.read_table(raw_path(sid), columns=["datetime"])
    return tab.column("datetime").to_numpy()


def count(out: Path, prereg_dir=None) -> dict:
    if not look_logged(prereg_dir):
        raise SystemExit("REFUSED: log the availability look first (look --write)")
    target = out / "avail.json"
    if target.exists():
        raise SystemExit(f"{target} exists; the availability look runs once")
    st = stations()
    ids = candidates(st)
    months = {s: BL.available_months(timestamps(s)) for s in ids}
    xy = {r.sid: (r.x, r.y) if np.isfinite(r.x) else None for r in st.itertuples()}
    nest = BL.nest_wells({s: xy.get(s) for s in ids}, ring_xy())
    county = dict(zip(st["sid"], st["county"], strict=True))
    feas = BL.feasibility(months, county, nest)
    rep = {"what": "I1.CH_backcast.v1 availability (timestamps only)",
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rule": {"day_min_hours": K.BC_DAY_MIN_HOURS,
                    "month_min_days": K.BC_MONTH_MIN_DAYS, "window": list(K.BC_WINDOW)},
           "months_by_well": months, "n_months_by_well": {s: len(m) for s, m in months.items()},
           "nest_excluded": sorted(nest), "feasibility": feas,
           "files_sha256": {s: IO.sha256(raw_path(s)) for s in ids},
           "stations": K.STATIONS, "stations_sha256": IO.sha256(IO.rp(K.STATIONS))}
    out.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(rep, f, indent=1)
    print(f"availability: {feas['n_eligible']} eligible Changhua wells (>= {K.BC_MIN_MONTHS} "
          f"months; {feas['n_excluded_nest']} nest wells excluded) -> "
          f"{'PASS' if feas['passed'] else 'STOP (not evaluable)'}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("look")
    g = lk.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    c = sub.add_parser("count")
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "count":
        count(a.out)
        return 0
    rec = look_record(candidates())
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    if a.validate:
        from harness import prereg3
        errs = prereg3.member_errors(rec["members"][K.BC_AVAIL_UNIT])
        print("member errors:", errs or "none", "| logged already:", look_logged())
        return 1 if errs else 0
    if look_logged():
        raise SystemExit("REFUSED: the availability look is already logged")
    from harness import prereg3
    r = prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                            members=rec["members"], note=rec["note"])
    print(f"logged look at {r['time']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
