"""Availability of fresh synthetic windows for the S1a r3 re-test (availability only).

Reads the 14 ring files and turns them IMMEDIATELY into a boolean read/not-read mask
(``notna``); no depth value, fit or statistic is computed. For candidate 13-month windows in
the pre-2012 record (before the 2012-03..2014-04 hiatus) it counts, per window, the sites that
would enter under entry rules (pre surveys >= P in the 36 months before, post surveys >= Q in
the 28 months after, >= 3 rings read in >= 80 % of the surveys of both windows), for several
(P, Q). The look is logged before the read.

    python -m s1_rings.r3_availability        (from dataprep_staging/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from harness import gaps, prereg3  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

OUT = K.EVIDENCE / "repair_r3" / "availability"
START_PRE2012 = pd.Timestamp("2003-01-01")
END_PRE2012 = pd.Timestamp("2012-02-29")       # last survey month before the 2012-14 hiatus
UNIT = "S1a:availability.pre2012_window_entry_2026-10-18"


def masks() -> dict:
    out = {}
    for s in K.SITES:
        d = pd.read_parquet(K.COMPACTION_DIR / f"{s}.parquet",
                            columns=["datetime", "depth", "value"])
        d["read"] = d["value"].notna()
        d = d.drop(columns=["value"])                      # values dropped at once
        m = d.pivot_table(index="datetime", columns="depth", values="read", aggfunc="max")
        m.index = pd.DatetimeIndex(pd.to_datetime(m.index)).astype("datetime64[ns]")
        out[s] = m.fillna(False).astype(bool).sort_index()
    return out


def geometry(w: dict, start=START_PRE2012, end=END_PRE2012) -> dict:
    s = pd.Period(w["start"], "M").start_time
    e = pd.Period(w["end"], "M").end_time.normalize()
    pre0 = max(s - pd.DateOffset(months=K.SYN_PRE_MONTHS), start)
    post1 = min((pd.Period(w["end"], "M") + K.SYN_POST_MONTHS).end_time.normalize(), end)
    return {"mask": (s, e), "pre": (pre0, s - pd.Timedelta(days=1)),
            "post": (e + pd.Timedelta(days=1), post1)}


def enters(m: pd.DataFrame, g: dict, P: int, Q: int) -> bool:
    m = m.loc[:END_PRE2012]
    pre = m.loc[g["pre"][0]:g["pre"][1]]
    post = m.loc[g["post"][0]:g["post"][1]]
    pre = pre[pre.any(axis=1)]
    post = post[post.any(axis=1)]
    if len(pre) < P or len(post) < Q:
        return False
    rings = [c for c in m.columns if pre[c].mean() >= K.COVER and post[c].mean() >= K.COVER]
    return len(rings) >= 3


def main() -> None:
    prereg3.record_look(
        [UNIT], "rings_availability:2003-01..2012-02", "S1a", "availability",
        members={UNIT: {"none": "survey dates and ring read/not-read masks of the 14 ring "
                                "files before 2012-03 (no depth value)",
                        "description": "entry counts of candidate pre-2012 synthetic windows"}},
        note=("S1a repair round r3 (S1ar), before declaring the fresh re-test: which 13-month "
              "windows of the pre-2012 record could be scored, from survey dates and ring "
              "read masks only (values are dropped on read; no fit or statistic)."))
    M = masks()
    wins = gaps.ring_windows("2003-01", "2011-12", K.SYN_LEN)
    rows = []
    for w in wins:
        g = geometry(w)
        if g["post"][0] > END_PRE2012:
            continue
        row = {"start": w["start"], "end": w["end"]}
        for P, Q in ((20, 18), (18, 16), (15, 15), (12, 12)):
            ss = [s for s in K.SITES if enters(M[s], g, P, Q)]
            row[f"n_{P}_{Q}"] = len(ss)
            row[f"sites_{P}_{Q}"] = ",".join(x[:2] for x in ss)
        rows.append(row)
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "pre2012_window_entry.csv", index=False)
    summ = {f"{P}_{Q}": {str(n): int((T[f"n_{P}_{Q}"] >= n).sum()) for n in (7, 8, 9, 10)}
            for P, Q in ((20, 18), (18, 16), (15, 15), (12, 12))}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    keep = T[(T.filter(like="n_").max(axis=1) >= 7)]
    print(keep.to_string(index=False))


if __name__ == "__main__":
    main()
