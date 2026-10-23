"""H1a repair round (2026-10-13), post-look correction review: v3 decisions from the v2
decisions (python -I).

Why. The after-the-fact verification of the registered v2 build (``v2_verify.py``) compared
every flag-1 step correction with the co-located nested wells of the same site. For 7 of the
8 co-located-backed corrections (6 pre-origin, 1 post-origin) the well-minus-sibling
difference did not shift by half the jump over the following year (median over siblings of
delta / j below 0.5); the 8th (07100211, 2025-05) cannot be checked over a year because its
sibling records only 7 days after the jump (on those days the difference moved -0.50 m
against a jump of -11.05 m). In 09160121 (2013-05) and 09200111 (2012-01..2015-08) the
corrected values moved the well 3-4 m away from a sibling whose one-year median difference
before the jump was 0.1-1.4 m. A nested well that records through the jump without a
simultaneous jump is what a datum artefact shows, but it is also what a
well-specific real event shows (a pump stopping, recovery from local drawdown). The
co-located test of plan WP H1 as implemented in v1/v2 ("no simultaneous jump") therefore
does not back a correction. Plan WP H1: "the default is to flag, not correct".

Rule (post-look, not pre-registered; no tunable constant): an accepted step whose only
backing is co-located (``backing`` starts with "co-located") is no longer corrected: its
displaced segment is flagged 9, exactly as an accepted step without backing. Steps backed by
physical metadata (impossible level, metadata date) keep flag 1; steps corroborated only by
the neighbour regression keep flag 3 (plan rule; never a calibration target). Nothing else
changes: detections, episodes, runs and the jump catalogue are v2's, so the registered
injection recall (which never reads flags) is unchanged by construction, and the leakage
probe and the conformance checks are recomputed on the v3 build.

Pre-origin decisions are reviewed from their own (pre-origin) decision rows only, so the
pre-origin rule of v2 (no 2023+ sample decides a pre-origin row) still holds.

Usage: s2c_review_v3.py WORK_V2 WORK_V3 STAGE1_SUBDIR
Writes WORK_V3/decisions_v2_<sub>.parquet (the file name is the interface the registered
assembly and validation scripts read; its rows are the reviewed v3 decisions, with the v2
values kept in ``flag_v2``, ``backing_v2`` and ``review_v3``) and links WORK_V3/<sub>,
station_meta_clean.parquet and meta_summary.json to WORK_V2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REVIEW_NOTE = ("none (repair round v3: co-located backing alone no longer corrects a step; "
               "the after-the-fact sibling-difference check did not confirm 7 of the 8 v2 "
               "co-located corrections and could not test the 8th; flagged, not corrected)")


def review(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["flag_v2"] = D["flag"]
    D["backing_v2"] = D["backing"]
    D["review_v3"] = ""
    m = (D["kind"] == "step") & (D["flag"] == 1) & \
        D["backing"].fillna("").str.startswith("co-located")
    D.loc[m, "flag"] = 9.0
    D.loc[m, "backing"] = REVIEW_NOTE
    D.loc[m, "review_v3"] = "co-located-only correction demoted to flag 9"
    return D


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        raise SystemExit(f"{dst} exists and is not the expected link")
    dst.symlink_to(src)


def main():
    w2, w3, sub = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    w3.mkdir(parents=True, exist_ok=True)
    for name in (sub, "station_meta_clean.parquet", "meta_summary.json"):
        link((w2 / name).resolve(), w3 / name)
    D = pd.read_parquet(w2 / f"decisions_v2_{sub}.parquet")
    R = review(D)
    R.to_parquet(w3 / f"decisions_v2_{sub}.parquet", index=False)
    ch = R[R.review_v3 != ""]
    ch[["sid", "period", "kind", "t1", "j1", "side", "t_lo", "t_hi", "flag_v2", "flag",
        "backing_v2"]].to_csv(w3 / f"review_v3_{sub}.csv", index=False)
    print(f"{sub}: {len(ch)} step corrections demoted to flag 9 "
          f"({ch.groupby('period').size().to_dict()})")
    print(R[R.kind == "step"].groupby(["period", "flag"]).size().to_string())


if __name__ == "__main__":
    main()
