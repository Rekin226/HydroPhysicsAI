"""WP S3 fix round of the S3r repair (2026-10-18), diagnosis BEFORE any method change: the v3
R6 WRA detection common mode (daily median of the pass residuals of the 5 nearest member
stations) at changes of neighbour composition.

    dataprep_staging/tools/cpu_slot.sh python -I -B \
        dataprep_staging/s3_gnss/v4/diag_cm_composition.py OUT_DIR

Re-runs the v3 WRA family (full records and pre-origin, no injection) with the unchanged v3
code (imported read-only from ``../v3``) and the v3 tuned parameters, keeping the pass-2
residual frame from which the final (pass-3) detection common mode is computed. Writes:

* ``gaps.csv``: every gap longer than 10 days of every WRA member series (the pass-2
  residual frame: screened observations not after an untied offset);
* ``events_<variant>.csv``: for every station and every day on which one or more of its 5
  neighbours leaves or returns (gap > 10 d), the common-mode jump (median over the 10
  days after minus the 10 days before) and the same jump of the median over the neighbours
  that report on both sides (constant subset); their difference is the composition part;
* ``cm_only_steps_<variant>.csv``: every common-mode-only step of the v3 product with the
  composition event within +-5 days, if any;
* ``summary.json``.
No injection and no A1/A2/A3 score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1]))   # dataprep_staging
sys.path.insert(0, str(HERE.parent / "v3"))   # the unchanged v3 code
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
import run_validation as RV  # noqa: E402

TUNE = REPO / "results/dataprep/reports/S3_evidence/repair_2026-10-18/tune_v3c/tune.json"
PRODUCT = REPO / "results/dataprep/gnss_clean"
W = 10          # days on each side of an event
GAP = 10        # a neighbour gap longer than this is a composition event


def run_wra_keep(p: M.Params, window_end) -> dict:
    """``pipeline.run_wra`` (v3) unrolled to keep the pass-2 residual frame and the final
    detection common mode."""
    ser = P.wra_series(window_end)
    cat = {k: [(d, t) for d, t in v if window_end is None or pd.Timestamp(d) <= window_end]
           for k, v in P.WRA_CATALOGUE.items()}
    drop, flag = P.wra_policy(window_end)
    proc = {k: v for k, v in ser.items() if k not in drop}
    members = [k for k in proc if k not in flag]
    xy = P.wra_station_xy()
    fam = P.Family("wra", {})
    cm, frames = None, []
    for ipass in range(3):
        fam.results = {k: M.process_station(s, p, cm=None if cm is None else cm.get(k),
                                            catalogue=cat.get(k)) for k, s in proc.items()}
        if ipass < 2:
            R = fam.resid_frame()
            frames.append(R)
            cm = P._knn_median(R, members, xy)
    return {"fam": fam, "R2": frames[1], "cm": cm, "members": members, "xy": xy,
            "stations": list(proc)}


def neighbours(s: str, members: list[str], xy: dict) -> list[str]:
    cand = [m for m in members if m != s]
    if s in xy and all(m in xy for m in cand):
        cand = sorted(cand, key=lambda m: np.hypot(xy[m][0] - xy[s][0],
                                                   xy[m][1] - xy[s][1]))[:P.KNN_WRA]
    return cand


def gaps_of(avail: pd.Series, min_len: int = GAP) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(first missing day, first day back, length) of every run of missing days longer than
    ``min_len`` between the first and last observation."""
    a = avail.to_numpy()
    idx = avail.index
    obs = np.flatnonzero(a)
    out = []
    if len(obs) < 2:
        return out
    d = np.diff(obs)
    for j in np.flatnonzero(d > min_len + 1):
        i0, i1 = obs[j], obs[j + 1]
        out.append((idx[i0 + 1], idx[i1], int(i1 - i0 - 1)))
    return out


def events(run: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    R = run["R2"].sort_index()
    full_idx = pd.date_range(R.index.min(), R.index.max(), freq="D")
    R = R.reindex(full_idx)
    members = run["members"]
    gap_rows = []
    member_gaps = {}
    for k in members:
        if k not in R:
            continue
        av = R[k].notna()
        first, last = av[av].index.min(), av[av].index.max()
        member_gaps[k] = gaps_of(av.loc[first:last])
        for g0, g1, n in member_gaps[k]:
            gap_rows.append({"variant": variant, "station": k, "first_missing": g0.date(),
                             "back": g1.date(), "days": n})
        # record start and end inside the window are composition events too
        member_gaps[k] = member_gaps[k] + [(full_idx[0], first, 0), (last + pd.Timedelta(days=1),
                                                                     full_idx[-1], -1)]
    rows = []
    for s in run["stations"]:
        nb = [m for m in neighbours(s, members, run["xy"]) if m in R]
        if not nb:
            continue
        cm_s = R[nb].median(axis=1)
        ev_days: dict[pd.Timestamp, list[str]] = {}
        for k in nb:
            for g0, g1, n in member_gaps.get(k, []):
                for d, kind in ((g0, "leaves"), (g1, "returns")):
                    if n == 0 and kind == "leaves":
                        continue          # record start: only its "returns" (= first day)
                    if n == -1 and kind == "returns":
                        continue          # record end: only its "leaves"
                    if full_idx[0] < d <= full_idx[-1]:
                        ev_days.setdefault(d, []).append(f"{k} {kind}")
        for d, what in sorted(ev_days.items()):
            lo, hi = d - pd.Timedelta(days=W), d + pd.Timedelta(days=W - 1)
            before = R.loc[lo:d - pd.Timedelta(days=1), nb]
            after = R.loc[d:hi, nb]
            if before.empty or after.empty:
                continue
            both = [k for k in nb if before[k].notna().sum() >= 3 and after[k].notna().sum() >= 3]
            n_b = int((before.notna().sum() >= 3).sum())
            n_a = int((after.notna().sum() >= 3).sum())
            j_all = float(cm_s.loc[d:hi].median() - cm_s.loc[lo:d - pd.Timedelta(days=1)].median())
            if both:
                cs = R[both].median(axis=1)
                j_const = float(cs.loc[d:hi].median() - cs.loc[lo:d - pd.Timedelta(days=1)].median())
            else:
                j_const = float("nan")
            own = R[s] if s in R else pd.Series(np.nan, index=R.index)
            j_own = float(own.loc[d:hi].median() - own.loc[lo:d - pd.Timedelta(days=1)].median())
            rows.append({"variant": variant, "station": s, "date": d.date(),
                         "what": "; ".join(what), "neighbours": " ".join(nb),
                         "n_reporting_before": n_b, "n_reporting_after": n_a,
                         "n_constant_subset": len(both),
                         "cm_jump_mm": 1000 * j_all, "constant_subset_jump_mm": 1000 * j_const,
                         "composition_part_mm": 1000 * (j_all - j_const),
                         "own_resid_jump_mm": 1000 * j_own})
    return pd.DataFrame(gap_rows), pd.DataFrame(rows)


def cm_only(variant: str, ev: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_csv(PRODUCT / variant / "v3" / "steps.csv")
    st = st[(st.family == "wra") & (st.basis == "common_mode_only")].copy()
    out = []
    for r in st.itertuples():
        d = pd.Timestamp(r.date)
        e = ev[ev.station == r.station] if len(ev) else ev
        near = e[(pd.to_datetime(e.date) - d).abs().dt.days <= 5] if len(e) else e
        best = near.loc[near.composition_part_mm.abs().idxmax()] if len(near) else None
        out.append({"variant": variant, "station": r.station, "date": r.date,
                    "size_mm": 1000 * r.size_m, "own_series_size_mm": 1000 * r.size_own_series_m,
                    "z_own_series": r.z_own_series,
                    "event_date": None if best is None else best.date,
                    "event": None if best is None else best.what,
                    "n_before_after": None if best is None else
                    f"{best.n_reporting_before}->{best.n_reporting_after}",
                    "cm_jump_mm": None if best is None else best.cm_jump_mm,
                    "composition_part_mm": None if best is None else best.composition_part_mm})
    return pd.DataFrame(out)


def flag_days(variant: str, stations: list[str], start: str, end: str) -> dict:
    d = pd.read_parquet(PRODUCT / variant / "v3" / "wra" / "daily.parquet",
                        columns=["station", "date", "flag"])
    d = d[(d.date >= start) & (d.date <= end) & d.station.isin(stations)]
    return {s: {str(f): int(n) for f, n in g.flag.value_counts().sort_index().items()}
            for s, g in d.groupby("station")}


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for variant, end in (("full", None), ("preorigin", P.ORIGIN_END)):
        p, _, _ = RV.load_tuned(str(TUNE), variant)
        run = run_wra_keep(p, end)
        g, ev = events(run, variant)
        g.to_csv(od / f"gaps_{variant}.csv", index=False)
        ev.to_csv(od / f"events_{variant}.csv", index=False)
        co = cm_only(variant, ev)
        co.to_csv(od / f"cm_only_steps_{variant}.csv", index=False)
        R = run["R2"]
        nrep = {}
        for s in run["stations"]:
            nb = [m for m in neighbours(s, run["members"], run["xy"]) if m in R]
            own = R[s].notna() if s in R else None
            cnt = R[nb].notna().sum(axis=1)
            if own is not None:
                cnt = cnt[own]
            nrep[s] = {"days": int(len(cnt)), "share_lt5": float((cnt < 5).mean()),
                       "share_le2": float((cnt <= 2).mean()), "share_0": float((cnt == 0).mean())}
        big = ev[ev.composition_part_mm.abs() >= 10] if len(ev) else ev
        summary[variant] = {
            "n_events": int(len(ev)), "n_events_composition_ge_10mm": int(len(big)),
            "events_composition_ge_10mm": big[["station", "date", "what", "n_reporting_before",
                                               "n_reporting_after", "cm_jump_mm",
                                               "constant_subset_jump_mm",
                                               "composition_part_mm"]].to_dict("records")
            if len(big) else [],
            "n_cm_only_steps": int(len(co)),
            "n_cm_only_at_composition_event_ge_10mm": int(
                (pd.to_numeric(co.composition_part_mm, errors="coerce").abs() >= 10).sum())
            if len(co) else 0,
            "neighbour_reporting": nrep,
            "members": run["members"]}
        if variant == "full":
            summary[variant]["flags_2023_25_v3"] = flag_days(
                "full", ["ANES", "HLES", "HNES", "YCES", "KTES", "SSES"], "2023-01-01",
                "2025-03-31")
    (od / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for v in summary:
        print(v, {k: summary[v][k] for k in ("n_events", "n_events_composition_ge_10mm",
                                             "n_cm_only_steps",
                                             "n_cm_only_at_composition_event_ge_10mm")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
