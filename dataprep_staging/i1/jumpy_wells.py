"""After-the-fact diagnosis (2026-10-20, user request): do the QC-dated jumps in the head records
explain the large out-of-fold errors at the "jumpy" Changhua wells?

Report only: no model, no product and no gate changes. Logs one look (after_the_fact).

What the twin scores. ``hydrophysics.twin.heads.build_head_field`` reads the raw per-well
parquet files of ``AMP_V2/data/wells`` (no L1 product), keeps 2012-01..2022-12, drops samples
outside median +/- 15 MAD, and takes month-end means; ``calibrate_flow.prepare_series`` then
back-fills every never-observed month (linear inside the record, constant at the ends), and
months 2..132 are the target and the score (``stage3_per_entry.npz`` ``obs``). This script
rebuilds those monthly heads from the raw samples (and checks them against ``obs``), so the
monthly footprint of every QC event can be computed on exactly the samples the twin used.

QC events (``results/dataprep/heads/L1_clean/v3/events.csv``, period ``pre``):

* jumps: every ``jump_v2_persistent`` (level change that persists; catalogued, never corrected
  by L1 v3), plus the edges of the flagged segments below when no persistent jump sits there;
* flagged segments (flag 9 in L1 v3, never corrected): ``offset_episode`` and ``dropout``
  (a jump out and back), ``step`` and ``excursion`` / ``jump_edge`` (one displaced side).

Step models of the out-of-fold residual r = obs - pred (R0 = ``cwa``, R1c = ``ccz``), each at
the QC dates only (no breakpoint search):

* F_seg (fixed, nothing fitted): every flagged segment displaces its samples by the QC's own
  offset (episodes and dropouts: (j1 - j2) / 2; one-sided segments: j1 on the displaced side;
  the mean offset where flagged segments overlap), averaged over the twin's kept samples of
  each month and back-filled like ``obs``; r - f.
* F (fixed): F_seg plus every unpaired persistent jump (not at a segment edge) as a permanent
  step of its QC size.
* S (free sizes): least squares of r on a constant, one regressor H(t - t_k) per QC date and a
  box per one-sided segment (same monthly averaging and back-fill). An upper bound: every free
  size absorbs some noise.
* P (placebo for S): as many dates as S has regressors, drawn at random among the well's
  sample times, 200 draws; the median RMSE says how much S gains by chance alone.

The metric is the gate's: own-mean anomaly RMSE over all 131 scored months (back-filled
months included, as the gate scores them), i.e. the standard deviation of r.

L1 v3 comparison: ``heads_monthly.parquet`` (flag 0/1 = usable; flag 9 months keep a value but
are excluded downstream), against the twin's observed (not back-filled) months, and the
predictions rescored against L1 v3.

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh ~/miniforge3/envs/hydro/bin/python \
        dataprep_staging/i1/jumpy_wells.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(REPO))

from harness import prereg3  # noqa: E402

from hydrophysics.twin.heads import _despike  # noqa: E402  (read-only reuse of the twin's QC)

OUT = REPO / "results/dataprep/reports/I1_jumpy_wells"
RUN = {"R0": "cwa", "R1c": "ccz"}
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
WELLS_DIR = REPO / "AMP_V2/data/wells"
L1 = REPO / "results/dataprep/heads/L1_clean/v3"
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
T0, T1 = pd.Timestamp("2012-01-01"), pd.Timestamp("2023-01-01")
MONTHS = pd.period_range(T0, T1 - pd.Timedelta(seconds=1), freq="M")      # 132
SEG_KINDS = ("offset_episode", "dropout", "step", "excursion", "jump_edge")
N_PLACEBO = 200
RNG = np.random.default_rng(20261020)
TOL = pd.Timedelta(hours=1)


def run_dir(s: str) -> Path:
    return REPO / f"results/twin_runs/stage3_datum_gate_{s}"


def load_entries(suffix: str) -> pd.DataFrame:
    z = np.load(run_dir(suffix) / "stage3_per_entry.npz", allow_pickle=True)
    w = pd.read_csv(run_dir(suffix) / "stage3_kfold_wells.csv", dtype={"sid": str}
                    ).set_index("entry")
    return pd.DataFrame({"entry": z["entry"].astype(int),
                         "sid": w.loc[z["entry"], "sid"].to_numpy(),
                         "layer": z["layer"].astype(int),
                         "obs": list(z["obs"].astype(float)),
                         "pred": list(z["pred"].astype(float))}).set_index("entry")


def backfill(x: np.ndarray) -> np.ndarray:
    """``calibrate_flow.prepare_series(backfill=True)``."""
    if np.isfinite(x).all():
        return x
    return pd.Series(x).interpolate(limit_direction="both").to_numpy()


class Samples:
    """The twin's kept samples of one well (window, dropna, 15-MAD despike) by month."""

    def __init__(self, sid: str):
        s = pd.read_parquet(WELLS_DIR / f"{sid}.parquet")["value"].dropna().sort_index()
        s = s[(s.index >= T0) & (s.index < T1)]
        self.n_window = len(s)
        self.raw = s
        self.s = _despike(s)
        self.t = self.s.index.to_numpy()
        per = self.s.index.to_period("M")
        mi = MONTHS.get_indexer(per)
        self.bounds = np.searchsorted(mi, np.arange(len(MONTHS) + 1))      # month edges
        self.count = np.diff(self.bounds)
        self.observed = self.count > 0
        self._interp = None
        self.monthly = np.array([self.s.iloc[a:b].mean() if b > a else np.nan
                                 for a, b in zip(self.bounds[:-1], self.bounds[1:],
                                                 strict=True)])

    def frac_after(self, tk: np.ndarray) -> np.ndarray:
        """(K, 132) share of each month's kept samples at or after each date (NaN when the
        month has no sample): the monthly mean of H(t - t_k)."""
        idx = np.searchsorted(self.t, np.asarray(tk, dtype="datetime64[ns]"))
        a, b = self.bounds[:-1][None, :], self.bounds[1:][None, :]
        n_after = np.clip(b - np.maximum(a, idx[:, None]), 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = n_after / (b - a)
        out[:, ~self.observed] = np.nan
        return out

    def scored(self, monthly: np.ndarray) -> np.ndarray:
        """A (K, 132) monthly array as the twin scores it: back-filled (``backfill``, as a
        linear map from the observed months, identical for every row), months 2..132."""
        m = np.atleast_2d(monthly)
        if self._interp is None:
            obs_i = np.flatnonzero(self.observed)
            eye = np.eye(len(obs_i))
            self._interp = np.stack([np.interp(np.arange(len(MONTHS)), obs_i, eye[k])
                                     for k in range(len(obs_i))], axis=1)     # (132, n_obs)
        return (m[:, self.observed] @ self._interp.T)[:, 1:]

    def monthly_of(self, v: np.ndarray) -> np.ndarray:
        """(132,) monthly mean over the kept samples of a per-sample array (NaN if none)."""
        c = np.r_[0.0, np.cumsum(v)]
        a, b = self.bounds[:-1], self.bounds[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (c[b] - c[a]) / (b - a)
        out[~self.observed] = np.nan
        return out

    def share_kept(self, t_a, t_b) -> float:
        """Share of the window's samples in [t_a, t_b] that the twin kept."""
        n_raw = ((self.raw.index >= t_a) & (self.raw.index <= t_b)).sum()
        n_keep = ((self.s.index >= t_a) & (self.s.index <= t_b)).sum()
        return float(n_keep / n_raw) if n_raw else np.nan


def well_events(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(jumps: date, size, source; segments) of one well. Jumps = persistent jumps plus the
    edges of flagged segments not already among them (within one hour)."""
    pj = ev[ev["kind"] == "jump_v2_persistent"]
    jumps = pd.DataFrame({"t": pd.to_datetime(pj["t_jump1"]), "j": pj["j1"].astype(float),
                          "source": "persistent_jump"})
    seg = ev[ev["kind"].isin(SEG_KINDS)].copy()
    extra = []
    for _, r in seg.iterrows():
        edges = []
        if r["kind"] in ("offset_episode", "dropout"):
            edges = [(pd.Timestamp(r["t_jump1"]), r["j1"]), (pd.Timestamp(r["t_jump2"]), r["j2"])]
        elif pd.notna(r["t_jump1"]):
            edges = [(pd.Timestamp(r["t_jump1"]), r["j1"])]
        for t, j in edges:
            if pd.isna(t) or pd.isna(j):
                continue
            if not (jumps["t"].sub(t).abs() <= TOL).any() and not any(
                    abs(t - e[0]) <= TOL for e in extra):
                extra.append((t, float(j)))
    if extra:
        jumps = pd.concat([jumps, pd.DataFrame({"t": [e[0] for e in extra],
                                                "j": [e[1] for e in extra],
                                                "source": "segment_edge"})])
    jumps = jumps[(jumps["t"] >= T0) & (jumps["t"] < T1)].sort_values("t")
    jumps = jumps.drop_duplicates("t").reset_index(drop=True)
    return jumps, seg


def events_summary(jumps: pd.DataFrame, seg: pd.DataFrame, unp: pd.DataFrame) -> str:
    """One readable line of a well's QC events (sizes in m)."""
    parts = []
    if len(seg) > 6:
        parts.append(f"{len(seg)} flagged segments ({seg['kind'].value_counts().to_dict()})")
    else:
        for _, r in seg.iterrows():
            t_a, t_b, off = seg_offset(r)
            parts.append(f"{r['kind']} {t_a:%Y-%m-%d}..{t_b:%Y-%m-%d} {off:+.2f} (flag 9)")
    if len(unp) > 6:
        parts.append(f"{len(unp)} unpaired persistent jumps (sum |j| "
                     f"{unp['j'].abs().sum():.1f})")
    else:
        for _, jr in unp.iterrows():
            parts.append(f"jump {jr['t']:%Y-%m-%d} {jr['j']:+.2f} (catalogued only)")
    return "; ".join(parts)


def seg_offset(r) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """(start, end, offset of the displaced segment against the good level)."""
    t_a, t_b = pd.Timestamp(r["t_start"]), pd.Timestamp(r["t_end"])
    if r["kind"] in ("offset_episode", "dropout"):
        off = (float(r["j1"]) - float(r["j2"])) / 2.0
    else:
        off = float(r["j1"]) if r["side"] == "after" else -float(r["j1"])
    return t_a, t_b, off


def unpaired_jumps(jumps: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Persistent jumps not at the edge (within one hour) of a flagged segment."""
    edges = pd.to_datetime(pd.concat([seg[c] for c in ("t_start", "t_end", "t_jump1",
                                                       "t_jump2")]).dropna())
    keep = [jr["source"] == "persistent_jump" and not (edges.sub(jr["t"]).abs() <= TOL).any()
            for _, jr in jumps.iterrows()]
    return jumps[np.asarray(keep, dtype=bool)] if len(jumps) else jumps


def fixed_footprints(S: Samples, seg: pd.DataFrame, unp: pd.DataFrame
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Scored (131,) monthly footprints with the QC's own sizes, nothing fitted:
    (flagged segments, unpaired persistent jumps). A segment displaces its samples by its
    offset (the mean offset where flagged segments overlap); an unpaired jump shifts every
    later sample by its size."""
    t = S.t
    tot, cnt = np.zeros(len(t)), np.zeros(len(t))
    for _, r in seg.iterrows():
        t_a, t_b, off = seg_offset(r)
        m = (t >= np.datetime64(t_a)) & (t <= np.datetime64(t_b))
        tot[m] += off
        cnt[m] += 1
    d_seg = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    d_jmp = np.zeros(len(t))
    for _, jr in unp.iterrows():
        d_jmp[t >= np.datetime64(jr["t"])] += float(jr["j"])
    return (S.scored(S.monthly_of(d_seg))[0], S.scored(S.monthly_of(d_jmp))[0])


def std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))


def lsq_rmse(r: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    """RMSE of r after least squares on a constant and the columns of X (T, K)."""
    A = np.column_stack([np.ones(len(r)), X]) if X.size else np.ones((len(r), 1))
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return std(r - A @ coef), int(np.linalg.matrix_rank(A)) - 1


def own_rmse(o: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(o) & np.isfinite(p)
    return std(o[m] - p[m]) if m.sum() >= 12 else np.nan


def shift6(x: np.ndarray, m: int) -> float:
    """mean of months m+1..m+6 minus months m-6..m-1 (scored-month index m)."""
    a, b = x[max(0, m - 6):m], x[m + 1:m + 7]
    return float(np.nanmean(b) - np.nanmean(a)) if len(a) >= 3 and len(b) >= 3 else np.nan


def main(log_look: bool) -> None:
    st = pd.read_parquet(STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    r0, r1 = load_entries(RUN["R0"]), load_entries(RUN["R1c"])
    assert (r0["sid"] == r1["sid"]).all()
    same_obs = max(float(np.nanmax(np.abs(a - b))) for a, b in zip(r0["obs"], r1["obs"],
                                                                   strict=True))
    qc = pd.read_csv(L1 / "well_qc_summary.csv", dtype={"sid": str})
    qc = qc[qc["period"] == "pre"].set_index("sid")
    ev_all = pd.read_csv(L1 / "events.csv", dtype={"sid": str}, low_memory=False)
    ev_all = ev_all[ev_all["period"] == "pre"]
    l1m = pd.read_parquet(L1 / "heads_monthly.parquet",
                          columns=["sid", "month", "period", "value", "flag", "c20",
                                   "n_days_flag9"])
    l1m = l1m[l1m["period"] == "pre"]

    wells, events = [], []
    for e, w in r0.iterrows():
        sid = w["sid"]
        county = st.loc[sid, "CountyName"]
        if county not in COUNTIES.values():
            continue
        q = qc.loc[sid] if sid in qc.index else None
        pjy = float(q["persistent_jumps_per_year"]) if q is not None else np.nan
        n_off = int(q["offset_episodes"]) if q is not None else 0
        jumpy = bool((pjy > 0) or (n_off > 0))
        S = Samples(sid)
        rebuilt = S.scored(S.monthly)[0]
        obs = w["obs"]
        rec = {"entry": e, "sid": sid, "county": [k for k, v in COUNTIES.items()
                                                  if v == county][0],
               "layer": int(w["layer"]), "jumpy": jumpy,
               "qc_persistent_jumps_per_year": pjy, "qc_offset_episodes": n_off,
               "qc_share_days_flag9": float(q["share_days_flag9"]) if q is not None else np.nan,
               "months_observed": int(S.observed[1:].sum()),
               "rebuild_max_abs_diff_m": float(np.nanmax(np.abs(rebuilt - obs))),
               "despike_dropped_samples": int(S.n_window - len(S.s))}
        resid = {k: obs - df.loc[e, "pred"] for k, df in (("r0", r0), ("r1c", r1))}
        for k in resid:
            rec[f"rmse_{k}"] = std(resid[k])
        # ---- QC events and the step models
        ev = ev_all[ev_all["sid"] == sid]
        jumps, seg = well_events(ev)
        rec["n_qc_dates"] = int(len(jumps))
        rec["n_segments"] = int(len(seg))
        rec["n_persistent_jumps"] = int((jumps["source"] == "persistent_jump").sum())
        unp = unpaired_jumps(jumps, seg)
        rec["n_unpaired_jumps"] = int(len(unp))
        rec["unpaired_jumps_abs_sum_m"] = float(unp["j"].abs().sum()) if len(unp) else 0.0
        rec["qc_events"] = events_summary(jumps, seg, unp)
        if len(jumps):
            F = S.frac_after(jumps["t"].to_numpy())                       # (K, 132)
            one_sided = seg[~seg["kind"].isin(("offset_episode", "dropout"))]
            for _, r in one_sided.iterrows():                             # a box each
                t_a, t_b, _o = seg_offset(r)
                F2 = S.frac_after(np.array([t_a, t_b + pd.Timedelta(seconds=1)],
                                           dtype="datetime64[ns]"))
                F = np.vstack([F, F2[0] - F2[1]])
            Xs = S.scored(F).T                                            # (131, K)
            fp_seg, fp_jmp = fixed_footprints(S, seg, unp)
            rec["footprint_seg_sd_m"] = std(fp_seg)
            rec["footprint_unpaired_sd_m"] = std(fp_jmp)
            for k, r in resid.items():
                rec[f"rmse_{k}_fixed_seg"] = std(r - fp_seg)
                rec[f"rmse_{k}_fixed"] = std(r - fp_seg - fp_jmp)
                rec[f"rmse_{k}_free"], rec["free_rank"] = lsq_rmse(r, Xs)
                pl = []
                for _ in range(N_PLACEBO):
                    tk = np.sort(RNG.choice(S.t, size=Xs.shape[1], replace=False))
                    pl.append(lsq_rmse(r, S.scored(S.frac_after(tk)).T)[0])
                rec[f"rmse_{k}_placebo_median"] = float(np.median(pl))
                rec[f"rmse_{k}_placebo_p05"] = float(np.quantile(pl, 0.05))
                rec[f"free_beats_placebo_share_{k}"] = float(np.mean(
                    np.asarray(pl) >= rec[f"rmse_{k}_free"]))
            # per-jump evidence in the monthly series
            for _, jr in jumps.iterrows():
                mk = MONTHS.get_loc(pd.Period(jr["t"], freq="M")) - 1      # scored index
                rawd = S.raw
                b = rawd[(rawd.index >= jr["t"] - pd.Timedelta(days=1)) & (rawd.index < jr["t"])]
                a = rawd[(rawd.index >= jr["t"]) & (rawd.index < jr["t"] + pd.Timedelta(days=1))]
                events.append({
                    "sid": sid, "county": rec["county"], "jumpy": jumpy, "type": "jump",
                    "kind": jr["source"], "t_start": jr["t"], "t_end": jr["t"],
                    "size_m": float(jr["j"]),
                    "raw_24h_shift_m": float(a.median() - b.median()) if len(a) and len(b)
                    else np.nan,
                    "l1_action": "catalogued, not corrected" if jr["source"] ==
                    "persistent_jump" else "edge of a flagged segment",
                    "obs_shift_6m_m": shift6(obs, mk) if mk >= 0 else np.nan,
                    "resid_r0_shift_6m_m": shift6(resid["r0"], mk) if mk >= 0 else np.nan})
        else:
            for k in resid:
                for s in ("fixed_seg", "fixed", "free", "placebo_median"):
                    rec[f"rmse_{k}_{s}"] = rec[f"rmse_{k}"]
        for _, sr in seg.iterrows():
            t_a, t_b, off = seg_offset(sr)
            mon = pd.period_range(t_a.to_period("M"), t_b.to_period("M"), freq="M")
            mi = MONTHS.get_indexer(mon)
            mi = mi[mi >= 0]
            # this segment's own monthly footprint on the twin's kept samples
            F2 = S.frac_after(np.array([t_a, t_b + pd.Timedelta(seconds=1)],
                                       dtype="datetime64[ns]"))
            seg_fp = off * (F2[0] - F2[1])
            l1w = l1m[(l1m["sid"] == sid)].set_index("month")
            diffs, l1_flags = [], []
            for m in mi:
                key = str(MONTHS[m])
                if key in l1w.index:
                    l1_flags.append(int(l1w.loc[key, "flag"]))
                    if l1w.loc[key, "flag"] in (0, 1) and S.observed[m]:
                        diffs.append(S.monthly[m] - l1w.loc[key, "value"])
            events.append({
                "sid": sid, "county": rec["county"], "jumpy": jumpy, "type": "segment",
                "kind": sr["kind"], "t_start": t_a, "t_end": t_b, "size_m": off,
                "duration_days": (t_b - t_a).total_seconds() / 86400.0,
                "months_touched": int(len(mi)),
                "twin_kept_share": S.share_kept(t_a, t_b),
                "twin_monthly_footprint_max_m": float(np.nanmax(np.abs(seg_fp)))
                if np.isfinite(seg_fp).any() else np.nan,
                "l1_action": "flag 9 (excluded), not corrected",
                "l1_months_flag9": int(sum(f == 9 for f in l1_flags)),
                "l1_months_usable": int(sum(f in (0, 1) for f in l1_flags)),
                "twin_minus_l1_in_usable_months_max_m": float(np.max(np.abs(diffs)))
                if diffs else np.nan})
        # ---- L1 v3 against the twin's observed months; rescoring
        l1w = l1m[l1m["sid"] == sid].set_index("month").reindex([str(p) for p in MONTHS])
        lv = l1w["value"].to_numpy(dtype=float)
        lflag = l1w["flag"].to_numpy(dtype=float)
        usable = np.isin(lflag, (0, 1)) & np.isfinite(lv)
        l1_val = np.where(usable, lv, np.nan)[1:]
        tw_obs = np.where(S.observed, S.monthly, np.nan)[1:]
        both = np.isfinite(l1_val) & np.isfinite(tw_obs)
        d = tw_obs[both] - l1_val[both]
        no9 = (l1w["n_days_flag9"].to_numpy(dtype=float)[1:] == 0)[both]
        rec.update({
            "l1_months_usable": int(np.isfinite(l1_val).sum()),
            "l1_months_flag9": int((lflag[1:] == 9).sum()),
            "twin_months_l1_flag9": int(((lflag[1:] == 9) & np.isfinite(tw_obs)).sum()),
            "twin_months_l1_missing": int((~np.isin(lflag[1:], (0, 1, 9))
                                           & np.isfinite(tw_obs)).sum()),
            "twin_minus_l1_median_abs_m": float(np.median(np.abs(d))) if d.size else np.nan,
            "twin_minus_l1_max_abs_m": float(np.max(np.abs(d))) if d.size else np.nan,
            "twin_minus_l1_n_gt_0p5m": int((np.abs(d) > 0.5).sum()),
            # months with no flagged day: the twin's sample-weighted mean against L1's
            # day-weighted mean (logging cadence changes, e.g. hourly -> 10 min in 2019)
            "twin_minus_l1_n_gt_0p5m_no_flag9_day": int(((np.abs(d) > 0.5) & no9).sum())})
        l1_bf = backfill(np.where(usable, lv, np.nan))[1:] if usable.any() else l1_val
        not9 = lflag[1:] != 9
        for k, df in (("r0", r0), ("r1c", r1)):
            p = df.loc[e, "pred"]
            rec[f"rmse_{k}_observed"] = own_rmse(tw_obs, p)
            rec[f"rmse_{k}_observed_not_flag9"] = own_rmse(np.where(not9, tw_obs, np.nan), p)
            rec[f"rmse_{k}_twin_l1months"] = own_rmse(np.where(both, tw_obs, np.nan), p)
            rec[f"rmse_{k}_l1"] = own_rmse(np.where(both, l1_val, np.nan), p)
            rec[f"rmse_{k}_l1_backfilled"] = own_rmse(l1_bf, p)
        wells.append(rec)
        print(f"{sid} {rec['county']} jumpy={jumpy} K={rec['n_qc_dates']} "
              f"R0 {rec['rmse_r0']:.3f}->{rec['rmse_r0_fixed_seg']:.3f}/"
              f"{rec['rmse_r0_fixed']:.3f}/{rec['rmse_r0_free']:.3f} "
              f"R1c {rec['rmse_r1c']:.3f}->{rec['rmse_r1c_fixed']:.3f}/"
              f"{rec['rmse_r1c_free']:.3f}", flush=True)

    W = pd.DataFrame(wells).set_index("entry")
    E = pd.DataFrame(events)
    summary = {}
    for cname in COUNTIES:
        d = W[W["county"] == cname]
        j = d[d["jumpy"]]
        c = d[~d["jumpy"]]
        out = {"n_wells": int(len(d)), "n_jumpy": int(len(j))}
        for k in ("r0", "r1c"):
            out[k] = {
                "all_wells_mean_rmse": float(d[f"rmse_{k}"].mean()),
                "jumpy_mean_rmse": float(j[f"rmse_{k}"].mean()),
                "clean_mean_rmse": float(c[f"rmse_{k}"].mean()),
                "jumpy_mean_rmse_fixed_segments_removed": float(j[f"rmse_{k}_fixed_seg"].mean()),
                "jumpy_mean_rmse_fixed_steps_removed": float(j[f"rmse_{k}_fixed"].mean()),
                "jumpy_mean_rmse_free_steps_removed": float(j[f"rmse_{k}_free"].mean()),
                "jumpy_mean_rmse_placebo_median": float(j[f"rmse_{k}_placebo_median"].mean()),
                "all_wells_mean_rmse_fixed_seg": float(d[f"rmse_{k}_fixed_seg"].mean()),
                "all_wells_mean_rmse_fixed": float(d[f"rmse_{k}_fixed"].mean()),
                "all_wells_mean_rmse_free": float(d[f"rmse_{k}_free"].mean()),
                "all_wells_mean_rmse_placebo": float(d[f"rmse_{k}_placebo_median"].mean()),
                "jumpy_var_explained_fixed_seg_median": float(np.median(
                    1 - (j[f"rmse_{k}_fixed_seg"] / j[f"rmse_{k}"]) ** 2)),
                "jumpy_var_explained_fixed_median": float(np.median(
                    1 - (j[f"rmse_{k}_fixed"] / j[f"rmse_{k}"]) ** 2)),
                "jumpy_var_explained_free_median": float(np.median(
                    1 - (j[f"rmse_{k}_free"] / j[f"rmse_{k}"]) ** 2)),
                "jumpy_var_explained_placebo_median": float(np.median(
                    1 - (j[f"rmse_{k}_placebo_median"] / j[f"rmse_{k}"]) ** 2)),
                "wells_free_beats_95pct_placebo": int((j[f"free_beats_placebo_share_{k}"]
                                                       >= 0.95).sum()),
                "wells_fixed_lowers_rmse": int((j[f"rmse_{k}_fixed"] < j[f"rmse_{k}"]).sum()),
                "wells_fixed_seg_lowers_rmse": int((j[f"rmse_{k}_fixed_seg"]
                                                    < j[f"rmse_{k}"] - 1e-9).sum()),
                "all_wells_mean_rmse_observed": float(d[f"rmse_{k}_observed"].mean()),
                "all_wells_mean_rmse_observed_not_flag9": float(
                    d[f"rmse_{k}_observed_not_flag9"].mean()),
                "jumpy_mean_rmse_observed": float(j[f"rmse_{k}_observed"].mean()),
                "jumpy_mean_rmse_observed_not_flag9": float(
                    j[f"rmse_{k}_observed_not_flag9"].mean()),
                "all_wells_mean_rmse_twin_l1months": float(d[f"rmse_{k}_twin_l1months"].mean()),
                "all_wells_mean_rmse_l1": float(d[f"rmse_{k}_l1"].mean()),
                "all_wells_mean_rmse_l1_backfilled": float(d[f"rmse_{k}_l1_backfilled"].mean()),
                "jumpy_mean_rmse_twin_l1months": float(j[f"rmse_{k}_twin_l1months"].mean()),
                "jumpy_mean_rmse_l1": float(j[f"rmse_{k}_l1"].mean())}
        out["gap_r1c_minus_r0"] = {
            s: out["r1c"][f"all_wells_mean_rmse{s}"] - out["r0"][f"all_wells_mean_rmse{s}"]
            for s in ("", "_fixed_seg", "_fixed", "_free", "_placebo", "_observed",
                      "_observed_not_flag9", "_twin_l1months", "_l1", "_l1_backfilled")}
        out["gap_r1c_minus_r0_jumpy_only"] = {
            "as_scored": out["r1c"]["jumpy_mean_rmse"] - out["r0"]["jumpy_mean_rmse"],
            "fixed_seg": out["r1c"]["jumpy_mean_rmse_fixed_segments_removed"]
            - out["r0"]["jumpy_mean_rmse_fixed_segments_removed"],
            "fixed": out["r1c"]["jumpy_mean_rmse_fixed_steps_removed"]
            - out["r0"]["jumpy_mean_rmse_fixed_steps_removed"],
            "free": out["r1c"]["jumpy_mean_rmse_free_steps_removed"]
            - out["r0"]["jumpy_mean_rmse_free_steps_removed"]}
        ok = j[["rmse_r1c", "qc_persistent_jumps_per_year"]].dropna()
        out["twin_vs_l1"] = {
            "wells_with_twin_month_l1_flag9": int((d["twin_months_l1_flag9"] > 0).sum()),
            "twin_months_l1_flag9_total": int(d["twin_months_l1_flag9"].sum()),
            "median_abs_twin_minus_l1_m_median_over_wells": float(
                d["twin_minus_l1_median_abs_m"].median()),
            "months_gt_0p5m_total": int(d["twin_minus_l1_n_gt_0p5m"].sum()),
            "months_gt_0p5m_without_flagged_days": int(
                d["twin_minus_l1_n_gt_0p5m_no_flag9_day"].sum()),
            "wells_with_months_gt_0p5m": int((d["twin_minus_l1_n_gt_0p5m"] > 0).sum())}
        out["spearman_vs_jumps_per_year_all_wells"] = {}
        for col in ("rmse_r0", "rmse_r1c", "rmse_r0_fixed_seg", "rmse_r1c_fixed_seg",
                    "rmse_r0_free", "rmse_r1c_free", "rmse_r0_l1", "rmse_r1c_l1"):
            m = d[[col, "qc_persistent_jumps_per_year"]].dropna()
            rho, p = spearmanr(m.iloc[:, 0], m.iloc[:, 1])
            out["spearman_vs_jumps_per_year_all_wells"][col] = {"rho": float(rho),
                                                                 "p": float(p)}
        if len(ok) >= 8:
            rho, p = spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])
            out["jumpy_only_spearman_rmse_r1c_vs_jumps_per_year"] = {"rho": float(rho),
                                                                      "p": float(p)}
        summary[cname] = out
    ej = E[E["jumpy"]] if len(E) else E
    jj = ej[ej["type"] == "jump"].dropna(subset=["raw_24h_shift_m"])
    ev_sum = {
        "jumps_total": int((ej["type"] == "jump").sum()),
        "segments_total": int((ej["type"] == "segment").sum()),
        "segments_by_kind": ej[ej["type"] == "segment"]["kind"].value_counts().to_dict(),
        "qc_jump_vs_raw_24h_shift_corr": float(np.corrcoef(jj["size_m"],
                                                           jj["raw_24h_shift_m"])[0, 1])
        if len(jj) > 2 else np.nan,
        "segments_twin_kept_share_median": float(ej.loc[ej["type"] == "segment",
                                                        "twin_kept_share"].median())}
    rec = {"schema": "i1-jumpy-wells/1", "runs": RUN,
           "role": "after_the_fact_analysis (U7: never a gate, threshold or choice)",
           "twin_head_source": {
               "files": "AMP_V2/data/wells/<sid>.parquet (raw logger values), stations "
                        "AMP_V2/data/fan_stations.parquet, groundwater zone 50",
               "qc": "window 2012-01..2022-12, dropna, median +/- 15 MAD despike, coverage "
                     ">= 0.80 of hours and longest gap <= 180 days, month-end mean",
               "backfill": "linear inside the record, constant at the ends (no_backfill=false "
                           "in both runs); months 2..132 are target and score",
               "uses_L1_clean_v3": False,
               "rebuild_check_max_abs_diff_m": float(W["rebuild_max_abs_diff_m"].max()),
               "r0_r1c_obs_identical_max_abs_diff_m": same_obs},
           "metric": "own-mean anomaly RMSE (sd of obs - pred) over the 131 scored months, "
                     "back-filled months included, as the gate scores them",
           "n_placebo": N_PLACEBO, "events": ev_sum, "counties": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    W.round(4).to_csv(OUT / "wells.csv")
    E.round(4).to_csv(OUT / "events.csv", index=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if not log_look:
        print("dry run: no look logged (pass --log-look)")
        return
    look = prereg3.record_look(
        ["I1:design_oof_heads_R0_R1e"], "heads:2012-01..2022-12", "I1", "after_the_fact",
        note="Jumpy-well diagnosis (user request 2026-10-20): traced the twin's head source "
             "(raw AMP_V2 wells, not L1_clean v3); fitted step functions at L1 v3 QC-dated "
             "jumps only (fixed QC sizes, free sizes, random-date placebo) to the out-of-fold "
             "R0/R1c residuals of Changhua and Yunlin wells; compared and rescored against L1 "
             "v3 monthly heads. Report only. results/dataprep/reports/I1_jumpy_wells/")
    print("look logged:", look["time"])


if __name__ == "__main__":
    main(log_look="--log-look" in sys.argv[1:])
