"""H1a acceptance tests 2-5 and reported checks (python -I, CPU slot), after stage 3.

Usage: s4_validate.py WORKDIR BUILD_DIR OUT_JSON

2. Audit agreement: share of the audit's raw-record events (``_pilots/audit_2026-10-11/
   artefact_events.csv``: every spike and offset jump, period 'hist' = 2012-22 and 'new' =
   2023-26) matched by an H1a jump of the same well within +-1 day (criterion >= 0.95 on the
   pre-origin events; the post-origin share is reported). Also the class agreement and the
   reverse share (H1a offsets and spikes matched by the audit).
3. Known real events stay unflagged: no flag-9 day, and no artefact event or correction, in
   the windows of ``config.REAL_EVENTS`` (post-origin by construction; plan-mandated).
4. Barometric O1 phase coherence: over wells with an accepted barometric correction and
   >= 12 monthly O1 phases in 2012-22, the paired median of coherence(corrected) -
   coherence(raw) is > 0 (sign test reported); the all-well medians and the 2023-26 values
   are reported.
5. No flagged offset episode coincides with a pressure drop >= 20 hPa: the largest pressure
   fall within +-24 h of either jump of every offset episode (any flag), at the well's
   ERA5 cell; criterion: zero on 2012-22 episodes; 2023-26 reported. The number of such
   falls in the 2012-22 pressure record is reported (power of the check).
Reported only: CWA station pressure against the Open-Meteo ERA5 cells (daily 2012-2026 at
467480 and 467490; hourly weeks at 467480, 467490, 467270), the frozen-challenge rescore on
L1 monthly means (the audit expects < 0.01 change in the ratio), event counts against the
plan's numbers, flag shares and C20 coverage.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import L0_NEW, PILOTS, REPO  # noqa: E402

from h1_heads_qc import baro  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402

DAY = 86400
AUDIT_EVENTS = PILOTS / "audit_2026-10-11" / "artefact_events.csv"
EVAL = REPO / "results" / "twin" / "hybrid_datum_gate_innov" / "evaluation.csv"
CODIS_DIR = sorted(L0_NEW.glob("pressure_hourly/cwa_codis_*"))[-1]
CWA_STATIONS = {"467270": (23.873803, 120.581286), "467480": (23.495925, 120.432906),
                "467490": (24.145736, 120.684075)}
ARTEFACT_EVENT_KINDS = set(C.ARTEFACT_KINDS)


def tsec(x) -> int:
    return int(pd.Timestamp(x).value // 10**9)


# ----------------------------------------------------------------------------- 2 audit
def audit_agreement(work: Path) -> dict:
    ev = pd.read_csv(AUDIT_EVENTS, dtype={"sid": str})
    ev["ts"] = ev["time"].map(tsec)
    ours = {}
    for p in sorted((work / "stage1").glob("*.pkl")):
        res = pickle.loads(Path(p).read_bytes())
        for part, per in (("pre_cat", "hist"), ("post_cat", "new")):
            if part in res:
                J = res[part]["jumps"]
                ours[(res["sid"], per)] = (np.array([j["t"] for j in J], np.int64),
                                           np.array([j["cls"] for j in J], object),
                                           np.array([j["j"] for j in J], float))
    out = {}
    rows = []
    for per, g in ev.groupby("period"):
        hit = cls_hit = 0
        for _, r in g.iterrows():
            t, c, _ = ours.get((r.sid, per), (np.array([], np.int64), np.array([]), None))
            near = np.abs(t - r.ts) <= C.AUDIT_TOL_DAYS * DAY
            h = bool(near.any())
            ch = bool((near & (c == r.kind)).any()) if h else False
            hit += h
            cls_hit += ch
            rows.append({"period": per, "sid": r.sid, "kind": r.kind, "time": r.time,
                         "jump_m": r.jump_m, "matched": h, "class_matched": ch})
        n = len(g)
        # reverse: our offsets / spikes matched by the audit
        rev_n = rev_hit = 0
        at = {s: gg.ts.to_numpy() for s, gg in g.groupby("sid")}
        for (sid, pp), (t, c, _) in ours.items():
            if pp != per:
                continue
            sel = np.isin(c, ["offset", "spike"])
            for x in t[sel]:
                rev_n += 1
                a = at.get(sid)
                rev_hit += bool(a is not None and (np.abs(a - x) <= DAY).any())
        out[per] = {"n_audit_events": n, "matched": hit, "agreement": hit / n if n else None,
                    "class_agreement": cls_hit / n if n else None,
                    "by_kind": {k: float(gg.kind.map(lambda _: 1).sum()) for k, gg in
                                g.groupby("kind")},
                    "reverse_n_h1a_offsets_spikes": rev_n,
                    "reverse_share_matched_by_audit": rev_hit / rev_n if rev_n else None}
    R = pd.DataFrame(rows)
    for per in out:
        sub = R[R.period == per]
        out[per]["agreement_by_kind"] = sub.groupby("kind").matched.mean().round(4).to_dict()
        out[per]["unmatched_examples"] = sub[~sub.matched].head(15).to_dict("records")
    return out


# ----------------------------------------------------------------------------- 3 real events
def real_events(build: Path) -> dict:
    D = pd.read_parquet(build / "heads_daily.parquet")
    E = pd.read_csv(build / "events.csv", dtype={"sid": str})
    E["t0"] = pd.to_datetime(E["t_start"])
    E["t1_"] = pd.to_datetime(E["t_end"].where(E["t_end"].astype(str) != "", E["t_start"]),
                              errors="coerce").fillna(E["t0"])
    res = []
    for ev in C.REAL_EVENTS:
        a, b = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]) + pd.Timedelta(days=1)
        d = D[(D.sid == ev["sid"]) & (D.date >= a) & (D.date < b)]
        e = E[(E.sid == ev["sid"]) & (E.kind.isin(ARTEFACT_EVENT_KINDS))
              & (E.t0 < b) & (E.t1_ >= a)]
        corr = d[(d.corr_m.abs() > 0)]
        flagged = bool((d.flag == 9).any() or len(e) or len(corr))
        res.append(ev | {"n_days_in_record": int(len(d)), "n_days_flag9": int((d.flag == 9).sum()),
                         "n_artefact_events": int(len(e)),
                         "artefact_events": e[["kind", "t_start", "t_end", "flag"]]
                         .to_dict("records"),
                         "n_days_corrected": int(len(corr)),
                         "daily_values": d[["date", "value", "flag", "n_hours"]]
                         .astype({"date": str}).to_dict("records"),
                         "flagged": flagged})
    return {"events": res, "n_flagged": int(sum(r["flagged"] for r in res)),
            "n_events": len(res)}


# ----------------------------------------------------------------------------- 4 O1
def o1_check(build: Path) -> dict:
    Cc = pd.read_csv(build / "o1_coherence.csv", dtype={"sid": str})
    B = pd.read_csv(build / "barometric_efficiency.csv", dtype={"sid": str})
    acc = set(B[B.accepted].sid)          # wells with at least one accepted epoch
    out = {}
    for per in ("pre", "post"):
        g = Cc[(Cc.period == per) & (Cc.n_months >= 12)]
        w = g.pivot(index="sid", columns="version", values="coh_O1").dropna()
        allm = {"n_wells": int(len(w)), "median_raw": float(w["raw"].median()),
                "median_corr": float(w["corr"].median())}
        wc = w[w.index.isin(acc)]
        dlt = (wc["corr"] - wc["raw"])
        nz = dlt[dlt != 0]
        sign_p = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) \
            if len(nz) else None
        out[per] = {"all_wells": allm,
                    "corrected_wells": {"n_wells": int(len(wc)),
                                        "median_raw": float(wc["raw"].median()) if len(wc)
                                        else None,
                                        "median_corr": float(wc["corr"].median()) if len(wc)
                                        else None,
                                        "median_paired_diff": float(dlt.median()) if len(wc)
                                        else None,
                                        "n_improved": int((dlt > 0).sum()),
                                        "n_worse": int((dlt < 0).sum()),
                                        "sign_test_p": sign_p}}
    return out


# ----------------------------------------------------------------------------- 5 pressure drops
def max_fall(p: pd.Series, t: int, half_h: int = 24) -> float:
    a = pd.Timestamp(t, unit="s") - pd.Timedelta(hours=half_h)
    b = pd.Timestamp(t, unit="s") + pd.Timedelta(hours=half_h)
    w = p[(p.index >= a) & (p.index <= b)].dropna().to_numpy()
    if len(w) < 2:
        return np.nan
    run_max = np.maximum.accumulate(w)
    return float(np.max(run_max - w))


def pressure_drops(work: Path, P: pd.DataFrame) -> dict:
    dec = pd.read_parquet(work / "decisions_stage1.parquet")
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    out = {}
    for per in ("pre", "post"):
        g = dec[(dec.period == per) & (dec.kind.isin(["offset_episode", "dropout", "step"]))]
        rows = []
        for _, r in g.iterrows():
            cid = meta.loc[r.sid, "era5_cell"] if r.sid in meta.index else None
            if cid is None or cid not in P.columns:
                continue
            f1 = max_fall(P[cid], int(r.t1))
            f2 = max_fall(P[cid], int(r.t2)) if r.kind != "step" else np.nan
            rows.append({"sid": r.sid, "kind": r.kind, "t1": str(pd.Timestamp(int(r.t1),
                                                                              unit="s")),
                         "flag": int(r.flag), "fall1_hpa": f1, "fall2_hpa": f2,
                         "max_fall_hpa": np.nanmax([f1, f2])})
        R = pd.DataFrame(rows)
        ep = R[R.kind == "offset_episode"] if len(R) else R
        out[per] = {"n_offset_episodes": int(len(ep)),
                    "n_offset_episodes_on_fall_ge_20hpa": int(
                        (ep.max_fall_hpa >= C.PRESSURE_DROP_HPA).sum()) if len(ep) else 0,
                    "max_fall_at_offset_episodes_hpa": float(ep.max_fall_hpa.max())
                    if len(ep) else None,
                    "n_steps_on_fall_ge_20hpa": int(((R.kind == "step") &
                                                     (R.max_fall_hpa >= C.PRESSURE_DROP_HPA))
                                                    .sum()) if len(R) else 0,
                    "n_dropouts_on_fall_ge_20hpa": int(((R.kind == "dropout") &
                                                        (R.max_fall_hpa >= C.PRESSURE_DROP_HPA))
                                                       .sum()) if len(R) else 0,
                    "episodes_on_fall_ge_10hpa": ep[ep.max_fall_hpa >= 10].to_dict("records")
                    if len(ep) else []}
    # power: falls >= 20 hPa within 48 h in the 2012-22 record, per cell (daily scan)
    power = {}
    for cid in P.columns:
        p = P[cid]
        p = p[(p.index >= "2012-01-01") & (p.index < "2023-01-01")]
        r48 = p.rolling(49, min_periods=40)
        fall = (r48.max() - p)          # fall from the running 48-h max to now
        ev = fall[fall >= C.PRESSURE_DROP_HPA]
        days = sorted(set(ev.index.normalize()))
        # group consecutive days into events
        n = 0
        last = None
        for dd in days:
            if last is None or (dd - last).days > 2:
                n += 1
            last = dd
        power[cid] = {"n_fall_events_ge_20hpa_2012_2022": n,
                      "max_48h_fall_hpa": float(fall.max())}
    out["power_pressure_record"] = power
    return out


# ----------------------------------------------------------------------------- CWA check
def cwa_check(P: pd.DataFrame) -> dict:
    rec = json.loads((CODIS_DIR / "SOURCE.json").read_text())
    cells = baro.cells_available(P)
    daily, hourly = {}, {}
    for f in rec["files"]:
        j = json.loads((CODIS_DIR / f["file"]).read_text())
        for st in j.get("data", []):
            sid = st.get("StationID")
            for r in st.get("dts", []):
                sp = r.get("StationPressure") or {}
                if f["kind"].startswith("daily"):
                    v = sp.get("Mean")
                    if v is not None:
                        daily.setdefault(sid, {})[r["DataDate"][:10]] = float(v)
                else:
                    v = sp.get("Instantaneous")
                    if v is not None:
                        hourly.setdefault(sid, {})[r["DataTime"]] = float(v)
    out = {"codis_dir": str(CODIS_DIR.relative_to(REPO))}
    for sid, (lat, lon) in CWA_STATIONS.items():
        cid = baro.nearest_cell(lat, lon, cells)
        o = {"era5_cell": cid}
        pc = P[cid]
        if sid in daily:
            s = pd.Series(daily[sid])
            s.index = pd.to_datetime(s.index)
            s = s[(s > 850) & (s < 1100)].sort_index()
            e = pc.groupby(pc.index.floor("D")).mean().reindex(s.index)
            ok = s.notna() & e.notna()
            a, b = s[ok], e[ok]
            off = float((a - b).mean())
            da, db = a.diff(), b.diff()
            okd = da.notna() & db.notna()
            o["daily"] = {"n_days": int(ok.sum()), "first": str(a.index.min().date()),
                          "last": str(a.index.max().date()), "mean_offset_hpa": off,
                          "rmse_after_offset_hpa": float(np.sqrt(((a - b - off) ** 2).mean())),
                          "corr": float(np.corrcoef(a, b)[0, 1]),
                          "corr_day_to_day_change": float(np.corrcoef(da[okd], db[okd])[0, 1]),
                          "rmse_day_to_day_change_hpa": float(np.sqrt(((da - db)[okd] ** 2)
                                                                      .mean()))}
        if sid in hourly:
            s = pd.Series(hourly[sid])
            # CODIS hourly DataTime is the end of the hour (01:00 = first hour), ERA5 is
            # instantaneous at the hour: compare at the same clock time
            s.index = pd.to_datetime(s.index)
            s = s[(s > 850) & (s < 1100)].sort_index()
            e = pc.reindex(s.index)
            ok = s.notna() & e.notna()
            a, b = s[ok], e[ok]
            wk = (a.index.to_series().diff() > pd.Timedelta(hours=6)).cumsum()
            res_w = []
            for _, idx in a.groupby(wk.values).groups.items():
                aa, bb = a.loc[idx], b.loc[idx]
                if len(aa) < 48:
                    continue
                res_w.append({"start": str(aa.index.min()), "n": int(len(aa)),
                              "offset": float((aa - bb).mean()),
                              "rmse_after_offset": float(np.sqrt(((aa - bb - (aa - bb).mean())
                                                                  ** 2).mean())),
                              "corr": float(np.corrcoef(aa, bb)[0, 1]),
                              "max_48h_fall_cwa": float((np.maximum.accumulate(aa.to_numpy())
                                                         - aa.to_numpy()).max()),
                              "max_48h_fall_era5": float((np.maximum.accumulate(bb.to_numpy())
                                                          - bb.to_numpy()).max())})
            W = pd.DataFrame(res_w)
            dha, dhb = a.diff(), b.diff()
            okh = dha.notna() & dhb.notna() & (a.index.to_series().diff() == pd.Timedelta(hours=1))
            o["hourly"] = {"n_hours": int(ok.sum()), "n_windows": int(len(W)),
                           "median_window_rmse_after_offset_hpa": float(W.rmse_after_offset
                                                                        .median()),
                           "p90_window_rmse_after_offset_hpa": float(W.rmse_after_offset
                                                                     .quantile(0.9)),
                           "median_window_corr": float(W["corr"].median()),
                           "corr_hourly_change": float(np.corrcoef(dha[okh], dhb[okh])[0, 1]),
                           "rmse_hourly_change_hpa": float(np.sqrt(((dha - dhb)[okh] ** 2)
                                                                   .mean())),
                           "fall_regression_era5_on_cwa_slope": float(np.polyfit(
                               W.max_48h_fall_cwa, W.max_48h_fall_era5, 1)[0]) if len(W) > 3
                           else None,
                           "largest_cwa_fall_window": W.sort_values("max_48h_fall_cwa")
                           .tail(1).to_dict("records")}
        out[sid] = o
    return out


# ----------------------------------------------------------------------------- rescore
def rescore(build: Path) -> dict:
    e = pd.read_csv(EVAL, dtype={"station_id": str})
    e["date"] = pd.to_datetime(e["date"])
    M = pd.read_parquet(build / "heads_monthly.parquet")
    M = M[(M.period == "post") & M.flag.isin([0, 1]) & M.c20]
    key = M.set_index(["sid", "date"])["value"]
    e["obs_l1"] = [key.get((s, d), np.nan)
                   for s, d in zip(e.station_id, e.date, strict=True)]
    preds = ["model_datum", "clim", "stat", "phys"]

    def ew(df, obs):
        df = df[df[obs].notna()]
        per = df.groupby("station_id").apply(
            lambda g: pd.Series({p: np.sqrt(np.mean((g[p] - g[obs]) ** 2)) for p in preds}),
            include_groups=False)
        r = per.mean()
        return {"months": int(len(df)), "wells": int(per.shape[0]),
                **{p: float(r[p]) for p in preds},
                "model_over_clim": float(r.model_datum / r.clim),
                "stat_over_clim": float(r.stat / r.clim)}
    orig = ew(e, "observed_m")
    l1 = ew(e, "obs_l1")
    same = ew(e[e.obs_l1.notna()], "observed_m")
    diff = (e.obs_l1 - e.observed_m).abs()
    return {"original": orig, "l1_monthly": l1, "original_on_l1_months": same,
            "ratio_change": l1["model_over_clim"] - orig["model_over_clim"],
            "months_changed_gt_0p25m": int((diff > 0.25).sum()),
            "wells_with_month_changed_gt_0p25m": int(e[diff > 0.25].station_id.nunique()),
            "months_dropped_by_l1": int((e.observed_m.notna() & e.obs_l1.isna()).sum()),
            "note": "secondary score only; the registered scorer keeps its registered QC"}


# ----------------------------------------------------------------------------- summary
def summary(work: Path, build: Path) -> dict:
    D = pd.read_parquet(build / "heads_daily.parquet", columns=["sid", "period", "flag",
                                                               "day_valid", "kinds"])
    M = pd.read_parquet(build / "heads_monthly.parquet")
    E = pd.read_csv(build / "events.csv", dtype={"sid": str})
    dec = pd.read_parquet(work / "decisions_stage1.parquet")
    B = pd.read_csv(build / "barometric_efficiency.csv", dtype={"sid": str})
    meta = pd.read_parquet(work / "station_meta_clean.parquet")
    out = {"daily_flag_counts": {per: g.flag.value_counts().sort_index().to_dict()
                                 for per, g in D.groupby("period")},
           "monthly_flag_counts": {per: g.flag.value_counts().sort_index().to_dict()
                                   for per, g in M.groupby("period")}}
    cov = {}
    for per, g in M.groupby("period"):
        rec = g.groupby("sid").apply(lambda x: (x.flag != 8).any(), include_groups=False)
        gg = g[g.sid.isin(rec[rec].index)]
        cov[per] = {"wells_with_record": int(rec.sum()),
                    "well_months": int(len(gg)),
                    "observed_flag01": int(gg.flag.isin([0, 1]).sum()),
                    "observed_flag01_c20": int((gg.flag.isin([0, 1]) & gg.c20).sum()),
                    "flag3": int((gg.flag == 3).sum()), "flag9": int((gg.flag == 9).sum()),
                    "flag8": int((gg.flag == 8).sum()),
                    "sigma_obs_median_flag01": float(gg[gg.flag.isin([0, 1])].sigma_obs
                                                     .median())}
    out["monthly_coverage"] = cov
    ev = {}
    for per in ("pre", "post"):
        d = dec[dec.period == per]
        e = E[E.period == per]
        ep = d[d.kind == "offset_episode"]
        st = d[d.kind == "step"]
        ev[per] = {
            "offset_episodes": int(len(ep)), "offset_episode_wells": int(ep.sid.nunique()),
            "offset_episodes_ge_15d": int((ep.duration_d >= 15).sum()),
            "offset_episodes_by_flag": ep.flag.value_counts().to_dict(),
            "dropout_episodes": int((d.kind == "dropout").sum()),
            "dropout_episode_wells": int(d[d.kind == "dropout"].sid.nunique()),
            "steps_ge_3m": int(len(st)), "step_wells": int(st.sid.nunique()),
            "steps_by_flag": st.flag.value_counts().to_dict(),
            "steps_by_side_rule": st.get("side_rule", pd.Series(dtype=object))
            .value_counts().to_dict(),
            "neighbour_verdicts": d.neighbour_verdict.value_counts().to_dict(),
            "backing": d.backing.value_counts().to_dict(),
            "spike_jumps": int((e.kind == "jump_spike").sum()),
            "spike_wells": int(e[e.kind == "jump_spike"].sid.nunique()),
            "unpaired_offsets_lt3m": int((e.kind == "unpaired_offset_lt3m").sum()),
            "floor_dropout_runs": int((e.kind == "dropout_flat").sum()),
            "floor_dropout_wells": int(e[e.kind.isin(["dropout_flat"])].sid.nunique()),
            "dropout_wells_any": int(e[e.kind.isin(["dropout_flat", "dropout",
                                                    "below_well_bottom"])].sid.nunique()),
            "sentinel_runs": int((e.kind == "sentinel_block").sum()),
            "flat_runs": int((e.kind == "flat_run").sum()),
            "mad_outlier_runs": int((e.kind == "mad_outlier").sum()),
            "far_outlier_runs": int((e.kind == "far_outlier").sum()),
            "below_bottom_runs": int((e.kind == "below_well_bottom").sum()),
            "daily_departure_days": int((e.kind == "daily_departure").sum()),
            "steps_carried": int((e.kind == "step_carried").sum())}
    out["events"] = ev
    out["barometric"] = {"status_counts": B.status.value_counts().to_dict(),
                         "accepted": int(B.accepted.sum()),
                         "A_K_accepted_quantiles": B[B.accepted].A_K.quantile(
                             [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict(),
                         "sensor_type_hint_accepted": B[B.accepted].sensor_type_hint
                         .value_counts().to_dict()}
    out["metadata"] = {"stations": int(len(meta)),
                       "xy_class_raw": meta.xy_class_raw.value_counts().to_dict(),
                       "xy_flag": meta.xy_flag.value_counts().to_dict(),
                       "label_repaired": int((meta.label_flag == 1).sum()),
                       "layer_flag": meta.layer_flag.value_counts().to_dict(),
                       "ground_height_zero": int(meta.ground_height_missing_code.sum()),
                       "swapped_with_hist_data": None}
    return out


def main():
    work, build, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    P = baro.load_pressure()
    out = {"audit_agreement": audit_agreement(work)}
    print("audit done", flush=True)
    out["real_events"] = real_events(build)
    out["o1_coherence"] = o1_check(build)
    out["pressure_drops"] = pressure_drops(work, P)
    print("baro checks done", flush=True)
    out["cwa_pressure_check"] = cwa_check(P)
    out["rescore_frozen_challenge"] = rescore(build)
    out["summary"] = summary(work, build)
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    brief = {"audit": {k: v["agreement"] for k, v in out["audit_agreement"].items()},
             "real_events_flagged": out["real_events"]["n_flagged"],
             "o1": out["o1_coherence"], "drops": {k: v.get(
                 "n_offset_episodes_on_fall_ge_20hpa") for k, v in out["pressure_drops"].items()
                 if k != "power_pressure_record"},
             "rescore_ratio_change": out["rescore_frozen_challenge"]["ratio_change"]}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
