"""H1a v2 validation (round 2), after the v2 stage 3 (python -I, CPU slot).

Usage: s4_validate_v2.py WORKDIR BUILD_DIR REF_WORKDIR V1_PRODUCT_DIR OUT_JSON

REF_WORKDIR is the v1 run's work directory (stage-1 identity check only); every v1 number
comes from the published v1 product (V1_PRODUCT_DIR).

Conformance checks of the v2 gate (pre-registered in H1a.qc_acceptance.v2 with the plan's
thresholds; their units were looked at in v1, so they are requirements, not fresh evidence):
* C3 known real events (``config.REAL_EVENTS``): no flag-9 day, no artefact event (v1 kinds
  plus ``excursion`` and ``jump_edge``) and no correction in the three windows;
* C4 audit agreement: share of the audit's 2012-22 raw-record events matched by a stage-1
  jump of the same well within +-1 day (stage 1 is unchanged from v1; the identity with the
  v1 run is checked);
* C5 barometric O1 coherence: paired median over wells with an accepted barometric correction
  (unchanged from v1; recomputed);
* C6 pressure drops: flagged v2 objects with an edge (offset episodes, dropouts, accepted
  steps, excursion and jump-edge runs) of 2012-22 with a pressure fall >= 20 hPa within
  +-24 h of either edge jump, at the well's ERA5 cell. v2 excludes typhoon-time jumps from
  episode and step edges, so this holds by construction (disclosed).
Reported, not gated:
* chance coincidences: the same pressure-fall count for the v1 offset episodes and for the
  v2 objects with their jump times shifted by whole years (1..10, cyclic within 2012-22),
  i.e. the count expected without any causal link;
* post-origin values of C3-C6 analogues, the CWA station pressure check (unchanged inputs),
  the frozen-challenge rescore on the v2 monthly means split into the effect of dropped
  months and of changed values (and the same split for v1);
* real-data false-flag proxies: v1 corrections and flags withdrawn in v2, the v2 jump
  catalogue (persistent / not persistent / typhoon-time), flag-9 shares per well, and the
  wells whose rate of persistent jumps marks a pumping-cycle record;
* v1 -> v2 transitions of flags and C20 months, corrected well-months.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import baro  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402
from h1_heads_qc import decide_v2 as DV  # noqa: E402
from h1_heads_qc import s4_validate as V1  # noqa: E402

ARTEFACT_KINDS_V2 = set(C.ARTEFACT_KINDS) | {"excursion", "jump_edge"}
EDGE_KINDS = ("offset_episode", "dropout", "step", "excursion", "jump_edge")
TYPHOON_EPISODES_V1 = [("07080132", "2015-08-07 08:00:00"), ("07080132", "2016-09-26 21:00:00"),
                       ("07240223", "2016-09-26 18:00:00")]


# ----------------------------------------------------------------------------- C3
def real_events_v2(build: Path) -> dict:
    D = pd.read_parquet(build / "heads_daily.parquet")
    E = pd.read_csv(build / "events.csv", dtype={"sid": str}, low_memory=False)
    E["t0"] = pd.to_datetime(E["t_start"])
    E["t1_"] = pd.to_datetime(E["t_end"].where(E["t_end"].astype(str) != "", E["t_start"]),
                              errors="coerce").fillna(E["t0"])
    res = []
    for ev in C.REAL_EVENTS:
        a, b = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]) + pd.Timedelta(days=1)
        d = D[(D.sid == ev["sid"]) & (D.date >= a) & (D.date < b)]
        e = E[(E.sid == ev["sid"]) & (E.kind.isin(ARTEFACT_KINDS_V2)) & (E.t0 < b) & (E.t1_ >= a)]
        corr = d[(d.corr_m.abs() > 0)]
        flagged = bool((d.flag == 9).any() or len(e) or len(corr))
        jw = E[(E.sid == ev["sid"]) & E.kind.str.startswith("jump") & (E.t0 >= a) & (E.t0 < b)]
        res.append(ev | {"n_days_in_record": int(len(d)),
                         "n_days_flag9": int((d.flag == 9).sum()),
                         "n_artefact_events": int(len(e)),
                         "artefact_events": e[["kind", "t_start", "t_end", "flag"]]
                         .to_dict("records"),
                         "jumps_in_window": jw[["kind", "t_start", "j1"]].to_dict("records"),
                         "n_days_corrected": int(len(corr)),
                         "daily_values": d[["date", "value", "flag", "n_hours"]]
                         .astype({"date": str}).to_dict("records"),
                         "flagged": flagged})
    return {"events": res, "n_flagged": int(sum(r["flagged"] for r in res)),
            "n_events": len(res),
            "n_untestable": int(sum(1 for r in res if r["n_days_in_record"] == 0))}


# ----------------------------------------------------------------------------- v1 decisions
def v1_decisions(v1: Path) -> pd.DataFrame:
    """The v1 episode, dropout and step decisions from the published v1 product's events.csv
    (sid, period, kind, flag, t1, t2 in seconds)."""
    E = pd.read_csv(v1 / "events.csv", dtype={"sid": str}, low_memory=False)
    E = E[E.kind.isin(["offset_episode", "dropout", "step"])].copy()

    def sec(x):
        return pd.to_datetime(x).astype("datetime64[s]").astype("int64")
    E["t1"] = sec(E["t_jump1"])
    t2 = E["t_jump2"].where(E["t_jump2"].notna() & (E["t_jump2"].astype(str) != ""),
                            E["t_jump1"])
    E["t2"] = sec(t2)
    return E[["sid", "period", "kind", "flag", "t1", "t2", "backing"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- C6
def _edge_times(r) -> list[int]:
    if r["kind"] in ("offset_episode", "dropout"):
        return [int(r["t1"]), int(r["t2"])]
    return [int(r["t1"])]


def pressure_drops_v2(work: Path, P: pd.DataFrame) -> dict:
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    pa = {c: DV.pressure_arrays(P[c]) for c in P.columns}
    out = {}
    for per in ("pre", "post"):
        g = d2[(d2.period == per) & d2.kind.isin(EDGE_KINDS)]
        g = g.drop_duplicates(["sid", "kind", "t1", "t2"])
        rows = []
        for _, r in g.iterrows():
            cid = meta.loc[r.sid, "era5_cell"] if r.sid in meta.index else None
            if cid not in pa:
                continue
            falls = [DV.max_fall(pa[cid], t) for t in _edge_times(r)]
            rows.append({"sid": r.sid, "kind": r.kind, "t1": str(pd.Timestamp(int(r.t1),
                                                                              unit="s")),
                         "flag": r.flag, "max_fall_hpa": float(np.nanmax(falls))
                         if np.isfinite(falls).any() else np.nan})
        R = pd.DataFrame(rows)
        hit = R[R.max_fall_hpa >= C.PRESSURE_DROP_HPA] if len(R) else R
        out[per] = {"n_objects": int(len(R)),
                    "n_objects_by_kind": R.kind.value_counts().to_dict() if len(R) else {},
                    "n_on_fall_ge_20hpa": int(len(hit)),
                    "on_fall_ge_20hpa": hit.to_dict("records") if len(hit) else [],
                    "n_offset_episodes_on_fall_ge_20hpa": int(
                        (hit.kind == "offset_episode").sum()) if len(hit) else 0,
                    "max_fall_hpa": float(R.max_fall_hpa.max()) if len(R) else None}
        j = d2[(d2.period == per) & (d2.kind == "jump")]
        out[per]["typhoon_time_jumps"] = j[j.v2_status == "typhoon_time"][
            ["sid", "t1", "j1", "pressure_fall_hpa", "persistent"]].assign(
            t1=lambda x: pd.to_datetime(x.t1, unit="s").astype(str)).to_dict("records")
    return out


def chance_coincidences(work: Path, v1dir: Path, P: pd.DataFrame) -> dict:
    """Pressure-fall coincidences of 2012-22 objects with their times shifted by k whole
    years (k = 1..10, cyclic inside 2012-01-01..2022-12-31): the count expected by chance."""
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    pa = {c: DV.pressure_arrays(P[c]) for c in P.columns}
    lo = int(pd.Timestamp("2012-01-01").value // 10**9)
    span = int(pd.Timestamp("2023-01-01").value // 10**9) - lo
    year = 365 * 86400

    def count(objs, k):
        n = 0
        for sid, times in objs:
            cid = meta.loc[sid, "era5_cell"] if sid in meta.index else None
            if cid not in pa:
                continue
            tt = [lo + ((t - lo + k * year) % span) for t in times]
            if any(DV.max_fall(pa[cid], t) >= C.PRESSURE_DROP_HPA for t in tt):
                n += 1
        return n
    v1 = v1_decisions(v1dir)
    v1 = v1[(v1.period == "pre") & (v1.kind == "offset_episode")]
    o1 = [(r.sid, [int(r.t1), int(r.t2)]) for _, r in v1.iterrows()]
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    d2 = d2[(d2.period == "pre") & d2.kind.isin(EDGE_KINDS)].drop_duplicates(
        ["sid", "kind", "t1", "t2"])
    o2 = [(r.sid, _edge_times(r)) for _, r in d2.iterrows()]
    c1 = [count(o1, k) for k in range(1, 11)]
    c2 = [count(o2, k) for k in range(1, 11)]
    return {"v1_offset_episodes": {"n": len(o1), "observed": count(o1, 0),
                                   "shifted_counts_k1_10": c1, "shifted_mean": float(np.mean(c1))},
            "v2_edge_objects": {"n": len(o2), "observed": count(o2, 0),
                                "shifted_counts_k1_10": c2, "shifted_mean": float(np.mean(c2))},
            "note": "shifting by whole years keeps the season of each jump; a shifted count is "
                    "what a detector with no causal link to the pressure record would show"}


# ----------------------------------------------------------------------------- identity
def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def stage1_identity(work: Path, ref: Path) -> dict:
    """Compare the stage-1 catalogues and barometric segments of this run with the v1 run's
    (REF_WORKDIR): equal digests mean the jump catalogue, the barometric fits and the O1
    inputs are those of v1."""
    out = {"wells": 0, "jumps_equal": True, "episodes_equal": True, "steps_equal": True,
           "be_equal": True, "o1_equal": True, "differing_wells": []}
    for p in sorted((work / "stage1").glob("*.pkl")):
        q = ref / "stage1" / p.name
        a = pickle.loads(p.read_bytes())
        b = pickle.loads(q.read_bytes()) if q.exists() else {}
        out["wells"] += 1
        bad = []
        for part in ("pre_cat", "post_cat"):
            for key, flag in (("jumps", "jumps_equal"), ("episodes", "episodes_equal"),
                              ("steps", "steps_equal")):
                if _digest(a.get(part, {}).get(key)) != _digest(b.get(part, {}).get(key)):
                    out[flag] = False
                    bad.append(f"{part}.{key}")
        if _digest(a.get("be")) != _digest(b.get("be")):
            out["be_equal"] = False
            bad.append("be")
        if _digest(a.get("o1_pre")) != _digest(b.get("o1_pre")):
            out["o1_equal"] = False
            bad.append("o1_pre")
        if bad:
            out["differing_wells"].append({"sid": a.get("sid"), "parts": bad})
    out["ref_workdir"] = str(ref)
    return out


# ----------------------------------------------------------------------------- rescore
def rescore_split(build: Path) -> dict:
    r = V1.rescore(build)
    o, s, l1 = (r["original"]["model_over_clim"], r["original_on_l1_months"]["model_over_clim"],
                r["l1_monthly"]["model_over_clim"])
    r["split"] = {"dropping_months_effect": s - o, "value_change_effect": l1 - s,
                  "total": l1 - o}
    return r


# ----------------------------------------------------------------------------- proxies
def proxies(work: Path, v1dir: Path, build: Path) -> dict:
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    d1 = v1_decisions(v1dir)
    D = pd.read_parquet(build / "heads_daily.parquet", columns=["sid", "period", "flag"])
    out = {}
    for per in ("pre", "post"):
        g2 = d2[d2.period == per]
        g1 = d1[d1.period == per]
        j = g2[g2.kind == "jump"]
        c1 = g1[g1.kind.isin(["offset_episode", "step"]) & g1.flag.isin([1, 3])]
        c2 = g2[(g2.kind == "step") & g2.flag.isin([1, 3])]
        keep = c1.merge(c2[["sid", "t1"]], on=["sid", "t1"], how="inner")
        out[per] = {
            "jump_catalogue": {f"{a}|{b}": int(n) for (a, b), n in
                               j.groupby(["cls1", "v2_status"]).size().items()},
            "v1_offset_episodes": int((g1.kind == "offset_episode").sum()),
            "v2_offset_episodes": int((g2.kind == "offset_episode").sum()),
            "v2_offset_episode_wells": int(g2[g2.kind == "offset_episode"].sid.nunique()),
            "v2_offset_episodes_ge_15d": int(((g2.kind == "offset_episode")
                                              & (g2.duration_d >= 15)).sum()),
            "v2_offset_episodes_unequal_edges": int((g2.v2_status == "unequal_edges").sum()),
            "v1_dropouts": int((g1.kind == "dropout").sum()),
            "v2_dropouts": int((g2.kind == "dropout").sum()),
            "v1_steps": int((g1.kind == "step").sum()),
            "v1_step_wells": int(g1[g1.kind == "step"].sid.nunique()),
            "v2_step_candidates": int(g2[g2.kind.isin(["step", "excursion", "jump_edge"])]
                                      .drop_duplicates(["sid", "t1"]).shape[0]
                                      + 2 * int((g2.v2_status == "unequal_edges").sum())),
            "v2_steps_accepted": int((g2.kind == "step").sum()),
            "v2_step_wells": int(g2[g2.kind == "step"].sid.nunique()),
            "v2_rejected_candidates_runs": g2[g2.kind.isin(["excursion", "jump_edge"])]
            .kind.value_counts().to_dict(),
            "v2_steps_by_flag": g2[g2.kind == "step"].flag.value_counts().to_dict(),
            "v2_steps_backing": g2[g2.kind == "step"].backing.fillna("").str.split(":")
            .str[0].value_counts().to_dict(),
            "v2_steps_neighbour_verdicts": g2[g2.kind == "step"].neighbour_verdict
            .fillna("none").value_counts().to_dict(),
            "v1_corrections": int(len(c1)),
            "v1_corrections_kept_as_corrected_steps_in_v2": int(len(keep)),
            "v2_corrections": int(len(c2))}
        dd = D[D.period == per]
        s9 = dd.groupby("sid").flag.apply(lambda f: float((f == 9).mean()))
        out[per]["flag9_share_by_well_top15"] = s9.sort_values(ascending=False).head(15) \
            .round(4).to_dict()
        out[per]["wells_with_flag9_share_gt_0p25"] = int((s9 > 0.25).sum())
    # pumping-cycle wells: persistent jumps per year of pre-origin record
    jj = d2[(d2.kind == "jump") & (d2.period == "pre")]
    Dp = D[D.period == "pre"]
    years = Dp[Dp.flag != 8].groupby("sid").size() / 365.25
    rate = (jj[jj.v2_status == "persistent"].groupby("sid").size() / years).dropna()
    allj = (jj.groupby("sid").size() / years).dropna()
    s9 = Dp.groupby("sid").flag.apply(lambda f: float((f == 9).mean()))
    tab = pd.DataFrame({"persistent_jumps_per_year": rate, "jumps_ge_1m_per_year": allj,
                        "flag9_share_pre": s9}).fillna(0.0)
    tab = tab[tab.jumps_ge_1m_per_year > 0].sort_values("persistent_jumps_per_year",
                                                       ascending=False)
    out["pumping_cycle_table_pre_top20"] = tab.head(20).round(3).reset_index() \
        .rename(columns={"index": "sid"}).to_dict("records")
    out["pumping_cycle_wells_ge_2_persistent_per_year"] = int(
        (tab.persistent_jumps_per_year >= 2).sum())
    return out


def corrected_months(build: Path) -> dict:
    M = pd.read_parquet(build / "heads_monthly.parquet")
    out = {}
    for per in ("pre", "post"):
        g = M[(M.period == per) & M.flag.isin([1, 3]) & (M.corr_m.abs() > 0)]
        out[per] = {"well_months": int(len(g)), "wells": int(g.sid.nunique()),
                    "median_abs_corr_m": float(g.corr_m.abs().median()) if len(g) else None,
                    "max_abs_corr_m": float(g.corr_m.abs().max()) if len(g) else None,
                    "by_flag": g.flag.value_counts().to_dict()}
    return out


def v1_v2_months(build: Path, v1: Path) -> dict:
    M2 = pd.read_parquet(build / "heads_monthly.parquet")
    M1 = pd.read_parquet(v1 / "heads_monthly.parquet")
    k = ["sid", "date"]
    m = M1[k + ["period", "flag", "c20", "value", "corr_m"]].merge(
        M2[k + ["flag", "c20", "value", "corr_m"]], on=k, suffixes=("_v1", "_v2"))
    out = {}
    for per, g in m.groupby("period"):
        o1 = g.flag_v1.isin([0, 1]) & g.c20_v1
        o2 = g.flag_v2.isin([0, 1]) & g.c20_v2
        both = o1 & o2
        dv = (g.value_v2 - g.value_v1).abs()
        out[per] = {"c20_flag01_v1": int(o1.sum()), "c20_flag01_v2": int(o2.sum()),
                    "lost_in_v2": int((o1 & ~o2).sum()), "gained_in_v2": int((~o1 & o2).sum()),
                    "lost_wells": int(g[o1 & ~o2].sid.nunique()),
                    "gained_wells": int(g[~o1 & o2].sid.nunique()),
                    "kept_months_value_changed_gt_0p25m": int((both & (dv > 0.25)).sum()),
                    "kept_months_value_changed_wells": int(g[both & (dv > 0.25)].sid
                                                           .nunique()),
                    "flag_transition": {f"{a}->{b}": int(n) for (a, b), n in
                                        pd.crosstab(g.flag_v1, g.flag_v2).stack().items()
                                        if n > 0}}
    return out


def summary_v2(work: Path, build: Path) -> dict:
    D = pd.read_parquet(build / "heads_daily.parquet", columns=["sid", "period", "flag",
                                                               "day_valid", "n_suspect"])
    M = pd.read_parquet(build / "heads_monthly.parquet")
    E = pd.read_csv(build / "events.csv", dtype={"sid": str}, low_memory=False)
    B = pd.read_csv(build / "barometric_efficiency.csv", dtype={"sid": str})
    out = {"daily_flag_counts": {per: {int(k): int(v) for k, v in
                                       g.flag.value_counts().sort_index().items()}
                                 for per, g in D.groupby("period")},
           "daily_share_flag9": {per: float((g.flag == 9).mean())
                                 for per, g in D.groupby("period")},
           "daily_share_any_flag9_sample": {per: float((g.n_suspect > 0).mean())
                                            for per, g in D.groupby("period")},
           "monthly_flag_counts": {per: {int(k): int(v) for k, v in
                                         g.flag.value_counts().sort_index().items()}
                                   for per, g in M.groupby("period")}}
    cov = {}
    for per, g in M.groupby("period"):
        rec = g.groupby("sid").apply(lambda x: (x.flag != 8).any(), include_groups=False)
        gg = g[g.sid.isin(rec[rec].index)]
        cov[per] = {"wells_with_record": int(rec.sum()), "well_months": int(len(gg)),
                    "observed_flag01": int(gg.flag.isin([0, 1]).sum()),
                    "observed_flag01_c20": int((gg.flag.isin([0, 1]) & gg.c20).sum()),
                    "flag3": int((gg.flag == 3).sum()), "flag9": int((gg.flag == 9).sum()),
                    "flag8": int((gg.flag == 8).sum()),
                    "sigma_obs_median_flag01": float(gg[gg.flag.isin([0, 1])].sigma_obs
                                                     .median())}
    out["monthly_coverage"] = cov
    ev = {}
    for per in ("pre", "post"):
        e = E[E.period == per]
        ev[per] = {"by_kind": e.kind.value_counts().to_dict(),
                   "wells_by_kind": e.groupby("kind").sid.nunique().to_dict(),
                   "steps_carried_by_flag": e[e.kind == "step_carried"].flag.value_counts()
                   .to_dict()}
    out["events"] = ev
    out["barometric"] = {"status_counts": B.status.value_counts().to_dict(),
                         "accepted_rows": int(B.accepted.sum()),
                         "accepted_wells": int(B[B.accepted].sid.nunique()),
                         "A_K_accepted_quantiles": B[B.accepted].A_K.quantile(
                             [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()}
    return out


def typhoon_episodes(work: Path) -> list[dict]:
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    out = []
    for sid, t in TYPHOON_EPISODES_V1:
        ts_ = int(pd.Timestamp(t).value // 10**9)
        g = d2[(d2.sid == sid) & ((d2.t1 - ts_).abs() <= 3600)]
        out.append({"sid": sid, "jump": t, "v2_rows": g[["kind", "v2_status", "flag", "t1",
                                                          "t2", "j1", "pressure_fall_hpa"]]
                    .astype(str).to_dict("records")})
    return out


def main():
    work, build, ref, v1 = (Path(sys.argv[k]) for k in (1, 2, 3, 4))
    out_json = Path(sys.argv[5])
    P = baro.load_pressure()
    out = {"conformance": {}}
    out["conformance"]["real_events"] = real_events_v2(build)
    print("real events:", out["conformance"]["real_events"]["n_flagged"], flush=True)
    out["conformance"]["audit_agreement"] = V1.audit_agreement(work)
    out["conformance"]["o1_coherence"] = V1.o1_check(build)
    out["conformance"]["pressure_drops"] = pressure_drops_v2(work, P)
    print("conformance done", flush=True)
    out["identity_with_v1_run"] = stage1_identity(work, ref)
    out["chance_coincidences"] = chance_coincidences(work, v1, P)
    out["cwa_pressure_check"] = V1.cwa_check(P)
    out["rescore_frozen_challenge"] = rescore_split(build)
    out["rescore_v1_split"] = rescore_split(v1)
    out["proxies"] = proxies(work, v1, build)
    out["corrected_months_v2"] = corrected_months(build)
    out["corrected_months_v1"] = corrected_months(v1)
    out["v1_v2_months"] = v1_v2_months(build, v1)
    out["summary"] = summary_v2(work, build)
    out["typhoon_episodes_v1_in_v2"] = typhoon_episodes(work)
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    cf = out["conformance"]
    brief = {"real_events_flagged": cf["real_events"]["n_flagged"],
             "audit": {k: v["agreement"] for k, v in cf["audit_agreement"].items()},
             "o1_pre": cf["o1_coherence"]["pre"]["corrected_wells"],
             "drops": {k: v["n_on_fall_ge_20hpa"] for k, v in cf["pressure_drops"].items()},
             "identity": {k: v for k, v in out["identity_with_v1_run"].items()
                          if k != "differing_wells"},
             "chance": out["chance_coincidences"],
             "rescore": out["rescore_frozen_challenge"]["split"],
             "corr": out["corrected_months_v2"]}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
