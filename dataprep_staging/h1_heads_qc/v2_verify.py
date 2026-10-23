"""H1a repair round (2026-10-13): after-the-fact verification of the v2 build, run after every
registered score (python -I, CPU slot). Reported only: nothing here gates, tunes or changes
the product.

Answers the round-1 verifier's findings on the v2 build:

1. **Corrections** (critical finding). Every non-zero step correction of the build (accepted
   steps at flag 1/3 and carried steps) gets an independent level check that does not use
   the decision layer's estimator: plain medians of the hourly medians of the uncorrected
   series (raw minus barometric loading; samples the base screens mark invalid left out)
   within 10 days on each side of the jump. A correction passes when the shift has the
   jump's sign and at least half its size (the verifier's 0.5|j| check). Each corrected span
   is also searched for record gaps (> 1 day) across which the level moves back by at least
   half the jump (a possible undetected return inside the corrected span). The two v1
   examples of the verifier (07230332 2012-01..2016-11, 07260122 2018-09..2026-09) and the
   09190132 record are reported in v1 and v2.
2. **False flags on real data** (major finding: the injection count excludes base-run
   detections). Proxies: flagged offset episodes and dropouts whose interior median does not
   differ from the median of the 10 days before and after by at least half the mean jump
   (same estimator); rejected-step runs; flag-9 day shares; the same level check applied to
   v1's flagged and corrected objects for comparison.
3. **Barometric plausibility** (minor): accepted epochs with A_K < -1 or A_K > 0, the size of
   their corrections, O1 coherence over all wells (from the validation file).
4. **Injection diagnostics** (minor): misses of the registered r = 10..19 run whose effective
   injected duration exceeds the 365-day pairing window (a harness artefact), recall without
   them (reported; the registered recall stands), kinds found at the missed jumps.
5. **Rescore attribution** (minor): dropped-month and changed-value effects for v1 and v2.

Usage: v2_verify.py WORKDIR BUILD_DIR V1_PRODUCT_DIR INJECTION_JSON OUT_JSON [VALIDATION_JSON]
(VALIDATION_JSON defaults to validation_v2.json next to OUT_JSON.) The same script verifies
the reviewed v3 build (WORKDIR = the v3 work directory).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import levelcheck as LC  # noqa: E402

DAY = 86400
WIN = 10 * DAY
FRAC = 0.5
EVID_V2 = Path("results/dataprep/reports/H1a_evidence/v2")


# ----------------------------------------------------------------------------- series
class Well:
    """Uncorrected (raw minus barometric loading) valid samples of one well, both periods."""

    def __init__(self, res: dict):
        ts, vs = [], []
        for part in ("pre", "post"):
            if part in res and len(res[part]["t"]):
                t, v, ok = LC.part_arrays(res[part])
                ts.append(t[ok])
                vs.append(v[ok])
        if ts:
            t = np.concatenate(ts)
            v = np.concatenate(vs)
            o = np.argsort(t, kind="stable")
            self.t, self.v = t[o], v[o]
        else:
            self.t, self.v = np.zeros(0, np.int64), np.zeros(0)

    def med(self, a: int, b: int) -> tuple[float, int]:
        """Median of the hourly medians in [a, b) and their number."""
        i0, i1 = np.searchsorted(self.t, a, "left"), np.searchsorted(self.t, b, "left")
        _, hv = LC.hourly_medians(self.t[i0:i1], self.v[i0:i1])
        return (float(np.median(hv)) if len(hv) else np.nan), int(len(hv))


def level_check(W: Well, t0: int, j: float, hi_limit: int | None = None) -> dict:
    """Plain 10-day medians on each side of a jump at t0 (no trend, no truncation)."""
    mb, nb = W.med(t0 - WIN, t0)
    b = t0 + WIN if hi_limit is None else min(t0 + WIN, hi_limit)
    ma, na = W.med(t0, b)
    d = ma - mb
    ok = bool(np.isfinite(d) and nb >= 12 and na >= 12 and np.sign(d) == np.sign(j)
              and abs(d) >= FRAC * abs(j))
    status = "undetermined" if (nb < 12 or na < 12) else ("pass" if ok else "fail")
    return {"median_before": mb, "median_after": ma, "shift": d, "n_hours_before": nb,
            "n_hours_after": na, "ratio": d / j if j else np.nan, "status": status}


def gap_returns(W: Well, lo: int, hi: int, j: float) -> list[dict]:
    """Record gaps > 1 day inside [lo, hi] across which the level moves back by >= 0.5|j|."""
    i0, i1 = np.searchsorted(W.t, lo, "left"), np.searchsorted(W.t, hi, "right")
    t = W.t[i0:i1]
    if len(t) < 2:
        return []
    gi = np.where(np.diff(t) > DAY)[0]
    out = []
    for g in gi:
        a, b = int(t[g]), int(t[g + 1])
        mb, nb = W.med(a - WIN, a + 1)
        ma, na = W.med(b, b + WIN)
        d = ma - mb
        if nb >= 12 and na >= 12 and np.sign(d) == -np.sign(j) and abs(d) >= FRAC * abs(j):
            out.append({"gap_start": str(pd.Timestamp(a, unit="s")),
                        "gap_end": str(pd.Timestamp(b, unit="s")), "shift": d})
    return out


def episode_check(W: Well, t1: int, t2: int, j1: float, j2: float) -> dict:
    """Interior median vs the median of the 10 days before t1 and after t2."""
    mi, ni = W.med(t1, t2)
    mb, nb = W.med(t1 - WIN, t1)
    ma, na = W.med(t2, t2 + WIN)
    out_ = [x for x, n in ((mb, nb), (ma, na)) if n >= 12]
    mo = float(np.mean(out_)) if out_ else np.nan
    jm = 0.5 * (abs(j1) + abs(j2)) if np.isfinite(j2) else abs(j1)
    d = mi - mo
    if ni < 12 or not out_:
        st = "undetermined"
    else:
        st = "pass" if (np.sign(d) == np.sign(j1) and abs(d) >= FRAC * jm) else "fail"
    return {"interior_median": mi, "outside_median": mo, "shift": d, "status": st}


# ----------------------------------------------------------------------------- inputs
def load_wells(work: Path, sids) -> dict:
    out = {}
    for s in sorted(set(sids)):
        p = work / "stage1" / f"{s}.pkl"
        if p.exists():
            out[s] = Well(pickle.loads(p.read_bytes()))
    return out


def sec(x) -> int:
    return int(pd.Timestamp(x).value // 10**9)


def v2_corrections(work: Path, build: Path) -> pd.DataFrame:
    """Every non-zero step correction: accepted steps at flag 1/3 and carried steps."""
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    st = d2[(d2.kind == "step") & d2.flag.isin([1, 3])].copy()
    rows = []
    for _, r in st.iterrows():
        rows.append({"sid": r.sid, "period": r.period, "kind": "step", "t1": int(r.t1),
                     "j": float(r.j1), "flag": int(r.flag), "side": r.side,
                     "side_rule": r.side_rule, "t_lo": int(r.t_lo), "t_hi": int(r.t_hi),
                     "to_end": bool(r.to_end), "backing": r.backing,
                     "neighbour_verdict": r.get("neighbour_verdict")})
    E = pd.read_csv(build / "events.csv", dtype={"sid": str}, low_memory=False)
    ca = E[(E.kind == "step_carried") & E.flag.isin([1, 3])]
    for _, r in ca.iterrows():
        t_end = r.t_end if isinstance(r.t_end, str) and r.t_end else "2026-10-01"
        rows.append({"sid": r.sid, "period": "post", "kind": "step_carried",
                     "t1": None, "j": float(r.j1), "flag": int(r.flag), "side": "after",
                     "side_rule": "carried", "t_lo": sec(r.t_start), "t_hi": sec(t_end),
                     "to_end": not (isinstance(r.t_end, str) and r.t_end),
                     "backing": r.backing, "neighbour_verdict": None})
    return pd.DataFrame(rows)


def v1_objects(v1: Path) -> pd.DataFrame:
    E = pd.read_csv(v1 / "events.csv", dtype={"sid": str}, low_memory=False)
    E = E[E.kind.isin(["offset_episode", "dropout", "step"])].copy()
    E["t1"] = pd.to_datetime(E["t_jump1"]).astype("datetime64[s]").astype("int64")
    t2 = E["t_jump2"].where(E["t_jump2"].notna() & (E["t_jump2"].astype(str) != ""),
                            E["t_jump1"])
    E["t2"] = pd.to_datetime(t2).astype("datetime64[s]").astype("int64")
    return E


# ----------------------------------------------------------------------------- sections
def check_corrections(C: pd.DataFrame, wells: dict) -> dict:
    recs = []
    for _, r in C.iterrows():
        W = wells.get(r.sid)
        rec = r.to_dict()
        rec["span_days"] = (r.t_hi - r.t_lo) / DAY
        rec["t_lo"] = str(pd.Timestamp(int(r.t_lo), unit="s"))
        rec["t_hi"] = str(pd.Timestamp(int(r.t_hi), unit="s"))
        if W is None:
            rec["level_check"] = {"status": "no_series"}
            recs.append(rec)
            continue
        if r.kind == "step":
            rec["t1"] = str(pd.Timestamp(int(r.t1), unit="s"))
            rec["level_check"] = level_check(W, int(r.t1), float(r.j))
            lo, hi = sec(rec["t_lo"]), sec(rec["t_hi"])
        else:
            rec["level_check"] = {"status": "carried (checked at its pre-origin step)"}
            lo, hi = sec(rec["t_lo"]), sec(rec["t_hi"])
        rec["gap_returns_in_span"] = gap_returns(W, lo, hi, float(r.j))
        recs.append(rec)
    steps = [x for x in recs if x["kind"] == "step"]
    n_pass = sum(1 for x in steps if x["level_check"]["status"] == "pass")
    return {"n_corrections": len(steps), "n_pass": n_pass,
            "n_fail": sum(1 for x in steps if x["level_check"]["status"] == "fail"),
            "n_undetermined": sum(1 for x in steps
                                  if x["level_check"]["status"] == "undetermined"),
            "n_carried": len(recs) - len(steps),
            "n_with_gap_return_in_span": sum(1 for x in recs if x["gap_returns_in_span"]),
            "corrections": recs}


def sibling_check(C: pd.DataFrame, work: Path, build: Path) -> dict:
    """Co-located check of every step correction: the daily difference well minus nested
    sibling (same site) in the W days before and after the jump (W = 90 and 365); a datum
    artefact shifts the difference by about j, a well-specific real event does not. A
    correction counts as sibling-confirmed when the median over siblings (>= 30 valid days on
    each side, W = 365) of delta / j is at least 0.5."""
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    D = pd.read_parquet(build / "heads_daily.parquet",
                        columns=["sid", "date", "value_raw", "baro_m", "flag", "day_valid"])
    D = D[D.day_valid & (D.flag != 8)]
    U = D.assign(u=D.value_raw - D.baro_m).pivot(index="date", columns="sid", values="u")
    recs = []
    for _, r in C[C.kind == "step"].iterrows():
        sid, j = r.sid, float(r.j)
        t1 = pd.Timestamp(int(r.t1), unit="s").normalize()
        site = meta.loc[sid, "site_id"] if sid in meta.index else None
        sibs = [s for s in meta.index[meta.site_id == site] if s != sid and s in U.columns] \
            if site is not None else []
        rows, ratios = [], []
        for s in sibs:
            if sid not in U.columns:
                break
            d = (U[sid] - U[s]).dropna()
            for W in (90, 365):
                b = d[(d.index < t1) & (d.index >= t1 - pd.Timedelta(days=W))]
                a = d[(d.index > t1) & (d.index <= t1 + pd.Timedelta(days=W))]
                if len(b) >= 30 and len(a) >= 30:
                    dl = float(a.median() - b.median())
                    rows.append({"sibling": s, "layer": meta.loc[s, "layer_code"], "W": W,
                                 "d_before": float(b.median()), "d_after": float(a.median()),
                                 "delta": dl, "delta_over_j": dl / j, "n": [len(b), len(a)]})
                    if W == 365:
                        ratios.append(dl / j)
        med = float(np.median(ratios)) if ratios else None
        verdict = ("no_sibling_data" if med is None else
                   "sibling_confirmed" if med >= FRAC else "not_confirmed")
        recs.append({"sid": sid, "period": r.period, "t1": str(t1.date()), "j": j,
                     "flag": int(r.flag), "backing": r.backing,
                     "median_delta_over_j_365d": med, "verdict": verdict, "siblings": rows})
    V = pd.Series([x["verdict"] for x in recs], dtype=object)
    return {"rule": "median over siblings of (difference shift over 365 d) / j >= 0.5",
            "verdicts": V.value_counts().to_dict() if len(V) else {}, "corrections": recs}


def check_flagged(d2: pd.DataFrame, wells: dict, per: str) -> dict:
    g = d2[(d2.period == per) & d2.kind.isin(["offset_episode", "dropout"])]
    st = []
    for _, r in g.iterrows():
        W = wells.get(r.sid)
        if W is None:
            continue
        c = episode_check(W, int(r.t1), int(r.t2), float(r.j1), float(r.j2))
        st.append({"sid": r.sid, "kind": r.kind, "v2_status": r.v2_status,
                   "duration_d": float(r.duration_d), "status": c["status"]})
    S = pd.DataFrame(st)
    out = {"n": int(len(S))}
    if len(S):
        out.update({
            "episodes_fail_level_check": int(((S.kind == "offset_episode")
                                              & (S.status == "fail")).sum()),
            "episodes_checked": int(((S.kind == "offset_episode")
                                     & (S.status != "undetermined")).sum()),
            "dropouts_fail_level_check": int(((S.kind == "dropout")
                                              & (S.status == "fail")).sum()),
            "by_kind_status": {f"{a}|{b}": int(n) for (a, b), n in
                               S.groupby(["kind", "status"]).size().items()},
            "fail_by_duration": {str(k): int(n) for k, n in S[S.status == "fail"].groupby(
                pd.cut(S[S.status == "fail"].duration_d, [0, 15, 60, 180, 366]),
                observed=True).size().items()}})
    return out


def check_v1(v1: Path, wells: dict) -> dict:
    E = v1_objects(v1)
    out = {}
    for per in ("pre", "post"):
        g = E[E.period == per]
        st_steps, st_eps = [], []
        for _, r in g.iterrows():
            W = wells.get(r.sid)
            if W is None:
                continue
            if r.kind == "step":
                c = level_check(W, int(r.t1), float(r.j1))
                st_steps.append((int(r.flag) if np.isfinite(r.flag) else -1, c["status"]))
            else:
                j2 = float(r.j2) if np.isfinite(r.j2) else np.nan
                c = episode_check(W, int(r.t1), int(r.t2), float(r.j1), j2)
                st_eps.append((int(r.flag) if np.isfinite(r.flag) else -1, r.kind, c["status"]))
        out[per] = {"steps": {f"flag{a}|{b}": int(n) for (a, b), n in
                              pd.Series(st_steps).value_counts().items()} if st_steps else {},
                    "episodes_dropouts": {f"flag{a}|{k}|{b}": int(n) for (a, k, b), n in
                                          pd.Series(st_eps).value_counts().items()}
                    if st_eps else {}}
    return out


def examples(build: Path, v1: Path) -> dict:
    cases = [("07230332", "2012-01-01", "2016-12-01"), ("07260122", "2018-09-01", "2026-10-01"),
             ("09190132", "2019-10-01", "2026-10-01"), ("09190132", "2024-05-28", "2024-06-05")]
    D2 = pd.read_parquet(build / "heads_daily.parquet")
    D1 = pd.read_parquet(v1 / "heads_daily.parquet")
    out = []
    for sid, a, b in cases:
        rec = {"sid": sid, "window": f"{a}..{b}"}
        for tag, D in (("v1", D1), ("v2", D2)):
            d = D[(D.sid == sid) & (D.date >= a) & (D.date < b)]
            rec[tag] = {"n_days": int(len(d)),
                        "n_days_corrected": int((d.corr_m.abs() > 0).sum()),
                        "median_corr_m": float(d.corr_m.median()) if len(d) else None,
                        "median_value_raw": float(d.value_raw.median()) if len(d) else None,
                        "median_value": float(d.value.median()) if len(d) else None,
                        "flags": {int(k): int(n) for k, n in d.flag.value_counts().items()}}
        out.append(rec)
    return out


def barometric(build: Path, val: dict) -> dict:
    B = pd.read_csv(build / "barometric_efficiency.csv", dtype={"sid": str})
    acc = B[B.accepted]
    neg = acc[acc.A_K < -1]
    pos = acc[acc.A_K > 0]
    D = pd.read_parquet(build / "heads_daily.parquet", columns=["sid", "baro_m"])
    flagged = set(neg.sid) | set(pos.sid)
    bd = D[D.sid.isin(flagged)]
    o1 = val["conformance"]["o1_coherence"]["pre"]
    return {"n_accepted_epochs": int(len(acc)), "n_accepted_wells": int(acc.sid.nunique()),
            "n_accepted_A_K_lt_minus1": int(len(neg)),
            "accepted_A_K_lt_minus1": neg[["sid", "epoch", "A_K", "se_A_K", "partial_r2"]]
            .round(4).to_dict("records"),
            "n_accepted_A_K_positive": int(len(pos)),
            "wells_A_K_positive": sorted(pos.sid.unique().tolist()),
            "median_partial_r2_accepted": float(acc.partial_r2.median()),
            "max_abs_baro_m_flagged": float(bd.baro_m.abs().max()) if len(bd) else None,
            "p99_abs_baro_m_flagged": float(bd.baro_m.abs().quantile(0.99)) if len(bd) else None,
            "max_abs_baro_m_all": float(D.baro_m.abs().max()),
            "o1_all_wells_pre": o1["all_wells"], "o1_corrected_wells_pre": o1["corrected_wells"]}


def injection(inj_path: Path) -> dict:
    inj = json.loads(inj_path.read_text())
    R = pd.DataFrame(inj["rows"])
    R = R[R.status == "injected"].copy()
    out = {}
    for kind, g in R.groupby("kind"):
        g = g.copy()
        g["recovered"] = g["recovered"].astype(float)
        rec = {"n": int(len(g)), "recall": float(g.recovered.mean())}
        if kind == "paired":
            long_ = g[g.effective_days > 365]
            rec.update({"n_effective_days_gt_365": int(len(long_)),
                        "misses_effective_days_gt_365": long_[long_.recovered == 0][
                            ["real", "sid", "size", "effective_days"]].to_dict("records"),
                        "recall_without_effective_gt_365":
                            float(g[g.effective_days <= 365].recovered.mean())})
        rec["misses"] = g[g.recovered == 0][["real", "sid", "size", "kinds_at_jump",
                                             "effective_days"]].to_dict("records")
        out[kind] = rec
    return out


def rescore(val: dict) -> dict:
    out = {}
    for tag, key in (("v2", "rescore_frozen_challenge"), ("v1", "rescore_v1_split")):
        r = val[key]
        out[tag] = {"original": r["original"]["model_over_clim"],
                    "original_on_l1_months": r["original_on_l1_months"]["model_over_clim"],
                    "l1_monthly": r["l1_monthly"]["model_over_clim"], "split": r["split"]}
    return out


def main():
    work, build, v1 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    inj_path, out_json = Path(sys.argv[4]), Path(sys.argv[5])
    vpath = Path(sys.argv[6]) if len(sys.argv) > 6 else out_json.parent / "validation_v2.json"
    val = json.loads(vpath.read_text())
    d2 = pd.read_parquet(work / "decisions_v2_stage1.parquet")
    C = v2_corrections(work, build)
    E1 = v1_objects(v1)
    sids = set(C.sid) | set(d2[d2.kind.isin(["offset_episode", "dropout"])].sid) | set(E1.sid)
    wells = load_wells(work, sids)
    out = {"note": "after-the-fact verification (reported only; nothing gated or tuned); "
                   "independent estimator: plain 10-day medians of hourly medians of the "
                   "uncorrected series, pass = jump's sign and >= 0.5 |j|",
           "corrections_level_check": check_corrections(C, wells),
           "corrections_sibling_check": sibling_check(C, work, build),
           "false_flag_proxies": {per: check_flagged(d2, wells, per) for per in ("pre", "post")},
           "v1_level_check": check_v1(v1, wells),
           "examples": examples(build, v1),
           "barometric_plausibility": barometric(build, val),
           "injection_r10_19": injection(inj_path),
           "rescore": rescore(val)}
    D = pd.read_parquet(build / "heads_daily.parquet", columns=["sid", "period", "flag"])
    out["false_flag_proxies"]["flag9_day_share"] = {
        per: float((g.flag == 9).mean()) for per, g in D[D.flag != 8].groupby("period")}
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    cl = out["corrections_level_check"]
    print(json.dumps({"corrections": {k: v for k, v in cl.items() if k != "corrections"},
                      "sibling_check": out["corrections_sibling_check"]["verdicts"],
                      "false_flag_proxies": out["false_flag_proxies"],
                      "v1_level_check": out["v1_level_check"],
                      "barometric": {k: v for k, v in out["barometric_plausibility"].items()
                                     if not k.startswith("accepted_A_K")},
                      "injection": {k: {kk: vv for kk, vv in v.items() if kk != "misses"}
                                    for k, v in out["injection_r10_19"].items()},
                      "rescore": out["rescore"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
