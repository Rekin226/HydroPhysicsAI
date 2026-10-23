"""H1a v2 acceptance tests (round 2): injected paired offsets >= 1 m and steps >= 3 m
recovered by the v2 decision layer (``decide_v2``), recall >= 0.9 each; false flags
reported. Pre-origin data only (2012-01..2022-12; U7). Run with ``python -I`` in a CPU slot.

Harness as in v1 (``s5_inject.py``): family O of ``harness.gaps`` through the origin guard
``harness.gaps_guarded.inject_offsets_steps`` (seed 1000*ord('O') + r; 20 paired offsets of
1-3 m lasting 15-365 days and 20 steps of 3-6 m per realization, at most one event per well,
days flagged 9 in the base build excluded with a 30-day margin), the jump placed at the first
injectable sample pair (<= 70 min apart, 10:00-18:00) on or after the generator's day, and
the well's pre-origin stage 1 rerun on the injected record with its base barometric loading
held fixed. Differences from v1:

* realizations are chosen on the command line (``--reals A B`` = r in A..B-1). The v2
  registered test uses the fresh units r = 10..19; r = 0..9 were scored by v1 and serve as
  design-time evidence for v2 only;
* the exclusion mask comes from the base build given as BUILD_DIR (v2: its own build);
* recovery is judged on the v2 decisions of the injected run (``decide_v2.decide_well`` with
  the well's ERA5 cell pressure, no backing evidence needed):
  - paired (primary): an offset episode or dropout whose first jump is within +-1 day of the
    injected jump and whose return is within +-1 day of the reversal; lenient (reported): any
    flagged v2 object (episode, dropout, step, excursion, jump edge) starting within +-1 day
    of the injected jump; stage-1 detection (reported, the v1 metric): a stage-1 episode with
    both jumps matched;
  - step (primary): an accepted step, offset episode or dropout whose first jump is within
    +-1 day of the injected jump (v1's definition with v2 objects); strict (reported): an
    accepted step; also reported: the share of accepted steps whose displaced side is the
    injected side, and recovery including excursion runs (lenient);
* false flags (reported): flagged v2 objects of the injected run that match neither an
  injected jump/reversal (+-1 day) nor a flagged v2 object of the base run (+-1 day), per
  injected event. This counts only flags created by the injection; it is NOT the false-flag
  rate on real data (s4_validate_v2 reports the real-data proxies).

Usage: s5_inject_v2.py WORKDIR BUILD_DIR OUT_JSON --reals A B
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import gaps_guarded  # noqa: E402

from h1_heads_qc import baro, decide_v2, pipeline, rawio, screens  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402
from h1_heads_qc.s5_inject import DAYS, first_on_or_after, injectable_times  # noqa: E402

DAY = 86400
FLAGGING = ("offset_episode", "dropout", "step", "excursion", "jump_edge")
EPISODE_KINDS = ("offset_episode", "dropout")


def bounds_of(meta_row: dict):
    elev, depth = meta_row.get("WellElevation"), meta_row.get("WellDepth")
    bottom = (elev - depth) if (elev is not None and depth is not None and depth > 0) else None
    top = elev + C.TOP_MARGIN_M if elev is not None else None
    return bottom, top


def v2_events(res: dict, bottom, top, pressure) -> list[dict]:
    r = {k: v for k, v in res.items() if k not in ("post", "post_cat", "o1_post", "be_oos")}
    rows = decide_v2.decide_well(r, bottom=bottom, top=top, pressure=pressure)
    return [{"t": int(x["t1"]), "t2": int(x["t2"]), "kind": x["kind"], "side": x.get("side")}
            for x in rows if x["kind"] in FLAGGING]


def stage1_events(res: dict) -> list[dict]:
    cat = res.get("pre_cat", {})
    return [{"t": E["t1"], "t2": E["t2"]} for E in cat.get("episodes", [])]


def main():
    work, build, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    i = sys.argv.index("--reals")
    r_lo, r_hi = int(sys.argv[i + 1]), int(sys.argv[i + 2])
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    P = baro.load_pressure()
    daily = pd.read_parquet(build / "heads_daily.parquet",
                            columns=["sid", "date", "flag", "day_valid", "period", "n_suspect"])
    daily = daily[daily.period == "pre"]
    base_rate = {"share_days_flag9": float((daily.flag == 9).mean()),
                 "share_days_with_any_flag9_sample": float((daily.n_suspect > 0).mean()),
                 "n_days": int(len(daily))}
    t00 = time.time()
    base, inj_days, sids, mrows, press = {}, {}, [], {}, {}
    for p in sorted((work / "stage1").glob("*.pkl")):
        res = pickle.loads(Path(p).read_bytes())
        if "pre" not in res or len(res["pre"]["t"]) < 1000:
            continue
        sid = res["sid"]
        sh, _ = rawio.load_well(sid, keep_sentinels=True, pre_only=True)
        t, v = screens.to_arrays(sh)
        inj_days[sid] = injectable_times(t, v)
        be = res.get("be", {})
        segs = [{"start": x["start"], "end": x["end"], "a": x.get("a")}
                for x in be.get("segments", [])]
        mrow = meta.loc[sid].to_dict()
        for key in ("WellElevation", "WellDepth"):
            mrow[key] = float(mrow[key]) if mrow.get(key) is not None and \
                np.isfinite(mrow.get(key)) else None
        mrows[sid] = mrow
        cid = mrow.get("era5_cell")
        press[sid] = P[cid] if cid in P.columns else None
        bottom, top = bounds_of(mrow)
        base[sid] = {"segments": segs, "p_ref_hpa": res.get("p_ref_hpa"),
                     "events": v2_events(res, bottom, top, press[sid])}
        sids.append(sid)
    W, T = len(sids), len(DAYS)
    Y = np.full((W, T), np.nan)
    for k, s in enumerate(sids):
        for d in inj_days[s]:
            Y[k, d] = 0.0
    excl = np.zeros((W, T), bool)
    flagged = daily[(daily.flag == 9) | (daily.n_suspect > 0)]
    pos = {s: k for k, s in enumerate(sids)}
    for s, g in flagged.groupby("sid"):
        if s in pos:
            di = ((g.date - DAYS[0]).dt.days).to_numpy()
            di = di[(di >= 0) & (di < T)]
            excl[pos[s], di] = True
    months = [d.strftime("%Y-%m") for d in DAYS]
    print(f"{W} wells, matrix built in {time.time() - t00:.0f}s", flush=True)

    rows, spurious = [], []
    for r in range(r_lo, r_hi):
        _, evs = gaps_guarded.inject_offsets_steps(Y, r, months, exclude=excl)
        for k, e in enumerate(evs):
            sid = sids[e["well"]]
            inj = inj_days[sid]
            d0, tau0 = first_on_or_after(inj, e["start"], T - 1)
            rec = {"real": r, "k": k, "sid": sid, "kind": e["kind"], "size": e["size"],
                   "gen_start_day": str(DAYS[e["start"]].date()),
                   "gen_end_day": str(DAYS[e["end"]].date()) if e["end"] is not None
                   and e["end"] < T else None}
            if tau0 is None:
                rows.append(rec | {"status": "no_injectable_pair"})
                continue
            tau1 = None
            if e["kind"] == "paired":
                d1, tau1 = first_on_or_after(inj, max(e["end"], d0 + 1), T - 1)
                if tau1 is None:
                    rows.append(rec | {"status": "no_injectable_reversal"})
                    continue
            sh, _ = rawio.load_well(sid, keep_sentinels=True, pre_only=True)
            t, v = screens.to_arrays(sh)
            good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
            m = good & (t >= tau0)
            if tau1 is not None:
                m &= t < tau1
            v2 = v.copy()
            v2[m] += e["size"]
            sh2 = pd.Series(v2, index=sh.index)
            out = pipeline.process_well(sid, mrows[sid], P, pre_only=True, hist_override=sh2,
                                        be_override={"segments": base[sid]["segments"],
                                                     "p_ref_hpa": base[sid]["p_ref_hpa"]},
                                        o1=False)
            bottom, top = bounds_of(mrows[sid])
            ev = v2_events(out, bottom, top, press[sid])
            s1 = stage1_events(out)
            near0 = [x for x in ev if abs(x["t"] - tau0) <= DAY]
            kinds0 = ";".join(sorted({x["kind"] for x in near0}))
            if e["kind"] == "paired":
                prim = any(x["kind"] in EPISODE_KINDS and abs(x["t2"] - tau1) <= DAY
                           for x in near0)
                det1 = any(abs(x["t"] - tau0) <= DAY and abs(x["t2"] - tau1) <= DAY for x in s1)
                rec.update(recovered=bool(prim), recovered_lenient=bool(near0),
                           stage1_detected=bool(det1), kinds_at_jump=kinds0)
            else:
                steps0 = [x for x in near0 if x["kind"] == "step"]
                epis0 = [x for x in near0 if x["kind"] in EPISODE_KINDS]
                rec.update(recovered=bool(steps0 or epis0), recovered_strict=bool(steps0),
                           recovered_lenient=bool(near0),
                           side_correct=bool(steps0[0]["side"] == "after") if steps0 else None,
                           kinds_at_jump=kinds0)
            rec.update(status="injected", tau0=str(screens.ts(tau0)),
                       tau1=str(screens.ts(tau1)) if tau1 is not None else None,
                       n_samples_shifted=int(m.sum()),
                       effective_days=float(((tau1 if tau1 is not None else int(t[good][-1]))
                                             - tau0) / DAY))
            rows.append(rec)
            marks = [tau0] + ([tau1] if tau1 is not None else [])
            bt = [x["t"] for x in base[sid]["events"]]
            for x in ev:
                if any(abs(x["t"] - q) <= DAY for q in marks):
                    continue
                if any(abs(x["t"] - q) <= DAY for q in bt):
                    continue
                spurious.append({"real": r, "sid": sid, "t": str(screens.ts(x["t"])),
                                 "kind": x["kind"]})
        print(f"realization {r} done {time.time() - t00:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    inj = R[R.status == "injected"].copy()
    for col in ("recovered", "recovered_lenient", "recovered_strict", "side_correct",
                "stage1_detected"):
        if col in inj:
            inj[col] = inj[col].map(lambda x: np.nan if x is None or
                                    (isinstance(x, float) and np.isnan(x)) else float(x))
    summ = {}
    for kind, g in inj.groupby("kind"):
        d = {"n_generated": int((R.kind == kind).sum()), "n_injected": int(len(g)),
             "n_recovered": int(g.recovered.sum()),
             "recall": float(g.recovered.mean()) if len(g) else None,
             "recall_lenient": float(g.recovered_lenient.mean())}
        if kind == "paired":
            d["stage1_detection_rate"] = float(g.stage1_detected.mean())
        else:
            d["recall_strict_accepted_step"] = float(g.recovered_strict.mean())
            sc = g.side_correct.dropna()
            d["side_correct_share_of_strict"] = float(sc.mean()) if len(sc) else None
        d["recall_by_realization"] = [round(float(x), 3) for x in
                                      g.groupby("real").recovered.mean().tolist()]
        sz = pd.cut(g["size"].abs(), [1, 1.5, 2, 2.5, 3, 4, 5, 6.01], right=False)
        d["recall_by_abs_size"] = {str(k): round(float(x), 3) for k, x in
                                   g.groupby(sz, observed=True).recovered.mean().items()}
        d["kinds_at_jump_of_misses"] = g[g.recovered == 0].kinds_at_jump.fillna("") \
            .value_counts().to_dict()
        summ[kind] = d
    n_inj = int(len(inj))
    out = {
        "test": "H1a v2 injection (family O)", "realizations": [r_lo, r_hi - 1],
        "data_window": "2012-01..2022-12 (pre-origin only)", "n_wells_eligible": W,
        "base_build": str(build), "summary": summ,
        "not_injected": R[R.status != "injected"].status.value_counts().to_dict(),
        "spurious_events": len(spurious),
        "spurious_per_injected_event": len(spurious) / n_inj if n_inj else None,
        "spurious_note": "flags created by the injection only (base-run detections excluded "
                         "by construction); not the false-flag rate on real data",
        "base_run_share_of_pre_origin_days_flagged_9": base_rate,
        "runtime_s": time.time() - t00, "rows": R.to_dict("records"), "spurious": spurious}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("summary", "not_injected", "spurious_events",
                                          "spurious_per_injected_event")}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
