"""H1a acceptance test 1 (family O, plan 4.4): injected paired offsets >= 1 m and steps >= 3 m
are recovered with recall >= 0.9; the false-flag rate is reported. Pre-origin data only
(2012-01..2022-12), so that no 2023+ observation enters this gate (U7). Run with
``python -I`` in a CPU slot, after stage 1 (base run) and the pre-registration.

Per realization r = 0..9 (``harness.gaps.inject_offsets_steps``, seed 1000*ord('O') + r,
20 paired offsets of 1-3 m lasting 15-365 days and 20 steps of 3-6 m, one event per well):

* the generator works on a wells x days matrix of the pre-origin record whose finite cells
  are the days with an injectable sample pair (a non-sentinel sample at or after 10:00 and
  before 18:00 local time whose non-sentinel predecessor is at most 70 minutes earlier);
  days on which the base run flagged any sample 9 are passed as ``exclude`` (30-day
  margin), so events land outside existing artefacts;
* the injected jump time is the first injectable pair on or after the generator's start day
  (and, for a paired offset, on or after its end day: the reversal); the size is added to
  every non-sentinel raw sample from the jump to the reversal (paired) or to the end of the
  pre-origin record (step);
* the well's pre-origin pass is rerun on the injected series with the base run's barometric
  loading held fixed (no refit), so injected and base runs differ only by the injection.

Recovery (pre-registered):
* paired, primary: an offset episode (or dropout episode) whose first jump is within +-1 day
  of the injected jump and whose return jump is within +-1 day of the reversal;
  lenient (reported): any episode or step whose jump is within +-1 day of the injected jump;
* step, primary: a step, or an episode, whose first jump is within +-1 day of the injected
  jump (detected and flagged as an artefact starting there); strict (reported): a step;
  also reported: the share of recovered steps whose displaced side is the injected side.
* recall = recovered / injected, over all 200 + 200 events (criterion >= 0.9 for each kind).

False flags: events (episodes and steps) of the injected run that match neither an injected
jump/reversal (+-1 day) nor a base-run event (+-1 day), per injected event; and the base
run's share of pre-origin valid days flagged 9 (from heads_daily).

Usage: s5_inject.py WORKDIR BUILD_DIR OUT_JSON [--reals N]
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
from harness import gaps  # noqa: E402

from h1_heads_qc import baro, pipeline, rawio, screens  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402

DAY = 86400
DAYS = pd.date_range("2012-01-01", "2022-12-31", freq="D")
D0 = int(DAYS[0].value // 10**9)


def injectable_times(t: np.ndarray, v: np.ndarray) -> dict[int, int]:
    """day index -> time of the first injectable sample of that day."""
    ok = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
    tt = t[ok]
    if len(tt) < 2:
        return {}
    gap_ok = np.r_[False, np.diff(tt) <= C.JUMP_MAX_DT_S]
    sec = (tt - D0) % DAY
    cand = gap_ok & (sec >= C.INJECT_HOUR * 3600) & (sec < 18 * 3600)
    out = {}
    for x in tt[cand]:
        d = int((x - D0) // DAY)
        if d not in out:
            out[d] = int(x)
    return out


def first_on_or_after(inj: dict, day: int, max_day: int):
    for d in range(day, max_day + 1):
        if d in inj:
            return d, inj[d]
    return None, None


def events_of(cat: dict) -> list[dict]:
    ev = [{"t": E["t1"], "t2": E["t2"], "cls": "dropout" if E["dropout"] else "episode"}
          for E in cat.get("episodes", [])]
    ev += [{"t": S["t"], "t2": None, "cls": "step", "side": S["side"]}
           for S in cat.get("steps", [])]
    return ev


def main():
    work, build, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    nreal = int(sys.argv[sys.argv.index("--reals") + 1]) if "--reals" in sys.argv \
        else C.INJECT_REALIZATIONS
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    P = baro.load_pressure()
    daily = pd.read_parquet(build / "heads_daily.parquet",
                            columns=["sid", "date", "flag", "day_valid", "period", "n_suspect"])
    daily = daily[daily.period == "pre"]
    base_rate = {"share_days_flag9": float((daily.flag == 9).mean()),
                 "share_days_with_any_flag9_sample": float((daily.n_suspect > 0).mean()),
                 "n_days": int(len(daily))}
    t00 = time.time()
    # wells with a pre-origin record and a base run
    base, inj_days, sids = {}, {}, []
    for p in sorted((work / "stage1").glob("*.pkl")):
        res = pickle.loads(Path(p).read_bytes())
        if "pre" not in res or len(res["pre"]["t"]) < 1000:
            continue
        sid = res["sid"]
        sh, _ = rawio.load_well(sid, keep_sentinels=True)
        t, v = screens.to_arrays(sh)
        inj_days[sid] = injectable_times(t, v)
        be = res.get("be", {})
        segs = [{"start": x["start"], "end": x["end"], "a": x.get("a")}
                for x in be.get("segments", [])]
        base[sid] = {"segments": segs, "p_ref_hpa": res.get("p_ref_hpa"),
                     "events": events_of(res["pre_cat"])}
        sids.append(sid)
    W, T = len(sids), len(DAYS)
    Y = np.full((W, T), np.nan)
    for i, s in enumerate(sids):
        for d in inj_days[s]:
            Y[i, d] = 0.0
    excl = np.zeros((W, T), bool)
    flagged = daily[(daily.flag == 9) | (daily.n_suspect > 0)]
    pos = {s: i for i, s in enumerate(sids)}
    for s, g in flagged.groupby("sid"):
        if s in pos:
            di = ((g.date - DAYS[0]).dt.days).to_numpy()
            di = di[(di >= 0) & (di < T)]
            excl[pos[s], di] = True
    print(f"{W} wells, matrix built in {time.time() - t00:.0f}s", flush=True)

    rows, spurious = [], []
    for r in range(nreal):
        _, evs = gaps.inject_offsets_steps(Y, r, exclude=excl)
        for k, e in enumerate(evs):
            sid = sids[e["well"]]
            inj = inj_days[sid]
            d0, tau0 = first_on_or_after(inj, e["start"], T - 1)
            rec = {"real": r, "k": k, "sid": sid, "kind": e["kind"], "size": e["size"],
                   "gen_start_day": str(DAYS[e["start"]].date()),
                   "gen_end_day": str(DAYS[e["end"]].date()) if e["end"] is not None
                   and e["end"] < T else None}
            if tau0 is None:
                rec.update(status="no_injectable_pair")
                rows.append(rec)
                continue
            tau1 = None
            if e["kind"] == "paired":
                d1, tau1 = first_on_or_after(inj, max(e["end"], d0 + 1), T - 1)
                if tau1 is None:
                    rec.update(status="no_injectable_reversal")
                    rows.append(rec)
                    continue
            sh, _ = rawio.load_well(sid, keep_sentinels=True)
            t, v = screens.to_arrays(sh)
            good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
            m = good & (t >= tau0)
            if tau1 is not None:
                m &= t < tau1
            v2 = v.copy()
            v2[m] += e["size"]
            sh2 = pd.Series(v2, index=sh.index)
            mrow = meta.loc[sid].to_dict()
            for key in ("WellElevation", "WellDepth"):
                mrow[key] = float(mrow[key]) if mrow.get(key) is not None and \
                    np.isfinite(mrow.get(key)) else None
            out = pipeline.process_well(sid, mrow, P, pre_only=True, hist_override=sh2,
                                        be_override={"segments": base[sid]["segments"],
                                                     "p_ref_hpa": base[sid]["p_ref_hpa"]},
                                        o1=False)
            ev = events_of(out.get("pre_cat", {}))
            near0 = [x for x in ev if abs(x["t"] - tau0) <= DAY]
            if e["kind"] == "paired":
                prim = any(x["cls"] in ("episode", "dropout") and x["t2"] is not None
                           and abs(x["t2"] - tau1) <= DAY for x in near0)
                len_ = bool(near0)
                rec.update(recovered=bool(prim), recovered_lenient=len_)
            else:
                steps0 = [x for x in near0 if x["cls"] == "step"]
                inj_side = "after"
                rec.update(recovered=bool(near0), recovered_strict=bool(steps0),
                           side_correct=bool(steps0[0]["side"] == inj_side) if steps0 else None)
            rec.update(status="injected", tau0=str(screens.ts(tau0)),
                       tau1=str(screens.ts(tau1)) if tau1 is not None else None,
                       n_samples_shifted=int(m.sum()),
                       across_day_shift_days=int((tau0 - (D0 + e["start"] * DAY)) // DAY))
            rows.append(rec)
            # spurious: injected-run events matching neither an injection nor a base event
            marks = [tau0] + ([tau1] if tau1 is not None else [])
            bt = [x["t"] for x in base[sid]["events"]]
            for x in ev:
                if any(abs(x["t"] - q) <= DAY for q in marks):
                    continue
                if any(abs(x["t"] - q) <= DAY for q in bt):
                    continue
                spurious.append({"real": r, "sid": sid, "t": str(screens.ts(x["t"])),
                                 "cls": x["cls"]})
        print(f"realization {r} done {time.time() - t00:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    inj = R[R.status == "injected"].copy()
    # post-registration fix (summary only): rows that were never injected carry no outcome,
    # so these columns are object dtype (bool/None/NaN) and .round() failed; cast the
    # injected rows' outcomes to float (True 1.0, False 0.0, None NaN). Recovery logic unchanged.
    for col in ("recovered", "recovered_lenient", "recovered_strict", "side_correct"):
        if col in inj:
            inj[col] = inj[col].map(lambda x: np.nan if x is None else float(x))
    summ = {}
    for kind, g in inj.groupby("kind"):
        d = {"n_generated": int((R.kind == kind).sum()), "n_injected": int(len(g)),
             "n_recovered": int(g.recovered.sum()),
             "recall": float(g.recovered.mean()) if len(g) else None}
        if kind == "paired":
            d["recall_lenient"] = float(g.recovered_lenient.mean())
        else:
            d["recall_strict_step_class"] = float(g.recovered_strict.mean())
            sc = g.side_correct.dropna()
            d["side_correct_share_of_strict"] = float(sc.mean()) if len(sc) else None
        d["recall_by_realization"] = g.groupby("real").recovered.mean().round(3).tolist()
        summ[kind] = d
    n_inj = int(len(inj))
    out = {
        "test": "H1a.inject_O", "data_window": "2012-01..2022-12 (pre-origin only)",
        "n_wells_eligible": W, "realizations": nreal, "summary": summ,
        "not_injected": R[R.status != "injected"].status.value_counts().to_dict(),
        "spurious_events": len(spurious),
        "spurious_per_injected_event": len(spurious) / n_inj if n_inj else None,
        "base_run_share_of_pre_origin_days_flagged_9": base_rate,
        "runtime_s": time.time() - t00,
        "rows": R.to_dict("records"), "spurious": spurious}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("summary", "not_injected", "spurious_events",
                                          "spurious_per_injected_event",
                                          "base_run_share_of_pre_origin_days_flagged_9")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
