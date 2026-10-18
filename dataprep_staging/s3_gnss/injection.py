"""WP S3: injected-step tests (plan 4.4 row "GNSS (S3) | Equipment steps | Injected steps
>= 2 cm"), shared by the pre-origin tuning (``tune.py``) and the acceptance run
(``validate.py``).

One trial per station per family run: a step of the given size is added to the raw series
from a random observed day and the whole family pipeline (all passes, detection common
mode, catalogue and metadata steps) is rerun. A trial is *recovered* when a step is found
within ``date_tol`` days of the injection day whose size is within ``size_tol`` of the
injected size. Injection days avoid the first and last 90 days, anything within 60 days of a
step, excursion or untied segment of the un-injected run, and need >= 20 observations on
each side within 30 days.
"""
from __future__ import annotations

import model as M
import numpy as np
import pandas as pd
import pipeline as P


def eligible_days(r: M.StationResult, margin: int = 90, clear: int = 60,
                  need: int = 20) -> np.ndarray:
    n = len(r.dates)
    ok = r.ok & ~r.untied & np.isfinite(r.y)
    elig = ok.copy()
    elig[:margin] = False
    elig[max(0, n - margin):] = False
    bad = r.transient | r.untied
    for st in r.steps:
        j = int(r.dates.searchsorted(st.date))
        bad[max(0, j - clear): j + clear] = True
    # dilate the bad mask by `clear`
    idx = np.flatnonzero(r.transient)
    for j in idx[::7]:
        bad[max(0, j - clear): j + clear] = True
    elig &= ~bad
    cs = np.concatenate([[0], np.cumsum(ok.astype(int))])
    i = np.arange(n)
    left = cs[i] - cs[np.clip(i - 30, 0, n)]
    right = cs[np.clip(i + 30, 0, n)] - cs[i]
    elig &= (left >= need) & (right >= need)
    return elig


def run_trials(family: str, p: M.Params, base: P.Family, window_end, amps: list[float],
               n_rep: int, seed: int, ref: list[str] | None = None,
               stations: list[str] | None = None, date_tol: int = 3,
               size_tol: float = 0.005, series=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sts = [k for k in base.results if stations is None or k in stations]
    elig = {k: eligible_days(base.results[k]) for k in sts}
    base_steps = {k: [s.date for s in base.results[k].steps] for k in sts}
    trial = 0
    for _rep in range(n_rep):
        for a in amps:
            inj = {}
            for k in sts:
                e = np.flatnonzero(elig[k])
                if e.size == 0:
                    continue
                j = int(rng.choice(e))
                c = float(a) * (1 if rng.random() < 0.5 else -1)
                inj[k] = [(base.results[k].dates[j], c)]
            if family == "wra":
                fam = P.run_wra(p, window_end=window_end, inject=inj, stations=sts,
                                series=series)
            else:
                fam = P.run_ngl(p, ref=ref, window_end=window_end, inject=inj,
                                stations=sts, series=series)
            for k, ((d, c),) in inj.items():
                r = fam.results[k]
                cand = [(abs((s.date - d).days), s) for s in r.steps
                        if abs((s.date - d).days) <= 15]
                near = min(cand, key=lambda q: q[0])[1] if cand else None
                hit = [s for s in r.steps if abs((s.date - d).days) <= date_tol]
                best = min(hit, key=lambda s: abs(s.size - c)) if hit else None
                extra = [s for s in r.steps
                         if abs((s.date - d).days) > date_tol
                         and all(abs((s.date - b).days) > date_tol for b in base_steps[k])]
                in_exc = bool(r.transient[int(r.dates.searchsorted(d))])
                rows.append({
                    "family": family, "trial": trial, "station": k, "inject_date": d,
                    "size_m": c, "abs_size_m": abs(c),
                    "found": best is not None,
                    "date_err_d": (best.date - d).days if best is not None else np.nan,
                    "size_err_m": (best.size - c) if best is not None else np.nan,
                    "near_date_err_d": (near.date - d).days if near is not None else np.nan,
                    "near_size_m": near.size if near is not None else np.nan,
                    "flagged_as_excursion": in_exc,
                    "recovered": bool(best is not None and abs(best.size - c) <= size_tol),
                    "extra_steps": len(extra)})
            trial += 1
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> dict:
    out = {"n": int(len(t)), "recovered": float(t.recovered.mean()) if len(t) else np.nan,
           "found_within_3d": float(t.found.mean()) if len(t) else np.nan}
    if len(t):
        f = t[t.found]
        out["size_err_mm_p90"] = float(np.nanpercentile(1000 * f.size_err_m.abs(), 90)) \
            if len(f) else np.nan
        out["size_err_mm_median"] = float(np.nanmedian(1000 * f.size_err_m.abs())) \
            if len(f) else np.nan
        out["date_err_abs_d_p90"] = float(np.nanpercentile(f.date_err_d.abs(), 90)) \
            if len(f) else np.nan
        out["extra_steps_per_trial"] = float(t.extra_steps.mean())
        out["by_size"] = {f"{a * 100:.0f}cm": {"n": int(len(g)),
                                               "recovered": float(g.recovered.mean())}
                          for a, g in t.groupby(t.abs_size_m.round(3))}
    return out


def run_trials_harness_g(family: str, p: M.Params, base: P.Family, window_end, reps: range,
                         ref: list[str] | None = None, series=None, date_tol: int = 3,
                         size_tol: float = 0.005) -> pd.DataFrame:
    """Secondary injection test with the A1 harness generator (plan 4.4 family G:
    ``harness.gaps.inject_gnss_steps``, 3 steps of 2-6 cm per series, >= 90 days apart,
    seed ``1000*ord('G') + r``), so later re-runs on the final product use the same masks.
    Epochs are calendar days of the station's daily grid; excluded epochs are those within
    60 days of a step, excursion or untied segment of the un-injected run. A step injected
    on an unobserved day takes effect at the next observed day (the true date)."""
    from harness.gaps import inject_gnss_steps
    rows = []
    sts = [k for k in base.results]
    elig = {k: eligible_days(base.results[k]) for k in sts}
    for r in reps:
        inj, truth = {}, {}
        for k in sts:
            b = base.results[k]
            n = len(b.dates)
            ex = np.zeros(n, bool)
            bad = b.transient | b.untied
            for st in b.steps:
                j = int(b.dates.searchsorted(st.date))
                bad[max(0, j - 60): j + 60] = True
            for j in np.flatnonzero(bad)[::7]:
                ex[max(0, j - 60): j + 60] = True
            ex |= bad
            if n < 2 * 90 + 10:
                continue
            _, steps = inject_gnss_steps(n, r, exclude=ex)
            obs = np.isfinite(b.y)
            lst, tr = [], []
            for s in steps:
                t = int(s["epoch"])
                nxt = np.flatnonzero(obs[t:])
                if nxt.size == 0:
                    continue
                d_true = b.dates[t + int(nxt[0])]
                lst.append((d_true, float(s["size"])))
                tr.append((d_true, float(s["size"]), bool(elig[k][t + int(nxt[0])])))
            inj[k], truth[k] = lst, tr
        if family == "wra":
            fam = P.run_wra(p, window_end=window_end, inject=inj, stations=sts, series=series)
        else:
            fam = P.run_ngl(p, ref=ref, window_end=window_end, inject=inj, stations=sts,
                            series=series)
        for k, tr in truth.items():
            res = fam.results[k]
            for d, c, e in tr:
                hit = [s for s in res.steps if abs((s.date - d).days) <= date_tol]
                best = min(hit, key=lambda s: abs(s.size - c)) if hit else None
                rows.append({"family": family, "harness_r": r, "station": k,
                             "inject_date": d, "size_m": c, "abs_size_m": abs(c),
                             "dense_enough": e, "found": best is not None,
                             "size_err_m": (best.size - c) if best is not None else np.nan,
                             "recovered": bool(best is not None
                                               and abs(best.size - c) <= size_tol)})
    return pd.DataFrame(rows)
