"""S1a repair round r3 (S1ar): diagnosis of the failed check (e), BEFORE any method change.

Look: ``r3_looks diagnose`` (logged 2026-10-17T20:57:24Z, before this module first read the
data). Reads only the spent units of S1a.e_synthetic_mixed.v1: ring surveys 2014-05..2021-12
and the pre-origin pile rows; never the pre-2012 surveys.

Parts (each writes a JSON under results/dataprep/reports/S1a_evidence/repair_r3/diag/):

* ``rerun``: re-executes the registered pipeline (``synthetic.run`` logic, ``s1lib.evaluate``
  unchanged) for every scored window, primary and stress variants, keeping the internals the
  registered rows do not store (line likelihood ratios, sd_deep, pool sizes and members), and
  checks that it reproduces ``synthetic_rows.parquet``; the same loop with ``forced = {}``
  (the verifier's check, pools recomputed).
* ``truth``: b_deep of the UNINJECTED surveys per window and site (the rate-continuity error
  of the estimator: the true scale there is 0), per deep interval.
* ``timing``: per site and survey, the depth scale of the survey-to-survey change of the deep
  rings, to tell a physical deep anomaly (gradual, deep intervals only) from an instrument scale
  step in the reference surveys (abrupt, common date, all rings).

    tools/cpu_slot.sh python -m s1_rings.r3_diagnose rerun|truth|timing   (from dataprep_staging/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import gaps_guarded  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402
from s1_rings import synthetic as SY  # noqa: E402

OUT = K.EVIDENCE / "repair_r3" / "diag"
ROWS_REG = K.EVIDENCE / "synthetic" / "synthetic_rows.parquet"
ASSIGN = K.EVIDENCE / "assignment.json"


def _inputs():
    from s1_rings import s1io
    piles, notes = s1io.piles(preorigin=True)
    pos = {s: L.load_positions(s).loc[:SY.END] for s in K.SITES}
    p_A = float(json.loads(ASSIGN.read_text())["p_A_rule"]["p_A"])
    return pos, piles, p_A, notes


def window_entries(pos_all: dict, g: dict) -> dict:
    """Sites entering a window under the registered entry rules (synthetic.run)."""
    ents = {}
    for s, p in pos_all.items():
        p = p.loc[:SY.END]
        if (SY.n_surveys(p, *g["pre"]) < K.SYN_MIN_PRE
                or SY.n_surveys(p, *g["post"]) < K.SYN_MIN_POST):
            continue
        keep = (p.index < g["mask"][0]) | (p.index > g["mask"][1])
        pt = p.loc[keep].loc[g["pre"][0]:g["post"][1]]
        try:
            L.gap_stats(pt, g["pre"], g["post"])
        except ValueError:
            continue
        ents[s] = pt
    return ents


def run_realization(ents, g, piles, inj, forced_mode="registered"):
    """One realization of the registered pipeline; returns (res, gsc, truth, forced)."""
    pos, truth = {}, {}
    for d in inj:
        s = d["site"]
        pt = ents[s]
        po = pt.copy()
        after = po.index > g["mask"][1]
        po.loc[after] = d["a"] + (1.0 + d["b"]) * po.loc[after]
        pos[s], truth[s] = po, pt
    gsc, forced = {}, set()
    for s, po in pos.items():
        gsc[s] = L.gap_stats(po, g["pre"], g["post"])
        if forced_mode == "registered" and abs(gsc[s]["dS_obs"]) <= K.NEAR_THRESHOLD_M:
            forced.add(s)
    pl = {s: (piles[v[0]], v[1]) for s, v in K.PILES.items() if s in pos}
    pw = {s: (g["pre"], g["post"]) for s in pl}
    res = L.evaluate(pos, {s: g["pre"] for s in pos}, {s: g["post"] for s in pos},
                     pl, forced, pile_wins=pw, gs_cache=gsc)
    return res, gsc, truth, forced


def rerun(nreal=K.SYN_REAL) -> dict:
    pos_all, piles, p_A, _ = _inputs()
    wins = gaps_guarded.ring_windows(K.SYN_WINDOW[0], K.SYN_WINDOW[1], K.SYN_LEN)
    rows = []
    t0 = time.time()
    for wi, w in enumerate(wins):
        g = SY.window_geometry(w)
        ents = window_entries(pos_all, g)
        if len(ents) < K.SYN_MIN_SITES:
            continue
        sites = sorted(ents)
        for k in range(nreal):
            r = 100 * wi + k
            for var in ("primary", "stress"):
                inj = (SY.inject_groupcond if var == "primary" else SY.inject_harness)(
                    sites, r, p_A)
                for mode in ("registered", "noforce"):
                    res, gsc, truth, forced = run_realization(ents, g, piles, inj, mode)
                    for d in inj:
                        s = d["site"]
                        rr, gs = res[s], gsc[s]
                        S = truth[s][gs["bot"]] - truth[s][gs["top"]]
                        C_true = float(S.loc[gs["d_lp"]] - S.loc[gs["d_fp"]])
                        ln = rr.lines
                        rows.append({
                            "variant": var, "mode": mode, "window": wi, "start": w["start"],
                            "k": k, "site": s, "true_group": d["group"],
                            "b_true_ppm": 1e6 * d["b"], "a_true_cm": 100 * d["a"],
                            "assigned": rr.assign["assigned"] or "",
                            "evidence_group": rr.assign["evidence_group"] or "",
                            "forced": s in forced, "votes_A": rr.assign["votes_A"],
                            "votes_B": rr.assign["votes_B"], "p_A": rr.p_A,
                            "lr_offset": ln["offset"]["lr"], "vote_offset": ln["offset"]["vote"],
                            "lr_bdeep": ln["b_deep"]["lr"], "vote_bdeep": ln["b_deep"]["vote"],
                            "lr_pile": ln["pile"]["lr"] if "pile" in ln else np.nan,
                            "vote_pile": ln["pile"]["vote"] if "pile" in ln else None,
                            "b_deep_ppm": 1e6 * gs["b_deep"], "sd_deep_ppm": 1e6 * gs["sd_deep"],
                            "n_deep": gs["n_deep"], "deep_len_m": gs["deep_len"],
                            "L_m": gs["L"], "sd_s_mm": 1e3 * gs["sd_s"],
                            "b_pile_ppm": (1e6 * rr.pile["b"]) if rr.pile and rr.pile.get("ok")
                            else np.nan,
                            "sd_pile_ppm": (1e6 * rr.pile["sd"]) if rr.pile and rr.pile.get("ok")
                            else np.nan,
                            "pool_A_n": rr.members["A"]["pool"]["n"],
                            "pool_B_n": rr.members["B"]["pool"]["n"],
                            "b_A_ppm": 1e6 * rr.members["A"]["b"],
                            "b_B_ppm": 1e6 * rr.members["B"]["b"],
                            "sig_A_ppm": 1e6 * rr.members["A"]["sigma"],
                            "sig_B_ppm": 1e6 * rr.members["B"]["sigma"],
                            "fallback_A": rr.members["A"]["pool"]["fallback"],
                            "fallback_B": rr.members["B"]["pool"]["fallback"],
                            "C_A_cm": 100 * rr.members["A"]["C"],
                            "C_B_cm": 100 * rr.members["B"]["C"],
                            "C_true_cm": 100 * C_true, "C_primary_cm": 100 * rr.primary["C"],
                            "err_cm": 100 * (rr.primary["C"] - C_true),
                            "sigma_cm": 100 * rr.primary["sigma_C"],
                            "err_oracle_cm": 100 * (rr.members[d["group"]]["C"] - C_true)})
        print(f"window {wi} {w['start']}: {len(ents)} sites ({time.time() - t0:.0f} s)",
              flush=True)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_parquet(OUT / "rerun_rows.parquet", index=False)
    # reproduction check against the registered rows
    reg = pd.read_parquet(ROWS_REG)
    mine = R[R["mode"] == "registered"].copy()
    key = ["variant", "window", "k", "site"]
    mm = reg.merge(mine, on=key, suffixes=("_reg", "_new"))
    rep = {"n_registered": int(len(reg)), "n_rerun": int(len(mine)), "n_matched": int(len(mm)),
           "max_abs_err_diff_cm": float(np.abs(mm["err_cm_reg"] - mm["err_cm_new"]).max()),
           "max_abs_p_A_diff": float(np.abs(mm["p_A_reg"] - mm["p_A_new"]).max()),
           "same_assigned": bool((mm["assigned_reg"] == mm["assigned_new"]).all())}
    (OUT / "rerun_reproduction.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return rep


# ----------------------------------------------------------------------------- truth field
def truth_field() -> dict:
    pos_all, _, _, _ = _inputs()
    wins = gaps_guarded.ring_windows(K.SYN_WINDOW[0], K.SYN_WINDOW[1], K.SYN_LEN)
    rows, ivrows = [], []
    for wi, w in enumerate(wins):
        g = SY.window_geometry(w)
        ents = window_entries(pos_all, g)
        if len(ents) < K.SYN_MIN_SITES:
            continue
        for s, pt in sorted(ents.items()):
            gs = L.gap_stats(pt, g["pre"], g["post"])
            rows.append({"window": wi, "start": w["start"], "site": s,
                         "b_deep_truth_ppm": 1e6 * gs["b_deep"],
                         "sd_deep_ppm": 1e6 * gs["sd_deep"], "n_deep": gs["n_deep"],
                         "deep_len_m": gs["deep_len"], "L_m": gs["L"],
                         "n_rings": len(gs["rings"]), "z_bot_m": float(gs["z"][-1]),
                         "gap_yr": gs["gap"], "r_pre_mm_yr": 1e3 * gs["r_pre"],
                         "r_post_mm_yr": 1e3 * gs["r_post"],
                         "C_true_total_cm": 100 * (gs["S_lp"] - gs["S_fp"]),
                         "C_mid_cm": 100 * gs["C_mid"]})
            # per-interval excess change over the gap (truth): dS_k + C_k (m), mid depth
            z = gs["z"]
            J = gs["J"]
            fits = None  # rates are re-derived below from the window fits of the rings
            pre = pt.loc[g["pre"][0]:g["pre"][1]][gs["rings"]]
            post = pt.loc[g["post"][0]:g["post"][1]][gs["rings"]]
            tp, tq = L.years(pre.index), L.years(post.index)
            t_m = float(L.years([gs["t_mid"]])[0])
            rp = np.array([L.window_fit(tp, pre[c].to_numpy(), t_m).rate for c in gs["rings"]])
            rq = np.array([L.window_fit(tq, post[c].to_numpy(), t_m).rate for c in gs["rings"]])
            Lk, dSk = np.diff(z), np.diff(J)
            rk_pre, rk_post = -np.diff(rp), -np.diff(rq)
            Ck = 0.5 * (rk_pre + rk_post) * gs["gap"]
            mid = 0.5 * (z[1:] + z[:-1])
            deep = (mid > np.median(z)) & (np.abs(rk_pre) < K.RATE_THR) \
                & (np.abs(rk_post) < K.RATE_THR)
            for k in range(len(Lk)):
                ivrows.append({"window": wi, "start": w["start"], "site": s, "k": k,
                               "mid_m": float(mid[k]), "L_k_m": float(Lk[k]),
                               "excess_mm": 1e3 * float(dSk[k] + Ck[k]),
                               "rate_pre_mm_yr": 1e3 * float(rk_pre[k]),
                               "rate_post_mm_yr": 1e3 * float(rk_post[k]),
                               "deep": bool(deep[k]), "below_median": bool(mid[k] > np.median(z))})
            del fits
        print(f"truth window {wi} {w['start']}", flush=True)
    T = pd.DataFrame(rows)
    IV = pd.DataFrame(ivrows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "truth_bdeep.csv", index=False)
    IV.to_parquet(OUT / "truth_intervals.parquet", index=False)
    return {"n": int(len(T))}


# ----------------------------------------------------------------------------- timing
def timing() -> dict:
    """Survey-to-survey depth scale of the deep rings, per site, 2014-05..2021-12.

    For consecutive surveys t1 < t2 (<= 62 days apart) and the rings read at both below the
    site's median ring depth, OLS of dz_i = z_i(t2) - z_i(t1) on z_i gives a slope s (ppm):
    an instrument scale step shows as one large s common to many sites on one date; physical
    deep compaction between two surveys is a few ppm at most."""
    pos_all, _, _, _ = _inputs()
    rows = []
    for s, p in pos_all.items():
        p = p.loc["2014-05-01":"2021-12-31"]
        cover = p.notna().mean()
        rings = [c for c in p.columns if cover[c] >= 0.8]
        zbar = p[rings].mean().sort_values()
        deep = list(zbar.index[zbar > zbar.median()])
        allr = list(zbar.index)
        idx = p.index
        for i in range(1, len(idx)):
            t1, t2 = idx[i - 1], idx[i]
            if (t2 - t1).days > 62:
                continue
            for name, rs in (("deep", deep), ("all", allr)):
                a, b = p.loc[t1, rs], p.loc[t2, rs]
                ok = a.notna() & b.notna()
                if ok.sum() < 4:
                    continue
                z = a[ok].to_numpy(float)
                dz = (b - a)[ok].to_numpy(float)
                X = np.stack([np.ones_like(z), z], 1)
                beta, res, *_ = np.linalg.lstsq(X, dz, rcond=None)
                resid = dz - X @ beta
                rows.append({"site": s, "set": name, "t1": t1.date().isoformat(),
                             "t2": t2.date().isoformat(), "month": str(t2.to_period("M")),
                             "n": int(ok.sum()), "slope_ppm": 1e6 * float(beta[1]),
                             "intercept_mm": 1e3 * float(beta[0]),
                             "resid_sd_mm": 1e3 * float(np.std(resid, ddof=2))
                             if ok.sum() > 2 else np.nan})
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "timing_survey_scale.csv", index=False)
    return {"n": int(len(T))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["rerun", "truth", "timing"])
    ap.add_argument("--nreal", type=int, default=K.SYN_REAL)
    a = ap.parse_args(argv)
    out = {"rerun": lambda: rerun(a.nreal), "truth": truth_field, "timing": timing}[a.part]()
    print(json.dumps(out, default=str)[:2000])


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
