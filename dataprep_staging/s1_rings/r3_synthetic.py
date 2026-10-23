"""S1a r3 re-test of check (e): mixed A/B synthetic pseudo-gaps on FRESH windows.

Test S1a.e_synthetic_fresh.v3 (registered before it runs; threshold unchanged: worst-window
RMSE of the primary bridged gap compaction <= 1.5 cm, post hoc). Families:

* ``pre2012`` (GATING with method r3, primary injection): every 13-month window of the pre-2012
  record (``ring_windows('2003-01', '2011-12', 13)``) with its pre window (<= 36 months, from
  2003-01) and post window (<= 28 months, to 2012-02-29, the last survey month before the
  2012-03..2014-04 hiatus). A site enters with >= 12 pre and >= 12 post surveys; a window is
  scored with >= 8 sites. No S1a test, design or diagnosis has used these windows.
* ``late2019`` (reported): 13-month windows starting 2019-07..2019-12 (start months never used;
  the registered entry rule stopped at 2019-06) in the registered geometry with >= 20 pre and
  >= 12 post surveys and >= 10 sites. Their surveys (2016-07..2021-12) are design-exposed.
* ``spent`` (SECOND LOOK, reported, never gating): the 42 windows and the registered seeds of
  S1a.e_synthetic_mixed.v1, re-run with method r3 (paired with the registered rows).

Methods: ``r3`` (``r3_lib.evaluate_r3``) and ``registered`` (``s1lib.evaluate`` with v1's
settings, the control on the same draws). Injections: ``primary`` (offsets from the own group's
cluster, ``synthetic.inject_groupcond``) and ``stress`` (``harness.gaps
.ring_instrument_injection``), p_A = the registered 5/7. Seeds ``harness.gaps.rng_for('R', r)``:
pre2012 ``r = 30000 + 100 w + k``, late2019 ``r = 40000 + 100 w + k``, spent ``r = 100 w + k``
(registered), w the window index in its family's list, k = 0..9.

    tools/cpu_slot.sh python -m s1_rings.r3_synthetic fresh       (pre2012 + late2019; scores)
    tools/cpu_slot.sh python -m s1_rings.r3_synthetic spent       (second look; logs it first)
    python -m s1_rings.r3_synthetic fake                          (smoke test, toy sites)
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
from harness import coverage3, gaps_guarded  # noqa: E402

from s1_rings import r3_lib as R3  # noqa: E402
from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402
from s1_rings import synthetic as SY  # noqa: E402

PREREG_ID = "S1a.e_synthetic_fresh.v3"
OUT = K.EVIDENCE / "repair_r3" / "retest"
CALIB = K.EVIDENCE / "repair_r3" / "calibration.json"
THR_CM = 100 * K.WORST_WINDOW_RMSE_M

FAMILIES = {
    "pre2012": {"windows": ("2003-01", "2011-12"), "start_range": None,
                "start": "2003-01-01", "end": "2012-02-29", "min_pre": 12, "min_post": 12,
                "min_sites": 8, "r0": 30000},
    "late2019": {"windows": ("2015-01", "2021-12"), "start_range": ("2019-07", "2019-12"),
                 "start": "2014-05-01", "end": "2021-12-31", "min_pre": 20, "min_post": 12,
                 "min_sites": 10, "r0": 40000},
    "spent": {"windows": ("2015-01", "2021-12"), "start_range": None,
              "start": "2014-05-01", "end": "2021-12-31", "min_pre": K.SYN_MIN_PRE,
              "min_post": K.SYN_MIN_POST, "min_sites": K.SYN_MIN_SITES, "r0": 0},
}


def geometry(w: dict, fam: dict) -> dict:
    start, end = pd.Timestamp(fam["start"]), pd.Timestamp(fam["end"])
    s = pd.Period(w["start"], "M").start_time
    e = pd.Period(w["end"], "M").end_time.normalize()
    pre0 = max(s - pd.DateOffset(months=K.SYN_PRE_MONTHS), start)
    post1 = min((pd.Period(w["end"], "M") + K.SYN_POST_MONTHS).end_time.normalize(), end)
    return {"mask": (s, e), "pre": (pre0, s - pd.Timedelta(days=1)),
            "post": (e + pd.Timedelta(days=1), post1)}


def family_windows(name: str) -> list[dict]:
    fam = FAMILIES[name]
    wins = gaps_guarded.ring_windows(fam["windows"][0], fam["windows"][1], K.SYN_LEN)
    out = []
    for wi, w in enumerate(wins):
        if fam["start_range"] and not (fam["start_range"][0] <= w["start"] <= fam["start_range"][1]):
            continue
        g = geometry(w, fam)
        if g["post"][0] > pd.Timestamp(fam["end"]):
            continue
        out.append({"wi": wi, **w, "geom": g})
    return out


def entries(pos_all: dict, g: dict, fam: dict) -> dict:
    ents = {}
    end = pd.Timestamp(fam["end"])
    for s, p in pos_all.items():
        p = p.loc[:end]
        if (SY.n_surveys(p, *g["pre"]) < fam["min_pre"]
                or SY.n_surveys(p, *g["post"]) < fam["min_post"]):
            continue
        keep = (p.index < g["mask"][0]) | (p.index > g["mask"][1])
        pt = p.loc[keep].loc[g["pre"][0]:g["post"][1]]
        try:
            L.gap_stats(pt, g["pre"], g["post"])
        except ValueError:
            continue
        ents[s] = pt
    return ents


def realization(ents, g, inj, method, piles, consts):
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
        if abs(gsc[s]["dS_obs"]) <= K.NEAR_THRESHOLD_M:
            forced.add(s)
    pre = {s: g["pre"] for s in pos}
    post = {s: g["post"] for s in pos}
    if method == "r3":
        res = R3.evaluate_r3(pos, pre, post, forced, consts["sigma_x_m"], consts["sigma_c_m"],
                             gs_cache=gsc)
    else:
        pl = {s: (piles[v[0]], v[1]) for s, v in K.PILES.items() if s in pos} if piles else {}
        pw = {s: (g["pre"], g["post"]) for s in pl}
        res = L.evaluate(pos, pre, post, pl, forced, pile_wins=pw, gs_cache=gsc)
    return res, gsc, truth, forced


def run_family(name: str, pos_all: dict, piles: dict | None, p_A: float, consts: dict,
               methods=("r3", "registered"), variants=("primary", "stress"),
               nreal: int = K.SYN_REAL) -> pd.DataFrame:
    fam = FAMILIES[name]
    rows = []
    t0 = time.time()
    for w in family_windows(name):
        g = w["geom"]
        ents = entries(pos_all, g, fam)
        if len(ents) < fam["min_sites"]:
            continue
        sites = sorted(ents)
        for k in range(nreal):
            r = fam["r0"] + 100 * w["wi"] + k
            for var in variants:
                inj = (SY.inject_groupcond if var == "primary" else SY.inject_harness)(
                    sites, r, p_A)
                for method in methods:
                    res, gsc, truth, forced = realization(ents, g, inj, method, piles, consts)
                    for d in inj:
                        s = d["site"]
                        rr, gs = res[s], gsc[s]
                        S = truth[s][gs["bot"]] - truth[s][gs["top"]]
                        C_true = float(S.loc[gs["d_lp"]] - S.loc[gs["d_fp"]])
                        pool = {g_: rr.members[g_]["pool"] for g_ in "AB"}
                        rows.append({
                            "family": name, "method": method, "variant": var,
                            "window": w["wi"], "start": w["start"], "end": w["end"], "k": k,
                            "r": r, "site": s, "n_sites": len(sites),
                            "true_group": d["group"], "b_true_ppm": 1e6 * d["b"],
                            "a_true_cm": 100 * d["a"],
                            "assigned": rr.assign["assigned"] or "",
                            "evidence_group": rr.assign["evidence_group"] or "",
                            "forced": s in forced, "p_A": rr.p_A,
                            "prior_A": rr.assign.get("prior_A", K.PRIOR_A),
                            "lines": ",".join(f"{n}:{v['vote'] or '-'}" for n, v in rr.lines.items()),
                            "lr_bdeep": rr.lines["b_deep"]["lr"],
                            "b_deep_ppm": 1e6 * gs["b_deep"], "deep_len_m": gs["deep_len"],
                            "C_true_cm": 100 * C_true, "C_primary_cm": 100 * rr.primary["C"],
                            "err_cm": 100 * (rr.primary["C"] - C_true),
                            "sigma_cm": 100 * rr.primary["sigma_C"],
                            "err_oracle_cm": 100 * (rr.members[d["group"]]["C"] - C_true),
                            "pool_A_n": pool["A"]["n"], "pool_B_n": pool["B"]["n"],
                            "fallback_A": bool(pool["A"]["fallback"]),
                            "fallback_B": bool(pool["B"]["fallback"]),
                            "b_A_ppm": 1e6 * rr.members["A"]["b"],
                            "b_B_ppm": 1e6 * rr.members["B"]["b"]})
        print(f"{name} window {w['wi']} {w['start']}: {len(sites)} sites "
              f"({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def rms(v) -> float:
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def summarize(x: pd.DataFrame) -> dict:
    """Statistics of one (family, method, variant): the registered definitions of
    synthetic.summarize plus fallback shares."""
    if x.empty:
        return {"n_windows": 0}
    per = x.groupby(["window", "start", "end"]).agg(
        n=("err_cm", "size"), n_sites=("site", "nunique"), rmse_cm=("err_cm", rms),
        bias_cm=("err_cm", "mean"), rmse_oracle_cm=("err_oracle_cm", rms)).reset_index()
    x = x.copy()
    x["mis"] = (x.assigned != "") & (x.assigned != x.true_group)
    x["una"] = x.assigned == ""
    per["misassigned"] = per["window"].map(x.groupby("window")["mis"].sum()).astype(int)
    worst = per.loc[per.rmse_cm.idxmax()]
    hits = (np.abs(x.err_cm) <= 1.645 * x.sigma_cm).to_numpy()
    cov = coverage3.coverage_gate(hits, x.site.to_numpy())
    assigned = x[~x.una]
    out = {
        "n_windows": int(per.shape[0]), "n_site_realizations": int(len(x)),
        "sites": sorted(x.site.unique().tolist()),
        "worst_window_rmse_m": float(worst.rmse_cm) / 100.0,
        "worst_window": {"window": int(worst.window), "start": worst.start, "end": worst.end,
                         "rmse_cm": float(worst.rmse_cm), "bias_cm": float(worst.bias_cm),
                         "rmse_oracle_cm": float(worst.rmse_oracle_cm), "n": int(worst.n),
                         "misassigned": int(worst.misassigned)},
        "n_windows_over_1p5cm": int((per.rmse_cm > THR_CM).sum()),
        "median_window_rmse_cm": float(per.rmse_cm.median()),
        "overall_rmse_cm": rms(x.err_cm), "overall_bias_cm": float(x.err_cm.mean()),
        "oracle_worst_window_rmse_cm": float(per.rmse_oracle_cm.max()),
        "oracle_n_windows_over_1p5cm": int((per.rmse_oracle_cm > THR_CM).sum()),
        "assigned_share": float(len(assigned) / len(x)),
        "misclassification_rate": float(x.mis.sum() / max(len(assigned), 1)),
        "misassigned_n": int(x.mis.sum()), "unassigned_share": float(x.una.mean()),
        "rmse_misassigned_cm": rms(x.loc[x.mis, "err_cm"]) if x.mis.any() else None,
        "rmse_unassigned_cm": rms(x.loc[x.una, "err_cm"]) if x.una.any() else None,
        "fallback_share_A": float(x.fallback_A.mean()),
        "fallback_share_B": float(x.fallback_B.mean()),
        "mean_pool_n": {"A": float(x.pool_A_n.mean()), "B": float(x.pool_B_n.mean())},
        "by_true_group": {g: {"n": int(len(v)), "rmse_cm": rms(v.err_cm),
                              "bias_cm": float(v.err_cm.mean()),
                              "rmse_oracle_cm": rms(v.err_oracle_cm)}
                          for g, v in x.groupby("true_group")},
        "coverage_90": cov,
        "per_window": per.round(4).to_dict(orient="records"),
    }
    out["pass"] = bool(out["worst_window_rmse_m"] <= K.WORST_WINDOW_RMSE_M)
    return out


def summaries(R: pd.DataFrame) -> dict:
    out = {}
    for (fam, meth, var), x in R.groupby(["family", "method", "variant"]):
        out[f"{fam}.{meth}.{var}"] = summarize(x)
    return out


# ----------------------------------------------------------------------------- inputs
def real_inputs(end: str):
    from s1_rings import s1io
    piles, notes = s1io.piles(preorigin=True)
    pos = {s: L.load_positions(s).loc[:end] for s in K.SITES}
    p_A = float(json.loads((K.EVIDENCE / "assignment.json").read_text())["p_A_rule"]["p_A"])
    return pos, piles, p_A, notes


def fake_inputs():
    """Toy sites on a 2003-2012 calendar (no real data) for the smoke test."""
    from s1_rings.tests.test_s1lib import fake_site
    pos = {}
    for i in range(10):
        tr, _ = fake_site(b=0.0, a=0.0, seed=100 + i, gap=("2030-01-01", "2030-01-02"))
        tr.index = tr.index - pd.DateOffset(years=11)
        pos[f"fake{i:02d}"] = tr.loc[:"2012-02-29"]
    return pos


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fresh", "spent", "fake"])
    ap.add_argument("--nreal", type=int, default=K.SYN_REAL)
    a = ap.parse_args(argv)
    consts = json.loads(CALIB.read_text())
    if a.what == "fake":
        pos = fake_inputs()
        R = run_family("pre2012", pos, None, 5 / 7, consts, nreal=2)
        S = summaries(R)
        print(json.dumps({k: {kk: v.get(kk) for kk in ("n_windows", "worst_window_rmse_m",
                                                         "fallback_share_A", "unassigned_share")}
                          for k, v in S.items()}, indent=1))
        return
    from harness import prereg3
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "fresh":
        chk = prereg3.assert_can_score(PREREG_ID, "S1a")
        pos, piles, p_A, notes = real_inputs("2021-12-31")
        R1 = run_family("pre2012", {s: p.loc[:"2012-02-29"] for s, p in pos.items()}, piles,
                        p_A, consts, nreal=a.nreal)
        R2 = run_family("late2019", pos, piles, p_A, consts, nreal=a.nreal)
        R = pd.concat([R1, R2], ignore_index=True)
        R.to_parquet(OUT / "fresh_rows.parquet", index=False)
        S = summaries(R)
        g = S.get("pre2012.r3.primary", {})
        result = {"prereg": PREREG_ID, "prereg_sha256": chk["sha256"], "p_A": p_A,
                  "constants": {k: consts[k] for k in ("sigma_x_m", "sigma_c_m")},
                  "guard_notes": notes, "post_registration_looks": chk["post_registration_looks"],
                  "unverifiable_prior_looks": chk["unverifiable_prior_looks"],
                  "gating": {"test": "pre2012.r3.primary", "threshold_m": K.WORST_WINDOW_RMSE_M,
                             "threshold_status": "post hoc (plan), unchanged",
                             "worst_window_rmse_m": g.get("worst_window_rmse_m"),
                             "pass": g.get("pass")},
                  "summaries": S}
        (OUT / "fresh_result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False,
                                                          default=str))
        for k, v in S.items():
            print(k, {kk: v.get(kk) for kk in ("n_windows", "worst_window_rmse_m", "pass",
                                               "n_windows_over_1p5cm", "median_window_rmse_cm",
                                               "oracle_worst_window_rmse_cm")})
    else:
        from s1_rings.r3_looks import log_second_look_e
        look = log_second_look_e()
        pos, piles, p_A, notes = real_inputs("2021-12-31")
        R = run_family("spent", pos, piles, p_A, consts, methods=("r3",), nreal=a.nreal)
        R.to_parquet(OUT / "spent_second_look_rows.parquet", index=False)
        S = summaries(R)
        result = {"status": "SECOND LOOK on spent units (S1a.e_synthetic_mixed.v1 windows and "
                            "seeds); conditional, never gating",
                  "look": {k: look.get(k) for k in ("time", "purpose")},
                  "constants": {k: consts[k] for k in ("sigma_x_m", "sigma_c_m")},
                  "summaries": S}
        (OUT / "spent_second_look_result.json").write_text(
            json.dumps(result, indent=1, ensure_ascii=False, default=str))
        for k, v in S.items():
            print(k, {kk: v.get(kk) for kk in ("n_windows", "worst_window_rmse_m", "pass",
                                               "n_windows_over_1p5cm", "median_window_rmse_cm",
                                               "oracle_worst_window_rmse_cm")})


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
