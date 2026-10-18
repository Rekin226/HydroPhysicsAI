"""Check (e): mixed A/B synthetic pseudo-gaps (test S1a.e_synthetic_mixed.v1, family R).

Every 13-month window of harness.gaps_guarded.ring_windows('2015-01', '2021-12', 13) is
masked in the ring surveys; a mixed A/B instrument change z -> a + (1 + b) z is injected into
every survey after it; the product pipeline (s1lib.evaluate: evidence lines, assignment, LOO
pooling, members, membership probability, primary) runs on the injected surveys; the error of
the primary bridged gap compaction against the uninjected truth is scored per window.

Pre-origin data only: ring surveys <= 2021-12-31 and pile rows of the deep_benchmarks/v2
pre-origin run (design_time_evidence through the 2022-12 slice).

    tools/cpu_slot.sh python -m s1_rings.synthetic [--windows N] [--fake]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import coverage3, gaps, gaps_guarded  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402

PREREG_ID = "S1a.e_synthetic_mixed.v1"
CLUSTERS = {"A": (-0.163, -0.155), "B": (0.50, 0.54)}   # harness ranges
Q_CLUSTER = 11.0 / 14.0
END = pd.Timestamp("2021-12-31")
START = pd.Timestamp("2014-05-01")


def inject_groupcond(sites, r: int, p_A: float) -> list[dict]:
    """Primary injection: offsets from the own group's cluster with probability 11/14."""
    rng = gaps.rng_for("R", r)
    out = []
    for s in sites:
        grp = "A" if rng.random() < p_A else "B"
        m, sd = K.REF[grp]
        b = float(rng.normal(m, sd))
        u = rng.random()
        lo, hi = CLUSTERS[grp] if u < Q_CLUSTER else K.OFFSET_OTHER
        a = float(rng.uniform(lo, hi))
        out.append({"site": s, "group": grp, "b": b, "a": a})
    return out


def inject_harness(sites, r: int, p_A: float) -> list[dict]:
    return gaps.ring_instrument_injection(list(sites), r, p_A=p_A)


def window_geometry(w: dict) -> dict:
    s = pd.Period(w["start"], "M").start_time
    e = pd.Period(w["end"], "M").end_time.normalize()
    pre0 = max(s - pd.DateOffset(months=K.SYN_PRE_MONTHS), START)
    post1 = min((pd.Period(w["end"], "M") + K.SYN_POST_MONTHS).end_time.normalize(), END)
    return {"mask": (s, e), "pre": (pre0, s - pd.Timedelta(days=1)),
            "post": (e + pd.Timedelta(days=1), post1)}


def n_surveys(p, a, b):
    return int(p.loc[a:b].dropna(how="all").shape[0])


def run(pos_all: dict, piles: dict, p_A: float, wins: list, nreal: int, variants) -> dict:
    rows = []
    for wi, w in enumerate(wins):
        g = window_geometry(w)
        ents = {}
        for s, p in pos_all.items():
            p = p.loc[:END]
            if n_surveys(p, *g["pre"]) < K.SYN_MIN_PRE or n_surveys(p, *g["post"]) < K.SYN_MIN_POST:
                continue
            keep = (p.index < g["mask"][0]) | (p.index > g["mask"][1])
            pt = p.loc[keep].loc[g["pre"][0]:g["post"][1]]
            try:
                L.gap_stats(pt, g["pre"], g["post"])
            except ValueError:
                continue
            ents[s] = pt
        if len(ents) < K.SYN_MIN_SITES:
            continue
        sites = sorted(ents)
        for k in range(nreal):
            r = 100 * wi + k
            for var in variants:
                inj = (inject_groupcond if var == "primary" else inject_harness)(sites, r, p_A)
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
                pl = {s: (piles[v[0]], v[1]) for s, v in K.PILES.items() if s in pos}
                pw = {s: (g["pre"], g["post"]) for s in pl}
                res = L.evaluate(pos, {s: g["pre"] for s in pos}, {s: g["post"] for s in pos},
                                 pl, forced, pile_wins=pw, gs_cache=gsc)
                for d in inj:
                    s = d["site"]
                    rr, gs = res[s], gsc[s]
                    S = truth[s][gs["bot"]] - truth[s][gs["top"]]
                    C_true = float(S.loc[gs["d_lp"]] - S.loc[gs["d_fp"]])
                    err = rr.primary["C"] - C_true
                    rows.append({
                        "variant": var, "window": wi, "start": w["start"], "end": w["end"],
                        "k": k, "r": r, "site": s, "true_group": d["group"],
                        "b_true_ppm": 1e6 * d["b"], "a_true_cm": 100 * d["a"],
                        "assigned": rr.assign["assigned"] or "",
                        "evidence_group": rr.assign["evidence_group"] or "",
                        "forced": s in forced, "votes_A": rr.assign["votes_A"],
                        "votes_B": rr.assign["votes_B"], "p_A": rr.p_A,
                        "lines": ",".join(f"{n}:{v['vote'] or '-'}"
                                          for n, v in rr.lines.items()),
                        "C_true_cm": 100 * C_true, "C_primary_cm": 100 * rr.primary["C"],
                        "err_cm": 100 * err, "sigma_cm": 100 * rr.primary["sigma_C"],
                        "err_oracle_cm": 100 * (rr.members[d["group"]]["C"] - C_true),
                        "b_deep_ppm": 1e6 * gs["b_deep"] if np.isfinite(gs["b_deep"]) else np.nan,
                        "J_top_cm": 100 * gs["J_top_mid"],
                        "b_pile_ppm": (1e6 * rr.pile["b"]) if rr.pile and rr.pile.get("ok")
                        else np.nan})
        print(f"window {wi} {w['start']}..{w['end']}: {len(ents)} sites", flush=True)
    return {"rows": rows}


def summarize(R: pd.DataFrame, var: str) -> dict:
    x = R[R.variant == var].copy()
    if x.empty:
        return {}
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    per = x.groupby(["window", "start", "end"]).agg(
        n=("err_cm", "size"), n_sites=("site", "nunique"),
        rmse_cm=("err_cm", rm), bias_cm=("err_cm", "mean"),
        rmse_oracle_cm=("err_oracle_cm", rm),
        misassigned=("site", lambda s: 0)).reset_index()
    x["mis"] = (x.assigned != "") & (x.assigned != x.true_group)
    x["una"] = x.assigned == ""
    mis_w = x.groupby("window")["mis"].sum()
    per["misassigned"] = per["window"].map(mis_w).astype(int)
    worst = per.loc[per.rmse_cm.idxmax()]
    hits = (np.abs(x.err_cm) <= 1.645 * x.sigma_cm).to_numpy()
    cov = coverage3.coverage_gate(hits, x.site.to_numpy())
    assigned = x[x.assigned != ""]
    by_true = x.groupby("true_group").agg(n=("err_cm", "size"), rmse_cm=("err_cm", rm),
                                          unassigned=("una", "mean"), misassigned=("mis", "mean"))
    out = {
        "n_windows": int(per.shape[0]), "n_site_realizations": int(len(x)),
        "worst_window_rmse_m": float(worst.rmse_cm) / 100.0,
        "worst_window": {"window": int(worst.window), "start": worst.start, "end": worst.end,
                         "rmse_cm": float(worst.rmse_cm), "bias_cm": float(worst.bias_cm),
                         "rmse_oracle_cm": float(worst.rmse_oracle_cm),
                         "n": int(worst.n), "misassigned": int(worst.misassigned)},
        "median_window_rmse_cm": float(per.rmse_cm.median()),
        "overall_rmse_cm": rm(x.err_cm), "overall_bias_cm": float(x.err_cm.mean()),
        "oracle_worst_window_rmse_cm": float(per.rmse_oracle_cm.max()),
        "assigned_share": float(len(assigned) / len(x)),
        "misclassification_rate": float(x.mis.sum() / max(len(assigned), 1)),
        "misassigned_n": int(x.mis.sum()), "unassigned_share": float(x.una.mean()),
        "rmse_misassigned_cm": rm(x.loc[x.mis, "err_cm"]) if x.mis.any() else None,
        "rmse_correctly_assigned_cm": rm(assigned.loc[~assigned.mis, "err_cm"]),
        "rmse_unassigned_cm": rm(x.loc[x.una, "err_cm"]) if x.una.any() else None,
        "by_true_group": by_true.round(4).to_dict(orient="index"),
        "coverage_90": cov,
        "per_window": per.round(4).to_dict(orient="records"),
    }
    out["pass"] = bool(out["worst_window_rmse_m"] <= K.WORST_WINDOW_RMSE_M)
    return out


def fake_inputs():
    """Toy sites for a dry run of the driver (no real data)."""
    from s1_rings.tests.test_s1lib import fake_site
    pos = {}
    for i in range(12):
        tr, _ = fake_site(b=0.0, a=0.0, seed=i, gap=("2030-01-01", "2030-01-02"))
        pos[f"fake{i:02d}"] = tr.loc[:"2021-12-31"]
    return pos, {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--nreal", type=int, default=K.SYN_REAL)
    args = ap.parse_args(argv)
    wins = gaps_guarded.ring_windows(K.SYN_WINDOW[0], K.SYN_WINDOW[1], K.SYN_LEN)
    if args.windows:
        wins = wins[:args.windows]
    if args.fake:
        pos, piles = fake_inputs()
        res = run(pos, piles, 10 / 14, wins, args.nreal, ("primary", "stress"))
        R = pd.DataFrame(res["rows"])
        print(json.dumps({v: {k: summarize(R, v).get(k) for k in
                              ("worst_window_rmse_m", "misclassification_rate",
                               "unassigned_share", "n_windows")}
                          for v in ("primary", "stress")}, indent=1))
        return
    from harness import prereg3

    from s1_rings import s1io
    chk = prereg3.assert_can_score(PREREG_ID, "S1a")
    assign = json.loads((K.EVIDENCE / "assignment.json").read_text())
    p_A = float(assign["p_A_rule"]["p_A"])
    piles, notes = s1io.piles(preorigin=True)
    pos = {s: L.load_positions(s).loc[:END] for s in K.SITES}
    res = run(pos, piles, p_A, wins, args.nreal, ("primary", "stress"))
    R = pd.DataFrame(res["rows"])
    out = K.EVIDENCE / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    R.to_parquet(out / "synthetic_rows.parquet", index=False)
    result = {"prereg": PREREG_ID, "prereg_sha256": chk["sha256"], "p_A": p_A,
              "n_windows_listed": len(wins), "guard_notes": notes,
              "post_registration_looks": chk["post_registration_looks"],
              "primary": summarize(R, "primary"), "stress": summarize(R, "stress"),
              "threshold_m": K.WORST_WINDOW_RMSE_M, "threshold_status": "post hoc (plan)"}
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False,
                                                default=str))
    for v in ("primary", "stress"):
        s = result[v]
        print(v, {k: s[k] for k in ("worst_window_rmse_m", "worst_window", "pass",
                                    "misclassification_rate", "unassigned_share",
                                    "oracle_worst_window_rmse_cm", "n_windows")})
        print("  coverage", {k: s["coverage_90"][k] for k in ("coverage", "n_eff",
                                                               "calibrated", "pass")})


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
