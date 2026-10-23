"""WP S1b repair round: synthetic operating characteristics of the plan-literal readings.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r2_synth.py

After the registered scoring (logged ``integrity_check`` look, availability only). Re-uses the
round-1 generator of ``s1b_synth.py`` (imported read-only: same truth model, same noise, same
availability reads; every value synthetic) with a new seed, and keeps for each realization the
bracket ends ``C_pre, C_post`` and their sigmas, so that every bracket rule of
``s1b_r2_core.RULES`` can be evaluated on the same draws. Reports:

* per site: containment of the correct member (1 - size) and of the other member, the raw value
  and the +-25 % values, under each rule;
* per family (the registered 80 % rule plus RMSE(primary) < RMSE(raw) against the midpoint):
  pass rate per scenario and rule;
* (d1) on the InSAR family: the pass rate of each clause (r >= r_crit; r_bridged > r_raw) for
  the correct member and for the registered mixture.

Writes ``S1b_evidence/repair/synthetic_plan_literal.json``. It answers "would a correct product
pass the plan-literal reading at this noise level?"; it does not change the registered rule.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_core as C  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r2_looks as LK  # noqa: E402
from s1_rings.s1b import s1b_synth as S  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

SEED = 1803
N_REAL = 200
OUT = K.EVIDENCE / "repair"
SCEN = ("correct", "other", "raw", "up25", "down25", "mix")


def one(rng, kind: str, r: dict, avail: dict, grp: str) -> dict | None:
    """``s1b_synth.one`` with the bracket components kept (same draws in the same order)."""
    y_lp, y_fp = C.years([r["t_lp"]])[0], C.years([r["t_fp"]])[0]
    tr = S.truth(rng, y_lp, y_fp)
    c, raw = S.synth_ring(rng, tr, r, S.B_PPM[grp])
    mem = S.members(raw, r["L_mm"])
    other = "B" if grp == "A" else "A"
    dL = S.ALT_PPM * 1e-6 * r["L_mm"]
    Cv = {"correct": mem[grp], "other": mem[other], "raw": raw, "up25": mem[grp] + dL,
          "down25": mem[grp] - dL, "mix": r["p_A"] * mem["A"] + (1.0 - r["p_A"]) * mem["B"]}
    if kind == "gnss":
        t_all = C.years(S.DAYS)
        d_all = (S.ring_true(tr, t_all) + S.excess_true(tr, t_all)
                 + S.ar1(rng, len(S.DAYS), avail["sigma_mm"], avail["rho"]))
        if avail["var_steps"] > 0:
            d_all = d_all + (rng.normal(0, math.sqrt(avail["var_steps"]))
                             * (t_all >= 0.5 * (y_lp + y_fp)))
        d = pd.Series(d_all, index=S.DAYS).reindex(avail["t"]).to_numpy()
        res = C.instrument_vs_ring(
            ring_t=r["t"], ring_c=c, inst_t=avail["t"], inst_d=d, t_lp=r["t_lp"],
            t_fp=r["t_fp"], pre_win=K.PRE_WIN, post_win=K.POST_WIN,
            min_inst=K.MIN_GNSS_DAYS, min_ring=K.MIN_RING_SURVEYS, min_span=K.MIN_SPAN_DAYS,
            end_half_d=K.GNSS_END_HALF_D, end_min=K.GNSS_END_MIN,
            end_half_fallback=K.GNSS_END_HALF_D_FALLBACK, segment_intercepts=False)
        if not res["ok"]:
            return None
        f = (1 + avail["rho"]) / (1 - avail["rho"])
        var_end = avail["sigma_mm"] ** 2 * f * (1 / res["n_end_lp"] + 1 / res["n_end_fp"])
        chk = C.site_check(res["G_mm"], res["excess"], var_end, r["sd_s_mm"], r["L_mm"],
                           {"raw": raw}, k=K.TOL_SIGMA, var_steps=avail["var_steps"])
    else:
        t = C.years(avail["t"])
        seg = avail["seg"]
        d = (S.ring_true(tr, t) + S.excess_true(tr, t)
             + rng.normal(0, K.INSAR_EPOCH_SD_MM, len(t)))
        offs = np.concatenate([[0.0], np.cumsum(rng.normal(0, K.INSAR_BRIDGE_SD_MM,
                                                           int(seg.max()) + 1))])
        d = d + offs[seg]
        d[~avail["ok"]] = np.nan
        res = C.instrument_vs_ring(
            ring_t=r["t"], ring_c=c, inst_t=avail["t"], inst_d=d, inst_seg=seg,
            t_lp=r["t_lp"], t_fp=r["t_fp"], pre_win=K.PRE_WIN, post_win=K.POST_WIN,
            min_inst=K.MIN_INSAR_EPOCHS, min_ring=K.MIN_RING_SURVEYS,
            min_span=K.MIN_SPAN_DAYS, end_half_d=K.INSAR_END_HALF_D, end_min=K.INSAR_END_MIN,
            segment_intercepts=True)
        if not res["ok"]:
            return None
        var_end = K.INSAR_EPOCH_SD_MM ** 2 * (1 / res["n_end_lp"] + 1 / res["n_end_fp"])
        chk = C.site_check(res["G_mm"], res["excess"], var_end, r["sd_s_mm"], r["L_mm"],
                           {"raw": raw}, k=K.TOL_SIGMA, n_bridges=res["n_bridges"],
                           bridge_sd=K.INSAR_BRIDGE_SD_MM)
    return {"site": r["site"], "grp": grp, "L_mm": r["L_mm"], "Cv": Cv, "raw": raw,
            "G_mm": res["G_mm"], "gap_yr": y_fp - y_lp,
            "C_pre": chk["C_pre_mm"], "C_post": chk["C_post_mm"],
            "s_pre": chk["sigma_pre_mm"], "s_post": chk["sigma_post_mm"]}


def rows_for(o: dict, scen: str) -> dict:
    return {"evaluable": True, "site": o["site"], "C_primary_mm": o["Cv"][scen],
            "C_raw_mm": o["raw"], "C_pre_mm": o["C_pre"], "C_post_mm": o["C_post"],
            "C_mid_mm": 0.5 * (o["C_pre"] + o["C_post"]), "sigma_pre_mm": o["s_pre"],
            "sigma_post_mm": o["s_post"], "L_m": o["L_mm"] / 1000.0}


def family_pass(rows: list[dict], rule: str) -> bool:
    fc = R.family_counts(rows, rule, "C_primary_mm", K.FAMILY_MIN_PASS_FRAC)
    return bool(fc["containment_pass"]
                and R.rmse(rows, "C_primary_mm", "C_mid_mm") < R.rmse(rows, "C_raw_mm",
                                                                       "C_mid_mm"))


def d1_clauses(items: list[dict], scen: str) -> dict | None:
    if len(items) < 3:
        return None
    x = np.array([o["G_mm"] / o["gap_yr"] for o in items])
    yb = np.array([o["Cv"][scen] / o["gap_yr"] for o in items])
    yr = np.array([o["raw"] / o["gap_yr"] for o in items])
    rb, rr = np.corrcoef(x, yb)[0, 1], np.corrcoef(x, yr)[0, 1]
    rc = C.r_crit(len(items), K.CORR_ALPHA)
    return {"crit": bool(rb >= rc), "gt_raw": bool(rb > rr), "both": bool(rb >= rc and rb > rr),
            "r_b": float(rb), "r_r": float(rr)}


def main() -> int:
    LK.synth()
    rng = np.random.default_rng(SEED)
    rings, prs, gav, iav = S.availability()
    out = {"seed": SEED, "realizations": N_REAL, "rules": list(R.RULES),
           "plan_literal_rules": list(R.PLAN_LITERAL), "scenarios": list(SCEN),
           "truth": "the product's more probable group at every site (P(A) >= 0.5 -> A)",
           "per_site": [], "families": {}}
    for fam in ("wra_colocated", "ngl_near", "neighbour", "insar"):
        items = ([(p["site"], p["station"], gav[(fam, p["site"], p["station"])])
                  for p in prs[fam]] if fam != "insar" else
                 [(s, "069A", iav[s]) for s in K.SITES])
        per_real = {k: [] for k in range(N_REAL)}
        for site, sta, av in items:
            r = rings[site]
            grp = "A" if r["p_A"] >= 0.5 else "B"
            got = []
            for k in range(N_REAL):
                o = one(rng, "gnss" if fam != "insar" else "insar", r, av, grp)
                if o is None:
                    continue
                got.append(o)
                per_real[k].append(o)
            row = {"family": fam, "site": site, "station": sta, "group": grp,
                   "n_ok": len(got)}
            for rule in R.RULES:
                for sc in ("correct", "other", "raw", "up25", "down25"):
                    v = [R.contains(rule, o["Cv"][sc], o["C_pre"], o["C_post"], o["s_pre"],
                                    o["s_post"], o["L_mm"]) for o in got]
                    row[f"{rule}|{sc}"] = float(np.mean(v)) if v else None
            out["per_site"].append(row)
        fr = {}
        for rule in R.RULES:
            for sc in SCEN:
                ps = [family_pass([rows_for(o, sc) for o in v], rule)
                      for v in per_real.values() if v]
                fr[f"{rule}|{sc}"] = float(np.mean(ps)) if ps else None
        fr["n_realizations"] = sum(1 for v in per_real.values() if v)
        if fam == "insar":
            for sc in ("correct", "mix"):
                cl = [d1_clauses(v, sc) for v in per_real.values()]
                cl = [c for c in cl if c]
                fr[f"d1|{sc}"] = {k: float(np.mean([c[k] for c in cl]))
                                  for k in ("crit", "gt_raw", "both")}
                fr[f"d1|{sc}"]["median_r_b"] = float(np.median([c["r_b"] for c in cl]))
                fr[f"d1|{sc}"]["median_r_r"] = float(np.median([c["r_r"] for c in cl]))
        out["families"][fam] = fr
    OUT.mkdir(parents=True, exist_ok=True)
    dump(out, OUT / "synthetic_plan_literal.json")
    pd.set_option("display.width", 250)
    for fam, fr in out["families"].items():
        print(fam, "n", fr["n_realizations"])
        for rule in R.RULES:
            print(f"   {rule:14s}", {sc: fr[f"{rule}|{sc}"] for sc in SCEN})
        for sc in ("correct", "mix"):
            if f"d1|{sc}" in fr:
                print("   d1", sc, fr[f"d1|{sc}"])
    ps = pd.DataFrame(out["per_site"])
    cols = ["family", "site", "group"] + [f"{r}|correct" for r in R.RULES]
    print(ps[cols].round(3).to_string())
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
