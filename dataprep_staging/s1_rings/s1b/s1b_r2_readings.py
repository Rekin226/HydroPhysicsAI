"""WP S1b repair round: after-the-fact readings from the registered rows (no product read).

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r2_readings.py

Logs an ``after_the_fact`` look first, then reads only ``S1b_evidence/registered_rows.csv`` and
computes, for every family:

1. the plan-literal readings of the (c) bracket (verifier major 1) and the registered k-sigma
   form for k = 0..2.5, for the bridged primary value and for the raw value, with the per-site
   minimum k and the family k* (the smallest k at which the 80 % rule would hold);
2. RMSE of primary and raw against the bracket midpoint (registered) and against each window's
   implied value C_pre and C_post separately (verifier minor 2);
3. for (d1): Williams' t for the dependent correlations r(I, bridged) vs r(I, raw), a site
   bootstrap of their difference and a Fisher interval of r(I, bridged) (verifier minor 1).

Writes ``S1b_evidence/repair/readings.json``. The registered result is not changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r2_looks as LK  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

OUT = K.EVIDENCE / "repair"
BOOT_SEED = 1804
N_BOOT = 10000


def load_rows() -> dict:
    df = pd.read_csv(K.EVIDENCE / "registered_rows.csv")
    df["evaluable"] = df["evaluable"].astype(str).str.lower().eq("true")
    out = {}
    for t, g in df.groupby("test"):
        out[t] = g.to_dict(orient="records")
    return out


def family_block(rows: list[dict]) -> dict:
    ev = [r for r in rows if r.get("evaluable")]
    rules = {}
    for rule in R.RULES:
        rules[rule] = {"primary": R.family_counts(rows, rule, "C_primary_mm"),
                       "raw": R.family_counts(rows, rule, "C_raw_mm")}
    per_site = []
    for r in ev:
        L = 1000.0 * r["L_m"]
        per_site.append({
            "site": r["site"], "site_en": r["site_en"], "station": r["station"],
            "C_primary_mm": r["C_primary_mm"], "C_raw_mm": r["C_raw_mm"],
            "C_pre_mm": r["C_pre_mm"], "C_post_mm": r["C_post_mm"],
            "C_mid_mm": r["C_mid_mm"], "sigma_pre_mm": r["sigma_pre_mm"],
            "sigma_post_mm": r["sigma_post_mm"],
            "range_half_width_mm": 0.5 * abs(r["C_pre_mm"] - r["C_post_mm"]),
            "range_half_width_ppm": 1e6 * 0.5 * abs(r["C_pre_mm"] - r["C_post_mm"]) / L,
            "bracket2_half_width_ppm": 1e6 * 0.5 * (r["hi_mm"] - r["lo_mm"]) / L,
            "primary_minus_mid_ppm": 1e6 * (r["C_primary_mm"] - r["C_mid_mm"]) / L,
            "min_k_primary": R.min_k(r["C_primary_mm"], r["C_pre_mm"], r["C_post_mm"],
                                     r["sigma_pre_mm"], r["sigma_post_mm"]),
            "min_k_raw": R.min_k(r["C_raw_mm"], r["C_pre_mm"], r["C_post_mm"],
                                 r["sigma_pre_mm"], r["sigma_post_mm"]),
            **{f"in_{rule}": R.contains(rule, r["C_primary_mm"], r["C_pre_mm"],
                                        r["C_post_mm"], r["sigma_pre_mm"],
                                        r["sigma_post_mm"], L) for rule in R.RULES}})
    rm = {f"{key}_vs_{ref}": R.rmse(rows, f"C_{key}_mm", f"C_{ref}_mm")
          for key in ("primary", "raw") for ref in ("mid", "pre", "post")}
    rp = [p["range_half_width_ppm"] for p in per_site]
    return {
        "n_evaluable": len(ev), "rules": rules, "per_site": per_site, "rmse_mm": rm,
        "rmse_rule_holds": {ref: bool(rm[f"primary_vs_{ref}"] < rm[f"raw_vs_{ref}"])
                            for ref in ("mid", "pre", "post")},
        "k_star_primary": R.k_star([p["min_k_primary"] for p in per_site]),
        "k_star_raw": R.k_star([p["min_k_raw"] for p in per_site]),
        "range_half_width_ppm": {"min": min(rp) if rp else None,
                                 "median": float(np.median(rp)) if rp else None,
                                 "max": max(rp) if rp else None}}


def d1_block(rows: list[dict], use_change: bool = False) -> dict:
    rr = [r for r in rows if np.isfinite(r.get("I_gap_mm", np.nan))]
    f = (lambda r: 1.0) if use_change else (lambda r: 1.0 / r["gap_yr"])
    x = np.array([r["I_gap_mm"] * f(r) for r in rr])
    yb = np.array([r["C_primary_mm"] * f(r) for r in rr])
    yr = np.array([r["C_raw_mm"] * f(r) for r in rr])
    n = len(rr)
    r12, r13 = np.corrcoef(x, yb)[0, 1], np.corrcoef(x, yr)[0, 1]
    r23 = np.corrcoef(yb, yr)[0, 1]
    return {"n": n, "r_bridged": float(r12), "r_raw": float(r13),
            "r_bridged_raw": float(r23),
            "williams": R.williams_t(float(r12), float(r13), float(r23), n),
            "bootstrap_diff": R.corr_diff_bootstrap(x, yb, yr, N_BOOT, BOOT_SEED),
            "fisher90_r_bridged": R.fisher_ci(float(r12), n, 0.90),
            "fisher90_r_raw": R.fisher_ci(float(r13), n, 0.90)}


def main() -> int:
    LK.readings()
    rows = load_rows()
    fams = {t: family_block(rows[t]) for t in ("c_wra", "c_ngl", "c_neighbour", "d_insar")}
    sj = [r for r in rows["c_wra"] if r["station"] != "SJES"]
    fams["c_wra_without_SJES"] = family_block(sj)
    out = {"source": "results/dataprep/reports/S1b_evidence/registered_rows.csv",
           "rules_doc": R.__doc__, "families": fams,
           "d1_rates": d1_block(rows["d_insar"]),
           "d1_changes": d1_block(rows["d_insar"], use_change=True),
           "boot_seed": BOOT_SEED, "n_boot": N_BOOT}
    # verdict of the registered conjunction under each reading (RMSE part unchanged)
    d1_pass = True    # registered d1 (unchanged by the bracket rule)
    ver = {}
    for rule in R.RULES:
        c = fams["c_wra"]["rules"][rule]["primary"]["containment_pass"] and \
            fams["c_wra"]["rmse_rule_holds"]["mid"]
        d2 = fams["d_insar"]["rules"][rule]["primary"]["containment_pass"] and \
            fams["d_insar"]["rmse_rule_holds"]["mid"]
        ver[rule] = {"c_wra": bool(c), "d2": bool(d2), "d1": d1_pass,
                     "verdict": bool(c and d2 and d1_pass)}
    out["verdict_by_reading"] = ver
    OUT.mkdir(parents=True, exist_ok=True)
    dump(out, OUT / "readings.json")
    for t in ("c_wra", "c_wra_without_SJES", "c_ngl", "c_neighbour", "d_insar"):
        f = fams[t]
        print(t, "n", f["n_evaluable"], "k*", round(f["k_star_primary"], 2),
              "k*raw", round(f["k_star_raw"], 2), "halfwidth ppm", f["range_half_width_ppm"])
        for rule in R.RULES:
            p, q = f["rules"][rule]["primary"], f["rules"][rule]["raw"]
            print(f"   {rule:14s} primary {p['n_contain']}/{p['n']} raw {q['n_contain']}/"
                  f"{q['n']} outside {[K.SITE_EN[s] for s in p['sites_outside']]}")
        print("   rmse", {k: round(v, 1) for k, v in f["rmse_mm"].items()})
    for k in ("d1_rates", "d1_changes"):
        d = out[k]
        print(k, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v)
                             for kk, v in d.items()}, default=str)[:900])
    print(json.dumps(ver, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
