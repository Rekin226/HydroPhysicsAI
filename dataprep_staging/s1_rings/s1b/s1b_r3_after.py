"""WP S1b repair round r3 (S1br): the real second-look counts read against the fresh synthetic
operating characteristics, and per-site after-the-fact diagnostics.

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_after.py

Logs an ``after_the_fact`` look, then reads only this round's evidence
(``repair_r3/retest/second_look.json``, ``repair_r3/synthetic/oc_*.json``) and the round-1
registered rows (for the per-site change against round 1). As declared in
``S1b.r3.rings_v3_second_look.v1``, for every reading, ring-noise model and family it reports
the probability of the observed count under each scenario, P(N = n_obs | s) and
P(N <= n_obs | s), and the likelihood ratios correct : +-25 % and mixture : +-25 %. Zero
synthetic frequencies are reported as "< 1/N" (upper bound 1/N in the ratio's denominator is
NOT used; the ratio is reported as None). Writes ``repair_r3/retest/after.json``.
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
from s1_rings.s1b.s1b_run import dump  # noqa: E402

RT = K.EVIDENCE / "repair_r3" / "retest"
SYN = K.EVIDENCE / "repair_r3" / "synthetic"
MODELS = ("N1_sd_s", "N2_resid")
SCEN = ("correct", "up25", "down25", "up25A", "down25A", "other", "raw", "mix", "mix_drawn")


def real_counts(sl: dict) -> dict:
    """Observed containment counts of the primary value (and raw) per reading for the three
    synthetic families: wra (primary run), wra_noSJES (the SJES-excluded variant) and InSAR."""
    p = sl["primary"]["families"]
    q = sl["variants"]["plan_flagged_SJES_excluded"]["families"]
    out = {}
    for fam, src in (("wra", p["c_wra"]), ("wra_noSJES", q["c_wra"]), ("insar", p["d_insar"])):
        out[fam] = {rule: {"n": src["rules"][rule]["primary"]["n"],
                           "count": src["rules"][rule]["primary"]["n_contain"],
                           "raw_count": src["rules"][rule]["raw"]["n_contain"]}
                    for rule in R.RULES}
    return out


def probs(hist: list[int], n_obs: int) -> dict:
    h = np.asarray(hist, float)
    tot = h.sum()
    eq = float(h[n_obs] / tot) if 0 <= n_obs < len(h) else 0.0
    le = float(h[: n_obs + 1].sum() / tot)
    return {"p_eq": eq, "p_le": le, "n_real": int(tot)}


def ratio(a: float, b: float) -> float | None:
    return None if b == 0 else float(a / b)


def lr_table(rc: dict, oc: dict) -> dict:
    out = {}
    for fam in ("wra", "wra_noSJES", "insar"):
        res = oc["families"][fam]
        out[fam] = {}
        for rule in R.RULES:
            obs = rc[fam][rule]["count"]
            n_syn = res[f"{rule}|correct"]["n_sites"]
            row = {"n_obs": obs, "n_sites_real": rc[fam][rule]["n"], "n_sites_synthetic": n_syn}
            for sc in SCEN:
                pr = probs(res[f"{rule}|{sc}"]["count_hist"], obs)
                row[sc] = {**pr, "mean_count": res[f"{rule}|{sc}"]["mean_count"],
                           "family_pass": res[f"{rule}|{sc}"]["family_pass"]}
            for alt in ("up25", "down25", "up25A", "down25A"):
                row[f"LR_correct_vs_{alt}"] = ratio(row["correct"]["p_eq"], row[alt]["p_eq"])
                row[f"LR_mixdrawn_vs_{alt}"] = ratio(row["mix_drawn"]["p_eq"], row[alt]["p_eq"])
            out[fam][rule] = row
    return out


def per_site_change(sl: dict) -> list[dict]:
    """Per WRA site: round 1 (v2 product, v2 validators) against this round (v3, v4/v3)."""
    r1 = pd.read_csv(K.EVIDENCE / "registered_rows.csv")
    r1 = r1[r1.test == "c_wra"].set_index("station")
    rows = []
    for r in sl["primary"]["rows"]["c_wra"]:
        st = r["station"]
        o = {"site_en": r["site_en"], "station": st, "evaluable": r.get("evaluable")}
        if r.get("evaluable"):
            L = 1000.0 * r["L_m"]
            o.update({k: r.get(k) for k in ("C_primary_mm", "C_A_mm", "C_B_mm", "C_raw_mm",
                                            "C_pre_mm", "C_post_mm", "sigma_pre_mm",
                                            "sigma_post_mm", "p_A", "G_mm")})
            o["min_k_primary"] = R.min_k(r["C_primary_mm"], r["C_pre_mm"], r["C_post_mm"],
                                         r["sigma_pre_mm"], r["sigma_post_mm"])
            for rule in R.RULES:
                o[f"in_{rule}"] = R.contains(rule, r["C_primary_mm"], r["C_pre_mm"],
                                             r["C_post_mm"], r["sigma_pre_mm"],
                                             r["sigma_post_mm"], L)
                o[f"A_in_{rule}"] = R.contains(rule, r["C_A_mm"], r["C_pre_mm"],
                                               r["C_post_mm"], r["sigma_pre_mm"],
                                               r["sigma_post_mm"], L)
                o[f"B_in_{rule}"] = R.contains(rule, r["C_B_mm"], r["C_pre_mm"],
                                               r["C_post_mm"], r["sigma_pre_mm"],
                                               r["sigma_post_mm"], L)
            if st in r1.index:
                q = r1.loc[st]
                o["round1"] = {"C_primary_mm": float(q.C_primary_mm),
                               "C_pre_mm": float(q.C_pre_mm), "C_post_mm": float(q.C_post_mm),
                               "G_mm": float(q.G_mm)}
                o["delta_C_primary_mm"] = r["C_primary_mm"] - float(q.C_primary_mm)
                o["delta_C_pre_mm"] = r["C_pre_mm"] - float(q.C_pre_mm)
                o["delta_C_post_mm"] = r["C_post_mm"] - float(q.C_post_mm)
                o["delta_G_mm"] = r["G_mm"] - float(q.G_mm)
        else:
            o["reason"] = r.get("reason")
        rows.append(o)
    return rows


def main() -> int:
    from harness import prereg3

    from s1_rings.s1b import s1b_r3_looks as LK
    doc, _ = prereg3.load_prereg(LK.DECL_ID)
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise SystemExit(f"declared inputs or code changed since {LK.DECL_ID}: {bad}")
    LK.after("lr_and_site_diagnostics")
    sl = json.loads((RT / "second_look.json").read_text())
    rc = real_counts(sl)
    out = {"real_counts": rc, "lr": {}, "per_site_wra": per_site_change(sl)}
    for m in MODELS:
        oc = json.loads((SYN / f"oc_{m}.json").read_text())
        out["lr"][m] = lr_table(rc, oc)
    dump(out, RT / "after.json")
    for m in MODELS:
        print("model", m)
        for fam in ("wra", "wra_noSJES"):
            for rule in ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:30", "mid_ppm:40",
                         "k:1", "k:2"):
                row = out["lr"][m][fam][rule]
                print(f"  {fam:10s} {rule:13s} obs {row['n_obs']}/{row['n_sites_real']} "
                      + " ".join(f"{sc}:{row[sc]['mean_count']:.2f}/{row[sc]['p_eq']:.3f}"
                                 for sc in ("correct", "up25", "down25", "up25A", "mix_drawn"))
                      + f" LRc/u {row['LR_correct_vs_up25']} LRc/d {row['LR_correct_vs_down25']}"
                      + f" LRm/u {row['LR_mixdrawn_vs_up25']}")
    for o in out["per_site_wra"]:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in o.items()
               if k in ("site_en", "station", "C_primary_mm", "C_pre_mm", "C_post_mm",
                        "delta_C_primary_mm", "delta_C_pre_mm", "delta_C_post_mm",
                        "delta_G_mm", "min_k_primary", "in_k:0", "in_range_ppm:40")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
