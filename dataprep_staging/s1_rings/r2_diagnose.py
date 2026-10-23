"""Repair round (r2) of WP S1a: after-the-fact decomposition of checks (e) and (a).

The verifier of rings_bridged/v1 found that the v1 report attributed the failure of check (e)
"mainly" to the pooled deep-interval scale, while its own synthetic rows show that the
membership / mixture handling contributes more. This module recomputes the diagnosis from the
registered rows (``S1a_evidence/synthetic/synthetic_rows.parquet``) and decomposes:

* each row's error into the **scale component** (the error of the member of the TRUE group,
  ``err_oracle``: pooled LOO scale minus the injected scale, times the interval length) and the
  **membership component** (``err - err_oracle``: what the assignment, the near-threshold
  forcing rule and the P-weighted mixture add);
* each window's MSE into scale, membership and cross terms;
* the membership component by assignment status, true group and the b_deep line's vote;
* counterfactual primaries (AFTER THE FACT, NOT TESTS; never used to change a rule of v1 or
  v2): the true group (oracle), the more probable member instead of the mixture, no
  near-threshold forcing, and the mixture with the membership prior equal to the injected
  base rate. The other member's value is reconstructed exactly from the stored mixture:
  for a mixture row ``err - err_oracle = (1 - P)(C_B - C_A)`` (truly A) or
  ``P (C_A - C_B)`` (truly B); for an assigned row the wrong member differs from the true one
  by ``C_A - C_B`` taken from the same identity. Rows where the needed coefficient is below
  1e-6 keep their stored error (the counterfactual member then has weight below 1e-6 too).

Check (a): the declared power (0.721) is decomposed into its two causes: the per-site
alternative ``0.25 b_eff S`` (about zero at the B and B-leaning mixture sites) and the declared
endpoint term ``2 sd_s^2`` of the gap sigma.

Window coverage of (e): which of the 72 listed windows were scored and why the others were
not (site entry rules of the registered design).

Every run logs its looks in the A1 look log first (purpose ``diagnose``).

    tools/cpu_slot.sh python -m s1_rings.r2_diagnose        (from dataprep_staging/)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

V1 = K.REPO / "results" / "dataprep" / "rings_bridged" / "v1"
SYN = K.EVIDENCE / "synthetic"
OUT = K.EVIDENCE / "repair"
THR_CM = 100 * K.WORST_WINDOW_RMSE_M
EPS = 1e-6


# ----------------------------------------------------------------------------- helpers
def rms(v) -> float:
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def line_vote(lines: str, name: str) -> str:
    """Vote of evidence line ``name`` in the stored 'offset:A,b_deep:B,pile:-' string."""
    for tok in str(lines).split(","):
        k, _, v = tok.partition(":")
        if k == name:
            return v
    return ""


def status(row) -> str:
    if row["assigned"]:
        return "assigned_correct" if row["assigned"] == row["true_group"] else "assigned_wrong"
    return "forced" if row["forced"] else "unassigned"


def member_diff(x: pd.DataFrame) -> np.ndarray:
    """C_A - C_B (cm) per row, reconstructed from the stored primary and oracle errors;
    NaN where it is not identified (an assigned row of the true group, or a coefficient
    below EPS)."""
    m = (x["err_cm"] - x["err_oracle_cm"]).to_numpy(float)
    p = x["p_A"].to_numpy(float)
    ta = (x["true_group"] == "A").to_numpy()
    asg = x["assigned"].to_numpy(object)
    d = np.full(len(x), np.nan)
    mix = asg == ""
    # mixture rows: truly A -> m = (1-p)(C_B - C_A); truly B -> m = p (C_A - C_B)
    c = np.where(ta, 1.0 - p, p)
    ok = mix & (c > EPS)
    d[ok & ta] = -m[ok & ta] / c[ok & ta]
    d[ok & ~ta] = m[ok & ~ta] / c[ok & ~ta]
    # assigned to the wrong group: truly A assigned B -> m = C_B - C_A; truly B -> C_A - C_B
    wrong = (~mix) & (asg != x["true_group"].to_numpy(object))
    d[wrong & ta] = -m[wrong & ta]
    d[wrong & ~ta] = m[wrong & ~ta]
    return d


def counterfactual(x: pd.DataFrame, weight_A: np.ndarray) -> np.ndarray:
    """Error (cm) of a primary that puts weight ``weight_A`` on member A (1 - weight_A on B):
    err_oracle + (w_A - 1[true A]) (C_A - C_B). Rows whose C_A - C_B is not identified keep
    their stored error when the counterfactual weight equals the stored one, else the oracle
    error when the counterfactual puts all weight on the true group."""
    d = member_diff(x)
    ta = (x["true_group"] == "A").to_numpy().astype(float)
    eo = x["err_oracle_cm"].to_numpy(float)
    e = x["err_cm"].to_numpy(float)
    w = np.asarray(weight_A, float)
    out = eo + (w - ta) * d
    nd = ~np.isfinite(d)
    true_w = np.isclose(w, ta)
    out[nd & true_w] = eo[nd & true_w]
    out[nd & ~true_w] = e[nd & ~true_w]
    return out


def n_fallback(x: pd.DataFrame, weight_A: np.ndarray) -> int:
    """Rows whose C_A - C_B is not identified and whose counterfactual weight differs from
    both the stored weight and the true group's (their stored error is kept: approximate)."""
    d = member_diff(x)
    ta = (x["true_group"] == "A").to_numpy().astype(float)
    w = np.asarray(weight_A, float)
    return int((~np.isfinite(d) & ~np.isclose(w, ta) & ~np.isclose(w, stored_weight(x))).sum())


def stored_weight(x: pd.DataFrame) -> np.ndarray:
    asg = x["assigned"].to_numpy(object)
    p = x["p_A"].to_numpy(float)
    return np.where(asg == "A", 1.0, np.where(asg == "B", 0.0, p))


def window_table(x: pd.DataFrame, err_col: str) -> pd.DataFrame:
    g = x.groupby(["window", "start", "end"])[err_col]
    return pd.DataFrame({"rmse_cm": g.apply(rms), "bias_cm": g.mean(),
                         "n": g.size()}).reset_index()


def summarize_errors(x: pd.DataFrame, col: str) -> dict:
    t = window_table(x, col)
    w = t.loc[t.rmse_cm.idxmax()]
    return {"worst_window_rmse_cm": round(float(w.rmse_cm), 3),
            "worst_window": f"{w.start}..{w.end}", "worst_window_index": int(w.window),
            "worst_window_bias_cm": round(float(w.bias_cm), 3),
            "n_windows_over_1p5cm": int((t.rmse_cm > THR_CM).sum()),
            "n_windows": int(len(t)), "median_window_rmse_cm": round(float(t.rmse_cm.median()), 3),
            "overall_rmse_cm": round(rms(x[col]), 3)}


# ----------------------------------------------------------------------------- (e)
def decompose_e(R: pd.DataFrame, variant: str, p_inj: float) -> dict:
    x = R[R.variant == variant].copy().reset_index(drop=True)
    x["assigned"] = x["assigned"].fillna("")
    x["status"] = x.apply(status, axis=1)
    x["memb_cm"] = x["err_cm"] - x["err_oracle_cm"]
    x["bdeep_vote"] = x["lines"].map(lambda s: line_vote(s, "b_deep"))
    x["offset_vote"] = x["lines"].map(lambda s: line_vote(s, "offset"))
    # --- per window
    rows = []
    for (w, s, e), g in x.groupby(["window", "start", "end"]):
        o, m = g.err_oracle_cm.to_numpy(), g.memb_cm.to_numpy()
        ta = g[g.true_group == "A"]
        rows.append({
            "window": int(w), "start": s, "end": e, "n": int(len(g)),
            "rmse_cm": rms(g.err_cm), "bias_cm": float(g.err_cm.mean()),
            "rmse_scale_cm": rms(o), "bias_scale_cm": float(o.mean()),
            "rms_membership_cm": rms(m), "bias_membership_cm": float(m.mean()),
            "mse_scale": float(np.mean(o ** 2)), "mse_membership": float(np.mean(m ** 2)),
            "mse_cross": float(2 * np.mean(o * m)),
            "share_unassigned_or_forced": float((g.assigned == "").mean()),
            "share_forced": float(g.forced.mean()),
            "truly_A_bdeep_votes_B": int((ta.bdeep_vote == "B").sum()), "truly_A_n": int(len(ta)),
            "truly_A_mean_pA_when_mixed": float(ta.loc[ta.assigned == "", "p_A"].mean())
            if (ta.assigned == "").any() else None})
    W = pd.DataFrame(rows)
    worst = int(W.loc[W.rmse_cm.idxmax(), "window"])
    worst_scale = int(W.loc[W.rmse_scale_cm.idxmax(), "window"])
    # --- worst window by status x true group
    xw = x[x.window == worst]
    by = []
    for (st, tg), g in xw.groupby(["status", "true_group"]):
        by.append({"status": st, "true_group": tg, "n": int(len(g)),
                   "bias_cm": round(float(g.err_cm.mean()), 3),
                   "bias_scale_cm": round(float(g.err_oracle_cm.mean()), 3),
                   "bias_membership_cm": round(float(g.memb_cm.mean()), 3),
                   "rmse_cm": round(rms(g.err_cm), 3), "rmse_scale_cm": round(rms(g.err_oracle_cm), 3),
                   "mean_pA": round(float(g.p_A.mean()), 3),
                   "sum_sq_cm2": round(float(np.sum(g.err_cm ** 2)), 3)})
    # --- overall by status
    by_all = []
    for (st, tg), g in x.groupby(["status", "true_group"]):
        by_all.append({"status": st, "true_group": tg, "n": int(len(g)),
                       "share": round(len(g) / len(x), 4),
                       "bias_cm": round(float(g.err_cm.mean()), 3),
                       "bias_membership_cm": round(float(g.memb_cm.mean()), 3),
                       "rmse_cm": round(rms(g.err_cm), 3),
                       "rmse_scale_cm": round(rms(g.err_oracle_cm), 3),
                       "mean_pA": round(float(g.p_A.mean()), 3)})
    # --- b_deep line votes among truly-A rows
    ta = x[x.true_group == "A"]
    bvotes = {"overall": {"truly_A": int(len(ta)), "b_deep_votes_B": int((ta.bdeep_vote == "B").sum()),
                          "b_deep_votes_A": int((ta.bdeep_vote == "A").sum())},
              "worst_window": {"truly_A": int((xw.true_group == "A").sum()),
                               "b_deep_votes_B": int(((xw.true_group == "A")
                                                      & (xw.bdeep_vote == "B")).sum()),
                               "b_deep_votes_A": int(((xw.true_group == "A")
                                                      & (xw.bdeep_vote == "A")).sum())}}
    over = W[W.rmse_cm > THR_CM]
    bvotes["windows_over_threshold"] = {
        "truly_A": int(over.truly_A_n.sum()), "b_deep_votes_B": int(over.truly_A_bdeep_votes_B.sum())}
    # --- P(A) reliability of mixture rows (calibration of the membership probability)
    mix = x[x.assigned == ""].copy()
    bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0000001]
    mix["bin"] = pd.cut(mix.p_A, bins, right=False)
    rel = [{"p_A_bin": str(b), "n": int(len(g)), "mean_pA": round(float(g.p_A.mean()), 3),
            "share_truly_A": round(float((g.true_group == "A").mean()), 3)}
           for b, g in mix.groupby("bin", observed=True)]
    # --- counterfactuals (after the fact)
    w0 = stored_weight(x)
    p = x["p_A"].to_numpy(float)
    mixed = (x["assigned"] == "").to_numpy()
    forced = x["forced"].to_numpy(bool)
    evg = x["evidence_group"].fillna("").to_numpy(object)
    w_hard = np.where(mixed, (p >= 0.5).astype(float), w0)
    w_noforce = np.where(mixed & forced & (evg == "A"), 1.0,
                         np.where(mixed & forced & (evg == "B"), 0.0, w0))
    w_both = np.where(mixed & forced & (evg != ""), w_noforce, w_hard)
    lr = np.where(p >= 1 - 1e-15, np.inf, p / np.maximum(1 - p, 1e-300))
    p_prior = np.where(np.isinf(lr), 1.0, p_inj * lr / (p_inj * lr + (1 - p_inj)))
    w_prior = np.where(mixed, p_prior, w0)
    x["cf_stored"] = counterfactual(x, w0)
    x["cf_oracle"] = x["err_oracle_cm"]
    x["cf_hard"] = counterfactual(x, w_hard)
    x["cf_noforce"] = counterfactual(x, w_noforce)
    x["cf_hard_noforce"] = counterfactual(x, w_both)
    x["cf_prior_inj"] = counterfactual(x, w_prior)
    recon = float(np.nanmax(np.abs(x["cf_stored"] - x["err_cm"])))
    fallback = {k: n_fallback(x, w) for k, w in (("hard", w_hard), ("noforce", w_noforce),
                                                   ("hard_noforce", w_both),
                                                   ("prior_inj", w_prior))}
    dd = member_diff(x)
    cfs = {
        "stored (registered primary, check)": summarize_errors(x, "cf_stored"),
        "true group known (scale component only)": summarize_errors(x, "cf_oracle"),
        "more probable member instead of the mixture": summarize_errors(x, "cf_hard"),
        "no near-threshold forcing (evidence group where 2 lines agree)":
            summarize_errors(x, "cf_noforce"),
        "no forcing + more probable member": summarize_errors(x, "cf_hard_noforce"),
        f"mixture with prior P(A) = injected base rate {p_inj:.4f}":
            summarize_errors(x, "cf_prior_inj"),
    }
    return {
        "variant": variant, "n_rows": int(len(x)),
        "reconstruction_max_abs_err_cm": recon,
        "counterfactual_rows_approximated": fallback,
        "member_diff_C_A_minus_C_B_cm": {"n_identified": int(np.isfinite(dd).sum()),
                                          "median": round(float(np.nanmedian(dd)), 3),
                                          "p05": round(float(np.nanpercentile(dd, 5)), 3),
                                          "p95": round(float(np.nanpercentile(dd, 95)), 3)},
        "worst_window": W[W.window == worst].round(4).to_dict(orient="records")[0],
        "worst_scale_window": W[W.window == worst_scale].round(4).to_dict(orient="records")[0],
        "n_windows_over_threshold": {"actual": int((W.rmse_cm > THR_CM).sum()),
                                     "scale_component_only": int((W.rmse_scale_cm > THR_CM).sum()),
                                     "of": int(len(W))},
        "windows_scale_component_over_threshold": W.loc[W.rmse_scale_cm > THR_CM,
                                                        ["start", "end", "rmse_scale_cm"]]
        .round(3).to_dict(orient="records"),
        "mse_terms_overall": {"scale": float(np.mean(x.err_oracle_cm ** 2)),
                              "membership": float(np.mean(x.memb_cm ** 2)),
                              "cross": float(2 * np.mean(x.err_oracle_cm * x.memb_cm))},
        "worst_window_by_status": by, "by_status_overall": by_all,
        "b_deep_votes_truly_A": bvotes,
        "membership_probability_reliability": rel,
        "counterfactuals_after_the_fact": cfs,
        "per_window": W.round(4).to_dict(orient="records"),
    }


# ----------------------------------------------------------------------------- coverage
def window_coverage() -> dict:
    """Which listed windows the registered entry rules scored, and why not the rest."""
    from harness import gaps_guarded

    from s1_rings import s1lib as L
    from s1_rings import synthetic as S
    wins = gaps_guarded.ring_windows(K.SYN_WINDOW[0], K.SYN_WINDOW[1], K.SYN_LEN)
    pos = {s: L.load_positions(s).loc[:S.END] for s in K.SITES}
    rows, never = [], set(K.SITES)
    for wi, w in enumerate(wins):
        g = S.window_geometry(w)
        ents, why = [], {}
        for s, p in pos.items():
            npre, npost = S.n_surveys(p, *g["pre"]), S.n_surveys(p, *g["post"])
            if npre < K.SYN_MIN_PRE or npost < K.SYN_MIN_POST:
                why[s] = f"surveys pre {npre} (>= {K.SYN_MIN_PRE}) / post {npost} (>= {K.SYN_MIN_POST})"
                continue
            keep = (p.index < g["mask"][0]) | (p.index > g["mask"][1])
            try:
                L.gap_stats(p.loc[keep].loc[g["pre"][0]:g["post"][1]], g["pre"], g["post"])
            except ValueError as e:
                why[s] = f"gap_stats: {e}"
                continue
            ents.append(s)
        never -= set(ents)
        rows.append({"window": wi, "start": w["start"], "end": w["end"],
                     "n_sites": len(ents), "scored": len(ents) >= K.SYN_MIN_SITES,
                     "excluded": why})
    scored = [r for r in rows if r["scored"]]
    return {"n_listed": len(rows), "n_scored": len(scored),
            "scored_starts": f"{scored[0]['start']}..{scored[-1]['start']}" if scored else None,
            "unscored": [{k: r[k] for k in ("start", "end", "n_sites")} for r in rows
                         if not r["scored"]],
            "sites_never_entering": sorted(never),
            "per_window": rows}


# ----------------------------------------------------------------------------- (a)
def decompose_a() -> dict:
    """Power of check (a) under the declared sigma and alternative, and two after-the-fact
    variants: the alternative 0.25 x the site's group-A LOO scale at every site, and the
    declared alternative without the endpoint term."""
    from s1_rings import s1lib as L
    val = json.loads((V1 / "validation.json").read_text())
    C = pd.read_csv(V1 / "ring_corrections.csv").set_index("site")
    a = val["a_pooled_standardized_difference"]
    rows = []
    for r in a["sites"]:
        s = r["site"]
        p = L.load_positions(s)
        gs = L.gap_stats(p, K.PRE_WIN, K.POST_WIN_BEICHEN if s == K.BEICHEN else K.POST_WIN)
        smid = math.sqrt(((gs["C_hi"] - gs["C_lo"]) / math.sqrt(12)) ** 2
                         + gs["gap"] ** 2 * (gs["se_r_pre"] ** 2 + gs["se_r_post"] ** 2) / 4)
        sig = C.loc[s, "sigma_C_primary_cm"] / 100.0
        p_A = C.loc[s, "p_A"]
        asg = C.loc[s, "assigned"] if isinstance(C.loc[s, "assigned"], str) else None
        b_eff = (C.loc[s, "b_A_ppm"] if asg == "A" else C.loc[s, "b_B_ppm"] if asg == "B"
                 else p_A * C.loc[s, "b_A_ppm"] + (1 - p_A) * C.loc[s, "b_B_ppm"]) * 1e-6
        # sigma without the endpoint term: members' sigma_b L only, plus the mixture spread
        sA, sB = C.loc[s, "sigma_b_A_ppm"] * 1e-6 * gs["L"], C.loc[s, "sigma_b_B_ppm"] * 1e-6 * gs["L"]
        CA, CB, Cp = C.loc[s, "C_gap_A_cm"] / 100, C.loc[s, "C_gap_B_cm"] / 100, \
            C.loc[s, "C_gap_primary_cm"] / 100
        if asg:
            sig_noend = sA if asg == "A" else sB
        else:
            sig_noend = math.sqrt(p_A * (sA ** 2 + (CA - Cp) ** 2)
                                  + (1 - p_A) * (sB ** 2 + (CB - Cp) ** 2))
        rows.append({"site": s, "S_fp_m": gs["S_fp"], "s_declared": math.hypot(sig, smid),
                     "s_no_endpoint": math.hypot(sig_noend, smid),
                     "shift_declared": K.SCALE_ERROR * b_eff * gs["S_fp"],
                     "shift_A_scale": K.SCALE_ERROR * C.loc[s, "b_A_ppm"] * 1e-6 * gs["S_fp"],
                     "b_eff_ppm": 1e6 * b_eff})
    D = pd.DataFrame(rows)
    n = len(D)

    def power(shift, s):
        mu = float(np.sum(shift / s) / math.sqrt(n))
        return mu, float(norm.cdf(mu - K.POOLED_Z_MAX) + norm.cdf(-mu - K.POOLED_Z_MAX))

    out = {}
    for name, sh, s in [("declared (0.25 b_eff S; sigma with 2 sd_s^2)", D.shift_declared,
                         D.s_declared),
                        ("0.25 x group-A LOO scale at all 14 sites; declared sigma",
                         D.shift_A_scale, D.s_declared),
                        ("declared alternative; sigma without the endpoint term",
                         D.shift_declared, D.s_no_endpoint),
                        ("0.25 x group-A scale; sigma without the endpoint term",
                         D.shift_A_scale, D.s_no_endpoint)]:
        mu, pw = power(sh.to_numpy(), s.to_numpy())
        out[name] = {"mu": round(mu, 3), "power": round(pw, 3)}
    D2 = D.copy()
    for c in ("S_fp_m", "s_declared", "s_no_endpoint", "shift_declared", "shift_A_scale"):
        D2[c] = (D2[c] * (1 if c == "S_fp_m" else 100)).round(3)
    return {"declared_power_reported": a["power_25pct"], "variants_after_the_fact": out,
            "sites_cm": D2.rename(columns={"s_declared": "s_declared_cm",
                                           "s_no_endpoint": "s_no_endpoint_cm",
                                           "shift_declared": "shift_declared_cm",
                                           "shift_A_scale": "shift_A_scale_cm"})
            .to_dict(orient="records"),
            "note": "after the fact, not a test; the declared result (power 0.721, FAIL) stands"}


# ----------------------------------------------------------------------------- looks
def log_looks() -> list[dict]:
    from harness import prereg3

    from s1_rings.seed_looks import site_units
    out = [prereg3.record_look(
        ["S1a:synth.R.mixed_groupcond.v1", "S1a:synth.R.mixed_harness.v1"],
        "rings+piles:2014-05..2021-12", "S1a", "diagnose",
        test_id="S1a.e.mixed_groupcond", prereg_id="S1a.e_synthetic_mixed.v1",
        note=("S1a repair round (r2) after the verifier: decomposition of the registered (e) rows "
              "into scale and membership components, by status, b_deep votes, P(A) reliability, "
              "counterfactual primaries (after the fact, not tests) and the window coverage of "
              "the registered entry rules; no rule or threshold of v1 changed"))]
    units = site_units()
    out.append(prereg3.record_look(
        sorted(units), "rings+piles:2019-01..2025-05", "S1a", "diagnose", members=units,
        note=("S1a repair round (r2): power of check (a) decomposed after the fact (declared "
              "alternative 0.25 b_eff S vs 0.25 x group-A scale; sigma with and without the "
              "endpoint term 2 sd_s^2); real-gap gap_stats recomputed for S_fp; the declared "
              "FAIL stands")))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-look", action="store_true", help="skip logging (unit tests only)")
    args = ap.parse_args(argv)
    looks = [] if args.no_look else log_looks()
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(SYN / "synthetic_rows.parquet")
    res = json.loads((SYN / "result.json").read_text())
    p_inj = float(res["p_A"])
    e = {"source": "S1a_evidence/synthetic/synthetic_rows.parquet (registered "
                   "S1a.e_synthetic_mixed.v1 rows, scored 2026-10-13T07:52Z)",
         "status": "after the fact (diagnosis); the registered result (FAIL, 2.79 cm) stands",
         "threshold_cm": THR_CM, "membership_prior_declared": K.PRIOR_A,
         "injected_base_rate_p_A": p_inj,
         "primary": decompose_e(R, "primary", p_inj),
         "stress": decompose_e(R, "stress", p_inj),
         "window_coverage": window_coverage(),
         "looks": [{k: lk.get(k) for k in ("time", "purpose", "test_id")} for lk in looks]}
    (OUT / "e_decomposition.json").write_text(json.dumps(e, indent=1, ensure_ascii=False,
                                                         default=str))
    a = decompose_a()
    (OUT / "a_power_decomposition.json").write_text(json.dumps(a, indent=1, ensure_ascii=False,
                                                               default=str))
    pr = e["primary"]
    print(json.dumps({"worst": pr["worst_window"], "over": pr["n_windows_over_threshold"],
                      "cf": pr["counterfactuals_after_the_fact"],
                      "recon": pr["reconstruction_max_abs_err_cm"],
                      "bvotes": pr["b_deep_votes_truly_A"],
                      "coverage": {k: e["window_coverage"][k] for k in
                                   ("n_listed", "n_scored", "scored_starts",
                                    "sites_never_entering")},
                      "a": a["variants_after_the_fact"]}, indent=1, ensure_ascii=False,
                     default=str))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
