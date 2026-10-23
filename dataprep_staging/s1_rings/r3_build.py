"""Build results/dataprep/rings_bridged/v3/ (WP S1a repair round r3, S1ar).

Rules: S1a.rings_bridged_rules.v3 (declaration written before this build and before the fresh
re-test). The evidence, membership and pooling rules are ``r3_lib`` (M1-M5); series, flags, the
2022 path (with v2's 20-day feed screen) and the long tables are v2's (``build.py``,
``build_r2.py``, reused unchanged). Changes against v2 beyond M1-M5:

* checks: (e) from the fresh re-test S1a.e_synthetic_fresh.v3; (a) as registered AND the
  corrected (a) (trend-based, declared; a SECOND LOOK on the 14 sites); (b) unchanged
  tolerances, now independent of the membership because the pile is no longer a line (M2);
  residual offset unchanged;
* ``rings_corrected_*`` files carry an sd column for corrected depths (r2 verifier minor 2);
* ``fit_window_end`` is the latest of the inputs' (deep_benchmarks/v2: 2026-01; r2 verifier
  minor 1);
* allowed uses: validator and scoring_target only if the gate passes with no second-look
  criterion, every flag-0/1 upstream gate passed and U7 is ruled for S1; otherwise
  after_the_fact_analysis only.

    tools/cpu_slot.sh python -m s1_rings.r3_build          (from dataprep_staging/)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import manifest  # noqa: E402
from common.flags import validate_long_table  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402
from scipy.stats import norm  # noqa: E402

from s1_rings import build as B  # noqa: E402
from s1_rings import build_r2 as R2  # noqa: E402
from s1_rings import r3_lib as R3  # noqa: E402
from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1io  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RULES_V3 = "S1a.rings_bridged_rules.v3"
TEST_V3 = "S1a.e_synthetic_fresh.v3"
PREREG_DIR = K.REPO / "results" / "dataprep" / "preregistration"
R3DIR = K.EVIDENCE / "repair_r3"
CALIB = R3DIR / "calibration.json"
FRESH = R3DIR / "retest" / "fresh_result.json"
SPENT = R3DIR / "retest" / "spent_second_look_result.json"


def consts() -> dict:
    return json.loads(CALIB.read_text())


def real_evaluation(piles: dict, c: dict, offset_lr: float = K.OFFSET_LR):
    pos = {s: L.load_positions(s) for s in K.SITES}
    pre = {s: K.PRE_WIN for s in K.SITES}
    post = {s: B.post_win(s) for s in K.SITES}
    pl = {s: (piles[v[0]], v[1]) for s, v in K.PILES.items()}
    pw = {s: (K.PILE_PRE_WIN, K.PILE_POST_WIN) for s in K.PILES}
    res = R3.evaluate_r3(pos, pre, post, set(K.FORCED_BOTH), c["sigma_x_m"], c["sigma_c_m"],
                         offset_lr=offset_lr, piles=pl, pile_wins=pw)
    return pos, res


def site_row(s, r) -> dict:
    row = B.site_row(s, r)
    row.update({"evidence_lines": "offset,b_deep (pile is not a line in v3)",
                "b_deep_sd_v3_ppm": B.ppm(r.assign["sd_bdeep"]),
                "prior_A_em": round(r.assign["prior_A"], 6)})
    for g in "AB":
        pool = r.members[g]["pool"]
        row.update({f"pool_{g}_sites": ";".join(pool["sites"]),
                    f"pool_{g}_fallback": bool(pool["fallback"]),
                    f"sigma_cm_{g}_ppm": B.ppm(pool.get("sigma_cm", float("nan")))})
    return row


# ----------------------------------------------------------------------------- checks
def test_a_corrected(pos: dict, res: dict) -> dict:
    """Corrected (a): trend-based bridged gap compaction against the rate midpoint; AR(1)
    inflation of the trend and rate se; declared alternative and thresholds unchanged."""
    zs, shifts, ss, rows = [], [], [], []
    for s, r in res.items():
        gs = r.gs
        te = R3.trend_endpoints(pos[s], gs, K.PRE_WIN, B.post_win(s))
        ca = R3.corrected_a_site(r, te)
        kp, kq = te["pre"]["ar1_factor"], te["post"]["ar1_factor"]
        smid = math.sqrt(((gs["C_hi"] - gs["C_lo"]) / math.sqrt(12)) ** 2
                         + gs["gap"] ** 2 * ((kp * gs["se_r_pre"]) ** 2
                                             + (kq * gs["se_r_post"]) ** 2) / 4)
        si = math.sqrt(ca["sigma_C"] ** 2 + smid ** 2)
        z = (ca["C"] - gs["C_mid"]) / si
        shift = K.SCALE_ERROR * r.primary["b_eff"] * gs["S_fp"]
        zs.append(z)
        shifts.append(shift)
        ss.append(si)
        rows.append({"site": s, "C_trend_primary_cm": B.cm(ca["C"]),
                     "C_survey_primary_cm": B.cm(r.primary["C"]), "C_mid_cm": B.cm(gs["C_mid"]),
                     "sigma_trend_cm": B.cm(ca["sigma_C"]), "sigma_mid_cm": B.cm(smid),
                     "ar1_pre": round(kp, 3), "ar1_post": round(kq, 3), "z": round(z, 3),
                     "shift_25pct_cm": B.cm(shift)})
    n = len(zs)
    Z = float(np.sum(zs) / math.sqrt(n))
    mu = float(np.sum(np.array(shifts) / np.array(ss)) / math.sqrt(n))
    power = float(norm.cdf(mu - K.POOLED_Z_MAX) + norm.cdf(-mu - K.POOLED_Z_MAX))
    return {"pooled_z": Z, "n": n, "mu_25pct": mu, "power_25pct": power,
            "pass_z": abs(Z) < K.POOLED_Z_MAX, "pass_power": power >= K.POWER_MIN,
            "pass": bool(abs(Z) < K.POOLED_Z_MAX and power >= K.POWER_MIN),
            "status": "SECOND LOOK on the 14 design-time sites (a pass is conditional)",
            "sites": rows}


def test_b(res: dict, c: dict) -> dict:
    pools = {g: R3.global_pool(res, g, c["sigma_x_m"]) for g in "AB"}
    rows = []
    for s, r in res.items():
        pe = r.pile
        if pe is None or not pe.get("ok"):
            if pe is not None:
                rows.append({"site": s, "tested": False, "reason": pe.get("reason")})
            continue
        g = r.assign["assigned"] or ("A" if r.p_A >= 0.5 else "B")
        m = pools[g]["mean"]
        if g == "A":
            ok = abs(pe["b"] - m) <= K.PILE_REL_TOL_A * m
            tol = f"+-25 % of {1e6 * m:.1f} ppm"
        else:
            ok = abs(pe["b"] - m) <= K.PILE_ABS_TOL_B
            tol = f"+-30 ppm of {1e6 * m:.1f} ppm"
        rows.append({"site": s, "tested": True, "group": g,
                     "group_source": "assigned" if r.assign["assigned"] else
                     f"more probable (P(A) {r.p_A:.4f})",
                     "b_pile_ppm": B.ppm(pe["b"]), "b_pile_sd_ppm": B.ppm(pe["sd"]),
                     "pooled_ppm": B.ppm(m), "tolerance": tol, "pass": bool(ok),
                     "pile_shapes_path": K.PILES[s][2] in ("total", "shallow")})
    tested = [x for x in rows if x.get("tested")]
    return {"pools": {g: {"mean_ppm": B.ppm(v["mean"]), "se_ppm": B.ppm(v["se"]),
                          "tau_ppm": B.ppm(v["tau"]), "n": v["n"], "sites": v["sites"],
                          "prior_A": round(v["prior_A"], 4), "fallback": v["fallback"]}
                      for g, v in pools.items()},
            "sites": rows, "n_tested": len(tested),
            "pass": bool(tested and all(x["pass"] for x in tested)),
            "note": "v3: the pile is not a membership line (M2), so (b) no longer tests the "
                    "pile against a group the pile chose; the piles still shape the 2022 path "
                    "at Tuku, Xizhou, Guangfu and Xiutan, and the 14 sites are design-time units"}


def sensitivity(res: dict, piles: dict, c: dict) -> dict:
    rows = []
    for s, r in res.items():
        J = r.gs["J_top_mid"]
        g_off = min("AB", key=lambda g: abs(J - B.OFFSET_CENTRE[g]))
        C_off = r.members[g_off]["C"]
        rows.append({"site": s, "group_offset_only": g_off, "primary": r.primary["tag"],
                     "C_primary_cm": B.cm(r.primary["C"]), "C_offset_only_cm": B.cm(C_off),
                     "delta_cm": B.cm(C_off - r.primary["C"])})
    lr = {}
    for v in (3.0, 30.0):
        _, rv = real_evaluation(piles, c, offset_lr=v)
        lr[str(v)] = [{"site": s, "p_A": round(rv[s].p_A, 4),
                       "assigned": rv[s].assign["assigned"],
                       "C_primary_cm": B.cm(rv[s].primary["C"]),
                       "delta_vs_main_cm": B.cm(rv[s].primary["C"] - res[s].primary["C"])}
                      for s in res]
    return {"offset_only": rows, "offset_lr": lr}


def v2_comparison(res: dict) -> list[dict]:
    from common import guard
    prod = guard.open_product(K.REPO / "results" / "dataprep" / "rings_bridged" / "v2",
                              "after_the_fact_analysis")
    v2 = prod.read_table("ring_corrections.csv").set_index("site")
    out = []
    for s, r in res.items():
        out.append({"site": s, "assigned_v2": v2.loc[s, "assigned"]
                    if isinstance(v2.loc[s, "assigned"], str) else None,
                    "assigned_v3": r.assign["assigned"],
                    "p_A_v2": float(v2.loc[s, "p_A"]), "p_A_v3": round(r.p_A, 6),
                    "C_primary_v2_cm": float(v2.loc[s, "C_gap_primary_cm"]),
                    "C_primary_v3_cm": B.cm(r.primary["C"]),
                    "b_A_v2_ppm": float(v2.loc[s, "b_A_ppm"]), "b_A_v3_ppm": B.ppm(r.members["A"]["b"]),
                    "b_B_v2_ppm": float(v2.loc[s, "b_B_ppm"]), "b_B_v3_ppm": B.ppm(r.members["B"]["b"])})
    return out


# ----------------------------------------------------------------------------- uses
def allowed_uses_rule(gate_passed: bool, second_look_in_gate: bool, upstream: dict,
                      u7_ruled: bool = False) -> tuple[list[str], list[str], str]:
    failed = sorted(p for p, v in upstream.items() if not v["gate_passed"])
    parts = []
    if not gate_passed:
        parts.append("product gate failed")
    if gate_passed and second_look_in_gate:
        parts.append("gate passes only conditionally (a criterion is a second look)")
    if failed:
        parts.append("flag-0/1 upstream gate failed: " + ", ".join(failed))
    if not u7_ruled:
        parts.append("U7 reading for S1's post-gap checks not ruled by the user")
    if not parts:
        allowed = ["validator", "scoring_target", "after_the_fact_analysis"]
        why = "gate passed on fresh units, upstream gates passed, U7 ruled"
    else:
        allowed = ["after_the_fact_analysis"]
        why = "; ".join(parts)
    return allowed, sorted(set(USES) - set(allowed)), why


UPSTREAM_FLAG01: dict = {}    # v3: no upstream product shapes flag 0-1 values (M2)


def corrected_file(s, r, g, pos, version_assigned) -> pd.DataFrame:
    m = r.members[g]
    q = L.corrected_positions(pos[s], m["a"], m["b"], B.FIRST_POST)
    raw = pos[s]
    lr = raw.stack(future_stack=True).rename("depth_raw_m").reset_index()
    lr.columns = ["datetime", "ring", "depth_raw_m"]
    lc = q.stack(future_stack=True).rename("depth_corrected_m").reset_index()
    lr["depth_corrected_m"] = lc["depth_corrected_m"].to_numpy()
    lr = lr.dropna(subset=["depth_raw_m"])
    post = lr["datetime"] >= pd.Timestamp(B.FIRST_POST)
    lr["flag"] = 0
    lr.loc[post, "flag"] = 1 if version_assigned == g else 3
    zpre = pos[s].loc[K.PRE_WIN[0]:K.PRE_WIN[1]].mean()
    lr["depth_pre_mean_m"] = lr["ring"].map(zpre)
    ztop = r.gs["z_top"]
    se_j = r.gs["se_J_top"]
    lr["sd_m"] = np.nan
    lr.loc[post, "sd_m"] = np.sqrt(se_j ** 2 + (m["sigma"] * (lr.loc[post, "depth_pre_mean_m"]
                                                              - ztop)) ** 2)
    lr["member"] = g
    lr["b_ppm"] = round(1e6 * m["b"], 3)
    lr["sigma_b_ppm"] = round(1e6 * m["sigma"], 3)
    lr["a_m"] = round(m["a"], 6)
    lr["p_A"] = round(r.p_A, 6)
    lr["read_both_sides"] = lr["ring"].isin(r.gs["rings"])
    lr["datetime"] = lr["datetime"].dt.strftime("%Y-%m-%d")
    return lr


def main(argv=None) -> None:
    from s1_rings.r3_looks import log_second_look_a
    for f in (PREREG_DIR / f"{RULES_V3}.json", FRESH):
        if not f.exists():
            raise SystemExit(f"missing {f}: declaration and fresh re-test come first")
    look = log_second_look_a()
    c = consts()
    piles, _ = s1io.piles(preorigin=False)
    daily, _ = s1io.pile_table(preorigin=False)
    printed, printed_notes = s1io.printed_monthly()
    pos, res = real_evaluation(piles, c)
    rows = [site_row(s, r) for s, r in res.items()]
    terms = json.loads(B.PATH_TERMS.read_text())["terms"]
    fresh = json.loads(FRESH.read_text())
    spent = json.loads(SPENT.read_text()) if SPENT.exists() else None
    fc = R2.feed_check_r2(piles, daily, printed)
    fc["guard_notes"] = printed_notes
    d = manifest.new_version_dir(K.PRODUCT)
    version = f"{K.PRODUCT}/{d.name}"
    long_all, path_methods, corr_files = [], {}, []
    for s, r in res.items():
        a = r.assign
        use = K.PILES.get(s, (None, None, None))[2]
        sensor = K.PILES.get(s, (None,))[0]
        path_method = {"total": "pile", "shallow": "hybrid"}.get(use, "climatology")
        if path_method != "climatology" and not fc["sensors"].get(sensor, {}).get("ok", False):
            path_method = "climatology"
        path_methods[s] = path_method
        ser = {}
        for g in "AB":
            fpost = 1 if (a["assigned"] == g) else 3
            ser[g], _ = B.member_series(s, pos[s], r.gs, g, r.members[g], terms, piles,
                                        path_method, fpost)
        post_flag = 1 if a["assigned"] else 3
        prim = B.combine(ser, r.p_A, a["assigned"], post_flag)
        long_all.append(B.long_rows(s, "primary", prim, r.p_A, version))
        carried = [a["assigned"]] if a["assigned"] else ["A", "B"]
        if not a["assigned"]:
            for g in "AB":
                long_all.append(B.long_rows(s, f"member_{g}", ser[g], r.p_A, version))
        for g in carried:
            lr = corrected_file(s, r, g, pos, a["assigned"])
            name = f"rings_corrected_{s}.parquet" if a["assigned"] else \
                f"rings_corrected_{s}_{g}.parquet"
            lr.to_parquet(d / name, index=False)
            corr_files.append(name)
    LT = pd.concat(long_all, ignore_index=True)
    probs = validate_long_table(LT, require_all_columns=True)
    LT[LT.band == "total"].to_csv(d / "ring_total_monthly_bridged.csv", index=False)
    LT.to_parquet(d / "ring_bands_monthly_bridged.parquet", index=False)
    C = pd.DataFrame(rows)
    C.to_csv(d / "ring_corrections.csv", index=False)
    va_reg = B.test_a(res)
    va_cor = test_a_corrected(pos, res)
    vb = test_b(res, c)
    vr = B.test_resid(pos, res)
    sens = sensitivity(res, piles, c)
    g_e = fresh["gating"]
    e_pass = bool(g_e["pass"])
    gate = bool(e_pass and va_cor["pass"] and vb["pass"] and vr["pass"])
    second_look_in_gate = True           # the corrected (a) is a second look by construction
    status = ("passed, conditional (second look: corrected (a))" if gate else "failed")
    ups = {p: {"gate_passed": bool(manifest.load_manifest(K.REPO / p)[0]
                                   .get("gate", {}).get("passed")), "shapes": why}
           for p, why in UPSTREAM_FLAG01.items()}
    allowed, forbidden, why_uses = allowed_uses_rule(gate, second_look_in_gate, ups)
    n_flag = LT[(LT.series == "primary") & (LT.band == "total")].flag.value_counts().to_dict()
    val = {"wp": "S1a", "rules": RULES_V3, "fresh_test": TEST_V3,
           "constants": {k: c[k] for k in ("sigma_x_m", "sigma_c_m", "sigma_x_mm",
                                           "sigma_c_mm")},
           "e_fresh": {"gating": g_e, "summaries": fresh["summaries"]},
           "e_second_look_spent": spent["summaries"] if spent else None,
           "a_registered": va_reg, "a_corrected_second_look": va_cor,
           "b_pile_implied_scale": vb, "residual_offset": vr,
           "misassignment_sensitivity": sens, "feed_check_2022": fc,
           "path_methods": path_methods,
           "path_terms_mm": {k: v["rmse_mm"] for k, v in terms.items()},
           "long_table_problems": probs, "primary_total_flags": {int(k): int(v)
                                                                 for k, v in n_flag.items()},
           "v2_comparison": v2_comparison(res),
           "gate": {"passed": gate, "status": status, "e_fresh": e_pass,
                    "a_corrected_second_look": va_cor["pass"],
                    "a_registered_reported": va_reg["pass"], "b": vb["pass"],
                    "residual_offset": vr["pass"],
                    "note": "(e)'s 1.5 cm threshold is post hoc (plan); thresholds unchanged"},
           "allowed_uses_rule": {"allowed": allowed, "reason": why_uses, "upstream": ups},
           "look": {k: look.get(k) for k in ("time", "purpose")}}
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, default=str))
    (d / "README.md").write_text(readme_text(version, C, val, terms))
    inputs = [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]
    inputs += [rel_to_repo(K.DEEP_BENCH_V2 / "daily.parquet"),
               rel_to_repo(K.DEEP_BENCH_V2 / "MANIFEST.json"),
               rel_to_repo(K.PRINTED_V2 / "printed_monthly.parquet"),
               rel_to_repo(K.PRINTED_V2 / "MANIFEST.json"),
               rel_to_repo(B.PATH_TERMS), rel_to_repo(CALIB),
               rel_to_repo(R3DIR / "diag" / "truth_bdeep.csv"),
               rel_to_repo(K.EVIDENCE / "assignment.json"),
               rel_to_repo(FRESH), rel_to_repo(R3DIR / "retest" / "fresh_rows.parquet"),
               f"results/dataprep/preregistration/{RULES_V3}.json",
               f"results/dataprep/preregistration/{TEST_V3}.json"]
    if SPENT.exists():
        inputs.append(rel_to_repo(SPENT))
    code = [rel_to_repo(HERE / n) for n in (
        "s1const.py", "s1lib.py", "s1io.py", "build.py", "build_r2.py", "path_test.py",
        "synthetic.py", "seed_looks.py", "r3_lib.py", "r3_build.py", "r3_synthetic.py",
        "r3_calibrate.py", "r3_diagnose.py", "r3_looks.py", "r3_prereg.py")]
    code += ["dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
             "dataprep_staging/common/flags.py", "dataprep_staging/common/uses.py",
             "dataprep_staging/common/paths.py",
             "dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py",
             "dataprep_staging/harness/coverage3.py"]
    m = manifest.build_manifest(
        d, product=K.PRODUCT, layer="L2", inputs=inputs, code=code,
        parameters={"rules": RULES_V3, "forced_both": sorted(K.FORCED_BOTH),
                    "evidence_lines": ["offset", "b_deep"],
                    "reference_groups_ppm": {"A": [182.2, 11.4], "B": [1.9, 14.3]},
                    "offset_clusters_m": K.OFFSET_CLUSTERS, "offset_lr": K.OFFSET_LR,
                    "lr_vote": K.LR_VOTE, "lr_cap": K.LR_CAP, "z_plausible": K.Z_PLAUSIBLE,
                    "sigma_x_m": c["sigma_x_m"], "sigma_c_m": c["sigma_c_m"],
                    "pool_posterior_min": R3.POOL_P_MIN,
                    "membership_prior": "EM over the other sites, Laplace rule",
                    "pre_win": K.PRE_WIN, "post_win": K.POST_WIN,
                    "post_win_beichen": K.POST_WIN_BEICHEN,
                    "feed_check_min_vendor_days": R2.MIN_FEED_DAYS,
                    "path_methods": path_methods,
                    "path_terms_mm": {k: v["rmse_mm"] for k, v in terms.items()},
                    "input_caveats": [
                        "deep_benchmarks/v2 (gate failed on test A, 3/61 units) is read as "
                        "after_the_fact_analysis for check (b) and the 2022 path shape; in v3 "
                        "it no longer shapes the group assignment (no flag 0-1 value).",
                        "GNSS (gnss_clean, gate failed) is not used by the product."]},
        fit_window_end="2026-01", validation_report="validation.json",
        gate={"passed": gate, "status": status,
              "criteria": "plan 3 WP S1 (S1a), rules v3: (e) worst-window RMSE <= 1.5 cm "
                          "(post hoc) on FRESH pre-2012 windows (S1a.e_synthetic_fresh.v3, "
                          "primary injection, method r3); (a) corrected (trend-based) |pooled "
                          "standardized difference| < 2 with power >= 0.8 for a 25 % scale "
                          "error (a declared second look; the registered (a) is reported); "
                          "(b) pile-implied scale within 25 % (A) / 30 ppm (B) of the pooled "
                          "scale; residual offset within +-1 cm at >= 12 of 14 sites"},
        allowed_uses=allowed, forbidden_uses=forbidden, personal_data="none",
        extra={"product_card": "README.md",
               "supersedes": {"version": "rings_bridged/v2", "note": relation_note(fresh)},
               "allowed_uses_rule": {"reason": why_uses, "upstream_flag01": ups,
                                     "rule": "validator/scoring_target only if the gate passes "
                                             "with no second-look criterion, every flag-0/1 "
                                             "upstream gate passed and U7 is ruled for S1 "
                                             f"(declaration {RULES_V3})"},
               "series_conventions": {
                   "ring_holdout": "plan 5.5: an ALTERNATIVE series to the ring_holdout "
                                   "re-reference at the first 2023 survey; report side by side, "
                                   "never mix in one score",
                   "flags": "0 observed; 1 corrected, group independently assigned (offset and "
                            "b_deep agree); 3 corrected under an unassigned / forced group or "
                            "pile-shaped 2022 path; 4 climatology 2022 path; 6 Beichen's "
                            "30-month path; 8 missing"}})
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise SystemExit(f"manifest errors: {errs}")
    manifest.write_manifest(d, m)
    print(d, "gate", gate, status, "allowed", allowed)
    print(json.dumps({"gate": val["gate"], "a_registered": {k: va_reg[k] for k in
                                                            ("pooled_z", "power_25pct", "pass")},
                      "a_corrected": {k: va_cor[k] for k in ("pooled_z", "power_25pct", "pass")},
                      "b": vb["pass"], "resid": [vr["n_within"], vr["n"]]},
                     ensure_ascii=False, default=str))


def relation_note(fresh: dict) -> str:
    """Post-registration metadata text (r3 round, disclosed): v3 is the newer rule set, not a
    validated improvement over v2."""
    s = fresh["summaries"]
    r3, rg = s["pre2012.r3.primary"], s["pre2012.registered.primary"]
    return (f"newer rule set (r3), NOT a validated improvement: neither v2's nor v3's method "
            f"passed (e) on the fresh pre-2012 draws (worst window {100 * r3['worst_window_rmse_m']:.2f} cm "
            f"with v3's method, {100 * rg['worst_window_rmse_m']:.2f} cm with v2's); v2's lower value "
            f"comes mainly from its pools falling back to the reference scale, which equals the "
            f"injection mean (A-pool fallback {rg['fallback_share_A']:.2f} against "
            f"{r3['fallback_share_A']:.2f}); see results/dataprep/reports/S1a.repair.md")


def readme_text(version, C, val, terms) -> str:
    asg = C[["site", "assigned", "forced_both", "p_A", "prior_A_em", "C_gap_primary_cm",
             "sigma_C_primary_cm"]].to_string(index=False)
    g = val["gate"]
    e = val["e_fresh"]["gating"]
    ar, ac = val["a_registered"], val["a_corrected_second_look"]
    vb, vr = val["b_pile_implied_scale"], val["residual_offset"]
    al = val["allowed_uses_rule"]
    cs = val["constants"]
    return f"""# {version}: compaction rings with the 2022 instrument change corrected and the gap bridged

WP S1a (plan 3, WP S1), repair round r3 (S1ar). Rules:
results/dataprep/preregistration/{RULES_V3}.json. Code: dataprep_staging/s1_rings/ (r3_build.py,
r3_lib.py; series and paths from build.py / build_r2.py). Newer version than
rings_bridged/v2, but {relation_note({'summaries': val['e_fresh']['summaries']})}.

## What it is

After the 2022 survey hiatus every ring depth reads z_post = a_site + (1 + b_group) z_true;
post-gap depths are corrected with z' = (z - a)/(1 + b). v3 changes how the group and the scale
are inferred (results/dataprep/reports/S1a.repair.md, Part 1 diagnosis and Part 2 changes):

- evidence lines: the offset cluster of the top-ring jump and the site's own deep-interval scale
  b_deep, whose sd now grows as 1/deep length (sigma_x {cs['sigma_x_mm']:.2f} mm in quadrature
  with the jackknife); the pile-implied scale is NOT a line (it is check (b) and shapes the 2022
  path);
- membership probability P(A) with an empirical-Bayes prior (EM over the other sites);
- pooled scale from every other site with P(group) >= 0.95, weighted by the length-aware sd,
  plus a common-mode term (sigma_c {cs['sigma_c_mm']:.2f} mm) in its sigma.

A site is assigned when the offset and b_deep lines agree and it is not on the plan's forced
list; the other sites are carried as both members with the P(A)-weighted mixture as primary.

This is a DATA product for post-origin uses only: an alternative ring series to the
ring_holdout re-reference (plan 5.5), reported side by side and never mixed in one score.
fit_window_end 2026-01 (inputs included); no pre-origin use is allowed.

## Allowed uses: {", ".join(al['allowed'])}

Reason: {al['reason']}.

## Gate: {g['status']}

- (e) fresh re-test (pre-2012 windows, never used before): worst-window RMSE
  {100 * e['worst_window_rmse_m']:.2f} cm (threshold 1.5 cm, post hoc): {'pass' if e['pass'] else 'FAIL'}.
- (a) registered: pooled Z {ar['pooled_z']:.2f}, power {ar['power_25pct']:.3f}: {'pass' if ar['pass'] else 'FAIL'} (reported).
- (a) corrected (trend-based; a SECOND LOOK on the 14 sites): pooled Z {ac['pooled_z']:.2f}, power
  {ac['power_25pct']:.3f}: {'pass' if ac['pass'] else 'FAIL'}.
- (b) {vb['n_tested']} pile sites: {'pass' if vb['pass'] else 'FAIL'}.
- residual offset {vr['n_within']}/{vr['n']} within 1 cm: {'pass' if vr['pass'] else 'FAIL'}.

## Assignment

{asg}

## Files

- ring_corrections.csv: per site evidence, v3 sd of b_deep, EM prior, P(A), pools (sites,
  fallback), both members' b, sigma_b, a and bridged gap compaction, primary.
- rings_corrected_<site>[_A|_B].parquet: every survey, raw and corrected depth, flag, and for
  corrected depths sd_m = sqrt(se_J_top^2 + (sigma_b (z - z_top))^2).
- ring_total_monthly_bridged.csv, ring_bands_monthly_bridged.parquet: as in v2.
- validation.json: every check, the fresh re-test summaries, the second look on the spent
  windows, sensitivities, the 2022 feed check and the v2 comparison.

## Flags (plan 2.4) and uncertainty

Flags as in v2: 0 observed; 1 corrected with an independently assigned group; 3 corrected under
an unassigned or forced group (mixture), a non-assigned member, or a pile-shaped 2022 path; 4
climatology 2022 path; 6 Beichen's 30-month path; 8 missing. Flags 2-6 never enter scoring
targets (U1). Post-gap sd = sqrt(sd_s^2 + (sigma_b L)^2); 2022 path sd = sqrt((sigma_b L tau)^2 +
path^2) (family-Y path terms: pile {terms['total.pile']['rmse_mm']:.2f} mm, climatology
{terms['total.climatology']['rmse_mm']:.2f} mm). Observed-month sd is the residual sd of the
2019-21 fit, an upper bound: a weighting by sd would trust the pile-shaped fills (flag 3) more
than the observed months at Tuku (r2 verifier minor 3; not changed). Every sd is UNCALIBRATED.
"""


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
