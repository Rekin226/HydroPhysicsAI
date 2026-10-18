"""Build results/dataprep/rings_bridged/v2/ (WP S1a repair round).

Rules: S1a.rings_bridged_rules.v1 as amended by the declaration S1a.rings_bridged_rules.v2
(written before this build). Every evidence line, assignment, pooling, member, mixture, flag,
sd and acceptance check is v1's (``build.py``, ``s1lib.py``; their code is reused unchanged).
The amendments:

1. **2022 feed check** (path method): a vendor monthly-mean increment is compared with the
   printed 007001 value only when both of its months have >= 20 vendor days with flag 0-1 (the
   deep_benchmarks convention for a comparable monthly mean). v1 compared every month and
   tripped on TKJS(1)'s 2022-05-04..23 outage (a 9-day May mean), which sent Tuku's 2022 path
   to climatology. POST-RESULT amendment, declared as such; no acceptance check uses the path.
2. **Allowed uses**: validator and scoring_target are declared only when the product gate
   passes AND every upstream product that shapes flag 0-1 values passed its own gate.
   deep_benchmarks/v2 (gate failed) shapes the group assignment through the pile line, so
   v2 declares after_the_fact_analysis only (closes the route by which a later version could
   pass its own gate and carry a failed upstream into scoring targets).
3. **Provenance**: the manifest hashes every module the outputs depend on (adds
   harness/coverage3.py, common/flags.py, common/uses.py, common/paths.py and the repair
   modules).
4. **Product card and validation**: corrected (13 synthetic clusters; the (e) diagnosis of
   ``r2_diagnose``), plus after-the-fact items labelled as such: residual offsets under both
   members, how far check (b) is implied by the pile vote, and the assignment with the plan's
   unwidened offset ranges. None of them enters the gate.

    tools/cpu_slot.sh python -m s1_rings.build_r2          (from dataprep_staging/)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import manifest  # noqa: E402
from common.flags import validate_long_table  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

from s1_rings import build as B  # noqa: E402
from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1io  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RULES_V1 = "S1a.rings_bridged_rules.v1"
RULES_V2 = "S1a.rings_bridged_rules.v2"
PREREG_DIR = K.REPO / "results" / "dataprep" / "preregistration"
REPAIR = K.EVIDENCE / "repair"
E_DECOMP = REPAIR / "e_decomposition.json"
A_DECOMP = REPAIR / "a_power_decomposition.json"
MIN_FEED_DAYS = 20                 # deep_benchmarks: >= 20 full days (flag 0-1) per month
#: The plan's offset ranges before the declared 0.5 cm widening (after-the-fact sensitivity).
PLAN_CLUSTERS = {"A": (-0.163, -0.155), "B": (0.50, 0.54)}
#: Upstream products whose values shape flag 0-1 values of this product (allowed-uses rule).
UPSTREAM_FLAG01 = {"results/dataprep/deep_benchmarks/v2":
                   "pile-implied scale = a group-evidence line (assignment, pooled groups)"}
SCORING_USES = ("validator", "scoring_target")


# ----------------------------------------------------------------------------- amendment 1
def vendor_days(daily: pd.DataFrame, sensor: str) -> pd.Series:
    """Vendor days with flag 0-1 per month (PeriodIndex) of one sensor."""
    x = daily[(daily["sensor_id"] == sensor) & (daily["feed"] == "vendor")
              & (daily["flag"].isin([0, 1]))]
    d = pd.DatetimeIndex(pd.to_datetime(x["date"])).normalize()
    d = pd.Series(1, index=d[~d.duplicated()])
    return d.groupby(d.index.to_period("M")).size()


def feed_check_r2(piles: dict, daily: pd.DataFrame, printed: pd.DataFrame) -> dict:
    """v1's feed check (vendor monthly-mean increments vs printed 007001 values, flag 0;
    > 2 mm in a 2022 month -> climatology path) with the day screen of amendment 1: a month's
    increment is compared only when both months have >= MIN_FEED_DAYS vendor days (flag 0-1)."""
    pm = printed[printed["flag"] == 0]
    out = {"min_days": MIN_FEED_DAYS, "sensors": {}}
    for sensor, _depth, use in K.PILES.values():
        if use == "evidence" or sensor in out["sensors"]:
            continue
        q = piles[sensor]
        mm = q.groupby(q.index.to_period("M")).mean()
        inc = mm.diff()
        nd = vendor_days(daily, sensor)
        rows = []
        for m in pd.period_range("2021-12", "2023-02", freq="M"):
            pv = pm[(pm.sensor_id == sensor) & (pm.month == str(m))]["value"]
            if pv.empty or m not in inc.index or not np.isfinite(inc.get(m, np.nan)):
                continue
            n0, n1 = int(nd.get(m - 1, 0)), int(nd.get(m, 0))
            comparable = n0 >= MIN_FEED_DAYS and n1 >= MIN_FEED_DAYS
            rows.append({"month": str(m), "printed_mm": float(pv.iloc[0]),
                         "feed_mm": round(float(inc[m]), 3),
                         "diff_mm": round(float(inc[m] - pv.iloc[0]), 3),
                         "vendor_days_prev": n0, "vendor_days": n1,
                         "comparable": comparable,
                         "in_rule": bool(m.year == 2022 and comparable)})
        d = np.array([r["diff_mm"] for r in rows if r["in_rule"]])
        out["sensors"][sensor] = {
            "months": rows, "n_rule_months": len(d),
            "n_screened_out_2022": sum(1 for r in rows if r["month"].startswith("2022")
                                       and not r["comparable"]),
            "max_abs_diff_mm": float(np.abs(d).max()) if len(d) else None,
            "ok": bool(len(d) and np.abs(d).max() <= 2.0),
            "rule": "2022 months whose increment has >= 20 vendor days (flag 0-1) in both "
                    "months (declaration S1a.rings_bridged_rules.v2, amendment 1)"}
    return out


# ----------------------------------------------------------------------------- amendment 2
def upstream_gates() -> dict:
    out = {}
    for p, why in UPSTREAM_FLAG01.items():
        m, _ = manifest.load_manifest(K.REPO / p)
        out[p] = {"gate_passed": bool(m.get("gate", {}).get("passed")), "shapes": why}
    return out


def allowed_uses_rule(gate_passed: bool, upstream: dict) -> tuple[list[str], list[str], str]:
    """(allowed, forbidden, reason): scoring uses need this gate AND every flag-0/1 upstream
    gate; after_the_fact_analysis is always allowed; every other use is forbidden."""
    failed = sorted(p for p, v in upstream.items() if not v["gate_passed"])
    if gate_passed and not failed:
        allowed = list(SCORING_USES) + ["after_the_fact_analysis"]
        why = "product gate passed and every flag-0/1 upstream gate passed"
    else:
        allowed = ["after_the_fact_analysis"]
        parts = []
        if not gate_passed:
            parts.append("product gate failed")
        if failed:
            parts.append("flag-0/1 upstream gate failed: " + ", ".join(failed))
        why = "; ".join(parts)
    forbidden = sorted(set(USES) - set(allowed))
    return allowed, forbidden, why


# ----------------------------------------------------------------------------- after the fact
@contextmanager
def offset_clusters(clusters: dict):
    old = dict(K.OFFSET_CLUSTERS)
    K.OFFSET_CLUSTERS.clear()
    K.OFFSET_CLUSTERS.update(clusters)
    try:
        yield
    finally:
        K.OFFSET_CLUSTERS.clear()
        K.OFFSET_CLUSTERS.update(old)


def resid_both_members(pos: dict, res: dict) -> list[dict]:
    rows = []
    for s, r in res.items():
        row = {"site": s}
        for g in "AB":
            m = r.members[g]
            q = L.corrected_positions(pos[s], m["a"], m["b"], B.FIRST_POST)
            ro = L.residual_offset(q, r.gs["rings"], r.gs["t_lp"], r.gs["t_fp"])
            row[f"offset_{g}_cm"] = B.cm(ro["offset_m"])
        row["difference_A_minus_B_cm"] = round(row["offset_A_cm"] - row["offset_B_cm"], 3)
        rows.append(row)
    return rows


def b_vote_implication(res: dict, vb: dict) -> list[dict]:
    """For every tested pile site: the pile line's vote band (reference group mean +- 3 sd
    with the 10 ppm floor) against check (b)'s band; (b) is implied by the vote when the vote
    band lies inside (b)'s band (to 0.5 ppm)."""
    rows = []
    pools = {g: v["mean_ppm"] for g, v in vb["pools"].items()}
    for x in vb["sites"]:
        if not x.get("tested"):
            continue
        s, g = x["site"], x["group"]
        pe = res[s].pile
        sd = max(pe["sd"], K.SD_FLOOR)
        mu, tau = K.REF[g]
        half = K.Z_PLAUSIBLE * math.sqrt(sd * sd + tau * tau)
        vote = res[s].lines.get("pile", {}).get("vote")
        lo, hi = (1e6 * (mu - half), 1e6 * (mu + half))
        m = pools[g]
        blo, bhi = ((1 - K.PILE_REL_TOL_A) * m, (1 + K.PILE_REL_TOL_A) * m) if g == "A" else \
            (m - 1e6 * K.PILE_ABS_TOL_B, m + 1e6 * K.PILE_ABS_TOL_B)
        implied = vote == g and lo >= blo - 0.5 and hi <= bhi + 0.5
        rows.append({"site": s, "group": g, "pile_vote": vote, "b_pile_ppm": x["b_pile_ppm"],
                     "vote_band_ppm": [round(lo, 1), round(hi, 1)],
                     "b_band_ppm": [round(blo, 1), round(bhi, 1)],
                     "b_implied_by_vote": bool(implied),
                     "pile_vote_needed_for_assignment": bool(
                         vote == g and res[s].assign["votes_" + g] == 2
                         and res[s].assign["evidence_group"] == g)})
    return rows


def plan_cluster_sensitivity(piles: dict, res: dict) -> list[dict]:
    with offset_clusters(PLAN_CLUSTERS):
        _, rv = B.real_evaluation(piles)
    out = []
    for s in res:
        a, b = res[s], rv[s]
        out.append({"site": s, "offset_vote_declared": a.lines["offset"]["vote"],
                    "offset_vote_plan_ranges": b.lines["offset"]["vote"],
                    "assigned_declared": a.assign["assigned"],
                    "assigned_plan_ranges": b.assign["assigned"],
                    "p_A_declared": round(a.p_A, 4), "p_A_plan_ranges": round(b.p_A, 4),
                    "C_primary_declared_cm": B.cm(a.primary["C"]),
                    "C_primary_plan_ranges_cm": B.cm(b.primary["C"]),
                    "delta_cm": B.cm(b.primary["C"] - a.primary["C"])})
    return out


def log_after_the_fact_look() -> dict:
    from harness import prereg3

    from s1_rings.seed_looks import site_units
    units = site_units()
    return prereg3.record_look(
        sorted(units), "rings+piles:2014-05..2025-05", "S1a", "diagnose", members=units,
        note=("S1a repair round (r2), rings_bridged/v2 build: v1's declared real-gap checks "
              "recomputed under unchanged rules, plus after-the-fact items (residual offsets "
              "under both members, check (b) implied by the pile vote, assignment with the "
              "plan's unwidened offset ranges); none enters the gate"))


# ----------------------------------------------------------------------------- card
def readme_text(version, C, va, vb, vr, syn, gate, terms, fc, why_uses, edec,
                allowed) -> str:
    asg = C[["site", "assigned", "forced_both", "p_A", "C_gap_primary_cm",
             "sigma_C_primary_cm"]].to_string(index=False)
    p = edec["primary"]
    cf = p["counterfactuals_after_the_fact"]
    orc = cf["true group known (scale component only)"]
    ww = p["worst_window"]
    tk = fc["sensors"]["TKJS(1)"]
    cov = syn["primary"]["coverage_90"]
    return f"""# {version}: compaction rings with the 2022 instrument change corrected and the gap bridged

WP S1a (plan 3, WP S1), repair round. Rules: results/dataprep/preregistration/{RULES_V1}.json
as amended by {RULES_V2}.json. Code: dataprep_staging/s1_rings/ (build_r2.py reusing build.py
and s1lib.py). Supersedes rings_bridged/v1 (same rules, same corrections; see "Changes").

## What it is

After the 2022 survey hiatus every ring depth reads z_post = a_site + (1 + b_group) z_true.
Post-gap depths are corrected with z' = (z - a)/(1 + b). The group (A: about +180 ppm, B: about
0 ppm) is assigned from three lines: the offset cluster of the top-ring jump, the site's own
deep-interval scale b_deep, and the pile-implied scale. A site is assigned when >= 2 lines
agree, none disagrees and it is not on the plan's forced list; the other sites are carried as
both members with a membership probability P(A). Note: the pile line compares the raw ring
change across the gap (for the 300 m piles, the whole-interval change itself) with the change
the pile predicts; the plan sanctions this line, but it is not independent of the jump being
corrected. The months between the last 2021 and the first 2023 survey are bridged:
pile-shaped at Tuku (restored in v2), Xizhou and Guangfu; Xiutan's shallow band pile-shaped;
elsewhere linear + seasonal climatology.

This is a DATA product for post-origin uses only: an alternative ring series to the
ring_holdout re-reference (plan 5.5), reported side by side and never mixed in one score.
fit_window_end 2025-05; no pre-origin use is allowed.

## Allowed uses: {", ".join(allowed)}

Reason: {why_uses}. Validator and scoring_target need this product's gate AND every upstream
product that shapes flag 0-1 values to pass (deep_benchmarks/v2 shapes the assignment through
the pile line).

## Files

- ring_corrections.csv: per site evidence (offset, b_deep, pile-implied scale, votes), group,
  membership probability P(A), both members' b, sigma_b, a and bridged gap compaction, primary.
- rings_corrected_<site>.parquet (assigned sites) / rings_corrected_<site>_<A|B>.parquet (sites
  carried as both members): every survey, raw and corrected depth, flag.
- ring_total_monthly_bridged.csv: monthly total-interval compaction (mm, + = shortening,
  re-zeroed to the first month), sd, flag, method; series 'primary' and, for sites carried as
  both members, 'member_A'/'member_B'.
- ring_bands_monthly_bridged.parquet: the same for the total and the three depth bands of
  hydrophysics.twin.mechanism.band_compaction (0-100, 100-200, 200-bottom m below the top ring).
  In 2022 the bands use climatology even where the total is pile-shaped, so the bands do not sum
  to the total in those months.
- validation.json: checks (a), (b), (e), residual offset, sensitivities, 2022 feed check, and
  the repair round's after-the-fact items (key repair_r2).

## Assignment

{asg}

## Flags (plan 2.4)

0 observed; 1 corrected, group independently assigned; 3 corrected under an unassigned or
forced group (mixture), a non-assigned member, or a pile-shaped 2022 path; 4 climatology 2022
path; 6 Beichen's 30-month path (beyond the 13-month tested envelope); 8 missing. Flags 2-6
never enter scoring targets (U1): post-gap levels of sites carried as both members are flag 3.

## Uncertainty

Post-gap sd = sqrt(sd_s^2 + (sigma_b L)^2); 2022 path sd = sqrt((sigma_b L tau)^2 + path^2),
path = family-Y RMSE (pile {terms['total.pile']['rmse_mm']:.2f} mm, hybrid
{terms['total.hybrid']['rmse_mm']:.2f} mm, climatology {terms['total.climatology']['rmse_mm']:.2f}
mm for the total); the primary adds the membership mixture variance. Observed-month sd is the
residual sd of the 2019-21 fit (an upper bound). Every sd is UNCALIBRATED: the synthetic
coverage of the nominal 90 % interval is {cov['coverage']:.3f} with {cov['k_clusters']} clusters
(sites; below the A1 envelope of 20).

## Gate: {gate} (unchanged from v1; no test was re-run)

(a) pooled Z {va['pooled_z']:.2f}, power {va['power_25pct']:.3f} (needs 0.8): FAIL; (b)
{vb['n_tested']} pile sites, pass {vb['pass']} (not independent: every pile is a group line);
residual offset {vr['n_within']}/{vr['n']} within 1 cm, pass {vr['pass']} (cannot distinguish the
groups: the offset a absorbs the jump); (e) worst-window RMSE
{100 * syn['primary']['worst_window_rmse_m']:.2f} cm (threshold 1.5 cm, post hoc): FAIL.

Why (e) fails (after the fact, results/dataprep/reports/S1a_evidence/repair/e_decomposition.json):
in the worst window ({ww['start']}..{ww['end']}) the error of the member of the TRUE group (the
pooled-scale component) has RMSE {ww['rmse_scale_cm']:.2f} cm and bias {ww['bias_scale_cm']:.2f} cm;
the membership component (mixture under weak evidence) adds RMS {ww['rms_membership_cm']:.2f} cm and
bias {ww['bias_membership_cm']:.2f} cm. With the true groups known the worst window would be
{orc['worst_window_rmse_cm']:.2f} cm and {orc['n_windows_over_1p5cm']} of {orc['n_windows']} windows would
exceed 1.5 cm, against {p['n_windows_over_threshold']['actual']} in the registered run.

## Changes from v1

- Tuku's 2022 path is pile-shaped (flag 3) instead of climatology (flag 4): the feed check now
  screens months with < 20 vendor days (TKJS(1) 2022: {tk['n_screened_out_2022']} increments
  screened out, max |diff| of the rest {tk['max_abs_diff_mm']} mm).
- allowed_uses: after_the_fact_analysis only (see above); v1 declared validator and
  scoring_target, which its failed gate never opened.
- Manifest code provenance completed; this card corrected (13 synthetic clusters; the (e)
  diagnosis).
- Every other value is identical to v1.
"""


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-look", action="store_true")
    args = ap.parse_args(argv)
    decl = PREREG_DIR / f"{RULES_V2}.json"
    if not decl.exists():
        raise SystemExit(f"declaration {decl} missing: run prereg_r2 first")
    look = None if args.no_look else log_after_the_fact_look()
    piles, _pile_notes = s1io.piles(preorigin=False)
    daily, _ = s1io.pile_table(preorigin=False)
    printed, printed_notes = s1io.printed_monthly()
    pos, res = B.real_evaluation(piles)
    rows = [B.site_row(s, r) for s, r in res.items()]
    pa = B.p_A_rule(res)
    terms = json.loads(B.PATH_TERMS.read_text())["terms"]
    syn = json.loads(B.SYN_RESULT.read_text())
    edec = json.loads(E_DECOMP.read_text())
    adec = json.loads(A_DECOMP.read_text())
    fc = feed_check_r2(piles, daily, printed)
    fc["guard_notes"] = printed_notes
    d = manifest.new_version_dir(K.PRODUCT)
    version = f"{K.PRODUCT}/{d.name}"
    long_all, path_methods = [], {}
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
            lr.loc[post, "flag"] = 1 if a["assigned"] == g else 3
            lr["member"] = g
            lr["b_ppm"] = round(1e6 * m["b"], 3)
            lr["a_m"] = round(m["a"], 6)
            lr["p_A"] = round(r.p_A, 6)
            lr["read_both_sides"] = lr["ring"].isin(r.gs["rings"])
            lr["depth_pre_mean_m"] = lr["ring"].map(
                pos[s].loc[K.PRE_WIN[0]:K.PRE_WIN[1]].mean())
            lr["datetime"] = lr["datetime"].dt.strftime("%Y-%m-%d")
            name = f"rings_corrected_{s}.parquet" if a["assigned"] else \
                f"rings_corrected_{s}_{g}.parquet"
            lr.to_parquet(d / name, index=False)
    LT = pd.concat(long_all, ignore_index=True)
    probs = validate_long_table(LT, require_all_columns=True)
    LT[LT.band == "total"].to_csv(d / "ring_total_monthly_bridged.csv", index=False)
    LT.to_parquet(d / "ring_bands_monthly_bridged.parquet", index=False)
    C = pd.DataFrame(rows)
    C.to_csv(d / "ring_corrections.csv", index=False)
    va, vb, vr = B.test_a(res), B.test_b(res), B.test_resid(pos, res)
    sens = B.sensitivity(res, piles)
    e_pass = bool(syn["primary"]["pass"])
    gate = bool(va["pass"] and vb["pass"] and vr["pass"] and e_pass)
    ups = upstream_gates()
    allowed, forbidden, why_uses = allowed_uses_rule(gate, ups)
    pr = edec["primary"]
    repair = {
        "status": "after the fact; none of these items enters the gate",
        "e_diagnosis": {
            "source": rel_to_repo(E_DECOMP),
            "worst_window": pr["worst_window"],
            "n_windows_over_threshold": pr["n_windows_over_threshold"],
            "counterfactuals_after_the_fact": pr["counterfactuals_after_the_fact"],
            "worst_window_by_status": pr["worst_window_by_status"],
            "b_deep_votes_truly_A": pr["b_deep_votes_truly_A"],
            "membership_probability_reliability": pr["membership_probability_reliability"],
            "window_coverage": {k: edec["window_coverage"][k] for k in
                                ("n_listed", "n_scored", "scored_starts", "sites_never_entering")},
            "stress_n_windows_over_threshold": edec["stress"]["n_windows_over_threshold"]},
        "a_power_decomposition": {"source": rel_to_repo(A_DECOMP),
                                  "variants": adec["variants_after_the_fact"]},
        "residual_offset_both_members": resid_both_members(pos, res),
        "b_implied_by_pile_vote": b_vote_implication(res, vb),
        "plan_offset_ranges_sensitivity": plan_cluster_sensitivity(piles, res),
        "path_methods": path_methods,
        "look": {k: look.get(k) for k in ("time", "purpose")} if look else None}
    val = {"wp": "S1a", "rules": f"{RULES_V1} as amended by {RULES_V2}",
           "synthetic_test": "S1a.e_synthetic_mixed.v1 (registered result, not re-run)",
           "p_A_rule": pa,
           "a_pooled_standardized_difference": va, "b_pile_implied_scale": vb,
           "residual_offset": vr, "e_synthetic": {k: syn[k] for k in ("primary", "stress")
                                                  if k in syn},
           "misassignment_sensitivity": sens, "feed_check_2022": fc,
           "path_terms_mm": {k: v["rmse_mm"] for k, v in terms.items()},
           "long_table_problems": probs,
           "gate": {"passed": gate, "e": e_pass, "a": va["pass"], "b": vb["pass"],
                    "residual_offset": vr["pass"],
                    "note": "e's 1.5 cm threshold is post hoc (plan); gate rules unchanged "
                            "from v1"},
           "allowed_uses_rule": {"allowed": allowed, "reason": why_uses, "upstream": ups},
           "repair_r2": repair}
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, default=str))
    (d / "README.md").write_text(readme_text(version, C, va, vb, vr, syn, gate, terms, fc,
                                             why_uses, edec, allowed))
    inputs = [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]
    inputs += [rel_to_repo(K.DEEP_BENCH_V2 / "daily.parquet"),
               rel_to_repo(K.DEEP_BENCH_V2 / "MANIFEST.json"),
               rel_to_repo(K.PRINTED_V2 / "printed_monthly.parquet"),
               rel_to_repo(K.PRINTED_V2 / "MANIFEST.json"),
               rel_to_repo(B.PATH_TERMS), rel_to_repo(B.SYN_RESULT),
               rel_to_repo(K.EVIDENCE / "synthetic" / "synthetic_rows.parquet"),
               rel_to_repo(E_DECOMP), rel_to_repo(A_DECOMP),
               f"results/dataprep/preregistration/{RULES_V1}.json",
               f"results/dataprep/preregistration/{RULES_V2}.json",
               "results/dataprep/preregistration/S1a.e_synthetic_mixed.v1.json"]
    code = [rel_to_repo(HERE / n) for n in ("s1const.py", "s1lib.py", "s1io.py", "build.py",
                                            "build_r2.py", "path_test.py", "synthetic.py",
                                            "r2_diagnose.py", "seed_looks.py")]
    code += ["dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
             "dataprep_staging/common/flags.py", "dataprep_staging/common/uses.py",
             "dataprep_staging/common/paths.py",
             "dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py",
             "dataprep_staging/harness/coverage3.py"]
    m = manifest.build_manifest(
        d, product=K.PRODUCT, layer="L2", inputs=inputs, code=code,
        parameters={"rules": f"{RULES_V1} as amended by {RULES_V2}",
                    "forced_both": sorted(K.FORCED_BOTH),
                    "reference_groups_ppm": {"A": [182.2, 11.4], "B": [1.9, 14.3]},
                    "offset_clusters_m": K.OFFSET_CLUSTERS, "offset_lr": K.OFFSET_LR,
                    "lr_vote": K.LR_VOTE, "lr_cap": K.LR_CAP, "z_plausible": K.Z_PLAUSIBLE,
                    "pre_win": K.PRE_WIN, "post_win": K.POST_WIN,
                    "post_win_beichen": K.POST_WIN_BEICHEN,
                    "feed_check_min_vendor_days": MIN_FEED_DAYS,
                    "path_methods": path_methods,
                    "path_terms_mm": {k: v["rmse_mm"] for k, v in terms.items()},
                    "input_caveats": [
                        "deep_benchmarks/v2 (gate failed on test A, 3/61 units) is read as "
                        "after_the_fact_analysis for the pile-implied scales (a group-evidence "
                        "line) and the 2022 path shape; because it shapes the assignment, the "
                        "allowed-uses rule keeps this product to after_the_fact_analysis.",
                        "GNSS (gnss_clean, gate failed) is not used by the product."],
                    "membership_prior_A": K.PRIOR_A},
        fit_window_end="2025-05", validation_report="validation.json",
        gate={"passed": gate, "criteria": "plan 3 WP S1 (S1a), unchanged from v1: (e) "
              "worst-window RMSE <= 1.5 cm (post hoc) in the mixed A/B synthetic rerun "
              "(registered S1a.e_synthetic_mixed.v1, not re-run); (a) |pooled standardized "
              "difference| < 2 with power >= 0.8 for a 25 % scale error; (b) pile-implied "
              "scale within 25 % (A) / 30 ppm (B) of the pooled scale; residual offset within "
              "+-1 cm at >= 12 of 14 sites"},
        allowed_uses=allowed, forbidden_uses=forbidden, personal_data="none",
        extra={"product_card": "README.md", "supersedes": "rings_bridged/v1",
               "allowed_uses_rule": {"reason": why_uses, "upstream_flag01": ups,
                                     "rule": "validator/scoring_target only if the product "
                                             "gate AND every flag-0/1 upstream gate passed "
                                             f"(declaration {RULES_V2})"},
               "series_conventions": {
                   "ring_holdout": "plan 5.5: an ALTERNATIVE series to the ring_holdout "
                                   "re-reference at the first 2023 survey; report side by side, "
                                   "never mix in one score",
                   "flags": "0 observed; 1 corrected, group independently assigned; 3 "
                            "corrected under an unassigned / forced group or pile-shaped 2022 "
                            "path; 4 climatology 2022 path; 6 Beichen's 30-month path; 8 "
                            "missing"}})
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise SystemExit(f"manifest errors: {errs}")
    manifest.write_manifest(d, m)
    print(d, "gate", gate, "allowed", allowed)
    print(json.dumps({"gate": val["gate"], "path_methods": path_methods,
                      "feed": {k: (v["ok"], v["max_abs_diff_mm"], v["n_screened_out_2022"])
                               for k, v in fc["sensors"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
