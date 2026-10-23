"""Build results/dataprep/rings_bridged/v<N>/ (WP S1a; rules S1a.rings_bridged_rules.v1).

    tools/cpu_slot.sh python -m s1_rings.build --assign-only   # evidence + assignment only
    tools/cpu_slot.sh python -m s1_rings.build                 # the product (needs the path
                                                               # test and the synthetic test)

Inputs: the 14 L0 ring files (data_fetch_api/new_measurements/compaction, hashed in
l0_registry/v1), deep_benchmarks/v2 daily vendor values (after_the_fact_analysis: path shape
and pile-implied scales), deep_benchmarks_printed/v2 printed_monthly (validator: the 2022
feed-month check), the family-Y path terms and the synthetic test result (evidence files).
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
from common import manifest  # noqa: E402
from common.flags import validate_long_table  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from scipy.stats import norm  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1io  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent
FIRST_POST = "2023-01-01"
LAST_MONTH = pd.Period("2025-05", "M")
OFFSET_CENTRE = {"A": -0.159, "B": 0.52}
PATH_TERMS = K.EVIDENCE / "path_test" / "path_terms.json"
SYN_RESULT = K.EVIDENCE / "synthetic" / "result.json"
ASSIGN_OUT = K.EVIDENCE / "assignment.json"


def post_win(site):
    return K.POST_WIN_BEICHEN if site == K.BEICHEN else K.POST_WIN


def ppm(x):
    return None if x is None or not np.isfinite(x) else round(1e6 * float(x), 1)


def cm(x):
    return None if x is None or not np.isfinite(x) else round(100 * float(x), 2)


# ----------------------------------------------------------------------------- evidence
def feed_check(piles: dict) -> dict:
    """Vendor monthly-mean increments vs printed 007001 values (flag 0) for the path sensors;
    > 2 mm in any 2022 month -> that sensor's path falls back to climatology."""
    pm, notes = s1io.printed_monthly()
    pm = pm[pm["flag"] == 0]
    out = {"guard_notes": notes, "sensors": {}}
    for sensor, _depth, use in K.PILES.values():
        if use == "evidence" or sensor in out["sensors"]:
            continue
        q = piles[sensor]
        mm = q.groupby(q.index.to_period("M")).mean()
        inc = mm.diff()
        rows = []
        for m in pd.period_range("2021-12", "2023-02", freq="M"):
            pv = pm[(pm.sensor_id == sensor) & (pm.month == str(m))]["value"]
            if pv.empty or m not in inc.index or not np.isfinite(inc.get(m, np.nan)):
                continue
            rows.append({"month": str(m), "printed_mm": float(pv.iloc[0]),
                         "feed_mm": round(float(inc[m]), 3),
                         "diff_mm": round(float(inc[m] - pv.iloc[0]), 3),
                         "in_rule": m.year == 2022})
        d = np.array([r["diff_mm"] for r in rows if r["in_rule"]])
        out["sensors"][sensor] = {"months": rows, "n_rule_months": len(d),
                                  "max_abs_diff_mm": float(np.abs(d).max()) if len(d) else None,
                                  "ok": bool(len(d) and np.abs(d).max() <= 2.0),
                                  "rule": "2022 months only (declaration path_2022); 2021-12 "
                                          "and 2023-02 listed for information"}
    return out


def real_evaluation(piles: dict, offset_lr=K.OFFSET_LR, use_lines=("offset", "b_deep", "pile"),
                    extra_lines=None):
    pos = {s: L.load_positions(s) for s in K.SITES}
    pre = {s: K.PRE_WIN for s in K.SITES}
    post = {s: post_win(s) for s in K.SITES}
    pl = {s: (piles[v[0]], v[1]) for s, v in K.PILES.items()}
    pw = {s: (K.PILE_PRE_WIN, K.PILE_POST_WIN) for s in K.PILES}
    res = L.evaluate(pos, pre, post, pl, set(K.FORCED_BOTH), pile_wins=pw,
                     offset_lr=offset_lr, use_lines=use_lines, extra_lines=extra_lines)
    return pos, res


def p_A_rule(res) -> dict:
    a = [r.assign["assigned"] for r in res.values()]
    nA, nB = a.count("A"), a.count("B")
    return {"n_A": nA, "n_B": nB, "p_A": nA / (nA + nB) if nA + nB else 10 / 14}


def site_row(s, r) -> dict:
    gs = r.gs
    row = {"site": s, "rings_used": len(gs["rings"]), "top_ring": gs["top"],
           "bottom_ring": gs["bot"], "z_top_m": round(gs["z_top"], 2),
           "z_bottom_m": round(float(gs["z"][-1]), 2), "L_m": round(gs["L"], 2),
           "last_pre_survey": str(gs["t_lp"].date()), "first_post_survey": str(gs["t_fp"].date()),
           "gap_yr": round(gs["gap"], 3), "J_top_mid_cm": cm(gs["J_top_mid"]),
           "se_J_top_cm": cm(gs["se_J_top"]), "offset_cluster": r.lines["offset"]["cluster"],
           "offset_vote": r.lines["offset"]["vote"],
           "b_deep_ppm": ppm(gs["b_deep"]), "b_deep_sd_ppm": ppm(gs["sd_deep"]),
           "n_deep": gs["n_deep"], "deep_len_m": round(gs["deep_len"], 1),
           "b_deep_lr": round(r.lines["b_deep"]["lr"], 4), "b_deep_vote": r.lines["b_deep"]["vote"]}
    pe = r.pile
    if pe is not None:
        row.update({"pile_sensor": K.PILES[s][0], "pile_depth_m": K.PILES[s][1],
                    "pile_ok": bool(pe.get("ok")), "pile_reason": pe.get("reason"),
                    "b_pile_ppm": ppm(pe.get("b", np.nan)), "b_pile_sd_ppm": ppm(pe.get("sd", np.nan)),
                    "pile_ratio_pre": round(pe["ratio_pre"], 3) if pe.get("ok") else None,
                    "pile_ratio_post": round(pe["ratio_post"], 3) if pe.get("ok") else None,
                    "pile_gap_mm": round(pe["pile_gap_mm"], 2) if pe.get("ok") else None,
                    "ring_gap_raw_mm": round(pe["ring_gap_raw_mm"], 2) if pe.get("ok") else None,
                    "pile_interval": f"{pe['top']}-{pe['bot']}"})
        if "pile" in r.lines:
            row.update({"pile_lr": round(r.lines["pile"]["lr"], 4),
                        "pile_vote": r.lines["pile"]["vote"]})
    a = r.assign
    row.update({"votes_A": a["votes_A"], "votes_B": a["votes_B"],
                "evidence_group": a["evidence_group"], "forced_both": a["forced_both"],
                "assigned": a["assigned"], "p_A": round(r.p_A, 6),
                "members_carried": "A" if a["assigned"] == "A" else
                ("B" if a["assigned"] == "B" else "A,B"),
                "dS_obs_cm": cm(gs["dS_obs"]), "sd_s_mm": round(1000 * gs["sd_s"], 2)})
    for g in "AB":
        m = r.members[g]
        row.update({f"b_{g}_ppm": ppm(m["b"]), f"sigma_b_{g}_ppm": ppm(m["sigma"]),
                    f"pool_{g}_n": m["pool"]["n"], f"pool_{g}_tau_ppm": ppm(m["pool"]["tau"]),
                    f"a_{g}_cm": cm(m["a"]), f"C_gap_{g}_cm": cm(m["C"]),
                    f"sigma_C_{g}_cm": cm(m["sigma_C"])})
    row.update({"primary": r.primary["tag"], "C_gap_primary_cm": cm(r.primary["C"]),
                "sigma_C_primary_cm": cm(r.primary["sigma_C"]),
                "C_rate_mid_cm": cm(gs["C_mid"]), "C_rate_lo_cm": cm(gs["C_lo"]),
                "C_rate_hi_cm": cm(gs["C_hi"])})
    return row


# ----------------------------------------------------------------------------- series
def intervals(site, gs):
    rings, zbar = gs["rings"], gs["zbar"]
    ivs = {"total": (rings[0], rings[-1])}
    labels = {"total": f"{zbar.iloc[0]:.0f}-{zbar.iloc[-1]:.0f}m"}
    for i, (a, b) in enumerate(L.band_indices(zbar)):
        ivs[f"band{i}"] = (rings[a], rings[b])
        labels[f"band{i}"] = f"{zbar.iloc[a] - zbar.iloc[0]:.0f}-{zbar.iloc[b] - zbar.iloc[0]:.0f}m"
    if site in K.PILES and K.PILES[site][2] == "shallow":
        r70 = str((zbar - K.PILES[site][1]).abs().idxmin())
        ivs["shallow"] = (rings[0], r70)
        ivs["deep"] = (r70, rings[-1])
    return ivs, labels


def path_term(terms, name, method):
    key = f"{name}.{method}" if name != "total" else f"total.{method}"
    return float(terms[key]["rmse_mm"]) / 1000.0


def member_series(site, p, gs, g, mem, terms, piles, path_method, flag_post):
    """Monthly series of every interval for member g: dict name -> DataFrame."""
    q = L.corrected_positions(p, mem["a"], mem["b"], FIRST_POST)
    mpos, mdates = L.monthly(q)
    mraw, _ = L.monthly(p)
    pre_days = pd.DatetimeIndex(p.loc[K.PRE_WIN[0]:K.PRE_WIN[1]].index).day
    vday = int(np.median(pre_days))
    ivs, labels = intervals(site, gs)
    months = pd.period_range(mpos.index.min(), LAST_MONTH, freq="M")
    out = {}
    raw_ser = {}
    for name, (a, b) in ivs.items():
        s = L.interval_series(mpos, a, b).reindex(months)
        raw_ser[name] = s
    for name, (a, b) in ivs.items():
        s = raw_ser[name]
        zl = (gs["zbar"][b] - gs["zbar"][a])
        sep_pre = (p[b] - p[a]).loc[K.PRE_WIN[0]:K.PRE_WIN[1]]
        f = L.window_fit(L.years(sep_pre.index), sep_pre.to_numpy(), 21.0)
        sd_s = f.resid_sd if f is not None else float("nan")
        v = s.dropna()
        t0 = v.loc[:pd.Period(K.GAP_LAST_PRE[:7], "M")].index[-1]
        t1 = v.loc[pd.Period(FIRST_POST[:7], "M"):].index[0]
        gap_m = months[(months > t0) & (months < t1)]
        dates = pd.Series(pd.NaT, index=months, dtype="datetime64[ns]")
        dd = mdates.reindex(months)
        dates.loc[dd.notna()] = dd[dd.notna()]
        vd = L.virtual_dates(gap_m, vday)
        dates.loc[gap_m] = vd
        clim = L.climatology(s)
        if name == "total" and path_method == "pile":
            vals, tau = L.fill(s[t0], s[t1], dates[t0], dates[t1], vd, "pile",
                               q=piles[K.PILES[site][0]])
            meth, fflag = (f"pile-shaped 2022 path ({K.PILES[site][0]} "
                           f"{K.PILES[site][1]:.0f} m, unit slope)"), 3
            term = path_term(terms, "total", "pile")
        elif name == "total" and path_method == "hybrid":
            sh, de = raw_ser["shallow"], raw_ser["deep"]
            lin, tau = L.fill(s[t0], s[t1], dates[t0], dates[t1], vd, "linear")
            fs, _ = L.fill(sh[t0], sh[t1], dates[t0], dates[t1], vd, "pile",
                           q=piles[K.PILES[site][0]])
            fl, _ = L.fill(sh[t0], sh[t1], dates[t0], dates[t1], vd, "linear")
            fd, _ = L.fill(de[t0], de[t1], dates[t0], dates[t1], vd, "climatology",
                           clim=L.climatology(de))
            fdl, _ = L.fill(de[t0], de[t1], dates[t0], dates[t1], vd, "linear")
            vals = lin + (fs - fl) + (fd - fdl)
            meth = (f"hybrid 2022 path: shallow band pile-shaped ({K.PILES[site][0]} "
                    f"{K.PILES[site][1]:.0f} m) + deep band climatology")
            fflag, term = 3, path_term(terms, "total", "hybrid")
        else:
            vals, tau = L.fill(s[t0], s[t1], dates[t0], dates[t1], vd, "climatology", clim=clim)
            meth, fflag = "linear + seasonal climatology 2022 path (2014-05..2021-12)", 4
            term = path_term(terms, name if name.startswith("band") else "total", "climatology") \
                if name in ("total", "band0", "band1", "band2") else \
                path_term(terms, "total", "climatology")
        n_fill = len(gap_m)
        if n_fill + 1 > K.TESTED_GAP_MONTHS + 1:          # Beichen: 30 months
            term = term * math.sqrt((n_fill + 1) / K.TESTED_GAP_MONTHS)
            fflag = 6
            meth += f"; {n_fill}-month gap beyond the 13-month tested envelope"
        if not np.all(np.isfinite(vals)):
            vals, tau = L.fill(s[t0], s[t1], dates[t0], dates[t1], vd, "climatology", clim=clim)
            meth = "linear + seasonal climatology 2022 path (pile missing)"
            fflag = 4 if fflag != 6 else 6
            term = path_term(terms, "total", "climatology")
        df = pd.DataFrame({"month": months, "date": dates.to_numpy(), "value": s.to_numpy(),
                           "sd": np.nan, "flag": 8, "method": "missing (no survey)",
                           "tau": np.nan}).set_index("month")
        obs = s.notna()
        pre = obs & (months <= t0)
        post = obs & (months >= t1)
        df.loc[pre, "flag"] = 0
        df.loc[pre, "sd"] = sd_s
        df.loc[pre, "method"] = "observed (monthly median of surveys)"
        df.loc[post, "flag"] = flag_post
        df.loc[post, "sd"] = math.sqrt(sd_s ** 2 + (mem["sigma"] * zl) ** 2)
        df.loc[post, "method"] = (f"observed, instrument-corrected as group {g} "
                                  f"(b {1e6 * mem['b']:.1f} ppm, a {100 * mem['a']:.2f} cm)")
        df.loc[gap_m, "value"] = vals
        df.loc[gap_m, "tau"] = tau
        df.loc[gap_m, "sd"] = np.sqrt((mem["sigma"] * zl * tau) ** 2 + term ** 2)
        df.loc[gap_m, "flag"] = fflag
        df.loc[gap_m, "method"] = meth
        df["interval_m"] = labels.get(name, name)
        df["sd_obs"] = sd_s
        out[name] = df
    return out, {"t0": None}


def combine(series_by_g: dict, p_A: float, report: str | None, post_flag: int) -> dict:
    """Primary series: assigned member or mixture; sd from the mixture's second moment."""
    pg = {"A": p_A, "B": 1.0 - p_A}
    out = {}
    for name in series_by_g["A"]:
        A, B = series_by_g["A"][name], series_by_g["B"][name]
        df = A.copy()
        if report is None:
            val = pg["A"] * A["value"] + pg["B"] * B["value"]
        else:
            val = series_by_g[report][name]["value"]
        var = sum(pg[g] * (series_by_g[g][name]["sd"] ** 2
                           + (series_by_g[g][name]["value"] - val) ** 2) for g in "AB")
        df["value"] = val
        changed = df["flag"].isin([1, 3, 4, 6]) & ~(A["flag"] == 0)
        df.loc[changed, "sd"] = np.sqrt(var[changed])
        post = (A["flag"].isin([1, 3])) & (A["method"].str.startswith("observed, instrument"))
        df.loc[post, "flag"] = post_flag
        tag = "mixture of members" if report is None else f"group {report} (assigned)"
        df.loc[post, "method"] = (f"observed, instrument-corrected; {tag}, "
                                  f"P(A) {p_A:.4f}")
        fillm = df["flag"].isin([3, 4, 6]) & ~post
        if report is None:
            df.loc[fillm, "method"] = df.loc[fillm, "method"] + f"; mixture P(A) {p_A:.4f}"
        out[name] = df
    return out


def long_rows(site, series_tag, dfs, p_A, version):
    rows = []
    for name, df in dfs.items():
        if name in ("shallow", "deep"):
            continue
        d = df.reset_index()
        d["site"] = site
        d["series"] = series_tag
        d["band"] = name
        d["p_A"] = round(p_A, 6)
        rows.append(d)
    x = pd.concat(rows, ignore_index=True)
    x["month"] = x["month"].astype(str)
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
    x["value_mm"] = (1000.0 * x["value"]).round(3)
    x["sd_mm"] = (1000.0 * x["sd"]).round(3)
    x["value"] = x["value_mm"]
    x["sd"] = x["sd_mm"]
    x["product_version"] = version
    x["source_ids"] = (f"data_fetch_api/new_measurements/compaction/{site}.parquet"
                       + (f";deep_benchmarks/v2:{K.PILES[site][0]}"
                          if site in K.PILES and K.PILES[site][2] != "evidence" else ""))
    return x[["site", "series", "band", "interval_m", "month", "date", "value", "sd", "flag",
              "method", "p_A", "tau", "product_version", "source_ids"]]


# ----------------------------------------------------------------------------- validation
def test_a(res) -> dict:
    zs, shifts, ss, rows = [], [], [], []
    for s, r in res.items():
        gs = r.gs
        smid = math.sqrt(((gs["C_hi"] - gs["C_lo"]) / math.sqrt(12)) ** 2
                         + gs["gap"] ** 2 * (gs["se_r_pre"] ** 2 + gs["se_r_post"] ** 2) / 4)
        si = math.sqrt(r.primary["sigma_C"] ** 2 + smid ** 2)
        z = (r.primary["C"] - gs["C_mid"]) / si
        shift = K.SCALE_ERROR * r.primary["b_eff"] * gs["S_fp"]
        zs.append(z)
        shifts.append(shift)
        ss.append(si)
        rows.append({"site": s, "C_primary_cm": cm(r.primary["C"]), "C_mid_cm": cm(gs["C_mid"]),
                     "sigma_primary_cm": cm(r.primary["sigma_C"]), "sigma_mid_cm": cm(smid),
                     "z": round(z, 3), "shift_25pct_cm": cm(shift),
                     "inside_rate_bracket": bool(gs["C_lo"] <= r.primary["C"] <= gs["C_hi"])})
    n = len(zs)
    Z = float(np.sum(zs) / math.sqrt(n))
    mu = float(np.sum(np.array(shifts) / np.array(ss)) / math.sqrt(n))
    power = float(norm.cdf(mu - K.POOLED_Z_MAX) + norm.cdf(-mu - K.POOLED_Z_MAX))
    return {"pooled_z": Z, "n": n, "mu_25pct": mu, "power_25pct": power,
            "pass_z": abs(Z) < K.POOLED_Z_MAX, "pass_power": power >= K.POWER_MIN,
            "pass": bool(abs(Z) < K.POOLED_Z_MAX and power >= K.POWER_MIN),
            "n_inside_rate_bracket": int(sum(r["inside_rate_bracket"] for r in rows)),
            "sites": rows}


def test_b(res) -> dict:
    assigned = {s: r.assign["assigned"] for s, r in res.items()}
    pools = {}
    for g in "AB":
        sites = [s for s, a in assigned.items() if a == g and np.isfinite(res[s].gs["b_deep"])]
        pools[g] = {**L.pool([res[s].gs["b_deep"] for s in sites],
                             [res[s].gs["sd_deep"] for s in sites]), "sites": sites}
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
        rows.append({"site": s, "tested": True, "group": g, "group_source":
                     "assigned" if r.assign["assigned"] else "more probable (unassigned)",
                     "b_pile_ppm": ppm(pe["b"]), "b_pile_sd_ppm": ppm(pe["sd"]),
                     "pooled_ppm": ppm(m), "tolerance": tol, "pass": bool(ok),
                     "pile_shapes_path": K.PILES[s][2] in ("total", "shallow")})
    tested = [r for r in rows if r.get("tested")]
    return {"pools": {g: {"mean_ppm": ppm(v["mean"]), "se_ppm": ppm(v["se"]),
                          "tau_ppm": ppm(v["tau"]), "n": v["n"], "sites": v["sites"]}
                      for g, v in pools.items()},
            "sites": rows, "n_tested": len(tested),
            "pass": bool(tested and all(r["pass"] for r in tested)),
            "note": "not independent where the pile is also a group-evidence line (every pile "
                    "site) or shapes the 2022 path (Tuku, Xizhou, Guangfu, Xiutan); plan 4.1"}


def test_resid(pos, res) -> dict:
    rows = []
    for s, r in res.items():
        g = r.assign["assigned"] or ("A" if r.p_A >= 0.5 else "B")
        m = r.members[g]
        q = L.corrected_positions(pos[s], m["a"], m["b"], FIRST_POST)
        ro = L.residual_offset(q, r.gs["rings"], r.gs["t_lp"], r.gs["t_fp"])
        rows.append({"site": s, "group": g, "offset_cm": cm(ro["offset_m"]),
                     "scale_ppm": ppm(ro["scale"]), "n_rings": ro["n_rings"],
                     "within_1cm": bool(abs(ro["offset_m"]) <= K.RESID_OFFSET_M)})
    n_ok = sum(r["within_1cm"] for r in rows)
    return {"sites": rows, "n_within": n_ok, "n": len(rows),
            "pass": n_ok >= K.RESID_OFFSET_MIN_SITES}


def sensitivity(res, piles) -> dict:
    rows = []
    for s, r in res.items():
        J = r.gs["J_top_mid"]
        g_off = min("AB", key=lambda g: abs(J - OFFSET_CENTRE[g]))
        C_off = r.members[g_off]["C"]
        rows.append({"site": s, "group_offset_only": g_off, "primary": r.primary["tag"],
                     "C_primary_cm": cm(r.primary["C"]), "C_offset_only_cm": cm(C_off),
                     "delta_cm": cm(C_off - r.primary["C"])})
    lr = {}
    for v in (3.0, 30.0):
        _, rv = real_evaluation(piles, offset_lr=v)
        lr[str(v)] = [{"site": s, "p_A": round(rv[s].p_A, 4), "assigned": rv[s].assign["assigned"],
                       "C_primary_cm": cm(rv[s].primary["C"]),
                       "delta_vs_main_cm": cm(rv[s].primary["C"] - res[s].primary["C"])}
                      for s in res]
    return {"offset_only": rows, "offset_lr": lr}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-only", action="store_true")
    args = ap.parse_args(argv)
    K.EVIDENCE.mkdir(parents=True, exist_ok=True)
    piles, pile_notes = s1io.piles(preorigin=False)
    pos, res = real_evaluation(piles)
    rows = [site_row(s, r) for s, r in res.items()]
    pa = p_A_rule(res)
    if args.assign_only:
        ASSIGN_OUT.write_text(json.dumps({"p_A_rule": pa, "sites": rows,
                                          "pile_guard_notes": pile_notes},
                                         indent=1, ensure_ascii=False, default=str))
        print(json.dumps(pa))
        print(pd.DataFrame(rows)[["site", "J_top_mid_cm", "offset_vote", "b_deep_ppm",
                                  "b_deep_sd_ppm", "b_deep_vote", "b_pile_ppm", "b_pile_sd_ppm",
                                  "pile_vote", "votes_A", "votes_B", "assigned", "p_A"]]
              .to_string())
        return
    terms = json.loads(PATH_TERMS.read_text())["terms"]
    syn = json.loads(SYN_RESULT.read_text())
    fc = feed_check(piles)
    d = manifest.new_version_dir(K.PRODUCT)
    version = f"{K.PRODUCT}/{d.name}"
    long_all, corr_files = [], []
    for s, r in res.items():
        a = r.assign
        use = K.PILES.get(s, (None, None, None))[2]
        sensor = K.PILES.get(s, (None,))[0]
        path_method = {"total": "pile", "shallow": "hybrid"}.get(use, "climatology")
        if path_method != "climatology" and not fc["sensors"].get(sensor, {}).get("ok", False):
            path_method = "climatology"
        ser = {}
        for g in "AB":
            fpost = 1 if (a["assigned"] == g) else 3
            ser[g], _ = member_series(s, pos[s], r.gs, g, r.members[g], terms, piles,
                                      path_method, fpost)
        post_flag = 1 if a["assigned"] else 3
        prim = combine(ser, r.p_A, a["assigned"], post_flag)
        long_all.append(long_rows(s, "primary", prim, r.p_A, version))
        carried = [a["assigned"]] if a["assigned"] else ["A", "B"]
        if not a["assigned"]:
            for g in "AB":
                long_all.append(long_rows(s, f"member_{g}", ser[g], r.p_A, version))
        for g in carried:
            m = r.members[g]
            q = L.corrected_positions(pos[s], m["a"], m["b"], FIRST_POST)
            raw = pos[s]
            lr = raw.stack(future_stack=True).rename("depth_raw_m").reset_index()
            lr.columns = ["datetime", "ring", "depth_raw_m"]
            lc = q.stack(future_stack=True).rename("depth_corrected_m").reset_index()
            lr["depth_corrected_m"] = lc["depth_corrected_m"].to_numpy()
            lr = lr.dropna(subset=["depth_raw_m"])
            post = lr["datetime"] >= pd.Timestamp(FIRST_POST)
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
            corr_files.append(name)
    LT = pd.concat(long_all, ignore_index=True)
    probs = validate_long_table(LT.rename(columns={}), require_all_columns=True)
    LT[LT.band == "total"].to_csv(d / "ring_total_monthly_bridged.csv", index=False)
    LT.to_parquet(d / "ring_bands_monthly_bridged.parquet", index=False)
    C = pd.DataFrame(rows)
    C.to_csv(d / "ring_corrections.csv", index=False)
    va, vb, vr = test_a(res), test_b(res), test_resid(pos, res)
    sens = sensitivity(res, piles)
    e_pass = bool(syn["primary"]["pass"])
    gate = bool(va["pass"] and vb["pass"] and vr["pass"] and e_pass)
    val = {"wp": "S1a", "rules": "S1a.rings_bridged_rules.v1", "synthetic_test":
           "S1a.e_synthetic_mixed.v1", "p_A_rule": pa,
           "a_pooled_standardized_difference": va, "b_pile_implied_scale": vb,
           "residual_offset": vr, "e_synthetic": {k: syn[k] for k in ("primary", "stress")
                                                  if k in syn},
           "misassignment_sensitivity": sens, "feed_check_2022": fc,
           "path_terms_mm": {k: v["rmse_mm"] for k, v in terms.items()},
           "long_table_problems": probs,
           "gate": {"passed": gate, "e": e_pass, "a": va["pass"], "b": vb["pass"],
                    "residual_offset": vr["pass"],
                    "note": "e's 1.5 cm threshold is post hoc (plan)"}}
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, default=str))
    readme = readme_text(version, C, va, vb, vr, syn, gate, terms)
    (d / "README.md").write_text(readme)
    inputs = [rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet") for s in K.SITES]
    inputs += [rel_to_repo(K.DEEP_BENCH_V2 / "daily.parquet"),
               rel_to_repo(K.DEEP_BENCH_V2 / "MANIFEST.json"),
               rel_to_repo(K.PRINTED_V2 / "printed_monthly.parquet"),
               rel_to_repo(K.PRINTED_V2 / "MANIFEST.json"),
               rel_to_repo(PATH_TERMS), rel_to_repo(SYN_RESULT),
               "results/dataprep/preregistration/S1a.rings_bridged_rules.v1.json",
               "results/dataprep/preregistration/S1a.e_synthetic_mixed.v1.json"]
    code = [rel_to_repo(HERE / n) for n in ("s1const.py", "s1lib.py", "s1io.py", "build.py",
                                            "path_test.py", "synthetic.py")]
    code += ["dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
             "dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py"]
    m = manifest.build_manifest(
        d, product=K.PRODUCT, layer="L2", inputs=inputs, code=code,
        parameters={"rules": "S1a.rings_bridged_rules.v1", "forced_both": sorted(K.FORCED_BOTH),
                    "reference_groups_ppm": {"A": [182.2, 11.4], "B": [1.9, 14.3]},
                    "offset_clusters_m": K.OFFSET_CLUSTERS, "offset_lr": K.OFFSET_LR,
                    "lr_vote": K.LR_VOTE, "lr_cap": K.LR_CAP, "z_plausible": K.Z_PLAUSIBLE,
                    "pre_win": K.PRE_WIN, "post_win": K.POST_WIN,
                    "post_win_beichen": K.POST_WIN_BEICHEN,
                    "path_terms_mm": {k: v["rmse_mm"] for k, v in terms.items()},
                    "input_caveats": [
                        "deep_benchmarks/v2 (gate failed on test A, 3/61 units) is read as "
                        "after_the_fact_analysis for the pile-implied scales and the 2022 path "
                        "shape; its 2022 vendor monthly-mean increments of the path sensors "
                        "were re-checked here against deep_benchmarks_printed/v2 (gate passed; "
                        "validation.json feed_check_2022) and S2 found the 2022 feed months "
                        "reproduce the agency values (62/62).",
                        "GNSS (gnss_clean, gate failed) is not used by the product."],
                    "membership_prior_A": K.PRIOR_A},
        fit_window_end="2025-05", validation_report="validation.json",
        gate={"passed": gate, "criteria": "plan 3 WP S1 (S1a): (e) worst-window RMSE <= 1.5 cm "
              "(post hoc) in the mixed A/B synthetic rerun; (a) |pooled standardized "
              "difference| < 2 with power >= 0.8 for a 25 % scale error; (b) pile-implied "
              "scale within 25 % (A) / 30 ppm (B) of the pooled scale; residual offset within "
              "+-1 cm at >= 12 of 14 sites (declaration S1a.rings_bridged_rules.v1)"},
        allowed_uses=["validator", "scoring_target", "after_the_fact_analysis"],
        forbidden_uses=["calibration_target", "datum_fit", "baseline", "initial_conditions",
                        "idw_field", "prior", "prior_width_input", "model_design", "selection",
                        "forcing_frozen", "forcing_recorded_arm", "prospective_scorer",
                        "statistical_layer_history", "design_time_evidence"],
        personal_data="none",
        extra={"product_card": "README.md", "series_conventions": {
            "ring_holdout": "plan 5.5: an ALTERNATIVE series to the ring_holdout re-reference "
                            "at the first 2023 survey; report side by side, never mix in one "
                            "score",
            "flags": "0 observed; 1 corrected, group independently assigned; 3 corrected "
                     "under an unassigned / forced group or pile-shaped 2022 path; 4 "
                     "climatology 2022 path; 6 Beichen's 30-month path; 8 missing"}})
    errs = manifest.validate_manifest(m, d)
    if errs:
        raise SystemExit(f"manifest errors: {errs}")
    manifest.write_manifest(d, m)
    print(d, "gate", gate)
    print(json.dumps(val["gate"]))


def readme_text(version, C, va, vb, vr, syn, gate, terms) -> str:
    asg = C[["site", "assigned", "forced_both", "p_A", "C_gap_primary_cm",
             "sigma_C_primary_cm"]].to_string(index=False)
    return f"""# {version}: compaction rings with the 2022 instrument change corrected and the gap bridged

WP S1a (plan 3, WP S1; rules results/dataprep/preregistration/S1a.rings_bridged_rules.v1.json).
Code: dataprep_staging/s1_rings/ (build.py, s1lib.py).

## What it is

After the 2022 survey hiatus every ring depth reads z_post = a_site + (1 + b_group) z_true.
Post-gap depths are corrected with z' = (z - a)/(1 + b); the group (A: about +180 ppm, B: about
0 ppm) is assigned from the offset cluster, the site's own deep-interval scale and the
pile-implied scale (>= 2 agreeing lines, none against, not on the plan's forced list); other
sites are carried as both members with a membership probability. The months between the last
2021 and the first 2023 survey are bridged: pile-shaped at Tuku, Xizhou and Guangfu, Xiutan's
shallow band pile-shaped, elsewhere linear + seasonal climatology.

This is a DATA product for post-origin uses: an alternative ring series to the ring_holdout
re-reference (plan 5.5), reported side by side and never mixed in one score. fit_window_end
2025-05; no pre-origin use is allowed.

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
- validation.json: checks (a), (b), (e), residual offset, sensitivities, 2022 feed check.

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
residual sd of the 2019-21 fit (an upper bound). UNCALIBRATED except as noted in validation.json
(synthetic coverage with 14 clusters is below the A1 envelope).

## Gate: {gate}

(a) pooled Z {va['pooled_z']:.2f} (power {va['power_25pct']:.2f}) pass {va['pass']}; (b) pass
{vb['pass']} ({vb['n_tested']} pile sites); residual offset {vr['n_within']}/{vr['n']} within 1 cm
pass {vr['pass']}; (e) worst-window RMSE {100 * syn['primary']['worst_window_rmse_m']:.2f} cm
(threshold 1.5 cm, post hoc) pass {syn['primary']['pass']}.
"""


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
