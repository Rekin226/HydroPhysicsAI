"""WP S1b repair round r3 (S1br): synthetic operating characteristics on FRESH seeds.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r3_synth.py N1_sd_s
    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r3_synth.py N2_resid

Answers the r2 verifier's major issue: for every bracket reading of ``s1b_r2_core.RULES`` and
every scenario, how often does the WRA family (and InSAR) contain the product's value, and with
what count distribution, under two ring-noise models:

* ``N1_sd_s``: ring survey noise iid N(0, sd_s) (the product's sd, an upper bound; r2's model);
* ``N2_resid``: iid N(0, s_res), s_res = residual sd of the ring's own trend + annual fit in the
  pre-gap window (the verifier's sensitivity; still an upper bound on survey noise).

The bracket's sigma always uses the product's sd_s, as the registered rule does on real data.
Truth model, GNSS and InSAR noise are the round-1 generator's (``s1b_synth.truth``,
``ring_true``, ``excess_true``, ``ar1``, ``members``; imported read-only). Availability comes
from the REPAIRED products: rings_bridged/v3 survey dates, gnss_clean/full/v4 flag-0/1/3 days,
station noise and in-gap step se, insar_069A/v3 valid epochs under the v3 pixel rule (one
network segment, so no bridge offsets). Every compaction, height and displacement value is
synthetic; the ring values are read only for the residual sd of N2.

Scenarios (truth = the product's more probable group unless stated):
``correct`` (the true group's member), ``other``, ``raw``, ``up25`` / ``down25`` (r2's uniform
+-0.25 x 182.2 ppm x L at every site), ``up25A`` / ``down25A`` (the same shift only where the
truth is A), ``mix`` (the product's P(A)-weighted mixture), ``mix_drawn`` (the mixture, truth
drawn with the product's P(A)).

Writes ``S1b_evidence/repair_r3/synthetic/oc_<model>.json``.
"""
from __future__ import annotations

import json
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
from s1_rings.s1b import s1b_io as IO1  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r3_io as IO3  # noqa: E402
from s1_rings.s1b import s1b_synth as S  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

OUT = K.EVIDENCE / "repair_r3" / "synthetic"
MODELS = {"N1_sd_s": 18031, "N2_resid": 18032}
TRUTH_SEED_OFFSET = 100
N_REAL = 400
SCEN = ("correct", "other", "raw", "up25", "down25", "up25A", "down25A", "mix", "mix_drawn")
FAMILIES = ("wra", "wra_noSJES", "insar")


def availability() -> tuple[dict, dict, dict, dict, dict]:
    rings, _ = IO3.rings(IO3.RINGS_V3)
    rsd = {s: IO3.ring_resid_sd(r) for s, r in rings.items()}
    for r in rings.values():
        r["c"] = None
        r["C"] = None
    xy = IO1.site_xy()
    g = IO3.gnss_availability(IO3.GNSS_FULL_V4)
    prs = IO1.pairs(g["stations"], xy)
    gav = {}
    for p in prs["wra_colocated"]:
        d = g["daily"]
        x = d[(d.family == p["family"]) & (d.station == p["station"])
              & d["flag"].isin(list(K.GNSS_FLAGS))]
        st = g["stations"][(g["stations"].family == p["family"])
                           & (g["stations"].station == p["station"])].iloc[0]
        r = rings[p["site"]]
        sp = g["steps"]
        sp = sp[(sp.family == p["family"]) & (sp.station == p["station"])]
        vs = 0.0
        for q in sp.itertuples():
            if r["t_lp"] < pd.Timestamp(q.date) < r["t_fp"]:
                se = float(q.se_m) if np.isfinite(float(q.se_m)) else 0.0
                vs += (1000.0 * se) ** 2
        gav[(p["site"], p["station"])] = {
            "t": pd.DatetimeIndex(x["date"]).sort_values(),
            "sigma_mm": 1000.0 * float(st["sigma_hf_m"]), "rho": float(st["rho_hf"]),
            "var_steps": float(vs)}
    g["daily"] = None
    ins = IO3.insar(IO3.INSAR_V3, values=False)
    iav = {}
    for s in K.SITES:
        x, y = xy[s]
        px = IO3.insar_pixel_series(ins, x, y, values=False)
        ok = px["n_valid"] >= K.INSAR_PIX_MIN_VALID
        iav[s] = {"t": ins["dates"], "seg": np.zeros(len(ins["dates"]), dtype=int), "ok": ok,
                  "n_ok_epochs": int(ok.sum())}
    return rings, rsd, prs, gav, iav


def synth_ring(rng, tr: dict, r: dict, b_ppm: float, noise_sd: float):
    """``s1b_synth.synth_ring`` with the ring noise sd as an argument."""
    t = C.years(r["t"])
    c = S.ring_true(tr, t) + rng.normal(0, noise_sd, len(t))
    tt = pd.DatetimeIndex(r["t"])
    c = c - (tt >= pd.Timestamp(r["t_fp"])) * b_ppm * 1e-6 * r["L_mm"]
    i_lp = int(np.flatnonzero(tt == pd.Timestamp(r["t_lp"]))[0])
    i_fp = int(np.flatnonzero(tt == pd.Timestamp(r["t_fp"]))[0])
    return c, float(c[i_fp] - c[i_lp])


def one(rng, kind: str, r: dict, avail: dict, grp: str, noise_sd: float) -> dict | None:
    """One synthetic site realization with the true group ``grp``: the bracket ends, their
    sigmas (with the product's sd_s), the raw change and both members."""
    y_lp, y_fp = C.years([r["t_lp"]])[0], C.years([r["t_fp"]])[0]
    tr = S.truth(rng, y_lp, y_fp)
    c, raw = synth_ring(rng, tr, r, S.B_PPM[grp], noise_sd)
    mem = S.members(raw, r["L_mm"])
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
    return {"mem": mem, "raw": raw, "G": res["G_mm"], "gap_yr": y_fp - y_lp,
            "C_pre": chk["C_pre_mm"], "C_post": chk["C_post_mm"],
            "s_pre": chk["sigma_pre_mm"], "s_post": chk["sigma_post_mm"]}


def scenario_values(o_prob: dict, o_drawn: dict, grp: str, p_A: float, L_mm: float) -> dict:
    """{scenario: (value, draw)}: the product value under each scenario and the realization
    whose bracket it is compared with."""
    other = "B" if grp == "A" else "A"
    dL = S.ALT_PPM * 1e-6 * L_mm
    m = o_prob["mem"]
    mix = p_A * m["A"] + (1.0 - p_A) * m["B"]
    md = o_drawn["mem"]
    return {"correct": (m[grp], o_prob), "other": (m[other], o_prob),
            "raw": (o_prob["raw"], o_prob), "up25": (m[grp] + dL, o_prob),
            "down25": (m[grp] - dL, o_prob),
            "up25A": (m[grp] + (dL if grp == "A" else 0.0), o_prob),
            "down25A": (m[grp] - (dL if grp == "A" else 0.0), o_prob),
            "mix": (mix, o_prob),
            "mix_drawn": (p_A * md["A"] + (1.0 - p_A) * md["B"], o_drawn)}


def family_eval(items: list[tuple], rule: str) -> dict:
    """items: [(value, draw, L_mm)] for one realization and scenario. Count inside, the
    containment-only rule and S1b's family rule (count AND RMSE clause)."""
    n = len(items)
    inside = [R.contains(rule, v, o["C_pre"], o["C_post"], o["s_pre"], o["s_post"], L)
              for v, o, L in items]
    cnt = int(sum(bool(x) for x in inside))
    need = math.ceil(K.FAMILY_MIN_PASS_FRAC * n)
    mid = [0.5 * (o["C_pre"] + o["C_post"]) for _, o, _ in items]
    rm_v = math.sqrt(np.mean([(v - m) ** 2 for (v, _, _), m in zip(items, mid, strict=True)]))
    rm_r = math.sqrt(np.mean([(o["raw"] - m) ** 2 for (_, o, _), m in zip(items, mid,
                                                                           strict=True)]))
    return {"count": cnt, "n": n, "contain_pass": cnt >= need,
            "family_pass": bool(cnt >= need and rm_v < rm_r)}


def d1_eval(items: list[tuple]) -> dict | None:
    if len(items) < 3:
        return None
    x = np.array([o["G"] / o["gap_yr"] for _, o, _ in items])
    yb = np.array([v / o["gap_yr"] for v, o, _ in items])
    yr = np.array([o["raw"] / o["gap_yr"] for _, o, _ in items])
    rb, rr = np.corrcoef(x, yb)[0, 1], np.corrcoef(x, yr)[0, 1]
    rc = C.r_crit(len(items), K.CORR_ALPHA)
    return {"crit": bool(rb >= rc), "gt_raw": bool(rb > rr), "both": bool(rb >= rc and rb > rr),
            "r_b": float(rb), "r_r": float(rr)}


def run(model: str, *, seed: int | None = None, n_real: int = N_REAL,
        out_path: Path | None = None) -> dict:
    seed = MODELS[model] if seed is None else seed
    rng = np.random.default_rng(seed)
    rng_truth = np.random.default_rng(seed + TRUTH_SEED_OFFSET)
    rings, rsd, prs, gav, iav = availability()
    sites = {"wra": [(p["site"], p["station"], gav[(p["site"], p["station"])])
                     for p in prs["wra_colocated"]],
             "insar": [(s, "069A", iav[s]) for s in K.SITES]}
    # realizations[family][k] = list of per-site dicts {site, grp, p_A, L, o_prob, o_drawn}
    reals = {f: [[] for _ in range(n_real)] for f in ("wra", "insar")}
    n_fail = {}
    for fam in ("wra", "insar"):
        for site, sta, av in sites[fam]:
            r = rings[site]
            noise = r["sd_s_mm"] if model == "N1_sd_s" else rsd[site]
            grp = "A" if r["p_A"] >= 0.5 else "B"
            nf = 0
            for k in range(n_real):
                draws = {}
                for gg in "AB":
                    o = one(rng, "gnss" if fam == "wra" else "insar", r, av, gg, noise)
                    if o is not None:
                        draws[gg] = o
                gd = "A" if rng_truth.uniform() < r["p_A"] else "B"
                if grp not in draws or gd not in draws:
                    nf += 1
                    continue
                reals[fam][k].append({"site": site, "station": sta, "grp": grp,
                                      "p_A": r["p_A"], "L": r["L_mm"],
                                      "sv": scenario_values(draws[grp], draws[gd], grp,
                                                            r["p_A"], r["L_mm"])})
            n_fail[f"{fam}:{site}"] = nf
    out = {"model": model, "seed": seed, "truth_seed": seed + TRUTH_SEED_OFFSET,
           "realizations": n_real, "rules": list(R.RULES), "scenarios": list(SCEN),
           "ring_noise_sd_mm": {s: (rings[s]["sd_s_mm"] if model == "N1_sd_s" else rsd[s])
                                for s in K.SITES},
           "ring_sd_s_mm": {s: rings[s]["sd_s_mm"] for s in K.SITES},
           "ring_resid_sd_mm": rsd, "not_evaluable_draws": n_fail,
           "insar_valid_epochs": {s: iav[s]["n_ok_epochs"] for s in K.SITES},
           "gnss_var_steps_mm2": {f"{k[0]}:{k[1]}": v["var_steps"] for k, v in gav.items()},
           "families": {}, "per_site": {}}
    fam_sets = {"wra": ("wra", None), "wra_noSJES": ("wra", "SJES"), "insar": ("insar", None)}
    for fname, (src, drop) in fam_sets.items():
        res = {}
        for rule in R.RULES:
            for sc in SCEN:
                counts, cpass, fpass = [], [], []
                for k in range(n_real):
                    it = [(x["sv"][sc][0], x["sv"][sc][1], x["L"]) for x in reals[src][k]
                          if x["station"] != drop]
                    if not it:
                        continue
                    e = family_eval(it, rule)
                    counts.append(e["count"])
                    cpass.append(e["contain_pass"])
                    fpass.append(e["family_pass"])
                n = len([1 for _, sta, _ in sites[src] if sta != drop])
                hist = np.bincount(np.asarray(counts, int), minlength=n + 1).tolist()
                res[f"{rule}|{sc}"] = {"family_pass": float(np.mean(fpass)),
                                       "contain_pass": float(np.mean(cpass)),
                                       "mean_count": float(np.mean(counts)),
                                       "count_hist": hist, "n_sites": n,
                                       "n_realizations": len(counts)}
        if src == "insar":
            for sc in ("correct", "mix", "mix_drawn"):
                cl = [d1_eval([(x["sv"][sc][0], x["sv"][sc][1], x["L"]) for x in reals[src][k]])
                      for k in range(n_real)]
                cl = [c for c in cl if c]
                res[f"d1|{sc}"] = {q: float(np.mean([c[q] for c in cl]))
                                   for q in ("crit", "gt_raw", "both")}
                res[f"d1|{sc}"]["median_r_b"] = float(np.median([c["r_b"] for c in cl]))
                res[f"d1|{sc}"]["median_r_r"] = float(np.median([c["r_r"] for c in cl]))
        out["families"][fname] = res
    for src in ("wra", "insar"):
        per = {}
        for site, sta, _ in sites[src]:
            xs = [x for k in range(n_real) for x in reals[src][k] if x["site"] == site]
            row = {"station": sta, "group": xs[0]["grp"] if xs else None, "n": len(xs)}
            for rule in R.RULES:
                for sc in SCEN:
                    v = [R.contains(rule, x["sv"][sc][0], x["sv"][sc][1]["C_pre"],
                                    x["sv"][sc][1]["C_post"], x["sv"][sc][1]["s_pre"],
                                    x["sv"][sc][1]["s_post"], x["L"]) for x in xs]
                    row[f"{rule}|{sc}"] = float(np.mean(v)) if v else None
            per[site] = row
        out["per_site"][src] = per
    path = out_path or (OUT / f"oc_{model}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(out, path)
    return out


def main(argv=None) -> int:
    from harness import prereg3

    from s1_rings.s1b import s1b_r3_looks as LK
    argv = sys.argv[1:] if argv is None else argv
    model = argv[0] if argv else ""
    if model == "smoke":
        LK.synth_smoke()
        out = run("N2_resid", seed=1, n_real=2,
                  out_path=K.EVIDENCE / "repair_r3" / "logs" / "smoke_synth.json")
        print(json.dumps(out["not_evaluable_draws"]), json.dumps(out["ring_resid_sd_mm"],
                                                                 ensure_ascii=False))
        return 0
    if model not in MODELS:
        raise SystemExit(f"usage: s1b_r3_synth.py {'|'.join(MODELS)}")
    doc, _ = prereg3.load_prereg(LK.DECL_ID)
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise SystemExit(f"declared inputs or code changed since {LK.DECL_ID}: {bad}")
    LK.synth(f"model {model}, seed {MODELS[model]}, truth seed "
             f"{MODELS[model] + TRUTH_SEED_OFFSET}, {N_REAL} realizations")
    out = run(model)
    for fam, res in out["families"].items():
        print(fam)
        for rule in R.RULES:
            print(f"   {rule:14s}", {sc: (round(res[f'{rule}|{sc}']['family_pass'], 3),
                                      round(res[f'{rule}|{sc}']['mean_count'], 2))
                                    for sc in SCEN})
        for sc in ("correct", "mix", "mix_drawn"):
            if f"d1|{sc}" in res:
                print("   d1", sc, res[f"d1|{sc}"])
    print(json.dumps(out["not_evaluable_draws"]))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
