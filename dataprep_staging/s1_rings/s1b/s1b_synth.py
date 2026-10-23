"""WP S1b: availability-preserving synthetic run (code test and operating characteristics).

Design-time, before the S1b registration; logged as an ``integrity_check`` look
(``s1b_looks.py availability``). The real inputs are read only for their dates, flags and
valid-pixel masks: the ring survey dates where both rings are read, the GNSS days with flags
0/1/3, the InSAR epochs with >= 4 valid pixels and the network segment ids. Every compaction,
height and displacement value is synthetic, with a known group scale:

* ring compaction: piecewise-linear rates (pre, gap, post) + an annual harmonic, iid noise
  with the site's sd_s (the product's upper bound), and the post-gap reading shifted by
  -b_true L (the raw gap change);
* instrument subsidence = ring compaction + an excess with its own piecewise rates (the gap
  rate drawn from 25 % below to 25 % beyond the pre/post range) and harmonic, plus noise:
  GNSS AR(1) daily noise with the station's sigma_hf and rho_hf; InSAR 6.6 mm per epoch plus a
  6 mm offset at every network gap;
* the product's members are C(b) = raw + b L for the declared group scales;
* GNSS stations whose product removed a step inside the gap get a post-gap offset error drawn
  with the steps' se (the registered rule widens the bracket by the same se).

Operating characteristics reported (design-time, before registration):

* per site: size (1 - containment of the correct member) and power against the other group's
  member and against the raw value;
* per family (the registered >= 80 % rule plus RMSE(primary) < RMSE(raw)): pass rate when the
  product carries the correct member at every site, and power (fail rate) when the product is
  wrong by a uniform +-25 % of the group-A scale (+-45.6 ppm x L) at every site, carries the
  other group's member at every site, or is the raw value;
* (d1) pass rate of the correlation rule; it depends on the synthetic cross-site spread
  (ring gap rates uniform 10-45 mm/yr, excess 0-40 mm/yr) and is indicative only.

The registered pipeline functions (``s1b_core.instrument_vs_ring``, ``site_check``,
``family_verdict``) run on these series; the output gives, per site and instrument, how often
the 2-sigma bracket contains the correct member (1 - size) and excludes the wrong one (power).

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_synth.py
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
from s1_rings.s1b import s1b_io as IO  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

DAYS = pd.date_range("2019-06-01", "2025-12-31", freq="D")
B_PPM = {"A": K.REF_PPM["A"][0], "B": K.REF_PPM["B"][0]}
ALT_PPM = 0.25 * K.REF_PPM["A"][0]          # uniform 25 % group-A scale error
ALTS = ("correct", "mix", "up25", "down25", "other", "raw")


def truth(rng, y_lp: float, y_fp: float) -> dict:
    r1, r2 = rng.uniform(10, 45), rng.uniform(5, 25)
    e1, e2 = rng.uniform(0, 40), rng.uniform(0, 40)
    return {"r": (r1, r1 + rng.uniform(0, 1) * (r2 - r1), r2),
            "e": (e1, e1 + rng.uniform(-0.25, 1.25) * (e2 - e1), e2),
            "Ac": rng.uniform(1, 6), "pc": rng.uniform(0, 1),
            "Ae": rng.uniform(0, 10), "pe": rng.uniform(0, 1), "y_lp": y_lp, "y_fp": y_fp}


def _pw(t: np.ndarray, rates, y_lp: float, y_fp: float) -> np.ndarray:
    r1, rg, r2 = rates
    return np.where(t < y_lp, r1 * (t - y_lp),
                    np.where(t < y_fp, rg * (t - y_lp), rg * (y_fp - y_lp) + r2 * (t - y_fp)))


def ring_true(tr: dict, t: np.ndarray) -> np.ndarray:
    return (_pw(t, tr["r"], tr["y_lp"], tr["y_fp"])
            + tr["Ac"] * np.sin(2 * np.pi * (t - tr["pc"])))


def excess_true(tr: dict, t: np.ndarray) -> np.ndarray:
    return (_pw(t, tr["e"], tr["y_lp"], tr["y_fp"])
            + tr["Ae"] * np.sin(2 * np.pi * (t - tr["pe"])))


def ar1(rng, n: int, sigma: float, rho: float) -> np.ndarray:
    from scipy.signal import lfilter
    rho = min(max(rho, 0.0), 0.95)
    e = rng.normal(0, sigma * math.sqrt(1 - rho ** 2), n)
    e[0] = rng.normal(0, sigma)
    return lfilter([1.0], [1.0, -rho], e)


def synth_ring(rng, tr: dict, r: dict, b_ppm: float) -> tuple[np.ndarray, float]:
    t = C.years(r["t"])
    c = ring_true(tr, t) + rng.normal(0, r["sd_s_mm"], len(t))
    post = pd.DatetimeIndex(r["t"]) >= pd.Timestamp(r["t_fp"])
    c = c - post * b_ppm * 1e-6 * r["L_mm"]
    i_lp = int(np.flatnonzero(pd.DatetimeIndex(r["t"]) == pd.Timestamp(r["t_lp"]))[0])
    i_fp = int(np.flatnonzero(pd.DatetimeIndex(r["t"]) == pd.Timestamp(r["t_fp"]))[0])
    return c, float(c[i_fp] - c[i_lp])


def members(raw: float, L_mm: float) -> dict:
    return {g: raw + B_PPM[g] * 1e-6 * L_mm for g in "AB"}


def one(rng, kind: str, r: dict, avail: dict, grp: str) -> dict | None:
    y_lp, y_fp = C.years([r["t_lp"]])[0], C.years([r["t_fp"]])[0]
    tr = truth(rng, y_lp, y_fp)
    c, raw = synth_ring(rng, tr, r, B_PPM[grp])
    mem = members(raw, r["L_mm"])
    other = "B" if grp == "A" else "A"
    dL = ALT_PPM * 1e-6 * r["L_mm"]
    Cv = {"primary": mem[grp], "A": mem["A"], "B": mem["B"], "raw": raw,
          "up25": mem[grp] + dL, "down25": mem[grp] - dL,
          "mix": r["p_A"] * mem["A"] + (1.0 - r["p_A"]) * mem["B"]}
    if kind == "gnss":
        t_all = C.years(DAYS)
        d_all = (ring_true(tr, t_all) + excess_true(tr, t_all)
                 + ar1(rng, len(DAYS), avail["sigma_mm"], avail["rho"]))
        if avail["var_steps"] > 0:
            d_all = d_all + (rng.normal(0, math.sqrt(avail["var_steps"]))
                             * (t_all >= 0.5 * (y_lp + y_fp)))
        d = pd.Series(d_all, index=DAYS).reindex(avail["t"]).to_numpy()
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
        chk = C.site_check(res["G_mm"], res["excess"], var_end, r["sd_s_mm"], r["L_mm"], Cv,
                           k=K.TOL_SIGMA, var_steps=avail["var_steps"])
    else:
        t = C.years(avail["t"])
        seg = avail["seg"]
        d = ring_true(tr, t) + excess_true(tr, t) + rng.normal(0, K.INSAR_EPOCH_SD_MM, len(t))
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
        chk = C.site_check(res["G_mm"], res["excess"], var_end, r["sd_s_mm"], r["L_mm"], Cv,
                           k=K.TOL_SIGMA, n_bridges=res["n_bridges"],
                           bridge_sd=K.INSAR_BRIDGE_SD_MM)
    key = {"correct": grp, "mix": "mix", "up25": "up25", "down25": "down25", "other": other,
           "raw": "raw"}
    rows = {a: {"evaluable": True, "site": r["site"], "contains_primary": chk[
        f"contains_{key[a]}"], "contains_raw": chk["contains_raw"],
        "C_primary_mm": Cv[key[a]], "C_raw_mm": raw,
        "C_mid_mm": chk["C_mid_mm"]} for a in ALTS}
    return {"ok": True, "grp": grp, "contains_true": chk[f"contains_{grp}"],
            "contains_other": chk[f"contains_{other}"], "contains_raw": chk["contains_raw"],
            "contains_up25": chk["contains_up25"], "contains_down25": chk["contains_down25"],
            "rows": rows, "row": rows["correct"], "G_mm": res["G_mm"],
            "gap_yr": y_fp - y_lp, "Cv": Cv,
            "half_width_mm": 0.5 * (chk["hi_mm"] - chk["lo_mm"]),
            "b_err_ppm": chk["b_mid_ppm"] - B_PPM[grp]}


def availability() -> tuple[dict, dict, dict, dict]:
    """Dates, flags and masks only; every value array is dropped right after reading."""
    rings, _ = IO.rings()
    for r in rings.values():
        r["c"] = None
        r["C"] = None
    for s, r in rings.items():
        r["site"] = s
    xy = IO.site_xy()
    g = IO.gnss()
    prs = IO.pairs(g["stations"], xy)
    gav = {}
    for fam in ("wra_colocated", "ngl_near", "neighbour"):
        for p in prs[fam]:
            d = g["daily"]
            x = d[(d.family == p["family"]) & (d.station == p["station"])
                  & d["flag"].isin(list(K.GNSS_FLAGS))]
            st = g["stations"][(g["stations"].family == p["family"])
                               & (g["stations"].station == p["station"])].iloc[0]
            r = rings[p["site"]]
            sp = g["steps"]
            sp = sp[(sp.family == p["family"]) & (sp.station == p["station"])]
            vs = sum((1000.0 * float(q.se_m)) ** 2 for q in sp.itertuples()
                     if r["t_lp"] < pd.Timestamp(q.date) < r["t_fp"])
            gav[(fam, p["site"], p["station"])] = {
                "t": pd.DatetimeIndex(x["date"]).sort_values(),
                "sigma_mm": 1000.0 * float(st["sigma_hf_m"]), "rho": float(st["rho_hf"]),
                "var_steps": float(vs)}
    g["daily"] = None
    ins = IO.insar()
    iav = {}
    for s in K.SITES:
        x, y = xy[s]
        px = IO.insar_pixel_series(ins, x, y, values=False)
        iav[s] = {"t": ins["dates"], "seg": np.asarray(ins["network_segment"], int),
                  "ok": px["n_valid"] >= K.INSAR_PIX_MIN_VALID,
                  "n_ok_epochs": int((px["n_valid"] >= K.INSAR_PIX_MIN_VALID).sum())}
    ins["value"] = None
    return rings, prs, gav, iav


def _d1(items: list[dict]) -> bool | None:
    """Registered (d1) rule on one realization: Pearson r of the InSAR gap-mean rate with the
    product's (mixture) gap-mean compaction rate >= r_crit and > r with the raw rate."""
    if len(items) < 3:
        return None
    x = [o["G_mm"] / o["gap_yr"] for o in items]
    yb = [o["Cv"]["mix"] / o["gap_yr"] for o in items]
    yr = [o["Cv"]["raw"] / o["gap_yr"] for o in items]
    return bool(C.corr_check(x, yb, yr, K.CORR_ALPHA)["pass"])


def main() -> int:
    rng = np.random.default_rng(K.SEED)
    rng_truth = np.random.default_rng(K.SEED + 1)
    rings, prs, gav, iav = availability()
    out = {"seed": K.SEED, "realizations": K.SYN_REAL, "alt_ppm": ALT_PPM,
           "per_site": [], "families": {},
           "scenarios": {
               "correct": "truth = the product's more probable group; product = that member",
               "mix_prodgrp": "truth = the more probable group; product = the P(A)-weighted "
                              "mixture (the registered primary)",
               "mix_drawn": "truth drawn with the product's P(A); product = the mixture",
               "up25 / down25": "truth = the more probable group; product = that member "
                                "+- 0.25 x 182.2 ppm x L",
               "other": "truth = the more probable group; product = the other member",
               "raw": "truth = the more probable group; product = the raw change"}}
    fam_rows = {}
    for fam in ("wra_colocated", "ngl_near", "neighbour", "insar"):
        items = ([(p["site"], p["station"], gav[(fam, p["site"], p["station"])])
                  for p in prs[fam]] if fam != "insar" else
                 [(s, "069A", iav[s]) for s in K.SITES])
        scen = ("correct", "mix_prodgrp", "mix_drawn", "up25", "down25", "other", "raw")
        per_real = {a: {k: [] for k in range(K.SYN_REAL)} for a in scen}
        d1_items = {a: {k: [] for k in range(K.SYN_REAL)} for a in ("prodgrp", "drawn")}
        for site, sta, av in items:
            r = rings[site]
            prod_grp = "A" if r["p_A"] >= 0.5 else "B"
            res = {"A": [], "B": []}
            n_fail = 0
            for k in range(K.SYN_REAL):
                ok = {}
                for grp in "AB":
                    o = one(rng, "gnss" if fam != "insar" else "insar", r, av, grp)
                    if o is None:
                        n_fail += 1
                        continue
                    res[grp].append(o)
                    ok[grp] = o
                drawn = "A" if rng_truth.uniform() < r["p_A"] else "B"
                if prod_grp in ok:
                    o = ok[prod_grp]
                    for a in ("correct", "up25", "down25", "other", "raw"):
                        per_real[a][k].append(o["rows"][a])
                    per_real["mix_prodgrp"][k].append(o["rows"]["mix"])
                    d1_items["prodgrp"][k].append(o)
                if drawn in ok:
                    per_real["mix_drawn"][k].append(ok[drawn]["rows"]["mix"])
                    d1_items["drawn"][k].append(ok[drawn])
            row = {"family": fam, "site": site, "station": sta, "product_group": prod_grp,
                   "p_A": r["p_A"], "not_evaluable": n_fail}
            for grp in "AB":
                xs = res[grp]
                if xs:
                    row[f"size_{grp}"] = 1 - float(np.mean([x["contains_true"] for x in xs]))
                    row[f"power_{grp}"] = 1 - float(np.mean([x["contains_other"] for x in xs]))
                    row[f"raw_excluded_{grp}"] = 1 - float(np.mean([x["contains_raw"]
                                                                    for x in xs]))
                    row[f"up25_excluded_{grp}"] = 1 - float(np.mean([x["contains_up25"]
                                                                     for x in xs]))
                    row[f"down25_excluded_{grp}"] = 1 - float(np.mean([x["contains_down25"]
                                                                       for x in xs]))
                    row[f"half_width_mm_{grp}"] = float(np.median([x["half_width_mm"]
                                                                   for x in xs]))
                    row[f"b_rmse_ppm_{grp}"] = float(np.sqrt(np.mean([x["b_err_ppm"] ** 2
                                                                      for x in xs])))
            out["per_site"].append(row)
        fr = {}
        for a in scen:
            passes = [C.family_verdict(rows_k, K.FAMILY_MIN_PASS_FRAC,
                                       K.FAMILY_MIN_SITES_GATE)["pass"]
                      for rows_k in per_real[a].values() if rows_k]
            fr[f"p_family_pass_{a}"] = float(np.mean(passes)) if passes else None
            fr["n_realizations"] = len(passes)
        if fam == "insar":
            for a in ("prodgrp", "drawn"):
                v = [_d1(x) for x in d1_items[a].values()]
                v = [x for x in v if x is not None]
                fr[f"p_d1_pass_{a}"] = float(np.mean(v)) if v else None
        fam_rows[fam] = fr
    out["families"] = fam_rows
    out["insar_valid_epochs"] = {s: iav[s]["n_ok_epochs"] for s in K.SITES}
    out["gnss_var_steps_mm2"] = {f"{k[0]}:{k[1]}:{k[2]}": v["var_steps"]
                                 for k, v in gav.items()}
    K.EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump(out, K.EVIDENCE / "synthetic_operating_characteristics_v2.json")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out["per_site"]).round(3).to_string())
    print(json.dumps(fam_rows, indent=1))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
