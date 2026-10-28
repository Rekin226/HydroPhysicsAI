"""Run the X5 design (declaration X5.shutdown_sdid.v1) ONCE on the real data and write
results/dataprep/reports/X5.shutdown_sdid.v1/report.json (refuses to overwrite).

Order of the computation is fixed by the declaration:
  primary   arm T2, R 3 km, buffer 3-6 km, pre 2009-2018, post 2019-2021, block placebos (500);
  verdict   ATT < 0 and one-sided p < 0.10 and lead placebo two-sided p >= 0.10, and the
            synthetic check passed (otherwise "not evaluable");
  reported  lead placebo, per-year ATTs, post 2019-20, radii 2/5 km, period pair 2009-13 vs
            2015-19, Golden Corridor sensitivities (G1 donors without GC towns, G2 census-kWh
            residualised outcome, G3 GC-only arm 北港 post 2020-21, G4 treated without GC towns),
            arm T1 (B02 townships: 2019-21, 2019 only, lead), rings (T1 sites, by band, deep
            minus shallow), secondary long difference vs census kWh (3 km, Yunlin) with the
            alternative pair, a T2 indicator and the annual TWFE lead E(t+1).

``--smoke`` replaces every outcome read by random numbers to exercise the code paths; it
writes nothing and reads no outcome value.

Usage: python run_x5.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402


def clean(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, (list, tuple)):
        return [clean(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.bool_):
        return bool(d)
    return d


class Design:
    def __init__(self, u: pd.DataFrame, S: pd.DataFrame):
        self.u, self.S = u, S

    def mat(self, idx, years):
        return self.S.loc[idx, years].to_numpy(float)

    def run(self, tr, don, pre, post, seed, B=K.N_PLACEBO, S=None):
        S = self.S if S is None else S
        yrs = pre + post
        Yt = S.loc[tr, yrs].to_numpy(float)
        Yd = S.loc[don, yrs].to_numpy(float)
        r = L.sdid_with_placebo(Yt, Yd, self.u.loc[don, "blk"].to_numpy(), len(pre), B, seed)
        r["pre_years"], r["post_years"] = pre, post
        return r


def ring_block(smoke: bool, rng) -> dict:
    if smoke:
        sites = [f"s{i}" for i in range(14)]
        R = pd.DataFrame([{"site": s, "band": b, "year": y, "rate": rng.normal(1, 1)}
                          for s in sites for b in K.RING_BANDS for y in K.RING_YEARS])
        t1 = set(sites[:8])
    else:
        R = L.read_ring_values()
        meta = pd.read_parquet(K.LEVELING_DATA_DIR / "ls_cache/ls-wra-mlcw-obs__meta.parquet",
                               columns=["_station_id", "TownName"])
        town = dict(zip(meta._station_id, meta.TownName, strict=True))
        t1 = {s for s in R.site.unique() if town.get(s) in K.T1_TOWNS}
    out = {"t1_sites": sorted(t1)}
    for b in K.RING_BANDS:
        W = R[R.band == b].pivot(index="site", columns="year", values="rate")
        W = W.reindex(columns=K.RING_YEARS).dropna()
        tr = [s for s in W.index if s in t1]
        dn = [s for s in W.index if s not in t1]
        if len(tr) < 1 or len(dn) < 2:
            out[b] = {"skipped": f"treated {len(tr)} donors {len(dn)}"}
            continue
        r = L.loo_placebo(W.loc[tr].to_numpy(float), W.loc[dn].to_numpy(float),
                          len(K.RING_PRE))
        r.update({"treated_sites": tr, "donor_sites": dn,
                  "min_attainable_p": 1.0 / (1 + len(dn))})
        out[b] = r
    if all("tau" in out.get(b, {}) for b in ("band0", "band2")):
        out["deep_minus_shallow_tau"] = out["band2"]["tau"] - out["band0"]["tau"]
        out["deep_responds_more"] = bool(out["band2"]["tau"] < out["band0"]["tau"])
    return out


def long_difference(u_all: pd.DataFrame, S_all: pd.DataFrame, E_all: pd.DataFrame,
                    Elead: pd.DataFrame) -> dict:
    out = {"energy_unit": "census value_all within 3 km, annualised (GWh/yr, model units)"}

    def ld(early, late, county_yunlin: bool, with_d: bool, seed: int):
        sel = u_all.index[(~u_all.changhua) if county_yunlin else u_all.changhua]
        rows = []
        for s in sel:
            sv, ev = S_all.loc[s], E_all.loc[s]
            ok = sv.notna() & ev.notna()
            e_y = [y for y in range(early[0], early[1] + 1) if y in ok.index and ok[y]]
            l_y = [y for y in range(late[0], late[1] + 1) if y in ok.index and ok[y]]
            if len(e_y) < K.LD_MIN_INTERVALS or len(l_y) < K.LD_MIN_INTERVALS:
                continue
            rows.append({"sid": s, "dr": sv[l_y].mean() - sv[e_y].mean(),
                         "dE": ev[l_y].mean() - ev[e_y].mean(), "E_early": ev[e_y].mean(),
                         "D": float(u_all.at[s, "dW"] <= K.R_TREAT), "blk": u_all.at[s, "blk"]})
        J = pd.DataFrame(rows)
        cols = ["dE", "D"] if with_d else ["dE"]

        def fn(df):
            return L.ols_slope(df.dr.to_numpy(float), df[cols].to_numpy(float))

        b = fn(J)
        bs = L.block_boot(J, fn, K.LD_BOOT, seed)
        res = {"n_benchmarks": len(J), "mean_E_early": float(J.E_early.mean()),
               "mean_dE": float(J.dE.mean()), "mean_dr": float(J.dr.mean())}
        for i, c in enumerate(cols):
            res[f"slope_{c}"] = float(b[i])
            res[f"ci90_{c}"] = [float(np.percentile(bs[:, i], 5)),
                                float(np.percentile(bs[:, i], 95))]
        return res

    out["yunlin_primary_pair_2010_14_vs_2017_21"] = ld(K.LD_EARLY, K.LD_LATE, True, False,
                                                       K.SEED + 50)
    out["yunlin_alt_pair_2009_13_vs_2015_19"] = ld(K.LD_ALT_EARLY, K.LD_ALT_LATE, True, False,
                                                   K.SEED + 51)
    out["yunlin_primary_pair_with_T2_indicator"] = ld(K.LD_EARLY, K.LD_LATE, True, True,
                                                      K.SEED + 52)
    out["changhua_primary_pair"] = ld(K.LD_EARLY, K.LD_LATE, False, False, K.SEED + 53)

    # annual TWFE with the lead E(t+1), Yunlin, units with >= LD_MIN_INTERVALS valid years
    yl = u_all.index[~u_all.changhua]
    S = S_all.loc[yl, K.YEARS]
    E0, E1 = E_all.loc[yl, K.YEARS], Elead.loc[yl, K.YEARS]
    ok = S.notna() & E0.notna() & E1.notna()
    keep = ok.sum(1) >= K.LD_MIN_INTERVALS
    S, E0, E1 = S[keep].where(ok[keep]), E0[keep], E1[keep]
    b = L.twfe(S, {"E0": E0, "Elead": E1})
    rng = np.random.default_rng(K.SEED + 54)
    blk = u_all.loc[S.index, "blk"]
    labs = blk.unique()
    bs = []
    for _ in range(K.LD_BOOT // 2):
        pick = rng.choice(labs, len(labs))
        parts = []
        for j, lb in enumerate(pick):
            ids = blk.index[blk == lb]
            parts.append((ids, [f"{s}#{j}" for s in ids]))
        ids = [i for p in parts for i in p[0]]
        new = [n for p in parts for n in p[1]]

        def sub(df, ids=ids, new=new):
            return pd.DataFrame(df.loc[ids].to_numpy(), index=new, columns=df.columns)

        bs.append(L.twfe(sub(S), {"E0": sub(E0), "Elead": sub(E1)}))
    bs = np.asarray(bs)
    out["yunlin_twfe_lead"] = {
        "n_benchmarks": int(len(S)), "n_obs": int(S.notna().sum().sum()),
        "b_E0": float(b[0]), "ci90_E0": [float(np.percentile(bs[:, 0], 5)),
                                         float(np.percentile(bs[:, 0], 95))],
        "b_Elead": float(b[1]), "ci90_Elead": [float(np.percentile(bs[:, 1], 5)),
                                               float(np.percentile(bs[:, 1], 95))],
        "n_boot": len(bs)}
    p = out["yunlin_primary_pair_2010_14_vs_2017_21"]
    a = out["yunlin_alt_pair_2009_13_vs_2015_19"]
    lead = out["yunlin_twfe_lead"]
    crit = {"primary_slope_positive_ci_excludes_0": p["slope_dE"] > 0 and p["ci90_dE"][0] > 0,
            "alt_pair_slope_positive": a["slope_dE"] > 0,
            "lead_ci_includes_0": lead["ci90_Elead"][0] <= 0 <= lead["ci90_Elead"][1]}
    out["criteria"] = crit
    out["secondary_pass"] = bool(all(crit.values()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    out_path = K.REPORT_DIR / "report.json"
    if not a.smoke:
        if out_path.exists():
            raise SystemExit(f"{out_path} exists; scored once")
        if not (REPO_PREREG := K.REPO / "results/dataprep/preregistration"
                / f"{K.PREREG_ID}.json").exists():
            raise SystemExit(f"{REPO_PREREG} missing: register first")
    syn_path = K.OUT_DIR / "synthetic" / "synthetic_check.json"
    syn = json.loads(syn_path.read_text()) if syn_path.exists() else None
    if not a.smoke and syn is None:
        raise SystemExit("synthetic check missing: it runs first")
    t_start = time.time()
    rng = np.random.default_rng(0)
    u0 = L.unit_table()
    on = u0[u0.ongrid]

    # ------------------------------------------------------------- outcomes (OUTCOME READ)
    if a.smoke:
        S_all = pd.DataFrame(rng.normal(1.5, 1.0, (len(on), len(K.YEARS))), index=on.index,
                             columns=K.YEARS)
    else:
        S_all = L.read_leveling_values(on.index)[K.YEARS]
    S_all = S_all.where(S_all.abs() <= K.GROSS_RATE)  # gross-error intervals -> NaN
    bal = on[on.date_balanced]
    gross_units = bal.index[S_all.loc[bal.index].isna().any(axis=1)]
    u = bal.drop(gross_units)
    S = S_all.loc[u.index]
    D = Design(u, S)
    rep: dict = {"declaration": K.PREREG_ID, "sign": K.SIGN,
                 "label": "design-time evidence (leveling 2009-2021 and rings 2014-2021 were "
                          "seen in design); not a skill test; cannot promote a model",
                 "units": {"ongrid": int(len(on)), "date_balanced": int(len(bal)),
                           "dropped_gross_or_missing": int(len(gross_units)),
                           "usable": int(len(u))},
                 "synthetic_check": {"path": str(syn_path.relative_to(K.REPO)),
                                     "gate_pass": None if syn is None else syn["gate_pass"]}}

    # ------------------------------------------------------------- primary arm T2
    tr, don = L.arm_sets(u, "T2")
    prim = D.run(tr, don, K.PRE, K.POST, K.SEED)
    lead = D.run(tr, don, K.LEAD_PRE, K.LEAD_POST, K.SEED + 1)
    rep["T2"] = {"treated": int(len(tr)), "donors": int(len(don)), "primary": prim,
                 "lead_placebo_2018": lead,
                 "post_2019_2020": D.run(tr, don, K.PRE, K.POST_ALT, K.SEED + 2),
                 "per_year": {str(y): D.run(tr, don, K.PRE, [y], K.SEED + 3 + i)
                              for i, y in enumerate(K.POST)},
                 "period_pair_2009_13_vs_2015_19": D.run(tr, don, K.PAIR_PRE, K.PAIR_POST,
                                                         K.SEED + 7)}
    for j, (nm, (rt, rb)) in enumerate(K.R_ALTS.items()):
        t_, d_ = L.arm_sets(u, "T2", rt, rb)
        rep["T2"][f"radius_{nm}"] = {"treated": int(len(t_)), "donors": int(len(d_)),
                                     **D.run(t_, d_, K.PRE, K.POST, K.SEED + 10 + j)}
    ok = (prim["tau"] < 0 and prim["p_one_sided_less_sinking"] < K.P_SUCCESS
          and lead["p_two_sided"] >= K.P_LEAD_NULL)
    syn_ok = bool(syn and syn["gate_pass"]) if not a.smoke else True
    rep["verdict"] = {
        "criteria": {"att_negative": prim["tau"] < 0,
                     "p_one_sided_lt_0.10": prim["p_one_sided_less_sinking"] < K.P_SUCCESS,
                     "lead_null_p2_ge_0.10": lead["p_two_sided"] >= K.P_LEAD_NULL,
                     "synthetic_gate_pass": syn_ok},
        "result": ("not evaluable: the estimator failed its declared synthetic check"
                   if not syn_ok else ("supported" if ok else "not supported"))}

    # ------------------------------------------------------------- Golden Corridor
    gc = {}
    d_g1 = don[~u.loc[don, "GC"].to_numpy()]
    gc["G1_donors_without_GC_towns"] = {"donors": int(len(d_g1)),
                                        **D.run(tr, d_g1, K.PRE, K.POST, K.SEED + 20)}
    t0, t1 = L.interval_bounds(u.index)
    En = L.Energy(u, K.LD_RADIUS)
    E = En.table(t0[K.YEARS], t1[K.YEARS])
    if a.smoke:
        E = pd.DataFrame(rng.gamma(2, 1, E.shape), index=E.index, columns=E.columns)
    beta = float(L.twfe(S.loc[don], {"E": E.loc[don]})[0])
    S_res = S - beta * E
    gc["G2_census_kwh_residualised"] = {"beta_cm_per_GWh": beta,
                                        **D.run(tr, don, K.PRE, K.POST, K.SEED + 21, S=S_res)}
    t3, d3 = L.arm_sets(u, "GC3")
    gc["G3_GC_only_arm_beigang_post_2020_21"] = {
        "treated": int(len(t3)), "donors": int(len(d3)),
        **D.run(t3, d3, K.PRE + [2019], [2020, 2021], K.SEED + 22)}
    t_g4 = tr[~u.loc[tr, "GC"].to_numpy()]
    gc["G4_treated_without_GC_towns"] = {"treated": int(len(t_g4)),
                                         **D.run(t_g4, don, K.PRE, K.POST, K.SEED + 23)}
    rep["golden_corridor"] = gc

    # ------------------------------------------------------------- arm T1 (B02 townships)
    t1_, d1_ = L.arm_sets(u, "T1")
    rep["T1"] = {"treated": int(len(t1_)), "donors": int(len(d1_)),
                 "post_2019_2021": D.run(t1_, d1_, K.PRE, K.POST, K.SEED + 30),
                 "post_2019_only_pre_GC_extension": D.run(t1_, d1_, K.PRE, [2019], K.SEED + 31),
                 "lead_placebo_2018": D.run(t1_, d1_, K.LEAD_PRE, K.LEAD_POST, K.SEED + 32),
                 "post_2020_2021": D.run(t1_, d1_, K.PRE + [2019], [2020, 2021], K.SEED + 33)}

    # ------------------------------------------------------------- rings (reported)
    rep["rings_T1"] = ring_block(a.smoke, rng)

    # ------------------------------------------------------------- secondary long difference
    t0a, t1a = L.interval_bounds(on.index)
    Ea = L.Energy(on, K.LD_RADIUS)
    E_all = Ea.table(t0a[K.YEARS], t1a[K.YEARS])
    E_lead = Ea.table(t0a[K.YEARS], t1a[K.YEARS], shift_years=1)
    if a.smoke:
        E_all = pd.DataFrame(rng.gamma(2, 1, E_all.shape), index=E_all.index,
                             columns=E_all.columns)
        E_lead = E_all.shift(-1, axis=1)
    rep["secondary_long_difference"] = long_difference(on, S_all, E_all, E_lead)
    rep["energy_unit_note"] = En.unit
    rep["seconds"] = round(time.time() - t_start, 1)
    rep = clean(rep)
    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    print(txt[:6000])
    if a.smoke:
        print("smoke run: nothing written")
        return 0
    K.OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = S.copy()
    panel.columns = [f"S_{c}" for c in panel.columns]
    panel = panel.join(u[["town", "dW", "T1", "GC", "blk"]])
    panel["arm_T2"] = np.where(panel.index.isin(tr), "treated",
                               np.where(panel.index.isin(don), "donor", "excluded"))
    panel.to_parquet(K.OUT_DIR / "leveling_rate_panel.parquet")
    K.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
