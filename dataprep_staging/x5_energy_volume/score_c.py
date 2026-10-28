"""Score ``X5.energy_volume.c_vs_a.v1`` once: estimator (c) against (a) on the fresh unit.

Steps (each a separate process):

* ``check`` (under the registry lock): ``prereg3.assert_can_score`` (logs the scoring look)
  and ``record_look`` for the two reported, non-gating units (unconfined 2010-2011 wells;
  confined AMP_V2 wells 2012-2019, design-time); writes work/score_check.json.
* ``run``: computes the registered primary statistic and every declared reported variant,
  writes results/dataprep/reports/X5.energy_volume.c_vs_a.v1/report.json once (refuses to
  overwrite) and the per-well table.

Primary: ratio_w = c_w / a(county) with a = WRA Jan-May 2011-22 irrigation groundwater per
kept irrigation kWh (step0_ab report); statistic = median over confined test wells with an
estimate; PASS if 1/1.5 <= median <= 1.5; NOT INFORMATIVE if fewer than 5 wells have one.

Usage: python -I score_c.py check|run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402

from x5_energy_volume import amp_inversion as A  # noqa: E402
from x5_energy_volume import x5const as K  # noqa: E402

CHECK = K.WORK / "score_check.json"
STEP0 = K.REPORT_DIR["step0_ab"] / "report.json"
OUTD = K.OUT / "score"
U_UNIT = "X5EV:unconfined_ls_cache_2010_2011_reported"
D_UNIT = "X5EV:confined_amp_2012_2019_designtime"
DT_WINDOW = ("2012-01-01", "2020-01-01")


def prereg_doc() -> dict:
    p = K.DATAPREP / "preregistration" / f"{K.PREREG['test']}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def check() -> int:
    from harness import prereg3
    doc = prereg_doc()
    r = prereg3.assert_can_score(K.PREREG["test"], K.WP, test_id=K.TEST_ID)
    rep = doc["reported_units"]
    for uid, scope, purpose in ((U_UNIT, K.TEST_SCOPE, "reported_after_score"),
                                (D_UNIT, "heads:2012-01..2019-12", "design_time_after_score")):
        prereg3.record_look([uid], scope, K.WP, purpose, members={uid: rep[uid]},
                            prereg_id=K.PREREG["test"],
                            note=f"[X5EV] {purpose}: {rep[uid]['description'][:160]}")
    K.WORK.mkdir(parents=True, exist_ok=True)
    CHECK.write_text(json.dumps({"sha256": r["sha256"], "tests": r["tests"],
                                 "post_registration_looks": r["post_registration_looks"],
                                 "unverifiable_prior_looks": r["unverifiable_prior_looks"]},
                                indent=1, default=str) + "\n", encoding="utf-8")
    print("can score:", r["tests"], "sha", r["sha256"][:12])
    return 0


def a_ref() -> dict:
    s = json.loads(STEP0.read_text(encoding="utf-8"))["counties"]
    return {cn: {"kept": s[cn]["a_m3_per_kwh"]["kept"]["janmay_2011_22"],
                 "dedup": s[cn]["a_m3_per_kwh"]["dedup"]["janmay_2011_22"]} for cn in K.COUNTIES}


def estimator(version: str):
    """(geometry, G, well_estimate) of the registered estimator version."""
    if version == "v2":
        from x5_energy_volume import amp_inversion_v2 as A2
        return A2.well_geometry, A2.g_series, A2.well_estimate
    return A.well_geometry, A.g_series, A.well_estimate


def estimate_wells(st: pd.DataFrame, P: dict, Pd: dict | None, bet: pd.Series, source: str,
                   window: tuple[str, str], variants: bool, s_override: float | None = None,
                   version: str = "v1") -> pd.DataFrame:
    geom, gser, west = estimator(version)
    months = pd.date_range(window[0], window[1], freq="MS", inclusive="left")
    rows = []
    for w in st.itertuples(index=False):
        w = pd.Series(w._asdict())
        row = {"sid": w["sid"], "layer": w["layer"], "attr": w["attr"]}
        row["county"] = A.county_of_point(w["x"], w["y"], P["x"], P["y"], P["county"])
        geo = geom(w, P)
        row.update({"n_pumps_2km": geo["n_2km"], "n_pumps_4km": geo["n_4km"]})
        if geo["n_2km"] < K.MIN_PUMPS:
            row["excluded"] = "fewer than 20 kept irrigation pumps within 2 km"
            rows.append(row)
            continue
        S, src = A.storage(w["sid"], bet)
        if s_override is not None:
            S, src = s_override, "declared Sy (reported only)"
        row.update({"S": S, "S_source": src})
        try:
            s = A.read_series(w["sid"], source)
        except (FileNotFoundError, OSError) as e:
            row["excluded"] = f"no series: {e.__class__.__name__}"
            rows.append(row)
            continue
        x, valid, idx = A.qc_hourly(s, *window)
        G = gser(geo, months, K.T_PRIMARY, S)
        r = west(x, valid, idx, G)
        row.update({k: r[k] for k in r})
        if np.isnan(r["c"]):
            row["excluded"] = f"{r['n_months']} valid Jan-May months < {K.MIN_JM_MONTHS}"
        if variants and np.isfinite(r["c"]):
            for t in K.T_LIT:
                row[f"c_T{int(t)}"] = west(x, valid, idx, gser(geo, months, t, S))["c"]
            for dh in K.DUTY_RANGE_H:
                row[f"c_duty{int(dh)}h"] = west(
                    x, valid, idx, gser(geo, months, K.T_PRIMARY, S, dh))["c"]
            other = "v1" if version == "v2" else "v2"
            og, ogs, ow = estimator(other)
            if other == "v2":
                from x5_energy_volume import amp_inversion_v2 as A2
                gq = og(w, A2.load_pumps())
            else:
                gq = og(w, A.load_pumps())
            row[f"c_{other}"] = ow(x, valid, idx, ogs(gq, months, K.T_PRIMARY, S))["c"]
            row["c_lsq"] = west(x, valid, idx, G, extractor="lsq")["c"]
            row["c_allmonths"] = west(x, valid, idx, G, months_sel=None)["c"]
            row["c_S_x0.1"] = west(x, valid, idx, gser(geo, months, K.T_PRIMARY, S * 0.1))["c"]
            row["c_S_x10"] = west(x, valid, idx, gser(geo, months, K.T_PRIMARY, S * 10))["c"]
            if Pd is not None:
                gd = geom(w, Pd)
                row["c_dedup"] = west(x, valid, idx, gser(gd, months, K.T_PRIMARY, S))["c"]
            # per-pump on-rate implied by c_w on pumping hours (median over the pumps)
            mi = pd.Index(A.MONTHS).get_indexer(months[np.isin(months.month, K.JANMAY)])
            e = geo["E"][:, mi].sum(axis=1) / max(len(mi), 1)
            e = e[e > 0]
            days = 30.4
            row["pump_on_rate_m3h_median"] = float(np.median(
                r["c"] * e / (days * K.DUTY_H))) if len(e) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, a: dict, col: str = "c", basis: str = "kept") -> dict:
    d = df[df[col].notna()].copy() if col in df else df.iloc[:0]
    if not len(d):
        return {"n": 0}
    d["ratio"] = d[col] / d["county"].map({k: v[basis] for k, v in a.items()})
    out = {"n": int(len(d)), "median_ratio": float(d["ratio"].median()),
           "median_c": float(d[col].median())}
    for cn, g in d.groupby("county"):
        out[cn] = {"n": int(len(g)), "median_ratio": float(g["ratio"].median()),
                   "median_c": float(g[col].median())}
    return out


def boot_ci(r: np.ndarray, seed: int, n: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    b = [np.median(rng.choice(r, len(r), replace=True)) for _ in range(n)]
    return [float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]


def run() -> int:
    if not CHECK.exists():
        raise SystemExit("run 'check' under the lock first")
    rep_path = K.REPORT_DIR["test"] / "report.json"
    if rep_path.exists():
        raise SystemExit(f"{rep_path} exists; scored once")
    doc = prereg_doc()
    a = a_ref()
    ver = doc.get("estimator_version", "v1")
    if ver == "v2":
        from x5_energy_volume import amp_inversion_v2 as A2
        P = A2.load_pumps()
    else:
        P = A.load_pumps()
    from x5_energy_volume import pump_cache
    if not pump_cache.DEDUP_CACHE.exists():
        pump_cache.build("dedup")
    zd = np.load(pump_cache.DEDUP_CACHE, allow_pickle=False)
    Pd = {k: zd[k] for k in zd.files}
    bet = A.be_table()
    st = A.stations().set_index("sid")
    test_ids = doc["tests"][0]["external_units"][doc["tests"][0]["fresh_units"][0]]["ids"]
    # ------------------------------------------------ primary (fresh unit)
    prim = estimate_wells(st.loc[test_ids].reset_index(), P, Pd, bet, "ls_cache",
                          K.TEST_WINDOW, variants=True, version=ver)
    s = summarise(prim, a)
    n = s["n"]
    if n < 5:
        verdict = "NOT INFORMATIVE (fewer than 5 wells with an estimate)"
        passed = None
    else:
        passed = bool(1 / K.RATIO_BAND <= s["median_ratio"] <= K.RATIO_BAND)
        verdict = "PASS" if passed else "FAIL"
    d = prim[prim["c"].notna()]
    ratios = (d["c"] / d["county"].map({k: v["kept"] for k, v in a.items()})).to_numpy()
    res = {"prereg": K.PREREG["test"], "test_id": K.TEST_ID, "estimator_version": ver,
           "prereg_sha256": json.loads(CHECK.read_text())["sha256"],
           "reference_a_kept_janmay_2011_22": {k: v["kept"] for k, v in a.items()},
           "primary": {**s, "verdict": verdict, "pass": passed,
                       "band": [1 / K.RATIO_BAND, K.RATIO_BAND],
                       "ci90_median_ratio_well_bootstrap": boot_ci(ratios, K.SEED_BOOT)
                       if len(ratios) >= 3 else None}}
    rv = {}
    for col in ("c_T58", "c_T6034", "c_duty4h", "c_duty12h", "c_lsq", "c_allmonths",
                "c_S_x0.1", "c_S_x10", "c_v1", "c_v2"):
        rv[col] = summarise(prim, a, col)
    rv["c_dedup_vs_a_dedup"] = summarise(prim, a, "c_dedup", basis="dedup")
    rv["c_kept_vs_a_dedup"] = summarise(prim, a, "c", basis="dedup")
    res["reported_variants_not_gating"] = rv
    res["pump_on_rate_m3h_median_over_wells"] = float(np.nanmedian(
        prim.get("pump_on_rate_m3h_median", pd.Series(dtype=float))))
    # ------------------------------------------------ reported: unconfined 2010-2011
    rep_u = doc["reported_units"][U_UNIT]["ids"]
    uu = estimate_wells(st.loc[rep_u].reset_index(), P, None, bet, "ls_cache", K.TEST_WINDOW,
                        variants=False, version=ver)
    uy = estimate_wells(st.loc[rep_u].reset_index(), P, None, bet, "ls_cache", K.TEST_WINDOW,
                        variants=False, s_override=K.SY_UNCONFINED, version=ver)
    res["reported_unconfined_2010_2011"] = {"elastic_S": summarise(uu, a),
                                            "Sy_0.1": summarise(uy, a),
                                            "label": "reported only; never gating"}
    # ------------------------------------------------ design-time: confined AMP 2012-2019
    rep_d = doc["reported_units"][D_UNIT]["ids"]
    dd = estimate_wells(st.loc[rep_d].reset_index(), P, None, bet, "amp", DT_WINDOW,
                        variants=False, version=ver)
    res["designtime_confined_amp_2012_2019"] = {
        **summarise(dd, a), "label": "design-time (these wells sat in earlier fresh units "
                                     "for 2012-2022); never gating"}
    OUTD.mkdir(parents=True, exist_ok=True)
    prim.to_csv(OUTD / "wells_primary_2010_2011.csv", index=False)
    uu.assign(S_variant="elastic").pipe(lambda x: pd.concat(
        [x, uy.assign(S_variant="Sy0.1")])).to_csv(OUTD / "wells_unconfined_2010_2011.csv",
                                                   index=False)
    dd.to_csv(OUTD / "wells_confined_amp_2012_2019.csv", index=False)
    res["outputs"] = [rel_to_repo(OUTD / f) for f in (
        "wells_primary_2010_2011.csv", "wells_unconfined_2010_2011.csv",
        "wells_confined_amp_2012_2019.csv")]
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("check", "run"))
    raise SystemExit(check() if ap.parse_args().step == "check" else run())
