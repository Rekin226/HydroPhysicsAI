"""P2d validation (plan WP P2d): freeze the census at mid-2013 and predict the 2014-16 cohorts
per township with the layer's own rule; compare counts and kWh.

Rule applied at the freeze F = 2013-07 (only pumps installed by 2013-06 and readings to 2013-06
are known to the rule):

* rate per township x class = kept pumps installed in the 6 calendar years before the freeze
  year (2007-2012) / 6 (the real layer uses 2010-2015 for a 2016-07 freeze; the plan's 2013
  surge exclusion has no counterpart in 2007-2012, so the back-test's low and high forms
  coincide);
* placement in proportion to the township's 2007-2012 installs per 1 km cell (cell -1 = outside
  the active grid);
* relative age ramp g(c, a), a = 1..6, from the 2007-2012 cohorts with readings to 2013-06
  (ages without a complete window hold the last measured age; classes with < 200 cohort pumps
  use the pooled ramp);
* energy of a synthetic cohort of n pumps of age a in month t = n * g(c, a) * P_F(c, t), where
  P_F(c, t) is the mean energy per pump of class c in month t over the pumps the frozen census
  holds (installed by 2013-06, or without a valid install month);
* monthly cohorts of rate/12 pumps installed 2014-01..2016-06 (the cohorts the census holds:
  its record of installs stops at mid-2016).

Truth: kept pumps with a valid install month in 2014-01..2016-06 (counts per township; their
recorded-arm kWh 2014-01..2022-11, in grid and county-wide). Metrics (plan): median over
townships with >= 10 actual installs of |pred - act| / act <= 0.30 and fan-total (in-grid)
kWh |pred / act - 1| <= 0.20 over 2014-01..2022-11; Yunlin and Changhua gate; Chiayi (in-grid
pumps only) is reported.

Usage: python -I d2_backtest.py predict-only|score <out_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

PREREG_ID = "P2def.v1"
TEST = {"yunlin": "P2d.bt.yunlin", "changhua": "P2d.bt.changhua", "chiayi": "P2d.bt.chiayi"}
USE = "prior"
COUNT_MAX = 0.30
KWH_MAX = 0.20


def predict(k: str, l1: pd.DataFrame) -> dict:
    p = D.Pumps("preorigin", k, USE, l1=l1)
    E = p.energy()
    T = p.T
    F = D.tidx(D.BT_FREEZE)
    existing = (~p.inst_valid) | (p.inst < F)
    P = D.per_pump_index(E, p.cls, p.inst, existing, T)
    iy = D.install_year(p.inst)
    coh = p.inst_valid & np.isin(iy, D.BT_RATE_YEARS)
    R = D.ramp(E, p.cls, p.inst, coh, P, data_end=F - 1)
    rate = D.rates_by_town_class(p.town, p.cls, p.inst, D.BT_RATE_YEARS)
    w = D.cell_weights(p.town, p.cell, p.inst, D.BT_RATE_YEARS)
    c0, c1 = D.tidx(D.BT_COHORT[0]), D.tidx(D.BT_COHORT[1])
    t1 = D.tidx(D.BT_KWH_END)
    S = D.synth_monthly(rate, w, R["g"], P, c0, c1, c0, t1)
    n_years = (c1 - c0 + 1) / 12.0
    pred_town = rate.groupby("town")["rate_per_yr"].sum() * n_years
    pred_ingrid = sum(S[c]["cells"].sum(0) for c in D.CLASSES)
    pred_county = sum(S[c]["county"] for c in D.CLASSES)
    return {"pumps": p, "E": E, "P": P, "R": R, "rate": rate, "w": w, "S": S,
            "pred_town": pred_town, "pred_ingrid": pred_ingrid, "pred_county": pred_county,
            "c0": c0, "c1": c1, "t1": t1, "n_existing": int(existing.sum()),
            "n_cohort_ramp": int(coh.sum())}


def actual(r: dict) -> dict:
    p, E = r["pumps"], r["E"]
    act = p.inst_valid & (p.inst >= r["c0"]) & (p.inst <= r["c1"])
    c0, t1 = r["c0"], r["t1"]
    act_town = pd.Series(p.town[act]).value_counts()
    act_ingrid = E[act & (p.cell >= 0), c0:t1 + 1].sum(0)
    act_county = E[act, c0:t1 + 1].sum(0)
    act_cls = pd.Series(p.cls[act]).value_counts()
    return {"mask": act, "town": act_town, "ingrid": act_ingrid, "county": act_county,
            "cls": act_cls}


def score(k: str, r: dict, a: dict) -> dict:
    towns = sorted(set(r["pred_town"].index) | set(a["town"].index))
    tab = pd.DataFrame({"town": towns})
    tab["pred"] = r["pred_town"].reindex(towns).fillna(0.0).to_numpy()
    tab["act"] = a["town"].reindex(towns).fillna(0).astype(int).to_numpy()
    tab["rel_err"] = np.where(tab["act"] > 0, np.abs(tab["pred"] - tab["act"]) /
                              tab["act"].clip(lower=1), np.nan)
    tab["signed_rel"] = np.where(tab["act"] > 0, tab["pred"] / tab["act"].clip(lower=1) - 1,
                                 np.nan)
    scored = tab[tab["act"] >= D.MIN_TOWN_ACTUAL]
    med = float(scored["rel_err"].median()) if len(scored) else float("nan")
    pi, ai = float(r["pred_ingrid"].sum()), float(a["ingrid"].sum())
    pc, ac = float(r["pred_county"].sum()), float(a["county"].sum())
    kwh_err = pi / ai - 1 if ai > 0 else float("nan")
    dates = D.DATES[r["c0"]: r["t1"] + 1]
    yr = pd.Series(r["pred_ingrid"], index=dates).groupby(dates.year).sum()
    ya = pd.Series(a["ingrid"], index=dates).groupby(dates.year).sum()
    annual = {int(y): {"pred_GWh": yr[y] / 1e6, "act_GWh": ya[y] / 1e6,
                       "rel_err": (yr[y] / ya[y] - 1) if ya[y] > 0 else None} for y in yr.index}
    gating = k in ("yunlin", "changhua")
    ok_count = bool(np.isfinite(med) and med <= COUNT_MAX)
    ok_kwh = bool(np.isfinite(kwh_err) and abs(kwh_err) <= KWH_MAX)
    return {
        "county": k, "test_id": TEST[k], "gating": gating,
        "median_township_count_rel_err": med, "n_towns_scored": int(len(scored)),
        "n_towns_total": int(len(tab)), "towns_unscored_act_lt_10": int((tab["act"] <
                                                                         D.MIN_TOWN_ACTUAL).sum()),
        "pred_count_total": float(tab["pred"].sum()), "act_count_total": int(tab["act"].sum()),
        "count_total_rel_err": float(tab["pred"].sum() / max(tab["act"].sum(), 1) - 1),
        "fan_kwh_pred_GWh": pi / 1e6, "fan_kwh_act_GWh": ai / 1e6, "fan_kwh_rel_err": kwh_err,
        "county_kwh_pred_GWh": pc / 1e6, "county_kwh_act_GWh": ac / 1e6,
        "county_kwh_rel_err": pc / ac - 1 if ac > 0 else None,
        "annual_ingrid": annual,
        "act_by_class": a["cls"].to_dict(),
        "pred_by_class": r["rate"].groupby("cls")["rate_per_yr"].sum().mul(
            (r["c1"] - r["c0"] + 1) / 12.0).to_dict(),
        "ramp_g": {c: r["R"]["g"][c].tolist() for c in D.CLASSES},
        "ramp_source": r["R"]["source"], "ramp_n_pumps": r["R"]["n_pumps"],
        "n_existing_pumps": r["n_existing"], "n_ramp_cohort_pumps": r["n_cohort_ramp"],
        "pass_count": ok_count, "pass_kwh": ok_kwh, "pass": bool(ok_count and ok_kwh),
        "thresholds": {"median_township_count_rel_err_max": COUNT_MAX,
                       "fan_kwh_abs_rel_err_max": KWH_MAX,
                       "min_actual_installs_per_scored_township": D.MIN_TOWN_ACTUAL},
        "towns": tab.round(6).to_dict(orient="records"),
    }


def main(mode: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    l1 = D.l1_table(use=USE)
    if mode == "score":
        from harness import prereg3
        auth = {}
        for k in D.COUNTIES:
            got = prereg3.assert_can_score(PREREG_ID, D.WP, test_id=TEST[k])
            print("assert_can_score ok:", got["sha256"][:12], got["tests"])
            auth[TEST[k]] = {"prereg_sha256": got["sha256"],
                             "post_registration_looks": got["post_registration_looks"],
                             "unverifiable_prior_looks": len(got["unverifiable_prior_looks"])}
    res = {"mode": mode, "code": D.p2def_code()}
    for k in D.COUNTIES:
        r = predict(k, l1)
        chk = {c: bool(np.isfinite(r["S"][c]["county"]).all()) for c in D.CLASSES}
        print(k, "predicted series finite:", all(chk.values()), "existing pumps",
              r["n_existing"], "ramp cohort pumps", r["n_cohort_ramp"],
              f"{time.time() - t_start:.0f}s")
        if mode == "score":
            a = actual(r)
            s = score(k, r, a)
            res[k] = s
            print(k, {x: s[x] for x in ("median_township_count_rel_err", "n_towns_scored",
                                        "count_total_rel_err", "fan_kwh_rel_err",
                                        "county_kwh_rel_err", "pass")})
        del r
    if mode == "score":
        res["authorisation"] = auth
        res["gate_P2d_backtest"] = bool(res["yunlin"]["pass"] and res["changhua"]["pass"])
        D.save_json(res, out / "P2d_backtest.json")
        print("gate:", res["gate_P2d_backtest"])
    else:
        D.save_json({"mode": mode, "ok": True, "seconds": time.time() - t_start},
                    out / "P2d_backtest_predict_only.json")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
