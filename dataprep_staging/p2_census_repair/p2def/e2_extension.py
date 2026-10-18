"""P2e (plan WP P2e, P1 step 6): extension of the cleaned census past its last complete month
(2026-06) to 2026-07..2026-08 from TPC d007012 farm-sector (農林漁牧) sales, for the
recorded-forcing diagnostic arm only (U4).

Census: C0 pump_electricity/full/v3 county_class_monthly, scope 'county', class 'all' (kept,
cleaned, recorded arm), complete months 2016-01..2026-06. TPC: L0 d007012 (2026-10-12 fetch),
months 2016-01..2026-08.

Candidates (per county; Yunlin is extended first, plan P1 step 6):

* ``flat``    r(moy) = mean over the fit window of census / TPC in that calendar month
              (P1 v1's estimator; fit window 2016-01..origin);
* ``recent36`` the same mean over the 36 months ending at the origin;
* ``trend``   log(census / TPC) = calendar-month effect + linear time trend, OLS over the fit
              window, extrapolated;
* ``scaled`` (Changhua only, plan P1 step 6 "Changhua = a x extended Yunlin") a x Yunlin's
              selected candidate, a = sum of the Changhua census / sum of the Yunlin census
              over the 36 months ending at the origin.

Backtests (rolling origin, fit on months <= origin only):

* x2 (selection, sd): origins 2019-06..2026-04 (monthly); two-month-ahead total error
  e = log(sum pred / sum census) over months origin+1, origin+2. Selection metric: RMS of e
  over the last 36 origins (2023-05..2026-04); the candidate with the lowest value is the
  county's extension. Monthly sd (uncalibrated) = RMS of the monthly log errors at lead 1-2 of
  the selected candidate over the same origins.
* x1 (acceptance, the plan's P1 extension rule): one-year-ahead annual error of the selected
  candidate, origins every June 2019..2025 (predict Jul..Jun); RMS of the relative annual
  error <= 10 %.

Usage: python -I e2_extension.py score <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

PREREG_ID = "P2def.v1"
USE = "forcing_recorded_arm"
FIT0 = "2016-01"
LAST_CENSUS = "2026-06"
EXT_MONTHS = ("2026-07", "2026-08")
X2_ORIGINS = pd.period_range("2019-06", "2026-04", freq="M")
X2_SELECT_LAST = 36
X1_ORIGIN_YEARS = tuple(range(2019, 2026))          # origins at June of these years
X1_MAX = 0.10
CANDIDATES = {"yunlin": ("flat", "recent36", "trend"),
              "changhua": ("flat", "recent36", "trend", "scaled")}


def census_monthly(prod) -> dict:
    df = prod.read_table("county_class_monthly.parquet",
                         columns=["county", "scope", "class", "date", "value", "flag"])
    out = {}
    for k in ("yunlin", "changhua", "chiayi"):
        s = df[(df["county"] == k) & (df["scope"] == "county") & (df["class"] == "all")]
        out[k] = s.set_index("date")["value"].sort_index()
    return out


def _months(a: str, b: str) -> list[str]:
    return pd.period_range(a, b, freq="M").strftime("%Y-%m").tolist()


def fit_predict(cand: str, Y: pd.Series, X: pd.Series, origin: str, targets: list[str],
                yl_pred: dict | None = None, Y_yl: pd.Series | None = None) -> pd.Series:
    """Prediction of census months ``targets`` from data <= origin (census) and the TPC
    series (TPC is observed in the target months: that is the extension's premise)."""
    fit = [m for m in _months(FIT0, origin) if m in Y.index and m in X.index and X[m] > 0]
    if cand == "scaled":
        last = [m for m in fit if m in Y_yl.index][-36:]
        a = Y.reindex(last).sum() / Y_yl.reindex(last).sum()
        return pd.Series({t: a * yl_pred[t] for t in targets})
    ratio = (Y.reindex(fit) / X.reindex(fit)).dropna()
    moy = pd.Index([int(m[5:]) for m in ratio.index])
    if cand in ("flat", "recent36"):
        r = ratio if cand == "flat" else ratio.iloc[-36:]
        mm = pd.Index([int(m[5:]) for m in r.index])
        rm = r.groupby(mm).mean()
        return pd.Series({t: X[t] * rm[int(t[5:])] for t in targets})
    if cand == "trend":
        lr = np.log(ratio.to_numpy())
        tt = np.array([(int(m[:4]) - 2016) + (int(m[5:]) - 1) / 12.0 for m in ratio.index])
        Z = np.column_stack([np.eye(12)[moy.to_numpy() - 1], tt])
        b, *_ = np.linalg.lstsq(Z, lr, rcond=None)
        out = {}
        for t in targets:
            z = np.concatenate([np.eye(12)[int(t[5:]) - 1],
                                [(int(t[:4]) - 2016) + (int(t[5:]) - 1) / 12.0]])
            out[t] = X[t] * float(np.exp(z @ b))
        return pd.Series(out)
    raise ValueError(cand)


def next_months(origin: str, n: int) -> list[str]:
    p = pd.Period(origin, freq="M")
    return [(p + i).strftime("%Y-%m") for i in range(1, n + 1)]


def backtests(Yc: dict, Xc: dict) -> dict:
    res = {}
    sel = {}
    preds_yl = {}       # Yunlin selected-candidate predictions per origin (for 'scaled')
    for k in ("yunlin", "changhua"):
        Y, X = Yc[k], Xc[k]
        x2 = {}
        for cand in CANDIDATES[k]:
            rows = []
            for o in X2_ORIGINS.strftime("%Y-%m"):
                tg = next_months(o, 2)
                if any(t not in Y.index for t in tg) or tg[-1] > LAST_CENSUS:
                    continue
                kw = {}
                if cand == "scaled":
                    kw = {"yl_pred": preds_yl[(o, 2)], "Y_yl": Yc["yunlin"]}
                p = fit_predict(cand, Y, X, o, tg, **kw)
                a = Y.reindex(tg)
                rows.append({"origin": o, "e2": float(np.log(p.sum() / a.sum())),
                             "e_m1": float(np.log(p.iloc[0] / a.iloc[0])),
                             "e_m2": float(np.log(p.iloc[1] / a.iloc[1]))})
            df = pd.DataFrame(rows)
            last = df.iloc[-X2_SELECT_LAST:]
            x2[cand] = {"rms_e2_last36": float(np.sqrt((last["e2"] ** 2).mean())),
                        "rms_e2_all": float(np.sqrt((df["e2"] ** 2).mean())),
                        "mean_e2_last36": float(last["e2"].mean()),
                        "rms_monthly_last36": float(np.sqrt(np.mean(np.concatenate(
                            [last["e_m1"].to_numpy(), last["e_m2"].to_numpy()]) ** 2))),
                        "n_origins": int(len(df)), "first_origin_last36": last["origin"].iloc[0],
                        "rows": df.to_dict(orient="records")}
        best = min(x2, key=lambda c: x2[c]["rms_e2_last36"])
        sel[k] = best
        if k == "yunlin":
            # store the selected Yunlin candidate's predictions for every origin and horizon
            for o in X2_ORIGINS.strftime("%Y-%m"):
                for h, tg in ((2, next_months(o, 2)), (12, next_months(o, 12))):
                    if all(t in X.index for t in tg):
                        preds_yl[(o, h)] = fit_predict(best, Y, X, o, tg)
            for y in X1_ORIGIN_YEARS:
                o = f"{y}-06"
                tg = next_months(o, 12)
                preds_yl[(o, 12)] = fit_predict(best, Y, X, o, tg)
            for h, tg in (("ext", list(EXT_MONTHS)),):
                preds_yl[(LAST_CENSUS, h)] = fit_predict(best, Y, X, LAST_CENSUS, tg)
        # x1: annual one-year-ahead for the selected candidate (and reported for the others)
        x1 = {}
        for cand in CANDIDATES[k]:
            rows = []
            for y in X1_ORIGIN_YEARS:
                o = f"{y}-06"
                tg = next_months(o, 12)
                if any(t not in Y.index for t in tg):
                    continue
                kw = {}
                if cand == "scaled":
                    kw = {"yl_pred": preds_yl[(o, 12)], "Y_yl": Yc["yunlin"]}
                p = fit_predict(cand, Y, X, o, tg, **kw)
                rows.append({"origin": o, "pred_GWh": p.sum() / 1e6,
                             "census_GWh": Y.reindex(tg).sum() / 1e6,
                             "rel_err": float(p.sum() / Y.reindex(tg).sum() - 1)})
            df = pd.DataFrame(rows)
            x1[cand] = {"rms_rel": float(np.sqrt((df["rel_err"] ** 2).mean())),
                        "mean_rel": float(df["rel_err"].mean()),
                        "rows": df.to_dict(orient="records")}
        res[k] = {"x2": x2, "x1": x1, "selected": best,
                  "x1_selected_rms": x1[best]["rms_rel"],
                  "x1_pass": bool(x1[best]["rms_rel"] <= X1_MAX),
                  "sd_monthly_log": x2[best]["rms_monthly_last36"],
                  "sd_two_month_log": x2[best]["rms_e2_last36"]}
    res["selected"] = sel
    res["_preds_yl"] = preds_yl
    return res


def extend(Yc: dict, Xc: dict, bt: dict) -> pd.DataFrame:
    rows = []
    preds_yl = bt["_preds_yl"]
    for k in ("yunlin", "changhua"):
        best = bt[k]["selected"]
        kw = {}
        if best == "scaled":
            kw = {"yl_pred": preds_yl[(LAST_CENSUS, "ext")], "Y_yl": Yc["yunlin"]}
        p = fit_predict(best, Yc[k], Xc[k], LAST_CENSUS, list(EXT_MONTHS), **kw)
        for t in EXT_MONTHS:
            rows.append({"county": k, "month": t, "candidate": best,
                         "tpc_farm_kwh": float(Xc[k][t]), "census_hat_kwh": float(p[t]),
                         "sd_log": bt[k]["sd_monthly_log"]})
    return pd.DataFrame(rows)


def main(mode: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if mode != "score":
        raise SystemExit("mode must be 'score'")
    from harness import prereg3
    got = {}
    for k in ("yunlin", "changhua"):
        for tid in (f"P2e.x2.select_{k}", f"P2e.x1.annual_{k}"):
            got[tid] = prereg3.assert_can_score(PREREG_ID, D.WP, test_id=tid)
    prod = D.open_c0("full", USE)
    Yc = census_monthly(prod)
    Xc = {k: D.tpc_farm(k) for k in ("yunlin", "changhua")}
    for k in ("yunlin", "changhua"):
        Yc[k] = Yc[k][(Yc[k].index >= FIT0) & (Yc[k].index <= LAST_CENSUS)]
    bt = backtests(Yc, Xc)
    ext = extend(Yc, Xc, bt)
    bt.pop("_preds_yl")
    bt["authorisation"] = {t: {"prereg_sha256": g["sha256"],
                               "post_registration_looks": g["post_registration_looks"]}
                           for t, g in got.items()}
    bt["gate_P2e_x1"] = bool(bt["yunlin"]["x1_pass"] and bt["changhua"]["x1_pass"])
    bt["code"] = D.p2def_code()
    D.save_json(bt, out / "P2e_backtest.json")
    ext.to_csv(out / "P2e_county_extension.csv", index=False)
    for k in ("yunlin", "changhua"):
        print(k, "selected", bt[k]["selected"],
              {c: round(v["rms_e2_last36"], 4) for c, v in bt[k]["x2"].items()},
              "x1 rms", round(bt[k]["x1_selected_rms"], 4), "pass", bt[k]["x1_pass"])
    print(ext.to_string())


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
