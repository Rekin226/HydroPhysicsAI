"""Step 7b (after the registered scoring): are 09030212's once-daily months 2019-2022
consistent with its hourly era? After-the-fact analysis, never a gate (look logged first).

``H3.snapshot`` tested whether a month of 00:00 readings represents the monthly mean when
the instrument is the same. It cannot test whether the once-daily record of 2019-2022 comes
from the same instrument and datum. This check compares the well with its neighbours:

* neighbours: the 4 wells most correlated with 09030212 (monthly C20 values 2012-2018,
  >= 24 common months, r >= 0.6, within 25 km; H2's neighbour rule) that are observed (C20)
  in 2019-2022;
* a ridge regression (alpha 1e-3, standardised predictors) of 09030212 on them, fitted on
  2012-2018; residual RMSE by leave-one-year-out in 2012-2018 against the residual RMSE of
  the once-daily months 2019-2022 (admitted months and months the persistence screen
  excluded, separately), and the mean residual (a datum shift would show as a bias).

Writes ``H3_evidence/after_fact/snapshot_consistency.json``.

Usage: python -I s7b_snapshot_consistency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import prereg3  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory, snapshot  # noqa: E402

RH = "09030212"


def ridge_fit(X, y, alpha=1e-3):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    A = Z1.T @ Z1 + alpha * np.diag([0] + [1] * Z.shape[1])
    b = np.linalg.solve(A, Z1.T @ y)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b


def main():
    mon = snapshot.l1_monthly()
    Y = mon.pivot(index="date", columns="sid", values="value")
    md = inventory.load_meta().set_index("sid")
    x0, y0 = md.loc[RH, "x_twd97"], md.loc[RH, "y_twd97"]
    pre = Y.loc[:"2018-12-01"]
    post = Y.loc["2019-01-01":]
    cands = []
    for s in Y.columns:
        if s == RH or not np.isfinite(md.loc[s, "x_twd97"]):
            continue
        if np.hypot(md.loc[s, "x_twd97"] - x0, md.loc[s, "y_twd97"] - y0) > 25000:
            continue
        ok = pre[[RH, s]].dropna()
        if len(ok) < 24 or post[s].notna().sum() < 36:
            continue
        r = ok[RH].corr(ok[s])
        if r >= 0.6:
            cands.append((r, s))
    nb = [s for _, s in sorted(cands, reverse=True)[:4]]
    unit = "H3:after_fact_09030212_neighbours"
    prereg3.record_look([unit], "heads:2012-01..2022-12", C.WP, "after_the_fact",
                        members={unit: {"description": "09030212 against its 4 most "
                                        "correlated neighbours 2012-2022 (C20 months) and its "
                                        "once-daily months 2019-2022", "ids": [RH] + nb,
                                        "window": "2012-01..2022-12"}},
                        note="after the H3.recovery.v1 scoring; never a gate")
    tr = pre[[RH] + nb].dropna()
    loyo = []
    for yr in sorted(set(tr.index.year)):
        a, b = tr[tr.index.year != yr], tr[tr.index.year == yr]
        f = ridge_fit(a[nb].to_numpy(), a[RH].to_numpy())
        loyo.extend(b[RH].to_numpy() - f(b[nb].to_numpy()))
    loyo = np.asarray(loyo)
    f = ridge_fit(tr[nb].to_numpy(), tr[RH].to_numpy())
    daily = snapshot.l1_daily([RH])
    d = daily[(daily.sid == RH) & daily.flag.isin([0, 1]) & ~daily.day_valid.astype(bool)
              & (daily.date >= snapshot.SNAP_START) & (daily.date <= snapshot.SNAP_END)]
    s = d.set_index("date")["value"]
    g = s.groupby(s.index.to_period("M")).agg(["mean", "count"])
    g = g[g["count"] >= snapshot.MIN_DAYS]
    scr = snapshot.screen_daily(s)
    rows = []
    for per, r in g.iterrows():
        ts = per.to_timestamp()
        if ts not in post.index or post.loc[ts, nb].isna().any():
            continue
        pred = float(f(post.loc[[ts], nb].to_numpy())[0])
        rows.append({"month": str(per), "value": float(r["mean"]), "pred": pred,
                     "resid": float(r["mean"]) - pred,
                     "screen_excluded": str(per) in scr["excluded_months"]})
    R = pd.DataFrame(rows)

    def stats(v):
        v = np.asarray(v, float)
        return {"n": int(len(v)), "rmse": float(np.sqrt(np.mean(v ** 2))) if len(v) else None,
                "bias": float(np.mean(v)) if len(v) else None}

    rep = {"well": RH, "neighbours": nb,
           "neighbour_corr_2012_2018": {s: float(r) for r, s in cands if s in nb},
           "loyo_2012_2018": stats(loyo),
           "once_daily_admitted_months": stats(R[~R.screen_excluded].resid),
           "once_daily_screen_excluded_months": stats(R[R.screen_excluded].resid),
           "monthly_residuals": R.to_dict("records")}
    od = C.EVIDENCE / "after_fact"
    od.mkdir(parents=True, exist_ok=True)
    (od / "snapshot_consistency.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "monthly_residuals"}, indent=1))


if __name__ == "__main__":
    main()
