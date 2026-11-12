"""Design study for a gating test of the statistical layer against the page column.

Unit       : WRA's printed township statement of one survey year: for each of the 45
             townships of Yunlin and Changhua an interval for the peak of the contoured
             rate field (listed: 3 cm/yr to the county maximum; not listed: below 3 cm/yr;
             the county's maximum township: the printed value).
Score      : per arm and year, the mean over townships of the distance (cm/yr) from the
             arm's township peak to the printed interval.
Arms       : L  the layer (C0: frozen rate field; HL: with the head term),
             ZR the page column's forward run, Z zero.
Year rule  : L wins a year iff its score is below both ZR's and Z's.
Test rule  : supported iff L wins every decided year and the paired bootstrap 95 % lower
             bound of mean(e_best_other - e_L) over township-years is above 0.

Replay     : the prospective test predicts survey years 2027 and 2028 from rates of
             2019-2021, a lead of 6 and 7 years. Each replay world takes an origin o, rates
             of o-2..o, and scores survey years o+6 and o+7 against the pre-origin printed
             bounds (survey years 2015-2022; design_time_evidence). The column is the page
             model's forward run, fitted on those years, which favours the column.
Null worlds: the printed intervals are rebuilt from a synthetic truth in which the layer
             carries no information (township peaks permuted within county) or sinking has
             stopped (all peaks scaled by a factor), to measure the false-pass rate.

Run once: ``python -m dataprep_staging.r12_layer.gate_power`` writes
``reports/SL/gate_power/result.json`` and refuses to overwrite it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from dataprep_staging.r12_layer import model as M
from dataprep_staging.r12_leveling.paths import DATAPREP, FWD

SEED = 20261113
N_BOOT = 2000
N_NULL = 2000
LEADS = (6, 7)
ORIGINS = tuple(range(2009, 2016))
BOUNDS = os.path.join(DATAPREP, "subsidence_public_reports", "preorigin", "v1",
                      "township_bounds.csv")
FORWARD = FWD
POWER_MIN, FALSE_MAX = 0.80, 0.10
QUIET = (1.0, 0.75, 0.5, 0.25)       # truth = factor x the layer's own field (null of level)


def interval_error(pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.maximum(lo - pred, 0.0) + np.maximum(pred - hi, 0.0)


def printed(bounds: pd.DataFrame, year: int) -> pd.DataFrame:
    b = bounds[bounds.survey_year == year].copy()
    b["county"] = b.county.str.replace("縣", "") + "縣"
    v = b.peak_rate_value_cm_yr
    b["lo"] = np.where(v.notna(), v, b.peak_rate_lower_cm_yr.fillna(-np.inf))
    b["hi"] = np.where(v.notna(), v, b.peak_rate_upper_cm_yr.fillna(np.inf))
    return b.set_index(["county", "township"])[["lo", "hi"]]


def bounds_from_truth(peaks: pd.Series) -> pd.DataFrame:
    """The printed statement a truth field of township peaks would produce."""
    out = pd.DataFrame({"lo": -np.inf, "hi": 3.0}, index=peaks.index)
    for _, g in peaks.groupby(level=0):
        mx = g.max()
        listed = g.index[g > 3.0]
        out.loc[listed, "lo"] = 3.0
        out.loc[listed, "hi"] = mx
        out.loc[g.idxmax(), ["lo", "hi"]] = mx
    return out


class Column:
    def __init__(self, cells: pd.DataFrame):
        z = np.load(FORWARD, allow_pickle=True)
        names = [str(s) for s in z["scenario_names"]]
        self.s = z["subs_mean"][names.index("baseline")] * 100.0
        self.months = pd.PeriodIndex(pd.to_datetime([str(d) for d in z["dates"]]), freq="M")
        self.cell = cells.cell.to_numpy(int)

    def rates(self, year: int) -> np.ndarray:
        a = self.months.get_loc(pd.Period(f"{year - 1}-05", freq="M"))
        b = self.months.get_loc(pd.Period(f"{year}-05", freq="M"))
        return (self.s[self.cell, b] - self.s[self.cell, a]) * 12.0 / (b - a)


def peaks(cells: pd.DataFrame, rate_cm: np.ndarray) -> pd.Series:
    """Township peak; a township without a predicted cell is predicted 0."""
    t = cells.assign(r=rate_cm).groupby(["county", "township"]).r.max()
    return t.fillna(0.0)


def layer_fields(L: M.Layer, origin: int, year: int, gamma: float) -> np.ndarray:
    yrs = (origin - 2, origin - 1, origin)
    rate = M.rate_at(L.D, L.xy, yrs)
    if gamma != 0.0:
        ref = np.nanmean(L.F[:, [M.YEARS.index(y) for y in yrs]], axis=1)
        drv = L.F[:, M.YEARS.index(year)] - ref
        rate = rate + gamma * np.nan_to_num(drv) * np.clip(rate, 0.002, None)
    return 100.0 * M.to_cells(rate, L.xy, L.cxy)


def decide(e_l: list, e_zr: list, e_z: list, rng) -> dict:
    """Lists hold one error vector (townships) per year."""
    wins = [a.mean() < min(b.mean(), c.mean()) for a, b, c in zip(e_l, e_zr, e_z, strict=True)]
    best = [b if b.mean() <= c.mean() else c for b, c in zip(e_zr, e_z, strict=True)]
    d = np.concatenate([b - a for a, b in zip(e_l, best, strict=True)])
    ix = rng.integers(0, len(d), (N_BOOT, len(d)))
    lo = float(np.quantile(d[ix].mean(1), 0.05))
    return {"wins": [bool(w) for w in wins], "mean_gain_cm_yr": float(d.mean()),
            "gain_lower95": lo, "supported": bool(all(wins) and lo > 0)}


def run() -> dict:
    rng = np.random.default_rng(SEED)
    L = M.Layer()
    with open(os.path.join(DATAPREP, "reports", "SL", "freeze", "model.json")) as fh:
        gamma = json.load(fh)["gamma_per_m"]
    col = Column(L.cells)
    bounds = pd.read_csv(BOUNDS)
    out = {"seed": SEED, "gamma": gamma, "leads": LEADS, "replay": {}, "null": {}}

    for arm, g in (("C0", 0.0), ("HL", gamma)):
        worlds = []
        for o in ORIGINS:
            el, ezr, ez, per_year = [], [], [], {}
            for lead in LEADS:
                y = o + lead
                pb = printed(bounds, y)
                pl = peaks(L.cells, layer_fields(L, o, y, g)).reindex(pb.index).fillna(0.0)
                pz = peaks(L.cells, col.rates(y)).reindex(pb.index).fillna(0.0)
                lo, hi = pb.lo.to_numpy(), pb.hi.to_numpy()
                a = interval_error(pl.to_numpy(), lo, hi)
                b = interval_error(pz.to_numpy(), lo, hi)
                c = interval_error(np.zeros(len(pb)), lo, hi)
                el.append(a), ezr.append(b), ez.append(c)
                per_year[y] = {"L": float(a.mean()), "ZR": float(b.mean()),
                               "Z": float(c.mean()), "n": int(len(pb))}
            worlds.append({"origin": o, "scores": per_year, **decide(el, ezr, ez, rng)})
        out["replay"][arm] = {"worlds": worlds,
                              "pass_rate": float(np.mean([w["supported"] for w in worlds]))}

    # null worlds on the prospective geometry: frozen layer (origin 2021) against the column
    pl = peaks(L.cells, layer_fields(L, 2021, 2027, 0.0))
    pz = [peaks(L.cells, col.rates(y)).reindex(pl.index).fillna(0.0) for y in (2027, 2028)]
    sd = 1.0                                           # cm/yr township noise of the truth
    for name, factor, permute in [(f"level_x{f:.2f}", f, False) for f in QUIET] + \
                                 [("permuted_within_county", 1.0, True)]:
        n_pass = 0
        for _ in range(N_NULL):
            el, ezr, ez = [], [], []
            for k in range(2):
                truth = factor * pl + rng.normal(0, sd, len(pl))
                if permute:
                    truth = truth.groupby(level=0).transform(
                        lambda s: rng.permutation(s.to_numpy()))
                tb = bounds_from_truth(truth)
                lo, hi = tb.lo.to_numpy(), tb.hi.to_numpy()
                el.append(interval_error(pl.to_numpy(), lo, hi))
                ezr.append(interval_error(pz[k].to_numpy(), lo, hi))
                ez.append(interval_error(np.zeros(len(pl)), lo, hi))
            n_pass += decide(el, ezr, ez, rng)["supported"]
        out["null"][name] = n_pass / N_NULL
    power = out["replay"]["C0"]["pass_rate"]
    false = out["null"]["permuted_within_county"]
    out["decision"] = {"replay_pass_rate_C0": power, "false_pass_permuted": false,
                       "rule": f"gating iff replay pass rate >= {POWER_MIN} and permuted-world "
                               f"pass rate <= {FALSE_MAX}",
                       "role": "gating" if power >= POWER_MIN and false <= FALSE_MAX
                       else "report_only"}
    return out


if __name__ == "__main__":
    d = os.path.join(DATAPREP, "reports", "SL", "gate_power")
    path = os.path.join(d, "result.json")
    if os.path.exists(path) and "--force" not in sys.argv:
        raise SystemExit(f"{path} exists: the design study is run once")
    res = run()
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
