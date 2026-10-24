"""Model-only power check of the reserved judge unit for a Changhua fix (no elevation read).

Unit (dates only): Changhua benchmarks on the grid surveyed within 45 days of May 2010, May
2011 and May 2012: two May-to-May intervals, scored with step B's estimator (benchmark and
campaign fixed effects, E3 and E10). Declared decision rule, fixed BEFORE this ran
(2026-10-23): the unit is spent on a fix only if BOTH
  (a) the simulated 90 % CI half-width of beta_obs, with 0.5 cm independent survey noise and
      the unit's real exposure contrasts, is <= 0.40 cm/yr per GWh/yr; and
  (b) the census model's (R1c, three-zone column) own two-campaign estimates of the same
      estimator, over every pair of consecutive May-to-May intervals of its 2013-22
      hindcast at the unit's cells, are > 0 in >= 95 % of windows (so the unit could reject
      a model as strong as the failed one if the data showed no positive response).
Otherwise the unit is not spent and a fix can only be judged prospectively.
Reads survey dates/coordinates, C0 electricity and model outputs. Writes units + verdict.
Usage: results/twin_runs/par24/py.sh dataprep_staging/n1_policy/power_lev_ch_1012.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402
from n1_policy import run_step_b as B  # noqa: E402

OUT = REPORTS_DIR / "N1_policy" / "fix_judge" / "power_lev_ch_1012.json"
MAYS = [2010, 2011, 2012]
NOISE_M = 0.005
HALF_WIDTH_MAX = 0.40          # cm/yr per GWh/yr
SHARE_POSITIVE_MIN = 0.95
N_SIM = 200
SEED = 20261024


def unit_rows(grid, ch) -> pd.DataFrame:
    from hydrophysics.config import Config

    meta = pd.read_parquet(os.path.join(str(Config().data_dir), "ls_cache",
                                        "ls-wra-lsp-obs__choushui_panel.parquet"),
                           columns=["sid", "datetime", "x_3826", "y_3826"])
    meta["datetime"] = pd.to_datetime(meta.datetime)
    meta["sid"] = meta.sid.astype(str)
    rows = []
    cent = grid.centroids()
    for sid, g in meta.groupby("sid"):
        picks = []
        for y in MAYS:
            d = np.abs((g.datetime - pd.Timestamp(f"{y}-05-01")).dt.days.to_numpy())
            if d.min() > 45:
                break
            picks.append(g.datetime.iloc[int(d.argmin())])
        if len(picks) < len(MAYS):
            continue
        c = PL.nearest_cell(cent, g[["x_3826", "y_3826"]].to_numpy()[:1], grid.dx)[0]
        if c < 0 or not ch[c]:
            continue
        for a, b in zip(picks[:-1], picks[1:], strict=True):
            rows.append({"sid": sid, "cell": int(c), "t0": a, "t1": b,
                         "years": (b - a).days / 365.25, "x": g.x_3826.iloc[0],
                         "y": g.y_3826.iloc[0]})
    return pd.DataFrame(rows)


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    os.chdir(REPO)
    rng = np.random.default_rng(SEED)
    grid, ch = B.grid_and_county()
    cent = grid.centroids()
    E, dates = B.c0_energy(len(cent))
    df = unit_rows(grid, ch)
    pos = {p: k for k, p in enumerate(dates)}
    m0 = np.array([pos[pd.Period(t, "M")] for t in df.t0])
    m1 = np.array([pos[pd.Period(t, "M")] for t in df.t1])
    W3 = PL.ring_weights(cent, df.cell.to_numpy(), 0.0, B.R3)
    W10 = PL.ring_weights(cent, df.cell.to_numpy(), B.R3, B.R10)
    df["E3"] = PL.interval_exposure(E, W3, m0, m1, df.years.to_numpy())
    df["E10"] = PL.interval_exposure(E, W10, m0, m1, df.years.to_numpy())
    df["campaign"] = [t.year for t in df.t1]
    X = df[["E3", "E10"]].to_numpy()
    fe = [df.sid.to_numpy(), df.campaign.to_numpy()]
    cl = PL.blocks_of(df[["x", "y"]].to_numpy())
    # (a) simulated CI half-width: survey noise on each benchmark's 3 surveys
    sids, inv = np.unique(df.sid.to_numpy(), return_inverse=True)
    halfw = []
    for _ in range(N_SIM):
        s = rng.normal(0, NOISE_M, (len(sids), 3))
        k = df.groupby("sid").cumcount().to_numpy()
        y = (s[inv, k + 1] - s[inv, k]) / df.years.to_numpy() + 0.8e-8 * df.E3.to_numpy()
        _, ci = PL.cluster_bootstrap(lambda r, y=y: PL.fe_ols(y[r], X[r], [f[r] for f in fe]),
                                     cl, B=200, seed=int(rng.integers(1 << 30)))
        halfw.append((ci[1][0] - ci[0][0]) / 2 * 1e8)
    # (b) the census model's own two-campaign estimates at the unit's cells
    theta, col = B.MODELS["changhua"]
    tm = PL.TwinModel(str(REPO / theta), str(REPO / col))
    cells = df.drop_duplicates("sid").cell.to_numpy()
    Em = tm.energy()
    Em = Em.sum(axis=0) if Em.ndim == 3 else Em
    S = tm.subsidence(tm.heads())
    may = np.array([k for k, p in enumerate(PL.MONTHS) if p.month == 5])
    W3m = PL.ring_weights(tm.cent, cells, 0.0, B.R3)
    W10m = PL.ring_weights(tm.cent, cells, B.R3, B.R10)
    r = np.stack([S[cells, b] - S[cells, a] for a, b in zip(may[:-1], may[1:], strict=True)], 1)
    e3 = np.stack([W3m @ Em[:, a:b].sum(1) for a, b in zip(may[:-1], may[1:], strict=True)], 1)
    e10 = np.stack([W10m @ Em[:, a:b].sum(1) for a, b in zip(may[:-1], may[1:], strict=True)], 1)
    n = len(cells)
    wins = []
    for s0 in range(r.shape[1] - 1):
        sl = slice(s0, s0 + 2)
        sid = np.repeat(np.arange(n)[:, None], 2, 1).ravel()
        yr = np.repeat(np.arange(2)[None, :], n, 0).ravel()
        b = PL.fe_ols(r[:, sl].ravel(), np.column_stack([e3[:, sl].ravel(), e10[:, sl].ravel()]),
                      [sid, yr])
        wins.append(float(b[0] * 1e8))
    wins = np.array(wins)
    hw = float(np.median(halfw))
    share_pos = float(np.mean(wins > 0))
    ok = hw <= HALF_WIDTH_MAX and share_pos >= SHARE_POSITIVE_MIN
    res = {"n_benchmarks": int(df.sid.nunique()), "n_intervals": int(len(df)),
           "n_clusters": int(len(np.unique(cl))),
           "within_sd_E3_GWh": float(np.std(df.E3 - df.groupby("sid").E3.transform("mean")) / 1e6),
           "simulated_half_width_median": hw,
           "simulated_half_width_p90": float(np.quantile(halfw, 0.9)),
           "census_model_two_campaign_estimates": wins.tolist(),
           "share_positive": share_pos, "rule": {"half_width_max": HALF_WIDTH_MAX,
                                                 "share_positive_min": SHARE_POSITIVE_MIN},
           "verdict": "spend_unit" if ok else "do_not_spend",
           "unit_ids": sorted(df.sid.unique().tolist())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "unit_ids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
