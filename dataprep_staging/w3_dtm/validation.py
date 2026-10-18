"""Held-out validation of the epoch-corrected DTM against leveling benchmarks (WP W3).

Design (pre-registered in results/dataprep/preregistration/W3.dtm_heldout_bias.json before
this ran; the constants below are checked against it by build.py):

* benchmarks: main-campaign observations (April-July) of 2012-2021 that lie inside an active
  1 km cell (point in cell);
* blocks: 10 km squares anchored at the twin grid origin (x0, y0); leave-one-block-out: the
  epoch correction for the benchmarks of a block is built only from benchmarks outside it;
* residual r = (cell mean of the 20 m DTM, epoch-corrected to the observation epoch) minus
  benchmark elevation; per benchmark the median over its observations; the primary statistic
  is the median over benchmarks; acceptance |median| <= 0.30 m (plan 3, WP W3);
* secondary (reported, never gating): cell median, uncorrected DTM, point DTM (bilinear),
  SRTM cell value, a 10 km buffer variant, per-year medians, spreads, a block-bootstrap
  interval, coverage of value +- 1.645 sd_point with a block-cluster-adjusted Wilson interval,
  strata, and the held-out error of the epoch correction itself.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from w3_dtm import leveling

BLOCK_M = 10000.0
BUFFER_M = 10000.0
THRESHOLD_M = 0.30
VAL_YEARS = range(2012, 2022)
DISP_YEARS = range(2011, 2022)
BOOT_N = 2000
BOOT_SEED = 1000
Z90 = 1.6448536269514722


def block_ids(x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> np.ndarray:
    bx = np.floor((np.asarray(x) - x0) / BLOCK_M).astype(int)
    by = np.floor((np.asarray(y) - y0) / BLOCK_M).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(bx, by, strict=True)])


def cell_of_point(x, y, mask, x0, y0, dx) -> np.ndarray:
    """Active-cell index (np.nonzero(mask) order) containing each point, -1 if none."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    col = np.floor((x - x0) / dx).astype(int)
    row = np.floor((y - y0) / dx).astype(int)
    ny, nx = mask.shape
    idx = np.full(mask.shape, -1, dtype=int)
    r, c = np.nonzero(mask)
    idx[r, c] = np.arange(len(r))
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    out = np.full(len(x), -1, dtype=int)
    out[ok] = idx[row[ok], col[ok]]
    return out


def cell_correction(dm: leveling.DisplacementModel, pts: np.ndarray, t: np.ndarray,
                    epochs: list, shares: list) -> np.ndarray:
    """u(p, t) - sum_k w_k u(p, t_k) for each point p with its own epoch mix."""
    return dm.correction(pts, t, epochs, shares)


def _own_elev(series: dict, sid: str, t: float) -> float:
    ts, es = series.get(sid, (None, None))
    if ts is None or len(ts) < 2 or not (ts[0] <= t <= ts[-1]):
        return float("nan")
    return float(np.interp(t, ts, es))


def wilson(k: float, n: float, z: float = Z90) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def cluster_coverage(hit: np.ndarray, cluster: np.ndarray) -> dict:
    """Coverage with a cluster-adjusted Wilson interval (plan 4.4 / A1 recipe: effective n =
    n / design effect, design effect 1 + (m_bar - 1) ICC, ICC by one-way ANOVA)."""
    hit = np.asarray(hit, dtype=float)
    df = pd.DataFrame({"h": hit, "c": cluster})
    n = len(df)
    g = df.groupby("c")["h"]
    k = g.ngroups
    m = g.size().to_numpy()
    mbar = (n - (m ** 2).sum() / n) / (k - 1) if k > 1 else 1.0
    grand = hit.mean()
    msb = (m * (g.mean().to_numpy() - grand) ** 2).sum() / max(k - 1, 1)
    msw = ((df["h"] - g.transform("mean")) ** 2).sum() / max(n - k, 1)
    icc = (msb - msw) / (msb + (mbar - 1) * msw) if (msb + (mbar - 1) * msw) > 0 else 0.0
    icc = max(icc, 0.0)
    deff = 1 + (mbar - 1) * icc
    neff = n / deff
    lo, hi = wilson(grand * neff, neff)
    return {"coverage": float(grand), "n": int(n), "n_clusters": int(k), "icc": float(icc),
            "design_effect": float(deff), "n_eff": float(neff), "wilson90": [lo, hi],
            "contains_0.90": bool(lo <= 0.90 <= hi),
            "point_window_0.85_0.95_required": bool(neff >= 30),
            "passes_rule_4.4": bool(lo <= 0.90 <= hi and (neff < 30 or 0.85 <= grand <= 0.95))}


def summarize(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"n": 0}
    q25, q75 = np.percentile(r, [25, 75])
    return {"n": int(len(r)), "median": float(np.median(r)), "mean": float(r.mean()),
            "iqr": [float(q25), float(q75)],
            "mad_sd": float(1.4826 * np.median(np.abs(r - np.median(r)))),
            "frac_abs_gt_1m": float(np.mean(np.abs(r) > 1)),
            "frac_abs_gt_3m": float(np.mean(np.abs(r) > 3))}


def run(*, camp: pd.DataFrame, cells: pd.DataFrame, centroids: np.ndarray, mask: np.ndarray,
        x0: float, y0: float, dx: float, mos, srtm: np.ndarray | None,
        buffer: bool = False, log=print) -> pd.DataFrame:
    """Per-observation residuals for every held-out benchmark observation."""
    obs = camp[camp["year"].isin(list(VAL_YEARS))].copy()
    obs["cell"] = cell_of_point(obs["x"], obs["y"], mask, x0, y0, dx)
    obs = obs[obs["cell"] >= 0].copy()
    obs["block"] = block_ids(obs["x"], obs["y"], x0, y0)
    series = {s: (g["t"].to_numpy(), g["elev"].to_numpy())
              for s, g in camp.sort_values("t").groupby("sid")}
    allxy = camp.groupby("sid")[["x", "y"]].first()
    hp, hep = mos.bilinear(obs["x"].to_numpy(), obs["y"].to_numpy())
    obs["dtm_point_sheet"] = hp
    obs["dtm_point_epoch"] = hep
    out = []
    for blk, ob in obs.groupby("block"):
        bx, by = (int(v) for v in blk.split("_"))
        xa, ya = x0 + bx * BLOCK_M, y0 + by * BLOCK_M
        grow = BUFFER_M if buffer else 0.0
        inside = ((allxy["x"] >= xa - grow) & (allxy["x"] < xa + BLOCK_M + grow)
                  & (allxy["y"] >= ya - grow) & (allxy["y"] < ya + BLOCK_M + grow))
        train = allxy.index[~inside]
        dm = leveling.DisplacementModel(camp, DISP_YEARS, train_sids=train)
        ci = ob["cell"].to_numpy()
        cpts = centroids[ci]
        ep = [cells.at[c, "epochs"] for c in ci]
        sh = [cells.at[c, "epoch_shares"] for c in ci]
        t = ob["t"].to_numpy()
        corr_cell = cell_correction(dm, cpts, t, ep, sh)
        bpts = ob[["x", "y"]].to_numpy()
        corr_bm = cell_correction(dm, bpts, t, ep, sh)            # same epoch mix, at the BM
        u_t, _ = dm.u(bpts, t)
        u_pe, _ = dm.u(bpts, ob["dtm_point_epoch"].to_numpy())
        corr_pt = u_t[:, 0] - u_pe[:, 0]
        # the benchmark's own displacement from its cell's epoch mix to t (truth for the
        # epoch-correction error), NaN unless every sheet epoch is bracketed by its campaigns
        own = []
        for sid_, e, s in zip(ob["sid"], ep, sh, strict=True):
            vals = [_own_elev(series, sid_, float(tk)) for tk in e]
            own.append(float(np.dot(s, vals)) if np.all(np.isfinite(vals)) else np.nan)
        own = np.asarray(own)
        zc = cells.loc[ci]
        d = pd.DataFrame({
            "sid": ob["sid"].to_numpy(), "x": ob["x"].to_numpy(), "y": ob["y"].to_numpy(),
            "year": ob["year"].to_numpy(), "t": t, "elev": ob["elev"].to_numpy(),
            "cell": ci, "block": blk, "n_train_benchmarks": len(train),
            "cell_mean_sheet": zc["mean"].to_numpy(), "cell_median_sheet":
            zc["median"].to_numpy(), "sd_within": zc["sd"].to_numpy(),
            "epoch_mean": zc["epoch_mean"].to_numpy(), "corr_cell": corr_cell,
            "corr_at_bm": corr_bm, "corr_point": corr_pt,
            "dtm_point_sheet": ob["dtm_point_sheet"].to_numpy(),
            "dtm_point_epoch": ob["dtm_point_epoch"].to_numpy(),
            "own_disp": ob["elev"].to_numpy() - own,
        })
        d["r_mean"] = d["cell_mean_sheet"] + d["corr_cell"] - d["elev"]
        d["r_median"] = d["cell_median_sheet"] + d["corr_cell"] - d["elev"]
        d["r_mean_uncorrected"] = d["cell_mean_sheet"] - d["elev"]
        d["r_point"] = d["dtm_point_sheet"] + d["corr_point"] - d["elev"]
        d["r_srtm"] = (srtm[ci] - d["elev"]) if srtm is not None else np.nan
        d["epoch_err"] = d["corr_at_bm"] - d["own_disp"]
        d["dt"] = d["t"] - d["epoch_mean"]
        out.append(d)
    res = pd.concat(out, ignore_index=True)
    if log:
        log(f"validation{' (buffer)' if buffer else ''}: {res['sid'].nunique()} benchmarks, "
            f"{len(res)} observations, {res['block'].nunique()} blocks")
    return res


def per_benchmark(res: pd.DataFrame, col: str) -> pd.Series:
    return res.groupby("sid")[col].median()


def epoch_error_slope(res: pd.DataFrame) -> dict:
    """sigma_epoch(dt) = a |dt|, a by least squares on squared errors through the origin."""
    d = res[np.isfinite(res["epoch_err"]) & (res["dt"].abs() > 0.25)]
    if not len(d):
        return {"a_m_per_yr": float("nan"), "n": 0}
    a = float(np.sqrt((d["epoch_err"] ** 2).sum() / (d["dt"] ** 2).sum()))
    bins = [0.25, 1, 2, 3, 4, 5, 7]
    tab = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        s = d[(d["dt"].abs() > lo) & (d["dt"].abs() <= hi)]
        if len(s):
            tab.append({"abs_dt": [lo, hi], "n": int(len(s)),
                        "rms_m": float(np.sqrt((s["epoch_err"] ** 2).mean())),
                        "median_m": float(s["epoch_err"].median())})
    cov = float(np.mean(np.abs(d["epoch_err"]) <= Z90 * a * d["dt"].abs()))
    return {"a_m_per_yr": a, "n": int(len(d)), "by_abs_dt": tab,
            "coverage_of_1.645_a_dt": cov}


def epoch_error_cv(res: pd.DataFrame) -> dict:
    """Leave-one-block-out check of sigma_epoch = a |dt|: a is refitted without the block and
    the 90 % interval is checked on the block (added after the pre-registration, reported
    only)."""
    d = res[np.isfinite(res["epoch_err"]) & (res["dt"].abs() > 0.25)]
    hits, blocks = [], []
    for b, g in d.groupby("block"):
        o = d[d["block"] != b]
        a = float(np.sqrt((o["epoch_err"] ** 2).sum() / (o["dt"] ** 2).sum()))
        hits.append((g["epoch_err"].abs() <= Z90 * a * g["dt"].abs()).to_numpy())
        blocks.append(np.full(len(g), b))
    if not hits:
        return {"n": 0}
    return cluster_coverage(np.concatenate(hits), np.concatenate(blocks))


def block_bootstrap_median(res: pd.DataFrame, col: str, n: int = BOOT_N,
                           seed: int = BOOT_SEED) -> list[float]:
    rb = res.groupby("sid").agg(r=(col, "median"), block=("block", "first"))
    blocks = rb["block"].unique()
    groups = {b: rb.loc[rb["block"] == b, "r"].to_numpy() for b in blocks}
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        meds.append(float(np.median(np.concatenate([groups[b] for b in pick]))))
    return [float(np.percentile(meds, 5)), float(np.percentile(meds, 95))]
