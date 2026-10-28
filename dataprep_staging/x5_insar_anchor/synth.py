"""X5 synthetic recovery check S0 (declared in X5.synthetic_anchor.v1 before it is run).

A known displacement field is sampled through the real geometry of the T1 test (run P2's
epochs and valid-pixel pattern, the anchor stations and their daily availability, the
benchmarks with both the 2020-05 and 2021-05 surveys), then corrupted the way the audit
found the InSAR to be corrupted: a common mode of +-22-32 mm per quarter (random sign) with
5 mm per-epoch jitter, a smooth per-epoch field (about 10 km, 3 mm), 4 mm pixel noise, a
3 mm/sqrt(yr) random walk between each GNSS station and its InSAR pixels and two stations
drifting +-40 mm/yr. GNSS carries 5 mm daily noise, leveling 3 mm and survey dates jittered
+-20 days. The T1 pipeline (x5lib: median-polish offset anchoring, May windows, scaling,
metrics) recovers the leveling increments.

Pass (x5const.SYN_PASS): C1 median over 20 seeds of the OLS slope divided by its known
errors-in-variables attenuation lambda in [0.95, 1.05]; C2 median RMSE <= 5 mm; C3 the T1 rule
(slope 0.8-1.2 and RMSE <= 10 mm) in >= 19 of 20 seeds.

No InSAR, GNSS or leveling VALUE is read: only P2's valid-pixel pattern and epochs, station
positions and daily availability, and benchmark positions.

    python -I synth.py [--smoke]   (--smoke: one seed, prints, writes nothing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "s4_insar")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4br_lib as RL  # noqa: E402
import x5const as K  # noqa: E402
import x5lib as X  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from scipy import ndimage  # noqa: E402

OUTD = K.OUT / "synthetic"
T0 = pd.Timestamp(K.REF_EPOCH)


def geometry() -> dict:
    run = RL.load_run("P2")
    g = RL.load_geom(run["frame"])
    seg = X._segments(run["gap"], run["cum"].shape)
    valid = run["mask"][None] & np.isfinite(run["cum"]) & (seg == 0)
    st = X.anchor_stations()
    avail = {}
    for fam, path in (("wra", K.GNSS_WRA), ("ngl", K.GNSS_NGL)):
        df = pd.read_parquet(path, columns=["station", "date", "flag"])
        df = df[df["flag"].isin(K.GNSS_FLAGS)]
        for s in st.loc[st["family"] == fam, "station"]:
            d = pd.DatetimeIndex(pd.to_datetime(df.loc[df["station"] == s, "date"]))
            avail[s] = d[(d >= pd.Timestamp("2019-12-01")) & (d <= pd.Timestamp("2023-01-31"))]
    bench = X.bench_geometry()
    return {"dates": pd.DatetimeIndex(run["dates"]), "valid": valid, "x": g["x"], "y": g["y"],
            "stations": st, "avail": avail, "bench": bench}


class Truth:
    """u(x, y, t) in mm (positive up), zero at T0."""

    def __init__(self, rng, bx, by):
        x0, x1 = np.percentile(bx, (5, 95))
        y0, y1 = np.percentile(by, (5, 95))
        fr = [(0.35, 0.25, 60.0, 8e3), (0.6, 0.45, 40.0, 10e3), (0.4, 0.7, 25.0, 6e3)]
        self.b = [(x0 + fx * (x1 - x0) + rng.normal(0, 2e3), y0 + fy * (y1 - y0)
                   + rng.normal(0, 2e3), r * rng.uniform(0.8, 1.2), s) for fx, fy, r, s in fr]

    def rate(self, x, y):
        r = 5.0 + sum(R * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))
                      for cx, cy, R, s in self.b)
        return -r                                    # mm/yr, sinking

    def u(self, x, y, t):
        t = pd.DatetimeIndex(np.atleast_1d(t))
        yr = ((t - T0).days.to_numpy(float) / 365.25)
        amp = K.SYN_SEASONAL_MM * np.clip(np.abs(self.rate(x, y)) / 40.0, 0.3, 1.5)
        ph = lambda tt: np.sin(2 * np.pi * (tt - 0.2))  # noqa: E731
        r = self.rate(x, y)
        return (np.multiply.outer(yr, r) + np.multiply.outer(ph(yr) - ph(0.0), amp))


def common_mode(rng, dates) -> np.ndarray:
    q = pd.period_range("2019Q4", "2023Q1", freq="Q")
    mid = q.start_time + (q.end_time - q.start_time) / 2
    vals = rng.choice([-1, 1], len(q)) * rng.uniform(*K.SYN_CM_ABS_MM, len(q))
    xs = (mid - T0).days.to_numpy(float)
    cm = np.interp((dates - T0).days.to_numpy(float), xs, vals)
    return cm + rng.normal(0, K.SYN_CM_JITTER_MM, len(dates))


def one_seed(seed: int, G: dict) -> dict:
    rng = np.random.default_rng(seed)
    bench, st, dates = G["bench"], G["stations"], G["dates"]
    tr = Truth(rng, bench["x"].to_numpy(), bench["y"].to_numpy())
    T, L, W = G["valid"].shape
    cm = common_mode(rng, dates)
    field = np.full((T, L, W), np.nan, np.float32)
    base = tr.u(G["x"], G["y"], dates)               # (T, L, W)
    for t in range(T):
        aps = ndimage.zoom(rng.normal(0, 1, (9, 8)), (L / 9, W / 8), order=3)[:L, :W]
        aps = aps / max(aps.std(), 1e-9) * K.SYN_APS_MM
        f = base[t] + cm[t] + aps + rng.normal(0, K.SYN_PIX_NOISE_MM, (L, W))
        field[t] = np.where(G["valid"][t], f, np.nan)
    ins = {"value": field, "valid": G["valid"], "dates": dates, "x": G["x"], "y": G["y"]}
    # synthetic GNSS on each station's real available days
    bad = set(rng.choice(st["station"].to_numpy(), K.SYN_N_BAD_STATIONS, replace=False))
    raw = {}
    for _, r in st.iterrows():
        d = G["avail"].get(r["station"])
        if d is None or len(d) < 3:
            continue
        yr = (d - T0).days.to_numpy(float) / 365.25
        steps = rng.normal(0, K.SYN_DRIFT_MM_SQRT_YR * np.sqrt(np.maximum(np.diff(yr,
                           prepend=yr[0]), 0)))
        drift = np.cumsum(steps)
        if r["station"] in bad:
            drift = drift + rng.choice([-1, 1]) * K.SYN_BAD_RATE_MM_YR * yr
        u = tr.u(np.array([r["x_twd97"]]), np.array([r["y_twd97"]]), d)[:, 0]
        raw[r["station"]] = pd.Series(u + drift + 1e5 + rng.normal(0, K.SYN_GNSS_NOISE_MM,
                                                                    len(d)), index=d)
    series = X.smooth_raw(raw)
    S = X.station_inputs(ins, st, series)
    bx, by = bench["x"].to_numpy(float), bench["y"].to_numpy(float)
    Ib = X.point_matrix(ins, bx, by)
    j = rng.integers(-K.SYN_LEV_DATE_JITTER_D, K.SYN_LEV_DATE_JITTER_D + 1, (2, len(bench)))
    d0 = pd.Timestamp("2020-05-15") + pd.to_timedelta(j[0], "D")
    d1 = pd.Timestamp("2021-05-15") + pd.to_timedelta(j[1], "D")
    lev = np.array([tr.u(np.array([x]), np.array([y]), pd.DatetimeIndex([a, b]))[:, 0]
                    @ np.array([-1.0, 1.0]) for x, y, a, b in zip(bx, by, d0, d1,
                                                                    strict=True)])
    lev_true = lev.copy()
    lev = lev + rng.normal(0, K.SYN_LEV_NOISE_MM, len(lev))
    out = {"seed": seed, "bad_stations": sorted(bad)}
    corr = X.epoch_corrections(S["I"], S["G"], S["x"], S["y"], "offset")
    for name, A in (("anchored", Ib + X.correction_at(corr, bx, by)), ("raw", Ib)):
        inc = X.may_increment(A, dates, *K.T1_YEARS, scaled=True)["inc"]
        ok = np.isfinite(inc)
        m = X.core_metrics(lev[ok], inc[ok])
        m["t1_rule"] = X.t1_verdict(m)
        m["lambda"] = float(np.var(lev_true[ok]) / np.var(lev[ok]))
        m["slope_over_lambda"] = m["slope_ols_insar_on_leveling"] / m["lambda"]
        m["rmse_vs_true_mm"] = float(np.sqrt(np.mean((inc[ok] - lev_true[ok]) ** 2)))
        out[name] = m
    out["epochs_with_anchor"] = int(np.isfinite(corr["c"]).sum())
    out["median_stations_per_epoch"] = float(np.median(corr["n"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    G = geometry()
    seeds = K.SYN_SEEDS[:1] if a.smoke else K.SYN_SEEDS
    if not a.smoke:
        OUTD.mkdir(parents=True, exist_ok=True)
        path = OUTD / "S0_result.json"
        if path.exists():
            raise SystemExit(f"{path} exists; the synthetic check runs once")
    res = [one_seed(s, G) for s in seeds]
    for r in res:
        print(r["seed"], {k: round(r["anchored"][k], 3) for k in
                          ("n", "slope_ols_insar_on_leveling", "lambda", "slope_over_lambda",
                           "rmse_mm", "rmse_vs_true_mm", "bias_mm")},
              "raw:", {k: round(r["raw"][k], 3) for k in
                       ("slope_ols_insar_on_leveling", "rmse_mm", "bias_mm")})
    if a.smoke:
        return 0
    sl = [r["anchored"]["slope_over_lambda"] for r in res]
    rm = [r["anchored"]["rmse_mm"] for r in res]
    npass = sum(r["anchored"]["t1_rule"]["pass"] for r in res)
    lo, hi = K.SYN_PASS["median_slope_over_lambda"]
    crit = {"C1_median_slope_over_lambda_in_band": bool(lo <= np.median(sl) <= hi),
            "C2_median_rmse_ok": bool(np.median(rm) <= K.SYN_PASS["median_rmse_mm"]),
            "C3_seeds_passing_t1_rule_ok": bool(npass >= K.SYN_PASS["min_seeds_passing_t1"])}
    summary = {
        "prereg": K.PREREG_SYN, "criterion": K.SYN_PASS, "n_seeds": len(res),
        "median_slope_over_lambda": float(np.median(sl)),
        "median_slope_ols": float(np.median([r["anchored"]["slope_ols_insar_on_leveling"]
                                             for r in res])),
        "median_lambda": float(np.median([r["anchored"]["lambda"] for r in res])),
        "median_bias_mm": float(np.median([r["anchored"]["bias_mm"] for r in res])),
        "median_rmse_vs_true_mm": float(np.median([r["anchored"]["rmse_vs_true_mm"]
                                                   for r in res])),
        "median_rmse_mm": float(np.median(rm)),
        "seeds_passing_t1_rule": int(npass), "criteria": crit, "pass": bool(all(crit.values())),
        "raw_unanchored": {
            "median_slope": float(np.median([r["raw"]["slope_ols_insar_on_leveling"]
                                             for r in res])),
            "median_rmse_mm": float(np.median([r["raw"]["rmse_mm"] for r in res])),
            "seeds_passing_t1_rule": int(sum(r["raw"]["t1_rule"]["pass"] for r in res)),
            "note": "reported, not part of the criterion"},
        "settings": {k: getattr(K, k) for k in dir(K) if k.startswith("SYN_")},
        "code": [rel_to_repo(HERE / f) for f in ("synth.py", "x5lib.py", "x5const.py")],
        "seeds": res}
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("median_slope_over_lambda", "median_slope_ols",
                                              "median_lambda", "median_rmse_mm",
                                              "seeds_passing_t1_rule", "criteria", "pass",
                                              "raw_unanchored")}, indent=1))
    print("wrote", rel_to_repo(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
