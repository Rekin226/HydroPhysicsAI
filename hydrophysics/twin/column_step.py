"""Score a compaction column on the 182 km step, leveling, rings and policy response (CPU).

    python -m hydrophysics.twin.column_step \\
        --run datum results/twin_forward/datum_gate.npz \\
              results/twin_runs/stage3_datum_gate/coupled_leveling \\
        --run band_1e-3 results/twin_forward/datum_band_1e-3.npz \\
              results/twin_runs/stage3_datum_gate/coupled_leveling_band_1e-3 \\
        --out results/twin/column_step_band

Each ``--run LABEL FORWARD_NPZ COLUMN_DIR`` pairs a ``twin.forward`` run with the
``calibrate_coupled`` directory of its column. Reported per run (first column of a run
with a rheology axis, which is the one the app shows):

- **leveling out of fold / in sample, rings**: the column's own scores, from
  ``COLUMN_DIR/stage4_column.csv`` (the row of the config its ``vep_*.json`` records);
- **forward hindcast leveling R2**: the forward run's baseline hindcast against the
  leveling network (``explorer3d.validate_against_leveling``), when the data are there;
- **the step across the line** (``--line-km``, default 182): per-cell subsidence rates
  (least-squares slope, cm/yr) over the hindcast (record start to the forecast origin)
  and the forecast (origin to horizon). In each window ``|x - line| <= w`` km
  (``--windows``, default 2,4,6) the rates are regressed on
  ``a + b x + c y + d y^2 + step * [x >= line]`` and ``step`` is reported with its
  standard error. The same regression on the leveling benchmarks' rates (per-site
  least-squares, record window) is the observed step, and the model's hindcast rates at
  those benchmarks' own cells give a like-for-like model step;
- **first-column contrast**: median forward change (origin to horizon, cm) in the first
  column of cells east and west of the line, the number the app's caveat quotes
  (``app.prep.boundary_contrast``);
- **policy response**: for every non-baseline scenario, the fan-mean subsidence avoided
  at the horizon and by the end of the first policy year, and the fan-mean sinking rate
  over the last five years with and without the policy; plus the ``policy_gate`` verdict
  when ``COLUMN_DIR/scorecard.json`` exists.

Writes ``<out>.json`` (everything) and ``<out>.csv`` (one row per run and window).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

LINE_KM = 182.0
WINDOWS_KM = (2.0, 4.0, 6.0)


class _Grid:
    """What ``validate_against_leveling`` needs of a grid, without ``pyproj``."""

    def __init__(self, cent: np.ndarray, dx: float):
        self._cent, self.dx = cent, float(dx)

    def centroids(self) -> np.ndarray:
        return self._cent


def cell_centroids(fw) -> np.ndarray:
    """``(A, 2)`` cell centres (m) of a forward npz, in the solver's active-cell order."""
    mask = np.asarray(fw["mask"], dtype=bool)
    rows, cols = np.nonzero(mask)
    dx = float(fw["dx"])
    return np.column_stack([float(fw["x0"]) + (cols + 0.5) * dx,
                            float(fw["y0"]) + (rows + 0.5) * dx])


def slope_per_year(series: np.ndarray, t_years: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row of ``series`` ``(n, T)`` against ``t_years``."""
    t = np.asarray(t_years, dtype="float64")
    tc = t - t.mean()
    s = np.asarray(series, dtype="float64")
    return (s - s.mean(axis=1, keepdims=True)) @ tc / float((tc ** 2).sum())


def step_fit(x_km: np.ndarray, y_km: np.ndarray, rate: np.ndarray,
             line_km: float = LINE_KM, window_km: float = 4.0) -> dict:
    """``rate = a + b x + c y + d y^2 + step [x >= line]`` over ``|x - line| <= window``.

    Coordinates are centred on the window (x on the line, y on its mean) so that the
    quadratic in y is well conditioned. Returns the step, its standard error and the
    sample sizes on each side (NaN when a side is empty or the fit is not identified)."""
    x = np.asarray(x_km, dtype="float64")
    y = np.asarray(y_km, dtype="float64")
    r = np.asarray(rate, dtype="float64")
    sel = (np.abs(x - line_km) <= window_km) & np.isfinite(r)
    east = x >= line_km
    out = {"window_km": float(window_km), "n": int(sel.sum()),
           "n_east": int((sel & east).sum()), "n_west": int((sel & ~east).sum()),
           "step": float("nan"), "se": float("nan")}
    if out["n_east"] == 0 or out["n_west"] == 0:
        return out
    xs, ys = x[sel] - line_km, y[sel] - y[sel].mean()
    X = np.column_stack([np.ones(xs.size), xs, ys, ys ** 2, east[sel].astype("float64")])
    if X.shape[0] <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return out
    beta, *_ = np.linalg.lstsq(X, r[sel], rcond=None)
    resid = r[sel] - X @ beta
    dof = X.shape[0] - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    out.update(step=float(beta[-1]), se=float(np.sqrt(max(cov[-1, -1], 0.0))))
    return out


def first_column_contrast(field: np.ndarray, x_km: np.ndarray, line_km: float = LINE_KM,
                          dx_km: float = 1.0) -> dict:
    """Median of ``field`` in the first column of cells east and west of the line
    (the same definition as ``app.prep.boundary_contrast``)."""
    east = (x_km > line_km) & (x_km <= line_km + dx_km)
    west = (x_km <= line_km) & (x_km > line_km - dx_km)

    def med(m):
        return float(np.median(field[m])) if m.any() else float("nan")

    return {"east": med(east), "west": med(west), "n_east": int(east.sum()),
            "n_west": int(west.sum())}


def _first_column_subs(fw) -> np.ndarray:
    """``(S, A, T)`` ensemble-mean subsidence in cm, the first column of a rheology axis."""
    labels = [str(v) for v in fw["rheology_labels"]] if "rheology_labels" in fw.files else []
    if len(labels) > 1 and "subs_mean_by_rheology" in fw.files:
        return fw["subs_mean_by_rheology"][0].astype("float64") * 100.0
    return fw["subs_mean"].astype("float64") * 100.0


def _years(dates) -> np.ndarray:
    d = pd.to_datetime([str(v) for v in dates])
    return np.asarray((d - d[0]).days, dtype="float64") / 365.25


def leveling_rates(ddir: str, t0: str, t1: str, min_obs: int = 5,
                   max_rate: float | None = 0.5) -> pd.DataFrame:
    """Per-benchmark least-squares rate (cm/yr, positive = sinking) over ``[t0, t1)``,
    with the same site screen as the column calibration."""
    from .leveling import load_panel, site_subsidence, site_xy

    panel = load_panel(ddir)
    obs = site_subsidence(panel, t0, t1, min_obs=min_obs, max_rate=max_rate)
    xy = site_xy(panel)
    rows = []
    for sid, s in obs.items():
        if sid not in xy or len(s) < 3:
            continue
        t = np.array([(i - s.index[0]).days / 365.25 for i in s.index], dtype="float64")
        if t.max() - t.min() < 2.0:
            continue
        rows.append({"sid": str(sid), "x": xy[sid][0], "y": xy[sid][1],
                     "rate_cm_yr": 100.0 * float(np.polyfit(t, s.to_numpy("float64"), 1)[0])})
    return pd.DataFrame(rows)


def policy_summary(subs: np.ndarray, dates, origin: int, names: list[str],
                   descs: list[str]) -> list[dict]:
    """Fan-mean avoided subsidence per non-baseline scenario (cm): at the horizon and
    by the end of the first policy year, and the last-five-year sinking rates."""
    d = pd.to_datetime([str(v) for v in dates])
    fan = subs.mean(axis=1)                                           # (S, T)
    fwd = fan - fan[:, [origin]]
    t = _years(dates)
    last5 = t >= t[-1] - 5.0 + 1e-9
    out = []
    for s in range(1, subs.shape[0]):
        # the scenario's description: "cut30: irrigation x0.7 from 2026-01"
        m = re.search(r"(?:from |@)(\d{4})-(\d{2})", descs[s])
        start = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-01") if m else None
        first_dec = (int(np.nonzero((d.month == 12) & (d.year == start.year))[0][-1])
                     if start is not None and ((d.month == 12) & (d.year == start.year)).any()
                     else None)
        out.append({
            "scenario": names[s], "desc": descs[s],
            "avoided_horizon_cm": float(fwd[0, -1] - fwd[s, -1]),
            "avoided_first_year_cm": (float(fwd[0, first_dec] - fwd[s, first_dec])
                                      if first_dec is not None else float("nan")),
            "baseline_forward_cm": float(fwd[0, -1]),
            "rate_last5_base_cm_yr": float(slope_per_year(fan[[0]][:, last5], t[last5])[0]),
            "rate_last5_policy_cm_yr": float(slope_per_year(fan[[s]][:, last5], t[last5])[0]),
        })
    return out


def column_scores(column_dir: str) -> dict:
    """The column's own scores from its ``calibrate_coupled`` directory."""
    out: dict = {"column_dir": column_dir}
    vep = sorted(glob.glob(os.path.join(column_dir, "vep_*.json")))
    config = None
    if vep:
        with open(vep[0], encoding="utf-8") as fh:
            p = json.load(fh)
        config = p.get("config")
        out.update(vep_json=vep[0], config=config,
                   n_columns=len(p.get("banded") or p.get("zonal") or [p]),
                   band_lambda=p.get("band_lambda"), band_kind=p.get("band_kind"))
    csv = os.path.join(column_dir, "stage4_column.csv")
    if os.path.exists(csv):
        cc = pd.read_csv(csv)
        r = cc[cc.config == config] if config in set(cc.config) else cc.iloc[[0]]
        out.update(leveling_r2_outoffold=float(r.r2_outoffold.iloc[0]),
                   leveling_r2_insample=float(r.r2_insample.iloc[0]),
                   rings_r2=float(r.rings_independent_r2.iloc[0]))
    sc = os.path.join(column_dir, "scorecard.json")
    if os.path.exists(sc):
        with open(sc, encoding="utf-8") as fh:
            pr = json.load(fh).get("policy_response") or {}
        out["policy_gate"] = {k: pr.get(k) for k in ("verdict", "column", "vep_json",
                                                     "ds_m", "dh2_m", "rel_ds")}
    return out


def score_run(label: str, forward_npz: str, column_dir: str, ddir: str | None,
              line_km: float = LINE_KM, windows=WINDOWS_KM, lev: pd.DataFrame | None = None,
              log=print) -> tuple[dict, list[dict]]:
    fw = np.load(forward_npz, allow_pickle=False)
    cent = cell_centroids(fw)
    x_km, y_km = cent[:, 0] / 1000.0, cent[:, 1] / 1000.0
    subs = _first_column_subs(fw)                                     # (S, A, T) cm
    dates = [str(v)[:10] for v in fw["dates"]]
    origin = int(fw["origin"])
    t = _years(dates)
    base = subs[0]
    rate_h = slope_per_year(base[:, :origin + 1], t[:origin + 1])
    rate_f = slope_per_year(base[:, origin:], t[origin:])
    fwd = base[:, -1] - base[:, origin]
    res = {"label": label, "forward_npz": forward_npz, **column_scores(column_dir),
           "line_km": float(line_km), "hindcast": f"{dates[0]}..{dates[origin]}",
           "forecast": f"{dates[origin]}..{dates[-1]}",
           "first_column_forward_cm": first_column_contrast(fwd, x_km, line_km),
           "first_column_hindcast_rate": first_column_contrast(rate_h, x_km, line_km),
           "first_column_forecast_rate": first_column_contrast(rate_f, x_km, line_km)}
    opt = {}
    if "forward_options" in fw.files:
        opt = json.loads(str(fw["forward_options"]))
    res["forward_options"] = opt
    names = ([str(s) for s in fw["scenario_names"]] if "scenario_names" in fw.files
             else [str(s).split(":")[0] for s in fw["scenarios"]])
    res["policy"] = policy_summary(subs, dates, origin, names,
                                   [str(s) for s in fw["scenarios"]])
    if ddir is not None:
        from .explorer3d import validate_against_leveling

        sk = validate_against_leveling(ddir, _Grid(cent, float(fw["dx"])),
                                       base[:, :origin + 1] / 100.0,
                                       pd.to_datetime(dates[:origin + 1]))
        res["forward_hindcast_leveling"] = {k: sk.get(k) for k in ("r2", "rmse_cm", "bias_cm",
                                                                   "n_sites", "n_pairs")}
    lev_cells = None
    if lev is not None and len(lev):
        d2 = ((cent[None, :, :] - lev[["x", "y"]].to_numpy()[:, None, :]) ** 2).sum(-1)
        cell = np.argmin(d2, axis=1)
        on = np.sqrt(d2[np.arange(len(lev)), cell]) <= float(fw["dx"])
        lev_cells = (lev[on].reset_index(drop=True), cell[on])
    rows = []
    for w in windows:
        row = {"label": label, "window_km": float(w)}
        for name, r in (("hindcast", rate_h), ("forecast", rate_f)):
            f = step_fit(x_km, y_km, r, line_km, w)
            row.update({f"{name}_step_cm_yr": f["step"], f"{name}_se": f["se"],
                        f"{name}_n": f["n"]})
        if lev_cells is not None:
            lv, cell = lev_cells
            lx, ly = lv.x.to_numpy() / 1000.0, lv.y.to_numpy() / 1000.0
            fo = step_fit(lx, ly, lv.rate_cm_yr.to_numpy(), line_km, w)
            fm = step_fit(lx, ly, rate_h[cell], line_km, w)
            row.update({"leveling_step_cm_yr": fo["step"], "leveling_se": fo["se"],
                        "leveling_n": fo["n"], "model_at_benchmarks_step_cm_yr": fm["step"],
                        "model_at_benchmarks_se": fm["se"]})
        rows.append(row)
        log(f"[{label}] +-{w:g} km: hindcast step {row['hindcast_step_cm_yr']:+.2f} "
            f"+- {row['hindcast_se']:.2f}, forecast step {row['forecast_step_cm_yr']:+.2f} "
            f"+- {row['forecast_se']:.2f} cm/yr"
            + (f"; leveling {row['leveling_step_cm_yr']:+.2f} +- {row['leveling_se']:.2f}, "
               f"model at benchmarks {row['model_at_benchmarks_step_cm_yr']:+.2f}"
               if "leveling_step_cm_yr" in row else ""))
    fc = res["first_column_forward_cm"]
    log(f"[{label}] first column across {line_km:g} km, forward change: east "
        f"{fc['east']:.1f} cm, west {fc['west']:.1f} cm; leveling out of fold "
        f"{res.get('leveling_r2_outoffold', float('nan')):+.3f}, rings "
        f"{res.get('rings_r2', float('nan')):+.3f}")
    for p in res["policy"]:
        log(f"[{label}] {p['scenario']}: avoids {p['avoided_horizon_cm']:.2f} cm by the "
            f"horizon ({p['avoided_first_year_cm']:.2f} by the first December); last-5-yr "
            f"rate {p['rate_last5_base_cm_yr']:.2f} -> {p['rate_last5_policy_cm_yr']:.2f} "
            "cm/yr")
    res["steps"] = rows
    return res, rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="score columns on the 182 km step (CPU)")
    ap.add_argument("--run", nargs=3, action="append", required=True,
                    metavar=("LABEL", "FORWARD_NPZ", "COLUMN_DIR"))
    ap.add_argument("--line-km", type=float, default=LINE_KM)
    ap.add_argument("--windows", default=",".join(f"{w:g}" for w in WINDOWS_KM))
    ap.add_argument("--data", default=None, help="data dir with the leveling panel (else "
                                                 "HYDROMIND_GW_DATA); without it the "
                                                 "leveling step and hindcast R2 are skipped")
    ap.add_argument("--out", required=True, help="output stem: <out>.json, <out>.csv")
    args = ap.parse_args(argv)

    ddir = args.data or os.environ.get("HYDROMIND_GW_DATA")
    if ddir and not os.path.isdir(ddir):
        ddir = None
    windows = [float(w) for w in args.windows.split(",") if w.strip()]
    lev = None
    if ddir:
        fw0 = np.load(args.run[0][1], allow_pickle=False)
        dates = [str(v)[:10] for v in fw0["dates"]]
        t1 = dates[int(fw0["origin"])]
        lev = leveling_rates(ddir, dates[0], t1)
        print(f"leveling rates: {len(lev)} benchmarks over {dates[0]}..{t1}", flush=True)
    results, table = [], []
    for label, npz, cdir in args.run:
        r, rows = score_run(label, npz, cdir, ddir, args.line_km, windows, lev,
                            log=lambda m: print(m, flush=True))
        results.append(r)
        table.extend(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"line_km": args.line_km, "windows_km": windows, "runs": results}, fh,
                  indent=1, default=str)
    pd.DataFrame(table).to_csv(args.out + ".csv", index=False)
    print(f"wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
