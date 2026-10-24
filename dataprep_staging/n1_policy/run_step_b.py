"""Step B of N1.policy_response.v1: pre-2012 leveling dose-response vs the twin's own.

Observed: at the benchmarks of ``N1P:lev_pre2012``, the subsidence rate of each survey
interval of 0.5-1.5 yr inside 2007-01..2010-12 (site_subsidence, max_rate 0.5 m/yr) regressed
on the annualised pump energy within 3 km (E3) and in the 3-10 km ring (E10) of the
benchmark's cell (pump_electricity/preorigin/v3 value_all), with benchmark and campaign
(year of the interval's end) fixed effects, per county. Placebo: the same regression plus the
next 12 months' E3 (lead); its interval must include 0.

Model: the county's model (Yunlin: R0 = stage3_datum_gate_cwa with its leveling column;
Changhua: R1c = stage3_datum_gate_ccz with the batch-24 three-zone column) run with S random
pumping pulses: every 10 km block's energy times 0.8 or 1.2, drawn per water year (June to
May, 2012-06..2021-05). At the same benchmarks' cells, the May-to-May model rates 2013-2021
of each pulse run minus the base run, regressed on the matching E3 and E10 differences:
beta_model = the E3 coefficient (the model's own local derivative, by construction free of
confounding).

Verdict per county (policy_lib.ratio_verdict): 'consistent' if beta_model / [obs 90 % CI]
lies in [1/1.5, 1.5], 'inconsistent' if it excludes 1, 'inconclusive' otherwise,
'not_informative' if the observed interval includes 0, 'not_evaluable' if the lead placebo's
interval excludes 0.

``--dry`` runs every step on synthetic rates (no elevation read, no look, nothing written),
with S = 2 pulse runs. Outputs: results/dataprep/reports/N1_policy/step_b/.

Usage: CUDA_VISIBLE_DEVICES="" results/twin_runs/par24/py.sh -I run_step_b.py [--dry]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # leave cores to the GPU jobs

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402

PREREG = "N1.policy_response.v1"
TESTS = {"yunlin": "N1P.B_dose_yunlin", "changhua": "N1P.B_dose_changhua"}
SEED = 20261023
S_PULSES = 16
PULSE = (0.8, 1.2)
PULSE_BLOCK_M = 10000.0
R3, R10 = 3000.0, 10000.0
UNITS = REPORTS_DIR / "N1_policy" / "units_policy.json"
C0 = REPO / "results/dataprep/pump_electricity/preorigin/v3/cells_total.npz"
OUT = REPORTS_DIR / "N1_policy" / "step_b"
MODELS = {
    "yunlin": ("results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json",
               "results/twin_runs/stage3_datum_gate_cwa/coupled_leveling/vep_zonal_leveling.json"),
    "changhua": ("results/twin_runs/stage3_datum_gate_ccz/stage3_theta.json",
                 "results/twin_runs/stage3_datum_gate_ccz/coupled_leveling_z3/"
                 "vep_zonal_leveling.json"),
}
SCALE = 1e8                           # m/yr per kWh/yr -> cm/yr per GWh/yr
UNIT_BAND = (0.8, 1.25)               # model energy / C0 energy over 2012-2021, per county


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


# ----------------------------------------------------------------------------- geometry
def grid_and_county():
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid

    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
    flags, _ = county_flags_for_grid(grid, DEFAULT_MAP)
    return grid, np.asarray(flags, dtype=bool)


def c0_energy(n_active: int):
    z = np.load(C0, allow_pickle=False)
    if int(z["n_active"]) != n_active:
        raise SystemExit(f"{C0}: {int(z['n_active'])} cells, the grid has {n_active}")
    dates = pd.PeriodIndex([str(d) for d in z["dates"]], freq="M")
    return np.nan_to_num(z["value_all"]), dates


# ----------------------------------------------------------------------------- observed
def observed_intervals(ids: list[str], dry: bool, rng) -> pd.DataFrame:
    """(sid, t0, t1, rate m/yr, x, y) of every 0.5-1.5 yr interval inside 2007-2010."""
    from hydrophysics.config import Config
    from hydrophysics.twin.leveling import load_panel, site_subsidence, site_xy

    ddir = str(Config().data_dir)
    if dry:          # dates and coordinates only; synthetic values
        meta = pd.read_parquet(os.path.join(ddir, "ls_cache",
                                            "ls-wra-lsp-obs__choushui_panel.parquet"),
                               columns=["sid", "datetime", "x_3826", "y_3826"])
        meta["elev_m"] = -0.03 * (pd.to_datetime(meta.datetime).dt.year - 2007) \
            + rng.normal(0, 0.005, len(meta))
        panel = meta.rename(columns={"x_3826": "x", "y_3826": "y"})
        panel["datetime"] = pd.to_datetime(panel.datetime)
        panel["sid"] = panel.sid.astype(str)
    else:
        panel = load_panel(ddir)
        panel["sid"] = panel.sid.astype(str)
    panel = panel[panel.sid.isin(set(ids))]
    sub = site_subsidence(panel, "2007-01-01", "2011-01-01", min_obs=3, max_rate=0.5)
    xy = site_xy(panel)
    rows = []
    for sid, s in sub.items():
        t, v = list(s.index), s.to_numpy()
        for a in range(len(t) - 1):
            yrs = (t[a + 1] - t[a]).days / 365.25
            if 0.5 <= yrs <= 1.5:
                rows.append({"sid": sid, "t0": t[a], "t1": t[a + 1], "years": yrs,
                             "rate": (v[a + 1] - v[a]) / yrs,
                             "x": xy[sid][0], "y": xy[sid][1]})
    return pd.DataFrame(rows)


def observed_beta(df: pd.DataFrame, seed: int) -> dict:
    X = df[["E3", "E10"]].to_numpy()
    Xp = df[["E3", "E10", "E3_lead"]].to_numpy()
    fe = [df.sid.to_numpy(), df.campaign.to_numpy()]
    y = df.rate.to_numpy()
    cl = PL.blocks_of(df[["x", "y"]].to_numpy())

    def stat(rows):
        return PL.fe_ols(y[rows], X[rows], [f[rows] for f in fe])

    def stat_p(rows):
        return PL.fe_ols(y[rows], Xp[rows], [f[rows] for f in fe])

    full, ci = PL.cluster_bootstrap(stat, cl, seed=seed)
    fp, cip = PL.cluster_bootstrap(stat_p, cl, seed=seed + 1)
    n_cl = len(np.unique(cl))
    return {"n_intervals": int(len(df)), "n_benchmarks": int(df.sid.nunique()),
            "n_clusters": int(n_cl), "campaigns": sorted(int(c) for c in df.campaign.unique()),
            "beta_E3": float(full[0]), "ci_E3": [float(ci[0][0]), float(ci[1][0])],
            "beta_E10": float(full[1]), "ci_E10": [float(ci[0][1]), float(ci[1][1])],
            "placebo_lead_E3": float(fp[2]), "placebo_ci": [float(cip[0][2]),
                                                             float(cip[1][2])],
            "enough_clusters": bool(n_cl >= PL.MIN_CLUSTERS)}


# ----------------------------------------------------------------------------- model
def pulse_factors(cent: np.ndarray, n_steps: int, rng) -> np.ndarray:
    """(A, n_steps) factor field: per 10 km block and water year (June-May), 0.8 or 1.2."""
    blk = PL.blocks_of(cent, PULSE_BLOCK_M)
    ub, inv = np.unique(blk, return_inverse=True)
    step_month = PL.MONTHS[1:n_steps + 1]
    wy = np.array([p.year + (1 if p.month >= 6 else 0) for p in step_month])
    uw, winv = np.unique(wy, return_inverse=True)
    f = rng.choice(PULSE, size=(len(ub), len(uw)))
    return f[inv][:, winv]


def model_beta(county: str, ids_xy: pd.DataFrame, on_county: np.ndarray, S: int,
               seed: int, c0: np.ndarray, c0_dates: pd.PeriodIndex) -> dict:
    theta, col = MODELS[county]
    tm = PL.TwinModel(str(REPO / theta), str(REPO / col))
    cells = tm.cell_of(ids_xy[["x", "y"]].to_numpy())
    keep = (cells >= 0) & on_county[np.clip(cells, 0, None)]
    cells = cells[keep]
    E = tm.energy()
    E_cell = E.sum(axis=0) if E.ndim == 3 else E                         # (A, steps)
    # unit check (not a score): the model's energy and C0's over the same county cells and
    # months 2012-02..2021-12 must agree within UNIT_BAND, or the two betas differ in units
    k0 = list(c0_dates).index(pd.Period("2012-02", "M"))
    n = list(PL.MONTHS).index(pd.Period("2021-12", "M"))           # steps 0..n-1
    unit = float(E_cell[on_county, :n].sum() / c0[on_county, k0:k0 + n].sum())
    if not UNIT_BAND[0] <= unit <= UNIT_BAND[1]:
        raise SystemExit(f"[{county}] model/C0 energy ratio {unit:.3f} outside {UNIT_BAND}")
    # month index of each May 2012..2021 on the heads axis; the energy of step j is
    # month j + 1, so the months (May y-1, May y] are steps may[y-1] .. may[y] - 1
    may = np.array([k for k, p in enumerate(PL.MONTHS) if p.month == 5])
    W3 = PL.ring_weights(tm.cent, cells, 0.0, R3)
    W10 = PL.ring_weights(tm.cent, cells, R3, R10)

    def panel(E_run, S_run):
        r, e3, e10 = [], [], []
        for a, b in zip(may[:-1], may[1:], strict=True):
            r.append(S_run[cells, b] - S_run[cells, a])
            tot = E_run[:, a:b].sum(axis=1)                               # kWh over the year
            e3.append(W3 @ tot)
            e10.append(W10 @ tot)
        return np.stack(r, 1), np.stack(e3, 1), np.stack(e10, 1)

    log(f"[{county}] base run")
    S0 = tm.subsidence(tm.heads())
    r0, e30, e100 = panel(E_cell, S0)
    rng = np.random.default_rng(seed)
    dr, d3, d10, draw = [], [], [], []
    for s in range(S):
        F = pulse_factors(tm.cent, E_cell.shape[1], rng)
        Es = E * F if E.ndim == 2 else E * F[None]
        Ss = tm.subsidence(tm.heads(Es))
        r, e3, e10 = panel(Es.sum(axis=0) if Es.ndim == 3 else Es, Ss)
        dr.append(r - r0)
        d3.append(e3 - e30)
        d10.append(e10 - e100)
        draw.append(np.full(r.shape, s))
        log(f"[{county}] pulse run {s + 1}/{S}")
    y = np.concatenate([a.ravel() for a in dr])
    X = np.column_stack([np.concatenate([a.ravel() for a in d3]),
                         np.concatenate([a.ravel() for a in d10]),
                         np.ones_like(y)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    dr_ = np.concatenate([a.ravel() for a in draw])
    _, ci = PL.cluster_bootstrap(
        lambda rows: np.linalg.lstsq(X[rows], y[rows], rcond=None)[0][:2], dr_, seed=seed + 7,
        B=200) if S >= 2 else (None, [[np.nan] * 2, [np.nan] * 2])
    # secondary (reported): the observed regression's design on the base hindcast
    sid = np.repeat(np.arange(len(cells))[:, None], r0.shape[1], 1).ravel()
    yr = np.repeat(np.arange(r0.shape[1])[None, :], len(cells), 0).ravel()
    hb = PL.fe_ols(r0.ravel(), np.column_stack([e30.ravel(), e100.ravel()]), [sid, yr])
    return {"model": {"theta": theta, "column": col}, "n_cells": int(len(cells)),
            "unit_ratio_model_over_c0": unit,
            "pulses": int(S), "beta_E3": float(beta[0]), "beta_E10": float(beta[1]),
            "ci_draws_E3": [float(ci[0][0]), float(ci[1][0])],
            "hindcast_fe_beta_E3": float(hb[0])}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    if not a.dry:
        if (OUT / "results.json").exists():
            raise SystemExit(f"{OUT / 'results.json'} exists; results are never recomputed")
        for t in TESTS.values():
            r = prereg3.assert_can_score(PREREG, "N1", test_id=t)
            log(f"assert_can_score {t}: ok (post-registration looks "
                f"{len(r['post_registration_looks'])})")
    rng = np.random.default_rng(SEED)
    units = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    ids = units["N1P:lev_pre2012"]["ids"]
    grid, changhua = grid_and_county()
    cent = grid.centroids()
    E, dates = c0_energy(len(cent))
    df = observed_intervals(ids, a.dry, rng)
    df["cell"] = PL.nearest_cell(cent, df[["x", "y"]].to_numpy(), grid.dx)
    df = df[df.cell >= 0].reset_index(drop=True)
    pos = {p: k for k, p in enumerate(dates)}
    m0 = np.array([pos[pd.Period(t, "M")] for t in df.t0])
    m1 = np.array([pos[pd.Period(t, "M")] for t in df.t1])
    W3 = PL.ring_weights(cent, df.cell.to_numpy(), 0.0, R3)
    W10 = PL.ring_weights(cent, df.cell.to_numpy(), R3, R10)
    df["E3"] = PL.interval_exposure(E, W3, m0, m1, df.years.to_numpy())
    df["E10"] = PL.interval_exposure(E, W10, m0, m1, df.years.to_numpy())
    df["E3_lead"] = PL.interval_exposure(E, W3, m1, np.minimum(m1 + 12, len(dates) - 1),
                                         np.ones(len(df)))
    df["campaign"] = [t.year for t in df.t1]
    df["changhua"] = changhua[df.cell.to_numpy()]
    log(f"observed intervals: {len(df)} at {df.sid.nunique()} benchmarks")
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "prereg": PREREG, "dry": bool(a.dry), "units": rel_to_repo(UNITS),
           "scale_note": "betas in m/yr per kWh/yr; *_cm_per_GWh = beta * 1e8", "tests": {}}
    for k, (county, tid) in enumerate(TESTS.items()):
        sel = df[df.changhua == (county == "changhua")]
        if a.dry:
            sel = sel.assign(rate=2e-9 * sel.E3 + rng.normal(0, 0.005, len(sel)))
        ob = observed_beta(sel, SEED + 10 * k)
        on = changhua if county == "changhua" else ~changhua
        mb = model_beta(county, sel.drop_duplicates("sid"), on, 2 if a.dry else S_PULSES,
                        SEED + 100 + k, E, dates)
        if not ob["enough_clusters"]:
            v = {"verdict": "not_evaluable", "reason": "fewer than 5 clusters"}
        elif ob["placebo_ci"][0] > 0 or ob["placebo_ci"][1] < 0:
            v = {"verdict": "not_evaluable", "reason": "lead placebo interval excludes 0"}
        else:
            v = PL.ratio_verdict(mb["beta_E3"], ob["ci_E3"])
        res["tests"][tid] = {"county": county, "observed": ob, "model": mb, **v,
                             "beta_obs_cm_per_GWh": ob["beta_E3"] * SCALE,
                             "beta_model_cm_per_GWh": mb["beta_E3"] * SCALE}
        log(f"{tid}: {v['verdict']}")
    if a.dry:
        print(json.dumps(res, indent=1, default=str)[:3000])
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    df.to_csv(OUT / "intervals.csv", index=False)
    log(f"wrote {rel_to_repo(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
