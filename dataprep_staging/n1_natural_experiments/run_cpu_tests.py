"""Run the registered CPU tests of ``N1.cpu_response.v1`` and the checks C1/C2 of
``N1.designtime_checks.v1`` (after their registration; scoring looks logged first).

1. ``harness.prereg3.assert_can_score`` for every registered test (refuses a registration
   that postdates a look at its data, or changed inputs; logs the scoring look), and
   ``record_look`` for the consistency and design-time checks, BEFORE any head is read.
2. The deliverable's inputs (``inputs_p4.load_inputs``), the batch-8 merged cwa theta and the
   engine (``SparseTwin``); rollouts M, ZR (one per event), M+PW (primary and three
   sensitivities); the same M/ZR for each fold theta (C2). M is checked against engine/v2's
   stored hindcast.
3. Observed heads from ``heads/L1_clean/v3`` (guard, ``scoring_target``; c20 months,
   2013-01..2022-12), and the n1lib scores.

Outputs go to ``results/dataprep/reports/N1_evidence/cpu_tests/`` (results.json, per-well
tables); the product build copies them. CPU only (CUDA hidden), inside a CPU slot.

Usage: CUDA_VISIBLE_DEVICES="" cpu_slot.sh python -I run_cpu_tests.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import n1lib as L  # noqa: E402
from n1_natural_experiments import units_n1 as U  # noqa: E402

OUT = K.EVID / "cpu_tests"
PREREG_T = "N1.cpu_response.v1"
PREREG_D = "N1.designtime_checks.v1"
T_IDS = [f"N1.T1_ext_{e}" for e in K.EVENTS] + ["N1.T2_pw_pulse2021", "N1.T3_pw_step2019"]
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    LOG.append(line)
    print(line, flush=True)


# ------------------------------------------------------------------------- calendar
MONTHS = pd.period_range("2012-01", "2022-12", freq="M")          # H[..., k] <-> MONTHS[k]
T = np.arange(len(MONTHS), dtype=float)
YEAR = np.array([p.year for p in MONTHS])
MON = np.array([p.month for p in MONTHS])
REF = np.isin(YEAR, K.REFERENCE_YEARS)


def event_mask(ev: str) -> np.ndarray:
    e = K.EVENTS[ev]
    return (YEAR == e["year"]) & np.isin(MON, e["months"])


# ------------------------------------------------------------------------- model runs
def load_model():
    import torch

    from hydrophysics.twin.forward import build_model, load_members
    from p4_changhua import engine, inputs_p4

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing (no GPU in Phase 0)")
    os.chdir(REPO)
    member = load_members([str(K.TWIN_THETA)])[0]
    folds = load_members([str(K.TWIN_THETA), str(K.TWIN_FOLD_THETAS)])[1:]
    inp = inputs_p4.load_inputs(member.meta)
    E_cls = np.stack([inp.E_by_class[c] for c in inp.E_by_class])
    E = E_cls.sum(axis=0)[:, 1:]
    R = inp.recharge_field.numpy()[:, 1:]
    ge = inp.ground_elev.numpy()
    h0 = inp.initial_heads(0, n_layers=4).numpy()
    model, sc, _ = build_model(inp.grid, member, "cpu")
    tw = engine.SparseTwin(model, sc, ge, pump_layer=int(member.meta.get("pump_layer", 1)),
                           recharge_layer=int(member.meta.get("recharge_layer", 0)))
    return {"inp": inp, "member": member, "folds": folds, "E": E, "R": R, "ge": ge, "h0": h0,
            "tw": tw, "build_model": build_model, "engine": engine}


def zr_energy(E: np.ndarray, ev: str) -> np.ndarray:
    """E with the event months replaced cell by cell by the reference-year mean of the same
    calendar month. E[:, j] is month MONTHS[j + 1] (step j of the rollout)."""
    Z = E.copy()
    em = event_mask(ev)[1:]
    ref = REF[1:]
    mon = MON[1:]
    for m in K.EVENTS[ev]["months"]:
        Z[:, em & (mon == m)] = E[:, ref & (mon == m)].mean(axis=1, keepdims=True)
    return Z


def pw_member(series: pd.DataFrame, dwells: pd.DataFrame, n_layers: int, A: int, *,
              scale: float, janmay_only: bool, layers: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    """Direct sources q (L, A, T) m3/day (abstraction negative) of the Yunlin Taiwan Water
    public-well member, per the registered rule."""
    s = series[(series.county == "Yunlin") & series.agency.str.contains("台水")
               & (series.township == "(county)")].set_index("year").value_1e4m3
    tw = dwells[(dwells.county == "Yunlin") & dwells.agency.str.contains("台水")]
    n_listed = len(tw)
    cells = tw.cell.to_numpy()
    cells = cells[cells >= 0]
    q = np.zeros((n_layers, A, len(MONTHS) - 1))
    for j in range(len(MONTHS) - 1):
        p = MONTHS[j + 1]
        if janmay_only and p.month > 5:
            continue
        v = float(s.loc[p.year]) * 1e4                                   # m3 in Jan-May
        days = sum(pd.Period(f"{p.year}-{m:02d}", "M").days_in_month for m in range(1, 6))
        rate_per_well = v / days / n_listed / scale                      # m3/day per well
        for c in cells:
            for lay in layers:
                q[lay, c, j] -= rate_per_well / len(layers)
    info = {"n_listed": int(n_listed), "n_in_grid": int(len(cells)),
            "share_in_grid": float(len(cells) / n_listed), "scale_divisor": scale,
            "janmay_only": janmay_only, "layers": list(layers),
            "in_grid_rate_m3d_2021": float(-q[:, :, [j for j in range(len(MONTHS) - 1)
                                                      if MONTHS[j + 1] == pd.Period("2021-03", "M")][0]].sum())}
    return q, info


def heads_at(H: np.ndarray, cells: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """(W, months) simulated heads of wells; H (L, A, T+1) with H[..., k] = MONTHS[k]."""
    return H[layers, cells, :]


# ------------------------------------------------------------------------- observations
def observed(sids: list[str]) -> np.ndarray:
    p = guard.open_product(K.HEADS_DIR, "scoring_target")
    df = p.read_table("heads_monthly.parquet", columns=["sid", "month", "value", "flag", "c20"])
    df = df[df.sid.isin(set(sids)) & df.c20.astype(bool) & df.flag.isin([0, 1])]
    df = df[(df.month >= "2013-01") & (df.month <= "2022-12")]
    pos = {str(p_): k for k, p_ in enumerate(MONTHS)}
    out = np.full((len(sids), len(MONTHS)), np.nan)
    row = {s: i for i, s in enumerate(sids)}
    for s, m, v in zip(df.sid, df.month, df.value, strict=True):
        out[row[s], pos[m]] = v
    return out


# ------------------------------------------------------------------------- scoring
def eligible(obs: np.ndarray, ev_mask: np.ndarray) -> np.ndarray:
    n_ev = np.isfinite(obs[:, ev_mask]).sum(1)
    n_ref = np.isfinite(obs[:, REF]).sum(1)
    yrs = np.array([len({YEAR[k] for k in np.flatnonzero(REF & np.isfinite(o))}) for o in obs])
    return (n_ev >= K.MIN_EVENT_MONTHS) & (n_ref >= K.MIN_REF_MONTHS) & (yrs >= K.MIN_REF_YEARS)


def event_table(obs, series: dict, wells: pd.DataFrame, ev: str) -> pd.DataFrame:
    """Per-well errors and mean anomalies for one event (wells eligible only)."""
    em = event_mask(ev)
    ok = eligible(obs, em)
    rows = []
    for i in np.flatnonzero(ok):
        a = L.anomalies(T, MON, obs[i], {k: v[i] for k, v in series.items()}, REF, em,
                        K.SEASONAL_HARMONICS)
        if a is None or len(a["obs"]) == 0:
            continue
        r = {"sid": wells.sid.iloc[i], "site": wells.site.iloc[i],
             "county": wells.county.iloc[i], "layer": int(wells.layer.iloc[i]),
             "n_event_months": int(len(a["obs"])), "mean_obs": float(np.mean(a["obs"])),
             "e_CL": L.per_well_error(np.zeros_like(a["obs"]), a["obs"])}
        for k in series:
            r[f"e_{k}"] = L.per_well_error(a[k], a["obs"])
            r[f"mean_{k}"] = float(np.mean(a[k]))
            r[f"rms_{k}_minus_M"] = (float(np.sqrt(np.mean((a[k] - a["M"]) ** 2)))
                                     if "M" in a and k != "M" else 0.0)
        for c in ("site_disjoint", "layer_flag", "dist_pw_km"):
            if c in wells:
                r[c] = wells[c].iloc[i]
        rows.append(r)
    return pd.DataFrame(rows)


def compare(tab: pd.DataFrame, model: str, base: str, seed: int) -> dict:
    if tab.empty:
        return {"n_wells": 0, "n_sites": 0, "verdict": "not_evaluable"}
    return L.paired_site_bootstrap(tab[f"e_{model}"].to_numpy(), tab[f"e_{base}"].to_numpy(),
                                   tab.site.to_numpy(), B=K.BOOT_B, seed=seed,
                                   level=K.CI_LEVEL, min_sites=K.MIN_SITES)


def informative(tab: pd.DataFrame, k: str) -> float:
    """RMS over well-months of (a_k - a_M) (well-month weighted)."""
    if tab.empty:
        return float("nan")
    w = tab.n_event_months.to_numpy()
    return float(np.sqrt(np.sum(w * tab[f"rms_{k}_minus_M"].to_numpy() ** 2) / np.sum(w)))


def slope(tab: pd.DataFrame, k: str, seed: int) -> dict:
    if len(tab) < 2:
        return {"n": int(len(tab))}
    return L.response_slope(tab.mean_obs.to_numpy(), tab[f"mean_{k}"].to_numpy(),
                            tab.site.to_numpy(), B=K.BOOT_B, seed=seed, level=K.CI_LEVEL)


def step_table(obs, series: dict, wells: pd.DataFrame) -> pd.DataFrame:
    pre = np.isin(YEAR, [2017, 2018]) & (MON >= 6)
    post = np.isin(YEAR, [2019, 2020]) & (MON >= 6)
    ref_pre = np.isin(YEAR, [2013, 2014, 2016, 2017, 2018])
    rows = []
    for i in range(len(obs)):
        o = obs[i]
        d_obs = L.step_delta(T, o, pre, post, 8)
        if not np.isfinite(d_obs):
            continue
        f = L.fit_reference(T, MON, o, ref_pre, K.SEASONAL_HARMONICS)
        n_ref = int((ref_pre & np.isfinite(o)).sum())
        n_yrs = len({YEAR[k] for k in np.flatnonzero(ref_pre & np.isfinite(o))})
        mask = np.where(np.isfinite(o), 1.0, np.nan)
        r = {"sid": wells.sid.iloc[i], "site": wells.site.iloc[i],
             "county": wells.county.iloc[i], "layer": int(wells.layer.iloc[i]),
             "delta_obs": d_obs, "e_CL0": abs(d_obs)}
        if f is not None and n_ref >= K.MIN_REF_MONTHS and n_yrs >= K.MIN_REF_YEARS:
            r["delta_CLtrend"] = L.trend_delta(f, T, pre, post, o)
            r["e_CLtrend"] = abs(r["delta_CLtrend"] - d_obs)
        else:
            r["delta_CLtrend"], r["e_CLtrend"] = np.nan, np.nan
        for k, v in series.items():
            dk = L.step_delta(T, v[i] * mask, pre, post, 8)
            r[f"delta_{k}"] = dk
            r[f"e_{k}"] = abs(dk - d_obs)
        for c in ("site_disjoint", "layer_flag", "dist_pw_km"):
            if c in wells:
                r[c] = wells[c].iloc[i]
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- main
def main() -> int:
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit(f"{OUT / 'results.json'} exists; the registered tests were run "
                         "already (results are never recomputed in place)")
    # 1. looks before any head is read
    scored = {}
    for tid in T_IDS:
        r = prereg3.assert_can_score(PREREG_T, K.WP, test_id=tid)
        scored[tid] = {"sha256": r["sha256"], "inputs_verified": r["inputs_verified"],
                       "post_registration_looks": len(r["post_registration_looks"]),
                       "unverifiable_prior_looks": len(r["unverifiable_prior_looks"])}
        log(f"assert_can_score {tid}: ok ({scored[tid]})")
    for purpose, what in (("consistency_check", "C1 in-sample consistency (full theta)"),
                          ("design_time_check", "C2 fold-theta check (batch-8 folds)")):
        prereg3.record_look(["N1:cal_wells"], "heads:2013-01..2022-12", K.WP, purpose,
                            prereg_id=PREREG_D, note=what)
        log(f"record_look N1:cal_wells {purpose}")

    # 2. model runs
    t0 = time.perf_counter()
    mdl = load_model()
    tw, E, R, h0 = mdl["tw"], mdl["E"], mdl["R"], mdl["h0"]
    L_, A = tw.n_layers, tw.n_active
    log(f"inputs and model loaded in {time.perf_counter() - t0:.1f}s: A={A}, steps={E.shape[1]}")
    H = {"M": tw.rollout(h0, E=E, R=R)}
    stored = np.load(K.ENGINE_DIR / "engine_arrays.npz")["H_hindcast"]
    repro = float(np.max(np.abs(H["M"] - stored)))
    log(f"M vs engine/v2 stored hindcast: max |dh| {repro:.3e} m")
    for ev in K.EVENTS:
        H[f"ZR_{ev}"] = tw.rollout(h0, E=zr_energy(E, ev), R=R)
    series_pw = pd.read_csv(K.EVID / "public_well_janmay_series.csv")
    dwells = pd.read_csv(K.EVID / "drought_wells_yunlin_changhua.csv")
    pw_info = {}
    variants = {"PW": dict(scale=K.PW_WRA_TO_MODEL, janmay_only=False, layers=(2, 3)),
                "PWx1": dict(scale=1.0, janmay_only=False, layers=(2, 3)),
                "PWjm": dict(scale=K.PW_WRA_TO_MODEL, janmay_only=True, layers=(2, 3)),
                "PWl2": dict(scale=K.PW_WRA_TO_MODEL, janmay_only=False, layers=(2,))}
    for k, v in variants.items():
        q, info = pw_member(series_pw, dwells, L_, A, **v)
        H[k] = tw.rollout(h0, E=E, R=R, q_extra=q)
        pw_info[k] = info
    log(f"rollouts done: {sorted(H)}; member {pw_info['PW']}")

    # wells
    ext = U.ext_wells()
    ext["site"] = ext.site_id
    cal = U.calibration_wells()
    meta = U.station_meta().set_index("sid")
    cal["cell"] = U.cell_index(cal.x.to_numpy(), cal.y.to_numpy())
    cal["layer"] = cal.layer.astype(int) - 1
    cal["site"] = cal.sid.str[:6]
    cal["county"] = cal.sid.map(meta.CountyName).map({"彰化縣": "Changhua", "雲林縣": "Yunlin",
                                                       "南投縣": "Nantou"})
    tw_cells = dwells[(dwells.county == "Yunlin") & dwells.agency.str.contains("台水")
                      & (dwells.cell >= 0)][["x_twd97", "y_twd97"]].to_numpy()

    def dist_pw(df, xcol, ycol):
        xy = df[[xcol, ycol]].to_numpy()
        return np.sqrt(((xy[:, None, :] - tw_cells[None]) ** 2).sum(-1)).min(1) / 1000.0

    ext["dist_pw_km"] = dist_pw(ext, "x_twd97", "y_twd97")
    cal["dist_pw_km"] = dist_pw(cal, "x", "y")
    # model check: the engine's layer convention equals the calibration's obs_layer
    inp = mdl["inp"]
    oi = {s: (int(c), int(lay)) for s, c, lay in zip(inp.sids, inp.obs_idx, inp.obs_layer,
                                                      strict=True)}
    cal_conv = all(oi[s] == (c, lay) for s, c, lay in zip(cal.sid, cal.cell, cal.layer,
                                                          strict=True))
    log(f"calibration-well cell/layer convention equals obs_idx/obs_layer: {cal_conv}")

    # 3. observations (the scoring looks were logged above)
    obs_ext = observed(list(ext.sid))
    obs_cal = observed(list(cal.sid))
    log(f"observations: ext {np.isfinite(obs_ext).sum()} well-months, cal "
        f"{np.isfinite(obs_cal).sum()}")

    def sim(wells, keys):
        return {k: heads_at(H[k], wells.cell.to_numpy(), wells.layer.to_numpy()) for k in keys}

    res: dict = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "prereg": {PREREG_T: scored, PREREG_D: "looks logged (record_look)"},
                 "engine_reproduction_max_abs_m": repro, "cal_layer_convention_ok": cal_conv,
                 "pw_member": pw_info, "tests": {}, "checks": {}}
    tables = {}
    # ---- T1
    prim = ext.site_disjoint & (ext.layer_flag == 0)
    for i, ev in enumerate(K.EVENTS):
        seed = K.BOOT_SEED + i
        s = sim(ext, ["M", f"ZR_{ev}"])
        s = {"M": s["M"], "ZR": s[f"ZR_{ev}"]}
        tab = event_table(obs_ext, s, ext, ev)
        tables[f"T1_{ev}_ext"] = tab
        p = tab[tab.sid.isin(set(ext.sid[prim]))]
        py = p[p.county == "Yunlin"]
        out = {"primary_wells": "site-disjoint, layer_flag 0",
               "M_vs_CL": compare(p, "M", "CL", seed),
               "M_vs_ZR_yunlin": compare(py, "M", "ZR", seed),
               "zr_informative_rms_m_yunlin": informative(py, "ZR"),
               "slope_M": slope(p, "M", seed),
               "secondary_all_ext": {"M_vs_CL": compare(tab, "M", "CL", seed),
                                     "M_vs_ZR_yunlin": compare(tab[tab.county == "Yunlin"],
                                                               "M", "ZR", seed),
                                     "slope_M": slope(tab, "M", seed)},
               "strata": {}}
        if out["zr_informative_rms_m_yunlin"] < K.ZR_INFORMATIVE_M:
            out["M_vs_ZR_yunlin"]["verdict_registered"] = "not informative"
        for lab, g in (("Changhua", tab[tab.county == "Changhua"]),
                       ("Yunlin", tab[tab.county == "Yunlin"]),
                       ("Nantou", tab[tab.county == "Nantou"]),
                       ("layers_0_1", tab[tab.layer <= 1]), ("layers_2_3", tab[tab.layer >= 2])):
            out["strata"][lab] = {"M_vs_CL": compare(g, "M", "CL", seed)}
        res["tests"][f"N1.T1_ext_{ev}"] = out
        log(f"T1 {ev}: primary M vs CL {out['M_vs_CL'].get('verdict')} "
            f"({out['M_vs_CL'].get('rmse_model')}, {out['M_vs_CL'].get('rmse_base')}); "
            f"M vs ZR yunlin {out['M_vs_ZR_yunlin'].get('verdict')}")
    # ---- T2 / T3
    deep_ext = ext[(ext.county == "Yunlin") & ext.layer.isin([2, 3])].reset_index(drop=True)
    deep_cal = cal[(cal.county == "Yunlin") & cal.layer.isin([2, 3])].reset_index(drop=True)
    o_de = observed(list(deep_ext.sid))
    o_dc = observed(list(deep_cal.sid))
    keys = ["M", "PW", "PWx1", "PWjm", "PWl2"]

    def strata_dist(tab, model, base, seed):
        out = {}
        for lab, lo, hi in (("<=3km", -1, 3.0), ("3-10km", 3.0, 10.0), (">10km", 10.0, 1e9)):
            g = tab[(tab.dist_pw_km > lo) & (tab.dist_pw_km <= hi)]
            out[lab] = compare(g, model, base, seed)
        return out

    for name, wells, o in (("primary_ext_deep", deep_ext, o_de),
                           ("secondary_cal_deep", deep_cal, o_dc)):
        seed = K.BOOT_SEED + 3
        tab = event_table(o, sim(wells, keys), wells, "E2021")
        tables[f"T2_{name}"] = tab
        r = {"PW_vs_M": compare(tab, "PW", "M", seed),
             "informative_rms_m": informative(tab, "PW"),
             "PW_vs_CL": compare(tab, "PW", "CL", seed), "M_vs_CL": compare(tab, "M", "CL", seed),
             "sensitivities_vs_M": {k: compare(tab, k, "M", seed) for k in keys[2:]},
             "sensitivity_informative_rms_m": {k: informative(tab, k) for k in keys[2:]},
             "strata_distance_PW_vs_M": strata_dist(tab, "PW", "M", seed),
             "slope_PW": slope(tab, "PW", seed), "slope_M": slope(tab, "M", seed)}
        if r["informative_rms_m"] < K.ZR_INFORMATIVE_M:
            r["PW_vs_M"]["verdict_registered"] = "not informative"
        res["tests"].setdefault("N1.T2_pw_pulse2021", {})[name] = r
        seed = K.BOOT_SEED + 4
        st = step_table(o, sim(wells, keys), wells)
        tables[f"T3_{name}"] = st
        r3 = {"PW_vs_M": compare(st, "PW", "M", seed),
              "informative_rms_m": (float(np.sqrt(np.mean((st.delta_PW - st.delta_M) ** 2)))
                                    if len(st) else float("nan")),
              "PW_vs_CLtrend": compare(st.dropna(subset=["e_CLtrend"]), "PW", "CLtrend", seed),
              "PW_vs_CL0": compare(st, "PW", "CL0", seed),
              "M_vs_CLtrend": compare(st.dropna(subset=["e_CLtrend"]), "M", "CLtrend", seed),
              "sensitivities_vs_M": {k: compare(st, k, "M", seed) for k in keys[2:]},
              "strata_distance_PW_vs_M": strata_dist(st, "PW", "M", seed),
              "mean_delta": {k: (float(st[f"delta_{k}"].mean()) if len(st) else float("nan"))
                             for k in ["obs", "CLtrend"] + keys}}
        if r3["informative_rms_m"] < K.ZR_INFORMATIVE_M:
            r3["PW_vs_M"]["verdict_registered"] = "not informative"
        res["tests"].setdefault("N1.T3_pw_step2019", {})[name] = r3
        log(f"T2 {name}: PW vs M {r['PW_vs_M'].get('verdict')} (inf {r['informative_rms_m']:.3f}); "
            f"T3: PW vs M {r3['PW_vs_M'].get('verdict')} (inf {r3['informative_rms_m']:.3f})")

    # ---- C1 consistency (full theta, calibration wells)
    c1 = {}
    for i, ev in enumerate(K.EVENTS):
        s = sim(cal, ["M", f"ZR_{ev}"])
        tab = event_table(obs_cal, {"M": s["M"], "ZR": s[f"ZR_{ev}"]}, cal, ev)
        tables[f"C1_{ev}"] = tab
        c1[ev] = {"M_vs_CL": compare(tab, "M", "CL", K.BOOT_SEED + 20 + i),
                  "M_vs_ZR_yunlin": compare(tab[tab.county == "Yunlin"], "M", "ZR",
                                            K.BOOT_SEED + 20 + i),
                  "zr_informative_rms_m_yunlin": informative(tab[tab.county == "Yunlin"], "ZR"),
                  "slope_M": slope(tab, "M", K.BOOT_SEED + 20 + i),
                  "by_county": {c: compare(tab[tab.county == c], "M", "CL", K.BOOT_SEED + 20 + i)
                                for c in ("Changhua", "Yunlin")}}
        log(f"C1 {ev}: M vs CL {c1[ev]['M_vs_CL'].get('verdict')}")
    res["checks"]["C1_insample"] = {"label": "consistency check (theta fitted on these heads)",
                                    "events": c1}
    # ---- C2 fold theta
    kf = pd.read_csv(K.TWIN_KFOLD_WELLS, dtype={"sid": str})
    fold_of = dict(zip(kf.sid, kf.fold, strict=True))
    c2_tabs = {ev: [] for ev in K.EVENTS}
    for fm in mdl["folds"]:
        k = int(fm.label.replace("fold", ""))
        model_k, sc_k, _ = mdl["build_model"](inp.grid, fm, "cpu")
        tw_k = mdl["engine"].SparseTwin(model_k, sc_k, mdl["ge"],
                                        pump_layer=int(fm.meta.get("pump_layer", 1)),
                                        recharge_layer=int(fm.meta.get("recharge_layer", 0)))
        held = np.array([fold_of.get(s) == k for s in inp.sids])
        h0k = inp.initial_heads(0, n_layers=4, well_mask=~held).numpy()
        Hk = {"M": tw_k.rollout(h0k, E=E, R=R)}
        for ev in K.EVENTS:
            Hk[f"ZR_{ev}"] = tw_k.rollout(h0k, E=zr_energy(E, ev), R=R)
        sel = cal.sid.map(fold_of).eq(k).to_numpy()
        wk = cal[sel].reset_index(drop=True)
        for ev in K.EVENTS:
            s = {"M": heads_at(Hk["M"], wk.cell.to_numpy(), wk.layer.to_numpy()),
                 "ZR": heads_at(Hk[f"ZR_{ev}"], wk.cell.to_numpy(), wk.layer.to_numpy())}
            t_ = event_table(obs_cal[sel], s, wk, ev)
            t_["fold"] = k
            c2_tabs[ev].append(t_)
        log(f"C2 fold {k}: {int(sel.sum())} held-out wells")
    c2 = {}
    for i, ev in enumerate(K.EVENTS):
        tab = pd.concat(c2_tabs[ev], ignore_index=True)
        tables[f"C2_{ev}"] = tab
        c2[ev] = {"M_vs_CL": compare(tab, "M", "CL", K.BOOT_SEED + 30 + i),
                  "M_vs_ZR_yunlin": compare(tab[tab.county == "Yunlin"], "M", "ZR",
                                            K.BOOT_SEED + 30 + i),
                  "zr_informative_rms_m_yunlin": informative(tab[tab.county == "Yunlin"], "ZR"),
                  "slope_M": slope(tab, "M", K.BOOT_SEED + 30 + i),
                  "by_county": {c: compare(tab[tab.county == c], "M", "CL", K.BOOT_SEED + 30 + i)
                                for c in ("Changhua", "Yunlin")}}
        log(f"C2 {ev}: M vs CL {c2[ev]['M_vs_CL'].get('verdict')}")
    res["checks"]["C2_fold_theta"] = {"label": "design-time evidence (batch-8 k-fold partition "
                                               "gated the deliverable)", "events": c2,
                                      "ic": "IDW of the fold's kept wells in 2012-01"}
    res["runtime_s"] = round(time.perf_counter() - t_start, 1)
    for k, t_ in tables.items():
        t_.to_csv(OUT / f"{k}.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                 default=float) + "\n")
    (OUT / "run.log").write_text("\n".join(LOG) + "\n")
    log(f"done in {res['runtime_s']} s -> {rel_to_repo(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
