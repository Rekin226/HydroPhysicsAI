"""Run test G4 of ``N1.phase12_response.v1``: the compaction column against the printed deep
benchmarks, Jan-May 2020 and Jan-May 2021 relative to Jan-May 2022.

Model: the I1 primary model is R0 (no I1 rung passed: results/dataprep/reports/I1_Rstar), i.e.
the batch-8 cwa theta and its leveling-calibrated zonal column
(``coupled_leveling/vep_zonal_leveling.json``; calibrated on leveling <= 2021, piles never
fitted). The column is driven, as ``calibrate_coupled`` fitted it, by the theta's free-run
hindcast (gain 0) recomputed on the CPU engine and checked against engine/v2's stored
hindcast; ``forward.load_or_fit_vep`` (hpc0 fast-days 365, the forward archive's option) and
``forward.compaction`` build it. Piles are placed as ``deep_pile_holdout`` places them
(``mechanism.ring_cells``, nearest active cell within one cell width).

Depth interval: ``forward.compaction`` drives one column per cell with the layer-mean head; it
has no per-layer compaction, so a sensor's anchor-to-surface interval cannot be resolved. The
total column is used for every sensor (deviation, recorded in results.json; the caveat is
``deep_pile_holdout``'s (iii)).

Metric (registered): delta_e = C(Jan-May e) - C(Jan-May 2022) per sensor, e in {2020, 2021};
error |delta_mod - delta_obs|; RMSE over sensors; ZR: delta = 0. Observations: printed_monthly
(flag 0) opened through the guard as scoring_target; a sensor-year counts only when all five
months are flag 0.

1. ``prereg3.assert_can_score`` BEFORE any pile value is read (logs the scoring look).
2. The column at the pile cells from the recomputed hindcast.
3. The observed Jan-May sums, deltas, RMSEs, sign agreement, verdict.

``--dry`` runs 2-3 on synthetic pile values (assert_can_score with log=False, the guard is
not opened, nothing written). Outputs: ``results/dataprep/reports/N1_evidence/phase12_tests/
g4/`` (results.json, g4_sensors.csv, run.log).

Usage: CUDA_VISIBLE_DEVICES="" cpu_slot.sh python -I run_g4_piles.py [--dry]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import n1lib as L  # noqa: E402
from n1_natural_experiments import run_cpu_tests as C  # noqa: E402
from n1_natural_experiments import run_phase12_tests as P  # noqa: E402

PREREG = "N1.phase12_response.v1"
G4 = "N1.G4_piles_2020_2021"
OUT = K.EVID / "phase12_tests/g4"
VEP = REPO / "results/twin_runs/stage3_datum_gate_cwa/coupled_leveling/vep_zonal_leveling.json"
SITES = REPO / "chou-shui-data/data/subsidence/deep_pile_sites.csv"
FORWARD = REPO / "results/twin_forward/datum_gate_cwa.npz"     # sensitivity only
HPC0_FAST_DAYS = 365.0                       # the forward archive's column_hpc0_fast_days
EVENT_YEARS = (2020, 2021)
REF_YEAR = 2022
JANMAY = (1, 2, 3, 4, 5)
WINDOW = ("2020-01", "2022-05")              # the test's data_scope
MIN_SIGNS = 10                               # registered: >= 10 of 14 signs agree
N_SENSOR_EVENTS = 14
MONTHS = C.MONTHS                            # H[..., k] <-> MONTHS[k], 2012-01..2022-12
log = C.log


def registered_sensors() -> list[str]:
    doc = json.loads((K.DATAPREP / "preregistration" / f"{PREREG}.json").read_text())
    t = [t for t in doc["tests"] if t["test_id"] == G4][0]
    return list(t["external_units"]["N1:piles"]["ids"])


# ------------------------------------------------------------------------- model
def column_at_cells(H: np.ndarray, member, inp) -> tuple[np.ndarray, dict]:
    """Cumulative compaction (A, T) m of the R0 column on heads H (L, A, T)."""
    import torch

    from hydrophysics.twin.forward import build_model, compaction, load_or_fit_vep
    from hydrophysics.twin.zones import fan_zones

    _, _, zoc = build_model(inp.grid, member, "cpu")
    if zoc is None:                                   # calibrate_coupled's fallback
        zoc = fan_zones(inp.grid.centroids())
    zone3 = np.load(K.ENGINE_DIR / "engine_arrays.npz")["zone3"]
    if not np.array_equal(np.asarray(zoc), zone3):
        raise SystemExit("the column zone map differs from engine/v2's zone3")
    col, p = load_or_fit_vep(str(VEP), None, None, "cpu", zone_of_cell=np.asarray(zoc),
                             hpc0_fast_days=HPC0_FAST_DAYS,
                             county_sha256=member.meta.get("zone_county_sha256"))
    Cc = compaction(col, torch.as_tensor(H, dtype=torch.float64))
    info = {"vep_json": rel_to_repo(VEP), "vep_sha256": sha256_file(VEP),
            "config": p.get("config"), "target": p.get("target"),
            "hpc0_released": p.get("hpc0_released"),
            "zones_per_cell": {int(z): int((np.asarray(zoc) == z).sum())
                               for z in np.unique(zoc)}}
    return np.asarray(Cc, float), info


def pile_cells(sites: pd.DataFrame) -> np.ndarray:
    from hydrophysics.twin.mechanism import ring_cells

    z = np.load(K.ENGINE_DIR / "engine_arrays.npz")
    npz = {k: z[k] for k in ("mask", "dx", "x0", "y0")}
    return ring_cells(npz, sites[["x_twd97", "y_twd97"]].to_numpy(float))


def janmay_model(Cc: np.ndarray, cells: np.ndarray, years) -> dict[int, np.ndarray]:
    """Jan-May compaction (mm) per cell for each year: C(May) - C(Dec of the year before)."""
    pos = {str(p): k for k, p in enumerate(MONTHS)}
    out = {}
    for y in years:
        k1, k0 = pos[f"{y}-05"], pos[f"{y - 1}-12"]
        out[y] = (Cc[cells, k1] - Cc[cells, k0]) * 1000.0
    return out


def janmay_archive(cells: np.ndarray, years) -> dict:
    """Sensitivity: the forward archive's hindcast column (ensemble mean over its 18
    members, rheology coupled_leveling, baseline scenario). Not the registered model."""
    with np.load(FORWARD, allow_pickle=False) as z:
        rl = [str(s) for s in z["rheology_labels"]]
        ri = rl.index("coupled_leveling")
        dates = [str(d)[:7] for d in z["dates"]]
        S = np.asarray(z["subs_mean_by_rheology"][ri, 0], dtype=float)
    pos = {d: k for k, d in enumerate(dates)}
    return {y: (S[cells, pos[f"{y}-05"]] - S[cells, pos[f"{y - 1}-12"]]) * 1000.0
            for y in years}


# ------------------------------------------------------------------------- observations
OBS_COLS = ["sensor_id", "station", "depth_m", "month", "value", "flag", "positive_means"]


def observed_table() -> tuple[pd.DataFrame, list[str]]:
    p = guard.open_product(K.PILES_PRINTED, "scoring_target", window_end=WINDOW[1])
    df = p.read_table("printed_monthly.parquet", columns=OBS_COLS)
    return df, list(p.notes)


def synthetic_table(sensors: list[str], sites: pd.DataFrame, mon_model: dict) -> pd.DataFrame:
    """Dry run only: model monthly compaction + noise, one sensor-month flagged 9."""
    rng = np.random.default_rng(0)
    rows = []
    for s in sensors:
        r = sites.loc[s]
        for y in (*EVENT_YEARS, REF_YEAR):
            for m in JANMAY:
                v = mon_model[s][f"{y}-{m:02d}"] + rng.normal(0.0, 3.0)
                rows.append({"sensor_id": s, "station": r.pile, "depth_m": int(r.depth_m),
                             "month": f"{y}-{m:02d}", "value": round(float(v), 1), "flag": 0,
                             "positive_means": "compression (synthetic)"})
    df = pd.DataFrame(rows)
    df.loc[(df.sensor_id == sensors[0]) & (df.month == "2021-03"), "flag"] = 9
    return df


def janmay_obs(df: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """(sensor x year) Jan-May sums in mm (NaN unless all five months flag 0); drop log."""
    df = df[(df.month >= WINDOW[0]) & (df.month <= WINDOW[1])].copy()
    df["year"] = df.month.str[:4].astype(int)
    df["mon"] = df.month.str[5:7].astype(int)
    df = df[df.mon.isin(JANMAY) & df.year.isin((*EVENT_YEARS, REF_YEAR))]
    dup = df[df.flag == 0].duplicated(["sensor_id", "month"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate flag-0 sensor-months in the printed table")
    out = pd.DataFrame(np.nan, index=sensors, columns=[*EVENT_YEARS, REF_YEAR])
    drops = []
    for s in sensors:
        for y in out.columns:
            g = df[(df.sensor_id == s) & (df.year == y)]
            ok = g[g.flag == 0]
            if sorted(ok.mon) == list(JANMAY) and ok.value.notna().all():
                out.at[s, y] = float(ok.value.sum())
            else:
                drops.append({"sensor": s, "year": int(y), "n_flag0": int(len(ok)),
                              "flags_seen": sorted(int(f) for f in g.flag.unique())})
    return out, drops


# ------------------------------------------------------------------------- scoring
def score(obs: pd.DataFrame, mod: dict, sensors: list[str], sites: pd.DataFrame,
          cells: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    for i, s in enumerate(sensors):
        for e in EVENT_YEARS:
            do = obs.at[s, e] - obs.at[s, REF_YEAR]
            dm = float(mod[e][i] - mod[REF_YEAR][i])
            ok = bool(np.isfinite(do))
            rows.append({"sensor": s, "station": sites.loc[s, "pile"],
                         "depth_m": int(sites.loc[s, "depth_m"]), "cell": int(cells[i]),
                         "event": e, "evaluable": ok,
                         "obs_janmay_e_mm": obs.at[s, e], "obs_janmay_2022_mm": obs.at[s, REF_YEAR],
                         "mod_janmay_e_mm": float(mod[e][i]),
                         "mod_janmay_2022_mm": float(mod[REF_YEAR][i]),
                         "delta_obs_mm": do if ok else np.nan, "delta_mod_mm": dm,
                         "delta_ZR_mm": 0.0,
                         "err_mod_mm": abs(dm - do) if ok else np.nan,
                         "err_ZR_mm": abs(do) if ok else np.nan,
                         "sign_agree": (bool(np.sign(dm) == np.sign(do)) if ok else None)})
    tab = pd.DataFrame(rows)
    per_event = {}
    for e in EVENT_YEARS:
        g = tab[(tab.event == e) & tab.evaluable]
        rm = L.equal_well_rmse(g.err_mod_mm.to_numpy()) if len(g) else float("nan")
        rz = L.equal_well_rmse(g.err_ZR_mm.to_numpy()) if len(g) else float("nan")
        per_event[str(e)] = {"n_sensors": int(len(g)), "n_stations": int(g.station.nunique()),
                             "rmse_model_mm": rm, "rmse_ZR_mm": rz,
                             "ratio_model_to_ZR": rm / rz if rz > 0 else float("nan"),
                             "mean_delta_obs_mm": float(g.delta_obs_mm.mean()) if len(g) else None,
                             "mean_delta_mod_mm": float(g.delta_mod_mm.mean()) if len(g) else None}
    ev = tab[tab.evaluable]
    n_agree = int(ev.sign_agree.astype(bool).sum())
    ratios = [per_event[str(e)]["ratio_model_to_ZR"] for e in EVENT_YEARS]
    better = all(np.isfinite(r) and r < 1.0 for r in ratios) and n_agree >= MIN_SIGNS
    summ = {"per_event": per_event,
            "sign_agreement": {"n_agree": n_agree, "n_evaluable": int(len(ev)),
                               "n_registered": N_SENSOR_EVENTS, "needed": MIN_SIGNS,
                               "zero_obs_deltas": int((ev.delta_obs_mm == 0).sum())},
            "verdict_bootstrap": "not evaluable",
            "verdict": "model_better" if better else "not_model_better",
            "verdict_rule": "model_better only if the RMSE ratio to ZR < 1 in both events and "
                            f">= {MIN_SIGNS} of {N_SENSOR_EVENTS} signs agree (registered); "
                            "uncalibrated, weak evidence"}
    return tab, summ


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="code check on synthetic pile values")
    a = ap.parse_args(argv)
    if not a.dry:
        OUT.mkdir(parents=True, exist_ok=True)
        if (OUT / "results.json").exists():
            raise SystemExit(f"{OUT / 'results.json'} exists; results are never recomputed")
    r = prereg3.assert_can_score(PREREG, K.WP, test_id=G4, log=not a.dry)
    scored = {"sha256": r["sha256"], "inputs_verified": r["inputs_verified"],
              "post_registration_looks": len(r["post_registration_looks"]),
              "unverifiable_prior_looks": len(r["unverifiable_prior_looks"]),
              "look_logged": not a.dry}
    log(f"assert_can_score {G4}: ok ({scored})")

    sensors = registered_sensors()
    sites = pd.read_csv(SITES)
    sites["sensor"] = sites.identifier.str.replace("BM_", "", regex=False)
    sites = sites.set_index("sensor").loc[sensors]
    cells = pile_cells(sites)
    if (cells < 0).any():
        raise SystemExit(f"sensors off the grid: {list(sites.index[cells < 0])}")
    log(f"sensors {sensors} -> cells {cells.tolist()} ({sites.pile.nunique()} stations)")

    mdl = P.load_refit(K.TWIN_THETA, False)
    tw, E, R, h0 = mdl["tw"], mdl["E"], mdl["R"], mdl["h0"]
    H = tw.rollout(h0, E=E, R=R)
    stored = np.load(K.ENGINE_DIR / "engine_arrays.npz")["H_hindcast"]
    repro = float(np.max(np.abs(H - stored)))
    log(f"free-run hindcast vs engine/v2 stored: max |dh| {repro:.3e} m; shape {H.shape}")
    if repro > 1e-3:
        raise SystemExit("the recomputed hindcast does not reproduce engine/v2's")
    Cc, col_info = column_at_cells(H, mdl["member"], mdl["inp"])
    years = (*EVENT_YEARS, REF_YEAR)
    mod = janmay_model(Cc, cells, years)
    arch = janmay_archive(cells, years)
    log("model Jan-May compaction (mm): " + "; ".join(
        f"{s} " + "/".join(f"{mod[y][i]:.1f}" for y in years) for i, s in enumerate(sensors)))

    if a.dry:
        mon = {s: {str(MONTHS[k]): (Cc[cells[i], k] - Cc[cells[i], k - 1]) * 1000.0
                   for k in range(1, len(MONTHS))} for i, s in enumerate(sensors)}
        df, guard_notes = synthetic_table(sensors, sites, mon), ["dry: synthetic"]
    else:
        df, guard_notes = observed_table()
    pm = sorted(df.positive_means.dropna().astype(str).unique().tolist())
    dep = df[df.sensor_id.isin(sensors)].groupby("sensor_id").depth_m.agg(
        lambda s: sorted(set(int(v) for v in s.dropna())))
    depth_mismatch = {s: {"printed": dep.get(s), "sites_csv": int(sites.loc[s, "depth_m"])}
                      for s in sensors if dep.get(s) != [int(sites.loc[s, "depth_m"])]}
    obs, drops = janmay_obs(df, sensors)
    log(f"observed Jan-May sums: {obs.notna().sum().to_dict()} sensors per year; "
        f"{len(drops)} sensor-years dropped; positive_means {pm}")
    tab, summ = score(obs, mod, sensors, sites, cells)
    tab_a, summ_a = score(obs, arch, sensors, sites, cells)
    for e in EVENT_YEARS:
        pe = summ["per_event"][str(e)]
        log(f"G4 {e}: n {pe['n_sensors']}, RMSE model {pe['rmse_model_mm']:.2f} mm, "
            f"ZR {pe['rmse_ZR_mm']:.2f} mm, ratio {pe['ratio_model_to_ZR']:.3f}")
    log(f"G4 signs {summ['sign_agreement']['n_agree']}/{summ['sign_agreement']['n_evaluable']}; "
        f"verdict {summ['verdict']}")

    deviations = [
        "depth interval: the column (forward.compaction) is driven by the layer-mean head and "
        "returns one total compaction per cell; it has no per-layer compaction, so the "
        "sensor's anchor-to-surface interval cannot be resolved. The total column is used "
        "for every sensor (130 m and 70 m sensors included; deep_pile_holdout caveat iii)",
        "the column JSON (vep_zonal_leveling.json) is not among the registration's hashed "
        "inputs; its sha256 is recorded here",
    ]
    n_ev = summ["sign_agreement"]["n_evaluable"]
    if n_ev < N_SENSOR_EVENTS:
        deviations.append(f"only {n_ev} of {N_SENSOR_EVENTS} sensor-events are evaluable "
                          "(all-five-months rule); the >= 10 sign count is applied as "
                          "registered to the evaluable ones")
    if depth_mismatch:
        deviations.append(f"printed depth differs from deep_pile_sites.csv: {depth_mismatch}")
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "prereg": PREREG, "test_id": G4, "dry": a.dry,
           "model": {"name": "R0 (the I1 primary model; no I1 rung passed)",
                     "theta": rel_to_repo(K.TWIN_THETA),
                     "theta_sha256": sha256_file(K.TWIN_THETA),
                     "column": col_info,
                     "heads": "free-run hindcast (gain 0) of the theta on the CPU engine "
                              "(SparseTwin), as calibrate_coupled fitted the column",
                     "hindcast_vs_engine_v2_max_abs_m": repro,
                     "pile_placement": "mechanism.ring_cells on the engine/v2 grid (nearest "
                                       "active cell within one cell width), as "
                                       "deep_pile_holdout",
                     "janmay_model": "C(May) - C(Dec of the previous year), mm"},
           "depth_interval_method": "total column (layer-mean-head driver); per-layer "
                                    "compaction not available",
           "observations": {"product": rel_to_repo(K.PILES_PRINTED),
                            "table": "printed_monthly.parquet", "use": "scoring_target",
                            "window_end": WINDOW[1], "flag": 0,
                            "positive_means": pm, "guard_notes": guard_notes,
                            "sensor_years_dropped": drops,
                            "janmay_sums_mm": {s: {str(y): (None if pd.isna(v) else float(v))
                                                   for y, v in obs.loc[s].items()}
                                               for s in sensors}},
           "sensors": [{"sensor": s, "station": sites.loc[s, "pile"],
                        "depth_m": int(sites.loc[s, "depth_m"]), "cell": int(cells[i])}
                       for i, s in enumerate(sensors)],
           "scores": summ,
           "sensitivity_not_verdict": {
               "what": "the forward archive datum_gate_cwa.npz hindcast column (18-member "
                       "ensemble mean, rheology coupled_leveling, baseline): not the "
                       "registered single model",
               "per_event": summ_a["per_event"], "sign_agreement": summ_a["sign_agreement"],
               "verdict_if_applied": summ_a["verdict"]},
           "deviations": deviations,
           "caveats": ["uncalibrated, weak evidence (registered)",
                       "2019-2021 printed months are uncertain by several mm (S2)",
                       "column out of sample (piles never fitted), heads in sample",
                       "three of the four stations host compaction rings that were a Stage-2 "
                       "target through 2022"]}
    tab["delta_mod_archive_mm"] = tab_a.delta_mod_mm
    res = no_nan(res)
    if a.dry:
        print(json.dumps(res, indent=1, default=str, ensure_ascii=False)[:4000])
        print(tab.to_string())
        return 0
    res["prereg_scored"] = scored
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    tab.to_csv(OUT / "g4_sensors.csv", index=False)
    (OUT / "run.log").write_text("\n".join(C.LOG) + "\n")
    return 0


def no_nan(x):
    """NaN -> None recursively."""
    if isinstance(x, dict):
        return {k: no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [no_nan(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        return no_nan(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


if __name__ == "__main__":
    raise SystemExit(main())
