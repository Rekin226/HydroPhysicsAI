"""Step C of N1.policy_response.v1: the masked 2016-18 refits (design-time evidence).

Models (batch 24): Yunlin = the R0 recipe refitted with --mask-months 2016-01:2018-12
(stage3_datum_gate_m1618) and its leveling column fitted with --withhold-surveys
2016-01:2018-12; Changhua = the R1c recipe (census v1, county zones) refitted with the same
mask (stage3_datum_gate_ccz_m1618) and its three-zone column (--column-zones 3), same
withheld surveys. Each model's own rollout drives its own column (policy_lib.TwinModel).
ZR = the same rollout with the energy of 2016-01..2018-12 replaced cell by cell by the mean
of the same calendar month over the reference years 2013, 2014, 2019 and 2022.

Heads (N1P.C_heads_*): G1's anomaly (n1lib.anomalies: intercept, trend and 2 harmonics fitted
on the reference years, per series), eligibility >= 18 of the 36 masked months and >= 36
reference months in >= 3 reference years; per-well RMS error, equal-well RMSE, paired
site-cluster bootstrap (sites = first 6 characters of the sid), B 2000, 90 %; M vs CL and
M vs ZR; ZR not informative if RMS(a_M - a_ZR) < 0.05 m.

Subsidence (N1P.C_lev_*): at the benchmarks of N1P:lev_m1618, cumulative subsidence at the
surveys nearest May 2015..2019 (within 45 days; site screened with max_rate 0.5 m/yr), the
four yearly increments minus their mean (the split of the 2015-19 total, which the kept 2015
and 2019 surveys constrain); the model's at the same model months. Per-benchmark RMS error,
paired cluster bootstrap over 5 km blocks; M vs CL (zero deviation = a constant rate) and M
vs ZR; ZR not informative if RMS(d_M - d_ZR) < 0.2 cm.

Reported with every verdict (never gating): pooled slope of observed on modelled anomaly,
and each model scored on the other county.

``--dry`` runs on synthetic observations (no head or elevation value read, no look,
nothing written). Outputs: results/dataprep/reports/N1_policy/step_c/.

Usage: CUDA_VISIBLE_DEVICES="" results/twin_runs/par24/py.sh -I run_step_c.py [--dry]
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
from n1_natural_experiments import n1lib as L  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402

PREREG = "N1.policy_response.v1"
MASK = "2016-01:2018-12"
TESTS = {("heads", "yunlin"): "N1P.C_heads_yunlin", ("heads", "changhua"): "N1P.C_heads_changhua",
         ("lev", "yunlin"): "N1P.C_lev_yunlin", ("lev", "changhua"): "N1P.C_lev_changhua"}
SEED = 20261123
REF_YEARS = [2013, 2014, 2019, 2022]
MIN_EVENT, MIN_REF, MIN_REF_YEARS = 18, 36, 3
INFORMATIVE_HEAD_M, INFORMATIVE_LEV_M = 0.05, 0.002
MAY_YEARS = [2015, 2016, 2017, 2018, 2019]
UNITS = REPORTS_DIR / "N1_policy" / "units_policy.json"
OUT = REPORTS_DIR / "N1_policy" / "step_c"
R = "results/twin_runs"
MODELS = {
    "yunlin": (f"{R}/stage3_datum_gate_m1618/stage3_theta.json",
               f"{R}/stage3_datum_gate_m1618/coupled_leveling_w1618/vep_zonal_leveling.json"),
    "changhua": (f"{R}/stage3_datum_gate_ccz_m1618/stage3_theta.json",
                 f"{R}/stage3_datum_gate_ccz_m1618/coupled_leveling_z3_w1618/"
                 "vep_zonal_leveling.json"),
}
DRY_MODELS = {   # --dry before batch 24 ends: the unmasked fits exercise the same code
    "yunlin": (f"{R}/stage3_datum_gate_cwa/stage3_theta.json",
               f"{R}/stage3_datum_gate_cwa/coupled_leveling/vep_zonal_leveling.json"),
    "changhua": (f"{R}/stage3_datum_gate_cwa/stage3_theta.json",
                 f"{R}/stage3_datum_gate_cwa/coupled_leveling/vep_zonal_leveling.json"),
}

MONTHS = PL.MONTHS
T = np.arange(len(MONTHS), dtype=float)
YEAR = np.array([p.year for p in MONTHS])
MON = np.array([p.month for p in MONTHS])
EVENT = (pd.Period("2016-01", "M") <= MONTHS) & (pd.Period("2018-12", "M") >= MONTHS)
REF = np.isin(YEAR, REF_YEARS)


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def check_models(models: dict) -> None:
    for county, (theta, col) in models.items():
        meta = json.loads((REPO / theta).read_text())["meta"]
        cj = json.loads((REPO / col).read_text())
        if meta.get("mask_months") != MASK:
            raise SystemExit(f"{theta}: mask_months {meta.get('mask_months')!r} != {MASK}")
        if cj.get("withhold_surveys") != MASK or cj.get("theta") != theta:
            raise SystemExit(f"{col}: withhold {cj.get('withhold_surveys')!r}, theta "
                             f"{cj.get('theta')!r}; expected {MASK} on {theta}")
        if county == "changhua" and not cj.get("column_zones_collapsed"):
            raise SystemExit(f"{col}: not a three-zone column")


def zr_energy(E: np.ndarray) -> np.ndarray:
    step_month = MONTHS[1:E.shape[-1] + 1]
    mos = np.array([p.month for p in step_month])
    tgt = EVENT[1:E.shape[-1] + 1]
    ref = REF[1:E.shape[-1] + 1]
    return PL.replace_months(E, tgt, ref, mos)


# ----------------------------------------------------------------------------- heads
def head_table(obs: np.ndarray, series: dict, wells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(wells)):
        o = obs[i]
        n_ev = int(np.isfinite(o[EVENT]).sum())
        refm = REF & np.isfinite(o)
        if n_ev < MIN_EVENT or refm.sum() < MIN_REF or len(set(YEAR[refm])) < MIN_REF_YEARS:
            continue
        a = L.anomalies(T, MON, o, {k: v[i] for k, v in series.items()}, REF, EVENT, 2)
        if a is None or not len(a["obs"]):
            continue
        rows.append({"sid": wells.sid.iloc[i], "site": wells.site.iloc[i],
                     "county": wells.county.iloc[i], "n": int(len(a["obs"])),
                     "e_CL": L.per_well_error(np.zeros_like(a["obs"]), a["obs"]),
                     "e_M": L.per_well_error(a["M"], a["obs"]),
                     "e_ZR": L.per_well_error(a["ZR"], a["obs"]),
                     "rms_ZR_M": float(np.sqrt(np.mean((a["ZR"] - a["M"]) ** 2))),
                     "a_obs": a["obs"].tolist(), "a_M": a["M"].tolist()})
    return pd.DataFrame(rows)


def score_heads(tab: pd.DataFrame, seed: int) -> dict:
    if tab.empty:
        return {"verdict_vs_CL": "not_evaluable", "n_wells": 0}
    s = tab.site.to_numpy()
    cl = L.paired_site_bootstrap(tab.e_M, tab.e_CL, s, B=PL.B_BOOT, seed=seed, level=PL.LEVEL,
                                 min_sites=PL.MIN_CLUSTERS)
    zr = L.paired_site_bootstrap(tab.e_M, tab.e_ZR, s, B=PL.B_BOOT, seed=seed + 1,
                                 level=PL.LEVEL, min_sites=PL.MIN_CLUSTERS)
    w = tab.n.to_numpy()
    inf = float(np.sqrt(np.sum(w * tab.rms_ZR_M ** 2) / np.sum(w)))
    if inf < INFORMATIVE_HEAD_M:
        zr["verdict"] = "not_informative"
    ml = max(tab.n)
    pad = lambda col: np.array([v + [np.nan] * (ml - len(v)) for v in tab[col]])  # noqa: E731
    return {"M_vs_CL": cl, "M_vs_ZR": zr, "rms_ZR_minus_M_m": inf,
            "slope_obs_on_M": PL.pooled_slope(pad("a_M"), pad("a_obs"), s, seed=seed + 2)}


# ----------------------------------------------------------------------------- leveling
def may_cum(panel: pd.DataFrame, ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    from hydrophysics.twin.leveling import site_subsidence, site_xy

    sub = site_subsidence(panel[panel.sid.isin(set(ids))], "2015-01-01", "2020-01-01",
                          min_obs=5, max_rate=0.5)
    xy = site_xy(panel)
    rows, cum = [], []
    for sid, s in sub.items():
        vals = []
        for y in MAY_YEARS:
            d = np.abs((s.index - pd.Timestamp(f"{y}-05-01")).days)
            if d.min() > 45:
                break
            vals.append(float(s.iloc[int(d.argmin())]))
        if len(vals) == len(MAY_YEARS):
            rows.append({"sid": sid, "x": xy[sid][0], "y": xy[sid][1]})
            cum.append(vals)
    return pd.DataFrame(rows), np.array(cum)


def score_lev(d_obs, d_M, d_ZR, xy, seed: int) -> dict:
    cl = PL.blocks_of(xy)
    e_M, e_ZR = PL.rms_rows(d_M, d_obs), PL.rms_rows(d_ZR, d_obs)
    e_CL = PL.rms_rows(np.zeros_like(d_obs), d_obs)
    vs_cl = L.paired_site_bootstrap(e_M, e_CL, cl, B=PL.B_BOOT, seed=seed, level=PL.LEVEL,
                                    min_sites=PL.MIN_CLUSTERS)
    vs_zr = L.paired_site_bootstrap(e_M, e_ZR, cl, B=PL.B_BOOT, seed=seed + 1, level=PL.LEVEL,
                                    min_sites=PL.MIN_CLUSTERS)
    inf = float(np.sqrt(np.mean((d_M - d_ZR) ** 2)))
    if inf < INFORMATIVE_LEV_M:
        vs_zr["verdict"] = "not_informative"
    return {"M_vs_CL": vs_cl, "M_vs_ZR": vs_zr, "rms_ZR_minus_M_m": inf,
            "slope_obs_on_M": PL.pooled_slope(d_M, d_obs, cl, seed=seed + 2)}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    models = MODELS if not a.dry or (REPO / MODELS["changhua"][1]).exists() else DRY_MODELS
    if not a.dry:
        if (OUT / "results.json").exists():
            raise SystemExit(f"{OUT / 'results.json'} exists; results are never recomputed")
        check_models(models)
        for t in TESTS.values():
            r = prereg3.assert_can_score(PREREG, "N1", test_id=t)
            log(f"assert_can_score {t}: ok (post-registration looks "
                f"{len(r['post_registration_looks'])})")
    from n1_natural_experiments import run_cpu_tests as C
    from n1_natural_experiments import units_n1 as U

    from hydrophysics.config import Config
    from hydrophysics.twin.leveling import load_panel

    rng = np.random.default_rng(SEED)
    units = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    meta = U.station_meta().set_index("sid")
    wells = U.calibration_wells()
    wells["site"] = wells.sid.str[:6]
    wells["county"] = wells.sid.map(meta.CountyName).map({"彰化縣": "changhua",
                                                          "雲林縣": "yunlin"})
    if a.dry:
        obs = None
        ddir = str(Config().data_dir)
        panel = pd.read_parquet(os.path.join(ddir, "ls_cache",
                                             "ls-wra-lsp-obs__choushui_panel.parquet"),
                                columns=["sid", "datetime", "x_3826", "y_3826"])
        panel = panel.rename(columns={"x_3826": "x", "y_3826": "y"})
        panel["datetime"] = pd.to_datetime(panel.datetime)
        panel["elev_m"] = -0.02 * (panel.datetime.dt.year - 2015) + rng.normal(0, .004,
                                                                               len(panel))
    else:
        obs = C.observed(list(wells.sid))
        panel = load_panel(str(Config().data_dir))
    panel["sid"] = panel.sid.astype(str)
    lev_tab, cum_obs = may_cum(panel, units["N1P:lev_m1618"]["ids"])
    may_idx = [list(MONTHS).index(pd.Period(f"{y}-05", "M")) for y in MAY_YEARS]
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "prereg": PREREG, "dry": bool(a.dry), "models": models, "tests": {},
           "secondary_cross_county": {}}
    runs = {}
    for county, (theta, col) in models.items():
        tm = PL.TwinModel(str(REPO / theta), str(REPO / col))
        E = tm.energy()
        H = {"M": tm.heads(), "ZR": tm.heads(zr_energy(E))}
        Sf = {n: tm.subsidence(h) for n, h in H.items()}
        oi = {s: (int(c), int(ly)) for s, c, ly in zip(tm.inp.sids, tm.inp.obs_idx,
                                                     tm.inp.obs_layer, strict=True)}
        cells = np.array([oi[s][0] for s in wells.sid])
        lays = np.array([oi[s][1] for s in wells.sid])
        ser = {n: h[lays, cells, :] for n, h in H.items()}
        bcell = tm.cell_of(lev_tab[["x", "y"]].to_numpy())
        runs[county] = {"ser": ser, "Sf": Sf, "bcell": bcell}
        log(f"[{county}] model runs done ({theta})")
    if a.dry:
        obs = runs["yunlin"]["ser"]["M"] + rng.normal(0, 0.5, runs["yunlin"]["ser"]["M"].shape)
        obs[:, :12] = np.nan
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid

    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
    ch_cell = np.asarray(county_flags_for_grid(grid, DEFAULT_MAP)[0], dtype=bool)
    d_obs = PL.partition_deviation(cum_obs) if len(cum_obs) else np.zeros((0, 4))
    for j, county in enumerate(("yunlin", "changhua")):
        for mdl in ("yunlin", "changhua"):
            run = runs[mdl]
            sel = (wells.county == county).to_numpy()
            tab = head_table(obs[sel], {n: v[sel] for n, v in run["ser"].items()},
                             wells[sel].reset_index(drop=True))
            hs = score_heads(tab, SEED + 10 * j)
            bc = run["bcell"]
            on = (bc >= 0) & (ch_cell[np.clip(bc, 0, None)] == (county == "changhua"))
            if on.sum():
                dM = PL.partition_deviation(run["Sf"]["M"][bc[on]][:, may_idx])
                dZ = PL.partition_deviation(run["Sf"]["ZR"][bc[on]][:, may_idx])
                ls = score_lev(d_obs[on], dM, dZ, lev_tab[["x", "y"]].to_numpy()[on],
                               SEED + 20 * j)
                ls["n_benchmarks"] = int(on.sum())
            else:
                ls = {"verdict": "not_evaluable", "n_benchmarks": 0}
            if mdl == county:
                res["tests"][TESTS[("heads", county)]] = {"model": models[mdl], **hs}
                res["tests"][TESTS[("lev", county)]] = {"model": models[mdl], **ls}
            else:
                res["secondary_cross_county"][f"{county}_scored_with_{mdl}_model"] = {
                    "heads": hs, "lev": ls}
            log(f"{county} with {mdl} model: heads vs CL "
                f"{hs.get('M_vs_CL', {}).get('verdict')}, lev vs CL "
                f"{ls.get('M_vs_CL', {}).get('verdict')}")
    if a.dry:
        print(json.dumps(res["tests"], indent=1, default=str)[:2500])
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"wrote {rel_to_repo(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
