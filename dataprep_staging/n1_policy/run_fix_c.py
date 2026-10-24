"""N1.changhua_fix.v1: fixes F1 and F2 for Changhua scored on step C's masked 2016-18 window.

Changhua only (Yunlin stays R0). Models, all masked 2016-01..2018-12 with the 2016-18 surveys
withheld from their columns:
* R1c (reference, the registered step-C Changhua model): stage3_datum_gate_ccz_m1618 +
  coupled_leveling_z3_w1618 (three-zone column, free tau).
* F1: the same flow + a three-zone column fitted with --tau-min-days 3650 (slow rheology,
  fixed in advance from R0's mid-zone tau at the 10.8-yr ceiling and the literature's
  decades): coupled_leveling_z3_w1618_tau3650.
* F2: the R1c recipe + --return-rice-share (paddy-weighted return flow,
  n1_policy/rice_intensity/v1), masked refit stage3_datum_gate_ccr_m1618 + its own
  three-zone column coupled_leveling_z3_w1618.

Design gates, evaluated BEFORE any test is unlocked, on model outputs only:
  G-a  the column's rest-of-fan out-of-fold leveling R2 (r2_outoffold_by_zone
       'county:rest', non-withheld surveys) >= the reference column's - 0.02;
  G-b  no uplift: modelled mean subsidence May 2019 -> May 2022 > 0 over the fan and over
       Changhua.
A fix failing a gate is reported 'failed_design_gate' and not scored.

Tests (scored once, step C's scorer functions unchanged): F1 leveling partition; F2 heads and
leveling partition; each M vs CL, M vs ZR and M vs the R1c reference (paired by well or
benchmark, same bootstrap). Outputs: results/dataprep/reports/N1_policy/fix_c/.
``--dry`` checks gates and code paths on synthetic observations (no look, nothing written).
Usage: CUDA_VISIBLE_DEVICES="" results/twin_runs/par25/py.sh run_fix_c.py [--dry]
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
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402
from n1_natural_experiments import n1lib as L  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402
from n1_policy import run_step_c as C  # noqa: E402

PREREG = "N1.changhua_fix.v1"
R = "results/twin_runs"
REF = (f"{R}/stage3_datum_gate_ccz_m1618/stage3_theta.json",
       f"{R}/stage3_datum_gate_ccz_m1618/coupled_leveling_z3_w1618/vep_zonal_leveling.json")
FIXES = {
    "F1": (f"{R}/stage3_datum_gate_ccz_m1618/stage3_theta.json",
           f"{R}/stage3_datum_gate_ccz_m1618/coupled_leveling_z3_w1618_tau3650/"
           "vep_zonal_leveling.json"),
    "F2": (f"{R}/stage3_datum_gate_ccr_m1618/stage3_theta.json",
           f"{R}/stage3_datum_gate_ccr_m1618/coupled_leveling_z3_w1618/vep_zonal_leveling.json"),
}
TESTS = {("F1", "lev"): "N1F.F1_lev_changhua", ("F2", "heads"): "N1F.F2_heads_changhua",
         ("F2", "lev"): "N1F.F2_lev_changhua"}
SEED = 20261124
GATE_DR2 = 0.02
OUT = REPORTS_DIR / "N1_policy" / "fix_c"


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def check_recipe(name: str, theta: str, col: str) -> None:
    meta = json.loads((REPO / theta).read_text())["meta"]
    cj = json.loads((REPO / col).read_text())
    if meta.get("mask_months") != C.MASK or cj.get("withhold_surveys") != C.MASK:
        raise SystemExit(f"{name}: mask {meta.get('mask_months')!r} / withhold "
                         f"{cj.get('withhold_surveys')!r} != {C.MASK}")
    if cj.get("theta") != theta or not cj.get("column_zones_collapsed"):
        raise SystemExit(f"{name}: {col} is not the three-zone column of {theta}")
    if name == "F1" and float(cj.get("tau_min_days") or 0) != 3650.0:
        raise SystemExit(f"F1: tau_min_days {cj.get('tau_min_days')!r} != 3650")
    if name == "F2" and not meta.get("return_rice_share"):
        raise SystemExit("F2: the flow theta has no return_rice_share")


def rest_r2(col: str) -> float:
    z = json.loads((REPO / col).read_text()).get("r2_outoffold_by_zone") or {}
    v = (z.get("county:rest") or {}).get("r2")
    return float("nan") if v is None else float(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    if not a.dry and (OUT / "results.json").exists():
        raise SystemExit(f"{OUT / 'results.json'} exists; results are never recomputed")
    from n1_natural_experiments import run_cpu_tests as RC
    from n1_natural_experiments import units_n1 as U

    from hydrophysics.config import Config
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths
    from hydrophysics.twin.leveling import load_panel
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid

    models = {"R1c": REF, **FIXES}
    for name, (th, col) in models.items():
        check_recipe(name, th, col)
    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
    ch_cell = np.asarray(county_flags_for_grid(grid, DEFAULT_MAP)[0], dtype=bool)
    may = {y: list(C.MONTHS).index(pd.Period(f"{y}-05", "M")) for y in (2019, 2022)}
    runs, gates = {}, {}
    for name, (th, col) in models.items():
        tm = PL.TwinModel(str(REPO / th), str(REPO / col))
        E = tm.energy()
        H = {"M": tm.heads(), "ZR": tm.heads(C.zr_energy(E))}
        Sf = {n: tm.subsidence(h) for n, h in H.items()}
        oi = {s: (int(c), int(ly)) for s, c, ly in zip(tm.inp.sids, tm.inp.obs_idx,
                                                     tm.inp.obs_layer, strict=True)}
        runs[name] = {"tm": tm, "H": H, "Sf": Sf, "oi": oi}
        d = Sf["M"][:, may[2022]] - Sf["M"][:, may[2019]]
        gates[name] = {"rest_oof_r2": rest_r2(col), "fan_2019_2022_m": float(d.mean()),
                       "changhua_2019_2022_m": float(d[ch_cell].mean())}
        log(f"[{name}] model runs done; gate inputs {gates[name]}")
    ref_r2 = gates["R1c"]["rest_oof_r2"]
    for name in FIXES:
        g = gates[name]
        g["pass_a"] = bool(g["rest_oof_r2"] >= ref_r2 - GATE_DR2)
        g["pass_b"] = bool(g["fan_2019_2022_m"] > 0 and g["changhua_2019_2022_m"] > 0)
        g["passed"] = g["pass_a"] and g["pass_b"]
    live = {t: tid for t, tid in TESTS.items() if gates[t[0]]["passed"]}
    if not a.dry:
        for tid in live.values():
            r = prereg3.assert_can_score(PREREG, "N1", test_id=tid)
            log(f"assert_can_score {tid}: ok (post-registration looks "
                f"{len(r['post_registration_looks'])})")
    rng = np.random.default_rng(SEED)
    meta = U.station_meta().set_index("sid")
    wells = U.calibration_wells()
    wells["site"] = wells.sid.str[:6]
    wells["county"] = wells.sid.map(meta.CountyName).map({"彰化縣": "changhua",
                                                          "雲林縣": "yunlin"})
    sel = (wells.county == "changhua").to_numpy()
    wch = wells[sel].reset_index(drop=True)
    units = json.loads(C.UNITS.read_text(encoding="utf-8"))["units"]
    if a.dry:
        ddir = str(Config().data_dir)
        panel = pd.read_parquet(os.path.join(ddir, "ls_cache",
                                             "ls-wra-lsp-obs__choushui_panel.parquet"),
                                columns=["sid", "datetime", "x_3826", "y_3826"])
        panel = panel.rename(columns={"x_3826": "x", "y_3826": "y"})
        panel["datetime"] = pd.to_datetime(panel.datetime)
        panel["elev_m"] = -0.02 * (panel.datetime.dt.year - 2015) + rng.normal(0, .004,
                                                                               len(panel))
    else:
        panel = load_panel(str(Config().data_dir))
    panel["sid"] = panel.sid.astype(str)
    lev_tab, cum_obs = C.may_cum(panel, units["N1P:lev_m1618"]["ids"])
    may_idx = [list(C.MONTHS).index(pd.Period(f"{y}-05", "M")) for y in C.MAY_YEARS]
    d_obs = PL.partition_deviation(cum_obs)
    obs = None if a.dry else RC.observed(list(wch.sid))

    def series(name):
        oi = runs[name]["oi"]
        cells = np.array([oi[s][0] for s in wch.sid])
        lays = np.array([oi[s][1] for s in wch.sid])
        return {n: h[lays, cells, :] for n, h in runs[name]["H"].items()}

    if a.dry:
        obs = series("R1c")["M"] + rng.normal(0, 0.5, series("R1c")["M"].shape)
        obs[:, :12] = np.nan
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "prereg": PREREG, "dry": bool(a.dry), "models": models, "gates": gates,
           "tests": {}}
    for (fix, kind), tid in TESTS.items():
        if (fix, kind) not in live:
            res["tests"][tid] = {"verdict": "failed_design_gate", "gate": gates[fix]}
            continue
        if kind == "heads":
            sf, sr = series(fix), series("R1c")
            tab = C.head_table(obs, sf, wch)
            tr = C.head_table(obs, {"M": sr["M"], "ZR": sr["ZR"]}, wch).set_index("sid")
            out = C.score_heads(tab, SEED)
            common = tab.sid.isin(tr.index).to_numpy()
            out["M_vs_R1c"] = L.paired_site_bootstrap(
                tab.e_M[common], tr.loc[tab.sid[common], "e_M"].to_numpy(),
                tab.site[common].to_numpy(), B=PL.B_BOOT, seed=SEED + 5, level=PL.LEVEL,
                min_sites=PL.MIN_CLUSTERS)
        else:
            bc = runs[fix]["tm"].cell_of(lev_tab[["x", "y"]].to_numpy())
            on = (bc >= 0) & ch_cell[np.clip(bc, 0, None)]
            dM = PL.partition_deviation(runs[fix]["Sf"]["M"][bc[on]][:, may_idx])
            dZ = PL.partition_deviation(runs[fix]["Sf"]["ZR"][bc[on]][:, may_idx])
            dR = PL.partition_deviation(runs["R1c"]["Sf"]["M"][bc[on]][:, may_idx])
            xy = lev_tab[["x", "y"]].to_numpy()[on]
            out = C.score_lev(d_obs[on], dM, dZ, xy, SEED + 10)
            out["M_vs_R1c"] = L.paired_site_bootstrap(
                PL.rms_rows(dM, d_obs[on]), PL.rms_rows(dR, d_obs[on]), PL.blocks_of(xy),
                B=PL.B_BOOT, seed=SEED + 15, level=PL.LEVEL, min_sites=PL.MIN_CLUSTERS)
            out["n_benchmarks"] = int(on.sum())
        res["tests"][tid] = {"model": models[fix], **out}
        log(f"{tid}: vs CL {out.get('M_vs_CL', {}).get('verdict')}, vs R1c "
            f"{out['M_vs_R1c']['verdict']}")
    if a.dry:
        print(json.dumps({"gates": gates, "tests": {k: {kk: (vv.get("verdict") if
              isinstance(vv, dict) else vv) for kk, vv in v.items()}
              for k, v in res["tests"].items()}}, indent=1, default=str)[:3000])
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"wrote {rel_to_repo(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
