"""WP A1 acceptance: the harness scorers reproduce the pilot metrics named in A0.

    CUDA_VISIBLE_DEVICES= tools/cpu_slot.sh python -m harness.repro_pilots OUT_DIR

The three pilot outputs WP A0 named (plan WP A0 validation) are re-derived with the shared
scorers from the pilots' own predictions, not with the pilots' scoring code:

* **gap-fill** ``pilot_final_clean.csv`` (and ``pilot_final_all.csv``,
  ``pilot_summary_cleantruth.csv``): RMSE, median |err|, share within 0.5 m, native 90 %
  coverage, split-conformal sd multiplier, out-of-sample coverage and median width per gap
  family and method, from ``pilot_ensembles.parquet`` / ``pilot_all_scored.parquet``;
* **pseudo-Changhua** ``d_pseudo_changhua.json``: datum-free head anomaly RMSE (fit and
  site-CV) and the pseudo-truth recovery metrics, from arrays saved by a copy of
  ``d_pseudo_changhua.py`` patched only to save them (patches listed in the evidence);
* **temporal hold-out** ``f_temporal_holdout.json`` (physics pilot, pre-origin): the
  out-of-time RMSE with the training-mean datum, from arrays saved by a patched copy of
  ``f_temporal_holdout.py``;
* **TPC** ``compare_annual.csv``: annual totals and ratios of totals from the pilot's
  archived monthly intermediates. The file stores 4 decimals, so the pilot's own
  ``tools/compare.py`` is also rerun (patched only to write a full-precision copy) and the
  harness values are compared with that copy at 1e-6, and with the archive at its stored
  precision;
* **gap masks** (generators, not scores): the artificial-gap masks of families N, S, E and
  H (``pilot_{N,S,E,H}.parquet``) and of the test-period replica T
  (``pilot_testperiod.parquet``) are regenerated with :mod:`harness.gaps` from the pilot's
  own matrices and compared cell by cell.

Pilots are opened through the A0 loader guard for ``after_the_fact_analysis`` (output hashes
verified at open); every pilot script that is copied or run is checked against the sha256 in
its pilot manifest. Criterion: maximum relative difference <= 1e-6 over every compared value
(masks: identical). Nothing under ``results/dataprep/_pilots/`` is written; the patched
scripts run in ``OUT_DIR`` with symbolic links to the archived arrays.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.manifest import HashMismatch  # noqa: E402
from common.paths import PILOTS, rel_to_repo  # noqa: E402

from harness import gaps, scorers  # noqa: E402

PY = sys.executable
TOL = 1e-6
USE = "after_the_fact_analysis"
_OPEN: dict = {}


def pilot(name: str) -> guard.Product:
    """A pilot opened through the loader guard (hashes of every output verified once)."""
    if name not in _OPEN:
        _OPEN[name] = guard.open_product(PILOTS / name, USE)
    return _OPEN[name]


def out_path(name: str, rel: str) -> Path:
    """Path of a listed pilot output (the guard refuses anything the manifest does not
    list)."""
    return pilot(name).path(rel)


def code_path(name: str, rel: str) -> Path:
    """Path of a pilot script after checking its sha256 against the pilot manifest."""
    prod = pilot(name)
    p = prod.directory / rel
    recs = {c["path"]: c for c in prod.manifest["code"]}
    key = rel_to_repo(p)
    if key not in recs:
        raise KeyError(f"{key} is not listed as code in the {name} manifest")
    got = sha256_file(p)
    if got != recs[key]["sha256"]:
        raise HashMismatch(f"{key}: sha256 {got} != manifest {recs[key]['sha256']}")
    return p


def rel_diff(a, b) -> float:
    """|a - b| / |b| (b = pilot value); absolute difference when b == 0; 0 when both are
    NaN/None; inf when exactly one is."""
    na = a is None or (isinstance(a, float) and math.isnan(a))
    nb = b is None or (isinstance(b, float) and math.isnan(b))
    if na or nb:
        return 0.0 if (na and nb) else math.inf
    a, b = float(a), float(b)
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


class Tally:
    def __init__(self, name):
        self.name, self.n, self.max, self.worst, self.bad = name, 0, 0.0, None, []

    def add(self, key, mine, ref):
        d = rel_diff(mine, ref)
        self.n += 1
        if d > self.max:
            self.max, self.worst = d, {"key": key, "harness": mine, "pilot": ref}
        if d > TOL:
            self.bad.append({"key": key, "harness": mine, "pilot": ref, "rel_diff": d})

    def result(self, **extra):
        return {"part": self.name, "n_values": self.n, "max_rel_diff": self.max,
                "worst": self.worst, "n_above_tol": len(self.bad), "above_tol": self.bad[:20],
                "pass": self.n > 0 and not self.bad, **extra}


def _f(x):
    return None if x is None else float(x)


# ============================================================================ gap-fill
def part_gapfill() -> dict:
    G = "head_gapfill_pilot"
    D = pd.read_parquet(out_path(G, "out/pilot_ensembles.parquet"))
    out = {}
    for label, df in (("clean", D[~D.suspect & ~D.step_well]), ("all", D)):
        refp = out_path(G, f"out/pilot_final_{label}.csv")
        ref = pd.read_csv(refp).set_index(["Lname", "method"])
        t = Tally(f"gapfill_final_{label}")
        for (L, meth), g in df.groupby(["Lname", "method"]):
            es = scorers.error_summary(g.err, thresholds=(0.5,))
            nat = scorers.coverage_native(g.err, g.sd, z=scorers.Z90_PILOT, nan_sd="miss")
            cs = scorers.conformal_split(g.err.to_numpy(), g.gid.to_numpy(), g.sd.to_numpy(),
                                         level=0.9, n_split=20, seed=1, z=scorers.Z90_PILOT,
                                         finite_sample=False)
            mine = {"n": es["n"], "rmse": es["rmse"], "medae": es["medae"],
                    "le_0p5": es["le_0p5"], "cov90_native": nat["coverage"],
                    "sd_mult": cs["sd_mult_full"], "cov90_recal_oos": cs["cov_oos"],
                    "width90_recal_median": cs["width_median"]}
            r = ref.loc[(L, meth)]
            for k, v in mine.items():
                t.add(f"{L}/{meth}/{k}", v, _f(r[k]) if pd.notna(r[k]) else None)
        if t.n != ref.size:
            t.bad.append({"key": "row coverage", "harness": t.n, "pilot": int(ref.size)})
        out[label] = t.result(reference=rel_to_repo(refp), reference_sha256=sha256_file(refp),
                              settings="z=1.645 (pilot), finite_sample=False, seed=1, "
                                       "n_split=20, groups=gid, nan_sd='miss'")
    # pilot_summary_cleantruth.csv (s5b): equal-well RMSE (mean of per-well RMSE) etc.
    P = pd.read_parquet(out_path(G, "out/pilot_all_scored.parquet"))
    clean = P[~P.suspect & ~P.step_well]
    refp = out_path(G, "out/pilot_summary_cleantruth.csv")
    ref = pd.read_csv(refp).set_index(["Lname", "method"])
    t = Tally("gapfill_summary_cleantruth")
    for (L, meth), g in clean.groupby(["Lname", "method"]):
        es = scorers.error_summary(g.err, thresholds=(0.5,))
        sd = g.sd.to_numpy()
        nat = (scorers.coverage_native(g.err, sd, z=scorers.Z90_PILOT)["coverage"]
               if np.isfinite(sd).all() else None)
        mine = {"n": es["n"], "rmse": es["rmse"],
                "rmse_eqwell": scorers.equal_well_rmse_long(g.err, g.w, "mean_of_rmse"),
                "medae": es["medae"], "le_0p5": es["le_0p5"], "native_cov90": nat}
        r = ref.loc[(L, meth)]
        for k, v in mine.items():
            t.add(f"{L}/{meth}/{k}", v, _f(r[k]) if pd.notna(r[k]) else None)
    if t.n != ref.size:
        t.bad.append({"key": "row coverage", "harness": t.n, "pilot": int(ref.size)})
    out["summary_cleantruth"] = t.result(reference=rel_to_repo(refp),
                                         reference_sha256=sha256_file(refp),
                                         settings="rmse_eqwell = mean of per-well RMSE "
                                                  "(agg='mean_of_rmse'), z=1.645")
    return out


# ============================================================================ patched runs
def _patch(src: Path, dst: Path, patches: list[tuple[str, str]]) -> dict:
    s = src.read_text()
    for old, new in patches:
        if s.count(old) != 1:
            raise RuntimeError(f"patch anchor not unique in {src.name}: {old[:60]!r}")
        s = s.replace(old, new)
    dst.write_text(s)
    return {"source": rel_to_repo(src), "source_sha256": sha256_file(src),
            "patched": dst.name, "patched_sha256": sha256_file(dst),
            "patches": [{"anchor": o, "replacement": n} for o, n in patches]}


PH = "physics_inversion_hooks"


def _prepare(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for f in ("inputs_cwa.npz", "hindcast_cwa.npz", "K_pump_f32.npy", "M_cwa.npz"):
        os.symlink(out_path(PH, f), run_dir / f)
    shutil.copy2(code_path(PH, "inv_lib.py"), run_dir / "inv_lib.py")


def _run(script: Path, run_dir: Path, log: Path) -> dict:
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with open(log, "w") as f:
        rc = subprocess.run([PY, script.name, str(run_dir)], cwd=run_dir, stdout=f,
                            stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"{script.name} failed (rc {rc}); see {log}")
    return {"cmd": f"python {script.name} <run_dir>", "rc": rc,
            "seconds": round(time.time() - t0, 1), "log": rel_to_repo(log)}


PSEUDO_PATCHES = [
    ("RHO_G, J_KWH, MIN_LIFT = 9800.0, 3.6e6, 2.0",
     "RHO_G, J_KWH, MIN_LIFT = 9800.0, 3.6e6, 2.0\nDUMP = {}  # A1 patch: save arrays"),
    ("    truth_blk = np.array([Qf[lab == k].sum(0) for k in range(nb)])   # (nb, NT)",
     "    truth_blk = np.array([Qf[lab == k].sum(0) for k in range(nb)])   # (nb, NT)\n"
     "    DUMP[f'{rname}__truth_blk'] = truth_blk.copy()  # A1 patch\n"
     "    DUMP[f'{rname}__n_k'] = n_k.copy()  # A1 patch\n"
     "    DUMP[f'{rname}__pred_wh'] = pred_wh.copy()  # A1 patch\n"
     "    DUMP[f'{rname}__wR'] = wR.copy()  # A1 patch"),
    ("    def metrics(qb, label, fitted=None, extra=None):\n",
     "    def metrics(qb, label, fitted=None, extra=None):\n"
     "        DUMP[f'{rname}__{label}__qb'] = np.asarray(qb, float).copy()  # A1 patch\n"
     "        if fitted is not None:  # A1 patch\n"
     "            DUMP[f'{rname}__{label}__fitted'] = np.asarray(fitted, float).copy()\n"),
    ("        cv_rmse = anom_rmse(obs[wR], pred_wh[wR] + cv_pred[wR])",
     "        DUMP[f'{rname}__{label}__cv_pred'] = cv_pred.copy()  # A1 patch\n"
     "        cv_rmse = anom_rmse(obs[wR], pred_wh[wR] + cv_pred[wR])"),
    ("    th_m = (solm.x / sc)[:NT]",
     "    th_m = (solm.x / sc)[:NT]\n    DUMP[f'{rname}__I4__th_m'] = th_m.copy()  # A1 patch"),
    ('json.dump(summary, open(os.path.join(OUT, "d_pseudo_changhua.json"), "w"), indent=1,\n'
     '          default=float)',
     'json.dump(summary, open(os.path.join(OUT, "d_pseudo_changhua.json"), "w"), indent=1,\n'
     '          default=float)\n'
     'np.savez_compressed(os.path.join(OUT, "a1_dump.npz"), obs=obs, pred_full=pred_full,'
     ' **DUMP)  # A1 patch'),
]

HOLDOUT_PATCHES = [
    ("RHO_G, J_KWH, MIN_LIFT = 9800.0, 3.6e6, 2.0",
     "RHO_G, J_KWH, MIN_LIFT = 9800.0, 3.6e6, 2.0\nDUMP_T = []  # A1 patch: save arrays"),
    ("    r = o[wsel] - p[wsel]",
     "    DUMP_T.append((p.copy(), wsel.copy()))  # A1 patch\n    r = o[wsel] - p[wsel]"),
    ("def run_case(name, R, base_pred, wsel, out_y, truth=None):\n",
     "def run_case(name, R, base_pred, wsel, out_y, truth=None):\n"
     "    DUMP_T.append(('CASE', name))  # A1 patch\n"),
    ('json.dump(out, open(os.path.join(OUT, "f_temporal_holdout.json"), "w"), indent=1)',
     'json.dump(out, open(os.path.join(OUT, "f_temporal_holdout.json"), "w"), indent=1)\n'
     '_arr, _case, _i = {}, None, 0  # A1 patch\n'
     'for _x in DUMP_T:  # A1 patch\n'
     '    if isinstance(_x[0], str) and _x[0] == "CASE":\n'
     '        _case, _i = _x[1], 0\n'
     '        continue\n'
     '    _arr[f"{_case}__{_i}__p"], _arr[f"{_case}__{_i}__wsel"] = _x\n'
     '    _i += 1\n'
     'np.savez_compressed(os.path.join(OUT, "a1_dump.npz"), obs=obs, split=split, **_arr)'),
]


def _json_numbers(a, b, prefix=""):
    """(key, a, b) for every numeric leaf of two nested JSON structures."""
    if isinstance(b, dict):
        for k in b:
            yield from _json_numbers(a.get(k) if isinstance(a, dict) else None, b[k],
                                     f"{prefix}/{k}")
    elif isinstance(b, list):
        for i, v in enumerate(b):
            yield from _json_numbers(a[i] if isinstance(a, list) and i < len(a) else None, v,
                                     f"{prefix}[{i}]")
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        yield prefix, a, b


def part_pseudo(ev: Path) -> dict:
    run_dir = ev / "pseudo_run"
    _prepare(run_dir)
    patch = _patch(code_path(PH, "d_pseudo_changhua.py"), run_dir / "d_pseudo_changhua_a1.py",
                   PSEUDO_PATCHES)
    step = _run(run_dir / "d_pseudo_changhua_a1.py", run_dir, run_dir / "run.log")
    refp = out_path(PH, "d_pseudo_changhua.json")
    ref = json.loads(refp.read_text())
    rerun = json.loads((run_dir / "d_pseudo_changhua.json").read_text())
    tr = Tally("pseudo_patched_rerun_vs_archive")      # the patches changed nothing
    for k, a, b in _json_numbers(rerun, ref):
        tr.add(k, a, b)
    z = np.load(run_dir / "a1_dump.npz")
    obs, pred_full = z["obs"], z["pred_full"]
    dates = pd.date_range("2012-02-01", periods=obs.shape[1], freq="MS")
    t = Tally("pseudo_scorers_vs_archive")
    keymap = {"mean_Q_est_m3d": "mean_est", "mean_Q_true_m3d": "mean_true",
              "bias_pct": "bias_pct", "monthly_corr": "monthly_corr",
              "seasonal_corr": "seasonal_corr", "seasonal_amp_ratio": "seasonal_amp_ratio",
              "annual_corr": "annual_corr", "block_density_corr": "block_density_corr"}
    for rname, reg in ref.items():
        wR = z[f"{rname}__wR"]
        pred_wh = z[f"{rname}__pred_wh"]
        truth_blk, n_k = z[f"{rname}__truth_blk"], z[f"{rname}__n_k"]
        for row in reg["rows"]:
            lab = row["design"]
            mine = {}
            if lab == "B0_zero":
                mine["head_anomRMSE_wellsInR"] = scorers.anomaly_rmse(obs[wR],
                                                                      pred_wh[wR])["rmse"]
                # the zero baseline's bias: recovery of an all-zero estimate
                m0 = scorers.recovery_metrics(np.zeros_like(truth_blk), truth_blk, dates,
                                              n_k if truth_blk.shape[0] > 1 else None)
                mine["bias_pct"] = m0["bias_pct"]
            elif lab == "truth_full_census":
                mine["head_anomRMSE_wellsInR"] = scorers.anomaly_rmse(obs[wR],
                                                                      pred_full[wR])["rmse"]
            elif lab == "I4_free_monthly_regiontotal":
                m = scorers.recovery_metrics(z[f"{rname}__I4__th_m"], truth_blk.sum(0), dates)
                mine.update({k: m[v] for k, v in keymap.items() if v in m})
            else:
                qb = z[f"{rname}__{lab}__qb"]
                m = scorers.recovery_metrics(qb, truth_blk, dates,
                                             n_k if qb.shape[0] > 1 else None)
                mine.update({k: m[v] for k, v in keymap.items() if v in m})
                fk = f"{rname}__{lab}__fitted"
                if fk in z.files:
                    mine["head_anomRMSE_wellsInR"] = scorers.anomaly_rmse(
                        obs[wR], pred_wh[wR] + z[fk][wR])["rmse"]
                ck = f"{rname}__{lab}__cv_pred"
                if ck in z.files:
                    mine["head_anomRMSE_wellsInR_CV"] = scorers.anomaly_rmse(
                        obs[wR], pred_wh[wR] + z[ck][wR])["rmse"]
            for k, v in mine.items():
                if k in row:
                    t.add(f"{rname}/{lab}/{k}", v, row[k])
                elif v is not None:
                    t.bad.append({"key": f"{rname}/{lab}/{k}", "harness": v,
                                  "pilot": "absent"})
    covered = sum(1 for r in ref.values() for row in r["rows"] for k in row
                  if k in set(keymap) | {"head_anomRMSE_wellsInR", "head_anomRMSE_wellsInR_CV"})
    res = t.result(reference=rel_to_repo(refp), reference_sha256=sha256_file(refp),
                   metric_fields_in_reference=covered,
                   not_scorer_outputs="sd_mean_total_m3d, sigma_hat and ident_* are inversion "
                                      "or identifiability outputs, not scores")
    if t.n != covered:
        res["pass"] = False
        res["coverage_problem"] = f"compared {t.n} of {covered} metric fields"
    return {"scorers": res, "patched_rerun": tr.result(), "patch": patch, "step": step}


def part_holdout(ev: Path) -> dict:
    run_dir = ev / "holdout_run"
    _prepare(run_dir)
    patch = _patch(code_path(PH, "f_temporal_holdout.py"),
                   run_dir / "f_temporal_holdout_a1.py", HOLDOUT_PATCHES)
    step = _run(run_dir / "f_temporal_holdout_a1.py", run_dir, run_dir / "run.log")
    refp = out_path(PH, "f_temporal_holdout.json")
    ref = json.loads(refp.read_text())
    rerun = json.loads((run_dir / "f_temporal_holdout.json").read_text())
    tr = Tally("holdout_patched_rerun_vs_archive")
    for k, a, b in _json_numbers(rerun, ref):
        tr.add(k, a, b)
    z = np.load(run_dir / "a1_dump.npz")
    obs, split = z["obs"], int(z["split"])
    T = obs.shape[1]
    train = np.arange(T) < split
    t = Tally("holdout_scorers_vs_archive")
    for case, rows in ref.items():
        keys = [k for k, v in rows.items() if isinstance(v, list)]
        for i, k in enumerate(keys):
            p, wsel = z[f"{case}__{i}__p"], z[f"{case}__{i}__wsel"]
            r = scorers.out_of_time_rmse(obs[wsel], p[wsel], train, ~train, min_test=7)
            t.add(f"{case}/{k}/rmse", r["rmse"], rows[k][0])
            t.add(f"{case}/{k}/n_wells", r["n_wells"], rows[k][1])
    return {"scorers": t.result(reference=rel_to_repo(refp), reference_sha256=sha256_file(refp),
                                datum="training-mean datum (2012-02..2019-12), test "
                                      "2020-01..2022-12, wells with > 6 test months"),
            "patched_rerun": tr.result(), "patch": patch, "step": step}


# ============================================================================ TPC
TP = "tpc_aggregate_electricity"
TPC_PATCHES = [
    ('ya.to_csv(f"{out_dir}/compare_annual.csv", float_format="%.4f")',
     'ya.to_csv(f"{out_dir}/compare_annual.csv", float_format="%.4f")\n'
     'ya.to_csv(f"{out_dir}/compare_annual_full.csv", float_format="%.17g")  # A1 patch'),
]
TPC_COLS = ["tpc_ag_CH", "tpc_ag_YL", "tpc_all_CH", "tpc_all_YL", "raw_GWh", "dedup_GWh",
            "cleaned_GWh"]


def tpc_harness_annual(tpc_csv: Path, census_csv: Path) -> pd.DataFrame:
    """compare_annual from the pilot's monthly intermediates, with the harness totals."""
    t = pd.read_csv(tpc_csv, parse_dates=["date"])
    cen = pd.read_csv(census_csv, index_col=0)
    cen.index = pd.PeriodIndex(cen.index, freq="M").to_timestamp()
    ag = t[t.sector == "農林漁牧"].pivot(index="date", columns="county", values="kwh") / 1e6
    tot = t.groupby(["date", "county"])["kwh"].sum().unstack() / 1e6
    m = pd.DataFrame({"tpc_ag_CH": ag["彰化縣"], "tpc_ag_YL": ag["雲林縣"],
                      "tpc_all_CH": tot["彰化縣"], "tpc_all_YL": tot["雲林縣"]}).join(cen,
                                                                                    how="left")
    ya = scorers.annual_sum(m, TPC_COLS, min_count=1).rename(columns={"months": "months_tpc"})
    R = scorers.ratio_of_totals
    ya["CH_over_YL"] = R(ya.tpc_ag_CH, ya.tpc_ag_YL)
    ya["ag_share_CH"] = R(ya.tpc_ag_CH, ya.tpc_all_CH)
    ya["ag_share_YL"] = R(ya.tpc_ag_YL, ya.tpc_all_YL)
    ya["cov_raw"] = R(ya.raw_GWh, ya.tpc_ag_YL)
    ya["cov_dedup"] = R(ya.dedup_GWh, ya.tpc_ag_YL)
    ya["cov_clean"] = R(ya.cleaned_GWh, ya.tpc_ag_YL)
    return ya


def part_tpc(ev: Path) -> dict:
    tpc_csv = out_path(TP, "tpc_county_sector_monthly.csv")
    cen_csv = out_path(TP, "census_month.csv")
    ref_path = out_path(TP, "compare_annual.csv")
    ya = tpc_harness_annual(tpc_csv, cen_csv)
    # (1) the pilot's own compare.py, patched only to also write full precision
    run_dir = ev / "tpc_run"
    run_dir.mkdir(parents=True, exist_ok=False)
    patch = _patch(code_path(TP, "tools/compare.py"), run_dir / "compare_a1.py", TPC_PATCHES)
    t0 = time.time()
    with open(run_dir / "run.log", "w") as f:
        rc = subprocess.run([PY, "compare_a1.py", str(tpc_csv), str(cen_csv), str(run_dir)],
                            cwd=run_dir, stdout=f, stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).returncode
    if rc != 0:
        raise RuntimeError(f"compare_a1.py failed (rc {rc})")
    rerun_identical = sha256_file(run_dir / "compare_annual.csv") == sha256_file(ref_path)
    full = pd.read_csv(run_dir / "compare_annual_full.csv").set_index("year")
    t = Tally("tpc_scorers_vs_pilot_full_precision")
    for y in full.index:
        for c in full.columns:
            a = ya.loc[y, c]
            b = full.loc[y, c]
            t.add(f"{y}/{c}", None if pd.isna(a) else float(a), None if pd.isna(b) else float(b))
    # (2) the archive at its stored precision (4 decimals)
    ref = pd.read_csv(ref_path, dtype=str).set_index("year")
    mine = ya.copy()
    mine.index = mine.index.astype(str)
    n = same = 0
    bad = []
    for y in ref.index:
        for c in ref.columns:
            n += 1
            rs = ref.loc[y, c]
            v = mine.loc[y, c]
            ms = "" if pd.isna(v) else (str(int(v)) if c == "months_tpc" else f"{v:.4f}")
            rs = "" if pd.isna(rs) else rs
            same += ms == rs
            if ms != rs:
                bad.append({"key": f"{y}/{c}", "harness": ms, "pilot": rs})
    res = t.result(reference=rel_to_repo(ref_path), reference_sha256=sha256_file(ref_path),
                   full_precision_reference="compare_annual_full.csv written by the pilot's "
                                            "tools/compare.py (patched to add a %.17g copy)",
                   pilot_rerun_compare_annual_byte_identical=bool(rerun_identical),
                   archive_values=n, archive_identical_at_stored_precision=same,
                   archive_mismatches=bad[:20])
    res["pass"] = bool(res["pass"] and same == n and rerun_identical)
    res["patch"] = patch
    res["step"] = {"cmd": "python compare_a1.py <tpc> <census> <run_dir>", "rc": rc,
                   "seconds": round(time.time() - t0, 1), "log": rel_to_repo(run_dir / "run.log")}
    return res


# ============================================================================ gap masks
G = "head_gapfill_pilot"
CELL_COLS = ["gap", "L", "w", "t", "edge_dist"]


def _hist_matrix(M: pd.DataFrame):
    """``lib.load_matrix('hist')`` of the gap-fill pilot (same logic)."""
    dates = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    W = M[M.src == "hist"].pivot(index="month", columns="sid", values="C20").reindex(dates)
    W = W.loc[:, W.notna().sum() > 0]
    return W.to_numpy(float).T, list(W.columns), dates


def _comb_matrix(M: pd.DataFrame):
    """``lib.load_matrix('comb')`` of the gap-fill pilot (same logic)."""
    dates = pd.date_range("2012-01-01", "2026-09-01", freq="MS")
    H = M[M.src == "hist"].pivot(index="month", columns="sid", values="C20")
    N = M[M.src == "new"].pivot(index="month", columns="sid", values="C20")
    H = H[[c for c in N.columns if c in H.columns]]
    W = pd.concat([H[H.index < "2023-01-01"], N[N.index >= "2023-01-01"]])
    W = W.reindex(dates)[N.columns]
    W = W.loc[:, W.notna().sum() > 0]
    return W.to_numpy(float).T, list(W.columns), dates


def _daily_bridge_2026(daily: pd.DataFrame, sids) -> dict:
    """Verbatim logic of the pilot's ``s6_apply.daily_bridge_2026`` (Feb/Mar 2026 monthly
    means from surviving days + linear bridge + mean deviation curve), on its daily table."""
    D = daily.sort_values("src").drop_duplicates(["sid", "date"], keep="first")
    D = D[D.sid.isin(set(sids))]
    res = {}
    for sid, g in D.groupby("sid"):
        s = g.set_index("date")["head"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        curves = []
        for yr in range(2013, 2027):
            g0, g1 = pd.Timestamp(f"{yr}-02-11"), pd.Timestamp(f"{yr}-03-19")
            pre = s.reindex(pd.date_range(g0 - pd.Timedelta(days=3), g0 - pd.Timedelta(days=1)))
            post = s.reindex(pd.date_range(g1 + pd.Timedelta(days=1), g1 + pd.Timedelta(days=3)))
            if pre.notna().sum() < 2 or post.notna().sum() < 2:
                continue
            gd = pd.date_range(g0, g1)
            x = np.arange(len(gd)) + 1
            lin = pre.mean() + (post.mean() - pre.mean()) * x / (len(gd) + 1)
            if yr == 2026:
                res[sid] = (s, pd.Series(lin, index=gd))
                continue
            dev = s.reindex(gd) - lin
            if dev.notna().mean() > 0.8:
                v = dev.interpolate(limit_direction="both").to_numpy()
                curves.append(np.interp(np.linspace(0, 1, 37), np.linspace(0, 1, len(v)), v))
        if sid in res:
            s, lin = res[sid]
            shape = (np.interp(np.linspace(0, 1, len(lin)), np.linspace(0, 1, 37),
                               np.mean(curves, 0)) if len(curves) >= 3 else 0.0)
            filled = s.combine_first(lin + shape)
            feb = filled.reindex(pd.date_range("2026-02-01", "2026-02-28")).mean()
            mar = filled.reindex(pd.date_range("2026-03-01", "2026-03-31")).mean()
            res[sid] = (feb, mar, len(curves))
    return {k: v for k, v in res.items() if len(v) == 3}


def part_gap_masks() -> dict:
    """Regenerate the pilot's artificial-gap masks with harness.gaps and compare the masked
    cells (gap id, length, well, month, edge distance; same order) with the pilot's."""
    for f in ("scen.py", "s3_pilot.py", "s8_testperiod.py", "s6_apply.py", "lib.py"):
        code_path(G, f)                              # the logic copied here is the archived one
    M = pd.read_parquet(out_path(G, "out/monthly_long.parquet"))
    Y, _, _ = _hist_matrix(M)
    obs = np.isfinite(Y)
    targets = np.where(obs.mean(1) >= 0.5)[0]
    fams = {}
    for fam in ("N", "S", "E", "H"):
        ref = pd.read_parquet(out_path(G, f"out/pilot_{fam}.parquet"),
                              columns=CELL_COLS + ["real", "method"])
        ref = ref[ref.method == "linear"]
        rows = []
        for r, _mask, cells in gaps.head_masks(fam, obs, targets):
            mine = np.asarray(cells, dtype=np.int64).reshape(-1, 5)
            theirs = ref[ref.real == r][CELL_COLS].to_numpy(np.int64)
            rows.append({"real": r, "seed": gaps.seed_for(fam, r), "cells_harness": len(mine),
                         "cells_pilot": len(theirs),
                         "identical": bool(mine.shape == theirs.shape and
                                           np.array_equal(mine, theirs))})
        fams[fam] = {"realizations": rows, "identical": all(x["identical"] for x in rows),
                     "reference": rel_to_repo(out_path(G, f"out/pilot_{fam}.parquet"))}
    # T: test-period replica on the comb matrix with the Feb/Mar 2026 daily bridge
    Yc, sids, dates = _comb_matrix(M)
    daily = pd.read_parquet(out_path(G, "out/daily_clean.parquet"))
    db = _daily_bridge_2026(daily, sids)
    kf = dates.get_loc(pd.Timestamp("2026-02-01"))
    for k, (feb, mar, _n) in db.items():
        i = sids.index(k)
        for j, v in ((kf, feb), (kf + 1, mar)):
            if not np.isfinite(Yc[i, j]) and np.isfinite(v):
                Yc[i, j] = v
    oc = np.isfinite(Yc)
    t23 = dates.get_loc(pd.Timestamp("2023-01-01"))
    T = Yc.shape[1]
    tg = [i for i in range(len(sids)) if oc[i, t23:].mean() >= 0.7 and oc[i].sum() >= 60]
    ref = pd.read_parquet(out_path(G, "out/pilot_testperiod.parquet"),
                          columns=["kind", "gap", "w", "t", "real", "method"])
    ref = ref[ref.method == "router"]
    rows = []
    for r in range(3):
        _mask, cells = gaps.replica_test_period(oc, tg, r, window=(t23, T), avoid=(kf, kf + 1))
        mine = pd.DataFrame(cells, columns=["kind", "gap", "w", "t"])
        theirs = ref[ref.real == r][["kind", "gap", "w", "t"]].reset_index(drop=True)
        same = (len(mine) == len(theirs) and
                bool((mine.kind.to_numpy() == theirs.kind.to_numpy()).all()) and
                np.array_equal(mine[["gap", "w", "t"]].to_numpy(np.int64),
                               theirs[["gap", "w", "t"]].to_numpy(np.int64)))
        rows.append({"real": r, "seed": gaps.seed_for("T", r), "cells_harness": len(mine),
                     "cells_pilot": len(theirs), "identical": bool(same)})
    fams["T"] = {"realizations": rows, "identical": all(x["identical"] for x in rows),
                 "reference": rel_to_repo(out_path(G, "out/pilot_testperiod.parquet")),
                 "note": "post-origin truth (2023-01..2026-09): after-the-fact reproduction of "
                         "the generator only"}
    n = sum(len(v["realizations"]) for v in fams.values())
    ok = sum(x["identical"] for v in fams.values() for x in v["realizations"])
    return {"part": "gap_masks", "families": fams, "n_realizations": n, "n_identical": ok,
            "pass": n > 0 and ok == n}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ev = Path(argv[0])
    ev.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"criterion": f"max relative difference <= {TOL}", "parts": {}}
    res["parts"]["gapfill"] = part_gapfill()
    print("gapfill", {k: (v["n_values"], v["max_rel_diff"]) for k, v in
                      res["parts"]["gapfill"].items()}, flush=True)
    res["parts"]["tpc"] = part_tpc(ev)
    print("tpc", res["parts"]["tpc"]["n_values"], res["parts"]["tpc"]["max_rel_diff"],
          flush=True)
    res["parts"]["gap_masks"] = part_gap_masks()
    print("gap_masks", res["parts"]["gap_masks"]["n_identical"], "of",
          res["parts"]["gap_masks"]["n_realizations"], flush=True)
    res["parts"]["pseudo"] = part_pseudo(ev)
    print("pseudo", res["parts"]["pseudo"]["scorers"]["n_values"],
          res["parts"]["pseudo"]["scorers"]["max_rel_diff"], flush=True)
    res["parts"]["holdout"] = part_holdout(ev)
    print("holdout", res["parts"]["holdout"]["scorers"]["n_values"],
          res["parts"]["holdout"]["scorers"]["max_rel_diff"], flush=True)
    flat = [res["parts"]["gapfill"][k] for k in res["parts"]["gapfill"]] + \
        [res["parts"]["tpc"], res["parts"]["pseudo"]["scorers"],
         res["parts"]["holdout"]["scorers"]]
    res["n_values_total"] = int(sum(p["n_values"] for p in flat))
    res["max_rel_diff_overall"] = max(p["max_rel_diff"] for p in flat)
    res["pass"] = all(p["pass"] for p in flat) and res["parts"]["gap_masks"]["pass"]
    res["patched_reruns_identical"] = all(
        res["parts"][k]["patched_rerun"]["pass"] for k in ("pseudo", "holdout"))
    res["guard"] = {name: {"use": USE, "notes": prod.notes,
                           "manifest_version": prod.manifest["version"]}
                    for name, prod in _OPEN.items()}
    res["seconds"] = round(time.time() - t0, 1)
    code = Path(__file__).resolve()
    res["code"] = {rel_to_repo(p): sha256_file(p) for p in
                   (code, code.parent / "scorers.py", code.parent / "gaps.py")}
    out = ev / "repro_pilots.json"
    with open(out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: res[k] for k in ("n_values_total", "max_rel_diff_overall", "pass",
                                          "seconds")}))
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
