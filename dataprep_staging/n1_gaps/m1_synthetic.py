"""M1 gate 0: does a Ske-capped leveling column recover the subsidence a cut avoids?

Synthetic recovery check, model-only (post hoc, not a test; no observation value is read).
It reuses step A1's machinery (``n1_policy/run_step_a.py``, ``policy_lib.py``) on R0
(``stage3_datum_gate_cwa``):

- truth column: R0's leveling-fitted zonal column with each zone's elastic skeletal
  storage replaced by ``Ske = 0.5 x sum over layers of S`` (the flow theta's own
  storativity, area-weighted over the column zone; ``calibrate_coupled.theta_ske_caps``
  with kappa 0.5), its other parameters kept;
- truth heads: ``H_k = H0 + k (H - H0)`` (H0 = zero pumping energy), head damping
  ``k`` in {1, 2}: the true head response to pumping is k times the model's;
- synthetic leveling: the truth column on H_k at the real benchmark cells and survey
  months 2012-21 (A1's ``survey_mask``: the panel's DATES and coordinates only, the same
  read as A1), re-zeroed at each benchmark's first survey, plus N(0, 0.5 cm);
- fits on the model's own heads H (``calibrate_coupled._fit``, h_pc0 guard 365 d, as A1):
  (a) uncapped, (b) ``--ske-cap-from-theta 1`` (Ske <= 1 x sum S per zone);
- lever cut30 (irrigation energy x 0.7 from 2013-01) in both systems; reported per column
  zone (and zone x county, fan): fitted / true avoided subsidence 1, 3 and 7 years into the
  cut (2013-12, 2015-12, 2019-12), plus A1's 2021-12 for reference.

Pass (declared before the run): the capped ratio lies in [0.67, 1.5] for every column zone
at both k (at each of the three horizons).

Usage (CPU only): CUDA_VISIBLE_DEVICES= python dataprep_staging/n1_gaps/m1_synthetic.py
    [--epochs 2000]
Output: a NEW file results/dataprep/reports/N1_gaps_2026-10-24/M1/m1_synthetic_<utc>.json
of the main repository (``DATAPREP_REPO``, by default the checkout that owns this
worktree's git directory). This worktree's ``hydrophysics`` is imported, not the installed one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = Path(__file__).resolve()
CODE_ROOT = HERE.parents[2]                     # the checkout whose code runs


def _main_repo() -> Path:
    """The repository that holds results/ (gitignored, main checkout only)."""
    if os.environ.get("DATAPREP_REPO"):
        return Path(os.environ["DATAPREP_REPO"]).resolve()
    try:
        common = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        common_p = Path(common)
        if not common_p.is_absolute():
            common_p = CODE_ROOT / common_p
        return common_p.resolve().parent
    except (OSError, subprocess.CalledProcessError):
        return CODE_ROOT


os.environ.setdefault("DATAPREP_REPO", str(_main_repo()))
os.environ.setdefault("HYDROMIND_GW_DATA",
                      str(Path(os.environ["DATAPREP_REPO"]) / "chou-shui-data" / "data"))
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(1, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from n1_policy import policy_lib as PL  # noqa: E402
from n1_policy.run_step_a import CUT, CUT_FROM, cut_energy, survey_mask  # noqa: E402

from n1_gaps.gaps_lib import OUT_ROOT, log, rel_to_repo, source, write_new  # noqa: E402

SEED = 20261024
KS = (1.0, 2.0)
KAPPA_TRUE = 0.5
KAPPA_FIT = 1.0
NOISE_M = 0.005
HORIZONS = {"1y": 1, "3y": 3, "7y": 7}
A1_END = pd.Period("2021-12", "M")
BAND = (0.67, 1.5)
R = "results/twin_runs/stage3_datum_gate_cwa"
THETA = f"{R}/stage3_theta.json"
COLUMN = f"{R}/coupled_leveling/vep_zonal_leveling.json"
ZONES = ("proximal", "mid", "distal")


def code_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_dirty() -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(CODE_ROOT), "status", "--short"],
                             capture_output=True, text=True, check=True).stdout
        return [ln for ln in out.splitlines() if ln.strip()]
    except (OSError, subprocess.CalledProcessError):
        return []


def column_sim(params: list[dict], heads: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """A zonal column's subsidence (A, T) m on heads (L, A, T), float64, layer-mean driver
    (the recurrence ``forward.compaction`` and ``calibrate_coupled._predict`` run)."""
    import torch

    from hydrophysics.twin.compaction import column_compaction

    drv = torch.as_tensor(heads, dtype=torch.float64).mean(dim=0)
    out = torch.zeros_like(drv)
    for z, p in enumerate(params):
        sel = torch.as_tensor(zone == z)
        if bool(sel.any()):
            out[sel] = column_compaction(drv[sel], p)
    return out.numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    a = ap.parse_args(argv)
    import torch

    import hydrophysics
    import hydrophysics.twin.calibrate_coupled as cc
    from hydrophysics.twin.zone_county import DEFAULT_MAP, column_zone_map, county_flags_for_grid

    repo = Path(os.environ["DATAPREP_REPO"])
    os.chdir(repo)
    out_path = OUT_ROOT / "M1" / (
        f"m1_synthetic_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; results are never overwritten")
    log(f"code {hydrophysics.__file__} (head {code_head()[:12]}); results in {repo}")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    t_start = time.perf_counter()
    cc.HPC0_GUARD_DAYS = 365.0
    tm = PL.TwinModel(str(repo / THETA), str(repo / COLUMN))
    H, Hc = tm.heads(), tm.heads(cut_energy(tm))
    H0 = tm.heads(np.zeros_like(tm.energy()))
    log(f"rollouts done in {time.perf_counter() - t_start:.0f} s")
    cells, M = survey_mask(tm)
    zone = column_zone_map(tm.inp.grid, tm.meta) % 3
    ch = np.asarray(county_flags_for_grid(tm.inp.grid, DEFAULT_MAP)[0], bool)
    months = list(PL.MONTHS)
    hz = {k: months.index(CUT_FROM + 12 * y - 1) for k, y in HORIZONS.items()}
    hz["a1_2021-12"] = months.index(A1_END)

    sum_s = cc.theta_ske_caps(tm.member.theta, tm.meta, tm.inp.grid, zone, 3, 1.0)
    caps_fit = [KAPPA_FIT * s for s in sum_s]
    r0 = tm.col_meta["zonal"]
    truth = [{**{k: float(p[k]) for k in ("log_ske", "log_skv", "log_tau", "h_pc0")},
              "log_ske": math.log(KAPPA_TRUE * s)} for p, s in zip(r0, sum_s, strict=True)]
    log("sum S per zone " + ", ".join(f"{z} {s:.3e}" for z, s in zip(ZONES, sum_s,
                                                                      strict=True))
        + "; R0 Ske " + ", ".join(f"{math.exp(p['log_ske']):.3e}" for p in r0))

    groups = {z: zone == i for i, z in enumerate(ZONES)}
    groups.update({f"{z}_{c}": (zone == i) & (ch == (c == "changhua"))
                   for i, z in enumerate(ZONES) for c in ("yunlin", "changhua")})
    groups["fan"] = np.ones_like(zone, dtype=bool)
    t32 = torch.float32
    Ht = torch.tensor(H[:, cells, :].transpose(1, 0, 2), dtype=t32)
    Mt = torch.tensor(M, dtype=t32)
    Zt = torch.tensor(zone[cells], dtype=torch.long)
    rng = np.random.default_rng(SEED)

    def fit(obs_rows: np.ndarray, caps: list[float] | None) -> tuple[list[dict], float, float]:
        cc.SKE_CAPS = caps
        try:
            model = cc._make("zonal", "cpu", 3)
            t0 = time.perf_counter()
            loss = cc._fit(model, Ht, torch.tensor(obs_rows, dtype=t32), Mt, Zt, a.epochs,
                           a.lr, rezero=True)
        finally:
            cc.SKE_CAPS = None
        return cc.column_json(model)["zonal"], loss, time.perf_counter() - t0

    out = {"what": "M1 gate 0: synthetic recovery of cut30 avoided subsidence by an "
                   "uncapped and a Ske-capped (kappa 1) leveling column; post hoc, "
                   "model-only, not a test",
           "pass_rule": f"capped fitted/true avoided in [{BAND[0]}, {BAND[1]}] for every "
                        "column zone (proximal, mid, distal) at k = 1 and 2, at 1, 3 and 7 "
                        "years into the cut",
           "design": {"seed": SEED, "ks": list(KS), "kappa_true": KAPPA_TRUE,
                      "kappa_fit": KAPPA_FIT, "noise_m": NOISE_M, "epochs": a.epochs,
                      "lr": a.lr, "hpc0_guard_days": 365.0, "lever": "cut30",
                      "cut": CUT, "cut_from": str(CUT_FROM),
                      "horizons": {k: str(months[v]) for k, v in hz.items()},
                      "survey_geometry": "A1 survey_mask: panel survey dates and benchmark "
                                         "coordinates 2012-21 only, no observed value",
                      "n_benchmarks": int(len(cells)), "n_survey_entries": int(M.sum())},
           "zones": {"names": list(ZONES),
                     "n_cells": {z: int((zone == i).sum()) for i, z in enumerate(ZONES)},
                     "sum_S_layers": dict(zip(ZONES, sum_s, strict=True)),
                     "ske_cap_fit": dict(zip(ZONES, caps_fit, strict=True)),
                     "ske_true": {z: math.exp(p["log_ske"]) for z, p in zip(ZONES, truth,
                                                                              strict=True)},
                     "ske_R0_column": {z: math.exp(p["log_ske"]) for z, p in zip(ZONES, r0,
                                                                                 strict=True)}},
           "k": {}}
    for k in KS:
        Hk, Hkc = H0 + k * (H - H0), H0 + k * (Hc - H0)
        S_true, S_true_c = column_sim(truth, Hk, zone), column_sim(truth, Hkc, zone)
        rows = S_true[cells]
        first = M.argmax(axis=1)
        rows = rows - rows[np.arange(len(cells)), first][:, None]
        obs = np.where(M, rows + rng.normal(0, NOISE_M, rows.shape), 0.0)
        av_true = S_true - S_true_c
        # elastic part of the true avoided subsidence: Ske (h_cut - h) of the layer mean
        dh = (Hkc - Hk).mean(axis=0)
        ske_cell = np.array([math.exp(truth[z]["log_ske"]) for z in zone])
        av_true_el = ske_cell[:, None] * dh
        res_k = {}
        for label, caps in (("uncapped", None), ("capped_kappa1", caps_fit)):
            params, loss, secs = fit(obs, caps)
            av_fit = column_sim(params, H, zone) - column_sim(params, Hc, zone)
            per = {}
            for hname, e in hz.items():
                per[hname] = {}
                for g, sel in groups.items():
                    tt, ff = float(av_true[sel, e].sum()), float(av_fit[sel, e].sum())
                    per[hname][g] = {
                        "true_avoided_m_sum": tt, "fitted_avoided_m_sum": ff,
                        "true_elastic_m_sum": float(av_true_el[sel, e].sum()),
                        "ratio": ff / tt if abs(tt) > 1e-12 else float("nan"),
                        "n_cells": int(sel.sum())}
            res_k[label] = {
                "loss": loss, "fit_seconds": secs,
                "fitted_columns": {z: {**p, "ske": math.exp(p["log_ske"]),
                                       "ske_over_sum_S": math.exp(p["log_ske"]) / s,
                                       "tau_days": math.exp(p["log_tau"])}
                                   for z, p, s in zip(ZONES, params, sum_s, strict=True)},
                "by_horizon": per}
            log(f"k={k:g} {label}: loss {loss:.3e} in {secs:.0f} s; ratios "
                + "; ".join(f"{h} " + " ".join(f"{z} {per[h][z]['ratio']:.2f}"
                                                for z in ZONES + ('fan',))
                            for h in HORIZONS))
        out["k"][f"{k:g}"] = res_k

    checks = {}
    for k in KS:
        for h in HORIZONS:
            for z in ZONES:
                r = out["k"][f"{k:g}"]["capped_kappa1"]["by_horizon"][h][z]["ratio"]
                checks[f"k{k:g}_{h}_{z}"] = {"ratio": r,
                                             "in_band": bool(BAND[0] <= r <= BAND[1])}
    out["checks_capped"] = checks
    out["verdict"] = "pass" if all(c["in_band"] for c in checks.values()) else "fail"
    out["provenance"] = {
        "script": "dataprep_staging/n1_gaps/m1_synthetic.py",
        "code_head": code_head(), "code_uncommitted": code_dirty(),
        "hydrophysics": str(Path(hydrophysics.__file__).parent),
        "inputs": [source(repo / THETA), source(repo / COLUMN)],
        "runtime_s": time.perf_counter() - t_start,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "post hoc, model-only, not a test; reads no observation value"}
    write_new(out_path, out)
    log(f"verdict {out['verdict']}; {rel_to_repo(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
