"""WP P4m shared code: the frozen cwa twin, the C0 metered Changhua forcing, and the scorers.

Pre-registered in ``P4m.metered_diagnostics.v1`` and ``P4m.recorded_arm_2023_25.v1``
(harness.prereg3, 2026-10-14T15:58Z). CPU only (CUDA hidden before torch is imported).

Conventions (fixed by the declarations):

* steps: 131 monthly steps 2012-02..2022-12 from ``h0`` (2012-01), the calibration's own;
  ``E``/``R``/``obs`` are the model's arrays without their first month;
* arms: ``A0`` the frozen model (Yunlin census only), ``A1`` plus the C0 preorigin/v3
  Changhua + in-grid Chiayi energy (all classes), ``A2`` the same without the domestic class;
  the added energy goes in as ``E_extra`` (spread kernel and lift feedback act on it);
* scores: observed months 2013-01..2022-12, training-mean datum, wells with >= 24 months;
  groups by county, model zone and model layer; 10 km well clusters for the bootstrap.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import grid_hash, sha256_file  # noqa: E402
from common.paths import DATAPREP, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402
from harness import scorers  # noqa: E402

from p4_changhua import engine, engine_tlm, inputs_p4  # noqa: E402

__version__ = "0.1.0"   # WP P4m, 2026-10-14

WP = "P4m"
PRE_DIAG = "P4m.metered_diagnostics.v1"
PRE_REC = "P4m.recorded_arm_2023_25.v1"
THETA = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
THETA_SHA = "789d0e459a3b35eb3e16002869405c889a86131815fcebfef84a9ffd8370704c"
ENGINE_V2 = DATAPREP / "changhua_abstraction" / "engine" / "v2"
C0_PRE = DATAPREP / "pump_electricity" / "preorigin" / "v3"
C0_FULL = DATAPREP / "pump_electricity" / "full" / "v3"
H1A = DATAPREP / "heads" / "L1_clean" / "v3"
PRODUCT_ROOT = DATAPREP / "changhua_forcing_diagnostics"
EVID = REPORTS_DIR / "P4m_evidence"
STAGE = EVID / "stage"
CH, YL = "彰化縣", "雲林縣"
COUNTY_KEYS = {"changhua": CH, "yunlin": YL}
SCORE_FIRST, SCORE_LAST = "2013-01", "2022-12"
MIN_N = 24
SEED = 2027
N_BOOT = 2000
CLUSTER_KM = 10.0
S_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
ZONE_NAMES = ("proximal", "mid", "distal")

_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_jdefault) + "\n")


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ============================================================================ the twin
def build_twin(member, grid, ge, **mults):
    from hydrophysics.twin.forward import build_model

    model, sc, zone3 = build_model(grid, member, "cpu")
    return engine.SparseTwin(model, sc, ge, **mults), model, sc


def setup(verbose: bool = True) -> dict:
    """The frozen cwa twin and its inputs (repository root as working directory)."""
    os.chdir(REPO)
    import torch

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; run with CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    from hydrophysics.twin.forward import load_members

    if sha256_file(THETA) != THETA_SHA:
        raise SystemExit("the cwa theta changed since the declarations")
    member = load_members([str(THETA)])[0]
    meta = member.meta
    t0 = time.perf_counter()
    inp = inputs_p4.load_inputs(meta)
    if verbose:
        log(f"inputs loaded in {time.perf_counter() - t0:.1f}s")
    with np.load(ENGINE_V2 / "engine_arrays.npz", allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    grid = inp.grid
    ge = inp.ground_elev.numpy()
    h0 = inp.initial_heads(0, n_layers=4).numpy()
    E_cls = np.stack([inp.E_by_class[c] for c in inp.E_by_class])
    E = E_cls.sum(axis=0)[:, 1:]
    R = inp.recharge_field.numpy()[:, 1:]
    tw, model, sc = build_twin(member, grid, ge)
    sids = np.array(inp.sids)
    if not (np.array_equal(sids, arr["sids"]) and np.array_equal(inp.obs_idx, arr["obs_idx"])
            and np.array_equal(inp.obs_layer, arr["obs_layer"])):
        raise SystemExit("well set differs from engine/v2")
    gh = grid_hash(grid.nx, grid.ny, grid.dx, grid.x0, grid.y0, grid.mask)
    dates = np.array([str(x.date())[:7] for x in inp.dates])
    S = {"member": member, "meta": meta, "inp": inp, "grid": grid, "grid_hash": gh, "ge": ge,
         "h0": h0, "E": E, "R": R, "E_months": dates, "steps": dates[1:], "tw": tw,
         "model": model, "sc": sc, "oi": inp.obs_idx, "ol": inp.obs_layer, "sids": sids,
         "obs": inp.obs_h[:, 1:].astype(np.float64), "well_xy": np.asarray(inp.well_xy),
         "well_county": arr["well_county"], "cell_county": arr["cell_county"],
         "zone3": arr["zone3"], "H_hindcast_v2": arr["H_hindcast"], "arrays": arr,
         "classes_model": list(inp.E_by_class)}
    S["well_zone"] = S["zone3"][S["oi"]]
    return S


def rollout(S: dict, E_extra=None, h0=None, *, E=None, R=None, return_rates=False):
    """Stepwise rollout of the frozen twin (apex pinned to the calibration's h0)."""
    return S["tw"].rollout(S["h0"] if h0 is None else h0, E=S["E"] if E is None else E,
                           R=S["R"] if R is None else R, E_extra=E_extra,
                           apex_from=S["h0"], return_rates=return_rates)


def at_wells(S: dict, H: np.ndarray) -> np.ndarray:
    return H[S["ol"], S["oi"], 1:]


# ============================================================================ C0 forcing
def c0_open(variant: str, use: str):
    d = {"preorigin": C0_PRE, "full": C0_FULL}[variant]
    return guard.open_product(d, use)


def c0_cells(variant: str, use: str, *, counties=("changhua", "chiayi"),
             nodomestic: bool = False, first: str, last: str, grid_hash_expected: str):
    """(A, T) kWh/month of the C0 per-county class arrays summed over ``counties`` and
    classes (domestic dropped if ``nodomestic``), months ``first``..``last`` inclusive."""
    p = c0_open(variant, use)
    with np.load(p.path("cells_by_class.npz"), allow_pickle=False) as z:
        if str(z["grid_hash"]) != grid_hash_expected:
            raise SystemExit("C0 grid differs from the twin's grid")
        dates = [str(x) for x in z["dates"]]
        cls = [str(x) for x in z["classes"]]
        cty = [str(x) for x in z["counties"]]
        i0, i1 = dates.index(first), dates.index(last) + 1
        keep_c = [i for i, c in enumerate(cls) if not (nodomestic and c == "domestic")]
        v = z["value_by_county"]
        out = np.zeros(v.shape[2:3] + (i1 - i0,))
        for c in counties:
            k = cty.index(c)
            out += v[k][keep_c][:, :, i0:i1].sum(axis=0)
        flags = {}
        for c in counties:
            k = cty.index(c)
            f = z["flag_by_county"][k][keep_c][:, :, i0:i1]
            flags[c] = {int(u): int((f == u).sum()) for u in np.unique(f)}
    if not np.all(np.isfinite(out)):
        raise SystemExit("non-finite C0 values")
    return out, dates[i0:i1], {"notes": p.notes, "flags": flags, "classes": cls,
                               "classes_used": [cls[i] for i in keep_c]}


def c0_county_totals(variant: str, use: str, counties, scope: str, cls: str, first: str,
                     last: str) -> float:
    """Sum (kWh) of C0 county_class_monthly ``value`` over the months (EC2 cross-check)."""
    import pandas as pd

    p = c0_open(variant, use)
    df = pd.read_parquet(p.path("county_class_monthly.parquet"),
                         columns=["county", "scope", "class", "date", "value"])
    dd = df["date"].astype(str).str[:7]
    sel = (df["county"].isin(list(counties)) & (df["scope"] == scope) & (df["class"] == cls)
           & (dd >= first) & (dd <= last))
    return float(df.loc[sel, "value"].sum())


def changhua_forcing(S: dict, nodomestic: bool = False):
    """E_CH for the 131 steps 2012-02..2022-12 (A1, or A2 if ``nodomestic``)."""
    e, dates, info = c0_cells("preorigin", "model_design", nodomestic=nodomestic,
                              first="2012-01", last="2022-12",
                              grid_hash_expected=S["grid_hash"])
    if list(dates) != list(S["E_months"]):
        raise SystemExit("C0 months differ from the model's")
    return e[:, 1:], e, info


# ============================================================================ scoring
def step_mask(S: dict, first: str = SCORE_FIRST, last: str = SCORE_LAST) -> np.ndarray:
    st = S["steps"]
    return (st >= first) & (st <= last)


def groups(S: dict) -> dict[str, np.ndarray]:
    wc, wz, wl = S["well_county"], S["well_zone"], S["ol"]
    g = {"all": np.ones(len(wc), bool)}
    for key, name in (("CH", CH), ("YL", YL)):
        g[key] = wc == name
        for z, zn in enumerate(ZONE_NAMES):
            g[f"{key}_zone_{zn}"] = (wc == name) & (wz == z)
        for lay in range(4):
            g[f"{key}_layer{lay}"] = (wc == name) & (wl == lay)
    return g


def masked_obs(S: dict, tmask: np.ndarray) -> np.ndarray:
    o = S["obs"].copy()
    o[:, ~tmask] = np.nan
    return o


def eligible(obs_m: np.ndarray) -> np.ndarray:
    return np.isfinite(obs_m).sum(axis=1) >= MIN_N


def per_well_stats(obs_m: np.ndarray, pred: np.ndarray) -> dict:
    """Per well (NaN where < MIN_N months): anomaly MSE, std ratio, Pearson r, mean resid."""
    W = obs_m.shape[0]
    mse, amp, cor, mres, n = (np.full(W, np.nan) for _ in range(5))
    for w in range(W):
        m = np.isfinite(obs_m[w])
        n[w] = m.sum()
        if m.sum() < MIN_N:
            continue
        o, p = obs_m[w, m], pred[w, m]
        r = o - p
        mres[w] = r.mean()
        mse[w] = float(np.mean((r - r.mean()) ** 2))
        so = np.std(o)
        amp[w] = np.std(p) / so if so > 0 else np.nan
        cor[w] = np.corrcoef(o, p)[0, 1] if np.std(p) > 0 and so > 0 else np.nan
    return {"mse": mse, "amp": amp, "corr": cor, "mean_resid": mres, "n": n}


def group_metrics(pw: dict, gmask: np.ndarray) -> dict:
    sel = gmask & np.isfinite(pw["mse"])
    if not sel.any():
        return {"n_wells": 0}
    return {"n_wells": int(sel.sum()), "n_months": int(pw["n"][sel].sum()),
            "anomaly_rmse": float(np.sqrt(np.mean(pw["mse"][sel]))),
            "amplitude_ratio": float(np.median(pw["amp"][sel])),
            "corr_median": float(np.nanmedian(pw["corr"][sel]))}


def check_against_harness(obs_m, pred, gmask) -> float:
    """|anomaly RMSE (ours) - harness.scorers.anomaly_rmse| on a group (agreement check)."""
    sel = gmask & eligible(obs_m)
    a = scorers.anomaly_rmse(obs_m[sel], pred[sel], agg="rms", min_n=MIN_N)["rmse"]
    pw = per_well_stats(obs_m[sel], pred[sel])
    b = float(np.sqrt(np.nanmean(pw["mse"])))
    c = scorers.amplitude_ratio(obs_m[sel], pred[sel], min_n=MIN_N)
    return max(abs(a - b), abs(c - float(np.nanmedian(pw["amp"]))))


def clusters(S: dict) -> np.ndarray:
    xy = S["well_xy"] / (CLUSTER_KM * 1000.0)
    key = np.floor(xy[:, 0]).astype(int) * 100000 + np.floor(xy[:, 1]).astype(int)
    _, lab = np.unique(key, return_inverse=True)
    return lab


def boot_indices(cl: np.ndarray, gmask: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    """Cluster bootstrap draws (lists of well indices) within ``gmask``."""
    rng = np.random.default_rng(seed)
    wells = np.flatnonzero(gmask)
    ucl = np.unique(cl[wells])
    members = {c: wells[cl[wells] == c] for c in ucl}
    for _ in range(n):
        pick = rng.choice(ucl, ucl.size, replace=True)
        yield np.concatenate([members[c] for c in pick])


def boot_interval(stat, cl, gmask, n: int = N_BOOT, seed: int = SEED, q=(5, 95)) -> dict:
    vals = []
    for idx in boot_indices(cl, gmask, n, seed):
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return {"p05": float(np.percentile(vals, q[0])), "p95": float(np.percentile(vals, q[1])),
            "n_draws": int(vals.size), "n_clusters": int(np.unique(cl[gmask]).size),
            "label": "cluster bootstrap (10 km), in-sample, UNCALIBRATED"}


def drmse_boot(pw_a: dict, pw_b: dict, cl, gmask) -> dict:
    """Interval of anomaly RMSE(a) - RMSE(b) over cluster resamples of the group's wells."""
    ok = gmask & np.isfinite(pw_a["mse"]) & np.isfinite(pw_b["mse"])

    def stat(idx):
        return float(np.sqrt(np.mean(pw_a["mse"][idx])) - np.sqrt(np.mean(pw_b["mse"][idx])))

    est = stat(np.flatnonzero(ok))
    return {"estimate": est, **boot_interval(stat, cl, ok)}


def anomaly_sse(obs_m: np.ndarray, pred: np.ndarray, wsel: np.ndarray) -> np.ndarray:
    """Per-well sum of squared anomaly residuals (training-mean datum) for wells in wsel."""
    out = np.full(obs_m.shape[0], np.nan)
    for w in np.flatnonzero(wsel):
        m = np.isfinite(obs_m[w])
        r = obs_m[w, m] - pred[w, m]
        out[w] = float(((r - r.mean()) ** 2).sum())
    return out


def amp_ratio_group(obs_m, pred, gsel) -> float:
    vals = []
    for w in np.flatnonzero(gsel):
        m = np.isfinite(obs_m[w])
        vals.append(np.std(pred[w, m]) / np.std(obs_m[w, m]))
    return float(np.median(vals))


def seasonal_cv(series: np.ndarray, months: np.ndarray) -> float:
    """Month-of-year profile sd / mean of a monthly series (months 'YYYY-MM')."""
    mo = np.array([int(m[5:7]) for m in months])
    prof = np.array([series[mo == k].mean() for k in range(1, 13)])
    return float(prof.std() / prof.mean())


def git_safe_rel(p) -> str:
    return rel_to_repo(Path(p))


def tlm_response(S: dict, H_path: np.ndarray, Es_path: np.ndarray, dE: np.ndarray) -> np.ndarray:
    """Tangent-linear head response at the wells (W, T) to an energy perturbation ``dE``
    (unspread kWh/month, (A, T) or (A, T, J)) about the path ``H_path`` driven by spread
    energy ``Es_path``."""
    tw = S["tw"]
    dEs = tw.spread(dE) if dE.ndim == 2 else np.stack(
        [tw.spread(dE[..., j]) for j in range(dE.shape[-1])], axis=-1)
    return engine_tlm.tlm_rollout(tw, H_path, Es_path, dEs=dEs, obs=(S["ol"], S["oi"]))


def math_isfinite(x) -> bool:
    return x is not None and math.isfinite(float(x))
