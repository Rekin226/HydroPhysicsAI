"""FE.encompass_2026-12.v1: the synthetic decidability study (model-only, CPU, no observation).

Scored quantity: lambda_P in y = lambda_P x_P + lambda_C x_C + e within wells (fe_lib), with
y = obs - STAT, x_P = PHYS - STAT, x_C = CLIM - STAT. In every world E04's registered layer is
RE-RUN on the world's own history (stat_fv.e04_frame, as FV.stat.v2 does), so STAT and CLIM
are what E04 would have forecast in that world.

No post-2022 data in the DESIGN (U7, lead's ruling). No real contrast enters any arm or any
G0 condition: the real E02 / E04 predictions are read for the well ids and E04's per-well
origin months only. Every physics contrast in the study is a PHYSICS-WORLD ANALOGUE built in
FV v2's forward worlds from 2012-22 inputs only (the forwards start from the 2022-12 origin;
PHYS's datum is fitted on 2012-22 months). Every synthetic error is scaled by FV v1's
registered pre-origin noise inputs (noise_ar1.csv, sha 73965a08: per-well sd and lag-1 of the
page model's 2012-01..2022-12 fitted residuals, window_end 2022-12). The real contrasts enter
only at scoring time (score_fe: the informative gate on the scored rows, then the fit).

Arms (worlds per cell in brackets):

* PHYS (gating; [250 per truth]; built first): FV.decidability.v2's 20 batch forwards (truths
  R0, Q0, R1c, X5R1 x 250 worlds) at the wells' cell and layer. truth = h_T,k + FV v2-rule
  noise (noise_ar1 sd and phi, independent between wells) over 2012-01..2027-10; E04's layer
  re-run; PHYS = h_ZR0 (fv_zr0.npz) + E02's datum fitted on the world's observed 2012-22 months
  (obs_mask.csv). In the window this is FV v2 reading 1's world. Cells: truth x {full, mcar,
  w100}. Power at lambda in {1, 0.5, 0.25} (lambda-world = obs - (1 - lambda)(PHYS - STAT),
  an exact replicate shift). Each full-variant world also yields its analogue contrast
  dC = PHYS - CLIM, x_P, x_C, which feed the arms below and gate (e).
* AR (gating; lambda_P = 0; [K_AR]): FV.stat.v2 AR worlds (``stat_v2.world``) x sd; E04's
  layer re-run; PHYS_world = CLIM_world + dC_j, dC_j a PHYS-world analogue (world j assigned
  deterministically). Cells: FV.stat.v2's nine + basin (share 0.5, both counties) + mcar +
  w100.
* PE, persistence error (gating; [K_PE]): the history is AR(1) at rho_hist (what E04's
  re-run layer fits), the months after each well's origin continue at rho_win: cells
  (0.93 -> 0.80) and (0.80 -> 0.93). PHYS_world = CLIM_world + dC_j + s(a0, h) with the
  origin-anomaly shapes s in {0, 0.5 a0 0.6^h, 0.5 a0 0.99^h} (a0 = the world's anomaly at
  the well's origin, h = months after it). Six gated (cell, shape) combinations.
* SIGMA (report-only; [K_SIG]): y = x_P,j + sd x AR world on the window (true lambda_P 1,
  lambda_C 0), the analogue regressors of world j: interval calibration.

G0 (:data:`G0_RULE`). Optimistic and stated: history fully observed; no drift or datum step;
rain at climatology after 2022-12 in PHYS worlds, so E04's re-run layer sees little
real-weather persistence there and PHYS-arm power is conditional on that.

CLI (repository root, hydro env):
  decide_fe.py smoke --out <dir outside results/>        FABRICATED inputs; prints 'ran ok'
  decide_fe.py run --seed N --prereg-id FE.encompass_2026-12.v1   once, after registration
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_stat_v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fe_lib as L  # noqa: E402
import stat_v2 as S  # noqa: E402  (FV.stat.v2, registered a15e468d; never edited)

ST, O, V2 = S.ST, S.O, S.V2  # noqa: E741
VERSION = "3.0.0"
GOV = "dataprep_staging/gov_heads_window/heads_window.py"
FROZEN = (L.LIB, L.DECIDE, L.SCORER, *S.FROZEN, GOV, "dataprep_staging/e02d/score_e02d.py",
          "dataprep_staging/e02d/e02dlib.py", "dataprep_staging/e02d/e02dio.py")
TEST_PREFIX = "TEST."             # the only registration ids injected inputs may run under

# ----------------------------------------------------------------------------- design
RHO_GRID, SHARE_GRID = S.RHO_GRID, S.SHARE_GRID          # FV.stat.v2's AR grid, unchanged
K_AR = 500
K_PE = 500
K_PHYS = V2.K_TRUTH               # 250 worlds per truth (all FV v2 worlds)
K_SIG = 200
B_STUDY = L.BOOT_B                # the scorer's bootstrap size in every world
MCAR = 0.25                       # share of window months missing at random ('mcar' cells)
N_SUB = 100                       # wells kept in 'w100' cells
BASIN_SHARE = 0.5
PE_SHARE = 0.15
PE_SHAPES = {"none": None, "fast": 0.6, "slow": 0.99}    # s = 0.5 a0 r^h (reviewer's forms)
PE_AMP = 0.5
LAMBDA_POINTS = (1.0, 0.5, 0.25)
Q_REPORT = 0.90                   # report-only: q90 of U_P under the AR arm (the old lambda_0)
TRUTHS = tuple(O.TAGS)            # R0, Q0, R1c, X5R1
VARIANTS = ("full", "mcar", "w100")
STREAMS = {"ar": 72, "sig": 73, "boot_sig": 74, "boot_ar": 75, "boot_phys": 76, "score": 77,
           "mask_ar": 78, "subset_ar": 79, "phys_noise": 80, "mask_phys": 81,
           "subset_phys": 82, "pe": 83, "pe_window": 84, "boot_pe": 85}
POOL_OFFSET = {"ar": 0, "pe": 7919, "sig": 15881}      # world j = (offset + cell K + k) mod N
SMOKE_SEED = 61120029
REFUSED_SEEDS = {SMOKE_SEED: "FE's own engineering smoke seed", **S.REFUSED_SEEDS}
G0 = {"p_inf_ar_max": 0.075, "p_inf_ar_cell_max": 0.15,
      "p_inf_pe_max": 0.075, "p_inf_pe_cell_max": 0.15,
      "p_inf_phys_min": 0.80, "p_inf_phys_cell_min": 0.60,
      "p_enc_phys_max": 0.10, "p_enc_phys_cell_max": 0.20,
      "p_enc_ar_min": 0.80, "x_partial_rms_min_m": L.X_RMS_MIN_M,
      "max_bad_replicates": L.MAX_BAD_REPLICATES, "delta": L.DELTA}
G0_RULE = (
    "FE is GATING iff (i) FE's registration was created on or before 2026-11-25T23:59:59Z "
    "(review B2 hard rule), (ii) the study was written under FE's registration before "
    "2026-11-30T16:00Z, and (iii) ALL hold (rates: equal-weight mean over cells, 'cell' = "
    "every cell): (a) AR arm P(L_P > 0) <= 0.075 pooled and <= 0.15 per cell; (a2) PE arm "
    "P(L_P > 0) <= 0.075 pooled and <= 0.15 per (cell, shape); (b) PHYS arm at lambda = 1 "
    "P(L_P > 0) >= 0.80 pooled and >= 0.60 per cell; (c) PHYS arm at lambda = 1 P(reading "
    "stat_encompasses) <= 0.10 pooled and <= 0.20 per cell; (d) AR arm P(reading "
    "stat_encompasses, U_P < 0.25) >= 0.80 pooled; (e) the median over the PHYS full-variant "
    "worlds of the analogue partial contrast RMS (x_P net of x_C, every window month) >= "
    "0.05 m; (f) no world has more than 1 % degenerate replicates. Otherwise REPORT-ONLY. "
    "Report-only: lambda = 0.5 and 0.25 power, median lambda_P per truth, the one-contrast "
    "lambda in the AR and PE arms, the q90 of U_P under the AR arm, interval calibration (AR, "
    "SIGMA), every per-cell rate and Monte Carlo SE. No real contrast enters any condition.")


def rng_for(stream: int, cell: int, k: int, *, seed: int) -> np.random.Generator:
    return S.rng_for(stream, cell, k, seed=seed)


def check_seed(seed) -> int:
    return L.check_seed(seed, REFUSED_SEEDS)


def ar_cells() -> list[dict]:
    """FV.stat.v2's nine cells plus the declared basin, mcar and w100 cells (all gating)."""
    out = [{**c, "kind": "grid"} for c in S.cells()]
    n = len(out)
    out += [{"cell": n, "rho": 0.86, "share": BASIN_SHARE, "kind": "basin"},
            {"cell": n + 1, "rho": 0.86, "share": 0.15, "kind": "mcar"},
            {"cell": n + 2, "rho": 0.86, "share": 0.15, "kind": "w100"}]
    return out


def pe_cells() -> list[dict]:
    return [{"cell": 0, "rho_hist": 0.93, "rho_win": 0.80},
            {"cell": 1, "rho_hist": 0.80, "rho_win": 0.93}]


def phys_cells() -> list[dict]:
    return [{"cell": i * len(VARIANTS) + j, "truth": t, "truth_index": i, "kind": v}
            for i, t in enumerate(TRUTHS) for j, v in enumerate(VARIANTS)]


def sig_cells() -> list[dict]:
    return [{**c, "kind": "grid"} for c in S.cells()]


# ----------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    ids: list
    county: np.ndarray            # Yunlin / Changhua by station code (81 / 66)
    last_fit: np.ndarray          # index in stat_fv.AXIS of each well's E04 origin month
    sd: np.ndarray                # (W,) pre-origin residual sd (noise_ar1.csv)
    phi: np.ndarray               # (W,) pre-origin residual lag-1 (noise_ar1.csv)
    cells: np.ndarray             # (W,) forward cell (wells_158.csv)
    layers: np.ndarray            # (W,) forward layer
    fitmask: np.ndarray           # (W, 132) observed 2012-22 months (obs_mask.csv)
    source: dict

    def layout(self, basin: bool = False):
        county = np.full(len(self.ids), "basin") if basin else self.county
        return ST.E04Layout(ids=list(self.ids), county=county, last_fit=self.last_fit)


def _month(s) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def _verified(rel: str, sha: str, repo: Path = REPO) -> Path:
    p = Path(repo) / rel
    if not p.is_file() or O.sha256(p) != sha:
        raise L.EngineeringFailure(f"{rel} missing or not the registered file")
    return p


def load_inputs(repo: Path = REPO) -> Inputs:
    """The shared well ids and E04's per-well origin months (from the frozen predictions; no
    predicted value is used), and FV v1's registered pre-origin inputs (noise scale, datum
    months, well cells), each sha256-verified. No head and no real contrast is read."""
    ids_of, org = {}, None
    for tag, d, sha in (("e02", L.E02_DIR, L.E02_PRED_SHA), ("e04", L.E04_DIR, L.E04_PRED_SHA)):
        f = pd.read_csv(_verified(f"{d}/predictions.csv", sha, repo),
                        dtype={"station_id": str}, usecols=["station_id", "prediction_origin"])
        ids_of[tag] = set(f["station_id"])
        if tag == "e04":
            if f.groupby("station_id")["prediction_origin"].nunique().max() != 1:
                raise L.EngineeringFailure("an E04 well has more than one prediction origin")
            org = f.drop_duplicates("station_id").set_index("station_id")["prediction_origin"]
    ids = sorted(ids_of["e02"] & ids_of["e04"])
    if len(ids) != L.N_SHARED:
        raise L.EngineeringFailure(f"E02 and E04 share {len(ids)} wells, not {L.N_SHARED}")
    org = org.loc[ids]
    last = np.array([ST.AXIS.get_loc(pd.Timestamp(str(o)[:7] + "-01")) for o in org])
    if np.any(last >= ST.T_FIT):
        raise L.EngineeringFailure("an E04 origin lies inside the window")
    county = np.array([L.county_of(s) for s in ids])
    if not set(county) <= {"Yunlin", "Changhua"}:
        raise L.EngineeringFailure("a shared well is outside Yunlin / Changhua")
    noise = pd.read_csv(_verified(L.NOISE_CSV, L.NOISE_SHA, repo),
                        dtype={"station_id": str}).set_index("station_id")
    wells = pd.read_csv(_verified(L.WELLS_CSV, L.WELLS_SHA, repo),
                        dtype={"station_id": str}).set_index("station_id")
    if not set(ids) <= set(noise.index) or not set(ids) <= set(wells.index):
        raise L.EngineeringFailure("noise_ar1.csv / wells_158.csv lack a shared well")
    all_ids = sorted(noise.index)
    mask158 = O.read_mask(_verified(L.MASK_CSV, L.MASK_SHA, repo), all_ids)
    fit = mask158[[all_ids.index(s) for s in ids]]
    sd = noise.loc[ids, "sd_m"].to_numpy(float)
    phi = noise.loc[ids, "phi"].to_numpy(float)
    if not (np.isfinite(sd).all() and (sd > 0).all() and np.isfinite(phi).all()):
        raise L.EngineeringFailure("the pre-origin noise scale is not finite and positive")
    return Inputs(ids=ids, county=county, last_fit=last, sd=sd, phi=phi,
                  cells=wells.loc[ids, "cell"].to_numpy(int),
                  layers=wells.loc[ids, "layer"].to_numpy(int), fitmask=fit,
                  source={"e02_predictions_sha256": L.E02_PRED_SHA,
                          "e04_predictions_sha256": L.E04_PRED_SHA,
                          "used_from_predictions": "station ids and E04 prediction_origin",
                          "noise_ar1_sha256": L.NOISE_SHA, "obs_mask_sha256": L.MASK_SHA,
                          "wells_158_sha256": L.WELLS_SHA, "wells": len(ids),
                          "origins": {str(k): int(v) for k, v in
                                      pd.Series(org.astype(str)).value_counts().items()},
                          "counties": {str(k): int(v) for k, v in
                                       pd.Series(county).value_counts().items()}})


def fabricated_inputs(n_wells: int = 120, *, seed: int) -> Inputs:
    """FABRICATED inputs for smokes and tests (never real data)."""
    county = np.array(["Yunlin" if i % 2 == 0 else "Changhua" for i in range(n_wells)])
    ids = [f"{'09' if c == 'Yunlin' else '07'}{i:06d}" for i, c in enumerate(county)]
    last = np.full(n_wells, ST.AXIS.get_loc(pd.Timestamp("2026-09-01")))
    return Inputs(ids=ids, county=county, last_fit=last, sd=np.full(n_wells, 0.3),
                  phi=np.full(n_wells, 0.6), cells=np.arange(n_wells),
                  layers=np.zeros(n_wells, int), fitmask=np.ones((n_wells, O.T_FIT), bool),
                  source={"fabricated": True, "seed": seed})


# ----------------------------------------------------------------------------- worlds
T_AX = len(ST.AXIS)
_FE_IN_AXIS = slice(int(ST.AXIS.get_loc(L.MONTHS[0])), int(ST.AXIS.get_loc(L.MONTHS[-1])) + 1)


def layer_window(lay, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E04's layer re-run on one world's history -> (obs, STAT, CLIM), each (W, 11)."""
    f, info = ST.e04_frame(lay, hist)
    W = len(lay.ids)
    if info["n_kept"] != W or len(f) != W * (ST.WIN[1] - ST.WIN[0]):
        raise L.EngineeringFailure("E04's layer dropped a well in a synthetic world")
    if list(f["station_id"].to_numpy().reshape(W, -1)[:, 0]) != list(lay.ids):
        raise L.EngineeringFailure("e04_frame's row order differs from the layout")
    n = len(L.MONTHS)
    return tuple(f[c].to_numpy(float).reshape(W, -1)[:, -n:]
                 for c in ("observed_m", "predicted_m", "baseline_m"))


def variant(kind: str, W: int, cell: int, k: int, *, arm: str, seed: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """(wells kept, window mask) of a cell kind with the arm's own streams: full / grid /
    basin (all wells), mcar (25 % of window months missing at random; the wells left with
    fewer than 6 months are DROPPED, as the scorer's panel() drops them) or w100."""
    T = len(L.MONTHS)
    rows = np.arange(W)
    m = np.ones((W, T), bool)
    if kind == "mcar":
        m = rng_for(STREAMS[f"mask_{arm}"], cell, k, seed=seed).random((W, T)) >= MCAR
        rows = np.flatnonzero(m.sum(axis=1) >= L.MIN_MONTHS)
        m = m[rows]
    elif kind == "w100":
        rows = np.sort(rng_for(STREAMS[f"subset_{arm}"], cell, k, seed=seed).choice(
            W, N_SUB, replace=False))
        m = np.ones((len(rows), T), bool)
    return rows, m


def _p(est: dict, name: str = "P") -> dict:
    return {**est[name], "bad_share": est["bad_share"]}


@dataclass
class Pool:
    """The PHYS full-variant worlds' analogue contrasts (the only physics contrasts the design
    uses): dc = PHYS - CLIM, xp = PHYS - STAT, xc = CLIM - STAT, each (N, W, 11)."""
    dc: np.ndarray
    xp: np.ndarray
    xc: np.ndarray

    def pick(self, arm: str, cell: int, K: int, k: int) -> int:
        return (POOL_OFFSET[arm] + cell * K + k) % self.dc.shape[0]

    def partial_rms(self) -> np.ndarray:
        m = np.ones(self.xp.shape[1:], bool)
        return np.array([L.partial_rms(self.xp[j], self.xc[j], m)
                         for j in range(self.xp.shape[0])])


def phys_parts(inp: Inputs, h: np.ndarray, h_zr0: np.ndarray, ti: int, k: int, *,
               seed: int) -> dict:
    """FV v2 truth world (truth ti, world k) on the window: obs, STAT, CLIM, PHYS (W, 11).
    truth = h + FV v2-rule noise; E04's layer re-run on the truth; PHYS = h_ZR0 + E02's datum
    fitted on the world's observed 2012-22 months."""
    e = O.simulate_ar1(inp.sd, inp.phi, T_AX, rng_for(STREAMS["phys_noise"], ti, k,
                                                       seed=seed))
    truth = h[:, :T_AX] + e
    obs, stat, clim = layer_window(inp.layout(), truth)
    diff = (truth - h_zr0[:, :T_AX])[:, :O.T_FIT]
    anchor = np.where(inp.fitmask, diff, 0.0).sum(1) / inp.fitmask.sum(1)
    phys = h_zr0[:, _FE_IN_AXIS] + anchor[:, None]
    return {"obs": obs, "stat": stat, "clim": clim, "phys": phys}


def phys_world(inp: Inputs, h: np.ndarray, h_zr0: np.ndarray, ti: int, k: int, *,
               seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y, x_P, x_C, dC) of one FV v2 truth world (see :func:`phys_parts`)."""
    q = phys_parts(inp, h, h_zr0, ti, k, seed=seed)
    return (q["obs"] - q["stat"], q["phys"] - q["stat"], q["clim"] - q["stat"],
            q["phys"] - q["clim"])


def phys_records(inp: Inputs, h: np.ndarray, h_zr0: np.ndarray, ti: int, k: int, *,
                 seed: int, B: int = B_STUDY) -> tuple[dict, tuple]:
    """Every variant cell of one FV v2 world at lambda = 1, and its analogue contrasts."""
    y, xp, xc, dc = phys_world(inp, h, h_zr0, ti, k, seed=seed)
    out = {}
    for j, kind in enumerate(VARIANTS):
        cell = ti * len(VARIANTS) + j
        rows, m = variant(kind, len(inp.ids), cell, k, arm="phys", seed=seed)
        w = L.boot_weights(len(rows), B, rng_for(STREAMS["boot_phys"], cell, k, seed=seed))
        out[kind] = _p(L.estimate([xp[rows], xc[rows]], y[rows], m, weights=w, rng=None))
    return out, (dc, xp, xc)


def ar_record(inp: Inputs, c: dict, k: int, pool: Pool, *, seed: int, B: int = B_STUDY
              ) -> dict:
    """One AR world (lambda_P = 0) with a PHYS-world analogue dC."""
    lay = inp.layout(basin=c["kind"] == "basin")
    hist = S.world(lay, c["rho"], c["share"], STREAMS["ar"], c["cell"], k, seed=seed)
    obs, stat, clim = layer_window(lay, hist * inp.sd[:, None])
    dc = pool.dc[pool.pick("ar", c["cell"], K_AR, k)]
    rows, m = variant(c["kind"], len(inp.ids), c["cell"], k, arm="ar", seed=seed)
    xc = (clim - stat)[rows]
    xp = (dc + clim - stat)[rows]
    y = (obs - stat)[rows]
    w = L.boot_weights(len(rows), B, rng_for(STREAMS["boot_ar"], c["cell"], k, seed=seed))
    return {"ar": _p(L.estimate([xp, xc], y, m, weights=w, rng=None)),
            "one_ar": _p(L.estimate([xp], y, m, weights=w, rng=None))}


def pe_truth(inp: Inputs, c: dict, k: int, *, seed: int) -> np.ndarray:
    """(W, 190) persistence-error world: AR(1) at rho_hist (FV.stat.v2's generator, common
    share 0.15) up to each well's origin, then AR(1) at rho_win from the origin value."""
    lay = inp.layout()
    truth = S.world(lay, c["rho_hist"], PE_SHARE, STREAMS["pe"], c["cell"], k, seed=seed)
    truth = truth * inp.sd[:, None]
    r = rng_for(STREAMS["pe_window"], c["cell"], k, seed=seed)
    names = sorted(set(inp.county))
    zc = r.standard_normal((len(names), T_AX))[[names.index(x) for x in inp.county]]
    z = math.sqrt(1 - PE_SHARE) * r.standard_normal((len(inp.ids), T_AX)) + \
        math.sqrt(PE_SHARE) * zc
    s = math.sqrt(1.0 - c["rho_win"] ** 2)
    for t in range(int(inp.last_fit.min()) + 1, T_AX):
        post = t > inp.last_fit
        truth[post, t] = (c["rho_win"] * truth[post, t - 1]
                          + s * inp.sd[post] * z[post, t])
    return truth


def pe_shape(a0: np.ndarray, h: np.ndarray, r: float | None) -> np.ndarray:
    return np.zeros_like(h, float) if r is None else PE_AMP * a0[:, None] * r ** h


def pe_record(inp: Inputs, c: dict, k: int, pool: Pool, *, seed: int, B: int = B_STUDY
              ) -> dict:
    """One PE world under every origin-anomaly shape of dC (same history and draws)."""
    truth = pe_truth(inp, c, k, seed=seed)
    obs, stat, clim = layer_window(inp.layout(), truth)
    a0 = truth[np.arange(len(inp.ids)), inp.last_fit]
    h = (np.arange(_FE_IN_AXIS.start, _FE_IN_AXIS.stop)[None, :] - inp.last_fit[:, None])
    base = pool.dc[pool.pick("pe", c["cell"], K_PE, k)]
    xc, y = clim - stat, obs - stat
    m = np.ones_like(y, bool)
    w = L.boot_weights(len(inp.ids), B, rng_for(STREAMS["boot_pe"], c["cell"], k, seed=seed))
    out = {}
    for name, r in PE_SHAPES.items():
        xp = base + pe_shape(a0, h, r) + clim - stat
        out[name] = _p(L.estimate([xp, xc], y, m, weights=w, rng=None))
        out[f"one_{name}"] = _p(L.estimate([xp], y, m, weights=w, rng=None))
    return out


def sig_record(inp: Inputs, c: dict, k: int, pool: Pool, *, seed: int, B: int = B_STUDY
               ) -> dict:
    """Report-only: y = x_P,j + sd x AR world (window) on the analogue regressors of world j
    (true lambda_P 1, lambda_C 0)."""
    j = pool.pick("sig", c["cell"], K_SIG, k)
    a = S.world(inp.layout(), c["rho"], c["share"], STREAMS["sig"], c["cell"], k, seed=seed)
    y = pool.xp[j] + a[:, _FE_IN_AXIS] * inp.sd[:, None]
    m = np.ones_like(y, bool)
    w = L.boot_weights(len(inp.ids), B, rng_for(STREAMS["boot_sig"], c["cell"], k, seed=seed))
    est = L.estimate([pool.xp[j], pool.xc[j]], y, m, weights=w, rng=None)
    return {"sig": _p(est), "sig_c": est["C"]}


# ----------------------------------------------------------------------------- truth worlds
def fv2_batches_registered(repo: Path = REPO) -> dict:
    """{'<truth>_b<b>': {'path', 'npz_sha256'}} from FV v2's attempts.jsonl (last successful
    forward of each batch; metadata only, the sealed stdout is never opened)."""
    p = Path(repo) / L.FV2_ATTEMPTS
    if not p.is_file():
        return {}
    recs = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
    out = {}
    for t in TRUTHS:
        for b in range(V2.PLAN.n_batch):
            fw = [r for r in recs if r.get("event") == "forward" and r.get("truth") == t
                  and str(r.get("batch")) == str(b)]
            if fw and int(fw[-1].get("rc", 1)) == 0:
                out[f"{t}_b{b}"] = {"path": V2.batch_npz(t, b),
                                    "npz_sha256": fw[-1].get("npz_sha256")}
    return out


def real_truths(inp: Inputs, registered: dict, verified=None, heads=None):
    """(h_ZR0 (W, 190), generator of (truth index, world k, h (W, 190)), used) from the
    registered FV v2 batch npz: sha256-verified, scenario order and length checked. ``used``
    collects every npz and the ZR0 sha256 actually read (recorded in decidability.json).
    ``verified``/``heads`` replace the sha check / npz reader (tests only)."""
    verified = verified or _verified
    heads = heads or O.well_heads
    used = {"zr0": {"path": L.ZR0_NPZ, "sha256": L.ZR0_SHA}, "fv2_batches": {}}
    zmeta, zh = heads(verified(L.ZR0_NPZ, L.ZR0_SHA), inp.cells, inp.layers)
    if [str(x) for x in zmeta["scenario_names"]] != ["baseline"]:
        raise L.EngineeringFailure("fv_zr0.npz is not a baseline-only forward")
    h_zr0 = zh[0]

    def gen():
        for ti, t in enumerate(TRUTHS):
            for b in range(V2.PLAN.n_batch):
                rec = registered.get(f"{t}_b{b}")
                if rec is None:
                    raise L.EngineeringFailure(f"FV v2 batch {t}_b{b} is not registered in FE")
                meta, h = heads(verified(rec["path"], rec["npz_sha256"]), inp.cells,
                                inp.layers)
                if [str(x) for x in meta["scenario_names"]] != V2.PLAN.names(b):
                    raise L.EngineeringFailure(f"{rec['path']}: scenario order differs")
                if h.shape[-1] < T_AX:
                    raise L.EngineeringFailure(f"{rec['path']}: fewer than 190 months")
                used["fv2_batches"][f"{t}_b{b}"] = dict(rec)
                for j, k in enumerate(V2.PLAN.worlds(b)):
                    yield ti, k, h[1 + j]          # scenario 0 is the batch baseline
                del h
    return h_zr0, gen(), used


# ----------------------------------------------------------------------------- summaries
def _inf(r, shift: float = 0.0) -> bool:
    return r["lower"] is not None and r["lower"] - shift > 0


def _enc(r, shift: float = 0.0) -> bool:
    """The stat_encompasses reading (U_P < delta and not L_P > 0) on a shifted record."""
    return r["upper"] is not None and r["upper"] - shift < L.DELTA and not _inf(r, shift)


def _cover(r, v) -> bool:
    return r["lower"] is not None and r["lower"] <= v <= r["upper"]


def _rate(xs) -> dict:
    return O._rate([bool(x) for x in xs])


def _pool(per: list[dict], key: str) -> dict:
    ps = [x[key]["p"] for x in per]
    se = float(np.sqrt(sum(x[key]["mc_se"] ** 2 for x in per)) / len(per))
    return {"p": float(np.mean(ps)), "mc_se": se, "min_cell": float(min(ps)),
            "max_cell": float(max(ps))}


def summarise(ar: list[list[dict]], pe: list[list[dict]], phys: dict, sig: list[list[dict]],
              x_partial_rms: np.ndarray) -> dict:
    """Per-cell and pooled operating characteristics, G0 and the role if on time.
    ``ar``/``pe``/``sig``: per cell, per world records; ``phys``: {cell index: [records]};
    ``x_partial_rms``: the PHYS full-variant worlds' analogue partial contrast RMS."""
    per_ar = [{**c, "informative": _rate(_inf(w["ar"]) for w in ws),
               "encompasses": _rate(_enc(w["ar"]) for w in ws),
               "covers_0": _rate(_cover(w["ar"], 0.0) for w in ws),
               "upper_below_0": _rate(w["ar"]["upper"] is not None and w["ar"]["upper"] < 0
                                      for w in ws),
               "median_lambda_p": float(np.median([w["ar"]["lambda"] for w in ws])),
               "one_contrast_informative_report_only": _rate(_inf(w["one_ar"]) for w in ws)}
              for c, ws in zip(ar_cells(), ar, strict=True)]
    per_pe = [{**c, "shape": s, "informative": _rate(_inf(w[s]) for w in ws),
               "median_lambda_p": float(np.median([w[s]["lambda"] for w in ws])),
               "one_contrast_informative_report_only": _rate(_inf(w[f"one_{s}"]) for w in ws)}
              for c, ws in zip(pe_cells(), pe, strict=True) for s in PE_SHAPES]
    per_ph = []
    for c in phys_cells():
        ws = phys[c["cell"]]
        rec = {**c, "median_lambda_p": float(np.median([w["lambda"] for w in ws]))}
        for lam in LAMBDA_POINTS:
            s = 1.0 - lam
            rec[f"lambda_{lam}"] = {"informative": _rate(_inf(w, s) for w in ws),
                                    "encompasses": _rate(_enc(w, s) for w in ws)}
        per_ph.append(rec)
    per_sig = [{**c, "covers_1": _rate(_cover(w["sig"], 1.0) for w in ws),
                "lambda_c_covers_0": _rate(_cover(w["sig_c"], 0.0) for w in ws)}
               for c, ws in zip(sig_cells(), sig, strict=True)] if sig else []
    P = {"ar": {k: _pool(per_ar, k) for k in ("informative", "encompasses", "covers_0",
                                               "upper_below_0")},
         "pe": {"informative": _pool(per_pe, "informative")},
         "phys": {f"lambda_{lam}": {k: _pool([{k: x[f"lambda_{lam}"][k]} for x in per_ph], k)
                                    for k in ("informative", "encompasses")}
                  for lam in LAMBDA_POINTS},
         "one_contrast_report_only": {
             "ar": _pool(per_ar, "one_contrast_informative_report_only"),
             "pe": _pool(per_pe, "one_contrast_informative_report_only")}}
    if per_sig:
        P["sig_report_only"] = {k: _pool(per_sig, k) for k in ("covers_1", "lambda_c_covers_0")}
    med_truth = {t: float(np.median([w["lambda"] for c in phys_cells() if c["truth"] == t
                                     for w in phys[c["cell"]]])) for t in TRUTHS
                 if any(phys[c["cell"]] for c in phys_cells() if c["truth"] == t)}
    u = np.array([w["ar"]["upper"] for ws in ar for w in ws], float)
    bad = max([w["ar"]["bad_share"] for ws in ar for w in ws]
              + [w[s]["bad_share"] for ws in pe for w in ws for s in PE_SHAPES]
              + [w["bad_share"] for ws in phys.values() for w in ws])
    xr = float(np.median(x_partial_rms)) if len(x_partial_rms) else float("nan")
    p1 = P["phys"]["lambda_1.0"]
    g = {"a_false_informative": P["ar"]["informative"]["p"] <= G0["p_inf_ar_max"]
         and P["ar"]["informative"]["max_cell"] <= G0["p_inf_ar_cell_max"],
         "a2_persistence_error": P["pe"]["informative"]["p"] <= G0["p_inf_pe_max"]
         and P["pe"]["informative"]["max_cell"] <= G0["p_inf_pe_cell_max"],
         "b_power": p1["informative"]["p"] >= G0["p_inf_phys_min"]
         and p1["informative"]["min_cell"] >= G0["p_inf_phys_cell_min"],
         "c_false_encompass": p1["encompasses"]["p"] <= G0["p_enc_phys_max"]
         and p1["encompasses"]["max_cell"] <= G0["p_enc_phys_cell_max"],
         "d_null_encompass": P["ar"]["encompasses"]["p"] >= G0["p_enc_ar_min"],
         "e_contrast": bool(np.isfinite(xr) and xr >= G0["x_partial_rms_min_m"]),
         "f_replicates": bad <= G0["max_bad_replicates"]}
    return {"cells_ar": per_ar, "cells_pe": per_pe, "cells_phys": per_ph,
            "cells_sig_report_only": per_sig, "P": P,
            "median_lambda_p_by_truth_report_only": med_truth,
            "bootstrap_calibration": {
                "ar_covers_0": P["ar"]["covers_0"], "ar_lower_above_0": P["ar"]["informative"],
                "ar_upper_below_0": P["ar"]["upper_below_0"],
                "sig_covers_1": (P.get("sig_report_only") or {}).get("covers_1"),
                "nominal": "0.90 two-sided, 0.05 per side"},
            "q90_upper_ar_report_only": float(np.quantile(u, Q_REPORT)) if len(u) else None,
            "analogue_partial_contrast_rms_m": {
                "median": xr, "q10": float(np.quantile(x_partial_rms, 0.1))
                if len(x_partial_rms) else None, "worlds": int(len(x_partial_rms))},
            "max_bad_share": bad,
            "g0": {"conditions": g, "pass": bool(all(g.values())), "rule": G0_RULE,
                   "thresholds": G0},
            "role_if_on_time": "gating" if all(g.values()) else "report_only"}


def role_cap(created: str | None) -> tuple[str, str]:
    """The B2 hard rule, keyed to the registration time."""
    if not created:
        return "report_only", "no registration time"
    t = pd.Timestamp(created)
    t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
    if t > pd.Timestamp(L.GATING_ARM_BY):
        return "report_only", ("registered after 2026-11-25: the gating physics arm was not "
                               "merged in time (review B2 hard rule)")
    return "gating", "registered on or before 2026-11-25"


# ----------------------------------------------------------------------------- registration
def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    p = d / "INDEX.jsonl"
    if not p.is_file():
        return None
    rows = [r for r in prereg3.read_chained(p) if r["id"] == prereg_id]
    return rows[-1] if rows else None


def load_registered(prereg_id: str, seed: int, prereg_dir=None, rows_dir=None, *,
                    test_only: bool = False) -> tuple[dict, str]:
    """FE's registration intact, its frozen code byte-identical, its seed this seed.
    ``test_only`` admits a TEST.* id in a non-real registry (injected runs only)."""
    from harness import prereg3

    if prereg_id != L.ID and not (test_only and prereg_id.startswith(TEST_PREFIX)):
        raise L.Refused(f"this runner serves {L.ID} only, not {prereg_id!r}")
    if index_record(prereg_id, prereg_dir) is None:
        raise L.Refused(f"{prereg_id} is not in INDEX.jsonl: register it first")
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise L.Refused(f"{prereg_id} is not registered or not intact: {err}") from err
    fc = doc.get("frozen_code") or {}
    if set(fc) != set(FROZEN):
        raise L.EngineeringFailure("frozen_code does not list exactly FE's files")
    bad = [p for p, h in fc.items() if not O.rp(p).is_file() or O.sha256(O.rp(p)) != h]
    if bad:
        raise L.EngineeringFailure(f"code sha256 differs from the registration: {bad}")
    if (doc.get("design") or {}).get("seed") != seed:
        raise L.EngineeringFailure("the seed is not the registered one")
    if (doc.get("decidability") or {}).get("streams") != STREAMS:
        raise L.EngineeringFailure("the registered streams differ from this code's")
    return doc, sha


def attempts_path(out: str) -> Path:
    return O.rp(out).parent / L.ATTEMPTS_NAME


def log_attempt(path: Path, rec: dict) -> None:
    """Append one attempt record (never rewritten); no statistic is ever written here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": L.ID, "time": O._now(), **rec}, sort_keys=True,
                            ensure_ascii=False, default=O._jsonable) + "\n")


def attempt_no(path: Path) -> int:
    if not path.is_file():
        return 1
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return 1 + sum(r.get("event") == "start" for r in rows)


def _testing() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _real_registry(prereg_dir) -> bool:
    from harness import prereg3

    if prereg_dir is None:
        return True
    try:
        return Path(prereg_dir).resolve() == Path(prereg3.PREREG_DIR).resolve()
    except OSError:
        return True


def injection_refusal(prereg_id: str, prereg_dir, test_only: bool) -> str | None:
    """Why injected inputs / truths are refused here (None = allowed)."""
    if not test_only:
        return "injected inputs or truths need test_only=True"
    if not _testing():
        return "injected inputs or truths are for tests only"
    if prereg_id == L.ID or not str(prereg_id).startswith(TEST_PREFIX):
        return f"injected runs refuse the real registration id (use a {TEST_PREFIX}* id)"
    if _real_registry(prereg_dir):
        return "injected runs refuse the real registry directory"
    return None


def study(inp: Inputs, h_zr0: np.ndarray, truths, *, seed: int, k_ar: int = K_AR,
          k_pe: int = K_PE, k_sig: int = K_SIG, B: int = B_STUDY) -> dict:
    """PHYS first (it builds the analogue pool), then AR, PE and SIGMA; then summarise."""
    phys = {c["cell"]: [] for c in phys_cells()}
    dcs, xps, xcs = [], [], []
    for ti, k, h in truths:
        recs, (dc, xp, xc) = phys_records(inp, h, h_zr0, ti, k, seed=seed, B=B)
        for j, kind in enumerate(VARIANTS):
            phys[ti * len(VARIANTS) + j].append(recs[kind])
        dcs.append(dc)
        xps.append(xp)
        xcs.append(xc)
    if any(not v for v in phys.values()):
        raise L.EngineeringFailure("a physics-arm cell has no world")
    pool = Pool(dc=np.stack(dcs), xp=np.stack(xps), xc=np.stack(xcs))
    ar = [[ar_record(inp, c, k, pool, seed=seed, B=B) for k in range(k_ar)]
          for c in ar_cells()]
    pe = [[pe_record(inp, c, k, pool, seed=seed, B=B) for k in range(k_pe)]
          for c in pe_cells()]
    sig = [[sig_record(inp, c, k, pool, seed=seed, B=B) for k in range(k_sig)]
           for c in sig_cells()]
    return summarise(ar, pe, phys, sig, pool.partial_rms())


def run(seed, prereg_id: str, *, out: str = L.DECIDABILITY, prereg_dir=None, rows_dir=None,
        inputs: Inputs | None = None, truths=None, test_only: bool = False, now=None,
        log=print) -> dict:
    """The study, once, under FE's registration (mode 'x'); every attempt is logged.
    ``inputs`` / ``truths`` are accepted only with test_only=True inside pytest, under a
    TEST.* id and a registry directory that is not the real one."""
    outp = O.rp(out)
    ap = attempts_path(out)
    n = attempt_no(ap)
    log_attempt(ap, {"event": "start", "stage": "decidability", "attempt": n, "seed": seed,
                     "prereg_id": prereg_id, "code_sha256": O.sha256(Path(__file__))})
    try:
        injected = inputs is not None or truths is not None
        if injected:
            why = injection_refusal(prereg_id, prereg_dir, test_only)
            if why:
                raise L.Refused(why)
        seed = check_seed(seed)
        if outp.exists():
            raise L.Refused(f"{outp} exists: the decidability study is run once")
        doc, dsha = load_registered(prereg_id, seed, prereg_dir, rows_dir,
                                    test_only=injected)
        created = (index_record(prereg_id, prereg_dir) or {}).get("created")
        inp = inputs or load_inputs()
        if truths is None:
            h_zr0, gen, used = real_truths(inp, (doc.get("decidability") or {})
                                           .get("fv2_batches") or {})
        else:
            h_zr0, gen = truths
            used = {"fabricated": True}
        rep = {"id": L.ID, "stage": "decidability", "version": VERSION,
               "created": O._now(), "declaration_sha256": dsha, "registration_created": created,
               "seed": seed, "streams": STREAMS, "k_ar": K_AR, "k_pe": K_PE, "k_phys": K_PHYS,
               "k_sig": K_SIG, "B": B_STUDY, "inputs": inp.source,
               **study(inp, h_zr0, gen, seed=seed), "frozen_code": doc["frozen_code"]}
        rep["truth_files_used"] = used
        t = now or dt.datetime.now(dt.timezone.utc)
        rep["on_time"] = bool(t < dt.datetime.fromisoformat(L.REGISTER_BY))
        cap, why = role_cap(created)
        rep["role_cap"] = {"role": cap, "why": why}
        rep["role"] = ("gating" if rep["on_time"] and cap == "gating"
                       and rep["role_if_on_time"] == "gating" else "report_only")
        O._write_json_once(outp, rep)
    except (L.Refused, L.EngineeringFailure) as err:
        log_attempt(ap, {"event": type(err).__name__, "stage": "decidability", "attempt": n,
                         "message": str(err)})
        raise
    except Exception as err:
        log_attempt(ap, {"event": "error", "stage": "decidability", "attempt": n,
                         "message": f"{type(err).__name__}: {err}"})
        raise
    log_attempt(ap, {"event": "success", "stage": "decidability", "attempt": n,
                     "report_sha256": O.sha256(outp), "declaration_sha256": dsha})
    log(f"wrote {out}")
    return rep


def verify_decidability(dec: dict, dec_path: Path, doc: dict, decl_sha: str) -> None:
    """Traceability (scoring time): the study used exactly the registered FV v2 batch npz and
    ZR0, and this decidability.json is the one its attempts.jsonl success row hashed under
    this registration. Raises EngineeringFailure otherwise."""
    used = dec.get("truth_files_used") or {}
    reg = (doc.get("decidability") or {}).get("fv2_batches") or {}
    got = {k: (v or {}).get("npz_sha256") for k, v in (used.get("fv2_batches") or {}).items()}
    want = {k: (v or {}).get("npz_sha256") for k, v in reg.items()}
    if not want or got != want:
        raise L.EngineeringFailure("decidability.json did not use exactly the registered FV v2 "
                                   "batch npz")
    if (used.get("zr0") or {}).get("sha256") != L.ZR0_SHA:
        raise L.EngineeringFailure("decidability.json did not use the registered fv_zr0.npz")
    ap = Path(dec_path).parent / L.ATTEMPTS_NAME
    rows = ([json.loads(x) for x in ap.read_text(encoding="utf-8").splitlines() if x.strip()]
            if ap.is_file() else [])
    ok = [r for r in rows if r.get("event") == "success" and r.get("stage") == "decidability"]
    if not ok or ok[-1].get("report_sha256") != O.sha256(dec_path) \
            or ok[-1].get("declaration_sha256") != decl_sha:
        raise L.EngineeringFailure("decidability.json does not match the success row of its "
                                   "attempts.jsonl")


# ----------------------------------------------------------------------------- smoke
def _under_results(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to((REPO / "results").resolve())
        return True
    except ValueError:
        return False


def fabricated_truths(inp: Inputs, k: int, *, seed: int):
    """FABRICATED truth forwards (smoke/tests): seasonal series per well with a per-world
    projection anomaly after 2022-12."""
    r = np.random.default_rng(np.random.SeedSequence([seed, 998]))
    season = np.sin(2 * np.pi * np.arange(T_AX) / 12)[None, :]
    h_zr0 = r.standard_normal((len(inp.ids), 1)) * 3 + 0.5 * season
    post = (np.arange(T_AX) >= O.T_FIT)[None, :]

    def gen():
        for ti in range(len(TRUTHS)):
            for kk in range(k):
                bump = 0.2 * r.standard_normal((len(inp.ids), 1)) * post
                yield ti, kk, h_zr0 + 0.1 * ti + bump + 0.05 * r.standard_normal(
                    (len(inp.ids), T_AX))
    return h_zr0, gen()


def run_smoke(out: Path, k: int = 2, B: int = 50) -> dict:
    """Engineering only, on SMOKE_SEED and FABRICATED inputs and truths: every arm, the layer
    re-run and the bootstrap at a tiny K; summary plumbing. Writes no rate, P or quantile."""
    out = Path(out)
    if _under_results(out):
        raise L.EngineeringFailure("the smoke never writes under results/")
    out.mkdir(parents=True, exist_ok=False)
    inp = fabricated_inputs(seed=SMOKE_SEED)
    h_zr0, gen = fabricated_truths(inp, k, seed=SMOKE_SEED)
    s = study(inp, h_zr0, gen, seed=SMOKE_SEED, k_ar=k, k_pe=k, k_sig=1, B=B)
    ok = s["role_if_on_time"] in ("gating", "report_only") and len(s["cells_phys"]) == 12 \
        and len(s["cells_pe"]) == 6
    rep = {"id": L.ID, "smoke": True, "status": "ran ok" if ok else "plumbing failed",
           "seed": SMOKE_SEED, "k": k, "B": B, "inputs": "FABRICATED",
           "note": "SMOKE: engineering only; no rate or quantile is written"}
    O._write_json_once(out / "smoke.json", rep)
    return rep


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("--out", required=True, help="a NEW directory outside results/")
    rn = sub.add_parser("run")
    rn.add_argument("--seed", type=int, required=True, help="FE's registered seed")
    rn.add_argument("--prereg-id", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            print(rep["status"])
            return 0 if rep["status"] == "ran ok" else 1
        rep = run(a.seed, a.prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {L.DECIDABILITY} (role {rep['role']})")
        return 0
    except (L.Refused, L.EngineeringFailure) as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
