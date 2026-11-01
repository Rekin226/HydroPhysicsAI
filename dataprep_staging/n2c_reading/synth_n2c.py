"""X5.n2c_reading.v1: synthetic operating characteristics of N2C's decision rule, run with the
hashed shared N2/N2C scorer (dataprep_staging/n2_policy/bootstrap_scorer.py) exactly as it
will run in the 2028 session.

Synthetic worlds over N2C's 74 Changhua wells and window (2026-11..2027-10):

    obs_syn - ZR = e = k * c_j + n + kappa * g_j

- c_j: R1c's contrast at each well's cell and layer (RPj - baseline of ONE CPU forward of
  R1c's main set, N2C's registered command with the PR arm replaced by seven recorded arms
  RP2012..RP2018 replaying contiguous pre-origin census anomaly blocks; replay.py). ZR cancels
  from e, so the synthetic panel is scored with ZR = 0 and obs = e; RMSE(ZR) = rms(e) and
  RMSE(PR) = rms(e - c) as in the real scoring.
- n: residual noise, per-well AR(1) plus a 5 km block AR(1), stationary from the first month,
  total sd SIGMA_TOT (I1.R1c Changhua anomaly RMSE), block share BLOCK_SHARE (R1c residual
  pair correlation at 0-5 km), lag-1 PHI (per-well lag-1 anomaly autocorrelation, median).
- g_j: an omitted canal-like term common to the 74 wells: unit sd over the window,
  correlation RHO with replay j's Changhua pumping anomaly a_CH,j(t) (exactly, by
  orthogonalisation); RHO = -0.83 is the documented drought-year co-movement (low canal
  water with high pumping), which biases N2C's beta UPWARD (c ~ -a). KAPPA scales it.

Every parameter cites a published report field (``PARAMS``); ``prereg_n2c_reading.py``
checks the cited values against the files before registration.

Order of the registered run (``run``): (1) exact plumbing: with no noise and kappa 0 the
scorer returns beta == k for every replay; a failure is a code defect (ENGINEERING FAILURE, no
reading; only a code fix may follow). (2) the plumbing cell: k = 1, rho = 0, kappa 0, the
declared noise scaled by PLUMBING_NOISE_SCALE, PLUMBING_REPLICATES replicates on its own
generator; its median beta must lie in [0.9, 1.1]. At that noise OLS cannot miss the band by
Monte Carlo chance unless the contrast is degenerate, so a failure with (1) passing is a
declared TERMINAL reading, 'not_estimable' (fixed display text, result.json written, no grid
cell computed, no v2 may change any design parameter). (3) the grid at full noise (the
full-noise (k = 1, rho = 0) cell is an ordinary grid cell) and the report-only cells; (4) the
declared reading. result.json and result.md are written once (mode 'x').

Usage (repository root; the runner calls it):
    python dataprep_staging/n2c_reading/synth_n2c.py extract --forward <npz> --out <panel.npz>
    python dataprep_staging/n2c_reading/synth_n2c.py run
    python dataprep_staging/n2c_reading/synth_n2c.py smoke --out <dir>   (synthetic panel)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "dataprep_staging" / "n2_policy"), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bootstrap_scorer as BS  # noqa: E402

DECL_ID = "X5.n2c_reading.v1"
X = "dataprep_staging/n2c_reading"
LOG = "results/twin_runs/n2c_reading"
REPLAYS = f"{LOG}/replays"
FWD_STEM = "results/twin_forward/n2c_reading_r1c"
PANEL = f"{LOG}/panel.npz"
OUT_DIR = f"results/dataprep/reports/{DECL_ID}"
WELLS = "dataprep_staging/x5_class_share/wells_frozen.json"
WELLS_SHA = "fc2d8257dbe729238a96faf96f2e0b8e3ff82facc7faddb03794b453c2fffcd6"
WELLS_XY = "results/twin_runs/stage3_datum_gate_ccz/stage3_wells.csv"
CODE_FILES = (f"{X}/synth_n2c.py", f"{X}/run_n2c_reading.sh",
              "dataprep_staging/n2_policy/bootstrap_scorer.py",
              "dataprep_staging/x5_changhua_lever/replay.py", WELLS)
REPLAY_YEARS = (2012, 2013, 2014, 2015, 2016, 2017, 2018)
N2C_WINDOW = (178, 190)                        # forward months 2026-11 .. 2027-10
N_T = 252

_R2_R0_CH = 0.14                               # I1_changhua_drivers/report.md table
_RMSE_R0_CH = 1.0735429484231622
_R2_CANAL = 0.06580829097530272
PARAMS = {
    "sigma_tot_m": {
        "value": 1.3177827290567443,
        "source": "results/dataprep/reports/I1_changhua_local/report.json",
        "field": "part_a_local.changhua.rmse_r1c_mean",
        "note": "R1c's out-of-fold Changhua own-mean head-anomaly RMSE (the I1.R1c gate value "
                "1.318 m); design-time, spent heads"},
    "block_share": {
        "value": 0.43923302041464884,
        "source": "results/dataprep/reports/I1_changhua_interannual/report.json",
        "field": "counties.changhua.residual.R1c.by_distance.0-5 km.median_r",
        "note": "median correlation of R1c's Changhua residual series for well pairs within "
                "5 km = the 5 km block component's variance share; pairs beyond 20 km "
                "correlate -0.08 (same report), so no county-wide component is added"},
    "phi": {
        "value": 0.8592242401907416,
        "source": "results/twin/hybrid_datum_gate_ccz/report.json",
        "field": "rho.median_own",
        "note": "median per-well lag-1 autocorrelation of the OBSERVED monthly head anomaly "
                "on the fitted months over 158 wells in both counties (R1c's hybrid report), "
                "not of R1c's Changhua residuals (no such field is published); used for both "
                "AR(1) parts"},
    "kappa_m": {
        "value": math.sqrt(_R2_CANAL) * _RMSE_R0_CH / math.sqrt(1.0 - _R2_R0_CH),
        "source": "results/dataprep/reports/I1_changhua_drivers/report.json; "
                  "results/dataprep/reports/I1_changhua_local/report.json; "
                  "results/dataprep/reports/I1_changhua_drivers/report.md",
        "field": "sqrt(summary.changhua.median_cv_r2.raw:canal_w1_cell) x "
                 "part_a_local.changhua.rmse_r0_mean / sqrt(1 - R0 Changhua median-well "
                 "out-of-fold R2 0.14 (report.md table))",
        "inputs": {"canal_r2": _R2_CANAL, "rmse_r0_m": _RMSE_R0_CH, "r2_r0": _R2_R0_CH},
        "note": "no per-well ridge coefficient is published (report.md and report.json carry "
                "R2 only); kappa is the head-anomaly sd that the canal-water driver alone "
                "explains out of sample (median Changhua well): sqrt(R2_canal) x the anomaly "
                "sd implied by R0's RMSE and R2. Design-time on spent heads"},
    "rho_abs": {
        "value": 0.83,
        "source": "results/dataprep/reports/I1_changhua_interannual/report.json",
        "field": "counties.changhua.observed.regional_vs_drivers_r (census_energy "
                 "-0.8347, canal_deliveries +0.8291)",
        "note": "the judge's value. The cited field is the correlation of the observed "
                "regional head with each driver (energy -0.83, canal deliveries +0.83), used "
                "as a proxy: no canal-vs-energy (canal-vs-pumping) correlation is published. "
                "Sign convention: rho = corr(g, a_CH) (canal-like term vs the Changhua "
                "pumping anomaly), the drought-year direction that biases beta upward"},
}
GRID = {"k": (0.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0), "rho": (0.0, -0.83)}
REPORT_ONLY_CELLS = ({"k": 0.0, "rho": -0.83, "kappa_mult": 2.0},
                     {"k": 1.0, "rho": -0.83, "kappa_mult": 2.0})
REPLICATES = 500
SEED = 20261102
PLUMBING_BAND = (0.9, 1.1)
PLUMBING_NOISE_SCALE = 0.02
PLUMBING_REPLICATES = 200
READING = {"false_pass_max": 0.25, "wrong_size_max": 0.25, "coverage_min": 0.80}
DISPLAY = {
    "reported_not_shown": "an N2C 'response supported' is reported, not shown: C13a stays W "
                          "(the confounded false-pass rate P(supported | k = 0, rho = -0.83) "
                          "= {fp:.2f} >= 0.25; the stricter of the unconditional rate and the "
                          "rate among replicates that pass N2C's informative gate)",
    "rates_printed": "a shown Changhua head lever is printed with the measured operating "
                     "characteristics (false pass {fp:.2f}; wrong-size pass at k = 0.5 / 2: "
                     "{ws05:.2f} / {ws2:.2f}; 90 % interval coverage at k = 1: {cov:.2f}; "
                     "pass rates are the stricter of unconditional and given informative)",
    "power": "beside a non-pass N2C verdict: 'power {pw:.2f} (P(response supported) if the "
             "model's response were right; {pwc:.2f} with the canal confounder)'",
    "verdict_word": "the N2C verdict word never changes",
    "not_estimable": "X5.n2c_reading.v1: N2C's operating characteristics were not estimable "
                     "(the low-noise plumbing cell's median beta {mb} fell outside [0.9, 1.1] "
                     "with exact plumbing passing); no rate is printed; this reading neither "
                     "narrows nor widens N2C's display, which follows N2C's page_mapping and "
                     "the other declared restrictions",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(s: str | Path) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def code_sha256() -> dict:
    return {p: sha256(REPO / p) if (REPO / p).exists() else "missing" for p in CODE_FILES}


def design_spec() -> dict:
    """Everything the registered run depends on (frozen into the declaration)."""
    return {
        "world": "e = k * c_j + n + kappa * g_j over the 74 Changhua wells x 12 window months; "
                 "ZR = 0, obs = e; scored by bootstrap_scorer.score (flags 0, no CL, "
                 "detrended secondary off) with n_recorded_months 12",
        "params": PARAMS, "grid": {k: list(v) for k, v in GRID.items()},
        "report_only_cells": list(REPORT_ONLY_CELLS), "replicates_per_cell": REPLICATES,
        "seed": SEED, "seeding": "numpy SeedSequence(seed).spawn(n_cells + 1): children "
                                 "0..n_cells-1 in cell order (k outer, rho inner, then the "
                                 "report-only cells), child n_cells for the plumbing cell; "
                                 "replicate r of a cell uses replay r mod 7 and that cell's "
                                 "generator in order: block noise (n_blocks x 12), well noise "
                                 "(74 x 12), xi (12)",
        "replays": [f"RP{y}" for y in REPLAY_YEARS],
        "replay_source": "dataprep_staging/x5_changhua_lever/replay.py: model_ready/"
                         "release_20261012b2/preorigin/electricity_census/v1 county_class_"
                         "monthly (use design_time_evidence; in_grid, all classes; months <= "
                         "2022-12; ratio to the 2012-2022 month-of-year mean); RPY maps "
                         "2023-01+j onto Y-01+j, j = 0..57, Y = 2012..2018; never X1",
        "noise": "x(t) = phi x(t-1) + sqrt(1 - phi^2) s eps, x(0) ~ N(0, s^2); well part s^2 = "
                 "(1 - block_share) sigma_tot^2, block part s^2 = block_share sigma_tot^2 "
                 "(5 km TWD97 blocks, bootstrap_scorer.block_ids of the wells' x, y)",
        "confounder": "g = rho a~ + sqrt(1 - rho^2) xi~, a~ = the window's Changhua anomaly "
                      "demeaned and scaled to unit sd, xi~ = an N(0, 1) draw demeaned, "
                      "orthogonalised against a~ and scaled to unit sd; common to all wells",
        "scorer": {"path": "dataprep_staging/n2_policy/bootstrap_scorer.py",
                   "version": BS.VERSION, "B": BS.B_BOOT, "seed": BS.SEED,
                   "level": BS.LEVEL},
        "plumbing": {"exact": "no noise, kappa 0: beta == k (|diff| <= 1e-9) for every "
                              "replay with an informative contrast; failure = engineering "
                              "(code defect), no reading",
                     "cell": {"k": 1.0, "rho": 0.0, "kappa_mult": 0.0,
                              "noise_scale": PLUMBING_NOISE_SCALE,
                              "replicates": PLUMBING_REPLICATES, "band": list(PLUMBING_BAND),
                              "outside_band": "terminal reading 'not_estimable' (display "
                                              "text DISPLAY.not_estimable); result.json is "
                                              "written; no grid cell is computed; final"},
                     "full_noise_k1_rho0": "an ordinary grid cell; its median beta is reported "
                                           "and gates nothing"},
        "reading": {**READING, "display": DISPLAY,
                    "rules": [
                        "every pass rate P below is max(P(response_supported), "
                        "P(response_supported | the replicate passes N2C's informative "
                        "gate)) (the stricter; the conditional rate is undefined, and the "
                        "unconditional one used, when no replicate is informative); both are "
                        "reported",
                        "R1: P(k = 0, rho = -0.83) >= 0.25 -> reported_not_shown",
                        "R2: max over rho of P(k = 0.5) or of P(k = 2) >= 0.25, or the 90 % "
                        "interval coverage of k = 1 at rho = 0 < 0.80 -> rates_printed",
                        "R3: power = P(response_supported | k = 1, rho = 0) (unconditional; "
                        "the conditional one reported beside it), printed with rho = -0.83 "
                        "beside it, beside any non-pass verdict",
                        "plumbing cell outside its band (exact plumbing passing) -> the "
                        "terminal reading not_estimable instead of R1-R3",
                        "the verdict word never changes; the report-only kappa x2 cells "
                        "never change a reading"]},
        "window": ["2026-11", "2027-10"], "n2c_window_forward_months": list(N2C_WINDOW),
    }


# ----------------------------------------------------------------------------- panel
def changhua_screens() -> dict:
    p = REPO / WELLS
    if sha256(p) != WELLS_SHA:
        raise SystemExit(f"{WELLS} differs from {WELLS_SHA[:8]}")
    rows = [r for r in json.loads(p.read_text())["rows"] if r["county"] == "changhua"]
    if len(rows) != 74:
        raise SystemExit("expected 74 Changhua screens")
    return {"sid": np.array([r["sid"] for r in rows]),
            "cell": np.array([r["cell"] for r in rows], int),
            "layer": np.array([r["layer"] for r in rows], int)}


def well_blocks(sids, xy_csv: Path) -> np.ndarray:
    import pandas as pd

    t = pd.read_csv(xy_csv, usecols=["sid", "x", "y"], dtype={"sid": str})
    t = t.drop_duplicates("sid").set_index("sid")
    miss = [s for s in sids if s not in t.index]
    if miss:
        raise SystemExit(f"{xy_csv}: no coordinates for {miss[:5]}")
    return BS.block_ids(t.loc[list(sids), "x"].to_numpy(float),
                        t.loc[list(sids), "y"].to_numpy(float))


ARCHIVE_MEMBERS = "results/twin_forward/datum_gate_ccz.members.npz"
ATTEMPTS = f"{LOG}/attempts.jsonl"


def main_member_l2(path: Path, cells: np.ndarray) -> tuple[np.ndarray, float]:
    """R1c's main set, unperturbed (ic 0), baseline December layer-2 heads at ``cells``
    (int16 quanta) from a ``--save-members yearly`` sidecar."""
    z = np.load(path, allow_pickle=False)
    names = [str(x) for x in z["scenario_names"]]
    k = [i for i, (m, j) in enumerate(zip([str(x) for x in z["head_member"]], z["head_ic"],
                                          strict=True))
         if m == "stage3_datum_gate_ccz" and int(j) == 0]
    if len(k) != 1:
        raise SystemExit(f"{path}: {len(k)} unperturbed main members")
    q = np.asarray(z["headL2_members_yr"][names.index("baseline"), k[0]], np.int64)
    return q[cells, :], float(z["head_scale_m"])


def reproduction(members: Path, archive: Path, cells: np.ndarray) -> dict:
    """Engineering: the run is R1c's main set as archived (within one 1 cm quantum)."""
    (a, qa), (b, qb) = main_member_l2(members, cells), main_member_l2(archive, cells)
    if qa != qb:
        raise SystemExit("member quanta differ")
    dq = int(np.max(np.abs(a - b)))
    return {"max_abs_diff_quanta": dq, "quantum_m": qa, "ok": dq <= 1,
            "archive": ARCHIVE_MEMBERS, "archive_sha256": sha256(archive)}


def last_forward_sha(attempts: Path) -> str | None:
    if not attempts.exists():
        return None
    recs = [json.loads(x) for x in attempts.read_text().splitlines() if x.strip()]
    fw = [r for r in recs if r.get("event") == "forward"]
    return fw[-1].get("npz_sha256") if fw and int(fw[-1].get("rc", -1)) == 0 else None


FWD_MEMBER = "stage3_datum_gate_ccz"
FWD_OPTIONS = {"column_heads": "free", "update_method": "innovation", "apex_hold": "calibrated",
               "restart_taper_km": 5.0, "column_hpc0_fast_days": 365.0,
               "save_members": "yearly"}
ORIGIN, HORIZON = 131, 120                     # 2022-12; 120 months


def forward_meta_problems(f) -> list[str]:
    """N2C's command as run: member label, gains, origin, horizon, forward options
    (score_chlever.load_npz's checks)."""
    e = []
    dates = [str(s)[:7] for s in f["dates"]]
    if not dates or dates[0] != "2012-01" or dates[-1] != "2032-12":
        e.append(f"dates {dates[:1]}..{dates[-1:]}")
    if int(f["origin"]) != ORIGIN or int(f["horizon"]) != HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    labels = [str(s) for s in f["member_labels"]]
    if labels != [FWD_MEMBER]:
        e.append(f"member_labels {labels}")
    if float(f["gain"]) != 1.0 or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    fo = json.loads(str(f["forward_options"]))
    bad = {k: fo.get(k) for k, v in FWD_OPTIONS.items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    return e


def extract(fwd: Path, replays_json: Path, out: Path) -> dict:
    """The panel: c_j (7, 74, 12) at the Changhua screens, a_CH,j (7, 12), blocks (74,)."""
    sc = changhua_screens()
    f = np.load(fwd, allow_pickle=False)
    names = [str(s) for s in f["scenario_names"]]
    want = ["baseline"] + [f"RP{y}" for y in REPLAY_YEARS]
    if names != want:
        raise SystemExit(f"{fwd}: scenarios {names} != {want}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != N_T or dates[N2C_WINDOW[0]] != "2026-11" or \
            dates[N2C_WINDOW[1] - 1] != "2027-10":
        raise SystemExit(f"{fwd}: window months misplaced")
    meta_problems = forward_meta_problems(f)
    if meta_problems:
        raise SystemExit(f"{fwd}: " + "; ".join(meta_problems))
    rep = json.loads(replays_json.read_text())
    rec = (json.loads(str(f["forward_options"]))).get("recorded_pumping") or {}
    hm = f["heads_mean"]
    t0, t1 = N2C_WINDOW
    base = np.asarray(hm[0][sc["layer"], sc["cell"], t0:t1], float)
    c, a = [], []
    for y in REPLAY_YEARS:
        r = rep["replays"][f"rp{y}"]
        if rec.get(f"RP{y}", {}).get("sha256") != r["sha256"]:
            raise SystemExit(f"{fwd}: RP{y} was not run on {r['file']} as built")
        c.append(np.asarray(hm[names.index(f"RP{y}")][sc["layer"], sc["cell"], t0:t1],
                            float) - base)
        a.append(np.asarray(r["window"]["a_changhua"], float))
    repro = reproduction(Path(str(fwd)[:-4] + ".members.npz"), rp(ARCHIVE_MEMBERS),
                         sc["cell"])
    if not repro["ok"]:
        raise SystemExit(f"ENGINEERING: the run does not reproduce R1c's archived main member "
                         f"({repro['max_abs_diff_quanta']} quanta)")
    blocks = well_blocks(sc["sid"], rp(WELLS_XY))
    panel = {"c": np.stack(c), "a_ch": np.stack(a), "sid": sc["sid"], "block": blocks,
             "replays": np.array([f"RP{y}" for y in REPLAY_YEARS]),
             "forward_sha256": np.str_(sha256(fwd)),
             "replays_json_sha256": np.str_(sha256(replays_json))}
    if not np.isfinite(panel["c"]).all():
        raise SystemExit("non-finite contrast")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        np.savez(fh, **panel)
    return {"panel": str(out), "panel_sha256": sha256(out), "shape_c": list(panel["c"].shape),
            "n_blocks": int(len(set(blocks))), "reproduction": repro}


def load_panel(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def synthetic_panel(seed: int = 0, n_blocks: int = 24, scale: float = 0.3) -> dict:
    """A stand-in panel for smoke runs and tests (no model output)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, (7, 12))
    lam = rng.uniform(0.5, 1.5, 74) * scale / 0.1
    lag = np.array([[0.3 * a[j, max(t - 1, 0)] + 0.7 * a[j, t] for t in range(12)]
                    for j in range(7)])
    c = -lam[None, :, None] * lag[:, None, :] + rng.normal(0, 0.2, (7, 74, 1))
    blocks = np.array([f"{b}_0" for b in rng.integers(0, n_blocks, 74)])
    return {"c": c, "a_ch": a, "sid": np.array([f"07{i:06d}" for i in range(74)]),
            "block": blocks, "replays": np.array([f"RP{y}" for y in REPLAY_YEARS])}


# ----------------------------------------------------------------------------- worlds
def ar1(rng, n: int, T: int, s: float, phi: float) -> np.ndarray:
    eps = rng.standard_normal((n, T))
    x = np.empty((n, T))
    x[:, 0] = s * eps[:, 0]
    q = math.sqrt(1.0 - phi * phi) * s
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + q * eps[:, t]
    return x


def confounder(rng, a_ch: np.ndarray, rho: float) -> np.ndarray:
    a = a_ch - a_ch.mean()
    sd = a.std()
    a = a / sd if sd > 0 else a
    xi = rng.standard_normal(a.size)
    xi -= xi.mean()
    if sd > 0:
        xi -= (xi @ a) / (a @ a) * a
    xi /= xi.std()
    return rho * a + math.sqrt(1.0 - rho * rho) * xi


def world(panel: dict, j: int, k: float, rho: float, rng, kappa_mult: float = 1.0,
          noise_scale: float = 1.0) -> np.ndarray:
    c = panel["c"][j]
    ub, binv = np.unique(panel["block"], return_inverse=True)
    sig, share, phi = (PARAMS["sigma_tot_m"]["value"], PARAMS["block_share"]["value"],
                       PARAMS["phi"]["value"])
    b = ar1(rng, len(ub), c.shape[1], noise_scale * sig * math.sqrt(share), phi)[binv]
    w = ar1(rng, c.shape[0], c.shape[1], noise_scale * sig * math.sqrt(1.0 - share), phi)
    g = confounder(rng, panel["a_ch"][j], rho)
    return k * c + b + w + kappa_mult * PARAMS["kappa_m"]["value"] * g[None, :]


def score_world(panel: dict, j: int, e: np.ndarray) -> dict:
    c = panel["c"][j]
    nw, T = c.shape
    wells = np.repeat(panel["sid"], T)
    months = np.tile([f"{2026 + (10 + t) // 12}-{(10 + t) % 12 + 1:02d}" for t in range(T)],
                     nw)
    blocks = dict(zip(panel["sid"], panel["block"], strict=True))
    inf = BS.informative_gate(c.ravel(), wells, blocks, T)
    return BS.score(wells, months, e.ravel(), np.zeros(e.size), c.ravel(),
                    np.zeros(e.size, int), blocks, inf, detrended_secondary=False)


def cells() -> list[dict]:
    out = [{"k": k, "rho": r, "kappa_mult": 1.0, "report_only": False}
           for k in GRID["k"] for r in GRID["rho"]]
    return out + [{**c, "report_only": True} for c in REPORT_ONLY_CELLS]


def run_cell(panel: dict, cell: dict, rng, replicates: int) -> dict:
    n_rep = panel["c"].shape[0]
    v = {x: 0 for x in BS.VERDICTS}
    betas, cover, ratio, n_inf = [], 0, [], 0
    for r in range(replicates):
        j = r % n_rep
        e = world(panel, j, cell["k"], cell["rho"], rng, cell["kappa_mult"],
                  cell.get("noise_scale", 1.0))
        s = score_world(panel, j, e)
        v[s["verdict"]] += 1
        n_inf += int(bool(s["informative_gate"]["passes"]))
        b = s["beta_response"]
        betas.append(b["beta"])
        lo, hi = b["ci90"]
        cover += int(np.isfinite(lo) and lo <= cell["k"] <= hi)
        ratio.append(s["rmse_m"]["PR"]["equal_well"] / s["rmse_m"]["ZR"]["equal_well"])
    p = {x: n / replicates for x, n in v.items()}
    p_inf = n_inf / replicates
    sup_inf = v["response_supported"] / n_inf if n_inf else None
    return {**cell, "replicates": replicates, "p": p, "p_informative": p_inf,
            "p_supported_given_informative": sup_inf,
            "p_supported_stricter": (max(p["response_supported"], sup_inf)
                                     if sup_inf is not None else p["response_supported"]),
            "mc_se": {x: math.sqrt(q * (1 - q) / replicates) for x, q in p.items()},
            "median_beta": float(np.nanmedian(betas)),
            "beta_p10_p90": [float(np.nanpercentile(betas, 10)),
                             float(np.nanpercentile(betas, 90))],
            "coverage90_of_k": cover / replicates,
            "median_rmse_ratio_pr_over_zr": float(np.median(ratio))}


def exact_plumbing(panel: dict) -> dict:
    out = {}
    for j, name in enumerate(panel["replays"]):
        s = score_world(panel, j, 1.0 * panel["c"][j])
        out[str(name)] = {"beta": s["beta_response"]["beta"],
                          "informative": s["informative_gate"]["passes"]}
    ok = all(abs(v["beta"] - 1.0) <= 1e-9 for v in out.values() if v["informative"])
    return {"ok": bool(ok), "per_replay": out}


def replay_amplitudes(panel: dict) -> dict:
    out = {}
    for j, name in enumerate(panel["replays"]):
        c = panel["c"][j]
        ct = c - c.mean(axis=1, keepdims=True)
        out[str(name)] = {"rms_well_demeaned_c_m": float(np.sqrt(np.mean(ct ** 2))),
                          "window_mean_c_m": float(c.mean()),
                          "sd_a_changhua": float(np.std(panel["a_ch"][j]))}
    r = [v["rms_well_demeaned_c_m"] for v in out.values()]
    return {"per_replay": out, "rms_range_m": [min(r), max(r)]}


def _cell(cs: list[dict], k: float, rho: float, mult: float = 1.0) -> dict:
    return next(c for c in cs if abs(c["k"] - k) < 1e-12 and c["rho"] == rho
                and c["kappa_mult"] == mult)


def _strict(c: dict) -> float:
    """max(unconditional, given-informative) pass rate (the stricter for R1 and R2)."""
    return float(c.get("p_supported_stricter", c["p"]["response_supported"]))


def reading(cs: list[dict]) -> dict:
    c0 = _cell(cs, 0.0, -0.83)
    fp = _strict(c0)
    ws = {k: max(_strict(_cell(cs, k, r)) for r in GRID["rho"]) for k in (0.5, 2.0)}
    cov = _cell(cs, 1.0, 0.0)["coverage90_of_k"]
    pw = _cell(cs, 1.0, 0.0)["p"]["response_supported"]
    pwc = _cell(cs, 1.0, -0.83)["p"]["response_supported"]
    keys = []
    if fp >= READING["false_pass_max"]:
        keys.append("reported_not_shown")
    if (ws[0.5] >= READING["wrong_size_max"] or ws[2.0] >= READING["wrong_size_max"]
            or cov < READING["coverage_min"]):
        keys.append("rates_printed")
    keys.append("power")
    v = {"fp": fp, "ws05": ws[0.5], "ws2": ws[2.0], "cov": cov, "pw": pw, "pwc": pwc}
    return {"status": "estimated",
            "false_pass_k0_rho_m083": fp,
            "false_pass_unconditional": c0["p"]["response_supported"],
            "false_pass_given_informative": c0.get("p_supported_given_informative"),
            "wrong_size_pass": {"k0.5": ws[0.5], "k2": ws[2.0]},
            "coverage90_k1_rho0": cov, "power_k1_rho0": pw, "power_k1_rho_m083": pwc,
            "power_k1_rho0_given_informative":
                _cell(cs, 1.0, 0.0).get("p_supported_given_informative"),
            "display_keys": keys,
            "display_text": [DISPLAY[k].format(**v) for k in keys] + [DISPLAY["verdict_word"]]}


def simulate(panel: dict, replicates: int = REPLICATES, seed: int = SEED, log=print) -> dict:
    t0 = time.time()
    res = {"exact_plumbing": exact_plumbing(panel)}
    if not res["exact_plumbing"]["ok"]:
        # a code defect: no panel summary is written, so a fix cannot be shaped by the panel
        res["engineering_failure"] = "exact plumbing: beta != k with no noise"
        return res
    res["replay_amplitudes"] = replay_amplitudes(panel)
    cl = cells()
    gens = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(cl) + 1)]
    pc = {"k": 1.0, "rho": 0.0, "kappa_mult": 0.0, "noise_scale": PLUMBING_NOISE_SCALE,
          "report_only": True}
    c1 = run_cell(panel, pc, gens[len(cl)], PLUMBING_REPLICATES)
    mb = c1["median_beta"]
    res["plumbing_cell"] = {**c1, "band": list(PLUMBING_BAND),
                            "ok": bool(np.isfinite(mb) and PLUMBING_BAND[0] <= mb
                                       <= PLUMBING_BAND[1])}
    log(f"plumbing cell (k=1, rho=0, low noise): ok={res['plumbing_cell']['ok']}")
    if not res["plumbing_cell"]["ok"]:
        txt = DISPLAY["not_estimable"].format(mb=f"{mb:.3f}" if np.isfinite(mb) else "nan")
        res["reading"] = {"status": "not_estimable", "display_keys": ["not_estimable"],
                          "display_text": [txt, DISPLAY["verdict_word"]]}
        res["runtime_s"] = round(time.time() - t0, 1)
        return res
    out = []
    for i, c in enumerate(cl):
        out.append(run_cell(panel, c, gens[i], replicates))
        log(f"  cell {i + 1}/{len(cl)} done ({time.time() - t0:.0f} s)")
    res["cells"] = out
    res["full_noise_k1_rho0_median_beta_report_only"] = _cell(out, 1.0, 0.0)["median_beta"]
    res["reading"] = reading([c for c in out if not c["report_only"]])
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


# ----------------------------------------------------------------------------- registry
def registered_declaration(prereg_dir: Path | None = None) -> tuple[dict, str]:
    from common.hashing import sha256_file
    from common.paths import PREREG_DIR
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir is not None else PREREG_DIR
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == DECL_ID]
    if not rows:
        raise SystemExit(f"REFUSED: {DECL_ID} is not registered in {d}; register it first "
                         "(prereg_n2c_reading.py --write)")
    path = d / f"{DECL_ID}.json"
    sha = sha256_file(path)
    if sha != rows[-1]["sha256"]:
        raise SystemExit(f"REFUSED: {path} sha256 {sha} != INDEX {rows[-1]['sha256']}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("design") != json.loads(json.dumps(design_spec())):
        raise SystemExit("REFUSED: the registered design differs from synth_n2c.design_spec()")
    now = code_sha256()
    if doc.get("code_sha256") != now:
        bad = sorted(k for k in now if (doc.get("code_sha256") or {}).get(k) != now[k])
        raise SystemExit(f"REFUSED: code changed since registration: {bad}")
    return doc, sha


def render_md(res: dict) -> str:
    lines = [f"# {DECL_ID}: synthetic operating characteristics of N2C's rule", "",
             f"Created {res['created']}. Scorer {res['scorer']['sha256'][:12]} "
             f"(bootstrap_scorer {BS.VERSION}). Synthetic worlds only; no observation read.",
             ""]
    if "engineering_failure" in res:
        return "\n".join(lines + [f"ENGINEERING FAILURE: {res['engineering_failure']}", ""])
    if res["reading"]["status"] == "not_estimable":
        return "\n".join(lines + ["Reading (terminal, declared):"]
                         + [f"- {t}" for t in res["reading"]["display_text"]] + [""])
    lines += ["| k | rho | kappa x | P(supported) | P(direction only) | P(not supported) | "
              "P(not informative) | median beta | coverage of k |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        p = c["p"]
        lines.append(f"| {c['k']:.3g} | {c['rho']:g} | {c['kappa_mult']:g}"
                     f"{' (report only)' if c['report_only'] else ''} | "
                     f"{p['response_supported']:.3f} | {p['direction_supported_size_not']:.3f} | "
                     f"{p['not_supported']:.3f} | {p['not_informative']:.3f} | "
                     f"{c['median_beta']:.2f} | {c['coverage90_of_k']:.3f} |")
    lines += ["", "Reading:"] + [f"- {t}" for t in res["reading"]["display_text"]] + [""]
    return "\n".join(lines)


def write_result(res: dict, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "result.json"
    with p.open("x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    with (out_dir / "result.md").open("x") as fh:
        fh.write(render_md(res))
    return p, sha256(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="the c panel from the CPU replay forward (registered)")
    ex.add_argument("--forward", default=FWD_STEM + ".npz")
    ex.add_argument("--replays", default=f"{REPLAYS}/replays.json")
    ex.add_argument("--out", default=PANEL)
    ru = sub.add_parser("run", help="the registered synthetic run (once)")
    ru.add_argument("--panel", default=PANEL)
    ru.add_argument("--out-dir", default=OUT_DIR)
    sm = sub.add_parser("smoke", help="tiny run on a synthetic panel (engineering only)")
    sm.add_argument("--out", required=True)
    sm.add_argument("--replicates", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        doc, _ = registered_declaration()
        rj = rp(a.replays)
        if sha256(rj) != doc["replays"]["replays_json_sha256"]:
            raise SystemExit("REFUSED: replays.json differs from the registered one")
        info = extract(rp(a.forward), rj, rp(a.out))
        print(json.dumps(info, indent=1))
        return 0
    if a.cmd == "smoke":
        out = rp(a.out)
        if str(out).startswith(str(REPO / "results/dataprep")):
            raise SystemExit("smoke output must not go under results/dataprep")
        res = simulate(synthetic_panel(), a.replicates, seed=1)
        res.update(created=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   smoke=True, scorer=BS.describe())
        p, h = write_result(res, out)
        print(f"smoke written {p} sha256 {h}")
        return 0
    doc, decl_sha = registered_declaration()
    pan = rp(a.panel)
    out_dir = rp(a.out_dir)
    if (out_dir / "result.json").exists():
        raise SystemExit("REFUSED: result.json exists; the declaration is run once")
    panel = load_panel(pan)
    if str(panel["replays_json_sha256"]) != doc["replays"]["replays_json_sha256"]:
        raise SystemExit("REFUSED: the panel was not built from the registered replays")
    if str(panel["forward_sha256"]) != last_forward_sha(rp(ATTEMPTS)):
        raise SystemExit("REFUSED: the panel's forward is not the last successful forward "
                         "in attempts.jsonl")
    res = simulate(panel)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res.update(id=DECL_ID, declaration_sha256=decl_sha, design=design_spec(),
               code_sha256=code_sha256(), scorer=BS.describe(),
               panel={"path": a.panel, "sha256": sha256(pan),
                      "forward_sha256": str(panel["forward_sha256"])}, created=stamp)
    if "engineering_failure" in res:
        # engineering: no reading exists; the attempt is kept, result.json is not written
        f = out_dir / f"engineering_failure_{stamp.replace(':', '')}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with f.open("x") as fh:
            json.dump(res, fh, indent=1, sort_keys=True, default=float)
        print(f"ENGINEERING FAILURE: {res['engineering_failure']}; logged {f} (no reading)")
        return 3
    p, h = write_result(res, out_dir)
    print(f"written {p} sha256 {h}; display keys {res['reading']['display_keys']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
