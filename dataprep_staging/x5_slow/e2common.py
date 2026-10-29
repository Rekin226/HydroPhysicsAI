"""I1.X5SC.gate0 (coast step E2, synthetic identifiability gate 0): the frozen spec and the
helpers shared by dump_fit_heads.py, make_synthetic.py, jobmeta.py, run_e2.sh (through
e2flags.py), evaluate_e2.py and prereg_x5sc_gate0.py.

Everything they must agree on lives in SPEC. ``spec_sha256()`` is recorded by the
generator's manifests, by every job.json, by the declaration's ``frozen`` block and by the
report, so a change after registration is visible and refused.

Paths are derived from this file's location (``REPO`` = the repository root); nothing here
names a user or a host. The batch's own files live under ``results/twin_runs/par32/``
(gitignored): the frozen-code launcher ``py.sh``, the code diff, the fit heads, the
synthetic panels and the runs.

Data discipline:
- leveling panel: inventory only. ``panel_inventory`` reads the columns sid, datetime,
  x_3826, y_3826 and never elev_m; the rows are filtered to the survey window 2012-2021.
- truth heads: calibrate_coupled's own nudged hindcast (dump_fit_heads.py, clean_heads v1
  through the theta's meta, record 2012-01..2022-12), sliced to months <= 2021-12 when
  written; model output, no observation value.
- q0_nh column JSON: the zonal parameters and the in-sample leveling residual (rmse_cm,
  bias_cm), both model statistics. rings_independent_r2 is never read.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # dataprep_staging/x5_slow -> repository root

DECL_ID = "I1.X5SC.gate0.v1"
BASE_COMMIT = "1b08fb6af527961f61533afdd482283736e4915c"
PAR = "results/twin_runs/par32"              # repo-relative home of the batch
PY_SH = f"{PAR}/py.sh"                       # frozen-code launcher (PYTHONPATH = worktree)
CODE_DIFF = f"{PAR}/x5sc_code.diff"          # the working-tree diff applied on BASE_COMMIT
HEADS_NPZ = f"{PAR}/fit_heads.npz"
SYNTH_DIR = f"{PAR}/synth"
RUNS_DIR = f"{PAR}/runs"
REPORT = f"{RUNS_DIR}/e2_report.json"
LEDGER = "attempts.jsonl"                    # inside the runs dir, outside every arm dir
DECL_PATH = f"results/dataprep/preregistration/{DECL_ID}.json"

# ---------------------------------------------------------------------------------------
# frozen spec
# ---------------------------------------------------------------------------------------
TAU_A_YEARS = 20.0          # truth A aquitard time constant
SENS_TRUTH = 0.006          # truth A level sensitivity Ska/tau_a, 1/yr (0.6 cm/yr per m)
BG_TARGET_CM = 2.5          # truth A and V distal background in 2016, cm/yr
TAU_V_YEARS = 20.0          # truth V distal VEP tau (same time scale as A, no level term)
DISTAL = 2                  # column zone id of the distal zone (easting < 182 km)

SPEC: dict = {
    "id": DECL_ID,
    "heads": {
        "npz": HEADS_NPZ, "scenario": "baseline", "last_month": "2021-12",
        "source": "dump_fit_heads.py: calibrate_coupled.main itself, run with the theta and "
                  "calibrate_common below, its hindcast_with_nudging result captured (the "
                  "nudged hindcast the column fits on; record 2012-01..2022-12; the head "
                  "inputs of the theta's meta, clean_heads v1)",
        "wells_dir": "results/dataprep/twin_inputs/clean_heads/v1/wells",
        "driver": "layer mean of the 4 layers (as calibrate_coupled._predict)"},
    "q0_nh_column": "results/twin_runs/stage3_datum_gate_q0_nh/coupled_leveling/"
                    "vep_zonal_leveling.json",
    "theta": "results/twin_runs/stage3_datum_gate_q0_nh/stage3_theta.json",
    "data_src": "chou-shui-data/data",
    "panel": "ls_cache/ls-wra-lsp-obs__choushui_panel.parquet",
    "inventory_columns": ["sid", "datetime", "x_3826", "y_3826"],
    "survey_window": ["2012-01-01", "2022-01-01"],
    "truths": {
        "N": "q0_nh zonal VEP column unchanged (no slow term)",
        "A": f"q0_nh column; distal zone adds the aquitard term with tau_a {TAU_A_YEARS:g} "
             f"yr, Ska = {SENS_TRUTH:g}/yr x tau_a, h_a0 solved so the distal background "
             f"in 2016 is {BG_TARGET_CM:g} cm/yr",
        "V": f"q0_nh column; distal zone VEP tau {TAU_V_YEARS:g} yr (Ske, Skv kept), h_pc0 "
             f"> 0 solved so the distal background in 2016 is {BG_TARGET_CM:g} cm/yr; no "
             "level-dependent term"},
    "background_definition": (
        "constant-head rate: the column driven by the layer-mean truth heads up to the hold "
        "month and held at that month's value afterwards; rate = compaction change between "
        "two later months / years (cm/yr, + = sinking). Truth calibration: hold 2016-01, "
        "window 2016-01..2017-01, mean over the distal benchmarks with >= 5 surveys "
        "2012-2021 on the grid. Gate quantity: hold at the origin month, each distal "
        "benchmark pair's own months (ta, tb), mean over the pairs. Fitted and truth columns "
        "are both driven by the truth heads (no forecast heads)"),
    "noise": {"source": "q0_nh column JSON leveling.rmse_cm / bias_cm (in-sample, re-zeroed "
                        "cumulative subsidence, model statistic)",
              "rule": "residual sd s = sqrt(rmse^2 - bias^2); per-survey elevation noise "
                      "iid N(0, (s/sqrt(2))^2), so the re-zeroed synthetic series carry "
                      "noise sd s", "seed": 20261028,
              "same_draw_all_truths": True,
              "disclosure": "white per-survey noise over 31-60 day survey gaps makes "
                            "calibrate_coupled's 0.5 m/yr screen drop benchmarks the real "
                            "fit keeps (about 739 on-grid benchmarks enter a synthetic fit "
                            "vs 798 in the q0_nh fit); time-correlated noise is not "
                            "modelled"},
    "base_elev_m": 10.0,
    "origins": ["2016-04", "2019-04"],
    "withhold": "{year}-05:2022-12",
    "pairs": "score_x5nh.benchmark_pairs on the synthetic panel: last survey in the origin's "
             "year after the origin, last survey the next year, >= 0.5 yr apart, |rate| <= "
             "50 cm/yr; distal = benchmark cell in column zone distal; cell = nearest "
             "centroid within dx",
    "calibrate_common": ["--target", "leveling", "--configs", "zonal",
                         "--hpc0-guard-days", "365", "--seed", "2027", "--device", "cuda",
                         "--hindcast-gain", "1.0", "--hindcast-every", "1",
                         "--update-method", "innovation", "--apex-hold", "calibrated"],
    "forms": {
        "aq": ["--creep", "aquitard", "--creep-zones", "mid+distal",
               "--aquitard-tau-min-years", "10", "--aquitard-ha0-min", "0"],
        "vt": ["--tau-min-days-zone", "distal=3650", "--tau-max-years", "100"],
    },
    "primary_truths": ["A", "N", "V"],
    "runs": {"dir": RUNS_DIR, "report": REPORT, "ledger": f"{RUNS_DIR}/{LEDGER}",
             "launcher": PY_SH, "code_diff": CODE_DIFF, "base_commit": BASE_COMMIT},
    "gates": {
        "rate_band": [0.67, 1.5],
        "sens_band": [0.5, 2.0],
        "n_abs_cm": 0.5,
        "at_bound_frac": 0.01,
        "v_spec_max_ratio": 0.5,
        "G0-A1": "truth A: next-year distal background (hold at origin) of the fit / truth "
                 "in [0.67, 1.5] at both origins",
        "G0-A2": "truth A, aquitard form: fitted distal Ska/tau_a / 0.006 in [0.5, 2] at "
                 "both origins",
        "G0-A3": "truth A: the distal slow parameters are not at a bound at both origins "
                 "(aq: log_ska, log_tau_a, h_a0; vt: log_tau), at bound = within 1 % of the "
                 "recorded bound interval's width (fit coordinates: log for Ska, tau_a, tau; "
                 "metres for h_a0) of either bound",
        "G0-N": "truth N: |next-year distal background of the fit| < 0.5 cm/yr at both "
                "origins (no false positive)",
        "eligible_for_b": "a form enters I1.X5SC iff G0-A1, G0-N, G0-A3 and (aquitard form) "
                          "G0-A2 all pass; if no form passes, I1.X5SC is not run and D1's "
                          "fail wording applies",
        "only_vt_passes": "vt becomes I1.X5SC's primary by elimination, and D1 cannot state "
                          "a level-dependent (long-run policy) response",
        "binding_on_b": "I1.X5SC's nh_vt arm uses exactly the vt flags above (including "
                        "--tau-max-years 100) and its G6 'not at a bound' uses this "
                        "at-bound rule (1 % of the bound width in fit coordinates)",
        "G0-V (claim gate, aquitard form)": "truth V: fitted distal Ska/tau_a < 0.5 x 0.006 "
                 "at both origins. A fail means the aquitard form reports a ring-like level "
                 "sensitivity when the truth has none: its Ska/tau_a may not be described as "
                 "identified by leveling (D1 wording), whatever I1.X5SC gives; it does not "
                 "remove the form from I1.X5SC",
        "reported_only": [
            "truth V: background recovery ratio, both forms",
            "next-year distal and all-benchmark rate RMSE/bias of the fit vs the noise-free "
            "truth and vs the noisy synthetic pairs (heads = truth heads)",
            "fitted slow-term-only rate (aquitard term alone) and empirical level "
            "sensitivity (background change for +1 m held head)",
            "mid-zone aquitard parameters of the mid+distal fits",
            "synthetic leveling R2 from the column JSON (synthetic, not real data)"],
        "never_read": ["rings_independent_r2 (real rings vs a synthetic-fitted column; "
                       "replaced by a note in each arm's outputs)",
                       "the run logs' rings lines (redacted by jobmeta.py redact)"],
    },
}


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_path(rel: str | os.PathLike) -> Path:
    """An absolute path for a repo-relative one (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


def arm_name(truth: str, form: str, origin: str) -> str:
    return f"{truth}_{form}_r{origin[:4]}"


def all_arms() -> list[tuple[str, str, str, str]]:
    """(arm, truth, form, origin) of the 12 declared arms, in run order (forms interleaved
    so each of three round-robin workers gets a similar mix; aquitard fits are slower)."""
    return [(arm_name(t, f, o), t, f, o) for t in SPEC["primary_truths"]
            for o in SPEC["origins"] for f in SPEC["forms"]]


def form_flags(form: str) -> list[str]:
    return list(SPEC["forms"][form])


def fit_args(form: str, origin: str, data_dir: str, out_dir: str) -> list[str]:
    """calibrate_coupled's argv of one arm (after the python launcher)."""
    return (["-m", "hydrophysics.twin.calibrate_coupled", "--theta", SPEC["theta"]]
            + SPEC["calibrate_common"] + form_flags(form)
            + ["--withhold-surveys", SPEC["withhold"].format(year=origin[:4]),
               "--data", str(data_dir), "--out", str(out_dir)])


# ---------------------------------------------------------------------------------------
# code pin
# ---------------------------------------------------------------------------------------
def code_state() -> dict:
    """The hydrophysics package this interpreter imports (py.sh: the frozen worktree): its
    root, git HEAD, and the sha256 of every hydrophysics/**/*.py (the fits' import closure
    is a subset) with one tree hash over them."""
    import subprocess

    import hydrophysics

    root = Path(hydrophysics.__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): file_sha256(p)
             for p in sorted((root / "hydrophysics").rglob("*.py"))
             if "__pycache__" not in p.parts}
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"root": str(root), "is_main_repo": root == REPO, "git_head": head,
            "tree_sha256": tree, "n_files": len(files), "files": files}


# ---------------------------------------------------------------------------------------
# grid, zones, heads
# ---------------------------------------------------------------------------------------
class Grid:
    """What ``validate_against_leveling``/``column_zone_map`` need of a grid."""

    def __init__(self, mask: np.ndarray, x0: float, y0: float, dx: float):
        self.mask = np.asarray(mask, dtype=bool)
        self.x0, self.y0, self.dx = float(x0), float(y0), float(dx)
        rows, cols = np.nonzero(self.mask)
        self._cent = np.column_stack([self.x0 + (cols + 0.5) * self.dx,
                                      self.y0 + (rows + 0.5) * self.dx])
        self.n_active = int(self.mask.sum())

    def centroids(self) -> np.ndarray:
        return self._cent

    def cell_of(self, x: float, y: float) -> int:
        """Nearest active cell within dx (``leveling_rows``' rule), else -1."""
        d2 = ((self._cent - np.array([x, y])) ** 2).sum(1)
        c = int(np.argmin(d2))
        return c if math.sqrt(d2[c]) <= self.dx else -1


def truth_heads(npz_path: str | os.PathLike, scenario: str = "baseline",
                last_month: str = "2021-12", layers: bool = False) -> dict:
    """The layer-mean heads (A, T) float64 of months <= ``last_month``, their month starts
    and the grid (``layers``: also the (L, A, T) layer heads). Accepts the
    ``dump_fit_heads.py`` npz (heads_mean (1, L, A, T), dates, scenario_names, mask, x0,
    y0, dx); later months, if any, are sliced off at once and never used."""
    z = np.load(npz_path, allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime([str(d) for d in z["dates"]]))
    keep = dates.to_period("M") <= pd.Period(last_month, freq="M")
    n = int(keep.sum())
    if not keep[:n].all():
        raise ValueError("the npz months are not sorted")
    names = [str(s) for s in z["scenario_names"]]
    s = names.index(scenario)
    hm = z["heads_mean"]
    lay = np.asarray(hm[s, :, :, :n], dtype="float64")       # (L, A, n)
    del hm
    grid = Grid(z["mask"], float(z["x0"]), float(z["y0"]), float(z["dx"]))
    if lay.shape[1] != grid.n_active:
        raise ValueError(f"heads have {lay.shape[1]} cells, the mask {grid.n_active}")
    out = {"h": lay.mean(axis=0), "dates": dates[:n], "grid": grid,
           "n_layers": int(lay.shape[0])}
    if layers:
        out["layers"] = lay
    return out


def heads_provenance(npz_path: str | os.PathLike) -> dict:
    """The ``provenance_json`` record dump_fit_heads.py writes into its npz."""
    z = np.load(npz_path, allow_pickle=False)
    if "provenance_json" not in z.files:
        raise ValueError(f"{npz_path}: no provenance_json; not a dump_fit_heads.py npz")
    return json.loads(str(z["provenance_json"]))


def zone_map(grid: Grid, theta_path: str | os.PathLike) -> np.ndarray:
    """The column's zone id per active cell for the theta's meta (twin.zone_county)."""
    from hydrophysics.twin.zone_county import column_zone_map

    with open(theta_path, encoding="utf-8") as fh:
        meta = json.load(fh)["meta"]
    if meta.get("zone_county_split"):
        raise ValueError("E2 expects Q0's unsplit three-zone column")
    return np.asarray(column_zone_map(grid, meta), dtype="int64")


def nearest_month(dates: pd.DatetimeIndex, t: pd.Timestamp) -> int:
    """``leveling_rows``' month: argmin |month start - t|, refused beyond 45 days."""
    d = np.abs((dates - t).days)
    j = int(np.argmin(d))
    if d[j] > 45:
        raise ValueError(f"survey {t} is {d[j]} days from the nearest model month")
    return j


# ---------------------------------------------------------------------------------------
# leveling inventory (never elev_m)
# ---------------------------------------------------------------------------------------
def panel_inventory(panel_path: str | os.PathLike) -> pd.DataFrame:
    """sid, datetime, x_3826, y_3826 of the survey window's rows; elev_m is never read."""
    import pyarrow.parquet as pq

    cols = SPEC["inventory_columns"]
    t = pq.read_table(panel_path, columns=cols)
    df = t.to_pandas()
    if "elev_m" in df.columns:                       # impossible by construction
        raise RuntimeError("elev_m was read")
    df["datetime"] = pd.to_datetime(df["datetime"])
    t0, t1 = (pd.Timestamp(v) for v in SPEC["survey_window"])
    df = df[(df["datetime"] >= t0) & (df["datetime"] < t1)].copy()
    df["sid"] = df["sid"].astype(str)
    return df.sort_values(["sid", "datetime"]).reset_index(drop=True)


def benchmark_cells(inv: pd.DataFrame, grid: Grid) -> pd.Series:
    """sid -> active cell (-1 off the grid), from each benchmark's first row."""
    first = inv.groupby("sid").first()
    return pd.Series({sid: grid.cell_of(float(r.x_3826), float(r.y_3826))
                      for sid, r in first.iterrows()}, dtype="int64")


# ---------------------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------------------
def load_column_sets(path: str | os.PathLike) -> tuple[list[dict], dict]:
    """(zonal parameter sets, the JSON without rings_independent_r2)."""
    with open(path, encoding="utf-8") as fh:
        p = json.load(fh)
    p.pop("rings_independent_r2", None)              # real rings vs a synthetic fit
    if "zonal" not in p:
        raise ValueError(f"{path}: not a zonal column")
    return [dict(s) for s in p["zonal"]], p


def column_series(sets: list[dict], zone: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compaction (n, T) in metres (+ = sinking, 0 at t=0) of rows with zones ``zone`` and
    layer-mean heads ``h`` (n, T), each zone's parameter set through
    ``compaction.column_compaction`` (VEP term plus the aquitard term when present)."""
    import torch

    from hydrophysics.twin.compaction import column_compaction

    h = np.asarray(h, dtype="float64")
    out = np.zeros_like(h)
    zone = np.asarray(zone)
    for z in np.unique(zone):
        sel = zone == z
        ht = torch.as_tensor(h[sel], dtype=torch.float64)
        with torch.no_grad():
            out[sel] = column_compaction(ht, sets[int(z)]).numpy()
    return out


def predict_like_calibrate(sets: list[dict], zone: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """The same compaction through calibrate_coupled's own path: forward.column_from_params
    per zone in an nn.ModuleList, then calibrate_coupled._predict on the (n, L, T) layer
    heads (the generator's self-check of the driver and the parameter mapping)."""
    import torch

    from hydrophysics.twin.calibrate_coupled import _predict
    from hydrophysics.twin.forward import column_from_params

    model = torch.nn.ModuleList([column_from_params(s) for s in sets]).double()
    with torch.no_grad():
        return _predict(model, torch.as_tensor(np.asarray(layers), dtype=torch.float64),
                        torch.as_tensor(np.asarray(zone), dtype=torch.int64)).numpy()


def aquitard_series(p: dict, h: np.ndarray) -> np.ndarray:
    """The aquitard term alone (n, T) of one parameter set (zeros when it has none)."""
    import torch

    from hydrophysics.twin.compaction import AQUITARD_KEYS, aquitard_compaction

    h = np.asarray(h, dtype="float64")
    if not all(k in p for k in AQUITARD_KEYS):
        return np.zeros_like(h)
    ht = torch.as_tensor(h, dtype=torch.float64)
    with torch.no_grad():
        return aquitard_compaction(ht, *(torch.tensor([float(p[k])], dtype=torch.float64)
                                         for k in AQUITARD_KEYS)).numpy()


def held(h: np.ndarray, k: int, offset_m: float = 0.0) -> np.ndarray:
    """Heads equal to ``h`` up to month ``k``, then held at month ``k``'s value
    (+ ``offset_m``) for every later month."""
    out = np.array(h, dtype="float64", copy=True)
    out[:, k + 1:] = out[:, [k]] + float(offset_m)
    return out


def years_between(dates: pd.DatetimeIndex, a: int, b: int) -> float:
    return float((dates[b] - dates[a]).days) / 365.25


def truth_sets(base: list[dict], truth: str, h_a0: float | None = None,
               h_pc0: float | None = None) -> list[dict]:
    """The zonal parameter sets of truth N, A or V (``h_a0``/``h_pc0`` solved by the
    generator)."""
    sets = [dict(s) for s in base]
    if truth == "N":
        return sets
    if truth == "A":
        tau_a = TAU_A_YEARS * 365.25
        sets[DISTAL].update(log_ska=math.log(SENS_TRUTH * TAU_A_YEARS),
                            log_tau_a=math.log(tau_a), h_a0=float(h_a0))
        return sets
    if truth == "V":
        sets[DISTAL].update(log_tau=math.log(TAU_V_YEARS * 365.25), h_pc0=float(h_pc0))
        return sets
    raise ValueError(truth)


def background_cm(sets: list[dict], zone: np.ndarray, h: np.ndarray, hold: int,
                  a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex,
                  offset_m: float = 0.0) -> np.ndarray:
    """Per row: constant-head rate (cm/yr) between months ``a`` and ``b`` (arrays), the
    heads held at month ``hold`` (+ ``offset_m``) after it."""
    c = column_series(sets, zone, held(h, hold, offset_m))
    rows = np.arange(len(a))
    yrs = np.array([years_between(dates, int(i), int(j)) for i, j in zip(a, b, strict=True)])
    return (c[rows, b] - c[rows, a]) * 100.0 / yrs


def benchmark_pairs(panel: pd.DataFrame, origin: str, max_rate: float = 50.0) -> pd.DataFrame:
    """``score_x5nh.benchmark_pairs`` on a (synthetic) panel with elev_m: each benchmark's
    last survey in the origin's year after the origin and its last survey the next year,
    >= 0.5 yr apart -> observed rate (cm/yr, + = sinking), |rate| <= ``max_rate``."""
    o = pd.Period(origin, freq="M")
    y = o.year
    w = panel[(panel["datetime"] > o.end_time) & (panel["datetime"].dt.year <= y + 1)]
    rows = []
    for sid, g in w.groupby("sid"):
        g = g.sort_values("datetime")
        a = g[g["datetime"].dt.year == y]
        b = g[g["datetime"].dt.year == y + 1]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[-1], b.iloc[-1]
        yrs = (b["datetime"] - a["datetime"]).days / 365.25
        if yrs < 0.5:
            continue
        rate = (a["elev_m"] - b["elev_m"]) * 100.0 / yrs
        if abs(rate) > max_rate:
            continue
        rows.append({"sid": str(sid), "x": float(a["x_3826"]), "y": float(a["y_3826"]),
                     "ta": a["datetime"], "tb": b["datetime"], "obs": float(rate)})
    return pd.DataFrame(rows)
