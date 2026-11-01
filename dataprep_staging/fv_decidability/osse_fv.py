"""FV.decidability.v1: a synthetic decidability study (OSSE) of E02c's 2028 level and coverage
verdicts. If a model of the right kind were the truth, how often would E02c pass?

Truth worlds (model-only; no post-origin observation exists or is read):

    obs(w, t) = ZR(w, t) + D_k(w, t) + R_k(w, t; a) + e(w, t)       t in 2026-11..2027-10

* ZR: E02's registered forecast (predictions.csv sha 16ac3c37, forward cf2c7154), unchanged.
* D_k: the structural departure of a distinct-recipe theta already on disk (R0 = cwa, Q0,
  R1c = ccz, X5R1), run with E02's forward recipe (one member, no IC perturbation, climatology
  from the 2022-12 origin) at the frozen worktree, MINUS the same-code reference ZR0 (the
  page recipe: datum_gate_innov's 18 thetas, the same flags, --ic-members 0, baseline only,
  same worktree): (h_k - h_ZR0) in the window minus its mean over the well's observed
  2012-22 months (obs_mask.csv, the months E02's datum offset was fitted on), at each well's
  cell and layer (wells_158.csv of E02c). Both terms share code, IC treatment and restart, so
  D_k is the recipe difference only (not code drift since cf2c7154's commit, not the IC
  member average carried by heads_mean of the registered run). Model-only: no head value is
  read; the mask carries availability only. |h_ZR0 - h_ZR(registered)| is a declared
  report-only diagnostic.
* R_k(a): the head response of truth k to a synthetic Yunlin pumping-energy multiplier a(t)
  on 2023-01..2027-10, by linear superposition of 58 one-month impulse arms (multiplier
  1 + DELTA in one month, Yunlin cells, ``forward --recorded-pumping``). a(t) is a calendar-year
  block bootstrap of the pre-origin Yunlin in-grid all-class energy ratio to its month-of-year
  mean OVER THE POOL YEARS (pump_electricity/preorigin v3, use design_time_evidence, rows to
  2022-12; a declared freeze rule keeps only the years before any frozen year). The pool
  mean of a(t) is 1 by construction: the anomalies are interannual variability around the
  forward's climatology, with no level shift or ordered trend (stated). Rain stays at
  climatology (the forward has no recorded-rain arm): an optimism, stated.
* e: per-well AR(1) noise, sd and lag-1 coefficient from the page model's pre-origin fitted
  residuals at the 158 wells (heads/L1_clean v3 flags 0-1 to 2022-12, use
  design_time_evidence; computed by ``export`` and hashed BEFORE the declaration, look logged).
  Independent between wells: an optimism, stated. The residuals are free-run hindcast misfit,
  so they already hold structural error that D_k adds again: a pessimism, stated. P is
  therefore NOT claimed as a strict bound in either direction.

Statistics: exactly E02c's (dataprep_staging/e02c/score_e02c.py, imported, never copied):
T-level (``t_level``: beat CL and CLT, each with the 5 km block-bootstrap 90 % interval below
0; CL/CLT = E02c's frozen baselines_e02c.csv) and T-cov per county (G0 = E02's pooled
[0.85, 0.95] check AND ``county_coverage`` gate-3 clusters = well; ``cov_outcome``). All 158
wells x 12 months are observed in every world (stated).

P(pass | right-ish) = the equal-weight mean over the four truths of each truth's pass rate
(K_TRUTH worlds per truth). One reading per verdict, applied once (user ruling 2026-11-01):
P <= 0.5 -> 'weak_by_design' (the page says a fail would not refute; for the level verdict
C30's heads part stays N whatever the 2028 outcome); otherwise 'not_shown_weak'. One-sided:
there is no 'a fail is informative' reading.

Gates, in this order, before any truth-world statistic is computed:
  E   engineering: registration, frozen code, input hashes, forward metadata (truths and
      ZR0), arm files, check-draw files, the declared null tolerance recomputed, and
      g0_e02 = twin.validation.evaluate on two self-check draws (own stream); X5R1's new
      hindcast reproduces its batch-27 forward within REPRO_TOL_M (Q0 report-only). E passes
      completely before any S, N or truth-world number is computed;
  S   superposition: the impulse basis reproduces N_CHECK full forwards (the first truth-world
      anomaly draws) within SUP_TOL_M at the 158 wells in the window;
  N   null world (obs = CL + e): E02c's level false-pass rate within the tolerance declared
      from RMS(ZR - CL) and the noise inputs before any draw (``null_tolerance``).
A failed S or N writes a 'not_evaluable' report (final under this declaration); an E failure
writes nothing (engineering retry).

Subcommands (repository root, hydro env):
    osse_fv.py export --out <inputs dir> [--verify]   (lead, BEFORE registration; real data)
    osse_fv.py log-look                               (lead, after export, before registration)
    osse_fv.py arms                                   (after registration; writes arm CSVs)
    osse_fv.py specs --truth TAG                      (runner: --recorded-pumping specs;
                                                       ZR0 has none)
    osse_fv.py verify-registration                    (runner preconditions)
    osse_fv.py score                                  (after the forwards; once)
    osse_fv.py smoke --out <dir>                      (synthetic end to end; anyone)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_e02c as S  # noqa: E402

VERSION = "1.0.0"
ID = "FV.decidability.v1"
WP = "FV"
F = "dataprep_staging/fv_decidability"
SELF = f"{F}/osse_fv.py"
RUNNER = f"{F}/run_fv.sh"
PREREG = f"{F}/prereg_fv.py"
# bound by sha256 in the declaration (the scorer refuses any change)
FROZEN = (SELF, RUNNER, f"{S.E}/score_e02c.py", f"{S.E}/export_baselines.py",
          "dataprep_staging/harness/coverage3.py", "hydrophysics/twin/bands.py")

# ----------------------------------------------------------------------------- truths
TRUTHS = (("R0", "stage3_datum_gate_cwa"), ("Q0", "stage3_datum_gate_q0"),
          ("R1c", "stage3_datum_gate_ccz"), ("X5R1", "stage3_datum_gate_x5r1"))
TAGS = tuple(t for t, _ in TRUTHS)
# hindcast reproduction against an existing single-theta forward: (file, gating)
REPRO = {"X5R1": ("results/twin_forward/datum_gate_x5r1.npz", True),
         "Q0": ("results/twin_forward/datum_gate_q0.npz", False)}
# same-code reference of the page recipe (reviewer must-fix): D_k = h_k - h_ZR0
ZR0 = "ZR0"
ZR0_DIR = "stage3_datum_gate"
ZR0_THETAS = ("stage3_theta.json", "stage3_fold_thetas.json", "stage3_posterior_tn.json")
ZR0_COLUMN = "coupled_leveling/vep_zonal_leveling.json"
ZR0_NPZ = "results/twin_forward/fv_zr0.npz"

# ----------------------------------------------------------------------------- inputs
E02_PRED = f"{S.E02_DIR}/predictions.csv"
E02_PROTOCOL = f"{S.E02_DIR}/protocol.json"
E02C_BASE_DIR = "results/dataprep/reports/E02c/baselines"
E02C_BASE_SHA = {   # as registered in E02c.prospective_2026-11.v1 ('baselines.files')
    "baselines_e02c.csv": "c9eacb4e8948c725f4ae0a1e6e4bfd347eecaa3fc661ebb1c697192aaebdaf4b",
    "wells_158.csv": "8ea086330f8a9ecf5ff76313c9dc3e617e8e500775d70d7cc180f077340ac7d3"}
L1 = "results/dataprep/heads/L1_clean/v3"
L1_USE = "design_time_evidence"
HEAD_FLAGS = (0, 1)
PUMP = "results/dataprep/pump_electricity/preorigin/v3"
PUMP_USE = "design_time_evidence"
PUMP_TABLE = "county_class_monthly.parquet"
POOL_SELECT = {"county": "yunlin", "scope": "in_grid", "class": "all"}
POOL_FLAGS = (0, 4)
POOL_YEARS = (2012, 2022)
FREEZE_RTOL = 1e-6
FREEZE_RUN = 3                                # >= 3 identical consecutive months -> frozen
MIN_POOL_YEARS = 3
MIN_MASK_MONTHS = 1
INPUT_FILES = ("noise_ar1.csv", "anomaly_pool.csv", "obs_mask.csv")

# ----------------------------------------------------------------------------- design
AXIS_START = "2012-01"
T_FIT = 132                                   # 2012-01..2022-12
ORIGIN_IDX = T_FIT - 1
HORIZON = 58                                  # 2023-01..2027-10
T_RUN = T_FIT + HORIZON
ANOM_MONTHS = pd.date_range("2023-01-01", periods=HORIZON, freq="MS")
WIN = (178, 190)                              # 2026-11..2027-10 on the 2012-01 axis
N_WIN = WIN[1] - WIN[0]
DELTA = 0.2                                   # impulse arm multiplier 1 + DELTA
K_TRUTH = 250
K_NULL = 500
SEED = 20261101
P_WEAK = 0.5
SUP_TOL_M = 0.02
N_CHECK = 5
REPRO_TOL_M = 0.02
MIN_SD_MONTHS = 12
MIN_PHI_PAIRS = 24
PHI_CLIP = (0.0, 0.95)
STREAM_ANOM, STREAM_NOISE, STREAM_NULL, STREAM_SELFCHECK = 1, 2, 3, 4
N_SELFCHECK = 2
NEUTRAL_POWER = {"pass_no_gross_only": False, "fail_may_reflect_shock": False}
COUNTY_SCOPE = "yunlin"

# ----------------------------------------------------------------------------- run layout
WT_HEAD = "5f3af18"
PY_SH = "results/twin_runs/par28/py.sh"
FWD_FLAGS = ("--horizon 58 --gain 1.0 --ic-members 0 --update-method innovation "
             "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none")
RUNS = "results/twin_runs/fv_decidability"
ARMS_DIR = f"{RUNS}/arms"
ATTEMPTS = f"{RUNS}/attempts.jsonl"
REPORT_DIR = f"results/dataprep/reports/{ID}"
INPUTS_DIR = f"{REPORT_DIR}/inputs"
REPORT = f"{REPORT_DIR}/report.json"
DECL = f"results/dataprep/preregistration/{ID}.json"


def truth_npz(tag: str) -> str:
    return f"results/twin_forward/fv_truth_{tag}.npz"


def zr0_thetas() -> list[str]:
    return [f"results/twin_runs/{ZR0_DIR}/{t}" for t in ZR0_THETAS]


def zr0_column() -> str:
    return f"results/twin_runs/{ZR0_DIR}/{ZR0_COLUMN}"


# ----------------------------------------------------------------------------- wording
WORDING = {
    "level": {
        "weak_by_design": {
            "en": "Synthetic decidability study (FV.decidability.v1): a model of the right "
                  "kind passes E02c's level rule with probability {p:.2f} (optimistic in rain "
                  "at climatology and noise independent between wells; not a strict bound), "
                  "so the level test is weak by design. A fail would not refute the forecast, "
                  "and the baseline projection's head part stays narrowed (N) whatever the "
                  "2028 level outcome.",
            "zh": "合成可判定性研究（FV.decidability.v1）：正確類型的模型通過 E02c 水位規則的"
                  "機率為 {p:.2f}（偏樂觀：降雨取氣候值，各井雜訊彼此獨立；並非嚴格界限），"
                  "因此該水位檢驗在設計上即屬弱檢驗。未通過也不足以否定預測，且不論 2028 年"
                  "水位結果如何，基準推估的水位部分都維持限縮措辭（N）。"},
        "not_shown_weak": {
            "en": "Synthetic decidability study (FV.decidability.v1): a model of the right "
                  "kind passes E02c's level rule with probability {p:.2f} (optimistic in rain "
                  "at climatology and noise independent between wells; not a strict bound); "
                  "the study does not show the level test weak and adds no other reading.",
            "zh": "合成可判定性研究（FV.decidability.v1）：正確類型的模型通過 E02c 水位規則的"
                  "機率為 {p:.2f}（偏樂觀：降雨取氣候值，各井雜訊彼此獨立；並非嚴格界限）；"
                  "本研究未顯示該水位檢驗為弱檢驗，也不提供其他解讀。"}},
    "coverage": {
        "weak_by_design": {
            "en": "{county}: a model of the right kind passes E02c's coverage gate with "
                  "probability {p:.2f} in the synthetic study (not a bound: rain variability "
                  "and correlated noise could move coverage either way), so the coverage test "
                  "is weak by design there; a fail would not refute the interval.",
            "zh": "{county}：在合成研究中，正確類型的模型通過 E02c 涵蓋率檢定的機率為 {p:.2f}"
                  "（並非界限：降雨變異與井間相關雜訊可使涵蓋率往任一方向變動），因此該處的"
                  "涵蓋率檢驗在設計上即屬弱檢驗；未通過也不足以否定此區間。"},
        "not_shown_weak": {
            "en": "{county}: a model of the right kind passes E02c's coverage gate with "
                  "probability {p:.2f} in the synthetic study (not a bound: rain variability "
                  "and correlated noise could move coverage either way); not shown weak, no "
                  "other reading.",
            "zh": "{county}：在合成研究中，正確類型的模型通過 E02c 涵蓋率檢定的機率為 {p:.2f}"
                  "（並非界限：降雨變異與井間相關雜訊可使涵蓋率往任一方向變動）；未顯示為弱"
                  "檢驗，不提供其他解讀。"}},
    "not_evaluable": {
        "en": "Synthetic decidability study (FV.decidability.v1) not evaluable ({why}); E02 "
              "and E02c gain no interpretation line from it.",
        "zh": "合成可判定性研究（FV.decidability.v1）無法評估（{why}）；E02 與 E02c 不因此增加"
              "解讀說明。"},
}


class EngineeringFailure(RuntimeError):
    """An input or run is inconsistent; nothing is written (fix, log, rerun)."""


# ----------------------------------------------------------------------------- small helpers
def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_json_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=_jsonable)
        f.write("\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def rng_for(stream: int, truth: int, draw: int) -> np.random.Generator:
    """One independent stream per (stream, truth, draw), fixed by SEED."""
    return np.random.default_rng(np.random.SeedSequence([SEED, stream, truth, draw]))


def reading(p: float | None) -> str | None:
    if p is None or not np.isfinite(p):
        return None
    return "weak_by_design" if p <= P_WEAK else "not_shown_weak"


# ----------------------------------------------------------------------------- arms
def impulse_names() -> list[str]:
    return [f"imp_{m:%Y_%m}" for m in ANOM_MONTHS]


def check_plan() -> list[tuple[int, str, int]]:
    """(check j, truth tag, draw index): check j uses truth TAGS[j % 4]'s draw j // 4, i.e.
    exactly the anomaly series of that truth world."""
    return [(j, TAGS[j % len(TAGS)], j // len(TAGS)) for j in range(N_CHECK)]


def arm_names(tag: str) -> list[str]:
    """Scenario order of truth ``tag``'s forward: baseline, 58 impulses, its checks."""
    return (["baseline"] + impulse_names()
            + [f"chk_{j}" for j, t, _ in check_plan() if t == tag])


def write_impulse_csv(path: Path, month: pd.Timestamp) -> None:
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(f"month,multiplier\n{month:%Y-%m},{1.0 + DELTA!r}\n")


def write_series_csv(path: Path, mult: np.ndarray) -> None:
    mult = np.asarray(mult, float)
    if mult.shape != (HORIZON,):
        raise ValueError(f"series has shape {mult.shape}, expected ({HORIZON},)")
    rows = "".join(f"{m:%Y-%m},{float(v)!r}\n" for m, v in zip(ANOM_MONTHS, mult, strict=True))
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write("month,multiplier\n" + rows)


def read_series_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, dtype={"month": str}, float_precision="round_trip")
    if list(df["month"]) != [f"{m:%Y-%m}" for m in ANOM_MONTHS]:
        raise EngineeringFailure(f"{path}: months are not 2023-01..2027-10")
    return df["multiplier"].to_numpy(float)


def write_arms(out: Path, pool: np.ndarray) -> dict:
    """The 58 impulse CSVs and the N_CHECK check-draw CSVs, once, with arms.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, m in zip(impulse_names(), ANOM_MONTHS, strict=True):
        p = out / f"{name}.csv"
        write_impulse_csv(p, m)
        files[name] = {"file": p.name, "start": f"{m:%Y-%m}", "sha256": sha256(p)}
    for j, tag, d in check_plan():
        a, pick = draw_multipliers(pool, rng_for(STREAM_ANOM, TAGS.index(tag), d))
        p = out / f"chk_{j}.csv"
        write_series_csv(p, a)
        files[f"chk_{j}"] = {"file": p.name, "start": f"{ANOM_MONTHS[0]:%Y-%m}",
                             "sha256": sha256(p), "truth": tag, "draw": d,
                             "pool_rows_drawn": [int(x) for x in pick]}
    rec = {"id": ID, "created": _now(), "delta": DELTA, "county": COUNTY_SCOPE,
           "files": files, "osse_fv_sha256": sha256(Path(__file__))}
    _write_json_once(out / "arms.json", rec)
    return rec


def recorded_specs(tag: str, arms_dir: str = ARMS_DIR) -> list[str]:
    """``NAME=PATH@START#yunlin`` for every arm of truth ``tag`` (forward --recorded-pumping),
    in the scenario order of :func:`arm_names`."""
    meta = json.loads((rp(arms_dir) / "arms.json").read_text(encoding="utf-8"))
    out = []
    for name in arm_names(tag)[1:]:
        r = meta["files"][name]
        out.append(f"{name}={arms_dir}/{r['file']}@{r['start']}#{COUNTY_SCOPE}")
    return out


# ----------------------------------------------------------------------------- anomaly pool
def frozen_months(E: np.ndarray, scale: float) -> np.ndarray:
    """(years, 12) bool. Month (y, m) is frozen when (i) |E(y, m) - E(y-1, m)| <= FREEZE_RTOL x
    scale (year-on-year repeat), or (ii) it lies in a run of >= FREEZE_RUN consecutive months
    (calendar order, across year ends) whose values are identical to FREEZE_RTOL x scale (a
    freeze that starts mid-year). Missing months are never frozen."""
    tol = FREEZE_RTOL * scale
    fm = np.zeros(E.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        fm[1:] = np.abs(E[1:] - E[:-1]) <= tol
        flat = E.reshape(-1)
        eq = np.abs(flat[1:] - flat[:-1]) <= tol            # NaN -> False
    run = np.zeros(flat.shape, dtype=bool)
    i, n = 0, len(eq)
    while i < n:
        if not eq[i]:
            i += 1
            continue
        j = i
        while j < n and eq[j]:
            j += 1
        if j - i + 1 >= FREEZE_RUN:                         # months i..j are identical
            run[i:j + 1] = True
        i = j
    return fm | run.reshape(E.shape)


def anomaly_pool(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Yunlin in-grid all-class monthly energy -> per-year ratio to its month-of-year mean over
    the POOL years, with the declared pool rule:

    * rows 2012-01..2022-12 only (pre-origin slice, whatever the table holds);
    * a year is complete with all 12 months present and flags in POOL_FLAGS;
    * freeze rule (:func:`frozen_months`): year-on-year repeat, or a run of >= FREEZE_RUN
      identical consecutive months; the pool keeps only the complete years before the first
      year holding a frozen month (if that year is 2016 or later the pool is labelled
      '2012-16 blocks only');
    * ratio(y, m) = E(y, m) / mean over the pool years of E(., m): the pool mean of a(t) is 1
      in every calendar month, so a draw carries interannual variability around the
      forward's climatology and no level shift or ordered trend (declared; the pool-to-2012-22
      level is reported, never used);
    * at least MIN_POOL_YEARS years, else EngineeringFailure."""
    t = table.copy()
    for k, v in POOL_SELECT.items():
        t = t[t[k].astype(str) == v]
    t["date"] = t["date"].astype(str).str.slice(0, 7)
    t = t[(t["date"] >= f"{POOL_YEARS[0]}-01") & (t["date"] <= f"{POOL_YEARS[1]}-12")]
    if t.duplicated("date").any():
        raise EngineeringFailure("duplicate months in the pool series")
    years = list(range(POOL_YEARS[0], POOL_YEARS[1] + 1))
    E = np.full((len(years), 12), np.nan)
    FL = np.full((len(years), 12), -1)
    for _, r in t.iterrows():
        y, m = int(r["date"][:4]), int(r["date"][5:7])
        E[y - years[0], m - 1] = float(r["value"])
        FL[y - years[0], m - 1] = int(r["flag"])
    if not np.isfinite(E).any(axis=0).all():
        raise EngineeringFailure("a calendar month has no value in 2012-22")
    clim_all = np.nanmean(E, axis=0)
    complete = np.isfinite(E).all(axis=1) & np.isin(FL, POOL_FLAGS).all(axis=1)
    frozen_year = frozen_months(E, float(np.nanmean(E))).any(axis=1)
    first_frozen = next((years[i] for i in range(len(years)) if frozen_year[i]), None)
    in_pool = complete & np.array([first_frozen is None or y < first_frozen for y in years])
    if int(in_pool.sum()) < MIN_POOL_YEARS:
        raise EngineeringFailure(f"only {int(in_pool.sum())} pool years (< {MIN_POOL_YEARS})")
    clim = E[in_pool].mean(axis=0)
    if not (clim > 0).all():
        raise EngineeringFailure("non-positive pool climatology month")
    ratio = E / clim[None, :]
    df = pd.DataFrame(ratio, columns=[f"m{m:02d}" for m in range(1, 13)])
    df.insert(0, "year", years)
    df["complete"] = complete
    df["frozen"] = frozen_year
    df["in_pool"] = in_pool
    info = {"select": POOL_SELECT, "flags": list(POOL_FLAGS), "years": list(POOL_YEARS),
            "clim_pool_gwh": [float(x) / 1e6 for x in clim],
            "report_only_pool_level_vs_2012_22": [float(x) for x in clim / clim_all],
            "ratio": "E(y, m) / mean over the pool years of E(., m)",
            "freeze_rule": f"|E(y,m)-E(y-1,m)| <= {FREEZE_RTOL} x mean E, or a run of >= "
                           f"{FREEZE_RUN} consecutive months identical to that tolerance -> "
                           "frozen month; pool = complete years before the first year with a "
                           "frozen month",
            "frozen_years": [y for y, f in zip(years, frozen_year, strict=True) if f],
            "first_frozen_year": first_frozen,
            "label": (f"2012-16 blocks only (census frozen from {first_frozen})"
                      if first_frozen is not None and first_frozen >= 2016 else
                      "full pool" if first_frozen is None else
                      f"pool before {first_frozen} (frozen year before 2016)"),
            "pool_years": [y for y, k in zip(years, in_pool, strict=True) if k],
            "excluded_incomplete_or_flagged": [y for y, c in zip(years, complete, strict=True)
                                               if not c]}
    return df, info


def pool_matrix(pool_df: pd.DataFrame) -> np.ndarray:
    p = pool_df[pool_df["in_pool"].astype(bool)]
    return p[[f"m{m:02d}" for m in range(1, 13)]].to_numpy(float)


def draw_multipliers(pool: np.ndarray, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Calendar-year block bootstrap: each of 2023..2027 takes one pool year's 12 ratios
    (2027 its first 10) -> (58,) multipliers and the pool rows drawn."""
    pool = np.asarray(pool, float)
    pick = rng.integers(0, pool.shape[0], size=5)
    return np.concatenate([pool[j] for j in pick])[:HORIZON], pick


# ----------------------------------------------------------------------------- noise
def ar1_fit(res: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-well (sd, phi, n_months, n_pairs) of residual rows ``(W, T)``: sd = nanstd
    (ddof 1) after the well mean, NaN below MIN_SD_MONTHS; phi = lag-1 correlation of the
    centred residual over consecutive observed months, NaN below MIN_PHI_PAIRS, clipped to
    PHI_CLIP."""
    res = np.asarray(res, float)
    n = np.isfinite(res).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(res, axis=1)
        x = res - mu[:, None]
        sd = np.nanstd(x, axis=1, ddof=1)
    sd = np.where(n >= MIN_SD_MONTHS, sd, np.nan)
    a, b = x[:, 1:], x[:, :-1]
    ok = np.isfinite(a) & np.isfinite(b)
    npairs = ok.sum(1)
    num = np.where(ok, a * b, 0.0).sum(1)
    den = np.sqrt(np.where(ok, a * a, 0.0).sum(1) * np.where(ok, b * b, 0.0).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    phi = np.where(npairs >= MIN_PHI_PAIRS, np.clip(phi, *PHI_CLIP), np.nan)
    return sd, phi, n, npairs


def fill_by_county(v: np.ndarray, county: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN -> the county median, else the overall median (declared fill)."""
    orig = np.asarray(v, float)
    v = orig.copy()
    filled = ~np.isfinite(orig)
    for c in np.unique(county):
        sel = county == c
        med = (np.nanmedian(orig[sel]) if np.isfinite(orig[sel]).any()
               else np.nanmedian(orig))
        v[sel & filled] = med
    return v, filled


def noise_table(ids, county, obs_fit: np.ndarray, model_fit: np.ndarray) -> pd.DataFrame:
    """Per-well AR(1) noise inputs from pre-origin fitted residuals (obs - model, flags 0-1,
    2012-01..2022-12; the well's mean residual is its datum, removed)."""
    if obs_fit.shape != model_fit.shape or obs_fit.shape[1] != T_FIT:
        raise EngineeringFailure("residual inputs are not (W, 132)")
    sd, phi, n, npairs = ar1_fit(obs_fit - model_fit)
    county = np.asarray(county)
    sd_f, sd_fill = fill_by_county(sd, county)
    phi_f, phi_fill = fill_by_county(phi, county)
    return pd.DataFrame({"station_id": list(map(str, ids)), "county": county,
                         "sd_m": sd_f, "phi": phi_f, "n_months": n, "n_pairs": npairs,
                         "sd_filled": sd_fill, "phi_filled": phi_fill})


def simulate_ar1(sd: np.ndarray, phi: np.ndarray, m: int, rng: np.random.Generator
                 ) -> np.ndarray:
    """Stationary AR(1) per well -> (W, m)."""
    sd, phi = np.asarray(sd, float), np.asarray(phi, float)
    z = rng.standard_normal((len(sd), m))
    e = np.empty_like(z)
    e[:, 0] = sd * z[:, 0]
    s = sd * np.sqrt(1.0 - phi ** 2)
    for t in range(1, m):
        e[:, t] = phi * e[:, t - 1] + s * z[:, t]
    return e


# ----------------------------------------------------------------------------- null world
def null_tolerance(d: np.ndarray, sd: np.ndarray, phi: np.ndarray, k: int = K_NULL) -> dict:
    """Declared before any draw. Null world obs = CL + e, so per well MSE(ZR) - MSE(CL) =
    mean_t(d^2 - 2 d e) with d = ZR - CL. Equal-well mean over wells: expectation D2 =
    mean_w mean_t d^2 > 0, variance (4 / W^2) sum_w sd_w^2 d_w' P_w d_w / M_w^2 with P_w the
    AR(1) correlation matrix (wells independent, as the noise is). ZR passes only if the
    difference is below 0 (and its interval, and against CLT too), so its rate is at most
    about p = Phi(-D2 / sd_diff). Tolerance = p + 3 sqrt(p (1 - p) / K) + 3 / K, capped at 1."""
    d = np.asarray(d, float)
    W = d.shape[0]
    per, var = [], 0.0
    for w in range(W):
        ok = np.isfinite(d[w])
        if not ok.any():
            continue
        dw = d[w, ok]
        t = np.flatnonzero(ok)
        P = float(phi[w]) ** np.abs(t[:, None] - t[None, :])
        per.append(float(np.mean(dw ** 2)))
        var += 4.0 * float(sd[w]) ** 2 * float(dw @ P @ dw) / len(dw) ** 2
    nw = len(per)
    d2 = float(np.mean(per))
    sdd = math.sqrt(var) / nw
    z = d2 / sdd if sdd > 0 else math.inf
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    tol = min(1.0, p + 3.0 * math.sqrt(p * (1.0 - p) / k) + 3.0 / k)
    return {"rms_zr_minus_cl_equal_well_m": math.sqrt(d2), "mse_gap_m2": d2,
            "sd_of_gap_m2": sdd, "z": z, "p_approx": p, "k_null": int(k),
            "tolerance": tol, "n_wells": nw,
            "formula": "tolerance = p + 3 sqrt(p (1 - p) / K_NULL) + 3 / K_NULL, p = "
                       "Phi(-D2 / sd), D2 = equal-well mean of (ZR - CL)^2, sd from the "
                       "AR(1) noise inputs (wells independent); capped at 1"}


# ----------------------------------------------------------------------------- observed-month mask
def mask_columns() -> list[str]:
    return [f"{m:%Y-%m}" for m in pd.date_range(f"{AXIS_START}-01", periods=T_FIT, freq="MS")]


def mask_table(ids, obs_fit: np.ndarray) -> pd.DataFrame:
    """Availability only (1 = an observed flag 0-1 month in 2012-01..2022-12, else 0); the
    months E02's datum offset (twin.coverage.fitted_offsets) averaged over. No value."""
    obs_fit = np.asarray(obs_fit, float)
    if obs_fit.shape != (len(ids), T_FIT):
        raise EngineeringFailure("mask input is not (W, 132)")
    df = pd.DataFrame(np.isfinite(obs_fit).astype(int), columns=mask_columns())
    df.insert(0, "station_id", list(map(str, ids)))
    return df


def read_mask(path: Path, ids: list[str]) -> np.ndarray:
    df = pd.read_csv(path, dtype={"station_id": str}).set_index("station_id")
    if sorted(df.index) != sorted(ids) or list(df.columns) != mask_columns():
        raise EngineeringFailure("obs_mask.csv does not hold the prediction wells x 2012-22")
    m = df.loc[ids].to_numpy()
    if not np.isin(m, (0, 1)).all():
        raise EngineeringFailure("obs_mask.csv holds a value other than 0/1")
    m = m.astype(bool)
    if (m.sum(1) < MIN_MASK_MONTHS).any():
        raise EngineeringFailure("a well has no observed 2012-22 month in obs_mask.csv")
    return m


# ----------------------------------------------------------------------------- context
@dataclass
class Paths:
    pred: str = E02_PRED
    protocol: str = E02_PROTOCOL
    baselines: str = f"{E02C_BASE_DIR}/baselines_e02c.csv"
    wells: str = f"{E02C_BASE_DIR}/wells_158.csv"
    zr_forward: str = S.FORWARD
    zr0: str = ZR0_NPZ
    inputs_dir: str = INPUTS_DIR
    arms_dir: str = ARMS_DIR
    attempts: str = ATTEMPTS
    truths: dict = field(default_factory=lambda: {t: truth_npz(t) for t in TAGS})
    repro: dict = field(default_factory=lambda: dict(REPRO))
    report: str = REPORT


def load_context(P: Paths) -> dict:
    """Window rows of E02's registered predictions on the 158 wells, E02c's frozen CL/CLT,
    well table and blocks, the noise inputs and the anomaly pool (all aligned by well)."""
    pred = pd.read_csv(rp(P.pred), dtype={"station_id": str})
    pred["date"] = S._month(pred["date"])
    win = S.window_months(*S.WINDOW)
    pred = pred[pred["date"].isin(win)].sort_values(["station_id", "date"])
    ids = sorted(pred["station_id"].unique())
    W = len(ids)
    if len(pred) != W * N_WIN:
        raise EngineeringFailure(f"predictions: {len(pred)} window rows, expected {W} x 12")
    wells = pd.read_csv(rp(P.wells), dtype={"station_id": str}).set_index("station_id")
    if sorted(wells.index) != ids:
        raise EngineeringFailure("wells_158.csv does not hold exactly the prediction wells")
    wells = wells.loc[ids]
    if (wells["cell"] < 0).any() or (wells["layer"] < 0).any():
        raise EngineeringFailure("a well has no forward cell or layer")
    base = pd.read_csv(rp(P.baselines), dtype={"station_id": str})
    base["date"] = S._month(base["date"])
    bw = base[base["date"].isin(win)].pivot(index="station_id", columns="date")
    cl = bw["cl_m"].reindex(index=ids, columns=win).to_numpy(float)
    clt = bw["clt_m"].reindex(index=ids, columns=win).to_numpy(float)
    noise = pd.read_csv(rp(P.inputs_dir) / "noise_ar1.csv",
                        dtype={"station_id": str}).set_index("station_id")
    if sorted(noise.index) != ids:
        raise EngineeringFailure("noise_ar1.csv does not hold exactly the prediction wells")
    noise = noise.loc[ids]
    mask = read_mask(rp(P.inputs_dir) / "obs_mask.csv", ids)
    pool_df = pd.read_csv(rp(P.inputs_dir) / "anomaly_pool.csv")
    pool = pool_matrix(pool_df)
    if pool.shape[0] < MIN_POOL_YEARS or not np.isfinite(pool).all():
        raise EngineeringFailure("anomaly pool too small or not finite")

    def mat(c):
        return pred.pivot(index="station_id", columns="date")[c].reindex(
            index=ids, columns=win).to_numpy(float)

    frame0 = pred[["station_id", "date", "prediction_origin", "predicted_m", "baseline_m",
                   "lower_m", "upper_m", "model_sha256"]].reset_index(drop=True)
    blocks = dict(zip(ids, wells["block5"].astype(str), strict=True))
    for c in ("predicted_m", "lower_m", "upper_m"):
        if not np.isfinite(mat(c)).all():
            raise EngineeringFailure(f"non-finite {c} in the window")
    return {"ids": ids, "W": W, "county": np.array([S.county_of(s) for s in ids]),
            "cells": wells["cell"].to_numpy(int), "layers": wells["layer"].to_numpy(int),
            "zr": mat("predicted_m"), "lo": mat("lower_m"), "hi": mat("upper_m"),
            "cl": cl, "clt": clt, "baselines": base, "blocks": blocks,
            "sd": noise["sd_m"].to_numpy(float), "phi": noise["phi"].to_numpy(float),
            "pool": pool, "frame0": frame0, "mask": mask,
            "protocol": json.loads(rp(P.protocol).read_text(encoding="utf-8"))}


# ----------------------------------------------------------------------------- forwards
def well_heads(npz: Path, cells: np.ndarray, layers: np.ndarray) -> tuple[dict, np.ndarray]:
    """(metadata, heads_mean[:, layer_w, cell_w, :] -> (S, W, T))."""
    with np.load(npz, allow_pickle=False) as z:
        meta = {k: z[k] for k in z.files if k not in ("heads_mean", "heads_std", "subs_mean",
                                                       "subs_std", "subs_mean_by_rheology")}
        h = np.asarray(z["heads_mean"][:, layers, cells, :], float)
    return meta, h


def check_truth_meta(tag: str, meta: dict, heads: np.ndarray, arms: dict) -> dict:
    """Engineering checks of a truth forward before any of its values is used."""
    names = [str(x) for x in meta["scenario_names"]]
    want = arm_names(tag)
    probs = []
    if names != want:
        probs.append(f"scenario names differ from the declared arm order ({len(names)} vs "
                     f"{len(want)})")
    dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"])).to_period("M").to_timestamp()
    if len(dates) != T_RUN or dates[0] != pd.Timestamp(f"{AXIS_START}-01"):
        probs.append(f"dates are not {AXIS_START}..2027-10 ({len(dates)} months)")
    if int(meta["origin"]) != ORIGIN_IDX or int(meta["horizon"]) != HORIZON:
        probs.append("origin or horizon differs")
    labels = [str(x) for x in meta.get("member_labels", [])]
    if int(meta["n_members"]) != 1 or labels != [dict(TRUTHS)[tag]]:
        probs.append(f"members {int(meta['n_members'])} {labels}, expected one "
                     f"{dict(TRUTHS)[tag]}")
    opts = json.loads(str(meta["forward_options"]))
    rec = opts.get("recorded_pumping") or {}
    for name in want[1:]:
        r, a = rec.get(name), arms["files"].get(name)
        if r is None or a is None or r.get("sha256") != a["sha256"] or \
                r.get("county") != COUNTY_SCOPE or r.get("start") != a["start"]:
            probs.append(f"arm {name}: recorded series does not match arms.json")
            break
    if float(meta["gain"]) != 1.0 or float(meta["hindcast_gain"]) != 0.0:
        probs.append("gain / hindcast gain differ from E02's recipe")
    if opts.get("update_method") != "innovation" or opts.get("column_heads") != "free":
        probs.append("forward options differ from E02's recipe")
    if not np.isfinite(heads).all():
        probs.append("non-finite heads at a well cell")
    if probs:
        raise EngineeringFailure(f"truth {tag}: " + "; ".join(probs))
    return {"scenarios": len(names), "members": labels, "gate_git": json.loads(
        str(meta["gate"])).get("git_commit")}


def check_zr0_meta(meta: dict, heads: np.ndarray, zr_meta: dict) -> dict:
    """Engineering checks of the same-code page-recipe reference ZR0: baseline only, the
    registered run's parameter sets (labels and order), one member each (no IC member),
    E02's options, no recorded-pumping arm."""
    probs = []
    names = [str(x) for x in meta["scenario_names"]]
    if names != ["baseline"]:
        probs.append(f"scenarios {names}, expected ['baseline']")
    dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"])).to_period("M").to_timestamp()
    if len(dates) != T_RUN or dates[0] != pd.Timestamp(f"{AXIS_START}-01"):
        probs.append(f"dates are not {AXIS_START}..2027-10 ({len(dates)} months)")
    if int(meta["origin"]) != ORIGIN_IDX or int(meta["horizon"]) != HORIZON:
        probs.append("origin or horizon differs")
    want = [str(x) for x in zr_meta["member_labels"]]
    labels = [str(x) for x in meta.get("member_labels", [])]
    if labels != want or int(meta["n_members"]) != len(want):
        probs.append(f"members {int(meta['n_members'])} {labels[:3]}..., expected "
                     f"{len(want)} (the registered run's parameter sets, no IC member)")
    opts = json.loads(str(meta["forward_options"]))
    if opts.get("recorded_pumping"):
        probs.append("ZR0 carries a recorded-pumping arm")
    if float(meta["gain"]) != 1.0 or float(meta["hindcast_gain"]) != 0.0:
        probs.append("gain / hindcast gain differ from E02's recipe")
    zopts = json.loads(str(zr_meta["forward_options"]))
    for k in ("update_method", "column_heads", "restart_taper_km", "apex_hold",
              "column_hpc0_fast_days"):
        if opts.get(k) != zopts.get(k):
            probs.append(f"forward option {k} {opts.get(k)!r} differs from the registered "
                         f"run's {zopts.get(k)!r}")
    if not np.isfinite(heads).all():
        probs.append("non-finite heads at a well cell")
    if probs:
        raise EngineeringFailure("ZR0: " + "; ".join(probs))
    return {"scenarios": names, "members": len(labels), "gate_git": json.loads(
        str(meta["gate"])).get("git_commit")}


def _anchor(diff: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-well mean of ``diff`` over the observed fitted months (mask), (W, 1)."""
    m = np.asarray(mask, bool)
    if m.shape != (diff.shape[0], T_FIT) or (m.sum(1) < MIN_MASK_MONTHS).any():
        raise EngineeringFailure("observed-month mask is not (W, 132) with a month per well")
    fit = diff[:, :T_FIT]
    return (np.where(m, fit, 0.0).sum(1) / m.sum(1))[:, None]


def departure(h_k: np.ndarray, h_zr0: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """D_k on the window: (h_k - h_ZR0) minus its mean over each well's observed 2012-22
    months, exactly the months E02's datum offset (fitted_offsets) used, so D_k is the
    difference of the two datum-corrected series. Same code and IC treatment on both
    sides (the must-fix of the review); model-only."""
    diff = h_k[:, :T_RUN] - h_zr0[:, :T_RUN]
    return diff[:, WIN[0]:WIN[1]] - _anchor(diff, mask)


def zr0_diagnostic(h_zr0: np.ndarray, h_zr: np.ndarray, mask: np.ndarray) -> dict:
    """REPORT-ONLY: the same-code page reference against the registered run at the wells.
    Hindcast part = code drift and member averaging only (IC members perturb the origin
    restart only); window part (datum-removed as E02) adds the IC-member average."""
    diff = h_zr0[:, :T_RUN] - h_zr[:, :T_RUN]
    hc = diff[:, :T_FIT]
    win = diff[:, WIN[0]:WIN[1]] - _anchor(diff, mask)
    return {"report_only": True,
            "hindcast_max_abs_m": float(np.max(np.abs(hc))),
            "hindcast_rms_m": float(np.sqrt(np.mean(hc ** 2))),
            "window_datum_removed_max_abs_m": float(np.max(np.abs(win))),
            "window_datum_removed_rms_m": float(np.sqrt(np.mean(win ** 2))),
            "use": "disclosure only; D_k uses ZR0 so neither part enters a truth world"}


def basis(heads: np.ndarray) -> np.ndarray:
    """Unit responses (58, W, T): (impulse arm - baseline) / DELTA."""
    return (heads[1:1 + HORIZON] - heads[0][None]) / DELTA


def superpose(base: np.ndarray, G: np.ndarray, a: np.ndarray) -> np.ndarray:
    return base + np.tensordot(np.asarray(a, float) - 1.0, G, axes=(0, 0))


# ----------------------------------------------------------------------------- statistics
def g0_e02(f: pd.DataFrame, protocol: dict) -> dict:
    """E02's own checks (twin.validation.evaluate arithmetic): pooled coverage in the
    protocol bounds (G0) and the equal-station RMSE ratio to baseline_m <= max ratio."""
    st = f.groupby("station_id")
    n_st = int(st.ngroups)
    months = st["date"].nunique()
    enough = n_st >= int(protocol["min_stations"]) and bool(
        (months >= int(protocol["min_months"])).all())
    cov = float(f["hit"].mean())
    lo, hi = protocol["coverage_bounds"]
    msq = ((f["predicted_m"] - f["observed_m"]) ** 2).groupby(f["station_id"]).mean()
    bsq = ((f["baseline_m"] - f["observed_m"]) ** 2).groupby(f["station_id"]).mean()
    ratio = float(np.sqrt(msq.mean()) / np.sqrt(bsq.mean()))
    in_bounds = lo <= cov <= hi
    beats = ratio <= float(protocol["max_rmse_ratio"])
    return {"pass": (bool(in_bounds) if enough else None), "coverage": cov, "ratio": ratio,
            "e02_passed": bool(enough and in_bounds and beats)}


def world_stats(ctx: dict, obs: np.ndarray) -> dict:
    """E02c's verdicts on one world (rows = the 158 wells x 12 window months)."""
    f = ctx["frame0"].copy()
    f["observed_m"] = np.asarray(obs, float).reshape(-1)
    f = S.annotate(f)
    lev = S.t_level(f, ctx["baselines"], ctx["blocks"])
    g0 = g0_e02(f, ctx["protocol"])
    out = {"level": lev["verdict"], "g0": g0["pass"], "e02_passed": g0["e02_passed"],
           "coverage_pooled": g0["coverage"], "e02_ratio": g0["ratio"]}
    if lev["verdict"] != "not_evaluable":
        out.update({"rmse": lev["rmse_equal_well_m"],
                    "beat_cl": bool(lev["vs_CL"]["a_better_block"]
                                    and lev["rmse_equal_well_m"]["ZR"]
                                    < lev["rmse_equal_well_m"]["CL"]),
                    "beat_clt": bool(lev["vs_CLT"]["a_better_block"]
                                     and lev["rmse_equal_well_m"]["ZR"]
                                     < lev["rmse_equal_well_m"]["CLT"])})
    out["cov"] = {}
    for c in S.COUNTIES_COV:
        cc = S.county_coverage(f, c)
        out["cov"][c] = {"verdict": S.cov_outcome(g0["pass"], cc, NEUTRAL_POWER)["verdict"],
                         "coverage": cc.get("G1", {}).get("coverage")}
    return out


def evaluate_selfcheck(ctx: dict, obs: np.ndarray, protocol_path: Path, tmp_root: Path
                       ) -> dict:
    """g0_e02 against twin.validation.evaluate on the same rows (engineering); the temporary
    evaluation.csv lives under ``tmp_root`` and is deleted."""
    from hydrophysics.twin.validation import evaluate

    f = ctx["frame0"].copy()
    f["observed_m"] = np.asarray(obs, float).reshape(-1)
    f["date"] = f["date"].dt.strftime("%Y-%m-%d")
    mine = g0_e02(S.annotate(f.assign(date=pd.to_datetime(f["date"]))), ctx["protocol"])
    Path(tmp_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix=".selfcheck_") as td:
        p = Path(td) / "evaluation.csv"
        f.to_csv(p, index=False)
        r = evaluate(p, protocol_path)
    ok = (r["checks"].get("interval_coverage") == mine["pass"]
          and abs(r["coverage"] - mine["coverage"]) < 1e-12
          and abs(r["rmse_ratio"] - mine["ratio"]) < 1e-12
          and r["passed"] == mine["e02_passed"])
    if not ok:
        raise EngineeringFailure("g0_e02 disagrees with validation.evaluate on a self-check "
                                 "world")
    return {"identical": True, "rows": int(len(f))}


def _rate(xs) -> dict:
    x = np.asarray(xs, float)
    p = float(x.mean())
    return {"p": p, "mc_se": float(math.sqrt(p * (1 - p) / len(x))), "k": int(len(x))}


# ----------------------------------------------------------------------------- registration
def load_registered(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    from harness import prereg3

    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    try:
        doc, sha = prereg3.load_prereg(ID, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise EngineeringFailure(f"{ID} is not registered or not intact: {err}") from err
    bad = [p for p, h in (doc.get("frozen_code") or {}).items() if sha256(rp(p)) != h]
    if set(doc.get("frozen_code") or {}) != set(FROZEN):
        bad.append("frozen_code does not list exactly FROZEN")
    if bad:
        raise EngineeringFailure(f"frozen code changed since registration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    return doc, sha


def check_attempts(att: Path, npz: dict) -> dict:
    """Each truth npz must be the output of its last successful forward attempt."""
    if not att.exists():
        raise EngineeringFailure(f"{att} missing")
    recs = [json.loads(s) for s in att.read_text().splitlines() if s.strip()]
    out = {}
    for tag, p in npz.items():
        fw = [r for r in recs if r.get("event") == "forward" and r.get("truth") == tag]
        if not fw or int(fw[-1].get("rc", 1)) != 0:
            raise EngineeringFailure(f"no successful forward attempt for {tag}")
        if fw[-1].get("npz_sha256") != sha256(p):
            raise EngineeringFailure(f"{p} is not the output of {tag}'s last attempt")
        out[tag] = {"attempt": fw[-1].get("attempt"), "npz_sha256": fw[-1]["npz_sha256"]}
    return out


# ----------------------------------------------------------------------------- score
def score(P: Paths, *, smoke: bool = False, k_truth: int = K_TRUTH, k_null: int = K_NULL,
          decl: dict | None = None, decl_sha: str | None = None, log=print) -> dict:
    """Gates E, S, N, then the truth worlds and the readings. Returns the report (not
    written). Raises EngineeringFailure (nothing written) on E."""
    rep: dict = {"id": ID, "version": VERSION, "smoke": smoke, "created": _now(),
                 "declaration_sha256": decl_sha, "k_truth": k_truth, "k_null": k_null,
                 "seed": SEED, "delta": DELTA, "truths": dict(TRUTHS)}
    ctx = load_context(P)
    if decl is not None:
        exp = decl.get("inputs_frozen") or {}
        for name, h in exp.items():
            if sha256(rp(P.inputs_dir) / name) != h:
                raise EngineeringFailure(f"{name} differs from the declared sha256")
    arms = json.loads((rp(P.arms_dir) / "arms.json").read_text(encoding="utf-8"))
    eng: dict = {"truth_meta": {}, "repro": {}}
    zr_meta, zr_heads = well_heads(rp(P.zr_forward), ctx["cells"], ctx["layers"])
    h_zr = zr_heads[[str(x) for x in zr_meta["scenario_names"]].index("baseline")]
    del zr_heads
    zr0_meta, zr0_heads = well_heads(rp(P.zr0), ctx["cells"], ctx["layers"])
    eng["zr0_meta"] = check_zr0_meta(zr0_meta, zr0_heads, zr_meta)
    h_zr0 = zr0_heads[0]
    del zr0_heads
    T = {}
    for i, tag in enumerate(TAGS):
        meta, heads = well_heads(rp(P.truths[tag]), ctx["cells"], ctx["layers"])
        eng["truth_meta"][tag] = check_truth_meta(tag, meta, heads, arms)
        names = arm_names(tag)
        T[tag] = {"i": i, "base": heads[0], "G": basis(heads),
                  "checks": {n: heads[names.index(n)] for n in names if n.startswith("chk_")},
                  "D": departure(heads[0], h_zr0, ctx["mask"])}
        if tag in P.repro:
            path, gating = P.repro[tag]
            _, ref = well_heads(rp(path), ctx["cells"], ctx["layers"])
            err = float(np.max(np.abs(heads[0][:, :T_FIT] - ref[0][:, :T_FIT])))
            eng["repro"][tag] = {"file": str(path), "max_abs_hindcast_m": err,
                                 "tolerance_m": REPRO_TOL_M, "gating": gating,
                                 "pass": err <= REPRO_TOL_M}
            if gating and err > REPRO_TOL_M:
                raise EngineeringFailure(f"{tag} hindcast does not reproduce {path} "
                                         f"(max {err:.4f} m > {REPRO_TOL_M} m)")
        del heads
    # check-draw files carry exactly the declared draws
    for j, tag, d in check_plan():
        a, _ = draw_multipliers(ctx["pool"], rng_for(STREAM_ANOM, TAGS.index(tag), d))
        got = read_series_csv(rp(P.arms_dir) / arms["files"][f"chk_{j}"]["file"])
        if not np.allclose(a, got, rtol=1e-12, atol=0.0):
            raise EngineeringFailure(f"chk_{j}.csv is not draw {d} of truth {tag}")
    # the declared null tolerance, recomputed from the bound inputs (no draw)
    tol = null_tolerance(ctx["zr"] - ctx["cl"], ctx["sd"], ctx["phi"], k_null)
    if decl is not None:
        dtol = (decl.get("null_world") or {}).get("tolerance")
        if dtol is None or abs(dtol - tol["tolerance"]) > 1e-12:
            raise EngineeringFailure(f"null tolerance {tol['tolerance']} differs from the "
                                     f"declared {dtol}")
    # g0_e02 against twin.validation.evaluate on self-check draws (own stream; the
    # statistics of these worlds are not reported and enter no gate)
    eng["evaluate_selfcheck"] = [
        evaluate_selfcheck(ctx, ctx["cl"] + simulate_ar1(ctx["sd"], ctx["phi"], N_WIN,
                                                         rng_for(STREAM_SELFCHECK, 0, k)),
                           rp(P.protocol), rp(P.report).parent)
        for k in range(N_SELFCHECK)]
    rep["engineering"] = eng
    log("gate E passed (engineering)")
    rep["zr0_vs_registered_report_only"] = zr0_diagnostic(h_zr0, h_zr, ctx["mask"])
    del h_zr, h_zr0

    # ---- gate S: superposition
    sup = []
    for j, tag, d in check_plan():
        a, _ = draw_multipliers(ctx["pool"], rng_for(STREAM_ANOM, TAGS.index(tag), d))
        t = T[tag]
        rec = superpose(t["base"], t["G"], a)
        full = t["checks"][f"chk_{j}"]
        dw = np.abs(rec - full)[:, WIN[0]:WIN[1]]
        dp = np.abs(rec - full)[:, T_FIT:T_RUN]
        sup.append({"check": j, "truth": tag, "draw": d, "max_abs_window_m": float(dw.max()),
                    "rms_window_m": float(np.sqrt((dw ** 2).mean())),
                    "max_abs_projection_m": float(dp.max()),
                    "response_rms_window_m": float(np.sqrt(((full - t["base"])[
                        :, WIN[0]:WIN[1]] ** 2).mean()))})
    worst = max(s["max_abs_window_m"] for s in sup)
    rep["superposition"] = {"checks": sup, "max_abs_window_m": worst, "tolerance_m": SUP_TOL_M,
                            "pass": worst <= SUP_TOL_M}
    if worst > SUP_TOL_M:
        log(f"gate S FAILED: {worst:.4f} m > {SUP_TOL_M} m")
        return _close(rep, "superposition failed")
    log(f"gate S passed (max {worst:.4f} m)")

    # ---- gate N: null world (tolerance declared and rechecked under gate E)
    nulls = []
    for k in range(k_null):
        e = simulate_ar1(ctx["sd"], ctx["phi"], N_WIN, rng_for(STREAM_NULL, 0, k))
        nulls.append(world_stats(ctx, ctx["cl"] + e))
    fp = _rate([w["level"] == "level_forecast_supported" for w in nulls])
    rep["null_world"] = {
        "tolerance": tol, "level_false_pass": fp,
        "report_only": {
            "beat_cl": _rate([w.get("beat_cl", False) for w in nulls]),
            "e02_own_passed": _rate([w["e02_passed"] for w in nulls]),
            "coverage_supported": {c: _rate([w["cov"][c]["verdict"] == "coverage_supported"
                                             for w in nulls]) for c in S.COUNTIES_COV}},
        "pass": fp["p"] <= tol["tolerance"]}
    if fp["p"] > tol["tolerance"]:
        log(f"gate N FAILED: false pass {fp['p']:.3f} > {tol['tolerance']:.3f}")
        return _close(rep, "null-world false-pass rate above the declared tolerance")
    log(f"gate N passed (false pass {fp['p']:.3f} <= {tol['tolerance']:.3f})")

    # ---- truth worlds
    per: dict = {}
    for tag in TAGS:
        t = T[tag]
        ws, rms_r = [], []
        for k in range(k_truth):
            a, _ = draw_multipliers(ctx["pool"], rng_for(STREAM_ANOM, t["i"], k))
            r = np.tensordot(a - 1.0, t["G"][:, :, WIN[0]:WIN[1]], axes=(0, 0))
            e = simulate_ar1(ctx["sd"], ctx["phi"], N_WIN, rng_for(STREAM_NOISE, t["i"], k))
            ws.append(world_stats(ctx, ctx["zr"] + t["D"] + r + e))
            rms_r.append(float(np.sqrt((r ** 2).mean())))
        per[tag] = {
            "level_supported": _rate([w["level"] == "level_forecast_supported" for w in ws]),
            "beat_cl": _rate([w.get("beat_cl", False) for w in ws]),
            "beat_clt": _rate([w.get("beat_clt", False) for w in ws]),
            "coverage_supported": {c: _rate([w["cov"][c]["verdict"] == "coverage_supported"
                                             for w in ws]) for c in S.COUNTIES_COV},
            "g0_pass": _rate([bool(w["g0"]) for w in ws]),
            "e02_own_passed": _rate([w["e02_passed"] for w in ws]),
            "median_rmse_m": {m: float(np.median([w["rmse"][m] for w in ws if "rmse" in w]))
                              for m in ("ZR", "CL", "CLT")},
            "rms_departure_D_m": float(np.sqrt((t["D"] ** 2).mean())),
            "mean_rms_pumping_response_m": float(np.mean(rms_r))}
        log(f"truth {tag}: P(level) {per[tag]['level_supported']['p']:.3f}")
    rep["truth_worlds"] = per
    rep["noise_report_only"] = {"median_sd_m": float(np.median(ctx["sd"])),
                                "median_phi": float(np.median(ctx["phi"]))}

    def pooled(get):
        ps = [get(per[t]) for t in TAGS]
        se = math.sqrt(sum(x[1] ** 2 for x in ps)) / len(ps)
        return {"p": float(np.mean([x[0] for x in ps])), "mc_se": se,
                "by_truth": {t: x[0] for t, x in zip(TAGS, ps, strict=True)},
                "min_over_truths": float(min(x[0] for x in ps))}

    P_level = pooled(lambda r: (r["level_supported"]["p"], r["level_supported"]["mc_se"]))
    P_cov = {c: pooled(lambda r, c=c: (r["coverage_supported"][c]["p"],
                                       r["coverage_supported"][c]["mc_se"]))
             for c in S.COUNTIES_COV}
    P_e02 = pooled(lambda r: (r["e02_own_passed"]["p"], r["e02_own_passed"]["mc_se"]))
    rep["P"] = {"level": P_level, "coverage": P_cov, "e02_own_report_only": P_e02,
                "definition": "equal-weight mean over the four truths of each truth's pass "
                              f"rate over {k_truth} worlds; the reading uses the point "
                              "estimate"}
    rl = reading(P_level["p"])
    rc = {c: reading(P_cov[c]["p"]) for c in S.COUNTIES_COV}
    rep["readings"] = {
        "threshold": P_WEAK, "rule": "P <= 0.5 -> weak_by_design; otherwise not_shown_weak "
                                     "(one-sided; applied once)",
        "level": rl, "coverage": rc,
        "wording": {"level": {k: v.format(p=P_level["p"])
                              for k, v in WORDING["level"][rl].items()},
                    "coverage": {c: {k: v.format(p=P_cov[c]["p"], county=c)
                                     for k, v in WORDING["coverage"][rc[c]].items()}
                                 for c in S.COUNTIES_COV}},
        "page_effect": ("level weak_by_design: E02/E02c's level line carries the wording and "
                        "C30's heads part stays N whatever the 2028 level outcome; "
                        "not_shown_weak: the line states P only. Coverage readings add a "
                        "line per county; E02c's C14a mapping is unchanged.")}
    rep["status"] = "scored"
    return rep


def _close(rep: dict, why: str) -> dict:
    rep["status"] = "not_evaluable"
    rep["why"] = why
    rep["readings"] = {"level": None, "coverage": None,
                       "wording": {k: v.format(why=why)
                                   for k, v in WORDING["not_evaluable"].items()}}
    return rep


# ----------------------------------------------------------------------------- export (lead)
def _zr_forward() -> dict:
    with np.load(REPO / S.FORWARD, allow_pickle=False) as z:
        fw = {k: z[k] for k in ("heads_mean", "scenario_names", "dates", "origin")}
    if sha256(REPO / S.FORWARD) != S.MODEL_SHA:
        raise EngineeringFailure("ZR forward sha differs from E02's registered model")
    return fw


def build_inputs(out: Path) -> dict:
    """REAL DATA (lead, before registration): noise_ar1.csv (pre-origin residuals of the page
    model at the 158 wells; heads/L1_clean v3, use design_time_evidence, window_end 2022-12,
    flags 0-1) and anomaly_pool.csv (pump_electricity/preorigin v3, use design_time_evidence,
    rows 2012-01..2022-12) and obs_mask.csv (availability only: which 2012-22 months are
    observed at each of the 158 wells, the months E02's datum used; no value). Written
    once, with summary.json."""
    import export_baselines as EB
    from common import guard

    for n, h in E02C_BASE_SHA.items():
        if sha256(REPO / E02C_BASE_DIR / n) != h:
            raise EngineeringFailure(f"E02c {n} differs from its registered sha256")
    if sha256(REPO / E02_PRED) != S.E02_PRED_SHA:
        raise EngineeringFailure("E02 predictions changed")
    wells = pd.read_csv(REPO / E02C_BASE_DIR / "wells_158.csv", dtype={"station_id": str})
    wells = wells.sort_values("station_id").reset_index(drop=True)
    ids = list(wells["station_id"])
    fw = _zr_forward()
    sc = [str(x) for x in fw["scenario_names"]].index("baseline")
    h = np.asarray(fw["heads_mean"][sc][wells["layer"].to_numpy(int),
                                        wells["cell"].to_numpy(int), :], float)
    heads = guard.open_product(REPO / L1, L1_USE, window_end=S.ORIGIN)
    long = heads.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag"])
    long = long[long["flag"].isin(HEAD_FLAGS)]
    ax = pd.date_range(f"{AXIS_START}-01", periods=T_FIT, freq="MS")
    obs = EB.obs_matrix(long, ids, ax, t_fit=T_FIT)
    noise = noise_table(ids, wells["county"].to_numpy(), obs[:, :T_FIT], h[:, :T_FIT])
    mask = mask_table(ids, obs[:, :T_FIT])
    pump = guard.open_product(REPO / PUMP, PUMP_USE)
    tab = pump.read_table(PUMP_TABLE, columns=["county", "scope", "class", "date", "value",
                                               "flag"])
    pool_df, pinfo = anomaly_pool(tab)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("noise_ar1.csv", noise), ("anomaly_pool.csv", pool_df),
                     ("obs_mask.csv", mask)):
        with open(out / name, "x", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, float_format="%.17g")
    summ = {"created": _now(), "id": ID,
            "what": "OSSE noise inputs (per-well AR(1) of pre-origin fitted residuals), "
                    "the Yunlin pumping-anomaly pool and the observed-month mask",
            "heads": {"product": L1, "use": L1_USE, "window_end": S.ORIGIN,
                      "flags": list(HEAD_FLAGS), "notes": heads.notes,
                      "model": f"{S.FORWARD} baseline heads_mean at wells_158 cell/layer "
                               f"(sha {S.MODEL_SHA[:12]})",
                      "n_wells": len(ids), "sd_filled": int(noise["sd_filled"].sum()),
                      "phi_filled": int(noise["phi_filled"].sum()),
                      "rules": {"min_sd_months": MIN_SD_MONTHS,
                                "min_phi_pairs": MIN_PHI_PAIRS, "phi_clip": list(PHI_CLIP),
                                "fill": "county median, else overall median"},
                      "mask": "obs_mask.csv: 1 where a flag 0-1 month exists in 2012-01.."
                              "2022-12 (the months of E02's datum offset), else 0; "
                              f"{int(mask.iloc[:, 1:].to_numpy().sum())} well-months"},
            "pump": {"product": PUMP, "use": PUMP_USE, "table": PUMP_TABLE,
                     "slice": "rows 2012-01..2022-12 (filtered in code; the product declares "
                              "no time_slicing and covers 2007-01..2022-12)",
                     "notes": pump.notes, **pinfo},
            "files": {n: sha256(out / n) for n in INPUT_FILES},
            "code": {SELF: sha256(Path(__file__))}}
    _write_json_once(out / "summary.json", summ)
    return summ


def verify_inputs(out: Path) -> dict:
    """Rebuild into a temporary directory and compare the CSVs bit for bit (verify.json)."""
    out = Path(out)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".verify_") as td:
        build_inputs(Path(td) / "rebuild")
        got = {n: sha256(Path(td) / "rebuild" / n) for n in INPUT_FILES}
    have = {n: sha256(out / n) for n in INPUT_FILES}
    probs = [n for n in INPUT_FILES if not (s["files"][n] == have[n] == got[n])]
    rec = {"verified": _now(), "identical": not probs, "files": have, "problems": probs,
           "summary_sha256": sha256(out / "summary.json"), "code": {SELF: sha256(Path(__file__))}}
    _write_json_once(out / "verify.json", rec)
    return rec


LOOK_UNIT = "FV:preorigin_residuals_R0wells"
LOOK_SCOPE = "heads:2012-01..2022-12"
LOOK_PURPOSE = "design_time_noise_inputs"


def look_members(ids) -> dict:
    return {LOOK_UNIT: {"ids": sorted(map(str, ids)), "window": "2012-01..2022-12",
                        "description": "pre-origin monthly heads (heads/L1_clean v3, flags "
                                       "0-1) at E02's 158 wells, read once by osse_fv.py "
                                       "export to compute per-well AR(1) residual noise "
                                       "inputs (sd, lag-1) of the page model and the "
                                       "observed-month mask (availability) for "
                                       "FV.decidability.v1; design-time evidence, spent "
                                       "months (2012-22 are in-sample for every theta)"}}


def log_look(prereg_dir=None, rows_dir=None, inputs_dir=INPUTS_DIR,
             wells_csv=f"{E02C_BASE_DIR}/wells_158.csv") -> dict:
    """Log the design-time look of ``export`` (the lead; real registry by default)."""
    from harness import prereg3

    s = json.loads((rp(inputs_dir) / "summary.json").read_text(encoding="utf-8"))
    wells = pd.read_csv(rp(wells_csv), dtype={"station_id": str})
    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    return prereg3.record_look(
        [LOOK_UNIT], LOOK_SCOPE, WP, LOOK_PURPOSE, members=look_members(wells["station_id"]),
        note=f"{ID} noise inputs: {inputs_dir}/noise_ar1.csv sha256 "
             f"{s['files']['noise_ar1.csv']}; the anomaly pool (pump_electricity/preorigin v3, "
             f"design_time_evidence) sha256 {s['files']['anomaly_pool.csv']}; the "
             f"observed-month mask sha256 {s['files'].get('obs_mask.csv')}", **kw)




# ----------------------------------------------------------------------------- synthetic smoke
SYNTHETIC_MARKER = "SYNTHETIC"
Z90 = 1.6448536269514722


def _save_npz(path: Path, heads: np.ndarray, names: list[str], label: str, recorded: dict
              ) -> None:
    T = heads.shape[-1]
    dates = pd.date_range(f"{AXIS_START}-01", periods=T, freq="MS")
    np.savez_compressed(
        path, heads_mean=heads.astype(np.float32), heads_std=np.zeros_like(heads, np.float32),
        scenario_names=np.array(names), scenarios=np.array(names),
        dates=np.array([str(d) for d in dates]), origin=ORIGIN_IDX, horizon=T - T_FIT,
        n_members=1, member_labels=np.array([label]), gain=1.0, hindcast_gain=0.0,
        gate=json.dumps({"git_commit": "synthetic"}),
        forward_options=json.dumps({"update_method": "innovation", "column_heads": "free",
                                    **({"recorded_pumping": recorded} if recorded else {})}))


def build_synthetic(out: Path, n_yunlin: int = 64, n_changhua: int = 60, seed: int = 7,
                    nonlinear: float = 0.01) -> Paths:
    """A complete SYNTHETIC input set with the real shapes and file formats (no real data):
    predictions, protocol, E02c-style baselines and well table, noise and pool inputs, the
    arm files, a ZR forward and four truth forwards whose impulse responses are exponential
    plus a small quadratic term (``nonlinear``) so the superposition gate is exercised."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / SYNTHETIC_MARKER).write_text("synthetic OSSE smoke inputs; no real data\n")
    rng = np.random.default_rng(seed)
    ids = sorted([f"09{i:06d}" for i in range(n_yunlin)]
                 + [f"07{i:06d}" for i in range(n_changhua)])
    W = len(ids)
    cells = rng.permutation(W)
    layers = rng.integers(0, 4, W)
    x = rng.uniform(170000, 215000, W)
    y = rng.uniform(2600000, 2650000, W)
    wells = pd.DataFrame({"station_id": ids, "county": [S.county_of(s) for s in ids],
                          "x": x, "y": y, "layer": layers, "cell": cells,
                          "block5": S.block_ids(x, y)})
    wells.to_csv(out / "wells_158.csv", index=False)
    t = np.arange(252, dtype=float)
    season = np.sin(2 * np.pi * t / 12.0)
    level = rng.normal(10.0, 5.0, W)
    amp = rng.uniform(0.5, 2.0, W)
    trend = rng.normal(0.0, 0.08, W) / 12.0
    proj = np.where(t > ORIGIN_IDX, 0.5, 1.0)          # the model's post-origin slowdown
    h_zr = (level[:, None] + amp[:, None] * season[None]
            + trend[:, None] * (t[None] - ORIGIN_IDX) * proj[None])
    hm = np.zeros((1, 4, W, 252))
    hm[0, layers, cells, :] = h_zr
    _save_npz(out / "zr.npz", hm, ["baseline"], "stage3_datum_gate", {})
    # ZR0: the same recipe under "new code, no IC member": a small post-origin offset
    h0 = h_zr + 0.01 * rng.normal(0.0, 1.0, W)[:, None] * (t[None] > ORIGIN_IDX)
    hm0 = np.zeros((1, 4, W, T_RUN))
    hm0[0, layers, cells, :] = h0[:, :T_RUN]
    _save_npz(out / "zr0.npz", hm0, ["baseline"], "stage3_datum_gate", {})
    win = slice(*WIN)
    off = rng.normal(0.0, 1.0, W)
    tf = np.arange(T_FIT)
    sd = rng.uniform(0.3, 1.0, W)
    phi = rng.uniform(0.3, 0.8, W)
    cl = level[:, None] + off[:, None] + amp[:, None] * season[None, win] \
        + (trend * (tf.mean() - ORIGIN_IDX))[:, None]
    clt = cl + trend[:, None] * (t[None, win] - tf.mean())
    zr = h_zr[:, win] + off[:, None]
    sig = np.sqrt(sd ** 2 + 0.2 ** 2)[:, None] * np.ones((1, N_WIN))
    months = S.window_months(*S.WINDOW)
    rows, brows = [], []
    for w, s in enumerate(ids):
        for j, m in enumerate(months):
            rows.append({"station_id": s, "layer": int(layers[w]) + 1,
                         "date": f"{m:%Y-%m-%d}", "prediction_origin": "2022-12-01",
                         "predicted_m": zr[w, j], "baseline_m": cl[w, j],
                         "lower_m": zr[w, j] - Z90 * sig[w, j],
                         "upper_m": zr[w, j] + Z90 * sig[w, j],
                         "model_sha256": "synthetic"})
            brows.append({"station_id": s, "date": f"{m:%Y-%m-%d}", "cl_m": cl[w, j],
                          "clt_m": clt[w, j]})
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "baselines_e02c.csv", index=False)
    proto = {"schema": 1, "test_start": "2026-11-01", "test_end": "2027-10-31",
             "registered_before": "2026-10-11", "independent_holdout": True,
             "model_sha256": "synthetic", "baseline": "synthetic climatology",
             "max_rmse_ratio": 1.0, "min_stations": 100, "min_months": 6,
             "coverage_bounds": [0.85, 0.95]}
    (out / "protocol.json").write_text(json.dumps(proto, indent=1) + "\n")
    inp = out / "inputs"
    inp.mkdir()
    pd.DataFrame({"station_id": ids, "county": wells["county"], "sd_m": sd, "phi": phi,
                  "n_months": 132, "n_pairs": 131, "sd_filled": False, "phi_filled": False}
                 ).to_csv(inp / "noise_ar1.csv", index=False, float_format="%.17g")
    avail = rng.uniform(size=(W, T_FIT)) < 0.85
    avail[:, :24] = True
    mask_table(ids, np.where(avail, 0.0, np.nan)).to_csv(inp / "obs_mask.csv", index=False)
    years = np.arange(2012, 2023)
    ratios = np.clip(1.0 + 0.12 * rng.standard_normal((len(years), 12)), 0.5, 1.5)
    pool_df = pd.DataFrame(ratios, columns=[f"m{m:02d}" for m in range(1, 13)])
    pool_df.insert(0, "year", years)
    pool_df["complete"] = True
    pool_df["frozen"] = years >= 2017
    pool_df["in_pool"] = years < 2017
    pool_df.to_csv(inp / "anomaly_pool.csv", index=False, float_format="%.17g")
    arms = write_arms(out / "arms", pool_matrix(pool_df))
    gain = rng.uniform(-0.3, -0.05, W)
    tau = rng.uniform(3.0, 18.0, W)
    tr = np.arange(T_RUN, dtype=float)

    def irf(m_idx: int) -> np.ndarray:            # unit response to a month-m multiplier
        lag = tr[None] - m_idx
        return np.where(lag >= 0, gain[:, None] * np.exp(-np.maximum(lag, 0) / tau[:, None]),
                        0.0)

    imp = [irf(T_FIT + k) for k in range(HORIZON)]
    paths = {}
    for i, (tag, label) in enumerate(TRUTHS):
        names = arm_names(tag)
        dstruct = rng.normal(0.0, 0.1 + 0.1 * i, W)
        base = h_zr[:, :T_RUN] + 0.3 * rng.normal(0, 1, W)[:, None] \
            + dstruct[:, None] * np.clip((tr[None] - ORIGIN_IDX) / HORIZON, 0, None)
        H = np.zeros((len(names), 4, W, T_RUN))
        H[0][layers, cells, :] = base
        for k in range(HORIZON):
            r = DELTA * imp[k]
            H[1 + k][layers, cells, :] = base + r + nonlinear * r ** 2
        rec = {}
        for n in names[1:]:
            a = arms["files"][n]
            rec[n] = {"sha256": a["sha256"], "county": COUNTY_SCOPE, "start": a["start"]}
            if n.startswith("chk_"):
                mult = read_series_csv(out / "arms" / a["file"])
                r = sum((mult[k] - 1.0) * imp[k] for k in range(HORIZON))
                H[names.index(n)][layers, cells, :] = base + r + nonlinear * r ** 2
        paths[tag] = out / f"truth_{tag}.npz"
        _save_npz(paths[tag], H, names, label, rec)
    repro = {}
    for tag, gating in (("X5R1", True), ("Q0", False)):
        p = out / f"batch27_{tag}.npz"
        with np.load(paths[tag], allow_pickle=False) as z:
            hb = np.asarray(z["heads_mean"][:1], float)
        _save_npz(p, hb + (0.0 if gating else 0.003), ["baseline"], dict(TRUTHS)[tag], {})
        repro[tag] = (str(p), gating)
    return Paths(pred=str(out / "predictions.csv"), protocol=str(out / "protocol.json"),
                 baselines=str(out / "baselines_e02c.csv"), wells=str(out / "wells_158.csv"),
                 zr_forward=str(out / "zr.npz"), zr0=str(out / "zr0.npz"),
                 inputs_dir=str(inp),
                 arms_dir=str(out / "arms"), attempts=str(out / "attempts.jsonl"),
                 truths={t: str(p) for t, p in paths.items()}, repro=repro,
                 report=str(out / "smoke_result.json"))


def run_smoke(out: Path, k_truth: int = 20, k_null: int = 40, nonlinear: float = 0.01) -> dict:
    """Synthetic end to end (no registry, no real data): writes out/smoke_result.json once."""
    out = Path(out)
    P = build_synthetic(out / "inputs_synthetic", nonlinear=nonlinear)
    rep = score(P, smoke=True, k_truth=k_truth, k_null=k_null)
    rep["note"] = "SMOKE: synthetic inputs only; numbers mean nothing about the twin"
    _write_json_once(out / "smoke_result.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="lead, before registration: noise + pool (real data)")
    e.add_argument("--out", default=INPUTS_DIR)
    e.add_argument("--verify", action="store_true")
    sub.add_parser("log-look", help="lead, after export, before registration")
    sub.add_parser("arms", help="after registration: arm CSVs")
    sp = sub.add_parser("specs", help="runner: --recorded-pumping specs of one truth")
    sp.add_argument("--truth", required=True, choices=TAGS)
    sub.add_parser("verify-registration")
    sc = sub.add_parser("score", help="after the forwards; once")
    sc.add_argument("--out", default=REPORT)
    sm = sub.add_parser("smoke", help="synthetic end to end")
    sm.add_argument("--out", required=True)
    sm.add_argument("--k-truth", type=int, default=20)
    sm.add_argument("--k-null", type=int, default=40)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out), k_truth=a.k_truth, k_null=a.k_null)
            print(json.dumps({"status": rep["status"], "readings": rep["readings"],
                              "P": rep.get("P")}, indent=1, ensure_ascii=False,
                             default=_jsonable))
            return 0
        if a.cmd == "export":
            r = verify_inputs(rp(a.out)) if a.verify else build_inputs(rp(a.out))
            print(json.dumps({k: v for k, v in r.items() if k not in ("heads", "pump")},
                             indent=1, default=_jsonable))
            return 0 if (not a.verify or r["identical"]) else 1
        if a.cmd == "log-look":
            print(json.dumps(log_look(), indent=1, default=_jsonable))
            return 0
        if a.cmd == "specs":
            print("\n".join(recorded_specs(a.truth)))
            return 0
        doc, dsha = load_registered()
        if a.cmd == "verify-registration":
            print(f"{ID} registered (sha256 {dsha[:16]}); frozen code and inputs intact")
            return 0
        if a.cmd == "arms":
            if rp(ARMS_DIR).exists():
                raise EngineeringFailure(f"{ARMS_DIR} exists (archive it first)")
            pool = pool_matrix(pd.read_csv(REPO / INPUTS_DIR / "anomaly_pool.csv"))
            r = write_arms(REPO / ARMS_DIR, pool)
            print(f"wrote {len(r['files'])} arm files to {ARMS_DIR} (arms.json sha256 "
                  f"{sha256(REPO / ARMS_DIR / 'arms.json')})")
            return 0
        if a.cmd == "score":
            out = rp(a.out)
            if out.exists():
                raise EngineeringFailure(f"{out} exists: {ID} is scored once")
            P = Paths()
            att = check_attempts(rp(P.attempts), {**{t: rp(p) for t, p in P.truths.items()},
                                                  ZR0: rp(P.zr0)})
            rep = score(P, decl=doc, decl_sha=dsha)
            rep["attempts"] = att
            rep["frozen_code"] = doc["frozen_code"]
            _write_json_once(out, rep)
            print(f"wrote {a.out} (status {rep['status']}; level reading "
                  f"{rep['readings']['level']})")
            return 0
    except EngineeringFailure as err:
        print(f"ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
