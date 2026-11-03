"""CB.coverage_side.v1 (round-8 rank 10): on which side does an E02c Changhua coverage fail
fall, once DR's per-well drift and U1.HB.power's year / month shocks are overlaid on FV v2's
synthetic worlds? A NEW estimand (the side distribution of E02c's Changhua coverage outcome
by truth family), declared on its own; it is not FV v3 and changes no FV v2 reading.

Model-only, CPU. No observation is read, no forward is run.

World k of truth T (T in R0, Q0, R1c, X5R1; k < 250; FV v2's 1000 hashed full forwards):

    obs(w, t) = ZR(w, t) + D_T(w, t) + R_T,k(w, t) + n(w, t) + drift(w, t)

* ZR, D_T, R_T,k: exactly FV.decidability.v2_postfailure's (osse_fv2: E02's registered
  predicted_m; D_T = batch-0 baseline minus the same-code reference ZR0, anchored on each
  well's observed 2012-22 months; R_T,k = world k minus its own batch baseline in the window).
* n: FV v2's per-well AR(1) noise e (noise_ar1.csv sd, phi; osse_fv.simulate_ar1) with
  U1.HB.power.v1's year and month shares overlaid on the same marginal sd:

      n = sqrt(a_y) sd C + sqrt(1 - a_y) (sqrt(a_m) sd B_t + sqrt(1 - a_m) e)

  C ~ N(0, 1) one per world (common to every well-month, both counties), B_t ~ N(0, 1) one
  per month (common to every well); (a_y, a_m) on U1.HB.power.v1's grid ICC_year x ICC_month
  (power_u1hb, imported unchanged). At (0, 0) n is e bit for bit.
* drift: DR.relative_drift.v1's bound linear drift of each well's group-relative offset,
  b_w (t - 2023.0) with t the mid-month decimal year (DR's lead definition 'origin'):
  b = sqrt(tau_c^2) x z centred within the well's DR zone group and rescaled by
  sqrt(n_g / (n_g - 1)) (z ~ N(0, 1) per well; tau_c^2 from DR's report: Changhua 0.0200,
  Yunlin 0.0185 per yr^2; groups from wells_e02d.csv). Arms: 'none', 'origin' (gating) and
  'datum_centre_report_only' (DR's report-only lead from the datum centre: each county's
  median shift, lead_years_datum_centre_report_only - lead_years = +5.5 years). DR's tau is
  group-relative only: no drift common to a whole group or county is overlaid (stated).

Fresh seed (the lead's, no default): streams NOISE 11, YEAR 12, MONTH 13, DRIFT 14, each
SeedSequence([seed, stream, truth index, k]); common random numbers across ICC cells and
drift arms, so each world's no-shock outcome is its shocked outcome's pair.

Outcome per world (E02c's frozen code, imported, never copied): G0 = E02's pooled check
(osse_fv.g0_e02), score_e02c.county_coverage('Changhua') and score_e02c.cov_outcome with
neutral power flags. A 'coverage_not_supported' outcome carries E02c's frozen side ('below'
or 'above' the registered range, read back from its wording and verified): 'above' -> OVER,
'below' -> UNDER. 'coverage_supported' -> PASS; 'pooled_gate_failed_county_report_only' ->
POOLED_ONLY (no side); anything else -> NOT_EVALUABLE.

Estimand: P(over | T, cell, arm) = share of truth T's 250 worlds whose Changhua outcome is
OVER (unconditional: the probability of the 2028-observable outcome 'fail by
over-coverage'). The conditional side among non-passes is report-only.

Gates, in order:
  E0 engineering (before any fresh draw): with FV v2's own seed and noise stream, cell (0, 0)
     and no drift, the engine reproduces report_worlds.json P.coverage.<county>.by_truth for
     all four truths in both counties exactly (1e-12). Failure: nothing written (retry).
  E1 engineering (fresh seed): Q0 Changhua no-drift no-shock pass rate within 2 Monte Carlo
     SE of v2's published 0.004 (SE of the difference of two binomial rates at the pooled
     rate). Failure: a final not_evaluable report (no retry on another seed). Any failure
     after E1 passed (an exception in the grid or its statistics) is also a FINAL
     not_evaluable report: the fresh draws have been made, so nothing is re-run.
  CB corrected one-sided gate: min over T in {Q0, R0} and every ICC cell, drift 'origin', of
     P(over) >= 0.8 AND max over T in {R1c, X5R1}, every ICC cell and drift 'none' and
     'origin', of P(over) <= 0.2. Point estimates; margins reported in MC SE units.
Reported explicitly: the bias-cancelled-by-shock probability P(world fails at (0, 0) and
passes at the cell, same drift arm), per truth, arm and cell.

Readings apply to C14a Changhua only; there is no C20/C21 mapping (N2C decides response
class). No branch writes a page line: ledger / README / MODEL_CARD only (user ruling R8.8
named neither CB nor C14a and kept the FV v2 per-truth split off the page; a page line needs a
new user ruling).

Subcommands (repository root, hydro env):
    cb_side.py verify-registration
    cb_side.py run --seed N [--workers W]      once, after registration (writes report.json)
    cb_side.py smoke --out DIR [--workers W]   synthetic end to end (anyone; no real data)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability_v2"),
           str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging" / "u1_bands"),
           str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osse_fv as O  # noqa: E402  (FV v1, frozen; bound by sha256)
import osse_fv2 as V2  # noqa: E402  (FV v2, frozen; bound by sha256)
import power_u1hb as U  # noqa: E402  (U1.HB.power.v1, frozen; its ICC grid)
import score_e02c as S  # noqa: E402  (E02c, frozen; its coverage outcome and side)

VERSION = "1.0.0"
ID = "CB.coverage_side.v1"
WP = "CB"
F = "dataprep_staging/cb_side"
SELF = f"{F}/cb_side.py"
PREREG = f"{F}/prereg_cb.py"
TESTS = "tests/test_cb_side.py"
FROZEN = (SELF, V2.SELF, O.SELF, f"{S.E}/score_e02c.py", "dataprep_staging/harness/coverage3.py",
          "hydrophysics/twin/bands.py", "dataprep_staging/u1_bands/power_u1hb.py")
EngineeringFailure = O.EngineeringFailure

# ----------------------------------------------------------------------------- design
TAGS = O.TAGS                                   # ('R0', 'Q0', 'R1c', 'X5R1'): FV's order
PAGE_FAMILY = ("Q0", "R0")                      # no Changhua pumping response
ALT_FAMILY = ("R1c", "X5R1")                    # Changhua recipes
COUNTY = "Changhua"
K = V2.K_TRUTH                                  # 250 worlds per truth
ICC_YEAR = tuple(float(x) for x in U.ICC_YEAR)  # (0.0, 0.1, 0.2)
ICC_MONTH = tuple(float(x) for x in U.ICC_MONTH)  # (0.0, 0.05, 0.15, 0.3)
DRIFT_ARMS = ("none", "origin", "datum_centre_report_only")
GATE_A_ARMS = ("origin",)
GATE_B_ARMS = ("none", "origin")
ORIGIN_YEAR = 2023.0                            # DR lead definition 'origin'
DR_LEAD_MONTH = 6                               # 2027-05: DR's L ('window A mid-month 2027-05')
PAGE_RULE = ("no page line in any branch: ledger / README / MODEL_CARD only; a page line needs a "
             "new user ruling (R8.8 named neither CB nor C14a and kept the FV v2 per-truth "
             "split off the page)")
P_OVER_MIN = 0.8                                # page family: P(over) >= 0.8
P_OVER_MAX = 0.2                                # Changhua recipes: P(over) <= 0.2
E1_N_SE = 2.0
E0_TOL = 1e-12
STREAMS = {"noise": 11, "year": 12, "month": 13, "drift": 14}
SMOKE_SEED = 61130010
REFUSED_SEEDS = {
    O.SEED: "FV v1 / FV v2's seed: its noise draws are the published worlds, not fresh",
    U.SEED: "U1.HB.power.v1's registered seed",
    U.SELF_CHECK_SEED: "U1.HB.power.v1's self-check seed (also E02c's bootstrap seed)",
    20261105: "DR.relative_drift.v1's ICC_BOOT_SEED (e02dlib)",
    20261102: "DR / E02d's drift BOOT_SEED (e02dlib)",
    20261103: "FV.stat.v1's declared seed, refused by FV.stat.v2 (also power_e02d's SEED)",
    SMOKE_SEED: "CB's own engineering smoke seed",
}
PASS, OVER, UNDER, POOLED, NOT_EVAL = 0, 1, 2, 3, 4
OUTCOMES = ("pass", "over", "under", "pooled_only", "not_evaluable")
E02C_SIDE = {"above": OVER, "below": UNDER}

# ----------------------------------------------------------------------------- inputs (real)
E02_PRED = f"{S.E02_DIR}/predictions.csv"
E02_PROTOCOL = f"{S.E02_DIR}/protocol.json"
WELLS_CSV = f"{O.E02C_BASE_DIR}/wells_158.csv"
NOISE_CSV = f"{O.INPUTS_DIR}/noise_ar1.csv"
MASK_CSV = f"{O.INPUTS_DIR}/obs_mask.csv"
ZR0_NPZ = O.ZR0_NPZ
FV2_DECL = V2.DECL
FV2_REPORT = V2.REPORT_WORLDS
FV2_ATTEMPTS = V2.ATTEMPTS
FV_EXT_REPORT = "results/dataprep/reports/FV.ext.v1/report.json"
DR_ID = "DR.relative_drift.v1"
DR_DECL = f"results/dataprep/preregistration/{DR_ID}.json"
DR_REPORT = f"results/dataprep/reports/{DR_ID}/report.json"
DR_WELLS = f"results/dataprep/reports/{DR_ID}/wells_e02d.csv"
U1HB_ID = U.DECL_ID
U1HB_DECL = f"results/dataprep/preregistration/{U1HB_ID}.json"
REPORT_DIR = f"results/dataprep/reports/{ID}"
REPORT = f"{REPORT_DIR}/report.json"
ATTEMPTS = f"{REPORT_DIR}/attempts.jsonl"
DECL = f"results/dataprep/preregistration/{ID}.json"
PINS = {   # every real input, by sha256 (registered values; the runner refuses any change)
    "pred": S.E02_PRED_SHA,
    "protocol": S.E02_PROTOCOL_SHA,
    "wells": O.E02C_BASE_SHA["wells_158.csv"],
    "noise": "73965a08914149af2132c96f617268701277b49411cf17cfbf57d41fbeaa9d55",
    "mask": "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89",
    "zr0": V2.V1_FORWARD_SHA["ZR0"],
    "fv2_report": "320ac928059ca78d343b5c3a1b8b32054a0c1ce15d5633b1e5060dfa7fbf9dae",
    "dr_report": "32025941e6b2c83cb606be843fd2a0ccb289ca966d61493f2cec87f9916c882b",
    "dr_wells": "f3174c481827f22efb4a6c7436359e765414bff07fa9d8bd06a65a73c51a92b8",
}
FV2_DECL_SHA = "f7e35a17482cce307c6be84013f9a2919f16f4be66e9a01cd7eeacbd08fb49d4"
DR_DECL_SHA = "f835af572a8b6e49bec6ae38d6edf7c37bb9d8e26abedb658b71e6e9c5fe873b"
U1HB_DECL_SHA = "0edd66748249ce13bfe90c5845f7c97694ca26e887248ca805d20bf6d4f46d5d"
# the published v2 value the E1 gate is anchored to (report_worlds.json, sha above)
V2_Q0_CHANGHUA = 0.004
# FV v2 frozen code CB imports: must still be byte-identical to FV v2's registration
FV2_SHARED_CODE = (V2.SELF, O.SELF, f"{S.E}/score_e02c.py",
                   "dataprep_staging/harness/coverage3.py", "hydrophysics/twin/bands.py")


# ----------------------------------------------------------------------------- small helpers
def check_seed(seed) -> int:
    """A seed is a non-negative int chosen by the lead; there is never a default."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise EngineeringFailure(f"seed must be a non-negative int, got {seed!r}")
    if int(seed) in REFUSED_SEEDS:
        raise EngineeringFailure(f"seed {seed} refused: {REFUSED_SEEDS[int(seed)]}")
    return int(seed)


def rng_for(seed: int, stream: str, ti: int, k: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAMS[stream], ti, k]))


def icc_cells() -> list[dict]:
    """U1.HB.power.v1's (ICC_month outer, ICC_year inner) order; cell 0 is (0, 0)."""
    out = []
    for am in ICC_MONTH:
        for ay in ICC_YEAR:
            out.append({"cell": len(out), "icc_month": am, "icc_year": ay})
    return out


CELLS = icc_cells()


def window_dt() -> np.ndarray:
    """Years from the origin (2023.0) to each window month's middle (DR's 'origin' lead)."""
    m = S.window_months(*S.WINDOW)
    return np.array([d.year + (d.month - 0.5) / 12.0 - ORIGIN_YEAR for d in m])


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=O._jsonable) + "\n")


# ----------------------------------------------------------------------------- paths / context
@dataclass
class Paths:
    pred: str = E02_PRED
    protocol: str = E02_PROTOCOL
    wells: str = WELLS_CSV
    noise: str = NOISE_CSV
    mask: str = MASK_CSV
    zr0: str = ZR0_NPZ
    batches: dict = field(default_factory=lambda: {
        t: [V2.batch_npz(t, b) for b in range(V2.PLAN.n_batch)] for t in TAGS})
    fv2_attempts: str = FV2_ATTEMPTS
    fv2_report: str = FV2_REPORT
    dr_report: str = DR_REPORT
    dr_wells: str = DR_WELLS
    report: str = REPORT
    attempts: str = ATTEMPTS


def verify_pins(P: Paths, pins: dict = PINS) -> dict:
    """sha256 of every pinned input (EngineeringFailure on a mismatch or a missing file)."""
    out = {}
    for key, want in pins.items():
        p = O.rp(getattr(P, key))
        if not p.is_file():
            raise EngineeringFailure(f"{key}: {p} missing")
        got = O.sha256(p)
        if got != want:
            raise EngineeringFailure(f"{key}: {p} sha256 {got[:12]} != pinned {want[:12]}")
        out[key] = {"path": str(getattr(P, key)), "sha256": got}
    return out


@dataclass
class Ctx:
    ids: list
    county: np.ndarray
    cells: np.ndarray
    layers: np.ndarray
    zr: np.ndarray
    frame0: pd.DataFrame
    protocol: dict
    sd: np.ndarray
    phi: np.ndarray
    mask: np.ndarray
    group: np.ndarray
    tau2: np.ndarray
    dt: dict


def dr_inputs(report_path, wells_path, ids: list, county: np.ndarray) -> dict:
    """DR's bound drift: per-well tau^2 (its county's), the zone group of each well and the
    two lead definitions. Only station_id, county, grouping and group of wells_e02d.csv are
    read (no datum, sd or other column)."""
    rep = json.loads(O.rp(report_path).read_text(encoding="utf-8"))
    w = pd.read_csv(O.rp(wells_path), dtype={"station_id": str},
                    usecols=["station_id", "county", "grouping", "group"]).set_index("station_id")
    if sorted(w.index) != sorted(ids):
        raise EngineeringFailure("wells_e02d.csv does not hold exactly the prediction wells")
    w = w.loc[ids]
    if list(w["county"]) != list(county):
        raise EngineeringFailure("wells_e02d.csv county differs from the station code")
    tau2 = np.full(len(ids), np.nan)
    shift = {}
    for c in np.unique(county):
        dec = (rep.get("decisions") or {}).get(c) or {}
        if dec.get("grouping") != "zone" or dec.get("include_drift") is not True:
            raise EngineeringFailure(f"DR {c}: not zone grouping with the drift term included")
        if (w.loc[w["county"] == c, "grouping"] != "zone").any():
            raise EngineeringFailure(f"DR {c}: a well's grouping is not 'zone'")
        tau2[county == c] = float(dec["tau2_per_year2"])
        cr = rep["counties"][c]
        if cr.get("lead_definition") != "origin":
            raise EngineeringFailure(f"DR {c}: lead definition is not 'origin'")
        if abs(float(cr["lead_years"]) - float(window_dt()[DR_LEAD_MONTH])) > 1e-9:
            raise EngineeringFailure(f"DR {c}: lead_years {cr['lead_years']} is not the "
                                     "origin-to-2027-05 lead CB's dt uses")
        shift[c] = float(cr["lead_years_datum_centre_report_only"]) - float(cr["lead_years"])
    if not np.isfinite(tau2).all() or (tau2 < 0).any():
        raise EngineeringFailure("DR tau^2 missing or negative for a well")
    groups = w["group"].astype(str).to_numpy()
    _, gidx = np.unique(groups, return_inverse=True)
    if (np.bincount(gidx) < 2).any():
        raise EngineeringFailure("a DR group has fewer than 2 wells")
    base = window_dt()
    dc = np.stack([base + shift[c] for c in county])
    return {"tau2": tau2, "group": gidx,
            "dt": {"none": None, "origin": np.broadcast_to(base, (len(ids), len(base))),
                   "datum_centre_report_only": dc},
            "tau2_by_county": {c: float(tau2[county == c][0]) for c in np.unique(county)},
            "datum_centre_shift_years": shift,
            "groups": {str(g): int(n) for g, n in zip(*np.unique(groups, return_counts=True),
                                                      strict=True)}}


def load_ctx(P: Paths) -> tuple[Ctx, dict]:
    """E02's window rows (osse_fv.load_context's construction, without its CL / CLT baselines
    and pumping pool, which CB never uses), noise, mask, protocol and DR's drift inputs."""
    pred = pd.read_csv(O.rp(P.pred), dtype={"station_id": str})
    pred["date"] = S._month(pred["date"])
    win = S.window_months(*S.WINDOW)
    pred = pred[pred["date"].isin(win)].sort_values(["station_id", "date"])
    ids = sorted(pred["station_id"].unique())
    W = len(ids)
    if len(pred) != W * O.N_WIN:
        raise EngineeringFailure(f"predictions: {len(pred)} window rows, expected {W} x 12")
    wells = pd.read_csv(O.rp(P.wells), dtype={"station_id": str}).set_index("station_id")
    if sorted(wells.index) != ids:
        raise EngineeringFailure("wells_158.csv does not hold exactly the prediction wells")
    wells = wells.loc[ids]
    if (wells["cell"] < 0).any() or (wells["layer"] < 0).any():
        raise EngineeringFailure("a well has no forward cell or layer")
    noise = pd.read_csv(O.rp(P.noise), dtype={"station_id": str}).set_index("station_id")
    if sorted(noise.index) != ids:
        raise EngineeringFailure("noise_ar1.csv does not hold exactly the prediction wells")
    noise = noise.loc[ids]
    mask = O.read_mask(O.rp(P.mask), ids)
    zr = pred.pivot(index="station_id", columns="date")["predicted_m"].reindex(
        index=ids, columns=win).to_numpy(float)
    for c in ("predicted_m", "lower_m", "upper_m"):
        if not np.isfinite(pred[c].to_numpy(float)).all():
            raise EngineeringFailure(f"non-finite {c} in the window")
    frame0 = pred[["station_id", "date", "prediction_origin", "predicted_m", "baseline_m",
                   "lower_m", "upper_m", "model_sha256"]].reset_index(drop=True)
    county = np.array([S.county_of(s) for s in ids])
    dr = dr_inputs(P.dr_report, P.dr_wells, ids, county)
    ctx = Ctx(ids=ids, county=county, cells=wells["cell"].to_numpy(int),
              layers=wells["layer"].to_numpy(int), zr=zr, frame0=frame0,
              protocol=json.loads(O.rp(P.protocol).read_text(encoding="utf-8")),
              sd=noise["sd_m"].to_numpy(float), phi=noise["phi"].to_numpy(float), mask=mask,
              group=dr["group"], tau2=dr["tau2"], dt=dr["dt"])
    info = {k: dr[k] for k in ("tau2_by_county", "datum_centre_shift_years", "groups")}
    info.update({"wells": W, "by_county": {c: int((county == c).sum())
                                          for c in np.unique(county)}})
    return ctx, info


# ----------------------------------------------------------------------------- truth worlds
def fv2_batches_from_attempts(att, plan: V2.Plan = V2.PLAN) -> dict:
    """{'<truth>_b<b>': npz sha256} of each batch's last successful forward (metadata only;
    the sealed forward stdout is never opened)."""
    p = O.rp(att)
    if not p.is_file():
        return {}
    recs = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
    out = {}
    for t in TAGS:
        for b in range(plan.n_batch):
            fw = [r for r in recs if r.get("event") == "forward" and r.get("truth") == t
                  and str(r.get("batch")) == str(b)]
            if fw and int(fw[-1].get("rc", 1)) == 0:
                out[f"{t}_b{b}"] = fw[-1].get("npz_sha256")
    return out


def seal_lapsed(att) -> dict:
    """FV v2's seal ('batch npz values are read only by the scorers, this one and FV.ext.v1')
    has lapsed once both scorers ran: attempts rows score_worlds and score_ext with rc 0."""
    p = O.rp(att)
    recs = ([json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
            if p.is_file() else [])
    ok = {e: any(r.get("event") == e and int(r.get("rc", 1)) == 0 for r in recs)
          for e in ("score_worlds", "score_ext")}
    return {"scored": ok, "lapsed": all(ok.values())}


def load_truths(P: Paths, ctx: Ctx, plan: V2.Plan = V2.PLAN, verify: dict | None = None
                ) -> tuple[dict, dict]:
    """{tag: {'i', 'D' (W, 12), 'R' (K, W, 12)}} exactly as osse_fv2.score_worlds builds them
    (D from batch 0's baseline and ZR0; R = world minus its own batch baseline). ``verify``:
    {'<truth>_b<b>': sha256} every batch npz must match before it is opened."""
    used: dict = {}
    zmeta, zh = O.well_heads(O.rp(P.zr0), ctx.cells, ctx.layers)
    if [str(x) for x in zmeta["scenario_names"]] != ["baseline"]:
        raise EngineeringFailure("ZR0 is not a baseline-only forward")
    h_zr0 = zh[0]
    del zh
    out = {}
    for tag in TAGS:
        paths = P.batches[tag]
        if len(paths) != plan.n_batch:
            raise EngineeringFailure(f"truth {tag}: {len(paths)} batch files, expected "
                                     f"{plan.n_batch}")
        R = np.full((plan.k_truth, len(ctx.ids), O.N_WIN), np.nan)
        base0 = None
        for b, path in enumerate(paths):
            key = f"{tag}_b{b}"
            if verify is not None:
                got = O.sha256(O.rp(path))
                if verify.get(key) != got:
                    raise EngineeringFailure(f"{key}: sha256 {got[:12]} is not the registered "
                                             "FV v2 batch")
                used[key] = got
            meta, h = O.well_heads(O.rp(path), ctx.cells, ctx.layers)
            if [str(x) for x in meta["scenario_names"]] != plan.names(b):
                raise EngineeringFailure(f"{key}: scenario order differs from FV v2's plan")
            if h.shape[-1] < O.T_RUN or not np.isfinite(h[..., :O.T_RUN]).all():
                raise EngineeringFailure(f"{key}: fewer than {O.T_RUN} finite months")
            if base0 is None:
                base0 = h[0][:, :O.T_RUN].copy()
            elif float(np.max(np.abs(h[0][:, :O.T_FIT] - base0[:, :O.T_FIT]))) > 0.0:
                raise EngineeringFailure(f"{key}: baseline hindcast differs from batch 0")
            for j, k in enumerate(plan.worlds(b)):
                R[k] = h[1 + j][:, O.WIN[0]:O.WIN[1]] - h[0][:, O.WIN[0]:O.WIN[1]]
            del h
        if not np.isfinite(R).all():
            raise EngineeringFailure(f"truth {tag}: a world response is missing")
        out[tag] = {"i": TAGS.index(tag), "R": R,
                    "D": O.departure(base0, h_zr0, ctx.mask)}
    return out, used


# ----------------------------------------------------------------------------- world parts
def centre_rescale(z: np.ndarray, gidx: np.ndarray) -> np.ndarray:
    """z centred within each group and rescaled by sqrt(n / (n - 1)) (marginal variance kept,
    zero sum per group: a group-relative drift, as DR's relative residual)."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    for g in np.unique(gidx):
        s = gidx == g
        n = int(s.sum())
        out[s] = (z[s] - z[s].mean()) * math.sqrt(n / (n - 1.0))
    return out


def drift_field(ctx: Ctx, z: np.ndarray, arm: str) -> np.ndarray | None:
    """(W, 12) drift of arm ``arm`` (None for 'none')."""
    t = ctx.dt[arm]
    if t is None:
        return None
    b = centre_rescale(z, ctx.group) * np.sqrt(ctx.tau2)
    return b[:, None] * t


def shock_noise(e: np.ndarray, C: float, B: np.ndarray, sd: np.ndarray, ay: float,
                am: float) -> np.ndarray:
    """U1.HB.power.v1's year / month decomposition at the noise's marginal sd; (0, 0) -> e."""
    sdc = np.asarray(sd, float)[:, None]
    return (math.sqrt(ay) * sdc * C
            + math.sqrt(1.0 - ay) * (math.sqrt(am) * sdc * np.asarray(B, float)[None, :]
                                     + math.sqrt(1.0 - am) * e))


def draws(ctx: Ctx, seed: int, ti: int, k: int) -> dict:
    return {"e": O.simulate_ar1(ctx.sd, ctx.phi, O.N_WIN, rng_for(seed, "noise", ti, k)),
            "C": float(rng_for(seed, "year", ti, k).standard_normal()),
            "B": rng_for(seed, "month", ti, k).standard_normal(O.N_WIN),
            "z": rng_for(seed, "drift", ti, k).standard_normal(len(ctx.ids))}


def world_obs(ctx: Ctx, D: np.ndarray, r: np.ndarray, noise: np.ndarray,
              drift: np.ndarray | None) -> np.ndarray:
    obs = ctx.zr + D + r + noise                 # FV v2's association order
    return obs if drift is None else obs + drift


# ----------------------------------------------------------------------------- E02c outcome
def frozen_side(oc: dict, cc: dict) -> str:
    """E02c's frozen side of a 'coverage_not_supported' outcome, read back from its wording
    and verified against score_e02c.WORDING (neutral power flags: no suffix)."""
    g1 = cc["G1"]
    hits = [s for s in ("below", "above")
            if oc["wording"] == S.WORDING["cov_not_supported"].format(
                county=cc["county"], cov=g1["coverage"], n=g1["n"], side=s)]
    if len(hits) != 1:
        raise EngineeringFailure("E02c's coverage wording does not carry exactly one side")
    return hits[0]


def classify(oc: dict, cc: dict) -> int:
    v = oc["verdict"]
    if v == "coverage_supported":
        return PASS
    if v == "coverage_not_supported":
        return E02C_SIDE[frozen_side(oc, cc)]
    if v == "pooled_gate_failed_county_report_only":
        return POOLED
    return NOT_EVAL


def outcome(ctx: Ctx, obs: np.ndarray, counties=(COUNTY,)) -> dict:
    """{county: (outcome code, county coverage)} of one world, by E02c's frozen code."""
    f = ctx.frame0.copy()
    f["observed_m"] = np.asarray(obs, float).reshape(-1)
    f = S.annotate(f)
    g0 = O.g0_e02(f, ctx.protocol)["pass"]
    out = {}
    for c in counties:
        cc = S.county_coverage(f, c)
        oc = S.cov_outcome(g0, cc, O.NEUTRAL_POWER)
        cov = (cc.get("G1") or {}).get("coverage")
        out[c] = (classify(oc, cc), float(cov) if cov is not None else float("nan"))
    return out


# ----------------------------------------------------------------------------- tasks
_STATE: dict = {}


def _task_e0(args) -> tuple:
    """FV v2's own world (its seed, its noise stream): (ti, k, {county: code})."""
    ti, k = args
    ctx, t = _STATE["ctx"], _STATE["truths"][TAGS[ti]]
    e = O.simulate_ar1(ctx.sd, ctx.phi, O.N_WIN, O.rng_for(O.STREAM_NOISE, t["i"], k))
    oc = outcome(ctx, world_obs(ctx, t["D"], t["R"][k], e, None), counties=S.COUNTIES_COV)
    return ti, k, {c: v[0] for c, v in oc.items()}


def _task_e1(args) -> tuple:
    """Fresh seed, cell (0, 0), no drift: (ti, k, code)."""
    ti, k = args
    ctx, t, seed = _STATE["ctx"], _STATE["truths"][TAGS[ti]], _STATE["seed"]
    d = draws(ctx, seed, ti, k)
    n = shock_noise(d["e"], d["C"], d["B"], ctx.sd, 0.0, 0.0)
    return ti, k, outcome(ctx, world_obs(ctx, t["D"], t["R"][k], n, None))[COUNTY][0]


def _task_grid(args) -> tuple:
    """Every drift arm x ICC cell of world k of truth ti: (ti, k, codes (A, C), cov (A, C))."""
    ti, k = args
    ctx, t, seed = _STATE["ctx"], _STATE["truths"][TAGS[ti]], _STATE["seed"]
    d = draws(ctx, seed, ti, k)
    codes = np.full((len(DRIFT_ARMS), len(CELLS)), NOT_EVAL, np.int8)
    cov = np.full((len(DRIFT_ARMS), len(CELLS)), np.nan)
    noises = [shock_noise(d["e"], d["C"], d["B"], ctx.sd, c["icc_year"], c["icc_month"])
              for c in CELLS]
    for a, arm in enumerate(DRIFT_ARMS):
        drift = drift_field(ctx, d["z"], arm)
        for j, n in enumerate(noises):
            code, cv = outcome(ctx, world_obs(ctx, t["D"], t["R"][k], n, drift))[COUNTY]
            codes[a, j], cov[a, j] = code, cv
    return ti, k, codes, cov


def _pmap(fn, tasks: list, workers: int) -> list:
    if workers <= 1:
        return [fn(x) for x in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (8 * workers)))


# ----------------------------------------------------------------------------- statistics
def _rate(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    p = float(x.mean())
    return {"p": p, "mc_se": float(math.sqrt(p * (1 - p) / len(x))), "k": int(len(x))}


def e1_check(x_new: int, x_pub: int, k_new: int, k_pub: int, n_se: float = E1_N_SE) -> dict:
    """|p_new - p_pub| <= n_se x SE of the difference at the pooled rate (equality if 0)."""
    p_new, p_pub = x_new / k_new, x_pub / k_pub
    pbar = (x_new + x_pub) / (k_new + k_pub)
    se = math.sqrt(pbar * (1 - pbar) * (1 / k_new + 1 / k_pub))
    diff = abs(p_new - p_pub)
    ok = diff <= n_se * se + 1e-15 if se > 0 else diff == 0
    return {"p_new": p_new, "p_published": p_pub, "abs_diff": diff, "se_diff": se,
            "n_se": n_se, "diff_in_se": (diff / se if se > 0 else None), "pass": bool(ok),
            "rule": "SE of the difference of two binomial rates at the pooled rate"}


def summarize(codes: dict, cov: dict) -> dict:
    """codes / cov: {tag: (K, A, C)}. Rates, conditional side, paired shock and drift
    effects, coverage quantiles."""
    out: dict = {}
    for tag, x in codes.items():
        rec: dict = {}
        for a, arm in enumerate(DRIFT_ARMS):
            cells = []
            for j, c in enumerate(CELLS):
                col = x[:, a, j]
                rates = {o: _rate(col == i) for i, o in enumerate(OUTCOMES)}
                fails = col != PASS
                base = x[:, a, 0]
                cv = cov[tag][:, a, j]
                cells.append({
                    **c, "rates": rates,
                    "report_only_side_given_non_pass": (
                        {"n_non_pass": int(fails.sum()),
                         "over": float((col[fails] == OVER).mean()),
                         "under": float((col[fails] == UNDER).mean())}
                        if fails.any() else {"n_non_pass": 0}),
                    "bias_cancelled_by_shock": _rate((base != PASS) & (col == PASS)),
                    "over_lost_to_shock": _rate((base == OVER) & (col != OVER)),
                    "side_flipped_by_shock": _rate(((base == OVER) & (col == UNDER))
                                                   | ((base == UNDER) & (col == OVER))),
                    "coverage_q05_q50_q95": [float(q) for q in np.nanquantile(
                        cv, [0.05, 0.5, 0.95])] if np.isfinite(cv).any() else None})
            rec[arm] = cells
        a0, a1 = DRIFT_ARMS.index("none"), DRIFT_ARMS.index("origin")
        rec["bias_cancelled_by_drift_report_only"] = [
            {**c, **_rate((x[:, a0, j] != PASS) & (x[:, a1, j] == PASS))}
            for j, c in enumerate(CELLS)]
        out[tag] = rec
    return out


def gate(stats: dict) -> dict:
    """The corrected one-sided gate on point estimates, with margins in MC SE units."""
    def over(tag, arm, j):
        return stats[tag][arm][j]["rates"]["over"]

    a = [{"truth": t, "arm": arm, "cell": j, **over(t, arm, j)}
         for t in PAGE_FAMILY for arm in GATE_A_ARMS for j in range(len(CELLS))]
    b = [{"truth": t, "arm": arm, "cell": j, **over(t, arm, j)}
         for t in ALT_FAMILY for arm in GATE_B_ARMS for j in range(len(CELLS))]
    worst_a = min(a, key=lambda r: r["p"])
    worst_b = max(b, key=lambda r: r["p"])

    def margin(r, thr, sign):
        return sign * (r["p"] - thr) / r["mc_se"] if r["mc_se"] > 0 else None

    ga, gb = worst_a["p"] >= P_OVER_MIN, worst_b["p"] <= P_OVER_MAX
    return {"rule": f"min P(over | {'/'.join(PAGE_FAMILY)}, every ICC cell, drift "
                    f"{'/'.join(GATE_A_ARMS)}) >= {P_OVER_MIN} AND max P(over | "
                    f"{'/'.join(ALT_FAMILY)}, every ICC cell, drift {'/'.join(GATE_B_ARMS)}) "
                    f"<= {P_OVER_MAX}; point estimates",
            "page_family": {"pass": bool(ga), "worst": worst_a,
                            "margin_in_se": margin(worst_a, P_OVER_MIN, 1)},
            "changhua_recipes": {"pass": bool(gb), "worst": worst_b,
                                 "margin_in_se": margin(worst_b, P_OVER_MAX, -1)},
            "pass": bool(ga and gb)}


# ----------------------------------------------------------------------------- wording
# ledger / README / MODEL_CARD only, in every branch (PAGE_RULE); rates printed to 3 decimals
WORDING = {
    "side_separates": {
        "ledger": {
            "en": "CB.coverage_side.v1 (model-only; FV v2's 1000 worlds with DR's drift and "
                  "U1.HB.power's year/month shocks overlaid, fresh seed; ICC_well not "
                  "overlaid): a Changhua E02c coverage fail by over-coverage occurs in >= "
                  "{a:.3f} of page-family worlds (Q0, R0; drift included; every year/month ICC "
                  "cell) and <= {b:.3f} of Changhua-recipe worlds (R1c, X5R1; every year/month "
                  "ICC cell, with and without drift); bias-cancelled-by-shock probability up "
                  "to {c:.3f} (page family). C14a Changhua only, one-sided (only an "
                  "over-coverage fail is read); no C20/C21 mapping; not on the page.",
            "zh": "CB.coverage_side.v1（僅模型；FV v2 的 1000 個情境疊加 DR 漂移與 U1.HB.power "
                  "年／月衝擊，新種子；未疊加 ICC_well）：頁面參數族情境（Q0、R0；含漂移；每個"
                  "年／月 ICC 格）中至少 {a:.3f} 的 E02c 彰化涵蓋率未通過屬涵蓋過度，彰化參數設定"
                  "情境（R1c、X5R1；每個年／月 ICC 格，含與不含漂移）中至多 {b:.3f}；衝擊抵消偏差"
                  "的機率最高 {c:.3f}（頁面參數族）。僅適用於 C14a 彰化，單向解讀（僅解讀涵蓋過度"
                  "之未通過）；不對應 C20/C21；不上頁面。"}},
    "side_does_not_separate": {
        "ledger": {
            "en": "CB.coverage_side.v1 (model-only; ICC_well not overlaid): once drift and "
                  "year/month shocks are overlaid, the side of a Changhua E02c coverage fail "
                  "does not separate the families (min P(over | Q0/R0, drift) {a:.3f} vs >= "
                  "0.8 needed; max P(over | R1c/X5R1) {b:.3f} vs <= 0.2 needed; "
                  "bias-cancelled-by-shock up to {c:.3f}). No side reading; the FV v2 "
                  "per-truth disclosure stands; not on the page.",
            "zh": "CB.coverage_side.v1（僅模型；未疊加 ICC_well）：疊加漂移與年／月衝擊後，E02c "
                  "彰化涵蓋率未通過的方向無法區分兩種參數族（Q0／R0 含漂移之最小涵蓋過度機率 "
                  "{a:.3f}，需 ≥ 0.8；R1c／X5R1 之最大涵蓋過度機率 {b:.3f}，需 ≤ 0.2；衝擊抵消"
                  "偏差的機率最高 {c:.3f}）。不提供方向解讀；FV v2 各真值揭露維持不變；不上頁面。"}},
    "not_evaluable": {
        "ledger": {
            "en": "CB.coverage_side.v1 (model-only) not evaluable ({why}); no side reading, "
                  "no further attempt; the FV v2 per-truth disclosure stands; not on the page.",
            "zh": "CB.coverage_side.v1（僅模型）無法評估（{why}）；不提供方向解讀，亦不再嘗試；"
                  "FV v2 各真值揭露維持不變；不上頁面。"}},
}
# appended to either scored ledger line when the report-only datum-centre arm misses 0.8
DATUM_CENTRE_SUFFIX = {
    "en": " With DR's report-only datum-centre lead (+5.5 years), min P(over | Q0/R0) is "
          "{d:.3f}, below 0.8.",
    "zh": "若採 DR 僅供報告之資料基準中心前置期（+5.5 年），Q0／R0 之最小涵蓋過度機率為 "
          "{d:.3f}，低於 0.8。"}


def bias_cancelled_summary(stats: dict) -> dict:
    """Max over ICC cells of P(fail at (0, 0) and pass at the cell), per truth and arm."""
    out = {}
    for tag in TAGS:
        out[tag] = {}
        for arm in DRIFT_ARMS:
            best = max(stats[tag][arm], key=lambda c: c["bias_cancelled_by_shock"]["p"])
            out[tag][arm] = {"max_p": best["bias_cancelled_by_shock"]["p"],
                             "cell": best["cell"], "icc_year": best["icc_year"],
                             "icc_month": best["icc_month"]}
    pf = max(out[t][a]["max_p"] for t in PAGE_FAMILY for a in GATE_A_ARMS)
    return {"definition": "P(the world's Changhua outcome is not a pass at ICC (0, 0) and is "
                          "a pass at the shocked cell, same world, same drift arm): the shock "
                          "cancels the structural coverage bias",
            "max_over_cells": out, "page_family_gated_arm_max": pf}


def datum_centre_page_min(stats: dict) -> dict:
    """Report-only: gate A's statistic on the datum-centre arm (min P(over), page family)."""
    arm = "datum_centre_report_only"
    r = min(({"truth": t, "cell": j, **stats[t][arm][j]["rates"]["over"]}
             for t in PAGE_FAMILY for j in range(len(CELLS))), key=lambda x: x["p"])
    return {**r, "misses_0_8": bool(r["p"] < P_OVER_MIN), "report_only": True}


def reading(g: dict, stats: dict) -> dict:
    a = g["page_family"]["worst"]["p"]
    b = g["changhua_recipes"]["worst"]["p"]
    c = bias_cancelled_summary(stats)["page_family_gated_arm_max"]
    dc = datum_centre_page_min(stats)
    key = "side_separates" if g["pass"] else "side_does_not_separate"
    w = {part: {lang: s.format(a=a, b=b, c=c)
                + (DATUM_CENTRE_SUFFIX[lang].format(d=dc["p"]) if dc["misses_0_8"] else "")
                for lang, s in d.items()}
         for part, d in WORDING[key].items()}
    return {"reading": key, "applies_to": "C14a Changhua only (no C20/C21 mapping)",
            "one_sided": "only a 2028 Changhua over-coverage fail is read", "wording": w,
            "datum_centre_gate_a_report_only": dc, "page": PAGE_RULE,
            "status_change": "none (C14a stays as E02c decides it)"}


def not_evaluable(rep: dict, why: str) -> dict:
    rep["status"] = "not_evaluable"
    rep["why"] = why
    rep["reading"] = {"reading": "not_evaluable", "applies_to": "C14a Changhua only",
                      "wording": {p: {lang: s.format(why=why) for lang, s in d.items()}
                                  for p, d in WORDING["not_evaluable"].items()},
                      "page": PAGE_RULE, "status_change": "none"}
    return rep


# ----------------------------------------------------------------------------- study
def published_rates(path) -> dict:
    """{county: {tag: p}} of FV v2's report_worlds.json P.coverage.<county>.by_truth."""
    rep = json.loads(O.rp(path).read_text(encoding="utf-8"))
    return {c: {t: float(rep["P"]["coverage"][c]["by_truth"][t]) for t in TAGS}
            for c in S.COUNTIES_COV}


def run_study(ctx: Ctx, truths: dict, *, seed: int, published: dict, k: int = K,
              k_published: int = K, workers: int = 1, log=print) -> dict:
    """E0, E1, then the grid. Returns the report (not written). E0 raises EngineeringFailure
    (nothing written); an E1 failure, and any failure after E1 passed, returns a final
    not_evaluable report."""
    seed = int(seed)
    t0 = time.time()
    rep: dict = {"id": ID, "version": VERSION, "seed": seed, "k": k, "streams": STREAMS,
                 "cells": CELLS, "drift_arms": list(DRIFT_ARMS), "truths": list(TAGS),
                 "county": COUNTY, "outcomes": list(OUTCOMES)}
    _STATE.clear()
    _STATE.update({"ctx": ctx, "truths": truths, "seed": seed})
    try:
        # ---- E0: FV v2's own worlds reproduce its published rates exactly
        res = _pmap(_task_e0, [(ti, kk) for ti in range(len(TAGS)) for kk in range(k)],
                    workers)
        e0 = {c: {} for c in S.COUNTIES_COV}
        for c in S.COUNTIES_COV:
            for ti, tag in enumerate(TAGS):
                got = float(np.mean([r[2][c] == PASS for r in res if r[0] == ti]))
                want = published[c][tag]
                e0[c][tag] = {"reproduced": got, "published": want,
                              "pass": abs(got - want) <= E0_TOL}
        rep["engineering"] = {"E0": e0}
        bad = [f"{c} {t}" for c in e0 for t, v in e0[c].items() if not v["pass"]]
        if bad:
            raise EngineeringFailure(f"E0: FV v2's published coverage rates not reproduced: "
                                     f"{bad}")
        log("gate E0 passed (FV v2 reproduced exactly)")

        # ---- E1: fresh seed, Q0 Changhua, (0, 0), no drift
        qi = TAGS.index("Q0")
        r1 = _pmap(_task_e1, [(qi, kk) for kk in range(k)], workers)
        e1_codes = np.array([r[2] for r in sorted(r1, key=lambda r: r[1])])
        x_pub = int(round(published[COUNTY]["Q0"] * k_published))
        e1 = e1_check(int((e1_codes == PASS).sum()), x_pub, k, k_published, n_se=E1_N_SE)
        rep["engineering"]["E1"] = e1
        if not e1["pass"]:
            log("gate E1 FAILED")
            return not_evaluable(rep, "fresh-seed Q0 Changhua pass rate not within 2 MC SE "
                                      "of FV v2's published value")
        log(f"gate E1 passed ({e1['p_new']:.3f} vs {e1['p_published']:.3f})")

        # ---- the grid: from here on any failure is a FINAL not_evaluable (fresh draws made)
        try:
            rg = _pmap(_task_grid, [(ti, kk) for ti in range(len(TAGS)) for kk in range(k)],
                       workers)
            return _finish(rep, rg, e1_codes, k, t0, log)
        except Exception as err:      # noqa: BLE001 - declared: final not_evaluable
            log(f"failure after E1: {err}")
            return not_evaluable(rep, f"failure after E1 ({type(err).__name__}: {err})")
    finally:
        _STATE.clear()


def _finish(rep: dict, rg: list, e1_codes: np.ndarray, k: int, t0: float, log) -> dict:
    codes = {t: np.full((k, len(DRIFT_ARMS), len(CELLS)), NOT_EVAL, np.int8) for t in TAGS}
    cov = {t: np.full((k, len(DRIFT_ARMS), len(CELLS)), np.nan) for t in TAGS}
    for ti, kk, cd, cv in rg:
        codes[TAGS[ti]][kk], cov[TAGS[ti]][kk] = cd, cv
    a0 = DRIFT_ARMS.index("none")
    if not np.array_equal(codes["Q0"][:, a0, 0], e1_codes):
        raise EngineeringFailure("the grid's Q0 (0, 0) no-drift outcomes differ from E1's")
    stats = summarize(codes, cov)
    g = gate(stats)
    rep.update({"stats": stats, "gate": g, "bias_cancelled_by_shock": bias_cancelled_summary(
        stats), "reading": reading(g, stats), "status": "scored",
        "codes_report_only": {"layout": "[world][drift arm][ICC cell], codes index 'outcomes'",
                              **{t: codes[t].tolist() for t in TAGS}},
        "runtime_s": round(time.time() - t0, 1)})
    log(f"gate {'PASSED' if g['pass'] else 'not passed'}")
    return rep


# ----------------------------------------------------------------------------- registration
def load_registered(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """The registered declaration with CB's frozen code intact and inputs unchanged."""
    return V2.load_registered(prereg_id=ID, frozen=FROZEN, prereg_dir=prereg_dir,
                              rows_dir=rows_dir)


def check_registered(doc: dict, seed: int, P: Paths) -> dict:
    """Everything the run binds, against the registration and FV v2's own records."""
    design = doc.get("design") or {}
    if design.get("seed") != seed:
        raise EngineeringFailure(f"seed {seed} is not the registered seed")
    fv2 = json.loads(O.rp(FV2_DECL).read_text(encoding="utf-8"))
    if O.sha256(O.rp(FV2_DECL)) != FV2_DECL_SHA:
        raise EngineeringFailure("FV v2's registration is not the bound one")
    drift = [p for p in FV2_SHARED_CODE
             if (fv2.get("frozen_code") or {}).get(p) != O.sha256(O.rp(p))]
    if drift:
        raise EngineeringFailure(f"code differs from FV v2's registered frozen code: {drift}")
    for name, path, want in (("DR", DR_DECL, DR_DECL_SHA), ("U1.HB", U1HB_DECL, U1HB_DECL_SHA)):
        if O.sha256(O.rp(path)) != want:
            raise EngineeringFailure(f"{name}'s registration is not the bound one")
    u1 = json.loads(O.rp(U1HB_DECL).read_text(encoding="utf-8"))
    up = "dataprep_staging/u1_bands/power_u1hb.py"
    if (u1.get("code_sha256") or {}).get(up) != O.sha256(O.rp(up)):
        raise EngineeringFailure("power_u1hb.py differs from U1.HB.power.v1's bound code")
    seal = seal_lapsed(P.fv2_attempts)
    if not seal["lapsed"]:
        raise EngineeringFailure("FV v2 or FV.ext not scored: the seal has not lapsed")
    att = fv2_batches_from_attempts(P.fv2_attempts)
    reg = design.get("fv2_batches") or {}
    want = {f"{t}_b{b}" for t in TAGS for b in range(V2.PLAN.n_batch)}
    if set(reg) != want or any(reg[x] != att.get(x) for x in want):
        raise EngineeringFailure("registered FV v2 batch sha256s differ from FV v2's "
                                 "attempts.jsonl")
    return {"seal": seal, "fv2_batches": reg}


# ----------------------------------------------------------------------------- synthetic
def build_synthetic(out: Path, plan: V2.Plan, k_null: int = 6) -> tuple[Paths, dict]:
    """SYNTHETIC inputs with the real shapes (no real data): FV v2's synthetic set, a
    DR-style group table and report, and 'published' rates computed by osse_fv2.score_worlds
    itself on those inputs (so E0 checks CB against FV v2's own code)."""
    out = Path(out)
    p2 = V2.build_synthetic(out / "fv2", plan)
    rep = V2.score_worlds(p2, plan=plan, smoke=True, k_null=k_null, log=lambda *_: None)
    if rep.get("status") != "scored":
        raise EngineeringFailure(f"synthetic FV v2 not scored: {rep.get('why')}")
    O._write_json_once(out / "report_worlds.json", rep)
    ids = sorted(pd.read_csv(p2.v1.wells, dtype={"station_id": str})["station_id"])
    county = [S.county_of(s) for s in ids]
    groups = [f"{c}:{('prox', 'mid', 'dist')[i % 3]}" for i, c in enumerate(county)]
    pd.DataFrame({"station_id": ids, "county": county, "grouping": "zone",
                  "group": groups}).to_csv(out / "wells_e02d.csv", index=False)
    dr = {"decisions": {c: {"grouping": "zone", "include_drift": True,
                            "tau2_per_year2": t2}
                        for c, t2 in (("Changhua", 0.02), ("Yunlin", 0.0185))},
          "counties": {c: {"lead_definition": "origin", "lead_years": 4.375,
                           "lead_years_datum_centre_report_only": 9.875}
                       for c in ("Changhua", "Yunlin")}}
    (out / "dr_report.json").write_text(json.dumps(dr, indent=1) + "\n")
    v1 = p2.v1
    P = Paths(pred=v1.pred, protocol=v1.protocol, wells=v1.wells,
              noise=str(Path(v1.inputs_dir) / "noise_ar1.csv"),
              mask=str(Path(v1.inputs_dir) / "obs_mask.csv"), zr0=v1.zr0,
              batches=dict(p2.batches), fv2_attempts=str(out / "fv2_attempts.jsonl"),
              fv2_report=str(out / "report_worlds.json"),
              dr_report=str(out / "dr_report.json"), dr_wells=str(out / "wells_e02d.csv"),
              report=str(out / "smoke_report.json"), attempts=str(out / "attempts.jsonl"))
    return P, rep


def run_smoke(out: Path, k: int = 6, batch: int = 4, workers: int = 1) -> dict:
    """Synthetic end to end on SMOKE_SEED (no registry, no real data): smoke_report.json."""
    out = Path(out)
    plan = V2.Plan(k_truth=k, batch=batch)
    P, _ = build_synthetic(out / "inputs_synthetic", plan)
    ctx, info = load_ctx(P)
    truths, _ = load_truths(P, ctx, plan)
    pub = published_rates(P.fv2_report)
    # the synthetic 'published' rates are over k worlds (E1's anchor uses k_published = k)
    rep = run_study(ctx, truths, seed=SMOKE_SEED, published=pub, k=k, k_published=k,
                    workers=workers, log=lambda *_: None)
    rep.update({"smoke": True, "inputs": info,
                "note": "SMOKE: synthetic inputs only; numbers mean nothing about the twin"})
    O._write_json_once(out / "smoke_report.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def cmd_run(seed: int, workers: int, P: Paths | None = None, prereg_dir=None, rows_dir=None
            ) -> int:
    """Refuses until registered; writes report.json once. EVERY invocation appends one
    attempts row: refusals before registration, refused re-runs and interrupts included
    (BaseException is logged and re-raised)."""
    P = P or Paths()
    out, att = O.rp(P.report), O.rp(P.attempts)
    n = sum(1 for s in att.read_text().splitlines() if s.strip()) if att.is_file() else 0
    row = {"attempt": n + 1, "event": "run", "start": _now(), "seed": seed, "workers": workers}
    try:
        seed = check_seed(seed)
        doc, dsha = load_registered(prereg_dir=prereg_dir, rows_dir=rows_dir)
        row["declaration_sha256"] = dsha
        if out.exists():
            raise EngineeringFailure(f"{out} exists: CB is run once")
        bound = check_registered(doc, seed, P)
        pins = verify_pins(P)
        ctx, info = load_ctx(P)
        truths, used = load_truths(P, ctx, verify=bound["fv2_batches"])
        pub = published_rates(P.fv2_report)
        if abs(pub[COUNTY]["Q0"] - V2_Q0_CHANGHUA) > E0_TOL:
            raise EngineeringFailure("report_worlds.json Q0 Changhua is not the published "
                                     f"{V2_Q0_CHANGHUA}")
        rep = run_study(ctx, truths, seed=seed, published=pub, workers=workers)
        rep.update({"created": _now(), "declaration_sha256": dsha, "smoke": False,
                    "frozen_code": doc["frozen_code"], "inputs": {**pins, "ctx": info},
                    "fv2_batches_used": used, "seal": bound["seal"]})
        O._write_json_once(out, rep)
    except BaseException as err:
        _append_jsonl(att, {**row, "rc": 1, "error": f"{type(err).__name__}: {err}",
                            "time": _now()})
        raise
    _append_jsonl(att, {**row, "rc": 0, "status": rep["status"], "time": _now(),
                        "report_sha256": O.sha256(out)})
    print(f"wrote {P.report} (status {rep['status']})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-registration")
    r = sub.add_parser("run", help="once, after registration")
    r.add_argument("--seed", required=True, type=int)
    r.add_argument("--workers", type=int, default=1)
    sm = sub.add_parser("smoke", help="synthetic end to end")
    sm.add_argument("--out", required=True)
    sm.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out), workers=a.workers)
            print(json.dumps({"status": rep["status"], "reading": rep.get("reading", {})
                              .get("reading")}, indent=1))
            return 0
        if a.cmd == "verify-registration":
            _, dsha = load_registered()
            print(f"{ID} registered (sha256 {dsha[:16]}); frozen code and inputs intact")
            return 0
        if a.cmd == "run":
            return cmd_run(a.seed, a.workers)
    except EngineeringFailure as err:
        print(f"ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
