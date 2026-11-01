"""Score FM.origin_memory.v1: for how many months does the page model's head level remember
its December 2022 start?

Model-only. It reads three forward runs that already exist, written and scored under the
model-only declaration FS.start_invariance.v1 (results/twin_forward/fs_start_S0/S1/S2.npz,
bound here by sha256), and the page's registered forward run (results/twin_forward/
datum_gate_innov.npz) for the engineering gate. Every npz is opened through
fm_common.WhitelistedNpz with the keys heads_mean, dates, scenario_names, mask and origin only;
hindcast_r2 (an in-sample score against observations), gate and every other key are refused.
The fs_start CSVs and sealed logs are never opened. No observation of any family is read.

  S0  the page recipe (innovation restart at the Dec-2022 origin, gain 1)
  S1  the same with gain 0: every member starts from its own free-running Dec-2022 state
  S2  the same with the old untapered IDW field restart (gain 1)

Estimands (baseline scenario, the 158 frozen screens of N2/E02, each in its own layer;
lead L = 1..48 is 2023-01..2026-12):
  d_k(i, L) = h_k(i, L) - h_S0(i, L), k in {S1, S2};
  p90_k(L)  = 90th percentile over the 158 screens of |d_k(i, L)|;
  L*        = the first lead after which p90_S1 and p90_S2 both stay strictly below
              FLOOR_M = 0.05 m at every later lead through 2026-12 (None if not at 2026-12);
  e-folding time of |mean over N2's 84 Yunlin screens of d_k| (report).
S0's heads_mean averages its two initial fields per parameter set (the innovation restart
and the same with sigma 0.5 m seed-0 noise on the observations); S1's two fields are
identical (gain 0). So d_S1 is the page ensemble mean's start (the observation update plus
half the seed-0 perturbation's effect) against the free run.

The E02/E21 interpretation clause covers the registered 2026-11..2027-10 window (leads
47..58), which runs past the decision window: it says 'beyond the start memory' only if L* is
defined and p90_S1 and p90_S2 stay below FLOOR_M at every lead through 58 (2027-10, read from
the same files' continuation); 'partly within' if L* is defined but either reaches FLOOR_M
again by lead 58; 'within' if L* is undefined. The clause case never changes L*, G-def or the
E23/C30 wordings.

Order of work:
  0. registration: the declaration is registered and intact (prereg3.load_prereg re-hashes
     every declared input, the three fs_start npz and the page run included) and the
     sha256 of the frozen code equals its frozen_code;
  1. engineering checks on every file (names, dates, origin, grid, shape, finiteness);
  2. G-eng: S0's baseline heads at the 158 screens equal the page run's in every month
     (max |diff| <= REPRO_TOL_M = 0.0 m, as FS's reproduction recorded); exit 3 otherwise;
  3. only then the curves, L*, the wording case and the report, written once (open 'x').

``--verify-registration`` runs step 0 only (no npz is opened).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fm_common as C  # noqa: E402

PREREG_ID = "FM.origin_memory.v1"
SCORER_VERSION = "1.0.0"
FROZEN = (f"{C.X}/fm_common.py", f"{C.X}/score_fm.py")
DEFAULT_OUT = f"results/dataprep/reports/{PREREG_ID}/report.json"

FS_SCEN = ("baseline", "cut30", "retire_aqua", "cut30_2030", "retire_aqua_2030")
REF_SCEN = ("baseline", "cut30", "retire_aqua")
FIRST, LAST, ORIGIN_MONTH = "2012-01", "2032-12", "2022-12"
I0 = C.month_index(FIRST, ORIGIN_MONTH)          # 131
N_LEADS = 48                                     # 2023-01 .. 2026-12 (the decision window)
N_LEADS_ALL = 120                                # 2023-01 .. 2032-12 (report-only)
WINDOW_E02_FIRST_LEAD = 47                       # 2026-11, the registered E02 window ...
WINDOW_E02_LAST_LEAD = 58                        # ... through 2027-10
CASE_BOUNDS = (12, 47)                           # L* <= 12 | 12 < L* <= 47 | L* > 47
REPRO_TOL_M = 0.0

FILES = {
    "S0": ("results/twin_forward/fs_start_S0.npz",
           "60e5553f9658a195bfaa69059cc636f9dd98f1f50374ba3e53dc978037f6cd0a"),
    "S1": ("results/twin_forward/fs_start_S1.npz",
           "507e3360ec7c579cacc875d17404e446c7fe3ed8aac84e985e1e1023a4ae27c2"),
    "S2": ("results/twin_forward/fs_start_S2.npz",
           "3853214cb716063baa9907716829affb685ff94e74ccc9c5d04287fb61f8343f"),
}
REF = ("results/twin_forward/datum_gate_innov.npz",
       "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228")

# the wording of each outcome, fixed before any curve is computed; {L} is L* in months and
# {date} the month of lead L* (C.lead_month)
WORDINGS = {
    "within_first_year": {
        "E23": "start memory: in this model the December 2022 start shapes the head levels "
               "for about {L} months (until {date}); model-only declaration "
               "FM.origin_memory.v1",
        "C30_heads": "In this model, after about {L} months (from {date}) the head levels "
                     "are the model's average-weather scenario, not a forecast from the "
                     "observations; the start shapes only the first year",
    },
    "beyond_first_year": {
        "E23": "start memory: in this model the December 2022 start shapes the head levels "
               "for about {L} months (until {date}), more than a year; model-only "
               "declaration FM.origin_memory.v1",
        "C30_heads": "In this model, after about {L} months (from {date}) the head levels "
                     "are the model's average-weather scenario, not a forecast from the "
                     "observations",
    },
    "longer_than_record": {
        "E23": "start memory: in this model the December 2022 start still shapes the head "
               "levels in 2026; start memory longer than the 2023-26 record, no horizon "
               "stated (model-only declaration FM.origin_memory.v1)",
        "C30_heads": None,
    },
}
# the E02/E21 interpretation clause, by how the start memory meets the registered
# 2026-11..2027-10 window (leads 47..58); {L} is L* in months
E02_E21_CLAUSES = {
    "window_beyond_memory": (
        "in this model the 2026-11..2027-10 window lies beyond the start memory (about {L} "
        "months; from 2026-11 through 2027-10 the start moves the heads by less than 0.05 m "
        "at 90 % of the screens), so the level rule tests the model's average-weather level "
        "and datum, not the 2022-12 start"),
    "window_partly_within_memory": (
        "in this model the start memory measured to 2026-12 is about {L} months, but the "
        "start again moves the heads by 0.05 m or more at 90 % of the screens before "
        "2027-10, so the 2026-11..2027-10 level rule also tests the 2022-12 start in part"),
    "window_within_memory": (
        "in this model the 2026-11..2027-10 window lies within the start memory, so the "
        "level rule also tests the 2022-12 start"),
}


def case_of(lstar: int | None) -> str:
    if lstar is None or lstar > CASE_BOUNDS[1]:
        return "longer_than_record"
    return "within_first_year" if lstar <= CASE_BOUNDS[0] else "beyond_first_year"


def window_case(lstar: int | None, lstar_window: int | None) -> str:
    """The E02/E21 clause case: ``lstar`` is L* (decided through lead 48, None if
    undefined), ``lstar_window`` the 'stays below' lead of both arms over leads 1..58."""
    if lstar is None:
        return "window_within_memory"
    if lstar_window is not None and lstar_window <= WINDOW_E02_FIRST_LEAD:
        return "window_beyond_memory"
    return "window_partly_within_memory"


def wording(lstar: int | None, wcase: str) -> dict:
    case = case_of(lstar)
    if (case == "longer_than_record") != (wcase == "window_within_memory"):
        raise ValueError(f"window case {wcase} does not go with {case}")
    fill = {"L": lstar, "date": C.lead_month(ORIGIN_MONTH, lstar) if lstar else None}
    return {"case": case, **{k: (v.format(**fill) if v else None)
                             for k, v in WORDINGS[case].items()},
            "E02_E21_clause_case": wcase,
            "E02_E21_clause": E02_E21_CLAUSES[wcase].format(**fill)}


# ----------------------------------------------------------------------------- loading
def load_fs(path: Path) -> dict:
    return C.load_run(path, allowed=C.FM_KEYS, names=FS_SCEN, first=FIRST, last=LAST,
                      origin_month=ORIGIN_MONTH)


def load_ref(path: Path) -> dict:
    return C.load_run(path, allowed=C.FM_KEYS, names=REF_SCEN, first=FIRST, last=LAST,
                      origin_month=ORIGIN_MONTH)


def reproduction(s0: dict, ref: dict) -> dict:
    """G-eng: S0's baseline heads at the 158 screens against the page run's, every month."""
    a = s0["h"][s0["names"].index("baseline")]
    b = ref["h"][ref["names"].index("baseline")]
    d = float(np.max(np.abs(a - b)))
    return {"ok": bool(d <= REPRO_TOL_M), "max_abs_diff_m": d, "tol_m": REPRO_TOL_M,
            "reference": ref["path"], "reference_sha256": ref["sha256"]}


# ----------------------------------------------------------------------------- scoring
def memory_curves(runs: dict, scen: str = "baseline", n_leads: int = N_LEADS) -> dict:
    """Per arm (S1, S2): the per-screen differences from S0 at leads 1..n_leads."""
    return {k: C.diff(runs[k], runs["S0"], scen, I0, n_leads) for k in ("S1", "S2")}


def score(runs: dict) -> dict:
    m = C.masks()
    lay = np.asarray(C.LAYERS)
    d = memory_curves(runs)
    p90 = {k: C.p90_abs(v) for k, v in d.items()}
    lstar = C.stays_below([p90["S1"], p90["S2"]])
    defined = lstar is not None and lstar <= N_LEADS - 1
    # the continuation past the decision window (2027-01..2032-12) from the same files
    d_all = memory_curves(runs, n_leads=N_LEADS_ALL)
    p_all = {k: C.p90_abs(v) for k, v in d_all.items()}
    lstar_window = C.stays_below([v[:WINDOW_E02_LAST_LEAD] for v in p_all.values()])
    wcase = window_case(lstar if defined else None, lstar_window)
    w = wording(lstar if defined else None, wcase)
    n2 = {k: C.unit_abs(v, m["yunlin"]) for k, v in d.items()}
    primary = {
        "lstar_months": lstar if defined else None,
        "lstar_month": C.lead_month(ORIGIN_MONTH, lstar) if defined else None,
        "lstar_raw": lstar,
        "g_def_pass": bool(defined),
        "wording": w,
        "window_e02_beyond_memory": wcase == "window_beyond_memory",
        "lstar_through_2027_10": lstar_window,
        "p90_m": {k: [float(x) for x in v] for k, v in p90.items()},
        "p90_at_leads_m": {k: C.at_leads(v) for k, v in p90.items()},
        "lstar_per_arm": {k: C.stays_below([v]) for k, v in p90.items()},
        "n2_unit_abs_m": {k: [float(x) for x in v] for k, v in n2.items()},
        "n2_unit_efold": {k: C.efold(v) for k, v in n2.items()},
    }
    rep: dict = {}
    rep["n2_unit_lstar"] = C.stays_below([n2["S1"], n2["S2"]])
    rep["p90_by_county_m"] = {
        cty: {k: C.at_leads(C.p90_abs(v, m[cty])) for k, v in d.items()}
        for cty in ("yunlin", "changhua")}
    rep["lstar_by_county"] = {
        cty: C.stays_below([C.p90_abs(v, m[cty]) for v in d.values()])
        for cty in ("yunlin", "changhua")}
    rep["p90_by_layer_m"] = {
        f"L{k + 1}": ({a: C.at_leads(C.p90_abs(v, lay == k)) for a, v in d.items()}
                      if (lay == k).any() else None) for k in range(4)}
    rep["lstar_by_layer"] = {
        f"L{k + 1}": (C.stays_below([C.p90_abs(v, lay == k) for v in d.values()])
                      if (lay == k).any() else None) for k in range(4)}
    rep["fan_layer2_abs_m"] = {
        k: C.at_leads(np.abs(C.fan_diff(runs[k], runs["S0"], "baseline", I0, N_LEADS)))
        for k in ("S1", "S2")}
    # continuation past the decision window (2027-01..2032-12), report-only
    rep["continuation_to_2032"] = {
        "lstar_through_2032": C.stays_below([p_all["S1"], p_all["S2"]]),
        "max_p90_after_lstar_m": (
            {k: float(v[lstar - 1:].max()) for k, v in p_all.items()} if defined else None)}
    # the other four scenarios (start memory should not depend on the scenario)
    rep["scenarios_check"] = {}
    for s in FS_SCEN[1:]:
        ds = memory_curves(runs, scen=s)
        ps = [C.p90_abs(v) for v in ds.values()]
        rep["scenarios_check"][s] = {
            "lstar": C.stays_below(ps),
            "max_abs_dev_from_baseline_curve_m": float(max(
                np.max(np.abs(ds[k] - d[k])) for k in ds))}
    return {"primary": primary, "reported_not_gating": rep, "floor_m": C.FLOOR_M,
            "percentile": C.PCTL, "n_leads": N_LEADS, "case_bounds": list(CASE_BOUNDS)}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for arm, (p, _) in FILES.items():
        ap.add_argument(f"--{arm.lower()}", default=p)
    ap.add_argument("--ref", default=REF[0])
    ap.add_argument("--out", default=None, help=f"report JSON (scoring; e.g. {DEFAULT_OUT})")
    ap.add_argument("--verify-registration", action="store_true",
                    help="step 0 only; opens no npz")
    a = ap.parse_args(argv)
    C.static_checks()
    try:
        if not a.verify_registration and not a.out:
            raise C.EngineeringError("scoring needs --out")
        if a.out and C.rp(a.out).exists():
            raise C.EngineeringError(f"{a.out} exists: the declaration is scored once")
        doc, decl_sha = C.load_registered(PREREG_ID)
        frozen = C.check_frozen(doc, FROZEN)
        files = {arm: C.rp(getattr(a, arm.lower())) for arm in FILES}
        for arm, p in files.items():
            C.check_sha(p, FILES[arm][1], arm)
        C.check_sha(C.rp(a.ref), REF[1], "page run")
    except C.EngineeringError as err:
        print(f"ENGINEERING FAIL (step 0): {err}", file=sys.stderr)
        return 1
    print(f"step 0 ok: {PREREG_ID} registered (sha256 {decl_sha[:16]}), frozen code and "
          "inputs match")
    if a.verify_registration:
        return 0
    try:
        ref = load_ref(C.rp(a.ref))
        runs = {arm: load_fs(p) for arm, p in files.items()}
    except C.EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    rep = reproduction(runs["S0"], ref)
    print(f"step 2 G-eng: max |S0 - page run| baseline at the 158 screens "
          f"{rep['max_abs_diff_m']:.3e} m ({'ok' if rep['ok'] else 'FAIL'})")
    if not rep["ok"]:
        print("ENGINEERING FAIL (step 2): S0 is not the page run; no curve computed",
              file=sys.stderr)
        return 3
    res = score(runs)
    pr = res["primary"]
    report = {"prereg_id": PREREG_ID, "scorer_version": SCORER_VERSION,
              "registration": {"declaration_sha256": decl_sha, "frozen_code_sha256": frozen},
              "screens_sha256": C.SCREENS_SHA256,
              "files": {k: {"path": v["path"], "sha256": v["sha256"],
                            "keys_read": v["keys_read"]} for k, v in
                        {**runs, "page_run": ref}.items()},
              "g_eng": rep, **res}
    print(f"L* = {pr['lstar_months']} months ({pr['lstar_month']}); G-def "
          f"{'pass' if pr['g_def_pass'] else 'FAIL'}; wording case {pr['wording']['case']}")
    p = C.rp(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:                      # never overwrite a scored report
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
