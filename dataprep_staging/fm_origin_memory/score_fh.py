"""Score FH.forcing_sensitivity.v1: in the page model, from how many months after a start does
the head level depend more on the weather and pumping that follow than on what is known at
the start?

Model-only. Three runs share one free-running December 2019 state (the page model's
hindcast is causal, so its 2012-01..2019-12 months do not depend on where the record ends):

  C19  NEW forward: the page recipe with --record-end 2019-12 --gain 0; projection 2020-01..
       2029-12 under the month-of-year climatology of the 2012-19 record (free run)
  N19  NEW forward: the same with --gain 1 (the page's innovation restart at 2019-12)
  R1   EXISTING fs_start_S1.npz (FS.start_invariance.v1): its hindcast months 2020-01..
       2022-12 are the same free run under the recorded 2020-22 forcing

  F(i, L) = R1 - C19  (forcing sensitivity under the 2020-22 history), L = 1..36
  M(i, L) = N19 - C19 (origin memory at the same 2019-12 origin),        L = 1..120
  L*_force = the first lead L in 1..36 with p90_F(L) > p90_M(L), p90 over the 158 frozen
  screens of |.| (numpy.percentile, linear).

Every npz is opened through fm_common.WhitelistedNpz: fs_start_S0/S1 with heads_mean,
dates, scenario_names, mask and origin; the two new runs also with the recipe metadata
horizon, gain, hindcast_gain, n_members, member_labels, forward_options and scenarios.
hindcast_r2 and gate are never loaded. The forwards' stdout (it prints per-member in-sample
hindcast R2) is sealed by run_fh.sh. No observation of any family is read here.

Order of work:
  0. registration: declaration registered and intact (inputs re-hashed), frozen code
     (fm_common.py, this scorer, run_fh.sh) unchanged, the head-well directory digest
     unchanged, each new npz the output of its arm's last successful forward attempt in
     attempts.jsonl, both arms at the same code;
  1. engineering checks on every file (names, dates, origin, grid, shape, recipe metadata);
  2. E1: the 2012-01..2019-12 heads of C19 and of N19 equal fs_start_S1's in every layer and
     cell (max |diff| < E1_TOL_M = 1e-3 m, 50 times below FLOOR_M; the report records whether
     they are bit-identical); exit 3 otherwise, no F or M computed. The band admits float32
     storage rounding (one ulp of a head of tens of metres is 2-8e-6 m) and any unordered
     CUDA index_add accumulation; a state difference below it moves F and M by far less
     than the floor;
  3. only then F, M, L*_force, the wording case and the report, written once (open 'x').

``--verify-registration`` runs step 0's registration part (no npz is opened); the runner
calls it before every forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fm_common as C  # noqa: E402

PREREG_ID = "FH.forcing_sensitivity.v1"
SCORER_VERSION = "1.0.0"
FROZEN = (f"{C.X}/fm_common.py", f"{C.X}/score_fh.py", f"{C.X}/run_fh.sh")
DEFAULT_OUT = f"results/dataprep/reports/{PREREG_ID}/report.json"
DEFAULT_ATTEMPTS = "results/twin_runs/fh_force/attempts.jsonl"
OUT_STEM = "results/twin_forward/fh_force_{arm}"
ARMS = ("C19", "N19")
DIGEST_KINDS = ("sorted_name_sha256_lines",)

FIRST = "2012-01"
FS_LAST, FS_ORIGIN = "2032-12", "2022-12"
NEW_LAST, NEW_ORIGIN = "2029-12", "2019-12"
I0 = C.month_index(FIRST, NEW_ORIGIN)             # 95: index of 2019-12
N_F = C.month_index(NEW_ORIGIN, FS_ORIGIN)        # 36 leads of recorded forcing (2020-22)
N_M = 120                                         # the new runs' horizon
I0_FS = C.month_index(FIRST, FS_ORIGIN)           # 131: FS's origin, for the report-only M
E1_TOL_M = 1e-3                                   # numeric band, 50 x below the floor
CASE_BOUND = 12
FS_SCEN = ("baseline", "cut30", "retire_aqua", "cut30_2030", "retire_aqua_2030")
NEW_SCEN = ("baseline",)
NEW_SCEN_DESC = ("baseline: baseline (no change)",)

FS_FILES = {
    "S1": ("results/twin_forward/fs_start_S1.npz",
           "507e3360ec7c579cacc875d17404e446c7fe3ed8aac84e985e1e1023a4ae27c2"),
    "S0": ("results/twin_forward/fs_start_S0.npz",
           "60e5553f9658a195bfaa69059cc636f9dd98f1f50374ba3e53dc978037f6cd0a"),
}
_FO = {"column_heads": "free", "update_method": "innovation", "apex_hold": "calibrated",
       "restart_taper_km": 5.0, "column_hpc0_fast_days": 365.0, "save_members": "none",
       "record_end": NEW_ORIGIN}
SPEC = {"C19": {"gain": 0.0, "fo": _FO}, "N19": {"gain": 1.0, "fo": _FO}}

WORDINGS = {
    "within_first_year": (
        "In this model, under the 2020-22 forcing history (which includes the 2021 drought), "
        "from about {L} months after a start the head level depends more on the weather and "
        "pumping that follow than on anything known at the start (model-only declaration "
        "FH.forcing_sensitivity.v1)"),
    "within_record": (
        "In this model, under the 2020-22 forcing history (which includes the 2021 drought), "
        "from about {L} months after a start the head level depends more on the weather and "
        "pumping that follow than on anything known at the start (model-only declaration "
        "FH.forcing_sensitivity.v1)"),
    "not_reached": (
        "In this model, under the 2020-22 forcing history (which includes the 2021 drought), "
        "what is known at a start outweighed the departure of weather and pumping from their "
        "averages for all 36 months available; no forcing horizon is stated (model-only "
        "declaration FH.forcing_sensitivity.v1)"),
    "not_evaluable": (
        "not evaluable: in this model the observation update at the 2019-12 start moves the "
        "heads by less than 0.05 m at 90 % of the screens one month later (model-only "
        "declaration FH.forcing_sensitivity.v1)"),
}


def wording(case: str, lstar: int | None) -> str:
    return WORDINGS[case].format(L=lstar)


def case_of(evaluable: bool, lstar: int | None) -> str:
    if not evaluable:
        return "not_evaluable"
    if lstar is None:
        return "not_reached"
    return "within_first_year" if lstar <= CASE_BOUND else "within_record"


# ----------------------------------------------------------------------------- step 0
def dir_digest(path: Path, pattern: str) -> tuple[int, str]:
    """(file count, sha256 over the sorted lines '<name> <sha256>\\n') of ``path/pattern``."""
    files = sorted(Path(path).glob(pattern))
    lines = "".join(f"{f.name} {C.sha256(f)}\n" for f in files)
    return len(files), hashlib.sha256(lines.encode()).hexdigest()


def check_input_digests(doc: dict, repo: Path = C.REPO) -> list[dict]:
    recs = doc.get("input_digests")
    if not isinstance(recs, list) or not recs:
        raise C.EngineeringError("the registered declaration carries no input_digests")
    bad = []
    for r in recs:
        if r.get("kind") not in DIGEST_KINDS:
            raise C.EngineeringError(f"input digest of unknown kind: {r.get('kind')}")
        n, h = dir_digest(repo / r["path"], r["glob"])
        if (n, h) != (r["n_files"], r["sha256"]):
            bad.append({"path": r["path"], "registered": [r["n_files"], r["sha256"]],
                        "now": [n, h]})
    if bad:
        raise C.EngineeringError(f"forward input directory changed since registration: {bad}")
    return recs


def check_attempts(attempts: Path, files: dict[str, Path]) -> dict:
    """Every new npz is the output of its arm's last forward attempt, which exited 0; both
    arms ran the same code."""
    if not attempts.exists():
        raise C.EngineeringError(f"no attempt log {attempts}")
    recs = [json.loads(s) for s in attempts.read_text().splitlines() if s.strip()]
    out, codes = {}, set()
    for arm, p in files.items():
        fw = [r for r in recs if r.get("event") == "forward" and r.get("arm") == arm]
        if not fw:
            raise C.EngineeringError(f"attempt log has no forward record for {arm}")
        last = fw[-1]
        if int(last.get("rc", -1)) != 0:
            raise C.EngineeringError(f"last forward attempt of {arm} exited {last.get('rc')}")
        if last.get("npz_sha256") != C.sha256(p):
            raise C.EngineeringError(f"{p} is not the npz the last {arm} forward wrote")
        codes.add(last.get("code"))
        out[arm] = {"attempt": last.get("attempt"), "time": last.get("time"),
                    "code": last.get("code"), "worktree_head": last.get("worktree_head")}
    if len(codes) != 1:
        raise C.EngineeringError(f"the arms ran different code: {sorted(map(str, codes))}")
    return out


# ----------------------------------------------------------------------------- step 1
def load_new(path: Path, arm: str) -> dict:
    d = C.load_run(path, allowed=C.FM_KEYS + C.FH_META_KEYS, names=NEW_SCEN, first=FIRST,
                   last=NEW_LAST, origin_month=NEW_ORIGIN)
    m, e = d["meta"], []
    missing = sorted(set(C.FH_META_KEYS) - set(m))
    if missing:
        raise C.EngineeringError(f"{path}: recipe keys missing {missing}")
    if int(m["horizon"]) != N_M:
        e.append(f"horizon {int(m['horizon'])}")
    if float(m["gain"]) != SPEC[arm]["gain"] or float(m["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(m['gain'])} hindcast_gain {float(m['hindcast_gain'])}")
    if int(m["n_members"]) != 2 * len(C.PAGE_MEMBERS):
        e.append(f"n_members {int(m['n_members'])}")
    if tuple(str(s) for s in m["member_labels"]) != C.PAGE_MEMBERS:
        e.append("member_labels differ from the page's 18 parameter sets")
    if tuple(str(s) for s in m["scenarios"]) != NEW_SCEN_DESC:
        e.append(f"scenarios {tuple(str(s) for s in m['scenarios'])}")
    fo = json.loads(str(m["forward_options"]))
    bad = {k: fo.get(k) for k, v in SPEC[arm]["fo"].items() if fo.get(k) != v}
    for k in ("climatology_end", "extend_record_to", "recorded_pumping"):
        if k in fo:
            bad[k] = fo[k]
    if bad:
        e.append(f"forward_options differ: {bad}")
    if e:
        raise C.EngineeringError(f"{path}: " + "; ".join(e))
    d["meta"] = {"checked": sorted(SPEC[arm]["fo"]) + ["gain", "horizon", "n_members",
                                                         "member_labels", "scenarios"]}
    return d


def load_fs(path: Path) -> dict:
    return C.load_run(path, allowed=C.FM_KEYS, names=FS_SCEN, first=FIRST, last=FS_LAST,
                      origin_month=FS_ORIGIN)


# ----------------------------------------------------------------------------- step 2
def e1(new: dict, s1: dict) -> dict:
    """E1: the new runs' 2012-01..2019-12 heads equal fs_start_S1's in every layer and
    cell (baseline scenario)."""
    out, ok = {}, True
    ref = np.asarray(s1["_heads"][0, :, :, :I0 + 1], dtype=np.float64)
    for arm, d in new.items():
        x = np.asarray(d["_heads"][0, :, :, :I0 + 1], dtype=np.float64)
        v = float(np.max(np.abs(x - ref)))
        out[arm] = {"max_abs_diff_m": v, "ok": bool(v < E1_TOL_M),
                    "bit_identical": bool(v == 0.0)}
        ok &= v < E1_TOL_M
    return {"ok": bool(ok), "tol_m": E1_TOL_M, "months": f"{FIRST}..{NEW_ORIGIN}",
            "arms": out}


# ----------------------------------------------------------------------------- step 3
def score(c19: dict, n19: dict, s1: dict, s0: dict | None = None) -> dict:
    m = C.masks()
    f = C.diff(s1, c19, "baseline", I0, N_F)              # (158, 36)
    mm = C.diff(n19, c19, "baseline", I0, N_M)            # (158, 120)
    pf, pm = C.p90_abs(f), C.p90_abs(mm)
    evaluable = bool(pm[0] >= C.FLOOR_M)
    lstar = C.first_exceeds(pf, pm[:N_F]) if evaluable else None
    case = case_of(evaluable, lstar)
    primary = {
        "evaluable": evaluable, "lstar_force_months": lstar,
        "lstar_force_month": C.lead_month(NEW_ORIGIN, lstar) if lstar else None,
        "case": case, "wording": wording(case, lstar),
        "p90_F_m": [float(x) for x in pf], "p90_M_m": [float(x) for x in pm],
        "p90_at_leads_m": {"F": C.at_leads(pf), "M": C.at_leads(pm, (1, 3, 6, 12, 24, 36,
                                                                      48, 60, 120))},
    }
    rep: dict = {}
    rep["stays_above_lead"] = C.stays_above(pf, pm[:N_F])

    def agg(a_f, a_m):
        return {"F_at_leads_m": C.at_leads(a_f), "M_at_leads_m": C.at_leads(a_m),
                "first_exceeds": C.first_exceeds(a_f, a_m[:N_F]),
                "stays_above": C.stays_above(a_f, a_m[:N_F])}

    rep["n2_unit_yunlin84"] = agg(C.unit_abs(f, m["yunlin"]), C.unit_abs(mm, m["yunlin"]))
    rep["changhua74_mean"] = agg(C.unit_abs(f, m["changhua"]),
                                 C.unit_abs(mm, m["changhua"]))
    rep["fan_layer2_mean"] = agg(np.abs(C.fan_diff(s1, c19, "baseline", I0, N_F)),
                                 np.abs(C.fan_diff(n19, c19, "baseline", I0, N_M)))
    rep["p90_by_county"] = {
        cty: agg(C.p90_abs(f, m[cty]), C.p90_abs(mm, m[cty]))
        for cty in ("yunlin", "changhua")}
    rep["M_2019_efold_n2_unit"] = C.efold(C.unit_abs(mm, m["yunlin"]))
    rep["M_2019_stays_below_lead"] = C.stays_below([pm])
    if s0 is not None:
        # the 2022-12 origin memory of the FS forwards (|S1 - S0|), reported separately and
        # never compared with F (another origin)
        m22 = C.diff(s1, s0, "baseline", I0_FS, N_F)
        rep["M_2022_from_fs"] = {"p90_at_leads_m": C.at_leads(C.p90_abs(m22)),
                                 "n2_unit_at_leads_m": C.at_leads(C.unit_abs(m22,
                                                                             m["yunlin"]))}
    return {"primary": primary, "reported_not_gating": rep, "floor_m": C.FLOOR_M,
            "percentile": C.PCTL, "n_F": N_F, "n_M": N_M, "case_bound": CASE_BOUND}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for arm in ARMS:
        ap.add_argument(f"--{arm.lower()}", default=OUT_STEM.format(arm=arm) + ".npz")
    for k, (p, _) in FS_FILES.items():
        ap.add_argument(f"--fs-{k.lower()}", default=p)
    ap.add_argument("--attempts", default=DEFAULT_ATTEMPTS)
    ap.add_argument("--out", default=None, help=f"report JSON (scoring; e.g. {DEFAULT_OUT})")
    ap.add_argument("--verify-registration", action="store_true",
                    help="step 0's registration part only; opens no npz")
    a = ap.parse_args(argv)
    C.static_checks()
    reg = None
    try:
        if not a.verify_registration and not a.out:
            raise C.EngineeringError("scoring needs --out")
        if a.out and C.rp(a.out).exists():
            raise C.EngineeringError(f"{a.out} exists: the declaration is scored once")
        doc, decl_sha = C.load_registered(PREREG_ID)
        frozen = C.check_frozen(doc, FROZEN)
        digests = check_input_digests(doc)
        fs = {k: C.rp(getattr(a, f"fs_{k.lower()}")) for k in FS_FILES}
        for k, p in fs.items():
            C.check_sha(p, FS_FILES[k][1], f"fs_start_{k}")
        reg = {"declaration_sha256": decl_sha, "frozen_code_sha256": frozen,
               "input_digests": digests}
        print(f"step 0 ok: {PREREG_ID} registered (sha256 {decl_sha[:16]}), frozen code and "
              "declared inputs match")
        if a.verify_registration:
            return 0
        files = {arm: C.rp(getattr(a, arm.lower())) for arm in ARMS}
        reg["attempts"] = check_attempts(C.rp(a.attempts), files)
    except C.EngineeringError as err:
        print(f"ENGINEERING FAIL (step 0): {err}", file=sys.stderr)
        return 1
    try:
        new = {arm: load_new(p, arm) for arm, p in files.items()}
        s1 = load_fs(fs["S1"])
    except C.EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    chk = e1(new, s1)
    for arm, r in chk["arms"].items():
        print(f"step 2 E1: {arm} vs fs_start_S1 over {chk['months']}: max |diff| "
              f"{r['max_abs_diff_m']:.3e} m ({'ok' if r['ok'] else 'FAIL'})")
    if not chk["ok"]:
        print("ENGINEERING FAIL (step 2): the runs do not share the 2019-12 state; no F or M "
              "computed", file=sys.stderr)
        return 3
    try:
        s0 = load_fs(fs["S0"])
    except C.EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1, S0): {err}", file=sys.stderr)
        return 2
    res = score(new["C19"], new["N19"], s1, s0)
    pr = res["primary"]
    report = {"prereg_id": PREREG_ID, "scorer_version": SCORER_VERSION, "registration": reg,
              "screens_sha256": C.SCREENS_SHA256,
              "files": {k: {"path": v["path"], "sha256": v["sha256"],
                            "keys_read": v["keys_read"]}
                        for k, v in {**new, "fs_start_S1": s1, "fs_start_S0": s0}.items()},
              "e1": chk, **res}
    print(f"L*_force = {pr['lstar_force_months']} months; case {pr['case']}")
    p = C.rp(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:                     # never overwrite a scored report
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
