"""Score X5.changhua_lever_model.v1 (model-only declaration, round 5 rank 8): R1c's all-class
Changhua head lever beside the Changhua-pumping family (structural spread, L), beside its
2016-18 masked refit (T, an L comparison), and F, the gain of N2C's own per-well-demeaned
slope estimator on R1c's response to replayed pre-origin county anomalies relative to R1c's
steady-state lever per unit fractional cut.

Model-only. No observation of any family is read. Two modes:

``precheck`` (BEFORE registration; reads baseline-only forwards): every arm's pre-build
baseline (results/twin_runs/x5_chlever/pre/<arm>.npz, written by ``run_chlever.sh prebuild``
from the frozen worktree) against its archived forward at the 158 frozen screens:
single-member archives (rs, x5r1t, b1) heads_mean <= REF_TOL_M (heads_mean averages the
unperturbed and one perturbed ic member, so an arm can also fail on ic-random-stream drift;
either way it is excluded as 'baseline not reproduced'); 18-member archives (ccz, chc,
chs, cdm, csw, ccs, cpw) the main member's unperturbed (ic 0) December layer-2 heads from
<archive>.members.npz (1 cm quanta) <= REF_TOL_M; ccz_m1618 has no archive, so its pre-build
IS the archive, and two pre-builds with different columns (z3_w1618 and z3_w1618_tau3650)
must give the same heads (<= COL_INDEP_TOL_M everywhere): the head lever is
column-independent under --column-heads free (code: twin.forward.run accumulates heads_mean
from the restarted rollout only; the column sees heads_col). Writes precheck.json; prints
only differences. No lever exists in these files.

score (AFTER registration, once): step 0 registration, frozen code and attempt log; step 1
engineering checks on every scored npz (scenario names and strings, recorded-arm file shas
and scopes, dates, shapes, member label, forward options, the theta meta equal to the
pre-build's); step 2 reproduction: scored baseline at the screens vs the pre-build baseline
(<= REF_TOL_M), and the disclosed class-share files (q0c, x5r1) checked against their
registered sha256 and metadata (their values are already known); only baseline values of the
scored runs are read in steps 1-2. Step 3 (only if 0-2 pass): the levers, the spread, T, F and
the declared readings; report.json written once (mode 'x'). Evaluability is per window
(L_ccz(w) >= 0.05 m gates the spread and T lines of window w only); the only C13a cap is the
user-approved F < 0.5 (ruling 2026-11-01).
A failure in 0-2 is an engineering failure (exit 1-3): nothing else is computed or printed.

Usage (repository root; the runner calls it):
    python dataprep_staging/x5_changhua_lever/score_chlever.py precheck
    python dataprep_staging/x5_changhua_lever/score_chlever.py score \
        --out results/dataprep/reports/X5.changhua_lever_model.v1/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
X = "dataprep_staging/x5_changhua_lever"
sys.path.insert(0, str(REPO / "dataprep_staging" / "n2_policy"))
import bootstrap_scorer as BS  # noqa: E402

PREREG_ID = "X5.changhua_lever_model.v1"
SCORER_VERSION = "1.0.0"
WELLS = "dataprep_staging/x5_class_share/wells_frozen.json"
WELLS_SHA = "fc2d8257dbe729238a96faf96f2e0b8e3ff82facc7faddb03794b453c2fffcd6"
FROZEN = (f"{X}/score_chlever.py", f"{X}/run_chlever.sh", f"{X}/replay.py",
          "dataprep_staging/n2_policy/bootstrap_scorer.py", WELLS)
LOG = "results/twin_runs/x5_chlever"
PRE = f"{LOG}/pre"
REPLAYS = f"{LOG}/replays"
PRECHECK = f"{LOG}/precheck.json"
ATTEMPTS = f"{LOG}/attempts.jsonl"
OUT_STEM = "results/twin_forward/chlever_{arm}"
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"
DISCLOSED_NPZ = {"q0c": "results/twin_forward/class_share_q0c.npz",
                 "x5r1": "results/twin_forward/class_share_x5r1.npz"}

# arm -> role, rung, gate status, column dir, archive kind, in the spread band
_F = "failed at least one registered gate"
ARMS = {
    "ccz": dict(role="reference (R1c, N2C primary)", rung="I1.R1c.county_zones.v1",
                gates="FAILED 1 of 4 (Changhua head-anomaly RMSE 1.318 vs 1.074 m); G2 fresh "
                      "partition failed (k-fold R2 0.646, Changhua county-mean RMSE 0.643 m)",
                column="coupled_leveling", archive="ensemble18", band=False),
    "ccz_m1618": dict(role="transfer comparison T (report only)",
                      rung="N1.policy_response.v1 step C masked refit (R1c recipe, 2016-18 "
                           "masked)", gates="no rung gate (design-time masked refit)",
                      column="coupled_leveling_z3_w1618", archive="prebuild", band=False),
    "chc": dict(role="family", rung="I1.R1.changhua_census.v1", gates=_F + " (2 of 4)",
                column="coupled_leveling", archive="ensemble18", band=True),
    "chs": dict(role="family", rung="I1.R1b.census_canal.v1", gates=_F + " (2 of 4)",
                column="coupled_leveling", archive="ensemble18", band=True),
    "cdm": dict(role="family", rung="I1.R1d_dom.v1", gates=_F, column="coupled_leveling",
                archive="ensemble18", band=True),
    "csw": dict(role="family", rung="I1.R1d_sw.v1", gates=_F, column="coupled_leveling",
                archive="ensemble18", band=True),
    "ccs": dict(role="family", rung="I1.R1e_cn.v1", gates=_F, column="coupled_leveling",
                archive="ensemble18", band=True),
    "rs": dict(role="family", rung="I1.Rstar.v1", gates=_F + " (Changhua county-mean RMSE)",
               column="coupled_leveling", archive="single", band=True),
    "x5r1t": dict(role="family", rung="I1.X5R1.v1 (township conversion arm)", gates=_F,
                  column="coupled_leveling", archive="single", band=True),
    "cpw": dict(role="report only, excluded from the band (fixed extra abstraction not "
                     "scaled by all30)", rung="I1.R1e_pw.v1", gates=_F + "; G2 failed",
                column="coupled_leveling", archive="ensemble18", band=False),
    "b1": dict(role="report only, excluded from the band (fixed extra abstraction not "
                    "scaled by all30)", rung="I1.B.v1", gates=_F, column="coupled_leveling",
               archive="single", band=False),
}
DISCLOSED_ARMS = {"q0c": "known all30 lever (X5.lever_class_share.v1, scored 2026-10-29); "
                         "reported-only arm of I1.X5R1.v1 (fails cmrmse_ch and leveling)",
                  "x5r1": "known all30 lever (X5.lever_class_share.v1); I1.X5R1.v1 primary, "
                          "FAILED"}
COL_B = "coupled_leveling_z3_w1618_tau3650"     # ccz_m1618's second column (independence)

BASE, ALL30 = "baseline", "all30"
SCEN_DESC = {BASE: "baseline: baseline (no change)",
             ALL30: "all30: aquaculture x0.7, domestic x0.7, industry x0.7, irrigation x0.7, "
                    "livestock x0.7, other x0.7 from 2026-01"}
REPLAY_YEARS = (2012, 2013, 2014, 2015, 2016, 2017, 2018)
CUT_ARMS = {"CH30": ("ch30.csv", "changhua"), "YL30": ("yl30.csv", "yunlin")}
REPLAY_ARMS = {f"RP{y}": f"rp{y}.npz" for y in REPLAY_YEARS}
CCZ_SCEN = (BASE, ALL30, *CUT_ARMS, *REPLAY_ARMS)
OTHER_SCEN = (BASE, ALL30)

N_T, ORIGIN, HORIZON = 252, 131, 120           # 2012-01 .. 2032-12; origin 2022-12
WINDOWS = {"2026-27": (168, 192), "2030-32": (216, 252)}
WINDOW_FIRST = {"2026-27": "2026-01", "2030-32": "2030-01"}
N2C_WINDOW = (178, 190)                        # 2026-11 .. 2027-10
STEADY = "2030-32"
CUT_FRACTION = 0.3
MIN_LEVER_M = 0.05                             # evaluable_if, per window: L_ccz(w) >= 0.05 m
F_MIN_RMS_M = BS.INFORMATIVE_HEAD_M            # F evaluable if RMS of s~ >= 0.05 m
F_BANDS = (0.5, 0.8)
REF_TOL_M = 0.01
COL_INDEP_TOL_M = 1e-4
FWD_OPTIONS = {"column_heads": "free", "update_method": "innovation", "apex_hold": "calibrated",
               "restart_taper_km": 5.0, "column_hpc0_fast_days": 365.0,
               "save_members": "yearly"}
# the disclosed class-share forwards (X5.lever_class_share.v1's run): checked, never re-run
DISCLOSED_SCEN = ("baseline", "cut30", "all30")
DISCLOSED_OPTIONS = {**FWD_OPTIONS, "save_members": "none"}
DROP_LABEL = {"single": "baseline not reproduced (code or ic-random-stream drift: heads_mean "
                        "averages the unperturbed member and one perturbed ic member)",
              "ensemble18": "baseline not reproduced (code drift: the unperturbed main member "
                            "is compared)",
              "prebuild": "column independence failed"}

WORDING = {
    "F_ge_0p8": "an N2C 'response supported' may say the window exercises most of R1c's "
                "Changhua head lever (F = {F:.2f}, model-only)",
    "F_0p5_0p8": "an N2C 'response supported' is worded 'N2C decides the first-year part "
                 "(F = {F:.2f}); the rest of the lever is model-only extrapolation'",
    "F_lt_0p5": "C13a is capped at N even after an N2C 'response supported': it may show only "
                "'first-year response' wording (F = {F:.2f} < 0.5), never S",
    "F_not_evaluable": "F not evaluable (RMS of the steady-state replay contrast below "
                       "0.05 m): no F wording; this declaration adds no cap (the 2026-11-01 "
                       "ruling approved only the F < 0.5 cap)",
    "spread": "model-internal lever spread (L): across {n} Changhua-pumping variants, every "
              "one of which failed at least one registered gate, the all-class x0.7 head "
              "lever at N2C's 74 Changhua wells is x{lo1:.2f}-{hi1:.2f} of R1c's in 2026-27 "
              "and x{lo2:.2f}-{hi2:.2f} in 2030-32",
    "spread_one": "model-internal lever spread (L): across {n} Changhua-pumping variants, "
                  "every one of which failed at least one registered gate, the all-class x0.7 "
                  "head lever at N2C's 74 Changhua wells is x{lo:.2f}-{hi:.2f} of R1c's in {w}",
    "T": "L comparison, no gate: R1c's lever is x{t1:.2f} (2026-27) and x{t2:.2f} (2030-32) "
         "that of its 2016-18 masked refit (the model step C scored); step C's design-time "
         "support belongs to the masked refit",
    "T_one": "L comparison, no gate: R1c's lever is x{t:.2f} ({w}) that of its 2016-18 masked "
             "refit (the model step C scored); step C's design-time support belongs to the "
             "masked refit",
    "not_evaluable": "not evaluable in {w}: R1c's all-class lever at the 74 Changhua wells is "
                     "below 0.05 m there; no spread or T line for {w}; no cap follows from it",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(s: str | Path) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


class EngineeringError(RuntimeError):
    """A file or the registration is not what the declaration names; nothing is scored."""


# ----------------------------------------------------------------------------- wells
def wells() -> dict:
    p = REPO / WELLS
    if sha256(p) != WELLS_SHA:
        raise SystemExit(f"{WELLS} sha256 differs from {WELLS_SHA[:8]}")
    rows = json.loads(p.read_text())["rows"]
    cells = np.array([r["cell"] for r in rows], int)
    layers = np.array([r["layer"] for r in rows], int)
    county = np.array([r["county"] for r in rows])
    sids = np.array([r["sid"] for r in rows])
    if len(rows) != 158 or int((county == "changhua").sum()) != 74:
        raise SystemExit("wells_frozen.json is not 158 screens / 74 Changhua")
    return {"sid": sids, "cell": cells, "layer": layers, "county": county,
            "changhua": county == "changhua", "yunlin": county == "yunlin"}


# ----------------------------------------------------------------------------- loading
def load_npz(path: Path, names: tuple[str, ...], W: dict, *, arm: str,
             read: tuple[str, ...]) -> dict:
    """Engineering checks, then the screen heads of the scenarios in ``read`` only."""
    f = np.load(path, allow_pickle=False)
    e = []
    sn = tuple(str(s) for s in f["scenario_names"])
    if sn != names:
        e.append(f"scenario_names {sn} != {names}")
    sd = [str(s) for s in f["scenarios"]]
    for i, n in enumerate(sn):
        if n in SCEN_DESC and sd[i] != SCEN_DESC[n]:
            e.append(f"scenario {n!r} string {sd[i]!r}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != N_T or dates[0] != "2012-01" or dates[-1] != "2032-12":
        e.append(f"dates {dates[0]}..{dates[-1]} ({len(dates)})")
    else:
        for w, (a, _) in WINDOWS.items():
            if dates[a] != WINDOW_FIRST[w]:
                e.append(f"window {w} starts {dates[a]}")
        if dates[N2C_WINDOW[0]] != "2026-11" or dates[N2C_WINDOW[1] - 1] != "2027-10":
            e.append("N2C window months misplaced")
    if int(f["origin"]) != ORIGIN or int(f["horizon"]) != HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    if f["heads_mean"].shape != (len(names), 4, 2148, N_T):
        e.append(f"heads_mean shape {f['heads_mean'].shape}")
    labels = [str(s) for s in f["member_labels"]]
    if labels != [f"stage3_datum_gate_{arm}"]:
        e.append(f"member_labels {labels}")
    if float(f["gain"]) != 1.0 or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    fo = json.loads(str(f["forward_options"]))
    bad = {k: fo.get(k) for k, v in FWD_OPTIONS.items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    if e:
        raise EngineeringError(f"{path}: " + "; ".join(e))
    hm = f["heads_mean"]
    h = {n: np.asarray(hm[sn.index(n)][W["layer"], W["cell"], :], np.float64) for n in read}
    for n, v in h.items():
        if not np.isfinite(v).all():
            raise EngineeringError(f"{path}: non-finite {n} heads at the screens")
    return {"path": str(path), "sha256": sha256(path), "names": sn, "h": h,
            "gate": json.loads(str(f["gate"])), "forward_options": fo}


def members_l2(path: Path, label: str, W: dict) -> tuple[np.ndarray, float]:
    """The unperturbed (ic 0) member's December layer-2 heads at the screens' cells,
    baseline scenario, from a ``--save-members yearly`` sidecar: (int16 quanta, quantum m)."""
    z = np.load(path, allow_pickle=False)
    names = [str(s) for s in z["scenario_names"]]
    hm = [str(s) for s in z["head_member"]]
    ic = np.asarray(z["head_ic"])
    k = [i for i, (m, j) in enumerate(zip(hm, ic, strict=True)) if m == label and j == 0]
    if len(k) != 1:
        raise EngineeringError(f"{path}: {len(k)} members {label!r} ic 0")
    q = np.asarray(z["headL2_members_yr"][names.index(BASE), k[0]], np.int64)
    return q[W["cell"], :], float(z["head_scale_m"])


def load_disclosed(path: Path, arm: str, W: dict) -> dict:
    """A disclosed class-share forward: metadata checks, then baseline and all30 at the
    screens. Engineering only: the values are already known (X5.lever_class_share.v1)."""
    f = np.load(path, allow_pickle=False)
    e = []
    sn = tuple(str(s) for s in f["scenario_names"])
    if sn != DISCLOSED_SCEN:
        e.append(f"scenario_names {sn} != {DISCLOSED_SCEN}")
    else:
        sd = [str(s) for s in f["scenarios"]]
        for n in (BASE, ALL30):
            if sd[sn.index(n)] != SCEN_DESC[n]:
                e.append(f"scenario {n!r} string {sd[sn.index(n)]!r}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != N_T or dates[0] != "2012-01" or dates[-1] != "2032-12":
        e.append(f"dates {dates[0]}..{dates[-1]} ({len(dates)})")
    else:
        for w, (a, _) in WINDOWS.items():
            if dates[a] != WINDOW_FIRST[w]:
                e.append(f"window {w} starts {dates[a]}")
    if int(f["origin"]) != ORIGIN or int(f["horizon"]) != HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    if f["heads_mean"].shape != (len(DISCLOSED_SCEN), 4, 2148, N_T):
        e.append(f"heads_mean shape {f['heads_mean'].shape}")
    labels = [str(s) for s in f["member_labels"]]
    if labels != [f"stage3_datum_gate_{arm}"]:
        e.append(f"member_labels {labels}")
    if float(f["gain"]) != 1.0 or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    fo = json.loads(str(f["forward_options"]))
    bad = {k: fo.get(k) for k, v in DISCLOSED_OPTIONS.items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    if e:
        raise EngineeringError(f"{path}: " + "; ".join(e))
    hm = f["heads_mean"]
    h = {n: np.asarray(hm[sn.index(n)][W["layer"], W["cell"], :], np.float64)
         for n in (BASE, ALL30)}
    if not all(np.isfinite(v).all() for v in h.values()):
        raise EngineeringError(f"{path}: non-finite heads at the screens")
    return {"h": h, "sha256": sha256(path)}


# ----------------------------------------------------------------------------- precheck
def precheck(pre_dir: Path, W: dict) -> dict:
    """Pre-registration reproduction of every arm's baseline-only pre-build (no lever)."""
    out = {"what": "baseline-only pre-builds (no all30, no replay) against the archived "
                   "forwards; engineering, before registration", "ref_tol_m": REF_TOL_M,
           "arms": {}}
    for arm, spec in ARMS.items():
        p = pre_dir / f"{arm}.npz"
        d = load_npz(p, (BASE,), W, arm=arm, read=(BASE,))
        rec = {"prebuild": str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),
               "prebuild_sha256": d["sha256"],
               "prebuild_members_sha256": sha256(pre_dir / f"{arm}.members.npz"),
               "archive_kind": spec["archive"]}
        if spec["archive"] == "single":
            ref = REPO / f"results/twin_forward/datum_gate_{arm}.npz"
            r = np.load(ref, allow_pickle=False)
            names = [str(s) for s in r["scenario_names"]]
            hr = np.asarray(r["heads_mean"][names.index(BASE)][W["layer"], W["cell"], :],
                            np.float64)
            diff = float(np.max(np.abs(d["h"][BASE] - hr)))
            rec.update(reference=str(ref.relative_to(REPO)), reference_sha256=sha256(ref),
                       compared="heads_mean baseline at the 158 screens, 252 months")
        elif spec["archive"] == "ensemble18":
            ref = REPO / f"results/twin_forward/datum_gate_{arm}.members.npz"
            label = f"stage3_datum_gate_{arm}"
            (a, qa), (b, qb) = (members_l2(ref, label, W),
                                members_l2(pre_dir / f"{arm}.members.npz", label, W))
            if qa != qb:
                raise EngineeringError(f"{arm}: member quanta differ ({qa} vs {qb})")
            dq = int(np.max(np.abs(a - b)))           # in quanta: exact integer compare
            diff = dq * qa
            rec["max_abs_diff_quanta"] = dq
            rec["quanta_tol"] = int(round(REF_TOL_M / qa))
            rec.update(reference=str(ref.relative_to(REPO)), reference_sha256=sha256(ref),
                       compared="main member ic 0, December layer-2 heads (1 cm quanta) at "
                                "the screens' cells, 21 years")
        else:                                            # ccz_m1618: column independence
            q = np.load(pre_dir / f"{arm}_colB.npz", allow_pickle=False)
            f = np.load(p, allow_pickle=False)
            diff_all = float(np.max(np.abs(np.asarray(f["heads_mean"], np.float64)
                                           - np.asarray(q["heads_mean"], np.float64))))
            diff = 0.0
            rec.update(reference="none (no archived forward); this pre-build is the archive",
                       column_independence={
                           "columns": [ARMS[arm]["column"], COL_B],
                           "colB_sha256": sha256(pre_dir / f"{arm}_colB.npz"),
                           "max_abs_heads_mean_diff_m_all_cells": diff_all,
                           "tol_m": COL_INDEP_TOL_M, "ok": diff_all <= COL_INDEP_TOL_M})
        rec["max_abs_diff_m"] = diff
        close = (rec["max_abs_diff_quanta"] <= rec["quanta_tol"] if "quanta_tol" in rec
                 else diff <= REF_TOL_M)
        rec["ok"] = bool(close and rec.get("column_independence", {"ok": True})["ok"])
        out["arms"][arm] = rec
        print(f"precheck {arm}: max |diff| {diff:.2e} m ({spec['archive']})"
              + (f"; column independence {rec['column_independence']['ok']}"
                 if "column_independence" in rec else ""))
    out["ok_reference"] = bool(out["arms"]["ccz"]["ok"] and out["arms"]["ccz_m1618"]["ok"])
    out["band_arms"] = [a for a, s in ARMS.items() if s["band"] and out["arms"][a]["ok"]]
    out["drift_excluded"] = [a for a, s in ARMS.items() if s["band"] and not out["arms"][a]["ok"]]
    out["drift_excluded_reason"] = {a: DROP_LABEL[ARMS[a]["archive"]]
                                    for a in out["drift_excluded"]}
    return out


# ----------------------------------------------------------------------------- step 0
def load_registered(prereg_dir: Path | None = None, rows_dir: Path | None = None
                    ) -> tuple[dict, str]:
    sys.path.insert(0, str(REPO / "dataprep_staging"))
    from harness import prereg3

    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    try:
        return prereg3.load_prereg(PREREG_ID, **kw)
    except prereg3.PreregistrationError as err:
        raise EngineeringError(f"registration: {err}") from err


def check_frozen(doc: dict, repo: Path = REPO) -> dict:
    want = doc.get("frozen_code") or {}
    got = {p: sha256(repo / p) if (repo / p).exists() else "missing" for p in FROZEN}
    bad = {p: {"registered": want.get(p), "now": got[p]} for p in FROZEN
           if want.get(p) != got[p]}
    if bad:
        raise EngineeringError(f"frozen code differs from the registered declaration: {bad}")
    return got


def check_attempts(attempts: Path, files: dict[str, Path]) -> dict:
    if not attempts.exists():
        raise EngineeringError(f"no attempt log {attempts}")
    recs = [json.loads(s) for s in attempts.read_text().splitlines() if s.strip()]
    out = {}
    for arm, p in files.items():
        fw = [r for r in recs if r.get("event") == "forward" and r.get("arm") == arm]
        if not fw:
            raise EngineeringError(f"attempt log has no forward record for {arm}")
        last = fw[-1]
        if int(last.get("rc", -1)) != 0:
            raise EngineeringError(f"last forward attempt of {arm} exited {last.get('rc')}")
        if last.get("npz_sha256") != sha256(p):
            raise EngineeringError(f"{p} is not the npz the last {arm} forward wrote")
        out[arm] = {"attempt": last.get("attempt"), "time": last.get("time")}
    return out


def check_recorded(fo: dict, declared: dict) -> None:
    """The ccz run's recorded arms are the declared replay and cut files."""
    rec = fo.get("recorded_pumping") or {}
    want = set(CUT_ARMS) | set(REPLAY_ARMS)
    if set(rec) != want:
        raise EngineeringError(f"recorded arms {sorted(rec)} != {sorted(want)}")
    for name, r in rec.items():
        d = declared[name]
        if r.get("sha256") != d["sha256"] or r.get("county") != d["county"]:
            raise EngineeringError(f"recorded arm {name}: sha/county {r.get('sha256')}/"
                                   f"{r.get('county')} != declared {d}")
        if r.get("per_cell") != d["per_cell"] or r.get("start") != d["start"]:
            raise EngineeringError(f"recorded arm {name}: per_cell/start differ")


def declared_recorded(replays: dict) -> dict:
    out = {}
    for name, (f, county) in CUT_ARMS.items():
        key = f.split(".")[0]
        out[name] = {"sha256": replays["cuts"][key]["sha256"], "county": county,
                     "per_cell": False, "start": "2026-01"}
    for name, f in REPLAY_ARMS.items():
        key = f.split(".")[0]
        out[name] = {"sha256": replays["replays"][key]["sha256"], "county": None,
                     "per_cell": True, "start": "2023-01"}
    return out


# ----------------------------------------------------------------------------- step 3
def lever(d: dict, scen: str) -> np.ndarray:
    return d["h"][scen] - d["h"][BASE]                       # (screen, month), + = higher


def wmean(lv: np.ndarray, mask: np.ndarray, w: str) -> float:
    a, b = WINDOWS[w]
    return float(lv[mask, a:b].mean())


def f_gain(ccz: dict, W: dict, replays: dict) -> dict:
    """F: N2C's per-well-demeaned slope of R1c's replay contrast c on the steady-state
    contrast s = -(lambda_CH a_CH + lambda_YL a_YL), pooled over replays (one FE group per
    replay and well), over the N2C window at the 74 Changhua screens."""
    ch = W["changhua"]
    a, b = WINDOWS[STEADY]
    lam = {c: lever(ccz, arm)[ch, a:b].mean(axis=1) / CUT_FRACTION
           for c, arm in (("changhua", "CH30"), ("yunlin", "YL30"))}
    t0, t1 = N2C_WINDOW
    cs, ss, gs, per = [], [], [], {}
    for name, f in REPLAY_ARMS.items():
        win = replays["replays"][f.split(".")[0]]["window"]
        aa = {c: np.asarray(win[f"a_{c}"], float) for c in ("changhua", "yunlin")}
        if len(aa["changhua"]) != t1 - t0:
            raise EngineeringError(f"{name}: window anomalies have {len(aa['changhua'])} months")
        c = lever(ccz, name)[ch, t0:t1]                                  # (74, 12)
        s = -(lam["changhua"][:, None] * aa["changhua"][None, :]
              + lam["yunlin"][:, None] * aa["yunlin"][None, :])
        g = np.repeat([f"{name}|{i}" for i in range(c.shape[0])], c.shape[1])
        uq, inv = np.unique(g, return_inverse=True)
        st = BS.within(s.ravel(), inv, len(uq))
        ct = BS.within(c.ravel(), inv, len(uq))
        per[name] = {"F": BS.slope_fe(c.ravel(), s.ravel(), g),
                     "rms_s_tilde_m": float(np.sqrt(np.mean(st ** 2))),
                     "rms_c_tilde_m": float(np.sqrt(np.mean(ct ** 2))),
                     "n2c_informative_rms_ok_report_only":
                         bool(np.sqrt(np.mean(ct ** 2)) >= BS.INFORMATIVE_HEAD_M),
                     "window_mean_c_m": float(c.mean()),
                     "source": replays["replays"][f.split(".")[0]]["window"]["source"][0]}
        cs.append(c.ravel())
        ss.append(s.ravel())
        gs.append(g)
    c_all, s_all, g_all = np.concatenate(cs), np.concatenate(ss), np.concatenate(gs)
    uq, inv = np.unique(g_all, return_inverse=True)
    rms_s = float(np.sqrt(np.mean(BS.within(s_all, inv, len(uq)) ** 2)))
    gain = BS.slope_fe(c_all, s_all, g_all)
    evaluable = bool(rms_s >= F_MIN_RMS_M and np.isfinite(gain))
    if not evaluable:
        key = "F_not_evaluable"
    elif gain >= F_BANDS[1]:
        key = "F_ge_0p8"
    elif gain >= F_BANDS[0]:
        key = "F_0p5_0p8"
    else:
        key = "F_lt_0p5"
    Fs = [v["F"] for v in per.values()]
    return {"F": gain, "evaluable": evaluable, "rms_s_tilde_pooled_m": rms_s,
            "reading_key": key, "wording": WORDING[key].format(F=gain if np.isfinite(gain) else 0),
            "caps_c13a_at_N": key == "F_lt_0p5",
            "per_replay_report_only": per, "F_range_over_replays": [min(Fs), max(Fs)],
            "lambda_m_per_unit_cut": {c: {"median": float(np.median(v)),
                                          "mean": float(v.mean())} for c, v in lam.items()}}


def score(runs: dict, disclosed: dict, W: dict, replays: dict, band_arms: list[str]) -> dict:
    ch, yl = W["changhua"], W["yunlin"]
    L = {arm: {w: wmean(lever(d, ALL30), ch, w) for w in WINDOWS} for arm, d in runs.items()}
    Ly = {arm: {w: wmean(lever(d, ALL30), yl, w) for w in WINDOWS} for arm, d in runs.items()}
    for arm, d in disclosed.items():
        L[arm] = {w: wmean(lever(d, ALL30), ch, w) for w in WINDOWS}
        Ly[arm] = {w: wmean(lever(d, ALL30), yl, w) for w in WINDOWS}
    ok_w = [w for w in WINDOWS if L["ccz"][w] >= MIN_LEVER_M]
    low = [w for w in WINDOWS if w not in ok_w]
    res = {"L_m": L, "L_yunlin_m_report_only": Ly, "min_lever_m": MIN_LEVER_M,
           "evaluable_windows": ok_w, "evaluable": bool(ok_w)}
    ratio = {arm: {w: L[arm][w] / L["ccz"][w] for w in ok_w} for arm in L if arm != "ccz"}
    fam = {w: [ratio[a][w] for a in band_arms] for w in ok_w}
    fam_d = {w: fam[w] + [ratio[a][w] for a in disclosed] for w in ok_w}
    spread = {w: [min(fam[w]), max(fam[w])] for w in ok_w} if band_arms else {}
    res["ratio_to_ccz"] = ratio
    res["spread"] = {"band_arms": band_arms, "by_window": spread,
                     "with_disclosed_q0c_x5r1": {w: [min(v), max(v)] for w, v in fam_d.items()
                                                 if v},
                     "excluded_report_only": [a for a in ARMS if a not in band_arms
                                              and a not in ("ccz", "ccz_m1618")]}
    T = {w: L["ccz"][w] / L["ccz_m1618"][w] if L["ccz_m1618"][w] != 0 else float("nan")
         for w in ok_w}
    res["T_report_only"] = T
    ccz = runs["ccz"]
    sup = {w: float((lever(ccz, "CH30") + lever(ccz, "YL30"))[ch, WINDOWS[w][0]:WINDOWS[w][1]]
                    .mean() / L["ccz"][w]) for w in ok_w}
    res["superposition_report_only"] = {
        "what": "(CH30 + YL30 lever) / all30 lever at the 74 Changhua screens: county cuts "
                "through the recorded mechanism against the class scenario",
        "by_window": sup,
        "county_share_changhua": {w: float(lever(ccz, "CH30")[ch, WINDOWS[w][0]:WINDOWS[w][1]]
                                           .mean() / L["ccz"][w]) for w in ok_w}}
    res["F"] = f_gain(ccz, W, replays)
    lay = W["layer"]
    res["L_by_layer_report_only"] = {
        arm: {f"L{k + 1}": (wmean(lever(d, ALL30), ch & (lay == k), STEADY)
                            if (ch & (lay == k)).any() else None) for k in range(4)}
        for arm, d in runs.items()}
    readings = {"F": res["F"]["wording"], "caps_c13a_at_N": res["F"]["caps_c13a_at_N"]}
    if low:
        readings["not_evaluable"] = WORDING["not_evaluable"].format(w=", ".join(low))
    if spread and len(ok_w) == len(WINDOWS):
        readings["spread_L"] = WORDING["spread"].format(
            n=len(band_arms), lo1=spread["2026-27"][0], hi1=spread["2026-27"][1],
            lo2=spread["2030-32"][0], hi2=spread["2030-32"][1])
    elif spread:
        w = ok_w[0]
        readings["spread_L"] = WORDING["spread_one"].format(
            n=len(band_arms), w=w, lo=spread[w][0], hi=spread[w][1])
    if len(ok_w) == len(WINDOWS):
        readings["T_L"] = WORDING["T"].format(t1=T["2026-27"], t2=T["2030-32"])
    elif ok_w:
        readings["T_L"] = WORDING["T_one"].format(w=ok_w[0], t=T[ok_w[0]])
    res["readings"] = readings
    return res


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("precheck", help="pre-registration reproduction of the pre-builds")
    pc.add_argument("--pre", default=PRE)
    pc.add_argument("--out", default=PRECHECK)
    sc = sub.add_parser("score", help="score once (after registration)")
    sc.add_argument("--out", default=None)
    sc.add_argument("--attempts", default=ATTEMPTS)
    sc.add_argument("--stem", default=OUT_STEM)
    sc.add_argument("--verify-registration", action="store_true")
    a = ap.parse_args(argv)
    W = wells()
    if a.cmd == "precheck":
        if (REPO / f"results/dataprep/preregistration/{PREREG_ID}.json").exists():
            print("REFUSED: registered; the precheck is frozen in the declaration",
                  file=sys.stderr)
            return 1
        try:
            res = precheck(rp(a.pre), W)
        except (EngineeringError, OSError, KeyError) as err:
            print(f"PRECHECK FAIL: {err}", file=sys.stderr)
            return 2
        out = rp(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {a.out} (sha256 {sha256(out)}); reference ok {res['ok_reference']}; "
              f"band arms {res['band_arms']}; drift-excluded {res['drift_excluded']}")
        return 0 if res["ok_reference"] else 3
    try:
        if not a.verify_registration and not a.out:
            raise EngineeringError("scoring needs --out")
        if a.out and rp(a.out).exists():
            raise EngineeringError(f"{a.out} exists: the declaration is scored once")
        doc, decl_sha = load_registered()
        frozen = check_frozen(doc)
        print(f"step 0 ok: {PREREG_ID} registered (sha256 {decl_sha[:16]}), frozen code matches")
        if a.verify_registration:
            return 0
        reg = {"declaration_sha256": decl_sha, "frozen_code_sha256": frozen}
        pre = doc["precheck"]
        files = {arm: rp(a.stem.format(arm=arm) + ".npz") for arm in ARMS}
        reg["attempts"] = check_attempts(rp(a.attempts), files)
        rjs = rp(REPLAYS) / "replays.json"
        if sha256(rjs) != doc["replays"]["replays_json_sha256"]:
            raise EngineeringError("replays.json differs from the registered one")
        replays = json.loads(rjs.read_text())
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 0): {err}", file=sys.stderr)
        return 1
    runs, repro = {}, {}
    try:
        for arm in ARMS:
            names = CCZ_SCEN if arm == "ccz" else OTHER_SCEN
            d = load_npz(files[arm], names, W, arm=arm, read=(BASE,))
            p = load_npz(rp(PRE) / f"{arm}.npz", (BASE,), W, arm=arm, read=(BASE,))
            if p["sha256"] != pre["arms"][arm]["prebuild_sha256"]:
                raise EngineeringError(f"{arm}: pre-build differs from the registered one")
            if d["gate"] != p["gate"]:
                raise EngineeringError(f"{arm}: theta meta differs from the pre-build's")
            if arm == "ccz":
                check_recorded(d["forward_options"], declared_recorded(replays))
            diff = float(np.max(np.abs(d["h"][BASE] - p["h"][BASE])))
            repro[arm] = {"max_abs_diff_m": diff, "ok": diff <= REF_TOL_M}
            runs[arm] = d
    except (EngineeringError, OSError, KeyError) as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    for arm, r in repro.items():
        print(f"  {arm}: baseline vs pre-build max |diff| {r['max_abs_diff_m']:.2e} m")
    if not all(r["ok"] for r in repro.values()):
        print("ENGINEERING FAIL (step 2): baselines do not reproduce the pre-builds; no lever "
              "read", file=sys.stderr)
        return 3
    disclosed = {}
    try:
        for arm, path in DISCLOSED_NPZ.items():
            disclosed[arm] = load_disclosed(rp(path), arm, W)
            if disclosed[arm]["sha256"] != doc["disclosed_files_sha256"][arm]:
                raise EngineeringError(f"{path} differs from the registered sha256")
    except (EngineeringError, OSError, KeyError) as err:
        print(f"ENGINEERING FAIL (step 2, disclosed arms): {err}", file=sys.stderr)
        return 2
    for arm in ARMS:                                   # step 3 only: the scenario values
        names = CCZ_SCEN if arm == "ccz" else OTHER_SCEN
        runs[arm] = load_npz(files[arm], names, W, arm=arm, read=names)
    res = score(runs, disclosed, W, replays, list(doc["precheck"]["band_arms"]))
    report = {"prereg_id": PREREG_ID, "scorer_version": SCORER_VERSION, "registration": reg,
              "wells_sha256": WELLS_SHA, "arms": ARMS, "disclosed_arms": DISCLOSED_ARMS,
              "files": {a: {"path": d["path"], "sha256": d["sha256"]} for a, d in runs.items()},
              "disclosed_files": {a: {"path": DISCLOSED_NPZ[a], "sha256": d["sha256"]}
                                  for a, d in disclosed.items()},
              "reproduction": repro, **res}
    for k, v in res.get("readings", {}).items():
        print(f"  {k}: {v}")
    p = rp(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, default=float)
    print(f"wrote {a.out} (sha256 {sha256(p)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
