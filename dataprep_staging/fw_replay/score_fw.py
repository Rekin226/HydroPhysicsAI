"""Score FW.weather_replay.v1: does a dry year expose the compaction column's permanent
(inelastic) channel? Model-only.

The page model's projections run under month-of-year climatology, under which no cell sets
a new head minimum after 2026-01 (page_posthoc: margin 0.24 m at the 5th percentile), so the
avoided subsidence of a pumping cut is all elastic (C15). This declaration replays the
2012-22 forcing years instead (hydrophysics/twin/replay.py, every field from the same source
year, pumping as anomalies about its 2012-22 trend) in the 11 cyclic orders of 2012..2022,
and measures the inelastic share of cut30's avoided subsidence by Dec 2032.

Inputs (all written by the runner run_fw.sh; nothing else is read):
  results/twin_forward/fw_replay/<arm>.npz        the forward's own output (heads, subsidence)
  results/twin_forward/fw_replay/<arm>.fw.npz     the replay capture (elastic / inelastic parts)
  results/twin_runs/fw_replay/attempts.jsonl      the runner's attempt log
  results/twin_runs/fw_replay/gate_forcing.json   the forcing identity gate record
  results/twin_forward/datum_gate_innov.npz       the page run (reproduction of arm C only)
No observation of any family is read.

Arms (the declaration's recipe; every arm runs the page model, theta stage3_datum_gate with
--column-heads free and the rheology axis coupled_leveling (page column d290c592) +
coupled_leveling_tau30):
  C           climatology, page ensemble (18 sets x 2 initial fields)   reference + repro
  W2012..W2022  replay of the cyclic order starting at that year, detrended pumping,
              page ensemble                                              PRIMARY
  RD2012..    the same with the page theta alone (--ic-members 0)       report-only
  RR2012..    raw (not detrended) pumping, page theta alone             report-only
  RW2012..    weather only (pumping at climatology), page theta alone   report-only; only
              when the declaration replays weather (recipe.replay_fields 'energy+weather')

Order of work (the declaration's attempt policy):
  0. registration: the declaration is registered and intact, the scorer and runner hash to
     its frozen_code, the declared input files and the head-well digest are unchanged; every
     file is its arm's last successful forward output in attempts.jsonl and all arms ran the
     run code; a passing forcing gate and a passing reproduction of C are logged, and no
     failed reproduction (exit 3) is logged at all;
  1. engineering checks on every file (scenarios, dates, shapes, members, options, replay
     record, engine code sha256, capture totals against the forward's own subsidence);
  2. reproduction: C's heads and subsidence (baseline, cut30, retire_aqua) equal the page
     run's within REF_TOL_HEAD_M / REF_TOL_SUBS_M at every cell and month;
  3. only if 0-2 pass: the primary statistic, the wording case and the report-only tables;
     the report is written once (open mode 'x').
``--repro`` runs steps 0-2 on C alone; ``--verify-registration`` step 0's registration only.
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
X = "dataprep_staging/fw_replay"

PREREG_ID = "FW.weather_replay.v1"
SCORER_VERSION = "1.0.0"
FROZEN = (f"{X}/score_fw.py", f"{X}/run_fw.sh")
ENGINE = {"replay.py": "hydrophysics/twin/replay.py",
          "forward.py": "hydrophysics/twin/forward.py",
          "scenario.py": "hydrophysics/twin/scenario.py",
          "compaction.py": "hydrophysics/twin/compaction.py"}

SEQ_STARTS = tuple(range(2012, 2023))
PRIMARY = tuple(f"W{y}" for y in SEQ_STARTS)
RED_DET = tuple(f"RD{y}" for y in SEQ_STARTS)
RED_RAW = tuple(f"RR{y}" for y in SEQ_STARTS)
RED_WEA = tuple(f"RW{y}" for y in SEQ_STARTS)
ARMS = ("C", *PRIMARY, *RED_DET, *RED_RAW, *RED_WEA)       # every arm the runner knows
FIELD_MODES = ("energy+weather", "energy")


def arms(fields: str) -> tuple[str, ...]:
    """The arms a declaration with these replayed fields runs and scores: the weather-only
    arms exist only when weather is replayed (pumping-only mode has nothing to replay)."""
    return ARMS if fields == "energy+weather" else ARMS[:-len(RED_WEA)]

OUT_STEM = "results/twin_forward/fw_replay/{arm}"
LOGDIR = "results/twin_runs/fw_replay"
DEFAULT_ATTEMPTS = f"{LOGDIR}/attempts.jsonl"
GATE_JSON = f"{LOGDIR}/gate_forcing.json"
REF = "results/twin_forward/datum_gate_innov.npz"
REF_SHA256 = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
DIGEST_KINDS = ("sorted_name_sha256_lines",)

SCEN_NAMES = ("baseline", "cut30", "retire_aqua", "cut30_lift")
SCEN_DESC = ("baseline: baseline (no change)",
             "cut30: irrigation x0.7 from 2026-01",
             "retire_aqua: aquaculture x0 from 2026-01",
             "cut30_lift: irrigation x0.7 from 2026-01 to 2029-12")
REF_NAMES = SCEN_NAMES[:3]
PAGE_MEMBERS = ("stage3_datum_gate", "fold0", "fold1", "fold2", "fold3", "fold4",
                *(f"foldpost{i}" for i in range(12)))
ONE_MEMBER = ("stage3_datum_gate",)
RHEOLOGY = ("coupled_leveling", "tau30y")
PRIMARY_RHEOLOGY = "coupled_leveling"            # page column d290c592
THETA_GIT = "fa5cca2"
_FO = {"column_heads": "free", "update_method": "innovation", "apex_hold": "calibrated",
       "restart_taper_km": 5.0, "column_hpc0_fast_days": 365.0, "save_members": "none"}

N_T, ORIGIN, HORIZON, N_CELLS = 252, 131, 120, 2148       # 2012-01..2032-12; origin 2022-12
YEARS = tuple(range(2012, 2033))                          # Decembers in the capture
PROJ_YEARS = tuple(range(2023, 2033))
Y_ORIGIN, Y_LIFT, Y_END = YEARS.index(2022), YEARS.index(2029), YEARS.index(2032)
ZONE_NAMES = ("proximal", "mid", "distal")                # the column's three zones
GROUPS = ("fan", "zone0", "zone1", "zone2", "yunlin_rest", "changhua")

SHARE_THRESHOLD = 0.10            # the wording rule: median share >= this -> 'inelastic'
ALL_ELASTIC_ABS = 0.0005          # 'all elastic' (not 'mostly') when every |share| < this
SOURCE_YEAR_MIN_SHARE = 0.05      # C27 lists source years holding >= this of prevented lows
MIN_AVOIDED_M = 0.001             # a sequence whose fan-mean avoided subsidence is below
MIN_EVALUABLE = 9                 # ... is not evaluable; fewer evaluable -> not_evaluable
FROM_YEAR = 2026                  # crossings counted from the cut's start as well
REF_TOL_HEAD_M = 0.01             # C vs the page run, heads (m), every cell and month
REF_TOL_SUBS_M = 0.005            # ... subsidence (m)
CAPTURE_TOL_M = 1e-5              # capture parts vs the forward's subsidence (m)
PARTS_TOL_M = 1e-6                # capture parts total vs the column (m)


def cyclic(y0: int) -> tuple[int, ...]:
    base = list(SEQ_STARTS)
    k = base.index(int(y0))
    return tuple(base[k:] + base[:k])


def years_text(seq) -> str:
    """Folded text of a sequence (hydrophysics.twin.replay.format_years)."""
    ys, parts, i = [int(y) for y in seq], [], 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1:
            j += 1
        parts.append(str(ys[i]) if i == j else f"{ys[i]}-{ys[j]}")
        i = j + 1
    return ",".join(parts)


def arm_spec(arm: str, fields: str) -> dict:
    """What every arm's files must say."""
    if arm == "C":
        return {"members": PAGE_MEMBERS, "n_members": 36, "projection_forcing": "climatology",
                "replay": None, "role": "reference"}
    y = int(arm[-4:])
    seq = cyclic(y)
    page = arm.startswith("W")
    pumping = {"RR": "raw", "RW": "climatology"}.get(arm[:2], "detrended")
    f = "energy+weather" if arm.startswith("RW") else fields
    return {"members": PAGE_MEMBERS if page else ONE_MEMBER, "n_members": 36 if page else 1,
            "projection_forcing": f"replay:{years_text(seq)}",
            "replay": {"years": list(seq), "pumping": pumping, "fields": f,
                       "trend_years": [2012, 2022]},
            "role": "primary" if page else "report_only"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class EngineeringError(RuntimeError):
    """A file or the registration is not what the declaration names; nothing is scored."""


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


def declared_fields(doc: dict) -> str:
    f = (doc.get("recipe") or {}).get("replay_fields")
    if f not in FIELD_MODES:
        raise EngineeringError(f"the declaration's recipe.replay_fields {f!r} is not one of "
                               f"{FIELD_MODES}")
    return f


def declared_engine(doc: dict) -> dict:
    eng = (doc.get("recipe") or {}).get("engine_code_sha256") or {}
    if set(eng) != set(ENGINE):
        raise EngineeringError(f"the declaration's engine code lists {sorted(eng)}")
    return eng


def declared_county(doc: dict) -> dict:
    """The county map the declaration names for the report-only county split."""
    c = (doc.get("recipe") or {}).get("county_map") or {}
    if not c.get("path") or not c.get("sha256"):
        raise EngineeringError("the declaration's recipe.county_map names no path and sha256")
    return c


def dir_digest(path: Path, pattern: str) -> tuple[int, str]:
    files = sorted(Path(path).glob(pattern))
    lines = "".join(f"{f.name} {sha256(f)}\n" for f in files)
    return len(files), hashlib.sha256(lines.encode()).hexdigest()


def check_input_digests(doc: dict, repo: Path = REPO) -> list[dict]:
    recs = doc.get("input_digests")
    if not isinstance(recs, list) or not recs:
        raise EngineeringError("the registered declaration carries no input_digests")
    bad = []
    for r in recs:
        if r.get("kind") not in DIGEST_KINDS:
            raise EngineeringError(f"input digest of unknown kind: {r.get('kind')}")
        n, h = dir_digest(repo / r["path"], r["glob"])
        if (n, h) != (r["n_files"], r["sha256"]):
            bad.append({"path": r["path"], "registered": [r["n_files"], r["sha256"]],
                        "now": [n, h]})
    if bad:
        raise EngineeringError(f"forward input directory changed since registration: {bad}")
    return recs


def _records(attempts: Path) -> list[dict]:
    if not attempts.exists():
        raise EngineeringError(f"no attempt log {attempts}")
    return [json.loads(s) for s in attempts.read_text().splitlines() if s.strip()]


def check_attempts(attempts: Path, files: dict[str, tuple[Path, Path]], gate: Path | None,
                   need_repro: bool = True) -> dict:
    """Every (npz, capture) pair is its arm's last successful forward output; one code for
    all; a passing forcing gate (and, with ``need_repro``, a passing reproduction of this C
    file) is logged at that code."""
    recs = _records(attempts)
    failed = [r for r in recs if r.get("event") == "repro" and int(r.get("rc", -1)) == 3]
    if failed:
        raise EngineeringError(f"a failed reproduction (exit 3) is logged ({len(failed)}x, "
                               f"first {failed[0].get('time')}): nothing more is scored under "
                               f"{PREREG_ID}")
    out, codes = {}, set()
    for arm, (npz, cap) in files.items():
        fw = [r for r in recs if r.get("event") == "forward" and r.get("arm") == arm]
        if not fw:
            raise EngineeringError(f"attempt log has no forward record for {arm}")
        last = fw[-1]
        missing = [str(f) for f in (npz, cap) if not Path(f).exists()]
        if missing:
            raise EngineeringError(f"{arm}: missing output {missing}")
        if int(last.get("rc", -1)) != 0:
            raise EngineeringError(f"last forward attempt of {arm} exited {last.get('rc')}")
        if last.get("npz_sha256") != sha256(npz):
            raise EngineeringError(f"{npz} is not the npz the last {arm} forward wrote")
        if last.get("capture_sha256") != sha256(cap):
            raise EngineeringError(f"{cap} is not the capture the last {arm} forward wrote")
        codes.add(last.get("code"))
        out[arm] = {"attempt": last.get("attempt"), "time": last.get("time"),
                    "code": last.get("code"), "worktree_head": last.get("worktree_head")}
    if len(codes) != 1:
        raise EngineeringError(f"the arms ran different code: {sorted(map(str, codes))}")
    code = next(iter(codes))
    if gate is not None:
        g = [r for r in recs if r.get("event") == "gate" and int(r.get("rc", -1)) == 0
             and r.get("code") == code and gate.exists()
             and r.get("gate_sha256") == sha256(gate)]
        if not g:
            raise EngineeringError("no passing forcing gate is logged for this gate file and "
                                   "code")
        out["gate"] = {"time": g[-1].get("time"), "code": code}
    if need_repro:
        c = sha256(files["C"][0])
        ok = [r for r in recs if r.get("event") == "repro" and int(r.get("rc", -1)) == 0
              and r.get("c_npz_sha256") == c and r.get("code") == code]
        if not ok:
            raise EngineeringError("no passing reproduction (--repro) is logged for this C "
                                   "file and code")
        out["repro"] = {"time": ok[-1].get("time"), "code": code}
    return out


def check_gate_record(gate: Path, engine: dict) -> dict:
    if not gate.exists():
        raise EngineeringError(f"no forcing gate record {gate}")
    g = json.loads(gate.read_text())
    if not g.get("passed"):
        raise EngineeringError("the forcing identity gate did not pass")
    if g.get("years") != years_text(SEQ_STARTS) or int(g.get("months", 0)) != ORIGIN + 1:
        raise EngineeringError(f"the gate replayed {g.get('years')} over {g.get('months')} "
                               "months, not the record 2012-2022")
    if g.get("code_sha256") != engine:
        raise EngineeringError("the gate ran other engine code than the declaration names")
    if not (g.get("pipeline") or {}).get("passed"):
        raise EngineeringError("the gate record carries no passing pipeline layer (the "
                               "patched forward.future_forcing / future_sw path)")
    return {"path": str(gate), "sha256": sha256(gate), "fields": sorted(g["fields"]),
            "pipeline": sorted(k for k in g["pipeline"] if k != "passed")}


# ----------------------------------------------------------------------------- step 1
def _ye_index() -> np.ndarray:
    return np.array([12 * (y - 2012) + 11 for y in YEARS], dtype="int64")


def load_forward(path: Path, arm: str, fields: str, names=SCEN_NAMES, descs=SCEN_DESC,
                 page: bool = False) -> dict:
    """Engineering checks of a forward npz; keeps the subsidence by rheology at every
    December (and, for C and the page run, the heads of the reference scenarios)."""
    spec = arm_spec("C" if page else arm, fields)
    f = np.load(path, allow_pickle=False)
    e = []
    sn = tuple(str(s) for s in f["scenario_names"])
    sd = tuple(str(s) for s in f["scenarios"])
    if sn != tuple(names):
        e.append(f"scenario_names {sn} != {tuple(names)}")
    if sd != tuple(descs):
        e.append(f"scenarios {sd} != {tuple(descs)}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != N_T or dates[0] != "2012-01" or dates[-1] != "2032-12":
        e.append(f"dates {dates[0]}..{dates[-1]} ({len(dates)})")
    if int(f["origin"]) != ORIGIN or int(f["horizon"]) != HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    labels = tuple(str(s) for s in f["member_labels"])
    if labels != spec["members"]:
        e.append(f"member_labels {labels}")
    if int(f["n_members"]) != spec["n_members"]:
        e.append(f"n_members {int(f['n_members'])} != {spec['n_members']}")
    if float(f["gain"]) != 1.0 or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    rh = tuple(str(s) for s in f["rheology_labels"])
    if rh != RHEOLOGY:
        e.append(f"rheology_labels {rh} != {RHEOLOGY}")
    if json.loads(str(f["gate"])).get("git_commit") != THETA_GIT:
        e.append("theta git_commit")
    fo = json.loads(str(f["forward_options"]))
    want = dict(_FO, save_members="yearly") if page else _FO
    bad = {k: fo.get(k) for k, v in want.items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    if not page and "subs_mean_by_rheology" not in f.files:
        e.append("no subs_mean_by_rheology")
    if e:
        raise EngineeringError(f"{path}: " + "; ".join(e))
    out = {"path": str(path), "sha256": sha256(path), "names": sn}
    if not page:
        sbr = np.asarray(f["subs_mean_by_rheology"], dtype="float64")
        if sbr.shape != (len(RHEOLOGY), len(names), N_CELLS, N_T):
            raise EngineeringError(f"{path}: subs_mean_by_rheology shape {sbr.shape}")
        out["subs_ye"] = sbr[..., _ye_index()]
    if page or arm == "C":
        k = [sn.index(s) for s in REF_NAMES]
        out["heads"] = np.asarray(f["heads_mean"][k], dtype="float64")
        out["subs"] = np.asarray(f["subs_mean"][k], dtype="float64")
        if not (np.isfinite(out["heads"]).all() and np.isfinite(out["subs"]).all()):
            raise EngineeringError(f"{path}: non-finite heads or subsidence")
    return out


def load_capture(path: Path, arm: str, fields: str, engine: dict, fwd: dict,
                 county: dict | None = None) -> dict:
    """Engineering checks of a capture; returns its arrays (float64). ``county`` (the
    declaration's recipe.county_map): a capture that carries a county split must name that
    cache file at that sha256; one without a split is kept, its reason reported."""
    spec = arm_spec(arm, fields)
    f = np.load(path, allow_pickle=False)
    e = []
    rec = json.loads(str(f["record"]))
    if rec.get("projection_forcing") != spec["projection_forcing"]:
        e.append(f"projection_forcing {rec.get('projection_forcing')!r}")
    rp_ = rec.get("replay")
    if spec["replay"] is None:
        if rp_ is not None:
            e.append("a replay record on a climatology arm")
    else:
        got = None if rp_ is None else {k: rp_.get(k) for k in spec["replay"]}
        if got != spec["replay"]:
            e.append(f"replay record {got} != {spec['replay']}")
    if rec.get("code_sha256") != engine:
        e.append("engine code sha256 differs from the declaration")
    if tuple(str(s) for s in f["scenario_names"]) != SCEN_NAMES:
        e.append("scenario_names")
    if tuple(str(s) for s in f["rheology_labels"]) != RHEOLOGY:
        e.append(f"rheology_labels {tuple(str(s) for s in f['rheology_labels'])}")
    if [int(y) for y in f["years"]] != list(YEARS) or int(f["origin"]) != ORIGIN:
        e.append("years or origin")
    if [int(y) for y in f["proj_years"]] != list(PROJ_YEARS):
        e.append("proj_years")
    groups = tuple(str(s) for s in f["group_names"])
    if groups not in (GROUPS, GROUPS[:4]):
        e.append(f"group_names {groups}")
    n_runs = int(f["n_runs"])
    if n_runs != spec["n_members"]:
        e.append(f"n_runs {n_runs} != {spec['n_members']}")
    if float(f["total_check_max_abs_m"]) > PARTS_TOL_M:
        e.append(f"parts total vs column {float(f['total_check_max_abs_m']):.2e} m")
    if e:
        raise EngineeringError(f"{path}: " + "; ".join(e))
    el = np.asarray(f["elastic_ye_mean"], dtype="float64")       # (S, R, A, Y)
    inel = np.asarray(f["inelastic_ye_mean"], dtype="float64")
    want = (len(SCEN_NAMES), len(RHEOLOGY), N_CELLS, len(YEARS))
    if el.shape != want or inel.shape != want:
        raise EngineeringError(f"{path}: part shapes {el.shape} {inel.shape}")
    diff = float(np.max(np.abs((el + inel) - np.moveaxis(fwd["subs_ye"], 0, 1))))
    if diff > CAPTURE_TOL_M:
        raise EngineeringError(f"{path}: capture total differs from the forward's subsidence "
                               f"by {diff:.2e} m")
    low = np.asarray(f["newlow_runs"], dtype="int64")             # (S, R, A, Yp)
    ry = np.asarray(f["newlow_run_year"], dtype="uint8")          # (S, R, M, A, Yp)
    if low.shape != want[:3] + (len(PROJ_YEARS),) or \
            ry.shape != want[:2] + (n_runs, N_CELLS, len(PROJ_YEARS)):
        raise EngineeringError(f"{path}: new-low shapes {low.shape} {ry.shape}")
    ch = np.asarray(f["changhua"], dtype="int8")
    prov = rec.get("county") or {"source": None, "error": "no county provenance recorded"}
    if not (ch < 0).any() and county is not None and (
            prov.get("source") != county["path"]
            or prov.get("file_sha256") != county["sha256"]):
        raise EngineeringError(f"{path}: the county split came from {prov.get('source')!r} "
                               f"(sha256 {str(prov.get('file_sha256'))[:8]}), not the declared "
                               f"{county['path']} ({county['sha256'][:8]})")
    return {"county": prov, "path": str(path), "sha256": sha256(path), "record": rec,
            "el": el, "inel": inel,
            "aq": np.asarray(f["aquitard_ye_mean"], dtype="float64"), "low": low, "ry": ry,
            "zone": np.asarray(f["zone_ids"], dtype="int64"), "n_runs": n_runs,
            "changhua": None if (ch < 0).any() else ch.astype(bool), "groups": groups,
            "capture_vs_forward_max_abs_m": diff}


# ----------------------------------------------------------------------------- step 2
def reproduction(c: dict, ref: dict) -> dict:
    out, ok = {}, True
    for i, s in enumerate(REF_NAMES):
        dh = float(np.max(np.abs(c["heads"][i] - ref["heads"][i])))
        ds = float(np.max(np.abs(c["subs"][i] - ref["subs"][i])))
        good = dh <= REF_TOL_HEAD_M and ds <= REF_TOL_SUBS_M
        out[s] = {"heads_max_abs_diff_m": dh, "subs_max_abs_diff_m": ds, "ok": bool(good)}
        ok &= good
    return {"ok": bool(ok), "tol_heads_m": REF_TOL_HEAD_M, "tol_subs_m": REF_TOL_SUBS_M,
            "scenarios": out, "reference": ref["path"], "reference_sha256": ref["sha256"]}


# ----------------------------------------------------------------------------- step 3
def masks(cap: dict, r: int) -> dict[str, np.ndarray]:
    z = cap["zone"][r]
    out = {"fan": np.ones(N_CELLS, dtype=bool)}
    for k, name in enumerate(ZONE_NAMES):
        out[name] = z == k
    if cap["changhua"] is not None:
        out["yunlin_rest"] = ~cap["changhua"]
        out["changhua"] = cap["changhua"]
    return out


def avoided(cap: dict, r: int, scen: str, yi: int, mask: np.ndarray) -> dict:
    """Group-mean avoided subsidence (baseline minus scen) at December index ``yi``, total
    and inelastic (m), and the inelastic share (None when the total is under the floor)."""
    b, s = SCEN_NAMES.index("baseline"), SCEN_NAMES.index(scen)
    tot = cap["el"][:, r] + cap["inel"][:, r]
    d_tot = float((tot[b] - tot[s])[mask, yi].mean())
    d_in = float((cap["inel"][b, r] - cap["inel"][s, r])[mask, yi].mean())
    return {"total_m": d_tot, "inelastic_m": d_in,
            "share": d_in / d_tot if d_tot >= MIN_AVOIDED_M else None}


def crossings(cap: dict, r: int, scen: str, from_year: int | None = None) -> dict:
    """Cells setting a new head minimum after the origin (or from ``from_year``): in at
    least one member run, and in at least half of them."""
    s = SCEN_NAMES.index(scen)
    ry = cap["ry"][s, r]                                           # (M, A, Yp)
    if from_year is not None:
        ry = ry[..., PROJ_YEARS.index(from_year):]
    runs = ry.any(axis=-1).sum(axis=0)                             # (A,)
    return {"cells_any_run": int((runs > 0).sum()),
            "cells_half_runs": int((2 * runs >= cap["n_runs"]).sum()),
            "_half": 2 * runs >= cap["n_runs"]}


def arm_metrics(cap: dict, r: int) -> dict:
    m = masks(cap, r)
    out = {"cut30_dec2032": {g: avoided(cap, r, "cut30", Y_END, k) for g, k in m.items()},
           "retire_aqua_dec2032": {g: avoided(cap, r, "retire_aqua", Y_END, k)
                                   for g, k in m.items()},
           "lift": {}}
    for g, k in m.items():
        lf = {"at_lift_dec2029": avoided(cap, r, "cut30_lift", Y_LIFT, k),
              "three_years_after_dec2032": avoided(cap, r, "cut30_lift", Y_END, k),
              "cut30_kept_dec2032": avoided(cap, r, "cut30", Y_END, k)}
        a, b = lf["at_lift_dec2029"]["total_m"], lf["three_years_after_dec2032"]["total_m"]
        lf["remaining_fraction"] = b / a if a >= MIN_AVOIDED_M else None
        out["lift"][g] = lf
    tot = cap["el"][0, r] + cap["inel"][0, r]
    out["baseline_forward_subsidence_2023_32_cm"] = float(
        (tot[:, Y_END] - tot[:, Y_ORIGIN]).mean() * 100.0)
    cr = {}
    for scen in ("baseline", "cut30"):
        for fy in (None, FROM_YEAR):
            c = crossings(cap, r, scen, fy)
            key = f"{scen}_{'after_origin' if fy is None else f'from_{fy}'}"
            cr[key] = {k: v for k, v in c.items() if not k.startswith("_")}
            cr[key]["cells_half_runs_by_group"] = {g: int((c["_half"] & k).sum())
                                                   for g, k in m.items()}
    hb = crossings(cap, r, "baseline", FROM_YEAR)["_half"]
    hc = crossings(cap, r, "cut30", FROM_YEAR)["_half"]
    cr["avoided_by_cut30_from_2026_half_runs"] = int((hb & ~hc).sum())
    cr["avoided_by_cut30_from_2026_half_runs_by_group"] = {g: int((hb & ~hc & k).sum())
                                                           for g, k in m.items()}
    out["crossings"] = cr
    return out


def _med(xs):
    v = [x for x in xs if x is not None]
    return {"median": float(np.median(v)) if v else None,
            "min": float(min(v)) if v else None, "max": float(max(v)) if v else None,
            "n": len(v)}


def source_year_lows(caps: dict, r: int) -> dict:
    """Report-only: new-minimum cell-months of the baseline in every projected year
    2023-32, summed over the arms' runs, by the source year the projected year replayed."""
    out = {y: 0 for y in SEQ_STARTS}
    for arm, cap in caps.items():
        seq = cyclic(int(arm[-4:]))
        low = cap["low"][SCEN_NAMES.index("baseline"), r].sum(axis=0)       # (Yp,)
        for k, n in enumerate(low.tolist()):
            out[seq[k % len(seq)]] += int(n)
    tot = sum(out.values())
    return {"what": "baseline new-minimum cell-months, 2023-32, all cells, summed over runs",
            "cell_months": {str(y): n for y, n in out.items()},
            "share": {str(y): (n / tot if tot else None) for y, n in out.items()}}


def prevented_lows_by_source_year(caps: dict, r: int) -> dict:
    """The source years in which the cut prevents a crossing: per arm, run, cell and
    projected year from FROM_YEAR (the cut's start), the baseline sets a new head minimum
    and cut30 does not; counted (cell-run-years) by the source year the projected year
    replayed. ``listed``: the years holding >= SOURCE_YEAR_MIN_SHARE of the count, largest
    first (C27's list)."""
    out = {y: 0 for y in SEQ_STARTS}
    k0 = PROJ_YEARS.index(FROM_YEAR)
    b, c = SCEN_NAMES.index("baseline"), SCEN_NAMES.index("cut30")
    for arm, cap in caps.items():
        seq = cyclic(int(arm[-4:]))
        x = cap["ry"][b, r].astype(bool) & ~cap["ry"][c, r].astype(bool)  # (M, A, Yp)
        per = x[..., k0:].sum(axis=(0, 1))
        for j, n in enumerate(per.tolist()):
            out[seq[(k0 + j) % len(seq)]] += int(n)
    tot = sum(out.values())
    share = {y: (n / tot if tot else 0.0) for y, n in out.items()}
    listed = sorted((y for y in SEQ_STARTS if tot and share[y] >= SOURCE_YEAR_MIN_SHARE),
                    key=lambda y: (-out[y], y))
    return {"what": (f"cell-run-years from {FROM_YEAR} in which the baseline sets a new head "
                     "minimum and cut30 does not, by replayed source year"),
            "cell_run_years": {str(y): n for y, n in out.items()},
            "share": {str(y): v for y, v in share.items()}, "total": tot, "listed": listed}


def _pct(x):
    return None if x is None else round(100.0 * x, 1)


def under_text(fields: str) -> str:
    return ("replayed 2012-22 forcing years (weather and detrended pumping)"
            if fields == "energy+weather"
            else "replayed 2012-22 pumping anomalies only (rain at climatology)")


def fill_of(prim: dict) -> dict:
    """The numbers the pre-declared wordings carry, formatted."""
    sh, cm, yrs = prim["share"], prim["inelastic_cm"], prim["source_years_cut_prevents_lows"]
    shares = [v for v in prim["per_sequence_shares"] if v is not None]
    return {"X": f"{_pct(sh['median'])}", "min": f"{_pct(sh['min'])}",
            "max": f"{_pct(sh['max'])}", "n": str(sh["n"]),
            "Y": f"{cm['median']:.1f}", "years": ", ".join(map(str, yrs)),
            "all_zero": bool(shares) and all(abs(v) < ALL_ELASTIC_ABS for v in shares),
            "spread_line": prim["spread_line"]}


def wording(case: str, fill: dict, fields: str) -> dict:
    """The pre-declared ledger wordings (the declaration renders them with placeholders,
    the scorer with the report's numbers). Every text names the replayed mode: in the
    pumping-only mode ('energy') rain stayed at climatology, so no text says weather,
    dry year or that the dry-year caveat is closed."""
    weather = fields == "energy+weather"
    under = under_text(fields)
    kind = "replayed forcing years" if weather else "replayed pumping-anomaly years"
    closed = ("the mean-forcing caveat is closed" if weather else
              "the mean-forcing caveat is closed for pumping anomalies only; a rain-year "
              "replay was not run")
    rng = f"{fill['min']}-{fill['max']} % over {fill['n']} evaluable sequences of 11"
    cite = ("The compaction rings did not show the running-minimum response "
            "(X5.record_low_tf.v1, design-time).")
    if case == "inelastic_present":
        years = (f"source years {fill['years']}, from 2026" if fill["years"] else
                 "no source year in which the cut prevents a new minimum outright: the cut "
                 "makes crossings shallower")
        no_rain = "" if weather else " Rain was held at climatology: no dry year was replayed."
        return {
            "C15": (f"All elastic under mean forcing only; under {under} {fill['X']} % ({rng}) "
                    "of the cut's avoided subsidence by 2032 is inelastic, i.e. permanent "
                    f"(model-only, FW.weather_replay.v1).{no_rain} {cite}"),
            "C25": (f"Untested; in the model, {fill['Y']} cm of lasting benefit by 2032 comes "
                    f"from inelastic compaction the cut avoids in {kind}, under {under} "
                    f"(model form, not a bound; untested; FW.weather_replay.v1). {cite}"),
            "C27": (f"inelastic channel present only where {kind} draw heads to new minima "
                    f"({years}; model-only)"),
            "TWIN_EVIDENCE_s13": (f"run under {under}: {fill['X']} % ({rng}) of the cut's "
                                  "avoided subsidence by 2032 is inelastic "
                                  f"(FW.weather_replay.v1, model-only). {cite}"),
            "spread_line": fill["spread_line"]}
    label = ("All elastic" if fill["all_zero"] else
             f"Mostly elastic (median {fill['X']} %, below the "
             f"{_pct(SHARE_THRESHOLD):.0f} % rule)")
    return {
        "C15": (f"{label} also under {under} (model-only, FW.weather_replay.v1): the median "
                f"inelastic share of the cut's avoided subsidence by 2032 is {fill['X']} % "
                f"({rng}); {closed}."),
        "C25": f"Untested; {label[0].lower() + label[1:]} also under {under} (model-only); "
               f"{closed}.",
        "C27": (f"no inelastic channel under {under} (model-only)" if fill["all_zero"] else
                f"inelastic channel below the {_pct(SHARE_THRESHOLD):.0f} % rule under "
                f"{under} (median {fill['X']} %; model-only)"),
        "TWIN_EVIDENCE_s13": (f"run under {under}; {label[0].lower() + label[1:]} "
                              "(FW.weather_replay.v1)"
                              + ("" if weather else "; a rain-year replay was not run")),
        "spread_line": fill["spread_line"]}


PLACEHOLDER_FILL = {"X": "X", "min": "min", "max": "max", "n": "n", "Y": "Y",
                    "years": "<years>", "all_zero": False,
                    "spread_line": "forcing spread, model-internal, not an interval: fan-mean "
                                   "baseline subsidence 2023-32 ranges a-b cm across the 11 "
                                   "replayed orders of 2012-22 (mean forcing: c cm)"}


def attribution(caps: dict, met: dict, r: int, fields: str) -> dict:
    """Report-only: is the replayed signal weather or pumping? The single-theta arms with
    the same sequences: RD (detrended pumping and, in 'energy+weather' mode, weather), RR
    (raw pumping), RW (weather only, pumping at climatology; 'energy+weather' mode only).
    Shares and prevented-crossing source years side by side."""
    rh = RHEOLOGY[r]
    out = {}
    for name, group in (("RD", RED_DET), ("RR", RED_RAW), ("RW", RED_WEA)):
        if not all(a in caps for a in group):
            out[name] = None
            continue
        out[name] = {
            "share": _med([met[a][rh]["cut30_dec2032"]["fan"]["share"] for a in group]),
            "prevented_lows_by_source_year": prevented_lows_by_source_year(
                {a: caps[a] for a in group}, r)}
    out["note"] = ("RW is not run in the pumping-only mode" if fields != "energy+weather"
                   else "RD - RW contrasts pumping anomalies with weather at one member")
    return out


def score(caps: dict, fields: str) -> dict:
    """The primary statistic, the wording case and the report-only tables."""
    r0, r1 = RHEOLOGY.index(PRIMARY_RHEOLOGY), 1 - RHEOLOGY.index(PRIMARY_RHEOLOGY)
    met = {arm: {RHEOLOGY[r]: arm_metrics(c, r) for r in (r0, r1)} for arm, c in caps.items()}
    prim = {a: met[a][PRIMARY_RHEOLOGY]["cut30_dec2032"]["fan"] for a in PRIMARY}
    shares = [prim[a]["share"] for a in PRIMARY]
    n_eval = sum(s is not None for s in shares)
    sh = _med(shares)
    spread = [met[a][PRIMARY_RHEOLOGY]["baseline_forward_subsidence_2023_32_cm"]
              for a in PRIMARY]
    src = source_year_lows({a: caps[a] for a in PRIMARY}, r0)
    prev = prevented_lows_by_source_year({a: caps[a] for a in PRIMARY}, r0)
    summary = {
        "share": sh,
        "inelastic_cm": {k: (None if v is None else 100.0 * v) for k, v in
                         _med([prim[a]["inelastic_m"] for a in PRIMARY]).items() if k != "n"},
        "avoided_total_cm": {k: (None if v is None else 100.0 * v) for k, v in
                             _med([prim[a]["total_m"] for a in PRIMARY]).items() if k != "n"},
        "n_evaluable": n_eval, "per_sequence_shares": shares,
        "source_years_cut_prevents_lows": prev["listed"],
        "spread_line": (
            "forcing spread, model-internal, not an interval: fan-mean baseline subsidence "
            f"2023-32 ranges {min(spread):.1f}-{max(spread):.1f} cm across the 11 replayed "
            "orders of 2012-22 (mean forcing: "
            f"{met['C'][PRIMARY_RHEOLOGY]['baseline_forward_subsidence_2023_32_cm']:.1f} cm)")}
    if n_eval < MIN_EVALUABLE:
        verdict, case = "not_evaluable", None
    else:
        verdict = "scored"
        case = ("inelastic_present" if sh["median"] >= SHARE_THRESHOLD else "mostly_elastic")
    by = {}
    for g in ("proximal", "mid", "distal", "yunlin_rest", "changhua"):
        if g in met[PRIMARY[0]][PRIMARY_RHEOLOGY]["cut30_dec2032"]:
            by[g] = _med([met[a][PRIMARY_RHEOLOGY]["cut30_dec2032"][g]["share"]
                          for a in PRIMARY])
    rep = {
        "per_sequence_primary": {a: {**prim[a], "sequence": years_text(cyclic(int(a[-4:])))}
                                 for a in PRIMARY},
        "share_by_zone_and_county_median": by,
        "retire_aqua_share": _med([met[a][PRIMARY_RHEOLOGY]["retire_aqua_dec2032"]["fan"]
                                   ["share"] for a in PRIMARY]),
        "tau30_share": _med([met[a]["tau30y"]["cut30_dec2032"]["fan"]["share"]
                             for a in PRIMARY]),
        "lift": {a: met[a][PRIMARY_RHEOLOGY]["lift"] for a in PRIMARY},
        "lift_remaining_fraction_by_group_median": {
            g: _med([met[a][PRIMARY_RHEOLOGY]["lift"][g]["remaining_fraction"]
                     for a in PRIMARY])
            for g in met[PRIMARY[0]][PRIMARY_RHEOLOGY]["lift"]},
        "crossings": {a: met[a][PRIMARY_RHEOLOGY]["crossings"] for a in ("C", *PRIMARY)},
        "prevented_lows_by_source_year": prev,
        "baseline_new_lows_by_source_year": src,
        "climatology_reference": met["C"][PRIMARY_RHEOLOGY],
        "reduced_detrended_share": _med([met[a][PRIMARY_RHEOLOGY]["cut30_dec2032"]["fan"]
                                         ["share"] for a in RED_DET]),
        "reduced_raw_share": _med([met[a][PRIMARY_RHEOLOGY]["cut30_dec2032"]["fan"]["share"]
                                   for a in RED_RAW]),
        "attribution": attribution(caps, met, r0, fields),
        "pumping_annual_anomaly_by_source_year": {
            "what": ("Y_c(y)/L_c(y) per policy class: the annual energy over its fitted "
                     "2012-22 line, i.e. the pumping anomaly a detrended replay of year y "
                     "carries; a step (e.g. the census frozen in mid-2016) shows as a run on "
                     "one side of 1"),
            "values": (caps[PRIMARY[0]]["record"].get("replay") or {}).get(
                "annual_anomaly_Y_over_L")},
        "county_split": {a: caps[a]["county"] for a in ("C", PRIMARY[0])},
        "forcing_spread_baseline_2023_32_cm": dict(zip(PRIMARY, spread, strict=True)),
        "sampling_note": (
            "the 11 sequences are cyclic orders of the same 11 source years, so their range "
            "is a forcing-order spread of one model, not 11 independent draws and not an "
            "interval; each sequence projects 10 of the 11 years")}
    out = {"verdict": verdict, "wording_case": case, "replay_fields": fields,
           "threshold": SHARE_THRESHOLD, "min_avoided_m": MIN_AVOIDED_M,
           "min_evaluable": MIN_EVALUABLE, "summary": summary, "reported_not_gating": rep,
           "metrics": met}
    out["wording"] = wording(case, fill_of(summary), fields) if case else {
        "ledger_note": f"FW.weather_replay.v1 not evaluable ({n_eval} of 11 sequences with a "
                       f"fan-mean avoided subsidence >= {MIN_AVOIDED_M * 100:.1f} cm); "
                       "nothing else changes"}
    return out


# ----------------------------------------------------------------------------- main
def rp(s: str) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def files_of(arm: str, root: str | None = None) -> tuple[Path, Path]:
    stem = OUT_STEM.format(arm=arm) if root is None else f"{root}/{arm}"
    return rp(stem + ".npz"), rp(stem + ".fw.npz")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", default=None, help="directory of the arms' files (default "
                    "results/twin_forward/fw_replay)")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--gate", default=GATE_JSON)
    ap.add_argument("--attempts", default=DEFAULT_ATTEMPTS)
    ap.add_argument("--out", default=None, help="report JSON (scoring mode; required there)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--repro", action="store_true", help="steps 0-2 on C alone")
    mode.add_argument("--verify-registration", action="store_true",
                      help="step 0's registration part only; reads no output")
    a = ap.parse_args(argv)
    files: dict = {}
    try:
        if not (a.verify_registration or a.repro) and not a.out:
            raise EngineeringError("scoring mode needs --out")
        if a.out and rp(a.out).exists():
            raise EngineeringError(f"{a.out} exists: the declaration is scored once")
        doc, decl_sha = load_registered()
        frozen = check_frozen(doc)
        fields = declared_fields(doc)
        engine = declared_engine(doc)
        county = declared_county(doc)
        digests = check_input_digests(doc)
        files = {arm: files_of(arm, a.outputs) for arm in arms(fields)}
        reg = {"declaration_sha256": decl_sha, "frozen_code_sha256": frozen,
               "input_digests": digests, "replay_fields": fields, "engine_code": engine,
               "county_map": county, "arms": list(files)}
        print(f"step 0 ok: {PREREG_ID} registered (sha256 {decl_sha[:16]}), frozen code and "
              "declared inputs match")
        if a.verify_registration:
            return 0
        reg["gate"] = check_gate_record(rp(a.gate), engine)
        if a.repro:
            reg["attempts"] = check_attempts(rp(a.attempts), {"C": files["C"]}, rp(a.gate),
                                             need_repro=False)
        else:
            reg["attempts"] = check_attempts(rp(a.attempts), files, rp(a.gate))
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 0): {err}", file=sys.stderr)
        return 1
    try:
        ref = load_forward(rp(a.ref), "page", fields, REF_NAMES, SCEN_DESC[:3], page=True)
        if ref["sha256"] != REF_SHA256:
            raise EngineeringError(f"{a.ref} is not the registered page run")
        c = load_forward(files["C"][0], "C", fields)
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    rep = reproduction(c, ref)
    print(f"step 1 ok (C, page run); step 2 reproduction {'ok' if rep['ok'] else 'FAIL'}")
    for s, r in rep["scenarios"].items():
        print(f"  {s}: max |C - page| heads {r['heads_max_abs_diff_m']:.2e} m, subsidence "
              f"{r['subs_max_abs_diff_m']:.2e} m")
    if not rep["ok"]:
        print("ENGINEERING FAIL (step 2): C does not reproduce the page run; no other arm "
              "read", file=sys.stderr)
        return 3
    if a.repro:
        print(f"repro: ok (C sha256 {c['sha256']})")
        return 0
    try:
        fw = {"C": c, **{arm: load_forward(files[arm][0], arm, fields)
                         for arm in files if arm != "C"}}
        caps = {arm: load_capture(files[arm][1], arm, fields, engine, fw[arm], county)
                for arm in files}
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    res = score(caps, fields)
    report = {"prereg_id": PREREG_ID, "scorer_version": SCORER_VERSION, "registration": reg,
              "reproduction": rep,
              "files": {arm: {"npz": {"path": fw[arm]["path"], "sha256": fw[arm]["sha256"]},
                              "capture": {"path": caps[arm]["path"],
                                          "sha256": caps[arm]["sha256"],
                                          "vs_forward_max_abs_m":
                                              caps[arm]["capture_vs_forward_max_abs_m"]}}
                        for arm in files}, **res}
    s = res["summary"]["share"]
    print(f"VERDICT: {res['verdict']} (wording case {res['wording_case']}; median inelastic "
          f"share {s['median']}, {res['summary']['n_evaluable']} of 11 evaluable)")
    p = rp(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:                      # never overwrite a scored report
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
