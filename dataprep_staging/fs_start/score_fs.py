"""Score FS.start_invariance.v1: does the page's head lever (policy minus baseline, mean over
N2's 84 Yunlin wells, each in its own screened layer) depend on the forecast's 2022-12 start
state?

Model-only. In scoring mode it reads the forward runs the declaration names
(results/twin_forward/fs_start_{S0,S1,S2,S3,S3ref}.npz and fs_start_S3.members.npz), the
page's registered forward run it must reproduce (results/twin_forward/datum_gate_innov.npz),
the registered declaration and the runner's attempt log. The well screens are frozen below
(cell, layer, county; no station id). No observation of any family is read.

Arms (the declaration's recipe; every arm runs the page theta stage3_datum_gate with
--column-heads free, so the compaction column is driven by the free-running heads it was
fitted on and is not part of this test):
  S0     rerun of the page recipe (18 parameter sets x 2 initial fields, innovation restart);
  S1     the same with --gain 0 (free-run start: the model's own Dec-2022 state)    GATING;
  S2     the same with --update-method field and no taper (the old untapered IDW restart
         that put the -21/-25 m step into the proximal layers)                       GATING;
  S3     the page theta alone with --ic-members 8 --ic-sigma 2.0                     report-only;
  S3ref  the page theta alone with --ic-members 0 (S3's unperturbed member)          report-only.

Order of work (the declaration's attempt policy):
  0. registration: the declaration is registered and intact (prereg3.load_prereg, which also
     re-hashes every declared single-file input, the forward's inputs included) and the
     sha256 of this scorer and the runner equal its ``frozen_code``; the declared directory
     digest of the head-well files (``input_digests``) still matches; every npz is the output
     of its arm's last successful forward attempt in attempts.jsonl, all arms ran the same
     code, and a passing reproduction (``--repro``) is logged for this S0 file;
  1. engineering checks on every file (scenario strings, dates, shapes, members, gain,
     forward options, theta commit, rheology axis);
  2. reproduction: S0's baseline, cut30 and retire_aqua heads at the 158 screens match the
     page run (max |diff| <= REF_TOL_M) and the N2-unit Dec-2032 lever matches the page run
     (|diff| <= REF_TOL_M), whose own values match the published +1.19 / +1.41 m within half
     the published rounding unit. Only page-published quantities are printed;
  3. only if 0-2 pass: the gated ratios and p90s, the verdict, the report-only tables. The
     report is written once (open mode 'x').
A failure in 0-2 is an engineering failure (exit 1, 2 or 3): no arm other than S0 is read.

``--repro`` runs steps 0-2 on S0 alone (after the S0 forward, before any other arm runs).
``--check-reference`` reads only the page run and prints its N2-unit Dec-2032 levers against
the published values (a plumbing check usable before registration; no arm is read).
``--verify-registration`` runs step 0's registration part only (the runner calls it before
every forward, so a drifted input refuses before any GPU time is spent).
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
X = "dataprep_staging/fs_start"

PREREG_ID = "FS.start_invariance.v1"
SCORER_VERSION = "1.1.0"
FROZEN = (f"{X}/score_fs.py", f"{X}/run_fs.sh")

ARMS = ("S0", "S1", "S2", "S3", "S3ref")
GATING = ("S1", "S2")
OUT_STEM = "results/twin_forward/fs_start_{arm}"
DEFAULT_ATTEMPTS = "results/twin_runs/fs_start/attempts.jsonl"
REF = "results/twin_forward/datum_gate_innov.npz"
REF_SHA256 = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
# the head-well directory the forward reads (heads.py globs <dir>/*.parquet): its digest is
# sha256 over the sorted lines "<file name> <file sha256>\n" (bytes hashed, never parsed)
DIGEST_KINDS = ("sorted_name_sha256_lines",)

SCEN_NAMES = ("baseline", "cut30", "retire_aqua", "cut30_2030", "retire_aqua_2030")
SCEN_DESC = (
    "baseline: baseline (no change)",
    "cut30: irrigation x0.7 from 2026-01",
    "retire_aqua: aquaculture x0 from 2026-01",
    "cut30_2030: irrigation x0.7 from 2030-01",
    "retire_aqua_2030: aquaculture x0 from 2030-01",
)
REF_NAMES, REF_DESC = SCEN_NAMES[:3], SCEN_DESC[:3]
PAGE_MEMBERS = ("stage3_datum_gate", "fold0", "fold1", "fold2", "fold3", "fold4",
                *(f"foldpost{i}" for i in range(12)))
ONE_MEMBER = ("stage3_datum_gate",)
RHEOLOGY = ("coupled_leveling", "tau30y")
THETA_GIT = "fa5cca2"
_PAGE_FO = {"column_heads": "free", "update_method": "innovation", "apex_hold": "calibrated",
            "restart_taper_km": 5.0, "column_hpc0_fast_days": 365.0}
# per arm: parameter sets, members (sets x initial fields), gain, forward options
SPEC = {
    "S0": {"members": PAGE_MEMBERS, "n_members": 36, "gain": 1.0,
           "fo": {**_PAGE_FO, "save_members": "none"}},
    "S1": {"members": PAGE_MEMBERS, "n_members": 36, "gain": 0.0,
           "fo": {**_PAGE_FO, "save_members": "none"}},
    "S2": {"members": PAGE_MEMBERS, "n_members": 36, "gain": 1.0,
           "fo": {**_PAGE_FO, "update_method": "field", "restart_taper_km": None,
                  "save_members": "none"}},
    "S3": {"members": ONE_MEMBER, "n_members": 9, "gain": 1.0,
           "fo": {**_PAGE_FO, "save_members": "yearly"}},
    "S3ref": {"members": ONE_MEMBER, "n_members": 1, "gain": 1.0,
              "fo": {**_PAGE_FO, "save_members": "none"}},
    "page": {"members": PAGE_MEMBERS, "n_members": 36, "gain": 1.0,
             "fo": {**_PAGE_FO, "save_members": "yearly"}},
}

N_T, ORIGIN, HORIZON, N_CELLS = 252, 131, 120, 2148     # 2012-01 .. 2032-12; origin 2022-12
I_2023_01, I_DEC2026, I_DEC2032 = 132, 179, 251
WINDOWS = {"2026-27": (168, 192), "2030-32": (216, 252), "Dec-2032": (251, 252)}
WINDOW_FIRST = {"2026-27": "2026-01", "2030-32": "2030-01", "Dec-2032": "2032-12"}
WINDOW_YEARS = {"2026-27": (2026, 2027), "2030-32": (2030, 2031, 2032), "Dec-2032": (2032,)}
LEVERS = ("cut30", "retire_aqua")
STARTS = (2026, 2030)
# gated units: (lever, start, window); a 2030 start has no lever in 2026-27
UNITS = tuple((lv, s, w) for lv in LEVERS for s in STARTS for w in WINDOWS
              if not (s == 2030 and w == "2026-27"))

BAND = (0.95, 1.05)                 # co-primary 1, gating, inclusive, unrounded
REPORT_BAND = (0.9, 1.1)            # pre-declared, report-only
P90_MAX = 0.10                      # co-primary 2, gating, inclusive
LEVER_FLOOR_M = 0.05                # per-well denominator floor of co-primary 2 (m)
MIN_LEVER_M = 0.05                  # S0 N2-unit mean lever below this: not evaluable
REF_TOL_M = 0.01                    # S0 vs the page run (m)
PUBLISHED_DEC2032 = {"cut30": 1.19, "retire_aqua": 1.41}   # page/ledger C01, 2 decimals
PUBLISHED_TOL_M = 0.005             # half the published rounding unit

# --- frozen well screens: N2's 158 registered stations (84 Yunlin, 74 Changhua) as cells and
# layers (0-based) of the twin grid; the same rows as dataprep_staging/x5_class_share
# (wells_frozen.json, station ids dropped here); SCREENS_SHA256 below ---
CELLS = (
    2018, 2018, 1994, 1994, 1994, 2079, 2079, 2107, 2107, 2107, 2107, 2139, 2139, 2139, 1896,
    1896, 1896, 1896, 1954, 1934, 1954, 1934, 1954, 1954, 1749, 1749, 1749, 1723, 1723, 1740,
    1740, 1740, 1401, 1401, 1900, 1900, 1900, 1900, 1535, 1535, 1671, 1671, 1671, 1449, 1449,
    1545, 1545, 1529, 1529, 1522, 1522, 1522, 1893, 1893, 1633, 1633, 1633, 1305, 1305, 1407,
    1407, 1407, 1407, 1421, 1244, 1244, 1393, 1393, 1393, 1354, 1287, 1287, 1199, 1250, 686,
    686, 686, 686, 694, 694, 694, 378, 378, 814, 814, 632, 1133, 1133, 864, 864, 864, 537, 537,
    361, 541, 541, 541, 112, 112, 133, 514, 514, 514, 340, 329, 329, 329, 280, 280, 871, 871,
    1035, 1038, 990, 990, 1125, 1125, 1128, 967, 1064, 1056, 1056, 902, 902, 902, 895, 616,
    616, 653, 750, 750, 178, 178, 357, 357, 358, 201, 235, 235, 235, 255, 165, 165, 346, 346,
    346, 346, 353, 353, 353, 353, 353, 37, 37, 37, 37, 69, 8,
)
LAYERS = (
    1, 2, 0, 1, 2, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 1, 2, 0, 0, 1, 1, 1, 3, 0, 1, 1, 1, 1, 1,
    1, 2, 0, 2, 1, 1, 2, 3, 1, 1, 1, 1, 2, 0, 2, 1, 1, 0, 2, 0, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1,
    1, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 1, 0, 3, 2, 0, 1, 0, 1,
    2, 0, 2, 1, 1, 1, 2, 1, 1, 3, 0, 1, 1, 0, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1, 1, 0,
    1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 1, 2, 3, 1, 1, 3, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1,
    2, 3, 1, 1, 2, 3, 1, 1,
)
COUNTY = (
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCYYYYYY"
    "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
)
SCREENS_SHA256 = "d66d3daaaacff49d1cedff68a47af82394d20046bf124c53049c3958ea693d1e"


def _screens_sha() -> str:
    rows = [{"cell": c, "layer": k, "county": "changhua" if q == "C" else "yunlin"}
            for c, k, q in zip(CELLS, LAYERS, "".join(COUNTY), strict=True)]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def masks() -> dict[str, np.ndarray]:
    c = np.array(list("".join(COUNTY))) == "C"
    return {"yunlin": ~c, "changhua": c}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scen_name(lever: str, start: int) -> str:
    return lever if start == 2026 else f"{lever}_{start}"


class EngineeringError(RuntimeError):
    """A file or the registration is not what the declaration names; nothing is scored."""


# ----------------------------------------------------------------------------- step 0
def load_registered(prereg_dir: Path | None = None, rows_dir: Path | None = None
                    ) -> tuple[dict, str]:
    """The registered declaration, verified against INDEX.jsonl and the ledger."""
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
    """sha256 of the scorer and the runner against the declaration."""
    want = doc.get("frozen_code") or {}
    got = {p: sha256(repo / p) if (repo / p).exists() else "missing" for p in FROZEN}
    bad = {p: {"registered": want.get(p), "now": got[p]} for p in FROZEN
           if want.get(p) != got[p]}
    if bad:
        raise EngineeringError(f"frozen code differs from the registered declaration: {bad}")
    return got


def dir_digest(path: Path, pattern: str) -> tuple[int, str]:
    """(file count, sha256 over the sorted lines '<name> <sha256>\\n') of ``path/pattern``."""
    files = sorted(Path(path).glob(pattern))
    lines = "".join(f"{f.name} {sha256(f)}\n" for f in files)
    return len(files), hashlib.sha256(lines.encode()).hexdigest()


def check_input_digests(doc: dict, repo: Path = REPO) -> list[dict]:
    """The declaration's directory digests (the head-well files) against the disk now."""
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


def check_attempts(attempts: Path, files: dict[str, Path], members: Path | None = None,
                   need_repro: bool = True) -> dict:
    """Every npz is the output of its arm's last forward attempt, which exited 0; all arms
    ran the same code; with ``need_repro`` a passing reproduction is logged for this S0
    file at that code."""
    recs = _records(attempts)
    out, codes = {}, set()
    for arm, p in files.items():
        fw = [r for r in recs if r.get("event") == "forward" and r.get("arm") == arm]
        if not fw:
            raise EngineeringError(f"attempt log has no forward record for {arm}")
        last = fw[-1]
        if int(last.get("rc", -1)) != 0:
            raise EngineeringError(f"last forward attempt of {arm} exited {last.get('rc')}")
        if last.get("npz_sha256") != sha256(p):
            raise EngineeringError(f"{p} is not the npz the last {arm} forward wrote")
        if arm == "S3" and members is not None and \
                last.get("members_npz_sha256") != sha256(members):
            raise EngineeringError(f"{members} is not the members file the last S3 forward "
                                   "wrote")
        codes.add(last.get("code"))
        out[arm] = {"attempt": last.get("attempt"), "time": last.get("time"),
                    "code": last.get("code"), "worktree_head": last.get("worktree_head")}
    if len(codes) != 1:
        raise EngineeringError(f"the arms ran different code: {sorted(map(str, codes))}")
    if need_repro:
        s0 = sha256(files["S0"])
        ok = [r for r in recs if r.get("event") == "repro" and int(r.get("rc", -1)) == 0
              and r.get("s0_npz_sha256") == s0 and r.get("code") in codes]
        if not ok:
            raise EngineeringError("no passing reproduction (--repro) is logged for this S0 "
                                   "file and code")
        out["repro"] = {"time": ok[-1].get("time"), "code": ok[-1].get("code")}
    return out


# ----------------------------------------------------------------------------- step 1
def _load(path: Path, arm: str, names: tuple[str, ...] = SCEN_NAMES,
          descs: tuple[str, ...] = SCEN_DESC) -> dict:
    """Engineering checks, then the heads at the screens and the fan layer-2 mean."""
    spec = SPEC[arm]
    f = np.load(path, allow_pickle=False)
    e = []
    sn = tuple(str(s) for s in f["scenario_names"])
    sd = tuple(str(s) for s in f["scenarios"])
    if sn != names:
        e.append(f"scenario_names {sn} != {names}")
    if sd != descs:
        e.append(f"scenarios {sd} != {descs}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != N_T or dates[0] != "2012-01" or dates[-1] != "2032-12":
        e.append(f"dates {dates[0]}..{dates[-1]} ({len(dates)}), expected 2012-01..2032-12")
    else:
        for w, (a, _) in WINDOWS.items():
            if dates[a] != WINDOW_FIRST[w]:
                e.append(f"window {w} starts {dates[a]}")
        if dates[I_2023_01] != "2023-01" or dates[I_DEC2026] != "2026-12":
            e.append("2023-01 / 2026-12 indices moved")
    if int(f["origin"]) != ORIGIN or int(f["horizon"]) != HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    hm = f["heads_mean"]                       # read once (compressed in forward outputs)
    shape = hm.shape
    if shape != (len(names), 4, N_CELLS, N_T):
        e.append(f"heads_mean shape {shape}")
    labels = tuple(str(s) for s in f["member_labels"])
    if labels != spec["members"]:
        e.append(f"member_labels {labels} != {spec['members']}")
    if int(f["n_members"]) != spec["n_members"]:
        e.append(f"n_members {int(f['n_members'])} != {spec['n_members']}")
    if float(f["gain"]) != spec["gain"] or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    rh = tuple(str(s) for s in f["rheology_labels"])
    if rh != RHEOLOGY:
        e.append(f"rheology_labels {rh} != {RHEOLOGY}")
    gate = json.loads(str(f["gate"]))
    if gate.get("git_commit") != THETA_GIT:
        e.append(f"theta git_commit {gate.get('git_commit')}")
    fo = json.loads(str(f["forward_options"]))
    bad = {k: fo.get(k) for k, v in spec["fo"].items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    if e:
        raise EngineeringError(f"{path}: " + "; ".join(e))
    cells, layers = np.asarray(CELLS), np.asarray(LAYERS)
    h = np.asarray(hm[:, layers, cells, :], dtype=np.float64)       # (scenario, screen, month)
    fan = np.asarray(hm[:, 1, :, :], dtype=np.float64).mean(axis=1)   # (scenario, month)
    if not (np.isfinite(h).all() and np.isfinite(fan).all()):
        raise EngineeringError(f"{path}: non-finite heads at the frozen screens or layer 2")
    return {"path": str(path), "sha256": sha256(path), "names": sn, "h": h, "fanL2": fan}


def _load_members(path: Path) -> dict:
    """S3's yearly members (layer-2 December heads, quantised), checked."""
    f = np.load(path, allow_pickle=False)
    e = []
    names = tuple(str(s) for s in f["scenario_names"])
    if names != SCEN_NAMES:
        e.append(f"members scenario_names {names}")
    hm = tuple(str(s) for s in f["head_member"])
    ic = [int(i) for i in f["head_ic"]]
    if hm != ONE_MEMBER * 9 or ic != list(range(9)):
        e.append(f"members head_member/head_ic {hm}/{ic}")
    years = [int(y) for y in f["years"]]
    if years != list(range(2012, 2033)):
        e.append(f"members years {years[0]}..{years[-1]}")
    q = f["headL2_members_yr"]
    if q.shape != (len(SCEN_NAMES), 9, N_CELLS, 21):
        e.append(f"headL2_members_yr shape {q.shape}")
    if e:
        raise EngineeringError(f"{path}: " + "; ".join(e))
    return {"path": str(path), "sha256": sha256(path), "years": years,
            "q": np.asarray(q, dtype=np.float64) * float(f["head_scale_m"]),
            "scale_m": float(f["head_scale_m"])}


def _lever(d: dict, scen: str) -> np.ndarray:
    return d["h"][d["names"].index(scen)] - d["h"][d["names"].index("baseline")]


def n2_unit_dec2032(d: dict, scen: str) -> float:
    """The page's C01 number: mean over N2's 84 Yunlin wells of (scen - baseline) at Dec 2032,
    each well in its own layer."""
    return float(_lever(d, scen)[masks()["yunlin"], I_DEC2032].mean())


# ----------------------------------------------------------------------------- step 2
def reference_values(ref: dict) -> dict:
    out, ok = {}, True
    for s, pub in PUBLISHED_DEC2032.items():
        v = n2_unit_dec2032(ref, s)
        good = abs(v - pub) <= PUBLISHED_TOL_M
        out[s] = {"page_run_m": v, "published_m": pub, "ok": bool(good)}
        ok &= good
    return {"ok": bool(ok), "tol_m": PUBLISHED_TOL_M, "entries": out}


def reproduction(s0: dict, ref: dict) -> dict:
    """S0 against the page run: heads at the screens (baseline, cut30, retire_aqua; every
    month) and the N2-unit Dec-2032 lever; the page run against the published values."""
    vs, ok = {}, True
    for s in REF_NAMES:
        diff = float(np.max(np.abs(s0["h"][s0["names"].index(s)]
                                   - ref["h"][ref["names"].index(s)])))
        vs[s] = {"max_abs_diff_m": diff, "ok": diff <= REF_TOL_M}
        ok &= diff <= REF_TOL_M
    unit = {}
    for s in PUBLISHED_DEC2032:
        a, b = n2_unit_dec2032(s0, s), n2_unit_dec2032(ref, s)
        unit[s] = {"S0_m": a, "page_run_m": b, "abs_diff_m": abs(a - b),
                   "ok": abs(a - b) <= REF_TOL_M}
        ok &= abs(a - b) <= REF_TOL_M
    pub = reference_values(ref)
    ok &= pub["ok"]
    return {"ok": bool(ok), "ref_tol_m": REF_TOL_M, "screens": vs, "n2_unit_dec2032": unit,
            "page_run_vs_published": pub, "reference": ref["path"],
            "reference_sha256": ref["sha256"]}


# ----------------------------------------------------------------------------- step 3
def unit_stats(arm: dict, base: dict, mask: np.ndarray, scen: str, a: int, b: int) -> dict:
    """Co-primary 1 (ratio of the mean lever over the wells) and co-primary 2 (p90 over the
    wells of |arm lever - base lever| / max(|base lever|, floor)) for one window."""
    la = _lever(arm, scen)[mask, a:b].mean(axis=1)
    lb = _lever(base, scen)[mask, a:b].mean(axis=1)
    num, den = float(la.mean()), float(lb.mean())
    rel = np.abs(la - lb) / np.maximum(np.abs(lb), LEVER_FLOOR_M)
    return {"arm_mean_lever_m": num, "base_mean_lever_m": den,
            "ratio": num / den if den != 0 else float("nan"),
            "p90_rel_dev": float(np.percentile(rel, 90)),
            "n_wells": int(mask.sum()),
            "n_wells_below_floor": int((np.abs(lb) < LEVER_FLOOR_M).sum())}


def _key(lv: str, s: int, w: str) -> str:
    return f"{lv}@{s}/{w}"


def gated(arms: dict) -> dict:
    """The gating table: S1 and S2 against S0 at N2's 84 Yunlin wells, every unit."""
    yun = masks()["yunlin"]
    tab, evaluable, inside = {}, True, True
    for arm in GATING:
        tab[arm] = {}
        for lv, s, w in UNITS:
            a, b = WINDOWS[w]
            r = unit_stats(arms[arm], arms["S0"], yun, scen_name(lv, s), a, b)
            r["evaluable"] = bool(r["base_mean_lever_m"] >= MIN_LEVER_M
                                  and np.isfinite(r["ratio"]))
            r["ratio_in_band"] = bool(r["evaluable"] and BAND[0] <= r["ratio"] <= BAND[1])
            r["p90_ok"] = bool(r["evaluable"] and r["p90_rel_dev"] <= P90_MAX)
            r["pass"] = r["ratio_in_band"] and r["p90_ok"]
            r["ratio_in_report_band"] = bool(r["evaluable"]
                                             and REPORT_BAND[0] <= r["ratio"] <= REPORT_BAND[1])
            evaluable &= r["evaluable"]
            inside &= r["pass"]
            tab[arm][_key(lv, s, w)] = r
    verdict = "not_evaluable" if not evaluable else ("pass" if inside else "fail")
    rows = [r for t in tab.values() for r in t.values() if r["evaluable"]]
    summary = {
        "max_abs_ratio_dev_pct": (100.0 * max(abs(r["ratio"] - 1.0) for r in rows)
                                  if rows else None),
        "max_p90_rel_dev": max(r["p90_rel_dev"] for r in rows) if rows else None,
        "failing": sorted(f"{arm}:{k}" for arm, t in tab.items() for k, r in t.items()
                          if not r["pass"]),
        "arms_failing": sorted({arm for arm, t in tab.items()
                                if not all(r["pass"] for r in t.values())}),
    }
    # per arm, for the pre-declared split wording (S1 is the page's own start family; S2 is
    # the superseded untapered field restart)
    per_arm = {}
    for arm, t in tab.items():
        ev = [r for r in t.values() if r["evaluable"]]
        per_arm[arm] = {
            "pass": all(r["pass"] for r in t.values()),
            "max_abs_ratio_dev_pct": (100.0 * max(abs(r["ratio"] - 1.0) for r in ev)
                                      if ev else None),
            "max_p90_rel_dev": max(r["p90_rel_dev"] for r in ev) if ev else None,
            "ratio_band_fails": sum(not r["ratio_in_band"] for r in t.values()),
            "p90_fails": sum(not r["p90_ok"] for r in t.values())}
    summary["per_arm"] = per_arm
    if verdict == "not_evaluable":
        case = "not_evaluable"
    elif verdict == "pass":
        case = "pass"
    elif per_arm["S1"]["pass"]:
        case = "S2_only"            # S1 passes, S2 fails: the superseded restart only
    elif per_arm["S2"]["pass"]:
        case = "S1_only"
    else:
        case = "S1_and_S2"
    summary["wording_case"] = case
    return {"verdict": verdict, "table": tab, "summary": summary}


def _members_table(mem: dict, base_ratio_band=BAND) -> dict:
    """S3 per initial field: layer-2 December lever at the Yunlin screens' cells (each cell
    once per screen, layer 2 whatever the screen layer), member k over member 0."""
    cells = np.asarray(CELLS)[masks()["yunlin"]]
    q, years = mem["q"], mem["years"]
    out = {}
    for lv, s, w in UNITS:
        yi = [years.index(y) for y in WINDOW_YEARS[w]]
        si = SCEN_NAMES.index(scen_name(lv, s))
        lev = (q[si][:, cells][:, :, yi] - q[0][:, cells][:, :, yi]).mean(axis=(1, 2))   # (9,)
        r = lev[1:] / lev[0] if lev[0] != 0 else np.full(8, np.nan)
        out[_key(lv, s, w)] = {
            "ic0_lever_m": float(lev[0]), "ratios_ic1_8": [float(x) for x in r],
            "share_in_band": float(np.mean((r >= base_ratio_band[0])
                                           & (r <= base_ratio_band[1])))}
    return {"note": f"layer-2 December heads quantised to {mem['scale_m']:g} m by the forward "
                    "sidecar; a ~1 m lever carries about +-1 % quantisation error",
            "units": out}


def score(arms: dict, s3_members: dict | None = None) -> dict:
    res = gated(arms)
    yun, cha = masks()["yunlin"], masks()["changhua"]
    lay = np.asarray(LAYERS)
    rep: dict = {"report_band": list(REPORT_BAND)}
    # Changhua's 74 wells and the fan layer-2 mean, same arms and units
    rep["changhua_74"] = {arm: {_key(lv, s, w): unit_stats(arms[arm], arms["S0"], cha,
                                                          scen_name(lv, s), *WINDOWS[w])
                                for lv, s, w in UNITS} for arm in GATING}

    def fan_ratio(arm, sc, a, b):
        la = arms[arm]["fanL2"][arms[arm]["names"].index(sc)] - arms[arm]["fanL2"][0]
        lb = arms["S0"]["fanL2"][arms["S0"]["names"].index(sc)] - arms["S0"]["fanL2"][0]
        num, den = float(la[a:b].mean()), float(lb[a:b].mean())
        return {"arm_m": num, "S0_m": den, "ratio": num / den if den else None}

    rep["fan_layer2_mean"] = {arm: {_key(lv, s, w): fan_ratio(arm, scen_name(lv, s),
                                                              *WINDOWS[w])
                                    for lv, s, w in UNITS} for arm in GATING}
    rep["by_layer_yunlin"] = {
        arm: {_key(lv, s, w): {f"L{k + 1}": (unit_stats(arms[arm], arms["S0"],
                                                         yun & (lay == k), scen_name(lv, s),
                                                         *WINDOWS[w])["ratio"]
                                              if (yun & (lay == k)).any() else None)
                               for k in range(4)}
              for lv, s, w in UNITS} for arm in GATING}
    # the page tile's series: N2-unit lever at every December 2026..2032, every arm
    decs = list(range(I_DEC2026, N_T, 12))
    rep["n2_unit_year_end_m"] = {
        arm: {sc: [float(_lever(d, sc)[yun, i].mean()) for i in decs]
              for sc in SCEN_NAMES[1:]} for arm, d in arms.items() if arm in ("S0", *GATING)}
    rep["n2_unit_year_end_years"] = list(range(2026, 2033))
    # the start moves the level: baseline head, arm minus S0, N2 unit and fan layer 2
    rep["baseline_level_shift_m"] = {
        arm: {lab: {"n2_unit": float((arms[arm]["h"][0][yun, i]
                                      - arms["S0"]["h"][0][yun, i]).mean()),
                    "fan_layer2": float(arms[arm]["fanL2"][0][i] - arms["S0"]["fanL2"][0][i])}
              for lab, i in (("2023-01", I_2023_01), ("2026-12", I_DEC2026),
                             ("2032-12", I_DEC2032))}
        for arm in GATING}
    # S3 (report-only): the IC ensemble mean against its unperturbed member, then per member
    if "S3" in arms and "S3ref" in arms:
        rep["S3_ensemble_mean_vs_S3ref"] = {
            _key(lv, s, w): unit_stats(arms["S3"], arms["S3ref"], yun, scen_name(lv, s),
                                       *WINDOWS[w]) for lv, s, w in UNITS}
    if s3_members is not None:
        rep["S3_per_member_layer2"] = _members_table(s3_members)
    return {**res, "band": list(BAND), "p90_max": P90_MAX, "lever_floor_m": LEVER_FLOOR_M,
            "min_lever_m": MIN_LEVER_M, "reported_not_gating": rep}


# ----------------------------------------------------------------------------- main
def rp(s: str) -> Path:
    """A path argument, relative to the repository root unless absolute."""
    return Path(s) if Path(s).is_absolute() else REPO / s


def _static_checks() -> None:
    if _screens_sha() != SCREENS_SHA256:
        raise SystemExit("frozen screen list was edited (screens sha mismatch)")
    if len(CELLS) != 158 or "".join(COUNTY).count("C") != 74:
        raise SystemExit("frozen screen list is not 158 screens / 74 Changhua")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for arm in ARMS:
        ap.add_argument(f"--{arm.lower()}", default=OUT_STEM.format(arm=arm) + ".npz")
    ap.add_argument("--s3-members", default=OUT_STEM.format(arm="S3") + ".members.npz")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--attempts", default=DEFAULT_ATTEMPTS)
    ap.add_argument("--out", default=None, help="report JSON (scoring mode; required there)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--repro", action="store_true",
                      help="steps 0-2 on S0 alone (registration, checks, reproduction)")
    mode.add_argument("--check-reference", action="store_true",
                      help="the page run's N2-unit Dec-2032 levers against the published "
                           "values; reads no arm")
    mode.add_argument("--verify-registration", action="store_true",
                      help="step 0's registration part only; reads no npz")
    a = ap.parse_args(argv)
    _static_checks()
    files = {arm: rp(getattr(a, arm.lower())) for arm in ARMS}

    if a.check_reference:
        try:
            ref = _load(rp(a.ref), "page", REF_NAMES, REF_DESC)
        except EngineeringError as err:
            print(f"ENGINEERING FAIL (reference): {err}", file=sys.stderr)
            return 2
        pub = reference_values(ref)
        for s, r in pub["entries"].items():
            print(f"  page run N2 unit, {s} at Dec 2032: {r['page_run_m']:+.4f} m "
                  f"(published {r['published_m']:+.2f})")
        print(f"reference sha256 {ref['sha256']} "
              f"({'registered' if ref['sha256'] == REF_SHA256 else 'NOT the registered file'})")
        print("check-reference:", "ok" if pub["ok"] else "FAIL")
        return 0 if pub["ok"] else 3

    reg = None
    try:
        if not (a.verify_registration or a.repro) and not a.out:
            raise EngineeringError("scoring mode needs --out")
        if a.out and rp(a.out).exists():
            raise EngineeringError(f"{a.out} exists: the declaration is scored once")
        doc, decl_sha = load_registered()
        frozen = check_frozen(doc)
        digests = check_input_digests(doc)
        reg = {"declaration_sha256": decl_sha, "frozen_code_sha256": frozen,
               "input_digests": digests}
        print(f"step 0 ok: {PREREG_ID} registered (sha256 {decl_sha[:16]}), frozen code and "
              "declared inputs match")
        if a.verify_registration:
            return 0
        if a.repro:
            reg["attempts"] = check_attempts(rp(a.attempts), {"S0": files["S0"]},
                                             need_repro=False)
        else:
            reg["attempts"] = check_attempts(rp(a.attempts), files, rp(a.s3_members))
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 0): {err}", file=sys.stderr)
        return 1
    try:
        ref = _load(rp(a.ref), "page", REF_NAMES, REF_DESC)
        if ref["sha256"] != REF_SHA256:
            raise EngineeringError(f"{a.ref} is not the registered page run")
        s0 = _load(files["S0"], "S0")
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    rep = reproduction(s0, ref)
    print(f"step 1 ok (S0, page run); step 2 reproduction {'ok' if rep['ok'] else 'FAIL'}")
    for s, r in rep["screens"].items():
        print(f"  {s} heads at the 158 screens: max |S0 - page| {r['max_abs_diff_m']:.2e} m")
    for s, r in rep["n2_unit_dec2032"].items():
        print(f"  N2 unit {s} Dec 2032: S0 {r['S0_m']:+.4f} m, page run {r['page_run_m']:+.4f} "
              f"m, published {PUBLISHED_DEC2032[s]:+.2f}")
    if not rep["ok"]:
        print("ENGINEERING FAIL (step 2): S0 does not reproduce the page run; no other arm "
              "read", file=sys.stderr)
        return 3
    if a.repro:
        print(f"repro: ok (S0 sha256 {s0['sha256']})")
        return 0
    try:
        arms = {"S0": s0, **{arm: _load(files[arm], arm) for arm in ARMS[1:]}}
        mem = _load_members(rp(a.s3_members))
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    res = score(arms, mem)
    report = {"prereg_id": PREREG_ID, "scorer_version": SCORER_VERSION, "registration": reg,
              "screens_sha256": SCREENS_SHA256,
              "files": {**{arm: {"path": d["path"], "sha256": d["sha256"]}
                           for arm, d in arms.items()},
                        "S3_members": {"path": mem["path"], "sha256": mem["sha256"]}},
              "reproduction": rep, **res}
    for arm, t in res["table"].items():
        for k, r in t.items():
            print(f"  {arm} {k}: ratio {r['ratio']:.4f} p90 {r['p90_rel_dev']:.4f} "
                  f"pass {r['pass']}")
    print(f"VERDICT: {res['verdict']} (wording case {res['summary']['wording_case']})")
    p = rp(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:                      # never overwrite a scored report
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
