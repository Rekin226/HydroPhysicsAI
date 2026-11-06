"""The all30 scenario forward on the page theta (round 8 rank 14; user ruling R8.9).

    python dataprep_staging/all30/run_all30.py check              preconditions only (CPU)
    DRY=1 python dataprep_staging/all30/run_all30.py run          print the forward command
    python dataprep_staging/all30/run_all30.py run [--code DIR]   the forward (GPU, in tmux)
    python dataprep_staging/all30/run_all30.py verify             re-check the outputs (CPU)

Ruling R8.9 (user rulings 2026-11-03, twelfth set): "all30 (all-class x0.7) at N2's unit is the
headline; cut30 and retire_aqua kept as class scenarios labelled 'size depends on the pumping
conversion'. Scenario forward to a separate file; cf2c7154 must stay byte-identical."

What runs. ``twin.forward`` with the page recipe exactly as FS.start_invariance.v1's arm S0
reproduced it (max |heads difference| 0.0 m against the page run, code 4c2723e): the page theta
stage3_datum_gate with its five fold and twelve posterior parameter sets (18 sets x 2 initial
fields = 36 members), both columns (coupled_leveling, tau30y), the innovation restart with a
5 km taper, the calibrated apex hold, the column on free-running heads with the start-up load
released, the vendor rain cache the page run read, horizon 120 months, per-member yearly fields
as on the page run. The scenarios are the page's cut30 and retire_aqua plus all30
(every class x0.7 from 2026-01, the same start and horizon as cut30), so the new file carries
its own reproduction of the page's levers.

Why a forward and not the response basis. The page's fast estimate superposes one-class
retirements of ONE parameter set (``response_basis_innov.npz``), rescaled per class to the
ensemble's solved runs; three of the six classes have no solved run (assumed factor), and the
solver is not linear in the pumping multipliers (the basis's own check, irrigation x0.5 against
half of irrigation x0, misses by 6 % at N2's unit). cut30 itself is a full ensemble forward
(``datum_gate_innov.npz``), so all30 is produced the same way. ``python ... check`` prints the
superposition numbers it would have given.

The bound file. ``results/twin_forward/datum_gate_innov.npz`` (sha256 cf2c7154...) is the
registered forward of N2/E02/E02c/N3F and the page's. Every subcommand refuses unless it is
byte-identical to that sha256, ``run`` checks it again after the forward and fails (exit 3) if
it changed, and the output stem may not be that file's. Outputs are never overwritten.

Outputs (``results/twin_forward/all30_page.*``, gitignored): the forward's npz, summary.csv,
members.csv and members.npz, its stdout (``.log``) and stderr (``.stderr.log``), and the sidecar
``all30_page.provenance.json``: the command, the code commit that ran, the sha256 of every input
and output, the reference check before and after, the layout and reproduction checks against the
page run and N2's unit (the mean over N2's 84 registered Yunlin screens, each in its own layer;
the frozen screens of FS.start_invariance.v1) for each lever. The page builder
(``twin.viewer_app --all30``) refuses an all30 file whose sidecar does not vouch for it.

Model output only: no registration, no test, no status. Exit codes: 0 ok, 1 refused
(precondition), 2 the forward failed, 3 the reference changed or the outputs do not reproduce
the page run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

RULING = ("user ruling 2026-11-03 (twelfth set) (R8.9); "
          "results/dataprep/rulings/user_rulings_2026-11-03.json (sha256 ec610b69)")
REF = "results/twin_forward/datum_gate_innov.npz"
REF_SHA256 = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
OUT_STEM = "results/twin_forward/all30_page"
G = "results/twin_runs/stage3_datum_gate"
THETAS = (f"{G}/stage3_theta.json", f"{G}/stage3_fold_thetas.json",
          f"{G}/stage3_posterior_tn.json")
VEPS = (f"{G}/coupled_leveling/vep_zonal_leveling.json",
        f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json")
RF = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
# the forward's default data files (calibrate_flow.DEFAULT_PATHS): hashed, never parsed
DATA_FILES = ("chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json",
              "AMP_V2/data/fan_stations.parquet", "AMP_V2/data/tpc_pumps.parquet",
              "AMP_V2/data/pump_kwh_all.parquet", "chou-shui-data/data/rf_stations.csv",
              "chou-shui-data/data/gw_stations.csv", "results/et/openmeteo_et0_2012_2022.npz",
              "results/twin/basemap.npz")
DATA_DIRS = (("AMP_V2/data/wells", "*.parquet"),)
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")
ALL30 = "all30:" + ",".join(f"{c}=0.7" for c in CLASSES) + "@2026-01"
SCENARIOS = ("cut30:irrigation=0.7@2026-01", "retire_aqua:aquaculture=0@2026-01", ALL30)
NAMES = ("baseline", "cut30", "retire_aqua", "all30")
ALL30_DESC = ("all30: " + ", ".join(f"{c} x0.7" for c in sorted(CLASSES)) + " from 2026-01")
FLAGS = ("--horizon", "120", "--gain", "1.0", "--ic-members", "1", "--ic-sigma", "0.5",
         "--seed", "0", "--update-method", "innovation", "--restart-taper-km", "5",
         "--apex-hold", "calibrated", "--column-heads", "free", "--column-hpc0-fast-days",
         "365", "--rf-timeseries", RF, "--device", "cuda", "--compile-matvec",
         "--save-members", "yearly")
REF_TOL_M = 0.01            # the page run's levers, every layer, cell and month (FS's REF_TOL_M)
PUBLISHED_DEC2032 = {"cut30": 1.19, "retire_aqua": 1.41}     # ledger C01-sz, 2 decimals
PUBLISHED_TOL_M = 0.005
MIN_FREE_MIB = int(os.environ.get("ALL30_MIN_FREE_MIB", "5000"))
FS_SCORER = "dataprep_staging/fs_start/score_fs.py"     # frozen N2 screens (read, never edited)
LAYOUT_KEYS = ("dates", "origin", "mask", "nx", "ny", "dx", "x0", "y0", "n_members",
               "member_labels", "rheology_labels", "horizon", "gain", "hindcast_gain",
               "hindcast_every", "gate", "forward_options")


class Refused(RuntimeError):
    """A precondition failed; nothing ran."""


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dir_digest(path: Path, pattern: str) -> dict:
    """(file count, sha256 over the sorted lines '<name> <sha256>\\n'), as FS's input digest."""
    files = sorted(Path(path).glob(pattern))
    lines = "".join(f"{f.name} {sha256(f)}\n" for f in files)
    return {"n_files": len(files), "sha256": hashlib.sha256(lines.encode()).hexdigest()}


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def outputs(stem: str = OUT_STEM) -> dict[str, str]:
    """Every file one run writes, by role."""
    return {"npz": f"{stem}.npz", "summary_csv": f"{stem}.summary.csv",
            "members_csv": f"{stem}.members.csv", "members_npz": f"{stem}.members.npz",
            "stdout_log": f"{stem}.log", "stderr_log": f"{stem}.stderr.log"}


def sidecar_path(stem: str = OUT_STEM) -> str:
    return f"{stem}.provenance.json"


def check_reference(repo: Path = REPO, ref: str | None = None, want: str | None = None) -> str:
    """The bound forward run, byte-identical to its registered sha256; returns that sha."""
    ref, want = ref or REF, want or REF_SHA256
    p = repo / ref
    if not p.is_file():
        raise Refused(f"{ref} is missing; the bound forward run must exist unchanged")
    got = sha256(p)
    if got != want:
        raise Refused(f"{ref} sha256 {got[:8]} is not {want[:8]}: the bound forward run "
                      "changed; nothing runs until it is restored byte-identical")
    return got


def check_stem(stem: str, ref: str | None = None) -> None:
    ref = ref or REF
    if Path(stem).with_suffix(".npz").name == Path(ref).name or \
            os.path.normpath(stem + ".npz") == os.path.normpath(ref):
        raise Refused(f"output stem {stem} would write over the bound forward run {ref}")


def forward_argv(python: str, stem: str = OUT_STEM) -> list[str]:
    """The exact forward command, relative to the repository root."""
    argv = [python, "-P", "-m", "hydrophysics.twin.forward"]
    for t in THETAS:
        argv += ["--theta", t]
    for s in SCENARIOS:
        argv += ["--scenario", s]
    argv += list(FLAGS) + ["--vep-json", ",".join(VEPS), "--out", stem]
    return argv


def input_hashes(repo: Path = REPO) -> dict:
    """sha256 of every declared input (files) and the digest of the head-well directory."""
    out, missing = {}, []
    for p in (REF, *THETAS, *VEPS, RF, *DATA_FILES):
        if (repo / p).is_file():
            out[p] = sha256(repo / p)
        else:
            missing.append(p)
    for d, pat in DATA_DIRS:
        if (repo / d).is_dir():
            out[f"{d}/{pat}"] = dir_digest(repo / d, pat)
        else:
            missing.append(d)
    return {"sha256": out, "missing": missing}


def code_info(code: Path) -> dict:
    """The commit of the code that runs, and whether its hydrophysics/ tree is clean."""
    def git(*a):
        return subprocess.run(["git", "-C", str(code), *a], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", "hydrophysics")
    except (OSError, subprocess.CalledProcessError) as e:
        raise Refused(f"{code.name}: not a git checkout ({type(e).__name__})") from e
    fwd = code / "hydrophysics" / "twin" / "forward.py"
    return {"dir": code.name, "commit": head, "hydrophysics_clean": not dirty,
            "forward_py_sha256": sha256(fwd) if fwd.is_file() else None}


def check_code(code: Path, python: str) -> dict:
    info = code_info(code)
    if not info["hydrophysics_clean"]:
        raise Refused(f"{code.name}: hydrophysics/ has uncommitted changes; run committed code")
    env = {**os.environ, "PYTHONPATH": str(code)}
    r = subprocess.run([python, "-P", "-c", "import hydrophysics.twin.forward as m; "
                        "print(m.__file__)"], capture_output=True, text=True, env=env,
                       cwd=str(REPO))
    mod = r.stdout.strip()
    if r.returncode != 0 or not os.path.realpath(mod).startswith(os.path.realpath(code) + os.sep):
        raise Refused(f"hydrophysics.twin.forward does not import from {code.name} "
                      f"(rc {r.returncode})")
    return info


def gpu_free_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise Refused("nvidia-smi failed: check the GPU before any forward")
    total, used = (int(v) for v in r.stdout.splitlines()[0].replace(",", " ").split())
    return total - used


def preconditions(repo: Path = REPO, stem: str = OUT_STEM) -> dict:
    check_stem(stem)
    before = check_reference(repo)
    ins = input_hashes(repo)
    if ins["missing"]:
        raise Refused(f"inputs missing: {ins['missing']}")
    present = [p for p in (*outputs(stem).values(), sidecar_path(stem)) if (repo / p).exists()]
    return {"reference_sha256_before": before, "inputs": ins["sha256"], "present": present}


# ---------------------------------------------------------------------------------- checks
def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype.kind == b.dtype.kind and bool(np.all(a == b))


def layout_check(out: dict, ref: dict) -> dict:
    """The new file's layout against the page run's: same grid, months, members, columns and
    forward options; its scenarios are the page's three plus all30."""
    bad = [k for k in LAYOUT_KEYS if k in ref and (k not in out or not _same(out[k], ref[k]))]
    names = [str(s) for s in out.get("scenario_names", [])]
    desc = [str(s) for s in out.get("scenarios", [])]
    rdesc = [str(s) for s in ref["scenarios"]]
    want = [*rdesc, ALL30_DESC]
    if names != list(NAMES):
        bad.append(f"scenario_names {names}")
    if desc != want:
        bad.append(f"scenarios {desc}")
    return {"ok": not bad, "mismatch": bad}


def reproduction(out: dict, ref: dict, tol: float = REF_TOL_M) -> dict:
    """The page's three scenarios in the new file against the page run, every layer, cell and
    month (heads gate at ``tol``; subsidence reported)."""
    rn = [str(s) for s in ref["scenario_names"]]
    on = [str(s) for s in out["scenario_names"]]
    res, ok = {}, True
    for s in rn:
        i, j = on.index(s), rn.index(s)
        dh = float(np.max(np.abs(out["heads_mean"][i].astype("float64")
                                 - ref["heads_mean"][j].astype("float64"))))
        ds = float(np.max(np.abs(out["subs_mean"][i].astype("float64")
                                 - ref["subs_mean"][j].astype("float64"))))
        res[s] = {"heads_max_abs_diff_m": dh, "subs_max_abs_diff_m": ds, "ok": dh <= tol}
        ok &= dh <= tol
    return {"ok": bool(ok), "tol_m": tol, "scenarios": res}


def load_screens(repo: Path = REPO):
    """N2's 84 registered Yunlin screens as (cells, layers): FS.start_invariance.v1's frozen
    rows, checked against their own sha256. Read only; that scorer is hash-bound."""
    p = repo / FS_SCORER
    spec = importlib.util.spec_from_file_location("_fs_screens", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod._screens_sha() != mod.SCREENS_SHA256:
        raise Refused("the frozen N2 screens do not match their sha256")
    yun = mod.masks()["yunlin"]
    return (np.asarray(mod.CELLS)[yun], np.asarray(mod.LAYERS)[yun],
            {"source": FS_SCORER, "screens_sha256": mod.SCREENS_SHA256,
             "n_yunlin": int(yun.sum())})


def year_ends(dates) -> np.ndarray:
    return np.array([i for i, d in enumerate(dates) if str(d)[5:7] == "12"], dtype="int64")


def n2_unit(heads: np.ndarray, names: list[str], cells, layers, ye) -> dict:
    """Policy minus baseline, mean over the screens, each in its own layer, per year-end."""
    b = names.index("baseline")
    base = heads[b][layers, cells][:, ye].astype("float64")
    return {s: (heads[i][layers, cells][:, ye].astype("float64") - base).mean(axis=0)
            for i, s in enumerate(names) if s != "baseline"}


def unit_block(out: dict, ref: dict, repo: Path = REPO) -> dict:
    cells, layers, src = load_screens(repo)
    ye = year_ends(out["dates"])
    on = [str(s) for s in out["scenario_names"]]
    u = n2_unit(out["heads_mean"], on, cells, layers, ye)
    ur = n2_unit(ref["heads_mean"], [str(s) for s in ref["scenario_names"]], cells, layers, ye)
    pub = {s: {"all30_file_m": float(u[s][-1]), "page_run_m": float(ur[s][-1]),
               "published_m": v, "ok": abs(float(u[s][-1]) - v) <= PUBLISHED_TOL_M}
           for s, v in PUBLISHED_DEC2032.items()}
    l2 = out["heads_mean"][on.index("all30"), 1] - out["heads_mean"][0, 1]
    return {**src, "year_ends": [str(out["dates"][i])[:7] for i in ye],
            "all30_series_m": [round(float(v), 6) for v in u["all30"]],
            "dec2032_m": {s: float(v[-1]) for s, v in u.items()},
            "published_check": {"ok": all(p["ok"] for p in pub.values()), "tol_m":
                                PUBLISHED_TOL_M, "levers": pub},
            "all30_fan_layer2_dec2032_m": float(l2[:, ye[-1]].mean())}


def verify(repo: Path = REPO, stem: str = OUT_STEM) -> dict:
    """Layout, reproduction and N2's unit of an existing all30 file (CPU, read only)."""
    o = outputs(stem)
    npz = repo / o["npz"]
    if not npz.is_file():
        raise Refused(f"{o['npz']} does not exist (run the forward first)")
    out, ref = _load(npz), _load(repo / REF)
    lay = layout_check(out, ref)
    rep = reproduction(out, ref) if lay["ok"] else {"ok": False, "skipped": "layout"}
    unit = unit_block(out, ref, repo) if lay["ok"] else None
    ok = lay["ok"] and rep["ok"] and bool(unit and unit["published_check"]["ok"])
    return {"ok": ok, "layout": lay, "reproduction": rep, "n2_unit": unit,
            "outputs_sha256": {k: sha256(repo / p) for k, p in o.items()
                               if (repo / p).is_file()}}


def superposition_note(repo: Path = REPO) -> dict | None:
    """What the page's response basis alone would give for all30 at N2's unit (one parameter
    set, raw and per-class rescaled), and the basis's own nonlinearity check: why the page
    needs the forward. Read only; None when the files are absent."""
    bp = repo / "results/twin_forward/response_basis_innov.npz"
    if not (bp.is_file() and (repo / REF).is_file() and (repo / FS_SCORER).is_file()):
        return None
    cells, layers, _ = load_screens(repo)
    b, ref = _load(bp), _load(repo / REF)
    ye = year_ends(ref["dates"])
    bn = [str(s) for s in b["scenario_names"]]
    u = n2_unit(b["heads_mean"], bn, cells, layers, ye)
    ur = n2_unit(ref["heads_mean"], [str(s) for s in ref["scenario_names"]], cells, layers, ye)
    keys = dict(zip(CLASSES, ("irr0_2026", "aqua0_2026", "live0_2026", "dom0_2026",
                              "ind0_2026", "oth0_2026"), strict=True))
    r = {c: float(u[k][-1]) for c, k in keys.items()}
    raw = 0.3 * sum(r.values())
    k_irr = float(ur["cut30"][-1]) / (0.3 * r["irrigation"])
    k_aqua = float(ur["retire_aqua"][-1]) / r["aquaculture"]
    k_mean = (k_irr + k_aqua) / 2
    scaled = 0.3 * (k_irr * r["irrigation"] + k_aqua * r["aquaculture"]
                    + k_mean * sum(r[c] for c in CLASSES[2:]))
    half = float(u["check_irr50_2026"][-1])
    return {"basis_sets": 1, "forward_sets": 18, "all30_raw_superposition_m": raw,
            "all30_rescaled_superposition_m": scaled,
            "rescale_factors": {"irrigation": k_irr, "aquaculture": k_aqua,
                                "other_four_assumed": k_mean},
            "nonlinearity_irr50": {"check_m": half, "half_of_irr0_m": 0.5 * r["irrigation"],
                                   "rel": (0.5 * r["irrigation"] - half) / half},
            "exact": False}


# ---------------------------------------------------------------------------------- commands
def _write_sidecar(repo: Path, stem: str, rec: dict) -> None:
    p = repo / sidecar_path(stem)
    with p.open("x", encoding="utf-8") as f:       # never overwritten
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")


def cmd_check(repo: Path, stem: str, python: str) -> int:
    pre = preconditions(repo, stem)
    print(f"reference {REF}: sha256 {pre['reference_sha256_before'][:8]} (byte-identical)")
    print(f"inputs: {len(pre['inputs'])} hashed, none missing")
    if pre["present"]:
        print(f"outputs already present (a run would refuse): {pre['present']}")
    sp = superposition_note(repo)
    if sp:
        print("response basis alone (not used): all30 at N2's unit Dec 2032 "
              f"{sp['all30_raw_superposition_m']:+.3f} m raw, "
              f"{sp['all30_rescaled_superposition_m']:+.3f} m rescaled; irrigation x0.5 "
              f"check misses half of x0 by {100 * sp['nonlinearity_irr50']['rel']:.1f} %: "
              "not exact, the forward is needed")
    print("command: " + shlex.join(forward_argv(python, stem)))
    return 0


def cmd_run(repo: Path, stem: str, python: str, code: Path, dry: bool) -> int:
    pre = preconditions(repo, stem)
    argv = forward_argv(python, stem)
    if dry:
        print(f"reference sha256 {pre['reference_sha256_before'][:8]} ok; would run from "
              f"{repo.name} with PYTHONPATH={code}:")
        print(shlex.join(argv))
        return 0
    if pre["present"]:
        raise Refused(f"outputs exist (never overwritten; move them aside): {pre['present']}")
    info = check_code(code, python)
    free = gpu_free_mib()
    print(f"GPU memory free: {free} MiB (need {MIN_FREE_MIB})", flush=True)
    if free < MIN_FREE_MIB:
        raise Refused(f"not enough free GPU memory ({free} MiB)")
    o = outputs(stem)
    t0 = utc()
    env = {**os.environ, "PYTHONPATH": str(code)}
    with (repo / o["stdout_log"]).open("x") as so, (repo / o["stderr_log"]).open("x") as se:
        rc = subprocess.run(argv, cwd=str(repo), env=env, stdout=so, stderr=se).returncode
    t1 = utc()
    after = sha256(repo / REF)
    rec = {"schema": "all30_scenario_forward.v1", "ruling": RULING, "model_only": True,
           "registration": None, "purpose": "all30 head lever at N2's unit for the page "
           "(tile tW headline); cut30 and retire_aqua reproduced in the same file",
           "command": ["$PYTHON", *argv[1:]], "cwd": "repository root",
           "env": {"PYTHONPATH": code.name, "CUDA_VISIBLE_DEVICES":
                   os.environ.get("CUDA_VISIBLE_DEVICES")},
           "python": sys.version.split()[0], "code": info, "started_utc": t0,
           "finished_utc": t1, "returncode": rc, "inputs_sha256": pre["inputs"],
           "reference": {"path": REF, "sha256_expected": REF_SHA256,
                         "sha256_before": pre["reference_sha256_before"],
                         "sha256_after": after, "byte_identical": after == REF_SHA256},
           "scenario": {"name": "all30", "spec": ALL30, "desc": ALL30_DESC}}
    with (repo / o["stdout_log"]).open() as f:
        first = f.readline().strip()
    rec["device_line"] = first
    status, code_rc = "ok", 0
    if rc != 0 or first != "device: cuda":
        status, code_rc = "forward_failed", 2
    elif after != REF_SHA256:
        status, code_rc = "reference_changed", 3
    else:
        v = verify(repo, stem)
        rec.update({k: v[k] for k in ("layout", "reproduction", "n2_unit")})
        if not v["ok"]:
            status, code_rc = "does_not_reproduce", 3
    rec["outputs_sha256"] = {k: sha256(repo / p) for k, p in o.items() if (repo / p).is_file()}
    rec["status"] = status
    _write_sidecar(repo, stem, rec)
    print(f"all30 forward: {status} (exit {code_rc}); sidecar {sidecar_path(stem)} "
          f"sha256 {sha256(repo / sidecar_path(stem))[:8]}; reference "
          f"{'byte-identical' if after == REF_SHA256 else 'CHANGED'}")
    if status == "ok":
        d = rec["n2_unit"]["dec2032_m"]
        print("N2's unit, Dec 2032 (model output): " + ", ".join(
            f"{s} {v:+.3f} m" for s, v in d.items()))
    return code_rc


def cmd_verify(repo: Path, stem: str) -> int:
    check_reference(repo)
    v = verify(repo, stem)
    sc = repo / sidecar_path(stem)
    side = json.loads(sc.read_text()) if sc.is_file() else None
    agree = bool(side) and side.get("outputs_sha256", {}).get("npz") == v["outputs_sha256"].get(
        "npz")
    print(f"layout {'ok' if v['layout']['ok'] else v['layout']['mismatch']}; reproduction "
          f"{'ok' if v['reproduction']['ok'] else 'FAIL'}; sidecar "
          f"{'vouches for this npz' if agree else 'missing or for another file'}")
    if v["n2_unit"]:
        print("N2's unit, Dec 2032 (model output): " + ", ".join(
            f"{s} {x:+.3f} m" for s, x in v["n2_unit"]["dec2032_m"].items()))
    return 0 if (v["ok"] and agree) else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check", "run", "verify"))
    ap.add_argument("--code", default=str(REPO),
                    help="checkout whose hydrophysics/ runs (PYTHONPATH; default this one)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the command (also DRY=1)")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    try:
        if a.cmd == "check":
            return cmd_check(REPO, OUT_STEM, a.python)
        if a.cmd == "verify":
            return cmd_verify(REPO, OUT_STEM)
        return cmd_run(REPO, OUT_STEM, a.python, Path(a.code).resolve(),
                       a.dry_run or os.environ.get("DRY") == "1")
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
