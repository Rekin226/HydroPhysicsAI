"""The fold-member rerun of the frozen forward (E02d's relative member spread): CPU steps.

The frozen forward results/twin_forward/datum_gate_innov.npz (sha cf2c7154) stores only the
member mean and standard deviation (heads_mean, heads_std), so a well-relative ensemble
spread needs each member's heads. The page run had 18 parameter sets (the in-sample fit,
fold0-4, foldpost0-11 of results/twin_runs/stage3_datum_gate) x 2 initial fields
(``--ic-members 1``: the unperturbed field and one perturbation, seed 0, sigma 0.5 m), 36
member runs. ``run_e02d.sh members`` reruns each parameter set ALONE, with the page recipe
FS.start_invariance.v1 reproduced at 4c2723e (S0 repro rc 0, 2026-11-01), baseline scenario
only, twice: ``--ic-members 1`` (mean of the two runs) and ``--ic-members 0`` (the
unperturbed run). The perturbation is drawn once per run from rng(seed) before the member
loop (forward.run), so it is the page run's; the perturbed member is 2 * mean - unperturbed.

    thetas   (PYTHONPATH = the frozen worktree; python -P) write one stage3_theta.json per
             parameter set exactly as forward.load_members builds it (fold well_datum merged)
    extract  read the 36 runs at the 158 wells (cells and layers of DR's wells_e02d.csv),
             apply the engineering gate (member mean reproduces the stored heads_mean within
             1e-3 m through 2029-10; the two runs of a parameter set agree on the hindcast)
             and write results/twin_runs/e02d/members_wells.npz + members_gate.json

Model output only: no observation is read, so no look. The forwards' stdout (which prints
in-sample hindcast skill against observed heads and leveling) is sealed by the runner.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# no repository root on sys.path: under 'thetas' hydrophysics must come from PYTHONPATH (the
# frozen worktree); 'extract' needs no hydrophysics import at all
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402

THETA_FILES = ("stage3_theta.json", "stage3_fold_thetas.json", "stage3_posterior_tn.json")
N_PARAM = 18
RUN_DIR = "results/twin_runs/e02d"
THETAS = f"{RUN_DIR}/thetas"
MEMBERS_OUT = "results/twin_forward/e02d_members"
MEMBERS_NPZ = f"{RUN_DIR}/members_wells.npz"
GATE_JSON = f"{RUN_DIR}/members_gate.json"
LAST = "2029-10"


def dump_thetas(out_dir: Path, wt: Path) -> dict:
    """One {"theta", "meta"} file per parameter set, from the frozen worktree's
    forward.load_members (refuses unless hydrophysics imports from ``wt``)."""
    import hydrophysics.twin.forward as F

    if not Path(F.__file__).resolve().is_relative_to(wt.resolve()):
        raise io.Refused(f"hydrophysics.twin.forward imports from {F.__file__}, not {wt}")
    src = [io.REPO / L.THETA_DIR / f for f in THETA_FILES]
    members = F.load_members([str(p) for p in src])
    if len(members) != N_PARAM:
        raise io.EngineeringFailure(f"{len(members)} parameter sets, expected {N_PARAM}")
    labels = [m.label for m in members]
    if len(set(labels)) != len(labels):
        raise io.EngineeringFailure(f"duplicate member labels {labels}")
    out_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for m in members:
        d = out_dir / m.label
        d.mkdir()
        p = d / "stage3_theta.json"
        p.write_text(json.dumps({"theta": m.theta, "meta": m.meta}, indent=1) + "\n")
        back = F.load_members([str(p)])[0]
        if back.theta != m.theta or back.meta != m.meta or back.label != m.label:
            raise io.EngineeringFailure(f"{p} does not load back as member {m.label}")
        files[m.label] = io.sha256(p)
    man = {"labels": labels, "files": files,
           "sources": {f: io.sha256(p) for f, p in zip(THETA_FILES, src, strict=True)},
           "forward_module": str(Path(F.__file__).resolve().relative_to(wt.resolve())),
           "worktree_head": os.popen(f"git -C {wt} rev-parse HEAD").read().strip()}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def _run_heads(path: Path, layers0: np.ndarray, cells: np.ndarray, n_t: int) -> dict:
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["scenario_names"]]
        if names[0] != "baseline":
            raise io.EngineeringFailure(f"{path}: first scenario {names[0]!r}")
        h = z["heads_mean"][0][layers0, cells, :n_t].astype(float)
        return {"heads": h, "n_members": int(z["n_members"]),
                "labels": [str(s) for s in z["member_labels"]],
                "dates": [str(s) for s in z["dates"][:n_t]], "origin": int(z["origin"]),
                "options": json.loads(str(z["forward_options"]))}


def extract(wells: pd.DataFrame, labels: list[str], members_dir: Path, page: Path,
            last: str = LAST) -> tuple[dict, dict]:
    """(arrays, gate) from the 2 x 18 runs; arrays['members'] is (36, W, T) on the page
    axis 2012-01..``last``."""
    with np.load(page, allow_pickle=False) as z:
        pdates = pd.DatetimeIndex(pd.to_datetime(z["dates"])).to_period("M").to_timestamp()
        n_t = int(np.flatnonzero(pdates == pd.Timestamp(f"{last}-01"))[0]) + 1
        lay0 = wells["layer"].to_numpy(int) - 1
        cells = wells["cell"].to_numpy(int)
        pmean = z["heads_mean"][0][lay0, cells, :n_t].astype(float)
        pstd = z["heads_std"][0][lay0, cells, :n_t].astype(float)
        origin = int(z["origin"])
        page_dates = [str(s) for s in z["dates"][:n_t]]
    mem, tags, problems = [], [], []
    hind = 0.0
    for lab in labels:
        r1 = _run_heads(members_dir / f"{lab}_ic1.npz", lay0, cells, n_t)
        r0 = _run_heads(members_dir / f"{lab}_ic0.npz", lay0, cells, n_t)
        for tag, r, n in (("ic1", r1, 2), ("ic0", r0, 1)):
            if r["n_members"] != n or r["labels"] != [lab]:
                problems.append(f"{lab}_{tag}: n_members {r['n_members']} labels {r['labels']}")
            if r["dates"] != page_dates or r["origin"] != origin:
                problems.append(f"{lab}_{tag}: time axis differs from the page run")
        hind = max(hind, float(np.max(np.abs(r1["heads"][:, :origin + 1]
                                             - r0["heads"][:, :origin + 1]))))
        x0 = r0["heads"]
        x1 = 2.0 * r1["heads"] - x0
        mem += [x0, x1]
        tags += [(lab, 0), (lab, 1)]
    M = np.stack(mem)
    dm = np.abs(M.mean(axis=0) - pmean)
    ds = np.abs(M.std(axis=0) - pstd)
    gate = {"tol_m": L.MEMBER_MEAN_TOL_M, "members": int(M.shape[0]), "wells": int(M.shape[1]),
            "months": int(n_t), "last": last,
            "max_abs_mean_diff_m": float(dm.max()),
            "max_abs_hindcast_ic1_vs_ic0_m": hind,
            "max_abs_std_diff_m_report_only": float(ds.max()),
            "problems": problems}
    gate["ok"] = (not problems and gate["max_abs_mean_diff_m"] <= L.MEMBER_MEAN_TOL_M
                  and hind <= L.MEMBER_MEAN_TOL_M)
    arrays = {"members": M, "station_id": np.array(wells["station_id"].astype(str).tolist()),
              "dates": np.array(page_dates), "member": np.array([t[0] for t in tags]),
              "ic": np.array([t[1] for t in tags])}
    return arrays, gate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("thetas")
    t.add_argument("--worktree", required=True, type=Path)
    t.add_argument("--out", type=Path, default=io.REPO / THETAS)
    e = sub.add_parser("extract")
    e.add_argument("--wells", type=Path,
                   default=io.REPO / f"results/dataprep/reports/{L.DR_ID}/wells_e02d.csv")
    e.add_argument("--members-dir", type=Path, default=io.REPO / MEMBERS_OUT)
    e.add_argument("--thetas", type=Path, default=io.REPO / THETAS)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "thetas":
            man = dump_thetas(a.out, a.worktree)
            print(json.dumps({"labels": man["labels"], "head": man["worktree_head"]}))
            return 0
        page = io.REPO / L.FORWARD
        if io.sha256(page) != L.MODEL_SHA:
            raise io.EngineeringFailure("page forward sha256 differs from cf2c7154")
        wells = pd.read_csv(a.wells, dtype={"station_id": str})
        labels = json.loads((a.thetas / "manifest.json").read_text())["labels"]
        arrays, gate = extract(wells, labels, a.members_dir, page)
        gate["inputs"] = {"wells_e02d": io.rec(a.wells), "page_forward": io.rec(page),
                          "runs": {p.name: io.sha256(p)
                                   for p in sorted(a.members_dir.glob("*.npz"))}}
        out = io.REPO / MEMBERS_NPZ
        if out.exists():
            raise io.Refused(f"{out} exists (archive it before a re-extraction)")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        gate["members_wells_sha256"] = io.sha256(out)
        io.write_once(io.REPO / GATE_JSON, gate)
        print(json.dumps({k: gate[k] for k in ("ok", "max_abs_mean_diff_m",
                                               "max_abs_hindcast_ic1_vs_ic0_m",
                                               "max_abs_std_diff_m_report_only")}))
        return 0 if gate["ok"] else 3
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
