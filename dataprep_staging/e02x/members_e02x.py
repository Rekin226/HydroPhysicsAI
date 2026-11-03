"""The 36 fold members (18 parameter sets x 2 initial fields) at N3F's 66 external cells:
extraction and engineering gate (CPU). The forwards themselves are GPU runs launched by
``run_e02x.sh members`` (model only, sealed stdout, no --data).

The members are E02d's: the same 18 theta files (results/twin_runs/e02d/thetas, checked
against E02d's manifest), the same frozen worktree code (4c2723e), the same command line
(run_e02d.sh F_MEMBER and COMMON), ``--ic-members 1`` (mean of the unperturbed and the
perturbed run) and ``--ic-members 0`` (the unperturbed run); the perturbed member is
2 * mean - unperturbed. The forward writes the full field heads_mean (1, 4, 2148, 252), so the
external cells are read from the same files as the donors.

Gate (engineering; no statistic is computed before it passes):
  (a) the 36 members at the 158 donors reproduce results/twin_runs/e02d/members_wells.npz
      (E02d's registered member table) within 1e-3 m on every month 2012-01..2029-10;
  (b) the member mean at the 66 external cells reproduces the frozen page run's heads_mean
      (cf2c7154) within 1e-3 m;
  (c) the two runs of a parameter set agree on the hindcast (<= 1e-3 m) at both well sets;
  (d) source 'e02d' only: every run npz has the sha256 E02d's members_gate.json recorded.

Sources: 'e02d' (DEFAULT, lead decision Q1: E02d's own run files in results/twin_forward/
e02d_members, which already hold the full field; 0 GPU) or 'rerun' (the GPU reruns in
results/twin_forward/e02x_members). With 'e02d', gate (a) is an IDENTITY check on the same
files, not a reproduction; gate (b) is the real check. E02d's members_gate.json must have the
sha256 E02d's registration recorded (bae7b24d...); gate['source'] records the source.

Model output only: no observation is read, so no look.

    python -I dataprep_staging/e02x/members_e02x.py extract [--source e02d|rerun]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402

RUN_DIR = "results/twin_runs/e02x"
E02D_RUN_DIR = "results/twin_runs/e02d"
THETAS = f"{E02D_RUN_DIR}/thetas"                 # E02d's 18 theta files, reused unchanged
E02D_MEMBERS = f"{E02D_RUN_DIR}/members_wells.npz"
E02D_GATE = f"{E02D_RUN_DIR}/members_gate.json"
SOURCES = {"rerun": "results/twin_forward/e02x_members",
           "e02d": "results/twin_forward/e02d_members"}
E02D_GATE_SHA_PREFIX = "bae7b24d"           # E02d's registered members_gate.json (S5)
MEMBERS_NPZ = f"{RUN_DIR}/members_ext.npz"
GATE_JSON = f"{RUN_DIR}/members_gate.json"
LAST = "2029-10"
N_PARAM = 18


def run_heads(path: Path, lay0: np.ndarray, cells: np.ndarray, n_t: int) -> dict:
    """One member run's baseline heads at (layer, cell) pairs, months 0..n_t-1 (E02d's
    members_e02d._run_heads, generalised to any cell set)."""
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["scenario_names"]]
        if names[0] != "baseline":
            raise io.EngineeringFailure(f"{path}: first scenario {names[0]!r}")
        h = z["heads_mean"][0][lay0, cells, :n_t].astype(float)
        return {"heads": h, "n_members": int(z["n_members"]),
                "labels": [str(s) for s in z["member_labels"]],
                "dates": [str(s) for s in z["dates"][:n_t]], "origin": int(z["origin"])}


def extract(donors: pd.DataFrame, ext: pd.DataFrame, labels: list[str], members_dir: Path,
            page: Path, e02d_members: dict, last: str = LAST) -> tuple[dict, dict]:
    """(arrays, gate). ``donors``/``ext``: station_id, cell, layer (twin, 1-based).
    arrays['members_ext'] (36, 66, T) and arrays['members_donor'] (36, 158, T)."""
    with np.load(page, allow_pickle=False) as z:
        pdates = pd.DatetimeIndex(pd.to_datetime(z["dates"])).to_period("M").to_timestamp()
        n_t = int(np.flatnonzero(pdates == pd.Timestamp(f"{last}-01"))[0]) + 1
        le, ce = ext["layer"].to_numpy(int) - 1, ext["cell"].to_numpy(int)
        pmean_e = z["heads_mean"][0][le, ce, :n_t].astype(float)
        origin = int(z["origin"])
        page_dates = [str(s) for s in z["dates"][:n_t]]
    ld, cd = donors["layer"].to_numpy(int) - 1, donors["cell"].to_numpy(int)
    lay = np.concatenate([ld, le])
    cel = np.concatenate([cd, ce])
    nd = len(donors)
    mem, tags, problems = [], [], []
    hind = 0.0
    for lab in labels:
        r1 = run_heads(members_dir / f"{lab}_ic1.npz", lay, cel, n_t)
        r0 = run_heads(members_dir / f"{lab}_ic0.npz", lay, cel, n_t)
        for tg, r, n in (("ic1", r1, 2), ("ic0", r0, 1)):
            if r["n_members"] != n or r["labels"] != [lab]:
                problems.append(f"{lab}_{tg}: n_members {r['n_members']} labels {r['labels']}")
            if r["dates"] != page_dates or r["origin"] != origin:
                problems.append(f"{lab}_{tg}: time axis differs from the page run")
        hind = max(hind, float(np.max(np.abs(r1["heads"][:, :origin + 1]
                                             - r0["heads"][:, :origin + 1]))))
        x0 = r0["heads"]
        mem += [x0, 2.0 * r1["heads"] - x0]
        tags += [(lab, 0), (lab, 1)]
    M = np.stack(mem)
    Md, Me = M[:, :nd], M[:, nd:]
    # (a) the donors against E02d's registered member table, member by member
    ref_ids = [str(s) for s in e02d_members["station_id"]]
    ref_tags = list(zip([str(s) for s in e02d_members["member"]],
                        [int(i) for i in e02d_members["ic"]], strict=True))
    want = donors["station_id"].astype(str).tolist()
    if sorted(ref_ids) != sorted(want) or sorted(ref_tags) != sorted(tags):
        problems.append("donor wells or member tags differ from members_wells.npz")
        d_repro = float("inf")
    else:
        pos = [ref_ids.index(s) for s in want]
        mpos = [ref_tags.index(t) for t in tags]
        ref = np.asarray(e02d_members["members"], float)[np.ix_(mpos, pos)]
        T = min(ref.shape[-1], Md.shape[-1])
        if ref.shape[-1] != Md.shape[-1]:
            problems.append(f"members_wells.npz has {ref.shape[-1]} months, rerun {n_t}")
        d_repro = float(np.max(np.abs(Md[..., :T] - ref[..., :T])))
    # (b) the member mean at the external cells against the frozen page run
    e_mean = float(np.max(np.abs(Me.mean(axis=0) - pmean_e)))
    gate = {"tol_m": L.MEMBER_TOL_M, "members": int(M.shape[0]), "donors": int(nd),
            "ext_wells": int(Me.shape[1]), "months": int(n_t), "last": last,
            "max_abs_donor_vs_members_wells_m": d_repro,
            "max_abs_ext_mean_vs_page_m": e_mean,
            "max_abs_hindcast_ic1_vs_ic0_m": hind, "problems": problems}
    gate["ok"] = bool(not problems and d_repro <= L.MEMBER_TOL_M and e_mean <= L.MEMBER_TOL_M
                      and hind <= L.MEMBER_TOL_M)
    arrays = {"members_ext": Me, "ext_station_id": np.array(ext["station_id"].astype(str)
                                                            .tolist()),
              "members_donor": Md, "donor_station_id": np.array(want),
              "dates": np.array(page_dates), "member": np.array([t[0] for t in tags]),
              "ic": np.array([t[1] for t in tags])}
    return arrays, gate


def source_check(source: str, members_dir: Path, labels: list[str]) -> list[str]:
    """Source 'e02d': each run file must be the one E02d's gate recorded (sha256)."""
    if source != "e02d":
        return []
    g = json.loads(io.repo_path(E02D_GATE).read_text(encoding="utf-8"))
    runs = (g.get("inputs") or {}).get("runs") or {}
    bad = []
    for lab in labels:
        for ic in (0, 1):
            n = f"{lab}_ic{ic}.npz"
            p = members_dir / n
            if not p.is_file() or io.sha256(p) != runs.get(n):
                bad.append(n)
    return bad


def e02d_registered_gate(prereg_dir=None) -> str:
    """The sha256 of members_gate.json that E02d's registration recorded (must be
    bae7b24d...)."""
    doc, _ = io.registered(L.E02D_ID, prereg_dir)
    sha = str((((doc.get("interval") or {}).get("members") or {}).get("gate") or {})
              .get("sha256") or "")
    if not sha.startswith(E02D_GATE_SHA_PREFIX):
        raise io.EngineeringFailure(f"E02d's registered member gate sha {sha[:8]!r} is not "
                                    f"{E02D_GATE_SHA_PREFIX}")
    return sha


def thetas_labels() -> list[str]:
    man = json.loads(io.repo_path(f"{THETAS}/manifest.json").read_text(encoding="utf-8"))
    labels = list(man["labels"])
    if len(labels) != N_PARAM:
        raise io.EngineeringFailure(f"{len(labels)} parameter sets, expected {N_PARAM}")
    for lab in labels:
        p = io.repo_path(f"{THETAS}/{lab}/stage3_theta.json")
        if io.sha256(p) != man["files"][lab]:
            raise io.EngineeringFailure(f"{p} differs from E02d's theta manifest")
    return labels


def run_extract(source: str, members_dir: Path | None = None, prereg_dir=None) -> dict:
    import wells_e02x as WX

    page = io.repo_path(L.FORWARD)
    if io.sha256(page) != L.MODEL_SHA:
        raise io.EngineeringFailure("page forward sha256 differs from cf2c7154")
    mdir = Path(members_dir) if members_dir else io.repo_path(SOURCES[source])
    labels = thetas_labels()
    bad = source_check(source, mdir, labels)
    if bad:
        raise io.EngineeringFailure(f"run files differ from E02d's recorded sha256: {bad}")
    ext, prov = WX.ext_meta(prereg_dir)
    design, _ = WX.u1_design(prereg_dir)
    donors = WX.donors_table(design, ("station_id", "layer", "cell"))
    gp = io.repo_path(E02D_GATE)
    reg_gate = e02d_registered_gate(prereg_dir)
    if io.sha256(gp) != reg_gate:
        raise io.EngineeringFailure("E02d's members_gate.json differs from the sha E02d's "
                                    f"registration recorded ({reg_gate[:8]})")
    eg = json.loads(gp.read_text(encoding="utf-8"))
    mp = io.repo_path(E02D_MEMBERS)
    if not eg.get("ok") or io.sha256(mp) != eg.get("members_wells_sha256"):
        raise io.Refused("E02d's member gate did not pass (or members_wells.npz changed)")
    with np.load(mp, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    arrays, gate = extract(donors, ext, labels, mdir, page, ref)
    gate.update({"source": source, "members_dir": io.rel(mdir),
                 "gate_a_kind": ("identity: the donors are read from the same run files that "
                                 "produced members_wells.npz" if source == "e02d" else
                                 "reproduction: independent GPU reruns"),
                 "gate_b_kind": "the real check: external-cell member mean vs cf2c7154",
                 "e02d_members_gate_registered_sha256": reg_gate,
                 "inputs": {"page_forward": io.rec(page), "members_wells": io.rec(mp),
                            "e02d_members_gate": io.rec(gp), "ext_meta": prov,
                            "thetas_manifest": io.rec(io.repo_path(f"{THETAS}/manifest.json")),
                            "runs": {p.name: io.sha256(p)
                                     for p in sorted(mdir.glob("*.npz"))}}})
    out = io.repo_path(MEMBERS_NPZ)
    if out.exists():
        raise io.Refused(f"{out} exists (archive it before a re-extraction)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as f:
        np.savez_compressed(f, **arrays)
    gate["members_ext_sha256"] = io.sha256(out)
    io.write_once(io.repo_path(GATE_JSON), gate)
    return gate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--source", choices=sorted(SOURCES), default="e02d",
                   help="e02d (lead decision Q1, 0 GPU) or rerun (the GPU reruns)")
    e.add_argument("--members-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        gate = run_extract(a.source, a.members_dir)
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(json.dumps({k: gate[k] for k in ("ok", "source", "max_abs_donor_vs_members_wells_m",
                                           "max_abs_ext_mean_vs_page_m",
                                           "max_abs_hindcast_ic1_vs_ic0_m")}))
    return 0 if gate["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
