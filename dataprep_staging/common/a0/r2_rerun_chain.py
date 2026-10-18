"""WP A0 repair round: end-to-end rerun of the pseudo-Changhua chain of physics_inversion_hooks.

The first-round rerun recomputed d_pseudo_changhua.json from the archived intermediate
arrays only. This rerun regenerates every intermediate from the repository data, from copies
of the archived scripts (sha256 checked against the pilot manifest):

  a_build_inputs.py   twin inputs via hydrophysics.twin.inputs (read-only)   -> inputs_cwa.npz
  b_operator.py       operator and hindcast from the stage-3 theta (CPU torch) -> M_cwa.npz,
                                                                                hindcast_cwa.npz
  c_greens.py         exact Green's functions (sparse LU, adjoint rollouts)  -> K_pump_f32.npy,
                                                                                G0_f32.npy
  d_pseudo_changhua.py + inv_lib.py                                          -> d_pseudo_changhua.json

Edits to the copies (each recorded in the result):
* ``torch.set_num_threads(4)`` -> ``torch.set_num_threads(1)`` in a and b (the CPU-slot budget
  allows one thread; threading changes floating-point summation order only);
* in a, ``cell_county=cell_county`` -> ``cell_county=cell_county.astype(str)`` when saving.
  Attempt 1 (evidence in reruns/physics_chain_attempt1*) ran the archived a unchanged: every
  array of inputs_cwa.npz came out value-identical, but cell_county was saved as an object
  array, which b_operator.py (np.load with allow_pickle=False) refuses. The archived
  inputs_cwa.npz holds cell_county as <U3 and was written at 00:43:51, after a_build_inputs.py
  (00:41:19) and b_operator.py (00:43:08) were saved, so the scout converted the column in a
  step that was not archived. The patch reproduces that conversion.
CUDA_VISIBLE_DEVICES is empty. Arrays are compared key by key (scale-relative
max abs difference and elementwise relative difference above a floor of 1e-9 x max|ref|);
JSON files with the first-round comparator (relative 1e-6 per value).

Usage (through tools/cpu_slot.sh): python -m common.a0.r2_rerun_chain
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from common.a0.rerun_pilots import PY, RTOL, compare_json, copy_script, link, run  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PILOTS, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

S = "physics_inversion_hooks"
OUT = REPORTS_DIR / "A0_evidence" / "repair" / "reruns" / "physics_chain"
THREADS = ("torch.set_num_threads(4)", "torch.set_num_threads(1)")
DTYPE = ("cell_county=cell_county,", "cell_county=cell_county.astype(str),")
FLOOR = 1e-9


def _cmp_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"ok": False, "problem": f"shape {a.shape} != {b.shape}"}
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":
        eq = bool(np.array_equal(a.astype(str), b.astype(str)))
        return {"ok": eq, "kind": "text", "identical": eq}
    av, bv = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.isnan(av), np.isnan(bv)
    if not np.array_equal(na, nb):
        return {"ok": False, "problem": "NaN pattern differs", "n": int((na != nb).sum())}
    m = ~na
    if not m.any():
        return {"ok": True, "kind": "all-nan"}
    d = np.abs(av[m] - bv[m])
    scale = float(np.max(np.abs(av[m])))
    den = np.maximum(np.abs(av[m]), np.abs(bv[m]))
    sel = den > FLOOR * max(scale, 1e-300)
    rel = d[sel] / den[sel] if sel.any() else np.zeros(1)
    out = {"kind": str(a.dtype), "identical": bool(np.array_equal(av, bv)),
           "max_abs_diff": float(d.max()), "max_abs_ref": scale,
           "scale_rel_diff": float(d.max() / scale) if scale > 0 else float(d.max()),
           "max_elem_rel_diff": float(rel.max()), "n": int(m.sum())}
    out["ok"] = out["max_elem_rel_diff"] <= RTOL
    return out


def compare_npz(ref: Path, new: Path) -> dict:
    with np.load(ref, allow_pickle=False) as A, np.load(new, allow_pickle=False) as B:
        keys = sorted(set(A.files) | set(B.files))
        res = {}
        for k in keys:
            if k not in A.files or k not in B.files:
                res[k] = {"ok": False, "problem": "key missing in one file"}
            else:
                res[k] = _cmp_arrays(A[k], B[k])
    return {"ref": rel_to_repo(ref), "new": rel_to_repo(new),
            "identical_bytes": sha256_file(ref) == sha256_file(new),
            "ok": all(v["ok"] for v in res.values()), "arrays": res}


def compare_sparse(ref: Path, new: Path) -> dict:
    a, b = sp.load_npz(ref).tocsr(), sp.load_npz(new).tocsr()
    if a.shape != b.shape:
        return {"ok": False, "problem": "shape differs"}
    diff = (a - b)
    dmax = float(abs(diff).max()) if diff.nnz else 0.0
    scale = float(abs(a).max())
    return {"ref": rel_to_repo(ref), "new": rel_to_repo(new), "nnz_ref": int(a.nnz),
            "nnz_new": int(b.nnz), "max_abs_diff": dmax, "scale_rel_diff": dmax / scale,
            "ok": dmax / scale <= RTOL}


def compare_npy(ref: Path, new: Path) -> dict:
    a = np.load(ref, mmap_mode="r")
    b = np.load(new, mmap_mode="r")
    r = _cmp_arrays(np.asarray(a), np.asarray(b))
    r.update(ref=rel_to_repo(ref), new=rel_to_repo(new),
             identical_bytes=sha256_file(ref) == sha256_file(new))
    return r


def main() -> int:
    t0 = time.time()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; never overwritten (move it aside to rerun)")
    OUT.mkdir(parents=True)
    rec = {"pilot": S, "dir": rel_to_repo(OUT), "scripts": [], "steps": []}
    a_rec = copy_script(S, "a_build_inputs.py", OUT, patch=THREADS)
    a_path = OUT / "a_build_inputs.py"
    text = a_path.read_text(encoding="utf-8")
    if text.count(DTYPE[0]) != 1:
        raise RuntimeError("dtype patch target not unique")
    a_path.write_text(text.replace(DTYPE[0], DTYPE[1]), encoding="utf-8")
    a_rec["patch2"] = {"old": DTYPE[0], "new": DTYPE[1]}
    a_rec["run_copy_sha256"] = sha256_file(a_path)
    rec["scripts"].append(a_rec)
    rec["scripts"].append(copy_script(S, "b_operator.py", OUT, patch=THREADS))
    for rel in ("c_greens.py", "d_pseudo_changhua.py", "inv_lib.py"):
        rec["scripts"].append(copy_script(S, rel, OUT))
    theta = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
    rec["theta"] = {"path": rel_to_repo(theta), "sha256": sha256_file(theta)}
    steps = [("a_build_inputs.py", "a.log"), ("b_operator.py", "b.log"),
             ("c_greens.py", "c.log"), ("d_pseudo_changhua.py", "d.log")]
    for script, log in steps:
        st = run([PY, script, str(OUT)], OUT, OUT / log)
        rec["steps"].append(st)
        print(script, st["rc"], st["seconds"], flush=True)
        if st["rc"] != 0:
            break
    P = PILOTS / S
    cmp = {}
    if all(st["rc"] == 0 for st in rec["steps"]) and len(rec["steps"]) == len(steps):
        cmp["inputs_cwa.npz"] = compare_npz(P / "inputs_cwa.npz", OUT / "inputs_cwa.npz")
        cmp["hindcast_cwa.npz"] = compare_npz(P / "hindcast_cwa.npz", OUT / "hindcast_cwa.npz")
        cmp["M_cwa.npz"] = compare_sparse(P / "M_cwa.npz", OUT / "M_cwa.npz")
        cmp["K_pump_f32.npy"] = compare_npy(P / "K_pump_f32.npy", OUT / "K_pump_f32.npy")
        cmp["G0_f32.npy"] = compare_npy(P / "G0_f32.npy", OUT / "G0_f32.npy")
        cmp["c_residual_by_county.json"] = compare_json(P / "c_residual_by_county.json",
                                                        OUT / "c_residual_by_county.json")
        cmp["d_pseudo_changhua.json"] = compare_json(P / "d_pseudo_changhua.json",
                                                     OUT / "d_pseudo_changhua.json")
        # the archived intermediates are kept for reference; drop the regenerated 2 x 178 MB
        # kernels after comparison (their hashes are recorded)
        for big in ("K_pump_f32.npy", "G0_f32.npy"):
            f = OUT / big
            cmp[big]["new_sha256"] = sha256_file(f)
            cmp[big]["new_bytes"] = f.stat().st_size
            f.unlink()
            link(P / big, OUT / f"{big}.archived_link")
    rec["comparisons"] = cmp
    rec["key"] = "d_pseudo_changhua.json"
    rec["rtol"] = RTOL
    rec["seconds_total"] = round(time.time() - t0, 1)
    rec["passed"] = bool(cmp) and bool(cmp["d_pseudo_changhua.json"]["ok"])
    rec["all_intermediates_ok"] = bool(cmp) and all(v.get("ok") for v in cmp.values())
    res = OUT.parent / "physics_chain_result.json"
    res.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float) + "\n")
    k = cmp.get("d_pseudo_changhua.json", {})
    print(json.dumps({"passed": rec["passed"], "intermediates_ok": rec["all_intermediates_ok"],
                      "max_rel_diff": k.get("max_rel_diff"), "n_numeric": k.get("n_numeric"),
                      "steps": [(s["rc"], s["seconds"]) for s in rec["steps"]]}, default=float))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
