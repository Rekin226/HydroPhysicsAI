"""WP A0 step 4: spot-rerun three pilots from their archived code and compare key values to a
relative 1e-6 (plan A0 validation).

  tpc       tools/tpc_county.py (on the L0 copy of d007012) -> census_totals.py (on the L0
            census through hydrophysics.twin.pumping.clean_census) -> tools/compare.py;
            compares compare_annual.csv (and the two intermediate tables)
  pseudo    d_pseudo_changhua.py with inv_lib.py on the archived inputs (inputs_cwa.npz,
            hindcast_cwa.npz, K_pump_f32.npy, M_cwa.npz); compares d_pseudo_changhua.json
  gapfill   s5b_cleantruth.py (audit path re-pointed to _pilots/audit_2026-10-11, the only
            edit) -> s5c_ensemble.py with lib.py on the archived pilot predictions; compares
            out/pilot_final_clean.csv (and pilot_summary_cleantruth.csv, pilot_all_scored)

Each rerun runs in its own directory under results/dataprep/reports/A0_evidence/reruns/
with copies of the scripts (sha256 checked against the pilot manifest) and symlinks to the
inputs, so nothing in _pilots/ is written. Run through dataprep_staging/tools/cpu_slot.sh.

Usage: python -m common.a0.rerun_pilots {tpc,pseudo,gapfill,compare}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.hashing import sha256_file  # noqa: E402
from common.manifest import load_manifest  # noqa: E402
from common.paths import L0_NEW, PILOTS, REPORTS_DIR, rel_to_repo  # noqa: E402

PY = sys.executable                      # the hydro env interpreter running this harness
RERUNS = REPORTS_DIR / "A0_evidence" / "reruns"
RTOL = 1e-6
_AUD_LINE = re.compile(r'^AUD = "[^"]*"$', re.M)


# ----------------------------------------------------------------------------- helpers
def _manifest_hash(scout: str, rel: str) -> str:
    m, _ = load_manifest(PILOTS / scout)
    want = f"results/dataprep/_pilots/{scout}/{rel}"
    for r in m["code"] + [dict(o, path=f"results/dataprep/_pilots/{scout}/{o['path']}")
                          for o in m["outputs"]]:
        if r["path"] == want:
            return r["sha256"]
    raise KeyError(want)


def copy_script(scout: str, rel: str, dest_dir: Path, patch: tuple[str, str] | None = None
                ) -> dict:
    """Copy a pilot script; verify its hash against the pilot manifest; optional one-string
    patch (recorded)."""
    src = PILOTS / scout / rel
    want = _manifest_hash(scout, rel)
    got = sha256_file(src)
    if got != want:
        raise RuntimeError(f"{src} differs from its manifest")
    dest = dest_dir / Path(rel).name
    text = src.read_text(encoding="utf-8")
    rec = {"script": f"_pilots/{scout}/{rel}", "sha256": got, "patched": False}
    if patch:
        old, new = patch
        if text.count(old) != 1:
            raise RuntimeError(f"patch target not unique in {src}")
        text = text.replace(old, new)
        rec.update(patched=True, patch={"old": old, "new": new})
    dest.write_text(text, encoding="utf-8")
    rec["run_copy_sha256"] = sha256_file(dest)
    return rec


def link(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src)


def run(cmd: list[str], cwd: Path, log: Path) -> dict:
    t0 = time.time()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return {"cmd": " ".join(cmd), "rc": r.returncode, "seconds": round(time.time() - t0, 1),
            "log": rel_to_repo(log)}


def fresh(name: str) -> Path:
    d = RERUNS / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


# ----------------------------------------------------------------------------- comparisons
def _rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    den = max(abs(a), abs(b))
    return 0.0 if den < 1e-300 else abs(a - b) / den


def compare_csv(ref: Path, new: Path, index_col=None) -> dict:
    A = pd.read_csv(ref, index_col=index_col)
    B = pd.read_csv(new, index_col=index_col)
    out = {"ref": rel_to_repo(ref), "new": rel_to_repo(new), "shape_ref": list(A.shape),
           "shape_new": list(B.shape), "identical_bytes": sha256_file(ref) == sha256_file(new)}
    if A.shape != B.shape or list(A.columns) != list(B.columns):
        out.update(ok=False, problem="shape or columns differ")
        return out
    if index_col is not None and not A.index.equals(B.index):
        out.update(ok=False, problem="index differs")
        return out
    maxrel, n_num, n_bad, n_nan_mismatch, n_str_mismatch = 0.0, 0, 0, 0, 0
    worst = None
    for c in A.columns:
        a, b = A[c], B[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            na, nb = np.isnan(av), np.isnan(bv)
            n_nan_mismatch += int((na != nb).sum())
            m = ~na & ~nb
            for i in np.flatnonzero(m):
                r = _rel(av[i], bv[i])
                n_num += 1
                if r > RTOL:
                    n_bad += 1
                if r > maxrel:
                    maxrel, worst = r, {"column": c, "row": int(i), "ref": av[i], "new": bv[i]}
        else:
            n_str_mismatch += int((a.astype(str) != b.astype(str)).sum())
    out.update(n_numeric=n_num, max_rel_diff=maxrel, n_above_rtol=n_bad,
               nan_pattern_mismatches=n_nan_mismatch, text_mismatches=n_str_mismatch,
               worst=worst, ok=(n_bad == 0 and n_nan_mismatch == 0 and n_str_mismatch == 0))
    return out


def _walk(a, b, path, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            acc["structure"].append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            _walk(a[k], b[k], f"{path}/{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["structure"].append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):   # length checked above
            _walk(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or \
            isinstance(a, str) or isinstance(b, str):
        if a != b and not (isinstance(a, float) and isinstance(b, float)
                           and math.isnan(a) and math.isnan(b)):
            acc["structure"].append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
            if not (isinstance(a, float) and isinstance(b, float) and math.isnan(a)
                    and math.isnan(b)):
                acc["structure"].append(f"{path}: NaN mismatch {a} vs {b}")
            return
        r = _rel(float(a), float(b))
        acc["n"] += 1
        if r > RTOL:
            acc["bad"].append({"path": path, "ref": a, "new": b, "rel": r})
        if r > acc["max"]:
            acc["max"], acc["worst"] = r, {"path": path, "ref": a, "new": b}
    else:
        acc["structure"].append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")


def compare_json(ref: Path, new: Path) -> dict:
    a = json.loads(ref.read_text())
    b = json.loads(new.read_text())
    acc = {"n": 0, "max": 0.0, "worst": None, "bad": [], "structure": []}
    _walk(a, b, "", acc)
    return {"ref": rel_to_repo(ref), "new": rel_to_repo(new),
            "identical_bytes": sha256_file(ref) == sha256_file(new), "n_numeric": acc["n"],
            "max_rel_diff": acc["max"], "worst": acc["worst"], "n_above_rtol": len(acc["bad"]),
            "above_rtol": acc["bad"][:20], "structure_mismatches": acc["structure"][:20],
            "ok": not acc["bad"] and not acc["structure"]}


# ----------------------------------------------------------------------------- reruns
def rerun_tpc() -> dict:
    s = "tpc_aggregate_electricity"
    d = fresh("tpc")
    rec = {"pilot": s, "dir": rel_to_repo(d), "scripts": [], "steps": []}
    for rel in ("tools/tpc_county.py", "tools/compare.py", "census_totals.py"):
        rec["scripts"].append(copy_script(s, rel, d))
    l0 = L0_NEW / "tpc_d007012" / "2026-10-12" / "d007012.csv"
    rec["l0_input"] = {"path": rel_to_repo(l0), "sha256": sha256_file(l0)}
    rec["steps"].append(run([PY, "tpc_county.py", str(l0), "tpc_county_sector_monthly.csv"],
                            d, d / "tpc_county.log"))
    rec["steps"].append(run([PY, "census_totals.py"], d, d / "census_totals.log"))
    rec["steps"].append(run([PY, "compare.py", "tpc_county_sector_monthly.csv",
                             "census_month.csv", "."], d, d / "compare.log"))
    P = PILOTS / s
    rec["comparisons"] = {
        "compare_annual.csv": compare_csv(P / "compare_annual.csv", d / "compare_annual.csv"),
        "tpc_county_sector_monthly.csv": compare_csv(P / "tpc_county_sector_monthly.csv",
                                                     d / "tpc_county_sector_monthly.csv"),
        "census_month.csv": compare_csv(P / "census_month.csv", d / "census_month.csv"),
        "compare_monthly.csv": compare_csv(P / "compare_monthly.csv", d / "compare_monthly.csv"),
    }
    rec["key"] = "compare_annual.csv"
    return rec


def rerun_pseudo() -> dict:
    s = "physics_inversion_hooks"
    d = fresh("pseudo")
    rec = {"pilot": s, "dir": rel_to_repo(d), "scripts": [], "steps": []}
    for rel in ("d_pseudo_changhua.py", "inv_lib.py"):
        rec["scripts"].append(copy_script(s, rel, d))
    rec["inputs"] = []
    for f in ("inputs_cwa.npz", "hindcast_cwa.npz", "K_pump_f32.npy", "M_cwa.npz"):
        src = PILOTS / s / f
        if sha256_file(src) != _manifest_hash(s, f):
            raise RuntimeError(f"{src} differs from its manifest")
        link(src, d / f)
        rec["inputs"].append({"path": f"_pilots/{s}/{f}", "linked": True})
    rec["steps"].append(run([PY, "d_pseudo_changhua.py", str(d)], d, d / "d_pseudo.log"))
    rec["comparisons"] = {"d_pseudo_changhua.json": compare_json(
        PILOTS / s / "d_pseudo_changhua.json", d / "d_pseudo_changhua.json")}
    rec["key"] = "d_pseudo_changhua.json"
    return rec


def rerun_gapfill() -> dict:
    s = "head_gapfill_pilot"
    d = fresh("gapfill")
    out = d / "out"
    out.mkdir()
    new_audit = str(PILOTS / "audit_2026-10-11")
    old_line = _AUD_LINE.search((PILOTS / s / "s5b_cleantruth.py").read_text()).group(0)
    rec = {"pilot": s, "dir": rel_to_repo(d), "steps": [], "scripts": [
        copy_script(s, "lib.py", d),
        copy_script(s, "s5b_cleantruth.py", d,
                    patch=(old_line, f'AUD = "{new_audit}"')),
        copy_script(s, "s5c_ensemble.py", d)]}
    names = ["monthly_long.parquet", "well_meta.csv"] + [
        f"pilot_{x}{fam}.parquet" for fam in "NSEH" for x in ("", "ext_")]
    for f in names:
        link(PILOTS / s / "out" / f, out / f)
    rec["inputs"] = [f"_pilots/{s}/out/{f}" for f in names] + [
        "_pilots/audit_2026-10-11/artefact_events.csv"]
    rec["steps"].append(run([PY, "s5b_cleantruth.py"], d, d / "s5b.log"))
    rec["steps"].append(run([PY, "s5c_ensemble.py"], d, d / "s5c.log"))
    P = PILOTS / s / "out"
    rec["comparisons"] = {
        "pilot_final_clean.csv": compare_csv(P / "pilot_final_clean.csv",
                                             out / "pilot_final_clean.csv"),
        "pilot_final_all.csv": compare_csv(P / "pilot_final_all.csv", out / "pilot_final_all.csv"),
        "pilot_summary_cleantruth.csv": compare_csv(P / "pilot_summary_cleantruth.csv",
                                                    out / "pilot_summary_cleantruth.csv"),
        "offset_episodes_hist.csv": compare_csv(P / "offset_episodes_hist.csv",
                                                out / "offset_episodes_hist.csv"),
    }
    a = pd.read_parquet(P / "pilot_all_scored.parquet")
    b = pd.read_parquet(out / "pilot_all_scored.parquet")
    rec["comparisons"]["pilot_all_scored.parquet"] = {
        "ok": bool(a.equals(b)), "rows": len(a), "identical_frame": bool(a.equals(b))}
    rec["key"] = "pilot_final_clean.csv"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["tpc", "pseudo", "gapfill"])
    a = ap.parse_args()
    t0 = time.time()
    rec = {"tpc": rerun_tpc, "pseudo": rerun_pseudo, "gapfill": rerun_gapfill}[a.which]()
    rec["seconds_total"] = round(time.time() - t0, 1)
    rec["rtol"] = RTOL
    rec["passed"] = all(st["rc"] == 0 for st in rec["steps"]) and bool(
        rec["comparisons"][rec["key"]]["ok"])
    p = RERUNS / f"{a.which}_result.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float) + "\n")
    k = rec["comparisons"][rec["key"]]
    print(json.dumps({"which": a.which, "passed": rec["passed"], "key": rec["key"],
                      "max_rel_diff": k.get("max_rel_diff"), "n_numeric": k.get("n_numeric"),
                      "identical_bytes": k.get("identical_bytes"),
                      "steps": [(st["rc"], st["seconds"]) for st in rec["steps"]]}, default=float))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
