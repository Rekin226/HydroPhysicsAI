"""Round-1 integrity check (logged as [P2-r1-repro] before it ran).

The interrupted round-1 session left its intermediates and scored results in
``results/dataprep/yunlin_electricity/_work_p2abc/`` with no hash record, and some round-1
code files were modified after some of their outputs were written (lib.py 10:46 local time,
after s2/s3 ran at 10:38/10:41). This reruns the unchanged round-1 scripts
(``s1_build``, ``s2_chowlin``, ``s3_validate``, ``s4_july``, ``s5_zeros``) into a scratch
work directory with their current code and compares every output with the stored one, so
the round-1 results quoted in the report and the round-1 arrays used by the registered
round-1 head check (P2a.head_placebo.v1) are known to follow from the archived code.

Usage: python -I r1_reproduce.py <scratch_dir> <out.json> [--compare-only]
(``--compare-only`` compares an existing rerun; the round-1 npz files hold an object-dtype
date array written by round 1 itself, so they are opened with allow_pickle=True.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
import lib  # noqa: E402

STORED = lib.WORK


def _cmp_json(a, b, path="", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ("seconds",):
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                _cmp_json(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _cmp_json(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (a == b or abs(a - b) <= tol * max(1.0, abs(a), abs(b))):
            out.append(f"{path}: {a} != {b}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def main(scratch: str, out_json: str, compare_only: bool = False) -> None:
    w = Path(scratch)
    w.mkdir(parents=True, exist_ok=True)
    lib.WORK = w
    import s1_build
    import s2_chowlin
    import s3_validate
    import s4_july
    import s5_zeros
    res = {"scratch": str(w), "stored": str(STORED), "files": {}}
    for mod in (s1_build, s2_chowlin, s3_validate, s4_july, s5_zeros):
        mod.lib.WORK = w
        if not compare_only:
            mod.main()
    res["compare_only_run"] = compare_only
    for p in sorted(STORED.iterdir()):
        q = w / p.name
        if not q.exists():
            res["files"][p.name] = "not produced by the rerun"
            continue
        if p.suffix == ".npy":
            a, b = np.load(p), np.load(q)
            same = a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
            res["files"][p.name] = {"identical": same,
                                    "max_abs_diff": None if same or a.shape != b.shape else
                                    float(np.nanmax(np.abs(a.astype(float) - b)))}
        elif p.suffix == ".npz":
            a, b = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
            diffs = [k for k in a.files if not np.array_equal(
                a[k], b[k], equal_nan=a[k].dtype.kind in "fc")]
            res["files"][p.name] = {"identical": not diffs, "differing_keys": diffs}
        elif p.suffix == ".parquet":
            a, b = pd.read_parquet(p), pd.read_parquet(q)
            try:
                pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
                res["files"][p.name] = {"identical": True}
            except AssertionError as e:
                res["files"][p.name] = {"identical": False, "msg": str(e)[:500]}
        elif p.suffix == ".json":
            a, b = json.loads(p.read_text()), json.loads(q.read_text())
            d = _cmp_json(a, b)
            res["files"][p.name] = {"identical": not d, "differences": d[:30],
                                    "n_differences": len(d)}
    res["all_identical"] = all(isinstance(v, dict) and v.get("identical")
                               for k, v in res["files"].items()
                               if k not in ("s3_validate.log",))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else v.get("identical"))
                      for k, v in res["files"].items()}, indent=1))
    print("all_identical", res["all_identical"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--compare-only" in sys.argv[3:])
