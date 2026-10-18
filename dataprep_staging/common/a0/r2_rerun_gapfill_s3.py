"""WP A0 repair round: rerun the gap-fill pilot's fitted step (s3, family S) end to end.

The first-round gap-fill rerun recomputed s5b/s5c on the archived predictions only. This rerun
refits from the archived monthly matrix (out/monthly_long.parquet, out/well_meta.csv) and the
repository rain tables, from hash-checked copies of s3_pilot.py and lib.py:

* the two global Kalman MLE fits (no cached theta is provided, so s3 refits and saves
  kalman_theta_{rain,norain}.npy), compared with the archived thetas;
* the four single-well-gap realizations of family S (seeds 1000*ord('S')+r), every method,
  compared with the archived out/pilot_S.parquet.

One edit to the copy of s3_pilot.py: the GP call's ``n_jobs=2`` -> ``n_jobs=1`` (one CPU slot;
a parallel map, numerically neutral). The Kalman fits already run with ``threads=1``.

Usage (through tools/cpu_slot.sh, background): python -m common.a0.r2_rerun_gapfill_s3
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.a0.rerun_pilots import PY, RTOL, copy_script, link, run  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PILOTS, REPORTS_DIR, rel_to_repo  # noqa: E402

S = "head_gapfill_pilot"
OUT = REPORTS_DIR / "A0_evidence" / "repair" / "reruns" / "gapfill_s3_S"
GP = ('preds["gp"] = lib.m4_gp(Ym, rows=tw, n_jobs=2)',
      'preds["gp"] = lib.m4_gp(Ym, rows=tw, n_jobs=1)')


def _rel_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.isnan(a), np.isnan(b)
    m = ~na & ~nb
    den = np.maximum(np.abs(a[m]), np.abs(b[m]))
    d = np.abs(a[m] - b[m])
    rel = np.where(den > 0, d / np.where(den > 0, den, 1), 0.0)
    return {"n": int(m.sum()), "nan_mismatch": int((na != nb).sum()),
            "max_rel": float(rel.max()) if rel.size else 0.0,
            "max_abs": float(d.max()) if d.size else 0.0,
            "n_above_rtol": int((rel > RTOL).sum())}


def main() -> int:
    t0 = time.time()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; never overwritten")
    (OUT / "out").mkdir(parents=True)
    rec = {"pilot": S, "dir": rel_to_repo(OUT), "scripts": [
        copy_script(S, "lib.py", OUT), copy_script(S, "s3_pilot.py", OUT, patch=GP)]}
    for f in ("monthly_long.parquet", "well_meta.csv"):
        link(PILOTS / S / "out" / f, OUT / "out" / f)
    rec["step"] = run([PY, "s3_pilot.py", "S"], OUT, OUT / "s3_S.log")
    ref = PILOTS / S / "out"
    cmp = {}
    if rec["step"]["rc"] == 0:
        for tag in ("rain", "norain"):
            a = np.load(ref / f"kalman_theta_{tag}.npy")
            b = np.load(OUT / "out" / f"kalman_theta_{tag}.npy")
            cmp[f"kalman_theta_{tag}.npy"] = dict(_rel_stats(a, b), shape=list(a.shape))
        A = pd.read_parquet(ref / "pilot_S.parquet")
        B = pd.read_parquet(OUT / "out" / "pilot_S.parquet")
        res = {"rows_ref": len(A), "rows_new": len(B),
               "columns_equal": list(A.columns) == list(B.columns),
               "identical_frame": bool(A.equals(B))}
        if len(A) == len(B) and res["columns_equal"]:
            for c in A.columns:
                if pd.api.types.is_numeric_dtype(A[c]):
                    res[c] = _rel_stats(A[c].to_numpy(), B[c].to_numpy())
                else:
                    res[c] = {"text_mismatches": int((A[c].astype(str)
                                                      != B[c].astype(str)).sum())}
            if "method" in A.columns and "pred" in A.columns:
                per = {}
                for meth, g in A.groupby("method"):
                    h = B.loc[g.index]
                    per[str(meth)] = _rel_stats(g["pred"].to_numpy(), h["pred"].to_numpy())
                res["pred_by_method"] = per
        cmp["pilot_S.parquet"] = res
        # family-level RMSE by method, as the pilot log reports it
        for name, df in (("ref", A), ("new", B)):
            if {"method", "pred", "truth"} <= set(df.columns):
                e = (df["pred"] - df["truth"]) ** 2
                cmp[f"rmse_by_method_{name}"] = (e.groupby(df["method"]).mean() ** 0.5
                                                 ).round(6).to_dict()
        rec["new_pilot_S_sha256"] = sha256_file(OUT / "out" / "pilot_S.parquet")
    rec["comparisons"] = cmp
    rec["rtol"] = RTOL
    rec["seconds_total"] = round(time.time() - t0, 1)
    ps = cmp.get("pilot_S.parquet", {})
    rec["passed"] = bool(ps.get("identical_frame")) or (
        ps.get("rows_ref") == ps.get("rows_new") and all(
            v.get("n_above_rtol", 0) == 0 and v.get("nan_mismatch", 0) == 0
            and v.get("text_mismatches", 0) == 0
            for k, v in ps.items() if isinstance(v, dict) and k != "pred_by_method"))
    (OUT.parent / "gapfill_s3_S_result.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1, default=float) + "\n")
    print(json.dumps({"passed": rec["passed"], "rc": rec["step"]["rc"],
                      "seconds": rec["seconds_total"],
                      "theta": {k: v.get("max_rel") for k, v in cmp.items()
                                if k.startswith("kalman")}}, default=float))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
