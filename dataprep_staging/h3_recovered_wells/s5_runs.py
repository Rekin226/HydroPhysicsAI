"""Step 5: artificial-gap runs of the registered test ``H3.d_wells`` (family d, r = 100..159).

H2's runner is used unchanged (``h2_heads_fill.runner.run_daily`` on the pre-origin matrix
of ``h2_heads_fill.io_h2.load('pre')``): one random daily gap of 5-60 days per eligible well
per realization (``bridge.d_windows``, seed ``1000*ord('d') + r``), the frozen daily bridge,
and the monthly comparators fitted with the touched well-months masked. Every eligible well is
masked exactly as in H2's family-d tests; only the rows of the H3 test wells are kept (the
other wells' errors are neither stored nor looked at).

H2's runner caches its base matrix in ``H2_WORKDIR``; this script points that at
``H3_evidence/h2_workdir`` so H2's own work directory is never touched, and compares the
rebuilt base matrix with H2's archived one (``H2_evidence/run_tables/base_pre.npz``).

Usage: python -I s5_runs.py [R ...]        (default: config.FRESH_D)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402

os.environ["H2_WORKDIR"] = str(C.H2_WORKDIR)            # before h2_heads_fill.config loads

import numpy as np  # noqa: E402
from h2_heads_fill import config as H2C  # noqa: E402
from h2_heads_fill import io_h2, runner  # noqa: E402


def test_wells() -> list[str]:
    reg = json.loads((C.EVIDENCE / "registered_wells.json").read_text())
    return reg["test_wells"]


def main(argv):
    assert H2C.WORKDIR == C.H2_WORKDIR, (H2C.WORKDIR, C.H2_WORKDIR)
    rs = [int(x) for x in argv] if argv else list(C.FRESH_D)
    if any(r not in C.FRESH_D for r in rs):
        raise SystemExit(f"realizations outside the registered range {C.FRESH_D}")
    wells = set(test_wells())
    h, Yb, _ = runner.load_base("pre")
    arch = np.load(C.H2_EVIDENCE / "run_tables" / "base_pre.npz", allow_pickle=False)
    same = (list(arch["sids"]) == h.sids
            and np.array_equal(arch["Ybase"], Yb, equal_nan=True))
    clean = io_h2.clean_truth(h)
    C.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (C.RUN_DIR / "base_check.json").write_text(json.dumps(
        {"base_matrix_identical_to_H2_archive": bool(same), "n_wells": len(h.sids),
         "test_wells_in_matrix": sorted(wells & set(h.sids))}, indent=1))
    if not same:
        raise SystemExit("rebuilt base matrix differs from H2's archived base_pre.npz")
    for r in rs:
        tic = time.time()
        df, info = runner.run_daily("pre", "d", r, h, Yb, clean)
        keep = df[df.sid.isin(wells)].copy()
        keep.to_parquet(C.RUN_DIR / f"d_r{r}.parquet")
        info = {"n_windows_all_wells": info.get("n_windows"),
                "n_cells_all_wells": info.get("n_cells"), "n_cells_kept": int(len(keep)),
                "n_windows_kept": int(keep.groupby(["sid", "win_start"]).ngroups)
                if len(keep) else 0, "seconds": time.time() - tic}
        (C.RUN_DIR / f"d_r{r}.json").write_text(json.dumps(info, indent=1, default=float))
        print(f"d r{r}: kept {info['n_cells_kept']} cells, {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
