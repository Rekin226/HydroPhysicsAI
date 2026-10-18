"""Re-validation of the port: methods.py against the pilot's lib.py on the L1 v3 matrix.

The pilot functions are imported read-only from the archived pilot directory (their
hash is checked against the pilot manifest first) and run on the same masked L1 pre-origin
matrix as the port. Every method must agree to 1e-9 (relative) on every cell. The mask is a
design-time realization (family S, r = 0). Writes ``port_check.json`` to the evidence dir.

Usage: python -I port_check.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from harness import gaps_guarded  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import io_h2, methods  # noqa: E402

PILOT = C.REPO / "results/dataprep/_pilots/head_gapfill_pilot"


def load_pilot_lib():
    man = json.loads((PILOT / "MANIFEST.json").read_text())
    want = None
    for sect in ("code", "outputs", "files"):
        for r in man.get(sect, []) or []:
            if isinstance(r, dict) and str(r.get("path", "")).endswith("lib.py"):
                want = r.get("sha256")
    got = hashlib.sha256((PILOT / "lib.py").read_bytes()).hexdigest()
    if want is not None and want != got:
        raise RuntimeError(f"pilot lib.py hash {got} differs from its manifest {want}")
    spec = importlib.util.spec_from_file_location("pilot_lib", PILOT / "lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, got, want


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    if not both.any():
        return 0.0
    return float(np.max(np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1.0)))


def main():
    lib, got, want = load_pilot_lib()
    h = io_h2.load("pre", with_daily=False)
    Y = h.Y
    obs = np.isfinite(Y)
    targets = np.where(obs.mean(1) >= C.TARGET_MIN_COV)[0]
    r, mask, cells = next(gaps_guarded.head_masks("S", obs, targets, h.months, nreal=1))
    Ym = np.where(mask, np.nan, Y)
    xy = h.xy()
    lay = h.meta.layer.to_numpy()
    out = {"pilot_lib_sha256": got, "pilot_manifest_sha256": want, "mask": "S r=0",
           "cells_masked": int(mask.sum()), "methods": {}}
    pairs = {
        "climatology": (methods.climatology(Ym), lib.climatology(Ym)),
        "m0_clim": (methods.m0_clim(Ym), lib.m0_clim(Ym)),
        "m0_clim_ar": (methods.m0_clim_ar(Ym), lib.m0_clim_ar(Ym)),
        "m1_neighbour": (methods.m1_neighbour(Ym, xy=xy), lib.m1_neighbour(Ym, xy=xy)),
        "m6_idw_anom": (methods.m6_idw_anom(Ym, xy=xy, layers=lay),
                        lib.m6_idw_anom(Ym, xy=xy, layers=lay)),
        "m2_softimpute_tb": (methods.m2_softimpute_tb(Ym), lib.m2_softimpute_tb(Ym)),
        "backfill_vs_m0_linear": (methods.backfill(Ym), lib.m0_linear(Ym)),
    }
    for name, (a, b) in pairs.items():
        if isinstance(a, tuple):
            d = max(rel(x, y) for x, y in zip(a, b, strict=True) if x is not None)
        else:
            d = rel(a, b)
        out["methods"][name] = d
    rows = [int(i) for i in sorted({c[2] for c in cells})][:6]
    ga = methods.m4_gp(Ym, rows=rows)
    gb = lib.m4_gp(Ym, rows=rows, n_jobs=1)
    out["methods"]["m4_gp_6wells"] = max(rel(ga[0][rows], gb[0][rows]),
                                         rel(ga[1][rows], gb[1][rows]))
    out["max_rel"] = max(out["methods"].values())
    out["pass"] = out["max_rel"] <= 1e-9
    C.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (C.EVIDENCE_DIR / "port_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
