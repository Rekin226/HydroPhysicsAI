"""Build the kept-irrigation pump cache read by ``amp_inversion.load_pumps``.

Engineering fix (2026-10-25): ``amp_inversion.build_pump_cache`` saved the month labels as an
object array, which ``load_pumps`` (allow_pickle=False) refuses. This script writes the same
arrays with fixed-width string dtypes, so the registered ``amp_inversion.py`` is used
unchanged. Census only; no head data. Pass ``--basis dedup`` for the reported dedup-basis
variant (irrigation pumps on every meter, uncapped), written to a separate file.

Usage: python -I pump_cache.py [--basis kept|dedup]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_energy_volume import amp_inversion as A  # noqa: E402
from x5_energy_volume import census as C  # noqa: E402
from x5_energy_volume import x5const as K  # noqa: E402

DEDUP_CACHE = K.WORK / "dedup_irrigation_pumps_2010_2019.npz"


def build(basis: str = "kept") -> Path:
    K.WORK.mkdir(parents=True, exist_ok=True)
    t0 = C.tidx("2010-01")
    parts = []
    for cn in K.COUNTIES:
        p = C.load_pumps(cn)
        L = C.load_kwh(cn, len(p))
        E = C.bases(p, L)[basis][:, t0:t0 + len(A.MONTHS)]
        sel = ((p["purpose_class"] == "irrigation") & p["x_twd97"].notna()
               & p["y_twd97"].notna())
        if basis == "kept":
            sel &= p["kept"].astype(bool)
        sel = sel.to_numpy()
        parts.append({"x": p.loc[sel, "x_twd97"].to_numpy(dtype="float64"),
                      "y": p.loc[sel, "y_twd97"].to_numpy(dtype="float64"),
                      "cell": p.loc[sel, "cell"].to_numpy(dtype="int64"),
                      "county": np.full(sel.sum(), cn).astype("U8"),
                      "kw": (K.KW_PER_HP * pd.to_numeric(p.loc[sel, "pump_hp"], errors="coerce")
                             .fillna(0.0).clip(lower=0.0)).to_numpy(dtype="float64"),
                      "E": E[sel].astype("float32")})
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    path = A.PUMP_CACHE if basis == "kept" else DEDUP_CACHE
    np.savez_compressed(path, months=A.MONTHS.strftime("%Y-%m").to_numpy().astype("U7"), **out)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=("kept", "dedup"), default="kept")
    print(build(ap.parse_args().basis))
