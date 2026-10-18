"""Port check: the generalised helpers in lib.py reproduce the billing pilot.

1. ``stagger_deconvolve`` with the pilot's population (s04: kept, on the active grid,
   irrigation, M2 & phase A / M1 & phase B) and the pilot's window (2007-01..2025-06)
   reproduces ``profiles_irrigation.npz`` (u_cls).
2. ``ChowLin`` reproduces the pilot's vectorised validation skill (s11: operator window
   2011-01..2023-12, pilot profile, no projection; 0.2257) and the group-size table of
   s08b (per-meter window 2012-01..2022-12; group_size_nearest_phase.json).

Writes ``_work_p2abc/portcheck.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402


def main() -> None:
    mt, M, pumps = lib.load_work()
    ph = lib.detect_phase(M, lib.VARIANTS["full"]["phase_window"])
    mt = pd.concat([mt, ph], axis=1)
    grid = lib.fan_grid()
    col = np.floor((mt["x"].to_numpy() - grid.x0) / grid.dx).astype(int)
    row = np.floor((mt["y"].to_numpy() - grid.y0) / grid.dx).astype(int)
    inb = (row >= 0) & (row < grid.ny) & (col >= 0) & (col < grid.nx)
    act = np.zeros(len(mt), bool)
    act[inb] = grid.mask[row[inb], col[inb]]
    keep = mt["kept"].to_numpy()
    irr = mt["cls"].to_numpy() == "irrigation"
    phase, bill = mt["phase"].to_numpy(), mt["bill"].to_numpy()
    selA = keep & act & irr & (bill == "M2") & (phase == lib.PHASE_A)
    selB = keep & act & irr & (bill == "M1") & (phase == lib.PHASE_B)
    T222 = 222
    Z = np.nan_to_num(M[:, :T222].astype(float))
    u = lib.stagger_deconvolve(Z[selA].sum(0), Z[selB].sum(0), t_end=T222)
    u_pilot = np.load(lib.PILOT / "profiles_irrigation.npz")["u_cls"]
    out = {"profile_max_rel_diff": float(np.max(np.abs(u - u_pilot)) / np.max(np.abs(u_pilot)))}

    # ---- s11 vectorised validation
    W0 = lib.tidx("2012-01")
    W1 = W0 + 132
    L0, L1 = W0 - 12, W1 + 12
    Mz = np.nan_to_num(M.astype(float))
    XW = M[:, W0:W1]
    finite_x = np.isfinite(mt["x"].to_numpy())
    val = keep & finite_x & irr & (phase == lib.MONTHLY) & np.isfinite(XW).all(1) & (np.nansum(XW, 1) > 0)
    vi = np.where(val)[0]
    bim = keep & finite_x & irr & np.isin(phase, [lib.PHASE_A, lib.PHASE_B])
    tree = cKDTree(mt[["x", "y"]].to_numpy()[bim])
    nph = phase[bim][tree.query(mt[["x", "y"]].to_numpy()[vi], k=1)[1]]
    ev = slice(W0 + 1, W1 - 1)
    sse = {"even": 0.0, "chowlin": 0.0}
    prof_pilot = np.concatenate([u_pilot, [u_pilot[-1]]])   # pad to 223 (unused month)
    for p_, lab in (("A", lib.PHASE_A), ("B", lib.PHASE_B)):
        j = vi[nph == lab]
        truth = Mz[j]
        ev_est = np.nan_to_num(lib.even_split(truth, p_, 0, T222))
        op = lib.ChowLin(p_, prof_pilot, L0, L1)
        cl = ev_est.copy()
        cl[:, L0:L1] = op.run(ev_est, project=False)
        sse["even"] += float(np.sum((ev_est[:, ev] - truth[:, ev]) ** 2))
        sse["chowlin"] += float(np.sum((cl[:, ev] - truth[:, ev]) ** 2))
    pilot_s11 = json.load(open(lib.PILOT / "chowlin_production.json"))["validation_skill"]
    out["s11_skill"] = 1 - sse["chowlin"] / sse["even"]
    out["s11_skill_pilot"] = pilot_s11
    out["s11_n"] = int(len(vi))

    # ---- s08b group sizes (per-meter window 2012-01..2022-12, nearest phase)
    idx = vi
    xy = mt[["x", "y"]].to_numpy()[idx]
    ph8 = np.where(nph == lib.PHASE_A, "A", "B")
    est = {m: np.zeros((len(idx), lib.T_ALL)) for m in ("even", "prorata", "chowlin")}
    for p_ in ("A", "B"):
        k = np.where(ph8 == p_)[0]
        x = Mz[idx[k]]
        est["even"][k] = np.nan_to_num(lib.even_split(x, p_, 0, T222))
        est["prorata"][k] = np.nan_to_num(lib.prorata(x, p_, prof_pilot, 0, T222))
        op = lib.ChowLin(p_, prof_pilot, W0, W1)
        cl = est["even"][k].copy()
        st = op.st
        t_lo, t_hi = int(st[0]), int(st[-1] + 2)
        res = op.run(x, project=False)
        cl[:, t_lo:t_hi] = res[:, t_lo - W0:t_hi - W0]
        est["chowlin"][k] = cl
    truth = Mz[idx]
    ev8 = slice(W0 + 1, W1 - 1)
    gs = {}
    for n in (1, 3, 10, 30, 100, 300, len(idx)):
        k_ = max(1, len(idx) // n)
        lab = (KMeans(n_clusters=k_, n_init=3, random_state=0).fit_predict(xy)
               if k_ > 1 else np.zeros(len(idx), int))
        s = {m: 0.0 for m in est}
        for gid in np.unique(lab):
            gi = np.where(lab == gid)[0]
            tr = truth[gi].sum(0)[ev8]
            for m in est:
                s[m] += float(np.sum((est[m][gi].sum(0)[ev8] - tr) ** 2))
        gs[str(n)] = {m: 1 - s[m] / s["even"] for m in ("prorata", "chowlin")}
    pilot_gs = json.load(open(lib.PILOT / "group_size_nearest_phase.json"))
    out["s08b"] = {n: {"mine": gs[n], "pilot": {m: pilot_gs[n][m] for m in ("prorata", "chowlin")}}
                   for n in gs}
    out["s08b_max_abs_diff"] = float(max(abs(gs[n][m] - pilot_gs[n][m])
                                         for n in gs for m in ("prorata", "chowlin")))
    lib.save_json(out, lib.WORK / "portcheck.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
