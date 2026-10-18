"""P2a step 2: billing phase per meter, fan-wide irrigation profile p(t) and production
Chow-Lin disaggregation of every kept bimonthly irrigation meter, per variant.

Variant ``preorigin``: phase detection on pairs in 2012-01..2022-12; the profile uses only
pairs complete by 2022-12; the Chow-Lin window is 2011-01..2022-12, so beta, rho and the
split never see a post-origin bill. The phase-B month 2022-12 (first half of a Dec-Jan
bill) is left unpaired and receives the Chow-Lin extrapolation (flagged 5 later).

Variant ``full``: phase detection 2012-01..2025-06; profile from pairs complete by 2025-07
(phase A's last pair is May-Jun 2025, phase B's Jun-Jul 2025); Chow-Lin windows 2011-01..
2025-06 (A) and 2011-01..2025-07 (B). Phase A's July 2025 is imputed later (P2b).

Writes ``_work_p2abc/<variant>_phase.parquet``, ``<variant>_profile.npz``,
``<variant>_cl.npy`` (float32 meters x 223, recorded values outside the Chow-Lin set and
windows) and ``<variant>_rho.npy``; ``chowlin_summary.json``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

BATCH = 20000


def active_meter(mt: pd.DataFrame) -> np.ndarray:
    """Meter (first-pump coordinates) inside an active cell (pilot s04 'act')."""
    grid = lib.fan_grid()
    x, y = mt["x"].to_numpy(), mt["y"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    col = np.where(ok, np.floor((np.nan_to_num(x) - grid.x0) / grid.dx), -1).astype(int)
    row = np.where(ok, np.floor((np.nan_to_num(y) - grid.y0) / grid.dx), -1).astype(int)
    inb = ok & (row >= 0) & (row < grid.ny) & (col >= 0) & (col < grid.nx)
    act = np.zeros(len(mt), bool)
    act[inb] = grid.mask[row[inb], col[inb]]
    return act


def first_last_nz(M: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray]:
    nz = np.nan_to_num(M[:, :end + 1]) > 0
    first = np.where(nz.any(1), nz.argmax(1), -1)
    last = np.where(nz.any(1), end - nz[:, ::-1].argmax(1), -1)
    return first, last


def profile_for(name: str, mt: pd.DataFrame, M: np.ndarray, phase: np.ndarray, act: np.ndarray):
    v = lib.VARIANTS[name]
    end = lib.tidx(v["end"])
    keep = mt["kept"].to_numpy()
    irr = mt["cls"].to_numpy() == "irrigation"
    bill = mt["bill"].to_numpy()
    selA = keep & act & irr & (bill == "M2") & (phase == lib.PHASE_A)
    selB = keep & act & irr & (bill == "M1") & (phase == lib.PHASE_B)
    Z = np.nan_to_num(M.astype(float))
    a, b = Z[selA].sum(0), Z[selB].sum(0)
    share_a = a.copy()
    share_b = b.copy()
    # the end month opens a pair whose bill is missing (index parity: even index = odd
    # calendar month = phase-A pair start); its recorded half-bill never enters the share
    if end % 2 == 0:
        share_a[end] = share_a[end - 1]
    else:
        share_b[end] = share_b[end - 1]
    u = lib.stagger_deconvolve(a, b, t_end=end + 1, share_a=share_a, share_b=share_b)
    prof = np.concatenate([u, np.full(lib.T_ALL - len(u), u[-1])])
    return prof, a, b, int(selA.sum()), int(selB.sum())


def windows(name: str) -> dict:
    """Chow-Lin window [lo, hi) per phase on DATES."""
    v = lib.VARIANTS[name]
    end = lib.tidx(v["end"])
    lo = lib.tidx(v["cl0"])
    # a phase whose pair would start at the end month cannot pair it: stop that phase's
    # window one month earlier so the incomplete month is not extrapolated twice
    hi = {"A": end + 1, "B": end + 1}
    if end % 2 == 0:            # end = odd calendar month -> phase A pair (end, end+1) open
        hi["A"] = end
    return {"A": (lo, hi["A"]), "B": (lo, hi["B"])}


def main() -> None:
    t_start = time.time()
    mt, M, _ = lib.load_work()
    act = active_meter(mt)
    keep = mt["kept"].to_numpy()
    irr = mt["cls"].to_numpy() == "irrigation"
    summary = {}
    for name, v in lib.VARIANTS.items():
        end = lib.tidx(v["end"])
        ph = lib.detect_phase(M, v["phase_window"])
        first, last = first_last_nz(M, end)
        tab = pd.concat([mt[["meter_id"]], ph], axis=1)
        tab["first_nz"], tab["last_nz"] = first, last
        tab["active_meter"] = act
        tab.to_parquet(lib.WORK / f"{name}_phase.parquet", index=False)
        phase = ph["phase"].to_numpy()
        prof, a, b, nA, nB = profile_for(name, mt, M, phase, act)
        np.savez(lib.WORK / f"{name}_profile.npz", prof=prof, a_month=a, b_month=b,
                 dates=lib.DATES.strftime("%Y-%m").to_numpy(), t_end=end + 1)
        W = windows(name)
        cl = M.copy()
        rho = np.full(len(mt), np.nan)
        info = {"profile_meters": {"A": nA, "B": nB}, "windows": {}}
        for p_, lab in (("A", lib.PHASE_A), ("B", lib.PHASE_B)):
            lo, hi = W[p_]
            op = lib.ChowLin(p_, prof, lo, hi)
            sel = np.where(keep & irr & (phase == lab))[0]
            for k in range(0, len(sel), BATCH):
                bidx = sel[k:k + BATCH]
                out, r = op.run(M[bidx].astype(float), return_rho=True)
                cl[bidx, lo:hi] = out.astype(np.float32)
                rho[bidx] = r
            info["windows"][p_] = {"lo": lib.ym(lo), "hi_inclusive": lib.ym(hi - 1),
                                   "n_pairs": op.K, "n_meters": int(len(sel)),
                                   "unpaired_months": [lib.ym(lo + i) for i in range(op.n)
                                                       if not np.isin(lo + i, np.r_[op.st, op.st + 1])]}
        np.save(lib.WORK / f"{name}_cl.npy", cl)
        np.save(lib.WORK / f"{name}_rho.npy", rho)
        o0 = lib.tidx(v["out0"])
        clset = keep & irr & np.isin(phase, [lib.PHASE_A, lib.PHASE_B])
        rec = np.nan_to_num(M[clset, o0:end + 1].astype(float))
        new = np.nan_to_num(cl[clset, o0:end + 1].astype(float))
        info.update({
            "phase_counts_by_bill": pd.crosstab(mt["bill"], ph["phase"]).to_dict(),
            "n_chowlin_meters": int(clset.sum()),
            "rho_share": {str(r_): float(np.mean(rho[clset] == r_)) for r_ in (0.0, 0.5, 0.9)},
            "mean_abs_change_over_energy": float(np.abs(new - rec).sum() / rec.sum()),
            "energy_change_rel": float((new.sum() - rec.sum()) / rec.sum()),
        })
        summary[name] = info
        print(name, {k: info[k] for k in ("n_chowlin_meters", "rho_share",
                                          "mean_abs_change_over_energy", "energy_change_rel")},
              flush=True)
    summary["seconds"] = round(time.time() - t_start, 1)
    lib.save_json(summary, lib.WORK / "chowlin_summary.json")


if __name__ == "__main__":
    main()
