"""WP S4b repair round: network-gap structure of the LiCSBAS runs P, F and D.

LiCSBAS13 (NSBAS, gamma 1e-4) inverts the increments of every pixel with a weak constraint
that the cumulative displacement follows a linear velocity. Where no valid interferogram spans
an epoch interval (a gap), the increment over that interval is set by that linear-velocity
constraint alone, not by any interferogram. LiCSBAS stores the per-pixel gap intervals in
``cum.h5['gap']`` (n_im - 1, L, W) and the network-level gaps in
``info/12network_gap_info.txt``.

This script reads only that structure (no displacement, coherence, GNSS or census value; look
logged by ``s4b_r2_looks.py repair_gap_meta``) and writes, per run:

  s4b_run/gaps_<run>.npz          network gaps and segments; per-pixel gap intervals; per-cell
                                  gap intervals (any contributing pixel) and pixel fractions;
                                  per-cell segment id per epoch
  S4b_evidence/repair_gaps.json   summary (what the report quotes)

Cell rule (fixed here from the structure alone): a cell has a gap at interval k when ANY of its
contributing pixels (inside the LiCSBAS15 mask) has one; the cell's segment id at epoch t is
the number of cell gaps before t. A value whose segment id is > 0 includes at least one
increment bridged by the linear-velocity constraint (product flag 4); a change between two
epochs is interferometric only when both carry the same segment id.

    python -I s4b_r2_gaps.py      (hydro env, one CPU slot)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402


def network_info(name: str) -> dict:
    p = C.RUN_DIR / name / "TS_GEOCml2" / "info" / "12network_gap_info.txt"
    gaps, segs, mode = [], [], None
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("Gaps between"):
            mode = "g"
            continue
        if s.startswith("Connected network"):
            mode = "s"
            continue
        if not s:
            continue
        if mode == "g":
            a, b = s.split()[:2]
            gaps.append([a, b])
        elif mode == "s":
            span, rest = s.split(" ", 1)
            a, b = span.split("-")
            n = int(rest.strip("()").split(",")[1])
            segs.append({"first": a, "last": b, "n_epochs": n})
    return {"gaps": gaps, "segments": segs}


def run_gaps(name: str, A: int) -> dict:
    ts = C.RUN_DIR / name / "TS_GEOCml2"
    with h5py.File(ts / "cum.h5", "r") as f:
        gap = f["gap"][()].astype(bool)                       # (T-1, L, W)
        n_gap = f["n_gap"][()]
        dates = [str(d) for d in f["imdates"][()]]
    L, W = gap.shape[1:]
    m = np.fromfile(ts / "results" / "mask", dtype=np.float32).reshape(L, W)
    mask = np.isfinite(m) & (m == 1)
    g = Lb.load_geom(Lb.run_frame(name))
    cell = g["cell"]
    net = network_info(name)
    # network-level gap intervals (index k = interval between epochs k and k+1)
    k_net = [dates.index(a) for a, b in net["gaps"]]
    for (a, b), k in zip(net["gaps"], k_net, strict=True):
        assert dates[k + 1] == b, (name, a, b)
    seg_net = np.zeros(len(dates), int)
    for k in k_net:
        seg_net[k + 1:] += 1
    act = mask & (cell >= 0)
    pix = gap[:, act]                                         # (T-1, P)
    extra = pix.copy()
    extra[k_net] = False
    # per cell (A, T-1): any contributing pixel with a gap; fraction of pixels with one
    cid = cell[act]
    cnt = np.bincount(cid, minlength=A)
    gsum = np.stack([np.bincount(cid, weights=pix[k].astype(float), minlength=A)
                     for k in range(pix.shape[0])], 1)        # (A, T-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(cnt[:, None] > 0, gsum / cnt[:, None], np.nan)
    cell_gap = gsum > 0
    has = cnt >= C.CELL_MIN_PIXELS
    seg_cell = np.concatenate([np.zeros((A, 1), int), np.cumsum(cell_gap, 1)], 1)
    seg_cell[~has] = -1
    pix_seg = np.concatenate([np.zeros((1, L, W), np.int16),
                              np.cumsum(gap, 0).astype(np.int16)], 0)
    pix_seg[:, ~mask] = -1
    np.savez_compressed(
        C.RUN_DIR / f"gaps_{name}.npz", dates=np.array(dates), net_gap_k=np.array(k_net, int),
        seg_net=seg_net, cell_gap=cell_gap, frac_pix_gap=frac.astype(np.float32),
        n_pix=cnt.astype(np.int32), seg_cell=seg_cell.astype(np.int16),
        pix_seg=pix_seg, n_gap_pix=n_gap.astype(np.float32), licsbas15_mask=mask)
    ng = n_gap[act]
    ncell_gaps = cell_gap[has].sum(1)
    extra_any = extra.any(0)
    per_k = pix.mean(1)
    top = np.argsort(per_k)[::-1][:8]
    return {
        "n_epochs": len(dates), "first": dates[0], "last": dates[-1], **net,
        "net_gap_interval_index": k_net,
        "active_pixels_in_mask": int(act.sum()),
        "pixel_n_gap": {"min": float(np.nanmin(ng)), "median": float(np.nanmedian(ng)),
                        "p90": float(np.nanpercentile(ng, 90)), "max": float(np.nanmax(ng))},
        "frac_pixels_with_gaps_beyond_network": float(extra_any.mean()),
        "frac_pixels_with_all_network_gaps": float(pix[k_net].all(0).mean()) if k_net else None,
        "intervals_with_most_pixel_gaps": [
            {"interval": f"{dates[k]}_{dates[k + 1]}", "frac_active_pixels": float(per_k[k]),
             "network_gap": bool(k in k_net)} for k in top],
        "cells_with_data": int(has.sum()),
        "cell_n_gaps_any_pixel": {
            "min": int(ncell_gaps.min()), "median": float(np.median(ncell_gaps)),
            "p90": float(np.percentile(ncell_gaps, 90)), "max": int(ncell_gaps.max()),
            "frac_cells_equal_to_network": float(np.mean(ncell_gaps == len(k_net)))},
        "cell_rule": "a cell has a gap at interval k when any contributing pixel (LiCSBAS15 "
                     "mask) has one; segment id = number of cell gaps before the epoch"}


def window_note(js: dict, t0: str, t1: str, half: int = C.GNSS_HALF_WINDOW_D) -> list[str]:
    lo = (pd.Timestamp(t0) - pd.Timedelta(days=half)).strftime("%Y%m%d")
    hi = (pd.Timestamp(t1) + pd.Timedelta(days=half)).strftime("%Y%m%d")
    return [f"{a}_{b}" for a, b in js["gaps"] if lo <= a and b <= hi]


def main() -> int:
    A = Lb.n_active()
    out = {r: run_gaps(r, A) for r in ("P", "F", "D")}
    out["G1_window_gaps"] = {r: window_note(out[r], C.GNSS_T0, C.GNSS_T1) for r in ("P", "F")}
    out["pilot_window_gaps_F"] = window_note(out["F"], C.PILOT_T0, C.PILOT_T1)
    Lb.jdump(out, C.EVID / "repair_gaps.json")
    for r in ("P", "F", "D"):
        v = out[r]
        print(r, v["gaps"], v["pixel_n_gap"], "beyond:", round(
            v["frac_pixels_with_gaps_beyond_network"], 3), "cells:", v["cell_n_gaps_any_pixel"])
    print("G1 window gaps", out["G1_window_gaps"], "pilot", out["pilot_window_gaps_F"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
