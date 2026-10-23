"""WP S4b: native-resolution coherence of selected pairs at the twin's active cells.

For each selected pair, read ``<pair>.geo.cc.tif`` from L0 (uint8, NoData 0, coherence =
value / 255; all files of a frame stack by index on the window) and compute, per active cell,
the median coherence of the valid native pixels whose centre lies in the cell, and their count.
Cells without a valid pixel get NaN (outside the frame footprint or no data).

Pair rules (``s4b_config``):
  summer_preorigin  both epochs in June-September of the same year 2020, 2021 or 2022 and
                    a temporal baseline <= 24 days (the registered coherence test)
  summer_2025       the same rule for 2025 (development on the descending frame only)
  all_preorigin     every pre-origin pair (epoch2 <= 2022-12-31), reported

Run with the licsbas env (GDAL), one CPU slot:
    dataprep_staging/s4_insar/licsbas_env.sh python -I dataprep_staging/s4_insar/s4b_coherence.py \
        --frame F --rule R --out NPZ
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import s4b_config as C  # noqa: E402
from osgeo import gdal  # noqa: E402

gdal.UseExceptions()


def select_pairs(frame: str, rule: str) -> list[dict]:
    with open(C.S4A_RUN / f"pairs_{frame}.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["fetched"] == "True" and r.get("on_reference_grid") == "True"]
    out = []
    for r in rows:
        e1, e2 = r["epoch1"], r["epoch2"]
        y1, y2, m1, m2 = int(e1[:4]), int(e2[:4]), int(e1[4:6]), int(e2[4:6])
        bt = int(r["btemp_days"])
        if rule.startswith("summer"):
            years = C.COH_SUMMER_YEARS if rule == "summer_preorigin" else (2025,)
            ok = (y1 == y2 and y1 in years and m1 in C.COH_SUMMER_MONTHS
                  and m2 in C.COH_SUMMER_MONTHS and bt <= C.COH_MAX_BTEMP_D)
        elif rule == "all_preorigin":
            ok = e2 <= "20221231"
        else:
            raise ValueError(rule)
        if ok:
            out.append({"pair": r["pair"], "epoch1": e1, "epoch2": e2, "btemp_days": bt,
                        "cc_file": r["cc_file"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=[C.PRIMARY, C.DESC])
    ap.add_argument("--rule", required=True, choices=["summer_preorigin", "summer_2025",
                                                       "all_preorigin"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    notes = json.loads(C.S4A_NOTES.read_text())["frames"][a.frame]
    l0 = C.REPO / notes["l0_dir"]
    g = np.load(C.RUN_DIR / f"geom_{a.frame}.npz", allow_pickle=False)
    cn = g["cell_native"].ravel()
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    A = int(np.asarray(z["mask"]).sum())
    inside = cn >= 0
    pix = np.flatnonzero(inside)
    order = np.argsort(cn[pix], kind="stable")
    pix, cid = pix[order], cn[pix][order]
    bounds = np.searchsorted(cid, np.arange(A + 1))
    pairs = select_pairs(a.frame, a.rule)
    med = np.full((len(pairs), A), np.nan, np.float32)
    nv = np.zeros((len(pairs), A), np.int32)
    for k, p in enumerate(pairs):
        ds = gdal.Open(str(l0 / p["cc_file"]))
        cc = ds.GetRasterBand(1).ReadAsArray().astype(np.float32).ravel()
        v = cc[pix]
        v = np.where(v > 0, v / 255.0, np.nan)
        for c in range(A):
            seg = v[bounds[c]:bounds[c + 1]]
            seg = seg[np.isfinite(seg)]
            nv[k, c] = seg.size
            if seg.size:
                med[k, c] = np.median(seg)
    np.savez_compressed(a.out, pairs=np.array([p["pair"] for p in pairs]),
                        btemp_days=np.array([p["btemp_days"] for p in pairs]),
                        cell_median_coh=med, cell_n_valid=nv, rule=np.array(a.rule),
                        frame=np.array(a.frame))
    print(json.dumps({"frame": a.frame, "rule": a.rule, "n_pairs": len(pairs), "out": a.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
