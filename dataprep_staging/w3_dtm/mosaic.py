"""20 m DTM mosaic on a fixed lattice around the twin grid (WP W3).

The MOI 20 m DTM points sit on multiples of 20 m in TWD97 TM2 (EPSG:3826). The mosaic is a
dense array over ``EXTENT`` with one cell per lattice point: height (float32, NaN = no data),
acquisition epoch (decimal year of the sheet's 原始資料生產日期) and sheet index.

Rules (fixed before any comparison with leveling):

* sheets are selected by their header SW corner within ``EXTENT`` grown by 5 km, and their
  extent is taken from the points actually read (some headers store extents, not counts);
* a sheet acquired after ``max_acquisition`` (default 2022-12-31, U3) is not used; the
  builder reports whether any lattice point inside an active cell needed it;
* the same sheet id in two county archives is read once when the two members are
  byte-identical (otherwise both are read and reported);
* where two sheets give a value for the same lattice point, the later acquisition wins
  (sheets are written in ascending acquisition order) and every such overlap is counted
  with its height difference;
* heights below -100 m or above 4000 m are no-data sentinels.
"""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from w3_dtm import dtm_io

#: (E_min, E_max, N_min, N_max) of the lattice, metres EPSG:3826, multiples of 20 m. It covers
#: the twin grid (x 162191..221191, y 2600420..2677420) with about 2 km of margin.
EXTENT = (160000.0, 224000.0, 2598000.0, 2680000.0)
STEP = 20.0
NODATA_LO, NODATA_HI = -100.0, 4000.0
MAX_ACQUISITION = "20221231"


@dataclass
class Mosaic:
    extent: tuple
    step: float
    H: np.ndarray            # (rows, cols) float32, row 0 = N_min
    epoch: np.ndarray        # (rows, cols) float32 decimal year
    sheet: np.ndarray        # (rows, cols) int16 index into sheets (-1 = none)
    sheets: pd.DataFrame
    overlap: dict = field(default_factory=dict)
    excluded_late: pd.DataFrame | None = None
    duplicates: list = field(default_factory=list)

    @property
    def e(self) -> np.ndarray:
        return self.extent[0] + self.step * np.arange(self.H.shape[1])

    @property
    def n(self) -> np.ndarray:
        return self.extent[2] + self.step * np.arange(self.H.shape[0])

    def bilinear(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear height at points (NaN if any of the 4 corners is missing) and the epoch
        of the nearest lattice point."""
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        fx = (x - self.extent[0]) / self.step
        fy = (y - self.extent[2]) / self.step
        fin = np.isfinite(fx) & np.isfinite(fy)
        fx = np.where(fin, fx, -1.0)                 # missing coordinates -> outside
        fy = np.where(fin, fy, -1.0)
        j0 = np.floor(fx).astype(int)
        i0 = np.floor(fy).astype(int)
        ok = (j0 >= 0) & (i0 >= 0) & (j0 + 1 < self.H.shape[1]) & (i0 + 1 < self.H.shape[0])
        out = np.full(x.shape, np.nan)
        ep = np.full(x.shape, np.nan)
        j0c, i0c = np.clip(j0, 0, self.H.shape[1] - 2), np.clip(i0, 0, self.H.shape[0] - 2)
        tx, ty = fx - j0c, fy - i0c
        h00 = self.H[i0c, j0c]
        h01 = self.H[i0c, j0c + 1]
        h10 = self.H[i0c + 1, j0c]
        h11 = self.H[i0c + 1, j0c + 1]
        v = (h00 * (1 - tx) * (1 - ty) + h01 * tx * (1 - ty) + h10 * (1 - tx) * ty
             + h11 * tx * ty)
        out[ok] = v[ok]
        jn = np.clip(np.rint(fx).astype(int), 0, self.H.shape[1] - 1)
        inn = np.clip(np.rint(fy).astype(int), 0, self.H.shape[0] - 1)
        ep[ok] = self.epoch[inn[ok], jn[ok]]
        return out, ep


def _member_sha(archive: str, member: str) -> str:
    with zipfile.ZipFile(archive) as z:
        return hashlib.sha256(z.read(member)).hexdigest()


def build_mosaic(archives: list[str], extent: tuple = EXTENT, step: float = STEP,
                 max_acquisition: str = MAX_ACQUISITION, margin: float = 5000.0,
                 log=print) -> Mosaic:
    e0, e1, n0, n1 = extent
    ncol = int(round((e1 - e0) / step)) + 1
    nrow = int(round((n1 - n0) / step)) + 1
    H = np.full((nrow, ncol), np.nan, dtype="float32")
    EP = np.full((nrow, ncol), np.nan, dtype="float32")
    SH = np.full((nrow, ncol), -1, dtype="int16")

    cand = []
    for a in archives:
        for s in dtm_io.list_sheets(a):
            h = s.header
            if not (e0 - margin <= h["sw_e"] <= e1 + margin
                    and n0 - margin <= h["sw_n"] <= n1 + margin):
                continue
            cand.append({**h, "archive": a, "member_grd": s.member_grd,
                         "member_hdr": s.member_hdr})
    cand = pd.DataFrame(cand)
    late = cand[cand["acquisition_date"].astype(str) > max_acquisition].copy()
    use = cand[cand["acquisition_date"].astype(str) <= max_acquisition].copy()
    use = use.sort_values(["acquisition_date", "file_sheet_id", "archive"]).reset_index(drop=True)

    # same sheet id in two archives: read once if byte-identical
    keep, dups = [], []
    seen: dict[str, tuple[str, str]] = {}
    for i, r in use.iterrows():
        sid = str(r["file_sheet_id"])
        sha = _member_sha(r["archive"], r["member_grd"])
        if sid in seen:
            if seen[sid][1] == sha:
                dups.append({"sheet_id": sid, "archive": r["archive"], "identical_to":
                             seen[sid][0], "identical": True})
                continue
            dups.append({"sheet_id": sid, "archive": r["archive"], "identical_to":
                         seen[sid][0], "identical": False})
        else:
            seen[sid] = (r["archive"], sha)
        keep.append(i)
    use = use.loc[keep].reset_index(drop=True)
    use["grd_sha256"] = [_member_sha(a, m) for a, m in zip(use["archive"], use["member_grd"],
                                                              strict=True)]

    n_over, diffs = 0, []
    recs = []
    for k, r in use.iterrows():
        g = dtm_io.read_grid(r["archive"], r["member_grd"])
        E, N, Z = g[:, 0], g[:, 1], g[:, 2]
        sent = (Z < NODATA_LO) | (Z > NODATA_HI)
        on_lattice = (np.abs(E / step - np.rint(E / step)) < 1e-6) & \
            (np.abs(N / step - np.rint(N / step)) < 1e-6)
        j = np.rint((E - e0) / step).astype(int)
        i = np.rint((N - n0) / step).astype(int)
        inside = (j >= 0) & (j < ncol) & (i >= 0) & (i < nrow)
        ok = inside & on_lattice & ~sent
        recs.append({"k": k, "n_read": len(Z), "n_sentinel": int(sent.sum()),
                     "n_off_lattice": int((~on_lattice).sum()), "n_inside": int(ok.sum()),
                     "e_min": float(E.min()), "e_max": float(E.max()),
                     "n_min": float(N.min()), "n_max": float(N.max()),
                     "z_min": float(np.nanmin(np.where(sent, np.nan, Z))) if (~sent).any()
                     else float("nan"),
                     "z_max": float(np.nanmax(np.where(sent, np.nan, Z))) if (~sent).any()
                     else float("nan")})
        if not ok.any():
            continue
        ii, jj, zz = i[ok], j[ok], Z[ok].astype("float32")
        prev = H[ii, jj]
        both = np.isfinite(prev)
        if both.any():
            n_over += int(both.sum())
            diffs.append((zz[both] - prev[both]).astype("float64"))
        H[ii, jj] = zz
        EP[ii, jj] = np.float32(r["acquisition_year"])
        SH[ii, jj] = k
    st = pd.DataFrame(recs).set_index("k")
    use = use.join(st)
    d = np.concatenate(diffs) if diffs else np.array([])
    overlap = {"n_points": int(n_over)}
    if d.size:
        ad = np.abs(d)
        overlap.update({"max_abs_diff_m": float(ad.max()), "median_abs_diff_m":
                        float(np.median(ad)), "p99_abs_diff_m": float(np.percentile(ad, 99)),
                        "frac_exact": float(np.mean(ad == 0))})
    if log:
        log(f"mosaic: {len(use)} sheets used ({len(late)} late sheets excluded, "
            f"{sum(x['identical'] for x in dups)} identical duplicates skipped), "
            f"{int(np.isfinite(H).sum())} lattice points, overlaps {overlap}")
    return Mosaic(extent=extent, step=step, H=H, epoch=EP, sheet=SH, sheets=use,
                  overlap=overlap, excluded_late=late, duplicates=dups)


def cell_blocks(mos: Mosaic, x0: float, y0: float, dx: float, rows: np.ndarray,
                cols: np.ndarray) -> list[tuple[slice, slice]]:
    """Lattice slices (row slice, col slice) of the points inside each model cell
    ``[x0 + c dx, x0 + (c+1) dx) x [y0 + r dx, y0 + (r+1) dx)``."""
    e0, _, n0, _ = mos.extent
    st = mos.step
    out = []
    for r, c in zip(rows, cols, strict=True):
        xa, ya = x0 + c * dx, y0 + r * dx
        j0 = int(np.ceil((xa - e0) / st - 1e-9))
        j1 = int(np.ceil((xa + dx - e0) / st - 1e-9))
        i0 = int(np.ceil((ya - n0) / st - 1e-9))
        i1 = int(np.ceil((ya + dx - n0) / st - 1e-9))
        out.append((slice(i0, i1), slice(j0, j1)))
    return out


def aggregate_cells(mos: Mosaic, blocks: list[tuple[slice, slice]]) -> pd.DataFrame:
    """Per-cell statistics of the 20 m points: n, coverage, mean, median, sd, min, max, p05,
    p95, epoch mix (point-weighted mean epoch, min, max, list of (epoch, share))."""
    recs = []
    for k, (si, sj) in enumerate(blocks):
        h = mos.H[si, sj].ravel().astype("float64")
        ep = mos.epoch[si, sj].ravel().astype("float64")
        n_tot = h.size
        ok = np.isfinite(h)
        n = int(ok.sum())
        rec = {"cell": k, "n_lattice": n_tot, "n_points": n,
               "coverage": n / n_tot if n_tot else 0.0}
        if n:
            hv, ev = h[ok], ep[ok]
            u, cnt = np.unique(ev, return_counts=True)
            rec.update({"mean": hv.mean(), "median": float(np.median(hv)),
                        "sd": float(hv.std(ddof=1)) if n > 1 else float("nan"),
                        "min": hv.min(), "max": hv.max(),
                        "p05": float(np.percentile(hv, 5)), "p95": float(np.percentile(hv, 95)),
                        "epoch_mean": float(ev.mean()), "epoch_min": float(ev.min()),
                        "epoch_max": float(ev.max()),
                        "epochs": [float(x) for x in u], "epoch_shares":
                        [float(x) / n for x in cnt]})
        recs.append(rec)
    return pd.DataFrame(recs)
