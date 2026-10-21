"""Digitise the 農田水利署 irrigation-regime maps (灌溉制度公告圖, 2024) of the Changhua and
Yunlin management offices into service-area fractions on the twin's 1 km grid.

Each map is one georeferenced JPEG embedded in a one-page PDF (1754 x 1240 px), with TWD97
TM2 (EPSG:3826) grid lines every 10 km and a legend of the irrigation regimes:
雙期作田 (two-crop paddy), 單期作田 (single-crop paddy, Yunlin only) and 輪作田 (rotation field).
The digitiser

1. extracts the embedded image with poppler ``pdfimages -j`` (no rendering; the PDF is
   never executed), into a temporary directory;
2. finds the dashed grid lines (columns and rows dominated by dark grey pixels) and fits
   pixel -> TWD97 metres by least squares to the configured line coordinates, refusing a
   residual above ``MAX_GRID_RESID_PX``;
3. classifies pixels as a legend colour blended over a near-white basemap (the polygons are
   drawn semi-transparent: observed = W - alpha (W - C)); a pixel is a class when the best
   alpha lies in ``ALPHA_RANGE`` and the blend residual is below ``BLEND_TOL``;
4. drops thin linear features (orange roads, labels) with a density filter: a classified
   pixel is kept only where at least ``DENSITY_THR`` of its ``DENSITY_WIN`` x ``DENSITY_WIN``
   window is classified (any class) and half that for its own class;
5. sums class areas per twin cell (A,).

Roads, labels and the basemap drawn over the service area hide part of it, so the area of
a class is an under-count by a roughly uniform factor; the factor is reported against the
office's published 灌溉服務面積 statistics and is used only as a diagnostic (the product
uses within-district shares, which do not depend on it).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from x4_canal_leveling.config import L0_CANAL

MAX_GRID_RESID_PX = 1.5
WHITE = (250.0, 250.0, 248.0)
BLEND_TOL = 30.0
ALPHA_RANGE = (0.5, 0.92)
DENSITY_WIN = 21
DENSITY_THR = 0.3
#: Opaque legend swatch colours, read from the legend boxes of both maps (median RGB).
LEGEND_RGB = {"double": (56, 168, 8), "single": (104, 200, 168), "rotation": (248, 168, 120)}

MAPS = {
    "chc": {
        "pdf": L0_CANAL / "chc_system_map_2024.pdf",
        "x_lines_m": [170000, 180000, 190000, 200000, 210000, 220000, 230000],
        "y_lines_m": [2670000, 2660000, 2650000, 2640000, 2630000],
        "classes": ("double", "rotation"),
    },
    "yl": {
        "pdf": L0_CANAL / "yl_system_map_2024.pdf",
        "x_lines_m": [170000, 180000, 190000, 200000, 210000],
        "y_lines_m": [2630000, 2620000, 2610000, 2600000],
        "classes": ("double", "single", "rotation"),
    },
}
#: Legend box (pixel window) excluded from classification: x0, y0, x1, y1.
LEGEND_BOX = (1430, 1000, 1660, 1205)
#: Scale-bar box excluded: x0, y0, x1, y1.
SCALE_BOX = (120, 1120, 640, 1200)


def extract_image(pdf: Path) -> np.ndarray:
    from PIL import Image

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdfimages", "-j", str(pdf), str(Path(td) / "img")], check=True,
                       capture_output=True)
        files = sorted(Path(td).glob("img-*"))
        if len(files) != 1:
            raise ValueError(f"{pdf}: expected one embedded image, got {len(files)}")
        with Image.open(files[0]) as im:
            return np.asarray(im.convert("RGB")).astype(np.float64)


def _peaks(profile: np.ndarray, frac: float, lo: int, hi: int) -> list[int]:
    """Positions of the local maxima above ``frac`` of the window maximum (one per run)."""
    idx = [i for i in range(lo, hi) if profile[i] >= frac * profile[lo:hi].max()]
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [int(g[int(np.argmax(profile[g]))]) for g in groups]


def _progression(p: np.ndarray, n: int, tol: float = 3.0) -> np.ndarray:
    """The n candidate positions forming the best evenly spaced run (all n must match)."""
    best = None
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            step = p[j] - p[i]
            if step < 50:
                continue
            want = p[i] + step * np.arange(n)
            d = np.abs(p[:, None] - want[None, :])
            hit = d.min(0) <= tol
            if not hit.all():
                continue
            got = p[d.argmin(0)]
            res = float(np.abs(got - want).max())
            if best is None or res < best[0]:
                best = (res, got)
    if best is None:
        raise ValueError(f"no evenly spaced run of {n} grid lines among {p.tolist()}")
    return best[1]


def georeference(img: np.ndarray, x_lines: list[int], y_lines: list[int]) -> dict:
    """Pixel -> metres: x = ax + bx * col, y = ay + by * row (north-up map, no rotation)."""
    dark = (img.max(2) < 110) & ((img.max(2) - img.min(2)) < 30)
    h, w = dark.shape
    col, row = dark.sum(0).astype(float), dark.sum(1).astype(float)
    # frame lines are the strongest; grid lines are the next tier inside the frame
    fx = _peaks(col, 0.9, 0, w)
    fy = _peaks(row, 0.9, 0, h)
    x0f, x1f = min(fx) + 5, max(fx) - 5
    y0f, y1f = min(fy) + 5, max(fy) - 5
    cand_x = _peaks(col, 0.15, x0f, x1f)
    cand_y = _peaks(row, 0.15, y0f, y1f)
    gx = _progression(np.array(sorted(cand_x), float), len(x_lines))
    gy = _progression(np.array(sorted(cand_y), float), len(y_lines))
    bx, ax = np.polyfit(gx, np.asarray(x_lines, float), 1)
    by, ay = np.polyfit(gy, np.asarray(y_lines, float), 1)
    rx = (np.asarray(x_lines) - (ax + bx * gx)) / abs(bx)
    ry = (np.asarray(y_lines) - (ay + by * gy)) / abs(by)
    resid = float(max(np.abs(rx).max(), np.abs(ry).max()))
    if resid > MAX_GRID_RESID_PX:
        raise ValueError(f"grid fit residual {resid:.2f} px > {MAX_GRID_RESID_PX}")
    return {"ax": float(ax), "bx": float(bx), "ay": float(ay), "by": float(by),
            "grid_cols_px": gx.tolist(), "grid_rows_px": gy.tolist(),
            "resid_px": resid, "m_per_px_x": float(abs(bx)), "m_per_px_y": float(abs(by)),
            "frame_px": [int(x0f), int(y0f), int(x1f), int(y1f)],
            "aspect_check": float(abs(bx) / abs(by))}


def classify(img: np.ndarray, classes: tuple, frame: list[int]) -> dict[str, np.ndarray]:
    from scipy.ndimage import uniform_filter

    h, w, _ = img.shape
    keep = np.zeros((h, w), bool)
    x0, y0, x1, y1 = frame
    keep[y0:y1, x0:x1] = True
    for bx0, by0, bx1, by1 in (LEGEND_BOX, SCALE_BOX):
        keep[by0:by1, bx0:bx1] = False
    white = np.array(WHITE)
    res, alp = [], []
    for name in classes:
        d = white - np.array(LEGEND_RGB[name], float)
        a = np.clip(((white - img) @ d) / (d @ d), 0.0, 1.0)
        r = np.sqrt(((img - (white - a[..., None] * d)) ** 2).sum(2))
        res.append(r)
        alp.append(a)
    res_a, alp_a = np.stack(res), np.stack(alp)
    j = res_a.argmin(0)
    rmin = res_a.min(0)
    amin = np.take_along_axis(alp_a, j[None], 0)[0]
    base = keep & (rmin < BLEND_TOL) & (amin >= ALPHA_RANGE[0]) & (amin <= ALPHA_RANGE[1])
    dens_all = uniform_filter(base.astype(float), DENSITY_WIN)
    out = {}
    for i, name in enumerate(classes):
        m = base & (j == i)
        dk = uniform_filter(m.astype(float), DENSITY_WIN)
        out[name] = m & (dens_all >= DENSITY_THR) & (dk >= 0.5 * DENSITY_THR)
    return out


def digitise(office: str) -> dict:
    cfg = MAPS[office]
    img = extract_image(cfg["pdf"])
    geo = georeference(img, cfg["x_lines_m"], cfg["y_lines_m"])
    masks = classify(img, cfg["classes"], geo["frame_px"])
    px_area = geo["m_per_px_x"] * geo["m_per_px_y"]
    rows, cols = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xs = geo["ax"] + geo["bx"] * (cols + 0.5)
    ys = geo["ay"] + geo["by"] * (rows + 0.5)
    pts = {k: (xs[m], ys[m]) for k, m in masks.items()}
    return {"office": office, "geo": geo, "px_area_m2": px_area,
            "class_pixels": {k: int(m.sum()) for k, m in masks.items()},
            "class_area_ha": {k: float(m.sum() * px_area / 1e4) for k, m in masks.items()},
            "points": pts, "image_shape": list(img.shape)}


def to_cells(points: dict, px_area: float, grid) -> dict[str, np.ndarray]:
    """Class area (m2) per active cell (A,) from classified pixel centres."""
    act = np.flatnonzero(grid.mask.ravel())
    lut = np.full(grid.mask.size, -1, np.int64)
    lut[act] = np.arange(act.size)
    out = {}
    for k, (x, y) in points.items():
        c = np.floor((x - grid.x0) / grid.dx).astype(np.int64)
        r = np.floor((y - grid.y0) / grid.dx).astype(np.int64)
        ok = (c >= 0) & (c < grid.nx) & (r >= 0) & (r < grid.ny)
        flat = np.where(ok, r * grid.nx + c, 0)
        cell = np.where(ok, lut[flat], -1)
        good = cell >= 0
        out[k] = np.bincount(cell[good], minlength=act.size).astype(float) * px_area
    return out
