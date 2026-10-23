"""Shared helpers of WP P6: text layer of the L0 PDFs, grid cells, census proxies.

The PDFs are untrusted data: only poppler tools (pdftotext, pdfseparate, pdftoppm, pdfinfo)
read them and nothing in them is executed. The census is read through the loader guard
(``pump_electricity/census_L1/v3``), selecting columns explicitly; the TPC parquet files and
their raw/extracted directories are never opened here.
"""
from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.paths import TWIN_GRID_NPZ  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402

NUM = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\d.,])\d+(?:\.\d+)?")


# ------------------------------------------------------------------ text layer
@cache
def doc_pages(src: str, layout: bool = True) -> tuple[str, ...]:
    """Pages of a source's text layer (pdftotext; form feeds split pages)."""
    args = ["pdftotext"] + (["-layout"] if layout else []) + [str(K.SRC[src]), "-"]
    out = subprocess.run(args, check=True, capture_output=True).stdout.decode(
        "utf-8", errors="replace")
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return tuple(pages)


def page_text(src: str, page: int, layout: bool = True) -> str:
    return doc_pages(src, layout)[page - 1]


def squash(s: str) -> str:
    """Whitespace removed, NFKC-normalised (the PDFs mix CJK compatibility ideographs)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def find_quote(src: str, page: int, quote: str) -> bool:
    """True when ``quote`` occurs on the page once all whitespace is removed from both."""
    q = squash(quote)
    return bool(q) and (q in squash(page_text(src, page)) or
                        q in squash(page_text(src, page, layout=False)))


def find_page(src: str, quote: str) -> int | None:
    """First page holding ``quote`` (whitespace-insensitive)."""
    q = squash(quote)
    for i, p in enumerate(doc_pages(src), 1):
        if q in squash(p):
            return i
    return None


def numbers(s: str) -> list[float]:
    """Numbers printed in a line, left to right (thousands separators removed)."""
    return [float(m.group(0).replace(",", "")) for m in NUM.finditer(
        unicodedata.normalize("NFKC", s))]


def write_text_layers(out_dir: Path = K.TEXT_DIR) -> dict:
    """Write each source's text layer to the evidence directory (derived, gitignored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {}
    for src, pdf in K.SRC.items():
        p = out_dir / f"{pdf.stem}.txt"
        txt = "\f".join(doc_pages(src))
        p.write_text(txt, encoding="utf-8")
        rec[src] = {"pdf": pdf.name, "pages": len(doc_pages(src)), "txt": p.name}
    return rec


# ------------------------------------------------------------------ grid
@cache
def grid() -> dict:
    z = np.load(TWIN_GRID_NPZ, allow_pickle=False)
    g = {k: z[k] for k in ("nx", "ny", "dx", "x0", "y0", "mask")}
    g["mask"] = np.asarray(g["mask"], bool)
    act = -np.ones(g["mask"].shape, dtype=np.int64)
    act[g["mask"]] = np.arange(int(g["mask"].sum()))      # row-major over masked cells
    g["active_idx"] = act
    g["n_active"] = int(g["mask"].sum())
    return g


def cell_index(x, y) -> np.ndarray:
    """Active-cell index of TWD97 points (-1 outside the active grid)."""
    g = grid()
    nx, ny, dx = int(g["nx"]), int(g["ny"]), float(g["dx"])
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fin = np.isfinite(x) & np.isfinite(y)
    j = np.floor((np.where(fin, x, float(g["x0"])) - float(g["x0"])) / dx).astype(int)
    i = np.floor((np.where(fin, y, float(g["y0"])) - float(g["y0"])) / dx).astype(int)
    ok = fin & (j >= 0) & (j < nx) & (i >= 0) & (i < ny)
    out = -np.ones(len(x), dtype=np.int64)
    out[ok] = g["active_idx"][i[ok], j[ok]]
    return out


@cache
def cell_county() -> np.ndarray:
    """County of each active cell (engine/v2 ``cell_county``, Chinese names)."""
    p = guard.open_product(K.ENGINE, "model_design")
    z = np.load(p.path("engine_arrays.npz"), allow_pickle=False)
    cc = np.asarray(z["cell_county"]).astype(str)
    if len(cc) != grid()["n_active"]:
        raise ValueError("engine cell_county does not match the grid")
    return cc


def month_axis(t0: str = K.MEMBER_T0, t1: str = K.MEMBER_T1) -> list[str]:
    return [str(p) for p in pd.period_range(t0, t1, freq="M")]


# ------------------------------------------------------------------ census (C0 L1)
CENSUS_COLS = ["county", "row_idx", "uid", "town", "x_twd97", "y_twd97", "cell", "in_grid",
               "pump_hp", "purpose_raw", "purpose_class", "well_depth_m", "kept",
               "install_month"]


def census(use: str = "prior") -> pd.DataFrame:
    """C0 census_L1/v3 pump table (explicit columns; never names or addresses)."""
    p = guard.open_product(K.C0_L1, use)
    return p.read_table("census_pumps.parquet", columns=CENSUS_COLS)


def town_key(s: str) -> str:
    """Township name without its 鄉/鎮/市/區 suffix (the census and the reports differ)."""
    s = unicodedata.normalize("NFKC", str(s or "")).strip()
    return re.sub(r"[鄉鎮市區]$", "", s)


# ------------------------------------------------------------------ distances
def nearest_distance(px, py, qx, qy, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) and index of the nearest q point for each p point (brute force, chunked)."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    qx, qy = np.asarray(qx, float), np.asarray(qy, float)
    d = np.full(len(px), np.inf)
    k = np.full(len(px), -1, dtype=np.int64)
    for s in range(0, len(px), chunk):
        dx = px[s:s + chunk, None] - qx[None, :]
        dy = py[s:s + chunk, None] - qy[None, :]
        dd = np.hypot(dx, dy)
        kk = np.argmin(dd, axis=1)
        d[s:s + chunk] = dd[np.arange(len(kk)), kk]
        k[s:s + chunk] = kk
    return d, k
