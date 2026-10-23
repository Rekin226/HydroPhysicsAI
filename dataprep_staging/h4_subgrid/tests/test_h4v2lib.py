"""CPU unit tests of h4v2lib (synthetic geometry; no product is opened)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import h4v2lib as V  # noqa: E402


def _brute(T, S, s, dx=1000.0, r_min=100.0, n=401):
    o = (np.arange(n) + 0.5) / n * dx - dx / 2
    X, Y = np.meshgrid(o, o, indexing="ij")
    r = np.maximum(np.hypot(X - s[0], Y - s[1]), r_min)
    kap = math.sqrt(V.L.OMEGA * S / T)
    return V._f_seasonal(r, kap).mean(), V._f_slow(r).mean()


def test_cell_mean_table_matches_brute_force():
    tabs = V.CellMeanTables(1000.0, 2000.0, 100.0)
    for T, S in ((3000.0, 1.5e-4), (900.0, 7e-3)):
        pts = np.array([[0.0, 0.0], [130.0, -270.0], [480.0, 520.0], [1500.0, 300.0],
                        [-2400.0, 900.0]])
        ms, ml = tabs.lookup(T, S, pts)
        for k, s in enumerate(pts):
            bs, bl = _brute(T, S, s)
            assert abs(ms[k] - bs) < 2e-3 * max(1.0, abs(bs))
            assert abs(ml[k] - bl) < 2e-3 * max(1.0, abs(bl))


def test_cellmean_kernel_averages_to_zero_over_the_cell():
    """The sub-grid kernel, averaged over evaluation points in the cell, is ~0 for any pump."""
    tabs = V.CellMeanTables(1000.0, 2000.0, 100.0)
    cfg = V.with_config(V.PRIMARY, r_max_m=2000.0)
    T, S, w = 3000.0, 1.5e-4, 1.0
    pump = np.array([[350.0, -120.0]])
    o = (np.arange(60) + 0.5) / 60 * 1000.0 - 500.0
    gs_all, gl_all = [], []
    for x in o:
        for y in o:
            r = np.hypot(pump[:, 0] - x, pump[:, 1] - y)
            gs, gl = V.subgrid_values(r, pump - 0.0, pump, np.zeros(2), T, S, w, cfg, tabs,
                                      1000.0)
            gs_all.append(gs[0])
            gl_all.append(gl[0])
    c = 1.0 / (2 * math.pi * T)
    assert abs(np.mean(gs_all)) < 0.01 * c
    assert abs(np.mean(gl_all)) < 0.01 * c


def test_centroid_mode_same_cell_reference_is_peaceman():
    tabs = V.CellMeanTables(1000.0, 2000.0, 100.0)
    cfg = V.with_config(V.PRIMARY, kernel_mode="centroid")
    T, S = 3000.0, 1.5e-4
    r = np.array([198.0])
    gs, gl = V.subgrid_values(r, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(2), T, S, 1.0,
                              cfg, tabs, 1000.0)
    assert abs(gl[0]) < 1e-6 and abs(gs[0]) < 1e-3 / (2 * math.pi * T)
