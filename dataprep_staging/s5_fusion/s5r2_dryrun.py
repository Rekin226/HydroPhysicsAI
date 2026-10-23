"""WP S5 repair round 2: dry runs on SYNTHETIC values (real data structure, every value
replaced before any fit). Used before the registration S5.repair_r2.v1 to check the code
paths, run times and the VR geometry of the fresh test; no real displacement is used.

``install()`` replaces s5r2_data.load_all2 by a wrapper that overwrites every y (leveling,
GNSS, InSAR, rings, piles) with s5dryrun.truth plus instrument offsets, a reverting
displacement anomaly, AR(1) sensor noise and white noise. Prediction targets are synthetic
too, so the dry-run scores say nothing about the real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import zlib  # noqa: E402

import numpy as np  # noqa: E402
import s5dryrun as D1  # noqa: E402
import s5r2_data as RD  # noqa: E402

RNG = np.random.default_rng(20261018)
_LOAD = RD.load_all2


def _ar1(t, sd, tau, rng):
    t = np.asarray(t, float)
    e = np.empty(t.size)
    e[0] = rng.normal(0, sd)
    for i in range(1, t.size):
        r = np.exp(-max(t[i] - t[i - 1], 0.0) / tau)
        e[i] = r * e[i - 1] + rng.normal(0, sd * np.sqrt(1 - r * r))
    return e


def _det(key: str, sd: float) -> float:
    """Deterministic normal draw per key (same synthetic offsets in every load)."""
    return float(np.random.default_rng(zlib.crc32(key.encode())).normal(0, sd))


def _anom(x, y, t):
    """Reverting displacement anomaly (year-to-year, spatially smooth)."""
    yr = np.floor(np.asarray(t, float)).astype(int)
    ph = (yr * 2654435761 % 1000) / 1000.0
    return 6.0 * np.sin(2 * np.pi * ph + x / 20.0 + y / 30.0)


def synth(data: dict) -> dict:
    T = D1.truth
    lev = data["lev"].copy()
    off = {s: _det(s, 1000.0) for s in lev.sid.unique()}
    rb = {s: _det(s + "rb", 2.0) for s in lev.sid.unique()}
    camp = {c: _det(c, 2.0) for c in lev.camp.unique()}
    lev["y"] = (T(lev.x_km, lev.y_km, lev.t) + _anom(lev.x_km, lev.y_km, lev.t)
                + lev.sid.map(off) + lev.sid.map(rb) * (lev.t - 2016) + lev.camp.map(camp)
                + RNG.normal(0, 3, len(lev)))
    data["lev"] = lev
    g = data["gnss"].copy()
    yv = np.empty(len(g))
    for _k, idx in g.groupby(["family", "station", "seg"]).groups.items():
        r = g.loc[idx]
        yv[np.asarray([g.index.get_loc(i) for i in idx])] = (
            T(r.x_km, r.y_km, r.t) + _anom(r.x_km, r.y_km, r.t) + RNG.normal(0, 1000)
            + _ar1(r.t.to_numpy(), 4.0, 0.3, RNG) + RNG.normal(0, 2, len(r)))
    g["y"] = yv
    data["gnss"] = g
    units = data["insar"]["units"]
    uoff = RNG.normal(0, 50, len(units))
    urate = RNG.normal(0, 2, len(units))
    for gr in data["insar"]["groups"]:
        xy = units.loc[gr["unit"], ["x_km", "y_km"]].to_numpy()
        pl = (RNG.normal(0, 3) + RNG.normal(0, 0.2) * (xy[:, 0] - 185)
              + RNG.normal(0, 0.2) * (xy[:, 1] - 2620))
        gr["y"] = (T(xy[:, 0], xy[:, 1], gr["t"]) + _anom(xy[:, 0], xy[:, 1], gr["t"])
                   + uoff[gr["unit"]] + urate[gr["unit"]] * (gr["t"] - 2021) + pl
                   + RNG.normal(0, 3, len(xy)))
    c = data["cmp"].copy()
    yc = np.empty(len(c))
    for _k, idx in c.groupby(["sensor", "seg"]).groups.items():
        r = c.loc[idx]
        pos = np.asarray([c.index.get_loc(i) for i in idx])
        yc[pos] = (0.7 * (T(r.x_km, r.y_km, r.t) + _anom(r.x_km, r.y_km, r.t))
                   + RNG.normal(0, 100) + _ar1(r.t.to_numpy(), 3.0, 0.3, RNG)
                   + RNG.normal(0, 1, len(r)))
    c["y"] = yc
    data["cmp"] = c
    return data


def load_all2(variant, **kw):
    data, qc = _LOAD(variant, **kw)
    qc["SYNTHETIC"] = True
    return synth(data), qc


def install() -> None:
    RD.load_all2 = load_all2
