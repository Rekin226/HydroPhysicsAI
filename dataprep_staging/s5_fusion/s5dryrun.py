"""WP S5 dry run: s5hyper + s5tests on SYNTHETIC values (real data structure, synthetic y).

Run before the registration (2026-10-14) to test the code paths and the calibration of the
predictive intervals; no real displacement value is used (every y is replaced by a synthetic
truth plus instrument offsets and noise). Evidence: S5_evidence/dryrun_synthetic/.

    python -I s5dryrun.py <hyper|t1|t2|t3|r2> <maxfev cap> <out dir>
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5fit as F  # noqa: E402
import s5hyper as Hy  # noqa: E402
import s5tests as Tt  # noqa: E402

warnings.simplefilter("ignore")
RNG = np.random.default_rng(1)


def truth(x, y, t):
    d = np.sqrt((x - 185) ** 2 + (y - 2620) ** 2)
    rate = 40 * np.exp(-d / 15) + 5
    acc = 6 * np.exp(-d / 10) * np.clip(t - 2019, 0, None) ** 1.5
    return rate * (t - 2012) + acc + 8 * np.exp(-d / 20) * np.sin(2 * np.pi * t)


_LOAD = D.load_all


def load_all(variant):
    data, qc = _LOAD(variant)
    lev = data["lev"].copy()
    off = {s: RNG.normal(0, 1000) for s in lev.sid.unique()}
    rb = {s: RNG.normal(0, 2) for s in lev.sid.unique()}
    lev["y"] = (truth(lev.x_km, lev.y_km, lev.t) + lev.sid.map(off)
                + lev.sid.map(rb) * (lev.t - 2016) + RNG.normal(0, 5, len(lev)))
    data["lev"] = lev
    g = data["gnss"].copy()
    off = {s: RNG.normal(0, 1000) for s in g.station.unique()}
    g["y"] = (truth(g.x_km, g.y_km, g.t) + g.station.map(off)
              + 3 * (g.t - 2012) * (g.family == "ngl") + RNG.normal(0, 4, len(g)))
    data["gnss"] = g
    units = data["insar"]["units"]
    uoff = RNG.normal(0, 50, len(units))
    for gr in data["insar"]["groups"]:
        xy = units.loc[gr["unit"], ["x_km", "y_km"]].to_numpy()
        pl = (RNG.normal(0, 3) + RNG.normal(0, 0.2) * (xy[:, 0] - 185)
              + RNG.normal(0, 0.2) * (xy[:, 1] - 2620))
        gr["y"] = (truth(xy[:, 0], xy[:, 1], gr["t"]) + uoff[gr["unit"]] + pl
                   + RNG.normal(0, 3, len(xy)))
    c = data["cmp"].copy()
    so = {(s, k): RNG.normal(0, 100)
          for s, k in c[["sensor", "seg"]].drop_duplicates().itertuples(index=False)}
    c["y"] = (0.7 * truth(c.x_km, c.y_km, c.t)
              + [so[(a, b)] for a, b in zip(c.sensor, c.seg, strict=True)]
              + RNG.normal(0, 2, len(c)))
    data["cmp"] = c
    return data, qc


def main() -> int:
    what = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("dryrun_out")
    out.mkdir(parents=True, exist_ok=True)
    C.EVIDENCE = out
    opt = F.optimise

    def capped(pr, h0, names, fams, maxfev, log=None):
        return opt(pr, h0, names, fams, min(maxfev, cap), log=log)

    F.optimise = capped
    Tt.score_gate = lambda key: {"dry": True}
    D.load_all = load_all
    t0 = time.time()
    if what == "hyper":
        Hy.main_run(out / "hyper_pre.json")
        Hy.t1_run(out / "hyper_t1.json")
    else:
        getattr(Tt, what)()
    print(what, "done", round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
