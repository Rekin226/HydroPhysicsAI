"""WP S5 look log (A1, harness.prereg3): every look at S5 data outside a registered score.

Unit ids and members (prereg3 needs members for units outside the registries):

* ``s5.lev.<set>``     leveling benchmarks: ids = benchmark sid (the registry's own id format,
                       so overlaps with dt.folds.leveling_* are visible to the harness);
* ``s5.gnss.<set>``    ids ``gnss:<family>:<station>``;
* ``s5.insar.<set>``   ids ``insar069A:block<id>`` (4 km blocks of twin cells). InSAR blocks are
                       declared by id, not by ``cells``: they hold no head data, so they must
                       not block head-unit registrations of other WPs (S4b's open issue);
* ``s5.cmp.<set>``     rings and piles: ids ``ring:<site>``, ``pile:<sensor>``.

    python -I s5looks.py data_build      (logged BEFORE the loaders first run on real data)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
from harness import prereg3  # noqa: E402


def leveling_ids() -> list[str]:
    return sorted(pd.read_parquet(C.LEVELING_PANEL, columns=["sid"])["sid"].unique().tolist())


def gnss_ids() -> list[str]:
    out = []
    for d in (C.GNSS_PRE, C.GNSS_FULL):
        st = pd.read_csv(d / "stations.csv", usecols=["family", "station"])
        out += [f"gnss:{f}:{s}" for f, s in zip(st.family, st.station, strict=True)]
    return sorted(set(out))


def insar_block_ids() -> list[str]:
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    r, c = np.nonzero(np.asarray(z["mask"]))
    b = np.unique((r // C.INSAR_BLOCK) * 1000 + (c // C.INSAR_BLOCK))
    return [f"insar069A:block{int(x)}" for x in b]


def cmp_ids() -> list[str]:
    from s1_rings import s1const as K1
    piles = ["NTUH(1)", "NTUH(2)", "SJES", "STES", "TKJS(1)", "TKJS(2)", "TKJS(3)"]
    return [f"ring:{s}" for s in K1.SITES] + [f"pile:{p}" for p in piles]


def all_members(desc: str, window: str) -> dict:
    return {"description": desc, "ids": leveling_ids() + gnss_ids() + insar_block_ids()
            + cmp_ids(), "window": window}


def log(units: dict, scope: str, purpose: str, note: str, *, prereg_id: str | None = None,
        test_id: str | None = None) -> dict:
    """Log a look; ``units`` maps unit id -> members dict."""
    return prereg3.record_look(sorted(units), scope, C.WP, purpose, members=units,
                               prereg_id=prereg_id, test_id=test_id, note=note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data_build"])
    a = ap.parse_args(argv)
    if a.what == "data_build":
        u = {"s5.dt.data_build": all_members(
            "every S5 input: leveling panel, gnss_clean pre/full v2, insar_069A v2 and run P, "
            "rings L0, printed piles", "2012-01..2026-01")}
        rec = log(u, "subsidence_all:2012-01..2026-01", "data_build", note=(
            "S5 loaders (s5data.py, s5insarP.py) run on the real inputs to build the observation "
            "tables: the leveling QC rule (>= 2 surveys, |survey-to-survey rate| > 0.5 m/yr "
            "drops a benchmark), GNSS flag/n_days selection, InSAR block/segment aggregation, "
            "ring intervals and pile segments. Only counts, date ranges, ids and table shapes "
            "are inspected; no fit, residual, displacement statistic or score is computed. "
            "Disclosed earlier look (2026-10-14, before this log): the leveling panel's "
            "survey counts per year and five 2006-2009 elevations of one benchmark (NBI033) "
            "were printed while profiling the file."))
        print(json.dumps({k: rec[k] for k in ("time", "purpose", "units")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
