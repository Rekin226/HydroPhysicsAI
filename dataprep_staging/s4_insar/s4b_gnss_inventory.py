"""WP S4b: inventory of the WRA GNSS stations for the leave-one-station-out test (metadata and
data counts only; no InSAR value and no height value is compared here).

For every WRA station of the vendor cache (L0 ``chou-shui-data/data/ls_cache``): position
(metadata columns selected explicitly), active twin cell, nearest multilooked pixel of the
primary frame and its distance, the number of daily heights inside each +-30 d window of the
registered change (``GNSS_T0``, ``GNSS_T1``), and the plan's catalogued steps inside the change
span. Writes ``results/dataprep/insar_tools/s4b_run/gnss_inventory.json``.

    python -I s4b_gnss_inventory.py      (hydro env)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402


def inventory(t0: str = C.GNSS_T0, t1: str = C.GNSS_T1) -> pd.DataFrame:
    meta = Lb.wra_meta()
    files = Lb.wra_files()
    g = Lb.load_geom(C.PRIMARY)
    mask, dx, x0, y0 = Lb.twin_grid()
    lo = pd.Timestamp(t0) - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
    hi = pd.Timestamp(t1) + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
    rows = []
    for st, p in files.items():
        if st not in meta.index:
            rows.append({"station": st, "note": "no metadata row"})
            continue
        x, y = float(meta.loc[st, "X_3826"]), float(meta.loc[st, "Y_3826"])
        cell = int(Lb.cell_index(np.array([x]), np.array([y]), mask, dx, x0, y0)[0])
        i, j, dist = Lb.station_pixel(g, x, y)
        s = Lb.read_wra(p)
        n0 = int(s.loc[pd.Timestamp(t0) - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D):
                       pd.Timestamp(t0) + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)].size)
        n1 = int(s.loc[pd.Timestamp(t1) - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D):
                       pd.Timestamp(t1) + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)].size)
        steps = [d for d, _ in C.GNSS_STEP_CATALOGUE.get(st, [])
                 if lo <= pd.Timestamp(d) <= hi]
        rows.append({"station": st, "cell": cell, "on_fan": cell >= 0, "ml_i": i, "ml_j": j,
                     "pix_dist_m": round(dist, 1), "first": str(s.index.min().date()),
                     "last": str(s.index.max().date()), "n_days_w0": n0, "n_days_w1": n1,
                     "gnss_ok": n0 >= C.GNSS_MIN_DAYS_PER_WINDOW
                     and n1 >= C.GNSS_MIN_DAYS_PER_WINDOW,
                     "catalogue_steps_in_span": steps})
    return pd.DataFrame(rows).set_index("station")


def main() -> int:
    inv = inventory()
    out = C.RUN_DIR / "gnss_inventory.json"
    Lb.jdump(inv.reset_index().to_dict(orient="records"), out)
    pd.set_option("display.width", 200)
    print(inv.drop(columns=["ml_i", "ml_j"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
