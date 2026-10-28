"""Which vertical reference is more solid: WRA leveling or GNSS/InSAR? (exploratory diagnosis)

Route 6 (X5.T1) found GNSS and anchored InSAR sinking about 13-15 mm/yr more than the WRA
leveling at the same places in 2020-21. Both cannot be right in absolute terms. The magnetic
rings of the multi-level compaction wells are the frame-free arbiter: they measure shortening
of the 0-~300 m column against the well's own bottom ring, with no datum, frame or reference
benchmark. Surface subsidence cannot be smaller than that shortening unless the bottom of the
well rises (tectonic uplift), so at each ring site:

* leveling rate < ring-total rate  -> leveling under-reads sinking (it is relative to a moving
  reference, or its datum adjustment absorbs part of the motion);
* GNSS rate - ring-total rate     -> compaction below the bottom ring + tectonics + frame
  error; it must be >= 0 apart from uplift, and where a deep benchmark exists (土庫) part of
  it is measured directly.

Windows: 2015-05..2021-05 (rings and leveling, after the 2012-14 ring gap; GNSS absent) and
2020-05..2021-05 (all three). Rates are annualised: leveling = least squares over the May
surveys of benchmarks within ``R_LEV`` of the ring well; rings = change of the monthly total
between the surveys' months; GNSS = monthly medians at the same months.

Label: exploratory, design-time data (leveling, rings and GNSS 2015-21 were all used in design
and scoring); never validation; decides which reference absolute rates are quoted against.
The look is logged with harness.prereg3 before any value is read.

Run from the repository root:
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_reference/reference_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg3  # noqa: E402
from hydrophysics.twin.leveling import load_panel  # noqa: E402
from hydrophysics.twin.ring_holdout import gnss_vertical, ring_totals  # noqa: E402

DDIR = str(REPO / "chou-shui-data/data")
OUT = REPO / "results/dataprep/x5_reference/v1"
R_LEV, R_GNSS = 1000.0, 1500.0
WINDOWS = {"2015_2021": ("2015-05-01", "2021-05-31"), "2020_2021": ("2020-05-01", "2021-05-31")}


def near(x, y, xs, ys, r):
    d = np.hypot(np.asarray(xs) - x, np.asarray(ys) - y)
    return np.where(d <= r)[0], d


def lev_rate(panel, sids, t0, t1):
    rates = []
    for sid in sids:
        g = panel[(panel.sid == sid) & (panel.datetime >= t0) & (panel.datetime <= t1)]
        if len(g) < 2:
            continue
        t = (g.datetime - g.datetime.min()).dt.days.to_numpy() / 365.25
        if t.max() < 0.8 * (t1 - t0).days / 365.25:
            continue
        rates.append(-np.polyfit(t, g.elev_m.to_numpy(), 1)[0] * 1000)   # mm/yr, + = sinking
    return (float(np.median(rates)), len(rates)) if rates else (np.nan, 0)


def series_rate(s, t0, t1):
    """mm/yr from a monthly cumulative series (m, + = sinking), months of t0 and t1."""
    a = s[(s.index >= t0 - pd.Timedelta(days=45)) & (s.index <= t0 + pd.Timedelta(days=45))]
    b = s[(s.index >= t1 - pd.Timedelta(days=45)) & (s.index <= t1 + pd.Timedelta(days=45))]
    if a.empty or b.empty:
        return np.nan
    yrs = (b.index.mean() - a.index.mean()).days / 365.25
    return float((b.mean() - a.mean()) / yrs * 1000) if yrs > 0.5 else np.nan


def main() -> int:
    rings, rxy = ring_totals(DDIR)
    gn, gxy = gnss_vertical(DDIR)
    panel = load_panel(DDIR)
    first = panel.groupby("sid").first()
    lev_ids, gnss_ids = set(), set()
    for _, r in rxy.iterrows():
        i, _ = near(r.x, r.y, first.x, first.y, R_LEV)
        lev_ids |= set(first.index[i])
        j, _ = near(r.x, r.y, gxy.x, gxy.y, R_GNSS)
        gnss_ids |= set(gxy.site.iloc[j])
    look = prereg3.record_look(
        ["x5.ref.ring_site_leveling", "x5.ref.ring_site_gnss", "x5.ref.ring_sites"],
        "subsidence:2015-01..2021-06", "X5", "exploration",
        members={"x5.ref.ring_site_leveling": {"ids": sorted(lev_ids), "window": "2015-01..2021-06",
                 "description": f"leveling benchmarks within {R_LEV:.0f} m of a ring well"},
                 "x5.ref.ring_site_gnss": {"none": "GNSS stations are not registered units",
                                           "description": "WRA GNSS within 1.5 km of a ring well"},
                 "x5.ref.ring_sites": {"none": "ring wells are not registered units",
                                       "description": "multi-level compaction wells"}},
        note="reference-frame diagnosis (leveling vs GNSS vs frame-free ring compaction); "
             "exploratory, design-time")
    rows = []
    for _, r in rxy.iterrows():
        i, _ = near(r.x, r.y, first.x, first.y, R_LEV)
        j, dg = near(r.x, r.y, gxy.x, gxy.y, R_GNSS)
        for w, (a, b) in WINDOWS.items():
            t0, t1 = pd.Timestamp(a), pd.Timestamp(b)
            lv, nl = lev_rate(panel, list(first.index[i]), t0, t1)
            rg = series_rate(rings[r.site], t0, t1)
            gs = [series_rate(gn[gxy.site.iloc[k]], t0, t1) for k in j] if len(j) else []
            gs = [g for g in gs if np.isfinite(g)]
            rows.append({"site": r.site, "window": w, "ring_total_mm_yr": rg,
                         "leveling_mm_yr": lv, "n_benchmarks": nl,
                         "gnss_mm_yr": float(np.median(gs)) if gs else np.nan,
                         "n_gnss": len(gs)})
    df = pd.DataFrame(rows)
    df["lev_minus_ring"] = df.leveling_mm_yr - df.ring_total_mm_yr
    df["gnss_minus_ring"] = df.gnss_mm_yr - df.ring_total_mm_yr
    df["gnss_minus_lev"] = df.gnss_mm_yr - df.leveling_mm_yr
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sites.csv", index=False)
    summ = {}
    for w in WINDOWS:
        d = df[df.window == w]
        s = {}
        for c in ("lev_minus_ring", "gnss_minus_ring", "gnss_minus_lev"):
            v = d[c].dropna()
            s[c] = {"n": int(len(v)), "median": float(v.median()) if len(v) else None,
                    "share_below_zero": float((v < 0).mean()) if len(v) else None}
        summ[w] = s
    res = {"label": "exploratory, design-time; never validation", "look_time": look["time"],
           "radii_m": {"leveling": R_LEV, "gnss": R_GNSS}, "summary": summ}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(df.round(2).to_string())
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
