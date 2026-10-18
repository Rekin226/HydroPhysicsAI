"""WP S3 repair round 3 (S3r): trace of the ERLN design-unit trials of design run 2 (design-time
evidence on seed 515 only; no test unit).

    cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/design_erln_trace.py OUT_TXT [--code DIR]

``--code DIR`` runs the trace with another copy of the v3 model and pipeline, for example the
design-run-2 snapshot ``results/dataprep/reports/S3_evidence/repair_2026-10-18/tune_v3b/code``
(station-wide empirical null) instead of the current code (time-local null, R2b). Prints:

1. ERLN's base-run steps, the station-wide robust spread of the scan statistic u = D / sqrt(v),
   its robust spread by calendar year, and the same spread at four other NGL stations;
2. three seed-515 ERLN injections (+3 cm 2007-12-20, +3 cm 2008-11-28, -2 cm 2010-10-09)
   processed with the design-run-2 detector (scan +-30 d, z_min 4.0, delta_min 10 mm,
   level-difference size on +-10 d) with the 12-step cap and with the cap lifted.

Look: ``looks_r5.py erln_trace``. Its first run (with the run-2 code, from /tmp) happened at
about 02:00-02:07 CST, before that look was recorded at 02:07 CST (a disclosed slip; design
units only).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE.parents[2] / "results" / "dataprep" / "reports" / "S3_evidence" / "repair_2026-10-18"
CASES = (("2007-12-20", 0.03), ("2008-11-28", 0.03), ("2010-10-09", -0.02))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--code", default=str(HERE))
    a = ap.parse_args(argv)
    # io_gnss (unchanged since v2) always comes from this directory: a snapshot copy would
    # resolve the repository root from its own location
    sys.path.insert(0, str(HERE))
    import io_gnss  # noqa: F401
    sys.path.insert(0, str(Path(a.code).resolve()))
    import model as M
    import numpy as np
    import pandas as pd
    import pipeline as P

    lines = [f"code: {Path(a.code).resolve().relative_to(HERE.parents[2]).as_posix()}"]
    p = replace(M.Params(), win=30, z_min=4.0, delta_min=0.01, local_half=10, trend_order=2,
                size_order=0)
    ref = json.loads((EV / "tune_v3b" / "reference.json").read_text())["reference"]
    nser, _ = P.ngl_series(P.ORIGIN_END)
    base = P.run_ngl(p, ref=ref, window_end=P.ORIGIN_END, series=nser)
    r = base.results["ERLN"]
    rc = np.where(r.ok, r.fit.resid, np.nan) - (r.cmv if r.cmv is not None else 0)
    c, v = M.scan(rc, p.win, p.min_obs, p.trend_order)
    u = c / np.sqrt(v)
    _, su = M.null_scale(u)
    lines.append("ERLN base steps: " + str([(str(s.date.date()), round(1000 * s.size, 1),
                                             s.evidence) for s in r.steps]))
    lines.append(f"ERLN station-wide robust sd of u: {1000 * su:.2f} mm")
    yrs = pd.Series(u, index=r.dates).groupby(r.dates.year).apply(
        lambda x: 1000 * M.mad_sd(x.to_numpy()))
    lines.append("ERLN robust sd of u by year (mm): " + json.dumps(
        {int(k): round(float(x), 1) for k, x in yrs.items() if np.isfinite(x)}))
    for k in ("PINT", "S103", "HUWE", "DNAN"):
        rr = base.results[k]
        rcc = np.where(rr.ok, rr.fit.resid, np.nan) - (rr.cmv if rr.cmv is not None else 0)
        cc, vv = M.scan(rcc, p.win, p.min_obs, p.trend_order)
        lines.append(f"{k} station-wide robust sd of u: "
                     f"{1000 * M.null_scale(cc / np.sqrt(vv))[1]:.2f} mm")
    cm = base.detect_cm["ERLN"]
    equip, quakes = P.ngl_metadata(["ERLN"], P.ORIGIN_END)
    for d, c0 in CASES:
        s2 = P._inject(nser["ERLN"], [(pd.Timestamp(d), c0)])
        for ms in (12, 200):
            rr = M.process_station(s2, replace(p, max_steps=ms), cm=cm,
                                   equipment=equip.get("ERLN"), earthquakes=quakes.get("ERLN"))
            hit = [x for x in rr.steps if abs((x.date - pd.Timestamp(d)).days) <= 3]
            ramps = [n for n in rr.notes if "ramps" in n]
            lines.append(f"injection {d} {1000 * c0:+.0f} mm, step cap {ms}: found "
                         f"{[(str(x.date.date()), round(1000 * x.size, 1)) for x in hit]}; "
                         f"steps kept {len(rr.steps)}; {ramps}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
