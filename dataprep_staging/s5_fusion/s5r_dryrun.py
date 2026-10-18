"""WP S5 repair round: dry run of the monthly-field procedure and test on SYNTHETIC values.

The real data structure is loaded (as ``s5dryrun``), then every value is replaced: the
synthetic truth and instrument offsets/noise of ``s5dryrun`` plus, in variant ``anomaly``, a
synthetic monthly anomaly (8 Gaussian bumps of 6 km, AR(1) amplitudes rho 0.8, sd 8 mm) added
to the InSAR month groups and the GNSS months of the same month; variant ``null`` has none.
Run before the registration ``S5.monthly_field.v1`` to test the code paths and the test's
operating characteristics. No real displacement value is used.

    tools/cpu_slot.sh python -I s5r_dryrun.py <anomaly|null> [maxfev cap] [--score-only]
        -> S5_evidence/repair/dryrun_monthly/<variant>/
"""
from __future__ import annotations

import shutil
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5dryrun as DR  # noqa: E402  (keeps a reference to the real loader)
import s5r_const as R  # noqa: E402
import s5r_mtest as MT  # noqa: E402

warnings.simplefilter("ignore")


def anomaly_fn(seed: int = 7):
    g = D.grid()
    rng = np.random.default_rng(seed)
    centres = g["cell_xy_km"][rng.choice(g["A"], size=8, replace=False)]
    months = [str(p) for p in pd.period_range("2019-01", "2027-12", freq="M")]
    amp = np.zeros((len(months), 8))
    amp[0] = rng.normal(0, 8, 8)
    for k in range(1, len(months)):
        amp[k] = 0.8 * amp[k - 1] + np.sqrt(1 - 0.64) * rng.normal(0, 8, 8)
    mi = {m: i for i, m in enumerate(months)}

    def a(x, y, month):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        d2 = (x[:, None] - centres[None, :, 0]) ** 2 + (y[:, None] - centres[None, :, 1]) ** 2
        return np.exp(-d2 / (2 * 6.0 ** 2)) @ amp[mi[month]]
    return a


def make_loader(variant: str):
    afn = anomaly_fn() if variant == "anomaly" else None

    def load_all(v):
        data, qc = DR.load_all(v)
        if afn is None:
            return data, qc
        gn = data["gnss"]
        add = np.zeros(len(gn))
        for mo, idx in gn.groupby("month").groups.items():
            if str(mo) >= "2019-01":
                add[np.asarray(gn.index.get_indexer(idx))] = afn(
                    gn.loc[idx, "x_km"].to_numpy(), gn.loc[idx, "y_km"].to_numpy(), str(mo))
        gn["y"] = gn["y"].to_numpy() + add
        units = data["insar"]["units"]
        for gr in data["insar"]["groups"]:
            xy = units.loc[gr["unit"], ["x_km", "y_km"]].to_numpy()
            gr["y"] = gr["y"] + afn(xy[:, 0], xy[:, 1], gr["month"])
        return data, qc
    return load_all


def main() -> int:
    variant = sys.argv[1]
    num = [a for a in sys.argv[2:] if a.isdigit()]
    cap = int(num[0]) if num else None
    out = R.EVIDENCE_R / "dryrun_monthly" / variant
    (out / "repair").mkdir(parents=True, exist_ok=True)
    shutil.copy(C.EVIDENCE / "hyper_pre.json", out / "hyper_pre.json")
    D.load_all = make_loader(variant)
    if cap is not None:                       # a quicker, capped run
        R.S2_MAXFEV_NULL, R.S2_MAXFEV, R.S2_MAXFEV_RESTART = cap, cap, cap // 2
    t0 = time.time()
    if "--score-only" not in sys.argv:
        MT.hyper(out / "repair" / "hyper_stage2.json", evidence=out)
    MT.score(out / "repair" / "m1.json", evidence=out, gate=lambda: {"dry_run": True})
    print(variant, "done", round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
