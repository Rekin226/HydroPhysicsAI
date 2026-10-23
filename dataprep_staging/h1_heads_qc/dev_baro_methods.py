"""H1a design-time check (pre-origin data only): which barometric regression design is
reliable enough to use? Run with ``python -I`` in a CPU slot.

For every station with a pre-origin record and a pressure cell, the clean hourly head
(screen pass 1, no correction) of 2012-01..2022-12 is fitted with each design of
``baro.DESIGNS`` on the whole window and on the two halves 2012-01..2016-12 and
2017-01..2022-12. Split-half reliability: Pearson and Spearman correlation across wells of
A_K(first half) vs A_K(second half), and the median of |A1 - A2| / sqrt(se1^2 + se2^2)
(about 0.67 if the bootstrap se are right and the response is stable). Also the number of
wells accepted per design. Nothing dated 2023-01 or later is read.

Usage: dev_baro_methods.py WORKDIR OUT_JSON [--n N]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import baro, pipeline, rawio, screens  # noqa: E402

HALVES = [("2012-01-01", "2017-01-01"), ("2017-01-01", "2023-01-01")]


def main():
    work, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10**6
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    P = baro.load_pressure()
    sids = [s for s in rawio.all_sids() if s in meta.index and meta.loc[s, "era5_cell"]]
    rows = []
    t00 = time.time()
    for sid in sids[:n]:
        sh, _ = rawio.load_well(sid, keep_sentinels=True)
        if len(sh) < 1000:
            continue
        m = meta.loc[sid]
        t, v = screens.to_arrays(sh)
        med, mad = pipeline.robust_ref(v)
        elev = float(m.WellElevation) if np.isfinite(m.WellElevation) else None
        depth = float(m.WellDepth) if np.isfinite(m.WellDepth) and m.WellDepth > 0 else None
        r1 = screens.screen(t, v, med_ref=med, mad_ref=mad, elev=elev, depth=depth)
        hh = pipeline.hourly_clean(t, v, r1["kind"])
        p = P[m.era5_cell]
        for kind in baro.DESIGNS:
            rec = {"sid": sid, "design": kind}
            for tag, (a, b) in [("all", ("2012-01-01", "2023-01-01"))] + list(
                    zip(("h1", "h2"), HALVES, strict=True)):
                hs = hh[(hh.index >= a) & (hh.index < b)]
                f = baro.fit_be(hs, p, kind=kind)
                rec[f"AK_{tag}"] = f.get("A_K", np.nan)
                rec[f"se_{tag}"] = f.get("se_A_K", np.nan)
                rec[f"status_{tag}"] = f.get("status")
            rows.append(rec)
        print(f"{sid} {time.time() - t00:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    summ = {}
    for kind, g in D.groupby("design"):
        ok = g[np.isfinite(g.AK_h1) & np.isfinite(g.AK_h2)]
        z = (ok.AK_h1 - ok.AK_h2).abs() / np.sqrt(ok.se_h1 ** 2 + ok.se_h2 ** 2)
        acc = g.status_all == "accepted"
        both_sig = ok[(ok.status_h1 == "accepted") & (ok.status_h2 == "accepted")]
        summ[kind] = {
            "n_wells": int(len(g)), "n_both_halves": int(len(ok)),
            "pearson_halves": float(np.corrcoef(ok.AK_h1, ok.AK_h2)[0, 1]) if len(ok) > 2
            else None,
            "spearman_halves": float(stats.spearmanr(ok.AK_h1, ok.AK_h2)[0]) if len(ok) > 2
            else None,
            "median_abs_z_halves": float(np.median(z)) if len(z) else None,
            "median_se_all": float(g.se_all.median()),
            "n_accepted_all": int(acc.sum()),
            "n_accepted_both_halves": int(len(both_sig)),
            "sign_agree_both_accepted": float(np.mean(np.sign(both_sig.AK_h1)
                                                      == np.sign(both_sig.AK_h2)))
            if len(both_sig) else None,
            "median_AK_accepted": float(g[acc].AK_all.median()) if acc.any() else None,
        }
    out = {"note": "design-time, pre-origin only (2012-01..2022-12); see module docstring",
           "summary": summ, "rows": D.to_dict("records")}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
