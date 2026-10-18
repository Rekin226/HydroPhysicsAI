"""After-the-fact checks of the full variant (2012-2026 matrix; reported, never a gate: U7).

* Family D on 2013-2025: the 2026 outage window removed in every year, bridged with the
  deviation curves of all other years 2012-2026 (the full variant's bridge).
* Family T on the test period: the test-period replica (harness ``gaps_guarded.
  replica_test_period`` with ``allow_post_origin=True``) inside 2023-01..2026-09, avoiding
  Feb-Mar 2026, realizations r = 10..19 (the pilot used r = 0..2 there).

Both are scored with the given frozen design parameters (default v1), on clean and all
truth. The look is logged (purpose 'after_the_fact') before scoring.

Usage: python -I s7_after_fact.py [PARAMS_JSON [OUT_NAME]]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import gaps_guarded, prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import io_h2, runner, scoring  # noqa: E402

PARAMS = C.EVIDENCE_DIR / "design" / "design_params.json"
T_R = range(10, 20)


def t_family_full(h, Yb, clean, r):
    obs = np.isfinite(Yb)
    T = obs.shape[1]
    lo = io_h2.month_index(h.dates, "2023-01")
    kf = io_h2.month_index(h.dates, "2026-02")
    targets = [i for i in range(obs.shape[0]) if obs[i, lo:].mean() >= 0.7 and obs[i].sum() >= 60]
    mask, cells = gaps_guarded.replica_test_period(obs, targets, r, (lo, T), h.months,
                                                   allow_post_origin=True, avoid=(kf, kf + 1))
    C20 = np.isfinite(h.Y)
    rows = [{"stratum": "T_H06" if c[0].startswith("H") else "T_S", "gap": c[1],
             "L": int(c[0][1:]), "w": c[2], "t": c[3]} for c in cells if C20[c[2], c[3]]]
    df = pd.DataFrame(rows)
    Ym = np.where(mask, np.nan, Yb)
    tg = np.zeros_like(obs)
    tg[df.w, df.t] = True
    xy, lay = h.xy(), h.meta.layer.to_numpy()
    out, info = runner.fill_all(Ym, xy, lay, tg, gp_rows=sorted(set(df.w)))
    from h2_heads_fill import methods
    cls, runlen, edge = methods.classify_cells(np.isfinite(Ym))
    idd = runner.idw_defined(np.isfinite(Ym), xy, lay)
    wi, ti = df.w.to_numpy(), df.t.to_numpy()
    df["family"], df["r"], df["variant"] = "T", r, "full"
    df["sid"] = [h.sids[i] for i in wi]
    df["cls"], df["truth"] = cls[wi, ti], h.Y[wi, ti]
    df["clean"], df["in_model"] = clean[wi, ti], h.meta.in_model.to_numpy()[wi]
    df["idw_defined"] = idd[wi, ti]
    for k in runner.METHODS:
        if k in out:
            mm, ss = out[k]
            df[f"pred_{k}"] = mm[wi, ti]
            df[f"sd_{k}"] = ss[wi, ti] if ss is not None else np.nan
        else:
            df[f"pred_{k}"] = df[f"sd_{k}"] = np.nan
    df["cluster"] = [s if st == "T_S" else f"T{r}" for s, st in zip(df.sid, df.stratum,
                                                                     strict=True)]
    return df


def main(argv=()):
    ppath = Path(argv[0]) if argv else PARAMS
    out_name = argv[1] if len(argv) > 1 else "full_variant_checks.json"
    params = json.loads(ppath.read_text())
    h, Yb, _ = runner.load_base("full")
    clean = io_h2.clean_truth(h)
    units = {"H2:after_fact_T_2023-2026": {"description": "test-period replica r=10..19 on "
                                           "the full matrix (2023-01..2026-09 window)",
                                           "ids": sorted(h.sids), "window": "2023-01..2026-09"},
             "H2:after_fact_D_2013-2025": {"description": "family D in 2013-2025 on the full "
                                           "matrix", "ids": sorted(h.sids),
                                           "window": "2013-01..2025-12"}}
    prereg3.record_look(sorted(units), "heads:2013-01..2026-09", C.WP, "after_the_fact",
                        members=units, note="after-the-fact checks of the full variant; "
                        "reported only (U7: never a gate)")
    tic = time.time()
    tparts = [t_family_full(h, Yb, clean, r) for r in T_R]
    tdf = pd.concat(tparts, ignore_index=True)
    ddf, dinfo = runner.run_daily("full", "D", 0, h, Yb, clean)
    pt = scoring.product_monthly(tdf, params)
    pdd = scoring.product_daily(ddf, params)
    pdd["stratum"] = "D_2013_2025"
    rep = {"seconds": time.time() - tic, "params_file": str(ppath),
           "T_test_period": {
               "clean": {"coverage": scoring.coverage_rows(pt[pt.clean.astype(bool)]),
                         "rmse": scoring.rmse_table(pt[pt.clean.astype(bool)],
                                                    ("product",) + scoring.BASELINES
                                                    + scoring.REPORTED).to_dict("records")},
               "all": {"coverage": scoring.coverage_rows(pt),
                       "rmse": scoring.rmse_table(pt, ("product",) + scoring.BASELINES
                                                  ).to_dict("records")}},
           "D_2013_2025": {
               "bridgeable_share": float(ddf.ok.mean()),
               "clean": {"coverage": scoring.coverage_rows(pdd[pdd.clean.astype(bool)]),
                         "rmse": scoring.rmse_table(pdd[pdd.clean.astype(bool)],
                                                    ("product", "backfill", "clim_ar", "stk")
                                                    ).to_dict("records")},
               "all": {"coverage": scoring.coverage_rows(pdd),
                       "rmse": scoring.rmse_table(pdd, ("product", "backfill", "clim_ar",
                                                        "stk")).to_dict("records")},
               "by_year_rmse_product_clean": pdd[pdd.clean.astype(bool)].groupby(
                   pdd.month.str[:4]).apply(lambda g: scoring.rmse(g.pred_product - g.truth),
                                            include_groups=False).to_dict()}}
    scoring.jdump(rep, C.EVIDENCE_DIR / "after_the_fact" / out_name)
    tdf.to_parquet(C.WORKDIR / "after_fact_T.parquet")
    ddf.to_parquet(C.WORKDIR / "after_fact_D.parquet")
    print(json.dumps({k: rep[k]["clean"]["coverage"] if "clean" in rep[k] else None
                      for k in ("T_test_period", "D_2013_2025")}, default=str)[:3000])


if __name__ == "__main__":
    main(sys.argv[1:])
