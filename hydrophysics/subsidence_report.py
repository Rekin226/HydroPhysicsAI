"""Print the head->subsidence coupling verdict: the R² numbers + regression beta.

This is the honest gate for whether a distance-to-coast Sk regression generalizes.
Run on the real data:

    python -m hydrophysics.subsidence_report

The pass criterion is the leave-one-site-out **Sk-space** R² > 0 (does distance-to-coast
predict a held-out site's Sk better than the mean?). The compaction-space R² is reported
alongside for comparability with the single-Sk (-0.28) and spatial-IDW (-2.40) baselines,
but it is drawdown-inflated and is NOT the gate.
"""

from __future__ import annotations


def main(argv=None) -> None:
    import argparse

    import numpy as np

    from .config import Config, default_config
    from .data import load_dataset
    from .subsidence import (
        calibrate_sk,
        fit_sk_regression,
        load_mlcw_stations,
        loso_sk_regression,
        mlcw_compaction,
        site_distance_to_coast,
    )

    ap = argparse.ArgumentParser(description="Head->subsidence coupling verdict")
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)

    cfg = Config(data_dir=args.data) if args.data else default_config()
    data = load_dataset(cfg)
    stations = load_mlcw_stations(cfg.data_dir / "mlcw_stations.csv")
    compaction = mlcw_compaction(str(cfg.data_dir))
    dist = site_distance_to_coast(stations, cfg.data_dir / "water" / "sea_TWD97.shp")
    cal = calibrate_sk(data, stations, compaction)

    # per-site own-Sk in-sample compaction R² (an optimistic upper bound, not a gate)
    obs = np.concatenate([np.asarray(C) for _, C in cal["per_site"].values()])
    res_ins = 0.0
    for _, (D, C) in cal["per_site"].items():
        D = np.asarray(D, float)
        C = np.asarray(C, float)
        sk_i = (D @ C) / (D @ D + 1e-12)
        res_ins += float(((C - sk_i * D) ** 2).sum())
    r2_insample = 1.0 - res_ins / max(((obs - obs.mean()) ** 2).sum(), 1e-12)

    fit = fit_sk_regression(cal["per_site"], dist)
    loso = loso_sk_regression(cal["per_site"], dist)

    print("=== Head -> subsidence coupling verdict (MLCW month-pairs) ===")
    print(f"  sites paired: {loso['n_sites']}")
    print("  -- compaction-space R² (comparable to baselines) --")
    print(f"    single basin-wide Sk          = {cal['r2']:+.3f}")
    print("    spatial-IDW per-site (ref)     = -2.400")
    print(f"    per-site own Sk (in-sample)    = {r2_insample:+.3f}   [upper bound]")
    print(f"    coast regression, LOSO         = {loso['r2']:+.3f}")
    print("  -- Sk-space R² (the honest generalization gate) --")
    print(f"    coast regression, in-sample    = {fit['r2_insample']:+.3f}")
    print(f"    coast regression, LOSO         = {loso['r2_sk']:+.3f}   [GATE; pass if > 0]")
    print(f"  beta: log Sk = {fit['b0']:.3f} + ({fit['b1']:.3g}) * distance_to_coast_m")
    verdict = ("PASS -> a coast-calibrated subsidence surface generalizes"
               if loso["r2_sk"] > 0 else
               "FAIL -> distance-to-coast does not generalize across the 14 sites; "
               "no validated surface")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":  # pragma: no cover
    main()
