"""X5 synthetic recovery check (declared in X5.record_low_tf.v1 before it runs).

For every ring site-band that meets the support rules, the twin's own VEP column
(``hydrophysics.twin.compaction.vep_compaction``, read-only) is driven by the band's REAL
primary head series (2012-2021) with a known (Ske, Skv, tau), a 3 mm/yr trend is added, iid
noise is drawn, and the result is sampled on the band's REAL ring-survey months (only the
pattern of surveyed months is read from the rings, never a value). The X5 estimator is then
fitted on FIT exactly as on the real data. Pass rule (x5const): in each truth case, at
>= SYN_FRAC of the identifiable bands (heads >= SYN_D_ID_M below the burn-in minimum within
FIT), the median over seeds of Skv_hat/Skv lies in SYN_SKV_BAND and of tau_hat/tau in
SYN_TAU_BAND; fewer than SYN_MIN_BANDS identifiable bands -> not evaluable (fail).

Writes results/dataprep/x5_record_low/synthetic/ once (refuses to overwrite).

Usage: python synth_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402

OUT = K.OUT / "synthetic"


def support_ok(obs: np.ndarray, months) -> bool:
    fit = L.in_window(months, K.FIT) & np.isfinite(obs)
    test = L.in_window(months, K.TEST) & np.isfinite(obs)
    pers = L.in_window(months, K.PERSIST) & np.isfinite(obs)
    return (fit.sum() >= K.MIN_FIT_OBS and test.sum() >= K.MIN_TEST_OBS
            and pers.sum() >= K.MIN_PERSIST_OBS and L.anchor_index(obs, months) is not None)


def run(noise_m: float, recs, months) -> list[dict]:
    rows = []
    fit = L.in_window(months, K.FIT)
    for r in recs:
        if r["obs"] is None or r["h_primary"] is None or not support_ok(r["obs"], months):
            continue
        h = r["h_primary"]
        mask = np.isfinite(r["obs"])
        d_fit = float(np.max(L.precon_inputs(h, months)["x_i"][fit], initial=0.0))
        for ci, case in enumerate(K.SYN_CASES):
            for s in range(K.SYN_SEEDS):
                rng = np.random.default_rng(K.SYN_SEED0 + 1000 * ci + s)
                o = L.synth_obs(h, months, case, mask, rng, noise_m)
                f = L.fit_tf(h, o, months, fit)
                rows.append({"site": r["site"], "band": r["band"], "case": case["name"],
                             "seed": s, "noise_m": noise_m, "D_fit_m": d_fit,
                             "identifiable": d_fit >= K.SYN_D_ID_M,
                             "skv_ratio": f["B"] / case["skv"],
                             "tau_ratio": f["tau_v_days"] / case["tau_days"],
                             "ske_ratio_A": f["A"] / case["ske"],
                             "ske_ratio_A0": f["A0"] / case["ske"],
                             "trend_ratio": f["r_m_yr"] / case["trend_m_yr"]})
    return rows


def verdict(df: pd.DataFrame) -> dict:
    med = (df.groupby(["case", "site", "band"])
           .agg(identifiable=("identifiable", "first"), D_fit_m=("D_fit_m", "first"),
                skv_ratio=("skv_ratio", "median"), tau_ratio=("tau_ratio", "median"),
                ske_ratio_A=("ske_ratio_A", "median"), trend_ratio=("trend_ratio", "median"))
           .reset_index())
    med["ok"] = (med.skv_ratio.between(*K.SYN_SKV_BAND)
                 & med.tau_ratio.between(*K.SYN_TAU_BAND))
    out = {}
    for case, g in med.groupby("case"):
        gi = g[g.identifiable]
        n = int(len(gi))
        frac = float(gi.ok.mean()) if n else float("nan")
        out[case] = {"n_bands": int(len(g)), "n_identifiable": n,
                     "n_ok_identifiable": int(gi.ok.sum()), "frac_ok_identifiable": frac,
                     "n_ok_non_identifiable": int(g[~g.identifiable].ok.sum()),
                     "evaluable": n >= K.SYN_MIN_BANDS,
                     "pass": bool(n >= K.SYN_MIN_BANDS and frac >= K.SYN_FRAC)}
    return {"by_case": out, "pass": bool(out) and all(v["pass"] for v in out.values()),
            "medians": med}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "synthetic_check.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the synthetic check runs once")
    recs, months = L.build_inputs()
    res = {"declaration": K.PREREG_ID, "label": "synthetic recovery check (declared criterion)",
           "criterion": {"skv_band": K.SYN_SKV_BAND, "tau_band": K.SYN_TAU_BAND,
                         "frac": K.SYN_FRAC, "min_bands": K.SYN_MIN_BANDS,
                         "D_identifiable_m": K.SYN_D_ID_M, "noise_m": K.SYN_NOISE_M,
                         "seeds": K.SYN_SEEDS, "cases": K.SYN_CASES}}
    for tag, noise in (("gating", K.SYN_NOISE_M), ("sensitivity", K.SYN_NOISE_SENS_M)):
        df = pd.DataFrame(run(noise, recs, months))
        df.to_csv(OUT / f"synthetic_draws_{tag}.csv", index=False)
        v = verdict(df)
        v["medians"].to_csv(OUT / f"synthetic_band_medians_{tag}.csv", index=False)
        res[tag] = {k: v[k] for k in ("by_case", "pass")}
    res["pass"] = res["gating"]["pass"]
    res["sensitivity"]["note"] = "4 mm noise; reported, not gating"
    print(json.dumps(res, indent=1, default=float))
    path.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
