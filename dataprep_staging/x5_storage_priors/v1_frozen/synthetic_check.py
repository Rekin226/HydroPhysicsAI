"""X5 synthetic recovery check (declared in X5.storage_priors.v1 before it runs; runs once).

Synthetic hourly heads 2017-18 with a known Ss (and the BE it implies at n = 0.3), the full
eight-constituent earth tide, barometric loading by the REAL ERA5 2017-18 pressure of one
cell, irrigation pumping noise, red noise, an annual cycle, spikes, sentinels, gaps and 1 cm
quantisation; coastal wells add an ocean M2 term. The same estimators as the real-data run
(x5sp_lib.clean_hourly, fit_tides, fit_be, ss_tide, ss_be) recover Ss; criteria S1-S6 of
x5sp_const.SYN_CRITERIA decide. Writes results/dataprep/x5_storage_priors/synthetic/ once.

Usage: python -I synthetic_check.py [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5sp_const as K  # noqa: E402
import x5sp_lib as L  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def draws() -> list[dict]:
    rng = np.random.default_rng(K.SYN_SEED)
    out = []
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    for i in range(K.SYN_N_INLAND + K.SYN_N_COASTAL):
        ss = lu(*K.SYN_SS_RANGE)
        coastal = i >= K.SYN_N_INLAND
        d = {"i": i, "coastal": coastal, "ss_true": ss, "be_true": K.WATER_TERM / ss,
             "pump_amp": lu(*K.SYN_PUMP_AMP), "pump_tau_h": rng.uniform(*K.SYN_PUMP_TAU_H),
             "red_sd": lu(*K.SYN_RED_SD), "seas_amp": rng.uniform(*K.SYN_SEAS_AMP),
             "baro_tau_h": rng.uniform(*K.SYN_BARO_TAU_H), "seed": int(rng.integers(1 << 31))}
        m2_amp = L.c_nu() * L.strain_amp("M2", K.SYN_LAT) / ss
        d["m2_true_amp_m"] = m2_amp
        if coastal:
            d["ocean_amp_m"] = rng.uniform(*K.SYN_OCEAN_X) * m2_amp
            d["ocean_phase_deg"] = 180.0 + rng.uniform(-K.SYN_OCEAN_PHASE_SPREAD,
                                                       K.SYN_OCEAN_PHASE_SPREAD)
        else:
            d["ocean_amp_m"], d["ocean_phase_deg"] = 0.0, 0.0
        out.append(d)
    return out


def one(args) -> dict:
    d, p = args
    rng = np.random.default_rng(d["seed"])
    hr = L.synth_well(d["ss_true"], d["be_true"], K.SYN_LAT, K.SYN_LON, p, rng,
                      pump_amp=d["pump_amp"], pump_tau_h=d["pump_tau_h"], red_sd=d["red_sd"],
                      seas_amp=d["seas_amp"], baro_tau_h=d["baro_tau_h"],
                      ocean_amp_m=d["ocean_amp_m"], ocean_phase_deg=d["ocean_phase_deg"])
    h = L.clean_hourly(hr, None)
    t = L.fit_tides(h, K.SYN_LAT, K.SYN_LON)
    b = L.fit_be(h, p)
    r = dict(d)
    r["tide_status"] = t.get("status")
    for n in ("M2", "O1"):
        c = t.get(n, {})
        for k in ("gain", "snr", "eta_deg", "amp_m", "se_phase_deg"):
            r[f"{n}_{k}"] = c.get(k, np.nan)
    r.update({f"be_{k}": v for k, v in b.items()})
    r["ss_m2"] = L.ss_tide(r["M2_gain"]) if np.isfinite(r["M2_gain"]) else np.nan
    r["ss_o1"] = L.ss_tide(r["O1_gain"]) if np.isfinite(r["O1_gain"]) else np.nan
    be = r.get("be_BE_long")
    r["ss_be"] = L.ss_be(be) if be is not None and be > 0 else np.nan
    return r


def evaluate(df: pd.DataFrame) -> dict:
    C = K.SYN_CRITERIA
    inl = df[~df["coastal"]]
    co = df[df["coastal"]]
    m2_det = inl["M2_snr"] >= K.SNR_MIN
    e_t = np.abs(np.log10(inl.loc[m2_det, "ss_m2"] / inl.loc[m2_det, "ss_true"]))
    big = inl["m2_true_amp_m"] >= C["S2_m2_true_amp_min_m"]
    det_rate = float(m2_det[big].mean()) if big.any() else np.nan
    be_ok = (inl["be_accepted"].astype(bool) & (inl["be_A_K"] < 0)
             & inl["be_BE_long"].between(K.BE_MIN, K.BE_MAX))
    e_b = np.abs(np.log10(inl.loc[be_ok, "ss_be"] / inl.loc[be_ok, "ss_true"]))
    bebig = inl["be_true"] >= C["S3_be_true_min"]
    acc_rate = float(be_ok[bebig].mean()) if bebig.any() else np.nan
    both = m2_det & be_ok
    rho = (float(inl.loc[both, "ss_m2"].rank().corr(inl.loc[both, "ss_be"].rank()))
           if both.sum() >= 3 else np.nan)

    def flagged(x):
        o1 = x["O1_snr"] >= K.SNR_MIN
        return (x.loc[o1, "M2_gain"] / x.loc[o1, "O1_gain"] > K.OCEAN_RATIO), int(o1.sum())
    fc, nco = flagged(co)
    fi, nin = flagged(inl)
    eta = np.abs(inl.loc[m2_det, "M2_eta_deg"])
    res = {
        "S1": {"n": int(m2_det.sum()), "median_abs_log10": float(np.median(e_t)),
               "p90_abs_log10": float(np.quantile(e_t, 0.9)),
               "pass": bool(np.median(e_t) <= C["S1_tide_median_abs_log10"]
                            and np.quantile(e_t, 0.9) <= C["S1_tide_p90_abs_log10"])},
        "S2": {"n_true_amp_ge_min": int(big.sum()), "detect_rate": det_rate,
               "pass": bool(det_rate >= C["S2_m2_detect_rate_min"])},
        "S3": {"n": int(be_ok.sum()), "median_abs_log10": float(np.median(e_b)),
               "p90_abs_log10": float(np.quantile(e_b, 0.9)),
               "accept_rate_be_ge_min": acc_rate,
               "pass": bool(np.median(e_b) <= C["S3_be_median_abs_log10"]
                            and np.quantile(e_b, 0.9) <= C["S3_be_p90_abs_log10"]
                            and acc_rate >= C["S3_be_accept_rate_min"])},
        "S4": {"n": int(both.sum()), "spearman": rho,
               "pass": bool(both.sum() >= C["S4_n_min"] and rho >= C["S4_spearman_min"])},
        "S5": {"n_coastal_o1": nco, "coastal_flag_rate": float(fc.mean()) if nco else np.nan,
               "n_inland_o1": nin, "inland_flag_rate": float(fi.mean()) if nin else np.nan,
               "pass": bool(nco and nin and fc.mean() >= C["S5_coastal_flag_rate_min"]
                            and fi.mean() <= C["S5_inland_flag_rate_max"])},
        "S6": {"n": int(m2_det.sum()), "median_abs_eta_deg": float(np.median(eta)),
               "pass": bool(np.median(eta) <= C["S6_median_abs_eta_deg_max"])},
    }
    res["all_pass"] = all(v["pass"] for v in res.values())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    out = K.SYN_DIR / "synthetic_check.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the synthetic check runs once")
    K.SYN_DIR.mkdir(parents=True, exist_ok=True)
    P = L.load_pressure()
    p = P[K.SYN_PRESSURE_CELL]
    ds = draws()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(d, p) for d in ds]))
    df = pd.DataFrame(rows)
    csv = K.SYN_DIR / "synthetic_wells.csv"
    df.to_csv(csv, index=False)
    res = {"prereg_id": K.PREREG_ID, "criteria": K.SYN_CRITERIA, "result": evaluate(df),
           "n_wells": len(df), "seed": K.SYN_SEED,
           "code": {rel_to_repo(f): sha256_file(f) for f in
                    (Path(__file__), K.CODE_DIR / "x5sp_lib.py", K.CODE_DIR / "x5sp_const.py")},
           "wells_csv": {"path": rel_to_repo(csv), "sha256": sha256_file(csv)}}
    with open(out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["result"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
