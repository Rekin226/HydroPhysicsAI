"""Synthetic recovery check of estimator (c) (declaration ``X5.energy_volume.synthetic_c.v1``).

For every candidate test well (the fresh unit's members: confined wells with an ls_cache
series in 2010-2011), hourly heads are built for 2010-01..2011-12 from the SAME kept
irrigation pumps, X3 aquifer shares and 2010-2011 census kWh, with a known volume per kWh
c_true pumped through the periodic Theis response (exact in frequency: each pump's hourly
rate series times K0(r sqrt(i 2 pi f S / T)) / (2 pi T)), with T and S the estimator's own.
Each pump has its own on-rate (lognormal, median 40 m3/h), duty U(6, 12) h, start hour
N(7, 1.5) h with daily jitter U(-1, 1) h, and pumps on as many random days of the month as
its monthly volume needs. Added: O1/K1/P1/M2/S2/N2 tides, a 2 mm atmospheric S1, a 1 m
seasonal cycle, a random walk (2 cm/day), 3 mm white noise, 1 cm quantisation and 5 % gaps
in blocks. The registered estimator then runs unchanged (QC, AMP v1, background, G, c_w).
No head observation is read.

Pass (declared before the run): for each c_true in (6, 3) m3/kWh, the pooled median of
c_hat / c_true over wells with an estimate lies in [0.75, 1.25], and so does each county's
median. Reported only: a misspecified variant (start-hour spread sd 3 h) and the LSQ
extractor.

Usage: python -I synth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_energy_volume import amp_inversion as A  # noqa: E402
from x5_energy_volume import x5const as K  # noqa: E402

_LX = np.linspace(np.log(1e-5), np.log(60.0), 40001)
_KT = A.kelvin_k0(np.exp(_LX))


def kernel_interp(x: np.ndarray) -> np.ndarray:
    lx = np.log(np.clip(x, 1e-5, 60.0))
    out = np.interp(lx, _LX, _KT.real) + 1j * np.interp(lx, _LX, _KT.imag)
    return np.where(x >= 60.0, 0.0, out)


def candidates() -> pd.DataFrame:
    """Confined (LayerAttribute 'C') stations with an ls_cache raw series holding rows in
    the test window (row count only; no value is read)."""
    st = A.stations()
    st = st[st["attr"] == "C"].copy()
    keep = []
    for sid in st["sid"]:
        f = K.LS_CACHE / f"gw__{sid}__raw.parquet"
        if not f.exists():
            continue
        ix = pd.read_parquet(f, columns=[]).index
        ix = pd.to_datetime(ix)
        if ((ix >= pd.Timestamp(K.TEST_WINDOW[0])) & (ix < pd.Timestamp(K.TEST_WINDOW[1]))).any():
            keep.append(sid)
    return st[st["sid"].isin(keep)].reset_index(drop=True)


def synth_series(geo: dict, S: float, c_true: float, rng, start_sd: float = 1.5) -> np.ndarray:
    idx = pd.date_range(*K.TEST_WINDOW, freq="h", inclusive="left")
    nh = len(idx)
    months = pd.date_range(K.TEST_WINDOW[0], periods=24, freq="MS")
    mi = pd.Index(A.MONTHS).get_indexer(months)
    E = geo["E"][:, mi]
    m_start = np.array([(m - months[0]).days * 24 for m in months])
    dim = months.days_in_month.to_numpy()
    T = K.T_PRIMARY / 86400.0
    f = np.fft.rfftfreq(nh, d=3600.0)
    fcpd = f * 86400.0
    band = (fcpd >= 0.05) & (fcpd <= 6.0)
    spec = np.zeros(len(f), dtype=complex)
    n = len(geo["r"])
    q_on = np.exp(rng.normal(np.log(40.0), 0.5, n)) / 3600.0      # m3/s
    duty = rng.uniform(6.0, 12.0, n)
    st0 = np.clip(rng.normal(7.0, start_sd, n), 0.0, 14.0)
    for c0 in range(0, n, 400):
        sl = slice(c0, min(n, c0 + 400))
        R = np.zeros((sl.stop - sl.start, nh), dtype="float32")
        for j, i in enumerate(range(sl.start, sl.stop)):
            for m in range(24):
                V = c_true * E[i, m]
                if V <= 0:
                    continue
                H = V / (q_on[i] * 3600.0)                         # hours at the on-rate
                nd = int(min(dim[m], np.ceil(H / duty[i])))
                hd = H / nd
                days = rng.choice(dim[m], size=nd, replace=False)
                for dd in days:
                    s = m_start[m] + dd * 24 + st0[i] + rng.uniform(-1, 1)
                    e = s + hd
                    k0, k1 = int(np.floor(s)), int(np.ceil(e))
                    for k in range(max(k0, 0), min(k1, nh)):
                        ov = min(e, k + 1) - max(s, k)
                        if ov > 0:
                            R[j, k] += q_on[i] * ov
        Rf = np.fft.rfft(R, axis=1)
        x = np.maximum(geo["r"][sl], K.R_MIN)[:, None] * np.sqrt(
            2 * np.pi * f[band][None, :] * S / T)
        Hk = kernel_interp(x) * geo["share"][sl][:, None] / (2 * np.pi * T)
        spec[band] += (Rf[:, band] * Hk).sum(axis=0)
    draw = np.fft.irfft(spec, n=nh)
    th = np.arange(nh, dtype=float)
    head = -draw
    o1 = rng.uniform(0.002, 0.004)
    for ln, amp in (("O1", o1), ("K1", 1.41 * o1), ("P1", 0.47 * o1), ("M2", 0.005),
                    ("S2", 0.0023), ("N2", 0.001), ("S1", K.ATM_S1_M)):
        head += amp * np.cos(np.radians(A.DEG_H[ln]) * th + rng.uniform(0, 2 * np.pi))
    head += 1.0 * np.sin(2 * np.pi * th / 8766.0 + rng.uniform(0, 2 * np.pi))
    head += np.cumsum(rng.normal(0, 0.02 / np.sqrt(24), nh))
    head += rng.normal(0, 0.003, nh)
    head = np.round(head + 10.0, 2)
    gap = np.zeros(nh, bool)
    while gap.mean() < 0.05:
        a = rng.integers(0, nh - 200)
        gap[a:a + rng.integers(6, 200)] = True
    head[gap] = np.nan
    return pd.Series(head, index=idx)


def run_one(w: pd.Series, P: dict, bet: pd.Series, c_true: float, seed: int,
            start_sd: float = 1.5) -> dict:
    rng = np.random.default_rng(seed)
    geo = A.well_geometry(w, P)
    S, src = A.storage(w["sid"], bet)
    months = pd.date_range(K.TEST_WINDOW[0], periods=24, freq="MS")
    G = A.g_series(geo, months, K.T_PRIMARY, S)
    s = synth_series(geo, S, c_true, rng, start_sd).dropna()
    x, valid, idx = A.qc_hourly(s, *K.TEST_WINDOW)
    r = A.well_estimate(x, valid, idx, G)
    r2 = A.well_estimate(x, valid, idx, G, extractor="lsq")
    return {"sid": w["sid"], "c_true": c_true, "start_sd": start_sd, "c_hat": r["c"],
            "ratio": r["c"] / c_true if np.isfinite(r["c"]) else np.nan,
            "ratio_lsq": r2["c"] / c_true if np.isfinite(r2["c"]) else np.nan,
            "n_months": r["n_months"], "A1p_mean_m": r["A1p_mean_m"],
            "background_m": r["background_m"], "n_pumps_2km": geo["n_2km"],
            "n_pumps_4km": geo["n_4km"], "S": S, "S_source": src}


def main() -> int:
    out_dir = K.REPORT_DIR["synthetic"]
    rep = out_dir / "report.json"
    if rep.exists():
        raise SystemExit(f"{rep} exists; run once")
    P = A.load_pumps()
    bet = A.be_table()
    cand = candidates()
    cand["county_pumps"] = [A.county_of_point(r.x, r.y, P["x"], P["y"], P["county"])
                            for r in cand.itertuples()]
    rows = []
    for k, w in cand.iterrows():
        geo_n = A.well_geometry(w, P)["n_2km"]
        if geo_n < K.MIN_PUMPS:
            rows.append({"sid": w["sid"], "excluded": f"{geo_n} kept irrigation pumps "
                                                      f"within 2 km < {K.MIN_PUMPS}"})
            continue
        for c_true in K.SYNTH_C_TRUE:
            r = run_one(w, P, bet, c_true, K.SEED_SYNTH + 10 * k + int(c_true))
            r["county"] = w["county_pumps"]
            rows.append(r)
            print(json.dumps({k_: r[k_] for k_ in ("sid", "c_true", "ratio", "ratio_lsq",
                                                   "n_months")}), flush=True)
        r = run_one(w, P, bet, K.SYNTH_C_TRUE[0], K.SEED_SYNTH + 10 * k + 7, start_sd=3.0)
        r["county"] = w["county_pumps"]
        r["variant"] = "misspecified: start-hour sd 3 h (reported only)"
        rows.append(r)
    df = pd.DataFrame(rows)
    K.OUT.mkdir(parents=True, exist_ok=True)
    (K.OUT / "synthetic").mkdir(exist_ok=True)
    df.to_csv(K.OUT / "synthetic" / "synthetic_wells.csv", index=False)
    main_rows = df[df.get("variant").isna()] if "variant" in df else df
    main_rows = main_rows[main_rows["ratio"].notna()] if "ratio" in main_rows else main_rows
    crit = {}
    for c_true in K.SYNTH_C_TRUE:
        d = main_rows[main_rows["c_true"] == c_true]
        e = {"pooled": {"n": int(len(d)), "median_ratio": float(d["ratio"].median()),
                        "median_ratio_lsq": float(d["ratio_lsq"].median())}}
        for cn, g in d.groupby("county"):
            e[cn] = {"n": int(len(g)), "median_ratio": float(g["ratio"].median()),
                     "median_ratio_lsq": float(g["ratio_lsq"].median())}
        e["pass"] = bool(all(abs(v["median_ratio"] - 1) <= K.SYNTH_BAND
                             for k_, v in e.items() if isinstance(v, dict) and v["n"] > 0))
        crit[f"c_true_{c_true}"] = e
    mis = df[df.get("variant").notna()] if "variant" in df else df.iloc[:0]
    res = {"declaration": K.PREREG["synthetic"],
           "criterion": "for each c_true, pooled and per-county median c_hat/c_true in "
                        f"[{1 - K.SYNTH_BAND}, {1 + K.SYNTH_BAND}]",
           "results": crit, "pass": bool(all(v["pass"] for v in crit.values())),
           "reported_only": {"misspecified_start_sd_3h_median_ratio":
                             float(mis["ratio"].median()) if len(mis) else None,
                             "n": int(mis["ratio"].notna().sum()) if len(mis) else 0},
           "wells_csv": "results/dataprep/x5_energy_volume/synthetic/synthetic_wells.csv"}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(rep, "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
