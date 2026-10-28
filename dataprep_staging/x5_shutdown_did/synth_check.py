"""Synthetic recovery check of the X5 synthetic-DiD pipeline (declared in X5.shutdown_sdid.v1
before it runs).

Synthetic benchmark elevations are generated at the REAL survey dates of the date-balanced
on-grid benchmarks (annual picks; dates and locations only, no elevation is read) and passed
through the same interval-rate formula, treatment geometry (arm T2, 3 km, 3-6 km buffer),
SDID and block placebos as the real run.

Data-generating process (cumulative subsidence, cm, t in decimal years):
  s_i(t) = r_i (t - 2008) + a_i F(t) + G_blk(i)(t) + [Yunlin_i H(t)] + delta_i max(0, t - 2019)
  elevation = -s + eps, eps ~ N(0, SYN_SIGMA_SURVEY) per survey;
  r_i = 1.2 + 0.8 u_blk + 0.4 v_i (u, v ~ N(0, 1)); a_i ~ U(0.6, 1.2) treated, U(0.2, 1.2) other;
  F, G, H piecewise-linear cumulative sums of yearly N(0, sd) increments (FACTOR, BLOCK_RW, COUNTY).
Scenarios: A (gating): delta_i = SYN_DELTA on treated, no county term; A0: delta = 0 (size);
B (stress, reported): A plus the Yunlin-only term; C (spillover, reported): delta_i =
SYN_DELTA exp(-d_i / 2 km) on EVERY unit (truth = mean over treated).

Gate (declared): A bias |mean tau - truth| <= 0.25 |truth|; A coverage of tau +- 1.645 se in
[0.80, 0.98]; A0 one-sided p < 0.10 rate <= 0.15; A0 lead placebo two-sided p < 0.10 rate <= 0.15.
Power at SYN_DELTA and SYN_DELTA_SMALL is reported.

Usage: python synth_check.py [--panels N] (writes OUT_DIR/synthetic/synthetic_check.json once)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402


def setup() -> dict:
    u = L.unit_table()
    u = u[u.ongrid & u.date_balanced]
    tr, don = L.arm_sets(u, "T2")
    t0, t1 = L.interval_bounds(u.index)
    yrs = K.YEARS

    def dec(df):
        return df[yrs].apply(lambda c: c.dt.year + (c.dt.dayofyear - 1) / 365.25).to_numpy(float)

    return {"u": u, "tr": tr, "don": don, "T0": dec(t0), "T1": dec(t1), "years": yrs}


def _cum_interp(inc: np.ndarray, t: np.ndarray, y0: int) -> np.ndarray:
    """Piecewise-linear cumulative sum of yearly increments inc[..., k] for year y0 + k."""
    k = np.clip(np.floor(t - y0).astype(int), 0, inc.shape[-1] - 1)
    cum = np.concatenate([np.zeros(inc.shape[:-1] + (1,)), np.cumsum(inc, -1)], -1)
    frac = t - (y0 + k)
    base = np.take_along_axis(cum, k, -1) if cum.ndim > 1 else cum[k]
    step = np.take_along_axis(inc, k, -1) if inc.ndim > 1 else inc[k]
    return base + frac * step


def panel(S: dict, rng, scen: str) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic S (cm/yr, + = sinking) for every unit x year and the per-unit truth."""
    u = S["u"]
    n = len(u)
    y0, ny = 2007, 16
    blk = pd.factorize(u.blk)[0]
    nb = blk.max() + 1
    is_tr = u.index.isin(S["tr"])
    r = 1.2 + 0.8 * rng.normal(size=nb)[blk] + 0.4 * rng.normal(size=n)
    a = np.where(is_tr, rng.uniform(0.6, 1.2, n), rng.uniform(0.2, 1.2, n))
    F = rng.normal(0, K.SYN_FACTOR_SD, ny)
    G = rng.normal(0, K.SYN_BLOCK_RW_SD, (nb, ny))
    H = rng.normal(0, K.SYN_COUNTY_SD, ny)
    if scen in ("A", "A0", "B"):
        dlt = np.where(is_tr, 0.0 if scen == "A0" else K.SYN_DELTA, 0.0)
    elif scen == "C":
        dlt = K.SYN_DELTA * np.exp(-u.dW.to_numpy() / 2000.0)
    else:
        raise ValueError(scen)
    yun = (~u.changhua.to_numpy()).astype(float)

    def s_at(t):  # t: (n, k)
        s = r[:, None] * (t - 2008.0) + a[:, None] * _cum_interp(F, t, y0)
        s += _cum_interp(G[blk], t, y0)
        if scen == "B":
            s += yun[:, None] * _cum_interp(H, t, y0)
        return s + dlt[:, None] * np.maximum(0.0, t - 2019.0)

    ts = np.hstack([S["T0"][:, :1], S["T1"]])          # survey times: picks 2008..2021
    elev = -s_at(ts) + rng.normal(0, K.SYN_SIGMA_SURVEY, ts.shape)  # one noise per survey
    dt = np.diff(ts, axis=1)
    Sy = -np.diff(elev, axis=1) / dt
    eff = dlt[:, None] * np.diff(np.maximum(0.0, ts - 2019.0), axis=1) / dt
    return Sy, eff


def run_one(S: dict, Sy: np.ndarray, eff: np.ndarray, seed: int, B: int,
            lead: bool) -> dict:
    u = S["u"]
    yrs = S["years"]
    it = u.index.get_indexer(S["tr"])
    idn = u.index.get_indexer(S["don"])
    cols = [yrs.index(y) for y in K.PRE + K.POST]
    blocks = u.blk.to_numpy()[idn]
    res = L.sdid_with_placebo(Sy[it][:, cols], Sy[idn][:, cols], blocks, len(K.PRE), B, seed)
    post = [yrs.index(y) for y in K.POST]
    out = {"tau": res["tau"], "se": res["se_placebo"], "p1": res["p_one_sided_less_sinking"],
           "p2": res["p_two_sided"], "truth": float(eff[it][:, post].mean()),
           "placebo_taus": res["_taus"]}
    if lead:
        lc = [yrs.index(y) for y in K.LEAD_PRE + K.LEAD_POST]
        lr = L.sdid_with_placebo(Sy[it][:, lc], Sy[idn][:, lc], blocks, len(K.LEAD_PRE), B,
                                 seed + 1)
        out.update({"lead_tau": lr["tau"], "lead_p2": lr["p_two_sided"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", type=int, default=K.SYN_N_PANELS)
    ap.add_argument("--dry", action="store_true", help="2 panels, nothing written")
    a = ap.parse_args(argv)
    out_path = K.OUT_DIR / "synthetic" / "synthetic_check.json"
    if not a.dry and out_path.exists():
        raise SystemExit(f"{out_path} exists; the synthetic check runs once")
    S = setup()
    n_pan = 2 if a.dry else a.panels
    B = 20 if a.dry else K.SYN_N_PLACEBO
    t_start = time.time()
    recs = {"A": [], "B": [], "C": []}
    for k in range(n_pan):
        seed = K.SYN_SEED + 1000 * k
        # A and A0 share every random draw (same seed); only delta differs
        SyA, effA = panel(S, np.random.default_rng(seed), "A")
        Sy0, eff0 = panel(S, np.random.default_rng(seed), "A0")
        rA = run_one(S, SyA, effA, seed, B, lead=False)
        r0 = run_one(S, Sy0, eff0, seed, B, lead=True)
        rA["tau_null_same_draws"] = r0["tau"]
        recs["A"].append({"A": rA, "A0": r0})
        if k < max(1, n_pan // 2):
            SyB, effB = panel(S, np.random.default_rng(seed + 7), "B")
            recs["B"].append(run_one(S, SyB, effB, seed + 7, B, lead=False))
            SyC, effC = panel(S, np.random.default_rng(seed + 11), "C")
            recs["C"].append(run_one(S, SyC, effC, seed + 11, B, lead=False))
        print(f"panel {k + 1}/{n_pan} {time.time() - t_start:.0f}s", flush=True)

    A = pd.DataFrame([r["A"] for r in recs["A"]])
    A0 = pd.DataFrame([r["A0"] for r in recs["A"]])
    truth = float(A.truth.mean())
    lo, hi = A.tau - K.Z90 * A.se, A.tau + K.Z90 * A.se
    cover = float(((lo <= A.truth) & (A.truth <= hi)).mean())
    bias = float(A.tau.mean() - truth)
    size = float((A0.p1 < K.P_SUCCESS).mean())
    lead_fp = float((A0.lead_p2 < K.P_LEAD_NULL).mean())
    # power at the small effect: the same draws with the treated post effect scaled; exact by
    # linearity (neither weight vector depends on the treated post block; placebos are donors)
    ratio_small = K.SYN_DELTA_SMALL / K.SYN_DELTA
    tau_small = A0.tau + ratio_small * A.truth
    p_small = np.array([(1 + (np.asarray(pt) <= ts_).sum()) / (1 + len(pt))
                        for pt, ts_ in zip(A0.placebo_taus, tau_small, strict=True)])
    gate = {
        "A_bias_within_tol": abs(bias) <= K.SYN_BIAS_TOL * abs(truth),
        "A_coverage_in_band": K.SYN_COVER[0] <= cover <= K.SYN_COVER[1],
        "A0_size_ok": size <= K.SYN_SIZE_MAX,
        "A0_lead_fp_ok": lead_fp <= K.SYN_SIZE_MAX,
    }

    def summ(df):
        df = pd.DataFrame(df)
        if df.empty:
            return {}
        lo_, hi_ = df.tau - K.Z90 * df.se, df.tau + K.Z90 * df.se
        df = df.drop(columns=["placebo_taus"], errors="ignore")
        return {"n": len(df), "truth_mean": float(df.truth.mean()),
                "tau_mean": float(df.tau.mean()), "bias": float(df.tau.mean() - df.truth.mean()),
                "tau_sd": float(df.tau.std(ddof=1)), "se_mean": float(df.se.mean()),
                "coverage90": float(((lo_ <= df.truth) & (df.truth <= hi_)).mean()),
                "power_p1_lt_0.10": float((df.p1 < K.P_SUCCESS).mean())}

    res = {
        "declaration": K.PREREG_ID, "label": "synthetic recovery check (declared before running)",
        "n_panels": n_pan, "n_placebo": B, "units": {"treated": len(S["tr"]),
                                                     "donors": len(S["don"])},
        "scenario_A": {**summ(A), "linearity_max_abs": float(
            (A.tau - A.tau_null_same_draws - A.truth).abs().max())},
        "scenario_A0": {"n": len(A0), "tau_mean": float(A0.tau.mean()),
                        "tau_sd": float(A0.tau.std(ddof=1)), "size_p1_lt_0.10": size,
                        "lead_tau_mean": float(A0.lead_tau.mean()),
                        "lead_fp_p2_lt_0.10": lead_fp},
        "power_small_effect": {"delta": K.SYN_DELTA_SMALL,
                               "truth_mean": float(ratio_small * A.truth.mean()),
                               "tau_mean": float(tau_small.mean()),
                               "power_p1_lt_0.10": float((p_small < K.P_SUCCESS).mean())},
        "scenario_B_county_stress": summ(recs["B"]),
        "scenario_C_spillover": summ(recs["C"]),
        "gate": {k: bool(v) for k, v in gate.items()},
        "gate_pass": bool(all(gate.values())),
        "seconds": round(time.time() - t_start, 1),
    }
    print(json.dumps(res, indent=1))
    if not a.dry:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "x", encoding="utf-8") as f:
            f.write(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
