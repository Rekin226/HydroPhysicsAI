"""P2c: zero classification of the kept meters of every in-grid county (port of the billing
pilot's s05 section (e) and s09 synchronized-zero screen, round 1's ``s5_zeros.py``), built on
the C0 v3 products after registration.

Classes (one row per zero meter-month of the variant's window; window pre-origin
2012-01..2022-12, full 2012-01..2026-06 (2026-07 is a partial month)):

  structural      C0 method pre_install (before the meter's install month; value 0)
  not_yet_used    after install, before the first use (leading run)
  idle            interior run <= 12 months (real idling; irrigation zeros are seasonal)
  long_idle       interior run > 12 months that ends in renewed use
  terminated      trailing run >= 12 months to the window end: a real zero for this meter,
                  ``possible_replacement`` True (abstraction may continue on an unseen meter)
  trailing_short  trailing run < 12 months at the window end (undetermined)
  never_used      no use in the window and none before it

Only flag-0 months are observations. Post-install missing readings (flag 8) outside the
meter's first..last use span count as zeros (they enter the model as 0); inside the span they
are missing, not zeros. Flag-4 months (C0's December-2022 fill) and flag-9 months are not
observations: they are excluded, and at the window end they shorten the meter's window by one
month (round 1's open-edge rule). ``sync_suspect`` marks zero meter-months in the months the
county's synchronized-zero screen flags (zero share among meters active two months before and
after exceeds its calendar-month median by more than 3 points, in the monthly or the
bimonthly group) where the meter itself was active at t-2 and t+2; they carry flag 9 (plan P2c:
kept in the data, a sensitivity member only). Other rows carry flag 0. Chiayi's 316 in-grid
meters are screened with the pooled Yunlin + Chiayi population (decided after the first build
showed the screen flagging 75 of 192 months for Chiayi alone: noise of a small population).

Usage: python -I b2_zeros.py <out_dir>    (writes zeros_<variant>.parquet,
       meter_zero_summary_<variant>.parquet and zeros_build.json)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402

WINDOW = {"preorigin": ("2012-01", "2022-12"), "full": ("2012-01", "2026-06")}
ZCLASSES = ["structural", "not_yet_used", "idle", "long_idle", "terminated", "trailing_short",
            "never_used"]
SYNC_THRESH = 0.03


def runs_of(z: np.ndarray):
    d = np.diff(np.concatenate([[0], z.astype(int), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def sync_screen(code, phase, X: np.ndarray, end: int, thresh: float = SYNC_THRESH):
    """Months <= end whose zero share (meters active at t-2 and t+2) exceeds its calendar-
    month median by more than ``thresh`` in the monthly or the bimonthly group."""
    isA = (code == "M2") & (phase == "A")
    isB = (code == "M1") & (phase == "B")
    isM = phase == "monthly"
    Z = np.nan_to_num(X)
    rows = []
    for t in range(2, end - 1):
        act = (Z[:, t - 2] > 0) & (Z[:, t + 2] > 0)
        m, bm = isM & act, (isA | isB) & act
        rows.append((L.DATES[t], float(np.mean(Z[m, t] == 0)) if m.sum() else np.nan,
                     float(np.mean(Z[bm, t] == 0)) if bm.sum() else np.nan))
    sy = pd.DataFrame(rows, columns=["date", "zero_monthly", "zero_bimonthly"]).set_index("date")
    an = sy - sy.groupby(sy.index.month).transform("median")
    flagged = sy[(an > thresh).any(axis=1)]
    return np.array([L.DATES.get_loc(d) for d in flagged.index], int), \
        [d.strftime("%Y-%m") for d in flagged.index]


def screen_input(cd: L.CountyData, variant: str):
    """(code, phase, X) of a county for the synchronized-zero screen (unobserved months NaN)."""
    w1 = L.tidx(WINDOW[variant][1])
    X = cd.M[:, :w1 + 1].astype(np.float64).copy()
    F = cd.F[:, :w1 + 1]
    X[(F == 4) | (F == 9)] = np.nan
    return cd.code, cd.phase, X


def classify(cd: L.CountyData, variant: str, screen=None):
    """``screen`` = (code, phase, X) of the population the synchronized-zero screen runs on
    (default: the county itself)."""
    w0, w1 = (L.tidx(m) for m in WINDOW[variant])
    T_data = w1 + 1
    X = cd.M[:, :T_data].astype(np.float64).copy()
    F = cd.F[:, :T_data]
    Mm = cd.Mm[:, :T_data]
    unobs = (F == 4) | (F == 9)
    X[unobs] = np.nan
    sc = screen if screen is not None else (cd.code, cd.phase, X)
    fl_t, flagged = sync_screen(sc[0], sc[1], sc[2], w1)
    Z = np.nan_to_num(X)
    nz = Z > 0
    first = np.where(nz.any(1), nz.argmax(1), -1)
    last = np.where(nz.any(1), T_data - 1 - nz[:, ::-1].argmax(1), -1)
    tt = np.arange(w0, w1 + 1)
    Lw = len(tt)
    recs, summ = [], []
    for j in range(len(cd.meters)):
        x = X[j, w0:w1 + 1].copy()
        f, l_ = first[j], last[j]
        end_unobs = unobs[j, w1]
        L_j, wend = (Lw - 1, w1 - 1) if end_unobs else (Lw, w1)
        inspan = (tt >= f) & (tt <= l_) if f >= 0 else np.zeros(Lw, bool)
        miss_out = np.isnan(x) & ~inspan & ~unobs[j, w0:w1 + 1]
        x[miss_out] = 0.0                              # out-of-span missing enters as 0
        x[L_j:] = np.nan
        z = x == 0
        if not z.any():
            summ.append((j, f, l_, False, -1))
            continue
        zc = np.full(Lw, -1, int)
        rl = np.zeros(Lw, int)
        for s_, e_ in zip(*runs_of(z), strict=True):
            n = e_ - s_
            rl[s_:e_] = n
            if f < 0 or f > wend:
                c = 6
            elif s_ == 0 and f > w0:
                c = 1
            elif e_ == L_j and l_ < wend:
                c = 4 if n >= 12 else 5
            else:
                c = 2 if n <= 12 else 3
            zc[s_:e_] = c
        pre = z & (Mm[j, w0:w1 + 1] == 1)
        zc[pre] = 0
        idx = np.where(z)[0]
        t_abs = tt[idx]
        sync = np.isin(t_abs, fl_t) & (zc[idx] != 0)
        if sync.any():
            ok = (t_abs - 2 >= 0) & (t_abs + 2 <= w1)
            act = np.zeros(len(idx), bool)
            act[ok] = (Z[j, t_abs[ok] - 2] > 0) & (Z[j, t_abs[ok] + 2] > 0)
            sync &= act
        recs.append((np.full(len(idx), j, np.int32), t_abs.astype(np.int16),
                     zc[idx].astype(np.int8), rl[idx].astype(np.int16), sync))
        term = bool(zc[L_j - 1] == 4)
        summ.append((j, f, l_, term, int(l_ + 1) if term else -1))
    z = pd.DataFrame({k: np.concatenate([r[i] for r in recs]) for i, k in
                      enumerate(("row", "t", "zclass", "run_len", "sync_suspect"))})
    z.insert(0, "county", cd.county)
    z.insert(1, "meter_idx", cd.meters["meter_idx"].to_numpy()[z["row"].to_numpy()])
    z["month"] = L.DATES[z["t"].to_numpy()].strftime("%Y-%m")
    z["zclass"] = pd.Categorical.from_codes(z["zclass"], ZCLASSES)
    z["flag"] = np.where(z["sync_suspect"], 9, 0).astype(np.uint8)
    z["possible_replacement"] = z["zclass"] == "terminated"
    s = pd.DataFrame(summ, columns=["row", "first_nz_t", "last_nz_t", "terminated",
                                    "termination_t"])
    e12 = np.zeros(len(s))
    for i, (j, l_, term) in enumerate(zip(s["row"], s["last_nz_t"], s["terminated"],
                                          strict=True)):
        if term:
            e12[i] = Z[j, max(0, l_ - 11): l_ + 1].sum()
    s["last12_kwh_before_termination"] = e12
    for c in ("first_nz", "last_nz", "termination"):
        t_ = s[f"{c}_t"].to_numpy()
        s[c] = np.where(t_ >= 0, L.DATES[np.clip(t_, 0, None)].strftime("%Y-%m"), "")
    s.insert(0, "county", cd.county)
    s.insert(1, "meter_idx", cd.meters["meter_idx"].to_numpy()[s["row"].to_numpy()])
    s["c0_terminated"] = cd.meters["terminated"].to_numpy()[s["row"].to_numpy()]
    counts = z["zclass"].value_counts().reindex(ZCLASSES).fillna(0).astype(int)
    irr = cd.mclass == "irrigation"
    zi = z[irr[z["row"].to_numpy()] & z["zclass"].isin(["idle", "long_idle"]).to_numpy()]
    info = {"window": list(WINDOW[variant]), "n_kept_meters": int(len(cd.meters)),
            "zero_meter_months": int(len(z)), "class_counts": counts.to_dict(),
            "class_shares": (counts / max(counts.sum(), 1)).round(5).to_dict(),
            "sync_flagged_months": flagged, "sync_suspect_rows": int(z["sync_suspect"].sum()),
            "n_terminated_meters": int(s["terminated"].sum()),
            "terminated_last12_GWh": float(s["last12_kwh_before_termination"].sum() / 1e6),
            "agreement_with_c0_terminated": float(np.mean(s["terminated"].to_numpy()
                                                          == s["c0_terminated"].to_numpy())),
            "irrigation_idle_zero_rows_by_moy": zi.groupby(
                L.DATES[zi["t"].to_numpy()].month).size().to_dict()}
    return z.drop(columns=["row", "t"]), s.drop(columns=["row"]), info


def main(out_dir: str) -> None:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for variant in ("preorigin", "full"):
        prod = L.open_c0(variant)
        zs, ss = [], []
        res[variant] = {}
        cds = {c: L.CountyData(variant, c, prod=prod) for c in L.COUNTIES}
        for c in L.COUNTIES:
            cd = cds[c]
            screen = None
            if c == "chiayi":
                # 316 meters cannot support the screen (zero shares of a few dozen meters
                # swing by more than 3 points): Chiayi takes the screen of the pooled
                # Yunlin + Chiayi population, whose billing groups it shares
                a, b = screen_input(cds["yunlin"], variant), screen_input(cd, variant)
                screen = tuple(np.concatenate([x, y]) for x, y in zip(a, b, strict=True))
            z, s, info = classify(cd, variant, screen)
            info["screen_population"] = "yunlin+chiayi" if c == "chiayi" else c
            zs.append(z)
            ss.append(s)
            res[variant][c] = info
            print(variant, c, info["class_counts"], info["sync_flagged_months"], flush=True)
        pd.concat(zs, ignore_index=True).to_parquet(out / f"zeros_{variant}.parquet",
                                                    index=False)
        pd.concat(ss, ignore_index=True).to_parquet(out / f"meter_zero_summary_{variant}.parquet",
                                                    index=False)
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, out / "zeros_build.json")


if __name__ == "__main__":
    main(sys.argv[1])
