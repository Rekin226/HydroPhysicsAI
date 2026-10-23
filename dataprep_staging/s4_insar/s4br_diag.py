"""WP S4b repair round 2 (S4br): DIAGNOSIS of the failed criteria, on the round-1 runs P and
F, before any method change. Looks are logged beforehand by ``s4br_looks.py`` (diag_wra,
diag_insar). Nothing here is a gate, a threshold or a design choice, and nothing here reads the
NGL GNSS family (kept for the fresh re-test).

    python -I s4br_diag.py network     loop closure with and without the multi-prime bias; the
                                       interferograms spanning each network gap and the step
                                       that removed them (runs P and F; InSAR-internal)
    python -I s4br_diag.py budget      error budget of the registered G1 (run P, 18 WRA
                                       stations): gap / within-segment split, epoch noise
                                       (InSAR-only and GNSS-only), the spatially smooth
                                       high-pass part (a LiCSBAS16-like split), station-pixel
                                       loop-closure quality, the reference-plane pattern, S3
                                       v4 steps, the 2025 east rate, the 3x3 vs 7x7 window
    python -I s4br_diag.py coherence   C1: coherence of the Changhua zone cells by season,
                                       temporal baseline and year (pre-origin pairs)
    python -I s4br_diag.py yconf       Y: annual cycle the stable-strip plane injects into the
                                       Yunlin amplitudes, against the thickness proxies
    python -I s4br_diag.py geometry    conditioning of the stable-strip reference fit (extents,
                                       correlations, variance inflation, formal coefficient
                                       errors, leverage at the 18 WRA station pixels) and the
                                       east / north / height parts of the reference pattern
                                       against the G1 errors and the station northing

Outputs: ``results/dataprep/reports/S4b_evidence/repair_2026-10-18/diag_<key>.json``.
Hydro env, one CPU slot each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_evaluate as Ev  # noqa: E402
import s4b_lib as Lb  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

EV = C.EVID / "repair_2026-10-18"
LO = pd.Timestamp(C.GNSS_T0) - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
HI = pd.Timestamp(C.GNSS_T1) + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
TWO_PI = 2 * np.pi
SUMMER = (6, 7, 8, 9)


def rms(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def rsd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size > 2 else float("nan")


def ts_dir(name: str) -> Path:
    return C.RUN_DIR / name / "TS_GEOCml2"


def read_list(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def network_gaps(name: str) -> list[tuple[str, str]]:
    out, on = [], False
    for line in (ts_dir(name) / "info" / "12network_gap_info.txt").read_text().splitlines():
        if line.startswith("Gaps between"):
            on = True
            continue
        if line.startswith("Connected"):
            break
        q = line.split()
        if on and len(q) == 2:
            out.append((q[0], q[1]))
    return out


# ----------------------------------------------------------------------------- network
def loops_of(ifgs: list[str]) -> list[tuple[str, str, str]]:
    s = set(ifgs)
    out = []
    for a in ifgs:
        p, q = a[:8], a[9:]
        for b in ifgs:
            if b.startswith(q) and f"{p}_{b[9:]}" in s:
                out.append((a, b, f"{p}_{b[9:]}"))
    return out


def cmd_network(_a) -> dict:
    g = Lb.load_geom(C.PRIMARY)
    active = g["cell"] >= 0
    res = {}
    for name in ("P", "F"):
        ml = C.RUN_DIR / name / "GEOCml2"
        ifgs = sorted(p.name for p in ml.iterdir() if p.is_dir() and (p / f"{p.name}.unw").exists())
        L, W = g["x"].shape
        unw = {}
        for d in ifgs:
            u = np.fromfile(ml / d / f"{d}.unw", dtype=np.float32).reshape(L, W)
            u[u == 0] = np.nan
            unw[d] = u
        info = ts_dir(name) / "info"
        loopd = ts_dir(name) / "12loop"
        bad11 = set(read_list(info / "11bad_ifg.txt"))
        bad12 = set(read_list(info / "12bad_ifg.txt"))
        bad_1st = set(read_list(loopd / "bad_ifg_loop.txt"))
        bad_3rd = set(read_list(loopd / "bad_ifg_loopref.txt"))
        bad_noref = set(read_list(loopd / "bad_ifg_noref.txt"))
        cand = [d for d in ifgs if d not in bad11]
        rows = []
        for a, b, c in loops_of(cand):
            lp = unw[a] + unw[b] - unw[c]
            ok = np.isfinite(lp)
            if ok.sum() < 100:
                continue
            lp = lp - np.round(np.nanmedian(lp) / TWO_PI) * TWO_PI
            r0 = float(np.sqrt(np.nanmean(lp ** 2)))
            lp2 = lp - np.nanmedian(lp)
            r1 = float(np.sqrt(np.nanmean(lp2 ** 2)))
            err = np.abs(lp2) > np.pi
            dates = [a[:8], a[9:], b[9:]]
            rows.append({"loop": "_".join(dates), "ifgs": [a, b, c],
                         "rms_no_prime": r0, "rms_multi_prime": r1,
                         "frac_pix_abs_gt_pi": float(err[ok].mean()),
                         "frac_active_pix_abs_gt_pi": float(err[ok & active].mean())
                         if (ok & active).any() else None,
                         "summer": all(int(x[4:6]) in SUMMER for x in dates)})
        df = pd.DataFrame(rows)

        def first_pass_bad(col: str, thr: float) -> set[str]:
            badc, good = set(), set()
            for _, r in df.iterrows():
                (badc if r[col] >= thr else good).update(r["ifgs"])
            return badc - good

        bad_np_15 = first_pass_bad("rms_no_prime", 1.5)
        bad_mp_15 = first_pass_bad("rms_multi_prime", 1.5)
        bad_mp_pi = first_pass_bad("rms_multi_prime", np.pi)
        gaps = network_gaps(name)
        span = []
        for ga, gb in gaps:
            sp = [d for d in ifgs if d[:8] <= ga and d[9:] >= gb]
            span.append({"gap": [ga, gb], "n_spanning_ifgs": len(sp), "ifgs": [{
                "ifg": d, "removed_by": ("11" if d in bad11 else "12_noref" if d in bad_noref
                                         else "12_first_pass_no_ref" if d in bad_1st
                                         else "12_third_pass_with_ref" if d in bad_3rd
                                         else "12" if d in bad12 else "kept"),
                "max_loop_rms_no_prime": float(df[df["ifgs"].apply(lambda v, d=d: d in v)]
                                               ["rms_no_prime"].max()) if len(df) else None,
                "max_loop_rms_multi_prime": float(df[df["ifgs"].apply(lambda v, d=d: d in v)]
                                                  ["rms_multi_prime"].max()) if len(df) else None,
                "median_loop_frac_active_abs_gt_pi": float(
                    df[df["ifgs"].apply(lambda v, d=d: d in v)]
                    ["frac_active_pix_abs_gt_pi"].median()) if len(df) else None,
                "in_any_loop": bool(df["ifgs"].apply(lambda v, d=d: d in v).any())}
                for d in sp]})
        sm = df[df["summer"]]
        res[name] = {
            "n_ifgs": len(ifgs), "n_bad11": len(bad11), "n_bad12": len(bad12),
            "n_bad12_first_pass": len(bad_1st), "n_bad12_third_pass": len(bad_3rd),
            "n_bad12_noref": len(bad_noref), "n_loops": int(len(df)),
            "loop_rms_no_prime_median": float(df["rms_no_prime"].median()),
            "loop_rms_multi_prime_median": float(df["rms_multi_prime"].median()),
            "frac_loops_ge_1p5_no_prime": float((df["rms_no_prime"] >= 1.5).mean()),
            "frac_loops_ge_1p5_multi_prime": float((df["rms_multi_prime"] >= 1.5).mean()),
            "frac_loops_ge_pi_multi_prime": float((df["rms_multi_prime"] >= np.pi).mean()),
            "summer_loops": {"n": int(len(sm)),
                             "rms_no_prime_median": float(sm["rms_no_prime"].median()),
                             "rms_multi_prime_median": float(sm["rms_multi_prime"].median()),
                             "frac_active_pix_abs_gt_pi_median":
                                 float(sm["frac_active_pix_abs_gt_pi"].median())},
            "other_loops": {"n": int((~df["summer"]).sum()),
                            "rms_no_prime_median":
                                float(df.loc[~df["summer"], "rms_no_prime"].median()),
                            "rms_multi_prime_median":
                                float(df.loc[~df["summer"], "rms_multi_prime"].median()),
                            "frac_active_pix_abs_gt_pi_median":
                                float(df.loc[~df["summer"], "frac_active_pix_abs_gt_pi"].median())},
            "first_pass_bad_recomputed": {"no_prime_thr_1p5": len(bad_np_15),
                                          "no_prime_thr_1p5_matches_run":
                                              len(bad_np_15 ^ bad_1st),
                                          "multi_prime_thr_1p5": len(bad_mp_15),
                                          "multi_prime_thr_pi": len(bad_mp_pi)},
            "gaps": span,
            "note": "loop phase = unw12 + unw23 - unw13 (rad), 2 pi n removed by the median "
                    "(as LiCSBAS12's first pass); multi-prime also removes the median "
                    "(non-integer) bias. first_pass_bad_recomputed repeats LiCSBAS12's "
                    "first-pass rule (bad if only in loops with RMS >= threshold) over the "
                    "loops of the ifgs that passed step 11."}
        del unw
    return res


# ----------------------------------------------------------------------------- budget
def tied(Ins: pd.DataFrame, Gn: pd.DataFrame) -> pd.DataFrame:
    """Registered G1 tie: e_k(t) = Ins_k - mean_others(D) - Gn_k, D = Ins - Gn demeaned."""
    sel = (Ins.index >= LO) & (Ins.index <= HI)
    Ins, Gn = Ins.loc[sel], Gn.loc[sel]
    D = Ins - Gn
    D = D - D.mean()
    E = {}
    for k in Ins.columns:
        others = [o for o in Ins.columns if o != k]
        cal = Ins[k] - D[others].mean(axis=1)
        both = cal.notna() & Gn[k].notna()
        E[k] = (cal - Gn[k]).where(both)
    return pd.DataFrame(E)


def wmean(s: pd.Series, centre: str) -> tuple[float, int]:
    return Lb.window_mean(s, centre)


def segments(dates: pd.DatetimeIndex, gaps) -> np.ndarray:
    seg = np.zeros(len(dates), int)
    for n, (_ga, gb) in enumerate(gaps, 1):
        seg[dates >= pd.Timestamp(gb)] = n
    return seg


def detrended_sd(s: pd.Series, seg: np.ndarray) -> float:
    t = (s.index - s.index[0]).days.to_numpy(float) / 365.25
    res = []
    for k in np.unique(seg):
        m = (seg == k) & s.notna().to_numpy()
        if m.sum() >= 4:
            A = np.column_stack([np.ones(m.sum()), t[m]])
            b, *_ = np.linalg.lstsq(A, s.to_numpy()[m], rcond=None)
            res.append(s.to_numpy()[m] - A @ b)
    return rsd(np.concatenate(res)) if res else float("nan")


def lp_space_hp_time(v: np.ndarray, valid: np.ndarray, dates: pd.DatetimeIndex,
                     px_r: float, px_a: float, width_km: float = 2.0) -> np.ndarray:
    """LiCSBAS16-like split (diagnostic emulation, not the product filter): the spatially
    low-passed (Gaussian, sd width_km) part of the temporally high-passed series (Gaussian, sd
    3 x mean epoch interval)."""
    t = (dates - dates[0]).days.to_numpy(float) / 365.25
    wy = float(np.diff(t).mean() * 3)
    T = v.shape[0]
    out = np.full_like(v, np.nan)
    sx, sy = width_km * 1000.0 / px_r, width_km * 1000.0 / px_a
    data = np.where(valid[None], v, np.nan)
    for i in range(T):
        w = np.exp(-((t - t[i]) ** 2) / 2 / wy ** 2)
        w = w[:, None, None] * np.isfinite(data)
        num = np.nansum(data * w, 0)
        den = w.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lpt = num / den
        hp = data[i] - lpt
        ok = np.isfinite(hp)
        a = ndimage.gaussian_filter(np.where(ok, hp, 0.0), (sy, sx), mode="constant")
        b = ndimage.gaussian_filter(ok.astype(float), (sy, sx), mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(b > 1e-3, a / b, np.nan)
    return out


def cmd_budget(_a) -> dict:
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st = Ev.station_set(inv, "exclude")
    run, g, v, tab = Ev.processed("P")
    Ins, Gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
    reg = Ev.loso_eval(Ins, Gn, C.GNSS_T0, C.GNSS_T1, "loso")
    ev = reg["stations"]
    rows = {r["station"]: r for r in reg["rows"]}
    e = pd.Series({k: rows[k]["err_mm"] for k in ev})
    registered = json.loads((C.EVID / "test_gnss.json").read_text())["primary"]
    e_reg = pd.Series({r["station"]: r["err_mm"] for r in registered["rows"]})
    out = {"reproduces_registered": {"rms_mm": reg["rms_mm"],
                                     "max_abs_diff_mm": float((e - e_reg[e.index]).abs().max())}}
    E = tied(Ins[ev], Gn[ev])
    dates = E.index
    gaps = network_gaps("P")
    seg = segments(dates, gaps)
    # -- exact split of e_k into the increments across the gap intervals and the rest
    gap_part, jumps = {}, {}
    for k in ev:
        s = E[k]
        tot, js = 0.0, []
        for ga, gb in gaps:
            a, b = pd.Timestamp(ga), pd.Timestamp(gb)
            if a in s.index and b in s.index and np.isfinite(s[a]) and np.isfinite(s[b]):
                tot += float(s[b] - s[a])
            before = s[(s.index <= a)].dropna().iloc[-3:]
            after = s[(s.index >= b)].dropna().iloc[:3]
            js.append(float(after.mean() - before.mean()))
        gap_part[k] = tot
        jumps[k] = js
    gp = pd.Series(gap_part)
    rest = e - gp
    # -- epoch noise of the residual series, InSAR-only and GNSS-only differential noise
    sig = pd.Series({k: detrended_sd(E[k], seg) for k in ev})
    n0 = pd.Series({k: rows[k]["n_ep_w0"] for k in ev})
    n1 = pd.Series({k: rows[k]["n_ep_w1"] for k in ev})
    white = sig * np.sqrt(1.0 / n0 + 1.0 / n1)
    sel = (Ins.index >= LO) & (Ins.index <= HI)
    Ii, Gg = Ins.loc[sel, ev], Gn.loc[sel, ev]
    dI = Ii.sub(Ii.mean(axis=1), axis=0)
    dG = Gg.sub(Gg.mean(axis=1), axis=0)
    seg_i = segments(Ii.index, gaps)
    sig_i = pd.Series({k: detrended_sd(dI[k], seg_i) for k in ev})
    sig_g = pd.Series({k: detrended_sd(dG[k], seg_i) for k in ev})
    nan_ins = int(Ii.isna().sum().sum())
    nan_gn = int(Gg.isna().sum().sum())
    # -- LiCSBAS16-like split at the station pixels (InSAR only)
    px_r, px_a = 203.69, 221.97
    aps = lp_space_hp_time(v, run["mask"], run["dates"], px_r, px_a)
    A = {k: Lb.station_series(aps, run["mask"], int(inv.loc[k, "ml_i"]),
                              int(inv.loc[k, "ml_j"])) for k in ev}
    A = pd.DataFrame(A, index=run["dates"]).loc[sel]
    If = Ii - A
    dIf = If.sub(If.mean(axis=1), axis=0)
    sig_if = pd.Series({k: detrended_sd(dIf[k], seg_i) for k in ev})
    dA = A.sub(A.mean(axis=1), axis=0)
    # window-mean change of the smooth high-pass part alone (what the filter would remove)
    aps_change = {}
    for k in ev:
        a0, _ = wmean(dA[k].where(Ii[k].notna() & Gg[k].notna()), C.GNSS_T0)
        a1, _ = wmean(dA[k].where(Ii[k].notna() & Gg[k].notna()), C.GNSS_T1)
        aps_change[k] = a1 - a0
    aps_change = pd.Series(aps_change)
    # -- loop-closure quality at the station pixels
    L, W = g["x"].shape
    q = {}
    for nm in ("n_loop_err", "n_loop_err_rat", "resid_rms", "n_unw", "coh_avg", "n_gap"):
        f = ts_dir("P") / "results" / nm
        if f.exists():
            arr = np.fromfile(f, dtype=np.float32).reshape(L, W)
            q[nm] = pd.Series({k: float(np.nanmedian(arr[
                max(int(inv.loc[k, "ml_i"]) - 1, 0):int(inv.loc[k, "ml_i"]) + 2,
                max(int(inv.loc[k, "ml_j"]) - 1, 0):int(inv.loc[k, "ml_j"]) + 2])) for k in ev})
    qual = {nm: {"values": s.round(4).to_dict(),
                 "spearman_abs_err": float(spearmanr(s, e.abs()).correlation)}
            for nm, s in q.items()}
    big = {k: int((E[k].diff().abs() > 25.0).sum()) for k in ev}
    # -- reference plane pattern (stable-strip plane + height extrapolated to the stations)
    t = tab.set_index(pd.to_datetime(tab["date"]))
    w0 = (t.index >= LO) & (t.index <= pd.Timestamp(C.GNSS_T0) + pd.Timedelta(days=30))
    w1 = (t.index >= pd.Timestamp(C.GNSS_T1) - pd.Timedelta(days=30)) & (t.index <= HI)
    dc = {c: float(t.loc[w1, f"coef_{c}"].mean() - t.loc[w0, f"coef_{c}"].mean())
          for c in ("offset", "east_km", "north_km", "hgt_km")}
    stb = g["stable_geom"]
    xc, yc = float(np.mean(g["x"][stb])), float(np.mean(g["y"][stb]))
    pat, pat_e = {}, {}
    for k in ev:
        i, j = int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])
        u = float(g["U"][i, j])
        pe = dc["east_km"] * (g["x"][i, j] - xc) / 1000.0
        pn = dc["north_km"] * (g["y"][i, j] - yc) / 1000.0
        ph = dc["hgt_km"] * float(np.nan_to_num(g["hgt"][i, j])) / 1000.0
        pat[k] = -(pe + pn + ph) / u
        pat_e[k] = -pe / u
    pat = pd.Series(pat) - pd.Series(pat).mean()
    pat_e = pd.Series(pat_e) - pd.Series(pat_e).mean()
    xs = pd.Series({k: float(g["x"][int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])])
                    for k in ev}) / 1000.0
    ys = pd.Series({k: float(g["y"][int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])])
                    for k in ev}) / 1000.0
    # per-epoch scatter of the plane coefficients (detrended, robust) and the east-term
    # contribution it implies at the station spread
    ce = t["coef_east_km"].to_numpy()
    cn = t["coef_north_km"].to_numpy()
    tt = (t.index - t.index[0]).days.to_numpy(float) / 365.25
    A2 = np.column_stack([np.ones_like(tt), tt])
    ce_res = ce - A2 @ np.linalg.lstsq(A2, ce, rcond=None)[0]
    cn_res = cn - A2 @ np.linalg.lstsq(A2, cn, rcond=None)[0]
    # -- S3 gnss_clean full/v4 (after_the_fact_analysis): steps in span, step-corrected G1
    from common import guard
    p3 = guard.open_product(C.DATAPREP / "gnss_clean" / "full" / "v4", "after_the_fact_analysis")
    steps = p3.read_table("steps.csv")
    steps["date"] = pd.to_datetime(steps["date"])
    s_in = steps[(steps["family"] == "wra") & steps["station"].isin(ev)
                 & (steps["date"] >= LO) & (steps["date"] <= HI)]
    daily = p3.read_table("wra/daily.parquet")
    daily = daily[daily["station"].isin(ev)]
    G3 = {}
    for k in ev:
        d = daily[(daily["station"] == k) & daily["flag"].isin([0, 1])]
        s = pd.Series(d["value"].to_numpy(float) * 1000.0,
                      index=pd.DatetimeIndex(pd.to_datetime(d["date"]))).sort_index()
        G3[k] = Lb.at_epochs(Lb.smooth_daily(s), run["dates"]).to_numpy()
    G3 = pd.DataFrame(G3, index=run["dates"])
    s3 = Ev.loso_eval(Ins[ev], G3[ev], C.GNSS_T0, C.GNSS_T1, "loso")
    # -- horizontal leakage from the 2025 east rate (insar_069A/v2, after the fact)
    p2 = guard.open_product(C.DATAPREP / "insar_069A" / "v2", "after_the_fact_analysis")
    dz = np.load(p2.path("decomposition_2025.npz"), allow_pickle=False)
    leak = {}
    for k in ev:
        c = int(inv.loc[k, "cell"])
        i, j = int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])
        if bool(dz["ok"][c]):
            leak[k] = float(g["E"][i, j] / g["U"][i, j] * dz["east"][c]) * 14.0 / 12.0
    leak = pd.Series(leak)
    leak_d = leak - leak.mean()
    # -- point vs area: 3x3 vs 7x7 window change (InSAR only)
    win = {}
    for k in ev:
        i, j = int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])
        s3x3 = pd.Series(Lb.station_series(v, run["mask"], i, j, 1, 4), index=run["dates"])
        s7x7 = pd.Series(Lb.station_series(v, run["mask"], i, j, 3, 10), index=run["dates"])
        c3 = wmean(s3x3, C.GNSS_T1)[0] - wmean(s3x3, C.GNSS_T0)[0]
        c7 = wmean(s7x7, C.GNSS_T1)[0] - wmean(s7x7, C.GNSS_T0)[0]
        win[k] = c3 - c7
    win = pd.Series(win)
    win_d = win - win.mean()
    out.update({
        "stations": ev, "n": len(ev), "gaps_P": gaps,
        "err_mm": e.round(3).to_dict(),
        "split": {"gap_part_mm": gp.round(3).to_dict(), "rest_mm": rest.round(3).to_dict(),
                  "rms_e": rms(e), "rms_gap_part": rms(gp), "rms_rest": rms(rest),
                  "corr_e_gap_part": float(np.corrcoef(e, gp)[0, 1]),
                  "smoothed_gap_jumps_mm": {k: [round(x, 3) for x in v_]
                                            for k, v_ in jumps.items()},
                  "rms_smoothed_jump_gap1": rms([v_[0] for v_ in jumps.values()]),
                  "rms_smoothed_jump_gap2": rms([v_[1] for v_ in jumps.values()]),
                  "note": "gap part = sum over the two run-P gap intervals of the residual "
                          "increment (InSAR bridged by LiCSBAS's linear-velocity constraint, "
                          "minus the tie and GNSS increments); it carries the per-epoch noise "
                          "of the 4 epochs bounding the gaps. Smoothed jump = mean of 3 epochs "
                          "after minus 3 before each gap."},
        "epoch_noise": {"sigma_resid_mm": sig.round(3).to_dict(),
                        "median_sigma_resid": float(sig.median()),
                        "white_noise_window_change_mm": white.round(3).to_dict(),
                        "rms_white_noise_window_change": rms(white),
                        "sigma_insar_diff_median": float(sig_i.median()),
                        "sigma_gnss_diff_median": float(sig_g.median()),
                        "sigma_insar_diff_after_lpspace_hptime_median": float(sig_if.median()),
                        "rms_window_change_of_smooth_hp_part": rms(aps_change),
                        "smooth_hp_change_mm": aps_change.round(3).to_dict(),
                        "corr_e_smooth_hp_change": float(np.corrcoef(e, aps_change)[0, 1]),
                        "n_nan_insar_station_epochs": nan_ins, "n_nan_gnss_station_epochs": nan_gn,
                        "note": "sigma = robust sd of each series about a per-segment linear "
                                "trend over 2021-08-31..2022-12-30; InSAR-only and GNSS-only "
                                "are differences from the station mean (the tie removes the "
                                "common part); white-noise window change = sigma * "
                                "sqrt(1/n0 + 1/n1); the smooth high-pass part emulates "
                                "LiCSBAS16 (2 km spatial, 3 x mean-interval temporal "
                                "Gaussians) on the round-1 run P: diagnostic only"},
        "unwrapping": {"station_pixel_quality": qual,
                       "n_residual_increments_gt_25mm": big,
                       "total_increments_gt_25mm": int(sum(big.values()))},
        "reference": {"coef_change_w1_minus_w0": dc,
                      "plane_height_pattern_mm": pat.round(3).to_dict(),
                      "rms_plane_height_pattern": rms(pat),
                      "rms_east_term_pattern": rms(pat_e),
                      "corr_e_plane_pattern": float(np.corrcoef(e, pat)[0, 1]),
                      "pearson_e_easting": float(np.corrcoef(e, xs[e.index])[0, 1]),
                      "pearson_e_northing": float(np.corrcoef(e, ys[e.index])[0, 1]),
                      "station_x_km_range": [float(xs.min()), float(xs.max())],
                      "stable_centroid_x_km": xc / 1000.0, "stable_centroid_y_km": yc / 1000.0,
                      "coef_east_detrended_rsd_mm_per_km": rsd(ce_res),
                      "coef_north_detrended_rsd_mm_per_km": rsd(cn_res),
                      "registered_report_only_rms_mm": {"offset_hgt": 12.29, "offset": 12.41}},
        "gnss_steps_s3v4": {"steps_in_span": s_in[["station", "date", "size_m", "flag_after",
                                                    "basis"]].astype(str).to_dict("records"),
                            "g1_with_s3v4_corrected_heights": {
                                "rms_mm": s3["rms_mm"], "n": s3["n_evaluable"],
                                "err_mm": {r["station"]: r["err_mm"] for r in s3["rows"]}}},
        "horizontal_2025": {"leak_14mo_mm": leak.round(3).to_dict(),
                            "rms_demeaned_leak": rms(leak_d), "n": int(leak.size),
                            "corr_e_leak": float(np.corrcoef(e[leak.index], leak_d)[0, 1]),
                            "note": "(E_los / U) x 2025 east rate x 14/12 at the station "
                                    "cells, demeaned (the tie removes the mean); assumes the "
                                    "2025 east rates hold in 2021-22"},
        "point_vs_area": {"change_3x3_minus_7x7_mm": win.round(3).to_dict(),
                          "rms_demeaned": rms(win_d)},
    })
    return out


# ----------------------------------------------------------------------------- coherence
def cmd_coherence(_a) -> dict:
    U = Ev.units()
    z = np.load(C.RUN_DIR / "coh_P_all_preorigin.npz", allow_pickle=False)
    pairs = [str(p) for p in z["pairs"]]
    bt = z["btemp_days"]
    med = z["cell_median_coh"]
    d1 = pd.to_datetime([p[:8] for p in pairs])
    d2 = pd.to_datetime([p[9:] for p in pairs])
    summer = np.array([(a.month in SUMMER and b.month in SUMMER and a.year == b.year)
                       for a, b in zip(d1, d2, strict=True)])
    out = {}
    for zid in C.CH_ZONES:
        cells = np.array(U[zid]["cells"], int)
        coh = med[:, cells]
        frac = np.where(np.isfinite(coh), coh >= C.COH_MIN, False).mean(1)
        mcoh = np.nanmedian(np.where(np.isfinite(coh), coh, np.nan), 1)
        tab = []
        for lab, m in (("summer", summer), ("other", ~summer)):
            for b in (6, 12, 18, 24, 36, 48):
                mm = m & (bt == b)
                if mm.sum():
                    tab.append({"season": lab, "btemp_days": b, "n_pairs": int(mm.sum()),
                                "median_frac_coherent": float(np.median(frac[mm])),
                                "median_cell_coh": float(np.nanmedian(mcoh[mm]))})
        yrs = {}
        for y in (2020, 2021, 2022):
            m = summer & (d1.year == y) & (bt <= C.COH_MAX_BTEMP_D)
            yrs[y] = {"n_pairs": int(m.sum()),
                      "median_frac_coherent": float(np.median(frac[m])) if m.any() else None,
                      "median_btemp_days": float(np.median(bt[m])) if m.any() else None}
        out[zid] = {"n_cells": int(cells.size), "by_season_btemp": tab,
                    "registered_summer_pairs_by_year": yrs}
    return out


# ----------------------------------------------------------------------------- yconf
def cmd_yconf(_a) -> dict:
    run = Lb.load_run("P")
    g = Lb.load_geom(run["frame"])
    _, tab = Lb.reference(run, g, C.REF_TERMS)
    t = pd.to_datetime(tab["date"])
    stb = g["stable_geom"]
    xc = float(np.mean(g["x"][stb]))
    h = Lb.harmonic_fit(tab["coef_east_km"].to_numpy(float)[:, None], pd.DatetimeIndex(t))
    amp_e = float(h["amp"][0])
    from s4b_r2_looks import yunlin_cells
    cells = np.array(yunlin_cells(), int)
    x, _, _, _ = Ev.cell_centres_km()
    inj = amp_e * np.abs(x[cells] - xc / 1000.0) / float(np.nanmedian(g["U"][g["cell"] >= 0]))
    tp = Ev.thickness_proxies()
    from common import guard
    p2 = guard.open_product(C.DATAPREP / "insar_069A" / "v2", "after_the_fact_analysis")
    sa = np.load(p2.path("seasonal_amplitude_preorigin.npz"), allow_pickle=False)
    keys = list(sa.files)
    amp = None
    for k in ("cell_amp", "amp_cells", "amplitude_cells", "cells_amp"):
        if k in keys:
            amp = sa[k]
    out = {"east_coef_annual_amp_mm_per_km_los": amp_e, "stable_centroid_x_km": xc / 1000.0,
           "n_cells": int(cells.size),
           "injected_amp_mm_vertical_p10_p50_p90": [float(np.percentile(inj, q))
                                                     for q in (10, 50, 90)],
           "spearman_injected_vs_west_distance": float(spearmanr(
               inj, tp["thick_west_of_pinchout_km"][cells]).correlation),
           "seasonal_npz_keys": keys}
    if amp is not None:
        a = np.asarray(amp, float)[cells]
        ok = np.isfinite(a)
        out["spearman_injected_vs_amplitude"] = float(spearmanr(inj[ok], a[ok]).correlation)
        out["median_amplitude_mm"] = float(np.nanmedian(a))
    return out


# ----------------------------------------------------------------------------- geometry
def cmd_geometry(_a) -> dict:
    run = Lb.load_run("P")
    g = Lb.load_geom(run["frame"])
    st = g["stable_geom"] & run["mask"]
    X = Lb.ref_design(g, C.REF_TERMS)[st].astype(float)
    xs, ys, hs = X[:, 1], X[:, 2], X[:, 3]
    cols = {"east_km": xs, "north_km": ys, "hgt_km": hs}
    vif = {}
    for k, v in cols.items():
        others = np.column_stack([np.ones_like(v)] + [w for kk, w in cols.items() if kk != k])
        b, *_ = np.linalg.lstsq(others, v, rcond=None)
        r2 = 1 - np.var(v - others @ b) / np.var(v)
        vif[k] = float(1 / (1 - r2))
    XtX_inv = np.linalg.inv(X.T @ X)
    _, tab = Lb.reference(run, g, C.REF_TERMS)
    sd = float(np.nanmedian(tab["resid_sd_los_mm"].to_numpy()[1:]))
    se = {k: float(sd * np.sqrt(XtX_inv[i, i])) for i, k in enumerate(C.REF_TERMS)}
    # effective spatial correlation of the residuals makes the formal errors optimistic: the
    # detrended per-epoch scatter of the coefficients (diag_budget) is the empirical one
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    b = json.loads((EV / "diag_budget.json").read_text())
    ev = b["stations"]
    e = pd.Series(b["err_mm"])[ev]
    t = tab.set_index(pd.to_datetime(tab["date"]))
    w0 = (t.index >= LO) & (t.index <= pd.Timestamp(C.GNSS_T0) + pd.Timedelta(days=30))
    w1 = (t.index >= pd.Timestamp(C.GNSS_T1) - pd.Timedelta(days=30)) & (t.index <= HI)
    dc = {c: float(t.loc[w1, f"coef_{c}"].mean() - t.loc[w0, f"coef_{c}"].mean())
          for c in ("east_km", "north_km", "hgt_km")}
    stb = g["stable_geom"]
    xc, yc = float(np.mean(g["x"][stb])), float(np.mean(g["y"][stb]))
    parts = {"east": {}, "north": {}, "hgt": {}}
    lev, xk, yk = {}, {}, {}
    for k in ev:
        i, j = int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])
        u = float(g["U"][i, j])
        d = np.array([1.0, (g["x"][i, j] - xc) / 1000.0, (g["y"][i, j] - yc) / 1000.0,
                      float(np.nan_to_num(g["hgt"][i, j])) / 1000.0])
        lev[k] = float(d @ XtX_inv @ d)
        xk[k], yk[k] = d[1], d[2]
        parts["east"][k] = -dc["east_km"] * d[1] / u
        parts["north"][k] = -dc["north_km"] * d[2] / u
        parts["hgt"][k] = -dc["hgt_km"] * d[3] / u
    out = {"stable_pixels": int(st.sum()),
           "x_km_rel_range": [float(xs.min()), float(xs.max())],
           "y_km_rel_range": [float(ys.min()), float(ys.max())],
           "hgt_km_range": [float(hs.min()), float(hs.max())],
           "corr_x_hgt": float(np.corrcoef(xs, hs)[0, 1]),
           "corr_y_hgt": float(np.corrcoef(ys, hs)[0, 1]),
           "corr_x_y": float(np.corrcoef(xs, ys)[0, 1]), "vif": vif,
           "median_resid_sd_los_mm": sd, "formal_se_white": se,
           "station_x_rel_km": {k: round(v, 2) for k, v in xk.items()},
           "station_y_rel_km": {k: round(v, 2) for k, v in yk.items()},
           "station_leverage_over_max_stable_leverage": {
               k: round(v / float(np.max(np.einsum("ij,jk,ik->i", X, XtX_inv, X))), 2)
               for k, v in lev.items()},
           "coef_change_w1_minus_w0": dc, "parts": {}}
    yy = pd.Series(yk)[ev]
    for nm, v in parts.items():
        s_ = pd.Series(v)[ev]
        s_ = s_ - s_.mean()
        out["parts"][nm] = {"rms_mm": rms(s_), "corr_with_err": float(np.corrcoef(e, s_)[0, 1]),
                            "values_mm": s_.round(3).to_dict()}
    gp = pd.Series(b["split"]["gap_part_mm"])[ev]
    rest = pd.Series(b["split"]["rest_mm"])[ev]
    out["corr_northing"] = {"err": float(np.corrcoef(e, yy)[0, 1]),
                            "gap_part": float(np.corrcoef(gp, yy)[0, 1]),
                            "rest": float(np.corrcoef(rest, yy)[0, 1])}
    for nm in ("east", "north", "hgt"):
        s_ = pd.Series(parts[nm])[ev]
        s_ = s_ - s_.mean()
        out["parts"][nm]["rms_err_without_this_part_mm"] = rms(e - s_)
    # the reference surface's own increments across the two run-P gap intervals, and the gap
    # part of the error once they are taken out (what is left is the LiCSBAS bridge, the tie
    # and the GNSS increments across the gaps)
    gaps = network_gaps("P")
    sg = {}
    for k in ev:
        i, j = int(inv.loc[k, "ml_i"]), int(inv.loc[k, "ml_j"])
        u = float(g["U"][i, j])
        d = np.array([1.0, (g["x"][i, j] - xc) / 1000.0, (g["y"][i, j] - yc) / 1000.0,
                      float(np.nan_to_num(g["hgt"][i, j])) / 1000.0])
        tot = 0.0
        for ga, gb in gaps:
            ca = t.loc[pd.Timestamp(ga), [f"coef_{c}" for c in C.REF_TERMS]].to_numpy(float)
            cb = t.loc[pd.Timestamp(gb), [f"coef_{c}" for c in C.REF_TERMS]].to_numpy(float)
            tot += -float(d @ (cb - ca)) / u
        sg[k] = tot
    sg = pd.Series(sg)[ev]
    sg = sg - sg.mean()
    gp_raw = gp - sg
    out["gap_part_reference_surface_mm"] = {"values": sg.round(3).to_dict(), "rms": rms(sg),
                                            "corr_northing": float(np.corrcoef(sg, yy)[0, 1])}
    out["gap_part_without_reference_surface"] = {
        "values": gp_raw.round(3).to_dict(), "rms": rms(gp_raw),
        "corr_northing": float(np.corrcoef(gp_raw, yy)[0, 1]),
        "corr_with_err": float(np.corrcoef(gp_raw, e)[0, 1])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=["network", "budget", "coherence", "yconf", "geometry"])
    a = ap.parse_args(argv)
    fn = {"network": cmd_network, "budget": cmd_budget, "coherence": cmd_coherence,
          "yconf": cmd_yconf, "geometry": cmd_geometry}[a.key]
    res = fn(a)
    EV.mkdir(parents=True, exist_ok=True)
    Lb.jdump(res, EV / f"diag_{a.key}.json")
    print(json.dumps(res, default=Lb._js)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
