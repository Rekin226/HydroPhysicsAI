"""WP S4b repair round: after-the-fact diagnostics (report only; look logged beforehand by
``s4b_r2_looks.py repair_diag``). Nothing here is a gate, a threshold or a design choice.

  pf        runs P and F at active-cell pixels over the G1 span (InSAR only): change
            differences per interval between consecutive common epochs, classed as across a
            network gap or within a segment; the whole last segment; the G1 change itself
  pieces    the G1 error of runs P and F at the 18 registered stations, with the registered
            leave-one-out offset tie, split exactly into the increments across the two 2022
            network gaps and the rest; single-interval increment errors within segments
  refcoef   stable-area reference coefficients: annual harmonic of the run-P east/north plane
            coefficients; 2020-2025 linear drift of the run-F coefficients
  yconf     test Y's easting confound on the Yunlin cells: rho(amplitude, easting / northing),
            rho(census density 5 km, easting), partial Spearman of amplitude and density given
            easting; flag-9 census cell-months 2020-22

    python -I s4b_r2_diag.py     (hydro env, one CPU slot) -> S4b_evidence/repair_diag.json
"""
from __future__ import annotations

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
from scipy.stats import rankdata, spearmanr  # noqa: E402

LO = pd.Timestamp(C.GNSS_T0) - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
HI = pd.Timestamp(C.GNSS_T1) + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)


def gaps(name: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    js = json.loads((C.EVID / "repair_gaps.json").read_text())[name]
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in js["gaps"]]


def rms(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


# ----------------------------------------------------------------------------- pf
def pf(R: dict) -> dict:
    (rP, gP, vP, _), (rF, gF, vF, _) = R["P"], R["F"]
    assert np.array_equal(gP["x"], gF["x"]), "runs P and F share the primary grid"
    common = rP["dates"].intersection(rF["dates"])
    common = common[(common >= LO) & (common <= HI)]
    iP = [int(np.flatnonzero(rP["dates"] == d)[0]) for d in common]
    iF = [int(np.flatnonzero(rF["dates"] == d)[0]) for d in common]
    sel = rP["mask"] & rF["mask"] & (gP["cell"] >= 0)
    P = vP[iP][:, sel]
    F = vF[iF][:, sel]
    gp = gaps("P")
    rows = []
    for k in range(len(common) - 1):
        a, b = common[k], common[k + 1]
        cls = "within"
        for n, (ga, gb) in enumerate(gp, 1):
            if a <= ga and gb <= b:
                cls = f"across_gap{n}"
        d = (F[k + 1] - F[k]) - (P[k + 1] - P[k])
        rows.append({"interval": f"{a.date()}_{b.date()}", "days": int((b - a).days),
                     "class": cls, "rms_mm": rms(d), "median_mm": float(np.nanmedian(d)),
                     "n_pix": int(np.isfinite(d).sum())})
    df = pd.DataFrame(rows)
    w = df[df["class"] == "within"]
    w24 = w[(w["days"] >= 18) & (w["days"] <= 30)]
    seg3 = (common >= gp[-1][1])
    k0, k1 = int(np.flatnonzero(seg3)[0]), len(common) - 1
    d3 = (F[k1] - F[k0]) - (P[k1] - P[k0])
    w0 = (common >= LO) & (common <= pd.Timestamp(C.GNSS_T0) + pd.Timedelta(days=30))
    w1 = (common >= pd.Timestamp(C.GNSS_T1) - pd.Timedelta(days=30)) & (common <= HI)
    with np.errstate(all="ignore"):
        cP = np.nanmean(P[w1], 0) - np.nanmean(P[w0], 0)
        cF = np.nanmean(F[w1], 0) - np.nanmean(F[w0], 0)
    dg = cF - cP
    return {"n_common_epochs": len(common), "first": str(common[0].date()),
            "last": str(common[-1].date()), "n_pixels": int(sel.sum()),
            "intervals": df.round(3).to_dict("records"),
            "across_gap1": df[df["class"] == "across_gap1"][["interval", "rms_mm",
                                                            "median_mm"]].to_dict("records"),
            "across_gap2": df[df["class"] == "across_gap2"][["interval", "rms_mm",
                                                            "median_mm"]].to_dict("records"),
            "within_rms_mm": {"median": float(w["rms_mm"].median()),
                              "max": float(w["rms_mm"].max()), "n": int(len(w))},
            "within_18_30d_rms_mm": {"median": float(w24["rms_mm"].median()),
                                     "max": float(w24["rms_mm"].max()), "n": int(len(w24))},
            "last_segment": {"span": f"{common[k0].date()}_{common[k1].date()}",
                             "rms_mm": rms(d3), "median_mm": float(np.nanmedian(d3))},
            "g1_change": {"rms_mm": rms(dg), "median_mm": float(np.nanmedian(dg)),
                          "n_epochs_w0": int(w0.sum()), "n_epochs_w1": int(w1.sum())}}


# ----------------------------------------------------------------------------- pieces
def tied_errors(Ins: pd.DataFrame, Gn: pd.DataFrame) -> pd.DataFrame:
    """Registered G1 tie (``s4b_lib.loso``): error series e_k = Ins_k - mean_others(D) - Gn_k
    over the analysis span, D = Ins - Gn demeaned per station."""
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


def wmean(s: pd.Series, centre: str) -> float:
    c = pd.Timestamp(centre)
    w = s.loc[c - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D):c + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)]
    w = w.dropna()
    return float(w.mean()) if len(w) >= C.GNSS_MIN_EPOCHS_PER_WINDOW else np.nan


def pieces(R: dict, inv: pd.DataFrame, stations: dict) -> dict:
    out = {}
    for nm in ("P", "F"):
        run, g, v, _ = R[nm]
        st = stations[nm]
        Ins, Gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
        E = tied_errors(Ins, Gn)
        gp = gaps(nm)
        gp = [(a, b) for a, b in gp if a >= LO and b <= HI]
        rows = []
        for s in st:
            e = E[s]
            err = wmean(e, C.GNSS_T1) - wmean(e, C.GNSS_T0)
            gpart = [float(e.get(b, np.nan) - e.get(a, np.nan)) for a, b in gp]
            rows.append({"station": s, "err_mm": err,
                         **{f"gap{n}_mm": x for n, x in enumerate(gpart, 1)},
                         "gap_sum_mm": float(np.sum(gpart)),
                         "rest_mm": err - float(np.sum(gpart))})
        df = pd.DataFrame(rows)
        ok = df[["err_mm", "gap_sum_mm"]].notna().all(axis=1)
        # single-interval increments within segments inside the span (control)
        idx = E.index
        inc = []
        for k in range(len(idx) - 1):
            a, b = idx[k], idx[k + 1]
            if any(a <= ga and gb <= b for ga, gb in gp):
                continue
            d = (E.iloc[k + 1] - E.iloc[k]).to_numpy(float)
            inc.append({"interval": f"{a.date()}_{b.date()}", "days": int((b - a).days),
                        "rms_mm": rms(d), "n": int(np.isfinite(d).sum())})
        inc = pd.DataFrame(inc)
        inc24 = inc[(inc["days"] >= 18) & (inc["days"] <= 30)]
        out[nm] = {"n_stations": int(ok.sum()), "stations": st,
                   "g1_rms_mm": rms(df.loc[ok, "err_mm"]),
                   **{f"gap{n}_rms_mm": rms(df.loc[ok, f"gap{n}_mm"])
                      for n in range(1, len(gp) + 1)},
                   **{f"gap{n}_mean_mm": float(df.loc[ok, f"gap{n}_mm"].mean())
                      for n in range(1, len(gp) + 1)},
                   "gap_sum_rms_mm": rms(df.loc[ok, "gap_sum_mm"]),
                   "rest_rms_mm": rms(df.loc[ok, "rest_mm"]),
                   "corr_err_gap_sum": float(np.corrcoef(df.loc[ok, "err_mm"],
                                                         df.loc[ok, "gap_sum_mm"])[0, 1]),
                   "gaps": [f"{a.date()}_{b.date()}" for a, b in gp],
                   "within_single_interval_rms_mm": {
                       "median": float(inc["rms_mm"].median()),
                       "p90": float(inc["rms_mm"].quantile(0.9)), "n": int(len(inc))},
                   "within_18_30d_rms_mm": {"median": float(inc24["rms_mm"].median()),
                                            "p90": float(inc24["rms_mm"].quantile(0.9)),
                                            "n": int(len(inc24))},
                   "rows": df.round(3).to_dict("records")}
    return out


# ----------------------------------------------------------------------------- refcoef
def harmonic(t: np.ndarray, y: np.ndarray) -> dict:
    w = 2 * np.pi * t
    X = np.column_stack([np.ones_like(t), t - t.mean(), np.cos(w), np.sin(w), np.cos(2 * w),
                         np.sin(2 * w)])
    ok = np.isfinite(y)
    B, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return {"rate_per_yr": float(B[1]), "annual_amp": float(np.hypot(B[2], B[3])),
            "annual_peak_doy": float((np.degrees(np.arctan2(B[3], B[2])) % 360) / 360 * 365.25),
            "n": int(ok.sum())}


def refcoef(R: dict) -> dict:
    out = {}
    g = R["P"][1]
    st = g["stable_geom"]
    out["stable_centroid_x_km"] = float(np.mean(g["x"][st]) / 1000.0)
    out["stable_centroid_y_km"] = float(np.mean(g["y"][st]) / 1000.0)
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    mask = np.asarray(z["mask"]).astype(bool)
    r, c = np.nonzero(mask)
    xa = (float(z["x0"]) + (c + 0.5) * float(z["dx"])) / 1000.0
    out["active_cells_x_km_min_max"] = [float(xa.min()), float(xa.max())]
    for nm in ("P", "F"):
        tab = R[nm][3]
        t = Lb.decimal_years(pd.to_datetime(tab["date"]))
        out[nm] = {k: harmonic(t, tab[f"coef_{k}"].to_numpy(float))
                   for k in ("east_km", "north_km", "hgt_km", "offset")}
    a = out["P"]["east_km"]["annual_amp"]
    out["P_east_annual_amp_los_mm_at_30_50km"] = [30 * a, 50 * a]
    return out


# ----------------------------------------------------------------------------- yconf
def partial_spearman(a, b, z) -> float:
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    X = np.column_stack([np.ones_like(rz), rz])
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def yconf() -> dict:
    U = Ev.units()
    yl = sorted(set().union(*[set(U[u]["cells"]) for u in C.YUNLIN_UNITS]))
    amp = np.load(C.EVID / "seasonal_amp_cells_P.npy")
    E = Ev.census_kwh_per_year()
    D = Ev.densities(E)
    T = Ev.thickness_proxies()
    x, y, _, _ = Ev.cell_centres_km()
    c = np.array([k for k in yl if np.isfinite(amp[k])], int)
    dens = D[f"dens_{C.DENSITY_PRIMARY_KM:g}km"]
    out = {"n_cells": int(c.size),
           "rho_amp_easting": float(spearmanr(amp[c], x[c])[0]),
           "rho_amp_northing": float(spearmanr(amp[c], y[c])[0]),
           "rho_density5_easting": float(spearmanr(dens[c], x[c])[0]),
           "rho_density5_west_of_pinchout": float(spearmanr(
               dens[c], T["thick_west_of_pinchout_km"][c])[0]),
           "rho_placebo_easting": float(spearmanr(D["placebo_ring"][c], x[c])[0]),
           "partial_rho_amp_density5_given_easting": partial_spearman(amp[c], dens[c], x[c]),
           "partial_rho_amp_placebo_given_easting": partial_spearman(
               amp[c], D["placebo_ring"][c], x[c]),
           "note": "both thickness proxies are functions of easting only (zonal Ske stepped at "
                   "205/182 km; max(0, 205 - x)); exploratory, after the fact"}
    sys.path.insert(0, str(C.STAGING))
    from common import guard
    p = guard.open_product(C.DATAPREP / "pump_electricity" / "preorigin" / "v1", "validator")
    z = np.load(p.path("cells_total.npz"), allow_pickle=False)
    dates = pd.to_datetime([str(d) for d in z["dates"]], format="%Y-%m")
    sel = (dates.year >= C.CENSUS_YEARS[0]) & (dates.year <= C.CENSUS_YEARS[1])
    f = z["flag_all"][:, sel]
    out["census_flag9_cell_months_2020_22"] = {"all_active": int((f == 9).sum()),
                                               "yunlin": int((f[yl] == 9).sum()),
                                               "of_all_active": int(f.size)}
    return out


def main() -> int:
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    g1 = json.loads((C.EVID / "test_gnss.json").read_text())
    af = json.loads((C.EVID / "after.json").read_text())
    stations = {"P": g1["primary"]["stations"], "F": af["F_g1"]["loso"]["stations"]}
    R = {nm: Ev.processed(nm) for nm in ("P", "F")}
    out = {"pf": pf(R), "pieces": pieces(R, inv, stations), "refcoef": refcoef(R),
           "yconf": yconf()}
    # consistency: the split reproduces the registered and after-the-fact G1 numbers
    out["check"] = {"P_g1_registered": g1["primary"]["rms_mm"],
                    "P_g1_split": out["pieces"]["P"]["g1_rms_mm"],
                    "F_g1_after": af["F_g1"]["loso"]["rms_mm"],
                    "F_g1_split": out["pieces"]["F"]["g1_rms_mm"]}
    Lb.jdump(out, C.EVID / "repair_diag.json")
    brief = {"check": out["check"],
             "pf": {k: out["pf"][k] for k in ("across_gap1", "across_gap2", "within_rms_mm",
                                              "within_18_30d_rms_mm", "last_segment",
                                              "g1_change")},
             "pieces": {nm: {k: v for k, v in out["pieces"][nm].items() if k not in
                             ("rows", "stations")} for nm in ("P", "F")},
             "refcoef": out["refcoef"], "yconf": out["yconf"]}
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
