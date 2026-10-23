"""WP W2: build et0/v<N> - daily FAO-56 ET0 at the 61 calibration points, 2012-01..2026-09.

Implements pre-registration results/dataprep/preregistration/W2.et0_overlap.v2.json
(GMT days; v1, which assumed local days, was superseded before it ran), recorded with the
WP A1 tool as W2.et0_overlap.v3 (design unchanged; tests T0, T2, T3 and completeness). The
scoring look is checked and logged with harness.prereg.assert_can_score before any test
statistic is computed:

  2012-01-01..2022-12-31  calibration cache results/et/openmeteo_et0_2012_2022.npz   source 0
  2023-01-01..2025-06-30  forcing-experiment fetch (2026-09-30, GMT)                  source 1
  2025-07-01..2026-06-30  main fetch openmeteo_et0/w2_gmt_2025_2026 (2026-10-12)      source 2
  2026-07-01..2026-09-30  pre-freeze re-fetch openmeteo_et0/w2_gmt_refetch_2026q3     source 3
                          (provisional: recent days may still be revised)

Tests: T0 same grid cells; T1 2022 cache vs forcing experiment (plan acceptance: per-point
MAE < 0.1 mm/d); T2 2025-01..06 main vs forcing experiment; T3 re-fetch vs main; completeness.
Level match: per-point additive offset from the 2022 overlap, applied from 2023-01 on when
the pre-registered stability rule holds (both variants are written either way).

Usage: python build_et0.py --out <product_dir>        (writes the product files)
       python build_et0.py --dry                      (prints the analysis only)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from wcommon import ET_CACHE, FORCING_EXP_ET, OM_L0, OM_VAR, read_json_untrusted, write_json_new

MAIN = OM_L0 / "w2_gmt_2025_2026"
REFETCH = OM_L0 / "w2_gmt_refetch_2026q3"
HISTORY = OM_L0 / "w2_gmt_history_2012_2021_st1_st42"
PROBE = OM_L0 / "w2_probe_st1_2022"
START, END = "2012-01-01", "2026-09-30"
SPLICE = {0: ("2012-01-01", "2022-12-31"), 1: ("2023-01-01", "2025-06-30"),
          2: ("2025-07-01", "2026-06-30"), 3: ("2026-07-01", "2026-09-30")}
SOURCE_NAMES = {0: "calibration cache results/et/openmeteo_et0_2012_2022.npz (committed "
                   "2026-06-21; Open-Meteo via AquaScope, GMT days)",
                1: "data_fetch_api/forcing_experiment/et (Open-Meteo archive, GMT, fetched "
                   "2026-09-30)",
                2: "data_fetch_api/new_measurements/openmeteo_et0/w2_gmt_2025_2026 (Open-Meteo "
                   "archive, GMT, fetched 2026-10-12)",
                3: "data_fetch_api/new_measurements/openmeteo_et0/w2_gmt_refetch_2026q3 "
                   "(Open-Meteo archive, GMT, fetched 2026-10-12, provisional)"}
FETCH_DATE = {0: "2026-06-21 (commit date of the cache; fetch time not recorded)",
              1: "2026-09-30", 2: "2026-10-12", 3: "2026-10-12"}
THRESH = 0.1
PREREG_ID = "W2.et0_overlap.v3"
LEVEL_MIN = 0.02
STAB_TOL = 0.02


# ----------------------------------------------------------------------------- loading
def load_cache():
    c = np.load(ET_CACHE, allow_pickle=False)
    ids = [str(x) for x in c["well_ids"]]
    dates = pd.DatetimeIndex(pd.to_datetime(c["dates"]))
    return ids, pd.DataFrame(c["et0"].astype(float).T, index=dates, columns=ids)


def load_dir(d: Path, ids=None) -> tuple[pd.DataFrame, dict]:
    """Open-Meteo responses <st>.json in d -> (DataFrame days x points, meta per point)."""
    series, meta = {}, {}
    files = sorted(d.glob("st*.json"))
    for p in files:
        st = p.stem
        if ids is not None and st not in ids:
            continue
        j = read_json_untrusted(p)
        tz = j.get("timezone")
        if tz != "GMT":
            raise ValueError(f"{p}: timezone {tz!r}, expected GMT")
        vals = [np.nan if v is None else float(v) for v in j["daily"][OM_VAR]]
        series[st] = pd.Series(vals, index=pd.DatetimeIndex(pd.to_datetime(j["daily"]["time"])))
        meta[st] = {"lat": j["latitude"], "lon": j["longitude"], "elevation": j["elevation"]}
    return pd.DataFrame(series), meta


def _win(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df.loc[a:b]


def per_point(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per-point daily MAE, bias (a - b), max |d|, r over the common finite days."""
    cols = [c for c in a.columns if c in b.columns]
    idx = a.index.intersection(b.index)
    rows = []
    for c in cols:
        x, y = a.loc[idx, c].to_numpy(), b.loc[idx, c].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        d = x[ok] - y[ok]
        rows.append({"point": c, "n_days": int(ok.sum()), "mae": float(np.mean(np.abs(d))),
                     "bias": float(np.mean(d)), "max_abs": float(np.max(np.abs(d))),
                     "n_identical": int(np.sum(np.abs(d) < 1e-9)),
                     "r": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else None})
    return pd.DataFrame(rows).set_index("point")


def _stats(pp: pd.DataFrame) -> dict:
    return {"n_points": int(len(pp)), "mae_median": float(pp["mae"].median()),
            "mae_max": float(pp["mae"].max()), "mae_min": float(pp["mae"].min()),
            "bias_mean": float(pp["bias"].mean()), "bias_min": float(pp["bias"].min()),
            "bias_max": float(pp["bias"].max()), "max_abs_max": float(pp["max_abs"].max()),
            "r_min": float(pp["r"].min()) if pp["r"].notna().any() else None,
            "identical_frac": float(pp["n_identical"].sum() / pp["n_days"].sum())}


# ----------------------------------------------------------------------------- analysis
def analyse(require_refetch: bool = True) -> dict:
    ids, cache = load_cache()
    fe, fe_meta = load_dir(FORCING_EXP_ET, ids)
    main, main_meta = load_dir(MAIN, ids)
    if set(fe.columns) != set(ids) or set(main.columns) != set(ids):
        raise ValueError("forcing experiment or main fetch does not cover the 61 points")
    rf = None
    if REFETCH.exists() and (REFETCH / "SOURCE.json").exists():
        rf, _ = load_dir(REFETCH, ids)
    elif require_refetch:
        raise FileNotFoundError(f"{REFETCH} (pre-freeze re-fetch) not complete")
    res = {"points": ids}
    # A1 look log: refuse to score unless W2.et0_overlap.v3 predates every look at its units
    from harness import prereg as _prereg
    can = _prereg.assert_can_score(PREREG_ID, "W2W4")
    res["a1_scoring"] = {"prereg": PREREG_ID, "sha256": can["sha256"], "tests": can["tests"],
                         "other_tests_prior_looks": can["other_tests_prior_looks"],
                         "logged_utc": dt.datetime.now(dt.timezone.utc).strftime(
                             "%Y-%m-%dT%H:%M:%SZ")}

    # T0 same cells
    t0 = [st for st in ids if (fe_meta[st]["lat"], fe_meta[st]["lon"], fe_meta[st]["elevation"])
          != (main_meta[st]["lat"], main_meta[st]["lon"], main_meta[st]["elevation"])]
    res["T0_same_points"] = {"n_mismatch": len(t0), "mismatch": t0, "pass": not t0,
                             "unique_cells": int(len({(m["lat"], m["lon"]) for m in
                                                      main_meta.values()})),
                             "unique_cell_elevation": int(len({tuple(m.values()) for m in
                                                               main_meta.values()}))}
    # T1 plan acceptance on 2022
    pp1 = per_point(_win(cache, "2022-01-01", "2022-12-31"), _win(fe, "2022-01-01",
                                                                    "2022-12-31"))
    res["T1_cache_vs_fe_2022"] = {**_stats(pp1), "criterion": f"max per-point MAE < {THRESH}",
                                  "pass": bool(pp1["mae"].max() < THRESH)}
    # T2 chain 2025-01..06
    pp2 = per_point(_win(main, "2025-01-01", "2025-06-30"), _win(fe, "2025-01-01",
                                                                   "2025-06-30"))
    res["T2_main_vs_fe_2025H1"] = {**_stats(pp2), "criterion": f"max per-point MAE < {THRESH}",
                                   "pass": bool(pp2["mae"].max() < THRESH)}
    # T3 revision
    if rf is not None:
        pp3 = per_point(_win(rf, "2026-07-01", "2026-09-30"), _win(main, "2026-07-01",
                                                                     "2026-09-30"))
        n_changed = int((pp3["n_days"] - pp3["n_identical"]).sum())
        res["T3_refetch_vs_main_2026Q3"] = {**_stats(pp3), "n_changed_values": n_changed,
                                            "criterion": f"max per-point MAE < {THRESH}",
                                            "pass": bool(pp3["mae"].max() < THRESH)}
    # level-match offsets (2022, pre-origin)
    delta = pp1["bias"].copy()                       # mean(cache - fe) per point
    res["level_offsets"] = {"definition": "delta_p = mean(cache - forcing experiment) over "
                                          "2022, additive, per point",
                            "fan_mean": float(delta.mean()), "min": float(delta.min()),
                            "max": float(delta.max()),
                            "per_point": {k: float(v) for k, v in delta.items()}}
    c22 = _win(cache, "2022-01-01", "2022-12-31")
    f22 = _win(fe, "2022-01-01", "2022-12-31")
    add = (c22 - f22.add(delta, axis=1)).to_numpy()
    mul = (c22 - f22.mul(c22.sum() / f22.sum(), axis=1)).to_numpy()
    res["offset_form"] = {"rmse_raw": float(np.sqrt(np.nanmean((c22 - f22).to_numpy() ** 2))),
                          "rmse_after_additive": float(np.sqrt(np.nanmean(add ** 2))),
                          "rmse_after_multiplicative": float(np.sqrt(np.nanmean(mul ** 2)))}
    # stability 2012-2021 at st1, st42
    hist, _ = load_dir(HISTORY)
    stab = {}
    ok_all = True
    for st in hist.columns:
        d = (cache.loc["2012-01-01":"2021-12-31", st] - hist[st]).dropna()
        ann = d.groupby(d.index.year).mean()
        dev = ann - float(delta[st])
        ok = bool((dev.abs() <= STAB_TOL).all())
        ok_all &= ok
        stab[st] = {"offset_2022": float(delta[st]),
                    "annual_offset": {int(k): float(v) for k, v in ann.items()},
                    "max_abs_dev_from_2022": float(dev.abs().max()), "within_tol": ok,
                    "daily_r_2012_2021": float(np.corrcoef(
                        cache.loc[d.index, st], hist.loc[d.index, st])[0, 1]),
                    "daily_mae_2012_2021": float(d.abs().mean())}
    apply_level = bool(abs(float(delta.mean())) >= LEVEL_MIN and ok_all)
    res["level_match_rule"] = {"fan_mean_abs_ge_0.02": bool(abs(float(delta.mean())) >=
                                                            LEVEL_MIN),
                               "stability": stab, "stable_all": ok_all,
                               "apply_level_match_to_primary": apply_level}
    # probe: local vs GMT at st1 2022
    if PROBE.exists():
        pj = read_json_untrusted(PROBE / "st1.json")
        loc = pd.Series([np.nan if v is None else v for v in pj["daily"][OM_VAR]],
                        index=pd.DatetimeIndex(pd.to_datetime(pj["daily"]["time"])))
        c1 = cache.loc[loc.index, "st1"]
        f1 = fe.loc[loc.index, "st1"]
        mloc, mf, mc = (s.resample("MS").mean() for s in (loc, f1, c1))
        res["probe_local_vs_gmt_st1_2022"] = {
            "daily_mae_local_vs_cache": float((loc - c1).abs().mean()),
            "daily_mae_gmt_vs_cache": float((f1 - c1).abs().mean()),
            "r_cache_local": float(np.corrcoef(loc, c1)[0, 1]),
            "r_cache_gmt": float(np.corrcoef(f1, c1)[0, 1]),
            "monthly_mean_mae_local_vs_gmt": float((mloc - mf).abs().mean()),
            "monthly_mean_mae_gmt_vs_cache": float((mf - mc).abs().mean())}
    res["_frames"] = {"cache": cache, "fe": fe, "main": main, "refetch": rf,
                      "delta": delta, "main_meta": main_meta}
    return res


# ----------------------------------------------------------------------------- build
def splice(res: dict) -> dict:
    fr = res["_frames"]
    ids = res["points"]
    days = pd.date_range(START, END, freq="D")
    raw = pd.DataFrame(np.nan, index=days, columns=ids)
    src = pd.Series(255, index=days, dtype="int64")
    frames = {0: fr["cache"], 1: fr["fe"], 2: fr["main"], 3: fr["refetch"]}
    for code, (a, b) in SPLICE.items():
        f = frames[code]
        if f is None:
            raise ValueError(f"source {code} missing")
        raw.loc[a:b] = f.loc[a:b, ids].reindex(pd.date_range(a, b, freq="D")).to_numpy()
        src.loc[a:b] = code
    if (src == 255).any():
        raise ValueError("unassigned days in the splice")
    delta = fr["delta"].reindex(ids)
    lev = raw.copy()
    post = raw.index >= pd.Timestamp("2023-01-01")
    lev.loc[post] = raw.loc[post].add(delta, axis=1)
    lev = lev.clip(lower=0.0)
    apply_level = res["level_match_rule"]["apply_level_match_to_primary"]
    prim = lev if apply_level else raw
    flag = np.zeros(raw.shape, dtype=np.uint8)
    if apply_level:
        flag[post, :] = 1
    flag[~np.isfinite(prim.to_numpy())] = 8
    return {"days": days, "raw": raw, "lev": lev, "prim": prim, "flag": flag, "src": src,
            "apply_level": apply_level, "delta": delta}


def completeness(sp: dict) -> dict:
    w = sp["prim"].loc["2023-01-01":END]
    frac = np.isfinite(w.to_numpy()).mean(axis=0)
    return {"window": ["2023-01-01", END], "min_finite_frac": float(frac.min()),
            "n_missing_days_total": int((~np.isfinite(w.to_numpy())).sum()),
            "criterion": ">= 0.99 finite days per point", "pass": bool(frac.min() >= 0.99)}


def climatology_bias(sp: dict, cache: pd.DataFrame) -> pd.DataFrame:
    clim_m = cache.loc["2012-01-01":"2021-12-31"].resample("MS").mean().mean(axis=1)
    by = clim_m.groupby(clim_m.index.month)
    mu, sd = by.mean(), by.std(ddof=1)
    rows = []
    for name, arr in (("raw_splice", sp["raw"]), ("level_matched", sp["lev"])):
        m = arr.loc["2023-01-01":END].resample("MS").mean().mean(axis=1)
        for t, v in m.items():
            b = v - mu[t.month]
            rows.append({"variant": name, "month": t.strftime("%Y-%m"), "fan_mean_mm_d": v,
                         "clim_2012_2021_mm_d": mu[t.month], "bias_mm_d": b,
                         "z": b / sd[t.month]})
    return pd.DataFrame(rows)


def jsonable(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    res = analyse(require_refetch=not a.dry)
    if a.dry:
        print(json.dumps({k: v for k, v in jsonable(res).items() if k != "points"}, indent=1,
                         default=float)[:6000])
        return 0
    out = Path(a.out)
    sp = splice(res)
    comp = completeness(sp)
    cb = climatology_bias(sp, res["_frames"]["cache"])
    ids = res["points"]
    meta = res["_frames"]["main_meta"]
    from wcommon import points_wgs84
    pts = points_wgs84().set_index("st_id").loc[ids]
    dates = np.array([d.strftime("%Y-%m-%d") for d in sp["days"]])
    src = sp["src"].to_numpy().astype(np.uint8)
    common = dict(
        well_ids=np.array(ids), dates=dates,
        coords=np.column_stack([pts["lat"], pts["lon"]]).astype("float32"),
        grid_lat=np.array([meta[s]["lat"] for s in ids], "float64"),
        grid_lon=np.array([meta[s]["lon"] for s in ids], "float64"),
        point_elevation_m=np.array([meta[s]["elevation"] for s in ids], "float64"),
        day_convention=np.array("GMT (UTC) calendar days"),
        units=np.array("mm/day (FAO-56 reference ET0, Open-Meteo et0_fao_evapotranspiration)"))
    np.savez_compressed(
        out / "et0_daily_2012_2026.npz",
        et0=sp["prim"].to_numpy().T.astype("float32"),
        flag=sp["flag"].T.copy(),
        et0_raw_splice=sp["raw"].to_numpy().T.astype("float32"),
        et0_level_matched=sp["lev"].to_numpy().T.astype("float32"),
        level_offset_mm_d=sp["delta"].to_numpy().astype("float32"),
        primary_variant=np.array("level_matched" if sp["apply_level"] else "raw_splice"),
        source=src, source_codes=np.array([f"{k}: {v}" for k, v in SOURCE_NAMES.items()]),
        fetch_date=np.array([FETCH_DATE[int(s)] for s in src]),
        provisional=(src == 3), **common)
    pre = sp["days"] <= pd.Timestamp("2022-12-31")
    np.savez_compressed(
        out / "et0_daily_2012_2022_preorigin.npz",
        et0=sp["raw"].loc[pre].to_numpy().T.astype("float32"),
        flag=np.zeros((len(ids), int(pre.sum())), np.uint8), source=src[pre],
        source_codes=np.array([f"0: {SOURCE_NAMES[0]}"]),
        fetch_date=np.array([FETCH_DATE[0]] * int(pre.sum())),
        **{**common, "dates": dates[pre]})
    cb.to_csv(out / "climatology_bias_2023_2026.csv", index=False)
    gate = {k: res[k]["pass"] for k in ("T0_same_points", "T1_cache_vs_fe_2022",
                                        "T2_main_vs_fe_2025H1", "T3_refetch_vs_main_2026Q3")}
    gate["completeness"] = comp["pass"]
    v = {"preregistration": "results/dataprep/preregistration/W2.et0_overlap.v2.json",
         "preregistration_a1_record": "results/dataprep/preregistration/W2.et0_overlap.v3.json",
         "superseded_preregistration": "results/dataprep/preregistration/W2.et0_overlap.v1."
                                       "json (never run)",
         **jsonable(res), "completeness": comp,
         "climatology_bias_summary": {
             n: {"mean_bias_mm_d": float(g["bias_mm_d"].mean()),
                 "mean_abs_z": float(g["z"].abs().mean()),
                 "n_months_abs_z_gt_2": int((g["z"].abs() > 2).sum()),
                 "annual_mean_bias_mm_d": {str(y): float(x) for y, x in
                                           g.assign(y=g["month"].str[:4]).groupby("y")
                                           ["bias_mm_d"].mean().items()}}
             for n, g in cb.groupby("variant")},
         "splice": {str(k): list(v) for k, v in SPLICE.items()},
         "primary_variant": "level_matched" if sp["apply_level"] else "raw_splice",
         "gate": gate, "gate_passed": bool(all(gate.values())),
         "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    write_json_new(out / "validation.json", v)
    print(json.dumps({"gate": gate, "primary": v["primary_variant"],
                      "T1": res["T1_cache_vs_fe_2022"], "completeness": comp}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
