"""P2d build: the post-2016 synthetic census layer (plan WP P2d) for every in-grid county, as
two additive members ``low`` and ``high`` (the third member, ``zero``, is the census as it
stands: no file). Members are unweighted (plan P2d; no evidence in hand can weight them).

Rule (fixed in the P2def.v1 registration):

* monthly cohorts from 2016-07 to the variant's last complete month (pre-origin 2022-12, full
  2026-06) of rate/12 pumps per township x class;
* rate = kept pumps installed in 2010-2015 / 6 (``high``, the 2013 registration surge
  included) or in 2010-2015 without 2013 / 5 (``low``);
* placement in proportion to the township's installs of the member's years per 1 km cell
  (cell -1 = outside the active grid: counted in county totals, absent from the arrays);
* energy of a cohort of n pumps of age a in month t = n * g(c, a) * P(c, t): g = relative age
  ramp of the 2010-2015 cohorts (both members; ages 1..6, held after 6; classes with < 200
  cohort pumps use the pooled ramp), P = mean energy per installed kept pump of the class in
  month t (recorded arm, C0 v3 of the variant);
* flags: 0 (value 0, method pre_layer) before 2016-07, 5 (synthetic) from 2016-07, 8 (NaN) at
  2026-07 of the full variant (the census month is incomplete; P cannot be formed);
* sd (uncalibrated: one back-test, no coverage check possible) = s_rel(county) x value, s_rel =
  max(|fan kWh error|, median township count error) of the county's mid-2013 back-test
  (Chiayi: the larger of the Yunlin and Changhua values); counties add in quadrature, classes
  and cells of one county are fully correlated in the sd.

Usage: python -I d3_layer.py build preorigin|full <work_dir> <backtest_json>
       python -I d3_layer.py check <work_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

MEMBERS = {"low": D.LOW_YEARS, "high": D.HIGH_YEARS}
VAR = {"preorigin": {"T": D.T_PRE, "last": "2022-12", "use": "prior"},
       "full": {"T": D.T_FULL, "last": "2026-06", "use": "forcing_recorded_arm"}}
M_PRE_LAYER, M_SYNTH, M_MISSING = 0, 1, 2
METHODS = {M_PRE_LAYER: "pre_layer", M_SYNTH: "synthetic_post2016_install_rate_age_ramp",
           M_MISSING: "census_month_incomplete"}


def s_rel_from_backtest(bt: dict) -> dict:
    s = {}
    for k in ("yunlin", "changhua"):
        b = bt[k]
        s[k] = float(max(abs(b["fan_kwh_rel_err"]), b["median_township_count_rel_err"]))
    s["chiayi"] = max(s["yunlin"], s["changhua"])
    return s


def build(variant: str, work: Path, bt_path: Path) -> None:
    t_start = time.time()
    v = VAR[variant]
    T, t_last = v["T"], D.tidx(v["last"])
    t0 = D.tidx(D.LAYER_START)
    s_rel = s_rel_from_backtest(D.load_json(bt_path))
    l1 = D.l1_table(use="prior")
    K, C, A = len(D.COUNTIES), len(D.CLASSES), D.A_CELLS
    arr = {m: {"value": np.zeros((K, C, A, T)), "n": np.zeros((K, C, A, T))}
           for m in MEMBERS}
    rows, params = [], {"s_rel": s_rel, "counties": {}}
    for ki, k in enumerate(D.COUNTIES):
        p = D.Pumps(variant, k, v["use"], l1=l1)
        E = p.energy()
        P = D.per_pump_index(E, p.cls, p.inst, np.ones(len(p.uid), bool), T)
        iy = D.install_year(p.inst)
        coh = p.inst_valid & np.isin(iy, D.HIGH_YEARS)
        R = D.ramp(E, p.cls, p.inst, coh, P, data_end=t_last)
        coh_lo = p.inst_valid & np.isin(iy, D.LOW_YEARS)
        R_lo = D.ramp(E, p.cls, p.inst, coh_lo, P, data_end=t_last)   # sensitivity only
        cp = {"ramp_g": {c: R["g"][c] for c in D.CLASSES}, "ramp_source": R["source"],
              "ramp_n_pumps": R["n_pumps"],
              "ramp_g_ex2013_sensitivity": {c: R_lo["g"][c] for c in D.CLASSES},
              "per_pump_index_kwh": {c: P[c][:T] for c in D.CLASSES},
              "n_kept_pumps": int(len(p.uid)), "n_install_valid": int(p.inst_valid.sum()),
              "members": {}}
        for m, years in MEMBERS.items():
            rate = D.rates_by_town_class(p.town, p.cls, p.inst, years)
            w = D.cell_weights(p.town, p.cell, p.inst, years)
            S = D.synth_monthly(rate, w, R["g"], P, t0, t_last, t0, t_last)
            for ci, c in enumerate(D.CLASSES):
                arr[m]["value"][ki, ci, :, t0:t_last + 1] = S[c]["cells"]
                arr[m]["n"][ki, ci, :, t0:t_last + 1] = S[c]["n_cells"]
                for j, t in enumerate(range(t0, t_last + 1)):
                    rows.append({"county": k, "member": m, "class": c, "date": D.ym(t),
                                 "value_county_kwh": S[c]["county"][j],
                                 "value_in_grid_kwh": S[c]["cells"][:, j].sum(),
                                 "n_county": S[c]["n_county"][j],
                                 "n_in_grid": S[c]["n_cells"][:, j].sum()})
            win = w.groupby("town").apply(lambda g_: float(g_.loc[g_["cell"] >= 0, "w"].sum()),
                                          include_groups=False)
            cp["members"][m] = {"years": list(years),
                                "rate_by_town_class": rate.to_dict(orient="records"),
                                "in_grid_weight_by_town": win.to_dict(),
                                "rate_total_per_yr": float(rate["rate_per_yr"].sum())}
        params["counties"][k] = cp
        print(variant, k, "done", f"{time.time() - t_start:.0f}s", flush=True)
        del p, E
    work.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_parquet(work / f"{variant}_county_table.parquet", index=False)
    D.save_json(params, work / f"{variant}_params.json")
    for m in MEMBERS:
        np.savez_compressed(work / f"{variant}_{m}_raw.npz", value=arr[m]["value"],
                            n=arr[m]["n"])
    print("built", variant, f"{time.time() - t_start:.0f}s")


def assemble(variant: str, work: Path, out_dir: Path) -> dict:
    """Final arrays (value/flag/sd/method per class x cell x month, by county, totals)."""
    v = VAR[variant]
    T, t_last, t0 = v["T"], D.tidx(v["last"]), D.tidx(D.LAYER_START)
    params = D.load_json(work / f"{variant}_params.json")
    s_rel = params["s_rel"]
    g = np.load(D.C0[variant] / "cells_by_class.npz", allow_pickle=False)
    meta = {x: g[x] for x in ("mask_hash", "grid_hash", "nx", "ny", "dx", "x0", "y0",
                              "n_active")}
    dates = np.array([D.ym(t) for t in range(T)])
    s_k = np.array([s_rel[k] for k in D.COUNTIES])
    info = {}
    for m in MEMBERS:
        raw = np.load(work / f"{variant}_{m}_raw.npz")
        vbk, nbk = raw["value"], raw["n"]
        flag = np.zeros(vbk.shape[1:], dtype=np.uint8)
        flag[..., t0:t_last + 1] = 5
        method = np.zeros(vbk.shape[1:], dtype=np.uint8)
        method[..., t0:t_last + 1] = M_SYNTH
        if t_last < T - 1:
            flag[..., t_last + 1:] = 8
            method[..., t_last + 1:] = M_MISSING
        fbk = np.broadcast_to(flag, vbk.shape).copy()
        sdbk = vbk * s_k[:, None, None, None]
        value = vbk.sum(0)
        sd = np.sqrt((sdbk ** 2).sum(0))
        vt = value.sum(0)
        sdt = np.sqrt(((vbk.sum(1) * s_k[:, None, None]) ** 2).sum(0))
        dom = D.CLASSES.index("domestic")
        vnd = vt - value[dom]
        sdnd = np.sqrt((((vbk.sum(1) - vbk[:, dom]) * s_k[:, None, None]) ** 2).sum(0))
        vbk2, sdbk2 = vbk.copy(), sdbk.copy()
        if t_last < T - 1:
            for a in (value, sd, vt, sdt, vnd, sdnd, vbk2, sdbk2):
                a[..., t_last + 1:] = np.nan
        n_all = nbk.sum(0)
        out = out_dir / f"synthetic_post2016_{m}.npz"
        np.savez_compressed(
            out, value=value, flag=flag, sd=sd, method=method,
            value_by_county=vbk2, flag_by_county=fbk, sd_by_county=sdbk2,
            value_total_all=vt, flag_total_all=flag[0], sd_total_all=sdt,
            value_total_nodomestic=vnd, flag_total_nodomestic=flag[0], sd_total_nodomestic=sdnd,
            n_pumps=n_all, n_pumps_by_county=nbk, member=np.array(m),
            classes=np.array(D.CLASSES), counties=np.array(D.COUNTIES), dates=dates,
            unit=np.array("kWh/month, model units (cleaned-census kWh); additive layer"),
            method_codes=np.array([f"{c}:{n}" for c, n in METHODS.items()]),
            sd_label=np.array("uncalibrated (s_rel x value from one mid-2013 back-test; "
                              "counties in quadrature; cells and classes of a county fully "
                              "correlated)"),
            **meta)
        info[m] = {"file": out.name, "in_grid_GWh_2022": float(
            np.nansum(vt[:, D.tidx("2022-01"):D.tidx("2022-12") + 1]) / 1e6)}
    # county table (all members) with flags and sd
    tab = pd.read_parquet(work / f"{variant}_county_table.parquet")
    tab["flag"] = 5
    tab["method"] = METHODS[M_SYNTH]
    tab["sd_county_kwh"] = tab["value_county_kwh"] * tab["county"].map(s_rel)
    tab["sd_in_grid_kwh"] = tab["value_in_grid_kwh"] * tab["county"].map(s_rel)
    tab["unit"] = "kWh/month, model units; additive synthetic layer"
    tab["product_version"] = f"census_post2016/{variant}"
    tab.to_parquet(out_dir / "county_class_monthly.parquet", index=False)
    D.save_json(params, out_dir / "layer_parameters.json")
    return info


# ----------------------------------------------------------------------------- checks
def recompute_ingrid_total(params: dict, variant: str) -> dict:
    """Independent recomputation of the in-grid energy and expected counts per county x class
    x member x month from the stored parameters (explicit loop over monthly cohorts)."""
    v = VAR[variant]
    T, t_last, t0 = v["T"], D.tidx(v["last"]), D.tidx(D.LAYER_START)
    out = {}
    for k, cp in params["counties"].items():
        for m, mp in cp["members"].items():
            rate = pd.DataFrame(mp["rate_by_town_class"])
            win = mp["in_grid_weight_by_town"]
            for c in D.CLASSES:
                g = np.array(cp["ramp_g"][c], dtype=float)
                P = np.nan_to_num(np.array(cp["per_pump_index_kwh"][c], dtype=float), nan=0.0)
                rc = rate[rate["cls"] == c]
                rin = float(sum(r * win.get(tw, 0.0) for tw, r in zip(rc["town"],
                                                                        rc["rate_per_yr"], strict=False)))
                e = np.zeros(T)
                n = np.zeros(T)
                for mc in range(t0, t_last + 1):
                    ts = np.arange(mc, t_last + 1)
                    age = np.minimum((ts - mc) // 12 + 1, 6)
                    e[ts] += rin / 12.0 * g[age - 1] * P[ts]
                    n[ts] += rin / 12.0
                out[(k, m, c)] = (e, n)
    return out


def check(work: Path, out_dirs: dict) -> dict:
    res = {"items": {}, "pass": True}

    def item(name, ok, detail):
        res["items"][name] = {"pass": bool(ok), "detail": detail}
        res["pass"] = res["pass"] and bool(ok)

    vals = {}
    for variant, od in out_dirs.items():
        v = VAR[variant]
        T, t_last, t0 = v["T"], D.tidx(v["last"]), D.tidx(D.LAYER_START)
        params = D.load_json(od / "layer_parameters.json")
        g0 = np.load(D.C0[variant] / "cells_by_class.npz", allow_pickle=False)
        rec = recompute_ingrid_total(params, variant)
        for m in MEMBERS:
            z = np.load(od / f"synthetic_post2016_{m}.npz", allow_pickle=False)
            pre = f"{variant}.{m}"
            vals[(variant, m)] = z["value_by_county"]
            shape_ok = (z["value"].shape == (6, D.A_CELLS, T) and
                        z["value_by_county"].shape == (3, 6, D.A_CELLS, T) and
                        str(z["mask_hash"]) == str(g0["mask_hash"]) and
                        str(z["grid_hash"]) == str(g0["grid_hash"]) and
                        list(z["dates"]) == [D.ym(t) for t in range(T)])
            item(f"{pre}.a_shapes_grid_dates", shape_ok, {"shape": list(z["value"].shape)})
            val, fl, sd = z["value"], z["flag"], z["sd"]
            nanv = np.isnan(val)
            b_ok = (bool((nanv == (fl == 8)).all()) and bool((val[~nanv] >= 0).all())
                    and bool((fl[..., :t_last + 1] != 8).all()))
            item(f"{pre}.b_nan_iff_flag8_nonneg", b_ok,
                 {"n_flag8": int((fl == 8).sum()), "n_neg": int((val[~nanv] < 0).sum())})
            c_ok = (bool((val[..., :t0] == 0).all()) and bool((fl[..., :t0] == 0).all())
                    and bool((fl[..., t0:t_last + 1] == 5).all()))
            item(f"{pre}.c_zero_before_start_flag5_after", c_ok, {"t0": D.ym(t0)})
            sdk = z["sd_by_county"]
            vk = z["value_by_county"]
            sr = np.array([params["s_rel"][k] for k in D.COUNTIES])
            fin = np.isfinite(vk)
            d_ok = (bool(np.isfinite(sd[fl == 5]).all()) and bool((sd[fl == 5] >= 0).all())
                    and bool((sd[fl == 0] == 0).all()) and
                    float(np.max(np.abs(sdk[fin] - (vk * sr[:, None, None, None])[fin]),
                                 initial=0.0)) <= 1e-9 * max(1.0, float(np.nanmax(vk))))
            item(f"{pre}.d_sd_rule", d_ok, {"s_rel": params["s_rel"]})
            max_e = max_n = 0.0
            for ki, k in enumerate(D.COUNTIES):
                for ci, c in enumerate(D.CLASSES):
                    e, n = rec[(k, m, c)]
                    got_e = np.nansum(vk[ki, ci, :, :t_last + 1], axis=0)
                    got_n = z["n_pumps_by_county"][ki, ci, :, :t_last + 1].sum(0)
                    den_e = max(float(np.abs(e).max()), 1.0)
                    den_n = max(float(np.abs(n).max()), 1e-9)
                    max_e = max(max_e, float(np.abs(got_e - e[:t_last + 1]).max()) / den_e)
                    max_n = max(max_n, float(np.abs(got_n - n[:t_last + 1]).max()) / den_n)
            item(f"{pre}.e_f_independent_recompute", max_e <= 1e-9 and max_n <= 1e-9,
                 {"max_rel_energy": max_e, "max_rel_count": max_n})
            agg = np.nansum(vk, axis=0)
            ok_g = (np.allclose(np.nan_to_num(val), agg, rtol=1e-12, atol=1e-6) and
                    np.allclose(np.nan_to_num(z["value_total_all"]), np.nan_to_num(val).sum(0),
                                rtol=1e-12, atol=1e-6) and
                    np.allclose(np.nan_to_num(z["value_total_nodomestic"]),
                                np.nan_to_num(z["value_total_all"]) -
                                np.nan_to_num(val[D.CLASSES.index("domestic")]),
                                rtol=1e-12, atol=1e-6))
            item(f"{pre}.g_aggregates", ok_g, {})
            bad = [x for x in z.files if x.lower() in ("uid", "x", "y", "sid", "x_twd97",
                                                        "y_twd97", "meter", "meter_idx")]
            strs = [x for x in z.files if z[x].dtype.kind == "U" and
                    any(":" in s and s.split(":")[0] in D.COUNTIES and s.split(":")[1].isdigit()
                        for s in np.atleast_1d(z[x]).astype(str).ravel()[:1000])]
            item(f"{pre}.i_privacy", not bad and not strs, {"bad_keys": bad, "uid_like": strs})
        tab = pd.read_parquet(od / "county_class_monthly.parquet")
        cols_bad = [c for c in tab.columns if c in ("uid", "x_twd97", "y_twd97", "sid")]
        item(f"{variant}.i_privacy_table", not cols_bad, {"cols": cols_bad})
    if len(out_dirs) == 2:
        t_cut = D.tidx("2022-11")
        mx = 0.0
        for m in MEMBERS:
            a = vals[("preorigin", m)][..., :t_cut + 1]
            b = vals[("full", m)][..., :t_cut + 1]
            den = max(float(np.nanmax(np.abs(a))), 1.0)
            mx = max(mx, float(np.nanmax(np.abs(a - b))) / den)
        item("h_preorigin_equals_full_to_2022_11", mx <= 1e-9, {"max_rel": mx})
    return res


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit("use build (assembly and checks run from f9_finalize)")


if __name__ == "__main__":
    main()
