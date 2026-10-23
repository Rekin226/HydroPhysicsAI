"""WP C0 acceptance checks, as pre-registered in
results/dataprep/preregistration/C0.census_product.v1.json (A1 tool ``harness.prereg2``).

Each test calls ``prereg2.assert_can_score`` immediately before it computes its score (which
logs the scoring look) and writes results/dataprep/reports/C0_evidence/checks/<test>.json.
``summary`` evaluates the pre-registered gates and writes checks/summary.json.

Usage (inside tools/cpu_slot.sh):
    python -I checks.py T1a T1b T2a T2b T9 T5 T6 T7 T8 T3 T4
    python -I checks.py summary
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402
import lib  # noqa: E402
from common import flags as cflags  # noqa: E402
from common import guard  # noqa: E402
from harness import prereg2  # noqa: E402

OUT = lib.EVIDENCE / "checks"
CLS = list(lib.CLASSES)
C = len(CLS)
T0, T1 = lib.DUTY_T0, lib.DUTY_T1
W = slice(lib.tidx("2012-01"), lib.tidx("2022-12") + 1)
TID = {
    "T1a": "C0.v1.T1a_model_path_identity", "T1b": "C0.v1.T1b_c0_pipeline_on_amp",
    "T2a": "C0.v1.T2a_new_l0_common_rows", "T2b": "C0.v1.T2b_product_yunlin_vs_model",
    "T3": "C0.v1.T3_vs_p1_truth", "T4": "C0.v1.T4_tpc_coverage",
    "T5": "C0.v1.T5_preorigin_equals_full", "T6": "C0.v1.T6_masks_and_internal_consistency",
    "T7": "C0.v1.T7_county_coverage", "T8": "C0.v1.T8_privacy",
    "T9": "C0.v1.T9_aggregation_vs_model_function",
}
ABS_TOL, REL_TOL = 1e-6, 1e-9


# ----------------------------------------------------------------------------- helpers
def score(key: str) -> dict:
    r = prereg2.assert_can_score(lib.PREREG_ID, lib.WP, test_id=TID[key])
    return {"prereg_sha256": r["sha256"], "inputs_verified": r["inputs_verified"],
            "post_registration_looks": r["post_registration_looks"]}


def products() -> dict[str, Path]:
    return {k: lib.REPO / v for k, v in lib.load_json(lib.WORK / "products.json").items()}


def compare(a: np.ndarray, ref: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if a.shape != ref.shape:
        return {"shape_equal": False, "shape": list(a.shape), "ref_shape": list(ref.shape),
                "n_violations": int(max(a.size, ref.size)), "pass": False}
    nan_a, nan_r = np.isnan(a), np.isnan(ref)
    fin = ~nan_a & ~nan_r
    d = np.abs(a - ref)
    viol = fin & (d > ABS_TOL + REL_TOL * np.abs(ref))
    nan_mismatch = int((nan_a != nan_r).sum())
    return {"shape_equal": True, "n_elements": int(a.size), "nan_mismatch": nan_mismatch,
            "max_abs_delta": float(d[fin].max()) if fin.any() else 0.0,
            "sum_abs_delta": float(d[fin].sum()), "sum_ref": float(np.nansum(ref)),
            "n_violations": int(viol.sum()) + nan_mismatch,
            "pass": bool(viol.sum() == 0 and nan_mismatch == 0)}


def save(key: str, res: dict) -> None:
    res = {"test": key, "test_id": TID.get(key, key), "time_utc": prereg2._now(), **res}
    lib.save_json(res, OUT / f"{key}.json")
    print(key, "pass" if res.get("pass") else ("reported" if res.get("pass") is None
                                                 else "FAIL"), flush=True)


def model_inputs():
    import pickle
    if str(lib.REPO) not in sys.path:
        sys.path.insert(0, str(lib.REPO))
    with open(lib.MODEL_INPUTS_PKL, "rb") as f:
        return pickle.load(f)


def model_stack(eb: dict, A: int, T: int) -> np.ndarray:
    E = np.zeros((C, A, T))
    for i, c in enumerate(CLS):
        if c in eb:
            E[i] = np.asarray(eb[c], dtype=np.float64)
    return E


def model_E() -> tuple[np.ndarray, dict]:
    inp = model_inputs()
    A, T = inp.grid.n_active, len(inp.dates)
    return model_stack(inp.E_by_class, A, T), {"classes": sorted(inp.E_by_class),
                                                "n_months": T, "n_active": A}


def norm_sid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


# ----------------------------------------------------------------------------- T1, T2, T9
def t1a() -> dict:
    from hydrophysics.twin.pumping import clean_census
    from hydrophysics.twin.scenario import energy_by_class
    meta = score("T1a")
    inp = model_inputs()
    grid = lib.load_grid()
    want_dates = pd.date_range(T0, T1, freq="MS", inclusive="left")
    g = inp.grid
    grid_eq = bool(np.array_equal(np.asarray(g.mask, bool), grid.mask)
                   and (g.nx, g.ny, float(g.dx), float(g.x0), float(g.y0))
                   == (grid.nx, grid.ny, grid.dx, grid.x0, grid.y0))
    dates_eq = bool(len(inp.dates) == len(want_dates)
                    and (pd.DatetimeIndex(inp.dates) == want_dates).all())
    amp = pd.read_parquet(lib.AMP_CENSUS, columns=["sid", "PUMP_HP", "電號1", "PURPOSE",
                                                   "TWD97_X", "TWD97_Y"])
    kwh = pd.read_parquet(lib.AMP_KWH, columns=["datetime", "electricity_kwh", "pump"])
    pumps, kw, rep = clean_census(amp, kwh, cap_duty=1.0, t0=T0, t1=T1)
    eb, _ = energy_by_class(pumps, kw, inp.grid, T0, T1)
    keys_eq = sorted(eb) == sorted(inp.E_by_class)
    per = {c: compare(eb[c], inp.E_by_class[c]) for c in eb if c in inp.E_by_class}
    ok = grid_eq and dates_eq and keys_eq and all(v["pass"] for v in per.values())
    return {**meta, "grid_equal": grid_eq, "dates_equal": dates_eq, "class_keys_equal": keys_eq,
            "classes": sorted(eb), "per_class": per,
            "kept_kwh_2012_2022_GWh": float(rep["kwh_kept_GWh"].sum()), "pass": bool(ok)}


def run_amp_pipeline(grid):
    df = lib.harmonise("yunlin", lib.read_amp_census(), grid)
    k, L, info = lib.read_amp_kwh(df["sid"].to_numpy())
    k["pump"] = norm_sid(k["pump"])
    r = build.process("yunlin", df, k, L, "preorigin", grid)
    return r, info


def t1b() -> dict:
    meta = score("T1b")
    grid = lib.load_grid()
    E, emeta = model_E()
    r, info = run_amp_pipeline(grid)
    res = compare(r["cell"]["value"][:, :, W], E)
    return {**meta, "amp_kwh_info": info, "model": emeta,
            "pumps_kept": r["info"]["pumps_kept"], **res}


def t2a() -> dict:
    meta = score("T2a")
    grid = lib.load_grid()
    E, emeta = model_E()
    df = lib.harmonise("yunlin", lib.read_census_l0("yunlin"), grid)
    k, L = lib.read_kwh_l0("yunlin", df["sid"].to_numpy(), len(df))
    _, La, info = lib.read_amp_kwh(df["sid"].to_numpy())
    t_amp_end = lib.tidx(info["last_month"])
    tt = np.arange(lib.T_ALL)
    in_amp = tt[None, :] <= t_amp_end
    both = np.isfinite(L) & np.isfinite(La)
    differ = both & (La != L)
    absent = np.isfinite(L) & np.isnan(La) & in_amp
    absent_w = absent.copy()
    absent_w[:, :W.start] = False
    absent_w[:, W.stop:] = False
    mask = np.isnan(La) & (tt[None, :] <= lib.tidx("2022-12"))
    Lm = L.copy()
    Lm[mask] = np.nan
    pi = pd.Index(df["sid"].to_numpy(dtype=object)).get_indexer(k["pump"])
    ti = ((k["datetime"].dt.year - 2007) * 12 + k["datetime"].dt.month - 1).to_numpy()
    km = k[~mask[pi, ti]]
    r = build.process("yunlin", df, km, Lm, "preorigin", grid)
    res = compare(r["cell"]["value"][:, :, W], E)
    rows = np.unique(np.nonzero(absent)[0])
    return {**meta, "amp_kwh_info": info, "model": emeta,
            "pump_months_both_finite_differ": int(differ.sum()),
            "pump_months_absent_in_amp": int(absent.sum()),
            "pump_months_absent_in_amp_nonzero": int((absent & (L != 0)).sum()),
            "absent_kwh_all": float(np.nansum(L[absent])),
            "absent_kwh_2012_2022": float(np.nansum(L[absent_w])),
            "absent_pumps": df.loc[rows, "uid"].tolist(),
            "absent_pumps_first_on_meter": df.loc[rows, "meter_first_pump"].tolist(),
            "absent_pumps_meter_n_pumps": df.loc[rows, "meter_n_pumps"].tolist(),
            "pump_months_masked_2007_2022": int(mask.sum()), **res}


def t2b() -> dict:
    meta = score("T2b")
    prods = products()
    E, emeta = model_E()
    t2 = lib.load_json(OUT / "T2a.json")
    z = np.load(prods["preorigin"] / "cells_by_class.npz")
    counties = list(z["counties"])
    V = z["value_by_county"][counties.index("yunlin")][:, :, W]
    d = V - E
    month_tot = E.sum((0, 1))
    rel_month = np.abs(d.sum((0, 1))) / np.where(month_tot > 0, month_tot, np.nan)
    sum_abs = float(np.abs(d).sum())
    lim_b = 2 * t2["absent_kwh_2012_2022"] + 1.0
    ok_a = bool(np.nanmax(rel_month) <= 1e-4)
    ok_b = bool(sum_abs <= lim_b)
    worst = int(np.nanargmax(rel_month))
    return {**meta, "model": emeta,
            "a_max_rel_month_total_delta": float(np.nanmax(rel_month)),
            "a_worst_month": lib.ym(W.start + worst), "a_pass": ok_a,
            "b_sum_abs_delta_kwh": sum_abs, "b_limit_kwh": lim_b, "b_pass": ok_b,
            "max_element_abs_delta_kwh": float(np.abs(d).max()),
            "n_elements_nonzero_delta": int((np.abs(d) > ABS_TOL).sum()),
            "fan_total_model_2012_2022_GWh": float(E.sum() / 1e6),
            "fan_total_product_2012_2022_GWh": float(V.sum() / 1e6),
            "nan_in_product_window": int(np.isnan(V).sum()),
            "pass": ok_a and ok_b and int(np.isnan(V).sum()) == 0}


def t9() -> dict:
    from hydrophysics.twin.scenario import energy_by_class
    meta = score("T9")
    prods = products()
    grid = lib.load_grid()
    z = np.load(prods["preorigin"] / "cells_by_class.npz")
    counties = list(z["counties"])
    out = {}
    for tag in ("changhua", "chiayi"):
        df = lib.harmonise(tag, lib.read_census_l0(tag), grid)
        k, _ = lib.read_kwh_l0(tag, df["sid"].to_numpy(), len(df))
        pumps, kw, _ = lib.clean_census(lib.model_pumps(df), k, cap_duty=1.0, t0=T0, t1=T1)
        eb, dates = energy_by_class(pumps, kw, grid, T0, T1)
        E = model_stack(eb, grid.n_active, len(dates))
        V = z["value_by_county"][counties.index(tag)][:, :, W]
        out[tag] = {**compare(V, E), "classes": sorted(eb),
                    "fan_total_2012_2022_GWh": float(E.sum() / 1e6)}
    return {**meta, "per_county": out, "pass": all(v["pass"] for v in out.values())}


# ----------------------------------------------------------------------------- T5
def t5() -> dict:
    meta = score("T5")
    prods = products()
    pre, full = prods["preorigin"], prods["full"]
    out = {}
    for f in ("pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz"):
        a, b = np.load(pre / f), np.load(full / f)
        Tp = a["value"].shape[1]
        same_uid = bool(np.array_equal(a["uid"], b["uid"]))
        res = compare(b["value"][:, :Tp], a["value"]) if same_uid else {"pass": False}
        out[f] = {"uid_equal": same_uid, "n_pumps": int(len(a["uid"])), **res}
    for f, keys in (("cells_by_class.npz", ("value", "value_by_county")),
                    ("cells_total.npz", ("value_all", "value_nodomestic"))):
        a, b = np.load(pre / f), np.load(full / f)
        for key in keys:
            Tp = a[key].shape[-1]
            out[f"{f}:{key}"] = compare(b[key][..., :Tp], a[key])
    strict = {k: (v.get("max_abs_delta", 0.0) <= 1e-9 and v.get("nan_mismatch", 0) == 0
                  and v.get("uid_equal", True) and v.get("shape_equal", True))
              for k, v in out.items()}
    return {**meta, "items": out, "items_pass_1e-9": strict, "pass": all(strict.values())}


# ----------------------------------------------------------------------------- T6
def l1_table() -> pd.DataFrame:
    return pd.read_parquet(products()["census_L1"] / "census_pumps.parquet")


def meter_context(tag: str, l1: pd.DataFrame):
    """(L0 dense matrix rows of each meter's first pump keyed by meter_idx, meter install
    month index, HP sum per meter) for one county, independently of build.py."""
    sub = l1[l1["county"] == tag]
    raw_n = len(lib.read_census_l0(tag)) if tag == "chiayi" else len(sub)
    sids = None
    if tag == "chiayi":
        allc = lib.harmonise(tag, lib.read_census_l0(tag), lib.load_grid())
        sids = allc["sid"].to_numpy()
    else:
        sids = sub["sid"].to_numpy()
    _, L = lib.read_kwh_l0(tag, sids, raw_n)
    first = sub[sub["meter_first_pump"]]
    reading = pd.Series(list(range(len(first))), index=first["meter_idx"].to_numpy())
    Lm = L[first["row_idx"].to_numpy()]
    inst_t = []
    for m, v in zip(sub["install_month"], sub["install_valid"], strict=True):
        if v and m is not None:
            y, mo = int(m[:4]), int(m[5:7])
            inst_t.append((y - 2007) * 12 + mo - 1)
        else:
            inst_t.append(np.iinfo("int64").max)
    inst = pd.Series(inst_t).groupby(sub["meter_idx"].to_numpy()).min()
    inst = inst.where(inst != np.iinfo("int64").max, np.iinfo("int64").min // 2)
    hp = pd.to_numeric(sub["pump_hp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    hp_m = hp.groupby(sub["meter_idx"].to_numpy()).sum()
    return Lm, reading, inst, hp_m, first


def t6_variant(variant: str, d: Path, l1: pd.DataFrame, ctx: dict) -> dict:
    res = {}
    zc = np.load(d / "cells_by_class.npz")
    zt = np.load(d / "cells_total.npz")
    counties = list(zc["counties"])
    T = zc["value"].shape[-1]
    dates = list(zc["dates"])
    A = int(zc["n_active"])
    sums = np.zeros((len(counties), C, A, T))
    county_scope = {}
    phaseA_cells = np.zeros((len(counties), C, A), bool)
    for ki, tag in enumerate(counties):
        suffix = "_ingrid" if tag == "chiayi" else ""
        zp = np.load(d / f"pumps_{tag}{suffix}.npz")
        Lm, reading, inst, hp_m, first = ctx[tag]
        val, flg, meth = zp["value"], zp["flag"], zp["method"]
        midx = zp["meter_idx"]
        rows = reading.loc[midx].to_numpy()
        R = Lm[rows, :T]
        inst_p = inst.loc[midx].to_numpy()
        tt = np.arange(T)
        pre = tt[None, :] < inst_p[:, None]
        r = {}
        # (i)
        r["i_a_preinstall_nonzero"] = int(((meth == 1) & (val != 0)).sum())
        r["i_b_method12_not_before_install"] = int((np.isin(meth, [1, 2]) & ~pre).sum())
        r["i_c_method2_nonpositive_L0"] = int(((meth == 2) & ~(R > 0)).sum())
        r["i_c_preinstall_positive_not_method2"] = int((pre & (R > 0) & (meth != 2)).sum())
        r["i_d_flags_2_to_6"] = int(np.isin(flg, [2, 3, 4, 5, 6]).sum())
        # (iii)
        r["iii_nan_flag8_mismatch"] = int((np.isnan(val) != (flg == 8)).sum())
        r["iii_pump_flags_outside_0_8_9"] = int((~np.isin(flg, [0, 8, 9])).sum())
        # (iv) per pump = share x meter reading where method != 1
        share = zp["hp_share"]
        W_ = share[:, None] * R
        m1 = meth == 1
        expect_nan = (np.isnan(R) & (meth == 3)) | (meth == 4)
        r["iv_nan_pattern_mismatch"] = int(((np.isnan(val) != expect_nan) & ~m1).sum())
        fin = ~m1 & np.isfinite(val) & np.isfinite(W_)
        dd = np.abs(val - W_)
        r["iv_share_x_reading_violations"] = int((fin & (dd > ABS_TOL + REL_TOL * np.abs(W_)))
                                                 .sum())
        r["iv_share_x_reading_max_abs"] = float(dd[fin].max()) if fin.any() else 0.0
        # (ii) full: phase-A meters at the last month
        if variant == "full":
            mt = pd.read_parquet(d / f"meters_{tag}{suffix}.parquet")
            ph = mt.set_index("meter_idx")
            isA = ((ph["phase_code"] == "A") | ((ph["phase_code"] == "unknown")
                                                 & (ph["phase_detected"] == "A")))
            a_p = isA.reindex(midx).fillna(False).to_numpy(bool)
            j = T - 1
            target = a_p & ~pre[:, j]
            r["ii_phaseA_last_month_pump_months"] = int(target.sum())
            r["ii_phaseA_last_month_not_nan_flag8_m4"] = int(
                (target & ~(np.isnan(val[:, j]) & (flg[:, j] == 8) & (meth[:, j] == 4))).sum())
            r["ii_method4_outside_target"] = int(((meth[:, j] == 4) & ~target).sum()
                                                 + (meth[:, :j] == 4).sum())
            cell = zp["cell"]
            ci = np.array([CLS.index(c) for c in zp["purpose_class"]])
            sel = target & (cell >= 0)
            phaseA_cells[ki, ci[sel], cell[sel]] = True
        # sums of per-pump arrays into cells and county scope
        cell = zp["cell"]
        ci = np.array([CLS.index(c) for c in zp["purpose_class"]])
        v0 = np.nan_to_num(val, nan=0.0)
        g = cell >= 0
        flat = ci[g] * A + cell[g]
        S = build.group_sum(flat, C * A, v0[g]).reshape(C, A, T)
        sums[ki] = S
        if tag != "chiayi":
            county_scope[tag] = build.group_sum(ci, C, v0)
        res[tag] = r
    # (iv) cells == sums of pumps, county axis, totals
    agg = {}
    fbc = zc["flag_by_county"]
    vbc = zc["value_by_county"]
    for ki, tag in enumerate(counties):
        ok8 = fbc[ki] != 8
        agg[f"cells_vs_pumps:{tag}"] = compare(np.where(ok8, vbc[ki], 0.0),
                                               np.where(ok8, sums[ki], 0.0))
        agg[f"cells_nan_flag8_mismatch:{tag}"] = int((np.isnan(vbc[ki]) != ~ok8).sum())
    f, v = zc["flag"], zc["value"]
    ok8 = f != 8
    agg["total_vs_county_sum"] = compare(np.where(ok8, v, 0.0),
                                         np.where(ok8, np.nansum(vbc, 0), 0.0))
    fa, va = zt["flag_all"], zt["value_all"]
    agg["value_all_vs_class_sum"] = compare(np.where(fa != 8, va, 0.0),
                                            np.where(fa != 8, np.nansum(v, 0), 0.0))
    dom = CLS.index("domestic")
    nd = [i for i in range(C) if i != dom]
    fn, vn = zt["flag_nodomestic"], zt["value_nodomestic"]
    agg["nodomestic_vs_class_sum"] = compare(np.where(fn != 8, vn, 0.0),
                                             np.where(fn != 8, np.nansum(v[nd], 0), 0.0))
    # (iii) flags and validate_grid
    gr = {}
    for name, (vv, ss, ff) in {"cells_by_class": (zc["value"], zc["sd"], zc["flag"]),
                               "cells_total_all": (zt["value_all"], zt["sd_all"],
                                                   zt["flag_all"]),
                               "cells_total_nodomestic": (zt["value_nodomestic"],
                                                          zt["sd_nodomestic"],
                                                          zt["flag_nodomestic"])}.items():
        gr[name] = {"validate_grid": cflags.validate_grid(vv, ss, ff, dates),
                    "nan_flag8_mismatch": int((np.isnan(vv) != (ff == 8)).sum()),
                    "flags_outside_0_8_9": int((~np.isin(ff, [0, 8, 9])).sum())}
    gr["flag_by_county_outside_0_8_9"] = int((~np.isin(fbc, [0, 8, 9])).sum())
    if variant == "full":
        ze = np.load(d / "cells_by_class_evensplit.npz")
        gr["evensplit"] = {"validate_grid": cflags.validate_grid(ze["value"], ze["sd"],
                                                                 ze["flag"], dates),
                           "nan_flag8_mismatch": int((np.isnan(ze["value"])
                                                      != (ze["flag"] == 8)).sum()),
                           "flags_outside_0_1_8_9": int((~np.isin(ze["flag"], [0, 1, 8, 9]))
                                                        .sum())}
        # (ii) cells: flag 8 at the last month exactly where a phase-A pump-month is
        j = T - 1
        cnt = {}
        for ki, tag in enumerate(counties):
            cnt[f"county_cells_mismatch:{tag}"] = int(((fbc[ki][:, :, j] == 8)
                                                       != phaseA_cells[ki]).sum())
        anyk = phaseA_cells.any(0)
        cnt["class_cells_mismatch"] = int(((f[:, :, j] == 8) != anyk).sum())
        cnt["total_all_mismatch"] = int(((fa[:, j] == 8) != anyk.any(0)).sum())
        cnt["total_nodomestic_mismatch"] = int(((fn[:, j] == 8) != anyk[nd].any(0)).sum())
        cnt["earlier_months_flag8_cells"] = int((f[:, :, :j] == 8).sum())
        gr["ii_cells"] = cnt
    # long table vs arrays
    lt = pd.read_parquet(d / "county_class_monthly.parquet")
    lt_res = {}
    for ki, tag in enumerate(counties):
        q = lt[(lt["county"] == tag) & (lt["scope"] == "in_grid") & lt["class"].isin(CLS)]
        M = q.pivot(index="class", columns="date", values="value").reindex(CLS)
        ff = q.pivot(index="class", columns="date", values="flag").reindex(CLS).to_numpy()
        ref = np.nansum(vbc[ki], 1)
        lt_res[f"{tag}:in_grid"] = compare(np.where(ff != 8, M.to_numpy(), 0.0),
                                           np.where(ff != 8, ref, 0.0))
        if tag in county_scope:
            q = lt[(lt["county"] == tag) & (lt["scope"] == "county") & lt["class"].isin(CLS)]
            M = q.pivot(index="class", columns="date", values="value").reindex(CLS)
            ff = q.pivot(index="class", columns="date", values="flag").reindex(CLS).to_numpy()
            lt_res[f"{tag}:county"] = compare(np.where(ff != 8, M.to_numpy(), 0.0),
                                              np.where(ff != 8, county_scope[tag], 0.0))
    agg["long_table"] = lt_res
    return {"per_county": res, "aggregates": agg, "grids": gr}


def kept_recomputed(tag: str, l1: pd.DataFrame, ctx: dict) -> set:
    Lm, reading, inst, hp_m, first = ctx[tag]
    duty = lib.meter_duty(Lm, hp_m.loc[first["meter_idx"]].to_numpy())
    kept_m = set(first["meter_idx"].to_numpy()[~(duty > lib.CAP_DUTY)])
    sub = l1[l1["county"] == tag]
    return set(sub.loc[sub["meter_idx"].isin(kept_m), "uid"])


def t6() -> dict:
    meta = score("T6")
    prods = products()
    l1 = l1_table()
    ctx = {tag: meter_context(tag, l1) for tag in lib.COUNTIES}
    out = {v: t6_variant(v, prods[v], l1, ctx) for v in ("preorigin", "full")}
    # (v) kept set recomputed independently
    kept = {}
    for tag in lib.COUNTIES:
        rec = kept_recomputed(tag, l1, ctx)
        sub = l1[l1["county"] == tag]
        suffix = "_ingrid" if tag == "chiayi" else ""
        for v in ("preorigin", "full"):
            uids = set(np.load(prods[v] / f"pumps_{tag}{suffix}.npz")["uid"])
            want = rec if tag != "chiayi" else (rec & set(sub.loc[sub["in_grid"], "uid"]))
            kept[f"{v}:{tag}"] = {"n_product": len(uids), "n_recomputed": len(want),
                                  "symmetric_difference": len(uids ^ want)}
        l1k = set(sub.loc[sub["kept"], "uid"])
        kept[f"census_L1:{tag}"] = {"n_l1_kept": len(l1k), "n_recomputed": len(rec),
                                    "symmetric_difference": len(l1k ^ rec)}
    out["v_kept_set"] = kept

    def viol(x) -> int:
        n = 0
        if isinstance(x, dict):
            if "n_violations" in x:
                return int(x["n_violations"])
            for k, v in x.items():
                if k in ("validate_grid",):
                    n += len(v)
                elif k in ("n_product", "n_recomputed", "n_l1_kept", "n_pumps",
                           "ii_phaseA_last_month_pump_months", "iv_share_x_reading_max_abs"):
                    continue
                else:
                    n += viol(v)
        elif isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            n += int(x)
        return n
    parts = {"preorigin": viol(out["preorigin"]), "full": viol(out["full"]),
             "v_kept_set": viol(kept)}
    pre_ok = parts["preorigin"] == 0 and all(
        v["symmetric_difference"] == 0 for k, v in kept.items() if not k.startswith("full"))
    full_ok = parts["full"] == 0 and all(v["symmetric_difference"] == 0 for v in kept.values())
    l1_ok = all(v["symmetric_difference"] == 0 for k, v in kept.items()
                if k.startswith("census_L1"))
    return {**meta, **out, "violations": parts, "pass_preorigin_part": bool(pre_ok),
            "pass_full_part": bool(full_ok), "pass_l1_kept": bool(l1_ok),
            "pass": bool(pre_ok and full_ok)}


# ----------------------------------------------------------------------------- T7, T8
def t7() -> dict:
    meta = score("T7")
    prods = products()
    grid = lib.load_grid()
    l1 = l1_table()
    res = {}
    for tag in lib.COUNTIES:
        suffix = "_ingrid" if tag == "chiayi" else ""
        z = np.load(prods["preorigin"] / f"pumps_{tag}{suffix}.npz")
        res[tag] = {"census_pumps_in_active_cells": int(l1.loc[l1["county"] == tag,
                                                               "in_grid"].sum()),
                    "kept_pumps_in_active_cells": int((z["cell"] >= 0).sum())}
    pt = pd.read_parquet(lib.PARQ / "pingtung_census.parquet",
                         columns=["row_idx", "97X坐標", "97Y坐標"])
    pt = pt.rename(columns={"97X坐標": "TWD97_X", "97Y坐標": "TWD97_Y"})
    pt["sid"] = pt["row_idx"].astype(str)
    res["pingtung"] = {"census_pumps_in_active_cells": len(lib._cell_lookup(pt, grid))}
    ok = (res["yunlin"]["kept_pumps_in_active_cells"] > 0
          and res["changhua"]["kept_pumps_in_active_cells"] > 0
          and res["chiayi"]["census_pumps_in_active_cells"] == 320
          and res["pingtung"]["census_pumps_in_active_cells"] == 0)
    uid_unique = bool(l1["uid"].is_unique)
    return {**meta, "counts": res, "l1_uid_unique": uid_unique, "pass": bool(ok)}


FORBIDDEN_NAMES = set(lib.FORBIDDEN_COLS) | {"電號", "電號1", "電號2"}


def raw_meter_numbers() -> set:
    out = set()
    for tag, cols in (("yunlin", ["電號1"]), ("changhua", ["電號1"]), ("chiayi", ["電號", "電號2"]),
                      ("pingtung", ["電號"])):
        df = pd.read_parquet(lib.PARQ / f"{tag}_census.parquet", columns=cols)
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            out |= set(v.astype("int64").tolist())
    return out


def _json_keys(x, acc: set) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            acc.add(str(k))
            _json_keys(v, acc)
    elif isinstance(x, list):
        for v in x:
            _json_keys(v, acc)


def _hits(values, meters: set) -> float:
    s = pd.Series(values)
    v = pd.to_numeric(s, errors="coerce")
    v = v[np.isfinite(v) & (v == np.round(v)) & (v.abs() < 9e15)]
    if len(s) == 0:
        return 0.0
    return float(v.astype("int64").isin(meters).sum() / len(s))


def t8() -> dict:
    meta = score("T8")
    prods = products()
    meters = raw_meter_numbers()
    viol, checked = [], []
    for name, d in prods.items():
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = f"{name}/{p.relative_to(d)}"
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
                bad = sorted(set(df.columns) & FORBIDDEN_NAMES)
                if bad:
                    viol.append({"file": rel, "forbidden_columns": bad})
                for c in df.columns:
                    h = _hits(df[c].to_numpy(), meters)
                    if h >= 0.01:
                        viol.append({"file": rel, "column": c, "raw_meter_share": h})
                checked.append({"file": rel, "columns": len(df.columns)})
            elif p.suffix == ".npz":
                z = np.load(p)
                bad = sorted(set(z.files) & FORBIDDEN_NAMES)
                if bad:
                    viol.append({"file": rel, "forbidden_arrays": bad})
                for k in z.files:
                    a = z[k]
                    if a.ndim == 1:
                        h = _hits(a, meters)
                        if h >= 0.01:
                            viol.append({"file": rel, "array": k, "raw_meter_share": h})
                checked.append({"file": rel, "arrays": len(z.files),
                                "note": "1-D arrays scanned for raw meter numbers; 2-D/3-D "
                                        "kWh/flag arrays checked by name only"})
            elif p.suffix == ".json":
                keys = set()
                _json_keys(lib.load_json(p), keys)
                bad = sorted(keys & FORBIDDEN_NAMES)
                if bad:
                    viol.append({"file": rel, "forbidden_keys": bad})
                checked.append({"file": rel})
            else:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                bad = sorted(n for n in FORBIDDEN_NAMES if n in txt)
                if bad:
                    viol.append({"file": rel, "forbidden_strings": bad,
                                 "note": "text file; names may appear in documentation"})
                checked.append({"file": rel})
    return {**meta, "n_raw_meter_numbers": len(meters), "files_checked": checked,
            "violations": viol, "pass": len(viol) == 0}


# ----------------------------------------------------------------------------- T3, T4
def c0_long(variant: str) -> pd.DataFrame:
    return pd.read_parquet(products()[variant] / "county_class_monthly.parquet")


def t3() -> dict:
    meta = score("T3")
    p1 = guard.open_product(lib.P1_CENSUS_CHECK, "after_the_fact_analysis")
    truth = p1.read_table("census_truth_monthly.parquet")
    truth = truth[truth["series"] == "clean"].copy()
    truth["month"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m")
    out = {}
    for variant, last in (("preorigin", "2022-12"), ("full", "2026-06")):
        lt = c0_long(variant)
        lt = lt[lt["county"].isin(["yunlin", "changhua"]) & lt["class"].isin(CLS)
                & (lt["date"] <= last)]
        tv = truth[(truth["variant"] == variant) & (truth["month"] <= last)]
        m = lt.merge(tv, left_on=["county", "scope", "class", "date"],
                     right_on=["county", "scope", "class", "month"], how="outer",
                     suffixes=("_c0", "_p1"), indicator=True)
        both = m["_merge"] == "both"
        d = (m.loc[both, "value_c0"] / 1e6 - m.loc[both, "value_p1"]).abs()
        out[variant] = {"rows_c0": int(len(lt)), "rows_p1": int(len(tv)),
                        "rows_unmatched": int((~both).sum()),
                        "max_abs_delta_GWh": float(d.max()) if len(d) else None,
                        "n_over_1e-6_GWh": int((d > 1e-6).sum()),
                        "worst": (m.loc[both].assign(d=d).sort_values("d").tail(3)
                                  [["county", "scope", "class", "date_c0", "value_c0",
                                    "value_p1", "d"]].to_dict(orient="records"))}
    ok = all(v["rows_unmatched"] == 0 and v["n_over_1e-6_GWh"] == 0 for v in out.values())
    return {**meta, "p1_product": "county_electricity/census_check/v2",
            "p1_use": "after_the_fact_analysis", "variants": out, "pass": bool(ok),
            "gating": False}


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 12:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def t4() -> dict:
    meta = score("T4")
    tpc = guard.open_product(lib.P1_TPC_OBSERVED, "after_the_fact_analysis")
    tp = tpc.read_table("tpc_county_sector_monthly_observed.parquet")
    tp = tp[tp["sector"] == "農林漁牧"]
    zh = {"changhua": "彰化縣", "yunlin": "雲林縣"}
    lt = c0_long("full")
    months = sorted(lt["date"].unique())
    res = {"tpc_rows_used": int(len(tp)), "census_variant": "full"}
    ser = {}
    for tag, cz in zh.items():
        t = tp[tp["county"] == cz].set_index("month")["value"].sort_index()   # GWh
        ser[f"tpc:{tag}"] = t
        for scope in ("county", "in_grid"):
            for cl in ("all", "all_nodomestic"):
                q = lt[(lt["county"] == tag) & (lt["scope"] == scope) & (lt["class"] == cl)]
                ser[f"census:{tag}:{scope}:{cl}"] = q.set_index("date")["value"] / 1e6
    # annual coverage, calendar years
    annual = {}
    for tag in zh:
        t = ser[f"tpc:{tag}"]
        for key in [k for k in ser if k.startswith(f"census:{tag}:")]:
            c = ser[key]
            rows = {}
            for y in range(2016, 2027):
                ms = [m for m in t.index if m.startswith(str(y)) and m <= "2026-06"]
                ms = [m for m in ms if m in c.index and np.isfinite(c[m])]
                if not ms:
                    continue
                rows[str(y) if y < 2026 else "2026-01..06"] = {
                    "census_GWh": float(c[ms].sum()), "tpc_GWh": float(t[ms].sum()),
                    "coverage": float(c[ms].sum() / t[ms].sum()), "n_months": len(ms)}
            annual[key.replace("census:", "")] = rows
    res["annual_coverage"] = annual
    # month-of-year coverage profile 2016-2022 (ratio of sums)
    moy = {}
    for tag in zh:
        t, c = ser[f"tpc:{tag}"], ser[f"census:{tag}:county:all"]
        prof = {}
        for mo in range(1, 13):
            ms = [m for m in t.index if "2016-01" <= m <= "2022-12" and int(m[5:7]) == mo]
            prof[f"{mo:02d}"] = float(c[ms].sum() / t[ms].sum())
        moy[tag] = prof
    res["month_of_year_coverage_2016_2022_county_all"] = moy
    # Changhua / Yunlin ratios
    ratio = {}
    for y in range(2016, 2026):
        ms = [m for m in months if m.startswith(str(y))]
        cc = ser["census:changhua:county:all"][ms].sum() / ser["census:yunlin:county:all"][ms]\
            .sum()
        cin = ser["census:changhua:in_grid:all"][ms].sum() / ser["census:yunlin:in_grid:all"][
            ms].sum()
        cnd = ser["census:changhua:county:all_nodomestic"][ms].sum() / ser[
            "census:yunlin:county:all_nodomestic"][ms].sum()
        tt_ = ser["tpc:changhua"].reindex(ms).sum() / ser["tpc:yunlin"].reindex(ms).sum()
        ratio[str(y)] = {"census_county_all": float(cc), "census_in_grid_all": float(cin),
                         "census_county_nodomestic": float(cnd), "tpc_agri": float(tt_)}
    res["changhua_over_yunlin"] = ratio
    # monthly correlations
    cor = {}
    for tag in zh:
        t = ser[f"tpc:{tag}"]
        c = ser[f"census:{tag}:county:all"]
        for w0, w1 in (("2016-01", "2022-12"), ("2016-01", "2026-06")):
            ms = [m for m in months if w0 <= m <= w1]
            x = c.reindex(ms).to_numpy(dtype=float)
            moy_ = pd.Series(x, index=[m[5:7] for m in ms])
            xa = x - moy_.groupby(level=0).transform("mean").to_numpy()
            yall = t.reindex(months).to_numpy(dtype=float)
            out = {}
            for lag in (-1, 0, 1):
                idx = [months.index(m) + lag for m in ms]
                y = np.array([yall[i] if 0 <= i < len(months) else np.nan for i in idx])
                ys = pd.Series(y, index=[m[5:7] for m in ms])
                ya = y - ys.groupby(level=0).transform("mean").to_numpy()
                out[f"lag{lag:+d}"] = {"r_raw": _corr(x, y), "r_deseasonalised": _corr(xa, ya)}
            cor[f"{tag}:{w0}..{w1}"] = out
    res["monthly_correlation_census_county_all_vs_tpc"] = cor
    # reproduction of P1 C8 cov_clean_all (annual 2016-2025)
    p1 = guard.open_product(lib.P1_CENSUS_CHECK, "after_the_fact_analysis")
    ev = lib.load_json(p1.path("evaluation.json"))
    c8 = ev["C8_coverage_validator_role"]
    rep = {}
    worst = 0.0
    for tag in zh:
        for y in range(2016, 2026):
            ref = c8.get(tag, {}).get(str(y), {}).get("cov_clean_all")
            got = annual[f"{tag}:county:all"].get(str(y), {}).get("coverage")
            if ref is None or got is None:
                continue
            rel = abs(got / ref - 1)
            worst = max(worst, rel)
            rep[f"{tag}:{y}"] = {"c0": got, "p1_c8": ref, "rel_diff": rel}
    res["p1_c8_reproduction"] = {"items": rep, "max_rel_diff": worst,
                                 "pass_1e-6": bool(worst <= 1e-6)}
    return {**meta, **res, "pass": None, "gating": False}


# ----------------------------------------------------------------------------- summary
def summary() -> dict:
    r = {k: lib.load_json(OUT / f"{k}.json") for k in TID if (OUT / f"{k}.json").exists()}
    p = {k: v.get("pass") for k, v in r.items()}
    t6r = r.get("T6", {})
    pre_tests = {"T1a": p.get("T1a"), "T1b": p.get("T1b"), "T2a": p.get("T2a"),
                 "T2b": p.get("T2b"), "T5": p.get("T5"),
                 "T6_preorigin_part": t6r.get("pass_preorigin_part"), "T7": p.get("T7"),
                 "T8": p.get("T8"), "T9": p.get("T9")}
    full_tests = {**pre_tests, "T6_full_part": t6r.get("pass_full_part")}
    l1_tests = {"T7": p.get("T7"), "T8": p.get("T8"),
                "uid_unique": r.get("T7", {}).get("l1_uid_unique"),
                "T6_v_l1_kept": t6r.get("pass_l1_kept")}
    gates = {
        "pump_electricity/preorigin": {"tests": pre_tests,
                                       "passed": all(v is True for v in pre_tests.values())},
        "pump_electricity/full": {"tests": full_tests,
                                  "passed": all(v is True for v in full_tests.values())},
        "pump_electricity/census_L1": {"tests": l1_tests,
                                       "passed": all(v is True for v in l1_tests.values())},
    }
    out = {"prereg": lib.PREREG_ID, "tests": p, "gates": gates,
           "reported_only": {"T3": p.get("T3"),
                             "T4_p1_c8_reproduction": r.get("T4", {}).get(
                                 "p1_c8_reproduction", {}).get("pass_1e-6")}}
    lib.save_json(out, OUT / "summary.json")
    print(out["gates"])
    return out


FUNCS = {"T1a": t1a, "T1b": t1b, "T2a": t2a, "T2b": t2b, "T3": t3, "T4": t4, "T5": t5,
         "T6": t6, "T7": t7, "T8": t8, "T9": t9}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for key in argv:
        if key == "summary":
            summary()
            continue
        t0 = time.time()
        res = FUNCS[key]()
        res["seconds"] = round(time.time() - t0, 1)
        save(key, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
