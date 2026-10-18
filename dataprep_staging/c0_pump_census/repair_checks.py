"""WP C0 repair round: the checks pre-registered in
results/dataprep/preregistration/C0.census_product.v2.json (A1 tool ``harness.prereg3``).

Each test calls ``prereg3.assert_can_score`` immediately before it computes its score (which
logs the scoring look) and writes results/dataprep/reports/C0_evidence/repair/checks/<test>.json
with the sha256 of this file at scoring time (a later edit of this file archives the previous
copy under checks/code_archive/ first; see ``main``).

Usage (inside tools/cpu_slot.sh):
    python -I repair_checks.py R1 R4 T5 R5 R6 R2 R3 R7 R8 R9 R10
    python -I repair_checks.py summary
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402  (v1, read-only)
import checks  # noqa: E402  (v1 helpers, read-only: compare, meter_context, privacy scan)
import lib  # noqa: E402
import repair_assemble as ra  # noqa: E402
import repair_build as rb  # noqa: E402
import repair_lib as rl  # noqa: E402
from common import flags as cflags  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "C0.census_product.v2"
OUT = lib.EVIDENCE / "repair" / "checks"
SCRATCH = Path(os.environ.get("C0R_SCRATCH", str(lib.EVIDENCE / "repair" / "_scratch")))
CLS = list(lib.CLASSES)
C = len(CLS)
DOM = CLS.index("domestic")
ND = [i for i in range(C) if i != DOM]
J = rl.J_DEC22
T_PRE = rl.T_PRE
W_FULL = slice(lib.tidx("2012-01"), lib.tidx("2022-12") + 1)
W_NOV = slice(lib.tidx("2012-01"), lib.tidx("2022-11") + 1)
TID = {
    "R1": "C0.v2.R1_v2_vs_v1_decomposition",
    "R2": "C0.v2.R2_no_postorigin_dependency",
    "R3": "C0.v2.R3_yunlin_vs_model_forcing",
    "R4": "C0.v2.R4_masks_and_internal_consistency",
    "R5": "C0.v2.R5_privacy",
    "R6": "C0.v2.R6_county_coverage",
    "R7": "C0.v2.R7_fill_backtest_loyo",
    "R8": "C0.v2.R8_after_the_fact_dec2022",
    "R9": "C0.v2.R9_census_snapshot_sensitivity",
    "R10": "C0.v2.R10_descriptive",
    "T5": "C0.v2.T5_variant_consistency",
}
ABS, REL = 1e-6, 1e-9
compare = checks.compare


# ----------------------------------------------------------------------------- helpers
def score(key: str) -> dict:
    r = prereg3.assert_can_score(PREREG_ID, lib.WP, test_id=TID[key])
    return {"prereg_sha256": r["sha256"], "inputs_verified": r["inputs_verified"],
            "post_registration_looks": r["post_registration_looks"],
            "checks_code_sha256": sha256_file(Path(__file__))}


def save(key: str, res: dict) -> None:
    res = {"test": key, "test_id": TID[key], "time_utc": prereg3._now(), **res}
    lib.save_json(res, OUT / f"{key}.json")
    p = res.get("pass")
    print(key, "pass" if p else ("reported" if p is None else "FAIL"), flush=True)


def v2() -> dict[str, Path]:
    return {k: lib.REPO / v for k, v in
            lib.load_json(lib.EVIDENCE / "repair" / "products_v2.json").items()}


def v1() -> dict[str, Path]:
    return {k: lib.PRODUCT_ROOT / k / "v1" for k in ("census_L1", "preorigin", "full")}


def l1_v2() -> pd.DataFrame:
    return pd.read_parquet(v2()["census_L1"] / "census_pumps.parquet")


def pumps_file(tag: str) -> str:
    return f"pumps_{tag}{'_ingrid' if tag == 'chiayi' else ''}.npz"


def meters_file(tag: str) -> str:
    return f"meters_{tag}{'_ingrid' if tag == 'chiayi' else ''}.parquet"


def eq_nan(a, b, tol=1e-9) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_equal": False, "n_violations": int(max(a.size, b.size))}
    na, nb = np.isnan(a), np.isnan(b)
    fin = ~na & ~nb
    d = np.abs(a - b)
    return {"shape_equal": True, "nan_mismatch": int((na != nb).sum()),
            "max_abs": float(d[fin].max()) if fin.any() else 0.0,
            "n_violations": int((na != nb).sum() + (fin & (d > tol)).sum())}


def ctx_independent(tag: str, l1: pd.DataFrame) -> dict:
    """Pre-origin meter context recomputed from L0 and the L1 meter structure with lib
    functions only (not repair_build): reading rows, install, HP, class, code, detected
    phase on pairs inside 2012-01..2022-11, Dec-Jan-paired rule, duty without the paired
    meters' Dec-2022 value, kept meters.

    Chiayi: the L1 table holds only the in-grid pumps and their meter-mates, but the fill is
    fitted on the county's paired kept meters (as registered), so the whole-county meter
    structure is rebuilt from the census with ``lib.harmonise`` (first run of R1 used the L1
    subset for Chiayi: a scope defect of this check, fixed before the other tests were
    scored; the first result is kept under checks/superseded/)."""
    if tag == "chiayi":
        cols = ["county", "sid", "uid", "row_idx", "meter_first_pump", "meter_idx",
                "install_month", "install_valid", "pump_hp", "billing_code", "purpose_class",
                "in_grid"]
        l1 = lib.harmonise(tag, lib.read_census_l0(tag), lib.load_grid())[cols]
    Lm, reading, inst, hp_m, first = checks.meter_context(tag, l1)
    midx = first["meter_idx"].to_numpy()
    X = Lm[:, :T_PRE].copy()
    code = lib.phase_from_code(first["billing_code"].to_numpy())
    det = lib.detect_phase(X, lib.tidx("2012-01"), J)["phase_detected"].to_numpy()
    paired = (det == "B") | ((code == "B") & np.isin(det, ["mixed", "insufficient",
                                                          "both-flat"]))
    X[paired, J] = np.nan
    hp = hp_m.loc[midx].to_numpy()
    duty = lib.meter_duty(X, hp)
    kept = ~(duty > lib.CAP_DUTY)
    return {"Lm": Lm, "X": X, "midx": midx, "inst": inst.loc[midx].to_numpy(),
            "cls": first["purpose_class"].to_numpy(), "code": code, "det": det,
            "paired": paired, "duty": duty, "kept": kept, "first": first}


def ratio_fit(X, inst, cls, sel, years) -> dict:
    """Second implementation of the registered ratio fit (written apart from repair_lib)."""
    per = {c: [] for c in CLS}
    allr, num, den, rows = [], 0.0, 0.0, {}
    for y in years:
        to = lib.tidx(f"{y}-10")
        a = 0.5 * (X[:, to + 2] + X[:, to + 3])
        b = 0.5 * (X[:, to] + X[:, to + 1])
        ok = sel & ~np.isnan(a) & ~np.isnan(b) & (inst <= to)
        allr.append(a[ok].sum() / b[ok].sum())
        rows[y] = (a, b, ok)
        for c in CLS:
            m = ok & (cls == c)
            per[c].append((int(m.sum()), a[m].sum(), b[m].sum()))
    out = {}
    rho_y = {}
    for c in CLS:
        own = all(n >= rl.N_MIN_CLASS and B > 0 for n, _, B in per[c])
        r = np.array([A / B for _, A, B in per[c]]) if own else np.array(allr)
        rho_y[c] = r
        out[c] = {"rho_bar": float(r.mean()), "s": float(r.std(ddof=1)), "pooled": not own}
    for i, y in enumerate(years):
        a, b, ok = rows[y]
        for c in CLS:
            m = ok & (cls == c)
            e = a[m] - rho_y[c][i] * b[m]
            num += float((e ** 2).sum())
            den += float((b[m] ** 2).sum())
    return {"classes": out, "sigma": float(np.sqrt(num / den))}


def pump_meter_rows(tag: str, z, ctx: dict) -> np.ndarray:
    """Row (in ctx arrays) of each pump's meter."""
    pos = pd.Series(np.arange(len(ctx["midx"])), index=ctx["midx"])
    return pos.loc[z["meter_idx"]].to_numpy()


# ----------------------------------------------------------------------------- R1
def r1() -> dict:
    meta = score("R1")
    P2, P1 = v2(), v1()
    l1 = l1_v2()
    l1old = pd.read_parquet(P1["census_L1"] / "census_pumps.parquet")
    res, ok_pre, ok_full = {}, True, True
    for tag in lib.COUNTIES:
        sub = l1[l1["county"] == tag]
        ctx = ctx_independent(tag, l1)
        kept_m = set(ctx["midx"][ctx["kept"]])
        so = l1old[l1old["county"] == tag]
        kept_v1 = set(so.loc[so["kept"], "meter_idx"])
        kept_ms = kept_m & set(sub["meter_idx"])          # meters of the L1 table
        flips_in, flips_out = kept_ms - kept_v1, kept_v1 - kept_ms
        want = sub["meter_idx"].isin(kept_m)
        if tag == "chiayi":
            want &= sub["in_grid"]
        want_uid = set(sub.loc[want, "uid"])
        params = ratio_fit(ctx["X"], ctx["inst"], ctx["cls"], ctx["paired"] & ctx["kept"],
                           rl.FIT_YEARS)
        fp = lib.load_json(P2["preorigin"] / "fill_parameters.json")["counties"][tag]
        par_d = max([abs(params["classes"][c]["rho_bar"] - fp["classes"][c]["rho_bar"])
                     for c in CLS] + [abs(params["classes"][c]["s"] - fp["classes"][c]["s"])
                                      for c in CLS] + [abs(params["sigma"] - fp["sigma"])])
        lp = sub.groupby("meter_idx")["dec_jan_paired"].first()
        ld = sub.groupby("meter_idx")["meter_duty_2012_2022"].first()
        lpos = pd.Series(np.arange(len(ctx["midx"])), index=ctx["midx"]).loc[lp.index].to_numpy()
        r = {"flips_in_meters": len(flips_in), "flips_out_meters": len(flips_out),
             "fill_params_max_abs_diff_vs_product": par_d,
             "l1_paired_mismatch": int((lp.to_numpy() != ctx["paired"][lpos]).sum()),
             "l1_duty": eq_nan(ld.to_numpy(), ctx["duty"][lpos], tol=1e-12)}
        for variant in ("preorigin", "full"):
            z2 = np.load(P2[variant] / pumps_file(tag))
            z1 = np.load(P1[variant] / pumps_file(tag))
            u2, u1 = set(z2["uid"]), set(z1["uid"])
            rr = {"uid_set_vs_expected_symdiff": len(u2 ^ want_uid),
                  "uids_only_v2": len(u2 - u1), "uids_only_v1": len(u1 - u2)}
            common = sorted(u2 & u1)
            i2 = pd.Index(z2["uid"]).get_indexer(common)
            i1 = pd.Index(z1["uid"]).get_indexer(common)
            a2, a1 = z2["value"][i2], z1["value"][i1]
            f2, f1 = z2["flag"][i2], z1["flag"][i1]
            excl = np.zeros(a2.shape, bool)
            if variant == "preorigin":
                rows = pump_meter_rows(tag, {"meter_idx": z2["meter_idx"][i2]}, ctx)
                excl[:, J] = ctx["paired"][rows]
            d = np.abs(a2 - a1)
            na, nb = np.isnan(a2), np.isnan(a1)
            rr["outside_target_value_violations"] = int(((~excl) & ((na != nb) | (
                ~na & ~nb & (d > 1e-9)))).sum())
            rr["outside_target_flag_mismatch"] = int(((~excl) & (f2 != f1)).sum())
            rr["method_label_differences"] = int((z2["method"][i2] != z1["method"][i1]).sum())
            if variant == "preorigin":
                rows_all = pump_meter_rows(tag, z2, ctx)
                X = ctx["X"]
                base = 0.5 * (X[rows_all, J - 2] + X[rows_all, J - 1])
                paired_p = ctx["paired"][rows_all]
                inst_p = ctx["inst"][rows_all]
                has_base = paired_p & np.isfinite(base) & (inst_p <= J - 2)
                cls_p = ctx["cls"][rows_all]
                rho = np.array([params["classes"][c]["rho_bar"] for c in cls_p])
                s = np.array([params["classes"][c]["s"] for c in cls_p])
                share = z2["hp_share"]
                want_val = share * rho * base
                want_sd = share * base * np.sqrt(s ** 2 + params["sigma"] ** 2)
                fl4 = z2["flag"][:, J] == 4
                rr["flag4_vs_rule_mismatch"] = int((fl4 != has_base).sum())
                rr["flag4_outside_dec2022"] = int((np.delete(z2["flag"], J, axis=1) == 4).sum())
                vv, ss = z2["value"][:, J], z2["sd"][:, J]
                m = fl4 & has_base
                rr["fill_value_violations"] = int((m & ~(np.abs(vv - want_val) <= ABS + REL *
                                                         np.abs(want_val))).sum())
                rr["fill_sd_violations"] = int((m & ~(np.abs(ss - want_sd) <= ABS + 1e-9 *
                                                      np.abs(want_sd))).sum())
                nobase = paired_p & ~has_base
                rr["paired_nobase_not_flag8_or_preinstall_zero"] = int((nobase & ~(
                    (z2["flag"][:, J] == 8) | ((z2["method"][:, J] == 1) & (vv == 0)))).sum())
                rr["n_filled_pumps"] = int(fl4.sum())
                rr["n_paired_pumps"] = int(paired_p.sum())
                good = (rr["uid_set_vs_expected_symdiff"] == 0
                        and rr["outside_target_value_violations"] == 0
                        and rr["outside_target_flag_mismatch"] == 0
                        and rr["flag4_vs_rule_mismatch"] == 0 and rr["flag4_outside_dec2022"] == 0
                        and rr["fill_value_violations"] == 0 and rr["fill_sd_violations"] == 0
                        and rr["paired_nobase_not_flag8_or_preinstall_zero"] == 0
                        and par_d <= 1e-9 and r["l1_paired_mismatch"] == 0
                        and r["l1_duty"]["n_violations"] == 0)
                ok_pre &= good
            else:
                rr["flag4_anywhere"] = int((z2["flag"] == 4).sum())
                good = (rr["uid_set_vs_expected_symdiff"] == 0
                        and rr["outside_target_value_violations"] == 0
                        and rr["outside_target_flag_mismatch"] == 0 and rr["flag4_anywhere"] == 0)
                ok_full &= good
            rr["pass"] = bool(good)
            r[variant] = rr
        res[tag] = r
    return {**meta, "per_county": res, "pass_preorigin_part": bool(ok_pre),
            "pass_full_part": bool(ok_full), "pass": bool(ok_pre and ok_full)}


# ----------------------------------------------------------------------------- R4
def fill_coeffs(tag: str, z, fp: dict):
    """(mask of flag-4 pumps at 2022-12, cc, ic, meter key) recomputed from the per-pump
    arrays and fill_parameters.json (base share = value / rho_bar)."""
    m = z["flag"][:, J] == 4 if z["flag"].shape[1] > J else np.zeros(len(z["uid"]), bool)
    cls = z["purpose_class"]
    rho = np.array([fp["classes"][c]["rho_bar"] for c in cls])
    s = np.array([fp["classes"][c]["s"] for c in cls])
    bshare = np.where(m, z["value"][:, J] / np.where(rho > 0, rho, np.nan), 0.0) \
        if m.any() else np.zeros(len(cls))
    return m, s * bshare, fp["sigma"] * bshare


def r4_variant(variant: str, d: Path, l1: pd.DataFrame, ctxs: dict) -> dict:
    zc = np.load(d / "cells_by_class.npz")
    zt = np.load(d / "cells_total.npz")
    counties = list(zc["counties"])
    T = zc["value"].shape[-1]
    A = int(zc["n_active"])
    dates = list(zc["dates"])
    pre_v = variant == "preorigin"
    fp = lib.load_json(d / "fill_parameters.json")["counties"] if pre_v else None
    out, viol = {}, 0
    sums = np.zeros((len(counties), C, A, T))
    nfill = np.zeros((len(counties), C, A, T))
    mexp = np.zeros((len(counties), C, A, T))
    sd_frames = []
    for ki, tag in enumerate(counties):
        z = np.load(d / pumps_file(tag))
        ctx = ctxs[tag]
        rows = pump_meter_rows(tag, z, ctx)
        R = ctx["Lm"][rows, :T]
        val, flg, meth = z["value"], z["flag"], z["method"]
        tt = np.arange(T)
        pre = tt[None, :] < ctx["inst"][rows][:, None]
        r = {}
        r["i_preinstall_nonzero"] = int(((meth == 1) & (val != 0)).sum())
        r["i_method12_not_before_install"] = int((np.isin(meth, [1, 2]) & ~pre).sum())
        r["ii_nan_flag8_mismatch"] = int((np.isnan(val) != (flg == 8)).sum())
        allowed = [0, 4, 8] if pre_v else [0, 8, 9]
        r["ii_pump_flags_outside_allowed"] = int((~np.isin(flg, allowed)).sum())
        r["ii_flag_method_mismatch"] = int((rl.FLAG_OF_METHOD[meth] != flg).sum())
        if pre_v:
            sd = z["sd"]
            r["iii_sd_finite_vs_flag4_mismatch"] = int((np.isfinite(sd) != (flg == 4)).sum())
            r["iii_flag4_outside_dec2022"] = int((np.delete(flg, J, axis=1) == 4).sum())
            r["iii_flag4_not_paired"] = int(((flg[:, J] == 4) & ~ctx["paired"][rows]).sum())
            r["iii_negative_sd"] = int((np.nan_to_num(sd, nan=0.0) < 0).sum())
        else:
            r["iii_flag4_present"] = int((flg == 4).sum())
        # (iv) share x L0 meter reading where the value is an observation
        excl = np.isin(meth, [1, rl.FILL_METHOD, 4])
        if pre_v:
            excl[:, J] |= ctx["paired"][rows]
        W_ = z["hp_share"][:, None] * R
        nan_exp = np.isnan(W_) & ~excl
        r["iv_nan_pattern_mismatch"] = int(((np.isnan(val) != nan_exp) & ~excl).sum())
        fin = ~excl & np.isfinite(val) & np.isfinite(W_)
        r["iv_share_x_reading_violations"] = int((fin & (np.abs(val - W_) > ABS + REL *
                                                         np.abs(W_))).sum())
        if pre_v:
            pj = ctx["paired"][rows]
            r["iv_paired_dec2022_observed_value_leak"] = int(
                (pj & (meth[:, J] == 0)).sum())
        cell = z["cell"]
        ci = np.array([CLS.index(c) for c in z["purpose_class"]])
        g = cell >= 0
        flat = ci[g] * A + cell[g]
        sums[ki] = build.group_sum(flat, C * A, np.nan_to_num(val, nan=0.0)[g]).reshape(C, A, T)
        nfill[ki] = build.group_sum(flat, C * A, (meth == rl.FILL_METHOD)[g].astype(float)
                                    ).reshape(C, A, T)
        # missing_expected: HP x same-month class rate over reporting pumps (county-wide for
        # Yunlin/Changhua; the product's Chiayi rate is county-wide and is not recomputable
        # from in-grid pumps, so Chiayi is checked for sign and support only)
        hp = z["hp"]
        rep_ok = (flg == 0) & (meth != 1) & np.isfinite(val) & (hp[:, None] > 0)
        num = build.group_sum(ci, C, np.where(rep_ok, val, 0.0))
        den = build.group_sum(ci, C, np.where(rep_ok, hp[:, None], 0.0))
        rate = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        me = hp[:, None] * rate[ci] * (meth == 3)
        mexp[ki] = build.group_sum(flat, C * A, me[g]).reshape(C, A, T)
        if pre_v:
            m4, cc, ic = fill_coeffs(tag, z, fp[tag])
            sd_frames.append(pd.DataFrame({"county": ki, "cell": cell[m4], "cls": ci[m4],
                                           "meter": ki * 10**7 + z["meter_idx"][m4],
                                           "cc": cc[m4], "ic": ic[m4]}))
        viol += sum(v for v in r.values())
        out[tag] = r
    agg = {}
    fbc, vbc = zc["flag_by_county"], zc["value_by_county"]
    for ki, tag in enumerate(counties):
        ok8 = fbc[ki] != 8
        agg[f"cells_vs_pumps:{tag}"] = compare(np.where(ok8, vbc[ki], 0.0),
                                               np.where(ok8, sums[ki], 0.0))
        agg[f"cells_nan_flag8_mismatch:{tag}"] = {"n_violations": int(
            (np.isnan(vbc[ki]) != ~ok8).sum())}
        f4 = fbc[ki] == 4
        agg[f"flag4_vs_nfill:{tag}"] = {"n_violations": int(
            (f4 & (nfill[ki] == 0)).sum() + ((nfill[ki] > 0) & (fbc[ki] == 0)).sum())}
    f, v = zc["flag"], zc["value"]
    ok8 = f != 8
    agg["total_vs_county_sum"] = compare(np.where(ok8, v, 0.0),
                                         np.where(ok8, np.nansum(vbc, 0), 0.0))
    agg["nfill_total_vs_pumps"] = compare(zc["n_fill"], nfill.sum(0))
    me_ref = zc["missing_expected_kwh"]
    yc = [counties.index(t) for t in ("yunlin", "changhua")]
    # Chiayi's in-grid pumps can share cells with the other counties; compare where no
    # Chiayi pump-month is missing
    zcy = np.load(d / pumps_file("chiayi"))
    cy_cell = zcy["cell"]
    cy_has = np.zeros((C, A, T), bool)
    ci_cy = np.array([CLS.index(c) for c in zcy["purpose_class"]])
    for p in np.where(cy_cell >= 0)[0]:
        cy_has[ci_cy[p], cy_cell[p]] |= zcy["method"][p] == 3
    agg["missing_expected_vs_recomputed_yl_ch"] = compare(
        np.where(cy_has, 0.0, me_ref), np.where(cy_has, 0.0, mexp[yc].sum(0)))
    agg["missing_expected_negative"] = {"n_violations": int((me_ref < 0).sum())}
    agg["missing_expected_without_missing_hp"] = {"n_violations": int(
        ((me_ref > 0) & (zc["hp_missing"] == 0)).sum())}
    fa, va = zt["flag_all"], zt["value_all"]
    agg["value_all_vs_class_sum"] = compare(np.where(fa != 8, va, 0.0),
                                            np.where(fa != 8, np.nansum(v, 0), 0.0))
    fn, vn = zt["flag_nodomestic"], zt["value_nodomestic"]
    agg["nodomestic_vs_class_sum"] = compare(np.where(fn != 8, vn, 0.0),
                                             np.where(fn != 8, np.nansum(v[ND], 0), 0.0))
    gr = {}
    grids = {"cells_by_class": (zc["value"], zc["sd"], zc["flag"]),
             "cells_total_all": (zt["value_all"], zt["sd_all"], zt["flag_all"]),
             "cells_total_nodomestic": (zt["value_nodomestic"], zt["sd_nodomestic"],
                                        zt["flag_nodomestic"])}
    allowed_agg = [0, 4, 9] if pre_v else [0, 8, 9]
    for name, (vv, ss, ff) in grids.items():
        gr[name] = {"validate_grid": cflags.validate_grid(vv, ss, ff, dates),
                    "nan_flag8_mismatch": int((np.isnan(vv) != (ff == 8)).sum()),
                    "flags_outside_allowed": int((~np.isin(ff, allowed_agg)).sum())}
    gr["flag_by_county_outside_allowed"] = int((~np.isin(fbc, allowed_agg)).sum())
    gr["sd_by_county_validate"] = cflags.validate_grid(vbc, zc["sd_by_county"], fbc, dates)
    if pre_v:
        sf = pd.concat(sd_frames, ignore_index=True)
        sg = sf[sf["cell"] >= 0]
        want = ra.sd_array(sg, (sg["cls"] * A + sg["cell"]).to_numpy(), (C, A), T)
        gr["sd_recomputed_cells_by_class"] = eq_nan(zc["sd"], want, tol=1e-6)
        gr["sd_recomputed_cells_by_class"]["rel_max"] = float(np.nanmax(
            np.abs(zc["sd"] - want) / np.where(want > 0, want, np.nan))) if np.isfinite(
            want).any() else 0.0
        want_all = ra.sd_array(sg, sg["cell"].to_numpy(), (A,), T)
        gr["sd_recomputed_total_all"] = eq_nan(zt["sd_all"], want_all, tol=1e-6)
        sn = sg[sg["cls"] != DOM]
        want_nd = ra.sd_array(sn, sn["cell"].to_numpy(), (A,), T)
        gr["sd_recomputed_total_nodomestic"] = eq_nan(zt["sd_nodomestic"], want_nd, tol=1e-6)
        gr["sd_finite_vs_nfill"] = int((np.isfinite(zc["sd"]) != (zc["n_fill"] > 0)).sum())
    if variant == "full":
        ze = np.load(d / "cells_by_class_evensplit.npz")
        gr["evensplit"] = {"validate_grid": cflags.validate_grid(ze["value"], ze["sd"],
                                                                 ze["flag"], dates),
                           "nan_flag8_mismatch": int((np.isnan(ze["value"])
                                                      != (ze["flag"] == 8)).sum()),
                           "flags_outside_0_1_8_9": int((~np.isin(ze["flag"], [0, 1, 8, 9]))
                                                        .sum())}
        j = T - 1
        cnt = {}
        anyk = np.zeros((C, A), bool)
        for ki, tag in enumerate(counties):
            z = np.load(d / pumps_file(tag))
            mt = pd.read_parquet(d / meters_file(tag)).set_index("meter_idx")
            isA = ((mt["phase_code"] == "A") | ((mt["phase_code"] == "unknown")
                                                 & (mt["phase_detected"] == "A")))
            a_p = isA.reindex(z["meter_idx"]).fillna(False).to_numpy(bool)
            rows = pump_meter_rows(tag, z, ctxs[tag])
            prej = ctxs[tag]["inst"][rows] > j
            target = a_p & ~prej
            cnt[f"phaseA_last_month_not_nan_flag8_m4:{tag}"] = int((target & ~(
                np.isnan(z["value"][:, j]) & (z["flag"][:, j] == 8)
                & (z["method"][:, j] == 4))).sum())
            cnt[f"method4_outside_target:{tag}"] = int(((z["method"][:, j] == 4) & ~target).sum()
                                                       + (z["method"][:, :j] == 4).sum())
            ci = np.array([CLS.index(c) for c in z["purpose_class"]])
            sel = target & (z["cell"] >= 0)
            pa = np.zeros((C, A), bool)
            pa[ci[sel], z["cell"][sel]] = True
            cnt[f"county_cells_mismatch:{tag}"] = int(((fbc[ki][:, :, j] == 8) != pa).sum())
            anyk |= pa
        cnt["class_cells_mismatch"] = int(((f[:, :, j] == 8) != anyk).sum())
        cnt["earlier_months_flag8_cells"] = int((f[:, :, :j] == 8).sum())
        gr["ii_cells_2026_07"] = cnt
    lt = pd.read_parquet(d / "county_class_monthly.parquet")
    ltr = {"validate_long_table": cflags.validate_long_table(lt)}
    for ki, tag in enumerate(counties):
        q = lt[(lt["county"] == tag) & (lt["scope"] == "in_grid") & lt["class"].isin(CLS)]
        M = q.pivot(index="class", columns="date", values="value").reindex(CLS).to_numpy()
        ff = q.pivot(index="class", columns="date", values="flag").reindex(CLS).to_numpy()
        ltr[f"{tag}:in_grid"] = compare(np.where(ff != 8, M, 0.0),
                                        np.where(ff != 8, np.nansum(vbc[ki], 1), 0.0))
        nf = q.pivot(index="class", columns="date", values="n_fill").reindex(CLS).to_numpy()
        ltr[f"{tag}:in_grid_nfill"] = compare(nf, nfill[ki].sum(1))
        sdq = q.pivot(index="class", columns="date", values="sd").reindex(CLS).to_numpy()
        ltr[f"{tag}:sd_finite_vs_nfill"] = {"n_violations": int(
            (np.isfinite(sdq) != (nf > 0)).sum())}
    agg["long_table"] = ltr

    def count(x) -> int:
        if isinstance(x, dict):
            if "n_violations" in x:
                return int(x["n_violations"])
            n = 0
            for k, vv in x.items():
                if k in ("validate_grid", "validate_long_table", "sd_by_county_validate"):
                    n += len(vv)
                elif k in ("rel_max", "max_abs", "max_abs_delta", "sum_abs_delta", "sum_ref",
                           "n_elements", "shape_equal"):
                    continue
                else:
                    n += count(vv)
            return n
        if isinstance(x, list):
            return len(x)
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            return int(x)
        return 0
    nv = viol + count(agg) + count(gr)
    return {"per_county": out, "aggregates": agg, "grids": gr, "violations": int(nv)}


def r4() -> dict:
    meta = score("R4")
    P2 = v2()
    l1 = l1_v2()
    ctxs = {t: ctx_independent(t, l1) for t in lib.COUNTIES}
    out = {v: r4_variant(v, P2[v], l1, ctxs) for v in ("preorigin", "full")}
    kept = {}
    for tag in lib.COUNTIES:
        sub = l1[l1["county"] == tag]
        ctx = ctxs[tag]
        km = set(ctx["midx"][ctx["kept"]])
        rec = set(sub.loc[sub["meter_idx"].isin(km), "uid"])
        l1k = set(sub.loc[sub["kept"], "uid"])
        kept[f"census_L1:{tag}"] = {"symmetric_difference": len(l1k ^ rec)}
        for variant in ("preorigin", "full"):
            u = set(np.load(P2[variant] / pumps_file(tag))["uid"])
            want = rec if tag != "chiayi" else rec & set(sub.loc[sub["in_grid"], "uid"])
            kept[f"{variant}:{tag}"] = {"symmetric_difference": len(u ^ want)}
    out["kept_set"] = kept
    pre_ok = out["preorigin"]["violations"] == 0 and all(
        v["symmetric_difference"] == 0 for k, v in kept.items() if not k.startswith("full"))
    full_ok = out["full"]["violations"] == 0 and all(
        v["symmetric_difference"] == 0 for k, v in kept.items() if not k.startswith("pre"))
    l1_ok = all(v["symmetric_difference"] == 0 for k, v in kept.items()
                if k.startswith("census_L1"))
    return {**meta, **out, "pass_preorigin_part": bool(pre_ok), "pass_full_part": bool(full_ok),
            "pass_l1_kept": bool(l1_ok), "pass": bool(pre_ok and full_ok and l1_ok)}


# ----------------------------------------------------------------------------- T5
def t5() -> dict:
    meta = score("T5")
    P2 = v2()
    pre, full = P2["preorigin"], P2["full"]
    out = {}
    for tag in lib.COUNTIES:
        a, b = np.load(pre / pumps_file(tag)), np.load(full / pumps_file(tag))
        same = bool(np.array_equal(a["uid"], b["uid"]))
        r = {"uid_equal": same}
        if same:
            r["months_to_2022_11"] = eq_nan(b["value"][:, :J], a["value"][:, :J])
            r["flags_to_2022_11_mismatch"] = int((b["flag"][:, :J] != a["flag"][:, :J]).sum())
            nf = a["flag"][:, J] != 4
            mt = pd.read_parquet(pre / meters_file(tag)).set_index("meter_idx")
            paired = mt["dec_jan_paired"].reindex(a["meter_idx"]).to_numpy(bool)
            keep = ~paired
            r["dec2022_unpaired"] = eq_nan(b["value"][keep, J], a["value"][keep, J])
            r["dec2022_flag4_only_paired"] = int((~nf & ~paired).sum())
        out[tag] = r
    zc_p, zc_f = np.load(pre / "cells_by_class.npz"), np.load(full / "cells_by_class.npz")
    out["cells_value_by_county_to_2022_11"] = eq_nan(zc_f["value_by_county"][..., :J],
                                                     zc_p["value_by_county"][..., :J])
    nf0 = zc_p["n_fill"][..., J] == 0
    out["cells_dec2022_without_fill"] = eq_nan(zc_f["value"][..., J][nf0],
                                               zc_p["value"][..., J][nf0])
    zt_p, zt_f = np.load(pre / "cells_total.npz"), np.load(full / "cells_total.npz")
    for key in ("value_all", "value_nodomestic"):
        out[f"cells_total:{key}_to_2022_11"] = eq_nan(zt_f[key][..., :J], zt_p[key][..., :J])

    def bad(x) -> int:
        if isinstance(x, dict):
            if "n_violations" in x:
                return int(x["n_violations"])
            return sum(bad(v) for k, v in x.items() if k not in ("uid_equal",)) + (
                0 if x.get("uid_equal", True) else 1)
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            return int(x)
        return 0
    n = bad(out)
    return {**meta, "items": out, "violations": n, "pass": n == 0}


# ----------------------------------------------------------------------------- R5, R6
def r5() -> dict:
    meta = score("R5")
    meters = checks.raw_meter_numbers()
    viol, checked = [], []
    names = checks.FORBIDDEN_NAMES
    for name, d in v2().items():
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = f"{name}/{p.relative_to(d)}"
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
                bad = sorted(set(df.columns) & names)
                if bad:
                    viol.append({"file": rel, "forbidden_columns": bad})
                for c in df.columns:
                    h = checks._hits(df[c].to_numpy(), meters)
                    if h >= 0.01:
                        viol.append({"file": rel, "column": c, "raw_meter_share": h})
            elif p.suffix == ".npz":
                z = np.load(p)
                bad = sorted(set(z.files) & names)
                if bad:
                    viol.append({"file": rel, "forbidden_arrays": bad})
                for k in z.files:
                    a = z[k]
                    if a.ndim == 1:
                        h = checks._hits(a, meters)
                        if h >= 0.01:
                            viol.append({"file": rel, "array": k, "raw_meter_share": h})
            elif p.suffix == ".json":
                keys = set()
                checks._json_keys(lib.load_json(p), keys)
                bad = sorted(keys & names)
                if bad:
                    viol.append({"file": rel, "forbidden_keys": bad})
            else:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                bad = sorted(n for n in names if n in txt)
                if bad:
                    viol.append({"file": rel, "forbidden_strings": bad})
            checked.append(rel)
    return {**meta, "n_raw_meter_numbers": len(meters), "n_files": len(checked),
            "files": checked, "violations": viol, "pass": len(viol) == 0}


def r6() -> dict:
    meta = score("R6")
    P2 = v2()
    grid = lib.load_grid()
    l1 = l1_v2()
    res = {}
    for tag in lib.COUNTIES:
        z = np.load(P2["preorigin"] / pumps_file(tag))
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
    return {**meta, "counts": res, "l1_uid_unique": uid_unique, "pass": bool(ok and uid_unique)}


# ----------------------------------------------------------------------------- R2
SEED = 20221231


def perturbed_inputs(tag: str, grid):
    df, k, L = rb.load_county(tag, grid)
    fr = np.where(df["meter_first_pump"].to_numpy())[0]
    X = L[fr, :T_PRE]
    code = lib.phase_from_code(df["billing_code"].to_numpy()[fr])
    det = lib.detect_phase(X, lib.tidx("2012-01"), J)["phase_detected"].to_numpy()
    paired_m = np.where((det == "B") | ((code == "B") & np.isin(det, ["mixed", "insufficient",
                                                                      "both-flat"])))[0]
    rng = np.random.default_rng(SEED)
    L2 = L.copy()
    post = L2[:, T_PRE:]
    L2[:, T_PRE:] = np.where(np.isfinite(post), 3.0 * post + 100.0, 50.0)
    rows = np.where(df["meter_idx"].isin(paired_m).to_numpy())[0]
    dj = L2[rows, J]
    L2[rows, J] = np.where(np.isfinite(dj), dj * rng.uniform(0.0, 5.0, len(rows)), 77.0)
    ri = pd.Index(df["sid"].astype(str)).get_indexer(k["pump"].astype(str))
    ti = ((k["datetime"].dt.year - 2007) * 12 + k["datetime"].dt.month - 1).to_numpy()
    k2 = k.copy()
    k2["electricity_kwh"] = L2[ri, ti]
    n_changed = int((~((L2 == L) | (np.isnan(L2) & np.isnan(L)))).sum())
    return (df, k2, L2), {"pump_months_changed": n_changed,
                          "paired_meters": int(len(paired_m)), "pumps_on_paired": int(len(rows))}


def r2() -> dict:
    meta = score("R2")
    P2, P1 = v2(), v1()
    grid = lib.load_grid()
    out_dir = SCRATCH / "r2_perturbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    zc = np.load(P2["preorigin"] / "cells_by_class.npz")
    counties = list(zc["counties"])
    l1 = l1_v2()
    res, total = {}, 0
    for tag in lib.COUNTIES:
        data, pinfo = perturbed_inputs(tag, grid)
        rb.run(tag, out_dir, variants=("preorigin",), data=data)
        r = {"perturbation": pinfo}
        a = np.load(out_dir / f"preorigin_{tag}_pumps.npz")
        b = np.load(P2["preorigin"] / pumps_file(tag))
        cmp_ = {}       # the rebuilt Chiayi file already holds the in-grid pumps only
        for key in b.files:
            x = a[key]
            y = b[key]
            if key in ("uid", "purpose_class", "dates", "meter_idx", "cell", "flag", "method"):
                cmp_[key] = {"n_violations": 0 if np.array_equal(x, y) else 1}
            else:
                cmp_[key] = eq_nan(x, y)
        r["pumps"] = cmp_
        ma = pd.read_parquet(out_dir / f"preorigin_{tag}_meters.parquet")
        if tag == "chiayi":
            ma = ma[ma["in_grid_any"]].reset_index(drop=True)
        mb = pd.read_parquet(P2["preorigin"] / meters_file(tag))
        try:
            pd.testing.assert_frame_equal(ma, mb, check_exact=False, rtol=0, atol=1e-9)
            r["meters"] = {"n_violations": 0}
        except AssertionError as e:
            r["meters"] = {"n_violations": 1, "error": str(e)[:500]}
        cz = np.load(out_dir / f"preorigin_{tag}_cells.npz")
        parts = {kk[5:]: cz[kk] for kk in cz.files if kk.startswith("cell_")}
        f = ra.flags_from_parts(parts)
        ki = counties.index(tag)
        r["value_by_county"] = eq_nan(ra.masked(parts["value"], f), zc["value_by_county"][ki])
        r["flag_by_county"] = {"n_violations": int((f != zc["flag_by_county"][ki]).sum())}
        fz = np.load(out_dir / f"preorigin_{tag}_fill.npz")
        ffr = pd.DataFrame({"cell": fz["cell"], "cls": fz["cls"], "meter": fz["meter_idx"],
                            "cc": fz["cc"], "ic": fz["ic"]})
        ffr = ffr[ffr["cell"] >= 0]
        A = int(zc["n_active"])
        T = zc["value"].shape[-1]
        sdr = ra.sd_array(ffr, (ffr["cls"] * A + ffr["cell"]).to_numpy(), (C, A), T)
        r["sd_by_county"] = eq_nan(sdr, zc["sd_by_county"][ki])
        la = pd.read_parquet(out_dir / f"l1_{tag}.parquet").reset_index(drop=True)
        lb = l1[l1["county"] == tag].reset_index(drop=True)
        cols = ["uid", "kept", "meter_duty_2012_2022", "hp_share", "dec_jan_paired",
                "phase_detected_2012_2022", "meter_hp_sum"]
        try:
            pd.testing.assert_frame_equal(la[cols], lb[cols], check_exact=False, rtol=0,
                                          atol=1e-12)
            r["l1_columns"] = {"n_violations": 0}
        except AssertionError as e:
            r["l1_columns"] = {"n_violations": 1, "error": str(e)[:500]}
        nv = sum(x["n_violations"] for x in cmp_.values()) + sum(
            r[x]["n_violations"] for x in ("meters", "value_by_county", "flag_by_county",
                                           "sd_by_county", "l1_columns"))
        # positive control: v1's pre-origin builder (model convention) on the same perturbed
        # L0 against the delivered v1 product: must change (the test detects the leak)
        df, k2, L2 = data
        r1v = build.process(tag, df, k2, L2, "preorigin", grid)
        zv1 = np.load(P1["preorigin"] / pumps_file(tag))
        sub = r1v["sub"]
        kr = np.ones(len(sub), bool) if tag != "chiayi" else (sub["cell"].to_numpy() >= 0)
        vv = r1v["Vp"][kr]
        same_uid = bool(np.array_equal(sub["uid"].to_numpy(dtype=str)[kr], zv1["uid"]))
        ctrl = {"uid_equal": same_uid}
        if same_uid:
            dd = ~((vv == zv1["value"]) | (np.isnan(vv) & np.isnan(zv1["value"])))
            ctrl["pump_months_changed"] = int(dd.sum())
            ctrl["months_changed"] = sorted({lib.ym(t) for t in np.nonzero(dd)[1]})[:12]
        else:
            ctrl["kept_uid_symdiff"] = len(set(sub["uid"][kr]) ^ set(zv1["uid"]))
            ctrl["pump_months_changed"] = None
        ctrl["detects_leak"] = bool((not same_uid) or ctrl["pump_months_changed"] > 0)
        r["control_v1_builder"] = ctrl
        r["violations"] = int(nv)
        total += nv
        res[tag] = r
        del r1v, data
    ctrl_ok = all(res[t]["control_v1_builder"]["detects_leak"] for t in ("yunlin", "changhua"))
    l1_ok = all(res[t]["l1_columns"]["n_violations"] == 0 for t in lib.COUNTIES)
    return {**meta, "seed": SEED, "per_county": res, "violations": int(total),
            "control_detects_leak_yl_ch": bool(ctrl_ok), "pass_l1_part": bool(l1_ok),
            "pass": bool(total == 0 and ctrl_ok)}


# ----------------------------------------------------------------------------- R3
def r3() -> dict:
    meta = score("R3")
    P2, P1 = v2(), v1()
    E, emeta = checks.model_E()
    out = {}
    for name, d in (("v2", P2["preorigin"]), ("v1", P1["preorigin"])):
        z = np.load(d / "cells_by_class.npz")
        V = z["value_by_county"][list(z["counties"]).index("yunlin")][:, :, W_FULL]
        dd = V - E
        tot = E.sum((0, 1))
        rel = np.abs(dd.sum((0, 1))) / tot
        n = W_NOV.stop - W_NOV.start
        out[name] = {
            "max_rel_month_2012_01_2022_11": float(rel[:n].max()),
            "worst_month_2012_01_2022_11": lib.ym(W_FULL.start + int(rel[:n].argmax())),
            "sum_abs_kwh_2012_01_2022_11": float(np.abs(dd[..., :n]).sum()),
            "rel_month_2022_12": float(rel[n]),
            "delta_fan_total_2022_12_kwh": float(dd[..., n].sum()),
            "model_fan_total_2022_12_kwh": float(tot[n]),
            "sum_abs_kwh_2012_2022": float(np.abs(dd).sum()),
            "max_element_abs_kwh": float(np.abs(dd).max()),
            "fan_total_GWh_per_yr_2012_2022": float(V.sum() / 1e6 / 11)}
    t2a = lib.load_json(lib.EVIDENCE / "checks" / "T2a.json")
    out["amp_absent_kwh_2012_2022_v1_T2a"] = t2a["absent_kwh_2012_2022"]
    return {**meta, **out, "pass": None, "gating": False}


# ----------------------------------------------------------------------------- R7
def r7() -> dict:
    meta = score("R7")
    P2 = v2()
    l1 = l1_v2()
    z90 = 1.6448536269514722
    grp_rows = []
    for ki, tag in enumerate(lib.COUNTIES):
        ctx = ctx_independent(tag, l1)
        sel = ctx["paired"] & ctx["kept"]
        z = np.load(P2["preorigin"] / pumps_file(tag))
        rows = pump_meter_rows(tag, z, ctx)
        share = z["hp_share"]
        cell = z["cell"]
        ci = np.array([CLS.index(c) for c in z["purpose_class"]])
        X = ctx["X"]
        for y in rl.FIT_YEARS:
            train = tuple(t for t in rl.FIT_YEARS if t != y)
            par = ratio_fit(X, ctx["inst"], ctx["cls"], sel, train)
            to = lib.tidx(f"{y}-10")
            a = 0.5 * (X[:, to + 2] + X[:, to + 3])
            b = 0.5 * (X[:, to] + X[:, to + 1])
            okm = sel & np.isfinite(a) & np.isfinite(b) & (ctx["inst"] <= to)
            okp = okm[rows] & (cell >= 0)
            rho = np.array([par["classes"][c]["rho_bar"] for c in ctx["cls"]])
            s = np.array([par["classes"][c]["s"] for c in ctx["cls"]])
            bp = share * b[rows]
            F = share * rho[rows] * b[rows]
            Act = share * a[rows]
            cc = s[rows] * bp
            ic = par["sigma"] * bp
            d = pd.DataFrame({"key": ci * 10**5 + cell, "meter": ki * 10**7 + z["meter_idx"],
                              "F": F, "A": Act, "cc": cc, "ic": ic})[okp]
            d["county"] = tag
            d["year"] = y
            grp_rows.append(d)
    allp = pd.concat(grp_rows, ignore_index=True)
    res = {}
    for scope, keycols in (("class_x_cell", ["year", "key"]), ("county_in_grid", ["year",
                                                                                 "county"])):
        g = allp.groupby(keycols)
        F = g["F"].sum()
        Act = g["A"].sum()
        com = g["cc"].sum()
        idio = allp.groupby(keycols + ["meter"])["ic"].sum().pow(2).groupby(level=keycols).sum()
        sd = np.sqrt(com ** 2 + idio.reindex(com.index))
        m = F > 0
        zsc = ((Act - F) / sd)[m]
        cov = float((zsc.abs() <= z90).mean())
        rel = ((F - Act) / Act)[m & (Act > 0)]
        r = {"n": int(m.sum()), "coverage_90": cov,
             "median_abs_rel_error": float(rel.abs().median()),
             "mean_rel_bias": float(rel.mean())}
        if scope == "county_in_grid":
            t = pd.DataFrame({"F": F, "A": Act, "sd": sd, "z": (Act - F) / sd}).reset_index()
            r["table"] = t.to_dict(orient="records")
            ylch = t[t["county"].isin(["yunlin", "changhua"])]
            r["coverage_90_yunlin_changhua"] = float((ylch["z"].abs() <= z90).mean())
            r["n_yunlin_changhua"] = int(len(ylch))
        else:
            r["coverage_90_by_year"] = {int(y): float((zz.abs() <= z90).mean())
                                        for y, zz in zsc.groupby(level=0)}
        res[scope] = r
    cal = (0.80 <= res["class_x_cell"]["coverage_90"] <= 0.97
           and res["county_in_grid"]["coverage_90_yunlin_changhua"] >= 0.75)
    return {**meta, **res, "sd_label": "calibrated" if cal else "uncalibrated",
            "rule": "calibrated iff class x cell coverage of the nominal 90 % interval is in "
                    "[0.80, 0.97] and the Yunlin+Changhua county in-grid coverage >= 0.75",
            "pass": None, "gating": False}


# ----------------------------------------------------------------------------- R8
def r8() -> dict:
    meta = score("R8")
    P2, P1 = v2(), v1()
    l1 = l1_v2()
    res = {}
    for tag in lib.COUNTIES:
        ctx = ctx_independent(tag, l1)
        Lm = ctx["Lm"]
        z2 = np.load(P2["preorigin"] / pumps_file(tag))
        z1 = np.load(P1["preorigin"] / pumps_file(tag))
        g2 = z2["cell"] >= 0
        fl4 = (z2["flag"][:, J] == 4) & g2
        fill = float(np.nansum(z2["value"][fl4, J]))
        common = pd.Index(z1["uid"]).get_indexer(z2["uid"][fl4])
        ok = common >= 0
        actual = float(np.nansum(z1["value"][common[ok], J]))
        dec_v1 = float(np.nansum(z1["value"][z1["cell"] >= 0, J]))
        dec_v2 = float(np.nansum(z2["value"][g2, J]))
        # rule capture: kept in-grid meters with Dec-2022, Jan-2023 finite, not both zero
        mt = pd.read_parquet(P2["full"] / meters_file(tag))
        mt = mt[mt["in_grid_any"]]
        rows = pd.Series(np.arange(len(ctx["midx"])), index=ctx["midx"]).loc[
            mt["meter_idx"]].to_numpy()
        dd, jj = Lm[rows, J], Lm[rows, J + 1]
        valid = np.isfinite(dd) & np.isfinite(jj) & ((dd > 0) | (jj > 0))
        eqh = valid & np.isclose(dd, jj, rtol=0, atol=lib.PAIR_ATOL)
        paired = mt["dec_jan_paired"].to_numpy(bool)
        e_unt = float(np.nansum(np.where(eqh & ~paired, dd, 0.0)))
        res[tag] = {
            "fill_in_grid_kWh": fill, "v1_value_same_pumps_kWh": actual,
            "fill_minus_actual_rel": (fill - actual) / actual if actual else None,
            "dec2022_in_grid_v1_kWh": dec_v1, "dec2022_in_grid_v2_kWh": dec_v2,
            "v1_paired_share_of_dec2022": actual / dec_v1 if dec_v1 else None,
            "meters_valid": int(valid.sum()), "meters_equal_halves": int(eqh.sum()),
            "rule_sensitivity": float((eqh & paired).sum() / max(eqh.sum(), 1)),
            "rule_precision": float((eqh & paired).sum() / max((paired & valid).sum(), 1)),
            "untreated_equal_halves_meters": int((eqh & ~paired).sum()),
            "untreated_equal_halves_meter_kWh_dec2022": e_unt,
            "untreated_share_of_dec2022_in_grid_meter_level": e_unt / dec_v2 if dec_v2 else None}
    return {**meta, "per_county": res, "note": "after the fact: uses the January-2023 L0 "
            "readings and the v1 values (which contain them); never feeds back into v2",
            "pass": None, "gating": False}


# ----------------------------------------------------------------------------- R9
def r9() -> dict:
    meta = score("R9")
    P2 = v2()
    grid = lib.load_grid()
    out_dir = SCRATCH / "r9_no_postorigin_installs"
    out_dir.mkdir(parents=True, exist_ok=True)
    zc = np.load(P2["preorigin"] / "cells_by_class.npz")
    counties = list(zc["counties"])
    res = {}
    for tag in lib.COUNTIES:
        raw = lib.read_census_l0(tag)
        df = lib.harmonise(tag, raw, grid)
        k, L = lib.read_kwh_l0(tag, df["sid"].to_numpy(), len(df))
        late = df["install_valid"].to_numpy() & (df["_t_install"].to_numpy() > J)
        r = {"pumps_install_after_2022_12": int(late.sum()),
             "pumps_install_after_2022_12_in_grid": int((late & df["in_grid"]).sum()),
             "latest_valid_install": max((m for m, v in zip(df["install_month"],
                                                            df["install_valid"], strict=True)
                                          if v and m), default=None),
             "pumps_install_after_2016_12": int((df["install_valid"].to_numpy()
                                                 & (df["_t_install"].to_numpy()
                                                    > lib.tidx("2016-12"))).sum())}
        mlate = set(df.loc[late, "meter_idx"])
        r["meters_with_late_pump"] = len(mlate)
        r["meters_with_late_and_earlier_pumps"] = int(sum(
            1 for m in mlate if (~late[df["meter_idx"].to_numpy() == m]).any()))
        if late.any():
            keep = ~late
            raw2 = raw[keep].reset_index(drop=True)
            df2 = lib.harmonise(tag, raw2, grid)
            if tag == "chiayi":     # sids are 'r<position>': keep the original pump ids
                df2["sid"] = df.loc[keep, "sid"].to_numpy()
                df2["uid"] = tag + ":" + df2["sid"]
            L2 = L[keep]
            k2 = k[k["pump"].astype(str).isin(set(df2["sid"].astype(str)))].reset_index(
                drop=True)
            rb.run(tag, out_dir, variants=("preorigin",), data=(df2, k2, L2))
            cz = np.load(out_dir / f"preorigin_{tag}_cells.npz")
            v_alt = cz["cell_value"]
            v_ref = np.nan_to_num(zc["value_by_county"][counties.index(tag)], nan=0.0)
            tot_alt, tot_ref = v_alt.sum((0, 1)), v_ref.sum((0, 1))
            w = W_FULL
            r["in_grid_GWh_2012_2022_ref"] = float(tot_ref[w].sum() / 1e6)
            r["in_grid_GWh_2012_2022_without_late"] = float(tot_alt[w].sum() / 1e6)
            r["rel_change_2012_2022"] = float(tot_alt[w].sum() / tot_ref[w].sum() - 1)
            rel = np.abs(tot_alt[w] - tot_ref[w]) / np.where(tot_ref[w] > 0, tot_ref[w], np.nan)
            r["max_rel_month_2012_2022"] = float(np.nanmax(rel))
            r["sum_abs_cell_class_kwh_2012_2022"] = float(np.abs(v_alt[..., w] - v_ref[..., w])
                                                          .sum())
            r["by_class_rel_change_2012_2022"] = {
                c: float(v_alt[i][..., w].sum() / v_ref[i][..., w].sum() - 1)
                if v_ref[i][..., w].sum() > 0 else None for i, c in enumerate(CLS)}
        res[tag] = r
    return {**meta, "per_county": res, "pass": None, "gating": False}


# ----------------------------------------------------------------------------- R10
def r10() -> dict:
    meta = score("R10")
    P2 = v2()
    out = {}
    for variant in ("preorigin", "full"):
        d = P2[variant]
        zc = np.load(d / "cells_by_class.npz")
        counties = list(zc["counties"])
        vb = np.nan_to_num(zc["value_by_county"], nan=0.0)
        r = {}
        for ki, tag in enumerate(counties):
            e = vb[ki][..., W_FULL]
            r[tag] = {"in_grid_GWh_per_yr_2012_2022": float(e.sum() / 1e6 / 11),
                      "domestic_share_2012_2022": float(e[DOM].sum() / e.sum())}
        me = zc["missing_expected_kwh"][..., W_FULL].sum()
        r["missing_expected_share_2012_2022"] = float(me / vb[..., W_FULL].sum())
        f = zc["flag"]
        r["flag_counts_class_cell_months"] = {int(k): int(v) for k, v in
                                              zip(*np.unique(f, return_counts=True), strict=True)}
        if variant == "preorigin":
            v = np.nan_to_num(zc["value"], nan=0.0)
            r["dec2022_in_grid_kWh"] = float(v[..., J].sum())
            r["dec2022_filled_cells"] = int((zc["n_fill"][..., J] > 0).sum())
            zt = np.load(d / "cells_total.npz")
            sda = zt["sd_all"][:, J]
            r["dec2022_sd_fan_total_upper_kWh"] = float(np.nansum(sda))
            fp = lib.load_json(d / "fill_parameters.json")
            r["fill_parameters"] = {t: {"sigma": fp["counties"][t]["sigma"],
                                        "classes": {c: {k: fp["counties"][t]["classes"][c][k]
                                                        for k in ("rho_bar", "s",
                                                                  "pooled_to_county")}
                                                    for c in CLS}}
                                    for t in lib.COUNTIES}
        bi = lib.load_json(d / "build_info.json")
        r["counties_build"] = {t: {k: bi["counties"][t][k] for k in (
            "pumps_kept", "pumps_kept_in_grid", "meters_kept", "meters_terminated_in_grid",
            "meters_dec_jan_paired", "meters_filled_dec2022", "meters_paired_without_base",
            "phase_agree_frac_bimonthly", "phase_agree_frac_monthly")} for t in lib.COUNTIES}
        out[variant] = r
    return {**meta, **out, "pass": None, "gating": False}


# ----------------------------------------------------------------------------- summary
def summary() -> dict:
    r = {k: lib.load_json(OUT / f"{k}.json") for k in TID if (OUT / f"{k}.json").exists()}
    g = lambda k, f="pass": r.get(k, {}).get(f)  # noqa: E731
    pre = {"R1_preorigin_part": g("R1", "pass_preorigin_part"), "R2": g("R2"),
           "R4_preorigin_part": g("R4", "pass_preorigin_part"), "R5": g("R5"), "R6": g("R6"),
           "T5": g("T5")}
    full = {"R1_full_part": g("R1", "pass_full_part"),
            "R4_full_part": g("R4", "pass_full_part"), "R5": g("R5"), "R6": g("R6"),
            "T5": g("T5")}
    l1 = {"R2_l1_part": g("R2", "pass_l1_part"), "R4_l1_kept": g("R4", "pass_l1_kept"),
          "R5": g("R5"), "R6": g("R6")}
    gates = {
        "pump_electricity/preorigin": {"tests": pre, "passed": all(v is True
                                                                  for v in pre.values())},
        "pump_electricity/full": {"tests": full, "passed": all(v is True for v in full.values())},
        "pump_electricity/census_L1": {"tests": l1, "passed": all(v is True for v in l1.values())},
    }
    out = {"prereg": PREREG_ID, "tests": {k: v.get("pass") for k, v in r.items()},
           "gates": gates, "sd_label": g("R7", "sd_label"),
           "reported_only": ["R3", "R7", "R8", "R9", "R10"]}
    lib.save_json(out, OUT / "summary.json")
    print(gates, "sd:", out["sd_label"])
    return out


FUNCS = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6, "R7": r7, "R8": r8,
         "R9": r9, "R10": r10, "T5": t5}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arch = OUT / "code_archive"
    sha = sha256_file(Path(__file__))
    if not (arch / f"repair_checks.{sha[:12]}.py").exists():
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), arch / f"repair_checks.{sha[:12]}.py")
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
