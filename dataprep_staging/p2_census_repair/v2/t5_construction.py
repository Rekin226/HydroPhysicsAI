"""Registered tests P2r2.k1.construction_<variant>: construction checks of an assembled
product (staging directory ``_work_p2abc_v2/assembled_<variant>``) against the C0 v3 inputs and
the stage-1 arrays (pre-registration P2abc.r2.c0_based.v1; items (a)-(i) as registered).

Usage: python -I t5_construction.py <variant> <out_dir> [--repair]
(``--repair``: the identical checks registered as P2r2.k1r.construction_<variant> in
P2abc.r2.k1_repair.v1, scored on the re-assembled product; output *_repair.json.)
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_finalize as fin  # noqa: E402
import p2lib as L  # noqa: E402
import scoring  # noqa: E402

TOL_ABS, TOL_REL = 1e-6, 1e-9
FORBIDDEN = ("戶名1", "用電地址2", "郵遞區號", "x_twd97", "y_twd97", "TWD97_X", "TWD97_Y",
             "電號1", "電號", "電號2")
L0P = L.REPO / "data_fetch_api" / "new_measurements" / \
    "tpc_changhua_yunlin_chiayi_pingtung_2026" / "parquet"
METER_COL = {"yunlin": "電號1", "changhua": "電號1", "chiayi": "電號"}


def code_sha() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), L.CODE_DIR / "p2lib.py", L.CODE_DIR / "b3_assemble.py",
                      L.CODE_DIR / "b4_finalize.py")}


def close(a, b) -> np.ndarray:
    """Element-wise equality within tolerance, NaN == NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isnan(a) & np.isnan(b)
    ok = np.abs(a - b) <= TOL_ABS + TOL_REL * np.abs(b)
    return both | (ok & ~np.isnan(a) & ~np.isnan(b))


def run(variant: str, out_dir: Path, repair: bool = False) -> dict:
    t0 = time.time()
    res = {"variant": variant, "code_sha256": code_sha(), "items": {}, "repair": repair}
    if repair:
        res["authorization"] = scoring.authorize([f"P2r2.k1r.construction_{variant}"],
                                                 prereg_id="P2abc.r2.k1_repair.v1")
    else:
        res["authorization"] = scoring.authorize([f"P2r2.k1.construction_{variant}"])
    d = L.WORK / f"assembled_{variant}"
    prod = L.open_c0(variant)
    T = L.VARIANTS[variant]["T"]
    t26 = L.tidx("2026-07") if variant == "full" else None
    v = {k: 0 for k in "abcdefghi"}
    det = {k: {} for k in "abcdefghi"}
    cz = np.load(prod.path("cells_by_class.npz"))
    ct = np.load(prod.path("cells_total.npz"))
    rec = np.load(d / "kwh_by_class_recorded.npz")
    clz = np.load(d / "kwh_by_class_chowlin.npz")
    A = int(cz["n_active"])
    C = len(L.CLASSES)
    l1 = None
    for ki, c in enumerate(L.COUNTIES):
        cd = L.CountyData(variant, c, prod=prod, l1=l1)
        z = np.load(d / f"pumps_{L.FILETAG[c]}.npz", allow_pickle=False)
        val, flg, met, sd = z["value"], z["flag"], z["method"], z["sd"]
        assert np.array_equal(z["uid"], cd.uid)
        # (b) equal to C0 wherever the method is not chowlin_split / imputed
        new = (met == L.M_CHOWLIN) | (met == L.M_JULY_IMPUTED)
        same_v = close(val, cd.pvalue)
        bad_b = (~new & ~(same_v & (flg == cd.pflag) & (met == cd.pmethod)))
        v["b"] += int(bad_b.sum())
        st1 = np.load(L.WORK / f"{variant}_{c}_cl.npz")
        rs = np.zeros(cd.M.shape, bool)
        rs[st1["rows"]] = st1["resplit"]
        bad_m10 = (met == L.M_CHOWLIN) & ~rs[cd.prow]
        v["b"] += int(bad_m10.sum())
        if variant == "full":
            bad_m11 = (met == L.M_JULY_IMPUTED) & ~(
                (np.arange(T)[None, :] == t26) & (cd.pmethod == 4))
        else:
            bad_m11 = met == L.M_JULY_IMPUTED
        v["b"] += int(bad_m11.sum())
        det["b"][c] = {"value_flag_method_diff_outside_new": int(bad_b.sum()),
                       "chowlin_split_outside_resplit": int(bad_m10.sum()),
                       "imputation_outside_open_july": int(bad_m11.sum()),
                       "n_chowlin_split_pump_months": int((met == L.M_CHOWLIN).sum()),
                       "n_imputed_pump_months": int((met == L.M_JULY_IMPUTED).sum())}
        # (a) bill conservation and pairing of method 10 at meter level
        fp = cd.first_array_pump
        Mv = val[fp] / cd.share[fp, None]
        Mmeth = met[fp]
        mt = pd.read_parquet(L.WORK / f"{variant}_{c}_meters.parquet")
        cph = mt["chowlin_phase"].to_numpy()
        bad_a = bad_pair = 0
        maxd = 0.0
        for p_ in ("A", "B"):
            rows = np.where(cph == p_)[0]
            st = L.pair_starts(p_, 0, T, T)
            m10a = Mmeth[np.ix_(rows, st)] == L.M_CHOWLIN
            m10b = Mmeth[np.ix_(rows, st + 1)] == L.M_CHOWLIN
            bad_pair += int((m10a ^ m10b).sum())
            both = m10a & m10b
            s_new = Mv[np.ix_(rows, st)] + Mv[np.ix_(rows, st + 1)]
            s_old = cd.M[np.ix_(rows, st)] + cd.M[np.ix_(rows, st + 1)]
            dd = np.abs(s_new - s_old)
            okd = dd <= TOL_ABS + TOL_REL * np.abs(s_old)
            bad_a += int((both & ~okd).sum())
            if both.any():
                maxd = max(maxd, float(dd[both].max()))
            # stage-1 resplit pairs (incl. zero-sum ones) keep their sums too
            rsp = rs[np.ix_(rows, st)] & rs[np.ix_(rows, st + 1)]
            bad_a += int((rsp & ~okd).sum())
        v["a"] += bad_a + bad_pair
        det["a"][c] = {"pair_sum_violations": bad_a, "unpaired_chowlin_months": bad_pair,
                       "max_abs_pair_sum_diff_kwh": maxd}
        # (c) pre-install months: 0, flag 0 (Chow-Lin arm pumps)
        pre = cd.pmethod == 1
        bad_c = pre & ~((val == 0) & (flg == 0) & (met == 1))
        v["c"] += int(bad_c.sum())
        det["c"][c] = {"pre_install_pump_months": int(pre.sum()), "violations": int(bad_c.sum())}
        # (e) cells = sum of pumps (Chow-Lin arm, this county)
        cls = np.array([L.CLASSES.index(x) for x in cd.pclass])
        g = np.where(cd.cell >= 0, cls * A + cd.cell, -1)
        ok = g >= 0
        S = np.zeros((C * A, T))
        np.add.at(S, g[ok], np.nan_to_num(val[ok]))
        fb = clz["flag_by_county"][ki].reshape(C * A, T)
        vb = clz["value_by_county"][ki].reshape(C * A, T)
        m_ = fb != 8
        bad_e = int((~close(vb[m_], S[m_])).sum()) + int((~np.isnan(vb[~m_])).sum())
        v["e"] += bad_e
        det["e"][c] = {"cell_vs_pump_sum_violations": bad_e}
        # (f) pump sd and flags
        bad_f = int((np.isfinite(sd) != np.isin(flg, [3, 4])).sum())
        bad_f += int((~np.isin(flg, [0, 3, 4, 8, 9])).sum())
        bad_f += int((np.isnan(val) != (flg == 8)).sum())
        v["f"] += bad_f
        det["f"][c] = {"pump_violations": bad_f}
        # (g) raw meter numbers for the privacy scan
        raw = pd.read_parquet(L0P / f"{c}_census.parquet", columns=[METER_COL[c]])
        det["g"].setdefault("_raw", set()).update(raw[METER_COL[c]].dropna().astype(str))
    # (d) recorded arm cells equal C0 except cell-months holding an imputed pump
    nimp = rec["n_imp"]
    excl = nimp > 0
    excl_tot = excl.any(0)
    bad_d = 0
    for key_p, key_c, ex in (("value", "value", excl), ("flag", "flag", excl)):
        a_, b_ = rec[key_p], cz[key_c]
        bad_d += int(((~close(a_, b_)) & ~ex).sum()) if key_p == "value" else \
            int(((a_ != b_) & ~ex).sum())
    for key_p, key_c in (("value_by_county", "value_by_county"),
                         ("flag_by_county", "flag_by_county")):
        a_, b_ = rec[key_p], cz[key_c]
        ex = np.broadcast_to(excl, a_.shape)
        bad_d += int(((~close(a_, b_)) & ~ex).sum()) if key_p.startswith("value") else \
            int(((a_ != b_) & ~ex).sum())
    for mine, theirs in (("value_total_all", "value_all"), ("flag_total_all", "flag_all"),
                         ("value_total_nodomestic", "value_nodomestic"),
                         ("flag_total_nodomestic", "flag_nodomestic")):
        a_, b_ = rec[mine], ct[theirs]
        bad_d += int(((~close(a_, b_)) & ~excl_tot).sum()) if mine.startswith("value") else \
            int(((a_ != b_) & ~excl_tot).sum())
    v["d"] = bad_d
    det["d"] = {"violations": bad_d, "cell_months_excluded_imputed": int(excl.sum())}
    # (e) all-county arrays = sum by county; totals = sum of classes; nodomestic
    bad_e2 = 0
    for arr in (rec, clz):
        f_ = arr["flag"]
        m_ = f_ != 8
        s_ = np.nansum(arr["value_by_county"], 0)
        bad_e2 += int((~close(arr["value"][m_], s_[m_])).sum())
        ft = arr["flag_total_all"]
        mt_ = ft != 8
        tot = np.nansum(arr["value_by_county"], (0, 1))
        bad_e2 += int((~close(arr["value_total_all"][mt_], tot[mt_])).sum())
        dom = L.CLASSES.index("domestic")
        nd = np.nansum(np.delete(arr["value_by_county"], dom, axis=1), (0, 1))
        fn = arr["flag_total_nodomestic"]
        mn = fn != 8
        bad_e2 += int((~close(arr["value_total_nodomestic"][mn], nd[mn])).sum())
    v["e"] += bad_e2
    det["e"]["all_county_and_totals_violations"] = bad_e2
    # (f) cell sd finite iff a contributing component exists (all-county class arrays)
    bad_f2 = 0
    for arm, arr in (("recorded", rec), ("chowlin", clz)):
        comp = np.isfinite(cz["sd"]) | (arr["n_imp"] > 0)
        if arm == "chowlin":
            comp |= arr["n_cl"] > 0
        bad_f2 += int((np.isfinite(arr["sd"]) != comp).sum())
        bad_f2 += int((~np.isin(arr["flag"], [0, 3, 4, 8, 9])).sum())
        bad_f2 += int((np.isnan(arr["value"]) != (arr["flag"] == 8)).sum())
    v["f"] += bad_f2
    det["f"]["cell_violations"] = bad_f2
    # (g) privacy scan of every staging output
    raw = det["g"].pop("_raw")
    hits_col, hits_val = [], 0
    for p in sorted(d.iterdir()):
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            hits_col += [f"{p.name}:{c}" for c in df.columns if c in FORBIDDEN]
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype).startswith(("int", "string")):
                    hits_val += int(df[c].astype(str).isin(raw).sum())
        elif p.suffix == ".npz":
            zz = np.load(p, allow_pickle=False)
            hits_col += [f"{p.name}:{k}" for k in zz.files if k in FORBIDDEN]
            for k in zz.files:
                if zz[k].dtype.kind in "UO" and zz[k].size < 5_000_000:
                    vals = pd.Series(zz[k].ravel().astype(str))
                    vals = vals.str.split(":").str[-1]
                    hits_val += int(vals.isin(raw).sum())
    v["g"] = len(hits_col) + hits_val
    det["g"] = {"forbidden_columns": hits_col, "raw_meter_number_hits": hits_val,
                "raw_meter_numbers_scanned": len(raw),
                "files": sorted(p.name for p in d.iterdir())}
    # (h) pre-origin manifest inputs never list pump_electricity/full
    ins = fin.inputs_for(variant)
    if variant == "preorigin":
        bad_h = [x for x in ins if "pump_electricity/full" in x or "_full_" in Path(x).name
                 or Path(x).name.startswith("full_")]
    else:
        bad_h = []
    v["h"] = len(bad_h)
    det["h"] = {"inputs": ins, "post_origin_inputs": bad_h}
    # (i) zero table rows are 0 readings (or out-of-span missing) of kept meters in window
    zt = pd.read_parquet(d / "zero_class.parquet")
    w0, w1 = (L.tidx(m) for m in fin.ZERO_WINDOW[variant])
    bad_i = 0
    for c in L.COUNTIES:
        cd = L.CountyData(variant, c, prod=prod)
        sub = zt[zt["county"] == c]
        r = cd.mpos.reindex(sub["meter_idx"].to_numpy()).to_numpy()
        t = np.array([L.tidx(m) for m in sub["month"]]) if len(sub) else np.zeros(0, int)
        missing_meter = np.isnan(r.astype(float))
        bad_i += int(missing_meter.sum())
        r = r[~missing_meter].astype(int)
        t = t[~missing_meter]
        x = cd.M[r, t]
        f = cd.F[r, t]
        bad_i += int((~(((x == 0) | np.isnan(x)) & (f != 4) & (f != 9))).sum())
        bad_i += int(((t < w0) | (t > w1)).sum())
    v["i"] = bad_i
    det["i"] = {"rows": int(len(zt)), "violations": bad_i}
    res["violations"] = v
    res["details"] = det
    res["passed"] = bool(all(x == 0 for x in v.values()))
    res["seconds"] = round(time.time() - t0, 1)
    tag = "_repair" if repair else ""
    L.save_json(res, out_dir / f"t5_construction_{variant}{tag}.json")
    print(variant, v, "passed", res["passed"])
    return res


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]), "--repair" in sys.argv[3:])
