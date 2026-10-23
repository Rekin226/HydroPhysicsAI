"""Registered tests P2r3b.k1.construction_<variant> (pre-registration P2abc.r3.repair.v2):
construction checks of a repair-round assembled product (staging directory
``_work_p2abc_v3/assembled_<variant>``) against the C0 v3 inputs, the round-2 stage-1 arrays
and the shipped zero classifier.

Items (a)-(h) restate round 2's (P2r2.k1r) for the repair-round product; (i) is sharpened;
(j)-(l) are new.

(a) every Chow-Lin re-split pair keeps its C0 bill sum; method chowlin_split covers both months
    of a pair;
(b) the Chow-Lin arm per pump equals C0 (value, NaN, flag, method) wherever its method is not
    chowlin_split, own_3yr_same_month_mean or m0_july2025_suspect; chowlin_split only on
    stage-1 re-split months; the imputation only on C0's open July-2026 phase-A months (full);
(c) C0 pre-install months are 0, flag 0, method pre_install;
(d) the recorded arm's cell arrays equal C0's cells_by_class (value everywhere except cell-
    months holding an imputed pump; flag also except 2025-07 cells whose M0-suspect HP exceeds
    10 % of the installed HP, which must carry flag 9 unless C0's flag is 8);
(e) cells = sum of pumps (Chow-Lin arm, per county), all-county = sum of counties, totals =
    sum of classes, no-domestic = all - domestic (every arm);
(f) pump sd finite iff flag 3 or 4; flags in {0, 3, 4, 8, 9}; NaN iff flag 8 (pumps and every
    arm's cells); a cell sd finite iff a contributing component exists (C0 sd, imputed pump,
    flag-3 pump in Chow-Lin arms, filled pump in members);
(g) privacy: no personal column, raw meter number or coordinate in any output;
(h) the pre-origin manifest lists no input of pump_electricity/full or a full-variant stage
    file, and every input carrying post-origin electricity is named by an exemption entry;
(i) zero table: every row is a 0 reading (flag 0, not the first month of a bill posted to its
    second month) of a kept meter inside the window; re-running
    ``zeros3.classify`` on the C0 inputs reproduces zero_class.parquet and
    meter_zero_summary.parquet exactly; structural == C0 pre_install at that meter-month;
    sync-suspect rows carry flag 9 and are never structural; zclass consistent with zkind and
    run_len;
(j) M0 July 2025 (full): exactly the kept code-M0 monthly meters with an installed (not
    pre_install) flag-0 2025-07 reading carry flag 9 and method m0_july2025_suspect at 2025-07
    (value equal to C0); no such method anywhere else (none in the pre-origin product); the
    expected set is computed in this script (P2abc.r3.k1_repair.v1; the v2 wording "with a
    flag-0 2025-07 reading" included a pre-install month, contradicting item (c));
(k) members: rebuilt per pump from C0, the Chow-Lin arm pumps, sync_fill.parquet and
    july2026_imputation.parquet, every member cell (value, by county and all counties) equals
    the published one; fills sit only on sync-suspect zero rows, are finite and >= 0;
(l) incomplete_pair_mask lists every sync-suspect zero month (reason sync_zero_month) and, full,
    every M0-suspect meter at 2025-07.

Usage: python -I t7_construction3.py <variant> <out_dir> [--repair]
       (--repair: the checks as restated in P2abc.r3.k1_repair.v1, test id
       P2r3c.k1.construction_<variant>; output t7_construction3_<variant>_repair.json)
       python -I t7_construction3.py <variant> <out_dir> <staging_dir> --no-auth <county,..>
(the optional staging dir and --no-auth are for debug runs on scratch subsets only)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import b5_assemble3 as B5  # noqa: E402
import p3lib as P3  # noqa: E402
import scoring  # noqa: E402
import t5_construction as T5  # noqa: E402  (v2: close, FORBIDDEN, L0P, METER_COL)
import zeros3  # noqa: E402
from p3lib import L  # noqa: E402

PREREG3 = "P2abc.r3.repair.v2"
close = T5.close
TOL_ABS, TOL_REL = T5.TOL_ABS, T5.TOL_REL


def run(variant: str, out_dir: Path, staging: Path | None = None, auth: bool = True,
        counties=None, repair: bool = False) -> dict:
    t0 = time.time()
    res = {"variant": variant, "items": {},
           "code_sha256": P3.code_sha(Path(__file__), P3.V3 / "b5_assemble3.py",
                                      P3.V3 / "zeros3.py", P3.V3 / "b6_finalize3.py",
                                      L.CODE_DIR / "p2lib.py", L.CODE_DIR / "b3_assemble.py")}
    res["repair"] = repair
    if auth and repair:
        res["authorization"] = scoring.authorize([f"P2r3c.k1.construction_{variant}"],
                                                 prereg_id="P2abc.r3.k1_repair.v1")
    elif auth:
        res["authorization"] = scoring.authorize([f"P2r3b.k1.construction_{variant}"],
                                                 prereg_id=PREREG3)
    d = staging or (P3.WORK3 / f"assembled_{variant}")
    counties = tuple(counties or L.COUNTIES)
    prod = L.open_c0(variant)
    T = L.VARIANTS[variant]["T"]
    t26 = L.tidx("2026-07") if variant == "full" else None
    t25 = L.tidx("2025-07") if variant == "full" else None
    items = "abcdefghijkl"
    v = {k: 0 for k in items}
    det = {k: {} for k in items}
    cz = np.load(prod.path("cells_by_class.npz"))
    ct = np.load(prod.path("cells_total.npz"))
    arrs = {arm: np.load(d / f"kwh_by_class_{arm}.npz") for arm in B5.ARMS}
    rec, clz = arrs["recorded"], arrs["chowlin"]
    A = int(cz["n_active"])
    C = len(L.CLASSES)
    zt = pd.read_parquet(d / "zero_class.parquet")
    zs = pd.read_parquet(d / "meter_zero_summary.parquet")
    fill = pd.read_parquet(d / "sync_fill.parquet")
    mask = pd.read_parquet(d / "incomplete_pair_mask.parquet")
    july = pd.read_parquet(d / "july2026_imputation.parquet") if variant == "full" else None
    bi = L.load_json(d / "build_info.json")
    w0, w1 = (L.tidx(m) for m in zeros3.WINDOW[variant])
    # member cell rebuild accumulators (per arm base, value by county)
    rebuilt = {arm: np.zeros((len(L.COUNTIES), C * A, T)) for arm in B5.ARMS}
    susp_by_county_hp, inst_by_county_hp, sus_by_county = {}, {}, {}
    raw_meters = set()
    zre_all, sre_all = [], []
    cds_screen = {}
    for ki, c in enumerate(L.COUNTIES):
        if c not in counties:
            continue
        cd = L.CountyData(variant, c, prod=prod)
        z = np.load(d / f"pumps_{L.FILETAG[c]}.npz", allow_pickle=False)
        val, flg, met, sd = z["value"], z["flag"], z["method"], z["sd"]
        assert np.array_equal(z["uid"], cd.uid)
        if variant == "full":     # computed here, independently of b5 (repair k1_repair.v1)
            m0r = np.where((cd.code == "M0") & (cd.phase == "monthly") & (cd.F[:, t25] == 0)
                           & (cd.Mm[:, t25] != 1))[0]
        else:
            m0r = np.zeros(0, int)
        m0p = np.isin(cd.prow, m0r)
        # (b)
        new = np.isin(met, [L.M_CHOWLIN, L.M_JULY_IMPUTED, B5.M_M0SUSPECT])
        same_v = close(val, cd.pvalue)
        bad_b = ~new & ~(same_v & (flg == cd.pflag) & (met == cd.pmethod))
        st1 = np.load(L.WORK / f"{variant}_{c}_cl.npz")
        rs = np.zeros(cd.M.shape, bool)
        rs[st1["rows"]] = st1["resplit"]
        bad_m10 = (met == L.M_CHOWLIN) & ~rs[cd.prow]
        if variant == "full":
            bad_m11 = (met == L.M_JULY_IMPUTED) & ~((np.arange(T)[None, :] == t26)
                                                     & (cd.pmethod == 4))
        else:
            bad_m11 = met == L.M_JULY_IMPUTED
        bad_m12v = (met == B5.M_M0SUSPECT) & ~(same_v & (flg == 9) & (cd.pflag == 0))
        v["b"] += int(bad_b.sum() + bad_m10.sum() + bad_m11.sum() + bad_m12v.sum())
        det["b"][c] = {"diff_outside_new": int(bad_b.sum()),
                       "chowlin_split_outside_resplit": int(bad_m10.sum()),
                       "imputation_outside_open_july": int(bad_m11.sum()),
                       "m0_suspect_value_or_flag": int(bad_m12v.sum())}
        # (j)
        exp12 = np.zeros(met.shape, bool)
        if variant == "full":
            exp12[m0p, t25] = True
        bad_j = int(((met == B5.M_M0SUSPECT) != exp12).sum())
        v["j"] += bad_j
        det["j"][c] = {"violations": bad_j, "m0_suspect_meters": int(len(m0r)),
                       "m0_suspect_pumps": int(m0p.sum())}
        # (a)
        fp = cd.first_array_pump
        Mv = val[fp] / cd.share[fp, None]
        Mmeth = met[fp]
        mt = pd.read_parquet(L.WORK / f"{variant}_{c}_meters.parquet")
        cph = mt["chowlin_phase"].to_numpy()
        bad_a = bad_pair = 0
        for p_ in ("A", "B"):
            rows = np.where(cph == p_)[0]
            st = L.pair_starts(p_, 0, T, T)
            m10a = Mmeth[np.ix_(rows, st)] == L.M_CHOWLIN
            m10b = Mmeth[np.ix_(rows, st + 1)] == L.M_CHOWLIN
            bad_pair += int((m10a ^ m10b).sum())
            s_new = Mv[np.ix_(rows, st)] + Mv[np.ix_(rows, st + 1)]
            s_old = cd.M[np.ix_(rows, st)] + cd.M[np.ix_(rows, st + 1)]
            okd = np.abs(s_new - s_old) <= TOL_ABS + TOL_REL * np.abs(s_old)
            rsp = rs[np.ix_(rows, st)] & rs[np.ix_(rows, st + 1)]
            bad_a += int(((m10a & m10b) & ~okd).sum()) + int((rsp & ~okd).sum())
        v["a"] += bad_a + bad_pair
        det["a"][c] = {"pair_sum_violations": bad_a, "unpaired_chowlin_months": bad_pair}
        # (c)
        pre = cd.pmethod == 1
        bad_c = int((pre & ~((val == 0) & (flg == 0) & (met == 1))).sum())
        v["c"] += bad_c
        det["c"][c] = {"pre_install_pump_months": int(pre.sum()), "violations": bad_c}
        # (e) cells = sum of pumps (Chow-Lin arm)
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
        # (f) pumps
        bad_f = int((np.isfinite(sd) != np.isin(flg, [3, 4])).sum())
        bad_f += int((~np.isin(flg, [0, 3, 4, 8, 9])).sum())
        bad_f += int((np.isnan(val) != (flg == 8)).sum())
        v["f"] += bad_f
        det["f"][c] = {"pump_violations": bad_f}
        # (d) helpers: M0-suspect and installed HP per class x cell at 2025-07
        if variant == "full":
            sh_ = np.zeros(C * A)
            ih_ = np.zeros(C * A)
            np.add.at(sh_, g[ok & m0p], cd.hp[ok & m0p])
            inst = ok & (cd.pmethod[:, t25] != 1)
            np.add.at(ih_, g[inst], cd.hp[inst])
            susp_by_county_hp[ki], inst_by_county_hp[ki] = sh_, ih_
            sus_by_county[ki] = (ih_ > 0) & (sh_ > 0.10 * ih_)
        # (k) member rebuild: pump values per base arm, then fills and member July values
        recp = cd.pvalue.astype(np.float64).copy()
        if variant == "full":
            js = july[july["county"] == c]
            jr = cd.mpos.reindex(js["meter_idx"].to_numpy()).to_numpy().astype(int)
            for col, arr in (("imputed_recorded_kwh", recp),):
                okj = js["shipped"].to_numpy() & np.isfinite(js[col].to_numpy())
                vals_m = np.full(len(cd.meters), np.nan)
                vals_m[jr[okj]] = js[col].to_numpy()[okj]
                pm = np.isfinite(vals_m[cd.prow])
                arr[pm, t26] = vals_m[cd.prow][pm] * cd.share[pm]
        base = {"recorded": recp, "chowlin": val.astype(np.float64)}
        fs = fill[fill["county"] == c]
        fr = cd.mpos.reindex(fs["meter_idx"].to_numpy()).to_numpy().astype(int)
        ftt = np.array([L.tidx(m) for m in fs["month"]], int) if len(fs) else np.zeros(0, int)
        sz = zt[(zt["county"] == c) & zt["sync_suspect"]]
        zr = cd.mpos.reindex(sz["meter_idx"].to_numpy()).to_numpy().astype(int)
        ztt = np.array([L.tidx(m) for m in sz["month"]], int) if len(sz) else np.zeros(0, int)
        skey = set((zr.astype(np.int64) * T + ztt).tolist())
        fkey = (fr.astype(np.int64) * T + ftt).tolist()
        bad_k = int(sum(k not in skey for k in fkey))
        for arm in B5.ARMS:
            bname = B5.BASE[arm]
            pv = base[bname].copy()
            if arm.endswith("_syncfill"):
                col = f"fill_{bname}_kwh"
                fv = fs[col].to_numpy()
                okf = np.isfinite(fv)
                bad_k += int((fv[okf] < 0).sum())
                Fm = np.zeros(cd.M.shape)
                Fm[fr[okf], ftt[okf]] = fv[okf]
                has = np.zeros(cd.M.shape, bool)
                has[fr[okf], ftt[okf]] = True
                hp_ = has[cd.prow]
                pv[hp_] = (Fm[cd.prow] * cd.share[:, None])[hp_]
                if variant == "full":
                    jc = f"imputed_{arm}_kwh"
                    okj = js["shipped"].to_numpy() & np.isfinite(js[jc].to_numpy())
                    vals_m = np.full(len(cd.meters), np.nan)
                    vals_m[jr[okj]] = js[jc].to_numpy()[okj]
                    pm = np.isfinite(vals_m[cd.prow])
                    pv[pm, t26] = vals_m[cd.prow][pm] * cd.share[pm]
            Sx = np.zeros((C * A, T))
            np.add.at(Sx, g[ok], np.nan_to_num(pv[ok]))
            rebuilt[arm][ki] = Sx
        v["k"] += bad_k
        det["k"][c] = {"fill_rows_not_sync_or_negative": bad_k, "fill_rows": int(len(fs))}
        # (l)
        mk = mask[mask["county"] == c]
        mset = set(zip(mk.loc[mk["reason"] == "sync_zero_month", "meter_idx"],
                       mk.loc[mk["reason"] == "sync_zero_month", "month"], strict=True))
        bad_l = int(sum((a_, b_) not in mset
                        for a_, b_ in zip(sz["meter_idx"], sz["month"], strict=True)))
        if variant == "full":
            m0set = set(mk.loc[mk["reason"] == "m0_july2025_suspect", "meter_idx"])
            bad_l += len(set(cd.meters["meter_idx"].to_numpy()[m0r]) ^ m0set)
        v["l"] += bad_l
        det["l"][c] = {"violations": bad_l}
        # (g) raw meter numbers
        raw = pd.read_parquet(T5.L0P / f"{c}_census.parquet", columns=[T5.METER_COL[c]])
        raw_meters.update(raw[T5.METER_COL[c]].dropna().astype(str))
        cds_screen[c] = cd
    kis_ = [L.COUNTIES.index(c) for c in counties]
    # (i) zero table re-run, row checks
    bad_i = 0
    for c in counties:
        cd = cds_screen[c]
        screen = None
        if c == "chiayi":
            ycd = cds_screen.get("yunlin") or L.CountyData(variant, "yunlin", prod=prod)
            a_, b_ = zeros3.screen_input3(ycd, variant), zeros3.screen_input3(cd, variant)
            screen = tuple(np.concatenate([x, y]) for x, y in zip(a_, b_, strict=True))
        zre, sre, _ = zeros3.classify(cd, variant, screen)
        zre_all.append(zre)
        sre_all.append(sre)
        sub = zt[zt["county"] == c]
        r = cd.mpos.reindex(sub["meter_idx"].to_numpy()).to_numpy()
        miss = np.isnan(r.astype(float))
        bad_i += int(miss.sum())
        r = r[~miss].astype(int)
        t = np.array([L.tidx(m) for m in sub["month"]])[~miss] if len(sub) else np.zeros(0, int)
        bad_i += int((~((cd.M[r, t] == 0) & (cd.F[r, t] == 0) & (cd.Mm[r, t] != 7))).sum())
        bad_i += int(((t < w0) | (t > w1)).sum())
        bad_i += int((sub["structural"].to_numpy()[~miss] != (cd.Mm[r, t] == 1)).sum())
    zre = pd.concat(zre_all, ignore_index=True)
    sre = pd.concat(sre_all, ignore_index=True)
    ztc = zt[zt["county"].isin(counties)].reset_index(drop=True)
    zsc = zs[zs["county"].isin(counties)].reset_index(drop=True)
    same_z = len(zre) == len(ztc) and all(
        np.array_equal(zre[k].astype(str).to_numpy(), ztc[k].astype(str).to_numpy())
        for k in ztc.columns)
    same_s = len(sre) == len(zsc) and all(
        np.array_equal(sre[k].astype(str).to_numpy(), zsc[k].astype(str).to_numpy())
        for k in zsc.columns)
    bad_i += int(not same_z) + int(not same_s)
    bad_i += int((ztc["sync_suspect"] & ((ztc["flag"] != 9) | ztc["structural"])).sum())
    bad_i += int((~ztc["sync_suspect"] & (ztc["flag"] != 0)).sum())
    exp_cls = zeros3.zclass_of(
        pd.Categorical(ztc["zkind"], categories=list(zeros3.KINDS)).codes.astype(np.int8),
        ztc["run_len"].to_numpy())
    bad_i += int((pd.Categorical(ztc["zclass"], categories=list(zeros3.ZCLASSES)).codes
                  != exp_cls).sum())
    v["i"] = bad_i
    det["i"] = {"rows": int(len(ztc)), "violations": bad_i, "rerun_identical_zero_table": same_z,
                "rerun_identical_meter_summary": same_s}
    # (d) recorded arm cells vs C0 (by county; all counties and totals when complete)
    full_set = counties == tuple(L.COUNTIES)
    excl = rec["n_imp"].reshape(C * A, T) > 0
    sets = [(f"county:{L.COUNTIES[ki]}", rec["value_by_county"][ki].reshape(C * A, T),
             rec["flag_by_county"][ki].reshape(C * A, T),
             cz["value_by_county"][ki].reshape(C * A, T),
             cz["flag_by_county"][ki].reshape(C * A, T),
             sus_by_county.get(ki)) for ki in kis_]
    if full_set:
        sus_all = None
        if variant == "full":
            si = sum(susp_by_county_hp.values())
            ii = sum(inst_by_county_hp.values())
            sus_all = (ii > 0) & (si > 0.10 * ii)
        sets.append(("all", rec["value"].reshape(C * A, T), rec["flag"].reshape(C * A, T),
                     cz["value"].reshape(C * A, T), cz["flag"].reshape(C * A, T), sus_all))
    bad_d = 0
    for name, rv, rf, cv, cf, sus in sets:
        flag_ex = excl.copy()
        if variant == "full":
            flag_ex[:, t25] |= sus
            bad_sus = int((sus & (cf[:, t25] != 8) & (rf[:, t25] != 9)).sum())
            det["d"][f"suspect_cells_2025_07|{name}"] = int(sus.sum())
            bad_d += bad_sus
        bad_d += int(((~close(rv, cv)) & ~excl).sum()) + int(((rf != cf) & ~flag_ex).sum())
    if full_set:
        exa = excl.reshape(C, A, T).any(0)
        bad_d += int(((~close(rec["value_total_all"], ct["value_all"])) & ~exa).sum())
        bad_d += int(((~close(rec["value_total_nodomestic"], ct["value_nodomestic"]))
                      & ~exa).sum())
    v["d"] = bad_d
    det["d"]["violations"] = bad_d
    # (e) all-county / totals / no-domestic, every arm; (k) member rebuild; (f) cells
    bad_e2 = bad_k2 = bad_f2 = 0
    dom = L.CLASSES.index("domestic")
    kis = [L.COUNTIES.index(c) for c in counties]
    for arm, arr in arrs.items():
        f_ = arr["flag"]
        m_ = f_ != 8
        s_ = np.nansum(arr["value_by_county"], 0)
        bad_e2 += int((~close(arr["value"][m_], s_[m_])).sum())
        ft = arr["flag_total_all"]
        tot = np.nansum(arr["value_by_county"], (0, 1))
        bad_e2 += int((~close(arr["value_total_all"][ft != 8], tot[ft != 8])).sum())
        nd = np.nansum(np.delete(arr["value_by_county"], dom, axis=1), (0, 1))
        fn = arr["flag_total_nodomestic"]
        bad_e2 += int((~close(arr["value_total_nodomestic"][fn != 8], nd[fn != 8])).sum())
        for ki in kis:
            pub = arr["value_by_county"][ki].reshape(C * A, T)
            pf = arr["flag_by_county"][ki].reshape(C * A, T)
            mm = pf != 8
            bad_k2 += int((~close(pub[mm], rebuilt[arm][ki][mm])).sum())
        comp = np.isfinite(cz["sd"]) | (arr["n_imp"] > 0) | (arr["n_sync"] > 0)
        if B5.BASE[arm] == "chowlin":
            comp |= arr["n_cl"] > 0
        bad_f2 += int((np.isfinite(arr["sd"]) != comp).sum())
        bad_f2 += int((~np.isin(arr["flag"], [0, 3, 4, 8, 9])).sum())
        bad_f2 += int((np.isnan(arr["value"]) != (arr["flag"] == 8)).sum())
        if not arm.endswith("_syncfill"):
            bad_k2 += int((arr["n_sync"] > 0).sum())
        else:
            bad_k2 += int(((arr["n_sync"] > 0) & (arr["flag"] < 4)).sum())
    v["e"] += bad_e2
    det["e"]["all_county_and_totals_violations"] = bad_e2
    v["k"] += bad_k2
    det["k"]["member_rebuild_or_flag_violations"] = bad_k2
    v["f"] += bad_f2
    det["f"]["cell_violations"] = bad_f2
    # (g) privacy scan
    hits_col, hits_val = [], 0
    for p in sorted(d.iterdir()):
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            hits_col += [f"{p.name}:{c}" for c in df.columns if c in T5.FORBIDDEN]
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype).startswith(("int", "string")):
                    hits_val += int(df[c].astype(str).isin(raw_meters).sum())
        elif p.suffix == ".npz":
            zz = np.load(p, allow_pickle=False)
            hits_col += [f"{p.name}:{k}" for k in zz.files if k in T5.FORBIDDEN]
            for k in zz.files:
                if zz[k].dtype.kind in "UO" and zz[k].size < 5_000_000:
                    vals = pd.Series(zz[k].ravel().astype(str)).str.split(":").str[-1]
                    hits_val += int(vals.isin(raw_meters).sum())
    v["g"] = len(hits_col) + hits_val
    det["g"] = {"forbidden_columns": hits_col, "raw_meter_number_hits": hits_val,
                "raw_meter_numbers_scanned": len(raw_meters),
                "files": sorted(p.name for p in d.iterdir())}
    # (h) manifest inputs of the pre-origin product
    import b6_finalize3 as F6
    ins = F6.inputs_for(variant)
    bad_h = []
    if variant == "preorigin":
        exempt = {p for x in F6.extra_exemptions_preorigin() for p in x["input_paths"]}
        for x in ins:
            post = ("pump_electricity/full" in x or Path(x).name.startswith("full_")
                    or "_full_" in Path(x).name or "t6_zero_shipped" in x
                    or "t4_zero_port" in x)
            if post and x not in exempt:
                bad_h.append(x)
    v["h"] = len(bad_h)
    det["h"] = {"inputs": ins, "post_origin_inputs_without_exemption": bad_h}
    res["violations"] = v
    res["details"] = det
    res["build_info_arms"] = bi.get("arms")
    res["passed"] = bool(all(x == 0 for x in v.values()))
    res["seconds"] = round(time.time() - t0, 1)
    tag = "_repair" if repair else ""
    L.save_json(res, out_dir / f"t7_construction3_{variant}{tag}.json")
    print(variant, v, "passed", res["passed"])
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 3:      # debug: <variant> <out_dir> <staging_dir> --no-auth <county,..>
        run(a[0], Path(a[1]), Path(a[2]), auth=False, counties=a[4].split(","))
    else:
        run(a[0], Path(a[1]), repair="--repair" in a[2:])
