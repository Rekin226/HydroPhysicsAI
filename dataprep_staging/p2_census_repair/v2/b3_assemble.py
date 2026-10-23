"""Assemble the round-2 P2a-c products (after the registered tests were scored) into a staging
directory ``_work_p2abc_v2/assembled_<variant>/``; ``b4_finalize.py`` copies a staging
directory that passed its registered construction checks into
``results/dataprep/yunlin_electricity/<variant>/v<N>/`` and writes the manifest.

Inputs: the C0 v3 product of the variant, the stage-1 arrays (b1_build.py), the registered
results t1_meter_validation_<variant>.json (shipping verdicts, sd models), t3_july_backtest.json
(July-2026 verdicts and imputation sd), t4_zero_port.json, and the zero classification
(b2_zeros.py).

Two arms on the C0 grid and time axis:

* ``recorded``: C0's values as read (TPC even split of bimonthly bills), plus, in the full
  variant, the P2b imputation of the open July-2026 phase-A month (flag 4) where shipped;
* ``chowlin``: the recorded arm with every eligible bimonthly irrigation bill re-split by the
  county's Chow-Lin operator (flag 3, method chowlin_split) where shipped, and the July-2026
  imputation from the Chow-Lin July values (full).

sd (registered rule): flag 3 -> q90(n_eff, month)/1.645 x even-split energy of the Chow-Lin
component (q90 from the variant's validation groups of the county; Chiayi: Yunlin's); flag-4
imputation -> q90_rel/1.645 x imputed energy (meter q90 for pumps, cell q90 for cells, the
fan-total q90 of the backtest years for county scopes); C0's December-2022 fill keeps C0's sd;
components add in quadrature; an aggregate holding a flag-3 or imputed pump has a defined sd
even when that pump's energy is 0 (repair after the first registered construction check).
Aggregate flags: 8 > 9 > 4 > 3 > 0 (C0's rule plus flag 3); a flag-8 aggregate's value is NaN
(as in C0).

Usage: python -I b3_assemble.py <variant> <tests_dir> <zeros_dir>
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402

C = len(L.CLASSES)
K = len(L.COUNTIES)
DOM = L.CLASSES.index("domestic")
CLS_IDX = {c: i for i, c in enumerate(L.CLASSES)}
SIZES = np.array([1, 3, 10, 30, 100, 300], float)
UNIT = "kWh/month, model units (cleaned-census kWh)"
FLAG_METHOD = {0: "observed", 3: "with_chowlin_split", 4: "with_fill_or_imputation",
               8: "incomplete_billing_pair", 9: "observed_suspect"}


# ----------------------------------------------------------------------------- helpers
def q90_interp(q90_by_n_moy: dict, n_eff: np.ndarray, moy: np.ndarray) -> np.ndarray:
    """q90 interpolated linearly in log(n_eff) between the fitted sizes (clamped)."""
    Q = np.array([[q90_by_n_moy[str(int(s))][str(m)] if str(m) in q90_by_n_moy[str(int(s))]
                   else q90_by_n_moy[str(int(s))][m] for m in range(1, 13)] for s in SIZES])
    ln = np.log(np.clip(n_eff, 1.0, 300.0))
    out = np.full(n_eff.shape, np.nan)
    lx = np.log(SIZES)
    for m in range(1, 13):
        sel = moy == m
        if sel.any():
            out[sel] = np.interp(ln[sel], lx, Q[:, m - 1])
    return out


def group_matrix(g: np.ndarray, G: int) -> sparse.csr_matrix:
    ok = g >= 0
    P = len(g)
    return sparse.csr_matrix((np.ones(ok.sum()), (g[ok], np.where(ok)[0])), shape=(G, P))


def gm_matrices(g: np.ndarray, meter: np.ndarray, G: int):
    """(B: (n_gm x P) pump -> (group, meter) pair, G2: (G x n_gm) pair -> group)."""
    ok = np.where(g >= 0)[0]
    nm = int(meter.max()) + 1
    key = g[ok].astype(np.int64) * nm + meter[ok]
    uniq, codes = np.unique(key, return_inverse=True)
    n = len(uniq)
    B = sparse.csr_matrix((np.ones(len(ok)), (codes, ok)), shape=(n, len(g)))
    G2 = sparse.csr_matrix((np.ones(n), (uniq // nm, np.arange(n))), shape=(G, n))
    return B, G2


def gkey(key: str, cls: np.ndarray, gc: np.ndarray, A: int) -> np.ndarray:
    """Group index of each pump: class x cell ('cls'), cell over all classes ('all') or over
    the non-domestic classes ('nodom'); -1 outside the grid."""
    if key == "cls":
        return gc
    if key == "all":
        return np.where(gc >= 0, gc % A, -1)
    return np.where((gc >= 0) & (cls != DOM), gc % A, -1)


def flags_from_parts(p: dict) -> np.ndarray:
    f = np.zeros(p["value"].shape, dtype=np.uint8)
    f[p["n_cl"] > 0] = 3
    f[(p["n_fill"] + p["n_imp"]) > 0] = 4
    hpi, hpm = p["hp_installed"], p["hp_missing"]
    f[(hpi > 0) & (hpm > 0.10 * hpi)] = 9
    f[p["n_partial"] > 0] = 9
    f[p["n_incomplete"] > 0] = 8
    return f


# ----------------------------------------------------------------------------- pump arms
def pump_arms(cd: L.CountyData, variant: str, ship_cl: bool, sdm: dict, july: dict | None):
    """Per-pump recorded and Chow-Lin arms: value, flag, method, sd, e_cl (even-split energy
    of flag-3 pump-months), imp (imputed energy of flag-4 imputations)."""
    T = cd.T
    moy = L.DATES[:T].month.to_numpy()
    M, F, Mm = cd.M.astype(np.float64), cd.F, cd.Mm
    arms = {}
    meter_tab = pd.read_parquet(L.WORK / f"{variant}_{cd.county}_meters.parquet")
    for arm in ("recorded", "chowlin"):
        V, Fl, Me = M.copy(), F.copy(), Mm.copy()
        E = np.zeros_like(V)
        IMP = np.zeros_like(V)
        if arm == "chowlin" and ship_cl:
            z = np.load(L.WORK / f"{variant}_{cd.county}_cl.npz")
            rows, vals, rs = z["rows"], z["values"], z["resplit"]
            ph = meter_tab["chowlin_phase"].to_numpy()[rows]
            orig = M[rows]
            tt = np.arange(T)[None, :]
            s0 = np.where(ph == "A", 0, 1)[:, None]
            start = np.where((tt - s0) % 2 == 0, tt, tt - 1)
            start = np.clip(start, 0, T - 2)
            r_i = np.arange(len(rows))[:, None]
            even = (orig[r_i, start] + orig[r_i, start + 1]) / 2
            pos = rs & (even > 0)
            sub = V[rows]
            sub[rs] = vals[rs]
            V[rows] = sub
            fsub, msub = Fl[rows], Me[rows]
            fsub[pos] = 3
            msub[pos] = L.M_CHOWLIN
            Fl[rows], Me[rows] = fsub, msub
            esub = np.zeros_like(sub)
            esub[pos] = even[pos]
            E[rows] = esub
        if july is not None and july["ship"]:
            t26 = L.tidx("2026-07")
            r = july["rows"]
            val = july["imputed_" + arm]
            ok = np.isfinite(val)
            V[r[ok], t26] = val[ok]
            Fl[r[ok], t26] = 4
            Me[r[ok], t26] = L.M_JULY_IMPUTED
            IMP[r[ok], t26] = val[ok]
        # pump level
        sh = cd.share[:, None]
        pv = V[cd.prow] * sh
        pf, pm = Fl[cd.prow], Me[cd.prow]
        pe = E[cd.prow] * sh
        pimp = IMP[cd.prow] * sh
        sd = np.full(pv.shape, np.nan)
        if arm == "chowlin" and (pe > 0).any():
            q1 = np.array([sdm["q90_by_n_moy"]["1"][str(m)] for m in range(1, 13)])
            qq = q1[moy - 1][None, :] / L.Z90
            sel = pf == 3
            sd[sel] = (qq * np.where(sel, pe, 0.0))[sel]
        c0fill = pm == 9
        sd[c0fill] = cd.psd[c0fill]
        if july is not None and july["ship"]:
            qm = july["q_meter_" + arm] / L.Z90
            sel = pm == L.M_JULY_IMPUTED
            sd[sel] = qm * pimp[sel]
        assert np.array_equal(np.isnan(pv), pf == 8), "NaN <=> flag 8 (pump level)"
        arms[arm] = {"value": pv, "flag": pf, "method": pm, "sd": sd, "e_cl": pe,
                     "imp": pimp}
    return arms


# ----------------------------------------------------------------------------- aggregation
def aggregate(a: dict, cd: L.CountyData, g: np.ndarray, G: int, sdm: dict | None,
              q_imp: float | None, sd_c0: np.ndarray | None) -> dict:
    T = cd.T
    moy = L.DATES[:T].month.to_numpy()
    S = group_matrix(g, G)
    hp = cd.hp[:, None]
    meth, flag = a["method"], a["flag"]
    p = {"value": S @ np.nan_to_num(a["value"]),
         "n_cl": S @ (flag == 3).astype(float),
         "n_fill": S @ (meth == 9).astype(float),
         "n_imp": S @ (meth == L.M_JULY_IMPUTED).astype(float),
         "n_incomplete": S @ (meth == 4).astype(float),
         "n_partial": S @ (meth == 5).astype(float),
         "hp_installed": S @ (hp * (meth != 1)),
         "hp_missing": S @ (hp * (meth == 3))}
    f = flags_from_parts(p)
    val = p["value"].copy()
    val[f == 8] = np.nan
    var = np.zeros(val.shape)
    has = np.zeros(val.shape, bool)
    n_eff = np.full(val.shape, np.nan)
    # a contributing flag-3 or imputed pump makes the sd defined, even when its energy is 0
    # (zero HP share, zero imputation): its component is then 0 (repair of P2r2.k1 (f))
    has |= (p["n_cl"] > 0) | (p["n_imp"] > 0)
    if sdm is not None and (a["e_cl"] > 0).any():
        B, G2 = gm_matrices(g, cd.prow, G)
        U = B @ a["e_cl"]
        Esum = G2 @ U
        S2 = G2 @ (U ** 2)
        okc = Esum > 0
        n_eff[okc] = Esum[okc] ** 2 / S2[okc]
        q = q90_interp(sdm["q90_by_n_moy"], np.where(okc, n_eff, 1.0),
                       np.broadcast_to(moy, val.shape))
        sd_cl = q / L.Z90 * Esum
        var[okc] += sd_cl[okc] ** 2
        has |= okc
    if q_imp is not None and (a["imp"] > 0).any():
        I_ = S @ a["imp"]
        oki = I_ > 0
        var[oki] += (q_imp / L.Z90 * I_[oki]) ** 2
        has |= oki
    if sd_c0 is not None:
        okz = np.isfinite(sd_c0)
        var[okz] += sd_c0[okz] ** 2
        has |= okz
    sd = np.where(has, np.sqrt(var), np.nan)
    return {"value": val, "flag": f, "sd": sd, "n_eff": n_eff, "n_cl": p["n_cl"],
            "n_imp": p["n_imp"]}


# ----------------------------------------------------------------------------- main
def main(variant: str, tests_dir: str, zeros_dir: str) -> None:
    t0 = time.time()
    td, zd = Path(tests_dir), Path(zeros_dir)
    out = L.WORK / f"assembled_{variant}"
    if out.exists():
        shutil.rmtree(out)                      # staging only, never a product version
    out.mkdir(parents=True)
    t1 = L.load_json(td / f"t1_meter_validation_{variant}.json")
    t1pre = L.load_json(td / "t1_meter_validation_preorigin.json")
    t3 = L.load_json(td / "t3_july_backtest.json") if variant == "full" else None
    prod = L.open_c0(variant)
    cz = np.load(prod.path("cells_by_class.npz"))
    ct = np.load(prod.path("cells_total.npz"))
    c0long = pd.read_parquet(prod.path("county_class_monthly.parquet"))
    A = int(cz["n_active"])
    T = L.VARIANTS[variant]["T"]
    dates = np.array([L.ym(i) for i in range(T)])
    # registered shipping rules (P2r2.a1 pre-origin verdict; Chiayi follows Yunlin)
    ship_cl = {c: bool(t1pre["counties"][c]["passed"]) for c in ("yunlin", "changhua")}
    ship_cl["chiayi"] = ship_cl["yunlin"]
    sdm = {c: t1["counties"][L.PROFILE_OF[c]]["sd_model"] for c in L.COUNTIES}
    sd_label = {c: t1["counties"][L.PROFILE_OF[c]]["sd_coverage"]["label"]
                for c in L.COUNTIES}
    info = {"variant": variant, "ship_chowlin": ship_cl, "sd_label_chowlin": sd_label,
            "counties": {}}
    arrays = {arm: {"value": np.zeros((C, A, T)), "flag": np.zeros((C, A, T), np.uint8),
                    "sd": np.full((C, A, T), np.nan), "n_eff": np.full((C, A, T), np.nan),
                    "n_cl": np.zeros((C, A, T)), "n_imp": np.zeros((C, A, T)),
                    "value_by_county": np.zeros((K, C, A, T)),
                    "flag_by_county": np.zeros((K, C, A, T), np.uint8),
                    "sd_by_county": np.full((K, C, A, T), np.nan)}
              for arm in ("recorded", "chowlin")}
    long_rows = []
    meters_out = []
    mask_rows = []
    july_rows = []
    county_pumps = {}
    for ki, c in enumerate(L.COUNTIES):
        cd = L.CountyData(variant, c, prod=prod)
        july = None
        q_imp = {}
        if variant == "full":
            jz = np.load(L.WORK / f"full_{c}_july.npz")
            tr = t3["counties"][c]
            july = {"ship": bool(tr["passed"]), "rows": jz["rows"],
                    "imputed_recorded": jz["imputed_recorded"],
                    "imputed_chowlin": jz["imputed_chowlin"],
                    "q_meter_recorded": tr["recorded_pair_mean"]["imputation_sd"]["meter"][
                        "q90_rel"],
                    "q_meter_chowlin": tr["chowlin_arm"]["imputation_sd"]["meter"]["q90_rel"]}
            for arm, key in (("recorded", "recorded_pair_mean"), ("chowlin", "chowlin_arm")):
                bt = tr[key]
                errs = np.array([r["total_err_pct"] for r in bt["per_year"]])
                fan = np.abs(errs) / (100.0 + errs)        # |est - truth| / est
                q_imp[arm] = {"cell": bt["imputation_sd"]["cell"]["q90_rel"],
                              "fan": float(np.quantile(fan, 0.90))}
            july_rows.append(pd.DataFrame({
                "county": c, "meter_idx": cd.meters["meter_idx"].to_numpy()[jz["rows"]],
                "imputed_recorded_kwh": jz["imputed_recorded"],
                "imputed_chowlin_kwh": jz["imputed_chowlin"],
                "sd_recorded_kwh": july["q_meter_recorded"] / L.Z90 * jz["imputed_recorded"],
                "sd_chowlin_kwh": july["q_meter_chowlin"] / L.Z90 * jz["imputed_chowlin"],
                "shipped": july["ship"] & np.isfinite(jz["imputed_recorded"])}))
        arms = pump_arms(cd, variant, ship_cl[c], sdm[c], july)
        cls = np.array([CLS_IDX[x] for x in cd.pclass])
        g_cell = np.where(cd.cell >= 0, cls * A + cd.cell, -1)
        for arm, a in arms.items():
            sd_c0 = cz["sd_by_county"][ki].reshape(C * A, T)
            qi = q_imp.get(arm, {}).get("cell") if variant == "full" else None
            ag = aggregate(a, cd, g_cell, C * A, sdm[c] if arm == "chowlin" else None, qi,
                           sd_c0)
            R = arrays[arm]
            R["value_by_county"][ki] = ag["value"].reshape(C, A, T)
            R["flag_by_county"][ki] = ag["flag"].reshape(C, A, T)
            R["sd_by_county"][ki] = ag["sd"].reshape(C, A, T)
            # county long table: scopes in_grid / county; classes, all, all_nodomestic
            c0l = c0long[c0long["county"] == c]
            for scope in ("in_grid", "county"):
                base = (cd.cell >= 0) if scope == "in_grid" else np.ones(len(cd.cell), bool)
                for name, sel in [(L.CLASSES[i], cls == i) for i in range(C)] + \
                        [("all", np.ones(len(cls), bool)), ("all_nodomestic", cls != DOM)]:
                    g = np.where(base & sel, 0, -1)
                    sdz = c0l[(c0l["scope"] == scope) & (c0l["class"] == name)] \
                        .set_index("date").reindex(dates)["sd"].to_numpy()[None, :]
                    qf = q_imp.get(arm, {}).get("fan") if variant == "full" else None
                    r = aggregate(a, cd, g, 1, sdm[c] if arm == "chowlin" else None, qf, sdz)
                    long_rows.append(pd.DataFrame({
                        "county": c, "scope": scope, "class": name, "date": dates,
                        "arm": arm, "value": r["value"][0], "sd": r["sd"][0],
                        "flag": r["flag"][0].astype("int64"),
                        "method": [FLAG_METHOD[int(x)] for x in r["flag"][0]],
                        "n_chowlin_pumps": r["n_cl"][0], "n_imputed_pumps": r["n_imp"][0],
                        "unit": UNIT}))
        county_pumps[c] = (cd, arms, cls, g_cell)
        a = arms["chowlin"]
        np.savez_compressed(
            out / f"pumps_{L.FILETAG[c]}.npz", uid=cd.uid, meter_idx=cd.meter_idx,
            cell=cd.cell, purpose_class=cd.pclass, hp=cd.hp, hp_share=cd.share, dates=dates,
            value=a["value"], flag=a["flag"], method=a["method"], sd=a["sd"])
        mt = pd.read_parquet(L.WORK / f"{variant}_{c}_meters.parquet")
        mt["chowlin_shipped"] = ship_cl[c] & (mt["chowlin_phase"] != "")
        meters_out.append(mt)
        # incomplete-pair / open-month mask table (meter level)
        Fm, Mmm = cd.F, cd.Mm
        rows_ = []
        if variant == "preorigin":
            j = L.tidx("2022-12")
            pr = cd.meters["dec_jan_paired"].to_numpy()
            for reason, sel in (("open_bill_dec2022_phaseB_c0_fill", pr & (Mmm[:, j] == 9)),
                                ("open_bill_dec2022_phaseB_missing", pr & (Fm[:, j] == 8))):
                for r_ in np.where(sel)[0]:
                    rows_.append((r_, j, reason))
        else:
            j = L.tidx("2026-07")
            for reason, sel in (("open_bill_jul2026_phaseA", Mmm[:, j] == 4),
                                ("partial_cycle_jul2026_monthly", Mmm[:, j] == 5)):
                for r_ in np.where(sel)[0]:
                    rows_.append((r_, j, reason))
        ph = cd.phase
        for p_ in ("A", "B"):
            rr = np.where(ph == p_)[0]
            st = L.pair_starts(p_, 0, cd.T, cd.T)
            fa, fb = np.isfinite(cd.M[np.ix_(rr, st)]), np.isfinite(cd.M[np.ix_(rr, st + 1)])
            ri, kk = np.nonzero(fa ^ fb)
            for r_, k_ in zip(rr[ri], st[kk], strict=True):
                rows_.append((r_, k_, "pair_one_month_missing"))
                rows_.append((r_, k_ + 1, "pair_one_month_missing"))
            sec = (Mmm[np.ix_(rr, st)] == 7) & (Mmm[np.ix_(rr, st + 1)] == 7)
            ri, kk = np.nonzero(sec)
            for r_, k_ in zip(rr[ri], st[kk], strict=True):
                rows_.append((r_, k_, "bill_in_second_month"))
                rows_.append((r_, k_ + 1, "bill_in_second_month"))
        if rows_:
            rr_ = np.array([x[0] for x in rows_])
            tt_ = np.array([x[1] for x in rows_])
            fp = cd.first_array_pump[rr_]
            mask_rows.append(pd.DataFrame({
                "county": c, "meter_idx": cd.meters["meter_idx"].to_numpy()[rr_],
                "month": dates[tt_], "reason": [x[2] for x in rows_],
                "c0_flag": Fm[rr_, tt_].astype("int64"),
                "c0_method": [L.METHODS[int(x)] for x in Mmm[rr_, tt_]],
                "chowlin_arm_flag": a["flag"][fp, tt_].astype("int64"),
                "chowlin_arm_method": [L.METHODS[int(x)] for x in a["method"][fp, tt_]],
                "recorded_arm_flag": arms["recorded"]["flag"][fp, tt_].astype("int64")}))
        info["counties"][c] = {
            "n_pumps": int(len(cd.uid)), "n_meters": int(len(cd.meters)),
            "chowlin_pump_months_flag3": int((a["flag"] == 3).sum()),
            "july_imputed_pumps": int((a["method"] == L.M_JULY_IMPUTED)[:, -1].sum())
            if variant == "full" else 0, "q_imp": q_imp}
        print(variant, c, info["counties"][c], flush=True)
    # all-county class arrays and totals
    for arm, R in arrays.items():
        for key, Gsz in (("cls", C * A), ("all", A), ("nodom", A)):
            acc = None
            for c in L.COUNTIES:
                cd, arms, cls, gc = county_pumps[c]
                parts = _parts(arms[arm], cd, gkey(key, cls, gc, A), Gsz)
                acc = parts if acc is None else {k: acc[k] + parts[k] for k in acc}
            f = flags_from_parts(acc)
            v = acc["value"].copy()
            v[f == 8] = np.nan
            sd_c0 = {"cls": cz["sd"].reshape(C * A, T), "all": ct["sd_all"],
                     "nodom": ct["sd_nodomestic"]}[key]
            sd, n_eff = _sd_all_counties(arm, county_pumps, key, Gsz, A, sdm, variant, info,
                                         sd_c0, T)
            if key == "cls":
                shape = (C, A, T)
                R["value"], R["flag"], R["sd"] = v.reshape(shape), f.reshape(shape), \
                    sd.reshape(shape)
                R["n_eff"] = n_eff.reshape(shape)
                R["n_cl"] = acc["n_cl"].reshape(shape)
                R["n_imp"] = acc["n_imp"].reshape(shape)
            else:
                name = "all" if key == "all" else "nodomestic"
                R[f"value_total_{name}"] = v
                R[f"flag_total_{name}"] = f
                R[f"sd_total_{name}"] = sd
                R[f"n_eff_total_{name}"] = n_eff
        meta = {k: cz[k] for k in ("classes", "counties", "dx", "grid_hash", "mask_hash",
                                   "n_active", "nx", "ny", "x0", "y0")}
        np.savez_compressed(out / f"kwh_by_class_{arm}.npz", dates=dates, unit=UNIT,
                            missing_expected_kwh=cz["missing_expected_kwh"],
                            missing_expected_kwh_all=ct["missing_expected_kwh_all"],
                            missing_expected_kwh_nodomestic=ct[
                                "missing_expected_kwh_nodomestic"],
                            **meta, **R)
    pd.concat(long_rows, ignore_index=True).to_parquet(out / "county_class_monthly.parquet",
                                                       index=False)
    pd.concat(meters_out, ignore_index=True).to_parquet(out / "meter_phase.parquet",
                                                        index=False)
    if mask_rows:
        pd.concat(mask_rows, ignore_index=True).to_parquet(
            out / "incomplete_pair_mask.parquet", index=False)
    if july_rows:
        pd.concat(july_rows, ignore_index=True).to_parquet(
            out / "july2026_imputation.parquet", index=False)
    shutil.copy2(L.WORK / f"{variant}_profiles.npz", out / "profiles.npz")
    shutil.copy2(zd / f"zeros_{variant}.parquet", out / "zero_class.parquet")
    shutil.copy2(zd / f"meter_zero_summary_{variant}.parquet",
                 out / "meter_zero_summary.parquet")
    info["seconds"] = round(time.time() - t0, 1)
    info["env"] = L.env_info()
    L.save_json(info, out / "build_info.json")
    print("assembled", out, info["seconds"])


def _parts(a: dict, cd: L.CountyData, g: np.ndarray, G: int) -> dict:
    S = group_matrix(g, G)
    hp = cd.hp[:, None]
    meth, flag = a["method"], a["flag"]
    return {"value": S @ np.nan_to_num(a["value"]),
            "n_cl": S @ (flag == 3).astype(float),
            "n_fill": S @ (meth == 9).astype(float),
            "n_imp": S @ (meth == L.M_JULY_IMPUTED).astype(float),
            "n_incomplete": S @ (meth == 4).astype(float),
            "n_partial": S @ (meth == 5).astype(float),
            "hp_installed": S @ (hp * (meth != 1)),
            "hp_missing": S @ (hp * (meth == 3))}


def _sd_all_counties(arm, county_pumps, key, G, A, sdm, variant, info, sd_c0, T):
    """sd of an all-county aggregate: per county Chow-Lin and imputation variances (each
    county's own sd model / imputation q90) added, plus C0's sd, in quadrature."""
    moy = L.DATES[:T].month.to_numpy()
    var = np.zeros((G, T))
    has = np.zeros((G, T), bool)
    Etot = np.zeros((G, T))
    S2tot = np.zeros((G, T))
    for c, (cd, arms, cls, gc) in county_pumps.items():
        a = arms[arm]
        g = gkey(key, cls, gc, A)
        S0 = group_matrix(g, G)
        has |= (S0 @ (a["flag"] == 3).astype(float) > 0) | \
            (S0 @ (a["method"] == L.M_JULY_IMPUTED).astype(float) > 0)
        if arm == "chowlin" and (a["e_cl"] > 0).any():
            B, G2 = gm_matrices(g, cd.prow, G)
            U = B @ a["e_cl"]
            Es = G2 @ U
            S2 = G2 @ (U ** 2)
            ok = Es > 0
            ne = np.where(ok, Es ** 2 / np.where(ok, S2, 1.0), 1.0)
            q = q90_interp(sdm[c]["q90_by_n_moy"], ne, np.broadcast_to(moy, (G, T)))
            var[ok] += (q[ok] / L.Z90 * Es[ok]) ** 2
            has |= ok
            Etot += Es
            S2tot += S2
        if variant == "full" and (a["imp"] > 0).any():
            S = group_matrix(g, G)
            I_ = S @ a["imp"]
            ok = I_ > 0
            qi = info["counties"][c]["q_imp"][arm]["cell"]
            var[ok] += (qi / L.Z90 * I_[ok]) ** 2
            has |= ok
    okz = np.isfinite(sd_c0)
    var[okz] += sd_c0[okz] ** 2
    has |= okz
    n_eff = np.where(Etot > 0, Etot ** 2 / np.where(S2tot > 0, S2tot, 1.0), np.nan)
    return np.where(has, np.sqrt(var), np.nan), n_eff


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
