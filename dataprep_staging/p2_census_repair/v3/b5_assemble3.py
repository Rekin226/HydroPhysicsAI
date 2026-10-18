"""Assemble the repair-round P2a-c products (yunlin_electricity/{preorigin,full}/v2) into a
staging directory ``_work_p2abc_v3/assembled_<variant>/``; ``b6_finalize3.py`` publishes a
staging directory whose registered checks passed.

Unchanged from round 2 (v1): the two arms ``recorded`` (C0 as read) and ``chowlin`` (eligible
bimonthly irrigation bills re-split by the county's Chow-Lin operator, flag 3) with their
shipping decisions, sd models and sd rules (round-2 registered results t1/t3, read from
``reports/P2abc_evidence/tests``), the stage-1 arrays of ``_work_p2abc_v2``, the July-2026
imputation of the full product (flag 4, where the county's July backtest passed), the
aggregate flag order 8 > 9 > 4 > 3 > 0 and the zero-energy sd repair.

Repair-round changes (registered in P2abc.r3.repair.v1, amended by v2):

1. **Zero tables** from ``zeros3.py`` (pilot run kinds; ``structural`` column; pair-consistent
   synchronized-zero marks), not from round 2's ``b2_zeros.py``.
2. **M0 July 2025 (full product, plan P2b).** Every kept meter with billing code M0 and
   detected phase monthly whose 2025-07 reading is an installed flag-0 observation (not a
   pre-install month; repair P2abc.r3.k1_repair.v1) is flagged 9, method
   ``m0_july2025_suspect`` (code 12), value kept as read, in every arm. An aggregate is flagged 9
   when such pumps hold more than 10 % of its installed HP (C0's threshold for aggregates with
   missing readings), on top of C0's flag-9 reasons.
3. **Synchronized-zero sensitivity member (both products; plan P2c: synchronized zeros "stay in
   the data with flag 9 and enter as a sensitivity member only").** Two more arms,
   ``recorded_syncfill`` and ``chowlin_syncfill``: the arm with every synchronized-zero
   meter-month of the zero table (``sync_suspect``) replaced by the meter's own mean of the same
   calendar month over the three most recent earlier years in which that month is an eligible
   observation of the arm (flag 0 or 3, installed, not itself a synchronized zero), searching up
   to six years back (no eligible year: the zero is kept). Filled months carry flag 4, method
   ``sync_zero_fill`` (code 13); their sd is q90/1.645 x the fill, q90 = the in-sample 90 %
   quantile of |prediction - observation| / prediction of the same predictor over the eligible
   meter-months of the county, arm and calendar month inside the variant's zero window
   (label: uncalibrated). In the full product the members' July-2026 imputation is recomputed
   from the member's own record (same rule and shipping as the arm).
4. ``incomplete_pair_mask.parquet`` adds the reasons ``sync_zero_month`` (every synchronized-
   zero meter-month) and ``m0_july2025_suspect`` (full).
5. ``sync_fill.parquet``: per meter-month fill of both members (kWh at meter level; a pump's
   value is hp_share x the meter value), with sd.

Usage: python -I b5_assemble3.py <variant> <zeros3_dir>
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import b3_assemble as B3  # noqa: E402  (v2, read-only helpers)
import p3lib as P3  # noqa: E402
from p3lib import L  # noqa: E402

C = len(L.CLASSES)
K = len(L.COUNTIES)
DOM = L.CLASSES.index("domestic")
CLS_IDX = {c: i for i, c in enumerate(L.CLASSES)}
UNIT = B3.UNIT
M_M0SUSPECT = 12
M_SYNCFILL = 13
METHODS3 = {**L.METHODS, M_M0SUSPECT: "m0_july2025_suspect", M_SYNCFILL: "sync_zero_fill"}
ARMS = ("recorded", "chowlin", "recorded_syncfill", "chowlin_syncfill")
BASE = {"recorded": "recorded", "chowlin": "chowlin", "recorded_syncfill": "recorded",
        "chowlin_syncfill": "chowlin"}
TESTS2 = L.EVID / "tests"
JULY_YEARS = (2023, 2024, 2025)
FLAG_METHOD = {0: "observed", 3: "with_chowlin_split", 4: "with_fill_or_imputation",
               8: "incomplete_billing_pair", 9: "observed_suspect"}
PART_KEYS = ("value", "n_cl", "n_fill", "n_imp", "n_sync", "n_incomplete", "n_partial",
             "hp_installed", "hp_missing", "hp_suspect")


# ----------------------------------------------------------------------------- flags
def flags_from_parts3(p: dict) -> np.ndarray:
    f = np.zeros(p["value"].shape, dtype=np.uint8)
    f[p["n_cl"] > 0] = 3
    f[(p["n_fill"] + p["n_imp"] + p["n_sync"]) > 0] = 4
    hpi = p["hp_installed"]
    f[(hpi > 0) & (p["hp_missing"] > 0.10 * hpi)] = 9
    f[(hpi > 0) & (p["hp_suspect"] > 0.10 * hpi)] = 9
    f[p["n_partial"] > 0] = 9
    f[p["n_incomplete"] > 0] = 8
    return f


# ----------------------------------------------------------------------------- meter arms
def base_arm(cd: L.CountyData, variant: str, arm: str, ship_cl: bool, meter_tab: pd.DataFrame):
    """Meter-level arm before the July-2026 imputation: V, Fl, Me, E (even-split energy of
    flag-3 months)."""
    M, F, Mm = cd.M.astype(np.float64), cd.F, cd.Mm
    T = cd.T
    V, Fl, Me = M.copy(), F.copy(), Mm.copy()
    E = np.zeros_like(V)
    if arm == "chowlin" and ship_cl:
        z = np.load(L.WORK / f"{variant}_{cd.county}_cl.npz")
        rows, vals, rs = z["rows"], z["values"], z["resplit"]
        ph = meter_tab["chowlin_phase"].to_numpy()[rows]
        orig = M[rows]
        tt = np.arange(T)[None, :]
        s0 = np.where(ph == "A", 0, 1)[:, None]
        start = np.clip(np.where((tt - s0) % 2 == 0, tt, tt - 1), 0, T - 2)
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
    return V, Fl, Me, E


def m0_suspect_rows(cd: L.CountyData) -> np.ndarray:
    """Kept meters with billing code M0, detected phase monthly and an installed flag-0
    reading in 2025-07 (a pre-install month is a structural zero, not a reading: repair
    P2abc.r3.k1_repair.v1)."""
    t25 = L.tidx("2025-07")
    return np.where((cd.code == "M0") & (cd.phase == "monthly") & (cd.F[:, t25] == 0)
                    & (cd.Mm[:, t25] != 1))[0]


def own_mean_pred(V: np.ndarray, elig: np.ndarray, rows: np.ndarray, ts: np.ndarray,
                  k_max: int = 6, n_use: int = 3) -> np.ndarray:
    """Mean of the n_use most recent eligible same-calendar-month values (t - 12k, k = 1..k_max)
    of each (row, t); NaN if none."""
    s = np.zeros(len(rows))
    n = np.zeros(len(rows), np.int64)
    for k in range(1, k_max + 1):
        tk = ts - 12 * k
        ok = tk >= 0
        take = np.zeros(len(rows), bool)
        take[ok] = elig[rows[ok], tk[ok]] & (n[ok] < n_use)
        s[take] += V[rows[take], tk[take]]
        n[take] += 1
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def fill_q90(V: np.ndarray, elig: np.ndarray, w0: int, w1: int) -> dict:
    """In-sample q90 of |pred - obs| / pred (pred > 0) of the own same-month mean predictor
    over eligible meter-months of months max(w0, 2013-01)..w1, per calendar month."""
    t_lo = max(w0, L.tidx("2013-01"))
    rr, tt = np.nonzero(elig[:, t_lo:w1 + 1])
    tt = tt + t_lo
    pred = own_mean_pred(V, elig, rr, tt)
    ok = np.isfinite(pred) & (pred > 0)
    z = np.abs(pred[ok] - V[rr[ok], tt[ok]]) / pred[ok]
    moy = L.DATES[tt[ok]].month.to_numpy()
    return {int(m): float(np.quantile(z[moy == m], 0.90)) for m in range(1, 13)} | {
        "n": int(ok.sum())}


def member_arm(cd, V, Fl, Me, E, sync_rt, w0, w1):
    """Synchronized-zero member of a base arm: returns (Vm, Flm, Mem, FILL, info, fill table
    pieces, q90)."""
    Vm, Flm, Mem = V.copy(), Fl.copy(), Me.copy()
    FILL = np.zeros_like(V)
    S = np.zeros(V.shape, bool)
    r, t = sync_rt
    S[r, t] = True
    elig = np.isin(Fl, (0, 3)) & np.isfinite(V) & (Me != 1) & ~S
    q = fill_q90(V, elig, w0, w1)
    target = (Fl[r, t] == 0) & (V[r, t] == 0)
    rr, tt = r[target], t[target]
    pred = own_mean_pred(V, elig, rr, tt)
    ok = np.isfinite(pred)
    fr, ft, fv = rr[ok], tt[ok], pred[ok]
    Vm[fr, ft] = fv
    Flm[fr, ft] = 4
    Mem[fr, ft] = M_SYNCFILL
    FILL[fr, ft] = fv
    qv = np.array([q[int(m)] for m in L.DATES[ft].month])
    info = {"sync_rows": int(len(r)), "targets": int(target.sum()),
            "not_target_flag_or_value": int((~target).sum()), "filled": int(ok.sum()),
            "kept_zero_no_history": int((~ok).sum()),
            "filled_kwh": float(fv.sum()), "q90_by_moy": q}
    tab = (fr, ft, fv, qv / L.Z90 * fv)
    return Vm, Flm, Mem, FILL, info, tab, q


def july_imputation(Vb: np.ndarray, rows: np.ndarray, arm: str) -> np.ndarray:
    """b1's July-2026 rule on a meter record: recorded-type arms use the Jul-Aug pair mean,
    Chow-Lin-type arms the July value, of 2023-2025 (NaN if any base is NaN)."""
    if BASE[arm] == "recorded":
        b = np.column_stack([(Vb[rows, L.tidx(f"{y}-07")] + Vb[rows, L.tidx(f"{y}-08")]) / 2
                             for y in JULY_YEARS])
    else:
        b = np.column_stack([Vb[rows, L.tidx(f"{y}-07")] for y in JULY_YEARS])
    return np.mean(b, axis=1)


# ----------------------------------------------------------------------------- pumps
def to_pumps(cd, V, Fl, Me, E, IMP, FILL, sdm, q_imp_meter, q_fill, arm):
    T = cd.T
    moy = L.DATES[:T].month.to_numpy()
    sh = cd.share[:, None]
    pv = V[cd.prow] * sh
    pf, pm = Fl[cd.prow], Me[cd.prow]
    pe = E[cd.prow] * sh
    pimp = IMP[cd.prow] * sh
    pfill = FILL[cd.prow] * sh
    sd = np.full(pv.shape, np.nan)
    if BASE[arm] == "chowlin" and (pe > 0).any():
        q1 = np.array([sdm["q90_by_n_moy"]["1"][str(m)] for m in range(1, 13)])
        qq = q1[moy - 1][None, :] / L.Z90
        sel = pf == 3
        sd[sel] = (qq * np.where(sel, pe, 0.0))[sel]
    c0fill = pm == 9
    sd[c0fill] = cd.psd[c0fill]
    if q_imp_meter is not None:
        sel = pm == L.M_JULY_IMPUTED
        sd[sel] = q_imp_meter / L.Z90 * pimp[sel]
    if q_fill is not None:
        qf = np.array([q_fill[m] for m in range(1, 13)])[moy - 1][None, :] / L.Z90
        sel = pm == M_SYNCFILL
        sd[sel] = (qf * np.where(sel, pfill, 0.0))[sel]
    assert np.array_equal(np.isnan(pv), pf == 8), "NaN <=> flag 8 (pump level)"
    return {"value": pv, "flag": pf, "method": pm, "sd": sd, "e_cl": pe, "imp": pimp,
            "fill": pfill}


# ----------------------------------------------------------------------------- aggregation
def parts_var(a: dict, cd: L.CountyData, g: np.ndarray, G: int, sdm, q_imp, q_fill) -> dict:
    """Additive parts, the variance of the Chow-Lin, imputation and fill components, the
    'has' mask and the n_eff accumulators of one county's pumps grouped by g."""
    T = cd.T
    moy = L.DATES[:T].month.to_numpy()
    S = B3.group_matrix(g, G)
    hp = cd.hp[:, None]
    meth, flag = a["method"], a["flag"]
    p = {"value": S @ np.nan_to_num(a["value"]),
         "n_cl": S @ (flag == 3).astype(float),
         "n_fill": S @ (meth == 9).astype(float),
         "n_imp": S @ (meth == L.M_JULY_IMPUTED).astype(float),
         "n_sync": S @ (meth == M_SYNCFILL).astype(float),
         "n_incomplete": S @ (meth == 4).astype(float),
         "n_partial": S @ (meth == 5).astype(float),
         "hp_installed": S @ (hp * (meth != 1)),
         "hp_missing": S @ (hp * (meth == 3)),
         "hp_suspect": S @ (hp * (meth == M_M0SUSPECT))}
    var = np.zeros((G, T))
    has = (p["n_cl"] > 0) | (p["n_imp"] > 0) | (p["n_sync"] > 0)
    Etot = np.zeros((G, T))
    S2tot = np.zeros((G, T))
    if sdm is not None and (a["e_cl"] > 0).any():
        Bm, G2 = B3.gm_matrices(g, cd.prow, G)
        U = Bm @ a["e_cl"]
        Es = G2 @ U
        S2 = G2 @ (U ** 2)
        ok = Es > 0
        ne = np.where(ok, Es ** 2 / np.where(ok, S2, 1.0), 1.0)
        q = B3.q90_interp(sdm["q90_by_n_moy"], ne, np.broadcast_to(moy, (G, T)))
        var[ok] += (q[ok] / L.Z90 * Es[ok]) ** 2
        has |= ok
        Etot += Es
        S2tot += S2
    if q_imp is not None and (a["imp"] > 0).any():
        I_ = S @ a["imp"]
        ok = I_ > 0
        var[ok] += (q_imp / L.Z90 * I_[ok]) ** 2
        has |= ok
    if q_fill is not None and (a["fill"] > 0).any():
        qf = np.array([q_fill[m] for m in range(1, 13)])[moy - 1][None, :] / L.Z90
        F_ = S @ (a["fill"] * qf)
        ok = F_ > 0
        var[ok] += F_[ok] ** 2
        has |= ok
    return {"parts": p, "var": var, "has": has, "Etot": Etot, "S2tot": S2tot}


def finalize(pv: dict, sd_c0: np.ndarray | None) -> dict:
    p = pv["parts"]
    f = flags_from_parts3(p)
    val = p["value"].copy()
    val[f == 8] = np.nan
    var, has = pv["var"].copy(), pv["has"].copy()
    if sd_c0 is not None:
        okz = np.isfinite(sd_c0)
        var[okz] += sd_c0[okz] ** 2
        has |= okz
    n_eff = np.where(pv["Etot"] > 0, pv["Etot"] ** 2 / np.where(pv["S2tot"] > 0,
                                                                pv["S2tot"], 1.0), np.nan)
    return {"value": val, "flag": f, "sd": np.where(has, np.sqrt(var), np.nan),
            "n_eff": n_eff, "n_cl": p["n_cl"], "n_imp": p["n_imp"], "n_sync": p["n_sync"]}


def acc_add(acc: dict | None, pv: dict) -> dict:
    if acc is None:
        return {"parts": dict(pv["parts"]), "var": pv["var"].copy(), "has": pv["has"].copy(),
                "Etot": pv["Etot"].copy(), "S2tot": pv["S2tot"].copy()}
    for k in PART_KEYS:
        acc["parts"][k] = acc["parts"][k] + pv["parts"][k]
    acc["var"] += pv["var"]
    acc["has"] |= pv["has"]
    acc["Etot"] += pv["Etot"]
    acc["S2tot"] += pv["S2tot"]
    return acc


# ----------------------------------------------------------------------------- main
def main(variant: str, zeros_dir: str, counties=None, out_dir: str | None = None) -> None:
    """``counties``/``out_dir``: debug runs on a subset into a scratch directory only."""
    t0 = time.time()
    zd = Path(zeros_dir)
    counties = tuple(counties or L.COUNTIES)
    out = Path(out_dir) if out_dir else P3.WORK3 / f"assembled_{variant}"
    if out.exists():
        shutil.rmtree(out)                      # staging only, never a product version
    out.mkdir(parents=True)
    t1 = L.load_json(TESTS2 / f"t1_meter_validation_{variant}.json")
    t1pre = L.load_json(TESTS2 / "t1_meter_validation_preorigin.json")
    t3 = L.load_json(TESTS2 / "t3_july_backtest.json") if variant == "full" else None
    prod = L.open_c0(variant)
    cz = np.load(prod.path("cells_by_class.npz"))
    ct = np.load(prod.path("cells_total.npz"))
    c0long = pd.read_parquet(prod.path("county_class_monthly.parquet"))
    A = int(cz["n_active"])
    T = L.VARIANTS[variant]["T"]
    dates = np.array([L.ym(i) for i in range(T)])
    w0, w1 = (L.tidx(m) for m in P3_ZWIN[variant])
    ship_cl = {c: bool(t1pre["counties"][c]["passed"]) for c in ("yunlin", "changhua")}
    ship_cl["chiayi"] = ship_cl["yunlin"]
    sdm = {c: t1["counties"][L.PROFILE_OF[c]]["sd_model"] for c in L.COUNTIES}
    sd_label = {c: t1["counties"][L.PROFILE_OF[c]]["sd_coverage"]["label"]
                for c in ("yunlin", "changhua")}
    sd_label["chiayi"] = "uncalibrated (not checked on Chiayi meters; Yunlin's sd model)"
    zt = pd.read_parquet(zd / f"zeros_{variant}.parquet",
                         columns=["county", "meter_idx", "month", "sync_suspect"])
    info = {"variant": variant, "ship_chowlin": ship_cl, "sd_label_chowlin": sd_label,
            "counties": {}, "arms": list(ARMS)}
    R = {arm: {"value_by_county": np.zeros((K, C, A, T)),
               "flag_by_county": np.zeros((K, C, A, T), np.uint8),
               "sd_by_county": np.full((K, C, A, T), np.nan)} for arm in ARMS}
    ACC = {arm: {"cls": None, "all": None, "nodom": None} for arm in ARMS}
    long_rows, meters_out, mask_rows, july_rows, fill_rows = [], [], [], [], []
    for ki, c in enumerate(L.COUNTIES):
        if c not in counties:
            continue
        cd = L.CountyData(variant, c, prod=prod)
        meter_tab = pd.read_parquet(L.WORK / f"{variant}_{c}_meters.parquet")
        cls = np.array([CLS_IDX[x] for x in cd.pclass])
        g_cell = np.where(cd.cell >= 0, cls * A + cd.cell, -1)
        zc = zt[(zt["county"] == c) & zt["sync_suspect"]]
        sync_r = cd.mpos.reindex(zc["meter_idx"].to_numpy()).to_numpy().astype(np.int64)
        sync_t = np.array([L.tidx(m) for m in zc["month"]], np.int64) if len(zc) else \
            np.zeros(0, np.int64)
        m0r = m0_suspect_rows(cd) if variant == "full" else np.zeros(0, np.int64)
        ci = {"n_pumps": int(len(cd.uid)), "n_meters": int(len(cd.meters)),
              "m0_july2025_suspect_meters": int(len(m0r)),
              "m0_july2025_suspect_pumps": int(np.isin(cd.prow, m0r).sum()),
              "sync_suspect_meter_months": int(len(sync_r)), "arms": {}}
        july = None
        if variant == "full":
            jz = np.load(L.WORK / f"full_{c}_july.npz")
            tr = t3["counties"][c]
            july = {"ship": bool(tr["passed"]), "rows": jz["rows"],
                    "q_meter": {"recorded": tr["recorded_pair_mean"]["imputation_sd"]["meter"][
                        "q90_rel"], "chowlin": tr["chowlin_arm"]["imputation_sd"]["meter"][
                        "q90_rel"]},
                    "stage1": {"recorded": jz["imputed_recorded"],
                               "chowlin": jz["imputed_chowlin"]}}
            q_imp = {}
            for arm_b, key in (("recorded", "recorded_pair_mean"), ("chowlin", "chowlin_arm")):
                bt = tr[key]
                errs = np.array([r["total_err_pct"] for r in bt["per_year"]])
                q_imp[arm_b] = {"cell": bt["imputation_sd"]["cell"]["q90_rel"],
                                "fan": float(np.quantile(np.abs(errs) / (100.0 + errs), 0.90))}
            ci["q_imp"] = q_imp
        jcols = {}
        fill_cols = {}
        for base in ("recorded", "chowlin"):
            V, Fl, Me, E = base_arm(cd, variant, base, ship_cl[c], meter_tab)
            if len(m0r):
                t25 = L.tidx("2025-07")
                assert (Fl[m0r, t25] == 0).all()
                Fl[m0r, t25] = 9
                Me[m0r, t25] = M_M0SUSPECT
            mem = member_arm(cd, V, Fl, Me, E, (sync_r, sync_t), w0, w1)
            for arm in (base, base + "_syncfill"):
                is_mem = arm.endswith("_syncfill")
                if is_mem:
                    Va, Fla, Mea, FILL, minfo, ftab, q_fill = mem
                    Va, Fla, Mea = Va.copy(), Fla.copy(), Mea.copy()
                    ci["arms"][arm] = {"fill": minfo}
                    fill_cols[base] = ftab
                else:
                    Va, Fla, Mea = V.copy(), Fl.copy(), Me.copy()
                    FILL = np.zeros_like(V)
                    q_fill = None
                    ci["arms"][arm] = {}
                IMP = np.zeros_like(Va)
                q_imp_meter = None
                if july is not None:
                    t26 = L.tidx("2026-07")
                    r_ = july["rows"]
                    val = july["stage1"][base] if not is_mem else july_imputation(Va, r_, arm)
                    if not is_mem:
                        # identity with round 2 (stage-1 values) for the unchanged arms
                        chk = july_imputation(Va, r_, arm)
                        both = np.isfinite(chk) & np.isfinite(val)
                        ci["arms"][arm]["july_stage1_identity_max_abs"] = float(
                            np.max(np.abs(chk[both] - val[both]))) if both.any() else 0.0
                    jcols[arm] = val
                    if july["ship"]:
                        ok = np.isfinite(val)
                        Va[r_[ok], t26] = val[ok]
                        Fla[r_[ok], t26] = 4
                        Mea[r_[ok], t26] = L.M_JULY_IMPUTED
                        IMP[r_[ok], t26] = val[ok]
                        q_imp_meter = july["q_meter"][base]
                a = to_pumps(cd, Va, Fla, Mea, E, IMP, FILL, sdm[c], q_imp_meter, q_fill, arm)
                del Va, IMP
                sdm_arm = sdm[c] if base == "chowlin" else None
                qi_cell = ci["q_imp"][base]["cell"] if (variant == "full" and july["ship"]) \
                    else None
                sd_c0 = cz["sd_by_county"][ki].reshape(C * A, T)
                fin_ = finalize(parts_var(a, cd, g_cell, C * A, sdm_arm, qi_cell, q_fill), sd_c0)
                R[arm]["value_by_county"][ki] = fin_["value"].reshape(C, A, T)
                R[arm]["flag_by_county"][ki] = fin_["flag"].reshape(C, A, T)
                R[arm]["sd_by_county"][ki] = fin_["sd"].reshape(C, A, T)
                for key, Gsz in (("cls", C * A), ("all", A), ("nodom", A)):
                    ACC[arm][key] = acc_add(ACC[arm][key], parts_var(
                        a, cd, B3.gkey(key, cls, g_cell, A), Gsz, sdm_arm, qi_cell, q_fill))
                c0l = c0long[c0long["county"] == c]
                for scope in ("in_grid", "county"):
                    bsel = (cd.cell >= 0) if scope == "in_grid" else np.ones(len(cd.cell), bool)
                    for name, sel in [(L.CLASSES[i], cls == i) for i in range(C)] + \
                            [("all", np.ones(len(cls), bool)), ("all_nodomestic", cls != DOM)]:
                        g = np.where(bsel & sel, 0, -1)
                        sdz = c0l[(c0l["scope"] == scope) & (c0l["class"] == name)] \
                            .set_index("date").reindex(dates)["sd"].to_numpy()[None, :]
                        qf_ = ci["q_imp"][base]["fan"] if (variant == "full" and july["ship"]) \
                            else None
                        r = finalize(parts_var(a, cd, g, 1, sdm_arm, qf_, q_fill), sdz)
                        long_rows.append(pd.DataFrame({
                            "county": c, "scope": scope, "class": name, "date": dates,
                            "arm": arm, "value": r["value"][0], "sd": r["sd"][0],
                            "flag": r["flag"][0].astype("int64"),
                            "method": [FLAG_METHOD[int(x)] for x in r["flag"][0]],
                            "n_chowlin_pumps": r["n_cl"][0], "n_imputed_pumps": r["n_imp"][0],
                            "n_syncfill_pumps": r["n_sync"][0], "unit": UNIT}))
                ci["arms"][arm]["flag3_pump_months"] = int((a["flag"] == 3).sum())
                ci["arms"][arm]["july_imputed_pumps"] = int(
                    (a["method"][:, -1] == L.M_JULY_IMPUTED).sum()) if variant == "full" else 0
                ci["arms"][arm]["syncfill_pump_months"] = int((a["method"] == M_SYNCFILL).sum())
                if arm == "chowlin":
                    np.savez_compressed(
                        out / f"pumps_{L.FILETAG[c]}.npz", uid=cd.uid, meter_idx=cd.meter_idx,
                        cell=cd.cell, purpose_class=cd.pclass, hp=cd.hp, hp_share=cd.share,
                        dates=dates, value=a["value"], flag=a["flag"], method=a["method"],
                        sd=a["sd"])
                    chow_pf = (a["flag"], a["method"])
                if arm == "recorded":
                    rec_pf = a["flag"]
                del a
            del V, Fl, Me, E, mem
        # sync fill table (meter level)
        fr, ft, fv, fs = fill_cols["recorded"]
        gr, gt, gv, gs = fill_cols["chowlin"]
        ftab = pd.DataFrame({"row": fr, "t": ft, "fill_recorded_kwh": fv, "sd_recorded_kwh": fs})
        gtab = pd.DataFrame({"row": gr, "t": gt, "fill_chowlin_kwh": gv, "sd_chowlin_kwh": gs})
        fm = ftab.merge(gtab, on=["row", "t"], how="outer")
        if len(fm):
            fill_rows.append(pd.DataFrame({
                "county": c, "meter_idx": cd.meters["meter_idx"].to_numpy()[fm["row"]],
                "month": dates[fm["t"]], **{k: fm[k].to_numpy() for k in (
                    "fill_recorded_kwh", "sd_recorded_kwh", "fill_chowlin_kwh",
                    "sd_chowlin_kwh")}}))
        if july is not None:
            july_rows.append(pd.DataFrame({
                "county": c, "meter_idx": cd.meters["meter_idx"].to_numpy()[july["rows"]],
                "imputed_recorded_kwh": jcols["recorded"],
                "imputed_chowlin_kwh": jcols["chowlin"],
                "sd_recorded_kwh": july["q_meter"]["recorded"] / L.Z90 * jcols["recorded"],
                "sd_chowlin_kwh": july["q_meter"]["chowlin"] / L.Z90 * jcols["chowlin"],
                "imputed_recorded_syncfill_kwh": jcols["recorded_syncfill"],
                "imputed_chowlin_syncfill_kwh": jcols["chowlin_syncfill"],
                "shipped": july["ship"] & np.isfinite(jcols["recorded"])}))
        mt = meter_tab.copy()
        mt["chowlin_shipped"] = ship_cl[c] & (mt["chowlin_phase"] != "")
        meters_out.append(mt)
        # mask table (meter level)
        Fm, Mmm = cd.F, cd.Mm
        rows_ = []
        if variant == "preorigin":
            j = L.tidx("2022-12")
            pr = cd.meters["dec_jan_paired"].to_numpy()
            for reason, sel in (("open_bill_dec2022_phaseB_c0_fill", pr & (Mmm[:, j] == 9)),
                                ("open_bill_dec2022_phaseB_missing", pr & (Fm[:, j] == 8))):
                rows_ += [(r_, j, reason) for r_ in np.where(sel)[0]]
        else:
            j = L.tidx("2026-07")
            for reason, sel in (("open_bill_jul2026_phaseA", Mmm[:, j] == 4),
                                ("partial_cycle_jul2026_monthly", Mmm[:, j] == 5)):
                rows_ += [(r_, j, reason) for r_ in np.where(sel)[0]]
            rows_ += [(r_, L.tidx("2025-07"), "m0_july2025_suspect") for r_ in m0r]
        ph = cd.phase
        for p_ in ("A", "B"):
            rr = np.where(ph == p_)[0]
            st = L.pair_starts(p_, 0, cd.T, cd.T)
            fa, fb = np.isfinite(cd.M[np.ix_(rr, st)]), np.isfinite(cd.M[np.ix_(rr, st + 1)])
            ri, kk = np.nonzero(fa ^ fb)
            for r_, k_ in zip(rr[ri], st[kk], strict=True):
                rows_ += [(r_, k_, "pair_one_month_missing"), (r_, k_ + 1,
                                                               "pair_one_month_missing")]
            sec = (Mmm[np.ix_(rr, st)] == 7) & (Mmm[np.ix_(rr, st + 1)] == 7)
            ri, kk = np.nonzero(sec)
            for r_, k_ in zip(rr[ri], st[kk], strict=True):
                rows_ += [(r_, k_, "bill_in_second_month"), (r_, k_ + 1, "bill_in_second_month")]
        rows_ += [(r_, t_, "sync_zero_month") for r_, t_ in zip(sync_r, sync_t, strict=True)]
        if rows_:
            rr_ = np.array([x[0] for x in rows_])
            tt_ = np.array([x[1] for x in rows_])
            fp = cd.first_array_pump[rr_]
            mask_rows.append(pd.DataFrame({
                "county": c, "meter_idx": cd.meters["meter_idx"].to_numpy()[rr_],
                "month": dates[tt_], "reason": [x[2] for x in rows_],
                "c0_flag": Fm[rr_, tt_].astype("int64"),
                "c0_method": [METHODS3[int(x)] for x in Mmm[rr_, tt_]],
                "chowlin_arm_flag": chow_pf[0][fp, tt_].astype("int64"),
                "chowlin_arm_method": [METHODS3[int(x)] for x in chow_pf[1][fp, tt_]],
                "recorded_arm_flag": rec_pf[fp, tt_].astype("int64")}))
        info["counties"][c] = ci
        print(variant, c, {k: v for k, v in ci.items() if k != "arms"}, flush=True)
        del cd
    # all-county class arrays and totals per arm
    for arm in ARMS:
        Rr = R[arm]
        for key in ("cls", "all", "nodom"):
            sd_c0 = {"cls": cz["sd"].reshape(C * A, T), "all": ct["sd_all"],
                     "nodom": ct["sd_nodomestic"]}[key]
            f_ = finalize(ACC[arm][key], sd_c0)
            if key == "cls":
                shape = (C, A, T)
                for k_ in ("value", "flag", "sd", "n_eff", "n_cl", "n_imp", "n_sync"):
                    Rr[k_] = f_[k_].reshape(shape)
            else:
                name = "all" if key == "all" else "nodomestic"
                for k_ in ("value", "flag", "sd", "n_eff"):
                    Rr[f"{k_}_total_{name}"] = f_[k_]
        meta = {k: cz[k] for k in ("classes", "counties", "dx", "grid_hash", "mask_hash",
                                   "n_active", "nx", "ny", "x0", "y0")}
        np.savez_compressed(out / f"kwh_by_class_{arm}.npz", dates=dates, unit=UNIT, arm=arm,
                            missing_expected_kwh=cz["missing_expected_kwh"],
                            missing_expected_kwh_all=ct["missing_expected_kwh_all"],
                            missing_expected_kwh_nodomestic=ct[
                                "missing_expected_kwh_nodomestic"], **meta, **Rr)
        R[arm] = None
        ACC[arm] = None
    pd.concat(long_rows, ignore_index=True).to_parquet(out / "county_class_monthly.parquet",
                                                       index=False)
    pd.concat(meters_out, ignore_index=True).to_parquet(out / "meter_phase.parquet",
                                                        index=False)
    pd.concat(mask_rows, ignore_index=True).to_parquet(out / "incomplete_pair_mask.parquet",
                                                       index=False)
    pd.concat(fill_rows, ignore_index=True).to_parquet(out / "sync_fill.parquet", index=False)
    if july_rows:
        pd.concat(july_rows, ignore_index=True).to_parquet(out / "july2026_imputation.parquet",
                                                           index=False)
    shutil.copy2(L.WORK / f"{variant}_profiles.npz", out / "profiles.npz")
    shutil.copy2(zd / f"zeros_{variant}.parquet", out / "zero_class.parquet")
    shutil.copy2(zd / f"meter_zero_summary_{variant}.parquet", out / "meter_zero_summary.parquet")
    info["methods"] = {str(k): v for k, v in METHODS3.items()}
    info["seconds"] = round(time.time() - t0, 1)
    info["env"] = L.env_info()
    L.save_json(info, out / "build_info.json")
    print("assembled", out, info["seconds"])


P3_ZWIN = {"preorigin": ("2012-01", "2022-12"), "full": ("2012-01", "2026-06")}

if __name__ == "__main__":
    if len(sys.argv) > 3:          # debug: <variant> <zeros_dir> <county,..> <scratch_out>
        main(sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
