"""WP C0 builder: one county, both variants -> intermediates in results/dataprep/pump_electricity/_build_c0/.

For each county it reads the L0 parquet derivative, harmonises the census, runs the model's
``clean_census`` (kept set on 2012-2022, cap-None re-share of the whole record), builds the
dense per-pump series, the per-meter billing/termination tables, the per pump-month method
codes and flags, and the per-cell x class arrays (plus county-scope class sums). Nothing is
compared with a reference here; the acceptance checks live in ``checks.py``. The only
assertion is the build invariant "value NaN <=> flag 8" at pump level.

``process`` is generic (any harmonised census + kWh table), so ``checks.py`` can run the same
pipeline on the AMP_V2 cache.

Usage: python -I build.py <county> [<county> ...]      (inside tools/cpu_slot.sh)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

C = len(lib.CLASSES)
CLS_IDX = {c: i for i, c in enumerate(lib.CLASSES)}
FLAG_OF_METHOD = np.zeros(256, dtype=np.uint8)
for _code, (_name, _flag) in lib.METHODS.items():
    FLAG_OF_METHOD[_code] = _flag


def clean_and_allocate(df: pd.DataFrame, k: pd.DataFrame, end: str):
    """Model cleaning: (positions of the kept pumps in ``df``, dense (P_kept, T) allocated
    kWh, NaN where the meter's reading is missing, clean report)."""
    T = lib.tidx(end) + 1
    end_ts = lib.DATES[T - 1]
    kv = k if T == lib.T_ALL else k[k["datetime"] <= end_ts]
    mp = lib.model_pumps(df)
    kept, _, rep = lib.clean_census(mp, kv, cap_duty=lib.CAP_DUTY, t0=lib.DUTY_T0,
                                    t1=lib.DUTY_T1)
    _, kc, _ = lib.clean_census(kept, kv, cap_duty=None)
    kept_sids = kept["sid"].astype(str).to_numpy()
    V = lib.dense_from_long(kc, kept_sids, T)
    pos = pd.Index(df["sid"].astype(str)).get_indexer(kept_sids)
    if (pos < 0).any() or not np.all(np.diff(pos) > 0):
        raise ValueError("kept pumps do not map onto the census in order")
    return pos, V, rep


def group_sum(groups: np.ndarray, n_groups: int, X: np.ndarray) -> np.ndarray:
    """Sum the rows of X (P, T) into n_groups by ``groups`` (rows with group < 0 skipped)."""
    keep = groups >= 0
    S = sp.csr_matrix((np.ones(int(keep.sum())), (groups[keep], np.where(keep)[0])),
                      shape=(n_groups, X.shape[0]))
    return np.asarray(S @ X)


def meter_install(df: pd.DataFrame, kept_meters: np.ndarray) -> np.ndarray:
    """Install month index of each kept meter = the earliest valid install date among all the
    meter's pumps (a very negative number when none is valid: no pre-install months)."""
    big = np.iinfo("int64").max
    ti = np.where(df["install_valid"].to_numpy(), df["_t_install"].to_numpy(), big)
    inst = pd.Series(ti).groupby(df["meter_idx"].to_numpy()).min().loc[kept_meters].to_numpy()
    return np.where(inst == big, np.iinfo("int64").min // 2, inst)


def process(tag: str, df: pd.DataFrame, k: pd.DataFrame, L: np.ndarray, variant: str,
            grid) -> dict:
    v = lib.VARIANTS[variant]
    T = lib.tidx(v["end"]) + 1
    t_last = lib.tidx(v["last_complete"])
    t0 = time.time()
    pos, V, rep = clean_and_allocate(df, k, v["end"])
    sub = df.iloc[pos].reset_index(drop=True)
    P = len(sub)
    hp = lib.hp_numeric(sub["PUMP_HP"])
    meter = sub["meter_idx"].to_numpy()
    share = lib.hp_shares(meter, hp)
    kept_meters, m_of_p = np.unique(meter, return_inverse=True)
    M = len(kept_meters)
    first_row = df.index[df["meter_first_pump"].to_numpy()]
    first_of_meter = pd.Series(first_row, index=df.loc[first_row, "meter_idx"].to_numpy())
    Mm = L[first_of_meter.loc[kept_meters].to_numpy(), :T]

    # allocation identity: V == share x meter reading (NaN pattern identical)
    W = share[:, None] * Mm[m_of_p]
    nan_same = bool(np.array_equal(np.isnan(V), np.isnan(W)))
    fin = np.isfinite(V)
    alloc_maxabs = float(np.max(np.abs(V[fin] - W[fin]))) if fin.any() else 0.0
    del W

    inst = meter_install(df, kept_meters)

    # billing code / phase (meter-level; the first pump's code, checked consistent)
    codes = df.groupby("meter_idx")["billing_code"].agg(lambda s: s.dropna().unique().tolist())
    mixed = int((codes.map(len) > 1).sum())
    code_m = df.loc[first_of_meter.loc[kept_meters].to_numpy(), "billing_code"].to_numpy()
    phase_code = lib.phase_from_code(code_m)
    pw = v["phase_window"]
    det = lib.detect_phase(Mm, lib.tidx(pw[0]), lib.tidx(pw[1]) + 1)
    term, term_from = lib.termination(Mm, t_last)
    pm = lib.pair_methods(Mm, phase_code, t_last + 1)

    # meter-level method codes (ascending priority; later assignments win)
    tt = np.arange(T)
    pre = tt[None, :] < inst[:, None]
    nan = np.isnan(Mm)
    meth = np.zeros((M, T), dtype=np.uint8)
    meth[:, :t_last + 1][pm == 8] = 8
    meth[:, :t_last + 1][pm == 7] = 7
    in_run = term[:, None] & (tt[None, :] >= term_from[:, None]) & (tt[None, :] <= t_last)
    meth[in_run] = 6
    meth[nan & ~pre] = 3
    open_edge = np.zeros(M, bool)
    if v["mask_last"]:
        j = T - 1
        open_edge = (phase_code == "A") | ((phase_code == "unknown")
                                           & (det["phase_detected"].to_numpy() == "A"))
        monthly = (phase_code == "monthly") & ~open_edge
        meth[open_edge & ~pre[:, j], j] = 4
        # a monthly meter's partial reading cycle is flagged 9 only where a reading exists
        # (a missing reading stays 'missing', flag 8)
        meth[monthly & ~pre[:, j] & ~nan[:, j], j] = 5
    meth[pre & (nan | (Mm == 0))] = 1
    meth[pre & (Mm > 0)] = 2

    # per-pump values, flags, methods
    meth_p = meth[m_of_p]
    Vp = V.copy()
    Vp[meth_p == 1] = 0.0
    Vp[meth_p == 4] = np.nan
    flag_p = FLAG_OF_METHOD[meth_p]
    assert np.array_equal(np.isnan(Vp), flag_p == 8), "NaN <=> flag 8 violated (pump level)"

    # cell and county-scope aggregation per class
    A = grid.n_active
    cls = np.array([CLS_IDX[c] for c in sub["purpose_class"]])
    cell = sub["cell"].to_numpy()
    g_cell = np.where(cell >= 0, cls * A + cell, -1)
    term_p = (term[m_of_p][:, None] & (tt[None, :] >= term_from[m_of_p][:, None])).astype(
        np.float64)
    parts = {
        "value": np.nan_to_num(Vp, nan=0.0),
        "n_incomplete": (meth_p == 4).astype(np.float64),
        "n_partial": (meth_p == 5).astype(np.float64),
        "hp_installed": hp[:, None] * (meth_p != 1),
        "hp_missing": hp[:, None] * (meth_p == 3),
        "n_terminated": term_p,
        "n_resplit": np.zeros((P, T)),
    }
    alt = None
    if variant == "full":
        # even-split alternative: a pair posted wholly to its second month (method 7 on both
        # months) is re-split evenly over the two months, flag 1
        Ma = Mm.copy()
        resplit_m = np.zeros((M, T), bool)
        for ph in ("A", "B"):
            st = lib.pair_starts(ph, 0, t_last + 1)
            rows = np.where(phase_code == ph)[0]
            sel = ((meth[np.ix_(rows, st)] == 7) & (meth[np.ix_(rows, st + 1)] == 7))
            rr, kk = np.nonzero(sel)
            a = Mm[rows[rr], st[kk]]
            b = Mm[rows[rr], st[kk] + 1]
            Ma[rows[rr], st[kk]] = (a + b) / 2
            Ma[rows[rr], st[kk] + 1] = (a + b) / 2
            resplit_m[rows[rr], st[kk]] = True
            resplit_m[rows[rr], st[kk] + 1] = True
        rs = resplit_m[m_of_p]
        Va = Vp.copy()
        pr, tc = np.nonzero(rs)
        Va[pr, tc] = share[pr] * Ma[m_of_p[pr], tc]
        alt = Va
        parts["n_resplit"] = rs.astype(np.float64)
        parts["value_evensplit"] = np.nan_to_num(Va, nan=0.0)
    cellarr = {key: group_sum(g_cell, C * A, X).reshape(C, A, T) for key, X in parts.items()}
    county = {key: group_sum(cls, C, X) for key, X in parts.items()}

    # meter table (kept meters)
    last12 = np.full(M, np.nan)
    for i in np.where(term)[0]:
        a0 = max(0, term_from[i] - 12)
        last12[i] = np.nansum(Mm[i, a0:term_from[i]])
    mt = pd.DataFrame({
        "county": tag, "meter_idx": kept_meters,
        "n_pumps_kept": np.bincount(m_of_p, minlength=M),
        "hp_sum": np.bincount(m_of_p, weights=hp, minlength=M),
        "billing_code": code_m, "phase_code": phase_code,
        "phase_detected": det["phase_detected"].to_numpy(), "fA": det["fA"].to_numpy(),
        "fB": det["fB"].to_numpy(), "nA": det["nA"].to_numpy(), "nB": det["nB"].to_numpy(),
        "install_month": [lib.ym(t) if 0 <= t < lib.T_ALL else (None if t < -10**9 else
                          f"{2007 + t // 12:04d}-{t % 12 + 1:02d}") for t in inst],
        "terminated": term,
        "terminated_from": [lib.ym(t) if t >= 0 else None for t in term_from],
        "kwh_last12_before_termination": last12,
        "n_pair_months_bill_in_second_month": (pm == 7).sum(1),
        "n_pair_months_uneven": (pm == 8).sum(1),
        "n_months_missing": (meth == 3).sum(1),
        "n_months_pre_install": ((meth == 1) | (meth == 2)).sum(1),
        "incomplete_pair_last_month": open_edge if v["mask_last"] else False,
        "class_first_pump": sub.groupby(m_of_p)["purpose_class"].first().to_numpy(),
        "in_grid_any": pd.Series(cell >= 0).groupby(m_of_p).any().to_numpy(),
    })
    mt["phase_agree"] = np.where(mt["phase_code"].isin(["A", "B"]),
                                 mt["phase_code"] == mt["phase_detected"],
                                 np.where(mt["phase_code"] == "monthly",
                                          mt["phase_detected"] == "monthly", False))
    method_counts = {lib.METHODS[c][0]: int((meth_p == c).sum()) for c in lib.METHODS}
    info = {
        "county": tag, "variant": variant, "months": [lib.ym(0), lib.ym(T - 1)],
        "pumps_census": int(len(df)), "pumps_kept": int(P), "meters_kept": int(M),
        "pumps_kept_in_grid": int((cell >= 0).sum()),
        "meters_mixed_billing_codes": mixed,
        "alloc_identity_nan_pattern_equal": nan_same,
        "alloc_identity_max_abs_kwh": alloc_maxabs,
        "method_counts_pump_months": method_counts,
        "meters_terminated": int(term.sum()),
        "meters_terminated_in_grid": int((term & mt["in_grid_any"].to_numpy()).sum()),
        "phase_code_counts": pd.Series(phase_code).value_counts().to_dict(),
        "phase_detected_counts": det["phase_detected"].value_counts().to_dict(),
        "phase_agree_frac_bimonthly": float(mt.loc[mt.phase_code.isin(["A", "B"]),
                                                   "phase_agree"].mean()),
        "phase_agree_frac_monthly": float(mt.loc[mt.phase_code == "monthly",
                                                 "phase_agree"].mean()),
        "clean_report": rep.reset_index().to_dict(orient="records"),
        "seconds": round(time.time() - t0, 1),
    }
    return {"sub": sub, "hp": hp, "share": share, "m_of_p": m_of_p, "Vp": Vp,
            "flag_p": flag_p, "meth_p": meth_p, "alt": alt, "cell": cellarr,
            "county": county, "meters": mt, "info": info, "T": T}


def coding_sensitivity(df: pd.DataFrame, L: np.ndarray, grid) -> dict:
    """Kept set under the alternative pre-install coding (every pre-install month 0 before the
    duty statistic) against the primary (L0 as delivered): meters, pumps and 2012-2022
    de-duplicated energy (county and in grid). Reported, never used for the product."""
    first = df["meter_first_pump"].to_numpy()
    fr = np.where(first)[0]
    midx = df["meter_idx"].to_numpy()
    hp = lib.hp_numeric(df["PUMP_HP"])
    hp_m = pd.Series(hp).groupby(midx).sum().loc[midx[fr]].to_numpy()
    inst = meter_install(df, midx[fr])
    X = L[fr]
    pre = np.arange(lib.T_ALL)[None, :] < inst[:, None]
    w = slice(lib.tidx("2012-01"), lib.tidx("2022-12") + 1)
    in_grid_m = pd.Series(df["cell"].to_numpy() >= 0).groupby(midx).any().loc[midx[fr]]
    out = {}
    for name, Xc in (("primary_as_delivered", X),
                     ("alt_preinstall_zero", np.where(np.isnan(X) & pre, 0.0, X))):
        d = lib.meter_duty(Xc, hp_m)
        kept_m = ~(d > lib.CAP_DUTY)
        e = np.nansum(Xc[:, w], 1)
        kept_p = pd.Series(kept_m, index=midx[fr]).reindex(midx).to_numpy()
        out[name] = {"meters_kept": int(kept_m.sum()), "pumps_kept": int(kept_p.sum()),
                     "kwh_2012_2022_GWh": float(e[kept_m].sum() / 1e6),
                     "kwh_2012_2022_in_grid_meters_GWh": float(e[kept_m & in_grid_m.to_numpy()]
                                                               .sum() / 1e6)}
    a, b = out["primary_as_delivered"], out["alt_preinstall_zero"]
    out["meters_flipped"] = int(b["meters_kept"] - a["meters_kept"])
    out["energy_change_rel"] = (b["kwh_2012_2022_GWh"] / a["kwh_2012_2022_GWh"] - 1
                                if a["kwh_2012_2022_GWh"] else None)
    return out


def save(tag: str, variant: str, r: dict, out: Path) -> None:
    sub = r["sub"]
    keep_rows = np.ones(len(sub), bool) if tag != "chiayi" else (sub["cell"].to_numpy() >= 0)
    lib.save_npz(
        out / f"{variant}_{tag}_pumps.npz",
        uid=sub["uid"].to_numpy(dtype=str)[keep_rows],
        cell=sub["cell"].to_numpy()[keep_rows],
        purpose_class=sub["purpose_class"].to_numpy(dtype=str)[keep_rows],
        meter_idx=sub["meter_idx"].to_numpy()[keep_rows],
        hp=r["hp"][keep_rows], hp_share=r["share"][keep_rows],
        value=r["Vp"][keep_rows], flag=r["flag_p"][keep_rows], method=r["meth_p"][keep_rows],
        dates=np.array([lib.ym(i) for i in range(r["T"])]))
    if r["alt"] is not None:
        lib.save_npz(out / f"{variant}_{tag}_pumps_evensplit.npz",
                            uid=sub["uid"].to_numpy(dtype=str)[keep_rows],
                            value=r["alt"][keep_rows])
    lib.save_npz(out / f"{variant}_{tag}_cells.npz",
                        **{f"cell_{k}": v for k, v in r["cell"].items()},
                        **{f"county_{k}": v for k, v in r["county"].items()})
    r["meters"].to_parquet(out / f"{variant}_{tag}_meters.parquet", index=False)
    lib.save_json(r["info"], out / f"{variant}_{tag}_info.json")


def run(tag: str, out: Path) -> dict:
    t0 = time.time()
    grid = lib.load_grid()
    raw = lib.read_census_l0(tag)
    df = lib.harmonise(tag, raw, grid)
    del raw
    k, L = lib.read_kwh_l0(tag, df["sid"].to_numpy(), len(df))
    # L1 rows (variant-independent): kept status and duty from the 2012-2022 statistic
    hp_all = lib.hp_numeric(df["PUMP_HP"])
    hp_sum_m = pd.Series(hp_all).groupby(df["meter_idx"].to_numpy()).sum()
    first = df["meter_first_pump"].to_numpy()
    duty_m = pd.Series(lib.meter_duty(L[first], hp_sum_m.loc[df.loc[first, "meter_idx"]]
                                      .to_numpy()), index=df.loc[first, "meter_idx"].to_numpy())
    res = {}
    kept_uid, share_of = set(), pd.Series(dtype=float)
    for variant in ("preorigin", "full"):
        r = process(tag, df, k, L, variant, grid)
        save(tag, variant, r, out)
        res[variant] = r["info"]
        if variant == "preorigin":
            kept_uid = set(r["sub"]["uid"])
            share_of = pd.Series(r["share"], index=r["sub"]["uid"].to_numpy())
        print(tag, variant, {x: r["info"][x] for x in ("pumps_kept", "pumps_kept_in_grid",
                                                         "meters_kept", "seconds")}, flush=True)
        del r
    l1 = df[lib.l1_columns()].copy()
    l1["meter_hp_sum"] = df["meter_idx"].map(hp_sum_m).to_numpy()
    l1["meter_duty_2012_2022"] = df["meter_idx"].map(duty_m).to_numpy()
    l1["kept"] = df["uid"].isin(kept_uid).to_numpy()
    l1["hp_share"] = df["uid"].map(share_of).to_numpy()
    if tag == "chiayi":
        # in-grid pumps plus their meter-mates (needed to reproduce the HP split)
        m_in = set(df.loc[df["in_grid"], "meter_idx"])
        l1 = l1[df["meter_idx"].isin(m_in).to_numpy()]
    l1.to_parquet(out / f"l1_{tag}.parquet", index=False)
    summary = {"county": tag, "seconds": round(time.time() - t0, 1), "variants": res,
               "pumps": int(len(df)), "pumps_in_grid": int(df["in_grid"].sum()),
               "install_invalid": int((~df["install_valid"]).sum()),
               "l1_rows": int(len(l1)),
               "coding_sensitivity_2012_2022": coding_sensitivity(df, L, grid),
               "env": lib.env_info()}
    lib.save_json(summary, out / f"summary_{tag}.json")
    return summary


def main(argv: list[str]) -> int:
    out = lib.WORK
    out.mkdir(parents=True, exist_ok=True)
    for tag in argv:
        if tag not in lib.COUNTIES:
            raise SystemExit(f"unknown county {tag}")
        s = run(tag, out)
        print(tag, "done", s["seconds"], "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
