"""WP C0 repair round, v3 (copy of ``repair_build.py`` with the v3 fill rule of
``repair3_lib``: the Oct-Nov 2022 base is coded as the product codes those months, and the
2022-12 method of a Dec-Jan-paired meter is set after the pre-install codes: fill (9), else a
structural zero by date (1) when not yet installed, else missing (3)).

Original v2 docstring follows. Per-county build of the v2 intermediates (pre-origin and full variants
plus the L1 rows) into a build directory (default: the session scratchpad given on the command
line; the products are assembled from it by ``repair_assemble.py``).

Differences from ``build.py`` (v1), see ``repair_lib``: the pre-origin context (Dec-Jan-paired
meters, the kept set from a duty statistic without their Dec-2022 value, the Dec-2022 fill
parameters) is computed once per county from L0 months <= 2022-12 only, and shared by both
variants; the pre-origin variant replaces the Dec-2022 value of Dec-Jan-paired meters by the
fill (flag 4, sd) or leaves it missing (flag 8); both variants add ``n_fill`` and
``missing_expected_kwh`` cell parts. Everything else is v1's ``build.process``.

``perturb`` (used by the registered invariance test R2) rebuilds the pre-origin variant from an
L0 table whose months >= 2023-01 and whose Dec-2022 readings of Dec-Jan-paired meters are
overwritten; the pre-origin outputs must not change.

Usage: python -I repair_build.py <build dir> <county> [<county> ...]   (inside cpu_slot.sh)
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
import build  # noqa: E402  (v1 helpers, imported read-only: group_sum, meter_install)
import lib  # noqa: E402
import repair3_lib as r3l  # noqa: E402
import repair_lib as rl  # noqa: E402

C = len(lib.CLASSES)
CLS_IDX = {c: i for i, c in enumerate(lib.CLASSES)}
J = rl.J_DEC22


# ----------------------------------------------------------------------------- context
def preorigin_context(df: pd.DataFrame, k: pd.DataFrame, L: np.ndarray) -> dict:
    """Everything fitted or classified, from L0 months <= 2022-12 only."""
    Lp = L[:, :rl.T_PRE]
    fr = np.where(df["meter_first_pump"].to_numpy())[0]
    midx = df["meter_idx"].to_numpy()[fr]
    if not np.array_equal(midx, np.arange(len(fr))):
        raise ValueError("meter_idx of first pumps is not 0..M-1")
    X = Lp[fr].copy()
    code = lib.phase_from_code(df["billing_code"].to_numpy()[fr])
    # pairs lying wholly inside 2012-01..2022-11: no pair involving a December-2022 value
    det = lib.detect_phase(X, lib.tidx("2012-01"), J)["phase_detected"].to_numpy()
    paired = rl.dec_jan_paired(code, det)
    X[paired, J] = np.nan                      # not a pre-origin observation
    hp_all = lib.hp_numeric(df["PUMP_HP"])
    hp_m = pd.Series(hp_all).groupby(df["meter_idx"].to_numpy()).sum().loc[midx].to_numpy()
    duty = lib.meter_duty(X, hp_m)
    # the model's clean_census on the same pre-origin record (paired meters' Dec-2022 rows
    # removed from every pump on those meters)
    paired_sid = set(df.loc[df["meter_idx"].isin(midx[paired]), "sid"].astype(str))
    kd = k[k["datetime"] <= lib.DATES[J]]
    kd = kd[~((kd["datetime"] == lib.DATES[J]) & kd["pump"].astype(str).isin(paired_sid))]
    kept, _, rep = lib.clean_census(lib.model_pumps(df), kd, cap_duty=lib.CAP_DUTY,
                                    t0=lib.DUTY_T0, t1=lib.DUTY_T1)
    kept_sid = set(kept["sid"].astype(str))
    kept_m_cc = set(df.loc[df["sid"].astype(str).isin(kept_sid), "meter_idx"])
    kept_m_duty = set(midx[~(duty > lib.CAP_DUTY)])
    if kept_m_cc != kept_m_duty:
        raise ValueError("clean_census kept set differs from the duty recomputation")
    kept_meter = np.zeros(len(fr), bool)
    kept_meter[sorted(kept_m_duty)] = True
    inst = build.meter_install(df, midx)
    cls = df["purpose_class"].to_numpy()[fr]
    params = rl.fill_params(X, inst, cls, paired & kept_meter)
    fill = r3l.meter_fill(X, inst, cls, paired & kept_meter, params)
    return {"kept": kept, "clean_report": rep, "paired": paired, "code": code, "det": det,
            "duty": duty, "hp_m": hp_m, "kept_meter": kept_meter, "inst": inst, "cls": cls,
            "params": params, "fill": fill}


# ----------------------------------------------------------------------------- one variant
def process(tag: str, df: pd.DataFrame, k: pd.DataFrame, L: np.ndarray, variant: str, grid,
            ctx: dict) -> dict:
    v = lib.VARIANTS[variant]
    T = lib.tidx(v["end"]) + 1
    t_last = lib.tidx(v["last_complete"])
    pre_variant = variant == "preorigin"
    t0 = time.time()
    kv = k if T == lib.T_ALL else k[k["datetime"] <= lib.DATES[T - 1]]
    kept = ctx["kept"]
    _, kc, _ = lib.clean_census(kept, kv, cap_duty=None)
    kept_sids = kept["sid"].astype(str).to_numpy()
    V = lib.dense_from_long(kc, kept_sids, T)
    pos = pd.Index(df["sid"].astype(str)).get_indexer(kept_sids)
    if (pos < 0).any() or not np.all(np.diff(pos) > 0):
        raise ValueError("kept pumps do not map onto the census in order")
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
    W = share[:, None] * Mm[m_of_p]
    nan_same = bool(np.array_equal(np.isnan(V), np.isnan(W)))
    fin = np.isfinite(V)
    alloc_maxabs = float(np.max(np.abs(V[fin] - W[fin]))) if fin.any() else 0.0
    del W
    inst = build.meter_install(df, kept_meters)
    code_m = df.loc[first_of_meter.loc[kept_meters].to_numpy(), "billing_code"].to_numpy()
    phase_code = lib.phase_from_code(code_m)
    codes = df.groupby("meter_idx")["billing_code"].agg(lambda s: s.dropna().unique().tolist())
    mixed = int((codes.map(len) > 1).sum())
    paired = ctx["paired"][kept_meters]
    fl = {key: (val[kept_meters] if isinstance(val, np.ndarray) else val)
          for key, val in ctx["fill"].items()}
    do_fill = paired & fl["ok"] if pre_variant else np.zeros(M, bool)
    if pre_variant:
        Mobs = Mm.copy()
        Mobs[paired, J] = np.nan                       # the pre-origin observed record
        Mrec = Mobs.copy()
        Mrec[do_fill, J] = fl["fill"][do_fill]
    else:
        Mobs = Mrec = Mm
    pw = v["phase_window"]
    det = lib.detect_phase(Mrec, lib.tidx(pw[0]), lib.tidx(pw[1]) + 1)
    term, term_from = lib.termination(Mrec, t_last)
    pm = lib.pair_methods(Mrec, phase_code, t_last + 1)

    tt = np.arange(T)
    pre = tt[None, :] < inst[:, None]
    nan = np.isnan(Mobs)
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
        meth[monthly & ~pre[:, j] & ~nan[:, j], j] = 5
    meth[pre & (nan | (Mm == 0))] = 1
    meth[pre & (Mm > 0)] = 2
    if pre_variant:
        # 2022-12 of a Dec-Jan-paired meter (its reading is not pre-origin): the fill, else a
        # structural zero by date when not yet installed, else missing
        mj = np.where(do_fill, rl.FILL_METHOD, np.where(pre[:, J], 1, 3)).astype(np.uint8)
        meth[paired, J] = mj[paired]

    meth_p = meth[m_of_p]
    Vp = V.copy()
    sd_p = np.full((P, T), np.nan)
    if pre_variant:
        pp = paired[m_of_p]
        Vp[pp, J] = np.nan                              # drop the January-2023 half
        fp = meth_p[:, J] == rl.FILL_METHOD
        mf = m_of_p[fp]
        Vp[fp, J] = share[fp] * fl["fill"][mf]
        sd_p[fp, J] = share[fp] * fl["base"][mf] * np.sqrt(fl["s"][mf] ** 2 + fl["sigma"] ** 2)
    Vp[meth_p == 1] = 0.0
    Vp[meth_p == 4] = np.nan
    flag_p = rl.FLAG_OF_METHOD[meth_p]
    assert np.array_equal(np.isnan(Vp), flag_p == 8), "NaN <=> flag 8 violated (pump level)"
    assert np.array_equal(np.isfinite(sd_p), flag_p == 4), "sd <=> flag 4 violated"

    A = grid.n_active
    cls = np.array([CLS_IDX[c] for c in sub["purpose_class"]])
    cell = sub["cell"].to_numpy()
    g_cell = np.where(cell >= 0, cls * A + cell, -1)
    term_p = (term[m_of_p][:, None] & (tt[None, :] >= term_from[m_of_p][:, None])).astype(
        np.float64)
    # expected kWh of post-install missing readings: HP x same-month class mean kWh/HP
    rep_ok = (flag_p == 0) & (meth_p != 1) & np.isfinite(Vp) & (hp[:, None] > 0)
    num = build.group_sum(cls, C, np.where(rep_ok, Vp, 0.0))
    den = build.group_sum(cls, C, np.where(rep_ok, hp[:, None], 0.0))
    rate = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    miss_exp = hp[:, None] * rate[cls] * (meth_p == 3)
    parts = {
        "value": np.nan_to_num(Vp, nan=0.0),
        "n_incomplete": (meth_p == 4).astype(np.float64),
        "n_partial": (meth_p == 5).astype(np.float64),
        "hp_installed": hp[:, None] * (meth_p != 1),
        "hp_missing": hp[:, None] * (meth_p == 3),
        "n_terminated": term_p,
        "n_resplit": np.zeros((P, T)),
        "n_fill": (meth_p == rl.FILL_METHOD).astype(np.float64),
        "missing_expected_kwh": miss_exp,
    }
    alt = None
    if variant == "full":
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
    cellarr = {key: build.group_sum(g_cell, C * A, X).reshape(C, A, T)
               for key, X in parts.items()}
    county = {key: build.group_sum(cls, C, X) for key, X in parts.items()}

    # fill info for the sd of any aggregate (pre-origin, month 2022-12 only)
    fp = meth_p[:, J] == rl.FILL_METHOD if pre_variant else np.zeros(P, bool)
    mf = m_of_p[fp]
    bp = share[fp] * fl["base"][mf] if fp.any() else np.zeros(0)
    fill_info = {"pump": np.where(fp)[0], "cell": cell[fp], "cls": cls[fp],
                 "meter_idx": meter[fp], "cc": bp * fl["s"][mf] if fp.any() else bp,
                 "ic": bp * fl["sigma"], "value": Vp[fp, J] if fp.any() else bp}

    last12 = np.full(M, np.nan)
    for i in np.where(term)[0]:
        a0 = max(0, term_from[i] - 12)
        last12[i] = np.nansum(Mrec[i, a0:term_from[i]])
    mt = pd.DataFrame({
        "county": tag, "meter_idx": kept_meters,
        "n_pumps_kept": np.bincount(m_of_p, minlength=M),
        "hp_sum": np.bincount(m_of_p, weights=hp, minlength=M),
        "billing_code": code_m, "phase_code": phase_code,
        "phase_detected": det["phase_detected"].to_numpy(), "fA": det["fA"].to_numpy(),
        "fB": det["fB"].to_numpy(), "nA": det["nA"].to_numpy(), "nB": det["nB"].to_numpy(),
        "phase_detected_2012_2022": ctx["det"][kept_meters],
        "dec_jan_paired": paired,
        "meter_duty_2012_2022": ctx["duty"][kept_meters],
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
    if pre_variant:
        mt["fill_base_octnov_2022_kwh"] = fl["base"]
        mt["fill_dec2022_kwh"] = np.where(do_fill, fl["fill"], np.nan)
        mt["fill_dec2022_sd_kwh"] = np.where(
            do_fill, fl["base"] * np.sqrt(fl["s"] ** 2 + fl["sigma"] ** 2), np.nan)
    mt["phase_agree"] = np.where(mt["phase_code"].isin(["A", "B"]),
                                 mt["phase_code"] == mt["phase_detected"],
                                 np.where(mt["phase_code"] == "monthly",
                                          mt["phase_detected"] == "monthly", False))
    method_counts = {rl.METHODS[c][0]: int((meth_p == c).sum()) for c in rl.METHODS}
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
        "meters_dec_jan_paired": int(paired.sum()),
        "meters_filled_dec2022": int(do_fill.sum()),
        "meters_paired_without_base": int((paired & ~fl["ok"]).sum()),
        "phase_code_counts": pd.Series(phase_code).value_counts().to_dict(),
        "phase_detected_counts": det["phase_detected"].value_counts().to_dict(),
        "phase_agree_frac_bimonthly": float(mt.loc[mt.phase_code.isin(["A", "B"]),
                                                   "phase_agree"].mean()),
        "phase_agree_frac_monthly": float(mt.loc[mt.phase_code == "monthly",
                                                 "phase_agree"].mean()),
        "seconds": round(time.time() - t0, 1),
    }
    return {"sub": sub, "hp": hp, "share": share, "m_of_p": m_of_p, "Vp": Vp, "sd_p": sd_p,
            "flag_p": flag_p, "meth_p": meth_p, "alt": alt, "cell": cellarr,
            "county": county, "meters": mt, "info": info, "T": T, "fill_info": fill_info}


# ----------------------------------------------------------------------------- save / run
def save(tag: str, variant: str, r: dict, out: Path) -> None:
    sub = r["sub"]
    keep_rows = np.ones(len(sub), bool) if tag != "chiayi" else (sub["cell"].to_numpy() >= 0)
    arrays = dict(
        uid=sub["uid"].to_numpy(dtype=str)[keep_rows], cell=sub["cell"].to_numpy()[keep_rows],
        purpose_class=sub["purpose_class"].to_numpy(dtype=str)[keep_rows],
        meter_idx=sub["meter_idx"].to_numpy()[keep_rows],
        hp=r["hp"][keep_rows], hp_share=r["share"][keep_rows],
        value=r["Vp"][keep_rows], flag=r["flag_p"][keep_rows], method=r["meth_p"][keep_rows],
        dates=np.array([lib.ym(i) for i in range(r["T"])]))
    if variant == "preorigin":
        arrays["sd"] = r["sd_p"][keep_rows]
    lib.save_npz(out / f"{variant}_{tag}_pumps.npz", **arrays)
    if r["alt"] is not None:
        lib.save_npz(out / f"{variant}_{tag}_pumps_evensplit.npz",
                     uid=sub["uid"].to_numpy(dtype=str)[keep_rows], value=r["alt"][keep_rows])
    lib.save_npz(out / f"{variant}_{tag}_cells.npz",
                 **{f"cell_{k}": v for k, v in r["cell"].items()},
                 **{f"county_{k}": v for k, v in r["county"].items()})
    lib.save_npz(out / f"{variant}_{tag}_fill.npz", **r["fill_info"])
    r["meters"].to_parquet(out / f"{variant}_{tag}_meters.parquet", index=False)
    lib.save_json(r["info"], out / f"{variant}_{tag}_info.json")


def load_county(tag: str, grid):
    df = lib.harmonise(tag, lib.read_census_l0(tag), grid)
    k, L = lib.read_kwh_l0(tag, df["sid"].to_numpy(), len(df))
    return df, k, L


def run(tag: str, out: Path, variants=("preorigin", "full"), data=None) -> dict:
    """Build one county; ``data`` = (df, k, L) to build from given (e.g. perturbed) inputs."""
    t0 = time.time()
    grid = lib.load_grid()
    df, k, L = data if data is not None else load_county(tag, grid)
    ctx = preorigin_context(df, k, L)
    res = {}
    share_of = pd.Series(dtype=float)
    for variant in variants:
        r = process(tag, df, k, L, variant, grid, ctx)
        save(tag, variant, r, out)
        res[variant] = r["info"]
        if variant == "preorigin":
            share_of = pd.Series(r["share"], index=r["sub"]["uid"].to_numpy())
        print(tag, variant, {x: r["info"][x] for x in ("pumps_kept", "pumps_kept_in_grid",
                                                         "meters_kept", "seconds")}, flush=True)
        del r
    # L1 rows: pre-origin kept set and duty (the Dec-2022 value of Dec-Jan-paired meters out)
    l1 = df[lib.l1_columns()].copy()
    m = df["meter_idx"].to_numpy()
    l1["meter_hp_sum"] = ctx["hp_m"][m]
    l1["meter_duty_2012_2022"] = ctx["duty"][m]
    l1["kept"] = ctx["kept_meter"][m]
    l1["hp_share"] = df["uid"].map(share_of).to_numpy()
    l1["phase_detected_2012_2022"] = ctx["det"][m]
    l1["dec_jan_paired"] = ctx["paired"][m]
    if tag == "chiayi":
        m_in = set(df.loc[df["in_grid"], "meter_idx"])
        l1 = l1[df["meter_idx"].isin(m_in).to_numpy()]
    l1.to_parquet(out / f"l1_{tag}.parquet", index=False)
    summary = {"county": tag, "seconds": round(time.time() - t0, 1), "variants": res,
               "pumps": int(len(df)), "pumps_in_grid": int(df["in_grid"].sum()),
               "l1_rows": int(len(l1)), "kept_meters": int(ctx["kept_meter"].sum()),
               "dec_jan_paired_meters": int(ctx["paired"].sum()),
               "fill_params": ctx["params"],
               "clean_report": ctx["clean_report"].reset_index().to_dict(orient="records"),
               "env": lib.env_info()}
    lib.save_json(summary, out / f"summary_{tag}.json")
    return summary


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    for tag in argv[1:]:
        if tag not in lib.COUNTIES:
            raise SystemExit(f"unknown county {tag}")
        s = run(tag, out)
        print(tag, "done", s["seconds"], "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
