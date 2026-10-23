"""P2c: zero classification (port of pilot s05 section (e), the s05 pre-install count and the
s09 synchronized-zero screen), with the pilot reproduction check (pre-registered:
P2c.zero_reproduction.v1) and the per-variant product classification.

Product classes (kept meters; one row per zero meter-month of the variant's window; NaN months
outside the meter's first..last use span count as zeros, as they enter the twin as 0):

  structural      before the meter's install month (no pump yet)
  not_yet_used    after install, before the first use (leading run)
  idle            interior run <= 12 months (real idling; irrigation zeros are seasonal)
  long_idle       interior run > 12 months that ends in renewed use (real zero)
  terminated      trailing run >= 12 months to the window end: a real zero for this meter,
                  ``possible_replacement`` True (abstraction may continue on an unseen meter)
  trailing_short  trailing run < 12 months at the window end (undetermined)
  never_used      no use in the window and none before it (or none at all)

``sync_suspect`` marks zero meter-months in months flagged by the synchronized-zero screen
(zero share among meters active two months before and after exceeds its calendar-month median
by > 3 points) where the meter itself was active at t-2 and t+2; those rows carry flag 9 and
enter only a sensitivity member (plan P2c). All other rows carry flag 0.

Writes ``_work_p2abc/zeros_<variant>.parquet``, ``meter_zero_summary_<variant>.parquet``,
``zeros.json``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

BINS = [0, 1, 2, 4, 6, 12, 24, 1000]
BIN_LABELS = ["1", "2", "3-4", "5-6", "7-12", "13-24", ">24"]
WINDOW = {"preorigin": ("2012-01", "2022-12"), "full": ("2012-01", "2025-06")}
ZCLASSES = ["structural", "not_yet_used", "idle", "long_idle", "terminated", "trailing_short",
            "never_used"]


def runs_of(z: np.ndarray):
    d = np.diff(np.concatenate([[0], z.astype(int), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


# ----------------------------------------------------------------------------- pilot check
def pilot_reproduction(mt: pd.DataFrame, M: np.ndarray) -> dict:
    tab = pd.read_parquet(lib.WORK / "full_phase.parquet")
    phase = tab["phase"].to_numpy()
    T = lib.T_ALL
    out = {}
    # pre-install months (s05): meters with install index > 12, months [0, ti-1)
    ti = np.array([lib.DATES.searchsorted(d) if pd.notna(d) else -1 for d in mt["inst"]])
    pz = pn = pp = 0
    for j in np.where(ti > 12)[0]:
        seg = M[j, : ti[j] - 1]
        pz += int((seg == 0).sum())
        pn += int(np.isnan(seg).sum())
        pp += int((seg > 0).sum())
    out["pre_install_months"] = {"zero": pz, "nan": pn, "positive": pp}
    # zero runs (s05 e)
    keep = mt["kept"].to_numpy()
    kz = keep & np.isin(phase, [lib.PHASE_A, lib.PHASE_B, lib.MONTHLY])
    Mz = np.nan_to_num(M)
    nz = Mz > 0
    first_nz = np.where(nz.any(1), nz.argmax(1), -1)
    last_nz = np.where(nz.any(1), T - 1 - nz[:, ::-1].argmax(1), -1)
    W0, W1 = lib.tidx("2012-01"), lib.tidx("2025-06") + 1
    rows = []
    moy_zero = np.zeros((3, 12))
    moy_cnt = np.zeros((3, 12))
    cls_idx = {"irrigation": 0, "aquaculture": 1, "livestock": 2}
    cls = mt["cls"].to_numpy()
    mo = lib.DATES[W0:W1].month.to_numpy() - 1
    for j in np.where(kz)[0]:
        x = M[j, W0:W1]
        if not np.isfinite(x).any() or np.nansum(x) <= 0:
            continue
        f, l_ = first_nz[j] - W0, last_nz[j] - W0
        z = x == 0
        for s_, e_ in zip(*runs_of(z)):
            kind = "leading" if s_ == 0 and f > 0 else (
                "trailing" if e_ == len(x) and l_ < len(x) - 1 else "interior")
            rows.append((cls[j], kind, e_ - s_))
        ci = cls_idx.get(cls[j])
        if ci is not None:
            inside = np.zeros(len(x), bool)
            inside[max(f, 0): l_ + 1] = True
            ok = inside & np.isfinite(x)
            np.add.at(moy_zero[ci], mo[ok], z[ok])
            np.add.at(moy_cnt[ci], mo[ok], 1)
    runs = pd.DataFrame(rows, columns=["cls", "kind", "len"])
    runs["len_bin"] = pd.cut(runs["len"], BINS, labels=BIN_LABELS)
    total = int(runs["len"].sum())
    share = runs.groupby(["kind", "len_bin"], observed=True)["len"].sum() / total
    out["zero_months_total"] = total
    out["month_share_kind_bin"] = {f"{k}|{b}": float(v) for (k, b), v in share.items()}
    out["interior_le12_share"] = float(runs.loc[(runs.kind == "interior") & (runs.len <= 12),
                                                "len"].sum() / total)
    out["trailing_ge12_share"] = float(runs.loc[(runs.kind == "trailing") & (runs.len >= 12),
                                                "len"].sum() / total)
    zmoy = moy_zero / np.maximum(moy_cnt, 1)
    out["irrigation_zero_moy"] = zmoy[0].tolist()
    counts = runs.groupby(["cls", "kind", "len_bin"], observed=True).size()
    pil = pd.read_csv(lib.PILOT / "zero_runs_summary.csv")
    pil = pil.set_index(["cls", "kind", "len_bin"])["0"]
    both = pd.concat([counts.rename("mine"), pil.rename("pilot")], axis=1).fillna(0)
    rel = (both["mine"] - both["pilot"]).abs() / both["pilot"].clip(lower=1)
    out["run_table_cells"] = int(len(both))
    out["run_table_max_rel_diff"] = float(rel.max())
    pz_pilot = 878738
    checks = {
        "pre_install_zero": abs(pz - pz_pilot) / pz_pilot <= 0.01,
        "zero_months_total": abs(total - 2124833) / 2124833 <= 0.01,
        "run_table": float(rel.max()) <= 0.01,
        "interior_le12_share": abs(out["interior_le12_share"] - 0.579) / 0.579 <= 0.01,
        "irrigation_april": abs(zmoy[0][3] - 0.0558) / 0.0558 <= 0.01,
        "irrigation_august": abs(zmoy[0][7] - 0.1563) / 0.1563 <= 0.01,
    }
    out["checks"] = {k: bool(v) for k, v in checks.items()}
    out["passed"] = bool(all(checks.values()))
    out["sync_flags_full"] = sync_screen(mt, M, phase, T - 1)[1]
    return out


# ----------------------------------------------------------------------------- screen
def sync_screen(mt: pd.DataFrame, M: np.ndarray, phase: np.ndarray, end: int,
                thresh: float = 0.03):
    """s09: share of meters zero at t although active at t-2 and t+2, by group; months whose
    share exceeds the calendar-month median by more than ``thresh``. Uses months <= end."""
    keep = mt["kept"].to_numpy()
    bill = mt["bill"].to_numpy()
    isA = keep & (bill == "M2") & (phase == lib.PHASE_A)
    isB = keep & (bill == "M1") & (phase == lib.PHASE_B)
    isM = keep & (phase == lib.MONTHLY)
    X = np.nan_to_num(M.astype(float))
    rows = []
    for t in range(2, end - 1):
        m = isM & (X[:, t - 2] > 0) & (X[:, t + 2] > 0)
        bm = (isA | isB) & (X[:, t - 2] > 0) & (X[:, t + 2] > 0)
        rows.append((lib.DATES[t], float(np.mean(X[m, t] == 0)) if m.sum() else np.nan,
                     float(np.mean(X[bm, t] == 0)) if bm.sum() else np.nan))
    sy = pd.DataFrame(rows, columns=["date", "zero_monthly", "zero_bimonthly"]).set_index("date")
    an = sy - sy.groupby(sy.index.month).transform("median")
    flagged = sy[(an > thresh).any(axis=1)]
    return flagged.index, [d.strftime("%Y-%m") for d in flagged.index]


# ----------------------------------------------------------------------------- product
def classify(name: str, mt: pd.DataFrame, M: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                    dict]:
    tab = pd.read_parquet(lib.WORK / f"{name}_phase.parquet")
    phase = tab["phase"].to_numpy()
    w0, w1 = (lib.tidx(m) for m in WINDOW[name])
    data_end = w1                                    # last month of data classified
    vend = lib.tidx(lib.VARIANTS[name]["end"])
    # open edge pair (pre-origin: phase-B December 2022) and incomplete pairs are NaN
    Mv, edge, _inc = lib.masked_matrix(M, mt["bill"].to_numpy(), phase, vend)
    X = Mv[:, :data_end + 1]
    flagged_idx, flagged = sync_screen(mt, X, phase, data_end)
    fl_t = np.array([lib.DATES.get_loc(d) for d in flagged_idx], int)
    keep = np.where(mt["kept"].to_numpy())[0]
    first, last = lib.spans(X)
    inst_m = np.array([lib.DATES.searchsorted(d.to_period("M").to_timestamp())
                       if pd.notna(d) else -1 for d in mt["inst"]])
    tt = np.arange(w0, w1 + 1)
    L = len(tt)
    recs = []
    summ = []
    Xz = np.nan_to_num(X)
    for j in keep:
        x = X[j, w0:w1 + 1].copy()
        f, l_ = first[j], last[j]
        inspan = (tt >= f) & (tt <= l_) if f >= 0 else np.zeros(L, bool)
        # an open edge month is unknown, not idle: the meter's window ends one month early
        L_j, wend = (L - 1, w1 - 1) if (edge[j] and data_end == vend) else (L, w1)
        x[np.isnan(x) & ~inspan] = 0.0                # out-of-span NaN enters the twin as 0
        x[L_j:] = np.nan
        z = x == 0
        if not z.any():
            summ.append((j, f, l_, False, -1))
            continue
        zc = np.full(L, -1, int)
        rl = np.zeros(L, int)
        for s_, e_ in zip(*runs_of(z)):
            n = e_ - s_
            rl[s_:e_] = n
            if f < 0 or f > wend:
                c = 6                                                   # never_used
            elif s_ == 0 and f > w0:
                c = 1                                                   # not_yet_used
            elif e_ == L_j and l_ < wend:
                c = 4 if n >= 12 else 5                                 # terminated / short
            else:
                c = 2 if n <= 12 else 3                                 # idle / long_idle
            zc[s_:e_] = c
        pre = z & (tt < inst_m[j]) if inst_m[j] >= 0 else np.zeros(L, bool)
        zc[pre] = 0                                                     # structural
        idx = np.where(z)[0]
        t_abs = tt[idx]
        sync = np.isin(t_abs, fl_t) & (zc[idx] != 0)
        if sync.any():
            ok = (t_abs - 2 >= 0) & (t_abs + 2 <= data_end)
            act = np.zeros(len(idx), bool)
            act[ok] = (Xz[j, t_abs[ok] - 2] > 0) & (Xz[j, t_abs[ok] + 2] > 0)
            sync &= act
        recs.append((np.full(len(idx), j, np.int32), t_abs.astype(np.int16),
                     zc[idx].astype(np.int8), rl[idx].astype(np.int16), sync))
        term = bool(zc[L_j - 1] == 4)
        summ.append((j, f, l_, term, int(l_ + 1) if term else -1))
    z = pd.DataFrame({k: np.concatenate([r[i] for r in recs]) for i, k in
                      enumerate(("meter_id", "t", "zclass", "run_len", "sync_suspect"))})
    z["month"] = lib.DATES[z["t"].to_numpy()].strftime("%Y-%m")
    z["zclass"] = pd.Categorical.from_codes(z["zclass"], ZCLASSES)
    z["flag"] = np.where(z["sync_suspect"], 9, 0).astype(np.uint8)
    z["possible_replacement"] = z["zclass"] == "terminated"
    z["run_len"] = z["run_len"].astype(np.int16)
    s = pd.DataFrame(summ, columns=["meter_id", "first_nz_t", "last_nz_t", "terminated",
                                    "termination_t"])
    # energy of the last 12 used months before termination (survivorship lower bound, P2d)
    e12 = np.zeros(len(s))
    for i, (j, l_, term) in enumerate(zip(s["meter_id"], s["last_nz_t"], s["terminated"])):
        if term:
            e12[i] = Xz[j, max(0, l_ - 11): l_ + 1].sum()
    s["last12_kwh_before_termination"] = e12
    for c in ("first_nz", "last_nz", "termination"):
        t_ = s[f"{c}_t"].to_numpy()
        s[c] = np.where(t_ >= 0, lib.DATES[np.clip(t_, 0, None)].strftime("%Y-%m"), "")
    counts = z["zclass"].value_counts().reindex(ZCLASSES).fillna(0).astype(int)
    info = {"window": list(WINDOW[name]), "n_kept_meters": int(len(keep)),
            "zero_meter_months": int(len(z)), "class_counts": counts.to_dict(),
            "class_shares": (counts / counts.sum()).round(5).to_dict(),
            "sync_flagged_months": flagged, "sync_suspect_rows": int(z["sync_suspect"].sum()),
            "n_terminated_meters": int(s["terminated"].sum()),
            "terminated_last12_GWh": float(s["last12_kwh_before_termination"].sum() / 1e6)}
    # irrigation zero seasonality within span (product definition: kept irrigation meters)
    irr = set(np.where(mt["cls"].to_numpy() == "irrigation")[0])
    zi = z[z["meter_id"].isin(irr) & z["zclass"].isin(["idle", "long_idle"])]
    info["irrigation_idle_zero_rows_by_moy"] = zi.groupby(
        lib.DATES[zi["t"].to_numpy()].month).size().to_dict()
    return z, s, info


def main() -> None:
    mt, M, _ = lib.load_work()
    res = {"pilot_reproduction": pilot_reproduction(mt, M)}
    print({k: v for k, v in res["pilot_reproduction"].items()
           if k not in ("month_share_kind_bin", "irrigation_zero_moy")}, flush=True)
    for name in WINDOW:
        z, s, info = classify(name, mt, M)
        z.drop(columns=["t"]).to_parquet(lib.WORK / f"zeros_{name}.parquet", index=False)
        s.to_parquet(lib.WORK / f"meter_zero_summary_{name}.parquet", index=False)
        res[name] = info
        print(name, info, flush=True)
    lib.save_json(res, lib.WORK / "zeros.json")


if __name__ == "__main__":
    main()
