"""P2c zero classification, repair round: the classifier that writes ``zero_class.parquet``
and ``meter_zero_summary.parquet`` of yunlin_electricity/{preorigin,full}/v2, and the same
function the registered check P2r3b.c1 scores (``classify_core``).

Run kinds follow the billing pilot's s05 section (e) exactly
(``results/dataprep/_pilots/billing_and_census_gaps/s05_census_zeros.py``):

* zero = a reading exactly 0; a month that is not an observation (missing reading, flag 8;
  C0's December-2022 fill, flag 4; a partial-cycle or suspect month, flag 9) is NaN: not a
  zero, and it breaks a zero run; a bimonthly bill posted wholly to its second month (C0
  method bill_in_second_month: from 2025-07 for 12-33 % of bills, and a few thousand pairs
  before, mostly the 2021-12/2022-01 phase-B pair) is split evenly first, so its first month
  is not a zero reading (repair-round decision; the pilot's data had no such postings in its
  window except those few pairs);
* ``first`` / ``last`` = the meter's first and last positive month over the whole record of the
  variant (not only the window);
* within the window, a zero run is *leading* if it starts at the window start and the first
  use lies after the window start, *trailing* if it ends at the window end and the last use
  lies before the window's last month, else *interior*; the window of a meter is trimmed to
  its first..last observed month inside the window (amendment P2abc.r3.repair.v2: identical
  to the pilot's rule where the edge months are observations; needed where they are not,
  e.g. the pre-origin 2022-12 of Dec-Jan-paired meters, C0's flag-4 fill);
* a meter with no positive reading in the window is skipped by the pilot; here its zero months
  are kept with kind ``no_use_in_window`` (outside the pilot's counts).

Product classes (``zclass``): leading -> ``not_yet_used``; interior <= 12 months -> ``idle``
(real idling; irrigation zeros are seasonal); interior > 12 -> ``long_idle``; trailing >= 12
-> ``terminated`` (a real zero for this meter; ``possible_replacement`` True: abstraction may
continue on an unseen meter); trailing < 12 -> ``trailing_short`` (undetermined);
no_use_in_window -> ``never_used_in_window``.

Structural zeros (plan P2c: before the install date) are a separate boolean column
``structural`` = C0 method ``pre_install`` of that meter-month (C0 codes every pre-install
month 0); it does not override the run class, so the run classes keep the pilot's definition.

Synchronized zeros: ``sync_suspect`` marks zero meter-months in the months the county's
synchronized-zero screen flags (round 2's screen algorithm, ``v2/b2_zeros.sync_screen``,
unchanged, on the same observed matrix as the classifier:
zero share among meters active two months before and after exceeds its calendar-month median
by more than 3 points in the monthly or the bimonthly group; Chiayi screened with the pooled
Yunlin + Chiayi population) where the meter itself was active at t-2 and t+2; structural
(pre-install) zero months are never marked (as round 2). Repair-round change: for a bimonthly
meter (detected phase A or B) the mark is made pair-consistent - a
marked zero month also marks the other month of the same bill when that month is a zero
too (round 2 flagged 2025-08 but not 2025-07 of the same zero Jul-Aug bills). Marked rows carry
flag 9 (plan P2c: they stay in the data and enter as a sensitivity member only).

Usage: python -I zeros3.py <out_dir>   (writes zeros_<variant>.parquet,
       meter_zero_summary_<variant>.parquet, zeros_build.json)
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
import b2_zeros as B2  # noqa: E402  (v2, read-only: sync_screen, screen_input, WINDOW)
from p3lib import L  # noqa: E402

WINDOW = B2.WINDOW                    # pre-origin 2012-01..2022-12, full 2012-01..2026-06
KINDS = ("leading", "interior", "trailing", "no_use_in_window")
ZCLASSES = ("not_yet_used", "idle", "long_idle", "terminated", "trailing_short",
            "never_used_in_window")


def runs_of(z: np.ndarray):
    d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def first_last(X: np.ndarray):
    nz = np.nan_to_num(X) > 0
    T = X.shape[1]
    first = np.where(nz.any(1), nz.argmax(1), -1)
    last = np.where(nz.any(1), T - 1 - nz[:, ::-1].argmax(1), -1)
    return first, last


def zclass_of(kind: np.ndarray, run_len: np.ndarray) -> np.ndarray:
    out = np.full(len(kind), -1, np.int8)
    out[kind == 0] = 0
    out[(kind == 1) & (run_len <= 12)] = 1
    out[(kind == 1) & (run_len > 12)] = 2
    out[(kind == 2) & (run_len >= 12)] = 3
    out[(kind == 2) & (run_len < 12)] = 4
    out[kind == 3] = 5
    assert (out >= 0).all()
    return out


def classify_core(X: np.ndarray, w0: int, w1: int) -> dict:
    """Zero meter-months of X (n, T; NaN = not an observation) in the window [w0, w1]
    (inclusive month indices). Returns arrays row, t, kind (index into KINDS), run_len,
    zclass (index into ZCLASSES), and per meter first, last, use_in_window, obs_lo, obs_hi.

    Window edges (amendment P2abc.r3.repair.v2): a meter's window runs from its first to its
    last observed month inside [w0, w1] (obs_lo..obs_hi); a run is leading if it starts at
    obs_lo and the first use lies after obs_lo, trailing if it ends at obs_hi and the last use
    lies before obs_hi. Where both edge months are observations (every meter of the pilot's
    data but a handful with missing readings) this is the pilot's rule; it differs where an
    edge month is not an observation (pre-origin: 2022-12 of Dec-Jan-paired meters is C0's
    flag-4 fill), which would otherwise make every trailing run of such a meter interior."""
    first, last = first_last(X)
    W = X[:, w0:w1 + 1]
    Lw = W.shape[1]
    fin = np.isfinite(W)
    use = fin.any(1) & (np.nansum(W, axis=1) > 0)
    obs_lo = np.where(fin.any(1), fin.argmax(1), -1)
    obs_hi = np.where(fin.any(1), Lw - 1 - fin[:, ::-1].argmax(1), -1)
    rows, ts, kinds, rls = [], [], [], []
    Z = W == 0                                    # NaN -> False
    for j in np.where(Z.any(1))[0]:
        z = Z[j]
        s, e = runs_of(z)
        n = e - s
        if not use[j]:
            k = np.full(len(s), 3, np.int8)
        else:
            f, l_ = first[j] - w0, last[j] - w0
            lo, hi = obs_lo[j], obs_hi[j]
            k = np.where((s == lo) & (f > lo), 0,
                         np.where((e == hi + 1) & (l_ < hi), 2, 1)).astype(np.int8)
        for a, b, kk, nn in zip(s, e, k, n, strict=True):
            rows.append(np.full(b - a, j, np.int32))
            ts.append(np.arange(a, b, dtype=np.int32) + w0)
            kinds.append(np.full(b - a, kk, np.int8))
            rls.append(np.full(b - a, nn, np.int16))
    if rows:
        row, t = np.concatenate(rows), np.concatenate(ts)
        kind, rl = np.concatenate(kinds), np.concatenate(rls)
    else:
        row = t = np.zeros(0, np.int32)
        kind, rl = np.zeros(0, np.int8), np.zeros(0, np.int16)
    return {"row": row, "t": t, "kind": kind, "run_len": rl, "zclass": zclass_of(kind, rl),
            "first": first, "last": last, "use_in_window": use,
            "obs_lo": np.where(obs_lo >= 0, obs_lo + w0, -1),
            "obs_hi": np.where(obs_hi >= 0, obs_hi + w0, -1)}


def observed_matrix(cd: L.CountyData, t_end: int) -> np.ndarray:
    """The county's meter record to month index t_end (inclusive) as the classifier and the
    synchronized-zero screen see it: every month that is not an observation is NaN (flags 4,
    8, 9), and a bimonthly bill that TPC posted wholly to its second month (C0 method
    bill_in_second_month on both months of the pair, pairs by C0's billing-code phase) is
    split evenly, as TPC split every other bill: its first month is not a zero reading."""
    X = cd.M[:, :t_end + 1].astype(np.float64).copy()
    F = cd.F[:, :t_end + 1]
    Mm = cd.Mm[:, :t_end + 1]
    X[(F == 4) | (F == 8) | (F == 9)] = np.nan
    pc = cd.meters["phase_code"].astype(str).to_numpy()
    m7 = Mm == 7
    for ph, s0 in (("A", 0), ("B", 1)):
        rows = np.where(pc == ph)[0]
        st = np.arange(s0, t_end, 2)
        st = st[st + 1 <= t_end]
        both = m7[np.ix_(rows, st)] & m7[np.ix_(rows, st + 1)]
        r, k = np.nonzero(both)
        rr, tt = rows[r], st[k]
        half = (X[rr, tt] + X[rr, tt + 1]) / 2
        X[rr, tt] = half
        X[rr, tt + 1] = half
    return X


def screen_input3(cd: L.CountyData, variant: str):
    """(code, phase, X) of a county for the synchronized-zero screen (round 2's screen input
    with the repair round's observed_matrix: non-observations NaN, second-month postings split
    evenly), months to the window end."""
    return cd.code, cd.phase, observed_matrix(cd, L.tidx(WINDOW[variant][1]))


def sync_marks(cd: L.CountyData, X: np.ndarray, res: dict, fl_t: np.ndarray,
               pre: np.ndarray):
    """sync_suspect per zero row (see module docstring), pair-consistent for bimonthly
    meters; structural (pre-install) zero rows are never marked (as round 2)."""
    row, t = res["row"], res["t"]
    Z = np.nan_to_num(X)
    T = X.shape[1]
    sync = np.isin(t, fl_t) & ~pre
    ok = sync & (t - 2 >= 0) & (t + 2 < T)
    act = np.zeros(len(row), bool)
    act[ok] = (Z[row[ok], t[ok] - 2] > 0) & (Z[row[ok], t[ok] + 2] > 0)
    sync &= act
    # pair consistency: the other month of the same bill, if it is a zero row too
    ph = cd.phase
    s0 = np.where(ph[row] == "A", 0, np.where(ph[row] == "B", 1, -1))
    bim = s0 >= 0
    start = np.where((t - s0) % 2 == 0, t, t - 1)
    other = np.where(start == t, t + 1, t - 1)
    key = row.astype(np.int64) * (T + 1) + t
    marked = set(key[sync].tolist())
    okey = row.astype(np.int64) * (T + 1) + other
    add = bim & ~sync & ~pre & np.array([k in marked for k in okey.tolist()], bool) \
        if len(row) else np.zeros(0, bool)
    return sync | add, int(add.sum())


def classify(cd: L.CountyData, variant: str, screen=None):
    """Product zero table and meter summary of one county and variant."""
    w0, w1 = (L.tidx(m) for m in WINDOW[variant])
    T_rec = cd.T if variant == "preorigin" else L.tidx("2026-07") + 1
    X = observed_matrix(cd, T_rec - 1)
    res = classify_core(X, w0, w1)
    sc = screen if screen is not None else screen_input3(cd, variant)
    fl_t, flagged = B2.sync_screen(sc[0], sc[1], sc[2], w1)
    row, t = res["row"], res["t"]
    pre = cd.Mm[row, t] == 1
    sync, n_pair_added = sync_marks(cd, X, res, fl_t, pre)
    z = pd.DataFrame({
        "county": cd.county,
        "meter_idx": cd.meters["meter_idx"].to_numpy()[row],
        "month": L.DATES[t].strftime("%Y-%m"),
        "zkind": pd.Categorical.from_codes(res["kind"], KINDS),
        "run_len": res["run_len"],
        "zclass": pd.Categorical.from_codes(res["zclass"], ZCLASSES),
        "structural": pre,
        "sync_suspect": sync,
        "flag": np.where(sync, 9, 0).astype(np.uint8),
    })
    z["possible_replacement"] = (z["zclass"] == "terminated").to_numpy()
    # meter summary (terminated = a trailing zero run >= 12 months, the pilot's rule)
    n = len(cd.meters)
    term_t = np.full(n, -1, np.int64)
    tr = res["zclass"] == 3
    if tr.any():
        g = pd.Series(t[tr]).groupby(row[tr]).min()
        term_t[g.index.to_numpy()] = g.to_numpy()
    first, last = res["first"], res["last"]
    Zf = np.nan_to_num(X)
    e12 = np.zeros(n)
    for j in np.where(term_t >= 0)[0]:
        e12[j] = Zf[j, max(0, last[j] - 11): last[j] + 1].sum()
    s = pd.DataFrame({"county": cd.county, "meter_idx": cd.meters["meter_idx"].to_numpy(),
                      "first_nz_t": first, "last_nz_t": last,
                      "use_in_window": res["use_in_window"],
                      "terminated": term_t >= 0, "termination_t": term_t,
                      "last12_kwh_before_termination": e12,
                      "c0_terminated": cd.meters["terminated"].to_numpy()})
    for c in ("first_nz", "last_nz", "termination"):
        t_ = s[f"{c}_t"].to_numpy()
        s[c] = np.where(t_ >= 0, L.DATES[np.clip(t_, 0, None)].strftime("%Y-%m"), "")
    counts = z["zclass"].value_counts().reindex(list(ZCLASSES)).fillna(0).astype(int)
    info = {"window": list(WINDOW[variant]), "record_end": L.ym(T_rec - 1),
            "n_kept_meters": int(n), "zero_meter_months": int(len(z)),
            "class_counts": counts.to_dict(),
            "structural_rows": int(z["structural"].sum()),
            "structural_by_class": z[z["structural"]]["zclass"].value_counts()
            .reindex(list(ZCLASSES)).fillna(0).astype(int).to_dict(),
            "sync_flagged_months": flagged, "sync_suspect_rows": int(sync.sum()),
            "sync_rows_added_by_pair_rule": n_pair_added,
            "n_terminated_meters": int(s["terminated"].sum()),
            "terminated_last12_GWh": float(e12.sum() / 1e6),
            "agreement_with_c0_terminated": float(np.mean(s["terminated"].to_numpy()
                                                          == s["c0_terminated"].to_numpy()))}
    return z, s, info


def build(out_dir: str, counties=None) -> dict:
    """``counties``: debug runs on a subset (Yunlin is still loaded for Chiayi's screen)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counties = tuple(counties or L.COUNTIES)
    res = {}
    for variant in ("preorigin", "full"):
        prod = L.open_c0(variant)
        need = set(counties) | ({"yunlin"} if "chiayi" in counties else set())
        cds = {c: L.CountyData(variant, c, prod=prod) for c in L.COUNTIES if c in need}
        zs, ss = [], []
        res[variant] = {}
        for c in counties:
            screen = None
            if c == "chiayi":
                a = screen_input3(cds["yunlin"], variant)
                b = screen_input3(cds[c], variant)
                screen = tuple(np.concatenate([x, y]) for x, y in zip(a, b, strict=True))
            z, s, info = classify(cds[c], variant, screen)
            info["screen_population"] = "yunlin+chiayi" if c == "chiayi" else c
            zs.append(z)
            ss.append(s)
            res[variant][c] = info
            print(variant, c, info["class_counts"], info["sync_flagged_months"],
                  info["sync_rows_added_by_pair_rule"], flush=True)
        pd.concat(zs, ignore_index=True).to_parquet(out / f"zeros_{variant}.parquet",
                                                    index=False)
        pd.concat(ss, ignore_index=True).to_parquet(
            out / f"meter_zero_summary_{variant}.parquet", index=False)
    return res


def main(out_dir: str, counties=None) -> None:
    t0 = time.time()
    res = build(out_dir, counties)
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, Path(out_dir) / "zeros_build.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
