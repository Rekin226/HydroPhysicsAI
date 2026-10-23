"""H1a stage 3: apply the decisions, aggregate daily and monthly (C20, sigma_obs), events.

Sample flags (plan 2.4): 8 sentinel; 9 any remaining artefact reason; 3 corrected with
neighbour corroboration only; 1 corrected with metadata / co-located backing, or barometric
loading removed; 0 observed and clean. Values: raw - loading - correction (non-sentinel);
``value_raw`` keeps the raw mean of the same samples, so every correction is reversible.

Event bits are rebuilt here from the stage-1 base bits (sentinel, spike, flat run, MAD, far,
floor dropout, below bottom) plus the decided events: dropouts and flag-9 episodes/steps set
their bit; flag-1/3 episodes and steps are corrected instead (episode: the onset jump j1
tapering linearly to the negated return jump -j2; step: the jump on the displaced side).

Carry across the origin: a pre-origin step whose displaced side is "after" and runs to the
end of the pre-origin record keeps its decision on the post-origin rows (same flag, same
correction) until the first post-origin reversal (an offset of opposite sign and matching
size, |j1 + j2| < 0.3 max|j|) or the first post-origin step. A post-origin episode that
starts in the 2022 context at the carried step's jump is then the same displacement and is
not applied twice; post-origin step ranges are clipped to start at the carry's end. No
carry across a record gap longer than ``config.CARRY_MAX_GAP_DAYS``.

Daily value: mean of the hourly means of the samples of the best class present that day
(good = flags 0/1, else flag 3, else flag 9; flag 8 = only sentinels). ``day_valid`` = at
least 12 hours (C20). A daily rolling-median departure screen (gap-fill clean rule: centred
15-day median of valid good days, threshold max(3 m, 8 x MAD(daily diff) x sqrt 7), MAD
fitted on pre-origin days) sets flag 9 on departing days.

Monthly value: mean of valid days with flag 0/1; ``n_clean_days``; ``c20`` (>= 20 days);
``sigma_obs`` = max(0.10, rho(k)) from A0's pre-origin rho(k), linear in k between the knots
3, 5, 10, 15, 20 and the 0.10 floor reached at a complete month (U2). Months with no good
valid day take the flag-3, else the flag-9 days (value given, flag kept); months with no
valid day are flag 8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import screens as S

SPLIT_S = int(pd.Timestamp(C.ORIGIN_SPLIT).value // 10**9)
K_DAILY_DEP = 1024
DAY_KIND_NAMES = dict(S.KIND_NAMES) | {K_DAILY_DEP: "daily_departure"}
ART_BITS = (S.K_SPIKE | S.K_EPISODE | S.K_DROPOUT | S.K_STEP | S.K_FLAT | S.K_MAD | S.K_FAR
            | S.K_DROPFLAT | S.K_BOTTOM)


def sigma_obs(k: np.ndarray, ndays: np.ndarray) -> np.ndarray:
    k = np.asarray(k, float)
    nd = np.asarray(ndays, float)
    knots = sorted(C.RHO_K)
    xs = np.array(knots, float)
    ys = np.array([C.RHO_K[q] for q in knots])
    out = np.interp(np.clip(k, xs[0], xs[-1]), xs, ys)
    hi = k >= xs[-1]
    frac = np.where(nd > xs[-1], (k - xs[-1]) / np.maximum(nd - xs[-1], 1e-9), 1.0)
    out = np.where(hi, ys[-1] + (C.SIGMA_FLOOR - ys[-1]) * np.clip(frac, 0, 1), out)
    return np.maximum(out, C.SIGMA_FLOOR)


# ----------------------------------------------------------------------------- carry
def carry_from_pre(dec_pre: pd.DataFrame, gap_days: float | None = None) -> dict | None:
    """The pre-origin step to carry into the post-origin rows, if any. ``gap_days``: time
    between the last pre-origin and the first post-origin sample; no carry across a gap
    longer than ``config.CARRY_MAX_GAP_DAYS`` (instrument continuity unknown)."""
    if dec_pre is None or dec_pre.empty:
        return None
    if gap_days is not None and gap_days > C.CARRY_MAX_GAP_DAYS:
        return None
    st = dec_pre[dec_pre["kind"] == "step"].sort_values("t1")
    if st.empty:
        return None
    last = st.iloc[-1]
    if last["side"] != "after" or not bool(last.get("to_end", False)):
        return None
    return {"from": SPLIT_S, "until": None, "flag": int(last["flag"]), "corr": float(last["j1"]),
            "j": float(last["j1"]), "t_step": int(last["t1"]), "backing": last.get("backing")}


def carry_until(carry: dict, post_cat: dict | None) -> dict:
    """Set the end of a carry: first post-origin reversal offset or post-origin step."""
    if carry is None or not post_cat:
        return carry
    jc = carry["j"]
    cands = []
    for J in post_cat.get("jumps", []):
        if J["t"] < SPLIT_S or J["cls"] != "offset" or abs(J["j"]) < C.EPISODE_MIN_M:
            continue
        if np.sign(J["j"]) != np.sign(jc) and \
                abs(J["j"] + jc) < C.PAIR_SIZE_TOL * max(abs(J["j"]), abs(jc)):
            cands.append((J["t"], "reversal"))
    for St in post_cat.get("steps", []):
        if St["t"] >= SPLIT_S:
            cands.append((St["t"], "post_step"))
    if cands:
        t, why = min(cands)
        carry = dict(carry, until=int(t), until_reason=why)
    else:
        carry = dict(carry, until_reason="record_end")
    return carry


# ----------------------------------------------------------------------------- apply
def apply_part(part: dict, dec: pd.DataFrame, carry: dict | None = None):
    """Return (value, corr, sflag, kind, applied) arrays/list for one segment."""
    t, v_raw, L = part["t"], part["v_raw"], part["L"]
    kind = part["kind_base"].copy()
    corr = np.zeros(len(t))
    cls = np.zeros(len(t), np.uint8)       # 1 / 3 for corrected samples
    applied = []
    c_lo = carry["from"] if carry else None
    c_hi = (carry["until"] if carry and carry.get("until") is not None else 2**62) if carry \
        else None
    for _, d in (dec.iterrows() if dec is not None else []):
        k, fl = d["kind"], int(d["flag"])
        if k in ("dropout", "offset_episode"):
            if carry and k == "offset_episode" and bool(d.get("spans_origin", False)) and \
                    abs(int(d["t1"]) - carry["t_step"]) <= C.COLOC_WINDOW_S:
                applied.append({"t1": int(d["t1"]), "what": "skipped_same_as_carry", "n": 0})
                continue
            m = (t >= d["t1"]) & (t < d["t2"])
            if not m.any():
                continue
            if k == "offset_episode" and fl in (1, 3):
                frac = (t[m] - d["t1"]) / max(d["t2"] - d["t1"], 1)
                corr[m] += d["j1"] + (-d["j2"] - d["j1"]) * frac
                cls[m] = np.maximum(cls[m], fl)
                what = f"corrected_flag{fl}"
            else:
                kind[m] |= np.uint16(S.K_DROPOUT if k == "dropout" else S.K_EPISODE)
                what = "flagged"
            applied.append({"t1": int(d["t1"]), "what": what, "n": int(m.sum())})
        elif k == "step":
            lo, hi = int(d["t_lo"]), int(d["t_hi"])
            if carry and lo < c_hi and hi >= c_lo:
                lo = max(lo, c_hi)
            m = (t >= lo) & (t <= hi)
            if not m.any():
                applied.append({"t1": int(d["t1"]), "what": "step_range_empty_after_clip",
                                "n": 0})
                continue
            if fl in (1, 3):
                corr[m] += d["j1"] if d["side"] == "after" else -d["j1"]
                cls[m] = np.maximum(cls[m], fl)
                what = f"corrected_flag{fl}"
            else:
                kind[m] |= np.uint16(S.K_STEP)
                what = "flagged"
            applied.append({"t1": int(d["t1"]), "what": what, "n": int(m.sum())})
    if carry:
        m = (t >= c_lo) & (t < c_hi)
        if carry["flag"] in (1, 3):
            corr[m] += carry["corr"]
            cls[m] = np.maximum(cls[m], carry["flag"])
            what = f"carry_corrected_flag{carry['flag']}"
        else:
            kind[m] |= np.uint16(S.K_STEP)
            what = "carry_flagged"
        applied.append({"t1": int(carry["t_step"]), "what": what, "n": int(m.sum())})
    sent = (kind & S.K_SENTINEL) > 0
    value = np.where(sent, np.nan, v_raw - L - corr)
    sflag = np.zeros(len(t), np.uint8)
    sflag[(cls == 1) | (L != 0)] = 1
    sflag[cls == 3] = 3
    sflag[(kind & ART_BITS) > 0] = 9
    sflag[sent] = 8
    return value, corr, sflag, kind, applied


# ----------------------------------------------------------------------------- daily
def daily_table(t, v_raw, L, corr, value, sflag, kind) -> pd.DataFrame:
    if len(t) == 0:
        return pd.DataFrame()
    ts_ = pd.to_datetime(t, unit="s")
    df = pd.DataFrame({"hour": ts_.floor("h"), "value": value, "raw": v_raw, "L": L,
                       "corr": corr, "sflag": sflag, "kind": kind.astype(np.int64)})
    df["day"] = df["hour"].dt.floor("D")
    df["grp"] = np.select([df.sflag <= 1, df.sflag == 3, df.sflag == 9], ["g", "c", "s"], "x")
    df["is1"] = df.sflag == 1
    h = df[df.grp != "x"].groupby(["grp", "day", "hour"]).agg(
        value=("value", "mean"), raw=("raw", "mean"), L=("L", "mean"), corr=("corr", "mean"),
        n=("value", "size"), f1=("is1", "max"))
    d = h.groupby(level=["grp", "day"]).agg(value=("value", "mean"), raw=("raw", "mean"),
                                            L=("L", "mean"), corr=("corr", "mean"),
                                            n_samples=("n", "sum"), n_hours=("value", "size"),
                                            any1=("f1", "max"))
    d = d.reset_index()
    df["s9"] = (df.sflag == 9).astype(np.int64)
    df["s8"] = (df.sflag == 8).astype(np.int64)
    cnt = df.groupby("day").agg(n_suspect=("s9", "sum"), n_sentinel=("s8", "sum"),
                                kinds=("kind", lambda x: int(np.bitwise_or.reduce(x.to_numpy()))))
    d["r"] = d.grp.map({"g": 0, "c": 1, "s": 2})
    best = d.sort_values(["day", "r"]).drop_duplicates("day")
    out = cnt.join(best.set_index("day"), how="left")
    any1 = out.any1.astype("boolean").fillna(False).astype(bool)
    out["flag"] = np.select([out.grp == "g", out.grp == "c", out.grp == "s"],
                            [np.where(any1, 1, 0), 3, 9], 8)
    out.loc[out.flag == 8, ["value", "raw", "L", "corr"]] = np.nan
    out["n_samples"] = out["n_samples"].fillna(0).astype(int)
    out["n_hours"] = out["n_hours"].fillna(0).astype(int)
    out["day_valid"] = out["n_hours"] >= C.DAY_MIN_HOURS
    out = out.reset_index().rename(columns={"day": "date", "raw": "value_raw", "L": "baro_m",
                                            "corr": "corr_m"})
    return out[["date", "value", "n_samples", "n_hours", "flag", "day_valid", "value_raw",
                "baro_m", "corr_m", "n_suspect", "n_sentinel", "kinds"]]


def mad_daily_diff(d: pd.DataFrame) -> float:
    if d is None or d.empty:
        return np.nan
    g = d[(d.flag <= 1) & d.day_valid].set_index("date")["value"].sort_index()
    dd = g.diff().dropna()
    if len(dd) < 10:
        return np.nan
    m = float((dd - dd.median()).abs().median())
    return m if m > 0 else 0.02


def departure_screen(d: pd.DataFrame, mad: float, context: pd.DataFrame | None = None
                     ) -> pd.Series:
    """Boolean Series (index of ``d``) of good valid days departing from the centred 15-day
    rolling median by more than max(3 m, 8 MAD sqrt 7). ``context``: earlier daily rows
    (pre-origin) used only as rolling-window context."""
    good = d[(d.flag <= 1) & d.day_valid]
    s = good.set_index("date")["value"]
    if context is not None and len(context):
        c = context[(context.flag <= 1) & context.day_valid]
        s = pd.concat([c.set_index("date")["value"], s])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 10 or not np.isfinite(mad):
        return pd.Series(False, index=d.index)
    roll = s.rolling(f"{C.DAILY_ROLL_DAYS}D", center=True, min_periods=5).median()
    thr = max(C.DAILY_DEP_MIN_M, C.DAILY_DEP_MAD_MULT * mad * np.sqrt(7))
    bad = (s - roll).abs() > thr
    bad_days = set(bad[bad].index)
    return d["date"].isin(bad_days) & (d.flag <= 1) & d.day_valid


def apply_departures(d: pd.DataFrame, bad: pd.Series) -> pd.DataFrame:
    d = d.copy()
    d.loc[bad, "flag"] = 9
    d.loc[bad, "kinds"] = d.loc[bad, "kinds"].astype(np.int64) | K_DAILY_DEP
    return d


# ----------------------------------------------------------------------------- monthly
def monthly_table(d: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """One row per month of ``months`` (flag 8 where there is no valid day)."""
    rows = []
    if d is None or d.empty:
        g_all = {}
    else:
        dd = d.copy()
        dd["month"] = dd["date"].dt.to_period("M")
        g_all = dict(tuple(dd.groupby("month")))
    for m in months:
        nd = m.days_in_month
        g = g_all.get(m)
        base = {"month": str(m), "date": m.start_time, "days_in_month": nd}
        if g is None or g.empty:
            rows.append(base | {"value": np.nan, "n_clean_days": 0, "n_valid_days_any": 0,
                                "flag": 8, "c20": False, "sigma_obs": np.nan,
                                "value_raw": np.nan, "baro_m": np.nan, "corr_m": np.nan,
                                "n_days_flag9": 0, "n_days_flag3": 0, "n_days_any": 0})
            continue
        good = g[(g.flag <= 1) & g.day_valid]
        c3 = g[(g.flag == 3) & g.day_valid]
        sus = g[(g.flag == 9) & g.day_valid]
        common = {"n_valid_days_any": int(g.day_valid.sum()),
                  "n_days_flag9": int((g.flag == 9).sum()),
                  "n_days_flag3": int((g.flag == 3).sum()), "n_days_any": int(len(g))}
        if len(good):
            use, flag = good, (1 if (good.flag == 1).any() else 0)
        elif len(c3):
            use, flag = c3, 3
        elif len(sus):
            use, flag = sus, 9
        else:
            rows.append(base | common | {"value": np.nan, "n_clean_days": 0, "flag": 8,
                                         "c20": False, "sigma_obs": np.nan, "value_raw": np.nan,
                                         "baro_m": np.nan, "corr_m": np.nan})
            continue
        k = len(use)
        rows.append(base | common | {
            "value": float(use.value.mean()), "n_clean_days": int(len(good)), "flag": int(flag),
            "c20": bool(len(good) >= C.MONTH_MIN_DAYS),
            "sigma_obs": float(sigma_obs(np.array([k]), np.array([nd]))[0]),
            "value_raw": float(use.value_raw.mean()), "baro_m": float(use.baro_m.mean()),
            "corr_m": float(use.corr_m.mean())})
    return pd.DataFrame(rows)


def kinds_to_names(bits: int) -> str:
    return ";".join(n for b, n in DAY_KIND_NAMES.items() if int(bits) & b)
