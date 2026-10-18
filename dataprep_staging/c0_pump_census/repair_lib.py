"""WP C0 repair round (v2 products): the pre-origin variant carries no post-origin information.

The verifier's critical finding: in ``pump_electricity/preorigin/v1`` the 2022-12 value of every
phase-B meter is half of a December-2022/January-2023 bill read in 2023 (TPC even split), so
about 15 % of the in-grid December-2022 forcing is January-2023 consumption, and the kept set's
duty statistic also saw it. v2 removes that information everywhere in the pre-origin variant:

* **Dec-Jan-paired meters** (:func:`dec_jan_paired`, pre-origin information only): the meter's
  identical-pair phase detected on the pairs lying wholly inside 2012-01..2022-11 (no pair that
  involves a December-2022 value) is B, or its billing code is M1 and the detection is
  inconclusive (mixed, insufficient, both-flat). The rule was fixed after a design look at
  pre-origin counts only (code x detection cross-tab; logged).
* **Duty statistic** (kept set, both variants and the L1 table): the mean over 2012-01..2022-12
  of the observed pre-origin readings, with the Dec-2022 value of Dec-Jan-paired meters left
  out (it is not a pre-origin observation). Both variants share this pre-origin kept set, so
  the recorded-forcing arm differs from the frozen arm only in post-origin forcing.
* **December-2022 fill** (pre-origin variant only; flag 4 "filled from the series' own temporal
  model", method ``fill_dec2022_dec_jan_pair``): for a Dec-Jan-paired kept meter,
  ``fill = rho_bar(county, class) x (Oct-2022 + Nov-2022) / 2``, where ``rho_bar`` is the mean over
  2012-2021 of the ratio of sums ``sum(Dec_y + Jan_y+1) / sum(Oct_y + Nov_y)`` over the county's
  Dec-Jan-paired kept meters of that class (installed by October y, all four readings finite;
  the county's all-class ratio where a class has fewer than :data:`N_MIN_CLASS` such meters in
  any year). The fill estimates the convention value of every other December of these meters
  (the even-split half of the Dec-Jan pair) from data read by 2022-12 only. A meter without
  an Oct-Nov 2022 base stays missing (flag 8). Fixed before any back-test number was seen.
* **sd of the fill** (two components, fixed blind): a common factor, the inter-annual sd ``s``
  of the ratio (2012-2021, ddof 1), taken as fully correlated across meters, classes and
  counties; and an idiosyncratic meter term ``sigma`` (county-pooled), with
  ``sigma^2 = sum (a - rho_y b)^2 / sum b^2`` over meter-years (a = Dec-Jan half, b = Oct-Nov
  half). For a group of pumps: ``sd^2 = (sum_p s_p b_p)^2 + sum_meters (sigma sum_p b_p)^2``
  with ``b_p`` = HP share x meter base. Its coverage is back-tested leave-one-year-out on
  2012-2021 (pre-registered); an sd that fails the back-test is labelled "uncalibrated".

Nothing else changes relative to v1 (cleaning, classes, cells, masks, phases, termination,
even-split alternative), except two additions: per-cell ``missing_expected_kwh`` (HP x the
same-month county-class mean kWh per HP of reporting pumps, for post-install missing readings,
which the value counts as zero; an indicator, not a fill) and the meter-level fill columns.
"""
from __future__ import annotations

import lib
import numpy as np
import pandas as pd

J_DEC22 = lib.tidx("2022-12")
T_PRE = J_DEC22 + 1
FIT_YEARS = tuple(range(2012, 2022))      # Dec y / Jan y+1 pairs read by 2022-01
N_MIN_CLASS = 30
INCONCLUSIVE = ("mixed", "insufficient", "both-flat")
FILL_METHOD = 9
METHODS = {**lib.METHODS, FILL_METHOD: ("fill_dec2022_dec_jan_pair", 4)}
FLAG_OF_METHOD = np.zeros(256, dtype=np.uint8)
for _c, (_n, _f) in METHODS.items():
    FLAG_OF_METHOD[_c] = _f
CLASSES = list(lib.CLASSES)


def dec_jan_paired(code: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Meters whose December value is half of a December-January bill (pre-origin rule)."""
    code = np.asarray(code, dtype=object)
    det = np.asarray(det, dtype=object)
    return (det == "B") | ((code == "B") & np.isin(det, INCONCLUSIVE))


def year_ab(X: np.ndarray, y: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(a = (Dec_y + Jan_y+1)/2, b = (Oct_y + Nov_y)/2, index of Oct y) per meter row."""
    t_o = lib.tidx(f"{y}-10")
    a = (X[:, t_o + 2] + X[:, t_o + 3]) / 2.0
    b = (X[:, t_o] + X[:, t_o + 1]) / 2.0
    return a, b, t_o


def fill_params(X: np.ndarray, inst: np.ndarray, cls: np.ndarray, sel: np.ndarray,
                years=FIT_YEARS) -> dict:
    """Ratio parameters from the meter rows ``sel`` of one county (X: meter x month, months
    <= 2022-12 only are ever read for ``years`` <= 2021). Returns per class rho_bar, s (ddof 1),
    pooled flag, the per-year table and the county-pooled idiosyncratic sigma."""
    if max(years) > 2021:
        raise ValueError("fit years must end by 2021 (Jan y+1 <= 2022-01)")
    rows = []
    ab = {}
    for y in years:
        a, b, t_o = year_ab(X, y)
        ok = sel & np.isfinite(a) & np.isfinite(b) & (inst <= t_o)
        ab[y] = (a, b, ok)
        for c in CLASSES + ["all"]:
            m = ok if c == "all" else ok & (cls == c)
            rows.append({"year": y, "class": c, "n": int(m.sum()), "A": float(a[m].sum()),
                         "B": float(b[m].sum())})
    tab = pd.DataFrame(rows)
    tab["rho"] = np.where(tab["B"] > 0, tab["A"] / tab["B"].where(tab["B"] > 0, 1.0), np.nan)
    out = {"years": list(years), "classes": {}}
    allc = tab[tab["class"] == "all"].set_index("year")["rho"]
    if not (np.isfinite(allc).all() and (tab.loc[tab["class"] == "all", "n"] > 0).all()):
        raise ValueError("county-wide ratio undefined in some year")
    rho_y = {}
    for c in CLASSES:
        t = tab[tab["class"] == c].set_index("year")
        own = bool((t["n"] >= N_MIN_CLASS).all() and np.isfinite(t["rho"]).all())
        r = t["rho"] if own else allc
        rho_y[c] = r
        out["classes"][c] = {"rho_bar": float(r.mean()), "s": float(r.std(ddof=1)),
                             "pooled_to_county": not own,
                             "n_meters_min_year": int(t["n"].min())}
    num = den = 0.0
    for y in years:
        a, b, ok = ab[y]
        ry = np.array([rho_y[c][y] for c in CLASSES])
        ci = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in cls])
        m = ok & (ci >= 0)
        e = a[m] - ry[ci[m]] * b[m]
        num += float((e * e).sum())
        den += float((b[m] * b[m]).sum())
    out["sigma"] = float(np.sqrt(num / den)) if den > 0 else float("nan")
    out["table"] = tab.to_dict(orient="records")
    return out


def meter_fill(X: np.ndarray, inst: np.ndarray, cls: np.ndarray, sel: np.ndarray,
               params: dict, y: int = 2022) -> dict:
    """Per-meter fill of December y from the Oct-Nov y base (rows ``sel`` only; NaN
    elsewhere or without a base): base, fill, the common and idiosyncratic sd coefficients
    (sd of the meter fill = base x sqrt(s^2 + sigma^2))."""
    t_o = lib.tidx(f"{y}-10")
    base = (X[:, t_o] + X[:, t_o + 1]) / 2.0
    ok = sel & np.isfinite(base) & (inst <= t_o)
    rho = np.array([params["classes"][c]["rho_bar"] if c in params["classes"] else np.nan
                    for c in cls])
    s = np.array([params["classes"][c]["s"] if c in params["classes"] else np.nan for c in cls])
    nan = np.full(len(base), np.nan)
    return {"ok": ok, "base": np.where(ok, base, nan), "fill": np.where(ok, rho * base, nan),
            "s": np.where(ok, s, nan), "sigma": params["sigma"]}


def group_sd(key: np.ndarray, meter_key: np.ndarray, cc: np.ndarray, ic: np.ndarray
             ) -> pd.Series:
    """sd of the fill summed over each group ``key`` (pumps with key < 0 skipped):
    sqrt((sum cc)^2 + sum over meters of (sum ic)^2), cc = s x b_p, ic = sigma x b_p."""
    d = pd.DataFrame({"key": key, "meter": meter_key, "cc": cc, "ic": ic})
    d = d[d["key"] >= 0]
    if d.empty:
        return pd.Series(dtype=float)
    common = d.groupby("key")["cc"].sum()
    idio = d.groupby(["key", "meter"])["ic"].sum().pow(2).groupby(level=0).sum()
    return np.sqrt(common.pow(2) + idio.reindex(common.index).fillna(0.0))
