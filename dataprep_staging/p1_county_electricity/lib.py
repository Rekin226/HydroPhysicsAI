"""WP P1 library: TPC county electricity, national backcast, source separation, extension.

Pure functions on pandas/numpy objects; no file writes. Readers take explicit paths and are
called from scripts run with ``python -I`` (downloads are untrusted data: parsed, never run).

Notation (plan WP P1):
  CH_ag, YL_ag   TPC d007012 農林漁牧 sales, Changhua / Yunlin county (GWh/month)
  YL_clean       Yunlin census total in model units (cleaned census, all classes; GWh/month)
  N_YL           YL_ag - YL_clean, Yunlin's non-pump-like agricultural load
  a, b           CH_ag = a * YL_clean + b * N_YL (two-component NNLS)
  r(moy)         month-of-year mean of YL_clean / YL_ag (transfer ratio)
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

CH, YL = "彰化縣", "雲林縣"
AG = "農林漁牧"


# ----------------------------------------------------------------------------- readers
def read_tpc_d007012(path) -> pd.DataFrame:
    """Tidy d007012 -> long table date, county, sector, kwh, pct (all counties, sectors)."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").iloc[:, :5]
    df.columns = ["ym", "county", "sector", "kwh", "pct"]
    df = df.dropna(subset=["ym"])
    df["ym"] = df["ym"].str.strip()
    df["kwh"] = pd.to_numeric(df["kwh"].str.replace(",", "").str.strip(), errors="coerce")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    roc = df["ym"].str[:-2].astype(int)
    mm = df["ym"].str[-2:].astype(int)
    df["date"] = pd.to_datetime(dict(year=roc + 1911, month=mm, day=1))
    df["county"] = df["county"].str.strip().str.replace("臺", "台")
    df["sector"] = df["sector"].str.strip()
    return df[["date", "county", "sector", "kwh", "pct"]].sort_values(
        ["county", "sector", "date"]).reset_index(drop=True)


def tpc_pair(tpc: pd.DataFrame) -> pd.DataFrame:
    """Monthly GWh: CH_ag, YL_ag, CH_all, YL_all (index date)."""
    ag = tpc[tpc.sector == AG].pivot(index="date", columns="county", values="kwh") / 1e6
    tot = tpc.groupby(["date", "county"])["kwh"].sum().unstack() / 1e6
    out = pd.DataFrame({"CH_ag": ag[CH], "YL_ag": ag[YL], "CH_all": tot[CH],
                        "YL_all": tot[YL]})
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out.sort_index()


def read_national_ag(p256, p236) -> pd.Series:
    """Energy Administration national agricultural electricity (GWh/month, 2007-01 on)."""
    parts = []
    for p in (p256, p236):
        d = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
        s = pd.Series(pd.to_numeric(d["電力消費_農業部門(數值)"], errors="coerce").values,
                      index=pd.to_datetime(d.iloc[:, 0].str.strip(), format="%Y%m"))
        parts.append(s)
    nat = pd.concat(parts).sort_index()
    nat = nat[~nat.index.duplicated(keep="last")]
    nat.index = pd.DatetimeIndex(nat.index, name="date")
    return nat.rename("nat_ag")


def read_d003004_subindustry(zip_path) -> pd.DataFrame:
    """TPC national power-customer sales by sub-industry (agriculture 'A' rows), GWh/month."""
    z = zipfile.ZipFile(zip_path)
    frames = []
    for i in z.infolist():
        n = (i.filename.encode("cp437").decode("big5", "replace")
             if i.flag_bits & 0x800 == 0 else i.filename)
        if n.startswith("行業別售電_") and n.endswith(".csv"):
            yr = int(n[6:10])
            t = pd.read_csv(io.BytesIO(z.read(i)), encoding="utf-8-sig", dtype=str)
            t.columns = ["big", "mid", "small", "month", "kwh"]
            t["year"] = yr
            frames.append(t)
    df = pd.concat(frames)
    df = df[df.big.str.startswith("A")].drop_duplicates(["year", "month", "small"])
    df["kwh"] = pd.to_numeric(df.kwh.str.replace(",", ""), errors="coerce") / 1e6
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month.astype(int), day=1))
    nat = df.pivot_table(index="date", columns="small", values="kwh", aggfunc="sum")
    nat.index = pd.DatetimeIndex(nat.index, name="date")
    return nat.sort_index()


# ----------------------------------------------------------------------------- helpers
def tdec(idx: pd.DatetimeIndex) -> np.ndarray:
    """Decimal year of month starts."""
    return idx.year.to_numpy() + (idx.month.to_numpy() - 1) / 12.0


def annual_err_pct(pred: pd.Series, obs: pd.Series) -> pd.Series:
    """100 * (sum pred / sum obs - 1) per calendar year (complete pairs only)."""
    ok = pred.notna() & obs.notna()
    p, o = pred[ok], obs[ok]
    return 100.0 * (p.groupby(p.index.year).sum() / o.groupby(o.index.year).sum() - 1.0)


def rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


# ----------------------------------------------------------------------------- NNLS, 2 vars
def nnls2_stats(G: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Exact 2-variable NNLS from sufficient statistics G = X'X (..., 2, 2), h = X'y (..., 2).

    Vectorised over leading axes. The objective 0.5 b'Gb - b'h is minimised over b >= 0 by
    checking the interior solution and the three faces (b1 = 0, b2 = 0, both 0)."""
    G = np.asarray(G, float)
    h = np.asarray(h, float)
    g11, g12, g22 = G[..., 0, 0], G[..., 0, 1], G[..., 1, 1]
    h1, h2 = h[..., 0], h[..., 1]
    det = g11 * g22 - g12 * g12
    with np.errstate(divide="ignore", invalid="ignore"):
        bi1 = (g22 * h1 - g12 * h2) / det
        bi2 = (g11 * h2 - g12 * h1) / det
        f1 = np.where(g11 > 0, np.maximum(h1 / g11, 0.0), 0.0)
        f2 = np.where(g22 > 0, np.maximum(h2 / g22, 0.0), 0.0)
    cands = np.stack([np.stack([bi1, bi2], -1), np.stack([f1, np.zeros_like(f1)], -1),
                      np.stack([np.zeros_like(f2), f2], -1),
                      np.zeros(bi1.shape + (2,))], axis=-2)          # (..., 4, 2)
    feas = np.all(cands >= -1e-12, axis=-1) & np.all(np.isfinite(cands), axis=-1)
    obj = 0.5 * np.einsum("...ki,...ij,...kj->...k", cands, G, cands) \
        - np.einsum("...ki,...i->...k", cands, h)
    obj = np.where(feas, obj, np.inf)
    k = np.argmin(obj, axis=-1)
    return np.take_along_axis(cands, k[..., None, None], axis=-2)[..., 0, :]


def year_stats(X: np.ndarray, y: np.ndarray, years: np.ndarray):
    """Per-year sufficient statistics (unique years, G_y (Y,2,2), h_y (Y,2))."""
    uy = np.unique(years)
    G = np.stack([X[years == k].T @ X[years == k] for k in uy])
    h = np.stack([X[years == k].T @ y[years == k] for k in uy])
    return uy, G, h


def block_bootstrap_counts(n_years: int, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n_years) multinomial counts of a year-block bootstrap with replacement."""
    picks = rng.integers(0, n_years, size=(n_boot, n_years))
    return np.stack([np.bincount(p, minlength=n_years) for p in picks])


def nnls2_fit(P: np.ndarray, N: np.ndarray, y: np.ndarray, years: np.ndarray,
              n_boot: int = 2000, seed: int = 0, level: float = 0.90) -> dict:
    """Two-component NNLS y = a P + b N with a year-block bootstrap percentile interval."""
    X = np.column_stack([P, N])
    coef = nnls2_stats(X.T @ X, X.T @ y)
    pred = X @ coef
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    uy, Gy, hy = year_stats(X, y, years)
    rng = np.random.default_rng(seed)
    cnt = block_bootstrap_counts(len(uy), n_boot, rng)
    boots = nnls2_stats(np.einsum("by,yij->bij", cnt, Gy), np.einsum("by,yi->bi", cnt, hy))
    q = [50 * (1 - level), 50 * (1 + level)]
    lo, hi = np.percentile(boots, q, axis=0)
    resid = y - pred
    return {"a": float(coef[0]), "b": float(coef[1]), "r2": float(r2),
            "a_lo": float(lo[0]), "a_hi": float(hi[0]), "b_lo": float(lo[1]),
            "b_hi": float(hi[1]), "a_boot_sd": float(boots[:, 0].std(ddof=1)),
            "b_boot_sd": float(boots[:, 1].std(ddof=1)), "n_months": int(len(y)),
            "n_years": int(len(uy)), "years": [int(v) for v in uy],
            "resid_sd": float(resid.std(ddof=2)),
            "resid_ac1": float(np.corrcoef(resid[:-1], resid[1:])[0, 1]),
            "n_boot": int(n_boot), "seed": int(seed), "level": level}


# ----------------------------------------------------------------------------- backcast
def share_model(sh: pd.Series, moy_window: tuple[str, str], trend_window: tuple[str, str],
                trend: bool = True) -> dict:
    """Month-of-year mean share on ``moy_window`` plus a linear trend fitted (OLS with 12
    month-of-year intercepts) on ``trend_window``; prediction s(m) + beta (t - t_ref),
    t_ref = mean decimal year of the moy window."""
    w = sh.loc[moy_window[0]:moy_window[1]].dropna()
    s_moy = w.groupby(w.index.month).mean().reindex(range(1, 13))
    beta = 0.0
    if trend:
        tw = sh.loc[trend_window[0]:trend_window[1]].dropna()
        D = np.zeros((len(tw), 13))
        D[np.arange(len(tw)), tw.index.month.to_numpy() - 1] = 1.0
        D[:, 12] = tdec(tw.index)
        coef, *_ = np.linalg.lstsq(D, tw.to_numpy(), rcond=None)
        beta = float(coef[12])
    return {"s_moy": s_moy.to_numpy(), "beta": beta, "t_ref": float(tdec(w.index).mean()),
            "moy_window": list(moy_window), "trend_window": list(trend_window) if trend
            else None}


def share_predict(model: dict, idx: pd.DatetimeIndex) -> np.ndarray:
    return model["s_moy"][idx.month.to_numpy() - 1] + model["beta"] * (tdec(idx)
                                                                      - model["t_ref"])


# ----------------------------------------------------------------------------- transfer
def transfer_ratio(yl_clean: pd.Series, yl_ag: pd.Series, t0: str, t1: str,
                   exclude_years=(), exclude_mask=None) -> pd.Series:
    """r(moy) = month-of-year mean of YL_clean / YL_ag over [t0, t1] (complete months)."""
    r = (yl_clean / yl_ag).loc[t0:t1].dropna()
    if len(exclude_years):
        r = r[~r.index.year.isin(list(exclude_years))]
    if exclude_mask is not None:
        r = r[~exclude_mask.reindex(r.index).fillna(False).to_numpy()]
    return r.groupby(r.index.month).mean().reindex(range(1, 13))


def apply_ratio(r: pd.Series, yl_ag: pd.Series) -> pd.Series:
    return yl_ag * r.reindex(yl_ag.index.month).to_numpy()


# ----------------------------------------------------------------------------- synthetics
def ar1_noise(n: int, sd: float, phi: float, rng, size: int) -> np.ndarray:
    """(size, n) stationary Gaussian AR(1) noise with marginal sd ``sd``."""
    phi = float(np.clip(phi, -0.99, 0.99))
    e = rng.standard_normal((size, n)) * sd * np.sqrt(1 - phi ** 2)
    out = np.empty((size, n))
    out[:, 0] = rng.standard_normal(size) * sd
    for t in range(1, n):
        out[:, t] = phi * out[:, t - 1] + e[:, t]
    return out


def seasonal_component(p: pd.Series) -> np.ndarray:
    """c(m) aligned with p: month-of-year mean minus the overall mean."""
    clim = p.groupby(p.index.month).mean()
    return (clim - p.mean()).reindex(p.index.month).to_numpy()


def wilson(k: int, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (n may be an effective n)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n if n else float("nan")
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - hw), float(c + hw)
