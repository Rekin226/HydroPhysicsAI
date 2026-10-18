"""WP S5 repair round: the v2 operators on top of the registered round-1 code (read-only).

Amendment 1 - GNSS offset segments. Every GNSS station series is split at each gap longer
than ``GNSS_GAP_DAYS`` (30 d) between consecutive usable days (daily flag 0, 1 or 3) of its
gnss_clean daily record; each segment gets its own diffuse offset. Station excess rate,
family frame rate and family-month common modes are unchanged. A monthly value belongs to
the segment of its days (no calendar month can hold days on both sides of a gap > 30 d).

Amendment 2 - stepped leveling benchmarks. A benchmark with >= 4 surveys is scanned for one
step on the residuals of a fit (y - posterior-mean field), with its own diffuse offset and
random velocity (sd sig_rb) eliminated exactly as in the leveling operator: for every split
with >= 2 surveys on each side, delta = s'W r / s'W s, sd = (s'W s)^-1/2. The largest |z| is
flagged when |z| > 5 and |delta| >= 30 mm; the benchmark then gets a second diffuse offset
from the step survey on (one step per benchmark).

The Problem object of ``s5fit`` is patched, not copied: :func:`make_problem_v2` calls
``s5fit.make_problem`` and then adds the extra offset parameters (names ``goff:<f>:<s>:<k>``
for GNSS segment k >= 1) and the leveling segment ids. :func:`fit_v2` assembles every family
with :func:`contribution_v2` (leveling with segments; the other families as in round 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5fit as F  # noqa: E402
import s5model as M  # noqa: E402
import s5r_vconst as V  # noqa: E402


# ============================================================================ GNSS segments
def gnss_breaks(daily: pd.DataFrame, gap_days: int = V.GNSS_GAP_DAYS,
                usable=V.GNSS_USABLE_DAY_FLAGS) -> dict:
    """{(family, station): [(last day before, first day after, gap days), ...]} for every gap
    longer than ``gap_days`` between consecutive usable days (dates and flags only)."""
    d = daily[daily["flag"].isin(list(usable))][["family", "station", "date"]].copy()
    d["date"] = pd.to_datetime(d["date"].astype(str))
    out = {}
    for (f, s), g in d.groupby(["family", "station"], sort=True):
        dd = g["date"].sort_values().drop_duplicates().reset_index(drop=True)
        if len(dd) < 2:
            continue
        gaps = dd.diff().dt.days.to_numpy()
        idx = np.flatnonzero(gaps > gap_days)
        out[(str(f), str(s))] = [(dd.iloc[i - 1], dd.iloc[i], int(gaps[i])) for i in idx]
    return out


def read_daily(variant: str) -> pd.DataFrame:
    """Dates and flags of the gnss_clean/<variant>/v2 daily tables (no heights)."""
    pdir = C.GNSS_PRE if variant == "pre" else C.GNSS_FULL
    rows = []
    for fam, name in V.GNSS_DAILY.items():
        p = pdir / name
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["family", "station", "date", "flag"])
        d["family"] = fam if fam == "ncku" else d["family"].astype(str)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def gnss_segments(gn: pd.DataFrame, breaks: dict) -> np.ndarray:
    """Segment id of every monthly GNSS row: the number of breaks whose first day after the
    gap falls on or before the end of the row's month."""
    seg = np.zeros(len(gn), int)
    mend = (pd.PeriodIndex(gn["month"].astype(str), freq="M") + 1).to_timestamp(how="start")
    for (f, s), br in breaks.items():
        if not br:
            continue
        sel = ((gn["family"] == f) & (gn["station"] == s)).to_numpy()
        if not sel.any():
            continue
        after = np.array([b[1] for b in br], dtype="datetime64[ns]")
        me = np.asarray(mend[sel], dtype="datetime64[ns]")
        seg[sel] = (after[None, :] < me[:, None]).sum(1)
    return seg


def segment_table(gn: pd.DataFrame, breaks: dict) -> pd.DataFrame:
    """One row per used GNSS offset segment: family, station, seg, first/last month, n."""
    rows = []
    for (f, s, k), g in gn.groupby(["family", "station", "seg"], sort=True):
        br = breaks.get((f, s), [])
        prev = br[k - 1] if 0 < k <= len(br) else None
        rows.append({"family": f, "station": s, "seg": int(k), "first_month": g.month.min(),
                     "last_month": g.month.max(), "n_months": int(len(g)),
                     "gap_before": (f"{prev[0].date()}..{prev[1].date()} ({prev[2]} d)"
                                    if prev else "")})
    return pd.DataFrame(rows)


# ============================================================================ problem (v2)
def make_problem_v2(data: dict, knots, families=F.FAMILIES, *, masks: dict | None = None,
                    fdict: dict | None = None, lattice: M.Lattice | None = None
                    ) -> F.Problem:
    """``s5fit.make_problem`` plus GNSS segment offsets (column ``seg`` of data['gnss']) and
    leveling segments (column ``seg`` of data['lev']); without those columns it is round 1."""
    pr = F.make_problem(data, knots, families, masks=masks, fdict=fdict, lattice=lattice)
    if "seg" in pr.gnss.columns:
        for s in pr._gnss_sites:
            segs = pr.gnss.loc[s["rows"], "seg"].to_numpy().astype(int)
            us = np.unique(segs)
            if us.size <= 1:
                continue
            f, st = s["key"]
            G = s["G"]
            first = segs == us[0]
            G[:, 1] = first.astype(float)
            extra_cols, extra_idx = [], []
            for k in us[1:]:
                name = f"goff:{f}:{st}:{int(k)}"
                if name not in pr.gindex:
                    pr.gnames.append(name)
                    pr.gindex[name] = len(pr.gnames) - 1
                extra_cols.append((segs == k).astype(float))
                extra_idx.append(pr.gindex[name])
            s["G"] = np.column_stack([G] + extra_cols)
            s["gidx"] = np.concatenate([s["gidx"], np.array(extra_idx, int)])
            s["segs"] = segs
    for s in pr._lev_sites:
        if "seg" in pr.lev.columns:
            s["seg"] = pr.lev.loc[s["rows"], "seg"].to_numpy().astype(int)
        else:
            s["seg"] = np.zeros(s["t"].size, int)
    return pr


def contribution_v2(pr: F.Problem, fam: str, h: dict) -> M.Contribution:
    if fam != "lev":
        return F.contribution(pr, fam, h)
    F.set_ell(pr, h["ell_km"])
    asm = M.Assembler(pr.K, pr.m, pr.ng)
    for s in pr._lev_sites:
        W, ld, ne = M.site_local(s["t"], s["seg"], h["sig_lev"], h["sig_rb"])
        asm.add_site(s["phi"], s["H"], W, s["y"], gidx=[0], G=(s["t"] - C.T_REF)[:, None],
                     logdet=ld, n_eff=ne)
    return asm.result()


def total_contribution_v2(pr: F.Problem, h: dict, families=None) -> M.Contribution:
    out = M.empty_contribution(pr.N)
    for fam in families or pr.families:
        out += contribution_v2(pr, fam, h)
    return out


def lev_subset_contribution_v2(pr: F.Problem, h: dict, sids) -> M.Contribution:
    F.set_ell(pr, h["ell_km"])
    sids = set(sids)
    asm = M.Assembler(pr.K, pr.m, pr.ng)
    for s in pr._lev_sites:
        if s["sid"] not in sids:
            continue
        W, ld, ne = M.site_local(s["t"], s["seg"], h["sig_lev"], h["sig_rb"])
        asm.add_site(s["phi"], s["H"], W, s["y"], gidx=[0], G=(s["t"] - C.T_REF)[:, None],
                     logdet=ld, n_eff=ne)
    return asm.result()


def fit_v2(pr: F.Problem, h: dict, *, want_cov: bool = False) -> M.Posterior:
    return F.fit(pr, h, contrib=total_contribution_v2(pr, h), want_cov=want_cov)


# ============================================================================ stepped benchmarks
def step_scan(t: np.ndarray, r: np.ndarray, sig: float, sig_r: float) -> dict | None:
    """Best single step in the residual series r(t) of one benchmark (offset diffuse, random
    velocity ~ N(0, sig_r^2), white noise sig); None when the series is too short."""
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    n = t.size
    if n < V.STEP_MIN_SURVEYS:
        return None
    W, _, _ = M.site_local(t, np.zeros(n, int), sig, sig_r)
    best = None
    for k in range(V.STEP_MIN_SIDE, n - V.STEP_MIN_SIDE + 1):
        sv = (np.arange(n) >= k).astype(float)
        a = float(sv @ W @ sv)
        if a <= 1e-12:
            continue
        d = float(sv @ W @ r) / a
        z = d * np.sqrt(a)
        if best is None or abs(z) > abs(best["z"]):
            best = {"k": int(k), "delta_mm": d, "sd_mm": float(1.0 / np.sqrt(a)), "z": z}
    return best


def leveling_steps(pr: F.Problem, post: M.Posterior, h: dict) -> pd.DataFrame:
    """Scan every leveling benchmark of ``pr`` for one step; returns one row per scanned
    benchmark (flagged column = rule of amendment 2)."""
    F.set_ell(pr, h["ell_km"])
    rows = []
    for s in pr._lev_sites:
        r = s["y"] - F.site_mean(pr, post, s, "lev")
        b = step_scan(s["t"], r, h["sig_lev"], h["sig_rb"])
        if b is None:
            continue
        dates = pr.lev.loc[s["rows"], "date"].to_numpy()
        flagged = abs(b["z"]) > V.STEP_Z and abs(b["delta_mm"]) >= V.STEP_MIN_MM
        rows.append({"sid": s["sid"], "n_surveys": int(s["t"].size), "k": b["k"],
                     "step_after": str(dates[b["k"] - 1]), "step_at": str(dates[b["k"]]),
                     "delta_mm": b["delta_mm"], "sd_mm": b["sd_mm"], "z": b["z"],
                     "flagged": bool(flagged)})
    return pd.DataFrame(rows)


def lev_segments(lev: pd.DataFrame, steps: pd.DataFrame) -> np.ndarray:
    """Segment id per leveling row: 1 from the flagged step survey on, else 0."""
    seg = np.zeros(len(lev), int)
    if steps is None or not len(steps):
        return seg
    fl = steps[steps.flagged]
    for _, r in fl.iterrows():
        sel = (lev["sid"] == r.sid).to_numpy() & (lev["date"].astype(str).to_numpy()
                                                  >= r.step_at)
        seg[sel] = 1
    return seg
