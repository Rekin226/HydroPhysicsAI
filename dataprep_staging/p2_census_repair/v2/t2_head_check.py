"""Placebo-controlled head check of the Chow-Lin split (plan P2a, "The head check needs a
regional placebo").

Two registered tests share this core:

* ``round1``: P2a.head_placebo.v1 (registered 2026-10-12T02:35:58Z by round-1 prereg.py,
  outside the A1 index; never scored before). Energy = the round-1 Yunlin AMP_V2 pipeline
  (``_work_p2abc``: pre-origin Chow-Lin arrays, reproduced byte-identically by r1_reproduce);
  every fan-station well with a defined statistic; regional covariates = the Yunlin in-grid
  total and the kWh of kept pumps > 15 km away.
* ``round2``: P2r2.a3.heads_yunlin / _changhua (P2abc.r2.c0_based.v1). Energy = the C0 v3
  pre-origin products of all three in-grid counties and this WP's stage-1 Chow-Lin arrays;
  wells of Yunlin and Changhua separately; regional covariates = the in-grid total of the
  well's county and the kWh of kept pumps (any county) > 15 km away.

Common definition (both registrations):

* heads: AMP_V2 fan stations, daily record truncated at 2022-12-31 (no 2023+ observation),
  pilot s06 rule (daily median, |value| < 1000, > 30 days, 15 x MAD despike, 15-day rolling
  median +-5 m, monthly mean of >= 10 days), months 2012-01..2022-11; wells with >= 60 months,
  layer 1-4, valid coordinates ('y x' swapped when the first number is >= 1e6);
* local energy at r km = sum over kept pumps within r of the well of HP share x meter series,
  for the recorded record, the real Chow-Lin split and 20 placebo splits (profile with calendar
  years 2007..2022 permuted by a seeded derangement, seeds 1..20, same production operator);
  regional covariates computed from the same energy version;
* covariates: intercept, high-passed rain lags 0-2 (nearest CODIS-rebuilt gauge with >= 90 %
  valid months, monthly sum of >= 20 valid days, gaps interpolated), the two regional series;
  high pass = x minus its centred 25-month rolling mean (min 12 months);
* statistic: partial R2 of the high-passed local energy (>= 48 usable months, non-constant
  energy); D_w(r) = pr2(real) - mean_k pr2(placebo k), defined when every version's fit is;
* primary: mean over wells of D_w = mean of D_w(r) over r in {1, 2} km where defined;
  site-cluster bootstrap (site = first 6 characters of the station id), 4000 resamples,
  seed 2026; one-sided p = (1 + #{boot mean <= 0}) / 4001; criterion p < 0.05.

Usage: python -I t2_head_check.py round1|round2 <out_dir>
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402

AMP = L.REPO / "AMP_V2" / "data"
RAIN = L.REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / \
    "rf_timeseries_codis.csv"
RAIN_ST = L.REPO / "chou-shui-data" / "data" / "rf_stations.csv"
HM = pd.date_range("2012-01-01", "2022-11-01", freq="MS")        # head months (131)
H0 = L.tidx("2012-01")                                            # index on the 2007 axis
NH = len(HM)
RADII = (1.0, 2.0, 3.0, 5.0)
PRIMARY_R = (1.0, 2.0)
FAR_KM = 15.0
N_PLACEBO = 20
N_BOOT = 4000
SEED = 2026
DAILY_END = pd.Timestamp("2022-12-31 23:59:59")


def code_sha() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), L.CODE_DIR / "p2lib.py")}


# ----------------------------------------------------------------------------- heads
def _xy(v):
    try:
        a, b = [float(t) for t in str(v).split()]
    except Exception:  # noqa: BLE001 - unparsable coordinates
        return np.nan, np.nan
    return (a, b) if a < 1e6 else (b, a)


def heads() -> tuple[pd.DataFrame, np.ndarray]:
    st = pd.read_parquet(AMP / "fan_stations.parquet",
                         columns=["sid", "CountyName", "GroundwaterLayerCode",
                                  "LocationByTWD97"])
    rows, H = [], []
    for r in st.itertuples():
        x, y = _xy(r.LocationByTWD97)
        h = np.full(NH, np.nan)
        p = AMP / "wells" / f"{r.sid}.parquet"
        d = pd.read_parquet(p)["value"].astype(float) if p.exists() else pd.Series(dtype=float)
        if len(d):
            d = d[d.index <= DAILY_END]
            d = d[d.abs() < 1000]
            day = d.resample("D").median().dropna()
            if len(day) > 30:
                med = day.median()
                mad = (day - med).abs().median() or day.std() or 1.0
                day = day[(day - med).abs() <= 15 * mad]
                rm = day.rolling(15, center=True, min_periods=5).median()
                day = day[(day - rm).abs() <= 5.0]
                mon = day.resample("MS").agg(["mean", "size"])
                mon = mon[mon["size"] >= 10]["mean"]
                h = mon.reindex(HM).to_numpy()
        rows.append({"sid": str(r.sid), "x": x, "y": y, "layer": str(r.GroundwaterLayerCode),
                     "county": r.CountyName, "site": str(r.sid)[:6],
                     "n_months": int(np.isfinite(h).sum())})
        H.append(h)
    return pd.DataFrame(rows), np.array(H)


def rain_for(wells: pd.DataFrame) -> tuple[np.ndarray, list]:
    rf = pd.read_csv(RAIN, parse_dates=["date time"]).set_index("date time")
    rf = rf[(rf.index >= HM[0]) & (rf.index < HM[-1] + pd.offsets.MonthBegin(1))]
    cnt = rf.resample("MS").count()
    tot = rf.resample("MS").sum(min_count=1)
    Pm = tot.where(cnt >= 20).reindex(HM)
    st = pd.read_csv(RAIN_ST, encoding="utf-8-sig")
    st = st[st["rf_id"].isin(Pm.columns)]
    good = [g for g in st["rf_id"] if Pm[g].notna().mean() >= 0.9]
    st = st[st["rf_id"].isin(good)].reset_index(drop=True)
    tree = cKDTree(st[["TM_X97", "TM_Y97"]].to_numpy())
    idx = tree.query(wells[["x", "y"]].to_numpy(), k=1)[1]
    P, used = [], []
    for i in idx:
        g = st.loc[i, "rf_id"]
        P.append(Pm[g].interpolate().bfill().ffill().to_numpy())
        used.append(g)
    return np.array(P), used


# ----------------------------------------------------------------------------- regression
def hp(v, w: int = 25) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.rolling(w, center=True, min_periods=12).mean()).to_numpy()


def pr2(y: np.ndarray, e: np.ndarray, P: list, regs: list) -> float:
    yy, ee = hp(y), hp(e)
    Z = np.column_stack([np.ones(len(yy)), ee] + P + [hp(r) for r in regs])
    ok = np.isfinite(yy) & np.isfinite(Z).all(1)
    ok[:2] = False
    if ok.sum() < 48 or np.nanstd(ee[ok]) == 0:
        return np.nan
    b, *_ = np.linalg.lstsq(Z[ok], yy[ok], rcond=None)
    res = yy[ok] - Z[ok] @ b
    Z0 = np.delete(Z[ok], 1, axis=1)
    b0, *_ = np.linalg.lstsq(Z0, yy[ok], rcond=None)
    r0 = yy[ok] - Z0 @ b0
    den = r0 @ r0
    return float(1 - (res @ res) / den) if den > 0 else np.nan


# ----------------------------------------------------------------------------- energy
class Energy:
    """Pumps (coordinates, HP share, meter row, county, in-grid) and a generator of
    (version name, meter matrix) pairs (rows = meters, columns = the 131 head months), built
    lazily so that one matrix is in memory at a time."""

    def __init__(self, px, py, pshare, prow, pcounty, pingrid, n_meters: int, versions):
        self.px, self.py, self.pshare, self.prow = px, py, pshare, prow
        self.pcounty, self.pingrid = pcounty, pingrid
        self.versions = versions            # callable -> iterator of (name, matrix)
        self.n_meters = int(n_meters)

    def weights(self, wells: pd.DataFrame, r_km: float) -> sparse.csr_matrix:
        ok = np.isfinite(self.px) & np.isfinite(self.py)
        pi = np.where(ok)[0]
        tree = cKDTree(np.column_stack([self.px[pi], self.py[pi]]))
        lists = tree.query_ball_point(wells[["x", "y"]].to_numpy(), r=r_km * 1000.0)
        rows, cols, vals = [], [], []
        for w, lst in enumerate(lists):
            if lst:
                p = pi[np.asarray(lst)]
                rows.append(np.full(len(p), w))
                cols.append(self.prow[p])
                vals.append(self.pshare[p])
        if not rows:
            return sparse.csr_matrix((len(wells), self.n_meters))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                         np.concatenate(cols))),
                                 shape=(len(wells), self.n_meters))

    def total_weights(self, mask: np.ndarray) -> np.ndarray:
        w = np.zeros(self.n_meters)
        np.add.at(w, self.prow[mask], self.pshare[mask])
        return w


def series_for(E: Energy, wells: pd.DataFrame, region_of_well: np.ndarray, region_w: dict):
    """Per version: local (r -> wells x months), far (wells x months), regional total
    (wells x months)."""
    W = {r: E.weights(wells, r) for r in RADII}
    W15 = E.weights(wells, FAR_KM)
    coords = np.isfinite(E.px) & np.isfinite(E.py)
    wall = E.total_weights(coords)
    out = {}
    for name, M in E.versions():
        assert M.shape == (E.n_meters, NH), (name, M.shape)
        loc = {r: np.asarray(W[r] @ M) for r in RADII}
        far = (wall @ M)[None, :] - np.asarray(W15 @ M)
        reg = np.vstack([region_w[g] @ M for g in region_of_well])
        out[name] = {"local": loc, "far": far, "region": reg}
    nmet = {r: np.asarray((W[r] > 0).sum(1)).ravel() for r in RADII}
    return out, nmet


def statistics(wells: pd.DataFrame, H: np.ndarray, P: np.ndarray, ser: dict, nmet: dict,
               fixed_nuisance: bool = False) -> pd.DataFrame:
    names = list(ser)
    rows = []
    for i, w in enumerate(wells.itertuples()):
        Pl = [hp(P[i]), hp(np.roll(P[i], 1)), hp(np.roll(P[i], 2))]
        rec = {"sid": w.sid, "site": w.site, "county": w.county, "layer": w.layer,
               "n_months": w.n_months}
        for r in RADII:
            rec[f"n_meters_{r:g}km"] = int(nmet[r][i])
            vals = {}
            for nm in names:
                src = ser["recorded"] if fixed_nuisance else ser[nm]
                vals[nm] = pr2(H[i], ser[nm]["local"][r][i], Pl,
                               [src["far"][i], src["region"][i]])
            plac = np.array([vals[f"placebo{k}"] for k in range(1, N_PLACEBO + 1)])
            rec[f"pr2_rec_{r:g}"] = vals["recorded"]
            rec[f"pr2_cl_{r:g}"] = vals["chowlin"]
            rec[f"pr2_plac_mean_{r:g}"] = float(np.mean(plac)) if np.isfinite(plac).all() \
                else np.nan
            ok = np.isfinite(vals["chowlin"]) and np.isfinite(plac).all()
            rec[f"D_{r:g}"] = vals["chowlin"] - float(np.mean(plac)) if ok else np.nan
            rec[f"cl_minus_rec_{r:g}"] = (vals["chowlin"] - vals["recorded"]
                                          if np.isfinite(vals["recorded"])
                                          and np.isfinite(vals["chowlin"]) else np.nan)
        Ds = [rec[f"D_{r:g}"] for r in PRIMARY_R if np.isfinite(rec[f"D_{r:g}"])]
        rec["D_w"] = float(np.mean(Ds)) if Ds else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def site_bootstrap(vals: np.ndarray, sites: np.ndarray, n_boot: int = N_BOOT,
                   seed: int = SEED) -> dict:
    ok = np.isfinite(vals)
    v, s = vals[ok], sites[ok]
    if len(v) == 0:
        return {"n_wells": 0, "n_sites": 0, "mean": np.nan, "p_one_sided": np.nan}
    us, inv = np.unique(s, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(us), size=(n_boot, len(us)))
    bm = sums[draw].sum(1) / cnts[draw].sum(1)
    return {"n_wells": int(len(v)), "n_sites": int(len(us)), "mean": float(v.mean()),
            "boot_q05_q95": [float(np.quantile(bm, 0.05)), float(np.quantile(bm, 0.95))],
            "p_one_sided": float((1 + np.sum(bm <= 0)) / (n_boot + 1))}


def summarize(df: pd.DataFrame, groups: dict) -> dict:
    out = {}
    for gname, sel in groups.items():
        d = df[sel]
        g = {"primary_D_w": site_bootstrap(d["D_w"].to_numpy(), d["site"].to_numpy())}
        for r in RADII:
            g[f"D_{r:g}km"] = site_bootstrap(d[f"D_{r:g}"].to_numpy(), d["site"].to_numpy())
            c = d[f"cl_minus_rec_{r:g}"].to_numpy()
            g[f"cl_minus_rec_{r:g}km"] = site_bootstrap(c, d["site"].to_numpy())
            okc = np.isfinite(c)
            g[f"share_cl_beats_rec_{r:g}km"] = {"n": int(okc.sum()),
                                                "share": float(np.mean(c[okc] > 0))
                                                if okc.any() else np.nan}
        g["primary_pass_p_lt_0.05"] = bool(g["primary_D_w"]["p_one_sided"] < 0.05)
        out[gname] = g
    return out


# ----------------------------------------------------------------------------- builders
def energy_round2() -> tuple[Energy, dict, dict]:
    l1 = L.l1_pumps()
    prod = L.open_c0("preorigin")
    cds = {c: L.CountyData("preorigin", c, prod=prod, l1=l1) for c in L.COUNTIES}
    pz = np.load(L.WORK / "preorigin_profiles.npz")
    profs = {c: pz[f"prof_{c}"] for c in ("yunlin", "changhua")}
    off, px, py, psh, prow, pc, pin = 0, [], [], [], [], [], []
    sl = slice(H0, H0 + NH)
    v = L.VARIANTS["preorigin"]
    info = {"n_meters": {}, "placebo_seconds": 0.0}
    for c, cd in cds.items():
        px.append(cd.x)
        py.append(cd.y)
        psh.append(cd.share)
        prow.append(cd.prow + off)
        pc.append(np.full(len(cd.uid), c, dtype=object))
        pin.append(cd.cell >= 0)
        info["n_meters"][c] = int(len(cd.meters))
        off += len(cd.meters)

    def versions():
        rec = {c: np.nan_to_num(cd.M) for c, cd in cds.items()}
        yield "recorded", np.vstack([rec[c][:, sl] for c in cds])
        blocks = []
        for c in cds:
            z = np.load(L.WORK / f"preorigin_{c}_cl.npz")
            cl = rec[c].copy()
            cl[z["rows"]] = np.nan_to_num(z["values"])
            blocks.append(cl[:, sl])
        yield "chowlin", np.vstack(blocks)
        for k in range(1, N_PLACEBO + 1):
            t0 = time.time()
            blocks = []
            for c, cd in cds.items():
                pp = L.year_shuffle(profs[L.PROFILE_OF[c]], k)
                Mk = rec[c].copy()
                rows = L.chowlin_rows(cd)
                for p_ in ("A", "B"):
                    vals, _, _, _ = L.apply_chowlin(cd.M, cd.F, cd.Mm, rows[p_], p_, pp,
                                                    v["cl0"], v["win_hi"][p_])
                    Mk[rows[p_]] = np.nan_to_num(vals)
                blocks.append(Mk[:, sl])
            info["placebo_seconds"] += time.time() - t0
            yield f"placebo{k}", np.vstack(blocks)

    pcounty = np.concatenate(pc)
    pingrid = np.concatenate(pin)
    E = Energy(np.concatenate(px), np.concatenate(py), np.concatenate(psh),
               np.concatenate(prow), pcounty, pingrid, off, versions)
    region_w = {c: E.total_weights(pingrid & (pcounty == c)) for c in ("yunlin", "changhua")}
    return E, region_w, info


def energy_round1() -> tuple[Energy, dict, dict]:
    sys.path.insert(0, str(L.CODE_DIR.parent))
    import lib as r1  # round-1 library (unchanged)
    w = r1.WORK
    mt = pd.read_parquet(w / "meters.parquet")
    pumps = pd.read_parquet(w / "pumps.parquet")
    M = np.load(w / "meter_mat.npy").astype(np.float64)
    phase = pd.read_parquet(w / "preorigin_phase.parquet")["phase"].to_numpy()
    prof = np.load(w / "preorigin_profile.npz")["prof"]
    keep = mt["kept"].to_numpy()
    irr = mt["cls"].to_numpy() == "irrigation"
    end = r1.tidx("2022-12")
    lo = r1.tidx("2011-01")
    win = {"A": (lo, end + 1), "B": (lo, end + 1)}            # round-1 s2.windows
    sl = slice(H0, H0 + NH)                                    # same 2007-01 base
    info = {"placebo_seconds": 0.0, "n_meters": int(len(mt))}

    def versions():
        yield "recorded", np.nan_to_num(M[:, sl])
        cl = np.load(w / "preorigin_cl.npy").astype(np.float64)
        yield "chowlin", np.nan_to_num(cl[:, sl])
        del cl
        for k in range(1, N_PLACEBO + 1):
            t0 = time.time()
            pp = r1.year_shuffle(prof[:end + 1], k)
            pp = np.concatenate([pp, np.full(r1.T_ALL - len(pp), pp[-1])])
            Mk = M.copy()
            for p_, lab in (("A", r1.PHASE_A), ("B", r1.PHASE_B)):
                sel = np.where(keep & irr & (phase == lab))[0]
                a, b = win[p_]
                op = r1.ChowLin(p_, pp, a, b)
                for j in range(0, len(sel), 20000):
                    bi = sel[j:j + 20000]
                    Mk[bi, a:b] = op.run(M[bi])
            info["placebo_seconds"] += time.time() - t0
            yield f"placebo{k}", np.nan_to_num(Mk[:, sl])

    kp = pumps["kept"].to_numpy()
    p = pumps[kp]
    E = Energy(p["x"].to_numpy(float), p["y"].to_numpy(float), p["share"].to_numpy(float),
               p["meter_id"].to_numpy(np.int64), np.full(len(p), "yunlin", dtype=object),
               p["cell"].to_numpy() >= 0, len(mt), versions)
    region_w = {"yunlin": E.total_weights(E.pingrid)}
    return E, region_w, info


# ----------------------------------------------------------------------------- main
def main(mode: str, out_dir: str) -> None:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"mode": mode, "code_sha256": code_sha()}
    if mode == "round2":
        import scoring
        res["authorization"] = scoring.authorize(["P2r2.a3.heads_yunlin",
                                                  "P2r2.a3.heads_changhua"])
    st, H = heads()
    use = (st["n_months"] >= 60) & st["x"].notna() & st["y"].notna() & \
        st["layer"].isin(["1", "2", "3", "4"])
    if mode == "round2":
        use &= st["county"].isin(["雲林縣", "彰化縣"])
    wells = st[use].reset_index(drop=True)
    Hw = H[use.to_numpy()]
    P, gauges = rain_for(wells)
    res["n_wells_candidates"] = int(use.sum())
    res["wells_by_county"] = wells["county"].value_counts().to_dict()
    res["rain_gauges_used"] = sorted(set(gauges))
    if mode == "round2":
        E, region_w, info = energy_round2()
        reg_of = np.where(wells["county"].to_numpy() == "雲林縣", "yunlin", "changhua")
    else:
        E, region_w, info = energy_round1()
        reg_of = np.full(len(wells), "yunlin")
    res["energy_info"] = info
    ser, nmet = series_for(E, wells, reg_of, region_w)
    df = statistics(wells, Hw, P, ser, nmet)
    dff = statistics(wells, Hw, P, ser, nmet, fixed_nuisance=True)
    df.to_csv(out / f"t2_head_check_{mode}_wells.csv", index=False)
    dff.to_csv(out / f"t2_head_check_{mode}_wells_fixed_nuisance.csv", index=False)
    if mode == "round2":
        groups = {"yunlin": (df["county"] == "雲林縣").to_numpy(),
                  "changhua": (df["county"] == "彰化縣").to_numpy(),
                  "pooled": np.ones(len(df), bool)}
    else:
        groups = {"all_wells": np.ones(len(df), bool),
                  "yunlin_wells": (df["county"] == "雲林縣").to_numpy()}
    res["summary"] = summarize(df, groups)
    res["summary_fixed_nuisance"] = summarize(dff, groups)
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, out / f"t2_head_check_{mode}.json")
    for g, s in res["summary"].items():
        print(mode, g, s["primary_D_w"], "pass", s["primary_pass_p_lt_0.05"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
