"""X5.CH_twfe_mirror.v1 driver (round 7 rank 16): within-township response of Changhua heads
to local census energy on the 74 spent calibration screens, mirrored on model-world heads,
with placebos and a synthetic G0 gate. Narrowing only; kappa is reported, never applied.

Steps (repository root; ``run_twfe.sh`` calls them and logs every attempt):

``design``        BEFORE registration. Census and station METADATA only (sid, site, township,
                  x, y): the kernel regressors (3 km primary, 5 km report-only) for the 158
                  frozen screens from census_L1 v3 coordinates and preorigin v3 per-pump kWh,
                  and the design arrays. Prints counts only.
``model-panel``   BEFORE registration. One model's monthly and annual heads at the screens,
                  from the baseline scenario of its stored forward (whitelisted npz keys:
                  dates, origin, scenario_names, heads_mean), hindcast months
                  2012-01..2022-12 only. Prints the shape and sha256 only.
``g0``            AFTER registration, once: the synthetic gate in R1c's world. g0.json is
                  written once (mode 'x').
``score``         AFTER registration and G0, once. g0.json must carry the registered
                  declaration's sha256 and a time after the registration. If G0 failed the
                  reading is written with no head read. Otherwise the look is logged
                  (design_time_evidence) and the observed heads of the 158 screens (flags 0-1,
                  2012-01..2022-12) are read; each model's annual mean is taken on the same
                  months the observation uses.
``smoke``         synthetic design, models and heads in a scratch dir (no product is read).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "dataprep_staging" / "n2_policy"), str(REPO / "dataprep_staging"),
           str(REPO / "dataprep_staging" / "n2c_reading"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import twfe_lib as T  # noqa: E402

DECL_ID = "X5.CH_twfe_mirror.v1"
X = "dataprep_staging/ch_audit"
WELLS = "dataprep_staging/x5_class_share/wells_frozen.json"
WELLS_SHA = "fc2d8257dbe729238a96faf96f2e0b8e3ff82facc7faddb03794b453c2fffcd6"
BS_PATH = "dataprep_staging/n2_policy/bootstrap_scorer.py"
BS_SHA = "f8990d6182b7b080608f77b7e5f7a449ea722f97f845cd83478997cdc9ae75bd"
SYNTH_N2C = "dataprep_staging/n2c_reading/synth_n2c.py"
CODE_FILES = (f"{X}/twfe_lib.py", f"{X}/twfe_mirror.py", f"{X}/run_twfe.sh", BS_PATH,
              SYNTH_N2C, WELLS)
STATION_META = "results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet"
META_COLS = ["sid", "site_id", "TownName", "poly_town", "x_twd97", "y_twd97"]
HEADS = "results/dataprep/heads/L1_clean/v3/heads_monthly.parquet"
HEAD_COLS = ["sid", "date", "value", "flag"]
CENSUS_L1 = "results/dataprep/pump_electricity/census_L1/v3/census_pumps.parquet"
PUMPS_DIR = "results/dataprep/pump_electricity/preorigin/v3"
PUMP_FILES = ("pumps_changhua.npz", "pumps_yunlin.npz", "pumps_chiayi_ingrid.npz")
MODELS = {"R1c": "results/twin_forward/datum_gate_ccz.npz",
          "ccz_m1618": "results/twin_forward/chlever_ccz_m1618.npz",
          "q0c": "results/twin_forward/datum_gate_q0c.npz"}
MODEL_KEYS = ("dates", "origin", "scenario_names", "heads_mean")
BASELINE = "baseline"
LOG = "results/twin_runs/ch_twfe"
DESIGN = f"{LOG}/design.npz"
G0_JSON = f"{LOG}/g0.json"
REPORT = f"results/dataprep/reports/{DECL_ID}/report.json"
AUDIT = "results/dataprep/reports/CH_audit/audit.json"
LOOK_UNITS = {"changhua": "CH_twfe.screens_changhua_74", "yunlin": "CH_twfe.screens_yunlin_84"}
LOOK_SCOPE = "heads:2012-01..2022-12"
SPENT_AS = ("N1:cal_wells", "H4:wells158_2020-2022")     # registered units, same screens


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256(repo: Path = REPO) -> dict:
    return {c: sha256(repo / c) for c in CODE_FILES}


def noise_params() -> dict:
    """The declared G0 noise: n2c_reading's published parameters (R1c's Changhua anomaly
    RMSE, 5 km block share, lag-1 autocorrelation), cited, not re-derived."""
    import synth_n2c as S
    return {"sigma_m": S.PARAMS["sigma_tot_m"]["value"],
            "block_share": S.PARAMS["block_share"]["value"], "phi": S.PARAMS["phi"]["value"],
            "sources": {k: {kk: S.PARAMS[k][kk] for kk in ("source", "field")}
                        for k in ("sigma_tot_m", "block_share", "phi")}}


def frozen_wells(repo: Path = REPO) -> pd.DataFrame:
    p = repo / WELLS
    if sha256(p) != WELLS_SHA:
        raise SystemExit(f"REFUSED: {WELLS} changed")
    return pd.DataFrame(json.loads(p.read_text())["rows"])


def _codes(v) -> np.ndarray:
    return np.unique(np.asarray(v).astype(str), return_inverse=True)[1].astype(np.int64)


# ----------------------------------------------------------------------------- design
def build_design(wells: pd.DataFrame, meta: pd.DataFrame, pumps: dict) -> dict:
    """Design arrays and regressors. ``meta``: META_COLS rows for the screens; ``pumps``:
    {px, py, E (P, len(REG_YEARS)) GWh/yr}."""
    d = wells.merge(meta, on="sid", how="left")
    if d[["site_id", "x_twd97", "y_twd97"]].isna().any().any():
        raise SystemExit("REFUSED: a frozen screen lacks site or coordinates")
    town = d["TownName"].where(d["TownName"].notna(), d["poly_town"]).astype(str)
    x, y = d["x_twd97"].to_numpy(float), d["y_twd97"].to_numpy(float)
    out = {"sid": d["sid"].astype(str).to_numpy(), "county": d["county"].astype(str).to_numpy(),
           "cell": d["cell"].to_numpy(np.int64), "layer": d["layer"].to_numpy(np.int64),
           "site": _codes(d["site_id"]), "town": _codes(d["county"] + "|" + town),
           "town_name": town.to_numpy(), "x": x, "y": y,
           "block": _codes(BS.block_ids(x, y, BS.BLOCK_KM)),
           "reg_years": np.array(T.REG_YEARS), "years": np.array(T.YEARS)}
    for h, key in ((T.H_KM_PRIMARY, "X3"), (T.H_KM_REPORT, "X5")):
        out[key] = T.kernel_energy(x, y, pumps["px"], pumps["py"], pumps["E"], h)
    return out


def load_pumps(repo: Path = REPO) -> dict:
    """In-grid kept pumps of preorigin v3 (the model's forcing set) with census_L1 v3
    coordinates; annual GWh 2007..2022 (missing months count as zero, the product's
    aggregate convention)."""
    cen = pd.read_parquet(repo / CENSUS_L1, columns=["uid", "x_twd97", "y_twd97", "kept"])
    cen = cen.set_index(cen["uid"].astype(str))
    xs, ys, Es = [], [], []
    for f in PUMP_FILES:
        z = np.load(repo / PUMPS_DIR / f, allow_pickle=False)
        uid = np.asarray(z["uid"]).astype(str)
        cell = np.asarray(z["cell"], np.int64)
        dates = pd.to_datetime([str(s)[:7] for s in z["dates"]])
        val = np.nan_to_num(np.asarray(z["value"], float), nan=0.0)
        keep = cell >= 0
        c = cen.reindex(uid[keep])
        if not c["kept"].fillna(False).astype(bool).all():
            raise SystemExit(f"REFUSED: {f} holds pumps census_L1 v3 does not keep")
        ok = c[["x_twd97", "y_twd97"]].notna().all(axis=1).to_numpy()
        vk = val[keep]
        ann = np.column_stack([vk[:, dates.year == yy].sum(axis=1)
                               for yy in T.REG_YEARS]) / 1e6
        xs.append(c["x_twd97"].to_numpy(float)[ok])
        ys.append(c["y_twd97"].to_numpy(float)[ok])
        Es.append(ann[ok])
    return {"px": np.concatenate(xs), "py": np.concatenate(ys), "E": np.vstack(Es)}


def save_npz_once(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"REFUSED: {path} exists (frozen)")
    with open(path, "xb") as f:
        np.savez(f, **arrays)
    return sha256(path)


def cmd_design(repo: Path = REPO, out: Path | None = None) -> dict:
    wells = frozen_wells(repo)
    meta = pd.read_parquet(repo / STATION_META, columns=META_COLS)
    meta = meta[meta["sid"].astype(str).isin(set(wells["sid"]))]
    des = build_design(wells, meta, load_pumps(repo))
    h = save_npz_once(out or (repo / DESIGN), **des)
    summ = {}
    for c in ("changhua", "yunlin"):
        m = des["county"] == c
        p = T.make_panel(des["site"][m], des["town"][m], np.ones((int(m.sum()), len(T.YEARS))))
        summ[c] = T.identifying(p)
    print(json.dumps({"design_sha256": h, "screens": int(len(des["sid"])), "design": summ}))
    return {"sha256": h, "design": summ}


# ----------------------------------------------------------------------------- model panels
def model_annual(npz_path: Path, cells, layers) -> tuple[np.ndarray, np.ndarray, dict]:
    """(W, len(YEARS)) annual mean and (W, 132) monthly baseline heads at (layer, cell),
    hindcast months 2012-01..2022-12 only."""
    z = np.load(npz_path, allow_pickle=False)
    names = [str(s) for s in z["scenario_names"]]
    dates = [str(s)[:7] for s in z["dates"]]
    origin = int(z["origin"])
    if BASELINE not in names:
        raise SystemExit(f"REFUSED: {npz_path} has no {BASELINE!r} scenario")
    if dates[0] != "2012-01" or dates[origin] != "2022-12":
        raise SystemExit(f"REFUSED: {npz_path} dates {dates[0]}, origin {dates[origin]}")
    hm = z["heads_mean"]
    h = np.asarray(hm[names.index(BASELINE)][np.asarray(layers), np.asarray(cells), :origin + 1],
                   dtype=np.float64)
    yrs = np.array([int(d[:4]) for d in dates[:origin + 1]])
    out = np.column_stack([h[:, yrs == y].mean(axis=1) for y in T.YEARS])
    return out, h, {"n_months": int(origin + 1), "scenario": BASELINE,
                    "keys_read": list(MODEL_KEYS)}


def cmd_model_panel(name: str, repo: Path = REPO, out: Path | None = None,
                    src: Path | None = None) -> dict:
    wells = frozen_wells(repo)
    src = src or (repo / MODELS[name])
    H, Hmon, info = model_annual(src, wells["cell"].to_numpy(), wells["layer"].to_numpy())
    h = save_npz_once(out or (repo / f"{LOG}/model_{name}.npz"), H=H, Hmon=Hmon,
                      sid=wells["sid"].astype(str).to_numpy(), source_sha256=sha256(src),
                      source=str(MODELS.get(name, src)))
    print(json.dumps({"model": name, "shape": list(H.shape), "panel_sha256": h,
                      "source_sha256": sha256(src)}))
    return {"sha256": h, **info}


# ----------------------------------------------------------------------------- registration
def registered(repo: Path = REPO, prereg_dir=None, rows_dir=None) -> dict:
    """The registered declaration, its code shas and frozen inputs against the files."""
    from harness import prereg3
    kw = {}
    if prereg_dir is not None:
        kw = {"prereg_dir": prereg_dir, "rows_dir": rows_dir or Path(prereg_dir) / "_rows"}
    doc, sha = prereg3.load_prereg(DECL_ID, **kw)
    bad = [c for c, h in doc["code_sha256"].items() if sha256(repo / c) != h]
    bad += [p for p, h in doc["frozen_inputs"].items() if sha256(repo / p) != h]
    if bad:
        raise SystemExit(f"REFUSED: files differ from the registration: {bad}")
    return {"doc": doc, "sha256": sha}


def _load_design(repo: Path, design: Path | None = None) -> dict:
    z = np.load(design or (repo / DESIGN), allow_pickle=False)
    return {k: z[k] for k in z.files}


def _county(des: dict, c: str) -> np.ndarray:
    return des["county"] == c


def _shift_maps(Xw: np.ndarray) -> dict:
    return {+T.SHIFT_YEARS: T.shifted(Xw, +T.SHIFT_YEARS),
            -T.SHIFT_YEARS: T.shifted(Xw, -T.SHIFT_YEARS)}


def run_g0(des: dict, H_r1c: np.ndarray, draws=T.G0_DRAWS, B=T.B_BOOT,
           report_arms=True) -> dict:
    m = _county(des, "changhua")
    Xw = T.on_years(des["X3"][m])
    res = T.g0(H_r1c[m], Xw, des["site"][m], des["town"][m], des["block"][m], noise_params(),
               _shift_maps(des["X3"][m]), draws=draws, B=B, report_arms=report_arms)
    return res


def cmd_g0(repo: Path = REPO, prereg_dir=None, out: Path | None = None, design=None,
           models_dir=None, draws=T.G0_DRAWS, B=T.B_BOOT, check_registration=True) -> dict:
    reg = registered(repo, prereg_dir) if check_registration else None
    target = out or (repo / G0_JSON)
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; G0 runs once")
    des = _load_design(repo, design)
    mdir = Path(models_dir) if models_dir else repo / LOG
    H = np.load(mdir / "model_R1c.npz")["H"]
    res = run_g0(des, H, draws=draws, B=B)
    res.update({"id": DECL_ID, "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "declaration_sha256": reg["sha256"] if reg else None})
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=float)
    print(json.dumps({k: res.get(k) for k in ("passed", "coverage_base", "coverage_ramp",
                                              "power", "placebo_mode", "beta_model")},
                     default=float))
    return res


# ----------------------------------------------------------------------------- scoring
def observed_annual(heads: pd.DataFrame, sids, return_mask: bool = False):
    """(W, len(YEARS)) annual means of flags 0-1 monthly heads, >= 9 months a year; with
    ``return_mask`` also the (W, 132) mask of the months used (accepted flag, finite, in a
    screen-year that qualifies)."""
    h = heads.copy()
    h["sid"] = h["sid"].astype(str)
    h["date"] = pd.to_datetime(h["date"])
    h = h[(h["date"] >= "2012-01-01") & (h["date"] < "2023-01-01")]
    months = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    pos = {s: i for i, s in enumerate(sids)}
    V = np.full((len(sids), len(months)), np.nan)
    F = np.full((len(sids), len(months)), 9, dtype=np.int64)
    h = h[h["sid"].isin(pos)]
    ti = ((h["date"].dt.year - 2012) * 12 + h["date"].dt.month - 1).to_numpy()
    wi = h["sid"].map(pos).to_numpy()
    V[wi, ti] = h["value"].to_numpy(float)
    F[wi, ti] = h["flag"].to_numpy(np.int64)
    yom = months.year.to_numpy()
    Y = T.annual_from_monthly(V, F, yom)
    if not return_mask:
        return Y
    ok = np.isfinite(V) & np.isin(F, T.FLAGS_SCORED)
    ok &= np.isfinite(Y[:, yom - T.YEARS[0]])
    return Y, ok


def masked_model_annual(Hmon: np.ndarray, okm: np.ndarray) -> np.ndarray:
    """(W, len(YEARS)) a model's annual means on exactly the months the observation uses
    (``okm`` from :func:`observed_annual`); NaN where the observation has no annual value."""
    yom = pd.date_range("2012-01-01", "2022-12-01", freq="MS").year.to_numpy()
    F = np.where(okm, 0, 9)
    return T.annual_from_monthly(np.asarray(Hmon, float), F, yom)


def check_g0(g0_res: dict, reg: dict) -> None:
    """g0.json must belong to the registered declaration and postdate its registration."""
    from harness import prereg3
    if g0_res.get("declaration_sha256") != reg["sha256"]:
        raise SystemExit("REFUSED: g0.json was not run under the registered declaration "
                         f"({g0_res.get('declaration_sha256')} != {reg['sha256']})")
    created = reg["doc"].get("created")
    if not created or not g0_res.get("time") or (
            prereg3.parse_time(g0_res["time"]) <= prereg3.parse_time(created)):
        raise SystemExit("REFUSED: g0.json does not postdate the registration")


def score_arrays(des: dict, Yobs: np.ndarray, Hm: dict, g0_res: dict) -> dict:
    """Everything after G0, from arrays (the smoke and the real run share it)."""
    out: dict = {"g0": {k: g0_res.get(k) for k in ("passed", "coverage_base",
                                                   "coverage_ramp", "power", "placebo_mode",
                                                   "beta_model")}}
    m = _county(des, "changhua")
    Xw = T.on_years(des["X3"][m])
    est = T.estimate(Xw, Yobs[m], des["site"][m], des["town"][m],
                     Hm={k: v[m] for k, v in Hm.items()})
    out["changhua"] = est
    plc = None
    dchk = T.design_check(g0_res, est) if g0_res.get("passed") else None
    out["design_check"] = dchk
    if (dchk is not None and dchk["passed"]
            and est["models"]["R1c"]["model_ci_excludes_0"]):
        plc = T.placebos(Xw, _shift_maps(des["X3"][m]), Yobs[m], des["site"][m],
                         des["town"][m], est["halfwidth"], g0_res["placebo_mode"])
    out["placebos"] = plc
    out["reading"] = T.reading(g0_res, est, plc)
    rep: dict = {}
    rep["kernel_5km"] = T.estimate(T.on_years(des["X5"][m]), Yobs[m], des["site"][m],
                                   des["town"][m], Hm={"R1c": Hm["R1c"][m]})
    rep["township_bootstrap"] = T.estimate(Xw, Yobs[m], des["site"][m], des["town"][m],
                                           plan_kind="township", Hm={"R1c": Hm["R1c"][m]})
    y = _county(des, "yunlin")
    rep["yunlin_parity"] = T.estimate(T.on_years(des["X3"][y]), Yobs[y], des["site"][y],
                                      des["town"][y], Hm={k: v[y] for k, v in Hm.items()})
    rep["note"] = ("report only: the Yunlin parity reading is a ledger line with Yunlin's "
                   "kappa interval, never README or page (no Yunlin G0, placebo or design "
                   "check), and is never suppressed")
    out["report_only"] = rep
    return out


def log_look(sids_by_county: dict, prereg_dir=None) -> dict:
    from harness import prereg3
    kw = {}
    if prereg_dir is not None:
        kw = {"prereg_dir": prereg_dir, "rows_dir": Path(prereg_dir) / "_rows"}
    units = [LOOK_UNITS[c] for c in ("changhua", "yunlin")]
    members = {LOOK_UNITS[c]: {"ids": sorted(map(str, sids_by_county[c])),
                               "window": "2012-01..2022-12",
                               "description": f"{c} frozen calibration screens (spent, "
                                              "design-time); the same screens as the "
                                              f"registered units {', '.join(SPENT_AS)}"}
               for c in ("changhua", "yunlin")}
    return prereg3.record_look(units, LOOK_SCOPE, "X5", "design_time_evidence",
                               members=members, prereg_id=DECL_ID,
                               note=f"{DECL_ID} score: annual flags 0-1 heads", **kw)


def cmd_score(repo: Path = REPO, prereg_dir=None, out: Path | None = None) -> dict:
    reg = registered(repo, prereg_dir)
    target = out or (repo / REPORT)
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; scored once")
    g0p = repo / G0_JSON
    if not g0p.exists():
        raise SystemExit("REFUSED: run g0 first")
    g0_res = json.loads(g0p.read_text())
    check_g0(g0_res, reg)
    des = _load_design(repo)
    rep = {"id": DECL_ID, "declaration_sha256": reg["sha256"], "g0_sha256": sha256(g0p),
           "time": dt.datetime.now(dt.timezone.utc).isoformat()}
    if not g0_res.get("passed"):
        rep["reading"] = T.reading(g0_res, None, None)
        rep["heads_read"] = False
    else:
        sids = des["sid"]
        Hmon = {}
        for n in MODELS:                 # checked before the look: a defect burns no look
            z = np.load(repo / f"{LOG}/model_{n}.npz")
            if [str(v) for v in z["sid"]] != [str(v) for v in sids]:
                raise SystemExit(f"ENGINEERING FAILURE: model_{n}.npz screens differ from "
                                 "the design")
            if z["Hmon"].shape != (len(sids), len(T.YEARS) * 12) or not np.isfinite(
                    z["Hmon"]).all():
                raise SystemExit(f"ENGINEERING FAILURE: model_{n}.npz monthly heads are not "
                                 "a complete (W, 132) panel")
            Hmon[n] = z["Hmon"]
        log_look({c: sids[_county(des, c)] for c in ("changhua", "yunlin")}, prereg_dir)
        heads = pd.read_parquet(repo / HEADS, columns=HEAD_COLS)
        Yobs, okm = observed_annual(heads, list(sids), return_mask=True)
        Hm = {n: masked_model_annual(v, okm) for n, v in Hmon.items()}
        rep.update(score_arrays(des, Yobs, Hm, g0_res))
        rep["heads_read"] = True
    rep["page_mapping"] = reg["doc"]["evidence_and_ledger_mapping"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x") as f:
        json.dump(rep, f, indent=1, sort_keys=True, default=float)
    print(json.dumps({"reading": rep["reading"]["text"]}, ensure_ascii=False))
    return rep


# ----------------------------------------------------------------------------- smoke
def synthetic(seed: int = 3, kappa: float = 0.5, n_sites_per_town=(3, 2, 2, 1, 1, 3, 2, 1)
              ) -> tuple[dict, dict, np.ndarray]:
    """Synthetic design (two counties), model heads with a known within-township response to
    the regressor, and observed heads = kappa x the model response + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    s_id = 0
    for county in ("changhua", "yunlin"):
        for t, ns in enumerate(n_sites_per_town):
            for _ in range(ns):
                for layer in (1, 2):
                    rows.append({"sid": f"{county[0]}{s_id:03d}{layer}", "county": county,
                                 "site": s_id, "town": f"{county}|T{t}", "layer": layer,
                                 "x": 180000 + 3000 * t + rng.uniform(0, 2500),
                                 "y": 2620000 + rng.uniform(0, 4000)})
                s_id += 1
    d = pd.DataFrame(rows)
    W = len(d)
    site_x = d.groupby("site")["x"].first()
    d["x"] = d["site"].map(site_x)
    trend = rng.normal(0, 0.15, (s_id, 1)) * np.arange(len(T.REG_YEARS))[None, :]
    base = rng.uniform(2, 6, (s_id, 1)) + trend + rng.normal(0, 0.3, (s_id, len(T.REG_YEARS)))
    X = base[d["site"].to_numpy()]
    des = {"sid": d["sid"].to_numpy(), "county": d["county"].to_numpy(),
           "cell": np.arange(W), "layer": d["layer"].to_numpy(),
           "site": d["site"].to_numpy(np.int64), "town": _codes(d["town"]),
           "town_name": d["town"].to_numpy(), "x": d["x"].to_numpy(),
           "y": np.full(W, 2620000.0), "block": _codes(BS.block_ids(d["x"], np.full(W, 2.62e6))),
           "reg_years": np.array(T.REG_YEARS), "years": np.array(T.YEARS),
           "X3": X, "X5": X * 1.6}
    Xy = T.on_years(X)
    tw = des["town"]
    g = rng.normal(0, 1.0, (tw.max() + 1, len(T.YEARS)))[tw]
    a = rng.normal(0, 3.0, (W, 1))
    beta = -0.8
    H = a + g + beta * Xy + rng.normal(0, 0.05, Xy.shape)
    Hm = {"R1c": H, "ccz_m1618": H * 1.1, "q0c": a + g + 0.9 * beta * Xy}
    Yobs = a + 0.5 * g + kappa * beta * Xy + rng.normal(0, 0.1, Xy.shape)
    Yobs[rng.random(Yobs.shape) < 0.1] = np.nan
    return des, Hm, Yobs


def cmd_smoke(out_dir: Path, draws: int = 60, B: int = 200) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    des, Hm, Yobs = synthetic()
    g0_res = run_g0(des, Hm["R1c"], draws=draws, B=B, report_arms=False)
    rep = score_arrays(des, Yobs, Hm, g0_res)
    with open(out_dir / "smoke_report.json", "x") as f:
        json.dump({"g0": g0_res, "score": rep}, f, indent=1, sort_keys=True, default=float)
    print(json.dumps({"g0_passed": g0_res["passed"], "reading": rep["reading"]["text"]},
                     default=float))
    return {"g0": g0_res, "score": rep}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("design")
    mp = sub.add_parser("model-panel")
    mp.add_argument("--name", required=True, choices=sorted(MODELS))
    sub.add_parser("g0")
    sub.add_parser("score")
    sm = sub.add_parser("smoke")
    sm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "design":
        cmd_design()
    elif a.cmd == "model-panel":
        cmd_model_panel(a.name)
    elif a.cmd == "g0":
        cmd_g0()
    elif a.cmd == "score":
        cmd_score()
    else:
        cmd_smoke(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
