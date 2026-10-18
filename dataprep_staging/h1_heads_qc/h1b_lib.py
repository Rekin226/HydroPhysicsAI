"""WP H1b library: GroundHeight repair from the W3 DTM well points (plan 3, WP H1, part b).

The L1 metadata product ``results/dataprep/heads/L1_station_ground/v<N>/`` gives every station of
``heads/L1_clean/v3`` one ground elevation:

* agency ``GroundHeight`` as stored where it is a real value (flag 0);
* where it is the missing-value code 0.0 or missing (NaN): the W3 DTM well point (bilinear
  20 m DTM, epoch-corrected to 2017.5, ``ground_elevation/v3``), recomputed here at the H1a v3
  repaired coordinates, with an sd (flag 5);
* nothing where neither exists (flag 8).

The sd of a flag-5 value is ``sd_cal = sqrt(sd_w3**2 + tau**2)``: ``sd_w3`` is W3's well-point
sd (local 100 m relief, benchmark point spread, epoch term) and ``tau`` is a split-conformal
inflation fitted on the stations whose agency GroundHeight is real, so that the 90 % interval
``DTM +- 1.645 sd_cal`` covers the agency value. The calibration is checked leave-one-10 km-
block-out (pre-registration ``H1b.ground_height.v1``). No head data is read anywhere in H1b.

Inputs are opened through the loader guard: H1a v3 for ``model_design`` (only its pre-origin
station table is readable for that use) and W3 v3 for ``design_time_evidence``, the only use
W3 grants besides after-the-fact analysis, because W3's gate failed. The product therefore
carries W3's gate: it stays closed to every gated use until the user records a decision on W3.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never the GPU (Phase-0 rule)
HERE = Path(__file__).resolve()
STAGING = HERE.parents[1]
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))
if str(HERE.parents[2]) not in sys.path:
    sys.path.insert(1, str(HERE.parents[2]))    # repository (read-only imports)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402
from common.paths import DATAPREP, REPO, TWIN_GRID_NPZ  # noqa: E402

WP = "H1b"
PRODUCT = "heads/L1_station_ground"
PREREG_ID = "H1b.ground_height.v1"
TEST_ID = "H1b.T1.sd_coverage_lobo"
FRESH_UNIT = "H1b:valid_groundheight_stations_calib"
HISTORY_UNIT = "H1b:session_profile_2026-10-13"
NS = "station_meta:"                 # namespace of station-metadata records in look members

H1A_DIR = DATAPREP / "heads" / "L1_clean" / "v3"
H1A_TABLE = "station_meta_preorigin.parquet"
W3_DIR = DATAPREP / "ground_elevation" / "v3"
W3_MOSAIC = "dtm20m_mosaic.npz"
W3_WELLS = "well_ground_elevation.parquet"
LEVELING = REPO / "chou-shui-data" / "data" / "ls_cache" / "ls-wra-lsp-obs__choushui_panel.parquet"

T_REF = 2017.5                       # W3's well-point epoch (2017-07-01)
LEVEL = 0.90                         # nominal coverage of the published interval
Z90 = 1.6448536269514722             # two-sided 90 % normal quantile (= W3 validation.Z90)
EST_CUTOFF = 20221231                # calibration stations established by the origin (U7)
GROSS_M = 5.0                        # informational: |agency - DTM| above this is reported
SEEN_BEFORE_REGISTRATION = ("070213M3", "070215T1")   # residual displayed on 2026-10-13
STATION_COLS = ["sid", "site_id", "CountyName", "x_twd97", "y_twd97", "xy_flag", "xy_method",
                "layer_code", "GroundHeight", "ground_height_missing_code", "WellDepth",
                "WellElevation", "EstablishDate"]
COMPARE_COLS = ["dtm_point_sheet_m", "dtm_point_epoch", "dtm_local_sd_m", "epoch_corr_ref_m",
                "ground_elev_dtm_ref_m", "sd_m", "rate_2012_2021_m_per_yr"]


# ----------------------------------------------------------------------------- inputs
def model_grid() -> dict:
    """Twin grid origin and the 158 model wells (keys of the datum gate's well_datum)."""
    with np.load(TWIN_GRID_NPZ, allow_pickle=False) as z:
        gate = json.loads(str(z["gate"]))
        g = {"x0": float(z["x0"]), "y0": float(z["y0"]), "dx": float(z["dx"]),
             "nx": int(z["nx"]), "ny": int(z["ny"])}
    g["model_wells"] = sorted(gate.get("well_datum", {}).keys())
    return g


def open_inputs() -> dict:
    """Open H1a v3 (model_design: only the pre-origin station table is readable) and W3 v3
    (design_time_evidence) through the guard; return tables, mosaic arrays and notes."""
    h1a = guard.open_product(H1A_DIR, "model_design")
    meta = h1a.read_table(H1A_TABLE, columns=STATION_COLS)
    meta["sid"] = meta["sid"].astype(str)
    w3 = guard.open_product(W3_DIR, "design_time_evidence")
    w3tab = w3.read_table(W3_WELLS)                  # flag filter drops W3's flag-8 rows
    w3tab["sid"] = w3tab["sid"].astype(str)
    with np.load(w3.path(W3_MOSAIC), allow_pickle=False) as z:
        mos = {k: z[k] for k in ("H", "epoch", "sheet", "extent", "step")}
    return {"meta": meta, "w3": w3tab, "mosaic": mos, "w3_manifest": w3.manifest,
            "h1a_manifest": h1a.manifest,
            "notes": {"h1a_v3": {"use": "model_design", "notes": h1a.notes,
                                 "readable_outputs": sorted(h1a.allowed_outputs or [])},
                      "w3_v3": {"use": "design_time_evidence", "notes": w3.notes}}}


def w3_policy_probe() -> dict:
    """Which uses W3 v3 grants (guard.check_use on its manifest; no file is read)."""
    from common.manifest import load_manifest
    m, _ = load_manifest(W3_DIR)
    out = {}
    for use in ("design_time_evidence", "after_the_fact_analysis", "validator", "model_design",
                "forcing_frozen", "selection"):
        try:
            guard.check_use(m, use)
            out[use] = "granted"
        except PermissionError as exc:
            out[use] = type(exc).__name__
    return {"gate": m["gate"], "uses": out}


# ----------------------------------------------------------------------------- DTM well points
def displacement_model():
    """W3's leveling displacement model, built exactly as ground_elevation/v3 built it."""
    from w3_dtm import leveling, validation
    obs = leveling.load_panel(str(LEVELING))
    camp = leveling.campaigns(obs, validation.DISP_YEARS)
    return leveling.DisplacementModel(camp, validation.DISP_YEARS)


def dtm_well_points(x: np.ndarray, y: np.ndarray, mos: dict, dm, sd_pt_mad: float,
                    a_epoch: float) -> pd.DataFrame:
    """W3's well-point computation (w3_dtm/build.py, section 'well points (H1b)') at (x, y),
    from the published mosaic: bilinear height and nearest-point epoch, 100 m local sd,
    leveling epoch correction to 2017.5, sd = sqrt(local**2 + point_mad**2 + epoch**2)."""
    from w3_dtm import build as w3build
    from w3_dtm import mosaic as w3mosaic
    M = w3mosaic.Mosaic(extent=tuple(float(v) for v in mos["extent"]),
                        step=float(mos["step"]), H=mos["H"], epoch=mos["epoch"],
                        sheet=mos["sheet"], sheets=pd.DataFrame())
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    hp, hep = M.bilinear(x, y)
    lsd = w3build.local_sd(M, x, y)
    ok = np.isfinite(hp)
    corr = np.full(len(x), np.nan)
    rate = np.full(len(x), np.nan)
    if ok.any():
        P = np.column_stack([x[ok], y[ok]])
        corr[ok] = dm.correction(P, np.full(int(ok.sum()), T_REF), [[e] for e in hep[ok]],
                                 [[1.0]] * int(ok.sum()))
        rate[ok] = dm.rate(P, 2012.0, 2021.0)
    sd = np.sqrt(lsd ** 2 + sd_pt_mad ** 2 + (a_epoch * np.abs(T_REF - hep)) ** 2)
    return pd.DataFrame({"dtm_point_sheet_m": hp, "dtm_point_epoch": hep,
                         "dtm_local_sd_m": lsd, "epoch_corr_ref_m": corr,
                         "ground_elev_dtm_ref_m": hp + corr, "sd_m": sd,
                         "rate_2012_2021_m_per_yr": rate})


def reproduction(meta: pd.DataFrame, mine: pd.DataFrame, w3tab: pd.DataFrame) -> dict:
    """C2: recomputed well points against W3's table (same sid), coordinates included."""
    a = meta[["sid", "x_twd97", "y_twd97"]].reset_index(drop=True).join(mine)
    b = w3tab.set_index("sid")
    fin_mine = set(a.loc[np.isfinite(a["ground_elev_dtm_ref_m"]), "sid"])
    fin_w3 = set(b.index[np.isfinite(b["ground_elev_dtm_ref_m"])])
    j = a.set_index("sid").join(b[["x", "y", *COMPARE_COLS]], rsuffix="_w3", how="inner")
    dxy = np.hypot(j["x_twd97"] - j["x"], j["y_twd97"] - j["y"])
    out = {"n_mine_finite": len(fin_mine), "n_w3_finite": len(fin_w3),
           "finite_sets_equal": fin_mine == fin_w3,
           "only_mine": sorted(fin_mine - fin_w3), "only_w3": sorted(fin_w3 - fin_mine),
           "n_joined": int(len(j)), "coord_max_diff_m": float(np.nanmax(dxy)) if len(j) else None,
           "coord_n_compared": int(np.isfinite(dxy).sum()), "max_abs_diff": {}}
    for c in COMPARE_COLS:
        d = (j[c] - j[f"{c}_w3"]).abs()
        both_nan = j[c].isna() & j[f"{c}_w3"].isna()
        out["max_abs_diff"][c] = float(d[~both_nan].max()) if (~both_nan).any() else 0.0
        out.setdefault("nan_pattern_equal", {})[c] = bool(
            (j[c].isna() == j[f"{c}_w3"].isna()).all())
    out["max_abs_diff_all"] = max(out["max_abs_diff"].values())
    out["pass"] = bool(out["finite_sets_equal"] and out["max_abs_diff_all"] <= 1e-6
                       and all(out["nan_pattern_equal"].values())
                       and (out["coord_max_diff_m"] is None or out["coord_max_diff_m"] <= 0.01))
    return out


# ----------------------------------------------------------------------------- status
def raw_status(gh: pd.Series) -> np.ndarray:
    """'valid' (a real agency value), 'zero_missing_code' (exactly 0.0) or 'missing_nan'."""
    g = pd.to_numeric(gh, errors="coerce")
    return np.where(g.isna(), "missing_nan", np.where(g == 0.0, "zero_missing_code", "valid"))


def calibration_mask(df: pd.DataFrame) -> np.ndarray:
    """Stations entering the sd calibration (registered): a real agency GroundHeight, a DTM
    value, established by 2022-12, and not one of the two stations whose residual was shown
    before the registration."""
    return ((df["raw_status"] == "valid").to_numpy()
            & np.isfinite(df["ground_elev_dtm_ref_m"].to_numpy())
            & (pd.to_numeric(df["EstablishDate"], errors="coerce") <= EST_CUTOFF).to_numpy()
            & ~df["sid"].isin(SEEN_BEFORE_REGISTRATION).to_numpy())


# ----------------------------------------------------------------------------- sd calibration
def tau_needed(r: np.ndarray, s: np.ndarray, z: float = Z90) -> np.ndarray:
    """Smallest tau >= 0 with |r| <= z sqrt(s**2 + tau**2), per station."""
    r = np.abs(np.asarray(r, dtype="float64"))
    s = np.asarray(s, dtype="float64")
    return np.sqrt(np.maximum((r / z) ** 2 - s ** 2, 0.0))


def conformal_tau(taus: np.ndarray, level: float = LEVEL) -> float:
    """Split-conformal quantile: the ceil((n + 1) level)-th smallest needed tau (the largest
    when that rank exceeds n)."""
    t = np.sort(np.asarray(taus, dtype="float64"))
    n = len(t)
    if n == 0:
        return float("nan")
    k = min(math.ceil((n + 1) * level), n)
    return float(t[k - 1])


def lobo_coverage(r: np.ndarray, s: np.ndarray, blocks: np.ndarray, z: float = Z90,
                  level: float = LEVEL) -> tuple[np.ndarray, pd.DataFrame]:
    """Leave-one-block-out: tau from the other blocks' stations, hits on the block's own."""
    r = np.asarray(r, dtype="float64")
    s = np.asarray(s, dtype="float64")
    blocks = np.asarray(blocks)
    need = tau_needed(r, s, z)
    hits = np.zeros(len(r), dtype=bool)
    rows = []
    for b in sorted(set(blocks.tolist())):
        inb = blocks == b
        tb = conformal_tau(need[~inb], level)
        half = z * np.sqrt(s[inb] ** 2 + tb ** 2)
        hits[inb] = np.abs(r[inb]) <= half
        rows.append({"block": b, "n_stations": int(inb.sum()), "n_train": int((~inb).sum()),
                     "tau_train_m": tb, "hits": int(hits[inb].sum())})
    return hits, pd.DataFrame(rows)


def mad_sd(x) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def describe(x) -> dict:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(len(x)), "median": float(q[2]), "iqr": [float(q[1]), float(q[3])],
            "p05": float(q[0]), "p95": float(q[4]), "mad_sd": mad_sd(x),
            "mean": float(x.mean()), "abs_gt_1m": float(np.mean(np.abs(x) > 1.0)),
            "abs_gt_5m": float(np.mean(np.abs(x) > 5.0))}


# ----------------------------------------------------------------------------- product table
def assemble(meta: pd.DataFrame, dtm: pd.DataFrame, model_wells: set, tau: float,
             sd_status: str, product_version: str) -> pd.DataFrame:
    """One row per station (plan 2.4 long-table columns plus provenance and DTM columns)."""
    d = meta.reset_index(drop=True).join(dtm)
    d["raw_status"] = raw_status(d["GroundHeight"])
    gh = pd.to_numeric(d["GroundHeight"], errors="coerce").to_numpy(dtype="float64")
    z = d["ground_elev_dtm_ref_m"].to_numpy(dtype="float64")
    sd_cal = np.sqrt(d["sd_m"].to_numpy(dtype="float64") ** 2 + tau ** 2)
    valid = d["raw_status"].to_numpy() == "valid"
    repl = ~valid & np.isfinite(z)
    flag = np.where(valid, 0, np.where(repl, 5, 8)).astype(np.uint8)
    value = np.where(valid, gh, np.where(repl, z, np.nan))
    sd = np.where(repl, sd_cal, np.nan)
    noxy = ~np.isfinite(d["x_twd97"].to_numpy(dtype="float64"))
    method = np.where(
        valid, "agency_GroundHeight_as_stored",
        np.where(repl, np.where(d["raw_status"] == "zero_missing_code",
                                "w3_dtm20m_point_2017.5_replaces_zero_code",
                                "w3_dtm20m_point_2017.5_replaces_missing"),
                 np.where(noxy, "not_replaced_no_coordinates", "not_replaced_outside_dtm")))
    src = np.where(valid, "fan_stations.GroundHeight via heads/L1_clean/v3",
                   np.where(repl, "ground_elevation/v3 (dtm20m_mosaic, leveling epoch "
                                  "correction) at heads/L1_clean/v3 coordinates", ""))
    est = pd.to_numeric(d["EstablishDate"], errors="coerce")
    resid = np.where(valid & np.isfinite(z), gh - z, np.nan)
    wd = pd.to_numeric(d["WellDepth"], errors="coerce").to_numpy(dtype="float64")
    out = pd.DataFrame({
        "sid": d["sid"].astype(str), "site_id": d["site_id"].astype(str),
        "county": d["CountyName"], "model_well": d["sid"].isin(model_wells),
        "established_after_origin": (est > EST_CUTOFF).to_numpy(),
        "raw_ground_height_m": gh, "raw_status": d["raw_status"],
        "value": value, "sd": sd, "flag": flag, "method": method,
        "epoch": np.where(repl, T_REF, np.nan), "sd_status": np.where(repl, sd_status, ""),
        "product_version": product_version, "source_ids": src,
        "dtm_point_ref_m": z, "dtm_sd_w3_m": d["sd_m"].to_numpy(dtype="float64"),
        "dtm_sd_cal_m": np.where(np.isfinite(z), sd_cal, np.nan),
        "dtm_local_sd_m": d["dtm_local_sd_m"].to_numpy(dtype="float64"),
        "dtm_point_epoch": d["dtm_point_epoch"].to_numpy(dtype="float64"),
        "rate_2012_2021_m_per_yr": d["rate_2012_2021_m_per_yr"].to_numpy(dtype="float64"),
        "agency_minus_dtm_m": resid,
        "agency_dtm_gross_discrepancy": np.isfinite(resid) & (np.abs(resid) > GROSS_M),
        "well_depth_m": wd, "well_bottom_elev_m": value - wd,
        "well_elevation_m": pd.to_numeric(d["WellElevation"], errors="coerce").to_numpy(),
        "xy_flag": d["xy_flag"].astype("int64"),
    })
    return out


def completeness(tab: pd.DataFrame) -> dict:
    """C1: every zero/missing station is replaced (flag 5, finite sd > 0) or flag 8 with a
    reason; all model wells with GroundHeight 0.0 are replaced."""
    zm = tab["raw_status"] != "valid"
    rep = tab["flag"] == 5
    sd_ok = np.isfinite(tab["sd"]) & (tab["sd"] > 0)
    m0 = tab["model_well"] & (tab["raw_status"] == "zero_missing_code")
    mn = tab["model_well"] & (tab["raw_status"] == "missing_nan")
    f8 = zm & (tab["flag"] == 8)
    reasons = tab.loc[f8, "method"].value_counts().to_dict()
    accounted = int((zm & ((rep & sd_ok) | (f8 & tab["method"].str.startswith("not_replaced"))))
                    .sum())
    out = {"zero_or_missing": int(zm.sum()),
           "zero_code": int((tab["raw_status"] == "zero_missing_code").sum()),
           "missing_nan": int((tab["raw_status"] == "missing_nan").sum()),
           "replaced_flag5": int((zm & rep).sum()), "flag5_with_sd": int((rep & sd_ok).sum()),
           "flag8_not_replaced": int(f8.sum()), "flag8_reasons": reasons,
           "accounted": accounted,
           "model_wells_zero": int(m0.sum()), "model_wells_zero_replaced": int((m0 & rep & sd_ok)
                                                                                .sum()),
           "model_wells_missing_nan": int(mn.sum()),
           "model_wells_missing_replaced": int((mn & rep & sd_ok).sum()),
           "model_wells_without_value": int((tab["model_well"] & tab["value"].isna()).sum())}
    out["pass"] = bool(out["accounted"] == out["zero_or_missing"]
                       and out["model_wells_zero_replaced"] == out["model_wells_zero"]
                       and out["model_wells_zero"] > 0)
    return out


def integrity(tab: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """C3: long-table validator, agency values untouched, no 0.0 code left as a value,
    flag-5 rows with sd > 0, no personal or contact columns."""
    from common.flags import validate_long_table
    errs = validate_long_table(tab)
    m = meta.set_index("sid")
    t = tab.set_index("sid")
    f0 = t["flag"] == 0
    gh = pd.to_numeric(m.loc[t.index[f0], "GroundHeight"], errors="coerce")
    agency_equal = bool(np.array_equal(t.loc[f0, "value"].to_numpy(), gh.to_numpy()))
    zero_left = int(((t["value"] == 0.0) & (t["raw_status"] != "valid")).sum())
    f5 = t["flag"] == 5
    sd_bad = int((f5 & ~(np.isfinite(t["sd"]) & (t["sd"] > 0))).sum())
    forbidden = [c for c in tab.columns if any(k in c.lower() for k in (
        "contact", "phone", "tel", "name", "address", "戶名", "地址", "owner"))]
    out = {"validator_errors": errs, "agency_values_unchanged": agency_equal,
           "n_flag0": int(f0.sum()), "zero_code_values_left": zero_left,
           "flag5_rows_bad_sd": sd_bad, "personal_columns": forbidden,
           "rows": int(len(tab)), "unique_sids": int(tab["sid"].nunique())}
    out["pass"] = bool(not errs and agency_equal and zero_left == 0 and sd_bad == 0
                       and not forbidden and out["rows"] == out["unique_sids"] == len(meta))
    return out


# ----------------------------------------------------------------------------- use policy
ALLOWED_USES = ["validator", "model_design", "design_time_evidence", "after_the_fact_analysis"]
FORBIDDEN_USES = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                  "calibration_target", "selection", "prior", "prior_width_input",
                  "forcing_frozen", "forcing_recorded_arm", "initial_conditions", "idw_field",
                  "statistical_layer_history"]


def policy_probe(policy: dict) -> dict:
    """C4: guard.check_use on the product's use policy (the fields the manifest will carry):
    gate-exempt uses granted, gated uses refused while the gate is closed (granted when it is
    open), every forbidden use refused."""
    res = {}
    for use in ALLOWED_USES + FORBIDDEN_USES:
        try:
            guard.check_use(policy, use)
            res[use] = "granted"
        except PermissionError as exc:
            res[use] = type(exc).__name__
    gate_open = bool(policy["gate"]["passed"])
    want = {}
    for use in ALLOWED_USES:
        if use in ("design_time_evidence", "after_the_fact_analysis"):
            want[use] = "granted"
        else:
            want[use] = "granted" if gate_open else "GateNotPassed"
    for use in FORBIDDEN_USES:
        want[use] = "UseNotAllowed"
    bad = {u: {"got": res[u], "want": want[u]} for u in want if res[u] != want[u]}
    return {"results": res, "expected": want, "mismatches": bad, "pass": not bad}
