"""Freeze, from pre-origin rows only, the baselines and well tables E02c and the external-well
test register (the lead runs this BEFORE registration; nothing here reads a window head).

``e02c``  CL (month-of-year climatology) and CLT (climatology plus one linear trend), the
          ``twin.drift_diag.fair_baselines`` pair, for the 158 wells of E02 on the window
          months, from heads/L1_clean/v3 opened through ``common.guard`` with use
          ``baseline`` and ``window_end`` 2022-12 (flags 0-1). Also the well table
          (county, TWD97 coordinates from station_meta_preorigin, the forward run's cell and
          layer, the 5 km block) and two steps on pre-origin or model-only quantities:
          (a) the declared reference rule below on |CL_L1 - E02 baseline_m| (both
          pre-origin; it decides which CL / CLT are frozen), (b) the spread over window
          months of E02 predicted_m
          minus the forward run's heads at the well's cell and layer (model-only; ~0 when the
          cell matches the one E02 used).
``ext``   E02's protocol at the never-calibrated external wells: the N1:ext_wells ids of
          N1.cpu_response.v1, the rule (active cell of the forward grid, layer code 1-4,
          >= 24 observed pre-origin months, residual sd from >= 12 months) applied to
          pre-origin rows only; datum and residual sd (use ``datum_fit``) and CL/CLT (use
          ``baseline``) from L1_clean v3 sliced to 2022-12; the frozen forward run
          (sha cf2c7154) for the model; written in E02's predictions/protocol/manifest format.

Reference rule (declared before the export is run; pre-origin quantities only). ZR carries
E02's datum (fitted on the AMP_V2 record) while CL / CLT come from L1_clean v3; a per-well
datum offset between the two would decide T-level by reference, not skill. So: m = median
over wells of mean_window |CL_L1 - E02 baseline_m| (both pre-origin month-of-year
climatologies). If m <= 0.05 m, CL = CL_L1 and CLT = CLT_L1 (the judge's specification);
otherwise CL = E02 baseline_m and CLT = baseline_m + (CLT_L1 - CL_L1) (E02's own reference,
L1's trend departure). The choice is written to summary.json and registered.

Every output is written once (open mode 'x') with a summary.json of sha256s. ``--verify``
rebuilds into a temporary directory next to ``--out`` and writes verify.json (the CSVs must
reproduce bit-identically; the registration scripts refuse ``--write`` without it). Station
availability in the window is NOT looked at here (the well list is fixed by rule first).

Usage (repository root):
    python -I dataprep_staging/e02c/export_baselines.py e02c --out results/dataprep/reports/E02c/baselines
    python -I dataprep_staging/e02c/export_baselines.py e02c --verify --out <same dir>
    python -I dataprep_staging/e02c/export_baselines.py ext --window 2026-11 \
        --out results/twin/prospective_2026-11_ext
    python -I dataprep_staging/e02c/export_baselines.py ext --window 2026-11 --verify --out <same>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_e02c as S  # noqa: E402

L1 = "results/dataprep/heads/L1_clean/v3"
META_COLS = ["sid", "x_twd97", "y_twd97", "layer_code", "CountyName"]
MIN_FIT_MONTHS = 24
MIN_SD_MONTHS = 12
SIGMA_FLOOR = 1e-3
Z90 = 1.6448536269514722
REF_THRESHOLD_M = 0.05
REF_L1 = "L1_clean_v3"
REF_E02 = "e02_baseline_aligned"


# ----------------------------------------------------------------------------- pure parts
def axis(last: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{S.AXIS_START}-01", f"{last}-01", freq="MS")


def obs_matrix(long: pd.DataFrame, ids: list[str], ax: pd.DatetimeIndex,
               t_fit: int = S.T_FIT) -> np.ndarray:
    """(W, T) observed monthly heads (flags 0-1) on ``ax``; only the first ``t_fit`` months
    (to 2022-12) are filled, whatever the table holds (a second guard after the slice)."""
    d = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp()
    ok = long["flag"].isin([0, 1]).to_numpy() & (d <= ax[t_fit - 1]).to_numpy()
    sub = pd.DataFrame({"sid": long["sid"].astype(str).to_numpy()[ok], "date": d[ok],
                        "value": long["value"].to_numpy(float)[ok]})
    if sub.duplicated(["sid", "date"]).any():
        raise ValueError("duplicate station-months in the pre-origin table")
    si = {s: i for i, s in enumerate(ids)}
    ti = {t: j for j, t in enumerate(ax[:t_fit])}
    sub = sub[sub["sid"].isin(si)]
    M = np.full((len(ids), len(ax)), np.nan)
    M[sub["sid"].map(si).to_numpy(), sub["date"].map(ti).to_numpy()] = sub["value"]
    return M


def baselines_table(ids: list[str], obs: np.ndarray, ax: pd.DatetimeIndex,
                    first: str, last: str, t_fit: int = S.T_FIT) -> pd.DataFrame:
    """CL and CLT (drift_diag.fair_baselines, month_offset 0: column 0 is January 2012)
    on the window months."""
    from hydrophysics.twin.drift_diag import fair_baselines

    b = fair_baselines(obs.copy(), t_fit, month_offset=0)
    win = (ax >= pd.Timestamp(f"{first}-01")) & (ax <= pd.Timestamp(f"{last}-01"))
    rows = []
    for w, s in enumerate(ids):
        for j in np.flatnonzero(win):
            rows.append({"station_id": s, "date": str(ax[j].date()),
                         "cl_m": float(b["clim"][w, j]), "clt_m": float(b["clim_trend"][w, j]),
                         "n_fit_months": int(np.isfinite(obs[w, :t_fit]).sum())})
    return pd.DataFrame(rows)


def reference_rule(pred: pd.DataFrame, base: pd.DataFrame,
                   threshold: float = REF_THRESHOLD_M) -> tuple[pd.DataFrame, dict]:
    """The declared reference rule (module docstring) on pre-origin quantities: ``base``
    holds CL_L1 / CLT_L1 (``cl_m``, ``clt_m``) on the window rows, ``pred`` E02's registered
    rows (``baseline_m``). Returns the frozen table (``cl_m``, ``clt_m`` as chosen, plus the
    L1 and E02 columns for the record) and the decision."""
    p = pred[["station_id", "date", "baseline_m"]].copy()
    p["station_id"] = p["station_id"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    b = base.copy()
    b["station_id"] = b["station_id"].astype(str)
    b["date"] = pd.to_datetime(b["date"]).dt.strftime("%Y-%m-%d")
    m = b.merge(p, on=["station_id", "date"], how="left")
    per = (m["cl_m"] - m["baseline_m"]).abs().groupby(m["station_id"]).mean()
    med = float(per.median()) if per.notna().any() else float("nan")
    chosen = REF_L1 if np.isfinite(med) and med <= threshold else REF_E02
    t = m.rename(columns={"cl_m": "cl_l1_m", "clt_m": "clt_l1_m",
                          "baseline_m": "e02_baseline_m"})
    if chosen == REF_L1:
        t["cl_m"], t["clt_m"] = t["cl_l1_m"], t["clt_l1_m"]
    else:
        t["cl_m"] = t["e02_baseline_m"]
        t["clt_m"] = t["e02_baseline_m"] + (t["clt_l1_m"] - t["cl_l1_m"])
    cols = ["station_id", "date", "cl_m", "clt_m", "n_fit_months", "cl_l1_m", "clt_l1_m",
            "e02_baseline_m"]
    info = {"rule": "median over wells of mean_window |CL_L1 - E02 baseline_m| <= "
                    f"{threshold} m -> {REF_L1}; else {REF_E02} (CL = baseline_m, CLT = "
                    "baseline_m + CLT_L1 - CL_L1)",
            "threshold_m": threshold, "median_abs_cl_l1_minus_e02_baseline_m": med,
            "n_wells_compared": int(per.notna().sum()),
            "n_wells_over_threshold": int((per > threshold).sum()), "chosen": chosen}
    return t[cols], info


def grid_of(fw: dict):
    from hydrophysics.twin.grid import FanGrid

    return FanGrid(int(fw["nx"]), int(fw["ny"]), float(fw["dx"]), float(fw["x0"]),
                   float(fw["y0"]), np.asarray(fw["mask"], bool))


def well_table(meta: pd.DataFrame, ids: list[str], fw: dict) -> pd.DataFrame:
    """County, coordinates, forward-grid cell, twin layer (layer_code - 1), 5 km block."""
    g = grid_of(fw)
    m = meta.copy()
    m["sid"] = m["sid"].astype(str)
    m = m.drop_duplicates("sid").set_index("sid")
    rows = []
    for s in ids:
        r = m.loc[s] if s in m.index else None
        x = float(r["x_twd97"]) if r is not None else np.nan
        y = float(r["y_twd97"]) if r is not None else np.nan
        lc = str(r["layer_code"]) if r is not None else ""
        cell = g.active_index(x, y) if np.isfinite(x) and np.isfinite(y) else None
        rows.append({"station_id": s, "county": S.county_of(s), "x": x, "y": y,
                     "layer_code": lc,
                     "layer": int(lc) - 1 if lc in ("1", "2", "3", "4") else -1,
                     "cell": -1 if cell is None else int(cell)})
    t = pd.DataFrame(rows)
    ok = np.isfinite(t["x"]) & np.isfinite(t["y"])
    t["block5"] = "none"
    t.loc[ok, "block5"] = S.block_ids(t.loc[ok, "x"], t.loc[ok, "y"])
    return t


def model_series(fw: dict, cell: int, layer: int) -> tuple[np.ndarray, np.ndarray]:
    sc = list(map(str, fw["scenario_names"])).index("baseline")
    return (np.asarray(fw["heads_mean"][sc, layer, cell], float),
            np.asarray(fw["heads_std"][sc, layer, cell], float))


def ext_predictions(wells: pd.DataFrame, obs: np.ndarray, fw: dict, first: str, last: str,
                    t_fit: int = S.T_FIT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """E02's recipe (twin.prospective.register) at wells given by cell/layer: datum = mean
    fitted residual, sd = fitted residual sd (>= 12 months), sigma = ens and sd in
    quadrature, central 90 %. Returns (predictions, baselines, selection)."""
    from hydrophysics.twin.coverage import _residual_sd, fitted_offsets
    from hydrophysics.twin.drift_diag import fair_baselines

    dates = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    if dates[0] != pd.Timestamp(f"{S.AXIS_START}-01") or int(fw["origin"]) != t_fit - 1:
        raise ValueError("forward run axis is not 2012-01 with origin 2022-12")
    T = len(dates)
    if obs.shape != (len(wells), T):
        raise ValueError(f"obs {obs.shape} != ({len(wells)}, {T})")
    W = len(wells)
    pred = np.full((W, T), np.nan)
    sig_e = np.full((W, T), np.nan)
    placed = (wells["cell"].to_numpy() >= 0) & wells["layer"].isin([0, 1, 2, 3]).to_numpy()
    for w in np.flatnonzero(placed):
        pred[w], sig_e[w] = model_series(fw, int(wells["cell"].iat[w]),
                                         int(wells["layer"].iat[w]))
    hist = obs.copy()
    hist[:, t_fit:] = np.nan
    offsets, count = fitted_offsets(hist, pred, t_fit)
    model = pred + offsets[:, None]
    sd = _residual_sd(hist, model, t_fit, min_months=MIN_SD_MONTHS)
    b = fair_baselines(hist.copy(), t_fit, month_offset=0)
    sigma = np.sqrt(np.maximum(sig_e, SIGMA_FLOOR) ** 2 + sd[:, None] ** 2)
    keep = placed & (count >= MIN_FIT_MONTHS) & np.isfinite(sd) & np.isfinite(offsets)
    win = (dates >= pd.Timestamp(f"{first}-01")) & (dates <= pd.Timestamp(f"{last}-01"))
    if win.sum() != 12:
        raise ValueError("the forward run does not cover the 12 window months")
    sel = wells.assign(n_fit_months=count, residual_sd_m=sd, datum_m=offsets, kept=keep)
    sel["reason"] = np.where(~placed, "no active cell or layer code outside 1-4",
                             np.where(count < MIN_FIT_MONTHS, "< 24 pre-origin months",
                                      np.where(~np.isfinite(sd), "sd from < 12 months",
                                               "kept")))
    P, Bl = [], []
    msha = str(fw.get("_sha256", S.MODEL_SHA))
    for w in np.flatnonzero(keep):
        s = wells["station_id"].iat[w]
        for t in np.flatnonzero(win):
            P.append({"station_id": s, "layer": int(wells["layer"].iat[w]) + 1,
                      "date": str(dates[t].date()), "prediction_origin": str(dates[t_fit - 1]
                                                                            .date()),
                      "predicted_m": float(model[w, t]), "baseline_m": float(b["clim"][w, t]),
                      "lower_m": float(model[w, t] - Z90 * sigma[w, t]),
                      "upper_m": float(model[w, t] + Z90 * sigma[w, t]),
                      "model_sha256": msha})
            Bl.append({"station_id": s, "date": str(dates[t].date()),
                       "cl_m": float(b["clim"][w, t]), "clt_m": float(b["clim_trend"][w, t]),
                       "n_fit_months": int(count[w])})
    return pd.DataFrame(P), pd.DataFrame(Bl), sel


def ext_protocol(first: str, last: str, registered_before: str) -> dict:
    """E02's protocol (schema 1), min_stations 30 for the external wells."""
    end = (pd.Timestamp(f"{last}-01") + pd.offsets.MonthEnd(0)).date()
    return {"schema": 1, "task": "free-running-head-validation",
            "test_start": f"{first}-01", "test_end": str(end),
            "registered_before": registered_before, "independent_holdout": True,
            "model_sha256": S.MODEL_SHA,
            "baseline": "month-of-year climatology of each well's observed months before "
                        "the origin (heads/L1_clean/v3, flags 0-1, sliced to 2022-12)",
            "max_rmse_ratio": 1.0, "min_stations": S.MIN_WELLS_EXT,
            "min_months": S.MIN_MONTHS, "coverage_bounds": [0.85, 0.95],
            "interval": "central 90 %, combined sigma", "update_method": "innovation",
            "forecast_origin": "2022-12-01",
            "datum": "per-well mean residual over observed months 2012-01..2022-12 "
                     "(L1_clean v3, never used in calibration)",
            "wells": "N1:ext_wells (N1.cpu_response.v1) with >= 24 pre-origin months",
            "forcing_after_origin": "month-of-year climatology of the 2012-2022 record "
                                    "(baseline scenario)",
            "note": "Observations in the window did not exist at registration; score with "
                    "dataprep_staging/e02c/score_e02c.py score-ext after E02c."}


# ----------------------------------------------------------------------------- I/O (lead)
def _load_forward() -> dict:
    with np.load(REPO / S.FORWARD, allow_pickle=False) as z:
        fw = {k: z[k] for k in z.files}
    fw["_sha256"] = S.sha256(REPO / S.FORWARD)
    if fw["_sha256"] != S.MODEL_SHA:
        raise SystemExit(f"forward sha {fw['_sha256']} != registered {S.MODEL_SHA}")
    return fw


def _open_l1(use: str):
    from common import guard

    heads = guard.open_product(REPO / L1, use, window_end=S.ORIGIN)
    meta = guard.open_product(REPO / L1, use)          # per-output route (fit_window_end 2022-12)
    long = heads.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag"])
    m = meta.read_table("station_meta_preorigin.parquet", columns=META_COLS)
    return long, m, heads.notes + meta.notes


def _write_csv(df: pd.DataFrame, p: Path) -> None:
    with open(p, "x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)


def _write_json(obj: dict, p: Path) -> None:
    with open(p, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=str)
        f.write("\n")


def export_e02c(out: Path) -> dict:
    pred = pd.read_csv(REPO / S.E02_DIR / "predictions.csv", dtype={"station_id": str})
    if S.sha256(REPO / S.E02_DIR / "predictions.csv") != S.E02_PRED_SHA:
        raise SystemExit("E02 predictions changed")
    ids = sorted(pred["station_id"].unique())
    fw = _load_forward()
    long, meta, notes = _open_l1("baseline")
    ax = axis(S.WINDOW[1])
    obs = obs_matrix(long, ids, ax)
    base_l1 = baselines_table(ids, obs, ax, *S.WINDOW)
    wells = well_table(meta, ids, fw)
    # (a) the declared reference rule (pre-origin vs pre-origin), (b) cell check (model only)
    base, ref_info = reference_rule(pred, base_l1)
    ref = (base["cl_l1_m"] - base["e02_baseline_m"]).abs().groupby(base["station_id"]).mean()
    dates = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    spread = {}
    for _, r in wells.iterrows():
        if r["cell"] < 0 or r["layer"] < 0:
            spread[r["station_id"]] = np.nan
            continue
        h, _ = model_series(fw, int(r["cell"]), int(r["layer"]))
        p = pred[pred["station_id"] == r["station_id"]]
        j = dates.get_indexer(pd.to_datetime(p["date"]))
        spread[r["station_id"]] = float(np.std(p["predicted_m"].to_numpy() - h[j]))
    wells["cell_check_spread_m"] = wells["station_id"].map(spread)
    wells["cell_matches_e02"] = wells["cell_check_spread_m"] < 1e-3
    wells["ref_abs_cl_minus_e02_baseline_m"] = wells["station_id"].map(ref)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(base, out / "baselines_e02c.csv")
    _write_csv(wells, out / "wells_158.csv")
    summ = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "what": "E02c pre-origin baselines CL/CLT and the 158-well table",
            "product": L1, "use": "baseline", "window_end": S.ORIGIN, "guard_notes": notes,
            "n_wells": len(ids), "n_rows": int(len(base)),
            "wells_without_clt_trend": int((base.groupby("station_id").n_fit_months.first()
                                            < 13).sum()),
            "reference": ref_info,
            "cell_check": {"n_match": int(wells["cell_matches_e02"].sum()),
                           "n_wells": len(wells)},
            "files": {n: S.sha256(out / n) for n in ("baselines_e02c.csv", "wells_158.csv")},
            "code": {f"{S.E}/export_baselines.py": S.sha256(Path(__file__))}}
    _write_json(summ, out / "summary.json")
    return summ


def ext_ids() -> list[str]:
    doc = json.loads((REPO / f"results/dataprep/preregistration/{S.N1_PREREG}.json")
                     .read_text(encoding="utf-8"))
    for t in doc["tests"]:
        u = (t.get("external_units") or {}).get(S.N1_EXT_UNIT)
        if u:
            return sorted(map(str, u["ids"]))
    raise SystemExit(f"{S.N1_EXT_UNIT} not found in {S.N1_PREREG}")


def export_ext(out: Path, window: str) -> dict:
    import prereg_e02c_ext as PX
    spec = PX.WINDOWS[window]
    ids = ext_ids()
    fw = _load_forward()
    long_d, meta, notes_d = _open_l1("datum_fit")
    long_b, _, notes_b = _open_l1("baseline")
    ax = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    obs = obs_matrix(long_d, ids, ax)
    if not np.array_equal(np.isnan(obs), np.isnan(obs_matrix(long_b, ids, ax))):
        raise SystemExit("baseline and datum_fit slices differ")
    wells = well_table(meta, ids, fw)
    pred, base, sel = ext_predictions(wells, obs, fw, spec["first"], spec["last"])
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(pred, out / "predictions.csv")
    _write_csv(base, out / "baselines.csv")
    _write_csv(sel, out / "wells.csv")
    _write_json(ext_protocol(spec["first"], spec["last"], dt.date.today().isoformat()),
                out / "protocol.json")
    man = {"registered_on": dt.date.today().isoformat(),
           "stations": int(pred["station_id"].nunique()) if len(pred) else 0,
           "rows": int(len(pred)), "predictions_sha256": S.sha256(out / "predictions.csv"),
           "protocol_sha256": S.sha256(out / "protocol.json"), "forward": S.FORWARD,
           "model_sha256": S.MODEL_SHA}
    _write_json(man, out / "manifest.json")
    summ = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "what": "external-well predictions (E02 recipe) from pre-origin rows",
            "window": [spec["first"], spec["last"]], "n_candidates": len(ids),
            "n_kept": int(sel["kept"].sum()),
            "reasons": sel["reason"].value_counts().to_dict(),
            "kept_by_county": sel[sel.kept].county.value_counts().to_dict(),
            "guard_notes": notes_d + notes_b,
            "files": {n: S.sha256(out / n) for n in ("predictions.csv", "baselines.csv",
                                                      "wells.csv", "protocol.json",
                                                      "manifest.json")},
            "code": {f"{S.E}/export_baselines.py": S.sha256(Path(__file__))}}
    _write_json(summ, out / "summary.json")
    return summ


def verify_export(out: Path, build) -> dict:
    """Engineering gate: rebuild with ``build(tmp_out)`` into a temporary directory next to
    ``out`` and compare every CSV listed in out/summary.json (and the reference decision)
    bit for bit. Writes out/verify.json once; ``identical`` False is a refusal for
    ``--write``."""
    out = Path(out)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    want = {n: h for n, h in (s.get("files") or {}).items() if n.endswith(".csv")}
    have = {n: S.sha256(out / n) for n in want}
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".verify_") as td:
        s2 = build(Path(td) / "rebuild")
        got = {n: S.sha256(Path(td) / "rebuild" / n) for n in want}
    problems = [f"{n}: summary {want[n][:12]} / on disk {have[n][:12]} / rebuilt "
                f"{got[n][:12]}" for n in want if not want[n] == have[n] == got[n]]
    if s.get("reference") != s2.get("reference"):
        problems.append("reference decision differs on rebuild")
    rec = {"verified": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "identical": not problems, "files": want, "problems": problems,
           "summary_sha256": S.sha256(out / "summary.json"),
           "code": {f"{S.E}/export_baselines.py": S.sha256(Path(__file__))}}
    _write_json(rec, out / "verify.json")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("e02c")
    a1.add_argument("--out", required=True, type=Path)
    a1.add_argument("--verify", action="store_true", help="rebuild and compare (gate)")
    a2 = sub.add_parser("ext")
    a2.add_argument("--out", required=True, type=Path)
    a2.add_argument("--window", choices=("2026-11", "2026-12"), default="2026-11")
    a2.add_argument("--verify", action="store_true", help="rebuild and compare (gate)")
    a = ap.parse_args(argv)
    if a.cmd == "e02c":
        def build(o):
            return export_e02c(o)
    else:
        def build(o):
            return export_ext(o, a.window)
    if a.verify:
        r = verify_export(a.out, build)
        print(json.dumps(r, indent=1))
        return 0 if r["identical"] else 1
    s = build(a.out)
    print(json.dumps({k: v for k, v in s.items() if k != "guard_notes"}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
