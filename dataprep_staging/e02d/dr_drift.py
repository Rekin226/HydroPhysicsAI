"""DR.relative_drift.v1 (round 7, rank 4): ONE declared design-time read of the page model's
fitted-month residuals, 2012-01..2022-12, at E02's 158 wells. It fixes E02d's grouping rule
and its drift term by rules declared before the read (``prereg_design.py dr``); it scores
nothing and is never validation.

What it reads (the lead only, after the declaration is registered): the frozen forward
``results/twin_forward/datum_gate_innov.npz`` (sha cf2c7154) and the pre-origin monthly
heads E02's own registration read (``twin.coverage.assemble``: AMP_V2/data/wells through
``twin.heads.build_head_field`` with t1 = 2023-01-01, use ``datum_fit`` /
``design_time_evidence``). Nothing after 2022-12 is used, no window head exists, nothing
is written under results/dataprep/preregistration except the one look this script logs.

Steps (all declared):
 0. Engineering gate: E02's recipe is rebuilt from these arrays (datum = mean fitted
    residual, sd = fitted residual sd, combined sigma) and must reproduce every
    predicted_m / lower_m / upper_m of results/twin/prospective_2026-11/predictions.csv to
    1e-6 m, on exactly its 158 wells. Otherwise nothing below is computed.
 1. Residuals r = observed - (forward + datum), fitted months. Per county (Changhua
    separate): ICC of r over county-months; ICC of the county-relative r (LOO median)
    over county x layer-group-months and county x zone-months (groups below 8 wells
    merged by the declared order); the grouping rule (``e02dlib.choose_grouping``); the
    ICC left after the chosen grouping (bounds U1.decomp_power.v1's grid).
 2. The relative residual e (LOO median of the chosen group, >= 5 others), its per-well
    root mean square ``sd_fit_rel`` (E02d's in-sample part), the well's datum centre.
 3. Split-half drift (``e02dlib.drift_county``), tau^2 a lower bound, and the 10 % rule
    at the window-A lead. Where the lead is measured from is the declaration's
    ``lead_definition``, chosen by the lead at ``prereg_design.py dr --write --lead-from``:
    each well's datum centre (the datum is the mean of 2012-22, so a linear drift
    accumulates from there) or the forecast origin; the judge's literal 4-year reading is
    printed beside it, report-only. The page number is the noise-corrected excess
    sqrt(max(0, mean(d^2 - E))) = 6 tau (a lower bound); RMS(d) includes sampling noise
    and is report-only. Column zone ids outside 0..2 are an engineering failure.
 4. Outputs, written once: results/dataprep/reports/DR.relative_drift.v1/{report.json,
    wells_e02d.csv}. wells_e02d.csv is the frozen per-well table E02d's artefact uses.

Usage (repository root):
    python -I dataprep_staging/e02d/dr_drift.py run           (the lead, once; logs the look)
    python -I dataprep_staging/e02d/dr_drift.py rehearsal --out <dir outside results/>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402

VERSION = "1.0.0"
FROZEN = (f"{io.X}/dr_drift.py", f"{io.X}/e02dlib.py", f"{io.X}/e02dio.py")
REPORT_DIR = f"results/dataprep/reports/{L.DR_ID}"
WELL_COLS = ("station_id", "county", "layer", "layer_group", "zone", "cell", "n_fit",
             "datum_m", "sd_abs_m", "partition_layer", "partition_zone", "grouping", "group",
             "subgroup", "sd_fit_rel_m", "n_rel", "t_centre_year", "phi", "drift_included")


# ----------------------------------------------------------------------------- E02 arrays
def load_e02_arrays(forward: Path, stations: Path, wells_dir: Path) -> dict:
    """E02's own read (``twin.coverage.assemble``), plus each well's model cell and column
    zone. THE LOOK: reads pre-origin monthly heads (t1 = the month after the origin)."""
    from hydrophysics.twin.grid import FanGrid
    from hydrophysics.twin.heads import build_head_field
    from hydrophysics.twin.zone_county import column_zone_map

    with np.load(forward, allow_pickle=False) as saved:
        fw = {k: saved[k] for k in saved.files}
    dates = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    origin = int(fw["origin"])
    fit = origin + 1
    grid = FanGrid(int(fw["nx"]), int(fw["ny"]), float(fw["dx"]), float(fw["x0"]),
                   float(fw["y0"]), fw["mask"])
    hf = build_head_field(str(wells_dir), pd.read_parquet(stations),
                          t0=str(dates[0].date()),
                          t1=str((dates[origin] + pd.offsets.MonthBegin()).date()))
    meta = json.loads(str(fw["gate"]))
    zoc = column_zone_map(grid, meta)
    selected = [(i, grid.active_index(*xy)) for i, xy in enumerate(hf.xy)]
    selected = [(i, c) for i, c in selected if c is not None]
    scen = list(fw["scenario_names"]).index("baseline")
    W, T = len(selected), len(dates)
    historical = np.full((W, T), np.nan)
    predicted = np.empty((W, T))
    sigma_ens = np.empty((W, T))
    layers = np.empty(W, dtype=int)
    cells = np.empty(W, dtype=int)
    for w, (i, cell) in enumerate(selected):
        historical[w, :fit] = hf.heads[i]
        lay = int(hf.layers[i]) - 1
        layers[w], cells[w] = lay + 1, cell
        predicted[w] = fw["heads_mean"][scen, lay, cell]
        sigma_ens[w] = fw["heads_std"][scen, lay, cell]
    return {"ids": [hf.sids[i] for i, _ in selected], "dates": dates, "origin": origin,
            "fit": fit, "layers": layers, "cells": cells,
            "zones": np.asarray(zoc)[cells].astype(int), "historical": historical,
            "predicted": predicted, "sigma_ens": sigma_ens, "model_sha256": io.sha256(forward)}


def fitted_offsets(historical, predicted, fit):
    """twin.coverage.fitted_offsets (mean fitted-period residual and its support)."""
    res = historical[:, :fit] - predicted[:, :fit]
    count = np.isfinite(res).sum(axis=1)
    off = np.divide(np.nansum(res, axis=1), count, out=np.full(len(count), np.nan),
                    where=count > 0)
    return off, count


def residual_sd(historical, model, fit, min_months: int = 12):
    """twin.coverage._residual_sd (ddof 1, NaN below 12 months)."""
    res = historical[:, :fit] - model[:, :fit]
    n = np.isfinite(res).sum(axis=1)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(res, axis=1, ddof=1)
    sd[n < min_months] = np.nan
    return sd


def e02_recipe(a: dict) -> dict:
    off, count = fitted_offsets(a["historical"], a["predicted"], a["fit"])
    model = a["predicted"] + off[:, None]
    sd = residual_sd(a["historical"], model, a["fit"])
    sigma = np.sqrt(np.maximum(a["sigma_ens"], 1e-3) ** 2 + sd[:, None] ** 2)
    keep = (count >= 24) & np.isfinite(sd) & np.isfinite(off)
    return {"offsets": off, "count": count, "model": model, "sd": sd, "sigma": sigma,
            "keep": keep}


def datum_gate(a: dict, rec: dict, pred: pd.DataFrame) -> dict:
    """Step 0: the rebuilt recipe reproduces E02's registered rows to DATUM_TOL_M."""
    ids = [str(x) for x in a["ids"]]
    kept = sorted(ids[i] for i in np.flatnonzero(rec["keep"]))
    e02_ids = sorted(pred["station_id"].astype(str).unique())
    out = {"tol_m": L.DATUM_TOL_M, "wells_rebuilt": len(kept), "wells_e02": len(e02_ids),
           "same_wells": kept == e02_ids}
    if not out["same_wells"]:
        out.update({"ok": False, "why": "the keep rule does not give E02's wells",
                    "only_rebuilt": sorted(set(kept) - set(e02_ids))[:20],
                    "only_e02": sorted(set(e02_ids) - set(kept))[:20]})
        return out
    wi = {w: i for i, w in enumerate(ids)}
    ti = {d: i for i, d in enumerate(a["dates"])}
    p = pred.copy()
    m = pd.to_datetime(p["date"]).dt.to_period("M").dt.to_timestamp()
    r = p["station_id"].astype(str).map(wi).to_numpy(int)
    c = m.map(ti).to_numpy()
    if np.isnan(c.astype(float)).any():
        return {**out, "ok": False, "why": "registered dates outside the forward axis"}
    c = c.astype(int)
    z = L.Z90
    diffs = {"predicted_m": np.abs(rec["model"][r, c] - p["predicted_m"].to_numpy(float)),
             "lower_m": np.abs(rec["model"][r, c] - z * rec["sigma"][r, c]
                               - p["lower_m"].to_numpy(float)),
             "upper_m": np.abs(rec["model"][r, c] + z * rec["sigma"][r, c]
                               - p["upper_m"].to_numpy(float))}
    out["max_abs_diff_m"] = {k: float(np.nanmax(v)) for k, v in diffs.items()}
    out["rows"] = int(len(p))
    out["ok"] = all(v <= L.DATUM_TOL_M for v in out["max_abs_diff_m"].values())
    if not out["ok"]:
        out["why"] = "rebuilt rows differ from E02's registered predictions"
    return out


# ----------------------------------------------------------------------------- analysis
def _part(labels_county, order):
    lab, log = L.merge_small(labels_county, order)
    return lab, log, len(set(lab))


def analyse(a: dict, rec: dict, unit_ids: list[str],
            lead_definition: str = "datum_centre") -> tuple[dict, pd.DataFrame]:
    """Steps 1-3 on the unit's wells (pure; ``a`` real or synthetic). ``lead_definition``
    (e02dlib.LEAD_DEFINITIONS) is the one the registered declaration records."""
    ids = np.array([str(x) for x in a["ids"]])
    sel = np.flatnonzero(np.isin(ids, unit_ids) & rec["keep"])
    fit = a["fit"]
    fdates = a["dates"][:fit]
    hist = a["historical"][sel, :fit]
    model = rec["model"][sel, :fit]
    r = hist - model
    ids_u = ids[sel]
    county = np.array([L.county_of(s) for s in ids_u])
    layer = np.asarray(a["layers"])[sel]
    lgroup = np.array([L.LAYER_GROUP.get(int(x), "L34") for x in layer])
    zone = np.array([L.ZONE_NAMES[int(z)] for z in np.asarray(a["zones"])[sel]])
    W = len(sel)
    groups = np.empty(W, dtype=object)
    subgroups = np.empty(W, dtype=object)
    part_layer = np.empty(W, dtype=object)
    part_zone = np.empty(W, dtype=object)
    grouping = np.empty(W, dtype=object)
    E = np.full(r.shape, np.nan)
    rep: dict = {"counties": {}}
    tcent = np.array([L.year_frac(fdates[np.isfinite(r[w])]).mean()
                      if np.isfinite(r[w]).any() else np.nan for w in range(W)])
    anchor = L.lead_anchor(lead_definition, tcent)
    for c in L.COUNTIES:
        mask = county == c
        idx = np.flatnonzero(mask)
        cr: dict = {"wells": int(idx.size)}
        if idx.size < L.MIN_GROUP_WELLS:
            cr["evaluable"] = False
            rep["counties"][c] = cr
            continue
        rc = r[idx]
        Wc, T = rc.shape
        cr["icc_county_month"] = L.anova_icc(rc, L.cell_labels(np.zeros(Wc, int), Wc, T))
        e_cty = L.relative(rc, np.zeros(Wc, int))
        pl, logl, nl = _part(lgroup[idx], L.LAYER_ORDER)
        pz, logz, nz = _part(zone[idx], L.ZONE_NAMES)
        cr["partition_layer"] = {"groups": {g: int((pl == g).sum()) for g in sorted(set(pl))},
                                 "merges": logl}
        cr["partition_zone"] = {"groups": {g: int((pz == g).sum()) for g in sorted(set(pz))},
                                "merges": logz}
        icl = L.anova_icc(e_cty, L.cell_labels(pl, Wc, T))
        icz = L.anova_icc(e_cty, L.cell_labels(pz, Wc, T))
        cr["icc_layer_month"], cr["icc_zone_month"] = icl, icz
        ch = L.choose_grouping(icl["icc"] if nl >= 2 else float("nan"),
                               icz["icc"] if nz >= 2 else float("nan"), nl, nz)
        cr["grouping"] = ch
        if ch["grouping"] == "layer":
            gl, other = pl, pz
        elif ch["grouping"] == "zone":
            gl, other = pz, pl
        else:
            gl = np.array(["all"] * Wc, dtype=object)
            other = pl if (icl["icc"] if math.isfinite(icl["icc"]) else -1) >= (
                icz["icc"] if math.isfinite(icz["icc"]) else -1) else pz
        Ec = L.relative(rc, gl)
        E[idx] = Ec
        if ch["grouping"] == "county":
            rem = max([x for x in (icl["icc"], icz["icc"]) if math.isfinite(x)], default=0.0)
            boot = L.icc_boot_upper(e_cty, [pl, pz])
            cr["icc_remaining"] = {"icc": float(rem), "how": "county grouping: the larger of "
                                                             "the two partition ICCs"}
        else:
            ri = L.anova_icc(Ec, L.cell_labels(other, Wc, T))
            boot = L.icc_boot_upper(Ec, [other])
            cr["icc_remaining"] = {**ri, "how": "ICC of the chosen-group relative residual "
                                                "over the other partition's group-months"}
        cr["icc_remaining"]["boot_upper90"] = boot["upper"]
        cr["icc_remaining"]["boot"] = boot
        cr["icc_well_report_only"] = L.anova_icc(
            Ec, np.broadcast_to(np.arange(Wc).reshape(Wc, 1), (Wc, T)))
        dr = L.drift_county(Ec, fdates, np.ones(Wc, bool))
        rows = dr.pop("rows")
        n_rel = np.isfinite(Ec).sum(axis=1)
        sd_rel = np.where(n_rel >= L.MIN_FIT_MONTHS_SD,
                          np.sqrt(np.nanmean(np.where(np.isfinite(Ec), Ec ** 2, np.nan),
                                             axis=1)), np.nan)
        sig_c = float(np.nanmedian(sd_rel))
        inc = [x["w"] for x in rows if x["included"]]
        lead = float(np.median(L.window_mid("A") - anchor[idx][inc])) if inc else float("nan")
        cr["drift"] = dr
        cr["sigma_rel_median_m"] = sig_c
        cr["lead_definition"] = lead_definition
        cr["lead_years"] = lead
        cr["lead_years_datum_centre_report_only"] = (
            float(np.median(L.window_mid("A") - tcent[idx][inc])) if inc else float("nan"))
        if dr.get("evaluable"):
            rule = L.drift_rule(dr["tau2_per_year2"], sig_c, lead)
        else:
            rule = {"include_drift": False, "increase": float("nan"),
                    "why": f"not evaluable: {dr['share_included']:.2f} of the county's wells "
                           f"meet {L.MIN_HALF_MONTHS}+{L.MIN_HALF_MONTHS} months "
                           f"(< {L.INCLUSION_MIN_SHARE})"}
        cr["rule"] = rule
        cr["rule_literal_4y_report_only"] = L.drift_rule(dr.get("tau2_per_year2", float("nan")),
                                                         sig_c, L.LITERAL_LEAD_YEARS)
        t2 = dr.get("tau2_per_year2", float("nan"))
        aw = (t2 * lead ** 2 / (sig_c ** 2 + t2 * lead ** 2)
              if math.isfinite(t2) and math.isfinite(lead) and sig_c > 0 else 0.0)
        cr["window_well_share_hat"] = float(aw)
        exc6 = math.sqrt(max(0.0, dr["excess"])) if "excess" in dr else None
        cr["page_numbers"] = {
            "page_value_key": "excess_drift_over_6y_m",
            "excess_drift_over_6y_m": exc6,
            "label": "in-sample, group-relative, noise-corrected (sqrt(max(0, mean(d^2 - E))) "
                     "= 6 tau): a lower bound",
            "report_only": {
                "rms_split_half_drift_m": (math.sqrt(dr["mean_d2"]) if "mean_d2" in dr
                                           else None),
                "label": "includes sampling noise: overstates the drift, NOT a lower bound; "
                         "never on the page"}}
        cr["evaluable"] = True
        rep["counties"][c] = cr
        for j, w in enumerate(idx):
            groups[w] = c if ch["grouping"] == "county" else f"{c}:{gl[j]}"
            subgroups[w] = f"{c}:{other[j]}"
            part_layer[w], part_zone[w] = pl[j], pz[j]
            grouping[w] = ch["grouping"]
        phi_by = {x["w"]: x["phi"] for x in rows}
        for j, w in enumerate(idx):
            E[w] = Ec[j]
            rep.setdefault("_well", {})[int(w)] = {
                "sd_fit_rel_m": float(sd_rel[j]), "n_rel": int(n_rel[j]),
                "phi": float(phi_by.get(j, float("nan"))),
                "drift_included": bool(rule["include_drift"])}
    wl = rep.pop("_well", {})
    off = rec["offsets"][sel]
    sd_abs = rec["sd"][sel]
    cells = np.asarray(a["cells"])[sel]
    rows = []
    for w in range(W):
        x = wl.get(w, {})
        rows.append({"station_id": ids_u[w], "county": county[w], "layer": int(layer[w]),
                     "layer_group": lgroup[w], "zone": zone[w], "cell": int(cells[w]),
                     "n_fit": int(np.isfinite(r[w]).sum()), "datum_m": float(off[w]),
                     "sd_abs_m": float(sd_abs[w]), "partition_layer": part_layer[w],
                     "partition_zone": part_zone[w], "grouping": grouping[w],
                     "group": groups[w], "subgroup": subgroups[w],
                     "sd_fit_rel_m": x.get("sd_fit_rel_m", float("nan")),
                     "n_rel": x.get("n_rel", 0), "t_centre_year": float(tcent[w]),
                     "phi": x.get("phi", float("nan")),
                     "drift_included": x.get("drift_included", False)})
    wells = pd.DataFrame(rows, columns=list(WELL_COLS)).sort_values("station_id")
    rep["unit_wells"] = int(W)
    rep["wells_without_sd_fit_rel"] = sorted(wells.loc[~np.isfinite(wells["sd_fit_rel_m"]),
                                                       "station_id"].tolist())
    return io.jsonable(rep), wells.reset_index(drop=True)


# ----------------------------------------------------------------------------- synthetic
def synthetic_arrays(seed: int = 7, n_y: int = 84, n_c: int = 74, drift: float = 0.0,
                     layer_shock: float = 0.0) -> tuple[dict, pd.DataFrame]:
    """A synthetic stand-in for E02's arrays and its predictions.csv (tests and rehearsal):
    132 fitted + 120 forecast months, county-month common shocks, optional layer-group
    shocks and per-well linear drift (m per year)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=252, freq="MS")
    fit = 132
    ids = [f"09{i:06d}" for i in range(n_y)] + [f"07{i:06d}" for i in range(n_c)]
    W, T = len(ids), len(dates)
    layers = rng.integers(1, 5, size=W)
    zones = rng.integers(0, 3, size=W)
    pred = 10 + rng.normal(0, 3, size=(W, 1)) + np.sin(np.arange(T) / 6.0)[None, :]
    cm = rng.normal(0, 1.0, size=(2, T))
    lg = rng.normal(0, layer_shock, size=(3, T))
    yr = L.year_frac(dates)
    beta = rng.normal(0, drift, size=(W, 1))
    noise = np.zeros((W, T))
    phi = 0.6
    eps = rng.normal(0, 0.3, size=(W, T))
    noise[:, 0] = eps[:, 0]
    for t in range(1, T):
        noise[:, t] = phi * noise[:, t - 1] + math.sqrt(1 - phi ** 2) * eps[:, t]
    ci = np.array([0 if s.startswith("09") else 1 for s in ids])
    li = np.array([{1: 0, 2: 1, 3: 2, 4: 2}[int(x)] for x in layers])
    obs = (pred + rng.normal(0, 0.8, size=(W, 1)) + cm[ci] + lg[li] + beta * (yr - 2017.5)
           + noise)
    hist = np.full((W, T), np.nan)
    hist[:, :fit] = obs[:, :fit]
    hist[:, :fit][rng.random((W, fit)) < 0.1] = np.nan
    a = {"ids": ids, "dates": dates, "origin": fit - 1, "fit": fit, "layers": layers,
         "cells": np.arange(W), "zones": zones, "historical": hist, "predicted": pred,
         "sigma_ens": np.full((W, T), 0.2), "model_sha256": "synthetic"}
    rec = e02_recipe(a)
    rows = []
    for w in range(W):
        for t in range(fit + 46, fit + 58):
            rows.append({"station_id": ids[w], "layer": int(layers[w]),
                         "date": str(dates[t].date()),
                         "prediction_origin": str(dates[fit - 1].date()),
                         "predicted_m": rec["model"][w, t],
                         "baseline_m": rec["model"][w, t] - 0.1,
                         "lower_m": rec["model"][w, t] - L.Z90 * rec["sigma"][w, t],
                         "upper_m": rec["model"][w, t] + L.Z90 * rec["sigma"][w, t],
                         "model_sha256": "synthetic"})
    return a, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def compute(a: dict, pred: pd.DataFrame,
            lead_definition: str = "datum_centre") -> tuple[dict, pd.DataFrame | None]:
    rec = e02_recipe(a)
    gate = datum_gate(a, rec, pred)
    out = {"id": L.DR_ID, "version": VERSION, "datum_gate": gate,
           "lead_definition": lead_definition}
    if not gate["ok"]:
        out["engineering_failure"] = True
        return out, None
    ids = np.array([str(x) for x in a["ids"]])
    units = sorted(pred["station_id"].astype(str).unique())
    zs = np.asarray(a["zones"])[np.isin(ids, units)]
    bad = sorted({int(z) for z in zs} - set(L.ZONE_IDS))
    if bad:
        out.update({"engineering_failure": True,
                    "zone_check": {"ok": False, "unexpected_zone_ids": bad,
                                   "why": f"column zone ids outside {list(L.ZONE_IDS)} (a "
                                          "county-split meta?): no zone partition is defined"}})
        return out, None
    out["zone_check"] = {"ok": True}
    rep, wells = analyse(a, rec, units, lead_definition)
    out.update(rep)
    out["engineering_failure"] = False
    out["decisions"] = {c: {"grouping": v["grouping"]["grouping"],
                            "include_drift": v["rule"]["include_drift"],
                            "tau2_per_year2": v["drift"].get("tau2_per_year2")}
                        for c, v in rep["counties"].items() if v.get("evaluable")}
    return out, wells


def write_outputs(out_dir: Path, report: dict, wells: pd.DataFrame | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shas = {}
    if wells is not None:
        p = out_dir / "wells_e02d.csv"
        if p.exists():
            raise FileExistsError(f"{p} exists (written once)")
        wells.to_csv(p, index=False)
        shas["wells_e02d.csv"] = io.sha256(p)
        report["wells_e02d_sha256"] = shas["wells_e02d.csv"]
    shas["report.json"] = io.write_once(out_dir / "report.json", report)
    return shas


def run_real(prereg_dir=None, rows_dir=None, out_dir: Path | None = None) -> int:
    from harness import prereg3

    doc, dsha = io.registered(L.DR_ID, prereg_dir, FROZEN)
    out_dir = out_dir or io.REPO / REPORT_DIR
    if (out_dir / "report.json").exists():
        raise io.Refused(f"{out_dir}/report.json exists: {L.DR_ID} ran; no re-read")
    lead_def = doc.get("lead_definition")
    if lead_def not in L.LEAD_DEFINITIONS:
        raise io.Refused(f"the registered declaration has no valid lead_definition "
                         f"({lead_def!r})")
    fwd = io.REPO / L.FORWARD
    if io.sha256(fwd) != L.MODEL_SHA:
        raise io.EngineeringFailure("forward npz sha256 differs from cf2c7154")
    pred = pd.read_csv(io.REPO / L.E02_DIR / "predictions.csv", dtype={"station_id": str})
    if io.sha256(io.REPO / L.E02_DIR / "predictions.csv") != L.E02_PRED_SHA:
        raise io.EngineeringFailure("E02 predictions.csv sha256 changed")
    kw = {"prereg_dir": prereg_dir} if prereg_dir else {}
    if rows_dir:
        kw["rows_dir"] = rows_dir
    lk = prereg3.record_look([L.DR_UNIT], L.DR_SCOPE, L.DR_WP, "design_time_estimation",
                             prereg_id=L.DR_ID,
                             note=f"{L.DR_ID} run (dr_drift.py {VERSION}): E02's pre-origin "
                                  "fitted-month residuals at the 158 wells, read once under "
                                  f"declaration sha256 {dsha}", **kw)
    a = load_e02_arrays(fwd, io.REPO / L.STATIONS, io.REPO / L.WELLS_DIR)
    report, wells = compute(a, pred, lead_def)
    report.update({"declaration": {"id": L.DR_ID, "sha256": dsha,
                                   "created": doc.get("created")},
                   "look": {"time": lk["time"], "purpose": lk["purpose"]},
                   "inputs": {"forward": io.rec(fwd),
                              "e02_predictions": io.rec(io.REPO / L.E02_DIR /
                                                        "predictions.csv")}})
    shas = write_outputs(out_dir, report, wells)
    print(json.dumps({"written": str(out_dir), **shas}, indent=1))
    if report["engineering_failure"]:
        print("ENGINEERING FAILURE (datum gate or zone ids): no statistic was computed; fix "
              "the cause, archive report.json and re-run (the look is logged again)")
        return 3
    for c, d in report["decisions"].items():
        print(f"{c}: grouping {d['grouping']}, drift term {d['include_drift']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="the registered read (the lead, once)")
    r = sub.add_parser("rehearsal", help="synthetic arrays only")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        try:
            return run_real()
        except io.Refused as exc:
            print(f"REFUSED: {exc}")
            return 1
    out = a.out.resolve()
    res = (io.REPO / "results").resolve()
    if out == res or res in out.parents:
        print("REFUSED: a rehearsal never writes under results/")
        return 1
    arr, pred = synthetic_arrays(a.seed)
    report, wells = compute(arr, pred)
    report["rehearsal"] = "SYNTHETIC: no observation was read"
    print(json.dumps(write_outputs(out, report, wells), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
