"""The ONE logged design-time read of E02x (the lead only, after U1.ext_transfer_power.v1 is
registered and the prereg3 blocking probe on E02x's window unit has cleared): the in-sample
transferred spread ``sd_fit_rel`` and the datum centre at N3F's 66 external wells.

    r_i[t] = observed_i - (forward_i + datum_i)         external well i, t in 2012-01..2022-12
    r_v[t] = observed_v - (forward_v + datum_v)         the 158 E02 wells (DR's own rows)
    e_i[t] = r_i[t] - median{ r_v[t] : v in i's donors, v reports in t }   (>= 5 donors)
    sd_fit_rel_i = sqrt(mean_t e_i[t]^2)  over >= 12 months, else NaN (well not scored)
    t_centre_i = mean decimal year of the months where r_i is observed

Donors = the E02 wells of i's DR group (zone partition, DR's merges; wells_e02x.ext_table);
Nantou (report-only) = the E02 wells of the same raw column zone in either county.

EXACTLY what it reads (nothing after 2022-12; no window head exists or is opened):
  1. results/twin_forward/datum_gate_innov.npz (model, sha cf2c7154): dates, origin, grid,
     mask, gate meta, baseline heads_mean / heads_std at the 158 + 66 cells.
  2. the 158 E02 wells' pre-origin monthly heads by DR's own path
     (dr_drift.load_e02_arrays: AMP_V2/data/fan_stations.parquet and AMP_V2/data/wells/
     *.parquet through twin.heads.build_head_field, t1 = 2023-01-01); engineering gate:
     DR's rebuilt recipe reproduces E02's registered predictions (dr_drift.datum_gate,
     1e-6 m) and DR's frozen wells_e02d.csv datum_m (1e-9 m).
  3. the 66 external wells' pre-origin monthly heads by N3F's own path
     (export_baselines._open_l1('datum_fit'): results/dataprep/heads/L1_clean/v3
     heads_monthly.parquet columns sid, date, value, flag and station_meta_preorigin.parquet
     columns sid, x_twd97, y_twd97, layer_code, CountyName, through common.guard with
     window_end 2022-12); engineering gate: N3F's recipe (export_baselines.ext_predictions)
     reproduces N3F's registered wells.csv datum_m and residual_sd_m (1e-9 m) and kept set.
  4. N3F's registered export results/twin/prospective_2026-11_ext/wells.csv (sha-checked),
     DR's frozen results/dataprep/reports/DR.relative_drift.v1/wells_e02d.csv (group, zone,
     datum_m) and the registered U1.decomp_power.v1 and N3F declarations (registry).
Refuses unless heads/L1_clean v3's MANIFEST allows datum_fit and design_time_evidence to
2022-12 (DR's governance check, metadata only). Every check that needs no head value runs
first (pre_look_checks). Logged next (prereg3.record_look, purpose design_time_estimation,
units PRE_UNIT and DR's unit, scope heads:2012-01..2022-12, prereg_id U1.ext_transfer_power.v1), immediately before
any of 2-3 loads; the two reproduction gates complete before any sd_fit_rel is computed.
Writes: results/dataprep/reports/E02x.sd_fit_rel/wells_e02x.csv (temporary file, then rename;
a stale copy without report.json is archived before the look, so a retry is safe) and
report.json (once). No ICC, no coverage and no drift statistic is computed here.

    python -I dataprep_staging/e02x/sdfit_e02x.py plan        (prints the read list; reads nothing)
    python -I dataprep_staging/e02x/sdfit_e02x.py run --window primary|fallback   (the lead, once)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d"), str(HERE.parent / "e02c")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402

VERSION = "1.0.0"
OUT_DIR = "results/dataprep/reports/E02x.sd_fit_rel"
FROZEN = (f"{io.X}/sdfit_e02x.py", f"{io.X}/e02xlib.py", f"{io.X}/e02xio.py",
          f"{io.X}/wells_e02x.py")
WELL_COLS = ("station_id", "county", "role", "layer", "layer_group", "zone", "cell", "group",
             "subgroup", "donor_key", "n_fit", "n_rel", "sd_fit_rel_m", "t_centre_year")
READS = (
    "results/twin_forward/datum_gate_innov.npz (model, cf2c7154)",
    "AMP_V2/data/fan_stations.parquet + AMP_V2/data/wells/*.parquet via "
    "twin.heads.build_head_field, t1 = 2023-01-01 (158 E02 wells, pre-origin; DR's path)",
    "results/dataprep/heads/L1_clean/v3 heads_monthly.parquet [sid, date, value, flag] and "
    "station_meta_preorigin.parquet [sid, x_twd97, y_twd97, layer_code, CountyName] via "
    "common.guard use datum_fit window_end 2022-12 (66 external wells; N3F's path)",
    "results/twin/prospective_2026-11_ext/wells.csv (N3F registered export; gate only)",
    "results/dataprep/reports/DR.relative_drift.v1/wells_e02d.csv (DR frozen table)",
    "results/twin/prospective_2026-11/predictions.csv (E02 registered; DR's datum gate)",
    "results/dataprep/preregistration: U1.decomp_power.v1, N3F, U1.ext_transfer_power.v1, "
    "INDEX.jsonl, looks.jsonl (registry)")
PROBE_DIR = "results/twin_runs/e02x"


def compute(r_ext: np.ndarray, r_donor: np.ndarray, ext: pd.DataFrame, donors: pd.DataFrame,
            dates) -> pd.DataFrame:
    """sd_fit_rel and t_centre per external well (pure). ``r_ext`` (66, Tfit), ``r_donor``
    (158, Tfit) pre-origin residuals in the row order of ``ext`` / ``donors``."""
    masks = L.donor_masks(ext, donors)
    e = L.transfer_relative(r_ext, r_donor, masks, ext["donor_key"].tolist())
    n_rel = np.isfinite(e).sum(axis=1)
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.nansum(np.where(np.isfinite(e), e ** 2, 0.0), axis=1)
                     / np.maximum(n_rel, 1))
    sd = np.where(n_rel >= L.MIN_FIT_MONTHS_SD, sd, np.nan)
    yf = L.D.year_frac(pd.DatetimeIndex(dates))
    tc = np.array([yf[np.isfinite(r_ext[w])].mean() if np.isfinite(r_ext[w]).any() else np.nan
                   for w in range(len(ext))])
    out = ext.copy()
    out["n_fit"] = np.isfinite(r_ext).sum(axis=1)
    out["n_rel"] = n_rel
    out["sd_fit_rel_m"] = sd
    out["t_centre_year"] = tc
    return out[list(WELL_COLS)].sort_values("station_id").reset_index(drop=True)


def governance() -> dict:
    """The source's governance (MANIFEST metadata only): heads/L1_clean v3 must allow
    datum_fit and design_time_evidence for rows up to 2022-12 (DR's check, reused)."""
    import prereg_design as PD

    return PD._governance()


def probe_ok(window: str) -> tuple[bool, str]:
    p = io.repo_path(f"{PROBE_DIR}/probe_{window}.json")
    if not p.is_file():
        return False, f"{p} missing: run 'prereg_e02x.py probe --window {window}' first"
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("blocking") or r.get("unit") != L.unit_id(window):
        return False, f"the probe on {L.unit_id(window)} did not clear: {r.get('blocking')}"
    return True, f"probe cleared at {r.get('time')}"


def _donor_residuals(donors: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    """DR's own read and gates (the 158 E02 wells; pre-origin)."""
    import dr_drift as DRM

    a = DRM.load_e02_arrays(io.repo_path(L.FORWARD), io.repo_path(L.D.STATIONS),
                            io.repo_path(L.D.WELLS_DIR))
    rec = DRM.e02_recipe(a)
    pred = pd.read_csv(io.repo_path(L.E02_DIR) / "predictions.csv", dtype={"station_id": str})
    gate = DRM.datum_gate(a, rec, pred)
    if not gate["ok"]:
        raise io.EngineeringFailure(f"DR's datum gate failed: {gate.get('why')}")
    ids = [str(s) for s in a["ids"]]
    pos = [ids.index(s) for s in donors["station_id"]]
    fit = a["fit"]
    r = a["historical"][pos, :fit] - rec["model"][pos, :fit]
    dd = np.max(np.abs(rec["offsets"][pos] - donors["datum_m"].to_numpy(float)))
    if not dd <= 1e-9:
        raise io.EngineeringFailure(f"donor datum differs from wells_e02d.csv by {dd} m")
    return r, a["dates"][:fit], {"datum_gate": gate, "max_abs_datum_vs_wells_e02d_m": float(dd)}


def _ext_residuals(ext: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """N3F's own read and gate (the 66 external wells; pre-origin, L1_clean v3 datum_fit)."""
    import export_baselines as EB
    import prereg_e02c_ext as PX

    ids = sorted(ext["station_id"])
    fw = EB._load_forward()
    long_d, meta, notes = EB._open_l1("datum_fit")
    ax = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    obs = EB.obs_matrix(long_d, EB.ext_ids(), ax)
    wells = EB.well_table(meta, EB.ext_ids(), fw)
    spec = PX.WINDOWS["2026-11"]
    _, _, sel = EB.ext_predictions(wells, obs, fw, spec["first"], spec["last"])
    reg = pd.read_csv(io.repo_path(L.N3F_DIR) / "wells.csv", dtype={"station_id": str})
    m = sel.merge(reg, on="station_id", suffixes=("", "_reg"))
    if sorted(m.loc[m["kept"].astype(bool), "station_id"]) != ids:
        raise io.EngineeringFailure("N3F's keep rule does not give the 66 kept wells")
    k = m[m["kept"].astype(bool)]
    gate = {c: float(np.nanmax(np.abs(k[c].to_numpy(float) - k[f"{c}_reg"].to_numpy(float))))
            for c in ("datum_m", "residual_sd_m")}
    if not all(v <= 1e-9 for v in gate.values()):
        raise io.EngineeringFailure(f"N3F's export is not reproduced: {gate}")
    t_fit = int(fw["origin"]) + 1
    sel = sel.set_index("station_id")
    allids = EB.ext_ids()
    r = np.full((len(ids), t_fit), np.nan)
    sc = list(map(str, fw["scenario_names"])).index("baseline")
    for j, s in enumerate(ids):
        w = allids.index(s)
        lay, cell = int(sel.at[s, "layer"]), int(sel.at[s, "cell"])
        model = np.asarray(fw["heads_mean"][sc, lay, cell], float) + float(sel.at[s, "datum_m"])
        r[j] = obs[w, :t_fit] - model[:t_fit]
    return r, {"n3f_reproduced_max_abs_m": gate, "guard_notes": notes}


def pre_look_checks(window: str, prereg_dir=None) -> dict:
    """Every check that needs no head value, run BEFORE the look is logged: registration and
    frozen code, no earlier report, cleared probe, governance (MANIFEST), the sha256 of every
    registered input (forward, E02 predictions, N3F wells.csv, DR wells_e02d.csv via the U1
    design), the external and donor tables (metadata). Returns the prepared inputs."""
    import wells_e02x as WX

    doc, dsha = io.registered(L.POWER_ID, prereg_dir, FROZEN)
    out = io.repo_path(OUT_DIR)
    if (out / "report.json").exists():
        raise io.Refused(f"{out}/report.json exists: the read happened; no re-read")
    ok, why = probe_ok(window)
    if not ok:
        raise io.Refused(why)
    gov = governance()
    if not gov.get("ok"):
        raise io.Refused(f"heads/L1_clean v3 does not allow datum_fit and design_time_evidence "
                         f"to 2022-12: {gov.get('why')}")
    if io.sha256(io.repo_path(L.FORWARD)) != L.MODEL_SHA:
        raise io.EngineeringFailure("forward npz sha256 differs from cf2c7154")
    if io.sha256(io.repo_path(L.E02_DIR) / "predictions.csv") != L.D.E02_PRED_SHA:
        raise io.EngineeringFailure("E02 predictions.csv changed (DR's datum gate input)")
    ext, prov = WX.ext_table(prereg_dir)          # N3F wells.csv sha-checked
    design, _ = WX.u1_design(prereg_dir)
    donors = WX.donors_table(design, ("station_id", "county", "zone", "group", "datum_m"))
    return {"dsha": dsha, "out": out, "probe": why, "gov": gov, "ext": ext, "prov": prov,
            "donors": donors}


def _write_csv_retry_safe(df: pd.DataFrame, p: Path) -> str:
    """Write to a temporary file in the same directory, then rename into place (a crash leaves
    no half-written wells_e02x.csv); returns the sha256."""
    tmp = p.with_name(p.name + ".partial")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    os.replace(tmp, p)
    return io.sha256(p)


def run_real(window: str, prereg_dir=None, rows_dir=None) -> int:
    """The read. A wells_e02x.csv left by an interrupted run (no report.json) is moved aside,
    logged, before the look, so a retry is safe; the two reproduction gates (DR's datum gate
    and N3F's export) read heads and therefore run after the look, but both complete before any
    sd_fit_rel is computed; a gate failure writes nothing and the retry logs a new look."""
    from harness import prereg3

    pre = pre_look_checks(window, prereg_dir)
    out = pre["out"]
    stale = out / "wells_e02x.csv"
    moved = None
    if stale.exists():
        arch = out / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        moved = arch / f"wells_e02x.{io.now().strftime('%Y%m%dT%H%M%SZ')}.csv"
        os.replace(stale, moved)
        io.append_attempt(io.repo_path(f"{PROBE_DIR}/attempts.jsonl"),
                          {"event": "sdfit", "outcome": "stale_csv_archived",
                           "moved": io.rel(moved)})
    kw = {"prereg_dir": prereg_dir} if prereg_dir else {}
    if rows_dir:
        kw["rows_dir"] = rows_dir
    lk = prereg3.record_look([L.PRE_UNIT, L.DR_UNIT], L.PRE_SCOPE, L.POWER_WP,
                             "design_time_estimation", prereg_id=L.POWER_ID,
                             note=f"E02x sd_fit_rel read (sdfit_e02x.py {VERSION}) under "
                                  f"declaration sha256 {pre['dsha']}: pre-origin heads at the "
                                  "66 external wells (L1_clean v3, datum_fit) and DR's 158-well "
                                  "fitted residuals (E02's own read); nothing after 2022-12",
                             **kw)
    try:
        r_d, fdates, gd = _donor_residuals(pre["donors"])      # gate: DR's datum gate
        r_e, ge = _ext_residuals(pre["ext"])                    # gate: N3F's export
    except io.Refused as exc:
        io.append_attempt(io.repo_path(f"{PROBE_DIR}/attempts.jsonl"),
                          {"event": "sdfit", "outcome": "engineering_failure",
                           "look_time": lk["time"], "why": str(exc)})
        raise
    ext_sorted = pre["ext"].sort_values("station_id").reset_index(drop=True)
    wells = compute(r_e, r_d, ext_sorted, pre["donors"], fdates)
    out.mkdir(parents=True, exist_ok=True)
    wsha = _write_csv_retry_safe(wells, out / "wells_e02x.csv")
    rep = {"version": VERSION, "declaration": {"id": L.POWER_ID, "sha256": pre["dsha"]},
           "look": {"time": lk["time"], "purpose": lk["purpose"]}, "window": window,
           "probe": pre["probe"], "governance": pre["gov"], "gates": {"donors": gd, "ext": ge},
           "ext_provenance": pre["prov"], "wells_e02x_sha256": wsha,
           "stale_csv_archived": io.rel(moved) if moved else None,
           "wells_without_sd_fit_rel": sorted(wells.loc[~np.isfinite(wells["sd_fit_rel_m"]),
                                                        "station_id"].tolist()),
           "reads": list(READS)}
    sha = io.write_once(out / "report.json", rep)
    io.append_attempt(io.repo_path(f"{PROBE_DIR}/attempts.jsonl"),
                      {"event": "sdfit", "outcome": "written", "report_sha256": sha})
    print(json.dumps({"report_sha256": sha, "wells_e02x_sha256": wsha}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="print what 'run' would read; reads nothing")
    r = sub.add_parser("run", help="the logged design-time read (the lead, once)")
    r.add_argument("--window", choices=sorted(L.WINDOWS), required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps({"reads": list(READS), "look": {
            "units": [L.PRE_UNIT, L.DR_UNIT], "scope": L.PRE_SCOPE,
            "purpose": "design_time_estimation", "prereg_id": L.POWER_ID},
            "writes": [f"{OUT_DIR}/wells_e02x.csv", f"{OUT_DIR}/report.json"]}, indent=1))
        return 0
    try:
        return run_real(a.window)
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
