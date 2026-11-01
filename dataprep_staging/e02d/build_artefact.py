"""Freeze E02d's relative interval artefact (sigma_idio) for windows A, B and C (lead-run, CPU,
after DR.relative_drift.v1, the fold-member rerun and the absolute registrations B/C).

    sigma_idio[w, t] = sqrt(sd_fit_rel[w]^2 + sigma_ens_rel[w, t]^2 [+ tau_c^2 dt[w, t]^2])

* sd_fit_rel: DR's per-well root mean square of the relative fitted-month residual (LOO
  median of the DR-chosen group), from wells_e02d.csv. Nothing after 2022-12.
* sigma_ens_rel: the relative member spread of the 36 rerun members (e02dlib.sigma_ens_rel,
  same LOO operator and groups) in each window month. heads_std is NOT used (it carries the
  spread shared by every well).
* the drift term only for a county whose DR 10 % rule triggered: tau_c^2 (DR, a lower
  bound) times (window month - anchor)^2 in years; the anchor is DR's registered
  ``lead_definition``: the well's datum centre or the forecast origin (2023.0).
* centre: E02's registered predicted_m (A: results/twin/prospective_2026-11, rows
  2026-12..2027-10, no correction) and the same recipe's predicted_m for B and C
  (``python -m hydrophysics.twin.prospective register`` with test_start/test_end, the
  absolute artefacts of rank 6). Engineering gate: the same register call for
  2026-12..2027-10 (register refuses a started window, so not 2026-11) reproduces E02's
  predictions.csv rows of those months (predicted/lower/upper/baseline) to 1e-9 m.

Reads model output, E02's frozen files and DR's frozen table; no observation. Writes once:
results/twin/e02d_relative/{sigma_idio_A.csv, sigma_idio_B.csv, sigma_idio_C.csv,
summary.json}.
"""
from __future__ import annotations

import argparse
import json
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
import members_e02d as ME  # noqa: E402

VERSION = "1.0.0"
OUT = "results/twin/e02d_relative"
DR_DIR = f"results/dataprep/reports/{L.DR_ID}"
NUM_COLS = ("predicted_m", "lower_m", "upper_m", "baseline_m")
ART_COLS = ("station_id", "county", "layer", "group", "date", "lead_months", "predicted_m",
            "sd_fit_rel_m", "sigma_ens_rel_m", "drift_m", "sigma_idio_m", "half_width_m")


def abs_check(check_dir: Path, e02_dir: Path) -> dict:
    """The register recipe reproduces E02's registered rows of the check's months."""
    a = pd.read_csv(check_dir / "predictions.csv", dtype={"station_id": str})
    b = pd.read_csv(e02_dir / "predictions.csv", dtype={"station_id": str})
    b = b[b["date"].isin(set(a["date"]))]
    key = ["station_id", "date"]
    m = a.merge(b, on=key, how="outer", suffixes=("", "_e02"), indicator=True)
    out = {"rows_check": int(len(a)), "rows_e02_same_months": int(len(b)),
           "months": sorted(set(a["date"])),
           "same_rows": bool(len(a) > 0 and (m["_merge"] == "both").all())}
    if out["same_rows"]:
        out["max_abs_diff_m"] = {c: float(np.max(np.abs(m[c] - m[f"{c}_e02"])))
                                 for c in NUM_COLS}
        out["ok"] = all(v <= L.ABS_REPRO_TOL_M for v in out["max_abs_diff_m"].values())
    else:
        out["ok"] = False
    return out


def window_centre(win: str, wells: list[str], abs_dirs: dict) -> tuple[np.ndarray, list]:
    months = L.month_index(L.WINDOWS[win]["first"], L.WINDOWS[win]["last"])
    p = pd.read_csv(io.repo_path(abs_dirs[win]) / "predictions.csv",
                    dtype={"station_id": str})
    arr = L.frame_to_arrays(p, wells, months, ("predicted_m",))["predicted_m"]
    return arr, months


def build(wells: pd.DataFrame, dr: dict, members: dict, abs_dirs: dict) -> dict:
    """{window: artefact frame} (pure given the inputs)."""
    w = wells.copy()
    w = w[np.isfinite(w["sd_fit_rel_m"].to_numpy(float))].reset_index(drop=True)
    ids = w["station_id"].astype(str).tolist()
    mid = [str(s) for s in members["station_id"]]
    pos = [mid.index(s) for s in ids]
    M = np.asarray(members["members"], float)[:, pos, :]
    mdates = pd.DatetimeIndex(pd.to_datetime(members["dates"])).to_period("M").to_timestamp()
    groups = w["group"].astype(str).to_numpy()
    dec = dr["decisions"]
    tau2 = np.array([float(dec[c]["tau2_per_year2"]) if dec.get(c, {}).get("include_drift")
                     else 0.0 for c in w["county"]])
    out = {}
    for win in ("A", "B", "C"):
        centre, months = window_centre(win, ids, abs_dirs)
        ti = [int(np.flatnonzero(mdates == m)[0]) for m in months]
        ens = L.sigma_ens_rel(M[:, :, ti], groups)
        anchor = L.lead_anchor(dr.get("lead_definition", "datum_centre"),
                               w["t_centre_year"].to_numpy(float))
        dt = L.year_frac(months)[None, :] - anchor[:, None]
        drift = np.sqrt(tau2[:, None] * dt ** 2)
        sig = L.sigma_idio(w["sd_fit_rel_m"].to_numpy(float), ens, tau2, dt)
        lead = ((months.year - 2022) * 12 + months.month - 12).to_numpy(int)
        rows = []
        for i, s in enumerate(ids):
            for j, m in enumerate(months):
                if not (np.isfinite(centre[i, j]) and np.isfinite(sig[i, j])):
                    continue
                rows.append({"station_id": s, "county": w.at[i, "county"],
                             "layer": int(w.at[i, "layer"]), "group": groups[i],
                             "date": str(m.date()), "lead_months": int(lead[j]),
                             "predicted_m": float(centre[i, j]),
                             "sd_fit_rel_m": float(w.at[i, "sd_fit_rel_m"]),
                             "sigma_ens_rel_m": float(ens[i, j]),
                             "drift_m": float(drift[i, j]),
                             "sigma_idio_m": float(sig[i, j]),
                             "half_width_m": float(L.Z90 * sig[i, j])})
        out[win] = pd.DataFrame(rows, columns=list(ART_COLS))
    return out


def run(out_dir: Path) -> int:
    dr_dir = io.REPO / DR_DIR
    dr = json.loads((dr_dir / "report.json").read_text(encoding="utf-8"))
    if dr.get("engineering_failure") or "decisions" not in dr:
        raise io.Refused("DR.relative_drift.v1 has no valid report")
    if io.sha256(dr_dir / "wells_e02d.csv") != dr.get("wells_e02d_sha256"):
        raise io.EngineeringFailure("wells_e02d.csv differs from DR's report")
    gate = json.loads((io.REPO / ME.GATE_JSON).read_text(encoding="utf-8"))
    mpath = io.REPO / ME.MEMBERS_NPZ
    if not gate.get("ok") or io.sha256(mpath) != gate.get("members_wells_sha256"):
        raise io.Refused("the fold-member rerun gate did not pass (or the npz changed)")
    chk = abs_check(io.REPO / L.ABS_CHECK_DIR, io.REPO / L.E02_DIR)
    if not chk["ok"]:
        raise io.EngineeringFailure(f"the register recipe does not reproduce E02: {chk}")
    for win in ("B", "C"):
        man = json.loads((io.REPO / L.ABS_DIRS[win] / "manifest.json").read_text())
        if man.get("model_sha256") != L.MODEL_SHA:
            raise io.EngineeringFailure(f"{L.ABS_DIRS[win]} is not bound to cf2c7154")
    wells = pd.read_csv(dr_dir / "wells_e02d.csv", dtype={"station_id": str})
    with np.load(mpath, allow_pickle=False) as z:
        members = {k: z[k] for k in z.files}
    arts = build(wells, dr, members, L.ABS_DIRS)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise io.Refused(f"{out_dir} is not empty (the artefact is written once)")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for win, f in arts.items():
        p = out_dir / f"sigma_idio_{win}.csv"
        f.to_csv(p, index=False)
        files[p.name] = io.sha256(p)
    summ = {"version": VERSION, "files": files, "abs_check": chk,
            "inputs": {"dr_report": io.rec(dr_dir / "report.json"),
                       "wells_e02d": io.rec(dr_dir / "wells_e02d.csv"),
                       "members_wells": io.rec(mpath),
                       "members_gate": io.rec(io.REPO / ME.GATE_JSON),
                       **{f"abs_{w}": io.rec(io.REPO / d / "predictions.csv")
                          for w, d in L.ABS_DIRS.items()}},
            "decisions": dr["decisions"],
            "lead_definition": dr.get("lead_definition", "datum_centre"),
            "per_window": {w: {"rows": int(len(f)), "wells": int(f["station_id"].nunique()),
                               "by_county": {c: int((f["county"] == c).sum())
                                             for c in L.COUNTIES}}
                           for w, f in arts.items()},
            "z": L.Z90, "min_others": L.MIN_OTHERS}
    sha = io.write_once(out_dir / "summary.json", summ)
    print(json.dumps({"summary_sha256": sha, **files}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=io.REPO / OUT)
    a = ap.parse_args(argv)
    try:
        return run(a.out)
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
