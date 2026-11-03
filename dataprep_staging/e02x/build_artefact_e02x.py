"""Freeze E02x's transferred relative band artefact (sigma_idio at the external wells) for the
primary window (2026-12..2027-10) and the declared fallback (2027-01..2027-10). Lead-run, CPU,
after the member gate (members_e02x.py), the logged sd_fit_rel read (sdfit_e02x.py) and the
power result.

    sigma_idio[i, t] = sqrt(sd_fit_rel[i]^2 + sigma_ens_rel[i, t]^2 + tau_c^2 dt[i, t]^2)

* sd_fit_rel: the logged design-time read (results/dataprep/reports/E02x.sd_fit_rel/
  wells_e02x.csv); a well without it is not scored (E02d's rule).
* sigma_ens_rel: the relative member spread of the 36 members at the external cells, around
  the median of the same members at the well's donors (e02xlib.sigma_ens_rel, min donors 5);
  heads_std is not used.
* tau_c^2: DR.relative_drift.v1's report (sha 32025941), the county's tau^2 where DR's 10 %
  rule triggered (E02d's rule; Nantou report-only: the larger triggered value); dt from DR's
  registered lead definition (datum centre: the well's t_centre_year from the read; origin
  (2023.0)); a report without a valid lead_definition is an engineering failure.
* centre: N3F's registered predicted_m (the frozen forecast plus the well's own pre-origin
  datum), rows of the window months; at scoring the transferred departure of the 158 E02
  wells in the same month is added (it does not exist before scoring).

Reads model output, registered files and pre-origin derived tables; no observation. Writes
once: results/twin/e02x_relative/{sigma_idio_primary.csv, sigma_idio_fallback.csv,
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

import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402
import members_e02x as ME  # noqa: E402
import sdfit_e02x as SF  # noqa: E402

VERSION = "1.0.0"
OUT = "results/twin/e02x_relative"
ART_COLS = ("station_id", "county", "role", "zone", "group", "donor_key", "date",
            "lead_months", "predicted_m", "sd_fit_rel_m", "sigma_ens_rel_m", "drift_m",
            "sigma_idio_m", "half_width_m")


def build(wells: pd.DataFrame, donors: pd.DataFrame, members: dict, pred: pd.DataFrame,
          dr: dict) -> dict:
    """{window: artefact frame} (pure given the inputs). ``wells``: wells_e02x.csv;
    ``donors``: station_id, group, zone in members['donor_station_id'] order."""
    w = wells[np.isfinite(wells["sd_fit_rel_m"].to_numpy(float))].sort_values("station_id")
    w = w.reset_index(drop=True)
    ids = w["station_id"].astype(str).tolist()
    eid = [str(s) for s in members["ext_station_id"]]
    did = [str(s) for s in members["donor_station_id"]]
    if donors["station_id"].astype(str).tolist() != did:
        raise io.EngineeringFailure("donor table order differs from the member table")
    Me = np.asarray(members["members_ext"], float)[:, [eid.index(s) for s in ids], :]
    Md = np.asarray(members["members_donor"], float)
    mdates = pd.DatetimeIndex(pd.to_datetime(members["dates"])).to_period("M").to_timestamp()
    masks = L.donor_masks(w, donors)
    keys = w["donor_key"].tolist()
    tau2 = np.array([L.drift_tau2(dr["decisions"], c) for c in w["county"]])
    lead_def = dr.get("lead_definition")
    if lead_def not in L.D.LEAD_DEFINITIONS:
        raise io.EngineeringFailure(f"DR's report has no valid lead_definition ({lead_def!r})")
    anchor = L.D.lead_anchor(lead_def, w["t_centre_year"].to_numpy(float))
    out = {}
    for win in L.WINDOWS:
        ms = L.months(win)
        ti = [int(np.flatnonzero(mdates == m)[0]) for m in ms]
        ens = L.sigma_ens_rel(Me[:, :, ti], Md[:, :, ti], masks, keys)
        dt_ = L.D.year_frac(ms)[None, :] - anchor[:, None]
        drift = np.sqrt(tau2[:, None] * dt_ ** 2)
        sig = L.sigma_idio(w["sd_fit_rel_m"].to_numpy(float), ens, tau2, dt_)
        centre = L.frame_to_arrays(pred, ids, ms, ("predicted_m",))["predicted_m"]
        lead = L.lead_months(ms)
        rows = []
        for i, s in enumerate(ids):
            for j, m in enumerate(ms):
                if not (np.isfinite(centre[i, j]) and np.isfinite(sig[i, j])):
                    continue
                rows.append({"station_id": s, "county": w.at[i, "county"],
                             "role": w.at[i, "role"], "zone": w.at[i, "zone"],
                             "group": w.at[i, "group"], "donor_key": w.at[i, "donor_key"],
                             "date": str(m.date()), "lead_months": int(lead[j]),
                             "predicted_m": float(centre[i, j]),
                             "sd_fit_rel_m": float(w.at[i, "sd_fit_rel_m"]),
                             "sigma_ens_rel_m": float(ens[i, j]), "drift_m": float(drift[i, j]),
                             "sigma_idio_m": float(sig[i, j]),
                             "half_width_m": float(L.Z90 * sig[i, j])})
        out[win] = pd.DataFrame(rows, columns=list(ART_COLS))
    return out


def run(out_dir: Path, prereg_dir=None) -> int:
    import wells_e02x as WX

    dr_dir = io.repo_path(L.DR_DIR)
    if not io.sha256(dr_dir / "report.json").startswith(L.DR_REPORT_SHA):
        raise io.EngineeringFailure(f"DR's report.json is not {L.DR_REPORT_SHA}")
    dr = json.loads((dr_dir / "report.json").read_text(encoding="utf-8"))
    if dr.get("engineering_failure") or "decisions" not in dr:
        raise io.Refused("DR.relative_drift.v1 has no valid report")
    gate = json.loads(io.repo_path(ME.GATE_JSON).read_text(encoding="utf-8"))
    mp = io.repo_path(ME.MEMBERS_NPZ)
    if not gate.get("ok") or io.sha256(mp) != gate.get("members_ext_sha256"):
        raise io.Refused("the member gate did not pass (or members_ext.npz changed)")
    sd_dir = io.repo_path(SF.OUT_DIR)
    sdr = json.loads((sd_dir / "report.json").read_text(encoding="utf-8"))
    if io.sha256(sd_dir / "wells_e02x.csv") != sdr.get("wells_e02x_sha256"):
        raise io.EngineeringFailure("wells_e02x.csv differs from the read's report")
    wells = pd.read_csv(sd_dir / "wells_e02x.csv", dtype={"station_id": str})
    n3f, _ = WX.n3f_registration(prereg_dir)
    pp = io.repo_path(L.N3F_DIR) / "predictions.csv"
    want = n3f["external_predictions"]["files"]["predictions.csv"]["sha256"]
    if io.sha256(pp) != want:
        raise io.EngineeringFailure("N3F predictions.csv is not the registered file")
    pred = pd.read_csv(pp, dtype={"station_id": str})
    with np.load(mp, allow_pickle=False) as z:
        members = {k: z[k] for k in z.files}
    design, _ = WX.u1_design(prereg_dir)
    dt_ = WX.donors_table(design, ("station_id", "group", "zone"))
    donors = dt_.set_index("station_id").loc[[str(s) for s in members["donor_station_id"]]]
    donors = donors.reset_index()
    arts = build(wells, donors, members, pred, dr)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise io.Refused(f"{out_dir} is not empty (the artefact is written once)")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for win, f in arts.items():
        p = out_dir / f"sigma_idio_{win}.csv"
        with open(p, "x", encoding="utf-8", newline="") as fh:
            f.to_csv(fh, index=False)
        files[p.name] = io.sha256(p)
    summ = {"version": VERSION, "files": files,
            "inputs": {"dr_report": io.rec(dr_dir / "report.json"),
                       "members_ext": io.rec(mp), "members_gate": io.rec(ME.GATE_JSON),
                       "sd_fit_rel_report": io.rec(sd_dir / "report.json"),
                       "wells_e02x": io.rec(sd_dir / "wells_e02x.csv"),
                       "n3f_predictions": io.rec(pp)},
            "decisions": dr["decisions"],
            "lead_definition": dr["lead_definition"],
            "per_window": {w: {"rows": int(len(f)), "wells": int(f["station_id"].nunique()),
                               "by_county": {c: int(f.loc[f["county"] == c, "station_id"]
                                                    .nunique())
                                             for c in (*L.GATED, *L.REPORT_ONLY)}}
                           for w, f in arts.items()},
            "z": L.Z90, "min_donors": L.MIN_DONORS}
    sha = io.write_once(out_dir / "summary.json", summ)
    print(json.dumps({"summary_sha256": sha, **files}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        return run(a.out or io.repo_path(OUT))
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
