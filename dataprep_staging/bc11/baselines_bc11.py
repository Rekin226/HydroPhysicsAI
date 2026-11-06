"""The 2012-22 ingredients of I1.backcast_2010_11.v1, frozen (by sha256) in the declaration
BEFORE any 2010-11 head is compared with anything (the lead runs this after the backcast
panel and before the G0 studies).

From heads/L1_clean v3 (2012-01..2022-12 only: the guard's window_end is 2022-12 and the
product starts 2012-01, so no 2010-11 value exists in it), flags 0-1, use 'baseline', for
the availability look's qualifying wells:

* E04's recipe refitted on 2012-22 (``bc11_lib.fit_stat``): per-well calendar-month CL,
  lag-1 rho (pooled-median fallback), residual sd, fitted-month counts. E04's frozen rho
  (prospective_2026-11_hybrid/parameters.json) is never opened.
* the physics datum and residual sd of the page ensemble at each placed screen
  (``bc11_lib.physics_datum`` on the panel's ``native_fit``: the page forward's own 2012-22
  heads), for G2, G3, the CF arm and G2/G3's G0 noise scale.

CL is fitted on 2012-22 and applied BACKWARD to 2010-11 (time-reversed; disclosed). Writes
bc11_baselines.npz + .json once (mode 'x'); prints only counts and sha256s.

    python dataprep_staging/bc11/baselines_bc11.py --panel <bc11_panel.npz> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "ch_x"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
#: CPU entry point: this checkout's hydrophysics (the hash-bound copy), not an editable install
if str(HERE.parents[1]) not in sys.path[:1]:
    sys.path.insert(0, str(HERE.parents[1]))

import bc11_const as K  # noqa: E402
import bc11_lib as BL  # noqa: E402
import chx_io as IO  # noqa: E402


def eligible(avail_path: str | Path = K.AVAIL_OUT) -> tuple[list[str], dict, str]:
    """(qualifying wells, county of each, avail.json sha256) from the availability output
    (counts and month lists only); refuses unless its stop rule passed."""
    p = IO.rp(avail_path)
    a = json.loads(p.read_text())
    f = a["feasibility"]
    if not f["passed"]:
        raise SystemExit("REFUSED: the availability stop rule did not pass (not evaluable)")
    ids = sorted(map(str, f["eligible_wells"]))
    return ids, {s: f["per_well"][s]["county"] for s in ids}, IO.sha256(p)


def build(panel: Path, out: Path, loader=None) -> dict:
    target = out / Path(K.BASELINES).name
    if target.exists():
        raise SystemExit(f"{target} exists")
    ids, county, avail_sha = eligible()
    z = np.load(panel, allow_pickle=False)
    psid = [str(s) for s in z["sid"]]
    if not set(psid) <= set(ids):
        raise SystemExit("REFUSED: panel wells outside the qualifying wells")
    load = loader or (lambda sids, months: IO.l1_monthly(
        sids, months, product=K.CL_PRODUCT, use=K.CL_USE, window_end=K.CL_WINDOW_END))
    obs_fit, _, notes = load(ids, BL.FIT_M)
    fit = BL.fit_stat(obs_fit)
    pi = {s: i for i, s in enumerate(ids)}
    W = len(ids)
    datum = np.full(W, np.nan)
    rsd = np.full(W, np.nan)
    rows = [pi[s] for s in psid]
    d, sd = BL.physics_datum(np.asarray(z["native_fit"], float), obs_fit[rows])
    datum[rows], rsd[rows] = d, sd
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"sid": np.array(ids), "county": np.array([county[s] for s in ids]),
              "cl_month": fit["cl_month"], "rho": fit["rho"], "rho_used": fit["rho_used"],
              "own_rho": fit["own_rho"], "rho_pooled": np.array(fit["rho_pooled"]),
              "sd": fit["sd"], "n_fit": fit["n_fit"], "n_pairs": fit["n_pairs"],
              "phys_datum": datum, "phys_resid_sd": rsd}
    with target.open("xb") as fh:
        np.savez(fh, **arrays)
    summ = {"what": "I1.backcast_2010_11.v1 2012-22 baselines (CL, rho, sd; physics datum)",
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "cl_product": K.CL_PRODUCT, "cl_use": K.CL_USE,
            "cl_manifest_sha256": IO.sha_or_missing(IO.rp(K.CL_PRODUCT) / "MANIFEST.json"),
            "window_end": K.CL_WINDOW_END, "guard_notes": list(notes),
            "availability": K.AVAIL_OUT, "availability_sha256": avail_sha,
            "panel": str(panel), "panel_sha256": IO.sha256(panel),
            "n_wells": W, "n_wells_kept_fit": int((fit["n_fit"] >= K.MIN_FIT_MONTHS).sum()),
            "n_own_rho": int(fit["own_rho"].sum()),
            "n_phys": int(np.isfinite(datum).sum()),
            "baselines": str(target), "baselines_sha256": IO.sha256(target),
            "code_sha256": K.code_shas()}
    with (out / Path(K.BASELINES_JSON).name).open("x") as f:
        json.dump(summ, f, indent=1)
    print(f"baselines: {W} wells, {summ['n_wells_kept_fit']} with >= {K.MIN_FIT_MONTHS} fitted "
          f"months, {summ['n_own_rho']} own rho, {summ['n_phys']} physics datums; sha256 "
          f"{summ['baselines_sha256']}")
    return summ


def load(path: str | Path = K.BASELINES) -> dict:
    with np.load(IO.rp(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    build(a.panel, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
