"""Freeze E02z's pre-origin baselines CL and CLT on the year-2/3 window months (the lead runs
this BEFORE registration; nothing here reads a window head or any month after 2022-12).

E02c's export code is imported read-only (dataprep_staging/e02c/export_baselines.py, bound by
sha in E02c's registration, never edited): heads/L1_clean/v3 opened through ``common.guard``
with use ``baseline`` and ``window_end`` 2022-12 (flags 0-1), ``obs_matrix`` fills only the
132 fitted months 2012-01..2022-12, ``baselines_table`` evaluates
``twin.drift_diag.fair_baselines`` (clim, clim_trend; month_offset 0) on the 2027-11..2029-10
months, and E02c's declared ``reference_rule`` (pre-origin vs pre-origin) is applied to each
window's rows against E02y's registered baseline_m (y2 / y3 predictions.csv). The 158 ids are
E02y's registered predictions' wells (E02c's wells). The well table and its 5 km blocks are
E02c's frozen wells_158.csv (not rebuilt).

Writes, once (open mode 'x'): baselines_e02z.csv (both windows), summary.json (sha256s, the
reference decision per window), and with ``--verify`` a bit-identical rebuild record
verify.json (E02c's ``verify_export``). The draft registration refuses without verify.json
reporting ``identical``.

    python -I dataprep_staging/e02z/export_baselines_e02z.py \\
        --out results/dataprep/reports/E02z/baselines
    python -I dataprep_staging/e02z/export_baselines_e02z.py --verify --out <same dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02z_const as K  # noqa: E402
import e02z_lib as L  # noqa: E402
import export_baselines as EB  # noqa: E402  (E02c's, read-only)
import pandas as pd  # noqa: E402

LAST = K.WINDOWS["C"][1]


def registered_ids(pred_by_win: dict) -> list[str]:
    ids = None
    for w, p in pred_by_win.items():
        s = sorted(p["station_id"].astype(str).unique())
        if len(s) != K.N_WELLS:
            raise SystemExit(f"window {w}: {len(s)} wells, expected {K.N_WELLS}")
        if ids is not None and s != ids:
            raise SystemExit("the two windows' registered wells differ")
        ids = s
    return ids


def load_predictions() -> dict:
    out = {}
    for w, d in K.ABS_DIRS.items():
        p = L.rp(d) / "predictions.csv"
        if L.sha256(p) != K.ABS_SHA[w]["predictions.csv"]:
            raise SystemExit(f"{p} is not E02y's registered file")
        out[w] = pd.read_csv(p, dtype={"station_id": str})
    return out


def build(pred_by_win: dict, obs, ax) -> tuple[pd.DataFrame, dict]:
    """Pure part: CL/CLT on each window from the pre-origin matrix ``obs`` (W, len(ax)) and
    E02c's reference rule per window. Returns the frozen table and the per-window decisions."""
    ids = registered_ids(pred_by_win)
    tables, ref = [], {}
    for w, (first, last) in K.WINDOWS.items():
        base_l1 = EB.baselines_table(ids, obs, ax, first, last)
        t, info = EB.reference_rule(pred_by_win[w], base_l1)
        t.insert(1, "window", w)
        tables.append(t)
        ref[w] = info
    return pd.concat(tables, ignore_index=True), ref


def export(out: Path) -> dict:
    pred = load_predictions()
    ids = registered_ids(pred)
    long, _, notes = EB._open_l1("baseline")
    ax = EB.axis(LAST)
    obs = EB.obs_matrix(long, ids, ax)
    base, ref = build(pred, obs, ax)
    out.mkdir(parents=True, exist_ok=True)
    EB._write_csv(base, out / K.BASE_FILE)
    summ = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "what": "E02z pre-origin baselines CL/CLT on 2027-11..2029-10 (E02c's code)",
            "product": EB.L1, "use": "baseline", "window_end": "2022-12",
            "guard_notes": notes, "n_wells": len(ids), "n_rows": int(len(base)),
            "reference": ref,
            "files": {K.BASE_FILE: L.sha256(out / K.BASE_FILE)},
            "code": {f"{K.X}/export_baselines_e02z.py": L.sha256(Path(__file__)),
                     K.E02C_EXPORT: L.sha256(L.rp(K.E02C_EXPORT))}}
    EB._write_json(summ, out / "summary.json")
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--verify", action="store_true", help="rebuild and compare (gate)")
    a = ap.parse_args(argv)
    if L.sha256(L.rp(K.E02C_EXPORT)) != K.E02C_EXPORT_SHA:
        raise SystemExit("E02c's export_baselines.py is not at its registered sha256")
    if a.verify:
        r = EB.verify_export(a.out, export)
        print(json.dumps({"identical": r["identical"], "problems": r["problems"]}, indent=1))
        return 0 if r["identical"] else 1
    s = export(a.out)
    print(json.dumps({"n_wells": s["n_wells"], "n_rows": s["n_rows"],
                      "reference_chosen": {w: v["chosen"] for w, v in s["reference"].items()},
                      "files": s["files"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
