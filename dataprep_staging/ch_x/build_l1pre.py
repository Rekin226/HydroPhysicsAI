"""Build the governed local product heads/L1_pre2012 v1 (user ruling R7.l1_pre2012): monthly
2010-01..2011-12 heads of I1.CH_backcast.v1's registered Changhua wells from ls_cache, with a
MANIFEST whose allowed uses are validator, design_time_evidence and initial_conditions.

Kept local and never committed: the product lives under results/ (gitignored); ls_cache came
from a backup and has no committed fetcher. The build reads head values, so it runs only
AFTER I1.CH_backcast.v1 is registered and after its product_build look is logged; it computes
no model-versus-observation statistic and prints only counts.

QC (``bc_lib.qc_monthly``): H1a's pass-1 sample screens with the robust reference fitted on
the 2010-11 window itself, clean samples -> hourly -> daily (>= 12 h) -> monthly (>= 20 days)
means, flag 0; no barometric correction, no fill (declared in the MANIFEST).

    python dataprep_staging/ch_x/build_l1pre.py --log-look [--skip-if-logged]   (once, first)
    python dataprep_staging/ch_x/build_l1pre.py --build

The build writes into a staging directory beside v1 and renames it to v1 only when the
MANIFEST is written, so a crashed build (an engineering retry) leaves no partial v1; a
rerun after ``--log-look`` needs ``--skip-if-logged`` (the look is logged once).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import avail_l1pre as AV  # noqa: E402
import bc_lib as BL  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402

PURPOSE = "product_build"
CODE = (f"{K.X}/build_l1pre.py", f"{K.X}/bc_lib.py", f"{K.X}/chx_const.py",
        "dataprep_staging/h1_heads_qc/screens.py", "dataprep_staging/h1_heads_qc/config.py")


def registered_ids(prereg_dir=None, rows_dir=None) -> tuple[list[str], str]:
    from harness import prereg3

    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    doc, sha = prereg3.load_prereg(K.BC_ID, **kw)
    for t in doc["tests"]:
        u = (t.get("external_units") or {}).get(K.BC_UNIT)
        if u:
            return sorted(map(str, u["ids"])), sha
    raise SystemExit(f"{K.BC_UNIT} not in {K.BC_ID}")


def look_logged(prereg_dir=None) -> bool:
    from common.paths import PREREG_DIR
    from harness import prereg3

    d = Path(prereg_dir or PREREG_DIR)
    return any(lk.get("purpose") == PURPOSE and lk.get("prereg_id") == K.BC_ID
               for lk in prereg3.read_chained(d / "looks.jsonl"))


def log_look(prereg_dir=None, rows_dir=None, skip_if_logged: bool = False) -> dict | None:
    from harness import prereg3

    if look_logged(prereg_dir):
        if skip_if_logged:
            return None
        raise SystemExit("REFUSED: the product_build look is already logged")
    ids, _ = registered_ids(prereg_dir, rows_dir)
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    return prereg3.record_look(
        [K.BC_UNIT], K.BC_SCOPE, K.WP, PURPOSE, prereg_id=K.BC_ID,
        note="heads/L1_pre2012 v1 build: QC and monthly aggregation of the registered wells' "
             "2010-11 ls_cache records; no model-vs-observation statistic", **kw)


def load_raw(sid: str) -> pd.Series:
    f = pd.read_parquet(AV.raw_path(sid), columns=["value", "datetime"])
    if "datetime" in f.columns:
        f = f.set_index("datetime")
    s = pd.to_numeric(f["value"], errors="coerce")
    s.index = pd.to_datetime(s.index)
    return s


def build_tables(ids: list[str], st: pd.DataFrame, loader=load_raw) -> dict:
    """(heads_monthly, station_meta, qc_summary) for ``ids`` (no file written)."""
    st = st.set_index("sid")
    rows, meta, qc = [], [], []
    for s in ids:
        r = st.loc[s]
        elev = pd.to_numeric(r.get("WellElevation"), errors="coerce")
        depth = pd.to_numeric(r.get("WellDepth"), errors="coerce")
        out = BL.qc_monthly(loader(s), None if pd.isna(elev) else float(elev),
                            None if pd.isna(depth) else float(depth))
        for m, v in out["monthly"].items():
            rows.append({"sid": s, "date": m, "value": float(v), "flag": 0,
                         "n_days": int(out["n_days"][m])})
        meta.append({"sid": s, "county": r["county"], "x": float(r["x"]),
                     "y": float(r["y"]), "layer_code": str(r["layer_code"])})
        qc.append({"sid": s, "n_raw": out["n_raw"], "n_clean": out["n_clean"],
                   "n_months": int(len(out["monthly"])), **out["kind_counts"]})
    hm = pd.DataFrame(rows, columns=["sid", "date", "value", "flag", "n_days"])
    return {"heads_monthly": hm, "station_meta": pd.DataFrame(meta),
            "qc_summary": pd.DataFrame(qc)}


def validate_tables(tb: dict) -> dict:
    hm = tb["heads_monthly"]
    win = set(BL.window_months())
    checks = {"finite_values": bool(np.isfinite(hm["value"]).all()),
              "months_in_window": bool(hm["date"].isin(win).all()),
              "flags_0": bool((hm["flag"] == 0).all()),
              "unique_rows": bool(not hm.duplicated(["sid", "date"]).any())}
    n = hm.groupby("sid").size()
    post = {"n_wells_ge_min_months": int((n >= K.BC_MIN_MONTHS).sum()),
            "min_wells": K.BC_MIN_WELLS, "min_months": K.BC_MIN_MONTHS}
    return {"checks": checks, "passed": all(checks.values()), "post_qc_counts": post}


def write_product(d: Path, tb: dict, val: dict, ids: list[str], decl_sha: str,
                  inputs: list[str]) -> dict:
    """Write the tables and the MANIFEST into the new version directory ``d``."""
    from common import manifest

    tb["heads_monthly"].to_parquet(d / "heads_monthly.parquet", index=False)
    tb["station_meta"].to_parquet(d / "station_meta.parquet", index=False)
    tb["qc_summary"].to_csv(d / "qc_summary.csv", index=False)
    (d / "validation.json").write_text(json.dumps(val, indent=1) + "\n")
    m = manifest.build_manifest(
        d, product=K.L1PRE_PRODUCT, layer="L1", inputs=inputs,
        code=[str(HERE.parents[1] / c) for c in CODE], version="v1",
        parameters={"window": list(K.BC_WINDOW), "wells": ids,
                    "qc": "H1a pass-1 sample screens (h1_heads_qc.screens.screen) with the "
                          "2010-11 window's own robust level/scale; clean -> hourly -> daily "
                          f"(>= {K.BC_DAY_MIN_HOURS} h) -> monthly (>= "
                          f"{K.BC_MONTH_MIN_DAYS} days); flag 0",
                    "not_done": "no barometric correction (no 2010-11 pressure product "
                                "held), no gap fill, no datum",
                    "registration": {"id": K.BC_ID, "sha256": decl_sha},
                    "ruling": f"{K.RULINGS} {K.RULING_L1PRE}"},
        fit_window_end=K.BC_WINDOW[1], validation_report="validation.json",
        gate={"passed": bool(val["passed"]),
              "criteria": "finite values, months inside 2010-01..2011-12, flags 0, unique "
                          "(sid, month) rows"},
        allowed_uses=list(K.L1PRE_ALLOWED), forbidden_uses=list(K.L1PRE_FORBIDDEN),
        personal_data="none (station ids and coordinates of public monitoring wells)",
        time_slicing={"column": "date", "params_fit_window_end": K.BC_WINDOW[1],
                      "encoding": "iso"},
        extra={"local_only": "never committed: ls_cache is backup data with no committed "
                             "fetcher (user ruling R7.l1_pre2012, 2026-11-01)",
               "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    manifest.write_manifest(d, m)
    return m


def build(prereg_dir=None, rows_dir=None, loader=load_raw) -> Path:
    ids, decl_sha = registered_ids(prereg_dir, rows_dir)
    if not look_logged(prereg_dir):
        raise SystemExit("REFUSED: log the product_build look first (--log-look)")
    d = IO.rp(K.L1PRE_V1)
    if d.exists():
        raise SystemExit(f"REFUSED: {K.L1PRE_V1} exists (built once; score_bc reads exactly "
                         "v1)")
    stage = d.parent / ".partial" / d.name      # named v1: the MANIFEST checks the name
    if stage.parent.exists():
        shutil.rmtree(stage.parent)     # a crashed earlier attempt (engineering retry)
    stage.mkdir(parents=True)
    st = AV.stations()
    tb = build_tables(ids, st, loader=loader)
    val = validate_tables(tb)
    write_product(stage, tb, val, ids, decl_sha,
                  [str(AV.raw_path(s)) for s in ids] + [str(IO.rp(K.STATIONS))])
    stage.rename(d)
    stage.parent.rmdir()
    q = val["post_qc_counts"]
    print(f"built {d}: {len(ids)} wells, {len(tb['heads_monthly'])} monthly rows; "
          f"{q['n_wells_ge_min_months']} wells with >= {K.BC_MIN_MONTHS} months; gate "
          f"{val['passed']}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--log-look", action="store_true")
    g.add_argument("--build", action="store_true")
    ap.add_argument("--skip-if-logged", action="store_true",
                    help="with --log-look: do nothing if this registration's look is logged")
    a = ap.parse_args(argv)
    if a.log_look:
        r = log_look(skip_if_logged=a.skip_if_logged)
        print("product_build look already logged" if r is None else
              f"logged product_build look at {r['time']}")
    else:
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
