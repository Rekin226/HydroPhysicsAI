"""Build the governed local product heads/L1_pre2012 v2 (user ruling R8.6a): monthly
2010-01..2011-12 heads from ls_cache for the wells that qualified in I1.backcast_2010_11.v1's
logged availability look (Yunlin and Changhua, ring-nest wells excluded), with a MANIFEST
whose allowed uses are validator, design_time_evidence and initial_conditions (the v1
precedent's set).

Kept local and never committed: the product lives under results/ (gitignored); ls_cache came
from a backup and has no committed fetcher. The build READS HEAD VALUES, so it runs only
after (1) the availability look is logged and its count passed, and (2) its own
``product_build`` look is logged on its OWN unit ``I1BC11P:l1pre2012_v2_build_2010_2011``
(members: the qualifying wells, window 2010-01..2011-12, never universal) -- not the test's
fresh units, not the availability unit, not E26's I1CHB units. The build precedes the
registration (plan #12: register after the product build and the model-only G0s, before any
2010-11 head is compared with anything); it refuses once I1.backcast_2010_11.v1 is
registered. It computes no model-versus-observation statistic and prints only counts.

QC exactly as v1 (``ch_x/build_l1pre.build_tables`` -> ``bc_lib.qc_monthly``): H1a's pass-1
sample screens with the 2010-11 window's own robust reference, clean samples -> hourly ->
daily (>= 12 h) -> monthly (>= 20 days) means, flag 0; no barometric correction, no fill.

    python dataprep_staging/bc11/build_l1pre_v2.py --log-look [--skip-if-logged]   (once)
    python dataprep_staging/bc11/build_l1pre_v2.py --build

The build writes into a staging directory and renames it to v2 only when the MANIFEST is
written, so a crashed build (an engineering retry) leaves no partial v2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
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

import avail_bc11 as A  # noqa: E402
import avail_l1pre as AV  # noqa: E402
import avail_lib as AL  # noqa: E402
import bc11_const as K  # noqa: E402
import bc_lib as BCL  # noqa: E402
import build_l1pre as BLD  # noqa: E402  (v1's frozen table builder, imported unchanged)
import chx_io as IO  # noqa: E402

CODE = (f"{K.X}/build_l1pre_v2.py", f"{K.X}/bc11_const.py", f"{K.CHX}/build_l1pre.py",
        f"{K.CHX}/bc_lib.py", f"{K.CHX}/chx_const.py", f"{K.CHX}/avail_bc11.py",
        "dataprep_staging/h1_heads_qc/screens.py", "dataprep_staging/h1_heads_qc/config.py")
NOTE = ("heads/L1_pre2012 v2 build (user ruling R8.6a): QC and monthly aggregation of the "
        "2010-11 ls_cache records of the wells that qualified in the logged availability "
        "look; no model-vs-observation statistic; before I1.backcast_2010_11.v1 is "
        "registered, on this build unit only (not the test's fresh units)")


def _pdir(prereg_dir=None) -> Path:
    from common.paths import PREREG_DIR

    return Path(prereg_dir or PREREG_DIR)


# ----------------------------------------------------------------------------- preconditions
def availability(prereg_dir=None) -> dict:
    """The logged availability look and its passed count (refuses otherwise)."""
    lk = AL.logged_look(K.AVAIL_UNIT, prereg_dir)
    if lk is None:
        raise SystemExit("REFUSED: the availability look is not logged")
    if not AL.note_matches(lk, A.constants(), A.RULE_TEXT):
        raise SystemExit("REFUSED: the logged availability look's note does not carry "
                         "avail_bc11.py's constants hash and rule text")
    p = IO.rp(K.AVAIL_OUT)
    if not p.exists():
        raise SystemExit(f"REFUSED: {K.AVAIL_OUT} missing (run the count first)")
    a = json.loads(p.read_text())
    if a.get("look_time") != lk["time"]:
        raise SystemExit("REFUSED: avail.json does not belong to the logged look")
    f = a["feasibility"]
    if not f["passed"]:
        raise SystemExit("REFUSED: the availability stop rule did not pass (not evaluable)")
    return {"look_time": lk["time"], "path": K.AVAIL_OUT, "sha256": IO.sha256(p),
            "wells": sorted(map(str, f["eligible_wells"])),
            "constants_sha256": f["constants_sha256"]}


def registered(prereg_dir=None) -> bool:
    from harness import prereg3

    d = _pdir(prereg_dir)
    return (d / f"{K.ID}.json").exists() or any(
        r.get("id") == K.ID for r in prereg3.read_chained(d / "INDEX.jsonl"))


def members(ids) -> dict:
    return {K.PRODUCT_UNIT: {
        "description": "heads/L1_pre2012 v2 build: the qualifying wells of "
                       f"{K.AVAIL_UNIT}'s availability count, 2010-11 ls_cache values (QC "
                       "and monthly aggregation only)",
        "ids": sorted(map(str, ids)), "window": K.WINDOW_TEXT}}


def product_looks(prereg_dir=None) -> list[dict]:
    from harness import prereg3

    return [lk for lk in prereg3.read_chained(_pdir(prereg_dir) / "looks.jsonl")
            if K.PRODUCT_UNIT in (lk.get("units") or [])
            and lk.get("purpose") == K.PRODUCT_PURPOSE]


def product_look(prereg_dir=None) -> dict | None:
    lks = product_looks(prereg_dir)
    if len(lks) > 1:
        raise SystemExit(f"REFUSED: {len(lks)} product_build looks logged on {K.PRODUCT_UNIT}")
    return lks[0] if lks else None


def log_look(prereg_dir=None, rows_dir=None, registries=None, skip_if_logged: bool = False
             ) -> dict | None:
    """Log the product_build look (once), after the availability look, before registration."""
    from harness import prereg3

    av = availability(prereg_dir)
    if registered(prereg_dir):
        raise SystemExit(f"REFUSED: {K.ID} is registered; the product is built before it")
    if product_look(prereg_dir) is not None:
        if skip_if_logged:
            return None
        raise SystemExit("REFUSED: the product_build look is already logged")
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    return prereg3.record_look(
        [K.PRODUCT_UNIT], K.SCOPE, K.WP, K.PRODUCT_PURPOSE, members=members(av["wells"]),
        note=f"{NOTE}; availability {av['path']} sha256 {av['sha256']}",
        registries=registries, **kw)


# ----------------------------------------------------------------------------- tables
def validate_v2(tb: dict, county: dict) -> dict:
    """Engineering checks plus post-QC counts (the scored-year rule; counts only)."""
    hm = tb["heads_monthly"]
    win = set(BCL.window_months(*K.WINDOW))
    checks = {"finite_values": bool(np.isfinite(hm["value"]).all()),
              "months_in_window": bool(hm["date"].isin(win).all()),
              "flags_0": bool((hm["flag"] == 0).all()),
              "unique_rows": bool(not hm.duplicated(["sid", "date"]).any())}
    ok_wells = {c: 0 for c in K.COUNTIES}
    for s, g in hm.groupby("sid"):
        ms = list(g["date"])
        n11 = AL.n_in(ms, *K.SCORED)
        origin_ok = AL.n_in(ms, "2010-09", K.ORIGIN) >= 1      # within MAX_GAP of 2010-12
        if n11 >= K.MIN_WINDOW_MONTHS and origin_ok and county.get(s) in ok_wells:
            ok_wells[county[s]] += 1
    post = {"n_wells_scoreable_upper_bound": ok_wells,
            "rule": f">= {K.MIN_WINDOW_MONTHS} monthly values in 2011 and one in "
                    f"2010-09..{K.ORIGIN} (an upper bound: CL needs >= {K.MIN_FIT_MONTHS} "
                    "fitted 2012-22 months too)"}
    return {"checks": checks, "passed": all(checks.values()), "post_qc_counts": post}


def write_product(d: Path, tb: dict, val: dict, ids: list[str], av: dict, look: dict,
                  inputs: list[str]) -> dict:
    from common import manifest

    tb["heads_monthly"].to_parquet(d / "heads_monthly.parquet", index=False)
    tb["station_meta"].to_parquet(d / "station_meta.parquet", index=False)
    tb["qc_summary"].to_csv(d / "qc_summary.csv", index=False)
    (d / "validation.json").write_text(json.dumps(val, indent=1) + "\n")
    m = manifest.build_manifest(
        d, product=K.PRODUCT_NAME, layer="L1", inputs=inputs,
        code=[str(HERE.parents[1] / c) for c in CODE], version=K.PRODUCT_VERSION,
        parameters={"window": list(K.WINDOW), "wells": ids,
                    "counties": sorted(set(tb["station_meta"]["county"])),
                    "qc": "as heads/L1_pre2012 v1: H1a pass-1 sample screens "
                          "(h1_heads_qc.screens.screen) with the 2010-11 window's own robust "
                          "level/scale; clean -> hourly -> daily (>= 12 h) -> monthly (>= 20 "
                          "days); flag 0",
                    "not_done": "no barometric correction (no 2010-11 pressure product "
                                "held), no gap fill, no datum",
                    "availability": {"path": av["path"], "sha256": av["sha256"],
                                     "look_time": av["look_time"],
                                     "constants_sha256": av["constants_sha256"]},
                    "build_look": {"unit": K.PRODUCT_UNIT, "time": look["time"]},
                    "for": K.ID, "ruling": f"{K.RULINGS} {K.RULING_PRODUCT}"},
        fit_window_end=K.WINDOW[1], validation_report="validation.json",
        gate={"passed": bool(val["passed"]),
              "criteria": "finite values, months inside 2010-01..2011-12, flags 0, unique "
                          "(sid, month) rows"},
        allowed_uses=list(K.PRODUCT_ALLOWED), forbidden_uses=list(K.PRODUCT_FORBIDDEN),
        personal_data="none (station ids and coordinates of public monitoring wells)",
        time_slicing={"column": "date", "params_fit_window_end": K.WINDOW[1],
                      "encoding": "iso"},
        extra={"local_only": "never committed: ls_cache is backup data with no committed "
                             "fetcher (user ruling R8.6a, 2026-11-03)",
               "supersedes": "none: v1 (Changhua, I1.CH_backcast.v1) stays as built",
               "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    manifest.write_manifest(d, m)
    return m


def build(prereg_dir=None, loader=BLD.load_raw) -> Path:
    av = availability(prereg_dir)
    if registered(prereg_dir):
        raise SystemExit(f"REFUSED: {K.ID} is registered; the product is built before it")
    look = product_look(prereg_dir)
    if look is None:
        raise SystemExit("REFUSED: log the product_build look first (--log-look)")
    m = (look.get("members") or {}).get(K.PRODUCT_UNIT) or {}
    if sorted(map(str, m.get("ids") or [])) != av["wells"] or m.get("universal"):
        raise SystemExit("REFUSED: the logged product_build look's members are not the "
                         "qualifying wells")
    d = IO.rp(K.PRODUCT)
    if d.exists():
        raise SystemExit(f"REFUSED: {K.PRODUCT} exists (built once; the scorer reads exactly "
                         "v2)")
    stage = d.parent / ".partial_v2" / d.name     # named v2: the MANIFEST checks the name
    if stage.parent.exists():
        shutil.rmtree(stage.parent)     # a crashed earlier attempt (engineering retry)
    stage.mkdir(parents=True)
    st = AV.stations()
    ids = av["wells"]
    tb = BLD.build_tables(ids, st, loader=loader)
    county = dict(zip(st["sid"], st["county"], strict=True))
    val = validate_v2(tb, county)
    write_product(stage, tb, val, ids, av, look,
                  [str(AV.raw_path(s)) for s in ids] + [str(IO.rp(K.STATIONS))])
    stage.rename(d)
    stage.parent.rmdir()
    q = val["post_qc_counts"]["n_wells_scoreable_upper_bound"]
    print(f"built {d}: {len(ids)} wells, {len(tb['heads_monthly'])} monthly rows; scoreable "
          f"upper bound {q}; gate {val['passed']}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--log-look", action="store_true")
    g.add_argument("--build", action="store_true")
    ap.add_argument("--skip-if-logged", action="store_true")
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
