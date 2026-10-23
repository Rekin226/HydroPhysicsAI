"""Round-4 checks of WP S2 (after the products are built; no rule is set here).

Subcommands:
  disclose   log, in the A1 look log, the looks made before round 4's declaration by the
             interrupted repair attempt (scratch run of printed_v2.py, about 2026-10-12T11:12Z)
             and by round 4's development run of build_v2.py with v1's labels
  guard      open deep_benchmarks/v2 and deep_benchmarks_printed/v2 for every use through
             the loader guard and record what is granted or refused
  repro      rebuild deep_benchmarks/v2 and deep_benchmarks_printed/v2 into a scratch
             directory and compare every table with the published version
  pandas     re-run the pandas non-nanosecond DatetimeIndex union probe (round 3)

Usage: python -I r4_checks.py <subcommand> [--out PATH] [--scratch DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.paths import DATAPREP, REPORTS_DIR, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

HERE = Path(__file__).resolve().parent
EV = REPORTS_DIR / "S2_evidence" / "r4"
PY = sys.executable
SIDS = ["SJES", "TKJS(1)", "TKJS(2)", "TKJS(3)", "NTUH(1)", "NTUH(2)", "STES"]


def disclose(which: str) -> list[dict]:
    from harness import prereg as pr
    out = []
    if which in ("repair_scratch", "all"):
        out.append(pr.record_look(
            [f"dbm_cy007001_printed:{s}:{p}" for s in SIDS for p in ("pre", "2023")],
            "deep_piles_cy007001_printed_reprints:2019-05..2025-11", "S2", "after_the_fact",
            note="disclosed by S2 round 4: the interrupted repair attempt ran printed_v2.py into "
                 "a scratch directory at about 2026-10-12T11:12Z (one primary printing per "
                 "sensor-month; coverage of the rounding sd on 245 pairs of two printings of "
                 "the same month, 0.80; 21 pre-origin sensor-months known only from a 2023 "
                 "re-printing). Printed values only; no feed compared."))
    if which in ("dev_run", "all"):
        out.append(pr.record_look(
            [f"dbm_cy007001_printed:{s}:pre" for s in SIDS] +
            [f"dbm_integrated_printed:{s}" for s in ("DLIO", "NTUH(1)", "SJES", "STES",
                                                       "TKJS(1)", "TKJS(3)", "WNES")],
            "deep_benchmarks_v2_development:2016-06..2026-01", "S2", "after_the_fact",
            note="S2 round 4: development runs of build_v2.py with v1's labels in place of "
                 "the pre-origin labels, before the declaration S2r4.deep_benchmarks_v2. "
                 "The full run reproduced v1; the pre-origin rows changed only in 17 TKJS(2) "
                 "flags. Later development runs exercise acceptance and test C's statistic "
                 "on these rows (looks logged separately by build_v2)."))
    return out


def guard_probe() -> dict:
    res = {}
    for prod in ("deep_benchmarks", "deep_benchmarks_printed"):
        d = sorted((DATAPREP / prod).glob("v*"), key=lambda p: int(p.name[1:]))[-1]
        r = {"version": rel_to_repo(d)}
        for use in sorted(USES):
            for w in (None, "2022-12"):
                key = use + ("" if w is None else "@2022-12")
                try:
                    p = guard.open_product(d, use, window_end=w)
                    tbl = "monthly.parquet" if prod == "deep_benchmarks" else \
                        "printed_monthly.parquet"
                    t = p.read_table(tbl)
                    r[key] = {"granted": True, "rows": int(len(t)),
                              "max_month": str(t["month"].max()) if len(t) else None,
                              "flags": sorted(int(x) for x in t["flag"].unique())}
                except Exception as exc:  # noqa: BLE001 - the refusal is the result
                    r[key] = {"granted": False, "error": type(exc).__name__,
                              "message": str(exc)[:200]}
        res[prod] = r
    return res


def repro(scratch: Path) -> dict:
    import pandas as pd
    out = {}
    for prod, script, extra in (("deep_benchmarks", "build_v2.py", ["--no-log"]),
                                ("deep_benchmarks_printed", "printed_v2.py", [])):
        d = sorted((DATAPREP / prod).glob("v*"), key=lambda p: int(p.name[1:]))[-1]
        sd = scratch / prod
        cmd = [PY, "-I", str(HERE / script), "--scratch", str(sd), *extra]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        rec = {"version": rel_to_repo(d), "returncode": rc.returncode,
               "stderr_tail": rc.stderr[-400:] if rc.returncode else ""}
        tabs = {}
        for f in sorted(d.iterdir()):
            if f.suffix not in (".parquet", ".csv"):
                continue
            g = sd / f.name
            if not g.exists():
                tabs[f.name] = "missing in rebuild"
                continue
            a = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            b = pd.read_parquet(g) if g.suffix == ".parquet" else pd.read_csv(g)
            cols = [c for c in a.columns if c != "product_version"]
            try:
                pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)
                tabs[f.name] = "identical" if f.read_bytes() == g.read_bytes() else \
                    "value-identical (product_version label differs)"
            except (AssertionError, KeyError) as exc:
                tabs[f.name] = f"differs: {str(exc)[:200]}"
        rec["tables"] = tabs
        va = json.loads((d / "validation.json").read_text(encoding="utf-8"))
        vb = json.loads((sd / "validation.json").read_text(encoding="utf-8"))
        rec["validation_differs_in"] = sorted(k for k in set(va) | set(vb)
                                              if va.get(k) != vb.get(k))
        out[prod] = rec
    return out


def pandas_probe() -> dict:
    import pandas as pd
    a = pd.date_range("2016-01-01", "2020-12-31 23:00", freq="h", unit="us")
    b = pd.date_range("2018-01-01", "2025-12-31 23:00", freq="h", unit="us")
    u = a.union(b)
    an, bn = a.as_unit("ns"), b.as_unit("ns")
    un = an.union(bn)
    return {"pandas": pd.__version__, "union_us_len": int(len(u)), "union_ns_len": int(len(un)),
            "union_us_max": str(u.max()), "union_ns_max": str(un.max()),
            "defect_present": bool(len(u) != len(un))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["disclose", "guard", "repro", "pandas"])
    ap.add_argument("--which", default="all")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "disclose":
        res = disclose(a.which)
    elif a.cmd == "guard":
        res = guard_probe()
    elif a.cmd == "repro":
        if a.scratch is None:
            raise SystemExit("repro needs --scratch")
        res = repro(a.scratch)
    else:
        res = pandas_probe()
    txt = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt + "\n", encoding="utf-8")
    print(txt[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
