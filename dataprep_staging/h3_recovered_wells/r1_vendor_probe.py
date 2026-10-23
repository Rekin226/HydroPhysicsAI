"""Repair round 1: the credentialed vendor route of the R-h probe, with existing access only.

The first round did not query the vendor API because no ``WISENVR_*`` variable was set. The
verifier pointed out that the project's existing vendor access (the one that served the
2026-09-30 station list and the ``heads_202210_202610`` fetch) is available. This script uses
the repository's own client, ``hydrophysics.twin.fetch_amp.Client`` (imported read-only),
which takes host, account and token URL **only from the environment**; nothing here holds
or prints a credential, a host or an account.

Two windows of ``gw-wra-gw10min-obs``:

* ``rh_2019_2022``: 2019-01-01 .. 2023-01-01 for the 23 wells whose records stop in 2018-12
  (R-h; ``H3_evidence/rh_probe/rh_probe_wells.csv``), pre-origin;
* ``post_origin_2023_2026``: 2023-01-01 .. 2026-10-01 for the 16 H3 test wells, the raw
  material of the plan's optional secondary 2023-26 test set. Only the ``datetime`` column is
  read back (availability counts); no value is read, and nothing of it enters a gate,
  threshold or choice (U7).

Responses go to the L0 directory ``data_fetch_api/new_measurements/wra_gw10min_h3_probe/
2026-10-14/<window>/<sid>.parquet`` (non-empty responses only) with a ``SOURCE.json``; the
per-station availability goes to ``H3_evidence/repair1/vendor_probe.json``.

Usage (credentials in the environment): python -I r1_vendor_probe.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import pandas as pd  # noqa: E402
from common import source  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

DATASET = "gw-wra-gw10min-obs"
FETCH_DAY = "2026-10-14"
L0 = C.REPO / "data_fetch_api/new_measurements/wra_gw10min_h3_probe" / FETCH_DAY
WINDOWS = {"rh_2019_2022": ("2019-01-01T00:00:00", "2023-01-01T00:00:00"),
           "post_origin_2023_2026": ("2023-01-01T00:00:00", "2026-10-01T00:00:00")}
URL = ("<WISENVR_BASE_URL>/" + DATASET + "/station/{sid}/data?orient=parquet"
       "&start_datetime={start}&end_datetime={end}")


def stations() -> dict[str, list[str]]:
    rh = pd.read_csv(C.EVIDENCE / "rh_probe" / "rh_probe_wells.csv", dtype={"sid": str})
    reg = json.loads((C.EVIDENCE / "registered_wells.json").read_text())
    return {"rh_2019_2022": sorted(rh.sid), "post_origin_2023_2026": sorted(reg["test_wells"])}


def availability(path: Path) -> dict:
    t = pd.to_datetime(pd.read_parquet(path, columns=["datetime"])["datetime"])
    per = t.dt.to_period("M").astype(str)
    days = t.dt.normalize()
    by_m = per.value_counts().sort_index()
    dm = days.groupby(per).nunique()
    return {"rows": int(len(t)), "first": str(t.min()), "last": str(t.max()),
            "months_with_samples": int(len(by_m)),
            "months_with_ge_20_days": int((dm >= 20).sum()),
            "median_step_min": float(t.sort_values().diff().dt.total_seconds().median() / 60)
            if len(t) > 1 else None,
            "samples_by_month": {k: int(v) for k, v in by_m.items()}}


def main():
    from hydrophysics.twin.fetch_amp import Client
    cl = Client.from_env()
    log = {"dataset": DATASET, "fetch_day": FETCH_DAY, "windows": {}, "errors": []}
    files = []
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for win, sids in stations().items():
        a, b = WINDOWS[win]
        out = {}
        for sid in sids:
            t0 = time.time()
            try:
                df = cl.station_interval(DATASET, sid, a, b)
            except FileNotFoundError:
                out[sid] = {"rows": 0, "status": "404 (station unknown to the dataset)"}
                continue
            except ValueError as e:
                msg = str(e)
                if "no datetime column" in msg:
                    out[sid] = {"rows": 0, "status": "empty response"}
                    continue
                log["errors"].append({"window": win, "sid": sid, "error": msg[:200]})
                continue
            except Exception as e:  # network or HTTP error: recorded, not fatal
                log["errors"].append({"window": win, "sid": sid,
                                      "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            if df.empty:
                out[sid] = {"rows": 0, "status": "no rows in window"}
                continue
            p = L0 / win / f"{sid}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            df.to_parquet(p, index=False)
            p.chmod(0o444)
            rec = availability(p)
            rec.update(status="rows written", seconds=round(time.time() - t0, 1))
            out[sid] = rec
            files.append({"file": f"{win}/{sid}.parquet",
                          "url": URL.format(sid=sid, start=a, end=b),
                          "note": "host redacted; the client splits capped responses into "
                                  "sub-intervals (fetch_amp.Client.station_interval)"})
            print(win, sid, rec["rows"], rec["first"], rec["last"], flush=True)
        log["windows"][win] = {"start": a, "end_exclusive": b, "stations": out}
    log["finished"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log["started"] = started
    ev = C.EVIDENCE / "repair1"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "vendor_probe.json").write_text(json.dumps(log, indent=1))
    if files:
        rec = source.build_source(
            L0, source_id="wra_gw10min_h3_probe/2026-10-14",
            title="WRA groundwater 10-minute observations (gw-wra-gw10min-obs) for the H3 "
                  "R-h probe (2019-2022, 23 wells) and the H3 test wells (2023-2026)",
            publisher="Water Resources Agency, via the project's existing WiseEnvr API access",
            licence="as for AMP_V2/data/wells (vendor-served WRA monitoring data; project "
                    "access; not for redistribution)",
            fetched=log["finished"], files=files,
            client="hydrophysics.twin.fetch_amp.Client (credentials from the environment "
                   "only)",
            fetched_by="WP H3 repair round, dataprep_staging/h3_recovered_wells/"
                       "r1_vendor_probe.py",
            empty_or_missing={w: sorted(s for s, r in v["stations"].items() if not r["rows"])
                              for w, v in log["windows"].items()},
            errors=log["errors"])
        source.write_source_json(L0, rec)
    print(json.dumps({w: {"with_rows": sum(1 for r in v["stations"].values() if r["rows"]),
                          "n": len(v["stations"])} for w, v in log["windows"].items()}),
          "errors", len(log["errors"]))


if __name__ == "__main__":
    main()
