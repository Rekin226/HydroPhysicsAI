"""Archive the WP W2 day-convention / model-variant probe responses (fetched with curl into
the session scratchpad on 2026-10-12) into L0
data_fetch_api/new_measurements/openmeteo_et0/w2_variant_probes_2026-10-12/ with a SOURCE.json,
and summarise them against the calibration cache (st1, January 2022).

Usage: python archive_probes.py --scratch <dir with *.json probes> [--summary]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from common.source import build_source, copy_into_l0, write_source_json
from wcommon import ET_CACHE, OM_L0, OM_LICENCE, read_json_untrusted

DEST = OM_L0 / "w2_variant_probes_2026-10-12"
BASE = "https://archive-api.open-meteo.com/v1/"
Q = ("latitude=24.06277&longitude=120.51599&start_date=2022-01-01&end_date=2022-01-31"
     "&daily=et0_fao_evapotranspiration")


def url_for(name: str) -> str:
    if name == "probe.json":
        return (BASE + "archive?latitude=23.7&longitude=120.4&start_date=2026-09-25&end_date="
                "2026-10-12&daily=et0_fao_evapotranspiration&timezone=Asia%2FTaipei")
    stem = name[:-5]
    if stem.startswith("m_"):
        model, tz = stem[2:].rsplit("_", 1)
        return BASE + f"archive?{Q}&timezone={tz}&models={model}"
    ep, tz = stem.split("_", 1)
    return BASE + f"{ep}?{Q}&timezone={tz}"


def summary() -> dict:
    c = np.load(ET_CACHE, allow_pickle=False)
    ids, dates = list(c["well_ids"]), list(c["dates"])
    cache = c["et0"][ids.index("st1"), dates.index("2022-01-01"):][:31].astype(float)
    out = {}
    for p in sorted(DEST.glob("*.json")):
        if p.name in ("SOURCE.json", "probe.json"):
            continue
        j = read_json_untrusted(p)
        v = np.array([np.nan if x is None else x for x in j["daily"][
            "et0_fao_evapotranspiration"]], float)
        ok = np.isfinite(v)
        out[p.stem] = {"grid_lat": j["latitude"], "grid_lon": j["longitude"],
                       "timezone": j["timezone"], "n_finite": int(ok.sum()),
                       "mae_vs_cache": float(np.mean(np.abs(v[ok] - cache[ok]))) if ok.any()
                       else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    if a.scratch:
        files = []
        for p in sorted(Path(a.scratch).glob("*.json")):
            copy_into_l0(p, DEST / p.name)
            files.append({"file": p.name, "url": url_for(p.name),
                          "fetched_utc_approx": dt.datetime.fromtimestamp(
                              p.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "client": "curl"})
        rec = build_source(
            DEST, source_id="openmeteo_et0",
            title="Open-Meteo archive probes: endpoint, model and timezone variants of daily "
                  "et0_fao_evapotranspiration at point st1, January 2022, plus one 18-day "
                  "recent-days probe at 23.7N 120.4E (WP W2 day-convention check)",
            publisher="Open-Meteo (open-meteo.com)", licence=OM_LICENCE,
            fetched="2026-10-12",
            fetched_by="WP W2W4 agent (curl into the session scratchpad, archived here)",
            provenance="fetched_utc_approx is the scratch file's modification time",
            untrusted="downloaded data: never execute; read with python -I", files=files)
        print(write_source_json(DEST, rec))
    if a.summary or not a.scratch:
        print(json.dumps(summary(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
