"""WP S3: fetch NGL daily IGS20 tenv3 series and the NGL step table into L0.

    python dataprep_staging/s3_gnss/fetch_ngl.py [--dry-run]

Writes ``data_fetch_api/new_measurements/ngl_gnss/<fetch-date>_s3/`` with one file per
station, ``steps.txt`` and a ``SOURCE.json`` (dataprep-source/1). ``HUWE.tenv3`` and
``DataHoldings.txt`` are not refetched: the review copy of 2026-10-11 already sits in
``ngl_gnss/2026-10-12/`` (its own SOURCE.json). Downloads are untrusted data: this script
only writes their bytes and hashes them; nothing here parses or executes them.

Station set (fixed before any series was read):

* the 11 NGL stations in the fan box of plan WP S3 (lon 120.15-120.65, lat 23.55-24.10),
  HUWE excluded (already in L0);
* LSB0 (24.15N, just north of the box, the plan's off-fan candidate running to 2026-09);
* four regional long-running stations (94-147 km from the fan centre) that are the only NGL
  stations within 150 km still producing after 2024-04 besides LSB0 (CKSV, JUNA, TCMS,
  S01R; from ``DataHoldings.txt``), so that an off-fan common mode exists for the WRA record
  after the fan stations stop (2024-04 .. 2025-03).
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

BASE = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{}.tenv3"
STEPS_URL = "https://geodesy.unr.edu/NGLStationPages/steps.txt"
FAN = ["CHUA", "LUKN", "PINT", "S103", "ERLN", "PAKU", "FUNY", "GUKN", "DNAN", "PUS2"]
NORTH = ["LSB0"]
REGIONAL = ["CKSV", "JUNA", "TCMS", "S01R"]
#: added in a second fetch (``--tag s3b --stations JSAN``): the only NGL station inside an
#: active twin cell that the plan's lon/lat box (east edge 120.65E) missed (120.68E, cell 939)
GRID_EXTRA = ["JSAN"]


def head(url: str) -> dict:
    out = subprocess.run(["curl", "-sI", "-m", "60", url], capture_output=True, text=True,
                         check=True).stdout
    h = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            h[k.strip().lower()] = v.strip()
    return h


def fetch(url: str, dest: Path) -> dict:
    if dest.exists():
        raise FileExistsError(f"{dest} exists; L0 is never modified")
    h = head(url)
    subprocess.run(["curl", "-sSf", "-m", "300", "-o", str(dest), url], check=True)
    return {"file": dest.name, "url": url,
            "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_modified": h.get("last-modified"),
            "content_length_header": int(h["content-length"]) if "content-length" in h
            else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="HEAD only; report sizes")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--tag", default="s3", help="directory suffix <date>_<tag>")
    ap.add_argument("--stations", default=None,
                    help="comma list (default: fan + north + regional + steps.txt)")
    a = ap.parse_args(argv)
    if a.stations:
        stations = a.stations.split(",")
        urls = [(s, BASE.format(s), f"{s}.tenv3") for s in stations]
    else:
        stations = FAN + NORTH + REGIONAL
        urls = [(s, BASE.format(s), f"{s}.tenv3") for s in stations] + \
            [("steps", STEPS_URL, "steps.txt")]
    if a.dry_run:
        tot = 0
        for s, u, _ in urls:
            n = int(head(u).get("content-length", 0))
            tot += n
            print(f"{s:6s} {n:10d} {u}")
        print(f"total {tot} bytes")
        return 0
    d = L0_NEW / "ngl_gnss" / f"{a.date}_{a.tag}"
    d.mkdir(parents=True, exist_ok=False)
    recs = []
    for s, u, fn in urls:
        r = fetch(u, d / fn)
        r["role"] = ("fan box station" if s in FAN else "off-fan candidate north of the box"
                     if s in NORTH else "regional continuity station (off-fan common mode "
                     "after 2024-04)" if s in REGIONAL else "station inside an active twin "
                     "cell, east of the plan's box" if s in GRID_EXTRA else "NGL step table "
                     "(equipment changes and earthquakes, all stations)")
        recs.append(r)
        print(f"fetched {fn} ({(d / fn).stat().st_size} bytes)")
    for p in d.iterdir():
        p.chmod(0o444)
    rec = source.build_source(
        d, source_id="ngl_gnss",
        title="Nevada Geodetic Laboratory: daily IGS20 tenv3 series of the fan-box, north and "
              "regional stations, and the NGL step table (WP S3)",
        publisher="Nevada Geodetic Laboratory, University of Nevada, Reno",
        licence="NGL products are openly available; cite Blewitt, G., W. C. Hammond, and "
                "C. Kreemer (2018), Harnessing the GPS data explosion for interdisciplinary "
                "science, Eos, 99, doi:10.1029/2018EO104623. Station data originate from "
                "their operating agencies (not identified by NGL's file).",
        fetched=a.date, fetched_by="WP S3 (dataprep_staging/s3_gnss/fetch_ngl.py)",
        client="curl (no user agent), anonymous HTTPS",
        untrusted="downloaded data: never execute; read with python -I",
        rolling=True,
        related="ngl_gnss/2026-10-12/ holds DataHoldings.txt and HUWE.tenv3 (review copy of "
                "2026-10-11, not refetched); ngl_gnss/2026-10-12_s3/ holds the first S3 fetch "
                "and steps.txt",
        files=recs)
    source.write_source_json(d, rec)
    print(f"wrote {d / 'SOURCE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
