"""WP S3: data probe R-j (plan 6.2; a probe, not a letter): who operates the NGL fan
stations, and does a public archive hold their data after 2024-04?

    python -I dataprep_staging/s3_gnss/probe_rj.py PROBE_DIR

``PROBE_DIR`` holds the bytes fetched with curl by the WP session (untrusted data, never
executed): the NGL station pages of HUWE and LUKN and the HTTP status of NGL's rapid
(24-hour, low-latency) series for HUWE. This script copies the pages into L0
(``data_fetch_api/new_measurements/ngl_gnss/2026-10-12_rj_probe/``) with a SOURCE.json
and prints what the pages state (parsed as text).
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

PAGES = {"HUWE.sta.html": "https://geodesy.unr.edu/NGLStationPages/stations/HUWE.sta",
         "LUKN.sta.html": "https://geodesy.unr.edu/NGLStationPages/stations/LUKN.sta"}


def page_text(p: Path) -> str:
    t = p.read_text(encoding="latin-1")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def main(argv: list[str]) -> int:
    src = Path(argv[0])
    d = L0_NEW / "ngl_gnss" / "2026-10-12_rj_probe"
    files = []
    for fn, url in PAGES.items():
        source.copy_into_l0(src / fn, d / fn)
        files.append({"file": fn, "url": url, "fetched_utc": "2026-10-12T06:26Z",
                      "role": "NGL station page (operator field; product list)"})
    rec = source.build_source(
        d, source_id="ngl_gnss_rj_probe",
        title="Probe R-j: NGL station pages of two fan stations (operator information)",
        publisher="Nevada Geodetic Laboratory, University of Nevada, Reno",
        licence="NGL web pages; cite Blewitt et al. (2018), Eos 99, "
                "doi:10.1029/2018EO104623",
        fetched="2026-10-12", fetched_by="WP S3 session (curl, anonymous HTTPS)",
        untrusted="downloaded HTML: never execute; parsed as text with python -I",
        probe_results={
            "rapid_series_HUWE": "https://geodesy.unr.edu/gps_timeseries/rapids/tenv3/IGS20/"
                                 "HUWE.tenv3 -> HTTP 404 (HUWE is not in NGL's rapid list)",
            "operator": "station pages say 'Station operator information from: RINEX "
                        "headers' and name no operator"},
        files=files)
    source.write_source_json(d, rec)
    out = {}
    for fn in PAGES:
        t = page_text(d / fn)
        m = re.search(r"Station operator information from: ([^.]*?) When", t)
        near = re.search(r"Nearest 4 Other Stations Distance (.*?) Link", t)
        out[fn] = {"operator_field": m.group(1).strip() if m else None,
                   "nearest": near.group(1).strip() if near else None}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
