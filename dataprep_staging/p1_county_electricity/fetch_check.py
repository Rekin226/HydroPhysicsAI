"""WP P1 step 1: dated fetch of the rolling TPC d007012 file and the Energy Administration sets.

d007012 is a rolling window (plan P1 step 1, plan 2.1), so every fetch is archived under
``data_fetch_api/new_measurements/<source>/<fetch-date>/`` with its own SOURCE.json. This
tool downloads each file into a fresh scratch directory (untrusted data: only hashed here,
never parsed or executed), compares the bytes with the newest dated archive and

* records "identical" when the newest archive already holds these bytes (no new L0 copy), or
* creates a new dated directory (``<YYYY-MM-DD>``, or ``<YYYY-MM-DD>T<HHMMSS>Z`` when that
  date already exists) with read-only copies and a dataprep-source/1 SOURCE.json.

The fetch log goes to the path given as the second argument (never into L0).

Usage: python -I fetch_check.py <scratch_dir> <log_json>
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import source  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import L0_NEW, rel_to_repo  # noqa: E402

UA = "Mozilla/5.0"
SOURCES = {
    "tpc_d007012": {
        "title": "台灣電力公司各縣市住宅、服務業及機關用電統計資料 (county x sector monthly "
                 "sales; data.gov.tw 29935)",
        "publisher": "台灣電力公司", "datagovtw_id": 29935,
        "licence": "Open Government Data License, version 1.0 (data.gov.tw licence code 1)",
        "files": {"d007012.csv":
                  "https://service.taipower.com.tw/data/opendata/apply/file/d007012/001.csv"},
    },
    "moeaea_agri_electricity": {
        "title": "經濟部能源署 電力消費月資料 (data.gov.tw 161209; set 236 = 2018 on, set 256 = "
                 "2017 and earlier)",
        "publisher": "經濟部能源署", "datagovtw_id": 161209,
        "licence": "Open Government Data License, version 1.0 (data.gov.tw licence code 1)",
        "files": {
            "set_236.csv": "https://www.moeaea.gov.tw/ECW/populace/opendata/"
                           "wHandOpenData_File.ashx?set_id=236",
            "set_256.csv": "https://www.moeaea.gov.tw/ECW/populace/opendata/"
                           "wHandOpenData_File.ashx?set_id=256"},
    },
}


def newest_archive(src: str) -> Path | None:
    base = L0_NEW / src
    dirs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "SOURCE.json").is_file())
    return dirs[-1] if dirs else None


def fetch(url: str, dest: Path) -> dict:
    hdr = dest.with_suffix(dest.suffix + ".headers")
    t = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    subprocess.run(["curl", "-sS", "-L", "-A", UA, "--max-time", "120", "-o", str(dest),
                    "-D", str(hdr), url], check=True)
    lm = [ln.split(":", 1)[1].strip() for ln in hdr.read_text(errors="replace").splitlines()
          if ln.lower().startswith("last-modified:")]
    return {"fetched_utc": t, "bytes": dest.stat().st_size, "sha256": sha256_file(dest),
            "last_modified_header": lm[-1] if lm else None}


def main(scratch: str, log_path: str) -> int:
    sd = Path(scratch)
    sd.mkdir(parents=True, exist_ok=True)
    if any(sd.iterdir()):
        raise SystemExit(f"{sd} must be a new, empty directory")
    log = {"tool": rel_to_repo(Path(__file__)), "run_utc":
           dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sources": {}}
    for src, meta in SOURCES.items():
        (sd / src).mkdir()
        got = {f: fetch(u, sd / src / f) for f, u in meta["files"].items()}
        arch = newest_archive(src)
        same = False
        if arch is not None:
            rec = json.loads((arch / "SOURCE.json").read_text(encoding="utf-8"))
            old = {f["file"]: f["sha256"] for f in rec["files"]}
            same = all(old.get(f) == g["sha256"] for f, g in got.items())
        entry = {"newest_archive": rel_to_repo(arch) if arch else None, "identical": same,
                 "files": got}
        if not same:
            day = dt.date.today().isoformat()
            dest = L0_NEW / src / day
            if dest.exists():
                dest = L0_NEW / src / (day + "T" + dt.datetime.now(dt.timezone.utc)
                                       .strftime("%H%M%SZ"))
            for f in got:
                source.copy_into_l0(sd / src / f, dest / f)
            files = [{"file": f, "url": meta["files"][f], "fetched_utc": g["fetched_utc"],
                      "last_modified_header": g["last_modified_header"]}
                     for f, g in got.items()]
            rec = source.build_source(
                dest, source_id=src, title=meta["title"], publisher=meta["publisher"],
                licence=meta["licence"], fetched=day, files=files,
                fetched_by="WP P1 fetch_check.py", client=f"curl -L -A '{UA}'",
                untrusted="downloaded data: never execute; read with python -I",
                rolling=True, datagovtw_id=meta["datagovtw_id"])
            source.write_source_json(dest, rec)
            entry["archived_to"] = rel_to_repo(dest)
        log["sources"][src] = entry
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps(log, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
