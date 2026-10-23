"""WP A0 repair round: L0 registry of files under data_fetch_api/new_measurements/ that no
dataprep SOURCE.json covers (plan 2.1 asks for a SOURCE.json per download directory).

These directories predate the SOURCE.json convention (2026-09-30 vendor-API pulls, the
2026-10-11 CWA CODIS rain rebuild that the twin now uses, the 2023-26 test-network heads,
ring compaction files) and are read-only for the data-prep work, so A0 does not write into
them. Instead this product, ``results/dataprep/l0_registry/v<N>/`` (layer ``registry``),
lists every such file as an input with its sha256, plus the files of the two legacy
SOURCE.json records that carry no sha256 (wra_monthly_reports, cwa_station_inventory). The
checker re-hashes the registry with every run, so a later change to any registered L0 file
fails the check, and it stops reporting these files as orphans.

Nothing is read except bytes for hashing: the station metadata with contact fields
(ls-wra-dbm-obs-stations.parquet) and the owner-only download log (heads_download.json) are
hashed, never parsed or copied.

Usage: python -m common.a0.r2_l0_registry
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import check  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.manifest import build_manifest, new_version_dir, write_manifest  # noqa: E402
from common.paths import DATAPREP, L0_NEW, REPO, rel_to_repo  # noqa: E402
from common.source import SCHEMA_VERSION, listed_files  # noqa: E402
from common.uses import USES  # noqa: E402

PROVENANCE = {
    "cwa_codis_rain": (
        "CWA CODIS daily rainfall for the 26 twin gauges, fetched 2026-10-11 (repository "
        "commit 13860ee, 'rebuild the 26 rain gauges from CWA CODIS'); each raw/<station>_"
        "<year>.json has a sidecar .meta.json with url (https://codis.cwa.gov.tw/api/station), "
        "fetched_at, sha256 and bytes; rf_timeseries_codis*.csv are the derived daily table "
        "and its per-day station map. In use by the twin (chou-shui-data/data/rf_timeseries.csv "
        "lineage)."),
    "heads_202210_202610": (
        "WRA groundwater-network heads 2022-10..2026-10 for the 158 registered test wells, "
        "fetched 2026-09-30 through the vendor API (data_fetch_api/; the download log "
        "heads_download.json is owner-only and was hashed, not read). The 2023-26 scoring "
        "target: scoring_target / validator uses only (plan 4.6)."),
    "compaction": (
        "Multi-layer compaction-well (ring) files per site, fetched 2026-09-30 (the "
        "'2026-fetched compaction files' of the subsidence pilot's rings_common.py)."),
    "(root files)": (
        "Vendor-API pulls of 2026-09-30: station lists, samples and latest records for TPC "
        "meters (etc-tpc-etc1mon), groundwater 10-min (gw-wra-gw10min), leveling (ls-wra-lsp), "
        "deep benchmarks (ls-wra-dbm; contains contact names and phones: never copy) and "
        "multi-layer compaction (ls-wra-mlcw); overlap_audit.json and heads_download.json are "
        "the session's own audit and download log."),
    "wra_monthly_reports": (
        "Legacy SOURCE.json (2026-10-11) without sha256: the registry adds the hashes."),
    "cwa_station_inventory": (
        "Legacy SOURCE.json (2026-10-11) without sha256: the registry adds the hashes."),
}


def _group(rel: Path) -> str:
    parts = rel.relative_to(rel_to_repo(L0_NEW)).parts
    return parts[0] if len(parts) > 1 else "(root files)"


def main() -> int:
    sources = sorted(L0_NEW.rglob("SOURCE.json"))
    orphans = check.find_orphans(L0_NEW, sources, [])
    legacy = []
    for s in sources:
        rec = json.loads(s.read_text(encoding="utf-8"))
        if rec.get("schema") != SCHEMA_VERSION:
            legacy += [rel_to_repo(p) for p in listed_files(s) if p.is_file()]
    files = sorted(set(orphans) | set(legacy))
    groups: dict[str, dict] = {}
    entries = []
    for f in files:
        p = REPO / f
        st = p.stat()
        g = _group(Path(f))
        mt = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        role = "legacy_sha256_supplement" if f in legacy else "orphan"
        entries.append({"path": f, "bytes": st.st_size, "mtime_utc": mt, "group": g,
                        "role": role})
        gr = groups.setdefault(g, {"group": g, "n_files": 0, "bytes": 0, "mtime_min": mt,
                                   "mtime_max": mt, "provenance": PROVENANCE.get(g, "unknown")})
        gr["n_files"] += 1
        gr["bytes"] += st.st_size
        gr["mtime_min"] = min(gr["mtime_min"], mt)
        gr["mtime_max"] = max(gr["mtime_max"], mt)
    d = new_version_dir("l0_registry", root=DATAPREP)
    reg = {"created": dt.date.today().isoformat(), "by": "WP A0 repair round",
           "scope": rel_to_repo(L0_NEW),
           "rule": ("files covered by no dataprep-source/1 SOURCE.json; hashes in the "
                    "MANIFEST.json inputs; L0 itself is untouched"),
           "groups": sorted(groups.values(), key=lambda x: x["group"]),
           "files": entries}
    (d / "registry.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8")
    code = [Path(__file__).resolve(), REPO / "dataprep_staging/common/check.py",
            REPO / "dataprep_staging/common/source.py"]
    m = build_manifest(
        d, product="l0_registry", layer="registry",
        inputs=[{"path": e["path"], "role": e["role"]} for e in entries], code=code,
        parameters={"n_orphans": len(orphans), "n_legacy_supplement": len(legacy),
                    "groups": {g["group"]: g["n_files"] for g in groups.values()}},
        fit_window_end="2026-10", validation_report=None,
        gate={"passed": False, "criteria": "registry of pre-convention L0 files: provenance "
                                           "and hashes only, not a data product"},
        allowed_uses=[], forbidden_uses=sorted(USES),
        personal_data=("ls-wra-dbm-obs-stations.parquet holds station contact names and phone "
                       "numbers (hashed only, never parsed or copied); heads_download.json is "
                       "owner-only (mode 600), hashed only"))
    write_manifest(d, m)
    print(f"{d}: {len(entries)} files ({len(orphans)} orphans, {len(legacy)} legacy), "
          f"{sum(e['bytes'] for e in entries) / 1e6:.1f} MB; registry.json sha256 "
          f"{sha256_file(d / 'registry.json')[:12]}")
    for g in sorted(groups.values(), key=lambda x: x["group"]):
        print(f"  {g['group']:24s} {g['n_files']:4d} files {g['bytes'] / 1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
