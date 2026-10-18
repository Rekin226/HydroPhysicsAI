"""WP S4a: tally every byte the WP transferred from the LiCSAR portal (budget, decision D1).

Sources (all under ``results/dataprep/insar_tools/s4a_run/``):
  fetch_log_listings.jsonl, fetch_log_pairdirs.jsonl   exact response sizes
  fetch_log_<frame>.jsonl                              GDAL network stats + raw range bytes
  aborted/fetch_log_*.run1.jsonl                       run 1 (removed misaligned clips)
  probes/probes.json                                   design-time probes (one upper bound)
  probes/resume_probes.json                            resume-session probes (curl)
  size surveys                                         3.6 MB measured (run 3 of the survey);
                                                       runs 1-2 estimated at 25 KB per file
  validation spot checks                               spot_checks.json (if present)
Writes ``transfer_tally.json``.

Usage (any env): python -I tally.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FRAMES, RUN_DIR  # noqa: E402

PER_HEADER_FILE = 25_000      # survey run 3: 3.6 MB over 144 files


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def main() -> int:
    t: dict[str, dict] = {}
    t["listings_and_metadata_text"] = {"bytes": sum(r["bytes"] for r in jl(
        RUN_DIR / "fetch_log_listings.jsonl")), "kind": "measured"}
    t["pair_directory_pages"] = {"bytes": sum(r.get("bytes", 0) for r in jl(
        RUN_DIR / "fetch_log_pairdirs.jsonl")), "kind": "measured"}
    t["size_survey_run3"] = {"bytes": 3_600_000, "kind": "measured (log, rounded to 0.1 MB)"}
    t["size_survey_runs1_2"] = {"bytes": (49 + 49) * 2 * PER_HEADER_FILE,
                                "kind": "estimate: 49 + 49 sampled pairs x 2 files x 25 KB "
                                        "(run 1 by the interrupted earlier S4a session, run 2 "
                                        "stopped at the first missing GeoTIFF)"}
    pr = json.loads((RUN_DIR / "probes" / "probes.json").read_text())["probes"]
    t["design_probes"] = {"bytes": sum(p["bytes"] for p in pr if p.get("host") != "github.com"),
                          "kind": "measured, one upper bound (the HTTP 429 attempt counted as "
                                  "the whole 33.8 MB file)"}
    rp = RUN_DIR / "probes" / "resume_probes.json"
    if rp.exists():
        t["resume_probes"] = {"bytes": sum(p["bytes"] for p in json.loads(
            rp.read_text())["probes"]), "kind": "measured (curl), HEAD bodies as upper bound"}
    t["run1_aborted_clips"] = {"bytes": sum(r.get("bytes_transferred") or 0 for r in jl(
        RUN_DIR / "aborted" / "fetch_log_069A_06694_131402.run1.jsonl")), "kind": "measured"}
    for frame in FRAMES:
        recs = jl(RUN_DIR / f"fetch_log_{frame}.jsonl")
        if recs:
            ok = [r for r in recs if r.get("status") == "ok"]
            t[f"subsets_{frame}"] = {
                "bytes": sum(r.get("bytes_transferred") or 0 for r in recs),
                "kind": "measured (GDAL network statistics + raw range bytes)",
                "n_files_ok": len(ok), "n_records": len(recs),
                "remote_full_bytes_of_ok_files": sum(r["remote_bytes"] for r in ok),
                "stored_bytes": sum(r["out_bytes"] for r in ok)}
    sc = RUN_DIR / "spot_checks.json"
    if sc.exists():
        t["validation_spot_checks"] = {"bytes": json.loads(sc.read_text()).get("bytes", 0),
                                       "kind": "measured"}
    total = sum(v["bytes"] for v in t.values())
    out = {"items": t, "total_bytes": total, "cap_bytes": 3_000_000_000,
           "d1_primary_guidance_bytes": 1_400_000_000,
           "primary_subset_bytes": t.get("subsets_069A_06694_131402", {}).get("bytes", 0),
           "remaining_bytes": 3_000_000_000 - total,
           "not_counted": "LiCSBAS git clone and conda-forge packages (not the LiCSAR portal); "
                          "a GitHub zip probe of 4.5 MB"}
    (RUN_DIR / "transfer_tally.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in t.items():
        print(f"{k:40s} {v['bytes']/1e6:10.1f} MB")
    print(f"{'TOTAL':40s} {total/1e6:10.1f} MB   remaining {out['remaining_bytes']/1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
