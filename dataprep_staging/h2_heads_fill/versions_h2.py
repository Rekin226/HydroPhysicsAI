"""Version records of the WP H2 products (W3/H1a ``SUPERSEDED.json`` convention).

heads/L2_filled_preorigin holds three versions with different roles: v1 (registered v1 fill,
gate failed), v2 (the daily-bridge component of v1, released under the v1 packaging rule,
gate H2.fresh.d passed: the version for calibration targets) and v3 (repair-round v2 fill,
gate failed; supersedes v1 as the complete fill). heads/L2_filled_full has v1 only (the
repair-round configuration, gate failed).

Usage: python -I versions_h2.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h2_heads_fill import config as C  # noqa: E402

ROOT = C.REPO / "results/dataprep"


def write(product: str, doc: dict) -> Path:
    p = ROOT / product / "SUPERSEDED.json"
    if p.exists():
        raise SystemExit(f"{p} exists; write a new record by hand after review")
    doc = {"schema": "dataprep-superseded/1", "product": product,
           "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": "dataprep_staging/h2_heads_fill/versions_h2.py"} | doc
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


def main():
    write(C.PRODUCT_PRE, {
        "current": "v2",
        "current_note": "v2 is the version for use, for calibration targets only: the "
                        "observations (flags 0/1) and the daily-bridged months (flag 2), "
                        "released under the packaging rule declared in H2.fill_acceptance.v1 "
                        "with the gate H2.fresh.d (passed). No version of the complete fill "
                        "passed its gate: v1 and v3 are design-time evidence and after-the-fact "
                        "analysis only (the guard refuses initial_conditions and idw_field).",
        "superseded": [{
            "version": "v1", "superseded_by": "v3",
            "why": "registered v1 configuration (H2.fill_acceptance.v1): coverage failed in "
                   "N_net1 0.812, S24 0.939, E_lead66 0.844 (point window) and E_trail48 "
                   "0.960; RMSE passed in all 14 strata. v3 is the repair round (v2 sd "
                   "factors); its gate also failed.",
            "allowed_uses": ["design_time_evidence", "after_the_fact_analysis"]}],
        "not_superseded": [{
            "version": "v3", "role": "complete fill of the repair round "
            "(H2.fill_acceptance.v2): coverage failed in S01 0.983 and the RMSE criterion in "
            "N_net2 (0.783 vs clim+AR(1) 0.764); the other 12 strata passed",
            "allowed_uses": ["design_time_evidence", "after_the_fact_analysis"]}]})
    write(C.PRODUCT_FULL, {
        "current": "v1",
        "current_note": "repair-round configuration (H2.fill_acceptance.v2) fitted on "
                        "2012-2026; gate failed, so after_the_fact_analysis only (the guard "
                        "refuses statistical_layer_history). The v1 configuration was not "
                        "published for the full variant.",
        "superseded": []})


if __name__ == "__main__":
    main()
