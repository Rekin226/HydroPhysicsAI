"""Diagnosis evidence for the L3 repair (release_20261012b); reads only, changes nothing.

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release_b.diag --json OUT

Evidence for the three verifier majors of release_20261012 and the pre-origin station minor:

* every printed pile value in the pre-origin slice of ``deep_benchmarks_printed/v2`` whose
  report was published after the origin, by the agency listing's publication month
  (發布年月, recorded per file in the L0 SOURCE.json of the 007001 series), for the primary
  printing and for the second printing (``reprint_*``);
* the same count by the report's caption month (the verifier's reading), for comparison;
* the stations of H1a's pre-origin station table established after 2022-12;
* the gate records of the three disputed upstreams (S2 printed, W1 pre-origin, H3).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import guard
from common.paths import DATAPREP, ORIGIN, REPO, rel_to_repo

PILE_SOURCE = "data_fetch_api/new_measurements/wra_cy_monthly_reports_007001/SOURCE.json"


def roc_to_ym(s: str) -> str:
    """'112/1' (ROC year / month) -> '2023-01'."""
    y, m = s.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}"


def publication_by_file(source_json: Path) -> dict[str, dict]:
    """{file: {published, listing_rows, last_modified}}; a file linked from several listing
    rows takes the LATEST listed month (conservative)."""
    src = json.loads(source_json.read_text(encoding="utf-8"))
    out = {}
    for f in src["files"]:
        if "published_roc" not in f:          # listing pages, not reports
            continue
        months = [roc_to_ym(f["published_roc"])]
        note = f.get("note", "")
        if "published_roc" in note:
            months += [roc_to_ym(x) for x in re.findall(r"'published_roc': '([0-9]+/[0-9]+)'",
                                                         note)]
        out[f["file"]] = {"published": max(months), "listing_months": sorted(set(months)),
                          "last_modified": f.get("last_modified")}
    return out


def piles(pub: dict) -> dict:
    p = guard.open_product(DATAPREP / "deep_benchmarks_printed/v2", "validator")
    df = p.read_table("printed_monthly.parquet")
    pre = df[df["month"].astype(str) <= ORIGIN].copy()
    missing = sorted(set(pre["report"]) - set(pub))
    pre["published"] = pre["report"].map(lambda r: pub.get(r, {}).get("published"))
    rp = pre["reprint_report"].fillna("")
    pre["reprint_published"] = [pub[r]["published"] if r else None for r in rp]
    late_primary = pre[pre["published"] > ORIGIN]
    late_caption = pre[pre["vintage_month"].astype(str) > ORIGIN]
    late_reprint = pre[pre["reprint_published"].fillna("") > ORIGIN]
    late_reprint_caption = pre[pre["reprint_vintage_month"].fillna("").astype(str) > ORIGIN]
    by_report = (late_primary.groupby(["report", "published"]).agg(
        rows=("month", "size"), months=("month", lambda s: sorted(set(map(str, s)))))
        .reset_index().to_dict("records"))
    lm = sorted({(r, pub[r]["published"], pub[r]["last_modified"]) for r in pre["report"]})
    return {
        "rows_in_preorigin_slice_validator_read": int(len(pre)),
        "reports_without_publication_record": missing,
        "primary_published_after_origin": {
            "rows": int(len(late_primary)),
            "flags": {str(k): int(v) for k, v in late_primary["flag"].value_counts().items()},
            "by_report": by_report},
        "primary_caption_after_origin_verifier_reading": int(len(late_caption)),
        "primary_published_after_origin_but_caption_not": sorted(
            f"{r.sensor_id}:{r.month}:{r.report}" for r in
            late_primary[late_primary["vintage_month"].astype(str) <= ORIGIN].itertuples()),
        "reprint_published_after_origin_rows": int(len(late_reprint)),
        "reprint_caption_after_origin_rows_verifier_reading": int(len(late_reprint_caption)),
        "rows_left_after_drop": int(len(pre) - len(late_primary)),
        "n_printings_counting_post_origin_reprints": int(
            ((pre["published"] <= ORIGIN) & (pre["reprint_published"].fillna("") > ORIGIN))
            .sum()),
        "server_last_modified_of_preorigin_reports": [
            {"report": r, "published_listing": p_, "last_modified": m} for r, p_, m in lm],
    }


def stations() -> dict:
    p = guard.open_product(DATAPREP / "heads/L1_clean/v3", "model_design")
    st = p.read_table("station_meta_preorigin.parquet")
    est = st["EstablishDate"].astype("int64")
    late = st[est > 20221231]
    h = guard.open_product(DATAPREP / "heads/L1_clean/v3", "design_time_evidence",
                           window_end=ORIGIN).read_table("heads_monthly.parquet")
    rows_late = int(h["sid"].isin(set(late["sid"])).sum())
    return {"stations": int(len(st)), "established_after_origin": int(len(late)),
            "establish_dates": sorted(int(x) for x in late["EstablishDate"]),
            "preorigin_head_rows_of_those_stations": rows_late}


def gates() -> dict:
    out = {}
    for d in ("deep_benchmarks_printed/v2", "deep_benchmarks/v2",
              "surface_water_v4/preorigin/v2", "surface_water_v4/full/v4",
              "heads/recovered_wells/v3"):
        m = json.loads((DATAPREP / d / "MANIFEST.json").read_text(encoding="utf-8"))
        g = m["gate"]
        out[d] = {k: g.get(k) for k in ("passed", "acceptance_passed", "release_rule_passed",
                                        "plan_acceptance_passed", "criteria")}
        out[d]["allowed_uses"] = m["allowed_uses"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    pub = publication_by_file(REPO / PILE_SOURCE)
    res = {"origin": ORIGIN, "pile_publication_source": PILE_SOURCE,
           "piles": piles(pub), "stations": stations(), "gates": gates()}
    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    pl = res["piles"]
    print(f"primary published after origin {pl['primary_published_after_origin']['rows']} "
          f"(caption reading {pl['primary_caption_after_origin_verifier_reading']}); reprints "
          f"{pl['reprint_published_after_origin_rows']} (caption reading "
          f"{pl['reprint_caption_after_origin_rows_verifier_reading']}); stations after origin "
          f"{res['stations']['established_after_origin']} -> {rel_to_repo(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
