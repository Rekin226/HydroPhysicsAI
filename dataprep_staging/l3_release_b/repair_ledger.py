"""Ledger rows for the numbers quoted in reports/L3.repair.md (plan 4.1, rule U6).

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release_b.repair_ledger

Appends one row per source per quoted number to ``results/dataprep/ledger/rows/L3.csv``
(exact duplicates skipped, conflicts refused), then reassembles
``results/dataprep/ledger/data_roles.csv`` from every WP fragment.
"""
from __future__ import annotations

from common import ledger

from l3_release_b import config as C

EV = "results/dataprep/reports/L3_evidence/repair_b"
REL = f"{C.RELEASE_DIR}/{C.RELEASE}"
PV = f"model_ready/{C.RELEASE}"
PILES_UP = "results/dataprep/deep_benchmarks_printed/v2"
SRC = C.PILE_PUBLICATION_SOURCE

NUMBERS = {
    # number id suffix: (sources, note)
    "piles_published_after_origin": (
        [f"{EV}/diag.json", f"{EV}/regress_final.json", SRC, PILES_UP],
        "35 pre-origin primary pile values published after 2022-12 (agency listing month), "
        "dropped from preorigin/piles_printed"),
    "piles_verifier_named": (
        [f"{EV}/diag.json", SRC, PILES_UP],
        "21 of them known only from 2023 re-printings (data months 2022-02, 2022-03, 2022-10)"),
    "piles_dec2022_report": (
        [f"{EV}/diag.json", SRC, PILES_UP],
        "14 printed only in the December-2022 report, published 2023-01 (data months 2021-12, "
        "2022-12)"),
    "piles_reprints_post_origin": (
        [f"{EV}/diag.json", f"{EV}/regress_final.json", SRC, PILES_UP],
        "49 second printings published after 2022-12 removed (reprint_* null)"),
    "piles_rows_kept": (
        [f"{EV}/regress_final.json", f"{REL}/preorigin/piles_printed/v1"],
        "231 of 266 pre-origin pile rows kept; values and flags identical to release_20261012"),
    "piles_hidden_second_printings": (
        [f"{REL}/preorigin/piles_printed/v1", f"{PILES_UP}/table1_rows.parquet", SRC],
        "18 rows with n_printings 2 but no reprint columns upstream; their second printings "
        "were published 2020-10..2021-11"),
    "piles_lastmodified_sensitivity": (
        [f"{REL}/preorigin/piles_printed/v1", SRC],
        "server Last-Modified reading would remove 53 more rows (data months 2022-04..2022-11; "
        "178 left, ending 2021-11) and 45 more second printings; not applied"),
    "stations_post_origin": (
        [f"{EV}/diag.json", f"{EV}/regress_final.json", "results/dataprep/heads/L1_clean/v3"],
        "13 stations established 2023-07-29..2024-08-01 dropped from the pre-origin station "
        "table (344 -> 331; 0 head rows)"),
    "rebuild_files": (
        [f"{EV}/compare.json", f"{REL}/REBUILD_CHECK.json"],
        "77/77 files identical over builds A and B"),
    "regress_vs_release_20261012": (
        [f"{EV}/regress_final.json", f"{C.RELEASE_DIR}/{C.SUPERSEDES}"],
        "30 files identical, 6 re-packed npz equal except the release name, 2 intended "
        "changes; release_20261012 tree unchanged"),
    "guard_probes": (
        [f"{EV}/check_final.json"],
        "306 guard probes over 18 manifests, 0 mismatches"),
    "inventory": (
        [f"{EV}/inventory.json"],
        "13 upstream versions included, 81 product versions excluded, 0 unclassified"),
    "code_pinned_upstreams": (
        [f"{REL}/RELEASE.json", "results/dataprep/pump_electricity/preorigin/v3",
         "results/dataprep/pump_electricity/full/v3",
         "results/dataprep/heads/L2_filled_preorigin/v4",
         "results/dataprep/heads/L2_filled_full/v2",
         "results/dataprep/reports/H2_evidence/code_as_registered/015c1980819a/config.py"],
        "4 upstreams whose moved-on code files were verified from git 709f660 or the archived "
        "registered bytes (C0 1 file each, H2 3 files each)"),
    "checker": (
        [f"{EV}/common_check_release_20261012b.json",
         f"{EV}/common_check_release_20261012.json"],
        "A0 checker: 18 manifests, 0 failures, in each release"),
    "excluded_items": (
        [f"{REL}/RELEASE.json"],
        "57 exclusion entries covering 84 product versions or WP-level items"),
}


def rows() -> list[dict]:
    out = []
    for suffix, (sources, note) in NUMBERS.items():
        for s in sources:
            out.append({"number_id": f"L3.{C.RELEASE}.repair.{suffix}", "source": s,
                        "role": "input", "product_version": PV, "note": note})
    return out


def main() -> int:
    p = ledger.append_rows(C.LEDGER_WP, rows())
    ids = [f"L3.{C.RELEASE}.repair.{k}" for k in NUMBERS]
    print(f"appended to {p}; unledgered {ledger.unledgered(ids)}")
    out, n = ledger.assemble()
    print(f"assembled {out} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
