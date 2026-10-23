"""WP P1 repair round (r3): re-issue ``county_electricity/preorigin`` as v2 so that the
pre-origin product carries no number computed from data after 2022-12.

The verifier found that ``preorigin/v1/validation.json`` also carried ``T3_BV_LOYO``, the
leave-one-year-out backcast, computed on the full variant's TPC data for 2016-2025. A closer
read shows the same file also carried the full-window halves of ``T8_SYN_coverage`` and
``T9_MIS`` (synthetics fitted on 2016-01..2025-06). None fed a pre-origin gate, parameter or
threshold (the gate used T1_BV1, T8 pre-origin and T9 pre-origin), but the product opens for
``design_time_evidence``, a pre-origin use, so those numbers must not be in it. All three
already sit, unchanged, in ``full/v1/validation.json``.

v2 = v1 with:

* ``validation.json``: ``T1_BV1`` unchanged; ``T8_SYN_coverage`` and ``T9_MIS`` reduced to their
  ``preorigin`` halves (values unchanged); ``T3_BV_LOYO`` replaced by the same leave-one-year-out
  backcast on the pre-origin data only (held-out years 2016-2022; reported only, never an
  acceptance criterion, as in the plan); the removed items and their location in full/v1 are
  listed under ``r3_reissue``; the gate is unchanged (failed: T1_BV1, T8).
* long tables: ``product_version`` rewritten to ``county_electricity/preorigin/v2``; every
  other column is checked equal to v1 before writing.
* every other file byte-identical to v1.

Usage: python -I r3_preorigin_v2.py <evidence_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402

import data as dmod  # noqa: E402
import lib  # noqa: E402

PRODUCT = "county_electricity/preorigin"
V1 = REPO / "results/dataprep" / PRODUCT / "v1"
FULL_V1 = REPO / "results/dataprep/county_electricity/full/v1"
TABLES = ["changhua_candidates.parquet", "tpc_county_sector_monthly.parquet",
          "yunlin_census_class_monthly.parquet"]
COPY = ["changhua_candidates_wide.parquet", "census_monthly_info.json", "decomposition.json",
        "misspecification.json", "phi_in.json"]
PRE_YEARS = range(2016, 2023)


def loyo_backcast(d: pd.DataFrame, years) -> dict:
    """``validate.loyo_backcast`` with the held-out years as an argument (same arithmetic)."""
    out = {}
    for c, col in (("CH", "CH_ag"), ("YL", "YL_ag")):
        sh = (d[col] / d["nat_ag"]).dropna()
        errs = {}
        for yr in years:
            tr = sh[sh.index.year != yr]
            s = tr.groupby(tr.index.month).mean()
            te = d.loc[str(yr)].dropna(subset=[col, "nat_ag"])
            pred = te["nat_ag"] * s.reindex(te.index.month).to_numpy()
            errs[yr] = float(100 * (pred.sum() / te[col].sum() - 1))
        out[c] = {"annual_err_pct": errs, "rms_pct": lib.rms(list(errs.values()))}
    return out


def main(evidence: str) -> int:
    evd = Path(evidence)
    evd.mkdir(parents=True, exist_ok=True)
    m1 = json.loads((V1 / "MANIFEST.json").read_text(encoding="utf-8"))
    v1val = json.loads((V1 / "validation.json").read_text(encoding="utf-8"))
    fval = json.loads((FULL_V1 / "validation.json").read_text(encoding="utf-8"))
    # the removed items must already sit, identical, in full/v1
    moved = {"T3_BV_LOYO": v1val["tests"]["T3_BV_LOYO"] == fval["tests"]["T3_BV_LOYO"],
             "T8_SYN_coverage.full": v1val["tests"]["T8_SYN_coverage"]["full"]
             == fval["tests"]["T8_SYN_coverage"]["full"],
             "T9_MIS.full": v1val["tests"]["T9_MIS"]["full"] == fval["tests"]["T9_MIS"]["full"]}
    if not all(moved.values()):
        raise SystemExit(f"removed items are not identical in full/v1: {moved}")
    D = dmod.load("preorigin")
    assert D["d"].index.max() <= pd.Timestamp("2022-12-01")
    t3 = loyo_backcast(D["d"], PRE_YEARS)

    d = manifest.new_version_dir(PRODUCT)
    pv = f"{PRODUCT}/{d.name}"
    for f in COPY:
        shutil.copy2(V1 / f, d / f)
    for f in TABLES:
        t = pd.read_parquet(V1 / f)
        if set(t["product_version"].unique()) != {f"{PRODUCT}/v1"}:
            raise SystemExit(f"{f}: unexpected product_version")
        t2 = t.assign(product_version=pv)
        pd.testing.assert_frame_equal(t.drop(columns="product_version"),
                                      t2.drop(columns="product_version"))
        t2.to_parquet(d / f, index=False)
    tests = {"T1_BV1": v1val["tests"]["T1_BV1"],
             "T3_BV_LOYO": t3 | {
                 "held_out_years": [min(PRE_YEARS), max(PRE_YEARS)],
                 "status": "reported only (plan: LOYO measures interpolation and is not used "
                           "for acceptance); r3: recomputed on data <= 2022-12 only"},
             "T8_SYN_coverage": {"preorigin": v1val["tests"]["T8_SYN_coverage"]["preorigin"]},
             "T9_MIS": {"preorigin": v1val["tests"]["T9_MIS"]["preorigin"]}}
    val = {"product": pv, "prereg": v1val["prereg"], "prereg_sha256": v1val["prereg_sha256"],
           "gate": v1val["gate"], "tests": tests,
           "note": "every number in this file uses data <= 2022-12 only; the gate uses T1_BV1, "
                   "T8 (pre-origin window) and T9 (pre-origin window)",
           "r3_reissue": {
               "reissue_of": f"{PRODUCT}/v1",
               "removed_items": {
                   "T3_BV_LOYO (held-out years 2016-2025, full-variant TPC data)":
                       "county_electricity/full/v1/validation.json tests.T3_BV_LOYO",
                   "T8_SYN_coverage.full (synthetic on 2016-01..2025-06)":
                       "county_electricity/full/v1/validation.json tests.T8_SYN_coverage.full",
                   "T9_MIS.full (synthetic on 2016-01..2025-06)":
                       "county_electricity/full/v1/validation.json tests.T9_MIS.full"},
               "removed_items_identical_in_full_v1": moved,
               "reason": "verifier minor issue (repair round): a pre-origin product that opens "
                         "for design_time_evidence must not carry post-origin numbers; none of "
                         "them fed a gate, parameter or threshold of v1",
               "v1_gate_unchanged": True}}
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    params = dict(m1["parameters"])
    params["r3_reissue"] = {
        "reissue_of": f"{PRODUCT}/v1", "v1_manifest_sha256": sha256_file(V1 / "MANIFEST.json"),
        "changes": ["validation.json: post-origin items removed (T3 2016-2025, T8 full, T9 "
                    "full; all still in full/v1); T3 recomputed on 2016-2022 (reported only)",
                    "long tables: product_version column rewritten to v2 (values unchanged)",
                    "other files byte-identical to v1"],
        "status": "superseded as a Changhua forcing source by WP C0 (plan revision "
                  "2026-10-12); gate failed as in v1 (T1_BV1, T8)",
        "flags": "backcast 2012-15 and every Changhua estimator row keep flag 5 (TPC-derived "
                 "series, plan 2.4); see the P1 report, repair round"}
    inputs = [i["path"] for i in m1["inputs"]] + [
        rel_to_repo(V1 / "MANIFEST.json")] + [rel_to_repo(V1 / f) for f in COPY + TABLES
                                             + ["validation.json"]]
    code = [c["path"] for c in m1["code"]] + [rel_to_repo(Path(__file__).resolve())]
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L2", inputs=inputs, code=code, parameters=params,
        fit_window_end=m1["fit_window_end"], validation_report="validation.json",
        gate={"passed": False, "criteria": m1["gate"]["criteria"]
              + " (unchanged from v1; r3 re-issue)"},
        allowed_uses=m1["allowed_uses"], forbidden_uses=m1["forbidden_uses"],
        personal_data=m1["personal_data"], exemptions=m1["exemptions"])
    p = manifest.write_manifest(d, m)
    ev = {"product_dir": rel_to_repo(d), "manifest": rel_to_repo(p),
          "T3_BV_LOYO_preorigin": t3, "removed_items_identical_in_full_v1": moved,
          "tables_rewritten": TABLES, "files_copied": COPY}
    (evd / "preorigin_v2_build.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1,
                                                            default=float), encoding="utf-8")
    print(json.dumps(ev, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
