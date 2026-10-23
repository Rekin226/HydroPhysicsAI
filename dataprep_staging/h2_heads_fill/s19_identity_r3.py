"""Row identity of the WP H2 repair-round-3 releases with the versions they re-package.

* heads/L2_filled_preorigin: the round-3 bridge release against v2 (the v1 configuration's
  bridge component): same (sid, month) rows and identical values, sds, flags, methods,
  sigma_obs_m and u2_weight; only ``product_version`` may differ.
* heads/L2_filled_full: the round-3 bridge release against the flag 0-2 rows of v1 (the
  repair configuration's complete full fill, whose bridge entries are identical).

Reads every version through the loader guard with ``after_the_fact_analysis`` where the version
allows it, else with its own allowed use. Writes ``H2_evidence/repair3/identity_r3.json``.

Usage: python -I s19_identity_r3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402

ROOT = C.REPO / "results/dataprep"
NUM = ("value", "sd", "flag", "sigma_obs_m", "u2_weight", "bridged_share", "l1_n_clean_days")
TXT = ("method", "gap_class", "source_ids")


def read(d: Path) -> tuple[pd.DataFrame, str]:
    """Earlier versions through the guard (after_the_fact_analysis); a round-3 version that
    does not allow that use is read as the packaging QA it is (the file just written, read
    directly as the manifest checker reads it), never under a modelling use."""
    m = json.loads((d / "MANIFEST.json").read_text())
    if "after_the_fact_analysis" in m["allowed_uses"]:
        p = guard.open_product(d, "after_the_fact_analysis")
        return p.read_table("fill.parquet"), "after_the_fact_analysis"
    return pd.read_parquet(d / "fill.parquet"), "packaging_qa_direct_read"


def newest(product: str, prefix: str) -> Path:
    hits = []
    for d in sorted((ROOT / product).glob("v*"), key=lambda p: int(p.name[1:])):
        v = d / "validation.json"
        if v.exists() and str(json.loads(v.read_text()).get("release", "")).startswith(prefix):
            hits.append(d)
    if not hits:
        raise SystemExit(f"no {prefix} release in {product}")
    return hits[-1]


def compare(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    k = ["sid", "month"]
    m = new.merge(old, on=k, how="outer", suffixes=("_new", "_old"), indicator=True)
    both = m[m._merge == "both"]
    out = {"rows_new": int(len(new)), "rows_old": int(len(old)),
           "only_new": int((m._merge == "left_only").sum()),
           "only_old": int((m._merge == "right_only").sum())}
    for c in NUM:
        a = both[f"{c}_new"].to_numpy(float)
        b = both[f"{c}_old"].to_numpy(float)
        nan_diff = int((np.isnan(a) != np.isnan(b)).sum())
        ok = ~np.isnan(a) & ~np.isnan(b)
        out[f"max_abs_diff_{c}"] = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
        out[f"nan_mismatch_{c}"] = nan_diff
    for c in TXT:
        out[f"mismatch_{c}"] = int((both[f"{c}_new"].astype(str)
                                    != both[f"{c}_old"].astype(str)).sum())
    out["identical"] = (out["only_new"] == 0 and out["only_old"] == 0
                        and all(out[f"max_abs_diff_{c}"] == 0 and out[f"nan_mismatch_{c}"] == 0
                                for c in NUM)
                        and all(out[f"mismatch_{c}"] == 0 for c in TXT))
    return out


def main():
    rec = {}
    pre_new = newest(C.PRODUCT_PRE, "r3_bridge_pre")
    pre_old = ROOT / C.PRODUCT_PRE / "v2"
    a, ua = read(pre_new)
    b, ub = read(pre_old)
    rec["preorigin"] = {"new": pre_new.name, "old": "v2", "uses_read": [ua, ub]} | compare(a, b)
    try:
        full_new = newest(C.PRODUCT_FULL, "r3_bridge_full")
    except SystemExit:
        full_new = None
    if full_new is not None:
        a, ua = read(full_new)
        b, ub = read(ROOT / C.PRODUCT_FULL / "v1")
        b = b[b.flag.isin([0, 1, 2])]
        rec["full"] = {"new": full_new.name, "old": "v1 (flags 0-2)",
                       "uses_read": [ua, ub]} | compare(a, b)
        fm = a[a.month.isin(["2026-02", "2026-03"]) & a.in_model.astype(bool)]
        rec["full"]["febmar_2026_model_wells"] = {
            "by_flag": {str(k): int(v) for k, v in fm.flag.value_counts().sort_index().items()},
            "bridged_sd_median": float(fm[fm.flag == 2].sd.median())}
    R3.jdump(rec, R3.EV_R3 / "identity_r3.json")
    print(json.dumps({k: (v["new"], v["identical"], v["rows_new"]) for k, v in rec.items()}))


if __name__ == "__main__":
    main()
