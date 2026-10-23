"""Repair round (r2) of WP S1a: rings_bridged/v2 against v1, and the registered code.

* every output table of v2 compared with v1: identical except the declared amendment
  (Tuku's 2022 total path, its sd, flag and method; the product_version column);
* the files hashed by v1's manifest and by the two v1 pre-registrations are unchanged (the
  repair round adds modules only);
* v2's validation.json carries v1's gate numbers unchanged.

    tools/cpu_slot.sh python -m s1_rings.r2_checks       (from dataprep_staging/)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import from_repo  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

V1 = K.REPO / "results" / "dataprep" / "rings_bridged" / "v1"
V2 = K.REPO / "results" / "dataprep" / "rings_bridged" / "v2"
PREREG = K.REPO / "results" / "dataprep" / "preregistration"
TUKU = "土庫國中"
KEYS = ["site", "series", "band", "month"]


def compare_long(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    a = a.set_index(KEYS).sort_index()
    b = b.set_index(KEYS).sort_index()
    out = {"same_index": bool(a.index.equals(b.index)), "n_rows": int(len(a))}
    if not out["same_index"]:
        return out
    diff = pd.Series(False, index=a.index)
    for c in a.columns:
        if c == "product_version":
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x):
            d = ~(np.isclose(x.astype(float), y.astype(float), atol=1e-9, equal_nan=True))
        else:
            d = ~((x == y) | (x.isna() & y.isna()))
        diff |= d
    rows = diff[diff].reset_index()
    out["n_rows_changed"] = int(diff.sum())
    out["changed_sites"] = sorted(rows["site"].unique().tolist()) if len(rows) else []
    out["changed_bands"] = sorted(rows["band"].unique().tolist()) if len(rows) else []
    out["changed_months"] = sorted(rows["month"].unique().tolist()) if len(rows) else []
    out["changed_series"] = sorted(rows["series"].unique().tolist()) if len(rows) else []
    t = a.loc[diff].join(b.loc[diff], lsuffix="_v1", rsuffix="_v2")
    out["flag_changes"] = (t.groupby(["flag_v1", "flag_v2"]).size().rename("n").reset_index()
                           .to_dict(orient="records")) if len(t) else []
    if len(t):
        dv = (t["value_v2"] - t["value_v1"]).abs()
        out["max_abs_value_change_mm"] = float(dv.max())
        out["sd_v1_mm_range"] = [float(t["sd_v1"].min()), float(t["sd_v1"].max())]
        out["sd_v2_mm_range"] = [float(t["sd_v2"].min()), float(t["sd_v2"].max())]
    return out


def main() -> None:
    res = {}
    # ring_corrections and the per-site corrected positions
    c1 = pd.read_csv(V1 / "ring_corrections.csv")
    c2 = pd.read_csv(V2 / "ring_corrections.csv")
    res["ring_corrections_identical"] = bool(c1.equals(c2))
    same = []
    for p in sorted(V1.glob("rings_corrected_*.parquet")):
        q = V2 / p.name
        same.append(q.exists() and pd.read_parquet(p).equals(pd.read_parquet(q)))
    res["rings_corrected_files"] = {"n": len(same), "identical": int(sum(same))}
    res["total_csv"] = compare_long(pd.read_csv(V1 / "ring_total_monthly_bridged.csv"),
                                    pd.read_csv(V2 / "ring_total_monthly_bridged.csv"))
    res["bands_parquet"] = compare_long(pd.read_parquet(V1 / "ring_bands_monthly_bridged.parquet"),
                                        pd.read_parquet(V2 / "ring_bands_monthly_bridged.parquet"))
    ok_change = (res["total_csv"].get("changed_sites") in ([], [TUKU])
                 and set(res["total_csv"].get("changed_bands", [])) <= {"total"}
                 and all("2022-01" <= m <= "2023-01" for m in res["total_csv"]["changed_months"]))
    res["only_declared_change"] = bool(res["ring_corrections_identical"]
                                       and res["rings_corrected_files"]["n"]
                                       == res["rings_corrected_files"]["identical"]
                                       and ok_change)
    # gate numbers unchanged
    v1 = json.loads((V1 / "validation.json").read_text())
    v2 = json.loads((V2 / "validation.json").read_text())
    keys = ("pooled_z", "power_25pct", "n_inside_rate_bracket")
    res["gate_numbers_identical"] = bool(
        v1["gate"]["passed"] == v2["gate"]["passed"]
        and all(v1["a_pooled_standardized_difference"][k] == v2["a_pooled_standardized_difference"][k]
                for k in keys)
        and v1["b_pile_implied_scale"]["sites"] == v2["b_pile_implied_scale"]["sites"]
        and v1["residual_offset"]["sites"] == v2["residual_offset"]["sites"]
        and v1["e_synthetic"] == v2["e_synthetic"])
    # registered code unchanged
    m1 = json.loads((V1 / "MANIFEST.json").read_text())
    bad = [c["path"] for c in m1["code"] if sha256_file(from_repo(c["path"])) != c["sha256"]]
    for pid in ("S1a.rings_bridged_rules.v1", "S1a.e_synthetic_mixed.v1"):
        doc = json.loads((PREREG / f"{pid}.json").read_text())
        for p, h in doc.get("code_at_registration", {}).items():
            if sha256_file(from_repo(p)) != h:
                bad.append(f"{pid}:{p}")
    res["registered_code_changed"] = bad
    res["ok"] = bool(res["only_declared_change"] and res["gate_numbers_identical"] and not bad)
    out = K.EVIDENCE / "repair" / "v1_v2_compare.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
