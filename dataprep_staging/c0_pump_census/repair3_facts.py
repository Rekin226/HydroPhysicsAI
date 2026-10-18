"""WP C0 repair round (resume 2026-10-14): descriptive L0 facts that the verifier's minor
findings asked the report to state correctly. After the fact, read-only, not a registered test:

1. identical Dec_y / Jan_y+1 pairs of M1-coded meters (phase B, the Dec-Jan pair of the TPC
   billing calendar), y = 2012..2024, per county: fraction equal within ``lib.PAIR_ATOL``
   among pairs with both readings finite and not both zero (the verifier quoted 3.5 % Yunlin /
   6.2 % Changhua for 2021-12/2022-01 against >= 99.6 % in other years);
2. census rows without TWD97 coordinates per county (no cell; dropped from the cell arrays,
   the model's convention), with their share of the county's 2012-2022 raw L0 kWh;
3. billing-code agreement of the detected phase, as recorded by the v3 builder
   (``build_info.json`` of preorigin/v3 and full/v3).

Output: reports/C0_evidence/repair3/report_facts.json.

``python -I repair3_facts.py log`` then ``python -I repair3_facts.py run``
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from harness import prereg3  # noqa: E402

OUT = lib.EVIDENCE / "repair3" / "report_facts.json"
UNIT = "C0r3:report_facts_descriptive"
YEARS = range(2012, 2025)
W = slice(lib.tidx("2012-01"), lib.tidx("2022-12") + 1)


def log_look() -> None:
    prereg3.record_look(
        [UNIT], "tpc_census_kwh:2012-01..2025-01", lib.WP, "diagnose",
        members={UNIT: {"none": "C0 electricity-forcing construction object; contains no "
                                "registered well, benchmark or model cell of the "
                                "validation-unit registries",
                        "description": "report facts: M1 Dec/Jan pair equality by year, census "
                                       "rows without coordinates, billing-code agreement"}},
        note="[C0-repair3-facts] descriptive L0 facts for the report (verifier minor findings): "
             "M1 Dec_y/Jan_y+1 identical-pair fractions 2012-2024, rows without coordinates, "
             "phase agreement from the v3 build_info; after the fact; nothing feeds back.")
    print("logged [C0-repair3-facts]")


def county(tag: str, grid) -> dict:
    df = lib.harmonise(tag, lib.read_census_l0(tag), grid)
    _, L = lib.read_kwh_l0(tag, df["sid"].to_numpy(), len(df))
    first = df["meter_first_pump"].to_numpy(bool)
    X = L[first]
    code = lib.phase_from_code(df.loc[first, "billing_code"].to_numpy())
    b = code == "B"
    pairs = {}
    for y in YEARS:
        d, j = X[b, lib.tidx(f"{y}-12")], X[b, lib.tidx(f"{y + 1}-01")]
        valid = np.isfinite(d) & np.isfinite(j) & ((d > 0) | (j > 0))
        eq = valid & np.isclose(d, j, rtol=0, atol=lib.PAIR_ATOL)
        pairs[f"{y}-12/{y + 1}-01"] = {"valid": int(valid.sum()), "equal": int(eq.sum()),
                                       "frac_equal": float(eq.sum() / max(valid.sum(), 1))}
    xy = df[["x_twd97", "y_twd97"]].to_numpy(dtype="float64")
    nocoord = ~np.isfinite(xy).all(1)
    tot = float(np.nansum(L[:, W]))
    return {"m1_meters": int(b.sum()), "m1_dec_jan_pair_equality": pairs,
            "census_rows": int(len(df)), "rows_without_coordinates": int(nocoord.sum()),
            "rows_without_coordinates_raw_kwh_share_2012_2022":
                float(np.nansum(L[nocoord][:, W]) / tot) if tot else None}


def run() -> dict:
    t0 = time.time()
    grid = lib.load_grid()
    res = {"time_utc": prereg3._now(),
           "note": "descriptive, after the fact (uses L0 to 2025-01); not a registered test",
           "per_county": {t: county(t, grid) for t in lib.COUNTIES}}
    agree = {}
    for v in ("preorigin", "full"):
        bi = lib.load_json(lib.PRODUCT_ROOT / v / "v3" / "build_info.json")
        agree[v] = {t: {k: bi["counties"][t].get(k) for k in ("phase_agree_frac_bimonthly",
                                                            "phase_agree_frac_monthly")}
                    for t in lib.COUNTIES}
    res["phase_agreement_v3_build_info"] = agree
    res["seconds"] = round(time.time() - t0, 1)
    lib.save_json(res, OUT)
    return res


def main(argv: list[str]) -> int:
    if argv == ["log"]:
        log_look()
        return 0
    if argv == ["run"]:
        r = run()
        for t, v in r["per_county"].items():
            print(t, {k: v[k] for k in v if k != "m1_dec_jan_pair_equality"})
            print("  ", {k: round(x["frac_equal"], 4) for k, x in
                         v["m1_dec_jan_pair_equality"].items()})
        print(r["phase_agreement_v3_build_info"])
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
