"""Build the data product ``heads/L1_yearbook`` from the extracted yearbook tables.

Source: public WRA hydrological yearbooks, groundwater volume (one page per well and year).
The product holds the printed monthly means of the fan's monitoring wells for 2001-2011 and,
for validation against held data, 2013, 2015 and 2020. Nothing is filled. User ruling R12.1
(2026-11-12): uses statistical_layer_history, model_design, validator, with the listed
defects disclosed.

Run from the repository root: ``python dataprep_staging/r12_yearbook/build_product.py``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import manifest  # noqa: E402
from common.paths import L0_NEW  # noqa: E402
from common.uses import USES  # noqa: E402

HERE = Path(__file__).resolve().parent
L0 = L0_NEW / "wra_hydro_yearbook_gw"
ALLOWED = ["statistical_layer_history", "model_design", "validator"]
KEEP = ("heads_monthly_yearbook.parquet", "match_table.csv", "validation.json",
        "validation_per_well.csv", "coverage_by_year.csv", "year_boundary_continuity.json",
        "our_wells_never_in_parsed_yearbooks.csv", "yearbook_urls.csv", "downloads.csv")
DEFECTS = [
    "no groundwater-level volume exists before 2001",
    "the 2006 volume holds only 3 Yunlin wells",
    "the 2009 yearbook marks most wells incomplete from May to December (parse_flag 1)",
    "38 wells have no January 2011 mean",
    "printed well-top elevations change between years; cross-year datum before 2012 is not "
    "independently verified",
    "years 2012, 2014, 2016-2019 and 2021-2024 were not downloaded",
    "manual-well pages (4-digit station ids) were not parsed",
]


def main() -> None:
    out = manifest.new_version_dir("heads/L1_yearbook")
    for f in KEEP:
        shutil.copy2(L0 / "out" / f, out / f)
    h = pd.read_parquet(out / "heads_monthly_yearbook.parquet")
    v = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    ok = h[h.value_m.notna()]
    dup = int(ok.duplicated(["sid", "year", "month"]).sum())
    gate = {
        "passed": bool(dup == 0 and v["median_rmse_m"] <= 0.02
                       and v["share_wells_abs_bias_le_0p05"] >= 0.9
                       and not v["wells_constant_offset"]),
        "criteria": "unique (sid, year, month) rows; on overlap years 2013, 2015, 2020 "
                    "against L1_clean v3 flags 0-1: median RMSE <= 0.02 m, at least 90 % of "
                    "wells within 0.05 m bias, no well with a constant datum offset",
        "measured": {"duplicates": dup, "median_rmse_m": v["median_rmse_m"],
                     "share_wells_abs_bias_le_0p05": v["share_wells_abs_bias_le_0p05"],
                     "wells_constant_offset": len(v["wells_constant_offset"])},
    }
    raw = sorted((L0 / "raw").glob("gw_*.pdf"))
    code = [HERE / n for n in ("crawl_index.py", "download.py", "parse_yearbook.py",
                               "build_outputs.py", "build_product.py")]
    m = manifest.build_manifest(
        out, product="heads/L1_yearbook", layer="L1", inputs=raw, code=code,
        parameters={
            "years": sorted(int(y) for y in h.year.unique()),
            "wells": int(ok.sid.nunique()), "rows": int(len(h)),
            "unit": "metres, elevation as printed by the yearbook of that year",
            "parse_flag": {"0": "clean", "1": "yearbook '*' (incomplete)",
                           "2": "yearbook '@' (gap-filled days)",
                           "3": "printed mean inconsistent with daily values",
                           "9": "not printed (value missing)"},
            "defects": DEFECTS,
            "ruling": "results/dataprep/rulings/user_rulings_2026-11-12.json R12.1",
            "code_note": "the extraction scripts were run once outside the repository on "
                         "2026-11-12; the copies hashed here differ only in configurable "
                         "paths",
        },
        fit_window_end="2020-12", validation_report="validation.json", gate=gate,
        allowed_uses=ALLOWED, forbidden_uses=sorted(set(USES) - set(ALLOWED)),
        personal_data="none (station ids and names of public monitoring wells)",
        extra={"local_only": "never committed: derived from agency publications"})
    errs = manifest.validate_manifest(m, out)
    if errs:
        shutil.rmtree(out)
        raise SystemExit("manifest invalid: " + "; ".join(errs))
    p = manifest.write_manifest(out, m)
    print(json.dumps({"product_dir": str(out), "manifest": str(p), "gate": gate}, indent=1))


if __name__ == "__main__":
    main()
