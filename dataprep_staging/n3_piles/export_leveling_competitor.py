"""The level test's competitor: the pre-origin leveling-cell rate at each pile cell (lead runs).

User ruling 2026-11-01 (fourth set): the pre-origin leveling-cell rate replaces printed-pile
persistence (2026-10-18 ruling (b): pre-origin printed piles are validator and
after_the_fact_analysis only). Source: the column's calibration panel (the WRA leveling
network, ls_cache ls-wra-lsp-obs__choushui_panel.parquet, read by twin.leveling.load_panel)
with surveys before 2022-01-01 only (window_end 2021-12), the calibration's own site screen
(twin.column_step.leveling_rates: >= 5 surveys, survey-to-survey rate <= 0.5 m/yr, >= 2
years spanned, least-squares rate, positive = sinking).

Declared rule (fixed before any rate is computed):
1. the benchmarks whose nearest active cell is the pile cell, within one cell width (the
   calibration's placement rule, calibrate_coupled.leveling_rows);
2. if none, the benchmarks within 1.5 km of the pile;
3. if none, no competitor: the level test is not evaluable at that pile.
Rate = the median of the selected benchmarks' 2016-01..2021-12 rates (cm/yr); the 12-month
competitor is the rate x 1 yr, the 24-month one x 2 yr. Leveling is ground-surface (total
depth), like the column; the pile is anchor-to-surface (the same depth caveat as M).

The output is a small product with a MANIFEST that grants ``baseline`` (pre-origin,
fit_window_end 2021-12) so that the scorer can open it through common.guard; it is hashed
into N3.piles_prospective.v1. This reads pre-origin observations (spent calibration data,
'baseline' use); it computes no score and is run by the lead before registration.

The read is logged as a look (prereg3.record_look, purpose 'baseline_export', unit
LOOK_UNIT, data scope 'leveling:2016-01..2021-12', members = every benchmark the screen
kept) before the product is written, so it sits in the chronology that N3 hashes; the panel's
sha256 is recorded in the look note, the MANIFEST inputs and its parameters.

Usage: python -I dataprep_staging/n3_piles/export_leveling_competitor.py
           --mapping <dir>/mapping.json [--data-dir <dir holding ls_cache/>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n3lib as L  # noqa: E402

PRODUCT = "n3_leveling_cell_rate"
LOOK_UNIT = "N3:leveling_cell_rate_2016-01_2021-12"
LOOK_SCOPE = "leveling:2016-01..2021-12"
LOOK_PURPOSE = "baseline_export"
T0, T1 = "2016-01-01", "2022-01-01"
WINDOW_END = "2021-12"
FALLBACK_RADIUS_M = 1500.0
PANEL_REL = os.path.join("ls_cache", "ls-wra-lsp-obs__choushui_panel.parquet")
ALLOWED = ["baseline", "after_the_fact_analysis"]
FORBIDDEN = ["selection", "model_design", "prior", "prior_width_input", "initial_conditions",
             "idw_field", "forcing_frozen", "forcing_recorded_arm", "calibration_target",
             "datum_fit", "statistical_layer_history", "prospective_scorer"]


def select(rates: pd.DataFrame, pile: dict, grid: dict) -> tuple[pd.DataFrame, str]:
    """The declared benchmark selection for one pile (rates: sid, x, y, rate_cm_yr)."""
    from hydrophysics.twin.mechanism import ring_cells

    if rates.empty:
        return rates, "none"
    cells = ring_cells(grid, rates[["x", "y"]].to_numpy(float))
    hit = rates[cells == pile["site"]["cell"]]
    if len(hit):
        return hit, "pile_cell"
    d = np.hypot(rates["x"] - pile["site"]["x"], rates["y"] - pile["site"]["y"])
    hit = rates[d <= FALLBACK_RADIUS_M]
    return (hit, "within_1.5km") if len(hit) else (hit, "none")


def competitor_table(rates: pd.DataFrame, mapping: dict, grid: dict) -> pd.DataFrame:
    rows = []
    for key in L.PRIMARY:
        sel, rule = select(rates, mapping["piles"][key], grid)
        rate = float(sel["rate_cm_yr"].median()) if len(sel) else np.nan
        rows.append({"pile_key": key, "pile": L.PILES[key]["pile"],
                     "cell": mapping["piles"][key]["site"]["cell"], "rule": rule,
                     "n_benchmarks": int(len(sel)),
                     "benchmarks": ";".join(sorted(map(str, sel["sid"]))),
                     "rate_cm_yr": rate, "competitor_12m_cm": rate * 1.0,
                     "competitor_24m_cm": rate * 2.0})
    return pd.DataFrame(rows)


def log_look(rates: pd.DataFrame, panel_sha: str | None, prereg_dir=None) -> dict:
    """The pre-origin leveling read as a look (before the product is written)."""
    from harness import prereg3

    ids = sorted(map(str, rates["sid"])) if len(rates) else []
    members = {LOOK_UNIT: {
        "description": "WRA leveling panel (ls_cache choushui panel), surveys 2016-01..2021-12 "
                       "read by twin.column_step.leveling_rates for N3's pre-origin "
                       "leveling-cell competitor (spent calibration data, baseline use; no "
                       "score)",
        **({"ids": ids} if ids else {"none": "the screen kept no benchmark"}),
        "window": "2016-01..2021-12"}}
    kw = {"prereg_dir": prereg_dir} if prereg_dir is not None else {}
    return prereg3.record_look([LOOK_UNIT], LOOK_SCOPE, L.WP, LOOK_PURPOSE, members=members,
                               note=f"{PRODUCT} for {L.ID}; panel sha256 {panel_sha}", **kw)


def build(mapping_json: Path, data_dir: Path, out_root: Path, *, rates: pd.DataFrame | None
          = None, panel: Path | None = None, grid: dict | None = None,
          prereg_dir=None) -> Path:
    from common.manifest import build_manifest, new_version_dir, write_manifest

    mapping = json.loads(Path(mapping_json).read_text(encoding="utf-8"))
    if grid is None:
        with np.load(REPO / L.FORWARD, allow_pickle=False) as z:
            grid = {k: z[k] for k in ("mask", "dx", "x0", "y0")}
    if rates is None:
        from hydrophysics.twin.column_step import leveling_rates

        rates = leveling_rates(str(data_dir), T0, T1, min_obs=5, max_rate=0.5)
    panel_sha = L.sha256(panel) if panel else None
    log_look(rates, panel_sha, prereg_dir)
    tab = competitor_table(rates, mapping, grid)
    d = new_version_dir(PRODUCT, out_root)
    tab.to_csv(d / "competitor.csv", index=False)
    ok = bool(tab["rate_cm_yr"].notna().all())
    gate = {"passed": ok, "criteria": "every primary pile has a competitor under the declared "
                                      "rule (pile cell, else 1.5 km); nothing is fitted"}
    (d / "validation.json").write_text(json.dumps(
        {"gate": gate, "rules": tab[["pile_key", "rule", "n_benchmarks"]].to_dict("records")},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    inputs = [str(mapping_json)] + ([str(panel)] if panel else [])
    m = build_manifest(
        d, product=PRODUCT, layer="L2", inputs=inputs,
        code=[f"{L.E}/export_leveling_competitor.py", f"{L.E}/n3lib.py",
              "hydrophysics/twin/leveling.py", "hydrophysics/twin/column_step.py",
              "hydrophysics/twin/mechanism.py"],
        parameters={"t0": T0, "t1_exclusive": T1, "min_obs": 5, "max_rate_m_yr": 0.5,
                    "fallback_radius_m": FALLBACK_RADIUS_M, "statistic": "median rate",
                    "panel_sha256": panel_sha, "look": {"unit": LOOK_UNIT, "scope": LOOK_SCOPE,
                                                        "purpose": LOOK_PURPOSE},
                    "registration": L.ID, "ruling": "user 2026-11-01 (fourth set): the "
                    "pre-origin leveling-cell rate replaces printed-pile persistence"},
        fit_window_end=WINDOW_END, validation_report="validation.json", gate=gate,
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN)
    write_manifest(d, m)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="directory holding ls_cache/ (default $HYDROMIND_GW_DATA or "
                         "chou-shui-data/data)")
    ap.add_argument("--out-root", type=Path, default=REPO / "results/dataprep")
    a = ap.parse_args(argv)
    dd = a.data_dir or Path(os.environ.get("HYDROMIND_GW_DATA", REPO / "chou-shui-data/data"))
    panel = Path(dd) / PANEL_REL
    if not panel.is_file():
        raise SystemExit(f"no leveling panel at {panel}")
    d = build(a.mapping, Path(dd), a.out_root, panel=panel)
    print("built", d, "MANIFEST sha256", L.sha256(d / "MANIFEST.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
