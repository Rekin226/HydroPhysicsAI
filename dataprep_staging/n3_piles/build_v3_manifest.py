"""Builder stub of deep_benchmarks_printed v3: the FUTURE WRA monthly-report months only.

User ruling 2026-11-01 (fourth set, rank 6): a future-month deep_benchmarks_printed v3 may
list ``prospective_scorer``. This builder makes that product from twin.report_extract's
two-model transcription of the reports for data months 2026-12 or later (the N3 window and
its continuation), and from the table-3 well-column transcription (well_columns.py) when the
head fallback is needed. It refuses any row dated before 2026-12, so no looked or pre-window
month can enter a product that grants prospective_scorer. It is run by the lead at scoring
time (never before the window's reports exist); before registration it is only tested on
synthetic transcriptions.

value = compression (mm), sign as printed (negative = rebound); flags: 0 two reads agree, 8
printed as '—', 9 reads disagree. Gate: share of non-null window values two-read verified
>= 0.9 (the v2 gate). Allowed uses: prospective_scorer, validator (N2's report-only pile arm
stays report text), after_the_fact_analysis; every fitting, design and pre-origin use is
forbidden.

Usage (2028, the lead):
    python -I dataprep_staging/n3_piles/build_v3_manifest.py --piles <report_extract
        deep_piles.csv> [--wells <accepted well-column CSV>] [--out-root <dir>]
"""
from __future__ import annotations

import argparse
import json
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

PRODUCT = "deep_benchmarks_printed"
VERSION = "v3"
FIRST_MONTH = L.WINDOW[0]
GATE_MIN_VERIFIED = 0.9
ALLOWED = ["prospective_scorer", "validator", "after_the_fact_analysis"]
FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "calibration_target", "selection",
             "model_design", "prior", "prior_width_input", "initial_conditions", "idw_field",
             "forcing_frozen", "forcing_recorded_arm", "statistical_layer_history",
             "design_time_evidence"]
PILE_TABLE = "integrated_monthly.parquet"
WELL_TABLE = "wells_monthly.parquet"


class FutureOnly(ValueError):
    """A row dated before the first window month."""


class UndeclaredSign(ValueError):
    """A well-column table whose printed sign legend is missing or not usable."""


def pile_rows(df: pd.DataFrame, first: str = FIRST_MONTH) -> pd.DataFrame:
    """report_extract deep_piles.csv rows -> product rows (mm); refuses pre-window months."""
    old = sorted(set(df.loc[df["month"].astype(str) < first, "month"].astype(str)))
    if old:
        raise FutureOnly(f"rows dated before {first}: {old[:5]} (future months only)")
    out = pd.DataFrame({
        "sensor_id": [f"{p}:{int(d) if pd.notna(d) else 'NA'}"
                      for p, d in zip(df["pile"], df["depth_m"], strict=True)],
        "pile": df["pile"].astype(str), "depth_m": pd.to_numeric(df["depth_m"]),
        "month": df["month"].astype(str),
        "value": pd.to_numeric(df["month_cm"], errors="coerce") * 10.0,
        "year_value": pd.to_numeric(df["year_cm"], errors="coerce") * 10.0,
        "agree": df["verified"].astype(bool),
        "chain_ok": df.get("chain_ok", None),
        "file": df.get("file", None)})
    out["flag"] = np.where(~out["agree"], 9, np.where(out["value"].isna(), 8, 0)).astype(int)
    out["sd"] = 0.5            # half the printed 0.1 cm step, mm; UNCALIBRATED (plan 4.7)
    out["method"] = "twin.report_extract two-model read (copy only)"
    out["product_version"] = f"{PRODUCT}/{VERSION}"
    return out.sort_values(["pile", "depth_m", "month"]).reset_index(drop=True)


def well_rows(df: pd.DataFrame, first: str = FIRST_MONTH) -> pd.DataFrame:
    """Accepted table-3 well columns (well_columns.accept) -> product rows; refuses
    pre-window months and any table without a usable printed sign legend (positive_means
    'rise' or 'fall', well_columns.SIGN_LEGENDS): the sign is never assumed."""
    import well_columns as WC

    old = sorted(set(df.loc[df["month"].astype(str) < first, "month"].astype(str)))
    if old:
        raise FutureOnly(f"well rows dated before {first}: {old[:5]}")
    if "positive_means" not in df.columns:
        raise UndeclaredSign("well table has no positive_means column (sign legend)")
    bad = sorted(set(df.loc[~df["positive_means"].isin(WC.SIGN_LEGENDS), "month"].astype(str)))
    if bad:
        raise UndeclaredSign(f"well-column sign legend missing or not usable in {bad[:5]}")
    out = df[["month", "pile", "well", "month_m", "year_m", "positive_means", "verified",
              "flag"]].copy()
    out["month"] = out["month"].astype(str)
    out["flag"] = out["flag"].astype(int)
    return out


def build(piles_csv: Path, out_root: Path, *, wells_csv: Path | None = None,
          code: list[str] | None = None, created: str | None = None) -> Path:
    from common.manifest import build_manifest, write_manifest

    d = Path(out_root) / PRODUCT / VERSION
    if d.exists():
        raise FileExistsError(f"{d} exists; a product version is built once")
    piles = pile_rows(pd.read_csv(piles_csv))
    wells = well_rows(pd.read_csv(wells_csv)) if wells_csv else None
    nn = piles["value"].notna() | (piles["flag"] == 9)
    share = float((piles.loc[nn, "flag"] == 0).mean()) if nn.any() else 0.0
    gate = {"passed": bool(share >= GATE_MIN_VERIFIED),
            "criteria": f"share of non-null window values two-read verified >= "
                        f"{GATE_MIN_VERIFIED} (v2 gate); got {share:.3f}"}
    d.mkdir(parents=True)
    piles.to_parquet(d / PILE_TABLE, index=False)
    if wells is not None:
        wells.to_parquet(d / WELL_TABLE, index=False)
    (d / "validation.json").write_text(json.dumps(
        {"verified_share": share, "n_rows": int(len(piles)), "gate": gate,
         "months": [piles["month"].min(), piles["month"].max()]}, indent=1) + "\n")
    (d / "README.md").write_text(
        f"# {PRODUCT}/{VERSION}: future WRA monthly-report deep-pile months only\n\n"
        f"Data months {FIRST_MONTH} or later, transcribed by twin.report_extract (two reads; "
        "the model never computed a value). value = compression mm. Built for "
        f"{L.ID}; grants prospective_scorer by user ruling 2026-11-01.\n", encoding="utf-8")
    code = code or [f"{L.E}/build_v3_manifest.py", f"{L.E}/n3lib.py",
                    "hydrophysics/twin/report_extract.py"]
    inputs = [str(piles_csv)] + ([str(wells_csv)] if wells_csv else [])
    m = build_manifest(
        d, product=PRODUCT, layer="L1", inputs=inputs, code=code,
        parameters={"first_month": FIRST_MONTH, "future_months_only": True,
                    "registration": L.ID, "ruling": "user 2026-11-01 (fourth set, rank 6)",
                    "fitted_parameters": "none"},
        fit_window_end=str(piles["month"].max()), validation_report="validation.json",
        gate=gate, allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN, created=created,
        extra={"notes": ["future months only (>= 2026-12); no pre-window month",
                         "N2's pile arm keeps its report-only role (validator); N3 gates",
                         "007001 (彰雲 monthly report table 1) is the declared second source"]})
    write_manifest(d, m)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--piles", type=Path, required=True)
    ap.add_argument("--wells", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO / "results/dataprep")
    a = ap.parse_args(argv)
    d = build(a.piles, a.out_root, wells_csv=a.wells)
    print("built", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
