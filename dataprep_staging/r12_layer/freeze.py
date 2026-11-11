"""Freeze SL.subsidence_layer.v1 and its predictions.

``python -m dataprep_staging.r12_layer.freeze base``
    writes ``reports/SL/freeze/model.json`` (coefficient, input hashes) and
    ``predictions_base.csv``: the constant-rate arm C0 (interval-independent, valid for every
    survey year) and the head-driven arm HL for survey year 2026 (record only; the 2026
    summary is sealed by N3S-L and never scored).
``python -m dataprep_staging.r12_layer.freeze year --survey-year 2027``
    writes ``predictions_hl_2027.csv`` once heads to May 2027 are held. It must run before
    the official summary of that survey year is read; an existing file is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from dataprep_staging.r12_layer import model as M
from dataprep_staging.r12_leveling.paths import DATAPREP, DDIR

OUT = os.path.join(DATAPREP, "reports", "SL", "freeze")
COUNTIES = ("雲林縣", "彰化縣")
LEVELING = os.path.join(DDIR, "ls_cache", "ls-wra-lsp-obs__choushui_panel.parquet")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(L: M.Layer, arm: str, year: int, gamma: float, status: str) -> pd.DataFrame:
    r = L.cell_rates(year, gamma)
    out = [{"arm": arm, "survey_year": year, "level": "county", "county": c, "township": "",
            "quantity": "area_km2_over_3cm", "value": L.area(r, c), "status": status}
           for c in COUNTIES]
    out += [{"arm": arm, "survey_year": year, "level": "county", "county": c, "township": "",
             "quantity": "max_rate_cm_yr", "status": status,
             "value": float(100 * np.nanmax(r[L.cells.county.to_numpy() == c]))}
            for c in COUNTIES]
    for _, t in L.township_peaks(r).iterrows():
        out.append({"arm": arm, "survey_year": year, "level": "township", "county": t.county,
                    "township": t.township, "quantity": "peak_cm_yr",
                    "value": float(t.peak_cm_yr), "status": status})
    return pd.DataFrame(out)


def base() -> None:
    os.makedirs(OUT, exist_ok=False)
    L = M.Layer()
    gamma, per_year = M.fit_gamma(L.D, L.F, L.xy)
    model = {
        "id": "SL.subsidence_layer.v1", "gamma_per_m": gamma,
        "gamma_fit_years": sorted(per_year), "ref_years": list(M.REF_YEARS),
        "threshold_m_yr": M.THRESH, "n_benchmarks_with_rate": int(np.isfinite(L.rate).sum()),
        "n_cells": int(len(L.cells)),
        "driver_2026_county_mean_m": {c: L.county_driver(2026, c) for c in COUNTIES},
        "inputs": {"leveling_panel": sha(LEVELING), "heads_yearbook": sha(M.YEARBOOK),
                   "heads_clean": sha(M.CLEAN + "heads_monthly.parquet"),
                   "station_meta": sha(M.CLEAN + "station_meta_clean.parquet"),
                   "cells": sha(M.CELLS)},
    }
    P = pd.concat([rows(L, "C0", 0, 0.0, "frozen_all_years"),
                   rows(L, "HL", 2026, gamma, "record_only_never_scored")])
    P.to_csv(os.path.join(OUT, "predictions_base.csv"), index=False)
    model["predictions_base_sha256"] = sha(os.path.join(OUT, "predictions_base.csv"))
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(model, f, indent=1, ensure_ascii=False)
    print(json.dumps(model, indent=1, ensure_ascii=False))
    print(P[P.level == "county"].to_string(index=False))


def year(survey_year: int) -> None:
    path = os.path.join(OUT, f"predictions_hl_{survey_year}.csv")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; a frozen prediction is never replaced")
    with open(os.path.join(OUT, "model.json")) as f:
        model = json.load(f)
    L = M.Layer(last_month=f"{survey_year}-05")
    drv = L.driver(survey_year)
    if np.isfinite(drv).sum() < 0.9 * np.isfinite(L.rate).sum():
        raise SystemExit(f"heads to {survey_year}-05 are not held for enough wells")
    P = rows(L, "HL", survey_year, model["gamma_per_m"], "frozen_before_summary_read")
    P.to_csv(path, index=False)
    print(json.dumps({"file": path, "sha256": sha(path)}, indent=1))
    print(P[P.level == "county"].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("base", "year"))
    ap.add_argument("--survey-year", type=int)
    a = ap.parse_args()
    base() if a.cmd == "base" else year(a.survey_year)
