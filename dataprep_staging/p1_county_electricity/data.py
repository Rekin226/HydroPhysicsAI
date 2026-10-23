"""WP P1 inputs per variant (``preorigin``: nothing dated after 2022-12; ``full``: everything).

Paths are fixed L0 locations plus the P1 census build (``census_monthly.py``) and the
provisional phi_in (``phi_in.py``). Run consumers with ``python -I``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib  # noqa: E402

REPO = HERE.parents[1]
L0 = REPO / "data_fetch_api/new_measurements"
TPC = L0 / "tpc_d007012/2026-10-12/d007012.csv"
NAT236 = L0 / "moeaea_agri_electricity/2026-10-12/set_236.csv"
NAT256 = L0 / "moeaea_agri_electricity/2026-10-12/set_256.csv"
SUBIND = L0 / "tpc_d003004/2026-10-12/001.zip"
BUILD = REPO / "results/dataprep/county_electricity/_build"
CENSUS = {"preorigin": BUILD / "census/census_class_month_preorigin.parquet",
          "full": BUILD / "census/census_class_month_full.parquet"}
PHI = BUILD / "phi/phi_in.json"
L0_INPUTS = [TPC, NAT236, NAT256, SUBIND, REPO / "AMP_V2/data/tpc_pumps.parquet",
             REPO / "AMP_V2/data/pump_kwh_all.parquet"]
END = {"preorigin": "2022-12-01", "full": "2026-08-01"}
CENSUS_LAST = "2025-06-01"            # July 2025 is partial: masked from 2025-07


def load(variant: str) -> dict:
    if variant not in END:
        raise ValueError(variant)
    end = pd.Timestamp(END[variant])
    tpc = lib.read_tpc_d007012(TPC)
    tpc = tpc[tpc.date <= end].reset_index(drop=True)
    pair = lib.tpc_pair(tpc)
    nat = lib.read_national_ag(NAT256, NAT236)
    nat = nat[nat.index <= end]
    cen = pd.read_parquet(CENSUS[variant])
    cen = cen[cen.index <= min(end, pd.Timestamp(CENSUS_LAST))]
    sub = lib.read_d003004_subindustry(SUBIND)
    sub = sub[sub.index <= end]
    phi = json.loads(PHI.read_text(encoding="utf-8"))
    idx = pd.date_range("2007-01-01", end, freq="MS", name="date")
    d = pd.DataFrame(index=idx)
    d = d.join(pair).join(nat)
    d["YL_clean"] = cen["clean_total"].reindex(idx)
    d["YL_clean_irr"] = cen["clean_irrigation"].reindex(idx)
    d["YL_clean_other"] = cen["clean_other_classes"].reindex(idx)
    d["YL_dedup_agri"] = cen["dedup_agri"].reindex(idx)
    return {"variant": variant, "end": end, "tpc": tpc, "d": d, "census": cen, "sub": sub,
            "phi": phi}


def window(d: pd.DataFrame, t0: str, t1: str, cols) -> pd.DataFrame:
    return d.loc[t0:t1, list(cols)].dropna()


def nnls_inputs(d: pd.DataFrame, t0: str, t1: str, p_col: str = "YL_clean"):
    w = window(d, t0, t1, ["CH_ag", "YL_ag", p_col])
    P = w[p_col].to_numpy()
    N = (w["YL_ag"] - w[p_col]).to_numpy()
    return w, P, N, w["CH_ag"].to_numpy(), w.index.year.to_numpy()


def sha_list(paths) -> list[dict]:
    sys.path.insert(0, str(HERE.parent))
    from common.hashing import sha256_file
    from common.paths import rel_to_repo
    return [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in paths]


__all__ = ["load", "window", "nnls_inputs", "np"]
