"""WP S3: run both families with the tuned detector and assemble every product table.

Shared by ``run_validation.py`` (tables in memory) and ``build.py`` (tables written to the
product directory); both call :func:`build`, which is deterministic, and ``build.py``
checks its tables against the hashes the validation run recorded.
"""
from __future__ import annotations

import math

import io_gnss as io
import model as M
import numpy as np
import pandas as pd
import pipeline as P
import products as PR

SRC_WRA = "chou-shui-data/data/ls_cache/ls-wra-gnss-obs__{}.parquet"
SRC_NCKU = "chou-shui-data/data/ls_cache/ls-ncku-gnss1day-obs__{}.parquet"
SRC_NGL = "data_fetch_api/new_measurements/ngl_gnss/{}/{}.tenv3"


def _ngl_src(k: str) -> str:
    p = io.ngl_tenv3_path(k)
    return SRC_NGL.format(p.parent.name, k)


def clean_json(o):
    """Recursively make ``o`` strict-JSON serialisable (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date()) if o == o.normalize() else str(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    return o


def build(p: M.Params, ref: list[str], window_end, version: str) -> dict:
    """All product tables for one variant (``window_end`` None = full records)."""
    grid = np.load(io.REPO / "results/twin_forward/datum_gate_innov.npz")
    wser = P.wra_series(window_end)
    wra = P.run_wra(p, window_end=window_end, series=wser)
    nser, _ = P.ngl_series(window_end)
    ngl = P.run_ngl(p, ref=ref, window_end=window_end, series=nser)
    ncku = P.run_ncku(p, window_end=window_end) if window_end is None else None
    cms = P.product_cms(wra, ngl, ref, p)

    steps, dailies, stations = [], {"wra": [], "ngl": [], "ncku": []}, []
    sig, rho = {}, {}
    fams = [("wra", wra), ("ngl", ngl)] + ([("ncku", ncku)] if ncku is not None else [])
    xy = {"wra": PR.wra_xy(), "ngl": PR.ngl_xy(list(ngl.results)),
          "ncku": PR.ncku_xy() if ncku is not None else {}}
    for fam_name, fam in fams:
        for k, r in fam.results.items():
            srow = PR.step_rows(fam_name, r, p)
            steps += srow
            src = (SRC_WRA.format(k) if fam_name == "wra" else
                   SRC_NCKU.format(k.replace("NCKU_", "")) if fam_name == "ncku" else
                   _ngl_src(k))
            dailies[fam_name].append(PR.daily_table(fam_name, r, srow, version, src))
            sig[k], rho[k] = r.sigma_hf, r.rho
            c = io.grid_cells(np.array([xy[fam_name][k]]), grid)[0]
            if fam_name == "wra":
                role = "flagged (excluded from the common modes)" if k in P.WRA_FLAG else \
                    "member"
                note = P.WRA_FLAG.get(k, "")
            elif fam_name == "ngl":
                role = ("off-fan reference" if k in ref else
                        "fan box" if k in P.NGL_FAN else
                        "on grid, east of the plan box" if k in P.NGL_GRID_EXTRA else
                        "north of the box" if k in P.NGL_NORTH else "regional")
                note = ""
            else:
                role, note = "ncku (own processing, no common mode)", ""
            stations.append(PR.station_row(fam_name, k, r, xy[fam_name][k], c, role, note))
    for k, why in P.WRA_DROP.items():
        if k in wser:
            c = io.grid_cells(np.array([xy["wra"][k]]), grid)[0]
            stations.append(PR.station_row("wra", k, None, xy["wra"][k], c, "dropped", why))
    out = {"cms": cms, "cms_monthly": PR.cm_monthly(cms),
           "steps": pd.DataFrame(steps), "stations": pd.DataFrame(stations),
           "wra_members": list(wra.info["members"])}
    for fam_name in ("wra", "ngl", "ncku"):
        if not dailies[fam_name]:
            continue
        d = pd.concat(dailies[fam_name], ignore_index=True)
        out[f"{fam_name}_daily"] = d
        out[f"{fam_name}_monthly"] = PR.monthly_table(d, rho, sig)
    if len(out["steps"]):
        s = out["steps"]
        for c in ("date", "gap_lo"):
            s[c] = s[c].map(lambda v: None if v is None or pd.isna(v) else str(
                pd.Timestamp(v).date()))
        out["steps"] = s[["family", "station", "date", "size_m", "se_m", "evidence", "detail",
                          "z", "gap_lo", "gap_days", "tied", "size_own_series_m",
                          "se_own_series_m", "z_own_series", "basis", "flag_after"]]
    return out


TABLES = ("wra_daily", "wra_monthly", "ngl_daily", "ngl_monthly", "ncku_daily",
          "ncku_monthly", "steps", "stations", "cms", "cms_monthly")


def table_hashes(tab: dict) -> dict:
    out = {}
    for k in TABLES:
        if k in tab:
            df = tab[k]
            out[k] = {"rows": int(len(df)),
                      "hash": str(int(pd.util.hash_pandas_object(
                          df.round(9), index=True).sum() % (2 ** 63)))}
    return out


def steps_summary(steps: pd.DataFrame) -> dict:
    if not len(steps):
        return {}
    s = steps
    out = {"n_steps": int(len(s)),
           "by_family_basis": {f"{f}:{b}": int(n) for (f, b), n in
                               s.groupby(["family", "basis"]).size().items()},
           "by_family_evidence": {f"{f}:{e}": int(n) for (f, e), n in
                                  s.groupby(["family", "evidence"]).size().items()},
           "untied": int((~s.tied.astype(bool)).sum())}
    big = s[s.size_m.abs() >= 0.02]
    out["steps_ge_2cm"] = clean_json(big[["family", "station", "date", "size_m", "evidence",
                                          "basis"]].to_dict(orient="records"))
    return out
