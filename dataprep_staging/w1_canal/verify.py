"""Post-publication checks of surface_water_v4 (run in a CPU slot after publish.py).

    python -m w1_canal.verify            (from dataprep_staging/)

* re-hash both manifests with the A0 checker (common.check);
* loader-guard probes: what each variant may and may not be opened for (U1, U3-U5);
* exactness of the benchmark: Denton series and every state-space draw sum to the annual
  district totals; no negative month; the pre-origin outputs end by 2022-12;
writes results/dataprep/reports/W1_evidence/verify.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import check, guard  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal.sources import jiji_annual  # noqa: E402

EVIDENCE = DATAPREP / "reports" / "W1_evidence"


def latest(variant: str) -> Path:
    base = C.product_dir_base(variant)
    vs = sorted((p for p in base.glob("v*") if p.name[1:].isdigit()),
                key=lambda p: int(p.name[1:]))
    if not vs:
        raise FileNotFoundError(base)
    return vs[-1]


def probe(d: Path, use: str, action=None) -> dict:
    try:
        p = guard.open_product(d, use)
        out = {"opened": True}
        if action is not None:
            out["action"] = action(p)
        return {"use": use, **out}
    except Exception as e:  # noqa: BLE001
        return {"use": use, "opened": False, "error": type(e).__name__, "msg": str(e)[:240]}


def try_path(name):
    def f(p):
        try:
            p.path(name)
            return {"path": name, "ok": True}
        except Exception as e:  # noqa: BLE001
            return {"path": name, "ok": False, "error": type(e).__name__, "msg": str(e)[:240]}
    return f


def try_table(name):
    def f(p):
        try:
            t = p.read_table(name)
            return {"table": name, "ok": True, "rows": int(len(t)),
                    "date_max": str(t["date"].max()) if "date" in t else None}
        except Exception as e:  # noqa: BLE001
            return {"table": name, "ok": False, "error": type(e).__name__, "msg": str(e)[:240]}
    return f


def exactness(d: Path, years) -> dict:
    jj, _ = jiji_annual()
    out = {}
    m = pd.read_csv(d / "district_monthly.csv")
    z = np.load(d / "district_draws.npz", allow_pickle=True)
    for dist, col, key in (("changhua", C.JIJI_COL["chc"], "draws_changhua"),
                           ("yunlin", C.JIJI_COL["yl"], "draws_yunlin")):
        t = jj[col].reindex(list(years)).to_numpy(float)
        v = m[m.district == dist].value.to_numpy()
        ann = v.reshape(len(years), 12).sum(1)
        X = z[key]
        annX = X.reshape(len(X), len(years), 12).sum(2)
        out[dist] = {"denton_annual_max_abs_diff": float(np.abs(ann - t).max()),
                     "draws_annual_max_abs_diff": float(np.abs(annX - t[None, :]).max()),
                     "negative_months_denton": int((v < -1e-12).sum()),
                     "negative_months_draws": int((X < -1e-12).sum()),
                     "n_draws": int(len(X))}
    dates = []
    for f in ("district_monthly.csv", "percolation_basis.csv", "janmay_validation.csv"):
        t = pd.read_csv(d / f)
        if "date" in t:
            dates.append(str(t["date"].max()))
    for f in ("inflow_composite_daily.parquet", "gauges_daily_L1.parquet"):
        dates.append(str(pd.read_parquet(d / f, columns=["date"])["date"].max()))
    g = np.load(d / "grid_sw_v4.npz", allow_pickle=True)
    dates.append(str(g["dates"].max()))
    out["max_date_in_outputs"] = max(dates)
    return out


def main() -> dict:
    pre, full = latest("preorigin"), latest("full")
    res = {"products": {"preorigin": rel_to_repo(pre), "full": rel_to_repo(full)}}
    res["checker"] = {v: check.check_manifest(d / "MANIFEST.json") for v, d in
                      (("preorigin", pre), ("full", full))}
    res["guard_probes"] = {
        "preorigin": [
            probe(pre, "forcing_frozen", try_table("district_monthly.csv")),
            probe(pre, "forcing_frozen", try_path("grid_sw_v4.npz")),
            probe(pre, "model_design", try_path("district_draws.npz")),
            probe(pre, "selection", try_path("grid_sw_v4.npz")),
            probe(pre, "scoring_target"),
            probe(pre, "calibration_target"),
            probe(pre, "forcing_recorded_arm"),
        ],
        "full": [
            probe(full, "forcing_recorded_arm", try_table("district_monthly.csv")),
            probe(full, "forcing_frozen"),
            probe(full, "model_design"),
            probe(full, "scoring_target"),
        ],
    }
    res["exactness"] = {"preorigin": exactness(pre, C.VARIANTS["preorigin"]["years"]),
                        "full": exactness(full, C.VARIANTS["full"]["years"])}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE / "verify.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "checker"}, indent=1,
                     ensure_ascii=False, default=str)[:6000])
    for v, c in r["checker"].items():
        print(v, {k: c[k] for k in c if k != "files"} if isinstance(c, dict) else c)
