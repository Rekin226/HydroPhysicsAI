"""Post-publication checks of surface_water_v4/full/v2 (the 2026 extension).

    cpu_slot.sh python -I dataprep_staging/w1_canal/ext_verify.py

* the A0 checker re-hashes full/v2 and the new L0 SOURCE.json;
* loader-guard probes (U4: recorded-forcing arm only);
* 2012-2025 identical to full/v1 (monthly table, draws, grid), 2026 months flag 6, no
  negative value or draw;
* post hoc (written after the results were known; nothing here changes the product): the
  extrapolation range of the 2026 regressors and predictions, and the May-2026 value if the
  predicted inflow were floored at the lowest observed May inflow;
* the forward-archive probes of the WRA real-time feeds (status codes only).
Writes results/dataprep/reports/W1_evidence/ext2026_verify.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import check  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402
from common.source import check_source_file  # noqa: E402

from w1_canal import diversion  # noqa: E402
from w1_canal import ext_config as E  # noqa: E402
from w1_canal import ext_model as M  # noqa: E402
from w1_canal.build import jsonable  # noqa: E402
from w1_canal.verify import latest, probe, try_path  # noqa: E402

EVIDENCE = DATAPREP / "reports" / "W1_evidence"
PROBES = ["https://fhy.wra.gov.tw/WraApi/v1/Water/Station",
          "https://fhy.wra.gov.tw/WraApi/v1/Water/RealTimeData",
          "https://fhy.wra.gov.tw/WraApi/v1/Reservoir/Daily",
          "https://opendata.wra.gov.tw/api/v2/51023e88-4c76-4dbc-bbb9-470da690d539"
          "?format=JSON",
          "https://sta.ci.taiwan.gov.tw/STA_WaterResource_v2/v1.0/"]


def http_probe(url: str) -> dict:
    """Status code, size and content type only; the body is discarded."""
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-m", "30", "-A", "Mozilla/5.0",
                        "-w", "%{http_code} %{size_download} %{content_type}", url],
                       capture_output=True, text=True)
    parts = (r.stdout or "").split(" ", 2)
    return {"url": url, "http_code": parts[0] if parts else None,
            "bytes": parts[1] if len(parts) > 1 else None,
            "content_type": parts[2] if len(parts) > 2 else None,
            "curl_exit": r.returncode,
            "usable": False}


def main():
    v2 = latest("full")
    v1 = M.FULL_V1
    out = {"product": rel_to_repo(v2)}
    out["checker"] = check.check_manifest(v2 / "MANIFEST.json")
    out["source_json"] = check_source_file(E.L0_DIR / "SOURCE.json")["status"]

    def table(use):
        def f(p):
            t = p.read_table("district_monthly.csv")
            return {"table": "district_monthly.csv", "ok": True, "rows": int(len(t)),
                    "date_max": str(t["date"].max())}
        return f
    out["guard_probes"] = [
        probe(v2, "forcing_recorded_arm", table("forcing_recorded_arm")),
        probe(v2, "forcing_recorded_arm", try_path("grid_sw_v4.npz")),
        probe(v2, "statistical_layer_history"),
        probe(v2, "forcing_frozen"), probe(v2, "model_design"), probe(v2, "scoring_target"),
        probe(v2, "calibration_target")]
    m1 = pd.read_csv(v1 / "district_monthly.csv")
    m2 = pd.read_csv(v2 / "district_monthly.csv")
    old = m2[m2.date <= "2025-12-01"].reset_index(drop=True)
    cols = ["date", "district", "value", "sd", "flag", "method", "indicator", "ss_mean",
            "ss_p05", "ss_p50", "ss_p95"]
    same_tab = bool(old[cols].equals(m1[cols]))
    new = m2[m2.date > "2025-12-01"]
    d1 = np.load(v1 / "district_draws.npz", allow_pickle=True)
    d2 = np.load(v2 / "district_draws.npz", allow_pickle=True)
    n1 = d1["draws_changhua"].shape[1]
    g1 = np.load(v1 / "grid_sw_v4.npz", allow_pickle=True)
    g2 = np.load(v2 / "grid_sw_v4.npz", allow_pickle=True)
    out["exactness"] = {
        "monthly_2012_2025_identical_to_v1": same_tab,
        "draws_2012_2025_identical": bool(np.array_equal(d1["draws_changhua"],
                                                         d2["draws_changhua"][:, :n1])
                                          and np.array_equal(d1["draws_yunlin"],
                                                             d2["draws_yunlin"][:, :n1])),
        "grid_2012_2025_identical": bool(np.array_equal(g1["value"], g2["value"][:, :n1])
                                         and np.array_equal(g1["sd"], g2["sd"][:, :n1])),
        "rows_2026": int(len(new)), "flags_2026": sorted(new.flag.unique().tolist()),
        "dates_2026": [new.date.min(), new.date.max()],
        "negative_values": int((m2.value < 0).sum()),
        "negative_draws": int((d2["draws_changhua"] < 0).sum() + (d2["draws_yunlin"] < 0)
                              .sum()),
        "grid_shape": list(g2["value"].shape), "grid_flag_2026": sorted(
            np.unique(g2["flag"][:, n1:]).tolist()),
        "mask_hash_unchanged": str(g1["mask_hash"]) == str(g2["mask_hash"]),
        "district_draw_mean_vs_value_2026": {
            d: float(np.abs(d2[f"draws_{nm}"][:, n1:].mean(0)
                            - new[new.district == nm].value.to_numpy()).max())
            for d, nm in (("chc", "changhua"), ("yl", "yunlin"))}}
    # post hoc: extrapolation and the May-2026 floor sensitivity (not used in the product)
    ri = pd.read_csv(v2 / "basin_rain_index_monthly.csv", parse_dates=["date"])
    P = ri.set_index("date")["value"]
    base = M.load_full_v1()
    Q = base["Q"]
    inflow = pd.read_csv(v2 / "inflow_predicted_monthly.csv", parse_dates=["date"])
    rng = {}
    for _, r in inflow.iterrows():
        m = r.date.month
        hist_q = Q[Q.index.month == m]
        hist_p = P["2012":"2025"][P["2012":"2025"].index.month == m].dropna()
        rng[f"{r.date:%Y-%m}"] = {
            "rain_mm": float(P.loc[r.date]), "rain_hist_min_max": [float(hist_p.min()),
                                                                   float(hist_p.max())],
            "q_pred": float(r.value), "q_hist_min_max": [float(hist_q.min()),
                                                         float(hist_q.max())],
            "q_pred_below_hist_min": bool(r.value < hist_q.min())}
    ff = base["final_fit"]
    months26 = pd.DatetimeIndex(inflow.date)
    qf = inflow.value.to_numpy().copy()
    for k, mo in enumerate(months26):
        qf[k] = max(qf[k], float(Q[Q.index.month == mo.month].min()))
    R25 = M.r_cms(base["jiji"], E.R_EXT_FROM_YEAR, 365)
    dd = M.daily_from_monthly(qf, months26, R25)
    mc, my = diversion.simulate(dd, ff["beta"], ff["a_chc_cms"], ff["a_yl_cms"], ff["s_chc"],
                                ff["s_yl"], ff["winter"])
    out["post_hoc"] = {
        "note": "computed after the results were known; not used in the product",
        "extrapolation": rng,
        "floored_at_hist_min_inflow": {"chc": mc[:len(months26)].round(3).tolist(),
                                       "yl": my[:len(months26)].round(3).tolist(),
                                       "jan_sep_chc": float(mc.sum()),
                                       "jan_sep_yl": float(my.sum())}}
    out["forward_archive_probes"] = [http_probe(u) for u in PROBES]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE / "ext2026_verify.json", "w") as f:
        json.dump(jsonable(out), f, indent=1, ensure_ascii=False)
    print(json.dumps(jsonable({k: out[k] for k in ("checker", "source_json", "exactness")}),
                     indent=1, ensure_ascii=False)[:3000])
    print(json.dumps(jsonable(out["post_hoc"]["floored_at_hist_min_inflow"])))
    for g in out["guard_probes"]:
        print(g["use"], g.get("opened"), g.get("action", g.get("error")))
    for p in out["forward_archive_probes"]:
        print(p["http_code"], p["content_type"], p["url"][:70])


if __name__ == "__main__":
    main()
