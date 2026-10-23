"""Post-publication checks of surface_water_v4/full/v3 (W1 repair round).

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep_verify.py

* the A0 checker re-hashes every surface_water_v4 manifest (preorigin/v1, full/v1-v3);
* the three repair-round registrations load with their recorded sha256 (INDEX and ledger);
* loader-guard probes of full/v3 (U4: recorded-forcing arm only);
* values identical to full/v2 (table, grid value, 2012-2025 draws = full/v1); sd changed only
  on structural-zero months (now 0) and in 2026; structural-zero members and sd 0; the 2026
  sd equals the members' standard deviation; no negative member; every row carries sd_basis
  and interval_calibration (plan 4.7); the Denton-target RMSEs are out of the scores (U1);
* the forward-archive probes of the WRA real-time feeds (status codes only).
Writes results/dataprep/reports/W1_evidence/rep_verify.json.
"""
from __future__ import annotations

import datetime as dt
import json
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
from harness import prereg2  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal.build import jsonable  # noqa: E402
from w1_canal.ext_verify import PROBES, http_probe  # noqa: E402
from w1_canal.verify import probe, try_path, try_table  # noqa: E402

EVIDENCE = DATAPREP / "reports" / "W1_evidence"
IDS = ["W1_janmay_benchmark_2026-10-12.record", "W1.ext2026_uncertainty.r1",
       "W1.ext2026_prospective_obs.v1"]


def main():
    base = C.product_dir_base("full")
    v1, v2, v3 = base / "v1", base / "v2", base / "v3"
    out = {"product": rel_to_repo(v3), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    out["checker"] = {rel_to_repo(d): check.check_manifest(d / "MANIFEST.json")
                      for d in (C.product_dir_base("preorigin") / "v1", v1, v2, v3)}
    regs = {}
    for i in IDS:
        try:
            doc, sha = prereg2.load_prereg(i)
            regs[i] = {"ok": True, "sha256": sha, "created": doc["created"],
                       "changed_inputs": prereg2.changed_inputs(doc)}
        except Exception as e:  # noqa: BLE001
            regs[i] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["registrations"] = regs
    out["guard_probes"] = [
        probe(v3, "forcing_recorded_arm", try_table("district_monthly.csv")),
        probe(v3, "forcing_recorded_arm", try_path("grid_sw_v4.npz")),
        probe(v3, "statistical_layer_history"), probe(v3, "after_the_fact_analysis"),
        probe(v3, "forcing_frozen"), probe(v3, "model_design"), probe(v3, "scoring_target"),
        probe(v3, "calibration_target"), probe(v3, "prior")]
    m2 = pd.read_csv(v2 / "district_monthly.csv")
    m3 = pd.read_csv(v3 / "district_monthly.csv")
    sz = m3.structural_zero.astype(bool).to_numpy()
    is26 = (pd.to_datetime(m3.date).dt.year == 2026).to_numpy()
    keep = ~sz & ~is26
    d1 = np.load(v1 / "district_draws.npz", allow_pickle=True)
    d3 = np.load(v3 / "district_draws.npz", allow_pickle=True)
    n1 = d1["draws_changhua"].shape[1]
    g2 = np.load(v2 / "grid_sw_v4.npz", allow_pickle=True)
    g3 = np.load(v3 / "grid_sw_v4.npz", allow_pickle=True)
    ex = {
        "rows": int(len(m3)), "same_keys_as_v2": bool(m3[["date", "district"]].equals(
            m2[["date", "district"]])),
        "values_identical_to_v2": bool(np.array_equal(m3.value.to_numpy(), m2.value.to_numpy())),
        "flags_identical_to_v2": bool(np.array_equal(m3.flag.to_numpy(), m2.flag.to_numpy())),
        "sd_identical_to_v2_outside_structural_and_2026": bool(np.array_equal(
            m3.sd.to_numpy()[keep], m2.sd.to_numpy()[keep])),
        "structural_rows": int(sz.sum()),
        "structural_sd_max": float(m3.sd.to_numpy()[sz].max()),
        "structural_alt_equals_v2_sd_2012_2025": bool(np.allclose(
            m3.structural_alt_value.to_numpy()[sz & ~is26], m2.sd.to_numpy()[sz & ~is26])),
        "rows_without_sd_basis": int(m3.sd_basis.isna().sum()),
        "rows_without_interval_calibration": int(m3.interval_calibration.isna().sum()),
        "interval_calibration_labels": m3.interval_calibration.value_counts().to_dict(),
        "draws_2012_2025_identical_to_v1": bool(
            np.array_equal(d1["draws_changhua"], d3["draws_changhua"][:, :n1])
            and np.array_equal(d1["draws_yunlin"], d3["draws_yunlin"][:, :n1])),
        "grid_value_identical_to_v2": bool(np.array_equal(g2["value"], g3["value"])),
        "grid_flag_identical_to_v2": bool(np.array_equal(g2["flag"], g3["flag"])),
        "grid_value_v3level_identical_to_v2": bool(np.array_equal(g2["value_v3level"],
                                                                  g3["value_v3level"])),
        "mask_hash_unchanged": str(g2["mask_hash"]) == str(g3["mask_hash"]),
        "grid_shape": list(g3["value"].shape)}
    dates3 = pd.to_datetime(d3["dates"])
    for nm in ("changhua", "yunlin"):
        X = d3[f"draws_{nm}"]
        t = m3[m3.district == nm].sort_values("date")
        sz_d = t.structural_zero.astype(bool).to_numpy()
        is26_d = (pd.to_datetime(t.date).dt.year == 2026).to_numpy()
        ex[nm] = {
            "negative_members": int((X < 0).sum()),
            "structural_nonzero_members": int((X[:, sz_d] != 0).sum()),
            "sd_2026_equals_member_sd_max_abs": float(np.abs(
                X[:, is26_d].std(0) - t.sd.to_numpy()[is26_d]).max()),
            "member_mean_minus_value_2026": (X[:, is26_d].mean(0)
                                             - t.value.to_numpy()[is26_d]).round(4).tolist(),
            "distinct_2026_trajectories": int(len(np.unique(np.round(X[:, is26_d], 12),
                                                            axis=0))),
            "dates_match_table": bool(np.array_equal(dates3, pd.to_datetime(t.date)))}
    val = json.loads((v3 / "validation.json").read_text())
    sc = val["ext2026"]["chain_test"]["scores"]
    ex["denton_target_rmse_in_scores"] = sorted(
        k for d in sc for k in sc[d] if "denton" in k)
    out["exactness"] = ex
    out["repair_r1_summary"] = {k: val["repair_r1"][k] for k in (
        "label_2026", "label_hist", "annual_check_passed")}
    out["forward_archive_probes"] = [http_probe(u) for u in PROBES]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE / "rep_verify.json", "w") as f:
        json.dump(jsonable(out), f, indent=1, ensure_ascii=False)
    print(json.dumps(jsonable({k: out[k] for k in ("checker", "registrations")}), indent=1,
                     ensure_ascii=False)[:3000])
    print(json.dumps(jsonable(ex), indent=1, ensure_ascii=False)[:4000])
    for g in out["guard_probes"]:
        print(g["use"], g.get("opened"), g.get("action", g.get("error")))
    for p in out["forward_archive_probes"]:
        print(p["http_code"], p["content_type"], p["url"][:70])


if __name__ == "__main__":
    main()
