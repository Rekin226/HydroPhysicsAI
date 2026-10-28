"""Cleaned head inputs continued past the origin (twin_inputs/clean_heads v2, 2026-10-28),
for the forcing-fix rung's hindcast nudged toward the observed heads through 2025.

``clean_heads`` v1 holds each raw well file cut before 2023-01-01 with the L1 v3 flag-9
events removed (``dataprep_staging/i1/build_clean_head_inputs.py``). The raw AMP_V2 files
end in 2022, so the months after it come from the L1 v3 build itself:

* ``wells/<sid>.parquet``: v1's samples unchanged (so a twin load over 2012-2022 is
  identical to v1's), then one sample per day from 2023-01-01: ``heads_daily.value_raw``
  of the days flagged 0 or 1. ``value_raw`` keeps barometric loading in, as v1's raw
  samples do; days flagged 3 (neighbour-only step correction), 8 or 9 are left out.
* ``fan_stations.parquet``: v1's table, copied.

Allowed uses: the recorded-forcing diagnostic arm (U4) and after-the-fact analysis only; a
pre-origin use (calibration, initial conditions, selection) reads v1, which these samples
equal up to 2022-12.

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/x5_nudge/build_clean_heads_ext.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from common import manifest  # noqa: E402

V1 = REPO / "results/dataprep/twin_inputs/clean_heads/v1"
L1 = REPO / "results/dataprep/heads/L1_clean/v3"
ORIGIN_END = pd.Timestamp("2023-01-01")


def main() -> None:
    daily = pd.read_parquet(L1 / "heads_daily.parquet",
                            columns=["sid", "date", "value_raw", "flag"])
    daily["sid"] = daily["sid"].astype(str)
    daily = daily[(daily["date"] >= ORIGIN_END) & daily["flag"].isin([0, 1])
                  & daily["value_raw"].notna()]
    post = {sid: g for sid, g in daily.groupby("sid")}
    d = manifest.new_version_dir("twin_inputs/clean_heads")
    wd = d / "wells"
    wd.mkdir()
    rows = []
    for f in sorted((V1 / "wells").glob("*.parquet")):
        pre = pd.read_parquet(f)
        if len(pre) and pre.index.max() >= ORIGIN_END:
            raise SystemExit(f"{f}: v1 holds samples past 2022")
        g = post.get(f.stem)
        if g is None or g.empty:
            pre.to_parquet(wd / f.name)
            rows.append({"sid": f.stem, "n_pre": len(pre), "n_post_days": 0})
            continue
        s = pd.DataFrame({"value": g["value_raw"].to_numpy(dtype="float64")},
                         index=pd.DatetimeIndex(g["date"], name=pre.index.name))
        pd.concat([pre, s]).to_parquet(wd / f.name)
        rows.append({"sid": f.stem, "n_pre": len(pre), "n_post_days": len(s),
                     "post_first": str(s.index.min().date()),
                     "post_last": str(s.index.max().date())})
    shutil.copy2(V1 / "fan_stations.parquet", d / "fan_stations.parquet")
    per = pd.DataFrame(rows)
    per.to_csv(d / "wells_extended.csv", index=False)
    no_raw = sorted(set(post) - set(per["sid"]))
    val = {"passed": True, "wells": len(per),
           "wells_with_post_days": int((per["n_post_days"] > 0).sum()),
           "post_days": int(per["n_post_days"].sum()),
           "post_last_day": str(daily["date"].max().date()),
           "l1_wells_without_a_v1_file": no_raw}
    (d / "validation.json").write_text(json.dumps(val, indent=1))
    m = manifest.build_manifest(
        d, product="twin_inputs/clean_heads", layer="L1",
        inputs=[str(V1 / "MANIFEST.json"), str(L1 / "heads_daily.parquet"),
                str(L1 / "MANIFEST.json")],
        code=[str(Path(__file__))],
        parameters={"pre_origin": "clean_heads v1 samples unchanged",
                    "post_origin": "L1 v3 heads_daily value_raw, flags 0-1, one sample per "
                                   "day from 2023-01-01"},
        fit_window_end="2022-12", non_exempt_fit_window_end="2022-12",
        exemptions=[], validation_report="validation.json",
        gate={"passed": True, "criteria": "v1 prefix unchanged; post-origin days flags 0-1 "
                                          "only; nothing else changed"},
        # heads past the origin enter a run only as its recorded-forcing diagnostic arm
        # (U4); pre-origin uses keep reading v1, whose samples these are
        allowed_uses=["forcing_recorded_arm", "after_the_fact_analysis"],
        forbidden_uses=["prospective_scorer", "calibration_target", "initial_conditions",
                        "selection"])
    manifest.write_manifest(d, m)
    print(json.dumps(val, indent=1), "\nwritten", d)


if __name__ == "__main__":
    main()
